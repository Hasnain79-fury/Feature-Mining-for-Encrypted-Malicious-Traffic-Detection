"""
PCAP Transform — extracts per-flow packet records from a raw packet capture.

Groups packets into TCP flows (5-tuple, direction-normalized) the same way
a session/unique_link_mark groups packets in the training CSVs, and hands
each flow's packets to FeatureTransformer.transform_pcap() for real (not
approximated) TTL/TCP-window/header-length feature extraction — see that
method's docstring for what this buys over browser-only capture.

Uses scapy (pure Python, no external `tshark` binary required) so this repo
keeps working after a plain `pip install -r requirements.txt`.
"""

import io

# Safety bounds — protect the server from a huge/malicious upload.
MAX_PACKETS = 200_000
MAX_FLOWS = 500
MIN_PACKETS_PER_FLOW = 3


def _extract_sni(payload: bytes):
    """
    Best-effort SNI extraction from a TLS ClientHello. Returns the hostname
    string, or None if this packet doesn't look like a ClientHello or
    parsing fails for any reason (fragmented handshake, malformed data,
    etc.) — callers fall back to an "ip:port -> ip:port" label.
    """
    try:
        if len(payload) < 6 or payload[0] != 0x16 or payload[1] != 0x03:
            return None  # not a TLS handshake record
        pos = 5  # skip TLS record header (type, version, length)
        if payload[pos] != 0x01:
            return None  # not a ClientHello
        pos += 4  # skip handshake header (type, 3-byte length)
        pos += 2 + 32  # client_version(2) + random(32)
        session_id_len = payload[pos]
        pos += 1 + session_id_len
        cipher_suites_len = int.from_bytes(payload[pos:pos + 2], 'big')
        pos += 2 + cipher_suites_len
        compression_len = payload[pos]
        pos += 1 + compression_len
        if pos + 2 > len(payload):
            return None
        extensions_len = int.from_bytes(payload[pos:pos + 2], 'big')
        pos += 2
        end = pos + extensions_len
        while pos + 4 <= end and pos + 4 <= len(payload):
            ext_type = int.from_bytes(payload[pos:pos + 2], 'big')
            ext_len = int.from_bytes(payload[pos + 2:pos + 4], 'big')
            ext_data_start = pos + 4
            if ext_type == 0x0000:  # server_name
                p = ext_data_start + 2  # skip server_name_list length
                name_type = payload[p]
                name_len = int.from_bytes(payload[p + 1:p + 3], 'big')
                if name_type == 0:
                    return payload[p + 3:p + 3 + name_len].decode('ascii', errors='ignore')
            pos = ext_data_start + ext_len
        return None
    except Exception:
        return None


def extract_flows(pcap_bytes: bytes):
    """
    Parse a .pcap/.pcapng file's bytes into a list of TCP flows.

    Returns
    -------
    list[dict]: each {
        'flow_id': str,      # "ip:port -> ip:port" (always present)
        'label': str,        # SNI hostname if found, else same as flow_id
        'packet_count': int,
        'packets': [ {timestamp, direction, ip_len, ip_header_len, ttl,
                       tcp_header_len, tcp_payload_len, tcp_window}, ... ]
    }

    Raises ValueError on an unparseable file (caller maps this to HTTP 400).
    """
    try:
        from scapy.utils import PcapReader
        from scapy.layers.inet import IP, TCP
    except ImportError as e:
        raise RuntimeError(f"scapy not installed: {e}")

    flows = {}  # key -> {'initiator': (ip, port), 'packets': [...], 'sni': str|None}
    total_packets = 0

    try:
        with PcapReader(io.BytesIO(pcap_bytes)) as reader:
            for pkt in reader:
                if total_packets >= MAX_PACKETS:
                    break
                if IP not in pkt or TCP not in pkt:
                    continue
                total_packets += 1

                ip_layer = pkt[IP]
                tcp_layer = pkt[TCP]
                a = (ip_layer.src, tcp_layer.sport)
                b = (ip_layer.dst, tcp_layer.dport)
                key = tuple(sorted([a, b]))

                if key not in flows:
                    if len(flows) >= MAX_FLOWS:
                        continue  # stop tracking brand-new flows once capped
                    flows[key] = {'initiator': a, 'packets': [], 'sni': None}

                fw = flows[key]
                direction = 'fwd' if a == fw['initiator'] else 'bwd'

                tcp_payload = bytes(tcp_layer.payload)
                if fw['sni'] is None and direction == 'fwd' and tcp_payload:
                    sni = _extract_sni(tcp_payload)
                    if sni:
                        fw['sni'] = sni

                dataofs = tcp_layer.dataofs if tcp_layer.dataofs else 5
                fw['packets'].append({
                    'timestamp': float(pkt.time),
                    'direction': direction,
                    'ip_len': float(ip_layer.len or 0),
                    'ip_header_len': float((ip_layer.ihl or 5) * 4),
                    'ttl': float(ip_layer.ttl or 0),
                    'tcp_header_len': float(dataofs * 4),
                    'tcp_payload_len': float(len(tcp_payload)),
                    'tcp_window': float(tcp_layer.window or 0),
                })
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse pcap: {e}")

    if total_packets == 0:
        raise ValueError("No TCP/IP packets found in this capture")

    results = []
    for (a, b), fw in flows.items():
        if len(fw['packets']) < MIN_PACKETS_PER_FLOW:
            continue
        flow_id = f"{a[0]}:{a[1]} -> {b[0]}:{b[1]}"
        results.append({
            'flow_id': flow_id,
            'label': fw['sni'] or flow_id,
            'packet_count': len(fw['packets']),
            'packets': fw['packets'],
        })

    # Largest flows first — most likely to be the interesting ones, and
    # ensures a truncated MAX_FLOWS result still favors substantial sessions.
    results.sort(key=lambda f: f['packet_count'], reverse=True)
    return results
