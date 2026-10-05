"""
Host Identification — best-effort extraction of infected-host identity
artifacts (MAC address, hostname, username) from a packet capture.

Deliberately SEPARATE from pcap_transform.py / feature_transform.py / the
ML classifier — this module is never imported by, and never touches, the
traffic-classification pipeline. It answers a different question ("who is
this IP") using different, well-known protocol parsers (ARP/Ethernet,
DHCP, NTLM authentication) instead of statistical traffic features.

NOT attempted: the user's full name. That generally isn't a network-traffic
artifact — it usually requires an LDAP/Active Directory lookup, or happens
to leak through content specific to one capture (an email, a form
submission). Reporting a guessed name here would be misleading, so this
module always leaves it null and says so.
"""

import base64
import io

MAX_PACKETS = 200_000


def _mac_str(raw_mac: bytes) -> str:
    return ':'.join(f'{b:02x}' for b in raw_mac[:6])


def _parse_ntlm_type3(data: bytes):
    """
    Extract Domain/User/Workstation from an NTLM Type 3 (Authenticate)
    message per [MS-NLMP]. Returns None if `data` doesn't contain one.
    """
    sig_idx = data.find(b'NTLMSSP\x00')
    if sig_idx == -1:
        return None
    msg = data[sig_idx:]
    if len(msg) < 12:
        return None
    msg_type = int.from_bytes(msg[8:12], 'little')
    if msg_type != 3:
        return None

    def read_buf(offset):
        # NTLM "security buffer": 2-byte Len, 2-byte MaxLen, 4-byte Offset
        # (offset is relative to the start of the NTLM message).
        if offset + 8 > len(msg):
            return None
        length = int.from_bytes(msg[offset:offset + 2], 'little')
        buf_offset = int.from_bytes(msg[offset + 4:offset + 8], 'little')
        if length == 0 or buf_offset + length > len(msg):
            return None
        try:
            return msg[buf_offset:buf_offset + length].decode('utf-16-le').strip()
        except Exception:
            return None

    domain = read_buf(28)
    user = read_buf(36)
    workstation = read_buf(44)
    if not user:
        return None
    return {'domain': domain, 'user': user, 'workstation': workstation}


def _find_ntlm_in_payload(payload: bytes):
    """NTLM Type 3 can appear raw (SMB2 Session Setup) or base64-encoded
    after an HTTP 'Authorization: NTLM ' header — try both."""
    result = _parse_ntlm_type3(payload)
    if result:
        return result
    marker = b'Authorization: NTLM '
    idx = payload.find(marker)
    if idx == -1:
        return None
    tail = payload[idx + len(marker):]
    end = tail.find(b'\r\n')
    b64 = tail[:end] if end != -1 else tail
    try:
        return _parse_ntlm_type3(base64.b64decode(b64, validate=False))
    except Exception:
        return None


def identify_hosts(pcap_bytes: bytes) -> dict:
    """
    Parse a pcap for host-identity artifacts, keyed by IP address.

    Returns
    -------
    dict[str, dict]: ip -> {
        'mac': str | None,            # from ARP, falls back to Ethernet frames
        'mac_source': str | None,     # 'arp' | 'ethernet_frame' | 'dhcp'
        'hostname': str | None,       # from DHCP option 12, or NTLM workstation field
        'hostname_source': str | None,# 'dhcp' | 'ntlm_workstation'
        'username': str | None,       # from NTLM authentication (SMB2 / HTTP)
        'username_source': str | None,# 'ntlm'
        'domain': str | None,         # NTLM domain, if present
        'full_name': None,            # never populated — see module docstring
    }

    Raises ValueError on an unparseable file (caller maps this to HTTP 400).
    """
    try:
        from scapy.utils import PcapReader
        from scapy.layers.l2 import Ether, ARP
        from scapy.layers.inet import IP, TCP
        from scapy.layers.dhcp import DHCP, BOOTP
    except ImportError as e:
        raise RuntimeError(f"scapy not installed: {e}")

    hosts = {}
    _IGNORED_IPS = ('0.0.0.0', '255.255.255.255')

    def get(ip):
        if ip in _IGNORED_IPS:
            return None
        if ip not in hosts:
            hosts[ip] = {
                'mac': None, 'mac_source': None,
                'hostname': None, 'hostname_source': None,
                'username': None, 'username_source': None,
                'domain': None,
                'full_name': None,
            }
        return hosts[ip]

    total = 0
    try:
        with PcapReader(io.BytesIO(pcap_bytes)) as reader:
            for pkt in reader:
                if total >= MAX_PACKETS:
                    break
                total += 1

                # ── MAC address: ARP is the canonical source (an ARP reply
                # is literally "this IP is at this MAC"); fall back to
                # whatever Ethernet frame we see from that source IP ──
                if ARP in pkt:
                    arp = pkt[ARP]
                    h = get(arp.psrc) if arp.psrc else None
                    if h and h['mac'] is None:
                        h['mac'] = arp.hwsrc
                        h['mac_source'] = 'arp'
                if Ether in pkt and IP in pkt:
                    h = get(pkt[IP].src)
                    if h and h['mac'] is None:
                        h['mac'] = pkt[Ether].src
                        h['mac_source'] = 'ethernet_frame'

                # ── Hostname via DHCP option 12 (client announces its own
                # hostname when requesting/renewing a lease) ──
                if DHCP in pkt and BOOTP in pkt:
                    opts = {o[0]: o[1] for o in pkt[DHCP].options
                            if isinstance(o, tuple) and len(o) == 2}
                    hostname = opts.get('hostname')
                    if hostname:
                        if isinstance(hostname, bytes):
                            hostname = hostname.decode('utf-8', errors='ignore')
                        bootp = pkt[BOOTP]
                        mac = _mac_str(bootp.chaddr) if bootp.chaddr else None
                        # ciaddr/yiaddr may be the only IPs we ever see for this
                        # host if it's mid-DHCP-negotiation (no data traffic yet)
                        for candidate_ip in (bootp.ciaddr, getattr(bootp, 'yiaddr', None)):
                            h2 = get(candidate_ip) if candidate_ip else None
                            if h2 is None:
                                continue
                            if h2['hostname'] is None:
                                h2['hostname'] = hostname
                                h2['hostname_source'] = 'dhcp'
                            if h2['mac'] is None and mac:
                                h2['mac'] = mac
                                h2['mac_source'] = 'dhcp'
                        # Backfill any host already seen (via ARP/Ethernet)
                        # with this same MAC but no hostname yet.
                        if mac:
                            for h2 in hosts.values():
                                if h2['mac'] == mac and h2['hostname'] is None:
                                    h2['hostname'] = hostname
                                    h2['hostname_source'] = 'dhcp'

                # ── Username (+ workstation hostname) via NTLM authentication,
                # e.g. SMB2 Session Setup to an internal file share ──
                if TCP in pkt and IP in pkt:
                    raw_payload = bytes(pkt[TCP].payload)
                    if b'NTLMSSP' in raw_payload:
                        ntlm = _find_ntlm_in_payload(raw_payload)
                        h = get(pkt[IP].src) if ntlm else None
                        if h:
                            if h['username'] is None:
                                h['username'] = ntlm['user']
                                h['username_source'] = 'ntlm'
                                h['domain'] = ntlm['domain']
                            if h['hostname'] is None and ntlm['workstation']:
                                h['hostname'] = ntlm['workstation']
                                h['hostname_source'] = 'ntlm_workstation'
    except Exception as e:
        raise ValueError(f"Failed to parse pcap for host identification: {e}")

    return hosts
