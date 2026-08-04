"""
Feature Transform — maps browser-collected packet data to model input tensors.

Replicates the preprocessing from:
  - src/preprocessing/lstm_pipeline.py   (ALL_TIME_COLS, 15-pkt window, mean-pad)
  - src/preprocessing/resnet_pipeline.py (38 payload features → outer product)
  - src/preprocessing/xgb_pipeline.py    (RATIO_COLS + ENC_TIME_SESS_COLS → 104 features)
"""

import numpy as np
import joblib
import os

# ── Column definitions (mirrored from src/preprocessing/feature_columns.py) ──

PKT_TIME_COLS = [
    'Time_cost',
    'Time_difference_between_packets_per_session',
    'Interval_of_arrival_time_of_forward_traffic',
    'Interval_of_arrival_time_of_backward_traffic',
    'inter_arrival_time_of_forward_traffic_enc',
    'inter_arrival_time_of_backward_traffic_enc',
    'ratio_to_previous_packet_enc',
]

PAYLOAD_CANDIDATES = [
    'mean_Length_of_IP_packets',    'median_Length_of_IP_packets',
    'max_Length_of_IP_packets',     'min_Length_of_IP_packets',
    'std_Length_of_IP_packets',     'var_Length_of_IP_packets',
    'mean_Length_of_TCP_payload',   'median_Length_of_TCP_payload',
    'max_Length_of_TCP_payload',    'min_Length_of_TCP_payload',
    'std_Length_of_TCP_payload',    'var_Length_of_TCP_payload',
    'mean_Length_of_TCP_packet_header',  'median_Length_of_TCP_packet_header',
    'max_Length_of_TCP_packet_header',   'min_Length_of_TCP_packet_header',
    'std_Length_of_TCP_packet_header',   'var_Length_of_TCP_packet_header',
    'mean_Length_of_IP_packet_header',   'median_Length_of_IP_packet_header',
    'max_Length_of_IP_packet_header',    'min_Length_of_IP_packet_header',
    'std_Length_of_IP_packet_header',    'var_Length_of_IP_packet_header',
    'mean_TCP_windows_size_value',  'median_TCP_windows_size_value',
    'max_TCP_windows_size_value',   'min_TCP_windows_size_value',
    'std_TCP_windows_size_value',   'var_TCP_windows_size_value',
    'mean_Length_of_TCP_segment(packet)', 'median_Length_of_TCP_segment(packet)',
    'max_Length_of_TCP_segment(packet)',  'min_Length_of_TCP_segment(packet)',
    'std_Length_of_TCP_segment(packet)',  'var_Length_of_TCP_segment(packet)',
    'std_forward_packet_length',    'mean_forward_packet_length',
    'max_forward_packet_length',    'min_forward_packet_length',
    'var_forward_packet_length',    'median_forward_packet_length',
    'std_backward_packet_length',   'mean_backward_packet_length',
    'max_backward_packet_length',   'min_backward_packet_length',
    'var_backward_packet_length',   'median_backward_packet_length',
]


def _safe_log1p(x):
    return np.log1p(np.abs(x))


def _safe_ratio(a, b):
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(b_arr != 0, a_arr / b_arr, 1.0)
    res = np.nan_to_num(r, nan=1.0, posinf=1.0, neginf=1.0)
    if np.isscalar(a) and np.isscalar(b):
        return float(res)
    return res


def _stats6(arr):
    """Return [mean, median, max, min, std, var] for an array."""
    if len(arr) == 0:
        return [0.0] * 6
    a = np.array(arr, dtype=np.float32)
    return [float(np.mean(a)), float(np.median(a)), float(np.max(a)),
            float(np.min(a)), float(np.std(a)), float(np.var(a))]


def _stats7(arr):
    """Return [total, std, mean, max, min, var, median] — TTL stat order."""
    if len(arr) == 0:
        return [0.0] * 7
    a = np.array(arr, dtype=np.float32)
    return [float(np.sum(a)), float(np.std(a)), float(np.mean(a)),
            float(np.max(a)), float(np.min(a)), float(np.var(a)), float(np.median(a))]


class FeatureTransformer:
    """Converts browser packet data into model-ready tensors."""

    # Default TTL values (not observable from browser)
    DEFAULT_TTL_FWD = 128.0
    DEFAULT_TTL_BWD = 64.0
    # Default TCP window size
    DEFAULT_TCP_WIN = 65535.0
    # Default IP header length
    DEFAULT_IP_HDR = 20.0

    def __init__(self, model_dir, scalers_dir, tensors_dir):
        self.scaler_lstm = joblib.load(os.path.join(scalers_dir, 'scaler_lstm.pkl'))
        self.scaler_resnet = joblib.load(os.path.join(scalers_dir, 'scaler_resnet_mms.pkl'))
        self.selector_resnet = joblib.load(os.path.join(scalers_dir, 'selector_resnet.pkl'))
        medians_path = os.path.join(tensors_dir, 'xgb_col_medians.npy')
        self.xgb_medians = np.load(medians_path) if os.path.exists(medians_path) else None

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────
    def transform(self, session_data: dict):
        """
        Parameters
        ----------
        session_data : dict with keys 'packets' (list of packet dicts).
            Each packet dict: {timestamp, duration, requestSize, responseSize,
                               transferSize, encodedBodySize, headerSize, ...}

        Returns
        -------
        X_lstm   : np.ndarray (1, 15, F_lstm)
        X_resnet : np.ndarray (1, 1, 38, 38)
        X_xgb    : np.ndarray (1, N_xgb)
        """
        packets = session_data['packets']
        if len(packets) == 0:
            raise ValueError("No packets in session")

        # ── Extract raw per-packet arrays ──
        durations = np.array([p.get('duration', 0) for p in packets], dtype=np.float32)
        # Timestamps arrive as millisecond epoch values (~1.7e12, e.g. Date.now()).
        # float32 only carries ~7 significant digits, so at that magnitude it can't
        # represent anything finer than ~131 seconds — every sub-second inter-packet
        # gap would silently round to zero. Rebase to "ms since first packet" in
        # float64 first (values then stay in the tens-of-thousands range, well
        # within float32's exact-integer range up to 2**24) before downcasting.
        timestamps64 = np.array([p.get('timestamp', 0) for p in packets], dtype=np.float64)
        if len(timestamps64) > 0:
            timestamps64 = timestamps64 - timestamps64[0]
        timestamps = timestamps64.astype(np.float32)
        req_sizes = np.array([p.get('requestSize', 0) for p in packets], dtype=np.float32)
        resp_sizes = np.array([p.get('responseSize', 0) for p in packets], dtype=np.float32)
        transfer_sizes = np.array([p.get('transferSize', 0) for p in packets], dtype=np.float32)
        encoded_body = np.array([p.get('encodedBodySize', 0) for p in packets], dtype=np.float32)
        header_sizes = np.array([p.get('headerSize', 0) for p in packets], dtype=np.float32)

        # ── Per-packet time deltas ──
        time_diffs = np.diff(timestamps, prepend=timestamps[0])
        iat_fwd = time_diffs.copy()
        iat_bwd = time_diffs.copy()  # Approximation: same as forward
        iat_fwd_enc = _safe_log1p(iat_fwd)
        iat_bwd_enc = _safe_log1p(iat_bwd)
        ratio_prev = np.ones_like(durations)
        ratio_prev[1:] = _safe_ratio(durations[1:], durations[:-1])

        # ── Build per-packet feature matrix (N_packets × 7) ──
        pkt_features = np.column_stack([
            durations, time_diffs, iat_fwd, iat_bwd,
            iat_fwd_enc, iat_bwd_enc, ratio_prev,
        ])

        # ── Session-level time stats (broadcast to every packet row) ──
        sess_time = self._session_time_stats(
            durations, time_diffs, iat_fwd, iat_bwd, timestamps)

        # Broadcast: repeat session vector for each packet
        sess_broadcast = np.tile(sess_time, (len(packets), 1))

        # Concatenate → (N_packets, 7 + len(sess_time))
        full_pkt = np.hstack([pkt_features, sess_broadcast]).astype(np.float32)

        # ── LSTM: 15-packet window with mean-padding ──
        X_lstm = self._build_lstm(full_pkt)

        # ── ResNet: 38 payload features → outer product ──
        X_resnet = self._build_resnet(
            transfer_sizes, encoded_body, header_sizes, req_sizes, resp_sizes)

        # ── XGBoost: session-level features ──
        X_xgb = self._build_xgb(sess_time, durations, time_diffs,
                                 iat_fwd, iat_bwd, transfer_sizes,
                                 encoded_body, header_sizes, req_sizes, resp_sizes)

        return X_lstm, X_resnet, X_xgb

    # ──────────────────────────────────────────────
    #  Session-level time statistics
    # ──────────────────────────────────────────────
    def _session_time_stats(self, durations, time_diffs, iat_fwd, iat_bwd, timestamps):
        """Compute session-level stats matching SESS_TIME_COLS order."""
        flow_dur = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        flow_dur_fwd = flow_dur
        flow_dur_bwd = flow_dur * 0.5  # Approximation

        td_stats = _stats6(time_diffs)
        iat_fwd_stats = _stats6(iat_fwd)
        iat_bwd_stats = _stats6(iat_bwd)

        # TTL stats — not available from browser, use defaults
        n = len(durations)
        ttl_fwd_arr = np.full(n, self.DEFAULT_TTL_FWD, dtype=np.float32)
        ttl_bwd_arr = np.full(n, self.DEFAULT_TTL_BWD, dtype=np.float32)
        ttl_fwd_stats = _stats7(ttl_fwd_arr)
        ttl_bwd_stats = _stats7(ttl_bwd_arr)

        # Raw section (35 values)
        raw = ([flow_dur, flow_dur_fwd, flow_dur_bwd]
               + td_stats + iat_fwd_stats + iat_bwd_stats
               + ttl_fwd_stats + ttl_bwd_stats)

        # Encoded section (29 values) — log1p of raw
        fd_enc = _safe_log1p(flow_dur)
        fd_fwd_enc = _safe_log1p(flow_dur_fwd)
        fd_bwd_enc = _safe_log1p(flow_dur_bwd)
        iat_fwd_enc_stats = _stats6(_safe_log1p(np.array(iat_fwd)))
        iat_bwd_enc_stats = _stats6(_safe_log1p(np.array(iat_bwd)))
        ttl_fwd_enc_stats = _stats7(_safe_log1p(ttl_fwd_arr))
        ttl_bwd_enc_stats = _stats7(_safe_log1p(ttl_bwd_arr))

        enc = ([fd_enc, fd_fwd_enc, fd_bwd_enc]
               + iat_fwd_enc_stats + iat_bwd_enc_stats
               + ttl_fwd_enc_stats + ttl_bwd_enc_stats)

        # Ratio section (22 values) — raw / enc
        fd_ratio = _safe_ratio(flow_dur, fd_enc)
        fd_fwd_ratio = _safe_ratio(flow_dur_fwd, fd_fwd_enc)
        fd_bwd_ratio = _safe_ratio(flow_dur_bwd, fd_bwd_enc)
        iat_fwd_ratio_stats = [_safe_ratio(r, e) for r, e in zip(
            _stats6(iat_fwd), iat_fwd_enc_stats)]
        iat_bwd_ratio_stats = [_safe_ratio(r, e) for r, e in zip(
            _stats6(iat_bwd), iat_bwd_enc_stats)]
        ttl_fwd_ratio_stats = [_safe_ratio(r, e) for r, e in zip(
            ttl_fwd_stats, ttl_fwd_enc_stats)]

        ratio = ([fd_ratio, fd_fwd_ratio, fd_bwd_ratio]
                 + iat_fwd_ratio_stats + iat_bwd_ratio_stats
                 + ttl_fwd_ratio_stats)

        # Flatten everything
        all_vals = []
        for v in raw + enc + ratio:
            all_vals.append(float(v) if np.isscalar(v) else float(v))

        return np.array(all_vals, dtype=np.float32)

    # ──────────────────────────────────────────────
    #  LSTM branch
    # ──────────────────────────────────────────────
    def _build_lstm(self, full_pkt):
        """Window to 15 packets, mean-pad if needed, apply scaler."""
        max_pkts = 15
        mat = full_pkt.copy()

        if len(mat) > max_pkts:
            mat = mat[:max_pkts]
        elif len(mat) < max_pkts:
            pad_val = mat.mean(axis=0)
            pad_rows = np.tile(pad_val, (max_pkts - len(mat), 1))
            mat = np.vstack([mat, pad_rows])

        # Apply scaler (handles 3D input)
        mat_3d = mat[np.newaxis, :, :]  # (1, 15, F)
        mat_3d = self.scaler_lstm.transform(mat_3d)
        return mat_3d.astype(np.float32)

    # ──────────────────────────────────────────────
    #  ResNet branch
    # ──────────────────────────────────────────────
    def _build_resnet(self, transfer_sizes, encoded_body, header_sizes,
                      req_sizes, resp_sizes):
        """Compute 48 payload candidates → select 38 → scale → outer product."""
        n = len(transfer_sizes)

        # Map browser data to the 8 payload groups
        ip_pkt_lens = transfer_sizes                      # IP packet ≈ transfer size
        tcp_payload = encoded_body                        # TCP payload ≈ encoded body
        tcp_hdr = np.maximum(header_sizes, 20.0)          # TCP header ≈ header size
        ip_hdr = np.full(n, self.DEFAULT_IP_HDR)          # IP header = constant 20
        tcp_win = np.full(n, self.DEFAULT_TCP_WIN)         # Not observable
        tcp_seg = transfer_sizes                          # TCP segment ≈ transfer size
        fwd_pkt = req_sizes                               # Forward = request
        bwd_pkt = resp_sizes                              # Backward = response

        groups = [ip_pkt_lens, tcp_payload, tcp_hdr, ip_hdr,
                  tcp_win, tcp_seg, fwd_pkt, bwd_pkt]

        # Compute 6 stats per group → 48 candidates
        candidates = []
        for g in groups:
            candidates.extend(_stats6(g))

        # Reorder to match PAYLOAD_CANDIDATES order (first 6 groups use mean/median/max/min/std/var,
        # last 2 groups use std/mean/max/min/var/median)
        # First 6 groups: mean, median, max, min, std, var (same order as _stats6)
        # Last 2 groups (fwd/bwd packet length): std, mean, max, min, var, median
        first_36 = candidates[:36]  # 6 groups × 6 stats
        # Reorder last 2 groups from [mean,median,max,min,std,var] to [std,mean,max,min,var,median]
        for i in range(2):
            base = 36 + i * 6
            grp = candidates[base:base + 6]  # [mean, median, max, min, std, var]
            reordered = [grp[4], grp[0], grp[2], grp[3], grp[5], grp[1]]
            first_36.extend(reordered)

        all_48 = np.array(first_36, dtype=np.float32).reshape(1, -1)

        # Apply selector (picks 38 of 48)
        selected = self.selector_resnet.transform(all_48)  # (1, 38)

        # Apply MinMaxScaler
        selected = self.scaler_resnet.transform(selected)

        # Outer product → (1, 1, 38, 38)
        vec = selected[0]
        img = np.outer(vec, vec)[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return img

    # ──────────────────────────────────────────────
    #  XGBoost branch
    # ──────────────────────────────────────────────
    def _build_xgb(self, sess_time_stats, durations, time_diffs,
                   iat_fwd, iat_bwd, transfer_sizes, encoded_body,
                   header_sizes, req_sizes, resp_sizes):
        """
        Build the 104-feature XGBoost vector in the EXACT column order the
        model was trained on: 72 session-level '_ratio' columns (verified
        against the real column order in Train Set/session_based_trainset.csv)
        followed by the 32 ENC_TIME_SESS_COLS (feature_columns.py).

        XGBoost/RF split on feature *position*, not name, so every slot must
        line up with training. Slots start at the training median — the
        "unknown" default the model already learned to handle — and are
        overridden only where a browser-observable proxy genuinely exists
        (payload sizes, timing). TTL, TCP window size, and per-direction
        header/segment splits aren't observable from the browser at all and
        are left at their training median.
        """
        expected = 104
        n = len(durations)
        if self.xgb_medians is not None and len(self.xgb_medians) == expected:
            vec = self.xgb_medians.copy().astype(np.float32)
        else:
            vec = np.zeros(expected, dtype=np.float32)

        # sess_time_stats layout (86 values): [0:35]=raw, [35:64]=enc, [64:86]=ratio
        ratio_section = sess_time_stats[64:]  # 22 values, see _session_time_stats

        def _len_ratio_block(arr):
            """[mean,median,max,min,std,var] ratios reordered to [std,mean,max,min,var,median]."""
            raw = _stats6(arr)
            enc = _stats6(_safe_log1p(np.asarray(arr, dtype=np.float32)))
            r = [_safe_ratio(a, b) for a, b in zip(raw, enc)]
            return [r[4], r[0], r[2], r[3], r[5], r[1]]

        # idx 0-11: forward/backward packet-length ratios (std,mean,max,min,var,median)
        vec[0:6] = _len_ratio_block(req_sizes)    # forward ≈ request
        vec[6:12] = _len_ratio_block(resp_sizes)  # backward ≈ response

        # idx 12-14: flow_duration ratio (total, forward, backward)
        vec[12:15] = ratio_section[0:3]

        # idx 15-26: IAT forward/backward ratio stats (mean,median,max,min,std,var)
        vec[15:21] = ratio_section[3:9]
        vec[21:27] = ratio_section[9:15]

        # idx 27-33: TTL forward ratio (total,std,mean,max,min,var,median)
        vec[27:34] = ratio_section[15:22]

        # idx 34-40: TTL backward ratio — not computed for LSTM, build it here
        ttl_bwd_arr = np.full(n, self.DEFAULT_TTL_BWD, dtype=np.float32)
        ttl_bwd_stats = _stats7(ttl_bwd_arr)
        ttl_bwd_enc_stats = _stats7(_safe_log1p(ttl_bwd_arr))
        vec[34:41] = [_safe_ratio(r, e) for r, e in zip(ttl_bwd_stats, ttl_bwd_enc_stats)]

        # idx 41-54: TCP window size ratio fwd/bwd — not observable, left at median

        # idx 55-60: IP packet length ratio group (Total,max,min,var,std,median)
        ip_raw = _stats6(transfer_sizes)  # mean, median, max, min, std, var
        ip_enc = _stats6(_safe_log1p(transfer_sizes))
        ip_total_raw = float(np.sum(transfer_sizes))
        ip_total_enc = float(_safe_log1p(ip_total_raw))
        vec[55] = _safe_ratio(ip_total_raw, ip_total_enc)  # Total
        vec[56] = _safe_ratio(ip_raw[2], ip_enc[2])        # max
        vec[57] = _safe_ratio(ip_raw[3], ip_enc[3])        # min
        vec[58] = _safe_ratio(ip_raw[5], ip_enc[5])        # var
        vec[59] = _safe_ratio(ip_raw[4], ip_enc[4])        # std
        vec[60] = _safe_ratio(ip_raw[1], ip_enc[1])        # median

        # idx 61: Total_Time_to_live_ratio — not observable, left at median

        # idx 62-63: total forward/backward payload ratio
        fwd_payload_total = float(np.sum(req_sizes))
        bwd_payload_total = float(np.sum(resp_sizes))
        vec[62] = _safe_ratio(fwd_payload_total, _safe_log1p(fwd_payload_total))
        vec[63] = _safe_ratio(bwd_payload_total, _safe_log1p(bwd_payload_total))

        # idx 64-65: total forward/backward IP header ratio — exactly computable
        # (constant 20 bytes/packet, so unlike TTL/window size this isn't a guess)
        ip_hdr_total = float(self.DEFAULT_IP_HDR * n)
        ip_hdr_ratio = _safe_ratio(ip_hdr_total, _safe_log1p(ip_hdr_total))
        vec[64] = ip_hdr_ratio
        vec[65] = ip_hdr_ratio

        # idx 66-69: TCP header/segment ratio fwd/bwd — no fwd/bwd split available
        # from Resource Timing data, left at median

        # idx 70: total_payload_per_session_ratio
        total_payload = fwd_payload_total + bwd_payload_total
        vec[70] = _safe_ratio(total_payload, _safe_log1p(total_payload))

        # idx 71: IPratio_ratio — unclear semantics, left at median

        # idx 72-103: ENC_TIME_SESS_COLS (32 values) — already correctly computed
        enc_section = sess_time_stats[35:64]  # 29 enc values
        total_ttl_enc = _safe_log1p(self.DEFAULT_TTL_FWD * n)
        vec[72:101] = enc_section
        vec[101] = total_ttl_enc
        vec[102] = _safe_log1p(fwd_payload_total)
        vec[103] = _safe_log1p(bwd_payload_total)

        xgb_arr = vec.reshape(1, -1).astype(np.float32)
        return np.nan_to_num(xgb_arr, nan=0.0, posinf=0.0, neginf=0.0)
