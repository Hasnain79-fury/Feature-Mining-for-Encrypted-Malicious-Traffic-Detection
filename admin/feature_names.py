"""
Human-readable names for the model's feature vectors, used only to label
SHAP output. Transcribed index-by-index from backend/feature_transform.py's
_build_xgb() docstring/comments — never imported by the live backend, and
never affects any prediction; display metadata only.
"""

XGB_FEATURE_NAMES = (
    # idx 0-5: forward packet-length ratio (std,mean,max,min,var,median)
    ["forward_packet_length_std_ratio", "forward_packet_length_mean_ratio",
     "forward_packet_length_max_ratio", "forward_packet_length_min_ratio",
     "forward_packet_length_var_ratio", "forward_packet_length_median_ratio"]
    # idx 6-11: backward packet-length ratio
    + ["backward_packet_length_std_ratio", "backward_packet_length_mean_ratio",
       "backward_packet_length_max_ratio", "backward_packet_length_min_ratio",
       "backward_packet_length_var_ratio", "backward_packet_length_median_ratio"]
    # idx 12-14: flow duration ratio
    + ["flow_duration_ratio", "flow_duration_forward_ratio", "flow_duration_backward_ratio"]
    # idx 15-20: IAT forward ratio (mean,median,max,min,std,var)
    + ["iat_forward_mean_ratio", "iat_forward_median_ratio", "iat_forward_max_ratio",
       "iat_forward_min_ratio", "iat_forward_std_ratio", "iat_forward_var_ratio"]
    # idx 21-26: IAT backward ratio
    + ["iat_backward_mean_ratio", "iat_backward_median_ratio", "iat_backward_max_ratio",
       "iat_backward_min_ratio", "iat_backward_std_ratio", "iat_backward_var_ratio"]
    # idx 27-33: TTL forward ratio (total,std,mean,max,min,var,median)
    + ["ttl_forward_total_ratio", "ttl_forward_std_ratio", "ttl_forward_mean_ratio",
       "ttl_forward_max_ratio", "ttl_forward_min_ratio", "ttl_forward_var_ratio",
       "ttl_forward_median_ratio"]
    # idx 34-40: TTL backward ratio
    + ["ttl_backward_total_ratio", "ttl_backward_std_ratio", "ttl_backward_mean_ratio",
       "ttl_backward_max_ratio", "ttl_backward_min_ratio", "ttl_backward_var_ratio",
       "ttl_backward_median_ratio"]
    # idx 41-47: TCP window size ratio forward
    + ["tcp_window_forward_total_ratio", "tcp_window_forward_std_ratio",
       "tcp_window_forward_mean_ratio", "tcp_window_forward_max_ratio",
       "tcp_window_forward_min_ratio", "tcp_window_forward_var_ratio",
       "tcp_window_forward_median_ratio"]
    # idx 48-54: TCP window size ratio backward
    + ["tcp_window_backward_total_ratio", "tcp_window_backward_std_ratio",
       "tcp_window_backward_mean_ratio", "tcp_window_backward_max_ratio",
       "tcp_window_backward_min_ratio", "tcp_window_backward_var_ratio",
       "tcp_window_backward_median_ratio"]
    # idx 55-60: IP packet length ratio (total,max,min,var,std,median)
    + ["ip_packet_length_total_ratio", "ip_packet_length_max_ratio",
       "ip_packet_length_min_ratio", "ip_packet_length_var_ratio",
       "ip_packet_length_std_ratio", "ip_packet_length_median_ratio"]
    # idx 61: total TTL ratio (always training-median, see ALWAYS_MEDIAN_INDICES)
    + ["total_time_to_live_ratio"]
    # idx 62-63: total forward/backward payload ratio
    + ["forward_payload_total_ratio", "backward_payload_total_ratio"]
    # idx 64-65: total forward/backward IP header ratio
    + ["ip_header_forward_total_ratio", "ip_header_backward_total_ratio"]
    # idx 66-69: TCP header/segment ratio fwd/bwd (always training-median)
    + ["tcp_header_forward_ratio", "tcp_header_backward_ratio",
       "tcp_segment_forward_ratio", "tcp_segment_backward_ratio"]
    # idx 70: total payload per session ratio
    + ["total_payload_per_session_ratio"]
    # idx 71: unclear semantics (always training-median)
    + ["ip_ratio_ratio"]
    # idx 72-74: flow duration enc (total, forward, backward)
    + ["flow_duration_enc", "flow_duration_forward_enc", "flow_duration_backward_enc"]
    # idx 75-80: IAT forward enc (mean,median,max,min,std,var)
    + ["iat_forward_mean_enc", "iat_forward_median_enc", "iat_forward_max_enc",
       "iat_forward_min_enc", "iat_forward_std_enc", "iat_forward_var_enc"]
    # idx 81-86: IAT backward enc
    + ["iat_backward_mean_enc", "iat_backward_median_enc", "iat_backward_max_enc",
       "iat_backward_min_enc", "iat_backward_std_enc", "iat_backward_var_enc"]
    # idx 87-93: TTL forward enc (total,std,mean,max,min,var,median)
    + ["ttl_forward_total_enc", "ttl_forward_std_enc", "ttl_forward_mean_enc",
       "ttl_forward_max_enc", "ttl_forward_min_enc", "ttl_forward_var_enc",
       "ttl_forward_median_enc"]
    # idx 94-100: TTL backward enc
    + ["ttl_backward_total_enc", "ttl_backward_std_enc", "ttl_backward_mean_enc",
       "ttl_backward_max_enc", "ttl_backward_min_enc", "ttl_backward_var_enc",
       "ttl_backward_median_enc"]
    # idx 101-103
    + ["total_ttl_enc", "forward_payload_total_enc", "backward_payload_total_enc"]
)

assert len(XGB_FEATURE_NAMES) == 104, f"expected 104 names, got {len(XGB_FEATURE_NAMES)}"

# Indices that backend/feature_transform.py's _build_xgb() leaves at the
# training median NO MATTER what input path is used (browser or pcap) —
# their SHAP contribution is therefore identical for every session, never
# session-specific. Flagged in the UI rather than presented as if derived
# from this particular flow.
ALWAYS_MEDIAN_INDICES = frozenset({61, 66, 67, 68, 69, 71})

# The Layer-2 Random Forest's 6 meta-inputs: concatenated [p_lstm, p_resnet, p_xgb],
# each a [prob_benign, prob_malicious] pair (see inference.py's `meta = np.hstack(...)`).
META_FEATURE_NAMES = [
    "LSTM → benign probability", "LSTM → malicious probability",
    "ResNet-34 → benign probability", "ResNet-34 → malicious probability",
    "XGBoost → benign probability", "XGBoost → malicious probability",
]
