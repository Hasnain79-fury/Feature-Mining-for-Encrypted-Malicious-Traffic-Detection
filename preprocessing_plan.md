# Dataset Preprocessing Plan
## Encrypted Malicious Traffic Detection Framework
### Wang & Thing (2023) — arXiv:2304.03691

**Dataset**: Mendeley Data — doi:10.17632/xw7r4tt54g.1
**Status**: Dataset-confirmed — all column names verified against actual CSV files

---

## Overview

The preprocessing pipeline is different for each of the three Layer 1 branches. The critical insight is that **the LSTM branch requires features from BOTH CSVs merged together**, while ResNet and XGBoost use the session CSV only.

```
packet_based_trainset.csv (4,435,304 rows × 25 cols)
         │
         ├──► merged with session time cols ──► LSTM Branch   (N, 15, 85)
         │
session_based_trainset.csv (488,524 rows × 280 cols)
         │
         ├──► 38 payload cols ──────────────► ResNet Branch  (N, 1, 38, 38)
         │
         └──► 65 ratio + 32 ENC time cols ──► XGBoost Branch (N, 97)
```

**Run order**:
1. Build LSTM tensor first → captures `uid_list` session order
2. Reindex session CSV using `uid_list` to align all branches
3. Build ResNet + XGBoost tensors from reindexed session CSV
4. Apply one unified train/val split across all three branches
5. Save tensors to disk before training

---

## Part 1: Environment Setup

### 1.1 Directory Structure

```
encrypted_traffic_detection/
├── data/
│   ├── Train Set/
│   │   ├── packet_based_trainset.csv
│   │   └── session_based_trainset.csv
│   └── Test Set/
│       ├── packet_based_testset.csv
│       └── session_based_testset.csv
├── tensors/                        ← preprocessed numpy arrays
│   ├── X_lstm_train.npy
│   ├── X_lstm_val.npy
│   ├── X_lstm_test.npy
│   ├── X_resnet_train.npy
│   ├── X_resnet_val.npy
│   ├── X_resnet_test.npy
│   ├── X_xgb_train.npy
│   ├── X_xgb_val.npy
│   ├── X_xgb_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
├── scalers/
│   ├── scaler_lstm.pkl
│   ├── scaler_resnet_mms.pkl
│   └── selector_resnet.pkl
├── probs/                          ← saved Layer 1 probability outputs
│   ├── lstm_probs_{train,val,test}.npy
│   ├── resnet_probs_{train,val,test}.npy
│   └── xgb_probs_{train,val,test}.npy
└── src/
    └── preprocessing/
        ├── feature_columns.py
        ├── lstm_pipeline.py
        ├── resnet_pipeline.py
        ├── xgb_pipeline.py
        └── run_all.py
```

### 1.2 Dependencies

```bash
pip install pandas numpy scikit-learn xgboost torch torchvision joblib
```

### 1.3 Dataset Facts (confirmed)

| Property | Value |
|----------|-------|
| Packet CSV rows | 4,435,304 |
| Packet CSV columns | 25 |
| Session CSV rows | 488,524 |
| Session CSV columns | 280 |
| Session ID column | `unique_link_mark` |
| Label column | `label` |
| Label encoding | 0 = Benign, 1 = Malicious |
| Class balance | ~243K benign / ~245K malicious |
| Classification type | Binary (num_classes = 2) |

---

## Part 2: Feature Column Definitions

Create `src/preprocessing/feature_columns.py` with all column lists.

### 2.1 LSTM Feature Columns (85 time-related features)

The 85 features are split across two groups:
- **Group A** (7 cols): True per-packet features from the **packet CSV** — different value per packet row
- **Group B** (78 cols): Session-level time aggregates from the **session CSV** — same value broadcast across all 15 packet rows of a session

```python
# ── feature_columns.py ─────────────────────────────────────────────────────

# ── GROUP A: Per-packet time features (from packet CSV) ──────────────────────
# These vary per packet row. They are the only truly sequential part of the input.
PKT_TIME_COLS = [
    # Packet-level time (4)
    'Time_cost',
    'Time_difference_between_packets_per_session',
    'Interval_of_arrival_time_of_forward_traffic',
    'Interval_of_arrival_time_of_backward_traffic',
    # Per-packet ENC time features (3) — also in packet CSV
    'inter_arrival_time_of_forward_traffic_enc',
    'inter_arrival_time_of_backward_traffic_enc',
    'ratio_to_previous_packet_enc',
]
# len(PKT_TIME_COLS) == 7

# ── GROUP B: Session-level time features (from session CSV) ───────────────────
# These are identical across all 15 packet rows for the same session.
# They provide session-level context to the LSTM at every timestep.
SESS_TIME_COLS = [
    # ── Flow duration group (3) ──────────────────────────────────────────────
    'flow duration',
    'flow_duration_of_forward_traffic',
    'flow_duration_of_backward_traffic',

    # ── Time difference between packets — stats (6) ──────────────────────────
    'mean_Time_difference_between_packets_per_session',
    'median_Time_difference_between_packets_per_session',
    'max_Time_difference_between_packets_per_session',
    'min_Time_difference_between_packets_per_session',
    'std_Time_difference_between_packets_per_session',
    'var_Time_difference_between_packets_per_session',

    # ── IAT forward traffic — stats (6) ──────────────────────────────────────
    'mean_Interval_of_arrival_time_of_forward_traffic',
    'median_Interval_of_arrival_time_of_forward_traffic',
    'max_Interval_of_arrival_time_of_forward_traffic',
    'min_Interval_of_arrival_time_of_forward_traffic',
    'std_Interval_of_arrival_time_of_forward_traffic',
    'var_Interval_of_arrival_time_of_forward_traffic',

    # ── IAT backward traffic — stats (6) ─────────────────────────────────────
    'mean_Interval_of_arrival_time_of_backward_traffic',
    'median_Interval_of_arrival_time_of_backward_traffic',
    'max_Interval_of_arrival_time_of_backward_traffic',
    'min_Interval_of_arrival_time_of_backward_traffic',
    'std_Interval_of_arrival_time_of_backward_traffic',
    'var_Interval_of_arrival_time_of_backward_traffic',

    # ── TTL forward traffic — stats (7) ──────────────────────────────────────
    'total_ttl_forward_traffic',
    'std_ttl_forward_traffic',
    'mean_ttl_forward_traffic',
    'max_ttl_forward_traffic',
    'min_ttl_forward_traffic',
    'var_ttl_forward_traffic',
    'median_ttl_forward_traffic',

    # ── TTL backward traffic — stats (7) ─────────────────────────────────────
    'total_ttl_backward_traffic',
    'std_ttl_backward_traffic',
    'mean_ttl_backward_traffic',
    'max_ttl_backward_traffic',
    'min_ttl_backward_traffic',
    'var_ttl_backward_traffic',
    'median_ttl_backward_traffic',

    # ── ENC flow duration group (3) ───────────────────────────────────────────
    'flow_duration_enc',
    'flow_duration_of_forward_traffic_enc',
    'flow_duration_of_backward_traffic_enc',

    # ── ENC IAT forward — stats (6) ──────────────────────────────────────────
    'mean_Interval_of_arrival_time_of_forward_traffic_enc',
    'median_Interval_of_arrival_time_of_forward_traffic_enc',
    'max_Interval_of_arrival_time_of_forward_traffic_enc',
    'min_Interval_of_arrival_time_of_forward_traffic_enc',
    'std_Interval_of_arrival_time_of_forward_traffic_enc',
    'var_Interval_of_arrival_time_of_forward_traffic_enc',

    # ── ENC IAT backward — stats (6) ─────────────────────────────────────────
    'mean_Interval_of_arrival_time_of_backward_traffic_enc',
    'median_Interval_of_arrival_time_of_backward_traffic_enc',
    'max_Interval_of_arrival_time_of_backward_traffic_enc',
    'min_Interval_of_arrival_time_of_backward_traffic_enc',
    'std_Interval_of_arrival_time_of_backward_traffic_enc',
    'var_Interval_of_arrival_time_of_backward_traffic_enc',

    # ── ENC TTL forward — stats (7) ───────────────────────────────────────────
    'total_ttl_forward_traffic_enc',
    'std_ttl_forward_traffic_enc',
    'mean_ttl_forward_traffic_enc',
    'max_ttl_forward_traffic_enc',
    'min_ttl_forward_traffic_enc',
    'var_ttl_forward_traffic_enc',
    'median_ttl_forward_traffic_enc',

    # ── ENC TTL backward — stats (7) ──────────────────────────────────────────
    'total_ttl_backward_traffic_enc',
    'std_ttl_backward_traffic_enc',
    'mean_ttl_backward_traffic_enc',
    'max_ttl_backward_traffic_enc',
    'min_ttl_backward_traffic_enc',
    'var_ttl_backward_traffic_enc',
    'median_ttl_backward_traffic_enc',

    # ── Ratio flow duration group (3) ─────────────────────────────────────────
    'flow_duration_ratio',
    'flow_duration_of_forward_traffic_ratio',
    'flow_duration_of_backward_traffic_ratio',

    # ── Ratio IAT forward — stats (6) ────────────────────────────────────────
    'mean_Interval_of_arrival_time_of_forward_traffic_ratio',
    'median_Interval_of_arrival_time_of_forward_traffic_ratio',
    'max_Interval_of_arrival_time_of_forward_traffic_ratio',
    'min_Interval_of_arrival_time_of_forward_traffic_ratio',
    'std_Interval_of_arrival_time_of_forward_traffic_ratio',
    'var_Interval_of_arrival_time_of_forward_traffic_ratio',

    # ── Ratio IAT backward — stats (6) ───────────────────────────────────────
    'mean_Interval_of_arrival_time_of_backward_traffic_ratio',
    'median_Interval_of_arrival_time_of_backward_traffic_ratio',
    'max_Interval_of_arrival_time_of_backward_traffic_ratio',
    'min_Interval_of_arrival_time_of_backward_traffic_ratio',
    'std_Interval_of_arrival_time_of_backward_traffic_ratio',
    'var_Interval_of_arrival_time_of_backward_traffic_ratio',

    # ── Ratio TTL forward — stats (7) ────────────────────────────────────────
    'total_ttl_forward_traffic_ratio',
    'std_ttl_forward_traffic_ratio',
    'mean_ttl_forward_traffic_ratio',
    'max_ttl_forward_traffic_ratio',
    'min_ttl_forward_traffic_ratio',
    'var_ttl_forward_traffic_ratio',
    'median_ttl_forward_traffic_ratio',
]
# len(SESS_TIME_COLS) == 78

# Combined for LSTM
ALL_TIME_COLS = PKT_TIME_COLS + SESS_TIME_COLS
# len(ALL_TIME_COLS) == 85  ✓


# ── RESNET: Payload columns (48 candidates → select top 38) ──────────────────
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
# len(PAYLOAD_CANDIDATES) == 48
# → SelectKBest(k=38) during preprocessing selects final RESNET_COLS


# ── XGBOOST: Ratio cols (65) + ENC time session cols (32) = 97 ──────────────
RATIO_COLS = None  # populated at runtime: [c for c in df if c.endswith('_ratio')]
# len == 65

ENC_TIME_SESS_COLS = [
    'flow_duration_enc',
    'flow_duration_of_forward_traffic_enc',
    'flow_duration_of_backward_traffic_enc',
    'mean_Interval_of_arrival_time_of_forward_traffic_enc',
    'median_Interval_of_arrival_time_of_forward_traffic_enc',
    'max_Interval_of_arrival_time_of_forward_traffic_enc',
    'min_Interval_of_arrival_time_of_forward_traffic_enc',
    'std_Interval_of_arrival_time_of_forward_traffic_enc',
    'var_Interval_of_arrival_time_of_forward_traffic_enc',
    'mean_Interval_of_arrival_time_of_backward_traffic_enc',
    'median_Interval_of_arrival_time_of_backward_traffic_enc',
    'max_Interval_of_arrival_time_of_backward_traffic_enc',
    'min_Interval_of_arrival_time_of_backward_traffic_enc',
    'std_Interval_of_arrival_time_of_backward_traffic_enc',
    'var_Interval_of_arrival_time_of_backward_traffic_enc',
    'total_ttl_forward_traffic_enc',
    'std_ttl_forward_traffic_enc',
    'mean_ttl_forward_traffic_enc',
    'max_ttl_forward_traffic_enc',
    'min_ttl_forward_traffic_enc',
    'var_ttl_forward_traffic_enc',
    'median_ttl_forward_traffic_enc',
    'total_ttl_backward_traffic_enc',
    'std_ttl_backward_traffic_enc',
    'mean_ttl_backward_traffic_enc',
    'max_ttl_backward_traffic_enc',
    'min_ttl_backward_traffic_enc',
    'var_ttl_backward_traffic_enc',
    'median_ttl_backward_traffic_enc',
    'Total_Time_to_live_enc',
    'Total_length_of_forward_payload_enc',
    'Total_length_of_backward_payload_enc',
]
# len(ENC_TIME_SESS_COLS) == 32
# XGB_COLS = RATIO_COLS (65) + ENC_TIME_SESS_COLS (32) = 97 total
```

---

## Part 3: LSTM Branch Preprocessing

**File**: `src/preprocessing/lstm_pipeline.py`

**Goal**: Build `(N_sessions, 15, 85)` tensor from packet CSV + session CSV merged.

### 3.1 Why Merge Both CSVs

The 85 LSTM features come from two places:

| Feature group | Source | Behaviour in the (15, 85) matrix |
|---------------|--------|----------------------------------|
| 7 per-packet time cols | Packet CSV | **Changes per row** — true temporal sequence |
| 78 session-level time cols | Session CSV | **Same value in all 15 rows** — context broadcast |

This is intentional. The LSTM sees per-packet variation in the first 7 columns (the actual sequence), while the remaining 78 session stats provide fixed context at every timestep.

### 3.2 Step-by-Step

#### Step 1 — Load both CSVs

```python
import pandas as pd
import numpy as np
from feature_columns import PKT_TIME_COLS, SESS_TIME_COLS, ALL_TIME_COLS

df_pkt  = pd.read_csv('Train Set/packet_based_trainset.csv')
df_sess = pd.read_csv('Train Set/session_based_trainset.csv')

print(f"Packet CSV:  {df_pkt.shape}")    # (4,435,304, 25)
print(f"Session CSV: {df_sess.shape}")   # (488,524, 280)
```

#### Step 2 — Verify all 85 columns exist

```python
# Check Group A — packet CSV
missing_pkt = [c for c in PKT_TIME_COLS if c not in df_pkt.columns]
print(f"Missing from packet CSV: {missing_pkt}")   # expect []

# Check Group B — session CSV
missing_sess = [c for c in SESS_TIME_COLS if c not in df_sess.columns]
print(f"Missing from session CSV: {missing_sess}") # expect []

# If any missing: check for typos in column names
# df_sess.columns[df_sess.columns.str.contains('ttl', case=False)]
```

#### Step 3 — Merge session time columns onto packet rows

```python
# Only bring the needed columns from session CSV
sess_slim = df_sess[['unique_link_mark'] + SESS_TIME_COLS]

# Left join: every packet row inherits its session's aggregate stats
df_merged = df_pkt.merge(sess_slim, on='unique_link_mark', how='left')
print(f"Merged shape: {df_merged.shape}")
# Expected: (4,435,304, 25 + 78) = (4,435,304, 103)
```

#### Step 4 — Clean NaN and Inf values

```python
# Check scope
nan_count = df_merged[ALL_TIME_COLS].isna().sum().sum()
inf_count = np.isinf(
    df_merged[ALL_TIME_COLS].select_dtypes('number')
).sum().sum()
print(f"NaN: {nan_count:,}  |  Inf: {inf_count:,}")

# Fix
df_merged[ALL_TIME_COLS] = (
    df_merged[ALL_TIME_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(df_merged[ALL_TIME_COLS].median())
)
print("Clean ✓")
```

#### Step 5 — Build (N, 15, 85) tensor with cutoff and average padding

```python
def build_lstm_tensor(df, feat_cols, max_pkts=15):
    """
    Groups packets by unique_link_mark, applies 15-packet cutoff,
    applies average padding, returns 3D tensor.

    Parameters
    ----------
    df        : merged DataFrame with both packet + session columns
    feat_cols : list of 85 column names (ALL_TIME_COLS)
    max_pkts  : fixed sequence length (paper: 15)

    Returns
    -------
    X       : np.ndarray (N_sessions, 15, 85)
    y       : np.ndarray (N_sessions,)  — labels
    uid_list: list of unique_link_mark values in output order
    """
    sessions, labels, uid_list = [], [], []

    for uid, grp in df.groupby('unique_link_mark', sort=False):
        # Sort chronologically within session
        grp = grp.sort_values('Time_cost')
        mat = grp[feat_cols].values.astype(np.float32)  # (n_pkts, 85)

        # --- Step 5a: Packet cutoff ---
        if len(mat) > max_pkts:
            mat = mat[:max_pkts]

        # --- Step 5b: Average padding (NOT zero-padding) ---
        elif len(mat) < max_pkts:
            pad_val  = mat.mean(axis=0)                      # (85,)
            pad_rows = np.tile(pad_val, (max_pkts - len(mat), 1))
            mat      = np.vstack([mat, pad_rows])            # (15, 85)

        sessions.append(mat)
        labels.append(int(grp['label'].iloc[0]))
        uid_list.append(uid)

    X = np.array(sessions, dtype=np.float32)   # (N, 15, 85)
    y = np.array(labels,   dtype=np.int32)     # (N,)
    return X, y, uid_list


X_lstm, y_lstm, uid_list = build_lstm_tensor(df_merged, ALL_TIME_COLS)
print(f"LSTM tensor:  {X_lstm.shape}")   # (N_sessions, 15, 85)
print(f"Labels:       {y_lstm.shape}")
print(f"Sessions:     {len(uid_list)}")
```

> **Note on broadcast columns**: The 78 session-level columns (SESS_TIME_COLS) will have
> identical values in all 15 rows of each session — this is correct and expected.
> The LSTM captures temporal change from the 7 packet-level columns while using
> the session stats as fixed contextual features at every timestep.

#### Step 6 — Train/val split

```python
from sklearn.model_selection import train_test_split

idx = np.arange(len(X_lstm))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_lstm, random_state=42
)

X_lstm_tr,  X_lstm_val  = X_lstm[idx_tr],  X_lstm[idx_val]
y_tr,       y_val       = y_lstm[idx_tr],  y_lstm[idx_val]

print(f"Train: {X_lstm_tr.shape}  |  Val: {X_lstm_val.shape}")
# Save idx for ResNet and XGBoost alignment
np.save('tensors/idx_tr.npy', idx_tr)
np.save('tensors/idx_val.npy', idx_val)
```

#### Step 7 — Normalise (fit on train only)

```python
from sklearn.preprocessing import StandardScaler
import joblib

N_tr, T, F = X_lstm_tr.shape

sc_lstm = StandardScaler()
X_lstm_tr_sc  = sc_lstm.fit_transform(
    X_lstm_tr.reshape(-1, F)).reshape(N_tr, T, F)
X_lstm_val_sc = sc_lstm.transform(
    X_lstm_val.reshape(-1, F)).reshape(-1, T, F)

joblib.dump(sc_lstm, 'scalers/scaler_lstm.pkl')
print("Scaler saved ✓")
```

#### Step 8 — Save tensors

```python
np.save('tensors/X_lstm_train.npy', X_lstm_tr_sc)
np.save('tensors/X_lstm_val.npy',   X_lstm_val_sc)
np.save('tensors/y_train.npy',      y_tr)
np.save('tensors/y_val.npy',        y_val)
np.save('tensors/uid_list.npy',     np.array(uid_list))
print("LSTM tensors saved ✓")
```

#### Step 9 — Process test set (same pipeline)

```python
df_pkt_test  = pd.read_csv('Test Set/packet_based_testset.csv')
df_sess_test = pd.read_csv('Test Set/session_based_testset.csv')

sess_slim_test = df_sess_test[['unique_link_mark'] + SESS_TIME_COLS]
df_merged_test = df_pkt_test.merge(sess_slim_test, on='unique_link_mark', how='left')
df_merged_test[ALL_TIME_COLS] = (
    df_merged_test[ALL_TIME_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(df_merged_test[ALL_TIME_COLS].median())
)

X_lstm_test, y_test, uid_list_test = build_lstm_tensor(df_merged_test, ALL_TIME_COLS)

N_test = X_lstm_test.shape[0]
X_lstm_test_sc = sc_lstm.transform(
    X_lstm_test.reshape(-1, F)).reshape(N_test, T, F)

np.save('tensors/X_lstm_test.npy', X_lstm_test_sc)
np.save('tensors/y_test.npy',      y_test)
np.save('tensors/uid_list_test.npy', np.array(uid_list_test))
print("LSTM test tensor saved ✓")
```

### 3.3 LSTM Tensor Summary

| Property | Value |
|----------|-------|
| Final shape | `(N_sessions, 15, 85)` |
| Axis 0 | Sessions (unique flows) |
| Axis 1 | Timesteps (15 packets, chronological) |
| Axis 2 | Features (85 time-related) |
| Cols 0–6 | True per-packet values (vary per timestep) |
| Cols 7–84 | Session-level stats (identical across all 15 timesteps) |
| Normalisation | StandardScaler, fit on train split only |
| Padding method | Average padding (NOT zero-padding) |

---

## Part 4: Session CSV Alignment

**Critical step before building ResNet and XGBoost tensors.**

The `build_lstm_tensor()` function processes sessions in groupby order, stored in `uid_list`. The session CSV must be reindexed to exactly this order so all three branch tensors refer to the same session at every row index.

```python
# src/preprocessing/align_session.py

import pandas as pd
import numpy as np

uid_list = np.load('tensors/uid_list.npy', allow_pickle=True).tolist()

df_sess = pd.read_csv('Train Set/session_based_trainset.csv')

# Reindex session CSV to match LSTM tensor row order
df_sess_aligned = (
    df_sess
    .set_index('unique_link_mark')
    .loc[uid_list]
    .reset_index()
)

# Verify alignment
y_lstm = np.load('tensors/y_train.npy')  # only train portion
# Full y_lstm before split:
y_full = np.concatenate([
    np.load('tensors/y_train.npy'),
    np.load('tensors/y_val.npy')
])

assert len(df_sess_aligned) == len(y_full), \
    f"Row mismatch: {len(df_sess_aligned)} vs {len(y_full)}"
assert (df_sess_aligned['label'].values == y_full).all(), \
    "Label mismatch between LSTM tensor and session CSV!"

print(f"Session CSV aligned: {df_sess_aligned.shape}")
print("Alignment verified ✓")
```

> **Why this matters**: Without alignment, row 500 of the LSTM probability array
> could refer to session A, while row 500 of the XGBoost probability array refers
> to session B. Layer 2 would then combine probabilities from different sessions —
> completely corrupting the ensemble.

---

## Part 5: ResNet Branch Preprocessing

**File**: `src/preprocessing/resnet_pipeline.py`

**Goal**: Build `(N_sessions, 1, 38, 38)` image tensor from 38 selected payload columns.

### 5.1 Step-by-Step

#### Step 1 — Select top 38 payload columns via mutual information

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from feature_columns import PAYLOAD_CANDIDATES
import joblib

# Use aligned session CSV
y_full = np.concatenate([np.load('tensors/y_train.npy'),
                          np.load('tensors/y_val.npy')])

X_payload_all = df_sess_aligned[PAYLOAD_CANDIDATES].values

selector = SelectKBest(mutual_info_classif, k=38)
selector.fit(X_payload_all, y_full)

RESNET_COLS = [
    PAYLOAD_CANDIDATES[i]
    for i in selector.get_support(indices=True)
]
joblib.dump(selector, 'scalers/selector_resnet.pkl')
print(f"Selected {len(RESNET_COLS)} ResNet columns")
print(RESNET_COLS)
```

#### Step 2 — Normalise to [0,1]

```python
from sklearn.preprocessing import MinMaxScaler

idx_tr  = np.load('tensors/idx_tr.npy')
idx_val = np.load('tensors/idx_val.npy')

X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)

# Clean
X_payload = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)

mms = MinMaxScaler()
X_payload[idx_tr]  = mms.fit_transform(X_payload[idx_tr])
X_payload[idx_val] = mms.transform(X_payload[idx_val])
joblib.dump(mms, 'scalers/scaler_resnet_mms.pkl')
```

#### Step 3 — Generate 38×38 images via outer product

```python
# Per session: (38,) outer product (38,) = (38, 38)
# Captures feature co-occurrence across the session's payload statistics
X_img = np.einsum('ni,nj->nij', X_payload, X_payload)  # (N, 38, 38)
X_img = X_img[:, np.newaxis, :, :]                      # (N, 1, 38, 38)
print(f"Image tensor shape: {X_img.shape}")
```

> **Also generate 1×38 format for comparison** (paper tests both):
> ```python
> X_img_1x38 = X_payload[:, np.newaxis, np.newaxis, :]  # (N, 1, 1, 38)
> ```

#### Step 4 — Split and save

```python
X_resnet_tr  = X_img[idx_tr]
X_resnet_val = X_img[idx_val]

np.save('tensors/X_resnet_train.npy', X_resnet_tr)
np.save('tensors/X_resnet_val.npy',   X_resnet_val)
print(f"ResNet train: {X_resnet_tr.shape}")   # (N_train, 1, 38, 38)
print(f"ResNet val:   {X_resnet_val.shape}")
```

#### Step 5 — Process test set

```python
uid_list_test = np.load('tensors/uid_list_test.npy', allow_pickle=True)
df_sess_test  = pd.read_csv('Test Set/session_based_testset.csv')
df_sess_test_aligned = (
    df_sess_test
    .set_index('unique_link_mark')
    .loc[uid_list_test]
    .reset_index()
)

X_payload_test = df_sess_test_aligned[RESNET_COLS].values.astype(np.float32)
X_payload_test = np.nan_to_num(X_payload_test, nan=0.0, posinf=0.0, neginf=0.0)
X_payload_test = mms.transform(X_payload_test)

X_img_test = np.einsum('ni,nj->nij', X_payload_test, X_payload_test)
X_img_test = X_img_test[:, np.newaxis, :, :]

np.save('tensors/X_resnet_test.npy', X_img_test)
print(f"ResNet test: {X_img_test.shape}")
```

### 5.2 ResNet Tensor Summary

| Property | Value |
|----------|-------|
| Final shape | `(N_sessions, 1, 38, 38)` |
| Channel | 1 (grayscale) |
| Height × Width | 38 × 38 |
| Generation method | Outer product: `M.T @ M` per session |
| Column selection | SelectKBest mutual_info_classif k=38 from 48 candidates |
| Normalisation | MinMaxScaler [0,1], fit on train split only |
| Alternative format | `(N, 1, 1, 38)` — direct strip image |

---

## Part 6: XGBoost Branch Preprocessing

**File**: `src/preprocessing/xgb_pipeline.py`

**Goal**: Build `(N_sessions, 97)` flat feature matrix from 65 ratio cols + 32 ENC time session cols.

### 6.1 Step-by-Step

#### Step 1 — Define XGBoost columns

```python
from feature_columns import ENC_TIME_SESS_COLS

RATIO_COLS = [c for c in df_sess_aligned.columns if c.endswith('_ratio')]
print(f"Ratio cols: {len(RATIO_COLS)}")         # → 65

XGB_COLS = RATIO_COLS + ENC_TIME_SESS_COLS
print(f"Total XGBoost features: {len(XGB_COLS)}")  # → 97
```

#### Step 2 — Extract and clean

```python
X_xgb = df_sess_aligned[XGB_COLS].values.astype(np.float32)

# Replace Inf with NaN, then impute with column median
X_xgb = np.where(np.isinf(X_xgb), np.nan, X_xgb)
col_medians = np.nanmedian(X_xgb, axis=0)
nan_mask = np.isnan(X_xgb)
X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

print(f"XGBoost matrix: {X_xgb.shape}")         # (N, 97)
print(f"NaN remaining:  {np.isnan(X_xgb).sum()}")  # → 0
```

> **No StandardScaler needed** — XGBoost is a tree-based model and is invariant
> to monotonic feature transformations. Scaling does not affect results.

#### Step 3 — Split and save

```python
X_xgb_tr  = X_xgb[idx_tr]
X_xgb_val = X_xgb[idx_val]

np.save('tensors/X_xgb_train.npy', X_xgb_tr)
np.save('tensors/X_xgb_val.npy',   X_xgb_val)
print(f"XGBoost train: {X_xgb_tr.shape}")   # (N_train, 97)
print(f"XGBoost val:   {X_xgb_val.shape}")
```

#### Step 4 — Process test set

```python
X_xgb_test = df_sess_test_aligned[XGB_COLS].values.astype(np.float32)
X_xgb_test = np.where(np.isinf(X_xgb_test), np.nan, X_xgb_test)
X_xgb_test[np.isnan(X_xgb_test)] = np.take(
    col_medians, np.where(np.isnan(X_xgb_test))[1])

np.save('tensors/X_xgb_test.npy', X_xgb_test)
print(f"XGBoost test: {X_xgb_test.shape}")
```

### 6.2 XGBoost Feature Summary

| Feature group | Source | Count | Why included |
|---------------|--------|-------|-------------|
| `_ratio` columns | Session CSV | 65 | Paper-exact ENC ratio features |
| ENC IAT fwd/bwd stats | Session CSV | 12 | ENC time context |
| ENC TTL fwd/bwd stats | Session CSV | 14 | ENC TTL patterns |
| ENC flow duration | Session CSV | 3 | Session encryption duration |
| Total_Time_to_live_enc | Session CSV | 1 | Session-level ENC TTL |
| ENC payload totals | Session CSV | 2 | Payload encryption scope |
| **Total** | | **97** | |

---

## Part 7: Run All Preprocessing

**File**: `src/preprocessing/run_all.py`

```python
"""
Master preprocessing script. Run once before any model training.
Estimated runtime: 20–40 minutes depending on hardware (4.4M packet rows).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

from feature_columns import (
    PKT_TIME_COLS, SESS_TIME_COLS, ALL_TIME_COLS,
    PAYLOAD_CANDIDATES, ENC_TIME_SESS_COLS
)

os.makedirs('tensors', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

print("=" * 60)
print("STEP 1: Loading CSVs")
print("=" * 60)
df_pkt  = pd.read_csv('Train Set/packet_based_trainset.csv')
df_sess = pd.read_csv('Train Set/session_based_trainset.csv')
print(f"Packet:  {df_pkt.shape}")
print(f"Session: {df_sess.shape}")

# ── LSTM ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: LSTM preprocessing — merging and building (N,15,85) tensor")
print("=" * 60)

sess_slim = df_sess[['unique_link_mark'] + SESS_TIME_COLS]
df_merged = df_pkt.merge(sess_slim, on='unique_link_mark', how='left')
df_merged[ALL_TIME_COLS] = (
    df_merged[ALL_TIME_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(df_merged[ALL_TIME_COLS].median())
)

def build_lstm_tensor(df, feat_cols, max_pkts=15):
    sessions, labels, uid_list = [], [], []
    for uid, grp in df.groupby('unique_link_mark', sort=False):
        grp = grp.sort_values('Time_cost')
        mat = grp[feat_cols].values.astype(np.float32)
        if len(mat) > max_pkts:
            mat = mat[:max_pkts]
        elif len(mat) < max_pkts:
            pad_val  = mat.mean(axis=0)
            pad_rows = np.tile(pad_val, (max_pkts - len(mat), 1))
            mat      = np.vstack([mat, pad_rows])
        sessions.append(mat)
        labels.append(int(grp['label'].iloc[0]))
        uid_list.append(uid)
    return (np.array(sessions, dtype=np.float32),
            np.array(labels,   dtype=np.int32),
            uid_list)

X_lstm, y_lstm, uid_list = build_lstm_tensor(df_merged, ALL_TIME_COLS)
print(f"LSTM tensor: {X_lstm.shape}")

# Split
idx = np.arange(len(X_lstm))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_lstm, random_state=42)

y_tr, y_val = y_lstm[idx_tr], y_lstm[idx_val]

# Scale LSTM
N_tr, T, F = X_lstm[idx_tr].shape
sc_lstm = StandardScaler()
X_lstm_tr_sc  = sc_lstm.fit_transform(
    X_lstm[idx_tr].reshape(-1, F)).reshape(N_tr, T, F)
X_lstm_val_sc = sc_lstm.transform(
    X_lstm[idx_val].reshape(-1, F)).reshape(-1, T, F)
joblib.dump(sc_lstm, 'scalers/scaler_lstm.pkl')

# Save
np.save('tensors/X_lstm_train.npy', X_lstm_tr_sc)
np.save('tensors/X_lstm_val.npy',   X_lstm_val_sc)
np.save('tensors/y_train.npy',      y_tr)
np.save('tensors/y_val.npy',        y_val)
np.save('tensors/idx_tr.npy',       idx_tr)
np.save('tensors/idx_val.npy',      idx_val)
np.save('tensors/uid_list.npy',     np.array(uid_list))
print("LSTM tensors saved ✓")

# ── ALIGN SESSION CSV ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Aligning session CSV to LSTM session order")
print("=" * 60)

df_sess_aligned = (
    df_sess
    .set_index('unique_link_mark')
    .loc[uid_list]
    .reset_index()
)
assert (df_sess_aligned['label'].values == y_lstm).all(), "Label mismatch!"
print(f"Session CSV aligned: {df_sess_aligned.shape} ✓")

# ── RESNET ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: ResNet preprocessing — 48→38 payload cols → (N,1,38,38)")
print("=" * 60)

selector = SelectKBest(mutual_info_classif, k=38)
selector.fit(df_sess_aligned[PAYLOAD_CANDIDATES].values, y_lstm)
RESNET_COLS = [PAYLOAD_CANDIDATES[i] for i in selector.get_support(indices=True)]
joblib.dump(selector, 'scalers/selector_resnet.pkl')
print(f"ResNet cols selected: {len(RESNET_COLS)}")

X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)
X_payload  = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)

mms = MinMaxScaler()
X_payload[idx_tr]  = mms.fit_transform(X_payload[idx_tr])
X_payload[idx_val] = mms.transform(X_payload[idx_val])
joblib.dump(mms, 'scalers/scaler_resnet_mms.pkl')

X_img = np.einsum('ni,nj->nij', X_payload, X_payload)[:, np.newaxis]
np.save('tensors/X_resnet_train.npy', X_img[idx_tr])
np.save('tensors/X_resnet_val.npy',   X_img[idx_val])
print(f"ResNet train: {X_img[idx_tr].shape} ✓")

# ── XGBOOST ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: XGBoost preprocessing — 65+32=97 features → (N,97)")
print("=" * 60)

RATIO_COLS = [c for c in df_sess_aligned.columns if c.endswith('_ratio')]
XGB_COLS   = RATIO_COLS + ENC_TIME_SESS_COLS
print(f"XGBoost features: {len(XGB_COLS)}")

X_xgb       = df_sess_aligned[XGB_COLS].values.astype(np.float32)
X_xgb       = np.where(np.isinf(X_xgb), np.nan, X_xgb)
col_medians  = np.nanmedian(X_xgb, axis=0)
nan_mask     = np.isnan(X_xgb)
X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

np.save('tensors/X_xgb_train.npy', X_xgb[idx_tr])
np.save('tensors/X_xgb_val.npy',   X_xgb[idx_val])
print(f"XGBoost train: {X_xgb[idx_tr].shape} ✓")

# ── DONE ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  LSTM:    tensors/X_lstm_train.npy  {X_lstm_tr_sc.shape}")
print(f"  ResNet:  tensors/X_resnet_train.npy {X_img[idx_tr].shape}")
print(f"  XGBoost: tensors/X_xgb_train.npy   {X_xgb[idx_tr].shape}")
print(f"  Labels:  tensors/y_train.npy        {y_tr.shape}")
```

---

## Part 8: Verification Checklist

Run these checks after `run_all.py` completes:

```python
# verification.py
import numpy as np

X_lstm   = np.load('tensors/X_lstm_train.npy')
X_resnet = np.load('tensors/X_resnet_train.npy')
X_xgb    = np.load('tensors/X_xgb_train.npy')
y        = np.load('tensors/y_train.npy')

print("Shape checks:")
print(f"  LSTM:    {X_lstm.shape}   ← expect (N, 15, 85)")
print(f"  ResNet:  {X_resnet.shape} ← expect (N, 1, 38, 38)")
print(f"  XGBoost: {X_xgb.shape}   ← expect (N, 97)")
print(f"  Labels:  {y.shape}        ← expect (N,)")

assert X_lstm.shape[0]   == X_resnet.shape[0] == X_xgb.shape[0] == y.shape[0], \
    "Row count mismatch across branches!"
assert X_lstm.shape[1]   == 15,  f"Wrong timesteps: {X_lstm.shape[1]}"
assert X_lstm.shape[2]   == 85,  f"Wrong features: {X_lstm.shape[2]}"
assert X_resnet.shape[1] == 1,   f"Wrong channels: {X_resnet.shape[1]}"
assert X_resnet.shape[2] == 38,  f"Wrong H: {X_resnet.shape[2]}"
assert X_resnet.shape[3] == 38,  f"Wrong W: {X_resnet.shape[3]}"
assert X_xgb.shape[1]    == 97,  f"Wrong XGB features: {X_xgb.shape[1]}"

print("\nNaN/Inf checks:")
print(f"  LSTM NaN:    {np.isnan(X_lstm).sum()}")
print(f"  ResNet NaN:  {np.isnan(X_resnet).sum()}")
print(f"  XGBoost NaN: {np.isnan(X_xgb).sum()}")

print("\nClass distribution:")
print(f"  Benign:    {(y==0).sum()} ({(y==0).mean():.1%})")
print(f"  Malicious: {(y==1).sum()} ({(y==1).mean():.1%})")

print("\nAll checks passed ✓")
```

---

## Part 9: Summary Table

| Branch | Source CSVs | Key steps | Output shape | Scaler |
|--------|-------------|-----------|--------------|--------|
| **LSTM** | Packet + Session (merged) | merge → clean → groupby → cutoff → avg-pad → scale | `(N, 15, 85)` | StandardScaler |
| **ResNet** | Session only (aligned) | SelectKBest k=38 → MinMaxScale → outer product | `(N, 1, 38, 38)` | MinMaxScaler |
| **XGBoost** | Session only (aligned) | select 97 cols → clean → no scaling | `(N, 97)` | None |
| **Layer 2** | Prob outputs | concat probs → RF or avg ensemble | `(N, 6)` | None |

### Critical Rules

1. **Build LSTM first** — it produces `uid_list` that all other branches must follow
2. **Align session CSV** — reindex by `uid_list` before building ResNet and XGBoost
3. **One split index** — `idx_tr` and `idx_val` from LSTM must be applied to all branches
4. **Fit scalers on train only** — StandardScaler and MinMaxScaler fit on `[idx_tr]` rows, then transform `[idx_val]` and test set
5. **Average padding, not zero** — `pad_val = mat.mean(axis=0)` per session
6. **Session columns broadcast** — it is correct that the 78 SESS_TIME_COLS are identical across all 15 rows of the same session
7. **No test leakage** — test set processed separately using scalers/selector fitted on train data only
