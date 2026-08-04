# Encrypted Malicious Traffic Detection Framework

## Implementation Plan

**Paper Reference**: Wang & Thing - "Feature Mining for Encrypted Malicious Traffic Detection with Deep Learning and Other Machine Learning Algorithms" (2023)

**Dataset**: Mendeley Data - Encrypted Traffic Feature Dataset for Machine Learning and Deep Learning based Encrypted Traffic Analysis

**Status**: Dataset-confirmed version — all feature counts, shapes, and column assignments verified against actual CSV files.

---

## 1. Framework Architecture Overview

### 1.1 Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INPUT: Two CSV Files (pre-processed)                     │
│  packet_based_trainset.csv (4,435,304 rows × 25 cols)                       │
│  session_based_trainset.csv (488,524 rows × 280 cols)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                                    │
│  Step 1 — LSTM: merge packet+session CSVs on unique_link_mark               │
│            → groupby → cutoff 15 → average padding → (N, 15, 85)            │
│  Step 2 — Align: reindex session CSV by uid_list from LSTM groupby          │
│  Step 3 — ResNet/XGBoost: slice column groups from ALIGNED session CSV      │
│            (no grouping needed — already session-level)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Layer 1.1      │       │  Layer 1.2      │       │  Layer 1.3      │
│  LSTM Branch    │       │  ResNet Branch  │       │  XGBoost Branch │
│                 │       │                 │       │                 │
│ SOURCE:         │       │ SOURCE:         │       │ SOURCE:         │
│ packet CSV +    │       │ session CSV     │       │ session CSV     │
│ session CSV     │       │ (aligned)       │       │ (aligned)       │
│ (merged)        │       │                 │       │                 │
│                 │       │                 │       │                 │
│ 85 time feats   │       │ 38 payload cols │       │ 65 ratio +      │
│ (7 packet +     │       │ shape (1,38,38) │       │ 32 ENC time     │
│  78 session)    │       │                 │       │ = 97 cols       │
│ shape (15, 85)  │       │                 │       │ shape (N, 97)   │
│                 │       │                 │       │                 │
│ Multi-layer     │       │ ResNet34        │       │ XGBClassifier   │
│ LSTM(128×2)     │       │ 1-channel input │       │ binary:logistic │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         ▼                         ▼                         ▼
    probs (N,2)              probs (N,2)               probs (N,2)
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │  concat → (N, 6)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: ENSEMBLE DETECTOR                               │
│                                                                              │
│  Option A: Random Forest (Priority: TPR/Recall)                             │
│  Option B: Average Ensemble (Priority: FPR minimization)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              FINAL PREDICTION: 0 = Benign  /  1 = Malicious                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Critical**: The LSTM branch requires features from BOTH CSVs merged together — 7 true
> per-packet columns (vary per timestep) plus 78 session-level columns (broadcast identically
> across all 15 timesteps of a session). All session order must be captured via `uid_list`
> from the LSTM groupby step, then used to reindex the session CSV before building the
> ResNet and XGBoost tensors. This guarantees all three branches reference identical
> sessions in identical row order — essential for a valid Layer 2 ensemble.

### 1.2 Target Performance Metrics

| Metric | Target Value | Layer 2 Method |
|--------|-------------|----------------|
| Accuracy | 99.73% | Average Ensemble |
| F1 Score | 99.72% | Average Ensemble |
| Precision | 99.89% | Average Ensemble |
| Recall/TPR | 99.56% | Average Ensemble |
| FPR | 0.11% | Average Ensemble |
| ROC-AUC | 99.94% | Average Ensemble |

---

## 2. Dataset Description

### 2.1 Files and Shapes

| File | Rows | Columns | Used by |
|------|------|---------|---------|
| `Train Set/packet_based_trainset.csv` | 4,435,304 | 25 | LSTM branch (merged with session CSV) |
| `Train Set/session_based_trainset.csv` | 488,524 | 280 | LSTM branch (merged) + ResNet + XGBoost branches |
| `Test Set/packet_based_testset.csv` | ~900K | 25 | LSTM evaluation (merged with session CSV) |
| `Test Set/session_based_testset.csv` | ~100K | 280 | LSTM evaluation (merged) + ResNet + XGBoost evaluation |

> **Note**: Use the pre-defined Train/Test split. Do NOT create your own 70/15/15 split. Carve a validation set from the Train Set only (85/15 stratified split).

### 2.2 Label Column

- **Column name**: `label`
- **Values**: `0` = Benign, `1` = Malicious
- **Distribution** (session CSV): 243,054 benign / 245,470 malicious — near-balanced
- **Binary classification** — `num_classes = 2` throughout

### 2.3 Session ID Column

- **Column name**: `unique_link_mark`
- Packets in the packet CSV are grouped into sessions using this identifier
- Both CSVs share this column, linking packet rows to session rows

### 2.4 Dataset Composition

**Malicious Traffic** (~245K sessions across 26 malware families):
Ammyy, Artemis Trojan, Barys, Bunitu Botnet, Caphaw/Kazy, Cerber Ransomware, Dridex, HtBot, Miuref, omQUd, PUA.Taobao, Ransom.Locky, Razy, Sathurbot, TrickBot, Trickster, Trojan.Banker, Trojan.Yakes, Trojan.Downloader, Upatre, Ursnif, Vawtrak, WisdomEyes, Zbot, HPEmotet

**Benign Traffic** (~243K sessions):
CIRA-CIC-DoHBRW-2020, CICIDS-2017, CICIDS-2012, and benign captures

---

## 3. Preprocessing Pipeline

### 3.1 Overview

**Run order matters.** The LSTM branch requires merging both CSVs, which produces a session
ordering (`uid_list`) that the other two branches must align to:

```
STEP 1 — LSTM (build first — establishes session order)
  packet_based_trainset.csv  ──┐
                                ├──► merge on unique_link_mark
  session_based_trainset.csv ──┘
        │
        ▼
  merged DataFrame: 7 per-packet cols (vary per row) +
                     78 session-level cols (broadcast per session)
        │
        ▼
  group by unique_link_mark → sort by Time_cost
  → cutoff at 15 packets → average padding for short sessions
        │
        ▼
  shape: (N_sessions, 15, 85)  →  LSTM Branch
  also produces: uid_list  (session order — required for Step 2)

STEP 2 — Align session CSV to uid_list
  df_sess.set_index('unique_link_mark').loc[uid_list].reset_index()
  → guarantees ResNet/XGBoost rows match LSTM session order exactly

STEP 3 — ResNet / XGBoost (from ALIGNED session CSV — no grouping needed)
  slice PAYLOAD_COLS (38 of 48 candidates, via SelectKBest) → ResNet Branch  shape: (N, 38) → image (38,38)
  slice RATIO_COLS (65) + ENC_TIME_SESS_COLS (32) = 97       → XGBoost Branch shape: (N, 97)
```

> **Why this order is mandatory**: `build_lstm_tensor()` processes sessions via
> `groupby('unique_link_mark')`, which does not preserve the session CSV's original row
> order. If ResNet and XGBoost are built independently with their own `train_test_split`
> calls, their session ordering will NOT match the LSTM tensor's ordering — even with an
> identical `random_state`. Concatenating mismatched probability vectors in Layer 2 silently
> corrupts the ensemble: row *i* of `lstm_probs` would describe a different session than row
> *i* of `xgb_probs`. Building LSTM first and aligning the session CSV to its `uid_list`
> eliminates this risk entirely.

### 3.2 LSTM Pipeline — Merge Packet CSV + Session CSV

The paper's 85 time-related features split into two groups by source:

| Group | Count | Source | Behaviour across the 15 timesteps |
|-------|-------|--------|-----------------------------------|
| Per-packet time features | 7 | Packet CSV | **Varies per row** — the true temporal sequence |
| Session-level time stats | 78 | Session CSV | **Identical across all 15 rows** — broadcast context |

This is intentional, not a workaround: the LSTM observes genuine temporal variation in the
7 packet-level columns while every timestep also carries the session's aggregate time
statistics as fixed context. **85 features are fully achievable — no dataset constraint
applies here.**

#### Step 1: Load both CSVs

```python
import pandas as pd
import numpy as np

df_pkt  = pd.read_csv('Train Set/packet_based_trainset.csv')
df_sess = pd.read_csv('Train Set/session_based_trainset.csv')
print(df_pkt.shape)   # (4,435,304, 25)
print(df_sess.shape)  # (488,524, 280)
```

#### Step 2: Define the 85 time-related columns

```python
# ── Group A: 7 true per-packet columns (packet CSV) ─────────────────────────
PKT_TIME_COLS = [
    'Time_cost',
    'Time_difference_between_packets_per_session',
    'Interval_of_arrival_time_of_forward_traffic',
    'Interval_of_arrival_time_of_backward_traffic',
    'inter_arrival_time_of_forward_traffic_enc',
    'inter_arrival_time_of_backward_traffic_enc',
    'ratio_to_previous_packet_enc',
]  # len == 7

# ── Group B: 78 session-level columns (session CSV) — broadcast per session ─
SESS_TIME_COLS = [
    # Flow duration (3)
    'flow duration', 'flow_duration_of_forward_traffic',
    'flow_duration_of_backward_traffic',
    # Time difference between packets — stats (6)
    'mean_Time_difference_between_packets_per_session',
    'median_Time_difference_between_packets_per_session',
    'max_Time_difference_between_packets_per_session',
    'min_Time_difference_between_packets_per_session',
    'std_Time_difference_between_packets_per_session',
    'var_Time_difference_between_packets_per_session',
    # IAT forward — stats (6)
    'mean_Interval_of_arrival_time_of_forward_traffic',
    'median_Interval_of_arrival_time_of_forward_traffic',
    'max_Interval_of_arrival_time_of_forward_traffic',
    'min_Interval_of_arrival_time_of_forward_traffic',
    'std_Interval_of_arrival_time_of_forward_traffic',
    'var_Interval_of_arrival_time_of_forward_traffic',
    # IAT backward — stats (6)
    'mean_Interval_of_arrival_time_of_backward_traffic',
    'median_Interval_of_arrival_time_of_backward_traffic',
    'max_Interval_of_arrival_time_of_backward_traffic',
    'min_Interval_of_arrival_time_of_backward_traffic',
    'std_Interval_of_arrival_time_of_backward_traffic',
    'var_Interval_of_arrival_time_of_backward_traffic',
    # TTL forward — stats (7)
    'total_ttl_forward_traffic', 'std_ttl_forward_traffic',
    'mean_ttl_forward_traffic', 'max_ttl_forward_traffic',
    'min_ttl_forward_traffic', 'var_ttl_forward_traffic',
    'median_ttl_forward_traffic',
    # TTL backward — stats (7)
    'total_ttl_backward_traffic', 'std_ttl_backward_traffic',
    'mean_ttl_backward_traffic', 'max_ttl_backward_traffic',
    'min_ttl_backward_traffic', 'var_ttl_backward_traffic',
    'median_ttl_backward_traffic',
    # ENC flow duration (3)
    'flow_duration_enc', 'flow_duration_of_forward_traffic_enc',
    'flow_duration_of_backward_traffic_enc',
    # ENC IAT forward — stats (6)
    'mean_Interval_of_arrival_time_of_forward_traffic_enc',
    'median_Interval_of_arrival_time_of_forward_traffic_enc',
    'max_Interval_of_arrival_time_of_forward_traffic_enc',
    'min_Interval_of_arrival_time_of_forward_traffic_enc',
    'std_Interval_of_arrival_time_of_forward_traffic_enc',
    'var_Interval_of_arrival_time_of_forward_traffic_enc',
    # ENC IAT backward — stats (6)
    'mean_Interval_of_arrival_time_of_backward_traffic_enc',
    'median_Interval_of_arrival_time_of_backward_traffic_enc',
    'max_Interval_of_arrival_time_of_backward_traffic_enc',
    'min_Interval_of_arrival_time_of_backward_traffic_enc',
    'std_Interval_of_arrival_time_of_backward_traffic_enc',
    'var_Interval_of_arrival_time_of_backward_traffic_enc',
    # ENC TTL forward — stats (7)
    'total_ttl_forward_traffic_enc', 'std_ttl_forward_traffic_enc',
    'mean_ttl_forward_traffic_enc', 'max_ttl_forward_traffic_enc',
    'min_ttl_forward_traffic_enc', 'var_ttl_forward_traffic_enc',
    'median_ttl_forward_traffic_enc',
    # ENC TTL backward — stats (7)
    'total_ttl_backward_traffic_enc', 'std_ttl_backward_traffic_enc',
    'mean_ttl_backward_traffic_enc', 'max_ttl_backward_traffic_enc',
    'min_ttl_backward_traffic_enc', 'var_ttl_backward_traffic_enc',
    'median_ttl_backward_traffic_enc',
    # Ratio flow duration (3)
    'flow_duration_ratio', 'flow_duration_of_forward_traffic_ratio',
    'flow_duration_of_backward_traffic_ratio',
    # Ratio IAT forward — stats (6)
    'mean_Interval_of_arrival_time_of_forward_traffic_ratio',
    'median_Interval_of_arrival_time_of_forward_traffic_ratio',
    'max_Interval_of_arrival_time_of_forward_traffic_ratio',
    'min_Interval_of_arrival_time_of_forward_traffic_ratio',
    'std_Interval_of_arrival_time_of_forward_traffic_ratio',
    'var_Interval_of_arrival_time_of_forward_traffic_ratio',
    # Ratio IAT backward — stats (6)
    'mean_Interval_of_arrival_time_of_backward_traffic_ratio',
    'median_Interval_of_arrival_time_of_backward_traffic_ratio',
    'max_Interval_of_arrival_time_of_backward_traffic_ratio',
    'min_Interval_of_arrival_time_of_backward_traffic_ratio',
    'std_Interval_of_arrival_time_of_backward_traffic_ratio',
    'var_Interval_of_arrival_time_of_backward_traffic_ratio',
    # Ratio TTL forward — stats (7)
    'total_ttl_forward_traffic_ratio', 'std_ttl_forward_traffic_ratio',
    'mean_ttl_forward_traffic_ratio', 'max_ttl_forward_traffic_ratio',
    'min_ttl_forward_traffic_ratio', 'var_ttl_forward_traffic_ratio',
    'median_ttl_forward_traffic_ratio',
]  # len == 78

ALL_TIME_COLS = PKT_TIME_COLS + SESS_TIME_COLS
print(len(ALL_TIME_COLS))  # → 85  ✓
```

#### Step 3: Merge session time columns onto packet rows

```python
sess_slim  = df_sess[['unique_link_mark'] + SESS_TIME_COLS]
df_merged  = df_pkt.merge(sess_slim, on='unique_link_mark', how='left')
print(df_merged.shape)  # (4,435,304, 25 + 78) = (4,435,304, 103)
```

#### Step 4: Clean NaN / Inf in the merged columns

```python
df_merged[ALL_TIME_COLS] = (
    df_merged[ALL_TIME_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(df_merged[ALL_TIME_COLS].median())
)
```

#### Step 5: Group, cutoff, and average-pad sessions → (N, 15, 85)

```python
def build_lstm_tensor(df, feat_cols, max_pkts=15):
    """
    Groups merged rows by session, applies 15-packet cutoff and
    average padding. Returns X (N,15,85), y (N,), and uid_list —
    the session order that ResNet/XGBoost must align to.
    """
    sessions, labels, uid_list = [], [], []

    for uid, grp in df.groupby('unique_link_mark', sort=False):
        grp_sorted = grp.sort_values('Time_cost')
        mat = grp_sorted[feat_cols].values.astype(np.float32)  # (n_pkts, 85)

        if len(mat) > max_pkts:
            mat = mat[:max_pkts]
        elif len(mat) < max_pkts:
            # Average padding (NOT zero-padding)
            pad_val  = mat.mean(axis=0)
            pad_rows = np.tile(pad_val, (max_pkts - len(mat), 1))
            mat      = np.vstack([mat, pad_rows])

        sessions.append(mat)
        labels.append(int(grp_sorted['label'].iloc[0]))
        uid_list.append(uid)

    X = np.array(sessions, dtype=np.float32)  # (N, 15, 85)
    y = np.array(labels,   dtype=np.int32)     # (N,)
    return X, y, uid_list

X_lstm, y_lstm, uid_list = build_lstm_tensor(df_merged, ALL_TIME_COLS)
print(X_lstm.shape)  # (N_sessions, 15, 85)  ← 85 confirmed, no compromise needed
```

> Note on broadcast columns: the 78 `SESS_TIME_COLS` will have identical values across
> all 15 rows of a session — this is correct and expected, not a bug.

#### Step 6: Train/val split — capture `idx_tr` / `idx_val` for all branches

```python
from sklearn.model_selection import train_test_split

idx = np.arange(len(X_lstm))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_lstm, random_state=42)

X_lstm_tr,  X_lstm_val  = X_lstm[idx_tr],  X_lstm[idx_val]
y_tr,       y_val       = y_lstm[idx_tr],  y_lstm[idx_val]

# Save for ResNet/XGBoost alignment in Section 3.3
np.save('tensors/idx_tr.npy',   idx_tr)
np.save('tensors/idx_val.npy',  idx_val)
np.save('tensors/uid_list.npy', np.array(uid_list))
```

#### Step 7: Normalise LSTM features (fit on train only)

```python
from sklearn.preprocessing import StandardScaler
import joblib

N_tr, T, F = X_lstm_tr.shape
sc_lstm = StandardScaler()
X_lstm_tr_sc  = sc_lstm.fit_transform(X_lstm_tr.reshape(-1, F)).reshape(N_tr, T, F)
X_lstm_val_sc = sc_lstm.transform(X_lstm_val.reshape(-1, F)).reshape(-1, T, F)
joblib.dump(sc_lstm, 'scalers/scaler_lstm.pkl')

np.save('tensors/X_lstm_train.npy', X_lstm_tr_sc)
np.save('tensors/X_lstm_val.npy',   X_lstm_val_sc)
np.save('tensors/y_train.npy',      y_tr)
np.save('tensors/y_val.npy',        y_val)
```

### 3.3 Align Session CSV to LSTM Session Order

**This step is mandatory before building ResNet or XGBoost tensors.**

```python
uid_list = np.load('tensors/uid_list.npy', allow_pickle=True).tolist()

df_sess_aligned = (
    df_sess
    .set_index('unique_link_mark')
    .loc[uid_list]
    .reset_index()
)

# Verify alignment
y_full = np.concatenate([np.load('tensors/y_train.npy'),
                          np.load('tensors/y_val.npy')])
assert len(df_sess_aligned) == len(y_full), "Row count mismatch!"
assert (df_sess_aligned['label'].values == y_full).all(), "Label mismatch!"
print("Session CSV aligned to LSTM order ✓")
```

> **Why this matters**: without this step, row *i* of the ResNet probability array could
> describe a different session than row *i* of the LSTM probability array. Each branch would
> appear to "train successfully" in isolation, but Layer 2 would silently combine
> probabilities from mismatched sessions, corrupting every downstream metric.

### 3.4 ResNet + XGBoost Pipeline — From Aligned Session CSV

No grouping needed here — `df_sess_aligned` is already one row per session, in the exact
order established by the LSTM tensor.

#### Step 1: Select 38 ResNet columns from 48 candidates

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

PAYLOAD_CANDIDATES = [c for c in df_sess_aligned.columns
    if any(x in c for x in [
        'Length_of_IP_packet', 'Length_of_TCP_payload',
        'Length_of_TCP_packet_header', 'Length_of_IP_packet_header',
        'TCP_windows_size_value', 'Length_of_TCP_segment',
        'forward_packet_length', 'backward_packet_length'])
    and '_enc' not in c and '_ratio' not in c]
print(len(PAYLOAD_CANDIDATES))  # → 48

selector = SelectKBest(mutual_info_classif, k=38)
selector.fit(df_sess_aligned[PAYLOAD_CANDIDATES], y_full)
RESNET_COLS = [PAYLOAD_CANDIDATES[i] for i in selector.get_support(indices=True)]
joblib.dump(selector, 'scalers/selector_resnet.pkl')
print(len(RESNET_COLS))  # → 38  ✓
```

#### Step 2: Define XGBoost columns — 65 ratio + 32 ENC time session cols

```python
RATIO_COLS = [c for c in df_sess_aligned.columns if c.endswith('_ratio')]
print(len(RATIO_COLS))  # → 65  ✓

ENC_TIME_SESS_COLS = [
    'flow_duration_enc', 'flow_duration_of_forward_traffic_enc',
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
    'total_ttl_forward_traffic_enc', 'std_ttl_forward_traffic_enc',
    'mean_ttl_forward_traffic_enc', 'max_ttl_forward_traffic_enc',
    'min_ttl_forward_traffic_enc', 'var_ttl_forward_traffic_enc',
    'median_ttl_forward_traffic_enc',
    'total_ttl_backward_traffic_enc', 'std_ttl_backward_traffic_enc',
    'mean_ttl_backward_traffic_enc', 'max_ttl_backward_traffic_enc',
    'min_ttl_backward_traffic_enc', 'var_ttl_backward_traffic_enc',
    'median_ttl_backward_traffic_enc', 'Total_Time_to_live_enc',
    'Total_length_of_forward_payload_enc',
    'Total_length_of_backward_payload_enc',
]
print(len(ENC_TIME_SESS_COLS))  # → 32

XGB_COLS = RATIO_COLS + ENC_TIME_SESS_COLS
print(len(XGB_COLS))  # → 97
```

#### Step 3: Extract, clean, and generate ResNet images

```python
from sklearn.preprocessing import MinMaxScaler

idx_tr  = np.load('tensors/idx_tr.npy')
idx_val = np.load('tensors/idx_val.npy')

X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)
X_payload = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)

mms = MinMaxScaler()
X_payload[idx_tr]  = mms.fit_transform(X_payload[idx_tr])
X_payload[idx_val] = mms.transform(X_payload[idx_val])
joblib.dump(mms, 'scalers/scaler_resnet_mms.pkl')

# Outer product per session → (N, 38, 38)
X_img = np.einsum('ni,nj->nij', X_payload, X_payload)[:, np.newaxis]  # (N,1,38,38)

np.save('tensors/X_resnet_train.npy', X_img[idx_tr])
np.save('tensors/X_resnet_val.npy',   X_img[idx_val])
print(X_img[idx_tr].shape)  # (N_train, 1, 38, 38)
```

#### Step 4: Extract and clean XGBoost features

```python
X_xgb = df_sess_aligned[XGB_COLS].values.astype(np.float32)
X_xgb = np.where(np.isinf(X_xgb), np.nan, X_xgb)
col_medians = np.nanmedian(X_xgb, axis=0)
nan_mask    = np.isnan(X_xgb)
X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

np.save('tensors/X_xgb_train.npy', X_xgb[idx_tr])
np.save('tensors/X_xgb_val.npy',   X_xgb[idx_val])
print(X_xgb[idx_tr].shape)  # (N_train, 97)
```

> **No StandardScaler needed for XGBoost** — tree-based models are invariant to monotonic
> feature scaling.

---

## 4. Feature Assignment Summary

### 4.1 Complete Column-to-Branch Map

| Branch | Source CSV(s) | Column groups | Count | Shape |
|--------|---------------|---------------|-------|-------|
| LSTM | packet + session (**merged**) | 7 per-packet + 78 session-level time | **85** | (15, 85) per session |
| ResNet | session (aligned) | payload length stats, SelectKBest k=38 of 48 | **38** | (1, 38, 38) |
| XGBoost | session (aligned) | 65 `_ratio` + 32 ENC time session cols | **97** | (N, 97) |
| Layer 2 | — | concat of 3 prob vecs | **6** | (N, 6) |

### 4.2 Session CSV Column Breakdown (280 total)

| Group | Count | Suffix | Role |
|-------|-------|--------|------|
| Traditional flow/time stats (used in LSTM merge) | 78 | none | LSTM Group B — broadcast per session |
| Payload length stats (ResNet candidates) | 48 → select 38 | none | ResNet input |
| ENC features | 78 | `_enc` | 32 reused for XGBoost; rest feed ratio calc |
| Ratio features | 65 | `_ratio` | XGBoost input |
| Excluded | 5 | — | label, unique_link_mark, Goodput, mean_of_fwd/bwd_IP_header |

### 4.3 Why the 85 LSTM Features Split Across Two CSVs

The paper extracted all 85 features per-packet internally. The public Mendeley release
pre-aggregated most of them into session-level statistics, while retaining only 7 as true
per-packet columns in the packet CSV. Both groups are still genuinely time-related — the
distinction is **temporal granularity**, not relevance:

- **7 packet CSV columns**: change per packet → form the true (15-step) temporal sequence
- **78 session CSV columns**: one value per session → broadcast identically across all 15
  timesteps, giving the LSTM fixed session-level time context at every step

This reconstructs the full 85-feature input the paper specifies, with no reduction in
feature count and no documented "dataset constraint" required.

---

## 5. Layer 1: Feature-Specific Detection Models

### 5.1 Branch 1: Multi-layer LSTM

**Architecture** (exact from Figure 1):

```
Input: (N, 15, 85)          ← 85 confirmed via packet+session merge — full paper spec met
├── LSTM(128, return_sequences=True)
├── Dropout(0.3)
├── LSTM(128, return_sequences=False)
├── Dropout(0.3)
├── Dense(256, activation='relu')
└── Dense(2, activation='softmax')    ← num_classes = 2 (binary)
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model_lstm = Sequential([
    LSTM(128, input_shape=(15, 85), return_sequences=True),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])
model_lstm.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Training Configuration**:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | sparse_categorical_crossentropy |
| Batch size | 32 |
| Early stopping patience | 5 (val loss) |

**Why Multi-layer LSTM (paper finding)**:
- Tested: LSTM, BiLSTM, GRU, BiGRU
- LSTM wins: Precision +0.53%, FPR −0.53% vs BiGRU
- BiLSTM has +0.18% recall but worse FPR — not chosen

**Expected Standalone Performance**:

| Metric | Value |
|--------|-------|
| Accuracy | 99.41% |
| F1 | 99.40% |
| Precision | 99.52% |
| Recall/TPR | 99.29% |
| FPR | 0.48% |

**Save probabilities**:

```python
lstm_probs_train = model_lstm.predict(X_lstm_train_sc)  # (N, 2)
lstm_probs_val   = model_lstm.predict(X_lstm_val_sc)
lstm_probs_test  = model_lstm.predict(X_lstm_test_sc)
np.save('probs/lstm_probs_train.npy', lstm_probs_train)
np.save('probs/lstm_probs_val.npy',   lstm_probs_val)
np.save('probs/lstm_probs_test.npy',  lstm_probs_test)
```

### 5.2 Branch 2: ResNet34 for Payload Features

**Image generation — two formats**:

```python
import torch

# Format A: 1×38 direct (paper winner)
# Each session row → (1, 38) image, single channel
def make_1x38_images(X):
    # X shape: (N, 38)
    return X[:, np.newaxis, np.newaxis, :]  # (N, 1, 1, 38)
    # PyTorch: (N, C=1, H=1, W=38)

# Format B: 38×38 via outer product (alternative)
def make_38x38_images(X):
    # X shape: (N, 38)
    imgs = np.einsum('ni,nj->nij', X, X)   # (N, 38, 38)
    return imgs[:, np.newaxis, :, :]        # (N, 1, 38, 38)
```

**Architecture** (exact from Figure 1, adapted for small input):

```python
import torchvision.models as models
import torch.nn as nn

resnet34 = models.resnet34(pretrained=False)
# Adapt for 1-channel small input
resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet34.maxpool = nn.Identity()   # remove — input too small for stride reduction
resnet34.fc = nn.Linear(512, 2)    # binary: num_classes = 2
```

**ResNet34 block structure** (unchanged from paper):

| Layer | Config |
|-------|--------|
| Conv 1 | 7×7, 64, stride 2 → replaced with 3×3, stride 1 for small input |
| Pool 1 | 3×3 max pool, stride 2 → removed for small input |
| Conv 2_x | [3×3, 64] × 3 |
| Conv 3_x | [3×3, 128] × 4 |
| Conv 4_x | [3×3, 256] × 6 |
| Conv 5_x | [3×3, 512] × 3 |
| Pool 2 | GlobalAveragePooling2D |
| Output | FC(2) + softmax |

**Expected Standalone Performance**:

| Metric | ResNet34 (38×38) | ResNet18 (38×38) |
|--------|------------------|------------------|
| Accuracy | 99.36% | 99.32% |
| F1 | 99.36% | 99.32% |
| Precision | 99.43% | 99.52% |
| Recall/TPR | 99.28% | 99.12% |
| FPR | 0.56% | 0.48% |

ResNet34 selected for highest overall F1, AUC, and TPR.

### 5.3 Branch 3: XGBoost for Ratio + ENC Time Features

**Input options**:

| Configuration | Columns | Shape | Notes |
|---------------|---------|-------|-------|
| Baseline | 65 `_ratio` cols | (N, 65) | Paper-exact |
| Extended | 65 `_ratio` + 32 ENC time session cols | (N, 97) | Recommended |

**The 32 additional ENC time session columns** (from session CSV, suffix `_enc`):

- flow_duration_enc, flow_duration_of_forward/backward_traffic_enc
- mean/median/max/min/std/var of IAT forward enc traffic (6 cols)
- mean/median/max/min/std/var of IAT backward enc traffic (6 cols)
- total/std/mean/max/min/var/median of TTL forward enc traffic (7 cols)
- total/std/mean/max/min/var/median of TTL backward enc traffic (7 cols)
- Total_Time_to_live_enc

> These are session-level aggregates, appropriate for XGBoost flat input. They are NOT suitable for the LSTM sequential branch.

**Parameters** (exact from Figure 1, updated for binary):

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    base_score=0.5,
    learning_rate=0.05,
    n_estimators=100,
    max_depth=10,
    objective='binary:logistic',   # binary classification confirmed
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(
    X_xgb_ext_train, y_xgb_train,
    eval_set=[(X_xgb_ext_val, y_xgb_val)],
    early_stopping_rounds=10,
    verbose=False
)
```

**Why XGBoost over Random Forest** (paper finding):

| Metric | XGBoost | Random Forest |
|--------|---------|--------------|
| Accuracy | 99.63% | 99.59% |
| F1 | 99.62% | 99.59% |
| Recall/TPR | 99.48% | 99.31% |
| Precision | 99.76% | 99.87% |
| FPR | 0.23% | **0.13%** |

XGBoost chosen for overall superiority. RF has lower FPR but lower TPR.

**Save probabilities**:

```python
xgb_probs_train = xgb_model.predict_proba(X_xgb_ext_train)  # (N, 2)
xgb_probs_val   = xgb_model.predict_proba(X_xgb_ext_val)
xgb_probs_test  = xgb_model.predict_proba(X_xgb_ext_test)
np.save('probs/xgb_probs_train.npy', xgb_probs_train)
np.save('probs/xgb_probs_val.npy',   xgb_probs_val)
np.save('probs/xgb_probs_test.npy',  xgb_probs_test)
```

---

## 6. Layer 2: Ensemble Detection

### 6.1 Input Construction

```python
# Load saved probabilities
lstm_p   = np.load('probs/lstm_probs_train.npy')    # (N, 2)
resnet_p = np.load('probs/resnet_probs_train.npy')  # (N, 2)
xgb_p    = np.load('probs/xgb_probs_train.npy')     # (N, 2)

# Concatenate → Layer 2 input
X_layer2_train = np.hstack([lstm_p, resnet_p, xgb_p])  # (N, 6)
# Repeat for val and test splits
```

> **Critical**: Always use probabilities from the matching split. Never use test predictions during Layer 2 training.

### 6.2 Option A: Random Forest Layer 2

**Use when**: TPR / F1 are priority.

```python
from sklearn.ensemble import RandomForestClassifier

rf_layer2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    bootstrap=True,
    max_features='sqrt',         # 'auto' deprecated in newer sklearn
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
rf_layer2.fit(X_layer2_train, y_train)
```

**Expected Performance**:

| Metric | Value |
|--------|-------|
| Accuracy | 99.65% |
| F1 | 99.65% |
| TPR/Recall | **99.68%** |
| Precision | 99.63% |
| FPR | 0.32% |

### 6.3 Option B: Average Ensemble

**Use when**: FPR minimization is priority. No training required.

```python
avg_probs  = (lstm_p_test + resnet_p_test + xgb_p_test) / 3
y_pred_avg = np.argmax(avg_probs, axis=1)
```

**Expected Performance**:

| Metric | Value |
|--------|-------|
| Accuracy | **99.73%** |
| F1 | **99.72%** |
| Precision | **99.89%** |
| TPR/Recall | 99.56% |
| FPR | **0.11%** |

### 6.4 Selection Guide

| Priority | Method | Key Metric |
|----------|--------|------------|
| Minimize missed detections | Random Forest | TPR: 99.68% |
| Minimize false alarms | Average Ensemble | FPR: 0.11% |
| Balanced (recommended) | Average Ensemble | F1: 99.72% |

---

## 7. Evaluation Methodology

### 7.1 Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Full report
print(classification_report(y_test, y_pred, target_names=['Benign','Malicious']))

# FPR for benign class (label=0) — the critical metric
cm = confusion_matrix(y_test, y_pred)
TN, FP = cm[0, 0], cm[0, 1]   # benign row: TN=correctly benign, FP=benign→malicious
FPR = FP / (FP + TN)
print(f"FPR (benign mislabelled as malicious): {FPR:.4%}")
# Target: ≤ 0.11%

# ROC-AUC
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"ROC-AUC: {auc:.4%}")
```

### 7.2 Ablation Study Order

1. Evaluate LSTM standalone (test RNN variants: LSTM, BiLSTM, GRU, BiGRU)
2. Evaluate ResNet34 standalone (test both 1×38 and 38×38 image formats)
3. Evaluate XGBoost standalone (test base 65 cols vs extended 97 cols)
4. Evaluate Layer 2 RF ensemble
5. Evaluate Layer 2 Average Ensemble
6. Compare all — prove two-layer beats any single branch

### 7.3 Feature Importance

```python
import shap

# XGBoost branch — which ratio features matter?
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_vals_xgb = explainer_xgb.shap_values(X_xgb_ext_test)
shap.summary_plot(shap_vals_xgb, X_xgb_ext_test,
                  feature_names=XGB_EXTENDED_COLS)

# Layer 2 RF — which branch's probabilities does it trust most?
explainer_rf = shap.TreeExplainer(rf_layer2)
shap_vals_rf = explainer_rf.shap_values(X_layer2_test)
# Feature names: ['lstm_p0','lstm_p1','resnet_p0','resnet_p1','xgb_p0','xgb_p1']
shap.summary_plot(shap_vals_rf,
                  feature_names=['LSTM-benign','LSTM-malicious',
                                 'ResNet-benign','ResNet-malicious',
                                 'XGB-benign','XGB-malicious'])
```

---

## 8. Implementation Checklist

### 8.1 Data Loading

- [ ] Load `Train Set/packet_based_trainset.csv` (4.4M rows, 25 cols)
- [ ] Load `Train Set/session_based_trainset.csv` (488K rows, 280 cols)
- [ ] Load `Test Set/packet_based_testset.csv`
- [ ] Load `Test Set/session_based_testset.csv`
- [ ] Confirm label column: `label` (0=benign, 1=malicious)
- [ ] Confirm session ID column: `unique_link_mark`

### 8.2 Data Quality

- [ ] Check NaN count in both CSVs (`df.isna().sum().sum()`)
- [ ] Check Inf count (`np.isinf(df.select_dtypes('number')).sum().sum()`)
- [ ] Replace Inf with NaN, impute NaN with column median
- [ ] Confirm class balance: ~243K benign / ~245K malicious

### 8.3 Feature Preparation

- [ ] Define PKT_TIME_COLS (7) and SESS_TIME_COLS (78) — combined ALL_TIME_COLS = 85
- [ ] Merge packet CSV + session CSV on `unique_link_mark` (left join)
- [ ] Clean NaN/Inf in merged 85 time columns
- [ ] Group merged data by `unique_link_mark`, sort by `Time_cost`
- [ ] Apply 15-packet cutoff
- [ ] Apply average padding (NOT zero-padding)
- [ ] Confirm LSTM input shape: (N_sessions, 15, 85)
- [ ] Save `uid_list` from the LSTM groupby — required for alignment
- [ ] **Align session CSV**: reindex by `uid_list`, verify labels match
- [ ] Run SelectKBest (k=38) on 48 payload candidates from **aligned** session CSV → RESNET_COLS
- [ ] Confirm ResNet input shape: (N, 38) before image conversion
- [ ] Extract 65 `_ratio` columns from **aligned** session CSV → RATIO_COLS
- [ ] Extract 32 ENC time session columns → ENC_TIME_SESS_COLS
- [ ] Confirm XGBoost extended input shape: (N, 97)
- [ ] Compute one `idx_tr` / `idx_val` split from the LSTM tensor; apply identically to ResNet and XGBoost
- [ ] Fit StandardScaler on train only for LSTM; fit MinMaxScaler on train only for ResNet
- [ ] No scaling needed for XGBoost (tree-based)

### 8.4 Layer 1 Training

- [ ] Train Multi-layer LSTM — input (N, 15, 85), output probs (N, 2)
- [ ] Train ResNet34 (38×38 format) — input (N, 1, 38, 38), output probs (N, 2)
- [ ] Train ResNet34 (1×38 format) — input (N, 1, 1, 38), compare with 38×38
- [ ] Train XGBoost base (N, 65) — save probs (N, 2)
- [ ] Train XGBoost extended (N, 97) — compare with base
- [ ] Save all probability arrays for train/val/test splits (6 files × 3 branches)
- [ ] Verify all three branches' probability arrays share identical row order (same session per row index)

### 8.5 Layer 2 Training

- [ ] Concatenate prob vectors: `np.hstack([lstm_p, resnet_p, xgb_p])` → (N, 6)
- [ ] Train Random Forest Layer 2 (n_estimators=200, max_depth=None)
- [ ] Implement Average Ensemble (no training)
- [ ] Compare RF vs Ensemble on val set

### 8.6 Evaluation

- [ ] Run ablation: each branch standalone
- [ ] Run full two-layer framework
- [ ] Compute FPR correctly: `FP/(FP+TN)` on benign class (label=0)
- [ ] Target: FPR ≤ 0.11%, F1 ≥ 99.72%, TPR ≥ 99.56%
- [ ] SHAP analysis on XGBoost + Layer 2 RF
- [ ] Generate confusion matrices for all models

---

## 9. Expected Final Results

### 9.1 Average Ensemble Layer 2 (Recommended)

| Metric | Paper Result | Implementation Target |
|--------|--------------|----------------------|
| Accuracy | 99.73% | ≥ 99.70% |
| F1 Score | 99.72% | ≥ 99.70% |
| Precision | 99.89% | ≥ 99.85% |
| Recall/TPR | 99.56% | ≥ 99.50% |
| FPR | 0.11% | ≤ 0.15% |
| ROC-AUC | 99.94% | ≥ 99.90% |

### 9.2 Random Forest Layer 2

| Metric | Paper Result | Implementation Target |
|--------|--------------|----------------------|
| Accuracy | 99.65% | ≥ 99.60% |
| F1 Score | 99.65% | ≥ 99.60% |
| Precision | 99.63% | ≥ 99.60% |
| Recall/TPR | 99.68% | ≥ 99.65% |
| FPR | 0.32% | ≤ 0.35% |
| ROC-AUC | 99.95% | ≥ 99.90% |

---

## 10. File Structure

```
encrypted_traffic_detection/
├── data/
│   ├── Train Set/
│   │   ├── packet_based_trainset.csv       # LSTM source (merged with session)
│   │   └── session_based_trainset.csv      # LSTM merge + ResNet + XGBoost source
│   └── Test Set/
│       ├── packet_based_testset.csv
│       └── session_based_testset.csv
├── tensors/                                 # Preprocessed numpy arrays
│   ├── X_lstm_{train,val,test}.npy          # (N, 15, 85)
│   ├── X_resnet_{train,val,test}.npy        # (N, 1, 38, 38)
│   ├── X_xgb_{train,val,test}.npy           # (N, 97)
│   ├── y_{train,val,test}.npy
│   ├── idx_tr.npy / idx_val.npy             # Unified split indices
│   └── uid_list.npy                         # Session order — alignment key
├── probs/                                   # Saved probability outputs
│   ├── lstm_probs_{train,val,test}.npy
│   ├── resnet_probs_{train,val,test}.npy
│   └── xgb_probs_{train,val,test}.npy
├── scalers/
│   ├── scaler_lstm.pkl
│   ├── scaler_resnet_mms.pkl
│   └── selector_resnet.pkl
├── src/
│   ├── preprocessing/
│   │   ├── lstm_pipeline.py                # merge → groupby → cutoff → padding → uid_list
│   │   ├── align_session.py                # reindex session CSV by uid_list
│   │   ├── resnet_pipeline.py              # SelectKBest → MinMaxScale → outer product
│   │   ├── xgb_pipeline.py                 # 65 ratio + 32 ENC time cols
│   │   └── run_all.py                      # master script, correct run order
│   ├── features/
│   │   ├── select_resnet_cols.py           # SelectKBest k=38 on 48 candidates
│   │   ├── lstm_features.py                # PKT_TIME_COLS (7) + SESS_TIME_COLS (78) = 85
│   │   └── session_features.py             # RESNET_COLS, RATIO_COLS, ENC_TIME_SESS_COLS
│   ├── models/
│   │   ├── lstm_model.py                   # LSTM(128)×2 → Dense(256) → Dense(2)
│   │   ├── resnet_model.py                 # ResNet34 1-channel adapted
│   │   ├── xgboost_model.py                # XGBClassifier binary:logistic
│   │   └── layer2_ensemble.py              # RF + Average Ensemble
│   ├── training/
│   │   ├── train_lstm.py
│   │   ├── train_resnet.py
│   │   ├── train_xgboost.py
│   │   └── train_layer2.py
│   └── evaluation/
│       ├── metrics.py                      # FPR, F1, TPR, ROC-AUC
│       └── shap_analysis.py               # SHAP for XGBoost and Layer 2 RF
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── main.py                                 # End-to-end pipeline runner
```

---

## 11. Hyperparameters Summary

### LSTM

| Parameter | Value |
|-----------|-------|
| Input shape | (15, 85) |
| LSTM hidden units | 128 |
| LSTM layers | 2 |
| return_sequences (L1) | True |
| return_sequences (L2) | False |
| Dropout | 0.3 |
| Dense units | 256 |
| Output units | 2 (binary) |
| Learning rate | 0.001 |
| Batch size | 32 |
| Early stopping patience | 5 |

### ResNet34

| Parameter | Value |
|-----------|-------|
| Input shape | (1, 38, 38) preferred |
| Input channels | 1 (grayscale) |
| conv1 | 3×3, 64, stride 1 (adapted) |
| maxpool | Removed (input too small) |
| Conv2_x | [3×3, 64] × 3 |
| Conv3_x | [3×3, 128] × 4 |
| Conv4_x | [3×3, 256] × 6 |
| Conv5_x | [3×3, 512] × 3 |
| Pool 2 | GlobalAveragePooling2D |
| Output | FC(2) + softmax |

### XGBoost

| Parameter | Value |
|-----------|-------|
| Input shape | (N, 97) extended |
| base_score | 0.5 |
| learning_rate | 0.05 |
| n_estimators | 100 |
| max_depth | 10 |
| objective | binary:logistic |
| eval_metric | logloss |
| early_stopping_rounds | 10 |

### Layer 2 Random Forest

| Parameter | Value |
|-----------|-------|
| Input shape | (N, 6) |
| n_estimators | 200 |
| max_depth | None |
| max_features | 'sqrt' |
| bootstrap | True |
| min_samples_split | 2 |

---

## 12. Key Implementation Notes

### Critical Notes

1. **Binary classification confirmed**: `num_classes = 2` everywhere. Layer 2 input is `(N, 6)` — 3 branches × 2 probability values. Not multiclass.

2. **LSTM merges BOTH source CSVs**: 7 true per-packet columns come from the packet CSV; 78 session-level time columns come from the session CSV and are broadcast identically across all 15 timesteps of a session. ResNet and XGBoost use the session CSV only (after alignment — see Note 5). Merging both CSVs achieves the full 85-feature input the paper specifies — no reduction needed.

3. **Pre-defined test split**: Use `Test Set/` folder for final evaluation. Do not create your own train/test split from the full data.

4. **85 LSTM features achieved via merge, not 19**: Earlier drafts of this plan used only the 19 packet-CSV columns and treated 19-vs-85 as an unavoidable dataset constraint. That was incorrect — merging the session CSV's 78 time-related columns onto the packet CSV (via `unique_link_mark`) reconstructs the full 85-feature input with no compromise.

5. **Build LSTM first, then align the session CSV**: `build_lstm_tensor()` processes sessions via `groupby('unique_link_mark')`, producing a session order captured in `uid_list`. The session CSV must be reindexed to this exact order (`df_sess.set_index('unique_link_mark').loc[uid_list]`) before building ResNet and XGBoost tensors. Skipping this step risks each branch "training successfully" while Layer 2 silently combines probabilities from mismatched sessions.

6. **One unified train/val split across all branches**: Compute `idx_tr` / `idx_val` once from the LSTM tensor and apply identically to the aligned ResNet and XGBoost tensors. Do not call `train_test_split` independently per branch — even with the same `random_state`, independent calls are not guaranteed to select the same sessions in the same order once the underlying data differs in row count or ordering.

7. **ENC time features go to XGBoost, not LSTM, when sourced from the session CSV**: The 32 session-level ENC time aggregates (flow_duration_enc, IAT stats, TTL stats) used in the XGBoost extended feature set are session-level — one value per session. They are distinct from the 78 session-level columns merged into the LSTM input; some column names overlap conceptually but the LSTM and XGBoost branches consume them differently (LSTM broadcasts across timesteps as context; XGBoost takes them as flat features).

8. **ResNet has 48 payload column candidates, not exactly 38**: Use `SelectKBest(mutual_info_classif, k=38)` to select the top 38 from the 48 candidates, computed on the aligned session CSV.

9. **Average padding, not zero-padding**: Time-based features (IAT, timestamps) must use per-session mean as padding. Zero-padding creates artificial data points that distort temporal statistics.

10. **No test leakage in Layer 2**: Probability vectors for Layer 2 training must come from Layer 1 predictions on the training split only. Never use test-set predictions in Layer 2 training.

11. **StandardScaler fit on train only**: Fit scalers on training data, then transform val and test. Never fit on all data before splitting.

12. **FPR calculation**: `FPR = FP / (FP + TN)` where FP and TN are from the benign class row of the confusion matrix (label=0 row). This is not the same as macro-averaged FPR.

### Common Pitfalls

- Using zero-padding instead of average padding
- Fitting StandardScaler on full dataset before split
- Using test predictions to train Layer 2 RF
- Confusing macro FPR with class-specific FPR for benign
- Building ResNet/XGBoost tensors with independent `train_test_split` calls instead of reusing the LSTM-derived `idx_tr`/`idx_val` — causes silent session misalignment in Layer 2
- Treating the packet CSV's 19 raw columns as the full LSTM input instead of merging in the session CSV's 78 time columns
- Forgetting to adapt ResNet conv1 and maxpool for small image input
- Using `multi:softprob` objective — must use `binary:logistic` (num_classes=2)
- Using `max_features='auto'` in newer sklearn — use `'sqrt'` instead

---

## 13. References

- Wang, Z., Thing, V. (2023). Feature Mining for Encrypted Malicious Traffic Detection with Deep Learning and Other Machine Learning Algorithms. arXiv:2304.03691
- Mendeley Dataset: doi:10.17632/xw7r4tt54g.1
