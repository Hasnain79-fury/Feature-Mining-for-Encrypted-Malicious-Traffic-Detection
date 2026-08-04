import json

cells = []

def add_md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [text]})

def add_code(text):
    source = [line + '\n' for line in text.split('\n')]
    if source:
        source[-1] = source[-1].rstrip('\n')
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source})

add_md('# Encrypted Malicious Traffic Detection - Preprocessing Pipeline\n\nRun this notebook on Google Colab. Make sure to upload your `Train Set` and `Test Set` folders to Google Drive first.')

add_code('''import os
from google.colab import drive
drive.mount('/content/drive')

# Change this to where you uploaded the data in your Drive!
BASE_DIR = '/content/drive/MyDrive/Malicious_Traffic_Data'
os.chdir(BASE_DIR)

os.makedirs('tensors', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('probs', exist_ok=True)
print(f"Working directory set to: {os.getcwd()}")''')

add_code('''import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

PKT_TIME_COLS = [
    'Time_cost', 'Time_difference_between_packets_per_session',
    'Interval_of_arrival_time_of_forward_traffic', 'Interval_of_arrival_time_of_backward_traffic',
    'inter_arrival_time_of_forward_traffic_enc', 'inter_arrival_time_of_backward_traffic_enc',
    'ratio_to_previous_packet_enc',
]

SESS_TIME_COLS = [
    'flow duration', 'flow_duration_of_forward_traffic', 'flow_duration_of_backward_traffic',
    'mean_Time_difference_between_packets_per_session', 'median_Time_difference_between_packets_per_session',
    'max_Time_difference_between_packets_per_session', 'min_Time_difference_between_packets_per_session',
    'std_Time_difference_between_packets_per_session', 'var_Time_difference_between_packets_per_session',
    'mean_Interval_of_arrival_time_of_forward_traffic', 'median_Interval_of_arrival_time_of_forward_traffic',
    'max_Interval_of_arrival_time_of_forward_traffic', 'min_Interval_of_arrival_time_of_forward_traffic',
    'std_Interval_of_arrival_time_of_forward_traffic', 'var_Interval_of_arrival_time_of_forward_traffic',
    'mean_Interval_of_arrival_time_of_backward_traffic', 'median_Interval_of_arrival_time_of_backward_traffic',
    'max_Interval_of_arrival_time_of_backward_traffic', 'min_Interval_of_arrival_time_of_backward_traffic',
    'std_Interval_of_arrival_time_of_backward_traffic', 'var_Interval_of_arrival_time_of_backward_traffic',
    'total_ttl_forward_traffic', 'std_ttl_forward_traffic', 'mean_ttl_forward_traffic',
    'max_ttl_forward_traffic', 'min_ttl_forward_traffic', 'var_ttl_forward_traffic', 'median_ttl_forward_traffic',
    'total_ttl_backward_traffic', 'std_ttl_backward_traffic', 'mean_ttl_backward_traffic',
    'max_ttl_backward_traffic', 'min_ttl_backward_traffic', 'var_ttl_backward_traffic', 'median_ttl_backward_traffic',
    'flow_duration_enc', 'flow_duration_of_forward_traffic_enc', 'flow_duration_of_backward_traffic_enc',
    'mean_Interval_of_arrival_time_of_forward_traffic_enc', 'median_Interval_of_arrival_time_of_forward_traffic_enc',
    'max_Interval_of_arrival_time_of_forward_traffic_enc', 'min_Interval_of_arrival_time_of_forward_traffic_enc',
    'std_Interval_of_arrival_time_of_forward_traffic_enc', 'var_Interval_of_arrival_time_of_forward_traffic_enc',
    'mean_Interval_of_arrival_time_of_backward_traffic_enc', 'median_Interval_of_arrival_time_of_backward_traffic_enc',
    'max_Interval_of_arrival_time_of_backward_traffic_enc', 'min_Interval_of_arrival_time_of_backward_traffic_enc',
    'std_Interval_of_arrival_time_of_backward_traffic_enc', 'var_Interval_of_arrival_time_of_backward_traffic_enc',
    'total_ttl_forward_traffic_enc', 'std_ttl_forward_traffic_enc', 'mean_ttl_forward_traffic_enc',
    'max_ttl_forward_traffic_enc', 'min_ttl_forward_traffic_enc', 'var_ttl_forward_traffic_enc', 'median_ttl_forward_traffic_enc',
    'total_ttl_backward_traffic_enc', 'std_ttl_backward_traffic_enc', 'mean_ttl_backward_traffic_enc',
    'max_ttl_backward_traffic_enc', 'min_ttl_backward_traffic_enc', 'var_ttl_backward_traffic_enc', 'median_ttl_backward_traffic_enc',
    'flow_duration_ratio', 'flow_duration_of_forward_traffic_ratio', 'flow_duration_of_backward_traffic_ratio',
    'mean_Interval_of_arrival_time_of_forward_traffic_ratio', 'median_Interval_of_arrival_time_of_forward_traffic_ratio',
    'max_Interval_of_arrival_time_of_forward_traffic_ratio', 'min_Interval_of_arrival_time_of_forward_traffic_ratio',
    'std_Interval_of_arrival_time_of_forward_traffic_ratio', 'var_Interval_of_arrival_time_of_forward_traffic_ratio',
    'mean_Interval_of_arrival_time_of_backward_traffic_ratio', 'median_Interval_of_arrival_time_of_backward_traffic_ratio',
    'max_Interval_of_arrival_time_of_backward_traffic_ratio', 'min_Interval_of_arrival_time_of_backward_traffic_ratio',
    'std_Interval_of_arrival_time_of_backward_traffic_ratio', 'var_Interval_of_arrival_time_of_backward_traffic_ratio',
    'total_ttl_forward_traffic_ratio', 'std_ttl_forward_traffic_ratio', 'mean_ttl_forward_traffic_ratio',
    'max_ttl_forward_traffic_ratio', 'min_ttl_forward_traffic_ratio', 'var_ttl_forward_traffic_ratio', 'median_ttl_forward_traffic_ratio',
]
ALL_TIME_COLS = PKT_TIME_COLS + SESS_TIME_COLS

PAYLOAD_CANDIDATES = [
    'mean_Length_of_IP_packets', 'median_Length_of_IP_packets', 'max_Length_of_IP_packets', 'min_Length_of_IP_packets',
    'std_Length_of_IP_packets', 'var_Length_of_IP_packets', 'mean_Length_of_TCP_payload', 'median_Length_of_TCP_payload',
    'max_Length_of_TCP_payload', 'min_Length_of_TCP_payload', 'std_Length_of_TCP_payload', 'var_Length_of_TCP_payload',
    'mean_Length_of_TCP_packet_header', 'median_Length_of_TCP_packet_header', 'max_Length_of_TCP_packet_header', 
    'min_Length_of_TCP_packet_header', 'std_Length_of_TCP_packet_header', 'var_Length_of_TCP_packet_header',
    'mean_Length_of_IP_packet_header', 'median_Length_of_IP_packet_header', 'max_Length_of_IP_packet_header', 
    'min_Length_of_IP_packet_header', 'std_Length_of_IP_packet_header', 'var_Length_of_IP_packet_header',
    'mean_TCP_windows_size_value', 'median_TCP_windows_size_value', 'max_TCP_windows_size_value', 'min_TCP_windows_size_value',
    'std_TCP_windows_size_value', 'var_TCP_windows_size_value', 'mean_Length_of_TCP_segment(packet)', 
    'median_Length_of_TCP_segment(packet)', 'max_Length_of_TCP_segment(packet)',  'min_Length_of_TCP_segment(packet)',
    'std_Length_of_TCP_segment(packet)', 'var_Length_of_TCP_segment(packet)', 'std_forward_packet_length', 
    'mean_forward_packet_length', 'max_forward_packet_length', 'min_forward_packet_length', 'var_forward_packet_length', 
    'median_forward_packet_length', 'std_backward_packet_length', 'mean_backward_packet_length', 'max_backward_packet_length', 
    'min_backward_packet_length', 'var_backward_packet_length', 'median_backward_packet_length',
]

ENC_TIME_SESS_COLS = [
    'flow_duration_enc', 'flow_duration_of_forward_traffic_enc', 'flow_duration_of_backward_traffic_enc',
    'mean_Interval_of_arrival_time_of_forward_traffic_enc', 'median_Interval_of_arrival_time_of_forward_traffic_enc',
    'max_Interval_of_arrival_time_of_forward_traffic_enc', 'min_Interval_of_arrival_time_of_forward_traffic_enc',
    'std_Interval_of_arrival_time_of_forward_traffic_enc', 'var_Interval_of_arrival_time_of_forward_traffic_enc',
    'mean_Interval_of_arrival_time_of_backward_traffic_enc', 'median_Interval_of_arrival_time_of_backward_traffic_enc',
    'max_Interval_of_arrival_time_of_backward_traffic_enc', 'min_Interval_of_arrival_time_of_backward_traffic_enc',
    'std_Interval_of_arrival_time_of_backward_traffic_enc', 'var_Interval_of_arrival_time_of_backward_traffic_enc',
    'total_ttl_forward_traffic_enc', 'std_ttl_forward_traffic_enc', 'mean_ttl_forward_traffic_enc',
    'max_ttl_forward_traffic_enc', 'min_ttl_forward_traffic_enc', 'var_ttl_forward_traffic_enc', 'median_ttl_forward_traffic_enc',
    'total_ttl_backward_traffic_enc', 'std_ttl_backward_traffic_enc', 'mean_ttl_backward_traffic_enc',
    'max_ttl_backward_traffic_enc', 'min_ttl_backward_traffic_enc', 'var_ttl_backward_traffic_enc', 'median_ttl_backward_traffic_enc',
    'Total_Time_to_live_enc', 'Total_length_of_forward_payload_enc', 'Total_length_of_backward_payload_enc',
]

class CustomScaler:
    def __init__(self, m, s):
        self.mean_val = m
        self.std_val = s
    def transform(self, X_in):
        X_out = X_in.copy()
        flat_mean = self.mean_val.reshape(-1)
        flat_std = self.std_val.reshape(-1)
        X_out -= flat_mean
        X_out /= flat_std
        return X_out''')

add_code('''def process_lstm(df_pkt, df_sess, is_train=True, sc_lstm=None):
    print(f"LSTM Pipeline (is_train={is_train})")
    sess_slim = df_sess[['unique_link_mark'] + SESS_TIME_COLS]
    df_merged = df_pkt.merge(sess_slim, on='unique_link_mark', how='left')
    
    print("Downcasting and cleaning columns...")
    for col in ALL_TIME_COLS:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(np.float32)
            df_merged[col] = df_merged[col].replace([np.inf, -np.inf], np.nan)
            med = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(med if not pd.isna(med) else 0.0)

    max_pkts = 15
    sessions, labels, uid_list = [], [], []
    
    print(f"Grouping packets...")
    for uid, grp in df_merged.groupby('unique_link_mark', sort=False):
        grp = grp.sort_values('Time_cost')
        mat = grp[ALL_TIME_COLS].values.astype(np.float32)
        
        if len(mat) > max_pkts:
            mat = mat[:max_pkts]
        elif len(mat) < max_pkts:
            pad_val = mat.mean(axis=0)
            mat = np.vstack([mat, np.tile(pad_val, (max_pkts - len(mat), 1))])
            
        sessions.append(mat)
        labels.append(int(grp['label'].iloc[0]))
        uid_list.append(uid)
        
    X = np.array(sessions, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    uid_arr = np.array(uid_list)

    if is_train:
        idx = np.arange(len(X))
        idx_tr, idx_val = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
        y_tr, y_val = y[idx_tr], y[idx_val]
        
        X_tr_sc = X[idx_tr].copy()
        mean_val = X_tr_sc.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
        std_val = X_tr_sc.std(axis=(0, 1), keepdims=True, dtype=np.float32)
        std_val[std_val == 0] = 1.0
        
        X_tr_sc -= mean_val
        X_tr_sc /= std_val
        
        X_val_sc = X[idx_val].copy()
        X_val_sc -= mean_val
        X_val_sc /= std_val
        
        sc_lstm = CustomScaler(mean_val, std_val)
        joblib.dump(sc_lstm, 'scalers/scaler_lstm.pkl')
        
        np.save('tensors/X_lstm_train.npy', X_tr_sc)
        np.save('tensors/X_lstm_val.npy', X_val_sc)
        np.save('tensors/y_train.npy', y_tr)
        np.save('tensors/y_val.npy', y_val)
        np.save('tensors/idx_tr.npy', idx_tr)
        np.save('tensors/idx_val.npy', idx_val)
        np.save('tensors/uid_list.npy', uid_arr)
        
        return uid_arr, idx_tr, idx_val, y, sc_lstm
    else:
        X_sc = sc_lstm.transform(X)
        np.save('tensors/X_lstm_test.npy', X_sc)
        np.save('tensors/y_test.npy', y)
        np.save('tensors/uid_list_test.npy', uid_arr)
        return uid_arr

def process_resnet(df_sess, uid_list, is_train=True, idx_tr=None, idx_val=None, y_train_full=None, selector=None, mms=None):
    print(f"ResNet Pipeline (is_train={is_train})")
    df_sess_aligned = df_sess.set_index('unique_link_mark').loc[uid_list].reset_index()
    
    if is_train:
        selector = SelectKBest(mutual_info_classif, k=38)
        selector.fit(df_sess_aligned[PAYLOAD_CANDIDATES].values, y_train_full)
        RESNET_COLS = [PAYLOAD_CANDIDATES[i] for i in selector.get_support(indices=True)]
        joblib.dump(selector, 'scalers/selector_resnet.pkl')
        
        X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)
        X_payload = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)
        
        mms = MinMaxScaler()
        X_payload[idx_tr] = mms.fit_transform(X_payload[idx_tr])
        X_payload[idx_val] = mms.transform(X_payload[idx_val])
        joblib.dump(mms, 'scalers/scaler_resnet_mms.pkl')
        
        X_img = np.einsum('ni,nj->nij', X_payload, X_payload)[:, np.newaxis, :, :]
        np.save('tensors/X_resnet_train.npy', X_img[idx_tr])
        np.save('tensors/X_resnet_val.npy', X_img[idx_val])
        return selector, mms
    else:
        RESNET_COLS = [PAYLOAD_CANDIDATES[i] for i in selector.get_support(indices=True)]
        X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)
        X_payload = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)
        X_payload = mms.transform(X_payload)
        X_img = np.einsum('ni,nj->nij', X_payload, X_payload)[:, np.newaxis, :, :]
        np.save('tensors/X_resnet_test.npy', X_img)

def process_xgb(df_sess, uid_list, is_train=True, idx_tr=None, idx_val=None, col_medians=None):
    print(f"XGBoost Pipeline (is_train={is_train})")
    df_sess_aligned = df_sess.set_index('unique_link_mark').loc[uid_list].reset_index()
    RATIO_COLS = [c for c in df_sess_aligned.columns if c.endswith('_ratio')]
    XGB_COLS = RATIO_COLS + ENC_TIME_SESS_COLS
    
    X_xgb = df_sess_aligned[XGB_COLS].values.astype(np.float32)
    X_xgb = np.where(np.isinf(X_xgb), np.nan, X_xgb)
    
    if is_train:
        col_medians = np.nanmedian(X_xgb, axis=0)
        nan_mask = np.isnan(X_xgb)
        X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        np.save('tensors/X_xgb_train.npy', X_xgb[idx_tr])
        np.save('tensors/X_xgb_val.npy', X_xgb[idx_val])
        np.save('tensors/xgb_col_medians.npy', col_medians)
        return col_medians
    else:
        nan_mask = np.isnan(X_xgb)
        X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        np.save('tensors/X_xgb_test.npy', X_xgb)''')

add_code('''print("=" * 60)
print("PHASE 1: TRAIN & VAL PREPROCESSING")
print("=" * 60)

print("Loading Train CSVs...")
df_pkt = pd.read_csv('Train Set/packet_based_trainset.csv')
df_sess = pd.read_csv('Train Set/session_based_trainset.csv')

# Ensure we only process UIDs present in both dataframes
valid_uids = set(df_pkt['unique_link_mark']).intersection(set(df_sess['unique_link_mark']))
df_pkt = df_pkt[df_pkt['unique_link_mark'].isin(valid_uids)]
df_sess = df_sess[df_sess['unique_link_mark'].isin(valid_uids)]

# LSTM 
uid_list, idx_tr, idx_val, y_full, sc_lstm = process_lstm(df_pkt, df_sess, is_train=True)
del df_pkt
gc.collect()

# ResNet
selector, mms = process_resnet(df_sess, uid_list, is_train=True, idx_tr=idx_tr, idx_val=idx_val, y_train_full=y_full)

# XGBoost
col_medians = process_xgb(df_sess, uid_list, is_train=True, idx_tr=idx_tr, idx_val=idx_val)
del df_sess
gc.collect()''')

add_code('''print("=" * 60)
print("PHASE 2: TEST SET PREPROCESSING")
print("=" * 60)

print("Loading Test CSVs...")
df_pkt_test = pd.read_csv('Test Set/packet_based_testset.csv')
df_sess_test = pd.read_csv('Test Set/session_based_testset.csv')

# Ensure we only process UIDs present in both dataframes
valid_uids_test = set(df_pkt_test['unique_link_mark']).intersection(set(df_sess_test['unique_link_mark']))
df_pkt_test = df_pkt_test[df_pkt_test['unique_link_mark'].isin(valid_uids_test)]
df_sess_test = df_sess_test[df_sess_test['unique_link_mark'].isin(valid_uids_test)]

# LSTM 
uid_list_test = process_lstm(df_pkt_test, df_sess_test, is_train=False, sc_lstm=sc_lstm)
del df_pkt_test
gc.collect()

# ResNet 
process_resnet(df_sess_test, uid_list_test, is_train=False, selector=selector, mms=mms)

# XGBoost 
process_xgb(df_sess_test, uid_list_test, is_train=False, col_medians=col_medians)
del df_sess_test
gc.collect()''')

add_code('''X_lstm_tr = np.load('tensors/X_lstm_train.npy')
X_resnet_tr = np.load('tensors/X_resnet_train.npy')
X_xgb_tr = np.load('tensors/X_xgb_train.npy')
y_tr = np.load('tensors/y_train.npy')

print(f"Shape checks (Train):")
print(f"  LSTM:    {X_lstm_tr.shape}   <- expect (N, 15, 85)")
print(f"  ResNet:  {X_resnet_tr.shape} <- expect (N, 1, 38, 38)")
print(f"  XGBoost: {X_xgb_tr.shape}   <- expect (N, 97)")
print(f"  Labels:  {y_tr.shape}        <- expect (N,)")

assert X_lstm_tr.shape[0] == X_resnet_tr.shape[0] == X_xgb_tr.shape[0] == y_tr.shape[0], "Train Row count mismatch!"
print("\\nAll checks passed ✓")''')

notebook = {
    'cells': cells,
    'metadata': {},
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('preprocessing_colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
