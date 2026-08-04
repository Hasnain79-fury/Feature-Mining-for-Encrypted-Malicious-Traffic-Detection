import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from feature_columns import ALL_TIME_COLS, SESS_TIME_COLS

def process_lstm(df_pkt, df_sess, is_train=True, sc_lstm=None):
    print(f"LSTM Pipeline (is_train={is_train})")
    sess_slim = df_sess[['unique_link_mark'] + SESS_TIME_COLS]
    df_merged = df_pkt.merge(sess_slim, on='unique_link_mark', how='left')
    
    # Clean NaN and Inf values memory-efficiently
    for col in ALL_TIME_COLS:
        if col in df_merged.columns:
            # Downcast to float32 to save memory
            df_merged[col] = df_merged[col].astype(np.float32)
            # In-place replacement
            df_merged[col] = df_merged[col].replace([np.inf, -np.inf], np.nan)
            
            med = df_merged[col].median()
            if pd.isna(med):
                med = 0.0
            df_merged[col] = df_merged[col].fillna(med)

    max_pkts = 15
    sessions, labels, uid_list = [], [], []
    
    print(f"Grouping packets for {len(df_sess)} sessions...")
    for uid, grp in df_merged.groupby('unique_link_mark', sort=False):
        grp = grp.sort_values('Time_cost')
        mat = grp[ALL_TIME_COLS].values.astype(np.float32)
        
        if len(mat) > max_pkts:
            mat = mat[:max_pkts]
        elif len(mat) < max_pkts:
            pad_val = mat.mean(axis=0)
            pad_rows = np.tile(pad_val, (max_pkts - len(mat), 1))
            mat = np.vstack([mat, pad_rows])
            
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
        
        N_tr, T, F = X[idx_tr].shape
        
        # Calculate mean and std over Train set (axes 0 and 1) in float32
        # We process manually to prevent scikit-learn from copying and upcasting to float64 (saves ~8GB RAM)
        X_tr_sc = X[idx_tr].copy()
        mean_val = X_tr_sc.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
        std_val = X_tr_sc.std(axis=(0, 1), keepdims=True, dtype=np.float32)
        std_val[std_val == 0] = 1.0
        
        # In-place scaling
        X_tr_sc -= mean_val
        X_tr_sc /= std_val
        
        # Apply scaling to Validation set
        X_val_sc = X[idx_val].copy()
        X_val_sc -= mean_val
        X_val_sc /= std_val
        
        # Create a lightweight dummy scaler class that mimics StandardScaler for future compatibility
        class CustomScaler:
            def __init__(self, m, s):
                self.mean_val = m
                self.std_val = s
            def transform(self, X_in):
                X_out = X_in.copy()
                # If 2D (from testing script), reshape to 3D temporarily if needed, but since
                # mean and std are shape (1,1,F), broadcasting works if X_out is (N, T, F)
                # Let's flatten mean/std for safety
                flat_mean = self.mean_val.reshape(-1)
                flat_std = self.std_val.reshape(-1)
                
                if X_out.ndim == 3:
                    X_out -= flat_mean
                    X_out /= flat_std
                else:
                    X_out -= flat_mean
                    X_out /= flat_std
                return X_out

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
        # CustomScaler handles 3D arrays automatically without reshaping
        X_sc = sc_lstm.transform(X)
        
        np.save('tensors/X_lstm_test.npy', X_sc)
        np.save('tensors/y_test.npy', y)
        np.save('tensors/uid_list_test.npy', uid_arr)
        
        return uid_arr
