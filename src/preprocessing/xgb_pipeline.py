import numpy as np
import pandas as pd

from feature_columns import ENC_TIME_SESS_COLS

def process_xgb(df_sess, uid_list, is_train=True, idx_tr=None, idx_val=None, col_medians=None):
    print(f"XGBoost Pipeline (is_train={is_train})")
    
    # Align session CSV
    df_sess_aligned = df_sess.set_index('unique_link_mark').loc[uid_list].reset_index()
    
    RATIO_COLS = [c for c in df_sess_aligned.columns if c.endswith('_ratio')]
    XGB_COLS = RATIO_COLS + ENC_TIME_SESS_COLS
    
    X_xgb = df_sess_aligned[XGB_COLS].values.astype(np.float32)
    X_xgb = np.where(np.isinf(X_xgb), np.nan, X_xgb)
    
    if is_train:
        print("Imputing NaN values...")
        col_medians = np.nanmedian(X_xgb, axis=0)
        nan_mask = np.isnan(X_xgb)
        X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        
        np.save('tensors/X_xgb_train.npy', X_xgb[idx_tr])
        np.save('tensors/X_xgb_val.npy', X_xgb[idx_val])
        np.save('tensors/xgb_col_medians.npy', col_medians)
        
        return col_medians
    else:
        print("Applying medians to test set...")
        nan_mask = np.isnan(X_xgb)
        X_xgb[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        np.save('tensors/X_xgb_test.npy', X_xgb)
