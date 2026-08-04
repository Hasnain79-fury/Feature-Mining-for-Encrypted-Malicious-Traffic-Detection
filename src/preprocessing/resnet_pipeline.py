import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

from feature_columns import PAYLOAD_CANDIDATES

def process_resnet(df_sess, uid_list, is_train=True, idx_tr=None, idx_val=None, y_train_full=None, selector=None, mms=None):
    print(f"ResNet Pipeline (is_train={is_train})")
    
    # Align session CSV
    df_sess_aligned = df_sess.set_index('unique_link_mark').loc[uid_list].reset_index()
    
    if is_train:
        print("Selecting features and fitting scaler...")
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
        print("Applying scaler to test set...")
        RESNET_COLS = [PAYLOAD_CANDIDATES[i] for i in selector.get_support(indices=True)]
        
        X_payload = df_sess_aligned[RESNET_COLS].values.astype(np.float32)
        X_payload = np.nan_to_num(X_payload, nan=0.0, posinf=0.0, neginf=0.0)
        X_payload = mms.transform(X_payload)
        
        X_img = np.einsum('ni,nj->nij', X_payload, X_payload)[:, np.newaxis, :, :]
        np.save('tensors/X_resnet_test.npy', X_img)
