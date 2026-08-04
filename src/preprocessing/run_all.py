import os
import gc
import pandas as pd
from lstm_pipeline import process_lstm
from resnet_pipeline import process_resnet
from xgb_pipeline import process_xgb

def run_train():
    print("=" * 60)
    print("PHASE 1: TRAIN & VAL PREPROCESSING")
    print("=" * 60)
    
    print("Loading Train CSVs...")
    df_pkt = pd.read_csv('Train Set/packet_based_trainset.csv')
    df_sess = pd.read_csv('Train Set/session_based_trainset.csv')
    
    # Ensure we only process UIDs present in both dataframes
    valid_uids = set(df_pkt['unique_link_mark']).intersection(set(df_sess['unique_link_mark']))
    df_pkt = df_pkt[df_pkt['unique_link_mark'].isin(valid_uids)]
    df_sess = df_sess[df_sess['unique_link_mark'].isin(valid_uids)]
    
    # LSTM Train/Val
    uid_list, idx_tr, idx_val, y_full, sc_lstm = process_lstm(df_pkt, df_sess, is_train=True)
    
    del df_pkt
    gc.collect()
    
    # ResNet Train/Val
    selector, mms = process_resnet(df_sess, uid_list, is_train=True, idx_tr=idx_tr, idx_val=idx_val, y_train_full=y_full)
    
    # XGBoost Train/Val
    col_medians = process_xgb(df_sess, uid_list, is_train=True, idx_tr=idx_tr, idx_val=idx_val)
    
    del df_sess
    gc.collect()
    
    return sc_lstm, selector, mms, col_medians

def run_test(sc_lstm, selector, mms, col_medians):
    print("\n" + "=" * 60)
    print("PHASE 2: TEST SET PREPROCESSING")
    print("=" * 60)
    
    print("Loading Test CSVs...")
    df_pkt_test = pd.read_csv('Test Set/packet_based_testset.csv')
    df_sess_test = pd.read_csv('Test Set/session_based_testset.csv')
    
    # Ensure we only process UIDs present in both dataframes
    valid_uids_test = set(df_pkt_test['unique_link_mark']).intersection(set(df_sess_test['unique_link_mark']))
    df_pkt_test = df_pkt_test[df_pkt_test['unique_link_mark'].isin(valid_uids_test)]
    df_sess_test = df_sess_test[df_sess_test['unique_link_mark'].isin(valid_uids_test)]
    
    # LSTM Test
    uid_list_test = process_lstm(df_pkt_test, df_sess_test, is_train=False, sc_lstm=sc_lstm)
    
    del df_pkt_test
    gc.collect()
    
    # ResNet Test
    process_resnet(df_sess_test, uid_list_test, is_train=False, selector=selector, mms=mms)
    
    # XGBoost Test
    process_xgb(df_sess_test, uid_list_test, is_train=False, col_medians=col_medians)
    
    del df_sess_test
    gc.collect()

if __name__ == '__main__':
    # Ensure dirs exist
    os.makedirs('tensors', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)
    os.makedirs('probs', exist_ok=True)
    
    # Change CWD to project root to match CSV paths if it's executed from inside src/preprocessing
    current_dir = os.path.basename(os.getcwd())
    if current_dir == 'preprocessing':
        os.chdir('../../')
    
    sc_lstm, selector, mms, col_medians = run_train()
    run_test(sc_lstm, selector, mms, col_medians)
    
    print("\n" + "=" * 60)
    print("ALL PREPROCESSING COMPLETE")
    print("=" * 60)
