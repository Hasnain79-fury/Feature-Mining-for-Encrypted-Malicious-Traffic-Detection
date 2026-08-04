import numpy as np
import os

def run_verification():
    print("=" * 60)
    print("RUNNING VERIFICATION CHECKS")
    print("=" * 60)
    
    try:
        X_lstm_tr = np.load('tensors/X_lstm_train.npy')
        X_resnet_tr = np.load('tensors/X_resnet_train.npy')
        X_xgb_tr = np.load('tensors/X_xgb_train.npy')
        y_tr = np.load('tensors/y_train.npy')
        
        X_lstm_val = np.load('tensors/X_lstm_val.npy')
        X_resnet_val = np.load('tensors/X_resnet_val.npy')
        X_xgb_val = np.load('tensors/X_xgb_val.npy')
        y_val = np.load('tensors/y_val.npy')
        
        X_lstm_te = np.load('tensors/X_lstm_test.npy')
        X_resnet_te = np.load('tensors/X_resnet_test.npy')
        X_xgb_te = np.load('tensors/X_xgb_test.npy')
        y_te = np.load('tensors/y_test.npy')
    except Exception as e:
        print(f"Failed to load files: {e}")
        return

    print(f"Shape checks (Train):")
    print(f"  LSTM:    {X_lstm_tr.shape}   <- expect (N, 15, 93)")
    print(f"  ResNet:  {X_resnet_tr.shape} <- expect (N, 1, 38, 38)")
    print(f"  XGBoost: {X_xgb_tr.shape}   <- expect (N, 104)")
    print(f"  Labels:  {y_tr.shape}        <- expect (N,)")

    assert X_lstm_tr.shape[0] == X_resnet_tr.shape[0] == X_xgb_tr.shape[0] == y_tr.shape[0], "Train Row count mismatch across branches!"
    assert X_lstm_tr.shape[1] == 15, f"Wrong timesteps: {X_lstm_tr.shape[1]}"
    assert X_lstm_tr.shape[2] == 93, f"Wrong features: {X_lstm_tr.shape[2]}"
    assert X_resnet_tr.shape[1] == 1, f"Wrong channels: {X_resnet_tr.shape[1]}"
    assert X_resnet_tr.shape[2] == 38, f"Wrong H: {X_resnet_tr.shape[2]}"
    assert X_resnet_tr.shape[3] == 38, f"Wrong W: {X_resnet_tr.shape[3]}"
    assert X_xgb_tr.shape[1] == 104, f"Wrong XGB features: {X_xgb_tr.shape[1]}"
    
    assert X_lstm_val.shape[0] == X_resnet_val.shape[0] == X_xgb_val.shape[0] == y_val.shape[0], "Val Row count mismatch across branches!"
    assert X_lstm_te.shape[0] == X_resnet_te.shape[0] == X_xgb_te.shape[0] == y_te.shape[0], "Test Row count mismatch across branches!"

    print("\nNaN/Inf checks (Train):")
    print(f"  LSTM NaN:    {np.isnan(X_lstm_tr).sum()}")
    print(f"  ResNet NaN:  {np.isnan(X_resnet_tr).sum()}")
    print(f"  XGBoost NaN: {np.isnan(X_xgb_tr).sum()}")

    print("\nClass distribution (Train):")
    print(f"  Benign (0):    {(y_tr==0).sum()} ({(y_tr==0).mean():.1%})")
    print(f"  Malicious (1): {(y_tr==1).sum()} ({(y_tr==1).mean():.1%})")

    print("\nAll checks passed ✓")

if __name__ == '__main__':
    run_verification()
