import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Resolve paths relative to the project root (not the script location)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load test-set probabilities from all three branches
lstm_p   = np.load(os.path.join(BASE, 'probs/lstm_probs_test.npy'))     # (N_test, 2)
resnet_p = np.load(os.path.join(BASE, 'probs/resnet_probs_test.npy'))
xgb_p    = np.load(os.path.join(BASE, 'probs/xgb_probs_test.npy'))

# Ground-truth labels
y_test = np.load(os.path.join(BASE, 'tensors/y_test.npy'))

# ---------- Random Forest Layer-2 ----------
rf = joblib.load(os.path.join(BASE, 'models/rf_layer2.pkl'))
X_test = np.hstack([lstm_p, resnet_p, xgb_p])
rf_probs = rf.predict_proba(X_test)

# ---------- Average Ensemble (no training) ----------
avg_probs = X_test.reshape(-1, 3, 2).mean(axis=1)

def report(name, probs):
    preds = np.argmax(probs, axis=1)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=['Benign','Malicious']))
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    auc = roc_auc_score(y_test, probs[:, 1])
    print(f'FPR: {fpr:.4%}   AUC: {auc:.4%}')

report('Random Forest Layer-2 (Test)', rf_probs)
report('Average Ensemble (Test)',      avg_probs)
