import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Resolve paths relative to the project root (not the script location)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure output folder exists
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)

# ---- Load Layer-1 TRAINING probabilities for fitting Layer-2 (plan §6.1) ----
# Critical: train the RF on train-split probs, NOT val probs
lstm_p_train   = np.load(os.path.join(BASE, 'probs/lstm_probs_train.npy'))     # (N_train, 2)
resnet_p_train = np.load(os.path.join(BASE, 'probs/resnet_probs_train.npy'))   # (N_train, 2)
xgb_p_train    = np.load(os.path.join(BASE, 'probs/xgb_probs_train.npy'))     # (N_train, 2)

X_layer2_train = np.hstack([lstm_p_train, resnet_p_train, xgb_p_train])  # (N_train, 6)
y_train = np.load(os.path.join(BASE, 'tensors/y_train.npy'))

# ---- Load Layer-1 VALIDATION probabilities for evaluation ----
lstm_p_val   = np.load(os.path.join(BASE, 'probs/lstm_probs_val.npy'))
resnet_p_val = np.load(os.path.join(BASE, 'probs/resnet_probs_val.npy'))
xgb_p_val    = np.load(os.path.join(BASE, 'probs/xgb_probs_val.npy'))

X_layer2_val = np.hstack([lstm_p_val, resnet_p_val, xgb_p_val])  # (N_val, 6)
y_val = np.load(os.path.join(BASE, 'tensors/y_val.npy'))

# ---- Random-Forest Ensemble (Layer-2) — plan §6.2 ----
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    bootstrap=True,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_layer2_train, y_train)  # fit on TRAIN probs
joblib.dump(rf, os.path.join(BASE, 'models/rf_layer2.pkl'))
print('✅ Random Forest Layer-2 model saved to models/rf_layer2.pkl')

# ---- Average Ensemble (no training) — plan §6.3 ----
avg_probs_val = X_layer2_val.reshape(-1, 3, 2).mean(axis=1)  # (N_val, 2)
np.save(os.path.join(BASE, 'probs/avg_ensemble_probs_val.npy'), avg_probs_val)
print('✅ Average ensemble probabilities saved')

# ---- Evaluation on VALIDATION set ----
def eval_set(name, probs, y_true):
    preds = np.argmax(probs, axis=1)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, preds, target_names=['Benign','Malicious']))
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    auc = roc_auc_score(y_true, probs[:, 1])
    print(f'FPR: {fpr:.4%}   AUC: {auc:.4%}')

eval_set('Random Forest Layer-2 (Validation)', rf.predict_proba(X_layer2_val), y_val)
eval_set('Average Ensemble (Validation)',      avg_probs_val, y_val)
