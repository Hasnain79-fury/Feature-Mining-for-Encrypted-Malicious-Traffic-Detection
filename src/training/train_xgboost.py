import os
import numpy as np
import joblib
import xgboost as xgb

# Resolve paths relative to the project root (not the script location)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure output folders exist
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'probs'),  exist_ok=True)

# Load pre-processed tensors (no scaling needed for XGBoost)
X_train = np.load(os.path.join(BASE, 'tensors/X_xgb_train.npy'))   # (N, 97)
X_val   = np.load(os.path.join(BASE, 'tensors/X_xgb_val.npy'))
X_test  = np.load(os.path.join(BASE, 'tensors/X_xgb_test.npy'))
y_train = np.load(os.path.join(BASE, 'tensors/y_train.npy'))
y_val   = np.load(os.path.join(BASE, 'tensors/y_val.npy'))

# XGBoost hyperparameters — exact from plan §5.3 / §11
xgb_clf = xgb.XGBClassifier(
    base_score=0.5,
    learning_rate=0.05,
    n_estimators=100,          # plan §11: n_estimators=100
    max_depth=10,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)

xgb_clf.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # plan §11: early_stopping_rounds=10
    verbose=False
)

# Persist the model
joblib.dump(xgb_clf, os.path.join(BASE, 'models/xgb_model.pkl'))

# Save probabilities for Layer-2 ensemble (train, val, AND test)
probs_train = xgb_clf.predict_proba(X_train)  # (N_train, 2)
probs_val   = xgb_clf.predict_proba(X_val)    # (N_val, 2)
probs_test  = xgb_clf.predict_proba(X_test)   # (N_test, 2)
np.save(os.path.join(BASE, 'probs/xgb_probs_train.npy'), probs_train)
np.save(os.path.join(BASE, 'probs/xgb_probs_val.npy'),   probs_val)
np.save(os.path.join(BASE, 'probs/xgb_probs_test.npy'),  probs_test)

print('✅ XGBoost training complete – model saved to models/xgb_model.pkl')
