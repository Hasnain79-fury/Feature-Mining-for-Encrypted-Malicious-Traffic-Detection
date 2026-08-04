"""
Inference — loads all 4 trained models and runs the multi-branch ensemble.

Models:
  Layer-1: LSTM (TensorFlow), ResNet-34 (PyTorch), XGBoost (sklearn)
  Layer-2: Random Forest ensemble over Layer-1 probabilities
"""

import os
import numpy as np
import joblib

# Lazy imports — only load frameworks when needed
_tf = None
_torch = None
_nn = None
_models = None


def _import_tf():
    global _tf
    if _tf is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        _tf = tf
    return _tf


def _import_torch():
    global _torch, _nn, _models
    if _torch is None:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        _torch = torch
        _nn = nn
        _models = models
    return _torch, _nn, _models


def _load_resnet(model_path, device):
    """Load ResNet-34 with the same architecture used in training."""
    torch, nn, models = _import_torch()

    net = models.resnet34(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net = net.to(device)
    net.eval()
    return net


class EnsemblePredictor:
    """Loads all models and provides a single predict() method."""

    def __init__(self, model_dir, scalers_dir, tensors_dir):
        from feature_transform import FeatureTransformer
        self.transformer = FeatureTransformer(model_dir, scalers_dir, tensors_dir)

        print("[INFO] Loading LSTM model...")
        tf = _import_tf()
        self.lstm = tf.keras.models.load_model(
            os.path.join(model_dir, 'lstm_model.h5'), compile=False)
        self.lstm_input_features = self.lstm.input_shape[-1]
        print(f"  LSTM expects input shape: {self.lstm.input_shape}")

        print("[INFO] Loading ResNet-34 model...")
        torch, _, _ = _import_torch()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = _load_resnet(
            os.path.join(model_dir, 'resnet34.pt'), self.device)
        print(f"  ResNet on device: {self.device}")

        print("[INFO] Loading XGBoost model...")
        self.xgb = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))

        print("[INFO] Loading RF Layer-2 ensemble...")
        self.rf = joblib.load(os.path.join(model_dir, 'rf_layer2.pkl'))

        print("[INFO] All models loaded successfully.")

    def predict(self, session_data: dict) -> dict:
        """
        Run the full 3-branch ensemble on a browser session.

        Parameters
        ----------
        session_data : dict
            Must contain 'packets' list with per-request metrics.

        Returns
        -------
        dict with verdict, confidence, branch scores.
        """
        try:
            X_lstm, X_resnet, X_xgb = self.transformer.transform(session_data)
        except Exception as e:
            return {"verdict": "error", "error": f"Feature transform failed: {e}"}

        # ── LSTM branch ──
        # Trim/pad features to match model expectation
        if X_lstm.shape[2] != self.lstm_input_features:
            if X_lstm.shape[2] > self.lstm_input_features:
                X_lstm = X_lstm[:, :, :self.lstm_input_features]
            else:
                pad = np.zeros((1, 15, self.lstm_input_features - X_lstm.shape[2]),
                               dtype=np.float32)
                X_lstm = np.concatenate([X_lstm, pad], axis=2)

        p_lstm = self.lstm.predict(X_lstm, verbose=0)[0]  # [prob_benign, prob_malicious]

        # ── ResNet branch ──
        torch, _, _ = _import_torch()
        with torch.no_grad():
            t_input = torch.from_numpy(X_resnet).float().to(self.device)
            logits = self.resnet(t_input)
            p_resnet = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # ── XGBoost branch ──
        p_xgb = self.xgb.predict_proba(X_xgb)[0]

        # ── Layer-2 RF ensemble ──
        meta = np.hstack([p_lstm, p_resnet, p_xgb]).reshape(1, -1)  # (1, 6)
        p_final = self.rf.predict_proba(meta)[0]

        verdict = "malicious" if p_final[1] > 0.5 else "benign"

        return {
            "verdict": verdict,
            "confidence": round(float(max(p_final)), 4),
            "malicious_prob": round(float(p_final[1]), 4),
            "branches": {
                "lstm":    round(float(p_lstm[1]), 4),
                "resnet":  round(float(p_resnet[1]), 4),
                "xgboost": round(float(p_xgb[1]), 4),
            }
        }
