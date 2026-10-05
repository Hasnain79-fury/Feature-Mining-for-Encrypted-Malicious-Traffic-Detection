"""
SHAP Explainability (Level 1) — standalone, read-only reuse of the live
backend's feature engineering and model-loading code.

Isolation: this module NEVER imports backend.app or backend.inference's
EnsemblePredictor, and never writes to any model/scaler file. It loads its
own copies of the same model objects into its own process (this admin
service), so a crash or bug here cannot affect /predict, /predict_pcap, or
/identify_hosts running in the separate backend process.

Explains XGBoost and the Layer-2 Random Forest only (both tree-based, so
shap.TreeExplainer is fast and exact). LSTM/ResNet-34 are out of scope for
level one — their *influence* still appears via the Layer-2 breakdown
(how much weight the RF gave their probability), just not as feature-level
attribution inside those two networks.

SHAP units differ by model — flagged explicitly rather than presented
uniformly, since treating them the same would be misleading:
  - XGBoost (binary:logistic): shap.TreeExplainer's default output is in
    log-odds ("margin") space, not probability space. A positive value
    still means "pushed toward malicious" and a negative value "pushed
    toward benign", but they don't sum to a 0-1 probability.
  - Layer-2 Random Forest: shap.TreeExplainer explains predict_proba
    directly for sklearn forests, so its SHAP values ARE probability
    contributions — expected_value + sum(shap_values) reconstructs the
    actual predict_proba output for the malicious class.
"""

import os
import sys
import numpy as np
import joblib

BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from feature_names import XGB_FEATURE_NAMES, META_FEATURE_NAMES, ALWAYS_MEDIAN_INDICES

TOP_N = 8


class AdminExplainer:
    def __init__(self, model_dir, scalers_dir, tensors_dir):
        # Reused read-only from backend/ — imported, never modified.
        from feature_transform import FeatureTransformer
        from inference import _import_tf, _import_torch, _load_resnet

        self.transformer = FeatureTransformer(model_dir, scalers_dir, tensors_dir)

        tf = _import_tf()
        self.lstm = tf.keras.models.load_model(
            os.path.join(model_dir, 'lstm_model.h5'), compile=False)
        self.lstm_input_features = self.lstm.input_shape[-1]

        torch, _, _ = _import_torch()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = _load_resnet(os.path.join(model_dir, 'resnet34.pt'), self.device)
        self._torch = torch

        self.xgb = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
        self.rf = joblib.load(os.path.join(model_dir, 'rf_layer2.pkl'))

        import shap
        self.xgb_explainer = shap.TreeExplainer(self.xgb)
        self.rf_explainer = shap.TreeExplainer(self.rf)

    def predict_flow(self, flow: dict) -> dict:
        """Verdict/confidence/branches only, no SHAP — fast path for listing
        every flow in an uploaded pcap."""
        result, _, _ = self._run_ensemble(flow)
        return result

    def explain_flow(self, flow: dict) -> dict:
        """Full verdict plus SHAP breakdown for one selected flow."""
        result, X_xgb, meta = self._run_ensemble(flow)

        # ── XGBoost SHAP (log-odds space, single output for binary:logistic) ──
        xgb_shap = self.xgb_explainer.shap_values(X_xgb)[0]  # (104,)
        xgb_contribs = self._rank_contributions(
            xgb_shap, XGB_FEATURE_NAMES,
            always_median=ALWAYS_MEDIAN_INDICES)

        # ── RF Layer-2 SHAP (probability space); take the malicious-class slice ──
        rf_shap_raw = self.rf_explainer.shap_values(meta)  # (1, 6, 2)
        rf_shap = np.asarray(rf_shap_raw)[0, :, 1]  # (6,) — push toward malicious
        rf_contribs = self._rank_contributions(rf_shap, META_FEATURE_NAMES)

        result["xgb_explanation"] = {
            "units": "log-odds",
            "base_value": round(float(self.xgb_explainer.expected_value), 4),
            "top_malicious": xgb_contribs["positive"],
            "top_benign": xgb_contribs["negative"],
        }
        result["rf_explanation"] = {
            "units": "probability",
            "base_value": round(float(np.asarray(self.rf_explainer.expected_value)[1]), 4),
            "top_malicious": rf_contribs["positive"],
            "top_benign": rf_contribs["negative"],
        }
        return result

    def _run_ensemble(self, flow: dict):
        """
        Run the full ensemble on a pcap flow (same math as
        EnsemblePredictor.predict_pcap_flow) while keeping the intermediate
        X_xgb tensor and the Layer-2 meta-vector, which the live backend's
        public API doesn't expose — needed here for SHAP.
        """
        X_lstm, X_resnet, X_xgb = self.transformer.transform_pcap(flow)

        if X_lstm.shape[2] != self.lstm_input_features:
            if X_lstm.shape[2] > self.lstm_input_features:
                X_lstm = X_lstm[:, :, :self.lstm_input_features]
            else:
                pad = np.zeros((1, 15, self.lstm_input_features - X_lstm.shape[2]), dtype=np.float32)
                X_lstm = np.concatenate([X_lstm, pad], axis=2)

        p_lstm = self.lstm.predict(X_lstm, verbose=0)[0]

        with self._torch.no_grad():
            t_input = self._torch.from_numpy(X_resnet).float().to(self.device)
            logits = self.resnet(t_input)
            p_resnet = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        p_xgb = self.xgb.predict_proba(X_xgb)[0]

        meta = np.hstack([p_lstm, p_resnet, p_xgb]).reshape(1, -1)
        p_final = self.rf.predict_proba(meta)[0]

        verdict = "malicious" if p_final[1] > 0.5 else "benign"
        result = {
            "verdict": verdict,
            "confidence": round(float(max(p_final)), 4),
            "malicious_prob": round(float(p_final[1]), 4),
            "branches": {
                "lstm": round(float(p_lstm[1]), 4),
                "resnet": round(float(p_resnet[1]), 4),
                "xgboost": round(float(p_xgb[1]), 4),
            },
        }
        return result, X_xgb, meta

    @staticmethod
    def _rank_contributions(shap_values, names, always_median=frozenset(), top_n=TOP_N):
        """Split a flat SHAP vector into top-N pushing toward malicious (positive)
        and top-N pushing toward benign (negative), sorted by magnitude."""
        pairs = [
            {
                "feature": names[i],
                "shap_value": round(float(shap_values[i]), 5),
                "always_median": i in always_median,
            }
            for i in range(len(shap_values))
        ]
        positive = sorted([p for p in pairs if p["shap_value"] > 0],
                           key=lambda p: -p["shap_value"])[:top_n]
        negative = sorted([p for p in pairs if p["shap_value"] < 0],
                           key=lambda p: p["shap_value"])[:top_n]
        return {"positive": positive, "negative": negative}
