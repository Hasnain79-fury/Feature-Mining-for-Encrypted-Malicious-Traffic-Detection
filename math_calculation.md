The confidence calculation lives entirely in backend/inference.py — the extension never touches it, only reclassifies verdict against the threshold (which I built in the last session). Here's the exact math, branch by branch.

Layer-1 — each branch outputs its own 2-class probability

LSTM: final layer is Dense(2, softmax), so p_lstm = softmax(z) = [e^z₀/(e^z₀+e^z₁), e^z₁/(e^z₀+e^z₁)]
ResNet-34: raw logits → torch.softmax(logits, dim=1) — same softmax formula
XGBoost: binary:logistic objective, predict_proba() applies sigmoid to the summed boosted-tree output: p_malicious = 1/(1+e^(-F(x))) where F(x) is the sum of leaf values across all 100 trees; returned as [1-p, p]
Each gives a 2-vector [p_benign, p_malicious]. The "Branch Scores" you see in the UI (LSTM/ResNet/XGBoost %) are literally p_lstm[1], p_resnet[1], p_xgb[1] — each branch's own malicious-class probability, nothing ensembled yet.

Layer-2 — Random Forest meta-classifier (inference.py:125-126)


meta = np.hstack([p_lstm, p_resnet, p_xgb]).reshape(1, -1)  # (1,6)
p_final = self.rf.predict_proba(meta)[0]
meta is [p_lstm_benign, p_lstm_mal, p_resnet_benign, p_resnet_mal, p_xgb_benign, p_xgb_mal]. A Random Forest's predict_proba is the average of every tree's leaf class-distribution:


p_final = (1/T) · Σₜ tree_t.predict_proba(meta)      for t = 1..200 trees
Each tree routes meta down to a leaf and returns the fraction of training samples of each class that landed in that leaf; the forest averages those 200 fractions.

Final numbers returned to the extension (inference.py:130-139)


verdict        = "malicious" if p_final[1] > 0.5 else "benign"
confidence     = max(p_final)        # NOT necessarily the malicious probability
malicious_prob = p_final[1]
The subtlety: confidence ≠ malicious_prob. confidence is max(p_final[0], p_final[1]) — the probability mass behind whichever class won. For a confidently-benign session, confidence could be 0.98 while malicious_prob is 0.02. They only converge near the malicious side of the boundary.

Where your threshold slider fits in: the reclassification I added in background.js (classify(malicious_prob) = malicious_prob >= threshold/100 ? 'malicious' : 'benign') only overrides verdict. It never recomputes confidence — that stays exactly max(p_final) from the RF, since it's meant to describe the model's own certainty, not your alert policy. So sliding the threshold can flip a session's verdict without changing the confidence number shown for it.

1. Backend computes it and puts it in the JSON response (inference.py:132):


"confidence": round(float(max(p_final)), 4)
2. background.js receives that JSON and copies it through, untouched:


const result = await res.json();          // result.confidence is already the final number
storeToSessionLog(payload, result, ...);   // writes confidence: result.confidence into chrome.storage
No math happens here — it's just read off the response and passed along.

3. popup.js reads it back out of storage and formats it for display (openDetailPanel):


document.getElementById('detailConfidence').textContent =
  item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—';
This is the only "calculation" the extension ever does on confidence: × 100 and .toFixed(1) — pure display formatting (0.9987 → "99.9%"), not a re-derivation of the value. The CSV export does the same ×100 formatting via its pct() helper.

So the number itself is computed exactly once, in the backend, by max(p_final) off the Random Forest's output — the extension is just a pass-through-and-format pipe for it, same as it does for malicious_prob and the branch scores.