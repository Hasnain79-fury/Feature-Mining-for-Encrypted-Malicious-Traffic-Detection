# Software Requirements Specification — Blueprint
## Encrypted Malicious Traffic Detection Framework

> **Purpose**: Feed this entire document into a separate AI to generate a formal IEEE 830 / ISO 29148 SRS report.

---

## 1. Project Identity

| Field | Value |
|-------|-------|
| **Title** | Feature Mining for Encrypted Malicious Traffic Detection |
| **Paper** | Wang & Thing (2023) — arXiv:2304.03691 |
| **Dataset** | Mendeley Data — doi:10.17632/xw7r4tt54g.1 |
| **Type** | ML-powered binary classification system with Chrome browser extension frontend |
| **Classification** | Benign (0) vs Malicious (1) encrypted network traffic |

---

## 2. Stakeholders & Users

| Stakeholder | Role | Interests |
|-------------|------|-----------|
| **End Users** (Browser users) | Install Chrome extension, receive real-time threat alerts | Minimal false alarms (FPR ≤ 0.11%), unobtrusive monitoring |
| **Security Analysts** | Review threat history, tune sensitivity, export logs | Detailed branch-level confidence, CSV export, SHAP explainability |
| **ML Engineers / Researchers** | Train/retrain models, evaluate performance | Reproducible pipeline, ablation studies, paper-matching metrics |
| **System Administrators** | Deploy/maintain the backend inference server | One-click launcher, health monitoring, resource management |
| **Project Supervisor / Academic** | Evaluate implementation fidelity to the paper | Correct architecture, matching performance benchmarks |

---

## 3. System Overview

The system is a **two-layer ensemble framework** with three components:

1. **Preprocessing Pipeline** (Python) — Transforms raw CSV datasets into model-ready tensors
2. **ML Training & Inference Backend** (Python/FastAPI) — Three Layer-1 models + Layer-2 ensemble
3. **Chrome Browser Extension** (Manifest V3 JS) — Captures live traffic, sends to backend, displays verdicts

---

## 4. Functional Requirements

### FR-1: Data Preprocessing Module

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Load packet CSV (4.4M rows × 25 cols) and session CSV (488K rows × 280 cols) | Must |
| FR-1.2 | Merge packet + session CSVs on `unique_link_mark` to produce 85 LSTM features (7 packet-level + 78 session-level) | Must |
| FR-1.3 | Group merged data by session, apply 15-packet cutoff, use **average padding** (not zero) → shape `(N, 15, 85)` | Must |
| FR-1.4 | Produce `uid_list` session ordering from LSTM groupby; align session CSV to this order before other branches | Must |
| FR-1.5 | Select top 38 payload columns from 48 candidates via `SelectKBest(mutual_info_classif, k=38)` for ResNet | Must |
| FR-1.6 | Generate 38×38 grayscale images via outer product for ResNet → shape `(N, 1, 38, 38)` | Must |
| FR-1.7 | Extract 65 `_ratio` + 32 ENC time session columns = 97 features for XGBoost → shape `(N, 97)` | Must |
| FR-1.8 | Apply unified train/val split (85/15, stratified) using single `idx_tr`/`idx_val` across all branches | Must |
| FR-1.9 | Fit StandardScaler (LSTM) and MinMaxScaler (ResNet) on training data only | Must |
| FR-1.10 | Clean NaN/Inf values: replace Inf→NaN, impute with column median | Must |
| FR-1.11 | Save all tensors, scalers, selectors, and indices to disk | Must |

### FR-2: Model Training Module

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Train Multi-layer LSTM: `LSTM(128)→Dropout(0.3)→LSTM(128)→Dropout(0.3)→Dense(256,relu)→Dense(2,softmax)` | Must |
| FR-2.2 | Train ResNet-34 adapted for 1-channel 38×38 input (conv1=3×3 stride 1, maxpool removed) | Must |
| FR-2.3 | Train XGBoost: `binary:logistic`, lr=0.05, n_estimators=100, max_depth=10 | Must |
| FR-2.4 | Save probability outputs `(N, 2)` from each Layer-1 model for train/val/test | Must |
| FR-2.5 | Train Layer-2 Random Forest ensemble on concatenated probs `(N, 6)` | Must |
| FR-2.6 | Implement Layer-2 Average Ensemble (no training) as alternative | Must |
| FR-2.7 | Early stopping: patience=5 (LSTM), patience=10 (XGBoost) | Should |

### FR-3: Backend Inference Server

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | FastAPI server on `127.0.0.1:8642` with CORS for Chrome extension | Must |
| FR-3.2 | `POST /predict` — Accept session JSON (packets array), return verdict + confidence + branch probs | Must |
| FR-3.3 | `GET /health` — Return server status and model-loaded flag | Must |
| FR-3.4 | `FeatureTransformer` class: map browser packet data → LSTM `(1,15,85)`, ResNet `(1,1,38,38)`, XGBoost `(1,97)` tensors | Must |
| FR-3.5 | `EnsemblePredictor` class: load all 4 models at startup, run 3 branches + Layer-2 ensemble | Must |
| FR-3.6 | Fill unavailable TCP/IP features (TTL, TCP window size) with training-set medians | Must |
| FR-3.7 | Reject requests with fewer than 3 packets (HTTP 400) | Should |
| FR-3.8 | One-click launcher script (`start_backend.bat`) | Should |

### FR-4: Chrome Extension — Data Collection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Manifest V3 service worker with `webRequest` permission to capture HTTP metadata | Must |
| FR-4.2 | Content script using Resource Timing API to collect per-request timing/size data | Must |
| FR-4.3 | `SessionManager`: group requests by `(tabId, domain)`, flush when ≥15 packets or idle >30s | Must |
| FR-4.4 | `feature_engineer.js`: transform raw packet data into session JSON payload for backend | Must |
| FR-4.5 | Send `POST /predict` to local backend, handle verdict response | Must |

### FR-5: Chrome Extension — User Interface

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Popup dashboard with dark-mode glassmorphism design | Must |
| FR-5.2 | Connection status indicator (green/red dot) based on `/health` polling | Must |
| FR-5.3 | Shield status icon: animated green=safe, red=threat detected | Must |
| FR-5.4 | Active domains list with verdict pills (safe/threat/monitoring) | Must |
| FR-5.5 | Threat detail card: branch-level confidence bars (LSTM/ResNet/XGBoost) | Should |
| FR-5.6 | History tab: last 50 verdicts in timeline | Should |
| FR-5.7 | Settings: backend URL, sensitivity threshold slider, notifications toggle | Should |
| FR-5.8 | Desktop notifications on malicious traffic detection | Should |
| FR-5.9 | Export threat history as CSV | Could |

### FR-6: Evaluation Module

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Compute Accuracy, F1, Precision, Recall/TPR, FPR, ROC-AUC | Must |
| FR-6.2 | FPR = FP/(FP+TN) on benign class (label=0 row of confusion matrix) | Must |
| FR-6.3 | Ablation study: each branch standalone, then full ensemble | Should |
| FR-6.4 | SHAP analysis on XGBoost features and Layer-2 RF feature importance | Could |

---

## 5. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | **Performance** | Accuracy ≥ 99.70%, F1 ≥ 99.70%, FPR ≤ 0.15%, TPR ≥ 99.50% |
| NFR-2 | **Latency** | Backend inference < 2 seconds per session |
| NFR-3 | **Scalability** | Handle concurrent extension requests (multiple tabs) |
| NFR-4 | **Privacy** | All processing local (127.0.0.1), no data sent externally |
| NFR-5 | **Reliability** | Graceful degradation when backend offline (queue sessions, retry) |
| NFR-6 | **Usability** | One-click backend start; extension installs via Chrome developer mode |
| NFR-7 | **Portability** | Backend runs on Windows with Python 3.8+; extension on any Chromium browser |
| NFR-8 | **Data Integrity** | Session alignment across all 3 branches guaranteed via `uid_list` |
| NFR-9 | **No Test Leakage** | Layer-2 trains only on Layer-1 training-split probabilities |
| NFR-10 | **Storage** | Threat history capped at 200 entries in `chrome.storage.local` |

---

## 6. UML Diagrams (descriptions for generation)

### 6.1 Use Case Diagram

**Actors**: End User, Security Analyst, ML Engineer, System Admin

**Use Cases**:
- End User: Browse web → Extension monitors traffic → View threat alerts → View active domains → Adjust sensitivity
- Security Analyst: View threat history → Export CSV → Inspect branch confidence → Run SHAP analysis
- ML Engineer: Run preprocessing → Train models → Run evaluation → Run ablation study
- System Admin: Start backend → Check health → Monitor logs

**Relationships**: 
- "View threat alerts" `<<includes>>` "Run ensemble prediction"
- "Run ensemble prediction" `<<includes>>` "Feature transformation"
- "Train models" `<<extends>>` "Run ablation study"

### 6.2 Class Diagram

```
FeatureTransformer
  - scaler_lstm: StandardScaler
  - scaler_resnet: MinMaxScaler
  - selector_resnet: SelectKBest
  - xgb_medians: ndarray
  + transform(session_data) → (X_lstm, X_resnet, X_xgb)
  + _extract_packet_features(packets) → ndarray
  + _compute_session_stats(pkt_features) → dict
  + _build_lstm_input(pkt_feats, sess_stats) → ndarray(1,15,85)
  + _build_resnet_input(sess_stats) → ndarray(1,1,38,38)
  + _build_xgb_input(sess_stats) → ndarray(1,97)

EnsemblePredictor
  - lstm: keras.Model
  - resnet: torch.nn.Module
  - xgb: XGBClassifier
  - rf: RandomForestClassifier
  - transformer: FeatureTransformer
  + predict(session_data) → dict{verdict, confidence, malicious_prob, branches}

FastAPI App
  + POST /predict(SessionPayload) → PredictionResponse
  + GET /health() → {status, models_loaded}

SessionManager (JS)
  - sessions: Map<key, {domain, packets[], lastActivity}>
  - IDLE_TIMEOUT: 30000
  - MIN_PACKETS: 15
  + addPacket(tabId, domain, packetData)
  + flush(sessionKey)
  + _checkIdle()

Pydantic Models:
  PacketData {timestamp, duration, requestSize, responseSize, transferSize, ...}
  SessionPayload {session_id, domain, packet_count, packets[]}
  PredictionResponse {verdict, confidence, malicious_prob, branches{}, error?}
```

### 6.3 Sequence Diagram — Real-Time Detection Flow

```
User → Browser: Navigate to website
Browser → Content.js: Page loads resources
Content.js → Background.js: RESOURCE_TIMINGS message (every 2s)
Background.js → SessionManager: addPacket(tabId, domain, data)
SessionManager: Check if packets ≥ 15
SessionManager → FeatureEngineer: buildSessionPayload(session)
FeatureEngineer → Backend /predict: POST session JSON
Backend → FeatureTransformer: transform(session_data)
FeatureTransformer → LSTM Model: predict(X_lstm) → probs(1,2)
FeatureTransformer → ResNet Model: predict(X_resnet) → probs(1,2)
FeatureTransformer → XGBoost Model: predict(X_xgb) → probs(1,2)
Backend → RF Layer-2: predict(concat_probs) → final verdict
Backend → Extension: {verdict, confidence, branches}
Background.js → Popup: Update UI (badge, shield, domain list)
Background.js → User: Desktop notification (if malicious)
```

### 6.4 Sequence Diagram — Training Pipeline

```
ML Engineer → run_all.py: Execute preprocessing
run_all.py → LSTM Pipeline: merge CSVs → groupby → cutoff → pad → (N,15,85) + uid_list
run_all.py → Alignment: reindex session CSV by uid_list
run_all.py → ResNet Pipeline: SelectKBest(38) → MinMaxScale → outer_product → (N,1,38,38)
run_all.py → XGBoost Pipeline: 65 ratio + 32 ENC = 97 cols → (N,97)
run_all.py → Disk: save tensors, scalers, indices
ML Engineer → train_lstm.py: fit LSTM model → save probs
ML Engineer → train_resnet.py: fit ResNet34 → save probs
ML Engineer → train_xgboost.py: fit XGBoost → save probs
ML Engineer → train_layer2.py: concat probs(N,6) → fit RF + compute avg ensemble
ML Engineer → metrics.py: evaluate all models → report
```

### 6.5 Component Diagram

```
[Chrome Extension]
  ├── content.js (Resource Timing collector)
  ├── background.js (Service worker, webRequest listener)
  ├── utils/session_manager.js (Session grouping)
  ├── utils/feature_engineer.js (Payload builder)
  └── popup/ (Dashboard UI)
       |
       | HTTP (localhost:8642)
       ↓
[FastAPI Backend]
  ├── app.py (API routes)
  ├── feature_transform.py (Browser→tensor mapping)
  └── inference.py (EnsemblePredictor)
       |
       | loads
       ↓
[Trained Models]
  ├── lstm_model.h5
  ├── resnet34.pt
  ├── xgb_model.pkl
  └── rf_layer2.pkl
       |
[Preprocessing Pipeline]
  ├── src/preprocessing/ (6 Python modules)
  ├── src/training/ (5 training scripts)
  └── src/evaluation/ (metrics)
       |
       | reads
       ↓
[Dataset]
  ├── Train Set/ (packet + session CSVs)
  └── Test Set/ (packet + session CSVs)
```

### 6.6 Deployment Diagram

```
<<device>> User's Windows PC
  ├── <<browser>> Chrome (Chromium)
  │     └── <<extension>> Traffic Guardian Extension (Manifest V3)
  │           ├── Service Worker (background.js)
  │           ├── Content Script (content.js)
  │           └── Popup UI (popup.html/js/css)
  │
  └── <<server>> Local Python Backend (127.0.0.1:8642)
        ├── <<runtime>> Python 3.8+
        ├── <<framework>> FastAPI + Uvicorn
        ├── <<model>> TensorFlow (LSTM)
        ├── <<model>> PyTorch (ResNet-34)
        ├── <<model>> XGBoost
        └── <<model>> scikit-learn (RF Layer-2)
```

---

## 7. Data Flow Diagram

### Level 0 (Context)
```
[User's Browser] → (Network Traffic) → [System] → (Verdict: Benign/Malicious) → [User]
```

### Level 1
```
[Browser] → {Raw HTTP metadata} → [1.0 Data Collection (Extension)]
  → {Session JSON: packets[]} → [2.0 Feature Transform (Backend)]
  → {X_lstm(1,15,85), X_resnet(1,1,38,38), X_xgb(1,97)} → [3.0 Model Inference]
  → {probs(1,2) × 3 branches} → [4.0 Ensemble (Layer-2)]
  → {verdict, confidence} → [5.0 Result Display (Popup)]
  → {Alert/Badge} → [User]
```

### Level 2 — Preprocessing (Offline)
```
[D1: packet_based_trainset.csv] + [D2: session_based_trainset.csv]
  → [2.1 Merge on unique_link_mark]
  → [2.2 LSTM Tensor Builder: groupby→cutoff→pad→(N,15,85)]
  → [2.3 Session Alignment by uid_list]
  → [2.4 ResNet: SelectKBest→MinMax→outer_product→(N,1,38,38)]
  → [2.5 XGBoost: 97 cols extract→clean→(N,97)]
  → [D3: tensors/*.npy] + [D4: scalers/*.pkl]
```

---

## 8. Dataset Specification

| Property | Value |
|----------|-------|
| Packet CSV (Train) | 4,435,304 rows × 25 columns |
| Session CSV (Train) | 488,524 rows × 280 columns |
| Session ID | `unique_link_mark` (shared key) |
| Label | `label` — 0=Benign, 1=Malicious |
| Balance | ~243K benign / ~245K malicious (near-balanced) |
| Malicious families | 26 (Ammyy, Cerber, Dridex, TrickBot, Ursnif, Zbot, etc.) |
| Benign sources | CIRA-CIC-DoHBRW-2020, CICIDS-2017, CICIDS-2012 |

### Feature Distribution Across Branches

| Branch | Features | Source | Shape |
|--------|----------|--------|-------|
| LSTM | 85 time features (7 packet + 78 session broadcast) | Both CSVs merged | (N, 15, 85) |
| ResNet | 38 payload stats (selected from 48 candidates) | Session CSV | (N, 1, 38, 38) |
| XGBoost | 97 (65 ratio + 32 ENC time) | Session CSV | (N, 97) |
| Layer-2 | 6 (concatenated branch probs) | Model outputs | (N, 6) |

---

## 9. API Contract

### POST /predict

**Request:**
```json
{
  "session_id": "123:example.com",
  "domain": "example.com",
  "packets": [
    {"timestamp": 1720300000000, "duration": 45.2, "requestSize": 512,
     "responseSize": 15360, "transferSize": 15872, "encodedBodySize": 15200,
     "decodedBodySize": 48000, "headerSize": 672, "protocol": "h2", "statusCode": 200}
  ]
}
```

**Response:**
```json
{
  "verdict": "malicious" | "benign",
  "confidence": 0.9987,
  "malicious_prob": 0.9987,
  "branches": {"lstm": 0.98, "resnet": 0.97, "xgboost": 0.99}
}
```

### GET /health
```json
{"status": "ok", "models_loaded": true, "version": "1.0.0"}
```

---

## 10. Complete File Inventory

### Preprocessing (`src/preprocessing/`)
| File | Purpose |
|------|---------|
| `feature_columns.py` | All column definitions (PKT_TIME_COLS, SESS_TIME_COLS, PAYLOAD_CANDIDATES, ENC_TIME_SESS_COLS) |
| `lstm_pipeline.py` | Merge → groupby → cutoff → average-pad → `(N, 15, 85)` |
| `resnet_pipeline.py` | SelectKBest → MinMaxScale → outer product → `(N, 1, 38, 38)` |
| `xgb_pipeline.py` | 65 ratio + 32 ENC → `(N, 97)` |
| `run_all.py` | Master orchestrator (correct execution order) |
| `verification.py` | Shape/NaN/alignment checks |

### Training (`src/training/`)
| File | Purpose |
|------|---------|
| `train_lstm.py` | LSTM(128×2) training with early stopping |
| `train_resnet.py` | ResNet-34 PyTorch training |
| `train_xgboost.py` | XGBClassifier training |
| `train_layer2.py` | RF ensemble + average ensemble |
| `run_all_training.py` | Sequential training orchestrator |

### Backend (`backend/`)
| File | Purpose |
|------|---------|
| `app.py` | FastAPI routes (/predict, /health) |
| `inference.py` | EnsemblePredictor class |
| `feature_transform.py` | Browser data → model tensors |
| `requirements.txt` | Python dependencies |
| `start_backend.bat` | One-click Windows launcher |

### Extension (`extension/`)
| File | Purpose |
|------|---------|
| `manifest.json` | Manifest V3 config |
| `background.js` | Service worker: webRequest + session management |
| `content.js` | Resource Timing API collector |
| `utils/session_manager.js` | Session grouping + flush logic |
| `utils/feature_engineer.js` | Packet list → feature vector JSON |
| `popup/popup.html` | Dashboard layout |
| `popup/popup.css` | Glassmorphism dark-mode styling |
| `popup/popup.js` | Render verdicts, history, settings |

### Trained Models (`models/`)
| File | Size |
|------|------|
| `lstm_model.h5` | 3.4 MB |
| `resnet34.pt` | 85.2 MB |
| `xgb_model.pkl` | 1.8 MB |
| `rf_layer2.pkl` | 5.3 MB |

---

## 11. Performance Targets

| Metric | Average Ensemble | RF Ensemble |
|--------|-----------------|-------------|
| Accuracy | 99.73% | 99.65% |
| F1 Score | 99.72% | 99.65% |
| Precision | 99.89% | 99.63% |
| Recall/TPR | 99.56% | 99.68% |
| FPR | **0.11%** | 0.32% |
| ROC-AUC | 99.94% | 99.95% |

---

## 12. Suggested SRS Document Outline

Use this structure when generating the full report. This mirrors the academic prototype format.

```
List of Stakeholders
    - End User (Browser User)
    - Security Analyst
    - ML Engineer / Researcher
    - System Administrator
    - Project Supervisor / Academic Advisor

Stakeholders Viewpoints
    After talking with the stakeholders, we collected the following viewpoints
    of requirements from them.

    End User (Browser User)
        - Wants real-time detection of malicious encrypted traffic while browsing
        - Expects minimal false alarms (FPR ≤ 0.11%)
        - Wants unobtrusive monitoring with clear threat alerts
        - Needs one-click backend start, no technical setup

    Security Analyst
        - Wants detailed branch-level confidence breakdown (LSTM, ResNet, XGBoost)
        - Needs threat history with export to CSV
        - Wants sensitivity threshold tuning
        - Requires SHAP explainability for model decisions

    ML Engineer / Researcher
        - Needs reproducible preprocessing and training pipeline
        - Wants ablation study capability (each branch standalone vs ensemble)
        - Expects paper-matching performance metrics
        - Requires modular code for experimentation

    System Administrator
        - Needs one-click launcher (start_backend.bat)
        - Wants health check endpoint (/health) for monitoring
        - Requires local-only deployment (127.0.0.1), no external data transfer
        - Expects graceful error handling when models fail to load

    Project Supervisor / Academic Advisor
        - Wants faithful reproduction of Wang & Thing (2023) framework
        - Expects correct two-layer ensemble architecture
        - Needs documented performance benchmarks matching the paper
        - Requires clean, well-documented codebase

Normal Requirements (Kano Model)
    1. System shall classify encrypted network traffic as Benign or Malicious
    2. System shall use a two-layer ensemble (LSTM + ResNet + XGBoost → RF/Avg)
    3. System shall preprocess raw CSVs into model-ready tensors
    4. System shall provide a REST API for real-time inference
    5. System shall display verdicts in a Chrome extension popup
    6. System shall achieve Accuracy ≥ 99.70%
    7. System shall compute and report FPR on the benign class
    8. System shall handle NaN/Inf values via median imputation

Expected Requirements
    1. System shall use average padding (not zero-padding) for LSTM sequences
    2. System shall align all three branches to the same session order via uid_list
    3. System shall fit scalers on training data only (no test leakage)
    4. System shall provide desktop notifications for detected threats
    5. System shall show connection status (green/red) in the extension popup
    6. System shall queue sessions when backend is offline and retry on reconnect
    7. System shall cap threat history at 200 entries

Exciting Requirements
    1. SHAP analysis for XGBoost feature importance visualization
    2. Branch-level confidence bars (LSTM / ResNet / XGBoost) in threat detail card
    3. Sensitivity threshold slider for adjustable detection aggressiveness
    4. Animated shield icon with pulse effect on threat detection
    5. Glassmorphism dark-mode UI with micro-animations
    6. CSV export of full threat history
    7. Ablation study automation comparing all model variants

User Stories
    1. Data Preprocessing & Loading
        - As an ML Engineer, I want to load both packet CSV (4.4M rows) and
          session CSV (488K rows) so that I can prepare training data.
        - As an ML Engineer, I want to merge both CSVs on unique_link_mark
          to produce 85 LSTM features so that the model receives complete input.
        - As an ML Engineer, I want the system to apply average padding to
          sessions shorter than 15 packets so that temporal features are
          not distorted by zeros.

    2. Model Training & Evaluation
        - As an ML Engineer, I want to train LSTM, ResNet-34, and XGBoost
          independently so that I can evaluate each branch's standalone performance.
        - As an ML Engineer, I want to concatenate branch probabilities (N,6)
          and train a Layer-2 RF ensemble so that I can improve overall accuracy.
        - As a Researcher, I want to run ablation studies comparing all model
          variants so that I can validate the paper's findings.

    3. Real-Time Traffic Analysis
        - As an End User, I want the Chrome extension to automatically capture
          my browsing traffic so that malicious connections are detected.
        - As an End User, I want to see a shield icon turn red when a threat
          is detected so that I am immediately alerted.
        - As a Security Analyst, I want to inspect which branch flagged a
          session so that I can assess detection confidence.

    4. Backend Inference
        - As a System Admin, I want a one-click batch script to start the
          backend server so that no CLI knowledge is required.
        - As a System Admin, I want a /health endpoint so that I can
          monitor if models are loaded and the server is running.
        - As an End User, I want the backend to respond within 2 seconds
          so that browsing is not noticeably delayed.

    5. Threat Management & History
        - As a Security Analyst, I want to view the last 50 verdicts in a
          timeline so that I can identify patterns.
        - As a Security Analyst, I want to export threat history as CSV
          so that I can perform offline analysis.
        - As an End User, I want to adjust the sensitivity threshold so
          that I can balance between false alarms and missed detections.

    6. Security & Privacy
        - As an End User, I want all processing to happen locally (127.0.0.1)
          so that my browsing data is never sent externally.
        - As a System Admin, I want CORS restricted to Chrome extension
          origins so that unauthorized clients cannot access the API.
        - As an ML Engineer, I want Layer-2 to train only on training-split
          probabilities so that there is no test leakage.

Actors
    - End User (Browser User)
    - Security Analyst
    - ML Engineer
    - System Administrator
    - Chrome Extension (automated agent)
    - FastAPI Backend (automated agent)
    - LSTM Model (internal)
    - ResNet-34 Model (internal)
    - XGBoost Model (internal)
    - RF Layer-2 Ensemble (internal)

Use Case Diagrams:
    Level: 0  (Context-level — entire system as single process)
        Actors: End User, ML Engineer, System Admin
        System boundary: "Encrypted Malicious Traffic Detection Framework"

    Level: 1  (Major subsystems)
        1.1 — Data Preprocessing & Feature Engineering
        1.2 — LSTM Branch Training & Inference
        1.3 — ResNet Branch Training & Inference
        1.4 — XGBoost Branch Training & Inference
        1.5 — Layer-2 Ensemble Detection
        1.6 — Chrome Extension Data Collection & UI
        1.7 — Backend API Server

    Level: 1.1  (Data Preprocessing expanded)
        1.1.1 — Load & Merge CSVs (packet + session on unique_link_mark)
        1.1.2 — Build LSTM Tensor (groupby → cutoff 15 → average pad → N,15,85)
        1.1.3 — Align Session CSV (reindex by uid_list)
        1.1.4 — Build ResNet Tensor (SelectKBest 38 → MinMax → outer product → N,1,38,38)
        1.1.5 — Build XGBoost Features (65 ratio + 32 ENC = 97 cols)
        1.1.6 — Train/Val Split (unified idx_tr, idx_val across all branches)
        1.1.7 — Fit & Save Scalers (StandardScaler LSTM, MinMaxScaler ResNet)

    Level: 1.1.1  (Load & Merge expanded)
        1.1.1.1 — Read packet_based_trainset.csv (4.4M × 25)
        1.1.1.2 — Read session_based_trainset.csv (488K × 280)
        1.1.1.3 — Validate column existence (7 PKT_TIME_COLS, 78 SESS_TIME_COLS)
        1.1.1.4 — Left-join on unique_link_mark → merged (4.4M × 103)
        1.1.1.5 — Clean NaN/Inf → median imputation

    Level: 1.2  (LSTM Branch)
        1.2.1 — Load LSTM tensor (N, 15, 85)
        1.2.2 — Train LSTM(128×2) with Dropout(0.3), Dense(256), Dense(2)
        1.2.3 — Early stopping (patience=5 on val loss)
        1.2.4 — Save model (lstm_model.h5)
        1.2.5 — Generate & save probability outputs (N, 2)

    Level: 1.3  (ResNet Branch)
        1.3.1 — Load ResNet tensor (N, 1, 38, 38)
        1.3.2 — Initialize ResNet-34 (conv1=3×3 stride 1, maxpool removed, FC→2)
        1.3.3 — Train with Adam optimizer
        1.3.4 — Save model (resnet34.pt)
        1.3.5 — Generate & save probability outputs (N, 2)

    Level: 1.4  (XGBoost Branch)
        1.4.1 — Load XGBoost features (N, 97)
        1.4.2 — Train XGBClassifier (binary:logistic, lr=0.05, max_depth=10)
        1.4.3 — Early stopping (patience=10 on eval logloss)
        1.4.4 — Save model (xgb_model.pkl)
        1.4.5 — Generate & save probability outputs (N, 2)

    Level: 1.5  (Layer-2 Ensemble)
        1.5.1 — Load all 3 branch probability arrays
        1.5.2 — Concatenate → (N, 6)
        1.5.3 — Train Random Forest (n_estimators=200)
        1.5.4 — Compute Average Ensemble (no training)
        1.5.5 — Evaluate both methods, select best

    Level: 1.6  (Chrome Extension)
        1.6.1 — content.js: Harvest Resource Timing entries every 2s
        1.6.2 — background.js: Listen webRequest.onCompleted for HTTP metadata
        1.6.3 — SessionManager: Group by (tabId, domain), flush at ≥15 packets
        1.6.4 — FeatureEngineer: Build session JSON payload
        1.6.5 — POST /predict to backend, receive verdict
        1.6.6 — Update popup UI (badge, shield, domain list)
        1.6.7 — Show notification if malicious

    Level: 1.7  (Backend API Server)
        1.7.1 — Startup: Load all 4 models + scalers + selectors
        1.7.2 — POST /predict: Accept SessionPayload → FeatureTransform → 3-branch inference → Layer-2 → verdict
        1.7.3 — GET /health: Return status + models_loaded flag
        1.7.4 — Error handling: 400 for <3 packets, 503 if models not loaded

Activity Diagrams:
    Level: 1    (System-wide activity flow)
    Level: 1.1  (Preprocessing pipeline activity)
    Level: 1.1.1 (CSV merge & clean activity)
    Level: 1.2  (LSTM training activity)
    Level: 1.3  (ResNet training activity)
    Level: 1.4  (XGBoost training activity)
    Level: 1.5  (Layer-2 ensemble activity)
    Level: 1.6  (Extension data collection activity)
    Level: 1.7  (Backend inference activity)

Swimlane Diagrams:
    SID (Swimlane ID): 1      (System overview — lanes: End User | Extension | Backend | Models)
    SID (Swimlane ID): 1.1    (Preprocessing — lanes: ML Engineer | LSTM Pipeline | ResNet Pipeline | XGBoost Pipeline | Disk)
    SID (Swimlane ID): 1.1.1  (CSV Merge — lanes: Packet CSV | Session CSV | Merge Process | Clean Process)
    SID (Swimlane ID): 1.2    (LSTM Training — lanes: Data Loader | LSTM Model | Evaluator | Disk)
    SID (Swimlane ID): 1.3    (ResNet Training — lanes: Data Loader | ResNet Model | Evaluator | Disk)
    SID (Swimlane ID): 1.4    (XGBoost Training — lanes: Data Loader | XGBoost Model | Evaluator | Disk)
    SID (Swimlane ID): 1.5    (Ensemble — lanes: Prob Loader | RF Trainer | Avg Ensemble | Evaluator)
    SID (Swimlane ID): 1.6    (Extension Flow — lanes: Content Script | Background Worker | Session Manager | Backend API | Popup UI)
    SID (Swimlane ID): 1.7    (Backend Inference — lanes: API Router | Feature Transformer | LSTM | ResNet | XGBoost | RF Ensemble)

Data-Based Modelling:
    Noun Identification:
        From requirements: User, Session, Packet, Domain, Verdict, Branch,
        Probability, Model, Tensor, Feature, Scaler, Label, Threshold,
        Notification, History Entry, Settings, CSV File, Extension,
        Backend, Confidence, Malware Family, Dataset, UID List, Index,
        Median, Prediction, Error, Health Status

    Final Data Objects:
        Session, Packet, Prediction, Model, Tensor, Scaler, Feature Column,
        Threshold Setting, History Entry, Health Status, Dataset

    Relations:
        Session (1) ←contains→ (N) Packet
        Session (1) ←produces→ (1) Prediction
        Prediction (1) ←uses→ (3) Model (LSTM, ResNet, XGBoost)
        Prediction (1) ←uses→ (1) Model (RF Layer-2)
        Model (1) ←requires→ (1) Tensor
        Model (1) ←requires→ (0..1) Scaler
        Tensor (1) ←built from→ (N) Feature Column
        Prediction (1) ←stored as→ (1) History Entry
        Dataset (1) ←contains→ (N) Session

    ER Diagram:
        Entities: Session, Packet, Prediction, LSTMModel, ResNetModel,
                  XGBoostModel, RFEnsemble, Tensor, Scaler, HistoryEntry,
                  Settings, Dataset
        (Generate full ER diagram with attributes and cardinalities)

    Schema Diagram:
        sessions(session_id PK, domain, packet_count, created_at)
        packets(packet_id PK, session_id FK, timestamp, duration, requestSize,
                responseSize, transferSize, encodedBodySize, decodedBodySize,
                headerSize, protocol, statusCode)
        predictions(prediction_id PK, session_id FK, verdict, confidence,
                    malicious_prob, lstm_prob, resnet_prob, xgboost_prob,
                    created_at)
        history_entries(entry_id PK, domain, verdict, confidence, timestamp)
        models(model_id PK, model_type, file_path, file_size, loaded_status)
        tensors(tensor_id PK, branch, shape, scaler_path, file_path)
        settings(key PK, value, default_value)

Class-Based Modelling:
    Identified Nouns:
        FeatureTransformer, EnsemblePredictor, SessionManager, FeatureEngineer,
        PacketData, SessionPayload, PredictionResponse, StandardScaler,
        MinMaxScaler, SelectKBest, LSTMModel, ResNet34, XGBClassifier,
        RandomForestClassifier, FastAPIApp, ContentScript, BackgroundWorker,
        PopupUI, HistoryManager, SettingsManager, HealthChecker

    Identified Verbs:
        transform, predict, addPacket, flush, buildPayload, harvestTimings,
        loadModels, checkHealth, renderHistory, renderActiveSessions,
        exportCSV, showNotification, updateBadge, storeToHistory,
        fitTransform, selectKBest, cleanNaN, mergeCsvs, buildTensor,
        alignSessions, splitData, saveModel, loadModel, computeMetrics

    General Classification:
        - Entity classes: PacketData, SessionPayload, PredictionResponse, HistoryEntry, Settings
        - Boundary classes: PopupUI, ContentScript, FastAPIApp
        - Control classes: FeatureTransformer, EnsemblePredictor, SessionManager,
                           FeatureEngineer, BackgroundWorker, HealthChecker

    Selection Criteria:
        (Apply retained information, needed services, multiple attributes,
         common attributes, common operations, essential requirements)

    Attribute and Method Identification:
        (For each class: list all attributes with types and all methods with signatures)

    CRC Cards:
        (Class-Responsibility-Collaborator cards for each of 15 classes)

    Class Cards:
        Class Card: FeatureTransformer
        Class Card: EnsemblePredictor
        Class Card: SessionManager
        Class Card: FeatureEngineer
        Class Card: PacketData
        Class Card: SessionPayload
        Class Card: PredictionResponse
        Class Card: FastAPIApp
        Class Card: PopupUI
        Class Card: ContentScript
        Class Card: BackgroundWorker
        Class Card: HistoryManager
        Class Card: SettingsManager
        Class Card: HealthChecker
        Class Card: LSTMModel

    CRC Diagrams:
        Diagram Id: 1   Name: FeatureTransformer
        Diagram Id: 2   Name: EnsemblePredictor
        Diagram Id: 3   Name: SessionManager
        Diagram Id: 4   Name: FeatureEngineer
        Diagram Id: 5   Name: PacketData
        Diagram Id: 6   Name: SessionPayload
        Diagram Id: 7   Name: PredictionResponse
        Diagram Id: 8   Name: FastAPIApp
        Diagram Id: 9   Name: PopupUI
        Diagram Id: 10  Name: ContentScript
        Diagram Id: 11  Name: BackgroundWorker
        Diagram Id: 12  Name: HistoryManager
        Diagram Id: 13  Name: SettingsManager
        Diagram Id: 14  Name: HealthChecker
        Diagram Id: 15  Name: LSTMModel

Behavioral Modelling:
    Event Table:
        (Events × Actor × Response matrix for all system interactions:
         Browse website, Packet captured, Session flushed, Prediction requested,
         Verdict received, Threat detected, History viewed, Settings changed,
         Backend started, Health checked, Model loaded, Training started,
         Preprocessing run, Export requested)

    State Transition Diagrams:
        ID: 1   Session lifecycle (Idle → Collecting → Ready → Analyzing → Verdicted → Archived)
        ID: 2   Backend server (Starting → Loading Models → Ready → Processing → Error → Ready)
        ID: 3   Extension (Installed → Connecting → Monitoring → Threat Alert → Monitoring)
        ID: 4   LSTM Training (Init → Loading Data → Training → Validating → Saving → Done)
        ID: 5   ResNet Training (Init → Loading Data → Training → Validating → Saving → Done)
        ID: 6   XGBoost Training (Init → Loading Data → Training → Validating → Saving → Done)
        ID: 7   Layer-2 Ensemble (Init → Loading Probs → Training RF → Eval Avg → Comparing → Done)
        ID: 8   Preprocessing Pipeline (Init → Loading CSVs → Merging → LSTM Build → Align → ResNet Build → XGB Build → Verify → Done)
        ID: 9   Prediction Request (Received → Validating → Transforming → LSTM Infer → ResNet Infer → XGB Infer → Ensemble → Response)
        ID: 10  Popup UI (Closed → Opening → Checking Health → Loading History → Live Monitoring → Closed)
        ID: 11  Packet Data (Captured → Enriched → Grouped → Flushed → Processed)
        ID: 12  Threat Notification (Triggered → Displayed → Acknowledged → Dismissed)
        ID: 13  Model File (Untrained → Training → Saved → Loaded → Serving)

    Sequence Diagrams:
        SD-1: Real-Time Traffic Detection (End User → Browser → Extension → Backend → Models → Extension → User)
        SD-2: Preprocessing Pipeline (ML Engineer → run_all.py → LSTM/ResNet/XGB pipelines → Disk)
        SD-3: Model Training Flow (ML Engineer → train scripts → Models → probs → Layer-2)
        SD-4: Threat History Export (Security Analyst → Popup → chrome.storage → CSV download)
        SD-5: Backend Health Check (Extension → /health → Popup status dot update)
        SD-6: Settings Change (End User → Popup → chrome.storage → SessionManager config update)
```

---

## 13. Key Constraints & Pitfalls to Document

1. **LSTM must be built first** — produces `uid_list` that aligns all branches
2. **Average padding, NOT zero-padding** — for temporal features
3. **No test leakage in Layer-2** — train only on training-split probabilities
4. **Scalers fit on train only** — transform val/test separately
5. **Binary classification throughout** — `num_classes = 2`, not multiclass
6. **85 LSTM features via merge** — 7 packet + 78 session, not just 19 packet columns
7. **Browser features are proxies** — TCP/IP fields unavailable from browser, filled with training medians
8. **FPR is class-specific** — `FP/(FP+TN)` on benign row, not macro-averaged
