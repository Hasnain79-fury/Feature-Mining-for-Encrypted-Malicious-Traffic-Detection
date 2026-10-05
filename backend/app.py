"""
Traffic Guardian — FastAPI Backend

Serves the ML ensemble for real-time encrypted traffic classification.
Run: python -m uvicorn app:app --host 127.0.0.1 --port 8642
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

MAX_PCAP_BYTES = 50 * 1024 * 1024  # 50MB upload cap
MAX_FLOWS_PER_REQUEST = 100  # bound worst-case request latency

# Resolve project root (parent of backend/) — works regardless of clone location
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

app = FastAPI(
    title="Traffic Guardian",
    description="Real-time malicious encrypted traffic detection API",
    version="1.0.0",
)

# Allow Chrome extension to call us from any origin. No credentials (cookies/auth
# headers) are used, so allow_origins=["*"] is safe here — combining it with
# allow_credentials=True would be an invalid CORS configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


class PacketData(BaseModel):
    timestamp: float = 0
    duration: float = 0
    requestSize: float = 0
    responseSize: float = 0
    transferSize: float = 0
    encodedBodySize: float = 0
    decodedBodySize: float = 0
    headerSize: float = 0
    protocol: str = "unknown"
    statusCode: int = 0


class SessionPayload(BaseModel):
    session_id: str
    domain: str
    packet_count: Optional[int] = 0
    packets: List[PacketData]


class PredictionResponse(BaseModel):
    verdict: str
    confidence: float = 0.0
    malicious_prob: float = 0.0
    branches: dict = {}
    error: Optional[str] = None


@app.on_event("startup")
def load_models():
    """Load all models on server startup."""
    global predictor
    try:
        from inference import EnsemblePredictor
        predictor = EnsemblePredictor(
            model_dir=os.path.join(PROJECT_ROOT, 'models'),
            scalers_dir=os.path.join(PROJECT_ROOT, 'scalers'),
            tensors_dir=os.path.join(PROJECT_ROOT, 'tensors'),
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print("  The server will start but /predict will return errors.")
        predictor = None


@app.get("/health")
async def health():
    """Health check endpoint — extension polls this to show connection status."""
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: SessionPayload):
    """Run ensemble prediction on a browser session."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if len(payload.packets) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 packets")

    session_dict = {
        "session_id": payload.session_id,
        "domain": payload.domain,
        "packets": [p.model_dump() for p in payload.packets],
    }

    result = predictor.predict(session_dict)
    return result


@app.post("/predict_pcap")
async def predict_pcap(file: UploadFile = File(...)):
    """
    Run ensemble prediction on every TCP flow found in an uploaded
    .pcap/.pcapng file. Uses the higher-fidelity pcap feature path (real
    TTL/TCP-window/header data) — see feature_transform.py::transform_pcap.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    filename = file.filename or ""
    if not filename.lower().endswith(('.pcap', '.pcapng', '.cap')):
        raise HTTPException(status_code=400, detail="File must be .pcap, .pcapng, or .cap")

    contents = await file.read()
    if len(contents) > MAX_PCAP_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_PCAP_BYTES // (1024*1024)}MB limit")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        from pcap_transform import extract_flows
        flows = extract_flows(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    analyzed = []
    for flow in flows[:MAX_FLOWS_PER_REQUEST]:
        result = predictor.predict_pcap_flow(flow)
        analyzed.append({
            "flow_id": flow["flow_id"],
            "label": flow["label"],
            "packet_count": flow["packet_count"],
            **result,
        })

    return {
        "flows_found": len(flows),
        "flows_analyzed": len(analyzed),
        "flows": analyzed,
    }


@app.post("/identify_hosts")
async def identify_hosts_endpoint(file: UploadFile = File(...)):
    """
    Best-effort host identification (MAC/hostname/username) from a pcap —
    a separate, complementary capability from the ML classifier. Uses
    ARP/DHCP/NTLM parsing, not traffic statistics. See host_identify.py's
    module docstring for what it can and can't answer (e.g. never a full
    name — that's not generally a network-traffic artifact).
    """
    filename = file.filename or ""
    if not filename.lower().endswith(('.pcap', '.pcapng', '.cap')):
        raise HTTPException(status_code=400, detail="File must be .pcap, .pcapng, or .cap")

    contents = await file.read()
    if len(contents) > MAX_PCAP_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_PCAP_BYTES // (1024*1024)}MB limit")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        from host_identify import identify_hosts
        hosts = identify_hosts(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"hosts": hosts}


@app.get("/")
async def root():
    return {
        "name": "Traffic Guardian Backend",
        "status": "running",
        "docs": "/docs",
    }
