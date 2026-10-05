"""
Traffic Guardian — Admin Explainability Module (Level 1)

A fully separate FastAPI service (port 8643) providing SHAP-based
explanations for XGBoost and the Layer-2 Random Forest, from an uploaded
pcap. Deliberately isolated from backend/app.py (port 8642): its own
process, its own requirements file, read-only reuse of feature engineering
and model-loading code — never imports backend.app or its EnsemblePredictor,
never writes to any shared file. A crash here cannot affect /predict,
/predict_pcap, or /identify_hosts.

Run: python -m uvicorn app:app --host 127.0.0.1 --port 8643
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MAX_PCAP_BYTES = 50 * 1024 * 1024
MAX_FLOWS_LISTED = 100

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = FastAPI(
    title="Traffic Guardian — Admin Explainability",
    description="Standalone SHAP explainability module for XGBoost + Layer-2 RF",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

explainer = None


@app.on_event("startup")
def load_explainer():
    global explainer
    try:
        from shap_explain import AdminExplainer
        explainer = AdminExplainer(
            model_dir=os.path.join(PROJECT_ROOT, 'models'),
            scalers_dir=os.path.join(PROJECT_ROOT, 'scalers'),
            tensors_dir=os.path.join(PROJECT_ROOT, 'tensors'),
        )
    except Exception as e:
        print(f"[ERROR] Failed to load admin explainer: {e}")
        print("  The server will start but explain endpoints will return errors.")
        explainer = None


def _extract_flows(contents: bytes):
    import sys
    backend_dir = os.path.join(PROJECT_ROOT, 'backend')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from pcap_transform import extract_flows
    return extract_flows(contents)


def _validate_pcap_upload(file: UploadFile, contents: bytes):
    filename = file.filename or ""
    if not filename.lower().endswith(('.pcap', '.pcapng', '.cap')):
        raise HTTPException(status_code=400, detail="File must be .pcap, .pcapng, or .cap")
    if len(contents) > MAX_PCAP_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_PCAP_BYTES // (1024*1024)}MB limit")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "explainer_loaded": explainer is not None,
        "version": "1.0.0",
    }


@app.post("/admin/list_flows")
async def list_flows(file: UploadFile = File(...)):
    """Upload a pcap, get every flow's verdict/confidence — no SHAP yet (fast).
    The frontend re-submits the same file plus a chosen flow_index to
    /admin/explain_flow to get the full breakdown for just that one flow."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded")

    contents = await file.read()
    _validate_pcap_upload(file, contents)

    try:
        flows = _extract_flows(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    listed = []
    for i, flow in enumerate(flows[:MAX_FLOWS_LISTED]):
        try:
            result = explainer.predict_flow(flow)
        except Exception as e:
            result = {"verdict": "error", "error": str(e)}
        listed.append({
            "flow_index": i,
            "flow_id": flow["flow_id"],
            "label": flow["label"],
            "packet_count": flow["packet_count"],
            **result,
        })

    return {"flows_found": len(flows), "flows": listed}


@app.post("/admin/explain_flow")
async def explain_flow(file: UploadFile = File(...), flow_index: int = 0):
    """Upload the same pcap again plus the flow_index chosen from
    /admin/list_flows; returns the full SHAP breakdown for that one flow.
    Stateless by design — no server-side session/cache to manage."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded")

    contents = await file.read()
    _validate_pcap_upload(file, contents)

    try:
        flows = _extract_flows(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if flow_index < 0 or flow_index >= len(flows):
        raise HTTPException(status_code=400, detail=f"flow_index out of range (0-{len(flows)-1})")

    flow = flows[flow_index]
    try:
        result = explainer.explain_flow(flow)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")

    return {
        "flow_index": flow_index,
        "flow_id": flow["flow_id"],
        "label": flow["label"],
        "packet_count": flow["packet_count"],
        **result,
    }


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
