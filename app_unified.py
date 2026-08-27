"""
app_unified.py
--------------
Unified, self-contained Cloud Run application for FoodGuard:
1. In-process BiLSTM classifier (SentenceTransformer + Keras).
2. BigQuery-backed evidence tools (Inspection history, recent complaints, cluster context).
3. LangGraph ReAct Agent powered by Google Gemini.
4. Deterministic Rule-Based Fallback logic.
5. Interactive Web UI & REST API (/health, /predict, /investigate, /).
"""
import os
import sys
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

# Ensure root config is available
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config import (
    GCP_PROJECT,
    EMBEDDING_MODEL,
    MODELS_DIR,
    GEMINI_MODEL,
    TRIAGE_LOG,
    TRIAGE_REVIEW,
    TRIAGE_ESCALATE,
    BQ_INSPECTIONS,
    BQ_CLUSTERS,
    BQ_COMPLAINTS_LABELLED,
    BQ_AGENT_DECISIONS,
    BQ_ESCALATIONS
)
from pipeline.routing.rule_based_triage import rule_based_triage

app = FastAPI(title="FoodGuard - Unified AI Triage System")

# Global models
embedder = None
classifier_model = None
model_loading_error = None
models_ready = False

MODEL_FILE = os.environ.get("MODEL_FILE", "bilstm_classweight.keras")
MODEL_PATH = MODELS_DIR / "bilstm" / MODEL_FILE
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


def init_models():
    """Background loader for deep learning models to ensure instant port binding."""
    global embedder, classifier_model, models_ready, model_loading_error
    try:
        print(f"[init_models] Loading SentenceTransformer: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"[init_models] Loading Keras classifier from: {MODEL_PATH}")
        import tensorflow as tf
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
        classifier_model = tf.keras.models.load_model(str(MODEL_PATH))
        models_ready = True
        print("[init_models] Models successfully initialized in memory.")
    except Exception as e:
        model_loading_error = str(e)
        print(f"[init_models] Error loading models: {e}")


@app.on_event("startup")
def startup_event():
    # Start model loading in a separate thread so Cloud Run health check passes immediately
    t = threading.Thread(target=init_models, daemon=True)
    t.start()


# ── BigQuery Tool Helpers ─────────────────────────────────────
_bq_client = None

def get_bq():
    global _bq_client
    if _bq_client is None:
        from google.cloud import bigquery
        _bq_client = bigquery.Client(project=GCP_PROJECT)
    return _bq_client


def get_inspection_history(camis_id: str) -> dict:
    camis_id = str(camis_id).strip()
    try:
        from google.cloud import bigquery
        query = f"""
            SELECT inspection_date, grade, action, critical_flag, score, violation_description
            FROM `{BQ_INSPECTIONS}`
            WHERE camis = @camis
            ORDER BY inspection_date DESC
            LIMIT 10
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("camis", "STRING", camis_id)]
        )
        rows = list(get_bq().query(query, job_config=job_config).result())
        if not rows:
            return {"camis": camis_id, "inspections": [], "note": "No prior inspections recorded."}
        return {
            "camis": camis_id,
            "inspections": [dict(r) for r in rows],
            "most_recent_grade": rows[0].get("grade") or "Not Graded",
            "last_action": rows[0].get("action") or ""
        }
    except Exception as e:
        return {"camis": camis_id, "error": str(e), "inspections": []}


def get_recent_complaints(camis_id: str, days: int = 30) -> dict:
    camis_id = str(camis_id).strip()
    try:
        from google.cloud import bigquery
        query = f"""
            SELECT complaint_type, descriptor, created_date, label
            FROM `{BQ_COMPLAINTS_LABELLED}`
            WHERE matched_camis = @camis
              AND created_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            ORDER BY created_date DESC
            LIMIT 20
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("camis", "STRING", camis_id),
                bigquery.ScalarQueryParameter("days", "INT64", days)
            ]
        )
        rows = list(get_bq().query(query, job_config=job_config).result())
        return {
            "camis": camis_id,
            "count": len(rows),
            "recent_complaints": [dict(r) for r in rows]
        }
    except Exception as e:
        return {"camis": camis_id, "count": 0, "error": str(e)}


def get_cluster_context(camis_id: str) -> dict:
    camis_id = str(camis_id).strip()
    try:
        from google.cloud import bigquery
        query = f"""
            SELECT cluster_id, descriptor, latitude, longitude
            FROM `{BQ_CLUSTERS}`
            WHERE matched_camis = @camis AND cluster_id >= 0
            LIMIT 5
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("camis", "STRING", camis_id)]
        )
        rows = list(get_bq().query(query, job_config=job_config).result())
        if not rows:
            return {"camis": camis_id, "in_active_cluster": False}
        return {
            "camis": camis_id,
            "in_active_cluster": True,
            "cluster_id": rows[0].get("cluster_id")
        }
    except Exception:
        return {"camis": camis_id, "in_active_cluster": False}


# ── Agent Factory ─────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def tool_inspection_history(camis: str) -> str:
            """Query official DOHMH past inspection history and grades for an establishment."""
            import json
            return json.dumps(get_inspection_history(camis), default=str)

        @tool
        def tool_recent_complaints(camis: str) -> str:
            """Query recent 311 citizen complaints filed against this establishment."""
            import json
            return json.dumps(get_recent_complaints(camis), default=str)

        @tool
        def tool_cluster_context(camis: str) -> str:
            """Check if establishment is part of a verified 5D spatiotemporal outbreak cluster."""
            import json
            return json.dumps(get_cluster_context(camis), default=str)

        api_key = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.1
        )
        tools = [tool_inspection_history, tool_recent_complaints, tool_cluster_context]
        system_prompt = (
            "You are FoodGuard, an AI decision support assistant for Environmental Health Officers.\n"
            "Analyze the complaint severity, retrieve inspection history and cluster context, and recommend a triage level:\n"
            "- LOG: Minor issue, record for next routine inspection.\n"
            "- REVIEW: Secondary risk or repeat complaints, review within 5 business days.\n"
            "- ESCALATE: Critical illness, severe pest infestation, or active outbreak; prioritize for immediate inspection within 48h.\n"
            "Always state your reasoning clearly tied to the evidence. You only recommend; human officers make regulatory decisions."
        )
        _agent = create_react_agent(llm, tools, prompt=system_prompt)
    return _agent


# ── Request / Response Schemas ─────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    severity_score: float
    severity_label: int
    model: str

class InvestigateRequest(BaseModel):
    complaint_id: Optional[str] = "COMP-DEMO"
    camis: Optional[str] = None
    text: str

class InvestigateResponse(BaseModel):
    complaint_id: str
    camis: Optional[str]
    severity_score: float
    severity_label: int
    triage_level: str
    reasoning: str
    evidence: dict
    used_rule_based_fallback: bool


# ── REST API Endpoints ────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": models_ready,
        "model_file": MODEL_FILE,
        "error": model_loading_error
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in 10 seconds.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty complaint text provided.")

    embedding = embedder.encode([req.text]).reshape(1, 1, -1)
    score = float(classifier_model.predict(embedding, verbose=0)[0][0])
    label = int(score >= 0.5)

    return PredictResponse(
        severity_score=round(score, 4),
        severity_label=label,
        model=MODEL_FILE
    )


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are still initializing.")

    # 1. Classify
    embedding = embedder.encode([req.text]).reshape(1, 1, -1)
    score = float(classifier_model.predict(embedding, verbose=0)[0][0])
    label = int(score >= 0.5)

    # 2. Gather Evidence
    evidence_data = {}
    if req.camis:
        evidence_data["inspections"] = get_inspection_history(req.camis)
        evidence_data["recent_complaints"] = get_recent_complaints(req.camis)
        evidence_data["cluster"] = get_cluster_context(req.camis)

    # 3. Agent Investigation
    tier = None
    reasoning = ""
    used_fallback = False

    try:
        agent = get_agent()
        user_msg = (
            f"Complaint regarding establishment CAMIS={req.camis or 'UNKNOWN'}: {req.text}\n"
            f"BiLSTM severity score: {score:.3f} (Label={label}).\n"
            f"Please investigate using your tools and recommend a triage tier (LOG, REVIEW, or ESCALATE)."
        )
        res = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        final_msg = res["messages"][-1].content
        reasoning = final_msg

        for candidate in (TRIAGE_ESCALATE, TRIAGE_REVIEW, TRIAGE_LOG):
            if candidate in final_msg.upper():
                tier = candidate
                break
    except Exception as e:
        reasoning = f"Agent reasoning error: {e}. Executing rule-based fallback."
        used_fallback = True

    if tier is None:
        used_fallback = True
        last_grade = evidence_data.get("inspections", {}).get("most_recent_grade", "")
        count = evidence_data.get("recent_complaints", {}).get("count", 0)
        tier = rule_based_triage(
            bilstm_pred=label,
            last_grade=last_grade,
            days_since_inspection=999,
            complaint_count_30d=count
        )
        if not reasoning:
            reasoning = f"Rule-based policy assigned {tier} based on severity score {score:.3f} and grade '{last_grade}'."

    return InvestigateResponse(
        complaint_id=req.complaint_id or "COMP-DEMO",
        camis=req.camis,
        severity_score=round(score, 4),
        severity_label=label,
        triage_level=tier,
        reasoning=reasoning,
        evidence=evidence_data,
        used_rule_based_fallback=used_fallback
    )


# ── Interactive Web UI ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FoodGuard — Cloud Run Decision Support</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
            .badge-LOG { background-color: #10b981; }
            .badge-REVIEW { background-color: #f59e0b; color: #000; }
            .badge-ESCALATE { background-color: #ef4444; }
            .btn-primary { background-color: #3b82f6; border-color: #3b82f6; }
            pre { background-color: #090d16; padding: 15px; border-radius: 8px; color: #38bdf8; white-space: pre-wrap; }
        </style>
    </head>
    <body class="py-5">
        <div class="container" style="max-width: 900px;">
            <div class="text-center mb-5">
                <h1 class="fw-bold text-primary">🛡️ FoodGuard</h1>
                <p class="text-secondary">Real-Time Food Safety Complaint Classification & Agentic Triage</p>
                <span class="badge bg-secondary">MSc Dissertation Prototype · Google Cloud Run</span>
            </div>

            <div class="card p-4 shadow-sm mb-4">
                <h5 class="mb-3">Submit Food Safety Complaint for Triage</h5>
                <div class="mb-3">
                    <label class="form-label text-secondary">Complaint Text</label>
                    <textarea id="complaintText" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="e.g. Severe vomiting and fever after eating raw tuna roll, saw live cockroaches behind counter."></textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label text-secondary">Establishment CAMIS (Optional, enables BigQuery history lookup)</label>
                    <input type="text" id="camisInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. 50002628 or 40368000">
                </div>
                <button onclick="runTriage()" class="btn btn-primary w-100 py-2 fw-semibold" id="btnSubmit">
                    Execute Agentic Triage
                </button>
            </div>

            <div id="resultsCard" class="card p-4 shadow-sm d-none mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0">Triage Assessment</h5>
                    <span id="triageBadge" class="badge fs-6 px-3 py-2"></span>
                </div>
                <div class="row text-center mb-3">
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">BiLSTM Severity Score</small>
                            <h3 id="severityScore" class="fw-bold mb-0 text-info"></h3>
                        </div>
                    </div>
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">Predicted Severity Flag</small>
                            <h3 id="severityLabel" class="fw-bold mb-0"></h3>
                        </div>
                    </div>
                </div>
                <h6 class="text-secondary">AI Agent Reasoning & Directives</h6>
                <pre id="agentReasoning"></pre>
            </div>
        </div>

        <script>
            async function runTriage() {
                const text = document.getElementById('complaintText').value.trim();
                const camis = document.getElementById('camisInput').value.trim();
                const btn = document.getElementById('btnSubmit');
                const resultsCard = document.getElementById('resultsCard');

                if (!text) { alert('Please enter complaint text.'); return; }

                btn.disabled = true;
                btn.innerText = 'Analyzing in GCP Cloud Run...';
                resultsCard.classList.add('d-none');

                try {
                    const resp = await fetch('/investigate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, camis: camis || null })
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        alert(data.detail || 'Error processing complaint.');
                        return;
                    }

                    document.getElementById('severityScore').innerText = data.severity_score;
                    document.getElementById('severityLabel').innerText = data.severity_label === 1 ? 'SEVERE ⚠️' : 'NON-SEVERE';
                    document.getElementById('severityLabel').className = 'fw-bold mb-0 ' + (data.severity_label === 1 ? 'text-danger' : 'text-success');

                    const badge = document.getElementById('triageBadge');
                    badge.innerText = data.triage_level;
                    badge.className = 'badge fs-6 px-3 py-2 badge-' + data.triage_level;

                    document.getElementById('agentReasoning').innerText = data.reasoning;
                    resultsCard.classList.remove('d-none');
                } catch (e) {
                    alert('Request failed: ' + e);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Execute Agentic Triage';
                }
            }
        </script>
    </body>
    </html>
    """
