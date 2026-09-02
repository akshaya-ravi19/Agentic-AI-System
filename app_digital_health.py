"""
app_digital_health.py
---------------------
Digital Health & Clinical Syndromic Surveillance Triage Service.

Features:
1. Digital Health BiLSTM model (SentenceTransformers + Keras bilstm_digital_health.keras)
2. Clinical Syndromic Hazard & Priority Pathogen Analysis (CDC FoodNet & FDA Model Code)
3. BigQuery Epidemiological Evidence & 5D HDBSCAN Outbreak Context
4. Gemini ReAct Clinical Decision Support Agent
5. Dedicated Digital Public Health Dashboard & REST API
"""
import os
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

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
    BQ_COMPLAINTS_LABELLED
)
from pipeline.routing.rule_based_triage import rule_based_triage

embedder = None
classifier_model = None
model_loading_error = None
models_ready = False

MODEL_FILE = "bilstm_digital_health.keras"
MODEL_PATH = MODELS_DIR / "bilstm" / MODEL_FILE
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


def _load_digital_health_models():
    global embedder, classifier_model, models_ready, model_loading_error
    try:
        print(f"[DigitalHealth] Loading SentenceTransformer: {EMBEDDING_MODEL}", flush=True)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"[DigitalHealth] Loading BiLSTM model: {MODEL_PATH}", flush=True)
        import tensorflow as tf
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
        classifier_model = tf.keras.models.load_model(str(MODEL_PATH))
        models_ready = True
        print("[DigitalHealth] All models successfully loaded into memory.", flush=True)
    except Exception as e:
        model_loading_error = str(e)
        print(f"[DigitalHealth] Error loading models: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_digital_health_models, daemon=True)
    t.start()
    yield


app = FastAPI(title="Digital Health Syndromic Triage Platform", lifespan=lifespan)

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
            WHERE CAST(matched_camis AS STRING) = @camis
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
            WHERE CAST(matched_camis AS STRING) = @camis AND cluster_id >= 0
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
            """Query DOHMH historical violation records and inspection grades."""
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
            "You are a Digital Public Health & Environmental Epidemiological Decision Support Agent.\n"
            "Evaluate citizen syndromic hazard complaints, investigate historical establishment inspection records and outbreak cluster evidence, "
            "and assign an actionable Public Health Priority Triage Level:\n"
            "- LOG: Minor administrative or non-pathogenic issue (Routine inspection tracking).\n"
            "- REVIEW: Secondary syndromic risk, repeat complaints, or hygiene hazards (Review within 5 business days).\n"
            "- ESCALATE: Acute foodborne illness, severe biological contamination (vermin/sewage), or active outbreak cluster (Immediate priority inspection within 48h).\n"
            "Provide clinical syndromic justification and risk reasoning tied directly to evidence."
        )
        _agent = create_react_agent(llm, tools, prompt=system_prompt)
    return _agent


class InvestigateRequest(BaseModel):
    complaint_id: Optional[str] = "SYNDROMIC-DEMO"
    camis: Optional[str] = None
    restaurant_name: Optional[str] = None
    location: Optional[str] = None
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": models_ready,
        "model_file": MODEL_FILE,
        "framework": "Digital Health Syndromic Surveillance"
    }


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are still initializing.")

    # 1. Embed & Predict Hazard
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
        est_desc = []
        if req.restaurant_name:
            est_desc.append(f"Name='{req.restaurant_name}'")
        if req.location:
            est_desc.append(f"Location='{req.location}'")
        if req.camis:
            est_desc.append(f"CAMIS={req.camis}")
        est_str = ", ".join(est_desc) if est_desc else "Establishment=Unspecified"

        user_msg = (
            f"Citizen complaint ({est_str}): {req.text}\n"
            f"Digital Health Syndromic Hazard Score: {score:.3f} (Class={label}).\n"
            f"Please conduct an epidemiological investigation and recommend a triage tier (LOG, REVIEW, or ESCALATE)."
        )
        res = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        raw_msg = res["messages"][-1].content
        if isinstance(raw_msg, list):
            final_msg = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in raw_msg])
        else:
            final_msg = str(raw_msg)
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

    return {
        "complaint_id": req.complaint_id,
        "camis": req.camis,
        "syndromic_score": round(score, 4),
        "hazard_flag": "ACTIONABLE HAZARD" if label == 1 else "ROUTINE / ADMINISTRATIVE",
        "triage_level": tier,
        "epidemiological_reasoning": reasoning,
        "evidence": evidence_data,
        "used_rule_based_fallback": used_fallback
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Digital Health Syndromic Surveillance Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0b132b; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            .card { background-color: #1c2541; border: 1px solid #3a506b; border-radius: 12px; }
            .badge-LOG { background-color: #10b981; }
            .badge-REVIEW { background-color: #f59e0b; color: #000; }
            .badge-ESCALATE { background-color: #ef4444; }
            .btn-primary { background-color: #00b4d8; border-color: #00b4d8; color: #03045e; font-weight: 700; }
            .btn-primary:hover { background-color: #90e0ef; border-color: #90e0ef; color: #03045e; }
            .section-title { color: #48cae4; font-weight: 700; border-bottom: 2px solid #0096c7; padding-bottom: 8px; display: inline-block; }
            pre { background-color: #070d1e; padding: 15px; border-radius: 8px; color: #90e0ef; white-space: pre-wrap; border: 1px solid #1f3160; }
        </style>
    </head>
    <body class="py-5">
        <div class="container" style="max-width: 950px;">
            <div class="text-center mb-4">
                <h2 class="fw-bold text-light">🩺 Digital Health Syndromic Surveillance & Triage</h2>
                <p class="text-secondary mb-1">Computational Public Health Intelligence · Early Outbreak Detection · Environmental Health CDSS</p>
                <span class="badge bg-info bg-opacity-25 text-info border border-info border-opacity-25 px-3 py-1">CDC FoodNet & FDA Model Code Aligned</span>
            </div>

            <div class="card p-4 shadow-sm mb-4">
                <div class="mb-3">
                    <h5 class="section-title mb-0">Submit Syndromic Complaint for Public Health Triage</h5>
                </div>
                
                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Complaint / Symptom Description <span class="text-danger">*</span></label>
                    <textarea id="complaintText" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="e.g. Acute onset of vomiting, high fever, and severe abdominal cramps after consuming undercooked seafood. Saw live cockroaches on kitchen cutting boards."></textarea>
                </div>

                <div class="row g-3 mb-3">
                    <div class="col-md-4">
                        <label class="form-label text-secondary fw-semibold">Establishment ID (CAMIS) <span class="text-muted fw-normal">(Optional)</span></label>
                        <input type="text" id="camisInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. 50002628">
                    </div>
                    <div class="col-md-4">
                        <label class="form-label text-secondary fw-semibold">Restaurant Name <span class="text-muted fw-normal">(Optional)</span></label>
                        <input type="text" id="restaurantNameInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. Ocean Blue Seafood">
                    </div>
                    <div class="col-md-4">
                        <label class="form-label text-secondary fw-semibold">Location / Address <span class="text-muted fw-normal">(Optional)</span></label>
                        <input type="text" id="locationInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. 420 Lexington Ave, Manhattan">
                    </div>
                </div>

                <button onclick="runTriage()" class="btn btn-primary w-100 py-2" id="btnSubmit">
                    Execute Digital Health Triage
                </button>
            </div>

            <div id="resultsCard" class="card p-4 shadow-sm d-none mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0 text-light">Epidemiological Triage Assessment</h5>
                    <span id="triageBadge" class="badge fs-6 px-3 py-2"></span>
                </div>
                <div class="row text-center mb-3">
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">Syndromic Hazard Score</small>
                            <h3 id="severityScore" class="fw-bold mb-0 text-info"></h3>
                        </div>
                    </div>
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">Public Health Hazard Flag</small>
                            <h3 id="severityLabel" class="fw-bold mb-0"></h3>
                        </div>
                    </div>
                </div>
                <h6 class="text-secondary">Epidemiological Decision Reasoning & Clinical Context</h6>
                <pre id="agentReasoning"></pre>
            </div>
        </div>

        <script>
            async function runTriage() {
                const text = document.getElementById('complaintText').value.trim();
                const camis = document.getElementById('camisInput').value.trim();
                const restaurant_name = document.getElementById('restaurantNameInput').value.trim();
                const location = document.getElementById('locationInput').value.trim();
                const btn = document.getElementById('btnSubmit');
                const resultsCard = document.getElementById('resultsCard');

                if (!text) { alert('Please enter symptom / complaint details.'); return; }

                btn.disabled = true;
                btn.innerText = 'Analyzing Syndromic Indicators...';
                resultsCard.classList.add('d-none');

                try {
                    const resp = await fetch('/investigate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: text,
                            camis: camis || null,
                            restaurant_name: restaurant_name || null,
                            location: location || null
                        })
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        alert(data.detail || 'Error processing request.');
                        return;
                    }

                    document.getElementById('severityScore').innerText = data.syndromic_score;
                    document.getElementById('severityLabel').innerText = data.hazard_flag;
                    document.getElementById('severityLabel').className = 'fw-bold mb-0 ' + (data.hazard_flag.includes('ACTIONABLE') ? 'text-danger' : 'text-success');

                    const badge = document.getElementById('triageBadge');
                    badge.innerText = data.triage_level;
                    badge.className = 'badge fs-6 px-3 py-2 badge-' + data.triage_level;

                    document.getElementById('agentReasoning').innerText = data.epidemiological_reasoning;
                    resultsCard.classList.remove('d-none');
                } catch (e) {
                    alert('Request failed: ' + e);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Execute Digital Health Triage';
                }
            }
        </script>
    </body>
    </html>
    """
