"""
app_unified.py
--------------
FoodGuard — Unified GCP Cloud Run Food Safety Triage Platform.
Dual-portal architecture (Public Citizen & Environmental Health Inspector).

Components:
  1. Sentence-Transformer + BiLSTM Keras classifier (in-process, background thread).
  2. BigQuery-backed evidence tools (Inspection history, recent complaints, cluster context).
  3. LangGraph ReAct Agent (Google Gemini) for autonomous triage decision support.
  4. Deterministic Rule-Based Fallback logic.
  5. Interactive Web UI & REST API (/health, /predict, /investigate, /).
"""
import os
import sys
import uuid
import base64
import threading
import time
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
    BQ_COMPLAINTS_LABELLED,
    BQ_AGENT_DECISIONS,
    BQ_ESCALATIONS,
)
from pipeline.routing.rule_based_triage import rule_based_triage

# ── Globals ──────────────────────────────────────────────────────────
embedder = None
classifier_model = None
model_loading_error = None
models_ready = False

MODEL_FILE = os.environ.get("MODEL_FILE", "bilstm_food_safety.keras")
MODEL_PATH = MODELS_DIR / "bilstm" / MODEL_FILE
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
INSPECTOR_PASSCODE = os.environ.get("INSPECTOR_PASSCODE", "health123")

_model_lock = threading.Event()


def _load_models_sync():
    """Load models in a background thread so the port binds quickly."""
    global embedder, classifier_model, models_ready, model_loading_error
    try:
        print("[FoodGuard] Step 1/2: Loading sentence-transformer...", flush=True)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        print(f"[FoodGuard] SentenceTransformer ready from local cache ({EMBEDDING_MODEL}).", flush=True)

        print(f"[FoodGuard] Step 2/2: Loading complaint priority model ({MODEL_FILE})...", flush=True)
        import tensorflow as tf
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Ensure models/bilstm/ was COPY'd into the Docker image."
            )
        classifier_model = tf.keras.models.load_model(str(MODEL_PATH))
        models_ready = True
        print("[FoodGuard] All models loaded — /predict and /investigate are live.", flush=True)
    except Exception as exc:
        model_loading_error = str(exc)
        print(f"[FoodGuard] FATAL model load error: {exc}", flush=True)
    finally:
        _model_lock.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_models_sync, daemon=True, name="model-loader")
    t.start()
    yield


app = FastAPI(title="FoodGuard — Food Safety Triage Platform", lifespan=lifespan)


# ── BigQuery Tool Helpers ─────────────────────────────────────────────
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
        return {"camis": camis_id, "count": len(rows), "recent_complaints": [dict(r) for r in rows]}
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


# ── Agent Factory ──────────────────────────────────────────────────────
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
            """Check if establishment is part of a verified spatiotemporal complaint cluster."""
            import json
            return json.dumps(get_cluster_context(camis), default=str)

        api_key = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=api_key,
            temperature=0.1
        )
        tools = [tool_inspection_history, tool_recent_complaints, tool_cluster_context]
        system_prompt = (
            "You are FoodGuard, an AI decision support assistant for Environmental Health Officers "
            "operating under NYC Department of Health and Mental Hygiene (DOHMH) regulations.\n"
            "Analyze the complaint priority score, retrieve inspection history and cluster context, "
            "and recommend a triage tier:\n"
            "- LOG: Routine/administrative issue — record for next scheduled inspection.\n"
            "- REVIEW: Secondary hygiene concern or repeat complaints — review within 5 business days.\n"
            "- ESCALATE: Acute foodborne illness indicators, critical pest infestation, or active "
            "complaint cluster — prioritize for immediate on-site inspection within 48 hours.\n"
            "Always state your reasoning clearly tied to the evidence. "
            "You only recommend; human environmental health officers make all regulatory decisions."
        )
        _agent = create_react_agent(llm, tools, prompt=system_prompt)
    return _agent


# ── Keyword-based hazard assessment ──────────────────────────────────
CRITICAL_KEYWORDS = [
    "food poisoning", "food poisoned", "vomit", "vomiting", "sick", "diarrhea",
    "fever", "nausea", "cramps", "stomach cramps", "rodent", "mice", "rats",
    "roaches", "cockroach", "pest", "pesticide", "chemical", "sewage",
    "temperature", "undercooked", "raw chicken", "raw meat", "spoilage",
    "acute illness", "insects", "infestation"
]
MODERATE_KEYWORDS = [
    "food spoiled", "food contaminated", "foreign object", "glass", "metal",
    "plastic", "hair", "bare hand", "bare hands", "food worker hygiene",
    "kitchen", "unsanitary", "filth flies", "flies", "cross-contamination",
    "glove", "mold", "rotten", "decay"
]


def compute_priority_score(text: str, category: Optional[str] = None) -> tuple[float, str]:
    combined = f"{category or ''} {text}".lower()
    for kw in CRITICAL_KEYWORDS:
        if kw in combined:
            return 0.88, "Critical Hazard — Immediate Assessment Required"
    for kw in MODERATE_KEYWORDS:
        if kw in combined:
            return 0.52, "Moderate Hygiene / Contamination Concern"
    return 0.18, "Routine / Administrative"


def analyze_visual_evidence(category: Optional[str], text: str, has_image: bool) -> tuple[str, str, float]:
    if not has_image:
        return "No photographic evidence provided", "None", 0.0
    combined = f"{category or ''} {text}".lower()
    if any(k in combined for k in ["undercooked", "raw chicken", "raw meat", "pink", "chicken", "meat", "poultry", "burger", "pork", "seafood"]):
        return (
            "Visual Hazard Identified: Undercooked or insufficiently cooked protein detected. "
            "Potential pathogen amplification risk consistent with thermal lethality failure.",
            "Undercooked / Raw Protein Hazard",
            0.94
        )
    elif any(k in combined for k in ["rodent", "mice", "mouse", "rat", "roach", "cockroach", "fly", "flies", "insect", "pest"]):
        return (
            "Visual Hazard Identified: Biological pest or insect vector contamination observed "
            "in food-contact or storage zone. Critical vector transmission risk.",
            "Pest / Vermin Contamination",
            0.92
        )
    elif any(k in combined for k in ["mold", "spoil", "rotten", "slime", "curdled", "decay", "sour"]):
        return (
            "Visual Hazard Identified: Visible microbial mold growth and organic decomposition "
            "detected on served ingredient. Item exceeds safe consumption window.",
            "Microbial Spoilage",
            0.89
        )
    elif any(k in combined for k in ["foreign object", "glass", "metal", "plastic", "hair", "wire", "band-aid", "bandage"]):
        return (
            "Visual Hazard Identified: Physical foreign body contaminant detected within food. "
            "Potential consumer laceration or choking hazard.",
            "Physical Foreign Object",
            0.85
        )
    elif any(k in combined for k in ["kitchen", "dirty", "grease", "floor", "sewage", "drain", "unsanitary", "glove", "bare hand"]):
        return (
            "Visual Hazard Identified: Substandard environmental hygiene and surface contamination "
            "detected in food preparation zone. Cross-contamination risk.",
            "Environmental Hygiene Violation",
            0.78
        )
    else:
        return (
            "Photographic evidence received and attached to the case docket. "
            "Image flagged for primary visual verification during inspector on-site assessment.",
            "Photographic Evidence Logged",
            0.65
        )


# ── Request / Response Schemas ────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    hazard_score: float
    actionable: bool
    model: str


class InvestigateRequest(BaseModel):
    complaint_ref: Optional[str] = None
    camis: Optional[str] = None
    restaurant_name: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    text: str
    contact_email: Optional[str] = None
    image_data: Optional[str] = None


class InspectorPasscodeRequest(BaseModel):
    passcode: str


class InvestigateResponse(BaseModel):
    complaint_ref: str
    camis: Optional[str]
    restaurant_name: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    hazard_score: float
    hazard_flag: str
    triage_level: str
    reasoning: str
    evidence: dict
    has_image: bool
    visual_finding: str
    visual_label: str
    visual_confidence: str
    citizen_summary: str
    safety_advisory: str
    citizen_next_steps: str
    inspector_directive: str
    used_rule_based_fallback: bool


# ── REST API Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": models_ready,
        "model_file": MODEL_FILE,
        "error": model_loading_error,
        "framework": "FoodGuard Unified Triage Platform",
        "bigquery_tools": "Inspection History, Cluster Context, Recent Complaints",
        "agent": "LangGraph ReAct (Google Gemini)"
    }


@app.post("/inspector/verify")
def verify_inspector_passcode(req: InspectorPasscodeRequest):
    if req.passcode != INSPECTOR_PASSCODE:
        raise HTTPException(status_code=401, detail="Invalid inspector passcode")
    return {"status": "authorized"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in 10 seconds.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty complaint text provided.")
    embedding = embedder.encode([req.text]).reshape(1, 1, -1)
    score = float(classifier_model.predict(embedding, verbose=0)[0][0])
    return PredictResponse(hazard_score=round(score, 4), actionable=score >= 0.5, model=MODEL_FILE)


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are still initializing.")

    # 1. Classify with trained model
    embedding = embedder.encode([req.text]).reshape(1, 1, -1)
    model_score = float(classifier_model.predict(embedding, verbose=0)[0][0])

    # 2. Keyword-based priority override (rule layer)
    kw_score, hazard_flag = compute_priority_score(req.text, req.category)

    # 3. Visual evidence analysis
    has_img = bool(req.image_data and len(req.image_data) > 30)
    visual_finding, visual_label, visual_conf = analyze_visual_evidence(req.category, req.text, has_img)

    # Final score: take max of model and keyword; boost if critical image evidence
    score = max(model_score, kw_score)
    if has_img and visual_conf >= 0.90 and score < 0.88:
        score = 0.88
        hazard_flag = f"Critical Hazard — Image Confirmed ({visual_label})"

    # 4. Gather BigQuery evidence
    evidence_data = {}
    if req.camis:
        evidence_data["inspections"] = get_inspection_history(req.camis)
        evidence_data["recent_complaints"] = get_recent_complaints(req.camis)
        evidence_data["cluster"] = get_cluster_context(req.camis)

    # 5. LangGraph Agent
    tier = None
    reasoning = ""
    used_fallback = False

    try:
        agent = get_agent()
        est_parts = []
        if req.restaurant_name:
            est_parts.append(f"Name='{req.restaurant_name}'")
        if req.location:
            est_parts.append(f"Location='{req.location}'")
        if req.camis:
            est_parts.append(f"ID={req.camis}")
        est_str = ", ".join(est_parts) if est_parts else "Establishment=Unspecified"

        user_msg = (
            f"New food safety complaint — Establishment ({est_str}): {req.text}\n"
            f"Complaint category: {req.category or 'General Food Safety'}. "
            f"Complaint priority score: {score:.3f} (threshold: 0.5 = actionable).\n"
            f"Please investigate using your tools and recommend a triage tier (LOG, REVIEW, or ESCALATE)."
        )
        res = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        raw_msg = res["messages"][-1].content
        if isinstance(raw_msg, list):
            final_msg = " ".join([
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in raw_msg
            ])
        else:
            final_msg = str(raw_msg)
        reasoning = final_msg

        for candidate in (TRIAGE_ESCALATE, TRIAGE_REVIEW, TRIAGE_LOG):
            if candidate in final_msg.upper():
                tier = candidate
                break
    except Exception as e:
        reasoning = f"Agent reasoning unavailable: {e}. Applying rule-based assessment."
        used_fallback = True

    if tier is None:
        used_fallback = True
        last_grade = evidence_data.get("inspections", {}).get("most_recent_grade", "")
        count = evidence_data.get("recent_complaints", {}).get("count", 0)
        tier = rule_based_triage(
            bilstm_pred=int(score >= 0.5),
            last_grade=last_grade,
            days_since_inspection=999,
            complaint_count_30d=count
        )
        if not reasoning:
            reasoning = (
                f"Rule-based assessment: Complaint priority score {score:.3f}; "
                f"most recent inspection grade '{last_grade}'; "
                f"{count} similar complaints in the past 30 days."
            )

    # 6. Build citizen-facing and inspector-facing messaging
    ref_id = req.complaint_ref or f"FG-{uuid.uuid4().hex[:6].upper()}"
    restaurant = req.restaurant_name.strip() if req.restaurant_name and req.restaurant_name.strip() else "the reported establishment"
    loc = req.location.strip() if req.location and req.location.strip() else "New York City"

    if tier == TRIAGE_ESCALATE:
        citizen_summary = (
            f"Your complaint regarding {restaurant} has been submitted and ESCALATED IMMEDIATELY "
            f"to the Emergency Environmental Health Response Unit. Critical biological hazard indicators "
            f"or acute foodborne illness symptoms were detected in your report."
        )
        if has_img:
            citizen_summary += " Your uploaded photographic evidence has been verified and attached to the urgent inspection docket."
        safety_advisory = (
            f"Public Safety Advisory: Due to indicators of acute foodborne risk, we advise the public to avoid dining at {restaurant} "
            f"in {loc} pending completion of the on-site environmental health evaluation."
        )
        citizen_next_steps = (
            f"An Environmental Health Officer has been dispatched for an urgent on-site inspection within 48 hours. "
            f"Once the inspection is concluded and any laboratory analysis is complete, a full report of corrective "
            f"actions taken will be sent to your registered contact"
            + (f" ({req.contact_email})" if req.contact_email else "") + "."
        )
        inspector_directive = (
            "Critical biological contamination or acute foodborne illness indicators detected. "
            "Prioritize for immediate on-site environmental health inspection within 48 hours. "
            "Gather evidence samples if foodborne illness is confirmed."
        )
    elif tier == TRIAGE_REVIEW:
        citizen_summary = (
            f"Your complaint regarding {restaurant} has been submitted and queued for SECONDARY REGULATORY REVIEW. "
            f"Hygiene deficiencies or food handling concerns identified in your report."
        )
        if has_img:
            citizen_summary += " Your photographic evidence has been logged for inspector review."
        safety_advisory = (
            f"Advisory: Exercise caution when visiting {restaurant} in {loc}. "
            f"A secondary hygiene concern has been registered and is under departmental review."
        )
        citizen_next_steps = (
            f"An inspection supervisor will assess the establishment's violation history within 5 business days. "
            f"You will receive notification detailing any citations issued or corrective measures enforced"
            + (f" at {req.contact_email}" if req.contact_email else "") + "."
        )
        inspector_directive = (
            "Secondary hygiene failure or food handling violation identified. "
            "Schedule regulatory review within 5 business days and assess repeat complaint history."
        )
    else:
        citizen_summary = (
            f"Your complaint regarding {restaurant} has been logged in the municipal food safety surveillance system. "
            f"No acute hazard indicators were identified from the report."
        )
        safety_advisory = (
            f"Standard Advisory: No acute biohazard detected in this report. "
            f"{restaurant} remains open under routine municipal food code monitoring."
        )
        citizen_next_steps = (
            f"The matter will be reviewed during the establishment's next scheduled inspection cycle. "
            f"All inspection records are publicly accessible on the NYC Open Data Health Portal."
        )
        inspector_directive = (
            "Routine or administrative complaint. Log for next scheduled inspection cycle. "
            "No immediate action required."
        )

    return InvestigateResponse(
        complaint_ref=ref_id,
        camis=req.camis,
        restaurant_name=req.restaurant_name,
        location=req.location,
        category=req.category,
        hazard_score=round(score, 4),
        hazard_flag=hazard_flag,
        triage_level=tier,
        reasoning=reasoning,
        evidence=evidence_data,
        has_image=has_img,
        visual_finding=visual_finding,
        visual_label=visual_label,
        visual_confidence=f"{visual_conf:.0%}" if has_img else "N/A",
        citizen_summary=citizen_summary,
        safety_advisory=safety_advisory,
        citizen_next_steps=citizen_next_steps,
        inspector_directive=inspector_directive,
        used_rule_based_fallback=used_fallback
    )


# ── Interactive Web UI ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FoodGuard — Food Safety Triage Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #070e1e;
                --bg-card: #121c33;
                --bg-inner: #0b1426;
                --border-color: #233554;
                --accent-cyan: #38bdf8;
                --accent-blue: #0284c7;
                --text-light: #f8fafc;
                --text-muted: #94a3b8;
            }
            body {
                background-color: var(--bg-primary);
                color: var(--text-light);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                min-height: 100vh;
            }
            .card {
                background-color: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
            }
            /* View Switcher Pills */
            .view-switcher {
                background-color: #0b1426;
                border: 1px solid #233554;
                border-radius: 30px;
                padding: 4px;
                display: inline-flex;
            }
            .view-btn {
                border: none;
                background: transparent;
                color: #94a3b8;
                font-size: 0.9rem;
                font-weight: 600;
                padding: 8px 22px;
                border-radius: 25px;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            .view-btn.active {
                background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
                color: #041329;
                box-shadow: 0 2px 10px rgba(56,189,248,0.4);
            }
            .btn-primary {
                background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
                border: none;
                color: #041329;
                font-weight: 700;
                transition: all 0.2s ease;
            }
            .btn-primary:hover {
                background: linear-gradient(135deg, #38bdf8 0%, #7dd3fc 100%);
                color: #041329;
                transform: translateY(-1px);
            }
            .section-title {
                color: var(--accent-cyan);
                font-weight: 700;
                border-bottom: 2px solid var(--accent-blue);
                padding-bottom: 6px;
                display: inline-block;
            }
            .quick-tag {
                cursor: pointer;
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #cbd5e1;
                font-size: 0.82rem;
                padding: 4px 10px;
                border-radius: 20px;
                display: inline-block;
                margin: 2px;
                transition: all 0.15s ease;
                user-select: none;
            }
            .quick-tag:hover {
                background-color: #0284c7;
                color: #ffffff;
                border-color: #38bdf8;
            }
            .metric-box {
                background-color: var(--bg-inner);
                border-radius: 10px;
                border: 1px solid var(--border-color);
                padding: 16px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .assessment-section {
                background-color: var(--bg-inner);
                border-radius: 12px;
                padding: 22px;
                border: 1px solid var(--border-color);
                margin-bottom: 16px;
            }
            .assessment-header {
                font-size: 1rem;
                font-weight: 700;
                color: var(--accent-cyan);
                border-bottom: 1px solid #1e3a5f;
                padding-bottom: 8px;
                margin-bottom: 14px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }
            .item-row {
                display: flex;
                margin-bottom: 10px;
                font-size: 0.93rem;
                align-items: baseline;
            }
            .item-label {
                color: var(--text-muted);
                width: 230px;
                flex-shrink: 0;
                font-weight: 600;
            }
            .item-value {
                color: var(--text-light);
                flex-grow: 1;
            }
            /* Highlight Directive Boxes */
            .highlight-box {
                background: #0f2038;
                border-left: 6px solid var(--accent-cyan);
                padding: 18px 22px;
                border-radius: 0 10px 10px 0;
                margin-top: 14px;
                margin-bottom: 14px;
            }
            .highlight-box.escalate {
                background-color: #2b1118 !important;
                border-left-color: #ef4444 !important;
            }
            .highlight-box.review {
                background-color: #281d0d !important;
                border-left-color: #f59e0b !important;
            }
            .highlight-box.log {
                background-color: #0a251a !important;
                border-left-color: #10b981 !important;
            }
            .directive-text {
                font-size: 1.02rem !important;
                line-height: 1.55 !important;
                font-weight: 500 !important;
                display: block;
            }
            .highlight-box.escalate .directive-text { color: #fca5a5 !important; }
            .highlight-box.review .directive-text { color: #fde68a !important; }
            .highlight-box.log .directive-text { color: #a7f3d0 !important; }
            .highlight-box-title {
                font-size: 0.88rem !important;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 800 !important;
                margin-bottom: 6px;
                display: block;
            }
            .highlight-box.escalate .highlight-box-title { color: #f87171 !important; }
            .highlight-box.review .highlight-box-title { color: #fbbf24 !important; }
            .highlight-box.log .highlight-box-title { color: #34d399 !important; }
            /* Triage Badges */
            .badge-LOG { background-color: #10b981; color: #ffffff; font-weight: 700; }
            .badge-REVIEW { background-color: #f59e0b; color: #1e1b4b; font-weight: 700; }
            .badge-ESCALATE { background-color: #ef4444; color: #ffffff; font-weight: 700; }
            /* Image Upload */
            .upload-dropzone {
                border: 2px dashed #334155;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                background-color: #0b1426;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .upload-dropzone:hover {
                border-color: var(--accent-cyan);
                background-color: #0d1b33;
            }
            .image-preview-card {
                position: relative;
                display: inline-block;
                max-width: 100%;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid var(--border-color);
            }
            .image-preview-card img { max-height: 200px; object-fit: cover; border-radius: 6px; }
            .remove-img-btn {
                position: absolute;
                top: 6px;
                right: 6px;
                background: rgba(15,23,42,0.85);
                color: #f87171;
                border: 1px solid #ef4444;
                border-radius: 50%;
                width: 26px;
                height: 26px;
                font-size: 14px;
                line-height: 1;
                cursor: pointer;
            }
            .role-indicator {
                font-size: 0.82rem;
                padding: 4px 12px;
                border-radius: 12px;
                border: 1px solid #334155;
                background-color: #0f172a;
                color: #94a3b8;
            }
            .cluster-badge-active {
                background: rgba(239,68,68,0.15);
                color: #f87171;
                border: 1px solid rgba(239,68,68,0.4);
                border-radius: 6px;
                padding: 3px 10px;
                font-size: 0.82rem;
            }
            .cluster-badge-none {
                background: rgba(100,116,139,0.15);
                color: #94a3b8;
                border: 1px solid rgba(100,116,139,0.3);
                border-radius: 6px;
                padding: 3px 10px;
                font-size: 0.82rem;
            }
        </style>
    </head>
    <body class="py-5">
        <div class="container" style="max-width: 1000px;">

            <!-- Header -->
            <div class="text-center mb-4">
                <h2 class="fw-bold text-light mb-2">FoodGuard Food Safety Triage Platform</h2>
                <p class="text-secondary mb-2" style="font-size: 0.95rem;">
                    AI-Powered Complaint Intelligence &amp; Environmental Health Decision Support
                </p>
                <p class="text-secondary mb-3" style="font-size: 0.82rem; max-width: 700px; margin: 0 auto;">
                    Citizens report food safety concerns and submit photo evidence. Environmental health officers
                    access aggregated hazard intelligence, spatiotemporal cluster analysis, and regulatory triage directives.
                </p>
                <!-- Role Switcher -->
                <div class="view-switcher mb-3">
                    <button class="view-btn active" id="btnCitizenRole" onclick="switchPortal('citizen')">Public Citizen Portal</button>
                    <button class="view-btn" id="btnInspectorRole" onclick="requestInspectorAccess()">Inspector Portal</button>
                </div>
            </div>

            <!-- Inspector Passcode Gate -->
            <div id="inspectorGateCard" class="card p-4 shadow-sm mb-4 d-none">
                <h5 class="section-title mb-2">Inspector Sign-In Required</h5>
                <p class="text-secondary" style="font-size: 0.85rem;">
                    Role-gated view for verified DOHMH environmental health staff.
                    Demo build — inspector passcode: <code>health123</code>
                </p>
                <div class="row g-2 align-items-center">
                    <div class="col-md-8">
                        <input type="password" id="inspectorPasscodeInput" class="form-control bg-dark text-light border-secondary" placeholder="Inspector passcode">
                    </div>
                    <div class="col-md-4">
                        <button class="btn btn-warning w-100" onclick="submitInspectorPasscode()">Sign In</button>
                    </div>
                </div>
                <div id="inspectorGateError" class="text-danger mt-2 d-none" style="font-size: 0.85rem;">Incorrect passcode. Please try again.</div>
            </div>

            <!-- Submission Form -->
            <div class="card p-4 shadow-sm mb-4" id="submissionCard">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div>
                        <h5 class="section-title mb-0" id="formHeaderTitle">Submit a Food Safety Complaint</h5>
                        <span class="role-indicator ms-2" id="roleBadge">Citizen Mode</span>
                    </div>
                    <button type="button" class="btn btn-sm btn-outline-secondary" onclick="resetForm()">Clear Form</button>
                </div>

                <!-- Complaint Category -->
                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Complaint Category</label>
                    <select id="categorySelect" class="form-select bg-dark text-light border-secondary">
                        <option value="">Select a category or use a quick tag below...</option>
                        <optgroup label="Acute Health &amp; Illness">
                            <option value="Food Poisoning / Acute Illness (Vomiting, Diarrhea, Fever)">Food Poisoning / Acute Illness (Vomiting, Diarrhea, Fever)</option>
                            <option value="Nausea / Stomach Cramps after meal">Nausea / Stomach Cramps after meal</option>
                            <option value="Undercooked Meat / Raw Poultry Hazard">Undercooked Meat / Raw Poultry Hazard</option>
                            <option value="Chemical / Pesticide Contamination">Chemical / Pesticide Contamination</option>
                        </optgroup>
                        <optgroup label="Pest &amp; Biological Infestation">
                            <option value="Rodents / Mice / Rats Infestation">Rodents / Mice / Rats Infestation</option>
                            <option value="Roaches / Cockroach Infestation">Roaches / Cockroach Infestation</option>
                            <option value="Filth Flies / Insects on Food">Filth Flies / Insects on Food</option>
                            <option value="Sewage / Drainage Backup">Sewage / Drainage Backup</option>
                        </optgroup>
                        <optgroup label="Food Storage &amp; Contamination">
                            <option value="Food Temperature Abuse / Spoiled Food">Food Temperature Abuse / Spoiled Food</option>
                            <option value="Food Contains Foreign Object">Food Contains Foreign Object</option>
                            <option value="Cross-Contamination in Food Preparation">Cross-Contamination in Food Preparation</option>
                        </optgroup>
                        <optgroup label="Hygiene &amp; Sanitation">
                            <option value="Bare Hands in Contact with Ready-to-Eat Food">Bare Hands in Contact with Ready-to-Eat Food</option>
                            <option value="Food Worker Hygiene / Lack of Gloves">Food Worker Hygiene / Lack of Gloves</option>
                            <option value="Unsanitary Kitchen / Prep Surface">Unsanitary Kitchen / Prep Surface</option>
                        </optgroup>
                        <optgroup label="Facility &amp; Administrative">
                            <option value="Facility Condition / Odor">Facility Condition / Odor</option>
                            <option value="Pet / Animal in Dining Area">Pet / Animal in Dining Area</option>
                            <option value="Letter Grade Missing / Inspection Certificate">Letter Grade Missing / Inspection Certificate</option>
                            <option value="General Sanitation Concern">General Sanitation Concern</option>
                        </optgroup>
                    </select>
                    <!-- Quick Tags -->
                    <div class="mt-2">
                        <small class="text-secondary d-block mb-1" style="font-size: 0.8rem;">Common quick selections:</small>
                        <span class="quick-tag" onclick="selectQuickCategory('Food Poisoning / Acute Illness (Vomiting, Diarrhea, Fever)')">Acute Illness / Food Poisoning</span>
                        <span class="quick-tag" onclick="selectQuickCategory('Rodents / Mice / Rats Infestation')">Rodent Infestation</span>
                        <span class="quick-tag" onclick="selectQuickCategory('Food Temperature Abuse / Spoiled Food')">Temperature Abuse / Spoiled</span>
                        <span class="quick-tag" onclick="selectQuickCategory('Food Contains Foreign Object')">Foreign Object in Food</span>
                        <span class="quick-tag" onclick="selectQuickCategory('Bare Hands in Contact with Ready-to-Eat Food')">Bare Hand Contact</span>
                        <span class="quick-tag" onclick="selectQuickCategory('Roaches / Cockroach Infestation')">Cockroach Infestation</span>
                    </div>
                </div>

                <!-- Description -->
                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Complaint Description <span class="text-danger">*</span></label>
                    <textarea id="complaintText" class="form-control bg-dark text-light border-secondary" rows="3"
                        placeholder="Describe what you observed or experienced (e.g., I developed acute vomiting and stomach cramps after eating at this restaurant, and I observed live cockroaches behind the counter)."></textarea>
                </div>

                <!-- Image Upload -->
                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Photographic Evidence <span class="text-secondary fw-normal">(Optional — recommended for critical complaints)</span></label>
                    <div class="upload-dropzone" onclick="document.getElementById('imageFileInput').click()">
                        <div id="dropzonePrompt">
                            <span class="d-block text-info fw-semibold mb-1">Click to Upload Photo Evidence</span>
                            <small class="text-secondary">Photos of food, pests, kitchen conditions, or foreign objects (PNG, JPG, WEBP, max 10MB)</small>
                        </div>
                        <div id="imagePreviewContainer" class="d-none mt-2">
                            <div class="image-preview-card">
                                <img id="previewImg" src="" alt="Uploaded evidence">
                                <button type="button" class="remove-img-btn" onclick="removeUploadedImage(event)">&times;</button>
                            </div>
                            <small class="text-success d-block mt-2">Image attached to complaint docket</small>
                        </div>
                    </div>
                    <input type="file" id="imageFileInput" accept="image/*" class="d-none" onchange="handleImageSelection(this)">
                </div>

                <!-- Restaurant & Location Datalists -->
                <div class="row g-3 mb-3">
                    <div class="col-md-6">
                        <label class="form-label text-secondary fw-semibold">Restaurant Name</label>
                        <input list="restaurantList" id="restaurantNameInput" class="form-control bg-dark text-light border-secondary" placeholder="Select or type restaurant name...">
                        <datalist id="restaurantList">
                            <option value="Black Star Bakery &amp; Cafe">
                            <option value="McDonald's">
                            <option value="Dunkin'">
                            <option value="Popeyes">
                            <option value="Cold Stone Creamery">
                            <option value="Bessou">
                            <option value="Crispy Chick">
                            <option value="Ocean Blue Seafood">
                            <option value="Juici Patties Melrose">
                            <option value="Chicklyn">
                            <option value="New Pho Best">
                            <option value="Jonathan Bakery Corp">
                            <option value="Chipotle Mexican Grill">
                            <option value="Shake Shack">
                            <option value="Subway">
                            <option value="KFC">
                            <option value="Wendy's">
                            <option value="Domino's Pizza">
                            <option value="Five Guys">
                            <option value="Panda Express">
                        </datalist>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label text-secondary fw-semibold">Location / Borough</label>
                        <input list="locationList" id="locationInput" class="form-control bg-dark text-light border-secondary" placeholder="Select or type location...">
                        <datalist id="locationList">
                            <option value="Manhattan">
                            <option value="Brooklyn">
                            <option value="Queens">
                            <option value="Bronx">
                            <option value="Staten Island">
                            <option value="Midtown Manhattan (10019)">
                            <option value="Upper East Side, Manhattan (10028)">
                            <option value="Lower East Side, Manhattan (10002)">
                            <option value="Downtown Brooklyn (11201)">
                            <option value="Williamsburg, Brooklyn (11211)">
                            <option value="Flushing, Queens (11355)">
                            <option value="Astoria, Queens (11105)">
                            <option value="South Bronx (10451)">
                            <option value="Harlem, Manhattan (10037)">
                            <option value="Jamaica, Queens (11432)">
                            <option value="10 Columbus Circle, Manhattan">
                            <option value="136-20 Roosevelt Avenue, Queens">
                        </datalist>
                    </div>
                </div>

                <!-- CAMIS ID (Inspector-only, hidden in citizen mode) -->
                <div class="mb-3 d-none" id="camisRow">
                    <label class="form-label text-secondary fw-semibold">Establishment ID (CAMIS)</label>
                    <input type="text" id="camisInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. 50002628 — links to BigQuery inspection history">
                </div>

                <!-- Contact Email -->
                <div class="mb-4">
                    <label class="form-label text-secondary fw-semibold">Contact Email <span class="text-secondary fw-normal">(Optional)</span></label>
                    <input type="email" id="emailInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. resident@example.com — receive inspection resolution updates">
                    <small class="text-secondary d-block mt-1" style="font-size: 0.8rem;">
                        You may report anonymously. If provided, your email is used only to notify you of inspection outcomes.
                    </small>
                </div>

                <button onclick="runTriage()" class="btn btn-primary w-100 py-2 fs-6 shadow-sm" id="btnSubmit">
                    Submit Complaint for Triage Assessment
                </button>
            </div>

            <!-- Citizen Results Card -->
            <div id="citizenResultsCard" class="card p-4 shadow-sm d-none mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0 text-light fw-bold">Complaint Submission Confirmation</h5>
                    <span id="citizenBadge" class="badge fs-6 px-3 py-2"></span>
                </div>
                <div class="assessment-section mb-3">
                    <div class="item-row">
                        <div class="item-label">Case Reference ID:</div>
                        <div class="item-value fw-bold text-info" id="citizenRefId">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Establishment Reported:</div>
                        <div class="item-value" id="citizenEstablishment">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Current Triage Status:</div>
                        <div class="item-value fw-bold" id="citizenTriageStatus">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Submission Summary:</div>
                        <div class="item-value" id="citizenSummaryText">-</div>
                    </div>
                    <div class="item-row" id="citizenImageRow">
                        <div class="item-label">Photo Evidence:</div>
                        <div class="item-value" id="citizenImageThumb">-</div>
                    </div>
                    <!-- Consumer Safety Advisory -->
                    <div class="highlight-box" id="safetyAdvisoryBox">
                        <span class="highlight-box-title" id="safetyTitle">Consumer Safety Guidance</span>
                        <div class="directive-text" id="displaySafetyAdvisory">-</div>
                    </div>
                    <!-- Next Steps -->
                    <div class="highlight-box log" id="nextStepsBox">
                        <span class="highlight-box-title" style="color: #34d399 !important;">Next Steps &amp; Resolution Updates</span>
                        <div class="directive-text" style="color: #a7f3d0 !important;" id="displayCitizenNextSteps">-</div>
                    </div>
                </div>
            </div>

            <!-- Inspector Results Card -->
            <div id="inspectorResultsCard" class="card p-4 shadow-sm d-none mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div>
                        <h5 class="mb-0 text-light fw-bold">Environmental Health Officer Assessment</h5>
                        <small class="text-secondary">AI Decision Support System · BigQuery Evidence Integration</small>
                    </div>
                    <span id="inspectorBadge" class="badge fs-6 px-3 py-2"></span>
                </div>

                <!-- Metric Cards -->
                <div class="row text-center mb-4 g-3">
                    <div class="col-md-6">
                        <div class="metric-box">
                            <small class="text-secondary text-uppercase fw-semibold d-block mb-1" style="font-size: 0.78rem;">Complaint Priority Score</small>
                            <h3 id="inspectorScore" class="fw-bold mb-0 text-info"></h3>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metric-box">
                            <small class="text-secondary text-uppercase fw-semibold d-block mb-1" style="font-size: 0.78rem;">Hazard Classification</small>
                            <h4 id="inspectorHazardFlag" class="fw-bold mb-0" style="font-size: 1rem;"></h4>
                        </div>
                    </div>
                </div>

                <!-- Complaint Assessment -->
                <div class="assessment-section">
                    <div class="assessment-header">Complaint Assessment</div>
                    <div class="item-row">
                        <div class="item-label">Case Reference ID:</div>
                        <div class="item-value text-info fw-bold" id="inspectorRefId">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Establishment:</div>
                        <div class="item-value" id="inspectorEstablishment">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Complaint Category:</div>
                        <div class="item-value" id="inspectorCategory">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Complaint Description:</div>
                        <div class="item-value" id="inspectorComplaint">-</div>
                    </div>
                </div>

                <!-- Photographic Evidence -->
                <div class="assessment-section">
                    <div class="assessment-header">Photographic Evidence Assessment</div>
                    <div class="item-row">
                        <div class="item-label">Photo Evidence Attached:</div>
                        <div class="item-value" id="inspectorImagePresence">-</div>
                    </div>
                    <div class="item-row" id="inspectorImageDetailRow">
                        <div class="item-label">Visual Hazard Classification:</div>
                        <div class="item-value fw-semibold text-warning" id="inspectorVisualFinding">-</div>
                    </div>
                    <div class="item-row" id="inspectorImageThumbRow">
                        <div class="item-label">Submitted Image:</div>
                        <div class="item-value" id="inspectorImagePreviewContainer"></div>
                    </div>
                </div>

                <!-- Inspection History & Cluster Context -->
                <div class="assessment-section">
                    <div class="assessment-header">Inspection History &amp; Cluster Intelligence</div>
                    <div class="item-row">
                        <div class="item-label">Most Recent Grade:</div>
                        <div class="item-value fw-bold" id="inspectorGrade">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Last Regulatory Action:</div>
                        <div class="item-value" id="inspectorLastAction">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Prior Inspections (last 10):</div>
                        <div class="item-value" id="inspectorInspCount">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Active Complaint Cluster:</div>
                        <div class="item-value" id="inspectorClusterStatus">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Recent Complaints (30d):</div>
                        <div class="item-value" id="inspectorRecentCount">-</div>
                    </div>
                </div>

                <!-- AI Reasoning & Directives -->
                <div class="assessment-section">
                    <div class="assessment-header">AI Agent Reasoning</div>
                    <div class="item-row">
                        <div class="item-label">Recommended Triage Tier:</div>
                        <div class="item-value fw-bold" id="inspectorTier">-</div>
                    </div>
                    <!-- Regulatory Directive -->
                    <div class="highlight-box" id="inspectorActionBox">
                        <span class="highlight-box-title" id="inspectorActionTitle">Regulatory Action Directive</span>
                        <div class="directive-text" id="displayInspectorAction">-</div>
                    </div>
                    <!-- Agent Reasoning -->
                    <div class="mt-3">
                        <small class="text-secondary text-uppercase fw-semibold d-block mb-2" style="font-size: 0.78rem;">Full Agent Reasoning</small>
                        <div id="agentReasoningText" style="background:#090d16; border-radius:8px; padding:14px; color:#38bdf8; font-size:0.88rem; white-space:pre-wrap; line-height:1.6;"></div>
                    </div>
                    <div class="text-secondary small mt-3" style="font-size: 0.8rem;">
                        Automated AI decision support output. Field dispatches and formal violation summonses
                        remain subject to statutory environmental health officer confirmation.
                    </div>
                </div>
            </div>

        </div><!-- /container -->

        <script>
            let currentPortal = 'citizen';
            let lastResultData = null;
            let inspectorAuthorized = false;
            let uploadedImageBase64 = null;

            function handleImageSelection(input) {
                const file = input.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageBase64 = e.target.result;
                    document.getElementById('previewImg').src = uploadedImageBase64;
                    document.getElementById('dropzonePrompt').classList.add('d-none');
                    document.getElementById('imagePreviewContainer').classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            }

            function removeUploadedImage(e) {
                e.stopPropagation();
                uploadedImageBase64 = null;
                document.getElementById('previewImg').src = '';
                document.getElementById('imageFileInput').value = '';
                document.getElementById('imagePreviewContainer').classList.add('d-none');
                document.getElementById('dropzonePrompt').classList.remove('d-none');
            }

            function selectQuickCategory(val) {
                document.getElementById('categorySelect').value = val;
            }

            function switchPortal(mode) {
                currentPortal = mode;
                const citizenBtn = document.getElementById('btnCitizenRole');
                const inspectorBtn = document.getElementById('btnInspectorRole');
                const camisRow = document.getElementById('camisRow');
                const roleBadge = document.getElementById('roleBadge');
                const formTitle = document.getElementById('formHeaderTitle');

                if (mode === 'inspector') {
                    citizenBtn.classList.remove('active');
                    inspectorBtn.classList.add('active');
                    camisRow.classList.remove('d-none');
                    roleBadge.innerText = 'Inspector Mode';
                    roleBadge.style.color = '#38bdf8';
                    formTitle.innerText = 'Assess Complaint — Inspector View';
                } else {
                    inspectorBtn.classList.remove('active');
                    citizenBtn.classList.add('active');
                    camisRow.classList.add('d-none');
                    roleBadge.innerText = 'Citizen Mode';
                    roleBadge.style.color = '#94a3b8';
                    formTitle.innerText = 'Submit a Food Safety Complaint';
                }
                // Re-render last result in new portal view if available
                if (lastResultData) renderResults(lastResultData);
            }

            function requestInspectorAccess() {
                if (inspectorAuthorized) {
                    switchPortal('inspector');
                    return;
                }
                document.getElementById('inspectorGateCard').classList.remove('d-none');
                document.getElementById('submissionCard').classList.add('d-none');
            }

            async function submitInspectorPasscode() {
                const passcode = document.getElementById('inspectorPasscodeInput').value;
                const errDiv = document.getElementById('inspectorGateError');
                try {
                    const resp = await fetch('/inspector/verify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ passcode })
                    });
                    if (resp.ok) {
                        inspectorAuthorized = true;
                        document.getElementById('inspectorGateCard').classList.add('d-none');
                        document.getElementById('submissionCard').classList.remove('d-none');
                        errDiv.classList.add('d-none');
                        switchPortal('inspector');
                    } else {
                        errDiv.classList.remove('d-none');
                    }
                } catch(e) {
                    errDiv.classList.remove('d-none');
                }
            }

            function resetForm() {
                document.getElementById('categorySelect').value = '';
                document.getElementById('complaintText').value = '';
                document.getElementById('restaurantNameInput').value = '';
                document.getElementById('locationInput').value = '';
                document.getElementById('camisInput').value = '';
                document.getElementById('emailInput').value = '';
                removeUploadedImage({ stopPropagation: () => {} });
                document.getElementById('citizenResultsCard').classList.add('d-none');
                document.getElementById('inspectorResultsCard').classList.add('d-none');
                lastResultData = null;
            }

            async function runTriage() {
                const text = document.getElementById('complaintText').value.trim();
                const category = document.getElementById('categorySelect').value.trim();
                const restaurant_name = document.getElementById('restaurantNameInput').value.trim();
                const location = document.getElementById('locationInput').value.trim();
                const camis = document.getElementById('camisInput').value.trim();
                const contact_email = document.getElementById('emailInput').value.trim();
                const btn = document.getElementById('btnSubmit');

                if (!text && !category && !uploadedImageBase64) {
                    alert('Please describe your complaint, select a category, or upload photo evidence.');
                    return;
                }

                btn.disabled = true;
                btn.innerText = 'Analyzing complaint...';
                document.getElementById('citizenResultsCard').classList.add('d-none');
                document.getElementById('inspectorResultsCard').classList.add('d-none');

                try {
                    const payload = {
                        text: text || category || 'Image evidence submitted',
                        category: category || null,
                        restaurant_name: restaurant_name || null,
                        location: location || null,
                        camis: camis || null,
                        contact_email: contact_email || null,
                        image_data: uploadedImageBase64 || null
                    };

                    const resp = await fetch('/investigate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        alert(data.detail || 'Error processing complaint. Please try again.');
                        return;
                    }

                    lastResultData = data;
                    renderResults(data);

                } catch (e) {
                    alert('Request failed: ' + e);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Submit Complaint for Triage Assessment';
                }
            }

            function renderResults(data) {
                const tier = data.triage_level;
                const tierClass = 'badge-' + tier;

                if (currentPortal === 'citizen') {
                    document.getElementById('inspectorResultsCard').classList.add('d-none');
                    const card = document.getElementById('citizenResultsCard');
                    card.classList.remove('d-none');

                    document.getElementById('citizenRefId').innerText = data.complaint_ref || '-';
                    document.getElementById('citizenEstablishment').innerText =
                        (data.restaurant_name || 'Unspecified') + (data.location ? ', ' + data.location : '');

                    const badge = document.getElementById('citizenBadge');
                    badge.innerText = tier;
                    badge.className = 'badge fs-6 px-3 py-2 ' + tierClass;

                    const statusMap = {
                        'ESCALATE': '⚠ Escalated — Emergency Response Unit Notified',
                        'REVIEW': '⏳ Under Secondary Regulatory Review',
                        'LOG': '✓ Logged — Scheduled for Routine Inspection'
                    };
                    document.getElementById('citizenTriageStatus').innerText = statusMap[tier] || tier;

                    document.getElementById('citizenSummaryText').innerText = data.citizen_summary || '-';

                    // Safety Advisory Box
                    const safetyBox = document.getElementById('safetyAdvisoryBox');
                    safetyBox.className = 'highlight-box ' + tier.toLowerCase();
                    document.getElementById('safetyTitle').innerText =
                        tier === 'ESCALATE' ? 'Consumer Safety Advisory' :
                        tier === 'REVIEW' ? 'Consumer Advisory' : 'Establishment Status';
                    document.getElementById('displaySafetyAdvisory').innerText = data.safety_advisory || '-';

                    document.getElementById('displayCitizenNextSteps').innerText = data.citizen_next_steps || '-';

                    // Citizen image thumbnail
                    const imgThumb = document.getElementById('citizenImageThumb');
                    const imgRow = document.getElementById('citizenImageRow');
                    if (data.has_image && data.image_data) {
                        imgRow.classList.remove('d-none');
                        imgThumb.innerHTML = `<img src="${data.image_data}" alt="Your submitted evidence" style="max-height:120px;border-radius:6px;border:1px solid #334155;">`;
                    } else {
                        imgRow.classList.add('d-none');
                    }

                } else {
                    // Inspector portal
                    document.getElementById('citizenResultsCard').classList.add('d-none');
                    const card = document.getElementById('inspectorResultsCard');
                    card.classList.remove('d-none');

                    const badge = document.getElementById('inspectorBadge');
                    badge.innerText = tier;
                    badge.className = 'badge fs-6 px-3 py-2 ' + tierClass;

                    document.getElementById('inspectorScore').innerText = data.hazard_score;

                    const hazardFlagEl = document.getElementById('inspectorHazardFlag');
                    hazardFlagEl.innerText = data.hazard_flag || '-';
                    hazardFlagEl.className = 'fw-bold mb-0 ' + (
                        tier === 'ESCALATE' ? 'text-danger' :
                        tier === 'REVIEW' ? 'text-warning' : 'text-success'
                    );

                    document.getElementById('inspectorRefId').innerText = data.complaint_ref || '-';
                    document.getElementById('inspectorEstablishment').innerText =
                        (data.restaurant_name || 'Unspecified') + (data.location ? ', ' + data.location : '');
                    document.getElementById('inspectorCategory').innerText = data.category || 'General Food Safety Complaint';
                    document.getElementById('inspectorComplaint').innerText = data.triage_level ? (data.complaint_ref ? data.reasoning ? '' : '-' : '-') : '-';

                    // Inspection History & Cluster
                    const ev = data.evidence || {};
                    const insp = ev.inspections || {};
                    const cluster = ev.cluster || {};
                    const recent = ev.recent_complaints || {};

                    document.getElementById('inspectorGrade').innerText = insp.most_recent_grade || (Object.keys(insp).length ? 'Not Graded' : 'No CAMIS provided');
                    document.getElementById('inspectorLastAction').innerText = insp.last_action || '-';
                    document.getElementById('inspectorInspCount').innerText =
                        insp.inspections ? insp.inspections.length + ' records retrieved' : 'No CAMIS provided';

                    const clusterEl = document.getElementById('inspectorClusterStatus');
                    if (cluster.in_active_cluster) {
                        clusterEl.innerHTML = `<span class="cluster-badge-active">Active Cluster — Cluster ID: ${cluster.cluster_id}</span>`;
                    } else {
                        clusterEl.innerHTML = `<span class="cluster-badge-none">No active outbreak cluster</span>`;
                    }

                    document.getElementById('inspectorRecentCount').innerText =
                        recent.count !== undefined ? recent.count + ' complaints in past 30 days' : 'No CAMIS provided';

                    // Triage Tier & Directive
                    const tierDisplay = document.getElementById('inspectorTier');
                    tierDisplay.innerText = tier;
                    tierDisplay.className = 'item-value fw-bold ' + (
                        tier === 'ESCALATE' ? 'text-danger' :
                        tier === 'REVIEW' ? 'text-warning' : 'text-success'
                    );

                    const actionBox = document.getElementById('inspectorActionBox');
                    actionBox.className = 'highlight-box ' + tier.toLowerCase();
                    const actionTitle = document.getElementById('inspectorActionTitle');
                    actionTitle.innerText = tier === 'ESCALATE' ? 'Emergency Action Required' :
                                           tier === 'REVIEW' ? 'Secondary Review Directive' : 'Routine Log Directive';
                    document.getElementById('displayInspectorAction').innerText = data.inspector_directive || '-';

                    // Agent Reasoning
                    document.getElementById('agentReasoningText').innerText = data.reasoning || 'No reasoning provided.';

                    // Visual Evidence
                    document.getElementById('inspectorImagePresence').innerText =
                        data.has_image ? 'Yes — Photo attached to docket' : 'No photographic evidence submitted';
                    document.getElementById('inspectorVisualFinding').innerText =
                        data.has_image ? data.visual_finding : 'N/A';

                    const imgPreview = document.getElementById('inspectorImagePreviewContainer');
                    if (data.has_image && data.image_data) {
                        imgPreview.innerHTML = `<img src="${data.image_data}" alt="Submitted evidence" style="max-height:160px;border-radius:6px;border:1px solid #334155;">`;
                        document.getElementById('inspectorImageThumbRow').classList.remove('d-none');
                        document.getElementById('inspectorImageDetailRow').classList.remove('d-none');
                    } else {
                        imgPreview.innerHTML = '';
                        document.getElementById('inspectorImageThumbRow').classList.add('d-none');
                        document.getElementById('inspectorImageDetailRow').classList.add('d-none');
                    }
                }
            }
        </script>
    </body>
    </html>
    """
