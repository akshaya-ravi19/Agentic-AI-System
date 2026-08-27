"""
pipeline/agent/service.py
-----------------------------
Deployable Cloud Run service wrapping the LangGraph ReAct agent
prototyped in notebook 05. Wires together:
  1. classifier-service (Cloud Run, pipeline/classifier_service) for
     the severity score
  2. this module's own BigQuery tools (inspection history, recent
     complaints, cluster context) for evidence
  3. output-router service (pipeline/routing) to log the decision
     and handle Tier 3 escalation

WHY THREE SEPARATE SERVICES INSTEAD OF ONE BIG ONE:
Matches the architecture diagram's own stage separation (Stage 2 /
Stage 3 / Stage 5 are drawn as distinct boxes). Each can scale, fail,
and redeploy independently -- e.g. a classifier model update doesn't
require redeploying the agent or routing logic, and a routing bug
doesn't take down classification.
"""
import os
import sys
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import GEMINI_MODEL, TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE
from pipeline.agent.tools import get_inspection_history, get_recent_complaints, get_cluster_context
from pipeline.routing.rule_based_triage import rule_based_triage

CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "").rstrip("/")
ROUTER_URL = os.environ.get("ROUTER_URL", "").rstrip("/")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # from Secret Manager at deploy time

SYSTEM_PROMPT = """You are a food-safety triage assistant supporting DOHMH
inspectors -- you RECOMMEND, you do not make final regulatory decisions.
Given a complaint's severity score, retrieve inspection history and
cluster context using your tools, then assign a triage_level of
LOG, REVIEW, or ESCALATE, with your reasoning tied to specific evidence
you retrieved (not general assumptions). Always call at least
get_inspection_history before recommending REVIEW or ESCALATE."""


@tool
def inspection_history_tool(camis: str) -> dict:
    """Get this restaurant's DOHMH inspection history."""
    return get_inspection_history(camis)


@tool
def recent_complaints_tool(camis: str) -> dict:
    """Get recent 311 complaints against this restaurant."""
    return get_recent_complaints(camis)


@tool
def cluster_context_tool(camis: str) -> dict:
    """Check whether this restaurant is part of an active outbreak cluster."""
    return get_cluster_context(camis)


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        tools = [inspection_history_tool, recent_complaints_tool, cluster_context_tool]
        # prompt= (not the deprecated state_modifier=) -- see CHANGES_APPLIED.md
        _agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return _agent


def classify_severity(text: str) -> dict:
    if not CLASSIFIER_URL:
        raise RuntimeError("CLASSIFIER_URL env var not set")
    resp = requests.post(f"{CLASSIFIER_URL}/predict", json={"text": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_to_router(decision: dict) -> dict:
    if not ROUTER_URL:
        raise RuntimeError("ROUTER_URL env var not set")
    resp = requests.post(f"{ROUTER_URL}/route", json=decision, timeout=30)
    resp.raise_for_status()
    return resp.json()


app = FastAPI(title="foodguard-agent")


class ComplaintIn(BaseModel):
    complaint_id: str
    camis: str | None = None
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/investigate")
def investigate(complaint: ComplaintIn):
    """
    Full pipeline for one complaint: classify -> agent investigates
    with tools -> route the resulting decision. Returns what was
    decided; the actual logging/escalation side-effect happens via
    the router service call, not here.
    """
    if not complaint.camis:
        raise HTTPException(status_code=400, detail="camis is required for triage")

    clf_result = classify_severity(complaint.text)

    agent = get_agent()
    user_msg = (
        f"Complaint (camis={complaint.camis}): {complaint.text}\n"
        f"Classifier severity_score={clf_result['severity_score']:.3f} "
        f"(label={clf_result['severity_label']}). "
        f"Investigate and recommend a triage_level (LOG/REVIEW/ESCALATE)."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    final_message = result["messages"][-1].content

    # Best-effort parse of the agent's tier decision out of its final
    # message; if that fails for any reason, fall back to the
    # deterministic rule-based baseline rather than silently dropping
    # the complaint -- a fallback matters here specifically because
    # this is meant to support inspectors, not go silent on them.
    tier = None
    for candidate in (TRIAGE_ESCALATE, TRIAGE_REVIEW, TRIAGE_LOG):
        if candidate in final_message.upper():
            tier = candidate
            break
    used_fallback = tier is None
    if used_fallback:
        history = get_inspection_history(complaint.camis)
        last_grade = history.get("most_recent_grade", "Not Yet Graded")
        recent = get_recent_complaints(complaint.camis, days=30)
        tier = rule_based_triage(
            bilstm_pred=clf_result["severity_label"],
            last_grade=last_grade,
            days_since_inspection=999,  # unknown here -- rule treats this conservatively
            complaint_count_30d=recent["count"],
        )

    decision = {
        "complaint_id": complaint.complaint_id,
        "camis": complaint.camis,
        "triage_level": tier,
        "severity_score": clf_result["severity_score"],
        "supporting_evidence": [final_message],
        "tools_called": ["inspection_history_tool", "recent_complaints_tool", "cluster_context_tool"],
        "recommended_action": final_message,
        "used_rule_based_fallback": used_fallback,
    }
    router_result = send_to_router(decision)

    return {"decision": decision, "router_result": router_result}
