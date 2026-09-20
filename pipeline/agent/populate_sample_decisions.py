"""
pipeline/agent/populate_sample_decisions.py
------------------------------------------------
Runs the real classifier + agent (or the rule-based fallback, if no
GOOGLE_API_KEY is set) against a sample of real complaints from
BigQuery, and writes genuine decisions to agent_decisions -- and any
resulting ESCALATE tier to escalations -- so the live dashboard has
real content to display, rather than requiring the full agent/router
Cloud Run services to be deployed and receiving live traffic first.

WHY THIS BYPASSES HTTP ENTIRELY (unlike pipeline/agent/service.py):
service.py calls the classifier over CLASSIFIER_URL and the router
over ROUTER_URL, which only work once those Cloud Run services are
deployed and reachable. This script instead loads the XGBoost model
and calls the agent directly in-process, and writes straight to
BigQuery -- useful for generating real sample data locally without
standing up the full microservice chain first.

HOW TO RUN:
    export GOOGLE_API_KEY=your_key_here   # optional -- omit to use the rule-based fallback only
    python pipeline/agent/populate_sample_decisions.py --n 15
"""
import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import (GCP_PROJECT, BQ_COMPLAINTS_LABELLED, MODELS_DIR, DATA_LABELLED,
                            DATA_RAW, EMBEDDING_MODEL, BQ_AGENT_DECISIONS, BQ_ESCALATIONS,
                            TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE)
from pipeline.agent.tools import get_inspection_history, get_recent_complaints, get_cluster_context
from pipeline.routing.rule_based_triage import rule_based_triage


def load_classifier():
    """Same pattern as demo_app.py: load the XGBoost model and
    reproduce its training-time scaler exactly, since the scaler
    itself was never saved to disk (see demo_app.py's docstring for
    why this recomputation is safe and deterministic)."""
    import xgboost as xgb
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import MinMaxScaler

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost" / "xgb_hybrid_final.json"))

    df = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv", parse_dates=["created_date"])
    df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"],
                            dtype={"camis": str}, low_memory=False)
    df["matched_camis"] = df["matched_camis"].astype(str)
    df = df.dropna(subset=["descriptor", "label"]).reset_index(drop=True)

    def prior_violation_count(row):
        hist = df_dohmh[(df_dohmh["camis"] == row["matched_camis"]) &
                         (df_dohmh["inspection_date"] < row["created_date"])]
        return len(hist)

    def days_since_last_inspection(row):
        hist = df_dohmh[(df_dohmh["camis"] == row["matched_camis"]) &
                         (df_dohmh["inspection_date"] < row["created_date"])]
        if hist.empty:
            return np.nan
        return (row["created_date"] - hist["inspection_date"].max()).days

    df["prior_violation_count"] = df.apply(prior_violation_count, axis=1)
    df["days_since_last_inspection"] = df.apply(days_since_last_inspection, axis=1)
    median_days = df["days_since_last_inspection"].median()
    df["days_since_last_inspection"] = df["days_since_last_inspection"].fillna(median_days)
    scaler = MinMaxScaler().fit(df[["prior_violation_count", "days_since_last_inspection"]].values)

    return embedder, model, scaler, df_dohmh, median_days


def classify(text, camis, embedder, model, scaler, df_dohmh, median_days):
    embedding = embedder.encode([text])
    as_of = pd.Timestamp.now()
    hist = df_dohmh[(df_dohmh["camis"] == str(camis).strip()) & (df_dohmh["inspection_date"] < as_of)]
    prior_count = len(hist) if not hist.empty else 0
    days_since = (as_of - hist["inspection_date"].max()).days if not hist.empty else median_days
    numeric_scaled = scaler.transform([[prior_count, days_since]])
    X = np.hstack([embedding, numeric_scaled])
    score = float(model.predict_proba(X)[0, 1])
    return score, int(score >= 0.5)


def get_agent_if_available():
    """Returns a LangGraph agent if GOOGLE_API_KEY is set, else None
    (caller falls back to the rule-based triage function)."""
    import os
    if not os.environ.get("GOOGLE_API_KEY"):
        print("No GOOGLE_API_KEY set -- using the rule-based fallback for every complaint, "
              "not the LLM agent. Set GOOGLE_API_KEY to use the real agent instead.")
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool
        from config.config import GEMINI_MODEL

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

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        system_prompt = ("You are a food-safety triage assistant supporting DOHMH inspectors -- "
                          "you RECOMMEND, you do not make final regulatory decisions. Assign a "
                          "triage_level of LOG, REVIEW, or ESCALATE with reasoning tied to evidence.")
        return create_react_agent(llm, [inspection_history_tool, recent_complaints_tool, cluster_context_tool],
                                   prompt=system_prompt)
    except Exception as e:
        print(f"Could not initialize the LLM agent ({e}) -- using the rule-based fallback instead.")
        return None


def run(n: int):
    from google.cloud import bigquery
    client = bigquery.Client(project=GCP_PROJECT)

    print("Loading classifier...")
    embedder, model, scaler, df_dohmh, median_days = load_classifier()
    agent = get_agent_if_available()

    print(f"Pulling {n} real complaints with a matched CAMIS from BigQuery...")
    sample_df = client.query(f"""
        SELECT descriptor, matched_camis, created_date
        FROM `{BQ_COMPLAINTS_LABELLED}`
        WHERE matched_camis IS NOT NULL
        ORDER BY RAND()
        LIMIT {n}
    """).to_dataframe()

    decisions, escalations = [], []
    for _, row in sample_df.iterrows():
        text, camis = str(row["descriptor"]), str(row["matched_camis"])
        score, label = classify(text, camis, embedder, model, scaler, df_dohmh, median_days)

        history = get_inspection_history(camis)
        recent = get_recent_complaints(camis, days=30)
        last_grade = history.get("most_recent_grade") or "Not Yet Graded"

        if agent is not None:
            user_msg = (f"Complaint (camis={camis}): {text}\nClassifier severity_score={score:.3f} "
                        f"(label={label}). Investigate and recommend a triage_level.")
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
                final_message = result["messages"][-1].content
                tier = next((t for t in (TRIAGE_ESCALATE, TRIAGE_REVIEW, TRIAGE_LOG)
                             if t in final_message.upper()), None)
            except Exception as e:
                print(f"  Agent call failed ({e}), falling back to rule-based for this complaint.")
                tier, final_message = None, ""
        else:
            tier, final_message = None, ""

        if tier is None:
            tier = rule_based_triage(bilstm_pred=label, last_grade=last_grade,
                                      days_since_inspection=999, complaint_count_30d=recent["count"])
            final_message = f"Rule-based fallback triage: {tier}"

        decision_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        decisions.append({
            "complaint_id": decision_id, "camis": camis, "decision_timestamp": now,
            "severity_score": score, "triage_tier": tier,
            "evidence_summary": f"prior_grade={last_grade}, recent_complaints_30d={recent['count']}",
            "tools_called": "inspection_history,recent_complaints,cluster_context",
            "supporting_inspection_ids": None, "cluster_id": None,
            "recommendation": final_message[:2000],
        })
        if tier == TRIAGE_ESCALATE:
            escalations.append({
                "escalation_id": str(uuid.uuid4()), "complaint_id": decision_id, "camis": camis,
                "escalation_timestamp": now, "triage_tier": tier,
                "reason": final_message[:500], "pubsub_message_id": None,
                "inspector_notified": False, "status": "pending",
            })
        print(f"  {camis}: score={score:.3f} -> {tier}")

    if decisions:
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                                             autodetect=True)
        client.load_table_from_json(decisions, BQ_AGENT_DECISIONS, job_config=job_config).result()
        print(f"Wrote {len(decisions)} rows to {BQ_AGENT_DECISIONS}")
    if escalations:
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                                             autodetect=True)
        client.load_table_from_json(escalations, BQ_ESCALATIONS, job_config=job_config).result()
        print(f"Wrote {len(escalations)} rows to {BQ_ESCALATIONS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="Number of complaints to run through the agent")
    args = parser.parse_args()
    run(args.n)
