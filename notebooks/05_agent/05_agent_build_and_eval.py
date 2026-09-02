# ============================================================
# NOTEBOOK 05 — LangGraph Agent Build and Evaluation
# Defines 5 tools, builds the ReAct agent, evaluates on
# 50 expert-labelled cases, compares vs rule-based baseline.
# The agent RECOMMENDS — humans make all final decisions.
# ============================================================
import os, json, pandas as pd, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *
EVAL_DIR.joinpath("agent").mkdir(parents=True, exist_ok=True)

# ── SECTION A: Define the 5 agent tools ──────────────────────
# In notebook mode, these query local CSV files.
# In production (pipeline/agent/), they query BigQuery.

# dtype=str on camis is essential here -- without it pandas silently
# reads this ID column as a float, and every lookup below (which
# receives camis_id as a string) would fail to match anything. This
# is the same bug that caused 0 labels in notebook 03 -- it applies
# anywhere a CAMIS column gets read from CSV.
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv",
                        parse_dates=["inspection_date"], dtype={"camis": str})
df_complaints = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv",
                             parse_dates=["created_date"],
                             dtype={"matched_camis": str})

def get_inspection_history(camis_id: str) -> dict:
    camis_id = str(camis_id).strip()
    rows = df_dohmh[df_dohmh["camis"] == camis_id].sort_values("inspection_date", ascending=False)
    if rows.empty:
        return {"error": "No inspection records found", "camis": camis_id}
    last = rows.iloc[0]
    days_since = (pd.Timestamp.now() - rows["inspection_date"].max()).days
    return {
        "camis": camis_id,
        "total_inspections": len(rows),
        "last_inspection_date": str(rows["inspection_date"].max().date()),
        "days_since_last_inspection": days_since,
        "critical_violations_ever": int((rows["critical_flag"]=="Critical").sum()),
        "last_grade": str(last.get("grade","Unknown")),
        "recent_3_inspections": rows.head(3)[["inspection_date","critical_flag","grade"]].to_dict("records")
    }

def get_recent_complaints(camis_id: str, days: int = 30) -> dict:
    camis_id = str(camis_id).strip()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    rows = df_complaints[
        (df_complaints.get("matched_camis","").astype(str) == camis_id) &
        (df_complaints["created_date"] >= cutoff)
    ]
    return {
        "complaint_count": len(rows),
        "severe_predicted": int((rows.get("label",pd.Series())==1).sum()),
        "complaint_texts": rows["descriptor"].head(3).tolist()
    }

def get_cluster_context(camis_id: str) -> dict:
    # In notebook mode: return placeholder (clusters computed in notebook 06)
    # In production: queries BQ_CLUSTERS table
    return {"in_cluster": False, "note": "Run notebook 06 first to populate clusters"}

def get_related_incidents(complaint_type: str, lat: float, lng: float, days: int = 14) -> dict:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    if "latitude" not in df_complaints.columns:
        return {"related_count": 0, "note": "Location data not available in this split"}
    nearby = df_complaints[
        (df_complaints["created_date"] >= cutoff) &
        (abs(df_complaints["latitude"].astype(float) - lat) < 0.01) &
        (abs(df_complaints["longitude"].astype(float) - lng) < 0.01)
    ]
    return {"related_incident_count": len(nearby)}

def write_triage_report(evidence: dict) -> dict:
    """Final tool call — produces the structured inspector recommendation."""
    return {
        "triage_level": evidence.get("triage_level", TRIAGE_LOG),
        "supporting_evidence": evidence.get("evidence_list", []),
        "confidence": evidence.get("confidence", "low"),
        "recommended_action": evidence.get("action", "Log complaint for next scheduled inspection."),
        "unsupported_claims": evidence.get("unsupported", []),
        "reasoning_chain": evidence.get("reasoning", []),
        "human_decision_required": True,
        "disclaimer": "Recommendation only. Inspector makes all final regulatory decisions."
    }

print("All 5 tools defined.")

# ── SECTION B: Rule-based baseline ───────────────────────────
# Imported from pipeline/routing/rule_based_triage.py rather than
# defined here -- this is the SAME function notebook 07 uses for
# evaluation, and the same one output_router.py can fall back to.
# Keeping one shared definition means the agent is always benchmarked
# against the exact rules being used elsewhere, not a copy that could
# quietly drift out of sync.

from pipeline.routing.rule_based_triage import rule_based_triage

# ── SECTION C: Build the LangGraph ReAct agent ───────────────
# Requires: GOOGLE_API_KEY in environment
# Install: pip install langchain langgraph langchain-google-genai

SYSTEM_PROMPT = f"""You are an AI assistant supporting regulators like Health and Safety Officers and Restaurant Inspectors.
You investigate food safety complaints using 5 tools:
- get_inspection_history(camis_id): past DOHMH inspections for a restaurant
- get_recent_complaints(camis_id, days): recent complaints about this restaurant
- get_cluster_context(camis_id): checks if restaurant is in an outbreak cluster
- get_related_incidents(complaint_type, lat, lng, days): similar nearby incidents
- write_triage_report(evidence): produces your final structured recommendation

Assign one of three triage levels:
- {TRIAGE_LOG}: minor issue, record for next scheduled inspection
- {TRIAGE_REVIEW}: inspector should review within 5 working days
- {TRIAGE_ESCALATE}: inspector should visit within 48 hours

RULES:
1. You RECOMMEND — you never make final regulatory decisions autonomously.
2. Cite which tool provided each piece of evidence.
3. Flag any claim not traceable to a tool output as UNSUPPORTED in your report.
4. Call write_triage_report as your final action.
5. Maximum {AGENT_MAX_STEPS} tool calls per investigation.
"""

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool

    @tool
    def tool_get_inspection_history(camis_id: str) -> str:
        """Retrieve full DOHMH inspection history for a restaurant by CAMIS ID."""
        return json.dumps(get_inspection_history(camis_id))

    @tool
    def tool_get_recent_complaints(camis_id: str, days: int = 30) -> str:
        """Get all recent complaints about a restaurant."""
        return json.dumps(get_recent_complaints(camis_id, days))

    @tool
    def tool_get_cluster_context(camis_id: str) -> str:
        """Check if restaurant belongs to an active HDBSCAN outbreak cluster."""
        return json.dumps(get_cluster_context(camis_id))

    @tool
    def tool_get_related_incidents(complaint_type: str, lat: float, lng: float, days: int = 14) -> str:
        """Find similar complaints at nearby restaurants in the time window."""
        return json.dumps(get_related_incidents(complaint_type, lat, lng, days))

    @tool
    def tool_write_triage_report(evidence_json: str) -> str:
        """Produce the final structured triage recommendation for the inspector."""
        evidence = json.loads(evidence_json) if isinstance(evidence_json, str) else evidence_json
        return json.dumps(write_triage_report(evidence))

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=AGENT_TEMPERATURE,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    tools = [tool_get_inspection_history, tool_get_recent_complaints,
             tool_get_cluster_context, tool_get_related_incidents,
             tool_write_triage_report]
    # NOTE: langgraph's create_react_agent renamed state_modifier -> prompt
    # in newer releases (state_modifier is deprecated/removed). Since
    # requirements.txt intentionally doesn't pin an exact langgraph
    # version, use "prompt" so this works on current installs.
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    print("LangGraph agent built successfully.")

    # Test on one complaint
    test_input = {
        "messages": [{
            "role": "user",
            "content": ("Complaint: I saw cockroaches on the kitchen counter at Mario's Pizza. "
                        "CAMIS: 12345678. Location: 40.7580,-73.9855. "
                        "Complaint type: pest_infestation. Please investigate.")
        }]
    }
    print("\nRunning test investigation...")
    result = agent.invoke(test_input)
    print("Agent output:", result["messages"][-1].content[:500])

except ImportError:
    print("LangGraph not installed. Run: pip install langchain langgraph langchain-google-genai")
except Exception as e:
    print(f"Agent setup error: {e}")
    print("Check that GOOGLE_API_KEY is set in your .env file.")

# ── SECTION D: Evaluation framework ──────────────────────────
print("""
=== EVALUATION STEPS ===
1. Create evaluation/agent/expert_labelled_50_cases.csv
   Columns: complaint_id, descriptor, camis, bilstm_pred,
            last_grade, days_since_inspection, complaint_count_30d,
            expert_triage (LOG/REVIEW/ESCALATE)

2. For each of 50 cases:
   - Run agent → save: agent_triage, tool_calls, reasoning_chain, report
   - Run rule_based_triage → save: rule_triage

3. Compute (vs expert_triage):
   - Cohen's Kappa for agent       (target >= 0.60)
   - Cohen's Kappa for rule-based  (comparison)
   - Per-level F1 for both
   - Tool selection accuracy
   - Hallucination rate (unsupported_claims / total claims)
   - Consistency (run same 20 cases twice, % agreement)

4. Delta(agent_kappa - rule_kappa) = what the agent adds over simple rules
""")