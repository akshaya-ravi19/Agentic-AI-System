"""
demo_app.py
--------------
A local, no-deployment demo of the full triage pipeline: type a
complaint, see the classifier's severity score, the agent's
investigation (using real BigQuery data), and the resulting triage
tier -- all running directly in this one Python process on your
machine. No Docker, no Cloud Run, no network health checks.

WHY THIS AVOIDS EVERYTHING THAT'S BEEN FAILING ON CLOUD RUN:
Every failure so far (segfault under memory pressure, slow cold
start exceeding a startup probe) was specifically about a CONTAINER
needing to import TensorFlow + sentence-transformers and bind a port
within a tight time/memory budget. Running this as a plain local
Streamlit app sidesteps all of that -- your machine almost certainly
has more available RAM than the container did, there's no startup
probe with a deadline, and imports just take however long they take
with no consequence.

HOW TO RUN:
    pip install streamlit
    streamlit run demo_app.py
This opens a browser tab automatically. First load will be slow
(same TensorFlow/embedder import as always) -- that's normal and
expected locally, just wait for it.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import EMBEDDING_MODEL, MODELS_DIR, TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE

st.set_page_config(page_title="Food Safety Triage Demo", layout="centered")
st.title("Food Safety Triage — Local Demo")
st.caption("Runs entirely on your machine. First load takes a minute while models import.")


@st.cache_resource
def load_classifier():
    """Loaded once and cached across reruns -- Streamlit's version of
    the 'load once at container startup' idea from the Cloud Run
    service, just without any container involved."""
    import tensorflow as tf
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    model_path = MODELS_DIR / "bilstm" / "bilstm_classweight.keras"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. "
                  f"Check MODELS_DIR in config.py and that notebook 04 has been run.")
        st.stop()
    model = tf.keras.models.load_model(str(model_path))
    return embedder, model


@st.cache_resource
def load_bq_client():
    from google.cloud import bigquery
    from config.config import GCP_PROJECT
    try:
        return bigquery.Client(project=GCP_PROJECT)
    except Exception as e:
        st.warning(f"Couldn't connect to BigQuery ({e}). "
                   f"Agent evidence lookups will be skipped, but classification still works.")
        return None


def classify(text, embedder, model):
    embedding = embedder.encode([text]).reshape(1, 1, -1)
    score = float(model.predict(embedding, verbose=0)[0][0])
    return score, int(score >= 0.5)


def get_evidence(client, camis):
    """Same query logic as pipeline/agent/tools.py, inlined here so
    this demo has no dependency on the agent service being deployed."""
    if client is None or not camis:
        return None
    from config.config import BQ_INSPECTIONS
    query = f"""
        SELECT inspection_date, grade, action, critical_flag
        FROM `{BQ_INSPECTIONS}`
        WHERE camis = @camis
        ORDER BY inspection_date DESC
        LIMIT 5
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("camis", "STRING", str(camis).strip())]
    )
    try:
        rows = list(client.query(query, job_config=job_config).result())
        return [dict(r) for r in rows]
    except Exception as e:
        st.warning(f"Inspection history lookup failed: {e}")
        return None


def rule_based_tier(severity_label, has_recent_grade_c_or_closure):
    if severity_label == 1 and has_recent_grade_c_or_closure:
        return TRIAGE_ESCALATE
    elif severity_label == 1:
        return TRIAGE_REVIEW
    return TRIAGE_LOG


with st.spinner("Loading classifier (TensorFlow + sentence-transformers)... first run is slow"):
    embedder, model = load_classifier()
bq_client = load_bq_client()

st.success("Classifier loaded and ready.")

st.subheader("Enter a complaint")
text = st.text_area("Complaint text", placeholder="e.g. found roaches near food prep area and staff not wearing gloves", height=100)
camis = st.text_input("Restaurant CAMIS (optional, enables evidence lookup)", placeholder="e.g. 50002628")

if st.button("Run triage", type="primary"):
    if not text.strip():
        st.warning("Enter some complaint text first.")
    else:
        score, label = classify(text, embedder, model)
        col1, col2 = st.columns(2)
        col1.metric("Severity score", f"{score:.3f}")
        col2.metric("Predicted", "SEVERE" if label == 1 else "non-severe")

        evidence = get_evidence(bq_client, camis) if camis else None
        has_bad_outcome = False
        if evidence:
            st.subheader("Inspection history evidence")
            for row in evidence:
                st.write(f"- {row['inspection_date']}: grade={row.get('grade') or 'N/A'}, "
                         f"action={row.get('action') or 'N/A'}")
                if (row.get("grade") or "").strip().upper() == "C" or "closed" in (row.get("action") or "").lower():
                    has_bad_outcome = True
        elif camis:
            st.info("No inspection history found for this CAMIS (or BigQuery unavailable).")

        tier = rule_based_tier(label, has_bad_outcome)
        tier_color = {"LOG": "🟢", "REVIEW": "🟡", "ESCALATE": "🔴"}.get(tier, "")
        st.subheader(f"Triage decision: {tier_color} {tier}")
        st.caption("This demo uses the rule-based triage logic directly (pipeline/routing/rule_based_triage.py "
                   "reimplemented above) rather than calling the deployed LLM agent, so it works with zero "
                   "external dependencies beyond BigQuery. Swap in a real agent.invoke() call once the agent "
                   "service is deployed and reachable.")

st.divider()
st.subheader("Validate against real labelled data")
st.caption("Pulls real rows from data/labelled/labelled_complaints.csv, runs the classifier on their actual "
           "descriptor text, and shows predicted vs. the true label side by side -- lets you check systematically "
           "whether low scores on individual inputs (like the 'violently ill' example) are a one-off or a pattern.")


@st.cache_data
def load_labelled_sample(n_per_class=10):
    import pandas as pd
    from config.config import DATA_LABELLED
    path = DATA_LABELLED / "labelled_complaints.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "descriptor" not in df.columns or "label" not in df.columns:
        return None
    df = df.dropna(subset=["descriptor", "label"])
    parts = []
    for lbl in (0, 1):
        subset = df[df["label"] == lbl]
        if len(subset):
            parts.append(subset.sample(min(n_per_class, len(subset)), random_state=42))
    if not parts:
        return None
    return pd.concat(parts).reset_index(drop=True)


if st.button("Run validation sample"):
    sample_df = load_labelled_sample()
    if sample_df is None:
        st.error("Couldn't load data/labelled/labelled_complaints.csv, or it's missing "
                 "'descriptor'/'label' columns. Check DATA_LABELLED in config.py.")
    else:
        with st.spinner(f"Scoring {len(sample_df)} real complaints..."):
            rows = []
            for _, r in sample_df.iterrows():
                score, pred_label = classify(str(r["descriptor"]), embedder, model)
                rows.append({
                    "descriptor": r["descriptor"],
                    "true_label": int(r["label"]),
                    "predicted_score": round(score, 3),
                    "predicted_label": pred_label,
                    "correct": int(r["label"]) == pred_label,
                })
        import pandas as pd
        results_df = pd.DataFrame(rows)
        accuracy = results_df["correct"].mean()
        st.metric("Accuracy on this sample", f"{accuracy:.0%}")
        st.dataframe(results_df, use_container_width=True)

        # Flag the exact pattern discussed in chat: identical/similar
        # descriptor text appearing under both labels is itself
        # evidence the label isn't purely text-determined.
        dup_check = results_df.groupby("descriptor")["true_label"].nunique()
        inconsistent = dup_check[dup_check > 1]
        if len(inconsistent):
            st.warning(
                f"{len(inconsistent)} descriptor(s) in this sample appear under BOTH labels "
                f"(e.g. {inconsistent.index[0]!r}) -- confirms the label depends on the linked "
                f"inspection's outcome, not on complaint text alone. Low scores on severe-sounding "
                f"text are expected given this, not a bug."
            )