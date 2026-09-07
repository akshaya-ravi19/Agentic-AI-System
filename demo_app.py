"""
demo_app.py
--------------
Local, no-deployment demo of the triage pipeline -- now using the
XGBoost hybrid classifier (text embedding + establishment-history
features) instead of the text-only BiLSTM, per the cross-validated
results showing a ~10x PR-AUC improvement (0.04 -> 0.41).

WHY THE SCALER MATTERS (read before touching classify()):
04d_xgboost_hybrid_cv.py trained on numeric features scaled with a
MinMaxScaler fit on ALL labelled data. That exact scaler was never
saved to disk -- so this file reproduces it by recomputing the same
two features over the same labelled_complaints.csv with the same
code. Since MinMaxScaler.fit() is deterministic (no randomness),
this reproduces an IDENTICAL scaler as long as labelled_complaints.csv
hasn't changed since 04d was run. If you re-run 04d after re-labelling
data, re-run this app fresh too (clear Streamlit's cache) so the two
stay in sync -- a silent mismatch here would misscale inputs without
ever raising an error, the same class of bug as the earlier CAMIS
dtype issue.

HOW TO RUN:
    pip install streamlit xgboost
    streamlit run demo_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import (EMBEDDING_MODEL, MODELS_DIR, DATA_LABELLED, DATA_RAW,
                            TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE)

st.set_page_config(page_title="Food Safety Triage Demo", layout="centered")
st.title("Food Safety Triage Triage — Local Demo")
st.caption("XGBoost hybrid classifier (text + establishment history). "
           "Runs entirely on your machine.")


@st.cache_resource
def load_classifier_and_scaler():
    """Loads the embedder, the trained XGBoost model, and reproduces
    the exact MinMaxScaler used at training time (see module
    docstring for why this has to be recomputed rather than just
    loaded from a file)."""
    import xgboost as xgb
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import MinMaxScaler

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    model_path = MODELS_DIR / "xgboost" / "xgb_hybrid_final.json"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Run "
                 f"notebooks/04_bilstm/04d_xgboost_hybrid_cv.py first.")
        st.stop()
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    labelled_path = DATA_LABELLED / "labelled_complaints.csv"
    dohmh_path = DATA_RAW / "dohmh_inspections.csv"
    if not labelled_path.exists() or not dohmh_path.exists():
        st.error("Can't reproduce the training scaler -- labelled_complaints.csv or "
                 "dohmh_inspections.csv missing.")
        st.stop()

    df = pd.read_csv(labelled_path, parse_dates=["created_date"])
    df_dohmh = pd.read_csv(dohmh_path, parse_dates=["inspection_date"],
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


@st.cache_resource
def load_bq_client():
    from google.cloud import bigquery
    from config.config import GCP_PROJECT
    try:
        return bigquery.Client(project=GCP_PROJECT)
    except Exception as e:
        st.warning(f"Couldn't connect to BigQuery ({e}). "
                   f"Live evidence lookups will be skipped, but classification still works.")
        return None


def get_establishment_features(camis, df_dohmh_local, median_days, as_of=None):
    if not camis:
        return 0, median_days
    as_of = as_of or pd.Timestamp.now()
    hist = df_dohmh_local[(df_dohmh_local["camis"] == str(camis).strip()) &
                           (df_dohmh_local["inspection_date"] < as_of)]
    if hist.empty:
        return 0, median_days
    days_since = (as_of - hist["inspection_date"].max()).days
    return len(hist), days_since


def classify(text, camis, embedder, model, scaler, df_dohmh_local, median_days):
    embedding = embedder.encode([text])
    prior_count, days_since = get_establishment_features(camis, df_dohmh_local, median_days)
    numeric_scaled = scaler.transform([[prior_count, days_since]])
    X = np.hstack([embedding, numeric_scaled])
    score = float(model.predict_proba(X)[0, 1])
    return score, int(score >= 0.5), prior_count, days_since


def get_evidence(client, camis):
    if client is None or not camis:
        return None
    from config.config import BQ_INSPECTIONS
    from google.cloud import bigquery
    query = f"""
        SELECT inspection_date, grade, action, critical_flag
        FROM `{BQ_INSPECTIONS}`
        WHERE camis = @camis
        ORDER BY inspection_date DESC
        LIMIT 5
    """
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


with st.spinner("Loading XGBoost model, embedder, and reproducing training scaler..."):
    embedder, model, scaler, df_dohmh_local, median_days = load_classifier_and_scaler()
bq_client = load_bq_client()

st.success("Classifier loaded and ready (XGBoost hybrid: text + establishment history).")

st.subheader("Enter a complaint")
text = st.text_area("Complaint text", placeholder="e.g. found roaches near food prep area and staff not wearing gloves", height=100)
camis = st.text_input("Restaurant CAMIS", placeholder="e.g. 50002628",
                       help="Required for meaningful results -- this model relies heavily on "
                            "establishment history, not just complaint text.")

if st.button("Run triage", type="primary"):
    if not text.strip():
        st.warning("Enter some complaint text first.")
    else:
        if not camis:
            st.info("No CAMIS entered -- establishment-history features will default to "
                    "0 prior violations / median days-since-inspection, making this "
                    "prediction much less reliable given how much the model relies on that history.")

        score, label, prior_count, days_since = classify(
            text, camis, embedder, model, scaler, df_dohmh_local, median_days
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Severity score", f"{score:.3f}")
        col2.metric("Predicted", "SEVERE" if label == 1 else "non-severe")
        col3.metric("Prior violations", int(prior_count))
        if isinstance(days_since, (int, float)):
            st.caption(f"Days since last inspection (used as a feature): {days_since:.0f}")

        evidence = get_evidence(bq_client, camis) if camis else None
        has_bad_outcome = False
        if evidence:
            st.subheader("Inspection history evidence (from BigQuery)")
            for row in evidence:
                st.write(f"- {row['inspection_date']}: grade={row.get('grade') or 'N/A'}, "
                         f"action={row.get('action') or 'N/A'}")
                if (row.get("grade") or "").strip().upper() == "C" or "closed" in (row.get("action") or "").lower():
                    has_bad_outcome = True
        elif camis:
            st.info("No BigQuery inspection history found for this CAMIS (or BigQuery unavailable).")

        tier = rule_based_tier(label, has_bad_outcome)
        tier_color = {"LOG": "🟢", "REVIEW": "🟡", "ESCALATE": "🔴"}.get(tier, "")
        st.subheader(f"Triage decision: {tier_color} {tier}")
        st.caption("Rule-based triage shown here for simplicity, not the full LLM agent. Given "
                   "cross-validated recall (~42%) and precision (~30%), treat this as one input "
                   "an inspector would weigh, not a standalone verdict.")

st.divider()
st.subheader("Validate against real labelled data")
st.caption("Pulls real rows from labelled_complaints.csv, runs the hybrid classifier on their "
           "actual descriptor + establishment features, and shows predicted vs. true label.")


@st.cache_data
def load_labelled_sample(n_per_class=10):
    path = DATA_LABELLED / "labelled_complaints.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["created_date"])
    if "descriptor" not in df.columns or "label" not in df.columns:
        return None
    df = df.dropna(subset=["descriptor", "label"])
    df["matched_camis"] = df["matched_camis"].astype(str)
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
        st.error("Couldn't load labelled_complaints.csv, or it's missing required columns.")
    else:
        with st.spinner(f"Scoring {len(sample_df)} real complaints..."):
            rows = []
            for _, r in sample_df.iterrows():
                score, pred_label, prior_count, days_since = classify(
                    str(r["descriptor"]), r["matched_camis"], embedder, model, scaler,
                    df_dohmh_local, median_days
                )
                rows.append({
                    "descriptor": r["descriptor"],
                    "true_label": int(r["label"]),
                    "predicted_score": round(score, 3),
                    "predicted_label": pred_label,
                    "correct": int(r["label"]) == pred_label,
                })
        results_df = pd.DataFrame(rows)
        accuracy = results_df["correct"].mean()
        st.metric("Accuracy on this sample", f"{accuracy:.0%}")
        st.dataframe(results_df, use_container_width=True)