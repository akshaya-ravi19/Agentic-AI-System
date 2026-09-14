"""
dashboard_app.py
--------------------
A live, read-only monitoring dashboard for the FoodGuard system --
distinct from demo_app.py (which tests a single complaint). This
gives an operational, at-a-glance view of the whole pipeline: recent
complaint volume, triage tier breakdown, active outbreak clusters,
and recent agent decisions/escalations, all pulled live from
BigQuery. Intended as a screenshot-able artifact for the dissertation
report as well as a genuinely usable monitoring view.

WHY EVERY SECTION HANDLES MISSING DATA GRACEFULLY:
Not every BigQuery table has been populated with real traffic yet
(agent_decisions/escalations specifically depend on the agent service
actually running against live complaints, which may not have happened
continuously). Rather than crashing or showing a blank page, each
section checks row count first and shows a clear "not yet populated"
message when empty -- this is itself honest, useful information about
the system's current operational status, worth keeping visible rather
than hiding.

HOW TO RUN:
    pip install streamlit
    streamlit run dashboard_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import (GCP_PROJECT, BQ_COMPLAINTS_LABELLED, BQ_INSPECTIONS,
                            BQ_CLUSTERS, BQ_AGENT_DECISIONS, BQ_ESCALATIONS)

st.set_page_config(page_title="Live Dashboard", layout="wide")
st.title("Live Operations Dashboard")
st.caption("Read-only monitoring view, live from BigQuery. Refresh the page for the latest data.")


@st.cache_resource
def get_client():
    from google.cloud import bigquery
    try:
        return bigquery.Client(project=GCP_PROJECT)
    except Exception as e:
        st.error(f"Could not connect to BigQuery: {e}")
        st.stop()


client = get_client()


def safe_query(sql, params=None):
    """Runs a query and returns a DataFrame, or None if the query
    fails (e.g. table doesn't exist yet) -- lets each dashboard
    section degrade gracefully instead of taking the whole page down."""
    from google.cloud import bigquery
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params or [])
        return client.query(sql, job_config=job_config).to_dataframe()
    except Exception as e:
        st.warning(f"Query failed (table may not exist yet): {e}")
        return None


# ── Top-line KPIs ───────────────────────────────────────────────
st.subheader("Overview")
kpi_cols = st.columns(4)

complaints_df = safe_query(f"""
    SELECT created_date, label, matched_camis, incident_zip, borough, latitude, longitude
    FROM `{BQ_COMPLAINTS_LABELLED}`
""")

with kpi_cols[0]:
    total = len(complaints_df) if complaints_df is not None else 0
    st.metric("Total labelled complaints", f"{total:,}")

with kpi_cols[1]:
    if complaints_df is not None and len(complaints_df):
        severe_pct = complaints_df["label"].mean() * 100
        st.metric("Severe (%)", f"{severe_pct:.1f}%")
    else:
        st.metric("Severe (%)", "—")

clusters_df = safe_query(f"SELECT DISTINCT cluster_id FROM `{BQ_CLUSTERS}`")
with kpi_cols[2]:
    n_clusters = len(clusters_df) if clusters_df is not None else 0
    st.metric("Active outbreak clusters", f"{n_clusters}")

escalations_df = safe_query(f"SELECT * FROM `{BQ_ESCALATIONS}`")
with kpi_cols[3]:
    n_esc = len(escalations_df) if escalations_df is not None else 0
    st.metric("Tier 3 escalations logged", f"{n_esc}")

st.caption("Note: 'Severe (%)' reflects the Grade-C/closure distant-supervision label "
           "(see Methodology, Section 3.4.1) -- treat as an administrative-outcome proxy, "
           "not a direct measure of complaint severity (label reliability \u03ba=0.025, "
           "see Results, Section 5.2.3).")

st.divider()

# ── Complaint volume over time ───────────────────────────────────
st.subheader("Complaint Volume Over Time")
if complaints_df is not None and len(complaints_df):
    volume = (complaints_df.dropna(subset=["created_date"])
              .assign(month=lambda d: pd.to_datetime(d["created_date"]).dt.to_period("M").astype(str))
              .groupby("month").size())
    st.bar_chart(volume)
else:
    st.info("No complaint data available yet.")

# ── Borough breakdown ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Complaints by Borough")
    if complaints_df is not None and len(complaints_df) and "borough" in complaints_df.columns:
        borough_counts = complaints_df["borough"].value_counts()
        st.bar_chart(borough_counts)
    else:
        st.info("No borough data available yet.")

with col2:
    st.subheader("Severity Split")
    if complaints_df is not None and len(complaints_df):
        split = complaints_df["label"].map({0: "Non-severe", 1: "Severe"}).value_counts()
        st.bar_chart(split)
    else:
        st.info("No severity data available yet.")

st.divider()

# ── Geospatial view ────────────────────────────────────────────────
st.subheader("Complaint Locations")
if complaints_df is not None and len(complaints_df):
    geo = complaints_df.dropna(subset=["latitude", "longitude"])
    if len(geo):
        st.map(geo[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}),
               size=20)
    else:
        st.info("No geolocated complaints available yet.")
else:
    st.info("No complaint data available yet.")

st.divider()

# ── Active clusters ─────────────────────────────────────────────────
st.subheader("Recent Outbreak Clusters (HDBSCAN)")
cluster_detail = safe_query(f"""
    SELECT cluster_id, COUNT(*) AS complaints_in_cluster,
           MAX(dominant_violation_category) AS dominant_violation_category,
           MAX(dominant_food_category) AS dominant_food_category,
           MAX(cluster_run_date) AS last_run
    FROM `{BQ_CLUSTERS}`
    GROUP BY cluster_id
    ORDER BY last_run DESC
    LIMIT 10
""")
if cluster_detail is not None and len(cluster_detail):
    st.dataframe(cluster_detail, use_container_width=True)
else:
    st.info("No clusters found yet -- the nightly clustering job (pipeline/clustering/cluster_job.py) "
            "may not have run against enough recent data, or hasn't been executed.")

st.divider()

# ── Recent agent decisions / escalations ────────────────────────────
st.subheader("Recent Agent Decisions")
agent_decisions = safe_query(f"""
    SELECT decision_timestamp, camis, triage_tier, severity_score, recommendation
    FROM `{BQ_AGENT_DECISIONS}`
    ORDER BY decision_timestamp DESC
    LIMIT 20
""")
if agent_decisions is not None and len(agent_decisions):
    st.dataframe(agent_decisions, use_container_width=True)
else:
    st.info("No agent decisions logged yet -- the agent service may not have processed live "
            "traffic, or the routing service hasn't written to this table yet. This is expected "
            "if the agent/routing services are still under development (see Discussion, Section 6.3.6).")

st.subheader("Tier 3 Escalations")
if escalations_df is not None and len(escalations_df):
    st.dataframe(escalations_df, use_container_width=True)
else:
    st.info("No Tier 3 escalations logged yet.")

st.divider()
st.caption("Live Dashboard — generated for dissertation reporting purposes. "
           "All figures are read directly from BigQuery at page load time.")
