"""
pipeline/agent/tools.py
--------------------------
BigQuery-backed versions of the agent tools prototyped in notebook
05. Notebook 05 reads from local CSVs (fine for offline
experimentation); this module reads from the live BigQuery tables so
the agent can run as a deployed Cloud Run service against real,
continuously-refreshed data from the ingestion jobs.
"""
import sys
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import GCP_PROJECT, BQ_INSPECTIONS, BQ_CLUSTERS, BQ_COMPLAINTS_LABELLED

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = bigquery.Client(project=GCP_PROJECT)
    return _client


def get_inspection_history(camis_id: str) -> dict:
    """Most recent inspections for a restaurant, most recent first."""
    camis_id = str(camis_id).strip()
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
    rows = list(_get_client().query(query, job_config=job_config).result())
    if not rows:
        return {"camis": camis_id, "inspections": [], "note": "no inspection history found"}
    return {
        "camis": camis_id,
        "inspections": [dict(r) for r in rows],
        "most_recent_grade": rows[0].get("grade"),
    }


def get_recent_complaints(camis_id: str, days: int = 30) -> dict:
    """Complaints against this establishment in the last N days."""
    camis_id = str(camis_id).strip()
    query = f"""
        SELECT complaint_type, descriptor, created_date, label
        FROM `{BQ_COMPLAINTS_LABELLED}`
        WHERE matched_camis = @camis
          AND created_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        ORDER BY created_date DESC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("camis", "STRING", camis_id),
        bigquery.ScalarQueryParameter("days", "INT64", days),
    ])
    rows = list(_get_client().query(query, job_config=job_config).result())
    return {"camis": camis_id, "window_days": days, "count": len(rows),
            "complaints": [dict(r) for r in rows]}


def get_cluster_context(camis_id: str) -> dict:
    """
    Whether this establishment currently belongs to an active HDBSCAN
    cluster (populated nightly by pipeline/clustering/cluster_job.py),
    and how big/persistent that cluster is -- used as multi-establishment
    outbreak-pattern evidence, not just single-complaint severity.
    """
    camis_id = str(camis_id).strip()
    query = f"""
        SELECT cluster_id, cluster_size, persistence_score,
               dominant_violation_category, dominant_food_category, cluster_run_date
        FROM `{BQ_CLUSTERS}`
        WHERE camis = @camis
        ORDER BY cluster_run_date DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("camis", "STRING", camis_id)]
    )
    rows = list(_get_client().query(query, job_config=job_config).result())
    if not rows:
        return {"camis": camis_id, "in_cluster": False}
    r = dict(rows[0])
    r["in_cluster"] = True
    return r
