"""
pipeline/clustering/cluster_job.py
--------------------------------------
Cloud Run Job version of notebook 06's HDBSCAN clustering. Reads
recent labelled complaints from BigQuery instead of a local CSV,
runs the same multi-dimensional clustering (time, location,
restaurant, complaint type, symptoms, food category, severity),
and writes cluster assignments back to BQ_CLUSTERS.

Intended to run nightly via Cloud Scheduler, matching "Stage 4
Nightly Batch" in the architecture diagram -- see deployment
instructions in chat for the gcloud commands (same build/deploy/
schedule pattern as pipeline/ingestion/).

NOTE: This mirrors notebook 06's feature engineering closely. If you
change the feature set in notebook 06 during experimentation, update
this file too -- they're not automatically kept in sync, since one
is for offline experimentation and this one is what actually runs in
production.
"""
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import (
    GCP_PROJECT, BQ_COMPLAINTS_LABELLED, BQ_CLUSTERS,
    HDBSCAN_WINDOW_DAYS, SYMPTOM_KEYWORDS, FOOD_CATEGORIES,
    VIOLATION_CATEGORIES_KEYWORDS,
)


def load_recent_complaints(client) -> pd.DataFrame:
    query = f"""
        SELECT descriptor, created_date, matched_camis, label,
               incident_zip, borough
        FROM `{BQ_COMPLAINTS_LABELLED}`
        WHERE created_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", HDBSCAN_WINDOW_DAYS)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def get_violation_category(text):
    if not isinstance(text, str):
        return "other"
    t = text.lower()
    for cat, kws in VIOLATION_CATEGORIES_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return "other"


def get_food_category(text):
    if not isinstance(text, str):
        return "other"
    t = text.lower()
    for cat, kws in FOOD_CATEGORIES.items():
        if any(k in t for k in kws):
            return cat
    return "other"


def build_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["symptom_flag"] = df["descriptor"].apply(
        lambda t: float(any(k in str(t).lower() for k in SYMPTOM_KEYWORDS))
    )
    df["violation_category"] = df["descriptor"].apply(get_violation_category)
    df["food_category"] = df["descriptor"].apply(get_food_category)
    df["severity_feature"] = df["label"].fillna(0).astype(float)

    camis_freq = df["matched_camis"].value_counts()
    df["restaurant_freq"] = df["matched_camis"].map(camis_freq).fillna(0)

    ts_scaled = MinMaxScaler().fit_transform(
        df[["created_date"]].apply(lambda c: c.astype(np.int64) / 1e12)
    )
    rest_scaled = MinMaxScaler().fit_transform(df[["restaurant_freq"]])

    type_dummies = pd.get_dummies(df["violation_category"], prefix="type")
    food_dummies = pd.get_dummies(df["food_category"], prefix="food")

    features = np.hstack([
        ts_scaled,
        df[["symptom_flag", "severity_feature"]].values,
        type_dummies.values * 0.5,
        food_dummies.values * 0.5,
        rest_scaled * 0.5,
    ])
    return features, df


def run():
    client = bigquery.Client(project=GCP_PROJECT)
    print("Loading recent complaints from BigQuery...")
    df = load_recent_complaints(client)
    print(f"Loaded {len(df):,} complaints from the last {HDBSCAN_WINDOW_DAYS} days")

    if len(df) < 10:
        print("Too few complaints to cluster meaningfully -- skipping this run.")
        return

    features, df = build_features(df)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(features)
    df["cluster_label"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters ({(labels == -1).sum()} noise points)")

    run_id = str(uuid.uuid4())[:8]
    run_date = datetime.now(timezone.utc).isoformat()

    rows = []
    for cluster_id, group in df[df["cluster_label"] != -1].groupby("cluster_label"):
        persistence = float(clusterer.cluster_persistence_[cluster_id]) \
            if cluster_id < len(clusterer.cluster_persistence_) else None
        dominant_violation = group["violation_category"].mode().iloc[0] if len(group) else None
        dominant_food = group["food_category"].mode().iloc[0] if len(group) else None
        for _, r in group.iterrows():
            rows.append({
                "cluster_id": f"{run_id}-{cluster_id}",
                "camis": r.get("matched_camis"),
                "complaint_id": None,  # 311 unique_key not selected in this query; join later if needed
                "cluster_run_date": run_date,
                "cluster_size": int(len(group)),
                "persistence_score": persistence,
                "dominant_violation_category": dominant_violation,
                "dominant_food_category": dominant_food,
            })

    if not rows:
        print("No non-noise clusters found this run -- nothing to write.")
        return

    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    load_job = client.load_table_from_json(rows, BQ_CLUSTERS, job_config=job_config)
    load_job.result()
    print(f"Wrote {len(rows)} cluster-membership rows to {BQ_CLUSTERS}")


if __name__ == "__main__":
    run()
