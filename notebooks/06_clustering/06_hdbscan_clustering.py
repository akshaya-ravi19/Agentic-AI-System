# ============================================================
# NOTEBOOK 06 — HDBSCAN Spatiotemporal Clustering
# Data source: Ground Truth A — labelled_complaints_ground_truth.csv
#   (73,450 NYC 311 food safety complaints, 3-tier municipal taxonomy)
#   Coherent with 04_food_safety_classifier.py (production BiLSTM).
#
# Feature space:
#   1+2: Latitude, Longitude (geographic)
#   3:   Unix timestamp      (temporal)
#   4:   Complaint type      (categorical, downweighted)
#   5:   Symptom flag        (illness signal — binary)
#   6:   Food category       (categorical, downweighted)
#   7:   Establishment frequency (address-based repeat-complaint proxy)
#   8:   Hazard severity     (priority_label from GT-A taxonomy)
# Validates clusters against hazard priority (chi-square).
# ============================================================
import pandas as pd, numpy as np, hdbscan, folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *
EVAL_DIR.joinpath("clustering").mkdir(parents=True, exist_ok=True)

# Use GT-A: full 73,450-row municipal taxonomy dataset — coherent with
# the production BiLSTM (04_food_safety_classifier.py).
# GT-B (labelled_complaints.csv, 1,458 rows) was used previously but
# it's the DOHMH-matched research subset with extreme imbalance;
# clustering on it produced only 105 complaints in the 30-day window.
# GT-A gives 50x more data and richer geographic/temporal coverage.
df = pd.read_csv(DATA_LABELLED/"labelled_complaints_ground_truth.csv", parse_dates=["created_date"], low_memory=False)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude","longitude","created_date"]).copy()
print(f"Loaded: {len(df):,} complaints for clustering")

# ── Dimension 4: Symptom flag ─────────────────────────────────
import re
pattern = "|".join(SYMPTOM_KEYWORDS)
df["symptom_flag"] = df["descriptor"].str.lower().str.contains(pattern, na=False).astype(int)
print(f"Illness symptom mentions: {df['symptom_flag'].sum():,} ({df['symptom_flag'].mean():.1%})")

# ── Dimension 5: Food category ────────────────────────────────
def get_food_cat(text):
    if not isinstance(text, str): return "other"
    t = text.lower()
    for cat, kws in FOOD_CATEGORIES.items():
        if any(k in t for k in kws): return cat
    return "other"

df["food_category"] = df["descriptor"].apply(get_food_cat)
food_dummies = pd.get_dummies(df["food_category"], prefix="food")

# ── Dimension 3: Complaint type ───────────────────────────────
# NOTE: "violation_category" never actually gets created anywhere
# upstream in this pipeline (notebooks 02/03 don't produce it), so
# this always silently fell through to an empty DataFrame -- the
# complaint-type dimension was missing from clustering entirely.
# FIX: derive it here from the complaint text using the
# VIOLATION_CATEGORIES keyword map already defined in config.py
# (which was likewise defined but never actually used anywhere).
def get_violation_category(text):
    if not isinstance(text, str):
        return "other"
    t = text.lower()
    for cat, kws in VIOLATION_CATEGORIES_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return "other"

if "violation_category" not in df.columns:
    df["violation_category"] = df["descriptor"].apply(get_violation_category)
type_dummies = pd.get_dummies(df["violation_category"], prefix="type")

# ── Dimension: Establishment identity ─────────────────────────
# GT-A does not have matched_camis (that's only in GT-B after DOHMH
# fuzzy-matching). Instead use incident_address as the establishment
# identity proxy — complaints at the same street address are almost
# certainly against the same restaurant, and rolling frequency of
# complaints at that address is a meaningful "repeat offender" signal.
if "incident_address" in df.columns:
    addr_freq = df["incident_address"].str.strip().str.upper().value_counts()
    df["restaurant_freq"] = df["incident_address"].str.strip().str.upper().map(addr_freq).fillna(0)
elif "matched_camis" in df.columns:
    camis_freq = df["matched_camis"].value_counts()
    df["restaurant_freq"] = df["matched_camis"].map(camis_freq).fillna(0)
else:
    df["restaurant_freq"] = 0

# ── Dimension: Severity ──────────────────────────────────────
# GT-A uses priority_label (1=Actionable Hazard, 0=Routine) from
# the 3-tier municipal taxonomy. This replaces the DOHMH outcome
# label (Grade C / Closure) used in GT-B — it's available for all
# 73,450 records and is coherent with the production BiLSTM target.
if "priority_label" in df.columns:
    df["severity_feature"] = df["priority_label"].fillna(0).astype(float)
elif "label" in df.columns:
    df["severity_feature"] = df["label"].fillna(0).astype(float)
else:
    df["severity_feature"] = 0.0

# ── Build feature matrix ──────────────────────────────────────
scaler = MinMaxScaler()
spatial = scaler.fit_transform(df[["latitude","longitude"]])
ts_scaled = scaler.fit_transform(df[["created_date"]].assign(
    ts=df["created_date"].astype(np.int64)/1e12)[["ts"]])

rest_scaled = scaler.fit_transform(df[["restaurant_freq"]])

cat_feats = np.hstack([
    type_dummies.reindex(df.index, fill_value=0).values * 0.5,
    df[["symptom_flag"]].values,
    food_dummies.reindex(df.index, fill_value=0).values * 0.5,
    df[["severity_feature"]].values,
    rest_scaled * 0.5,
])
features = np.hstack([spatial, ts_scaled, cat_feats])
print(f"Feature matrix: {features.shape}")

# ── Apply rolling window then HDBSCAN ─────────────────────────
cutoff = df["created_date"].max() - pd.Timedelta(days=HDBSCAN_WINDOW_DAYS)
mask   = df["created_date"] >= cutoff
df_w   = df[mask].copy().reset_index(drop=True)
feat_w = features[mask.values]
print(f"Window: last {HDBSCAN_WINDOW_DAYS} days -> {len(df_w):,} complaints")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    cluster_selection_method=HDBSCAN_CLUSTER_METHOD
)
df_w["cluster_id"] = clusterer.fit_predict(feat_w)
n_cl = df_w["cluster_id"].nunique() - (1 if -1 in df_w["cluster_id"].values else 0)
print(f"Clusters found: {n_cl}")
print(f"In cluster: {(df_w['cluster_id']>=0).sum():,} | Noise: {(df_w['cluster_id']==-1).sum():,}")

# ── Evaluation 1: Internal quality ───────────────────────────
in_cluster = df_w["cluster_id"] >= 0
if in_cluster.sum() > 1 and df_w.loc[in_cluster,"cluster_id"].nunique() > 1:
    sil = silhouette_score(feat_w[in_cluster.values], df_w.loc[in_cluster,"cluster_id"])
    db  = davies_bouldin_score(feat_w[in_cluster.values], df_w.loc[in_cluster,"cluster_id"])
    print(f"\nSilhouette score: {sil:.4f} (higher=better, ideal>0.5)")
    print(f"Davies-Bouldin:   {db:.4f} (lower=better, ideal<1.0)")

# ── Evaluation 2: Predictive validity (chi-square) ────────────
# Check whether cluster membership correlates with hazard priority.
# GT-A: uses priority_label (Actionable=1 vs Routine=0)
# GT-B fallback: uses label (Severe=1 vs Non-severe=0)
label_col = "priority_label" if "priority_label" in df_w.columns else "label"
if label_col in df_w.columns:
    df_w["in_cluster"] = (df_w["cluster_id"] >= 0).astype(int)
    ct = pd.crosstab(df_w["in_cluster"], df_w[label_col])
    print(f"\nCross-tabulation (in_cluster vs {label_col}):\n{ct}")
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"Chi-square: {chi2:.2f} | p-value: {p:.4f}")
    print("Clusters carry significant predictive signal" if p < 0.05
          else "No significant predictive signal found")

# ── Evaluation 3: Manual coherence check ─────────────────────
print("\n=== Manual coherence check ===")
for cid in sorted(df_w["cluster_id"].unique()):
    if cid == -1: continue
    subset = df_w[df_w["cluster_id"] == cid]
    texts = subset["descriptor"].head(3).tolist()
    boro = subset["borough"].mode()[0] if "borough" in subset.columns else "?"
    actionable_pct = subset[label_col].mean() if label_col in subset.columns else 0
    print(f"\nCluster {cid} | Borough: {boro} | Actionable: {actionable_pct:.0%} | n={len(subset)}")
    for t in texts:
        print(f"  - {t[:80]}")

# ── Save results ──────────────────────────────────────────────
df_w.to_csv(EVAL_DIR/"clustering"/"cluster_results.csv", index=False)

# ── Map visualisation ─────────────────────────────────────────
m = folium.Map(location=[df_w["latitude"].mean(), df_w["longitude"].mean()], zoom_start=12)
colors = ["red","blue","green","purple","orange","darkred","cadetblue","darkgreen","pink",
          "lightred","beige","lightblue","lightgreen","gray","black"]
hazard_cat = "hazard_category" if "hazard_category" in df_w.columns else label_col
for _, row in df_w[df_w["cluster_id"] >= 0].iterrows():
    popup_txt = (f"Pattern Group {row['cluster_id']}: {str(row.get('descriptor',''))[:40]}"
                 f" | {str(row.get(hazard_cat,''))[:30]}"
                 f" | {str(row.get('borough',''))}")
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5, color=colors[int(row["cluster_id"]) % len(colors)],
        fill=True, fill_opacity=0.7,
        popup=popup_txt
    ).add_to(m)
map_path = EVAL_DIR/"clustering"/"cluster_map.html"
m.save(str(map_path))
print(f"\nMap saved -> open {map_path} in your browser")

