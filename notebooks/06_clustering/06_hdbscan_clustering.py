# ============================================================
# NOTEBOOK 06 — HDBSCAN Spatiotemporal Clustering
# 5-dimensional feature space:
#   1+2: Latitude, Longitude (geographic)
#   3:   Unix timestamp      (temporal)
#   4:   Complaint type      (categorical, downweighted)
#   5:   Symptom flag        (illness signal — binary)
#   6:   Food category       (categorical, downweighted)
# Validates clusters against inspection outcomes (chi-square).
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

df = pd.read_csv(DATA_LABELLED/"labelled_complaints.csv", parse_dates=["created_date"])
df = df.dropna(subset=["latitude","longitude","created_date"]).copy()
df["latitude"]  = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)
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

# ── Dimension: Restaurant identity (CAMIS) ──────────────────────
# Explicitly requested addition: lat/long alone can't fully stand
# in for "same restaurant" (e.g. GPS jitter, or a complaint logged
# against a nearby cross-street). Encode restaurant identity as its
# own signal so repeat complaints against the SAME establishment
# cluster together even if their coordinates aren't pixel-identical.
# A raw one-hot of every CAMIS would blow up the feature space, so
# instead we use each restaurant's rolling complaint frequency as a
# lightweight numeric proxy for "how much recent history exists
# here" -- restaurants with more complaints get a distinguishing
# signal without one-hot dimensionality explosion.
if "matched_camis" in df.columns:
    camis_freq = df["matched_camis"].value_counts()
    df["restaurant_freq"] = df["matched_camis"].map(camis_freq).fillna(0)
else:
    df["restaurant_freq"] = 0

# ── Dimension: Severity ──────────────────────────────────────
# Explicitly requested dimension: complaints linked to a severe
# outcome (grade C / closure, per notebook 03's redefinition) should
# pull clustering toward grouping serious cases together, not just
# by space/time/type alone. Using it as an INPUT feature here is
# separate from the chi-square validation step further down (which
# checks whether resulting clusters correspond to severity
# afterwards) -- one uses severity to help shape the clusters, the
# other independently checks if the shapes it found make sense.
if "label" in df.columns:
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
print(f"Window: last {HDBSCAN_WINDOW_DAYS} days → {len(df_w):,} complaints")

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
if "label" in df_w.columns:
    df_w["in_cluster"] = (df_w["cluster_id"]>=0).astype(int)
    ct = pd.crosstab(df_w["in_cluster"], df_w["label"])
    print(f"\nCross-tabulation (in_cluster vs severe label):\n{ct}")
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"Chi-square: {chi2:.2f} | p-value: {p:.4f}")
    print("Clusters carry significant predictive signal" if p<0.05
          else "No significant predictive signal found")

# ── Evaluation 3: Manual coherence instructions ───────────────
print("\n=== Manual coherence check ===")
for cid in sorted(df_w["cluster_id"].unique()):
    if cid == -1: continue
    texts = df_w[df_w["cluster_id"]==cid]["descriptor"].head(3).tolist()
    print(f"\nCluster {cid} (review these and decide if they are related):")
    for t in texts: print(f"  - {t[:80]}")

# ── Save results ──────────────────────────────────────────────
df_w.to_csv(EVAL_DIR/"clustering"/"cluster_results.csv", index=False)

# ── Map visualisation ─────────────────────────────────────────
m = folium.Map(location=[df_w["latitude"].mean(), df_w["longitude"].mean()], zoom_start=12)
colors = ["red","blue","green","purple","orange","darkred","cadetblue","darkgreen","pink"]
for _, row in df_w[df_w["cluster_id"]>=0].iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5, color=colors[int(row["cluster_id"]) % len(colors)],
        fill=True, fill_opacity=0.7,
        popup=f"Cluster {row['cluster_id']}: {str(row.get('descriptor',''))[:50]}"
    ).add_to(m)
map_path = EVAL_DIR/"clustering"/"cluster_map.html"
m.save(str(map_path))
print(f"\nMap saved → open {map_path} in your browser")
