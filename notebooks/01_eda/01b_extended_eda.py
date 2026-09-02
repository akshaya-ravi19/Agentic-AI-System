"""
notebooks/01_eda/01b_extended_eda.py
----------------------------------------
Additional visuals beyond the original 01_exploratory_analysis.py --
picked to either reveal patterns worth discussing in your data
chapter, or to directly justify methodology decisions made elsewhere
in the project (the LABEL_WINDOW_DAYS choice, the syndromic-
surveillance framing, the fairness-audit subgroups).

Each figure is saved to data/processed/ as a PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

sns.set_style("whitegrid")

df_311 = pd.read_csv(DATA_RAW / "nyc_311_food_complaints.csv", parse_dates=["created_date"], low_memory=False)
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"],
                        dtype={"camis": str}, low_memory=False)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
print(f"311: {len(df_311):,} rows | DOHMH: {len(df_dohmh):,} rows")


# ── 1. Borough x Month heatmap ─────────────────────────────────
# Reveals whether complaint volume is roughly stable per borough
# over time, or whether specific boroughs spike at specific periods
# -- relevant context for the clustering chapter's time dimension.
print("\n[1/8] Borough x Month heatmap...")
df_311["month"] = df_311["created_date"].dt.to_period("M").astype(str)
pivot = df_311.pivot_table(index="borough", columns="month", values="unique_key",
                            aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Complaints"})
ax.set_title("Complaint Volume by Borough and Month")
plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_borough_month_heatmap.png", dpi=150)
plt.close(fig)


# ── 2. Top complaint descriptors, as an actual chart ────────────
print("[2/8] Top complaint descriptors bar chart...")
top_desc = df_311["descriptor"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top_desc.values, y=top_desc.index, ax=ax, color="#c0392b")
ax.set_title("Top 15 Complaint Descriptors")
ax.set_xlabel("Count")
plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_top_descriptors.png", dpi=150)
plt.close(fig)


# ── 3. DOHMH grade distribution ─────────────────────────────────
# Directly relevant to your severity-label design decision (grade C
# / closure) -- shows how rare a C grade actually is relative to
# A/B/pending, which is exactly why severe ended up ~3% prevalent.
print("[3/8] DOHMH grade distribution...")
if "grade" in df_dohmh.columns:
    grade_counts = df_dohmh["grade"].fillna("(blank/pending)").value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=ax, color="#2980b9")
    ax.set_title("DOHMH Inspection Grade Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_grade_distribution.png", dpi=150)
    plt.close(fig)
    c_pct = (df_dohmh["grade"].fillna("").str.strip().str.upper() == "C").mean() * 100
    print(f"  Grade C: {c_pct:.2f}% of all inspection rows -- this is why 'severe' "
          f"(grade C or closure) is a rare label, not a bug.")


# ── 4. Cuisine distribution ──────────────────────────────────────
# Relevant to the fairness/bias audit (notebook 07) -- shows which
# cuisines have enough volume for a meaningful subgroup comparison,
# and which are too sparse to draw conclusions from.
print("[4/8] Cuisine distribution...")
if "cuisine_description" in df_dohmh.columns:
    top_cuisine = df_dohmh["cuisine_description"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_cuisine.values, y=top_cuisine.index, ax=ax, color="#27ae60")
    ax.set_title("Top 15 Cuisine Types (Inspection Rows)")
    ax.set_xlabel("Count")
    plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_cuisine_distribution.png", dpi=150)
    plt.close(fig)
    sparse = (df_dohmh["cuisine_description"].value_counts() < 30).sum()
    print(f"  {sparse} cuisine categories have fewer than 30 inspection rows -- "
          f"exclude these from fairness-audit subgroup comparisons, too little data "
          f"to draw a reliable conclusion per group.")


# ── 5. Day-of-week complaint pattern ─────────────────────────────
# Relevant to the real-time ingestion design -- shows whether
# complaint volume is roughly even across the week or clusters on
# specific days (e.g. weekends), useful context for Cloud Scheduler
# frequency decisions.
print("[5/8] Day-of-week pattern...")
df_311["day_of_week"] = df_311["created_date"].dt.day_name()
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_counts = df_311["day_of_week"].value_counts().reindex(day_order)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax, color="#8e44ad")
ax.set_title("Complaint Volume by Day of Week")
ax.set_ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_day_of_week.png", dpi=150)
plt.close(fig)


# ── 6. Complaint-to-inspection lag distribution ──────────────────
# THE MOST METHODOLOGICALLY IMPORTANT NEW FIGURE: directly justifies
# (or challenges) the LABEL_WINDOW_DAYS choice in notebook 03. Shows
# the actual real-world distribution of "how many days after a
# complaint does the next inspection happen" for matched establishments.
print("[6/8] Complaint-to-inspection lag distribution (this loops over rows, ~1 min)...")
sample_311 = df_311.dropna(subset=["created_date"]).sample(
    min(2000, len(df_311)), random_state=RANDOM_SEED
)
lags = []
if "camis" in df_dohmh.columns:
    # This needs matched_camis, which only exists post-03 -- fall
    # back gracefully if run before that step.
    processed_path = DATA_PROCESSED / "311_with_camis.csv"
    if processed_path.exists():
        df_matched = pd.read_csv(processed_path, parse_dates=["created_date"],
                                  dtype={"matched_camis": str})
        df_matched = df_matched.dropna(subset=["matched_camis"]).sample(
            min(2000, len(df_matched)), random_state=RANDOM_SEED
        )
        for _, row in df_matched.iterrows():
            hist = df_dohmh[(df_dohmh["camis"] == row["matched_camis"]) &
                             (df_dohmh["inspection_date"] > row["created_date"])]
            if not hist.empty:
                lags.append((hist["inspection_date"].min() - row["created_date"]).days)

if lags:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(lags, bins=50, ax=ax, color="#d35400")
    for w in LABEL_WINDOW_SENSITIVITY:
        ax.axvline(w, linestyle="--", color="gray", alpha=0.7)
        ax.text(w, ax.get_ylim()[1]*0.9, f"{w}d", rotation=90, fontsize=8)
    ax.axvline(LABEL_WINDOW_DAYS, color="red", linewidth=2, label=f"Chosen window ({LABEL_WINDOW_DAYS}d)")
    ax.set_title("Days Between Complaint and Next Inspection")
    ax.set_xlabel("Days"); ax.legend()
    plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_complaint_inspection_lag.png", dpi=150)
    plt.close(fig)
    pct_within_window = (pd.Series(lags) <= LABEL_WINDOW_DAYS).mean() * 100
    print(f"  {pct_within_window:.1f}% of matched complaints get an inspection within "
          f"your chosen {LABEL_WINDOW_DAYS}-day window -- quote this directly when "
          f"justifying the window choice in your methodology chapter.")
else:
    print("  Skipped -- run notebook 03 first so 311_with_camis.csv exists.")


# ── 7. Symptom-keyword frequency ──────────────────────────────────
# Ties directly to the syndromic-surveillance framing discussed in
# chat -- shows which symptom terms actually appear in complaint
# text and how often, supporting that framing with real numbers.
print("[7/8] Symptom keyword frequency...")
text_lower = df_311["descriptor"].fillna("").str.lower()
symptom_counts = {kw: text_lower.str.contains(kw, regex=False).sum() for kw in SYMPTOM_KEYWORDS}
symptom_series = pd.Series(symptom_counts).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=symptom_series.values, y=symptom_series.index, ax=ax, color="#e67e22")
ax.set_title("Symptom Keyword Mentions in Complaint Text")
ax.set_xlabel("Count")
plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_symptom_keywords.png", dpi=150)
plt.close(fig)
total_symptom_mentions = (text_lower.apply(lambda t: any(k in t for k in SYMPTOM_KEYWORDS))).sum()
print(f"  {total_symptom_mentions:,} complaints ({total_symptom_mentions/len(df_311):.1%}) "
      f"mention at least one symptom keyword -- relevant if you frame clustering as "
      f"syndromic surveillance in your literature review.")


# ── 8. Geospatial scatter of complaints ──────────────────────────
print("[8/8] Geospatial scatter...")
geo = df_311.dropna(subset=["latitude", "longitude"])
if len(geo):
    fig, ax = plt.subplots(figsize=(7, 8))
    boroughs = geo["borough"].unique()
    palette = sns.color_palette("Set2", len(boroughs))
    for boro, color in zip(boroughs, palette):
        sub = geo[geo["borough"] == boro]
        ax.scatter(sub["longitude"], sub["latitude"], s=2, alpha=0.3, label=boro, color=color)
    ax.set_title("Complaint Locations by Borough")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(markerscale=8)
    plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_geo_scatter.png", dpi=150)
    plt.close(fig)

print("\nExtended EDA complete. New figures saved to data/processed/:")
print("  fig_borough_month_heatmap.png, fig_top_descriptors.png, fig_grade_distribution.png,")
print("  fig_cuisine_distribution.png, fig_day_of_week.png, fig_complaint_inspection_lag.png,")
print("  fig_symptom_keywords.png, fig_geo_scatter.png")
