"""
notebooks/04_bilstm/04d_xgboost_hybrid_cv.py
--------------------------------------------------
Rectification of the BiLSTM results, per the plan discussed in chat.
CHANGES vs. earlier notebooks: model class (XGBoost instead of
BiLSTM) and evaluation methodology (5-fold stratified cross-
validation instead of one train/test split). Ground truth is
UNCHANGED -- same labelled_complaints.csv, same grade-C/closure
severity definition as everywhere else in the project.

WHY XGBOOST:
- Well-suited to small-n tabular+embedding data (you have ~30-40
  severe examples total -- a deep net has very little to learn from).
- Removes TensorFlow from the classifier's dependency chain entirely,
  which also addresses the Cloud Run deployment crashes (segfaults,
  slow cold starts) that were specifically caused by TF+sentence-
  transformers loading together in a memory/time-constrained container.

WHY 5-FOLD CROSS-VALIDATION INSTEAD OF ONE SPLIT:
A single 80/20 split leaves only ~6-9 severe examples in the test
set -- one flipped prediction swings recall or precision by 10-20
percentage points. Cross-validation reports a mean +/- std across 5
different splits, which is a far more honest and stable estimate
given how little positive data exists. Report the mean +/- std in
your dissertation, not a single number -- that itself is worth a
sentence in your methodology chapter explaining why.

WHAT'S COMPARED:
Three ways of handling the imbalance, each with XGBoost:
  1. scale_pos_weight (XGBoost's built-in reweighting, no resampling)
  2. SMOTE (synthetic minority oversampling)
  3. RandomOverSampler (duplicates real minority examples -- the
     method flagged as missing from the original balancing comparison)
All three are applied ONLY within each fold's training data, never
to validation data, to avoid leakage.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import (DATA_LABELLED, DATA_RAW, MODELS_DIR, EVAL_DIR,
                            EMBEDDING_MODEL, EMBEDDING_DIM, RANDOM_SEED)

MODELS_DIR.joinpath("xgboost").mkdir(parents=True, exist_ok=True)
EVAL_DIR.joinpath("classifier").mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv", parse_dates=["created_date"])
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"],
                        dtype={"camis": str}, low_memory=False)
df["matched_camis"] = df["matched_camis"].astype(str)
df = df.dropna(subset=["descriptor", "label"]).reset_index(drop=True)
n_severe = int(df["label"].sum())
print(f"Loaded {len(df):,} labelled complaints, severe: {df['label'].mean():.1%} (n={n_severe})")

if n_severe < 15:
    print(f"WARNING: only {n_severe} severe examples total. 5-fold CV means ~{n_severe//5} "
          f"severe cases per fold -- results will still be high-variance, just less so than "
          f"a single split. Report this honestly as a data-volume limitation.")

# ── Establishment-history features (same as 04b/04c) ─────────────
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

print("Computing establishment-history features (loops over all rows, ~1-2 min)...")
df["prior_violation_count"] = df.apply(prior_violation_count, axis=1)
df["days_since_last_inspection"] = df.apply(days_since_last_inspection, axis=1)
df["days_since_last_inspection"] = df["days_since_last_inspection"].fillna(
    df["days_since_last_inspection"].median()
)

print(f"Encoding text with {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
text_embeddings = encoder.encode(df["descriptor"].astype(str).tolist(), show_progress_bar=True)

numeric_features = df[["prior_violation_count", "days_since_last_inspection"]].values
y = df["label"].values.astype(int)

HYBRID_DIM = text_embeddings.shape[1] + numeric_features.shape[1]
print(f"Hybrid feature dim: {HYBRID_DIM}")


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="aucpr", random_state=RANDOM_SEED,
        use_label_encoder=False,
    )


def pr_auc_score(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return auc(rec, prec)


BALANCING_METHODS = ["scale_pos_weight", "smote", "random_oversample"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

results = {m: {"pr_auc": [], "recall": [], "precision": []} for m in BALANCING_METHODS}

print(f"\n{'='*55}\n5-FOLD CROSS-VALIDATION\n{'='*55}")
for fold, (train_idx, test_idx) in enumerate(skf.split(text_embeddings, y), 1):
    # Scale numeric features fold-locally (fit on train fold only,
    # to avoid leaking test-fold statistics into the scaler).
    scaler = MinMaxScaler().fit(numeric_features[train_idx])
    num_train = scaler.transform(numeric_features[train_idx])
    num_test = scaler.transform(numeric_features[test_idx])

    X_train = np.hstack([text_embeddings[train_idx], num_train])
    X_test = np.hstack([text_embeddings[test_idx], num_test])
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nFold {fold}: train={len(X_train)} (severe={y_train.sum()}), "
          f"test={len(X_test)} (severe={y_test.sum()})")

    for method in BALANCING_METHODS:
        if method == "scale_pos_weight":
            spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            model = make_xgb()
            model.set_params(scale_pos_weight=spw)
            model.fit(X_train, y_train)
        elif method == "smote":
            k = min(5, y_train.sum() - 1)
            if k < 1:
                print(f"  [{method}] skipped fold {fold}: too few severe examples to SMOTE")
                continue
            X_res, y_res = SMOTE(random_state=RANDOM_SEED, k_neighbors=k).fit_resample(X_train, y_train)
            model = make_xgb().fit(X_res, y_res)
        elif method == "random_oversample":
            X_res, y_res = RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
            model = make_xgb().fit(X_res, y_res)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results[method]["pr_auc"].append(pr_auc_score(y_test, y_prob))
        results[method]["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        results[method]["precision"].append(precision_score(y_test, y_pred, zero_division=0))

print(f"\n{'='*55}\nCROSS-VALIDATED RESULTS (mean +/- std across 5 folds)\n{'='*55}")
summary_rows = []
for method, m in results.items():
    if not m["pr_auc"]:
        continue
    pr_mean, pr_std = np.mean(m["pr_auc"]), np.std(m["pr_auc"])
    rec_mean, rec_std = np.mean(m["recall"]), np.std(m["recall"])
    prec_mean, prec_std = np.mean(m["precision"]), np.std(m["precision"])
    print(f"\n{method}:")
    print(f"  PR-AUC:    {pr_mean:.4f} +/- {pr_std:.4f}")
    print(f"  Recall:    {rec_mean:.4f} +/- {rec_std:.4f}")
    print(f"  Precision: {prec_mean:.4f} +/- {prec_std:.4f}")
    summary_rows.append({
        "model": f"XGBoost + {method}",
        "pr_auc_mean": round(pr_mean, 4), "pr_auc_std": round(pr_std, 4),
        "recall_mean": round(rec_mean, 4), "recall_std": round(rec_std, 4),
        "precision_mean": round(prec_mean, 4), "precision_std": round(prec_std, 4),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(EVAL_DIR / "classifier" / "comparison_xgboost_cv.csv", index=False)
print(f"\nSaved to {EVAL_DIR / 'classifier' / 'comparison_xgboost_cv.csv'}")

if len(df_summary):
    best_method = df_summary.loc[df_summary["pr_auc_mean"].idxmax(), "model"]
    print(f"\nBest by mean PR-AUC: {best_method}")
    print(f"\nCompare against comparison.csv (text-only BiLSTM, PR-AUC ~0.03-0.04) and")
    print(f"comparison_hybrid.csv (single-split hybrid BiLSTM) from earlier notebooks.")
    print(f"If this cross-validated result still doesn't clear a reasonable bar, that's")
    print(f"honest evidence the label itself (kappa=0.025 vs human judgement) is the real")
    print(f"ceiling -- report that as a finding, not a modeling failure to keep chasing.")

# ── Fit final model on ALL data for deployment ─────────────────────
print(f"\n{'='*55}\nFitting final model on full dataset for deployment...")
final_scaler = MinMaxScaler().fit(numeric_features)
num_scaled = final_scaler.transform(numeric_features)
X_full = np.hstack([text_embeddings, num_scaled])
spw_full = (y == 0).sum() / max((y == 1).sum(), 1)
final_model = make_xgb()
final_model.set_params(scale_pos_weight=spw_full)
final_model.fit(X_full, y)

model_path = MODELS_DIR / "xgboost" / "xgb_hybrid_final.json"
final_model.save_model(str(model_path))
print(f"Saved final model to {model_path}")
print("NOTE: this model expects a 386-dim input: 384-dim text embedding + "
      "[prior_violation_count, days_since_last_inspection] scaled with the SAME "
      "MinMaxScaler fit here -- save/reuse that scaler's min/max when serving this "
      "model, or predictions will be silently wrong (same class of bug as the CAMIS "
      "dtype issue -- a scaling mismatch won't error, it'll just quietly mispredict).")
