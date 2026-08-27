"""
notebooks/04_bilstm/04c_hybrid_classifier.py
--------------------------------------------------
Builds a HYBRID classifier: text embedding (384-dim) concatenated
with two establishment-history features that 04b showed carry real
signal (days_since_last_inspection: ROC-AUC 0.74 standalone,
prior_violation_count: ROC-AUC 0.68 standalone) -- versus every
text-only model in 04's comparison.csv scoring far BELOW even the
trivial majority baseline (0.03-0.04 vs 0.52).

This is a genuinely different input representation from notebook 04,
not just a rerun -- so it's built as a self-contained script that
recomputes everything from labelled_complaints.csv directly, rather
than reusing notebook 02's saved embedding-only splits.

IMPORTANT CAVEAT TO REPORT HONESTLY IN YOUR DISSERTATION: with severe
prevalence around 3% of ~1,458 rows, that's roughly 30-40 positive
examples total, and a single test split leaves single digits of
severe cases in the test set. Metrics computed on that few positive
test examples are HIGH VARIANCE -- one or two cases flipping changes
recall/precision by 10-20 percentage points. Report this explicitly
as a limitation; consider it suggestive evidence, not a precise
estimate, and note that a larger complaint volume (more ingestion
history accumulating over time) would substantially firm this up.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, precision_recall_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import (DATA_LABELLED, DATA_RAW, DATA_SPLITS, MODELS_DIR, EVAL_DIR,
                            EMBEDDING_MODEL, EMBEDDING_DIM, RANDOM_SEED,
                            BILSTM_UNITS, BILSTM_DROPOUT, BILSTM_LR, BILSTM_EPOCHS,
                            BILSTM_BATCH_SIZE, BILSTM_PATIENCE)

MODELS_DIR.joinpath("bilstm").mkdir(parents=True, exist_ok=True)
EVAL_DIR.joinpath("classifier").mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv", parse_dates=["created_date"])
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"],
                        dtype={"camis": str}, low_memory=False)
df["matched_camis"] = df["matched_camis"].astype(str)
df = df.dropna(subset=["descriptor", "label"]).reset_index(drop=True)
print(f"Loaded {len(df):,} labelled complaints, severe: {df['label'].mean():.1%}")

# ── Establishment-history features (same logic as 04b) ───────────
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

# ── Text embeddings ────────────────────────────────────────────
print(f"Encoding text with {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
text_embeddings = encoder.encode(df["descriptor"].astype(str).tolist(), show_progress_bar=True)

# ── Scale numeric features separately (very different scale from
# embeddings, which are roughly unit-normed) before concatenating ──
numeric_features = df[["prior_violation_count", "days_since_last_inspection"]].values
numeric_scaled = MinMaxScaler().fit_transform(numeric_features)

X_hybrid = np.hstack([text_embeddings, numeric_scaled])
y = df["label"].values.astype(int)
HYBRID_DIM = X_hybrid.shape[1]
print(f"Hybrid feature dim: {HYBRID_DIM} ({EMBEDDING_DIM} text + {numeric_scaled.shape[1]} numeric)")

# ── Split (stratified -- essential given how rare severe is) ──────
X_train, X_test, y_train, y_test = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
)
print(f"Train: {len(X_train)} (severe: {y_train.sum()}) | "
      f"Val: {len(X_val)} (severe: {y_val.sum()}) | "
      f"Test: {len(X_test)} (severe: {y_test.sum()})")
if y_test.sum() < 5:
    print("WARNING: fewer than 5 severe examples in the test set -- metrics below "
          "will be very high-variance. See the caveat in this file's docstring.")

# ── Balancing: severe is rare again (~3%), so SMOTE/class-weighting
# are back to their traditional minority-oversampling role ─────────
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {int(c): w for c, w in zip(np.unique(y_train), class_weights)}

try:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, y_train.sum() - 1))
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(X_train)} -> {len(X_train_sm)} rows")
except ValueError as e:
    print(f"SMOTE failed ({e}) -- too few severe examples to oversample. "
          f"Falling back to class-weighting only.")
    X_train_sm, y_train_sm = X_train, y_train


def evaluate(name, y_true, y_pred, y_prob, results_list):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["non-severe", "severe"], zero_division=0))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    sr = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0
    sp = (y_true[y_pred == 1] == 1).mean() if (y_pred == 1).any() else 0.0
    print(f"PR-AUC: {pr_auc:.4f} | Severe recall: {sr:.4f} | Severe precision: {sp:.4f}")
    results_list.append({"model": name, "pr_auc": round(pr_auc, 4),
                          "severe_recall": round(sr, 4), "severe_precision": round(sp, 4)})


results = []

# ── Baseline: majority-class, for reference against text-only run ─
maj_pred = np.zeros_like(y_test)
maj_prob = np.zeros_like(y_test, dtype=float)
evaluate("Majority-class baseline", y_test, maj_pred, maj_prob, results)

# ── Logistic Regression on hybrid features ────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                         class_weight="balanced").fit(X_train, y_train)
evaluate("Logistic Regression (hybrid features)", y_test,
         lr.predict(X_test), lr.predict_proba(X_test)[:, 1], results)


def build_hybrid_bilstm():
    inp = keras.Input(shape=(1, HYBRID_DIM))
    x = layers.Bidirectional(layers.LSTM(BILSTM_UNITS, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(BILSTM_LR), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")])
    return m

es = keras.callbacks.EarlyStopping(patience=BILSTM_PATIENCE, restore_best_weights=True,
                                    monitor="val_pr_auc", mode="max")

Xtr_sm = X_train_sm.reshape(-1, 1, HYBRID_DIM)
Xv = X_val.reshape(-1, 1, HYBRID_DIM)
Xte = X_test.reshape(-1, 1, HYBRID_DIM)
Xtr = X_train.reshape(-1, 1, HYBRID_DIM)

print("\nTraining Hybrid BiLSTM + SMOTE...")
m_sm = build_hybrid_bilstm()
m_sm.fit(Xtr_sm, y_train_sm, validation_data=(Xv, y_val), epochs=BILSTM_EPOCHS,
         batch_size=BILSTM_BATCH_SIZE, callbacks=[es], verbose=1)
p_sm = m_sm.predict(Xte).flatten()
evaluate("Hybrid BiLSTM + SMOTE", y_test, (p_sm >= 0.5).astype(int), p_sm, results)
m_sm.save(str(MODELS_DIR / "bilstm" / "bilstm_hybrid_smote.keras"))

print("\nTraining Hybrid BiLSTM + Class Weighting...")
m_cw = build_hybrid_bilstm()
m_cw.fit(Xtr, y_train, validation_data=(Xv, y_val), class_weight=class_weight_dict,
         epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE, callbacks=[es], verbose=1)
p_cw = m_cw.predict(Xte).flatten()
evaluate("Hybrid BiLSTM + Class Weighting", y_test, (p_cw >= 0.5).astype(int), p_cw, results)
m_cw.save(str(MODELS_DIR / "bilstm" / "bilstm_hybrid_classweight.keras"))

df_res = pd.DataFrame(results)
print(f"\n{'='*55}\nHYBRID MODEL COMPARISON\n{'='*55}")
print(df_res.to_string(index=False))
df_res.to_csv(EVAL_DIR / "classifier" / "comparison_hybrid.csv", index=False)

best = df_res.loc[df_res["pr_auc"].idxmax(), "model"]
print(f"\nBest hybrid model by PR-AUC: {best}")
print(f"\nCompare this table against comparison.csv (text-only) from notebook 04 --")
print(f"if hybrid PR-AUC now exceeds the majority baseline's {results[0]['pr_auc']:.4f} "
      f"(text-only models did not), that's strong evidence establishment-history "
      f"features were the missing piece, not a fundamentally unlearnable label.")