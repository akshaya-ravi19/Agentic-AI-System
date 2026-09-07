"""
notebooks/04_bilstm/04_reframed_classifier.py
----------------------------------------------
Option 1: Reframed Classifier - Complaint Prioritisation Ranker

Trains and compares FOUR models on the reframed ground truth
(text-derivable labels from FDA/CDC taxonomy):

  1. Majority-class baseline
  2. TF-IDF + Logistic Regression  (fast interpretable baseline)
  3. SentenceTransformer + BiLSTM + SMOTE
  4. SentenceTransformer + BiLSTM + class weighting

PRIMARY METRICS: PR-AUC, severe-class recall, severe-class F1

The key hypothesis: because the labels now COME FROM the complaint
text itself (not from a future inspection outcome), the text-based
classifier should be able to learn meaningful patterns.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, precision_recall_curve,
                             auc, roc_auc_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import (DATA_LABELLED, MODELS_DIR, EVAL_DIR,
                            EMBEDDING_MODEL, EMBEDDING_DIM,
                            RANDOM_SEED, BILSTM_UNITS, BILSTM_DROPOUT,
                            BILSTM_LR, BILSTM_EPOCHS, BILSTM_BATCH_SIZE,
                            BILSTM_PATIENCE)

# ── Load reframed labelled data ───────────────────────────────────
data_file = DATA_LABELLED / "labelled_complaints_reframed.csv"
if not data_file.exists():
    print(f"ERROR: {data_file} not found. Run 03_reframed_ground_truth.py first.")
    sys.exit(1)

df = pd.read_csv(data_file).dropna(subset=["complaint_text"]).reset_index(drop=True)
print(f"Loaded {len(df):,} reframed complaints")
print(f"Priority label rate: {df['priority_label'].mean():.1%}")
print(f"Hazard tier breakdown:\n{df['hazard_tier'].value_counts().sort_index()}")

X_text = df["complaint_text"].astype(str).tolist()
y = df["priority_label"].values.astype(int)

# ── Stratified train / val / test split ──────────────────────────
X_tmp, X_test_txt, y_tmp, y_test = train_test_split(
    X_text, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15, random_state=RANDOM_SEED, stratify=y_tmp)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train_txt)} | Val: {len(X_val_txt)} | Test: {len(X_test_txt)}")
print(f"  Train positive rate: {y_train.mean():.1%} | Test positive rate: {y_test.mean():.1%}")

# ── Evaluation helper ─────────────────────────────────────────────
results = []
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_true, y_pred,
          target_names=["Administrative (0)","Hazard (1)"], zero_division=0))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    f1_haz = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    haz_recall = (y_pred[y_true==1]==1).mean() if (y_true==1).any() else 0.0
    haz_prec   = (y_true[y_pred==1]==1).mean() if (y_pred==1).any() else 0.0
    print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc:.4f}")
    print(f"Hazard recall: {haz_recall:.4f} | Hazard precision: {haz_prec:.4f} | F1: {f1_haz:.4f}")
    results.append({
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc, 4),
        "hazard_recall": round(haz_recall, 4),
        "hazard_precision": round(haz_prec, 4),
        "hazard_f1": round(f1_haz, 4),
    })

# ══════════════════════════════════════════════════════════════════
# MODEL 1: Majority-class baseline
# ══════════════════════════════════════════════════════════════════
maj_pred = np.ones_like(y_test) if y_test.mean() > 0.5 else np.zeros_like(y_test)
maj_prob = np.full_like(y_test, y_test.mean(), dtype=float)
evaluate("Majority-class baseline", y_test, maj_pred, maj_prob)

# ══════════════════════════════════════════════════════════════════
# MODEL 2: TF-IDF + Logistic Regression (fast interpretable baseline)
# ══════════════════════════════════════════════════════════════════
print("\nTraining TF-IDF + Logistic Regression...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_tr_tfidf = tfidf.fit_transform(X_train_txt)
X_te_tfidf = tfidf.transform(X_test_txt)

lr_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                               class_weight="balanced").fit(X_tr_tfidf, y_train)
evaluate("TF-IDF + Logistic Regression",
         y_test, lr_tfidf.predict(X_te_tfidf),
         lr_tfidf.predict_proba(X_te_tfidf)[:, 1])

# Top TF-IDF features
feature_names = tfidf.get_feature_names_out()
coefs = lr_tfidf.coef_[0]
top_hazard = [(feature_names[i], round(coefs[i], 3)) for i in np.argsort(coefs)[-15:][::-1]]
top_admin  = [(feature_names[i], round(coefs[i], 3)) for i in np.argsort(coefs)[:15]]
print("\nTop 15 features -> HAZARD class:")
for feat, w in top_hazard:
    print(f"  {feat:<30} weight={w}")
print("\nTop 15 features -> ADMINISTRATIVE class:")
for feat, w in top_admin:
    print(f"  {feat:<30} weight={w}")

# ══════════════════════════════════════════════════════════════════
# SENTENCE EMBEDDINGS (shared by models 3 & 4)
# ══════════════════════════════════════════════════════════════════
print(f"\nGenerating sentence embeddings with {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
X_train_emb = encoder.encode(X_train_txt, batch_size=128, show_progress_bar=True)
X_val_emb   = encoder.encode(X_val_txt,   batch_size=128, show_progress_bar=True)
X_test_emb  = encoder.encode(X_test_txt,  batch_size=128, show_progress_bar=True)

# LR on embeddings (direct comparison with original notebook's LR)
lr_emb = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                              class_weight="balanced").fit(X_train_emb, y_train)
evaluate("SentenceTransformer + Logistic Regression",
         y_test, lr_emb.predict(X_test_emb),
         lr_emb.predict_proba(X_test_emb)[:, 1])

# SMOTE on embeddings
cws = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = {int(c): w for c, w in zip(np.unique(y_train), cws)}
try:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, y_train.sum() - 1))
    X_train_sm, y_train_sm = smote.fit_resample(X_train_emb, y_train)
    print(f"\nSMOTE: {len(X_train_emb)} -> {len(X_train_sm)} rows")
except Exception as e:
    print(f"SMOTE skipped ({e}), using original training set")
    X_train_sm, y_train_sm = X_train_emb, y_train

# BiLSTM architecture
def build_bilstm(input_dim=EMBEDDING_DIM):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Bidirectional(layers.LSTM(BILSTM_UNITS, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(BILSTM_LR),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")])
    return m

es = keras.callbacks.EarlyStopping(patience=BILSTM_PATIENCE, restore_best_weights=True,
                                    monitor="val_pr_auc", mode="max")

Xtr_sm = X_train_sm.reshape(-1, 1, EMBEDDING_DIM)
Xtr    = X_train_emb.reshape(-1, 1, EMBEDDING_DIM)
Xv     = X_val_emb.reshape(-1, 1, EMBEDDING_DIM)
Xte    = X_test_emb.reshape(-1, 1, EMBEDDING_DIM)

# ══════════════════════════════════════════════════════════════════
# MODEL 3: BiLSTM + SMOTE
# ══════════════════════════════════════════════════════════════════
print("\nTraining BiLSTM + SMOTE on reframed labels...")
m_smote = build_bilstm()
m_smote.fit(Xtr_sm, y_train_sm, validation_data=(Xv, y_val),
            epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE,
            callbacks=[es], verbose=1)
p_smote = m_smote.predict(Xte).flatten()
evaluate("BiLSTM + SMOTE (reframed)", y_test, (p_smote >= 0.5).astype(int), p_smote)

MODELS_DIR.joinpath("bilstm").mkdir(parents=True, exist_ok=True)
m_smote.save(str(MODELS_DIR / "bilstm" / "bilstm_reframed_smote.keras"))

# ══════════════════════════════════════════════════════════════════
# MODEL 4: BiLSTM + class weighting
# ══════════════════════════════════════════════════════════════════
print("\nTraining BiLSTM + class weighting on reframed labels...")
m_cw = build_bilstm()
m_cw.fit(Xtr, y_train, validation_data=(Xv, y_val),
         class_weight=cw_dict,
         epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE,
         callbacks=[es], verbose=1)
p_cw = m_cw.predict(Xte).flatten()
evaluate("BiLSTM + Class Weighting (reframed)", y_test, (p_cw >= 0.5).astype(int), p_cw)
m_cw.save(str(MODELS_DIR / "bilstm" / "bilstm_reframed_classweight.keras"))

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
print(f"\n{'='*55}")
print("REFRAMED CLASSIFIER — FINAL COMPARISON")
print("(compare these against comparison.csv from original notebook 04)")
print(f"{'='*55}")
print(df_res.to_string(index=False))

EVAL_DIR.joinpath("classifier").mkdir(parents=True, exist_ok=True)
out_path = EVAL_DIR / "classifier" / "comparison_reframed.csv"
df_res.to_csv(out_path, index=False)
print(f"\nResults saved -> {out_path}")

best = df_res.loc[df_res["pr_auc"].idxmax(), "model"]
print(f"\nBest model by PR-AUC: {best}")
print("\nKEY COMPARISON vs original notebook 04:")
print("  Original BiLSTM + SMOTE     PR-AUC: 0.0284 | Recall: 0.167")
print("  Original BiLSTM + CW        PR-AUC: 0.0398 | Recall: 0.167")
print(f"  Reframed best: see above")
print("\nIf reframed PR-AUC >> 0.04, this confirms the original failure")
print("was label/text decoupling, NOT a fundamental architecture problem.")
