# ============================================================
# NOTEBOOK 04 — BiLSTM Severity Classifier
# Trains 4 models on identical data splits:
#   1. Majority-class baseline (always predicts non-severe)
#   2. Logistic Regression on embeddings (ML baseline)
#   3. BiLSTM + SMOTE
#   4. BiLSTM + class weighting
# Evaluates ALL on the same unmodified test set.
# PRIMARY METRICS: severe-class recall and PR-AUC
# ============================================================
import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
    precision_recall_curve, auc, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

# Load all splits
X_train = np.load(DATA_SPLITS/"X_train.npy"); y_train = np.load(DATA_SPLITS/"y_train.npy")
X_val   = np.load(DATA_SPLITS/"X_val.npy");   y_val   = np.load(DATA_SPLITS/"y_val.npy")
X_test  = np.load(DATA_SPLITS/"X_test.npy");  y_test  = np.load(DATA_SPLITS/"y_test.npy")
X_tr_sm = np.load(DATA_SPLITS/"X_train_smote.npy")
y_tr_sm = np.load(DATA_SPLITS/"y_train_smote.npy")
cw      = json.loads((DATA_SPLITS/"class_weights.json").read_text())
EVAL_DIR.joinpath("classifier").mkdir(parents=True, exist_ok=True)
MODELS_DIR.joinpath("bilstm").mkdir(parents=True, exist_ok=True)
# Sanity check: with the old "any Critical violation" severity rule,
# this printed 88% severe, which meant the majority-class baseline
# scored a HIGHER PR-AUC (0.94) than every real trained model --
# PR-AUC of a constant classifier tracks positive-class prevalence,
# so a near-total prevalence in one class makes it a misleading
# ranking metric. Severity was redefined in notebook 03 (grade C /
# closure, not "any Critical violation") specifically to fix this --
# if this number still shows >80% or <20% severe, something in that
# redefinition didn't take effect and these results shouldn't be
# trusted yet.
print(f"Test set: {len(X_test):,} | severe: {y_test.mean():.1%}")

# ── Evaluation helper ─────────────────────────────────────────
results = []
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["non-severe","severe"]))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    sr = (y_pred[y_true==1]==1).mean() if (y_true==1).any() else 0.0
    sp = (y_true[y_pred==1]==1).mean() if (y_pred==1).any() else 0.0
    print(f"PR-AUC:         {pr_auc:.4f}")
    print(f"Severe recall:  {sr:.4f}  ← PRIMARY METRIC")
    print(f"Severe prec:    {sp:.4f}")
    results.append({"model":name,"pr_auc":round(pr_auc,4),
                    "severe_recall":round(sr,4),"severe_precision":round(sp,4)})
    return y_prob

# ── Model 1: Majority-class baseline ─────────────────────────
maj_pred = np.zeros_like(y_test)
maj_prob = np.zeros_like(y_test, dtype=float)
evaluate("Majority-class baseline", y_test, maj_pred, maj_prob)

# ── Model 2: Logistic Regression ─────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_train, y_train)
evaluate("Logistic Regression (ML baseline)", y_test,
         lr.predict(X_test), lr.predict_proba(X_test)[:,1])

# ── BiLSTM architecture ───────────────────────────────────────
# build_bilstm takes an optional loss function so the SAME
# architecture can be trained with plain binary crossentropy
# (class-weighting variant) or focal loss (focal-loss variant) --
# keeping the network identical isolates the comparison to the loss
# function itself, which is the actual thing being compared here.
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss down-weights easy/well-classified examples so the
    model spends more learning capacity on hard, often-minority-class
    examples -- an alternative to SMOTE/class-weighting for handling
    imbalance, compared empirically here rather than assumed superior.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss_fn


def build_bilstm(loss="binary_crossentropy"):
    inp = keras.Input(shape=(1, EMBEDDING_DIM))
    x = layers.Bidirectional(layers.LSTM(BILSTM_UNITS, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(BILSTM_LR),
              loss=loss,
              metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")])
    return m

es = keras.callbacks.EarlyStopping(
    patience=BILSTM_PATIENCE, restore_best_weights=True,
    monitor="val_pr_auc", mode="max")

# Reshape: (samples, timesteps=1, features)
Xtr  = X_train.reshape(-1,1,EMBEDDING_DIM)
Xtrs = X_tr_sm.reshape(-1,1,EMBEDDING_DIM)
Xv   = X_val.reshape(-1,1,EMBEDDING_DIM)
Xte  = X_test.reshape(-1,1,EMBEDDING_DIM)

# ── Model 3: BiLSTM + SMOTE ───────────────────────────────────
print("\nTraining BiLSTM + SMOTE...")
m_smote = build_bilstm()
m_smote.fit(Xtrs, y_tr_sm, validation_data=(Xv,y_val),
            epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE,
            callbacks=[es], verbose=1)
p_smote = m_smote.predict(Xte).flatten()
evaluate("BiLSTM + SMOTE", y_test, (p_smote>=0.5).astype(int), p_smote)
m_smote.save(str(MODELS_DIR/"bilstm"/"bilstm_smote.keras"))

# ── Model 4: BiLSTM + class weighting ────────────────────────
print("\nTraining BiLSTM + class weighting...")
m_cw = build_bilstm()
m_cw.fit(Xtr, y_train, validation_data=(Xv,y_val),
         class_weight={int(k):v for k,v in cw.items()},
         epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE,
         callbacks=[es], verbose=1)
p_cw = m_cw.predict(Xte).flatten()
evaluate("BiLSTM + Class Weighting", y_test, (p_cw>=0.5).astype(int), p_cw)
m_cw.save(str(MODELS_DIR/"bilstm"/"bilstm_classweight.keras"))

# ── Model 5: BiLSTM + focal loss ─────────────────────────────
# Trained on the UNMODIFIED training split (no SMOTE, no explicit
# class weights) -- focal loss handles imbalance through the loss
# function itself, so combining it with SMOTE/class-weighting would
# confound which mechanism is doing the work. Compared on equal
# footing against the other two imbalance-handling approaches.
print("\nTraining BiLSTM + focal loss...")
m_focal = build_bilstm(loss=focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA))
m_focal.fit(Xtr, y_train, validation_data=(Xv,y_val),
            epochs=BILSTM_EPOCHS, batch_size=BILSTM_BATCH_SIZE,
            callbacks=[es], verbose=1)
p_focal = m_focal.predict(Xte).flatten()
evaluate("BiLSTM + Focal Loss", y_test, (p_focal>=0.5).astype(int), p_focal)
m_focal.save(str(MODELS_DIR/"bilstm"/"bilstm_focal.keras"))

# ── Summary ───────────────────────────────────────────────────
df_res = pd.DataFrame(results)
print(f"\n{'='*55}\nFINAL COMPARISON\n{'='*55}")
print(df_res.to_string(index=False))
df_res.to_csv(EVAL_DIR/"classifier"/"comparison.csv", index=False)

# NOTE: picking the "best" model by severe_recall ALONE is a trap --
# a model that predicts "severe" for every single input gets 100%
# recall but is useless (near-zero precision, floods inspectors with
# false alarms). PR-AUC balances both sides of that trade-off, so we
# rank by PR-AUC first and report recall/precision as context.
best = df_res.loc[df_res["pr_auc"].idxmax(), "model"]
print(f"\nBest model by PR-AUC (balances recall + precision): {best}")
print("(severe_recall alone is not a safe selection criterion -- see comment above)")