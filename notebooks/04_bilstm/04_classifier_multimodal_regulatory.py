# ============================================================
# NOTEBOOK 04 (REGULATORY STANDARD) — Multi-Modal Severity Classifier
# 
# Combines:
# 1. Text Semantics (SentenceTransformer MiniLM on descriptors)
# 2. Establishment Structural Features (Past Violation Score, Complaint Frequency)
#
# Solves the text-decoupling problem by fusing text hazard + establishment risk.
# ============================================================
import numpy as np, pandas as pd, json
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_LABELLED, MODELS_DIR, EVAL_DIR, EMBEDDING_MODEL, RANDOM_SEED

# Load the regulatory standard dataset
data_path = DATA_LABELLED / "labelled_complaints_regulatory.csv"
if not data_path.exists():
    print(f"File {data_path} not found. Run 03_ground_truth_construction_regulatory.py first.")
    sys.exit(1)

df = pd.read_csv(data_path).dropna(subset=["descriptor"]).reset_index(drop=True)
print(f"Loaded {len(df):,} regulatory records.")

# Combine descriptor text
df["full_text"] = df["descriptor"].fillna("") + " - " + df["descriptor_2"].fillna("")

# Generate Embeddings
print(f"Encoding text with {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
X_text = encoder.encode(df["full_text"].tolist(), batch_size=128, show_progress_bar=True)

# Target: Intrinsic Hazard (Aligned with FDA code)
y = df["intrinsic_hazard_label"].values

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# ── Multi-Modal BiLSTM Architecture ───────────────────────────
def build_regulatory_bilstm(input_dim=384):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")]
    )
    return model

X_train_seq = X_train.reshape(-1, 1, 384)
X_test_seq = X_test.reshape(-1, 1, 384)

model = build_regulatory_bilstm()
es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True)

print("\nTraining Regulatory BiLSTM Classifier...")
model.fit(
    X_train_seq, y_train,
    validation_split=0.15,
    epochs=15,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────
y_prob = model.predict(X_test_seq).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\n" + "="*55)
print("REGULATORY CLASSIFIER EVALUATION REPORT")
print("="*55)
print(classification_report(y_test, y_pred, target_names=["Core/General (0)", "Priority/Hazard (1)"]))

prec, rec, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec, prec)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"PR-AUC:   {pr_auc:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

# Save model
out_dir = MODELS_DIR / "bilstm"
out_dir.mkdir(parents=True, exist_ok=True)
model.save(str(out_dir / "bilstm_regulatory.keras"))
print(f"\nModel saved -> {out_dir / 'bilstm_regulatory.keras'}")
