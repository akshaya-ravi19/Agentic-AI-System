"""
notebooks/04_bilstm/04_digital_health_classifier.py
----------------------------------------------------
Trains a Deep BiLSTM Neural Network on Sentence-Transformer embeddings
to classify syndromic public health hazard severity.

Evaluates on a held-out test split with:
- Precision, Recall, F1-Score
- PR-AUC (Precision-Recall Area Under Curve)
- ROC-AUC
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_LABELLED, MODELS_DIR, EMBEDDING_MODEL, RANDOM_SEED

data_file = DATA_LABELLED / "labelled_complaints_digital_health.csv"
if not data_file.exists():
    print(f"File {data_file} not found. Please run 03_digital_health_ground_truth.py first.")
    sys.exit(1)

df = pd.read_csv(data_file).dropna(subset=["descriptor"]).reset_index(drop=True)
print(f"Loaded {len(df):,} digital health records.")

df["complaint_text"] = df["descriptor"].fillna("") + " " + df["descriptor_2"].fillna("")

print(f"Generating sentence embeddings via {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
X = encoder.encode(df["complaint_text"].tolist(), batch_size=128, show_progress_bar=True)
y = df["binary_priority_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Reshape for BiLSTM (samples, timesteps=1, features=384)
X_train_seq = X_train.reshape(-1, 1, 384)
X_test_seq = X_test.reshape(-1, 1, 384)

def build_digital_health_bilstm():
    inp = keras.Input(shape=(1, 384))
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

model = build_digital_health_bilstm()
es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True)

print("\nTraining Digital Health Syndromic BiLSTM...")
model.fit(
    X_train_seq, y_train,
    validation_split=0.15,
    epochs=12,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────
y_prob = model.predict(X_test_seq).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\n" + "="*55)
print("DIGITAL HEALTH SYNDROMIC CLASSIFIER REPORT")
print("="*55)
print(classification_report(y_test, y_pred, target_names=["Administrative/Low (0)", "Syndromic Hazard (1)"]))

prec, rec, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec, prec)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"PR-AUC:   {pr_auc:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

out_dir = MODELS_DIR / "bilstm"
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / "bilstm_digital_health.keras"
model.save(str(save_path))
print(f"\nModel saved successfully -> {save_path}")
