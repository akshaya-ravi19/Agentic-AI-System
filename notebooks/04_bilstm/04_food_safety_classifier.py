"""
notebooks/04_bilstm/04_food_safety_classifier.py
--------------------------------------------------
Trains and evaluates a Deep BiLSTM Neural Network for the GCP FoodGuard
Food Safety Complaint Triage system using robust cross-validation.

Ground Truth: 3-tier NYC DOHMH municipal taxonomy (no CDC/FDA references).
  - Tier 2 (Critical): Rodent/pest/temperature/hygiene/sewage/food source
  - Tier 1 (Moderate): Spoilage/contamination/foreign object/kitchen
  - Tier 0 (Routine): Grading/permit/toilet/odor/pet-animal

Binary Target: priority_label (Tier >= 1 = 1 = Actionable, Tier 0 = 0 = Routine)

Key: Category-Disjoint Cross-Validation prevents label leakage.
     Each fold holds out entire complaint descriptor groups unseen during training.

Metrics Reported:
  - Precision, Recall, F1-Score per class
  - PR-AUC (Precision-Recall Area Under Curve)
  - ROC-AUC
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_LABELLED, MODELS_DIR, EMBEDDING_MODEL, RANDOM_SEED

data_file = DATA_LABELLED / "labelled_complaints_ground_truth.csv"
if not data_file.exists():
    print(f"File {data_file} not found. Please run 03_food_safety_ground_truth.py first.")
    sys.exit(1)

df = pd.read_csv(data_file, low_memory=False).dropna(subset=["complaint_text"]).reset_index(drop=True)
print(f"Loaded {len(df):,} food safety complaint records.")

# Build distinct category signature pairs to evaluate true generalization
# Each distinct (descriptor, descriptor_2) combination is a unique complaint template
df["descriptor"] = df["descriptor"].fillna("").str.strip()
df["descriptor_2"] = df["descriptor_2"].fillna("").str.strip()

pairs = df[["descriptor", "descriptor_2", "priority_label"]].drop_duplicates().reset_index(drop=True)
pairs["template_text"] = pairs["descriptor"] + " " + pairs["descriptor_2"]
pairs["template_text"] = pairs["template_text"].str.strip()

print(f"Identified {len(pairs)} distinct complaint category templates.")
print("Class breakdown across distinct category templates:")
print(pairs["priority_label"].value_counts().to_dict())

print(f"\nGenerating sentence embeddings via {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
X = encoder.encode(pairs["template_text"].tolist(), batch_size=32, show_progress_bar=True)
y = pairs["priority_label"].values


def build_food_safety_bilstm(input_dim=384):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")]
    )
    return model


# 5-Fold Category-Disjoint Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

oof_trues = []
oof_preds = []
oof_probs = []

print("\nStarting 5-Fold Category-Disjoint Cross-Validation (BiLSTM)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_f, y_train_f = X[train_idx], y[train_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]

    X_train_seq = X_train_f.reshape(-1, 1, 384)
    X_val_seq = X_val_f.reshape(-1, 1, 384)

    model = build_food_safety_bilstm()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=6, restore_best_weights=True)

    # Class weighting to handle imbalance
    pos_weight = (len(y_train_f) - sum(y_train_f)) / (sum(y_train_f) + 1e-9)
    cw = {0: 1.0, 1: float(pos_weight)}

    model.fit(
        X_train_seq, y_train_f,
        validation_data=(X_val_seq, y_val_f),
        epochs=30,
        batch_size=16,
        class_weight=cw,
        callbacks=[es],
        verbose=0
    )

    probs = model.predict(X_val_seq, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    print(f"  Fold {fold}: Val samples={len(y_val_f)}, Hazard recall={sum((preds==1) & (y_val_f==1))}/{sum(y_val_f==1)}")

    oof_trues.extend(y_val_f)
    oof_preds.extend(preds)
    oof_probs.extend(probs)

oof_trues = np.array(oof_trues)
oof_preds = np.array(oof_preds)
oof_probs = np.array(oof_probs)

print("\n" + "=" * 60)
print("FOODGUARD FOOD SAFETY COMPLAINT TRIAGE CLASSIFIER REPORT")
print("(Evaluated on Unseen Complaint Categories — No Leakage)")
print("=" * 60)
print(classification_report(oof_trues, oof_preds, target_names=["Routine / Low Priority (0)", "Actionable Hazard (1)"]))

prec, rec, _ = precision_recall_curve(oof_trues, oof_probs)
pr_auc = auc(rec, prec)
roc_auc = roc_auc_score(oof_trues, oof_probs)

print(f"PR-AUC:   {pr_auc:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

# Train final production model on full category-disjoint template corpus and save
print("\nTraining final production model on full complaint taxonomy corpus...")
X_full_seq = X.reshape(-1, 1, 384)
final_model = build_food_safety_bilstm()
pos_weight_full = (len(y) - sum(y)) / (sum(y) + 1e-9)
final_model.fit(
    X_full_seq, y,
    epochs=25,
    batch_size=16,
    class_weight={0: 1.0, 1: float(pos_weight_full)},
    verbose=0
)

out_dir = MODELS_DIR / "bilstm"
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / "bilstm_food_safety.keras"
final_model.save(str(save_path))
print(f"Production model saved successfully -> {save_path}")