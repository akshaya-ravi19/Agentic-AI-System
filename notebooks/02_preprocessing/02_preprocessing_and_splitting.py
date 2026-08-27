# ============================================================
# NOTEBOOK 02 — Preprocessing, Embedding, Splitting, Balancing
# Run AFTER notebook 03 (needs labelled_complaints.csv).
# CRITICAL RULE: Split FIRST, balance training fold ONLY.
# ============================================================
import pandas as pd, numpy as np, re, spacy, nltk
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

nltk.download("stopwords", quiet=True); nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

spacy_nlp  = spacy.load(SPACY_MODEL)
lemmatizer = WordNetLemmatizer()
STOPS      = set(stopwords.words("english"))

df = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv")
print(f"Loaded: {len(df):,} | severe: {df['label'].mean():.1%}")
DATA_SPLITS.mkdir(parents=True, exist_ok=True)

# ── PII scrubbing ────────────────────────────────────────────
def scrub_pii(text):
    if not isinstance(text, str): return ""
    doc = spacy_nlp(text)
    for ent in doc.ents:
        if ent.label_ in PII_ENTITIES:
            text = text.replace(ent.text, f"[{ent.label_}]")
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    return text

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = scrub_pii(text).lower()
    text = re.sub(r'[^a-z\s\[\]]', ' ', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOPS and len(t) > 2]
    return ' '.join(tokens[:MAX_TEXT_LENGTH])

print("Cleaning text...")
df["text_clean"] = df["descriptor"].apply(clean_text)
df = df[df["text_clean"].str.len() > 10].reset_index(drop=True)
print(f"After cleaning: {len(df):,}")

# ── Sentence-Transformer embeddings ──────────────────────────
print(f"Generating embeddings with {EMBEDDING_MODEL}...")
encoder    = SentenceTransformer(EMBEDDING_MODEL)
embeddings = encoder.encode(df["text_clean"].tolist(),
                             batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True)
print(f"Embeddings: {embeddings.shape}")
np.save(DATA_PROCESSED / "embeddings.npy", embeddings)

# ── Train / val / test split (BEFORE any balancing) ──────────
labels = df["label"].values
X_temp, X_test, y_temp, y_test = train_test_split(
    embeddings, labels, test_size=TEST_RATIO,
    random_state=RANDOM_SEED, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
    random_state=RANDOM_SEED, stratify=y_temp)

print(f"\nSplit summary:")
print(f"  Train: {len(X_train):,} | severe: {y_train.mean():.1%}")
print(f"  Val:   {len(X_val):,}   | severe: {y_val.mean():.1%}")
print(f"  Test:  {len(X_test):,}  | severe: {y_test.mean():.1%}")

# ── Compare three balancing strategies on TRAIN ONLY ─────────
print("\n=== Balancing comparison (training fold only) ===")

# A: SMOTE
smote = SMOTE(sampling_strategy=SMOTE_STRATEGY,
              k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
X_tr_smote, y_tr_smote = smote.fit_resample(X_train, y_train)
print(f"A) SMOTE:           {len(X_tr_smote):,} samples | severe: {y_tr_smote.mean():.1%}")

# B: Random oversampling
ros = RandomOverSampler(random_state=RANDOM_SEED)
X_tr_ros, y_tr_ros = ros.fit_resample(X_train, y_train)
print(f"B) Random oversample: {len(X_tr_ros):,} samples | severe: {y_tr_ros.mean():.1%}")

# C: Class weighting (no resampling — weight applied during model training)
w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw = {0: float(w[0]), 1: float(w[1])}
print(f"C) Class weights:   weight_0={cw[0]:.3f}, weight_1={cw[1]:.3f}")
print("   (pass class_weight=cw to model.fit — no data augmentation)")

# ── Save all splits ───────────────────────────────────────────
for name, arr in [("X_train",X_train),("X_val",X_val),("X_test",X_test),
                   ("y_train",y_train),("y_val",y_val),("y_test",y_test),
                   ("X_train_smote",X_tr_smote),("y_train_smote",y_tr_smote),
                   ("X_train_ros",X_tr_ros),("y_train_ros",y_tr_ros)]:
    np.save(DATA_SPLITS / f"{name}.npy", arr)
import json
(DATA_SPLITS / "class_weights.json").write_text(json.dumps(cw))
print(f"\nAll splits saved to {DATA_SPLITS}")