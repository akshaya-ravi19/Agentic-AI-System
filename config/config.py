# ============================================================
# FoodGuard — Central Configuration
# All constants in one place. Change here, propagates everywhere.
# ============================================================
from pathlib import Path

ROOT            = Path(__file__).parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_LABELLED   = ROOT / "data" / "labelled"
DATA_SPLITS     = ROOT / "data" / "splits"
MODELS_DIR      = ROOT / "models"
EVAL_DIR        = ROOT / "evaluation"

# NYC Open Data
NYC_311_API     = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
DOHMH_API       = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
COMPLAINT_TYPE  = "Food Establishment"
START_YEAR      = 2019

# Free Socrata app token -- get one in ~1 minute at
# https://data.cityofnewyork.us/profile/edit (no approval wait).
# Without a token you're on Socrata's unauthenticated rate limit,
# which is slow and prone to exactly the kind of read-timeout you'll
# see on large paginated pulls. Paste your token below, or set the
# SOCRATA_APP_TOKEN environment variable instead of hardcoding it.
import os as _os
SOCRATA_APP_TOKEN = _os.environ.get("SOCRATA_APP_TOKEN", "")

# Severity labelling
LABEL_WINDOW_DAYS        = 30
LABEL_WINDOW_SENSITIVITY = [14, 30, 60]
SEVERE_LABEL             = 1
NON_SEVERE_LABEL         = 0

# Train/val/test split
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

# Preprocessing
SPACY_MODEL      = "en_core_web_sm"
MAX_TEXT_LENGTH  = 512
PII_ENTITIES     = ["PERSON", "PHONE", "EMAIL"]

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# BiLSTM
BILSTM_UNITS      = 128
BILSTM_DROPOUT    = 0.3
BILSTM_EPOCHS     = 20
BILSTM_BATCH_SIZE = 32
BILSTM_LR         = 1e-3
# Focal loss hyperparameters (see notebook 04's focal_loss()).
# gamma controls how much easy examples get down-weighted; alpha
# balances the positive/negative class weighting within the loss
# itself. These are the standard defaults from the original focal
# loss paper (Lin et al. 2017) -- reasonable starting values, not
# tuned specifically for this dataset. If focal loss underperforms
# unexpectedly, trying a couple of alpha values (e.g. matching your
# actual class prevalence) before ruling it out is worth a mention
# in your methodology chapter.
FOCAL_LOSS_GAMMA  = 2.0
FOCAL_LOSS_ALPHA  = 0.25
BILSTM_PATIENCE   = 4

# SMOTE
SMOTE_STRATEGY   = "minority"
SMOTE_K_NEIGHBORS = 5

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE  = 3
HDBSCAN_MIN_SAMPLES       = 2
HDBSCAN_CLUSTER_METHOD    = "eom"
HDBSCAN_WINDOW_DAYS       = 30

# Symptom keywords for illness signal extraction
SYMPTOM_KEYWORDS = [
    "sick","ill","vomit","nausea","diarrhea","diarrhoea",
    "stomach","food poisoning","fever","cramps","poisoned",
    "hospitalized","hospital","threw up"
]

# Food categories for clustering dimension
FOOD_CATEGORIES = {
    "poultry": ["chicken","turkey","duck","poultry"],
    "seafood":  ["fish","seafood","shrimp","sushi","oyster","clam"],
    "dairy":    ["milk","cheese","dairy","butter","cream","yogurt"],
    "produce":  ["vegetable","fruit","salad","lettuce","tomato"],
}

# Agent
# gemini-1.5-flash was retired long ago. gemini-3.5-flash-lite is
# GA as of Aug 2026 with no shutdown date announced -- check
# https://ai.google.dev/gemini-api/docs/deprecations before your
# viva in case this has changed by the time you read this.
GEMINI_MODEL       = "gemini-3.5-flash-lite"
AGENT_MAX_STEPS    = 10
AGENT_TEMPERATURE  = 0.1

# Triage levels
TRIAGE_LOG      = "LOG"
TRIAGE_REVIEW   = "REVIEW"
TRIAGE_ESCALATE = "ESCALATE"

# Violation categories (DOHMH standard)
VIOLATION_CATEGORIES = [
    "pest_infestation","food_temperature","personal_hygiene",
    "food_source","facility_condition","chemical_contamination",
    "cross_contamination","other"
]

# Keyword map used to derive violation_category from free-text
# complaint descriptors (used in notebook 06 clustering). This was
# previously defined as a flat list above with no keywords attached,
# so nothing could actually derive a category from text -- this dict
# is what notebook 06 now uses to do that derivation.
VIOLATION_CATEGORIES_KEYWORDS = {
    "pest_infestation":       ["roach","mice","mouse","rodent","rat","pest","insect","fly","flies"],
    "food_temperature":       ["cold","hot food","temperature","warm","refrigerat","thaw"],
    "personal_hygiene":       ["glove","hand wash","hygiene","hairnet","sick employee","hair in food"],
    "food_source":            ["expired","spoiled","unlabeled","source","recall"],
    "facility_condition":     ["dirty","unsanitary","mold","leak","broken","floor","ceiling"],
    "chemical_contamination": ["chemical","cleaning solution","poison","bleach"],
    "cross_contamination":    ["raw meat","cross contamina","cutting board","utensil"],
}

# GCP
GCP_PROJECT      = "thesis-project-499512"
GCP_REGION       = "us-central1"
BIGQUERY_DATASET = "foodsafety_data"
BQ_COMPLAINTS    = "thesis-project-499512.foodsafety_data.complaints"
# Your labelled research dataset (ground-truth severity labels tied to
# matched_camis) lives in a SEPARATE table from raw 311 ingestion --
# they got split apart after a schema conflict (see chat history).
# The classifier (notebook 04) and agent (notebook 05) both depend on
# the labelled data, so they should read from THIS table, not BQ_COMPLAINTS.
BQ_COMPLAINTS_LABELLED = "thesis-project-499512.foodsafety_data.complaints_labelled_upload"
# NOTE: inspection_history had to be loaded as all-STRING then cleaned
# into a second table via SAFE_CAST/SAFE.PARSE_TIMESTAMP (see chat) --
# _clean is the one with real types, use that one everywhere downstream.
BQ_INSPECTIONS   = "thesis-project-499512.foodsafety_data.inspection_history_clean"
BQ_AGENT_DECISIONS = "thesis-project-499512.foodsafety_data.agent_decisions"
BQ_ESCALATIONS   = "thesis-project-499512.foodsafety_data.escalations"
BQ_CLUSTERS      = "thesis-project-499512.foodsafety_data.clusters"

# Evaluation
EVAL_SAMPLE_SIZE  = 50
KAPPA_THRESHOLD   = 0.60