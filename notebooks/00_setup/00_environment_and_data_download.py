# ============================================================
# NOTEBOOK 00 — Environment Check & Data Download
# RUN THIS FIRST before opening any other notebook.
# ============================================================
# HOW TO USE:
# Option A (Jupyter): jupyter notebook, open this file
# Option B (script):  python 00_environment_and_data_download.py
# ============================================================

import sys, requests, json
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

# ── CELL 1: Check all packages are installed ────────────────
print("Checking packages...")
packages = ["pandas","numpy","requests","spacy","nltk",
            "sentence_transformers","sklearn","tensorflow",
            "imblearn","hdbscan","langchain","langgraph"]
all_ok = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK  {pkg}")
    except ImportError:
        print(f"  MISSING  {pkg}  <- run: pip install -r requirements.txt")
        all_ok = False
if all_ok:
    print("\nAll packages installed.")
else:
    print("\nFix missing packages before continuing.")
    sys.exit(1)

# ── CELL 2: Check spaCy model ───────────────────────────────
import spacy
try:
    spacy.load("en_core_web_sm")
    print("spaCy model OK")
except OSError:
    print("spaCy model missing — run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# ── Helper: resilient paginated fetch with retries + checkpointing ─
# Fixes a real failure you hit: a single ReadTimeout on any one page
# used to kill the whole download and lose every page already
# fetched (everything lived in a Python list in memory until the
# very end). Now each page is written to disk immediately, and each
# request retries with backoff before giving up.
import time

def fetch_paginated(api_url, extra_params, out_path, order_field, batch=25000, max_retries=4):
    """
    Pulls a Socrata dataset page by page, writing each page straight
    to out_path (append mode) so progress survives a crash/timeout.
    Returns the total row count written.
    """
    offset = 0
    total = 0
    # Start fresh each run (00 already skips this function entirely
    # if out_path exists -- see the calling cells below)
    if out_path.exists():
        out_path.unlink()
    first_page = True

    headers = {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}
    if not SOCRATA_APP_TOKEN:
        print("  (no SOCRATA_APP_TOKEN set -- this will be slower and more "
              "timeout-prone; see config.py for a 1-minute fix)")

    while True:
        params = dict(extra_params)
        params.update({"$limit": batch, "$offset": offset, "$order": order_field})

        page = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(api_url, params=params, headers=headers, timeout=120)
                r.raise_for_status()
                page = r.json()
                break
            except (requests.exceptions.RequestException,) as e:
                wait = 2 ** attempt  # 2s, 4s, 8s, 16s
                print(f"  request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    print(f"  giving up on offset {offset} after {max_retries} attempts.")
                    print(f"  {total:,} rows were already saved to {out_path} before this failure --")
                    print(f"  re-run this script; it will need out_path deleted first to resume from scratch,")
                    print(f"  or ask Claude to add resume-from-offset support if this keeps happening.")
                    raise
                print(f"  retrying in {wait}s...")
                time.sleep(wait)

        if not page:
            break

        pd.DataFrame(page).to_csv(out_path, mode="a", index=False, header=first_page)
        first_page = False
        total += len(page)
        offset += batch
        print(f"  {total:,} records downloaded...")

        if len(page) < batch:
            break
        time.sleep(0.3)  # be polite to the API between successful pages too

    return total

# ── CELL 3: Download NYC 311 food complaints ────────────────
DATA_RAW.mkdir(parents=True, exist_ok=True)
out_311 = DATA_RAW / "nyc_311_food_complaints.csv"

if out_311.exists():
    print(f"311 data already exists ({out_311}). Delete to re-download.")
else:
    print("Downloading NYC 311 food complaints (this may take a few minutes)...")
    where = f"complaint_type='{COMPLAINT_TYPE}' AND created_date >= '{START_YEAR}-01-01T00:00:00'"
    n = fetch_paginated(NYC_311_API, {"$where": where}, out_311, "created_date DESC")
    print(f"Saved {n:,} records -> {out_311}")

# ── CELL 4: Download DOHMH inspection results ───────────────
out_dohmh = DATA_RAW / "dohmh_inspections.csv"

if out_dohmh.exists():
    print(f"DOHMH data already exists ({out_dohmh}). Delete to re-download.")
else:
    print("Downloading DOHMH inspection results...")
    n = fetch_paginated(DOHMH_API, {}, out_dohmh, "inspection_date DESC")
    print(f"Saved {n:,} records -> {out_dohmh}")

# ── CELL 5: Sanity checks ───────────────────────────────────
# NOTE: dtype=str on camis/unique_key is important here. Without it,
# pandas auto-detects these ID columns as numbers and later scripts
# that read the same CSVs will silently get "50002628.0" instead of
# "50002628" -- IDs that look identical but never match in a join.
df_311   = pd.read_csv(out_311, dtype={"unique_key": str})
df_dohmh = pd.read_csv(out_dohmh, dtype={"camis": str})
print(f"\n=== SANITY CHECKS ===")
print(f"311 rows:            {len(df_311):,}")
print(f"311 columns:         {list(df_311.columns)}")
print(f"311 null descriptor: {df_311.get('descriptor', pd.Series()).isna().sum()}")
print(f"DOHMH rows:          {len(df_dohmh):,}")
print(f"DOHMH critical flag: {df_dohmh.get('critical_flag', pd.Series()).value_counts().to_dict()}")
print(f"DOHMH restaurants:   {df_dohmh.get('camis', pd.Series()).nunique():,}")
print("\nSetup complete.")