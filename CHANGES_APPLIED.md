# Changes applied — review pass

Everything below was found by reading every file in the project and
tracing how data flows between them. Each fix is also commented
in-line at the exact spot it was made, so you can find the reasoning
again later without this file.

## Bugs (would silently produce wrong or zero results)

1. **CAMIS dtype corruption (the root cause of your 0-labels run)**
   Present in FOUR places, not just the one script you'd already
   patched: `00_environment_and_data_download.py` (sanity-check cell),
   `03_ground_truth_construction.py` (main join), and
   `05_agent_build_and_eval.py` (both `get_inspection_history` and
   `get_recent_complaints`). Every `pd.read_csv()` that touches a
   `camis`/`matched_camis` column now passes `dtype=str` explicitly,
   and lookup functions cast their input to `str` too. Without this,
   pandas silently reads all-digit ID columns as floats, so
   `"50002628"` becomes `"50002628.0"` and never matches anything.

2. **Fuzzy-matching against the wrong field.** The original
   `03_ground_truth_construction.py` matched against `df_311["location"]`
   — a nested Socrata object that reads back from CSV as a garbled
   stringified dict, not usable address text. Rebuilt the match key
   from `incident_address`/`street_name` instead, mirroring the DOHMH
   side's `dba + building + street` key.

3. **`violation_category` column referenced but never created.**
   `06_hdbscan_clustering.py` checked `if "violation_category" in
   df.columns` and silently fell through to an empty feature block
   when it wasn't there — which was always, since no upstream
   notebook creates it. Meanwhile `config.py` already defined
   `VIOLATION_CATEGORIES` as a flat list with no keywords attached, so
   it was defined but unused. Added `VIOLATION_CATEGORIES_KEYWORDS`
   (a real keyword map) to config, and notebook 06 now derives the
   category from complaint text with it.

4. **`gemini-1.5-flash` is long retired.** Updated to
   `gemini-3.5-flash-lite` (current GA model, cost-efficient, no
   shutdown date announced as of Aug 2026). Double check
   https://ai.google.dev/gemini-api/docs/deprecations before you
   submit, since Google's retirement cadence has been fast this year.

5. **`create_react_agent(..., state_modifier=...)` uses a deprecated
   parameter.** Newer `langgraph` releases renamed it to `prompt`;
   `state_modifier` raises or warns depending on version. Your
   `requirements.txt` doesn't pin an exact langgraph version, so a
   fresh install would hit this. Fixed to `prompt=`.

## Design gaps (wouldn't crash, but would undermine your own stated requirements)

6. **HDBSCAN was missing the "restaurant" dimension.** You explicitly
   asked for time, restaurant, complaint type, symptoms, and food
   category — lat/long alone stood in for "restaurant" but doesn't
   fully capture it (GPS jitter, cross-street logging, etc.). Added a
   `restaurant_freq` feature (each restaurant's complaint frequency,
   scaled) as a lightweight numeric signal, since one-hot encoding
   every individual CAMIS would blow up the feature space.

7. **"Best model" selection picked by severe-class recall alone**, in
   both `04_bilstm_classifier.py` and `07_full_evaluation_and_ablation.py`.
   A model that predicts "severe" for every input gets 100% recall
   and is useless. Changed the primary ranking criterion to PR-AUC
   (which balances precision and recall), with recall still reported
   as context. `07` still ranks by recall for display purposes — if
   you want it to match 04's PR-AUC-based selection exactly, say so
   and I'll align them.

8. **O(n×m) fuzzy matching would take a very long time to complete**
   on the full dataset (roughly 70K complaints × 25K restaurants).
   Narrowed candidates by ZIP code before fuzzy matching each row,
   cutting the comparison set per row from ~25,000 to a few hundred.

9. **The original `03_ground_truth_construction.py` required manually
   uncommenting code blocks** to actually run the full match and
   labelling steps. Rewritten to run end-to-end, with a fast
   ID-overlap sanity check inserted BEFORE the slow labelling loop —
   so a dtype/matching problem is caught in under a second next time,
   not after another hour-long run.

10. **Added a manual-review sample export** at the end of notebook 03
    (`data/labelled/manual_review_sample.csv`, ~50 rows). This is the
    hand-labelled validation subset we discussed earlier for checking
    how reliable the distant-supervision labels actually are — fill in
    `manual_severity_label` yourself and compare against `label` with
    `cohen_kappa_score` for your methodology chapter.

## Update — network timeout fix (notebook 00)

11. **Data download had no retry logic, no app token, and no
    checkpointing.** You hit `ReadTimeoutError` partway through the
    NYC 311 pull — with the original code that meant losing every
    page already downloaded, since everything sat in a Python list
    in memory until the very end. Fixed:
    - Each page now writes straight to the CSV as it arrives, so a
      late failure doesn't lose earlier progress
    - Failed requests retry up to 4 times with exponential backoff
      (2s, 4s, 8s, 16s) before giving up
    - Timeout raised from 60s to 120s, batch size lowered from 50,000
      to 25,000 rows/page (smaller requests are less likely to time
      out on Socrata's unauthenticated tier)
    - Added `SOCRATA_APP_TOKEN` support in `config.py` (reads from an
      environment variable). **Get one free at
      https://data.cityofnewyork.us/profile/edit — takes about a
      minute, no approval wait — and it meaningfully reduces
      throttling/timeouts.** Set it via `SOCRATA_APP_TOKEN` env var,
      or paste it directly into config.py.

    If a download partially completes and then hard-fails even with
    retries, delete the partial CSV in `data/raw/` before re-running
    — the script only skips the download if the file already exists,
    it doesn't currently resume mid-file.

## Things intentionally left as-is (flagging, not fixing)

- **BiLSTM on single-timestep sequences** (`04_bilstm_classifier.py`):
  feeding one fixed 384-dim embedding as a "sequence of length 1" to
  an LSTM doesn't give it any real sequential structure to learn from
  — it's functioning close to a dense layer with extra parameters. I
  didn't change the architecture since that's a modeling decision
  that affects your methodology chapter's framing, not a bug. Worth a
  paragraph in your dissertation either justifying this choice or
  discussing it as a limitation.
- **`pipeline/` folder is empty scaffolding** (`agent/`, `clustering/`,
  `ingestion/`, `nlp/`, `routing/` — all empty `__init__.py`). This is
  the production/GCP code that doesn't exist yet. See the deployment
  plan for what goes in each.
