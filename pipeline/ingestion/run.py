"""
pipeline/ingestion/run.py
----------------------------
Entry point for the Cloud Run container. Reads INGEST_TARGET from
the environment and calls the matching ingestion script's main
function directly (as a Python import, not a subprocess) so errors
propagate properly and Cloud Run sees a real non-zero exit code on
failure -- important, since that's what triggers Cloud Scheduler's
retry/alerting behaviour.

WHY ONE ENTRY POINT INSTEAD OF TWO SEPARATE IMAGES:
Both ingestion scripts share the same dependencies (google-cloud-
bigquery, requests) and the same config. Building one image and
choosing behaviour via an environment variable means one Dockerfile
to maintain and one image to rebuild when a shared dependency
changes, instead of two that can silently drift out of sync.
"""
import os
import sys
from datetime import date, timedelta

target = os.environ.get("INGEST_TARGET", "311").lower()

if target == "311":
    from pipeline.ingestion.ingest_311_to_bq import ingest
    from config.config import BQ_COMPLAINTS
    # Default to pulling the last 2 days each run (a scheduled job
    # runs frequently, so it only needs a small recent window, not
    # the full history every time -- much faster and cheaper).
    since = os.environ.get("INGEST_SINCE") or str(date.today() - timedelta(days=2))
    print(f"[run.py] Starting 311 ingestion, since={since}")
    ingest(since, BQ_COMPLAINTS)

elif target == "dohmh":
    from pipeline.ingestion.ingest_dohmh_to_bq import ingest
    from config.config import BQ_INSPECTIONS
    print("[run.py] Starting DOHMH ingestion (full replace)")
    ingest(BQ_INSPECTIONS)

else:
    print(f"[run.py] Unknown INGEST_TARGET: {target!r} (expected '311' or 'dohmh')")
    sys.exit(1)

print("[run.py] Done.")
