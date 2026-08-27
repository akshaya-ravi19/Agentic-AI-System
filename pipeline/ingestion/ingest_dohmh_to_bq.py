"""
pipeline/ingestion/ingest_dohmh_to_bq.py
-------------------------------------------
Pulls DOHMH inspection data and writes it into BigQuery.

IMPORTANT DIFFERENCE FROM ingest_311_to_bq.py:
This writes with WRITE_TRUNCATE (replace the whole table), not
WRITE_APPEND. Why: DOHMH's dataset already represents current state
(each pull contains the latest full inspection history for active
restaurants), not a delta of "new since last time" like 311 complaints
are. Appending here would just keep duplicating the same violation
rows every time this job runs. Truncate-and-reload keeps the table
matching DOHMH's source of truth exactly.

This also loads directly into inspection_history_clean with REAL
types (TIMESTAMP, INTEGER) rather than the all-STRING workaround you
used for the one-off manual upload -- because here we're building the
JSON rows in Python first, we control the types directly and never
hit BigQuery's CSV autodetect at all, so that whole class of problem
doesn't come up.

HOW TO RUN:
    python pipeline/ingestion/ingest_dohmh_to_bq.py
"""

import sys
import time
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import GCP_PROJECT, BQ_INSPECTIONS, SOCRATA_APP_TOKEN
import requests

SOCRATA_BASE = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
PAGE_SIZE = 25000

SCHEMA = [
    bigquery.SchemaField("camis", "STRING"),           # STRING is deliberate -- see chat history re: the CAMIS dtype bug
    bigquery.SchemaField("dba", "STRING"),
    bigquery.SchemaField("boro", "STRING"),
    bigquery.SchemaField("building", "STRING"),
    bigquery.SchemaField("street", "STRING"),
    bigquery.SchemaField("zipcode", "STRING"),
    bigquery.SchemaField("cuisine_description", "STRING"),
    bigquery.SchemaField("inspection_date", "TIMESTAMP"),
    bigquery.SchemaField("action", "STRING"),
    bigquery.SchemaField("violation_code", "STRING"),
    bigquery.SchemaField("violation_description", "STRING"),
    bigquery.SchemaField("critical_flag", "STRING"),
    bigquery.SchemaField("score", "INTEGER"),
    bigquery.SchemaField("grade", "STRING"),
    bigquery.SchemaField("grade_date", "TIMESTAMP"),
    bigquery.SchemaField("inspection_type", "STRING"),
    bigquery.SchemaField("latitude", "FLOAT"),
    bigquery.SchemaField("longitude", "FLOAT"),
]
FIELDS_TO_KEEP = [f.name for f in SCHEMA]


def fetch_page(limit, offset):
    params = {"$limit": limit, "$offset": offset, "$order": "camis, inspection_date"}
    headers = {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}
    resp = requests.get(SOCRATA_BASE, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def clean_row(row: dict) -> dict:
    out = {}
    for k in FIELDS_TO_KEEP:
        v = row.get(k)
        # score arrives as a string from Socrata's JSON API; cast to
        # int here, but fall back to None instead of crashing the
        # whole batch on one bad/blank value (same SAFE_CAST idea we
        # used in the BigQuery SQL cleanup, just done in Python here).
        if k == "score" and v not in (None, ""):
            try:
                v = int(v)
            except (ValueError, TypeError):
                v = None
        out[k] = v
    return out


def ingest(table_id: str):
    client = bigquery.Client(project=GCP_PROJECT)
    offset = 0
    total = 0
    first_batch = True

    while True:
        for attempt in range(4):
            try:
                page = fetch_page(PAGE_SIZE, offset)
                break
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"  fetch failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        else:
            print("  giving up on this page after 4 retries")
            break

        if not page:
            break

        rows = [clean_row(r) for r in page]

        # Truncate on the FIRST batch only, then append the rest --
        # otherwise every batch would wipe out the previous one.
        write_disposition = (
            bigquery.WriteDisposition.WRITE_TRUNCATE if first_batch
            else bigquery.WriteDisposition.WRITE_APPEND
        )
        job_config = bigquery.LoadJobConfig(
            schema=SCHEMA,
            write_disposition=write_disposition,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        load_job = client.load_table_from_json(rows, table_id, job_config=job_config)
        load_job.result()
        first_batch = False

        total += len(rows)
        print(f"  loaded {len(rows)} rows (offset {offset}), total so far: {total}")

        offset += PAGE_SIZE
        time.sleep(0.5)
        if len(page) < PAGE_SIZE:
            break

    print(f"Done. Loaded {total} rows into {table_id} (table was replaced, not appended to)")


if __name__ == "__main__":
    print(f"Ingesting DOHMH inspections into {BQ_INSPECTIONS} (this replaces the table's contents)")
    ingest(BQ_INSPECTIONS)
