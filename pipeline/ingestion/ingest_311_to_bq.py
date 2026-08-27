"""
pipeline/ingestion/ingest_311_to_bq.py
----------------------------------------
Pulls food-related NYC 311 complaints and writes them straight into
BigQuery, instead of a local CSV/ndjson file. This replaces the
manual "download CSV, then upload through the console" step you were
doing by hand.

WHY BIGQUERY LOAD JOBS INSTEAD OF ROW-BY-ROW INSERTS:
There are two ways to get data into BigQuery from Python: streaming
inserts (insert_rows_json, one row/small batch at a time, shows up
instantly) or load jobs (load_table_from_json, batched, slight delay
before visible, but far cheaper and has no daily quota limit).
For a scheduled ingestion job like this one, load jobs are the right
choice -- we don't need sub-second visibility, and streaming inserts
have both a cost and a row-quota you can hit surprisingly fast.

WHY WE FORCE AN EXPLICIT SCHEMA HERE (not autodetect):
Autodetect is what caused the "too many errors" failure you hit
uploading dohmh_inspections.csv by hand -- it guesses types from a
sample of rows and breaks on anything that doesn't match later on.
An explicit schema, defined once here in code, always behaves the
same way regardless of what's in any particular batch of new data.
matched_camis is deliberately STRING -- this is the exact column
that broke the whole labelling pipeline earlier when it got read
back as a float. Keeping it STRING all the way into BigQuery avoids
reintroducing that bug at the database layer.

HOW TO RUN:
    python pipeline/ingestion/ingest_311_to_bq.py --since 2024-01-01
"""

import argparse
import sys
import time
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import GCP_PROJECT, BIGQUERY_DATASET, BQ_COMPLAINTS, SOCRATA_APP_TOKEN
import requests

SOCRATA_BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
PAGE_SIZE = 25000

FOOD_RELATED_COMPLAINT_TYPES = [
    "Food Poisoning", "Food Establishment", "Unsanitary Conditions",
    "Mobile Food Vendor", "Food Vendor",
]

# Explicit schema -- see module docstring for why this matters.
# Only the fields we actually use downstream; BigQuery load jobs are
# happy to receive a JSON row with extra keys not in the schema as
# long as we don't ask it to autodetect (we pass schema explicitly
# so unexpected extra fields from Socrata are just ignored, not fatal).
SCHEMA = [
    # INTEGER here is correct and safe -- unlike camis (which has
    # leading-zero/precision issues that make STRING mandatory),
    # 311's unique_key is a plain numeric ID with no such problem.
    # This also has to match the type BigQuery already auto-detected
    # for the existing complaints table, or load jobs reject the batch.
    bigquery.SchemaField("unique_key", "INTEGER"),
    bigquery.SchemaField("created_date", "TIMESTAMP"),
    bigquery.SchemaField("complaint_type", "STRING"),
    bigquery.SchemaField("descriptor", "STRING"),
    bigquery.SchemaField("incident_address", "STRING"),
    bigquery.SchemaField("street_name", "STRING"),
    bigquery.SchemaField("incident_zip", "STRING"),
    bigquery.SchemaField("borough", "STRING"),
    bigquery.SchemaField("latitude", "FLOAT"),
    bigquery.SchemaField("longitude", "FLOAT"),
]

FIELDS_TO_KEEP = [f.name for f in SCHEMA]


def build_where(since: str) -> str:
    types_list = ", ".join(f"'{t}'" for t in FOOD_RELATED_COMPLAINT_TYPES)
    return f"complaint_type in({types_list}) AND created_date >= '{since}'"


def fetch_page(where_clause, limit, offset):
    params = {"$where": where_clause, "$limit": limit, "$offset": offset,
              "$order": "created_date ASC"}
    headers = {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}
    resp = requests.get(SOCRATA_BASE, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def clean_row(row: dict) -> dict:
    """Keep only the fields in our schema, drop everything else Socrata sends."""
    out = {k: row.get(k) for k in FIELDS_TO_KEEP}
    # unique_key arrives as a string from Socrata's JSON API but the
    # BigQuery table expects INTEGER -- cast it, falling back to None
    # rather than crashing the whole batch on one bad value.
    if out.get("unique_key") not in (None, ""):
        try:
            out["unique_key"] = int(out["unique_key"])
        except (ValueError, TypeError):
            out["unique_key"] = None
    # matched_camis doesn't come from 311 -- it gets filled in later by
    # the labelling step. Leaving it out entirely here (rather than
    # forcing None) keeps this table's job strictly "raw ingestion."
    return out


def ingest(since: str, table_id: str):
    client = bigquery.Client(project=GCP_PROJECT)
    where_clause = build_where(since)
    offset = 0
    total = 0

    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )

    while True:
        for attempt in range(4):
            try:
                page = fetch_page(where_clause, PAGE_SIZE, offset)
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
        load_job = client.load_table_from_json(rows, table_id, job_config=job_config)
        load_job.result()  # blocks until the load finishes, raises on failure

        total += len(rows)
        print(f"  loaded {len(rows)} rows (offset {offset}), total so far: {total}")

        offset += PAGE_SIZE
        time.sleep(0.5)
        if len(page) < PAGE_SIZE:
            break

    print(f"Done. Loaded {total} rows into {table_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2024-01-01")
    parser.add_argument("--table", default=BQ_COMPLAINTS)
    args = parser.parse_args()

    print(f"Ingesting 311 complaints since {args.since} into {args.table}")
    ingest(args.since, args.table)
