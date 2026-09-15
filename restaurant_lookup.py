from typing import List, Dict
from google.cloud import bigquery
from config.config import BQ_INSPECTIONS, GCP_PROJECT

_bq_client = None

def get_bq():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=GCP_PROJECT)
    return _bq_client

def lookup_restaurant(restaurant_name: str, borough: str) -> List[Dict]:
    """Return a list of matching establishments for the given name and borough.
    Each dict contains keys: 'camis', 'dba', 'boro', 'street', 'zipcode'.
    Deduplicates unique establishment locations.
    """
    if not restaurant_name or not restaurant_name.strip():
        return []
    try:
        loc_clause = (
            "AND (UPPER(boro) LIKE UPPER(@borough_pattern) "
            "OR UPPER(street) LIKE UPPER(@borough_pattern) "
            "OR CAST(zipcode AS STRING) = @borough_exact)"
            if borough and borough.strip() else ""
        )
        query = f"""
            SELECT DISTINCT CAST(camis AS STRING) AS camis, dba, boro, street, zipcode
            FROM `{BQ_INSPECTIONS}`
            WHERE UPPER(dba) LIKE UPPER(@name_pattern)
            {loc_clause}
            ORDER BY dba, boro, street
            LIMIT 30
        """
        params = [bigquery.ScalarQueryParameter("name_pattern", "STRING", f"%{restaurant_name.strip()}%")]
        if borough and borough.strip():
            params.append(bigquery.ScalarQueryParameter("borough_pattern", "STRING", f"%{borough.strip()}%"))
            params.append(bigquery.ScalarQueryParameter("borough_exact", "STRING", borough.strip()))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(get_bq().query(query, job_config=job_config).result())
        
        seen = set()
        deduped = []
        for r in rows:
            d = dict(r)
            c = str(d.get("camis", "")).strip()
            if c and c not in seen:
                seen.add(c)
                deduped.append(d)
        return deduped
    except Exception as e:
        print(f"[lookup_restaurant] error: {e}", flush=True)
        return []
