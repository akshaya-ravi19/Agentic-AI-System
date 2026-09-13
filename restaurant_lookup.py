from typing import List, Dict
from google.cloud import bigquery
from .app_unified import BQ_INSPECTIONS, get_bq

def lookup_restaurant(restaurant_name: str, borough: str) -> List[Dict]:
    """Return a list of matching establishments for the given name and borough.
    Each dict contains keys: 'camis', 'dba', 'boro', 'street', 'zipcode'.
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
            SELECT camis, dba, boro, street, zipcode
            FROM `{BQ_INSPECTIONS}`
            WHERE UPPER(dba) LIKE UPPER(@name_pattern)
            {loc_clause}
            ORDER BY inspection_date DESC
        """
        params = [bigquery.ScalarQueryParameter("name_pattern", "STRING", f"%{restaurant_name.strip()}%")]
        if borough and borough.strip():
            params.append(bigquery.ScalarQueryParameter("borough_pattern", "STRING", f"%{borough.strip()}%"))
            params.append(bigquery.ScalarQueryParameter("borough_exact", "STRING", borough.strip()))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(get_bq().query(query, job_config=job_config).result())
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[lookup_restaurant] error: {e}", flush=True)
        return []
