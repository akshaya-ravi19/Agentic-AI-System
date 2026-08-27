# pipeline/ingestion/Dockerfile
#
# Packages the ingestion scripts into a container Cloud Run can run
# as a scheduled Job. One image, two possible entry points (311 or
# DOHMH) -- which one runs is chosen at deploy time via an
# environment variable, not baked into the image, so we don't need
# two separate images to maintain.
FROM python:3.12-slim

WORKDIR /app

# Only copy what ingestion actually needs -- not the whole repo
# (classifier weights, notebooks, etc. would bloat the image for no
# reason and slow every deploy).
COPY requirements-ingestion.txt .
RUN pip install --no-cache-dir -r requirements-ingestion.txt

COPY config/ ./config/
COPY pipeline/__init__.py ./pipeline/__init__.py
COPY pipeline/ingestion/ ./pipeline/ingestion/

# INGEST_TARGET decides which script runs: "311" or "dohmh".
# Set per-job at deploy time (see deploy commands).
ENV INGEST_TARGET=311

CMD python pipeline/ingestion/run.py
