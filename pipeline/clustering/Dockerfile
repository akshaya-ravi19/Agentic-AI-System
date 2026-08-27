# pipeline/clustering/Dockerfile
# Cloud Run JOB (like pipeline/ingestion) -- runs once nightly via
# Cloud Scheduler, then exits. Not a standing service.
FROM python:3.12-slim
WORKDIR /app
COPY pipeline/clustering/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config/ ./config/
COPY pipeline/__init__.py ./pipeline/__init__.py
COPY pipeline/clustering/ ./pipeline/clustering/
CMD python pipeline/clustering/cluster_job.py
