
# Agentic AI Food Safety Triage System

## What this project does
1. Classifies NYC 311 food safety complaints as severe / non-severe (BiLSTM)
2. Retrieves evidence from DOHMH inspection records (LangGraph agent tools)
3. Detects spatiotemporal outbreak patterns (HDBSCAN — 5 dimensions)
4. Produces structured triage recommendations for Environmental Health Officers

The agent RECOMMENDS. Humans make all final regulatory decisions.

## Environment setup


**VS Code + local venv (used for most of this project):**
```
python -m venv venv
venv\Scripts\activate        (Windows)  /  source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**GCP project setup:**
```
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable bigquery.googleapis.com run.googleapis.com \
    cloudscheduler.googleapis.com pubsub.googleapis.com \
    aiplatform.googleapis.com secretmanager.googleapis.com \
    artifactregistry.googleapis.com cloudbuild.googleapis.com
```
Core libraries (already in requirements.txt): NLTK, spaCy,
sentence-transformers, imbalanced-learn (SMOTE), hdbscan, langgraph,
langchain, google-cloud-bigquery.

## Deploying pipeline/ to GCP 

Five deployable pieces, each with its own Dockerfile:
- `pipeline/ingestion/` — Cloud Run JOBS, scheduled (311 every 15 min,
  DOHMH nightly). Real-time-ish data acquisition into BigQuery.
- `pipeline/classifier_service/` — Cloud Run SERVICE. Serves the
  trained BiLSTM over HTTP (`/predict`).
- `pipeline/clustering/` — Cloud Run JOB, scheduled nightly. Runs
  HDBSCAN over BigQuery data, writes cluster assignments back.
- `pipeline/agent/` — Cloud Run SERVICE. The LangGraph agent,
  calling the classifier service + BigQuery tools, then handing its
  decision to the router service. Needs GOOGLE_API_KEY (put it in
  Secret Manager, not a plain env var, once actually deployed).
- `pipeline/routing/` — Cloud Run SERVICE. Logs every decision to
  BigQuery; Tier 3 also publishes to Pub/Sub for the inspector
  dashboard.

Each follows the same build/deploy pattern (see chat history for the
exact `gcloud builds submit` / `gcloud run deploy` / `gcloud run jobs
create` commands used for ingestion — the classifier, agent, and
clustering deploys follow the identical shape, just swapping the
Dockerfile path and image/service name).

Deploy order matters: classifier service and routing service first
(the agent depends on both being reachable), then the agent service,
then wire up the ingestion/clustering schedules last.

## Datasets
- NYC 311: https://data.cityofnewyork.us/resource/erm2-nwe9.json
- DOHMH:   https://data.cityofnewyork.us/resource/43nn-pn8j.json

##Link to live Prototype
https://agentic-ai-system-500255319268.europe-west1.run.app/

## Key rule
NEVER apply SMOTE or any balancing to validation or test data.
Training fold only. Split first, balance second.
=======
# Agentic-AI-System
>>>>>>> cc7f66d8ea549713229f199fc3018f1fa292e9d7
