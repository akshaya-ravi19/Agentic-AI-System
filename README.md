# FoodGuard — Agentic AI Food Safety Triage System
MSc Dissertation · Data Science & Artificial Intelligence

## What this project does
FoodGuard is an agentic AI pipeline that:
1. Classifies NYC 311 food safety complaints as severe / non-severe (BiLSTM)
2. Retrieves evidence from DOHMH inspection records (LangGraph agent tools)
3. Detects spatiotemporal outbreak patterns (HDBSCAN — 5 dimensions)
4. Produces structured triage recommendations for Environmental Health Officers

The agent RECOMMENDS. Humans make all final regulatory decisions.

## Environment setup

**Option A — Google Colab (fastest to start, free GPU access):**
Upload the repo to Drive, mount it, `!pip install -r requirements.txt`
in a cell. Colab Pro gives longer runtimes/better GPUs if notebook 04
is too slow on the free tier.

**Option B — VS Code + local venv (used for most of this project):**
```
python -m venv venv
venv\Scripts\activate        (Windows)  /  source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**GCP project setup (one-time, needed before pipeline/ deployment):**
```
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable bigquery.googleapis.com run.googleapis.com \
    cloudscheduler.googleapis.com pubsub.googleapis.com \
    aiplatform.googleapis.com secretmanager.googleapis.com \
    artifactregistry.googleapis.com cloudbuild.googleapis.com
```
New GCP accounts get $300 free credit; GCP also has a separate
education/student credit program worth checking if you have a .edu
email. Set a budget alert (Billing > Budgets) before deploying
anything — Cloud Run jobs and services are cheap but not free, and
notebook 04/05 alone won't touch GCP cost at all (they run locally).

Core libraries (already in requirements.txt): NLTK, spaCy,
sentence-transformers, imbalanced-learn (SMOTE), hdbscan, langgraph,
langchain, google-cloud-bigquery.

## Build order — follow exactly
Step 1 → notebooks/00_setup/   (verify environment, download data)
Step 2 → notebooks/01_eda/     (understand data before modelling)
Step 3 → notebooks/03_labelling/ (construct ground truth labels)
Step 4 → notebooks/02_preprocessing/ (clean, embed, split, balance)
Step 5 → notebooks/04_bilstm/  (train and evaluate classifier)
Step 6 → notebooks/05_agent/   (build and evaluate agent)
Step 7 → notebooks/06_clustering/ (HDBSCAN + validation)
Step 8 → notebooks/07_evaluation/ (ablation study + final results)
Step 9 → pipeline/ + deployment/ (only after evaluation is complete)

## Deploying pipeline/ to GCP (after evaluation is solid)

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

## User evaluation

See `evaluation/user_questionnaire.md` for the questionnaire template
to administer to Environmental Health Officers / inspectors reviewing
sample agent outputs, as part of the evaluation chapter.

## Datasets
- NYC 311: https://data.cityofnewyork.us/resource/erm2-nwe9.json
- DOHMH:   https://data.cityofnewyork.us/resource/43nn-pn8j.json

## Key rule
NEVER apply SMOTE or any balancing to validation or test data.
Training fold only. Split first, balance second.
