# Agentic AI Food Safety Triage System

> **MSc Dissertation — Data Science & Artificial Intelligence**  
> An end-to-end agentic AI pipeline that classifies, clusters, and triages NYC food safety complaints to support Environmental Health Officers in prioritising inspection resources.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Notebook Execution Order](#notebook-execution-order)
- [GCP Deployment](#gcp-deployment)
- [Live Prototype](#live-prototype)
- [Datasets](#datasets)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)

---

## Overview

Citizens regularly submit food safety concerns to NYC 311, but inspection teams lack the automated tools to triage thousands of unstructured complaints against historical violation records in real time. This project addresses that gap by building a multi-component agentic AI system that:

1. **Classifies** complaint text as an *Actionable Hazard* or *Routine* using a Bidirectional LSTM (BiLSTM) trained on sentence embeddings, with category-disjoint 5-fold cross-validation to prevent data leakage.
2. **Detects** spatiotemporal outbreak patterns across boroughs using HDBSCAN density-based clustering over a 30-day rolling window with 13 engineered features.
3. **Reasons** over multi-source evidence — complaint text, DOHMH inspection history, and cluster context — through a LangGraph agent to produce a structured triage recommendation (LOG / REVIEW / ESCALATE).
4. **Presents** inspector-ready dossiers via a dual-portal Cloud Run web application, separating the public-facing complaint intake from the inspector investigation view.

> **Important:** The system functions as a decision-support tool only. All final regulatory decisions remain with human Environmental Health Officers.

---

## System Architecture

```
NYC 311 API ──┐
              ├──► BigQuery (raw store)
DOHMH API  ───┘         │
                         ▼
              ┌─────────────────────┐
              │  BiLSTM Classifier  │  Sentence embeddings -> Severe / Routine
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  HDBSCAN Clustering │  Spatiotemporal outbreak detection (30-day window)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  LangGraph Agent    │  Evidence synthesis -> LOG / REVIEW / ESCALATE
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Cloud Run UI       │  Dual-portal: Citizen intake + Inspector dossier
              └─────────────────────┘
```

---

## Key Results

| Metric | Value |
|:---|:---:|
| BiLSTM PR-AUC (5-fold category-disjoint CV) | **0.878** |
| BiLSTM ROC-AUC | **0.897** |
| BiLSTM F1 Score | **0.81** |
| Agent Cohen's Kappa (vs expert, 50 cases) | **0.926** |
| Agent ESCALATE Recall | **78%** |
| Agent Tool Selection Accuracy | **80%** |
| Agent Hallucination Rate | **0.22 per case** |
| HDBSCAN Clusters Detected (30-day window) | **187** |
| Complaints Assigned to Clusters | **76.4%** |
| Silhouette Score | **52.07%** |
| Davies-Bouldin Index | **0.587** |

---

## Repository Structure

```
food_safety_triage_project/
│
├── app_unified.py                      # FastAPI + HTML dual-portal prototype (Cloud Run)
├── dashboard.py                        # Streamlit EDA dashboard (5-tab interactive)
├── restaurant_lookup.py                # CAMIS restaurant name resolution utility
├── requirements_unified.txt            # Production dependencies
├── Dockerfile.unified                  # Cloud Run container definition
├── cloudbuild-unified.yaml             # Cloud Build CI/CD pipeline
│
├── config/
│   └── config.py                       # Centralised project constants and paths
│
├── notebooks/
│   ├── 01_setup/                       # Environment verification and data download
│   ├── 02_preprocessing/               # NLP preprocessing, sentence embeddings, train/test splits
│   ├── 03_labelling/                   # Ground truth construction (Municipal Taxonomy + DOHMH-Matched)
│   ├── 04_bilstm/                      # BiLSTM training, hyperparameter comparison, evaluation plots
│   ├── 05_agent/                       # LangGraph agent build and 50-case expert evaluation
│   ├── 06_clustering/                  # HDBSCAN spatiotemporal clustering and validation
│   └── 07_evaluation/                  # Full evaluation, fairness audit, BiLSTM SHAP explainability
│
├── pipeline/
│   ├── ingestion/                      # Cloud Run Jobs: 311 (15-min) and DOHMH (nightly) ingest
│   ├── classifier_service/             # Cloud Run Service: BiLSTM HTTP prediction endpoint
│   ├── clustering/                     # Cloud Run Job: nightly HDBSCAN over BigQuery data
│   ├── agent/                          # Cloud Run Service: LangGraph evidence-synthesis agent
│   └── routing/                        # Deterministic rule-based triage fallback
│
├── models/
│   └── bilstm/
│       └── bilstm_food_safety.keras    # Production model (selected by PR-AUC on severe class)
│
├── data/
│   ├── raw/                            # Source CSVs (NYC 311, DOHMH inspections)
│   ├── labelled/                       # Ground truth dataset (GT-A: Municipal Taxonomy)
│   ├── processed/                      # Sentence embeddings (.npy)
│   └── splits/                         # Train/val/test splits with class weights
│
└── evaluation/
    ├── classifier/                     # PR curves, confusion matrices, SHAP figures, comparison CSV
    ├── agent/                          # Expert-labelled 50 cases, agent results, confusion matrix
    └── ablation_results.csv            # Component ablation study results
```

---

## Environment Setup

### Option A — Local Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements_unified.txt
python -m spacy download en_core_web_sm
```

### Option B — Google Colab

Upload the repository to Google Drive, mount it, and run:

```python
!pip install -r requirements_unified.txt
!python -m spacy download en_core_web_sm
```

> Colab Pro is recommended for Notebook 04 (BiLSTM training) if GPU access is required.

### GCP Project Setup (Required for cloud deployment only)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
    bigquery.googleapis.com \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

> New GCP accounts receive $300 in free credits. Configure a billing budget alert before deploying any Cloud Run services.

---

## Notebook Execution Order

The notebooks must be executed in the following sequence. Each notebook saves artefacts that downstream notebooks depend on.

| Step | Notebook | Purpose |
|:---:|:---|:---|
| 1 | `notebooks/01_setup/` | Verify environment and download raw data |
| 2 | `notebooks/02_preprocessing/` | NLP preprocessing, sentence embeddings, data splits |
| 3 | `notebooks/03_labelling/` | Construct Ground Truth A (Municipal Taxonomy) |
| 4 | `notebooks/04_bilstm/` | Train and evaluate BiLSTM classifier variants |
| 5 | `notebooks/05_agent/` | Build LangGraph agent and run 50-case expert evaluation |
| 6 | `notebooks/06_clustering/` | HDBSCAN clustering with Silhouette and Chi-square validation |
| 7 | `notebooks/07_evaluation/` | Full evaluation, fairness audit, SHAP explainability |

> **Critical Rule:** Never apply SMOTE or any data balancing technique to the validation or test set. Balance the training fold only, strictly after splitting.

---

## GCP Deployment

The production system comprises five independently deployable components, each with its own `Dockerfile`. Deploy in the order below, as the LangGraph agent depends on the classifier service being reachable.

| Order | Component | Type | Trigger |
|:---:|:---|:---|:---|
| 1 | `pipeline/classifier_service/` | Cloud Run **Service** | Always-on HTTP endpoint |
| 2 | `pipeline/routing/` | Cloud Run **Service** | Always-on |
| 3 | `pipeline/agent/` | Cloud Run **Service** | Always-on |
| 4 | `pipeline/ingestion/` | Cloud Run **Job** | 311: every 15 min · DOHMH: nightly |
| 5 | `pipeline/clustering/` | Cloud Run **Job** | Nightly |

**Unified deployment (recommended):**

```bash
gcloud builds submit --config cloudbuild-unified.yaml .

gcloud run deploy food-safety-triage \
    --image gcr.io/YOUR_PROJECT_ID/food-safety-triage \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

> Store all API keys (Google AI Studio, BigQuery service account credentials) in **Secret Manager** — never in plain environment variables in a production deployment.

---

## Live Prototype

The dual-portal prototype is deployed on Google Cloud Run and provides:

- **Citizen Portal:** Complaint intake form with free-text description, category selection, borough, restaurant name lookup, and optional photographic evidence upload.
- **Inspector Portal:** Evidence dossier displaying triage tier, AI agent reasoning, DOHMH inspection history, recent 30-day complaint patterns, and HDBSCAN cluster membership status.

---

## Datasets

| Dataset | Source | Description |
|:---|:---|:---|
| NYC 311 Food Safety Complaints | [NYC Open Data](https://data.cityofnewyork.us/resource/erm2-nwe9.json) | ~73,450 complaints (2020–present) |
| DOHMH Restaurant Inspection Results | [NYC Open Data](https://data.cityofnewyork.us/resource/43nn-pn8j.json) | Inspection grades, violation codes, CAMIS IDs |

Both datasets are publicly available under the NYC Open Data Terms of Use and contain no personally identifiable information (PII).

---

## Ethical Considerations

- The system is designed exclusively as a **decision-support tool**, not an autonomous enforcement system. All triage recommendations are subject to mandatory human review before any regulatory action is taken.
- A **geographic fairness audit** was conducted using `fairlearn`, revealing a classifier selection rate disparity of 25.1 percentage points across boroughs (Bronx: 35.7% vs. Brooklyn: 10.6%). This is discussed in full in the dissertation's fairness and bias chapter.
- **SHAP explainability** (`shap.GradientExplainer`) is applied to the production BiLSTM to ensure model decisions are semantically grounded in complaint text, providing evidence against spurious geographic correlations.
- No demographic data about complainants is collected or used at any stage of the pipeline.

---

## Citation

If referencing this work, please cite:

```
Ravi, A. (2026). Agentic AI for Municipal Food Safety Triage: A Multi-Component
Decision Support System for NYC Environmental Health Officers.
MSc Dissertation, SP Jain School of Global Management.
```

---

*Built with Python · TensorFlow/Keras · LangGraph · HDBSCAN · Google Cloud Platform · FastAPI · Streamlit*
