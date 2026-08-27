"""
pipeline/classifier_service/app.py
------------------------------------
Serves the trained BiLSTM classifier over HTTP as a Cloud Run
service. Takes raw complaint text in, returns a severity score out
-- the agent (notebook 05's tools, once ported to pipeline/agent/)
calls this instead of loading the model + embedder in every process.

WHY A SEPARATE SERVICE RATHER THAN THE AGENT LOADING THE MODEL DIRECTLY:
The BiLSTM (~few MB) and the sentence-transformer embedder (~90MB) are
both real memory/startup-time costs. If every agent invocation loaded
them fresh, requests would be slow and wasteful. Running this as its
own always-warm-ish Cloud Run service means the model loads ONCE per
container instance and stays in memory across many requests -- and it
can scale independently of the agent (e.g. if classification volume
spikes separately from agent traffic).

WHICH MODEL FILE THIS LOADS:
Set via the MODEL_FILE env var, defaulting to whichever model
notebook 04 (or 07's evaluation) determined was best by PR-AUC --
update this default once you know that from your own results,
rather than assuming SMOTE is the winner.
"""
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import EMBEDDING_MODEL

MODEL_FILE = os.environ.get("MODEL_FILE", "bilstm_classweight.keras")
MODEL_PATH = Path("/app/models/bilstm") / MODEL_FILE

app = FastAPI(title="foodsafety-classifier")

# Loaded once at container startup, not per-request -- this is the
# whole point of running this as a standing service rather than a
# one-shot script.
embedder = None
model = None

@app.on_event("startup")
async def load_models():
    global embedder, model
    print(f"[startup] Loading embedder: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[startup] Loading classifier: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Did you COPY it into the "
            f"image in the Dockerfile? See pipeline/classifier_service/Dockerfile."
        )
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("[startup] Models loaded, ready to serve.")

class ComplaintRequest(BaseModel):
    text: str

class ComplaintResponse(BaseModel):
    severity_score: float
    severity_label: int  # 1 = severe, 0 = non-severe, thresholded at 0.5
    model_used: str


@app.get("/health")
def health():
    # Cloud Run and any future load balancer/health check hits this.
    return {"status": "ok", "model": MODEL_FILE}

@app.post("/predict", response_model=ComplaintResponse)
def predict(req: ComplaintRequest):
    if embedder is None or model is None:
        raise HTTPException(status_code=503, detail="Model still loading, retry shortly")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")

    # Embed exactly the same way notebook 02 did during training --
    # any mismatch here (different model, different pooling) would
    # silently produce meaningless predictions without ever raising
    # an error, since the shapes would still happen to match.
    embedding = embedder.encode([req.text])  # shape (1, 384)

    # BiLSTM expects a sequence dimension even though this is really
    # a single fixed-length vector (see CHANGES_APPLIED.md note on
    # this architecture choice) -- reshape to (1, 1, 384) to match
    # what notebook 04 trained on.
    embedding_seq = embedding.reshape(1, 1, -1)

    score = float(model.predict(embedding_seq, verbose=0)[0][0])
    label = int(score >= 0.5)

    return ComplaintResponse(
        severity_score=score,
        severity_label=label,
        model_used=MODEL_FILE,
    )