# pipeline/classifier_service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pipeline/classifier_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model INTO the image at build
# time, so the container never needs network access to fetch it at
# startup. This was very likely the real cause of the slow-cold-start
# failure -- app.py's SentenceTransformer(...) call was blocking port
# 8080 from opening until a runtime download finished, and Cloud Run's
# startup probe only allows one ~240s attempt before giving up.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY config/ ./config/
COPY pipeline/classifier_service/app.py ./pipeline/classifier_service/app.py
COPY models/bilstm/ ./models/bilstm/

ENV MODEL_FILE=bilstm_classweight.keras
ENV PORT=8080

CMD uvicorn pipeline.classifier_service.app:app --host 0.0.0.0 --port ${PORT}
