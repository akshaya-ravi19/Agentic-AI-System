# pipeline/agent/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pipeline/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config/ ./config/
COPY pipeline/__init__.py ./pipeline/__init__.py
COPY pipeline/agent/ ./pipeline/agent/
COPY pipeline/routing/rule_based_triage.py ./pipeline/routing/rule_based_triage.py
COPY pipeline/routing/__init__.py ./pipeline/routing/__init__.py
ENV PORT=8080
CMD uvicorn pipeline.agent.service:app --host 0.0.0.0 --port ${PORT}
