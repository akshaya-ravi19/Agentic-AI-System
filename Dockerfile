FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements_unified.txt .
RUN pip install --no-cache-dir -r requirements_unified.txt

# Pre-download sentence transformer model so container startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project assets
COPY --chown=user config/ ./config/
COPY --chown=user models/bilstm/ ./models/bilstm/
COPY --chown=user pipeline/routing/ ./pipeline/routing/
COPY --chown=user pipeline/__init__.py ./pipeline/__init__.py
COPY --chown=user app_unified.py ./app_unified.py
COPY --chown=user restaurant_lookup.py ./restaurant_lookup.py

ENV PORT=7860
ENV MODEL_FILE=bilstm_food_safety.keras

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app_unified:app --host 0.0.0.0 --port ${PORT}"]
