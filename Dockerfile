FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    DATA_DIR=/data \
    INBOX_DIR=/inbox \
    HF_HOME=/data/hf

# Tesseract (français + anglais) : pymupdf4llm l'utilise pour l'OCR des pages scannées.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Reranker optionnel (cross-encoder local, tourne sur CPU dans le conteneur).
#   docker compose build --build-arg WITH_RERANKER=1     (ou WITH_RERANKER=1 dans .env)
ARG WITH_RERANKER=0
RUN if [ "$WITH_RERANKER" = "1" ]; then \
        if [ "$(uname -m)" = "x86_64" ]; then \
            pip install torch --index-url https://download.pytorch.org/whl/cpu; \
        fi; \
        pip install sentence-transformers; \
    fi

COPY app ./app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)" || exit 1

CMD ["python", "-m", "app.ui"]
