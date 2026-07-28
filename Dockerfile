FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    OUTPUT_DIR=/data/outputs \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tini gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN chmod +x ./docker-entrypoint.sh \
    && mkdir -p /models/huggingface /data/outputs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv/app /models /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health/live || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/srv/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
