# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash balkanbench
WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY README.md /workspace/README.md

RUN uv pip install --system -e ".[dev]"

USER balkanbench

ENTRYPOINT ["balkanbench"]
CMD ["--help"]
