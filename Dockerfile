# ── Stage 1: Base ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: Builder ────────────────────────────────────────────────────────
FROM base AS builder

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY src/ ./src/
COPY configs/ ./configs/

# ── Stage 3: Test ────────────────────────────────────────────────────────────
FROM builder AS test

COPY tests/ ./tests/
RUN pip install --no-cache-dir ".[dev]" && \
    pytest tests/ -m "unit" --tb=short -q

# ── Stage 4: Runtime ────────────────────────────────────────────────────────
FROM base AS runtime

# Non-root user (spec §09 / security)
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser

# Install production dependencies only
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip cache purge

# Copy source and required assets
COPY --from=builder /app/src/ ./src/
COPY --from=builder /app/configs/ ./configs/

# Create model/data directories
RUN mkdir -p models data reference reports && \
    chown -R appuser:appuser /app

# Copy model artifacts
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health/ready'); exit(0 if r.status_code in (200, 503) else 1)"

ENTRYPOINT ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
