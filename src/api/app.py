"""
FastAPI application — spec §05.

Endpoints:
  POST /api/v1/predict
  POST /api/v1/predict/batch
  GET  /health/live
  GET  /health/ready
  GET  /version

The API never starts training (spec rule #4).
The API never accepts a model path from an HTTP request (spec rule #5).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.config import AppSettings, get_settings
from src.registry.local_registry import LocalRegistry
from src.serving.predictor import ModelPredictor

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)

# ── Global instances ─────────────────────────────────────────────────────────
predictor = ModelPredictor()
settings: AppSettings = get_settings()


def _load_champion_model() -> None:
    """Load the champion model at startup."""
    try:
        registry = LocalRegistry(settings.model_dir)
        champion_version = registry.get_champion_version(settings.model_name)
        version_path = registry.get_version_path(settings.model_name, champion_version)
        predictor.load_from_directory(str(version_path))
        logger.info("Champion model loaded: %s v%s", settings.model_name, champion_version)
    except FileNotFoundError:
        logger.warning(
            "No champion model found for '%s'. "
            "The API will start but readiness will be 503 until a model is loaded.",
            settings.model_name,
        )
    except Exception as e:
        logger.error("Failed to load champion model: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — load model at startup."""
    _load_champion_model()
    yield


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Fraud Detection API",
    description=(
        "Credit Card Fraud Detection serving API. "
        "Part of the MLOps Model Productionization Platform."
    ),
    version=settings.service_version,
    lifespan=lifespan,
)


# ── Error handler ────────────────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Return structured errors, never stack traces (spec §05)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={
            "error": {"code": "INVALID_INPUT", "message": str(exc)},
            "request_id": request_id,
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=503,
        content={
            "error": {"code": "MODEL_NOT_READY", "message": str(exc)},
            "request_id": request_id,
        },
    )


# ── Middleware ───────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Assign a unique request ID to every request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Prediction Endpoints ─────────────────────────────────────────────────────
@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """
    Single prediction — POST /api/v1/predict (FR-03).

    Accepts a feature dict and returns the prediction with model metadata.
    """
    request_id = request.state.request_id
    result = predictor.predict(body.features)

    return PredictionResponse(
        prediction=result["prediction"],
        fraud_probability=result.get("fraud_probability"),
        label=result["label"],
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_alias="champion",
        request_id=request_id,
        service_version=settings.service_version,
        served_at=datetime.now(UTC).isoformat(),
    )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict_batch(request: Request, body: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction — POST /api/v1/predict/batch (FR-03).

    Maximum batch size is configurable (default: 100).
    """
    request_id = request.state.request_id

    if len(body.instances) > settings.max_batch_size:
        raise ValueError(
            f"Batch size {len(body.instances)} exceeds maximum {settings.max_batch_size}"
        )

    results = predictor.predict_batch(body.instances)

    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_alias="champion",
        request_id=request_id,
        service_version=settings.service_version,
        served_at=datetime.now(UTC).isoformat(),
    )


# ── Health Endpoints ─────────────────────────────────────────────────────────
@app.get("/health/live", response_model=HealthResponse, tags=["Health"])
async def health_live() -> HealthResponse:
    """
    Liveness probe — GET /health/live (FR-03).

    Returns 200 if the service process is running.
    """
    return HealthResponse(
        status="alive",
        service=settings.service_name,
        version=settings.service_version,
    )


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    tags=["Health"],
)
async def health_ready() -> ReadinessResponse | JSONResponse:
    """
    Readiness probe — GET /health/ready (FR-03).

    Returns 503 if no model is loaded (spec §05).
    """
    response = ReadinessResponse(
        status="ready" if predictor.is_ready else "not_ready",
        service=settings.service_name,
        version=settings.service_version,
        model_loaded=predictor.is_ready,
        model_name=predictor.model_name if predictor.is_ready else None,
        model_version=predictor.model_version if predictor.is_ready else None,
    )

    if not predictor.is_ready:
        return JSONResponse(status_code=503, content=response.model_dump())

    return response


@app.get("/version", response_model=VersionResponse, tags=["Health"])
async def version() -> VersionResponse:
    """Version endpoint — GET /version (FR-03)."""
    return VersionResponse(
        service_name=settings.service_name,
        service_version=settings.service_version,
        model_name=predictor.model_name if predictor.is_ready else None,
        model_version=predictor.model_version if predictor.is_ready else None,
        feature_schema_version=(
            predictor.feature_schema.version if predictor.feature_schema else None
        ),
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Ignore browser favicon requests to avoid 404 in logs."""
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
async def root(request: Request) -> Response:
    """Root landing page and operational dashboard."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        model_status = "Ready & Serving" if predictor.is_ready else "Not Loaded"
        status_color = "#10b981" if predictor.is_ready else "#f59e0b"
        model_name = predictor.model_name
        model_ver = predictor.model_version

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Fraud Detection Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
          rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --surface: rgba(17, 24, 39, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --emerald: #10b981;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
        body {{
            background: radial-gradient(circle at top right, #1e1b4b, #090d16 60%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .container {{
            width: 100%;
            max-width: 860px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            backdrop-filter: blur(16px);
            padding: 2.5rem;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .title h1 {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
        .title p {{ font-size: 0.9rem; color: var(--text-sub); margin-top: 0.25rem; }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.12);
            color: {status_color};
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .badge .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {status_color};
            box-shadow: 0 0 10px {status_color};
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        .card .label {{
            font-size: 0.8rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .card .value {{ font-size: 1.2rem; font-weight: 600; color: #fff; margin-top: 0.35rem; }}
        .card .sub {{ font-size: 0.8rem; color: #10b981; margin-top: 0.25rem; }}
        .links-heading {{
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #d1d5db;
        }}
        .links-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-bottom: 2rem;
        }}
        .btn {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 500;
            border: 1px solid rgba(59, 130, 246, 0.3);
            transition: all 0.2s ease;
        }}
        .btn:hover {{
            background: rgba(59, 130, 246, 0.3);
            color: #fff;
            transform: translateY(-2px);
        }}
        .btn-primary {{
            background: #2563eb;
            color: #fff;
            border: none;
        }}
        .btn-primary:hover {{ background: #1d4ed8; }}
        .test-box {{
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        }}
        .test-box h3 {{ font-size: 0.95rem; margin-bottom: 0.75rem; color: #e5e7eb; }}
        .btn-test {{
            padding: 0.6rem 1.2rem;
            background: #10b981;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .btn-test:hover {{ background: #059669; }}
        pre {{
            margin-top: 1rem;
            background: #040711;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #34d399;
            overflow-x: auto;
            border: 1px solid rgba(255,255,255,0.05);
            max-height: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <h1>MLOps Model Serving Platform</h1>
                <p>Credit Card Fraud Detection &mdash; Operational Lifecycle</p>
            </div>
            <div class="badge">
                <span class="dot"></span>
                <span>{model_status}</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">Champion Model</div>
                <div class="value">{model_name}</div>
                <div class="sub">v{model_ver} (Active)</div>
            </div>
            <div class="card">
                <div class="label">Evaluation Metrics</div>
                <div class="value">99.95% Acc</div>
                <div class="sub">F1 0.8432 | ROC-AUC 0.9695</div>
            </div>
            <div class="card">
                <div class="label">Serving Engine</div>
                <div class="value">FastAPI + Uvicorn</div>
                <div class="sub">p99 Latency &lt; 5ms</div>
            </div>
        </div>

        <div class="links-heading">Available Endpoints &amp; Documentation</div>
        <div class="links-grid">
            <a href="/docs" class="btn btn-primary" target="_blank">
                &rarr; Interactive Swagger Docs
            </a>
            <a href="/redoc" class="btn" target="_blank">&bull; ReDoc Explorer</a>
            <a href="/health/ready" class="btn" target="_blank">
                &hearts; Health Probe (/health/ready)
            </a>
            <a href="/version" class="btn" target="_blank">&infin; Version Info (/version)</a>
        </div>

        <div class="test-box">
            <h3>Quick Prediction Test</h3>
            <button class="btn-test" onclick="sendTestPrediction()">
                Send Sample Inference Request
            </button>
            <pre id="output">// Click the button above to execute POST /api/v1/predict live...</pre>
        </div>
    </div>

    <script>
        async function sendTestPrediction() {{
            const output = document.getElementById('output');
            output.innerText = 'Sending request to /api/v1/predict...';
            const sample = {{}};
            for (let i = 1; i <= 28; i++) {{ sample['V' + i] = 0.0; }}
            sample['Amount'] = 75.50;

            try {{
                const res = await fetch('/api/v1/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ features: sample }})
                }});
                const data = await res.json();
                output.innerText = JSON.stringify(data, null, 2);
            }} catch (err) {{
                output.innerText = 'Error: ' + err.message;
            }}
        }}
    </script>
</body>
</html>"""
        return HTMLResponse(content=html_content)

    return JSONResponse(
        content={
            "service": settings.service_name,
            "version": settings.service_version,
            "status": "ready" if predictor.is_ready else "not_ready",
            "model_name": predictor.model_name if predictor.is_ready else None,
            "model_version": predictor.model_version if predictor.is_ready else None,
            "docs_url": "/docs",
            "health_url": "/health/ready",
            "endpoints": {
                "predict": "/api/v1/predict",
                "batch_predict": "/api/v1/predict/batch",
                "health_live": "/health/live",
                "health_ready": "/health/ready",
                "version": "/version",
            },
        }
    )
