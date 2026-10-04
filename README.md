# MLOps Model Productionization Platform

## Credit Card Fraud Detection — Production Lifecycle

This project industrializes an existing ML research pipeline for credit card fraud detection. It does **not** build a new ML model — instead, it adds the operational lifecycle around the existing model: serving, monitoring, retraining, versioning, promotion, and rollback.

### Core Lifecycle

```text
ml-research training config
      |
      v
 model artifact + metadata
      |
      v
 validation -> registry -> champion
      |
      v
 FastAPI /predict
      |
      +-> inference telemetry
      |
      v
 data + prediction drift
      |
      v
 retraining trigger
      |
      v
 candidate model
      |
      v
 validation gates
    /      \
 fail      pass
  reject   register
              |
              v
          promote champion
              |
              v
            serve
              |
          smoke failure
              |
              v
           rollback
```

### Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train the initial model
python -m src.ml_research.train \
  --config configs/credit_card_fraud.yaml \
  --data data/creditcard.csv \
  --output models/credit-card-fraud/1.0.0

# Register and promote to champion
python -c "
from src.registry.local_registry import LocalRegistry
registry = LocalRegistry('models')
registry.register('models/credit-card-fraud/1.0.0', '1.0.0')
registry.promote('credit-card-fraud', '1.0.0')
"

# Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"V1": -1.36, "V2": -0.07, ..., "Amount": 149.62}}'
```

### Stack

- **Serving**: FastAPI / Uvicorn
- **ML**: scikit-learn (existing ml-research pipeline)
- **Registry**: Local filesystem (MLflow-ready)
- **Monitoring**: Evidently-compatible drift detection (PSI/KS)
- **CI/CD**: GitHub Actions
- **Container**: Docker multi-stage build
- **Testing**: pytest / Ruff / mypy

### Project Structure

```
mlops-model-production/
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── serving/       # Model predictor
│   ├── monitoring/    # Drift detection
│   ├── retraining/    # Retrain pipeline
│   ├── artifact/      # Manifest + schema contracts
│   ├── registry/      # Model versioning
│   ├── integration/   # ml-research adapter
│   ├── ml_research/   # Training pipeline (simulated)
│   └── telemetry/     # Structured logging
├── configs/           # Training configs
├── models/            # Model artifacts
├── tests/             # Test suite
├── .github/workflows/ # CI/CD
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### Key Design Decisions

- **Training-serving skew prevention**: Preprocessing is packaged WITH the model in a single sklearn Pipeline
- **Rollback ≠ rebuild**: Rollback changes the champion pointer, not the Docker image
- **Drift ≠ degradation**: Drift monitoring is a signal, not proof of performance loss
- **Candidate validation**: No model reaches production without passing all validation gates
- **Version independence**: Service version and model version are separate
