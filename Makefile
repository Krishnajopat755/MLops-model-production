.PHONY: lint typecheck test test-e2e docker-test install dev clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -m "unit" -v --tb=short

test-integration:
	pytest tests/ -m "integration" -v --tb=short

test-e2e:
	pytest tests/ -m "e2e" -v --tb=short

test-all:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

docker-build:
	docker build -t mlops-fraud-detection:latest .

docker-test: docker-build
	docker compose -f docker-compose.yml up -d api
	sleep 5
	curl -f http://localhost:8000/health/live
	curl -f http://localhost:8000/health/ready
	docker compose down

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

train:
	python -m src.ml_research.train --config configs/credit_card_fraud.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache htmlcov .coverage
