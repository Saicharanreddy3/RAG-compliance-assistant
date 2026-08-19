.PHONY: install dev test lint ingest ask eval serve docker clean ci

PYTHON ?= python
export PYTHONPATH := src:.

install:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m ruff check src tests evals

ingest:
	$(PYTHON) -m rag_assistant.cli ingest --reset

ask:
	@$(PYTHON) -m rag_assistant.cli ask "$(Q)"

eval:
	$(PYTHON) -m evals.run_eval --output reports/eval.json

serve:
	$(PYTHON) -m uvicorn rag_assistant.api.main:app --reload --port 8000

docker:
	docker build -t rag-compliance-assistant .

# What CI runs.
ci: lint test ingest eval

clean:
	rm -rf data/index reports .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
