install:
	pip install -r requirements.txt

dev:
	uvicorn frag.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	bash scripts/lint.sh

check:
	ruff check .
	ruff format --check .

format:
	ruff format .

ingest-sample:
	python scripts/ingest_sample.py

ingest-live:
	python scripts/ingest_live.py

build-golden:
	python scripts/build_golden.py

eval:
	python scripts/run_eval.py
