.PHONY: setup install run lint test test-cov clean help

setup:
	python -m venv venv
	pip install -r requirements.txt

install:
	pip install -r requirements.txt

run:
	jupyter notebook notebooks/telco_churn_analysis.ipynb

lint:
	black notebooks/
	flake8 notebooks/ --max-line-length=100
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=notebooks --cov-report=html

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .ipynb_checkpoints/

help:
	@echo "Available commands:"
	@echo "  make setup  — Create venv and install dependencies"
	@echo "  make run    — Start Jupyter notebook"
	@echo "  make lint   — Format & check code"
	@echo "  make clean  — Remove cache files"
