.PHONY: setup
setup:
	@echo "Setting up..."
	@python3 -m venv venv
	@(. venv/bin/activate && pip install poetry && poetry install)


.PHONY: format
format:
	@echo "Formatting code..."
	@(. venv/bin/activate && black . && isort .)


.PHONY: run
run:
	@echo "Running..."
	@(. venv/bin/activate && poetry install && uvicorn app.server:app --reload)