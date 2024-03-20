.PHONY: setup
setup:
	@echo "Setting up..."
	@python3.10 -m venv venv
	@(. venv/bin/activate && pip install poetry && poetry install && pip install -r requirements.txt)


.PHONY: format
format:
	@echo "Formatting code..."
	@(. venv/bin/activate && black . && isort .)


.PHONY: run
run:
	@echo "Running..."
	@(. venv/bin/activate && poetry install && uvicorn app.server:app --reload)

.PHONY: poetry-to-requirements
poetry-to-requirements:
	@echo "Generating requirements.txt..."
	@(. venv/bin/activate && poetry export -f requirements.txt --output requirements.txt --without-hashes)
