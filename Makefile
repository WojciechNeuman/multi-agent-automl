# Makefile for atm-simulation project

ENV_NAME=multi_agent_automl
ENV_FILE=environment.yml

.PHONY: help env activate run runserver test clean

# Help command to display available make commands
help:
	@echo "Makefile commands:"
	@echo "  make env        - Create Conda environment from environment.yml"
	@echo "  make activate   - Display command to activate the Conda environment"
	@echo "  make run        - Run the app.py file"
	@echo "  make runserver  - Run the Django server"
	@echo "  make test       - Run tests in the tests/ folder"
	@echo "  make clean      - Remove __pycache__ and .pytest_cache"

# Create Conda environment
env:
	conda env create -f $(ENV_FILE)

update:
	conda env update --file $(ENV_FILE) --prune

build:
	conda env export --name $(ENV_NAME) > $(ENV_FILE)

# Command to activate the environment (just prints it)
activate:
	@echo "To activate the environment, run:"
	@echo "  conda activate $(ENV_NAME)"

# Run app.py for ML processing (not Django)
run:
	python app.py

# Run Django server (from the web/ directory)
runserver:
	cd web && python manage.py runserver

# Run tests with pytest
test:
	pytest tests/

# Clean up Python cache and pytest artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache

