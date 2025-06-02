# Makefile for atm-simulation project

ENV_NAME=multi_agent_automl
ENV_FILE=environment.yml
FRONTEND_DIR=frontend_ui

.PHONY: help env activate run run-server test clean

# Help command to display available make commands
help:
	@echo "Makefile commands:"
	@echo "  make env        - Create Conda environment from environment.yml"
	@echo "  make activate   - Display command to activate the Conda environment"
	@echo "  make run        - Run the app.py file"
	@echo "  make run-server  - Run the Django server"
	@echo "  make test       - Run tests in the tests/ folder"
	@echo "  make clean      - Remove __pycache__ and .pytest_cache"
	@echo "  make install-frontend-deps - Install Node.js dependencies for the React frontend"
	@echo "  make run-frontend - Start the React frontend development server"

# Create Conda environment
env:
	conda env create -f $(ENV_FILE)

update:
	conda env update --file $(ENV_FILE) --prune

# Command to activate the environment (just prints it)
activate:
	@echo "To activate the environment, run:"
	@echo "  conda activate $(ENV_NAME)"

# Run app.py for ML processing (not Django)
run:
	python app.py

# Run Django server (from the web/ directory)
run-server:
	python web_multi_agent_automl/manage.py run-server

# Install Node.js dependencies for the React frontend (assumes npm is installed)
install-frontend-deps:
	cd $(FRONTEND_DIR) && npm install

run-frontend:
	cd $(FRONTEND_DIR) && npm start

# Run tests with pytest
test:
	pytest tests/

# Clean up Python cache and pytest artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache
