# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

# Phony targets
.PHONY: data run

# Default target
.DEFAULT_GOAL := help


help:
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)dist$(NC)        - Build distribution"
	@echo "  $(GREEN)artifacts$(NC)   - Train and save model artifacts"
	@echo "  $(GREEN)run$(NC)         - Run all services"


dist/*.whl: 
	@echo "$(YELLOW)Building distribution...$(NC)"
	uv build


artifacts/*.pth:
	@echo "$(YELLOW)Training model...$(NC)"
	uv run src/intergalactic_namerator/main.py


run: dist/*.whl artifacts/*.pth
	@echo "$(YELLOW)Starting services...$(NC)"
	@if [ "$$(uname)" = "Darwin" ]; then \
			echo "$(YELLOW)Detected macOS, using 'docker compose'...$(NC)"; \
			docker compose up --build --detach; \
		else \
			echo "$(YELLOW)Detected Linux, using 'docker-compose'...$(NC)"; \
			sudo docker-compose up --build --detach; \
		fi


stop:
	@echo "$(YELLOW)Stopping services...$(NC)"
	@if [ "$$(uname)" = "Darwin" ]; then \
			echo "$(YELLOW)Detected macOS, using 'docker compose'...$(NC)"; \
			docker compose stop; \
		else \
			echo "$(YELLOW)Detected Linux, using 'docker-compose'...$(NC)"; \
			sudo docker-compose stop; \
		fi


clean: stop
	@echo "$(YELLOW)Cleaning up...$(NC)"
	rm -rf dist/*
	find artifacts -type f ! -name '.gitkeep' -delete
	find assets -type f ! -name '.gitkeep' -delete
	@if [ "$$(uname)" = "Darwin" ]; then \
			echo "$(YELLOW)Detected macOS, using 'docker compose'...$(NC)"; \
			docker compose down; \
		else \
			echo "$(YELLOW)Detected Linux, using 'docker-compose'...$(NC)"; \
			sudo docker-compose down; \
		fi