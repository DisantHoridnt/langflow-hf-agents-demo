.PHONY: unit-tests integration-tests all-tests build dev clean gemma-tests run-all-integration

# Run unit tests only
unit-tests:
	docker-compose run --rm unit-tests

# Run integration tests only (requires Langflow)
integration-tests:
	docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration -v -s"

# Run all tests (both unit and integration)
all-tests:
	docker-compose run --rm unit-tests && docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration -v -s"

# Build all Docker images
build:
	docker-compose build

# Start development environment with Langflow
dev:
	docker-compose --env-file .env up dev

# Clean up Docker resources
clean:
	docker-compose down
	docker system prune -f

# Run the HuggingFace model integration test file (with logs)
hf-model-tests:
	docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration/test_real_langflow_components.py -v -s"

# Run all integration tests
run-all-integration:
	docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration -v"

# Help
help:
	@echo "Available commands:"
	@echo "  make unit-tests         - Run unit tests only"
	@echo "  make integration-tests  - Run integration tests (requires Langflow)"
	@echo "  make all-tests          - Run all tests"
	@echo "  make gemma-tests        - Run Gemma integration tests (uses real HF API)"
	@echo "  make run-all-integration - Run all integration tests"
	@echo "  make build              - Build Docker images"
	@echo "  make dev                - Start development environment with Langflow"
	@echo "  make clean              - Clean up Docker resources"
