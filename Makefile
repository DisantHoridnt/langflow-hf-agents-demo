.PHONY: unit-tests integration-tests all-tests build dev clean hf-model-tests run-all-integration clean-tests demo-tests

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
	docker-compose --env-file .env up --build dev

# Clean up Docker resources
clean:
	docker-compose down
	docker system prune -f

# Run the HuggingFace model integration test file (with logs)
hf-model-tests:
	docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration/test_real_langflow_components.py -v -s"

# Run tests with clear, beautiful output format (no warnings)
clean-tests:
	@echo "\n\033[1;36m┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\033[0m"
	@echo "\033[1;36m┃           🧪 LANGFLOW COMPONENTS TEST SUITE         ┃\033[0m"
	@echo "\033[1;36m┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\033[0m\n"
	@echo "\033[1;34m📋 Running Core Unit Tests...\033[0m"
	@docker-compose run --rm unit-tests bash -c "source /activate-environment.sh && python -m pytest tests/unit -v -q --no-header --disable-warnings"
	@echo "\n\033[1;32m✅ All unit tests passed successfully!\033[0m\n"
	@echo "\033[1;35m⚡ Running Real-world Integration Tests...\033[0m"
	@docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration/test_real_langflow_components.py -v -q --no-header --disable-warnings"
	@echo "\n\033[1;32m✅ All integration tests passed successfully!\033[0m"
	@echo "\n\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo "\033[1;36m           🚀 LANGFLOW COMPONENTS READY              \033[0m"
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n"

# Run all integration tests
run-all-integration:
	docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration -v"

# Demo the components with detailed test outputs (for presentations)
demo:
	@echo "\n\033[1;36m┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\033[0m"
	@echo "\033[1;36m┃          💼 LANGFLOW COMPONENTS DEMO            ┃\033[0m"
	@echo "\033[1;36m┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\033[0m\n"
	@echo "\033[1;33m🔔 Running dockers to demonstrate component capabilities...\033[0m\n"
	@docker-compose --env-file .env run --rm integration-tests bash -c "source /activate-environment.sh && python -m pytest tests/integration/test_real_langflow_components.py -v --no-header --disable-warnings -s" || true
	@echo "\n\033[1;32m🏁 Demo completed! Components are ready for use in Langflow.\033[0m"

# Help
help:
	@echo "Available commands:"
	@echo "  make unit-tests         - Run unit tests only"
	@echo "  make integration-tests  - Run integration tests (requires Langflow)"
	@echo "  make all-tests          - Run all tests"
	@echo "  make hf-model-tests     - Run HuggingFace model integration tests (uses real HF API)"
	@echo "  make run-all-integration - Run all integration tests"
	@echo "  make build              - Build Docker images"
	@echo "  make dev                - Start development environment with Langflow"
	@echo "  make clean              - Clean up Docker resources"
	@echo "  make clean-tests        - Run tests with minimal output (clean display)"
	@echo "  make demo-tests        - Run a demo with detailed test outputs (for presentations)"
