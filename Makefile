.PHONY: unit-tests integration-tests all-tests build dev clean

# Run unit tests only
unit-tests:
	docker-compose run --rm unit-tests

# Run integration tests only (requires Langflow)
integration-tests:
	docker-compose run --rm integration-tests

# Run all tests (both unit and integration)
all-tests:
	docker-compose run --rm unit-tests && docker-compose run --rm integration-tests

# Build all Docker images
build:
	docker-compose build

# Start development environment with Langflow
dev:
	docker-compose up dev

# Clean up Docker resources
clean:
	docker-compose down
	docker system prune -f

# Help
help:
	@echo "Available commands:"
	@echo "  make unit-tests         - Run unit tests only"
	@echo "  make integration-tests  - Run integration tests (requires Langflow)"
	@echo "  make all-tests          - Run all tests"
	@echo "  make build              - Build Docker images"
	@echo "  make dev                - Start development environment with Langflow"
	@echo "  make clean              - Clean up Docker resources"
