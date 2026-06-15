.PHONY: help test test-fast benchmark docker-build docker-train docs docs-serve

IMAGE ?= opengoalrl
TRAIN_CONFIG ?= opengoalrl/configs/empty_goal_close.yaml

help:
	@echo "OpenGoalRL make targets:"
	@echo "  test          Run the full test suite (requires GRF)"
	@echo "  test-fast     Run the GRF-free test suite"
	@echo "  benchmark     Run the seeded benchmark harness and aggregate"
	@echo "  docker-build  Build the reproducible Docker image"
	@echo "  docker-train  Train $(TRAIN_CONFIG) inside Docker"
	@echo "  docs          Build the docs site (mkdocs build --strict)"
	@echo "  docs-serve    Serve the docs site locally"

test:
	pytest opengoalrl/tests/ -v

test-fast:
	pytest opengoalrl/tests/ -v -k "not test_env"

benchmark:
	python benchmarks/run_benchmarks.py --seeds 3
	python benchmarks/aggregate.py

docker-build:
	docker build -t $(IMAGE) .

docker-train: docker-build
	docker run --rm $(IMAGE) \
		python -m opengoalrl.scripts.train --config $(TRAIN_CONFIG)

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve
