# Makefile
.PHONY: help build push run train serve test clean deploy

# Variables
DOCKER_IMAGE ?= webarena-mas
VERSION ?= latest
REGISTRY ?= local
METHOD ?= p3o

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker build -t $(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) -f docker/Dockerfile .

push: ## Push Docker image to registry
	docker push $(REGISTRY)/$(DOCKER_IMAGE):$(VERSION)

run: ## Run training locally
	docker-compose -f docker/docker-compose.yml up mas-trainer

train-test: ## Run quick training test
	docker run --rm --gpus all \
		--env-file .env \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/models:/workspace/models \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		train --config /workspace/config/test_training.yaml

serve: ## Start API server
	docker-compose -f docker/docker-compose.yml up mas-api

test: ## Run tests
	docker run --rm \
		--env-file .env \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		test

deploy-runpod: ## Deploy to RunPod
	python scripts/deploy_runpod.py create \
		--config config/standard_training.yaml \
		--api-key $(RUNPOD_API_KEY)

monitor: ## Monitor training with TensorBoard
	docker run --rm -p 6006:6006 \
		-v $(PWD)/logs:/workspace/logs \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		tensorboard

jupyter: ## Start Jupyter notebook
	docker run --rm -p 8888:8888 \
		-v $(PWD):/workspace \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		jupyter

clean: ## Clean up containers and volumes
	docker-compose -f docker/docker-compose.yml down -v
	rm -rf logs/* checkpoints/* __pycache__ .pytest_cache

logs: ## View training logs
	docker-compose -f docker/docker-compose.yml logs -f mas-trainer

shell: ## Open shell in container
	docker run --rm -it --gpus all \
		--env-file .env \
		-v $(PWD):/workspace \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		bash