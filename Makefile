# Define Makefile targets for various tasks

black: ## Format Python files using Black
	black --line-length=100 ./

flake8: ## Run Flake8 on all Python files
	find . -type f -name "*.py" | xargs flake8

pylint: ## Run pylint on all Python files
	find . -type f -name "*.py" | xargs pylint

test: ## Run tests using pytest
	pytest -vv --tb=auto ./

build: ## Build package distribution files
	python -m build

publish: ## Publish package distribution files to PyPI
	twine upload dist/*
	make clean

clean: ## Remove caches, checkpoints, and distribution artifacts
	find . \( -name ".DS_Store" -o -name ".ipynb_checkpoints" -o -name "__pycache__" -o -name ".pytest_cache" \) | xargs rm -rf
	rm -rf dist/ build/ **/*.egg-info

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*##"}; {printf "%-15s %s\n", $$1, $$2}'
