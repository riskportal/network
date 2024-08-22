black: ## Black format only python files to line length 100
	black --line-length=100 ./
	make clean

flake8: ## Flake8 every python file
	find ./ -type f -name "*.py" -a | xargs flake8

pylint: ## pylint every python file
	find ./ -type f -name "*.py" -a | xargs pylint

test:  ## Run tests
	pytest -vv --tb=auto ./

build: ## Build package distribution files
	python -m build

publish: ## Publish package distribution files to pypi
	twine upload dist/*
	make clean

clean: ## Remove caches, checkpoints, and distribution artifacts
	find . -type f -name ".DS_Store" | xargs rm -f
	find . -type d \( -name ".ipynb_checkpoints" -o -name "__pycache__" -o -name ".pytest_cache" \) | xargs rm -rf
	rm -rf dist/ build/ *.egg-info
