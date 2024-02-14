black: ## Black format only python files to line length 100
	black --line-length=100 ./;
	make clean

flake8: ## Flake8 every python file
	find ./ -type f -name "*.py" -a | xargs flake8

pylint: ## pylint every python file
	find ./ -type f -name "*.py" -a | xargs pylint

build: ## Build package distribution files
	flit build;

publish: ## Publish package distribution files to pypi
	flit publish;
	make clean;

clean: ## Remove caches, checkpoints, and .DS_Store
	find . -type f -name ".DS_Store" | xargs rm;
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) | xargs rm -r;
