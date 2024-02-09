black: ## Black format only python files to line length 100
	black --line-length=100 .;
	make clean
flake8: ## Flake8 every python file
	find ./s_cerevisiae/python -type f -name "*.py" -a | xargs flake8

pylint: ## pylint every python file
	find ./s_cerevisiae/python -type f -name "*.py" -a | xargs pylint

clean: ## Remove caches, checkpoints, and .DS_Store
	find . -type f -name ".DS_Store" | xargs rm;
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) | xargs rm -r;
