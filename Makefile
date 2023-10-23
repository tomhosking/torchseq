.PHONY: docs test coverage syntax types test check


# Check that source code meets quality standards
check: formatcheck syntax types test

# Format source code automatically
format:
	black --line-length 119 --target-version py38 tests torchseq

formatcheck:
	black --check --line-length 119 --target-version py38 tests torchseq

# Check syntax
syntax:
	flake8 ./torchseq --count --select=E9,F63,F7,F82 --show-source --statistics

types:
	mypy ./torchseq --install-types --non-interactive

# Run tests for the library
test:
	WANDB_USERNAME='' pytest --cov=./torchseq ./tests

# Run tests for the library
testall:
	WANDB_USERNAME='' RUN_SLOW=1 pytest --cov=./torchseq ./tests

# Send coverage report to codecov
coverage:
	CODECOV_TOKEN="28535f9f-825a-435e-bb4e-e1de2aa63da3" codecov
	rm .coverage
	rm coverage.xml

# Build docs
docs:
	sphinx-apidoc -f -o ./docs/_source ./torchseq
	(cd ./docs && make html)