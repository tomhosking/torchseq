# Check that source code meets quality standards
check:
	black --check --line-length 119 --target-version py36 tests torchseq
	flake8  --config .flake8 tests torchseq


# Format source code automatically
format:
	black --line-length 119 --target-version py36 tests torchseq

# Check syntax
syntax:
	flake8 ./torchseq --count --select=E9,F63,F7,F82 --show-source --statistics

# Run tests for the library
test:
	pytest --cov=./torchseq ./tests

# Run tests for the library
testall:
	RUN_SLOW=1 pytest --cov=./torchseq ./tests

# Send coverage report to codecov
coverage:
	CODECOV_TOKEN="28535f9f-825a-435e-bb4e-e1de2aa63da3" codecov
	rm .coverage
	rm coverage.xml