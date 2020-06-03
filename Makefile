# Check that source code meets quality standards
check:
	black --check --line-length 119 --target-version py36 tests src
	flake8  --config .flake8 tests src


# Format source code automatically
format:
	black --line-length 119 --target-version py36 tests src

# Check syntax
syntax:
	flake8 ./src --count --select=E9,F63,F7,F82 --show-source --statistics

# Run tests for the library
test:
	pytest --cov

# Run tests for the library
testall:
	RUN_SLOW=1 pytest


