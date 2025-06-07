#@meta {author: "Paul Landes"}
#@meta {desc: "PyHealth build automation", date: "2025-05-22"}


## Build
#
# directory with the unit tests
PY_TEST_DIR ?=		tests
# test file glob pattern
PY_TEST_GLOB ?=		test_metrics.py


## Targets
#
# install dependencies
.PHONY:			deps
deps:
			pip install -r requirements-nlp.txt

# run the unit test cases
.PHONY:			test
test:
			@echo "Running tests in $(PY_TEST_DIR)/$(PY_TEST_GLOB)"
			python -m unittest discover \
				-s $(PY_TEST_DIR) -p '$(PY_TEST_GLOB)' -v

# clean derived objects
.PHONY:			clean
clean:
			@echo "removing __pycache__"
			@find . -type d -name __pycache__ -prune -exec rm -r {} \;
