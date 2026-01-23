#@meta {author: "REDACTED_AUTHOR"}
#@meta {desc: "PyHealth build automation", date: "2025-05-22"}


## Build
#
TARG_DIR ?=		target
DIST_DIR ?=		$(TARG_DIR)/dist


## Targets
#
$(DIST_DIR):
		mkdir -p $(DIST_DIR)


# install the pixi program
.PHONY:		installpixi
installpixi:
		@echo "checking for pixi..."
		@$(eval ex := $(shell which pixi > /dev/null 2>&1 ; echo $$?))
		@if [ $(ex) -eq 1 ] ; then \
			echo "pixi not found, install? (CNTL-C to stop)" ; \
			read ; \
			curl -fsSL https://pixi.sh/install.sh | sh ; \
		fi
		@echo "installed"

# install pixi and the Python environments
.PHONY:		init
init:		installpixi
		@echo "installing environment..."
		@pixi install

# run base module tests
.PHONY:		test
test:
		@echo "running unit tests..."
		@pixi run test

# run NLP specific tests
.PHONY:		testnlp
testnlp:
		@echo "running NLP unit tests..."
		@pixi run testnlp

# run all tests
.PHONY:		testall
testall:	test testnlp

# build the wheel (clean first to ensure fresh build)
.PHONY:		wheel
wheel:		clean $(DIST_DIR)
		@PX_DIST_DIR=$(DIST_DIR) pixi run build-wheel

.PHONY:		clean
clean:
		@echo "removing target: $(TARG_DIR)"
		@rm -fr $(TARG_DIR)


# upload to test PyPI
.PHONY: upload-test
upload-test: wheel
	@PX_DIST_DIR=$(DIST_DIR) pixi run -e build-pypi upload-test

# upload to PyPI
.PHONY: upload
upload: wheel
	@PX_DIST_DIR=$(DIST_DIR) pixi run -e build-pypi upload
.PHONY: bump-alpha-minor-dry
bump-alpha-minor-dry:
	@python tools/bump_version.py --alpha-minor --dry-run

.PHONY: bump-alpha-major-dry
bump-alpha-major-dry:
	@python tools/bump_version.py --alpha-major --dry-run

.PHONY: bump-minor-dry
bump-minor-dry:
	@python tools/bump_version.py --minor --dry-run

.PHONY: bump-major-dry
bump-major-dry:
	@python tools/bump_version.py --major --dry-run

.PHONY: upload-test-alpha-minor
upload-test-alpha-minor:
	@$(MAKE) bump-alpha-minor
	@$(MAKE) upload-test

.PHONY: upload-test-alpha-major
upload-test-alpha-major:
	@$(MAKE) bump-alpha-major
	@$(MAKE) upload-test

.PHONY: upload-test-minor
upload-test-minor:
	@$(MAKE) bump-minor
	@$(MAKE) upload-test

.PHONY: upload-test-major
upload-test-major:
	@$(MAKE) bump-major
	@$(MAKE) upload-test


## Smart bump + upload helpers
#
# Usage examples:
#   make upload-alpha-minor   # 2.0a04 -> 2.0a05, then build & upload
#   make upload-alpha-major   # 2.0a04 -> 2.0a10, then build & upload
#   make upload-minor         # 2.0a04 -> 2.0.0 (or next free patch), then upload
#   make upload-major         # 2.0.1  -> 2.1.0, then build & upload

.PHONY: bump-alpha-minor
bump-alpha-minor:
	@python tools/bump_version.py --alpha-minor

.PHONY: bump-alpha-major
bump-alpha-major:
	@python tools/bump_version.py --alpha-major

.PHONY: bump-minor
bump-minor:
	@python tools/bump_version.py --minor

.PHONY: bump-major
bump-major:
	@python tools/bump_version.py --major

.PHONY: upload-alpha-minor
upload-alpha-minor:
	@$(MAKE) bump-alpha-minor
	@$(MAKE) upload

.PHONY: upload-alpha-major
upload-alpha-major:
	@$(MAKE) bump-alpha-major
	@$(MAKE) upload

.PHONY: upload-minor
upload-minor:
	@$(MAKE) bump-minor
	@$(MAKE) upload

.PHONY: upload-major
upload-major:
	@$(MAKE) bump-major
	@$(MAKE) upload