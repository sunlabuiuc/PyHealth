#@meta {author: "Paul Landes"}
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

# build the wheel
.PHONY:		wheel
wheel:		$(DIST_DIR)
		@PX_DIST_DIR=$(DIST_DIR) pixi run build-wheel

.PHONY:		clean
clean:
		@echo "removing target: $(TARG_DIR)"
		@rm -fr $(TARG_DIR)
