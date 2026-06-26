SHELL := /bin/sh

CMAKE ?= cmake
GIT ?= git
BUILD_DIR ?= $(CURDIR)/build
SRC_DIR ?= $(CURDIR)/src
THIRD_PARTY_DIR ?= $(SRC_DIR)/third_party
THIRD_PARTY_PREFIX ?= $(THIRD_PARTY_DIR)/install
CMAKE_PREFIX_PATH ?= $(THIRD_PARTY_PREFIX)
BUILD_TYPE ?= Release
JOBS ?= $(shell command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

PIQP_DIR := $(THIRD_PARTY_DIR)/piqp
PIQP_CONFIG := $(THIRD_PARTY_PREFIX)/lib/cmake/piqp/piqpConfig.cmake

CPPADCG_DIR := $(THIRD_PARTY_DIR)/CppADCodeGen
CPPADCG_PC := $(THIRD_PARTY_PREFIX)/share/pkgconfig/cppadcg.pc

.PHONY: all help brew-deps third_party configure build clean distclean crisp_julia

all: third_party configure build

help:
	@echo "CRISP build helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make                  Build local third-party deps, configure, and build everything"
	@echo "  make configure         Configure CMake into $(BUILD_DIR)"
	@echo "  make build             Build all CMake targets"
	@echo "  make crisp_julia       Build the Julia CxxWrap binding"
	@echo "  make clean             Clean compiled CMake outputs"
	@echo "  make distclean         Remove build dir and local third-party deps"
	@echo "  make brew-deps         Install Homebrew deps used on macOS"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)"
	@echo "  THIRD_PARTY_PREFIX=$(THIRD_PARTY_PREFIX)"
	@echo "  JOBS=$(JOBS)"

brew-deps:
	brew install cppad boost yaml-cpp pybind11

third_party: $(PIQP_CONFIG) $(CPPADCG_PC)

$(PIQP_DIR):
	mkdir -p $(THIRD_PARTY_DIR)
	$(GIT) clone https://github.com/PREDICT-EPFL/piqp.git $(PIQP_DIR)

$(PIQP_CONFIG): | $(PIQP_DIR)
	$(GIT) -C $(PIQP_DIR) checkout v0.5.0
	$(CMAKE) -S $(PIQP_DIR) -B $(PIQP_DIR)/build \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_TESTS=OFF \
		-DBUILD_BENCHMARKS=OFF \
		-DCMAKE_INSTALL_PREFIX=$(THIRD_PARTY_PREFIX)
	$(CMAKE) --build $(PIQP_DIR)/build --config $(BUILD_TYPE) --target install -j$(JOBS)

$(CPPADCG_DIR):
	mkdir -p $(THIRD_PARTY_DIR)
	$(GIT) clone https://github.com/joaoleal/CppADCodeGen.git $(CPPADCG_DIR)

$(CPPADCG_PC): | $(CPPADCG_DIR)
	$(CMAKE) -S $(CPPADCG_DIR) -B $(CPPADCG_DIR)/build \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_TESTING=OFF \
		-DCMAKE_INSTALL_PREFIX=$(THIRD_PARTY_PREFIX)
	$(CMAKE) --build $(CPPADCG_DIR)/build --config $(BUILD_TYPE) --target install -j$(JOBS)

configure: third_party
	$(CMAKE) -S $(SRC_DIR) -B $(BUILD_DIR) -DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

crisp_julia: configure
	$(CMAKE) --build $(BUILD_DIR) --target crisp_julia -j$(JOBS)

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi

distclean:
	rm -rf $(BUILD_DIR) $(THIRD_PARTY_DIR) $(CURDIR)/model
