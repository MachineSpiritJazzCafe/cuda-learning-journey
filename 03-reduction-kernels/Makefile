# CUDA Reduction Kernels Makefile
NVCC = nvcc
FLAGS = -D WSL_BUILD -I..
SRC_DIR = src
BUILD_DIR = build

# Find all .cu files and convert to target names
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu,%, $(SOURCES))

# Default target - builds ALL .cu files in src/
default: $(TARGETS)
	@echo "Built all targets: $(TARGETS)"

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build any version from src/ to build/
%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Building $@..."
	$(NVCC) $(FLAGS) $< -o $(BUILD_DIR)/$@

# Clean up executables
clean:
	rm -rf $(BUILD_DIR)

# Quick test - runs ALL built executables
test: $(TARGETS)
	@echo "=== Testing All Reduction Kernels ==="
	@for target in $(TARGETS); do \
		echo ""; \
		echo "========================================"; \
		echo "Running $$target..."; \
		echo "========================================"; \
		$(BUILD_DIR)/$$target || echo "$$target failed!"; \
	done
	@echo ""
	@echo "=== All Tests Complete ==="

# Test a specific version
test-%: %
	@echo "Testing $*..."
	$(BUILD_DIR)/$*

# List available targets
list:
	@echo "Available targets:"
	@echo "$(TARGETS)" | tr ' ' '\n' | sort

# Help message
help:
	@echo "CUDA Reduction Kernels - Makefile Commands:"
	@echo ""
	@echo "Building:"
	@echo "  make              - Build all .cu files in src/"
	@echo "  make <target>     - Build specific target (e.g., make v1-naive-reduction)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run ALL built executables"
	@echo "  make test-<n>  - Run specific executable (e.g., make test-v1-naive-reduction)"
	@echo ""
	@echo "Utilities:"
	@echo "  make list         - Show available targets"
	@echo "  make clean        - Remove all built files"
	@echo "  make help         - Show this help message"

.PHONY: clean test default list help
