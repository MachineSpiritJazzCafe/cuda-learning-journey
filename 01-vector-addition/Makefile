# Simple CUDA Makefile with Profiling
NVCC = nvcc
FLAGS = -D WSL_BUILD -I..
SRC_DIR = src
BUILD_DIR = build
PROFILE_DIR = profiling
NCU_DIR = $(PROFILE_DIR)/ncu
NSYS_DIR = $(PROFILE_DIR)/nsys

# Find all .cu files and convert to target names
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu,%, $(SOURCES))

# NCU Metrics for Educational Comparison
NCU_METRICS = gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_per_inst_scheduled.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__sass_branch_targets_threads_divergent.avg,launch__registers_per_thread

# Metric Documentation:
# gpu__time_duration.avg                              - Kernel execution time
# dram__throughput.avg.pct_of_peak_sustained_elapsed  - Memory bandwidth utilization (% of peak)
# sm__throughput.avg.pct_of_peak_sustained_elapsed    - Compute unit utilization (% of peak)  
# l1tex__throughput.avg.pct_of_peak_sustained_elapsed - L1/Texture cache utilization
# sm__warps_active.avg.pct_of_peak_sustained_active   - Warp occupancy (% warps active)
# smsp__inst_executed_per_inst_scheduled.ratio        - Instruction efficiency (executed/scheduled)
# l1tex__data_pipe_lsu_wavefronts_mem_shared.sum      - Shared memory access count
# smsp__sass_branch_targets_threads_divergent.avg     - Branch divergence (threads taking different paths)
# launch__registers_per_thread                        - Register usage per thread


# Default target - builds ALL .cu files in src/
default: $(TARGETS)
	@echo "Built all targets: $(TARGETS)"

# Create directories if they don't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PROFILE_DIR):
	mkdir -p $(NCU_DIR) $(NSYS_DIR)

# Build any version from src/ to build/
%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Building $@..."
	$(NVCC) $(FLAGS) $< -o $(BUILD_DIR)/$@

# Clean up executables and all reports
clean:
	rm -rf $(BUILD_DIR) $(PROFILE_DIR)

# Quick test - runs ALL built executables
test: $(TARGETS)
	@echo "=== Testing All Targets ==="
	@for target in $(TARGETS); do \
		echo "Running $$target..."; \
		$(BUILD_DIR)/$$target || echo "$$target failed!"; \
		echo ""; \
	done

# Test a specific version
test-%: %
	@echo "Testing $*..."
	$(BUILD_DIR)/$*

# NCU profiling for specific target (saves to file)
profile-ncu-%: % | $(PROFILE_DIR)
	@echo "=== NCU Profiling: $* ==="
	@echo "Results will be saved to $(NCU_DIR)/$*_ncu.txt"
	ncu --metrics $(NCU_METRICS) $(BUILD_DIR)/$* > $(NCU_DIR)/$*_ncu.txt 2>&1
	@echo "NCU profiling complete. Report saved."

# NSys profiling for specific target
profile-nsys-%: % | $(PROFILE_DIR)
	@echo "=== NSys Profiling: $* ==="
	nsys profile --trace=cuda --stats=true -o $(NSYS_DIR)/$*_profile $(BUILD_DIR)/$*
	@echo "NSys report saved as: $(NSYS_DIR)/$*_profile.nsys-rep"

# Combined metrics collection for specific target
# Compare multiple targets (usage: make compare COMPARE_TARGETS="v1-basic v2-chunked")
compare:
	@if [ -z "$(COMPARE_TARGETS)" ]; then \
		echo "Usage: make compare COMPARE_TARGETS=\"target1 target2 target3\""; \
		exit 1; \
	fi
	@echo "========================================"
	@echo "PERFORMANCE COMPARISON"
	@echo "========================================"
	@for target in $(COMPARE_TARGETS); do \
		echo ""; \
		echo "--- $$target ---"; \
		if [ -f $(BUILD_DIR)/$$target ]; then \
			echo "Running NCU on $$target..."; \
			ncu --metrics $(NCU_METRICS) $(BUILD_DIR)/$$target | grep -E "^\s+(dram__|gpu__|l1tex__|launch__|sm__|smsp__)" | grep -v "n/a" | awk 'BEGIN{kernel=1} {if(NR>1 && (NR-1)%8==0) {print "  [Kernel " ++kernel "]"} print $$0} END{if(NR>8) print ""}'; \
		else \
			echo "Target $$target not built."; \
		fi; \
	done
	@echo "========================================"

# List available targets
list:
	@echo "Available targets:"
	@echo "$(TARGETS)" | tr ' ' '\n' | sort

# Show saved reports
list-reports:
	@echo "Saved profiling reports:"
	@if [ -d $(PROFILE_DIR) ]; then \
		echo "NCU reports:"; \
		ls -la $(NCU_DIR)/*.txt 2>/dev/null || echo "  No NCU reports found"; \
		echo "NSys reports:"; \
		ls -la $(NSYS_DIR)/*.nsys-rep 2>/dev/null || echo "  No NSys reports found"; \
	else \
		echo "No profiling directory found. Run a profiling command first."; \
	fi

# Help message
help:
	@echo "CUDA Learning Journey - Makefile Commands:"
	@echo ""
	@echo "Building:"
	@echo "  make              - Build all .cu files in src/"
	@echo "  make <target>     - Build specific target (e.g., make v1-basic)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run ALL built executables"
	@echo "  make test-<name>  - Run specific executable (e.g., make test-v1-basic)"
	@echo ""
	@echo "Profiling:"
	@echo "  make metrics-<name>      - Full metrics for target (e.g., make metrics-v1-basic)"
	@echo "  make profile-ncu-<name>  - NCU profiling only (saves to file)"
	@echo "  make profile-nsys-<name> - NSys profiling only"
	@echo "  make compare COMPARE_TARGETS=\"v1-basic v2-chunked\" - Compare multiple targets"
	@echo ""
	@echo "Utilities:"
	@echo "  make list         - Show available targets"
	@echo "  make list-reports - Show saved profiling reports"
	@echo "  make clean        - Remove all built files and reports"
	@echo "  make help         - Show this help message"

.PHONY: clean test default list help compare list-reports
