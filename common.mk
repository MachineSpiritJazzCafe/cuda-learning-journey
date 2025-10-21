# Simple CUDA Makefile with Profiling
NVCC = nvcc
ARCH = -arch=sm_75
BASE_FLAGS = -D WSL_BUILD -I.. $(ARCH)

# Build Configuration Flags
# =============================================================================
# DEFAULT: What you'll use 95% of the time - optimized profiling
DEFAULT_FLAGS = $(BASE_FLAGS) -lineinfo -Xptxas -v

# DEBUG: For cuda-gdb debugging (disables optimizations)
DEBUG_FLAGS = $(BASE_FLAGS) -G -g -v

# NOCACHE: For cache strategy testing (bypasses L1 cache)
NOCACHE_FLAGS = $(BASE_FLAGS) -lineinfo -Xptxas -v,-dlcm=cg

# PRODUCTION: Maximum optimizations for final benchmarks
PROD_FLAGS = $(BASE_FLAGS) -O3 -use_fast_math -lineinfo

# Select build mode (default to DEFAULT_FLAGS)
ifndef BUILD_MODE
	BUILD_MODE = default
endif

ifeq ($(BUILD_MODE),debug)
	FLAGS = $(DEBUG_FLAGS)
else ifeq ($(BUILD_MODE),nocache)
	FLAGS = $(NOCACHE_FLAGS)
else ifeq ($(BUILD_MODE),production)
	FLAGS = $(PROD_FLAGS)
else
	FLAGS = $(DEFAULT_FLAGS)
endif

# NVCC Flags Documentation:
# =============================================================================
# -D <n>: Defines macros for preprocessing (e.g., -D WSL_BUILD defines WSL_BUILD for conditional compilation).
# -I <dir>: Adds include search paths (e.g., -I.. includes parent directory for headers like helpers.cuh).
# -lineinfo: Adds line-level profiling info without disabling optimizations (better than -G for profiling).
# -G: Generates device debug info; disables optimizations (use ONLY for cuda-gdb debugging).
# -g: Host debug info (pairs with -G for debugging).
# -Xptxas <options>: Passes options to PTX assembler:
#   -dlcm=cg: Bypass L1 cache, use only L2 (for cache strategy experiments).
# -O3: Maximum optimization level (use in production builds).
# -use_fast_math: Aggressive math optimizations (slightly less precise, much faster).
# -v: Verbose output (prints stats like registers, shared memory usage).
# -arch <arch>: Targets GPU architecture (e.g., -arch=sm_75 for Turing RTX 2070 Super; generates native PTX/SASS).
#
# NOTE: To see register spills and local memory usage, use Nsight Compute profiling instead of compiler flags.

SRC_DIR = src
BUILD_DIR = build
PROFILE_DIR = profiling
NCU_DIR = $(PROFILE_DIR)/ncu
NSYS_DIR = $(PROFILE_DIR)/nsys

# Find all .cu files and convert to target names
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu,%, $(SOURCES))

# NCU Metrics for Educational Comparison
NCU_METRICS = gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_per_inst_scheduled.ratio,smsp__sass_branch_targets_threads_divergent.avg,launch__registers_per_thread,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_lg.sum,smsp__warp_cycles_per_issue_stall_memory_shared.avg.pct_of_peak_sustained_active
# Metric Documentation:
# =============================================================================
# gpu__time_duration.avg                              		- Kernel execution time (baseline for comparisons).
# dram__throughput.avg.pct_of_peak_sustained_elapsed  		- Memory bandwidth utilization (% of peak; high in global reductions like V1/V2).
# sm__throughput.avg.pct_of_peak_sustained_elapsed    		- Compute unit utilization (% of peak; low if stalled on memory/divergence).
# l1tex__throughput.avg.pct_of_peak_sustained_elapsed 		- L1/Texture cache utilization (high in shared kernels like V3/V4).
# sm__warps_active.avg.pct_of_peak_sustained_active   		- Warp occupancy (% warps active; aim >50% for good parallelism).
# smsp__inst_executed_per_inst_scheduled.ratio        		- Instruction efficiency (executed/scheduled; >1 indicates good reuse).
# smsp__sass_branch_targets_threads_divergent.avg     		- Branch divergence (high in interleaved like V1/V3).
# launch__registers_per_thread                        		- Register usage per thread (high can limit occupancy).
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum 	- Shared memory load bank conflicts (key for V3/V4 optimization; aim low).
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum 	- Shared memory store bank conflicts (similar to loads).
# l1tex__data_pipe_lsu_wavefronts_mem_lg.sum          		- Global/local memory access count (high in V1/V2; contrasts with shared).
# smsp__warp_cycles_per_issue_stall_memory_shared.avg.pct_of_peak_sustained_active - % cycles stalled on shared memory (indicates bank conflict impact).

# Default target - builds ALL .cu files in src/
default: $(TARGETS)
	@echo "Built all targets ($(BUILD_MODE) mode): $(TARGETS)"
	@echo "Flags used: $(FLAGS)"

# Create directories if they don't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PROFILE_DIR):
	mkdir -p $(NCU_DIR) $(NSYS_DIR)

# Build any version from src/ to build/
%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Building $@ ($(BUILD_MODE) mode)..."
	$(NVCC) $(FLAGS) $< -o $(BUILD_DIR)/$@

# Build mode shortcuts
debug:
	$(MAKE) BUILD_MODE=debug

nocache:
	$(MAKE) BUILD_MODE=nocache

production:
	$(MAKE) BUILD_MODE=production

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

# Profiling targets
# =============================================================================

# Full metrics collection for specific target
metrics-%: % | $(PROFILE_DIR)
	@echo "=== Full Metrics for $* ==="
	ncu --metrics $(NCU_METRICS) $(BUILD_DIR)/$* | tee $(NCU_DIR)/$*-metrics.txt

# NCU profiling (saves detailed report)
profile-ncu-%: % | $(PROFILE_DIR)
	@echo "=== NCU Profiling $* ==="
	ncu --set full -o $(NCU_DIR)/$*-full $(BUILD_DIR)/$*
	@echo "Report saved to: $(NCU_DIR)/$*-full.ncu-rep"

# NSys profiling (system-wide timeline)
profile-nsys-%: % | $(PROFILE_DIR)
	@echo "=== NSys Profiling $* ==="
	nsys profile -o $(NSYS_DIR)/$* $(BUILD_DIR)/$*
	@echo "Report saved to: $(NSYS_DIR)/$*.nsys-rep"

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
			echo "Target $$target not built. Run: make $$target"; \
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
		ls -lh $(NCU_DIR)/*.txt $(NCU_DIR)/*.ncu-rep 2>/dev/null || echo "  No NCU reports found"; \
		echo ""; \
		echo "NSys reports:"; \
		ls -lh $(NSYS_DIR)/*.nsys-rep 2>/dev/null || echo "  No NSys reports found"; \
	else \
		echo "No profiling directory found. Run a profiling command first."; \
	fi

# Help target
help:
	@echo "CUDA Learning Journey - Makefile Commands"
	@echo "=========================================="
	@echo ""
	@echo "Build Modes:"
	@echo "  make                    # Default build (optimized profiling with -lineinfo)"
	@echo "  make debug              # Debug build (with -G -g for cuda-gdb)"
	@echo "  make nocache            # Cache testing build (bypass L1 with -dlcm=cg)"
	@echo "  make production         # Production build (-O3 -use_fast_math)"
	@echo ""
	@echo "Building:"
	@echo "  make                    # Build all .cu files in src/"
	@echo "  make <target>           # Build specific target (e.g., make v1-global-interleaved)"
	@echo ""
	@echo "Testing:"
	@echo "  make test               # Run ALL built executables"
	@echo "  make test-<name>        # Run specific executable (e.g., make test-v1-global-interleaved)"
	@echo ""
	@echo "Profiling:"
	@echo "  make metrics-<name>     # Full metrics for target (e.g., make metrics-v1-global-interleaved)"
	@echo "  make profile-ncu-<name> # NCU profiling (saves detailed .ncu-rep file)"
	@echo "  make profile-nsys-<name># NSys profiling (system timeline)"
	@echo "  make compare COMPARE_TARGETS=\"v1-global-interleaved v2-global-sequential\""
	@echo ""
	@echo "Utilities:"
	@echo "  make list               # Show available targets"
	@echo "  make list-reports       # Show saved profiling reports"
	@echo "  make clean              # Remove all builds and profiling data"
	@echo "  make help               # Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make debug v1-global-interleaved     # Build v1 in debug mode"
	@echo "  make metrics-v1-global-interleaved   # Get full metrics for v1"
	@echo "  make nocache test-v1-global-interleaved  # Build and test v1 without L1 cache"
	@echo "  make compare COMPARE_TARGETS=\"v1-global-interleaved v2-global-sequential\""

.PHONY: default debug nocache production clean test help list list-reports compare metrics-% profile-ncu-% profile-nsys-% test-%
