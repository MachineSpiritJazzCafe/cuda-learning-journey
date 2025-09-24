# Simple CUDA Makefile
NVCC = nvcc
FLAGS = -D WSL_BUILD -I..

# Default target - builds v1-basic
default: v1-basic

# Build any .cu file by removing .cu extension
%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

# Clean up executables
clean:
	rm -f v1-basic v2-* v3-*

# Quick test run
test: v1-basic
	./v1-basic

.PHONY: clean test default
