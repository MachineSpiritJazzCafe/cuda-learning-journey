#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils/cuda_utils.cuh"


// FORWARD DECLARATION
__global__ void reduce_in_place(float* input, int n);



// CPU Reduction Helper
inline float cpu_reduce(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}


// ============================================================================
// Host function: Reduce entire Array using gpu kernel
// ============================================================================
inline float reduce(float* d_input, int n, int blockSize, int elementsPerThread) {
   
     printf("\n=== Strategy: Multi-Stage GPU Reduction ===\n");

    int currentSize = n;
    int stage = 0;
    int elementsPerBlock = blockSize * elementsPerThread; 
    // Keep reducing until we can fit in one block
    while (currentSize > blockSize) {
        int numBlocks = (currentSize + elementsPerBlock - 1) / elementsPerBlock;
        
        printf("  Stage %d: %d elements → %d blocks\n", 
               stage, currentSize, numBlocks);
        
        reduce_in_place<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        currentSize = numBlocks;
        stage++;
    }
    
    // Final reduction in one block
    printf("  Stage %d (final): %d elements → 1 block\n", stage, currentSize);
    reduce_in_place<<<1, blockSize, blockSize * sizeof(float)>>>(d_input, currentSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back final result (just 1 float)
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    printf("  Total stages: %d\n", stage + 1);
    printf("  Data transfer: 4 bytes (1 float)\n");
    return result;
}

// ============================================================================
// Check if we're in profile mode (for NCU profiling with single large test)
// ============================================================================
inline bool is_profile_mode() {
    const char* mode = getenv("PROFILE_MODE");
    return mode != nullptr && strcmp(mode, "1") == 0;
}

inline int get_profile_size() {
    const char* size_str = getenv("PROFILE_SIZE");
    if (size_str) {
        return atoi(size_str);
    }
    return 1024 * 1024 * 1024;
}

// ============================================================================
// Standard Test Harness for Reduction Kernels
// TWO MODES:
// - Normal: Multiple test sizes for validation
// - Profile: Single large test for clean NCU profiling
// ============================================================================
inline void run_reduction_tests(const char* kernel_name, int blockSize = 256,
                                int elementsPerThread = 1) {
    
    if (is_profile_mode()) {
        // ========================================================================
        // PROFILE MODE: Single large test for NCU
        // ========================================================================
        int n = get_profile_size();
        size_t bytes = n * sizeof(float);
        
        printf("=============================================================\n");
        printf("PROFILE MODE: %s\n", kernel_name);
        printf("=============================================================\n");
        printf("N = %d elements (%.2f MB)\n", n, bytes / 1e6);
        printf("=============================================================\n\n");
        
        // Allocate and initialize
        float* h_input = new float[n];
        for (int i = 0; i < n; i++) {
            h_input[i] = 1.0f;  // Sum should equal N
        }
        
        float* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        
        // Run reduction (no timing - NCU will time)
        float gpu_result = reduce(d_input, n, blockSize, elementsPerThread);
        
        // Verify
        float expected = (float)n;
        float error = fabs(gpu_result - expected) / expected;
        
        printf("\n--- Results ---\n");
        printf("GPU result:   %.0f\n", gpu_result);
        printf("Expected:     %.0f\n", expected);
        printf("Error:        %.6f%%\n", error * 100.0f);
        printf("Status:       %s\n", error < 1e-5 ? "✓ PASSED" : "✗ FAILED");
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        delete[] h_input;
        
    } else {
        // ========================================================================
        // TEST MODE: Multiple sizes for validation (YOUR ORIGINAL CODE)
        // ========================================================================
        printf("=============================================================\n");
        printf("Threshold-Based Reduction Strategy\n");
        printf("=============================================================\n");
        
        // Test different problem sizes to see strategy selection
        int test_sizes[] = {
            1 << 15,  // 32K   → ~128 blocks (right at threshold)
            1 << 18,  // 256K  → ~1K blocks (multi-stage)
            1 << 20,  // 1M    → ~4K blocks (multi-stage)
            1 << 22   // 4M    → ~16K blocks (multi-stage)
        };
        
        for (int test = 0; test < 4; test++) {
            int n = test_sizes[test];
            size_t bytes = n * sizeof(float);
            
            printf("\n\n");
            printf("#############################################################\n");
            printf("TEST %d: N = %d elements (%.2f MB)\n", test + 1, n, bytes / 1e6);
            printf("#############################################################\n");
            
            // Allocate and initialize
            float* h_input = new float[n];
            for (int i = 0; i < n; i++) {
                h_input[i] = 1.0f;  // Sum should equal N
            }
            
            float* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, bytes));
            CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            // Time the reduction
            CUDA_CHECK(cudaEventRecord(start));
            float gpu_result = reduce(d_input, n, blockSize, elementsPerThread);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            
            // Verify
            float expected = (float)n;
            float error = fabs(gpu_result - expected) / expected;
            
            printf("\n--- Results ---\n");
            printf("GPU result:   %.0f\n", gpu_result);
            printf("Expected:     %.0f\n", expected);
            printf("Error:        %.6f%%\n", error * 100.0f);
            printf("Time:         %.4f ms\n", milliseconds);
            printf("Status:       %s\n", error < 3e-5 ? "✓ PASSED" : "✗ FAILED");
            
            // Cleanup
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_input));
            delete[] h_input;
        }
    }
}

