#include <cuda_runtime.h>
#include <stdio.h>
#include "utils/cuda_utils.cuh"

// Tree-based reduction with increasing stride
// VERY SLOW - operates on global memory (400-800 cycle latency)
__global__ void reduce_in_place(float* input, int n) {
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Example with blockDim.x = 8:
    // Block 0: tid = 0,1,2,3,4,5,6,7  → index = 0,1,2,3,4,5,6,7
    // Block 1: tid = 0,1,2,3,4,5,6,7  → index = 8,9,10,11,12,13,14,15

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        
        // Why? Threads read values written by OTHER threads in previous iteration
        // Without sync: Race condition - reading garbage/stale data
        __syncthreads();
        
        
        // Condition breakdown:
        //   tid % (2 * stride) == 0  → Only "even stride" threads work
        //   index + stride < n       → Boundary check (don't read out of bounds)
        if (tid % (2 * stride) == 0 && index + stride < n) {
            
            // Pattern: Each active thread sums its element with one "stride" away
            input[index] += input[index + stride];
            
            // Visual example with stride = 1 (first iteration):
            //   Thread 0: input[0] += input[1]   → input[0] = sum of [0,1]
            //   Thread 2: input[2] += input[3]   → input[2] = sum of [2,3]
            //   Thread 4: input[4] += input[5]   → input[4] = sum of [4,5]
            //   (Threads 1,3,5,7 idle - condition fails)
            
            // Visual example with stride = 2 (second iteration):
            //   Thread 0: input[0] += input[2]   → input[0] = sum of [0,1,2,3]
            //   Thread 4: input[4] += input[6]   → input[4] = sum of [4,5,6,7]
            //   (Threads 1,2,3,5,6,7 idle)
            
            // Final iteration (stride = 4):
            //   Thread 0: input[0] += input[4]   → input[0] = sum of all 8
            //   (All other threads idle)
        }
    }
    
    
    // Block results to global output:
    // Only thread 0 of each block does this
    // After the loop, input[blockIdx.x * blockDim.x] contains the block's sum
    
    if (tid == 0) {
        // Each block writes its result to the beginning of its section
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
    }
    
    // Note: After this kernel, first N_BLOCKS elements of input[] contain
    // the partial sums. Need another kernel launch or CPU code to finish.
}

// CPU Reduction Helper
float cpu_reduce(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}




// ============================================================================
// Host function: Reduce entire Array using gpu kernel only
// ============================================================================
float naiveReduceGPU(float* d_input, int n, int blockSize) {
   
     printf("\n=== Strategy: Multi-Stage GPU Reduction ===\n");
    
    int currentSize = n;
    int stage = 0;
    
    // Keep reducing until we can fit in one block
    while (currentSize > blockSize) {
        int numBlocks = (currentSize + blockSize - 1) / blockSize;
        
        printf("  Stage %d: %d elements → %d blocks\n", 
               stage, currentSize, numBlocks);
        
        reduce_in_place<<<numBlocks, blockSize>>>(d_input, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        currentSize = numBlocks;
        stage++;
    }
    
    // Final reduction in one block
    printf("  Stage %d (final): %d elements → 1 block\n", stage, currentSize);
    reduce_in_place<<<1, blockSize>>>(d_input, currentSize);
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
// Host function: Reduce using gpu and cpu to reduce 
// Do bulk work on GPU, finish small remainder on CPU
// Good when: Few blocks left, minimize kernel launches, CPU available
// ============================================================================
float naiveReduceHybrid(float* d_input, int n, int blockSize) {
    
    printf("\n=== Strategy: Hybrid GPU+CPU Reduction ===\n");
    
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    printf("  GPU stage: %d elements → %d blocks\n", n, numBlocks);
    
    // Single GPU reduction
    reduce_in_place<<<numBlocks, blockSize>>>(d_input, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back partial sums
    float* h_partial = new float[numBlocks];
    size_t transferBytes = numBlocks * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(h_partial, d_input, transferBytes, 
                         cudaMemcpyDeviceToHost));
    
    // Finish on CPU
    printf("  CPU stage: %d partial sums\n", numBlocks);
    float result = cpu_reduce(h_partial, numBlocks);
    
    printf("  Total stages: 2 (1 GPU + 1 CPU)\n");
    printf("  Data transfer: %zu bytes (%d floats)\n", 
           transferBytes, numBlocks);
    
    delete[] h_partial;
    return result;
}

// ============================================================================
// Automatically choose best strategy based on problem size
// ============================================================================
float smartReduce(float* d_input, int n, int blockSize) {
    
    // Calculate how many blocks we'd need
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // If we have more than this many blocks, use multi-stage GPU
    // otherwise, use hybrid GPU+CPU
    const int THRESHOLD = 128;
    
    float result;
    
    if (numBlocks > THRESHOLD) {
        // Large problem: Multiple GPU reductions
        printf("\nDecision: MULTI-STAGE GPU (numBlocks=%d > threshold=%d)\n", 
               numBlocks, THRESHOLD);
        printf("Reason: Many blocks → minimize data transfer, keep GPU busy\n");
        
        result = naiveReduceGPU (d_input, n, blockSize);
        
    } else {
        // Small problem: Single GPU reduction + CPU finish
        printf("\nDecision: HYBRID GPU+CPU (numBlocks=%d <= threshold=%d)\n", 
               numBlocks, THRESHOLD);
        printf("Reason: Few blocks → avoid launch overhead, CPU finish is fast\n");
        
        result = naiveReduceHybrid(d_input, n, blockSize);
    }
    
    return result;
}

int main() {
    
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
    
    int blockSize = 256;
    
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
        float gpu_result = smartReduce(d_input, n, blockSize);
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
        printf("Status:       %s\n", error < 1e-5 ? "✓ PASSED" : "✗ FAILED");
        
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_input));
        delete[] h_input;
    }
    
    
    return 0;
}
