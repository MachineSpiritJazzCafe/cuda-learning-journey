// v8-multiple-adds.cu
//
// Kernel 7 from Mark Harris's "Optimizing Parallel Reduction in CUDA"
// KEY OPTIMIZATION: Each thread processes MULTIPLE elements
//                   before writing to shared memory
//
// HOW IT WORKS:
//   1. Each thread loads elementsPerThread values from global memory
//   2. Performs partial reduction in REGISTERS (private per-thread)
//   3. Writes the partially reduced value to shared memory
//   - Reduces shared memory traffic (fewer writes)
//   - Increases arithmetic intensity (more FLOPs per memory op)
//   - Better occupancy (fewer blocks needed for same problem)
//   - Hides memory latency with independent additions
//
// TRADEOFF:
//   - More work per thread means fewer threads active
//   - Must balance: too much work → poor occupancy
//                   too little work → underutilized ALUs

#include "commons/helpers.cuh"

template <unsigned int blockSize>
__device__ void reduceBlock(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    // Final warp (no __syncthreads() needed)
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
    }
}

// ============================================================================
// TEMPLATED KERNEL WITH MULTIPLE ADDS
// ============================================================================
template <unsigned int blockSize, unsigned int elementsPerThread>
__global__ void reduce_kernel(float *input, int n) {
    
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    
    // ========================================================================
    // STEP 1: MULTIPLE ADDS IN REGISTERS
    // ========================================================================
    // Each thread processes elementsPerThread consecutive elements
    // Starting index: blockIdx.x * (blockSize * elementsPerThread) + tid
    //
    // LAYOUT EXAMPLE (blockSize=256, elementsPerThread=8):
    //   Thread 0: processes indices [0, 256, 512, 768, 1024, 1280, 1536, 1792]
    //   Thread 1: processes indices [1, 257, 513, 769, 1025, 1281, 1537, 1793]
    //   ...
    //
    // This is a STRIDED access pattern with stride = blockSize
    // Why strided? Coalesced memory accesses within each iteration    
    float sum = 0.0f;  // Accumulate in register
    
    unsigned int gridSize = blockSize * elementsPerThread * gridDim.x;
    unsigned int i = blockIdx.x * (blockSize * elementsPerThread) + tid;
    
    // Grid-stride loop: process elementsPerThread values
    // Stop when we've processed all our elements OR reached array end
    while (i < n) {
        sum += input[i];
        
        // Move to next element for this thread (stride by blockSize)
        i += blockSize;
        
        // If we've done elementsPerThread iterations, jump to next grid
        // (This handles arrays larger than gridSize)
        if (i >= blockIdx.x * (blockSize * elementsPerThread) + blockSize * elementsPerThread) {
            i += gridSize - blockSize * elementsPerThread;
        }
    }
    
    // ========================================================================
    // STEP 2: Write partial sum to shared memory
    // ========================================================================
    sdata[tid] = sum;
    __syncthreads();
    
    
    // ========================================================================
    // STEP 3: Tree reduction in shared memory (same as v7)
    // ========================================================================
    reduceBlock<blockSize>(sdata, tid);
    
    
    // Write result
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// NON-TEMPLATED WRAPPER
// ============================================================================
__global__ void reduce_in_place(float *input, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Using elementsPerThread = 8 (Harris's recommended sweet spot)
    const unsigned int elementsPerThread = 8;
    
    // MULTIPLE ADDS IN REGISTERS
    float sum = 0.0f;
    unsigned int gridSize = blockDim.x * elementsPerThread * gridDim.x;
    
    for (unsigned int i = blockIdx.x * blockDim.x * elementsPerThread + tid; 
         i < n; 
         i += gridSize) {
        
        for (unsigned int j = 0; j < elementsPerThread; j++) {
            unsigned int idx = i + j * blockDim.x;
            if (idx < n) {
                sum += input[idx];
            }
        }
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    reduceBlock<256>(sdata, tid);
    
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}


int main() {
    run_reduction_tests("v8: Multiple Adds per Thread", 256, 8);
    return 0;
}

