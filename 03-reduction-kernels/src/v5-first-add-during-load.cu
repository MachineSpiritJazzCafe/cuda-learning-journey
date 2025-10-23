// v5-first-add-during-load.cu
// 
// Kernel 4 from Mark Harris's "Optimizing Parallel Reduction in CUDA"
// 
// KEY OPTIMIZATION: Each thread loads TWO elements and performs first reduction 
//                   step during the load itself
//
// WHY IT WORKS:
//   v4: Each thread loads 1 element, then half sit idle in first reduction step
//   v5: Each thread loads 2 elements and adds them immediately
//       → No threads idle during load
//       → Half as many blocks needed
//       → Better work distribution

#include "commons/helpers.cuh"

// ============================================================================
// KERNEL 4: First Add During Global Load
// ============================================================================

__global__ void reduce_in_place(float *g_input, int n) {
    
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    
    // ========================================================================
    // CRITICAL CHANGE FROM V4: Block stride is now 2*blockDim.x
    // ========================================================================
    // Each block handles 2*blockDim.x elements (not just blockDim.x)
    // This is coordinated with the host code via elementsPerThread = 2
    
    unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    
    // ========================================================================
    // FIRST REDUCTION STEP: Happens during global memory load
    // ========================================================================
    // Instead of:
    //   Load 1 element → wait → reduce
    // We do:
    //   Load 2 elements → add immediately → reduce
    
    sdata[tid] = 0.0f;  // Safe initialization for boundary cases
    
    if (index < n) {
        // Load first element
        sdata[tid] = g_input[index];
        
        // Load second element (if it exists) and add it
        // CRITICAL: Must check bounds on second element!
        if (index + blockDim.x < n) {
            sdata[tid] += g_input[index + blockDim.x];
        }
    }
    __syncthreads();
    
    
    // ========================================================================
    // SHARED MEMORY REDUCTION: Identical to v4
    // ========================================================================
    // Sequential addressing (no divergence, no bank conflicts)
    // Note: First reduction step already completed above!
    
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_input[blockIdx.x] = sdata[0];
    }
}


// ============================================================================
// MAIN: Run with elementsPerThread = 2
// ============================================================================

int main() {
    // CRITICAL: Must pass 2 as third parameter (elementsPerThread)
    // This tells helpers.cuh to calculate:
    //   numBlocks = n / (blockSize * 2)
    // instead of:
    //   numBlocks = n / blockSize
    run_reduction_tests("v5: First Add During Load", 256, 2);
    return 0;
}
