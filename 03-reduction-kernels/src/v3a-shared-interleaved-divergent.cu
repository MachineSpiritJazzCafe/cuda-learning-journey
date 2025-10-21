#include "commons/helpers.cuh"

// v3: Shared Memory + Interleaved Addressing (V1 progression)
// Lesson learned: Just adding shared memory doesn't magically improves metrics.
// STILL HAS: Expensive modulo operation and branch divergence
// Inte leaved implementation causes bank conflicts in shared memory
__global__ void reduce_in_place(float* input, int n) {
    
    // Shared memory: ~5 cycle latency vs ~400-800 for global memory
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ========================================================================
    // STEP 1: Load from GLOBAL to SHARED (with boundary check)
    // ========================================================================
    // This is the ONLY global memory access during reduction!
    // After this, everything happens in fast shared memory
    
    sdata[tid] = 0.0f;  // Initialize (handles partial blocks)
    if (index < n) {
        sdata[tid] = input[index];
    }
    __syncthreads();  // Ensure all threads have loaded their data
    
    
    // ========================================================================
    // STEP 2: Tree-based reduction IN SHARED MEMORY
    // ========================================================================
    // Same interleaved pattern as v1, but now operating on sdata[] not global
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
       
        // INTERLEAVED ADDRESSING (same as v1):
        // Active threads: 0, 2, 4, 6, ... (scattered)
        if (tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
            
            // Add element 'stride' away
            sdata[tid] += sdata[tid + stride];
        }

     __syncthreads();
   
    }
    
    // ========================================================================
    // STEP 3: Write result back to GLOBAL memory
    // ========================================================================
    // Only thread 0 has the final sum in sdata[0]
    
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}

int main() {
    run_reduction_tests("v3: Shared Memory + Interleaved Addressing");
    return 0;
}
