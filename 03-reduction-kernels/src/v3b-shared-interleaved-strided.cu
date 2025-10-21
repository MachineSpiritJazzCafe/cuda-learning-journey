#include "commons/helpers.cuh"

// v3-shared-interleaved-strided: Corresponds to NVIDIA Optimizing Parallel
// Reductiom in CUDA by Mark Harris
// IMPROVEMENT: Removes divergent branching
// PROBLEM: Creates shared memory bank conflicts! (subject to architecture)
__global__ void reduce_in_place(float *input, int n) {
    
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
 // ========================================================================
    // STEP 1: Load from GLOBAL to SHARED
    // ========================================================================
    sdata[tid] = 0.0f;
    if (index < n) {
        sdata[tid] = input[index];
    }
    __syncthreads();
    
    
    // ========================================================================
    // STEP 2: Interleaved addressing with STRIDED INDEX (Kernel 2 pattern)
    // ========================================================================
    // This removes divergent branching but CREATES BANK CONFLICTS!
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        
        // STRIDED INDEX PATTERN (causes bank conflicts)
        int idx = 2 * s * tid;  // Notice: ALL threads compute index
        
        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx + s];
        }
        
        __syncthreads();
    }
    
    // ========================================================================
    // STEP 3: Write result back to GLOBAL memory
    // ========================================================================
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }     
}

int main(){
    run_reduction_tests("v3-shared-interleaved-strided");
    return 0;
}
