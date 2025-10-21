  #include "commons/helpers.cuh"

// v4: Shared Memory + Sequential Addressing
// COMBINES: Fast shared memory (v3) + No modulo (v2)
// KEY INSIGHT: Still seeing bank conflicts!
__global__ void reduce_in_place(float* g_input, int n) {
    
    // Shared memory: ~5 cycle latency vs ~400-800 for global memory
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ========================================================================
    // STEP 1: Load from GLOBAL to SHARED (with boundary check)
    // ========================================================================
    sdata[tid] = 0.0f;  // Initialize (handles partial blocks)
    if (index < n) {
        sdata[tid] = g_input[index];
    }
    __syncthreads();  

    // ========================================================================
    // STEP 2: Tree-based reduction IN SHARED MEMORY
    // ========================================================================
    // KEY CHANGE FROM V3: Sequential addressing instead of interleaved
    // V3: if (tid % (2*stride) == 0)  ← Expensive modulo + bank conflicts
    // V4: if (tid < stride)           ← Cheap comparison + fewer bank conflicts
    
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            // Add element 'stride' away
            sdata[tid] += sdata[tid + stride];
        }

    __syncthreads();
    
    }
    
    // ========================================================================
    // STEP 3: Write result back to GLOBAL memory
    // ========================================================================
    if (tid == 0) {
        g_input[blockIdx.x] = sdata[0];
    }
}

int main() {
    run_reduction_tests("v4: Shared Memory + Sequential Addressing)");
    return 0;
}

