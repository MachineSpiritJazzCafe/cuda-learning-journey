#include "commons/helpers.cuh"

// v2: Global Memory + Sequential Addressing (No Modulo)
// ISOLATES: Cost of modulo operation vs simple comparison
// STILL USES: Global memory (400-800 cycle latency)
__global__ void reduce_in_place(float *input, int n){
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    
    // ========================================================================
    // KEY CHANGE FROM V1: Sequential addressing instead of interleaved
    // ========================================================================
    // V1: if (tid % (2*stride) == 0)  ← Expensive modulo, scattered threads
    // V2: if (tid < stride)           ← Cheap comparison, contiguous threads
    // ========================================================================
    
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();

        if (tid < stride) {
            unsigned int read_index = index + stride;
            
            // Boundary check: Don't read past end of array
            if (read_index < n) {
                input[index] += input[read_index];
            }
        }
    }       
        // Write block's result to beginning of output array
        if (tid == 0) {
            input[blockIdx.x] = input[blockIdx.x * blockDim.x];
        }
}

int main() {
    run_reduction_tests("v2: Global Memory + Sequential Addressing (No Modulo");
    return 0;
}

