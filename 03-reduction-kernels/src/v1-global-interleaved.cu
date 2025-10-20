#include "commons/helpers.cuh"

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
        //   tid % (2 * stride) == 0  → Only "even stride" threads work | % op are expensive!
        //   index + stride < n       → Boundary check (don't read out of bounds)
        if (tid % (2 * stride) == 0 && index + stride < n) {
            
            // Pattern: Each active thread sums its element with one "stride" away
            input[index] += input[index + stride];
        }
    }
    
    
    // Block results to global output:
    // Only thread 0 of each block does this
    // After the loop, input[blockIdx.x * blockDim.x] contains the block's sum
    
    if (tid == 0) {
        // Each block writes its result to the beginning of its section
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
    }
}

int main() {
    run_reduction_tests("v1: Naive Interleanved Addressing"); // Uses default blockSize
    return 0;
}
