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

int main() {
    run_reduction_tests("v1: Naive Interleanved Addressing"); // Uses default blockSize
    return 0;
}
