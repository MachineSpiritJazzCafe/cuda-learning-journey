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
            
    // ================================================================
    // VISUAL EXAMPLE: blockDim.x = 8, Block 0
    // ================================================================
    // 
    // Initial: input[0..7] = [a, b, c, d, e, f, g, h]
    // 
    // ITERATION 1 (stride = 4):
    //   Thread 0 (tid=0, index=0): input[0] += input[4]  → [a+e, b, c, d, e, f, g, h]
    //   Thread 1 (tid=1, index=1): input[1] += input[5]  → [a+e, b+f, c, d, e, f, g, h]
    //   Thread 2 (tid=2, index=2): input[2] += input[6]  → [a+e, b+f, c+g, d, e, f, g, h]
    //   Thread 3 (tid=3, index=3): input[3] += input[7]  → [a+e, b+f, c+g, d+h, e, f, g, h]
    //   Threads 4-7: IDLE (tid >= stride)
    // 
    // ITERATION 2 (stride = 2):
    //   Thread 0: input[0] += input[2]  → [a+e+c+g, b+f, c+g, d+h, ...]
    //   Thread 1: input[1] += input[3]  → [a+e+c+g, b+f+d+h, c+g, d+h, ...]
    //   Threads 2-7: IDLE
    // 
    // ITERATION 3 (stride = 1):
    //   Thread 0: input[0] += input[1]  → [a+b+c+d+e+f+g+h, ...]
    //   Threads 1-7: IDLE
    // 
    // ================================================================
        }
    }
    
    // Write block's result to beginning of output array
    if (tid == 0) {
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
    }
}

// ============================================================================
// OPTIMIZATION ANALYSIS: What changed from v1?
// ============================================================================
// 
// IMPROVEMENT 1: No Modulo Operation
// ----------------------------------
// V1: tid % (2*stride) == 0  → ~40-100 cycles (integer division)
// V2: tid < stride           → 1 cycle (simple comparison)
// 
// Impact: Eliminates expensive modulo from critical path
// 
// 
// IMPROVEMENT 2: Better Memory Coalescing
// ----------------------------------------
// V1 Memory Access Pattern (stride=1):
//   Thread 0: reads input[0], input[1]
//   Thread 2: reads input[2], input[3]   ← NOT CONSECUTIVE
//   Thread 4: reads input[4], input[5]
//   Thread 6: reads input[6], input[7]
//   → Multiple 32-byte memory transactions
// 
// V2 Memory Access Pattern (stride=4):
//   Thread 0: reads input[0], input[4]   ← CONSECUTIVE threads
//   Thread 1: reads input[1], input[5]      read consecutive memory
//   Thread 2: reads input[2], input[6]
//   Thread 3: reads input[3], input[7]
//   → Better chance of coalescing into fewer transactions
// 
// 
// IMPROVEMENT 3: Reduced Branch Divergence
// -----------------------------------------
// V1: Within each warp, threads alternate active/idle
//     Warp 0: [ACTIVE, idle, ACTIVE, idle, ACTIVE, idle, ...]
// 
// V2: Within each warp, first half active, second half idle
//     Warp 0: [ACTIVE, ACTIVE, ACTIVE, ACTIVE, idle, idle, ...]
//     → Still divergent, but more predictable
// 
// 
// EXPECTED SPEEDUP: ~1.5-2x over v1
// ==================================
// - Modulo elimination: ~1.2-1.3x
// - Better coalescing: ~1.2-1.5x
// - Combined: ~1.5-2x
// 
// STILL SLOW BECAUSE:
// - Global memory (400-800 cycles per access)
// - Half the threads idle each iteration
// - No shared memory caching
// 
// NEXT STEP: v3 will add shared memory
// ============================================================================

int main() {
    run_reduction_tests("v2: Global Memory + Sequential Addressing (No Modulo)", 256);
    return 0;
}
