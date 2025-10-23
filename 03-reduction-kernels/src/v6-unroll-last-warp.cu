// v6-unroll-last-warp.cu
//
// Kernel 5 from Mark Harris's "Optimizing Parallel Reduction in CUDA"
//
// KEY OPTIMIZATION: Unroll the last 6 iterations (when s <= 32)
//                   Within a warp, threads are SIMD synchronous, so:
//                   - No need for __syncthreads()
//                   - No need for if (tid < s) checks
//
// WHY IT WORKS:
//   When s <= 32, only one warp is active. Within a warp:
//   - All threads execute in lockstep (SIMD)
//   - __syncthreads() is unnecessary (and expensive!)
//   - Branch checks don't save work (all threads execute anyway)
//   By unrolling, we eliminate this overhead

#include "commons/helpers.cuh"

// ============================================================================
// DEVICE FUNCTION: Warp Reduction (Unrolled)
// ============================================================================
// This handles the last 6 iterations when s <= 32
// CRITICAL: sdata must be volatile to prevent compiler optimization issues!
//
// Why volatile?
// - Prevents compiler from optimizing away seemingly "redundant" reads
// - Ensures writes are visible to other threads in the warp immediately
// - Without volatile, compiler might cache values in registers
//
// Within a warp (32 threads):
// - All threads execute in SIMD lockstep
// - No need for __syncthreads() (they're already synchronized!)
// - No need for if statements (doesn't save work)

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    // Unroll last 6 iterations: s = 32, 16, 8, 4, 2, 1
    sdata[tid] += sdata[tid + 32];  // s = 32
    sdata[tid] += sdata[tid + 16];  // s = 16
    sdata[tid] += sdata[tid + 8];   // s = 8
    sdata[tid] += sdata[tid + 4];   // s = 4
    sdata[tid] += sdata[tid + 2];   // s = 2
    sdata[tid] += sdata[tid + 1];   // s = 1
}

__global__ void reduce_in_place(float *input, int n) {
    
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // ========================================================================
    // STEP 1: First add during load (from v5)
    // ========================================================================
    sdata[tid] = 0.0f;
    if (index < n) {
        sdata[tid] = input[index];
        if (index + blockDim.x < n) {
            sdata[tid] += input[index + blockDim.x];
        }
    }
    __syncthreads();
    
    
    // ========================================================================
    // STEP 2: Reduction loop (but stop at s > 32)
    // ========================================================================
    // KEY CHANGE: Loop only while s > 32
    // Once s <= 32, we have only one warp left → handle with warpReduce()
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // ========================================================================
    // STEP 3: Final warp reduction (unrolled, no syncs needed!)
    // ========================================================================
    // When we reach here, s = 32, and only threads 0-31 have work to do
    // But all 256 threads will execute warpReduce() - only tid < 32 matters
    //
    // The if statement here DOES save work for warps 1-7 (threads 32-255)
    // They can skip the function entirely
    
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}


int main() {
    // Still using elementsPerThread = 2 from v5
    run_reduction_tests("v6: Unroll Last Warp", 256, 2);
    return 0;
}
