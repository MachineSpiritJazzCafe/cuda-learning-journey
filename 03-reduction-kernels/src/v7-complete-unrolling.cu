// v7-complete-unrolling.cu
//
// Kernel 6 from Mark Harris's "Optimizing Parallel Reduction in CUDA"
//
// KEY OPTIMIZATION: Use C++ templates to unroll the ENTIRE reduction loop
//                   blockSize becomes a compile-time constant
//                   Compiler can eliminate all loop overhead and dead branches
//
// WHY IT WORKS:
//   With templates, the compiler knows blockSize at compile time:
//   - Can unroll the entire loop
//   - Can eliminate if statements (dead code elimination)
//   - Can optimize instruction scheduling
//   - Can perform constant folding

#include "commons/helpers.cuh"

// ============================================================================
// TEMPLATED REDUCTION FUNCTION
// ============================================================================
// This replaces the reduction loop entirely!
// The template parameter blockSize is known at compile time,
// so the compiler can optimize aggressively.
//
// For blockSize = 256:
//   Unrolls to: s = 128, 64, 32, 16, 8, 4, 2, 1
//
// The if statements look redundant, but they're compile-time constants
// The compiler eliminates dead branches.

template <unsigned int blockSize>
__device__ void reduceBlock(volatile float* sdata, unsigned int tid) {
    
    // Unroll iterations where s > 32 (need __syncthreads())
    // The "if (blockSize >= X)" checks are evaluated at COMPILE TIME!
    // If false, the entire block is eliminated by the compiler.
    
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    // Final warp (s = 32, 16, 8, 4, 2, 1)
    // No __syncthreads() needed - within one warp!
    // The if (tid < 32) check saves warps 1-7 from executing
    
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
// TEMPLATED KERNEL
// ============================================================================
// The kernel is now templated on blockSize
// This allows the compiler to specialize the kernel for each block size

template <unsigned int blockSize>
__global__ void reduce_kernel(float *input, int n) {
    
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    
    // ========================================================================
    // STEP 1: First add during load (from v5)
    // ========================================================================
    sdata[tid] = 0.0f;
    if (i < n) {
        sdata[tid] = input[i];
        if (i + blockSize < n) {
            sdata[tid] += input[i + blockSize];
        }
    }
    __syncthreads();
    
    
    // ========================================================================
    // STEP 2: Completely unrolled reduction
    // ========================================================================
    // This single function call replaces the entire reduction loop!
    // The compiler will inline it and optimize away all dead branches.
    
    reduceBlock<blockSize>(sdata, tid);
    
    
    // Write result
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}


// ============================================================================
// NON-TEMPLATED WRAPPER (for backward compatibility with helpers.cuh)
// ============================================================================
// helpers.cuh expects a kernel named "reduce_in_place"
// We create a wrapper that dispatches to the templated kernel

__global__ void reduce_in_place(float *input, int n) {
    // This is a bit hacky, but avoids refactoring of helpers.cuh
    // Also good as now we have examples without and with templating
    // ***
    // We know blockDim.x at runtime, dispatch to appropriate template
    // For simplicity, we only support blockSize = 256
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    sdata[tid] = 0.0f;
    if (i < n) {
        sdata[tid] = input[i];
        if (i + blockDim.x < n) {
            sdata[tid] += input[i + blockDim.x];
        }
    }
    __syncthreads();
    
    reduceBlock<256>(sdata, tid);
    
    if (tid == 0) {
        input[blockIdx.x] = sdata[0];
    }
}

int main() {
    run_reduction_tests("v7: Complete Unrolling", 256, 2);
    return 0;
}
