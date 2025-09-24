#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA API calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Safe prefetch function that handles WSL2 and other environments
static void safePrefetch(void* ptr, size_t size, int device) {
#ifdef WSL_BUILD
    // Do nothing in WSL2
    static bool warned = false;
    if (!warned) {
        printf("WSL2 detected - prefetch disabled\n");
        warned = true;
    }
#else
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, device, 0));
#endif
}

// Helper function to print device information
static void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("=== CUDA Device Info ===\n");
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s (Compute %d.%d)\n", i, prop.name, prop.major, prop.minor);
        printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
    printf("=========================\n\n");
}

#endif // CUDA_UTILS_CUH
