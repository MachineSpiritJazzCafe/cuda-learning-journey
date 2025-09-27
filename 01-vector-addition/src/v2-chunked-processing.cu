#include <stdio.h>
#include <cuda_runtime.h>
#include "utils/cuda_utils.cuh"

__global__ void add(int n, float* a, float* b, float* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += stride){
        c[i] = a[i] + b[i];
    }
}


void chunkedAdd(float* h_a, float* h_b, float* h_c, long long N) {
    // Memory strategy: Use ~70% of GPU memory for safety
    size_t freeMemory, totalMemory;
    CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));

    // Reserve memory for 3 vectors + overhead
    size_t maxElementsPerChunk = (freeMemory * 0.7) / (3 * sizeof(float));


    printf("GPU Memory: %zu MB free, using chunks of %zu elements\n", 
           freeMemory / (1024*1024), maxElementsPerChunk);
        
    // Allocate device memory for chunks
    float *d_a, *d_b, *d_c;
    size_t chunkSizeBytes = maxElementsPerChunk * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_a, chunkSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_b, chunkSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_c, chunkSizeBytes));
    
    for (long long offset = 0; offset < N; offset += maxElementsPerChunk) {
        // Calculate current chunk size
        long long currentChunkSize = std::min(maxElementsPerChunk, (size_t)(N - offset));
        size_t currentChunkBytes = currentChunkSize * sizeof(float);
        
        printf("Processing chunk: elements %lld to %lld (%lld elements)\n", 
               offset, offset + currentChunkSize - 1, currentChunkSize);
        
        // Copy chunk to device
        CUDA_CHECK(cudaMemcpy(d_a, &h_a[offset], currentChunkBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, &h_b[offset], currentChunkBytes, cudaMemcpyHostToDevice));
        
        // Calculate grid size for current chunk
        int blockSize = 256;
        int numBlocks = (currentChunkSize + blockSize - 1) / blockSize;
        
        // Launch kernel on current chunk
        add<<<numBlocks, blockSize>>>(currentChunkSize, d_a, d_b, d_c);
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for kernel to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(&h_c[offset], d_c, currentChunkBytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    // Test with vectors larger than GPU memory
    long long N = 500 * (1 << 20);  // 524M elements = ~6GB total (3 vectors)
    
    printf("Allocating vectors with %lld elements (~%.2f GB total)\n", 
           N, (3.0 * N * sizeof(float)) / (1024*1024*1024));
    
    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    
    // Initialize vectors
    printf("Initializing vectors...\n");
    for (long long i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Perform chunked addition
    printf("Starting chunked vector addition...\n");
    chunkedAdd(h_a, h_b, h_c, N);
    
    // Verify results (check a few elements)
    bool success = true;
    for (int i = 0; i < 100; i++) {
        if (abs(h_c[i] - 3.0f) > 1e-5) {
            success = false;
            break;
        }
    }
    
    printf("Verification: %s\n", success ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}
