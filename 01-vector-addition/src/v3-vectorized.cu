#include <cuda_runtime.h>
#include "utils/cuda_utils.cuh"


// const - a and b are read-only, more agressive cache, texture cache, l1 cache
// __restrict__ ensures non overlaping memory between marked variables
__global__ void add(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float* __restrict__ c, 
                    int n) {
       
    // Skipping by 4 since each thread processes 4 elements
    int idx = blockDim.x * blockIdx.x + threadIdx.x * 4;
    
    if (idx + 3 < n) {
        float4 a4 = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b4 = *reinterpret_cast<const float4*>(&b[idx]);

        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        
        *reinterpret_cast<float4*>(&c[idx]) = c4;
    }

    // Handle tail elements (last 0-3 elements)
    // Only first block processes tail to avoid race conditions
    if (blockIdx.x == 0) {  // only first block
        int remainder = (n / 4) * 4;
        for (int i = remainder + threadIdx.x; i < n; i += blockDim.x) {
            c[i] = a[i] + b[i];
        }
    }    
}

void vectorizedAdd(float* h_a, float* h_b, float* h_c, long long N) {
    
    //Use ~70% of GPU memory for safety
    size_t freeMemory, totalMemory;
    CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));

    //Chunking...
    size_t maxElementsPerChunk = (freeMemory * 0.7) / (3 * sizeof(float));
    size_t chunkSizeBytes = maxElementsPerChunk * sizeof(float);

    printf("GPU Memory: %zu MB free, using chunks of %zu elements, and %zu bytes per chunk\n",
           freeMemory / (1024 * 1024), maxElementsPerChunk, chunkSizeBytes);
        
    //Allocate on device memory for chunks
    float *d_a, *d_b, *d_c;

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
        // Need 4 times fewer threads
        int numBlocks = ((currentChunkSize + 3) / 4 + blockSize - 1) / blockSize;    

        // Launch kernel on current chunk
        add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, currentChunkSize);
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
    long long N = 500 * (1 << 20); // 3 * vector size > GPU memory
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
    vectorizedAdd(h_a, h_b, h_c, N);
    
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
