#include <cstdlib>
#include <iostream>
#include <math.h>
#include <cstdio>
#include "utils/cuda_utils.cuh"

// function to add the elements of two arrays
__global__ 
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
         y[i] = x[i] + y[i];
}

int main(void) {
    int N = 1<<20; // 1M elements

    float *x, *y;
    
    CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float))); 

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    //Prefetch the x and y arrays to the GPU
    safePrefetch(x, N * sizeof(float), 0);
    safePrefetch(y, N * sizeof(float), 0);
    
    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
              
    add<<<numBlocks, blockSize>>>(N, x, y);

    //Checking for error now since kernel don't return cudaError_t
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max Error: " << maxError << std::endl;

    // Free memory
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));

    return 0;
}
