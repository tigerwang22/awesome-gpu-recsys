// P1 Week 1 - inspect CUDA thread indexing.
// Build & run:
//   nvcc -std=c++17 02_thread_indexing.cu -o indexing && ./indexing

#include <cstdio>
#include <cuda_runtime.h>

__global__ void inspectIndexing(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf(
        "blockIdx.x=%d threadIdx.x=%d blockDim.x=%d -> idx=%d %s\n",
        blockIdx.x,
        threadIdx.x,
        blockDim.x,
        idx,
        idx < n ? "[active]" : "[skip]"
    );
}

int main() {
    const int n = 20;
    const int threadsPerBlock = 8;
    const int blocks = 3;

    inspectIndexing<<<blocks, threadsPerBlock>>>(n);
    cudaDeviceSynchronize();
    return 0;
}
