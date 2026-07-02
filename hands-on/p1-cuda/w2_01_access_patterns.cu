// P1 Week 2 - inspect memory access patterns.
// Build & run:
//   nvcc -std=c++17 w2_01_access_patterns.cu -o access && ./access
//
// This program prints which element each thread would read under different
// access patterns. It is not a performance benchmark; it is a visibility tool.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void showPattern(const char* label, int stride) {
    int idx = threadIdx.x * stride;
    printf("%s: thread %d reads element %d\n", label, threadIdx.x, idx);
}

int main() {
    printf("=== Coalesced pattern (stride = 1) ===\n");
    showPattern<<<1, 8>>>("coalesced", 1);
    cudaDeviceSynchronize();

    printf("\n=== Strided pattern (stride = 4) ===\n");
    showPattern<<<1, 8>>>("strided", 4);
    cudaDeviceSynchronize();

    return 0;
}
