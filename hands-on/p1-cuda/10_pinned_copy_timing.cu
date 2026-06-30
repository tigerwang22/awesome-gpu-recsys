// P1 Week 4 - compare pageable vs pinned host memory copy timing.
// Build & run:
//   nvcc -std=c++17 10_pinned_copy_timing.cu -o pinned_copy && ./pinned_copy

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

int main() {
    const int n = 1 << 22;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> pageable(n, 1.0f);

    float* pinned = nullptr;
    CHECK_CUDA(cudaMallocHost(&pinned, bytes));
    for (int i = 0; i < n; ++i) {
        pinned[i] = 1.0f;
    }

    float* dData = nullptr;
    CHECK_CUDA(cudaMalloc(&dData, bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < 50; ++iter) {
        CHECK_CUDA(cudaMemcpy(dData, pageable.data(), bytes, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float pageableMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&pageableMs, start, stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < 50; ++iter) {
        CHECK_CUDA(cudaMemcpy(dData, pinned, bytes, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float pinnedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&pinnedMs, start, stop));

    std::cout << "50 pageable H2D copies took about " << pageableMs << " ms\n";
    std::cout << "50 pinned   H2D copies took about " << pinnedMs << " ms\n";
    std::cout << "average pageable copy: " << (pageableMs / 50.0f) << " ms\n";
    std::cout << "average pinned   copy: " << (pinnedMs / 50.0f) << " ms\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dData));
    CHECK_CUDA(cudaFreeHost(pinned));
    return EXIT_SUCCESS;
}
