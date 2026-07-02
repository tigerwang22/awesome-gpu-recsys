// P1 Week 6 - use the occupancy API for a starting block size hint.
// Build & run:
//   nvcc -std=c++17 w6_02_occupancy_hint.cu -o occupancy_hint && ./occupancy_hint

#include <cstdlib>
#include <iostream>

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

__global__ void saxpyGridStride(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int minGridSize = 0;
    int suggestedBlockSize = 0;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &suggestedBlockSize,
        saxpyGridStride,
        0,
        0));

    std::cout << "occupancy API suggested block size = " << suggestedBlockSize << "\n";
    std::cout << "minimum grid size to reach that suggestion = " << minGridSize << "\n";
    std::cout << "treat this as a starting point, not a final answer\n";

    return EXIT_SUCCESS;
}
