// P1 Week 5 - grid-stride SAXPY.
// Build & run:
//   nvcc -std=c++17 w5_03_grid_stride_saxpy.cu -o grid_stride_saxpy && ./grid_stride_saxpy
//
// Suggested profiling:
//   nsys profile -o grid_stride_report ./grid_stride_saxpy
//   ncu ./grid_stride_saxpy

#include <cmath>
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

__global__ void saxpyGridStride(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int n = 1 << 24;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> hX(n, 1.0f);
    std::vector<float> hY(n, 2.0f);

    float* dX = nullptr;
    float* dY = nullptr;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dY, bytes));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY.data(), bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = 256;

    saxpyGridStride<<<blocks, threadsPerBlock>>>(n, 2.0f, dX, dY);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hY.data(), dY, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const float expected = 4.0f;
        if (std::fabs(hY[i] - expected) > 1e-6f) {
            std::cout << "Mismatch at " << i
                      << ": got " << hY[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "grid-stride SAXPY passed\n";
    }

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
