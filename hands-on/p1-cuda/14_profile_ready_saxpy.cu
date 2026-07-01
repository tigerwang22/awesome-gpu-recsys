// P1 Week 5 - profile-ready SAXPY baseline.
// Build & run:
//   nvcc -std=c++17 14_profile_ready_saxpy.cu -o profile_saxpy && ./profile_saxpy
//
// Suggested profiling:
//   nsys profile -o profile_saxpy_report ./profile_saxpy
//   ncu ./profile_saxpy

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

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    const int n = 1 << 22;
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
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    saxpy<<<blocks, threadsPerBlock>>>(n, 2.0f, dX, dY); // warmup
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int iter = 0; iter < 100; ++iter) {
        saxpy<<<blocks, threadsPerBlock>>>(n, 2.0f, dX, dY);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hY.data(), dY, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(hY[i])) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "profile-ready SAXPY completed 100 iterations\n";
    }

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
