// P1 Week 3 - add basic CUDA error checking.
// Build & run:
//   nvcc -std=c++17 08_checked_vector_add.cu -o checked_vadd && ./checked_vadd
//
// This is a more engineering-style vector add:
// - wraps CUDA API calls with a check macro
// - checks launch errors
// - checks execution errors after synchronize

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

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1 << 16;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> hA(n);
    std::vector<float> hB(n);
    std::vector<float> hC(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(dA, dB, dC, n);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-6f) {
            std::cout << "Mismatch at " << i
                      << ": got " << hC[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "checked vectorAdd passed\n";
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
