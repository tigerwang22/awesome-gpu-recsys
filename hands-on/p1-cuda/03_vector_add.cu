// P1 Week 1 - vector add.
// Build & run:
//   nvcc -std=c++17 03_vector_add.cu -o vadd && ./vadd
//
// Each thread handles one element:
//   c[idx] = a[idx] + b[idx]

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 16;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> hA(n);
    std::vector<float> hB(n);
    std::vector<float> hC(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(100 + i);
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 8;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(dA, dB, dC, n);

    cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-6f) {
            std::cout << "Mismatch at " << i
                      << ": got " << hC[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "vectorAdd passed\n";
        std::cout << "First few outputs: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << hC[i];
            if (i + 1 < 5) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
