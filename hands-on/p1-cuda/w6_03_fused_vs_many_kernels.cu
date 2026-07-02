// P1 Week 6 - compare many tiny launches vs one fused kernel.
// Build & run:
//   nvcc -std=c++17 w6_03_fused_vs_many_kernels.cu -o fused_vs_many && ./fused_vs_many

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

__global__ void incrementOnce(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

__global__ void incrementFused(float* data, int n, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        for (int s = 0; s < steps; ++s) {
            value += 1.0f;
        }
        data[idx] = value;
    }
}

float timeManyLaunches(float* dData, int n, int steps, int blocks, int threadsPerBlock) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < steps; ++s) {
        incrementOnce<<<blocks, threadsPerBlock>>>(dData, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

float timeFusedLaunch(float* dData, int n, int steps, int blocks, int threadsPerBlock) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    incrementFused<<<blocks, threadsPerBlock>>>(dData, n, steps);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

int main() {
    const int n = 1 << 20;
    const int steps = 100;
    const std::size_t bytes = n * sizeof(float);
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> zeros(n, 0.0f);
    std::vector<float> hOut(n, 0.0f);

    float* dData = nullptr;
    CHECK_CUDA(cudaMalloc(&dData, bytes));

    CHECK_CUDA(cudaMemcpy(dData, zeros.data(), bytes, cudaMemcpyHostToDevice));
    float manyMs = timeManyLaunches(dData, n, steps, blocks, threadsPerBlock);

    CHECK_CUDA(cudaMemcpy(dData, zeros.data(), bytes, cudaMemcpyHostToDevice));
    float fusedMs = timeFusedLaunch(dData, n, steps, blocks, threadsPerBlock);

    CHECK_CUDA(cudaMemcpy(hOut.data(), dData, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (float value : hOut) {
        if (std::fabs(value - static_cast<float>(steps)) > 1e-6f) {
            ok = false;
            break;
        }
    }

    std::cout << "many tiny launches time: " << manyMs << " ms\n";
    std::cout << "one fused launch time:   " << fusedMs << " ms\n";
    if (ok) {
        std::cout << "fusion comparison passed\n";
    }

    CHECK_CUDA(cudaFree(dData));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
