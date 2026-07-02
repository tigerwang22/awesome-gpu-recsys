// P1 Week 4 - use one stream as an ordered async pipeline.
// Build & run:
//   nvcc -std=c++17 w4_02_stream_pipeline.cu -o stream_pipeline && ./stream_pipeline

#include <cmath>
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

__global__ void squareKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

int main() {
    const int n = 1 << 20;
    const std::size_t bytes = n * sizeof(float);

    float* hInput = nullptr;
    float* hOutput = nullptr;
    CHECK_CUDA(cudaMallocHost(&hInput, bytes));
    CHECK_CUDA(cudaMallocHost(&hOutput, bytes));

    for (int i = 0; i < n; ++i) {
        hInput[i] = static_cast<float>(i % 100);
        hOutput[i] = 0.0f;
    }

    float* dInput = nullptr;
    float* dOutput = nullptr;
    CHECK_CUDA(cudaMalloc(&dInput, bytes));
    CHECK_CUDA(cudaMalloc(&dOutput, bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CUDA(cudaMemcpyAsync(dInput, hInput, bytes, cudaMemcpyHostToDevice, stream));
    squareKernel<<<blocks, threadsPerBlock, 0, stream>>>(dInput, dOutput, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(hOutput, dOutput, bytes, cudaMemcpyDeviceToHost, stream));

    std::cout << "copy -> kernel -> copy have all been enqueued on one stream\n";
    std::cout << "host is free until we explicitly synchronize the stream\n";

    CHECK_CUDA(cudaStreamSynchronize(stream));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = hInput[i] * hInput[i];
        if (std::fabs(hOutput[i] - expected) > 1e-5f) {
            std::cout << "Mismatch at " << i
                      << ": got " << hOutput[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "stream pipeline passed\n";
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dInput));
    CHECK_CUDA(cudaFree(dOutput));
    CHECK_CUDA(cudaFreeHost(hInput));
    CHECK_CUDA(cudaFreeHost(hOutput));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
