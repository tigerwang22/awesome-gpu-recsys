// P1 Week 4 - process two chunks with two streams.
// Build & run:
//   nvcc -std=c++17 12_two_stream_saxpy.cu -o two_stream_saxpy && ./two_stream_saxpy
//
// This example is about structure, not about proving perfect overlap.

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

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    const int n = 1 << 20;
    const int chunkSize = n / 2;
    const std::size_t bytes = n * sizeof(float);

    float* hX = nullptr;
    float* hY = nullptr;
    CHECK_CUDA(cudaMallocHost(&hX, bytes));
    CHECK_CUDA(cudaMallocHost(&hY, bytes));

    for (int i = 0; i < n; ++i) {
        hX[i] = 1.0f;
        hY[i] = 2.0f;
    }

    float* dX = nullptr;
    float* dY = nullptr;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dY, bytes));

    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));

    const int threadsPerBlock = 256;

    for (int s = 0; s < 2; ++s) {
        const int offset = s * chunkSize;
        const int elementsThisChunk = (s == 0) ? chunkSize : (n - offset);
        const std::size_t bytesThisChunk = elementsThisChunk * sizeof(float);
        const int blocks = (elementsThisChunk + threadsPerBlock - 1) / threadsPerBlock;

        CHECK_CUDA(cudaMemcpyAsync(
            dX + offset, hX + offset, bytesThisChunk, cudaMemcpyHostToDevice, streams[s]));
        CHECK_CUDA(cudaMemcpyAsync(
            dY + offset, hY + offset, bytesThisChunk, cudaMemcpyHostToDevice, streams[s]));

        saxpy<<<blocks, threadsPerBlock, 0, streams[s]>>>(
            elementsThisChunk, 2.0f, dX + offset, dY + offset);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpyAsync(
            hY + offset, dY + offset, bytesThisChunk, cudaMemcpyDeviceToHost, streams[s]));
    }

    std::cout << "two chunks have been enqueued on two streams\n";

    CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    CHECK_CUDA(cudaStreamSynchronize(streams[1]));

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
        std::cout << "two-stream SAXPY passed\n";
    }

    CHECK_CUDA(cudaStreamDestroy(streams[0]));
    CHECK_CUDA(cudaStreamDestroy(streams[1]));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFreeHost(hX));
    CHECK_CUDA(cudaFreeHost(hY));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
