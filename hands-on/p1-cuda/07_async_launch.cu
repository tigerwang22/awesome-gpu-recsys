// P1 Week 3 - kernel launch is asynchronous.
// Build & run:
//   nvcc -std=c++17 07_async_launch.cu -o async_launch && ./async_launch
//
// This example uses a busy-wait kernel to make the timing difference visible.

#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

__global__ void spinKernel(unsigned long long cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

int main() {
    using clock = std::chrono::steady_clock;

    const unsigned long long cycles = 500000000ULL;

    auto t0 = clock::now();
    std::cout << "launching kernel...\n";
    spinKernel<<<1, 1>>>(cycles);
    auto t1 = clock::now();

    std::cout << "host kept going immediately after launch\n";
    std::cout << "host-side launch call took about "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    cudaDeviceSynchronize();
    auto t2 = clock::now();

    std::cout << "after cudaDeviceSynchronize(), total elapsed time is about "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count()
              << " ms\n";

    return 0;
}
