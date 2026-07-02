// P1 Week 1 - simulate CUDA indexing on CPU.
// Build & run:
//   g++ -std=c++17 -O2 w1_01_indexing_sim.cpp -o sim && ./sim
//
// This file is for learning the CUDA execution model even on a machine
// without nvcc. It prints the same global index formula you will use in kernels.

#include <iostream>
#include <vector>

int main() {
    const int blocks = 3;
    const int threadsPerBlock = 8;
    const int n = 20;

    std::cout << "blocks = " << blocks << "\n";
    std::cout << "threadsPerBlock = " << threadsPerBlock << "\n";
    std::cout << "n = " << n << "\n\n";

    std::vector<int> covered;
    covered.reserve(blocks * threadsPerBlock);

    for (int blockIdx = 0; blockIdx < blocks; ++blockIdx) {
        std::cout << "=== block " << blockIdx << " ===\n";

        for (int threadIdx = 0; threadIdx < threadsPerBlock; ++threadIdx) {
            int globalIdx = blockIdx * threadsPerBlock + threadIdx;
            bool inRange = globalIdx < n;

            std::cout
                << "threadIdx.x=" << threadIdx
                << ", blockIdx.x=" << blockIdx
                << ", blockDim.x=" << threadsPerBlock
                << " -> global idx=" << globalIdx
                << (inRange ? "  (work)" : "  (out of range)")
                << "\n";

            if (inRange) {
                covered.push_back(globalIdx);
            }
        }

        std::cout << "\n";
    }

    std::cout << "Covered element indices: ";
    for (std::size_t i = 0; i < covered.size(); ++i) {
        std::cout << covered[i];
        if (i + 1 < covered.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    return 0;
}
