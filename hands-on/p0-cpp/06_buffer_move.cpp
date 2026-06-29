// P0 Week 2 - build a move-only resource owner.
// Build & run:
//   g++ -std=c++17 -O2 06_buffer_move.cpp -o buffer_move && ./buffer_move
//
// This class is a CPU-memory stand-in for something like a future CudaBuffer:
// it owns a heap allocation, forbids copy, and supports move.

#include <cstddef>
#include <iostream>
#include <utility>

class Buffer {
public:
    explicit Buffer(std::size_t size) : size_(size), data_(size ? new int[size] : nullptr) {
        std::cout << "[acquire] size=" << size_ << " ptr=" << data_ << "\n";
        for (std::size_t i = 0; i < size_; ++i) {
            data_[i] = static_cast<int>(i * 10);
        }
    }

    ~Buffer() {
        std::cout << "[release] size=" << size_ << " ptr=" << data_ << "\n";
        delete[] data_;
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : size_(other.size_), data_(other.data_) {
        std::cout << "[move ctor] take ptr=" << data_ << "\n";
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        std::cout << "[move assign] old ptr=" << data_ << " new ptr=" << other.data_ << "\n";
        delete[] data_;

        size_ = other.size_;
        data_ = other.data_;
        other.size_ = 0;
        other.data_ = nullptr;
        return *this;
    }

    void printFirst(const char* label) const {
        std::cout << label << ": ptr=" << data_ << ", size=" << size_;
        if (data_ && size_ > 0) {
            std::cout << ", first=" << data_[0];
        }
        std::cout << "\n";
    }

private:
    std::size_t size_ = 0;
    int* data_ = nullptr;
};

Buffer makeBuffer(std::size_t size) {
    Buffer tmp(size);
    tmp.printFirst("makeBuffer tmp");
    return tmp;
}

int main() {
    Buffer a(4);
    a.printFirst("a before move");

    Buffer b = std::move(a);
    a.printFirst("a after move");
    b.printFirst("b after move ctor");

    Buffer c(2);
    c = makeBuffer(6);
    c.printFirst("c after move assign");

    return 0;
}
