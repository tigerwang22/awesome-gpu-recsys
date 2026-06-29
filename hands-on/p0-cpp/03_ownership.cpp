// P0 Week 1 - the RAII trap: what happens when you COPY an owning object?
// Build & run:
//   g++ -std=c++17 -O2 03_ownership.cpp -o own ; ./own
//
// This program has a BUG on purpose. Watch the [release] addresses.

#include <iostream>

// An RAII class that owns a heap int.
class Owner {
public:
    Owner(int v) : data_(new int(v)) {
        std::cout << "  [acquire] ptr=" << data_ << " value=" << *data_ << "\n";
    }
    ~Owner() {
        std::cout << "  [release] ptr=" << data_ << "\n";
        delete data_;   // free the heap int
    }
private:
    int* data_;
};

int main() {
    std::cout << std::unitbuf;  // make cout flush every time, so we see output before any crash
    std::cout << "create a\n";
    Owner a(10);

    std::cout << "copy a into b (Owner b = a;)\n";
    Owner b = a;   // <-- default copy: copies the POINTER, not the heap int.
                   //     now a.data_ and b.data_ point to the SAME heap int!

    std::cout << "end of main (both destructors will run)\n";
    return 0;
    // ~b runs: delete that pointer
    // ~a runs: delete the SAME pointer AGAIN  -> double free -> crash/corruption
}
