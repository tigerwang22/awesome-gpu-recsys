// P0 Week 1 - Exercise 1: pass by value / pointer / reference, heap memory, RAII.
// Build & run:
//   g++ -std=c++17 -O2 01_pointers.cpp -o out && ./out
//
// See README.md for the concepts behind this file.

#include <iostream>
#include <vector>
#include <memory>

// ---- Part 1: pass by value / pointer / reference ----

void addOneByValue(int x) {
    x = x + 1; // changes only the local copy
}

void addOneByPointer(int* x) {
    *x = *x + 1;
}

void addOneByReference(int& x) {
    x = x + 1; // x is an alias for the caller's variable: change it directly, no * needed
}

// ---- Part 2: heap memory, manual lifetime ----

int* makeIntOnHeap(int value) {
    // Allocate an int on the heap. The caller becomes responsible for deleting it.
    return new int(value);
}

// ---- Part 3: const reference (read-only, no copy) ----

long sum(const std::vector<int>& v) {
    long s = 0;
    for (int x : v) s += x;
    return s;
}

// ---- Part 4: RAII via unique_ptr (no manual delete needed) ----

void raiiDemo() {
    std::unique_ptr<int> p = std::make_unique<int>(42);
    std::cout << "unique_ptr value = " << *p << "\n";
    // no delete here: when p goes out of scope, memory is freed automatically.
}

int main() {
    int n = 10;

    addOneByValue(n);
    std::cout << "after byValue:     " << n << "  (expect 10, unchanged)\n";

    addOneByPointer(&n);
    std::cout << "after byPointer:   " << n << "  (expect 11)\n";

    addOneByReference(n);
    std::cout << "after byReference: " << n << "  (expect 12)\n";

    int* heapInt = makeIntOnHeap(100);
    if (heapInt) {
        std::cout << "heap int = " << *heapInt << "  (expect 100)\n";
        delete heapInt; // C++ has no garbage collector: heap memory from new must be freed manually, or it leaks
    }

    std::vector<int> data = {1, 2, 3, 4, 5};
    std::cout << "sum = " << sum(data) << "  (expect 15)\n";

    raiiDemo();

    return 0;
}
