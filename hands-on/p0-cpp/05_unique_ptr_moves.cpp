// P0 Week 2 - unique_ptr ownership transfer.
// Build & run:
//   g++ -std=c++17 -O2 05_unique_ptr_moves.cpp -o uptr_move && ./uptr_move
//
// Focus:
// 1) unique_ptr cannot be copied
// 2) ownership can be moved
// 3) moved-from unique_ptr becomes empty

#include <iostream>
#include <memory>
#include <utility>

struct Widget {
    explicit Widget(int value) : value(value) {
        std::cout << "[Widget ctor] value=" << value << "\n";
    }

    ~Widget() {
        std::cout << "[Widget dtor] value=" << value << "\n";
    }

    int value;
};

std::unique_ptr<Widget> upgrade(std::unique_ptr<Widget> w) {
    std::cout << "upgrade() owns Widget at " << w.get() << "\n";
    w->value += 100;
    return w; // ownership moves back to the caller
}

int main() {
    auto p = std::make_unique<Widget>(42);
    std::cout << "main owns Widget at " << p.get() << "\n";

    // auto q = p; // uncommenting this line would fail to compile: copy is deleted

    auto q = std::move(p);
    std::cout << "after move, p.get() = " << p.get() << " (expect nullptr)\n";
    std::cout << "q now owns Widget at " << q.get() << "\n";

    q = upgrade(std::move(q));
    std::cout << "after upgrade, q->value = " << q->value << "\n";

    return 0;
}
