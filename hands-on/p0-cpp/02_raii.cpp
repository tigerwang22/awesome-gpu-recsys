// P0 Week 1 - RAII demo: watch the destructor run AUTOMATICALLY.
// Build & run:
//   g++ -std=c++17 -O2 02_raii.cpp -o raii && ./raii
//
// The whole point: you never call "release" yourself. It happens automatically
// when the object goes out of scope. Watch the order of the printed lines.

#include <iostream>
#include <string>
#include <memory>

// A tiny class that "owns" a resource.
// Constructor = acquire the resource.  Destructor = release it.
class Guard {
public:
    Guard(const std::string& name) : name_(name) {
        std::cout << "  [acquire] " << name_ << "\n";   // runs when object is created
    }
    ~Guard() {
        std::cout << "  [release] " << name_ << "\n";   // runs AUTOMATICALLY when object dies
    }
private:
    std::string name_;
};

void scopeDemo() {
    std::cout << "enter scopeDemo\n";
    Guard g("resource-A");                 // construct -> [acquire] prints here
    std::cout << "...doing work...\n";
    // NOTE: no manual release. When scopeDemo() ends, g is destroyed
    // and ~Guard() runs by itself.
}                                          // <-- [release] prints right here

void earlyReturnDemo(bool fail) {
    std::cout << "enter earlyReturnDemo(fail=" << (fail ? "true" : "false") << ")\n";
    Guard g("resource-B");
    if (fail) {
        std::cout << "...something went wrong, returning early...\n";
        return;                            // even on early return, ~Guard() STILL runs
    }
    std::cout << "...finished normally...\n";
}

int main() {
    std::cout << "=== 1) basic scope ===\n";
    scopeDemo();
    std::cout << "back in main\n\n";

    std::cout << "=== 2) early return still releases ===\n";
    earlyReturnDemo(true);
    std::cout << "back in main\n\n";

    std::cout << "=== 3) unique_ptr is just a ready-made RAII wrapper for heap memory ===\n";
    {
        std::unique_ptr<int> p = std::make_unique<int>(42);
        std::cout << "  unique_ptr holds " << *p << " (no new/delete written by us)\n";
    } // <-- p goes out of scope here, its destructor calls delete for us
    std::cout << "  unique_ptr already freed the memory automatically\n";

    return 0;
}
