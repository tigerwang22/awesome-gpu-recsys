// P0 Week 2 - observe copy vs move.
// Build & run:
//   g++ -std=c++17 -O2 04_move_basics.cpp -o move_basic && ./move_basic
//
// Focus:
// 1) lvalue usually copies
// 2) std::move makes an expression movable
// 3) move transfers resources instead of duplicating them

#include <iostream>
#include <string>
#include <utility>

class Tracer {
public:
    explicit Tracer(std::string name) : name_(std::move(name)) {
        std::cout << "[ctor]        " << name_ << "\n";
    }

    Tracer(const Tracer& other) : name_(other.name_ + " (copied)") {
        std::cout << "[copy ctor]   from " << other.name_ << " -> " << name_ << "\n";
    }

    Tracer(Tracer&& other) noexcept : name_(std::move(other.name_)) {
        std::cout << "[move ctor]   take ownership of payload\n";
        other.name_ = "<moved-from>";
    }

    ~Tracer() {
        std::cout << "[dtor]        " << name_ << "\n";
    }

    const std::string& name() const {
        return name_;
    }

private:
    std::string name_;
};

void takeByValue(Tracer t) {
    std::cout << "inside takeByValue: " << t.name() << "\n";
}

int main() {
    std::cout << "1) construct original\n";
    Tracer a("alpha");

    std::cout << "\n2) pass lvalue -> copy\n";
    takeByValue(a);

    std::cout << "\n3) pass std::move(a) -> move\n";
    takeByValue(std::move(a));

    std::cout << "after move, a = " << a.name() << "\n";

    std::cout << "\n4) construct b from temporary\n";
    Tracer b("beta");
    std::cout << "b = " << b.name() << "\n";

    return 0;
}
