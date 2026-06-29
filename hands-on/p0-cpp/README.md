# P0 Week 1 - C++ 内存、指针与所有权

> 这一周补齐带 GC 的高级语言替你隐藏、但 C++ 和 CUDA 必须直面的东西: 内存、指针、所有权、RAII。
> CUDA 编程本质就是手动管理 GPU 显存指针, 所以这是地基中的地基。

适用对象: 有编程基础(用过某种带 GC 的语言, 如 Java/Python/Go/C#)。

---

## 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| 01_pointers.cpp | 值/指针/引用传参、堆内存、const 引用、unique_ptr | `g++ -std=c++17 -O2 01_pointers.cpp -o out && ./out` |
| 02_raii.cpp | RAII: 析构函数自动释放资源(亲眼看到) | `g++ -std=c++17 -O2 02_raii.cpp -o raii && ./raii` |
| 03_ownership.cpp | 所有权陷阱: 拷贝导致 double free(故意写的 bug) | `g++ -std=c++17 -O2 03_ownership.cpp -o own && ./own` |

---

## 工具链

Mac 上 `g++` 实际指向 Apple clang, 完全标准可用:

```
g++ --version
g++ -std=c++17 -O2 文件.cpp -o out && ./out
```

---

## 概念 1: C++ 没有 GC

- 带 GC 的语言: new 一个对象, 运行时自动回收。
- C++: 谁分配谁释放。堆内存 new 了不 delete 就泄漏。
- 这就是为什么 C++ 有指针、RAII、智能指针: 都在管"谁拥有这块内存、何时释放"。
- CUDA 更极端: cudaMalloc 分配显存必须 cudaFree, 没有任何自动回收。

---

## 概念 2: 栈 vs 堆

```cpp
void f() {
    int a = 10;            // 栈: 函数返回自动回收
    int* p = new int(20);  // 堆: 必须手动 delete p, 否则泄漏
    delete p;
}
```

- 栈: 快、自动管理、生命周期绑定作用域。
- 堆: 手动管理、生命周期自由、要自己负责释放。

---

## 概念 3: 值 / 指针 / 引用 传参

把内存想象成一排带地址的格子, 变量就是某个格子里的值。

```cpp
void byValue(int x)   { x = 99; }   // 改的是副本, 外面不变
void byPointer(int* x){ *x = 99; }  // x 存的是地址, * 解引用后改原值
void byRef(int& x)    { x = 99; }   // x 是原变量的别名, 直接改原值
```

- 值传递: 拷贝一份, 外部不变(大对象拷贝开销大)。
- 指针: 传地址, 可改原值, 可为 null, 需用 * 解引用; 调用时用 &取地址。
- 引用: 地址的别名, 可改原值, 不能为 null, 语法像普通变量。
- 经验: 只读大对象用 `const T&`(高效且安全), 要改用引用或指针。

CUDA 关联: host 与 device 之间、kernel 之间, 全靠指针传 GPU 显存地址。

---

## 概念 4: RAII(本周核心)

### 它解决什么问题

手动管理资源很容易漏:

```cpp
void f() {
    int* p = new int(10);
    if (出错) return;   // 糟糕: 提前 return, 下面的 delete 没跑 -> 泄漏
    delete p;
}
```

只要有提前 return、抛异常、或忘写 delete, 就泄漏。

### 核心思想(一句话)

> 把资源的生命周期绑定到一个对象的生命周期上。对象构造时拿资源(构造函数), 对象销毁时自动还(析构函数)。

C++ 铁律: 对象离开作用域时, 析构函数被自动调用。RAII 就是利用这条 -- 你不用记得释放, 释放会自动发生。

### 关键现象(见 02_raii.cpp 输出)

- 函数里没写任何释放代码, 但作用域结束时 `[release]` 自动打印 -> 析构函数自动跑了。
- 即使提前 return, 析构函数照样跑 -> 不管从哪条路径离开, 释放一定发生。
- unique_ptr 就是现成的 RAII 包装: 不写 new/delete, 它的析构函数替你 delete。

### RAII 和"栈内存自动回收"的区别(重要)

这两个是"利用"关系, 不是同一回事:

| | 管的是什么 | 释放逻辑 | 是什么 |
|---|---|---|---|
| 栈变量(如 int a) | 栈上那几个字节自己 | 无(直接回收) | 语言机制 |
| RAII 对象 | 别处的资源(堆/文件/GPU) | 你写在析构函数里 | 设计模式(借栈机制实现) |

```
普通栈变量 int a:           RAII 对象 g(持有堆资源):
  栈 [ 10 ]                   栈 [ g 含指针 ] ---> 堆 [ 实际资源 ]
  离开作用域直接回收           离开作用域 -> 自动调 ~g() -> 释放堆那块
```

一句话: RAII 对象本身在栈上, 但它是"管家", 借栈"对象销毁自动调析构"这个扳机, 去释放住在堆/文件/GPU 上的资源。

---

## 概念 5: 所有权与 double free(深挖, 见 03_ownership.cpp)

当一个管着堆资源的 RAII 对象被拷贝时会出事:

```cpp
Owner a(10);
Owner b = a;   // 默认是浅拷贝: 只复制了指针这个地址, 没复制堆里的内容
```

```
  a.data_ --+
            +---> 堆 [ 10 ]   // 只有一块, 两个指针指向它
  b.data_ --+
```

结束时两个析构函数各 delete 一次 -> 同一块内存释放两次 -> double free -> 程序 abort(退出码 133/134)。

### C++ 的解决方案: Rule of Three / Five

> 如果一个类需要自定义析构函数(说明它管着资源), 通常也需要自定义拷贝构造、拷贝赋值(C++11 后还有移动构造、移动赋值), 否则默认浅拷贝会出事。

三条路:

1. 禁止拷贝(最常用): 资源只能一个主人
   ```cpp
   Owner(const Owner&) = delete;
   Owner& operator=(const Owner&) = delete;
   ```
2. 深拷贝: 复制时连堆内容一起复制, 各管各的
   ```cpp
   Owner(const Owner& o) : data_(new int(*o.data_)) {}
   ```
3. 移动语义: 不复制资源, 而是转移所有权(下一周内容, 也是 GPU 性能关键)。

unique_ptr 就是"禁止拷贝 + 只能移动"的现成品。所以现代 C++ 的答案: 别裸写 new/delete + 裸指针, 用 unique_ptr / shared_ptr。

CUDA 关联: GPU 显存 double free 同样致命。自己写的 CudaBuffer 类要么禁止拷贝、要么实现移动, 这正是下周学移动语义对 GPU 编程是刚需的原因。

---

## 一条完整逻辑链(本周收获)

```
RAII -> 所有权 -> 拷贝的危险(double free) -> Rule of Five -> 移动语义(下周)
```

---

## 一个小知识(调试用)

cout 默认是缓冲的, 程序崩溃前缓冲区可能没刷出, 导致看不到打印。
在 main 开头加 `std::cout << std::unitbuf;` 可让 cout 每次都立即刷新, 便于定位崩溃前的最后输出。

---

## CUDA 关联速查

| 本周概念 | CUDA 里对应什么 |
|---|---|
| 堆内存手动管理 | cudaMalloc / cudaFree, 无 GC |
| 指针传参 | host 与 device、kernel 之间传显存地址 |
| RAII | CudaBuffer 类: 构造 cudaMalloc, 析构 cudaFree, 显存不泄漏 |
| 禁止拷贝/移动 | 防止显存被 double free |
