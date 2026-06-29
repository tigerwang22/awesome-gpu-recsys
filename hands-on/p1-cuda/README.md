# P1 Week 1 - CUDA 执行模型与第一个 kernel

> 这一周先别急着背 API。目标只有一个: 真的理解 GPU 是怎么把一个 kernel 分发给成千上万个线程的。
> 你如果没有先吃透 thread / block / grid, 后面所有 CUDA 代码都会像天书。

适用对象:

- 已完成 P0 的内存、RAII、move semantics
- 刚开始学 CUDA
- 当前机器不一定装了 CUDA Toolkit

---

## 本周目标

这一周你要建立 4 个最核心的直觉:

1. CPU 函数是"调用一次, 跑一次"; CUDA kernel 是"写一份逻辑, 同时给很多线程跑"
2. 一个线程只负责很小的一份工作, 常见模式是"每个线程处理一个元素"
3. 线程不是直接散着管理的, 而是按 `thread -> block -> grid` 分层组织
4. 全局索引 `idx = blockIdx.x * blockDim.x + threadIdx.x` 是 CUDA 入门第一公式

---

## 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| 00_indexing_sim.cpp | 用 CPU 模拟 CUDA 的 block/thread 索引, 当前机器可直接运行 | `g++ -std=c++17 -O2 00_indexing_sim.cpp -o sim && ./sim` |
| 01_hello_kernel.cu | 第一个最小 kernel, 观察很多线程同时打印 | `nvcc -std=c++17 01_hello_kernel.cu -o hello && ./hello` |
| 02_thread_indexing.cu | 观察 `threadIdx.x / blockIdx.x / blockDim.x` 和全局索引 | `nvcc -std=c++17 02_thread_indexing.cu -o indexing && ./indexing` |
| 03_vector_add.cu | 第一个真正有数值意义的 kernel: 向量加法 | `nvcc -std=c++17 03_vector_add.cu -o vadd && ./vadd` |

如果你当前没有 `nvcc`, 先把 `00_indexing_sim.cpp` 吃透, 再看 `.cu` 文件。等你切到 CUDA 环境时, 学习路径是无缝的。

---

## 概念 1: kernel 到底是什么

普通 C++ 函数:

```cpp
add(a, b, c);
```

意思是:

- CPU 按通常方式调用一次函数
- 里面的逻辑由少量 CPU 核心执行

CUDA kernel:

```cpp
vectorAdd<<<gridSize, blockSize>>>(a, b, c, n);
```

意思是:

- 启动很多 GPU 线程
- 每个线程都执行同一份 `vectorAdd` 代码
- 但每个线程看到的 `threadIdx` / `blockIdx` 不一样, 所以各自处理不同的数据位置

一句话:

> kernel 不是"调用一次做一件事", 而是"启动很多线程, 每个线程做一小件事"。

---

## 概念 2: thread / block / grid

先记一维情况:

- `threadIdx.x`: 当前线程在自己 block 里的编号
- `blockIdx.x`: 当前 block 在整个 grid 里的编号
- `blockDim.x`: 每个 block 里有多少线程
- `gridDim.x`: 整个 grid 里有多少个 block

一个经典例子:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

如果:

- `blockDim.x = 8`
- `gridDim.x = 3`

那总线程数就是 `3 * 8 = 24`。

你可以把它想成:

```text
block 0: thread 0..7    -> global idx 0..7
block 1: thread 0..7    -> global idx 8..15
block 2: thread 0..7    -> global idx 16..23
```

---

## 概念 3: 为什么要有 block

不是简单为了分组好看, 而是因为:

- block 是调度和协作的基本单位
- 同一个 block 里的线程可以用 shared memory
- 同一个 block 里的线程可以 `__syncthreads()`
- 不同 block 之间默认不能直接同步

入门阶段你先记住:

> thread 是干活的人, block 是一个小组, grid 是这次启动的所有小组。

---

## 概念 4: 第一个实际模式 - 每线程处理一个元素

这是 CUDA 最常见的入门模式:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

逻辑很朴素:

- 算出当前线程负责哪个元素
- 如果没越界, 就处理这个元素

这就是 `03_vector_add.cu` 的核心。

---

## 概念 5: host 和 device 的分工

写 CUDA 时, 你会同时写两种代码:

- host 代码: 运行在 CPU 上, 负责分配显存、拷贝数据、启动 kernel
- device 代码: 运行在 GPU 上, 也就是 `__global__` / `__device__` 这些函数

典型流程:

1. CPU 上准备输入数据
2. `cudaMalloc` 分配 GPU 显存
3. `cudaMemcpy` 把数据拷到 GPU
4. 启动 kernel
5. `cudaMemcpy` 把结果拷回 CPU
6. 校验结果
7. `cudaFree` 释放显存

你会发现, 这和 P0 的内容直接连上了:

- 显存是手动管理资源
- `cudaMalloc/cudaFree` 就像 GPU 世界的 `new/delete`
- 后面我们完全可以把它封装成 RAII 风格的 `CudaBuffer`

---

## 这一周建议怎么学

### 第一步

先跑 `00_indexing_sim.cpp`, 直到你能脱口而出:

- `threadIdx.x` 是什么
- `blockIdx.x` 是什么
- 为什么全局索引要写成 `blockIdx.x * blockDim.x + threadIdx.x`

### 第二步

读 `01_hello_kernel.cu` 和 `02_thread_indexing.cu`, 把 CUDA 语法样子认熟:

- `__global__`
- `<<<grid, block>>>`
- `cudaDeviceSynchronize()`

### 第三步

最后读 `03_vector_add.cu`, 重点盯住:

- kernel 里那 3 行索引逻辑
- host 侧的 `cudaMalloc/cudaMemcpy/cudaFree`
- 为什么一定要做结果校验

---

## 学完这周, 你应该能做到

- 看懂最基本的一维 CUDA kernel
- 自己算出一个线程处理哪个数据位置
- 理解 CUDA 程序最基本的 host/device 分工
- 看懂并解释一个 vector add 示例

---

## 和后面几周的连接

这一周解决的是:

```text
GPU 上到底是谁在执行代码?
```

下一周会开始解决:

```text
GPU 内存层次是什么? 为什么访问方式会决定性能?
```
