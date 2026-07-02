# P1 - CUDA Fundamentals Hands-on

> 这一周先别急着背 API。目标只有一个: 真的理解 GPU 是怎么把一个 kernel 分发给成千上万个线程的。
> 你如果没有先吃透 thread / block / grid, 后面所有 CUDA 代码都会像天书。

适用对象:

- 已完成 P0 的内存、RAII、move semantics
- 刚开始学 CUDA
- 当前机器不一定装了 CUDA Toolkit

---

## 第 1 周 - CUDA 执行模型与第一个 kernel

### 本周目标

这一周你要建立 4 个最核心的直觉:

1. CPU 函数是"调用一次, 跑一次"; CUDA kernel 是"写一份逻辑, 同时给很多线程跑"
2. 一个线程只负责很小的一份工作, 常见模式是"每个线程处理一个元素"
3. 线程不是直接散着管理的, 而是按 `thread -> block -> grid` 分层组织
4. 全局索引 `idx = blockIdx.x * blockDim.x + threadIdx.x` 是 CUDA 入门第一公式

---

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w1_01_indexing_sim.cpp | 用 CPU 模拟 CUDA 的 block/thread 索引, 当前机器可直接运行 | `g++ -std=c++17 -O2 w1_01_indexing_sim.cpp -o sim && ./sim` |
| w1_02_hello_kernel.cu | 第一个最小 kernel, 观察很多线程同时打印 | `nvcc -std=c++17 w1_02_hello_kernel.cu -o hello && ./hello` |
| w1_03_thread_indexing.cu | 观察 `threadIdx.x / blockIdx.x / blockDim.x` 和全局索引 | `nvcc -std=c++17 w1_03_thread_indexing.cu -o indexing && ./indexing` |
| w1_04_vector_add.cu | 第一个真正有数值意义的 kernel: 向量加法 | `nvcc -std=c++17 w1_04_vector_add.cu -o vadd && ./vadd` |

如果你当前没有 `nvcc`, 先把 `w1_01_indexing_sim.cpp` 吃透, 再看 `.cu` 文件。等你切到 CUDA 环境时, 学习路径是无缝的。

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

这就是 `w1_04_vector_add.cu` 的核心。

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

先跑 `w1_01_indexing_sim.cpp`, 直到你能脱口而出:

- `threadIdx.x` 是什么
- `blockIdx.x` 是什么
- 为什么全局索引要写成 `blockIdx.x * blockDim.x + threadIdx.x`

### 第二步

读 `w1_02_hello_kernel.cu` 和 `w1_03_thread_indexing.cu`, 把 CUDA 语法样子认熟:

- `__global__`
- `<<<grid, block>>>`
- `cudaDeviceSynchronize()`

### 第三步

最后读 `w1_04_vector_add.cu`, 重点盯住:

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

---

## 第 2 周 - CUDA 内存层次、访问模式与 shared memory

### 本周目标

这一周要把一个关键事实吃透:

> CUDA 性能往往不是先被算力卡住, 而是先被内存访问方式卡住。

你会建立这 4 个核心直觉:

1. GPU 有多层内存: register / shared memory / global memory / constant memory
2. global memory 容量大, 但慢; shared memory 小得多, 但快而且能让同一个 block 协作
3. 相邻线程访问相邻地址, 通常比相邻线程跳着访问更好, 这就是 coalescing 的直觉
4. `__syncthreads()` 是 block 内线程协作的基本同步点

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w2_01_access_patterns.cu | 观察 coalesced vs strided 访问模式 | `nvcc -std=c++17 w2_01_access_patterns.cu -o access && ./access` |
| w2_02_shared_memory_reverse.cu | 用 shared memory 让同一个 block 里的线程协作 | `nvcc -std=c++17 w2_02_shared_memory_reverse.cu -o reverse && ./reverse` |
| w2_03_shared_memory_sum.cu | 用 shared memory 做 block 内归约求和 | `nvcc -std=c++17 w2_03_shared_memory_sum.cu -o block_sum && ./block_sum` |

---

## 概念 1: CUDA 内存层次(够用版)

先建立够用的层次感:

- register: 每个线程自己的最快小抽屉
- shared memory: 一个 block 内线程共享的小白板
- global memory: 所有线程都能访问的大仓库
- constant memory: 适合很多线程读同一份只读小数据

入门阶段最重要的是这句:

> 你几乎总是在和 global memory 打交道, 而很多优化本质上是在减少、重排、缓存对 global memory 的访问。

---

## 概念 2: 为什么 coalescing 重要

看这两种访问:

```text
thread 0 -> element 0
thread 1 -> element 1
thread 2 -> element 2
thread 3 -> element 3
```

这叫相邻线程访问相邻地址, 是理想模式。

再看这种:

```text
thread 0 -> element 0
thread 1 -> element 4
thread 2 -> element 8
thread 3 -> element 12
```

这叫 strided access, 相邻线程跳着读。通常会让内存访问更分散、效率更差。

本周第一个示例 `w2_01_access_patterns.cu` 就是把这个差别直接打印出来。

---

## 概念 3: shared memory 到底解决什么问题

shared memory 不是"另一块普通数组", 它的真正价值是:

- 比 global memory 快
- 同一个 block 里的线程都能访问
- 可以把会重复用的数据先搬进来, 再协作处理

基本套路是:

1. 每个线程从 global memory 读一点数据
2. 写进 shared memory
3. `__syncthreads()` 等大家都写完
4. 再从 shared memory 里协作读取

这就是 `w2_02_shared_memory_reverse.cu` 在演示的模式。

---

## 概念 4: `__syncthreads()` 是干什么的

这是 block 内的屏障:

- 没到这里的线程, 还在前面干活
- 到了这里的线程, 必须等同一个 block 里其他线程也到达
- 全部线程到齐后, 才能继续往后执行

最典型用途:

- 前半段: 大家往 shared memory 写数据
- `__syncthreads()`
- 后半段: 大家安全地读别人写进去的数据

如果没有这个同步点, 你可能会读到"别人还没来得及写完"的数据。

---

## 概念 5: 为什么 block 内归约能体现 shared memory 的价值

归约(reduction)是 GPU 基础训练题之一, 例如求和:

```text
1, 2, 3, 4, 5, 6, 7, 8 -> 36
```

如果每个线程只会读自己的元素, 它们根本没法一起完成总和。

shared memory + `__syncthreads()` 让一个 block 内的线程能像小组协作一样:

- 先各自拿一份数据
- 放到共享区
- 再一轮轮合并

这就是 `w2_03_shared_memory_sum.cu` 要演示的重点。

---

## 这一周建议怎么学

### 第一步

先跑 `w2_01_access_patterns.cu`, 看清:

- 相邻线程访问连续元素是什么样
- stride 访问长什么样
- 为什么说"访问模式"本身就是性能问题

### 第二步

再跑 `w2_02_shared_memory_reverse.cu`, 重点盯住:

- `__shared__` 怎么声明
- 为什么写完 shared memory 之后要 `__syncthreads()`
- 每个线程怎么读到别的线程先前写下来的数据

### 第三步

最后跑 `w2_03_shared_memory_sum.cu`, 理解:

- block 内线程怎样合作做一件单线程做不了的事
- 为什么 reduction 是共享内存训练题

---

## 学完这周, 你应该能做到

- 解释 coalesced access 和 strided access 的区别
- 看懂一个最小的 shared memory 示例
- 理解 `__syncthreads()` 为什么必要
- 读懂最基本的 block 内 shared-memory reduction

---

## 第 3 周 - Error Handling、Timing 与异步执行

### 本周目标

这一周开始把 CUDA 代码写得更像工程代码, 而不是只停留在"能跑"。

你会建立这 4 个核心直觉:

1. kernel launch 默认是异步的, CPU 不会傻等 GPU 自己干完
2. `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 是两道不同类型的检查
3. 只看 host 侧墙钟时间, 很容易把 GPU 时间量错
4. 用 CUDA events 才能更可靠地测 kernel 时间

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w3_01_async_launch.cu | 观察 kernel launch 的异步语义 | `nvcc -std=c++17 w3_01_async_launch.cu -o async_launch && ./async_launch` |
| w3_02_checked_vector_add.cu | 用统一的错误检查模式写一个更像工程代码的 kernel 示例 | `nvcc -std=c++17 w3_02_checked_vector_add.cu -o checked_vadd && ./checked_vadd` |
| w3_03_event_timing.cu | 用 CUDA events 正确测 kernel 时间 | `nvcc -std=c++17 w3_03_event_timing.cu -o event_timing && ./event_timing` |

---

## 概念 1: kernel launch 默认是异步的

这句:

```cpp
myKernel<<<grid, block>>>(...);
```

并不等于:

```text
CPU 在这里等 GPU 干完再继续
```

更接近于:

```text
CPU 把活派给 GPU, 然后自己先往下走
```

所以:

- launch 返回得很快, 不代表 GPU 已经算完
- 如果你接下来马上读结果, 往往需要同步

本周第一个示例 `w3_01_async_launch.cu` 就是专门把这个事实打印出来。

---

## 概念 2: 两道错误检查分别在查什么

最常见的工程模式是:

```cpp
myKernel<<<grid, block>>>(...);
cudaGetLastError();
cudaDeviceSynchronize();
```

它们不是重复:

- `cudaGetLastError()`: 查 launch 本身有没有立刻出错
- `cudaDeviceSynchronize()`: 等 GPU 真跑完, 顺便把执行阶段的错误带回来

一句话:

> 一个查"活有没有成功发出去", 一个查"活干的过程中有没有炸"。

---

## 概念 3: 为什么 host 墙钟时间容易量错

如果你这样写:

```cpp
auto t0 = now();
myKernel<<<grid, block>>>(...);
auto t1 = now();
```

你量到的很可能只是:

- CPU 发起 launch 花了多久

不是:

- GPU 真正执行 kernel 花了多久

因为 launch 默认异步。要么显式同步后再量, 要么直接用 CUDA events。

---

## 概念 4: 为什么 CUDA events 更适合测 kernel 时间

CUDA events 是 GPU 时间线上的打点工具:

- 在 kernel 前记录一个 event
- 在 kernel 后记录一个 event
- 等待后者完成
- 计算两个 event 的间隔

这样量到的更接近:

> GPU 真正执行这段工作花了多少时间

这就是 `w3_03_event_timing.cu` 的核心。

---

## 这一周建议怎么学

### 第一步

先跑 `w3_01_async_launch.cu`, 看清:

- 为什么 launch 之后 host 能立刻继续打印
- 为什么 `cudaDeviceSynchronize()` 之后才算 GPU 真的完成

### 第二步

再跑 `w3_02_checked_vector_add.cu`, 重点盯住:

- `CHECK_CUDA(...)` 这类宏怎么包错误检查
- 为什么 launch 后要先 `cudaGetLastError()`, 再 `cudaDeviceSynchronize()`

### 第三步

最后跑 `w3_03_event_timing.cu`, 理解:

- host 墙钟和 CUDA events 各自量到的是什么
- 为什么做性能实验时要先把时间测对

---

## 学完这周, 你应该能做到

- 解释 CUDA kernel launch 为什么默认是异步的
- 在自己的 CUDA 程序里加上基本错误检查
- 知道什么时候该用 `cudaDeviceSynchronize()`
- 用 CUDA events 写出一个最小可用的 kernel timing 示例

---

## 第 4 周 - Pinned Memory、Streams 与 Copy/Compute Overlap

### 本周目标

这一周开始触碰一个很真实的 CUDA 工程问题:

> 很多程序不是算得慢, 而是"搬数据 + 排队等待"太多。

你会建立这 4 个核心直觉:

1. pageable host memory 和 pinned host memory 不是一回事
2. `cudaMemcpyAsync` 真想异步得像样, 通常需要 pinned memory
3. stream 是一条有序工作队列; 同一 stream 内操作按顺序执行, 不同 stream 才有机会并行推进
4. 真正的 GPU 工程常常不是只优化 kernel, 还要优化 copy 和 compute 的重叠

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w4_01_pinned_copy_timing.cu | 对比 pageable vs pinned host memory 的拷贝时间 | `nvcc -std=c++17 w4_01_pinned_copy_timing.cu -o pinned_copy && ./pinned_copy` |
| w4_02_stream_pipeline.cu | 用单个 stream 串起 H2D copy、kernel、D2H copy | `nvcc -std=c++17 w4_02_stream_pipeline.cu -o stream_pipeline && ./stream_pipeline` |
| w4_03_two_stream_saxpy.cu | 用两个 stream 分块处理数据, 搭出 overlap-ready 结构 | `nvcc -std=c++17 w4_03_two_stream_saxpy.cu -o two_stream_saxpy && ./two_stream_saxpy` |

---

## 概念 1: 什么是 pinned memory

默认的 host 内存通常叫 pageable memory:

- 操作系统可以随时把页面换进换出
- 对 CPU 来说很普通
- 对 GPU DMA 拷贝来说不够理想

pinned memory 也叫 page-locked memory:

- 这段 host 内存被锁住, 不会随意分页
- GPU 更容易直接高效地做传输
- 很多真正异步的 host<->device copy 都依赖它

一句话:

> pinned memory 不是让 kernel 算得更快, 而是让数据传输路径更像高性能路径。

---

## 概念 2: 为什么 `cudaMemcpyAsync` 常和 pinned memory 一起出现

如果 host buffer 是 pageable:

- runtime 往往要先做额外处理
- 有时看起来像异步, 但并不是真正理想的异步传输路径

如果 host buffer 是 pinned:

- `cudaMemcpyAsync` 更容易走到真正适合 overlap 的路径

所以工程上经常成对出现:

```cpp
cudaMallocHost(...);
cudaMemcpyAsync(..., stream);
```

---

## 概念 3: stream 是什么

你可以把 stream 想成 GPU 的一条工作队列。

同一条 stream 里:

- H2D copy
- kernel launch
- D2H copy

会按你入队的顺序执行。

不同 stream 之间:

- 才有机会并发推进
- 具体能不能重叠, 取决于硬件能力、资源占用、copy engine、kernel 特征等

先记住最核心的一句:

> 同一 stream 保序, 不同 stream 才谈并发。

---

## 概念 4: overlap 到底想解决什么

没有 overlap 时, 时间线更像:

```text
copy in -> compute -> copy out -> copy in -> compute -> copy out
```

有机会 overlap 时, 目标更像:

```text
chunk0 compute 和 chunk1 copy in 部分重叠
chunk1 compute 和 chunk2 copy in / chunk0 copy out 部分重叠
```

也就是说:

- 不让 GPU 只在等数据
- 不让传输总在等计算

这就是 Week 4 最后一份例子 `w4_03_two_stream_saxpy.cu` 想带你看到的结构。

---

## 这一周建议怎么学

### 第一步

先跑 `w4_01_pinned_copy_timing.cu`, 建立:

- pinned memory 是什么
- 它和 copy timing 为什么相关

### 第二步

再跑 `w4_02_stream_pipeline.cu`, 看清:

- 同一 stream 内 copy / kernel / copy 为什么天然保序
- host 为什么只需要在最后同步一次

### 第三步

最后跑 `w4_03_two_stream_saxpy.cu`, 理解:

- 分块处理为什么是 overlap 的前提
- 两条 stream 怎么把结构搭出来

---

## 学完这周, 你应该能做到

- 解释 pageable host memory 和 pinned host memory 的区别
- 看懂最小的 `cudaMemcpyAsync + stream` 示例
- 理解为什么分块 + 多 stream 是 overlap 的基础结构
- 对"传输也是性能瓶颈"这件事建立真实直觉

---

## 第 5 周 - Profiling 入门: 看时间线, 找热点

### 本周目标

这一周开始学一件很关键的工程习惯:

> 先观察, 再优化。不要靠猜。

你会建立这 4 个核心直觉:

1. Nsight Systems (`nsys`) 更像全局时间线: 谁在等谁、copy 和 kernel 怎么排队
2. Nsight Compute (`ncu`) 更像单 kernel 深挖: 一个 kernel 自己跑得怎么样
3. "很多小 kernel" 和 "少量大一点的 kernel" 在时间线上看起来很不一样
4. profiling 的第一步不是看几百个指标, 而是先找出"时间花在哪"

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w5_01_many_small_kernels.cu | 为 `nsys` 准备的时间线样本: 很多小 kernel | `nvcc -std=c++17 w5_01_many_small_kernels.cu -o many_small && ./many_small` |
| w5_02_profile_ready_saxpy.cu | 为 `ncu` / `nsys` 准备的稳定 baseline kernel | `nvcc -std=c++17 w5_02_profile_ready_saxpy.cu -o profile_saxpy && ./profile_saxpy` |
| w5_03_grid_stride_saxpy.cu | 更像真实写法的 grid-stride loop kernel, 适合继续 profile | `nvcc -std=c++17 w5_03_grid_stride_saxpy.cu -o grid_stride_saxpy && ./grid_stride_saxpy` |

---

## 概念 1: `nsys` 和 `ncu` 各看什么

先用一句最短的话区分:

- `nsys`: 从"整台 GPU 程序"角度看时间线
- `ncu`: 从"某一个 kernel"角度看性能细节

如果你现在的问题是:

- 为什么程序在等
- copy 和 compute 有没有重叠
- kernel launch 是不是太碎

先看 `nsys`。

如果你现在的问题是:

- 某个 kernel 自己为什么慢
- 访存是不是有问题
- 算力有没有吃满

再看 `ncu`。

---

## 概念 2: 第一次看 timeline 先看哪三件事

第一次打开 `nsys` 时间线时, 不要急着看所有轨道。

先问自己三件事:

1. 时间主要花在 copy 还是 kernel
2. kernel 是不是碎成了很多很小的 launch
3. host 有没有在不必要地同步等待

如果这三件事都还没看明白, 就先别急着谈更细的优化。

---

## 概念 3: 为什么很多小 kernel 值得警惕

如果时间线看起来像这样:

```text
launch launch launch launch launch ...
tiny kernel tiny kernel tiny kernel ...
```

那很可能意味着:

- launch overhead 占比不低
- kernel 粒度太碎
- GPU 可能一直在处理很多小活, 但每个活都不够大

`w5_01_many_small_kernels.cu` 就是专门给你一个这种时间线样本。

---

## 概念 4: 为什么要准备 profile-ready baseline

做 profiling 时, 最怕的不是程序慢, 而是样本不稳定。

一个好样本通常会:

- 做足够多次迭代
- 有 warmup
- 输入规模别太小
- 最后还能做结果校验

`w5_02_profile_ready_saxpy.cu` 和 `w5_03_grid_stride_saxpy.cu` 就是在朝这个方向靠。

---

## 这一周建议怎么学

### 第一步

先跑 `w5_01_many_small_kernels.cu`, 然后用 `nsys` 看时间线, 盯住:

- 很多小 kernel 在时间线上是什么样
- host 端 launch 节奏是什么样

### 第二步

再跑 `w5_02_profile_ready_saxpy.cu`, 重点盯住:

- 一个简单稳定的 baseline kernel 在 `nsys` / `ncu` 里是什么感觉
- 总时间主要是不是落在 kernel 上

### 第三步

最后跑 `w5_03_grid_stride_saxpy.cu`, 理解:

- grid-stride loop 为什么是更真实的 CUDA kernel 写法
- profiling 的对象怎么逐渐从"玩具代码"走向"更像项目代码"

---

## 学完这周, 你应该能做到

- 知道 `nsys` 和 `ncu` 的职责分工
- 打开时间线后先找大头时间花在哪
- 识别"很多小 kernel"这类常见时间线特征
- 为后面的性能优化准备更像样的 profiling 样本

---

## 第 6 周 - 优化入口: Launch Config、Occupancy 与 Kernel Fusion

### 本周目标

这一周不追求"一下子优化十倍", 先建立三个很重要的性能直觉:

1. 同一个 kernel, block size 不同, 表现可能明显不同
2. occupancy 是一个有用的起点指标, 但不是唯一目标
3. 很多小 kernel 有时不如把工作融合到更少的 launch 里

### 本周练习文件

| 文件 | 主题 | 运行 |
|---|---|---|
| w6_01_block_size_sweep.cu | 扫不同 `threadsPerBlock`, 感受 launch config 对时间的影响 | `nvcc -std=c++17 w6_01_block_size_sweep.cu -o block_size_sweep && ./block_size_sweep` |
| w6_02_occupancy_hint.cu | 用 `cudaOccupancyMaxPotentialBlockSize` 拿到 block size 起步建议 | `nvcc -std=c++17 w6_02_occupancy_hint.cu -o occupancy_hint && ./occupancy_hint` |
| w6_03_fused_vs_many_kernels.cu | 对比很多小 launch 和一个融合 kernel 的结构差异 | `nvcc -std=c++17 w6_03_fused_vs_many_kernels.cu -o fused_vs_many && ./fused_vs_many` |

---

## 概念 1: Launch Config 为什么值得试

一维 kernel 里最常见的可调参数就是:

```cpp
<<<blocks, threadsPerBlock>>>
```

很多初学者会把 `threadsPerBlock` 当成随便填的数字, 但它会影响:

- 一个 block 里线程数
- 寄存器 / shared memory 资源切分
- 同时能驻留多少 block
- 最终的 occupancy 和吞吐

本周第一份示例 `w6_01_block_size_sweep.cu` 就是让你直接跑几个常见配置, 看时间差别。

---

## 概念 2: Occupancy 是什么

先记最够用的版本:

- occupancy 描述的是 SM 上活跃 warp 相对理论上限的比例
- 它反映"机器有没有被足够多的活填起来"

但要特别注意:

> occupancy 高不等于性能一定最好, 它只是一个起点信号, 不是终点答案。

如果一个 kernel 明显受内存访问限制, 光把 occupancy 提高不一定解决问题。

---

## 概念 3: `cudaOccupancyMaxPotentialBlockSize` 有什么用

这是一个很实用的 API:

- 你给它一个 kernel
- 它帮你给出一个"潜在可行的 block size 起点"

这不是神谕, 也不是最终答案, 但非常适合:

- 避免完全瞎猜 block size
- 给 block size sweep 一个合理起点

这就是 `w6_02_occupancy_hint.cu` 的重点。

---

## 概念 4: Kernel Fusion 在解决什么问题

如果程序长这样:

```text
launch launch launch launch ...
tiny work tiny work tiny work ...
```

那很可能:

- launch overhead 不低
- 时间线很碎
- GPU 一直在接很多小任务

一种常见优化思路就是 fusion:

- 把原本分散在多个 kernel 里的工作
- 尽量合到更少的 launch 里

`w6_03_fused_vs_many_kernels.cu` 就是用一个很小的例子把这个结构对比跑出来。

---

## 这一周建议怎么学

### 第一步

先跑 `w6_01_block_size_sweep.cu`, 感受:

- 同一个 kernel 在 64 / 128 / 256 / 512 threads per block 下会不会有差别
- 为什么 launch config 不是随便填的

### 第二步

再跑 `w6_02_occupancy_hint.cu`, 重点盯住:

- occupancy API 给了什么建议
- 建议值和你手工 sweep 的结果是不是一致

### 第三步

最后跑 `w6_03_fused_vs_many_kernels.cu`, 理解:

- 很多小 launch 的时间结构
- 融合 kernel 为什么经常更值得考虑

---

## 学完这周, 你应该能做到

- 知道为什么 block size 值得 sweep
- 会用 occupancy API 拿一个 block size 起步建议
- 理解 kernel fusion 在解决什么问题
- 为后面真正进入 kernel 优化打好第一层直觉
