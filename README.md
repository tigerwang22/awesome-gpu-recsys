# Awesome GPU Recommender Systems [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated 6-month roadmap and resource list for becoming a **GPU-accelerated recommender systems engineer** — C++/CUDA, GPU performance optimization, and HPC data pipelines.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is both a **learning path** (follow it stage by stage) and an **awesome-list** (curated, opinionated resources). It is built for engineers with a programming background who want to break into GPU/HPC and large-scale recommender systems (the kind of work done on frameworks like NVIDIA Merlin / HugeCTR).

If this helps you, please star the repo and contribute a resource via PR.

---

## Contents

- [Who this is for](#who-this-is-for)
- [The 6-Month Roadmap](#the-6-month-roadmap)
- [P0 · Modern C++](#p0--modern-c)
- [P1 · CUDA Fundamentals](#p1--cuda-fundamentals)
- [P2 · GPU Performance Optimization](#p2--gpu-performance-optimization)
- [P3 · Recommender Systems on GPU](#p3--recommender-systems-on-gpu)
- [P4 · Multi-GPU & Communication](#p4--multi-gpu--communication)
- [P5 · HPC Data Pipelines](#p5--hpc-data-pipelines)
- [P6 · Portfolio & Open Source](#p6--portfolio--open-source)
- [Hands-On Exercises](#hands-on-exercises)
- [Cloud GPU Options](#cloud-gpu-options)
- [Core Principles](#core-principles)
- [Contributing](#contributing)
- [License](#license)

---

## Who this is for

- You can already program in some language (Java/Python/Go/C#/...).
- You want to move into **GPU programming, performance engineering, or large-scale ML systems**.
- C++ rusty or zero is fine — **P0 covers it**.
- Assumes ~20 hours/week; reach an interview-ready level in about 6 months.
- No GPU to buy: cloud GPUs are enough (see [Cloud GPU Options](#cloud-gpu-options)).

---

## The 6-Month Roadmap

| Stage | Focus | Duration | Key Deliverable |
|---|---|---|---|
| P0 Modern C++ | pointers / memory / RAII / templates / CMake | 2-3 weeks | build a multi-file C++ program confidently |
| P1 CUDA Fundamentals | thread model / memory hierarchy / writing kernels | 4-6 weeks | vector add, matmul, reduction kernels |
| P2 GPU Performance | Nsight profiling / coalescing / shared memory / Roofline | 4-6 weeks | optimize a kernel 5-10x with a report |
| P3 RecSys on GPU | sparse embeddings / memory-bound / HugeCTR | 3-4 weeks | run a Merlin example and read its source |
| P4 Multi-GPU | NCCL / data & model parallelism / embedding sharding | 4 weeks | a multi-GPU all-reduce demo |
| P5 HPC Pipelines | RDMA / GPUDirect RDMA & Storage / NVMe | 4 weeks | concepts + an end-to-end experiment if HW available |
| P6 Portfolio | optimize a real kernel / contribute to HugeCTR | ongoing | public GitHub work + an open-source contribution |

---

## P0 · Modern C++

**Goal:** Face what GC languages hid from you — memory, pointers, ownership.

- [learncpp.com](https://www.learncpp.com/) — the best free, thorough modern C++ tutorial
- *A Tour of C++* (Bjarne Stroustrup) — concise modern C++ overview
- [cppreference.com](https://en.cppreference.com/) — the reference you will live in
- Topics to master: pointers vs references vs value, stack vs heap, `new`/`delete`, `const` correctness, **RAII**, smart pointers, templates, move semantics, CMake basics

**Deliverable:** build and run a small multi-file C++ project with CMake.

---

## P1 · CUDA Fundamentals

**Goal:** Write correct CUDA kernels and understand the execution + memory model.

- *Programming Massively Parallel Processors* (Kirk & Hwu) — **the** canonical CUDA textbook
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (NVIDIA, Mark Harris) — best first read
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — official spec
- *CUDA by Example* (Sanders & Kandrot) — gentle hands-on intro
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) — community CUDA/perf lecture series
- Topics: thread/block/grid/warp, global/shared/register/constant memory, `cudaMalloc`/`cudaMemcpy`/`cudaFree`, streams, error handling

**Deliverable:** vector add, tiled matmul, and a parallel reduction — all numerically verified.

---

## P2 · GPU Performance Optimization

**Goal:** Profile first, then optimize. Know if you are compute-bound or memory-bound.

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/) and [Nsight Compute](https://docs.nvidia.com/nsight-compute/) — official profilers
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — official optimization handbook
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) — the mental model for performance ceilings
- Topics: memory coalescing, shared-memory tiling, bank conflicts, occupancy, warp divergence, instruction-level parallelism

**Deliverable:** take a naive kernel, profile it, optimize 5-10x, and write up the before/after.

---

## P3 · Recommender Systems on GPU

**Goal:** Understand why recsys is **memory-bound** and how frameworks handle huge sparse embeddings.

- [NVIDIA Merlin](https://github.com/NVIDIA-Merlin/Merlin) — end-to-end GPU recsys framework
- [HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR) — high-performance GPU CTR training framework (read the embedding code)
- [DLRM](https://github.com/facebookresearch/dlrm) (Meta) — the reference deep learning recommendation model
- Topics: large sparse embedding tables, embedding lookup/update bottlenecks, why recsys differs from generic HPC

**Deliverable:** run a Merlin example and read through the embedding implementation.

---

## P4 · Multi-GPU & Communication

**Goal:** Scale across GPUs and understand collective communication.

- [NCCL](https://github.com/NVIDIA/nccl) and [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/) — collective communication library
- Topics: all-reduce / all-gather, data vs model parallelism, embedding table sharding, synchronization bottlenecks

**Deliverable:** a working multi-GPU all-reduce demo.

---

## P5 · HPC Data Pipelines

**Goal:** Move data fast between GPUs, NICs, and storage.

- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/) — GPU-to-NIC direct path
- [GPUDirect Storage (GDS)](https://docs.nvidia.com/gpudirect-storage/) — GPU-to-NVMe direct path
- [UCX](https://github.com/openucx/ucx) — unified communication framework
- Topics: RDMA verbs, CUDA-aware MPI, NVMe / NVMe-oF, high-speed compute-to-storage data movement

**Deliverable:** clear conceptual mastery; an end-to-end experiment if hardware is available.

---

## P6 · Portfolio & Open Source

**Goal:** Make your work visible and contribute back.

- Turn your P2/P4 optimizations into a clean GitHub project with a write-up
- Open a PR to Merlin / HugeCTR / DLRM (docs or a small optimization count too)
- Write a blog post per stage — teaching is the best proof of understanding

---

## Hands-On Exercises

This repo is not just a reading list — it ships **runnable, annotated exercises** so you learn by doing.

| Stage | Folder | What's inside |
|---|---|---|
| P0 Modern C++ | [hands-on/p0-cpp/](hands-on/p0-cpp/) | pointers vs references vs value, heap memory, **RAII**, ownership, move semantics, Rule of Five — with notes and 6 compilable programs |
| P1 CUDA Fundamentals | [hands-on/p1-cuda/](hands-on/p1-cuda/) | CUDA execution model, indexing, first kernel, vector add, memory access patterns, shared memory — with notes, 1 CPU simulator, and 6 CUDA examples |

Each folder has a `README.md` explaining the concepts (and how they map to CUDA), plus small `.cpp` programs you build and run with `g++`. More stages land here as the roadmap progresses.

---

## Cloud GPU Options

| Stage | Recommended | Cost |
|---|---|---|
| P0-P2 | Google Colab free T4 (run `nvcc` in notebooks) | free / Pro ~$10/mo |
| P2 deep profiling | RunPod / Lambda / Vast.ai by the hour (for Nsight) | ~$0.2-0.5/hr |
| P4-P5 | RunPod multi-GPU instances, on demand | pay as you go |

---

## Core Principles

1. **Never just read CUDA — write code, run a profiler, look at the numbers.**
2. **Recsys is memory-bound, not compute-bound.** This is the key difference from generic HPC.
3. **Every stage must end with something you can show.** Build a portfolio as you go.
4. **From P3 on, read HugeCTR / Merlin source** — that is real production code.

---

## Contributing

PRs welcome! Add a resource that genuinely helped you learn:

1. Fork and create a branch.
2. Add the link under the right stage, with a one-line description of *why* it is good.
3. Keep it high-signal — this list is curated, not exhaustive.
4. Open a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE) — free to use, share, and adapt.
