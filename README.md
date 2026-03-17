# CUDA + OpenGL N-Body Simulation

Real-time N-body gravity simulation using **CUDA C** for compute and **OpenGL** for rendering, with **C++** as the host layer. Developed on Linux using Codium with a custom Python build system.

## Overview

- CUDA kernels perform all simulation work directly on the GPU.
- Uses **CUDA–OpenGL interop** to write particle positions into a shared OpenGL **VBO**.
- Rendering reads from the same buffer, eliminating CPU involvement in the simulation → render pipeline.

## Why

- Built to explore **low-level GPU programming** without relying on high-level abstraction libraries.
- Focused on understanding:
  - Memory layout and data movement
  - Parallel execution constraints
  - CPU vs GPU responsibility boundaries in real-time systems

## Compute

- CUDA C kernels handle **force accumulation and integration** per particle.
- Started with a naive **O(n²)** all-pairs simulation as a correctness and benchmarking baseline.
- Introduced **Barnes–Hut approximation** to reduce complexity toward **O(n log n)**.
- Octree construction and traversal are now **executed on the GPU**.

## Rendering

- OpenGL renders directly from CUDA-written buffers using **interop VBOs**.
- No CPU-side staging, copying, or synchronization bottlenecks.
- Designed to keep the entire simulation → render pipeline GPU-resident.

## Camera

- Two interchangeable **3D free-fly camera systems**:
  - FPS-style movement using WASD + mouse
  - Full flight mode allowing unrestricted 6-axis movement
- Designed for navigating large-scale simulations from both grounded and orbital perspectives.

## Build System

- Custom **Python build pipeline** for compiling both CUDA (`.cu`) and standard C++ (`.cpp`) sources.
- Compiles independent files in parallel using multithreading.
- Performs a single link step only after all compilation stages succeed.
- Chosen to maintain full control over the build process and avoid treating CUDA integration as a black box.

## Simulation Progression

- Began with naive **O(n²)** simulation to establish correctness and baseline performance.
- Added **GPU instanced rendering** for efficient large-scale particle visualisation.
- Introduced **per-stage benchmarking** to identify performance bottlenecks.
- Moved to a **Barnes–Hut octree** for scalable force approximation.
- Transitioned octree construction and traversal onto the GPU to reduce CPU overhead.

## Current Direction

- Replacing octree construction with a **space-filling curve (Morton code) pipeline**.
- Implementing GPU **radix sort** with custom **prefix-sum (scan)**.
- Improving memory coherence and enabling faster parallel hierarchy construction.
- Moving toward a **fully GPU-resident spatial acceleration structure**.
