# CUDA + OpenGL N-Body Simulation

Real-time N-body gravity simulation using **CUDA C** for compute and **OpenGL** for rendering with **C++** as the host code. This is on linux using codium with a python build process.

## Overview
- CUDA kernels perform all simulation work on the GPU.
- Uses **CUDA–OpenGL interop** to write particle positions directly into an OpenGL **VBO**.
- Rendering reads from the same VBO, bypassing the CPU entirely.

## Compute
- CUDA C kernels for particle integration and force calculation.
- Barnes–Hut distant-region approximation for scalable N-body simulation.
- Octree construction currently runs on the GPU.

## Rendering
- OpenGL renders directly from CUDA-written VBOs.
- No CPU-side staging or readback.

## Camera
- 2x 3D free-fly camera versions.
- Minecraft creative-style movement thats more like an fps game with typical WASD and space controls.
- Flight mode for unrestricted navigation. This allows for rotation in all 6 axis.

## In Progress
- Replacing octree build with a **space-filling curve** pipeline.
- GPU **radix sort** with a custom **prefix scan**.
- Improved coherence and faster hierarchy construction.

