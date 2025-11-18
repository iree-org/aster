# Performance Exploration

This set of examples explores the main performance characteristics of AMDGPU
hardware that are useful to low-level software developers, from first principles,
using special purpose ASM.

## Setup

Make sure you have:
1. Built the project: `cd build && ninja install`
2. Set PYTHONPATH: `export PYTHONPATH=${PWD}/build/python_packages`
3. LLVM tools in PATH (for assembling to HSACO): `llvm-mc` and `ld.lld`
