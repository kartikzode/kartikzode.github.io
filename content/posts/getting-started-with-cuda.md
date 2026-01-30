---
title: "Getting Started with CUDA Programming"
date: 2024-01-15T10:00:00-00:00
draft: false
tags: ["CUDA", "GPU", "Parallel Computing"]
categories: ["CUDA Programming"]
author: "Thombya"
showToc: true
TocOpen: false
description: "An introduction to CUDA programming and GPU computing fundamentals"
---

# Getting Started with CUDA Programming

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables developers to harness the power of GPUs for general-purpose computing tasks.

## Why CUDA?

Modern GPUs contain thousands of cores capable of executing operations in parallel. While CPUs excel at sequential processing and complex logic, GPUs are designed for massive parallelism. CUDA allows us to leverage this parallelism for computational tasks beyond graphics rendering.

## Basic Concepts

### Thread Hierarchy

CUDA organizes threads in a hierarchical structure:

- **Threads**: The smallest unit of execution
- **Blocks**: Groups of threads that can cooperate through shared memory
- **Grid**: Collection of blocks launched in a single kernel invocation

```cpp
// Example: Vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Memory Hierarchy

Understanding CUDA's memory hierarchy is crucial for performance:

1. **Global Memory**: Accessible by all threads, highest latency
2. **Shared Memory**: Fast, on-chip memory shared within a block
3. **Registers**: Fastest, private to each thread
4. **Constant Memory**: Read-only, cached memory for uniform data

## Your First CUDA Program

Here's a simple example that adds two vectors:

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

## Key Performance Considerations

1. **Memory Coalescing**: Ensure consecutive threads access consecutive memory addresses
2. **Occupancy**: Maximize the number of active warps per SM
3. **Divergence**: Minimize thread divergence within warps
4. **Memory Transfers**: Minimize data transfers between host and device

## Next Steps

To master CUDA programming:

- Study warp-level primitives and synchronization
- Learn about streams and concurrent kernel execution
- Explore advanced memory patterns (texture memory, unified memory)
- Profile your code using NVIDIA Nsight tools

CUDA opens up exciting possibilities for accelerating compute-intensive workloads. In future posts, we'll dive deeper into optimization techniques and real-world applications.
