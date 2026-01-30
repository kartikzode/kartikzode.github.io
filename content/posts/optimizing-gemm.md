---
title: "Optimizing GEMM: A Journey to Peak Performance"
date: 2024-01-20T10:00:00-00:00
draft: false
tags: ["GEMM", "Performance", "Linear Algebra", "CUDA"]
categories: ["GEMM Optimizations"]
author: "Thombya"
showToc: true
TocOpen: false
description: "Exploring techniques to optimize General Matrix Multiplication (GEMM) for maximum performance"
---

# Optimizing GEMM: A Journey to Peak Performance

General Matrix Multiplication (GEMM) is one of the most fundamental operations in scientific computing and deep learning. The operation `C = αAB + βC` is deceptively simple but achieving peak performance requires careful optimization.

## Why GEMM Matters

GEMM is the computational backbone of:
- Deep learning (neural network training and inference)
- Scientific simulations
- Computer graphics
- Signal processing

Modern GPUs can achieve tens of TFLOPS, but naive implementations often achieve only a small fraction of this theoretical peak.

## The Naive Approach

Let's start with a straightforward CPU implementation:

```cpp
void gemm_naive(int M, int N, int K, 
                float *A, float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
```

This implementation is simple but suffers from poor cache locality and achieves only a tiny fraction of peak performance.

## Key Optimization Strategies

### 1. Blocking/Tiling

Break the computation into smaller blocks that fit in cache:

```cpp
void gemm_blocked(int M, int N, int K, 
                  float *A, float *B, float *C) {
    const int BLOCK_SIZE = 32;
    
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                // Multiply block A[i:i+BS, k:k+BS] 
                // with B[k:k+BS, j:j+BS]
                for (int ii = i; ii < min(i+BLOCK_SIZE, M); ii++) {
                    for (int jj = j; jj < min(j+BLOCK_SIZE, N); jj++) {
                        float sum = C[ii*N + jj];
                        for (int kk = k; kk < min(k+BLOCK_SIZE, K); kk++) {
                            sum += A[ii*K + kk] * B[kk*N + jj];
                        }
                        C[ii*N + jj] = sum;
                    }
                }
            }
        }
    }
}
```

### 2. SIMD Vectorization

Modern CPUs support SIMD instructions (AVX, AVX2, AVX-512) that can process multiple elements simultaneously:

```cpp
#include <immintrin.h>

void gemm_vectorized(int M, int N, int K, 
                     float *A, float *B, float *C) {
    // Process 8 floats at a time with AVX
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a = _mm256_broadcast_ss(&A[i*K + k]);
                __m256 b = _mm256_loadu_ps(&B[k*N + j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            _mm256_storeu_ps(&C[i*N + j], sum);
        }
    }
}
```

### 3. GPU Implementation with CUDA

For maximum performance, we leverage GPU parallelism with shared memory:

```cuda
__global__ void gemm_cuda(int M, int N, int K,
                          float *A, float *B, float *C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t*TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row*K + t*TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t*TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row*N + col] = sum;
}
```

### 4. Tensor Cores

Modern NVIDIA GPUs (Volta and later) include Tensor Cores for mixed-precision matrix operations:

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_tensor_core(half *A, half *B, float *C, 
                                  int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Load and compute
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}
```

## Performance Results

Typical performance improvements on an NVIDIA A100:

| Implementation | TFLOPS | % of Peak |
|---------------|--------|-----------|
| Naive CPU | 0.01 | 0.1% |
| Blocked CPU | 0.1 | 1% |
| Vectorized CPU | 0.5 | 5% |
| Basic CUDA | 5 | 25% |
| Optimized CUDA | 15 | 75% |
| cuBLAS (NVIDIA) | 19 | 95% |

## Lessons Learned

1. **Memory is King**: Optimizing memory access patterns is more important than raw compute
2. **Cache Hierarchy**: Understanding and exploiting the cache hierarchy is essential
3. **Parallelism**: Both thread-level and instruction-level parallelism matter
4. **Use Libraries**: cuBLAS, MKL, and other vendor libraries are heavily optimized

## Next Steps

In future posts, we'll explore:
- Auto-tuning GEMM parameters
- Mixed-precision optimizations
- Integration with deep learning frameworks
- Benchmarking and profiling techniques

GEMM optimization is a deep rabbit hole, but the principles learned apply broadly to high-performance computing.
