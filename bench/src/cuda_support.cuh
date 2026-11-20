#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#define CUDA(call)                                                                                 \
    do {                                                                                           \
        cudaError_t status_ = (call);                                                              \
        if (status_ != cudaSuccess)                                                                \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_));     \
    } while (0)
template <class T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t n) {
        if (n)
            CUDA(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    Buffer(const Buffer &) = delete;
};
