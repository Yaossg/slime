#pragma once

// CUDA translation of CPU forward search: one work item is one seed, its positions are serial.
__global__ void seedKernel(DeviceArgs a, forward::Hit *hits, unsigned long long *count) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t s = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; s < a.seeds; s += stride) {
        for (uint64_t xi = 0; xi < a.width; ++xi)
            for (uint64_t zi = 0; zi < a.height; ++zi) {
                const int32_t x = int32_t(int64_t(a.x0) + int64_t(xi));
                const int32_t z = int32_t(int64_t(a.z0) + int64_t(zi));
                if (forward::solid(a.seed + s, x, z, a.n))
                    emit(a, hits, count, a.seed + s, x, z);
            }
    }
}
