#pragma once

// Coordinate mapping: one work item is a fixed (x,z). Its private axis terms are computed
// once, then reused for every seed. No host table or global coordinate table.
template <int N>
__device__ __forceinline__ bool solidAxes(uint64_t seed, const uint64_t (&xs)[N],
                                          const uint64_t (&zs)[N]) {
#pragma unroll
    for (int dx = 0; dx < N; ++dx) {
#pragma unroll
        for (int dz = 0; dz < N; ++dz) {
            const uint64_t t = xs[dx] + zs[dz];
            if (!forward::original(seed, t))
                return false;
        }
    }
    return true;
}

template <int N>
__global__ void coordinateKernel(DeviceArgs a, forward::Hit *hits, unsigned long long *count) {
    const uint64_t area = a.width * a.height;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t p = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < area; p += stride) {
        const int32_t x = int32_t(int64_t(a.x0) + int64_t(p / a.height));
        const int32_t z = int32_t(int64_t(a.z0) + int64_t(p % a.height));
        uint64_t xs[N], zs[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            xs[i] = forward::xTerm(x + i);
            zs[i] = forward::zTerm(z + i);
        }
        for (uint64_t s = 0; s < a.seeds; ++s)
            if (solidAxes<N>(a.seed + s, xs, zs))
                emit(a, hits, count, a.seed + s, x, z);
    }
}

// Compile-time sizes let the compiler scalarize/unroll the private axis arrays.
// Dispatch once on the host, outside the kernel's timed seed loop.
void launchCoordinate(int blocks, int threads, DeviceArgs a, forward::Hit *hits,
                      unsigned long long *count) {
    switch (a.n) {
    case 1:
        coordinateKernel<1><<<blocks, threads>>>(a, hits, count);
        break;
    case 2:
        coordinateKernel<2><<<blocks, threads>>>(a, hits, count);
        break;
    case 3:
        coordinateKernel<3><<<blocks, threads>>>(a, hits, count);
        break;
    case 4:
        coordinateKernel<4><<<blocks, threads>>>(a, hits, count);
        break;
    case 5:
        coordinateKernel<5><<<blocks, threads>>>(a, hits, count);
        break;
    case 6:
        coordinateKernel<6><<<blocks, threads>>>(a, hits, count);
        break;
    case 7:
        coordinateKernel<7><<<blocks, threads>>>(a, hits, count);
        break;
    case 8:
        coordinateKernel<8><<<blocks, threads>>>(a, hits, count);
        break;
    default:
        throw std::runtime_error("invalid pattern size");
    }
}
