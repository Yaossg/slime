#pragma once

// Common rectangle path: use block coordinates directly, avoiding 64-bit tile divisions.
// One thread keeps its coordinate while scanning all seeds, as in the coordinate kernel.
__global__ void outerRectKernel(DeviceArgs a, forward::Hit *hits, unsigned long long *count) {
    __shared__ uint64_t xs[16 + forward::MAX_N - 1], zs[16 + forward::MAX_N - 1];
    const unsigned lane = threadIdx.y * 16 + threadIdx.x;
    const uint64_t xb = uint64_t(blockIdx.x) * 16, zb = uint64_t(blockIdx.y) * 16;
    if (lane < unsigned(16 + a.n - 1)) {
        xs[lane] = forward::xTerm(int32_t(int64_t(a.x0) + int64_t(xb + lane)));
        zs[lane] = forward::zTerm(int32_t(int64_t(a.z0) + int64_t(zb + lane)));
    }
    __syncthreads();
    const uint64_t xi = xb + threadIdx.x, zi = zb + threadIdx.y;
    if (xi >= a.width || zi >= a.height)
        return;
    for (uint64_t s = 0; s < a.seeds; ++s) {
        bool ok = true;
        for (int dx = 0; dx < a.n && ok; ++dx)
            for (int dz = 0; dz < a.n; ++dz) {
                if (!forward::original(a.seed + s, xs[threadIdx.x + dx] + zs[threadIdx.y + dz])) {
                    ok = false;
                    break;
                }
            }
        if (ok)
            emit(a, hits, count, a.seed + s, int32_t(int64_t(a.x0) + int64_t(xi)),
                 int32_t(int64_t(a.z0) + int64_t(zi)));
    }
}

// One block handles a 16x16 anchor tile; each axis needs 16+n-1 terms.
// Fallback for unusually tall rectangles exceeding the grid.y limit.
__global__ void outerKernel(DeviceArgs a, forward::Hit *hits, unsigned long long *count) {
    __shared__ uint64_t xs[16 + forward::MAX_N - 1], zs[16 + forward::MAX_N - 1];
    const uint64_t tilesX = (a.width + 15) / 16, tilesZ = (a.height + 15) / 16;
    const uint64_t total = tilesX * tilesZ;
    const unsigned lane = threadIdx.y * 16 + threadIdx.x;
    for (uint64_t tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const uint64_t tx = tile / tilesZ, tz = tile % tilesZ;
        if (lane < unsigned(16 + a.n - 1)) {
            xs[lane] = forward::xTerm(int32_t(int64_t(a.x0) + int64_t(tx * 16 + lane)));
            zs[lane] = forward::zTerm(int32_t(int64_t(a.z0) + int64_t(tz * 16 + lane)));
        }
        __syncthreads();
        const uint64_t xi = tx * 16 + threadIdx.x, zi = tz * 16 + threadIdx.y;
        for (uint64_t s = 0; s < a.seeds; ++s) {
            bool ok = xi < a.width && zi < a.height;
            for (int dx = 0; dx < a.n && ok; ++dx)
                for (int dz = 0; dz < a.n; ++dz) {
                    if (!forward::original(a.seed + s,
                                           xs[threadIdx.x + dx] + zs[threadIdx.y + dz])) {
                        ok = false;
                        break;
                    }
                }
            if (ok)
                emit(a, hits, count, a.seed + s, int32_t(int64_t(a.x0) + int64_t(xi)),
                     int32_t(int64_t(a.z0) + int64_t(zi)));
        }
        // No final barrier is needed for a block assigned only one tile.
        // When a giant workload exceeds the grid limit, synchronize before reuse.
        if (tile + gridDim.x < total)
            __syncthreads();
    }
}
