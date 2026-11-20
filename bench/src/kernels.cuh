#pragma once

#include "model.hpp"
#include "cuda_support.cuh"
__global__ void rootsKernel(const sr::CellTerm *cells, uint32_t positions, int count, int positive,
                            Job *out, unsigned long long *total) {
    uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= (uint64_t(positions) << 18))
        return;
    uint32_t pos = t >> 18, root = t & 262143;
    const auto *p = cells + uint64_t(pos) * count;
    for (int i = 0; i < positive; ++i)
        if ((((root + uint32_t(p[i].term)) ^ uint32_t(sr::kCombinedXor)) *
                 uint32_t(sr::kJavaMultiplier) +
             11U) &
            (1U << 17))
            return;
    out[atomicAdd(total, 1ULL)] = {pos, root};
}
__global__ void prepareKernel(const sr::CellTerm *cells, const Job *jobs, uint32_t n, int count,
                              int positive, uint32_t *prepared) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    const auto job = jobs[k];
    for (int i = 0; i < positive; ++i) {
        uint64_t a = job.root + cells[uint64_t(job.position) * count + i].term;
        prepared[(uint64_t(k) * count + i) * 2] = a >> 18;
        prepared[(uint64_t(k) * count + i) * 2 + 1] =
            ((((a ^ sr::kCombinedXor) & 262143) * sr::kJavaMultiplier + 11) >> 17);
    }
}
__host__ __device__ __forceinline__ uint32_t q(const uint32_t *p, int i, uint32_t m) {
    return (p[2 * i + 1] + 2U * uint32_t(sr::kJavaMultiplier) *
                               ((m + p[2 * i]) ^ uint32_t(sr::kCombinedXor >> 18))) &
           0x7fffffffU;
}
__host__ __device__ uint32_t groupMask(const uint32_t *p, uint32_t base, int count) {
    uint32_t mask = 65535, q0 = q(p, 0, base), qe = q(p, 0, base + 240);
    for (int i = 1; i < count && mask; ++i) {
        uint32_t cut = (256 - ((base + p[2 * i]) & 255) + 15) >> 4, lo = (1U << cut) - 1;
        uint32_t r0 = ((q(p, i, base) - q0) & 0x7fffffffU) % 10,
                 r1 = ((q(p, i, base + 240) - qe) & 0x7fffffffU) % 10;
        mask &= ((r0 == 0 || r0 == 8) ? lo : 0U) | ((r1 == 0 || r1 == 8) ? (65535U ^ lo) : 0U);
    }
    return mask;
}
struct Emit {
    Coord p;
    uint64_t originTerm;
    const sr::CellTerm *negative;
    int negativeCount;
    Match *out;
    uint64_t capacity;
    unsigned long long *total;
    __device__ void operator()(uint64_t seed) {
        if (!sr::verifyPattern(seed, negative, negativeCount))
            return;
        uint64_t slot = atomicAdd(total, 1ULL);
        if (slot < capacity)
            out[slot] = {(seed - originTerm) & sr::kMask48, p.x, p.z};
    }
};
__global__ void searchKernel(const sr::CellTerm *cells, const Coord *coords, const Job *jobs,
                             uint32_t n, int count, int positive, int originX, int originZ,
                             const uint32_t *prepared, Match *out, uint64_t capacity,
                             unsigned long long *total) {
    uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= (uint64_t(n) << 13))
        return;
    uint32_t g = index & 8191, k = index >> 13, base = ((g >> 4) << 8) + (g & 15);
    const uint32_t *p = prepared + uint64_t(k) * count * 2;
    uint32_t mask = groupMask(p, base, positive);
    Job job = jobs[k];
    Coord origin = coords[job.position];
    Emit emit{origin,
              sr::coordinateTerm(origin.x + originX, origin.z + originZ),
              cells + uint64_t(job.position) * count + positive,
              count - positive,
              out,
              capacity,
              total};
    while (mask) {
        uint32_t j = __ffs(mask) - 1;
        mask &= mask - 1;
        sr::solveIntervals(job.root + (uint64_t(base + 16 * j) << 18),
                           cells + uint64_t(job.position) * count, positive, emit);
    }
}
