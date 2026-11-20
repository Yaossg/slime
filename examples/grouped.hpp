#pragma once
#include "../include/slime/upper_interval.hpp"
#if defined(__CUDACC__)
#define GROUP_HD __host__ __device__ __forceinline__
#else
#define GROUP_HD inline
#endif
namespace reverse_demo {
namespace sr = slime;
struct Prepared {
    uint32_t h[25], b[25];
};
GROUP_HD void prepare(uint32_t root, const sr::CellTerm *cells, Prepared &out) {
    for (int i = 0; i < 25; ++i) {
        const uint64_t a = root + cells[i].term;
        out.h[i] = uint32_t(a >> 18);
        const uint64_t lo = (a ^ sr::kCombinedXor) & ((1U << 18) - 1);
        out.b[i] = uint32_t((lo * sr::kJavaMultiplier + 11) >> 17);
    }
}
GROUP_HD uint32_t preparedQ(const Prepared &p, int i, uint32_t m) {
    const uint32_t u = (m + p.h[i]) ^ uint32_t(sr::kCombinedXor >> 18);
    return (p.b[i] + 2U * uint32_t(sr::kJavaMultiplier) * u) & 0x7fffffffU;
}

// Bits 4..7 of kCombinedXor >> 18 are zero. For m=base+16*j,
// j in [0,16), m XOR C is affine. Each shifted input crosses at
// most one 256 boundary, so each output difference has at most two values.
// base bits 4..7 must be zero; cells[0].term must be zero (normalized anchor).
GROUP_HD uint32_t groupMask(const Prepared &p, uint32_t base, int count = 25) {
    uint32_t mask = 65535;
    const uint32_t q0 = preparedQ(p, 0, base), qend = preparedQ(p, 0, base + 240);
    for (int i = 1; i < count && mask; ++i) {
        const uint32_t n = (256 - ((base + p.h[i]) & 255) + 15) >> 4;
        const uint32_t before = (1U << n) - 1;
        const uint32_t r0 = ((preparedQ(p, i, base) - q0) & 0x7fffffffU) % 10;
        const uint32_t r1 = ((preparedQ(p, i, base + 240) - qend) & 0x7fffffffU) % 10;
        const uint32_t allowed =
            ((r0 == 0 || r0 == 8) ? before : 0U) | ((r1 == 0 || r1 == 8) ? (65535U ^ before) : 0U);
        mask &= allowed;
    }
    return mask;
}
} // namespace reverse_demo
#undef GROUP_HD
