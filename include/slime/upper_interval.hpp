#pragma once

#include "solver_core.hpp"

#if defined(__CUDACC__)
#define SLIME_INTERVAL_HD __host__ __device__
#define SLIME_INTERVAL_INLINE __forceinline__
#else
#define SLIME_INTERVAL_HD
#define SLIME_INTERVAL_INLINE inline
#endif

namespace slime {
namespace sr = slime;

// Positive (slime) constraints only; positions may be sparse.
// Each surviving residue has a single interval. Keeping the first cell's
// two residues makes intersection constant-space. Check negative cells on emit.
template <class Emit>
SLIME_INTERVAL_HD SLIME_INTERVAL_INLINE void solveIntervals(uint64_t low, const sr::CellTerm *cells,
                                                            int count, Emit &emit) {
    low &= (1ULL << kLowerBits) - 1;
    uint32_t q = uint32_t(sr::nextStateFromTerm(low, cells[0].term) >> 17);
    if (q & 1U)
        return;
    uint32_t b = q >> 18;
    uint32_t r = q & ((1U << 18) - 1);
    uint32_t threshold = kUpperSize - b;
    uint32_t residue[2] = {(r % 5 + 5 - b % 5) % 5, 0};
    residue[1] = (residue[0] + 2) % 5;
    uint32_t lo[2] = {0, threshold}, hi[2] = {threshold, kUpperSize};
    unsigned active = 0;
    for (int k = 0; k < 2; ++k)
        if (sr::firstWithResidue(lo[k], residue[k]) < hi[k])
            active |= 1U << k;
    for (int i = 1; i < count && active; ++i) {
        q = uint32_t(sr::nextStateFromTerm(low, cells[i].term) >> 17);
        if (q & 1U)
            return;
        b = q >> 18;
        r = q & ((1U << 18) - 1);
        threshold = kUpperSize - b;
        const uint32_t before = (r % 5 + 5 - b % 5) % 5;
        const uint32_t after = (before + 2) % 5;
        for (int k = 0; k < 2; ++k)
            if (active & (1U << k)) {
                if (residue[k] == before)
                    hi[k] = hi[k] < threshold ? hi[k] : threshold;
                else if (residue[k] == after)
                    lo[k] = lo[k] > threshold ? lo[k] : threshold;
                else {
                    active &= ~(1U << k);
                    continue;
                }
                if (sr::firstWithResidue(lo[k], residue[k]) >= hi[k])
                    active &= ~(1U << k);
            }
    }
    for (int k = 0; k < 2; ++k)
        if (active & (1U << k)) {
            for (uint32_t t = sr::firstWithResidue(lo[k], residue[k]); t < hi[k]; t += 5) {
                const uint64_t seed =
                    low + (uint64_t((kHalfDeltaInverse * t) & kUpperMask) << kLowerBits);
                if (sr::verifyPattern(seed, cells, count))
                    emit(seed);
            }
        }
}

// A cheap necessary condition before high-bit interval solving.
// Adding the common high-bit shift can wrap q_i-q_0 once: delta mod 10 is 0 or 8.
SLIME_INTERVAL_HD SLIME_INTERVAL_INLINE bool
differenceFilter(uint64_t low, const sr::CellTerm *cells, int count) {
    const uint32_t q0 = uint32_t(sr::nextStateFromTerm(low, cells[0].term) >> 17);
    for (int i = 1; i < count; ++i) {
        const uint32_t qi = uint32_t(sr::nextStateFromTerm(low, cells[i].term) >> 17);
        const uint32_t delta = (qi - q0) & 0x7fffffffU;
        const uint32_t r = delta % 10;
        if (r != 0 && r != 8)
            return false;
    }
    return true;
}
} // namespace slime

#undef SLIME_INTERVAL_HD
#undef SLIME_INTERVAL_INLINE
