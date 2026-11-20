#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define SLIME_HD __host__ __device__
#define SLIME_INLINE __forceinline__
#else
#define SLIME_HD
#define SLIME_INLINE inline
#endif

namespace slime {

constexpr uint64_t kJavaMultiplier = 0x5DEECE66DULL;
constexpr uint64_t kJavaAddend = 0xBULL;
constexpr uint64_t kMask48 = (1ULL << 48) - 1ULL;
constexpr uint64_t kSlimeXor = 987234911ULL;
constexpr uint64_t kCombinedXor = kSlimeXor ^ kJavaMultiplier;
constexpr int kLowerBits = 35;
constexpr int kUpperBits = 13;
constexpr uint32_t kUpperSize = 1U << kUpperBits;
constexpr uint32_t kUpperMask = kUpperSize - 1U;
constexpr uint32_t kHalfDeltaInverse = 4965U;
constexpr int kMaxPatternCells = 64;
constexpr uint32_t kX2Coeff = 4987142U;
constexpr uint32_t kXCoeff = 5947611U;
constexpr uint32_t kZ2Coeff = 4392871U;
constexpr uint32_t kZCoeff = 389711U;

enum Requirement : uint8_t {
    kNonSlime = 0,
    kSlime = 1,
};

struct CellTerm {
    uint64_t term;
    uint8_t requirement;
};

SLIME_HD SLIME_INLINE uint64_t javaIntAsLongBits(uint32_t value) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(value)));
}

SLIME_HD SLIME_INLINE uint64_t coordinateTerm(int32_t chunkX, int32_t chunkZ) {
    const uint32_t x = static_cast<uint32_t>(chunkX);
    const uint32_t z = static_cast<uint32_t>(chunkZ);
    return javaIntAsLongBits(x * x * kX2Coeff) + javaIntAsLongBits(x * kXCoeff) +
           javaIntAsLongBits(z * z) * kZ2Coeff + javaIntAsLongBits(z * kZCoeff);
}

SLIME_HD SLIME_INLINE uint64_t coordinateXTerm(int32_t chunkX) {
    const uint32_t x = static_cast<uint32_t>(chunkX);
    return javaIntAsLongBits(x * x * kX2Coeff) + javaIntAsLongBits(x * kXCoeff);
}

SLIME_HD SLIME_INLINE uint64_t coordinateZTerm(int32_t chunkZ) {
    const uint32_t z = static_cast<uint32_t>(chunkZ);
    return javaIntAsLongBits(z * z) * kZ2Coeff + javaIntAsLongBits(z * kZCoeff);
}

SLIME_HD SLIME_INLINE uint64_t nextStateFromTerm(uint64_t seed48, uint64_t term) {
    uint64_t state = ((seed48 + term) ^ kCombinedXor) & kMask48;
    return (state * kJavaMultiplier + kJavaAddend) & kMask48;
}

SLIME_HD SLIME_INLINE bool exactIsSlimeFromTerm(uint64_t seed48, uint64_t term) {
    uint64_t state = ((seed48 + term) ^ kCombinedXor) & kMask48;
    for (;;) {
        state = (state * kJavaMultiplier + kJavaAddend) & kMask48;
        const uint32_t bits = static_cast<uint32_t>(state >> 17);
        const uint32_t value = bits % 10U;
        if (bits - value + 9U <= 0x7FFFFFFFU) {
            return value == 0U;
        }
    }
}

SLIME_HD SLIME_INLINE bool verifyPattern(uint64_t seed48, const CellTerm *cells, int cellCount) {
    for (int i = 0; i < cellCount; ++i) {
        const bool slime = exactIsSlimeFromTerm(seed48, cells[i].term);
        if (slime != (cells[i].requirement == kSlime)) {
            return false;
        }
    }
    return true;
}

SLIME_HD SLIME_INLINE uint32_t firstWithResidue(uint32_t start, uint32_t residue) {
    return start + ((residue + 5U - (start % 5U)) % 5U);
}

} // namespace slime

#undef SLIME_HD
#undef SLIME_INLINE
