#include "src/kernels.cuh"
#include <iostream>
#include <set>
#include <random>
struct HostEmit {
    const sr::CellTerm *cells;
    int count;
    std::set<uint64_t> *result;
    __host__ __device__ void operator()(uint64_t seed) {
#ifndef __CUDA_ARCH__
        if (sr::verifyPattern(seed, cells, count))
            result->insert(seed);
#endif
    }
};
void check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
int main() {
    uint64_t points = 0, groups = 0, mixed = 0;
    std::mt19937_64 rng(123456);
    for (int r = 0; r <= 64; ++r) {
        Config c;
        c.x = -123;
        c.z = 98;
        c.radius = r;
        std::set<std::pair<int32_t, int32_t>> actual;
        for (uint64_t i = 0; i < anchors(r); ++i) {
            auto p = coordinate(c, i);
            actual.emplace(p.x, p.z);
        }
        check(actual.size() == anchors(r), "duplicate anchor");
        for (int dz = -r; dz <= r; ++dz)
            for (int dx = -r; dx <= r; ++dx)
                check(bool(actual.count({c.x + dx, c.z + dz})) ==
                          (std::abs(dx) + std::abs(dz) <= r),
                      "diamond coverage mismatch");
        points += actual.size();
    }
    Config large;
    large.radius = 1000000;
    for (int t = 0; t < 100000; ++t) {
        auto p = coordinate(large, rng() % anchors(large.radius));
        check(std::abs(int64_t(p.x)) + std::abs(int64_t(p.z)) <= int64_t(large.radius),
              "large diamond mismatch");
    }
    for (int n : {1, 2, 3, 4, 5, 6})
        for (int t = 0; t < 512; ++t) {
            Coord coord{int32_t(rng() % 3750000) - 1875000, int32_t(rng() % 3750000) - 1875000};
            int count = n * n;
            std::vector<sr::CellTerm> cells(count);
            terms(coord, n, cells.data());
            uint32_t root = rng() & 262143, base = (rng() & 131071) & ~240U;
            std::vector<uint32_t> prepared(count * 2);
            for (int i = 0; i < count; ++i) {
                uint64_t a = root + cells[i].term;
                prepared[2 * i] = a >> 18;
                prepared[2 * i + 1] =
                    ((((a ^ sr::kCombinedXor) & 262143) * sr::kJavaMultiplier + 11) >> 17);
                for (int j = 0; j < 16; ++j)
                    check(q(prepared.data(), i, base + 16 * j) ==
                              (sr::nextStateFromTerm(root + (uint64_t(base + 16 * j) << 18),
                                                     cells[i].term) >>
                               17),
                          "prepared q mismatch");
            }
            uint32_t expected = 0;
            for (int j = 0; j < 16; ++j)
                if (sr::differenceFilter(root + (uint64_t(base + 16 * j) << 18), cells.data(),
                                         count))
                    expected |= 1U << j;
            check(groupMask(prepared.data(), base, count) == expected,
                  "variable-size group mask mismatch");
            ++groups;
        }
    // Compare arbitrary sparse layouts and forbidden cells against all high bits.
    for (int t = 0; t < 128; ++t) {
        Coord coord{int32_t(rng()), int32_t(rng())};
        // Keep offsets away from overflow even for signed boundary samples.
        coord.x /= 2;
        coord.z /= 2;
        uint64_t world = rng() & sr::kMask48;
        std::string text;
        for (int z = 0; z < 5; ++z) {
            if (z)
                text += '/';
            for (int x = 0; x < 7; ++x)
                text +=
                    (x == 0 && z == 0) || rng() % 4 == 0 ? '?'
                    : sr::exactIsSlimeFromTerm(world, sr::coordinateTerm(coord.x + x, coord.z + z))
                        ? '#'
                        : '.';
        }
        if (text.find('#') == std::string::npos)
            continue;
        auto pattern = parsePattern(text);
        int count = pattern.cells.size();
        std::vector<sr::CellTerm> cells(count);
        terms(coord, pattern, cells.data());
        uint64_t normalized = (world + sr::coordinateTerm(coord.x + pattern.cells.front().x,
                                                          coord.z + pattern.cells.front().z)) &
                              sr::kMask48;
        check(sr::verifyPattern(normalized, cells.data(), count), "normalized pattern mismatch");
        uint64_t low = normalized & ((1ULL << 35) - 1);
        std::set<uint64_t> actual, expected;
        HostEmit emit{cells.data() + pattern.positive, count - pattern.positive, &actual};
        sr::solveIntervals(low, cells.data(), pattern.positive, emit);
        for (uint64_t high = 0; high < 8192; ++high) {
            uint64_t seed = low + (high << 35);
            // Ordinary branch requires a successful first draw at positive cells.
            bool ordinary = true;
            for (int i = 0; i < pattern.positive; ++i)
                if ((sr::nextStateFromTerm(seed, cells[i].term) >> 17) % 10 != 0)
                    ordinary = false;
            if (ordinary && sr::verifyPattern(seed, cells.data(), count))
                expected.insert(seed);
        }
        check(actual == expected, "mixed pattern high-bit solver mismatch");
        ++mixed;
        uint32_t root = low & 262143, base = uint32_t(low >> 18) & ~240U;
        std::vector<uint32_t> prepared(count * 2);
        for (int i = 0; i < pattern.positive; ++i) {
            uint64_t a = root + cells[i].term;
            prepared[2 * i] = a >> 18;
            prepared[2 * i + 1] =
                ((((a ^ sr::kCombinedXor) & 262143) * sr::kJavaMultiplier + 11) >> 17);
        }
        uint32_t wanted = 0;
        for (int j = 0; j < 16; ++j)
            if (sr::differenceFilter(root + (uint64_t(base + 16 * j) << 18), cells.data(),
                                     pattern.positive))
                wanted |= 1U << j;
        check(groupMask(prepared.data(), base, pattern.positive) == wanted,
              "sparse pattern group mask mismatch");
    }
    std::cout << "{\"diamond_points\":" << points << ",\"variable_size_groups\":" << groups
              << ",\"mixed_pattern_slices\":" << mixed << ",\"passed\":true}\n";
}
