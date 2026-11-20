#pragma once

#include "../include/slime/solver_core.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace forward {
constexpr uint64_t A = 0x5DEECE66DULL;
constexpr uint64_t B = 11;
constexpr uint64_t MASK48 = (1ULL << 48) - 1;
constexpr uint64_t SLIME_XOR = 987234911;
constexpr uint32_t REJECTION_START = 2147483640U;
constexpr int MAX_N = 8;

// The shared implementation defines Java int wrapping and sign extension.
HD inline uint64_t xTerm(int32_t x) { return slime::coordinateXTerm(x); }
HD inline uint64_t zTerm(int32_t z) { return slime::coordinateZTerm(z); }
HD inline uint64_t term(int32_t x, int32_t z) { return slime::coordinateTerm(x, z); }

// Forward examples: the original two-XOR JavaRandom path, with defined arithmetic.
struct JavaRandom {
    uint64_t state;
    HD explicit JavaRandom(uint64_t input) : state((input ^ A) & MASK48) {}
    HD uint32_t next31() {
        state = (state * A + B) & MASK48;
        return uint32_t(state >> 17);
    }
    HD bool slime() {
        for (;;) {
            const uint32_t q = next31();
            if (q < REJECTION_START)
                return q % 10 == 0;
        }
    }
};
HD inline bool original(uint64_t seed, uint64_t t) {
    JavaRandom random((seed + t) ^ SLIME_XOR);
    return random.slime();
}

HD inline bool solid(uint64_t seed, int32_t x, int32_t z, int n) {
    for (int dx = 0; dx < n; ++dx) {
        for (int dz = 0; dz < n; ++dz) {
            const uint64_t t = term(x + dx, z + dz);
            if (!original(seed, t))
                return false;
        }
    }
    return true;
}

struct Args {
    uint64_t seed = 61692151016558ULL, seeds = 1;
    int32_t x0 = 50941, z0 = 418996;
    uint64_t width = 257, height = 257;
    int n = 5, repeats = 5, cpuThreads = 16;
    uint64_t capacity = 100000;
    std::string modes = "all";
    uint64_t positions() const { return width * height; }
    uint64_t work() const { return seeds * positions(); }
};

inline uint64_t positive(const std::string &s) {
    if (s.empty() || s[0] == '-')
        throw std::runtime_error("expected unsigned integer");
    size_t end = 0;
    const uint64_t v = std::stoull(s, &end);
    if (end != s.size())
        throw std::runtime_error("invalid integer");
    return v;
}
inline Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "--seed N --seeds N --x0 X --z0 Z --width N --height N -n N "
                         "--repeats N --cpu-threads N --capacity N --modes all|2,3,4\n";
            std::exit(0);
        }
        if (++i >= argc)
            throw std::runtime_error("missing value: " + key);
        const std::string v = argv[i];
        if (key == "--seed")
            a.seed = positive(v);
        else if (key == "--seeds")
            a.seeds = positive(v);
        else if (key == "--x0")
            a.x0 = std::stoi(v);
        else if (key == "--z0")
            a.z0 = std::stoi(v);
        else if (key == "--width")
            a.width = positive(v);
        else if (key == "--height")
            a.height = positive(v);
        else if (key == "-n")
            a.n = std::stoi(v);
        else if (key == "--repeats")
            a.repeats = std::stoi(v);
        else if (key == "--cpu-threads")
            a.cpuThreads = std::stoi(v);
        else if (key == "--capacity")
            a.capacity = positive(v);
        else if (key == "--modes")
            a.modes = v;
        else
            throw std::runtime_error("unknown option: " + key);
    }
    if (a.n < 1 || a.n > MAX_N || !a.seeds || !a.width || !a.height || a.repeats < 1 ||
        a.cpuThreads < 1 || a.cpuThreads > 256 || !a.capacity || a.capacity > 10000000 ||
        a.width > 10000000 || a.height > 10000000 || a.seeds > 10000000)
        throw std::runtime_error("invalid range/size/count");
    if (a.seeds - 1 > UINT64_MAX - a.seed || a.seeds > UINT64_MAX / a.positions())
        throw std::runtime_error("seed/work range overflow");
    // Include the expanded 16x16 tile's unused fringe so CUDA precomputation is safe.
    if (int64_t(a.x0) + int64_t((a.width + 15) / 16 * 16) + a.n - 2 > INT32_MAX ||
        int64_t(a.z0) + int64_t((a.height + 15) / 16 * 16) + a.n - 2 > INT32_MAX)
        throw std::runtime_error("coordinate footprint/tile exceeds int32");
    return a;
}

struct Hit {
    uint64_t seed;
    int32_t x, z;
    bool operator<(const Hit &o) const { return std::tie(seed, x, z) < std::tie(o.seed, o.x, o.z); }
    bool operator==(const Hit &o) const { return seed == o.seed && x == o.x && z == o.z; }
};
inline double ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}
inline void array(const std::vector<double> &v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << v[i];
    }
    std::cout << ']';
}
inline void results(const std::vector<Hit> &v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << '[' << v[i].seed << ',' << v[i].x << ',' << v[i].z << ']';
    }
    std::cout << ']';
}
inline void workload(const Args &a) {
    std::cout << "\"seed_start\":" << a.seed << ",\"seeds\":" << a.seeds << ",\"x0\":" << a.x0
              << ",\"z0\":" << a.z0 << ",\"width\":" << a.width << ",\"height\":" << a.height
              << ",\"n\":" << a.n << ",\"candidate_count\":" << a.work();
}
} // namespace forward
