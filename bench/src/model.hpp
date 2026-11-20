#pragma once

#include "../../include/slime/upper_interval.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sr = slime;
struct Coord {
    int32_t x, z;
};
struct Job {
    uint32_t position, root;
};
struct Match {
    uint64_t seed;
    int32_t x, z;
};
struct PatternCell {
    int x, z;
    uint8_t requirement;
};
struct Pattern {
    int width = 0, height = 0, positive = 0;
    std::vector<std::string> rows;
    std::vector<PatternCell> cells;
};
Pattern parsePattern(std::string text) {
    std::replace(text.begin(), text.end(), '/', '\n');
    Pattern p;
    std::stringstream input(text);
    std::string row;
    while (std::getline(input, row)) {
        if (!row.empty() && row.back() == '\r')
            row.pop_back();
        if (row.empty())
            throw std::runtime_error("empty pattern row");
        p.rows.push_back(row);
    }
    if (p.rows.empty())
        throw std::runtime_error("empty pattern");
    p.width = p.rows[0].size();
    p.height = p.rows.size();
    if (p.width > 64 || p.height > 64)
        throw std::runtime_error("pattern dimensions must be 1..64");
    for (auto &r : p.rows)
        if (r.size() != size_t(p.width))
            throw std::runtime_error("pattern rows must have equal width");
    for (char wanted : {'#', '.'})
        for (int z = 0; z < p.height; ++z)
            for (int x = 0; x < p.width; ++x) {
                char ch = p.rows[z][x];
                if (ch != '#' && ch != '.' && ch != '?')
                    throw std::runtime_error("pattern characters must be #, . or ?");
                if (ch == wanted)
                    p.cells.push_back({x, z, uint8_t(ch == '#' ? sr::kSlime : sr::kNonSlime)});
                if (wanted == '#' && ch == '#')
                    ++p.positive;
            }
    if (!p.positive)
        throw std::runtime_error("pattern requires at least one # cell");
    return p;
}
struct Config {
    Pattern pattern;
    int32_t x = 0, z = 0;
    uint64_t radius = 0;
    int size = 5, positionBatch = 32, jobBatch = 2048, repeats = 1;
    uint64_t startAnchor = 0, anchorCount = 0;
    bool countSet = false;
    uint64_t maxResults = 0;
    size_t cacheClasses = 4096;
    std::string devices = "0", output;
    bool dry = false, list = false;
};
uint64_t number(const std::string &s) {
    size_t used = 0;
    if (s.empty() || s[0] == '-')
        throw std::runtime_error("expected nonnegative integer: " + s);
    auto n = std::stoull(s, &used);
    if (used != s.size())
        throw std::runtime_error("invalid integer: " + s);
    return n;
}
int32_t signed32(const std::string &s) {
    size_t used = 0;
    auto n = std::stoll(s, &used);
    if (used != s.size() || n < INT32_MIN || n > INT32_MAX)
        throw std::runtime_error("invalid int32 coordinate");
    return int32_t(n);
}
uint64_t anchors(uint64_t r) { return 1 + 2 * r * (r + 1); }
uint64_t isqrt(uint64_t v) {
    uint64_t r = std::sqrt(double(v));
    while ((r + 1) * (r + 1) <= v)
        ++r;
    while (r * r > v)
        --r;
    return r;
}
Coord coordinate(const Config &c, uint64_t index) {
    uint64_t top = (c.radius + 1) * (c.radius + 1);
    int64_t dx, dz;
    if (index < top) {
        uint64_t row = isqrt(index);
        dz = int64_t(row) - c.radius;
        dx = int64_t(index - row * row) - int64_t(row);
    } else {
        uint64_t back = anchors(c.radius) - 1 - index, row = isqrt(back);
        dz = int64_t(c.radius) - row;
        dx = int64_t(row) - int64_t(back - row * row);
    }
    return {int32_t(int64_t(c.x) + dx), int32_t(int64_t(c.z) + dz)};
}
uint64_t classKey(Coord p) {
    return (uint64_t(uint32_t(p.x) & 65535) << 17) | (uint32_t(p.z) & 131071);
}
void terms(Coord p, int n, sr::CellTerm *out) {
    uint64_t anchor = sr::coordinateTerm(p.x, p.z);
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x)
            out[z * n + x] = {(sr::coordinateTerm(p.x + x, p.z + z) - anchor) & sr::kMask48,
                              sr::kSlime};
}
void terms(Coord p, const Pattern &pattern, sr::CellTerm *out) {
    // The grouped filter needs its first positive term to be zero.
    auto first = pattern.cells.front();
    uint64_t anchor = sr::coordinateTerm(p.x + first.x, p.z + first.z);
    size_t i = 0;
    for (auto cell : pattern.cells)
        out[i++] = {(sr::coordinateTerm(p.x + cell.x, p.z + cell.z) - anchor) & sr::kMask48,
                    cell.requirement};
}
