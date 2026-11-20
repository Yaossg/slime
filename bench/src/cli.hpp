#pragma once

#include "model.hpp"
#include <fstream>
#include <iostream>
void help() {
    std::cout << R"(slime-bench --radius R [options]
Search effective 48-bit seeds for slime chunk patterns.
Ordinary candidate branch only; rejection-tail supplement is NOT included.

  --x X --z Z          Center chunk coordinates (default 0,0)
  --radius R           Manhattan radius in chunks (required, >=0)
  --size N             Solid square side, 1..6 (default 5)
  --pattern TEXT       Rows separated by /; # slime, . non-slime, ? ignored
  --pattern-file FILE  Same pattern, one row per line (dimensions 1..64)
                       Requires at least one #; exclusive with --size
  --devices 0,1,...     CUDA device indices, or all (default 0)
  --start-anchor N    Start at this zero-based anchor index (default 0)
  --anchor-count N    Scan only N anchors from start (default remaining)
  --repeat N           Complete passes; fresh root cache each pass (default 1)
  --position-batch N   Anchors per scheduled batch, 1..256 (default 32)
  --job-batch N        Low18-root jobs per GPU launch, 1..65536 (default 2048)
  --cache-classes N    Per-GPU bounded FIFO root cache entries (default 4096)
  --max-results N      Save up to N example matches; count ALL (default 0)
  --output FILE        Also save the complete JSON report (file must not exist)
  --dry-run            Print scope JSON without using CUDA
  --list-devices       List visible CUDA devices
  --help               Show this help

Scope: abs(anchor_x-X)+abs(anchor_z-Z)<=R, including boundary.
Anchor is the minimum-x/minimum-z corner; the square may extend outside
this diamond. Coordinates are not clipped to Minecraft's world border.
Small squares can have enormous numbers of matching seeds and take very long.
)";
}

bool parseArguments(int argc, char **argv, Config &c, bool &patternSet) {
    bool radiusSet = false, sizeSet = false;
    std::string patternText;
    patternSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            help();
            return false;
        }
        if (a == "--dry-run") {
            c.dry = true;
            continue;
        }
        if (a == "--list-devices") {
            c.list = true;
            continue;
        }
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + a);
        std::string v = argv[++i];
        auto bounded = [&](uint64_t lo, uint64_t hi) {
            auto n = number(v);
            if (n < lo || n > hi)
                throw std::runtime_error("out of range: " + a);
            return n;
        };
        if (a == "--x")
            c.x = signed32(v);
        else if (a == "--z")
            c.z = signed32(v);
        else if (a == "--radius") {
            c.radius = bounded(0, INT32_MAX);
            radiusSet = true;
        } else if (a == "--size") {
            c.size = bounded(1, 6);
            sizeSet = true;
        } else if (a == "--pattern" || a == "--pattern-file") {
            if (patternSet)
                throw std::runtime_error("specify only one pattern");
            patternSet = true;
            if (a == "--pattern")
                patternText = v;
            else {
                std::ifstream file(v);
                if (!file)
                    throw std::runtime_error("cannot read pattern: " + v);
                std::ostringstream text;
                text << file.rdbuf();
                patternText = text.str();
            }
        } else if (a == "--devices")
            c.devices = v;
        else if (a == "--repeat")
            c.repeats = bounded(1, 1000);
        else if (a == "--start-anchor")
            c.startAnchor = number(v);
        else if (a == "--anchor-count") {
            c.anchorCount = number(v);
            c.countSet = true;
        } else if (a == "--position-batch")
            c.positionBatch = bounded(1, 256);
        else if (a == "--job-batch")
            c.jobBatch = bounded(1, 65536);
        else if (a == "--cache-classes")
            c.cacheClasses = bounded(0, 1000000);
        else if (a == "--max-results")
            c.maxResults = bounded(0, 1000000);
        else if (a == "--output")
            c.output = v;
        else
            throw std::runtime_error("unknown option: " + a);
    }

    if (c.list)
        return true;
    if (!radiusSet)
        throw std::runtime_error("--radius is required");
    if (patternSet && sizeSet)
        throw std::runtime_error("--size and pattern options are mutually exclusive");
    if (!patternSet) {
        for (int z = 0; z < c.size; ++z) {
            if (z)
                patternText += '/';
            patternText += std::string(c.size, '#');
        }
    }
    c.pattern = parsePattern(patternText);
    for (auto axis : {std::make_pair(c.x, c.pattern.width), std::make_pair(c.z, c.pattern.height)})
        if (int64_t(axis.first) - int64_t(c.radius) < INT32_MIN ||
            int64_t(axis.first) + int64_t(c.radius) + axis.second - 1 > INT32_MAX)
            throw std::runtime_error("coordinates including pattern must fit int32");
    if (c.startAnchor >= anchors(c.radius))
        throw std::runtime_error("start anchor outside range");
    if (!c.countSet)
        c.anchorCount = anchors(c.radius) - c.startAnchor;
    if (!c.anchorCount || c.anchorCount > anchors(c.radius) - c.startAnchor)
        throw std::runtime_error("anchor count outside range");
    if (!c.output.empty() && std::ifstream(c.output).good())
        throw std::runtime_error("output already exists: " + c.output);

    return true;
}
