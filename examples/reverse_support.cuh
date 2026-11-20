#pragma once

// Fixed 5x5 workload shared by the reverse-search and coordinate-reuse examples.
#include "grouped.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sr = slime;
using Clock = std::chrono::steady_clock;
#define CUDA(call)                                                                                 \
    do {                                                                                           \
        auto e = (call);                                                                           \
        if (e != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));           \
    } while (0)
struct Position {
    int32_t x, z;
    uint64_t anchor;
    sr::CellTerm cells[25];
};
struct Job {
    uint32_t position, root;
};
struct Match {
    uint64_t position, seed;
    bool operator<(const Match &o) const {
        return std::tie(position, seed) < std::tie(o.position, o.seed);
    }
    bool operator==(const Match &o) const { return position == o.position && seed == o.seed; }
};
constexpr uint32_t MATCH_CAPACITY = 1U << 20;
constexpr uint32_t JOB_CAPACITY = 1U << 22;

Position makePosition(int32_t x, int32_t z) {
    Position p{};
    p.x = x;
    p.z = z;
    p.anchor = sr::coordinateTerm(x, z);
    for (int j = 0; j < 5; ++j)
        for (int i = 0; i < 5; ++i)
            p.cells[j * 5 + i] = {(sr::coordinateTerm(x + i, z + j) - p.anchor) & sr::kMask48,
                                  sr::kSlime};
    return p;
}

__global__ void rootKernel(const Position *positions, uint32_t count, Job *jobs,
                           unsigned long long *total) {
    uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t(count) << 18))
        return;
    const uint32_t pos = uint32_t(idx >> 18), root = uint32_t(idx) & ((1U << 18) - 1);
    const Position &p = positions[pos];
    for (int i = 0; i < 25; ++i) {
        const uint32_t in = root + uint32_t(p.cells[i].term);
        if ((((in ^ uint32_t(sr::kCombinedXor)) * uint32_t(sr::kJavaMultiplier) + 11U) >> 17) & 1U)
            return;
    }
    const auto at = atomicAdd(total, 1ULL);
    if (at < JOB_CAPACITY)
        jobs[at] = {pos, root};
}

struct Emitter {
    uint32_t position;
    uint64_t anchor;
    Match *out;
    unsigned long long *count;
    __device__ void operator()(uint64_t s) {
        auto at = atomicAdd(count, 1ULL);
        if (at < MATCH_CAPACITY)
            out[at] = {position, (s - anchor) & sr::kMask48};
    }
};

using reverse_demo::Prepared;
using reverse_demo::preparedQ;
__global__ void prepareKernel(const Position *positions, const Job *jobs, uint32_t count,
                              Prepared *out) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < count)
        reverse_demo::prepare(jobs[j].root, positions[jobs[j].position].cells, out[j]);
}
enum class SearchStage { Difference, Precomputed, Grouped };

template <SearchStage Mode>
__global__ void searchKernel(const Position *positions, const Job *jobs, uint32_t count, Match *out,
                             unsigned long long *total, const Prepared *prepared) {
    uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr int bits = Mode == SearchStage::Grouped ? 13 : 17;
    if (idx >= (uint64_t(count) << bits))
        return;
    const Job job = jobs[idx >> bits];
    const uint32_t m = idx & ((1U << bits) - 1);
    uint64_t low = job.root + (uint64_t(m) << 18);
    const Position &p = positions[job.position];
    Emitter emit{job.position, p.anchor, out, total};
    if constexpr (Mode == SearchStage::Grouped) {
        const uint32_t base = ((m >> 4) << 8) + (m & 15);
        uint32_t mask = reverse_demo::groupMask(prepared[idx >> 13], base);
        while (mask) {
            const uint32_t j = __ffs(mask) - 1;
            mask &= mask - 1;
            const uint64_t candidate = job.root + (uint64_t(base + 16 * j) << 18);
            sr::solveIntervals(candidate, p.cells, 25, emit);
        }
    } else if constexpr (Mode == SearchStage::Difference) {
        if (sr::differenceFilter(low, p.cells, 25))
            sr::solveIntervals(low, p.cells, 25, emit);
    } else if constexpr (Mode == SearchStage::Precomputed) {
        const Prepared &t = prepared[idx >> 17];
        const uint32_t q0 = preparedQ(t, 0, m);
        for (int i = 1; i < 25; ++i) {
            const uint32_t r = ((preparedQ(t, i, m) - q0) & 0x7fffffffU) % 10;
            if (r != 0 && r != 8)
                return;
        }
        sr::solveIntervals(low, p.cells, 25, emit);
    }
}
struct Device {
    Prepared *prepared = nullptr;
    size_t preparedCapacity = 0;
    Position *positions = nullptr;
    Job *jobs = nullptr;
    Match *matches = nullptr;
    unsigned long long *total = nullptr;
    cudaEvent_t start, stop;
    Device(const std::vector<Position> &p) {
        CUDA(cudaMalloc(&positions, p.size() * sizeof(Position)));
        CUDA(cudaMemcpy(positions, p.data(), p.size() * sizeof(Position), cudaMemcpyHostToDevice));
        CUDA(cudaMalloc(&jobs, JOB_CAPACITY * sizeof(Job)));
        CUDA(cudaMalloc(&matches, MATCH_CAPACITY * sizeof(Match)));
        CUDA(cudaMalloc(&total, sizeof(*total)));
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&stop));
    }
    ~Device() {
        cudaFree(prepared);
        cudaFree(positions);
        cudaFree(jobs);
        cudaFree(matches);
        cudaFree(total);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float finish() {
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(stop));
        CUDA(cudaEventSynchronize(stop));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
    uint64_t totalCount() {
        unsigned long long n;
        CUDA(cudaMemcpy(&n, total, sizeof(n), cudaMemcpyDeviceToHost));
        return n;
    }
    float roots(uint32_t n) {
        CUDA(cudaMemset(total, 0, sizeof(*total)));
        CUDA(cudaEventRecord(start));
        unsigned blocks = unsigned(((uint64_t(n) << 18) + 255) / 256);
        rootKernel<<<blocks, 256>>>(positions, n, jobs, total);
        return finish();
    }
    float search(uint32_t n, SearchStage mode) {
        if (n > preparedCapacity) {
            CUDA(cudaFree(prepared));
            prepared = nullptr;
            CUDA(cudaMalloc(&prepared, n * sizeof(Prepared)));
            preparedCapacity = n;
        }
        CUDA(cudaMemset(total, 0, sizeof(*total)));
        CUDA(cudaEventRecord(start));
        unsigned blocks = unsigned(((uint64_t(n) << 17) + 255) / 256);
        if (blocks) {
            if (mode != SearchStage::Difference)
                prepareKernel<<<(n + 127) / 128, 128>>>(positions, jobs, n, prepared);
            if (mode == SearchStage::Difference)
                searchKernel<SearchStage::Difference>
                    <<<blocks, 256>>>(positions, jobs, n, matches, total, prepared);
            if (mode == SearchStage::Precomputed)
                searchKernel<SearchStage::Precomputed>
                    <<<blocks, 256>>>(positions, jobs, n, matches, total, prepared);
            if (mode == SearchStage::Grouped)
                searchKernel<SearchStage::Grouped>
                    <<<unsigned(((uint64_t(n) << 13) + 255) / 256), 256>>>(
                        positions, jobs, n, matches, total, prepared);
        }
        return finish();
    }
    std::vector<Match> results() {
        uint64_t n = totalCount();
        if (n > MATCH_CAPACITY)
            throw std::runtime_error("match overflow");
        std::vector<Match> r(n);
        CUDA(cudaMemcpy(r.data(), matches, n * sizeof(Match), cudaMemcpyDeviceToHost));
        std::sort(r.begin(), r.end());
        return r;
    }
};
