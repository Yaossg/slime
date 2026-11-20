#include "reverse_support.cuh"

__device__ bool firstDrawPattern(uint64_t seed, const sr::CellTerm *cells) {
    for (int i = 0; i < 25; ++i)
        if ((sr::nextStateFromTerm(seed, cells[i].term) >> 17) % 10 != 0)
            return false;
    return sr::verifyPattern(seed, cells, 25);
}

// Low18 roots are already known; enumerate the remaining 30 bits.
__global__ void enumerateRemaining(const Position *positions, const Job *jobs, uint32_t count,
                                   Match *out, unsigned long long *total) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < (uint64_t(count) << 30); idx += stride) {
        const Job job = jobs[idx >> 30];
        const Position &p = positions[job.position];
        const uint64_t seed = job.root + ((idx & ((1ULL << 30) - 1)) << 18);
        if (firstDrawPattern(seed, p.cells)) {
            Emitter emit{job.position, p.anchor, out, total};
            emit(seed);
        }
    }
}

__global__ void intervalsWithoutFilter(const Position *positions, const Job *jobs, uint32_t count,
                                       Match *out, unsigned long long *total) {
    const uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t(count) << 17))
        return;
    const Job job = jobs[idx >> 17];
    const Position &p = positions[job.position];
    const uint64_t low = job.root + ((idx & ((1U << 17) - 1)) << 18);
    Emitter emit{job.position, p.anchor, out, total};
    sr::solveIntervals(low, p.cells, 25, emit);
}

int main(int argc, char **argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 21;
        if (repeats < 1)
            throw std::runtime_error("positive repeats required");
        Device d({makePosition(51069, 419124)});
        d.roots(1);
        const uint64_t roots = d.totalCount();
        if (roots != 84)
            throw std::runtime_error("expected 84 roots");
        d.search(roots, SearchStage::Grouped);
        const auto expected = d.results();
        if (expected.size() != 24)
            throw std::runtime_error("expected 24 seeds");
        const char *labels[] = {"low18_enumerate30", "intervals_only", "difference_intervals",
                                "precompute_intervals", "group_intervals"};
        std::vector<double> samples[5];
        for (int round = -1; round < repeats; ++round)
            for (int j = 0; j < 5; ++j) {
                const int mode = round < 0 ? j : (round + j) % 5;
                float ms;
                if (mode == 0) {
                    cudaDeviceProp prop{};
                    CUDA(cudaGetDeviceProperties(&prop, 0));
                    CUDA(cudaMemset(d.total, 0, sizeof(*d.total)));
                    CUDA(cudaEventRecord(d.start));
                    enumerateRemaining<<<prop.multiProcessorCount * 32, 256>>>(
                        d.positions, d.jobs, roots, d.matches, d.total);
                    ms = d.finish();
                } else if (mode == 1) {
                    CUDA(cudaMemset(d.total, 0, sizeof(*d.total)));
                    CUDA(cudaEventRecord(d.start));
                    intervalsWithoutFilter<<<unsigned(((roots << 17) + 255) / 256), 256>>>(
                        d.positions, d.jobs, roots, d.matches, d.total);
                    ms = d.finish();
                } else
                    ms = d.search(roots, mode == 2   ? SearchStage::Difference
                                         : mode == 3 ? SearchStage::Precomputed
                                                     : SearchStage::Grouped);
                if (d.results() != expected)
                    throw std::runtime_error("full result list mismatch");
                if (round >= 0)
                    samples[mode].push_back(ms);
            }
        std::cout << std::setprecision(10)
                  << "{\"anchor\":[51069,419124],\"roots\":84,\"lower35\":11010048,\"comparison_"
                     "equal\":true,\"rows\":[";
        for (int mode = 0; mode < 5; ++mode) {
            if (mode)
                std::cout << ",";
            auto sorted = samples[mode];
            std::sort(sorted.begin(), sorted.end());
            double median = sorted[sorted.size() / 2];
            if (sorted.size() % 2 == 0)
                median = (median + sorted[sorted.size() / 2 - 1]) / 2;
            std::cout << "{\"mode\":\"" << labels[mode] << "\",\"median_ms\":" << median
                      << ",\"samples_ms\":[";
            for (size_t i = 0; i < samples[mode].size(); ++i) {
                if (i)
                    std::cout << ",";
                std::cout << samples[mode][i];
            }
            std::cout << "]}";
        }
        std::cout << "],\"matches\":[";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i)
                std::cout << ",";
            std::cout << expected[i].seed;
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
