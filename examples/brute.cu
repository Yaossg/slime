#include "../include/slime/solver_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace sr = slime;

namespace {

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t error_ = (call);                                                         \
        if (error_ != cudaSuccess) {                                                               \
            std::cerr << #call << ": " << cudaGetErrorString(error_) << '\n';                      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

struct Args {
    int32_t x = 51069;
    int32_t z = 419124;
    int32_t size = 5;
    uint64_t start = 0;
    uint64_t count = 1ULL << 32;
    int threads = 256;
    int blocksPerSm = 16;
};

const char *value(int argc, char **argv, int &i) {
    if (++i >= argc)
        std::exit(2);
    return argv[i];
}

Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--x")
            args.x = std::stoi(value(argc, argv, i));
        else if (arg == "--z")
            args.z = std::stoi(value(argc, argv, i));
        else if (arg == "-n")
            args.size = std::stoi(value(argc, argv, i));
        else if (arg == "--start")
            args.start = std::stoull(value(argc, argv, i));
        else if (arg == "--count")
            args.count = std::stoull(value(argc, argv, i));
        else if (arg == "--threads")
            args.threads = std::stoi(value(argc, argv, i));
        else if (arg == "--blocks-per-sm")
            args.blocksPerSm = std::stoi(value(argc, argv, i));
        else {
            std::cerr << "Usage: brute [--x X --z Z -n N] "
                         "[--start N --count N --threads N --blocks-per-sm N]\n";
            std::exit(arg == "--help" ? 0 : 2);
        }
    }
    if (args.size <= 0 || int64_t(args.size) * args.size > sr::kMaxPatternCells ||
        args.count == 0 || args.threads <= 0 || args.threads > 1024 || args.blocksPerSm <= 0 ||
        int64_t(args.x) + args.size - 1 > INT32_MAX ||
        int64_t(args.z) + args.size - 1 > INT32_MAX) {
        std::exit(2);
    }
    return args;
}

__global__ void bruteForceKernel(uint64_t start, uint64_t count, const sr::CellTerm *cells,
                                 int cellCount, unsigned long long *matches) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < count;
         offset += stride) {
        if (sr::verifyPattern((start + offset) & sr::kMask48, cells, cellCount)) {
            atomicAdd(matches, 1ULL);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    const Args args = parseArgs(argc, argv);
    std::vector<sr::CellTerm> terms;
    for (int32_t dz = 0; dz < args.size; ++dz) {
        for (int32_t dx = 0; dx < args.size; ++dx) {
            terms.push_back({sr::coordinateTerm(args.x + dx, args.z + dz), sr::kSlime});
        }
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int blocks = properties.multiProcessorCount * args.blocksPerSm;

    sr::CellTerm *deviceTerms = nullptr;
    unsigned long long *deviceMatches = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceTerms, terms.size() * sizeof(sr::CellTerm)));
    CUDA_CHECK(cudaMalloc(&deviceMatches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(deviceTerms, terms.data(), terms.size() * sizeof(sr::CellTerm),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceMatches, 0, sizeof(unsigned long long)));

    // Warm up the context and the exact same kernel before timing.
    bruteForceKernel<<<blocks, args.threads>>>(args.start, 1U << 20, deviceTerms, int(terms.size()),
                                               deviceMatches);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(deviceMatches, 0, sizeof(unsigned long long)));

    cudaEvent_t begin;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin));
    bruteForceKernel<<<blocks, args.threads>>>(args.start, args.count, deviceTerms,
                                               int(terms.size()), deviceMatches);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float milliseconds = 0.0F;
    unsigned long long matches = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    CUDA_CHECK(cudaMemcpy(&matches, deviceMatches, sizeof(matches), cudaMemcpyDeviceToHost));
    const double seconds = milliseconds / 1000.0;
    const double seedsPerSecond = double(args.count) / seconds;
    const double fullSeconds = double(1ULL << 48) / seedsPerSecond;
    std::cout << "gpu=\"" << properties.name << "\" count=" << args.count << " matches=" << matches
              << " milliseconds=" << milliseconds << " seeds_per_second=" << seedsPerSecond
              << " projected_full48_seconds=" << fullSeconds << '\n';

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(deviceTerms);
    cudaFree(deviceMatches);
}
