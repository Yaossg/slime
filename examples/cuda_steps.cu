#include "common.hpp"
#include <cuda_runtime.h>
#include <map>

void check(cudaError_t err) {
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

// std::string in host Args must never be copied into a device kernel.
struct DeviceArgs {
    uint64_t seed, seeds, width, height, capacity;
    int32_t x0, z0;
    int n;
};
__device__ void emit(DeviceArgs a, forward::Hit *hits, unsigned long long *count, uint64_t seed,
                     int32_t x, int32_t z) {
    const auto slot = atomicAdd(count, 1ULL);
    if (slot < a.capacity)
        hits[slot] = {seed, x, z};
}

#include "cuda_seed.cuh"
#include "cuda_position.cuh"
#include "cuda_outer_sum.cuh"

struct Buffers {
    forward::Hit *hits = nullptr;
    unsigned long long *count = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    ~Buffers() {
        if (begin)
            cudaEventDestroy(begin);
        if (end)
            cudaEventDestroy(end);
        if (count)
            cudaFree(count);
        if (hits)
            cudaFree(hits);
    }
};
struct Run {
    double kernelMs, scanMs;
    std::vector<forward::Hit> hits;
};

Run run(int mode, const forward::Args &a, int maxBlocks, Buffers &b) {
    const auto wall = std::chrono::steady_clock::now();
    check(cudaMemset(b.count, 0, sizeof(unsigned long long)));
    const DeviceArgs d{a.seed, a.seeds, a.width, a.height, a.capacity, a.x0, a.z0, a.n};
    const int threads = 256;
    const uint64_t work = mode == 2 ? a.seeds : a.positions();
    const int blocks = int(std::min<uint64_t>((work + threads - 1) / threads, maxBlocks));
    check(cudaEventRecord(b.begin));
    switch (mode) {
    case 2:
        seedKernel<<<blocks, threads>>>(d, b.hits, b.count);
        break;
    case 3:
        launchCoordinate(blocks, threads, d, b.hits, b.count);
        break;
    case 4: {
        const uint64_t tiles = (a.width + 15) / 16 * ((a.height + 15) / 16);
        if ((a.height + 15) / 16 <= 65535) {
            const dim3 grid(unsigned((a.width + 15) / 16), unsigned((a.height + 15) / 16));
            outerRectKernel<<<grid, dim3(16, 16)>>>(d, b.hits, b.count);
        } else {
            outerKernel<<<int(std::min<uint64_t>(tiles, 0x7FFFFFFFULL)), dim3(16, 16)>>>(d, b.hits,
                                                                                         b.count);
        }
        break;
    }
    default:
        throw std::runtime_error("mode must be 2..4");
    }
    check(cudaGetLastError());
    check(cudaEventRecord(b.end));
    check(cudaEventSynchronize(b.end));
    float kernel = 0;
    check(cudaEventElapsedTime(&kernel, b.begin, b.end));
    unsigned long long count = 0;
    check(cudaMemcpy(&count, b.count, sizeof(count), cudaMemcpyDeviceToHost));
    if (count > a.capacity)
        throw std::runtime_error("hit buffer overflow; raise --capacity");
    std::vector<forward::Hit> hits(count);
    if (count)
        check(
            cudaMemcpy(hits.data(), b.hits, count * sizeof(forward::Hit), cudaMemcpyDeviceToHost));
    const double wallMs = forward::ms(wall);
    std::sort(hits.begin(), hits.end()); // equality checks are outside timed scan
    return {kernel, wallMs, std::move(hits)};
}

int main(int argc, char **argv) {
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto a = forward::parse(argc, argv);
        std::vector<int> modes;
        if (a.modes == "all")
            modes = {2, 3, 4};
        else {
            for (size_t p = 0; p < a.modes.size();) {
                const size_t end = a.modes.find(',', p);
                const std::string token =
                    a.modes.substr(p, end == std::string::npos ? end : end - p);
                const auto m = forward::positive(token);
                if (m < 2 || m > 4 || std::find(modes.begin(), modes.end(), int(m)) != modes.end())
                    throw std::runtime_error("invalid/duplicate CUDA mode");
                modes.push_back(int(m));
                if (end == std::string::npos)
                    break;
                p = end + 1;
            }
        }
        if (modes.empty())
            throw std::runtime_error("empty mode list");
        int device = 0;
        cudaDeviceProp prop{};
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&prop, device));
        const int maxBlocks = prop.multiProcessorCount * 32;
        Buffers b;
        check(cudaMalloc(&b.hits, a.capacity * sizeof(forward::Hit)));
        check(cudaMalloc(&b.count, sizeof(unsigned long long)));
        check(cudaEventCreate(&b.begin));
        check(cudaEventCreate(&b.end));
        const double setup = forward::ms(start);
        std::vector<forward::Hit> expected;
        bool first = true;
        for (int mode : modes) {
            const auto r = run(mode, a, maxBlocks, b); // warm every kernel on actual workload
            if (first) {
                expected = r.hits;
                first = false;
            } else if (r.hits != expected)
                throw std::runtime_error("CUDA warmups disagree");
        }
        std::map<int, std::vector<double>> kernel, wall;
        for (int repeat = 0; repeat < a.repeats; ++repeat) {
            for (size_t j = 0; j < modes.size(); ++j) {
                const int mode =
                    modes[(j + repeat) % modes.size()]; // rotate order to reduce clock/order bias
                auto r = run(mode, a, maxBlocks, b);
                if (r.hits != expected)
                    throw std::runtime_error("CUDA modes/repeats disagree");
                kernel[mode].push_back(r.kernelMs);
                wall[mode].push_back(r.scanMs);
            }
        }
        std::cout.precision(10);
        std::cout << '{';
        forward::workload(a);
        std::cout << ",\"gpu\":\"" << prop.name << "\",\"setup_ms\":" << setup
                  << ",\"comparison_equal\":true,\"rows\":[";
        for (size_t i = 0; i < modes.size(); ++i) {
            const int m = modes[i];
            if (i)
                std::cout << ',';
            std::cout << "{\"mode\":" << m << ",\"kernel_samples_ms\":";
            forward::array(kernel[m]);
            std::cout << ",\"kernel_median_ms\":" << forward::median(kernel[m])
                      << ",\"scan_samples_ms\":";
            forward::array(wall[m]);
            std::cout << ",\"scan_median_ms\":" << forward::median(wall[m]) << '}';
        }
        std::cout << "],\"hits\":";
        forward::results(expected);
        std::cout << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
