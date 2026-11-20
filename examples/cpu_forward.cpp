#include "common.hpp"
#include <atomic>
#include <thread>

// Original seed -> x -> z -> cells nesting, counting every matching anchor.
std::vector<forward::Hit> scan(const forward::Args &a) {
    std::atomic<uint64_t> next{0}, count{0};
    std::atomic<bool> overflow{false};
    std::vector<std::vector<forward::Hit>> local(a.cpuThreads);
    std::vector<std::thread> workers;
    for (int t = 0; t < a.cpuThreads; ++t)
        workers.emplace_back([&, t] {
            for (;;) {
                const uint64_t s = next.fetch_add(1, std::memory_order_relaxed);
                if (s >= a.seeds)
                    break;
                for (uint64_t xi = 0; xi < a.width; ++xi)
                    for (uint64_t zi = 0; zi < a.height; ++zi) {
                        const int32_t x = int32_t(int64_t(a.x0) + xi),
                                      z = int32_t(int64_t(a.z0) + zi);
                        if (forward::solid(a.seed + s, x, z, a.n)) {
                            if (count.fetch_add(1, std::memory_order_relaxed) < a.capacity)
                                local[t].push_back({a.seed + s, x, z});
                            else
                                overflow.store(true, std::memory_order_relaxed);
                        }
                    }
            }
        });
    for (auto &w : workers)
        w.join();
    if (overflow.load())
        throw std::runtime_error("hit buffer overflow; raise --capacity");
    std::vector<forward::Hit> result;
    for (const auto &v : local)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

int main(int argc, char **argv) {
    try {
        const auto a = forward::parse(argc, argv);
        std::cout.precision(10);
        auto expected = scan(a); // full-workload warmup
        std::sort(expected.begin(), expected.end());
        std::vector<double> times;
        for (int r = 0; r < a.repeats; ++r) {
            const auto begin = std::chrono::steady_clock::now();
            auto hits = scan(a);
            times.push_back(
                forward::ms(begin)); // thread creation + scan + collection, excludes sort
            std::sort(hits.begin(), hits.end());
            if (hits != expected)
                throw std::runtime_error("CPU repeats disagree");
        }
        std::cout << '{';
        forward::workload(a);
        std::cout << ",\"mode\":\"cpu_forward\",\"cpu_threads\":" << a.cpuThreads
                  << ",\"samples_ms\":";
        forward::array(times);
        std::cout << ",\"median_ms\":" << forward::median(times) << ",\"hits\":";
        forward::results(expected);
        std::cout << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
