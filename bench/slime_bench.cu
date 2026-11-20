#include "src/cli.hpp"
#include "src/report.cuh"

int main(int argc, char **argv) {
    try {
        Config c;
        bool patternSet = false;
        if (!parseArguments(argc, argv, c, patternSet))
            return 0;
        if (c.list) {
            int n;
            CUDA(cudaGetDeviceCount(&n));
            for (int i = 0; i < n; ++i) {
                cudaDeviceProp p{};
                CUDA(cudaGetDeviceProperties(&p, i));
                std::cout << i << "\t" << p.name << "\tsm_" << p.major << p.minor << "\t"
                          << p.totalGlobalMem / (1ULL << 20) << " MiB\n";
            }
            return 0;
        }
        auto report = scopeReport(c, patternSet);
        if (c.dry) {
            auto first = coordinate(c, c.startAnchor),
                 last = coordinate(c, c.startAnchor + c.anchorCount - 1);
            report << ",\"first_anchor\":[" << first.x << "," << first.z << "],\"last_anchor\":["
                   << last.x << "," << last.z << "]}";
        } else {
            searchReport(c, report);
        }
        if (!c.output.empty()) {
            std::ofstream file(c.output);
            if (!file)
                throw std::runtime_error("cannot write report: " + c.output);
            file << report.str() << "\n";
            file.close();
            if (!file)
                throw std::runtime_error("failed writing report");
        }
        std::cout << report.str() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
