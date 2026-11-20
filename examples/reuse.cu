#include "reverse_support.cuh"
#include <sstream>

__global__ void expandJobs(const uint32_t *positionClass, const uint32_t *offsets,
                           const uint32_t *classOffsets, const uint32_t *roots, uint32_t n,
                           Job *jobs) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    const uint32_t c = positionClass[p];
    for (uint32_t i = classOffsets[c]; i < classOffsets[c + 1]; ++i)
        jobs[offsets[p] + i - classOffsets[c]] = {p, roots[i]};
}
struct Extra {
    Position *reps;
    uint32_t *positionClass, *offsets, *classOffsets, *roots;
    Extra(size_t n) {
        CUDA(cudaMalloc(&reps, n * sizeof(Position)));
        CUDA(cudaMalloc(&positionClass, n * sizeof(uint32_t)));
        CUDA(cudaMalloc(&offsets, (n + 1) * sizeof(uint32_t)));
        CUDA(cudaMalloc(&classOffsets, (n + 1) * sizeof(uint32_t)));
        CUDA(cudaMalloc(&roots, JOB_CAPACITY * sizeof(uint32_t)));
    }
    ~Extra() {
        cudaFree(reps);
        cudaFree(positionClass);
        cudaFree(offsets);
        cudaFree(classOffsets);
        cudaFree(roots);
    }
};
struct Run {
    double hostMs, buildMs, rootMs, expandMs, searchMs;
    uint32_t classes, jobs, activePositions;
    std::vector<Match> matches;
    std::vector<Job> rootList;
};
Run run(const std::vector<std::pair<int32_t, int32_t>> &coords, Device &d, Extra &extra, int mode) {
    const auto start = Clock::now();
    auto ms = [](auto t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    std::vector<Position> positions;
    positions.reserve(coords.size());
    if (mode != 3)
        for (auto [x, z] : coords)
            positions.push_back(makePosition(x, z));
    std::vector<Position> reps;
    std::vector<uint32_t> pc;
    if (mode >= 2) {
        std::map<uint64_t, uint32_t> map;
        for (auto [x, z] : coords) {
            uint64_t key = (uint64_t(uint32_t(x) & 65535) << 17) | (uint32_t(z) & 131071);
            auto [it, inserted] = map.emplace(key, uint32_t(reps.size()));
            if (inserted)
                reps.push_back(makePosition(x, z));
            pc.push_back(it->second);
        }
    }
    if (!positions.empty())
        CUDA(cudaMemcpy(d.positions, positions.data(), positions.size() * sizeof(Position),
                        cudaMemcpyHostToDevice));
    Run r{};
    r.classes = mode >= 2 ? reps.size() : positions.size();
    r.buildMs = ms(start);
    const auto rootStart = Clock::now();
    if (mode >= 2) {
        CUDA(cudaMemcpy(extra.reps, reps.data(), reps.size() * sizeof(Position),
                        cudaMemcpyHostToDevice));
        CUDA(cudaMemset(d.total, 0, sizeof(*d.total)));
        rootKernel<<<unsigned(((uint64_t(reps.size()) << 18) + 255) / 256), 256>>>(
            extra.reps, reps.size(), d.jobs, d.total);
    } else
        d.roots(positions.size());
    uint64_t n = d.totalCount();
    if (n > JOB_CAPACITY)
        throw std::runtime_error("root capacity exceeded");
    r.rootMs = ms(rootStart);
    const auto expandStart = Clock::now();
    std::vector<uint32_t> originalIndices;
    if (mode >= 2) {
        std::vector<Job> cj(n);
        if (n)
            CUDA(cudaMemcpy(cj.data(), d.jobs, n * sizeof(Job), cudaMemcpyDeviceToHost));
        std::vector<std::vector<uint32_t>> lists(reps.size());
        for (auto j : cj)
            lists[j.position].push_back(j.root);
        if (mode == 3) {
            std::vector<uint32_t> compactClasses;
            for (uint32_t i = 0; i < coords.size(); ++i)
                if (!lists[pc[i]].empty()) {
                    auto [x, z] = coords[i];
                    positions.push_back(makePosition(x, z));
                    compactClasses.push_back(pc[i]);
                    originalIndices.push_back(i);
                }
            pc = std::move(compactClasses);
            if (!positions.empty())
                CUDA(cudaMemcpy(d.positions, positions.data(), positions.size() * sizeof(Position),
                                cudaMemcpyHostToDevice));
        }
        std::vector<uint32_t> roots, co{0}, offsets{0};
        for (auto &list : lists) {
            std::sort(list.begin(), list.end());
            roots.insert(roots.end(), list.begin(), list.end());
            co.push_back(roots.size());
        }
        uint64_t expanded = 0;
        for (auto c : pc) {
            expanded += lists[c].size();
            if (expanded > JOB_CAPACITY)
                throw std::runtime_error("expanded capacity exceeded; use smaller batch");
            offsets.push_back(expanded);
        }
        n = expanded;
        CUDA(cudaMemcpy(extra.positionClass, pc.data(), pc.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(extra.offsets, offsets.data(), offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(extra.classOffsets, co.data(), co.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
        if (!roots.empty())
            CUDA(cudaMemcpy(extra.roots, roots.data(), roots.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
        if (!positions.empty())
            expandJobs<<<unsigned((positions.size() + 255) / 256), 256>>>(
                extra.positionClass, extra.offsets, extra.classOffsets, extra.roots,
                positions.size(), d.jobs);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
    }
    r.expandMs = ms(expandStart);
    r.jobs = n;
    r.activePositions = positions.size();
    const auto searchStart = Clock::now();
    d.search(n, SearchStage::Grouped);
    r.matches = d.results();
    if (mode == 3)
        for (auto &match : r.matches)
            match.position = originalIndices[match.position];
    r.searchMs = ms(searchStart);
    r.hostMs = ms(start);
    // Validation-only transfers/sorting are outside reported pipeline time.
    r.rootList.resize(n);
    if (n)
        CUDA(cudaMemcpy(r.rootList.data(), d.jobs, n * sizeof(Job), cudaMemcpyDeviceToHost));
    if (mode == 3)
        for (auto &job : r.rootList)
            job.position = originalIndices[job.position];
    std::sort(r.rootList.begin(), r.rootList.end(), [](Job a, Job b) {
        return std::tie(a.position, a.root) < std::tie(b.position, b.root);
    });
    return r;
}
int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error("usage: reuse positions.txt [repeats=5]");
        const auto process = Clock::now();
        std::ifstream in(argv[1]);
        if (!in)
            throw std::runtime_error("cannot open input");
        std::vector<std::pair<int32_t, int32_t>> coords;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream row(line);
            row >> std::ws;
            if (row.eof())
                continue;
            int64_t x, z;
            std::string extra;
            if (!(row >> x >> z) || (row >> extra))
                throw std::runtime_error("expected x z on each input line");
            if (x < INT32_MIN || z < INT32_MIN || x > INT32_MAX - 4LL || z > INT32_MAX - 4LL)
                throw std::runtime_error("coordinates including 5x5 window must fit int32");
            coords.emplace_back(int32_t(x), int32_t(z));
            if (coords.size() > 65536)
                throw std::runtime_error("expected 1..65536 coordinates");
        }
        if (in.bad())
            throw std::runtime_error("failed reading coordinates");
        if (coords.empty() || coords.size() > 65536)
            throw std::runtime_error("expected 1..65536 coordinates");
        int repeats = argc > 2 ? std::stoi(argv[2]) : 5;
        if (repeats < 1)
            throw std::runtime_error("invalid repeats");
        Device d(std::vector<Position>(coords.size()));
        Extra e(coords.size());
        std::cout << std::setprecision(10);
        std::cout << "{\"type\":\"setup\",\"host_ms\":"
                  << std::chrono::duration<double, std::milli>(Clock::now() - process).count()
                  << "}" << std::endl;
        Run ref;
        for (int round = -1; round < repeats; ++round)
            for (int k = 0; k < 3; ++k) {
                int mode = 1 + (round < 0 ? k : (round + k) % 3);
                Run r = run(coords, d, e, mode);
                if (round == -1 && mode == 1)
                    ref = r;
                else {
                    if (r.matches != ref.matches || r.rootList.size() != ref.rootList.size())
                        throw std::runtime_error("pipeline mismatch");
                    for (size_t i = 0; i < r.rootList.size(); ++i)
                        if (r.rootList[i].position != ref.rootList[i].position ||
                            r.rootList[i].root != ref.rootList[i].root)
                            throw std::runtime_error("complete root list mismatch");
                }
                if (round >= 0)
                    std::cout << "{\"type\":\"pipeline\",\"mode\":" << mode
                              << ",\"round\":" << round << ",\"positions\":" << coords.size()
                              << ",\"classes\":" << r.classes
                              << ",\"active_positions\":" << r.activePositions
                              << ",\"jobs\":" << r.jobs << ",\"matches\":" << r.matches.size()
                              << ",\"host_ms\":" << r.hostMs << ",\"build_ms\":" << r.buildMs
                              << ",\"root_ms\":" << r.rootMs << ",\"expand_ms\":" << r.expandMs
                              << ",\"search_ms\":" << r.searchMs << "}" << std::endl;
            }
        std::cout << "{\"type\":\"validation\",\"complete_roots_equal\":true,\"complete_matches_"
                     "equal\":true,\"matches\":[";
        for (size_t i = 0; i < ref.matches.size(); ++i) {
            if (i)
                std::cout << ",";
            auto m = ref.matches[i];
            std::cout << "[" << m.seed << "," << coords[m.position].first << ","
                      << coords[m.position].second << "]";
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
