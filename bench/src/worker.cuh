#pragma once

#include "kernels.cuh"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>

using Clock = std::chrono::steady_clock;
struct Metrics {
    int device;
    std::string name;
    uint64_t positions = 0, classes = 0, cacheHits = 0, jobs = 0, matches = 0;
    double setupMs = 0, scanMs = 0, rootMs = 0, searchMs = 0;
    Clock::time_point start, end;
    std::vector<Match> sample;
};
struct Shared {
    std::atomic<uint64_t> next{0}, done{0};
    std::atomic<int> finished{0};
    int ready = 0, workerCount = 0;
    std::condition_variable cv;
    std::atomic<bool> cancel{false};
    std::mutex mutex;
    std::string error;
};
void worker(const Config &c, int device, Shared &shared, Metrics &result) {
    try {
        auto setup = Clock::now();
        CUDA(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA(cudaGetDeviceProperties(&prop, device));
        result.device = device;
        result.name = prop.name;
        int count = c.pattern.cells.size(), positive = c.pattern.positive;
        size_t rootsCapacity = size_t(c.positionBatch) * (1U << 17);
        Buffer<sr::CellTerm> dcells(size_t(c.positionBatch) * count);
        Buffer<Coord> dcoords(c.positionBatch);
        Buffer<Job> droots(rootsCapacity), djobs(c.jobBatch);
        Buffer<uint32_t> prep(size_t(c.jobBatch) * count * 2);
        Buffer<unsigned long long> total(1);
        Buffer<Match> matches(c.maxResults);
        std::map<uint64_t, std::vector<uint32_t>> cache;
        std::deque<uint64_t> fifo;
        size_t cachedRoots = 0;
        result.setupMs = std::chrono::duration<double, std::milli>(Clock::now() - setup).count();
        {
            std::unique_lock<std::mutex> lock(shared.mutex);
            ++shared.ready;
            shared.cv.notify_all();
            shared.cv.wait(
                lock, [&]() { return shared.ready == shared.workerCount || shared.cancel.load(); });
            if (shared.cancel)
                return;
        }
        result.start = Clock::now();
        const uint64_t end = c.startAnchor + c.anchorCount;
        while (!shared.cancel) {
            uint64_t first = shared.next.fetch_add(c.positionBatch);
            if (first >= end)
                break;
            uint32_t num = std::min<uint64_t>(c.positionBatch, end - first);
            std::vector<Coord> coords;
            std::vector<uint64_t> keys;
            std::map<uint64_t, std::vector<uint32_t>> lists;
            std::vector<Coord> missing;
            std::vector<uint64_t> missingKeys;
            for (uint32_t i = 0; i < num; ++i) {
                Coord p = coordinate(c, first + i);
                coords.push_back(p);
                uint64_t key = classKey(p);
                keys.push_back(key);
                if (lists.count(key)) {
                    ++result.cacheHits;
                    continue;
                }
                auto it = cache.find(key);
                if (it != cache.end()) {
                    lists[key] = it->second;
                    ++result.cacheHits;
                } else {
                    lists[key] = {};
                    missing.push_back(p);
                    missingKeys.push_back(key);
                }
            }
            if (!missing.empty()) {
                auto start = Clock::now();
                std::vector<sr::CellTerm> cells(missing.size() * count);
                for (size_t i = 0; i < missing.size(); ++i)
                    terms(missing[i], c.pattern, cells.data() + i * count);
                CUDA(cudaMemcpy(dcells.p, cells.data(), cells.size() * sizeof(sr::CellTerm),
                                cudaMemcpyHostToDevice));
                CUDA(cudaMemset(total.p, 0, sizeof(*total.p)));
                rootsKernel<<<unsigned((uint64_t(missing.size()) << 18) / 256), 256>>>(
                    dcells.p, missing.size(), count, positive, droots.p, total.p);
                CUDA(cudaGetLastError());
                unsigned long long found;
                CUDA(cudaMemcpy(&found, total.p, sizeof(found), cudaMemcpyDeviceToHost));
                if (found > rootsCapacity)
                    throw std::runtime_error("root capacity invariant failed");
                std::vector<Job> roots(found);
                if (found)
                    CUDA(cudaMemcpy(roots.data(), droots.p, found * sizeof(Job),
                                    cudaMemcpyDeviceToHost));
                for (auto job : roots)
                    lists[missingKeys[job.position]].push_back(job.root);
                result.classes += missing.size();
                for (auto key : missingKeys) {
                    auto &list = lists[key];
                    std::sort(list.begin(), list.end());
                    if (c.cacheClasses) {
                        while (!fifo.empty() && (cache.size() >= c.cacheClasses ||
                                                 cachedRoots + list.size() > (1U << 20))) {
                            auto old = fifo.front();
                            fifo.pop_front();
                            cachedRoots -= cache.at(old).size();
                            cache.erase(old);
                        }
                        cache[key] = list;
                        cachedRoots += list.size();
                        fifo.push_back(key);
                    }
                }
                result.rootMs +=
                    std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            }
            // Construct concrete positions only for classes with roots.
            std::vector<Coord> active;
            std::vector<sr::CellTerm> cells;
            std::vector<Job> jobs;
            jobs.reserve(c.jobBatch);
            for (uint32_t i = 0; i < num; ++i)
                if (!lists.at(keys[i]).empty())
                    active.push_back(coords[i]);
            if (!active.empty()) {
                cells.resize(active.size() * count);
                for (size_t i = 0; i < active.size(); ++i)
                    terms(active[i], c.pattern, cells.data() + i * count);
                CUDA(cudaMemcpy(dcells.p, cells.data(), cells.size() * sizeof(sr::CellTerm),
                                cudaMemcpyHostToDevice));
                CUDA(cudaMemcpy(dcoords.p, active.data(), active.size() * sizeof(Coord),
                                cudaMemcpyHostToDevice));
                auto search = [&]() {
                    if (jobs.empty() || shared.cancel)
                        return;
                    auto start = Clock::now();
                    CUDA(cudaMemcpy(djobs.p, jobs.data(), jobs.size() * sizeof(Job),
                                    cudaMemcpyHostToDevice));
                    CUDA(cudaMemset(total.p, 0, sizeof(*total.p)));
                    prepareKernel<<<unsigned((jobs.size() + 127) / 128), 128>>>(
                        dcells.p, djobs.p, jobs.size(), count, positive, prep.p);
                    searchKernel<<<unsigned((uint64_t(jobs.size()) << 13) / 256), 256>>>(
                        dcells.p, dcoords.p, djobs.p, jobs.size(), count, positive,
                        c.pattern.cells.front().x, c.pattern.cells.front().z, prep.p, matches.p,
                        c.maxResults, total.p);
                    CUDA(cudaGetLastError());
                    unsigned long long found;
                    CUDA(cudaMemcpy(&found, total.p, sizeof(found), cudaMemcpyDeviceToHost));
                    if (found > UINT64_MAX - result.matches)
                        throw std::runtime_error("match count exceeds uint64");
                    result.matches += found;
                    result.jobs += jobs.size();
                    size_t take = std::min<uint64_t>(found, c.maxResults - result.sample.size());
                    if (take) {
                        size_t old = result.sample.size();
                        result.sample.resize(old + take);
                        CUDA(cudaMemcpy(result.sample.data() + old, matches.p, take * sizeof(Match),
                                        cudaMemcpyDeviceToHost));
                    }
                    result.searchMs +=
                        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    jobs.clear();
                };
                for (uint32_t i = 0; i < active.size() && !shared.cancel; ++i)
                    for (uint32_t root : lists.at(classKey(active[i]))) {
                        jobs.push_back({i, root});
                        if (jobs.size() == size_t(c.jobBatch))
                            search();
                    }
                search();
            }
            result.positions += num;
            shared.done.fetch_add(num);
        }
        result.end = Clock::now();
        result.scanMs =
            std::chrono::duration<double, std::milli>(result.end - result.start).count();
    } catch (const std::exception &e) {
        shared.cancel = true;
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.error = "GPU " + std::to_string(device) + ": " + e.what();
        shared.cv.notify_all();
    }
}
