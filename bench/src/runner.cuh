#pragma once

#include "worker.cuh"
#include <iostream>
#include <set>
#include <thread>

std::vector<int> selectDevices(const Config &c) {
    int available;
    CUDA(cudaGetDeviceCount(&available));
    std::vector<int> devices;
    if (c.devices == "all") {
        for (int i = 0; i < available; ++i)
            devices.push_back(i);
    } else {
        std::stringstream ss(c.devices);
        std::string v;
        while (std::getline(ss, v, ',')) {
            auto id = number(v);
            if (id >= uint64_t(available))
                throw std::runtime_error("CUDA device unavailable: " + v);
            devices.push_back(id);
        }
    }
    if (devices.empty() || std::set<int>(devices.begin(), devices.end()).size() != devices.size())
        throw std::runtime_error("device list must be nonempty and unique");
    return devices;
}

struct RoundResult {
    std::vector<Metrics> metrics;
    uint64_t visited = 0, jobs = 0, matches = 0;
    std::vector<Match> sample;
    double elapsed = 0, scan = 0;
};

RoundResult runRound(const Config &c, const std::vector<int> &devices, int round) {
    RoundResult result;
    Shared shared;
    shared.next = c.startAnchor;
    shared.workerCount = devices.size();
    result.metrics.resize(devices.size());
    auto &metrics = result.metrics;
    std::vector<std::thread> workers;
    auto start = Clock::now();
    for (size_t i = 0; i < devices.size(); ++i)
        workers.emplace_back([&, i]() {
            worker(c, devices[i], shared, metrics[i]);
            ++shared.finished;
            shared.cv.notify_all();
        });
    {
        std::unique_lock<std::mutex> lock(shared.mutex);
        while (shared.finished < int(devices.size())) {
            if (!shared.cv.wait_for(lock, std::chrono::seconds(5),
                                    [&]() { return shared.finished == int(devices.size()); }))
                std::cerr << "progress: " << shared.done << "/" << c.anchorCount
                          << " anchors (round " << round + 1 << ")\n";
        }
    }
    for (auto &t : workers)
        t.join();
    if (!shared.error.empty())
        throw std::runtime_error(shared.error);
    result.elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto first = metrics[0].start, last = metrics[0].end;

    for (auto &m : metrics) {
        first = std::min(first, m.start);
        last = std::max(last, m.end);
        result.visited += m.positions;
        result.jobs += m.jobs;
        if (m.matches > UINT64_MAX - result.matches)
            throw std::runtime_error("total match count exceeds uint64");
        result.matches += m.matches;
        for (auto hit : m.sample)
            if (result.sample.size() < c.maxResults)
                result.sample.push_back(hit);
    }
    if (result.visited != c.anchorCount)
        throw std::runtime_error("incomplete anchor coverage");
    result.scan = std::chrono::duration<double, std::milli>(last - first).count();
    return result;
}
