#pragma once

#include "runner.cuh"
#include <iomanip>
#include <iostream>
std::string escape(const std::string &s) {
    std::ostringstream o;
    for (unsigned char ch : s) {
        if (ch == '"' || ch == '\\')
            o << '\\' << char(ch);
        else if (ch < 32)
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        else
            o << char(ch);
    }
    return o.str();
}

std::ostringstream scopeReport(const Config &c, bool patternSet) {
    std::ostringstream report;
    report << std::setprecision(12)
           << "{\"schema_version\":1,\"mode\":\"ordinary_all_effective_seeds\",\"rejection_tail_"
              "included\":false,\"center\":["
           << c.x << "," << c.z << "],\"radius_chunks\":" << c.radius
           << ",\"square_size\":" << (patternSet ? "null" : std::to_string(c.size))
           << ",\"anchors\":" << anchors(c.radius)
           << ",\"effective_seed_bits\":48,\"dry_run\":" << (c.dry ? "true" : "false");
    report << ",\"start_anchor\":" << c.startAnchor << ",\"anchor_count\":" << c.anchorCount;
    report << ",\"pattern_width\":" << c.pattern.width << ",\"pattern_height\":" << c.pattern.height
           << ",\"slime_cells\":" << c.pattern.positive
           << ",\"non_slime_cells\":" << c.pattern.cells.size() - c.pattern.positive
           << ",\"pattern\":[";
    for (size_t i = 0; i < c.pattern.rows.size(); ++i) {
        if (i)
            report << ",";
        report << "\"" << c.pattern.rows[i] << "\"";
    }
    report << "]";

    return report;
}

void searchReport(const Config &c, std::ostringstream &report) {
    auto devices = selectDevices(c);
    report << ",\"position_batch\":" << c.positionBatch << ",\"job_batch\":" << c.jobBatch
           << ",\"cache_classes\":" << c.cacheClasses
           << ",\"cuda_runtime_version\":" << CUDART_VERSION << ",\"rounds\":[";
    std::vector<double> timings;
    uint64_t expected = 0;
    for (int round = 0; round < c.repeats; ++round) {
        auto result = runRound(c, devices, round);
        const auto &metrics = result.metrics;
        const auto &sample = result.sample;
        const auto visited = result.visited, jobs = result.jobs, matches = result.matches;
        const double elapsed = result.elapsed, scan = result.scan;
        if (round && matches != expected)
            throw std::runtime_error("repeat match count mismatch");
        expected = matches;
        timings.push_back(scan);
        if (round)
            report << ",";
        report << "{\"round\":" << round + 1 << ",\"elapsed_ms\":" << elapsed
               << ",\"scan_wall_ms\":" << scan
               << ",\"anchors_per_second\":" << visited / (scan / 1000) << ",\"root_jobs\":" << jobs
               << ",\"matches\":" << matches
               << ",\"sample_truncated\":" << (matches > sample.size() ? "true" : "false")
               << ",\"sample\":[";
        for (size_t i = 0; i < sample.size(); ++i) {
            if (i)
                report << ",";
            auto m = sample[i];
            report << "[" << m.seed << "," << m.x << "," << m.z << "]";
        }
        report << "],\"gpus\":[";
        for (size_t i = 0; i < metrics.size(); ++i) {
            if (i)
                report << ",";
            auto &m = metrics[i];
            report << "{\"device\":" << m.device << ",\"name\":\"" << escape(m.name)
                   << "\",\"anchors\":" << m.positions
                   << ",\"root_classes_generated\":" << m.classes
                   << ",\"cache_hits\":" << m.cacheHits << ",\"root_jobs\":" << m.jobs
                   << ",\"matches\":" << m.matches << ",\"setup_ms\":" << m.setupMs
                   << ",\"scan_ms\":" << m.scanMs << ",\"root_stage_ms\":" << m.rootMs
                   << ",\"search_stage_ms\":" << m.searchMs << "}";
        }
        report << "]}";
        std::cerr << "round " << round + 1 << "/" << c.repeats << ": " << visited << " anchors, "
                  << matches << " matches, scan " << scan << " ms, elapsed " << elapsed << " ms\n";
    }
    std::sort(timings.begin(), timings.end());
    double median = timings[timings.size() / 2];
    if (timings.size() % 2 == 0)
        median = (median + timings[timings.size() / 2 - 1]) / 2;
    report << "],\"median_scan_wall_ms\":" << median << "}";
}
