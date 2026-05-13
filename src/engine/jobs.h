#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Simple worker-thread pool with an enqueue + wait_all model.
//
// Typical usage:
//   js.enqueue([&]{ doWork(0); });
//   js.enqueue([&]{ doWork(1); });
//   js.wait_all();           // blocks until every enqueued job finishes
//
// `parallel_for(N, [](size_t i){ ... })` is a convenience that splits a range
// across all workers and waits for completion.
//
// Threads are created once and stay alive until destruction. No allocations
// happen during enqueue beyond the std::function itself.
class JobSystem {
public:
    explicit JobSystem(uint32_t numThreads = 0);
    ~JobSystem();

    JobSystem(const JobSystem&)            = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    void enqueue(std::function<void()> job);
    void wait_all();

    // Split [0, count) into chunks and run `fn(i)` for each `i` in parallel.
    // Templated so the per-iteration call dispatches DIRECTLY rather than
    // through std::function — in Debug builds (/Od /RTC1) the type-erased
    // version is ~10x slower per call, which dominates when the per-entity
    // body is small (~1 µs). Only the chunk-level enqueue (worker_count
    // dispatches per parallel_for) still goes through std::function, and
    // that overhead is amortized over thousands of indices.
    template <typename F>
    void parallel_for(size_t count, F fn) {
        if (count == 0) return;
        if (m_workers.empty()) {
            for (size_t i = 0; i < count; ++i) fn(i);
            return;
        }
        size_t chunks = std::min<size_t>(count, m_workers.size());
        size_t per    = (count + chunks - 1) / chunks;
        for (size_t c = 0; c < chunks; ++c) {
            size_t begin = c * per;
            size_t end   = std::min(begin + per, count);
            if (begin >= end) continue;
            // Capture fn BY VALUE so each chunk owns a copy; the source may go
            // out of scope before the worker picks the task up. Inside the
            // chunk, fn(i) is a direct call into the lambda's operator() —
            // no virtual dispatch, fully inlinable in Release.
            enqueue([begin, end, fn]() mutable {
                for (size_t i = begin; i < end; ++i) fn(i);
            });
        }
        wait_all();
    }

    uint32_t worker_count() const { return static_cast<uint32_t>(m_workers.size()); }

private:
    void workerLoop();

    std::vector<std::thread>          m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex                        m_mutex;
    std::condition_variable           m_cvTasks;
    std::condition_variable           m_cvDone;
    std::atomic<uint64_t>             m_inFlight{0};
    bool                              m_shutdown = false;
};
