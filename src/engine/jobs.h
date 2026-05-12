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
    void parallel_for(size_t count, std::function<void(size_t)> fn);

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
