#include "jobs.h"

#include <algorithm>

JobSystem::JobSystem(uint32_t numThreads) {
    if (numThreads == 0) {
        unsigned hw = std::thread::hardware_concurrency();
        numThreads = (hw > 1) ? (hw - 1) : 1;
    }
    m_workers.reserve(numThreads);
    for (uint32_t i = 0; i < numThreads; ++i) {
        m_workers.emplace_back(&JobSystem::workerLoop, this);
    }
}

JobSystem::~JobSystem() {
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_shutdown = true;
    }
    m_cvTasks.notify_all();
    for (auto& t : m_workers) if (t.joinable()) t.join();
}

void JobSystem::enqueue(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_tasks.push(std::move(job));
        m_inFlight.fetch_add(1, std::memory_order_relaxed);
    }
    m_cvTasks.notify_one();
}

void JobSystem::wait_all() {
    std::unique_lock<std::mutex> lk(m_mutex);
    m_cvDone.wait(lk, [&] {
        return m_inFlight.load(std::memory_order_acquire) == 0;
    });
}

void JobSystem::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_cvTasks.wait(lk, [&] { return m_shutdown || !m_tasks.empty(); });
            if (m_shutdown && m_tasks.empty()) return;
            task = std::move(m_tasks.front());
            m_tasks.pop();
        }
        task();
        if (m_inFlight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_cvDone.notify_all();
        }
    }
}
