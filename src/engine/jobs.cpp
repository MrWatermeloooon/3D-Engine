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

void JobSystem::parallel_for(size_t count, std::function<void(size_t)> fn) {
    if (count == 0) return;
    if (m_workers.empty()) {
        for (size_t i = 0; i < count; ++i) fn(i);
        return;
    }
    // Split into ~one chunk per worker. Each chunk runs `fn` for its index range.
    size_t chunks = std::min<size_t>(count, m_workers.size());
    size_t per    = (count + chunks - 1) / chunks;
    for (size_t c = 0; c < chunks; ++c) {
        size_t begin = c * per;
        size_t end   = std::min(begin + per, count);
        if (begin >= end) continue;
        enqueue([begin, end, fn]() {
            for (size_t i = begin; i < end; ++i) fn(i);
        });
    }
    wait_all();
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
