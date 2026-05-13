#pragma once

// Per-frame GPU + CPU performance profiler.
//
// GPU side: VkQueryPool of timestamps per frame-in-flight. gpuBegin/gpuEnd wrap
// recorded passes; results are read back at the start of the NEXT use of that
// frame slot (one-frame lag, paid for by the existing in-flight fence).
//
// CPU side: scoped RAII timer based on std::chrono::steady_clock. Push/pop
// produces a flat list of named scopes per frame.
//
// Both APIs are non-nesting at the scope level — call end() before begin()ing a
// new scope on the same channel. The profiler asserts on misuse in debug.

#include <vulkan/vulkan.h>
#include <chrono>
#include <vector>
#include <cstdint>

class Profiler {
public:
    struct Entry {
        const char* name = nullptr;
        double      ms   = 0.0;
    };

    void init(VkDevice device, VkPhysicalDevice physical, int framesInFlight);
    void shutdown(VkDevice device);

    // Call once per frame, AFTER vkWaitForFences for `frame` (so the previous
    // submission's timestamps are guaranteed ready) and AFTER vkBeginCommandBuffer
    // for the new submission (so the query-pool reset can be recorded).
    void beginFrame(VkDevice device, VkCommandBuffer cmd, uint32_t frame);

    // Call once per frame, AFTER the command buffer is fully recorded (the
    // submission itself isn't required to have happened yet — endFrame just
    // marks the frame slot as valid for next-time readback).
    void endFrame();

    // Wrap a GPU pass. Non-nested. If too many scopes were already opened this
    // frame, the call is a no-op (logged once).
    void gpuBegin(VkCommandBuffer cmd, const char* name);
    void gpuEnd(VkCommandBuffer cmd);

    // RAII CPU scope. Use via auto _ = profiler.cpuScope("name");
    struct CpuGuard {
        Profiler* prof;
        const char* name;
        std::chrono::steady_clock::time_point t0;

        CpuGuard(Profiler* p, const char* n)
            : prof(p), name(n), t0(std::chrono::steady_clock::now()) {}
        ~CpuGuard() {
            if (!prof) return;
            using namespace std::chrono;
            double ms = duration<double, std::milli>(steady_clock::now() - t0).count();
            prof->recordCpu(name, ms);
        }
        CpuGuard(const CpuGuard&) = delete;
        CpuGuard& operator=(const CpuGuard&) = delete;
        CpuGuard(CpuGuard&& o) noexcept : prof(o.prof), name(o.name), t0(o.t0) {
            o.prof = nullptr; // moved-from guard does nothing on destruction
        }
        CpuGuard& operator=(CpuGuard&&) = delete;
    };
    [[nodiscard]] CpuGuard cpuScope(const char* name) {
        return CpuGuard(this, name);
    }
    void recordCpu(const char* name, double ms);

    // Display data — valid after endFrame().
    const std::vector<Entry>& gpuResults() const { return m_gpuDisplay; }
    const std::vector<Entry>& cpuResults() const { return m_cpuDisplay; }

    bool gpuAvailable() const { return m_gpuAvailable; }

private:
    static constexpr int    MAX_GPU_SCOPES = 16;  // → 32 timestamps per frame
    static constexpr int    MAX_CPU_SCOPES = 32;

    struct GpuFrame {
        VkQueryPool             pool      = VK_NULL_HANDLE;
        std::vector<const char*> names;            // one per opened scope
        int                     nextSlot  = 0;     // next timestamp slot to write
        int                     openSlot  = -1;    // index of currently-open scope, -1 if none
        bool                    submitted = false; // has this slot been recorded at least once
    };

    bool                     m_gpuAvailable = false;
    int                      m_framesInFlight = 0;
    uint32_t                 m_currentFrame   = 0;
    double                   m_timestampPeriodNs = 1.0; // ns per tick
    std::vector<GpuFrame>    m_gpuFrames;

    std::vector<Entry>       m_gpuDisplay;
    std::vector<Entry>       m_cpuLive;
    std::vector<Entry>       m_cpuDisplay;
};

// Macro: concatenates __LINE__ to give a unique guard name.
#define PROFILER_CONCAT_(a, b) a##b
#define PROFILER_CONCAT(a, b) PROFILER_CONCAT_(a, b)
#define PROFILE_CPU(prof, name) \
    auto PROFILER_CONCAT(_cpuGuard_, __LINE__) = (prof).cpuScope(name)
