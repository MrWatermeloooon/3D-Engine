#include "profiler.h"
#include "../utils/vk_check.h"

#include <algorithm>
#include <iostream>

void Profiler::init(VkDevice device, VkPhysicalDevice physical, int framesInFlight) {
    m_framesInFlight = framesInFlight;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);

    // timestampPeriod == 0 means "timestamps not supported on this queue".
    // RTX cards report a non-zero period (typically 1.0). Bail gracefully if
    // we hit an exotic queue family that lacks timestamp support.
    if (props.limits.timestampPeriod == 0.0f) {
        m_gpuAvailable = false;
        std::cerr << "[Profiler] timestampPeriod == 0 — GPU timing disabled.\n";
    } else {
        m_gpuAvailable      = true;
        m_timestampPeriodNs = static_cast<double>(props.limits.timestampPeriod);
    }

    m_gpuFrames.resize(framesInFlight);
    if (m_gpuAvailable) {
        for (auto& gf : m_gpuFrames) {
            VkQueryPoolCreateInfo ci{};
            ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            ci.queryCount = MAX_GPU_SCOPES * 2;
            VK_CHECK(vkCreateQueryPool(device, &ci, nullptr, &gf.pool));
        }
    }

    m_cpuLive.reserve(MAX_CPU_SCOPES);
    m_cpuDisplay.reserve(MAX_CPU_SCOPES);
    m_gpuDisplay.reserve(MAX_GPU_SCOPES);
}

void Profiler::shutdown(VkDevice device) {
    for (auto& gf : m_gpuFrames) {
        if (gf.pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, gf.pool, nullptr);
            gf.pool = VK_NULL_HANDLE;
        }
    }
    m_gpuFrames.clear();
}

void Profiler::beginFrame(VkDevice device, VkCommandBuffer cmd, uint32_t frame) {
    m_currentFrame = frame;
    if (!m_gpuAvailable) {
        m_gpuDisplay.clear();
        return;
    }

    auto& gf = m_gpuFrames[frame];

    // Read back results from the previous use of this slot. vkWaitForFences
    // for this frame's in-flight fence was called by drawFrame just above; the
    // submission that wrote into this pool is therefore complete.
    if (gf.submitted && gf.nextSlot > 0) {
        uint64_t ts[MAX_GPU_SCOPES * 2];
        VkResult r = vkGetQueryPoolResults(device, gf.pool, 0,
            static_cast<uint32_t>(gf.nextSlot),
            sizeof(uint64_t) * gf.nextSlot, ts,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);

        if (r == VK_SUCCESS) {
            m_gpuDisplay.clear();
            int pairs = gf.nextSlot / 2;
            int validNames = std::min<int>(pairs, static_cast<int>(gf.names.size()));
            for (int i = 0; i < validNames; ++i) {
                uint64_t diff = ts[2*i + 1] - ts[2*i];
                double   ms   = static_cast<double>(diff) * m_timestampPeriodNs * 1e-6;
                m_gpuDisplay.push_back({ gf.names[i], ms });
            }
        }
        // On VK_NOT_READY (shouldn't happen after the fence wait) or error,
        // just keep the previous display until the next valid readback.
    }

    // Reset for the new submission. Reset BEFORE any new writes — it has to
    // be inside the command buffer (host-side vkResetQueryPool would race
    // with reuse for frames still in flight on other slots).
    vkCmdResetQueryPool(cmd, gf.pool, 0, MAX_GPU_SCOPES * 2);

    gf.names.clear();
    gf.nextSlot  = 0;
    gf.openSlot  = -1;
    gf.submitted = false;

    // Note: do NOT clear m_cpuLive here. CPU scopes run throughout the frame,
    // including BEFORE drawFrame (scene.update, updateLightBuffer,
    // computeCascades in engine.cpp). beginFrame fires inside drawFrame —
    // clearing here would wipe everything that ran before. The accumulator
    // is reset in endFrame after the display swap instead.
}

void Profiler::endFrame() {
    if (m_gpuAvailable) {
        auto& gf = m_gpuFrames[m_currentFrame];
        gf.submitted = (gf.nextSlot > 0);
    }
    // Swap CPU live → display so the UI sees a coherent snapshot of this
    // frame's CPU scopes, then reset live for the next frame.
    m_cpuDisplay = m_cpuLive;
    m_cpuLive.clear();
}

void Profiler::gpuBegin(VkCommandBuffer cmd, const char* name) {
    if (!m_gpuAvailable) return;

    auto& gf = m_gpuFrames[m_currentFrame];
    if (gf.openSlot >= 0) {
        // Misuse: previous gpuBegin not matched with gpuEnd. Skip this scope.
        return;
    }
    if (gf.nextSlot + 2 > MAX_GPU_SCOPES * 2) {
        // Exceeded budget — drop this scope silently. Bumping MAX_GPU_SCOPES
        // is the fix if this trips for real workloads.
        return;
    }

    gf.openSlot = gf.nextSlot;
    gf.names.push_back(name);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gf.pool, gf.nextSlot);
    gf.nextSlot++;
}

void Profiler::gpuEnd(VkCommandBuffer cmd) {
    if (!m_gpuAvailable) return;

    auto& gf = m_gpuFrames[m_currentFrame];
    if (gf.openSlot < 0) return; // unmatched end; ignore

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, gf.pool, gf.nextSlot);
    gf.nextSlot++;
    gf.openSlot = -1;
}

void Profiler::recordCpu(const char* name, double ms) {
    if (static_cast<int>(m_cpuLive.size()) >= MAX_CPU_SCOPES) return;
    m_cpuLive.push_back({ name, ms });
}
