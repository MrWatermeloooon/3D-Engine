#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

#include "instancing.h"
#include "frustum.h"
#include "buffer.h"

// GPU-driven culling: frustum + LOD selection + (Phase 3) occlusion via HZB.
//
// Per frame the CPU writes:
//   * CandidateInstance per renderable entity
//   * BatchHeader per (LODGroup, texture) batch
//   * CullParamsUBO (camera frustum, position, prev view-proj, HZB metadata,
//     numCandidates)
// Plus pre-authored indirect commands (instanceCount = 0). The compute shader
// frustum-tests, LOD-picks, HZB-occlusion-tests, and atomicAdds into the
// appropriate draw command(s).

struct GpuCullData {
    VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline       = VK_NULL_HANDLE;

    VkDescriptorPool             descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;          // one per frame in flight

    // Per-frame params UBO. Lives here so the cull module owns its own data.
    std::vector<AllocatedBuffer> paramsBuffers;
    std::vector<void*>           paramsMapped;
};

// CullParamsUBO — must match cull.comp's CullParams layout (std140).
struct CullParamsUBO {
    glm::mat4 prevViewProj;        // 64 — for HZB-space projection
    glm::vec4 frustumPlanes[6];    // 96
    glm::vec4 cameraPos;           // 16 — .xyz used for LOD distance
    glm::vec4 hzbSizeMipCount;     // .xy = hzb extent, .z = mip count, .w = enable flag
    uint32_t  numCandidates;
    uint32_t  _pad0, _pad1, _pad2;
};
static_assert(sizeof(CullParamsUBO) == 208, "CullParamsUBO must match shader layout");

void createGpuCull(GpuCullData& cull, VkPhysicalDevice physicalDevice, VkDevice device,
                   uint32_t framesInFlight, const std::string& compShaderPath);

// Bind buffers (SSBOs) and the HZB sampler to each per-frame descriptor set.
// Called once at init, and again on swapchain rebuild because the HZB image
// (and possibly buffer identities) change.
void writeGpuCullDescriptors(GpuCullData& cull, VkDevice device,
                             const CandidateBuffer&   candidates,
                             const BatchHeaderBuffer& batches,
                             const InstanceBuffer&    mainInstance,
                             const InstanceBuffer&    shadowInstance,
                             const IndirectBuffer&    indirect,
                             VkImageView              hzbView,
                             VkSampler                hzbSampler);

// Upload CullParamsUBO for this frame, then dispatch.
void dispatchCull(VkCommandBuffer cmd, GpuCullData& cull, uint32_t frame,
                  const CullParamsUBO& params);

void destroyGpuCull(VkDevice device, GpuCullData& cull);
