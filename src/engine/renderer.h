#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>

#include <entt/entt.hpp>

#include "swapchain.h"
#include "mesh.h"
#include "descriptors.h"
#include "shadow.h"
#include "postfx.h"
#include "frustum.h"
#include "instancing.h"
#include "skeletal.h"
#include "gpu_cull.h"
#include "hzb.h"
#include "raytracing.h"
#include "sky.h"
#include "spatial.h"
#include "components.h"
#include "config.h"

class ResourceManager;
class Profiler;

// Per-entity data resolved exactly once per frame in the registry walk that
// also populates the spatial index. SpatialIndex::Entry carries a userIndex
// back-reference so buildCandidates reads this array directly with zero
// component-pool lookups.
struct EffectiveLOD {
    uint32_t   count = 1;
    MeshHandle mesh[engine_config::MAX_LOD]{};
    float      maxDist[engine_config::MAX_LOD]{};
    glm::vec3  aabbMin{0}, aabbMax{0};
};

struct ResolvedEntity {
    entt::entity        entity;
    EffectiveLOD        lod;
    TextureHandle       texture;
    TransformComponent  transform;   // copy (mat4 + mat3 cache fields)
    MaterialComponent   material;    // copy
    glm::vec3           wMin;
    glm::vec3           wMax;
    uint64_t            batchKey;
};

struct RendererData {
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    uint32_t currentFrame = 0;

    // Persistent storage for the CPU pre-cull pass. Cleared each frame; the
    // vectors keep capacity so we avoid reallocation under steady-state.
    SpatialIndex                     spatial;
    std::vector<SpatialIndex::Entry> visibleEntries;
    std::vector<ResolvedEntity>      resolved;
};

void createCommandPool(RendererData& data, VkDevice device, uint32_t graphicsFamily);
void allocateCommandBuffers(RendererData& data, VkDevice device);

struct DrawFrameInfo {
    VkPipeline       mainPipeline;
    VkPipelineLayout mainLayout;
    VkPipeline       shadowPipeline;
    VkPipelineLayout shadowLayout;
    SkyData*         sky = nullptr;

    DescriptorData*  descriptors;
    ShadowData*      shadow;
    OffscreenTarget* offscreen;
    BloomChain*      bloom;
    SSAOTarget*      ssao;
    CompositeData*   composite;
    LdrTarget*       ldr;
    PostFXPipelines* postfx;
    PostFXSettings*  settings;

    UniformBufferObject* ubo;
    CascadeUBO*          cascadeUbo;
    CullParamsUBO*       cullParams;       // built by engine.cpp every frame
    CandidateBuffer*     candidates;
    BatchHeaderBuffer*   batchHeaders;
    InstanceBuffer*      mainInstances;
    InstanceBuffer*      shadowInstances;
    IndirectBuffer*      indirect;
    GpuCullData*         gpuCull;
    HzbData*             hzb;

    // Ray tracing (Phase 1b: TLAS rebuilt per-frame from the registry's
    // visible meshes. Consumed by ray queries in Phase 1c).
    RtScene*             rtScene        = nullptr;
    RtSettings*          rtSettings     = nullptr;
    VkPhysicalDevice     physicalDevice = VK_NULL_HANDLE; // for TLAS resize

    // Skeletal animation
    SkinnedMesh*         skinnedMesh             = nullptr;
    VkPipeline           skinnedPipeline         = VK_NULL_HANDLE;
    VkPipelineLayout     skinnedLayout           = VK_NULL_HANDLE;
    VkPipeline           skinnedShadowPipeline   = VK_NULL_HANDLE;
    VkPipelineLayout     skinnedShadowLayout     = VK_NULL_HANDLE;
    VkDescriptorSet      boneDescriptorSet       = VK_NULL_HANDLE;

    class JobSystem*     jobSystem          = nullptr;

    // Per-frame stats (written by drawFrame, read by UI)
    int* visibleEntities = nullptr;
    int* totalEntities   = nullptr;

    entt::registry*  registry;
    ResourceManager* resources;

    // Per-object previous-frame model matrices for motion vectors. Engine
    // maintains the map across frames; drawFrame reads it during candidate
    // build and writes it back at the end of the frame.
    std::unordered_map<entt::entity, glm::mat4>* prevTransforms = nullptr;

    Profiler* profiler = nullptr;
};

bool drawFrame(RendererData& renderer, SwapchainData& swapchain,
               VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue,
               const DrawFrameInfo& info,
               bool& framebufferResized);
