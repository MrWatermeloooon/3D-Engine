#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>

#include "buffer.h"
#include "config.h"

// Per-instance GPU data. 144 bytes. Tightly packed; matches the vertex
// input description in getInstanceAttributeDescriptions().
//
// normalCols stores the precomputed mat3 normal matrix (transpose(inverse(mat3(model))))
// as three column vectors. Computing it on the CPU saves 16 multiplies per vertex
// per frame on the GPU — adds up fast when you have 10k+ instances.
struct InstanceData {
    glm::mat4 model;        // location 5..8  (4 vec4 slots) — bumped by 1 from 4 to make room for tangent at vertex location 4
    glm::vec4 colorTint;    // location 9
    glm::vec4 matParams;    // location 10 — x=metallic, y=roughness, z=textureIndex (bindless), w=normalTextureIndex
    glm::vec4 normalCol0;   // location 11
    glm::vec4 normalCol1;   // location 12
    glm::vec4 normalCol2;   // location 13
    // Previous-frame world matrix for per-object motion vectors. On the first
    // appearance of an entity, the engine seeds this to the current model so
    // motion is zero rather than huge garbage. The vertex shader reprojects
    // vWorldPos through this to compute prevClipPos for the motion vector
    // attachment; without this, motion vectors would only encode camera
    // motion and rotating/moving objects would ghost under TAA.
    glm::mat4 prevModel;    // location 14..17
    // Second material params slot. x = bindless heightTexture slot (0 = no
    // parallax), y = parallaxScale (tangent-space depth). z, w reserved.
    glm::vec4 matParams2;   // location 18
};

VkVertexInputBindingDescription getInstanceBindingDescription();
std::array<VkVertexInputAttributeDescription, 14> getInstanceAttributeDescriptions();

// Compute-cull input. CPU writes one of these per entity per frame. The cull
// compute shader reads it, frustum-tests, and copies `data` into one of the
// instance output buffers. `aabbMin.w` carries the batch index as float bits.
struct CandidateInstance {
    InstanceData data;       // 208 (model + tint + params + 3 normalCols + prevModel)
    glm::vec4    aabbMin;    // .xyz world-space AABB min, .w = batchIndex (uintBitsToFloat)
    glm::vec4    aabbMax;    // .xyz world-space AABB max, .w unused
};
static_assert(sizeof(InstanceData) == 224,
              "InstanceData must match shader std430 layout (model+tint+params+3normalCols+prevModel+matParams2)");
static_assert(sizeof(CandidateInstance) == 256, "CandidateInstance must match shader std430 layout");

// Max LOD levels per group. Bumping this expands BatchHeader (CPU + shader
// must agree). 4 is plenty for hand-authored LOD chains.
using engine_config::MAX_LOD;

// Per-batch metadata for the cull compute shader. CPU writes one per
// (LODGroup, texture) batch per frame. `lodCount` is in 1..MAX_LOD; entries
// past lodCount are ignored. lodDist[i] is the upper view-distance threshold
// for LOD i (last entry should be a sentinel like FLT_MAX). Shadow draws use
// LOD0 only — shadow silhouettes stay sharp at any distance.
struct BatchHeader {
    uint32_t shadowBase;            // offset into shadowInstance buffer
    uint32_t shadowCmd;             // indirect-cmd index for shadow draw
    uint32_t lodCount;              // 1..MAX_LOD
    // LOD index used for shadow draws. 0 = LOD0 (sharp silhouettes). For very
    // distant casters, 1 or 2 is significantly cheaper with no visible change.
    // Clamped to lodCount-1 by the shadow setup path; ignored when CSM is off.
    uint32_t shadowLodBias;
    uint32_t lodBase[MAX_LOD];      // offsets into mainInstance per LOD (pass 0)
    uint32_t lodCmd[MAX_LOD];       // indirect-cmd indices per LOD (pass 0)
    float    lodDist[MAX_LOD];      // upper distance thresholds per LOD
    // Two-pass occlusion (Phase 2.2). Pass 0 writes [lodBase, lodBase+count0)
    // and lodCmd[]; pass 1 writes [lodBaseLate, lodBaseLate+count1) and
    // lodCmdLate[]. The two ranges are disjoint, so the main-pass draws can
    // be issued separately: pass A reads lodCmd[] / lodBase[], pass B reads
    // lodCmdLate[] / lodBaseLate[].
    uint32_t lodBaseLate[MAX_LOD];  // offsets into mainInstance per LOD (pass 1)
    uint32_t lodCmdLate[MAX_LOD];   // indirect-cmd indices per LOD (pass 1)
};
static_assert(sizeof(BatchHeader) == 96, "BatchHeader must match shader std430 layout");

// Per-frame instance buffer. Used in two roles:
//   * compute shader writes into it as an SSBO (one buffer for main-pass
//     visible instances, one for shadow-pass all-instances)
//   * graphics pipeline reads it as a per-instance vertex binding
// Therefore usage = STORAGE_BUFFER | VERTEX_BUFFER. Memory remains
// HOST_VISIBLE | HOST_COHERENT — the candidate buffer also lives in this
// memory tier, and keeping everything coherent dodges staging logic.
struct InstanceBuffer {
    std::vector<AllocatedBuffer> buffers;       // one per frame in flight
    std::vector<void*>           mapped;        // unused for compute outputs, kept for parity
    uint32_t                     capacity = 0;  // max instances per frame
};

void createInstanceBuffer(InstanceBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyInstanceBuffer(VkDevice device, InstanceBuffer& b);

// Per-frame indirect command buffer. CPU pre-writes per-batch metadata
// (indexCount/firstIndex/vertexOffset) plus zeroed instanceCount; the cull
// compute shader atomically increments instanceCount. Usage therefore needs
// INDIRECT | STORAGE.
struct IndirectBuffer {
    std::vector<AllocatedBuffer> buffers;
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0; // max commands per frame
};

void createIndirectBuffer(IndirectBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyIndirectBuffer(VkDevice device, IndirectBuffer& b);

// Per-frame candidate buffer. Holds one CandidateInstance per renderable entity.
struct CandidateBuffer {
    std::vector<AllocatedBuffer> buffers;
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0;
};

void createCandidateBuffer(CandidateBuffer& b, VkPhysicalDevice physicalDevice,
                           VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyCandidateBuffer(VkDevice device, CandidateBuffer& b);

// Per-frame batch-header buffer. Holds one BatchHeader per (mesh,texture) batch.
struct BatchHeaderBuffer {
    std::vector<AllocatedBuffer> buffers;
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0;
};

void createBatchHeaderBuffer(BatchHeaderBuffer& b, VkPhysicalDevice physicalDevice,
                             VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyBatchHeaderBuffer(VkDevice device, BatchHeaderBuffer& b);
