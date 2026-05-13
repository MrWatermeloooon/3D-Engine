#include "renderer.h"
#include "resource_manager.h"
#include "components.h"
#include "frustum.h"
#include "instancing.h"
#include "skeletal.h"
#include "jobs.h"
#include "gpu_cull.h"
#include "vulkan_init.h"
#include "profiler.h"
#include "../utils/vk_check.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <limits>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

// ── Command pool & buffers ──────────────────────────────────────────────────

void createCommandPool(RendererData& data, VkDevice device, uint32_t graphicsFamily) {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsFamily;
    VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &data.commandPool));
}

void allocateCommandBuffers(RendererData& data, VkDevice device) {
    data.commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = data.commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(data.commandBuffers.size());
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, data.commandBuffers.data()));
}

// ── Batch construction (CPU side of GPU-driven culling + LOD) ──────────────
//
// A *batch* = (LOD chain, texture). For entities without an LOD group, the
// chain has a single level pointing at the MeshComponent's handle.
//
// Per-batch GPU resource layout:
//   * shadowInstance[shadowOffset .. +reservedCount)  — written by compute
//     (no cull). Drawn with LOD0's mesh in the shadow pass.
//   * mainInstance  [lodBase[i] .. +reservedCount)    — written by compute,
//     packed by LOD assignment. Drawn with lodMesh[i] in the main pass.
//   * indirect: lodCount main commands + 1 shadow command, contiguous.

// `EffectiveLOD` is defined in renderer.h (shared with ResolvedEntity).
// resolveLOD() used to live here as a helper; the work is now inlined into
// resolveAndPopulateSpatial below since the single-walk consolidation needs
// fine control over which fields it captures and where the work lands.

struct InstanceBatch {
    EffectiveLOD  lod;
    TextureHandle texture;
    uint32_t      shadowOffset;
    uint32_t      lodBase[MAX_LOD];
    uint32_t      lodBaseLate[MAX_LOD];     // pass-1 ranges (Phase 2.2)
    uint32_t      lodCmdIndex[MAX_LOD];
    uint32_t      lodCmdLateIndex[MAX_LOD]; // pass-1 cmds
    uint32_t      shadowCmdIndex;
    uint32_t      reservedCount;        // entities assigned to this batch
};

struct CullBuildResult {
    std::vector<InstanceBatch> batches;
    uint32_t numCandidates = 0;
    uint32_t numBatches    = 0;
};

// Key uniquely identifying a batch: LOD-group entities use group.id with a tag
// bit (so they can't collide with single-mesh entities of the same numeric id).
static uint64_t makeBatchKey(const EffectiveLOD& lod, uint32_t textureId,
                             bool hasGroup, uint32_t groupOrMeshId)
{
    (void)lod;
    uint64_t upper = hasGroup
        ? (uint64_t(0x1) << 63) | groupOrMeshId
        : uint64_t(groupOrMeshId);
    return (upper << 32) | textureId;
}

// `ResolvedEntity` is defined in renderer.h so RendererData can hold a
// std::vector<ResolvedEntity> directly (persisted across frames for capacity).

// Single registry walk: enumerate the drawable view, resolve LOD chain,
// fetch local AABB, transform to world space, compute batch key, snapshot
// transform + material. Also populates the spatial index with each entry's
// AABB + a userIndex back into `resolved`.
//
// `outTotal` reports the total entity count for the UI; visible-after-cull
// is the size of the queryFrustum result vector.
static void resolveAndPopulateSpatial(
    SpatialIndex& spatial,
    std::vector<ResolvedEntity>& resolved,
    entt::registry& registry,
    ResourceManager& resources,
    int& outTotal)
{
    auto view = registry.view<TransformComponent, MeshComponent, MaterialComponent>();
    resolved.clear();
    resolved.reserve(view.size_hint());

    for (auto entity : view) {
        const auto& tr  = view.get<TransformComponent>(entity);
        const auto& mat = view.get<MaterialComponent>(entity);

        ResolvedEntity r;
        r.entity    = entity;
        r.transform = tr;
        r.material  = mat;
        r.texture   = mat.texture;

        // LOD resolution — single try_get; the result feeds both the AABB
        // source (LOD0 mesh) and the batch key.
        auto* lodComp = registry.try_get<MeshLODComponent>(entity);
        bool hasGroup = lodComp && resources.hasLODGroup(lodComp->group);

        uint32_t keyId;
        if (hasGroup) {
            const auto& g = resources.getLODGroup(lodComp->group);
            r.lod.count = std::min<uint32_t>(static_cast<uint32_t>(g.levels.size()), MAX_LOD);
            for (uint32_t k = 0; k < r.lod.count; ++k) {
                r.lod.mesh[k]    = g.levels[k].mesh;
                r.lod.maxDist[k] = g.levels[k].maxDistance;
            }
            r.lod.maxDist[r.lod.count - 1] = std::numeric_limits<float>::max();
            keyId = lodComp->group.id;
        } else {
            const auto& mc = view.get<MeshComponent>(entity);
            r.lod.count       = 1;
            r.lod.mesh[0]     = mc.handle;
            r.lod.maxDist[0]  = std::numeric_limits<float>::max();
            keyId = mc.handle.id;
        }

        // AABB from LOD0's mesh (or the single mesh in the no-group case).
        const Mesh& m0 = resources.getMesh(r.lod.mesh[0]);
        r.lod.aabbMin  = m0.aabbMin;
        r.lod.aabbMax  = m0.aabbMax;
        transformAabb(tr.getMatrix(), r.lod.aabbMin, r.lod.aabbMax, r.wMin, r.wMax);

        r.batchKey = makeBatchKey(r.lod, mat.texture.id, hasGroup, keyId);

        resolved.push_back(r);
    }

    outTotal = static_cast<int>(resolved.size());

    // Mirror into the spatial index. userIndex preserves the back-link
    // through the BVH's median-split reordering.
    spatial.resizeForBulkInsert(resolved.size());
    for (size_t i = 0; i < resolved.size(); ++i) {
        SpatialIndex::Entry e;
        e.entity    = resolved[i].entity;
        e.wMin      = resolved[i].wMin;
        e.wMax      = resolved[i].wMax;
        e.userIndex = static_cast<uint32_t>(i);
        spatial.mutableEntry(i) = e;
    }
    spatial.build();
}

static CullBuildResult buildCandidates(
    ResourceManager& resources,
    const std::vector<ResolvedEntity>& resolved,
    const std::vector<SpatialIndex::Entry>& visibleEntries,
    CandidateInstance* candidatesMapped, uint32_t candidateCapacity,
    BatchHeader* headersMapped,          uint32_t headerCapacity,
    VkDrawIndexedIndirectCommand* indirectMapped, uint32_t indirectCapacity,
    uint32_t mainInstanceCapacity, uint32_t shadowInstanceCapacity,
    JobSystem* jobs,
    const std::unordered_map<entt::entity, glm::mat4>* prevTransforms)
{
    CullBuildResult out;

    // EntityRef is now a lightweight reference into the resolved array plus
    // the assigned batch index. All per-entity data was computed during the
    // single resolve walk in resolveAndPopulateSpatial; we just need to sort
    // by batchKey and assign batches.
    struct EntityRef {
        uint32_t resolvedIdx;   // index into `resolved`
        uint64_t batchKey;      // copied for sort locality
        uint32_t batchIndex;    // assigned in stage 3
    };

    std::vector<EntityRef> refs;
    refs.resize(visibleEntries.size());

    // Gather: zero registry hits, just copy index + key from the pre-resolved
    // array. The BVH preserved userIndex through its median-split reorder.
    for (size_t i = 0; i < visibleEntries.size(); ++i) {
        const ResolvedEntity& r = resolved[visibleEntries[i].userIndex];
        refs[i].resolvedIdx = visibleEntries[i].userIndex;
        refs[i].batchKey    = r.batchKey;
    }

    if (refs.empty() || candidatesMapped == nullptr) {
        return out;
    }

    // Sort by batch key so entities with the same (LOD chain, texture) end up
    // contiguous. We then walk and assign batch indices.
    std::vector<uint32_t> order(refs.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](uint32_t a, uint32_t b) { return refs[a].batchKey < refs[b].batchKey; });

    // Stage: create batches, assign batchIndex to each EntityRef. LOD chain
    // and texture come from the pre-resolved entity record.
    {
        uint64_t prev = ~refs[order[0]].batchKey; // force first miss
        for (uint32_t k = 0; k < order.size(); ++k) {
            uint32_t i = order[k];
            const ResolvedEntity& re = resolved[refs[i].resolvedIdx];
            if (refs[i].batchKey != prev) {
                InstanceBatch b{};
                b.lod           = re.lod;
                b.texture       = re.texture;
                b.reservedCount = 0;
                out.batches.push_back(b);
                prev = refs[i].batchKey;
            }
            refs[i].batchIndex = static_cast<uint32_t>(out.batches.size() - 1);
            out.batches.back().reservedCount++;
        }
    }
    out.numBatches = static_cast<uint32_t>(out.batches.size());

    // Stage: allocate instance-buffer ranges + indirect command slots per batch.
    // Two-pass occlusion (Phase 2.2) doubles the per-LOD reservation: pass 0
    // writes into lodBase[]/lodCmd[], pass 1 writes into the disjoint
    // lodBaseLate[]/lodCmdLate[] ranges. Worst-case total per batch:
    //   * main:   2 * lodCount * reservedCount slots
    //   * shadow: reservedCount slots (single pass)
    //   * indirect: 2 * lodCount main cmds + 1 shadow cmd
    {
        uint32_t mainCursor    = 0;
        uint32_t shadowCursor  = 0;
        uint32_t cmdCursor     = 0;
        for (auto& b : out.batches) {
            // Pass-0 main ranges
            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodBase[i] = mainCursor;
                mainCursor  += b.reservedCount;
            }
            // Pass-1 main ranges (disjoint, contiguous after pass-0's)
            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodBaseLate[i] = mainCursor;
                mainCursor      += b.reservedCount;
            }
            b.shadowOffset  = shadowCursor;
            shadowCursor   += b.reservedCount;

            // Pass-0 main cmds
            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodCmdIndex[i] = cmdCursor++;
            }
            // Pass-1 main cmds
            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodCmdLateIndex[i] = cmdCursor++;
            }
            b.shadowCmdIndex = cmdCursor++;
        }
        // Bail out if any pool would overflow — drops this frame's draws but
        // keeps the renderer alive. Log loudly so the user knows why the
        // screen suddenly went blank, instead of silently rendering nothing.
        if (mainCursor   > mainInstanceCapacity   ||
            shadowCursor > shadowInstanceCapacity ||
            cmdCursor    > indirectCapacity       ||
            out.numBatches > headerCapacity)
        {
            std::cerr << "[VulkanEngine] Cull buffer overflow — frame dropped. "
                      << "mainCursor=" << mainCursor << "/" << mainInstanceCapacity
                      << " shadowCursor=" << shadowCursor << "/" << shadowInstanceCapacity
                      << " cmdCursor=" << cmdCursor << "/" << indirectCapacity
                      << " batches=" << out.numBatches << "/" << headerCapacity
                      << ". Increase MAX_INSTANCES_PER_FRAME / MAX_BATCHES_PER_FRAME.\n";
            return CullBuildResult{};
        }
    }

    // Stage: author indirect commands + batch headers.
    for (uint32_t bi = 0; bi < out.numBatches; ++bi) {
        auto& b = out.batches[bi];

        // Pass-0 + pass-1 main cmds. Same index count per LOD; instanceCount=0,
        // populated atomically by cull.comp.
        for (uint32_t i = 0; i < b.lod.count; ++i) {
            const Mesh& mesh = resources.getMesh(b.lod.mesh[i]);
            assert(mesh.indices.size() <= std::numeric_limits<uint32_t>::max()
                   && "Mesh index count exceeds uint32_t");
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = static_cast<uint32_t>(mesh.indices.size());
            cmd.instanceCount = 0;
            cmd.firstIndex    = 0;
            cmd.vertexOffset  = 0;
            cmd.firstInstance = 0;
            indirectMapped[b.lodCmdIndex[i]]     = cmd;  // pass 0
            indirectMapped[b.lodCmdLateIndex[i]] = cmd;  // pass 1 (same shape)
        }
        // Shadow uses LOD0, no two-pass.
        {
            const Mesh& m0 = resources.getMesh(b.lod.mesh[0]);
            assert(m0.indices.size() <= std::numeric_limits<uint32_t>::max()
                   && "Mesh index count exceeds uint32_t");
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = static_cast<uint32_t>(m0.indices.size());
            cmd.instanceCount = 0;
            cmd.firstIndex    = 0;
            cmd.vertexOffset  = 0;
            cmd.firstInstance = 0;
            indirectMapped[b.shadowCmdIndex] = cmd;
        }

        BatchHeader h{};
        h.shadowBase    = b.shadowOffset;
        h.shadowCmd     = b.shadowCmdIndex;
        h.lodCount      = b.lod.count;
        h.shadowLodBias = 0;   // LOD0 by default — distant casters can opt up.
        for (uint32_t i = 0; i < b.lod.count; ++i) {
            h.lodBase[i]     = b.lodBase[i];
            h.lodCmd [i]     = b.lodCmdIndex[i];
            h.lodDist[i]     = b.lod.maxDist[i];
            h.lodBaseLate[i] = b.lodBaseLate[i];
            h.lodCmdLate [i] = b.lodCmdLateIndex[i];
        }
        headersMapped[bi] = h;
    }

    // Stage: write candidate stream. Reads the pre-resolved entity record
    // directly — every per-entity field (transform, material, texture, world
    // AABB) was filled during the single resolve walk.
    const uint32_t toWrite = std::min(static_cast<uint32_t>(refs.size()), candidateCapacity);
    for (uint32_t i = 0; i < toWrite; ++i) {
        const EntityRef& ref = refs[i];
        const ResolvedEntity& r = resolved[ref.resolvedIdx];

        // TransformComponent's matrix + normal-matrix cache are populated on
        // first access; for static entities subsequent frames are float
        // compare + early return.
        const glm::mat4& model = r.transform.getMatrix();
        const glm::mat3& nm    = r.transform.getNormalMatrix();

        // Previous-frame model from the engine-owned cache. On first
        // appearance, fall back to current model so motion is zero rather
        // than a huge spike from a stale identity.
        glm::mat4 prevModel = model;
        if (prevTransforms) {
            auto it = prevTransforms->find(r.entity);
            if (it != prevTransforms->end()) prevModel = it->second;
        }

        CandidateInstance c;
        c.data.model      = model;
        c.data.colorTint  = r.material.color;
        c.data.matParams  = glm::vec4(r.material.metallic, r.material.roughness,
                                       static_cast<float>(r.texture.id),
                                       static_cast<float>(r.material.normalTexture.id));
        c.data.normalCol0 = glm::vec4(nm[0], 0.0f);
        c.data.normalCol1 = glm::vec4(nm[1], 0.0f);
        c.data.normalCol2 = glm::vec4(nm[2], 0.0f);
        c.data.prevModel  = prevModel;
        c.data.matParams2 = glm::vec4(
            static_cast<float>(r.material.heightTexture.id),
            r.material.parallaxScale,
            0.0f, 0.0f);

        c.aabbMin = glm::vec4(r.wMin, glm::uintBitsToFloat(ref.batchIndex));
        c.aabbMax = glm::vec4(r.wMax, 0.0f);

        candidatesMapped[i] = c;
    }

    out.numCandidates = toWrite;
    (void)jobs; (void)resources;
    return out;
}

// ── Shadow pass ─────────────────────────────────────────────────────────────

static void recordShadowPass(VkCommandBuffer cmd, ShadowData& shadow,
                             VkPipeline shadowPipeline, VkPipelineLayout shadowLayout,
                             const CascadeUBO& cascadeUbo,
                             const std::vector<InstanceBatch>& batches,
                             VkBuffer shadowInstanceBuffer,
                             VkBuffer indirectBuffer,
                             ResourceManager& resources,
                             const DrawFrameInfo& info,
                             uint32_t frame)
{
    VkClearValue clear{};
    clear.depthStencil = { 1.0f, 0 };

    for (uint32_t c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
        VkRenderPassBeginInfo rp{};
        rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass        = shadow.renderPass;
        rp.framebuffer       = shadow.framebuffers[c];
        rp.renderArea.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE };
        rp.clearValueCount   = 1;
        rp.pClearValues      = &clear;

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        // ── Static (instanced) shadow draws ────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        vkCmdPushConstants(cmd, shadowLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &cascadeUbo.lightViewProj[c]);

        for (const auto& batch : batches) {
            if (batch.reservedCount == 0) continue;
            const Mesh& mesh = resources.getMesh(batch.lod.mesh[0]);

            VkBuffer vbs[]   = { mesh.vertexBuffer.buffer, shadowInstanceBuffer };
            VkDeviceSize off[] = { 0, batch.shadowOffset * sizeof(InstanceData) };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbs, off);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            VkDeviceSize cmdOffset = static_cast<VkDeviceSize>(batch.shadowCmdIndex)
                                   * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(cmd, indirectBuffer, cmdOffset, 1,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        // ── Skinned shadow draws ───────────────────────────────────────────
        // Walk every entity with SkinnedMeshComponent + TransformComponent.
        // Single skinned mesh per scene (engine.cpp owns one m_skinnedMesh);
        // each entity reuses the same vertex/index buffer + bone palette but
        // has its own world transform.
        if (info.skinnedMesh && info.skinnedShadowPipeline != VK_NULL_HANDLE) {
            auto& reg = *info.registry;
            auto sview = reg.view<TransformComponent, SkinnedMeshComponent>();
            if (sview.begin() != sview.end()) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  info.skinnedShadowPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        info.skinnedShadowLayout, 0, 1,
                                        &info.boneDescriptorSet, 0, nullptr);

                VkBuffer vbs[]   = { info.skinnedMesh->vertexBuffer.buffer };
                VkDeviceSize off[] = { 0 };
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, off);
                vkCmdBindIndexBuffer(cmd, info.skinnedMesh->indexBuffer.buffer, 0,
                                     VK_INDEX_TYPE_UINT32);

                for (auto e : sview) {
                    glm::mat4 push[2] = {
                        cascadeUbo.lightViewProj[c],
                        sview.get<TransformComponent>(e).getMatrix(),
                    };
                    vkCmdPushConstants(cmd, info.skinnedShadowLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
                    vkCmdDrawIndexed(cmd,
                                     static_cast<uint32_t>(info.skinnedMesh->indices.size()),
                                     1, 0, 0, 0);
                }
            }
        }

        vkCmdEndRenderPass(cmd);
    }
    (void)frame;
}

// ── Main pass ───────────────────────────────────────────────────────────────

// Map a (vrsMode, lodIndex) pair to a VkExtent2D shading rate. The shading-rate
// enum encodes rate as power-of-two block sizes: {1,1}=full, {2,1}/{1,2}=2x,
// {2,2}=4x, {4,4}=16x cheaper. Modes 2/3 force a single rate regardless of LOD.
static VkExtent2D shadingRateFor(int vrsMode, uint32_t lodIndex) {
    switch (vrsMode) {
        case 0: return {1, 1};
        case 2: return {2, 2};
        case 3: return {4, 4};
        case 1:
        default:
            switch (lodIndex) {
                case 0:  return {1, 1};
                case 1:  return {2, 1};
                case 2:  return {2, 2};
                default: return {4, 4};
            }
    }
}

static void recordMainPass(VkCommandBuffer cmd, OffscreenTarget& offscreen,
                           VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                           VkDescriptorSet sceneSet,
                           const std::vector<InstanceBatch>& batches,
                           VkBuffer mainInstanceBuffer,
                           VkBuffer indirectBuffer,
                           ResourceManager& resources,
                           int vrsMode,
                           bool latePass)
{
    // Two-pass occlusion (Phase 2.2): pass A uses the CLEAR renderpass and
    // pass-0 indirect slots; pass B uses the LOAD renderpass and pass-1 slots
    // (newly-revealed objects only).
    std::array<VkClearValue, 3> clearValues{};
    clearValues[0].color.float32[0] = 0.05f;
    clearValues[0].color.float32[1] = 0.06f;
    clearValues[0].color.float32[2] = 0.08f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].color.float32[0] = 0.0f; // motion: clear to zero
    clearValues[1].color.float32[1] = 0.0f;
    clearValues[2].depthStencil.depth   = 1.0f;
    clearValues[2].depthStencil.stencil = 0;

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = latePass ? offscreen.renderPassLate : offscreen.renderPass;
    rpBegin.framebuffer       = offscreen.framebuffer;
    rpBegin.renderArea.extent = offscreen.extent;
    // LOAD renderpass ignores clear values; pass them only for the CLEAR path.
    if (!latePass) {
        rpBegin.clearValueCount   = static_cast<uint32_t>(clearValues.size());
        rpBegin.pClearValues      = clearValues.data();
    }

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(offscreen.extent.width);
    viewport.height   = static_cast<float>(offscreen.extent.height);
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{}; scissor.extent = offscreen.extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkDescriptorSet sets[] = { sceneSet, resources.bindlessSet() };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 2, sets, 0, nullptr);

    // VRS dynamic state was baked into the pipeline only when supported.
    // Since the pipeline declares the dynamic state, we MUST call the setter
    // before the first draw if VRS_SUPPORTED — start at full rate (1×1).
    const VkFragmentShadingRateCombinerOpKHR combiners[2] = {
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,   // primitive op
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,   // attachment op
    };
    if (VRS_SUPPORTED && VRS_SetRate) {
        VkExtent2D rate{1, 1};
        VRS_SetRate(cmd, &rate, combiners);
    }

    // Per batch: iterate the batch's LODs and issue one indirect draw each.
    // Pass A reads lodBase / lodCmdIndex; pass B reads the disjoint late slots.
    for (const auto& batch : batches) {
        if (batch.reservedCount == 0) continue;
        for (uint32_t i = 0; i < batch.lod.count; ++i) {
            const Mesh& mesh = resources.getMesh(batch.lod.mesh[i]);

            uint32_t baseSlot = latePass ? batch.lodBaseLate[i]     : batch.lodBase[i];
            uint32_t cmdSlot  = latePass ? batch.lodCmdLateIndex[i] : batch.lodCmdIndex[i];

            VkBuffer vbs[] = { mesh.vertexBuffer.buffer, mainInstanceBuffer };
            VkDeviceSize offsets[] = { 0, baseSlot * sizeof(InstanceData) };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            if (VRS_SUPPORTED && VRS_SetRate) {
                VkExtent2D rate = shadingRateFor(vrsMode, i);
                VRS_SetRate(cmd, &rate, combiners);
            }

            VkDeviceSize cmdOffset = static_cast<VkDeviceSize>(cmdSlot)
                                   * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(cmd, indirectBuffer, cmdOffset, 1,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }
    }

    // Reset to full rate before any subsequent draws (e.g. skinned meshes).
    if (VRS_SUPPORTED && VRS_SetRate) {
        VkExtent2D rate{1, 1};
        VRS_SetRate(cmd, &rate, combiners);
    }
}

static void recordSkinnedDraws(VkCommandBuffer cmd, const DrawFrameInfo& info, uint32_t frame)
{
    if (!info.skinnedMesh || info.skinnedPipeline == VK_NULL_HANDLE) return;

    auto& reg = *info.registry;
    auto view = reg.view<TransformComponent, SkinnedMeshComponent, MaterialComponent>();
    if (view.begin() == view.end()) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedPipeline);
    VkDescriptorSet sets3[] = {
        info.descriptors->sceneSets[frame],
        info.resources->bindlessSet(),
        info.boneDescriptorSet,
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedLayout,
                            0, 3, sets3, 0, nullptr);

    VkBuffer vbs[]   = { info.skinnedMesh->vertexBuffer.buffer };
    VkDeviceSize off[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, off);
    vkCmdBindIndexBuffer(cmd, info.skinnedMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    for (auto e : view) {
        auto& tr  = view.get<TransformComponent>(e);
        auto& mat = view.get<MaterialComponent>(e);

        struct SkinnedPC {
            glm::mat4 model;
            glm::vec4 color;
            glm::vec4 matParams;
            glm::vec4 matParams2;
        } pc;
        pc.model     = tr.getMatrix();
        pc.color     = mat.color;
        pc.matParams = glm::vec4(mat.metallic, mat.roughness,
                                  static_cast<float>(mat.texture.id),
                                  static_cast<float>(mat.normalTexture.id));
        // Skinned vertex format has no per-vertex tangent yet, so leave
        // parallax disabled even if the material has a height map assigned.
        pc.matParams2 = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, info.skinnedLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(SkinnedPC), &pc);

        vkCmdDrawIndexed(cmd,
                         static_cast<uint32_t>(info.skinnedMesh->indices.size()),
                         1, 0, 0, 0);
    }
}

// ── Composite + FXAA passes ────────────────────────────────────────────────

static void recordCompositePass(VkCommandBuffer cmd, LdrTarget& ldr,
                                VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                                VkDescriptorSet compositeSet)
{
    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = ldr.renderPass;
    rpBegin.framebuffer       = ldr.framebuffer;
    rpBegin.renderArea.extent = ldr.extent;
    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &compositeSet, 0, nullptr);

    VkViewport vp{};
    vp.width = static_cast<float>(ldr.extent.width);
    vp.height = static_cast<float>(ldr.extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.extent = ldr.extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

static void recordFxaaPass(VkCommandBuffer cmd, SwapchainData& swapchain, LdrTarget& ldr,
                           VkPipeline fxaaPipeline, VkPipelineLayout fxaaLayout,
                           VkDescriptorSet fxaaSet, uint32_t imageIndex, bool enable)
{
    VkClearValue clear{};
    clear.color.float32[0] = 0.0f;
    clear.color.float32[1] = 0.0f;
    clear.color.float32[2] = 0.0f;
    clear.color.float32[3] = 1.0f;

    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = swapchain.renderPass;
    rp.framebuffer       = swapchain.framebuffers[imageIndex];
    rp.renderArea.extent = swapchain.extent;
    rp.clearValueCount   = 1;
    rp.pClearValues      = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaaPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaaLayout,
                            0, 1, &fxaaSet, 0, nullptr);

    VkViewport vp{};
    vp.width  = static_cast<float>(swapchain.extent.width);
    vp.height = static_cast<float>(swapchain.extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.extent = swapchain.extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    glm::vec4 pcParams(1.0f / static_cast<float>(ldr.extent.width),
                       1.0f / static_cast<float>(ldr.extent.height),
                       enable ? 1.0f : 0.0f, 0.0f);
    vkCmdPushConstants(cmd, fxaaLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(glm::vec4), &pcParams);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
}

// ── Draw frame ──────────────────────────────────────────────────────────────

bool drawFrame(RendererData& renderer, SwapchainData& swapchain,
               VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue,
               const DrawFrameInfo& info,
               bool& framebufferResized)
{
    uint32_t frame = renderer.currentFrame;

    vkWaitForFences(device, 1, &swapchain.inFlightFences[frame], VK_TRUE, UINT64_MAX);
    {
        auto _ = info.profiler ? info.profiler->cpuScope("updateUniformBuffer")
                                : Profiler::CpuGuard(nullptr, nullptr);
        updateUniformBuffer(*info.descriptors, frame, *info.ubo);
    }

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device, swapchain.swapchain, UINT64_MAX,
        swapchain.imageAvailableSemaphores[frame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return true;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    // visibleEntities is now reported by the CPU pre-cull below (post-spatial
    // query, pre-GPU occlusion cull). This is the "frustum-visible" count, not
    // the "passed occlusion + on-screen" count — the latter would need a
    // round-trip from the GPU and would lag a frame anyway.

    vkResetFences(device, 1, &swapchain.inFlightFences[frame]);

    // ── CPU pre-cull: walk all entities, build a per-frame BVH, query the
    //    camera frustum so the heavy candidate-build runs only on visibles ──
    int total = 0;
    {
        auto _ = info.profiler ? info.profiler->cpuScope("resolve + spatial build")
                                : Profiler::CpuGuard(nullptr, nullptr);
        resolveAndPopulateSpatial(renderer.spatial, renderer.resolved,
                                  *info.registry, *info.resources, total);
    }
    if (info.totalEntities) *info.totalEntities = total;

    // Frustum used for the CPU pre-cull is the SAME one the GPU cull receives
    // via CullParamsUBO (engine.cpp populated cullParams->frustumPlanes from
    // the current view-proj). Build a Frustum struct from those planes.
    Frustum cpuFrustum;
    for (int i = 0; i < 6; ++i) cpuFrustum.planes[i] = info.cullParams->frustumPlanes[i];

    {
        auto _ = info.profiler ? info.profiler->cpuScope("spatial query")
                                : Profiler::CpuGuard(nullptr, nullptr);
        renderer.spatial.queryFrustum(cpuFrustum, renderer.visibleEntries);
    }
    if (info.visibleEntities) *info.visibleEntities = static_cast<int>(renderer.visibleEntries.size());

    CullBuildResult build;
    {
        auto _ = info.profiler ? info.profiler->cpuScope("buildCandidates")
                                : Profiler::CpuGuard(nullptr, nullptr);
        build = buildCandidates(
            *info.resources,
            renderer.resolved,
            renderer.visibleEntries,
            static_cast<CandidateInstance*>(info.candidates->mapped[frame]),
            info.candidates->capacity,
            static_cast<BatchHeader*>(info.batchHeaders->mapped[frame]),
            info.batchHeaders->capacity,
            static_cast<VkDrawIndexedIndirectCommand*>(info.indirect->mapped[frame]),
            info.indirect->capacity,
            info.mainInstances->capacity, info.shadowInstances->capacity,
            info.jobSystem, info.prevTransforms);
    }

    VkBuffer mainInstBuf   = info.mainInstances->buffers[frame].buffer;
    VkBuffer shadowInstBuf = info.shadowInstances->buffers[frame].buffer;
    VkBuffer indirectBuf   = info.indirect->buffers[frame].buffer;

    VkCommandBuffer cmd = renderer.commandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Profiler: read back previous frame's GPU timestamps and reset the pool
    // for this frame's writes. Must run BEFORE any gpuBegin/gpuEnd, and AFTER
    // vkWaitForFences so the prior submission's writes are visible.
    if (info.profiler) info.profiler->beginFrame(device, cmd, frame);

    // ── TLAS build (RT) ─────────────────────────────────────────────────────
    // Walk the visible meshes and gather their world transforms + BLAS
    // addresses, then rebuild the TLAS into the per-frame buffer. The build
    // includes its own post-barrier so subsequent shaders can ray-query it.
    // Skip cleanly when RT is unsupported, disabled, or no instances exist.
    if (RT_SUPPORTED && info.rtScene && info.rtSettings && info.rtSettings->enabled) {
        std::vector<RtInstanceDesc> rtInstances;
        rtInstances.reserve(static_cast<size_t>(total));

        {
        auto _ = info.profiler ? info.profiler->cpuScope("TLAS instance gather")
                                : Profiler::CpuGuard(nullptr, nullptr);
        auto rtView = info.registry->view<TransformComponent, MeshComponent, MaterialComponent>();
        for (auto e : rtView) {
            // LOD0 mesh is what we trace against (shadow rays don't need LOD).
            MeshHandle h{0};
            if (auto* lod = info.registry->try_get<MeshLODComponent>(e);
                lod && info.resources->hasLODGroup(lod->group))
            {
                h = info.resources->getLODGroup(lod->group).levels[0].mesh;
            } else {
                h = rtView.get<MeshComponent>(e).handle;
            }
            const Mesh& m = info.resources->getMesh(h);
            if (m.rt.deviceAddress == 0) continue; // BLAS missing — skip
            const auto& mat = rtView.get<MaterialComponent>(e);
            RtInstanceDesc d{};
            d.transform           = rtView.get<TransformComponent>(e).getMatrix();
            d.blasAddress         = m.rt.deviceAddress;
            d.material.color      = mat.color;
            d.material.params     = glm::vec4(mat.metallic, mat.roughness, 0.0f, 0.0f);
            // Use the cached addresses populated at mesh upload time —
            // calling vkGetBufferDeviceAddress here was ~1.7 ms / frame at
            // 1.9k entities (two API calls per entity).
            d.material.vertexAddr = splitAddress(m.vertexAddress);
            d.material.indexAddr  = splitAddress(m.indexAddress);
            rtInstances.push_back(d);
        }
        } // TLAS instance gather scope

        if (info.profiler) info.profiler->gpuBegin(cmd, "TLAS build");
        buildTlas(*info.rtScene, frame, info.physicalDevice, device, cmd, rtInstances);
        if (info.profiler) info.profiler->gpuEnd(cmd);

        // Point this frame's scene descriptor at the freshly built TLAS so
        // mesh.frag's ray queries see it. The TLAS handle is stable across
        // frames unless storage grows; we write it unconditionally because the
        // write is cheap and idempotent.
        writeSceneTlas(*info.descriptors, device, frame, info.rtScene->tlas[frame]);
        writeSceneRtMaterials(*info.descriptors, device, frame,
                              info.rtScene->materialBuffer[frame].buffer,
                              info.rtScene->materialCapacity[frame]);
    }

    info.cullParams->numCandidates = build.numCandidates;
    const bool runCull = build.numCandidates > 0;

    auto cullToDrawBarrier = [&]() {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                         | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // ── Pass 0 cull (frustum + prev-frame HZB) ────────────────────────────
    if (runCull) {
        if (info.profiler) info.profiler->gpuBegin(cmd, "Cull pass 0");
        dispatchCull(cmd, *info.gpuCull, frame, *info.cullParams, 0u);
        if (info.profiler) info.profiler->gpuEnd(cmd);
        cullToDrawBarrier();
    }

    // ── Shadow pass (single-pass, uses pass-0's shadow instance writes) ────
    const bool rtShadowsActive = RT_SUPPORTED && info.rtSettings
                              && info.rtSettings->enabled && info.rtSettings->shadows;
    if (!rtShadowsActive) {
        if (info.profiler) info.profiler->gpuBegin(cmd, "Shadow pass (CSM)");
        recordShadowPass(cmd, *info.shadow, info.shadowPipeline, info.shadowLayout,
                         *info.cascadeUbo, build.batches, shadowInstBuf, indirectBuf,
                         *info.resources, info, frame);
        if (info.profiler) info.profiler->gpuEnd(cmd);
    }

    // ── Main pass A (CLEAR + sky + skinned) ───────────────────────────────
    if (info.profiler) info.profiler->gpuBegin(cmd, "Main pass A");
    recordMainPass(cmd, *info.offscreen, info.mainPipeline, info.mainLayout,
                   info.descriptors->sceneSets[frame],
                   build.batches, mainInstBuf, indirectBuf, *info.resources,
                   info.settings->vrsMode, /*latePass=*/false);
    recordSkinnedDraws(cmd, info, frame);
    // Sky in pass A: writes color where depth==1.0. Pass-B geometry may
    // overwrite some of those pixels with newly-visible surfaces — which is
    // exactly the correct behavior (geometry hides the sky behind it).
    if (info.sky && info.sky->pipeline) {
        recordSkyPass(cmd, *info.sky, info.descriptors->sceneSets[frame],
                      info.offscreen->extent);
    }
    vkCmdEndRenderPass(cmd);
    if (info.profiler) info.profiler->gpuEnd(cmd);

    // ── HZB reduce + pass-1 cull + main pass B ────────────────────────────
    // Renderpass A's external dependency already transitioned depth →
    // DEPTH_STENCIL_READ_ONLY_OPTIMAL and made it visible to FRAGMENT_SHADER.
    // Add a barrier so COMPUTE_SHADER reads see the depth write too.
    if (info.hzb && runCull) {
        VkMemoryBarrier depthBarrier{};
        depthBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        depthBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depthBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &depthBarrier, 0, nullptr, 0, nullptr);

        if (info.profiler) info.profiler->gpuBegin(cmd, "HZB reduce (mid)");
        recordHzbReduce(cmd, *info.hzb);
        if (info.profiler) info.profiler->gpuEnd(cmd);

        // HZB-reduce's writes must be visible to pass-1 cull's reads.
        VkMemoryBarrier hzbToCull{};
        hzbToCull.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hzbToCull.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hzbToCull.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &hzbToCull, 0, nullptr, 0, nullptr);

        // Pass-1 cull: re-tests culled candidates against the fresh HZB. The
        // mid-frame view-proj is identical to what pass-0 saw, so we pass the
        // same CullParamsUBO contents (dispatchCull skips the re-upload when
        // pass != 0).
        if (info.profiler) info.profiler->gpuBegin(cmd, "Cull pass 1");
        dispatchCull(cmd, *info.gpuCull, frame, *info.cullParams, 1u);
        if (info.profiler) info.profiler->gpuEnd(cmd);
        cullToDrawBarrier();

        if (info.profiler) info.profiler->gpuBegin(cmd, "Main pass B");
        recordMainPass(cmd, *info.offscreen, info.mainPipeline, info.mainLayout,
                       info.descriptors->sceneSets[frame],
                       build.batches, mainInstBuf, indirectBuf, *info.resources,
                       info.settings->vrsMode, /*latePass=*/true);
        vkCmdEndRenderPass(cmd);
        if (info.profiler) info.profiler->gpuEnd(cmd);
    }


    // SSAO uses the offscreen depth.
    if (info.profiler) info.profiler->gpuBegin(cmd, "SSAO");
    recordSSAO(cmd, *info.ssao, *info.postfx, frame);
    if (info.profiler) info.profiler->gpuEnd(cmd);

    // Bloom reads from offscreen HDR color.
    if (info.profiler) info.profiler->gpuBegin(cmd, "Bloom");
    recordBloom(cmd, *info.bloom, *info.postfx, *info.settings);
    if (info.profiler) info.profiler->gpuEnd(cmd);

    // Composite reads offscreen (HDR) + bloom + SSAO, writes LDR.
    if (info.profiler) info.profiler->gpuBegin(cmd, "Composite");
    recordCompositePass(cmd, *info.ldr,
                        info.postfx->composite, info.postfx->compositeLayout,
                        info.composite->descriptorSets[frame]);
    if (info.profiler) info.profiler->gpuEnd(cmd);

    // FXAA: LDR → swapchain.
    if (info.profiler) info.profiler->gpuBegin(cmd, "FXAA + UI");
    recordFxaaPass(cmd, swapchain, *info.ldr,
                   info.postfx->fxaa, info.postfx->fxaaLayout,
                   info.ldr->descriptorSet, imageIndex, info.settings->fxaaEnabled);
    if (info.profiler) info.profiler->gpuEnd(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));
    if (info.profiler) info.profiler->endFrame();

    VkSemaphore waitSemaphores[]      = { swapchain.imageAvailableSemaphores[frame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore signalSemaphores[]    = { swapchain.renderFinishedSemaphores[imageIndex] };

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, swapchain.inFlightFences[frame]));

    VkSwapchainKHR swapchains[] = { swapchain.swapchain };
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapchains;
    presentInfo.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        return true;
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image");
    }

    // Snapshot this frame's per-entity world matrices for next frame's motion
    // vectors. Done at the end so an aborted frame (early return above) won't
    // corrupt the cache. Stale entries for destroyed entities are pruned in
    // one pass — cheap relative to the ECS iteration cost of the frame.
    if (info.prevTransforms) {
        auto& cache = *info.prevTransforms;
        auto& reg   = *info.registry;
        for (auto it = cache.begin(); it != cache.end(); ) {
            if (!reg.valid(it->first)) it = cache.erase(it);
            else ++it;
        }
        auto view = reg.view<TransformComponent, MeshComponent, MaterialComponent>();
        for (auto e : view) {
            cache[e] = view.get<TransformComponent>(e).getMatrix();
        }
    }

    renderer.currentFrame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
    return false;
}
