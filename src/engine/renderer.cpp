#include "renderer.h"
#include "resource_manager.h"
#include "components.h"
#include "frustum.h"
#include "instancing.h"
#include "skeletal.h"
#include "jobs.h"
#include "gpu_cull.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <limits>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <cstring>

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

struct EffectiveLOD {
    uint32_t   count = 1;
    MeshHandle mesh[MAX_LOD]{};
    float      maxDist[MAX_LOD]{};
    glm::vec3  aabbMin{0}, aabbMax{0};   // taken from LOD0 mesh
};

static EffectiveLOD resolveLOD(entt::entity e, entt::registry& reg, ResourceManager& res) {
    EffectiveLOD out;
    if (auto* lod = reg.try_get<MeshLODComponent>(e); lod && res.hasLODGroup(lod->group)) {
        const auto& g = res.getLODGroup(lod->group);
        out.count = std::min<uint32_t>(static_cast<uint32_t>(g.levels.size()), MAX_LOD);
        for (uint32_t i = 0; i < out.count; ++i) {
            out.mesh[i]    = g.levels[i].mesh;
            out.maxDist[i] = g.levels[i].maxDistance;
        }
        // Force last threshold to infinity so distance fall-through always
        // selects a valid LOD even if the user under-specifies.
        out.maxDist[out.count - 1] = std::numeric_limits<float>::max();
        const Mesh& m0 = res.getMesh(out.mesh[0]);
        out.aabbMin = m0.aabbMin;
        out.aabbMax = m0.aabbMax;
    } else {
        auto* mc = reg.try_get<MeshComponent>(e);
        out.count       = 1;
        out.mesh[0]     = mc ? mc->handle : MeshHandle{0};
        out.maxDist[0]  = std::numeric_limits<float>::max();
        const Mesh& m0  = res.getMesh(out.mesh[0]);
        out.aabbMin     = m0.aabbMin;
        out.aabbMax     = m0.aabbMax;
    }
    return out;
}

struct InstanceBatch {
    EffectiveLOD  lod;
    TextureHandle texture;
    uint32_t      shadowOffset;
    uint32_t      lodBase[MAX_LOD];
    uint32_t      lodCmdIndex[MAX_LOD];
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

static CullBuildResult buildCandidates(
    entt::registry& registry, ResourceManager& resources,
    CandidateInstance* candidatesMapped, uint32_t candidateCapacity,
    BatchHeader* headersMapped,          uint32_t headerCapacity,
    VkDrawIndexedIndirectCommand* indirectMapped, uint32_t indirectCapacity,
    uint32_t mainInstanceCapacity, uint32_t shadowInstanceCapacity,
    int& outTotal,
    JobSystem* jobs)
{
    CullBuildResult out;

    struct EntityRef {
        EffectiveLOD       lod;
        TextureHandle      texture;
        TransformComponent transform;
        MaterialComponent  material;
        uint64_t           batchKey;
        uint32_t           batchIndex;  // assigned in stage 3
    };

    std::vector<EntityRef> refs;
    refs.reserve(256);

    auto view = registry.view<TransformComponent, MeshComponent, MaterialComponent>();
    for (auto entity : view) {
        auto& tr  = view.get<TransformComponent>(entity);
        auto& mat = view.get<MaterialComponent>(entity);

        EntityRef r;
        r.lod       = resolveLOD(entity, registry, resources);
        r.texture   = mat.texture;
        r.transform = tr;
        r.material  = mat;

        bool hasGroup = registry.all_of<MeshLODComponent>(entity)
                     && resources.hasLODGroup(registry.get<MeshLODComponent>(entity).group);
        uint32_t keyId = hasGroup
            ? registry.get<MeshLODComponent>(entity).group.id
            : view.get<MeshComponent>(entity).handle.id;
        r.batchKey = makeBatchKey(r.lod, mat.texture.id, hasGroup, keyId);
        refs.push_back(r);
    }
    outTotal = static_cast<int>(refs.size());

    if (refs.empty() || candidatesMapped == nullptr) {
        return out;
    }

    // Sort by batch key so entities with the same (LOD chain, texture) end up
    // contiguous. We then walk and assign batch indices.
    std::vector<uint32_t> order(refs.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](uint32_t a, uint32_t b) { return refs[a].batchKey < refs[b].batchKey; });

    // Stage: create batches, assign batchIndex to each EntityRef.
    {
        uint64_t prev = ~refs[order[0]].batchKey; // force first miss
        for (uint32_t k = 0; k < order.size(); ++k) {
            uint32_t i = order[k];
            if (refs[i].batchKey != prev) {
                InstanceBatch b{};
                b.lod           = refs[i].lod;
                b.texture       = refs[i].texture;
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
    // Main: per-LOD reservation = lodCount × reservedCount slots (worst case
    // all entities pick the same LOD). Shadow: reservedCount slots.
    // Indirect: lodCount main cmds + 1 shadow cmd per batch, contiguous.
    {
        uint32_t mainCursor    = 0;
        uint32_t shadowCursor  = 0;
        uint32_t cmdCursor     = 0;
        for (auto& b : out.batches) {
            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodBase[i] = mainCursor;
                mainCursor  += b.reservedCount;
            }
            b.shadowOffset  = shadowCursor;
            shadowCursor   += b.reservedCount;

            for (uint32_t i = 0; i < b.lod.count; ++i) {
                b.lodCmdIndex[i] = cmdCursor++;
            }
            b.shadowCmdIndex = cmdCursor++;
        }
        // Bail out if any pool would overflow — drops this frame's draws but
        // keeps the renderer alive.
        if (mainCursor   > mainInstanceCapacity   ||
            shadowCursor > shadowInstanceCapacity ||
            cmdCursor    > indirectCapacity       ||
            out.numBatches > headerCapacity)
        {
            return CullBuildResult{};
        }
    }

    // Stage: author indirect commands + batch headers.
    for (uint32_t bi = 0; bi < out.numBatches; ++bi) {
        auto& b = out.batches[bi];

        for (uint32_t i = 0; i < b.lod.count; ++i) {
            const Mesh& mesh = resources.getMesh(b.lod.mesh[i]);
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = static_cast<uint32_t>(mesh.indices.size());
            cmd.instanceCount = 0;
            cmd.firstIndex    = 0;
            cmd.vertexOffset  = 0;
            cmd.firstInstance = 0;
            indirectMapped[b.lodCmdIndex[i]] = cmd;
        }
        // Shadow uses LOD0.
        {
            const Mesh& m0 = resources.getMesh(b.lod.mesh[0]);
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = static_cast<uint32_t>(m0.indices.size());
            cmd.instanceCount = 0;
            cmd.firstIndex    = 0;
            cmd.vertexOffset  = 0;
            cmd.firstInstance = 0;
            indirectMapped[b.shadowCmdIndex] = cmd;
        }

        BatchHeader h{};
        h.shadowBase = b.shadowOffset;
        h.shadowCmd  = b.shadowCmdIndex;
        h.lodCount   = b.lod.count;
        for (uint32_t i = 0; i < b.lod.count; ++i) {
            h.lodBase[i] = b.lodBase[i];
            h.lodCmd [i] = b.lodCmdIndex[i];
            h.lodDist[i] = b.lod.maxDist[i];
        }
        headersMapped[bi] = h;
    }

    // Stage: write candidate stream (parallel for >256 entities).
    const uint32_t toWrite = std::min(static_cast<uint32_t>(refs.size()), candidateCapacity);
    auto computeOne = [&](size_t i) {
        const auto& r = refs[i];
        glm::mat4 model = r.transform.getMatrix();
        glm::vec3 wMin, wMax;
        transformAabb(model, r.lod.aabbMin, r.lod.aabbMax, wMin, wMax);
        glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));

        CandidateInstance c;
        c.data.model      = model;
        c.data.colorTint  = r.material.color;
        c.data.matParams  = glm::vec4(r.material.metallic, r.material.roughness,
                                       static_cast<float>(r.texture.id), 0.0f);
        c.data.normalCol0 = glm::vec4(nm[0], 0.0f);
        c.data.normalCol1 = glm::vec4(nm[1], 0.0f);
        c.data.normalCol2 = glm::vec4(nm[2], 0.0f);

        uint32_t bi = r.batchIndex;
        c.aabbMin = glm::vec4(wMin, 0.0f);
        std::memcpy(&c.aabbMin.w, &bi, sizeof(uint32_t));
        c.aabbMax = glm::vec4(wMax, 0.0f);

        candidatesMapped[i] = c;
    };

    if (jobs && toWrite > 256) {
        jobs->parallel_for(toWrite, computeOne);
    } else {
        for (uint32_t i = 0; i < toWrite; ++i) computeOne(i);
    }

    out.numCandidates = toWrite;
    return out;
}

// ── Shadow pass ─────────────────────────────────────────────────────────────

static void recordShadowPass(VkCommandBuffer cmd, ShadowData& shadow,
                             VkPipeline shadowPipeline, VkPipelineLayout shadowLayout,
                             const CascadeUBO& cascadeUbo,
                             const std::vector<InstanceBatch>& batches,
                             VkBuffer shadowInstanceBuffer,
                             VkBuffer indirectBuffer,
                             ResourceManager& resources)
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
        vkCmdEndRenderPass(cmd);
    }
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
                           int vrsMode)
{
    // Offscreen render pass now has 3 attachments: color, motion, depth.
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
    rpBegin.renderPass        = offscreen.renderPass;
    rpBegin.framebuffer       = offscreen.framebuffer;
    rpBegin.renderArea.extent = offscreen.extent;
    rpBegin.clearValueCount   = static_cast<uint32_t>(clearValues.size());
    rpBegin.pClearValues      = clearValues.data();

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
    for (const auto& batch : batches) {
        if (batch.reservedCount == 0) continue;
        for (uint32_t i = 0; i < batch.lod.count; ++i) {
            const Mesh& mesh = resources.getMesh(batch.lod.mesh[i]);

            VkBuffer vbs[] = { mesh.vertexBuffer.buffer, mainInstanceBuffer };
            VkDeviceSize offsets[] = { 0, batch.lodBase[i] * sizeof(InstanceData) };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            if (VRS_SUPPORTED && VRS_SetRate) {
                VkExtent2D rate = shadingRateFor(vrsMode, i);
                VRS_SetRate(cmd, &rate, combiners);
            }

            VkDeviceSize cmdOffset = static_cast<VkDeviceSize>(batch.lodCmdIndex[i])
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
        } pc;
        pc.model     = tr.getMatrix();
        pc.color     = mat.color;
        pc.matParams = glm::vec4(mat.metallic, mat.roughness,
                                  static_cast<float>(mat.texture.id), 0.0f);
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
    updateUniformBuffer(*info.descriptors, frame, *info.ubo);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device, swapchain.swapchain, UINT64_MAX,
        swapchain.imageAvailableSemaphores[frame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return true;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    // Late stats readback (one frame lag). The shadow commands are interleaved
    // with main commands now (one shadow per batch right after its LOD cmds),
    // so the old "first half = main" heuristic no longer works. We can't easily
    // separate visible main from shadow counts without per-frame metadata —
    // expose only the total instances drawn by main (sum LODs) via a future
    // mechanism. For now report visibleEntities = -1 (unknown) so the UI can
    // skip it.
    if (info.visibleEntities) *info.visibleEntities = -1;

    vkResetFences(device, 1, &swapchain.inFlightFences[frame]);

    // ── CPU: build candidate stream + pre-author indirect commands ─────────
    int total = 0;
    auto build = buildCandidates(
        *info.registry, *info.resources,
        static_cast<CandidateInstance*>(info.candidates->mapped[frame]),
        info.candidates->capacity,
        static_cast<BatchHeader*>(info.batchHeaders->mapped[frame]),
        info.batchHeaders->capacity,
        static_cast<VkDrawIndexedIndirectCommand*>(info.indirect->mapped[frame]),
        info.indirect->capacity,
        info.mainInstances->capacity, info.shadowInstances->capacity,
        total, info.jobSystem);
    if (info.totalEntities) *info.totalEntities = total;

    VkBuffer mainInstBuf   = info.mainInstances->buffers[frame].buffer;
    VkBuffer shadowInstBuf = info.shadowInstances->buffers[frame].buffer;
    VkBuffer indirectBuf   = info.indirect->buffers[frame].buffer;

    VkCommandBuffer cmd = renderer.commandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // ── TLAS build (RT) ─────────────────────────────────────────────────────
    // Walk the visible meshes and gather their world transforms + BLAS
    // addresses, then rebuild the TLAS into the per-frame buffer. The build
    // includes its own post-barrier so subsequent shaders can ray-query it.
    // Skip cleanly when RT is unsupported, disabled, or no instances exist.
    if (RT_SUPPORTED && info.rtScene && info.rtSettings && info.rtSettings->enabled) {
        std::vector<RtInstanceDesc> rtInstances;
        rtInstances.reserve(static_cast<size_t>(total));

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
            d.material.vertexAddr = splitAddress(
                getBufferDeviceAddress(device, m.vertexBuffer.buffer));
            d.material.indexAddr  = splitAddress(
                getBufferDeviceAddress(device, m.indexBuffer.buffer));
            rtInstances.push_back(d);
        }

        buildTlas(*info.rtScene, frame, info.physicalDevice, device, cmd, rtInstances);

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
    if (build.numCandidates > 0) {
        dispatchCull(cmd, *info.gpuCull, frame, *info.cullParams);

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                         | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // Skip the CSM pass entirely when RT shadows are doing the work — the
    // fragment shader's CSM branch is dead in that case, so the maps would
    // just be expensive scratch.
    const bool rtShadowsActive = RT_SUPPORTED && info.rtSettings
                              && info.rtSettings->enabled && info.rtSettings->shadows;
    if (!rtShadowsActive) {
        recordShadowPass(cmd, *info.shadow, info.shadowPipeline, info.shadowLayout,
                         *info.cascadeUbo, build.batches, shadowInstBuf, indirectBuf,
                         *info.resources);
    }

    recordMainPass(cmd, *info.offscreen, info.mainPipeline, info.mainLayout,
                   info.descriptors->sceneSets[frame],
                   build.batches, mainInstBuf, indirectBuf, *info.resources,
                   info.settings->vrsMode);
    recordSkinnedDraws(cmd, info, frame);
    vkCmdEndRenderPass(cmd);

    // ── HZB reduction (for NEXT frame's occlusion cull) ────────────────────
    // The render-pass external dependency already transitions the depth
    // attachment to DEPTH_STENCIL_READ_ONLY_OPTIMAL and makes it visible to
    // FRAGMENT_SHADER reads (for SSAO). Add a barrier so COMPUTE_SHADER reads
    // can see the depth write too.
    if (info.hzb) {
        VkMemoryBarrier depthBarrier{};
        depthBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        depthBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depthBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &depthBarrier, 0, nullptr, 0, nullptr);

        recordHzbReduce(cmd, *info.hzb);
    }


    // SSAO uses the offscreen depth.
    recordSSAO(cmd, *info.ssao, *info.postfx, frame);

    // Bloom reads from offscreen HDR color.
    recordBloom(cmd, *info.bloom, *info.postfx, *info.settings);

    // Composite reads offscreen (HDR) + bloom + SSAO, writes LDR.
    recordCompositePass(cmd, *info.ldr,
                        info.postfx->composite, info.postfx->compositeLayout,
                        info.composite->descriptorSets[frame]);

    // FXAA: LDR → swapchain.
    recordFxaaPass(cmd, swapchain, *info.ldr,
                   info.postfx->fxaa, info.postfx->fxaaLayout,
                   info.ldr->descriptorSet, imageIndex, info.settings->fxaaEnabled);

    VK_CHECK(vkEndCommandBuffer(cmd));

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

    renderer.currentFrame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
    return false;
}
