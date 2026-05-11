#include "renderer.h"
#include "resource_manager.h"
#include "components.h"
#include "frustum.h"
#include "instancing.h"
#include "skeletal.h"
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

// ── Batching ────────────────────────────────────────────────────────────────

struct PendingInstance {
    uint32_t     meshId;
    uint32_t     textureId;
    bool         visible;
    InstanceData data;
};

struct InstanceBatch {
    MeshHandle    mesh;
    TextureHandle texture;
    uint32_t      offset;       // start index in instance buffer
    uint32_t      visibleCount; // first N instances are visible (in frustum)
    uint32_t      totalCount;   // all instances (used for shadow pass)
};

// Build per-frame batches: gathers renderable entities, frustum-tests them,
// sorts by (mesh, texture, !visible), writes contiguous instance data to the
// mapped instance buffer, and returns per-batch ranges.
static std::vector<InstanceBatch> buildBatches(
    entt::registry& registry, ResourceManager& resources,
    const Frustum& frustum,
    InstanceData* mapped, uint32_t capacity,
    int& outVisible, int& outTotal)
{
    std::vector<PendingInstance> pending;
    pending.reserve(64);

    auto view = registry.view<TransformComponent, MeshComponent, MaterialComponent>();
    for (auto entity : view) {
        auto& tr   = view.get<TransformComponent>(entity);
        auto& mc   = view.get<MeshComponent>(entity);
        auto& mat  = view.get<MaterialComponent>(entity);
        const Mesh& mesh = resources.getMesh(mc.handle);

        glm::mat4 model = tr.getMatrix();
        glm::vec3 wMin, wMax;
        transformAabb(model, mesh.aabbMin, mesh.aabbMax, wMin, wMax);
        bool visible = aabbInFrustum(frustum, wMin, wMax);

        PendingInstance p;
        p.meshId    = mc.handle.id;
        p.textureId = mat.texture.id;
        p.visible   = visible;
        p.data.model      = model;
        p.data.colorTint  = mat.color;
        p.data.matParams  = glm::vec4(mat.metallic, mat.roughness, 0.0f, 0.0f);
        pending.push_back(p);
    }

    // Sort: group by (mesh, texture), visible-first within each group
    std::sort(pending.begin(), pending.end(),
        [](const PendingInstance& a, const PendingInstance& b) {
            if (a.meshId    != b.meshId)    return a.meshId    < b.meshId;
            if (a.textureId != b.textureId) return a.textureId < b.textureId;
            return a.visible > b.visible;
        });

    outTotal   = static_cast<int>(pending.size());
    outVisible = 0;
    for (const auto& p : pending) if (p.visible) outVisible++;

    // Write to mapped instance buffer + collect batches
    std::vector<InstanceBatch> batches;
    uint32_t toWrite = std::min(static_cast<uint32_t>(pending.size()), capacity);
    for (uint32_t i = 0; i < toWrite; ++i) {
        mapped[i] = pending[i].data;
    }

    uint32_t i = 0;
    while (i < toWrite) {
        InstanceBatch b;
        b.mesh.id    = pending[i].meshId;
        b.texture.id = pending[i].textureId;
        b.offset     = i;
        uint32_t end = i;
        uint32_t visCount = 0;
        while (end < toWrite &&
               pending[end].meshId    == b.mesh.id &&
               pending[end].textureId == b.texture.id) {
            if (pending[end].visible) visCount++;
            end++;
        }
        b.visibleCount = visCount;
        b.totalCount   = end - i;
        batches.push_back(b);
        i = end;
    }

    return batches;
}

// ── Shadow pass ─────────────────────────────────────────────────────────────

static void recordShadowPass(VkCommandBuffer cmd, ShadowData& shadow,
                             VkPipeline shadowPipeline, VkPipelineLayout shadowLayout,
                             const CascadeUBO& cascadeUbo,
                             const std::vector<InstanceBatch>& batches,
                             VkBuffer instanceBuffer,
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
            if (batch.totalCount == 0) continue;
            const Mesh& mesh = resources.getMesh(batch.mesh);

            VkBuffer vbs[]   = { mesh.vertexBuffer.buffer, instanceBuffer };
            VkDeviceSize off[] = { 0, batch.offset * sizeof(InstanceData) };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbs, off);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()),
                             batch.totalCount, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }
}

// ── Main pass ───────────────────────────────────────────────────────────────

static void recordMainPass(VkCommandBuffer cmd, OffscreenTarget& offscreen,
                           VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                           VkDescriptorSet sceneSet,
                           const std::vector<InstanceBatch>& batches,
                           VkBuffer instanceBuffer,
                           ResourceManager& resources)
{
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color.float32[0] = 0.05f;
    clearValues[0].color.float32[1] = 0.06f;
    clearValues[0].color.float32[2] = 0.08f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil.depth   = 1.0f;
    clearValues[1].depthStencil.stencil = 0;

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

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &sceneSet, 0, nullptr);

    for (const auto& batch : batches) {
        if (batch.visibleCount == 0) continue;
        const Mesh& mesh = resources.getMesh(batch.mesh);
        VkDescriptorSet matSet = resources.getMaterialSet(batch.texture);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                1, 1, &matSet, 0, nullptr);

        VkBuffer vbs[] = { mesh.vertexBuffer.buffer, instanceBuffer };
        VkDeviceSize offsets[] = { 0, batch.offset * sizeof(InstanceData) };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()),
                         batch.visibleCount, 0, 0, 0);
    }
    // (caller may issue more draws — render pass closes in drawFrame)
}

// Draw all entities with SkinnedMeshComponent using the skinned pipeline.
// Currently a single shared SkinnedMesh resource per engine; multiple entities
// still share its vertex/index buffers but each gets its own model transform.
static void recordSkinnedDraws(VkCommandBuffer cmd, const DrawFrameInfo& info, uint32_t frame)
{
    if (!info.skinnedMesh || info.skinnedPipeline == VK_NULL_HANDLE) return;

    auto& reg = *info.registry;
    auto view = reg.view<TransformComponent, SkinnedMeshComponent, MaterialComponent>();
    if (view.begin() == view.end()) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedLayout,
                            0, 1, &info.descriptors->sceneSets[frame], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedLayout,
                            2, 1, &info.boneDescriptorSet, 0, nullptr);

    VkBuffer vbs[]   = { info.skinnedMesh->vertexBuffer.buffer };
    VkDeviceSize off[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, off);
    vkCmdBindIndexBuffer(cmd, info.skinnedMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    for (auto e : view) {
        auto& tr  = view.get<TransformComponent>(e);
        auto& mat = view.get<MaterialComponent>(e);

        VkDescriptorSet matSet = info.resources->getMaterialSet(mat.texture);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.skinnedLayout,
                                1, 1, &matSet, 0, nullptr);

        struct SkinnedPC {
            glm::mat4 model;
            glm::vec4 color;
            glm::vec4 matParams;
        } pc;
        pc.model     = tr.getMatrix();
        pc.color     = mat.color;
        pc.matParams = glm::vec4(mat.metallic, mat.roughness, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, info.skinnedLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(SkinnedPC), &pc);

        vkCmdDrawIndexed(cmd,
                         static_cast<uint32_t>(info.skinnedMesh->indices.size()),
                         1, 0, 0, 0);
    }
}

// ── Composite + FXAA passes (unchanged) ────────────────────────────────────

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

    vkResetFences(device, 1, &swapchain.inFlightFences[frame]);

    // Build batches + write to instance buffer for this frame
    int vis = 0, tot = 0;
    auto batches = buildBatches(*info.registry, *info.resources, *info.cameraFrustum,
                                static_cast<InstanceData*>(info.instances->mapped[frame]),
                                info.instances->capacity,
                                vis, tot);
    if (info.visibleEntities) *info.visibleEntities = vis;
    if (info.totalEntities)   *info.totalEntities   = tot;
    VkBuffer instBuf = info.instances->buffers[frame].buffer;

    VkCommandBuffer cmd = renderer.commandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // 1. Shadow cascades (uses total count — off-screen objects still cast shadows)
    recordShadowPass(cmd, *info.shadow, info.shadowPipeline, info.shadowLayout,
                     *info.cascadeUbo, batches, instBuf, *info.resources);

    // 2. Main pass: opens offscreen render pass, draws instanced batches,
    //    then draws skinned meshes (same pass — they share the render target).
    recordMainPass(cmd, *info.offscreen, info.mainPipeline, info.mainLayout,
                   info.descriptors->sceneSets[frame],
                   batches, instBuf, *info.resources);
    recordSkinnedDraws(cmd, info, frame);
    vkCmdEndRenderPass(cmd);

    // 3. SSAO + bloom (always recorded so layouts stay valid for composite)
    recordSSAO(cmd, *info.ssao, *info.postfx, frame);
    recordBloom(cmd, *info.bloom, *info.postfx, *info.settings);

    // 4. Composite → LDR
    recordCompositePass(cmd, *info.ldr,
                        info.postfx->composite, info.postfx->compositeLayout,
                        info.composite->descriptorSets[frame]);

    // 5. FXAA (+ ImGui) → swapchain
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
