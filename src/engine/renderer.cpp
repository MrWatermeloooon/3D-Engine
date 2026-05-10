#include "renderer.h"
#include "resource_manager.h"
#include "components.h"
#include "../utils/vk_check.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <limits>
#include <stdexcept>
#include <array>

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

// ── Shadow pass ─────────────────────────────────────────────────────────────

static void recordShadowPass(VkCommandBuffer cmd, ShadowData& shadow,
                             VkPipeline shadowPipeline, VkPipelineLayout shadowLayout,
                             const CascadeUBO& cascadeUbo,
                             entt::registry& registry, ResourceManager& resources)
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

        auto view = registry.view<TransformComponent, MeshComponent>();
        for (auto entity : view) {
            auto& transform = view.get<TransformComponent>(entity);
            auto& meshComp  = view.get<MeshComponent>(entity);
            const Mesh& mesh = resources.getMesh(meshComp.handle);

            VkBuffer vbs[]   = { mesh.vertexBuffer.buffer };
            VkDeviceSize off[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vbs, off);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            glm::mat4 model = transform.getMatrix();
            vkCmdPushConstants(cmd, shadowLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               sizeof(glm::mat4), sizeof(glm::mat4), &model);

            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }
}

// ── Main pass (offscreen HDR) ───────────────────────────────────────────────

static void recordMainPass(VkCommandBuffer cmd, OffscreenTarget& offscreen,
                           VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                           VkDescriptorSet sceneSet,
                           entt::registry& registry, ResourceManager& resources)
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

    auto view = registry.view<TransformComponent, MeshComponent, MaterialComponent>();
    for (auto entity : view) {
        auto& transform = view.get<TransformComponent>(entity);
        auto& meshComp  = view.get<MeshComponent>(entity);
        auto& material  = view.get<MaterialComponent>(entity);

        const Mesh& mesh = resources.getMesh(meshComp.handle);
        VkDescriptorSet matSet = resources.getMaterialSet(material.texture);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                1, 1, &matSet, 0, nullptr);

        VkBuffer vbs[] = { mesh.vertexBuffer.buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        glm::mat4 model = transform.getMatrix();
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &model);

        struct FragPush {
            glm::vec4 color;
            float metallic, roughness, pad0, pad1;
        } fp;
        fp.color = material.color;
        fp.metallic = material.metallic;
        fp.roughness = material.roughness;
        fp.pad0 = fp.pad1 = 0.0f;
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           sizeof(glm::mat4), sizeof(FragPush), &fp);

        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

// ── Composite + UI pass (swapchain) ─────────────────────────────────────────

static void recordCompositePass(VkCommandBuffer cmd, SwapchainData& swapchain,
                                VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                                VkDescriptorSet compositeSet,
                                uint32_t imageIndex)
{
    VkClearValue clear{};
    clear.color.float32[0] = 0.0f;
    clear.color.float32[1] = 0.0f;
    clear.color.float32[2] = 0.0f;
    clear.color.float32[3] = 1.0f;

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = swapchain.renderPass;
    rpBegin.framebuffer       = swapchain.framebuffers[imageIndex];
    rpBegin.renderArea.extent = swapchain.extent;
    rpBegin.clearValueCount   = 1;
    rpBegin.pClearValues      = &clear;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Composite full-screen pass
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &compositeSet, 0, nullptr);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(swapchain.extent.width);
    viewport.height   = static_cast<float>(swapchain.extent.height);
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{}; scissor.extent = swapchain.extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    // ImGui on top
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

    VkCommandBuffer cmd = renderer.commandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // 1. Shadow cascades
    recordShadowPass(cmd, *info.shadow, info.shadowPipeline, info.shadowLayout,
                     *info.cascadeUbo, *info.registry, *info.resources);

    // 2. Main pass to HDR offscreen
    recordMainPass(cmd, *info.offscreen, info.mainPipeline, info.mainLayout,
                   info.descriptors->sceneSets[frame],
                   *info.registry, *info.resources);

    // 3. SSAO (always recorded so its image stays in SHADER_READ_ONLY layout;
    //    the composite shader gates the contribution with enableSSAO).
    recordSSAO(cmd, *info.ssao, *info.postfx, frame);

    // 4. Bloom chain — always run so bloom mip0 is in a valid layout for composite.
    recordBloom(cmd, *info.bloom, *info.postfx, *info.settings);

    // 5. Composite + ImGui to swapchain
    recordCompositePass(cmd, swapchain, info.postfx->composite, info.postfx->compositeLayout,
                        info.composite->descriptorSets[frame], imageIndex);

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
