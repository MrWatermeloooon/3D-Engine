#include "engine.h"
#include "frustum.h"

#include <imgui.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

// ── Init ────────────────────────────────────────────────────────────────────

void Engine::init() {
    m_window = new Window(1280, 720, "VulkanEngine");

    auto extensions = m_window->getRequiredExtensions();
    m_vk.instance       = createInstance(extensions);
    m_vk.debugMessenger = setupDebugMessenger(m_vk.instance);
    m_vk.surface        = m_window->createSurface(m_vk.instance);
    m_vk.physicalDevice = pickPhysicalDevice(m_vk.instance, m_vk.surface);
    m_vk.queueFamilies  = findQueueFamilies(m_vk.physicalDevice, m_vk.surface);
    m_vk.device         = createLogicalDevice(m_vk.physicalDevice, m_vk.queueFamilies);
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.graphicsFamily.value(), 0, &m_vk.graphicsQueue);
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.presentFamily.value(),  0, &m_vk.presentQueue);

    createCommandPool(m_renderer, m_vk.device, m_vk.queueFamilies.graphicsFamily.value());

    // Swapchain (color-only, used for composite + UI)
    auto extent = m_window->getFramebufferSize();
    createSwapchain(m_swapchain, m_vk.physicalDevice, m_vk.device, m_vk.surface,
                    m_vk.queueFamilies.graphicsFamily.value(),
                    m_vk.queueFamilies.presentFamily.value(), extent);
    createImageViews(m_swapchain, m_vk.device);
    createRenderPass(m_swapchain, m_vk.device);
    createFramebuffers(m_swapchain, m_vk.device);
    createPresentSemaphores(m_swapchain, m_vk.device);
    createSyncObjects(m_swapchain, m_vk.device);

    // Pick depth format for offscreen
    m_depthFormat = findDepthFormat(m_vk.physicalDevice);

    // Offscreen HDR target (main pass → here)
    createOffscreenTarget(m_offscreen, m_vk.physicalDevice, m_vk.device,
                          m_swapchain.extent, m_depthFormat);

    // Bloom chain
    createBloomChain(m_bloom, m_vk.physicalDevice, m_vk.device, m_swapchain.extent);

    // SSAO
    createSSAO(m_ssao, m_vk.physicalDevice, m_vk.device,
               m_renderer.commandPool, m_vk.graphicsQueue,
               m_swapchain.extent, MAX_FRAMES_IN_FLIGHT);

    // Composite + LDR (composite writes here, FXAA reads from here)
    createCompositeData(m_composite, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);
    createLdrTarget(m_ldr, m_vk.physicalDevice, m_vk.device, m_swapchain.extent);

    // Shadow
    createShadowResources(m_shadow, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);

    // Light buffers
    createLightBuffers(m_lightBuffers, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);

    // Instance buffer (capacity 16384 instances per frame — covers 10k+ objects)
    createInstanceBuffer(m_instances, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, 16384);
    // Indirect buffer: 2 commands per batch (main + shadow), up to 1024 batches per frame.
    createIndirectBuffer(m_indirect, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, 2048);

    // Descriptor set layouts (scene + material)
    m_descriptors.sceneSetLayout    = createSceneSetLayout(m_vk.device);
    m_descriptors.materialSetLayout = createMaterialSetLayout(m_vk.device);

    // Scene UBO buffers
    createUniformBuffers(m_descriptors, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);

    // Descriptor pool + scene sets (with light + cascade + shadow refs) + bindless texture set
    createDescriptorPool(m_descriptors, m_vk.device, MAX_FRAMES_IN_FLIGHT, 64);

    std::vector<VkBuffer> lightBufs, cascadeBufs;
    for (auto& b : m_lightBuffers.buffers)  lightBufs.push_back(b.buffer);
    for (auto& b : m_shadow.cascadeBuffers) cascadeBufs.push_back(b.buffer);
    createSceneDescriptorSets(m_descriptors, m_vk.device, MAX_FRAMES_IN_FLIGHT,
                              lightBufs, cascadeBufs,
                              m_shadow.arrayView, m_shadow.sampler);
    allocateBindlessTexturesSet(m_descriptors, m_vk.device);

    // Resource manager (default mesh + texture)
    m_resources.init(m_vk.physicalDevice, m_vk.device,
                     m_renderer.commandPool, m_vk.graphicsQueue, m_descriptors);

    // Main pipeline targets the OFFSCREEN render pass (not the swapchain)
    m_pipeline = createGraphicsPipeline(m_vk.device, m_offscreen.renderPass,
                                        m_offscreen.extent,
                                        m_descriptors.sceneSetLayout, m_descriptors.materialSetLayout,
                                        "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");

    // Shadow pipeline
    m_shadowPipeline = createShadowPipeline(m_vk.device, m_shadow.renderPass,
                                            SHADOW_MAP_SIZE, "shaders/shadow.vert.spv");

    // ── Skeletal animation setup ────────────────────────────────────────
    m_skinnedMesh = createTestBoneChain();
    uploadSkinnedMesh(m_skinnedMesh, m_vk.physicalDevice, m_vk.device,
                      m_renderer.commandPool, m_vk.graphicsQueue);

    // Bone palette descriptor set layout (set 2, binding 0, UBO)
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1; li.pBindings = &b;
        vkCreateDescriptorSetLayout(m_vk.device, &li, nullptr, &m_boneSetLayout);
    }

    // Per-frame bone palette UBOs + descriptor sets
    m_boneUbos.resize(MAX_FRAMES_IN_FLIGHT);
    m_boneMapped.resize(MAX_FRAMES_IN_FLIGHT);
    m_boneDescSets.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_boneUbos[i] = createBuffer(m_vk.physicalDevice, m_vk.device, sizeof(BonePalette),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(m_vk.device, m_boneUbos[i].memory, 0, sizeof(BonePalette), 0,
                    &m_boneMapped[i]);
    }
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_boneSetLayout);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_descriptors.descriptorPool;
        ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        ai.pSetLayouts = layouts.data();
        vkAllocateDescriptorSets(m_vk.device, &ai, m_boneDescSets.data());

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = m_boneUbos[i].buffer;
            bi.range  = sizeof(BonePalette);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_boneDescSets[i]; w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1; w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(m_vk.device, 1, &w, 0, nullptr);
        }
    }

    m_skinnedPipeline = createSkinnedPipeline(m_vk.device, m_offscreen.renderPass,
        m_offscreen.extent,
        m_descriptors.sceneSetLayout, m_descriptors.materialSetLayout, m_boneSetLayout,
        "shaders/mesh_skinned.vert.spv", "shaders/mesh.frag.spv");

    // Post-FX pipelines + descriptor sets
    createPostFXPipelines(m_postFX, m_vk.device, m_bloom, m_ssao, m_composite, m_ldr,
                          m_offscreen, m_swapchain.renderPass, MAX_FRAMES_IN_FLIGHT);

    allocateCommandBuffers(m_renderer, m_vk.device);

    m_scene.createDefaultScene(m_resources);

    m_camera.setAspectRatio(static_cast<float>(m_swapchain.extent.width) /
                            static_cast<float>(m_swapchain.extent.height));

    m_debugUI.init(m_window->getHandle(), m_vk.instance, m_vk.physicalDevice,
                   m_vk.device, m_vk.queueFamilies.graphicsFamily.value(),
                   m_vk.graphicsQueue, m_swapchain.renderPass,
                   static_cast<uint32_t>(m_swapchain.images.size()));

    m_lastFrameTime = glfwGetTime();
    std::cout << "[VulkanEngine] Initialization complete\n";
}

// ── Input ───────────────────────────────────────────────────────────────────

void Engine::processInput(double deltaTime) {
    float dt = static_cast<float>(deltaTime);

    if (m_window->isKeyDown(GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(m_window->getHandle(), GLFW_TRUE);
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantCaptureMouse) return;

    bool tabDown = m_window->isKeyDown(GLFW_KEY_TAB);
    if (tabDown && !m_tabWasDown) {
        m_camera.toggleMode();
        if (m_camera.getMode() == CameraMode::FPS) m_window->setCursorMode(GLFW_CURSOR_DISABLED);
        else                                        m_window->setCursorMode(GLFW_CURSOR_NORMAL);
        m_firstMouse = true;
    }
    m_tabWasDown = tabDown;

    if (m_camera.getMode() == CameraMode::FPS) {
        m_camera.processKeyboard(dt,
            m_window->isKeyDown(GLFW_KEY_W),
            m_window->isKeyDown(GLFW_KEY_S),
            m_window->isKeyDown(GLFW_KEY_A),
            m_window->isKeyDown(GLFW_KEY_D),
            m_window->isKeyDown(GLFW_KEY_SPACE),
            m_window->isKeyDown(GLFW_KEY_LEFT_SHIFT));

        double mx, my; m_window->getCursorPos(mx, my);
        if (m_firstMouse) { m_lastMouseX = mx; m_lastMouseY = my; m_firstMouse = false; }
        float dx = static_cast<float>(mx - m_lastMouseX);
        float dy = static_cast<float>(m_lastMouseY - my);
        m_lastMouseX = mx; m_lastMouseY = my;
        m_camera.processMouseMovement(dx, dy);
    } else {
        bool leftHeld  = glfwGetMouseButton(m_window->getHandle(), GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
        bool rightHeld = glfwGetMouseButton(m_window->getHandle(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (leftHeld || rightHeld) {
            double mx, my; m_window->getCursorPos(mx, my);
            if (m_firstMouse) { m_lastMouseX = mx; m_lastMouseY = my; m_firstMouse = false; }
            float dx = static_cast<float>(mx - m_lastMouseX);
            float dy = static_cast<float>(my - m_lastMouseY);
            m_lastMouseX = mx; m_lastMouseY = my;
            if (leftHeld)  m_camera.processOrbit(dx, dy);
            if (rightHeld) m_camera.processPan(dx, dy);
        } else { m_firstMouse = true; }
    }

    if (m_window->scrollDelta != 0.0f) {
        m_camera.processZoom(-m_window->scrollDelta);
        m_window->scrollDelta = 0.0f;
    }
}

// ── Main loop ───────────────────────────────────────────────────────────────

void Engine::run() {
    while (!m_window->shouldClose()) {
        m_window->pollEvents();

        double currentTime = glfwGetTime();
        double dt64        = currentTime - m_lastFrameTime;
        m_lastFrameTime    = currentTime;
        float  dt          = static_cast<float>(dt64);

        processInput(dt64);
        m_scene.update(dt);

        m_hotReloadTimer += dt;
        if (m_hotReloadTimer >= 2.0f) { m_resources.pollHotReload(); m_hotReloadTimer = 0.0f; }

        m_debugUI.beginFrame();
        m_debugUI.buildUI(m_scene.registry(), m_resources, m_camera, m_shadow,
                          m_postFXSettings,
                          m_visibleEntities, m_totalEntities,
                          dt);
        m_debugUI.endFrame();

        uint32_t frame = m_renderer.currentFrame;

        glm::vec3 dirLightDir = updateLightBuffer(m_lightBuffers, frame, m_scene.registry());

        CascadeUBO cascadeUbo{};
        computeCascades(m_shadow, frame,
                        m_camera.getViewMatrix(), m_camera.getProjectionMatrix(),
                        m_camera.nearClip(), m_camera.farClip(),
                        dirLightDir, cascadeUbo);

        UniformBufferObject ubo{};
        ubo.view      = m_camera.getViewMatrix();
        ubo.proj      = m_camera.getProjectionMatrix();
        ubo.cameraPos = glm::vec4(m_camera.getPosition(), 1.0f);

        // Update post-fx UBOs
        glm::mat4 proj    = m_camera.getProjectionMatrix();
        glm::mat4 invProj = glm::inverse(proj);
        updateSSAOUbo(m_ssao, frame, proj, invProj, m_swapchain.extent, m_postFXSettings);
        updateCompositeUbo(m_composite, frame, m_postFXSettings);

        // Advance animators and update bone palette UBO for this frame
        BonePalette palette{};
        bool anySkinned = false;
        auto animView = m_scene.registry().view<AnimatorComponent>();
        for (auto e : animView) {
            auto& a = animView.get<AnimatorComponent>(e);
            if (a.playing) a.time += dt * a.speed;
            if (!m_skinnedMesh.animations.empty()) {
                int ai = std::min(a.animationIndex,
                                  static_cast<int>(m_skinnedMesh.animations.size()) - 1);
                computeBoneMatrices(m_skinnedMesh.skeleton,
                                    m_skinnedMesh.animations[ai], a.time, palette);
                anySkinned = true;
                break; // single-entity skinned support for now
            }
        }
        if (anySkinned) {
            std::memcpy(m_boneMapped[frame], &palette, sizeof(palette));
        }

        // Build camera frustum for culling
        Frustum frustum = extractFrustum(proj * ubo.view);

        DrawFrameInfo info{};
        info.mainPipeline    = m_pipeline.graphicsPipeline;
        info.mainLayout      = m_pipeline.pipelineLayout;
        info.shadowPipeline  = m_shadowPipeline.graphicsPipeline;
        info.shadowLayout    = m_shadowPipeline.pipelineLayout;
        info.descriptors     = &m_descriptors;
        info.shadow          = &m_shadow;
        info.offscreen       = &m_offscreen;
        info.bloom           = &m_bloom;
        info.ssao            = &m_ssao;
        info.composite       = &m_composite;
        info.ldr             = &m_ldr;
        info.postfx          = &m_postFX;
        info.settings        = &m_postFXSettings;
        info.ubo             = &ubo;
        info.cascadeUbo      = &cascadeUbo;
        info.cameraFrustum   = &frustum;
        info.instances       = &m_instances;
        info.indirect        = &m_indirect;
        info.skinnedMesh     = &m_skinnedMesh;
        info.skinnedPipeline = m_skinnedPipeline.graphicsPipeline;
        info.skinnedLayout   = m_skinnedPipeline.pipelineLayout;
        info.boneDescriptorSet = m_boneDescSets[frame];
        info.jobSystem       = &m_jobs;
        info.visibleEntities = &m_visibleEntities;
        info.totalEntities   = &m_totalEntities;
        info.registry        = &m_scene.registry();
        info.resources       = &m_resources;

        bool needsRecreation = drawFrame(m_renderer, m_swapchain, m_vk.device,
                                          m_vk.graphicsQueue, m_vk.presentQueue,
                                          info, m_window->framebufferResized);
        if (needsRecreation) recreateSwapchain();
    }
    vkDeviceWaitIdle(m_vk.device);
}

// ── Swapchain recreation ────────────────────────────────────────────────────

void Engine::recreateSwapchain() {
    auto extent = m_window->getFramebufferSize();
    while (extent.width == 0 || extent.height == 0) {
        m_window->pollEvents();
        extent = m_window->getFramebufferSize();
    }

    vkDeviceWaitIdle(m_vk.device);

    cleanupSwapchain(m_swapchain, m_vk.device);

    // Tear down post-FX (depend on extent + offscreen image views)
    destroyPostFXPipelines(m_vk.device, m_postFX);
    destroyLdrTarget(m_vk.device, m_ldr);
    destroyOffscreenTarget(m_vk.device, m_offscreen);
    destroyBloomChain(m_vk.device, m_bloom);
    destroySSAO(m_vk.device, m_ssao);
    destroyCompositeData(m_vk.device, m_composite);
    vkDestroyPipeline(m_vk.device, m_pipeline.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_vk.device, m_pipeline.pipelineLayout, nullptr);

    // Recreate swapchain (color-only)
    createSwapchain(m_swapchain, m_vk.physicalDevice, m_vk.device, m_vk.surface,
                    m_vk.queueFamilies.graphicsFamily.value(),
                    m_vk.queueFamilies.presentFamily.value(), extent);
    createImageViews(m_swapchain, m_vk.device);
    createFramebuffers(m_swapchain, m_vk.device);
    createPresentSemaphores(m_swapchain, m_vk.device);

    // Recreate offscreen + bloom + ssao + composite + ldr
    createOffscreenTarget(m_offscreen, m_vk.physicalDevice, m_vk.device,
                          m_swapchain.extent, m_depthFormat);
    createBloomChain(m_bloom, m_vk.physicalDevice, m_vk.device, m_swapchain.extent);
    createSSAO(m_ssao, m_vk.physicalDevice, m_vk.device,
               m_renderer.commandPool, m_vk.graphicsQueue,
               m_swapchain.extent, MAX_FRAMES_IN_FLIGHT);
    createCompositeData(m_composite, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);
    createLdrTarget(m_ldr, m_vk.physicalDevice, m_vk.device, m_swapchain.extent);

    // Recreate main pipeline (its renderpass and viewport baked from offscreen)
    m_pipeline = createGraphicsPipeline(m_vk.device, m_offscreen.renderPass,
                                        m_offscreen.extent,
                                        m_descriptors.sceneSetLayout, m_descriptors.materialSetLayout,
                                        "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");

    // Recreate post-FX pipelines + descriptor sets
    createPostFXPipelines(m_postFX, m_vk.device, m_bloom, m_ssao, m_composite, m_ldr,
                          m_offscreen, m_swapchain.renderPass, MAX_FRAMES_IN_FLIGHT);

    m_camera.setAspectRatio(static_cast<float>(m_swapchain.extent.width) /
                            static_cast<float>(m_swapchain.extent.height));
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

void Engine::cleanup() {
    if (m_vk.device != VK_NULL_HANDLE) vkDeviceWaitIdle(m_vk.device);

    m_debugUI.cleanup(m_vk.device);

    destroyPostFXPipelines(m_vk.device, m_postFX);
    destroyLdrTarget(m_vk.device, m_ldr);
    destroyCompositeData(m_vk.device, m_composite);
    destroySSAO(m_vk.device, m_ssao);
    destroyBloomChain(m_vk.device, m_bloom);
    destroyOffscreenTarget(m_vk.device, m_offscreen);

    vkDestroyDescriptorPool(m_vk.device, m_descriptors.descriptorPool, nullptr);
    for (auto& ub : m_descriptors.uniformBuffers) destroyBuffer(m_vk.device, ub);

    destroyLightBuffers(m_vk.device, m_lightBuffers);
    destroyInstanceBuffer(m_vk.device, m_instances);
    destroyIndirectBuffer(m_vk.device, m_indirect);
    destroyShadowResources(m_vk.device, m_shadow);

    // Skeletal
    for (auto& b : m_boneUbos) destroyBuffer(m_vk.device, b);
    m_boneUbos.clear(); m_boneMapped.clear(); m_boneDescSets.clear();
    if (m_boneSetLayout) vkDestroyDescriptorSetLayout(m_vk.device, m_boneSetLayout, nullptr);
    if (m_skinnedPipeline.graphicsPipeline)
        vkDestroyPipeline(m_vk.device, m_skinnedPipeline.graphicsPipeline, nullptr);
    if (m_skinnedPipeline.pipelineLayout)
        vkDestroyPipelineLayout(m_vk.device, m_skinnedPipeline.pipelineLayout, nullptr);
    destroySkinnedMesh(m_vk.device, m_skinnedMesh);

    m_resources.cleanup();

    vkDestroyCommandPool(m_vk.device, m_renderer.commandPool, nullptr);

    vkDestroyPipeline(m_vk.device, m_shadowPipeline.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_vk.device, m_shadowPipeline.pipelineLayout, nullptr);
    vkDestroyPipeline(m_vk.device, m_pipeline.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_vk.device, m_pipeline.pipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(m_vk.device, m_descriptors.materialSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_vk.device, m_descriptors.sceneSetLayout, nullptr);

    destroyPresentSemaphores(m_swapchain, m_vk.device);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(m_vk.device, m_swapchain.imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(m_vk.device, m_swapchain.inFlightFences[i], nullptr);
    }

    for (auto fb : m_swapchain.framebuffers) vkDestroyFramebuffer(m_vk.device, fb, nullptr);
    vkDestroyRenderPass(m_vk.device, m_swapchain.renderPass, nullptr);
    for (auto iv : m_swapchain.imageViews) vkDestroyImageView(m_vk.device, iv, nullptr);
    vkDestroySwapchainKHR(m_vk.device, m_swapchain.swapchain, nullptr);

    vkDestroyDevice(m_vk.device, nullptr);
    vkDestroySurfaceKHR(m_vk.instance, m_vk.surface, nullptr);
    destroyDebugMessenger(m_vk.instance, m_vk.debugMessenger);
    vkDestroyInstance(m_vk.instance, nullptr);

    delete m_window;
    m_window = nullptr;
    std::cout << "[VulkanEngine] Cleanup complete\n";
}
