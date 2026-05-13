#include "engine.h"
#include "frustum.h"
#include "dlss.h"

#include <imgui.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>

// ── Init ────────────────────────────────────────────────────────────────────

void Engine::init() {
    m_window = new Window(1280, 720, "VulkanEngine");

    auto extensions = m_window->getRequiredExtensions();

    // NGX needs specific instance extensions to be enabled for DLSS. Query
    // them before instance creation and merge. Safe if DLSS is unavailable —
    // returns an empty list.
    auto dlssInstanceExts = dlssRequiredInstanceExtensions();
    for (const char* e : dlssInstanceExts) {
        bool present = false;
        for (const char* x : extensions) if (std::strcmp(x, e) == 0) { present = true; break; }
        if (!present) extensions.push_back(e);
    }

    m_vk.instance       = createInstance(extensions);
    m_vk.debugMessenger = setupDebugMessenger(m_vk.instance);
    m_vk.surface        = m_window->createSurface(m_vk.instance);
    m_vk.physicalDevice = pickPhysicalDevice(m_vk.instance, m_vk.surface);
    m_vk.queueFamilies  = findQueueFamilies(m_vk.physicalDevice, m_vk.surface);

    // Same dance for device extensions — must be queried after VkInstance +
    // physical device exist and passed into device creation.
    auto dlssDeviceExts = dlssRequiredDeviceExtensions(m_vk.instance, m_vk.physicalDevice);

    m_vk.device         = createLogicalDevice(m_vk.physicalDevice, m_vk.queueFamilies,
                                              dlssDeviceExts);
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.graphicsFamily.value(), 0, &m_vk.graphicsQueue);
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.presentFamily.value(),  0, &m_vk.presentQueue);

    // DLSS init — capability query only. No upscale path yet (phases 4b/4c).
    dlssInit(m_vk.instance, m_vk.physicalDevice, m_vk.device);

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
    createShadowResources(m_shadow, m_vk.physicalDevice, m_vk.device,
                          m_renderer.commandPool, m_vk.graphicsQueue,
                          MAX_FRAMES_IN_FLIGHT);

    // Light buffers
    createLightBuffers(m_lightBuffers, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT);

    // GPU-driven culling buffers. Main-instance capacity grew because each
    // batch over-reserves lodCount × instanceCount slots (worst case all
    // instances pick the same LOD). Indirect grew because each batch now
    // owns lodCount + 1 cmds (per-LOD main + 1 shadow).
    constexpr uint32_t MAX_INSTANCES_PER_FRAME = 16384;
    constexpr uint32_t MAX_BATCHES_PER_FRAME   = 1024;
    createCandidateBuffer(m_candidates, m_vk.physicalDevice, m_vk.device,
                          MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME);
    createBatchHeaderBuffer(m_batchHeaders, m_vk.physicalDevice, m_vk.device,
                            MAX_FRAMES_IN_FLIGHT, MAX_BATCHES_PER_FRAME);
    createInstanceBuffer(m_mainInstances,   m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME * MAX_LOD);
    createInstanceBuffer(m_shadowInstances, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME);
    createIndirectBuffer(m_indirect, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_BATCHES_PER_FRAME * (MAX_LOD + 1));

    // Compute-cull pipeline + per-frame UBO. Pointed at buffers after HZB is
    // built (cull descriptor set includes the HZB sampler binding).
    createGpuCull(m_gpuCull, m_vk.physicalDevice, m_vk.device, MAX_FRAMES_IN_FLIGHT,
                  "shaders/cull.comp.spv");

    // HZB: occlusion-culling pyramid built from the previous frame's depth
    // attachment. Created after offscreen so depthView/depthSampler exist.
    // HZB mip 0 is HALF the depth resolution — hzb_reduce.comp does a 2×2
    // max-pool with `srcBase = dst * 2`, which only works if the source is
    // twice the destination size. Passing the full depth extent here would
    // make ~75% of mip 0 clamp to the depth attachment's bottom-right pixel,
    // contaminating later mips and causing screen-position-dependent
    // false-occlusion in the cull pass.
    VkExtent2D hzbExtent = {
        std::max(1u, m_offscreen.extent.width  / 2u),
        std::max(1u, m_offscreen.extent.height / 2u),
    };
    createHzb(m_hzb, m_vk.physicalDevice, m_vk.device,
              m_renderer.commandPool, m_vk.graphicsQueue,
              hzbExtent,
              m_offscreen.depthView, m_offscreen.depthSampler,
              "shaders/hzb_reduce.comp.spv");

    writeGpuCullDescriptors(m_gpuCull, m_vk.device,
                            m_candidates, m_batchHeaders,
                            m_mainInstances, m_shadowInstances, m_indirect,
                            m_hzb.fullView, m_hzb.sampler);

    // Ray tracing scaffolding. Phase 1a: just sizes per-frame containers.
    // No BLAS/TLAS exist yet — those come in Phase 1b.
    createRtScene(m_rtScene, m_vk.device, MAX_FRAMES_IN_FLIGHT);

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
                          m_postFXSettings, m_rtSettings, m_dlssSettings,
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

        // DLSS sub-pixel jitter — applied directly to the projection matrix.
        // For column-major GLM, proj[2][0] / proj[2][1] are the X / Y
        // translation entries of clip space. Adding `j * 2 / extent` shifts
        // every vertex by exactly `j` pixels in screen space (NDC is [-1,1]
        // across the full extent). Jitter is gated by DLSS enable so it's
        // invisible until you toggle DLSS on.
        glm::vec2 jitterNDC(0.0f);
        if (m_dlssSettings.enabled && m_dlssSettings.jitterEnabled) {
            glm::vec2 j = haltonJitter(m_haltonIndex);
            m_haltonIndex = (m_haltonIndex % 8u) + 1u; // 8-frame cycle
            jitterNDC = glm::vec2(j.x * 2.0f / float(m_swapchain.extent.width),
                                  j.y * 2.0f / float(m_swapchain.extent.height));
            ubo.proj[2][0] += jitterNDC.x;
            ubo.proj[2][1] += jitterNDC.y;
        }
        ubo.jitterOffset = glm::vec4(jitterNDC, 0.0f, 0.0f);
        ubo.prevViewProj = m_dlssPrevViewProj;
        // RT shadow uniform — feeds the gated branch in mesh.frag. Setting x=0
        // makes the shader take the CSM path (and never touches binding 4).
        const bool rtShadowsActive = RT_SUPPORTED && m_rtSettings.enabled && m_rtSettings.shadows;
        // x=enabled, y=softness, z=samples, w=sunOnly flag
        ubo.rtParams  = glm::vec4(rtShadowsActive ? 1.0f : 0.0f,
                                  m_rtSettings.shadowSoftness,
                                  static_cast<float>(m_rtSettings.shadowSamples),
                                  m_rtSettings.sunOnly ? 1.0f : 0.0f);
        const bool rtReflectionsActive = RT_SUPPORTED && m_rtSettings.enabled && m_rtSettings.reflections;
        ubo.rtParams2 = glm::vec4(rtReflectionsActive ? 1.0f : 0.0f,
                                  static_cast<float>(m_rtSettings.reflectionSamples),
                                  m_rtSettings.reflectionMaxDist,
                                  m_rtSettings.reflectionIntensity);
        const bool rtGiActive = RT_SUPPORTED && m_rtSettings.enabled && m_rtSettings.gi;
        ubo.rtParams3 = glm::vec4(rtGiActive ? 1.0f : 0.0f,
                                  static_cast<float>(m_rtSettings.giSamples),
                                  m_rtSettings.giMaxDist,
                                  m_rtSettings.giIntensity);

        // Update post-fx UBOs.
        //
        // When RT shadows are active, suppress SSAO too. SSAO darkens
        // crevices using screen-space depth derivatives — its dark halo at
        // contact points reads as a second shadow stacked on the RT shadow.
        // Phase 3 (ray-traced GI) will replace SSAO with proper indirect
        // occlusion; until then we hide the conflict by temporarily forcing
        // SSAO off when RT shadows are on. Restoring the user's choice when
        // RT is off.
        PostFXSettings effectiveFx = m_postFXSettings;
        if (rtShadowsActive) effectiveFx.ssaoEnabled = false;

        glm::mat4 proj    = m_camera.getProjectionMatrix();
        glm::mat4 invProj = glm::inverse(proj);
        updateSSAOUbo(m_ssao, frame, proj, invProj, m_swapchain.extent, effectiveFx);
        updateCompositeUbo(m_composite, frame, effectiveFx);

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

        // Build camera frustum + CullParamsUBO for the GPU cull dispatch.
        glm::mat4 currViewProj = proj * ubo.view;
        Frustum   frustum      = extractFrustum(currViewProj);

        CullParamsUBO cullParams{};
        cullParams.prevViewProj = m_hasPrevVP ? m_prevViewProj : currViewProj;
        for (int i = 0; i < 6; ++i) cullParams.frustumPlanes[i] = frustum.planes[i];
        cullParams.cameraPos        = glm::vec4(m_camera.getPosition(), 1.0f);
        cullParams.hzbSizeMipCount  = glm::vec4(static_cast<float>(m_hzb.extent.width),
                                                 static_cast<float>(m_hzb.extent.height),
                                                 static_cast<float>(m_hzb.mipCount),
                                                 m_hasPrevVP ? 1.0f : 0.0f);
        cullParams.numCandidates    = 0;   // set by drawFrame after CPU build pass

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
        info.cullParams      = &cullParams;
        info.hzb             = &m_hzb;
        info.candidates      = &m_candidates;
        info.batchHeaders    = &m_batchHeaders;
        info.mainInstances   = &m_mainInstances;
        info.shadowInstances = &m_shadowInstances;
        info.indirect        = &m_indirect;
        info.gpuCull         = &m_gpuCull;
        info.rtScene         = &m_rtScene;
        info.rtSettings      = &m_rtSettings;
        info.physicalDevice  = m_vk.physicalDevice;
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
        if (needsRecreation) {
            recreateSwapchain();
        } else {
            // Save this frame's view-proj so the next frame's compute cull can
            // project AABBs into the HZB the GPU just built.
            m_prevViewProj = currViewProj;
            m_hasPrevVP    = true;
            // Also save the *jittered* view-proj for DLSS motion vectors.
            // We use the same `currViewProj` for both purposes — the cull
            // pass doesn't care about jitter (sub-pixel) but motion vectors
            // need the actual rasterisation matrix.
            m_dlssPrevViewProj = ubo.proj * ubo.view;
        }
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

    // Tear down post-FX (depend on extent + offscreen image views). HZB also
    // depends on the offscreen depthView, so it must be destroyed before
    // offscreen and rebuilt after.
    destroyPostFXPipelines(m_vk.device, m_postFX);
    destroyLdrTarget(m_vk.device, m_ldr);
    destroyHzb(m_vk.device, m_hzb);
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

    // Rebuild HZB (depends on offscreen depth + extent) and re-point the cull
    // descriptor sets at the new HZB views. Half-extent — see note at the
    // initial createHzb call.
    VkExtent2D hzbExtent = {
        std::max(1u, m_offscreen.extent.width  / 2u),
        std::max(1u, m_offscreen.extent.height / 2u),
    };
    createHzb(m_hzb, m_vk.physicalDevice, m_vk.device,
              m_renderer.commandPool, m_vk.graphicsQueue,
              hzbExtent,
              m_offscreen.depthView, m_offscreen.depthSampler,
              "shaders/hzb_reduce.comp.spv");
    writeGpuCullDescriptors(m_gpuCull, m_vk.device,
                            m_candidates, m_batchHeaders,
                            m_mainInstances, m_shadowInstances, m_indirect,
                            m_hzb.fullView, m_hzb.sampler);
    // Throw away the previous view-proj — the camera's aspect ratio is about
    // to change, and an HZB just cleared to 1.0 wouldn't match it anyway.
    m_hasPrevVP = false;

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
    destroyRtScene(m_vk.device, m_rtScene);
    destroyHzb(m_vk.device, m_hzb);
    destroyGpuCull(m_vk.device, m_gpuCull);
    destroyCandidateBuffer(m_vk.device, m_candidates);
    destroyBatchHeaderBuffer(m_vk.device, m_batchHeaders);
    destroyInstanceBuffer(m_vk.device, m_mainInstances);
    destroyInstanceBuffer(m_vk.device, m_shadowInstances);
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

    dlssShutdown(m_vk.device);

    vkDestroyDevice(m_vk.device, nullptr);
    vkDestroySurfaceKHR(m_vk.instance, m_vk.surface, nullptr);
    destroyDebugMessenger(m_vk.instance, m_vk.debugMessenger);
    vkDestroyInstance(m_vk.instance, nullptr);

    delete m_window;
    m_window = nullptr;
    std::cout << "[VulkanEngine] Cleanup complete\n";
}
