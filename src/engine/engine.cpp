#include "engine.h"
#include "frustum.h"
#include "gltf_loader.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>

#include <glm/gtc/matrix_transform.hpp>

namespace {
    // Recompile one GLSL file to SPIR-V with glslc, matching the build's
    // flags (--target-env=vulkan1.2). Tries $VULKAN_SDK\Bin\glslc.exe then
    // glslc on PATH. Returns true on a zero exit code.
    bool recompileShader(const std::string& src, const std::string& outSpv) {
        std::vector<std::string> glslcs;
        if (const char* sdk = std::getenv("VULKAN_SDK"))
            glslcs.push_back(std::string(sdk) + "\\Bin\\glslc.exe");
        glslcs.push_back("glslc");
        for (const auto& g : glslcs) {
            // Whole command is wrapped in quotes for cmd.exe because the
            // program path and the repo path both contain spaces.
            std::string cmd = "\"\"" + g + "\" --target-env=vulkan1.2 \"" +
                              src + "\" -o \"" + outSpv + "\"\"";
            if (std::system(cmd.c_str()) == 0) return true;
        }
        return false;
    }
}

// ── Init ────────────────────────────────────────────────────────────────────

void Engine::init() {
    m_window = std::make_unique<Window>(1280, 720, "VulkanEngine");

    auto extensions = m_window->getRequiredExtensions();

    m_vk.instance       = createInstance(extensions);
    m_vk.debugMessenger = setupDebugMessenger(m_vk.instance);
    m_vk.surface        = m_window->createSurface(m_vk.instance);
    m_vk.physicalDevice = pickPhysicalDevice(m_vk.instance, m_vk.surface);
    m_vk.queueFamilies  = findQueueFamilies(m_vk.physicalDevice, m_vk.surface);

    m_vk.device         = createLogicalDevice(m_vk.physicalDevice, m_vk.queueFamilies, {});
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.graphicsFamily.value(), 0, &m_vk.graphicsQueue);
    vkGetDeviceQueue(m_vk.device, m_vk.queueFamilies.presentFamily.value(),  0, &m_vk.presentQueue);

    // VMA allocator. Must be ready before any buffer/image is created, but
    // after the logical device since VMA caches device-level function pointers.
    createVmaAllocator(m_vk.instance, m_vk.physicalDevice, m_vk.device);

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

    // Offscreen HDR target (main pass → here).
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

    // GPU-driven culling buffers. Phase 2.2 (two-pass occlusion) doubles the
    // per-LOD reservations: each batch has separate pass-0 and pass-1 ranges
    // in the main instance buffer, and 2*lodCount + 1 indirect cmds (pass-0
    // + pass-1 main + shadow).
    using engine_config::MAX_INSTANCES_PER_FRAME;
    using engine_config::MAX_BATCHES_PER_FRAME;
    createCandidateBuffer(m_candidates, m_vk.physicalDevice, m_vk.device,
                          MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME);
    createBatchHeaderBuffer(m_batchHeaders, m_vk.physicalDevice, m_vk.device,
                            MAX_FRAMES_IN_FLIGHT, MAX_BATCHES_PER_FRAME);
    createInstanceBuffer(m_mainInstances,   m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME * MAX_LOD * 2);
    createInstanceBuffer(m_shadowInstances, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_INSTANCES_PER_FRAME);
    createIndirectBuffer(m_indirect, m_vk.physicalDevice, m_vk.device,
                         MAX_FRAMES_IN_FLIGHT, MAX_BATCHES_PER_FRAME * (2 * MAX_LOD + 1));

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

    // ── IBL probes ────────────────────────────────────────────────────────
    // Create the env / irradiance / prefilter / BRDF-LUT resources, bake them
    // from the procedural sky, then point all per-frame scene sets at them.
    // bakeIbl re-runs cheaply later if the user picks a different HDR.
    createIbl(m_ibl, m_vk.physicalDevice, m_vk.device);
    bakeIbl(m_ibl, m_vk.physicalDevice, m_vk.device,
            m_renderer.commandPool, m_vk.graphicsQueue, m_iblParams);
    for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        writeSceneIbl(m_descriptors, m_vk.device, f,
                      m_ibl.irradianceView, m_ibl.prefilterView, m_ibl.brdfLutView,
                      m_ibl.cubeSampler, m_ibl.lutSampler);
    }
    m_iblRebuildRequested = false;

    // Resource manager (default mesh + texture)
    m_resources.init(m_vk.physicalDevice, m_vk.device,
                     m_renderer.commandPool, m_vk.graphicsQueue, m_descriptors);

    // Main pipeline targets the OFFSCREEN render pass (not the swapchain)
    m_pipeline = createGraphicsPipeline(m_vk.device, m_offscreen.renderPass,
                                        m_offscreen.extent,
                                        m_descriptors.sceneSetLayout, m_descriptors.materialSetLayout,
                                        "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");

    // Sky pipeline — shares the offscreen render pass; drawn after geometry
    // inside the same pass.
    createSkyPipeline(m_sky, m_vk.device, m_offscreen.renderPass, m_offscreen.extent,
                      m_descriptors.sceneSetLayout,
                      "shaders/sky.vert.spv", "shaders/sky.frag.spv");

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
        m_boneMapped[i] = m_boneUbos[i].mapped;
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

    // Skinned shadow pipeline — same shadow render pass as the static
    // shadow pipeline; bone palette set + (lightVP, model) push constants.
    m_skinnedShadowPipeline = createSkinnedShadowPipeline(m_vk.device,
        m_shadow.renderPass, SHADOW_MAP_SIZE, m_boneSetLayout,
        "shaders/shadow_skinned.vert.spv");

    // Post-FX pipelines + descriptor sets
    createPostFXPipelines(m_postFX, m_vk.device, m_bloom, m_ssao, m_composite, m_ldr,
                          m_offscreen, m_swapchain.renderPass, MAX_FRAMES_IN_FLIGHT);

    allocateCommandBuffers(m_renderer, m_vk.device);

    m_scene.createDefaultScene(m_resources);

    // Physics: init the Jolt world, then create bodies for any scene entities
    // that already have a RigidBodyComponent (the default scene's floor + the
    // falling-box demo).
    m_physics.init();
    m_physics.syncNewBodies(m_scene.registry());

    // Audio: OpenAL device/context. Listener is driven by the camera each
    // frame; emitters come from AudioSourceComponent entities.
    m_audio.init();

    // Release backing Jolt/OpenAL objects when an entity (or its component)
    // is destroyed, e.g. via the Hierarchy "Delete" button. Without this the
    // OpenAL source keeps playing and the Jolt body leaks. on_destroy fires
    // while the component is still readable, so removeBody/removeSource can
    // read the stored handle.
    {
        auto& reg = m_scene.registry();
        reg.on_destroy<RigidBodyComponent>()
            .connect<&PhysicsWorld::removeBody>(m_physics);
        reg.on_destroy<AudioSourceComponent>()
            .connect<&AudioEngine::removeSource>(m_audio);
        reg.on_destroy<ScriptComponent>()
            .connect<&ScriptEngine::onDestroy>(m_scripting);
    }

    // Scripting: Lua via sol2. spawn_cube() in scripts routes here so
    // scripting needn't know about ResourceManager.
    m_scripting.init(
        m_window->getHandle(), &m_physics,
        [this](const glm::vec3& p) -> entt::entity {
            auto& reg = m_scene.registry();
            auto e = reg.create();
            reg.emplace<NameComponent>(e, std::string("ScriptCube"));
            TransformComponent t{};
            t.position = p;
            reg.emplace<TransformComponent>(e, t);
            reg.emplace<MeshComponent>(e, m_resources.getDefaultCube());
            MaterialComponent mat{};
            mat.texture   = m_resources.getDefaultTexture();
            mat.color     = glm::vec4(0.6f, 0.8f, 1.0f, 1.0f);
            mat.metallic  = 0.0f;
            mat.roughness = 0.5f;
            reg.emplace<MaterialComponent>(e, mat);
            return e;
        });

    m_camera.setAspectRatio(static_cast<float>(m_swapchain.extent.width) /
                            static_cast<float>(m_swapchain.extent.height));

    m_debugUI.init(m_window->getHandle(), m_vk.instance, m_vk.physicalDevice,
                   m_vk.device, m_vk.queueFamilies.graphicsFamily.value(),
                   m_vk.graphicsQueue, m_swapchain.renderPass,
                   static_cast<uint32_t>(m_swapchain.images.size()));

    // Profiler: GPU timestamps + CPU scopes. Independent of any GPU work; safe
    // to init last. Per-frame query pools sized for MAX_FRAMES_IN_FLIGHT.
    m_profiler.init(m_vk.device, m_vk.physicalDevice, MAX_FRAMES_IN_FLIGHT);

    // Seed undo history with the initial scene state.
    m_undo.seed(m_scene.registry());

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
        {
            PROFILE_CPU(m_profiler, "scene update");
            m_scene.update(dt);
        }
        {
            PROFILE_CPU(m_profiler, "scripts");
            m_scripting.update(m_scene.registry(), dt, currentTime);
        }
        {
            PROFILE_CPU(m_profiler, "physics");
            m_physics.syncNewBodies(m_scene.registry());
            m_physics.step(m_scene.registry(), dt);
        }
        {
            PROFILE_CPU(m_profiler, "audio");
            glm::vec3 camPos = m_camera.getPosition();
            glm::vec3 camFwd = m_camera.getForward();
            float invDt = (dt > 1e-4f) ? (1.0f / dt) : 0.0f;
            glm::vec3 camVel = m_hasPrevCamPos
                ? (camPos - m_prevCamPos) * invDt : glm::vec3(0.0f);
            m_prevCamPos = camPos;
            m_hasPrevCamPos = true;
            m_audio.update(m_scene.registry(), camPos, camFwd,
                           glm::vec3(0.0f, 1.0f, 0.0f), camVel);
        }

        m_hotReloadTimer += dt;
        // Subtract the interval instead of zeroing — zeroing accumulates
        // float drift over many frames and can skip a cycle.
        if (m_hotReloadTimer >= 2.0f) { m_resources.pollHotReload(); m_hotReloadTimer -= 2.0f; }

        m_debugUI.beginFrame();
        m_debugUI.buildUI(m_scene.registry(), m_resources, m_camera, m_shadow,
                          m_postFXSettings, m_rtSettings,
                          m_visibleEntities, m_totalEntities,
                          dt, &m_profiler, &m_skinnedMesh,
                          &m_iblParams, &m_iblRebuildRequested,
                          &m_pendingGltfLoad, &m_physics, &m_audio,
                          &m_scripting, &m_undo,
                          &m_shaderSrcDir, &m_shaderReloadRequested);

        if (!m_pendingGltfLoad.empty()) {
            vkDeviceWaitIdle(m_vk.device);
            SkinnedMesh fresh{};
            std::string err;
            if (loadGltfSkinned(m_pendingGltfLoad, fresh, &err)) {
                destroySkinnedMesh(m_vk.device, m_skinnedMesh);
                m_skinnedMesh = std::move(fresh);
                uploadSkinnedMesh(m_skinnedMesh, m_vk.physicalDevice, m_vk.device,
                                  m_renderer.commandPool, m_vk.graphicsQueue);
                std::cout << "[glTF] Loaded skinned mesh from " << m_pendingGltfLoad
                          << " — " << m_skinnedMesh.vertices.size() << " verts, "
                          << m_skinnedMesh.skeleton.joints.size() << " joints, "
                          << m_skinnedMesh.animations.size() << " anims.\n";
                // Reset all animators so they replay the new clip from t=0
                // rather than pointing at a clip index that no longer exists
                // or running at a time past the new clip's duration.
                auto av = m_scene.registry().view<AnimatorComponent>();
                for (auto e : av) {
                    auto& a = av.get<AnimatorComponent>(e);
                    a.animationIndex = 0;
                    a.time = 0.0f;
                }
            } else {
                std::cerr << "[glTF] Load failed: " << err << "\n";
            }
            m_pendingGltfLoad.clear();
        }

        // Honor an IBL rebuild request from the UI before recording the
        // frame. vkDeviceWaitIdle keeps it safe — the rebake re-records on
        // a single-time command buffer and re-writes the per-frame scene
        // descriptors. Infrequent: only when the user pushes Rebake.
        if (m_iblRebuildRequested) {
            vkDeviceWaitIdle(m_vk.device);
            bakeIbl(m_ibl, m_vk.physicalDevice, m_vk.device,
                    m_renderer.commandPool, m_vk.graphicsQueue, m_iblParams);
            for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
                writeSceneIbl(m_descriptors, m_vk.device, f,
                              m_ibl.irradianceView, m_ibl.prefilterView, m_ibl.brdfLutView,
                              m_ibl.cubeSampler, m_ibl.lutSampler);
            }
            m_iblRebuildRequested = false;
        }

        // Hot shader reload: recompile the mesh GLSL and rebuild the main
        // pipeline live. vkDeviceWaitIdle makes the destroy/recreate safe
        // (same approach as the IBL rebake / glTF swap above). Scope: the
        // main mesh pipeline (mesh.vert + mesh.frag) — the common edit case.
        if (m_shaderReloadRequested) {
            m_shaderReloadRequested = false;
            bool ok =
                recompileShader(m_shaderSrcDir + "/mesh.vert",
                                "shaders/mesh.vert.spv") &&
                recompileShader(m_shaderSrcDir + "/mesh.frag",
                                "shaders/mesh.frag.spv");
            if (ok) {
                vkDeviceWaitIdle(m_vk.device);
                vkDestroyPipeline(m_vk.device, m_pipeline.graphicsPipeline,
                                  nullptr);
                vkDestroyPipelineLayout(m_vk.device, m_pipeline.pipelineLayout,
                                        nullptr);
                m_pipeline = createGraphicsPipeline(
                    m_vk.device, m_offscreen.renderPass, m_offscreen.extent,
                    m_descriptors.sceneSetLayout,
                    m_descriptors.materialSetLayout,
                    "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");
                std::cout << "[Shaders] mesh pipeline reloaded from "
                          << m_shaderSrcDir << "\n";
            } else {
                std::cerr << "[Shaders] recompile failed — is glslc on PATH "
                             "or VULKAN_SDK set? Check the console for glslc "
                             "errors.\n";
            }
        }
        m_debugUI.endFrame();

        uint32_t frame = m_renderer.currentFrame;

        glm::vec3 dirLightDir;
        {
            PROFILE_CPU(m_profiler, "updateLightBuffer");
            dirLightDir = updateLightBuffer(m_lightBuffers, frame, m_scene.registry());
        }

        CascadeUBO cascadeUbo{};
        {
            PROFILE_CPU(m_profiler, "computeCascades");
            computeCascades(m_shadow, frame,
                            m_camera.getViewMatrix(), m_camera.getProjectionMatrix(),
                            m_camera.nearClip(), m_camera.farClip(),
                            dirLightDir, cascadeUbo);
        }

        UniformBufferObject ubo{};
        ubo.view      = m_camera.getViewMatrix();
        ubo.proj      = m_camera.getProjectionMatrix();
        ubo.cameraPos = glm::vec4(m_camera.getPosition(), 1.0f);

        // Motion-vector scaffolding. No jitter applied — the engine renders
        // un-jittered for now. A future TAA / Streamline pass will fill in
        // the jitter when needed.
        ubo.jitterOffset = glm::vec4(0.0f);
        ubo.prevViewProj = m_prevViewProjMotion;
        // RT shadow uniform — feeds the gated branch in mesh.frag. Setting x=0
        // makes the shader take the CSM path (and never touches binding 4).
        const bool rtShadowsActive = RT_SUPPORTED && m_rtSettings.enabled && m_rtSettings.shadows;
        // `sunOnly` matters only when at least one directional light exists.
        // Otherwise the shader would skip every light's shadow ray (it only
        // shadows the "first directional"), leaving the scene flat. Detect a
        // directional-light-less scene and force per-light RT shadows so
        // point/spot lights still cast.
        bool anyDirectional = false;
        {
            auto dirView = m_scene.registry().view<DirectionalLightComponent>();
            anyDirectional = dirView.begin() != dirView.end();
        }
        // sunOnly only applies when there IS a directional light. With no
        // directional in the scene, the shader's "shadow only the first
        // directional" branch would skip every light's shadow ray and the
        // scene would render unshadowed — fall through to per-light RT instead.
        const bool effectiveSunOnly = m_rtSettings.sunOnly && anyDirectional;
        // x=enabled, y=softness, z=samples, w=sunOnly flag
        ubo.rtParams  = glm::vec4(rtShadowsActive ? 1.0f : 0.0f,
                                  m_rtSettings.shadowSoftness,
                                  static_cast<float>(m_rtSettings.shadowSamples),
                                  effectiveSunOnly ? 1.0f : 0.0f);
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
        // DoF depth-linearisation needs the camera's near/far each frame —
        // the camera owns the canonical values, so plumb them through.
        effectiveFx.nearClip = m_camera.nearClip();
        effectiveFx.farClip  = m_camera.farClip();

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
            if (!m_skinnedMesh.animations.empty()) {
                // Clamp on BOTH ends — a negative animationIndex slips past
                // std::min unchecked and indexes the array out of bounds.
                int ai = std::clamp(a.animationIndex, 0,
                                    static_cast<int>(m_skinnedMesh.animations.size()) - 1);
                const float dur = m_skinnedMesh.animations[ai].duration;
                if (a.playing) {
                    a.time += dt * a.speed;
                    if (dur > 0.0f) {
                        if (a.loop) {
                            // fmod can return negative for reversed playback; rebias to [0, dur).
                            a.time = std::fmod(a.time, dur);
                            if (a.time < 0.0f) a.time += dur;
                        } else if (a.time >= dur) {
                            a.time   = dur;
                            a.playing = false;
                        } else if (a.time < 0.0f) {
                            a.time   = 0.0f;
                            a.playing = false;
                        }
                    }
                }
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
        info.skinnedPipeline       = m_skinnedPipeline.graphicsPipeline;
        info.skinnedLayout         = m_skinnedPipeline.pipelineLayout;
        info.skinnedShadowPipeline = m_skinnedShadowPipeline.graphicsPipeline;
        info.skinnedShadowLayout   = m_skinnedShadowPipeline.pipelineLayout;
        info.sky                   = &m_sky;
        info.boneDescriptorSet = m_boneDescSets[frame];
        info.jobSystem       = &m_jobs;
        info.visibleEntities = &m_visibleEntities;
        info.totalEntities   = &m_totalEntities;
        info.registry        = &m_scene.registry();
        info.resources       = &m_resources;
        info.prevTransforms  = &m_prevTransforms;
        info.profiler        = &m_profiler;

        bool needsRecreation = drawFrame(m_renderer, m_swapchain, m_vk.device,
                                          m_vk.graphicsQueue, m_vk.presentQueue,
                                          info, m_window->framebufferResized);
        if (needsRecreation) {
            // The projection (and HZB extent) is about to change — invalidate
            // the prev-frame view-proj so the next frame doesn't reproject
            // AABBs into an HZB built at a different resolution.
            m_hasPrevVP = false;
            recreateSwapchain();
        } else {
            // Save this frame's view-proj so the next frame's compute cull can
            // project AABBs into the HZB the GPU just built.
            m_prevViewProj = currViewProj;
            m_hasPrevVP    = true;
            // Save the un-jittered view-proj for the next frame's motion
            // vectors (kept for future TAA / Streamline integration).
            m_prevViewProjMotion = currViewProj;
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
    destroySkyPipeline(m_vk.device, m_sky);

    // Recreate swapchain (color-only)
    createSwapchain(m_swapchain, m_vk.physicalDevice, m_vk.device, m_vk.surface,
                    m_vk.queueFamilies.graphicsFamily.value(),
                    m_vk.queueFamilies.presentFamily.value(), extent);
    createImageViews(m_swapchain, m_vk.device);
    createFramebuffers(m_swapchain, m_vk.device);
    createPresentSemaphores(m_swapchain, m_vk.device);

    // Recreate offscreen + post-fx chain at the new swapchain extent.
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
    createSkyPipeline(m_sky, m_vk.device, m_offscreen.renderPass, m_offscreen.extent,
                      m_descriptors.sceneSetLayout,
                      "shaders/sky.vert.spv", "shaders/sky.frag.spv");

    // Recreate post-FX pipelines + descriptor sets
    createPostFXPipelines(m_postFX, m_vk.device, m_bloom, m_ssao, m_composite, m_ldr,
                          m_offscreen, m_swapchain.renderPass, MAX_FRAMES_IN_FLIGHT);

    m_camera.setAspectRatio(static_cast<float>(m_swapchain.extent.width) /
                            static_cast<float>(m_swapchain.extent.height));
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

void Engine::cleanup() {
    if (m_vk.device != VK_NULL_HANDLE) vkDeviceWaitIdle(m_vk.device);

    // Physics + audio + scripting are CPU-only; tear them down first.
    m_scripting.cleanup();
    m_physics.cleanup();
    m_audio.cleanup();

    m_profiler.shutdown(m_vk.device);
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
    if (m_skinnedShadowPipeline.graphicsPipeline)
        vkDestroyPipeline(m_vk.device, m_skinnedShadowPipeline.graphicsPipeline, nullptr);
    if (m_skinnedShadowPipeline.pipelineLayout)
        vkDestroyPipelineLayout(m_vk.device, m_skinnedShadowPipeline.pipelineLayout, nullptr);
    destroySkinnedMesh(m_vk.device, m_skinnedMesh);

    m_resources.cleanup();

    vkDestroyCommandPool(m_vk.device, m_renderer.commandPool, nullptr);

    vkDestroyPipeline(m_vk.device, m_shadowPipeline.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_vk.device, m_shadowPipeline.pipelineLayout, nullptr);
    vkDestroyPipeline(m_vk.device, m_pipeline.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_vk.device, m_pipeline.pipelineLayout, nullptr);
    destroySkyPipeline(m_vk.device, m_sky);
    destroyIbl(m_vk.device, m_ibl);

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

    // VMA allocator owns sub-allocations against the device — release before
    // the device handle goes away.
    destroyVmaAllocator();

    vkDestroyDevice(m_vk.device, nullptr);
    vkDestroySurfaceKHR(m_vk.instance, m_vk.surface, nullptr);
    destroyDebugMessenger(m_vk.instance, m_vk.debugMessenger);
    vkDestroyInstance(m_vk.instance, nullptr);

    m_window.reset();
    std::cout << "[VulkanEngine] Cleanup complete\n";
}
