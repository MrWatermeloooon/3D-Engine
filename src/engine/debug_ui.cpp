#include "debug_ui.h"
#include "camera.h"
#include "resource_manager.h"
#include "components.h"
#include "shadow.h"
#include "postfx.h"
#include "raytracing.h"
#include "vulkan_init.h"
#include "profiler.h"
#include "skeletal.h"
#include "ibl.h"
#include "../utils/vk_check.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <ImGuizmo.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <commdlg.h>
#endif

static std::string openObjFileDialog() {
#ifdef _WIN32
    char buf[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize  = sizeof(ofn);
    ofn.lpstrFilter  = "Wavefront OBJ (*.obj)\0*.obj\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile    = buf;
    ofn.nMaxFile     = sizeof(buf);
    ofn.Flags        = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle   = "Import Model";
    if (GetOpenFileNameA(&ofn)) return std::string(buf);
#endif
    return {};
}

static std::string openGltfFileDialog() {
#ifdef _WIN32
    char buf[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize  = sizeof(ofn);
    ofn.lpstrFilter  = "glTF (*.gltf;*.glb)\0*.gltf;*.glb\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile    = buf;
    ofn.nMaxFile     = sizeof(buf);
    ofn.Flags        = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle   = "Import glTF";
    if (GetOpenFileNameA(&ofn)) return std::string(buf);
#endif
    return {};
}

void DebugUI::init(GLFWwindow* window, VkInstance instance, VkPhysicalDevice physicalDevice,
                   VkDevice device, uint32_t graphicsFamily, VkQueue graphicsQueue,
                   VkRenderPass renderPass, uint32_t imageCount)
{
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 100;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_imguiPool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding  = 2.0f;
    style.GrabRounding   = 2.0f;
    style.Alpha          = 0.97f;

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance       = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device         = device;
    initInfo.QueueFamily    = graphicsFamily;
    initInfo.Queue          = graphicsQueue;
    initInfo.DescriptorPool = m_imguiPool;
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = imageCount;
    initInfo.PipelineInfoMain.RenderPass  = renderPass;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&initInfo);
}

void DebugUI::cleanup(VkDevice device) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, m_imguiPool, nullptr);
}

void DebugUI::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

static const char* lightTypeName(entt::registry& reg, entt::entity e) {
    if (reg.any_of<DirectionalLightComponent>(e)) return "Directional";
    if (reg.any_of<PointLightComponent>(e))       return "Point";
    if (reg.any_of<SpotLightComponent>(e))        return "Spot";
    return nullptr;
}

void DebugUI::buildUI(entt::registry& registry, ResourceManager& resources,
                      Camera& camera, ShadowData& shadow, PostFXSettings& postfx,
                      RtSettings& rt,
                      int visibleEntities, int totalEntities,
                      float deltaTime,
                      const Profiler* profiler,
                      const SkinnedMesh* skinnedMesh,
                      IblBakeParams* iblParams,
                      bool* iblRebuildRequest,
                      std::string* pendingGltfLoad)
{
    // FPS calc
    m_frameCount++;
    m_fpsTimer += deltaTime;
    if (m_fpsTimer >= 0.5f) {
        m_displayFps = static_cast<float>(m_frameCount) / m_fpsTimer;
        m_displayFrameTime = m_fpsTimer / static_cast<float>(m_frameCount) * 1000.0f;
        m_frameCount = 0;
        m_fpsTimer = 0.0f;
    }
    m_fpsHistory[m_fpsIndex] = m_displayFps;
    m_fpsIndex = (m_fpsIndex + 1) % 120;

    // ── Dock space spanning the full viewport ───────────────────────────
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);

    ImGuiWindowFlags hostFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus
        | ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##DockHost", nullptr, hostFlags);
    ImGui::PopStyleVar(3);

    ImGuiID dockId = ImGui::GetID("MainDock");
    ImGui::DockSpace(dockId, ImVec2(0, 0),
                     ImGuiDockNodeFlags_PassthruCentralNode);

    if (!m_initialDockBuilt) {
        m_initialDockBuilt = true;
        ImGui::DockBuilderRemoveNode(dockId);
        ImGui::DockBuilderAddNode(dockId, ImGuiDockNodeFlags_DockSpace
                                          | ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::DockBuilderSetNodeSize(dockId, vp->WorkSize);

        ImGuiID dockMain   = dockId;
        ImGuiID dockLeft   = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left,  0.20f, nullptr, &dockMain);
        ImGuiID dockRight  = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right, 0.28f, nullptr, &dockMain);
        ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.50f, nullptr, &dockRight);

        ImGui::DockBuilderDockWindow("Hierarchy",        dockLeft);
        ImGui::DockBuilderDockWindow("Performance",      dockLeft);
        ImGui::DockBuilderDockWindow("Frame Timing",     dockLeft);
        ImGui::DockBuilderDockWindow("Properties",       dockRight);
        ImGui::DockBuilderDockWindow("Lights",           dockRight);
        ImGui::DockBuilderDockWindow("Post-Processing",  dockBottom);
        ImGui::DockBuilderDockWindow("Shadows (CSM)",    dockBottom);
        ImGui::DockBuilderDockWindow("Ray Tracing",      dockBottom);
        ImGui::DockBuilderDockWindow("Camera",           dockBottom);
        ImGui::DockBuilderFinish(dockId);
    }
    ImGui::End();

    // ── Top menu bar ────────────────────────────────────────────────────
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Gizmo")) {
            if (ImGui::MenuItem("Translate (W)", "W", m_gizmoOperation == ImGuizmo::TRANSLATE))
                m_gizmoOperation = ImGuizmo::TRANSLATE;
            if (ImGui::MenuItem("Rotate (E)",    "E", m_gizmoOperation == ImGuizmo::ROTATE))
                m_gizmoOperation = ImGuizmo::ROTATE;
            if (ImGui::MenuItem("Scale (R)",     "R", m_gizmoOperation == ImGuizmo::SCALE))
                m_gizmoOperation = ImGuizmo::SCALE;
            ImGui::Separator();
            if (ImGui::MenuItem("World", nullptr, m_gizmoMode == ImGuizmo::WORLD))
                m_gizmoMode = ImGuizmo::WORLD;
            if (ImGui::MenuItem("Local", nullptr, m_gizmoMode == ImGuizmo::LOCAL))
                m_gizmoMode = ImGuizmo::LOCAL;
            ImGui::EndMenu();
        }
        ImGui::Text("|  FPS: %.0f  (%.2f ms)", m_displayFps, m_displayFrameTime);
        ImGui::EndMainMenuBar();
    }

    // Keyboard shortcuts for gizmo (only when not typing)
    if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) m_gizmoOperation = ImGuizmo::TRANSLATE;
        if (ImGui::IsKeyPressed(ImGuiKey_E)) m_gizmoOperation = ImGuizmo::ROTATE;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) m_gizmoOperation = ImGuizmo::SCALE;
    }

    // ── Hierarchy ───────────────────────────────────────────────────────
    if (ImGui::Begin("Hierarchy")) {
        auto view = registry.view<NameComponent>();
        for (auto entity : view) {
            auto& name = view.get<NameComponent>(entity);
            std::string label = name.name;
            if (auto* lt = lightTypeName(registry, entity)) {
                label += "  [";  label += lt; label += "]";
            }
            bool selected = (entity == m_selectedEntity);
            ImGui::PushID(static_cast<int>(static_cast<uint32_t>(entity)));
            if (ImGui::Selectable(label.c_str(), selected)) {
                m_selectedEntity = entity;
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        if (ImGui::Button("Import Model...")) {
            std::string path = openObjFileDialog();
            if (!path.empty()) {
                try {
                    auto handle = resources.loadMesh(path);
                    auto e = registry.create();
                    std::string name;
                    try { name = std::filesystem::path(path).stem().string(); }
                    catch (...) { name = "Imported"; }
                    if (name.empty()) name = "Imported";
                    registry.emplace<NameComponent>(e, name);
                    registry.emplace<TransformComponent>(e);
                    registry.emplace<MeshComponent>(e, handle);
                    MaterialComponent mat{};
                    mat.texture   = resources.getDefaultTexture();
                    mat.color     = glm::vec4(0.85f, 0.85f, 0.85f, 1.0f);
                    mat.metallic  = 0.05f;
                    mat.roughness = 0.55f;
                    registry.emplace<MaterialComponent>(e, mat);
                    m_selectedEntity = e;
                } catch (const std::exception& ex) {
                    ImGui::OpenPopup("Import Failed");
                    static char errBuf[512];
                    std::snprintf(errBuf, sizeof(errBuf), "Failed to import:\n%s", ex.what());
                    (void)errBuf;
                }
            }
        }
        ImGui::SameLine();
        if (pendingGltfLoad && ImGui::Button("Import glTF (Skinned)...")) {
            std::string p = openGltfFileDialog();
            if (!p.empty()) {
                // Engine picks this up after buildUI returns: it stalls the
                // GPU, swaps the shared SkinnedMesh, and re-uploads. We just
                // surface the request.
                *pendingGltfLoad = p;
            }
        }
        if (ImGui::Button("Spawn Point Light")) {
            auto e = registry.create();
            registry.emplace<NameComponent>(e, std::string("Point Light"));
            TransformComponent t{}; t.position = {0.0f, 2.0f, 0.0f};
            registry.emplace<TransformComponent>(e, t);
            registry.emplace<PointLightComponent>(e);
            m_selectedEntity = e;
        }
        if (ImGui::Button("Spawn Spot Light")) {
            auto e = registry.create();
            registry.emplace<NameComponent>(e, std::string("Spot Light"));
            TransformComponent t{}; t.position = {0.0f, 3.0f, 0.0f};
            registry.emplace<TransformComponent>(e, t);
            registry.emplace<SpotLightComponent>(e);
            m_selectedEntity = e;
        }
        ImGui::SameLine();
        if (ImGui::Button("Spawn Cube")) {
            auto e = registry.create();
            registry.emplace<NameComponent>(e, std::string("Cube"));
            registry.emplace<TransformComponent>(e);
            registry.emplace<MeshComponent>(e, resources.getDefaultCube());
            MaterialComponent mat{};
            mat.texture = resources.getDefaultTexture();
            mat.color = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
            mat.roughness = 0.5f;
            registry.emplace<MaterialComponent>(e, mat);
            m_selectedEntity = e;
        }
        if (ImGui::Button("Spawn Skinned Test")) {
            auto e = registry.create();
            registry.emplace<NameComponent>(e, std::string("Skinned"));
            TransformComponent t{}; t.position = { 3.0f, -0.4f, 0.0f };
            registry.emplace<TransformComponent>(e, t);
            registry.emplace<SkinnedMeshComponent>(e);
            registry.emplace<AnimatorComponent>(e);
            MaterialComponent mat{};
            mat.texture   = resources.getDefaultTexture();
            mat.color     = glm::vec4(0.7f, 0.5f, 0.95f, 1.0f);
            mat.metallic  = 0.1f;
            mat.roughness = 0.5f;
            registry.emplace<MaterialComponent>(e, mat);
            m_selectedEntity = e;
        }
        // Spawn a 30×30 grid of icosphere instances using a 4-level LOD group
        // (1280 / 320 / 80 / 20 tris). The LOD group is created on first click
        // and reused across subsequent clicks. Move the camera to see the
        // GPU-side LOD selection swap meshes by view distance.
        if (ImGui::Button("Spawn 900 LOD Spheres")) {
            static LODGroupHandle s_sphereLOD{};
            static bool s_sphereLODBuilt = false;
            if (!s_sphereLODBuilt) {
                MeshHandle lod0 = resources.addMesh("lod_sphere_0", createIcosphereMesh(3));
                MeshHandle lod1 = resources.addMesh("lod_sphere_1", createIcosphereMesh(2));
                MeshHandle lod2 = resources.addMesh("lod_sphere_2", createIcosphereMesh(1));
                MeshHandle lod3 = resources.addMesh("lod_sphere_3", createIcosphereMesh(0));
                MeshLODGroup g{};
                g.levels = {
                    { lod0, 12.0f },
                    { lod1, 30.0f },
                    { lod2, 60.0f },
                    { lod3, 1.0e30f },
                };
                s_sphereLOD = resources.addLODGroup(std::move(g));
                s_sphereLODBuilt = true;
            }
            for (int i = 0; i < 900; ++i) {
                auto e = registry.create();
                registry.emplace<NameComponent>(e, std::string("LODSphere"));
                TransformComponent t{};
                float x = (float)((i % 30) - 15) * 1.6f;
                float z = (float)((i / 30) - 15) * 1.6f;
                t.position = { x, 0.5f, z };
                t.scale    = { 0.6f, 0.6f, 0.6f };
                registry.emplace<TransformComponent>(e, t);
                registry.emplace<MeshComponent>(e, MeshComponent{ /*placeholder*/ resources.getDefaultCube() });
                registry.emplace<MeshLODComponent>(e, MeshLODComponent{ s_sphereLOD });
                MaterialComponent mat{};
                mat.texture   = resources.getDefaultTexture();
                float h       = (float)i / 900.0f;
                mat.color     = glm::vec4(0.3f + h * 0.6f, 0.6f, 0.9f - h * 0.4f, 1.0f);
                mat.metallic  = 0.0f;
                mat.roughness = 0.45f;
                registry.emplace<MaterialComponent>(e, mat);
            }
        }
        if (ImGui::Button("Spawn 1000 Cubes (instanced demo)")) {
            for (int i = 0; i < 1000; ++i) {
                auto e = registry.create();
                registry.emplace<NameComponent>(e, std::string("Inst"));
                TransformComponent t{};
                float x = (float)((i % 40) - 20) * 1.5f;
                float z = (float)((i / 40) - 12) * 1.5f;
                t.position = { x, 0.5f, z };
                t.scale    = { 0.5f, 0.5f, 0.5f };
                registry.emplace<TransformComponent>(e, t);
                registry.emplace<MeshComponent>(e, resources.getDefaultCube());
                MaterialComponent mat{};
                mat.texture   = resources.getDefaultTexture();
                float h       = (float)i / 1000.0f;
                mat.color     = glm::vec4(0.4f + h * 0.6f, 0.5f, 1.0f - h * 0.5f, 1.0f);
                mat.metallic  = (i % 3 == 0) ? 1.0f : 0.0f;
                mat.roughness = 0.2f + ((i * 37) % 100) / 100.0f * 0.6f;
                registry.emplace<MaterialComponent>(e, mat);
            }
        }
    }
    ImGui::End();

    // ── Frame Timing (GPU + CPU per-pass) ───────────────────────────────
    if (ImGui::Begin("Frame Timing")) {
        if (!profiler) {
            ImGui::TextDisabled("Profiler not attached.");
        } else {
            // GPU column. Results are one frame behind by construction (we
            // read back the previous use of this frame slot's query pool).
            ImGui::TextDisabled("GPU (1 frame lag)");
            if (!profiler->gpuAvailable()) {
                ImGui::TextDisabled("  timestamps unsupported");
            } else if (profiler->gpuResults().empty()) {
                ImGui::TextDisabled("  warming up…");
            } else {
                double gpuTotal = 0.0;
                for (auto& e : profiler->gpuResults()) gpuTotal += e.ms;
                for (auto& e : profiler->gpuResults()) {
                    ImGui::Text("  %-18s %6.3f ms", e.name, e.ms);
                }
                ImGui::Separator();
                ImGui::Text("  %-18s %6.3f ms", "GPU total (sum)", gpuTotal);
            }
            ImGui::Separator();
            ImGui::TextDisabled("CPU (this frame)");
            if (profiler->cpuResults().empty()) {
                ImGui::TextDisabled("  no scopes");
            } else {
                double cpuTotal = 0.0;
                for (auto& e : profiler->cpuResults()) cpuTotal += e.ms;
                for (auto& e : profiler->cpuResults()) {
                    ImGui::Text("  %-22s %6.3f ms", e.name, e.ms);
                }
                ImGui::Separator();
                ImGui::Text("  %-22s %6.3f ms", "CPU total (sum)", cpuTotal);
            }
        }
    }
    ImGui::End();

    // ── Performance ─────────────────────────────────────────────────────
    if (ImGui::Begin("Performance")) {
        ImGui::Text("FPS: %.1f", m_displayFps);
        ImGui::Text("Frame Time: %.2f ms", m_displayFrameTime);
        ImGui::PlotLines("##fps", m_fpsHistory, 120, m_fpsIndex,
                         nullptr, 0.0f, 200.0f, ImVec2(-1, 60));
        int entityCount = 0;
        for (auto e : registry.view<NameComponent>()) { (void)e; entityCount++; }
        ImGui::Text("Entities: %d", entityCount);
        ImGui::Text("Drawable: %d   Visible: %d   Culled: %d",
                    totalEntities, visibleEntities,
                    totalEntities - visibleEntities);
    }
    ImGui::End();

    // ── Properties (right panel) ────────────────────────────────────────
    if (ImGui::Begin("Properties")) {
        if (m_selectedEntity != entt::null && registry.valid(m_selectedEntity)) {
            auto e = m_selectedEntity;

            if (auto* n = registry.try_get<NameComponent>(e)) {
                char buf[128]; std::strncpy(buf, n->name.c_str(), sizeof(buf)); buf[127] = 0;
                if (ImGui::InputText("Name", buf, sizeof(buf))) n->name = buf;
            }

            if (auto* t = registry.try_get<TransformComponent>(e)) {
                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Position", &t->position.x, 0.05f);
                    ImGui::DragFloat3("Rotation", &t->rotation.x, 0.5f);
                    ImGui::DragFloat3("Scale",    &t->scale.x,    0.02f, 0.001f, 100.0f);
                }
            }

            if (auto* m = registry.try_get<MaterialComponent>(e)) {
                if (ImGui::CollapsingHeader("Material (PBR)", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::ColorEdit4("Albedo Tint", &m->color.x);
                    ImGui::SliderFloat("Metallic",   &m->metallic,  0.0f, 1.0f);
                    ImGui::SliderFloat("Roughness",  &m->roughness, 0.04f, 1.0f);
                    // Parallax (only useful when a height map is assigned).
                    bool hasHeight = (m->heightTexture.id != 0);
                    if (ImGui::Checkbox("Parallax (POM)", &hasHeight)) {
                        // The user can't pick a texture here yet — toggling
                        // just resets the slot to "none" if disabling. Asset
                        // hookup arrives with the glTF importer.
                        if (!hasHeight) m->heightTexture.id = 0;
                    }
                    if (m->heightTexture.id != 0) {
                        ImGui::SliderFloat("Parallax Scale", &m->parallaxScale,
                                           0.0f, 0.25f, "%.3f");
                    }
                }
            }

            if (auto* a = registry.try_get<AnimatorComponent>(e)) {
                if (ImGui::CollapsingHeader("Animator", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (skinnedMesh && !skinnedMesh->animations.empty()) {
                        const int clipCount = static_cast<int>(skinnedMesh->animations.size());
                        a->animationIndex = std::clamp(a->animationIndex, 0, clipCount - 1);
                        const auto& clip = skinnedMesh->animations[a->animationIndex];
                        const float duration = clip.duration;

                        // Clip combo
                        const char* preview = clip.name.empty() ? "(unnamed)" : clip.name.c_str();
                        if (ImGui::BeginCombo("Clip", preview)) {
                            for (int i = 0; i < clipCount; ++i) {
                                const auto& c = skinnedMesh->animations[i];
                                const std::string label = c.name.empty()
                                    ? ("clip " + std::to_string(i))
                                    : c.name;
                                bool sel = (i == a->animationIndex);
                                if (ImGui::Selectable(label.c_str(), sel)) {
                                    if (i != a->animationIndex) {
                                        a->animationIndex = i;
                                        a->time = 0.0f;
                                    }
                                }
                                if (sel) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        // Transport buttons
                        if (ImGui::Button(a->playing ? "Pause" : "Play")) {
                            // If not looping and we're at the end, rewind on play.
                            if (!a->playing && !a->loop && duration > 0.0f && a->time >= duration) {
                                a->time = 0.0f;
                            }
                            a->playing = !a->playing;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Stop")) {
                            a->time    = 0.0f;
                            a->playing = false;
                        }
                        ImGui::SameLine();
                        ImGui::Checkbox("Loop", &a->loop);

                        // Scrubber — dragging pauses, releasing restores prior state.
                        float scrub = a->time;
                        const float scrubMax = duration > 0.0f ? duration : 1.0f;
                        ImGui::SliderFloat("Time", &scrub, 0.0f, scrubMax, "%.3f s");
                        if (ImGui::IsItemActivated()) {
                            m_animScrubWasPlaying = a->playing;
                            a->playing = false;
                        }
                        if (ImGui::IsItemActive()) {
                            a->time = scrub;
                        }
                        if (ImGui::IsItemDeactivatedAfterEdit()) {
                            a->time    = scrub;
                            a->playing = m_animScrubWasPlaying;
                        }

                        ImGui::DragFloat("Speed", &a->speed, 0.01f, -4.0f, 4.0f);

                        ImGui::Separator();
                        ImGui::Text("Duration: %.3f s", duration);
                        ImGui::Text("Channels: %d", (int)clip.channels.size());
                        ImGui::Text("Joints:   %d", (int)skinnedMesh->skeleton.joints.size());
                    } else {
                        ImGui::TextDisabled("No animations loaded.");
                    }
                }
            }

            if (auto* d = registry.try_get<DirectionalLightComponent>(e)) {
                if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Direction", &d->direction.x, 0.05f, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Color",     &d->color.x);
                    ImGui::DragFloat("Intensity",  &d->intensity, 0.1f, 0.0f, 50.0f);
                    ImGui::Checkbox("Casts Shadows", &d->castsShadows);
                }
            }
            if (auto* p = registry.try_get<PointLightComponent>(e)) {
                if (ImGui::CollapsingHeader("Point Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::ColorEdit3("Color",    &p->color.x);
                    ImGui::DragFloat("Intensity", &p->intensity, 0.1f, 0.0f, 50.0f);
                    ImGui::DragFloat("Range",     &p->range,     0.1f, 0.1f, 100.0f);
                }
            }
            if (auto* s = registry.try_get<SpotLightComponent>(e)) {
                if (ImGui::CollapsingHeader("Spot Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Direction",  &s->direction.x, 0.05f, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Color",      &s->color.x);
                    ImGui::DragFloat("Intensity",   &s->intensity, 0.1f, 0.0f, 50.0f);
                    ImGui::DragFloat("Range",       &s->range,     0.1f, 0.1f, 100.0f);
                    ImGui::DragFloat("Inner Cone",  &s->innerConeDeg, 0.5f, 0.0f, 89.0f);
                    ImGui::DragFloat("Outer Cone",  &s->outerConeDeg, 0.5f, 0.0f, 89.0f);
                }
            }
            if (auto* r = registry.try_get<RotatorComponent>(e)) {
                if (ImGui::CollapsingHeader("Rotator")) {
                    ImGui::DragFloat3("Axis", &r->axis.x, 0.05f);
                    ImGui::DragFloat("Speed", &r->speed,  1.0f, 0.0f, 360.0f);
                }
            }

            ImGui::Separator();
            if (ImGui::Button("Delete")) {
                registry.destroy(e);
                m_selectedEntity = entt::null;
            }
        } else {
            ImGui::TextDisabled("Select an entity in the hierarchy");
        }
    }
    ImGui::End();

    // ── Lights summary panel ────────────────────────────────────────────
    if (ImGui::Begin("Lights")) {
        int dirCount = 0, pointCount = 0, spotCount = 0;
        for (auto e : registry.view<DirectionalLightComponent>()) { (void)e; dirCount++; }
        for (auto e : registry.view<PointLightComponent>())       { (void)e; pointCount++; }
        for (auto e : registry.view<SpotLightComponent>())        { (void)e; spotCount++; }

        ImGui::Text("Directional: %d", dirCount);
        ImGui::Text("Point:       %d", pointCount);
        ImGui::Text("Spot:        %d", spotCount);
        ImGui::Separator();
        ImGui::TextWrapped("Edit individual lights via Properties. The first directional light casts CSM shadows.");
    }
    ImGui::End();

    // ── Environment / IBL panel ─────────────────────────────────────────
    if (iblParams && iblRebuildRequest) {
        if (ImGui::Begin("Environment")) {
            ImGui::TextWrapped("Procedural sky + image-based lighting bake. "
                               "Press Rebake after editing parameters or "
                               "providing an HDR equirectangular path.");
            ImGui::Separator();

            ImGui::DragFloat3("Sun Direction", &iblParams->sunDir.x, 0.05f, -1.0f, 1.0f);
            ImGui::DragFloat ("Sun Intensity", &iblParams->sunIntensity, 0.05f, 0.0f, 32.0f);
            ImGui::SliderFloat("IBL Intensity", &iblParams->intensity, 0.0f, 2.0f, "%.3f");
            ImGui::ColorEdit3("Zenith",  &iblParams->zenithColor.x);
            ImGui::ColorEdit3("Horizon", &iblParams->horizonColor.x);
            ImGui::ColorEdit3("Ground",  &iblParams->groundColor.x);

            char buf[260];
            std::strncpy(buf, iblParams->hdrPath.c_str(), sizeof(buf));
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText("HDR equirect (.hdr)", buf, sizeof(buf))) {
                iblParams->hdrPath = buf;
            }

            if (ImGui::Button("Rebake")) {
                *iblRebuildRequest = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear HDR (procedural)")) {
                iblParams->hdrPath.clear();
                *iblRebuildRequest = true;
            }
        }
        ImGui::End();
    }

    // ── Post-Processing ─────────────────────────────────────────────────
    if (ImGui::Begin("Post-Processing")) {
        if (ImGui::CollapsingHeader("Debug View", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* views[] = { "Final", "HDR (raw)", "Bloom", "SSAO", "(unused)", "Lit (no postfx)" };
            ImGui::Combo("View", &postfx.debugView, views, IM_ARRAYSIZE(views));
            ImGui::TextDisabled("Use this to isolate which stage is wrong.");
        }
        if (ImGui::CollapsingHeader("Tone Map / Grading", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* tonemaps[] = { "ACES", "Reinhard", "Off" };
            ImGui::Combo("Tonemap", &postfx.tonemapMode, tonemaps, IM_ARRAYSIZE(tonemaps));
            ImGui::SliderFloat("Exposure",   &postfx.exposure,   -3.0f, 3.0f);
            ImGui::SliderFloat("Saturation", &postfx.saturation, 0.0f, 2.0f);
            ImGui::SliderFloat("Contrast",   &postfx.contrast,   0.5f, 1.5f);
            ImGui::SliderFloat("Gamma",      &postfx.gamma,      1.0f, 3.0f);
            ImGui::ColorEdit3("Color Lift",  &postfx.colorBalance.x);
        }
        if (ImGui::CollapsingHeader("Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable##bloom",   &postfx.bloomEnabled);
            ImGui::SliderFloat("Strength",     &postfx.bloomStrength,    0.0f, 0.5f);
            ImGui::SliderFloat("Threshold",    &postfx.bloomThreshold,   0.0f, 5.0f);
            ImGui::SliderFloat("Filter Radius",&postfx.bloomFilterRadius,0.0f, 4.0f);
        }
        if (ImGui::CollapsingHeader("SSAO", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable##ssao", &postfx.ssaoEnabled);
            ImGui::SliderFloat("Strength",  &postfx.ssaoStrength,  0.0f, 1.0f);
            ImGui::SliderFloat("Radius",    &postfx.ssaoRadius,    0.05f, 2.0f);
            ImGui::SliderFloat("Bias",      &postfx.ssaoBias,      0.0f, 0.1f);
            ImGui::SliderFloat("Intensity", &postfx.ssaoIntensity, 0.5f, 4.0f);
        }
        if (ImGui::CollapsingHeader("Anti-Aliasing", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("FXAA", &postfx.fxaaEnabled);
            ImGui::TextDisabled("Edge-aware luma blur, ~0.2ms at 720p");
        }
        if (ImGui::CollapsingHeader("Variable-Rate Shading")) {
            if (VRS_SUPPORTED) {
                const char* modes[] = { "Off (1x1)", "Auto / per-LOD", "Forced 2x2", "Forced 4x4" };
                ImGui::Combo("Mode", &postfx.vrsMode, modes, IM_ARRAYSIZE(modes));
                ImGui::TextDisabled("Auto: LOD0=1x1, LOD1=2x1, LOD2=2x2, LOD3=4x4");
            } else {
                ImGui::TextDisabled("VK_KHR_fragment_shading_rate not supported on this GPU");
            }
        }
        if (ImGui::CollapsingHeader("Vignette")) {
            ImGui::SliderFloat("Intensity", &postfx.vignetteIntensity, 0.0f, 1.0f);
            ImGui::SliderFloat("Falloff",   &postfx.vignetteFalloff,   0.0f, 0.5f);
        }
    }
    ImGui::End();

    // ── Ray Tracing ─────────────────────────────────────────────────────
    if (ImGui::Begin("Ray Tracing")) {
        if (!RT_SUPPORTED) {
            ImGui::TextDisabled("RT not supported by this device / driver.");
            ImGui::TextDisabled("Check the console for which extension or feature is missing.");
        } else {
            ImGui::Text("RT supported: BLAS/TLAS + ray queries available");
            ImGui::Separator();

            // Presets — apply curated rt + post-fx combos with one click. The
            // sliders below remain editable so you can tune any preset.
            ImGui::TextDisabled("Presets");
            if (ImGui::Button("Off (Classic)")) {
                rt.enabled            = false;
                rt.shadows            = false;
                rt.reflections        = false;
                rt.gi                 = false;
                postfx.ssaoEnabled    = true;
                postfx.ssaoStrength   = 0.5f;
                postfx.bloomEnabled   = false;       // kills the chrome-edge halo
            }
            ImGui::SameLine();
            if (ImGui::Button("RT (Clean)")) {
                rt.enabled            = true;
                rt.shadows            = true;
                rt.reflections        = true;
                rt.gi                 = false;       // skip GI → no Monte-Carlo grain
                rt.sunOnly            = true;
                rt.shadowSoftness     = 0.005f;
                rt.shadowSamples      = 16;
                rt.reflectionSamples  = 4;
                rt.reflectionIntensity = 1.0f;
                postfx.ssaoEnabled    = false;       // RT shadows replace SSAO
                postfx.bloomEnabled   = true;
                postfx.bloomStrength  = 0.04f;
                postfx.bloomThreshold = 1.5f;
            }
            ImGui::SameLine();
            if (ImGui::Button("RT (Full)")) {
                rt.enabled            = true;
                rt.shadows            = true;
                rt.reflections        = true;
                rt.gi                 = true;
                rt.sunOnly            = true;
                rt.shadowSamples      = 16;
                rt.reflectionSamples  = 4;
                // GI quality knobs: 48 samples is ~2× quieter than the old
                // 12-sample default (variance ~ 1/√N). Costs ~3× more rays
                // per pixel, but RT GI was severely under-sampled before.
                // mix at 0.25 lets the GI color-bleed actually show.
                rt.giSamples          = 48;
                rt.giIntensity        = 0.25f;
                postfx.ssaoEnabled    = false;
                postfx.bloomEnabled   = true;
                postfx.bloomStrength  = 0.05f;
                postfx.bloomThreshold = 1.5f;
            }
            ImGui::Separator();

            ImGui::Checkbox("Enable Ray Tracing", &rt.enabled);
            ImGui::BeginDisabled(!rt.enabled);
            ImGui::Checkbox("RT Shadows (replaces CSM)", &rt.shadows);
            ImGui::Checkbox("Sun only (CSM parity)", &rt.sunOnly);
            ImGui::TextDisabled("ON = shadow only the first directional light");
            ImGui::SliderFloat("Light angular radius", &rt.shadowSoftness, 0.0f, 0.1f,
                               "%.4f rad");
            ImGui::TextDisabled("0.0046 ≈ real sun, larger = softer penumbra");
            ImGui::SliderInt("Samples / light / fragment", &rt.shadowSamples, 1, 64);
            ImGui::TextDisabled("Higher = smoother shadows, more GPU cost");
            ImGui::Separator();
            ImGui::Checkbox("RT Reflections", &rt.reflections);
            ImGui::TextDisabled("Mirror + glossy on metals (Fresnel-weighted)");
            ImGui::BeginDisabled(!rt.reflections);
            ImGui::SliderInt("Reflection samples", &rt.reflectionSamples, 1, 16);
            ImGui::SliderFloat("Reflection max dist", &rt.reflectionMaxDist, 5.0f, 500.0f, "%.0f");
            ImGui::SliderFloat("Reflection intensity", &rt.reflectionIntensity, 0.0f, 2.0f);
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::Checkbox("RT Global Illumination", &rt.gi);
            ImGui::TextDisabled("One-bounce indirect — replaces hemisphere ambient");
            ImGui::BeginDisabled(!rt.gi);
            ImGui::SliderInt("GI samples", &rt.giSamples, 1, 64);
            ImGui::SliderFloat("GI max dist", &rt.giMaxDist, 5.0f, 100.0f, "%.0f");
            ImGui::SliderFloat("GI mix (0=smooth, 1=full RT)", &rt.giIntensity, 0.0f, 1.0f);
            ImGui::EndDisabled();
            ImGui::EndDisabled();
        }
    }
    ImGui::End();

    // ── Shadows (CSM) ───────────────────────────────────────────────────
    if (ImGui::Begin("Shadows (CSM)")) {
        ImGui::Text("Cascades: %d", static_cast<int>(SHADOW_CASCADE_COUNT));
        ImGui::Text("Map size: %u x %u", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        ImGui::Separator();
        ImGui::SliderFloat("Split Lambda", &shadow.cascadeSplitLambda, 0.0f, 1.0f);
        ImGui::TextDisabled("0 = uniform splits, 1 = logarithmic");
        ImGui::SliderFloat("Depth Bias",  &shadow.depthBias, 0.0f, 5.0f);
        ImGui::SliderFloat("Slope Bias",  &shadow.slopeBias, 0.0f, 5.0f);
        ImGui::SliderFloat("PCF Radius",  &shadow.pcfRadius, 0.0f, 4.0f);
    }
    ImGui::End();

    // ── Camera ──────────────────────────────────────────────────────────
    if (ImGui::Begin("Camera")) {
        const char* modeStr = camera.getMode() == CameraMode::FPS ? "FPS" : "Orbit";
        ImGui::Text("Mode: %s (Tab to toggle)", modeStr);
        auto pos = camera.getPosition();
        ImGui::Text("Position: (%.1f, %.1f, %.1f)", pos.x, pos.y, pos.z);
        ImGui::TextWrapped("Orbit: LMB rotate | RMB pan | Scroll zoom\nFPS: WASD move | Mouse look | Space/Shift up/down\nTab: toggle mode");
    }
    ImGui::End();

    // ── ImGuizmo over the central viewport ──────────────────────────────
    if (m_selectedEntity != entt::null && registry.valid(m_selectedEntity)) {
        auto* t = registry.try_get<TransformComponent>(m_selectedEntity);
        if (t) {
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
            ImGuiViewport* mainVp = ImGui::GetMainViewport();
            ImGuizmo::SetRect(mainVp->WorkPos.x, mainVp->WorkPos.y,
                              mainVp->WorkSize.x, mainVp->WorkSize.y);

            glm::mat4 view = camera.getViewMatrix();
            glm::mat4 proj = camera.getProjectionMatrixUnflipped();

            glm::mat4 model = t->getMatrix();
            ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                                 (ImGuizmo::OPERATION)m_gizmoOperation,
                                 (ImGuizmo::MODE)m_gizmoMode,
                                 glm::value_ptr(model));

            if (ImGuizmo::IsUsing()) {
                glm::vec3 newPos, newRot, newScale;
                ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(model),
                                                      &newPos.x, &newRot.x, &newScale.x);
                t->position = newPos;
                t->rotation = newRot;
                t->scale    = newScale;
            }
        }
    }
}

void DebugUI::endFrame() {
    ImGui::Render();
}
