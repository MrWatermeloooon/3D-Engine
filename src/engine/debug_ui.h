#pragma once

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <entt/entt.hpp>

#include "material_graph.h"
#include "asset_browser.h"

#include <string>

class Camera;
class ResourceManager;
class Profiler;
class PhysicsWorld;
class AudioEngine;
class ScriptEngine;
class UndoStack;
struct ShadowData;
struct PostFXSettings;
struct RtSettings;
struct SkinnedMesh;
struct IblBakeParams;

class DebugUI {
public:
    void init(GLFWwindow* window, VkInstance instance, VkPhysicalDevice physicalDevice,
              VkDevice device, uint32_t graphicsFamily, VkQueue graphicsQueue,
              VkRenderPass renderPass, uint32_t imageCount);

    void cleanup(VkDevice device);

    void beginFrame();
    void buildUI(entt::registry& registry, ResourceManager& resources,
                 Camera& camera, ShadowData& shadow, PostFXSettings& postfx,
                 RtSettings& rt,
                 int visibleEntities, int totalEntities,
                 float deltaTime,
                 const Profiler* profiler = nullptr,
                 const SkinnedMesh* skinnedMesh = nullptr,
                 IblBakeParams* iblParams = nullptr,
                 bool* iblRebuildRequest = nullptr,
                 std::string* pendingGltfLoad = nullptr,
                 PhysicsWorld* physics = nullptr,
                 AudioEngine* audio = nullptr,
                 ScriptEngine* scripting = nullptr,
                 UndoStack* undo = nullptr,
                 std::string* shaderSrcDir = nullptr,
                 bool* shaderReloadRequest = nullptr);
    void endFrame();

private:
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

    float m_fpsHistory[120]{};
    int   m_fpsIndex = 0;
    float m_fpsTimer = 0.0f;
    int   m_frameCount = 0;
    float m_displayFps = 0.0f;
    float m_displayFrameTime = 0.0f;

    entt::entity m_selectedEntity = entt::null;

    int  m_gizmoOperation = 7;     // ImGuizmo::TRANSLATE = 7
    int  m_gizmoMode      = 1;     // ImGuizmo::WORLD = 1
    bool m_initialDockBuilt = false;

    // Animator scrubber: remembers whether playback was active when the slider
    // was grabbed, so releasing the slider restores that state.
    bool m_animScrubWasPlaying = false;

    // Tracks ImGuizmo drag state so a transform edit commits one undo entry
    // on release (not one per frame of the drag).
    bool m_gizmoWasUsing = false;

    // Visual node-graph material editor (composes the PBR MaterialComponent).
    MaterialGraph m_materialGraph;

    // Project file explorer (spawn meshes / apply textures / attach audio).
    AssetBrowser  m_assetBrowser;

    // Last camera raycast result (Physics panel readout).
    bool  m_lastRayValid    = false;
    bool  m_lastRayHit      = false;
    float m_lastRayPoint[3] { 0, 0, 0 };
    float m_lastRayNormal[3]{ 0, 0, 0 };
    float m_lastRayDist     = 0.0f;
};
