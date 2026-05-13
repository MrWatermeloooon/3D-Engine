#pragma once

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <entt/entt.hpp>

#include <string>

class Camera;
class ResourceManager;
class Profiler;
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
                 std::string* pendingGltfLoad = nullptr);
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
};
