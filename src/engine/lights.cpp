#include "lights.h"
#include "components.h"
#include "../utils/vk_check.h"

#include <cstring>
#include <cmath>

void createLightBuffers(LightBufferData& data, VkPhysicalDevice physicalDevice,
                        VkDevice device, uint32_t framesInFlight)
{
    VkDeviceSize size = sizeof(LightsUBO);
    data.buffers.resize(framesInFlight);
    data.mapped.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        data.buffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, data.buffers[i].memory, 0, size, 0, &data.mapped[i]);
    }
}

void destroyLightBuffers(VkDevice device, LightBufferData& data) {
    for (auto& buf : data.buffers) destroyBuffer(device, buf);
    data.buffers.clear();
    data.mapped.clear();
}

glm::vec3 updateLightBuffer(LightBufferData& data, uint32_t frameIndex,
                            entt::registry& registry)
{
    LightsUBO ubo{};
    ubo.numLights = 0;
    glm::vec3 dirLightDir{0.0f, -1.0f, 0.0f};

    // Directional lights first (only one really used for shadows; rest still lit)
    auto dirView = registry.view<DirectionalLightComponent>();
    for (auto e : dirView) {
        if (ubo.numLights >= MAX_LIGHTS) break;
        auto& d = dirView.get<DirectionalLightComponent>(e);
        glm::vec3 dir = glm::normalize(d.direction);
        if (ubo.numLights == 0) dirLightDir = dir;

        LightData ld{};
        ld.positionAndType   = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        ld.directionAndRange = glm::vec4(dir, 0.0f);
        ld.colorAndIntensity = glm::vec4(d.color, d.intensity);
        ld.spotCones         = glm::vec4(0.0f);
        ubo.lights[ubo.numLights++] = ld;
    }

    // Point lights
    auto pointView = registry.view<TransformComponent, PointLightComponent>();
    for (auto e : pointView) {
        if (ubo.numLights >= MAX_LIGHTS) break;
        auto& t = pointView.get<TransformComponent>(e);
        auto& p = pointView.get<PointLightComponent>(e);

        LightData ld{};
        ld.positionAndType   = glm::vec4(t.position, 1.0f);
        ld.directionAndRange = glm::vec4(0.0f, 0.0f, 0.0f, p.range);
        ld.colorAndIntensity = glm::vec4(p.color, p.intensity);
        ld.spotCones         = glm::vec4(0.0f);
        ubo.lights[ubo.numLights++] = ld;
    }

    // Spot lights
    auto spotView = registry.view<TransformComponent, SpotLightComponent>();
    for (auto e : spotView) {
        if (ubo.numLights >= MAX_LIGHTS) break;
        auto& t = spotView.get<TransformComponent>(e);
        auto& s = spotView.get<SpotLightComponent>(e);

        LightData ld{};
        ld.positionAndType   = glm::vec4(t.position, 2.0f);
        ld.directionAndRange = glm::vec4(glm::normalize(s.direction), s.range);
        ld.colorAndIntensity = glm::vec4(s.color, s.intensity);
        ld.spotCones         = glm::vec4(std::cos(glm::radians(s.innerConeDeg)),
                                         std::cos(glm::radians(s.outerConeDeg)),
                                         0.0f, 0.0f);
        ubo.lights[ubo.numLights++] = ld;
    }

    std::memcpy(data.mapped[frameIndex], &ubo, sizeof(ubo));
    return dirLightDir;
}
