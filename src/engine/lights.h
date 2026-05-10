#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

#include <entt/entt.hpp>

#include "buffer.h"

constexpr uint32_t MAX_LIGHTS = 32;

// GPU-side representation of a light. 64 bytes, packed for std140.
struct LightData {
    glm::vec4 positionAndType;     // xyz = position, w = type (0=dir, 1=point, 2=spot)
    glm::vec4 directionAndRange;   // xyz = direction, w = range
    glm::vec4 colorAndIntensity;   // xyz = color,     w = intensity
    glm::vec4 spotCones;           // x = cos(inner),  y = cos(outer), zw = pad
};

struct LightsUBO {
    int numLights;
    int pad0;
    int pad1;
    int pad2;
    LightData lights[MAX_LIGHTS];
};

struct LightBufferData {
    std::vector<AllocatedBuffer> buffers;
    std::vector<void*> mapped;
};

void createLightBuffers(LightBufferData& data, VkPhysicalDevice physicalDevice,
                        VkDevice device, uint32_t framesInFlight);

void destroyLightBuffers(VkDevice device, LightBufferData& data);

// Gathers all light components into the light UBO and writes to the mapped buffer.
// Returns the directional light direction (or {0,-1,0} if none) for shadow CSM.
glm::vec3 updateLightBuffer(LightBufferData& data, uint32_t frameIndex,
                            entt::registry& registry);
