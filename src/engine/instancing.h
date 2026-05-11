#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>

#include "buffer.h"

// Per-instance GPU data. 96 bytes. Tightly packed; matches the vertex
// input description in getInstanceAttributeDescriptions().
struct InstanceData {
    glm::mat4 model;        // location 4..7 (4 vec4 slots)
    glm::vec4 colorTint;    // location 8
    glm::vec4 matParams;    // location 9 — x=metallic, y=roughness, zw=pad
};

VkVertexInputBindingDescription getInstanceBindingDescription();
std::array<VkVertexInputAttributeDescription, 6> getInstanceAttributeDescriptions();

// Per-frame host-visible buffer holding all instance data for the frame.
struct InstanceBuffer {
    std::vector<AllocatedBuffer> buffers;       // one per frame in flight
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0;  // max instances per frame
};

void createInstanceBuffer(InstanceBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyInstanceBuffer(VkDevice device, InstanceBuffer& b);
