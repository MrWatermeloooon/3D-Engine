#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>

#include "buffer.h"

// Per-instance GPU data. 144 bytes. Tightly packed; matches the vertex
// input description in getInstanceAttributeDescriptions().
//
// normalCols stores the precomputed mat3 normal matrix (transpose(inverse(mat3(model))))
// as three column vectors. Computing it on the CPU saves 16 multiplies per vertex
// per frame on the GPU — adds up fast when you have 10k+ instances.
struct InstanceData {
    glm::mat4 model;        // location 4..7 (4 vec4 slots)
    glm::vec4 colorTint;    // location 8
    glm::vec4 matParams;    // location 9 — x=metallic, y=roughness, z=textureIndex (bindless), w=pad
    glm::vec4 normalCol0;   // location 10
    glm::vec4 normalCol1;   // location 11
    glm::vec4 normalCol2;   // location 12
};

VkVertexInputBindingDescription getInstanceBindingDescription();
std::array<VkVertexInputAttributeDescription, 9> getInstanceAttributeDescriptions();

// Per-frame host-visible buffer holding all instance data for the frame.
struct InstanceBuffer {
    std::vector<AllocatedBuffer> buffers;       // one per frame in flight
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0;  // max instances per frame
};

void createInstanceBuffer(InstanceBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyInstanceBuffer(VkDevice device, InstanceBuffer& b);

// Per-frame indirect command buffer. Holds VkDrawIndexedIndirectCommand entries
// authored by the CPU (today) or written by a compute-cull pass (future).
struct IndirectBuffer {
    std::vector<AllocatedBuffer> buffers;
    std::vector<void*>           mapped;
    uint32_t                     capacity = 0; // max commands per frame
};

void createIndirectBuffer(IndirectBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity);
void destroyIndirectBuffer(VkDevice device, IndirectBuffer& b);
