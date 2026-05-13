#include "instancing.h"

#include <cstring>

VkVertexInputBindingDescription getInstanceBindingDescription() {
    VkVertexInputBindingDescription b{};
    b.binding   = 1;                                  // binding 0 is per-vertex
    b.stride    = sizeof(InstanceData);
    b.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    return b;
}

std::array<VkVertexInputAttributeDescription, 14> getInstanceAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 14> attrs{};

    // Instance attribute locations start at 5 — vertex attribute locations
    // are 0..4 (position, normal, texCoord, color, tangent).

    // mat4 model — 4 vec4 slots, locations 5..8
    for (uint32_t i = 0; i < 4; ++i) {
        attrs[i].binding  = 1;
        attrs[i].location = 5 + i;
        attrs[i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[i].offset   = static_cast<uint32_t>(offsetof(InstanceData, model) + i * sizeof(glm::vec4));
    }
    // vec4 colorTint — location 9
    attrs[4].binding  = 1;
    attrs[4].location = 9;
    attrs[4].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[4].offset   = offsetof(InstanceData, colorTint);

    // vec4 matParams — location 10
    attrs[5].binding  = 1;
    attrs[5].location = 10;
    attrs[5].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[5].offset   = offsetof(InstanceData, matParams);

    // Normal matrix columns — locations 11, 12, 13
    for (uint32_t i = 0; i < 3; ++i) {
        attrs[6 + i].binding  = 1;
        attrs[6 + i].location = 11 + i;
        attrs[6 + i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[6 + i].offset   = static_cast<uint32_t>(
            offsetof(InstanceData, normalCol0) + i * sizeof(glm::vec4));
    }

    // mat4 prevModel — locations 14..17
    for (uint32_t i = 0; i < 4; ++i) {
        attrs[9 + i].binding  = 1;
        attrs[9 + i].location = 14 + i;
        attrs[9 + i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[9 + i].offset   = static_cast<uint32_t>(
            offsetof(InstanceData, prevModel) + i * sizeof(glm::vec4));
    }

    // vec4 matParams2 — location 18 (parallax: heightSlot, scale, _, _)
    attrs[13].binding  = 1;
    attrs[13].location = 18;
    attrs[13].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[13].offset   = offsetof(InstanceData, matParams2);

    return attrs;
}

void createInstanceBuffer(InstanceBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity)
{
    b.capacity = capacity;
    b.buffers.resize(framesInFlight);
    b.mapped.resize(framesInFlight);

    VkDeviceSize size = static_cast<VkDeviceSize>(capacity) * sizeof(InstanceData);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        b.buffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b.mapped[i] = b.buffers[i].mapped;
    }
}

void destroyInstanceBuffer(VkDevice device, InstanceBuffer& b) {
    for (auto& buf : b.buffers) destroyBuffer(device, buf);
    b.buffers.clear();
    b.mapped.clear();
    b.capacity = 0;
}

void createIndirectBuffer(IndirectBuffer& b, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight, uint32_t capacity)
{
    b.capacity = capacity;
    b.buffers.resize(framesInFlight);
    b.mapped.resize(framesInFlight);

    VkDeviceSize size = static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        b.buffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b.mapped[i] = b.buffers[i].mapped;
    }
}

void destroyIndirectBuffer(VkDevice device, IndirectBuffer& b) {
    for (auto& buf : b.buffers) destroyBuffer(device, buf);
    b.buffers.clear();
    b.mapped.clear();
    b.capacity = 0;
}

void createCandidateBuffer(CandidateBuffer& b, VkPhysicalDevice physicalDevice,
                           VkDevice device, uint32_t framesInFlight, uint32_t capacity)
{
    b.capacity = capacity;
    b.buffers.resize(framesInFlight);
    b.mapped.resize(framesInFlight);

    VkDeviceSize size = static_cast<VkDeviceSize>(capacity) * sizeof(CandidateInstance);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        b.buffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b.mapped[i] = b.buffers[i].mapped;
    }
}

void destroyCandidateBuffer(VkDevice device, CandidateBuffer& b) {
    for (auto& buf : b.buffers) destroyBuffer(device, buf);
    b.buffers.clear();
    b.mapped.clear();
    b.capacity = 0;
}

void createBatchHeaderBuffer(BatchHeaderBuffer& b, VkPhysicalDevice physicalDevice,
                             VkDevice device, uint32_t framesInFlight, uint32_t capacity)
{
    b.capacity = capacity;
    b.buffers.resize(framesInFlight);
    b.mapped.resize(framesInFlight);

    VkDeviceSize size = static_cast<VkDeviceSize>(capacity) * sizeof(BatchHeader);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        b.buffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b.mapped[i] = b.buffers[i].mapped;
    }
}

void destroyBatchHeaderBuffer(VkDevice device, BatchHeaderBuffer& b) {
    for (auto& buf : b.buffers) destroyBuffer(device, buf);
    b.buffers.clear();
    b.mapped.clear();
    b.capacity = 0;
}
