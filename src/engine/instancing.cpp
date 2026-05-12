#include "instancing.h"

#include <cstring>

VkVertexInputBindingDescription getInstanceBindingDescription() {
    VkVertexInputBindingDescription b{};
    b.binding   = 1;                                  // binding 0 is per-vertex
    b.stride    = sizeof(InstanceData);
    b.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    return b;
}

std::array<VkVertexInputAttributeDescription, 9> getInstanceAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 9> attrs{};

    // mat4 model — 4 vec4 slots, locations 4..7
    for (uint32_t i = 0; i < 4; ++i) {
        attrs[i].binding  = 1;
        attrs[i].location = 4 + i;
        attrs[i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[i].offset   = static_cast<uint32_t>(offsetof(InstanceData, model) + i * sizeof(glm::vec4));
    }
    // vec4 colorTint — location 8
    attrs[4].binding  = 1;
    attrs[4].location = 8;
    attrs[4].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[4].offset   = offsetof(InstanceData, colorTint);

    // vec4 matParams — location 9
    attrs[5].binding  = 1;
    attrs[5].location = 9;
    attrs[5].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[5].offset   = offsetof(InstanceData, matParams);

    // Normal matrix columns — locations 10, 11, 12
    for (uint32_t i = 0; i < 3; ++i) {
        attrs[6 + i].binding  = 1;
        attrs[6 + i].location = 10 + i;
        attrs[6 + i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[6 + i].offset   = static_cast<uint32_t>(
            offsetof(InstanceData, normalCol0) + i * sizeof(glm::vec4));
    }

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
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, b.buffers[i].memory, 0, size, 0, &b.mapped[i]);
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
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, b.buffers[i].memory, 0, size, 0, &b.mapped[i]);
    }
}

void destroyIndirectBuffer(VkDevice device, IndirectBuffer& b) {
    for (auto& buf : b.buffers) destroyBuffer(device, buf);
    b.buffers.clear();
    b.mapped.clear();
    b.capacity = 0;
}
