#include "descriptors.h"
#include "lights.h"
#include "shadow.h"
#include "../utils/vk_check.h"

#include <array>
#include <cstring>

VkDescriptorSetLayout createSceneSetLayout(VkDevice device) {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};

    // Binding 0: scene UBO (view, proj, cameraPos)
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 1: lights UBO
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 2: cascade UBO (light VPs + split distances)
    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 3: shadow map array (combined image sampler with comparison)
    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout));
    return layout;
}

VkDescriptorSetLayout createMaterialSetLayout(VkDevice device) {
    // Bindless: one binding holding an array of BINDLESS_MAX_TEXTURES samplers.
    // PartiallyBound lets us leave unused slots empty; UpdateAfterBind lets us
    // register new textures even while the set is bound for a draw.
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding         = 0;
    samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = BINDLESS_MAX_TEXTURES;
    samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorBindingFlags bindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCi{};
    flagsCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsCi.bindingCount  = 1;
    flagsCi.pBindingFlags = &bindingFlags;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext        = &flagsCi;
    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &samplerBinding;

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout));
    return layout;
}

void createUniformBuffers(DescriptorData& data, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    data.uniformBuffers.resize(framesInFlight);
    data.uniformBuffersMapped.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; i++) {
        data.uniformBuffers[i] = createBuffer(physicalDevice, device, bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device, data.uniformBuffers[i].memory, 0, bufferSize, 0,
                    &data.uniformBuffersMapped[i]);
    }
}

void createDescriptorPool(DescriptorData& data, VkDevice device,
                          uint32_t framesInFlight, uint32_t maxMaterials) {
    (void)maxMaterials; // bindless: textures live in a single global set
    VkDescriptorPoolSize poolSizes[] = {
        // Scene UBO + lights UBO + cascade UBO + bone UBO per frame
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         framesInFlight * 4 },
        // Shadow map per frame + entire bindless texture array (allocated once)
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, framesInFlight + BINDLESS_MAX_TEXTURES },
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                           | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    poolInfo.maxSets       = framesInFlight + 1 /*bindless*/ + framesInFlight /*bones*/;

    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &data.descriptorPool));
}

void createSceneDescriptorSets(DescriptorData& data, VkDevice device, uint32_t framesInFlight,
                               const std::vector<VkBuffer>& lightBuffers,
                               const std::vector<VkBuffer>& cascadeBuffers,
                               VkImageView shadowArrayView, VkSampler shadowSampler)
{
    std::vector<VkDescriptorSetLayout> layouts(framesInFlight, data.sceneSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = data.descriptorPool;
    allocInfo.descriptorSetCount = framesInFlight;
    allocInfo.pSetLayouts        = layouts.data();

    data.sceneSets.resize(framesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, data.sceneSets.data()));

    for (uint32_t i = 0; i < framesInFlight; i++) {
        VkDescriptorBufferInfo sceneBuf{};
        sceneBuf.buffer = data.uniformBuffers[i].buffer;
        sceneBuf.offset = 0;
        sceneBuf.range  = sizeof(UniformBufferObject);

        VkDescriptorBufferInfo lightBuf{};
        lightBuf.buffer = lightBuffers[i];
        lightBuf.offset = 0;
        lightBuf.range  = sizeof(LightsUBO);

        VkDescriptorBufferInfo cascadeBuf{};
        cascadeBuf.buffer = cascadeBuffers[i];
        cascadeBuf.offset = 0;
        cascadeBuf.range  = sizeof(CascadeUBO);

        VkDescriptorImageInfo shadowImg{};
        shadowImg.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        shadowImg.imageView   = shadowArrayView;
        shadowImg.sampler     = shadowSampler;

        std::array<VkWriteDescriptorSet, 4> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = data.sceneSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &sceneBuf;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = data.sceneSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo     = &lightBuf;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = data.sceneSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo     = &cascadeBuf;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = data.sceneSets[i];
        writes[3].dstBinding      = 3;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo      = &shadowImg;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void allocateBindlessTexturesSet(DescriptorData& data, VkDevice device) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = data.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &data.materialSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &data.bindlessTexturesSet));
}

void writeBindlessTexture(DescriptorData& data, VkDevice device,
                          uint32_t slot, VkImageView view, VkSampler sampler) {
    VkDescriptorImageInfo info{};
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    info.imageView   = view;
    info.sampler     = sampler;

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = data.bindlessTexturesSet;
    w.dstBinding      = 0;
    w.dstArrayElement = slot;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo      = &info;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

void updateUniformBuffer(DescriptorData& data, uint32_t currentFrame,
                         const UniformBufferObject& ubo) {
    memcpy(data.uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}
