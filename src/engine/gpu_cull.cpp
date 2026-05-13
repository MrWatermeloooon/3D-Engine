#include "gpu_cull.h"
#include "pipeline.h"
#include "swapchain.h"
#include "config.h"
#include "../utils/vk_check.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

void createGpuCull(GpuCullData& cull, VkPhysicalDevice physicalDevice, VkDevice device,
                   uint32_t framesInFlight, const std::string& compShaderPath)
{
    // ── Descriptor set layout ───────────────────────────────────────────────
    // 0..4 = SSBOs (candidates, batches, mainInst, shadowInst, indirect)
    //   5  = UBO (CullParams)
    //   6  = HZB combined image sampler
    //   7  = vismask SSBO (two-pass occlusion)
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (uint32_t i = 0; i < 5; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[5].binding         = 5;
    bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[6].binding         = 6;
    bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[7].binding         = 7;
    bindings[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCi{};
    layoutCi.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCi.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCi.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutCi, nullptr, &cull.setLayout));

    // ── Pipeline layout — single 4-byte push constant for the pass index ──
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plCi{};
    plCi.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCi.setLayoutCount         = 1;
    plCi.pSetLayouts            = &cull.setLayout;
    plCi.pushConstantRangeCount = 1;
    plCi.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plCi, nullptr, &cull.pipelineLayout));

    // ── Compute pipeline ────────────────────────────────────────────────────
    VkShaderModule shader = loadShaderModule(device, compShaderPath);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName  = "main";

    VkComputePipelineCreateInfo pipeCi{};
    pipeCi.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeCi.stage  = stage;
    pipeCi.layout = cull.pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeCi, nullptr,
                                      &cull.pipeline));
    vkDestroyShaderModule(device, shader, nullptr);

    // ── Per-frame UBO buffers ──────────────────────────────────────────────
    cull.paramsBuffers.resize(framesInFlight);
    cull.paramsMapped.resize(framesInFlight);
    VkDeviceSize size = sizeof(CullParamsUBO);
    for (uint32_t f = 0; f < framesInFlight; ++f) {
        cull.paramsBuffers[f] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        cull.paramsMapped[f] = cull.paramsBuffers[f].mapped;
        (void)size;
    }

    // ── Per-frame vismask SSBOs (one uint per candidate, device-local) ────
    cull.visMaskCapacity = engine_config::MAX_INSTANCES_PER_FRAME;
    cull.visMaskBuffers.resize(framesInFlight);
    VkDeviceSize visBytes = static_cast<VkDeviceSize>(cull.visMaskCapacity) * sizeof(uint32_t);
    for (uint32_t f = 0; f < framesInFlight; ++f) {
        cull.visMaskBuffers[f] = createBuffer(physicalDevice, device, visBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    // ── Descriptor pool ─────────────────────────────────────────────────────
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 6 * framesInFlight;   // +1 for vismask
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = framesInFlight;
    poolSizes[2].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = framesInFlight;

    VkDescriptorPoolCreateInfo poolCi{};
    poolCi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets       = framesInFlight;
    poolCi.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCi.pPoolSizes    = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device, &poolCi, nullptr, &cull.descriptorPool));

    std::vector<VkDescriptorSetLayout> layouts(framesInFlight, cull.setLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = cull.descriptorPool;
    allocInfo.descriptorSetCount = framesInFlight;
    allocInfo.pSetLayouts        = layouts.data();
    cull.descriptorSets.resize(framesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, cull.descriptorSets.data()));

    std::cout << "[VulkanEngine] GPU-cull compute pipeline created\n";
}

void writeGpuCullDescriptors(GpuCullData& cull, VkDevice device,
                             const CandidateBuffer&   candidates,
                             const BatchHeaderBuffer& batches,
                             const InstanceBuffer&    mainInstance,
                             const InstanceBuffer&    shadowInstance,
                             const IndirectBuffer&    indirect,
                             VkImageView              hzbView,
                             VkSampler                hzbSampler)
{
    const uint32_t frames = static_cast<uint32_t>(cull.descriptorSets.size());
    for (uint32_t f = 0; f < frames; ++f) {
        std::array<VkDescriptorBufferInfo, 5> bi{};
        bi[0] = { candidates.buffers[f].buffer,    0, VK_WHOLE_SIZE };
        bi[1] = { batches.buffers[f].buffer,       0, VK_WHOLE_SIZE };
        bi[2] = { mainInstance.buffers[f].buffer,  0, VK_WHOLE_SIZE };
        bi[3] = { shadowInstance.buffers[f].buffer,0, VK_WHOLE_SIZE };
        bi[4] = { indirect.buffers[f].buffer,      0, VK_WHOLE_SIZE };

        VkDescriptorBufferInfo paramsBi{};
        paramsBi.buffer = cull.paramsBuffers[f].buffer;
        paramsBi.offset = 0;
        paramsBi.range  = sizeof(CullParamsUBO);

        VkDescriptorImageInfo hzbInfo{};
        hzbInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        hzbInfo.imageView   = hzbView;
        hzbInfo.sampler     = hzbSampler;

        VkDescriptorBufferInfo visBi{};
        visBi.buffer = cull.visMaskBuffers[f].buffer;
        visBi.offset = 0;
        visBi.range  = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 8> writes{};
        for (uint32_t i = 0; i < 5; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = cull.descriptorSets[f];
            writes[i].dstBinding      = i;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo     = &bi[i];
        }
        writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet          = cull.descriptorSets[f];
        writes[5].dstBinding      = 5;
        writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo     = &paramsBi;
        writes[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet          = cull.descriptorSets[f];
        writes[6].dstBinding      = 6;
        writes[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].descriptorCount = 1;
        writes[6].pImageInfo      = &hzbInfo;
        writes[7].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet          = cull.descriptorSets[f];
        writes[7].dstBinding      = 7;
        writes[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].descriptorCount = 1;
        writes[7].pBufferInfo     = &visBi;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void dispatchCull(VkCommandBuffer cmd, GpuCullData& cull, uint32_t frame,
                  const CullParamsUBO& params, uint32_t pass)
{
    if (params.numCandidates == 0) return;

    // The UBO is the same for both passes within a frame; only re-uploading on
    // pass 0 keeps the host write traffic down. (Pass 1 reads the same buffer.)
    if (pass == 0) {
        std::memcpy(cull.paramsMapped[frame], &params, sizeof(params));
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull.pipelineLayout,
                            0, 1, &cull.descriptorSets[frame], 0, nullptr);
    vkCmdPushConstants(cmd, cull.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(uint32_t), &pass);

    constexpr uint32_t LOCAL_SIZE_X = 64;
    uint32_t groups = (params.numCandidates + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
    vkCmdDispatch(cmd, groups, 1, 1);
}

void destroyGpuCull(VkDevice device, GpuCullData& cull) {
    for (auto& b : cull.paramsBuffers)  destroyBuffer(device, b);
    for (auto& b : cull.visMaskBuffers) destroyBuffer(device, b);
    cull.paramsBuffers.clear();
    cull.paramsMapped.clear();
    cull.visMaskBuffers.clear();
    if (cull.descriptorPool) {
        vkDestroyDescriptorPool(device, cull.descriptorPool, nullptr);
        cull.descriptorPool = VK_NULL_HANDLE;
    }
    cull.descriptorSets.clear();
    if (cull.pipeline)       vkDestroyPipeline(device, cull.pipeline, nullptr);
    if (cull.pipelineLayout) vkDestroyPipelineLayout(device, cull.pipelineLayout, nullptr);
    if (cull.setLayout)      vkDestroyDescriptorSetLayout(device, cull.setLayout, nullptr);
    cull.pipeline = VK_NULL_HANDLE;
    cull.pipelineLayout = VK_NULL_HANDLE;
    cull.setLayout = VK_NULL_HANDLE;
}
