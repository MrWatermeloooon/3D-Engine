#include "pipeline.h"
#include "vertex.h"
#include "../utils/vk_check.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <glm/glm.hpp>

// ── Shader loading ──────────────────────────────────────────────────────────

static std::vector<char> readFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

VkShaderModule loadShaderModule(VkDevice device, const std::string& filepath) {
    auto code = readFile(filepath);

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &shaderModule));
    return shaderModule;
}

// ── Main graphics pipeline ──────────────────────────────────────────────────

PipelineData createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D extent,
                                    VkDescriptorSetLayout sceneSetLayout,
                                    VkDescriptorSetLayout materialSetLayout,
                                    const std::string& vertPath, const std::string& fragPath)
{
    PipelineData result;

    auto vertModule = loadShaderModule(device, vertPath);
    auto fragModule = loadShaderModule(device, fragPath);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDescs   = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &blendAttachment;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates;

    // Push constant ranges: model matrix (vertex) + material (fragment)
    VkPushConstantRange pushRanges[2]{};
    pushRanges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRanges[0].offset     = 0;
    pushRanges[0].size       = sizeof(glm::mat4);            // 64

    pushRanges[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRanges[1].offset     = sizeof(glm::mat4);
    pushRanges[1].size       = sizeof(glm::vec4) + 4 * sizeof(float); // 32 (color + metallic/roughness/pads)

    VkDescriptorSetLayout setLayouts[] = { sceneSetLayout, materialSetLayout };

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 2;
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 2;
    layoutInfo.pPushConstantRanges    = pushRanges;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &result.pipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = result.pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                       &result.graphicsPipeline));

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);

    std::cout << "[VulkanEngine] Graphics pipeline created\n";
    return result;
}

// ── Shadow pipeline (depth-only) ────────────────────────────────────────────

PipelineData createShadowPipeline(VkDevice device, VkRenderPass shadowRenderPass,
                                  uint32_t shadowMapSize,
                                  const std::string& vertPath)
{
    PipelineData result;

    auto vertModule = loadShaderModule(device, vertPath);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName  = "main";

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDescs   = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width    = static_cast<float>(shadowMapSize);
    viewport.height   = static_cast<float>(shadowMapSize);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = { shadowMapSize, shadowMapSize };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_FRONT_BIT;       // peter-panning prevention
    rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable         = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor    = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 0;

    // No dynamic state — viewport/scissor are baked at shadow map size

    // Push constants: lightVP (offset 0) + model (offset 64), both vertex stage
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(glm::mat4) * 2;  // 128 bytes

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 0;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &result.pipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 1;
    pipelineInfo.pStages             = &vertStage;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.layout              = result.pipelineLayout;
    pipelineInfo.renderPass          = shadowRenderPass;
    pipelineInfo.subpass             = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                       &result.graphicsPipeline));

    vkDestroyShaderModule(device, vertModule, nullptr);

    std::cout << "[VulkanEngine] Shadow pipeline created\n";
    return result;
}
