#include "sky.h"
#include "pipeline.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <iostream>
#include <vector>

void createSkyPipeline(SkyData& sky, VkDevice device, VkRenderPass renderPass,
                       VkExtent2D /*extent*/, VkDescriptorSetLayout sceneSetLayout,
                       const std::string& vertPath, const std::string& fragPath)
{
    auto vertModule = loadShaderModule(device, vertPath);
    auto fragModule = loadShaderModule(device, fragPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule; stages[1].pName = "main";

    // No vertex input — sky.vert builds the fullscreen tri from gl_VertexIndex.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth   = 1.0f;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Critical: keep depth test ON with LESS_OR_EQUAL so the sky only fills
    // pixels where geometry hasn't already written closer depth. Depth write
    // OFF so we don't pollute the depth buffer (HZB reduction reads it).
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Two color attachments (color + motion), like the main pipeline.
    VkPipelineColorBlendAttachmentState blends[2]{};
    blends[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blends[1] = blends[0];

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 2;
    cb.pAttachments    = blends;

    // Main pipeline declares VRS dynamic state when VRS_SUPPORTED; we omit it
    // here because the sky always wants 1×1 (it's already trivially cheap),
    // and adding the dynamic state would require a setter call between the
    // last geometry draw and the sky draw — easier to keep both pipelines
    // independent on this axis.
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dyn;

    // Bind set 0 (scene) only — the sky shader reads view, proj, cameraPos,
    // and the lights UBO from there. No material textures, no push constants.
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts    = &sceneSetLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &sky.pipelineLayout));

    VkGraphicsPipelineCreateInfo info{};
    info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount          = 2;
    info.pStages             = stages;
    info.pVertexInputState   = &vertexInput;
    info.pInputAssemblyState = &ia;
    info.pViewportState      = &vp;
    info.pRasterizationState = &rs;
    info.pMultisampleState   = &ms;
    info.pDepthStencilState  = &ds;
    info.pColorBlendState    = &cb;
    info.pDynamicState       = &dynState;
    info.layout              = sky.pipelineLayout;
    info.renderPass          = renderPass;
    info.subpass             = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                       &sky.pipeline));

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);

    std::cout << "[VulkanEngine] Sky pipeline created\n";
}

void destroySkyPipeline(VkDevice device, SkyData& sky) {
    if (sky.pipeline)       vkDestroyPipeline(device, sky.pipeline, nullptr);
    if (sky.pipelineLayout) vkDestroyPipelineLayout(device, sky.pipelineLayout, nullptr);
    sky = {};
}

void recordSkyPass(VkCommandBuffer cmd, const SkyData& sky, VkDescriptorSet sceneSet,
                   VkExtent2D extent)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky.pipelineLayout,
                            0, 1, &sceneSet, 0, nullptr);

    VkViewport vp{};
    vp.width    = static_cast<float>(extent.width);
    vp.height   = static_cast<float>(extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}
