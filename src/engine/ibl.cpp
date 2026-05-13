#include "ibl.h"
#include "buffer.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <stb_image.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>

// ── Local helpers ───────────────────────────────────────────────────────────

namespace {

AllocatedImage createCubeImage(VkDevice, uint32_t faceSize, uint32_t mipCount,
                               VkFormat format, VkImageUsageFlags usage)
{
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { faceSize, faceSize, 1 };
    ici.mipLevels     = mipCount;
    ici.arrayLayers   = 6;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    AllocatedImage out{};
    VK_CHECK(vmaCreateImage(gVmaAllocator, &ici, &aci, &out.image, &out.allocation, nullptr));
    return out;
}

AllocatedImage create2DImage(VkDevice, uint32_t w, uint32_t h, VkFormat format,
                             VkImageUsageFlags usage)
{
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { w, h, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    AllocatedImage out{};
    VK_CHECK(vmaCreateImage(gVmaAllocator, &ici, &aci, &out.image, &out.allocation, nullptr));
    return out;
}

VkImageView createCubeView(VkDevice device, VkImage image, VkFormat format,
                           uint32_t baseMip, uint32_t mipCount)
{
    VkImageViewCreateInfo v{};
    v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.image    = image;
    v.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    v.format   = format;
    v.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.baseMipLevel   = baseMip;
    v.subresourceRange.levelCount     = mipCount;
    v.subresourceRange.baseArrayLayer = 0;
    v.subresourceRange.layerCount     = 6;
    VkImageView view{};
    VK_CHECK(vkCreateImageView(device, &v, nullptr, &view));
    return view;
}

VkImageView create2DView(VkDevice device, VkImage image, VkFormat format)
{
    VkImageViewCreateInfo v{};
    v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.image    = image;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D;
    v.format   = format;
    v.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.baseMipLevel   = 0;
    v.subresourceRange.levelCount     = 1;
    v.subresourceRange.baseArrayLayer = 0;
    v.subresourceRange.layerCount     = 1;
    VkImageView view{};
    VK_CHECK(vkCreateImageView(device, &v, nullptr, &view));
    return view;
}

VkSampler createSampler(VkDevice device, VkSamplerAddressMode mode,
                        bool linearMips, float maxLod)
{
    VkSamplerCreateInfo s{};
    s.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter    = VK_FILTER_LINEAR;
    s.minFilter    = VK_FILTER_LINEAR;
    s.mipmapMode   = linearMips ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.addressModeU = mode;
    s.addressModeV = mode;
    s.addressModeW = mode;
    s.minLod       = 0.0f;
    s.maxLod       = maxLod;
    VkSampler out{};
    VK_CHECK(vkCreateSampler(device, &s, nullptr, &out));
    return out;
}

VkShaderModule loadShader(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Failed to open shader: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> code(sz);
    f.seekg(0); f.read(code.data(), sz);

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m{};
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &m));
    return m;
}

// Image-layout barrier for the cubemap/2D bake helpers. baseMip/levelCount
// allow targeting subsets when transitioning the prefilter pyramid.
void barrier(VkCommandBuffer cmd, VkImage image,
             VkImageLayout oldLayout, VkImageLayout newLayout,
             VkAccessFlags srcAccess, VkAccessFlags dstAccess,
             VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
             uint32_t baseMip, uint32_t mipCount, uint32_t layerCount)
{
    VkImageMemoryBarrier b{};
    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout     = oldLayout;
    b.newLayout     = newLayout;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image         = image;
    b.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseMipLevel   = baseMip;
    b.subresourceRange.levelCount     = mipCount;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount     = layerCount;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Lightweight 2D HDR-equirect texture used as an *input* to the env bake. We
// don't keep it alive past the bake — it's transient.
struct EquirectTexture {
    AllocatedImage image{};
    VkImageView    view    = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    bool           valid   = false;
};

EquirectTexture loadHdrEquirect(VkPhysicalDevice physicalDevice, VkDevice device,
                                VkCommandPool commandPool, VkQueue queue,
                                const std::string& path)
{
    EquirectTexture out{};

    int w = 0, h = 0, channels = 0;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        std::cerr << "[IBL] Failed to load HDR equirect: " << path
                  << " — falling back to procedural sky.\n";
        return out;
    }
    const VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4 * sizeof(float);

    auto staging = createBuffer(physicalDevice, device, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, pixels, static_cast<size_t>(size));
    stbi_image_free(pixels);

    out.image = create2DImage(device, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                              VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    auto cmd = beginSingleTimeCommands(device, commandPool);
    barrier(cmd, out.image.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, 1);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    vkCmdCopyBufferToImage(cmd, staging.buffer, out.image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(cmd, out.image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, 1);
    endSingleTimeCommands(device, commandPool, queue, cmd);
    destroyBuffer(device, staging);

    out.view    = create2DView(device, out.image.image, VK_FORMAT_R32G32B32A32_SFLOAT);
    out.sampler = createSampler(device, VK_SAMPLER_ADDRESS_MODE_REPEAT, false, 1.0f);
    out.valid   = true;
    return out;
}

void destroyEquirect(VkDevice device, EquirectTexture& tex) {
    if (tex.sampler) vkDestroySampler(device, tex.sampler, nullptr);
    if (tex.view)    vkDestroyImageView(device, tex.view, nullptr);
    if (tex.image.image)
        vmaDestroyImage(gVmaAllocator, tex.image.image, tex.image.allocation);
    tex = {};
}

// One-shot compute pipeline holder. Made for the IBL bakes — built fresh per
// bake call, torn down before returning. Keeps the IBL state in IblData small
// since the pipelines are only needed at bake time.
struct ComputePipe {
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    VkPipelineLayout      layout    = VK_NULL_HANDLE;
    VkPipeline            pipeline  = VK_NULL_HANDLE;
    VkShaderModule        shader    = VK_NULL_HANDLE;
};

ComputePipe makeComputePipe(VkDevice device, const std::string& spirv,
                            const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                            uint32_t pushSize)
{
    ComputePipe p{};
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(bindings.size());
    dslci.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &p.dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = pushSize;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &p.dsl;
    plci.pushConstantRangeCount = pushSize ? 1u : 0u;
    plci.pPushConstantRanges    = pushSize ? &pcr : nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &p.layout));

    p.shader = loadShader(device, spirv);
    VkPipelineShaderStageCreateInfo ssci{};
    ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = p.shader;
    ssci.pName  = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = ssci;
    cpci.layout = p.layout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline));
    return p;
}

void destroyComputePipe(VkDevice device, ComputePipe& p) {
    if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
    if (p.layout)   vkDestroyPipelineLayout(device, p.layout, nullptr);
    if (p.dsl)      vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    if (p.shader)   vkDestroyShaderModule(device, p.shader, nullptr);
    p = {};
}

} // namespace

// ── Public API ──────────────────────────────────────────────────────────────

void createIbl(IblData& ibl, VkPhysicalDevice /*physicalDevice*/, VkDevice device)
{
    const VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Environment cubemap — sampled by sky.frag and used as the bake source.
    {
        AllocatedImage img = createCubeImage(device, ibl.envFaceSize, 1, fmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
          | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        ibl.envImage = img.image;
        ibl.envAlloc = img.allocation;
        ibl.envView  = createCubeView(device, ibl.envImage, fmt, 0, 1);
    }

    // Prefilter cubemap with mips. The full-chain view is for sampling; the
    // per-mip views are storage-image writes from the compute bake.
    {
        AllocatedImage img = createCubeImage(device, ibl.prefilterFaceSize,
            ibl.prefilterMipCount, fmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
          | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ibl.prefilterImage = img.image;
        ibl.prefilterAlloc = img.allocation;
        ibl.prefilterView  = createCubeView(device, ibl.prefilterImage, fmt, 0,
                                            ibl.prefilterMipCount);
        ibl.prefilterMipViews.resize(ibl.prefilterMipCount);
        for (uint32_t m = 0; m < ibl.prefilterMipCount; ++m) {
            ibl.prefilterMipViews[m] = createCubeView(device, ibl.prefilterImage, fmt, m, 1);
        }
    }

    // Irradiance cubemap (no mips).
    {
        AllocatedImage img = createCubeImage(device, ibl.irradianceFaceSize, 1, fmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        ibl.irradianceImage = img.image;
        ibl.irradianceAlloc = img.allocation;
        ibl.irradianceView  = createCubeView(device, ibl.irradianceImage, fmt, 0, 1);
    }

    // BRDF LUT — RG16F 2D.
    {
        const VkFormat lutFmt = VK_FORMAT_R16G16_SFLOAT;
        AllocatedImage img = create2DImage(device, ibl.brdfLutSize, ibl.brdfLutSize, lutFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        ibl.brdfLutImage = img.image;
        ibl.brdfLutAlloc = img.allocation;
        ibl.brdfLutView  = create2DView(device, ibl.brdfLutImage, lutFmt);
    }

    // Samplers.
    ibl.cubeSampler = createSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    true, static_cast<float>(ibl.prefilterMipCount));
    ibl.lutSampler  = createSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, false, 1.0f);
}

void destroyIbl(VkDevice device, IblData& ibl)
{
    if (ibl.cubeSampler) vkDestroySampler(device, ibl.cubeSampler, nullptr);
    if (ibl.lutSampler)  vkDestroySampler(device, ibl.lutSampler,  nullptr);

    if (ibl.brdfLutView)  vkDestroyImageView(device, ibl.brdfLutView,  nullptr);
    if (ibl.brdfLutImage) vmaDestroyImage(gVmaAllocator, ibl.brdfLutImage, ibl.brdfLutAlloc);

    if (ibl.irradianceView)  vkDestroyImageView(device, ibl.irradianceView, nullptr);
    if (ibl.irradianceImage) vmaDestroyImage(gVmaAllocator, ibl.irradianceImage, ibl.irradianceAlloc);

    for (auto v : ibl.prefilterMipViews) if (v) vkDestroyImageView(device, v, nullptr);
    ibl.prefilterMipViews.clear();
    if (ibl.prefilterView)  vkDestroyImageView(device, ibl.prefilterView, nullptr);
    if (ibl.prefilterImage) vmaDestroyImage(gVmaAllocator, ibl.prefilterImage, ibl.prefilterAlloc);

    if (ibl.envView)  vkDestroyImageView(device, ibl.envView, nullptr);
    if (ibl.envImage) vmaDestroyImage(gVmaAllocator, ibl.envImage, ibl.envAlloc);

    ibl = {};
}

// ── Bake ────────────────────────────────────────────────────────────────────

namespace {

struct EnvPush {
    uint32_t sourceMode;
    uint32_t faceSize;
    float    sunIntensity;
    float    intensity;       // multiplies the final env colour (sky + sun)
    glm::vec4 sunDirection;  // .w = sun strength multiplier (1.0 default)
    glm::vec4 zenith;
    glm::vec4 horizon;
    glm::vec4 ground;
};

struct IrrPush {
    uint32_t faceSize;
    float    sampleDelta;
    float    pad0, pad1;
};

struct PrePush {
    float    roughness;
    uint32_t faceSize;
    uint32_t envFaceSize;
    uint32_t sampleCount;
};

struct BrdfPush {
    uint32_t size;
    uint32_t sampleCount;
};

// Build one transient descriptor pool sized for the worst-case writes inside
// a bake. Used + freed within bakeIbl.
VkDescriptorPool makeBakePool(VkDevice device, uint32_t setCount)
{
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,        setCount * 2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, setCount * 2 },
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = setCount;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    VkDescriptorPool pool{};
    VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &pool));
    return pool;
}

VkDescriptorSet allocSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout dsl)
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &dsl;
    VkDescriptorSet set{};
    VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));
    return set;
}

} // namespace

void bakeIbl(IblData& ibl, VkPhysicalDevice physicalDevice, VkDevice device,
             VkCommandPool commandPool, VkQueue queue,
             const IblBakeParams& params)
{
    // Optional HDR source.
    EquirectTexture equirect{};
    if (!params.hdrPath.empty()) {
        equirect = loadHdrEquirect(physicalDevice, device, commandPool, queue,
                                   params.hdrPath);
    }
    const bool useHdr = equirect.valid;

    // ── Build compute pipelines ─────────────────────────────────────────
    // Env: binding 0 = storage cube (writeonly), binding 1 = sampler2D (equirect)
    ComputePipe envPipe = makeComputePipe(device, "shaders/ibl_env.comp.spv",
        {{
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }}, sizeof(EnvPush));

    // Irradiance: binding 0 = storage cube (writeonly), 1 = samplerCube
    ComputePipe irrPipe = makeComputePipe(device, "shaders/ibl_irradiance.comp.spv",
        {{
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }}, sizeof(IrrPush));

    // Prefilter: same shape as irradiance (one set per mip → multiple allocs)
    ComputePipe prePipe = makeComputePipe(device, "shaders/ibl_prefilter.comp.spv",
        {{
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }}, sizeof(PrePush));

    // BRDF LUT: just one storage image binding.
    ComputePipe brdfPipe = makeComputePipe(device, "shaders/ibl_brdf.comp.spv",
        {{
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }}, sizeof(BrdfPush));

    // Descriptor pool sized for: 1 env + 1 irr + N prefilter mips + 1 brdf
    const uint32_t setCount = 2 + ibl.prefilterMipCount + 1;
    VkDescriptorPool pool   = makeBakePool(device, setCount);

    // We need a fallback sampler when no equirect is loaded — the binding
    // still has to be filled with *something* even though the shader won't
    // sample it (sourceMode = 0 branches off). Reuse the env cube view here
    // as a benign sampled image; it's in UNDEFINED layout pre-bake so we use
    // a 1×1 dummy 2D instead.
    EquirectTexture dummy{};
    if (!useHdr) {
        // 1×1 magenta pixel (clearly visible if accidentally sampled).
        const float magenta[4] = { 1.0f, 0.0f, 1.0f, 1.0f };
        auto staging = createBuffer(physicalDevice, device, sizeof(magenta),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(staging.mapped, magenta, sizeof(magenta));
        dummy.image = create2DImage(device, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        auto c = beginSingleTimeCommands(device, commandPool);
        barrier(c, dummy.image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, 1);
        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent      = { 1, 1, 1 };
        vkCmdCopyBufferToImage(c, staging.buffer, dummy.image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        barrier(c, dummy.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, 1);
        endSingleTimeCommands(device, commandPool, queue, c);
        destroyBuffer(device, staging);
        dummy.view    = create2DView(device, dummy.image.image, VK_FORMAT_R32G32B32A32_SFLOAT);
        dummy.sampler = createSampler(device, VK_SAMPLER_ADDRESS_MODE_REPEAT, false, 1.0f);
        dummy.valid   = true;
    }

    auto cmd = beginSingleTimeCommands(device, commandPool);

    // ── Env bake ────────────────────────────────────────────────────────
    barrier(cmd, ibl.envImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, 6);

    VkDescriptorSet envSet = allocSet(device, pool, envPipe.dsl);
    {
        VkDescriptorImageInfo envWrite{};
        envWrite.imageView   = ibl.envView;
        envWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo eqSampled{};
        eqSampled.imageView   = useHdr ? equirect.view    : dummy.view;
        eqSampled.sampler     = useHdr ? equirect.sampler : dummy.sampler;
        eqSampled.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = envSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo = &envWrite;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = envSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[1].pImageInfo = &eqSampled;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envPipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envPipe.layout,
                            0, 1, &envSet, 0, nullptr);
    {
        EnvPush ep{};
        ep.sourceMode   = useHdr ? 1u : 0u;
        ep.faceSize     = ibl.envFaceSize;
        ep.sunIntensity = params.sunIntensity;
        ep.intensity    = params.intensity;
        // .w field carries sun strength multiplier so the procedural shader
        // can independently scale disc + halo. 1.0 keeps the present tuning.
        ep.sunDirection = glm::vec4(glm::normalize(params.sunDir), 1.0f);
        ep.zenith       = glm::vec4(params.zenithColor,  0.0f);
        ep.horizon      = glm::vec4(params.horizonColor, 0.0f);
        ep.ground       = glm::vec4(params.groundColor,  0.0f);
        vkCmdPushConstants(cmd, envPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ep), &ep);
    }
    {
        uint32_t g = (ibl.envFaceSize + 7) / 8;
        vkCmdDispatch(cmd, g, g, 6);
    }

    // Env GENERAL → SHADER_READ_ONLY for downstream sampling.
    barrier(cmd, ibl.envImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, 6);

    // ── Irradiance bake ─────────────────────────────────────────────────
    barrier(cmd, ibl.irradianceImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, 6);

    VkDescriptorSet irrSet = allocSet(device, pool, irrPipe.dsl);
    {
        VkDescriptorImageInfo irrWrite{};
        irrWrite.imageView   = ibl.irradianceView;
        irrWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo envRead{};
        envRead.imageView   = ibl.envView;
        envRead.sampler     = ibl.cubeSampler;
        envRead.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = irrSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[0].pImageInfo = &irrWrite;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = irrSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &envRead;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irrPipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irrPipe.layout,
                            0, 1, &irrSet, 0, nullptr);
    {
        IrrPush ip{};
        ip.faceSize    = ibl.irradianceFaceSize;
        ip.sampleDelta = 0.05f;     // ~1800 samples per texel
        vkCmdPushConstants(cmd, irrPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ip), &ip);
    }
    {
        uint32_t g = (ibl.irradianceFaceSize + 7) / 8;
        vkCmdDispatch(cmd, g, g, 6);
    }
    barrier(cmd, ibl.irradianceImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, 6);

    // ── Prefilter bake (one dispatch per mip) ───────────────────────────
    barrier(cmd, ibl.prefilterImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, ibl.prefilterMipCount, 6);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prePipe.pipeline);
    for (uint32_t m = 0; m < ibl.prefilterMipCount; ++m) {
        VkDescriptorSet set = allocSet(device, pool, prePipe.dsl);
        VkDescriptorImageInfo preWrite{};
        preWrite.imageView   = ibl.prefilterMipViews[m];
        preWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo envRead{};
        envRead.imageView   = ibl.envView;
        envRead.sampler     = ibl.cubeSampler;
        envRead.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[0].pImageInfo = &preWrite;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &envRead;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prePipe.layout,
                                0, 1, &set, 0, nullptr);
        PrePush pp{};
        pp.roughness   = static_cast<float>(m) / static_cast<float>(ibl.prefilterMipCount - 1);
        pp.faceSize    = std::max(1u, ibl.prefilterFaceSize >> m);
        pp.envFaceSize = ibl.envFaceSize;
        pp.sampleCount = (m == 0) ? 1u : 512u;  // mip 0 = mirror, no integration needed
        vkCmdPushConstants(cmd, prePipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pp), &pp);
        uint32_t g = (pp.faceSize + 7) / 8;
        vkCmdDispatch(cmd, g, g, 6);
    }
    barrier(cmd, ibl.prefilterImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, ibl.prefilterMipCount, 6);

    // ── BRDF LUT bake ───────────────────────────────────────────────────
    barrier(cmd, ibl.brdfLutImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, 1);

    VkDescriptorSet brdfSet = allocSet(device, pool, brdfPipe.dsl);
    {
        VkDescriptorImageInfo brdfWrite{};
        brdfWrite.imageView   = ibl.brdfLutView;
        brdfWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = brdfSet; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &brdfWrite;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfPipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfPipe.layout,
                            0, 1, &brdfSet, 0, nullptr);
    {
        BrdfPush bp{};
        bp.size        = ibl.brdfLutSize;
        bp.sampleCount = 1024;
        vkCmdPushConstants(cmd, brdfPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(bp), &bp);
        uint32_t g = (ibl.brdfLutSize + 7) / 8;
        vkCmdDispatch(cmd, g, g, 1);
    }
    barrier(cmd, ibl.brdfLutImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, 1);

    endSingleTimeCommands(device, commandPool, queue, cmd);

    // ── Cleanup transient state ─────────────────────────────────────────
    vkDestroyDescriptorPool(device, pool, nullptr);
    destroyComputePipe(device, envPipe);
    destroyComputePipe(device, irrPipe);
    destroyComputePipe(device, prePipe);
    destroyComputePipe(device, brdfPipe);
    if (equirect.valid) destroyEquirect(device, equirect);
    if (dummy.valid)    destroyEquirect(device, dummy);
}
