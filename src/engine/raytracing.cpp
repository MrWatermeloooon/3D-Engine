#include "raytracing.h"
#include "vulkan_init.h"
#include "mesh.h"
#include "../utils/vk_check.h"

#include <cstring>
#include <stdexcept>

// ── Lifecycle ───────────────────────────────────────────────────────────────

void createRtScene(RtScene& scene, VkDevice /*device*/, uint32_t framesInFlight) {
    if (!RT_SUPPORTED) return;
    scene.tlas.assign(framesInFlight, VK_NULL_HANDLE);
    scene.tlasBuffer.assign(framesInFlight, AllocatedBuffer{});
    scene.instanceBuffer.assign(framesInFlight, AllocatedBuffer{});
    scene.materialBuffer.assign(framesInFlight, AllocatedBuffer{});
    scene.scratchBuffer.assign(framesInFlight, AllocatedBuffer{});
    scene.scratchCapacity.assign(framesInFlight, 0);
    scene.tlasCapacity.assign(framesInFlight, 0);
    scene.instanceCapacity.assign(framesInFlight, 0);
    scene.materialCapacity.assign(framesInFlight, 0);
}

void destroyRtScene(VkDevice device, RtScene& scene) {
    if (!RT_SUPPORTED) { scene = {}; return; }
    for (auto as : scene.tlas) if (as && RT_DestroyAS) RT_DestroyAS(device, as, nullptr);
    for (auto& b : scene.tlasBuffer)     destroyBuffer(device, b);
    for (auto& b : scene.instanceBuffer) destroyBuffer(device, b);
    for (auto& b : scene.materialBuffer) destroyBuffer(device, b);
    for (auto& b : scene.scratchBuffer)  destroyBuffer(device, b);
    scene = {};
}

void destroyRtMeshGeometry(VkDevice device, RtMeshGeometry& geo) {
    if (geo.accel && RT_DestroyAS) RT_DestroyAS(device, geo.accel, nullptr);
    destroyBuffer(device, geo.accelBuffer);
    geo = {};
}

VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return RT_GetBufferDeviceAddress(device, &info);
}

// ── BLAS build ──────────────────────────────────────────────────────────────

void buildMeshBlas(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                   VkCommandPool commandPool, VkQueue queue)
{
    if (!RT_SUPPORTED) return;
    const uint32_t triCount = static_cast<uint32_t>(mesh.indices.size()) / 3;
    if (triCount == 0) return;

    // ── Describe geometry ──────────────────────────────────────────────────
    VkAccelerationStructureGeometryTrianglesDataKHR tri{};
    tri.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = getBufferDeviceAddress(device, mesh.vertexBuffer.buffer);
    tri.vertexStride = sizeof(Vertex);
    tri.maxVertex    = static_cast<uint32_t>(mesh.vertices.size()) - 1;
    tri.indexType    = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress  = getBufferDeviceAddress(device, mesh.indexBuffer.buffer);
    tri.transformData.deviceAddress = 0;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.geometry.triangles = tri;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount  = triCount;
    range.primitiveOffset = 0;
    range.firstVertex     = 0;
    range.transformOffset = 0;

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount            = 1;
    build.pGeometries              = &geom;

    // ── Query sizes ────────────────────────────────────────────────────────
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    RT_GetASBuildSizes(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &triCount, &sizes);

    // ── Allocate backing buffer + scratch ─────────────────────────────────
    mesh.rt.accelBuffer = createBuffer(physicalDevice, device, sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = mesh.rt.accelBuffer.buffer;
    ci.size   = sizes.accelerationStructureSize;
    ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK(RT_CreateAS(device, &ci, nullptr, &mesh.rt.accel));

    AllocatedBuffer scratch = createBuffer(physicalDevice, device, sizes.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    build.dstAccelerationStructure   = mesh.rt.accel;
    build.scratchData.deviceAddress  = getBufferDeviceAddress(device, scratch.buffer);

    // ── Submit build ──────────────────────────────────────────────────────
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    RT_CmdBuildAS(cmd, 1, &build, &pRange);
    endSingleTimeCommands(device, commandPool, queue, cmd);

    // Scratch is one-shot; drop it now that the build has completed.
    destroyBuffer(device, scratch);

    // Cache the BLAS device address for TLAS instance writes.
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = mesh.rt.accel;
    mesh.rt.deviceAddress = RT_GetASDeviceAddress(device, &addrInfo);
}

// ── TLAS build ──────────────────────────────────────────────────────────────

static AllocatedBuffer ensureBuffer(AllocatedBuffer& existing, VkDeviceSize& capacity,
                                    VkPhysicalDevice physicalDevice, VkDevice device,
                                    VkDeviceSize requested, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags properties)
{
    if (existing.buffer != VK_NULL_HANDLE && capacity >= requested) return existing;
    destroyBuffer(device, existing);
    existing = createBuffer(physicalDevice, device, requested, usage, properties);
    capacity = requested;
    return existing;
}

void buildTlas(RtScene& scene, uint32_t frame,
               VkPhysicalDevice physicalDevice, VkDevice device,
               VkCommandBuffer cmd,
               const std::vector<RtInstanceDesc>& instances)
{
    if (!RT_SUPPORTED) return;
    const uint32_t count = static_cast<uint32_t>(instances.size());

    // ── Pack VkAccelerationStructureInstanceKHR records ────────────────────
    // VkTransformMatrixKHR is row-major 3×4 (3 rows of 4 floats). glm matrices
    // are column-major 4×4, so we transpose-truncate here. customIndex is the
    // array index into the parallel material buffer (sequential 0..count-1).
    std::vector<VkAccelerationStructureInstanceKHR> records(count);
    std::vector<RtInstanceMaterial>                 materials(count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& src = instances[i];
        VkTransformMatrixKHR T{};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col)
                T.matrix[row][col] = src.transform[col][row];
        records[i].transform                              = T;
        records[i].instanceCustomIndex                    = i & 0xFFFFFF;
        records[i].mask                                   = 0xFF;
        records[i].instanceShaderBindingTableRecordOffset = 0;
        records[i].flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        records[i].accelerationStructureReference         = src.blasAddress;

        materials[i] = src.material;
    }

    // ── Upload instance records (host-visible, persistently mapped) ───────
    const VkDeviceSize recordBytes =
        std::max<VkDeviceSize>(1, sizeof(VkAccelerationStructureInstanceKHR) * count);
    ensureBuffer(scene.instanceBuffer[frame], scene.instanceCapacity[frame],
                 physicalDevice, device, recordBytes,
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
               | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (count > 0) {
        void* dst = nullptr;
        vkMapMemory(device, scene.instanceBuffer[frame].memory, 0, recordBytes, 0, &dst);
        std::memcpy(dst, records.data(),
                    sizeof(VkAccelerationStructureInstanceKHR) * count);
        vkUnmapMemory(device, scene.instanceBuffer[frame].memory);
    }

    // ── Upload parallel material buffer (host-visible) ────────────────────
    const VkDeviceSize matBytes =
        std::max<VkDeviceSize>(sizeof(RtInstanceMaterial),
                               sizeof(RtInstanceMaterial) * count);
    ensureBuffer(scene.materialBuffer[frame], scene.materialCapacity[frame],
                 physicalDevice, device, matBytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
               | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (count > 0) {
        void* dst = nullptr;
        vkMapMemory(device, scene.materialBuffer[frame].memory, 0, matBytes, 0, &dst);
        std::memcpy(dst, materials.data(),
                    sizeof(RtInstanceMaterial) * count);
        vkUnmapMemory(device, scene.materialBuffer[frame].memory);
    }

    // ── Geometry pointing at the instance buffer ──────────────────────────
    VkAccelerationStructureGeometryInstancesDataKHR inst{};
    inst.sType            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst.arrayOfPointers  = VK_FALSE;
    inst.data.deviceAddress = (count > 0)
        ? getBufferDeviceAddress(device, scene.instanceBuffer[frame].buffer)
        : 0;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = inst;

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                   | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount            = 1;
    build.pGeometries              = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    RT_GetASBuildSizes(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &count, &sizes);

    // ── Resize / create TLAS storage and scratch ──────────────────────────
    // If the TLAS storage grew we have to recreate the TLAS handle too.
    bool tlasResized = false;
    if (scene.tlasCapacity[frame] < sizes.accelerationStructureSize
        || scene.tlas[frame] == VK_NULL_HANDLE) {
        if (scene.tlas[frame] != VK_NULL_HANDLE) {
            RT_DestroyAS(device, scene.tlas[frame], nullptr);
            scene.tlas[frame] = VK_NULL_HANDLE;
        }
        destroyBuffer(device, scene.tlasBuffer[frame]);
        scene.tlasBuffer[frame] = createBuffer(physicalDevice, device,
            sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        scene.tlasCapacity[frame] = sizes.accelerationStructureSize;

        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = scene.tlasBuffer[frame].buffer;
        ci.size   = sizes.accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(RT_CreateAS(device, &ci, nullptr, &scene.tlas[frame]));
        tlasResized = true;
    }
    (void)tlasResized;

    ensureBuffer(scene.scratchBuffer[frame], scene.scratchCapacity[frame],
                 physicalDevice, device, sizes.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    build.dstAccelerationStructure  = scene.tlas[frame];
    build.scratchData.deviceAddress = getBufferDeviceAddress(device, scene.scratchBuffer[frame].buffer);

    // ── Record the build ──────────────────────────────────────────────────
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = count;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    // Make the instance-buffer host write visible to the AS build read.
    VkMemoryBarrier preBuild{};
    preBuild.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    preBuild.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    preBuild.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                           | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &preBuild, 0, nullptr, 0, nullptr);

    RT_CmdBuildAS(cmd, 1, &build, &pRange);

    // Make the AS write visible to subsequent shader reads (ray queries).
    VkMemoryBarrier postBuild{};
    postBuild.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    postBuild.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    postBuild.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
      | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &postBuild, 0, nullptr, 0, nullptr);
}
