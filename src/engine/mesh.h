#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include "vertex.h"
#include "buffer.h"

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;
};

Mesh loadMeshFromObj(const std::string& filepath);
Mesh createCubeMesh();

void uploadMesh(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue);
void destroyMesh(VkDevice device, Mesh& mesh);
