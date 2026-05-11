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

    // Local-space AABB (computed from vertices)
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};
};

Mesh loadMeshFromObj(const std::string& filepath);
Mesh createCubeMesh();

void uploadMesh(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue);
void destroyMesh(VkDevice device, Mesh& mesh);
