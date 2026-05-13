#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include "vertex.h"
#include "buffer.h"
#include "raytracing.h"

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;

    // Local-space AABB (computed from vertices)
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};

    // Per-mesh BLAS, populated by uploadMesh when RT_SUPPORTED. Empty otherwise.
    RtMeshGeometry rt;
};

Mesh loadMeshFromObj(const std::string& filepath);
Mesh createCubeMesh();

// Procedural unit icosphere. `subdivisions` = 0 yields a 20-tri icosahedron;
// each step quadruples the triangle count. Useful for generating LOD chains
// (e.g. subdivisions 3, 2, 1, 0 → 1280, 320, 80, 20 tris). UVs are spherical;
// the seam is left unfixed (acceptable for untextured / colored demo meshes).
Mesh createIcosphereMesh(uint32_t subdivisions);

void uploadMesh(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue);
void destroyMesh(VkDevice device, Mesh& mesh);
