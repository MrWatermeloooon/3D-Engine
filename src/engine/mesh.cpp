#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "mesh.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

// Compute smooth tangents using the standard Lengyel method
// (https://terathon.com/blogs/tangent-space.html). For each triangle, build a
// tangent from position + UV deltas, accumulate onto each vertex, then
// orthogonalize against the vertex normal. Vertices with zero-area UVs (e.g.
// degenerate triangles or meshes without UV unwrap) fall back to a vector
// orthogonal to the normal so the TBN basis stays well-formed.
void computeMeshTangents(Mesh& mesh) {
    for (auto& v : mesh.vertices) v.tangent = glm::vec3(0.0f);

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        Vertex& v0 = mesh.vertices[mesh.indices[i + 0]];
        Vertex& v1 = mesh.vertices[mesh.indices[i + 1]];
        Vertex& v2 = mesh.vertices[mesh.indices[i + 2]];

        glm::vec3 dp1 = v1.position - v0.position;
        glm::vec3 dp2 = v2.position - v0.position;
        glm::vec2 du1 = v1.texCoord - v0.texCoord;
        glm::vec2 du2 = v2.texCoord - v0.texCoord;

        float r = du1.x * du2.y - du2.x * du1.y;
        if (glm::abs(r) < 1e-8f) continue; // degenerate UV — skip
        float invR = 1.0f / r;

        glm::vec3 t = (dp1 * du2.y - dp2 * du1.y) * invR;
        v0.tangent += t; v1.tangent += t; v2.tangent += t;
    }

    // Orthogonalize against normal + normalize. Fall back to an arbitrary
    // orthonormal axis if the accumulated tangent collapses (zero-UV mesh).
    for (auto& v : mesh.vertices) {
        glm::vec3 n = v.normal;
        glm::vec3 t = v.tangent - n * glm::dot(n, v.tangent);
        float len = glm::length(t);
        if (len > 1e-6f) {
            v.tangent = t / len;
        } else {
            // Pick the world axis least aligned with n, then orthogonalize.
            glm::vec3 a = (glm::abs(n.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
            v.tangent = glm::normalize(a - n * glm::dot(n, a));
        }
    }
}

Mesh loadMeshFromObj(const std::string& filepath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())) {
        throw std::runtime_error("Failed to load OBJ: warn=\"" + warn + "\" err=\"" + err + "\"");
    }

    Mesh mesh;
    std::unordered_map<Vertex, uint32_t> uniqueVertices;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            // Defaults — overwritten if the OBJ provides them.
            vertex.normal   = { 0.0f, 1.0f, 0.0f };
            vertex.texCoord = { 0.0f, 0.0f };
            vertex.color    = { 1.0f, 1.0f, 1.0f };

            if (index.normal_index >= 0 && !attrib.normals.empty()) {
                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };
            }

            if (index.texcoord_index >= 0 && !attrib.texcoords.empty()) {
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
            }

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vertex);
            }
            mesh.indices.push_back(uniqueVertices[vertex]);
        }
    }

    // If the OBJ had no normals, compute smooth ones from triangle face normals.
    // Use the raw (un-normalized) cross product so each triangle's contribution
    // is proportional to its area — otherwise a tiny degenerate sliver weighs
    // equally with a large face it shares a vertex with and skews the result.
    if (attrib.normals.empty()) {
        for (auto& v : mesh.vertices) v.normal = glm::vec3(0.0f);
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            auto& v0 = mesh.vertices[mesh.indices[i + 0]];
            auto& v1 = mesh.vertices[mesh.indices[i + 1]];
            auto& v2 = mesh.vertices[mesh.indices[i + 2]];
            glm::vec3 e1 = v1.position - v0.position;
            glm::vec3 e2 = v2.position - v0.position;
            glm::vec3 n  = glm::cross(e1, e2);  // length = 2 * triangle area
            v0.normal += n; v1.normal += n; v2.normal += n;
        }
        for (auto& v : mesh.vertices) {
            float len = glm::length(v.normal);
            v.normal = (len > 1e-6f) ? v.normal / len : glm::vec3(0, 1, 0);
        }
    }

    // Compute per-vertex tangents from position+UV deltas (Lengyel's method).
    // Tangents are needed for tangent-space normal mapping in mesh.frag.
    // We accumulate per-triangle tangents weighted by triangle area
    // (un-normalized cross-style accumulation) and orthogonalize against the
    // vertex normal at the end.
    computeMeshTangents(mesh);

    // Compute local AABB
    if (!mesh.vertices.empty()) {
        mesh.aabbMin = mesh.aabbMax = mesh.vertices[0].position;
        for (const auto& v : mesh.vertices) {
            mesh.aabbMin = glm::min(mesh.aabbMin, v.position);
            mesh.aabbMax = glm::max(mesh.aabbMax, v.position);
        }
    }

    std::cout << "[VulkanEngine] Loaded mesh: " << mesh.vertices.size() << " vertices, "
              << mesh.indices.size() / 3 << " triangles"
              << (attrib.normals.empty() ? " (computed normals)" : "") << "\n";
    return mesh;
}

Mesh createCubeMesh() {
    Mesh mesh;

    mesh.vertices = {
        // Front face (+Z)
        {{-0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        // Back face (-Z)
        {{ 0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        // Top face (+Y)
        {{-0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        // Bottom face (-Y)
        {{-0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        // Right face (+X)
        {{ 0.5f, -0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        // Left face (-X)
        {{-0.5f, -0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
    };

    mesh.indices = {
         0,  1,  2,  2,  3,  0,
         4,  5,  6,  6,  7,  4,
         8,  9, 10, 10, 11,  8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20,
    };

    mesh.aabbMin = { -0.5f, -0.5f, -0.5f };
    mesh.aabbMax = {  0.5f,  0.5f,  0.5f };

    computeMeshTangents(mesh);
    return mesh;
}

Mesh createIcosphereMesh(uint32_t subdivisions) {
    // Start from a regular icosahedron (12 verts, 20 faces).
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<glm::vec3> positions = {
        glm::normalize(glm::vec3(-1,  t,  0)),
        glm::normalize(glm::vec3( 1,  t,  0)),
        glm::normalize(glm::vec3(-1, -t,  0)),
        glm::normalize(glm::vec3( 1, -t,  0)),
        glm::normalize(glm::vec3( 0, -1,  t)),
        glm::normalize(glm::vec3( 0,  1,  t)),
        glm::normalize(glm::vec3( 0, -1, -t)),
        glm::normalize(glm::vec3( 0,  1, -t)),
        glm::normalize(glm::vec3( t,  0, -1)),
        glm::normalize(glm::vec3( t,  0,  1)),
        glm::normalize(glm::vec3(-t,  0, -1)),
        glm::normalize(glm::vec3(-t,  0,  1)),
    };
    std::vector<uint32_t> indices = {
         0,11, 5,  0, 5, 1,  0, 1, 7,  0, 7,10,  0,10,11,
         1, 5, 9,  5,11, 4, 11,10, 2, 10, 7, 6,  7, 1, 8,
         3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
         4, 9, 5,  2, 4,11,  6, 2,10,  8, 6, 7,  9, 8, 1,
    };

    // Subdivide. Cache midpoints so each shared edge gets a single new vertex.
    auto edgeKey = [](uint32_t a, uint32_t b) -> uint64_t {
        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        return (hi << 32) | lo;
    };
    for (uint32_t s = 0; s < subdivisions; ++s) {
        std::unordered_map<uint64_t, uint32_t> midCache;
        std::vector<uint32_t> next;
        next.reserve(indices.size() * 4);
        auto midpoint = [&](uint32_t a, uint32_t b) {
            uint64_t k = edgeKey(a, b);
            auto it = midCache.find(k);
            if (it != midCache.end()) return it->second;
            glm::vec3 m = glm::normalize(positions[a] + positions[b]);
            uint32_t idx = static_cast<uint32_t>(positions.size());
            positions.push_back(m);
            midCache[k] = idx;
            return idx;
        };
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t a = indices[i], b = indices[i+1], c = indices[i+2];
            uint32_t ab = midpoint(a, b);
            uint32_t bc = midpoint(b, c);
            uint32_t ca = midpoint(c, a);
            next.insert(next.end(), { a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca });
        }
        indices = std::move(next);
    }

    // Build Vertex array. Position == normal for a unit sphere; UV is spherical.
    Mesh mesh;
    mesh.vertices.reserve(positions.size());
    for (const auto& p : positions) {
        Vertex v{};
        v.position = p * 0.5f;          // unit-diameter sphere
        v.normal   = p;
        v.texCoord = {
            0.5f + std::atan2(p.z, p.x) / (2.0f * 3.14159265358979323846f),
            0.5f - std::asin(p.y) / 3.14159265358979323846f
        };
        v.color = { 1.0f, 1.0f, 1.0f };
        mesh.vertices.push_back(v);
    }
    mesh.indices = std::move(indices);
    mesh.aabbMin = { -0.5f, -0.5f, -0.5f };
    mesh.aabbMax = {  0.5f,  0.5f,  0.5f };
    computeMeshTangents(mesh);
    return mesh;
}

void uploadMesh(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue) {
    // When RT is supported, the vertex + index buffers must also be readable
    // by acceleration-structure builds and reachable via device address.
    VkBufferUsageFlags vbUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VkBufferUsageFlags ibUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (RT_SUPPORTED) {
        VkBufferUsageFlags rtExtra = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vbUsage |= rtExtra;
        ibUsage |= rtExtra;
    }

    // Vertex buffer via staging
    VkDeviceSize vbSize = sizeof(Vertex) * mesh.vertices.size();

    auto staging = createBuffer(physicalDevice, device, vbSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(staging.mapped, mesh.vertices.data(), vbSize);

    mesh.vertexBuffer = createBuffer(physicalDevice, device, vbSize,
        vbUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(device, commandPool, queue, staging.buffer, mesh.vertexBuffer.buffer, vbSize);
    destroyBuffer(device, staging);

    // Index buffer via staging
    VkDeviceSize ibSize = sizeof(uint32_t) * mesh.indices.size();

    staging = createBuffer(physicalDevice, device, ibSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(staging.mapped, mesh.indices.data(), ibSize);

    mesh.indexBuffer = createBuffer(physicalDevice, device, ibSize,
        ibUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(device, commandPool, queue, staging.buffer, mesh.indexBuffer.buffer, ibSize);
    destroyBuffer(device, staging);

    // Cache buffer device addresses once. They're stable for the life of the
    // mesh (the buffers don't get reallocated), so the per-frame TLAS gather
    // can read them directly instead of calling vkGetBufferDeviceAddress().
    // Only valid when RT/BDA features are enabled — otherwise the API call
    // would itself fail. Guard with RT_SUPPORTED to match buffer-creation
    // flags in this function.
    if (RT_SUPPORTED) {
        mesh.vertexAddress = getBufferDeviceAddress(device, mesh.vertexBuffer.buffer);
        mesh.indexAddress  = getBufferDeviceAddress(device, mesh.indexBuffer.buffer);
    }

    // Build per-mesh BLAS if RT is available. Falls through silently otherwise.
    if (RT_SUPPORTED) {
        buildMeshBlas(mesh, physicalDevice, device, commandPool, queue);
    }
}

void destroyMesh(VkDevice device, Mesh& mesh) {
    destroyRtMeshGeometry(device, mesh.rt);
    destroyBuffer(device, mesh.vertexBuffer);
    destroyBuffer(device, mesh.indexBuffer);
}
