#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <functional>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription desc{};
        desc.binding = 0;
        desc.stride = sizeof(Vertex);
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attrs{};

        attrs[0].binding  = 0;
        attrs[0].location = 0;
        attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset   = offsetof(Vertex, position);

        attrs[1].binding  = 0;
        attrs[1].location = 1;
        attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset   = offsetof(Vertex, normal);

        attrs[2].binding  = 0;
        attrs[2].location = 2;
        attrs[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attrs[2].offset   = offsetof(Vertex, texCoord);

        attrs[3].binding  = 0;
        attrs[3].location = 3;
        attrs[3].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[3].offset   = offsetof(Vertex, color);

        return attrs;
    }

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal &&
               texCoord == other.texCoord && color == other.color;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& v) const {
            size_t h = 0;
            auto combine = [&h](auto val) {
                h ^= std::hash<decltype(val)>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            combine(v.position.x); combine(v.position.y); combine(v.position.z);
            combine(v.normal.x);   combine(v.normal.y);   combine(v.normal.z);
            combine(v.texCoord.x); combine(v.texCoord.y);
            combine(v.color.x);    combine(v.color.y);    combine(v.color.z);
            return h;
        }
    };
}
