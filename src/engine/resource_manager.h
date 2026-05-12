#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include "mesh.h"
#include "texture.h"
#include "descriptors.h"
#include "components.h"

struct MeshEntry {
    Mesh mesh;
    std::string path;
    std::filesystem::file_time_type lastWrite;
    bool isBuiltin = false;
};

struct TextureEntry {
    TextureData texture;
    uint32_t    bindlessSlot = 0;     // index into the global bindless array
    std::string path;
    std::filesystem::file_time_type lastWrite;
    bool isBuiltin = false;
};

class ResourceManager {
public:
    void init(VkPhysicalDevice physicalDevice, VkDevice device,
              VkCommandPool commandPool, VkQueue queue,
              DescriptorData& descriptors);

    MeshHandle getDefaultCube() const { return m_defaultCube; }
    TextureHandle getDefaultTexture() const { return m_defaultTexture; }

    MeshHandle loadMesh(const std::string& path);
    MeshHandle addMesh(const std::string& name, Mesh mesh);
    TextureHandle loadTexture(const std::string& path);

    const Mesh& getMesh(MeshHandle handle) const { return m_meshes[handle.id].mesh; }
    const TextureData& getTexture(TextureHandle handle) const { return m_textures[handle.id].texture; }
    uint32_t getBindlessSlot(TextureHandle handle) const { return m_textures[handle.id].bindlessSlot; }
    VkDescriptorSet bindlessSet() const { return m_descriptors->bindlessTexturesSet; }

    void pollHotReload();
    void cleanup();

private:
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    DescriptorData* m_descriptors = nullptr;

    std::vector<MeshEntry> m_meshes;
    std::vector<TextureEntry> m_textures;
    std::unordered_map<std::string, MeshHandle> m_meshLookup;
    std::unordered_map<std::string, TextureHandle> m_textureLookup;

    MeshHandle m_defaultCube{};
    TextureHandle m_defaultTexture{};
};
