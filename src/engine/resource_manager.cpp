#include "resource_manager.h"

#include <iostream>

void ResourceManager::init(VkPhysicalDevice physicalDevice, VkDevice device,
                           VkCommandPool commandPool, VkQueue queue,
                           DescriptorData& descriptors) {
    m_physicalDevice = physicalDevice;
    m_device = device;
    m_commandPool = commandPool;
    m_queue = queue;
    m_descriptors = &descriptors;

    // Default cube mesh
    auto cube = createCubeMesh();
    uploadMesh(cube, m_physicalDevice, m_device, m_commandPool, m_queue);

    MeshEntry cubeEntry;
    cubeEntry.mesh = std::move(cube);
    cubeEntry.path = "__builtin_cube";
    cubeEntry.isBuiltin = true;

    m_defaultCube = { static_cast<uint32_t>(m_meshes.size()) };
    m_meshes.push_back(std::move(cubeEntry));

    // Default checkerboard texture
    auto tex = createDefaultTexture(m_physicalDevice, m_device, m_commandPool, m_queue);
    auto descSet = allocateMaterialDescriptorSet(descriptors, m_device, tex.imageView, tex.sampler);

    TextureEntry texEntry;
    texEntry.texture = tex;
    texEntry.descriptorSet = descSet;
    texEntry.path = "__builtin_checkerboard";
    texEntry.isBuiltin = true;

    m_defaultTexture = { static_cast<uint32_t>(m_textures.size()) };
    m_textures.push_back(std::move(texEntry));

    std::cout << "[ResourceManager] Initialized with default assets\n";
}

MeshHandle ResourceManager::loadMesh(const std::string& path) {
    auto it = m_meshLookup.find(path);
    if (it != m_meshLookup.end()) return it->second;

    auto mesh = loadMeshFromObj(path);
    uploadMesh(mesh, m_physicalDevice, m_device, m_commandPool, m_queue);

    MeshEntry entry;
    entry.mesh = std::move(mesh);
    entry.path = path;
    entry.lastWrite = std::filesystem::last_write_time(path);

    MeshHandle handle = { static_cast<uint32_t>(m_meshes.size()) };
    m_meshes.push_back(std::move(entry));
    m_meshLookup[path] = handle;
    return handle;
}

MeshHandle ResourceManager::addMesh(const std::string& name, Mesh mesh) {
    uploadMesh(mesh, m_physicalDevice, m_device, m_commandPool, m_queue);

    MeshEntry entry;
    entry.mesh = std::move(mesh);
    entry.path = name;
    entry.isBuiltin = true;

    MeshHandle handle = { static_cast<uint32_t>(m_meshes.size()) };
    m_meshes.push_back(std::move(entry));
    return handle;
}

TextureHandle ResourceManager::loadTexture(const std::string& path) {
    auto it = m_textureLookup.find(path);
    if (it != m_textureLookup.end()) return it->second;

    auto tex = createTextureFromFile(m_physicalDevice, m_device, m_commandPool, m_queue, path);
    auto descSet = allocateMaterialDescriptorSet(*m_descriptors, m_device, tex.imageView, tex.sampler);

    TextureEntry entry;
    entry.texture = tex;
    entry.descriptorSet = descSet;
    entry.path = path;
    entry.lastWrite = std::filesystem::last_write_time(path);

    TextureHandle handle = { static_cast<uint32_t>(m_textures.size()) };
    m_textures.push_back(std::move(entry));
    m_textureLookup[path] = handle;
    return handle;
}

void ResourceManager::pollHotReload() {
    for (auto& entry : m_meshes) {
        if (entry.isBuiltin) continue;
        try {
            auto currentTime = std::filesystem::last_write_time(entry.path);
            if (currentTime != entry.lastWrite) {
                std::cout << "[ResourceManager] Hot-reloading mesh: " << entry.path << "\n";
                vkDeviceWaitIdle(m_device);
                destroyMesh(m_device, entry.mesh);
                entry.mesh = loadMeshFromObj(entry.path);
                uploadMesh(entry.mesh, m_physicalDevice, m_device, m_commandPool, m_queue);
                entry.lastWrite = currentTime;
            }
        } catch (...) {}
    }

    for (auto& entry : m_textures) {
        if (entry.isBuiltin) continue;
        try {
            auto currentTime = std::filesystem::last_write_time(entry.path);
            if (currentTime != entry.lastWrite) {
                std::cout << "[ResourceManager] Hot-reloading texture: " << entry.path << "\n";
                vkDeviceWaitIdle(m_device);

                vkFreeDescriptorSets(m_device, m_descriptors->descriptorPool, 1, &entry.descriptorSet);
                destroyTexture(m_device, entry.texture);

                entry.texture = createTextureFromFile(m_physicalDevice, m_device,
                                                      m_commandPool, m_queue, entry.path);
                entry.descriptorSet = allocateMaterialDescriptorSet(
                    *m_descriptors, m_device, entry.texture.imageView, entry.texture.sampler);
                entry.lastWrite = currentTime;
            }
        } catch (...) {}
    }
}

void ResourceManager::cleanup() {
    for (auto& entry : m_meshes)
        destroyMesh(m_device, entry.mesh);
    m_meshes.clear();

    for (auto& entry : m_textures)
        destroyTexture(m_device, entry.texture);
    m_textures.clear();

    m_meshLookup.clear();
    m_textureLookup.clear();
}
