#pragma once

#include <entt/entt.hpp>
#include <string>
#include "components.h"

class ResourceManager;

class Scene {
public:
    entt::entity createEntity(const std::string& name = "Entity");
    void destroyEntity(entt::entity entity);

    entt::registry& registry() { return m_registry; }
    const entt::registry& registry() const { return m_registry; }

    void createDefaultScene(ResourceManager& resources);
    void update(float deltaTime);

private:
    entt::registry m_registry;
};
