#include "scene.h"
#include "resource_manager.h"

#include <glm/gtc/matrix_transform.hpp>
#include <string>

entt::entity Scene::createEntity(const std::string& name) {
    auto entity = m_registry.create();
    m_registry.emplace<NameComponent>(entity, name);
    m_registry.emplace<TransformComponent>(entity);
    return entity;
}

void Scene::destroyEntity(entt::entity entity) {
    m_registry.destroy(entity);
}

void Scene::createDefaultScene(ResourceManager& resources) {
    auto cube = resources.getDefaultCube();
    auto tex  = resources.getDefaultTexture();

    // Sun (directional light)
    {
        auto e = createEntity("Sun");
        DirectionalLightComponent dl{};
        dl.direction = glm::normalize(glm::vec3(-0.4f, -0.8f, -0.4f));
        dl.color     = glm::vec3(1.0f, 0.95f, 0.85f);
        dl.intensity = 3.5f;
        m_registry.emplace<DirectionalLightComponent>(e, dl);
    }

    // Floor (large, dielectric/rough)
    {
        auto e = createEntity("Floor");
        auto& t = m_registry.get<TransformComponent>(e);
        t.position = {0.0f, -0.5f, 0.0f};
        t.scale    = {15.0f, 0.1f, 15.0f};
        m_registry.emplace<MeshComponent>(e, cube);
        MaterialComponent mat{};
        mat.texture   = tex;
        mat.color     = glm::vec4(0.7f, 0.7f, 0.72f, 1.0f);
        mat.metallic  = 0.0f;
        mat.roughness = 0.85f;
        m_registry.emplace<MaterialComponent>(e, mat);

        // Static physics collider matching the floor box so dynamic bodies
        // land on it.
        RigidBodyComponent rb{};
        rb.motion = RigidBodyComponent::Motion::Static;
        rb.shape  = RigidBodyComponent::Shape::Box;
        rb.autoShapeFromScale = true;
        m_registry.emplace<RigidBodyComponent>(e, rb);
    }

    // Metallic spinning cube
    {
        auto e = createEntity("Steel Cube");
        auto& t = m_registry.get<TransformComponent>(e);
        t.position = {-2.0f, 0.5f, 0.0f};
        m_registry.emplace<MeshComponent>(e, cube);
        MaterialComponent mat{};
        mat.texture   = tex;
        mat.color     = glm::vec4(0.95f, 0.95f, 0.97f, 1.0f);
        mat.metallic  = 1.0f;
        mat.roughness = 0.25f;
        m_registry.emplace<MaterialComponent>(e, mat);
        m_registry.emplace<RotatorComponent>(e, glm::vec3(0, 1, 0), 30.0f);
    }

    // Plastic spinning cube
    {
        auto e = createEntity("Plastic Cube");
        auto& t = m_registry.get<TransformComponent>(e);
        t.position = {0.0f, 0.5f, 0.0f};
        m_registry.emplace<MeshComponent>(e, cube);
        MaterialComponent mat{};
        mat.texture   = tex;
        mat.color     = glm::vec4(0.9f, 0.2f, 0.2f, 1.0f);
        mat.metallic  = 0.0f;
        mat.roughness = 0.4f;
        m_registry.emplace<MaterialComponent>(e, mat);
        m_registry.emplace<RotatorComponent>(e, glm::vec3(1, 1, 0), 50.0f);
    }

    // Gold spinning cube
    {
        auto e = createEntity("Gold Cube");
        auto& t = m_registry.get<TransformComponent>(e);
        t.position = {2.0f, 0.5f, 0.0f};
        m_registry.emplace<MeshComponent>(e, cube);
        MaterialComponent mat{};
        mat.texture   = tex;
        mat.color     = glm::vec4(1.0f, 0.78f, 0.34f, 1.0f);
        mat.metallic  = 1.0f;
        mat.roughness = 0.2f;
        m_registry.emplace<MaterialComponent>(e, mat);
        m_registry.emplace<RotatorComponent>(e, glm::vec3(0, 0, 1), 70.0f);
    }

    // Physics demo: a tower of dynamic boxes that falls and settles on the
    // floor. Slight per-box X jitter so the stack topples instead of landing
    // as a perfect column — makes the simulation visibly obvious.
    {
        const glm::vec3 colors[5] = {
            {0.30f, 0.60f, 0.95f}, {0.95f, 0.55f, 0.25f}, {0.45f, 0.85f, 0.45f},
            {0.85f, 0.35f, 0.55f}, {0.80f, 0.80f, 0.30f},
        };
        for (int i = 0; i < 5; ++i) {
            auto e = createEntity("Physics Box " + std::to_string(i));
            auto& t = m_registry.get<TransformComponent>(e);
            t.position = {4.5f + (i % 2 == 0 ? 0.08f : -0.08f),
                          1.0f + 1.2f * static_cast<float>(i),
                          0.0f};
            t.scale    = {0.5f, 0.5f, 0.5f};
            m_registry.emplace<MeshComponent>(e, cube);
            MaterialComponent mat{};
            mat.texture   = tex;
            mat.color     = glm::vec4(colors[i], 1.0f);
            mat.metallic  = 0.0f;
            mat.roughness = 0.5f;
            m_registry.emplace<MaterialComponent>(e, mat);
            RigidBodyComponent rb{};
            rb.motion = RigidBodyComponent::Motion::Dynamic;
            rb.shape  = RigidBodyComponent::Shape::Box;
            rb.autoShapeFromScale = true;
            rb.mass   = 1.0f;
            m_registry.emplace<RigidBodyComponent>(e, rb);
        }
    }

    // A warm fill light (subtle — sun should still dominate)
    {
        auto e = createEntity("Warm Fill");
        auto& t = m_registry.get<TransformComponent>(e);
        t.position = {2.0f, 2.0f, 2.0f};
        PointLightComponent pl{};
        pl.color     = glm::vec3(1.0f, 0.55f, 0.30f);
        pl.intensity = 1.5f;
        pl.range     = 8.0f;
        m_registry.emplace<PointLightComponent>(e, pl);
    }
}

void Scene::update(float deltaTime) {
    auto view = m_registry.view<TransformComponent, RotatorComponent>();
    for (auto entity : view) {
        auto& transform = view.get<TransformComponent>(entity);
        auto& rotator   = view.get<RotatorComponent>(entity);
        transform.rotation += rotator.axis * rotator.speed * deltaTime;
    }
}
