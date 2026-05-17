#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class CameraMode { FPS, Orbit };

class Camera {
public:
    Camera();

    void setMode(CameraMode mode);
    CameraMode getMode() const { return m_mode; }
    void toggleMode();

    void processKeyboard(float dt, bool fwd, bool back, bool left, bool right,
                         bool up, bool down);
    void processMouseMovement(float dx, float dy);
    void processOrbit(float dx, float dy);
    void processPan(float dx, float dy);   // moves the orbit target along screen axes
    void processZoom(float delta);

    void setAspectRatio(float aspect);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;          // Vulkan-flipped (for rendering)
    glm::mat4 getProjectionMatrixUnflipped() const; // Standard (for ImGuizmo)
    glm::vec3 getPosition() const;
    glm::vec3 getForward()  const;   // normalized world-space view direction

    float nearClip() const { return m_near; }
    float farClip()  const { return m_far;  }

private:
    void updateVectors();

    CameraMode m_mode = CameraMode::Orbit;

    // FPS
    glm::vec3 m_position{ 0.0f, 1.0f, 3.0f };
    glm::vec3 m_front{ 0.0f, 0.0f, -1.0f };
    glm::vec3 m_up{ 0.0f, 1.0f, 0.0f };
    glm::vec3 m_right{ 1.0f, 0.0f, 0.0f };
    float m_yaw   = -90.0f;
    float m_pitch = 0.0f;
    float m_speed = 2.5f;
    float m_sensitivity = 0.1f;

    // Orbit
    glm::vec3 m_target{ 0.0f, 0.0f, 0.0f };
    float m_orbitDistance = 3.0f;
    float m_orbitYaw   = 45.0f;
    float m_orbitPitch = 30.0f;

    // Projection
    float m_fov    = 45.0f;
    float m_aspect = 16.0f / 9.0f;
    float m_near   = 0.1f;
    float m_far    = 100.0f;
};
