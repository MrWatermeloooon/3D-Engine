#include "camera.h"
#include <algorithm>
#include <cmath>

Camera::Camera() {
    updateVectors();
}

void Camera::setMode(CameraMode mode) {
    if (m_mode == mode) return;

    if (mode == CameraMode::Orbit) {
        m_target = m_position + m_front * m_orbitDistance;
    } else {
        m_position = getPosition();
        glm::vec3 dir = glm::normalize(m_target - m_position);
        m_pitch = glm::degrees(asin(dir.y));
        m_yaw   = glm::degrees(atan2(dir.z, dir.x));
    }

    m_mode = mode;
    updateVectors();
}

void Camera::toggleMode() {
    setMode(m_mode == CameraMode::FPS ? CameraMode::Orbit : CameraMode::FPS);
}

void Camera::processKeyboard(float dt, bool fwd, bool back, bool left, bool right,
                              bool up, bool down) {
    float velocity = m_speed * dt;
    if (fwd)   m_position += m_front * velocity;
    if (back)  m_position -= m_front * velocity;
    if (left)  m_position -= m_right * velocity;
    if (right) m_position += m_right * velocity;
    if (up)    m_position.y += velocity;
    if (down)  m_position.y -= velocity;
}

void Camera::processMouseMovement(float dx, float dy) {
    m_yaw   += dx * m_sensitivity;
    m_pitch += dy * m_sensitivity;
    m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
    updateVectors();
}

void Camera::processOrbit(float dx, float dy) {
    m_orbitYaw   += dx * 0.3f;
    m_orbitPitch -= dy * 0.3f;
    m_orbitPitch = std::clamp(m_orbitPitch, -89.0f, 89.0f);
}

void Camera::processPan(float dx, float dy) {
    // Translate the orbit target along the camera's right + up vectors.
    // Pan speed scales with orbit distance so it feels consistent at any zoom.
    glm::vec3 pos     = getPosition();
    glm::vec3 forward = glm::normalize(m_target - pos);
    glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 up      = glm::normalize(glm::cross(right, forward));

    float scale = m_orbitDistance * 0.0015f;
    m_target += -right * dx * scale + up * dy * scale;
}

void Camera::processZoom(float delta) {
    if (m_mode == CameraMode::Orbit) {
        m_orbitDistance += delta * 0.3f;
        m_orbitDistance = std::clamp(m_orbitDistance, 0.5f, 50.0f);
    } else {
        m_fov += delta;
        m_fov = std::clamp(m_fov, 1.0f, 120.0f);
    }
}

void Camera::setAspectRatio(float aspect) {
    m_aspect = aspect;
}

glm::mat4 Camera::getViewMatrix() const {
    if (m_mode == CameraMode::FPS) {
        return glm::lookAt(m_position, m_position + m_front, m_up);
    }
    return glm::lookAt(getPosition(), m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::getProjectionMatrix() const {
    auto proj = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
    proj[1][1] *= -1; // Vulkan Y-flip
    return proj;
}

glm::mat4 Camera::getProjectionMatrixUnflipped() const {
    return glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
}

glm::vec3 Camera::getPosition() const {
    if (m_mode == CameraMode::FPS) {
        return m_position;
    }

    float pitchRad = glm::radians(m_orbitPitch);
    float yawRad   = glm::radians(m_orbitYaw);

    glm::vec3 offset;
    offset.x = m_orbitDistance * cos(pitchRad) * cos(yawRad);
    offset.y = m_orbitDistance * sin(pitchRad);
    offset.z = m_orbitDistance * cos(pitchRad) * sin(yawRad);

    return m_target + offset;
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);
    m_right = glm::normalize(glm::cross(m_front, glm::vec3(0.0f, 1.0f, 0.0f)));
    m_up    = glm::normalize(glm::cross(m_right, m_front));
}
