#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void pollEvents();

    GLFWwindow* getHandle() const { return m_window; }
    VkExtent2D getFramebufferSize() const;
    std::vector<const char*> getRequiredExtensions() const;
    VkSurfaceKHR createSurface(VkInstance instance) const;

    bool isKeyDown(int key) const;
    void getCursorPos(double& x, double& y) const;
    void setCursorMode(int mode);

    bool  framebufferResized = false;
    float scrollDelta = 0.0f;

private:
    GLFWwindow* m_window = nullptr;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};
