#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult err = x;                                                      \
        if (err != VK_SUCCESS) {                                               \
            throw std::runtime_error(                                          \
                std::string("Vulkan error ") + std::to_string(static_cast<int>(err)) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                      \
    } while (0)
