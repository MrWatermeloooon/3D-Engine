cmake_minimum_required(VERSION 3.20)
project(VulkanEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)

# ── Source files ──────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/engine/engine.cpp
    src/engine/window.cpp
    src/engine/vulkan_init.cpp
    src/engine/swapchain.cpp
    src/engine/pipeline.cpp
    src/engine/renderer.cpp
    src/engine/buffer.cpp
    src/engine/depth.cpp
    src/engine/descriptors.cpp
    src/engine/texture.cpp
    src/engine/mesh.cpp
    src/engine/camera.cpp
    src/engine/resource_manager.cpp
    src/engine/scene.cpp
    src/engine/debug_ui.cpp
    src/engine/lights.cpp
    src/engine/shadow.cpp
    src/engine/postfx.cpp
    src/engine/frustum.cpp
    src/engine/instancing.cpp
    src/engine/skeletal.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    EnTT::EnTT
    imgui::imgui
    imguizmo::imguizmo
)

# ── Shader compilation (GLSL → SPIR-V) ───────────────────────
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/mesh.vert
    ${SHADER_SOURCE_DIR}/mesh.frag
    ${SHADER_SOURCE_DIR}/shadow.vert
    ${SHADER_SOURCE_DIR}/fullscreen.vert
    ${SHADER_SOURCE_DIR}/composite.frag
    ${SHADER_SOURCE_DIR}/bloom_downsample.frag
    ${SHADER_SOURCE_DIR}/bloom_upsample.frag
    ${SHADER_SOURCE_DIR}/ssao.frag
    ${SHADER_SOURCE_DIR}/fxaa.frag
    ${SHADER_SOURCE_DIR}/mesh_skinned.vert
)

set(SHADER_OUTPUTS "")

foreach(SHADER_FILE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    set(OUTPUT_FILE ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${GLSLC_EXECUTABLE} ${SHADER_FILE} -o ${OUTPUT_FILE}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling shader: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
        VERBATIM
    )

    list(APPEND SHADER_OUTPUTS ${OUTPUT_FILE})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} Shaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_OUTPUT_DIR} $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
    COMMENT "Copying compiled shaders to output directory"
)
