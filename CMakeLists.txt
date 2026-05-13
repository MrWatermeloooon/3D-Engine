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
    src/engine/jobs.cpp
    src/engine/gpu_cull.cpp
    src/engine/hzb.cpp
    src/engine/raytracing.cpp
    src/engine/dlss.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
)

# ── DLSS (NVIDIA NGX) ─────────────────────────────────────────────
# The SDK is expected locally in DLSS_SDK/ relative to this CMakeLists. We link the
# release-flavour static import lib (nvsdk_ngx_d.lib) and copy the dev-signed
# DLL next to the exe — the dev DLL ignores app-allowlisting, perfect for
# in-development testing. Builds without DLSS_SDK/ on disk will fail at the
# include step; that's intentional, integration is hard-required.
set(DLSS_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DLSS_SDK)
# Pick the lib variant matching the active CRT. Release uses /MD +
# _ITERATOR_DEBUG_LEVEL=0; debug uses /MDd + IDL=2 — they must agree or the
# linker rejects the mismatch (LNK2038).
set(DLSS_LIB_RELEASE ${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
set(DLSS_LIB_DEBUG   ${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib)
set(DLSS_LIB         $<IF:$<CONFIG:Debug>,${DLSS_LIB_DEBUG},${DLSS_LIB_RELEASE}>)
set(DLSS_DLL         ${DLSS_SDK_DIR}/lib/Windows_x86_64/dev/nvngx_dlss.dll)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
    ${DLSS_SDK_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    EnTT::EnTT
    imgui::imgui
    imguizmo::imguizmo
    ${DLSS_LIB}
)

# Copy nvngx_dlss.dll next to the exe so NGX can find it at runtime.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLSS_DLL}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/nvngx_dlss.dll
    COMMENT "Copying nvngx_dlss.dll next to exe")

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
    ${SHADER_SOURCE_DIR}/cull.comp
    ${SHADER_SOURCE_DIR}/hzb_reduce.comp
)

set(SHADER_OUTPUTS "")

foreach(SHADER_FILE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    set(OUTPUT_FILE ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)

    # Target Vulkan 1.2 so ray queries (GL_EXT_ray_query) are accepted by glslc.
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 ${SHADER_FILE} -o ${OUTPUT_FILE}
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
