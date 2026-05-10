# 3D Engine

A C++20 Vulkan rendering engine focused on real-time 3D graphics, modular engine systems, and modern rendering features. The project is built with CMake and vcpkg, uses GLFW for windowing, GLM for math, EnTT for scene data, and ImGui/ImGuizmo for runtime debugging and editor-style controls.

## Current Features

- Vulkan renderer with swapchain, command buffers, descriptor management, and pipeline setup.
- Mesh, texture, buffer, camera, scene, and resource manager systems.
- Entity/component scene structure using EnTT.
- Directional lighting support with cascaded shadow maps.
- HDR offscreen rendering target.
- Bloom downsample/upsample pass.
- SSAO pass with configurable kernel and debug views.
- Composite post-processing with exposure, gamma, saturation, contrast, vignette, and tone mapping controls.
- ImGui debug UI and ImGuizmo integration.
- GLSL shader compilation to SPIR-V through CMake.

## Tech Stack

- C++20
- Vulkan
- CMake 3.20+
- Ninja
- vcpkg
- GLFW
- GLM
- tinyobjloader
- stb
- EnTT
- ImGui
- ImGuizmo

## Project Structure

```text
.
|-- shaders/        # GLSL shader sources
|-- src/
|   |-- engine/     # Engine modules and rendering systems
|   `-- utils/      # Shared utility headers
|-- CMakeLists.txt
|-- CMakePresets.json
`-- vcpkg.json
```

## Requirements

- Vulkan SDK installed and available on PATH.
- CMake 3.20 or newer.
- Ninja build system.
- vcpkg installed with `VCPKG_ROOT` set.
- A GPU and driver with Vulkan support.

## Build

Configure the debug preset:

```powershell
cmake --preset default
```

Build the debug target:

```powershell
cmake --build --preset debug
```

Configure and build release:

```powershell
cmake --preset release
cmake --build --preset release
```

Compiled shaders are generated into the build output and copied beside the executable after build.

## Run

After building, run the generated `VulkanEngine` executable from the build output directory.

## Notes

The repository intentionally ignores local build output, Visual Studio metadata, compiled SPIR-V files, and the Vulkan SDK installer. Install the Vulkan SDK locally instead of committing the installer into Git.

## Future Additions

### Core Rendering

- Skeletal animation: bone transforms, skinning shader, glTF skeleton support.
- Frustum culling: AABB tests and skipping invisible draw calls.
- Instanced rendering: draw 10k trees with one call.
- Anti-aliasing: MSAA, TAA, or FXAA depending on the quality/performance tradeoff.
- Normal mapping: tangent-space normals with optional parallax.
- Skybox + IBL: HDR environment map, irradiance, and specular probe.

### RTX / Ray Tracing

- Ray traced shadows: replace shadow maps with accurate ray traced soft shadows.
- Ray traced reflections: mirror and glossy reflections without screen-space limits.
- Global illumination: one-bounce indirect light bouncing through the scene.
- DLSS: AI upscaling for rendering at a lower internal resolution while outputting full-resolution frames.

### Visual Quality

- Screen space reflections: cheap and convincing reflections using visible screen data.
- Volumetric lighting: god rays, light shafts, and atmospheric fog.
- Depth of field: camera-style near/far bokeh blur.
- Motion blur: per-object or camera velocity-based blur.
- Particle / VFX system: GPU particles for fire, smoke, sparks, and explosions.
- Decals: projected textures for bullet holes, paint, dirt, and surface details.

### Engine Completeness

- Physics (Jolt): rigid bodies, collision, and raycasting via Jolt Physics.
- Audio: 3D spatial audio via OpenAL or miniaudio.
- Scripting (Lua): sol2 bindings and engine API exposure to scripts.
- Scene serialization: save and load scenes from JSON or a binary format.
- Prefab system: reusable object templates that can be instantiated and overridden.
- Material editor: visual node graph for building materials without editing shaders directly.
- Undo / redo: command pattern support for editor workflows.
- Asset browser: file explorer panel for dragging meshes, textures, and audio into scenes.
- Hot shader reload: edit GLSL files and see changes without restarting the engine.

### Performance

- GPU-driven rendering: indirect draw calls where the GPU decides what to draw.
- Bindless rendering: large descriptor arrays to reduce per-draw descriptor binding overhead.
- Job system: multi-threaded task scheduler for spreading engine work across CPU cores.
- LOD system: swap to lower-poly meshes at distance to keep far objects cheap.
- Occlusion culling: skip objects hidden behind other geometry.
- Variable rate shading: shade detailed edges at full rate and flat areas more cheaply.
