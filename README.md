# 3D Engine

A C++20 Vulkan rendering engine focused on real-time 3D graphics, modular engine systems, and modern rendering features. The project is built with CMake and vcpkg, uses GLFW for windowing, GLM for math, EnTT for scene data, and ImGui/ImGuizmo for runtime debugging and editor-style controls.

> **Work in progress:** this engine is actively being developed. Some systems are experimental, APIs may change, and features listed in the roadmap are planned or partially implemented rather than production-ready.

## Current Features

- Vulkan renderer with swapchain, command buffers, descriptor management, and pipeline setup.
- Mesh, texture, buffer, camera, scene, and resource manager systems.
- Mesh-local AABB generation and frustum culling against extracted view-projection planes.
- Instanced rendering path with per-instance transform, color, metallic, and roughness data.
- Indirect batched drawing with per-frame `VkDrawIndexedIndirectCommand` buffers.
- GPU compute culling that writes visible draw commands directly into the indirect buffer.
- HZB occlusion culling with a compute-built hierarchical depth buffer from the previous frame.
- Mesh LOD groups with compute-side LOD selection and per-LOD indirect draw slots.
- Bindless texture array with descriptor indexing and update-after-bind support.
- Hardware ray tracing path with BLAS/TLAS builds, Vulkan ray queries, ray-traced shadows, reflections, and one-bounce GI.
- Partial DLSS integration with NGX initialization, Halton jitter, motion-vector attachment, and DLSS UI controls.
- Job system with fire-and-forget tasks, blocking `parallel_for`, and frame-work barriers.
- Variable-rate shading support through `VK_KHR_fragment_shading_rate`, with per-LOD rates and UI override when available.
- Entity/component scene structure using EnTT.
- Directional lighting support with cascaded shadow maps.
- HDR offscreen rendering target.
- Bloom downsample/upsample pass.
- SSAO pass with configurable kernel and debug views.
- Composite post-processing with exposure, gamma, saturation, contrast, vignette, tone mapping, and FXAA controls.
- Skeletal animation infrastructure with skeletons, animation channels, skinned vertex data, bone palette uploads, and a test bone-chain demo.
- ImGui debug UI and ImGuizmo integration.
- GLSL shader compilation to SPIR-V through CMake.

## Recent Optimization Work

- Swapchain SSAO recreation now uses the shared `destroySSAO` and `createSSAO` path instead of a duplicated inline rebuild block.
- Single-time command submission now waits on a per-call fence instead of stalling the whole graphics queue with `vkQueueWaitIdle`.
- Instance data now carries a precomputed normal matrix so the vertex shader no longer computes `transpose(inverse(mat3(model)))` per vertex.
- Frustum culling, world AABB generation, model matrix generation, and normal matrix generation are parallelized for larger entity counts.
- Main and shadow batches now issue indirect indexed draws, setting up the renderer for future compute-driven visibility.
- Bindless textures are bound once per pass, with each instance selecting its texture by bindless index.
- CPU-side visible sorting has been replaced by GPU culling: the CPU writes candidate instances and batch headers, then compute fills visible main-pass instances and unconditional shadow instances.
- The renderer now builds a hierarchical Z buffer after the main pass and uses it for next-frame occlusion culling.
- LOD spheres and per-LOD indirect commands exercise distance-based mesh selection without changing the graphics draw path.
- Vulkan depth convention fixes now define `GLM_FORCE_DEPTH_ZERO_TO_ONE` and `GLM_FORCE_RADIANS`, correcting camera projection, cascade extraction, and shadow ortho math.
- Shadow cascade stability improved by pulling the light origin back to avoid clipping tall off-frustum casters.
- HZB now builds at half offscreen resolution to match the reducer shader and avoid screen-position-dependent false occlusion.
- Ray-traced shadows can replace the CSM shadow pass, with the shadow image still transitioned so descriptors remain valid.
- RT reflections and GI reuse mesh device addresses and material data to shade ray hits with interpolated normals and sun visibility.

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
- NVIDIA DLSS SDK cloned locally to `DLSS_SDK/` for DLSS-enabled builds. The SDK is intentionally ignored by Git because it contains large third-party binaries.

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

The local `DLSS_SDK/` checkout is also ignored. Clone or install NVIDIA's DLSS SDK into that folder before configuring if you want to build the DLSS integration.

If a Windows build fails with `LNK1168`, another program may be holding the executable open. Close capture/overlay tools such as Medal, OBS, or Overwolf, then rebuild.

## Future Additions

### Core Rendering

- glTF skeletal animation importer: load skins, joints, inverse-bind matrices, vertex weights, and animation channels from glTF files.
- Skeletal animation UI polish: expose animator time, speed, clip selection, and playback controls in the properties panel.
- Normal mapping: tangent-space normals with optional parallax.
- Skybox + IBL: HDR environment map, irradiance, and specular probe.

### RTX / Ray Tracing

- DLSS evaluation path: call `NVSDK_NGX_VULKAN_CreateFeature_DLSS` and `NVSDK_NGX_VULKAN_EvaluateFeature`.
- Render-scale plumbing: render at the DLSS-selected internal resolution and upscale to full output.
- Previous per-instance transforms for true object motion vectors instead of camera-only motion.
- Denoising and temporal accumulation for cleaner ray-traced GI and glossy reflections.

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

- Compute-authored per-cascade shadow visibility so shadow passes can cull per cascade instead of drawing every caster.
- Attachment-based variable-rate shading: generate a shading-rate image from scene detail or edge detection and migrate the offscreen pass to `VkRenderPass2`.
- HZB stabilization for fast camera movement to reduce brief false-culls on newly revealed geometry.

## Hard Addons

These are larger, more advanced systems that would push the engine toward professional game engine territory. Many of them are multi-month or multi-year projects on their own.

### Simulation

Nature, destruction, and complex geometry systems.

- Atmospheric scattering: physically accurate sky rendering for sunrise, sunset, and blue-sky lighting generated from math.
- Ocean / water: FFT wave simulation for realistic open water with foam and waves.
- Cloth simulation: position-based dynamics for capes, flags, and fabric that reacts to physics.
- Hair / fur: strand-based rendering for thousands of individual simulated strands.
- Subsurface scattering: light passing through skin, wax, marble, and other organic or translucent surfaces.
- Terrain system: heightmaps, tessellation, clipmap LOD, and splat-map texturing.
- Foliage system: grass, trees, wind animation, GPU instancing, and LOD.
- Destruction: real-time mesh fracture with Voronoi shatter and debris simulation.

### AI / Animation

Systems that make the world feel alive.

- NavMesh pathfinding: Recast/Detour integration so AI agents can navigate the world.
- Behavior trees: AI decision making for patrol, chase, attack, flee, and other logic.
- Animation state machine: blend trees, transition conditions, and layered animations.
- Inverse kinematics: FABRIK IK for feet planting on uneven ground and hands reaching for objects.
- Ragdoll physics: blend from animation to physics for fully dynamic bodies.

### Tools

Developer tools that make the engine faster and more professional to build with.

- Flame graph profiler: visual CPU timeline for finding frame-time bottlenecks.
- GPU timeline profiler: timestamp queries for profiling every render pass on the GPU.
- Dev console: in-engine command line for tweaking CVars, spawning objects, and running scripts.
- Cinematic sequencer: timeline editor for keyframing cameras, lights, and transforms.
- Build / packaging: standalone executable shipping with asset packing and editor-code stripping.

### Platform

Work that takes the engine beyond a single Windows desktop build.

- Linux support: CMake-based build and path fixes for cross-platform support.
- VR / OpenXR: stereo rendering, head tracking, and controller input through the OpenXR standard.
- Networking: ENet or custom UDP with authoritative server logic and client prediction.
- Plugin system: runtime DLL loading so the engine can be extended without recompiling.

### Moonshots

Unreal-level systems that are years of work each, but incredible long-term goals.

- Virtual geometry: Nanite-style streaming and rendering of massive triangle counts through mesh shaders.
- Dynamic GI: Lumen-style fully dynamic indirect lighting with no baked lightmaps.
- Full path tracer: reference-quality rendering where light transport is physically simulated.
- Virtual texturing: stream texture tiles on demand for huge texture detail with controlled memory use.
