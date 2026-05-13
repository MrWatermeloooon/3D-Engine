# VulkanEngine — Handoff

Custom Vulkan 1.3 renderer for Windows, built incrementally across many sessions.
This document is a complete take-over reference so a new contributor (or new chat) can
pick up without reading the prior conversation.

## Build & Run

**Hard prerequisites** (already installed on the target machine):
- Visual Studio 2022 (MSVC 14.42+)
- CMake ≥ 3.20
- Ninja (bundled with VS)
- vcpkg at `C:\vcpkg`
- Vulkan SDK 1.4.341.1 at `C:\VulkanSDK\1.4.341.1` (for `glslc`)
- **NVIDIA DLSS SDK at `DLSS_SDK/`** (in the project root). Cloned from
  https://github.com/NVIDIA/DLSS. Build will hard-fail at link step without it.
  The `dev/nvngx_dlss.dll` is copied next to the exe post-build.
- GPU driver: NVIDIA 555+ for DLSS 310.x SDK. Tested on driver 591.86 / RTX 5070 Ti.

**Configure + build** (from a VS dev shell or via the helper batch pattern the
prior chats used — `vcvars64.bat` first, then):

```
set VCPKG_ROOT=C:\vcpkg
set PATH=C:\VulkanSDK\1.4.341.1\Bin;%PATH%
cd /d "P:\coding projects\3d-engine\3d engine"
cmake --preset default
cmake --build build
```

**Run**: `build/VulkanEngine.exe`. Working directory must be `build/` so the
relative `shaders/*.spv` and `nvngx_dlss.dll` paths resolve.

> ⚠️ **LNK1168 — locked exe.** Medal recorder, OBS hook, and Overwolf overlay
> all install Vulkan layers that sometimes hold a handle on the exe after the
> engine exits. If `cmake --build` fails with `LNK1168: cannot open
> VulkanEngine.exe for writing`, close any running engine window and end
> stragglers via Task Manager. Reboot if needed.

> ⚠️ The old multi-config Visual Studio build at `build/Debug/VulkanEngine.exe`
> is stale. Always run `build/VulkanEngine.exe` (Ninja single-config output).

## Dependencies (vcpkg manifest)

`vcpkg.json` declares: `glfw3`, `glm`, `stb`, `tinyobjloader`, `entt`,
`imgui` (features: `glfw-binding`, `vulkan-binding`, `docking-experimental`),
`imguizmo`. All header-only or thin libs.

**Not via vcpkg**:
- NVIDIA NGX SDK lives at `DLSS_SDK/` (manually cloned). Linked from
  `DLSS_SDK/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib` (Debug) /
  `nvsdk_ngx_d.lib` (Release). CMake picks per-config via generator
  expression — mismatching CRT iterator level causes LNK2038.

## Compile-time defines

In `CMakeLists.txt`:
- `GLM_FORCE_DEPTH_ZERO_TO_ONE` — makes `glm::perspective` and `glm::ortho`
  emit Vulkan-style `[0, 1]` NDC z. **Critical** — without this every
  projection in the engine is wrong (half the depth range gets clipped, CSM
  cascades land on the wrong slab of view-space, frustum extraction returns
  the wrong near plane, etc).
- `GLM_FORCE_RADIANS` — `glm::perspective` expects radians, set explicitly
  to silence GLM deprecation.

Shaders compile at `--target-env=vulkan1.2` (for ray-query + buffer-reference
support).

## Project Layout

```
DLSS_SDK/                      # NVIDIA NGX SDK (cloned, NOT vcpkg)
src/
  main.cpp                    # entry, owns one Engine, calls init/run/cleanup
  utils/vk_check.h            # VK_CHECK macro
  engine/
    engine.{h,cpp}            # game loop + owns all GPU state (god class — see below)
    window.{h,cpp}            # GLFW wrapper, input, scroll callback
    vulkan_init.{h,cpp}       # instance, debug, device, queue families, ext/feature probes
    swapchain.{h,cpp}         # swapchain + color-only render pass + sync objects
    pipeline.{h,cpp}          # graphics, shadow, and skinned pipelines
    buffer.{h,cpp}            # buffer/image create/destroy, single-time cmds
    descriptors.{h,cpp}       # scene UBO + bindless material set + TLAS write helper
    mesh.{h,cpp}              # static mesh, cube builder, OBJ loader, BLAS build hook
    texture.{h,cpp}           # stb_image loader, mipmaps, default white texture
    vertex.h                  # Vertex (position/normal/uv/color)
    camera.{h,cpp}            # FPS + Orbit camera
    components.h              # ECS components (Transform, Mesh, Material, lights, etc.)
    scene.{h,cpp}             # entt::registry wrapper + default scene
    resource_manager.{h,cpp}  # handle-based loading, hot reload, bindless registration
    debug_ui.{h,cpp}          # ImGui dockspace + panels + ImGuizmo
    renderer.{h,cpp}          # per-frame drawFrame
    lights.{h,cpp}            # light buffer (UBO), 3 light types
    shadow.{h,cpp}            # CSM (4 cascades), stable texel-snapped; initial layout transition
    postfx.{h,cpp}            # HDR offscreen (+motion vectors), bloom, SSAO, composite, LDR, FXAA
    depth.{h,cpp}             # depth resources + format selection
    frustum.{h,cpp}            # 6-plane extraction + AABB test (Vulkan ZTO convention)
    instancing.{h,cpp}        # InstanceData + Candidate/BatchHeader/Indirect buffers
    skeletal.{h,cpp}          # Skeleton, Animation, SkinnedMesh, test bone chain
    jobs.{h,cpp}              # thread-pool with parallel_for + wait_all
    gpu_cull.{h,cpp}          # compute-cull pipeline, params UBO, dispatch helper
    hzb.{h,cpp}               # hierarchical Z-buffer (occlusion culling source)
    raytracing.{h,cpp}        # BLAS/TLAS build, RtSettings, RtInstanceMaterial
    dlss.{h,cpp}              # NGX init/shutdown, Halton jitter, DlssSettings

shaders/
  fullscreen.vert             # full-screen triangle from gl_VertexIndex
  mesh.vert / mesh.frag       # PBR + RT shadows + RT reflections + RT GI + motion vectors
  mesh_skinned.vert           # LBS skinning, shares mesh.frag
  shadow.vert                 # depth-only with per-cascade lightVP push constant
  bloom_downsample.frag       # Karis 13-tap weighted average
  bloom_upsample.frag         # 3×3 tent, additive blend
  ssao.frag                   # 16-sample hemisphere SSAO, depth-derivative normals
  composite.frag              # tonemap + grading + vignette + debug views
  fxaa.frag                   # FXAA 3.11 lite
  cull.comp                   # GPU culling: frustum + HZB-occlusion + LOD pick
  hzb_reduce.comp             # 2×2 max-depth reduction (one dispatch per mip)
```

## Render Pipeline

Per-frame ordering inside `renderer.cpp::drawFrame`:

1. **Build candidates** (CPU, parallelized via JobSystem if >256 entities).
2. **Build TLAS** if RT is enabled — walks `registry.view<Transform,Mesh,Material>()`,
   gathers per-entity transform + BLAS device address + material data, calls
   `buildTlas` which uploads instance records + materials and dispatches a
   GPU acceleration-structure build. Followed by `writeSceneTlas` and
   `writeSceneRtMaterials` to bind the freshly built AS into the per-frame
   scene descriptor set.
3. **Compute cull dispatch** — `cull.comp`, one thread per candidate.
4. **Shadow pass (CSM)** — **skipped when RT shadows are active**. When run,
   4 cascades, depth-only, one `vkCmdDrawIndexedIndirect` per batch.
5. **Main pass** (offscreen HDR RGBA16F **+ motion RG16F** + sampleable depth).
   `mesh.frag` does direct lighting (Cook-Torrance), shadow lookup (CSM PCF
   *or* `rayQueryEXT` ray-trace per light), indirect-specular reflections
   (RT, metallics only), indirect-diffuse GI (RT, mixed with hemisphere
   ambient), and writes a screen-space motion vector to attachment 1.
6. **HZB reduction** — at half offscreen resolution, see HZB note below.
7. **SSAO pass** — auto-disabled when RT shadows are on (would double-stack
   contact darkening).
8. **Bloom** — 4-mip downsample chain then upsample chain.
9. **Composite pass** — HDR → LDR.
10. **FXAA + ImGui pass**.

Swapchain image format is `B8G8R8A8_SRGB`.

## Features

### PBR (Cook-Torrance, energy-conserving split)

- GGX normal distribution, Smith with Schlick-GGX geometry, Schlick Fresnel
- Metallic + roughness per material via `InstanceData.matParams`
- **Indirect lighting split** (in `mesh.frag::main`):
  - `kD_indirect = (1 - F_indirect) * (1 - metallic)` → diffuse weight
  - **Diffuse indirect**: hemisphere ambient (default) or RT GI (when on),
    mixed by `rtParams3.w`. Metallics get zero diffuse indirect.
  - **Specular indirect**: RT reflection (metallics only). Skipped for
    `metallic < 0.5` or `roughness >= 0.85`. Multiplied by `F_indirect *
    rtParams2.w`. Falls back to flat `F * sky` when RT reflections are off
    so metals don't go black.
- Linear HDR output; tonemap happens in composite

### Lights

- `DirectionalLightComponent`, `PointLightComponent`, `SpotLightComponent`
- Packed into a `LightsUBO` (32 max), 64 bytes each
- **CSM path**: only the first directional light is shadowed
- **RT path**: every light is shadowed by a per-light ray query, unless
  `RtSettings::sunOnly` is on (default true) which restricts to the first
  directional light for CSM parity

### Shadows

Two paths gated by `RtSettings::shadows`:

- **CSM (legacy)**: 4 cascades, 2048×2048, stable texel-snapped sphere bounds,
  practical split scheme, 5×5 PCF with normal-offset bias. **Shadow ortho
  pulls the light origin back by `radius * 10` along `-ldir`** so casters
  above the receiving frustum (tall pillars, terrain features) aren't clipped
  out — this was a bug fix earlier in this chat.
- **RT (default)**: per-fragment ray query toward each light. Soft shadows
  via cone-sampling (`rtSettings.shadowSoftness` = light angular radius),
  `rtSettings.shadowSamples` rays per light per fragment (default 16).
  Sun-only toggle for CSM parity.

Shadow image is **transitioned to `DEPTH_STENCIL_READ_ONLY_OPTIMAL` once at
creation** in `shadow.cpp::createShadowResources` so the descriptor binding
is valid even when the shadow pass never runs (RT-shadows-only case).

### Ray Tracing (Phases 1–3)

**Phase 1 — RT Shadows.**
- `vulkan_init.cpp::probeRtSupport` checks `VK_KHR_acceleration_structure`,
  `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations`. If all present and
  the features report OK, `RT_SUPPORTED` is set, extensions enabled, and 7
  function pointers loaded (`RT_CreateAS`, `RT_DestroyAS`,
  `RT_GetASBuildSizes`, `RT_GetASDeviceAddress`, `RT_CmdBuildAS`,
  `RT_CmdWriteASProps`, `RT_GetBufferDeviceAddress`).
- `bufferDeviceAddress` and `scalarBlockLayout` features enabled on the
  device (also needed for buffer-reference vertex lookup).
- Mesh upload (`mesh.cpp::uploadMesh`) also builds a BLAS per mesh when
  `RT_SUPPORTED` — vertex/index buffers gain
  `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | …AS_BUILD_INPUT_READ_ONLY_BIT_KHR`.
- TLAS rebuilt every frame in `renderer.cpp::drawFrame` via `buildTlas`.
- `mesh.frag` declares `accelerationStructureEXT topLevelAS` at scene set
  binding 4 and does `rayQueryInitializeEXT` / `rayQueryProceedEXT` for
  shadow rays.

**Phase 2 — RT Reflections** (`rtSettings.reflections`).
- Parallel SSBO `RtInstanceMaterial[]` at scene set binding 5, indexed by
  `gl_RayQueryInstanceCustomIndexEXT`. Holds per-instance albedo, metallic,
  roughness, and **vertex/index buffer device addresses** (split into two
  `uvec2`s).
- `mesh.frag::shadeReflectionHit` reads the hit triangle's 3 vertex normals
  via `GL_EXT_buffer_reference` with `scalar` layout, interpolates with
  `rayQueryGetIntersectionBarycentricsEXT`, transforms to world space via
  `inverse(transpose(objectToWorld))`, casts a secondary shadow ray, and
  shades with Lambert + sun + small hemisphere ambient.
- Cone width = `r² × 0.3` half-angle (1° at roughness 0.25, 17° at 1.0).
- Metallic-only gate (`metallic > 0.5`) — dielectrics get no reflection
  contribution by design (avoids grazing-angle noise on plastics).

**Phase 3 — RT GI** (`rtSettings.gi`).
- Cosine-weighted hemisphere ray, 12 samples by default.
- Reuses `shadeReflectionHit` for the hit shading (Lambert + sun + ambient).
- Per-sample radiance clamped to `vec3(1.0)` to suppress fireflies.
- Mixed with the static hemisphere ambient via `rtParams3.w` slider
  (default 0.15 = mostly smooth, subtle color bleed). Without temporal
  accumulation, pure RT GI is too noisy at any practical sample count;
  the mix is the trade-off knob.
- Skipped for metallics (their indirect is purely specular reflection).

### DLSS (Phase 4, partial)

**Step 4a (complete).**
- `dlss.{h,cpp}` wraps NGX. `dlssInit` calls `NVSDK_NGX_VULKAN_Init_with_ProjectID`
  with a **UUID-format** project ID (NGX rejects free-form strings via
  `NGXValidateIdentifier`).
- `dlssRequiredInstanceExtensions` / `dlssRequiredDeviceExtensions` query
  NGX for the extensions it needs *before* instance/device creation. Engine
  init walks them and merges into the extension lists.
- Path list + log directory set via `NVSDK_NGX_FeatureCommonInfo`. NGX
  writes its own log to `build/dlss_logs/nvsdk_ngx.log`. Console mirror is
  silenced.
- Capability probed via `NVSDK_NGX_Parameter_SuperSampling_Available`.
- Console prints `[DLSS] NGX initialised, DLSS Super Sampling available`
  on success.

**Step 4b (foundation in place, motion vectors in flight).**
- `DlssSettings` (enabled / quality preset / jitterEnabled).
- `haltonJitter(index)` produces a Halton(2,3) offset in `[-0.5, +0.5]`.
- Engine applies sub-pixel jitter to the camera proj (`ubo.proj[2][0/1] +=
  jitterNDC`) when DLSS is on.
- `SceneUBO` gained `prevViewProj` (mat4) and `jitterOffset` (vec4). Each
  GLSL stage that declares the UBO must list them in matching order or the
  std140 layout breaks — see "Key conventions".
- **Motion vectors render target** added to `OffscreenTarget` as RG16F
  second color attachment. Render pass has 3 attachments now (color, motion,
  depth). Main + skinned pipelines both declare 2 color blend states.
- `mesh.frag` writes `(prevNDC - currNDC)` to attachment 1, with the jitter
  offset subtracted from `currNDC` so motion encodes only real motion.

**Step 4c (complete).** `dlssCreateFeature` / `dlssEvaluate` /
`dlssGetOptimalRenderSize` / `dlssReleaseFeature` wrap the NGX entry points.
`Engine::computeRenderExtent` picks the offscreen extent: full swapchain
extent when DLSS is off, NGX-reported optimal input size when on. Offscreen,
bloom, SSAO, composite, LDR, and HZB are all built at this `m_renderExtent`;
only the new `UpscaleTarget m_upscale` and the swapchain itself live at full
resolution. Toggling DLSS / changing the quality preset is detected at the
top of each frame, triggering `recreateSwapchain()` + `rebuildDlssFeatureIfNeeded()`
in one shot. The renderer inserts a DLSS evaluate between composite and FXAA
when a feature is alive: LDR (low-res, SHADER_READ_ONLY) → upscale (full-res,
GENERAL→SHADER_READ_ONLY), with motion + depth fed from the offscreen pass.
FXAA samples from `m_upscale.descriptorSet` instead of `m_ldr.descriptorSet`
when DLSS is active. Camera jitter is computed against `m_renderExtent`
(input pixels), and the same Halton offset is forwarded to NGX as
`InJitterOffsetX/Y` in input-pixel space.

### Post-FX

- HDR offscreen (`RGBA16F`) **+ motion vectors (`R16G16_SFLOAT`)** + sampleable depth
- Bloom: 4-mip downsample/upsample chain
- **SSAO** auto-suppressed when RT shadows are active (RT GI replaces it).
  When RT is off, SSAO runs normally.
- Composite: ACES/Reinhard/Off tonemap, color lift, saturation, contrast, vignette
- FXAA 3.11 lite

### ECS (EnTT)

Same as before: `TransformComponent`, `MeshComponent`, `MaterialComponent`,
`MeshLODComponent`, light components, skeletal components.

### Resource Manager

Same as before. Note: mesh upload now also triggers BLAS build when
`RT_SUPPORTED`. `destroyMesh` tears down the BLAS via
`destroyRtMeshGeometry`.

### Scene editor (ImGui + ImGuizmo)

New panels added:
- **Ray Tracing** — master enable, per-feature toggles (shadows /
  reflections / GI), sun-only checkbox, shadow softness + samples,
  reflection samples / max-dist / intensity, GI samples / max-dist /
  intensity (= mix factor).
- **DLSS** — master enable, sub-pixel jitter toggle, quality preset combo.
  Visible only if DLSS init succeeded.

### Camera, Skeletal — unchanged

### Performance infrastructure

GPU-driven culling unchanged except:
- **HZB is now built at half offscreen resolution** (was full). `hzb_reduce.comp`
  does a 2× downscale; mip 0's source is the depth attachment, so HZB mip 0
  must be half the depth size or clamping corrupts ~75% of mip 0. This was
  a latent bug that masked itself behind the earlier frustum-near-plane
  issue. See `engine.cpp::createHzb` call sites — `hzbExtent` is explicitly
  half of `m_offscreen.extent`.

## Key conventions

### SceneUBO layout

**Critical**: GLSL `SceneUBO` declarations in `mesh.vert`, `mesh.frag`, and
`mesh_skinned.vert` must list every field of the C++ `UniformBufferObject`
in matching order. Adding a field to the C++ struct silently shifts std140
offsets in every shader that reads the UBO. Current layout:

```glsl
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 rtParams;      // x=shadows on, y=light angular radius, z=shadow samples, w=sunOnly
    vec4 rtParams2;     // x=reflections on, y=samples, z=tMax, w=intensity
    vec4 rtParams3;     // x=GI on, y=samples, z=tMax, w=mix factor
    mat4 prevViewProj;  // motion vectors / TAA
    vec4 jitterOffset;  // .xy = NDC jitter applied to proj
} scene;
```

### Vulkan features required

- Vulkan 1.2 descriptor indexing (bindless textures)
- `samplerAnisotropy`
- `shaderInt64` — for buffer-reference pointer reconstruction
- `bufferDeviceAddress` — when `RT_SUPPORTED`
- `scalarBlockLayout` — for `GL_EXT_scalar_block_layout` in mesh.frag's
  vertex-buffer reads
- `accelerationStructure`, `rayQuery` — when `RT_SUPPORTED`

### Global capability flags

- `VRS_SUPPORTED` — `VK_KHR_fragment_shading_rate` available
- `RT_SUPPORTED` — three RT extensions + features
- `dlssAvailable()` — NGX init succeeded and DLSS Super Sampling is offered

All three guard their code paths defensively. The engine boots and runs
cleanly with any combination off.

### Descriptor sets

- Graphics set 0 (scene):
  - binding 0: scene UBO (with `prevViewProj`, `jitterOffset`)
  - binding 1: lights UBO
  - binding 2: cascade UBO
  - binding 3: shadow array
  - binding 4: TLAS (only when RT_SUPPORTED)
  - binding 5: RT material SSBO (only when RT_SUPPORTED)
- Graphics set 1 (bindless textures): 1024-slot sampler array
- Graphics set 2 (skinned only): bone palette UBO
- Compute set (cull): unchanged
- Compute set (HZB reduce): unchanged

### Push constants — unchanged

### Vertex input — unchanged

### Texture coordinates / camera projection — unchanged

### DLSS project ID format

NGX validates the project ID and rejects anything non-UUID. The engine uses
a hardcoded UUID `5b7e8f4a-2c9d-4e3f-a1b6-c8d7e9f0a1b2`. Replace with a
registered UUID when shipping with the release-signed DLL.

## Default scene — unchanged

Sun + warm fill point light + steel/red/gold spinning cubes + floor.

## Known issues / deferred work

### DLSS upscale runs at LDR (post-tonemap), not HDR

The current placement (per the original handoff plan) feeds DLSS a
tonemapped, gamma-corrected LDR image from the composite pass — not the
canonical HDR pre-composite input NVIDIA recommends. This loses some
quality (DLSS' internal exposure heuristics and temporal accumulation are
tuned for HDR linear input) but keeps the pipeline simple: a single
DLSS pass slots in cleanly between composite and FXAA, and bloom/SSAO/
composite all run unchanged at the low render extent.

To switch to HDR DLSS: move the DLSS pass to run between the main pass
and SSAO/bloom, write its output into a full-res HDR image, and rebuild
the bloom/SSAO/composite chain at full resolution off that image (or
keep them at low res and accept the bandwidth — common). The
`NVSDK_NGX_DLSS_Feature_Flags_IsHDR` create flag needs to be set in
`dlssCreateFeature` too.

### Per-instance previous transform not tracked

Motion vectors currently use the *current* model matrix for both prev and
curr clip positions — so they encode only camera motion, not object
motion. Rotating cubes won't have correct motion vectors. To finish:
extend `InstanceData` with `mat4 prevModel`, populate it from a CPU-side
`unordered_map<entt::entity, glm::mat4>` updated after each frame, and use
it in `mesh.frag::main`'s motion vector computation.

### Memory allocator
- Every buffer/image hits `vkAllocateMemory` directly. ~4096 allocation
  limit will bite eventually. VMA integration is the standard fix — call
  sites are in `buffer.cpp`, `shadow.cpp`, `postfx.cpp`, `texture.cpp`,
  `lights.cpp`, `instancing.cpp`, `skeletal.cpp`, `raytracing.cpp` (BLAS/TLAS
  backing buffers and instance/scratch buffers).

### Engine god-class
- `Engine` owns everything. Extract `Renderer` (GPU-resource owner + per-frame
  draw) eventually.

### Skeletal animation gaps
- Single skinned entity per frame.
- Skinned shadows still not drawn (don't write to shadow array; also won't
  contribute to RT shadow because skinned mesh doesn't have a BLAS).
- No glTF importer.

### GPU-driven culling caveats
- Single-pass occlusion (previous-frame HZB).
- Shadow pass ignores occlusion + LOD when it runs (CSM path).
- `visibleEntities` UI stat is `-1` (dead).
- `MAX_LOD = 4` hardcoded; bump requires CPU+GLSL stride agreement.
- Attachment-based VRS not implemented.

### GI noise
- Single-frame Monte Carlo without denoising. The mix-with-ambient slider
  (`rtParams3.w`) is the workaround. Proper fix: temporal accumulation
  with a history buffer. Sized to ~half-res with bilateral upsample is the
  standard.

### Reflections at hit point
- Hit shading is Lambert with the sun only (no point/spot at hits).
- No mirror-of-mirror recursion (single bounce).
- For very rough metals, `roughness >= 0.85` skips the trace entirely —
  noise would dominate the contribution.

### DLSS-specific gotchas
- NGX dev DLL bypasses app allowlisting; the release DLL would need a
  registered project ID with NVIDIA.
- Debug build links `nvsdk_ngx_d_dbg.lib`, release links `nvsdk_ngx_d.lib`.
  CRT iterator levels must match.
- Locked-exe LNK1168 is the most common build failure since the Overwolf /
  Medal / OBS Vulkan layers cling to handles.

## File-by-file index of significant entry points

- **Engine startup**: `Engine::init()` in `engine.cpp`. Order is
  important — RT extension queries happen *before* instance and device
  creation; DLSS extension queries happen between instance + device creation;
  `dlssInit` happens right after device creation; `createShadowResources` now
  also takes a command pool + queue for the initial layout transition.
- **Per-frame draw**: `drawFrame()` in `renderer.cpp`. Reads `DrawFrameInfo`,
  builds TLAS, dispatches cull, optionally skips shadow pass when RT shadows
  active, records main pass (now 3 attachments), HZB reduce, SSAO, bloom,
  composite, FXAA.
- **Batch construction**: `buildCandidates()` static in `renderer.cpp`.
- **RT instance gathering**: same loop in `drawFrame`, fills
  `RtInstanceDesc` array passed to `buildTlas`.
- **BLAS build**: `buildMeshBlas()` in `raytracing.cpp`, called from
  `mesh.cpp::uploadMesh` when `RT_SUPPORTED`.
- **TLAS build**: `buildTlas()` in `raytracing.cpp`. Issues pre-build
  HOST_WRITE → AS_WRITE barrier and post-build AS_WRITE → SHADER_READ.
- **NGX init**: `dlssInit` in `dlss.cpp`.
- **Swapchain rebuild**: `Engine::recreateSwapchain()` — rebuilds offscreen
  (which now includes the motion image), HZB at half-extent, all post-fx.
- **Shadow cascade math**: `computeCascades()` in `shadow.cpp` — `zExtend
  = radius * 10` for caster coverage; uses Vulkan ZTO `glm::ortho`.

## Quick "I want to add X" recipes

- **Add a new mesh and instance it**: same as before. Mesh upload now also
  builds a BLAS automatically when RT is supported — no caller changes.

- **Add an LOD chain**: same as before.

- **Add a new texture**: same as before.

- **Add a new post-FX pass**: same as before.

- **Add a new descriptor in the scene set**: edit
  `descriptors.cpp::createSceneSetLayout` (bindings array + size), pool
  size in `createDescriptorPool`, write logic in `createSceneDescriptorSets`,
  and **GLSL declarations in *every* shader that uses set 0** (mesh.vert,
  mesh.frag, mesh_skinned.vert). Mismatched UBO declarations corrupt
  std140 offsets silently.

- **Add an RT feature** (e.g. RT ambient occlusion): trace from `mesh.frag`
  using `rayQueryEXT` against `topLevelAS`. Hit material is reachable via
  `gl_RayQueryInstanceCustomIndexEXT → rtMats.materials[idx]`. Hit normal
  via the `Vertices(packUint2x32(mat.vertexAddr))` buffer-reference dance
  in `shadeReflectionHit`.

## Last-known-good build state

- Last successful build linked `VulkanEngine.exe` with Phase 1, 2, 3 and
  the full Step 4a/4b/4c stack. DLSS off-path renders identically to
  Phase 3 (full-res offscreen, no jitter, FXAA samples LDR). DLSS on-path
  renders the offscreen at the NGX-reported optimal input size, runs
  DLSS into `m_upscale`, and routes FXAA from there.
- Shader count: 12 (same as before — no new shader files added; existing
  mesh.frag grew substantially).
- Console on startup (5070 Ti, driver 591.86):
  - `[VulkanEngine] Logical device created (VRS supported) (RT supported)`
  - `[DLSS] NGX initialised, DLSS Super Sampling available`
- Expected validation noise:
  - 7× `Vertex attribute at location N not consumed` for the shadow
    pipeline (binds full InstanceData layout but shadow.vert only reads
    position + model).
  - 2× `loader_get_json` for the missing Medal recorder shims — Medal
    bug, harmless.
  - 2× `Layer ... API version 1.2 ... older than 1.3` for the Overwolf
    overlays — harmless.
- The Halton jitter is sub-pixel; with DLSS toggled on you might see a
  faint 1-pixel shimmer on contrasty edges. That's the only visible
  effect of DLSS today.
