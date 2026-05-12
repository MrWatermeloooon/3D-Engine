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
relative `shaders/*.spv` paths resolve (CMake post-build copies SPIR-V next
to the exe — clicking the .exe from Explorer works).

> ⚠️ The old multi-config Visual Studio build at `build/Debug/VulkanEngine.exe`
> is stale. Always run `build/VulkanEngine.exe` (Ninja single-config output).

## Dependencies (vcpkg manifest)

`vcpkg.json` declares: `glfw3`, `glm`, `stb`, `tinyobjloader`, `entt`,
`imgui` (features: `glfw-binding`, `vulkan-binding`, `docking-experimental`),
`imguizmo`. All header-only or thin libs — no native deps to manage.

## Project Layout

```
src/
  main.cpp                  # entry, owns one Engine, calls init/run/cleanup
  utils/vk_check.h          # VK_CHECK macro
  engine/
    engine.{h,cpp}          # game loop + owns all GPU state (god class — see below)
    window.{h,cpp}          # GLFW wrapper, input, scroll callback
    vulkan_init.{h,cpp}     # instance, debug, device, queue families
    swapchain.{h,cpp}       # swapchain + color-only render pass + sync objects
    pipeline.{h,cpp}        # graphics, shadow, and skinned pipelines
    buffer.{h,cpp}          # buffer/image create/destroy, single-time cmds
    descriptors.{h,cpp}     # scene UBO + bindless material set
    mesh.{h,cpp}            # static mesh, cube builder, OBJ loader
    texture.{h,cpp}         # stb_image loader, mipmaps, default white texture
    vertex.h                # Vertex (position/normal/uv/color)
    camera.{h,cpp}          # FPS + Orbit camera
    components.h            # ECS components (Transform, Mesh, Material, lights, etc.)
    scene.{h,cpp}           # entt::registry wrapper + default scene
    resource_manager.{h,cpp}# handle-based loading, hot reload, bindless registration
    debug_ui.{h,cpp}        # ImGui dockspace + panels + ImGuizmo
    renderer.{h,cpp}        # per-frame drawFrame; shadow → main → SSAO → bloom → composite → fxaa
    lights.{h,cpp}          # light buffer (UBO), 3 light types
    shadow.{h,cpp}          # CSM (4 cascades), stable texel-snapped
    postfx.{h,cpp}          # HDR offscreen, bloom, SSAO, composite, LDR, FXAA
    depth.{h,cpp}           # depth resources + format selection
    frustum.{h,cpp}         # 6-plane extraction + AABB test
    instancing.{h,cpp}      # InstanceData + Candidate/BatchHeader/Indirect buffers
    skeletal.{h,cpp}        # Skeleton, Animation, SkinnedMesh, test bone chain
    jobs.{h,cpp}            # thread-pool with parallel_for + wait_all
    gpu_cull.{h,cpp}        # compute-cull pipeline, params UBO, dispatch helper
    hzb.{h,cpp}             # hierarchical Z-buffer (occlusion culling source)

shaders/
  fullscreen.vert           # full-screen triangle from gl_VertexIndex
  mesh.vert / mesh.frag     # PBR static draw (Cook-Torrance + CSM shadows)
  mesh_skinned.vert         # LBS skinning, shares mesh.frag
  shadow.vert               # depth-only with per-cascade lightVP push constant
  bloom_downsample.frag     # Karis 13-tap weighted average
  bloom_upsample.frag       # 3×3 tent, additive blend
  ssao.frag                 # 16-sample hemisphere SSAO, depth-derivative normals
  composite.frag            # tonemap + grading + vignette + debug views
  fxaa.frag                 # FXAA 3.11 lite
  cull.comp                 # GPU culling: frustum + HZB-occlusion + LOD pick
  hzb_reduce.comp           # 2×2 max-depth reduction (one dispatch per mip)
```

## Render Pipeline

Per-frame ordering inside `renderer.cpp::drawFrame`:

1. **Build candidates** (CPU, parallelized via JobSystem if >256 entities) — collect
   entities, group into `(LODGroup, texture)` batches, write one `CandidateInstance`
   per entity (model + AABB + batch index) and one `BatchHeader` per batch
   (per-LOD slot offsets + indirect cmd indices). **No CPU frustum culling, no
   visible/total split** — that work moved to the GPU.
2. **Compute cull dispatch** — `cull.comp`, one thread per candidate, does:
   frustum cull, previous-frame HZB occlusion test, LOD pick by camera distance,
   atomic-append into the chosen LOD's `mainInstance` slot + draw cmd. Shadow
   draws are unconditional (no occlusion, always LOD0).
3. **Shadow pass** — 4 CSM cascades, depth-only, one `vkCmdDrawIndexedIndirect`
   per batch (LOD0 mesh, all instances).
4. **Main pass** (offscreen HDR RGBA16F + sampleable depth) — for each batch and
   each of its LODs, set fragment shading rate (per-LOD if VRS supported) then
   `vkCmdDrawIndexedIndirect`. Skinned meshes follow inside the same render pass.
5. **HZB reduction** — `hzb_reduce.comp` walks every mip (mip 0 sources from the
   just-written depth attachment, mips 1..N source from the previous HZB mip).
   Result is consumed by *next* frame's compute cull.
6. **SSAO pass** — half-res R8, reads offscreen depth + 4×4 noise texture.
7. **Bloom** — 4-mip downsample chain (Karis-weighted) then upsample chain
   (additive blend, 3×3 tent).
8. **Composite pass** — HDR → LDR (RGBA8): exposure, bloom add, SSAO multiply,
   ACES/Reinhard tonemap, color grading (lift/saturation/contrast), vignette.
9. **FXAA + ImGui pass** — FXAA reads LDR, writes swapchain. ImGui draws on top.

Swapchain image format is `B8G8R8A8_SRGB`. The composite uses `gamma = 2.2` as the
neutral value (relying on sRGB swapchain conversion); the slider applies a
*relative* gamma curve on top.

## Features (what works today)

### PBR (Cook-Torrance)
- GGX normal distribution, Smith with Schlick-GGX geometry, Schlick Fresnel
- Metallic + roughness per material via `InstanceData.matParams`
- Hemisphere ambient (sky/ground tint by `N.y`)
- Linear HDR output; tonemap happens in composite

### Lights
- `DirectionalLightComponent`, `PointLightComponent`, `SpotLightComponent`
- Packed into a `LightsUBO` (32 max), 64 bytes each, `LightData`
- Type stored in `positionAndType.w` (0/1/2 = dir/point/spot)
- First directional light casts CSM shadows; others light without shadowing

### Cascaded shadow maps
- 4 cascades, 2048×2048, single `VK_IMAGE_VIEW_TYPE_2D_ARRAY` for sampling +
  per-layer views for framebuffers
- Practical split scheme (lambda 0–1 lerps between uniform & logarithmic)
- **Stable cascades**: sphere center texel-snapped in light space each frame
- 5×5 PCF + normal-offset bias scaled by cascade index
- `sampler2DArrayShadow` (hardware comparison + linear filter)
- Front-face culling in shadow pass to reduce peter-panning

### Post-FX
- HDR offscreen target (`RGBA16F`) + sampleable depth
- Bloom: 4-mip downsample/upsample chain, Karis weighted at the source mip
- SSAO: 16 samples, hemisphere oriented by normal-from-depth-derivatives, range-checked
- Composite: ACES/Reinhard/Off tonemap modes, color lift, saturation, contrast,
  vignette
- FXAA 3.11 lite in its own pass

### ECS (EnTT)
- `TransformComponent` (pos/rot/scale + `getMatrix()`)
- `MeshComponent`, `MaterialComponent` (texture handle + color + metallic + roughness)
- `NameComponent`, `RotatorComponent` (per-axis spin)
- Light components (3 types)
- `SkinnedMeshComponent`, `AnimatorComponent`

### Resource Manager
- Handle-based: `MeshHandle`, `TextureHandle` (`uint32_t` IDs)
- Path-based dedup cache
- **Hot reload**: `std::filesystem::last_write_time` poll every 2 s; reloaded
  textures write into the same bindless slot (no descriptor churn)
- `TextureHandle.id` *is* the bindless array slot index

### Scene editor (ImGui + ImGuizmo)
- Docked sidepanel layout (auto-built first frame via `DockBuilder`)
- Panels: Hierarchy, Performance, Properties, Lights, Post-Processing, Shadows, Camera
- ImGuizmo for translate/rotate/scale of the selected entity (W/E/R hotkeys, World/Local)
- Import Model... (Win32 file dialog → OBJ via tinyobjloader; computes smooth
  normals when the OBJ doesn't provide them)
- Spawn Cube / Spawn 1000 Cubes / Spawn Skinned Test / Spawn Point/Spot Light

### Camera
- Orbit (default): LMB rotate, RMB pan, scroll zoom
- FPS: WASD + mouse look, Space/Shift up/down
- Tab toggles modes; ImGui captures input when hovering UI

### Skeletal animation
- `Skeleton` (joints with parent index + inverse-bind matrix + local TRS)
- `Animation` (channels with T/R/S keyframes, linear interp + slerp)
- Programmatic test bone chain (`createTestBoneChain()`)
- `SkinnedVertex` (position + normal + uv + uvec4 joint IDs + vec4 weights)
- `mesh_skinned.vert` — LBS, max 64 bones per palette
- Separate skinned pipeline (3 descriptor sets: scene, bindless textures, bones)
- Currently single-entity per frame (engine writes one shared bone palette UBO).
  Skinned meshes do **not** cast shadows yet.

### Performance infrastructure (GPU-driven pipeline)

The CPU only assembles candidates and batch metadata. All visibility decisions —
frustum, occlusion, LOD — live in `cull.comp`.

- **Candidate stream**: per-frame host-visible SSBO of `CandidateInstance`
  (`InstanceData` + world AABB + batch index packed into `aabbMin.w`)
- **Batch headers**: per-frame SSBO of `BatchHeader` (64 B): shadow base/cmd,
  `lodCount`, plus per-LOD `lodBase[4]`/`lodCmd[4]`/`lodDist[4]`
- **Compute cull** (`gpu_cull.{h,cpp}` + `cull.comp`):
  - 64 threads/group, one thread per candidate
  - Reads `CullParamsUBO` (prev view-proj, frustum planes, camera pos, HZB
    metadata, candidate count) — push constants would have exceeded 128 B
  - Per candidate: frustum test → HZB occlusion test → LOD pick by view distance
    → `atomicAdd` into chosen LOD's `instanceCount`, store `InstanceData` to
    `mainInstance[base + slot]`. Shadow output is unconditional (LOD0 only).
- **Two output instance buffers**: `mainInstance` (sized `MAX_INSTANCES × MAX_LOD`
  for worst-case "all picked the same LOD") and `shadowInstance` (sized
  `MAX_INSTANCES`). Both are vertex+SSBO usage; compute writes, vertex reads.
- **Indirect buffer** (capacity `MAX_BATCHES × (MAX_LOD + 1)`): per batch
  reserves `lodCount` main cmds + 1 shadow cmd, contiguous. CPU pre-writes the
  static fields (`indexCount`, `vertexOffset`); compute atomically grows
  `instanceCount`. Indirect+SSBO usage so compute can write to it.
- **LOD groups**: `MeshLODGroup` registered via `ResourceManager::addLODGroup`,
  referenced by `MeshLODComponent`. Each level = `(MeshHandle, maxDistance)`,
  finest first. Cap `MAX_LOD = 4`. Single-mesh entities (`MeshComponent` only)
  are treated as a synthetic 1-LOD chain. Procedural icosphere generator
  (`createIcosphereMesh(uint32_t subdivisions)`) provides demo content.
- **HZB occlusion** (`hzb.{h,cpp}` + `hzb_reduce.comp`): R32_SFLOAT image at
  offscreen resolution, full mip chain (~11 mips at 1280×720). Built each frame
  *after* the main pass from the just-written depth attachment, used by *next*
  frame's compute cull. Image lives in `VK_IMAGE_LAYOUT_GENERAL` to avoid
  per-mip transitions; barriers are memory-only between dispatches. Initial
  state cleared to depth=1.0; first frame disables occlusion via UBO flag.
  Single-pass (previous-frame) approximation — fast camera moves can briefly
  false-cull newly-revealed geometry. Two-pass (HZB on this frame's late draws)
  is the natural next upgrade.
- **Variable-rate shading**: `VK_KHR_fragment_shading_rate` is *probed* at
  device creation (`VRS_SUPPORTED` global + `VRS_SetRate` function pointer).
  When supported, the main pipeline declares
  `VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR` and `recordMainPass` calls
  `vkCmdSetFragmentShadingRateKHR` per LOD: LOD0=1×1, LOD1=2×1, LOD2=2×2,
  LOD3=4×4 (or a forced-rate override from the UI). All paths no-op safely on
  unsupported hardware. *Attachment-based* VRS (per-tile shading rate image
  authored from edge detection) is intentionally not implemented — would
  require migrating the offscreen render pass to `VkRenderPass2`.
- **Bindless textures**: Vulkan 1.2 descriptor indexing, single 1024-slot
  combined-image-sampler array, `nonuniformEXT` indexing in `mesh.frag`,
  `UPDATE_AFTER_BIND` so hot-reload doesn't need a queue stall
- **Job system**: worker pool sized to `hardware_concurrency() - 1`,
  `parallel_for(N, fn)` used in `buildCandidates` for per-entity matrix math

## Key conventions

- **Push constants**: only used by shadow vertex (lightViewProj, 64 bytes) and
  the skinned pipeline (model + colorTint + matParams, 96 bytes).
  The main pipeline uses zero push constants — everything is per-instance.
- **Per-frame resources**: always sized `MAX_FRAMES_IN_FLIGHT` (currently 2).
  Light/Scene/Cascade/Composite/SSAO/Candidate/BatchHeader/MainInstance/
  ShadowInstance/Indirect/CullParams/Bone palette UBOs are all per-frame.
- **Descriptor sets**:
  - Graphics set 0 (scene): scene UBO + lights UBO + cascade UBO + shadow array
  - Graphics set 1 (bindless textures): 1024-slot sampler array
  - Graphics set 2 (skinned only): bone palette UBO
  - Compute set (cull): 5 SSBOs (candidates, batch headers, main instance,
    shadow instance, indirect) + 1 UBO (CullParams) + 1 sampler2D (HZB)
  - Compute set (HZB reduce): one set per mip — sampled-image source +
    storage-image dest. Mip 0's source is the offscreen depth view.
- **Vertex input** (main pipeline):
  - Binding 0 (per-vertex): position / normal / uv / color → locations 0..3
  - Binding 1 (per-instance, `VK_VERTEX_INPUT_RATE_INSTANCE`):
    `mat4 model` at 4..7, `vec4 colorTint` at 8, `vec4 matParams` at 9
    (x=metallic, y=roughness, z=textureIndex, w=pad), normal matrix columns
    at 10..12
- **Texture coordinates**: V is flipped (`1.0 - v`) in the OBJ loader to match
  Vulkan UV convention
- **Camera projection**: `glm::perspective` with `proj[1][1] *= -1` for Vulkan
  Y-down NDC; `getProjectionMatrixUnflipped()` exists for ImGuizmo

## Default scene

`scene.cpp::createDefaultScene` spawns:
- Sun (directional, warm white, intensity 3.5)
- Gray rough floor (15×0.1×15)
- Steel cube (metallic 1.0 / rough 0.25, spinning)
- Red plastic cube (metallic 0 / rough 0.4, spinning)
- Gold cube (metallic 1.0 / rough 0.2, spinning)
- Warm Fill (point light, orange, intensity 1.5)

The "Spawn 1000 Cubes" button drops 40×25 cubes in a grid — useful for testing
that frustum culling and indirect/instanced rendering hold up.

## Known issues / deferred work

These were flagged in a code review but not yet addressed:

### Memory allocator
- Every buffer/image hits `vkAllocateMemory` directly. NVIDIA's per-device
  allocation cap is ~4096 — fine today, will hit in a real-asset scene.
- **Next step**: integrate **VMA** (`VulkanMemoryAllocator`). All allocation
  sites are in `buffer.cpp`, `shadow.cpp`, `postfx.cpp`, `texture.cpp`,
  `lights.cpp`, `instancing.cpp`, `skeletal.cpp`, `engine.cpp` (bone UBOs).
  ~10 call sites total. Mechanical refactor.

### Engine god-class
- `Engine` owns: window, vulkan context, swapchain, pipelines, descriptors,
  shadow, lights, bloom, SSAO, composite, LDR, instance buffer, indirect
  buffer, bone UBOs, debug UI, scene, resource manager, job system, post-fx
  pipelines, skinned mesh.
- **Next step**: extract a `Renderer` class that owns all GPU resources and
  the per-frame draw call. `Engine` stays for the game loop + input + scene
  ownership. Roughly halves the size of `engine.{h,cpp}`.

### Skeletal animation gaps
- Only supports **one** skinned entity per frame (single shared bone palette).
- Skinned meshes are **not drawn in the shadow pass** (need a `shadow_skinned.vert`
  + extra shadow pipeline using the same bone palette layout).
- **No glTF importer** — only the procedural test bone chain. tinygltf needs
  adding to `vcpkg.json`; loader fills `Skeleton` from `skin.joints` +
  `inverseBindMatrices`, animations from `animations[].channels[]`. The
  `skeletal.h` data structures are already shaped 1:1 with glTF semantics.

### GPU-driven culling caveats (now implemented; remaining work)
- **No two-pass occlusion.** The cull shader uses *previous-frame* HZB +
  view-proj. Whip the camera around fast and recently-revealed geometry can
  flicker for a frame. Industry fix: render last-frame visibles → build HZB
  from this-frame depth → late-cull all → render newly-visible. Needs a
  per-instance "was visible last frame" SSBO and two cull dispatches.
- **Shadow pass ignores both occlusion and LOD.** Always draws every instance
  with LOD0. Per-cascade GPU cull (one HZB per cascade) is a future upgrade;
  per-cascade LOD selection probably less impactful.
- **`visibleEntities` UI stat is dead** — set to -1. The previous CPU-side
  count is gone; reviving it needs a small SSBO write at the end of `cull.comp`
  (atomic counter), read back next frame after the fence wait.
- **Worst-case main instance reservation = `lodCount × N` per batch.** If you
  push `MAX_INSTANCES_PER_FRAME = 16384` with `MAX_LOD = 4`, the main
  instance buffer is `4 × 16384 × 144 = 9.4 MB` per frame in flight. Fine,
  but the over-reservation could be tightened with a CPU LOD-guess prepass.
- **`MAX_LOD` is hardcoded to 4** — bumping it grows `BatchHeader` (CPU + GLSL
  must agree on stride) and the main instance over-reservation.
- **Attachment-based VRS is not implemented.** Pipeline/dynamic VRS gives
  per-LOD shading rates today; per-tile shading-rate-image authored from
  prev-frame edge detection is the natural next step but requires migrating
  the offscreen render pass to `VkRenderPass2` (and chaining
  `VkFragmentShadingRateAttachmentInfoKHR` into the subpass).

## File-by-file index of significant entry points

- **Engine startup**: `Engine::init()` in `engine.cpp`. Order matters — command
  pool is created early because mesh/texture upload needs it.
- **Per-frame draw**: `drawFrame()` in `renderer.cpp`. Reads `DrawFrameInfo`.
- **Batch construction**: `buildBatches()` static in `renderer.cpp` —
  the only hot CPU path per frame, parallelized via `JobSystem::parallel_for`.
- **Swapchain rebuild**: `Engine::recreateSwapchain()` — also tears down and
  rebuilds all extent-dependent resources (offscreen, bloom, SSAO, LDR,
  composite, post-FX pipelines, main pipeline).
- **Shadow cascade math**: `computeCascades()` in `shadow.cpp` — texel-snapping
  is in the middle of that function and the comment block above it explains why.
- **Bindless registration**: `ResourceManager::loadTexture` /
  `ResourceManager::init` call `writeBindlessTexture(slot, view, sampler)`.
  Slot equals the texture handle ID.

## Quick "I want to add X" recipes

**Add a new mesh and instance it**: load via `m_resources.loadMesh(path)`
→ create an entity → add `TransformComponent`, `MeshComponent`,
`MaterialComponent` (with a `TextureHandle` from `loadTexture`). It joins
the next frame's batch automatically; entities sharing `(mesh, texture)`
get coalesced into one `vkCmdDrawIndexedIndirect`.

**Add an LOD chain**: `MeshLODGroup g; g.levels = { {meshHi, 12.0f}, {meshMid,
30.0f}, {meshLo, FLT_MAX} }; auto h = m_resources.addLODGroup(std::move(g));`.
Then on the entity: `registry.emplace<MeshLODComponent>(e, MeshLODComponent{h});`.
Levels MUST be ordered finest-first; the cull shader picks the lowest LOD
whose `maxDistance` exceeds the camera distance to AABB center. The
`MeshComponent.handle` on the same entity is ignored for the main pass when
`MeshLODComponent` is present, but is still required (and used as the
shadow-pass mesh — keep it set to LOD0 or finer).

**Add a new texture**: `m_resources.loadTexture("path.png")` returns a
`TextureHandle`. Store on `MaterialComponent.texture`. No descriptor work —
the handle IS the bindless slot.

**Add a new post-FX pass**: model it after `bloom_downsample.frag` /
`createBloomChain`. Add the target struct to `postfx.h`, create the render
pass + framebuffer in `postfx.cpp`, add the pipeline in
`createPostFXPipelines`, record it in `drawFrame` between bloom and composite.

**Add a new descriptor in the scene set**: edit `createSceneSetLayout` in
`descriptors.cpp` (add to the bindings array), update the pool size in
`createDescriptorPool`, update `createSceneDescriptorSets` to write the new
binding, and add the GLSL declaration to `mesh.vert`/`mesh.frag` at
`set = 0, binding = N`.

## Last-known-good build state

- Last successful build linked `VulkanEngine.exe` with all features above.
- 11 shaders compile clean: `mesh.vert`, `mesh.frag`, `shadow.vert`,
  `mesh_skinned.vert`, `fullscreen.vert`, `composite.frag`,
  `bloom_downsample.frag`, `bloom_upsample.frag`, `ssao.frag`, `fxaa.frag`,
  `cull.comp`, `hzb_reduce.comp`.
- Expected validation-layer noise from pipeline creation: 7×
  "Vertex attribute at location N not consumed by vertex shader". These come
  from the **shadow pipeline** (it binds the full `InstanceData` layout but
  `shadow.vert` only reads location 0 + the model matrix at 4..7). Harmless;
  fixing it cleanly would mean a separate per-instance binding for shadow.
- No runtime validation errors expected from the rendering loop. If they
  appear after a change, check descriptor pool sizing first
  (`createDescriptorPool` and `createGpuCull`'s pool) — it's the most common
  breakage source. Also watch for missing barriers around the new compute
  passes: compute-shader-write → vertex-attribute-read for instance buffers,
  → indirect-command-read for the indirect buffer, and depth-attachment-write
  → shader-read for the HZB source.
