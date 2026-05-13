// Single translation unit for VulkanMemoryAllocator's implementation.
// VMA is header-only by default; defining VMA_IMPLEMENTATION before the
// include emits all definitions exactly once. The header is included normally
// elsewhere (vulkan_init, buffer) — only this TU emits the bodies.
//
// We use VMA via the vcpkg `vulkan-memory-allocator` port, which exports
// `GPUOpen::VulkanMemoryAllocator` as a header-only interface target.

#define VMA_IMPLEMENTATION

// VMA pulls vulkan.h via its own include; we don't want the static
// loader-resolution disabled, so leave VMA_STATIC_VULKAN_FUNCTIONS at default.

// MSVC warns on VMA's unreferenced parameters / signed-unsigned compares; the
// VMA author silences them with vendor-specific pragmas at the top of the
// header, but defensively re-disable here for the duration of this TU.
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4100 4127 4189 4324 4505)
#endif

#include <vk_mem_alloc.h>

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
