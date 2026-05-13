#version 450

layout(location = 0) in vec3 vWorldDir;

layout(location = 0) out vec4 outColor;
// Same MRT layout as mesh.frag — motion vector attachment. Sky has no
// per-pixel motion (it's at infinity, so camera translation doesn't move it).
// Camera *rotation* would technically shift it, but for now zero is correct
// enough for TAA's purposes: the sky pixels just won't be temporally
// re-projected, which reads as "fine, the sky doesn't ghost."
layout(location = 1) out vec2 outMotion;

// Sample the prefiltered specular cubemap at mip 0 — that's the unfiltered
// environment (the bake puts the raw env there before convolution layers
// fill the remaining mips). For HDR backgrounds this is the most faithful
// representation: tonemap happens later in composite.
layout(set = 0, binding = 7) uniform samplerCube prefilterMap;

void main() {
    vec3 dir = normalize(vWorldDir);
    outColor  = vec4(textureLod(prefilterMap, dir, 0.0).rgb, 1.0);
    outMotion = vec2(0.0);
}
