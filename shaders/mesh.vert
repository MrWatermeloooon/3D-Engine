#version 450

// Per-vertex
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec3 inTangent;
// Per-instance (binding 1) — locations shifted up by 1 from earlier versions
// to make room for inTangent at vertex location 4.
layout(location = 5)  in vec4 instModel0;
layout(location = 6)  in vec4 instModel1;
layout(location = 7)  in vec4 instModel2;
layout(location = 8)  in vec4 instModel3;
layout(location = 9)  in vec4 instColorTint;
layout(location = 10) in vec4 instMatParams;  // x=metallic, y=roughness, z=textureIndex, w=normalTextureIndex
layout(location = 11) in vec4 instNormal0;    // precomputed normal matrix cols
layout(location = 12) in vec4 instNormal1;
layout(location = 13) in vec4 instNormal2;
// Previous-frame world matrix for per-object motion vectors.
layout(location = 14) in vec4 instPrevModel0;
layout(location = 15) in vec4 instPrevModel1;
layout(location = 16) in vec4 instPrevModel2;
layout(location = 17) in vec4 instPrevModel3;
// Second material params slot — parallax (x=heightSlot, y=parallaxScale, zw reserved).
layout(location = 18) in vec4 instMatParams2;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 rtParams;
    vec4 rtParams2;
    vec4 rtParams3;
    mat4 prevViewProj;
    vec4 jitterOffset;
} scene;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vTexCoord;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vWorldPos;
// vec4 (not vec3) so fragment shaders can recover linearized view-space depth
// from .w when DoF / SSAO depth reconstruction lands. The fragment shader
// reads .xyz for everything else; the .w just rides along.
layout(location = 4) out vec4 vViewPos;
layout(location = 5) flat out vec4 vColorTint;
layout(location = 6) flat out vec4 vMatParams;
// Per-vertex previous-frame clip-space position. We send the full clip-space
// vec4 (not the divided NDC) so the fragment can do its own /w with the
// interpolated W — that's the correct way to preserve per-pixel motion
// accuracy across a non-affine projection.
layout(location = 7) out vec4 vPrevClipPos;
// World-space tangent for normal mapping. Bitangent is reconstructed in the
// fragment shader as cross(N, T) — assumes the tangent generator used a
// consistent handedness (the standard for position+UV-derived tangents).
layout(location = 8) out vec3 vTangent;
// Parallax params: x=heightSlot, y=parallaxScale.
layout(location = 9) flat out vec4 vMatParams2;

void main() {
    mat4 model    = mat4(instModel0, instModel1, instModel2, instModel3);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    vWorldPos     = worldPos.xyz;

    vec4 viewPos = scene.view * worldPos;
    vViewPos     = viewPos;

    gl_Position  = scene.proj * viewPos;

    vColor      = inColor;
    vTexCoord   = inTexCoord;
    mat3 normalMat = mat3(instNormal0.xyz, instNormal1.xyz, instNormal2.xyz);
    vNormal     = normalMat * inNormal;
    // Tangent uses the normal matrix too. Strictly the tangent should use
    // mat3(model) since it's a tangent vector (not a covector like normal),
    // but for the rigid + uniform-scale transforms this engine handles, both
    // produce the same direction post-normalize. Stay consistent with normal
    // until non-uniform scale arrives.
    vTangent    = normalMat * inTangent;
    vColorTint  = instColorTint;
    vMatParams  = instMatParams;
    vMatParams2 = instMatParams2;

    // Previous-frame clip position via the per-instance prev world matrix,
    // then the un-jittered previous view-proj from the scene UBO. Pass the
    // full clip vec4; the fragment shader does the divide for proper per-pixel
    // motion accuracy.
    mat4 prevModel = mat4(instPrevModel0, instPrevModel1, instPrevModel2, instPrevModel3);
    vec4 prevWorld = prevModel * vec4(inPosition, 1.0);
    vPrevClipPos   = scene.prevViewProj * prevWorld;
}
