#version 450

const int MAX_BONES = 64;

// Per-vertex (skinned format)
layout(location = 0) in vec3  inPosition;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inTexCoord;
layout(location = 3) in uvec4 inJointIds;
layout(location = 4) in vec4  inWeights;

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

layout(set = 2, binding = 0) uniform BonePalette {
    mat4 bones[MAX_BONES];
} bp;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 colorTint;
    vec4 matParams;   // x=metallic, y=roughness
} push;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vTexCoord;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vWorldPos;
layout(location = 4) out vec3 vViewPos;
layout(location = 5) flat out vec4 vColorTint;
layout(location = 6) flat out vec4 vMatParams;

void main() {
    // Linear-blend skinning: blend up to 4 bone transforms by weights
    mat4 skin =
        bp.bones[inJointIds.x] * inWeights.x +
        bp.bones[inJointIds.y] * inWeights.y +
        bp.bones[inJointIds.z] * inWeights.z +
        bp.bones[inJointIds.w] * inWeights.w;

    vec4 skinnedLocal = skin * vec4(inPosition, 1.0);
    vec3 skinnedNorm  = mat3(skin) * inNormal;

    vec4 worldPos = push.model * skinnedLocal;
    vWorldPos    = worldPos.xyz;

    vec4 viewPos = scene.view * worldPos;
    vViewPos     = viewPos.xyz;
    gl_Position  = scene.proj * viewPos;

    vColor      = vec3(1.0);
    vTexCoord   = inTexCoord;
    vNormal     = mat3(transpose(inverse(push.model))) * skinnedNorm;
    vColorTint  = push.colorTint;
    vMatParams  = push.matParams;
}
