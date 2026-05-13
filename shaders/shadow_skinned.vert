#version 450

// Depth-only skinned shadow vertex shader. Linear-blend skin the position with
// the bone palette, transform by the instance model, then project through the
// cascade light-view-proj. No varyings — shadow pass is depth-only.

const int MAX_BONES = 64;

layout(location = 0) in vec3  inPosition;
layout(location = 1) in vec3  inNormal;       // unused, kept so binding layout matches SkinnedVertex
layout(location = 2) in vec2  inTexCoord;     // unused
layout(location = 3) in uvec4 inJointIds;
layout(location = 4) in vec4  inWeights;

layout(set = 0, binding = 0) uniform BonePalette {
    mat4 bones[MAX_BONES];
} bp;

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj;
    mat4 model;
} push;

void main() {
    mat4 skin =
        bp.bones[inJointIds.x] * inWeights.x +
        bp.bones[inJointIds.y] * inWeights.y +
        bp.bones[inJointIds.z] * inWeights.z +
        bp.bones[inJointIds.w] * inWeights.w;

    vec4 worldPos = push.model * skin * vec4(inPosition, 1.0);
    gl_Position   = push.lightViewProj * worldPos;
}
