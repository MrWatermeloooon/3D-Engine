#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} scene;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vTexCoord;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vWorldPos;
layout(location = 4) out vec3 vViewPos;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    vWorldPos    = worldPos.xyz;

    vec4 viewPos = scene.view * worldPos;
    vViewPos     = viewPos.xyz;

    gl_Position  = scene.proj * viewPos;

    vColor    = inColor;
    vTexCoord = inTexCoord;
    vNormal   = mat3(transpose(inverse(push.model))) * inNormal;
}
