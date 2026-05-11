#version 450

// Per-vertex
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;
// Per-instance (binding 1)
layout(location = 4) in vec4 instModel0;
layout(location = 5) in vec4 instModel1;
layout(location = 6) in vec4 instModel2;
layout(location = 7) in vec4 instModel3;
layout(location = 8) in vec4 instColorTint;
layout(location = 9) in vec4 instMatParams; // x=metallic, y=roughness

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} scene;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vTexCoord;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vWorldPos;
layout(location = 4) out vec3 vViewPos;
layout(location = 5) flat out vec4 vColorTint;
layout(location = 6) flat out vec4 vMatParams;

void main() {
    mat4 model    = mat4(instModel0, instModel1, instModel2, instModel3);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    vWorldPos     = worldPos.xyz;

    vec4 viewPos = scene.view * worldPos;
    vViewPos     = viewPos.xyz;

    gl_Position  = scene.proj * viewPos;

    vColor      = inColor;
    vTexCoord   = inTexCoord;
    vNormal     = mat3(transpose(inverse(model))) * inNormal;
    vColorTint  = instColorTint;
    vMatParams  = instMatParams;
}
