#version 450

// Per-vertex
layout(location = 0) in vec3 inPosition;
// Per-instance (binding 1) — only model matrix is read in shadow pass
layout(location = 4) in vec4 instModel0;
layout(location = 5) in vec4 instModel1;
layout(location = 6) in vec4 instModel2;
layout(location = 7) in vec4 instModel3;

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj;
} push;

void main() {
    mat4 model = mat4(instModel0, instModel1, instModel2, instModel3);
    gl_Position = push.lightViewProj * model * vec4(inPosition, 1.0);
}
