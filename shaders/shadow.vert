#version 450

layout(location = 0) in vec3 inPosition;
// Other attributes ignored — we only need position

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj; // offset 0
    mat4 model;         // offset 64
} push;

void main() {
    gl_Position = push.lightViewProj * push.model * vec4(inPosition, 1.0);
}
