#version 450

// Fullscreen triangle drawn at far depth. The fragment shader reconstructs a
// world-space view direction from gl_FragCoord. We don't pass anything extra
// from the vertex stage — the inverse view-proj reconstruction in the
// fragment shader is cheap and avoids a varying.
//
// gl_Position.z = 1.0 → NDC depth = 1.0, which combined with the sky
// pipeline's depthCompareOp = LESS_OR_EQUAL means the sky only fills pixels
// that no geometry wrote to.

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

layout(location = 0) out vec3 vWorldDir;

void main() {
    // Standard fullscreen-triangle expansion: (-1,-1), (3,-1), (-1,3)
    vec2 pos = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                    (gl_VertexIndex == 2) ? 3.0 : -1.0);

    // Project the corner ray back into world space. Vulkan NDC: z=1 = far.
    vec4 clip   = vec4(pos, 1.0, 1.0);
    mat4 invVP  = inverse(scene.proj * scene.view);
    vec4 worldH = invVP * clip;
    vec3 world  = worldH.xyz / worldH.w;
    vWorldDir   = world - scene.cameraPos.xyz;

    gl_Position = vec4(pos, 1.0, 1.0);
}
