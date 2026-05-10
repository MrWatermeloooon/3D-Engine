#version 450

const int  KERNEL_SIZE = 16;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out float outOcclusion;

layout(set = 0, binding = 0) uniform sampler2D depthTex;
layout(set = 0, binding = 1) uniform sampler2D noiseTex;
layout(set = 0, binding = 2) uniform SSAOUbo {
    mat4 projection;
    mat4 invProjection;
    vec4 samples[KERNEL_SIZE];
    vec2 noiseScale;
    float radius;
    float bias;
    float intensity;
    float pad0, pad1, pad2;
} ubo;

vec3 viewPosFromDepth(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    // Vulkan Y in NDC is bottom-up vs UV top-down. We pass projection that includes Y-flip,
    // so the view-space reconstruction works directly.
    vec4 view = ubo.invProjection * clip;
    return view.xyz / view.w;
}

void main() {
    float depth = texture(depthTex, vUV).r;
    if (depth >= 1.0) { outOcclusion = 1.0; return; }

    vec3 fragPosVS = viewPosFromDepth(vUV, depth);

    // Reconstruct view-space normal from depth derivatives.
    // In Vulkan, dFdy points down in screen space; with the Y-flipped projection,
    // cross(dy, dx) gives the camera-facing normal (cross(dx, dy) would be inverted).
    vec3 dx = dFdx(fragPosVS);
    vec3 dy = dFdy(fragPosVS);
    vec3 normalVS = normalize(cross(dy, dx));

    vec3 randomVec = texture(noiseTex, vUV * ubo.noiseScale).xyz;
    vec3 tangent   = normalize(randomVec - normalVS * dot(randomVec, normalVS));
    vec3 bitangent = cross(normalVS, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normalVS);

    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        vec3 samplePos = TBN * ubo.samples[i].xyz;
        samplePos = fragPosVS + samplePos * ubo.radius;

        vec4 offset = ubo.projection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;

        if (offset.x < 0.0 || offset.x > 1.0 || offset.y < 0.0 || offset.y > 1.0) continue;

        float sampleDepth = texture(depthTex, offset.xy).r;
        vec3 sampledVS = viewPosFromDepth(offset.xy, sampleDepth);

        float rangeCheck = smoothstep(0.0, 1.0, ubo.radius / abs(fragPosVS.z - sampledVS.z));
        occlusion += (sampledVS.z >= samplePos.z + ubo.bias ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / float(KERNEL_SIZE));
    outOcclusion = pow(occlusion, ubo.intensity);
}
