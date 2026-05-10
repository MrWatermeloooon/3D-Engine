#version 450

const float PI = 3.14159265359;
const int   MAX_LIGHTS    = 32;
const int   CASCADE_COUNT = 4;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vTexCoord;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec3 vWorldPos;
layout(location = 4) in vec3 vViewPos;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} scene;

struct Light {
    vec4 positionAndType;
    vec4 directionAndRange;
    vec4 colorAndIntensity;
    vec4 spotCones;
};

layout(set = 0, binding = 1) uniform LightsUBO {
    int   numLights;
    int   pad0;
    int   pad1;
    int   pad2;
    Light lights[MAX_LIGHTS];
} lightsUbo;

layout(set = 0, binding = 2) uniform CascadeUBO {
    mat4 lightViewProj[CASCADE_COUNT];
    vec4 splitsViewSpace;
} cascades;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadowMap;
layout(set = 1, binding = 0) uniform sampler2D albedoTex;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) vec4  colorTint;
    layout(offset = 80) float metallic;
    layout(offset = 84) float roughness;
} push;

layout(location = 0) out vec4 outColor;

// ── PBR (Cook-Torrance) ─────────────────────────────────────────────────────

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Shadow sampling ─────────────────────────────────────────────────────────

int selectCascade(float viewZ) {
    int cascade = CASCADE_COUNT - 1;
    for (int i = 0; i < CASCADE_COUNT; ++i) {
        if (viewZ > cascades.splitsViewSpace[i]) {
            cascade = i;
            break;
        }
    }
    return cascade;
}

// 5x5 PCF with normal offset to reduce acne. Uses a per-cascade bias scale
// because farther cascades cover more world-space area per shadow texel.
float samplePCF(vec3 worldPos, vec3 N, vec3 lightDir, int cascade) {
    // Normal-offset bias: shift the world position along its normal by a few texels
    // worth of distance, scaled by (1 - NdotL) so steep angles get more offset.
    float NdotL = max(dot(N, -lightDir), 0.0);
    float texelWorld = 2.0 / float(textureSize(shadowMap, 0).x);
    vec3 offsetWorld = N * texelWorld * (1.0 - NdotL) * (1.0 + float(cascade) * 0.5) * 1.5;

    vec4 shadowClip = cascades.lightViewProj[cascade] * vec4(worldPos + offsetWorld, 1.0);
    vec3 shadowNDC  = shadowClip.xyz / shadowClip.w;
    vec2 uv = shadowNDC.xy * 0.5 + 0.5;
    float depth = shadowNDC.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0) return 1.0;

    // Constant bias scaled per cascade
    float bias = 0.0008 * (1.0 + float(cascade));

    // 5x5 PCF (25 taps via hardware comparison sampler — tap cost is low on modern GPUs)
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float shadow = 0.0;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(x, y) * texelSize;
            shadow += texture(shadowMap, vec4(uv + offset, float(cascade), depth - bias));
        }
    }
    return shadow / 25.0;
}

// ── Per-light contribution ──────────────────────────────────────────────────

vec3 evaluateLight(Light L, vec3 N, vec3 V, vec3 worldPos, vec3 albedo, vec3 F0,
                   float metallic, float roughness, out vec3 lightDirOut)
{
    vec3 Lvec;
    float intensity = L.colorAndIntensity.w;
    float attenuation = 1.0;

    if (L.positionAndType.w < 0.5) {
        // Directional
        Lvec = -normalize(L.directionAndRange.xyz);
    } else if (L.positionAndType.w < 1.5) {
        // Point
        vec3 toLight = L.positionAndType.xyz - worldPos;
        float dist = length(toLight);
        Lvec = toLight / max(dist, 0.0001);
        float r = L.directionAndRange.w;
        float falloff = clamp(1.0 - pow(dist / max(r, 0.0001), 4.0), 0.0, 1.0);
        attenuation = (falloff * falloff) / (dist * dist + 1.0);
    } else {
        // Spot
        vec3 toLight = L.positionAndType.xyz - worldPos;
        float dist = length(toLight);
        Lvec = toLight / max(dist, 0.0001);
        vec3 spotDir = normalize(L.directionAndRange.xyz);
        float r = L.directionAndRange.w;
        float falloff = clamp(1.0 - pow(dist / max(r, 0.0001), 4.0), 0.0, 1.0);
        attenuation = (falloff * falloff) / (dist * dist + 1.0);

        float cosTheta = dot(spotDir, -Lvec);
        float coneAtt  = clamp((cosTheta - L.spotCones.y) / max(L.spotCones.x - L.spotCones.y, 1e-4), 0.0, 1.0);
        attenuation *= coneAtt;
    }

    lightDirOut = -Lvec;
    vec3 H = normalize(V + Lvec);

    float NdotL = max(dot(N, Lvec), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, Lvec, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 numerator   = D * G * F;
    float denom      = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular    = numerator / denom;

    vec3 radiance = L.colorAndIntensity.xyz * intensity * attenuation;
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    vec3 albedoSample = texture(albedoTex, vTexCoord).rgb;
    vec3 albedo       = albedoSample * vColor * push.colorTint.rgb;
    float metallic    = push.metallic;
    float roughness   = clamp(push.roughness, 0.04, 1.0);

    vec3 N = normalize(vNormal);
    vec3 V = normalize(scene.cameraPos.xyz - vWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Separate the first directional light's contribution so we can shadow it
    // independently of the other lights (point/spot don't cast shadows here).
    vec3 directionalLo = vec3(0.0);
    vec3 otherLo       = vec3(0.0);
    vec3 firstDirLightDir = vec3(0.0, -1.0, 0.0);
    bool firstDirLightFound = false;

    for (int i = 0; i < lightsUbo.numLights; ++i) {
        vec3 outDir;
        vec3 contrib = evaluateLight(lightsUbo.lights[i], N, V, vWorldPos, albedo, F0,
                                     metallic, roughness, outDir);
        bool isDirectional = lightsUbo.lights[i].positionAndType.w < 0.5;
        if (isDirectional && !firstDirLightFound) {
            directionalLo      = contrib;
            firstDirLightDir   = outDir;
            firstDirLightFound = true;
        } else {
            otherLo += contrib;
        }
    }

    // Apply shadow only to the (first) directional light
    vec3 Lo = otherLo;
    if (firstDirLightFound) {
        int cascade = selectCascade(vViewPos.z);
        float shadowFactor = samplePCF(vWorldPos, N, firstDirLightDir, cascade);
        Lo += directionalLo * shadowFactor;
    }

    // Hemisphere ambient — sky tint above, ground tint below, blended by N.y.
    // Simple but a big quality win over a flat constant ambient.
    vec3 skyColor    = vec3(0.45, 0.55, 0.70);
    vec3 groundColor = vec3(0.18, 0.16, 0.14);
    float hemi       = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient     = mix(groundColor, skyColor, hemi) * albedo * 0.18;

    vec3 color = ambient + Lo;

    // Output linear HDR — composite pass handles tone mapping + gamma
    outColor = vec4(color, 1.0);
}
