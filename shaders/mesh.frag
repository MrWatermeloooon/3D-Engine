#version 460
#extension GL_EXT_nonuniform_qualifier                       : require
#extension GL_EXT_ray_query                                  : require
#extension GL_EXT_buffer_reference                           : require
#extension GL_EXT_scalar_block_layout                        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64     : require

const float PI = 3.14159265359;
const int   MAX_LIGHTS    = 32;
const int   CASCADE_COUNT = 4;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vTexCoord;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec3 vWorldPos;
layout(location = 4) in vec3 vViewPos;
layout(location = 5) flat in vec4 vColorTint;
layout(location = 6) flat in vec4 vMatParams;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 rtParams;   // x=shadows on, y=light angular radius, z=shadow samples, w=sunOnly
    vec4 rtParams2;  // x=reflections on, y=reflection samples, z=reflection tMax, w=intensity
    vec4 rtParams3;  // x=GI on,          y=GI samples,         z=GI tMax,         w=intensity
    mat4 prevViewProj;  // previous-frame view × proj for motion vectors
    vec4 jitterOffset;  // .xy = current frame camera-jitter in NDC units
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
layout(set = 0, binding = 4) uniform accelerationStructureEXT topLevelAS;

// Vertex layout mirrors C++ `Vertex` (vertex.h) with scalar layout to avoid
// vec3 std140 padding. position, normal, texCoord, color = 12+12+8+12 = 44 B.
struct VertexRT {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
    vec3 color;
};

layout(buffer_reference, scalar) readonly buffer Vertices { VertexRT v[]; };
layout(buffer_reference, scalar) readonly buffer Indices  { uint     i[]; };

struct RtMaterial {
    vec4  color;       // rgb = albedo
    vec4  params;      // x = metallic, y = roughness, zw reserved
    uvec2 vertexAddr;  // packUint2x32 → Vertices
    uvec2 indexAddr;   // packUint2x32 → Indices
};
layout(set = 0, binding = 5, std430) readonly buffer RtMaterialsBuffer {
    RtMaterial materials[];
} rtMats;

// Bindless texture array — material chooses its texture via an index in vMatParams.z.
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];

layout(location = 0) out vec4 outColor;
// Motion vector for DLSS. NDC-space (prev - curr), jitter removed.
// Computed from vWorldPos so it works for both static and skinned meshes
// without per-instance previous-transform plumbing — at the cost of object
// motion not being captured (only camera motion).
layout(location = 1) out vec2 outMotion;

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

// ── Ray-queried shadows ─────────────────────────────────────────────────────
// One ray per light per fragment. Hits are treated as opaque; we terminate on
// first hit because we only care about visibility. Soft penumbras come from
// jittering the ray direction inside a cone around the light vector, sized
// by `rtParams.y` (angular radius). For a single sample per pixel this is
// noisy — TAA or a tiny temporal blur cleans it up. With samples > 1 the
// loop converges to a clean soft shadow.

uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint state = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (state >> 22u) ^ state;
}

float rand01(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) / 4294967295.0;
}

// Build an orthonormal basis around `n` without trig.
void onb(vec3 n, out vec3 t, out vec3 b) {
    float sign_ = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + n.z);
    float bx = n.x * n.y * a;
    t = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * bx, -sign_ * n.x);
    b = vec3(bx, sign_ + n.y * n.y * a, -n.y);
}

vec3 sampleCone(vec3 axis, float cosThetaMax, inout uint seed) {
    float u1 = rand01(seed);
    float u2 = rand01(seed);
    float cosTheta = mix(cosThetaMax, 1.0, u1);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 6.2831853 * u2;
    vec3 t, b;
    onb(axis, t, b);
    return normalize(t * cos(phi) * sinTheta + b * sin(phi) * sinTheta + axis * cosTheta);
}

// Trace one shadow ray from `origin` toward `dir`. Returns 1.0 if no hit,
// 0.0 if anything blocks within [tMin, tMax].
float traceShadowRay(vec3 origin, vec3 dir, float tMin, float tMax) {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, tMin, dir, tMax);
    rayQueryProceedEXT(q);
    return rayQueryGetIntersectionTypeEXT(q, true)
        == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

// Soft shadow toward a directional or punctual light. For directional lights
// `tMax` should be very large (the sun is effectively at infinity). For
// punctual lights `tMax` is the distance to the light, so we don't accumulate
// false shadows from geometry beyond it.
float rtShadow(vec3 worldPos, vec3 N, vec3 toLightDir, float tMax, uint seedBase) {
    // Offset the origin along the normal to avoid self-shadowing on coarse
    // geometry (cube faces in particular were getting acne at 0.002).
    vec3 origin = worldPos + N * 0.01;
    float cosMax = cos(scene.rtParams.y);
    int samples = clamp(int(scene.rtParams.z), 1, 64);
    float acc = 0.0;
    uint seed = seedBase;
    for (int i = 0; i < samples; ++i) {
        vec3 d = sampleCone(toLightDir, cosMax, seed);
        acc += traceShadowRay(origin, d, 0.001, tMax);
    }
    return acc / float(samples);
}

// ── Ray-traced reflections ──────────────────────────────────────────────────
// One ray per sample along reflect(-V, N). For glossy materials we sample a
// cone whose width scales with roughness — mirror at roughness 0, fairly
// blurred at 1.0. Misses return a hemisphere sky color so we don't get pure
// black voids in reflections. Hits sample the per-instance material buffer
// (no surface-normal lookup yet — that wants vertex/index buffer access via
// device addresses, a separate step).

vec3 reflectionSkyColor(vec3 dir) {
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.18, 0.16, 0.14), vec3(0.45, 0.55, 0.70), t);
}

// Shade a reflection hit with real Lambert + sun + RT shadow ray + a small
// hemisphere ambient. Surface normal is interpolated from the hit primitive's
// three vertex normals via barycentrics and pushed to world space using the
// instance's object-to-world matrix.
vec3 shadeReflectionHit(rayQueryEXT q, vec3 rayOrigin, vec3 rayDir,
                        vec3 sunToLight, vec3 sunRadiance)
{
    int idx = rayQueryGetIntersectionInstanceCustomIndexEXT(q, true);
    RtMaterial mat = rtMats.materials[idx];

    Vertices vbuf = Vertices(packUint2x32(mat.vertexAddr));
    Indices  ibuf = Indices (packUint2x32(mat.indexAddr));

    uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, true));
    uint i0 = ibuf.i[prim * 3u + 0u];
    uint i1 = ibuf.i[prim * 3u + 1u];
    uint i2 = ibuf.i[prim * 3u + 2u];

    vec3 n0 = vbuf.v[i0].normal;
    vec3 n1 = vbuf.v[i1].normal;
    vec3 n2 = vbuf.v[i2].normal;

    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(q, true);
    vec3 b  = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);

    vec3 nObj = normalize(n0 * b.x + n1 * b.y + n2 * b.z);

    // Object→world transform from the ray query. transpose(inverse(M))
    // handles non-uniform scale (e.g. the floor's 15×0.1×15 box).
    mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(q, true);
    mat3 normalMat = transpose(inverse(mat3(o2w)));
    vec3 hitN = normalize(normalMat * nObj);

    float t = rayQueryGetIntersectionTEXT(q, true);
    vec3 hitP = rayOrigin + rayDir * t;

    // Sun shadow at the hit (single ray — soft shadow would be expensive
    // inside an already-secondary trace).
    float sunShadow = traceShadowRay(hitP + hitN * 0.01, sunToLight, 0.001, 1.0e4);

    float NdotL = max(dot(hitN, sunToLight), 0.0);

    // Small hemisphere ambient so back-faces aren't pitch black.
    float hemi = clamp(hitN.y * 0.5 + 0.5, 0.0, 1.0);
    vec3  hemiAmbient = mix(vec3(0.18, 0.16, 0.14), vec3(0.45, 0.55, 0.70), hemi);

    vec3 albedo = mat.color.rgb;
    return albedo * (NdotL * sunShadow * sunRadiance + hemiAmbient * 0.25);
}

// Cosine-weighted hemisphere sample around N. PDF = NdotL / PI, so the
// Monte-Carlo estimator for Lambertian indirect simplifies to
// `albedo * mean(radiance)` — no extra cosine or PI factor needed in main().
vec3 sampleCosineHemisphere(vec3 N, inout uint seed) {
    float u1 = rand01(seed);
    float u2 = rand01(seed);
    float r   = sqrt(u1);
    float phi = 6.2831853 * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1));
    vec3 t, b;
    onb(N, t, b);
    return normalize(t * x + b * y + N * z);
}

// One-bounce global illumination. Cosine-sample the hemisphere around the
// surface normal, trace a ray, and return the average incoming radiance.
// `shadeReflectionHit` is reused for the hit shading (Lambert + sun shadow
// ray + small ambient) — that's exactly what we want as "indirect light".
vec3 rtGI(vec3 worldPos, vec3 N, vec3 sunToLight, vec3 sunRadiance,
          inout uint seed)
{
    int samples = clamp(int(scene.rtParams3.y), 1, 16);
    float tMax = scene.rtParams3.z;
    vec3 origin = worldPos + N * 0.01;

    vec3 acc = vec3(0.0);
    for (int i = 0; i < samples; ++i) {
        vec3 d = sampleCosineHemisphere(N, seed);
        rayQueryEXT q;
        rayQueryInitializeEXT(q, topLevelAS, gl_RayFlagsOpaqueEXT,
                              0xFF, origin, 0.001, d, tMax);
        rayQueryProceedEXT(q);
        vec3 sampleRadiance;
        if (rayQueryGetIntersectionTypeEXT(q, true)
            == gl_RayQueryCommittedIntersectionNoneEXT)
        {
            sampleRadiance = reflectionSkyColor(d);
        } else {
            sampleRadiance = shadeReflectionHit(q, origin, d, sunToLight, sunRadiance);
        }
        // Firefly clamp: any single sample's contribution is capped so that
        // a few outlier rays (e.g. one that hits the sunlit top of a cube)
        // can't dominate the average. This is the cheapest "denoise" trick.
        acc += min(sampleRadiance, vec3(1.0));
    }
    return acc / float(samples);
}

vec3 rtReflection(vec3 worldPos, vec3 N, vec3 V, float roughness,
                  vec3 sunToLight, vec3 sunRadiance, inout uint seed)
{
    vec3 R = reflect(-V, N);
    int samples = clamp(int(scene.rtParams2.y), 1, 16);
    // Tight cone — r² × 0.3 gives ~1.1° at r=0.25 (steel),
    // ~2.75° at r=0.4 (plastic), ~17° at r=1.0. Mirror at r=0.
    float halfAngle = roughness * roughness * 0.3;
    float cosMax = cos(halfAngle);
    float tMax = scene.rtParams2.z;
    vec3 origin = worldPos + N * 0.01;

    vec3 acc = vec3(0.0);
    for (int i = 0; i < samples; ++i) {
        vec3 d = (cosMax > 0.999) ? R : sampleCone(R, cosMax, seed);
        rayQueryEXT q;
        rayQueryInitializeEXT(q, topLevelAS, gl_RayFlagsOpaqueEXT,
                              0xFF, origin, 0.001, d, tMax);
        rayQueryProceedEXT(q);
        if (rayQueryGetIntersectionTypeEXT(q, true)
            == gl_RayQueryCommittedIntersectionNoneEXT)
        {
            acc += reflectionSkyColor(d);
        } else {
            acc += shadeReflectionHit(q, origin, d, sunToLight, sunRadiance);
        }
    }
    return acc / float(samples);
}

// ── CSM shadow sampling ─────────────────────────────────────────────────────

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
    int textureIndex = int(vMatParams.z);
    vec3 albedoSample = texture(bindlessTextures[nonuniformEXT(textureIndex)], vTexCoord).rgb;
    vec3 albedo       = albedoSample * vColor * vColorTint.rgb;
    float metallic    = vMatParams.x;
    float roughness   = clamp(vMatParams.y, 0.04, 1.0);

    vec3 N = normalize(vNormal);
    vec3 V = normalize(scene.cameraPos.xyz - vWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    bool useRT      = scene.rtParams.x > 0.5;
    bool rtSunOnly  = scene.rtParams.w > 0.5;

    // Per-fragment seed for RT cone sampling — derived from screen position so
    // adjacent pixels get different ray directions.
    uint seed = uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + 1u;

    vec3 Lo = vec3(0.0);
    vec3 directionalLo = vec3(0.0);
    vec3 firstDirLightDir = vec3(0.0, -1.0, 0.0);
    bool firstDirLightFound = false;

    // RT reflection's hit shading needs the sun direction + radiance.
    // Captured during the lighting loop on the first directional encounter.
    vec3 sunToLight  = vec3(0.0, 1.0, 0.0);
    vec3 sunRadiance = vec3(0.0);

    for (int i = 0; i < lightsUbo.numLights; ++i) {
        vec3 outDir;
        vec3 contrib = evaluateLight(lightsUbo.lights[i], N, V, vWorldPos, albedo, F0,
                                     metallic, roughness, outDir);
        bool isDirectional = lightsUbo.lights[i].positionAndType.w < 0.5;

        if (isDirectional && !firstDirLightFound) {
            sunToLight  = -outDir;
            sunRadiance = lightsUbo.lights[i].colorAndIntensity.rgb
                        * lightsUbo.lights[i].colorAndIntensity.w;
        }

        if (useRT) {
            // All lights get an RT shadow when ray queries are enabled. When
            // `sunOnly` is on, restrict to the first directional light (parity
            // with CSM) — every other light contributes unshadowed.
            bool shadowThisLight =
                !rtSunOnly || (isDirectional && !firstDirLightFound);
            if (isDirectional) firstDirLightFound = true;

            if (shadowThisLight) {
                vec3 toLight = -outDir;
                float tMax;
                if (isDirectional) {
                    tMax = 1.0e4;
                } else {
                    tMax = length(lightsUbo.lights[i].positionAndType.xyz - vWorldPos);
                }
                float shadow = rtShadow(vWorldPos, N, toLight, tMax,
                                        seed + uint(i) * 0x9E3779B9u);
                Lo += contrib * shadow;
            } else {
                Lo += contrib;
            }
        } else {
            // Legacy CSM path — only the first directional light is shadowed.
            if (isDirectional && !firstDirLightFound) {
                directionalLo      = contrib;
                firstDirLightDir   = outDir;
                firstDirLightFound = true;
            } else {
                Lo += contrib;
            }
        }
    }

    if (!useRT && firstDirLightFound) {
        int cascade = selectCascade(vViewPos.z);
        float shadowFactor = samplePCF(vWorldPos, N, firstDirLightDir, cascade);
        Lo += directionalLo * shadowFactor;
    }

    // ── Indirect lighting (energy-conserving split) ────────────────────────
    // Dielectric diffuse + metallic specular don't overlap. Without this
    // split, metals double-up (ambient + reflection) and look washed out;
    // dielectrics double-up too (ambient + tiny grazing reflection).
    vec3 F_indirect = fresnelSchlick(max(dot(N, V), 0.0), F0);

    // Hemisphere diffuse: only applies to the diffuse portion. Metals get
    // none of this — their indirect lighting is purely the RT reflection.
    // When RT GI is enabled, sample the real scene radiance via ray queries
    // instead of the static hemisphere — gives color bleed + true ambient
    // occlusion as a side effect.
    vec3 skyColor      = vec3(0.45, 0.55, 0.70);
    vec3 groundColor   = vec3(0.18, 0.16, 0.14);
    float hemi         = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 hemiAmbient   = mix(groundColor, skyColor, hemi);
    vec3 kD_indirect   = (1.0 - F_indirect) * (1.0 - metallic);

    // Static hemisphere ambient — smooth, noise-free fallback.
    vec3 staticAmbient = kD_indirect * albedo * hemiAmbient * 0.18;

    vec3 diffuseIndirect = staticAmbient;
    if (scene.rtParams3.x > 0.5 && metallic < 0.5) {
        vec3 giRadiance = rtGI(vWorldPos, N, sunToLight, sunRadiance, seed);
        // RT GI contribution scaled to roughly match the static ambient
        // magnitude so the mix doesn't drastically change overall brightness.
        vec3 giAmbient = kD_indirect * albedo * giRadiance * 0.5;
        // rtParams3.w acts as the GI/ambient mix factor:
        //   0 → pure smooth ambient (no GI flavor, no noise)
        //   1 → pure ray-traced GI (full color bleed, full noise)
        // Default 0.3 keeps the look mostly smooth with subtle color bleed.
        diffuseIndirect = mix(staticAmbient, giAmbient, scene.rtParams3.w);
    }

    // Specular indirect via RT. Skipped for very rough surfaces (would be
    // pure noise). Reflection intensity multiplier in rtParams2.w lets the
    // user dial reflections up/down for taste.
    vec3 specIndirect = vec3(0.0);
    // RT reflections only on metallic surfaces. Real dielectric plastic has
    // weak grazing-angle reflections, but for this engine's aesthetic we
    // gate them out — keeps plastics fully diffuse and reflection-noise free.
    if (scene.rtParams2.x > 0.5 && roughness < 0.85 && metallic > 0.5) {
        vec3 reflColor = rtReflection(vWorldPos, N, V, roughness,
                                      sunToLight, sunRadiance, seed);
        specIndirect = F_indirect * reflColor * scene.rtParams2.w;
    } else if (metallic > 0.5) {
        // Without RT reflection a metal would have zero indirect light.
        // Fall back to a flat F_indirect * hemiSky so metals don't go black
        // when reflections are disabled.
        specIndirect = F_indirect * hemiAmbient * 0.5;
    }

    vec3 color = diffuseIndirect + specIndirect + Lo;

    // Output linear HDR — composite pass handles tone mapping + gamma
    outColor = vec4(color, 1.0);

    // ── Motion vector ────────────────────────────────────────────────────
    // Reproject this fragment through both the current and previous-frame
    // view-projection. Output (prevNDC - currNDC) in NDC units. Subtract
    // the camera jitter from the current sample so the motion only encodes
    // real motion, not sub-pixel jitter.
    vec4 currClip = scene.proj * scene.view * vec4(vWorldPos, 1.0);
    vec4 prevClip = scene.prevViewProj         * vec4(vWorldPos, 1.0);
    vec2 currNDC = currClip.xy / currClip.w - scene.jitterOffset.xy;
    vec2 prevNDC = prevClip.xy / prevClip.w;
    outMotion = prevNDC - currNDC;
}
