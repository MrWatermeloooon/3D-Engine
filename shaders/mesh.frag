#version 460
#extension GL_EXT_nonuniform_qualifier                       : require
#extension GL_EXT_ray_query                                  : require
#extension GL_EXT_buffer_reference                           : require
#extension GL_EXT_scalar_block_layout                        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64     : require
// Subgroup quad ops let the 4 fragments of a 2x2 pixel quad share data with
// each other. We use them to spatially average the noisy RT-GI signal —
// effectively quadrupling its sample count for free, since each fragment in
// the quad has already done its own hemisphere sampling with a different
// seed. Standard Vulkan 1.1+; supported on every modern desktop GPU.
#extension GL_KHR_shader_subgroup_quad                       : require

const float PI = 3.14159265359;
const int   MAX_LIGHTS    = 32;
const int   CASCADE_COUNT = 4;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vTexCoord;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec3 vWorldPos;
layout(location = 4) in vec4 vViewPos;
layout(location = 5) flat in vec4 vColorTint;
layout(location = 6) flat in vec4 vMatParams;
layout(location = 7) in vec4 vPrevClipPos;
layout(location = 8) in vec3 vTangent;
layout(location = 9) flat in vec4 vMatParams2;  // x=heightSlot, y=parallaxScale

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
// IBL: irradiance for diffuse, prefilter for specular, BRDF LUT for split-sum.
layout(set = 0, binding = 6) uniform samplerCube irradianceMap;
layout(set = 0, binding = 7) uniform samplerCube prefilterMap;
layout(set = 0, binding = 8) uniform sampler2D   brdfLUT;

// Vertex layout mirrors C++ `Vertex` (vertex.h) with scalar layout to avoid
// vec3 std140 padding. position, normal, texCoord, color, tangent =
// 12+12+8+12+12 = 56 B. Must match C++ Vertex exactly — the buffer-reference
// walks this stride.
struct VertexRT {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
    vec3 color;
    vec3 tangent;
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
// Motion vector for future TAA/upscalers. NDC-space (prev - curr), jitter removed.
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
    // Clamp angular radius to [0, π/2]. Values past π/2 make sampleCone
    // return directions pointing back into the surface, which produces
    // false self-shadowing on every fragment.
    float angularRadius = clamp(scene.rtParams.y, 0.0, 1.5707);
    float cosMax = cos(angularRadius);
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
    // Degenerate BLAS construction can occasionally produce committed hits with
    // negative T (geometry behind the ray origin). Bias the secondary ray's
    // origin past that, but reject the shaded value entirely if t is non-positive.
    if (t <= 0.0001) return vec3(0.0);
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
    // Cap bumped from 16 to 64 so the "GI samples" slider can actually buy
    // less noise. Variance drops as 1/sqrt(N), so 64 samples cuts grain
    // ~halfway between 16-sample-noisy and pure-smooth.
    int samples = clamp(int(scene.rtParams3.y), 1, 64);
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

// ── Parallax-occlusion mapping ──────────────────────────────────────────────
// Tangent-space step march along the view ray. Returns the UV at which the
// view ray first dips below the height field. `Vt` is the tangent-space view
// direction (camera-toward-fragment, in TBN coords). Adaptive step count
// (more steps at grazing angles, fewer at near-normal incidence) keeps the
// average tap cost low while preventing layer-stepping artifacts at the
// silhouette.
//
// Uses the convention "height 1.0 = fully popped out toward the camera": the
// rendered surface is at depth 0 (top of layer stack) and the texture's
// height map sample is *subtracted* from a virtual layer depth that runs
// 0→1 along the view ray. We sample (1 - height) so a brighter texel reads
// as "higher" and pops toward the camera.
vec2 parallaxOffsetUV(vec2 uv, vec3 Vt, int heightSlot, float depthScale) {
    if (heightSlot <= 0 || depthScale <= 0.0) return uv;

    // Vt.z is dot(N, V) in tangent space — flatten the cone when looking
    // straight down to avoid divide-by-zero. Step count scales with grazing.
    float nDotV = max(Vt.z, 0.05);
    int   steps = int(mix(32.0, 8.0, nDotV));

    float layerDepth   = 1.0 / float(steps);
    float currentDepth = 0.0;
    // Parallax shifts UV opposite the view direction (we look "down" into the
    // layer stack). xy of Vt is the in-plane component; divide by z for the
    // perspective-correct shift, then scale by depth.
    vec2  deltaUV = (Vt.xy / nDotV) * depthScale * layerDepth;
    vec2  curUV   = uv;

    float h = 1.0 - texture(bindlessTextures[nonuniformEXT(heightSlot)], curUV).r;
    // March until the ray drops below the surface.
    for (int i = 0; i < 64; ++i) {
        if (currentDepth >= h) break;
        curUV         -= deltaUV;
        currentDepth  += layerDepth;
        h = 1.0 - texture(bindlessTextures[nonuniformEXT(heightSlot)], curUV).r;
        if (i >= steps) break;
    }

    // Binary-search refinement: one bisection between the last two layers
    // resolves the silhouette without needing a much higher base step count.
    vec2  prevUV    = curUV + deltaUV;
    float afterH    = h - currentDepth;
    float beforeH   = (1.0 - texture(bindlessTextures[nonuniformEXT(heightSlot)], prevUV).r)
                      - (currentDepth - layerDepth);
    float w = afterH / max(afterH - beforeH, 1e-5);
    return mix(curUV, prevUV, w);
}

void main() {
    int   heightSlot    = int(vMatParams2.x);
    float parallaxScale = vMatParams2.y;

    // Parallax happens BEFORE any UV-derived sample (albedo, normal). The
    // view direction needs to be in tangent space, which requires a TBN
    // built once here and reused below for normal mapping.
    vec3 geomN  = normalize(vNormal);
    vec3 geomT  = normalize(vTangent - dot(vTangent, geomN) * geomN);
    vec3 geomB  = cross(geomN, geomT);
    mat3 TBN    = mat3(geomT, geomB, geomN);

    vec3 Vworld = normalize(scene.cameraPos.xyz - vWorldPos);
    vec3 Vtan   = transpose(TBN) * Vworld;
    vec2 uv     = parallaxOffsetUV(vTexCoord, Vtan, heightSlot, parallaxScale);

    int textureIndex = int(vMatParams.z);
    vec3 albedoSample = texture(bindlessTextures[nonuniformEXT(textureIndex)], uv).rgb;
    vec3 albedo       = albedoSample * vColor * vColorTint.rgb;
    float metallic    = vMatParams.x;
    float roughness   = clamp(vMatParams.y, 0.04, 1.0);

    vec3 N = geomN;

    // ── Tangent-space normal mapping ────────────────────────────────────
    // matParams.w carries the bindless texture slot for the per-material
    // normal map. Slot 0 is reserved as the default white texture in the
    // resource manager — we treat "0" as "no normal map" and skip the
    // mapping. Uses the parallax-offset UV so normals follow the displaced
    // surface.
    int normalMapIndex = int(vMatParams.w);
    if (normalMapIndex > 0) {
        vec3 nmSample = texture(bindlessTextures[nonuniformEXT(normalMapIndex)], uv).xyz;
        vec3 nm  = normalize(nmSample * 2.0 - 1.0);
        N = normalize(TBN * nm);
    }

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

    // ── Indirect lighting (split-sum IBL) ──────────────────────────────────
    // Energy-conserving: dielectric diffuse and metallic specular don't
    // overlap. Diffuse weight is kD = (1 - F) * (1 - metallic); specular
    // weight is F. Roughness-aware Fresnel keeps grazing-angle highlights
    // sane on rough materials.
    //
    // IBL_EXPOSURE scales the IBL contribution to surface shading
    // independently of how bright the env cubemap is for sky display.
    // Without this split, choosing a sky brightness either over-lights
    // surfaces or leaves the sky too dim — they're separate concerns. 0.18
    // matches the magnitude of the engine's legacy `hemiAmbient * 0.18`
    // static-ambient term, so swapping IBL on/off doesn't change overall
    // scene brightness materially.
    const float IBL_EXPOSURE = 0.18;

    float NdotV = max(dot(N, V), 0.0);
    vec3  F_indirect = F0 + (max(vec3(1.0 - roughness), F0) - F0)
                           * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    vec3  kD_indirect = (1.0 - F_indirect) * (1.0 - metallic);

    // Diffuse: integrated environment radiance over the upper hemisphere.
    vec3 irradiance     = texture(irradianceMap, N).rgb;
    vec3 diffuseIndirect = kD_indirect * albedo * irradiance * IBL_EXPOSURE;

    // Mix RT GI in when enabled. The ray-traced radiance is single-frame
    // Monte Carlo. Two layers of noise reduction:
    //   * 0.5 attenuation on the giAmbient term so the contribution at any
    //     given rtParams3.w mix factor is half — matches the prior tuning.
    //   * subgroup quad-average shares each fragment's GI estimate with its
    //     3 neighbours in the 2x2 pixel quad. Each neighbour ran a different
    //     hemisphere sample set (the seed is per-pixel), so this effectively
    //     quadruples the sample budget at zero ray cost. Slight 2x2 spatial
    //     blur — fine for low-frequency GI, occasional artefacts on triangle
    //     edges where the quad straddles two primitives.
    //
    // The subgroupQuadSwap* ops require all 4 lanes of the 2x2 quad to be
    // active — quad ops read undefined data from inactive lanes. `metallic`
    // is per-fragment, so it must NOT gate the swap (a quad straddling a
    // metal/dielectric edge would diverge and corrupt the average). Only the
    // scene-uniform rtParams3.x gates the GI compute + reduction; the
    // per-fragment metallic test only decides whether to APPLY the result.
    if (scene.rtParams3.x > 0.5) {
        vec3 giRadiance = rtGI(vWorldPos, N, sunToLight, sunRadiance, seed);
        vec3 quadSum  = giRadiance;
        quadSum += subgroupQuadSwapHorizontal(giRadiance);
        quadSum += subgroupQuadSwapVertical(giRadiance);
        quadSum += subgroupQuadSwapDiagonal(giRadiance);
        giRadiance = quadSum * 0.25;

        if (metallic < 0.5) {
            vec3 giAmbient  = kD_indirect * albedo * giRadiance * 0.5;
            diffuseIndirect = mix(diffuseIndirect, giAmbient, scene.rtParams3.w);
        }
    }

    // Specular: split-sum approximation. textureLod into the prefiltered
    // cube at a mip determined by roughness; multiply by F*scale + bias from
    // the BRDF LUT. Matches Epic's UE4 IBL paper.
    vec3  R           = reflect(-V, N);
    float maxReflLod  = float(textureQueryLevels(prefilterMap)) - 1.0;
    vec3  prefiltered = textureLod(prefilterMap, R, roughness * maxReflLod).rgb;
    vec2  envBRDF     = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3  iblSpecular = prefiltered * (F_indirect * envBRDF.x + envBRDF.y) * IBL_EXPOSURE;

    // RT reflections override IBL specular for metallic surfaces when on —
    // they trace true scene reflections that the prefilter can't capture.
    if (scene.rtParams2.x > 0.5 && roughness < 0.85 && metallic > 0.5) {
        vec3 reflColor = rtReflection(vWorldPos, N, V, roughness,
                                      sunToLight, sunRadiance, seed);
        iblSpecular = F_indirect * reflColor * scene.rtParams2.w;
    }

    vec3 color = diffuseIndirect + iblSpecular + Lo;

    // Output linear HDR — composite pass handles tone mapping + gamma
    outColor = vec4(color, 1.0);

    // ── Motion vector ────────────────────────────────────────────────────
    // Per-OBJECT motion: prevClipPos comes from the per-instance prevModel
    // (per-vertex prev world position × scene.prevViewProj, interpolated
    // here). currClip is the current world position through the current
    // view-proj. Output (prev - curr) in NDC units.
    //
    // Jitter handling: the camera proj has `+jitterNDC` added to proj[2][0/1];
    // post-W-divide that yields a NDC shift of `-jitterNDC`. To recover the
    // unjittered current NDC, add jitterOffset back. scene.prevViewProj is
    // the UN-jittered prev view-proj (set engine-side from raw camera
    // matrices), so prevNDC needs no correction.
    vec4 currClip = scene.proj * scene.view * vec4(vWorldPos, 1.0);
    vec2 currNDC = currClip.xy / currClip.w + scene.jitterOffset.xy;
    vec2 prevNDC = vPrevClipPos.xy / vPrevClipPos.w;
    outMotion = prevNDC - currNDC;
}
