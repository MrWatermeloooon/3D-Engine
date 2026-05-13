#version 450

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 0, binding = 2) uniform sampler2D ssaoTex;

layout(set = 0, binding = 3) uniform CompositeUbo {
    float exposure;
    float bloomStrength;
    float ssaoStrength;
    float vignetteIntensity;
    float vignetteFalloff;
    float saturation;
    float contrast;
    float gamma;
    vec4  colorBalance;        // RGB lift, alpha = enable bloom
    int   tonemapMode;         // 0=ACES, 1=Reinhard, 2=Off
    int   enableSSAO;
    int   enableBloom;
    int   debugView;           // 0=final, 1=HDR, 2=Bloom, 3=SSAO, 4=Albedo, 5=Lit (no postfx)
    vec4  dofParams;           // x=enable, y=focusDist, z=focusRange, w=maxBokehPx
    vec4  cameraClip;          // x=near, y=far
} ubo;

layout(set = 0, binding = 4) uniform sampler2D depthTex;

vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Convert hardware depth (NDC z in [0,1] under GLM_FORCE_DEPTH_ZERO_TO_ONE) to
// linear view-space depth. Standard reverse-of-projection formula for the
// Vulkan zero-to-one convention with a perspective matrix.
float linearizeDepth(float ndcZ, float near, float far) {
    return (near * far) / (far - ndcZ * (far - near));
}

// 16-tap Poisson disc — cheap-enough single-pass DoF. Each tap is the HDR
// color sampled from a position scaled by per-pixel CoC. Results look soft
// rather than truly bokeh, but it responds correctly to focusDistance and
// scales with bokehRadius.
const vec2 POISSON_16[16] = vec2[16](
    vec2( 0.000,  0.000),
    vec2( 0.541,  0.121),
    vec2(-0.461,  0.198),
    vec2( 0.218, -0.523),
    vec2(-0.169, -0.617),
    vec2( 0.713, -0.314),
    vec2(-0.752, -0.301),
    vec2( 0.115,  0.733),
    vec2(-0.302,  0.683),
    vec2( 0.829,  0.412),
    vec2(-0.857,  0.367),
    vec2( 0.341, -0.881),
    vec2(-0.341, -0.881),
    vec2( 0.957,  0.078),
    vec2(-0.957,  0.078),
    vec2( 0.000, -0.945)
);

vec3 sampleDoF(vec3 centerHdr, vec2 uv) {
    if (ubo.dofParams.x < 0.5) return centerHdr;

    float ndcZ      = texture(depthTex, uv).r;
    float viewZ     = linearizeDepth(ndcZ, ubo.cameraClip.x, ubo.cameraClip.y);
    float focusDist = ubo.dofParams.y;
    float focusRange= ubo.dofParams.z;
    float maxRadius = ubo.dofParams.w;

    float coc = clamp(abs(viewZ - focusDist) / focusRange - 1.0, 0.0, 1.0) * maxRadius;
    if (coc < 0.5) return centerHdr;   // less than one pixel → skip the blur

    vec2 texel = 1.0 / vec2(textureSize(hdrTex, 0));
    vec3 acc   = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        vec2 offset = POISSON_16[i] * coc * texel;
        acc += texture(hdrTex, uv + offset).rgb;
    }
    return acc / 16.0;
}

void main() {
    vec3 hdr = texture(hdrTex, vUV).rgb;

    // DoF runs in HDR space, before bloom additive blend and tonemap. Skipped
    // when ubo.dofParams.x == 0.
    hdr = sampleDoF(hdr, vUV);

    // ── Debug views (bypass post-fx) ────────────────────────────────────
    if (ubo.debugView == 1) { outColor = vec4(hdr, 1.0); return; }
    if (ubo.debugView == 2) { outColor = vec4(texture(bloomTex, vUV).rgb, 1.0); return; }
    if (ubo.debugView == 3) { float a = texture(ssaoTex, vUV).r; outColor = vec4(a,a,a,1.0); return; }
    if (ubo.debugView == 5) {
        // Lit but no post-fx: tone map only
        vec3 ldr = acesFilm(hdr);
        ldr = pow(max(ldr, vec3(0.0)), vec3(1.0/2.2));
        outColor = vec4(ldr, 1.0);
        return;
    }

    if (ubo.enableBloom == 1) {
        vec3 bloom = texture(bloomTex, vUV).rgb;
        hdr += bloom * ubo.bloomStrength;
    }

    // Exposure
    hdr *= pow(2.0, ubo.exposure);

    // SSAO multiplies result (mostly affects ambient/dark areas).
    // The SSAO shader rotates its kernel per-pixel using a 4×4 random-noise
    // texture, which would otherwise produce a visible stippled / dithered
    // pattern in soft AO regions. We blur out that noise with a 4×4 box
    // filter exactly matching the noise tile size — every neighbourhood
    // averages one full rotation cycle, so the result is denoise-free at
    // negligible cost (16 bilinear texture taps).
    if (ubo.enableSSAO == 1) {
        vec2 texel = 1.0 / vec2(textureSize(ssaoTex, 0));
        float ao = 0.0;
        for (int y = -2; y <= 1; ++y) {
            for (int x = -2; x <= 1; ++x) {
                ao += texture(ssaoTex, vUV + vec2(x, y) * texel).r;
            }
        }
        ao /= 16.0;
        ao = mix(1.0, ao, ubo.ssaoStrength);
        hdr *= ao;
    }

    // Tone map
    vec3 ldr;
    if (ubo.tonemapMode == 0)      ldr = acesFilm(hdr);
    else if (ubo.tonemapMode == 1) ldr = hdr / (hdr + vec3(1.0));
    else                           ldr = clamp(hdr, 0.0, 1.0);

    // Color grading: lift, saturation, contrast
    ldr += ubo.colorBalance.rgb;
    float lum = dot(ldr, vec3(0.2126, 0.7152, 0.0722));
    ldr = mix(vec3(lum), ldr, ubo.saturation);
    ldr = (ldr - 0.5) * ubo.contrast + 0.5;

    // Vignette
    vec2 d = vUV - 0.5;
    float dist = dot(d, d);
    float vig = 1.0 - smoothstep(ubo.vignetteFalloff, ubo.vignetteFalloff + 0.5, dist) * ubo.vignetteIntensity;
    ldr *= vig;

    // Gamma adjustment relative to sRGB (which the swapchain image format applies automatically).
    // gamma = 2.2 → no change. <2.2 darker midtones, >2.2 brighter.
    ldr = pow(max(ldr, vec3(0.0)), vec3(2.2 / ubo.gamma));

    outColor = vec4(ldr, 1.0);
}
