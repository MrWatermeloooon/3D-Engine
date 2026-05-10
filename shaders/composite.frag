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
} ubo;

vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdrTex, vUV).rgb;

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

    // SSAO multiplies result (mostly affects ambient/dark areas)
    if (ubo.enableSSAO == 1) {
        float ao = texture(ssaoTex, vUV).r;
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
