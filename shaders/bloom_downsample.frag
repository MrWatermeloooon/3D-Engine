#version 450

// Karis-weighted 13-tap downsample (CoD: AW)
layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D srcTex;

layout(push_constant) uniform PC {
    vec2  texelSize;
    float threshold;     // > 0 enables prefilter (only for first pass)
    float pad;
} pc;

vec3 prefilter(vec3 c) {
    if (pc.threshold <= 0.0) return c;
    float brightness = max(c.r, max(c.g, c.b));
    float knee = max(0.001, brightness - pc.threshold);
    return c * (knee / max(brightness, 1e-4));
}

float karis(vec3 c) {
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + lum);
}

void main() {
    vec2 t = pc.texelSize;
    vec3 a = texture(srcTex, vUV + t * vec2(-1.0,  1.0)).rgb;
    vec3 b = texture(srcTex, vUV + t * vec2( 0.0,  1.0)).rgb;
    vec3 c = texture(srcTex, vUV + t * vec2( 1.0,  1.0)).rgb;
    vec3 d = texture(srcTex, vUV + t * vec2(-1.0,  0.0)).rgb;
    vec3 e = texture(srcTex, vUV).rgb;
    vec3 f = texture(srcTex, vUV + t * vec2( 1.0,  0.0)).rgb;
    vec3 g = texture(srcTex, vUV + t * vec2(-1.0, -1.0)).rgb;
    vec3 h = texture(srcTex, vUV + t * vec2( 0.0, -1.0)).rgb;
    vec3 i = texture(srcTex, vUV + t * vec2( 1.0, -1.0)).rgb;
    vec3 j = texture(srcTex, vUV + t * vec2(-0.5,  0.5)).rgb;
    vec3 k = texture(srcTex, vUV + t * vec2( 0.5,  0.5)).rgb;
    vec3 l = texture(srcTex, vUV + t * vec2(-0.5, -0.5)).rgb;
    vec3 m = texture(srcTex, vUV + t * vec2( 0.5, -0.5)).rgb;

    a = prefilter(a); b = prefilter(b); c = prefilter(c);
    d = prefilter(d); e = prefilter(e); f = prefilter(f);
    g = prefilter(g); h = prefilter(h); i = prefilter(i);
    j = prefilter(j); k = prefilter(k); l = prefilter(l); m = prefilter(m);

    // Karis weighted average of 5 quad blocks
    vec3 g1 = (a + b + d + e) * 0.25;
    vec3 g2 = (b + c + e + f) * 0.25;
    vec3 g3 = (d + e + g + h) * 0.25;
    vec3 g4 = (e + f + h + i) * 0.25;
    vec3 g5 = (j + k + l + m) * 0.25;

    float w1 = karis(g1), w2 = karis(g2), w3 = karis(g3), w4 = karis(g4), w5 = karis(g5);
    float wSum = w1 + w2 + w3 + w4 + w5;
    vec3 result = (g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4 + g5 * w5) / wSum;

    outColor = vec4(result, 1.0);
}
