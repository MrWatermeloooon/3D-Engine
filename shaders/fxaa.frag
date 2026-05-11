#version 450

// FXAA 3.11 lite, working on the LDR composite output.
// Single fullscreen pass. Push constants: vec4(texelSize.xy, enable, _).

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D srcTex;

layout(push_constant) uniform PC {
    vec4 params;   // xy = 1/textureSize, z = enable (0/1)
} pc;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 rgbM = texture(srcTex, vUV).rgb;
    if (pc.params.z < 0.5) { outColor = vec4(rgbM, 1.0); return; }

    vec2 t = pc.params.xy;
    vec3 rgbNW = texture(srcTex, vUV + vec2(-1.0, -1.0) * t).rgb;
    vec3 rgbNE = texture(srcTex, vUV + vec2( 1.0, -1.0) * t).rgb;
    vec3 rgbSW = texture(srcTex, vUV + vec2(-1.0,  1.0) * t).rgb;
    vec3 rgbSE = texture(srcTex, vUV + vec2( 1.0,  1.0) * t).rgb;

    float lumaNW = luma(rgbNW);
    float lumaNE = luma(rgbNE);
    float lumaSW = luma(rgbSW);
    float lumaSE = luma(rgbSE);
    float lumaM  = luma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    float range = lumaMax - lumaMin;

    // Skip if no perceptible edge (saves work in flat regions)
    if (range < max(0.0312, lumaMax * 0.125)) {
        outColor = vec4(rgbM, 1.0);
        return;
    }

    // Edge direction
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * (1.0/8.0)), 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * t;

    vec3 rgbA = 0.5 * (
        texture(srcTex, vUV + dir * (1.0/3.0 - 0.5)).rgb +
        texture(srcTex, vUV + dir * (2.0/3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(srcTex, vUV + dir * -0.5).rgb +
        texture(srcTex, vUV + dir *  0.5).rgb);

    float lumaB = luma(rgbB);
    outColor = vec4(((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB, 1.0);
}
