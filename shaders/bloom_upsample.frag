#version 450

// 3x3 tent-filter upsample (additive blend handled by pipeline blend state)
layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D srcTex;

layout(push_constant) uniform PC {
    vec2  texelSize;
    float filterRadius;
    float pad;
} pc;

void main() {
    float r = pc.filterRadius;
    vec2 t = pc.texelSize * r;

    vec3 a = texture(srcTex, vUV + vec2(-t.x,  t.y)).rgb;
    vec3 b = texture(srcTex, vUV + vec2( 0.0,  t.y)).rgb;
    vec3 c = texture(srcTex, vUV + vec2( t.x,  t.y)).rgb;
    vec3 d = texture(srcTex, vUV + vec2(-t.x,  0.0)).rgb;
    vec3 e = texture(srcTex, vUV).rgb;
    vec3 f = texture(srcTex, vUV + vec2( t.x,  0.0)).rgb;
    vec3 g = texture(srcTex, vUV + vec2(-t.x, -t.y)).rgb;
    vec3 h = texture(srcTex, vUV + vec2( 0.0, -t.y)).rgb;
    vec3 i = texture(srcTex, vUV + vec2( t.x, -t.y)).rgb;

    vec3 sum = e * 4.0 + (b + d + f + h) * 2.0 + (a + c + g + i);
    outColor = vec4(sum / 16.0, 1.0);
}
