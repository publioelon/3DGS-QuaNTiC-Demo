#version 430 core

// Minimal Gaussian billboard shader used by the OpenGL fallback renderer.
// The CUDA renderer is the main dynamic path. This shader keeps the fallback
// path alive by drawing each Gaussian as a small camera-facing quad.

layout(location = 0) in vec2 position;

layout(std430, binding = 0) buffer GaussianBuffer {
    float gaussian_data[];
};

layout(std430, binding = 1) buffer SortIndexBuffer {
    int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 cam_pos;
uniform vec3 hfovxy_focal;
uniform float scale_modifier;
uniform int render_mod;
uniform int sh_dim;

out vec2 v_quad;
out vec4 v_color;

vec3 sh_to_rgb(vec3 sh0) {
    // Matches the DC SH convention used in util_gau.naive_gaussian().
    return clamp(sh0 * 0.28209 + vec3(0.5), 0.0, 1.0);
}

void main() {
    int sorted_id = gi[gl_InstanceID];
    int stride = 3 + 4 + 3 + 1 + sh_dim;
    int base = sorted_id * stride;

    vec3 xyz = vec3(
        gaussian_data[base + 0],
        gaussian_data[base + 1],
        gaussian_data[base + 2]
    );

    vec3 scale = vec3(
        gaussian_data[base + 7],
        gaussian_data[base + 8],
        gaussian_data[base + 9]
    );

    float opacity = gaussian_data[base + 10];
    vec3 sh0 = vec3(
        gaussian_data[base + 11],
        gaussian_data[base + 12],
        gaussian_data[base + 13]
    );

    vec4 view_pos = view_matrix * vec4(xyz, 1.0);
    float splat_size = max(max(scale.x, scale.y), scale.z) * scale_modifier;

    // Simple screen-aligned quad. It is not a full anisotropic Gaussian shader,
    // but it is enough for startup/fallback rendering before the CUDA path takes over.
    vec2 offset = position * splat_size;
    view_pos.xy += offset;

    gl_Position = projection_matrix * view_pos;
    v_quad = position;
    v_color = vec4(sh_to_rgb(sh0), opacity);
}
