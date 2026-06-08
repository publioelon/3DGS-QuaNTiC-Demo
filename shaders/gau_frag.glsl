#version 430 core

in vec2 v_quad;
in vec4 v_color;
out vec4 fragColor;

void main() {
    float r2 = dot(v_quad, v_quad);
    if (r2 > 1.0) {
        discard;
    }

    float alpha = exp(-4.0 * r2) * v_color.a;
    fragColor = vec4(v_color.rgb, alpha);
}
