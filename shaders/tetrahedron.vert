#version 450

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 transformation_matrix;
    vec4 object_color;
} object_data;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = object_data.transformation_matrix * vec4(inPosition, 1.0);
    outColor = inColor * object_data.object_color.rgb;
}