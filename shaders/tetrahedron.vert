#version 450

layout(std140, set = 0, binding = 0) uniform SceneData {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_direction;
    vec4 directional_color;
    vec4 ambient_color;
    uvec4 light_counts;
} scene_data;

layout(std140, set = 1, binding = 0) uniform ModelData {
    mat4 model_matrix;
    vec4 albedo;
    vec4 specular;
    vec4 material_params;
} model_data;

layout(push_constant) uniform PipelineConsts {
    uint shader_flag;
} pipeline_consts;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec3 outColor;

void main() {
    vec4 world_position = model_data.model_matrix * vec4(inPosition, 1.0);
    gl_Position = scene_data.view_projection * world_position;
    outWorldPosition = world_position.xyz;
    outWorldNormal = transpose(inverse(mat3(model_data.model_matrix))) * inNormal;
    outColor = inColor;
}
