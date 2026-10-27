#version 450

struct PointLight {
    vec4 position;
    vec4 color_intensity;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color_intensity;
    vec4 angles;
};

layout(std140, set = 0, binding = 0) uniform SceneData {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_direction;
    vec4 directional_color;
    vec4 ambient_color;
    uvec4 light_counts;
} scene_data;

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelData {
    mat4 model_matrix;
    vec4 albedo;
    vec4 specular;
    vec4 material_params;
} model_data;

layout(push_constant) uniform PipelineConsts {
    uint shader_flag;
} pipeline_consts;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inColor;
layout(location = 0) out vec4 outColor;

vec3 calculateBlinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction,
    vec3 light_color, float intensity, vec3 albedo, vec3 specular_color, float gloss_value) {
    float diffuse_factor = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = albedo * light_color * diffuse_factor;
    vec3 specular = vec3(0.0);
    if (diffuse_factor > 0.0) {
        vec3 halfway_direction = normalize(light_direction + view_direction);
        float specular_factor = pow(max(dot(normal, halfway_direction), 0.0), gloss_value);
        specular = specular_color * light_color * specular_factor;
    }
    return (diffuse + specular) * intensity;
}

void main() {
    vec3 albedo = model_data.albedo.rgb * inColor;
    if (pipeline_consts.shader_flag != 0) {
        outColor = vec4(albedo, 1.0);
    } else {
        vec3 normal = normalize(inWorldNormal);
        vec3 view_direction = normalize(scene_data.camera_position.xyz - inWorldPosition);
        vec3 result = albedo * scene_data.ambient_color.rgb * scene_data.ambient_color.a;
        vec3 directional_light_direction = normalize(-scene_data.directional_direction.xyz);
        result += calculateBlinnPhong(normal, view_direction, directional_light_direction,
            scene_data.directional_color.rgb, scene_data.directional_color.a,
            albedo, model_data.specular.rgb, model_data.material_params.x);
        for (uint i = 0; i < scene_data.light_counts.x; i++) {
            vec3 offset = point_lights[i].position.xyz - inWorldPosition;
            float distance_squared = max(dot(offset, offset), 0.01);
            vec3 light_direction = normalize(offset);
            float attenuation = 1.0 / distance_squared;
            result += calculateBlinnPhong(normal, view_direction, light_direction,
                point_lights[i].color_intensity.rgb,
                point_lights[i].color_intensity.a * attenuation,
                albedo, model_data.specular.rgb, model_data.material_params.x);
        }
        for (uint i = 0; i < scene_data.light_counts.y; i++) {
            vec3 offset = spot_lights[i].position.xyz - inWorldPosition;
            float distance_squared = max(dot(offset, offset), 0.01);
            vec3 light_direction = normalize(offset);
            vec3 from_light_direction = normalize(inWorldPosition - spot_lights[i].position.xyz);
            float theta = dot(normalize(spot_lights[i].direction.xyz), from_light_direction);
            float inner_cos = spot_lights[i].angles.x;
            float outer_cos = spot_lights[i].angles.y;
            float spot_factor = 0.0;
            if (inner_cos > outer_cos) {
                spot_factor = smoothstep(outer_cos, inner_cos, theta);
            } else {
                spot_factor = step(outer_cos, theta);
            }
            float attenuation = spot_factor / distance_squared;
            result += calculateBlinnPhong(normal, view_direction, light_direction,
                spot_lights[i].color_intensity.rgb,
                spot_lights[i].color_intensity.a * attenuation,
                albedo, model_data.specular.rgb, model_data.material_params.x);
        }
        outColor = vec4(result, 1.0);
    }
}
