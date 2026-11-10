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
    uint texture_effect;
    float time;
    float effect_strength;
} pipeline_consts;

layout(set = 2, binding = 0) uniform sampler2D albedo_texture;
layout(set = 2, binding = 1) uniform sampler2D specular_texture;
layout(set = 2, binding = 2) uniform sampler2D emissive_texture;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inCoord;
layout(location = 0) out vec4 outColor;

const uint TEXTURE_EFFECT_NONE = 0;
const uint TEXTURE_EFFECT_BOX_BLUR = 1;
const uint TEXTURE_EFFECT_FROSTED_GLASS = 2;
const uint TEXTURE_EFFECT_ROTATING_TRAIL = 3;
const uint TEXTURE_EFFECT_WAVE_DISTORTION = 4;
const uint TEXTURE_EFFECT_KALEIDOSCOPE = 5;
const uint TEXTURE_EFFECT_CHROMATIC_ABERRATION = 6;

uint nextRandom(uint seed) {
    return seed * 1103515245u + 12345u;
}

float randomFloat(uint value) {
    return float(value >> 8u) / float(1u << 24u);
}

vec4 sampleBoxBlur(vec2 coord) {
    vec2 texel_size = 1.0 / vec2(textureSize(albedo_texture, 0));
    float pulse = 0.5 + 0.5 * sin(pipeline_consts.time * 1.5);
    float radius = max(pipeline_consts.effect_strength * (4.0 + pulse * 12.0), 1.0);
    vec4 result = vec4(0.0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size * radius;
            result += texture(albedo_texture, coord + offset);
        }
    }
    return result / 9.0;
}

vec4 sampleFrostedGlass(vec2 coord) {
    vec2 texture_size = vec2(textureSize(albedo_texture, 0));
    ivec2 cell = ivec2(floor(coord * texture_size * 0.05));
    uint time_salt = uint(pipeline_consts.time * 60.0);
    uint seed = uint(cell.x) ^ (uint(cell.y) << 16u) ^ (time_salt << 8u);
    uint r1 = nextRandom(seed);
    uint r2 = nextRandom(r1);
    float random_x = randomFloat(r1);
    float random_y = randomFloat(r2);
    vec2 offset = (vec2(random_x, random_y) - 0.5) * pipeline_consts.effect_strength * 0.08;
    vec2 texel_size = 1.0 / texture_size;
    vec4 result = texture(albedo_texture, coord + offset);
    result += texture(albedo_texture, coord + offset + vec2(texel_size.x, 0.0));
    result += texture(albedo_texture, coord + offset - vec2(texel_size.x, 0.0));
    result += texture(albedo_texture, coord + offset + vec2(0.0, texel_size.y));
    result += texture(albedo_texture, coord + offset - vec2(0.0, texel_size.y));
    return result / 5.0;
}

vec4 sampleRotatingTrail(vec2 coord) {
    float angle = pipeline_consts.time * 1.5;
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    mat2 rotation = mat2(cos_angle, -sin_angle, sin_angle, cos_angle);
    vec2 direction = normalize(rotation * vec2(1.0, 0.0));
    float distance = pipeline_consts.effect_strength * 0.025;
    vec4 result = texture(albedo_texture, coord) * 0.4;
    result += texture(albedo_texture, coord - direction * distance) * 0.25;
    result += texture(albedo_texture, coord - direction * distance * 2.0) * 0.15;
    result += texture(albedo_texture, coord - direction * distance * 3.0) * 0.12;
    result += texture(albedo_texture, coord - direction * distance * 4.0) * 0.08;
    return result;
}

vec2 distortWave(vec2 coord) {
    float strength = pipeline_consts.effect_strength * 0.05;
    coord.x += sin(coord.y * 30.0 + pipeline_consts.time * 3.0) * strength;
    coord.y += sin(coord.x * 25.0 + pipeline_consts.time * 2.0) * strength;
    return coord;
}

vec2 transformKaleidoscope(vec2 coord) {
    vec2 centered_coord = coord - vec2(0.5);
    float radius = length(centered_coord);
    float angle = atan(centered_coord.y, centered_coord.x);
    angle += pipeline_consts.time * 0.8;
    float zoom = 1.0 + 0.15 * sin(pipeline_consts.time * 1.7);
    radius *= zoom;
    const float sector_angle = 2.0 * 3.14 / 8.0;
    angle = mod(angle, sector_angle);
    angle = min(angle, sector_angle - angle);
    return vec2(0.5) + vec2(cos(angle), sin(angle)) * radius;
}

vec4 sampleChromaticAberration(vec2 coord) {
    float shift = pipeline_consts.effect_strength * 0.05;
    vec2 offset = vec2(
        sin(pipeline_consts.time * 2.0),
        cos(pipeline_consts.time * 1.5)
    ) * shift;
    vec4 red_sample = texture(albedo_texture, coord + offset);
    vec4 green_sample = texture(albedo_texture, coord);
    vec4 blue_sample = texture(albedo_texture, coord - offset);
    return vec4(red_sample.r, green_sample.g, blue_sample.b, green_sample.a);
}

vec4 sampleAlbedo(vec2 coord) {
    if (pipeline_consts.texture_effect == TEXTURE_EFFECT_BOX_BLUR) {
        return sampleBoxBlur(coord);
    } else if (pipeline_consts.texture_effect == TEXTURE_EFFECT_FROSTED_GLASS) {
        return sampleFrostedGlass(coord);
    } else if (pipeline_consts.texture_effect == TEXTURE_EFFECT_ROTATING_TRAIL) {
        return sampleRotatingTrail(coord);
    } else if (pipeline_consts.texture_effect == TEXTURE_EFFECT_WAVE_DISTORTION) {
        return texture(albedo_texture, distortWave(coord));
    } else if (pipeline_consts.texture_effect == TEXTURE_EFFECT_KALEIDOSCOPE) {
        return texture(albedo_texture, transformKaleidoscope(coord));
    } else if (pipeline_consts.texture_effect == TEXTURE_EFFECT_CHROMATIC_ABERRATION) {
        return sampleChromaticAberration(coord);
    }
    return texture(albedo_texture, coord);
}

vec3 calculateBlinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction,
    vec3 light_color, float intensity, vec3 albedo, vec3 specular_color,
    float gloss_value) {
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
    if (model_data.material_params.y > 0.5) {
        albedo = model_data.albedo.rgb * sampleAlbedo(inCoord).rgb;
    }
    if (pipeline_consts.shader_flag != 0) {
        outColor = vec4(albedo, 1.0);
        return;
    }
    vec3 specular_color = model_data.specular.rgb * texture(specular_texture, inCoord).rgb;
    vec3 emissive_color = texture(emissive_texture, inCoord).rgb;
    vec3 normal = normalize(inWorldNormal);
    vec3 view_direction = normalize(scene_data.camera_position.xyz - inWorldPosition);
    vec3 result = albedo * scene_data.ambient_color.rgb * scene_data.ambient_color.a;
    vec3 directional_light_direction = normalize(-scene_data.directional_direction.xyz);
    result += calculateBlinnPhong(normal, view_direction, directional_light_direction,
        scene_data.directional_color.rgb, scene_data.directional_color.a,
        albedo, specular_color, model_data.material_params.x);
    for (uint i = 0; i < scene_data.light_counts.x; i++) {
        vec3 offset = point_lights[i].position.xyz - inWorldPosition;
        float distance_squared = max(dot(offset, offset), 0.01);
        vec3 light_direction = normalize(offset);
        float attenuation = 1.0 / distance_squared;
        result += calculateBlinnPhong(normal, view_direction, light_direction,
            point_lights[i].color_intensity.rgb,
            point_lights[i].color_intensity.a * attenuation,
            albedo, specular_color, model_data.material_params.x);
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
            albedo, specular_color, model_data.material_params.x);
    }
    result += emissive_color;
    outColor = vec4(result, 1.0);
}
