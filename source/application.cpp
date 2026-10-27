#include "application.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

namespace {

struct Vertex {
	float position[3];
	float normal[3];
	float color[3];
};

struct Matrix {
	float numbers[16]{};
};

struct alignas(16) SceneData {
	float view_projection[16];
	float camera_position[4];
	float directional_direction[4];
	float directional_color[4];
	float ambient_color[4];
	uint32_t light_counts[4];
};

struct alignas(16) ModelData {
	float model_matrix[16];
	float albedo[4];
	float specular[4];
	float material_params[4];
};

struct alignas(16) PointLightData {
	float position[4];
	float color_intensity[4];
};

struct alignas(16) SpotLightData {
	float position[4];
	float direction[4];
	float color_intensity[4];
	float angles[4];
};

struct BufferResource {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
};

struct ObjectResources {
	size_t data_index = 0;
};

struct PipelineConsts {
	uint32_t shader_flag;
};

struct PointLightSettings {
	std::array<float, 3> position;
	std::array<float, 3> color;
	float intensity;
	bool hidden = false;
};

struct SpotLightSettings {
	std::array<float, 3> position;
	std::array<float, 3> direction;
	std::array<float, 3> color;
	float intensity;
	float inner_angle;
	float outer_angle;
	bool hidden = false;
};

constexpr size_t number_of_tetrahedrons = 3;
constexpr size_t max_light_count = 32;
constexpr size_t cone_segments = 32;
constexpr size_t cone_vertex_count = cone_segments * 4;
constexpr size_t sphere_rings = 8;
constexpr size_t sphere_segments = 12;
constexpr size_t sphere_vertex_count = (sphere_rings + 1) * (sphere_segments + 1);
constexpr size_t sphere_index_count = sphere_rings * sphere_segments * 6;
constexpr float pi = std::numbers::pi_v<float>;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;
uint32_t number_of_indices = 0;

BufferResource marker_vertex_buffer;
BufferResource marker_index_buffer;
uint32_t marker_number_of_indices = 0;
BufferResource sphere_vertex_buffer;
BufferResource sphere_index_buffer;
uint32_t sphere_number_of_indices = 0;
std::array<BufferResource, max_light_count> cone_buffers{};
BufferResource scene_buffer;
BufferResource model_buffer;
VkDeviceSize model_uniform_stride = sizeof(ModelData);
BufferResource point_light_buffer;
BufferResource spot_light_buffer;
size_t point_buffer_capacity = 1;
size_t spot_buffer_capacity = 1;

std::array<ObjectResources, number_of_tetrahedrons> objects{};
std::array<ObjectResources, max_light_count> point_markers{};
std::array<ObjectResources, max_light_count> spot_markers{};
ObjectResources directional_marker;
ObjectResources cone_object;
std::vector<ObjectResources*> model_resources;

VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;

int projection_mode = 0;
float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {20.0f, 20.0f, 20.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};

std::array<std::array<float, 3>, number_of_tetrahedrons> object_colors = {{
	{{1.0f, 0.5f, 0.3f}},
	{{0.3f, 1.0f, 0.5f}},
	{{0.5f, 0.3f, 1.0f}},
}};

std::array<std::array<float, 3>, number_of_tetrahedrons> specular_colors = {{
	{{1.0f, 1.0f, 1.0f}},
	{{1.0f, 1.0f, 1.0f}},
	{{1.0f, 1.0f, 1.0f}},
}};

std::array<float, number_of_tetrahedrons> gloss = {{20.0f, 20.0f, 20.0f}};

float camera_position[3] = {0.0f, 0.0f, 5.0f};
float camera_rotation[3] = {0.0f, 0.0f, 0.0f};
float view_angle = 45.0f;
float directional_position[3] = {-1.0f, 1.0f, -1.0f};
float directional_direction[3] = {0.0f, 0.0f, 0.0f};
float directional_color[3] = {1.0f, 1.0f, 1.0f};
float directional_intensity = 1.0f;
bool directional_hidden = false;
float ambient_color[3] = {0.2f, 0.2f, 0.2f};
float ambient_intensity = 1.0f;
bool visualization_enabled = true;

std::vector<PointLightSettings> point_lights = {
	{{1.0f, 0.5f, 1.0f}, {1.0f, 0.3f, 0.2f}, 3.0f},
};
std::vector<SpotLightSettings> spot_lights = {
	{{0.0f, 1.0f, 1.0f}, {0.0f, -0.45f, -1.0f}, {0.3f, 0.45f, 1.0f}, 8.0f, 15.0f, 25.0f},
};

bool pause = false;
float moving_speed = 1.0f;
float rotation_radius = 1.0f;
float vertical_amplitude = 1.0f;
float rotation_speed_x = 20.0f;
float rotation_speed_y = 20.0f;
float rotation_speed_z = 20.0f;
double trajectory_angle = 0.0;
double moving_rotation_x = 0.0;
double moving_rotation_y = 0.0;
double moving_rotation_z = 0.0;
double prev_time = -1.0;

void printError(const std::string& str) {
	std::cerr << str << '\n';
	std::ofstream log("Error", std::ios::app);
	if (log.is_open()) {
		log << str << '\n';
	}
}

void correctNumber(float& number, float correct) {
	if (!std::isfinite(number)) {
		number = correct;
	}
}

float correctDegrees(float degrees) {
	if (!std::isfinite(degrees)) {
		return 0.0f;
	}
	return std::fmod(degrees, 360.0f);
}

Matrix identityMatrix() {
	Matrix matrix{};
	matrix.numbers[0] = 1.0f;
	matrix.numbers[5] = 1.0f;
	matrix.numbers[10] = 1.0f;
	matrix.numbers[15] = 1.0f;
	return matrix;
}

Matrix multiplyMatrices(const Matrix& left, const Matrix& right) {
	Matrix result{};
	for (int column = 0; column < 4; column++) {
		for (int row = 0; row < 4; row++) {
			for (int i = 0; i < 4; i++) {
				result.numbers[column * 4 + row] +=
					left.numbers[i * 4 + row] * right.numbers[column * 4 + i];
			}
		}
	}
	return result;
}

Matrix transferMatrix(float x, float y, float z) {
	Matrix matrix = identityMatrix();
	matrix.numbers[12] = x;
	matrix.numbers[13] = y;
	matrix.numbers[14] = z;
	return matrix;
}

Matrix scaleMatrix(float x, float y, float z) {
	Matrix matrix{};
	matrix.numbers[0] = x;
	matrix.numbers[5] = y;
	matrix.numbers[10] = z;
	matrix.numbers[15] = 1.0f;
	return matrix;
}

Matrix rotationXMatrix(float angle) {
	Matrix matrix = identityMatrix();
	const float cos = std::cos(angle);
	const float sin = std::sin(angle);
	matrix.numbers[5] = cos;
	matrix.numbers[6] = sin;
	matrix.numbers[9] = -sin;
	matrix.numbers[10] = cos;
	return matrix;
}

Matrix rotationYMatrix(float angle) {
	Matrix matrix = identityMatrix();
	const float cos = std::cos(angle);
	const float sin = std::sin(angle);
	matrix.numbers[0] = cos;
	matrix.numbers[2] = -sin;
	matrix.numbers[8] = sin;
	matrix.numbers[10] = cos;
	return matrix;
}

Matrix rotationZMatrix(float angle) {
	Matrix matrix = identityMatrix();
	const float cos = std::cos(angle);
	const float sin = std::sin(angle);
	matrix.numbers[0] = cos;
	matrix.numbers[1] = sin;
	matrix.numbers[4] = -sin;
	matrix.numbers[5] = cos;
	return matrix;
}

Matrix perspectiveMatrix(float aspect, float view_angle_degrees) {
	const float factor = 1.0f / std::tan(view_angle_degrees * pi / 360.0f);
	constexpr float near_view = 0.1f;
	constexpr float far_view = 100.0f;
	Matrix matrix{};
	matrix.numbers[0] = factor / aspect;
	matrix.numbers[5] = -factor;
	matrix.numbers[10] = far_view / (near_view - far_view);
	matrix.numbers[11] = -1.0f;
	matrix.numbers[14] = far_view * near_view / (near_view - far_view);
	return matrix;
}

Matrix orthographicMatrix(float aspect) {
	constexpr float half_height = 2.5f;
	const float half_width = half_height * aspect;
	constexpr float near_view = 0.1f;
	constexpr float far_view = 100.0f;
	Matrix matrix = identityMatrix();
	matrix.numbers[0] = 1.0f / half_width;
	matrix.numbers[5] = -1.0f / half_height;
	matrix.numbers[10] = 1.0f / (near_view - far_view);
	matrix.numbers[14] = near_view / (near_view - far_view);
	return matrix;
}

std::array<float, 3> normalizeVector(const std::array<float, 3>& vector) {
	const float length = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] +
		vector[2] * vector[2]);
	if (length <= 0.00001f || !std::isfinite(length)) {
		return {0.0f, -1.0f, 0.0f};
	}
	return {vector[0] / length, vector[1] / length, vector[2] / length};
}

std::array<float, 3> cross(const std::array<float, 3>& left,
	const std::array<float, 3>& right) {
	return {
		left[1] * right[2] - left[2] * right[1],
		left[2] * right[0] - left[0] * right[2],
		left[0] * right[1] - left[1] * right[0],
	};
}

float dot(const std::array<float, 3>& left, const std::array<float, 3>& right) {
	return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

void addTriangle(std::vector<uint32_t>& indices, uint32_t a, uint32_t b, uint32_t c) {
	indices.push_back(a);
	indices.push_back(b);
	indices.push_back(c);
}

void addHexagon(std::vector<uint32_t>& indices, const std::array<uint32_t, 6>& face) {
	for (size_t i = 1; i + 1 < face.size(); i++) {
		addTriangle(indices, face[0], face[i], face[i + 1]);
	}
}

std::vector<uint32_t> createIndices() {
	std::vector<uint32_t> indices;
	addTriangle(indices, 0, 2, 4);
	addTriangle(indices, 1, 8, 6);
	addTriangle(indices, 3, 7, 10);
	addTriangle(indices, 5, 11, 9);
	addHexagon(indices, {0, 1, 6, 7, 3, 2});
	addHexagon(indices, {0, 4, 5, 9, 8, 1});
	addHexagon(indices, {2, 3, 10, 11, 5, 4});
	addHexagon(indices, {6, 8, 9, 11, 10, 7});
	return indices;
}

bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
	const void* data, BufferResource& resource) {
	if (size == 0) {
		printError("Cannot create an empty Vulkan buffer");
		return false;
	}
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	auto& context = graphics::internal::context;
	const VkResult result = vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
		&resource.buffer, &resource.allocation, nullptr);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan buffer");
		return false;
	}
	if (data != nullptr) {
		void* mapped_data = nullptr;
		const VkResult map_result = vmaMapMemory(context.allocator, resource.allocation,
			&mapped_data);
		if (map_result != VK_SUCCESS) {
			printError("Failed to map Vulkan buffer memory");
			vmaDestroyBuffer(context.allocator, resource.buffer, resource.allocation);
			resource = {};
			return false;
		}
		std::memcpy(mapped_data, data, static_cast<size_t>(size));
		vmaFlushAllocation(context.allocator, resource.allocation, 0, size);
		vmaUnmapMemory(context.allocator, resource.allocation);
	}
	return true;
}

bool updateHostBuffer(const BufferResource& resource, const void* data, VkDeviceSize size) {
	if (resource.buffer == VK_NULL_HANDLE || resource.allocation == VK_NULL_HANDLE || data == nullptr) {
		return false;
	}
	void* mapped_data = nullptr;
	const VkResult result = vmaMapMemory(graphics::internal::context.allocator,
		resource.allocation, &mapped_data);
	if (result != VK_SUCCESS) {
		printError("Failed to map Vulkan buffer for update");
		return false;
	}
	std::memcpy(mapped_data, data, static_cast<size_t>(size));
	vmaFlushAllocation(graphics::internal::context.allocator, resource.allocation, 0, size);
	vmaUnmapMemory(graphics::internal::context.allocator, resource.allocation);
	return true;
}

void destroyBuffer(BufferResource& resource) {
	if (resource.buffer != VK_NULL_HANDLE && graphics::internal::context.allocator != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, resource.buffer, resource.allocation);
	}
	resource = {};
}

bool createTetrahedron() {
	const std::array<std::array<float, 3>, 12> positions = {{
		{1.0f, 1.0f / 3.0f, 1.0f / 3.0f},
		{1.0f, -1.0f / 3.0f, -1.0f / 3.0f},
		{1.0f / 3.0f, 1.0f, 1.0f / 3.0f},
		{-1.0f / 3.0f, 1.0f, -1.0f / 3.0f},
		{1.0f / 3.0f, 1.0f / 3.0f, 1.0f},
		{-1.0f / 3.0f, -1.0f / 3.0f, 1.0f},
		{1.0f / 3.0f, -1.0f / 3.0f, -1.0f},
		{-1.0f / 3.0f, 1.0f / 3.0f, -1.0f},
		{1.0f / 3.0f, -1.0f, -1.0f / 3.0f},
		{-1.0f / 3.0f, -1.0f, 1.0f / 3.0f},
		{-1.0f, 1.0f / 3.0f, -1.0f / 3.0f},
		{-1.0f, -1.0f / 3.0f, 1.0f / 3.0f},
	}};
	std::array<Vertex, 12> vertices{};
	constexpr float size = 0.5f;
	for (size_t i = 0; i < positions.size(); i++) {
		for (size_t component = 0; component < 3; component++) {
			vertices[i].position[component] = positions[i][component] * size;
			vertices[i].color[component] = (positions[i][component] + 1.0f) * 0.5f;
		}
	}
	const std::vector<uint32_t> indices = createIndices();
	std::array<std::array<float, 3>, 12> normal_sums{};
	for (size_t i = 0; i + 2 < indices.size(); i += 3) {
		const Vertex& first = vertices[indices[i]];
		const Vertex& second = vertices[indices[i + 1]];
		const Vertex& third = vertices[indices[i + 2]];
		const std::array<float, 3> edge_a = {
			second.position[0] - first.position[0],
			second.position[1] - first.position[1],
			second.position[2] - first.position[2],
		};
		const std::array<float, 3> edge_b = {
			third.position[0] - first.position[0],
			third.position[1] - first.position[1],
			third.position[2] - first.position[2],
		};
		std::array<float, 3> normal = cross(edge_a, edge_b);
		const std::array<float, 3> center = {
			(first.position[0] + second.position[0] + third.position[0]) / 3.0f,
			(first.position[1] + second.position[1] + third.position[1]) / 3.0f,
			(first.position[2] + second.position[2] + third.position[2]) / 3.0f,
		};
		if (dot(normal, center) < 0.0f) {
			for (float& component : normal) {
				component = -component;
			}
		}
		for (size_t component = 0; component < 3; component++) {
			normal_sums[indices[i]][component] += normal[component];
			normal_sums[indices[i + 1]][component] += normal[component];
			normal_sums[indices[i + 2]][component] += normal[component];
		}
	}
	for (size_t i = 0; i < vertices.size(); i++) {
		const std::array<float, 3> normal = normalizeVector(normal_sums[i]);
		for (size_t component = 0; component < 3; component++) {
			vertices[i].normal[component] = normal[component];
		}
	}
	number_of_indices = static_cast<uint32_t>(indices.size());
	BufferResource vertex_resource;
	BufferResource index_resource;
	if (!createHostBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		vertices.data(), vertex_resource)) {
		return false;
	}
	if (!createHostBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		indices.data(), index_resource)) {
		destroyBuffer(vertex_resource);
		return false;
	}
	vertex_buffer = vertex_resource.buffer;
	vertex_allocation = vertex_resource.allocation;
	index_buffer = index_resource.buffer;
	index_allocation = index_resource.allocation;
	return true;
}

bool createCubeGeometry() {
	const std::array<std::array<std::array<float, 3>, 4>, 6> faces = {{
		{{{0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}}},
		{{{-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}}},
		{{{-0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}}},
		{{{-0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}}},
		{{{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}}},
		{{{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}}},
	}};
	const std::array<std::array<float, 3>, 6> normals = {{
		{{1.0f, 0.0f, 0.0f}}, {{-1.0f, 0.0f, 0.0f}},
		{{0.0f, 1.0f, 0.0f}}, {{0.0f, -1.0f, 0.0f}},
		{{0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, -1.0f}},
	}};
	std::array<Vertex, 24> vertices{};
	std::array<uint32_t, 36> indices{};
	for (size_t face = 0; face < faces.size(); face++) {
		for (size_t vertex = 0; vertex < 4; vertex++) {
			for (size_t component = 0; component < 3; component++) {
				vertices[face * 4 + vertex].position[component] = faces[face][vertex][component];
				vertices[face * 4 + vertex].normal[component] = normals[face][component];
				vertices[face * 4 + vertex].color[component] = 1.0f;
			}
		}
		const uint32_t offset = static_cast<uint32_t>(face * 4);
		indices[face * 6] = offset;
		indices[face * 6 + 1] = offset + 1;
		indices[face * 6 + 2] = offset + 2;
		indices[face * 6 + 3] = offset;
		indices[face * 6 + 4] = offset + 2;
		indices[face * 6 + 5] = offset + 3;
	}
	marker_number_of_indices = static_cast<uint32_t>(indices.size());
	if (!createHostBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		vertices.data(), marker_vertex_buffer)) {
		return false;
	}
	if (!createHostBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		indices.data(), marker_index_buffer)) {
		destroyBuffer(marker_vertex_buffer);
		return false;
	}
	return true;
}

bool createSphereGeometry() {
	std::array<Vertex, sphere_vertex_count> vertices{};
	std::array<uint32_t, sphere_index_count> indices{};
	for (size_t ring = 0; ring <= sphere_rings; ring++) {
		const float vertical_angle = pi * static_cast<float>(ring) / static_cast<float>(sphere_rings);
		const float y = std::cos(vertical_angle);
		const float ring_radius = std::sin(vertical_angle);
		for (size_t segment = 0; segment <= sphere_segments; segment++) {
			const float horizontal_angle = 2.0f * pi * static_cast<float>(segment) /
				static_cast<float>(sphere_segments);
			const size_t index = ring * (sphere_segments + 1) + segment;
			vertices[index].position[0] = 0.5f * ring_radius * std::cos(horizontal_angle);
			vertices[index].position[1] = 0.5f * y;
			vertices[index].position[2] = 0.5f * ring_radius * std::sin(horizontal_angle);
			vertices[index].normal[0] = ring_radius * std::cos(horizontal_angle);
			vertices[index].normal[1] = y;
			vertices[index].normal[2] = ring_radius * std::sin(horizontal_angle);
			vertices[index].color[0] = 1.0f;
			vertices[index].color[1] = 1.0f;
			vertices[index].color[2] = 1.0f;
		}
	}
	size_t index = 0;
	for (size_t ring = 0; ring < sphere_rings; ring++) {
		for (size_t segment = 0; segment < sphere_segments; segment++) {
			const uint32_t first = static_cast<uint32_t>(ring * (sphere_segments + 1) + segment);
			const uint32_t second = first + static_cast<uint32_t>(sphere_segments + 1);
			indices[index++] = first;
			indices[index++] = first + 1;
			indices[index++] = second;
			indices[index++] = first + 1;
			indices[index++] = second + 1;
			indices[index++] = second;
		}
	}
	sphere_number_of_indices = static_cast<uint32_t>(indices.size());
	if (!createHostBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		vertices.data(), sphere_vertex_buffer)) {
		return false;
	}
	if (!createHostBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		indices.data(), sphere_index_buffer)) {
		destroyBuffer(sphere_vertex_buffer);
		return false;
	}
	return true;
}

bool createShaderModule(const char* name, VkShaderModule& shader_module) {
	std::ifstream file(name, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		printError(std::string("Failed to open shader file: ") + name);
		return false;
	}
	const std::streamsize size = file.tellg();
	if (size <= 0 || size % 4 != 0) {
		printError(std::string("Invalid shader file: ") + name);
		return false;
	}
	std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	if (!file) {
		printError(std::string("Failed to read shader file: ") + name);
		return false;
	}
	const VkShaderModuleCreateInfo shader_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(size),
		.pCode = code.data(),
	};
	const VkResult result = vkCreateShaderModule(graphics::internal::context.device,
		&shader_info, nullptr, &shader_module);
	if (result != VK_SUCCESS) {
		printError(std::string("Failed to create Vulkan shader module: ") + name);
		return false;
	}
	return true;
}

bool createDataBuffers() {
	if (!createHostBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		nullptr, scene_buffer)) {
		return false;
	}
	if (!createHostBuffer(sizeof(PointLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		nullptr, point_light_buffer)) {
		return false;
	}
	if (!createHostBuffer(sizeof(SpotLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		nullptr, spot_light_buffer)) {
		return false;
	}
	for (BufferResource& buffer : cone_buffers) {
		if (!createHostBuffer(cone_vertex_count * sizeof(Vertex),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, nullptr, buffer)) {
			return false;
		}
	}
	model_resources.reserve(number_of_tetrahedrons + max_light_count * 2 + 2);
	for (ObjectResources& object : objects) {
		model_resources.push_back(&object);
	}
	for (ObjectResources& object : point_markers) {
		model_resources.push_back(&object);
	}
	for (ObjectResources& object : spot_markers) {
		model_resources.push_back(&object);
	}
	model_resources.push_back(&directional_marker);
	model_resources.push_back(&cone_object);
	for (size_t i = 0; i < model_resources.size(); i++) {
		model_resources[i]->data_index = i;
	}
	VkPhysicalDeviceProperties device_properties{};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device,
		&device_properties);
	const VkDeviceSize alignment = std::max<VkDeviceSize>(
		1, device_properties.limits.minUniformBufferOffsetAlignment);
	model_uniform_stride = ((sizeof(ModelData) + alignment - 1) / alignment) * alignment;
	const VkDeviceSize model_buffer_size = model_uniform_stride * model_resources.size();
	if (!createHostBuffer(model_buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		nullptr, model_buffer)) {
		return false;
	}
	return true;
}

bool createDescriptorResources() {
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};
	auto& context = graphics::internal::context;
	VkResult result = vkCreateDescriptorSetLayout(context.device, &scene_layout_info,
		nullptr, &scene_descriptor_set_layout);
	if (result != VK_SUCCESS) {
		printError("Failed to create scene descriptor set layout");
		return false;
	}
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	result = vkCreateDescriptorSetLayout(context.device, &model_layout_info,
		nullptr, &model_descriptor_set_layout);
	if (result != VK_SUCCESS) {
		printError("Failed to create model descriptor set layout");
		return false;
	}
	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 3,
		.pPoolSizes = pool_sizes,
	};
	result = vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan descriptor pool");
		return false;
	}
	const VkDescriptorSetLayout layouts[] = {
		scene_descriptor_set_layout,
		model_descriptor_set_layout,
	};
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = layouts,
	};
	VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	result = vkAllocateDescriptorSets(context.device, &set_info, descriptor_sets);
	if (result != VK_SUCCESS) {
		printError("Failed to allocate Vulkan descriptor sets");
		return false;
	}
	scene_descriptor_set = descriptor_sets[0];
	model_descriptor_set = descriptor_sets[1];
	const VkDescriptorBufferInfo scene_info = {
		.buffer = scene_buffer.buffer,
		.offset = 0,
		.range = sizeof(SceneData),
	};
	const VkDescriptorBufferInfo point_info = {
		.buffer = point_light_buffer.buffer,
		.offset = 0,
		.range = sizeof(PointLightData) * point_buffer_capacity,
	};
	const VkDescriptorBufferInfo spot_info = {
		.buffer = spot_light_buffer.buffer,
		.offset = 0,
		.range = sizeof(SpotLightData) * spot_buffer_capacity,
	};
	const VkDescriptorBufferInfo model_info = {
		.buffer = model_buffer.buffer,
		.offset = 0,
		.range = sizeof(ModelData),
	};
	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &point_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &spot_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_info,
		},
	};
	vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);
	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = (sizeof(PipelineConsts) + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1),
	};
	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};
	result = vkCreatePipelineLayout(context.device, &pipeline_layout_info,
		nullptr, &pipeline_layout);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan pipeline layout");
		return false;
	}
	return true;
}

bool createGraphicsPipeline(VkPrimitiveTopology topology, VkCullModeFlags cull_mode,
	VkPipeline& pipeline) {
	VkShaderModule vert_shader = VK_NULL_HANDLE;
	VkShaderModule frag_shader = VK_NULL_HANDLE;
	if (!createShaderModule("shaders/tetrahedron.vert.spv", vert_shader)) {
		return false;
	}
	if (!createShaderModule("shaders/tetrahedron.frag.spv", frag_shader)) {
		vkDestroyShaderModule(graphics::internal::context.device, vert_shader, nullptr);
		return false;
	}
	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag_shader,
			.pName = "main",
		},
	};
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = static_cast<uint32_t>(sizeof(Vertex)),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = static_cast<uint32_t>(offsetof(Vertex, position)),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = static_cast<uint32_t>(offsetof(Vertex, normal)),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = static_cast<uint32_t>(offsetof(Vertex, color)),
		},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
		.primitiveRestartEnable = VK_FALSE,
	};
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cull_mode,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};
	const VkPipelineColorBlendAttachmentState color_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};
	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = graphics::internal::context.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(graphics::internal::context.device,
		VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
	vkDestroyShaderModule(graphics::internal::context.device, vert_shader, nullptr);
	vkDestroyShaderModule(graphics::internal::context.device, frag_shader, nullptr);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan graphics pipeline");
		return false;
	}
	return true;
}

void updateModelData(ObjectResources& object, const Matrix& model_matrix,
	const float* albedo, const float* specular, float material_gloss) {
	ModelData data{};
	std::memcpy(data.model_matrix, model_matrix.numbers, sizeof(model_matrix.numbers));
	for (size_t i = 0; i < 3; i++) {
		data.albedo[i] = albedo[i];
		data.specular[i] = specular[i];
	}
	data.albedo[3] = 1.0f;
	data.specular[3] = 1.0f;
	data.material_params[0] = material_gloss;
	void* mapped_data = nullptr;
	const VkResult result = vmaMapMemory(graphics::internal::context.allocator,
		model_buffer.allocation, &mapped_data);
	if (result != VK_SUCCESS) {
		printError("Failed to map model uniform buffer");
		return;
	}
	const VkDeviceSize offset = object.data_index * model_uniform_stride;
	std::memcpy(static_cast<std::byte*>(mapped_data) + offset, &data, sizeof(data));
	vmaFlushAllocation(graphics::internal::context.allocator, model_buffer.allocation,
		offset, sizeof(data));
	vmaUnmapMemory(graphics::internal::context.allocator, model_buffer.allocation);
}

void bindModelData(VkCommandBuffer command_buffer, const ObjectResources& object,
	bool shader_flag) {
	const uint32_t dynamic_offset = static_cast<uint32_t>(object.data_index * model_uniform_stride);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline_layout, 1, 1, &model_descriptor_set, 1, &dynamic_offset);
	PipelineConsts constants{};
	if (shader_flag) {
		constants.shader_flag = 1u;
	}
	vkCmdPushConstants(command_buffer, pipeline_layout,
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		0, sizeof(constants), &constants);
}

void updateSceneDescriptor(uint32_t binding, const BufferResource& buffer, VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = range,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = scene_descriptor_set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

bool ensurePointBufferCapacity(size_t required_count) {
	required_count = std::max<size_t>(required_count, 1);
	if (required_count <= point_buffer_capacity) {
		return true;
	}
	size_t new_capacity = point_buffer_capacity;
	while (new_capacity < required_count) {
		new_capacity *= 2;
	}
	BufferResource new_buffer;
	if (!createHostBuffer(sizeof(PointLightData) * new_capacity,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, new_buffer)) {
		return false;
	}
	BufferResource old_buffer = point_light_buffer;
	point_light_buffer = new_buffer;
	point_buffer_capacity = new_capacity;
	updateSceneDescriptor(1, point_light_buffer, sizeof(PointLightData) * point_buffer_capacity);
	destroyBuffer(old_buffer);
	return true;
}

bool ensureSpotBufferCapacity(size_t required_count) {
	required_count = std::max<size_t>(required_count, 1);
	if (required_count <= spot_buffer_capacity) {
		return true;
	}
	size_t new_capacity = spot_buffer_capacity;
	while (new_capacity < required_count) {
		new_capacity *= 2;
	}
	BufferResource new_buffer;
	if (!createHostBuffer(sizeof(SpotLightData) * new_capacity,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, new_buffer)) {
		return false;
	}
	BufferResource old_buffer = spot_light_buffer;
	spot_light_buffer = new_buffer;
	spot_buffer_capacity = new_capacity;
	updateSceneDescriptor(2, spot_light_buffer, sizeof(SpotLightData) * spot_buffer_capacity);
	destroyBuffer(old_buffer);
	return true;
}

bool updateLightBuffers() {
	size_t visible_point_count = 0;
	for (const PointLightSettings& light : point_lights) {
		if (!light.hidden) ++visible_point_count;
	}
	size_t visible_spot_count = 0;
	for (const SpotLightSettings& light : spot_lights) {
		if (!light.hidden) ++visible_spot_count;
	}
	const size_t point_capacity = std::max<size_t>(visible_point_count, 1);
	const size_t spot_capacity = std::max<size_t>(visible_spot_count, 1);
	if (!ensurePointBufferCapacity(point_capacity) || !ensureSpotBufferCapacity(spot_capacity)) {
		return false;
	}
	std::vector<PointLightData> point_data(point_capacity);
	size_t point_index = 0;
	for (const PointLightSettings& light : point_lights) {
		if (light.hidden) continue;
		for (size_t c = 0; c < 3; c++) {
			point_data[point_index].position[c] = light.position[c];
			point_data[point_index].color_intensity[c] = light.color[c];
		}
		point_data[point_index].position[3] = 1.0f;
		point_data[point_index].color_intensity[3] = light.intensity;
		++point_index;
	}
	std::vector<SpotLightData> spot_data(spot_capacity);
	size_t spot_index = 0;
	for (const SpotLightSettings& light : spot_lights) {
		if (light.hidden) continue;
		const std::array<float, 3> direction = normalizeVector(light.direction);
		for (size_t c = 0; c < 3; c++) {
			spot_data[spot_index].position[c] = light.position[c];
			spot_data[spot_index].direction[c] = direction[c];
			spot_data[spot_index].color_intensity[c] = light.color[c];
		}
		spot_data[spot_index].position[3] = 1.0f;
		spot_data[spot_index].direction[3] = 0.0f;
		spot_data[spot_index].color_intensity[3] = light.intensity;
		spot_data[spot_index].angles[0] = std::cos(light.inner_angle * pi / 180.0f);
		spot_data[spot_index].angles[1] = std::cos(light.outer_angle * pi / 180.0f);
		++spot_index;
	}
	if (!updateHostBuffer(point_light_buffer, point_data.data(),
		point_data.size() * sizeof(PointLightData))) return false;
	if (!updateHostBuffer(spot_light_buffer, spot_data.data(),
		spot_data.size() * sizeof(SpotLightData))) return false;
	return true;
}

void updateSceneData(const Matrix& view_projection, bool light_buffers_ready) {
	SceneData data{};
	std::memcpy(data.view_projection, view_projection.numbers, sizeof(view_projection.numbers));
	for (size_t c = 0; c < 3; c++) {
		data.camera_position[c] = camera_position[c];
		data.directional_direction[c] = directional_direction[c];
		data.directional_color[c] = directional_color[c];
		data.ambient_color[c] = ambient_color[c];
	}
	data.camera_position[3] = 1.0f;
	data.directional_direction[3] = 0.0f;
	if (directional_hidden) {
		data.directional_color[3] = 0.0f;
	} else {
		data.directional_color[3] = directional_intensity;
	}
	data.ambient_color[3] = ambient_intensity;
	if (light_buffers_ready) {
		uint32_t point_count = 0;
		for (const PointLightSettings& light : point_lights) {
			if (!light.hidden) ++point_count;
		}
		uint32_t spot_count = 0;
		for (const SpotLightSettings& light : spot_lights) {
			if (!light.hidden) ++spot_count;
		}
		data.light_counts[0] = point_count;
		data.light_counts[1] = spot_count;
	}
	updateHostBuffer(scene_buffer, &data, sizeof(data));
}

void createConeVertices(const SpotLightSettings& light, std::array<Vertex, cone_vertex_count>& vertices) {
	const std::array<float, 3> direction = normalizeVector(light.direction);
	std::array<float, 3> reference = {0.0f, 1.0f, 0.0f};
	if (std::abs(dot(direction, reference)) > 0.95f) {
		reference = {1.0f, 0.0f, 0.0f};
	}
	const std::array<float, 3> side = normalizeVector(cross(direction, reference));
	const std::array<float, 3> up = cross(direction, side);
	constexpr float cone_length = 2.0f;
	const float outer_angle = std::clamp(light.outer_angle, 1.0f, 90.0f) * pi / 180.0f;
	const float cone_radius = cone_length * std::tan(outer_angle);
	const std::array<float, 3> end = {
		light.position[0] + direction[0] * cone_length,
		light.position[1] + direction[1] * cone_length,
		light.position[2] + direction[2] * cone_length,
	};
	size_t vertex_index = 0;
	for (size_t i = 0; i < cone_segments; i++) {
		const float angle_a = 2.0f * pi * static_cast<float>(i) / static_cast<float>(cone_segments);
		const float angle_b = 2.0f * pi * static_cast<float>(i + 1) / static_cast<float>(cone_segments);
		const std::array<float, 3> ring_a = {
			end[0] + cone_radius * (side[0] * std::cos(angle_a) + up[0] * std::sin(angle_a)),
			end[1] + cone_radius * (side[1] * std::cos(angle_a) + up[1] * std::sin(angle_a)),
			end[2] + cone_radius * (side[2] * std::cos(angle_a) + up[2] * std::sin(angle_a)),
		};
		const std::array<float, 3> ring_b = {
			end[0] + cone_radius * (side[0] * std::cos(angle_b) + up[0] * std::sin(angle_b)),
			end[1] + cone_radius * (side[1] * std::cos(angle_b) + up[1] * std::sin(angle_b)),
			end[2] + cone_radius * (side[2] * std::cos(angle_b) + up[2] * std::sin(angle_b)),
		};
		const std::array<std::array<float, 3>, 4> line_points = {{
			light.position, ring_a, ring_a, ring_b,
		}};
		for (const std::array<float, 3>& point : line_points) {
			for (size_t component = 0; component < 3; component++) {
				vertices[vertex_index].position[component] = point[component];
				vertices[vertex_index].normal[component] = 0.0f;
				vertices[vertex_index].color[component] = 1.0f;
			}
			vertex_index++;
		}
	}
}

void cleanUserValues() {
	for (float& value : position) {
		correctNumber(value, 0.0f);
		value = std::clamp(value, -5.0f, 5.0f);
	}
	for (float& value : rotation) {
		correctNumber(value, 0.0f);
		value = std::clamp(value, -180.0f, 180.0f);
	}
	for (float& value : scale) {
		correctNumber(value, 1.0f);
		value = std::clamp(std::abs(value), 0.001f, 5.0f);
	}
	for (float& value : camera_position) {
		correctNumber(value, 0.0f);
		value = std::clamp(value, -10.0f, 10.0f);
	}
	for (float& value : camera_rotation) {
		correctNumber(value, 0.0f);
		value = std::clamp(value, -180.0f, 180.0f);
	}
	if (!std::isfinite(view_angle)) {
		view_angle = 45.0f;
	}
	view_angle = std::clamp(view_angle, 15.0f, 100.0f);
	for (float& value : directional_position) {
		correctNumber(value, 0.0f);
		value = std::clamp(value, -8.0f, 8.0f);
	}
	const std::array<float, 3> source_direction = normalizeVector({
		directional_position[0], directional_position[1], directional_position[2]});
	for (size_t component = 0; component < 3; component++) {
		directional_direction[component] = -source_direction[component];
	}
	for (float& value : directional_color) {
		correctNumber(value, 1.0f);
		value = std::clamp(value, 0.0f, 1.0f);
	}
	for (float& value : ambient_color) {
		correctNumber(value, 0.15f);
		value = std::clamp(value, 0.0f, 1.0f);
	}
	correctNumber(directional_intensity, 1.0f);
	directional_intensity = std::clamp(directional_intensity, 0.0f, 100.0f);
	correctNumber(ambient_intensity, 1.0f);
	ambient_intensity = std::clamp(ambient_intensity, 0.0f, 5.0f);
	correctNumber(moving_speed, 1.0f);
	moving_speed = std::clamp(moving_speed, -10.0f, 10.0f);
	correctNumber(rotation_radius, 1.0f);
	rotation_radius = std::clamp(rotation_radius, 0.0f, 5.0f);
	correctNumber(vertical_amplitude, 0.5f);
	vertical_amplitude = std::clamp(vertical_amplitude, 0.0f, 5.0f);
	for (float* speed : {&rotation_speed_x, &rotation_speed_y, &rotation_speed_z}) {
		correctNumber(*speed, 20.0f);
		*speed = std::clamp(*speed, -360.0f, 360.0f);
	}
	for (auto& color : object_colors) {
		for (float& component : color) {
			correctNumber(component, 1.0f);
			component = std::clamp(component, 0.0f, 1.0f);
		}
	}
	for (auto& color : specular_colors) {
		for (float& component : color) {
			correctNumber(component, 1.0f);
			component = std::clamp(component, 0.0f, 1.0f);
		}
	}
	for (float& value : gloss) {
		correctNumber(value, 32.0f);
		value = std::clamp(value, 1.0f, 256.0f);
	}
	for (PointLightSettings& light : point_lights) {
		for (float& value : light.position) {
			correctNumber(value, 0.0f);
			value = std::clamp(value, -8.0f, 8.0f);
		}
		for (float& value : light.color) {
			correctNumber(value, 1.0f);
			value = std::clamp(value, 0.0f, 1.0f);
		}
		correctNumber(light.intensity, 1.0f);
		light.intensity = std::clamp(light.intensity, 0.0f, 1000.0f);
	}
	for (SpotLightSettings& light : spot_lights) {
		for (float& value : light.position) {
			correctNumber(value, 0.0f);
			value = std::clamp(value, -8.0f, 8.0f);
		}
		for (float& value : light.direction) {
			correctNumber(value, -1.0f);
			value = std::clamp(value, -1.0f, 1.0f);
		}
		if (std::abs(light.direction[0]) + std::abs(light.direction[1]) +
			std::abs(light.direction[2]) < 0.00001f) {
			light.direction = {0.0f, -1.0f, 0.0f};
		}
		for (float& value : light.color) {
			correctNumber(value, 1.0f);
			value = std::clamp(value, 0.0f, 1.0f);
		}
		correctNumber(light.intensity, 1.0f);
		light.intensity = std::clamp(light.intensity, 0.0f, 1000.0f);
		correctNumber(light.inner_angle, 15.0f);
		correctNumber(light.outer_angle, 25.0f);
		light.inner_angle = std::clamp(light.inner_angle, 1.0f, 90.0f);
		light.outer_angle = std::clamp(light.outer_angle, 1.0f, 90.0f);
		if (light.inner_angle > light.outer_angle) {
			light.inner_angle = light.outer_angle;
		}
	}
}

} // namespace

bool initialize() {
	if (!createTetrahedron()) {
		shutdown();
		return false;
	}
	if (!createCubeGeometry()) {
		shutdown();
		return false;
	}
	if (!createSphereGeometry()) {
		shutdown();
		return false;
	}
	if (!createDataBuffers()) {
		shutdown();
		return false;
	}
	if (!createDescriptorResources()) {
		shutdown();
		return false;
	}
	if (!createGraphicsPipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		VK_CULL_MODE_BACK_BIT, graphics_pipeline)) {
		shutdown();
		return false;
	}
	if (!createGraphicsPipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		VK_CULL_MODE_NONE, line_pipeline)) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	if (context.device != VK_NULL_HANDLE && context.graphics_queue != VK_NULL_HANDLE) {
		vkQueueWaitIdle(context.graphics_queue);
	}
	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		graphics_pipeline = VK_NULL_HANDLE;
	}
	if (line_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, line_pipeline, nullptr);
		line_pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		scene_descriptor_set = VK_NULL_HANDLE;
		model_descriptor_set = VK_NULL_HANDLE;
	}
	if (scene_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, scene_descriptor_set_layout, nullptr);
		scene_descriptor_set_layout = VK_NULL_HANDLE;
	}
	if (model_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, model_descriptor_set_layout, nullptr);
		model_descriptor_set_layout = VK_NULL_HANDLE;
	}
	model_resources.clear();
	destroyBuffer(scene_buffer);
	destroyBuffer(model_buffer);
	destroyBuffer(point_light_buffer);
	destroyBuffer(spot_light_buffer);
	destroyBuffer(marker_vertex_buffer);
	destroyBuffer(marker_index_buffer);
	destroyBuffer(sphere_vertex_buffer);
	destroyBuffer(sphere_index_buffer);
	for (BufferResource& buffer : cone_buffers) {
		destroyBuffer(buffer);
	}
	if (index_buffer != VK_NULL_HANDLE && context.allocator != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
		index_buffer = VK_NULL_HANDLE;
		index_allocation = VK_NULL_HANDLE;
	}
	if (vertex_buffer != VK_NULL_HANDLE && context.allocator != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_allocation = VK_NULL_HANDLE;
	}
	number_of_indices = 0;
	marker_number_of_indices = 0;
	sphere_number_of_indices = 0;
	model_uniform_stride = sizeof(ModelData);
	point_buffer_capacity = 1;
	spot_buffer_capacity = 1;
}

void update(double time) {
	ImGui::Begin("Truncated regular tetrahedron");
	const char* projection_modes[] = {
		"Perspective",
		"Orthographic",
	};
	ImGui::Combo("Projection", &projection_mode, projection_modes, 2);
	ImGui::Separator();
	ImGui::InputFloat3("Position", position, "%.2f");
	ImGui::InputFloat3("Rotation", rotation, "%.1f");
	ImGui::InputFloat3("Scale", scale, "%.2f");
	ImGui::Separator();
	ImGui::Checkbox("Pause", &pause);
	ImGui::InputFloat("Moving speed", &moving_speed, 0.0f, 0.0f, "%.2f");
	ImGui::InputFloat("Rotation radius", &rotation_radius, 0.0f, 0.0f, "%.2f");
	ImGui::InputFloat("Vertical amplitude", &vertical_amplitude, 0.0f, 0.0f, "%.2f");
	ImGui::InputFloat("X rotation speed", &rotation_speed_x, 0.0f, 0.0f, "%.1f");
	ImGui::InputFloat("Y rotation speed", &rotation_speed_y, 0.0f, 0.0f, "%.1f");
	ImGui::InputFloat("Z rotation speed", &rotation_speed_z, 0.0f, 0.0f, "%.1f");
	ImGui::Separator();
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		ImGui::PushID(static_cast<int>(i));
		const std::string label = "Object " + std::to_string(i + 1);
		ImGui::Text("%s", label.c_str());
		ImGui::ColorEdit3("Color", object_colors[i].data(),
			ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
		ImGui::ColorEdit3("Specular color", specular_colors[i].data(),
			ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
		ImGui::SliderFloat("Gloss", &gloss[i], 1.0f, 256.0f, "%.1f");
		ImGui::Separator();
		ImGui::PopID();
	}
	ImGui::End();
	ImGui::Begin("Lighting");
	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("Camera position", camera_position, -10.0f, 10.0f, "%.2f");
		ImGui::SliderFloat3("Camera rotation", camera_rotation, -180.0f, 180.0f, "%.1f");
		ImGui::SliderFloat("View angle", &view_angle, 15.0f, 100.0f, "%.1f");
	}
	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Directional light");
		ImGui::SliderFloat3("Position", directional_position, -8.0f, 8.0f, "%.2f");
		ImGui::ColorEdit3("Color", directional_color,
			ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
		ImGui::SliderFloat("Intensity", &directional_intensity, 0.0f, 10.0f, "%.2f");
		ImGui::Checkbox("Hide", &directional_hidden);
		ImGui::PopID();
	}
	if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Ambient light");
		ImGui::ColorEdit3("Color", ambient_color,
			ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
		ImGui::SliderFloat("Intensity", &ambient_intensity, 0.0f, 2.0f, "%.2f");
		ImGui::PopID();
	}
	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Point lights");
		for (size_t i = 0; i < point_lights.size(); i++) {
			ImGui::PushID(static_cast<int>(i));
			const std::string label = "Point light " + std::to_string(i + 1);
			bool remove_light = false;
			if (ImGui::TreeNode(label.c_str())) {
				ImGui::SliderFloat3("Position", point_lights[i].position.data(), -8.0f, 8.0f, "%.2f");
				ImGui::ColorEdit3("Color", point_lights[i].color.data(),
					ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
				ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 30.0f, "%.2f");
				ImGui::Checkbox("Hide", &point_lights[i].hidden);
				if (ImGui::Button("Delete")) {
					point_lights.erase(point_lights.begin() + static_cast<std::ptrdiff_t>(i));
					remove_light = true;
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
			if (remove_light) {
				break;
			}
		}
		if (point_lights.size() < max_light_count) {
			if (ImGui::Button("Add")) {
				point_lights.push_back({{0.0f, 0.5f, 1.5f}, {1.0f, 1.0f, 1.0f}, 3.0f});
			}
		}
		ImGui::PopID();
	}
	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Spot lights");
		for (size_t i = 0; i < spot_lights.size(); i++) {
			ImGui::PushID(static_cast<int>(i + max_light_count));
			const std::string label = "Spot light " + std::to_string(i + 1);
			bool remove_light = false;
			if (ImGui::TreeNode(label.c_str())) {
				ImGui::SliderFloat3("Position", spot_lights[i].position.data(), -8.0f, 8.0f, "%.2f");
				ImGui::SliderFloat3("Direction", spot_lights[i].direction.data(), -1.0f, 1.0f, "%.2f");
				ImGui::ColorEdit3("Color", spot_lights[i].color.data(),
					ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
				ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 30.0f, "%.2f");
				ImGui::SliderFloat("Inner angle", &spot_lights[i].inner_angle, 1.0f, 89.0f, "%.1f");
				ImGui::SliderFloat("Outer angle", &spot_lights[i].outer_angle, 1.0f, 89.0f, "%.1f");
				ImGui::Checkbox("Hide", &spot_lights[i].hidden);
				if (ImGui::Button("Delete")) {
					spot_lights.erase(spot_lights.begin() + static_cast<std::ptrdiff_t>(i));
					remove_light = true;
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
			if (remove_light) {
				break;
			}
		}
		if (spot_lights.size() < max_light_count) {
			if (ImGui::Button("Add")) {
				spot_lights.push_back({{0.0f, 0.8f, 1.0f}, {0.0f, -1.0f, 0.0f},
					{1.0f, 1.0f, 1.0f}, 5.0f, 15.0f, 30.0f});
			}
		}
		ImGui::PopID();
	}
	ImGui::Checkbox("Visualization", &visualization_enabled);
	ImGui::End();
	cleanUserValues();
	double delta_time = 0.0;
	if (prev_time >= 0.0 && time >= prev_time) {
		delta_time = time - prev_time;
	}
	prev_time = time;
	delta_time = std::min(delta_time, 0.1);
	if (!pause && delta_time > 0.0) {
		const double scaled_delta = delta_time * static_cast<double>(moving_speed);
		trajectory_angle = std::fmod(trajectory_angle + scaled_delta, pi * 2);
		if (trajectory_angle < 0.0) {
			trajectory_angle += pi * 2;
		}
		moving_rotation_x = std::fmod(moving_rotation_x +
			scaled_delta * static_cast<double>(rotation_speed_x), 360.0);
		moving_rotation_y = std::fmod(moving_rotation_y +
			scaled_delta * static_cast<double>(rotation_speed_y), 360.0);
		moving_rotation_z = std::fmod(moving_rotation_z +
			scaled_delta * static_cast<double>(rotation_speed_z), 360.0);
	}
}

void render(const graphics::internal::FrameData& frame_data) {
	auto& context = graphics::internal::context;
	if (context.swapchain_extent.width == 0 || context.swapchain_extent.height == 0) {
		return;
	}
	const float aspect = static_cast<float>(context.swapchain_extent.width) /
		static_cast<float>(context.swapchain_extent.height);
	Matrix projection_matrix;
	if (projection_mode == 0) {
		projection_matrix = perspectiveMatrix(aspect, view_angle);
	} else {
		projection_matrix = orthographicMatrix(aspect);
	}
	const float camera_x = correctDegrees(camera_rotation[0]) * pi / 180.0f;
	const float camera_y = correctDegrees(camera_rotation[1]) * pi / 180.0f;
	const float camera_z = correctDegrees(camera_rotation[2]) * pi / 180.0f;
	Matrix view = rotationXMatrix(-camera_x);
	view = multiplyMatrices(view, rotationYMatrix(-camera_y));
	view = multiplyMatrices(view, rotationZMatrix(-camera_z));
	view = multiplyMatrices(view, transferMatrix(-camera_position[0],
		-camera_position[1], -camera_position[2]));
	const Matrix view_projection = multiplyMatrices(projection_matrix, view);
	const bool light_buffers_ready = updateLightBuffers();
	updateSceneData(view_projection, light_buffers_ready);
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		const double object_phase = trajectory_angle + pi * 2.0 * static_cast<double>(i) /
			static_cast<double>(number_of_tetrahedrons);
		const float path_x = rotation_radius * static_cast<float>(std::cos(object_phase));
		const float path_y = vertical_amplitude * static_cast<float>(std::sin(2.0 * object_phase));
		const float path_z = rotation_radius * static_cast<float>(std::sin(object_phase));
		const float rotation_offset = static_cast<float>(i) * 18.0f;
		const float radians_x = correctDegrees(rotation[0] +
			static_cast<float>(moving_rotation_x) + rotation_offset) * pi / 180.0f;
		const float radians_y = correctDegrees(rotation[1] +
			static_cast<float>(moving_rotation_y) + rotation_offset) * pi / 180.0f;
		const float radians_z = correctDegrees(rotation[2] +
			static_cast<float>(moving_rotation_z) - rotation_offset) * pi / 180.0f;
		Matrix model = transferMatrix(position[0] + path_x, position[1] + path_y,
			position[2] + path_z);
		model = multiplyMatrices(model, rotationZMatrix(radians_z));
		model = multiplyMatrices(model, rotationYMatrix(radians_y));
		model = multiplyMatrices(model, rotationXMatrix(radians_x));
		model = multiplyMatrices(model, scaleMatrix(scale[0], scale[1], scale[2]));
		updateModelData(objects[i], model, object_colors[i].data(),
			specular_colors[i].data(), gloss[i]);
	}
	if (visualization_enabled) {
		const Matrix identity = identityMatrix();
		const float white[3] = {1.0f, 1.0f, 1.0f};
		if (!directional_hidden) {
			Matrix model = transferMatrix(directional_position[0], directional_position[1],
				directional_position[2]);
			model = multiplyMatrices(model, scaleMatrix(0.18f, 0.18f, 0.18f));
			updateModelData(directional_marker, model, directional_color, white, 1.0f);
		}
		for (size_t i = 0; i < point_lights.size(); i++) {
			if (point_lights[i].hidden) {
				continue;
			}
			Matrix model = transferMatrix(point_lights[i].position[0], point_lights[i].position[1],
				point_lights[i].position[2]);
			model = multiplyMatrices(model, scaleMatrix(0.12f, 0.12f, 0.12f));
			updateModelData(point_markers[i], model, point_lights[i].color.data(), white, 1.0f);
		}
		for (size_t i = 0; i < spot_lights.size(); i++) {
			if (spot_lights[i].hidden) {
				continue;
			}
			Matrix model = transferMatrix(spot_lights[i].position[0], spot_lights[i].position[1],
				spot_lights[i].position[2]);
			model = multiplyMatrices(model, scaleMatrix(0.12f, 0.12f, 0.12f));
			updateModelData(spot_markers[i], model, spot_lights[i].color.data(), white, 1.0f);
			std::array<Vertex, cone_vertex_count> vertices{};
			createConeVertices(spot_lights[i], vertices);
			updateHostBuffer(cone_buffers[i], vertices.data(), sizeof(vertices));
		}
		updateModelData(cone_object, identity, white, white, 1.0f);
	}
	vkResetCommandBuffer(frame_data.command_buffer, 0);
	const VkCommandBufferBeginInfo command_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(frame_data.command_buffer, &command_begin) != VK_SUCCESS) {
		printError("Failed to begin Vulkan command buffer");
		return;
	}
	const VkClearValue clear_numbers[] = {
		{
			.color = {.float32 = {0.015f, 0.02f, 0.03f, 1.0f}},
		},
		{
			.depthStencil = {1.0f, 0},
		},
	};
	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = frame_data.framebuffer,
		.renderArea = {.extent = context.swapchain_extent},
		.clearValueCount = 2,
		.pClearValues = clear_numbers,
	};
	vkCmdBeginRenderPass(frame_data.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor_rectangle = {
		.offset = {0, 0},
		.extent = context.swapchain_extent,
	};
	vkCmdSetViewport(frame_data.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(frame_data.command_buffer, 0, 1, &scissor_rectangle);
	vkCmdBindDescriptorSets(frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);
	vkCmdBindPipeline(frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1, &vertex_buffer, &vertex_offset);
	vkCmdBindIndexBuffer(frame_data.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		bindModelData(frame_data.command_buffer, objects[i], false);
		vkCmdDrawIndexed(frame_data.command_buffer, number_of_indices, 1, 0, 0, 0);
	}
	if (visualization_enabled) {
		if (!directional_hidden) {
			vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1,
				&sphere_vertex_buffer.buffer, &vertex_offset);
			vkCmdBindIndexBuffer(frame_data.command_buffer, sphere_index_buffer.buffer, 0,
				VK_INDEX_TYPE_UINT32);
			bindModelData(frame_data.command_buffer, directional_marker, true);
			vkCmdDrawIndexed(frame_data.command_buffer, sphere_number_of_indices, 1, 0, 0, 0);
		}
		for (size_t i = 0; i < point_lights.size(); i++) {
			if (point_lights[i].hidden) {
				continue;
			}
			vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1,
				&marker_vertex_buffer.buffer, &vertex_offset);
			vkCmdBindIndexBuffer(frame_data.command_buffer, marker_index_buffer.buffer, 0,
				VK_INDEX_TYPE_UINT32);
			bindModelData(frame_data.command_buffer, point_markers[i], true);
			vkCmdDrawIndexed(frame_data.command_buffer, marker_number_of_indices, 1, 0, 0, 0);
		}
		for (size_t i = 0; i < spot_lights.size(); i++) {
			if (spot_lights[i].hidden) {
				continue;
			}
			vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1,
				&sphere_vertex_buffer.buffer, &vertex_offset);
			vkCmdBindIndexBuffer(frame_data.command_buffer, sphere_index_buffer.buffer, 0,
				VK_INDEX_TYPE_UINT32);
			bindModelData(frame_data.command_buffer, spot_markers[i], true);
			vkCmdDrawIndexed(frame_data.command_buffer, sphere_number_of_indices, 1, 0, 0, 0);
		}
		vkCmdBindPipeline(frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
		for (size_t i = 0; i < spot_lights.size(); i++) {
			if (spot_lights[i].hidden) {
				continue;
			}
			vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1,
				&cone_buffers[i].buffer, &vertex_offset);
			bindModelData(frame_data.command_buffer, cone_object, true);
			vkCmdDraw(frame_data.command_buffer, static_cast<uint32_t>(cone_vertex_count), 1, 0, 0);
		}
	}
	vkCmdEndRenderPass(frame_data.command_buffer);
	if (vkEndCommandBuffer(frame_data.command_buffer) != VK_SUCCESS) {
		printError("Failed to finish Vulkan command buffer");
	}
}

} // namespace application
