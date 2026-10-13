#include "application.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

namespace {

struct Vertex {
	float position[3];
	float color[3];
};

struct Matrix {
	float numbers[16]{};
};

struct alignas(16) ObjectData {
	float transformation_matrix[16];
	float object_color[4];
};

struct ObjectResources {
	VkBuffer data_buffer = VK_NULL_HANDLE;
	VmaAllocation data_allocation = VK_NULL_HANDLE;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

constexpr size_t number_of_tetrahedrons = 3;
constexpr float pi = std::numbers::pi_v<float>;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;
uint32_t number_of_indices = 0;
std::array<ObjectResources, number_of_tetrahedrons> objects{};
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;

int projection_mode = 0;
float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {20.0f, 20.0f, 20.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};

std::array<std::array<float, 3>, number_of_tetrahedrons> object_colors = {{
	{{1.0f, 0.5f, 0.3f}},
	{{0.3f, 1.0f, 0.5f}},
	{{0.5f, 0.3f, 1.0f}},
}};

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

Matrix perspectiveMatrix(float aspect) {
	const float factor = 1.0f / std::tan(45.0f * pi / 360.0f);
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
	const void* data, VkBuffer& buffer, VmaAllocation& allocation) {
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
	VkResult result = vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                                 &buffer, &allocation, nullptr);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan buffer");
		return false;
	}
	if (data != nullptr) {
		void* mapped_data = nullptr;
		result = vmaMapMemory(context.allocator, allocation, &mapped_data);
		if (result != VK_SUCCESS) {
			printError("Failed to map Vulkan buffer memory");
			vmaDestroyBuffer(context.allocator, buffer, allocation);
			buffer = VK_NULL_HANDLE;
			allocation = VK_NULL_HANDLE;
			return false;
		}
		std::memcpy(mapped_data, data, static_cast<size_t>(size));
		vmaFlushAllocation(context.allocator, allocation, 0, size);
		vmaUnmapMemory(context.allocator, allocation);
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
	number_of_indices = static_cast<uint32_t>(indices.size());
	if (!createHostBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
		vertices.data(), vertex_buffer, vertex_allocation)) {
		return false;
	}
	if (!createHostBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
		indices.data(), index_buffer, index_allocation)) {
		return false;
	}
	return true;
}

bool createDataBuffers() {
	for (ObjectResources& object : objects) {
		if (!createHostBuffer(sizeof(ObjectData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
			nullptr, object.data_buffer, object.data_allocation)) {
			return false;
		}
	}
	return true;
}

bool createDescriptorResources() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	auto& context = graphics::internal::context;
	VkResult result = vkCreateDescriptorSetLayout(context.device, &descriptor_layout_info, 
		nullptr, &descriptor_set_layout);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan descriptor set layout");
		return false;
	}
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = static_cast<uint32_t>(number_of_tetrahedrons),
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = static_cast<uint32_t>(number_of_tetrahedrons),
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	result = vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan descriptor pool");
		return false;
	}
	std::array<VkDescriptorSetLayout, number_of_tetrahedrons> set_layouts{};
	set_layouts.fill(descriptor_set_layout);
	std::array<VkDescriptorSet, number_of_tetrahedrons> descriptor_sets{};
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = static_cast<uint32_t>(number_of_tetrahedrons),
		.pSetLayouts = set_layouts.data(),
	};
	result = vkAllocateDescriptorSets(context.device, &set_info, descriptor_sets.data());
	if (result != VK_SUCCESS) {
		printError("Failed to allocate Vulkan descriptor sets");
		return false;
	}
	std::array<VkDescriptorBufferInfo, number_of_tetrahedrons> buffer_infos{};
	std::array<VkWriteDescriptorSet, number_of_tetrahedrons> descriptor_writes{};
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		objects[i].descriptor_set = descriptor_sets[i];
		buffer_infos[i] = {
			.buffer = objects[i].data_buffer,
			.offset = 0,
			.range = sizeof(ObjectData),
		};
		descriptor_writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = objects[i].descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}
	vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(number_of_tetrahedrons), 
		descriptor_writes.data(), 0, nullptr);
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};
	result = vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan pipeline layout");
		return false;
	}
	return true;
}

bool createGraphicsPipeline() {
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
			.offset = static_cast<uint32_t>(offsetof(Vertex, color)),
		},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
		.cullMode = VK_CULL_MODE_BACK_BIT,
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
		VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);
	vkDestroyShaderModule(graphics::internal::context.device, vert_shader, nullptr);
	vkDestroyShaderModule(graphics::internal::context.device, frag_shader, nullptr);
	if (result != VK_SUCCESS) {
		printError("Failed to create Vulkan graphics pipeline");
		return false;
	}
	return true;
}

void updateDataBuffer(size_t object_index, const Matrix& transformation_matrix) {
	ObjectData object_data{};
	std::memcpy(object_data.transformation_matrix, transformation_matrix.numbers, 
		sizeof(transformation_matrix.numbers));
	object_data.object_color[0] = object_colors[object_index][0];
	object_data.object_color[1] = object_colors[object_index][1];
	object_data.object_color[2] = object_colors[object_index][2];
	object_data.object_color[3] = 1.0f;
	auto& context = graphics::internal::context;
	void* mapped_data = nullptr;
	const VkResult result = vmaMapMemory(context.allocator, 
		objects[object_index].data_allocation, &mapped_data);
	if (result != VK_SUCCESS) {
		printError("Failed to map data buffer memory");
		return;
	}
	std::memcpy(mapped_data, &object_data, sizeof(object_data));
	vmaFlushAllocation(context.allocator, objects[object_index].data_allocation, 
		0, sizeof(object_data));
	vmaUnmapMemory(context.allocator, objects[object_index].data_allocation);
}

} // namespace

bool initialize() {
	if (!createTetrahedron()) {
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
	if (!createGraphicsPipeline()) {
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
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		for (ObjectResources& object : objects) {
			object.descriptor_set = VK_NULL_HANDLE;
		}
	}
	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
		descriptor_set_layout = VK_NULL_HANDLE;
	}
	for (ObjectResources& object : objects) {
		if (object.data_buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(context.allocator, object.data_buffer, object.data_allocation);
			object.data_buffer = VK_NULL_HANDLE;
			object.data_allocation = VK_NULL_HANDLE;
		}
	}
	if (index_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
		index_buffer = VK_NULL_HANDLE;
		index_allocation = VK_NULL_HANDLE;
	}
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_allocation = VK_NULL_HANDLE;
	}
	number_of_indices = 0;
}

void update(double time) {
    ImGui::Begin("Truncated regular tetrahedron");
    const char* projection_modes[] = {
        "Perspective",
        "Orthographic"
    };
    ImGui::Combo("Projection", &projection_mode, projection_modes, 2);
    ImGui::Separator();
    ImGui::InputFloat3("Position", position, "%.3f");
    ImGui::InputFloat3("Rotation", rotation, "%.3f");
    ImGui::InputFloat3("Scale", scale, "%.3f");
    ImGui::Separator();
    ImGui::Checkbox("Pause", &pause);
    ImGui::InputFloat("Moving speed", &moving_speed, 0.1f, 1.0f, "%.3f");
    ImGui::InputFloat("Rotation radius", &rotation_radius, 0.05f, 0.25f, "%.3f");
    ImGui::InputFloat("Vertical amplitude", &vertical_amplitude, 0.05f, 0.25f, "%.3f");
    ImGui::InputFloat("X rotation speed", &rotation_speed_x, 1.0f, 5.0f, "%.2f");
    ImGui::InputFloat("Y rotation speed", &rotation_speed_y, 1.0f, 5.0f, "%.2f");
    ImGui::InputFloat("Z rotation speed", &rotation_speed_z, 1.0f, 5.0f, "%.2f");
    ImGui::Separator();
    for (size_t i = 0; i < number_of_tetrahedrons; i++) {
        const std::string str = "Object " + std::to_string(i + 1) + " colour";
        ImGui::ColorEdit3(str.c_str(), object_colors[i].data(), 
			ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB);
    }
    ImGui::End();
    for (float& value : position) {
        correctNumber(value, 0.0f);
    }
    for (float& value : rotation) {
        correctNumber(value, 0.0f);
    }
    for (float& value : scale) {
        correctNumber(value, 1.0f);
        value = std::max(std::abs(value), 0.001f);
    }
    correctNumber(moving_speed, 1.0f);
    correctNumber(rotation_radius, 1.0f);
    correctNumber(vertical_amplitude, 0.50f);
    correctNumber(rotation_speed_x, 20.0f);
    correctNumber(rotation_speed_y, 20.0f);
    correctNumber(rotation_speed_z, 20.0f);
    for (auto& color : object_colors) {
        for (float& component : color) {
            correctNumber(component, 1.0f);
            component = std::clamp(component, 0.0f, 1.0f);
        }
    }
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
			scaled_delta * static_cast<double>(rotation_speed_z),360.0);
    }
}

void render(const graphics::internal::FrameData& frame_data) {
	auto& context = graphics::internal::context;
	if (context.swapchain_extent.width == 0 || context.swapchain_extent.height == 0) {
		return;
	}
	const float aspect = static_cast<float>(context.swapchain_extent.width) / 
		static_cast<float>(context.swapchain_extent.height);
	const Matrix view = transferMatrix(0.0f, 0.0f, -4.0f);
	Matrix projection_matrix;
	if (projection_mode == 0) {
		projection_matrix = perspectiveMatrix(aspect);
	} else {
		projection_matrix = orthographicMatrix(aspect);
	}
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		const double object_phase = trajectory_angle + pi * 2 * static_cast<double>(i) / 
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
		const Matrix view_model = multiplyMatrices(view, model);
		const Matrix transformation_matrix = multiplyMatrices(projection_matrix, view_model);
		updateDataBuffer(i, transformation_matrix);
	}
	vkResetCommandBuffer(frame_data.command_buffer, 0);
	const VkCommandBufferBeginInfo command_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(frame_data.command_buffer, &command_begin);
	const VkClearValue clear_numbers[] = {
		{
			.color = {.float32 = {0, 0, 0, 1.0f}},
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
	vkCmdBindPipeline(frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
		graphics_pipeline);
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(frame_data.command_buffer, 0, 1, &vertex_buffer, &vertex_offset);
	vkCmdBindIndexBuffer(frame_data.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
	for (size_t i = 0; i < number_of_tetrahedrons; i++) {
		vkCmdBindDescriptorSets(frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
			pipeline_layout, 0, 1, &objects[i].descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(frame_data.command_buffer, number_of_indices, 1, 0, 0, 0);
	}
	vkCmdEndRenderPass(frame_data.command_buffer);
	vkEndCommandBuffer(frame_data.command_buffer);
}

} // namespace application