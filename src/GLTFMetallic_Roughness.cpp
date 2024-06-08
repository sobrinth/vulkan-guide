#include <GLTFMetallic_Roughness.h>

#include "vk_pipelines.h"

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", engine->device, &meshFragShader))
    {
        fmt::println("Error when building the mesh fragment shader module");
    }
}
