#include <vk_pipelines.h>
#include <fstream>
#include <vk_initializers.h>

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule)
{
    // open the file, with cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)file.tellg();

    // spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the file
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    //put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));

    file.close();

    // create a new shader module
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr
    };

    // codeSize has to be in bytes, so multiply the int in the buffer :)
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    // check that the creation goes well.
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

void PipelineBuilder::clear()
{
    // clear all the structs we need back to 0 with their correct stype
    _inputAssembly = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};

    _rasterizer = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};

    _colorBlendAttachment = {};

    _multisampling = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};

    _pipelineLayout = {};

    _depthStencil = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    _renderInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

    _shaderStages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device)
{
    // make viewport state from our stored viewport and scissor.
    // at the moment we won't support multiple viewports and scissors.
    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    // setup dummy color blending. We aren't using transparent objects yet
    // the binding is just "no blend", but we do write the color attachment
    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &_colorBlendAttachment
    };

    // completely clear VertexInputStateCreateInfo, as we have no need for it
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    // build the actual pipeline
    // we now use all the info structs we have been writing into this one to create the pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        // connect the renderInfo to the pNext extension mechanism
        .pNext = &_renderInfo,
        .stageCount = (uint32_t)_shaderStages.size(),
        .pStages = _shaderStages.data(),
        .pVertexInputState = &_vertexInputInfo,
        .pInputAssemblyState = &_inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &_rasterizer,
        .pMultisampleState = &_multisampling,
        .pDepthStencilState = &_depthStencil,
        .pColorBlendState = &colorBlending,
        .layout = _pipelineLayout
    };

    VkDynamicState state[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = &state[0]
    };

    pipelineInfo.pDynamicState = &dynamicInfo;

    // it's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
    {
        fmt::println("failed to create pipeline");
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    }
    return newPipeline;
}

void PipelineBuilder::set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader)
{
    _shaderStages.clear();

    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));

    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology)
{
    _inputAssembly.topology = topology;

    // we are not going to use primitive restart for now, so leave it on false
    _inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode)
{
    _rasterizer.polygonMode = mode;
    _rasterizer.lineWidth = 1.f;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace)
{
    _rasterizer.cullMode = cullMode;
    _rasterizer.frontFace = frontFace;
}

void PipelineBuilder::set_multisampling_none()
{
    _multisampling.sampleShadingEnable = VK_FALSE;
    // multisampling defaulted to no multisampling (1 sample per pixel)
    _multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    _multisampling.minSampleShading = 1.0f;
    _multisampling.pSampleMask = nullptr;
    // no alpha coverage either
    _multisampling.alphaToCoverageEnable = VK_FALSE;
    _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::disable_blending()
{
    // default write mask
    _colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    // no blending
    _colorBlendAttachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format)
{
    _colorAttachmentFormat = format;
    // connect the format to the renderinfo structure
    _renderInfo.colorAttachmentCount = 1;
    _renderInfo.pColorAttachmentFormats = &_colorAttachmentFormat;
}

void PipelineBuilder::set_depth_format(VkFormat format)
{
    _renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::disable_depthtest()
{
    _depthStencil.depthTestEnable = VK_FALSE;
    _depthStencil.depthWriteEnable = VK_FALSE;
    _depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
    _depthStencil.depthBoundsTestEnable = VK_FALSE;
    _depthStencil.stencilTestEnable = VK_FALSE;
    _depthStencil.front = {};
    _depthStencil.back = {};
    _depthStencil.minDepthBounds = 0.f;
    _depthStencil.maxDepthBounds = 1.f;
}
