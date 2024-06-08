//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

// bootstrap library
#include "VkBootstrap.h"

#include <chrono>
#include <thread>

#include "vk_images.h"
#include "vk_pipelines.h"

#define VMA_IMPLEMENTATION
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

VulkanEngine* loaded_engine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loaded_engine; }

constexpr bool B_USE_VALIDATION_LAYERS = true;

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loaded_engine == nullptr);
    loaded_engine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    constexpr SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN;

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        static_cast<int>(_windowExtent.width),
        static_cast<int>(_windowExtent.height),
        window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_imgui();

    init_default_data();
    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        // make sure the gpu has stopped doing its tings
        vkDeviceWaitIdle(_device);

        _mainDeletionQueue.flush();
        _globalDescriptorAllocator.destroy_pool(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);

        for (auto& [_swapchainSemaphore, _renderSemaphore, _renderFence, _commandPool, _, _deletionQueue] : _frames)
        {
            _deletionQueue.flush();

            vkDestroyCommandPool(_device, _commandPool, nullptr);

            // destroy sync objects
            vkDestroyFence(_device, _renderFence, nullptr);
            vkDestroySemaphore(_device, _renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _swapchainSemaphore, nullptr);
        }

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loaded_engine = nullptr;
}

void VulkanEngine::run()
{
    assert(loaded_engine != nullptr);
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit)
    {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0)
        {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT)
            {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED)
                {
                    _stopRendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED)
                {
                    _stopRendering = false;
                }
            }

            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (_stopRendering)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("background"))
        {
            auto& [name, pipeline, layout, data] = _backgroundEffects[_activeBackgroundEffect];

            ImGui::Text("Selected effect: %s", name);

            ImGui::SliderInt("Effect Index", &_activeBackgroundEffect, 0, _backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", reinterpret_cast<float*>(&data.data1));
            ImGui::InputFloat4("data2", reinterpret_cast<float*>(&data.data2));
            ImGui::InputFloat4("data3", reinterpret_cast<float*>(&data.data3));
            ImGui::InputFloat4("data4", reinterpret_cast<float*>(&data.data4));
        }
        ImGui::End();

        // demo imgui UI
        // ImGui::ShowDemoWindow();

        // make imgui calculate internal draw structures
        ImGui::Render();

        draw();
    }
}

void VulkanEngine::draw()
{
    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame().renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame().renderFence));

    get_current_frame().deletionQueue.flush();

    // request image from the swapchain
    uint32_t swapchainImageIndex;
    VK_CHECK(
        vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &
            swapchainImageIndex));

    const auto cmd = get_current_frame().mainCommandBuffer;

    // now that we are sure that the commands finished executing, we can safely reset the buffer
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // begin the command buffer recording. We will use the buffer exactly once, so we tell vulkan
    const auto cmdBeginInfo = vkinit::command_buffer_begin_info(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    _drawExtent.width = _drawImage.imageExtent.width;
    _drawExtent.height = _drawImage.imageExtent.height;

    // start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main image into general layout, so we can write into it,
    // we will overwrite it all, so we don't care about what was the older layout
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd);

    // transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent,
                                _swapchainExtent);

    // set swapchain image layout to Attachment Optimal, so we can draw it
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // draw imgui into the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);


    // set swapchain image layout to Present, so we can show it on the screen
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue.
    // we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    // we will signal the _renderSemaphore, to signal that rendering has finished
    auto cmdInfo = vkinit::command_buffer_submit_info(cmd);

    auto waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                                                  get_current_frame().swapchain_semaphore);
    auto signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                                    get_current_frame().renderSemaphore);

    const auto submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution.
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame().renderFence));

    // prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that,
    // as its necessary that drawing commands have finished before the image is displayed to the user
    const VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &get_current_frame().renderSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &swapchainImageIndex
    };

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    // increase the number of frames drawn
    _frameNumber++;
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    // make vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Another stupid vulkan things")
                           .request_validation_layers(B_USE_VALIDATION_LAYERS)
                           .use_default_debug_messenger()
                           .require_api_version(1, 3, 0)
                           .build();

    // TODO: Error checking on Result<T> types!
    vkb::Instance vkb_instance = inst_ret.value();

    // grab the instance
    _instance = vkb_instance.instance;
    _debugMessenger = vkb_instance.debug_messenger;


    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13 = {
        .synchronization2 = true,
        .dynamicRendering = true
    };

    // vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12 = {
        .descriptorIndexing = true,
        .bufferDeviceAddress = true
    };
    // use vkbootstrap to select a gpu because I am lazy
    // GPU must support all requested features (obviously)
    vkb::PhysicalDeviceSelector selector{vkb_instance};
    vkb::PhysicalDevice physicalDevice = selector
                                         .set_minimum_version(1, 3)
                                         .set_required_features_13(features13)
                                         .set_required_features_12(features12)
                                         .set_surface(_surface)
                                         .select()
                                         .value();

    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the device handle used in the rest of the vulkan application
    _device = vkbDevice.device;
    _chosenGpu = physicalDevice.physical_device;

    // create graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = _chosenGpu,
        .device = _device,
        .instance = _instance
    };

    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]()
    {
        vmaDestroyAllocator(_allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    // draw image size wil match the window
    const VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    // hardcoding the draw format to 32-bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages = {};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    const VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages,
                                                                  drawImageExtent);
    const auto dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // for the draw image, allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocation = {
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };

    // allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocation, &_drawImage.image, &_drawImage.allocation, nullptr);
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocation, &_depthImage.image, &_depthImage.allocation, nullptr);

    // build an image-view for the draw image to use for rendering
    const VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image,
                                                                           VK_IMAGE_ASPECT_COLOR_BIT);

    const auto dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image,
                                                          VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    // add to the deletion queues
    _mainDeletionQueue.push_function([this]()
    {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
    });

    // not implemented yet :3
}

void VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers.
    const auto commandPoolInfo = vkinit::command_pool_create_info(
        _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (auto& [swapchain_semaphore, renderSemaphore, renderFence, commandPool, mainCommandBuffer, deletionQueue] : _frames)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &commandPool));

        // allocate the default command buffer that we will use for rendering
        auto cmdAllocInfo = vkinit::command_buffer_allocate_info(commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &mainCommandBuffer));
    }

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    const auto cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([this]()
    {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::draw_background(const VkCommandBuffer cmd)
{
    const auto& [name, pipeline, layout, data] = _backgroundEffects[_activeBackgroundEffect];

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors,
                            0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants),
                       &data);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size, so we need to divide by it
    vkCmdDispatch(cmd, static_cast<int>(std::ceil(_drawExtent.width / 16.0)),
                  static_cast<int>(std::ceil(_drawExtent.height / 16.0)), 1);
}


void VulkanEngine::init_sync_structures()
{
    // create synchronization structures
    // one fence to control when the gpu has finished rendering the frame
    // and 2 semaphores to synchronize rendering with swapchain
    // we want the fence to start signalled, so we can wait on it on the first frame
    const auto fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    const auto semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (auto& [swapchain_semaphore, renderSemaphore, renderFence, commandPool, mainCommandBuffer, deletionQueue] : _frames)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &swapchain_semaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &renderSemaphore));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([this]() { vkDestroyFence(_device, _immFence, nullptr); });
}


void VulkanEngine::create_swapchain(const uint32_t width, const uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGpu, _device, _surface};

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
                                  //.use_default_format_selection()
                                  .set_desired_format(VkSurfaceFormatKHR{
                                      .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
                                  })
                                  // use vsync present mode
                                  .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                  .set_desired_extent(width, height)
                                  .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                                  .build()
                                  .value();

    _swapchainExtent = vkbSwapchain.extent;
    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain() const
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy individual swapchain resources
    for (auto& imageView : _swapchainImageViews)
    {
        vkDestroyImageView(_device, imageView, nullptr);
    }
}

void VulkanEngine::init_descriptors()
{
    // create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}
    };

    _globalDescriptorAllocator.init_pool(_device, 10, sizes);

    // make the descriptor set layout for our compute draw
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // allocate a descriptor set for our draw image
    _drawImageDescriptors = _globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    VkDescriptorImageInfo imgInfo = {
        .imageView = _drawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };

    VkWriteDescriptorSet drawImageWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = _drawImageDescriptors,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &imgInfo
    };

    vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);
}

bool VulkanEngine::init_background_pipelines()
{
    VkPushConstantRange pushConstant = {

        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputePushConstants)
    };

    VkPipelineLayoutCreateInfo computeLayout = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &_drawImageDescriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant
    };

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    const VkPipelineShaderStageCreateInfo stageinfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = gradientShader,
        .pName = "main"
    };

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = stageinfo,
        .layout = _gradientPipelineLayout,
    };

    ComputeEffect gradient = {
        .name = "gradient",
        .layout = _gradientPipelineLayout,
        .data = {}
    };

    // default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device,VK_NULL_HANDLE,1,&computePipelineCreateInfo, nullptr, &gradient.pipeline))
    ;

    // change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky = {
        .name = "sky",
        .layout = _gradientPipelineLayout,
        .data = {}
    };

    // default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    // add the 2 background effects into the array
    _backgroundEffects.push_back(gradient);
    _backgroundEffects.push_back(sky);

    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);

    _mainDeletionQueue.push_function([this, sky, gradient]()
    {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    });
    return false;
}

void VulkanEngine::init_pipelines()
{
    // COMPUTE PIPELINES
    init_background_pipelines();

    // GRAPHICS PIPELINES
    init_mesh_pipeline();
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) const
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    const VkCommandBuffer cmd = _immCommandBuffer;

    const auto cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    std::move(function)(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    auto cmdInfo = vkinit::command_buffer_submit_info(cmd);
    const auto submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1000,
        .poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes)),
        .pPoolSizes = pool_sizes
    };

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {
        .Instance = _instance,
        .PhysicalDevice = _chosenGpu,
        .Device = _device,
        .Queue = _graphicsQueue,
        .DescriptorPool = imguiPool,
        .MinImageCount = 3,
        .ImageCount = 3,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .UseDynamicRendering = true
    };

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add destruction of imgui structures
    _mainDeletionQueue.push_function([this, imguiPool]()
    {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) const
{
    auto colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(const VkCommandBuffer cmd) const
{
    // begin a render pass, connected to our draw image
    auto colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr,
                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView,
                                                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    const VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    // draw mesh pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

    // set dynamic viewport and scissor
    VkViewport viewport;
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // draw loaded meshes
    const glm::mat4 view = glm::translate(glm::vec3{0, 0, -3});
    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), static_cast<float>(_drawExtent.width) / static_cast<float>(_drawExtent.height),
                                            10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    GPUDrawPushConstants push_constants;
    projection[1][1] *= -1;

    push_constants.worldMatrix = projection * view;


    push_constants.vertexBuffer = _testMeshes[2]->meshBuffers.vertexBufferAddress;
    vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants),
                       &push_constants);
    vkCmdBindIndexBuffer(cmd, _testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, _testMeshes[2]->surfaces[0].count, 1, _testMeshes[2]->surfaces[0].startIndex, 0, 0);

    vkCmdEndRendering(cmd);
}

AllocatedBuffer VulkanEngine::create_buffer(const size_t allocSize, const VkBufferUsageFlags usage,
                                            const VmaMemoryUsage memoryUsage) const
{
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = allocSize,
        .usage = usage,
    };

    const VmaAllocationCreateInfo vmaAllocInfo = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = memoryUsage,
    };

    AllocatedBuffer newBuffer;

    VK_CHECK(
        vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.
            info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer) const
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(const std::span<uint32_t> indices, const std::span<Vertex> vertices) const
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    // create vertex buffer
    newSurface.vertexBuffer = create_buffer(vertexBufferSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // find the address of the vertex buffer
    const VkBufferDeviceAddressInfo deviceAddressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newSurface.vertexBuffer.buffer
    };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAddressInfo);

    // create index buffer
    newSurface.indexBuffer = create_buffer(indexBufferSize,
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_GPU_ONLY);

    const auto stagingBuffer = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = stagingBuffer.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy(static_cast<char*>(data) + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd)
    {
        const VkBufferCopy vertexCopy = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertexBufferSize
        };

        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        const VkBufferCopy indexCopy = {
            .srcOffset = vertexBufferSize,
            .dstOffset = 0,
            .size = indexBufferSize
        };

        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });
    destroy_buffer(stagingBuffer);
    return newSurface;
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", _device, &meshFragShader))
    {
        fmt::print("Error when building the triangle fragment shader module");
    }
    else
    {
        fmt::print("Mesh fragment shader successfully loaded\n");
    }

    VkShaderModule meshVertShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle_mesh.vert.spv", _device, &meshVertShader))
    {
        fmt::print("Error when building the triangle vertex shader module");
    }
    else
    {
        fmt::print("Mesh vertex shader successfully loaded\n");
    }

    VkPushConstantRange bufferRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants)
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));
    PipelineBuilder pipelineBuilder;

    //use the triangle layout we created
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    //connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_shaders(meshVertShader, meshFragShader);
    //it will draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipelineBuilder.set_multisampling_none();
    //no blending
    pipelineBuilder.disable_blending();

    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //connect the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    //finally build the pipeline
    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    //clean structures
    vkDestroyShaderModule(_device, meshFragShader, nullptr);
    vkDestroyShaderModule(_device, meshVertShader, nullptr);

    _mainDeletionQueue.push_function([&]()
    {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
    });
}

void VulkanEngine::init_default_data()
{
    _testMeshes = loadGltfMeshes(this, R"(..\..\assets\basicmesh.glb)").value();

    _mainDeletionQueue.push_function([this]()
    {
        for (const auto& asset : _testMeshes)
        {
            destroy_buffer(asset->meshBuffers.indexBuffer);
            destroy_buffer(asset->meshBuffers.vertexBuffer);
        }
    });
}
