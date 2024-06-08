// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

// ReSharper disable CppUninitializedNonStaticDataMember
#pragma once

#include <vk_types.h>

#include <ranges>

#include "vk_descriptors.h"
#include "vk_loader.h"

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;
    // not really efficient for a lot of objects, but ok for this small thing

    void push_function(std::function<void()>&& function)
    {
        deletors.push_back(std::move(function));
    }

    void flush()
    {
        for (auto& deletor : std::ranges::reverse_view(deletors))
        {
            deletor(); // call functors
        }

        deletors.clear();
    }
};

struct FrameData
{
    VkSemaphore swapchain_semaphore, renderSemaphore;
    VkFence renderFence;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    DeletionQueue deletionQueue;
    GrowableDescriptorAllocator frameDescriptors;
};

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect
{
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine
{
public:
    VulkanEngine() = default;

    static VulkanEngine& Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //run main loop
    void run();
    [[nodiscard]] GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices) const;

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false) const;
    AllocatedImage create_image(const void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false) const;
    void destroy_image(const AllocatedImage& img) const;

private:
    bool _isInitialized{false};
    int _frameNumber{0};
    bool _stopRendering{false};
    bool _resizeRequested{false};
    VkExtent2D _windowExtent{1920, 1080};

    struct SDL_Window* _window{nullptr};

    VkInstance _instance; // Vulkan library handle
    VkDebugUtilsMessengerEXT _debugMessenger; // Vulkan debug output handle
    VkPhysicalDevice _chosenGpu; // GPU chosen as the default device
    VkDevice _device; // Vulkan device for commands
    VkSurfaceKHR _surface; // Vulkan window surface

    // FrameData
    FrameData _frames[FRAME_OVERLAP];
    FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    // Swapchain
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;
    VkExtent2D _drawExtent;
    float _renderScale = 1.f;

    DescriptorAllocator _globalDescriptorAllocator;

    GPUSceneData _sceneData;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipelineLayout _gradientPipelineLayout;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    std::vector<ComputeEffect> _backgroundEffects;
    int _activeBackgroundEffect{0};

    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;

    std::vector<std::shared_ptr<MeshAsset>> _testMeshes;

    //draw loop
    void draw();

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) const;
    void init_vulkan();
    void init_swapchain();
    void resize_swapchain();
    void init_commands();
    void init_sync_structures();

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain() const;

    void draw_background(VkCommandBuffer cmd);
    void init_descriptors();
    bool init_background_pipelines();
    void init_pipelines();

    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) const;

    void draw_geometry(VkCommandBuffer cmd);

    [[nodiscard]] AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const;
    void destroy_buffer(const AllocatedBuffer& buffer) const;

    void init_mesh_pipeline();
    void init_default_data();
};
