#include <vk_descriptors.h>

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type)
{
    const VkDescriptorSetLayoutBinding newbind = {
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1
    };

    bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext,
                                                     VkDescriptorSetLayoutCreateFlags flags)
{
    for (auto& b : bindings)
    {
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = pNext,
        .flags = flags,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios)
    {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type, .descriptorCount = static_cast<uint32_t>(ratio.ratio * static_cast<float>(maxSets))
        });
    }

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device) const
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device) const
{
    vkDestroyDescriptorPool(device, pool, nullptr);
}


VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout) const
{
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };

    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

    return ds;
}

void GrowableDescriptorAllocator::init(const VkDevice device, const uint32_t initialSets, const std::span<PoolSizeRatio> poolRatios)
{
    _ratios.clear();

    for (auto r : poolRatios)
    {
        _ratios.push_back(r);
    }
    const auto pool = create_pool(device, initialSets, poolRatios);
    _setsPerPool = static_cast<uint32_t>(initialSets * 1.5); // grow it next allocation

    _readyPools.push_back(pool);
}

void GrowableDescriptorAllocator::clear_pools(const VkDevice device)
{
    for (const auto p : _readyPools)
    {
        vkResetDescriptorPool(device, p, 0);
    }
    for (auto p : _fullPools)
    {
        vkResetDescriptorPool(device, p, 0);
        _readyPools.push_back(p); // pools are no longer full, therefore they are ready again
    }
}

void GrowableDescriptorAllocator::destroy_pools(const VkDevice device)
{
    for (const auto p : _readyPools)
    {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    _readyPools.clear();

    for (const auto p : _fullPools)
    {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    _fullPools.clear();
}

VkDescriptorSet GrowableDescriptorAllocator::allocate(const VkDevice device, const VkDescriptorSetLayout layout, void* pNext)
{
    // get or create a pool to allocate from
    auto poolToUse = get_pool(device);

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = pNext,
        .descriptorPool = poolToUse,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };

    VkDescriptorSet ds;

    // allocation failed. Try again
    if (const VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &ds); result == VK_ERROR_OUT_OF_POOL_MEMORY || result ==
        VK_ERROR_FRAGMENTED_POOL)
    {
        _fullPools.push_back(poolToUse);

        poolToUse = get_pool(device);

        allocInfo.descriptorPool = poolToUse;

        // this should not fail a second time. If it does -> panic and crash!
        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));
    }
    _readyPools.push_back(poolToUse);
    return ds;
}

VkDescriptorPool GrowableDescriptorAllocator::get_pool(const VkDevice device)
{
    VkDescriptorPool pool;
    if (!_readyPools.empty())
    {
        pool = _readyPools.back();
        _readyPools.pop_back();
    }
    else
    {
        pool = create_pool(device, _setsPerPool, _ratios);

        _setsPerPool = static_cast<uint32_t>(_setsPerPool * 1.5);
        if (_setsPerPool > 4092)
        {
            _setsPerPool = 4092;
        }
    }

    return pool;
}

VkDescriptorPool GrowableDescriptorAllocator::create_pool(const VkDevice device, const uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto [type, ratio] : poolRatios)
    {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = type,
            .descriptorCount = static_cast<uint32_t>(ratio * setCount)
        });
    }

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = setCount,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
    return pool;
}
