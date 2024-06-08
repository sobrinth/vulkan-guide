#pragma once
#include <vk_types.h>
#include <unordered_map>
#include <filesystem>

#include "vk_descriptors.h"


struct GLTFMaterial
{
    MaterialInstance data;
};

struct GeoSurface
{
    uint32_t startIndex;
    uint32_t count;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset
{
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

class VulkanEngine;

struct LoadedGLTF : public IRenderable
{
    // storage for all the data on a given glTF fille
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // nodes that don't have a parent, for iterating through the file in tree order
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    GrowableDescriptorAllocator descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine* creator;

    ~LoadedGLTF() { clearAll(); };

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;

private:
    void clearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath);
