#include "skeletal.h"
#include "../utils/vk_check.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstring>
#include <algorithm>
#include <cmath>

VkVertexInputBindingDescription getSkinnedVertexBinding() {
    VkVertexInputBindingDescription b{};
    b.binding   = 0;
    b.stride    = sizeof(SkinnedVertex);
    b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return b;
}

std::array<VkVertexInputAttributeDescription, 5> getSkinnedVertexAttributes() {
    std::array<VkVertexInputAttributeDescription, 5> a{};
    a[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(SkinnedVertex, position) };
    a[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(SkinnedVertex, normal)   };
    a[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(SkinnedVertex, texCoord) };
    a[3] = { 3, 0, VK_FORMAT_R32G32B32A32_UINT,   offsetof(SkinnedVertex, jointIds) };
    a[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(SkinnedVertex, weights)  };
    return a;
}

// ── Test bone chain ────────────────────────────────────────────────────────
// A simple skinned mesh: a tapered cylinder along +Y, 4 bones evenly spaced.
// Each vertex is weighted to the 2 nearest bones for smooth deformation.

SkinnedMesh createTestBoneChain() {
    SkinnedMesh mesh;

    constexpr int   BONES   = 4;
    constexpr int   RINGS   = 12;        // along Y
    constexpr int   SEGS    = 12;        // around
    constexpr float HEIGHT  = 3.0f;
    constexpr float R_TOP   = 0.15f;
    constexpr float R_BOT   = 0.5f;

    // ── Build skeleton (4 joints along Y, each parent of next) ──────────
    mesh.skeleton.joints.resize(BONES);
    for (int i = 0; i < BONES; ++i) {
        Joint& j = mesh.skeleton.joints[i];
        j.name   = "bone" + std::to_string(i);
        j.parent = i - 1;
        float y  = (i == 0) ? 0.0f : (HEIGHT / static_cast<float>(BONES - 1));
        j.localT = glm::vec3(0.0f, y, 0.0f);
        j.localR = glm::quat(1, 0, 0, 0);
        j.localS = glm::vec3(1.0f);
    }
    // Compute inverse bind matrices from the rest pose
    std::vector<glm::mat4> bindWorld(BONES, glm::mat4(1.0f));
    for (int i = 0; i < BONES; ++i) {
        glm::mat4 local =
            glm::translate(glm::mat4(1.0f), mesh.skeleton.joints[i].localT) *
            glm::mat4_cast(mesh.skeleton.joints[i].localR) *
            glm::scale(glm::mat4(1.0f), mesh.skeleton.joints[i].localS);
        bindWorld[i] = (mesh.skeleton.joints[i].parent < 0)
                       ? local
                       : bindWorld[mesh.skeleton.joints[i].parent] * local;
        mesh.skeleton.joints[i].inverseBind = glm::inverse(bindWorld[i]);
    }

    // ── Build vertices (cylinder shell) ─────────────────────────────────
    for (int r = 0; r <= RINGS; ++r) {
        float tY = static_cast<float>(r) / RINGS;
        float y  = tY * HEIGHT;
        float radius = glm::mix(R_BOT, R_TOP, tY);

        // Map y to bone weights: between the two nearest joints
        float boneSpan  = HEIGHT / static_cast<float>(BONES - 1);
        float boneIdxF  = y / boneSpan;
        int   bi0       = std::clamp(static_cast<int>(std::floor(boneIdxF)), 0, BONES - 1);
        int   bi1       = std::clamp(bi0 + 1, 0, BONES - 1);
        float w1        = (bi0 == bi1) ? 0.0f : (boneIdxF - static_cast<float>(bi0));
        float w0        = 1.0f - w1;

        for (int s = 0; s < SEGS; ++s) {
            float tA = static_cast<float>(s) / SEGS * 2.0f * 3.14159265f;
            float cx = std::cos(tA), cz = std::sin(tA);
            SkinnedVertex v{};
            v.position = glm::vec3(cx * radius, y, cz * radius);
            v.normal   = glm::normalize(glm::vec3(cx, 0.1f, cz));
            v.texCoord = glm::vec2(static_cast<float>(s) / SEGS, tY);
            v.jointIds = glm::uvec4(bi0, bi1, 0, 0);
            v.weights  = glm::vec4(w0, w1, 0.0f, 0.0f);
            mesh.vertices.push_back(v);
        }
    }
    for (int r = 0; r < RINGS; ++r) {
        for (int s = 0; s < SEGS; ++s) {
            int s1 = (s + 1) % SEGS;
            uint32_t a = r       * SEGS + s;
            uint32_t b = r       * SEGS + s1;
            uint32_t c = (r + 1) * SEGS + s;
            uint32_t d = (r + 1) * SEGS + s1;
            mesh.indices.push_back(a); mesh.indices.push_back(c); mesh.indices.push_back(b);
            mesh.indices.push_back(b); mesh.indices.push_back(c); mesh.indices.push_back(d);
        }
    }

    // AABB conservatively
    mesh.aabbMin = glm::vec3(-R_BOT, 0.0f,    -R_BOT);
    mesh.aabbMax = glm::vec3( R_BOT, HEIGHT,   R_BOT);

    // ── Build a wave animation ──────────────────────────────────────────
    Animation a;
    a.name     = "wave";
    a.duration = 2.0f;

    for (int i = 1; i < BONES; ++i) {
        AnimChannel ch;
        ch.joint = i;
        ch.path  = AnimChannelPath::Rotation;

        const int KEYS = 16;
        for (int k = 0; k <= KEYS; ++k) {
            float t = static_cast<float>(k) / KEYS * a.duration;
            float phase = t * 3.0f + static_cast<float>(i) * 0.6f;
            float angle = std::sin(phase) * 0.35f; // ~20 degrees
            glm::quat q = glm::angleAxis(angle, glm::vec3(0, 0, 1));
            ch.sampler.times.push_back(t);
            ch.sampler.values.push_back(glm::vec4(q.x, q.y, q.z, q.w));
        }
        a.channels.push_back(ch);
    }
    mesh.animations.push_back(a);

    return mesh;
}

// ── Upload / destroy ───────────────────────────────────────────────────────

void uploadSkinnedMesh(SkinnedMesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                       VkCommandPool commandPool, VkQueue queue)
{
    // Vertex buffer via staging
    VkDeviceSize vbSize = sizeof(SkinnedVertex) * mesh.vertices.size();
    auto staging = createBuffer(physicalDevice, device, vbSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, mesh.vertices.data(), vbSize);

    mesh.vertexBuffer = createBuffer(physicalDevice, device, vbSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    copyBuffer(device, commandPool, queue, staging.buffer, mesh.vertexBuffer.buffer, vbSize);
    destroyBuffer(device, staging);

    // Index buffer
    VkDeviceSize ibSize = sizeof(uint32_t) * mesh.indices.size();
    staging = createBuffer(physicalDevice, device, ibSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, mesh.indices.data(), ibSize);

    mesh.indexBuffer = createBuffer(physicalDevice, device, ibSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    copyBuffer(device, commandPool, queue, staging.buffer, mesh.indexBuffer.buffer, ibSize);
    destroyBuffer(device, staging);
}

void destroySkinnedMesh(VkDevice device, SkinnedMesh& mesh) {
    destroyBuffer(device, mesh.vertexBuffer);
    destroyBuffer(device, mesh.indexBuffer);
}

// ── Animation evaluation ────────────────────────────────────────────────────

static glm::vec3 sampleVec3(const AnimSampler& s, float t) {
    if (s.times.empty()) return glm::vec3(0.0f);
    if (t <= s.times.front()) return glm::vec3(s.values.front());
    if (t >= s.times.back())  return glm::vec3(s.values.back());
    for (size_t i = 1; i < s.times.size(); ++i) {
        if (t < s.times[i]) {
            float u = (t - s.times[i - 1]) / (s.times[i] - s.times[i - 1]);
            glm::vec3 a = glm::vec3(s.values[i - 1]);
            glm::vec3 b = glm::vec3(s.values[i]);
            return glm::mix(a, b, u);
        }
    }
    return glm::vec3(s.values.back());
}

static glm::quat sampleQuat(const AnimSampler& s, float t) {
    if (s.times.empty()) return glm::quat(1, 0, 0, 0);
    auto pickQuat = [](const glm::vec4& v) { return glm::quat(v.w, v.x, v.y, v.z); };
    if (t <= s.times.front()) return pickQuat(s.values.front());
    if (t >= s.times.back())  return pickQuat(s.values.back());
    for (size_t i = 1; i < s.times.size(); ++i) {
        if (t < s.times[i]) {
            float u = (t - s.times[i - 1]) / (s.times[i] - s.times[i - 1]);
            return glm::normalize(glm::slerp(pickQuat(s.values[i - 1]),
                                             pickQuat(s.values[i]), u));
        }
    }
    return pickQuat(s.values.back());
}

void computeBoneMatrices(const Skeleton& skeleton, const Animation& anim, float t,
                         BonePalette& outPalette)
{
    const size_t N = skeleton.joints.size();

    std::vector<glm::vec3> T(N), S(N);
    std::vector<glm::quat> R(N);
    for (size_t i = 0; i < N; ++i) {
        T[i] = skeleton.joints[i].localT;
        R[i] = skeleton.joints[i].localR;
        S[i] = skeleton.joints[i].localS;
    }

    // Loop through animation duration
    float loopT = (anim.duration > 0.0f) ? std::fmod(t, anim.duration) : t;

    for (const auto& ch : anim.channels) {
        if (ch.joint < 0 || static_cast<size_t>(ch.joint) >= N) continue;
        switch (ch.path) {
            case AnimChannelPath::Translation:
                T[ch.joint] = sampleVec3(ch.sampler, loopT); break;
            case AnimChannelPath::Rotation:
                R[ch.joint] = sampleQuat(ch.sampler, loopT); break;
            case AnimChannelPath::Scale:
                S[ch.joint] = sampleVec3(ch.sampler, loopT); break;
        }
    }

    std::vector<glm::mat4> world(N, glm::mat4(1.0f));
    for (size_t i = 0; i < N; ++i) {
        glm::mat4 local =
            glm::translate(glm::mat4(1.0f), T[i]) *
            glm::mat4_cast(R[i]) *
            glm::scale(glm::mat4(1.0f), S[i]);
        int p = skeleton.joints[i].parent;
        world[i] = (p < 0) ? local : world[p] * local;
    }

    for (size_t i = 0; i < N && i < MAX_BONES; ++i) {
        outPalette.bones[i] = world[i] * skeleton.joints[i].inverseBind;
    }
    // Fill any unused bone slots with identity (safety for shader array indexing)
    for (size_t i = N; i < MAX_BONES; ++i) {
        outPalette.bones[i] = glm::mat4(1.0f);
    }
}
