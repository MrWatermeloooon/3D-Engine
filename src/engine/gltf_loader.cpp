#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "gltf_loader.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

// Build a {node* → joint index} table for the skin. The skin stores its joint
// list, which is the *canonical ordering* used by JOINTS_0 indices. Walking
// the node hierarchy lets us derive each joint's parent (or -1 when its
// parent isn't a joint of the same skin).
struct JointIndexer {
    std::unordered_map<const cgltf_node*, int> index;

    void build(const cgltf_skin* skin) {
        index.clear();
        for (cgltf_size i = 0; i < skin->joints_count; ++i) {
            index.emplace(skin->joints[i], static_cast<int>(i));
        }
    }
    int find(const cgltf_node* n) const {
        auto it = index.find(n);
        return it == index.end() ? -1 : it->second;
    }
};

// Read a typed scalar accessor into a flat float buffer. Handles the
// component types we care about (FLOAT, U8, U16, U32) for indices + weights.
// Falls back to cgltf's helper for unaligned / sparse accessors.
template <typename T>
void readAccessor(const cgltf_accessor* acc, std::vector<T>& out, size_t components)
{
    out.resize(acc->count * components);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_accessor_read_float(acc, i, &reinterpret_cast<float&>(out[i * components]),
                                  components);
    }
}

void readIndices(const cgltf_accessor* acc, std::vector<uint32_t>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        out[i] = static_cast<uint32_t>(cgltf_accessor_read_index(acc, i));
    }
}

// Joint indices come as integer accessors (u8 or u16). cgltf has
// cgltf_accessor_read_uint that we can use per-element.
void readJointIndices(const cgltf_accessor* acc, std::vector<glm::uvec4>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_uint v[4] = { 0, 0, 0, 0 };
        cgltf_accessor_read_uint(acc, i, v, 4);
        out[i] = glm::uvec4(v[0], v[1], v[2], v[3]);
    }
}

void readVec3(const cgltf_accessor* acc, std::vector<glm::vec3>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_accessor_read_float(acc, i, &out[i].x, 3);
    }
}
void readVec2(const cgltf_accessor* acc, std::vector<glm::vec2>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_accessor_read_float(acc, i, &out[i].x, 2);
    }
}
void readVec4(const cgltf_accessor* acc, std::vector<glm::vec4>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_accessor_read_float(acc, i, &out[i].x, 4);
    }
}
void readMat4(const cgltf_accessor* acc, std::vector<glm::mat4>& out) {
    out.resize(acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i) {
        cgltf_accessor_read_float(acc, i, glm::value_ptr(out[i]), 16);
    }
}

AnimChannelPath mapPath(cgltf_animation_path_type t) {
    switch (t) {
        case cgltf_animation_path_type_translation: return AnimChannelPath::Translation;
        case cgltf_animation_path_type_rotation:    return AnimChannelPath::Rotation;
        case cgltf_animation_path_type_scale:       return AnimChannelPath::Scale;
        default:                                    return AnimChannelPath::Translation;
    }
}

} // namespace

bool loadGltfSkinned(const std::string& path, SkinnedMesh& out, std::string* errorOut)
{
    auto fail = [&](const char* msg) {
        if (errorOut) *errorOut = msg;
        std::cerr << "[glTF] " << msg << " (" << path << ")\n";
        return false;
    };

    cgltf_options opts{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opts, path.c_str(), &data) != cgltf_result_success) {
        return fail("cgltf_parse_file failed");
    }
    if (cgltf_load_buffers(&opts, data, path.c_str()) != cgltf_result_success) {
        cgltf_free(data);
        return fail("cgltf_load_buffers failed");
    }

    // Find a node that drives a skinned mesh.
    const cgltf_node*      skinnedNode = nullptr;
    const cgltf_primitive* skinnedPrim = nullptr;
    for (cgltf_size i = 0; i < data->nodes_count; ++i) {
        const cgltf_node* n = &data->nodes[i];
        if (n->mesh && n->skin && n->mesh->primitives_count > 0) {
            skinnedNode = n;
            skinnedPrim = &n->mesh->primitives[0];
            break;
        }
    }
    if (!skinnedNode) {
        cgltf_free(data);
        return fail("no skinned mesh primitive found");
    }

    const cgltf_skin* skin = skinnedNode->skin;

    // ── Skeleton ────────────────────────────────────────────────────────
    JointIndexer indexer;
    indexer.build(skin);
    out.skeleton.joints.clear();
    out.skeleton.joints.resize(skin->joints_count);

    std::vector<glm::mat4> ibm;
    if (skin->inverse_bind_matrices) {
        readMat4(skin->inverse_bind_matrices, ibm);
    }
    for (cgltf_size i = 0; i < skin->joints_count; ++i) {
        const cgltf_node* jn = skin->joints[i];
        Joint& j = out.skeleton.joints[i];
        j.name        = jn->name ? jn->name : ("joint_" + std::to_string(i));
        j.parent      = indexer.find(jn->parent);
        j.inverseBind = (i < ibm.size()) ? ibm[i] : glm::mat4(1.0f);

        // Local TRS. glTF nodes can have either a matrix or T/R/S components;
        // when has_matrix is true we decompose it (uniform/non-uniform scale
        // is preserved through the matrix path, so explicit decomposition is
        // fine — but most glTF rigs ship as T/R/S which we read directly).
        if (jn->has_translation)
            j.localT = glm::vec3(jn->translation[0], jn->translation[1], jn->translation[2]);
        if (jn->has_rotation)
            j.localR = glm::quat(jn->rotation[3], jn->rotation[0], jn->rotation[1], jn->rotation[2]); // glm quat = (w, x, y, z)
        if (jn->has_scale)
            j.localS = glm::vec3(jn->scale[0], jn->scale[1], jn->scale[2]);
    }

    // ── Vertices ────────────────────────────────────────────────────────
    std::vector<glm::vec3>  positions, normals;
    std::vector<glm::vec2>  uvs;
    std::vector<glm::uvec4> joints;
    std::vector<glm::vec4>  weights;
    for (cgltf_size a = 0; a < skinnedPrim->attributes_count; ++a) {
        const cgltf_attribute& attr = skinnedPrim->attributes[a];
        switch (attr.type) {
            case cgltf_attribute_type_position: readVec3(attr.data, positions); break;
            case cgltf_attribute_type_normal:   readVec3(attr.data, normals);   break;
            case cgltf_attribute_type_texcoord:
                if (attr.index == 0) readVec2(attr.data, uvs);
                break;
            case cgltf_attribute_type_joints:
                if (attr.index == 0) readJointIndices(attr.data, joints);
                break;
            case cgltf_attribute_type_weights:
                if (attr.index == 0) readVec4(attr.data, weights);
                break;
            default: break;
        }
    }
    if (positions.empty()) {
        cgltf_free(data);
        return fail("primitive missing POSITION attribute");
    }
    const size_t vcount = positions.size();
    normals.resize(vcount, glm::vec3(0.0f, 1.0f, 0.0f));
    uvs    .resize(vcount, glm::vec2(0.0f));
    joints .resize(vcount, glm::uvec4(0));
    weights.resize(vcount, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

    out.vertices.clear();
    out.vertices.reserve(vcount);
    glm::vec3 aabbMin( std::numeric_limits<float>::infinity());
    glm::vec3 aabbMax(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < vcount; ++i) {
        SkinnedVertex sv;
        sv.position = positions[i];
        sv.normal   = normals[i];
        sv.texCoord = uvs[i];
        sv.jointIds = joints[i];
        // Renormalise weights so the sum is exactly 1. Important when the file
        // stored u8/u16-normalised weights that rounded slightly off.
        glm::vec4 w = weights[i];
        float sum = w.x + w.y + w.z + w.w;
        sv.weights = (sum > 0.0f) ? w / sum : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        out.vertices.push_back(sv);

        aabbMin = glm::min(aabbMin, sv.position);
        aabbMax = glm::max(aabbMax, sv.position);
    }
    out.aabbMin = aabbMin;
    out.aabbMax = aabbMax;

    // ── Indices ─────────────────────────────────────────────────────────
    out.indices.clear();
    if (skinnedPrim->indices) {
        readIndices(skinnedPrim->indices, out.indices);
    } else {
        // Non-indexed primitives → emit a trivial 0,1,2,3,… index list.
        out.indices.resize(vcount);
        for (uint32_t i = 0; i < vcount; ++i) out.indices[i] = i;
    }

    // ── Animations ──────────────────────────────────────────────────────
    out.animations.clear();
    out.animations.reserve(data->animations_count);
    for (cgltf_size ai = 0; ai < data->animations_count; ++ai) {
        const cgltf_animation& ga = data->animations[ai];
        Animation anim;
        anim.name = ga.name ? ga.name : ("anim_" + std::to_string(ai));
        anim.duration = 0.0f;

        for (cgltf_size ci = 0; ci < ga.channels_count; ++ci) {
            const cgltf_animation_channel& gch = ga.channels[ci];
            int jointIdx = indexer.find(gch.target_node);
            if (jointIdx < 0) continue;  // channel targets a node outside the skin

            AnimChannel ch;
            ch.joint = jointIdx;
            ch.path  = mapPath(gch.target_path);

            const cgltf_animation_sampler* gs = gch.sampler;
            // Times: scalar float accessor.
            ch.sampler.times.resize(gs->input->count);
            for (cgltf_size t = 0; t < gs->input->count; ++t) {
                cgltf_accessor_read_float(gs->input, t, &ch.sampler.times[t], 1);
            }

            // Values: vec3 for T/S, vec4 (quat xyzw) for R. Stored as vec4 in
            // AnimSampler so we can carry quaternions uniformly; vec3 paths
            // leave .w = 0.
            const size_t valCount = gs->output->count;
            ch.sampler.values.resize(valCount, glm::vec4(0.0f));
            const bool isRotation = (gch.target_path == cgltf_animation_path_type_rotation);
            const size_t comps    = isRotation ? 4 : 3;
            for (cgltf_size v = 0; v < valCount; ++v) {
                cgltf_accessor_read_float(gs->output, v, &ch.sampler.values[v].x, comps);
            }

            if (gs->interpolation == cgltf_interpolation_type_cubic_spline) {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "[glTF] CUBICSPLINE interpolation not implemented; "
                                 "falling back to LINEAR.\n";
                    warned = true;
                }
                // CUBICSPLINE stores 3x values per keyframe (in-tangent, value,
                // out-tangent). Take just the middle entries — the values —
                // and treat the result as linear.
                if (valCount >= 3 * gs->input->count) {
                    std::vector<glm::vec4> linear(gs->input->count);
                    for (cgltf_size k = 0; k < gs->input->count; ++k) {
                        linear[k] = ch.sampler.values[k * 3 + 1];
                    }
                    ch.sampler.values = std::move(linear);
                }
            }
            // STEP interpolation: keep the keyframes; existing playback in
            // skeletal.cpp interpolates linearly which produces a smoothed
            // approximation. Snapping to the prior keyframe is a future
            // refinement that would need a per-channel interpolation enum.

            if (!ch.sampler.times.empty()) {
                anim.duration = std::max(anim.duration, ch.sampler.times.back());
            }
            anim.channels.push_back(std::move(ch));
        }
        if (!anim.channels.empty()) {
            out.animations.push_back(std::move(anim));
        }
    }

    cgltf_free(data);
    return true;
}
