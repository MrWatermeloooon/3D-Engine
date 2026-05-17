#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <string>

class ResourceManager;

// A small visual node graph that composes the engine's fixed PBR
// MaterialComponent (base color / metallic / roughness / parallax + texture
// slots) WITHOUT hand-editing shaders. It does not generate GLSL — that
// (arbitrary shader codegen) is the separate "material graph → shader" hard
// addon. Here the graph evaluates on the CPU to concrete MaterialComponent
// values and writes them onto the selected entity, so the existing render
// path is reused unchanged.
//
// Linking model: every node produces one typed value (float or color).
// Composite/Output inputs pick their source via a dropdown (Inline value, or
// another compatible node) — deterministic and robust, drawn as link lines
// between draggable node boxes.
class MaterialGraph {
public:
    MaterialGraph();
    // Renders the editor window contents and applies/loads to the selected
    // entity's MaterialComponent on demand. Returns true if "Apply" wrote a
    // new MaterialComponent this frame (so the caller can commit an undo).
    bool draw(entt::registry& reg, entt::entity selected,
              ResourceManager& resources);

private:
    enum class Type { Color, Scalar, MixColor, MulScalar, Output };

    struct Node {
        int     id;
        Type    type;
        float   x, y;                 // canvas position
        glm::vec4 color{0.8f, 0.8f, 0.8f, 1.0f};
        float   scalar = 0.5f;
        // Composite/Output input sources: node id, or -1 = use inline value.
        int     inA = -1, inB = -1, inT = -1;     // Mix/Mul use inA/inB(/inT)
        // Output-only inline fields:
        int     baseColorSrc = -1;    // node id producing a color, or -1
        int     metallicSrc  = -1;
        int     roughnessSrc = -1;
        glm::vec4 baseColor{0.8f, 0.8f, 0.8f, 1.0f};
        float   metallic   = 0.0f;
        float   roughness  = 0.5f;
        float   parallax   = 0.0f;
        unsigned albedoTex = 0, normalTex = 0, heightTex = 0;
    };

    Node* find(int id);
    glm::vec4 evalColor(int nodeId, int depth);
    float     evalScalar(int nodeId, int depth);
    int       addNode(Type t, float x, float y);

    std::vector<Node> m_nodes;
    int  m_nextId   = 1;
    int  m_outputId = 0;
    bool m_seeded   = false;
};
