#include "material_graph.h"
#include "components.h"

#include <imgui.h>
#include <algorithm>
#include <utility>
#include <string>
#include <cstdio>

MaterialGraph::MaterialGraph() = default;

MaterialGraph::Node* MaterialGraph::find(int id) {
    for (auto& n : m_nodes) if (n.id == id) return &n;
    return nullptr;
}

int MaterialGraph::addNode(Type t, float x, float y) {
    Node n;
    n.id = m_nextId++;
    n.type = t;
    n.x = x; n.y = y;
    m_nodes.push_back(n);
    return n.id;
}

// Resolve a node's value. depth guards against cycles introduced by the
// source dropdowns.
glm::vec4 MaterialGraph::evalColor(int nodeId, int depth) {
    if (depth > 16) return glm::vec4(1.0f);
    Node* n = find(nodeId);
    if (!n) return glm::vec4(1.0f);
    switch (n->type) {
        case Type::Color: return n->color;
        case Type::MixColor: {
            glm::vec4 a = n->inA >= 0 ? evalColor(n->inA, depth + 1) : n->color;
            glm::vec4 b = n->inB >= 0 ? evalColor(n->inB, depth + 1)
                                      : glm::vec4(1.0f);
            float t = n->inT >= 0 ? evalScalar(n->inT, depth + 1) : n->scalar;
            return glm::mix(a, b, glm::clamp(t, 0.0f, 1.0f));
        }
        default: return glm::vec4(glm::vec3(evalScalar(nodeId, depth)), 1.0f);
    }
}

float MaterialGraph::evalScalar(int nodeId, int depth) {
    if (depth > 16) return 0.0f;
    Node* n = find(nodeId);
    if (!n) return 0.0f;
    switch (n->type) {
        case Type::Scalar: return n->scalar;
        case Type::MulScalar: {
            float a = n->inA >= 0 ? evalScalar(n->inA, depth + 1) : n->scalar;
            float b = n->inB >= 0 ? evalScalar(n->inB, depth + 1) : 1.0f;
            return a * b;
        }
        case Type::Color:
        case Type::MixColor: return evalColor(nodeId, depth).x;
        default: return n->scalar;
    }
}

namespace {
    const char* typeName(int t) {
        switch (t) {
            case 0: return "Color";
            case 1: return "Scalar";
            case 2: return "Mix(Color)";
            case 3: return "Mul(Scalar)";
            default: return "Output";
        }
    }
    void sourceCombo(const char* label, int& src,
                     const std::vector<std::pair<int, int>>& opts) {
        std::string cur = "Inline";
        for (auto& o : opts)
            if (o.first == src) cur = std::string(typeName(o.second)) +
                                      " #" + std::to_string(o.first);
        if (ImGui::BeginCombo(label, cur.c_str())) {
            if (ImGui::Selectable("Inline", src < 0)) src = -1;
            for (auto& o : opts) {
                std::string nm = std::string(typeName(o.second)) + " #" +
                                 std::to_string(o.first);
                if (ImGui::Selectable(nm.c_str(), src == o.first))
                    src = o.first;
            }
            ImGui::EndCombo();
        }
    }

    // Known PBR presets: linear base colour, metallic, roughness. Metals use
    // their measured specular albedo with metallic=1; dielectrics metallic=0.
    struct Preset { const char* name; float r,g,b, metallic, rough; };
    const Preset kPresets[] = {
        { "Gold",          1.000f, 0.766f, 0.336f, 1.0f, 0.22f },
        { "Silver",        0.972f, 0.960f, 0.915f, 1.0f, 0.18f },
        { "Copper",        0.955f, 0.637f, 0.538f, 1.0f, 0.30f },
        { "Chrome",        0.550f, 0.556f, 0.554f, 1.0f, 0.06f },
        { "Iron",          0.560f, 0.570f, 0.580f, 1.0f, 0.45f },
        { "Aluminium",     0.913f, 0.921f, 0.925f, 1.0f, 0.32f },
        { "Plastic White", 0.900f, 0.900f, 0.900f, 0.0f, 0.35f },
        { "Plastic Red",   0.800f, 0.090f, 0.090f, 0.0f, 0.40f },
        { "Plastic Black", 0.040f, 0.040f, 0.040f, 0.0f, 0.45f },
        { "Glossy Paint",  0.100f, 0.300f, 0.850f, 0.0f, 0.12f },
        { "Rubber",        0.050f, 0.050f, 0.050f, 0.0f, 0.90f },
        { "Ceramic",       0.930f, 0.930f, 0.900f, 0.0f, 0.18f },
    };
}

bool MaterialGraph::draw(entt::registry& reg, entt::entity selected,
                         ResourceManager& /*resources*/) {
    bool applied = false;
    if (!m_seeded) {
        m_outputId = addNode(Type::Output, 0, 0);
        m_seeded = true;
    }
    Node* out = find(m_outputId);
    bool hasSel = selected != entt::null && reg.valid(selected) &&
                  reg.all_of<MaterialComponent>(selected);

    if (!hasSel) {
        ImGui::TextDisabled("Select an entity with a material.");
        return false;
    }
    auto& mat = reg.get<MaterialComponent>(selected);

    // ── Material presets (the quick path: make it gold / plastic / …) ───
    ImGui::SeparatorText("Presets");
    static int presetIdx = 0;
    ImGui::SetNextItemWidth(180);
    if (ImGui::BeginCombo("##preset", kPresets[presetIdx].name)) {
        for (int i = 0; i < IM_ARRAYSIZE(kPresets); ++i)
            if (ImGui::Selectable(kPresets[i].name, presetIdx == i))
                presetIdx = i;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply preset")) {
        const Preset& p = kPresets[presetIdx];
        mat.color     = glm::vec4(p.r, p.g, p.b, 1.0f);
        mat.metallic  = p.metallic;
        mat.roughness = p.rough;
        if (out) {                       // mirror into the editor
            out->baseColor = mat.color;
            out->metallic  = mat.metallic;
            out->roughness = mat.roughness;
            out->baseColorSrc = out->metallicSrc = out->roughnessSrc = -1;
        }
        applied = true;
    }

    // ── Direct PBR (always editable, no graph needed) ───────────────────
    ImGui::SeparatorText("PBR parameters");
    if (out) {
        ImGui::ColorEdit3("Base color", &out->baseColor.x);
        ImGui::SliderFloat("Metallic",  &out->metallic,  0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &out->roughness, 0.04f, 1.0f);
        ImGui::SliderFloat("Parallax",  &out->parallax,  0.0f, 0.2f);
        if (ImGui::Button("Load from selected")) {
            out->baseColor = mat.color;
            out->metallic  = mat.metallic;
            out->roughness = mat.roughness;
            out->parallax  = mat.parallaxScale;
            out->baseColorSrc = out->metallicSrc = out->roughnessSrc = -1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply to selected")) {
            mat.color     = out->baseColorSrc >= 0
                ? evalColor(out->baseColorSrc, 0) : out->baseColor;
            mat.metallic  = glm::clamp(out->metallicSrc >= 0
                ? evalScalar(out->metallicSrc, 0) : out->metallic, 0.0f, 1.0f);
            mat.roughness = glm::clamp(out->roughnessSrc >= 0
                ? evalScalar(out->roughnessSrc, 0) : out->roughness,
                0.04f, 1.0f);
            mat.parallaxScale = out->parallax;
            // NOTE: texture handles are deliberately left untouched so the
            // entity keeps its existing albedo/normal/height maps.
            applied = true;
        }
    }

    // ── Optional node graph feeding the Output sources ──────────────────
    ImGui::SeparatorText("Node graph (optional)");
    ImGui::TextDisabled("Build value nodes, then set an Output source below.");
    if (ImGui::Button("Add Color"))  addNode(Type::Color, 0, 0);
    ImGui::SameLine();
    if (ImGui::Button("Add Scalar")) addNode(Type::Scalar, 0, 0);
    ImGui::SameLine();
    if (ImGui::Button("Add Mix"))    addNode(Type::MixColor, 0, 0);
    ImGui::SameLine();
    if (ImGui::Button("Add Mul"))    addNode(Type::MulScalar, 0, 0);

    std::vector<std::pair<int,int>> colorOpts, scalarOpts;
    for (auto& n : m_nodes) {
        if (n.type == Type::Color || n.type == Type::MixColor)
            colorOpts.push_back({ n.id, static_cast<int>(n.type) });
        if (n.type == Type::Scalar || n.type == Type::MulScalar)
            scalarOpts.push_back({ n.id, static_cast<int>(n.type) });
    }

    int deleteId = -1;
    for (auto& n : m_nodes) {
        if (n.type == Type::Output) continue;   // Output shown above
        ImGui::PushID(n.id);
        char hdr[48];
        std::snprintf(hdr, sizeof(hdr), "%s #%d",
                      typeName(static_cast<int>(n.type)), n.id);
        if (ImGui::CollapsingHeader(hdr, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();
            switch (n.type) {
                case Type::Color:
                    ImGui::ColorEdit3("Color", &n.color.x);
                    break;
                case Type::Scalar:
                    ImGui::SliderFloat("Value", &n.scalar, 0.0f, 1.0f);
                    break;
                case Type::MixColor:
                    sourceCombo("A (color)", n.inA, colorOpts);
                    if (n.inA < 0) ImGui::ColorEdit3("A inline", &n.color.x);
                    sourceCombo("B (color)", n.inB, colorOpts);
                    sourceCombo("T (scalar)", n.inT, scalarOpts);
                    if (n.inT < 0)
                        ImGui::SliderFloat("T inline", &n.scalar, 0.0f, 1.0f);
                    break;
                case Type::MulScalar:
                    sourceCombo("A (scalar)", n.inA, scalarOpts);
                    if (n.inA < 0)
                        ImGui::SliderFloat("A inline", &n.scalar, 0.0f, 1.0f);
                    sourceCombo("B (scalar)", n.inB, scalarOpts);
                    break;
                default: break;
            }
            if (ImGui::SmallButton("delete node")) deleteId = n.id;
            ImGui::Unindent();
        }
        ImGui::PopID();
    }

    if (out) {
        ImGui::SeparatorText("Output sources (override inline)");
        sourceCombo("BaseColor src", out->baseColorSrc, colorOpts);
        sourceCombo("Metallic src",  out->metallicSrc,  scalarOpts);
        sourceCombo("Roughness src", out->roughnessSrc, scalarOpts);
        ImGui::TextDisabled("Set a src then press 'Apply to selected'.");
    }

    if (deleteId >= 0) {
        for (auto& n : m_nodes) {
            if (n.inA == deleteId) n.inA = -1;
            if (n.inB == deleteId) n.inB = -1;
            if (n.inT == deleteId) n.inT = -1;
            if (n.baseColorSrc == deleteId) n.baseColorSrc = -1;
            if (n.metallicSrc  == deleteId) n.metallicSrc  = -1;
            if (n.roughnessSrc == deleteId) n.roughnessSrc = -1;
        }
        m_nodes.erase(std::remove_if(m_nodes.begin(), m_nodes.end(),
            [&](const Node& n){ return n.id == deleteId; }), m_nodes.end());
    }
    return applied;
}
