#include "asset_browser.h"
#include "components.h"
#include "resource_manager.h"
#include "camera.h"

#include <imgui.h>

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <exception>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {
    std::string lower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower((unsigned char)c));
        return s;
    }
    AssetBrowser::Kind kindOf(const std::string& ext) {
        if (ext == ".obj" || ext == ".gltf" || ext == ".glb")
            return AssetBrowser::Kind::Mesh;
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".tga" || ext == ".bmp" || ext == ".hdr")
            return AssetBrowser::Kind::Texture;
        if (ext == ".wav") return AssetBrowser::Kind::Audio;
        return AssetBrowser::Kind::Other;
    }
    const char* tag(AssetBrowser::Kind k) {
        switch (k) {
            case AssetBrowser::Kind::Dir:     return "[DIR]";
            case AssetBrowser::Kind::Mesh:    return "[MESH]";
            case AssetBrowser::Kind::Texture: return "[TEX]";
            case AssetBrowser::Kind::Audio:   return "[SND]";
            default:                          return "[?]";
        }
    }
}

AssetBrowser::AssetBrowser() = default;

// Fully guarded: a bad/again/permission-denied path must never throw out of
// here (that was the crash). On any failure the listing is just empty + a
// status message; the directory stays where it was.
void AssetBrowser::rescan() {
    m_entries.clear();
    try {
        std::error_code ec;
        if (m_dir.empty() || !fs::is_directory(m_dir, ec)) {
            std::string cur = fs::current_path(ec).string();
            if (!ec) m_dir = cur;
        }
        fs::path base(m_dir);

        // ".." unless we're already at a filesystem/drive root.
        if (base.has_parent_path() && base.parent_path() != base)
            m_entries.push_back({ "..", Kind::Dir, true });

        std::vector<Entry> dirs, files;
        for (fs::directory_iterator it(base,
                 fs::directory_options::skip_permission_denied, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code e2;
            const fs::path p = it->path();
            bool isDir = fs::is_directory(p, e2);
            Entry en;
            en.name  = p.filename().string();
            if (en.name.empty()) continue;
            en.isDir = isDir;
            en.kind  = isDir ? Kind::Dir
                             : kindOf(lower(p.extension().string()));
            (isDir ? dirs : files).push_back(std::move(en));
        }
        auto byName = [](const Entry& a, const Entry& b) {
            return lower(a.name) < lower(b.name);
        };
        std::sort(dirs.begin(), dirs.end(), byName);
        std::sort(files.begin(), files.end(), byName);
        for (auto& d : dirs)  m_entries.push_back(std::move(d));
        for (auto& f : files) m_entries.push_back(std::move(f));
    } catch (const std::exception& ex) {
        m_status = std::string("Cannot read folder: ") + ex.what();
    } catch (...) {
        m_status = "Cannot read folder.";
    }
}

bool AssetBrowser::draw(entt::registry& reg, entt::entity selected,
                        ResourceManager& resources, Camera& camera) {
    if (!m_scanned) { rescan(); m_scanned = true; }
    bool changed = false;

    // Deferred navigation target — never rescan while iterating m_entries.
    std::string navTo;

    static char pathBuf[512];
    if (!ImGui::IsAnyItemActive())
        std::snprintf(pathBuf, sizeof(pathBuf), "%s", m_dir.c_str());
    ImGui::SetNextItemWidth(-140.0f);
    if (ImGui::InputText("##path", pathBuf, sizeof(pathBuf),
                         ImGuiInputTextFlags_EnterReturnsTrue))
        navTo = pathBuf;
    ImGui::SameLine();
    if (ImGui::Button("Go"))      navTo = pathBuf;
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) rescan();

#ifdef _WIN32
    // Drive shortcuts so other drives are reachable (and safe).
    DWORD mask = GetLogicalDrives();
    for (char d = 'A'; d <= 'Z'; ++d) {
        if (!(mask & (1u << (d - 'A')))) continue;
        char drv[4] = { d, ':', '\\', 0 };
        ImGui::SameLine();
        if (ImGui::SmallButton(drv)) navTo = drv;
    }
#endif

    fs::path base(m_dir);
    AssetBrowser::Kind selKind = Kind::Other;
    for (auto& e : m_entries)
        if (!e.isDir && e.name == m_selected) selKind = e.kind;

    ImGui::BeginChild("ab_list", ImVec2(0, 240), true);
    for (const auto& e : m_entries) {
        std::string label = std::string(tag(e.kind)) + " " + e.name;
        bool sel = (!e.isDir && e.name == m_selected);
        if (ImGui::Selectable(label.c_str(), sel,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            if (e.isDir) {
                if (ImGui::IsMouseDoubleClicked(0)) {
                    std::error_code ec;
                    fs::path np = (e.name == "..")
                        ? base.parent_path() : (base / e.name);
                    navTo = np.string();
                }
            } else {
                m_selected = e.name;
            }
        }
    }
    ImGui::EndChild();

    if (m_selected.empty()) {
        ImGui::TextDisabled("Select a file. Double-click [DIR] to enter.");
    } else {
        std::string full = (base / m_selected).string();
        ImGui::Text("Selected: %s", m_selected.c_str());

        if (selKind == Kind::Mesh) {
            std::string ext = lower(fs::path(m_selected).extension().string());
            if (ext == ".obj") {
                if (ImGui::Button("Spawn mesh in scene")) {
                    try {
                        auto h = resources.loadMesh(full);
                        auto e = reg.create();
                        reg.emplace<NameComponent>(e, m_selected);
                        TransformComponent t{};
                        t.position = camera.getPosition() +
                                     camera.getForward() * 4.0f;
                        reg.emplace<TransformComponent>(e, t);
                        reg.emplace<MeshComponent>(e, h);
                        MaterialComponent mat{};
                        mat.texture   = resources.getDefaultTexture();
                        mat.color     = glm::vec4(0.85f, 0.85f, 0.85f, 1.0f);
                        mat.metallic  = 0.05f;
                        mat.roughness = 0.55f;
                        reg.emplace<MaterialComponent>(e, mat);
                        m_status = "Spawned " + m_selected +
                                   " (OBJ loads geometry only — no material/"
                                   "texture; assign one in Material Editor)";
                        changed = true;
                    } catch (const std::exception& ex) {
                        m_status = std::string("Mesh load failed: ") + ex.what();
                    } catch (...) {
                        m_status = "Mesh load failed.";
                    }
                }
                ImGui::TextDisabled("OBJ = geometry only. Use the Material\n"
                                    "Editor / a texture below for appearance.");
            } else {
                ImGui::TextDisabled("glTF: use Hierarchy > Import glTF "
                                    "(skinned import path).");
            }
        } else if (selKind == Kind::Texture) {
            bool hasMat = selected != entt::null && reg.valid(selected) &&
                          reg.all_of<MaterialComponent>(selected);
            ImGui::BeginDisabled(!hasMat);
            if (ImGui::Button("Set as albedo of selected")) {
                try {
                    auto th = resources.loadTexture(full);
                    reg.get<MaterialComponent>(selected).texture = th;
                    m_status = "Applied texture to selected";
                    changed = true;
                } catch (const std::exception& ex) {
                    m_status = std::string("Texture load failed: ") + ex.what();
                } catch (...) {
                    m_status = "Texture load failed.";
                }
            }
            ImGui::EndDisabled();
            if (!hasMat)
                ImGui::TextDisabled("Select an entity with a material first.");
        } else if (selKind == Kind::Audio) {
            if (ImGui::Button("Spawn audio emitter here")) {
                auto e = reg.create();
                reg.emplace<NameComponent>(e, m_selected);
                TransformComponent t{};
                t.position = camera.getPosition() + camera.getForward() * 3.0f;
                reg.emplace<TransformComponent>(e, t);
                AudioSourceComponent a{};
                a.clip = full; a.loop = true; a.spatial = true;
                a.playing = true;
                reg.emplace<AudioSourceComponent>(e, a);
                m_status = "Spawned audio emitter";
                changed = true;
            }
            bool hasSel = selected != entt::null && reg.valid(selected);
            ImGui::SameLine();
            ImGui::BeginDisabled(!hasSel);
            if (ImGui::Button("Attach to selected")) {
                AudioSourceComponent a{};
                a.clip = full; a.loop = true; a.spatial = true;
                a.playing = true;
                if (reg.all_of<AudioSourceComponent>(selected))
                    reg.remove<AudioSourceComponent>(selected);
                reg.emplace<AudioSourceComponent>(selected, a);
                m_status = "Attached audio to selected";
                changed = true;
            }
            ImGui::EndDisabled();
        } else {
            ImGui::TextDisabled("Unsupported file type.");
        }
    }
    if (!m_status.empty()) ImGui::TextWrapped("%s", m_status.c_str());

    // Apply deferred navigation AFTER the UI loop (m_entries safe to rebuild).
    if (!navTo.empty()) {
        std::error_code ec;
        if (fs::is_directory(navTo, ec) && !ec) {
            m_dir = navTo;
            m_selected.clear();
            rescan();
        } else {
            m_status = "Not a folder: " + navTo;
        }
    }
    return changed;
}
