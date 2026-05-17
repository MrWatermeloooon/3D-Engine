#pragma once

#include <entt/entt.hpp>
#include <string>
#include <vector>

class ResourceManager;
class Camera;

// File-explorer panel for browsing the project folder and bringing assets
// into the scene: spawn .obj meshes, set a texture as the selected entity's
// albedo, or attach a .wav as an audio source. Drag-and-drop onto the 3D
// viewport isn't an ImGui target, so this uses select-file + action buttons
// (same interaction model as the rest of the editor).
class AssetBrowser {
public:
    AssetBrowser();
    // Returns true if it mutated the scene this frame (so the caller can
    // commit an undo snapshot).
    bool draw(entt::registry& reg, entt::entity selected,
              ResourceManager& resources, Camera& camera);

    enum class Kind { Dir, Mesh, Texture, Audio, Other };
    struct Entry { std::string name; Kind kind; bool isDir; };

private:
    void rescan();

    std::string        m_dir;
    std::vector<Entry> m_entries;
    std::string        m_selected;     // file name within m_dir
    std::string        m_status;
    bool               m_scanned = false;
};
