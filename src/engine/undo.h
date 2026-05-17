#pragma once

#include <entt/entt.hpp>
#include <string>
#include <vector>

// Snapshot-based undo/redo. Each history entry is a full serialized scene
// (the same binary codec the scene serializer / prefabs use) — robust in an
// ECS editor without per-action command classes, at whole-scene granularity.
//
// commit() is called EXPLICITLY right after each discrete editor action
// (spawn / delete / add-remove component / prefab instantiate / scene load /
// gizmo release). It is deliberately NOT driven by a heuristic: the scene is
// continuously mutated by physics/rotators/scripts, so any "did the scene
// change" guess captures simulation drift as fake edits and corrupts redo.
class UndoStack {
public:
    void seed(const entt::registry& reg);                  // first state
    void commit(const entt::registry& reg, const std::string& label);

    bool undo(entt::registry& reg);
    bool redo(entt::registry& reg);
    bool canUndo() const { return m_cursor > 0; }
    bool canRedo() const {
        return m_cursor >= 0 &&
               m_cursor + 1 < static_cast<int>(m_states.size());
    }

    const std::vector<std::string>& labels() const { return m_labels; }
    int  cursor() const { return m_cursor; }
    void clear();

private:
    std::vector<std::string> m_states;   // full serialized scene snapshots
    std::vector<std::string> m_labels;
    int m_cursor = -1;                   // index of the current state
    static constexpr int kMax = 64;
};
