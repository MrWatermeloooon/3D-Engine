#include "undo.h"
#include "serialization.h"

void UndoStack::clear() {
    m_states.clear();
    m_labels.clear();
    m_cursor = -1;
}

void UndoStack::seed(const entt::registry& reg) {
    clear();
    m_states.push_back(SceneSerializer::saveToBuffer(reg));
    m_labels.push_back("initial");
    m_cursor = 0;
}

void UndoStack::commit(const entt::registry& reg, const std::string& label) {
    std::string snap = SceneSerializer::saveToBuffer(reg);
    // Guard against an accidental double-call for the same action.
    if (m_cursor >= 0 && snap == m_states[m_cursor]) return;

    // Doing a new action after undo discards the redo tail (standard).
    if (m_cursor + 1 < static_cast<int>(m_states.size())) {
        m_states.resize(m_cursor + 1);
        m_labels.resize(m_cursor + 1);
    }
    m_states.push_back(std::move(snap));
    m_labels.push_back(label);
    m_cursor = static_cast<int>(m_states.size()) - 1;

    if (static_cast<int>(m_states.size()) > kMax) {
        m_states.erase(m_states.begin());
        m_labels.erase(m_labels.begin());
        --m_cursor;
    }
}

bool UndoStack::undo(entt::registry& reg) {
    if (!canUndo()) return false;
    --m_cursor;
    return SceneSerializer::loadFromBuffer(reg, m_states[m_cursor]);
}

bool UndoStack::redo(entt::registry& reg) {
    if (!canRedo()) return false;
    ++m_cursor;
    return SceneSerializer::loadFromBuffer(reg, m_states[m_cursor]);
}
