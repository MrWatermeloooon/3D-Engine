#include "engine/engine.h"

#include <iostream>
#include <stdexcept>

int main() {
    // Unbuffered stdout so that taskkill-style aborts during dev still leave
    // a readable log. Cheap; no perf concern for a per-frame engine.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    Engine engine;

    try {
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
