#include "engine/engine.h"

#include <iostream>
#include <stdexcept>

int main() {
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
