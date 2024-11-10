// #include <iostream>


// int main() {
//     std::cout << "Hello, World!" << std::endl;
//     return 0;
// }


#include <iostream>

#if defined(_MSC_VER)
    #include <intrin.h>
    #define DEBUG_BREAK() __debugbreak()
#elif defined(__clang__) || defined(__GNUC__)
    #include <csignal>
    #define DEBUG_BREAK() raise(SIGTRAP)
#else
    #error "Platform not supported!"
#endif

int main() {
    std::cout << "Hello, World!" << std::endl;
    int x = 999; 

    DEBUG_BREAK();  // Program execution will pause here if a debugger is attached.

    std::cout << "After breakpoint" << std::endl;

    return 0;
}

