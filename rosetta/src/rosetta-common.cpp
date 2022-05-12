#include "rosetta.h"


#ifdef _MSC_VER
//__declspec(selectany)
#else
__attribute__((weak))
#endif
void run(State& state, int n);


struct  Rosetta {
        static void run(int n) {
            if (&::run) {
                State state;
                ::run(state, n);
            }
        }
};




int main(int argc, char* argv[]) {
    // TODO: benchnark-specific default size
    int n = 100;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

Rosetta::run( n );

    return EXIT_SUCCESS;
}

