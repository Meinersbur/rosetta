#include "rosetta.h"



static void kernel(int n, double *A) {
    #pragma omp target map(tofrom:A[0:n]) 
    {
      // is_host = omp_is_initial_device();
      //  if (omp_is_initial_device())
      //      fprintf(stderr, "Warning: Not offloading\n");

    for (int i = 0; i < n; i += 1)
        A[i] += 42;
    }

        // if (is_host)
       // fprintf(stderr, "Warning: Not offloading\n");
}

void run(State& state, int n) {
    double *A = state.malloc<double>(n);
    state.fakedata(A, n);

    for (auto &&_ : state) 
        kernel(n, A);
      
    state.verifydata(A, n);
    state.free(A);
}


