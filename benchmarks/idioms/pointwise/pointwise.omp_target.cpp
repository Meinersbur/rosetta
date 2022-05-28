#include "rosetta.h"
#include <omp.h>

static void kernel(int n, double *B, double *A) {

bool is_host;
    #pragma omp target map(from:B[0:n]) map(to:A[0:n])  map(is_host)
    {
        is_host = omp_is_initial_device();
      //  if (omp_is_initial_device())
      //      fprintf(stderr, "Warning: Not offloading\n");
    #pragma omp teams distribute parallel for 
    for (int i = 0; i < n; i += 1) {
        B[i] = 42 * A[i];
    }
    }
    #pragma omp taskwait

    if (is_host)
        fprintf(stderr, "Warning: Not offloading\n");
}


void run(State& state, int n) {
    if (n < 0)
        n = (DEFAULT_N);

    double *A = new double[n];
    double *B = new double[n];

    for (auto &&_ : state) {
        kernel(n, B, A);
    }

    delete[] A;
    delete[] B;
}
