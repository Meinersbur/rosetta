#include "rosetta.h"





static void kernel(int tsteps, int n,
      multarray<real, 3> A, multarray<real, 3> B) {
#pragma scop
    for (int t = 1; t <= tsteps; t++) {
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    B[i][j][k] =   (real)(0.125) * (A[i+1][j][k] - (real)(2.0) * A[i][j][k] + A[i-1][j][k])
                                 + (real)(0.125) * (A[i][j+1][k] - (real)(2.0) * A[i][j][k] + A[i][j-1][k])
                                 + (real)(0.125) * (A[i][j][k+1] - (real)(2.0) * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (int i = 1; i < n-1; i++) {
           for (int j = 1; j < n-1; j++) {
               for (int k = 1; k < n-1; k++) {
                   A[i][j][k] =   (real)(0.125) * (B[i+1][j][k] - (real)(2.0) * B[i][j][k] + B[i-1][j][k])
                                + (real)(0.125) * (B[i][j+1][k] - (real)(2.0) * B[i][j][k] + B[i][j-1][k])
                                + (real)(0.125) * (B[i][j][k+1] - (real)(2.0) * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }
#pragma endscop
}


void run(State &state, int pbsize) {
     size_t tsteps  = 1; // 500
  size_t n = pbsize; // 120






    auto A = state.allocate_array<real>({n,n,n}, /*fakedata*/ true , /*verify*/ false);
  auto B  = state.allocate_array<real>({n,n,n}, /*fakedata*/ true, /*verify*/ true);


  for (auto &&_ : state)
    kernel(tsteps,n,A,B);
}
