// BUILD: add_benchmark(ppm=serial,sources=[__file__, "cholesky-common.cxx"])
#include <rosetta.h>
#include "cholesky-common.h"


static real sqr(real v) {
    return v*v;
}



#if 0
/* Polly parallelizes this as follows:
#pragma minimal dependence distance: 1
for (int c0 = 0; c0 < n; c0 += 1) {
  Stmt_if_then(c0, c0);
  #pragma simd
  #pragma known-parallel
  for (int c1 = c0 + 1; c1 < n; c1 += 1)
    Stmt_if_else(c1, c0);
  #pragma omp parallel for
  for (int c1 = c0 + 1; c1 < n; c1 += 1)
    #pragma simd
  for (int c2 = c0 + 1; c2 <= c1; c2 += 1)
    Stmt_for_body8(c1, c2, c0);
}
*/
extern
void kernel2(pbsize_t n, double A[128][128]) {
#pragma scop
    for (idx_t i = 0; i < n; i++) { // c1
        for (idx_t j = 0; j <= i; j++) { //c0
            for (int k = 0; k < j; k++) // c2
                A[i][j] -= A[i][k] * A[j][k]; // Stmt_for_body8(c1, c2, c0)
            if (i == j) {
                A[i][j] = std::sqrt(A[j][j]); // Stmt_if_then(c0, c0)
            } else {
                A[i][j] /= A[j][j]; // Stmt_if_else(c1, c0)
            }
        } 
    } 
#pragma endscop
}

static 
void kernel3(pbsize_t n, double A[128][128]) {
#pragma scop
    for (idx_t i = 0; i < n; i++) { // c1
        for (idx_t j = 0; j <= i; j++) { //c0
            if (i == j) {
                for (idx_t k = 0; k < j; k++) // c2
                        A[j][j] -= A[j][k] * A[j][k]; // Stmt_for_body8(c1, c2, c0)
                A[j][j] = std::sqrt(A[j][j]); // Stmt_if_then(c0, c0)
            } else {
                for (idx_t k = 0; k < j; k++) // c2
                    A[i][j] -= A[i][k] * A[j][k]; // Stmt_for_body8(c1, c2, c0)
                A[i][j] /= A[j][j]; // Stmt_if_else(c1, c0)
            }
        } 
    } 
#pragma endscop
}
#endif 


static void kernel_polly(pbsize_t n, multarray<real, 2> A) {
        for (idx_t j = 0; j < n; j++) { // c0
            A[j][j] = std::sqrt(A[j][j]); // Stmt_if_then

            for (idx_t i = j + 1; i < n; i++)
                A[i][j] /= A[j][j]; // Stmt_if_else

            for (idx_t i = j + 1; i < n; i++)
                for (idx_t k = j + 1; k <= i; k++) // c2
                    A[i][k] -= A[i][j] * A[k][j]; // Stmt_for_body8
        }
}



static 
 void kernel(pbsize_t n,
                   multarray<real, 2> A) {
#pragma scop
  for (idx_t i = 0; i < n; i++) {

    // j<i case
    for (idx_t j = 0; j < i; j++) {
      for (idx_t k = 0; k < j; k++) 
S:        A[i][j] -= A[i][k] * A[j][k];
T:      A[i][j] /= A[j][j]; 
   }

    // i==j case
    for (idx_t k = 0; k < i; k++) 
U:      A[i][i] -= sqr(A[i][k]);
V:    A[i][i] = std::sqrt(A[i][i]);
  }
#pragma endscop
}

// T(4,0):   A[4][0] /= A[0][0]; 
// 
// S(4,1,0): A[4][1] -= A[4][0] * A[1][0];
// T(4,1):   A[4][1] /= A[1][1]; 
// 
// S(4,2,0): A[4][2] -= A[4][0] * A[2][0];
// S(4,2,1): A[4][2] -= A[4][1] * A[2][1];
// T(4,2):   A[4][2] /= A[2][2]; 
// 
// S(4,3,0): A[4][3] -= A[4][0] * A[3][0];
// S(4,3,1): A[4][3] -= A[4][1] * A[3][1];
// S(4,3,2): A[4][3] -= A[4][2] * A[3][2];
// T(4,3):   A[4][3] /= A[3][3]; 

// U(4,0):   A[4][4] -= sqr(A[4][0]);
// U(4,1):   A[4][4] -= sqr(A[4][1]);
// U(4,2):   A[4][4] -= sqr(A[4][2]);
// U(4,3):   A[4][4] -= sqr(A[4][3]);
// V(4):     A[4][4] = std::sqrt(A[4][4]);




void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



  for (auto&& _ : state.manual()) {
      ensure_posdefinite(n, A);
      {
          auto &&scope = _.scope();
          // FIXME: cholesky of pos-definite matrix is not necessarily itself pos-definite
          kernel(n, A);
      }
  }
}


