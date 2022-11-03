// BUILD: add_benchmark(ppm=serial)
#include "rosetta.h"



static real sqr(real v) {
    return v*v;
}

static void kernel(int n,
                   multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < n; i++) {

    // j<i case
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) 
S:        A[i][j] -= A[i][k] * A[j][k];
T:      A[i][j] /= A[j][j]; 
    }

    // i==j case
    for (int k = 0; k < i; k++) 
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



// https://math.stackexchange.com/a/358092
// https://math.stackexchange.com/q/357980
static void ensure_posdefinite(int n, multarray<real, 2> A) {
if (n==3) {
  A[0][0] =  4;
A[0][1] = 12;
A[0][2] = -16;
  A[1][0] =  12;
A[1][1] = 37;
A[1][2] = -43;
  A[2][0] =  -16;
A[2][1] = -43;
A[2][2] = 98;
return ;
} else if (n==4) {
    real B[4][4] = {0};
    B[0][0] =  1;
    B[1][0] =   2;
    B[2][0] =   3;
    B[3][0] =  4;
    B[1][1] =    5;
    B[2][1] =    6;
    B[3][1] =    7;
    B[2][2] =      8;
    B[3][2] =     9;
    B[3][3] =      10;

    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            A[i][j] = 0;

    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            for (idx_t k = 0;k < n; ++k )
                A[i][j] += B[i][k] * B[j][k];
    return ;
} else if (n==5) {
    real B[5][5] = {0};
    int k = 1;
    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0; j <=i; ++j ) {
            B[i][j] = k;
            k+=1;
        }



    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            A[i][j] = 0;

    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            for (idx_t k = 0;k < n; ++k )
                A[i][j] += B[i][k] * B[j][k];
    return ;
}  else if (n==6) {
    real B[6][6] = {0};
    int k = 1;
    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0; j < n; ++j ) 
            if (j >= i) {
                B[j][i] = k;
                k += 1;
            }



    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            A[i][j] = 0;

    for (idx_t i = 0;i < n; ++i )
        for (idx_t j = 0;j < n; ++j )
            for (idx_t k = 0;k < n; ++k )
                A[i][j] += B[i][k] * B[j][k];
    return ;
}

    // make symmetric (not really necessary, the kernel doesn't read the upper triangular elements anyway)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            auto val = (std::abs(A[i][j]) + std::abs(A[j][i]))/2;
            A[i][j] = val;
            A[j][i] = val;
        }

        

    real maximum=0;
    for (int i = 0; i < n ; i++)
        for (int j = 0; j < n; j++) {
            auto val = std::abs(A[i][j]);
            if (val > maximum) maximum = val;
        }


    // Make the diagnonal elements too large to be a linear combination of the other columns (strictly diagonally dominant).
    for (int i = 0; i < n ; i++)
        A[i][i] = std::abs( A[i][i]) + 1 + n*maximum;


    // FIXME: Repeated invocation will grow the numbers
}


void run(State &state, int pbsize) {
  size_t n = pbsize; // 2000

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
