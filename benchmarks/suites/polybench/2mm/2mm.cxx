#include "rosetta.h"



static
void kernel(int ni, int nj, int nk, int nl,
			real alpha,		real beta,
           multarray<real,2> tmp,
                multarray<real,2> A,
                             multarray<real,2> B,   multarray<real,2> C,   multarray<real,2> D){
#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++)
      {
	tmp[i][j] = 0;
	for (int k = 0; k < nk; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++)
      {
	D[i][j] *= beta;
	for (int k = 0; k < nj; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop
}



void run(State& state, int pbsize) {
   size_t ni = pbsize - pbsize /3; // 800
   size_t nj = pbsize - pbsize/4; // 900
      size_t nk = pbsize - pbsize/12; //  1100
         size_t nl = pbsize; // 1200

    real alpha = 1.5;
  real beta = 1.2;
    auto tmp = state.allocate_array<double>({ni,nj},  /*fakedata*/false, /*verify*/false );
    auto A = state.allocate_array<double>({ni,nk},  /*fakedata*/true, /*verify*/false );
    auto B = state.allocate_array<double>({nk,nj},  /*fakedata*/true, /*verify*/false );
    auto C = state.allocate_array<double>({nj,nl},  /*fakedata*/true, /*verify*/false );
    auto D = state.allocate_array<double>({ni,nl},  /*fakedata*/true, /*verify*/true );

    for (auto &&_ : state)
        kernel(ni,nj,nk,nl , alpha , beta,tmp,A,B,C,D);
}

