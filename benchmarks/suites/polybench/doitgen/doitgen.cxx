#include "rosetta.h"


static void kernel(int nr, int nq, int np,
 multarray<real, 3> A,
  multarray<real, 2> C4,
  real sum[]) {
#pragma scop
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++)  {
      for (int p = 0; p < np; p++)  {
	sum[p] = 0;
	for (int s = 0; s < np; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (int p = 0; p < np; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop
}


void run(State &state, int pbsize) {
      size_t nq  =pbsize-pbsize/8; // 140
  size_t nr = pbsize-pbsize/16; // 150
    size_t np=pbsize ; // 160


  auto A = state.allocate_array<double>({nr, nq,np}, /*fakedata*/ true, /*verify*/ true );
  auto C4 = state.allocate_array<double>({np,np}, /*fakedata*/ true , /*verify*/ false );
    auto sum = state.allocate_array<double>({np}, /*fakedata*/ false, /*verify*/ false);



  for (auto &&_ : state)
    kernel(nr,nq,np,A,C4,sum);
}
