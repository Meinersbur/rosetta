// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl, pbsize_t nm,
                   multarray<real, 2> E,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   multarray<real, 2> F,
                   multarray<real, 2> C,
                   multarray<real, 2> D,
                   multarray<real, 2> G) {
real *pA = &A[0][0];
real *pB = &B[0][0];
real *pC = &C[0][0];
real *pD = &D[0][0];
real *pE = &E[0][0];
real *pF = &F[0][0];
real *pG = &G[0][0];

#pragma omp target data map(to:pA[0:ni*nk],pB[0:nk*nj],pC[0:nj*nm],pD[0:nm*nl]) \
                        map(alloc:pE[0:ni*nj],pF[0:nj*nl]) \
                        map(from:pG[0:ni*nl])
{
#define AccA(i,j) (pA[(i)*nk+(j)])
#define AccB(i,j) (pB[(i)*nj+(j)])
#define AccC(i,j) (pC[(i)*nm+(j)])
#define AccD(i,j) (pD[(i)*nl+(j)])
#define AccE(i,j) (pE[(i)*nj+(j)])
#define AccF(i,j) (pF[(i)*nl+(j)])
#define AccG(i,j) (pG[(i)*nl+(j)])

/* E := A*B */
#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(ni,nj,nk,pE,pA,pB)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++) {
        AccE(i,j) = 0;
        for (idx_t k = 0; k < nk; ++k)
            AccE(i,j) += AccA(i,k) * AccB(k,j);
      }

/* F := C*D */
#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(nj,nl,nm,pF,pC,pD)
    for (idx_t i = 0; i < nj; i++)
      for (idx_t j = 0; j < nl; j++) {
          AccF(i,j) = 0;
        for (idx_t k = 0; k < nm; ++k)
            AccF(i,j) += AccC(i,k) * AccD(k,j);
      }

/* G := E*F */
#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(ni,nl,nj,pG,pE,pF)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nl; j++) {
          AccG(i,j) = 0;
        for (idx_t k = 0; k < nj; ++k)
            AccG(i,j) += AccE(i,k) * AccF(k,j);
      }
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 6;  // 1000
  pbsize_t nl = pbsize - pbsize / 12; // 1100
  pbsize_t nm = pbsize;               // 1200



  auto E = state.allocate_array<real>({ni, nj}, /*fakedata*/ false, /*verify*/ false, "E");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto F = state.allocate_array<real>({nj, nl}, /*fakedata*/ false, /*verify*/ false, "F");
  auto C = state.allocate_array<real>({nj, nm}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({nm, nl}, /*fakedata*/ true, /*verify*/ false, "D");
  auto G = state.allocate_array<real>({ni, nl}, /*fakedata*/ false, /*verify*/ true, "G");

  for (auto &&_ : state)
    kernel(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
}
