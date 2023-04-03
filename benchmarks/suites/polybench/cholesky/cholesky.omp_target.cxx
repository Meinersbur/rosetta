// BUILD: add_benchmark(ppm=omp_target,sources=[__file__, "cholesky-common.cxx"])

#include "cholesky-common.h"

#include <omp.h>
#include <rosetta.h>



static real sqr(real v) {
  return v * v;
}




static void kernel_polly(pbsize_t n, multarray<real, 2> A) {
  auto *Adata = &A[0][0];



#pragma omp target data map(tofrom:Adata[0:n*n]) 
  {


    for (idx_t j = 0; j < n; j++) {

#pragma omp target
      Adata[j*n+j] = std::sqrt(Adata[j*n+j]); 

#pragma omp target teams distribute parallel for
      for (idx_t i = j + 1; i < n; i++)
        Adata[i*n+j] /= Adata[j*n+j]; 

#pragma omp target teams distribute parallel for collapse(2)
      for (idx_t i = j + 1; i < n; i++)
        for (idx_t k = j + 1; k <= i; k++) 
          Adata[i*n+k] -= Adata[i*n+j] * Adata[k*n+j];   


    }



  }
}




void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");


  for (auto &&_ : state.manual()) {
    ensure_posdefinite(n, A);
    {
      auto &&scope = _.scope();
      // FIXME: cholesky of pos-definite matrix is not necessarily itself pos-definite
      kernel_polly(n, A);
    }
  }
}
