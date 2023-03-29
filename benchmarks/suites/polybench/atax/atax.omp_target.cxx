// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t m, pbsize_t n, multarray<real, 2> A, real x[], real y[], real tmp[]) {
    real *Adata = &A[0][0];

#pragma omp target data map(alloc: tmp[0:m]) map(tofrom : Adata[0:m*n] )  map(tofrom :x[0:n] ) map(tofrom: y[0:n])
    {


        //TODO: memset
#pragma omp target teams distribute parallel for
        for (idx_t j = 0; j < n; ++j)
            y[j] = 0;
#pragma omp target teams distribute parallel for
        for (idx_t i = 0; i < m; ++i) 
            tmp[i] = 0;

#pragma omp target teams distribute parallel for 
            for (idx_t i = 0; i < m; ++i)
                for (idx_t j = 0; j < n; ++j)
                    tmp[i] += Adata[i*n+j] * x[j];

#pragma omp  target teams distribute parallel for 
            for (idx_t j = 0; j < n; ++j)       
            for (idx_t i = 0; i < m; ++i)            
                    y[j] += Adata[i*n + j] * tmp[i] ;

    }
}




    void run(State & state, int pbsize) {
      // n is 5%-20% larger than m
      pbsize_t n = pbsize;
      pbsize_t m = pbsize - pbsize / 10;

      auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "A");
      auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "x");
      auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");
      auto tmp = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ false, "tmp");

      for (auto &&_ : state)
        kernel( m,n, A, x, y, tmp);
    }


