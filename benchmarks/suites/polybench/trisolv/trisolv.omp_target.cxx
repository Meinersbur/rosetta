// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n,
                   multarray<real, 2> L, real x[], real b[]) {
    real *Ldata = &L[0][0];

#pragma omp target data map(to:Ldata[0:n*n],b[0:n]) map(from:x[0:n]) 
                       {


                           for (idx_t i = 0; i < n; i++) {
                               real sum = 0; // TODO: Keep sum on device
#pragma omp target teams distribute parallel for reduction(+ : sum) map(tofrom:sum) map(to:x[0:n]) map(to:Ldata[0:n*n]) default(none) defaultmap(none) firstprivate(n,Ldata,i,x)
                               for (idx_t j = 0; j < i; j++)
                                   sum += Ldata[i*n+j] * x[j];
#pragma omp target map(to:sum,b[0:n],i,n,Ldata[0:n*n]) defaultmap(none) map(tofrom:x[0:n])
                               x[i] = (b[i] - sum) / Ldata[i*n+i];
                           }



                       }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000


  auto L = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "L");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");

  for (auto &&_ : state)
    kernel(n, L, x, b);
}
