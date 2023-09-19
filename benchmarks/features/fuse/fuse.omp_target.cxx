// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n, real A[], real B[]) {
int TwinPrimeA[] = {3,5,11,17,29,41,59,71,101,107,137};
int TwinPrimeB[] = {5,7,13,19,31,43,61,73,103,109,139};

#pragma omp target data map(to:TwinPrimeA[0:11],TwinPrimeB[0:11]) map(tofrom:A[0:n*n+139],B[0:n*n+139])
{

#pragma omp target teams distribute parallel for collapse(2) 
  for (int k = 0; k < 11; ++k) {
    for (int i = 0; i < n; i += 1) {
      auto a = TwinPrimeA[k];
      A[i*n+a] = sin(i*2*M_PI/n + a);

      auto b = TwinPrimeB[k];
      if (i < n-1)
         B[i*n+b] = cos(i*2*M_PI/n + b);
    }
  }

}
}




void run(State &state, pbsize_t n) {
  auto A = state.allocate_array<real>({n*n+137}, /*fakedata*/ true, /*verify*/ true, "A");
  auto B = state.allocate_array<real>({n*n+139}, /*fakedata*/ true, /*verify*/ true, "B");

  for (auto &&_ : state)
    kernel(n, A, B);
}
