// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n, real A[], real B[]) {
int TwinPrimeA[] = {3,5,11,17,29,41,59,71,101,107,137};
int TwinPrimeB[] = {5,7,13,19,31,43,61,73,103,109,139};

#pragma omp target teams map(to:TwinPrimeA[0:11],TwinPrimeB[0:11]) map(tofrom:A[0:n*n+139],B[0:n*n+139])
{
#pragma omp distribute parallel for collapse(2)
  for (auto a : TwinPrimeA)
    for (int i = 0; i < n; i += 1)
      A[i*n+a] = sin(i*2*M_PI/n + a);

#pragma omp distribute parallel for collapse(2) 
  for (auto b : TwinPrimeB)
    for (int j = 1; j < n-1; j += 1)
      B[j*n+b] = cos(j*2*M_PI/n + b);
}
}




void run(State &state, pbsize_t n) {
  auto A = state.allocate_array<real>({n*n+137}, /*fakedata*/ true, /*verify*/ true, "A");
  auto B = state.allocate_array<real>({n*n+139}, /*fakedata*/ true, /*verify*/ true, "B");

  for (auto &&_ : state)
    kernel(n, A, B);
}