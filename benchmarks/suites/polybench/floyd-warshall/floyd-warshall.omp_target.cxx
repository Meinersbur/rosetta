// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n, multarray<real, 2> path) {
    real* ppath = &path[0][0];

#pragma omp target data map(tofrom:ppath[0:n*n])
  {

    for (idx_t k = 0; k < n; k++) {

      // FIXME: Allow benign races? Double lock?
#pragma omp target teams distribute parallel for
      for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
          real newval = ppath[i*n+k] + ppath[k*n+j];
          real &ref = ppath[i*n+j];
          if (ref > newval)
            ref = newval;
        }
    }


  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2800



  auto path = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "path");


  for (auto &&_ : state)
    kernel(n, path);
}



