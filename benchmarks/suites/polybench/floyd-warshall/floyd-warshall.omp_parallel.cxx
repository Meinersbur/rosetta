// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(pbsize_t  n, multarray<real, 2> path) {
#pragma omp parallel default(none) firstprivate(n,path)
    {
        for (idx_t k = 0; k < n; k++) {

            // FIXME: Allow benign races? Double lock?
#pragma omp parallel for schedule(static) 
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = 0; j < n; j++) {   
                    auto newval = path[i][k] + path[k][j];
                        auto &ref = path[i][j] ;
                    if (ref> newval)  
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
