// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>



#define match(b1, b2) (((b1) + (b2)) == 3 ? 1 : 0)


static void kernel(pbsize_t n, real seq[], multarray<real, 2> table) {
        for (idx_t i = n - 1; i >= 0; i--) {
            for (idx_t j = i + 1; j < n; j++) {

                if (j - 1 >= 0)
                    table[i][j] =  std::max(table[i][j], table[i][j - 1]);
                if (i + 1 < n)
                    table[i][j] =  std::max(table[i][j], table[i + 1][j]);

                if (j - 1 >= 0 && i + 1 < n) {
                    /* don't allow adjacent elements to bond */
                    if (i < j - 1)
                        table[i][j] =  std::max(table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j]));
                    else
                        table[i][j] =  std::max(table[i][j], table[i + 1][j - 1]);
                }

                real maximum = table[i][j] ;
#pragma omp parallel for default(none) firstprivate(i,j,n,table) schedule(static) reduction(max:maximum)
                for (idx_t k = i + 1; k < j; k++)
                    maximum =  std::max(maximum, table[i][k] + table[k + 1][j]);
                table[i][j] = maximum;
            }
        }
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2500



  auto seq = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "seq");
  auto table = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "table");


  for (auto &&_ : state)
    kernel(n, seq, table);
}
