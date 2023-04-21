// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>



static void kernel(pbsize_t n, real seq[], multarray<real, 2> table) {
  // Wavefronting
#pragma omp parallel
  for (idx_t w = n; w < 2 * n - 1; ++w) {
#pragma omp for schedule(guided)
    for (idx_t j = std::max(0, w - ((idx_t)n - 1)); j < std::min(n, n + w - ((idx_t)n - 1)); ++j) {
      idx_t i = ((idx_t)n - 1) + j - w;

      real maximum = table[i][j];

      if (j - 1 >= 0)
        maximum = std::max(maximum, table[i][j - 1]);
      if (i + 1 < n)
        maximum = std::max(maximum, table[i + 1][j]);

      if (j - 1 >= 0 && i + 1 < n) {
        auto upd = table[i + 1][j - 1];

        /* don't allow adjacent elements to bond */
        if (i < j - 1)
          upd += (seq[i] + seq[j] == 3) ? (real)1 : (real)0;

        maximum = std::max(maximum, upd);
      }

      for (idx_t k = i + 1; k < j; k++)
        maximum = std::max(maximum, table[i][k] + table[k + 1][j]);

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
