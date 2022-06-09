#include "rosetta.h"



static
void kernel(int m, int n,
			real float_n,
             multarray<real,2> data,
                multarray<real,2> corr,
                real *mean,
                real *stddev){
  real eps = 0.1;


#pragma scop
  for (int j = 0; j < m; j++)    {
      mean[j] = 0.0;
      for (int i = 0; i < n; i++)
	    mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (int j = 0; j < m; j++)    {
      stddev[j] = 0.0;
      for (int i = 0; i < n; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = std::sqrt(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)      {
        data[i][j] -= mean[j];
        data[i][j] /= std::sqrt(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (int i = 0; i < m-1; i++)    {
      corr[i][i] = 1.0;
      for (int j = i+1; j < m; j++)        {
          corr[i][j] = 0.0;
          for (int k = 0; k < n; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[m-1][m-1] = 1.0;
#pragma endscop
}


void run(State& state, int pbsize) {
    // n is 8%-25% larger than m
    size_t n = pbsize;
    size_t m = pbsize - pbsize/8;

    real float_n = n;
    auto data = state.allocate_array<real>({m,n}, /*fakedata*/true, /*verify*/false);
    auto corr = state.allocate_array<real>({ m,m },/*fakedata*/false,/*verify*/true);
    auto mean = state.allocate_array<real>({ m },/*fakedata*/false,/*verify*/true );
    auto stddev = state.allocate_array<real>({ m },/*fakedata*/false,/*verify*/true);
  

    for (auto &&_ : state) 
        kernel(m, n, float_n,  data, corr, mean, stddev);
}
