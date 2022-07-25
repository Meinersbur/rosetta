#include "rosetta.h"


static
void kernel(int n, real alpha, real beta,
 multarray<real,2> A,   real *u1, real *v1,  real *u2, real *v2,  real *w, real *x,  real *y, real *z ){
#pragma scop
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (int i = 0; i < n; i++)
    x[i] = x[i] + z[i];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];
#pragma endscop
}


void run(State& state, int n) {
    real alpha = 1.5;
  real beta = 1.2;
    auto A = state.allocate_array<double>({n,n},  /*fakedata*/true, /*verify*/false );
    auto u1 = state.allocate_array<double>({ n },  /*fakedata*/true,/*verify*/false);
    auto v1 = state.allocate_array<double>({ n },  /*fakedata*/true,/*verify*/false);
    auto u2 = state.allocate_array<double>({ n },  /*fakedata*/true,/*verify*/false);
    auto v2 = state.allocate_array<double>({ n },  /*fakedata*/true,/*verify*/false);
    auto w = state.allocate_array<double>({ n },  /*fakedata*/false,/*verify*/true);
    auto x = state.allocate_array<double>({ n },  /*fakedata*/false,/*verify*/true);
    auto y = state.allocate_array<double>({ n },  /*fakedata*/false,/*verify*/true);
    auto z = state.allocate_array<double>({ n },  /*fakedata*/false,/*verify*/true);



    for (auto &&_ : state)
        kernel(n,  alpha , beta,A, u1,v1,u2,v2,w,x,y,z);
}
