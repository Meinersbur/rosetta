#include "rosetta.h"


static
void kernel(int m, int n, multarray<real,2> A,    real* x,    real* y,    real* tmp){
#pragma scop
    for (int i = 0; i < n; i++)
        y[i] = 0;
    for (int i = 0; i < m; i++){
        tmp[i] = 0.0;
        for (int j = 0; j < n; j++)
            tmp[i] = tmp[i] + A[i][j] * x[j];
        for (int j = 0; j < n; j++)
            y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop
}


void run(State& state, int pbsize) {
    // n is 5%-20% larger than m
    size_t n = pbsize ;
    size_t m = pbsize - pbsize/10;


    auto A = state.allocate_array<double>({n,m}, /*fakedata*/true, /*verify*/false); 
    auto x = state.allocate_array<double>({ n },/*fakedata*/false,/*verify*/false);
    auto y = state.allocate_array<double>({ n },/*fakedata*/false,/*verify*/true);
    auto tmp = state.allocate_array<double>({ n },/*fakedata*/false,/*verify*/false);

    for (auto &&_ : state) 
        kernel(n, m, A, x, y, tmp);
}
