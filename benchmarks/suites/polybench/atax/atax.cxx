#include "rosetta.h"


static
void kernel(int m, int n, 
    multarray<real,2> A,
    multarray<real,1> x,
    multarray<real,1> y,
    multarray<real,1> tmp){

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
    // n 5%-20% larger than n
    size_t n = pbsize + pbsize/10;
    size_t m = pbsize;


    auto A  = state.fakedata_array<double>(n*m,/*verify*/false);   
    auto x  = state.alloc_array<double>(n,/*verify*/false);   
    auto y  = state.alloc_array<double>(n,/*verify*/true);  
    auto tmp  = state.alloc_array<double>(n);  



    for (auto &&_ : state) 
        kernel(n, m, multarray<real, 2>(A.data(), { m, n }), multarray<real, 1>(x.data(), { n }), multarray<real, 1>(y.data(), { n }), multarray<real, 1>(tmp.data(), {m}));
}

