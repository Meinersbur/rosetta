// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"

struct cast_real : public thrust:: unary_function<real,real>{
  __host__ __device__ real operator()(idx_t i) const  {
    return  i;
  }
};


static real kernel(pbsize_t n) {
    real sum  = thrust::transform_reduce(
        thrust::device,
          thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0)+n,
        cast_real(),
          (real)0,
         thrust::plus<real>());

    real sum = 0;
    for (idx_t i = 0; i < n; i += 1)
        sum += i;
    return sum;
}

void run(State& state, pbsize_t n) {
 auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");
 multarray<real, 1> sum = sum_owner;

    for (auto &&_ : state)
        sum[0] = kernel(n);
}
