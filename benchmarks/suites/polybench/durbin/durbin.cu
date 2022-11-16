// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"

#include <cub/cub.cuh>   
#include <thrust/iterator/iterator_traits.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>
#include <thrust/detail/config.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda.h>


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}




template<typename T>
struct r_times_y : public thrust:: unary_function<T,T>
{
    T *r;
    T *y;
    idx_t k;

r_times_y(T *r, T*y, idx_t k) : r(r), y(y), k(k)  {  }


  __host__ __device__ T operator()(const T &x) const
  {
    idx_t i = x;
    return  r[k - i - 1] * y[i];
  }
};



__global__ void kernel_z(pbsize_t n,
                   real r[],
                   real y[], real z[], idx_t k, real alpha) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < k)
   z[i] = y[i] + alpha * y[k - i - 1];
}






static void kernel(pbsize_t n,
                   real r[],
                   real y[], real z[]) {
                           const  unsigned threadsPerBlock = 256;
// https://github.com/NVIDIA/thrust/blob/master/examples/cuda/wrap_pointer.cu/
thrust::device_ptr<real> wrapped_r = thrust::device_pointer_cast(r);
thrust::device_ptr<real> wrapped_y = thrust::device_pointer_cast(y);

    real beta = 1;
    real alpha = -wrapped_r[0];
  





  wrapped_y[0] = alpha;
  for (idx_t k = 1; k < n; k++) {
    // FIXME: Quite approximate sum
       r_times_y<real> op{r,y,k};
        real sum  = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0), 
                                    thrust::make_counting_iterator(0)+k,
                                     op,
                                      (real)0,
                                     thrust::plus<real>());
                               

        beta = (1 - alpha * alpha) * beta;
        alpha = -(wrapped_r[k] + sum) / beta;

      kernel_z<<<threadsPerBlock,num_blocks(k,threadsPerBlock)>>>(n,r,y,z,k,alpha);

      BENCH_CUDA_TRY(   cudaMemcpy( y ,z, k*sizeof(real), cudaMemcpyDeviceToDevice)); 
 
       wrapped_y[k] = alpha;     
    }




}


void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000



  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");
 // auto z = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false, "z");


  real* dev_r = state.allocate_dev<real>(n);
real* dev_y = state.allocate_dev<real>(n);
real* dev_z = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
 BENCH_CUDA_TRY( cudaMemcpy(dev_r, r.data(),  n* sizeof(real), cudaMemcpyHostToDevice));

    kernel(n, dev_r, dev_y, dev_z);

     BENCH_CUDA_TRY(    cudaMemcpy( y.data() ,dev_y,  n*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }


              state.free_dev(dev_r);     
                        state.free_dev(dev_y);     
                                  state.free_dev(dev_z);     
}

