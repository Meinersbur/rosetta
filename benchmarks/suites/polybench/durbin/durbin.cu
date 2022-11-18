// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>



static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}
 




struct r_times_y : public thrust:: unary_function<real,real>
{
  thrust::device_ptr<real> r;
  thrust::device_ptr<real> y;
    idx_t k;

r_times_y(thrust::device_ptr<real> r, thrust::device_ptr<real> y, idx_t k) : r(r), y(y), k(k)  {  }


  __host__ __device__ real operator()(idx_t i) const  {
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
  thrust::device_ptr<real> r,
  thrust::device_ptr<real> y, thrust::device_ptr<real> z) {
                           const  unsigned threadsPerBlock = 256;

    real beta = 1;
    real alpha = -r[0];
  





  y[0] = alpha;
  for (idx_t k = 1; k < n; k++) {
    // FIXME: Quite approximate sum
       r_times_y op{r,y,k};
        real sum  = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0), 
                                    thrust::make_counting_iterator(0)+k,
                                     op,
                                      (real)0,
                                     thrust::plus<real>());
                               

        beta = (1 - alpha * alpha) * beta;
        alpha = -(r[k] + sum) / beta;

      kernel_z<<<threadsPerBlock,num_blocks(k,threadsPerBlock)>>>(n,r.get(),y.get(),z.get(),k,alpha);

      BENCH_CUDA_TRY(   cudaMemcpy( y.get() ,z.get(), k*sizeof(real), cudaMemcpyDeviceToDevice));
 
       y[k] = alpha;
    }




}


void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000



  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");



  real* dev_r = state.allocate_dev<real>(n);// thrust::device_malloc ?
real* dev_y = state.allocate_dev<real>(n);
real* dev_z = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
 BENCH_CUDA_TRY( cudaMemcpy(dev_r, r.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
    kernel(n,  thrust::device_pointer_cast(dev_r),  thrust::device_pointer_cast(dev_y),  thrust::device_pointer_cast(dev_z));
     BENCH_CUDA_TRY(    cudaMemcpy( y.data() ,dev_y,  n*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }


              state.free_dev(dev_r);     
                        state.free_dev(dev_y);     
                                  state.free_dev(dev_z);     
}

