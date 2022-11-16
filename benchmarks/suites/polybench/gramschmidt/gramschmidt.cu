// TODO: add_benchmark(ppm=cuda,sources=[__file__, "gramschmidt-common.cxx"])

#include <rosetta.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "gramschmidt-common.h"



 

static 
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


 __device__  real sqr(real v) { return v*v;}






template<typename T>
struct outer_sqr : public thrust:: unary_function<T,T>
{pbsize_t m; pbsize_t n;
    T *A;
    idx_t k;

outer_sqr(pbsize_t m, pbsize_t n, T *A, idx_t k) :  m(m), n(n), A(A), k(k) {  }


  __device__ T operator()(int i) const {
    return   sqr(A[i*n+k]) ;
  }
};



__global__ void kernel_Q(pbsize_t m, pbsize_t n,
                  real* A, real* R, real* Q, idx_t k) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < m)
       Q[i*n+k] = A[i*n+k] / R[k*n+k];
}




__global__ void kernel_R(pbsize_t m, pbsize_t n,
                  real* A, real* R, real* Q, idx_t k) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x + k + 1;

if (j < n) {
     R[k*n+j] = 0;
                                   for (idx_t i = 0; i < m; i++)
                                       R[k*n+j] += Q[i*n+k] * A[i*n+j];
}
}


__global__ void kernel_A(pbsize_t m, pbsize_t n,
                  real* A, real* R, real* Q, idx_t k) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x + k + 1;

    if (j < n) {
            for (idx_t i = 0; i < m; i++)
                                        A[i*n+j] -= Q[i*n+k] * R[k*n+j];
    }
}





static void kernel(pbsize_t m, pbsize_t n,
                  real* A, real* R, real* Q) {
                            const  unsigned threadsPerBlock = 256;
thrust::device_ptr<real> wrapped_A = thrust::device_pointer_cast(A);
thrust::device_ptr<real> wrapped_R = thrust::device_pointer_cast(R);
thrust::device_ptr<real> wrapped_Q = thrust::device_pointer_cast(Q);

       for (idx_t k = 0; k < n; k++) {    
 outer_sqr<real> op{m,n,A,k};
real sum  =     
#if 1
  thrust::transform_reduce( thrust::device,
                                         thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(0)+m,
                                     op,
                                      (real)0,
                                     thrust::plus<real>());
                                     #endif




                                     wrapped_R[k*n+k] = std::sqrt(sum);


                                    kernel_Q<<<threadsPerBlock, num_blocks(m,threadsPerBlock)>>>(m,n,A,R,Q,k);
#if 1
                                  kernel_R<<<threadsPerBlock, num_blocks(n-(k + 1),threadsPerBlock)>>>(m,n,A,R,Q,k);
                                  kernel_A<<<threadsPerBlock, num_blocks(n-(k + 1),threadsPerBlock)>>>(m,n,A,R,Q,k);
#endif
       }



                       
                       #if 0
                           for (idx_t k = 0; k < n; k++) {

                               double sum = 0;
                               // FIXME: For some reason OpenMP-reduction numericall destabilizes this
                               // Possibly inherent to Gram-Schmidt numeric instability
                               // https://en.wikipedia.org/wiki/Gram–Schmidt_process#Numerical_stability
                               // Generate fakedata that a not that similat to each other
#pragma omp parallel for schedule(static) default(none) firstprivate(k,m,A) reduction(+:sum)
                               for (int i = 0; i < m; i++) {
//#pragma omp critical
//                                   printf("%lu %d: sqr(%g) = %g\n",k,i,A[i][k],sqr(A[i][k]) );
                                   sum += sqr(A[i][k]) ;
                               }

                           //    printf("%lu: sum=%g\n",k,sum );
                               R[k][k] = std::sqrt(sum);


#pragma omp parallel for schedule(static)  default(none)  firstprivate(k,m,A,Q,R)
                               for (int i = 0; i < m; i++)
                                   Q[i][k] = A[i][k] / R[k][k];


#pragma omp parallel for schedule(static)  default(none)  firstprivate(k,m,n,A,Q,R)
                               for (int j = k + 1; j < n; j++) {
                                   R[k][j] = 0;
                                   for (idx_t i = 0; i < m; i++)
                                       R[k][j] += Q[i][k] * A[i][j];
                               }

#pragma omp parallel for schedule(static)  default(none) firstprivate(k,m,n,A,Q,R)
                               for (int j = k + 1; j < n; j++) 
                                   for (idx_t i = 0; i < m; i++)
                                       A[i][j] -= Q[i][k] * R[k][j];
                               
                           }
                       #endif
}








void run(State &state, pbsize_t pbsize) {
    pbsize_t m = pbsize - pbsize / 6; // 1000
    pbsize_t n = pbsize;              // 1200


    auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
    auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
    auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");

  real* dev_A = state.allocate_dev<real>(m*n);
    real* dev_R  = state.allocate_dev<real>(n*n);
      real* dev_Q = state.allocate_dev<real>(m*n);

    for (auto&& _ : state.manual()) {
        condition(m,n,A);
        {
            auto &&scope = _.scope();

 BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), m*n* sizeof(real), cudaMemcpyHostToDevice));


            kernel(m, n, dev_A, dev_R, dev_Q);

               BENCH_CUDA_TRY(    cudaMemcpy( A.data() ,dev_A,  m*n*sizeof(real), cudaMemcpyDeviceToHost )); 
BENCH_CUDA_TRY(    cudaMemcpy( R.data() ,dev_R,  n*n*sizeof(real), cudaMemcpyDeviceToHost )); 
BENCH_CUDA_TRY(    cudaMemcpy( Q.data() ,dev_Q,  m*n*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
        }
    }

           state.free_dev(dev_A);  
                  state.free_dev(dev_R);       
                    state.free_dev(dev_Q);  
}


