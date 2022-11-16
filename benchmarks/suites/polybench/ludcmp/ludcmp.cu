// BUILD: add_benchmark(ppm=cuda,sources=[__file__,"ludcmp-common.cxx"])

#include "ludcmp-common.h"
#include <rosetta.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>








struct minus_Aik_times_Akj : public thrust:: unary_function<real,real>{
  pbsize_t  n;
    thrust::device_ptr<real> A;
    idx_t i;
    idx_t j;

minus_Aik_times_Akj(pbsize_t n,thrust::device_ptr<real> A, idx_t i, idx_t j) : n(n), A(A), i(i), j(j)  {  }

  __host__ __device__ real operator()(pbsize_t k) const
  {
    return -( A[i*n+k] * A[k*n+j]);
  }
};

struct minus_Aij_times_yj : public thrust:: unary_function<real,real>{
  pbsize_t  n;
    thrust::device_ptr<real> A;
     thrust::device_ptr<real> y;
    idx_t i;


minus_Aij_times_yj(pbsize_t n,thrust::device_ptr<real> A,  thrust::device_ptr<real> y, idx_t i) : n(n), A(A), y(y), i(i)  {  }

  __host__ __device__ real operator()(pbsize_t j) const  {
    return -(A[i*n+j] * y[j]);
  }
};



struct minus_Aij_times_xj : public thrust:: unary_function<real,real>{
  pbsize_t  n;
    thrust::device_ptr<real> A;
     thrust::device_ptr<real> x;
    idx_t i;


minus_Aij_times_xj(pbsize_t n,thrust::device_ptr<real> A,  thrust::device_ptr<real> x, idx_t i) : n(n), A(A),x(x), i(i)  {  }

  __host__ __device__ real operator()(pbsize_t j) const  {
    return -(A[i*n+j] * x[j]);
  }
};







static void kernel(pbsize_t  n,      thrust::device_ptr<real> A,    thrust::device_ptr<real> b,     thrust::device_ptr<real> x,     thrust::device_ptr<real> y) {
// TODO: The LU-Decomposition is identical to suites.polyench.lu, should use comparable algorithms.
  for (idx_t i = 0; i < n; i++) {
    for (idx_t  j = 0; j < i; j++) {
        real w  =  A[i*n+j];       
              minus_Aik_times_Akj op{n,A,i,j}; 
        w = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0), 
                                    thrust::make_counting_iterator(0)+j,
                                     op,
                                     w,
                                     thrust::plus<real>());
         A[i*n+j] =  w / A[j*n+j];
    }
 for (idx_t j = i; j < n; j++) {      
        real w  =  A[i*n+j];        
          minus_Aik_times_Akj op{n,A,i,j};
        w = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0), 
                                    thrust::make_counting_iterator(0)+i,
                                     op,
                                     w,
                                     thrust::plus<real>());
         A[i*n+j] = w;
 }

   for (idx_t i = 0; i < n; i++) {
        real w = b[i];
                  minus_Aij_times_yj op{n,A,y,i};
                w = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0), 
                                    thrust::make_counting_iterator(0)+i,
                                     op,
                                     w,
                                     thrust::plus<real>());
                                     y[i] = w;
   }
     for (idx_t i = n - 1; i >= 0; i--) { 
          real w = y[i];
             minus_Aij_times_xj op{n,A,x,i};
                w = thrust::transform_reduce(
                                    thrust::device,
                                      thrust::make_counting_iterator(0)+i+1, 
                                    thrust::make_counting_iterator(0)+n,
                                     op,
                                     w,
                                     thrust::plus<real>());
              x[i] = w / A[i*n+i];
     }
}


#if 0
  for (idx_t i = 0; i < n; i++) {
    for (idx_t  j = 0; j < i; j++) {
      real w = A[i][j];
#pragma omp parallel for default(none) firstprivate(i,j,n,A)  schedule(static) reduction(+:w)
      for (idx_t k = 0; k < j; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w / A[j][j];
    }
    for (idx_t j = i; j < n; j++) {
      real w = A[i][j];
#pragma omp parallel for default(none) firstprivate(i,j,n,A)  schedule(static) reduction(+:w)
      for (idx_t k = 0; k < i; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w;
    }
  }

  for (idx_t i = 0; i < n; i++) {
    real w = b[i];
#pragma omp parallel for default(none) firstprivate(i,n,A,y)  schedule(static) reduction(+:w)
    for (idx_t j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }

  for (idx_t i = n - 1; i >= 0; i--) {
    real w = y[i];
#pragma omp parallel for default(none) firstprivate(i,n,A,x)  schedule(static) reduction(+:w)
    for (idx_t j = i + 1; j < n; j++)
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
#endif
}





void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


   
  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");



    real* dev_A = state.allocate_dev<real>(n*n);
        real* dev_b = state.allocate_dev<real>(n);
            real* dev_x = state.allocate_dev<real>(n);
                real* dev_y = state.allocate_dev<real>(n);

  for (auto&& _ : state.manual()) {
      ensure_fullrank(n, A);
      {
          auto &&scope = _.scope();

          BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), n*n* sizeof(real), cudaMemcpyHostToDevice));
          BENCH_CUDA_TRY( cudaMemcpy(dev_b, b.data(), n* sizeof(real), cudaMemcpyHostToDevice));
          kernel(n,  thrust::device_pointer_cast(dev_A), thrust::device_pointer_cast(dev_b),  thrust::device_pointer_cast(dev_x), thrust::device_pointer_cast(dev_y));
              BENCH_CUDA_TRY(    cudaMemcpy( x.data() ,dev_x,  n*sizeof(real), cudaMemcpyDeviceToHost )); 


    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
      }
  }

         state.free_dev(dev_A);  
         state.free_dev(dev_b);  
         state.free_dev(dev_x);  
         state.free_dev(dev_y);  
}
