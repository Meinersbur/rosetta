// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"

static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



__global__ void kernel_A(pbsize_t n, real alpha, real beta,
                   real * A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
        idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;

if (i < n && j < n)
           A[i*n+j] +=  u1[i] * v1[j] + u2[i] * v2[j];
}


__global__ void kernel_x(pbsize_t n, real alpha, real beta,
                   real * A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;


if (i < n ) {
          for (idx_t j = 0; j < n; j++)
                                   x[i] +=  beta * A[j*n+i] * y[j];
}
}


__global__ void kernel_y(pbsize_t n, real alpha, real beta,
                   real * A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  

if (i < n )
             x[i] +=  z[i];
}

__global__ void kernel_w(pbsize_t n, real alpha, real beta,
                   real * A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    

if (i < n ) {
         for (idx_t j = 0; j < n; j++)
                                   w[i] +=  alpha * A[i*n+j] * x[j] ;
}
}







static void kernel(pbsize_t n, real alpha, real beta,
                   real * A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
              const  unsigned threadsPerBlock = 256;


{
             dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(n, block.y), 1};
    kernel_A <<<block ,grid >>> (n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z);
}


 kernel_x <<<threadsPerBlock ,num_blocks(n,threadsPerBlock) >>> (n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z);
 kernel_y <<<threadsPerBlock ,num_blocks(n,threadsPerBlock) >>> (n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z);
 kernel_w <<<threadsPerBlock ,num_blocks(n,threadsPerBlock) >>> (n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z);




#if 0
#pragma omp parallel default(none) firstprivate(n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z)
                       {
#pragma omp for collapse(2) schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   A[i][j] +=  u1[i] * v1[j] + u2[i] * v2[j];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   x[i] +=  beta * A[j][i] * y[j];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               x[i] +=  z[i];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   w[i] +=  alpha * A[i][j] * x[j] ;
                       }
                       #endif
}




void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto y = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "y");
  auto z = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "z");
  auto u1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u1");
  auto v1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v1");
  auto u2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u2");
  auto v2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v2");
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto w = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "w");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "x");

  real* dev_y = state.allocate_dev<real>(n);
 real* dev_z = state.allocate_dev<real>(n);
  real* dev_u1 = state.allocate_dev<real>(n);
   real* dev_v1 = state.allocate_dev<real>(n);
    real* dev_u2 = state.allocate_dev<real>(n);
     real* dev_v2 = state.allocate_dev<real>(n);
      real* dev_A = state.allocate_dev<real>(n*n);
       real* dev_w = state.allocate_dev<real>(n);
        real* dev_x = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
 BENCH_CUDA_TRY( cudaMemcpy(dev_y, y.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_z, z.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_u1, u1.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_v1 , v1.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_u2, u2.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_v2, v2.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), n* n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_w, w.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
 BENCH_CUDA_TRY( cudaMemcpy(dev_x, x.data(),  n* sizeof(real), cudaMemcpyHostToDevice));

    kernel(n, alpha, beta, dev_A, dev_u1, dev_v1, dev_u2, dev_v2, dev_w, dev_x, dev_y, dev_z);

            BENCH_CUDA_TRY(    cudaMemcpy( A.data() ,dev_A , n*n*sizeof(real), cudaMemcpyDeviceToHost )); 
               BENCH_CUDA_TRY(    cudaMemcpy( w.data() ,dev_w , n*sizeof(real), cudaMemcpyDeviceToHost )); 
                  BENCH_CUDA_TRY(    cudaMemcpy( x.data() ,dev_x , n*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }


state.free_dev(dev_y);    
state.free_dev(dev_z);     
state.free_dev(dev_u1);     
state.free_dev(dev_v1);     
state.free_dev(dev_u2);     
state.free_dev(dev_v2);     
state.free_dev(dev_A);     
state.free_dev(dev_w);     
state.free_dev(dev_x);     
}
