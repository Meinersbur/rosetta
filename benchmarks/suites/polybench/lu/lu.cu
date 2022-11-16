// BUILD: add_benchmark(ppm=cuda,sources=[__file__,"lu-common.cxx"])

#include <rosetta.h>
#include "lu-common.h"

static 
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



__global__ void kernel_div(pbsize_t n, real* A,idx_t k) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x +k+1;
 


    if (i < n  ) 
              A[i*n+k] /= A[k*n+k];
}


__global__ void kernel_A1(pbsize_t n, real* A,idx_t k) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x +k+1;
   idx_t j = blockDim.y * blockIdx.y + threadIdx.y +k+1;


    if (i < n && i <= j && j < n) 
              A[i*n+j] -= A[i*n+k] * A[k*n+j];
}

__global__ void kernel_A2(pbsize_t n, real* A,idx_t k) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x +k+1;
   idx_t j = blockDim.y * blockIdx.y + threadIdx.y +k+1;


    if (i < n && j < i ) 
              A[i*n+j] -= A[i*n+k] * A[k*n+j];
}




static void 
 kernel(pbsize_t  n,  real* A) {
const  unsigned  int threadsPerBlock = 256;

  for (idx_t k = 0; k < n-1; k++) {
    kernel_div<<<threadsPerBlock,num_blocks(n-(k+1),threadsPerBlock)>>>(n,A,k);

    
      dim3 block {threadsPerBlock/32, 32,1};     
                    dim3 grid {num_blocks(n-(k+1),block.x), num_blocks(n-(k+1),block.y), 1}; 
                    kernel_A1<<<block,grid>>>(n,A,k);        
          kernel_A2<<<block,grid>>>(n,A,k);   
  }



#if 0
#pragma omp parallel default(none) firstprivate(n,A)
                       {
                           for (idx_t k = 0; k < n-1; k++) {
#pragma omp for schedule(static)
                               for (idx_t i = k + 1; i < n; i++) {
                                                          A[i][k] /= A[k][k];
                               }

#pragma omp for collapse(2) /* schedule(static) */
                               for (idx_t i = k + 1; i < n; i++)
                                   for (idx_t j = i; j < n; j++) {
                                                                 A[i][j] -= A[i][k] * A[k][j];
                                   }

#pragma omp for collapse(2) /* schedule(static) */
                               for (idx_t i = k + 1; i < n; i++)
                                   for (idx_t j = k+1; j < i; j++) {
                                                           A[i][j] -= A[i][k] * A[k][j];
                                   }
                           }



                       }
                       #endif
}




void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");

    real* dev_A = state.allocate_dev<real>(n*n);

    for (auto&& _ : state.manual()) {
        ensure_fullrank(n, A);
        {
            auto &&scope = _.scope();


BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), n*n* sizeof(real), cudaMemcpyHostToDevice));
            kernel(n, dev_A);
             BENCH_CUDA_TRY(    cudaMemcpy( A.data() ,dev_A,  n*n*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
        }
    }

         state.free_dev(dev_A);  
}
