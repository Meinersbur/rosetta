// BUILD: add_benchmark(ppm=cuda,sources=[__file__, "cholesky-common.cxx"])
#include <rosetta.h>
#include "cholesky-common.h"


// https://dl.acm.org/doi/pdf/10.1145/3038228.3038237
// https://people.ast.cam.ac.uk/~stg20/cuda/cholesky/




__global__ void kernel0(pbsize_t n, idx_t j, real * A) {
    A[j*n +j] = std::sqrt( A[j*n +j] );
}



__global__ void kernel1(pbsize_t n, idx_t j, real * A) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < n && i > j   ) 
        A[i*n +j] /= A[j*n +j];
}


__global__ void kernel2(pbsize_t n, idx_t j, real * A) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t k = blockDim.y * blockIdx.y + threadIdx.y;


      if (j < n && j < i && i < n && j < k && k <= i)
        A[i*n + k] -= A[i*n + j] * A[k*n + j];
}


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



static void kernel_polly(pbsize_t n, real *dev_A) {
   const  unsigned  int threadsPerBlock = 256;

    
        for (idx_t j = 0; j < n; j++) { 
            kernel0<<<1,1>>>(n,j,dev_A);

            kernel1<<<threadsPerBlock,num_blocks(n,threadsPerBlock)>>>(n,j,dev_A);

            dim3 block {threadsPerBlock/32, 32, 1};
            dim3 grid {num_blocks(n,block.x), num_blocks(n,block.y), 1}; 
            kernel2<<<block,grid>>>(n,j,dev_A);
        }
}






void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


    auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
   real *dev_A = state.allocate_dev<real>(n * n );


    for (auto&& _ : state.manual()) {
        ensure_posdefinite(n, A);

        {
            auto &&scope = _.scope();

            cudaMemcpy(dev_A, A.data(), n * n * sizeof(real), cudaMemcpyHostToDevice);
       
            kernel_polly(n, dev_A);

            cudaMemcpy( A.data() ,dev_A,  n *  n * sizeof(real), cudaMemcpyDeviceToHost); 
        }

        cudaDeviceSynchronize();
    }

      state.free_dev(dev_A);
}
