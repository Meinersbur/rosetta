// BUILD: add_benchmark(ppm=cuda,sources=[__file__, "cholesky-common.cxx"])
#include <rosetta.h>
#include "cholesky-common.h"


// https://dl.acm.org/doi/pdf/10.1145/3038228.3038237
// https://people.ast.cam.ac.uk/~stg20/cuda/cholesky/



__global__ void kernel1(pbsize_t n, idx_t j, real * A) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i==j)
        A[j*n +j] = std::sqrt( A[j*n +j] );
    else if (i > j && i < n) 
        A[i*n +j] /=  A[j*n +j];
}


__global__ void kernel2(pbsize_t n, idx_t j, real * A) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t k = blockDim.y * blockIdx.y + threadIdx.y;

    if (j < i && i < n && j < k && j <= i) 
        A[k*n  + i] -= A[j*n + i] * A[j*n + k];
}


static
int num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



static void kernel_polly(pbsize_t n, real *dev_A) {
    int threadsPerBlock = 256;

    dim3 block (threadsPerBlock/32, 32, 1);
    dim3 grid (num_blocks(n,block.x), num_blocks(n,block.y), 1); 

    
        for (idx_t j = 0; j < n; j++) { //c0
            kernel1<<<num_blocks(n,threadsPerBlock),threadsPerBlock>>>(n,j,dev_A);
            kernel2<<<grid,block>>>(n,j,dev_A);
        }
}






void run(State &state, int pbsize) {
    pbsize_t n = pbsize; // 2000

    auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
 

    real *dev_A;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * n * sizeof(real)));


    for (auto&& _ : state.manual()) {
        ensure_posdefinite(n, A);
        {
            auto &&scope = _.scope();

            cudaMemcpy(dev_A, A.data(), n * n * sizeof(real), cudaMemcpyHostToDevice);
       
            kernel_polly(n, dev_A);

            cudaMemcpy( dev_A, A.data() , n *  n * sizeof(double), cudaMemcpyDeviceToHost); 
        }

        cudaDeviceSynchronize();
    }

    BENCH_CUDA_TRY(cudaFree(dev_A));
}
