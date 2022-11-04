// BUILD: add_benchmark(ppm=cuda)
#include "rosetta.h"







__global__ void kernel3(int m, int n, real * A, real *x, real *y, real *tmp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ( i >= m || j >= n) return; 
            tmp[i] += A[i * m + j] * x[j];
}


__global__ void kernel4(int m, int n, real * A, real *x, real *y, real *tmp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ( i >= m || j >= n) return; 

            y[j] +=   A[i*m +j] * tmp[i];
}


int num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


void run(State& state, int pbsize) {
    // n is 5%-20% larger than m
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 10;

    auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false);
    auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false);
    auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true);
    auto tmp = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ false);


    real *dev_A, *dev_x, *dev_y, *dev_tmp;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * m * sizeof(real))); // TODO: Runtime should do this
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_x, n *  sizeof(real)));
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_y, n *  sizeof(real)));
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_tmp, n *  sizeof(real)));

   


    int threadsPerBlock = 256;
    dim3 block (threadsPerBlock/32, 32, 1);
    dim3 grid (num_blocks(m,block.x), num_blocks(n,block.y), 1); 
   


    for (auto &&_ : state.manual()) {
        

        {
            auto &&scope = _.scope();

            cudaMemcpy(dev_A, A.data(), n * m * sizeof(real), cudaMemcpyHostToDevice);
            cudaMemset(dev_y, 0, n * sizeof(real) );
            cudaMemset(dev_tmp, 0, m * sizeof(real));

            kernel3<<<grid, block>>>(m,n,dev_A,dev_x, dev_y, dev_tmp);
            kernel4<<<grid, block>>>(m,n,dev_A,dev_x, dev_y, dev_tmp);

            cudaMemcpy( dev_y, y.data() , n * sizeof(double), cudaMemcpyDeviceToHost ); 
        }

        // TODO: Is the invocation already blocking?
       cudaDeviceSynchronize();

    }

    BENCH_CUDA_TRY(   cudaFree(dev_A));
    BENCH_CUDA_TRY(   cudaFree(dev_x));
    BENCH_CUDA_TRY(   cudaFree(dev_y));
    BENCH_CUDA_TRY(   cudaFree(dev_tmp));
}


