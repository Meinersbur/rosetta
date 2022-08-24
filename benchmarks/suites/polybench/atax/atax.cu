// BUILD: add_benchmark(ppm=cuda)
#include "rosetta.h"





__global__ void kernel1(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: use CUDA memset instead
    if ( i < n) 
            y[i] = 0;
}

__global__ void kernel2(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: use CUDA memset instead
    if ( i < m) 
        tmp[i] = 0;
}

__global__ void kernel3(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ( i >= m || j >= n) return; 


            tmp[i] += A[i][j] * x[j];
}

__global__ void kernel4(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ( i >= m || j >= n) return; 

            y[j] +=   A[i][j] * tmp[i];
}



void run(State& state, int n) {
    // n is 5%-20% larger than m
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 10;

    auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);
    auto x = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false);
    auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true);
    auto tmp = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false);


    double *dev_A;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * sizeof(double)));


    cudaMemcpy(dev_A, A.data(), n * sizeof(double), cudaMemcpyHostToDevice);



    // TODO: dim3 dimBlock(16, 16, 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    //state.PauseTiming();
    for (auto &&_ : state.manual()) {
        //state.PauseTiming();
        cudaStream_t stream = 0;
        //    cuda_event_timer raii(state, true, stream); 
        //state.ResumeTiming();

        {
            auto &&scope = _.scope();
            kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, dev_A);
        }

        // TODO: Is the invocation already blocking?
        cudaMemcpy( A.data(), dev_A, n * sizeof(double), cudaMemcpyDeviceToHost ); cudaDeviceSynchronize();

    }

    cudaFree(dev_A);



}


