template<typename T, int BS>
__global__ void calc_grad_kernel(T** Ainv_list, T** grad_lapl_list, T* grad, int N, int row_stride, int elec)
{
  int tid = threadIdx.x;
  int NB  = N / BS + ((N % BS) ? 1 : 0);
  __shared__ T *Ainv, *grad_lapl;
  if (tid == 0)
  {
    Ainv      = Ainv_list[blockIdx.x];
    grad_lapl = grad_lapl_list[blockIdx.x] + 4 * elec * row_stride;
  }
  __syncthreads();
  const int BS1 = BS + 1;
  const int BS2 = 2 * BS1;
#ifdef QMC_COMPLEX
  __shared__ uninitialized_array<T, BS> Ainv_colk_shared;
  __shared__ uninitialized_array<T, 3 * BS1> ratio_prod;
#else
  __shared__ T Ainv_colk_shared[BS];
  __shared__ T ratio_prod[3 * BS1];
#endif
  ratio_prod[tid]       = 0.0f;
  ratio_prod[BS1 + tid] = 0.0f;
  ratio_prod[BS2 + tid] = 0.0f;
  // This is *highly* uncoallesced, but we just have to eat it to allow
  // other kernels to operate quickly.
  __syncthreads();
  for (int block = 0; block < NB; block++)
  {
    int col = block * BS + tid;
    if (col < N)
      Ainv_colk_shared[tid] = Ainv[col * row_stride + elec];
    __syncthreads();
    if (col < N)
    {
      ratio_prod[tid] += Ainv_colk_shared[tid] * grad_lapl[0 * row_stride + col];
      ratio_prod[BS1 + tid] += Ainv_colk_shared[tid] * grad_lapl[1 * row_stride + col];
      ratio_prod[BS2 + tid] += Ainv_colk_shared[tid] * grad_lapl[2 * row_stride + col];
    }
    __syncthreads();
  }
  // Now, we have to sum
  for (unsigned int s = BS / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      ratio_prod[tid] += ratio_prod[tid + s];             // grad_x
      ratio_prod[BS1 + tid] += ratio_prod[BS1 + tid + s]; // grad_y
      ratio_prod[BS2 + tid] += ratio_prod[BS2 + tid + s]; // grad_z
    }
    __syncthreads();
  }
  if (tid < 3)
    grad[3 * blockIdx.x + tid] = ratio_prod[tid * BS1];
}

void calc_gradient(float* Ainv_list[],
                   float* grad_lapl_list[],
                   float grad[],
                   int N,
                   int row_stride,
                   int elec,
                   int numWalkers)
{
  const int BS = 32;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);
  calc_grad_kernel<float, BS>
      <<<dimGrid, dimBlock, 0, gpu::kernelStream>>>(Ainv_list, grad_lapl_list, grad, N, row_stride, elec);
}
