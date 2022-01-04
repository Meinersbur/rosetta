template<typename T, int COLBS, int DIM = 3>
__global__ void calcGradients_kernel(const int n,
                                     const T* const Ainvrow[],
                                     const T* const dpsiMrow[],
                                     T* const grads_now)
{
  const int iw                    = blockIdx.x;
  const T* __restrict__ invRow    = Ainvrow[iw];
  const T* __restrict__ dpsiM_row = dpsiMrow[iw];

  constexpr int SUM_SIZE = DIM * COLBS;
  __shared__ uninitialized_array<T, SUM_SIZE> sum;
  const int tid = threadIdx.x;
  for (int idim = 0; idim < DIM; idim++)
    sum[idim * COLBS + tid] = T(0);

  const int num_col_blocks = (n + COLBS - 1) / COLBS;
  for (int ib = 0; ib < num_col_blocks; ib++)
  {
    const int col_id = ib * COLBS + tid;
    for (int idim = 0; idim < DIM; idim++)
      if (col_id < n)
        sum[idim * COLBS + tid] += invRow[col_id] * dpsiM_row[col_id * DIM + idim];
  }

  for (int iend = COLBS / 2; iend > 0; iend /= 2)
  {
    __syncthreads();
    for (int idim = 0; idim < DIM; idim++)
      if (tid < iend)
        sum[idim * COLBS + tid] += sum[idim * COLBS + tid + iend];
  }

  if (tid == 0)
    for (int idim = 0; idim < DIM; idim++)
      grads_now[iw * DIM + idim] = sum[idim * COLBS];
}

cudaError_t calcGradients_cuda(cudaStream_t& hstream,
                               const int n,
                               const float* const Ainvrow[],
                               const float* const dpsiMrow[],
                               float* const grads_now,
                               const int batch_count)
{
  if (batch_count == 0)
    return cudaSuccess;

  const int COLBS = 64;
  dim3 dimBlock(COLBS);
  dim3 dimGrid(batch_count);
  calcGradients_kernel<float, COLBS><<<dimGrid, dimBlock, 0, hstream>>>(n, Ainvrow, dpsiMrow, grads_now);
  return cudaPeekAtLastError();
}
