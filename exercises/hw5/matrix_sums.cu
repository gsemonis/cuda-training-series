#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

const size_t DSIZE = 16384; // matrix side dimension
const int block_size = 256; // CUDA maximum is 1024
// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds)
{
  // we're going to do 1 block per row, so we'll index into the "matrix" by blockIDx
  float sum = 0.0f;
  unsigned mask = 0xFFFFFFFFU;
  int warpLane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;
  int warpCount = (blockDim.x + (warpSize - 1)) / warpSize;
  __shared__ float sharedMemory[32];
  size_t rowStartOffset = blockIdx.x * ds;
  size_t endoOfRow = rowStartOffset + ds;
  size_t idx = threadIdx.x + rowStartOffset;

  // stride but only to the end of the row
  while (idx < endoOfRow)
  {
    sum += A[idx];
    idx += blockDim.x;
  }

  // warp shuffle reduce
  for (size_t offset = warpSize / 2; offset > 0; offset >>= 1)
  {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  // now we know thead in lane 0 has the sum of it's warp
  if (warpLane == 0)
  {
    sharedMemory[warpID] = sum;
  }
  __syncthreads();

  // now in shared memory we have the sum of every warp chunk
  if (warpID == 0)
  {
    sum = 0.0f;
    // stride the shared memory
    idx = warpLane;
    while (idx < warpCount)
    {
      sum += sharedMemory[idx];
      idx += warpSize;
    }
    // reduce again
    // warp shuffle reduce
    for (size_t offset = warpSize / 2; offset > 0; offset >>= 1)
    {
      sum += __shfl_down_sync(mask, sum, offset);
    }
    if (warpLane == 0)
      sums[blockIdx.x] = sum;
  }
}
// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds)
  {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx + ds * i]; // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
  }
}
bool validate(float *data, size_t sz)
{
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz)
    {
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz);
      return false;
    }
  return true;
}
int main()
{

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE * DSIZE; i++) // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float)); // allocate device space for A
  cudaMalloc(&d_sums, DSIZE * sizeof(float));      // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure");           // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // cuda processing sequence step 1 is complete
  row_sums<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE))
    return -1;
  printf("row sums correct!\n");
  cudaMemset(d_sums, 0, DSIZE * sizeof(float));
  column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE))
    return -1;
  printf("column sums correct!\n");
  return 0;
}
