//////////////////////////////////////////////////////////////////////
// A simple example to show how CUDA WMMA API works with Tensor Cores
//    Created by Zong-Sheng Wang @ 2018/11/25
// Performance Tips:
//    To minimize bank conflicts, you should try to shift row or 
// column of matrics in shared memory
// cmd: 
//    $ nvcc -o main main.cu -arch sm_75

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// NVML
#include <nvml.h>

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 8
#define N 8
#define K 4

// GEMM configuration.
#define M_TILES 512
#define N_TILES 512
#define K_TILES 512

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)


//__global__ void WMMAINT8()
using namespace nvcuda;

__host__ void InitMatrix(double *A, double *B, double *C)
{
  for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
    A[i] = (rand() % 1000 / 1000.0f);
  for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
    B[i] = (rand() % 1000 / 1000.0f);
  for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
    C[i] = rand() % 1000 / 1000.0f;
}



__global__ void WMMAF16TensorCore(double *A, double *B, double *C, double *D)
{
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);
  
  wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> ab_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;
  
  wmma::fill_fragment(ab_frag, 0.0f);


  // AB = A*B
  int a_col, a_row, b_col, b_row, c_col, c_row;
  a_row = ix * M;
  b_row = iy * N;

  wmma::load_matrix_sync(a_frag, A + 0 + a_row * M_TOTAL, M_TOTAL);
  wmma::load_matrix_sync(b_frag, B + 0 + 0 * K_TOTAL, K_TOTAL);
  wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);
  for (int k=0; k<K_TOTAL; k+=K) {
    a_col = b_col = k;

    if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
      // Load the inputs
      // wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
      // wmma::load_matrix_sync(b_frag, B + b_col + b_col * K_TOTAL, K_TOTAL);

      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
  }

  // D = AB + C
  c_col = b_row;
  c_row = a_row;
  if (c_row < M_TOTAL && c_col < N_TOTAL) {
    // wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);

    // for (int i = 0; i < c_frag.num_elements; i++) {
    //   c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
    // }

    // Store the output
    wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
  }
}

cudaError_t CalcWMMA(double *A, double *B, double *C, double *D)
{
  cudaError_t cuda_status;
  dim3 gridDim, blockDim;
  // 16 warps in one block
  blockDim.x = 4 * WARP_SIZE; 
  blockDim.y = 4;

  gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
  gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);
  int repeat=500;
  // for Performance Metrics
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
   unsigned int power1, power2;
    nvmlReturn_t result;
  nvmlDevice_t device;
  nvmlEnableState_t mode;

    result=nvmlInit();
  result = nvmlDeviceGetHandleByIndex(0, &device);
  // assert(NVML_SUCCESS == result);
  result=nvmlDeviceGetPowerManagementMode(device, &mode);
  // printf("enabled = %d\n", mode);
  result=nvmlDeviceGetPowerUsage(device,&power1);
  // assert(NVML_SUCCESS == result);
  for(int i=0; i<repeat; i++)
  {
    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D);
  }
  cudaEventRecord(start);
  for(int i=0; i<repeat; i++)
  {
    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D);
  }
  
  
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cuda_status = cudaDeviceSynchronize();
  result=nvmlDeviceGetPowerUsage(device,&power2);
  // assert(NVML_SUCCESS == result);
  nvmlShutdown();

  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);

  // for Performance Metrics
  printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
  // references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
  printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2)*repeat / milliseconds / 1e9);
  printf("power from %u W to %u W\n", 
                      power1/1000, power2/1000);
    // printf("%f, ", gflops);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cuda_status;
}


int main()
{
  cudaError_t cuda_status;
  cuda_status = cudaSetDevice(0);
  if (cuda_status != cudaSuccess) {
    printf("cudaSetDevice failed! ");
    return 1;
  }


  // Matrix on device
  double *A;
  double *B;
  double *C;
  double *D;

  // CUDA Unified Memory 
  cudaMallocManaged((void **)&A, sizeof(double) * M_TOTAL * K_TOTAL);
  cudaMallocManaged((void **)&B, sizeof(double) * K_TOTAL * N_TOTAL);
  cudaMallocManaged((void **)&C, sizeof(double) * M_TOTAL * N_TOTAL);
  cudaMallocManaged((void **)&D, sizeof(double) * M_TOTAL * N_TOTAL);
  
  // Init matrix A B C on host
  //InitHostMatrix(host_A, host_B, host_C);
  printf("[*] Initializing Matrix...\n");
  InitMatrix(A, B, C);
  printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
  printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
  printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);
  
  // computing gemm using tensor core
  printf("[*] Computing D = A * B +C with Tensor Cores...\n");
  // D = A * B +C, D holds the result after ret
  cuda_status = CalcWMMA(A, B, C, D);
  
  cuda_status = cudaDeviceReset();
  if (cuda_status != cudaSuccess) {
    printf("cudaDeviceReset failed! ");
    return 1;
  }
  // Todo: Add a function to verify the result by using the result of CPU version implementation.

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);

  return 0;
}