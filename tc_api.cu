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
#define M 16
#define N 16
#define K 8

// GEMM configuration.
#define M_TILES 512
#define N_TILES 512
#define K_TILES 512
#define TILE 4
// #define M_TOTAL (M * M_TILES)
// #define N_TOTAL (N * N_TILES)
// #define K_TOTAL (K * K_TILES)
#define ILP 4

//__global__ void WMMAINT8()
using namespace nvcuda;
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}


__host__ void InitMatrix(float *A, float *B, float *C, int tiles)
{
  int M_TOTAL=(M );
  int N_TOTAL=(N );
  int K_TOTAL=(K );

  for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
    A[i] = (rand() % 1000 / 1000.0f);
  for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
    B[i] = (rand() % 1000 / 1000.0f);
  for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
    C[i] = rand() % 1000 / 1000.0f;
}



__global__ void WMMAF16TensorCore(float *A, float *B, float *C, float *D, int tiles)
{
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);
  
  int M_TOTAL=(M * tiles );
  int N_TOTAL=(N * tiles );
  int K_TOTAL=(K * tiles );

  wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag[ILP];
  wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b_frag[ILP];
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag[ILP];
  #pragma unroll
  for(int ilp=0; ilp<ILP; ilp++)
  {
    wmma::fill_fragment(c_frag[ilp], 0.0f);
  }
  

  // AB = A*B 
  int a_col, a_row, b_col, b_row, c_col, c_row;
  a_row = ix * M;
  b_row = iy * N;

  // wmma::load_matrix_sync(a_frag, A + 0 + a_row * M_TOTAL, M_TOTAL);
  #pragma unroll
  for(int ilp=0; ilp<ILP; ilp++)
  {
    wmma::load_matrix_sync(a_frag[ilp], A + 0 + 0 * M_TOTAL, M_TOTAL);
    for (int t = 0; t < a_frag[ilp].num_elements; t++) {
        a_frag[ilp].x[t] = wmma::__float_to_tf32(a_frag[ilp].x[t]);
    }
  }
  
  #pragma unroll
  for(int ilp=0; ilp<ILP; ilp++)
  {
    wmma::load_matrix_sync(b_frag[ilp], B + 0 + 0 * K_TOTAL, K_TOTAL);
    for (int t = 0; t < b_frag[ilp].num_elements; t++) {
          b_frag[ilp].x[t] = wmma::__float_to_tf32(b_frag[ilp].x[t]);
    }
  }

  #pragma unroll
  for(int ilp=0; ilp<ILP; ilp++)
  {
    for (int k=0; k<K_TOTAL; k+=K) {
      a_col = b_col = k;
      {
        wmma::mma_sync(c_frag[ilp], a_frag[ilp], b_frag[ilp], c_frag[ilp]);
      }
    }
  }

  c_col = b_row;
  c_row = a_row;
  // if (c_row < M_TOTAL && c_col < N_TOTAL) 
  #pragma unroll
  for(int ilp=0; ilp<ILP; ilp++)
  {
    for (int i = 0; i < c_frag[ilp].num_elements; i++) {
      c_frag[ilp].x[i] = c_frag[ilp].x[i];// + c1_frag.x[i]+c2_frag.x[i]+c3_frag.x[i];
    }
    wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag[ilp], N_TOTAL, wmma::mem_row_major);
  }
}

cudaError_t CalcWMMA(float *A, float *B, float *C, float *D, int tiles)
{
  int M_TOTAL=(M * tiles );
  int N_TOTAL=(N * tiles );
  int K_TOTAL=(K * tiles );

  cudaError_t cuda_status;
  dim3 gridDim, blockDim;
  // 16 warps in one block

  
  blockDim.x = 4 * WARP_SIZE; 
  blockDim.y = 4;

  gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
  gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);


  int repeat=1000;
  // for Performance Metrics
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  unsigned int power1, power2;
  // nvmlReturn_t result;
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
    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D, tiles);
  }
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);
  for(int i=0; i<repeat; i++)
  {
    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D, tiles);
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
  printf("[+] TFLOPS: %.2f\n", ((float)M_TOTAL * ILP * N_TOTAL* K_TOTAL * 2)*repeat / milliseconds / 1e9);
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
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  int tiles=sm_count * TILE;
  int M_TOTAL=(M * tiles);
  int N_TOTAL=(N * tiles);
  int K_TOTAL=(K * tiles);

  // Matrix on device
  float *A;
  float *B;
  float *C;
  float *D;

  // CUDA Unified Memory 
  cudaMallocManaged((void **)&A, sizeof(float) * M_TOTAL * K_TOTAL);
  cudaMallocManaged((void **)&B, sizeof(float) * K_TOTAL * N_TOTAL);
  cudaMallocManaged((void **)&C, sizeof(float) * M_TOTAL * N_TOTAL);
  cudaMallocManaged((void **)&D, sizeof(float) * M_TOTAL * N_TOTAL);
  
  // Init matrix A B C on host
  //InitHostMatrix(host_A, host_B, host_C);
  printf("[*] Initializing Matrix...\n");
  InitMatrix(A, B, C, tiles);
  printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
  printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
  printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);
  
  // computing gemm using tensor core
  printf("[*] Computing D = A * B +C with Tensor Cores...\n");
  // D = A * B +C, D holds the result after ret
  cuda_status = CalcWMMA(A, B, C, D,tiles);
  
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