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
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
// NVML
#include <nvml.h>

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 8
#define N 8
#define K 4

// // GEMM configuration.
// #define M_TILES 512
// #define N_TILES 512
// #define K_TILES 512
// #define TILE 4
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




__global__ void WMMAF16TensorCore(double*A, double*B,  double*C, int tile_c, int repeat)
{

  wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag[ILP];
  int id_warps=blockIdx.x/32;
  #pragma unroll
  for(int i=0; i<ILP;i++)
  {
    wmma::fill_fragment(c_frag[i], 0.0f);
  }
  
  wmma::load_matrix_sync(a_frag, A , M);
  wmma::load_matrix_sync(b_frag, B , N);
  for(int i=0; i<repeat; i++)
  {
    #pragma unroll
    for(int i=0; i<ILP;i++)
    {
      wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
    }
  }
  #pragma unroll
  for(int i=0; i<ILP;i++)
  {
    wmma::store_matrix_sync(C+N*(blockIdx.x+i*gridDim.x) + tile_c*(id_warps), c_frag[i], tile_c, wmma::mem_row_major);
  }
}
#include<unistd.h>   

int main(int argc, char const *argv[])
{
  cudaError_t cuda_status;
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  int num_warps=8;
  int tilesC=sm_count*ILP*N;
  int sizeofmem=sizeof(double)*16*16;
  int sizeofrmem=sizeofmem*sm_count*ILP*num_warps;

  double*h_mat=(double*)malloc(sizeofmem);
  double*h_mat2=(double*)malloc(sizeofmem);
  double*d_A;
  double*d_B;
  double*d_C;
  cudaMalloc((void**)&d_A, sizeofmem);
  cudaMalloc((void**)&d_B, sizeofmem);
  cudaMalloc((void**)&d_C, sizeofrmem);
  double*h_C=(double*)malloc(sizeofrmem);

  for(int i=0; i<16*16; i++)
  {
    h_mat[i]=i+1;
  }

  int repeat=10000; 
  int outrepeat=10000;
  cudaMemcpy(d_A, h_mat, sizeofmem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_mat, sizeofmem, cudaMemcpyHostToDevice);
  
  float milliseconds = 0;
  nvmlReturn_t result;
  nvmlDevice_t device;
  result=nvmlInit();
  nvmlEnableState_t mode;
  result = nvmlDeviceGetHandleByIndex(0, &device);
  result=nvmlDeviceGetPowerManagementMode(device, &mode);

  #pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 1)
    {
      for(int i=0; i<20;i++)
      {
        unsigned int power1, power2;
        result=nvmlDeviceGetPowerUsage(device,&power1);
        // cuda_status = cudaDeviceSynchronize();
        result=nvmlDeviceGetPowerUsage(device,&power2);

        assert(NVML_SUCCESS == result);
        printf("%d power from %u W to %u W\n", i,
                        power1/1000, power2/1000);
        sleep(1);
      }
    }
    if (omp_get_thread_num() == 0){
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      for(int i=0; i<outrepeat; i++)
      {
        WMMAF16TensorCore<<<sm_count,32*num_warps>>>(d_A, d_B, d_C, tilesC, repeat);
      }
      cudaEventRecord(start);
      for(int i=0; i<outrepeat; i++)
      {
        WMMAF16TensorCore<<<sm_count,32*num_warps>>>(d_A, d_B, d_C, tilesC, repeat);
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
      printf("[+] TFLOPS: %.2f\n", ((float)M*tilesC*K*num_warps *2)*repeat*outrepeat / milliseconds / 1e9);
  
    }
  }
  nvmlShutdown();

  cudaMemcpy(h_C, d_C, sizeofrmem, cudaMemcpyDeviceToHost);

  return 0;
}
