#include <cooperative_groups.h>
#include<math.h>
namespace cg = cooperative_groups;
#include "stdio.h"
// #include "cudarntim"
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}
// #define NUMPTHREAD (4)
#define SLP1 asm volatile("nanosleep.u32 1000;");

template<class REAL, int ILP>
__global__ void simulator(REAL* in,REAL* in2, REAL* out, int N)
{
  // out[threadIdx.x+blockIdx.x*blockDim.x]=in[threadIdx.x+blockIdx.x*blockDim.x];
  for(int tid=threadIdx.x+blockIdx.x*blockDim.x; tid<N; tid+=blockDim.x*gridDim.x*ILP)
  {
    double tmp[ILP];
    #pragma unroll
    for(int ip=0; ip<ILP; ip++)
    {
      int local_tid=tid+blockDim.x*gridDim.x*ip;
      if(local_tid>=N)break;
      tmp[ip]=in[local_tid]+in2[local_tid];
    }
    //store
    #pragma unroll 
    for(int ip=0; ip<ILP; ip++)
    {
      int local_tid=tid+blockDim.x*gridDim.x*ip;
      if(local_tid>=N)break;
      out[local_tid]=tmp[ip];
    }
  }
}
template<class REAL, int ILP>
void RunBasic(int N, int tbpsm=1)
{
  int executeSM=0;
  int bdimx=256;
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );

  auto execute_kernel=simulator<REAL,ILP>;
  int numBlocksPerSm_current=1000;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdimx, executeSM);
  numBlocksPerSm_current=numBlocksPerSm_current>=tbpsm?tbpsm:numBlocksPerSm_current;
  dim3 block_dim(bdimx);
  // dim3 grid_dim(N/bdimx/1);
  dim3 grid_dim(sm_count*numBlocksPerSm_current);//numBlocksPerSm_current*sm_count);
  

  REAL* h_data_in=(REAL*)malloc(sizeof(REAL)*N);
  REAL* h_data_in_2=(REAL*)malloc(sizeof(REAL)*N);
  REAL* h_data_out=(REAL*)malloc(sizeof(REAL)*N);

  REAL* d_data_in;
  REAL* d_data_in_2;
  REAL* d_data_out;
  cudaCheckError();

  cudaMalloc(&d_data_in,sizeof(REAL)*N);
  cudaMalloc(&d_data_in_2,sizeof(REAL)*N);
  cudaMalloc(&d_data_out,sizeof(REAL)*N);

  for(int i=0; i<N; i++)
  {
    h_data_in[i]=N;
    h_data_in_2[i]=N;
  }
  cudaMemcpy(d_data_in, h_data_in, sizeof(REAL)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data_in_2, h_data_in_2, sizeof(REAL)*N, cudaMemcpyHostToDevice);
  // cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  for(int i=0; i<4000; i++)
    execute_kernel<<<grid_dim, block_dim>>>(d_data_in,d_data_in_2, d_data_out, N);
  cudaCheckError();
  cudaEventRecord(_forma_timer_start_,0);
  for(int i=0; i<1000; i++)
    execute_kernel<<<grid_dim, block_dim>>>(d_data_in,d_data_in_2, d_data_out, N);
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  cudaCheckError();
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  cudaMemcpy(h_data_out,d_data_out , sizeof(REAL)*N, cudaMemcpyDeviceToHost);
  cudaCheckError();
  printf("<%d,%d>\t%f\t%f\t%f\n",bdimx,grid_dim.x, (double)N*sizeof(REAL)/1024/1024,elapsedTime, (double)1000*3*N*sizeof(REAL)/elapsedTime/1000/1000/1000*1000);
  cudaFree(d_data_in);
  cudaFree(d_data_out);
  free(h_data_out);
  free(h_data_in);
  cudaDeviceReset();
}
template<class REAL>
void RunOverSub(int N)
{
  int executeSM=0;
  int bdimx=256;
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );

  auto execute_kernel=simulator<REAL,4>;
  int numBlocksPerSm_current=1000;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdimx, executeSM);
  // numBlocksPerSm_current=2;
  printf("blkpsm %d ",numBlocksPerSm_current);
  dim3 block_dim(bdimx);
  dim3 grid_dim(min(N/bdimx/1,8192*8192));
  // dim3 grid_dim(sm_count);//numBlocksPerSm_current*sm_count);
  

  REAL* h_data_in=(REAL*)malloc(sizeof(REAL)*N);
  REAL* h_data_in_2=(REAL*)malloc(sizeof(REAL)*N);
  REAL* h_data_out=(REAL*)malloc(sizeof(REAL)*N);

  REAL* d_data_in;
  REAL* d_data_in_2;
  REAL* d_data_out;

  cudaMalloc(&d_data_in,sizeof(REAL)*N);
  cudaMalloc(&d_data_in_2,sizeof(REAL)*N);
  cudaMalloc(&d_data_out,sizeof(REAL)*N);
  cudaCheckError();
  for(int i=0; i<N; i++)
  {
    h_data_in[i]=N;
  }
  cudaMemcpy(d_data_in, h_data_in, sizeof(REAL)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data_in_2, h_data_in_2, sizeof(REAL)*N, cudaMemcpyHostToDevice);
  cudaCheckError();
  // cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEvent_t warmupstart, warmupstop;
  cudaEventCreate(&warmupstart);
  cudaEventCreate(&warmupstop);
  cudaEventRecord(warmupstart, 0);
  { 
      execute_kernel<<<grid_dim, block_dim>>>(d_data_in,d_data_in_2, d_data_out, N);
  }
  cudaEventRecord(warmupstop, 0);
  cudaDeviceSynchronize();
  cudaEventSynchronize(warmupstop);
  float inc;
  cudaEventElapsedTime(&inc, warmupstart, warmupstop);

  for (int s = 0; s < 2000/inc; s++) {
    execute_kernel<<<grid_dim, block_dim>>>(d_data_in,d_data_in_2, d_data_out, N);
  }
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaEventRecord(_forma_timer_start_,0);
  int iter=(int)min(2000/inc,1.0);
  for(int i=0; i<iter; i++)
    execute_kernel<<<grid_dim, block_dim>>>(d_data_in,d_data_in_2, d_data_out, N);
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  cudaMemcpy(h_data_out,d_data_out , sizeof(REAL)*N, cudaMemcpyDeviceToHost);
  for(int i=0; i<N; i++)
  {
    if(h_data_in[i]!=h_data_out[i]){
      printf("%f,%f\n",h_data_in[i],h_data_out[i]);
      exit(0);
    }
  }
  cudaCheckError();
  printf("<%d,%d>\t%f\t%f\t%f\n",bdimx,grid_dim.x, (double)2*N*sizeof(REAL)/1024/1024,elapsedTime, (double)iter*3*N*sizeof(REAL)/elapsedTime/1000/1000/1000*1000);
  cudaFree(d_data_in);
  cudaFree(d_data_out);
  free(h_data_out);
  free(h_data_in);
  cudaDeviceReset();
}

int main(int argc, char const *argv[])
{
  
  // size_t N=1024*256*16*32;
  // size_t N=2304*2304;
  // RunOverSub<double>(N);
  // RunBasic<double,4>(N);

  for(int N=256*108*4; N<=8192*8192*2*2; N*=2)
  {
    // printf();
    RunOverSub<double>(N);
    // break;
  }
  // for(int i=1; i<=8; i*=2)
  // {
  //   size_t N=2304*2304*16;
  //   // printf();
  //   RunBasic<double,4>(N,i);
  //   // break;
  // }
  return 0;
}
