#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>

#include "cudaLens.hpp"


__global__ void lensPartialCalc(float* d_lx1, float* d_lx2, float* d_lm, float* d_x1, float* d_x2, float* d_p1, float* d_p2, const int Nl);
__global__ void lensFinishCalc(float* d_x1, float* d_x2, float* d_p1, float* d_p2, float ks, float g);
__device__ inline void atomicFloatAdd(float *address, float val);


//MAIN CALCULATION: Calculating total deflections
//==========================================================
void runCudaLens(float* d_lx1,float* d_lx2,float* d_lm,float* d_x1,float* d_x2,float* d_p1,float* d_p2,long Nl,float ks,float g,Stopwatch& kernelTimer1,Stopwatch& kernelTimer2){
  cudaError_t err;
  dim3 threads( BLOCK_SIZE, 1, 1);
  dim3 grid( CHUNK_SIZE/threads.x, 1, 1);
  grid.y = (int) ceil( (float)Nl/(float)BLOCK_SIZE );

  kernelTimer1.start();
  lensPartialCalc<<<grid,threads>>>(d_lx1, d_lx2, d_lm, d_x1, d_x2, d_p1, d_p2, Nl);
  cudaThreadSynchronize();
  kernelTimer1.stop();

  err = cudaGetLastError();
  if( err != cudaSuccess ){
	fprintf(stderr,"Error: %s - in \"lensPartialCalc\" \n",cudaGetErrorString(err));
  }


  grid.y = 1;
  kernelTimer2.start();
  lensFinishCalc<<< grid, threads>>>(d_x1, d_x2, d_p1, d_p2, ks, g);
  cudaThreadSynchronize();
  kernelTimer2.stop();
  
  err = cudaGetLastError();
  if( err != cudaSuccess ){
      fprintf(stderr,"Error: %s - in \"lensFinishCalc\" \n",cudaGetErrorString(err));
  }	
}


//MAIN CALCULATION: Calculating partial deflections
//==========================================================
__global__ void lensPartialCalc(float* d_lx1, float* d_lx2, float* d_lm, float* d_x1, float* d_x2, float* d_p1, float* d_p2, const int Nl){
  unsigned int tx = threadIdx.x;
  unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int gy = blockIdx.y * blockDim.x + threadIdx.x;


  __shared__ float3 lens[BLOCK_SIZE];
  lens[tx].x = d_lx1[gy];
  lens[tx].y = d_lx2[gy];
  lens[tx].z = d_lm[gy];
  __syncthreads();


  float x1 = d_x1[gx];
  float x2 = d_x2[gx];
  float p1 = 0.;
  float p2 = 0.;
  float d,d1,d2;
  float3 ll;
  unsigned int by = Nl - blockIdx.y * BLOCK_SIZE;
  if(by > BLOCK_SIZE){  by = BLOCK_SIZE;  }


#pragma unroll 8            //10 FLOPS (removes one division at cost of 2 mult)
  for(unsigned int n=0; n < by; n++){
    ll = lens[n];
    d1 = x1 - ll.x;    //1 FLOP, 1 shared
    d2 = x2 - ll.y;    //1 FLOP, 1 shared
    
    d = d1*d1 + d2*d2;      //3 FLOPS
    d = ll.z/d;        //1 FLOP
    
    p1 += d1*d;             //2 FLOPS
    p2 += d2*d;             //2 FLOPS
  }

  
  atomicFloatAdd( &d_p1[gx], p1);
  atomicFloatAdd( &d_p2[gx], p2);
}


//MAIN CALCULATION: Finalizing deflections
//==========================================================
__global__ void lensFinishCalc(float* d_x1,float* d_x2,float* d_p1,float* d_p2, float ks, float g){
	unsigned int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	d_x1[gx] = d_x1[gx] * (1.0 - ks - g) - d_p1[gx] ;
	d_x2[gx] = d_x2[gx] * (1.0 - ks + g) - d_p2[gx];
}


//MAIN CALCULATION: Atomic add on GPU memory locations
//==========================================================
__device__ inline void atomicFloatAdd(float* address,float val){
  //       int i_val = __float_as_int(val);
       int i_val = __float2int_rn(val);
       int tmp0 = 0;
       int tmp1;

       while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0 )
       {
               tmp0  = tmp1;
	       //               i_val = __float_as_int(val + __int_as_float(tmp1));
               i_val = __float2int_rn(val + __int2float_rn(tmp1));
       }
}








//Supporting functions
//==========================================================
void myGetDeviceCount(int* raw_device_count){
  cudaGetDeviceCount(raw_device_count);
}

int _cudaGetDeviceCount(){
  int count;
  cudaGetDeviceCount(&count);
  return count; 
}

void _cudaSetDevice(int device){
  cudaSetDevice(device);
}

void _cudaGetDevName(char* name,int* rate, int* mps){
  int devNo;
  cudaGetDevice(&devNo);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,devNo);
  strcpy(name,prop.name);
  *rate = prop.clockRate;
  *mps  = prop.multiProcessorCount;
}

void lensCudaMalloc(float** data, size_t size){
  cudaMalloc((void**) data,size);
}

void lensCudaFree(float** data){
  cudaFree(*data);
  data = NULL;
}

