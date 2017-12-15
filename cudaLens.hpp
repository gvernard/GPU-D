//#define CHUNK_SIZE 524288  //2^19
#define CHUNK_SIZE 131072    //2^17
//#define CHUNK_SIZE 65536   //2^16
//#define CHUNK_SIZE 16384   //2^14
//#define CHUNK_SIZE 128     //2^7

//#define BLOCK_SIZE 512     //2^9
#define BLOCK_SIZE 256       //2^8

#define LENS_CHUNK 4191304   //2^22
//#define LENS_CHUNK 1048576 //2^20
//#define LENS_CHUNK 262144  //2^18
//#define LENS_CHUNK 131072  //2^17
//#define LENS_CHUNK 32768   //2^15

#include "stopwatch.hpp"

void runCudaLens(double* d_lx1, double* d_lx2, double* d_lm, double* d_x1, double* d_x2, double* d_p1, double* d_p2, long Nl, double ks,double g,Stopwatch& kernelTimer1,Stopwatch& kernelTimer2);

void myGetDeviceCount(int* raw_device_count);
void _cudaGetDevName(char* name,int* rate, int* mps);
int  _cudaGetDeviceCount();
void _cudaSetDevice(int device);
void lensCudaMalloc(double** data, size_t size);
void lensCudaFree(double** data);
