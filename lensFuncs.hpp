#include <string>
#include <vector>
#include <cstdlib>

#include "stopwatch.hpp"

typedef struct{
  float* lx1;
  float* lx2;
  float* lm;
  long Nl;
  long Nleff;
} ImagePlane;

template<typename T>
T** create2DArray(int xdim,int ydim){
	float** data = (T** )calloc(xdim, sizeof(T*));
	for( int i=0; i<xdim; ++i ) {
		data[i] = (T*)calloc(ydim, sizeof(T));
		for( int j=0; j<ydim; ++j ) {
			data[i][j] = (T)0.0;
		}
	}
	return data;
}

void       genRandomPositions(float* data1,float* data2,int chunkSize,float rx,float ry,long int seed);
void       genRandomPositions(float** data1,float** data2,int d,int chunkSize,float rx,float ry,long int seed);
void       binSourcePlane(long long int* nray,long long int* exclude,float** source_plane,float* y1,float* y2,int res,float ss2);
void       binSourcePlane(long long int* nray,long long int* exclude,int ndevices,float** source_plane,float** y1,float** y2,int res,float ss2,long m,long cycles);
ImagePlane setupLensPositions(const char* lp_file,long Nl,float rx,float ry,float m);
ImagePlane readLensPositions(const char* lp_file,long* lens_count,float rx,float ry,float m);
float**    createMatrixFloat(int xdim,int ydim);
int        timeMachineSeed(std::vector<int> dev_list);

void writeSnapshot(const std::string& target_dir,float** y,float k,float g,float s,float ss2,float is_x,float is_y,long long int nray,long long int exclude,int res);
void reportProgress(long cycles,long long int nray,long long int exclude,int res,Stopwatch& totalTimer);
void readBin(const std::string& target_dir,float** y,int Nx,int Ny);
void writeBin(const std::string& target_dir,float** y,int dim,float k,float g,float s,float ss2,float avgmag,float avgrayspp);
void writePlain(const std::string& name,float** y,int dim,float k,float g,float ss2,float avgmag,float avgrayspp);
