#include <string>
#include <vector>
#include <cstdlib>

#include "stopwatch.hpp"

typedef struct{
  double *lx1;
  double *lx2;
  double *lm;
  long Nl;
  long Nleff;
} ImagePlane;

template<typename T>
T** create2DArray(int xdim,int ydim){
	double** data = (T** )calloc(xdim, sizeof(T*));
	for( int i=0; i<xdim; ++i ) {
		data[i] = (T*)calloc(ydim, sizeof(T));
		for( int j=0; j<ydim; ++j ) {
			data[i][j] = (T)0.0;
		}
	}
	return data;
}

void       genRandomPositions(double* data1,double* data2,int chunkSize,double rx,double ry,long int seed);
void       genRandomPositions(double** data1,double** data2,int d,int chunkSize,double rx,double ry,long int seed);
void       binSourcePlane(long long int* nray,long long int* exclude,double** source_plane,double* y1,double* y2,int res,double ss2);
void       binSourcePlane(long long int* nray,long long int* exclude,int ndevices,double** source_plane,double** y1,double** y2,int res,double ss2,long m,long cycles);
ImagePlane setupLensPositions(const char* lp_file,long Nl,double rx,double ry,double m);
ImagePlane readLensPositions(const char* lp_file,long* lens_count,double rx,double ry,double m);
double**    createMatrixDouble(int xdim,int ydim);
int        timeMachineSeed(std::vector<int> dev_list);

void writeSnapshot(const std::string& target_dir,double** y,double k,double g,double s,double ss2,double is_x,double is_y,long long int nray,long long int exclude,int res);
void reportProgress(long cycles,long long int nray,long long int exclude,int res,Stopwatch& totalTimer);
void readBin(const std::string& target_dir,double** y,int Nx,int Ny);
void writeBin(const std::string& target_dir,double** y,int dim,double k,double g,double s,double ss2,double avgmag,double avgrayspp);
void writePlain(const std::string& name,double** y,int dim,double k,double g,double ss2,double avgmag,double avgrayspp);
