#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include <vector>
#include <unistd.h>

#include "lensFuncs.hpp"
#include "cudaLens.hpp"

using std::endl;
using std::cout;
using std::ofstream;
using std::ifstream;
using std::setprecision;
using std::setw;


// Quick local-use functions
template<typename T>
T min(T a, T b) { return a<b?a:b; }
template<typename T>
T max(T a, T b) { return a<b?b:a; }


int timeMachineSeed(std::vector<int> dev_list){
  int hnamelen = 10;
  char hostname[hnamelen];
  gethostname(hostname, hnamelen);
  int sum=0;
  for(int i=0;i<hnamelen;i++){
    sum += (int)hostname[i] ;
  }
  int mytime = ((long)time(NULL)) - 1261440000;
  int dev = dev_list[dev_list.size()-1];

  //  cout << sum << endl;
  //  cout << dev << endl;
  //  cout << mytime << endl;

  int final = (mytime % ((1+2*dev)*sum)) + (int) (mytime / ((1+2*dev)*sum));
  return final;
}

void genRandomPositions(float* data1,float* data2,int chunkSize,float Ix,float Iy,long int seed){
  srand48(seed);
  for( long i=0; i<chunkSize; i++ ){
    data1[i] = (drand48()*2.0-1.0)*Ix;
    data2[i] = (drand48()*2.0-1.0)*Iy;
  }
}

void genRandomPositions(float** data1,float** data2,int ndevices,int chunkSize,float Ix,float Iy,long int seed){

  for(int d=0;d<ndevices;d++){
    srand48(seed+d);
    for(long i=0;i<chunkSize;i++){
      data1[d][i] = (drand48()*2.0-1.0)*Ix;
      data2[d][i] = (drand48()*2.0-1.0)*Iy;
    }
  }
}

void binSourcePlane(long long int* nray,long long int* exclude,float** source_plane,float* y1,float* y2,int res,float ss2){
  float scalingFy = ((float)(res/2)/(float)(ss2));
  int i,j;

  for(long n=0; n<CHUNK_SIZE; n++){
    ++*nray;

    if( (fabs(y1[n]) < ss2) && (fabs(y2[n]) < ss2) ){
      i = (int)((y1[n]+ss2)*scalingFy);
      j = (int)((y2[n]+ss2)*scalingFy);
      ++source_plane[i][j];
    } else { 
      ++*exclude;
    }
  }
}

void binSourcePlane(long long int* nray,long long int* exclude,int ndevices,float** source_plane,float** y1,float** y2,int res,float ss2,long m,long cycles){
  float scalingFy = ((float)(res/2)/(float)(ss2));
  int i,j;

  for(int d=0;d<ndevices;d++){
    if( m+d < cycles ){

      for(long n=0; n<CHUNK_SIZE; n++){
	++*nray;
	
	if( (fabs(y1[d][n]) < ss2) && (fabs(y2[d][n]) < ss2) ){
	  i = (int) floor((y1[d][n]+ss2)*scalingFy);
	  j = (int) floor((y2[d][n]+ss2)*scalingFy);
	  ++source_plane[i][j];
	} else { 
	  ++*exclude;
	}
      }

    }
  }

}



ImagePlane setupLensPositions(const char* lp_file,long Nl, float rx, float ry, float mass){
  ImagePlane IP;
  long i;
  
  /* Random lens positions within the image plane */
  IP.lx1 = (float*) calloc(Nl, sizeof(float));
  if(IP.lx1 == NULL){ cout << "Error: out of mem\n"; }
  IP.lx2 = (float*) calloc(Nl, sizeof(float));
  if(IP.lx2 == NULL){ cout << "Error: out of mem\n"; }
  IP.lm  = (float*) calloc(Nl, sizeof(float));
  if(IP.lm == NULL){  cout << "Error: out of mem\n"; }
  IP.Nl = Nl;
  IP.Nleff = (int) ceil( (float)Nl/(float)BLOCK_SIZE ) * BLOCK_SIZE;//required for correct memory management on the GPU


  ofstream o_stream(lp_file);
  float x,y;
  long it;
  for (i=0;i<Nl;i++){
    // Create a uniform distribution inside an ellipse
    x = (float)(drand48()*2.0-1.0) * rx;
    y = (float)(drand48()*2.0-1.0) * ry;
    while( x*x/(rx*rx)+y*y/(ry*ry) > 1.0 ){
      x = (float)(drand48()*2.0-1.0) * rx;
      y = (float)(drand48()*2.0-1.0) * ry;
    }
    it        = (long)(x*1.e4);
    IP.lx1[i] = it/1.e4;
    it        = (long)(y*1.e4);
    IP.lx2[i] = it/1.e4;
    IP.lm[i]  = floor(mass*1.e3)/1.e3;
    drand48();// <--ATTENTION: ONLY FOR CONSISTENCY OF RANDOM NUMBER GENERATOR WITH PREVIOUS VERSIONS OF THE CODE
    
    //Generated random numbers are written to the 7th decimal (precision of float)
    o_stream << std::fixed << setw(10) << setprecision(4) << IP.lx1[i] << setw(10) << setprecision(4) << IP.lx2[i] << setw(5) << setprecision(2) << IP.lm[i] << endl;
  }
  o_stream.close();
  
  return IP;
}

ImagePlane readLensPositions(const char* lp_file, long* lens_count, float rx, float ry, float mass){
    ImagePlane IP;

    std::string str;
    ifstream myfile(lp_file);
    long i=0;
    while(true){
      getline(myfile,str);
      if ( myfile.eof() ) break;
      i++;
    }
    myfile.close();

    long Nl = i;
    *lens_count = Nl;
    IP.lx1 = (float*) calloc(Nl, sizeof(float));
    if(IP.lx1 == NULL){ cout << "Error: out of mem\n"; }
    IP.lx2 = (float*) calloc(Nl, sizeof(float));
    if(IP.lx2 == NULL){ cout << "Error: out of mem\n"; }
    IP.lm = (float*) calloc(Nl, sizeof(float));
    if(IP.lm == NULL){  cout << "Error: out of mem\n"; }
    IP.Nl = Nl;
    IP.Nleff = (int) ceil( (float)Nl/(float)BLOCK_SIZE ) * BLOCK_SIZE;

    float x,y,m;
    myfile.open(lp_file);
    for(i=0;i<Nl;i++){
      myfile >> x >> y >> m;
      IP.lx1[i] = x;
      IP.lx2[i] = y;
      IP.lm[i]  = m;
    }
    myfile.close();

    return IP;
}

float** createMatrixFloat(int xdim,int ydim){
 int i = 0, j = 0;

 float** data = (float**) calloc(xdim, sizeof(float*));
 for (i=0;i<xdim;i++){
   data[i] = (float*) calloc(ydim, sizeof(float));    
   for (j=0;j<ydim;j++){
     data[i][j] = 0.0;
   }
 }
 return data;
}


void writeSnapshot(const std::string& target_dir,float** y,float k,float g,float s,float ss2,float is_x,float is_y,long long int nray,long long int exclude,int res){
  //Write output
  float raysmag1 = is_x*is_y*(float)(res*res)/(ss2*ss2*(float)nray);
  float avgmag = raysmag1*(float)(nray-exclude)/(float)(res*res);
  float avgrayspp = (float)(nray-exclude)/(float)(res*res);
  //  writePlain(target_dir+"tmp.out", y, res, k, g, ss2, avgmag, avgrayspp);
  //  rename((target_dir+"tmp.out").c_str(),(target_dir+"map.dat").c_str());
  writeBin(target_dir, y, res, k, g, s, ss2, avgmag, avgrayspp);
}
void reportProgress(long cycles,long long int nray,long long int exclude,int res,Stopwatch& totalTimer){
  //Report progress
  float avgrayspp = (float)(nray-exclude)/(float)(res*res);
  int pc          = (int)floor(100*(nray/CHUNK_SIZE)/cycles);
  float ex_pc     = 0;
  if( nray != 0 ){
    ex_pc = (float)100.0*exclude/nray;
  }
  printf("%3i%% complete, total rays: %12lli, excluded: %12lli (%7.4f%%), avg rays/pxl: %14.8f, ttot: %8.1f\n",pc,nray,exclude,ex_pc,avgrayspp,totalTimer.getTime());
  fflush(stdout);
}


void writeBin(const std::string& target_dir,float** y,int dim,float k,float g,float s,float ss2,float avgmag,float avgrayspp){
  ofstream out_dat( (target_dir+"mapmeta.dat").c_str() );
  out_dat << avgmag << " " << avgrayspp << endl;
  out_dat << dim    << endl;
  out_dat << ss2*2  << endl;
  out_dat << k      << " " << g  << " " << s << endl;
  out_dat.close();

  int dum;
  ofstream out_bin( (target_dir+"map.bin").c_str() ,std::ios::out|std::ios::binary);
  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      dum = int(y[i][j]);
      out_bin.write((const char*) (&dum),sizeof(int));
    }
  }
  out_bin.close();
}

void readBin(const std::string& target_dir,float** y,int Nx,int Ny){
  FILE* ptr_myfile = fopen( (target_dir+"map.bin").c_str() ,"rb");
  int* imap = (int*) calloc(Nx*Ny,sizeof(int));
  fread(imap,sizeof(int),Nx*Ny,ptr_myfile);
  fclose(ptr_myfile);

  for(long i=0;i<Nx;i++){
    for(long j=0;j<Ny;j++){
      y[i][j] = (float) imap[i*Ny+j];
    }
  }

  free(imap);
}


void writePlain(const std::string& name, float** y, int dim, float k, float g, float ss2, float avgmag, float avgrayspp){
  ofstream out(name.c_str());

  out << avgmag << " " << avgrayspp << endl;
  out << dim    << " " << dim       << endl;
  out << ss2*2                      << endl;
  out << k      << " " << g         << endl;
  for( int i=0; i<dim; ++i ){
    for( int j=0; j<dim; ++j ){
      out << y[i][j] << endl;
    }
  }

  out.close();
}

