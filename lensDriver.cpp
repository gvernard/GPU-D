#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "device.hpp"
#include "stopwatch.hpp"
#include "string.hpp"
#include "help.hpp"
#include "lensFuncs.hpp"
#include "cudaLens.hpp"


using std::cout;
using std::endl;
using mycuda::toString;
using mycuda::Device;




int main(int argc, char* argv[]){
  Stopwatch totalTimer;
  Stopwatch initTimer;
  Stopwatch genposTimer;
  Stopwatch transferTimer;
  Stopwatch kernelTimer1;
  Stopwatch kernelTimer2;
  Stopwatch binTimer;
  Stopwatch writeTimer;

  totalTimer.start();

  initTimer.start();
  //======================================== Verify Input and report errors ========================================
  params parameters;
  if ( setParams(argc,argv,parameters) ){//if successfull, setParams has already set the parameters.
    return -1;
  }
  //===================================================== END ======================================================

  
  //======================================== Parameter Initialization ========================================
  std::string mode             = parameters.mode;
  std::string target_dir       = parameters.target_dir;
  std::string lens_pos         = parameters.lens_pos;
  std::vector<int> device_list = parameters.devices;
  double kappa_star            = parameters.kappa_star;
  double kappa_s               = parameters.kappa_s;
  double s                     = parameters.s;
  double gamma                 = parameters.gamma;
  int    resolution            = parameters.resolution;
  double source_scale          = parameters.source_scale;
  double avg_ray_count         = parameters.avg_ray_count;
  double lens_scale_fudge_1    = parameters.lens_scale_fudge_1;
  double lens_scale_fudge_2    = parameters.lens_scale_fudge_2;
  double image_scale_fudge     = parameters.image_scale_fudge;
  double mass                  = parameters.mass;
  long   cycles                = parameters.cycles;
  long   cycles_done           = parameters.cycles_done;
  //  long long int total_rays     = parameters.total_rays;
  long long int nray           = parameters.nray;
  long long int exclude        = parameters.exclude;
  int    ndevices              = device_list.size();

  // Calculate derived physical quantities
  double  kappa         = kappa_star + kappa_s;
  double  ray_grid_x    = (0.5 + 2.0*image_scale_fudge) * source_scale / (1.0 - kappa - gamma);
  double  ray_grid_y    = (0.5 + 2.0*image_scale_fudge) * source_scale / (1.0 - kappa + gamma);
  double  lens_rad_x    = (0.5*source_scale + lens_scale_fudge_1) / (1.0 - kappa + gamma);
  double  lens_rad_y    = (0.5*source_scale + lens_scale_fudge_1) / (1.0 - kappa - gamma);
  double  lens_rad      = sqrt(lens_rad_x*lens_rad_x + lens_rad_y*lens_rad_y) + lens_scale_fudge_2;
  long    lens_count    = (size_t)(kappa_star * lens_rad*lens_rad / mass + 0.5);
  double  ss2           = source_scale/2.0;
  // TODO: Eventually 'mass' should be the avg. lens mass
  // The rest of the code considers source_scale to be a radius, i.e., half the side length


  // Allocate and initialise lens data (determine own lens positions) on CPU.
  // Lenses x,y are important only to the 7th decimal, and m to the 2nd.
  ImagePlane IP;
  std::string temp = target_dir+"lens_pos.dat";
  char *lp_file    = (char*) temp.c_str();

  if ( lens_pos.compare("r") == 0 ) {
    srand48(timeMachineSeed(device_list));
    IP = setupLensPositions(lp_file,lens_count,lens_rad,lens_rad,mass);
  } else if ( atoi(lens_pos.c_str()) != 0 ) {
    srand48(atoi(lens_pos.c_str()));
    IP = setupLensPositions(lp_file,lens_count,lens_rad,lens_rad,mass);
  } else {
    std::ifstream in_stream(lp_file);
    if (in_stream){
      IP = readLensPositions(lp_file,&lens_count,lens_rad,lens_rad,mass);
    } else {
      cout << "Input file for lens positions does not exist." << endl;
      return -1;
    }
    in_stream.close();
  }


  // Allocate source plane array
  double** source_plane;
  source_plane = create2DArray<double>(resolution+1, resolution+1);
  if( mode == "restart" ){
    readBin(target_dir,source_plane,resolution,resolution);
  }


  cout << "gpu-d code version : 6.0"                     << endl;
  cout << "Output directory : "    << target_dir         << endl;
  cout << "****** Initialization ******"                 << endl;
  cout << " Kappa              = " << kappa              << endl;
  cout << " Kappa compact      = " << kappa_star         << endl;
  cout << " Kappa smooth       = " << kappa_s            << endl;
  cout << " Smooth matter fr.  = " << s                  << endl;
  cout << " Gamma              = " << gamma              << endl;
  cout << " Resolution         = " << resolution         << endl;
  cout << " Source scale       = " << source_scale       << endl;
  cout << " Lens Positions     = " << lens_pos           << endl;
  cout << " Lens count         = " << lens_count         << endl;
  cout << " Lens mass          = " << mass               << endl;
  cout << " Image scale x      = " << ray_grid_x*2.0     << endl;
  cout << " Image scale y      = " << ray_grid_y*2.0     << endl;
  cout << " Lens scale fudge 1 = " << lens_scale_fudge_1 << endl;
  cout << " Lens scale fudge 2 = " << lens_scale_fudge_2 << endl;
  cout << " Image scale fudge  = " << image_scale_fudge  << endl;
  cout << " Lens region rad x  = " << lens_rad_x         << endl;
  cout << " Lens region rad y  = " << lens_rad_y         << endl;
  cout << " Lens region radius = " << lens_rad           << endl;
  cout << " Avg. ray count     = " << avg_ray_count      << endl;
  printf("Doing %4li cycles on %1i devices with CHUNK_SIZE=%7i\n",cycles,ndevices,CHUNK_SIZE);
  //================================================= END =================================================


  //======================================== Device Initialization ========================================
  omp_set_num_threads(ndevices); // IMPORTANT!!!

  std::vector<Device> devices;
  for(int d=0;d<ndevices;++d){
    Device device(device_list[d]);
    cout << "Using device " << toString(device.getIndex()) << " (" << device.getName() << ")";
    device.initialise();
    cout << " ... initialized" << endl;
    devices.push_back(device);
  }
  //================================================= END =================================================
  initTimer.stop();


  //========================================== Memory Allocation ==========================================
  assert(IP.Nl < LENS_CHUNK && "Number of lenses must be less than 1,000,000");
  
  // Host memory pointers for ray positions
  double* x1[ndevices];
  double* x2[ndevices];
  double* y1[ndevices];
  double* y2[ndevices];
  // Device memory pointers for rays, lens positions and mass
  double* d_x1[ndevices];
  double* d_x2[ndevices];
  double* d_p1[ndevices];
  double* d_p2[ndevices];
  double* d_lx1[ndevices];
  double* d_lx2[ndevices];
  double* d_lm[ndevices];

  // Allocate Host memory for ray positions
  for(int d=0;d<ndevices;++d){
    x1[d] = (double*) calloc(CHUNK_SIZE, sizeof(double));
    x2[d] = (double*) calloc(CHUNK_SIZE, sizeof(double));
    y1[d] = (double*) calloc(CHUNK_SIZE, sizeof(double));
    y2[d] = (double*) calloc(CHUNK_SIZE, sizeof(double));
  }	

  // Allocate Device memory for rays, lens positions and mass (usage of Nleff for correct memory allocation)
#pragma omp parallel for
  for(int d=0;d<ndevices;d++){
    cudaMalloc( (void**) &d_x1[d], CHUNK_SIZE*sizeof(double));
    cudaMalloc( (void**) &d_x2[d], CHUNK_SIZE*sizeof(double));
    cudaMalloc( (void**) &d_p1[d], CHUNK_SIZE*sizeof(double));
    cudaMalloc( (void**) &d_p2[d], CHUNK_SIZE*sizeof(double));
    cudaMalloc( (void**) &d_lx1[d], IP.Nleff*sizeof(double));
    cudaMalloc( (void**) &d_lx2[d], IP.Nleff*sizeof(double));
    cudaMalloc( (void**) &d_lm[d],  IP.Nleff*sizeof(double));
      
    cudaError cuda_err = cudaGetLastError();
    if( cudaSuccess != cuda_err ){
      cout << "CUDA Error: Failed to allocate Device memory, device " << devices[d].getIndex() << endl;
    }
  }

  // Transfer lens positions and mass from Host to Device
  transferTimer.start();
#pragma omp parallel for
  for(int d=0;d<ndevices;d++){
    cudaMemcpy(d_lx1[d], IP.lx1, IP.Nl*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_lx2[d], IP.lx2, IP.Nl*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_lm[d],  IP.lm,  IP.Nl*sizeof(double),cudaMemcpyHostToDevice);
  
    cudaError cuda_err = cudaGetLastError();
    if( cudaSuccess != cuda_err ){
      cout << "CUDA Error: Failed to transfer lens position and mass Host memory to Device, device " << devices[d].getIndex() << endl;
    }
  }
  transferTimer.stop();
  //================================================= END =================================================



  //======================================== Main loop ========================================
  cout << "****** Running ******" << endl;

  if( mode == "restart" ){
    totalTimer.setTime(parameters.ttot);
  }

  double intervals;
  if( lens_count > 1300000 ){
    intervals = 40.;
  } else {
    intervals = 10.;
  }
  double last_out = cycles/intervals;
  while ( last_out < (double)cycles_done ){
    last_out += (double)cycles/intervals;
  }
  reportProgress(cycles,nray,exclude,resolution,totalTimer);



  double* yZeros = (double*) calloc(CHUNK_SIZE,sizeof(double));//if this is done inside the loop then I keep allocating memory until I run out of it
  for(long m=cycles_done;m<cycles;m+=ndevices){

    // Generate random starting ray positions
    genposTimer.start();
    /*    
#pragma omp parallel for
    for(int d=0;d<ndevices;d++){
      genRandomPositions(x1[d], x2[d], CHUNK_SIZE, ray_grid_x, ray_grid_y, (long) m+d);
    }
    */    
    genRandomPositions(x1, x2, ndevices, CHUNK_SIZE, ray_grid_x, ray_grid_y, (long) m);
    genposTimer.stop();
    
    
    // Transfer starting ray positions from Host to Device
    transferTimer.start();
#pragma omp parallel for
    for(int d=0;d<ndevices;d++){
      cudaMemcpy(d_x1[d], x1[d], CHUNK_SIZE*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_x2[d], x2[d], CHUNK_SIZE*sizeof(double), cudaMemcpyHostToDevice);
      //Overwrite the partial ys with zeros <- NECESSARY
      cudaMemcpy(d_p1[d], yZeros, CHUNK_SIZE*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_p2[d], yZeros, CHUNK_SIZE*sizeof(double), cudaMemcpyHostToDevice);
      
      cudaError cuda_err = cudaGetLastError();
      if( cudaSuccess != cuda_err ){
	cout << "CUDA Error: Failed to transfer ray position Host memory to Device, device " << devices[d].getIndex() << endl;
      }
    }
    transferTimer.stop();
    

    // Calculate deflections. Main calculation.
#pragma omp parallel for
    for(int d=0;d<ndevices;d++){
      runCudaLens(d_lx1[d], d_lx2[d], d_lm[d], d_x1[d], d_x2[d], d_p1[d], d_p2[d], IP.Nl, kappa_s, gamma, kernelTimer1, kernelTimer2);
    }


    // Transfer final ray positions from Device to Host
#pragma omp parallel for
    for(int d=0;d<ndevices;d++){
      cudaMemcpy(y1[d], d_x1[d], CHUNK_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(y2[d], d_x2[d], CHUNK_SIZE*sizeof(double), cudaMemcpyDeviceToHost);

      cudaError cuda_err = cudaGetLastError();
      if( cudaSuccess != cuda_err ){
	cout << "CUDA Error: Failed to transfer final ray position Device memory to Host, device " << devices[d].getIndex() << endl;
      } 
    }


    // Bin the results in pixels
    binTimer.start();
    /*
    for(int d=0;d<ndevices;d++){
      binSourcePlane(&nray, &exclude, source_plane, y1[d], y2[d], resolution, ss2);
    }
    */
    binSourcePlane(&nray, &exclude, ndevices, source_plane, y1, y2, resolution, ss2, m, cycles);
    binTimer.stop();
    
	
    // Print current status every intervals %
    if( m >= last_out ) {

      last_out += (double)cycles/intervals;
      writeTimer.start();
      writeSnapshot(target_dir,source_plane,kappa,gamma,s,ss2,ray_grid_x,ray_grid_y,nray,exclude,resolution);
      writeTimer.stop();
      reportProgress(cycles,nray,exclude,resolution,totalTimer);
    }


  }


  // Print out the final ray statistics
  writeTimer.start();
  writeSnapshot(target_dir,source_plane,kappa,gamma,s,ss2,ray_grid_x,ray_grid_y,nray,exclude,resolution);
  writeTimer.stop();
  reportProgress(cycles,nray,exclude,resolution,totalTimer);
  //================================================= END =======================================================


  //=========================================== Cleanup GPU =====================================================
  // Loop over devices
  initTimer.start();
#pragma omp parallel for
  for(int d=0;d<ndevices;d++){
    cudaFree(d_x1[d]);
    cudaFree(d_x2[d]);
    cudaFree(d_p1[d]);
    cudaFree(d_p2[d]);
  }
  initTimer.stop();
  //=============================================== END =========================================================


  totalTimer.stop();


  cout << "****** Summary ******" << endl;
  cout << "Profiling times (not all accurate for multi-GPU runs):"   << endl;
  cout << " Init/cleanup time        : " << initTimer.getTime()     << endl;
  cout << " Gen random positions time: " << genposTimer.getTime()   << endl;
  cout << " Memory transfer time     : " << transferTimer.getTime() << endl;
  cout << " Kernel part 1 time       : " << kernelTimer1.getTime()  << endl;
  cout << " Kernel part 2 time       : " << kernelTimer2.getTime()  << endl;
  cout << " Bin time                 : " << binTimer.getTime()      << endl;
  cout << " Write output time        : " << writeTimer.getTime()    << endl;
  cout << " Total run time           : " << totalTimer.getTime()    << endl;	
  cout << "Done." << endl;


  return 0;
}
    
