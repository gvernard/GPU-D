#define DEFAULT_MODE               "normal"
#define DEFAULT_TARGET_DIR         ""
#define DEFAULT_LENS_POS           "123"
#define DEFAULT_GAMMA              0.000
#define DEFAULT_S                  0.000
#define DEFAULT_KAPPA              0.413
#define DEFAULT_AVG_RAY_COUNT      100
#define DEFAULT_SOURCE_SCALE       24.0
#define DEFAULT_RESOLUTION         512
#define DEFAULT_MASS               1.0
#define DEFAULT_IMAGE_SCALE_FUDGE  0.1
#define DEFAULT_LENS_SCALE_FUDGE_1 5.0
#define DEFAULT_LENS_SCALE_FUDGE_2 2.0

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "string.hpp"
#include "help.hpp"
#include "cudaLens.hpp" //using definition of CHUNK_SIZE

using std::cout;
using std::endl;
using mycuda::fromString;


void printUsage(char* argv[]){
  cout << "Usage: " << argv[0] << " [options]" << endl;
  cout << "  -mode,     --mode                        Set the mode to 'normal','resubmit' (same) or 'restart'" << endl;
  cout << "  -td,       --target_dir                  Set the I/O directory to an existing directory" << endl;
  cout << "  -lens_pos, --lens_pos sn                 Set the lens positions to seed:snnn,random:r,or read from file 'lens_pos.dat':c" << endl;
  cout << "  -g,        --gamma f                     Set the global shear to f" << endl;
  cout << "  -ss,       --source-scale f              Set the physical source scale to f" << endl;
  cout << "  -m,        --lens-mass f                 Set the lens mass to f" << endl;
  cout << "  -lf1,      --lens-scale-fudge-1 f        Set lens scale fudge factor 1 to f" << endl;
  cout << "  -lf2,      --lens-scale-fudge-2 f        Set lens scale fudge factor 2 to f" << endl;
  cout << "  -if,       --image-scale-fudge f         Set image scale fudge factor to f" << endl;
  cout << "  -s,        --smooth matter fraction f    Set the convergence in continuous matter to f" << endl;
  cout << "  -k,        --kappa f                     Set the convergence in compact objects to f" << endl;
  cout << "  -navg,     --avg-ray-count n             Set average rays per pixel to n" << endl;
  cout << "  -res,      --resolution n                Set resolution to n*n" << endl;
  cout << "  -d,        --use-devices n1 n2...        Use the GPU devices with indices n1, n2..." << endl;
  cout << "  -h,        --help                        Show this usage information" << endl;
}


void getLastLine(const std::string& target_dir,long int* nray,long int* exclude,double* ttot){
  std::string line;
  std::ifstream infile( (target_dir+"meta0.dat").c_str() );
  
  while( infile >> std::ws && std::getline(infile,line) ){
    //do nothing, just get to the last non-empty line of the file
  }
  //  cout << line << endl;
  char dum[50];
  sscanf(line.c_str(),"%27c%12li%12c%12li%49c%lf\n",dum,nray,dum,exclude,dum,ttot);
}


int setParams(int argc,char* argv[],params &parameters){
  std::string mode          = DEFAULT_MODE;
  std::string target_dir    = DEFAULT_TARGET_DIR;
  std::string lens_pos      = DEFAULT_LENS_POS;
  double  kappa              = DEFAULT_KAPPA;
  double  s                  = DEFAULT_S;
  double  gamma              = DEFAULT_GAMMA;
  int    resolution         = DEFAULT_RESOLUTION;
  double  source_scale       = DEFAULT_SOURCE_SCALE;
  double  avg_ray_count      = DEFAULT_AVG_RAY_COUNT;
  double  lens_scale_fudge_1 = DEFAULT_LENS_SCALE_FUDGE_1;
  double  lens_scale_fudge_2 = DEFAULT_LENS_SCALE_FUDGE_2;
  double  image_scale_fudge  = DEFAULT_IMAGE_SCALE_FUDGE;
  double  mass               = DEFAULT_MASS;
  std::vector<int> devices;
  std::vector<int> device_idx_list;

  // Print usage information when no params are passed
  if( 1 == argc ) {printUsage(argv);return 1;}
  
  // 'Simple' command-line parsing
  for( int i=1; i<argc; i++ ) {
    if( strcmp(argv[i], "--mode")==0 || strcmp(argv[i], "-mode")==0 ) {
      fromString(argv[++i],mode);
      if ( !( mode == "normal" || mode == "restart" ) ){
	cout << "Error: wrong mode of operation" << endl;
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "--target_dir")==0 || strcmp(argv[i], "-td")==0 ) {
      fromString(argv[++i],target_dir);
      char *temp = (char*) target_dir.c_str();
      if ( opendir(temp) == NULL ){
	cout << "Error: target directory does not exist" << endl;
	printUsage(argv);
	return 1;
      }
      if ( target_dir.substr(target_dir.length()-1,1).compare("/")!=0 ) {
	target_dir += "/";
      }
    } else if( strcmp(argv[i], "--lens_pos")==0 || strcmp(argv[i], "-lens_pos")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], lens_pos) ) {
	  cout << "Error parsing param 'lens_pos'" << endl;
	  printUsage(argv);
	  return 1;
	}
	bool blp = true;
	if ( lens_pos.compare("r")==0 )  { blp = false; }
	if ( atoi(lens_pos.c_str())!=0 ) { 
	  blp = false;
	  std::stringstream ss;
	  ss << atoi(lens_pos.c_str());
	  lens_pos = ss.str();
	}
	if ( lens_pos.compare("c")==0 )  { blp = false; }
	if ( lens_pos.length()>4 && lens_pos.substr(lens_pos.length()-4,4).compare(".txt")==0 ) {
	  blp = false;
	  lens_pos = "c";
	}
	if ( blp ) {
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "-g") == 0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], gamma) ) {
	  cout << "Error parsing param 'gamma'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else {
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "-k") == 0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], kappa) ) {
	  cout << "Error parsing param 'kappa'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "-s") == 0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], s) ) {
	  cout << "Error parsing param 's'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "--lens-scale-fudge-1")==0 || strcmp(argv[i], "-lf1")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], lens_scale_fudge_1) ) {
	  cout << "Error parsing param 'lens-scale-fudge-1'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv); 
	return 1; 
      }
    } else if( strcmp(argv[i], "--lens-scale-fudge-2")==0 || strcmp(argv[i], "-lf2")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], lens_scale_fudge_2) ) {
	  cout << "Error parsing param 'lens-scale-fudge-2'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv);
	return 1;
      }
    } else if( strcmp(argv[i], "--image-scale-fudge")==0 || strcmp(argv[i], "-if")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], image_scale_fudge) ) {
	  cout << "Error parsing param 'image-scale-fudge'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv); 
	return 1;
      }
    } else if( strcmp(argv[i], "--avg-ray-count")==0 || strcmp(argv[i], "-navg")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], avg_ray_count) ) {
	  cout << "Error parsing param 'avg-ray-count'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv); 
	return 1; 
      }
    } else if( strcmp(argv[i], "--resolution")==0 || strcmp(argv[i], "-res")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], resolution) ) {
	  cout << "Error parsing param 'resolution'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv); 
	return 1;
      }
    } else if( strcmp(argv[i], "--source-scale")==0 || strcmp(argv[i], "-ss")==0 ) {
      if( i+1 < argc ) {
	if( !fromString(argv[++i], source_scale) ) {
	  cout << "Error parsing param 'source-scale'" << endl;
	  printUsage(argv);
	  return 1;
	}
      } else { 
	printUsage(argv); 
	return 1;
      }
    } else if( strcmp(argv[i], "--use-devices")==0 || strcmp(argv[i], "-d")==0 ) {
      while( i+1 < argc ) {
	int idx;
	if( !fromString(argv[++i], idx) ) {
	  //You just read the next option, go back one space
  	  --i;
	  break;
	}
	device_idx_list.push_back(idx);
      }
    } else if( strcmp(argv[i], "--help")==0 || strcmp(argv[i], "-h")==0 ) {
      printUsage(argv);
      return 1;
    } else {
      cout << "Unknown option '" << argv[i] << "'" << endl;
      printUsage(argv);
      return 1;
    }
  }



  if( device_idx_list.size() > 0 ) { // Use given list of devices
    for(unsigned int d=0;d<device_idx_list.size();++d){
      devices.push_back(device_idx_list[d]);
    }
  } else { // Use first device
    devices.push_back(0);
  }



  double kappa_s    = kappa*s;
  double kappa_star = kappa - kappa_s;

  parameters.mode               = mode;
  parameters.target_dir         = target_dir;
  parameters.kappa_star         = kappa_star;
  parameters.kappa_s            = kappa_s;
  parameters.s                  = s;
  parameters.gamma              = gamma;
  parameters.resolution         = resolution;
  parameters.source_scale       = source_scale;
  parameters.lens_pos           = lens_pos;
  parameters.avg_ray_count      = avg_ray_count;
  parameters.lens_scale_fudge_1 = lens_scale_fudge_1;
  parameters.lens_scale_fudge_2 = lens_scale_fudge_2;
  parameters.image_scale_fudge  = image_scale_fudge;
  parameters.mass               = mass;
  parameters.devices            = devices;

  //  int ndevices = devices.size();
  parameters.total_rays  = avg_ray_count*avg_ray_count*1.e4;
  //cycles = (long)ceilf((double)((long)avg_ray_count*(long)resolution*(long)resolution)/(CHUNK_SIZE*ndevices));
  //parameters.cycles      = (long)ceilf((double)(avg_ray_count*avg_ray_count*1.e4)/(CHUNK_SIZE*ndevices));
  parameters.cycles      = (long)ceilf((double)(avg_ray_count*avg_ray_count*1.e4)/(CHUNK_SIZE));
  parameters.cycles_done = 0;
  parameters.nray        = 0;
  parameters.exclude     = 0;
  parameters.ttot        = 0;

  if( mode == "restart" ){
    parameters.lens_pos = 'c';
    getLastLine(parameters.target_dir,&parameters.nray,&parameters.exclude,&parameters.ttot);
    parameters.cycles_done = (long) ceilf( (double) (parameters.nray/CHUNK_SIZE) );
  }



  return 0;
}


