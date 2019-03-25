#include <string>
#include <vector>
#include <cstdlib>

typedef struct{
  std::string mode;
  std::string target_dir;
  std::string lens_pos;
  std::vector<int> devices;
  float kappa_star;
  float kappa_s;
  float gamma;
  float s;
  int   resolution;
  float source_scale;
  float avg_ray_count;
  float lens_scale_fudge_1;
  float lens_scale_fudge_2;
  float image_scale_fudge;
  float mass;
  long long int total_rays;
  long  cycles;
  long  cycles_done;
  long  nray;
  long  exclude;
  float ttot;
} params;

void printUsage(char* argv[]);
void getLastLine(const std::string& target_dir,int& nrays,int& exclude,float& ttot);
int  setParams(int argc,char* argv[],params &parameters);
