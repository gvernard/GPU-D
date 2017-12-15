#include <string>
#include <vector>
#include <cstdlib>

typedef struct{
  std::string mode;
  std::string target_dir;
  std::string lens_pos;
  std::vector<int> devices;
  double kappa_star;
  double kappa_s;
  double gamma;
  double s;
  int   resolution;
  double source_scale;
  double avg_ray_count;
  double lens_scale_fudge_1;
  double lens_scale_fudge_2;
  double image_scale_fudge;
  double mass;
  long long int total_rays;
  long  cycles;
  long  cycles_done;
  long  nray;
  long  exclude;
  double ttot;
} params;

void printUsage(char* argv[]);
void getLastLine(const std::string& target_dir,int& nrays,int& exclude,double& ttot);
int  setParams(int argc,char* argv[],params &parameters);
