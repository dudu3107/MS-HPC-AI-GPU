#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>
#include "utils/monitoring/CudaTimer.h"


#include "lbm/LBMSolver.h" 

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  CudaTimer gpu_timer;
  gpu_timer.start();

  LBMSolver* solver = new LBMSolver(params);

  solver->run();

  gpu_timer.stop();
  float gpu_time = gpu_timer.elapsed();
  //printf("time: %f\n", gpu_time);
  
  std::ofstream outfile;
  outfile.open("BenchmarkGPU.log", std::ios_base::app);
  outfile << "GPU time: " << gpu_time << "  nx: " << params.nx << "  ny: " << params.ny << std::endl;

  delete solver;

  return EXIT_SUCCESS;
}
