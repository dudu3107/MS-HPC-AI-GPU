#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#include "lbm/LBMSolver.h" 
#include "utils/monitoring/SimpleTimer.h"

// TODO : uncomment when building with OpenACC
//#include "utils/openacc_utils.h"

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  // TODO : uncomment the last two lines when activating OpenACC
  // print OpenACC version / info
  // print_openacc_version();
  //init_openacc();

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  SimpleTimer simple_timer;

  LBMSolver* solver = new LBMSolver(params);

  solver->run();

  simple_timer.stop();
  float sequencial_time = simple_timer.elapsed();
  printf("time: %f\n", sequencial_time);

  std::ofstream outfile;
  outfile.open("BenchmarkSimple.log", std::ios_base::app);
  outfile << "Simple time: " << sequencial_time << "  nx: " << params.nx << "  ny: " << params.ny << std::endl;


  delete solver;

  return EXIT_SUCCESS;
}
