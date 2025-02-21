#include <fstream>
#include <iostream>
#include <vector>

// here we should include only the interface 
#include "dealII/MinSur.hpp"

int
main(int argc, char *argv[])
{
  const std::string default_mesh_file_name = "../mesh/mesh-square-h0.025000.msh";

  const std::string mesh_file_name = (argc > 1) ? argv[1] : default_mesh_file_name;

  MinSur problem(mesh_file_name);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

//   MinSurFF problem_ff(10);

  return 0;
}
