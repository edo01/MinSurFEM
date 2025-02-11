#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

// here we should include only the interface 
#include "dealII/MinSur.hpp"

int
main(int /*argc*/, char * /*argv*/[])
{
  const std::string mesh_file_name =
    "../../mesh/mesh-square-h0.100000.msh";
  const unsigned int r = 1;

  MinSur problem(mesh_file_name, r);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}