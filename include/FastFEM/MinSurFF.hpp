#ifndef MINSURFF_HPP
#define MINSURFF_HPP

#include <iomanip>
#include <functional>

#include "FastFem/linalg/Vector.hpp"
#include "FastFem/linalg/sparseMatrices/CSRMatrix.hpp"
#include "FastFem/mesh/Mesh.hpp"
#include "FastFem/fe/FESimplexP.hpp"

using namespace fastfem;

/**
 * Class managing the differential problem.
 */
class MinSurFF
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Constructor.
  MinSurFF(unsigned n_vertices, std::function<double(double, double)> g, double newton_step = 1.0)
    : n_vertices(n_vertices), function_g(g), newton_step(newton_step)
  {
    if(newton_step <= 0)
      throw std::runtime_error("MinSurFF(): newton_step must be positive.");
  }

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output();

protected:

  void solve_linear_system();

  /* Polynomial degree. */
  const unsigned int r = 1;

  unsigned int n_vertices;

  /* g(x). */
  // Dirichlet boundary conditions.
  std::function<double(double, double)> function_g;

  /* Triangulation */
  mesh::Mesh<dim> mesh;

  /* Finite element space. */
  std::shared_ptr<fe::FESimplexP<dim>> fe;

  dof::DoFHandler<dim> dof_handler;

  linalg::CSRPattern sparsity_pattern;

  /* System matrix. */
  linalg::CSRMatrix stiffness_matrix;

  // System right-hand side.
  linalg::Vector residual_vector;

  // System solution.
  linalg::Vector solution;

  // System solution
  linalg::Vector delta;

  double newton_step;
};

#endif // MINSURFF_HPP