#ifndef MINSUR_HPP
#define MINSUR_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/tria.h>


#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class MinSur
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  /*
    In deal.ii, functions are implemented by deriving the dealii::Function
    class, which provides an interface for the computation of function values
    and their derivatives.
  */

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };

  // Constructor.
  MinSur(const std::string &mesh_file_name_)
    : mesh_file_name(mesh_file_name_)
  {}

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
  output() const;

protected:

  void solve_linear_system();

  /* Path to the mesh file. */
  const std::string mesh_file_name;

  /* Polynomial degree. */
  const unsigned int r = 1;

  /* g(x). */
  FunctionG function_g;

  /* Triangulation */
  // A triangulation is a collection of cells that, jointly, cover the domain
  // on which one typically wants to solve a partial differential equation. 
  // This domain, and the mesh that covers it, represents a dim-dimensional 
  // manifold and lives in spacedim spatial dimensions.
  // In this case, we are working with a 2D domain embedded in 2D space.
  Triangulation<dim> mesh;

  /* Finite element space. */
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter). The
  // class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit.
  std::unique_ptr<FiniteElement<dim>> fe;

  /* Quadrature formula. */
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  /* DoF handler. */
  // Given a triangulation and a description of a finite element, this 
  // class enumerates degrees of freedom on all vertices, edges, faces, and
  // cells of the triangulation.  As a result, it also provides a basis for
  // a discrete space Vh whose elements are finite element functions defined 
  // on each cell by a FiniteElement object (in our case FE_SimplexP)

  // The numbering of degrees of freedom (DoFs) is done by traversing the
  // triangulation cell by cell and numbers the dofs of that cell if not
  // yet numbered. This numbering implies very large bandwidths of the 
  // resulting matrices and is thus vastly suboptimal for some 
  // solution algorithms.
  // For this reason, the DoFRenumbering class offers several algorithms
  // to reorder the dof numbering according.
  DoFHandler<dim> dof_handler;

  /* Sparsity pattern. */
  // SparseMatrix, only stores a value for each matrix entry, but not
  // where these entries are located. For this, it relies on the information
  // it gets from a SparsityPattern object associated with this matrix.

  // SparsityPattern objects are built in two phases: first, in a "dynamic" phase, 
  // one allocates positions where one expects matrices built on it to have
  // nonzero entries; in a second "static" phase, the representation of these nonzero
  // locations is "compressed" into the usual Compressed Sparse Row (CSR) format
  SparsityPattern sparsity_pattern;

  /* System matrix. */
  SparseMatrix<double> jacobian_matrix;

  // System right-hand side.
  Vector<double> residual_vector;

  // System solution.
  Vector<double> solution;

  // System solution
  Vector<double> delta;
};

#endif