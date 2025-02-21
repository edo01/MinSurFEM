#include "dealII/MinSurDealII.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

int
main(int argc, char *argv[])
{
  const std::string default_mesh_file_name = "../mesh/mesh-square-h0.012500.msh";
  const std::string mesh_file_name = (argc > 1) ? argv[1] : default_mesh_file_name;

  MinSurDealII problem(mesh_file_name);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();
}


void
MinSurDealII::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh from " << mesh_file_name << std::endl;

    // read the mesh from file:
    // At present, UCD (unstructured cell data), DB Mesh, XDA, Gmsh, Tecplot, 
    // UNV, VTK, ASSIMP, and Cubit are supported as input format for grid data
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);

    std::ifstream mesh_file(mesh_file_name);
    grid_in.read_msh(mesh_file);

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    // Construct the finite element object. Notice that we use the FE_SimplexP
    // class here, that is suitable for triangular (or tetrahedral) meshes.
    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    // Construct the quadrature formula of the appopriate degree of exactness.
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // Initialize and then number the degrees of freedom. For a given finite 
    // element space, initializes info on the control variables: how many they are,
    // where they are collocated, their "global indices", ...).
    // The numbering of degrees of freedom (DoFs) is treated as an "implementation
    // detail" from deal.II, so we could use other methods offered by DoFRenumbering,
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    // We first initialize a "sparsity pattern", i.e. a data structure that
    // indicates which entries of the matrix are zero and which are different
    // from zero. To do so, we construct first a DynamicSparsityPattern (a
    // sparsity pattern stored in a memory- and access-inefficient way, but
    // fast to write) and then convert it to a SparsityPattern (which is more
    // efficient, but cannot be modified), which means CSR format.
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix 
    std::cout << "  Initializing the system matrix" << std::endl;
    stiffness_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the delta vector" << std::endl;
    delta.reinit(dof_handler.n_dofs());
  }
}

void
MinSurDealII::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  //
  // IT IS IMPORTANT that only the flags that are really needed must be set,
  // in order to avoid unnecessary computations.
  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    // Here we specify what quantities we need FEValues to compute on
    // quadrature points. For our test, we need:
    // - the values of shape functions (update_values);
    // - the derivative of shape functions (update_gradients);
    // - the position of quadrature points (update_quadrature_points);
    // - the product J_c(x_q)*w_q (update_JxW_values).
    update_gradients | update_quadrature_points | update_JxW_values);

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  stiffness_matrix   = 0.0;
  residual_vector   = 0.0;

  // We will use this vector to store the values of the gradient of the
  // solution at the quadrature points.
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Reinitialize the FEValues object on current element. This
      // precomputes all the quantities we requested when constructing
      // FEValues (see the update_* flags above) for all quadrature nodes of
      // the current cell.
      fe_values.reinit(cell);

      // We reset the cell matrix and vector (discarding any leftovers from
      // previous element).
      cell_matrix   = 0.0;
      cell_residual = 0.0;

      // get the values of the gradient of the solution at the quadrature points
      fe_values.get_function_gradients(solution, solution_gradient_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we assemble the local contribution for current cell and
          // current quadrature point, filling the local matrix and vector.

          // compute the square norm of the gradient of the solution at the
          // current quadrature point
          const double grad_norm_square = scalar_product(solution_gradient_loc[q],
                                                      solution_gradient_loc[q]);
          const double grad_norm = sqrt(grad_norm_square);

          const double sqrt_1_plus_grad = sqrt(1 + grad_norm_square);

          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                // FEValues::shape_grad(i, q) returns the gradient of the i-th
                // basis function at the q-th quadrature node, already mapped
                // on the physical element: we don't have to deal with the
                // mapping, it's all hidden inside FEValues.
                
                // (grad(v) * grad(delta))/sqrt(1 + |grad(u)|^2)
                cell_matrix(i, j) += ((fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q))/sqrt_1_plus_grad) *
                                      fe_values.JxW(q);

                // (grad(v) \dot grad(u))*(grad(u) \dot grad(delta)) / (|grad(u)|*sqrt((1 + |grad(u)|^2)^3))
                cell_matrix(i, j) += ((scalar_product(fe_values.shape_grad(i, q), solution_gradient_loc[q]) *
                                       scalar_product(solution_gradient_loc[q], fe_values.shape_grad(j, q)))/
                                      (grad_norm * (1 + grad_norm_square)* sqrt_1_plus_grad))
                                      *fe_values.JxW(q);
              }

              // -grad(u) \dot grad(v) / sqrt(1 + |grad(u)|^2)
              cell_residual(i) -= (scalar_product(solution_gradient_loc[q], fe_values.shape_grad(i, q))/
                                  sqrt_1_plus_grad) * fe_values.JxW(q);
            }
        }


      // At this point the local matrix and vector are constructed: we
      // need to sum them into the global matrix and vector. To this end,
      // we need to retrieve the global indices of the DoFs of current
      // cell.
      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      stiffness_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);
    }

  
  // We apply homogeneous Dirichlet boundary conditions.
  // The linear system solution is delta, which is the difference between
  // u_{n+1}^{(k+1)} and u_{n+1}^{(k)}. Both must satisfy the same Dirichlet
  // boundary conditions: therefore, on the boundary, delta = u_{n+1}^{(k+1)} -
  // u_{n+1}^{(k+1)} = 0. We impose homogeneous Dirichlet BCs.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.

    // Functions::ZeroFunction<dim> zero_function(dim);
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim>                        zero_function;

    for (unsigned int i = 0; i < 4; ++i)
      boundary_functions[i] = &zero_function;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, stiffness_matrix, delta, residual_vector, true);
  }
}

void
MinSurDealII::solve_linear_system()
{
  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "\nSolving the linear system";
  // We are not using any preconditioner, so we pass the identity matrix.
  solver.solve(stiffness_matrix, delta, residual_vector, PreconditionIdentity());
  std::cout << " - " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
MinSurDealII::solve()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-6;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  // We apply the Dirichlet boundary conditions to the linear system before
  // starting the Newton iteration. Since the delta vector is the difference
  // between two solutions, in the Newton iteration we will apply homogeneous
  // Dirichlet boundary conditions.
  {
    // We extract the DoFs corresponding to the Dirichlet boundary conditions.
    //IndexSet dirichlet_dofs = DoFTools::extract_boundary_dofs(dof_handler);
    // We create a vector that stores the values of the Dirichlet boundary
    //Vector vector_dirichlet(solution);
    // We interpolate the Dirichlet boundary function g on the entire domain.
    VectorTools::interpolate(dof_handler, function_g, solution);//vector_dirichlet);

    // differently from the assemble function, we apply the boundary conditions
    // only to the solution vector, not to the system matrix and right-hand side.
    //for (const auto &idx : dirichlet_dofs) // we iterate over the DoFs on the boundary
    //  solution[idx] = vector_dirichlet[idx];

    // A GOOD STARTING POINT FOR THE NEWTON ITERATION MAY BE THE FUNCTION G
    // ITSELF, SO WE SET THE SOLUTION VECTOR TO THE INTERPOLATION OF G.
    // IN FACT, AS WE KNOW, WE HAVE CONVERGENCE RESULTS FOR THE NEWTON METHOD
    // ONLY IF THE INITIAL GUESS IS CLOSE ENOUGH TO THE SOLUTION.
    // THIS IS A TRIVIAL WAY TO SET THE INITIAL GUESS. IN PRACTICE, WE MAY
    // USE A MORE SOPHISTICATED STRATEGIES.
  }

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble();
      residual_norm = residual_vector.l2_norm();

      std::cout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      if (residual_norm > residual_tolerance)
        {
          solve_linear_system();
          solution += delta; // update the solution
        }
      else
        {
          std::cout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

void
MinSurDealII::output() const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add the function g to the output.
  Vector<double> g_values(solution);
  VectorTools::interpolate(dof_handler, function_g, g_values);
  data_out.add_data_vector(dof_handler, g_values, "function_g");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string           output_file_name = "output-MinSurDealII.vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}