#include "FastFEM/MinSurFF.hpp"
#include "FastFem/mesh/MeshMaker.hpp"
#include "FastFem/linalg/iterativeSolvers/CGSolver.hpp"
#include "FastFem/mesh/MeshIO.hpp"

double distance(const mesh::Point<2> &v1, const mesh::Point<2> &v2)
{
    return std::sqrt((v2.coords[0] - v1.coords[0]) * (v2.coords[0] - v1.coords[0]) + (v2.coords[1] - v1.coords[1]) * (v2.coords[1] - v1.coords[1]));
}

double dot(const mesh::Point<2> &v1, const mesh::Point<2> &v2, const mesh::Point<2> &w1, const mesh::Point<2> &w2)
{
    return (v2.coords[0] - v1.coords[0]) * (w2.coords[0] - w1.coords[0]) + (v2.coords[1] - v1.coords[1]) * (w2.coords[1] - w1.coords[1]);
}

double dot(double x1, double y1, double x2, double y2)
{
    return x1 * x2 + y1 * y2;
}

void
MinSurFF::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the square mesh with " << n_vertices << " x " << n_vertices
              << " vertices" << std::endl;


    mesh::SquareMaker square_maker(n_vertices);

    mesh = square_maker.make_mesh();

    std::cout << "  Number of elements = " << mesh.elem_count() << std::endl;

    std::cout << "-----------------------------------------------" << std::endl;
  }

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    // Construct the finite element object. Notice that we use the FE_SimplexP
    // class here, that is suitable for triangular (or tetrahedral) meshes.
    fe = std::make_shared<fe::FESimplexP1<dim>>();

    std::cout << "  Degree                     = " << fe->get_degree() << std::endl;
    std::cout << "  DoFs per cell              = " << fe->get_n_dofs_per_element() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.attach_mesh(mesh);

    dof_handler.distribute_dofs(fe);

    std::cout << "  Number of DoFs = " << dof_handler.get_n_dofs() << std::endl;

    std::cout << "-----------------------------------------------" << std::endl;
   }

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;
    sparsity_pattern = linalg::CSRPattern::create_from_dof_handler(dof_handler);


    // Then, we use the sparsity pattern to initialize the system matrix 
    std::cout << "  Initializing the system matrix" << std::endl;
    stiffness_matrix = linalg::CSRMatrix(dof_handler.get_n_dofs(), sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    residual_vector = linalg::Vector(dof_handler.get_n_dofs());

    std::cout << "  Initializing the solution vector" << std::endl;
    solution = linalg::Vector(dof_handler.get_n_dofs());

    std::cout << "  Initializing the delta vector" << std::endl;
    delta = linalg::Vector(dof_handler.get_n_dofs());
  }
}

void
MinSurFF::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_element = fe->get_n_dofs_per_element();

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  linalg::FullMatrix local_matrix(dofs_per_element, dofs_per_element);
  linalg::Vector     local_rhs(dofs_per_element);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_element);

  // Reset the global matrix and vector, just in case.
  //stiffness_matrix   = 0.0;
  //residual_vector   = 0.0;
  stiffness_matrix = 0.0;
  residual_vector.fill(0.0);

  // We will use this vector to store the values of the gradient of the
  // solution at the quadrature points.
  //std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  for (auto it = dof_handler.elem_begin(); it != dof_handler.elem_end(); ++it)
  {
    auto &elem = *it;

    mesh::Simplex<2, 2> triangle = mesh.get_Simplex(elem);

    mesh::Point<2> vA = triangle.get_vertex(0);
    mesh::Point<2> vB = triangle.get_vertex(1);
    mesh::Point<2> vC = triangle.get_vertex(2);

    double volume = triangle.volume();

    double lenght_01 = distance(vA, vB);
    double lenght_12 = distance(vB, vC);
    double lenght_20 = distance(vC, vA);

    //compute the stiffness coefficients
    double S_00 = lenght_12 * lenght_12 / (4 * volume);
    double S_11 = lenght_20 * lenght_20 / (4 * volume);
    double S_22 = lenght_01 * lenght_01 / (4 * volume);

    double S_01 = dot(vB, vC, vC, vA) / (4 * volume);
    double S_02 = dot(vC, vB, vB, vA) / (4 * volume);
    double S_10 = dot(vA, vC, vC, vB) / (4 * volume);
    double S_12 = dot(vC, vA, vA, vB) / (4 * volume);
    double S_20 = dot(vA, vB, vB, vC) / (4 * volume);
    double S_21 = dot(vB, vA, vA, vC) / (4 * volume);

    std::vector<types::global_dof_index> global_dofs_on_cell = dof_handler.get_ordered_dofs_on_element(elem);

    //retrieve the solution values on the current element
    double u_0 = solution[global_dofs_on_cell[0]];
    double u_1 = solution[global_dofs_on_cell[1]];
    double u_2 = solution[global_dofs_on_cell[2]];

    //compute the gradient of the shape functions
    double grad_phi_0_x = (vC.coords[1] - vB.coords[1]) / (2 * volume);
    double grad_phi_1_x = (vA.coords[1] - vC.coords[1]) / (2 * volume);
    double grad_phi_2_x = (vB.coords[1] - vA.coords[1]) / (2 * volume);

    double grad_phi_0_y = (vB.coords[0] - vC.coords[0]) / (2 * volume);
    double grad_phi_1_y = (vC.coords[0] - vA.coords[0]) / (2 * volume);
    double grad_phi_2_y = (vA.coords[0] - vB.coords[0]) / (2 * volume);

    //compute the gradient of the solution on the current element
    double grad_u_x = grad_phi_0_x * u_0 + grad_phi_1_x * u_1 + grad_phi_2_x * u_2;
    double grad_u_y = grad_phi_0_y * u_0 + grad_phi_1_y * u_1 + grad_phi_2_y * u_2;

    // coeff = 1/sqrt(1 + |grad(u)|^2)
    double coeff = 1 / std::sqrt(1 + grad_u_x * grad_u_x + grad_u_y * grad_u_y);
    // coeff^3
    double coeff3 = coeff * coeff * coeff;

    local_matrix.set_to_zero();
    local_rhs.fill(0.0);

    //first term: (grad(phi_i), coeff*grad(phi_j))
    local_matrix(0, 0) += S_00 * coeff;
    local_matrix(1, 1) += S_11 * coeff;
    local_matrix(2, 2) += S_22 * coeff;

    local_matrix(0, 1) += S_01 * coeff;
    local_matrix(0, 2) += S_02 * coeff;
    local_matrix(1, 0) += S_10 * coeff;
    local_matrix(1, 2) += S_12 * coeff;
    local_matrix(2, 0) += S_20 * coeff;
    local_matrix(2, 1) += S_21 * coeff;

    //second term: (grad(u) \dot grad(phi_i), coeff^3 * grad(u) \dot grad(phi_j))
    local_matrix(0, 0) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * volume;
    local_matrix(0, 1) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * volume;
    local_matrix(0, 2) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * volume;

    local_matrix(1, 0) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * volume;
    local_matrix(1, 1) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * volume;
    local_matrix(1, 2) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * volume;

    local_matrix(2, 0) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * dot(grad_u_x, grad_u_y, grad_phi_0_x, grad_phi_0_y) * volume;
    local_matrix(2, 1) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * dot(grad_u_x, grad_u_y, grad_phi_1_x, grad_phi_1_y) * volume;
    local_matrix(2, 2) += coeff3 * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * dot(grad_u_x, grad_u_y, grad_phi_2_x, grad_phi_2_y) * volume;

    //rhs term: -(grad(phi_i), coeff * grad(u))
    local_rhs[0] -= coeff * dot(grad_phi_0_x, grad_phi_0_y, grad_u_x, grad_u_y) * volume;
    local_rhs[1] -= coeff * dot(grad_phi_1_x, grad_phi_1_y, grad_u_x, grad_u_y) * volume;
    local_rhs[2] -= coeff * dot(grad_phi_2_x, grad_phi_2_y, grad_u_x, grad_u_y) * volume;

    // At this point the local matrix and vector are constructed: we
    // need to sum them into the global matrix and vector.
    
    linalg::MatrixTools::add_local_matrix_to_global(stiffness_matrix, local_matrix, global_dofs_on_cell);
    linalg::MatrixTools::add_local_vector_to_global(residual_vector, local_rhs, global_dofs_on_cell);
  }

    // We apply homogeneous Dirichlet boundary conditions.
    // The linear system solution is delta, which is the difference between
    // u_{n+1}^{(k+1)} and u_{n+1}^{(k)}. Both must satisfy the same Dirichlet
    // boundary conditions: therefore, on the boundary, delta = u_{n+1}^{(k+1)} -
    // u_{n+1}^{(k+1)} = 0. We impose homogeneous Dirichlet BCs.
    linalg::MatrixTools::apply_homogeneous_dirichlet(stiffness_matrix, residual_vector, dof_handler, 0);
}

void
MinSurFF::solve()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-6;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  // We apply the Dirichlet boundary conditions to the linear system before
  // starting the Newton iteration. Since the delta vector is the difference
  // between two solutions, in the Newton iteration we will apply homogeneous
  // Dirichlet boundary conditions.

  linalg::MatrixTools::interpolate(solution, dof_handler, function_g);

  // A GOOD STARTING POINT FOR THE NEWTON ITERATION MAY BE THE FUNCTION G
  // ITSELF, SO WE SET THE SOLUTION VECTOR TO THE INTERPOLATION OF G.
  // IN FACT, AS WE KNOW, WE HAVE CONVERGENCE RESULTS FOR THE NEWTON METHOD
  // ONLY IF THE INITIAL GUESS IS CLOSE ENOUGH TO THE SOLUTION.
  // THIS IS A TRIVIAL WAY TO SET THE INITIAL GUESS. IN PRACTICE, WE MAY
  // USE A MORE SOPHISTICATED STRATEGIES.

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble();
      residual_norm = residual_vector.norm();

      std::cout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      if (residual_norm > residual_tolerance)
        {
          solve_linear_system();
          //solution = solution + delta; // update the solution
          linalg::Vector::axpby(newton_step, delta, 1, solution);
        }
      else
        {
          std::cout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

void
MinSurFF::solve_linear_system()
{
  std::cout << "===============================================" << std::endl;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.

  linalg::CGSolver cg_solver(1000, 1e-6 * residual_vector.norm());

  std::cout << "  Solving the linear system" << std::endl;

  delta = cg_solver.solve(stiffness_matrix, residual_vector);
}

void
MinSurFF::output()
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  mesh::DataIO<dim, dim> data_out(mesh, dof_handler, solution);

  // Add the function g to the output.
    // linalg::MatrixTools::interpolate()
//   VectorTools::interpolate(dof_handler, function_g, g_values);
//   data_out.add_data_vector(dof_handler, g_values, "function_g");

  const std::string           output_file_name = "output-MinSurFF.vtk";
  data_out.save_vtx(output_file_name);
  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

int main()
{

    std::function<double(double, double)> g = [](double x, double y) { return x * x + y * y; };
    // std::function<double(double, double)> g = [](double x, double y) { return std::sin(2.0 * M_PI * x); };
    // std::function<double(double, double)> g = [](double x, double y) { return 1 - x * x - y * y; };
    // std::function<double(double, double)> g = [](double x, double y) { 
    //     if(x == -1 || x == 1)
    //         return -1;
    //     else return 1;
    //  };
    // std::function<double(double, double)> g = [](double x, double y) { 
    //     if(x == -1 || x == 1)
    //         return y;
    //     else return x;
    // };

    MinSurFF problem_ff(100, g, 1.3);

    problem_ff.setup();
    problem_ff.assemble();
    problem_ff.solve();
    problem_ff.output();

}