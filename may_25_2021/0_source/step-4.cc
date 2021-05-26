#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>
using namespace dealii;


template <int dim>
class Step4
{
public:
  Step4(unsigned int element_degree,
        unsigned int id_mesh_being_created,
        unsigned int grid_parameter);
  
  void run();
  
private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void computing_the_error();
  void storing_error_cpu_time_to_a_file();
  
  const unsigned int element_degree;
  const unsigned int id_mesh_being_created;
  const unsigned int grid_parameter;
  
  unsigned int current_refinement_level;
  
  
  Triangulation<dim> triangulation;
  
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;
  
  double solution_L2_error_abs = 0;
  double solution_H1_semi_error_abs = 0;
  double solution_H2_semi_error_abs = 0;
  
  TimerOutput          computing_timer;
  double total_CPU_time_per_run = 0;
};



template <int dim>
class ExactSolution : public Function<dim>
{
public:
  
  virtual double value (const Point<dim>   &p,
                             const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                       const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                               const unsigned int  component = 0) const;
protected:
    
};


template <int dim>
class Coeff_Diff_Real
{
  public:

    Tensor<2,dim> value (const Point<dim> &p,
                  const unsigned int  component = 0) const;
    Tensor<2,dim,Tensor<1,dim>> gradient (const Point<dim> &p,
                   const unsigned int  component = 0) const;       
    Tensor<2,dim> gradient_x_direction (const Point<dim> &p,
                               const unsigned int  component = 0) const;       
    Tensor<2,dim> gradient_y_direction (const Point<dim> &p,
                               const unsigned int  component = 0) const;
};


template <int dim>
class RightHandSide:
public ExactSolution<dim>,
public Coeff_Diff_Real<dim>
{
public:
  virtual double value_rhs(const Point<dim> & p,
                           const unsigned int component = 0);
private:
  Tensor<1,dim> gradient_p;
  SymmetricTensor<2,dim> hessian_p;
  
  Tensor<2,dim> value_d;
  Tensor<2,dim> gradient_d_x_direction;
  Tensor<2,dim> gradient_d_y_direction;
  
  Tensor<1,dim> contribution_gradient_d;
    
};


template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p,
                                                const unsigned int ) const
{
  double return_value = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0)));
     
  return return_value;
}

template <int dim>
Tensor<1,dim> ExactSolution<dim>::gradient (const Point<dim> &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;

  return_value[0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[0] - 0.5));
  return_value[1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[1] - 0.5));
  
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution<dim>::hessian (const Point<dim>   &p,
                                       const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;

  return_value[0][0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (4.0 * std::pow((p[0] - 0.5), 2) - 2.0);
  return_value[0][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[0] - 0.5)) * (-2.0 * (p[1] - 0.5));
  return_value[1][0] = return_value[0][1];
  return_value[1][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (4.0 * std::pow((p[1] - 0.5), 2) - 2.0);

  return return_value;
}

template <int dim>
Tensor<2,dim>
Coeff_Diff_Real<dim>::value (const Point<dim> &/*p*/,
                             const unsigned int ) const
{           
  Tensor<2,dim> return_value;
  
#if 1
  
    return_value[0][0] = 1.0;
    return_value[0][1] = 0.0;
    return_value[1][0] = return_value[0][1];
    return_value[1][1] = return_value[0][0];      
      
#else      
      
    return_value[0][0] = 1.0 + (p[0] + p[1]);
    return_value[0][1] = p[0]*p[1];
    return_value[1][0] = return_value[0][1];
    return_value[1][1] = return_value[0][0];
    
#endif
    
  return return_value;
}


template <int dim>
Tensor<2,dim,Tensor<1,dim>>
Coeff_Diff_Real<dim>::gradient (const Point<dim> &/*p*/,
                                const unsigned int) const
{
  Tensor<2,dim,Tensor<1,dim>> return_value;

#if 1
  
    return_value[0][0][0] = 0.0;
    return_value[0][1][0] = 0.0;
    return_value[1][0][0] = return_value[0][1][0];
    return_value[1][1][0] = return_value[0][0][0];   

    return_value[0][0][1] = 0.0;
    return_value[0][1][1] = 0.0;
    return_value[1][0][1] = return_value[0][1][1];
    return_value[1][1][1] = return_value[0][0][1];   

#else
    
    return_value[0][0][0] = 1.0;
    return_value[0][1][0] = p[1];
    return_value[1][0][0] = return_value[0][1][0];
    return_value[1][1][0] = return_value[0][0][0];   

    return_value[0][0][1] = 1.0;
    return_value[0][1][1] = p[0];
    return_value[1][0][1] = return_value[0][1][1];
    return_value[1][1][1] = return_value[0][0][1];
    
#endif    
    
    return return_value;
    
}

template <int dim>
Tensor<2,dim>
Coeff_Diff_Real<dim>::gradient_x_direction (const Point<dim> &p,
                                            const unsigned int) const
{
  Tensor<2,dim> return_value;
    
  Tensor<2,dim,Tensor<1,dim>> gradient_d = Coeff_Diff_Real<dim>::gradient(p);

  for (unsigned int j=0; j<dim; ++j)
  {
    for (unsigned int k=0; k<dim; ++k)
    {
      return_value[j][k] = gradient_d[j][k][0];    
    }
  }
  
  return return_value;
}

template <int dim>
Tensor<2,dim>
Coeff_Diff_Real<dim>::gradient_y_direction (const Point<dim> &p,
                                            const unsigned int) const
{
  Tensor<2,dim> return_value;
    
  Tensor<2,dim,Tensor<1,dim>> gradient_d = Coeff_Diff_Real<dim>::gradient(p);

  for (unsigned int j=0; j<dim; ++j)
  {
    for (unsigned int k=0; k<dim; ++k)
    {
      return_value[j][k] = gradient_d[j][k][1];    
    }
  }
  
  return return_value;
}


template <int dim>
double RightHandSide<dim>::value_rhs(const Point<dim> &p,
                                   const unsigned int /*component*/)
{
  
  gradient_p = ExactSolution<dim>::gradient(p);
  hessian_p = ExactSolution<dim>::hessian(p);
  
  value_d = Coeff_Diff_Real<dim>::value(p);
  gradient_d_x_direction = Coeff_Diff_Real<dim>::gradient_x_direction(p);
  gradient_d_y_direction = Coeff_Diff_Real<dim>::gradient_y_direction(p);
  
  contribution_gradient_d = gradient_d_x_direction[0] + gradient_d_y_direction[1];
  
//   std::cout << "vector_value_d: \n";
//   for(unsigned int i=0; i<values.size(); ++i)
//   {
//       std::cout << vector_value_d[i] << "\n";
//   }
  
  double return_value;
  
      
//     return_value = -exp(-(pow(p[0]-0.5, 2.0) + pow(p[1]-0.5, 2.0))) * (4.0 * (std::pow(p[0] - 0.5, 2) + std::pow(p[1] - 0.5, 2) - 1.0));           // D = Identity matrix
    
    return_value = -(contribution_gradient_d * gradient_p 
                     + trace(value_d*hessian_p));
    
    return return_value;
  
}


template <int dim>
Step4<dim>::Step4(unsigned int element_degree,
                  unsigned int id_mesh_being_created,
                  unsigned int grid_parameter)
:
  element_degree(element_degree),
  id_mesh_being_created(id_mesh_being_created),
  grid_parameter(grid_parameter),
  fe(element_degree),
  dof_handler(triangulation),
  computing_timer  (std::cout, TimerOutput::summary, TimerOutput::cpu_times)
{}



template <int dim>
void Step4<dim>::make_grid()
{
    
  std::cout << "Making grid\n";
  TimerOutput::Scope t(computing_timer, "make_grid");
      
    
  if(id_mesh_being_created == 0)
  {   
    GridGenerator::hyper_cube(triangulation, 0, 1);
    
    triangulation.begin_active()->face(2)->set_boundary_id(1);                    // bottom
    triangulation.begin_active()->face(3)->set_boundary_id(1);                    // top
    
    current_refinement_level = grid_parameter;
    
    triangulation.refine_global(current_refinement_level);
    
  }else if(id_mesh_being_created == 1)
  {
    
    current_refinement_level = 0;
      
    unsigned int n_dofs_custom = int(grid_parameter + 0.5);
        
    unsigned int n_dofs_custom_in_one_direction = int(sqrt(n_dofs_custom) + 0.5);
        
    unsigned int n_cells_custom_in_one_direction = int((n_dofs_custom_in_one_direction - 1)/element_degree + 0.5);                   // '+ 0.5' is for relatively correct rounding
        
    std::cout << "      n_dofs_custom: " << n_dofs_custom << "\n";
    std::cout << "      n_dofs_custom_in_one_direction: " << n_dofs_custom_in_one_direction << "\n";
    std::cout << "   n_cells_custom_in_one_direction: " << n_cells_custom_in_one_direction << "\n";

    std::vector<unsigned int> repetitions(2);
    repetitions[0] = n_cells_custom_in_one_direction;
    repetitions[1] = n_cells_custom_in_one_direction;
    
    const Point<dim> p1 (0.0, 0.0);                         
    const Point<dim> p2 (1.0, 1.0);
    
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              p1,
                                              p2);
    
    
    
  
    Triangulation<2>::active_face_iterator face = triangulation.begin_active_face();      // method 2 for setting boundary ids
    Triangulation<2>::active_face_iterator endface = triangulation.end_face();    
    
    for (; face!=endface; ++face)                                 // this iterates from the first active face to the last active face
    {
      if(face->at_boundary())
      {
        if(face->vertex(0)(1)==p2[1] && face->vertex(1)(1)==p2[1])
        {
          face->set_boundary_id(1);        
        }else if(face->vertex(0)(1)==p1[1] && face->vertex(1)(1)==p1[1])
        {
          face->set_boundary_id(1);
        }
      }
    }

  }
  
  
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
  
}



template <int dim>
void Step4<dim>::setup_system()
{
  std::cout << "setup_system\n";
  TimerOutput::Scope t(computing_timer, "setup_system");
    
  dof_handler.distribute_dofs(fe);
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void Step4<dim>::assemble_system()
{
    
  std::cout << "assemble_system\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");    
    
  QGauss<dim> quadrature_formula(fe.degree + 2);
  QGauss<dim-1> face_quadrature_formula(fe.degree + 2);
  
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
  Coeff_Diff_Real<dim> obj_coeff_diff;
  ExactSolution<dim> exact_solution;
  RightHandSide<dim> right_hand_side;
  

  
  Tensor<2,dim> value_coeff_diff;
  Tensor<2,dim> value_coeff_diff_face;
  double value_rhs;
  
  for (const auto &cell : dof_handler.active_cell_iterators())
    {     
        
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        
        value_coeff_diff = obj_coeff_diff.value(fe_values.quadrature_point(q_index));
        value_rhs = right_hand_side.value_rhs(fe_values.quadrature_point(q_index));
          
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
            {
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 value_coeff_diff *
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
            }
            
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            value_rhs *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }
      }
      
      for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
      {
        if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 1))
        {
            
            fe_face_values.reinit (cell, face_n);
            
            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            {
                value_coeff_diff_face = obj_coeff_diff.value (fe_face_values.quadrature_point(q_point));
                
                const double neumann_value
                = (value_coeff_diff_face * exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *             //
                    fe_face_values.normal_vector(q_point));
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += (neumann_value *
                                fe_face_values.shape_value(i,q_point) *
                                fe_face_values.JxW(q_point));
                }
            }
        }
      }          
      
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    
  std::cout << "   applying the boundary condition\n";
    
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           ExactSolution<dim>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
  
}



template <int dim>
void Step4<dim>::solve()
{
    
  std::cout << "solve\n";
  TimerOutput::Scope t(computing_timer, "solve");     
    
#if 1
  
  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult (solution, system_rhs);
  
#else
  
  SolverControl           solver_control (1e7, 1e-18);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;  
  
#endif

}




template <int dim>
void Step4<dim>::computing_the_error()
{
    
  std::cout << "computing_the_error\n";
  TimerOutput::Scope t(computing_timer, "computing_the_error");     
  
  ExactSolution<dim> exact_solution;
  
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
  QGauss<dim> qgauss_for_integrating_difference = QGauss<dim>(fe.degree+1);
  
  VectorTools::integrate_difference (dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    qgauss_for_integrating_difference,
                                    VectorTools::L2_norm);
  solution_L2_error_abs = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);

  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  solution_H1_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_seminorm);

  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  solution_H2_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_seminorm);
}



template <int dim>
void Step4<dim>::storing_error_cpu_time_to_a_file()
{    
    
  std::ofstream myfile;
  
  std::string obj_string = "data_output_error_0_sm_0_real.txt";                         // _alter
  
  myfile.open (obj_string, std::ofstream::app);
  
  myfile << current_refinement_level << " ";
  myfile << triangulation.n_vertices() <<" ";
  myfile << dof_handler.n_dofs() << " ";
  

  myfile << solution_L2_error_abs << " ";
  myfile << solution_H1_semi_error_abs << " ";
  myfile << solution_H2_semi_error_abs << " ";      


  myfile << 0 << " ";
  myfile << 0 << " ";
  myfile << 0 << " ";
  myfile << total_CPU_time_per_run;


  myfile << "\n";

  myfile.close();
}



template <int dim>
void Step4<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;
  make_grid();
  setup_system();
  assemble_system();
  solve();
  
  total_CPU_time_per_run = computing_timer.return_total_cpu_time ();
  
  computing_the_error();
  
  storing_error_cpu_time_to_a_file();
  
}



int main(int argc, char *argv[])
{
    
  unsigned int element_degree;
  unsigned int id_mesh_being_created;
  unsigned int grid_parameter;
  
  const unsigned int n_arguments_for_checking = 4;
  
  if (argc != n_arguments_for_checking)
  {
    std::cout << "usage: "
              << "<element_degree> "
              << "<id_mesh_being_created> "
              << "<grid_parameter> "
              << "\n";
    
    exit(EXIT_FAILURE);
              
  }

//   std::cout << "The arguments read\n";
//   for (int i = 0; i < argc; ++i)
//   {
//     std::cout << argv[i] << "\n";
//   }
  
  element_degree = std::stoi(argv[1]);
  id_mesh_being_created = std::stoi(argv[2]);
  grid_parameter = std::stoi(argv[3]);
  
#if 1
    
  deallog.depth_console(0);
  {
    Step4<2> laplace_problem_2d(element_degree,
                                id_mesh_being_created,
                                grid_parameter);
    laplace_problem_2d.run();
  }
  
#endif
  
  return 0;
}
