

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>


#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>


#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include <quadmath.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>                               // these two for the local mesh refinement
#include <deal.II/numerics/error_estimator.h>

using namespace dealii;                                                 // important putting this before calling external functions

#include"common_cpp_function.h"
#include"selected_function.h"

#define pi 3.141592653589793238462643


template <int dim>
class Step4
{
public:
  Step4(double coeff_solu,
        unsigned int element_degree,
        unsigned int id_mesh_being_created,
        unsigned int grid_parameter,
        unsigned int n_total_refinements);
  
  void run();
  
private:
  void make_grid();
  void refining_mesh_locally();
  
  void setup_system();

  void assemble_system();
  
  
  void solve();
  
  void output_results();
  
  void computing_the_error_using_built_in_functions();

  void storing_error_cpu_time_to_a_file();
  
  const double coeff_solu;
  const unsigned int element_degree;
  const unsigned int id_mesh_being_created;
  const unsigned int grid_parameter;
  const unsigned int n_total_refinements;
  
  unsigned int current_refinement_level;
  
  
  Triangulation<dim> triangulation;
  
  AffineConstraints<double> constraints;                            // for adaptive mesh refinement
  unsigned int is_constraints_used = 1;
  
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;
  
  Vector<float> estimated_error_per_cell_from_the_computed_error;
  
  
  std::streamsize ss = std::cout.precision();
  
  double solution_L2_norm = 0.0;
  
  double solution_L2_error_abs_last_refinement = 1.0;
  
  double solution_L2_error_abs = 1.0;
  double solution_H1_semi_error_abs = 1.0;
  double solution_H2_semi_error_abs = 1.0;
  
  float minimum_grid_size = 1.0;
  


  int cycle_global = 0;
  
  TimerOutput          computing_timer;
  double total_CPU_time_per_run = 0;
};


template <int dim>
class ExactSolution : public Function<dim>
{
public:
    
  ExactSolution (const double coeff_solu);  
  
  virtual double value (const Point<dim>   &p,
                             const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                       const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                               const unsigned int  component = 0) const;
                                               
protected:
    const double coeff_solu;                                               
    
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
class RightHandSide_Direct_Evaluation : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};



template <int dim>
class RightHandSide:
public ExactSolution<dim>,
public Coeff_Diff_Real<dim>
{
public:
    
  RightHandSide (const double coeff_solu);   
  
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
ExactSolution<dim>::ExactSolution(const double coeff_solu):
coeff_solu(coeff_solu)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const double coeff_solu):
ExactSolution<dim>(coeff_solu)
{}


template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p,
                                                const unsigned int ) const
{
  double return_value;
  
  if(dim == 1)
  {  
//     return_value = exp(-pow(p[0] - 0.5, 2.0));
    
    return_value = exp(-pow(p[0] - 0.5, 2.0)*1000.0);                               // 1e5
    
//     return_value = exp(-pow(p[0] - 0.5, 2.0)*1000.0) + 0.1 * cos(pi * p[0]);
    
//     return_value = pow(p[0] - 0.5, 2.0);
    
//     return_value = pow(p[0], 2.0) - p[0];
    
//     return_value = 1.0 + p[0];
    
  }else if(dim == 2)
  {
#if 0
    return_value = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0)));
#else
//     return_value = 0.0;

    return_value = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu);   // + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001)  + 0.5 * (1.0 + cos (pi * p[0])) + 0.5 * sin (pi * p[1]) * (1.0 - p[0]);
#endif
  }
     
  return return_value;
}

template <int dim>
Tensor<1,dim> ExactSolution<dim>::gradient (const Point<dim> &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;
  
  if(dim == 1)
  {
//     return_value[0] = exp(-pow(p[0] - 0.5, 2.0)) * (-2.0 * (p[0] - 0.5));
    
    return_value[0] = exp(-pow(p[0] - 0.5, 2.0)*1000.0) * (-2.0 * 1000.0 * (p[0] - 0.5));
    
//     return_value[0] = exp(-pow(p[0] - 0.5, 2.0)*1000.0) * (-2.0 * 1000.0 * (p[0] - 0.5)) - 0.1 * pi * sin(pi * p[0]);
    
//     return_value[0] = 2.0 * (p[0] - 0.5);
      
//     return_value[0] = 2.0 * p[0] - 1.0;
    
//     return_value[0] = 1.0;
    
  }else if(dim == 2)
  {
#if 0
    return_value[0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[0] - 0.5));
    return_value[1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[1] - 0.5));
#else
//     return_value[0] = 0.0;
//     return_value[1] = 0.0;

    return_value[0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu) * (-2.0 * (p[0] - 0.5) / coeff_solu);     //+ exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) * (-2.0 * (p[0] - 0.9) / 0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001) * (-2.0 * (p[0] - 0.9) / 0.001) - 0.5 * pi * sin (pi * p[0]) - 0.5 * sin (pi * p[1]);
    return_value[1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu) * (-2.0 * (p[1] - 0.5) / coeff_solu);     //+ exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) * (-2.0 * (p[1] - 0.2) / 0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001) * (-2.0 * (p[1] - 0.8) / 0.001) + 0.5 * pi * cos (pi * p[1]) * (1.0 - p[0]);
#endif      
    
  }

  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution<dim>::hessian (const Point<dim>   &p,
                                       const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;

  if(dim == 1)
  {
//     return_value[0][0] = exp(-pow(p[0] - 0.5, 2.0)) * (std::pow(2.0 * (p[0] - 0.5), 2) - 2.0 * 1.0);
    
    return_value[0][0] = exp(-pow(p[0] - 0.5, 2.0)*1000.0) * (std::pow(2.0 * (p[0] - 0.5) *1000.0, 2) - 2.0 *1000.0);
    
//     return_value[0][0] = exp(-pow(p[0] - 0.5, 2.0)*1000.0) * (std::pow(2.0 * (p[0] - 0.5) *1000.0, 2) - 2.0 *1000.0) - 0.1 * std::pow(pi, 2.0) * cos(pi * p[0]);
    
//     return_value[0][0] = 2.0;
    
//     return_value[0][0] = 0.0;
    
  }else if(dim == 2)
  {  
#if 0
    return_value[0][0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (4.0 * std::pow((p[0] - 0.5), 2) - 2.0);
    return_value[0][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (-2.0 * (p[0] - 0.5)) * (-2.0 * (p[1] - 0.5));
    return_value[1][0] = return_value[0][1];
    return_value[1][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))) * (4.0 * std::pow((p[1] - 0.5), 2) - 2.0);
#else
//     return_value[0][0] = 0.0;
//     return_value[0][1] = 0.0;
//     return_value[1][0] = return_value[0][1];
//     return_value[1][1] = 0.0;


    return_value[0][0] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu) * (std::pow(2.0 * (p[0] - 0.5)/coeff_solu, 2) - 2.0 / coeff_solu);  //  + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) * (std::pow(2.0 * (p[0] - 0.9)/0.001, 2) - 2.0 / 0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001) * (std::pow(2.0 * (p[0] - 0.9)/0.001, 2) - 2.0 / 0.001) - 0.5 * std::pow(pi, 2.0) * cos (pi * p[0]);
    return_value[0][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu) * (-2.0 * (p[0] - 0.5)/coeff_solu) * (-2.0 * (p[1] - 0.5)/coeff_solu);  //  + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) * (-2.0 * (p[0] - 0.9)/0.001) * (-2.0 * (p[1] - 0.2)/0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001) * (-2.0 * (p[0] - 0.9)/0.001) * (-2.0 * (p[1] - 0.8)/0.001) - 0.5 * pi * cos (pi * p[1]);
    return_value[1][0] = return_value[0][1];
    return_value[1][1] = exp(-(pow(p[0] - 0.5, 2.0) + pow(p[1] - 0.5, 2.0))/coeff_solu) * (std::pow(2.0 * (p[1] - 0.5)/coeff_solu, 2) - 2.0 / coeff_solu);  //  + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.2, 2.0))/0.001) * (std::pow(2.0 * (p[1] - 0.2)/0.001, 2) - 2.0 / 0.001) + exp(-(pow(p[0] - 0.9, 2.0) + pow(p[1] - 0.8, 2.0))/0.001) * (std::pow(2.0 * (p[1] - 0.8)/0.001, 2) - 2.0 / 0.001) - 0.5 * pow(pi, 2.0) * sin (pi * p[1]) * (1.0 - p[0]);    
#endif      
    
  }
  
  return return_value;
}


template <int dim>
Tensor<2,dim>
Coeff_Diff_Real<dim>::value (const Point<dim> &/*p*/,
                             const unsigned int ) const
{           
  Tensor<2,dim> return_value;
  
  if (dim == 1)
  {
    return_value[0][0] = 1.0;   
  }else if(dim == 2)
  {
  
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
      
  }
    
  return return_value;
}


template <int dim>
Tensor<2,dim,Tensor<1,dim>>
Coeff_Diff_Real<dim>::gradient (const Point<dim> &/*p*/,
                                const unsigned int) const
{
  Tensor<2,dim,Tensor<1,dim>> return_value;

  if (dim == 1)
  {
    return_value[0][0][0] = 0.0;
  }else if(dim == 2)
  {
      
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
  }
    
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
  
  if(dim == 1)
  {
    contribution_gradient_d = gradient_d_x_direction[0];
  }else if(dim == 2)
  {
    gradient_d_y_direction = Coeff_Diff_Real<dim>::gradient_y_direction(p);
    contribution_gradient_d = gradient_d_x_direction[0] + gradient_d_y_direction[1];
  }
  
  double return_value;
  
//     return_value = -exp(-(pow(p[0]-0.5, 2.0) + pow(p[1]-0.5, 2.0))) * (4.0 * (std::pow(p[0] - 0.5, 2) + std::pow(p[1] - 0.5, 2) - 1.0));           // D = Identity matrix
    
    return_value = -(contribution_gradient_d * gradient_p 
                     + trace(value_d*hessian_p));
    
    return return_value;
}


template <int dim>
double RightHandSide_Direct_Evaluation<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double return_value = 0.0;
  
  return_value = - exp(-pow(p[0] - 0.5, 2.0)) * (4.0 * std::pow((p[0] - 0.5), 2) - 2.0);
    
  return return_value;
}


template <int dim>
Step4<dim>::Step4(double coeff_solu,
                  unsigned int element_degree,
                  unsigned int id_mesh_being_created,
                  unsigned int grid_parameter,
                  unsigned int n_total_refinements)
:
  coeff_solu(coeff_solu),
  element_degree(element_degree),
  id_mesh_being_created(id_mesh_being_created),
  grid_parameter(grid_parameter),
  n_total_refinements(n_total_refinements),
  fe(element_degree),                                                           // element_degree  QIterated<1>(QTrapez<1>(), element_degree)
  dof_handler(triangulation),
  computing_timer  (std::cout, TimerOutput::never, TimerOutput::cpu_times)
{}


template <int dim>
void Step4<dim>::make_grid()
{
    
  std::cout << "Making grid\n";
  TimerOutput::Scope t(computing_timer, "make_grid");
      
    
  if(id_mesh_being_created == 0)
  {
    
#if 1
    GridGenerator::hyper_cube(triangulation, 0, 1);
    
    if (dim == 1)
    {
      triangulation.last_active()->face(1)->set_boundary_id(0);
    }else if(dim == 2)
    {
      triangulation.begin_active()->face(2)->set_boundary_id(1);                    // bottom
      triangulation.begin_active()->face(3)->set_boundary_id(1);                    // top
    }
    
#else

    const Point<dim> p1 (0.0);                                                      // not successful             
    const Point<dim> p2 (0.3);
    const Point<dim> p3 (0.7);
    const Point<dim> p4 (1.0);

    const std::vector<Point<dim>> vertices = {p1, p2, p3, p4};
    
    const std::vector<unsigned int> cell_vertices[3] = {{0, 1}, {1, 2}, {2, 3}};
    
    std::vector<CellData<dim>> cells(3, CellData<dim>());
    for (unsigned int i = 0; i < cells.size(); ++i)
    {
        cells[i].vertices[0]    = cell_vertices[i][0];
        cells[i].vertices[1]    = cell_vertices[i][1];
        cells[i].material_id = 0;
    }
    triangulation.create_triangulation(
    vertices,
    cells,
    SubCellData());                          // No boundary information

    triangulation.last_active()->face(1)->set_boundary_id(0); 
    
#endif
    
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
    
    for (; face!=endface; ++face)                                                         // this iterates from the first active face to the last active face
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
  
  std::cout << "    current_refinement_level: " << current_refinement_level << "\n";
  std::cout << "    Number of active cells: " << triangulation.n_active_cells()
//             << std::endl
//             << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
  
}


template <int dim>
void Step4<dim>::refining_mesh_locally()
{
  std::cout << "refining_mesh_locally\n";
  TimerOutput::Scope t(computing_timer, "refining_mesh_locally");
    
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
  
  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      {},
                                      solution,
                                      estimated_error_per_cell);
  
  
//   std::cout << "    estimated_error_per_cell: ";
//   estimated_error_per_cell.print(std::cout,3);
//   std::cout << "    estimated_error_per_cell_from_the_computed_error: ";
//   estimated_error_per_cell_from_the_computed_error.print(std::cout,3);
  
#if 0
  std::ofstream fid("0_data_output/data_output_estimated_error_per_cell_cycle_" + std::to_string(cycle_global) + ".txt", std::ofstream::trunc);
  
  for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)                                                                  
  {
    fid << estimated_error_per_cell[i] << "\n";
  }

  fid.close();
  fid.clear();    
#endif
  

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
//                                                    estimated_error_per_cell,
                                                   estimated_error_per_cell_from_the_computed_error,
                                                   0.3, 0.0);
  triangulation.execute_coarsening_and_refinement ();    
}


template <int dim>
void Step4<dim>::setup_system()
{
  std::cout << "setup_system\n";
  TimerOutput::Scope t(computing_timer, "setup_system");
  
  dof_handler.distribute_dofs(fe);
  
  std::cout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
            
  
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  
  if (is_constraints_used == 1)
  {
    constraints.clear();                                                                  // for the adaptive mesh refinement
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            ExactSolution<dim>(coeff_solu),
                                            constraints);
    constraints.close();
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);  
  }else
  {
    DoFTools::make_sparsity_pattern(dof_handler, dsp);  
  }
  
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  
}


template <int dim>
void Step4<dim>::assemble_system()
{
  std::cout << "assemble_system\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");    
  
  QGauss<dim> quadrature_formula(fe.degree + 1);
  
  QGauss<dim-1> face_quadrature_formula(fe.degree + 1);
  
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values | update_jacobians);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values | update_jacobians);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
//   Coeff_Diff_Real<dim> obj_coeff_diff;
  ExactSolution<dim> exact_solution(coeff_solu);
  RightHandSide<dim> right_hand_side(coeff_solu);
  
//   Tensor<2,dim> value_coeff_diff;
//   Tensor<2,dim> value_coeff_diff_face;
  double value_rhs;
  
//   std::cout << value_coeff_diff << "\n";
  
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
        
//       std::cout << "    Cell no. " << cell->active_cell_index() << "\n";
        
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
//         value_coeff_diff = obj_coeff_diff.value(fe_values.quadrature_point(q_index));
        value_rhs = right_hand_side.value_rhs(fe_values.quadrature_point(q_index));                         // value_rhs
          
        for (const unsigned int i : fe_values.dof_indices())
          {
              

            
            for (const unsigned int j : fe_values.dof_indices())
            {
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
//                  value_coeff_diff *
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
                
            }
            
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            value_rhs *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }
      }
      
#if 1
      for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
      {
        if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 1))
        {
            
//             std::cout << "    applying the Neumann boundary conditions\n";
//             std::cout << "    on the face with coordinates " << cell->face(face_n)->vertex(0) << "\n";
            
            fe_face_values.reinit (cell, face_n);
            
            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            {
//                 value_coeff_diff_face = obj_coeff_diff.value (fe_face_values.quadrature_point(q_point));
                
                const double neumann_value
                = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *             //
//                    value_coeff_diff_face *
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
#endif
      
      cell->get_dof_indices(local_dof_indices);
      
      
      if(is_constraints_used == 1)
      {
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);    
      }else
      {
        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
          {
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          }
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
            
      } 

    }
     
  if (is_constraints_used == 0)
  {
    //   std::cout << "\n";
    std::cout << "    applying the Dirichlet boundary conditions\n";
        
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            ExactSolution<dim>(coeff_solu),
                                            boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }
  
//   std::cout << '\n';
//   std::cout << "system_matrix after applying the Dirichlet BC:\n";
//   system_matrix.print_formatted(std::cout, 2);
//   std::cout << "system_rhs after applying the Dirichlet BC:\n";
//   system_rhs.print(std::cout, 2);  
//   
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
  if(is_constraints_used == 1)
  {
    constraints.distribute(solution);    
  }  
  
//   std::cout << "  solution: ";
//   solution.print(std::cout, 2);            
  
}


template <int dim>
void Step4<dim>::output_results ()
{
  std::cout << "outputting results\n";
  TimerOutput::Scope t(computing_timer, "output_results");  
    
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  
  data_out.build_patches ();
  
  std::string obj_string = "0_data_output/0_solution_in_vtk/solution_" + std::to_string(dim) + "d_0_sm_deg_" + std::to_string(element_degree) + "_R_" + std::to_string(current_refinement_level) + "_cycle_" + std::to_string(cycle_global) + ".vtk";
  
  std::ofstream output (obj_string);
  data_out.write_vtk (output);
}


template <int dim>
void Step4<dim>::computing_the_error_using_built_in_functions()
{
    
  std::cout << "computing_the_error_using_built_in_functions\n";
  TimerOutput::Scope t(computing_timer, "computing_the_error_using_built_in_functions");  
  
  ExactSolution<dim> exact_solution(coeff_solu);
  const Functions::ZeroFunction<dim> zero_function;                                             // for computing the L2 norm of the solution
  
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
  
  QGauss<dim> qgauss_for_integrating_difference = QGauss<dim>(fe.degree + 1);
  
    
  std::cout << std::scientific << std::setprecision(2);
  
  
  VectorTools::integrate_difference (dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    qgauss_for_integrating_difference,
                                    VectorTools::L2_norm);
  
  solution_L2_error_abs = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);
  
//   std::cout << "    difference_per_cell for L2_norm: ";
//   difference_per_cell.print(std::cout, 3);    
  
  estimated_error_per_cell_from_the_computed_error.reinit(triangulation.n_active_cells());
  estimated_error_per_cell_from_the_computed_error = difference_per_cell;
  
#if 0
  std::ofstream fid("0_data_output/data_output_difference_per_cell_cycle_" + std::to_string(cycle_global) + ".txt", std::ofstream::trunc);
  
  for (unsigned int i = 0; i < difference_per_cell.size(); ++i)                                                                  
  {
    fid << difference_per_cell[i] << "\n";
  }

  fid.close();
  fid.clear();  
#endif    
  

#if 1
  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  solution_H1_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_seminorm);
  
  
//   std::cout << "    difference_per_cell for H1_seminorm: ";
//   difference_per_cell.print(std::cout, 3);

  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  solution_H2_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_seminorm);
  
//   std::cout << "    difference_per_cell for H2_seminorm:\n";
//   difference_per_cell.print(std::cout, 3);

#endif

  VectorTools::integrate_difference (dof_handler,
                                    solution,
                                    zero_function,
                                    difference_per_cell,
                                    qgauss_for_integrating_difference,
                                    VectorTools::L2_norm);
  
  solution_L2_norm = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);
  
  std::cout << "\n";
  
  std::cout << std::right << std::setw(30) << "solution_L2_error_abs: " << solution_L2_error_abs << "\n";
  std::cout << std::right << std::setw(30) << "solution_H1_semi_error_abs: " << solution_H1_semi_error_abs << "\n";
  std::cout << std::right << std::setw(30) << "solution_H2_semi_error_abs: " << solution_H2_semi_error_abs << "\n";
  
  std::cout << "\n";
  std::cout << std::right << std::setw(30) << "solution_L2_norm: " << solution_L2_norm << "\n";
  
//   std::cout << std::defaultfloat << std::setprecision(ss);
  
}


template <int dim>
void Step4<dim>::storing_error_cpu_time_to_a_file()
{    
    
  std::ofstream myfile;
  
  std::string obj_string = "data_output_error_0_dealii_0_sm_0_real.txt";                         // _alter_0       _case_3_PCT_50
  
  myfile.open (obj_string, std::ofstream::app);
  
//   if (n_total_refinements > 1)
//   {
//     myfile << cycle_global + 1 << " ";    
//   }
  
  myfile << current_refinement_level << " ";
  myfile << triangulation.n_vertices() <<" ";
  myfile << minimum_grid_size <<" ";
  myfile << dof_handler.n_dofs() << " ";
  

  myfile << solution_L2_error_abs << " ";
  myfile << solution_H1_semi_error_abs << " ";
  myfile << solution_H2_semi_error_abs << " ";      


  myfile << solution_L2_norm << " ";
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
  
  for (unsigned int cycle = 0; cycle < n_total_refinements; cycle++)                        // cycle used for adaptive mesh refinement
  {
    cycle_global = cycle;
    
    std::cout << "\n";
    std::cout << "    cycle " << cycle << "\n";
    
#if 1
    
    
    if (cycle != 0 && cycle < 60)                                          // we set an upper limit for the number of local mesh refinement
    {
      refining_mesh_locally();  
    }
//     else
//     {
//       triangulation.refine_global(1);                       // we can control the number of total global mesh refinements
//       current_refinement_level++;
//     }
    
    solution_L2_error_abs_last_refinement = solution_L2_error_abs;
#endif
    
    minimum_grid_size = returning_the_minimum_grid_size(triangulation);
//     std::cout << "    minimum_grid_size: " << minimum_grid_size << "\n";
      
    std::cout << std::right << std::setw(30) << "minimum_grid_size: " << minimum_grid_size << "\n";
    
#if 0
    std::ofstream fid("0_data_output/data_output_coordinate_of_vertices_cycle_" + std::to_string(cycle) + ".txt", std::ofstream::trunc);   
      
    std::cout << "    coordinates of vertices of each active cell\n";
    
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    
    for (; cell!=endc; ++cell)                                                                  
    {
        std::cout << "    [" << cell->active_cell_index() << "] ";
        
        fid << cell->vertex(0) << "\n";
        fid << cell->vertex(1) << "\n";
        
        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
        {
            std::cout << cell->vertex_index(vertex) << " (" << cell->vertex(vertex) << ") ";     
        }
        
        std::cout << "\n";
    }
    
    fid.close();
    fid.clear();
#endif
  
    setup_system();
    
#if 1
    assemble_system();
    
    solve();
    
//     output_results();
    
    computing_the_error_using_built_in_functions();
    
    computing_timer.print_summary();
    
    total_CPU_time_per_run = computing_timer.return_total_cpu_time ();
    
    std::cout << std::right << std::setw(30) << "total_CPU_time_per_run: " << total_CPU_time_per_run << "\n";
    
    computing_timer.reset();
    
    storing_error_cpu_time_to_a_file();
    
    if(cycle >= 4 && solution_L2_error_abs > solution_L2_error_abs_last_refinement)
    {
        std::cout << "    solution_L2_error_abs: " << solution_L2_error_abs << "\n";
        std::cout << "    solution_L2_error_abs_last_refinement: " << solution_L2_error_abs_last_refinement << "\n";
        break;
    }
#endif
  }
}


int main(int argc, char *argv[])
{
    
  double coeff_solu;
  unsigned int element_degree;
  unsigned int id_mesh_being_created;
  unsigned int grid_parameter;
  unsigned int n_total_refinements;
  
  const unsigned int n_arguments_for_checking = 6;
  
  if (argc != n_arguments_for_checking)
  {
    std::cout << "usage: "
              << "<coeff_solu> "
              << "<element_degree> "
              << "<id_mesh_being_created> "
              << "<grid_parameter> "
              << "<n_total_refinements> "
              << "\n";
    
    exit(EXIT_FAILURE);
              
  }

//   std::cout << "The arguments read\n";
//   for (int i = 0; i < argc; ++i)
//   {
//     std::cout << argv[i] << "\n";
//   }
  
  coeff_solu = std::stof(argv[1]);
  element_degree = std::stoi(argv[2]);
  id_mesh_being_created = std::stoi(argv[3]);
  grid_parameter = std::stoi(argv[4]);
  n_total_refinements = std::stoi(argv[5]);
  
#if 1
    
  deallog.depth_console(0);
  {
    Step4<2> laplace_problem_2d(coeff_solu,
                                element_degree,
                                id_mesh_being_created,
                                grid_parameter,
                                n_total_refinements);
    laplace_problem_2d.run();
  }
  
#endif
  
  return 0;
}
