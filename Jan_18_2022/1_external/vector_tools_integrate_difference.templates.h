// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#ifndef dealii_vector_tools_integrate_difference_templates_h
#define dealii_vector_tools_integrate_difference_templates_h

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools_integrate_difference.h>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  namespace internal
  {
    template <int dim, int spacedim, typename Number>
    struct IDScratchData
    {
      IDScratchData(const dealii::hp::MappingCollection<dim, spacedim> &mapping,
                    const dealii::hp::FECollection<dim, spacedim> &     fe,
                    const dealii::hp::QCollection<dim> &                q,
                    const UpdateFlags update_flags);

      IDScratchData(const IDScratchData &data);

      void
      resize_vectors(const unsigned int n_q_points,
                     const unsigned int n_components);

      std::vector<Vector<Number>>                           function_values;
      std::vector<std::vector<Tensor<1, spacedim, Number>>> function_grads;
      std::vector<std::vector<Tensor<2,spacedim,Number> > > function_hessians;
      std::vector<double>                                   weight_values;
      std::vector<Vector<double>>                           weight_vectors;

      std::vector<Vector<Number>>                           psi_values;
      std::vector<std::vector<Tensor<1, spacedim, Number>>> psi_grads;
      std::vector<std::vector<Tensor<2,spacedim,Number> > > psi_hessians;
      std::vector<Number>                                   psi_scalar;

      std::vector<Number>                      tmp_values;
      std::vector<Vector<Number>>              tmp_vector_values;
      std::vector<Tensor<1, spacedim, Number>> tmp_gradients;
      std::vector<SymmetricTensor<2,spacedim,Number> > tmp_hessians;
      std::vector<std::vector<Tensor<1, spacedim, Number>>>
        tmp_vector_gradients;

      dealii::hp::FEValues<dim, spacedim> x_fe_values;
    };


    template <int dim, int spacedim, typename Number>
    IDScratchData<dim, spacedim, Number>::IDScratchData(
      const dealii::hp::MappingCollection<dim, spacedim> &mapping,
      const dealii::hp::FECollection<dim, spacedim> &     fe,
      const dealii::hp::QCollection<dim> &                q,
      const UpdateFlags                                   update_flags)
      : x_fe_values(mapping, fe, q, update_flags)
    {}

    template <int dim, int spacedim, typename Number>
    IDScratchData<dim, spacedim, Number>::IDScratchData(
      const IDScratchData &data)
      : x_fe_values(data.x_fe_values.get_mapping_collection(),
                    data.x_fe_values.get_fe_collection(),
                    data.x_fe_values.get_quadrature_collection(),
                    data.x_fe_values.get_update_flags())
    {}

    template <int dim, int spacedim, typename Number>
    void
    IDScratchData<dim, spacedim, Number>::resize_vectors(
      const unsigned int n_q_points,
      const unsigned int n_components)
    {
      function_values.resize(n_q_points, Vector<Number>(n_components));
      function_grads.resize(
        n_q_points, std::vector<Tensor<1, spacedim, Number>>(n_components));
      
      function_hessians.resize (n_q_points,
                             std::vector<Tensor<2,spacedim,Number> >(n_components));      

      weight_values.resize(n_q_points);
      weight_vectors.resize(n_q_points, Vector<double>(n_components));

      psi_values.resize(n_q_points, Vector<Number>(n_components));
      psi_grads.resize(n_q_points,
                       std::vector<Tensor<1, spacedim, Number>>(n_components));
      psi_hessians.resize (n_q_points,
                        std::vector<Tensor<2,spacedim,Number> >(n_components));      
      psi_scalar.resize(n_q_points);

      tmp_values.resize(n_q_points);
      tmp_vector_values.resize(n_q_points, Vector<Number>(n_components));
      tmp_gradients.resize(n_q_points);
      tmp_hessians.resize (n_q_points);
      tmp_vector_gradients.resize(
        n_q_points, std::vector<Tensor<1, spacedim, Number>>(n_components));
    }

    template <int dim, int spacedim, typename Number>
    struct DEAL_II_DEPRECATED DeprecatedIDScratchData
    {
      DeprecatedIDScratchData(
        const dealii::hp::MappingCollection<dim, spacedim> &mapping,
        const dealii::hp::FECollection<dim, spacedim> &     fe,
        const dealii::hp::QCollection<dim> &                q,
        const UpdateFlags                                   update_flags);

      DeprecatedIDScratchData(const DeprecatedIDScratchData &data);

      void
      resize_vectors(const unsigned int n_q_points,
                     const unsigned int n_components);

      std::vector<Vector<Number>>                           function_values;
      std::vector<std::vector<Tensor<1, spacedim, Number>>> function_grads;
      std::vector<double>                                   weight_values;
      std::vector<Vector<double>>                           weight_vectors;

      std::vector<Vector<Number>>                           psi_values;                 // from this expression, we allow the value at a quadrature point to be a vector, instead           
                                                                                        // of a scalar
      std::vector<std::vector<Tensor<1, spacedim, Number>>> psi_grads;
      std::vector<Number>                                   psi_scalar;

      std::vector<double>                           tmp_values;
      std::vector<Vector<double>>                   tmp_vector_values;
      std::vector<Tensor<1, spacedim>>              tmp_gradients;
      std::vector<std::vector<Tensor<1, spacedim>>> tmp_vector_gradients;

      dealii::hp::FEValues<dim, spacedim> x_fe_values;
    };


    template <int dim, int spacedim, typename Number>
    DeprecatedIDScratchData<dim, spacedim, Number>::DeprecatedIDScratchData(
      const dealii::hp::MappingCollection<dim, spacedim> &mapping,
      const dealii::hp::FECollection<dim, spacedim> &     fe,
      const dealii::hp::QCollection<dim> &                q,
      const UpdateFlags                                   update_flags)
      : x_fe_values(mapping, fe, q, update_flags)
    {}

    template <int dim, int spacedim, typename Number>
    DeprecatedIDScratchData<dim, spacedim, Number>::DeprecatedIDScratchData(
      const DeprecatedIDScratchData &data)
      : x_fe_values(data.x_fe_values.get_mapping_collection(),
                    data.x_fe_values.get_fe_collection(),
                    data.x_fe_values.get_quadrature_collection(),
                    data.x_fe_values.get_update_flags())
    {}

    template <int dim, int spacedim, typename Number>
    void
    DeprecatedIDScratchData<dim, spacedim, Number>::resize_vectors(
      const unsigned int n_q_points,
      const unsigned int n_components)
    {
      function_values.resize(n_q_points, Vector<Number>(n_components));
      function_grads.resize(
        n_q_points, std::vector<Tensor<1, spacedim, Number>>(n_components));

      weight_values.resize(n_q_points);
      weight_vectors.resize(n_q_points, Vector<double>(n_components));

      psi_values.resize(n_q_points, Vector<Number>(n_components));
      psi_grads.resize(n_q_points,
                       std::vector<Tensor<1, spacedim, Number>>(n_components));
      psi_scalar.resize(n_q_points);

      tmp_values.resize(n_q_points);
      tmp_vector_values.resize(n_q_points, Vector<double>(n_components));
      tmp_gradients.resize(n_q_points);
      tmp_vector_gradients.resize(
        n_q_points, std::vector<Tensor<1, spacedim>>(n_components));
    }

    namespace internal
    {
      template <typename number>
      double
      mean_to_double(const number &mean_value)
      {
        return mean_value;
      }

      template <typename number>
      double
      mean_to_double(const std::complex<number> &mean_value)
      {
        // we need to return double as a norm, but mean value is a complex
        // number. Panic and return real-part while warning the user that
        // they shall never do that.
        Assert(
          false,
          ExcMessage(
            "Mean value norm is not implemented for complex-valued vectors"));
        return mean_value.real();
      }
    } // namespace internal


    // avoid compiling inner function for many vector types when we always
    // really do the same thing by putting the main work into this helper
    // function
    template <int dim, int spacedim, typename Number>
    double
    integrate_difference_inner(const Function<spacedim, Number> &exact_solution,
                               const NormType &                  norm,
                               const Function<spacedim> *        weight,
                               const UpdateFlags                 update_flags,
                               const double                      exponent,
                               const unsigned int                n_components,
                               IDScratchData<dim, spacedim, Number> &data)
    {
        
//       std::cout << "internal::integrate_difference_inner()\n";                                  // step 3
        
      const bool                             fe_is_system = (n_components != 1);
      const dealii::FEValues<dim, spacedim> &fe_values =
        data.x_fe_values.get_present_fe_values();
      const unsigned int n_q_points = fe_values.n_quadrature_points;
      
//       std::cout << "coordinates of quadrature points\n";
//       for(const Point<dim>& i:fe_values.get_quadrature_points())
//       {
//         std::cout << i << "\n";
//       }

      if (weight != nullptr)
        {
          if (weight->n_components > 1)
            weight->vector_value_list(fe_values.get_quadrature_points(),
                                      data.weight_vectors);
          else
            {
              weight->value_list(fe_values.get_quadrature_points(),
                                 data.weight_values);
              for (unsigned int k = 0; k < n_q_points; ++k)
                data.weight_vectors[k] = data.weight_values[k];
            }
        }
      else
        {
          for (unsigned int k = 0; k < n_q_points; ++k)
            data.weight_vectors[k] = 1.;
        }
        
//         std::cout << "weight vectors of the components\n";
//         for(Vector<Number>& value:data.weight_vectors)
//         {
//             std::cout << value << "\n";
//         }  

      if (update_flags & update_values)
        {
          // first compute the exact solution (vectors) at the quadrature
          // points. try to do this as efficient as possible by avoiding a
          // second virtual function call in case the function really has only
          // one component
          //
          // TODO: we have to work a bit here because the Function<dim,double>
          //   interface of the argument denoting the exact function only
          //   provides us with double/Tensor<1,dim> values, rather than
          //   with the correct data type. so evaluate into a temp
          //   object, then copy around
          if (fe_is_system)
            {
              exact_solution.vector_value_list(
                fe_values.get_quadrature_points(), data.tmp_vector_values);
              for (unsigned int i = 0; i < n_q_points; ++i)
                data.psi_values[i] = data.tmp_vector_values[i];
            }
          else
            {
              exact_solution.value_list(fe_values.get_quadrature_points(),
                                        data.tmp_values);
              for (unsigned int i = 0; i < n_q_points; ++i)
                data.psi_values[i](0) = data.tmp_values[i];
            }
            
/*            std::cout << "values of the variable at quadrature points\n";
            for(Vector<Number>& value:data.psi_values)
            {
              std::cout << value << "\n";
            }     */        

          // then subtract finite element fe_function
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < data.psi_values[q].size(); ++i)
              data.psi_values[q][i] -= data.function_values[q][i];
        }

      // Do the same for gradients, if required
      if (update_flags & update_gradients)
        {
          // try to be a little clever to avoid recursive virtual function
          // calls when calling gradient_list for functions that are really
          // scalar functions
          if (fe_is_system)
            {
              exact_solution.vector_gradient_list(
                fe_values.get_quadrature_points(), data.tmp_vector_gradients);
              for (unsigned int i = 0; i < n_q_points; ++i)
                for (unsigned int comp = 0; comp < data.psi_grads[i].size();
                     ++comp)
                  data.psi_grads[i][comp] = data.tmp_vector_gradients[i][comp];
            }
          else
            {
              exact_solution.gradient_list(fe_values.get_quadrature_points(),
                                           data.tmp_gradients);
              for (unsigned int i = 0; i < n_q_points; ++i)
              {
                data.psi_grads[i][0] = data.tmp_gradients[i];
              }
            }
            
//             std::cout << "gradients of the variable at quadrature points\n";                    // since psi_grads is evaluated from the exact solution, its values are the same
//                                                                                                 // for either computing the norm or the error of the numerical solution
//             for(std::vector<Tensor<1, spacedim, Number>>& value:data.psi_grads)
//             {
//                 for (unsigned int comp = 0; comp < n_components; ++comp)
//                 {
//                     std::cout << "comp " << comp << ", ";
//                     std::cout << value[comp] << "\n";     
//                 }
//             }
            

          // then subtract finite element function_grads. We need to be
          // careful in the codimension one case, since there we only have
          // tangential gradients in the finite element function, not the full
          // gradient. This is taken care of, by subtracting the normal
          // component of the gradient from the exact function.
          if (update_flags & update_normal_vectors)
            for (unsigned int k = 0; k < n_components; ++k)
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // compute (f.n) n
                  const typename ProductType<Number, double>::type f_dot_n =
                    data.psi_grads[q][k] * fe_values.normal_vector(q);
                  const Tensor<1, spacedim, Number> f_dot_n_times_n =
                    f_dot_n * fe_values.normal_vector(q);

                  data.psi_grads[q][k] -=
                    (data.function_grads[q][k] + f_dot_n_times_n);
                }
          else
            for (unsigned int k = 0; k < n_components; ++k)
              for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int d = 0; d < spacedim; ++d)
                  data.psi_grads[q][k][d] -= data.function_grads[q][k][d];                              // function_grads is initialized in do_integrate_difference()
        }
        
      // Do the same for second-order derivatives, if required,              ## done by the author
      if (update_flags & update_hessians)
        {  
            
//             std::cout << "update hessians\n";
            
            exact_solution.hessian_list (fe_values.get_quadrature_points(),                     // obtaining hessians of the exact solution
                                                data.tmp_hessians);
            
            for (unsigned int i=0; i<n_q_points; ++i)
            {
                data.psi_hessians[i][0] = data.tmp_hessians[i];
            }
            
/*            std::cout << "exact hessians of the variable at quadrature points\n";
            for(std::vector<Tensor<2, spacedim, Number>>& value:data.psi_hessians)
            {
                for (unsigned int comp = 0; comp < n_components; ++comp)
                {
                    std::cout << "comp " << comp << ", ";
                    std::cout << value[comp] << "\n";     
                }
            }  */          
            
            for (unsigned int k=0; k<n_components; ++k)
                for (unsigned int q=0; q<n_q_points; ++q)
                    for (unsigned int d1=0; d1<spacedim; ++d1)
                        for (unsigned int d2=0; d2<spacedim; ++d2)
                        {
                            data.psi_hessians[q][k][d1][d2] -= data.function_hessians[q][k][d1][d2];    // obtaining the difference between the exact solution and the numerical solution
                                                                                                        // function_hessians is dealt on line 1117
                        }
            
        }        

      double diff      = 0;
      Number diff_mean = 0;

      // First work on function values:
      switch (norm)
        {
          case mean:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                Number sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += data.psi_values[q](k) * data.weight_vectors[q](k);
                diff_mean += sum * fe_values.JxW(q);
              }
            break;

          case Lp_norm:
          case L1_norm:
          case W1p_norm:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += std::pow(static_cast<double>(
                                      numbers::NumberTraits<Number>::abs_square(
                                        data.psi_values[q](k))),
                                    exponent / 2.) *
                           data.weight_vectors[q](k);
                diff += sum * fe_values.JxW(q);
              }

            // Compute the root only if no derivative values are added later
            if (!(update_flags & update_gradients))
              diff = std::pow(diff, 1. / exponent);
            break;

          case L2_norm:
          case H1_norm:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                  {
                    sum += numbers::NumberTraits<Number>::abs_square(data.psi_values[q](k)) *
                           data.weight_vectors[q](k);
                  }
                diff += sum * fe_values.JxW(q);
              }
            // Compute the root only, if no derivative values are added later
            if (norm == L2_norm)
              diff = std::sqrt(diff);
            break;

          case Linfty_norm:
          case W1infty_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              for (unsigned int k = 0; k < n_components; ++k)
                if (data.weight_vectors[q](k) != 0)
                  diff = std::max(diff,
                                  double(std::abs(data.psi_values[q](k) *
                                                  data.weight_vectors[q](k))));
            break;

          case H1_seminorm:
          case H1_seminorm_velocity_00:
          case H1_seminorm_velocity_01:
          case H1_seminorm_velocity_10:
          case H1_seminorm_velocity_11:
              
              
              
          case H2_divnorm:
          case H2_seminorm:
          case H2_seminorm_00:
          case H2_seminorm_01:
          case H2_seminorm_10:
          case H2_seminorm_11:
          case Hdiv_seminorm:
          case W1p_seminorm:
          case W1infty_seminorm:
            // function values are not used for these norms
            break;

          default:
            Assert(false, ExcNotImplemented());
            break;
        }

      // Now compute terms depending on first-order derivatives:
      switch (norm)
        {
          case W1p_seminorm:
          case W1p_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += std::pow(data.psi_grads[q][k].norm_square(),
                                    exponent / 2.) *
                           data.weight_vectors[q](k);
                diff += sum * fe_values.JxW(q);
              }
            diff = std::pow(diff, 1. / exponent);
            break;

          case H1_seminorm:
          case H1_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                {
                  if (data.weight_vectors[q](k) != 0)
                  {
                    sum += data.psi_grads[q][k].norm_square() *        // norm_square() is a function for a tensor, returning the sum of the absolute squares of all entries
                           data.weight_vectors[q](k);                  // this is an "extended" version of abs_square()
                                                                       // note that, psi_grads[q] has 1 component for the standard FEM, and 2 components for the mixed FEM
                  }
                }
                diff += sum * fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;
            
          case H1_seminorm_velocity_00:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                diff += numbers::NumberTraits<Number>::abs_square(data.psi_grads[q][0][0])* fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;              
          case H1_seminorm_velocity_01:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                diff += numbers::NumberTraits<Number>::abs_square(data.psi_grads[q][0][1])* fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;              
          case H1_seminorm_velocity_10:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                diff += numbers::NumberTraits<Number>::abs_square(data.psi_grads[q][1][0])* fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;              
          case H1_seminorm_velocity_11:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                diff += numbers::NumberTraits<Number>::abs_square(data.psi_grads[q][1][1])* fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;              
            

          case Hdiv_seminorm:
              
//             std::cout << "Hdiv_seminorm\n";
              
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                unsigned int idx = 0;
                if (weight != nullptr)
                  for (; idx < n_components; ++idx)
                    if (data.weight_vectors[0](idx) > 0)
                      break;

                Assert(
                  n_components >= idx + dim,
                  ExcMessage(
                    "You can only ask for the Hdiv norm for a finite element "
                    "with at least 'dim' components. In that case, this function "
                    "will find the index of the first non-zero weight and take "
                    "the divergence of the 'dim' components that follow it."));

                Number sum = 0;
                // take the trace of the derivatives scaled by the weight and
                // square it
                
//                 std::cout << "idx: " << idx << ", ";
                
                for (unsigned int k = idx; k < idx + dim; ++k)
                {    
//                   std::cout << "k:" << k << ", ";
//                   std::cout << "k-idx:" << k-idx << ", ";
            
                  if (data.weight_vectors[q](k) != 0)
                  {
                    sum += data.psi_grads[q][k][k - idx] *                      // for each component, we only count the diagonal part
                           std::sqrt(data.weight_vectors[q](k));
                  }  
                }
//                 std::cout << "\n";
                
                diff += numbers::NumberTraits<Number>::abs_square(sum) *
                        fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;

          case W1infty_seminorm:
          case W1infty_norm:
            {
              double t = 0;
              for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    for (unsigned int d = 0; d < dim; ++d)
                      t = std::max(t,
                                   double(std::abs(data.psi_grads[q][k][d]) *
                                          data.weight_vectors[q](k)));

              // then add seminorm to norm if that had previously been
              // computed
              diff += t;
            }
            break;
            
          case H2_divnorm:
          case H2_seminorm:
          case H2_seminorm_00:
          case H2_seminorm_01:
          case H2_seminorm_10:
          case H2_seminorm_11:
          {
            break;
          }   
        
          default:
            break;
        }
        
        // Now compute terms depending on 2nd-order derivatives, done by the author
        switch (norm)
        {
          
//           std::cout << "\n";
//           std::cout << "n_components: " << n_components << "\n";
            
          case H2_divnorm:
//             std::cout << "H2_divnorm\n";
              
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                
              Number value_trace = 0;
                
              value_trace = trace(data.psi_hessians[q][0]);
              
              diff += numbers::NumberTraits<Number>::abs_square(value_trace) * fe_values.JxW(q);
              
//               Number sum = 0;                                              // an alternative of the above 'value_trace'
//               for (unsigned int k = 0; k < dim; ++k)
//               {    
//                 sum += data.psi_hessians[q][0][k][k];                      // counting the diagonal part, only one component
//                                                                            // analogous to Hdiv_seminorm
//               }              
//               std::cout << "difference between 'sum' and trace" << sum - value_trace << "\n";
//               
            }
            diff = std::sqrt(diff);
            break;
            
            
          case H2_seminorm:
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              diff += data.psi_hessians[q][0].norm_square() * fe_values.JxW(q);     // note: this type of norm is only for the standard FEM
            }                                                                       // we only have one component for the standard FEM
            diff = std::sqrt(diff);
            break;
            
          case H2_seminorm_00:
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              diff += numbers::NumberTraits<Number>::abs_square(data.psi_hessians[q][0][0][0]) * fe_values.JxW(q);
            }
            diff = std::sqrt(diff);
            break;
          case H2_seminorm_01:
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              diff += numbers::NumberTraits<Number>::abs_square(data.psi_hessians[q][0][0][1]) * fe_values.JxW(q);
            }
            diff = std::sqrt(diff);
            break;
          case H2_seminorm_10:
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              diff += numbers::NumberTraits<Number>::abs_square(data.psi_hessians[q][0][1][0]) * fe_values.JxW(q);
            }
            diff = std::sqrt(diff);
            break;
          case H2_seminorm_11:
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              diff += numbers::NumberTraits<Number>::abs_square(data.psi_hessians[q][0][1][1]) * fe_values.JxW(q);
            }
            diff = std::sqrt(diff);
            break;
          default:
            break;
        }        

      if (norm == mean)
        diff = internal::mean_to_double(diff_mean);

      // append result of this cell to the end of the vector
      AssertIsFinite(diff);
      return diff;
    }

    template <int dim, int spacedim, typename Number>
    DEAL_II_DEPRECATED
      typename std::enable_if<!std::is_same<Number, double>::value,
                              double>::type
      integrate_difference_inner(
        const Function<spacedim> &                      exact_solution,
        const NormType &                                norm,
        const Function<spacedim> *                      weight,
        const UpdateFlags                               update_flags,
        const double                                    exponent,
        const unsigned int                              n_components,
        DeprecatedIDScratchData<dim, spacedim, Number> &data)
    {
      const bool                             fe_is_system = (n_components != 1);
      const dealii::FEValues<dim, spacedim> &fe_values =
        data.x_fe_values.get_present_fe_values();
      const unsigned int n_q_points = fe_values.n_quadrature_points;

      if (weight != nullptr)
        {
          if (weight->n_components > 1)
            weight->vector_value_list(fe_values.get_quadrature_points(),
                                      data.weight_vectors);
          else
            {
              weight->value_list(fe_values.get_quadrature_points(),
                                 data.weight_values);
              for (unsigned int k = 0; k < n_q_points; ++k)
                data.weight_vectors[k] = data.weight_values[k];
            }
        }
      else
        {
          for (unsigned int k = 0; k < n_q_points; ++k)
            data.weight_vectors[k] = 1.;
        }


      if (update_flags & update_values)
        {
          // first compute the exact solution (vectors) at the quadrature
          // points. try to do this as efficient as possible by avoiding a
          // second virtual function call in case the function really has only
          // one component
          //
          // TODO: we have to work a bit here because the Function<dim,double>
          //   interface of the argument denoting the exact function only
          //   provides us with double/Tensor<1,dim> values, rather than
          //   with the correct data type. so evaluate into a temp
          //   object, then copy around
          if (fe_is_system)
            {
              exact_solution.vector_value_list(
                fe_values.get_quadrature_points(), data.tmp_vector_values);
              for (unsigned int i = 0; i < n_q_points; ++i)
                data.psi_values[i] = data.tmp_vector_values[i];
            }
          else
            {
              exact_solution.value_list(fe_values.get_quadrature_points(),
                                        data.tmp_values);
              for (unsigned int i = 0; i < n_q_points; ++i)
                data.psi_values[i](0) = data.tmp_values[i];
            }

          // then subtract finite element fe_function
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < data.psi_values[q].size(); ++i)
              data.psi_values[q][i] -= data.function_values[q][i];
        }

      // Do the same for gradients, if required
      if (update_flags & update_gradients)
        {
          // try to be a little clever to avoid recursive virtual function
          // calls when calling gradient_list for functions that are really
          // scalar functions
          if (fe_is_system)
            {
              exact_solution.vector_gradient_list(
                fe_values.get_quadrature_points(), data.tmp_vector_gradients);
              for (unsigned int i = 0; i < n_q_points; ++i)
                for (unsigned int comp = 0; comp < data.psi_grads[i].size();
                     ++comp)
                  data.psi_grads[i][comp] = data.tmp_vector_gradients[i][comp];
            }
          else
            {
              exact_solution.gradient_list(fe_values.get_quadrature_points(),
                                           data.tmp_gradients);
              for (unsigned int i = 0; i < n_q_points; ++i)
                data.psi_grads[i][0] = data.tmp_gradients[i];
            }

          // then subtract finite element function_grads. We need to be
          // careful in the codimension one case, since there we only have
          // tangential gradients in the finite element function, not the full
          // gradient. This is taken care of, by subtracting the normal
          // component of the gradient from the exact function.
          if (update_flags & update_normal_vectors)
            for (unsigned int k = 0; k < n_components; ++k)
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // compute (f.n) n
                  const typename ProductType<Number, double>::type f_dot_n =
                    data.psi_grads[q][k] * fe_values.normal_vector(q);
                  const Tensor<1, spacedim, Number> f_dot_n_times_n =
                    f_dot_n * fe_values.normal_vector(q);

                  data.psi_grads[q][k] -=
                    (data.function_grads[q][k] + f_dot_n_times_n);
                }
          else
            for (unsigned int k = 0; k < n_components; ++k)
              for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int d = 0; d < spacedim; ++d)
                  data.psi_grads[q][k][d] -= data.function_grads[q][k][d];
        }

      double diff      = 0;
      Number diff_mean = 0;

      // First work on function values:
      switch (norm)
        {
          case mean:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                Number sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += data.psi_values[q](k) * data.weight_vectors[q](k);
                diff_mean += sum * fe_values.JxW(q);
              }
            break;

          case Lp_norm:
          case L1_norm:
          case W1p_norm:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += std::pow(static_cast<double>(
                                      numbers::NumberTraits<Number>::abs_square(
                                        data.psi_values[q](k))),
                                    exponent / 2.) *
                           data.weight_vectors[q](k);
                diff += sum * fe_values.JxW(q);
              }

            // Compute the root only if no derivative values are added later
            if (!(update_flags & update_gradients))
              diff = std::pow(diff, 1. / exponent);
            break;

          case L2_norm:
          case H1_norm:
            // Compute values in quadrature points and integrate
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += numbers::NumberTraits<Number>::abs_square(
                             data.psi_values[q](k)) *
                           data.weight_vectors[q](k);
                diff += sum * fe_values.JxW(q);
              }
            // Compute the root only, if no derivative values are added later
            if (norm == L2_norm)
              diff = std::sqrt(diff);
            break;

          case Linfty_norm:
          case W1infty_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              for (unsigned int k = 0; k < n_components; ++k)
                if (data.weight_vectors[q](k) != 0)
                  diff = std::max(diff,
                                  double(std::abs(data.psi_values[q](k) *
                                                  data.weight_vectors[q](k))));
            break;

          case H1_seminorm:
          case H1_seminorm_velocity_00:
          case H1_seminorm_velocity_01:
          case H1_seminorm_velocity_10:
          case H1_seminorm_velocity_11:
              
          case Hdiv_seminorm:
          case W1p_seminorm:
          case W1infty_seminorm:
            // function values are not used for these norms
            break;

          default:
            Assert(false, ExcNotImplemented());
            break;
        }

      // Now compute terms depending on derivatives:
      switch (norm)
        {
          case W1p_seminorm:
          case W1p_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += std::pow(data.psi_grads[q][k].norm_square(),
                                    exponent / 2.) *
                           data.weight_vectors[q](k);
                diff += sum * fe_values.JxW(q);
              }
            diff = std::pow(diff, 1. / exponent);
            break;

          case H1_seminorm:
          case H1_norm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n_components; ++k)
                {
                  if (data.weight_vectors[q](k) != 0)
                  {
                    sum += data.psi_grads[q][k].norm_square() *
                           data.weight_vectors[q](k);
                  }
                }
                diff += sum * fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;

          case Hdiv_seminorm:
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                unsigned int idx = 0;
                if (weight != nullptr)
                  for (; idx < n_components; ++idx)
                    if (data.weight_vectors[0](idx) > 0)
                      break;

                Assert(
                  n_components >= idx + dim,
                  ExcMessage(
                    "You can only ask for the Hdiv norm for a finite element "
                    "with at least 'dim' components. In that case, this function "
                    "will find the index of the first non-zero weight and take "
                    "the divergence of the 'dim' components that follow it."));

                Number sum = 0;
                // take the trace of the derivatives scaled by the weight and
                // square it
                for (unsigned int k = idx; k < idx + dim; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    sum += data.psi_grads[q][k][k - idx] *
                           std::sqrt(data.weight_vectors[q](k));
                diff += numbers::NumberTraits<Number>::abs_square(sum) *
                        fe_values.JxW(q);
              }
            diff = std::sqrt(diff);
            break;

          case W1infty_seminorm:
          case W1infty_norm:
            {
              double t = 0;
              for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int k = 0; k < n_components; ++k)
                  if (data.weight_vectors[q](k) != 0)
                    for (unsigned int d = 0; d < dim; ++d)
                      t = std::max(t,
                                   double(std::abs(data.psi_grads[q][k][d]) *
                                          data.weight_vectors[q](k)));

              // then add seminorm to norm if that had previously been
              // computed
              diff += t;
            }
            break;
          default:
            break;
        }

      if (norm == mean)
        diff = internal::mean_to_double(diff_mean);

      // append result of this cell to the end of the vector
      AssertIsFinite(diff);
      return diff;
    }

    template <int dim,
              class InVector,
              class OutVector,
              typename DoFHandlerType,
              int spacedim>
    static void
    do_integrate_difference(
      const dealii::hp::MappingCollection<dim, spacedim> &     mapping,
      const DoFHandlerType &                                   dof,
      const InVector &                                         fe_function,
      const Function<spacedim, typename InVector::value_type> &exact_solution,
      OutVector &                                              difference,
      const dealii::hp::QCollection<dim> &                     q,
      const NormType &                                         norm,
      const Function<spacedim> *                               weight,
      const double                                             exponent_1)
    {
        
//       std::cout << "internal::do_integrate_difference()\n";                                     // step 2
        
      using Number = typename InVector::value_type;
      // we mark the "exponent" parameter to this function "const" since it is
      // strictly incoming, but we need to set it to something different later
      // on, if necessary, so have a read-write version of it:
      double exponent = exponent_1;

      const unsigned int n_components = dof.get_fe(0).n_components();
      
//       std::cout << "n_components: " << n_components << "\n";

      Assert(exact_solution.n_components == n_components,
             ExcDimensionMismatch(exact_solution.n_components, n_components));

      if (weight != nullptr)
        {
          Assert((weight->n_components == 1) ||
                   (weight->n_components == n_components),
                 ExcDimensionMismatch(weight->n_components, n_components));
        }

      difference.reinit(dof.get_triangulation().n_active_cells());

      switch (norm)
        {
          case L2_norm:
          case H1_seminorm:
          case H1_seminorm_velocity_00:
          case H1_seminorm_velocity_01:
          case H1_seminorm_velocity_10:
          case H1_seminorm_velocity_11:
              
          case H2_divnorm:
          case H2_seminorm:
          case H2_seminorm_00:
          case H2_seminorm_01:
          case H2_seminorm_10:
          case H2_seminorm_11:
          case H1_norm:
          case Hdiv_seminorm:
            exponent = 2.;
            break;

          case L1_norm:
            exponent = 1.;
            break;

          default:
            break;
        }

      UpdateFlags update_flags =
        UpdateFlags(update_quadrature_points | update_JxW_values);
      switch (norm)
        {
            
          case H2_divnorm:
          case H2_seminorm:
          case H2_seminorm_00:
          case H2_seminorm_01:
          case H2_seminorm_10:
          case H2_seminorm_11:
            update_flags |= UpdateFlags (update_hessians);
            break;
            
          case H1_seminorm:
          case H1_seminorm_velocity_00:
          case H1_seminorm_velocity_01:
          case H1_seminorm_velocity_10:
          case H1_seminorm_velocity_11:
              
          case Hdiv_seminorm:
          case W1p_seminorm:
          case W1infty_seminorm:
            update_flags |= UpdateFlags(update_gradients);
            if (spacedim == dim + 1)
              update_flags |= UpdateFlags(update_normal_vectors);

            break;

          case H1_norm:
          case W1p_norm:
          case W1infty_norm:
            update_flags |= UpdateFlags(update_gradients);
            if (spacedim == dim + 1)
              update_flags |= UpdateFlags(update_normal_vectors);
            DEAL_II_FALLTHROUGH;

          default:
            update_flags |= UpdateFlags(update_values);
            break;
        }

      const dealii::hp::FECollection<dim, spacedim> &fe_collection =
        dof.get_fe_collection();
      IDScratchData<dim, spacedim, Number> data(mapping,
                                                fe_collection,
                                                q,
                                                update_flags);

      // loop over all cells
      for (const auto &cell : dof.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // initialize for this cell
            data.x_fe_values.reinit(cell);

            const dealii::FEValues<dim, spacedim> &fe_values =
              data.x_fe_values.get_present_fe_values();
            const unsigned int n_q_points = fe_values.n_quadrature_points;
            data.resize_vectors(n_q_points, n_components);

            if (update_flags & update_values)
              fe_values.get_function_values(fe_function, data.function_values);
            if (update_flags & update_gradients)                                        // initializing gradients of the numerical solution of the variable
              fe_values.get_function_gradients(fe_function,
                                               data.function_grads);
            if (update_flags & update_hessians)
            {
              fe_values.get_function_hessians (fe_function, data.function_hessians);   
            }
            
/*            std::cout << "numeric hessians of the variable at quadrature points\n";
            for(std::vector<Tensor<2, spacedim, Number>>& value:data.function_hessians)
            {
                for (unsigned int comp = 0; comp < n_components; ++comp)
                {
                    std::cout << "comp " << comp << ", ";
                    std::cout << value[comp] << "\n";     
                }
            } */            
            
//             std::cout << "cell " << cell->active_cell_index() << ", ";
            
            difference(cell->active_cell_index()) =
              integrate_difference_inner<dim, spacedim, Number>(exact_solution,
                                                                norm,
                                                                weight,
                                                                update_flags,
                                                                exponent,
                                                                n_components,
                                                                data);
          }
        else
          // the cell is a ghost cell or is artificial. write a zero into the
          // corresponding value of the returned vector
          difference(cell->active_cell_index()) = 0;
    }

    template <int dim,
              class InVector,
              class OutVector,
              typename DoFHandlerType,
              int spacedim>
    DEAL_II_DEPRECATED static typename std::enable_if<
      !std::is_same<typename InVector::value_type, double>::value>::type
    do_integrate_difference(
      const dealii::hp::MappingCollection<dim, spacedim> &mapping,
      const DoFHandlerType &                              dof,
      const InVector &                                    fe_function,
      const Function<spacedim> &                          exact_solution,
      OutVector &                                         difference,
      const dealii::hp::QCollection<dim> &                q,
      const NormType &                                    norm,
      const Function<spacedim> *                          weight,
      const double                                        exponent_1)
    {
      using Number = typename InVector::value_type;
      // we mark the "exponent" parameter to this function "const" since it is
      // strictly incoming, but we need to set it to something different later
      // on, if necessary, so have a read-write version of it:
      double exponent = exponent_1;

      const unsigned int n_components = dof.get_fe(0).n_components();

      Assert(exact_solution.n_components == n_components,
             ExcDimensionMismatch(exact_solution.n_components, n_components));

      if (weight != nullptr)
        {
          Assert((weight->n_components == 1) ||
                   (weight->n_components == n_components),
                 ExcDimensionMismatch(weight->n_components, n_components));
        }

      difference.reinit(dof.get_triangulation().n_active_cells());

      switch (norm)
        {
          case L2_norm:
          case H1_seminorm:
          case H1_norm:
          case Hdiv_seminorm:
            exponent = 2.;
            break;

          case L1_norm:
            exponent = 1.;
            break;

          default:
            break;
        }

      UpdateFlags update_flags =
        UpdateFlags(update_quadrature_points | update_JxW_values);
      switch (norm)
        {
          case H1_seminorm:
          case Hdiv_seminorm:
          case W1p_seminorm:
          case W1infty_seminorm:
            update_flags |= UpdateFlags(update_gradients);
            if (spacedim == dim + 1)
              update_flags |= UpdateFlags(update_normal_vectors);

            break;

          case H1_norm:
          case W1p_norm:
          case W1infty_norm:
            update_flags |= UpdateFlags(update_gradients);
            if (spacedim == dim + 1)
              update_flags |= UpdateFlags(update_normal_vectors);
            DEAL_II_FALLTHROUGH;

          default:
            update_flags |= UpdateFlags(update_values);
            break;
        }

      const dealii::hp::FECollection<dim, spacedim> &fe_collection =
        dof.get_fe_collection();
      DeprecatedIDScratchData<dim, spacedim, Number> data(mapping,
                                                          fe_collection,
                                                          q,
                                                          update_flags);

      // loop over all cells
      for (const auto &cell : dof.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // initialize for this cell
            data.x_fe_values.reinit(cell);

            const dealii::FEValues<dim, spacedim> &fe_values =
              data.x_fe_values.get_present_fe_values();
            const unsigned int n_q_points = fe_values.n_quadrature_points;
            data.resize_vectors(n_q_points, n_components);

            if (update_flags & update_values)
              fe_values.get_function_values(fe_function, data.function_values);
            if (update_flags & update_gradients)
              fe_values.get_function_gradients(fe_function,
                                               data.function_grads);

            difference(cell->active_cell_index()) =
              integrate_difference_inner<dim, spacedim, Number>(exact_solution,
                                                                norm,
                                                                weight,
                                                                update_flags,
                                                                exponent,
                                                                n_components,
                                                                data);
          }
        else
          // the cell is a ghost cell or is artificial. write a zero into the
          // corresponding value of the returned vector
          difference(cell->active_cell_index()) = 0;
    }

  } // namespace internal

  template <int dim, class InVector, class OutVector, int spacedim>
  void
  integrate_difference(
    const Mapping<dim, spacedim> &                           mapping,
    const DoFHandler<dim, spacedim> &                        dof,
    const InVector &                                         fe_function,
    const Function<spacedim, typename InVector::value_type> &exact_solution,
    OutVector &                                              difference,
    const Quadrature<dim> &                                  q,
    const NormType &                                         norm,
    const Function<spacedim> *                               weight,
    const double                                             exponent)
  {
      
//     std::cout << "internal::integrate_difference(), with mapping\n";
      
    internal::do_integrate_difference(hp::MappingCollection<dim, spacedim>(
                                        mapping),
                                      dof,
                                      fe_function,
                                      exact_solution,
                                      difference,
                                      hp::QCollection<dim>(q),
                                      norm,
                                      weight,
                                      exponent);
  }

  template <int dim, class InVector, class OutVector, int spacedim>
  DEAL_II_DEPRECATED typename std::enable_if<
    !std::is_same<typename InVector::value_type, double>::value>::type
  integrate_difference(const Mapping<dim, spacedim> &   mapping,
                       const DoFHandler<dim, spacedim> &dof,
                       const InVector &                 fe_function,
                       const Function<spacedim> &       exact_solution,
                       OutVector &                      difference,
                       const Quadrature<dim> &          q,
                       const NormType &                 norm,
                       const Function<spacedim> *       weight,
                       const double                     exponent)
  {
    internal::do_integrate_difference(hp::MappingCollection<dim, spacedim>(
                                        mapping),
                                      dof,
                                      fe_function,
                                      exact_solution,
                                      difference,
                                      hp::QCollection<dim>(q),
                                      norm,
                                      weight,
                                      exponent);
  }


  template <int dim, class InVector, class OutVector, int spacedim>
  void
  integrate_difference(
    const DoFHandler<dim, spacedim> &                        dof,
    const InVector &                                         fe_function,
    const Function<spacedim, typename InVector::value_type> &exact_solution,
    OutVector &                                              difference,
    const Quadrature<dim> &                                  q,
    const NormType &                                         norm,
    const Function<spacedim> *                               weight,                        // receiving the id of the component of interest
    const double                                             exponent)
  {
      
//     std::cout << "internal::integrate_difference()\n";                                  // step 1
      
    internal::do_integrate_difference(
      hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
      dof,
      fe_function,
      exact_solution,
      difference,
      hp::QCollection<dim>(q),
      norm,
      weight,
      exponent);
  }


  template <int dim, class InVector, class OutVector, int spacedim>
  DEAL_II_DEPRECATED typename std::enable_if<
    !std::is_same<typename InVector::value_type, double>::value>::type
  integrate_difference(const DoFHandler<dim, spacedim> &dof,
                       const InVector &                 fe_function,
                       const Function<spacedim> &       exact_solution,
                       OutVector &                      difference,
                       const Quadrature<dim> &          q,
                       const NormType &                 norm,
                       const Function<spacedim> *       weight,
                       const double                     exponent)
  {
    internal::do_integrate_difference(
      hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
      dof,
      fe_function,
      exact_solution,
      difference,
      hp::QCollection<dim>(q),
      norm,
      weight,
      exponent);
  }



  template <int dim, class InVector, class OutVector, int spacedim>
  void
  integrate_difference(
    const dealii::hp::MappingCollection<dim, spacedim> &     mapping,
    const dealii::hp::DoFHandler<dim, spacedim> &            dof,
    const InVector &                                         fe_function,
    const Function<spacedim, typename InVector::value_type> &exact_solution,
    OutVector &                                              difference,
    const dealii::hp::QCollection<dim> &                     q,
    const NormType &                                         norm,
    const Function<spacedim> *                               weight,
    const double                                             exponent)
  {
    internal::do_integrate_difference(hp::MappingCollection<dim, spacedim>(
                                        mapping),
                                      dof,
                                      fe_function,
                                      exact_solution,
                                      difference,
                                      q,
                                      norm,
                                      weight,
                                      exponent);
  }

  template <int dim, class InVector, class OutVector, int spacedim>
  DEAL_II_DEPRECATED typename std::enable_if<
    !std::is_same<typename InVector::value_type, double>::value>::type
  integrate_difference(
    const dealii::hp::MappingCollection<dim, spacedim> &mapping,
    const dealii::hp::DoFHandler<dim, spacedim> &       dof,
    const InVector &                                    fe_function,
    const Function<spacedim> &                          exact_solution,
    OutVector &                                         difference,
    const dealii::hp::QCollection<dim> &                q,
    const NormType &                                    norm,
    const Function<spacedim> *                          weight,
    const double                                        exponent)
  {
    internal::do_integrate_difference(hp::MappingCollection<dim, spacedim>(
                                        mapping),
                                      dof,
                                      fe_function,
                                      exact_solution,
                                      difference,
                                      q,
                                      norm,
                                      weight,
                                      exponent);
  }


  template <int dim, class InVector, class OutVector, int spacedim>
  void
  integrate_difference(
    const dealii::hp::DoFHandler<dim, spacedim> &            dof,
    const InVector &                                         fe_function,
    const Function<spacedim, typename InVector::value_type> &exact_solution,
    OutVector &                                              difference,
    const dealii::hp::QCollection<dim> &                     q,
    const NormType &                                         norm,
    const Function<spacedim> *                               weight,
    const double                                             exponent)
  {
    internal::do_integrate_difference(
      hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
      dof,
      fe_function,
      exact_solution,
      difference,
      q,
      norm,
      weight,
      exponent);
  }

  template <int dim, class InVector, class OutVector, int spacedim>
  DEAL_II_DEPRECATED typename std::enable_if<
    !std::is_same<typename InVector::value_type, double>::value>::type
  integrate_difference(const dealii::hp::DoFHandler<dim, spacedim> &dof,
                       const InVector &                             fe_function,
                       const Function<spacedim> &          exact_solution,
                       OutVector &                         difference,
                       const dealii::hp::QCollection<dim> &q,
                       const NormType &                    norm,
                       const Function<spacedim> *          weight,
                       const double                        exponent)
  {
    internal::do_integrate_difference(
      hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
      dof,
      fe_function,
      exact_solution,
      difference,
      q,
      norm,
      weight,
      exponent);
  }

  template <int dim, int spacedim, class InVector>
  double
  compute_global_error(const Triangulation<dim, spacedim> &tria,
                       const InVector &                    cellwise_error,
                       const NormType &                    norm,
                       const double                        exponent)
  {
    Assert(cellwise_error.size() == tria.n_active_cells(),
           ExcMessage("input vector cell_error has invalid size!"));
#ifdef DEBUG
    {
      // check that off-processor entries are zero. Otherwise we will compute
      // wrong results below!
      typename InVector::size_type                                i = 0;
      typename Triangulation<dim, spacedim>::active_cell_iterator it =
        tria.begin_active();
      for (; i < cellwise_error.size(); ++i, ++it)
        if (!it->is_locally_owned())
          Assert(
            std::fabs(cellwise_error[i]) < 1e-20,
            ExcMessage(
              "cellwise_error of cells that are not locally owned need to be zero!"));
    }
#endif

    MPI_Comm comm = MPI_COMM_SELF;
#ifdef DEAL_II_WITH_MPI
    if (const parallel::TriangulationBase<dim, spacedim> *ptria =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &tria))
      comm = ptria->get_communicator();
#endif

    switch (norm)
      {
        case L2_norm:
        case H1_seminorm:
        case H1_seminorm_velocity_00:
        case H1_seminorm_velocity_01:
        case H1_seminorm_velocity_10:
        case H1_seminorm_velocity_11:            
            
            
        case H2_divnorm:
        case H2_seminorm:
        case H2_seminorm_00:
        case H2_seminorm_01:
        case H2_seminorm_10:
        case H2_seminorm_11:
        case H1_norm:
        case Hdiv_seminorm:
          {
            const double local = cellwise_error.l2_norm();
            return std::sqrt(Utilities::MPI::sum(local * local, comm));
          }

        case L1_norm:
          {
            const double local = cellwise_error.l1_norm();
            return Utilities::MPI::sum(local, comm);
          }

        case Linfty_norm:
        case W1infty_seminorm:
          {
            const double local = cellwise_error.linfty_norm();
            return Utilities::MPI::max(local, comm);
          }

        case W1infty_norm:
          {
            AssertThrow(false,
                        ExcMessage(
                          "compute_global_error() is impossible for "
                          "the W1infty_norm. See the documentation for "
                          "NormType::W1infty_norm for more information."));
            return std::numeric_limits<double>::infinity();
          }

        case mean:
          {
            // Note: mean is defined as int_\Omega f = sum_K \int_K f, so we
            // need the sum of the cellwise errors not the Euclidean mean
            // value that is returned by Vector<>::mean_value().
            const double local =
              cellwise_error.mean_value() * cellwise_error.size();
            return Utilities::MPI::sum(local, comm);
          }

        case Lp_norm:
        case W1p_norm:
        case W1p_seminorm:
          {
            double                       local = 0;
            typename InVector::size_type i;
            typename Triangulation<dim, spacedim>::active_cell_iterator it =
              tria.begin_active();
            for (i = 0; i < cellwise_error.size(); ++i, ++it)
              if (it->is_locally_owned())
                local += std::pow(cellwise_error[i], exponent);

            return std::pow(Utilities::MPI::sum(local, comm), 1. / exponent);
          }

        default:
          AssertThrow(false, ExcNotImplemented());
          break;
      }
    return 0.0;
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif // dealii_vector_tools_integrate_difference_templates_h
