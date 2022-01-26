#ifndef SELECTED_FUNCTION_H
#define SELECTED_FUNCTION_H

template <int dim>
float returning_the_minimum_grid_size(const Triangulation<dim> &triangulation)
{

    float grid_size = 1.0;
    float minimum_grid_size = 1.0;
    
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    
    for (; cell!=endc; ++cell)                                                                  
    {
//       std::cout << "    [" << cell->active_cell_index() << "] ";
    
//         grid_size = cell->vertex(1)[0] - cell->vertex(0)[0];                                                // only for 1d        
      grid_size = cell->minimum_vertex_distance();
    
//       std::cout << "grid_size: " << grid_size << "\n";
    
      if (grid_size < minimum_grid_size)
      {
        minimum_grid_size = grid_size;
      }
    }
    
//     std::cout << "    minimum_grid_size: " << minimum_grid_size << "\n";

    return minimum_grid_size;
    
    return 0;
}


// template <int dim>
// double returning_the_average_grid_size()
// {
//     float grid_size = 1.0;
//     float total_grid_size = 0.0;
//     float average_grid_size = 0.0;
//       
//     typename Triangulation<dim>::active_cell_iterator
//     cell = triangulation.begin_active(),
//     endc = triangulation.end();
//     
//     for (; cell!=endc; ++cell)                                                                  
//     { 
//       total_grid_size += grid_size;
//     }
//     
// //     std::cout << "    total_grid_size: " << total_grid_size << "\n";
// //     std::cout << "    n_active_cells: " << triangulation.n_active_cells() << "\n";
//     
//     average_grid_size = total_grid_size/triangulation.n_active_cells();
// 
//     return average_grid_size;
// }



template <typename T>
double returning_the_l2_norm_of_a_vector(std::vector<T> input_vector)
{
    double sum_of_squares = 0;
    for(unsigned int i = 0; i < input_vector.size(); ++i)
    {
        sum_of_squares += pow(input_vector[i], 2);
    }
    
    return std::sqrt(sum_of_squares);
}


#endif
