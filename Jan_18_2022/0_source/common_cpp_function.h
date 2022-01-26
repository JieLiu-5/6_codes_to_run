
#ifndef COMMON_CPP_FUNCTION_H
#define COMMON_CPP_FUNCTION_H

#include <map>

template <typename T>
void print_vector_vertically(std::vector<T> &vector)
{
  for (unsigned int i=0; i < vector.size(); ++i)
  {
    std::cout << "  [" << i << "]: " << vector[i] << "\n";
  }
}

// void print_vector_vertically_test(std::vector<double> &vector,                                           // failed, Sep. 14, 2021
//                                   const std::ios_base obj_floatfield)                               // trying to use std::fixed as an argument
// {
//   std::cout << std::fixed << "print_vector_vertically_test\n";
//   
//   for (unsigned int i=0; i<vector.size();++i)
//   {
//     std::cout << "    [" << i << "]: " << vector[i] << "\n";
//   }  
// }

template <>
void print_vector_vertically<double>(std::vector<double> &vector)
{
  for (unsigned int i=0; i<vector.size();++i)
  {
    std::cout << "    [" << i << "]: " << vector[i] << "\n";
  }
}

template <typename T>
void print_vector_horizontally(const std::vector<T> &vector)
{
  if(vector.size() > 0)
  {
    for (unsigned int i = 0; i < vector.size(); ++i)
    {
        std::cout << vector[i] << " ";
    }
    std::cout << "\n";
  }
  
}

template <typename T>
void print_vector_horizontally(const std::vector<T> &vector,
                               unsigned int integer_for_precision)
{
  for (unsigned int i = 0; i < vector.size(); ++i)
  {
    std::cout << std::setprecision(integer_for_precision) << vector[i] << " ";
  }
  std::cout << "\n";
}


template <typename T>
void print_vector_in_vector(std::vector<std::vector<T> > &tensor,
                            unsigned int integer_for_precision)
{
  unsigned int row_no = tensor.size();
  
  if(row_no>0)
  {
    for (unsigned int k=0; k<row_no; ++k)
    {
      std::cout << "    [" << k << "] ";
      
      if(tensor[k].empty())
      {
        std::cout << "(empty)\n";
          
//         break;
      }else
      {
        for (unsigned int j=0; j<tensor[k].size(); ++j)
        {
          std::cout << std::setprecision(integer_for_precision)
               << tensor[k][j] << " ";          // << setprecision(2) 
        }          
        std::cout << "\n";
      }
    }
  }else
  {
//     std::cout << "  number of rows smaller than 1\n";
  }
}

void print_an_array(float obj_array[][3])
{
//     std::cout << "printing an array\n";
    
//     unsigned int rows = sizeof(obj_array) / 3;                       // not successful
    unsigned int cols = sizeof obj_array[0] / sizeof obj_array[0][0];
    
//     std::cout << "rows: " << rows << "\n";
//     std::cout << "cols: " << cols << "\n";
    
    for (unsigned int i = 0; i < 5; ++i)
    {
        for(unsigned int j = 0; j < cols; ++j)
        {
            std::cout << obj_array[i][j] << " ";
        }
        std::cout << "\n";
    }
}


template <typename T>
void print_map(std::map<T, double>& obj_map)                    // 
{
  for (auto& x: obj_map) 
  {
    std::cout << x.first << ": " << x.second << '\n';
  }    
}


void printPattern(int radius)
{
  // dist represents distance to the center 
  float dist; 
  
  // for horizontal movement 
  for (int i = 0; i <= 2 * radius; i++) { 
  
    // for vertical movement 
    for (int j = 0; j <= 2 * radius; j++) { 
      dist = sqrt((i - radius) * (i - radius) +  
                  (j - radius) * (j - radius)); 
  
      // dist should be in the range (radius - 0.5) 
      // and (radius + 0.5) to print stars(*) 
      if (dist > radius - 0.5 && dist < radius + 0.5)  
        std::cout << "*"; 
      else 
        std::cout << " ";       
    } 
  
    std::cout << "\n"; 
  } 
}


template <typename T>
void save_vector_of_numbers_to_a_file(std::string& obj_string,
                                           std::string& obj_tag,
                                           std::vector<T> &obj_vector,
                                           const std::ofstream::openmode& ofstream_openmode)
{
  unsigned int row_no = obj_vector.size();

  std::ofstream fid;
  fid.open(obj_string, ofstream_openmode);
  
  fid << obj_tag << "\n";

  for (unsigned int k=0; k<row_no; ++k)
  {
    fid << std::setprecision(3) << obj_vector[k] << "\n";
  }
  fid.close();
  fid.clear();
}


template <typename T>
void save_vector_of_numbers_to_a_file_without_inserting_a_tag(std::string& obj_string,
                                                    std::vector<T> &obj_vector)
{
  unsigned int row_no = obj_vector.size();

  std::ofstream fid;
  fid.open(obj_string, std::ofstream::trunc);             // app

  for (unsigned int k=0; k<row_no; ++k)
  {
    fid << obj_vector[k] << "\n";
  }
  fid.close();
  fid.clear();
  
}


template <typename T>
void save_vector_of_vector_to_a_file(std::string &obj_string,
                                     std::string &obj_tag,
                                     std::vector<T> &obj_vector,
                                     const std::ofstream::openmode& ofstream_openmode)
{
    unsigned int column_no= obj_vector[0].size(); 
    unsigned int row_no = obj_vector.size();
    
    std::ofstream fid;
    fid.open(obj_string, ofstream_openmode);
    
    fid << std::setprecision(6) << obj_tag << "\n";
    
    for (unsigned int k=0; k<row_no; ++k)
    {
        for (unsigned int j=0; j<column_no; ++j)
        {
//             std::cout << obj_vector[k][j] << " ";
            fid << obj_vector[k][j];
            if (j!=column_no-1)
            {
                fid<< " ";
            }
        }
//         std::cout << "\n";
        fid << "\n";
    }
    fid.close();
    fid.clear();
}


bool reading_data_in_a_file_into_a_vector(std::string fileName,
                                          std::vector<double> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : " << fileName << std::endl;
        return false;
    }
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(stod(str));
    }
    //Close The File
    in.close();
    return true;
}


template <typename T>
void reading_data_in_a_file_to_a_matrix(std::string &obj_string_for_reading_from_a_file,
                                     unsigned int row_start,
                                     std::vector<std::vector<T>> &obj_matrix)                           // We require using the data from the first column
{
//     std::cout << "reading " << obj_string_for_reading_from_a_file << " to an array\n";
    
    std::ifstream input_file;
    std::string data_line;
    
    input_file.open(obj_string_for_reading_from_a_file);
    
    if (!input_file)
    {
        std::cout << "Unable to open file\n";
        exit(1);
    }
    
    std::string dummyLine;
    getline(input_file, dummyLine);                             // excluding the head of the data
    
//     std::cout << "dummyLine: " << dummyLine << "\n";
    
    for(unsigned int i = 0; i < row_start; ++i)                 // excluding rows before row_start
    {
        getline(input_file, dummyLine);
    }
    
    std::string element_of_data_per_line = "";
    double value_intermediate;
    
    unsigned int n_row = obj_matrix.size();
    unsigned int n_column = obj_matrix[0].size();
//     std::cout << "n_row: " << n_row << "\n";
//     std::cout << "n_column: " << n_column << "\n";
    
    unsigned int order_column = 0;
    
    for (unsigned int i = 0; i < n_row; ++i)
    {
        getline(input_file, data_line);
        
//         std::cout << "  data in one line: " << data_line << "\n";
        
        for (std::string::iterator it = data_line.begin(); it != data_line.end(); ++it)
        {
            element_of_data_per_line = element_of_data_per_line + *it;
            
            if (isspace(*it) or (it == --data_line.end()))                                          // *it represents one character
            {
//                 std::cout << "  element: " << element_of_data_per_line << "\n";
                
                value_intermediate = stod(element_of_data_per_line);
                
//                 std::cout << "  order_column: " << order_column << "\n";
                
                if (order_column < n_column)                                                        //  && value_intermediate < 1e7
                {
//                     std::cout << "an_int_value: " << int(value_intermediate + 0.5) << "\n";
                                                                                                 
                                                                                                 // the following two lines are selected manually
//                     obj_matrix[i][order_column] = int(value_intermediate + 0.5);              // used when T is int    
                                                                                                 // this does not throw an error even when order_column >= n_column
                    
                    obj_matrix[i][order_column] = value_intermediate;                            // used when T is double
                }
                
                order_column++;
                
                element_of_data_per_line = "";
            }
        }
        order_column = 0;
    }
    
    input_file.close();
    
//     std::cout << "obj_matrix: \n";
//     print_vector_in_vector(obj_matrix);
    
}


template <typename T>
void absorbing_one_vector(std::vector<T> &vector_1,
                          std::vector<T> &vector_2)
{
    for(unsigned int i = 0; i < vector_2.size(); ++i)
    {
        vector_1.push_back(vector_2[i]);
    }
}

template<typename T>
void putting_elements_of_two_vectors_together(std::vector<std::vector<T>> &vector_1,
                                              std::vector<T> &vector_2,
                                              std::vector<T> &vector_3)
{
    for (unsigned int i = 0; i < vector_1.size(); ++i)
    {
        vector_1[i] = {vector_2[i],vector_3[i]};
    }    
}


void creating_a_geometric_series(const double &first_term, 
                                 const double &common_ratio,
                                 const unsigned int &number_of_terms,
                                 std::vector<double> &geometric_series)
{
    geometric_series.resize(number_of_terms);
    
    for (unsigned int i = 0; i < number_of_terms; ++i)
    {
        geometric_series[i] = first_term * pow(common_ratio, i);
    }
    
}
  

void sequencing_numbers_in_a_txt_file(std::string& obj_string)
{

  std::vector<double> coords_of_dofs_sequenced;
  
  reading_data_in_a_file_into_a_vector(obj_string,
                                       coords_of_dofs_sequenced);

  
//   std::cout << "coords_of_dofs_sequenced before sorting:\n";
//   print_vector(coords_of_dofs_sequenced);

  sort( coords_of_dofs_sequenced.begin(), coords_of_dofs_sequenced.end() );

//   std::cout << "coords_of_dofs_sequenced after sorting:\n";
//   print_vector(coords_of_dofs_sequenced);
}


template <typename T>
void adjusting_the_value_of_a_matrix(std::vector<std::vector<T>> &input_matrix,
                                     double &factor,
                                     std::vector<std::vector<T>> &output_matrix)
{
    
    unsigned int column_no= input_matrix[0].size(); 
    unsigned int row_no = input_matrix.size();

    for (unsigned int i = 0; i < row_no; ++i)
    {
        for (unsigned int j = 0; j < column_no; ++j)
        {
            output_matrix[i][j] = input_matrix[i][j] * factor;
        }
    }
    
}


void deleting_a_txt_file(const char *filename)
{
    if( remove( filename ) != 0 )
    {
        perror( "  error deleting file" );
    }
    else
    {
        std::cout << "  " << (filename) << " successfully deleted" << "\n";  
    } 
}



#endif
