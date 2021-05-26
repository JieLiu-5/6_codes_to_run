The version of Deal.II is 9.2.0.

In the file folder "0_source", the source file, which is "step-4.cc", is placed.
To run "step-4.cc" successfully, the header files in the file folder "1_external" are to replace that in the corresponding file folders after installation. The locations of these header files can be found below.
    1. timer.h is located in include/deal.II/base
    2. the rest are in include/deal.II/numerics

To run the source file, type 
make -j4; ./step-4 2 0 2
in the command line

The arguments following ./step-4 denote the element degree, the method for creating the mesh, and the grid parameter, respectively.
For the second argument, "0" denotes creating the mesh by refining the unit interval(1D)/squred(2D) with a factor 2; "1" denotes using a certain numer of DoFs.
For the third argument, it denotes the refinement level when the second argument is "0" and the number of DoFs when the second argument is "1".
For example, "0 2" for the last two arguments denotes the mesh is created by globally refining an unit interval/squre with a factor 2 two times.

