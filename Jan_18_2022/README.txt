The version of Deal.II is 9.2.0.

In the file folder "0_source", the source file, which is "step-4.cc", and two auxiliary header files are placed.
To run "step-4.cc" successfully, the header files in the file folder "1_external" are to replace that in the corresponding file folders after installation. The locations of these header files can be found below.
    1. timer.h is located in include/deal.II/base
    2. the rest are in include/deal.II/numerics

To run the source file, we may type 
make -j4; ./step-4 1e-3 2 0 2 1
in the command line.

The arguments following ./step-4 denote the parameter of the solution, element degree,  method for creating the mesh, grid parameter, and total number of refinements, respectively.
For the third argument, "0" denotes creating the mesh by refining the unit interval(1D)/square(2D) with a factor 2; "1" denotes using a certain numer of DoFs.
For the fourth argument, it denotes the refinement level when the second argument is "0" and the number of DoFs when the second argument is "1".

Experiments to run:
./step-4 1e-3 5 0 10 1
./step-4 1e-2 3 0 11 1

