# FocusFusion2D
This is a project to simulate a cross-section of a filament in a dense plasma focus. It is not operational yet.

18/05/17:

Most of the cpp code represents relics from past simulation attempts. 
In particular, the large body of code from solver.cpp is devoted to production solving of ODEs for EM fields under Ampere-Darwin
with implicit charge and current.

The present setup is that we initialise a System object (contains arrays of Vertex and Triangle), run the initial ODE solve on it, 
then pour the data into a Systdata object (contains flatpacks of variables) which is then copied on to GPU. The file newkernel2.cu
contains the kernels and kernel calls to perform a set of timesteps. Then we are going to transfer back to CPU to do Delaunay flips
- which must now be altered to work on a Systdata object because transferring back to System object, presently loses information.

At the moment a separate project cudaproj is compiled first by nvcc. This is not ideal.
The only role of the CPU code now shall be: initialise, maintain mesh at regular intervals;
and soft stuff: provide front-end ie windowing and menus, display graphics using DirectX 9.
