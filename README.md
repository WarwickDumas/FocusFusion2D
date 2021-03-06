# FocusFusion2D
This is a project to simulate a 2-dimensional cross-section of a filament in a dense plasma focus. 

2021: The main files are NoJxyCu.cpp , kernel.cu, heatflux.cu and cudahost.cu .

The large body of code from solver.cpp is devoted to production solving of ODEs for EM fields under Ampere-Darwin
with implicit charge and current.

The present setup is that we initialise a System object (contains arrays of Vertex and Triangle), run the initial ODE solve on it, 
then pour the data into a Systdata object (contains flatpacks of variables) which is then copied on to GPU. The file newkernel2.cu
contains the kernels and kernel calls to perform a set of timesteps. Then we are going to transfer back to CPU to do Delaunay flips.

The only role of the CPU code now shall be: initialise, maintain mesh at regular intervals;
and soft stuff: provide front-end ie windowing and menus, display graphics using DirectX 9.
