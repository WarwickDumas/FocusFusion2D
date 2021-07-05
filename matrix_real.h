
#ifndef MATREAL_H
#define MATREAL_H
#define real double

// CompactMatrix class based on the Bandec class given on p.59 of 
// Press, Teukolsky, Vetterling and Flannery 
// Numerical Recipes for Scientific Computing, third edition (2007)
// [ they cite for this, Keller, H.B. (1968) ].

// call constructor with bandsize, m1, m2 as arguments
// then call invoke - size is now fixed forever

// then can define matrix A
// then can call bandec, bandsolve(RHS vector,soln vector) (can repeat with different RHS vector)
// & can repeat this.


class Matrix_real 
{
private:
	
public:
	real ** LU; // the elements
	long LUSIZE;  // the size of the matrix
	real * vv;
	long * indx;   // the index array when decomposed

	Matrix_real ();

	int Invoke(long newLUSIZE);
	
	void CopyFrom(Matrix_real & src);

	~Matrix_real ();

	long LUdecomp() ;

	long LUSolve (real b[], real x[]);
	long LUSolveII (real b[], real x[],int iWhich, real value);

};

#endif