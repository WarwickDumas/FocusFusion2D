
#include <iostream>
#include <conio.h>
#include "matrix_real.h"

using std::cout;

// CompactMatrix class based on the Bandec class given on p.59 of 
// Press, Teukolsky, Vetterling and Flannery 
// Numerical Recipes for Scientific Computing, third edition (2007)
// [ they cite for this, Keller, H.B. (1968) ].

// call constructor with bandsize, m1, m2 as arguments
// then call invoke - size is now fixed forever

// then can define matrix A
// then can call bandec, bandsolve(RHS vector,soln vector) (can repeat with different RHS vector)
// & can repeat this.

#define unity 1.0
#define zero 0.0


	Matrix_real::Matrix_real ()
	{
		LUSIZE = 0;
	}

	int Matrix_real::Invoke(long newLUSIZE)
	{
		long i;
		// run this first and then assign values to the elements of the matrix

		if (LUSIZE == newLUSIZE){
			for (i = 0; i < newLUSIZE; i++)
				memset(LU[i],0,sizeof(real)*newLUSIZE);
			return 0;
		}

		if (LUSIZE > 0)
		{
			for (i = 0; i < LUSIZE; i++)
				delete[] LU[i];
			delete[] LU;
			delete[] indx;
			delete[] vv;
			LUSIZE = 0;
			//printf("Note: Matrix::Invoke called twice.");
		};

		LU = new real * [newLUSIZE];
		for (i = 0; i < newLUSIZE; i++)
		{
			LU[i] = new real[newLUSIZE];
			memset(LU[i],0,sizeof(real)*newLUSIZE);
		}
		indx = new long [newLUSIZE];
		vv = new real [newLUSIZE];
		if (vv == NULL) return 2;
		LUSIZE = newLUSIZE;

		return 0;
	};
	
	void Matrix_real::CopyFrom(Matrix_real & src)
	{
		for (long i = 0; i < LUSIZE; i++)
			for (long j = 0; j < LUSIZE; j++)
				LU[i][j] = src.LU[i][j];
	};

	Matrix_real::~Matrix_real ()
	{
		long i;
		if (LUSIZE > 0)
		{
			for (i = 0; i < LUSIZE; i++)
				delete[] LU[i];
			delete[] LU;
			delete[] indx;
			delete[] vv;
		};
		LUSIZE = 0;
	};

	long Matrix_real::LUdecomp(/*real LU[LUSIZE][LUSIZE], long indx[]*/) 
	// LU must already be dimensioned as a matrix[LUSIZE][LUSIZE]
	// index must already be dimensioned as an array[LUSIZE]
	{
		// Method taken from Press et al, Numerical Recipes, 3rd edition 2007
		// pages 52-3

		// After this is run, the LU matrix is decomposed. It has to be assigned values beforehand.


		static real const TINY = (real)1.0e-80;

		long i,imax,j,k;
		real big,temp1;
		//real vv[LUSIZE]; // stores implicit scaling of each row
		real d;
		
		d = unity;		// no row interchanges yet

		for (i = 0; i < LUSIZE; i++) // loop over rows to get implicit scaling information
		{
			big = zero;
			for (j = 0; j < LUSIZE; j++)
				if ((temp1=fabs(LU[i][j])) > big) big = temp1;
			if (big == zero) 
				return 1; // no nonzero largest element
			vv[i] = unity/big;
		};
		for (k = 0; k < LUSIZE; k++) // the outermost kij loop
		{
		
			if (k < LUSIZE - 1) // if it == LUSIZE-1 then we don't need to run this, and it's dangerous too.
			{
				imax = -1;
				big = zero;					// initialise for the search for largest pivot element
				for (i = k; i < LUSIZE; i++)
				{
					temp1 = vv[i] * fabs(LU[i][k]);
					if (temp1 > big) // is the figure of merit for the pivot better than the best so far?
					{
						big = temp1;
						imax = i;
					};
				};
				if (k != imax)	// do we need to interchange rows?
				{
					// Swap row imax with row k.

					// With a big matrix, what we want to do for efficiency is just exchange pointers,
					// from an array of pointers to rows.

					for (j = 0; j < LUSIZE; j++)
					{
						temp1 = LU[imax][j];
						LU[imax][j] = LU[k][j];
						LU[k][j] = temp1;
					};
					d = -d;	// change the parity of d [ never used for anything ]

					// surely here we should be actually interchanging properly?

					// Old version:
					//vv[imax] = vv[k];  // interchange the scale factor

					// Warwick code:
					temp1 = vv[imax];
					vv[imax] = vv[k];
					vv[k] = temp1;  // does that help anything? Guess probably not.
					// Otherwise the info from vv[imax] is lost completely - can that be right? This is all a bit crazy.
					// never used again??

				};
			} else {
				imax = k;
			}
			if (imax < 0) {
				printf("problem -- imax not found.\n");
				getch();
			} else {
				indx[k] = imax;
			};

			if (LU[k][k] == zero) {
				LU[k][k] = TINY;
				printf("singular matrix!\n");
			};

			// pivot element == zero, ie singular matrix
			// huh.
			// Well, this is simply changing the matrix then
			// And making it appear that we are solving with some
			// arbitrary influence of the unused value.

			for (i = k+1; i < LUSIZE; i++)
			{
				temp1 = LU[i][k] /= LU[k][k]; // "divide by the pivot element"
				for (j = k+1; j < LUSIZE; j++)  // innermost loop: reduce remaining submatrix
					LU[i][j] -= temp1*LU[k][j];
			};

			// Here could launch each row or column as a block.


		};
	//	printf("LUdecomp done\n");

		return 0; // successful?
	}

	long Matrix_real::LUSolve (real b[], real x[])
		// Make x solve A x = b, where A was the matrix originally defined by assignments before decomposition
	{
		long i, ii, ip, j;
		real sum;
		ii = 0;
		
		for (i = 0; i < LUSIZE; i++)
			x[i] = b[i];
		for (i = 0; i < LUSIZE; i++) 
			// when ii is +ve, it will become the index of the first nonvanishing element of b
		{		// we now do the forward substitution (2.3.6), unscrambling as we go
			ip = indx[i];
			sum = x[ip];
			x[ip] = x[i];
			if (ii != 0)
				for (j = ii-1; j < i; j++) sum -= LU[i][j]*x[j];
			else if (sum != zero) // a nonzero element was encountered, so from now on we will do the sums in the loop above
				ii = i+1;
			x[i] = sum;
		};
		// My guess: the L matrix is the one with 1's on the diagonal, since LU[i][i] does not appear up to this point.

		// For further debugging why not output this intermediate x

		// It works up through the rows  -- we need x[j > i].
		// We could do 16 consecutive followed by next 16 together using those 16, followed by 16 consecutive.
		// Next 16, we use 32 final rows, then do those 16 together. Can do 'consecutive' part in a kernel.
		// Maybe it goes:
		// for thread 0 alone: do calc ; syncthreads
		// all other threads: use result ; syncthreads
		// for thread 1 alone: do calc ; syncthreads
		// all other threads: use result ; syncthreads ...

		for (i = LUSIZE-1; i >= 0; i--) // now do the back-substitution, (2.3.7)
		{
			sum = x[i];
			for (j = i+1; j < LUSIZE; j++) sum -= LU[i][j]*x[j];
			
			// Add extra thing: this wastes many cycles but the point is
			// that we may have an undetermined element of x.
			if (LU[i][i] == 0.0) {
				x[i] = 0.0;
				// Maybe this will fix it.
			} else {
				x[i] = sum / LU[i][i];		// store a component of the solution vector X.
			};
			// For some reason no, we have in the decomp matrix a thing we should
			// not have -- LU[i][i] == 0 for the previous element.
		};		
		return 0;
	}

	long Matrix_real::LUSolveII (real b[], real x[],int iWhich, real value)
		// Make x solve A x = b, where A was the matrix originally defined by assignments before decomposition
	{
		long i, ii, ip, j;
		real sum;
		ii = 0;
		
		for (i = 0; i < LUSIZE; i++)
			x[i] = b[i];
		for (i = 0; i < LUSIZE; i++) 
			// when ii is +ve, it will become the index of the first nonvanishing element of b
		{		// we now do the forward substitution (2.3.6), unscrambling as we go
			ip = indx[i];
			sum = x[ip];
			x[ip] = x[i];
			if (ii != 0)
				for (j = ii-1; j < i; j++) sum -= LU[i][j]*x[j];
			else if (sum != zero) // a nonzero element was encountered, so from now on we will do the sums in the loop above
				ii = i+1;
			x[i] = sum;
		};
		// My guess: the L matrix is the one with 1's on the diagonal, since LU[i][i] does not appear up to this point.

		// For further debugging why not output this intermediate x

		for (i = LUSIZE-1; i >= 0; i--) // now do the back-substitution, (2.3.7)
		{
			if (i == iWhich) {
				x[i] = value;
			} else {

				sum = x[i];
				for (j = i+1; j < LUSIZE; j++) sum -= LU[i][j]*x[j];
				x[i] = sum/LU[i][i];		// store a component of the solution vector X.

				// Add extra thing: this wastes many cycles but the point is
				// that we may have an undetermined element of x.
			//	if (LU[i][i] == 0.0) x[i] = 0.0;
				// Maybe this will fix it.

				// For some reason no, we have in the decomp matrix a thing we should
				// not have -- LU[i][i] == 0 for the previous element.
			};
		};
		
		return 0;
	}

