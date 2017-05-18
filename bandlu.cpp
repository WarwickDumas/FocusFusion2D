
#include <iostream>
#include "headers.h"

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

int CompactMatrix::Invoke(int length, int bandsizetemp, int m1temp, int m2temp) 
	{

		int i;
		
		bandsize = bandsizetemp;
		m1 = m1temp;
		m2 = m2temp;

		if (LUSIZE2 > 0)
		{
			cout << "error: Invoke called twice for CompactMatrix class";
			return 1;
		};

		LUSIZE2 = length;
		A = new qd_or_d * [length];
		au = new qd_or_d * [length];
		al = new qd_or_d * [length];
		for (i = 0; i < length; i++)
		{
			A[i] = new qd_or_d[bandsize];
			au[i] = new qd_or_d[bandsize];
			al[i] = new qd_or_d[bandsize];
		};
		indx = new int [length];
		
		if (al[length-1] == NULL || indx == NULL) return 2;

		return 0;
	}

	CompactMatrix::CompactMatrix()
	{
		LUSIZE2 = 0;
		bandsize = 0;
		m1 = 0;
		m2 = 0;
	};
	
	CompactMatrix::~CompactMatrix()
	{
		int i;

		if (LUSIZE2 > 0)
		{
			for (i = 0; i < LUSIZE2; i++)
			{
				delete[] A[i];
				delete[] au[i];
				delete[] al[i];
			};
			delete[]A;
			delete[]au;
			delete[]al;
			delete[]indx;
		};
	}


	int CompactMatrix::bandec(void)
	{
		// assuming A has already been populated,
		// populate au, al, indx using A

		static qd_or_d const TINY = (qd_or_d)1.0e-60;
		int i,j,k,l,mm;
		qd_or_d dum,temp1;
		
		// to begin with, copy A:
		
		for (i = 0; i < LUSIZE2; i++)
			for (j = 0; j < bandsize; j++)
				au[i][j] = A[i][j];
		// might as well call it without A really, since storing A afterwards is pointless
		// but who cares for now 

		mm = m1 + m2 + 1;
		l = m1;
		for (i = 0; i < m1; i++)
		{
			for (j=m1-i; j < mm; j++)
				au[i][j-l]=au[i][j];
			l--;
			for (j=mm-l-1;j < mm; j++)
				au[i][j] = zero;
		};
		//d = 1.0;
		l = m1;
		for (k = 0; k < LUSIZE2; k++)
		{
			dum = au[k][0];
			i = k;
			if (l < LUSIZE2) l++;
			for (j = k+1; j < l; j++)
			{
				if (fabs(au[j][0]) > fabs(dum)) 
				{
					dum = au[j][0];
					i = j;
				};
			};
			indx[k] = i+1;
			if (dum == zero) au[k][0] = TINY;
			if (i != k) {
				//d = -d;
				for (j = 0; j < mm; j++)
				{
					temp1 = au[i][j];
					au[i][j] = au[k][j];
					au[k][j] = temp1;
				};
			};
			for (i = k+1; i < l; i++)
			{
				dum = au[i][0]/au[k][0];
				al[k][i-k-1]=dum;
				for (j = 1; j < mm; j++)
					au[i][j-1] = au[i][j]-dum*au[k][j];
				au[i][mm-1] = zero;
			};
		};
		return 0;
	}


	int CompactMatrix::bandsolve(qd_or_d b[], qd_or_d x[])
	// solves A x = b for x
	{
		int i,j,k,l,mm;
		qd_or_d dum,temp1;
		mm = m1+m2+1;
		l = m1;
		for (k = 0; k < LUSIZE2; k++) x[k] = b[k];
		for (k = 0; k < LUSIZE2; k++) 
		{
			j = indx[k]-1;
			if (j!=k) {
				temp1 = x[k];
				x[k] = x[j];
				x[j] = temp1;
			};
			if (l < LUSIZE2) l++;
			for (j = k+1; j <l; j++)
				x[j] -= al[k][j-k-1]*x[k];
		};
		l = 1;
		for (i = LUSIZE2-1; i>=0; i--)
		{
			dum = x[i];
			for (k = 1; k < l; k++)
				dum -= au[i][k]*x[k+i];
			x[i] = dum/au[i][0];
			if (l<mm) l++;
		};
		return 0;
	}


// for non- band-diagonal matrix:

	Matrix::Matrix ()
	{
		LUSIZE = 0;
	}

	int Matrix::Invoke(long newLUSIZE)
	{
		long i;
		// run this first and then assign values to the elements of the matrix

		if (LUSIZE == newLUSIZE){
			for (i = 0; i < newLUSIZE; i++)
				memset(LU[i],0,sizeof(qd_or_d)*newLUSIZE);
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

		LU = new qd_or_d * [newLUSIZE];
		for (i = 0; i < newLUSIZE; i++)
		{
			LU[i] = new qd_or_d[newLUSIZE];
			memset(LU[i],0,sizeof(qd_or_d)*newLUSIZE);
		}
		indx = new long [newLUSIZE];
		vv = new qd_or_d [newLUSIZE];
		if (vv == NULL) return 2;
		LUSIZE = newLUSIZE;

		return 0;
	};
	
	void Matrix::CopyFrom(Matrix & src)
	{
		for (long i = 0; i < LUSIZE; i++)
			for (long j = 0; j < LUSIZE; j++)
				LU[i][j] = src.LU[i][j];
	};

	Matrix::~Matrix ()
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

	long Matrix::LUdecomp() 
	// LU must already be dimensioned as a matrix[LUSIZE][LUSIZE]
	// index must already be dimensioned as an array[LUSIZE]
	{
		// Method taken from Press et al, Numerical Recipes, 3rd edition 2007
		// pages 52-3

		// After this is run, the LU matrix is decomposed. It has to be assigned values beforehand.


		static qd_or_d const TINY = (qd_or_d)1.0e-100;

		long i,imax,j,k;
		qd_or_d big,temp1;
		//qd_or_d vv[LUSIZE]; // stores implicit scaling of each row
		qd_or_d d;
		
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
			big = zero;					// initialise for the search for largest pivot element
			for (i = k; i < LUSIZE; i++)
			{
				temp1 = vv[i]*fabs(LU[i][k]);
				if (temp1 > big) // is the figure of merit for the pivot better than the best so far?
				{
					big = temp1;
					imax = i;
				};
			};
			if (k != imax)	// do we need to interchange rows?
			{
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
			indx[k] = imax;

			if (LU[k][k] == zero) LU[k][k] = TINY;
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
		};
		
		return 0; // successful?
	}

	long Matrix::LUSolve (qd_or_d b[], qd_or_d x[])
		// Make x solve A x = b, where A was the matrix originally defined by assignments before decomposition
	{
		long i, ii, ip, j;
		qd_or_d sum;
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
			sum = x[i];
			for (j = i+1; j < LUSIZE; j++) sum -= LU[i][j]*x[j];
			x[i] = sum/LU[i][i];		// store a component of the solution vector X.

			// Add extra thing: this wastes many cycles but the point is
			// that we may have an undetermined element of x.
			if (LU[i][i] == 0.0) x[i] = 0.0;
			// Maybe this will fix it.

			// For some reason no, we have in the decomp matrix a thing we should
			// not have -- LU[i][i] == 0 for the previous element.
		};
		return 0;
	}

	long Matrix::LUSolveII (qd_or_d b[], qd_or_d x[],int iWhich, qd_or_d value)
		// Make x solve A x = b, where A was the matrix originally defined by assignments before decomposition
	{
		long i, ii, ip, j;
		qd_or_d sum;
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


		static real const TINY = (real)1.0e-100;

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
			imax = -1;
			big = zero;					// initialise for the search for largest pivot element
			for (i = k; i < LUSIZE; i++)
			{
				temp1 = vv[i]*fabs(LU[i][k]);
				if (temp1 > big) // is the figure of merit for the pivot better than the best so far?
				{
					big = temp1;
					imax = i;
				};
			};
			if (k != imax)	// do we need to interchange rows?
			{
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
			if (imax < 0) {
				printf("problem -- imax not found.\n");
				getch();
			} else {
				indx[k] = imax;
			};

			if (LU[k][k] == zero) LU[k][k] = TINY;
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
		};

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

		for (i = LUSIZE-1; i >= 0; i--) // now do the back-substitution, (2.3.7)
		{
			sum = x[i];
			for (j = i+1; j < LUSIZE; j++) sum -= LU[i][j]*x[j];
			x[i] = sum/LU[i][i];		// store a component of the solution vector X.

			// Add extra thing: this wastes many cycles but the point is
			// that we may have an undetermined element of x.
			if (LU[i][i] == 0.0) x[i] = 0.0;
			// Maybe this will fix it.

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

