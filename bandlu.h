
#ifndef BANDLU_H
#define BANDLU_H

// use qd as include, not as part of an library, so we need to define QD_API empty
#include "qd/qd_config.h"
#undef QD_API
#define QD_API
#include "qd/dd_real.h"
//#include "qd/qd_real.h"
#include "qd/fpu.h"
#define real double
#define qd_or_d dd_real


// CompactMatrix class based on the Bandec class given on p.59 of 
// Press, Teukolsky, Vetterling and Flannery 
// Numerical Recipes for Scientific Computing, third edition (2007)
// [ they cite for this, Keller, H.B. (1968) ].

// call constructor with bandsize, m1, m2 as arguments
// then call invoke - size is now fixed forever

// then can define matrix A
// then can call bandec, bandsolve(RHS vector,soln vector) (can repeat with different RHS vector)
// & can repeat this.

class CompactMatrix
{
private:
	int * indx;
	int LUSIZE2;
	qd_or_d ** au; // upper
	qd_or_d ** al; // lower
	
	int bandsize, m1, m2;

public:

	qd_or_d ** A;
	//int m1, m2;

	int Invoke(int length, int bandsizetemp, int m1temp, int m2temp) ;

	CompactMatrix();
	
	~CompactMatrix();


	int bandec(void);

	int bandsolve(qd_or_d b[], qd_or_d x[]);


/*
	void output(const char * str, qd_or_d * jilly)
	{
		FILE * hfile;
		int i,j;

		hfile = fopen(str,"w");

		// dump matrix values to file
		for (i = 0; i < LUSIZE2; i++)
		{
			for (j = 0; j < bandsize; j++)
				fprintf(hfile,"%1.10E ",A[i][j].x[0]);
			fprintf(hfile,"%1.10E \n",jilly[i]);
		};
		fclose(hfile);
	}*/
};


// for non- band-diagonal matrix:

class Matrix 
{
private:
	long * indx;   // the index array when decomposed
	
public:
	qd_or_d ** LU; // the elements
	long LUSIZE;  // the size of the matrix
	qd_or_d * vv;

	Matrix ();

	int Invoke(long newLUSIZE);
	
	void CopyFrom(Matrix & src);

	~Matrix ();

	long LUdecomp() ;

	long LUSolve (qd_or_d b[], qd_or_d x[]);

	long LUSolveII (qd_or_d b[], qd_or_d x[],int iWhich, qd_or_d value);

};


class Matrix_real 
{
private:
	long * indx;   // the index array when decomposed
	
public:
	real ** LU; // the elements
	long LUSIZE;  // the size of the matrix
	real * vv;

	Matrix_real ();

	int Invoke(long newLUSIZE);
	
	void CopyFrom(Matrix_real & src);

	~Matrix_real ();

	long LUdecomp() ;

	long LUSolve (real b[], real x[]);
	long LUSolveII (real b[], real x[],int iWhich, real value);

};

#endif