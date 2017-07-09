
#include "cuda_runtime.h"
#include "../cuda_struct.h"
#include "device_launch_parameters.h"
#include <stdio.h>


Systdata Syst1,Syst2;


int main()
{
	// Idea will be:

	// . Load systdata

	Syst1.InvokeHost(NUMBER_OF_VERTICES);
	Syst2.InvokeHost(NUMBER_OF_VERTICES);
	
	Syst1.LoadHost("testsyst.sdt");	
	
	// . Call PerformAdvance:
	// This expects systdata populated on host memory.
	
	PerformCUDA_Advance_2 (
		&Syst1, 
		Syst1.Nverts,
		1e-13, 
		10,
		&Syst2,
		f64 t // time of first timeslice
		);
	
	Syst2.AsciiOutput("output.txt");
	
	printf("done.");
	
	Syst1.RevokeHost();
	Syst2.RevokeHost();

	getch();

	return 0;
}
