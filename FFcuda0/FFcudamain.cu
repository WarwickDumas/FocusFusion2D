
#include "cuda_runtime.h"
#include "../cuda_struct.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../FFxtubes.h"


Systdata Syst1a,Syst2a;


int main()
{
	// Idea will be:

	// . Load systdata

	Syst1a.InvokeHost(NUMBER_OF_VERTICES_AIMED);
	Syst2a.InvokeHost(NUMBER_OF_VERTICES_AIMED);
	
	Syst1a.LoadHost("testsyst.sdt");	
	
	// . Call PerformAdvance:
	// This expects systdata populated on host memory.
	
	PerformCUDA_Advance_2 (
		&Syst1a, 
		Syst1a.Nverts,
		1e-13, 
		10,
		&Syst2a);
	
	Syst2a.AsciiOutput("output.txt");
	
	printf("done.");
	
	Syst1a.RevokeHost();
	Syst2a.RevokeHost();

	getch();

	return 0;
}
