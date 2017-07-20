
// Version 0.51

// Been over some attempts to ameliorate local accesses -- not v successful basically.
// Correction in "Get Lap phi" routine.

// Version 0.52:

// Change Lap A, Grad A routines to load CHAR4 p_tri_per_neigh instead of loading data
// to interrogate neighbour periodic status.

// Change major area calc in the INNERMOST/OUTERMOST case.

// Note that central area calc does not look right.

// Version 0.53:

// Made changes to Reladvect_nT because it was taking wrong connection for OUTERMOST.
// Changed flag tests & treatment of Inner verts in preceding routines.

// Version 0.54:

// Adjusted area calculations as written in spec. 
// We set ins crossing tri minor area = 0, centroid on ins;
// frill area = 0, centroid on boundary.

// Version 0.6:

// Debugging and making corrections.



// PLAN: 
// Allow that on GPU we can move outside domain and it's fine, we do not change PB data.
// PB data will be only changed on CPU.
// Nonetheless we kept PBCTri lists which can be updated, unlike has_periodic alone, in case
// of moving something to its image within the domain.



// NOTES:

		// Ensure that outside the domain, n_major is recorded as 0

		// Ensure that outside the domain, resistive_heat is recorded as 0

// Notes as of 11/04/17:

// Getting two problems basically according to ptxas --v.

// Both Lap_A and Midpt have got (1252, 2500); (1000,2332) spill (stores,loads).
// Both of them will try to run 512 threads/SM which is what you get with 63 registers used.
// The registers represent ~~ 128KB / SM or 32 doubles/thread.
// The extra for L1 would be, for midpt, 48KB => 12 doubles.
// For Lap_A we do not even get that.
// But we have spill stores of e.g. 1024 = 128 doubles extra.

// We basically want to halve the amount of stored data for midpt.
// We went to some lengths to AVOID global random read/writes and now
// we are looking at reading 256 DOUBLES PER KERNEL RUN.
// PROCESS THAT !! So what can be done???

// For midpt there is scope to use more shared memory but it will clearly
// only represent 4 doubles extra per thread in total.

// We could reduce to 256 threads running at once and this might well be faster.
// Then we get L1 with 24 doubles for midpt. -- more likely there to be worth having.

// For Lap_A routine some kind of drastic action is called for.

// Notes as of 11/04/17.


#include <math.h>
#include <time.h>
#include <stdio.h>

#include "flags.h"

#define OUTPUT 1

// Note that this file has to first be compiled with nvcc
// Then with -dlink, apply nvcc to the obj file to produce another obj file;
// Include both obj files and cudart.lib in the main project.

// -dlink command line:

// E:\focusfusion\FFxtubes\cudaproj\x64\Release>
// nvcc -dlink -gencode=arch=compute_20,code=\"sm_20,compute_20\" --machine 64 -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" newwrapper2.cu.obj -o newwrapper2.obj

// E:\focusfusion\FFxtubes\cudaproj\x64\Debug>
//nvcc -dlink -gencode=arch=compute_20,code=\"sm_20,compute_20\" --machine 64 -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MTd "  -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" newkernel2.cu.obj -o newkernel2.obj

#define SIXTH 0.166666666666667
#define TWELTH 0.083333333333333
#define FIVETWELTHS 0.416666666666667
#define REPORT 69500
#define DEVICE_INSULATOR_OUTER_RADIUS 3.44

#include "cuda_struct.h"

#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   
						// { Call(cudaStatus, "cudaStatus") } ?
extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;

//long BLOCK_START_OF_FRILL_SEARCH = 256;
// Do 288-256 = 32 blocks -- assuming 256*288 tris
// This won't get the inner frills!!!!


// Global host helper:
__host__ bool Call(cudaError_t cudaStatus,char str[]);

#include "systdata.cpp"

char * report_time_(int action)
{
	/* action = 0: reset ; action = 1: report */
	/* requires timebuffy to be defined as char[255] globally */

	static char timebuffer[255];
	static clock_t start;
	double timenow;
	long ops;

	if (action == 0) 
		{
		start = clock();
		}
	else
		{
		timenow = ((double)(clock()-start)/(double)CLOCKS_PER_SEC);
		ops = (long)(clock()-start);
		/* create a null-terminated string */
		sprintf (timebuffer, "%6.4f sec.",timenow);
		};
	return &(timebuffer[0]);	
};


// Global helper functions for kernels:

__device__ __forceinline__ f64 Get_lnLambda_ion_d(f64 n_ion,f64 T_ion);
__device__ __forceinline__ f64 Get_lnLambda_d(real n_e,real T_e);
__device__ __forceinline__ f64_vec2 Anticlock_rotate2(const f64_vec2 arg);
__device__ __forceinline__ f64_vec2 Clockwise_rotate2(const f64_vec2 arg);
__device__ __forceinline__ f64_vec3 Anticlock_rotate3(const f64_vec3 arg);
__device__ __forceinline__ f64 Estimate_Neutral_MT_Cross_section(f64 T);
__device__ __forceinline__ f64 Estimate_Neutral_Neutral_Viscosity_Cross_section(f64 T) ;
__device__ __forceinline__ f64 Estimate_Ion_Neutral_Viscosity_Cross_section(f64 T);
__device__ __forceinline__ f64 Calculate_Kappa_Neutral(f64 n_i, f64 T_i, f64 n_n, f64 T_n);


#ifdef __CUDACC__
__device__ __forceinline__ f64 GetEzShape(f64 r) {
	return 1.0-1.0/(1.0+exp(-16.0*(r-4.2))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}
#else
f64 inline GetEzShape_(f64 r) {
	return 1.0-1.0/(1.0+exp(-16.0*(r-4.2))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}
#endif



// Device-accessible constants not known at compile time:
__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices

__constant__ f64_tens2 Anticlockwise2, Clockwise2; // use this to do rotation.

// Set from host constant definitions:
__constant__ f64 sC, kB, c,Z, e,q,m_e, m_ion, m_n,
				 eoverm, qoverM, moverM, eovermc, qoverMc, 
				 FOURPI_Q_OVER_C, FOURPI_Q, FOURPI_OVER_C,
				 NU_EI_FACTOR, // Note: NU_EI_FACTOR goes with T in eV -- !!
				 nu_eiBarconst, csq, m_s, 
				  // New:
				 FOUR_PI;
//__constant__ long BLOCK_START_OF_FRILL_SEARCH_d;

__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
				 cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];

// Set from calculations in host routine:
__constant__ f64 Nu_ii_Factor, kB_to_3halves, 
				 one_over_kB, one_over_kB_cubed,
				over_sqrt_m_ion,over_sqrt_m_e,over_sqrt_m_neutral;

// Other:
__constant__ f64 T_ion_avg_sq_d, T_neut_avg_sq_d, T_elec_avg_sq_d,
				v_ion_avg_sq_d,v_neut_avg_sq_d,
				MAXERRPPNSQ_d, AVGFAC_d, ABSTHRESHFLUX_SQ_d,ENDPT_MAXERRPPN_SQ_d,
				avgTe,avgTi,avgTn;

//__constant__ long ReverseJzIndexStart, ReverseJzIndexEnd; // MaxNeigh_d
// use #define MAXNEIGH_d but we will have to allow that there is a different
// maximum used for arrays loaded-in than for actual max in list.
// Could be 10 vs 20.

__constant__ f64 four_pi_over_c_ReverseJz;

__constant__ f64 FRILL_CENTROID_OUTER_RADIUS_d, 
				FRILL_CENTROID_INNER_RADIUS_d;

__device__ real * p_summands, * p_Iz0_summands, * p_Iz0_initial,
				* p_scratch_d, 
				* p_resistive_heat_neut_minor,
				* p_resistive_heat_ion_minor,
				* p_resistive_heat_elec_minor,
				* p_Lapphi;
__device__ f64_vec2 * p_grad_phidot;
__device__ f64_vec3 * p_MAR_neut, * p_MAR_ion, * p_MAR_elec;
__device__ nn *p_nn_ionrec_minor;

#include "E:/focusfusion/FFxtubes/helpers.cu"

#define Set_f64_constant(dest, src) { \
		Call(cudaGetSymbolAddress((void **)(&f64address), dest ), \
			"cudaGetSymbolAddress((void **)(&f64address), dest )");\
		Call(cudaMemcpy( f64address, &src, sizeof(f64),cudaMemcpyHostToDevice),\
			"cudaMemcpy( f64address, &src, sizeof(f64),cudaMemcpyHostToDevice) src dest");\
						}

Systdata Syst1, Systhalf, Syst2, SystAdv;

__host__ bool Call(cudaError_t cudaStatus,char str[])
{
	if (cudaStatus == cudaSuccess) return false;	
	printf("Error: %s\nReturned %d : %s\n",
		str, cudaStatus,cudaGetErrorString(cudaStatus));
	printf("Anykey.\n");	getch();
	return true;
}

real GetIzPrescribed(real const t)
{
	real Iz = -PEAKCURRENT_STATCOULOMB * sin ((t + ZCURRENTBASETIME) * PIOVERPEAKTIME );
	//printf("\nGetIzPrescribed : t + ZCURRENTBASETIME = %1.5E : %1.12E\n", t + ZCURRENTBASETIME, Iz);
	return Iz;
}

// Do we want to create a 1:1 link between major tiles and minor tiles, or do we want
// a border on each. ...
/*__device__ void __forceinline__ atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    // return __longlong_as_double(old);
}*/

#include "newkernel2.cu"


nn * p_nn_host;
f64_vec3 * p_MAR_ion_host, * p_MAR_neut_host, * p_MAR_elec_host;

void Systdata::AsciiOutput (const char filename[]) const 
{
	FILE * file = fopen(filename,"w");
	if (file == 0) {
		printf("could not open %s",filename);
		getch();
		getch();
		return;
	} ;
	printf("%s opened",filename);
	
	fprintf(file,"index flag | n_neut T_neut n_ion T_ion n_elec T_elec | ionise recombine | "
		"Bx By Bz | vnx vny vnz vix viy viz vex vey vez | "
		"gradphi_x gradphi_y Lap_A_x Lap_A_y Lap_A_z Az Adot_x Adot_y Adot_z | X1_Adot_z | "
		"MAR_neutx MAR_neuty MAR_neutz MAR_ionx MAR_iony MAR_ionz MAR_elecx MAR_elecy MAR_elecz | "
		"GradTe_x GradTe_y phi \n");
	
	for (int iMinor = 0; iMinor < this->Nminor; iMinor++)
	{
		f64 temp1;
	//	cudaMemcpy(&temp1, pX1->p_Adot+iMinor, 
	//	sizeof(f64_vec3),			cudaMemcpyDeviceToHost);
		
		if (iMinor % 500 == 0) printf("%d ",iMinor);
		
		int flag;
		if (iMinor < BEGINNING_OF_CENTRAL) {
			flag = this->p_tri_perinfo[iMinor].flag;
		} else {
			flag = this->p_info[iMinor-BEGINNING_OF_CENTRAL].flag;
		};

		fprintf(file,"%d %d | %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E |  %1.14E %1.14E | "
			" %1.14E %1.14E %1.14E | %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
			iMinor, flag,
			this->p_nT_neut_minor[iMinor].n,this->p_nT_neut_minor[iMinor].T,
			this->p_nT_ion_minor[iMinor].n,this->p_nT_ion_minor[iMinor].T,
			this->p_nT_elec_minor[iMinor].n,this->p_nT_elec_minor[iMinor].T,
			p_nn_host[iMinor].n_ionise, p_nn_host[iMinor].n_recombine,
			this->p_B[iMinor].x,this->p_B[iMinor].y,this->p_B[iMinor].z,
			this->p_v_neut[iMinor].x,this->p_v_neut[iMinor].y,this->p_v_neut[iMinor].z,
			this->p_v_ion[iMinor].x,this->p_v_ion[iMinor].y,this->p_v_ion[iMinor].z,
			this->p_v_elec[iMinor].x,this->p_v_elec[iMinor].y,this->p_v_elec[iMinor].z
			);
			
		fprintf(file,	
			" %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | %1.14E | ",
			this->p_grad_phi[iMinor].x,this->p_grad_phi[iMinor].y,
			this->p_Lap_A[iMinor].x,this->p_Lap_A[iMinor].y,this->p_Lap_A[iMinor].z,
			this->p_A[iMinor].z,
			this->p_Adot[iMinor].x,this->p_Adot[iMinor].y,this->p_Adot[iMinor].z,
			0.0);
			//temp1);
		
		fprintf(file,	
			" %1.14E %1.14E %1.14E ",
			p_MAR_neut_host[iMinor].x,p_MAR_neut_host[iMinor].y,p_MAR_neut_host[iMinor].z);
		
		fprintf(file,	
			" %1.14E %1.14E %1.14E ",
			p_MAR_ion_host[iMinor].x,p_MAR_ion_host[iMinor].y,p_MAR_ion_host[iMinor].z);
		
		fprintf(file,	
			" %1.14E %1.14E %1.14E ",
			p_MAR_elec_host[iMinor].x,p_MAR_elec_host[iMinor].y,p_MAR_elec_host[iMinor].z);
		
		fprintf(file,		" %1.14E %1.14E ",
			this->p_GradTe[iMinor].x,this->p_GradTe[iMinor].y);
		
		if (iMinor < BEGINNING_OF_CENTRAL) {
			fprintf(file," %1.10E %1.10E ",this->p_tri_centroid[iMinor].x,this->p_tri_centroid[iMinor].y);
		} else {
			fprintf(file," %1.10E | %1.10E %1.10E ",
				this->p_phi[iMinor-BEGINNING_OF_CENTRAL],
				this->p_info[iMinor-BEGINNING_OF_CENTRAL].pos.x,
				this->p_info[iMinor-BEGINNING_OF_CENTRAL].pos.y);
		};
		fprintf(file,"\n");
	};
	fclose(file);
	
}


void SendToHost(const Systdata * pX_nvT, const Systdata * pXhalf, const Systdata * pX_host)
{
	cudaMemcpy(pX_host->p_phi,					pXhalf->p_phi,
		sizeof(f64)*pX_host->Nverts,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_nT_neut_minor,		pX_nvT->p_nT_neut_minor,
		sizeof(nT)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_nT_ion_minor,			pX_nvT->p_nT_ion_minor,
		sizeof(nT)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_nT_elec_minor,		pX_nvT->p_nT_elec_minor,
		sizeof(nT)*pX_host->Nminor,			cudaMemcpyDeviceToHost);

	cudaMemcpy(pX_host->p_tri_centroid,		pXhalf->p_tri_centroid,
		sizeof(f64_vec2)*pX_host->Ntris,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_B,				pXhalf->p_B,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);

	cudaMemcpy(pX_host->p_v_neut,			pX_nvT->p_v_neut,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_v_ion,			pX_nvT->p_v_ion,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_v_elec,			pX_nvT->p_v_elec,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);

	cudaMemcpy(pX_host->p_grad_phi,			pXhalf->p_grad_phi,
		sizeof(f64_vec2)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_Lap_A,			pXhalf->p_Lap_A,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(pX_host->p_Adot,				pXhalf->p_Adot,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);

	cudaMemcpy(pX_host->p_GradTe,			pX_nvT->p_GradTe,
		sizeof(f64_vec2)*pX_host->Nminor,			cudaMemcpyDeviceToHost);

	cudaMemcpy(p_nn_host,					p_nn_ionrec_minor,
		sizeof(nn)*pX_host->Nminor,					cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_neut_host,				p_MAR_neut,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_ion_host,				p_MAR_ion,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_elec_host,				p_MAR_elec,
		sizeof(f64_vec3)*pX_host->Nminor,			cudaMemcpyDeviceToHost);
	
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize memcpies S2H");

}

void PerformCUDA_Advance_2 (
		const Systdata * pX_host, // populate in CPU MSVC routine...
		long numVerts,
		const real hsub, 
		const int numSubsteps,
		const Systdata * pX_host_target
		)
{
	// Preliminaries:
	
	char buffer[256];
	FILE * fpdebug;
	
	long iVertex;
	Systdata * pX1,*pX2,*pXhalf,*pXusable;
	real * p_summands_host, * p_Iz0_summands_host, *p_scratch, *p_Iz0_initial_host, * p_scratch_host;
	structural * p_scratch_info;
	int iSubsubstep;
	size_t uFree, uTotal;	
	long numVertices = numVerts;
	real const hstep = hsub/(real)numSubsteps;
	
	real evaltime = pX_host->evaltime;
	real t = evaltime;

	printf("pXhost->p_Adot[20000 + BEGINNING_OF_CENTRAL].z %1.10E\n",pX_host->p_Adot[20000 + BEGINNING_OF_CENTRAL].z);

	printf("sizeof(CHAR4): %d \n"
		"sizeof(structural): %d \n"
		"sizeof(LONG3): %d \n"
		"sizeof(nn): %d \n",
		   sizeof(CHAR4),sizeof(structural),sizeof(LONG3),sizeof(nn));
	getch();


	if (Syst1.bInvoked == false) {
		
		Call(cudaMemGetInfo (&uFree,&uTotal),"cudaMemGetInfo (&uFree,&uTotal)");
		printf("Before Invokes: uFree %d uTotal %d\n",uFree,uTotal);
		
		Syst1.Invoke(numVertices);
		Systhalf.Invoke(numVertices);
		Syst2.Invoke(numVertices);
		SystAdv.Invoke(numVertices);
		
		Call(cudaMemGetInfo (&uFree,&uTotal),"cudaMemGetInfo (&uFree,&uTotal)");
		printf("After Invokes: uFree %d uTotal %d\n",uFree,uTotal);
	}
	
	// -----  What needs to change in all this stuff?  -----
	
	// Populate video constant memory:
	// ________________________________
	
	long * address;
	f64 * f64address;
	
	// not used? :

//	Call(cudaGetSymbolAddress((void **)(&address),Nverts), 
//		"cudaGetSymbolAddress((void **)(&address),Nverts)");
//	Call(cudaMemcpy( address, &numVertices, sizeof(long),cudaMemcpyHostToDevice),
//		"cudaMemcpy( address, &numVertices, sizeof(long),cudaMemcpyHostToDevice) 2 ");
// good
	
	//memcpy(IndexNeigh,pIndexNeigh + long_neighs_stride*index,long_neighs_stride);
	//memcpy(PBCNeigh,pPBCneigh + char_neighs_stride*index,char_neighs_stride);
	
	// Eventually change this :
	//long MaxNeigh = MAXNEIGH_d;
	//CallMAC(cudaGetSymbolAddress((void **)(&address),MAXNEIGH_d));
	//CallMAC(cudaMemcpy( address, &MaxNeigh, sizeof(long),cudaMemcpyHostToDevice));
	
	// not used:
	//Call(cudaGetSymbolAddress((void **)(&address),uDataLen_d), 
	//	"cudaGetSymbolAddress((void **)(&address),uDataLen_d)");
	//Call(cudaMemcpy( address, &numDataLen, sizeof(long),cudaMemcpyHostToDevice),
	//	"cudaMemcpy( address, &numDataLen, sizeof(long),cudaMemcpyHostToDevice) 3 ");
	
	//Tensor2 const Anticlockwise(cos(FULLANGLE),-sin(FULLANGLE),sin(FULLANGLE),cos(FULLANGLE));
	f64_tens2 anticlock2;
	// Note that objects appearing in constant memory must have empty constructor & destructor.
	anticlock2.xx = cos(FULLANGLE);
	anticlock2.xy = -sin(FULLANGLE);
	anticlock2.yx = sin(FULLANGLE);
	anticlock2.yy = cos(FULLANGLE);
	
	Tensor2 * T2address;
	Call(cudaGetSymbolAddress((void **)(&T2address),Anticlockwise2), 
		"cudaGetSymbolAddress((void **)(&T2address),Anticlockwise2)");
	Call(cudaMemcpy( T2address, &anticlock2, sizeof(f64_tens2),cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &anticlock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");
	
	f64_tens2 clock2;
	// Note that objects appearing in constant memory must have empty constructor & destructor.
	clock2.xx = cos(FULLANGLE);
	clock2.xy = sin(FULLANGLE);
	clock2.yx = -sin(FULLANGLE);
	clock2.yy = cos(FULLANGLE);
	
	Call(cudaGetSymbolAddress((void **)(&T2address),Clockwise2), 
		"cudaGetSymbolAddress((void **)(&T2address),Clockwise2)");
	Call(cudaMemcpy( T2address, &clock2, sizeof(f64_tens2),cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &clock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");
	
	//CallMAC(cudaGetSymbolAddress((void **)(&address),ReverseJzIndexStart));
	//CallMAC(cudaMemcpy( address, &numStartZCurrentRow, sizeof(long),cudaMemcpyHostToDevice));
	//long past_end = numEndZCurrentRow+1;
	//CallMAC(cudaGetSymbolAddress((void **)(&address),ReverseJzIndexEnd));
	//CallMAC(cudaMemcpy( address, &past_end, sizeof(long),cudaMemcpyHostToDevice));
	

	//// numEndZCurrentRow = numVertices-1; // the previous one.
	//// numStartZCurrentRow = numVertices-numRow[numRow1];
	
	// For floating point constants you have two choices:
	// 1. #define MAY be faster, but can only be used if no danger of
	// false match.
	// 2. __constant__. 
	// global const is not even supposed to work for integers.
	
	Set_f64_constant(FRILL_CENTROID_OUTER_RADIUS_d,pX_host->OutermostFrillCentroidRadius);
	Set_f64_constant(FRILL_CENTROID_INNER_RADIUS_d,pX_host->InnermostFrillCentroidRadius);
	Set_f64_constant(sC,sC_); // ever used?
	Set_f64_constant(kB,kB_);
	Set_f64_constant(c,c_); // ever used? likely not
	Set_f64_constant(Z,Z_);
	Set_f64_constant(e,e_);
	Set_f64_constant(q,q_);
	Set_f64_constant(m_e,m_e_);
	Set_f64_constant(m_ion,m_ion_);
	Set_f64_constant(m_n,m_n_);
	Set_f64_constant(eoverm, eoverm_);
	Set_f64_constant(qoverM, qoverM_);
	Set_f64_constant(moverM, moverM_);
	Set_f64_constant(eovermc, eovermc_);
	Set_f64_constant(qoverMc, qoverMc_);
	Set_f64_constant(FOURPI_Q_OVER_C, FOUR_PI_Q_OVER_C_);
	Set_f64_constant(FOURPI_Q, FOUR_PI_Q_);
	Set_f64_constant(FOURPI_OVER_C, FOURPI_OVER_C_);
	Set_f64_constant(NU_EI_FACTOR, NU_EI_FACTOR_);
	Set_f64_constant(nu_eiBarconst, nu_eiBarconst_);
	// Supposedly things will now be easier since device constants have the 
	// easiest labels.
	// Granted some of these could safely be #define.

	f64 temp;
	temp = 1.0/(sqrt(2.0)*2.09e7);
	Set_f64_constant(Nu_ii_Factor,temp);
	temp = sqrt(kB_*kB_*kB_);
	Set_f64_constant(kB_to_3halves, temp);
	temp = 1.0/kB_;
	Set_f64_constant(one_over_kB,temp);
	temp = temp*temp*temp;
	Set_f64_constant(one_over_kB_cubed,temp);
	temp = 1.0/sqrt(m_ion_);
	Set_f64_constant(over_sqrt_m_ion,temp);
	temp = 1.0/sqrt(m_e_);
	Set_f64_constant(over_sqrt_m_e,temp);
	temp = 1.0/sqrt(m_n_);
	Set_f64_constant(over_sqrt_m_neutral,temp);
	temp = c_*c_;
	Set_f64_constant(csq,temp);
	
	//Call(cudaGetSymbolAddress((void **)(&address), BLOCK_START_OF_FRILL_SEARCH_d),
	//		"cudaGetSymbolAddress((void **)(&address), dest )");
	//Call(cudaMemcpy( address, &BLOCK_START_OF_FRILL_SEARCH, sizeof(long),cudaMemcpyHostToDevice),
	//		"cudaMemcpy( address, &BLOCK_START_OF_FRILL_SEARCH, sizeof(long),cudaMemcpyHostToDevice) src dest");
		
//	Set_f64_constant(MAXERRPPNSQ_d, MAXERRPPNSQ);
//	Set_f64_constant(AVGFAC_d,AVGFAC);
//	Set_f64_constant(ABSTHRESHFLUX_SQ_d,ABSTHRESHFLUX_SQ);
//	Set_f64_constant(ENDPT_MAXERRPPN_SQ_d,ENDPT_MAXERRPPN_SQ);

	// These have to be set if doing that type of controlling the flux change.

	Call(cudaMemcpyToSymbol(cross_T_vals_d,cross_T_vals, 10*sizeof(f64)),
		"cudaMemcpyToSymbol(cross_T_vals_d,cross_T_vals, 10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d,cross_s_vals_viscosity_ni,
								10*sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d,cross_s_vals_viscosity_ni, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d,cross_s_vals_viscosity_nn,
								10*sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d,cross_s_vals_viscosity_nn, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_MT_ni_d,cross_s_vals_momtrans_ni,
								10*sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_MT_ni_d,cross_s_vals_momtrans_ni, \
		10*sizeof(f64))");
	
	// 1. More cudaMallocs for d/dt arrays and main data:
	// and aggregation arrays...
	
	CallMAC(cudaMalloc((void **)&p_summands,numTilesMinor*sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands,numTilesMinor*sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial,numTilesMinor*sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_scratch_d,numVertices*sizeof(f64)));
		// used for?
	CallMAC(cudaMalloc((void **)&p_nn_ionrec_minor,Syst1.Nminor*sizeof(nn)));
	
	CallMAC(cudaMalloc((void **)&p_resistive_heat_neut_minor,Syst1.Nminor*sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_resistive_heat_ion_minor,Syst1.Nminor*sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_resistive_heat_elec_minor,Syst1.Nminor*sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_MAR_neut,Syst1.Nminor*sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion,Syst1.Nminor*sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec,Syst1.Nminor*sizeof(f64_vec3)));
	
	CallMAC(cudaMalloc((void **)&p_Lapphi,Syst1.Nminor*sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_grad_phidot,Syst1.Nminor*sizeof(f64_vec2)));
	
	p_summands_host = (f64 *)malloc(numTilesMinor*sizeof(f64));
	p_Iz0_summands_host = (f64 *)malloc(numTilesMinor*sizeof(f64));
	p_Iz0_initial_host = (f64 *)malloc(numTilesMinor*sizeof(f64));
	p_scratch = (f64 *)malloc((numVertices+1000)*sizeof(f64));
	p_scratch_info = (structural *)malloc((numVertices+1000)*sizeof(structural));
	p_scratch_host = (f64 *)malloc((pX_host->Nminor+1000)*sizeof(f64));

	p_nn_host = (nn *)malloc(pX_host->Nminor*sizeof(nn));
	p_MAR_neut_host = (f64_vec3 *)malloc(pX_host->Nminor*sizeof(f64_vec3));
	p_MAR_ion_host = (f64_vec3 *)malloc(pX_host->Nminor*sizeof(f64_vec3));
	p_MAR_elec_host = (f64_vec3 *)malloc(pX_host->Nminor*sizeof(f64_vec3));
	
	// 2. cudaMemcpy system state from host: this happens always
	// __________________________________________________________
	
 	CallMAC(cudaMemcpy(Syst1.p_phi, pX_host->p_phi, numVertices*sizeof(f64), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_phidot, pX_host->p_phidot, numVertices*sizeof(f64), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_A, pX_host->p_A, Syst1.Nminor*sizeof(f64_vec3), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_Adot, pX_host->p_Adot, Syst1.Nminor*sizeof(f64_vec3), cudaMemcpyHostToDevice));
	// Transfer initial B field!
	CallMAC(cudaMemcpy(Syst1.p_B, pX_host->p_B, Syst1.Nminor*sizeof(f64), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_area, pX_host->p_area, numVertices*sizeof(f64), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_area_minor, pX_host->p_area_minor, Syst1.Nminor*sizeof(f64), cudaMemcpyHostToDevice));
		
	Syst1.InnermostFrillCentroidRadius = pX_host->InnermostFrillCentroidRadius;
	Syst1.OutermostFrillCentroidRadius = pX_host->OutermostFrillCentroidRadius;
	Systhalf.InnermostFrillCentroidRadius = Syst1.InnermostFrillCentroidRadius;
	Systhalf.OutermostFrillCentroidRadius = Syst1.OutermostFrillCentroidRadius;
	Syst2.InnermostFrillCentroidRadius = Syst1.InnermostFrillCentroidRadius;
	Syst2.OutermostFrillCentroidRadius = Syst1.OutermostFrillCentroidRadius;
	SystAdv.InnermostFrillCentroidRadius = Syst1.InnermostFrillCentroidRadius;
	SystAdv.OutermostFrillCentroidRadius = Syst1.OutermostFrillCentroidRadius;

	Syst1.EzTuning = pX_host->EzTuning; // fail?
		
	printf("Syst1.Ez %1.9E pX_host Ez %1.9E \n",
		Syst1.EzTuning,pX_host->EzTuning);
	getch();

	CallMAC(cudaMemcpy(Syst1.p_nT_neut_minor, pX_host->p_nT_neut_minor, Syst1.Nminor*sizeof(nT), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_nT_ion_minor, pX_host->p_nT_ion_minor, Syst1.Nminor*sizeof(nT), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_nT_elec_minor, pX_host->p_nT_elec_minor, Syst1.Nminor*sizeof(nT), cudaMemcpyHostToDevice));
	
	CallMAC(cudaMemcpy(Syst1.p_v_neut, pX_host->p_v_neut, Syst1.Nminor*sizeof(f64_vec3), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_v_ion, pX_host->p_v_ion, Syst1.Nminor*sizeof(f64_vec3), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_v_elec, pX_host->p_v_elec, Syst1.Nminor*sizeof(f64_vec3), cudaMemcpyHostToDevice));
	
	CallMAC(cudaMemcpy(Syst1.p_info, pX_host->p_info, numVertices*sizeof(structural), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_tri_perinfo, pX_host->p_tri_perinfo,  Syst1.Nminor*sizeof(CHAR4), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_tri_corner_index, pX_host->p_tri_corner_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.p_tri_per_neigh, pX_host->p_tri_per_neigh, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyHostToDevice));
	
	CallMAC(cudaMemcpy(Syst1.pIndexNeigh, pX_host->pIndexNeigh, numVertices*MAXNEIGH_d*sizeof(long),cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.pPBCneigh, pX_host->pPBCneigh, numVertices*MAXNEIGH_d*sizeof(char),cudaMemcpyHostToDevice));
	
	CallMAC(cudaMemcpy(Syst1.pIndexTri, pX_host->pIndexTri, numVertices*MAXNEIGH_d*sizeof(long), cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(Syst1.pPBCtri, pX_host->pPBCtri, numVertices*MAXNEIGH_d*sizeof(char), cudaMemcpyHostToDevice));
		
	CallMAC(cudaMemcpy(Syst1.p_neigh_tri_index, pX_host->p_neigh_tri_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyHostToDevice));
	
	// Now copy across to the other systems we initialized.

	CallMAC(cudaMemcpy(Systhalf.p_info, Syst1.p_info, numVertices*sizeof(structural), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.p_tri_perinfo, Syst1.p_tri_perinfo, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.p_tri_corner_index, Syst1.p_tri_corner_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.p_tri_per_neigh, Syst1.p_tri_per_neigh, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.p_neigh_tri_index, Syst1.p_neigh_tri_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	
	CallMAC(cudaMemcpy(Systhalf.pIndexTri, Syst1.pIndexTri, numVertices*MAXNEIGH_d*sizeof(long), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.pPBCtri, Syst1.pPBCtri, numVertices*MAXNEIGH_d*sizeof(char), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.pIndexNeigh, Syst1.pIndexNeigh,numVertices*MAXNEIGH_d*sizeof(long),cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Systhalf.pPBCneigh, Syst1.pPBCneigh,numVertices*MAXNEIGH_d*sizeof(char),cudaMemcpyDeviceToDevice));
	
	// Of course, this is duplicated information for the whole cycle, which makes it clear
	// that we should just have 1 copy of this really.

	CallMAC(cudaMemcpy(Syst2.p_info, Syst1.p_info, numVertices*sizeof(structural), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.p_tri_perinfo, Syst1.p_tri_perinfo, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.p_tri_corner_index, Syst1.p_tri_corner_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.p_tri_per_neigh, Syst1.p_tri_per_neigh, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.p_neigh_tri_index, Syst1.p_neigh_tri_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	
	CallMAC(cudaMemcpy(Syst2.pIndexTri, Syst1.pIndexTri, numVertices*MAXNEIGH_d*sizeof(long), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.pPBCtri, Syst1.pPBCtri, numVertices*MAXNEIGH_d*sizeof(char), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.pIndexNeigh, Syst1.pIndexNeigh,numVertices*MAXNEIGH_d*sizeof(long),cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(Syst2.pPBCneigh, Syst1.pPBCneigh,numVertices*MAXNEIGH_d*sizeof(char),cudaMemcpyDeviceToDevice));
	
	CallMAC(cudaMemcpy(SystAdv.p_info, Syst1.p_info, numVertices*sizeof(structural), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.p_tri_perinfo, Syst1.p_tri_perinfo, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.p_tri_corner_index, Syst1.p_tri_corner_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.p_tri_per_neigh, Syst1.p_tri_per_neigh, Syst1.Ntris*sizeof(CHAR4), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.p_neigh_tri_index, Syst1.p_neigh_tri_index, Syst1.Ntris*sizeof(LONG3), cudaMemcpyDeviceToDevice));
	
	CallMAC(cudaMemcpy(SystAdv.pIndexTri, Syst1.pIndexTri, numVertices*MAXNEIGH_d*sizeof(long), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.pPBCtri, Syst1.pPBCtri, numVertices*MAXNEIGH_d*sizeof(char), cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.pIndexNeigh, Syst1.pIndexNeigh,numVertices*MAXNEIGH_d*sizeof(long),cudaMemcpyDeviceToDevice));
	CallMAC(cudaMemcpy(SystAdv.pPBCneigh, Syst1.pPBCneigh,numVertices*MAXNEIGH_d*sizeof(char),cudaMemcpyDeviceToDevice));
	
	// None of these are being modified during a CUDA run cycle.
	// So what we want is another class called something like "system_structure" with only 1 object existing.
	
	printf("Done main cudaMemcpy to video memory.\n");
	
	// Let's test what we've been given:
	//FILE * fp = fopen("contribsIz.txt","w");
	f64 Iz0 = 0.0, Ne=0.0;
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pX_host->p_info[iVertex].flag == DOMAIN_VERTEX) 
		{
			Iz0 += q_*(pX_host->p_nT_ion_minor[iVertex + BEGINNING_OF_CENTRAL].n*pX_host->p_v_ion[iVertex + BEGINNING_OF_CENTRAL].z
			        - pX_host->p_nT_elec_minor[iVertex + BEGINNING_OF_CENTRAL].n*pX_host->p_v_elec[iVertex + BEGINNING_OF_CENTRAL].z)*
					pX_host->p_area[iVertex];
			
			Ne += pX_host->p_nT_elec_minor[iVertex + BEGINNING_OF_CENTRAL].n * pX_host->p_area[iVertex];
		};
	
		// save off the contribs:
		//	fprintf(fp,"%d %1.10E \n",iVertex,
		//		q_*(pX_host->p_nT_ion[iVertex].n*pX_host->p_v_ion[iVertex].z
		//	        - pX_host->p_nT_elec[iVertex].n*pX_host->p_v_elec[iVertex].z)*
		//			pX_host->p_area[iVertex]);
	};
	printf("pX_host Iz0 %1.12E Ne %1.8E IzPresc %1.12E \n",Iz0, Ne, GetIzPrescribed(t)); 
	//fclose(fp);
	
	pX1 = &Syst1;
	pXhalf = &Systhalf;
	pX2 = &Syst2;
	pXusable = &SystAdv;
	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(::Kernel_Ionisation,
							cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(::Kernel_Midpoint_v_and_Adot,
							cudaFuncCachePreferL1);
	
	// For anything else that uses < 16kB shared, it always might help
	// to set to prefer L1.
	
	// Not sure if this will help speed or just prevent 32-bit allocation:
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
				
	CallMAC(cudaMemset(p_summands,0,sizeof(f64)*numTilesMinor));
	Kernel_GetZCurrent<<<numTilesMinor,threadsPerTileMinor>>>(
		pX1->p_tri_perinfo,
		pX1->p_nT_ion_minor,
		pX1->p_nT_elec_minor,
		pX1->p_v_ion,
		pX1->p_v_elec, // Not clear if this should be nv or {n,v} yet - think.
		pX1->p_area_minor,
		p_summands
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize GetZCurrent 1.");
	
	CallMAC(cudaMemcpy(p_summands_host,p_summands,sizeof(f64)*numTilesMinor,cudaMemcpyDeviceToHost));
	Iz0 = 0.0;
	for (int ii = 0; ii < numTilesMinor; ii++)
	{
		Iz0 += p_summands_host[ii];
	};	
	printf("Iz X1 before area calc %1.12E \n",Iz0); 
	
	// 3. Advance:	
	// For now do a version where the mesh motion is done every innermost step. 
	cudaEvent_t start, stop;
	float elapsedTime;
	
	// Bring areas back to host, spit them out alongside previous:
	//CallMAC(cudaMemcpy(p_scratch,pX1->p_area,sizeof(f64)*numVertices,cudaMemcpyDeviceToHost));
	//FILE * fp = fopen("area_compare.txt","w");
	//for (iVertex = 0; iVertex < numVertices; iVertex++)
	//{
	//	fprintf(fp,"%d %d %1.14E %1.14E \n",
	//		iVertex, pX_host->p_info[iVertex].flag,
	//		pX_host->p_area[iVertex], p_scratch[iVertex]);
	//}
	//fclose(fp);
	//printf("Compared areas output...\n");
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
			
// _____________________________________________________________________________

	// Kernel calling code:

	// k1 ln 2.8 + k2 = -V
	// k1 ln 4.6 + k2 = V
	// 2V/(ln 4.6-ln 2.8) = k1
	
	f64 V = pX1->EzTuning*3.5; // 7cm assumed effective distance
	// EzTuning is what goes into E since EzShape ~= 1 near tooth. Check ?
	f64 k1 = 2.0*V/(log(4.6)-log(2.8));
	f64 k2 = V - k1*log(4.6);
	Kernel_InitialisePhi<<<numTilesMajor, threadsPerTileMajor>>>
		(
			pX1->p_info,
			k1,k2,
			V,
			pX1->p_phi
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize InitialisePhi.");
	
	//f64 tempf64;
	//cudaMemcpy(&tempf64,pX1->p_phi+10000,sizeof(f64),cudaMemcpyDeviceToHost);
	//printf("pX1->p_phi[10000] %1.9E k1 %1.5E k2 %1.5E\n"
	//	"==============================================\n",
	//	tempf64,k1,k2);
	// First thing is to see why this is zero, then see why #IND in Xhalf.
	
	
	Kernel_CalculateTriMinorAreas_AndCentroids<<<numTriTiles, threadsPerTileMinor>>>		
		(
			pX1->p_info,
			pX1->p_tri_corner_index,
			pX1->p_tri_perinfo,
		 // Output:
			pX1->p_area_minor,
			pX1->p_tri_centroid
		);
		//// Minor tiles need to carry information about what # of major values are used.
		//// And vice versa.
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalculateTriMinorAreas 1.");
	
	if (cudaPeekAtLastError() != cudaSuccess) {
		printf("cudaPALEarea %s\n",cudaGetErrorString(cudaGetLastError()));
		getch();
	} else {
		printf("Kernel_CalculateTriMinorAreas_AndCentroids No error found,\n");
	}
	
	Kernel_CalculateCentralMinorAreas<<<numTilesMajor, threadsPerTileMajor>>>(
		 pX1->p_info,
		 pX1->pIndexTri, // lists of length 12
		 pX1->p_area_minor,
		 // Output:
		 pX1->p_area_minor + BEGINNING_OF_CENTRAL);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalculateCentralMinorAreas 1.");
	
	if (cudaPeekAtLastError() != cudaSuccess) {
		printf("cudaPALEarea %s\n",cudaGetErrorString(cudaGetLastError()));
		getch();
	} else {
		printf("Kernel_CalculateCentralMinorAreas No error found,\n");
	}
	
	//CallMAC(cudaMemcpy(p_scratch_host,pX1->p_area_minor,sizeof(f64)*pX1->Nminor,cudaMemcpyDeviceToHost));
	//f64 areasum = 0.0;
	//int iTest;
	//for (iTest = 0; iTest < pX1->Ntris; iTest++)
	//{
	//	if (p_scratch_host[iTest] < 0.0) {
	//		printf("iTest %d %1.5E \n",iTest,p_scratch_host[iTest]);
	//	};
	//	areasum += p_scratch_host[iTest];
	//}
	//printf("Areasum tris only %1.12E \n",areasum); // -2500.
	//for (; iTest < pX1->Nminor; iTest++)
	//{
	//	areasum += p_scratch_host[iTest];
	//}
	//printf("Areasum %1.12E \n",areasum);

	//CallMAC(cudaMemcpy(p_scratch_host,pX1->p_area,sizeof(f64)*pX1->Nverts,cudaMemcpyDeviceToHost));
	//areasum = 0.0;
	//FILE * fp = fopen("oldareas.txt","w");
	//for (int iTest = 0; iTest < pX1->Nverts; iTest++)
	//{
	//	areasum += p_scratch_host[iTest];
	//	fprintf(fp,"%d %1.15E\n",iTest,p_scratch_host[iTest]);
	//}
	//fclose(fp);
	//printf("Areasum_major old %1.12E \n",areasum);

	Kernel_CalculateMajorAreas<<<numTilesMajor,threadsPerTileMajor>>>(
			pX1->p_info,
			pX1->p_tri_centroid,
			pX1->pIndexTri,
			pX1->pPBCtri,
			pX1->p_area
			);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalculateMajorAreas");
	
	FILE * fp = fopen("newareas.txt","w");
	CallMAC(cudaMemcpy(p_scratch_host,pX1->p_area,sizeof(f64)*pX1->Nverts,cudaMemcpyDeviceToHost));
	f64 areasum = 0.0;
	for (int iTest = 0; iTest < pX1->Nverts; iTest++)
	{
		areasum += p_scratch_host[iTest];
		fprintf(fp,"%d %1.15E\n",iTest,p_scratch_host[iTest]);
	}
	fclose(fp);
	printf("Areasum_major II %1.12E \n",areasum);
	
	
	// The number of triangles will not be exactly numTriTiles*threadsPerTileMinor.
	// THAT WOULD BE VERY BAD NEWS: It means that the array has a hole in it!!
		
	// NEED geometry to give exact # tris and ideally # vertices also.
	// However, #tris = #firstrow + #lastrow + 2* sum of other #in_row
	// On that it looks unlikely that we'd happen to achieve a multiple of 128 if #tris % 256=0;
	// Bear in mind.
	// Alternative would be to put tris hanging off first row and outside last row.
	// This doesn't serve an obvious purpose and causes aggro: what is corner index in meaningless tri.
	// But it might actually make getting a #tris easier.
	
	// Central cells use a bigger shared memory footprint by looking at tri data with indices.
	// So we probably don't want larger blocks for them, as large as tri blocks.
	// Makes more sense to put tris in blocks
	
	
	// ***  Document design decisions and reasons. --- Weds
	
	// Here's a problem: we have said that we need to start minor block at 2* start index
	// Otherwise ... maybe it would still work if we loaded a start point into shared data,
	// but we do not know.
	// We can only load consecutive values anyway, basically. We could have an edge fetch
	// I suppose. Which would only complicated matters.
	// But it probably won't recognise contiguous access if there is a variable in the index???
	// It might or might not. I'm not sure.

	// However for GetZCurrent we can use large blocks --- this may overrun the total array
	// of minors, just put a test in there to stop it.
			
	CallMAC(cudaMemset(p_summands,0,sizeof(f64)*numTilesMinor));
	
	Kernel_GetZCurrent<<<numTilesMinor,threadsPerTileMinor>>>(
		pX1->p_tri_perinfo,
		pX1->p_nT_ion_minor,
		pX1->p_nT_elec_minor,
		pX1->p_v_ion,
		pX1->p_v_elec, // Not clear if this should be nv or {n,v} yet - think.
		pX1->p_area_minor,
		p_summands
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize GetZCurrent 1.");
	// To get current, just take n v area in each minor cell and central cell.
	// No reason we would not know areas already but nvm that.
	
	// HMMMM
	// How to handle nT data?
	
	CallMAC(cudaMemcpy(p_summands_host,p_summands,sizeof(f64)*numTilesMinor,cudaMemcpyDeviceToHost));
	Iz0 = 0.0;
	int ii;
	for (ii = 0; ii < numTriTiles; ii++)
	{
		Iz0 += p_summands_host[ii];
	};	
	printf("Iz after tri areas %1.12E \n",Iz0);
	for (; ii < numTilesMinor; ii++)
	{
		Iz0 += p_summands_host[ii];
	};	
	printf("Iz after areas %1.12E \n",Iz0);
	getch();


	// DEBUG  :
	// ========

	::Kernel_Populate_A_frill<<<numTriTiles, threadsPerTileMinor>>>
		(
			pX1->p_tri_perinfo,
			pX1->p_A, // update own, read others
			pX1->p_tri_centroid,
			pX1->p_neigh_tri_index
		);

	::Kernel_Compute_Lap_A_and_Grad_A_to_get_B_on_all_minor<<<numTriTiles, threadsPerTileMinor>>>
		(
			pX1->p_A,
			pX1->p_A + BEGINNING_OF_CENTRAL,
			pX1->p_info, // does this make it work ?
			pX1->p_tri_centroid,
			pX1->p_tri_perinfo,
			pX1->p_tri_per_neigh,
			pX1->p_tri_corner_index,
			pX1->p_neigh_tri_index,
			pX1->pIndexTri,
			pX1->p_Lap_A,
			pX1->p_Lap_A + BEGINNING_OF_CENTRAL,
			pX1->p_B,
			pX1->p_B + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute Lap A I");

	::Kernel_Compute_grad_phi_Te_tris<<<numTriTiles, threadsPerTileMinor>>>
		(
		pX1->p_info,
		pX1->p_phi,     // on majors
		pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // on majors
		pX1->p_tri_corner_index,
		pX1->p_tri_perinfo,
		pX1->p_grad_phi,
		pX1->p_GradTe
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi tri I");
	
	::Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor, threadsPerTileMajor>>>
		(
		pX1->p_info,
		pX1->p_phi,
		pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
		pX1->pIndexNeigh,
		// output:
		pX1->p_grad_phi + BEGINNING_OF_CENTRAL,
		pX1->p_GradTe + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi central I");
	// CHECK PARAMETERS <<< >>> 
			
	// Get thermal pressure on each accelerating region...
	// Better off probably to do the ionisation stage first, it will give a better idea
	// of the half-time thermal pressure we are ultimately aiming for.
	::Kernel_GetThermalPressureTris<<<numTriTiles,threadsPerTileMinor>>>
		( 
		pX1->p_info,			
		pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
		pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
		pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
		pX1->p_tri_corner_index,
		pX1->p_tri_perinfo,
		p_MAR_neut,
		p_MAR_ion,
		p_MAR_elec
		);
	// So far it only works on DOMAIN_TRIANGLE, CROSSING_INS gets 0.
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure tris");
	
	Kernel_GetThermalPressureCentrals<<<numTilesMajor,threadsPerTileMajor>>>
		(
		pX1->p_info,
		pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
		pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
		pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
		pX1->pIndexNeigh,
		p_MAR_neut + BEGINNING_OF_CENTRAL,
		p_MAR_ion + BEGINNING_OF_CENTRAL,
		p_MAR_elec + BEGINNING_OF_CENTRAL
		); // works on DOMAIN_VERTEX only
	Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure");
	
	printf("done GTPC\n");
		
	// End debug
	
	SendToHost(pX1, pX1, pX_host);
	pX_host->AsciiOutput("inputs_pX1.txt");
	printf("done ascii output of pX1\n\n");
	getch();

	int iSubstep;
	for (iSubstep = 0; iSubstep < numSubsteps; iSubstep++)
	{
		printf("Step %d / %d : ",iSubstep,numSubsteps);

		// First set up Iz_presc_1/2 etc:
	
		f64 thalf = t + hstep*0.5;
		// Set ReverseJz before each call to Advance Potentials.
		f64 Iz_prescribed = GetIzPrescribed(thalf);
		f64 fourpioverc_reverse_Jz = -FOUR_PI_OVER_C_*Iz_prescribed/(f64)(Syst1.numReverseJzTris);
		Set_f64_constant(four_pi_over_c_ReverseJz,fourpioverc_reverse_Jz);
		// thalf because we use it to advance dA/dt from k to k+1, via ReverseJz.
		// HOWEVER, we also aim for Iz_prescribed
		// So we need to set up 2 different variables.
		Iz_prescribed = GetIzPrescribed(t+hstep);
		
		::Kernel_Average_nT_to_tri_minors<<<numTriTiles,threadsPerTileMinor>>>(
			//pX1->p_info,
			pX1->p_tri_corner_index,
			pX1->p_tri_perinfo,
			pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL, 
			pX1->p_nT_neut_minor, pX1->p_nT_ion_minor,	pX1->p_nT_elec_minor			
			);
		// If one of the corners is an outermost then outermost n should be pop'd with benign value.
		// At insulator-crossing tri we require having set n = 0 inside insulator.
		
		Kernel_GetZCurrent<<<numTilesMinor,threadsPerTileMinor>>>(
			pX1->p_tri_perinfo,
			pX1->p_nT_ion_minor,
			pX1->p_nT_elec_minor,
			pX1->p_v_ion,
			pX1->p_v_elec, // Not clear if this should be nv or {n,v} ? {n,v}
			pX1->p_area_minor,
			p_summands	);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize GetZCurrent k.");
		CallMAC(cudaMemcpy(p_summands_host,p_summands,sizeof(f64)*numTilesMinor,cudaMemcpyDeviceToHost));
		Iz0 = 0.0;
		for (int ii = 0; ii < numTilesMinor; ii++)
		{
			Iz0 += p_summands_host[ii];
		};	
		printf("Iz t_k: %1.12E \n",Iz0);
		
		// We are going to actually try using nv, nT.
		
		// Mesh and fluid advection, h/2:
		// ==============================
		
		// 1. Create v_overall, on major cells (use central) and
		
		Kernel_Create_v_overall_and_newpos<<<numTilesMajor,threadsPerTileMajor>>>(
			pX1->p_info,
			hstep*0.5,
			
			pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL, 
			
			pX1->p_v_neut + BEGINNING_OF_CENTRAL,
			pX1->p_v_ion + BEGINNING_OF_CENTRAL,
			pX1->p_v_elec + BEGINNING_OF_CENTRAL, // central v
			
			pXhalf->p_info,
			pX1->p_v_overall + BEGINNING_OF_CENTRAL // make it for everything		
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Create_v_overall.");
		
		// 2. transmit to minor : this is the mesh move v.
		//   ^^^ tri, function of central
		::Kernel_Average_v_overall_to_tris<<<numTriTiles,threadsPerTileMinor>>>(
			pX1->p_tri_corner_index,
			pX1->p_tri_perinfo,	
			pX1->p_v_overall + BEGINNING_OF_CENTRAL, // major v_overall
			pX1->p_tri_centroid,
			pX1->p_v_overall
			); // so motion will take place relative to this velocity.
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Create_v_overall.");
		// Tri centroid was set up when?
		// Needs to be done for 1st system and thereafter.
		
		// Extra steps because fields unmoving :
		// 3. anti-advect phi, phidot [Lap phi advances dphi/dt also -- inconsistent with rest of scheme]
		//    + advance phi to half-time using updated phidot
		//   ^^^ major, function of major  but Lap phi uses edges from tri centroid
		
		// What we are supposed to do:
		// * phidot requires Lap_phi_k and rho_k to advance.
		// * phi uses the resulting phidot to advance to phi_half
		//  * We need grad phi and grad phidot, on major, to do anti-advect
		
		// * A uses dA/dt_k to advance to A_half
		//  * We need grad A and grad Adot, on all, to do anti-advect.
				
		Get_Lap_phi_on_major<<<numTilesMajor,threadsPerTileMajor>>>
			(
			pX1->p_phi,
			pX1->p_info,
			pX1->pIndexNeigh, // neighbours of vertices
			pX1->pPBCneigh, // rel periodic orientation of vertex neighbours
			p_Lapphi
			); // == 0 for INNERMOST & OUTERMOST
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Get_Lap_phi_on_major.");
		
		::Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor,threadsPerTileMajor>>>(
			pX1->p_info,
			pX1->p_phidot,   // phidot is always for major
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pX1->pIndexNeigh,
			p_grad_phidot + BEGINNING_OF_CENTRAL,
			pX1->p_GradTe +  BEGINNING_OF_CENTRAL
			); 
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Compute_grad_phi_centrals.");
		// sets == 0 outside of DOMAIN_VERTEX
		
		// Having created grad phi, we then want to use it to do the anti-move of phi_vertex.
		Kernel_Advance_Antiadvect_phidot<<<numTilesMajor,threadsPerTileMajor>>>(
				pX1->p_phidot,	
				pX1->p_v_overall + BEGINNING_OF_CENTRAL, // !!! NOTE BENE
				hstep*0.5,
				p_grad_phidot + BEGINNING_OF_CENTRAL, // on majors please
				p_Lapphi,
				pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
				pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // --> rho _k
				pXhalf->p_phidot
				// This is just a ton of loading and a simple formula --
				// we should prefer to combine with Get_Lap_phi routine.
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Advance_Antiadvect_phidot.");
		//
		//::Kernel_Compute_grad_phi_Te_tris<<<numTriTiles,threadsPerTileMinor>>>(
		//	pX1->p_info,
		//	pX1->p_phi, // NOTE BENE
		//	pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
		//	pX1->p_tri_corner_index,
		//	pX1->p_tri_perinfo,
		//	pX1->p_grad_phi // NOTE BENE --- this is for minors
		//	);
		//Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute_grad_phi_on_tris.");
		//
		Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor,threadsPerTileMajor>>>(
			pX1->p_info,
			pX1->p_phi,
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pX1->pIndexNeigh,
			pX1->p_grad_phi + BEGINNING_OF_CENTRAL, // NOTE BENE	
			pX1->p_GradTe + BEGINNING_OF_CENTRAL			
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Compute_grad_phi_centrals.");
		// sets == 0 outside of DOMAIN_VERTEX
		
		f64 Vhalf = pX1->EzTuning*3.5*(GetIzPrescribed(thalf)/GetIzPrescribed(t));
		printf("EzTuning = %1.5E , V = %1.15E Vhalf = %1.15E \n",pX1->EzTuning,V,Vhalf);	// guesstimate 

		getch();

		Kernel_Advance_Antiadvect_phi<<<numTilesMajor,threadsPerTileMajor>>>
			(
				pX1->p_info, // for innermost & outermost, set = +-V:,
				Vhalf,
				pX1->p_phi,
				pX1->p_v_overall + BEGINNING_OF_CENTRAL, // SHOULD THIS NEED + numTris? Most obvious way is yes.
				hstep*0.5,
				pX1->p_grad_phi + BEGINNING_OF_CENTRAL,
				pXhalf->p_phidot	, // Think I'm correct to say, we use updated phidot here.
				pXhalf->p_phi
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Antiadvect_phi.");
		
		::Kernel_Populate_A_frill<<<numTriTiles, threadsPerTileMinor>>>
			(
				pX1->p_tri_perinfo,
				pX1->p_A, // update own, read others
				pX1->p_tri_centroid,
				pX1->p_neigh_tri_index
			);
		// Note that if we are not sharing tri info for major tile, we can simply run with bigger blocks for major.
		// Only if we concatenate several neighbouring "tiles" that we organised.
		
		// 4. anti-advect A, dA/dt    [Advance A to half-time also]
		
		// _ Kernel_Compute_Lap_A_and_Grad_A_to_get_B_on_all_minor
		
		// Simpler(?) way: 2 separate kernels, large reload of grad A.
		// Better way: have a switch to apply results during same kernel.
		
		// Adot:
		Kernel_Compute_Grad_A_minor_antiadvect<<<numTriTiles,threadsPerTileMinor>>>(
			pX1->p_Adot,        // for creating grad
			pX1->p_Adot + BEGINNING_OF_CENTRAL,
			hstep*0.5,
			pX1->p_v_overall,    // hv = amt to anti-advect
			pX1->p_info,       // 
			pX1->p_tri_centroid, // 
			pX1->p_tri_perinfo,     // 
			pX1->p_tri_per_neigh,
			pX1->p_tri_corner_index,    // 
			pX1->p_neigh_tri_index, // 
			pX1->pIndexTri,         // we said carry on using this for now.
			false,
			0,
			// output:
			pXhalf->p_Adot// fill in for both tri and vert...			
			);
		
		if (cudaPeekAtLastError() != cudaSuccess) {
			printf("Kernel_Compute_Grad_A_minor_antiadvect %s\n",cudaGetErrorString(cudaGetLastError()));
		} else {
			printf("Kernel_Compute_Grad_A_minor_antiadvect No error found at invoc,\n");
		}

		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Antiadvect Adot.");
		if (cudaPeekAtLastError() != cudaSuccess) {
			printf("Kernel_Compute_Grad_A_minor_antiadvect ___ %s\n",cudaGetErrorString(cudaGetLastError()));
		} else {
			printf("Kernel_Compute_Grad_A_minor_antiadvect No error found at synchr,\n");
		}


		Kernel_Compute_Grad_A_minor_antiadvect<<<numTriTiles,threadsPerTileMinor>>>(
			pX1->p_A,        // for creating grad
			pX1->p_A + BEGINNING_OF_CENTRAL,
			hstep*0.5,
			pX1->p_v_overall,    // hv = amt to anti-advect
			pX1->p_info,       // 
			pX1->p_tri_centroid, // 
			pX1->p_tri_perinfo,     //
			pX1->p_tri_per_neigh,
			pX1->p_tri_corner_index,    // 
			pX1->p_neigh_tri_index, // 
			pX1->pIndexTri,         // we said carry on using this for now.
			
			true,
			pX1->p_Adot, // add h*0.5*Adot to A ...
			// output:
			pXhalf->p_A// fill in for both tri and vert...			
			);
		
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Antiadvect A etc.");
		
		// Get this right: Why we like to save Lap A half -- because it
		// contributes to advancing Adot during the main step.
		// So we have to be careful here and call for Lap A again AFTER
		// we advance A with Adot to halfway.
		// IS THAT NOT an equally good occasion to do the antiadvect of A?
		// i.e. sequence:
		
		// 1. Add h/2 * Adot to give A_half [at old positions] in X1
		
		// 2. Take grad Adot_k[old pos] -> Antiadvect Adot to
		//		pXhalf->Adot_k[new pos] = still the time k value but in new positions
		
		// 3. Take Lap A_half[old pos], grad A_half to get Ahalf[new positions]
		
		// A bit messed up:		
		// . We are taking Lap A at the old positions and want to apply it at new.
		
		// So let's do it the simple way first that involves 3 calls.
		// Then we can come back and experiment with this.
		


		// Clarity here : A needs to advance with A-dot_k.
		//  ( Adot does not need to advance until we do the midpt step )
		
		// That is the end of field evolution & mesh & species advection, 1st part.
		
		// =========================================================================
		
		
		// Old chat:
		
		// Well think again as regards nT. It's easily inferred from having n on same place.
		// It's the case that having nv creates certain amt of heat, but we can deal with heat via T given n.
		
		// But hang on a minute.
		// If we are given ns*vs and we need to know ns*(vs-v) then how do we get that?
		
		// Feels like breaking down and using {n,v} here is better after all.		
		// We need to be jolly careful about this.
		// How we estimate n on minor:
		// ===========================
		// 
		// We want n for central = n for major say. Apportion the rest of the mass so
		//   n_tri_cell = Sum (Area_intersection * n_major )/(area_tri_cell)
		// This means that they add back up -- correct?
		// We'll get mass_tri_cell = sum (area_intersection * n_major)
		// mass total for each major cell then = n_major * area_major
		
		// OK so that is a very simple way.
		
		// Let's stick with the {n,v,T} so popular up til now.
		// BUT,
		// what then do we do to ensure conservation of say Nv when we do advection of nv?
		
		// We know how much "Nv" is in the cell at the start .. ?
		// Simple way:
		// v is constant in edge cell;
		// therefore Nv = n.v.area
		
		// How do we know how much we have to finish off with?
		// Do N,NT advection first.
		// Same formula: and n_new = lc : Sum (Area_intersection * n_major )/(area_tri_cell)
		// We should allow that a certain amount of momentum has flowed in.
		// Then we choose v = (arrived mom)/(k+1 mass) to give the req amt of momentum.
		// So really there isn't a powerful reason to use nv ... even with use in corrector method it really makes no difference.
				
		// 5. Calculate half-time minor areas and estimated densities.

		// GOT TO CREATE TRI CENTROIDS BEFORE WE CREATE AREAS:

		::Kernel_CalculateTriMinorAreas_AndCentroids<<<numTriTiles,threadsPerTileMinor>>>
			(
			pXhalf->p_info,
			pXhalf->p_tri_corner_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_area_minor,
			pXhalf->p_tri_centroid
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalcMinorAreas+Centroids.");
		
		Kernel_CalculateMajorAreas<<<numTilesMajor,threadsPerTileMajor>>>(
			pXhalf->p_info,
			pXhalf->p_tri_centroid,
			pXhalf->pIndexTri,
			pXhalf->pPBCtri,
			pXhalf->p_area
			);
		// Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Advance_A_with_Adot.");
		
		::Kernel_CalculateCentralMinorAreas<<<numTilesMajor,threadsPerTileMajor>>>( // central areas
			pXhalf->p_info, // used how?
			pXhalf->pIndexTri,
			pXhalf->p_area_minor,
			pXhalf->p_area_minor + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalcCentralMinorAreas, MajorAreas.");
		
		// 6. Rel advection for each species:
		// 6a. Density & heat advection based on edge cells nv vs the move rate of the wall.
		//    ^^^ major cells sharing both major cell N,NT,area
		//        and tri cell nv, centroid --- correct?
		
		Kernel_RelAdvect_nT<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pX1->p_info, 
			pX1->pIndexTri,
			//pX1->pPBCtri,
			pX1->p_tri_centroid,
			pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_neut_minor,
			pX1->p_nT_ion_minor,    
			pX1->p_nT_elec_minor,
			pX1->p_v_neut,  // should always be minor...
			pX1->p_v_ion,
			pX1->p_v_elec,
			pX1->p_v_overall, 
			pX1->p_area,
			pXhalf->p_area,
			// dest:
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL
			// Consider: {n,v,T} = 5 vars. One more is the magic number.
			// It is probably not the end of the world if we split into 2's and 3's, nT vs v.
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect nT pXhalf");

		::Kernel_Average_nT_to_tri_minors<<<numTriTiles,threadsPerTileMinor>>>(
										pXhalf->p_tri_corner_index,
										pXhalf->p_tri_perinfo, 
										pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
										pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
										pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
										pXhalf->p_nT_neut_minor,
										pXhalf->p_nT_ion_minor,
										pXhalf->p_nT_elec_minor);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize avg nT pXhalf");

			
		// 6b. Momentum advection for minor cells..
		//  2 kinds of walls: central-to-tri and tri-to-tri.
		
		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_nT_neut_minor,   // -> momentum input
			pXhalf->p_nT_neut_minor, // destination n needed to divide Nv
			pX1->p_v_overall,
			pX1->p_v_neut,         // -> momentum input
			pX1->p_tri_centroid,
			pX1->p_tri_corner_index,
			pX1->p_neigh_tri_index,
			pX1->p_tri_perinfo,
			pX1->p_tri_per_neigh,  // ? does it need to exist?
			pX1->p_area_minor,
			pXhalf->p_area_minor,
			pXhalf->p_v_neut      // output
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri neut");
		
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_tri_centroid,
			pX1->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_neut_minor,
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pX1->p_v_neut,
			pX1->p_v_overall,
			pX1->pIndexTri,
			pX1->pPBCtri,
			pX1->p_area,
			pXhalf->p_area,
			pXhalf->p_v_neut + BEGINNING_OF_CENTRAL 
			);		

		if (cudaPeekAtLastError() != cudaSuccess) {
			printf("Kernel_Rel_advect_v_central %s\n",cudaGetErrorString(cudaGetLastError()));
		} else {
			printf("Kernel_Rel_advect_v_central No error found at invoc,\n");
		}

		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent neut");
		
		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_nT_ion_minor,
			pXhalf->p_nT_ion_minor,
			pX1->p_v_overall,
			pX1->p_v_ion,
			pX1->p_tri_centroid,
			pX1->p_tri_corner_index,
			pX1->p_neigh_tri_index,
			pX1->p_tri_perinfo,
			pX1->p_tri_per_neigh,
			pX1->p_area_minor,
			pXhalf->p_area_minor,
			pXhalf->p_v_ion
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri ion");
		
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_tri_centroid,
			pX1->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_ion_minor,
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pX1->p_v_ion,
			pX1->p_v_overall,
			pX1->pIndexTri,
			pX1->pPBCtri,
			pX1->p_area,
			pXhalf->p_area,
			pXhalf->p_v_ion + BEGINNING_OF_CENTRAL 
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent ion");
		
		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_nT_elec_minor,
			pXhalf->p_nT_elec_minor,
			pX1->p_v_overall,
			pX1->p_v_elec,
			pX1->p_tri_centroid,
			pX1->p_tri_corner_index,
			pX1->p_neigh_tri_index,
			pX1->p_tri_perinfo,
			pX1->p_tri_per_neigh,
			pX1->p_area_minor,
			pXhalf->p_area_minor,
			pXhalf->p_v_elec
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri elec");
		
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pX1->p_info,
			pX1->p_tri_centroid,
			pX1->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pX1->p_nT_elec_minor,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pX1->p_v_elec,
			pX1->p_v_overall,
			pX1->pIndexTri,
			pX1->pPBCtri,
			pX1->p_area,
			pXhalf->p_area,
			pXhalf->p_v_elec + BEGINNING_OF_CENTRAL 
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent elec");
		/*
		cudaMemcpy(pX_host->p_nT_ion_minor,			pX1->p_nT_ion_minor,
			sizeof(nT)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(pX_host->p_nT_elec_minor,			pXhalf->p_nT_ion_minor,
			sizeof(nT)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(pX_host->p_v_ion,			pX1->p_v_ion,
			sizeof(f64_vec3)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(pX_host->p_v_elec,			pXhalf->p_v_ion,
			sizeof(f64_vec3)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(pX_host->p_area_minor,			pXhalf->p_area_minor,
			sizeof(f64)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize memcpies");

		printf("43654: n_ion %1.10E  %d : %1.10E \n",pX_host->p_nT_elec_minor[43654].n,
			BEGINNING_OF_CENTRAL + 20000, pX_host->p_nT_elec_minor[BEGINNING_OF_CENTRAL + 20000].n);
		

		FILE * file = fopen("inputs0.txt","w");
		fprintf(file,"index | pX1_v_ion_x y z | pXhalf_v_ion_x y z | 1n 1T n T \n");
		
		for (int iMinor = 0; iMinor < Syst1.Nminor; iMinor++)
		{
			fprintf(file,"%d %d | %1.10E %1.10E %1.10E | %1.10E %1.10E %1.10E | "
				"%1.10E %1.10E %1.10E %1.10E | %1.10E \n",
				iMinor,pX_host->p_tri_perinfo[iMinor].flag,
				pX_host->p_v_ion[iMinor].x,pX_host->p_v_ion[iMinor].y,pX_host->p_v_ion[iMinor].z,
				pX_host->p_v_elec[iMinor].x,pX_host->p_v_elec[iMinor].y,pX_host->p_v_elec[iMinor].z,
				pX_host->p_nT_ion_minor[iMinor].n,pX_host->p_nT_ion_minor[iMinor].T,
				pX_host->p_nT_elec_minor[iMinor].n,pX_host->p_nT_elec_minor[iMinor].T,
				pX_host->p_area_minor[iMinor]);
		};
		cudaMemcpy(pX_host->p_area_minor,			pX1->p_area_minor,
			sizeof(f64)*Syst1.Nminor,
			cudaMemcpyDeviceToHost);
		for (int iMinor = 0; iMinor < Syst1.Nminor; iMinor++)
		{
			fprintf(file,"%d %1.10E \n",iMinor,pX_host->p_area_minor[iMinor]);
		};
		fclose(file);*/
		
		// RESULT SO FAR: n, T look normal. v_ion is OK in X1, becomes IND/INF in Xhalf.
		// From 85392 onwards, it's 0,0,large . To 73532 : IND=xy, INF=z.
		
		// ============================================================================
		// Now do estimates for half-time system ready for midpoint calls:
		
		// Get Grad phi _half, etc, for each minor cell:
		
		cudaMemset(p_MAR_neut,0,sizeof(f64_vec3)*pX1->Nminor);
		cudaMemset(p_MAR_ion,0,sizeof(f64_vec3)*pX1->Nminor);
		cudaMemset(p_MAR_elec,0,sizeof(f64_vec3)*pX1->Nminor);

		//FILE * fp = fopen("tri_data.txt","w");
		//for (int iii = 0; iii < Syst1.Ntris; iii++)
		//{
		//	fprintf(fp,"%d %d %d %d\n",iii,pX_host->p_tri_corner_index[iii].i1,
		//		pX_host->p_tri_corner_index[iii].i2,
		//		pX_host->p_tri_corner_index[iii].i3);
		//}
		//fclose(fp);
		//cudaMemcpy(pX_host->p_tri_corner_index,pX1->p_tri_corner_index,
		//	sizeof(LONG3)*Syst1.Ntris,
		//	cudaMemcpyDeviceToHost
		//	);
		//fp = fopen("tri_data2.txt","w");
		//for (int iii = 0; iii < Syst1.Ntris; iii++)
		//{
		//	fprintf(fp,"%d %d %d %d\n",iii,pX_host->p_tri_corner_index[iii].i1,
		//		pX_host->p_tri_corner_index[iii].i2,
		//		pX_host->p_tri_corner_index[iii].i3);
		//}
		//fclose(fp);

		::Kernel_Populate_A_frill<<<numTriTiles, threadsPerTileMinor>>>
			(
				pXhalf->p_tri_perinfo,
				pXhalf->p_A, // update own, read others
				pXhalf->p_tri_centroid,
				pXhalf->p_neigh_tri_index
			);

		::Kernel_Compute_Lap_A_and_Grad_A_to_get_B_on_all_minor<<<numTriTiles, threadsPerTileMinor>>>
			(
				pXhalf->p_A,
				pXhalf->p_A + BEGINNING_OF_CENTRAL,
				pXhalf->p_info, // does this make it work ?
				pXhalf->p_tri_centroid,
				pXhalf->p_tri_perinfo,
				pXhalf->p_tri_per_neigh,
				pX1->p_tri_corner_index,
				pXhalf->p_neigh_tri_index,
				pXhalf->pIndexTri,
				pXhalf->p_Lap_A,
				pXhalf->p_Lap_A + BEGINNING_OF_CENTRAL,
				pXhalf->p_B,
				pXhalf->p_B + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute Lap A I");

		::Kernel_Compute_grad_phi_Te_tris<<<numTriTiles, threadsPerTileMinor>>>
			(
			pXhalf->p_info,
			pXhalf->p_phi,     // on majors
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // on majors
			pXhalf->p_tri_corner_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_grad_phi,
			pXhalf->p_GradTe
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi tri I");
		
		::Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor, threadsPerTileMajor>>>
			(
			pXhalf->p_info,
			pXhalf->p_phi,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->pIndexNeigh,
			// output:
			pXhalf->p_grad_phi + BEGINNING_OF_CENTRAL,
			pXhalf->p_GradTe + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi central I");
		// CHECK PARAMETERS <<< >>> 
				
		// Get thermal pressure on each accelerating region...
		// Better off probably to do the ionisation stage first, it will give a better idea
		// of the half-time thermal pressure we are ultimately aiming for.
		::Kernel_GetThermalPressureTris<<<numTriTiles,threadsPerTileMinor>>>
			( 
			pXhalf->p_info,			
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_tri_corner_index,
			pXhalf->p_tri_perinfo,
			p_MAR_neut,
			p_MAR_ion,
			p_MAR_elec
			);
		// So far it only works on DOMAIN_TRIANGLE, CROSSING_INS gets 0.
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure tris");
		
		Kernel_GetThermalPressureCentrals<<<numTilesMajor,threadsPerTileMajor>>>
			(
			pXhalf->p_info,
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->pIndexNeigh,
			p_MAR_neut + BEGINNING_OF_CENTRAL,
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL
			); // works on DOMAIN_VERTEX only
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure");
		
		printf("done GTPC\n");
		
		
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// It might be reasonable to instead be using TRIANGLE nT in getting
		// thermal pressure on centrals? You'd think so.
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		// This should be pretty similar to grad phi, and like grad phi needs to
		// apply to every minor cell.
		
		// nv:
		// Would we also wish to switch to nT? Because this reflects the actual
		// amount of heating that is happening, and is then conserved.
		// Again, to get grad T, we load T into shared memory by dividing.
		
		// We choose v because we need to be taking grad v ?
		// Or we'd need estimated n to be picked up at the same time. But that would be OK.
		// No - not as good: it involves divisions...
		
		// -----------------------------------------------------
		// Now ionisation , accel & heating
		
		Kernel_Ionisation<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep,
			pXhalf->p_info,
			pXhalf->p_area, // correct input? used?
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL, // src 
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,		
			0,0,0,
	
			// No output, I think now, except nn_ionrec.			
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL, 
			0 // b2ndpass  --  ??
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Ionisation 1");

		// What do we do with information about momentum changing? Or do we only need nnionrec?

		Kernel_Average_nnionrec_to_tris<<<numTriTiles,threadsPerTileMinor>>>
			(
			pXhalf->p_tri_perinfo,
			pXhalf->p_tri_corner_index,
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL,
			p_nn_ionrec_minor
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize splitout nn");
		
		// OKAY what we see is that nn_ionrec is needed in centrals and we want to set it to the
		// major value.
		// Therefore nn_ionrec logically is organised with major/central values at the end.
		
		// Now run midpoint v step on minor cells.
		
		printf("about to do midpoint.\n");
		getch();
		
		// I think for debugging it would be good here to dump all the inputs to host
		// and spit it out to a spreadsheet.
		
		f64 temp1, temp2;
		nT nTtemp3, nTtemp4;

		//cudaMemcpy(&temp1,&(pX1->p_phi[10000]),sizeof(f64),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&temp2,&(pXhalf->p_phi[10000]),sizeof(f64),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&nTtemp3,&(pX1->p_nT_elec_minor[89000]),sizeof(nT),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&nTtemp4,&(pXhalf->p_nT_elec_minor[89000]),sizeof(nT),cudaMemcpyDeviceToHost);
		//
		//printf("phi[10000] %1.9E %1.9E \nTe[89000] %1.5E %1.5E\n",
		//	temp1,temp2,nTtemp3.T,nTtemp4.T);
		//getch();
		
		SendToHost(pXhalf, pXhalf, pX_host);		
		pX_host->AsciiOutput("Inputs_half.txt");
		
		printf("start midpt step:\n");
		pXhalf->evaltime = pX1->evaltime + 0.5*hstep;
		Kernel_Midpoint_v_and_Adot<<<numTilesMinor,threadsPerTileMinor>>>
			(
				hstep,
				pXhalf->p_tri_perinfo,
				pXhalf->p_nT_neut_minor, // src
				pXhalf->p_nT_ion_minor, 
				pXhalf->p_nT_elec_minor, 
				// Both n_k and n_k+1 appear in the midpt formula, so we need n_k.
				
				//pXhalf->p_nT_neut, // k or k+1/2
				//pXhalf->p_nT_ion, // k or k+1/2
				//pXhalf->p_nT_elec, // k or k+1/2 --- for forming nu etc...
				0,0,0, // on b2ndPass == 0 we'll try not loading.
				
				p_nn_ionrec_minor, // Have to load 2 additional doubles due to doing ionisation outside.
							
				pXhalf->p_tri_centroid, // Defined ever?
				pXhalf->p_info,   // were these positions actually created?
				
				pXhalf->p_B,
				pXhalf->p_v_neut, // Do not update: we need v_k again.
				pXhalf->p_v_ion,
				pXhalf->p_v_elec,
				// Thing is, we have to create 0.5*(v_k+v_k+1) on 1st pass.
				// We want to leave v_k[advected] intact, so no update here.
				// We want to go again from v_k on 2nd and 3rd pass.
					
				pXhalf->p_area_minor,	// It's assumed to be area_k+1 but I guess it's area_k+1/2 ... too bad?
								// THIS MATTERS !!
								// I think area_1/2 is relevant both times in midpt but need to check that.
				pXhalf->p_grad_phi,
				pXhalf->p_Lap_A, // check this input?
				pXhalf->p_Adot,  // anti-advected Adot_k ...
				p_MAR_neut,
				p_MAR_ion,
				p_MAR_elec, // assume take integral(-grad(nT))/m_s
				pXhalf->p_GradTe,
						
				// output: ( Here interim values of v )
				pXusable->p_v_neut,
				pXusable->p_v_ion,
				pXusable->p_v_elec,
				p_resistive_heat_neut_minor, 
				p_resistive_heat_ion_minor,
				p_resistive_heat_elec_minor,
				pXusable->p_Adot,
				false, // 1st pass
				pX1->EzTuning, // put what here?
 				p_Iz0_summands,
				p_summands
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize MidptAccel 1");
		
		printf("midpt 1 done.");
		getch();

		// The amount of resistive heating depends on Ez of course...
		// but we don't want to have to run twice at this juncture.
		// Therefore?		
	
		// . The heating routine also cements the effects of ionisation on n.
		// . Assume central ionisation == vertcell ionisation.
		Kernel_Heating_routine<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep,
			pXhalf->p_info,
			pXhalf->pIndexTri, // fetch htg amounts from minor cells...
			
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,  
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL,
			
			pXhalf->p_B + BEGINNING_OF_CENTRAL,
			
		//	p_visccond_heatrate_neut, // from conduction routine...
		//	p_visccond_heatrate_ion,  // applies to major cells
		//  p_visccond_heatrate_elec,
			
			p_resistive_heat_neut_minor, // from midpoint v acceleration routine
			p_resistive_heat_ion_minor,
			p_resistive_heat_elec_minor,
			
			pXhalf->p_area, // major areas
			
			// output:
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			false // not 2nd pass
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Heating 1");
		
		::Kernel_Average_nT_to_tri_minors<<<numTriTiles,threadsPerTileMinor>>>(
										pX1->p_tri_corner_index,
										pX1->p_tri_perinfo, 
										pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_neut_minor,
										pXusable->p_nT_ion_minor,
										pXusable->p_nT_elec_minor);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize avg nT pXusable");
		
		printf("Heating 1 done\n");
		
		//cudaMemcpy(p_MAR_ion_host,				p_MAR_ion,
	//		sizeof(f64_vec3)*Syst1.Nminor,			cudaMemcpyDeviceToHost);
	//	Call(cudaThreadSynchronize(),"cudaThreadSynchronize memcpies");
		
		/*file = fopen("ionMAR.txt","w");
		if (file == 0) {
			printf("could not open ionMAR.txt");
			while (1) getch();
		} else {
			printf("ionMAR.txt opened");
			getch();
		};*/


		// OKay let's think about visc htg and conductive htg.
		// Conduction has to be done on major cells, using the B field etc from the minor cells.
		// Whereas viscous heating? Each minor wall generates some heating. Look at the two 
		// effects 2T dT/dt|wall and share the sum to both major cells.
		// Viscous heating at edge of central cell: share the sum to the major cell it is within.
		// OK that seems to work out well.
		
		// -------------------------------------------------------------------------------------------
		// Aside:
		// Is it easier just to do heating on minors and then average back to major? NO
		// There would be nothing wrong with putting ionisation in minors as well ... but it's blurring
		// something that does not need to be, since T is averaged before we do it.
		// -----------------------------------------------------------------------------------
		
		
		// Next thing:
		
		// . Calculate ionisation again with half-time heat
		// . Two runs of midpt - first one establishes Ohm relationship.
				
		cudaMemset(p_MAR_neut,0,sizeof(f64_vec3)*pX1->Nminor);
		cudaMemset(p_MAR_ion,0,sizeof(f64_vec3)*pX1->Nminor);
		cudaMemset(p_MAR_elec,0,sizeof(f64_vec3)*pX1->Nminor);
	
		// We have not changed A - it's still the same so no need to go again for Lap A.
		// We have not changed phi. But we have changed Te so re-estimate its gradient.
		::Kernel_Compute_grad_phi_Te_tris<<<numTriTiles, threadsPerTileMinor>>>
			(
			pXhalf->p_info,
			pXhalf->p_phi,     // on majors
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // on majors
			pXhalf->p_tri_corner_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_grad_phi,
			pXusable->p_GradTe
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi tri I");
		
		::Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor, threadsPerTileMajor>>>
			(
			pXhalf->p_info,
			pXhalf->p_phi,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->pIndexNeigh,
			// output:
			pXhalf->p_grad_phi + BEGINNING_OF_CENTRAL,
			pXusable->p_GradTe + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Compute grad phi central I");
		// CHECK PARAMETERS <<< >>> 
		
		// Get thermal pressure on each accelerating region...
		// Better off probably to do the ionisation stage first, it will give a better idea
		// of the half-time thermal pressure we are ultimately aiming for.
		// Not so much because it doesn't include recombination heating update -- I don't think so anyway.
		::Kernel_GetThermalPressureTris<<<numTriTiles,threadsPerTileMinor>>>
			( 
			pXhalf->p_info,			
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_tri_corner_index,
			pXhalf->p_tri_perinfo,
			p_MAR_neut,
			p_MAR_ion,
			p_MAR_elec // overwrite
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure tris");
		
		Kernel_GetThermalPressureCentrals<<<numTilesMajor,threadsPerTileMajor>>>
			(
			pXhalf->p_info,
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXhalf->pIndexNeigh,
			p_MAR_neut + BEGINNING_OF_CENTRAL,
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Thermal pressure");
		
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// It might be reasonable to instead be using TRIANGLE nT in getting
		// thermal pressure on centrals? You'd think so.
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		// Set Ez given initially estimated relationship -- 
		// is there actually a point in this? Does it affect what we now find?
		// Really not. ?

		// 2nd call to ionisation calc:
		Kernel_Ionisation<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep,
			pXhalf->p_info,
			pXhalf->p_area, // correct input? used?
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL, // src 
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,		
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL, // src 
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,		
			// No output, I think now, except nn_ionrec.			
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL, 
			0 // b2ndpass  --  ??
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Ionisation 2");
				
		Kernel_Average_nnionrec_to_tris<<<numTriTiles,threadsPerTileMinor>>>
			(
			pXhalf->p_tri_perinfo,
			pXhalf->p_tri_corner_index,
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL,
			p_nn_ionrec_minor
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize splitout nn");

		SendToHost(pXusable,pXhalf,pX_host);

		pX_host->AsciiOutput("inputs_3__.txt");
		// v and gradphi come back as IND / viz,vez INF.

		// Establish Ohmic relationship:
		printf("ready to do midpt again\n");
		getch();

		pXusable->evaltime = pXhalf->evaltime;
		Kernel_Midpoint_v_and_Adot<<<numTilesMinor,threadsPerTileMinor>>>
		(
			hstep,
			pXhalf->p_tri_perinfo,
			pXhalf->p_nT_neut_minor, // src
			pXhalf->p_nT_ion_minor, 
			pXhalf->p_nT_elec_minor, 
			// Both n_k and n_k+1 appear in the midpt formula, so we need n_k.
			
			pXusable->p_nT_neut_minor, // use
			pXusable->p_nT_ion_minor, 
			pXusable->p_nT_elec_minor, 
				
			p_nn_ionrec_minor, // Have to load 2 additional doubles due to doing ionisation outside.
							
			pXhalf->p_tri_centroid, // Defined ever?
			pXhalf->p_info,   // were these positions actually created?
				
			pXhalf->p_B,
			pXhalf->p_v_neut, // src
			pXhalf->p_v_ion,
			pXhalf->p_v_elec,
				// Thing is, we have to create 0.5*(v_k+v_k+1) on 1st pass.
				// We want to leave v_k[advected] intact, so no update here.
				// We want to go again from v_k on 2nd and 3rd pass.
					
			pXhalf->p_area_minor,	// It's assumed to be area_k+1 but I guess it's area_k+1/2 ... too bad?
								// THIS MATTERS !!
								// I think area_1/2 is relevant both times in midpt but need to check that.
			pXhalf->p_grad_phi,
			pXhalf->p_Lap_A, // check this input?
			pXhalf->p_Adot,  // anti-advected Adot_k ...
			p_MAR_neut,
			p_MAR_ion,
			p_MAR_elec, // assume take integral(-grad(nT))/m_s
			pXusable->p_GradTe,

			// output: not used, of course
			pXusable->p_v_neut,
			pXusable->p_v_ion,
			pXusable->p_v_elec,
				// Why are pXusable->v even useful? I don't see anywhere they are used.
				// Therefore do not store them on 1st pass!
			
			p_resistive_heat_neut_minor, 
			p_resistive_heat_ion_minor,
			p_resistive_heat_elec_minor,
			pXusable->p_Adot,
			1, // 2nd pass    --- send an integer here
			pX1->EzTuning, 
 			p_Iz0_summands, // here is what we want
			p_summands
		);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize MidptAccel 2");
		
		// Establish Ohmic relationship:
		CallMAC(cudaMemcpy(p_Iz0_summands_host,p_Iz0_summands,sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost));
		CallMAC(cudaMemcpy(p_summands_host,p_summands,sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost));
		f64 Iz0 = 0.0, IzPerEzTuning = 0.0;
		for (int ii = 0; ii < numTilesMinor; ii++)
		{
			Iz0 += p_Iz0_summands_host[ii];
			IzPerEzTuning += p_summands_host[ii];
		};

		// Set pXhalf->EzTuning:
		pXhalf->EzTuning = pX1->EzTuning + (Iz_prescribed-Iz0)/IzPerEzTuning;
		
		printf("pX1->EzTuning %1.8E Iz_prescribed %1.8E \n"
			"Iz0 %1.8E IzPerEzTuning %1.8E \n"
			"pXhalf->EzTuning %1.8E \n",
			pX1->EzTuning,Iz_prescribed, Iz0, IzPerEzTuning,
			pXhalf->EzTuning);
		getch();

		// Call with same parameters over again:
		Kernel_Midpoint_v_and_Adot<<<numTilesMinor,threadsPerTileMinor>>>
		(
			hstep,
			pXhalf->p_tri_perinfo,
			pXhalf->p_nT_neut_minor, // src
			pXhalf->p_nT_ion_minor, 
			pXhalf->p_nT_elec_minor, 
			// Both n_k and n_k+1 appear in the midpt formula, so we need n_k.
			pXusable->p_nT_neut_minor, // use
			pXusable->p_nT_ion_minor, 
			pXusable->p_nT_elec_minor, 
			p_nn_ionrec_minor, // Have to load 2 additional doubles due to doing ionisation outside.
			pXhalf->p_tri_centroid, // Defined ever?
			pXhalf->p_info,   // were these positions actually created?
			pXhalf->p_B,
			pXhalf->p_v_neut, // src
			pXhalf->p_v_ion,
			pXhalf->p_v_elec,
			pXhalf->p_area_minor,	// It's assumed to be area_k+1 but I guess it's area_k+1/2 ... too bad?
			pXhalf->p_grad_phi,
			pXhalf->p_Lap_A, 
			pXhalf->p_Adot,  
			p_MAR_neut, p_MAR_ion, p_MAR_elec,
			pXusable->p_GradTe,

			// output: 
			pXusable->p_v_neut,
			pXusable->p_v_ion,
			pXusable->p_v_elec,
			p_resistive_heat_neut_minor, // save off resistive.
			p_resistive_heat_ion_minor,  // Does it need to be zeroed beforehand?
			p_resistive_heat_elec_minor,
			pXusable->p_Adot,
			2, // 3rd pass    --- send an integer here
			pXhalf->EzTuning, 
 			p_Iz0_summands, // here is what we want
			p_summands
		);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize MidptAccel 3");
		
		CallMAC(cudaMemcpy(p_Iz0_summands_host,p_Iz0_summands,sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost));
		Iz0 = 0.0;
		for (int ii = 0; ii < numTilesMinor; ii++)
		{
			Iz0 += p_Iz0_summands_host[ii];
		};
		printf("Iz attained %1.8E Presc %1.8E Diff %1.4E\n",
			Iz0,Iz_prescribed,Iz0-Iz_prescribed);
		getch();
		// Can double-check here that Iz is being achieved:
		
		Kernel_Heating_routine<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep,
			pXhalf->p_info,
			pXhalf->pIndexTri, // fetch htg amounts from minor cells...
			
			pXhalf->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pXhalf->p_nT_ion_minor + BEGINNING_OF_CENTRAL,  
			pXhalf->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			p_nn_ionrec_minor + BEGINNING_OF_CENTRAL,
			// On 2nd pass, let us use the output as the "used" nT, and update it.
			
			pXhalf->p_B + BEGINNING_OF_CENTRAL,
			
		//	p_visccond_heatrate_neut, // from conduction routine...
		//	p_visccond_heatrate_ion,  // applies to major cells
		//    p_visccond_heatrate_elec,
			
			p_resistive_heat_neut_minor, // from midpoint v acceleration routine
			p_resistive_heat_ion_minor,
			p_resistive_heat_elec_minor,
			
			pXhalf->p_area,
			
			// output:
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			true // 2nd pass
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Heating 2");
		printf("heating done\n");
		
		::Kernel_Average_nT_to_tri_minors<<<numTriTiles,threadsPerTileMinor>>>(
										pX1->p_tri_corner_index,
										pX1->p_tri_perinfo, 
										pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
										pXusable->p_nT_neut_minor,
										pXusable->p_nT_ion_minor,
										pXusable->p_nT_elec_minor);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize avg nT pXusable");
		
		printf("end");
		while(1) getch();


		// We now created pXusable -> n,v,T and Adot. pXhalf->EzTuning.
		
		// Now the rest of the move.
		// Finish updating Adot:
		
		// Do advection, 2nd half. Going from pXusable to pX2.

		// v_overall will come from t_half.
		// If we want, we can extrapolate compared to t_k v_overall.
		Kernel_Create_v_overall_and_newpos<<<numTilesMajor,threadsPerTileMajor>>>(
			pXhalf->p_info,
			hstep*0.5,
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL, 
			
			pXusable->p_v_neut + BEGINNING_OF_CENTRAL,
			pXusable->p_v_ion + BEGINNING_OF_CENTRAL,
			pXusable->p_v_elec + BEGINNING_OF_CENTRAL, // central v
			
			pXusable->p_info,
			pXusable->p_v_overall + BEGINNING_OF_CENTRAL // make it for everything		
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Create_v_overall pXusable");
		::Kernel_Average_v_overall_to_tris<<<numTriTiles,threadsPerTileMinor>>>(
			pXusable->p_tri_corner_index,
			pXusable->p_tri_perinfo,	
			pXusable->p_v_overall + BEGINNING_OF_CENTRAL, // major v_overall
			pXusable->p_tri_centroid, 

			// HAS THIS BEEN POPULATED ??

			pXusable->p_v_overall
			); // so motion will take place relative to this velocity.
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Create_v_overall pXusable");
				
		
		Kernel_Compute_Grad_A_minor_antiadvect<<<numTriTiles,threadsPerTileMinor>>>(
			pXhalf->p_A,        // for creating grad
			pXhalf->p_A + BEGINNING_OF_CENTRAL,  // ?
			hstep*0.5, 
			pXusable->p_v_overall,    // hv = amt to anti-advect
			
			// Take geometry from pXhalf which was advected.
			pXhalf->p_info,       // 
			pXhalf->p_tri_centroid, //  Defined?
			pXhalf->p_tri_perinfo,     // 
			pXhalf->p_tri_per_neigh,
			pXhalf->p_tri_corner_index,    // 
			pXhalf->p_neigh_tri_index, // 
			pXhalf->pIndexTri,         // we said carry on using this for now.
			
			true,
			pXusable->p_Adot, // add h*0.5*Adot_k+1 to A ...
			// output:
			pX2->p_A // fill in for both tri and vert...						
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Antiadvect A etc II.");
			
		// Adot:
		Kernel_Compute_Grad_A_minor_antiadvect<<<numTriTiles,threadsPerTileMinor>>>(
			pXusable->p_Adot,        // for creating grad
			pXusable->p_Adot + BEGINNING_OF_CENTRAL,
			
			hstep*0.5,
			pXusable->p_v_overall,    // hv = amt to anti-advect
			
			pXhalf->p_info,       // 
			pXhalf->p_tri_centroid, // 
			pXhalf->p_tri_perinfo,     // 
			pXhalf->p_tri_per_neigh,
			pXhalf->p_tri_corner_index,    // 
			pXhalf->p_neigh_tri_index, // 
			pXhalf->pIndexTri,         // we said carry on using this for now.
			false,
			0,
			// output:
			pX2->p_Adot// fill in for both tri and vert...			
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Antiadvect Adot II");
		
		// Note pXhalf->grad_phi is already populated.
				
		Vhalf = pXhalf->EzTuning*3.5;
		Kernel_Advance_Antiadvect_phi<<<numTilesMajor,threadsPerTileMajor>>>
			(
				pXhalf->p_info,
				Vhalf,
				pXhalf->p_phi,
				pXusable->p_v_overall + BEGINNING_OF_CENTRAL, // SHOULD THIS NEED + numTris? Most obvious way is yes.
				hstep*0.5,
				pXhalf->p_grad_phi + BEGINNING_OF_CENTRAL,
				pXhalf->p_phidot	, // still using half-time value
				pX2->p_phi
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Antiadvect_phi II.");
		
		// Now get Lap phi_k+1
		// ...
		// If we were smart we'd avoid duplicating that effort.		
		// Now rel advect to produce pX2->nvT for doing rho_k+1 since we used unadvected for rho_k.
		
		::Kernel_CalculateTriMinorAreas_AndCentroids<<<numTriTiles,threadsPerTileMinor>>>
			(
			pX2->p_info,
			pX2->p_tri_corner_index,
			pX2->p_tri_perinfo,
			pX2->p_area_minor,
			pX2->p_tri_centroid
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalcMinorAreas+Centroids.pX2");
		
		Kernel_CalculateMajorAreas<<<numTilesMajor,threadsPerTileMajor>>>(
			pX2->p_info,
			pX2->p_tri_centroid,
			pX2->pIndexTri,
			pX2->pPBCtri,
			pX2->p_area
			);
		// Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Advance_A_with_Adot.");
		
		::Kernel_CalculateCentralMinorAreas<<<numTilesMajor,threadsPerTileMajor>>>( // central areas
			pX2->p_info, // used how?
			pX2->pIndexTri,
			pX2->p_area_minor,
			pX2->p_area_minor + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize CalcCentralMinorAreas, MajorAreas.pX2");
		
		
		Kernel_RelAdvect_nT<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pXhalf->p_info, 
			pXhalf->pIndexTri,
	//		pXhalf->pPBCtri,			
			pXhalf->p_tri_centroid, 

			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_neut_minor,
			pXusable->p_nT_ion_minor,    
			pXusable->p_nT_elec_minor,
			pXusable->p_v_neut,  // should always be minor...
			pXusable->p_v_ion,
			pXusable->p_v_elec,
			pXusable->p_v_overall, 

			pXhalf->p_area,
			pX2->p_area,
			// dest:
			pX2->p_nT_neut_minor + BEGINNING_OF_CENTRAL, 
			pX2->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
			pX2->p_nT_elec_minor + BEGINNING_OF_CENTRAL
			// Consider: {n,v,T} = 5 vars. One more is the magic number.
			// It is probably not the end of the world if we split into 2's and 3's, nT vs v.
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect nT pX2");
				
		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXusable->p_nT_neut_minor,   // -> momentum input
			pX2->p_nT_neut_minor, // destination n needed to divide Nv
			pXusable->p_v_overall,
			pXusable->p_v_neut,          // -> momentum input
			
			pXhalf->p_tri_centroid,
			pXhalf->p_tri_corner_index,
			pXhalf->p_neigh_tri_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_tri_per_neigh,  // ? does it need to exist?
			pXhalf->p_area_minor,
			pX2->p_area_minor,
			pX2->p_v_neut      // output
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri neut pX2");
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXhalf->p_tri_centroid,
			pXusable->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_neut_minor,
			pX2->p_nT_neut_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_v_neut + BEGINNING_OF_CENTRAL,
			pXusable->p_v_overall,
			pXhalf->pIndexTri,
			pXhalf->pPBCtri,
			pXhalf->p_area,
			pX2->p_area,
			pX2->p_v_neut + BEGINNING_OF_CENTRAL // ?
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent neut pX2");
		
		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXusable->p_nT_ion_minor,
			pX2->p_nT_ion_minor,
			pXusable->p_v_overall,
			pXusable->p_v_ion,

			pXhalf->p_tri_centroid,
			pXhalf->p_tri_corner_index,
			pXhalf->p_neigh_tri_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_tri_per_neigh,
			
			pXhalf->p_area_minor,
			pX2->p_area_minor,
			pX2->p_v_ion
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri ion");
		
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXhalf->p_tri_centroid,
			pXusable->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_ion_minor,
			pX2->p_nT_ion_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_v_ion + BEGINNING_OF_CENTRAL,
			pXusable->p_v_overall,
			pXhalf->pIndexTri,
			pXhalf->pPBCtri,
			pXhalf->p_area,
			pX2->p_area,
			pX2->p_v_ion + BEGINNING_OF_CENTRAL // ? Check USAGE
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent neut");
		

		::Kernel_Rel_advect_v_tris<<<numTriTiles,threadsPerTileMinor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXusable->p_nT_elec_minor,
			pX2->p_nT_elec_minor,
			pXusable->p_v_overall,
			pXusable->p_v_elec,

			pXhalf->p_tri_centroid,
			pXhalf->p_tri_corner_index,
			pXhalf->p_neigh_tri_index,
			pXhalf->p_tri_perinfo,
			pXhalf->p_tri_per_neigh,
			
			pXhalf->p_area_minor,
			pX2->p_area_minor,
			pX2->p_v_elec
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v tri ion");
		
		::Kernel_Rel_advect_v_central<<<numTilesMajor,threadsPerTileMajor>>>(
			hstep*0.5,
			pXhalf->p_info,
			pXhalf->p_tri_centroid,
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_nT_elec_minor,
			pX2->p_nT_elec_minor + BEGINNING_OF_CENTRAL,
			pXusable->p_v_elec + BEGINNING_OF_CENTRAL,
			pXusable->p_v_overall,
			pXhalf->pIndexTri,
			pXhalf->pPBCtri,
			pXhalf->p_area,
			pX2->p_area,
			pX2->p_v_elec + BEGINNING_OF_CENTRAL // ?
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Reladvect v cent neut");
		
		// =========================================================================
				
		Get_Lap_phi_on_major<<<numTilesMajor,threadsPerTileMajor>>>
			(
			pX2->p_phi,
			pX2->p_info,
			pX2->pIndexNeigh, // neighbours of vertices
			pX2->pPBCneigh, // rel periodic orientation of vertex neighbours
			p_Lapphi
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Get_Lap_phi_on_major II.");
		
		// Get grad_phidot...		
		::Kernel_Compute_grad_phi_Te_centrals<<<numTilesMajor,threadsPerTileMajor>>>(
			pXhalf->p_info,
			pXhalf->p_phidot,   // phidot is always for major
			pXusable->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // not important
			pXhalf->pIndexNeigh,
			p_grad_phidot + BEGINNING_OF_CENTRAL,
			pXhalf->p_GradTe + BEGINNING_OF_CENTRAL // nvm
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Compute_grad_phi_centrals.");
		
		// Is that call avoidable from what is above?


		Kernel_Advance_Antiadvect_phidot<<<numTilesMajor,threadsPerTileMajor>>>(
				pXhalf->p_phidot,	
				pXusable->p_v_overall + BEGINNING_OF_CENTRAL, // !!! NOTE BENE
				hstep*0.5,
				p_grad_phidot + BEGINNING_OF_CENTRAL, // on majors please
				
				p_Lapphi,
				pX2->p_nT_ion_minor + BEGINNING_OF_CENTRAL, 
				pX2->p_nT_elec_minor + BEGINNING_OF_CENTRAL, // --> rho _k
				pX2->p_phidot
				// This is just a ton of loading and a simple formula --
				// we should prefer to combine with Get_Lap_phi routine.
			);
		Call(cudaThreadSynchronize(),"cudaThreadSynchronize Kernel_Advance_Antiadvect_phidot II");
			
		pX2->evaltime = pXusable->evaltime + hstep*0.5;
		
		
		// Document sequence with inputs labelled fully and showing where calc'd.
		// Then check that the calcs are as claimed in each routine.
		t += hstep;
	};
	
	
	// We intermittently return to CPU to do re-Delaunerization - to begin with.
	// Otherwise only send data back, every 2.5e-11 s, for graphing. -> 20fps gives 2s/ns. 50s/25ns
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start,stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	//printf("Time elapsed: %s",report_time_(1));
	
	
	// 4. cudaMemcpy from device to host
	
	CallMAC(cudaMemcpy(pX_host_target->p_phi, pX1->p_phi, numVertices*sizeof(f64), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_phidot, pX1->p_phidot, numVertices*sizeof(f64), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_A, pX1->p_A, numVertices*sizeof(f64_vec3), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_Adot, pX1->p_Adot, numVertices*sizeof(f64_vec3), cudaMemcpyDeviceToHost));
	
	CallMAC(cudaMemcpy(pX_host_target->p_nT_neut_minor, pX1->p_nT_neut_minor, numVertices*sizeof(nT), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_nT_ion_minor, pX1->p_nT_ion_minor, numVertices*sizeof(nT), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_nT_elec_minor, pX1->p_nT_elec_minor, numVertices*sizeof(nT), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_v_neut, pX1->p_v_neut, numVertices*sizeof(f64_vec3), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_v_ion, pX1->p_v_ion, numVertices*sizeof(f64_vec3), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_v_elec, pX1->p_v_elec, numVertices*sizeof(f64_vec3), cudaMemcpyDeviceToHost));
		
	CallMAC(cudaMemcpy(pX_host_target->p_info, pX1->p_info, numVertices*sizeof(structural), cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->pIndexNeigh, pX1->pIndexNeigh, numVertices*MAXNEIGH_d*sizeof(long),cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->pPBCneigh, pX1->pPBCneigh, numVertices*MAXNEIGH_d*sizeof(char),cudaMemcpyDeviceToHost));
	CallMAC(cudaMemcpy(pX_host_target->p_area, pX1->p_area, numVertices*sizeof(f64),cudaMemcpyDeviceToHost));
	// Do we need to copy back neighbour arrays? Will they actually ever change?
	// pX->host_target has to point to an object already invoked ie with dimensioned arrays.
	
	
	cudaFree(p_summands);
	cudaFree(p_Iz0_summands);
	cudaFree( p_resistive_heat_neut_minor);
	cudaFree( p_resistive_heat_ion_minor);
	cudaFree( p_resistive_heat_elec_minor);
	cudaFree(p_scratch_d);
	cudaFree(p_Iz0_initial);

	cudaFree(p_nn_ionrec_minor);
	cudaFree(p_MAR_neut);
	cudaFree(p_MAR_ion);
	cudaFree(p_MAR_elec);

	cudaFree(p_Lapphi);
	cudaFree(p_grad_phidot);

	free(p_summands_host);
	free(p_Iz0_summands_host);
	free(p_scratch);
	free(p_scratch_info);
	free(p_Iz0_initial_host);
	free(p_scratch_host);
	free(p_nn_host);
	free(p_MAR_neut_host);
	free(p_MAR_ion_host);
	free(p_MAR_elec_host);

	printf("Transferred back.\n");
	
	Call(cudaMemGetInfo (&uFree,&uTotal),"cudaMemGetInfo (&uFree,&uTotal)");
	printf("uFree %d uTotal %d\n",uFree,uTotal);
	
	printf("END OF CUDA");

	//" do not call cudaResetDevice(): save invoked stuff for next step."

}