
#include <conio.h>
#include <stdio.h>

// It is possible for class members to be __device__. But:

// http://stackoverflow.com/questions/6978643/cuda-and-classes
// Mark Harris: "Any method that must be called from device code should be defined with
// both __device__ and __host__ declspecs, including the constructor and destructor if 
// you plan to use new/delete on the device"

// Unlikely this will break down instantly ... can put a test in CPU program.

#include "vector_tensor.cu"
		
#ifndef CLASSES_H
#define CLASSES_H

struct momflux{
	f64_vec3 dv_ion_bydt, dv_neut_bydt;
	f64 heatrate_neut, heatrate_ion;
} ; 

struct heatflux{
	f64 d_NnTn_bydt,
		d_NiTi_bydt,
		d_NeTe_bydt;
	// repeat from species_f64 really.
} ;

struct fluid
{
	f64 mass,heat;
	f64_vec3 mom;
#ifdef __CUDACC__
	__host__ __device__ void Add(const f64 scale,const fluid & add);
	__host__ __device__ void operator /= (const f64 divisor);
#endif
};


struct nvT
{
	f64 n,T;
	f64_vec3 v;
#ifdef __CUDACC__
	__host__ __device__ void Add(const f64 scale,const nvT & add);

	__host__ __device__ void operator /= (const f64 divisor);
#endif
};

struct fluid3BE 
{
	fluid neut, ion, elec;
	f64_vec3 B; 
	f64 Ez;
	
	// Note: putting same class in non-nvcc compile, remove "__host__ __device__".
	// Per remark by Mark Harris.

	// Aargh - the same must apply to operator * for a fluid.
	// We don't even need this more than once. Let's be sensible.
#ifdef __CUDACC__
	__host__ __device__ void AddnvT(const f64 scale, const fluid3BE addition);
	__host__ __device__ void Divide_nvT(const f64 divisor);
	__host__ __device__ void Zero_nvT();
#endif
}; // about 19 x 8 = 152 bytes

// surely going to also want to have kappa, nu_heart ... see about it

struct vertdata 
{
	nvT neut, ion, elec;
	f64_vec3 B; // these are the best defined.
	
#ifdef __CUDACC__
	__host__ __device__ void AddnvT(const f64 scale, const vertdata & addition);
	__host__ __device__ void AddnvT_from_fluid3BE(const f64, const fluid3BE &, f64 );

	__host__ __device__ void Zero_nvT();
	__host__ __device__ void Divide_nvT(const f64 divisor);
#endif
};
class ROCArray {
public:

	momflux * pmf; // you don't need to put __device__ for a pointer.
	bool bInvoked;
	long Ndim;
	// Methods not to be called from kernels.
	
	ROCArray();
	void Invoke (long N);
	void Zero () ;

	void Extrapolate (ROCArray * pROCflux0, ROCArray * pROCflux1, 
		real h_used, 
		ROCArray * pROCbase, real h_extrap);

	void Get_dfdt(ROCArray * pROCflux0, ROCArray * pROCflux1, real hstep);
	
	void ROCArray::SetLinear(ROCArray * pROCflux0, real hstep, ROCArray * dfdt);
	void ROCArray::SetLinear(ROCArray * pROCflux0, real hstep1, ROCArray * dfdt1,
		real hstep2, ROCArray * dfdt2);

	void GetAvgSq(real * pv_neut_avg_sq, real * pv_ion_avg_sq);

	void FileOutput(FILE * fp, long start, long end);

	~ROCArray() ;
}; // This way only need to dimension and deallocate once over whole program.
 
.
// Have to get rid of cpp same-named class when we switch to this.
class DTArray {
public:
	heatflux * phf; // you don't need to put __device__ for a pointer.
	bool bInvoked;
	long Ndim;
	// Methods not to be called from kernels.
	// The stuff in here also is no good for the cpp compiler to read.

	// DTArray(){};

	void Invoke (long N);

	void Zero () ;

	void Extrapolate (DTArray * pROCflux0, DTArray * pROCflux1, 
		real h_used, DTArray * pROCbase, real h_extrap);

	void Get_dfdt(DTArray * pROCflux0, DTArray * pROCflux1, real hstep);
	
	void SetLinear(DTArray * pROCflux0, real hstep1, DTArray * dfdt1);
	void SetLinear(DTArray * pROCflux0, real hstep1, DTArray * dfdt1,
		real hstep2, DTArray * dfdt2);

	void GetAvgSq(real * pT_neut_avg_sq, real * pT_ion_avg_sq, real * pT_elec_avg_sq);

	void DTArray::FileOutput( FILE * fp, long start, long end);
	~DTArray() ;
}; 
// This way only need to dimension and deallocate once over whole program.
 

#include "cuda_struct.h"


struct Matrix_real_6
{
	static int const LUSIZE = 6;

	long indx[6];   // the index array when decomposed
	real LU[6][6]; // the elements
	real vv[6];

	//void CopyFrom(Matrix_real & src);

#ifdef __CUDACC__
	__host__ __device__ __forceinline__ long LUdecomp() ;
	__host__ __device__ __forceinline__ long LUSolve (real b[], real x[]);
#endif

};

	

struct CalculateAccelsClass_d
{
	// exists only to do a calculation repeatedly from some stored data
	// Need to trim what is stored, to reduce register pressure....

	f64_vec3 omega_ce;//, omega_ci; // can make do with 1 or none
	//f64_tens3 omega_ci_cross;  // totally want to do without this!
	
	f64 nu_eiBar, nu_eHeart, nu_ieBar, nu_en_visc, s_en_visc, // added these for debug 
		nu_en_MT, nu_in_MT, nu_ne_MT, nu_ni_MT,
		n_i, n_n, n_e,
		heat_transfer_rate_in,heat_transfer_rate_ni,
		heat_transfer_rate_en,heat_transfer_rate_ne,
		heat_transfer_rate_ei,heat_transfer_rate_ie,
		fric_dTe_by_dt_ei,StoredEz;
	
	f64_vec3 ROC_v_ion_due_to_Rie,
			 a_neut_pressure,a_ion_pressure,
			 vrel_e, ROC_v_ion_thermal_force; 

	//f64_tens3 Upsilon_nu_eHeart, Rie_thermal_force_matrix,
	// avoidable? some must be!
	//       Rie_friction_force_matrix, Ratio_times_Upsilon_eHeart;
	
	f64 Rie_friction_force_matrix_zz;
	
	// What is used for temperatures?
	// content ourselves with leaving all calcs to happen every time for now.
	// But putting nu array into memory would be very reasonable to try.

	bool bNeutrals;

	// EASIER WAY:
	// Let's just stick simple Ohm's Law v_e(v_i,v_n)
	// in CalculateCoefficients.

	f64 SimpleOhms_vez0, SimpleOhms_beta_neutz, Ohms_vez0, Ohms_sigma;
	f64_vec3 SimpleOhms_beta_ion;
	// we do need those.

	// Now it became clear:
	// We are not apparently supposed to declare members as only
	// __device__ but __host__ __device__ (see Mark Harris note elsewhere).
	// Therefore since we cannot compile as host, put outside the class.
};

#ifdef __CUDACC__
__device__ __forceinline__ void CalculateCoefficients(
											fluid3BE use, 
											species_vec2 pressure,
										//	structural info,
											CalculateAccelsClass_d & Y);
		
__device__ __forceinline__ void Populate_Acceleration_Heating_Coefficients_noU
							(CalculateAccelsClass_d const Y,
							real H[6][6], real a0[6], real h0[4],
							fluid3BE use) ;
#endif

#endif
