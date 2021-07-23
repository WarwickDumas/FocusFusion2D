#ifndef cuda_struct_h
#define cuda_struct_h
// Avoid allocating and deallocating:

#define MAXNEIGH_d 12

// Here is what should be basically common object...
// but not necessarily.

// vector_tensor.cu probably going to be needed
#include "vector_tensor.cu"
#include "FFxtubes.h"
#include "mesh.h"

	// we do not just do the mesh advection every time because
	// this 
	// -- creates floating point error
	// -- necessitates more calculations (of Lap etc).
	// We can do it every 1e-12.

	// Temporary measure: hardcoded MAXNEIGH
	// but we should revise it to be empirical and match the
	// actual maximum seen, which is constant until we reDelaunerise

	// Well but that could be a problem: dynamic arrays ...

	// needing cudaMalloc of a nVerts*maxneighs block.
	// That should happen during the CPU->GPU transfer.

	// array of long followed by array of char for periodic.
	// That's not good as on loading in kernel, have to unpack
	// into 2 different variables.
	// Are we better off with 2 separate fetches?
	// Or create inner object: long index; char periodic;

	// Therefore 2 fetches is fine.

struct CHAR4 {
	char flag, per0, per1, per2;
}; // 4 bytes
struct CHAR3 {
	char c1, c2, c3;
};	// Can see the genuine case for padding it to 4 bytes ie using CHAR4 and never CHAR3.

struct LONG3 {
	long i1, i2, i3;
}; // 12 bytes
struct nn {
	f64 n_ionise, n_recombine;
};
	// Vertices must be sent in anticlockwise sequence.
struct twoshort {
	short flag;
	short neigh_len;
};

struct structural {
	f64_vec2 pos; // Only ever used with flag, therefore put together.
	// NO: because flag isn't only used with pos. So basically flatpack would have been better in general.
	short neigh_len;
	char flag;     // we want to now include "has_periodic"
	// does that mean flag, has_periodic become char ? 
	char has_periodic; // Let's really hope it doesn't pad this to be like 5 longs.
}; // 8+8+2+1+1

struct nT {
	f64 n; f64 T;
};

struct v4 {
	f64_vec2 vxy;
	double viz, vez;
};

struct nvals {
	f64 n, n_n;
	__device__ __host__ nvals(f64 n_, f64 nn_) {
		n = n_; n_n = nn_;
	}
	void __device__ __forceinline__ operator+= (const nvals &nval) {
		n += nval.n;
		n_n += nval.n_n;
	}
	__device__ __host__ nvals() {}
};
nvals __device__ inline operator* (const real hh, const nvals & nval) 
{
	return nvals(hh*nval.n, hh*nval.n_n);
}

struct f64_12 {
	f64 n[12];
};

struct T2 {
	f64 Ti, Te;
};
struct T3 {
	f64 Tn, Ti, Te;
	__device__ __host__ T3(f64 Tn_, f64 Ti_, f64 Te_) {
		Tn = Tn_; Ti = Ti_; Te = Te_;
	}
	__device__ __host__ T3(){}
	void __device__ __forceinline__ operator+= (const T3 &T) {
		Tn += T.Tn;
		Ti += T.Ti;
		Te += T.Te;
	}
};
T3 __device__ inline operator* (const real hh, const T3 &T) 
{
	return T3(hh*T.Tn, hh*T.Ti, hh*T.Te);
};


struct AAdot {
	f64 Az, Azdot;
};

struct OhmsCoeffs {
	f64 sigma_e_zz, sigma_i_zz;
	f64_vec2 beta_xy_z;
	f64 beta_ne, beta_ni;
};

struct species3 {
	f64 n, i, e;
};

struct Systdata {
	
	// shared 32k -> 4000 doubles / 400 points
	// so keep only 10 or 12 doubles in a struct.
	// nvT*3 = 15. 4000/16 = 250 which is too small,
	// could be just 1 row, we need it to be at least 2.5 ...
	// Somehow we Must do a pass for flux contrib to
	// each species in turn I suppose.

	f64 * p_phi, * p_phidot;
	f64_vec3 * p_A, * p_Adot;
	
	nT * p_nT_neut_minor, * p_nT_ion_minor, * p_nT_elec_minor;
	// _minor reminds us that as with v, it is tris first and then major=central
	f64_vec3 * p_v_neut, * p_v_ion, * p_v_elec;

	f64_vec3 * p_MomAdditionRate_neut,
		     * p_MomAdditionRate_ion,
			 * p_MomAdditionRate_elec;
	f64 * p_intercell_heatrate_neut,
		* p_intercell_heatrate_ion,
		* p_intercell_heatrate_elec;

	// Would we gain any advantage from clubbing them together? 
	// Not that I can see -- loads of f64 HeatAdditionRate will always be contiguous.

	// As long as they are not too big,
	// structs are a good thing even for performance,
	// compared to loading from several flatpacks.
	// But not if we would sometimes load whole struct and only need one variable.

	f64 * p_area, * p_area_minor;       // Useful for evol routine.
	structural * p_info; // neigh_len + flag = 32 bits.
	long * pIndexNeigh, * pIndexTri; // pointer to numVertices*maxNeighs
	// We would actually save quite a bit of fetch by making it short not long!
	// But that is limiting... can we accept 65536 on each level?
	char * pPBCneigh, * pPBCtri;   // pointer to numVertices*maxNeighs
	
	// Always would be nice to go over what I wrote and streamline what
	// structural information is needed.

	f64_vec2 * p_grad_phi; // Can this be combined into M.A.R.? No
	f64_vec3 * p_B, * p_Lap_A;
	f64_vec2 * p_GradTe, * p_v_overall;
	
	f64_vec2 * p_tri_centroid;
	CHAR4 * p_tri_perinfo;
	LONG3 * p_tri_corner_index;
	LONG3 * p_neigh_tri_index; // NOT CLEAR if we need
	CHAR4 * p_tri_per_neigh;   // Not clear we need.

	long Nverts,Ntris,Nminor, numReverseJzTris;

	bool bInvoked, bInvokedHost;
	f64 EzTuning, evaltime; // each system is to have an evaltime now.
	f64 InnermostFrillCentroidRadius, OutermostFrillCentroidRadius;

	// HOST FUNCTIONS:
	Systdata();
	void Invoke(long N);
	void InvokeHost(long N);
	void Zero();
	void ZeroHost();
	void RevokeHost();
	int Systdata::LoadHost(const char [], bool);
	int Systdata::SaveHost(const char str[]);
	void Systdata::AsciiOutput (const char filename[]) const ;
	void Systdata::AsciiOutputEdges (const char filename[]) const ;
	void Systdata::AsciiOutput4Values (FILE * file, real eval) const ;
	void Systdata::AsciiOutputSpecific (FILE * file,real eval) const ;

	~Systdata();
};

// We need to move these funcs out into a .cu file.



class cuSyst {
private:
public:
	bool bInvoked, bInvokedHost;
	long Nverts, Ntris, Nminor;
	
	structural * p_info;
	 
	 long * p_izTri_vert;
	 long * p_izNeigh_vert;
	 char * p_szPBCtri_vert;   // MAXNEIGH*numVertices
	 char * p_szPBCneigh_vert;

	 long * p_izNeigh_TriMinor;
	 char * p_szPBC_triminor; // 6*numTriangles
	 
	 LONG3 * p_tri_corner_index;
	 CHAR4 * p_tri_periodic_corner_flags;

	 LONG3 * p_tri_neigh_index;
	 CHAR4 * p_tri_periodic_neigh_flags;    
	 
	 LONG3 * p_who_am_I_to_corner;
 
	 nvals * p_n_minor;
	 nvals * p_n_major;
	 T3 * p_T_minor; 
	 f64_vec3 * p_v_n;
	 v4 * p_vie;
	 AAdot * p_AAdot;
	 f64_vec3 * p_B;

	 f64 * p_Lap_Az;
	 f64_vec2 * p_v_overall_minor;
	 nvals * p_n_upwind_minor;

	 //f64_vec2 * p_pos; // vertex positions are a subset?
	 // We made structural * p_info assuming if we want pos we also want flags...
	 
	 f64 * p_AreaMinor;
	 f64 * p_AreaMajor;
	 f64_vec2 * p_cc;

	 char * p_iVolley;

	cuSyst();
	int Invoke();
	int InvokeHost();
	void SendToHost(cuSyst & Xhost);
	void SendToDevice(cuSyst & Xdevice);
	void CopyStructuralDetailsFrom(cuSyst & src); // on device

	void ReportDifferencesHost(cuSyst &X2);

	void PopulateFromTriMesh(TriMesh * pX);
	void PopulateTriMesh(TriMesh * pX);

	void PerformCUDA_Advance(//const 
		cuSyst * pX_target, //const
		cuSyst * pX_half);

	void cuSyst::PerformCUDA_AdvectionCompressionInstantaneous(//const 
		f64 const Timestep,
		cuSyst * pX_target,
		cuSyst * pX_half);

	void PerformCUDA_Advance_noadvect(//const 
		cuSyst * pX_target, //const
		cuSyst * pX_half);

	void PerformCUDA_Advance_Debug(const cuSyst * pX_target, const cuSyst * pX_half,
		const cuSyst * p_cuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
		TriMesh * pDestMesh);

	void Output(const char * filename);

	void cuSyst::Load(const char filename[]);
	void cuSyst::Save(const char filename[]);

	void cuSyst::SaveGraphs(const char filename[]);

	void ZeroData();
	~cuSyst();
};



void PerformCUDA_Invoke_Populate (
	cuSyst * pX_host, // populate in calling routine...
	long numVerts,
	f64 InnermostFrillCentroidRadius,
	f64 OutermostFrillCentroidRadius,
	long numStartZCurrentTriangles_,
	long numEndZCurrentTriangles_
		);

void PerformCUDA_RunStepsAndReturnSystem(cuSyst * pX_host);
void PerformCUDA_RunStepsAndReturnSystem_Debug(cuSyst * pcuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
	TriMesh * pDestMesh);

void PerformCUDA_Revoke();

void FreeVideoResources ();

extern __host__ bool Call(cudaError_t cudaStatus, char str[]);

#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   

#define Set_f64_constant(dest, src) { \
		Call(cudaGetSymbolAddress((void **)(&f64address), dest ), \
			"cudaGetSymbolAddress((void **)(&f64address), dest )");\
		Call(cudaMemcpy( f64address, &src, sizeof(f64),cudaMemcpyHostToDevice),\
			"cudaMemcpy( f64address, &src, sizeof(f64),cudaMemcpyHostToDevice) src dest");\
						}

#endif
