#ifndef cuda_struct_h
#define cuda_struct_h
// Avoid allocating and deallocating:

// Here is what should be basically common object...
// but not necessarily.

// vector_tensor.cu probably going to be needed
#include "vector_tensor.cu"
#include <conio.h>

#define MAXNEIGH    42  // large number needed to cater for links on aux mesh if we do not minimize connections.
#define MAXNEIGH_d  12

// 12*32768*5 = 2MB .. just to keep things in perspective.
// We should keep the number down just to reduce fetch size.
// Let's keep it real. nvT is best for our fetches and therefore is best.

long const numTriTiles = 288; // note that there are also centrals
long const numTilesMajor = 288;
long const numTilesMinor = 432; // 432 = 288+144
					// 456*256 = 304*256 + 304*128
			
// numTriTiles == numTilesMajor because the two sets are bijective.
// Then we also have to assign central minors to tiles, twice the size of the major tiles...

long const threadsPerTileMinor = 256;
long const threadsPerTileMajor = 128; // see about it - usually we take info from minor.
long const SIZE_OF_MAJOR_PER_TRI_TILE = 128;
long const SIZE_OF_TRI_TILE_FOR_MAJOR = 256;
long const BEGINNING_OF_CENTRAL = threadsPerTileMinor*numTriTiles;



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
	short neigh_len;
	char flag;     // we want to now include "has_periodic"
	// does that mean flag, has_periodic become char ? 
	char has_periodic; // Let's really hope it doesn't pad this to be like 5 longs.
}; // 8+8+2+1+1

struct nT {
	f64 n; f64 T;
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
	int Systdata::LoadHost(const char str[]);
	int Systdata::SaveHost(const char str[]);
	void Systdata::AsciiOutput (const char filename[]) const ;
	~Systdata();
};

// We need to move these funcs out into a .cu file.


void PerformCUDA_Advance_2 (
		const Systdata * pX_host, // populate in CPU MSVC routine...
		long numVerts,
		const real hsub, 
		const int numSubsteps,
		const Systdata * pX_host_target
		);
#endif
