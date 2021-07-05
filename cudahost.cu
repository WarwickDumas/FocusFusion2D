// Version 1.0 23/04/19:
// Changing to use upwind T for advection. We could do better in future. Interp gives negative T sometimes.
// Corrected ionisation rate.
            
#pragma once     
#include "switches.h" // defines AZCG

#include <cusolverDn.h>

#define LAPACKE
#define PRECISE_VISCOSITY 
#define DEBUGTE               0
  
#define EQNS_TOTAL 192

#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <time.h>
#include <windows.h>
   
//#include "lapacke.h"
  
#include "mesh.h"
#include "FFxtubes.h"
#include "cuda_struct.h"
#include "flags.h"
#include "kernel.h"
#include "matrix_real.h"
                     
#ifdef LAPACKE  
// Auxiliary routines prototypes 
extern void print_matrix(char* desc, int m, int n, double* a, int lda);
extern void print_int_vector(char* desc, int n, int* a);
#endif 
 
extern bool GlobalCutaway;
extern HWND hwndGraphics;
extern TriMesh X4; 
 
#define BWD_SUBCYCLE_FREQ  1
#define BWD_STEP_RATIO     1    // divide substeps by this for bwd
#define NUM_BWD_ITERATIONS 4
#define FWD_STEP_FACTOR    2    // multiply substeps by this for fwd

// This will be slow but see if it solves it.
  
#define CHOSEN 52178
// 19020 for ez visc
#define CHOSEN1 14332
#define CHOSEN2 14334 
 

#define VERTCHOSEN 26102 // The original highest position
// 17906  // 26081 // 22026 // 16196
#define VERTCHOSEN2 50

#define ITERATIONS_BEFORE_SWITCH  18
#define REQUIRED_IMPROVEMENT_RATE  0.98
#define REQUIRED_IMPROVEMENT_RATE_J  0.985

// This is the file for CUDA host code.
#include "simulation.cu"
  
#define p_sqrtDN_Tn p_NnTn
#define p_sqrtDN_Ti p_NTi
#define p_sqrtDN_Te p_NTe
 
#define DEFAULTSUPPRESSVERBOSITY false


bool inline within(f64 l, f64 a, f64 b)
{
	if (a < b) {
		if ((l <= b) && (l >= a)) return true;
	} else {
		if ((l >= b) && (l <= a)) return true;
	};
	return false;
}

__device__ int *d_a;
void cudaMemoryTest()
{
	const unsigned int N = 1048576;
	const unsigned int bytes = N * sizeof(int);
	int *h_a = (int*)malloc(bytes);
	CallMAC(cudaMalloc((int**)&d_a, bytes));

	memset(h_a, 0, bytes);
	CallMAC(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	CallMAC(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	CallMAC(cudaFree(d_a));
	free(h_a);
}

void RunBackwardR8LSForViscosity_Geometric(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use);

void inline SubroutineComputeDbyDbetaNeutral(
	f64 const hsub, f64_vec2 * p_regrxy, f64 * p_regriz, f64_vec3 * p_v_n, cuSyst * pX_use,
	int i,
	int iUsexy, int iUse_z);

extern surfacegraph Graph[8];
extern D3D Direct3D;
extern HWND hWnd;
  
FILE * fp_trajectory;
FILE * fp_dbg;
bool GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
bool bGlobalSaveTGraphs;
bool bViscousHistory;
long iHistoryVN = 0;

long VERTS[3] = {15559, 15405, 15251};
long iEquations[3];
  
extern long NumInnerFrills, FirstOuterFrill;
__constant__ long NumInnerFrills_d, FirstOuterFrill_d;
__constant__ long DebugFlag;
__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices 
f64 over_iEquations_n, over_iEquations_i, over_iEquations_e;
__constant__ f64_tens2 Anticlockwise_d, Clockwise_d; // use this to do rotation.   
__constant__ f64 kB, c, q, m_e, m_ion, m_i, m_n, 
eoverm, qoverM, moverM, qovermc, qoverMc,
FOURPI_Q_OVER_C, FOURPI_Q, FOURPI_OVER_C,
one_over_kB, one_over_kB_cubed, kB_to_3halves,
NU_EI_FACTOR, nu_eiBarconst, Nu_ii_Factor,
M_i_over_in,// = m_i / (m_i + m_n);
M_e_over_en,// = m_e / (m_e + m_n);
M_n_over_ni,// = m_n / (m_i + m_n);
M_n_over_ne,// = m_n / (m_e + m_n);
M_en, //= m_e * m_n / ((m_e + m_n)*(m_e + m_n));
M_in, // = m_i * m_n / ((m_i + m_n)*(m_i + m_n));
M_ei, // = m_e * m_i / ((m_e + m_i)*(m_e + m_i));
m_en, // = m_e * m_n / (m_e + m_n);
m_ei, // = m_e * m_i / (m_e + m_i);
over_sqrt_m_ion, over_sqrt_m_e, over_sqrt_m_neutral,
over_m_e, over_m_i, over_m_n,
four_pi_over_c_ReverseJz, RELTHRESH_AZ_d,
FRILL_CENTROID_OUTER_RADIUS_d, FRILL_CENTROID_INNER_RADIUS_d;

__constant__ long lChosen;
__constant__ f64 UNIFORM_n_d;
__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
                 cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];
__constant__ f64 beta_n_c[32], beta_i_c[8], beta_e_c[8];
__constant__ bool bSwitch = false;
__constant__ f64 recomb_coeffs[32][3][5];
f64 recomb_coeffs_host[32][3][5];
__constant__ f64 ionize_coeffs[32][5][5];
f64 ionize_coeffs_host[32][5][5];  
__constant__ f64 ionize_temps[32][10];
f64 ionize_temps_host[32][10];
__constant__ long MyMaxIndex;
__device__ __constant__ f64 billericay;
__constant__ f64 Ez_strength;
f64 EzStrength_;
__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)
__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;
f64 * p_eqns_host;

#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   
// { Call(cudaStatus, "cudaStatus") } ?
extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;
extern bool flaglist[NMINOR];

cuSyst cuSyst1, cuSyst2, cuSyst3;
extern cuSyst cuSyst_host;
// Given our restructure, we are going to need to dimension
// a cuSyst type thing that lives on the host??
// Not necessarily and not an easy way to write.
// This time around find another way to populate.
// We do need a temporary such object in the routine where we populate the device one.
// I guess as before we want an InvokeHost routine because of that.
__device__ real * p_summands, *p_Iz0_summands, *p_Iz0_initial, *p_scratch_d;
f64 * p_summands_host, *p_Iz0_summands_host, *p_Iz0_initial_host;
__device__ f64 * p_temp1, *p_temp2, *p_temp3, *p_temp4,*p_temp5, *p_temp6, *p_denom_i, *p_denom_e, *p_coeff_of_vez_upon_viz, *p_beta_ie_z;
__device__ f64_vec3 * p_temp3_1, *p_temp3_2, *p_temp3_3;
__device__ f64_vec3 * p_ROCMAR1, *p_ROCMAR2, *p_ROCMAR3;
__device__ f64 * p_graphdata1, *p_graphdata2, *p_graphdata3, *p_graphdata4, *p_graphdata5, *p_graphdata6;
f64 * p_graphdata1_host, *p_graphdata2_host, *p_graphdata3_host, *p_graphdata4_host, *p_graphdata5_host, *p_graphdata6_host;
__device__ f64_vec3* p_MAR_ion_temp_central, *p_MAR_elec_temp_central;
__device__ f64 * p_Tgraph[9];
f64 * p_Tgraph_host[9];
__device__ f64_vec2 * p_stored_move_neut_xy, *p_stored_move_neut_xy2;
__device__ f64 * p_stored_move_neut_z, *p_stored_move_neut_z2;
__device__ f64_tens2 * p__matrix_xy_i, *p__matrix_xy_e,*p__invmatrix;
__device__ f64 * p__coeffself_iz, *p__coeffself_ez, *p__invcoeffself, 
*p__invcoeffselfviz, *p__invcoeffselfvez;
__device__ double4 * p__xzyzzxzy_i, *p__xzyzzxzy_e;
__device__ f64_vec2 * p_regressors2;
__device__ f64 * p_regressors_iz, *p_regressors_ez;
__device__ f64 * p_dump, *p_dump2;
__device__ f64 * p__invcoeffself_x, *p__invcoeffself_y;
__device__ f64 * p_ita_i, *p_ita_n, *p_ita_e, *p_nu_n;
__device__ long * p_blockTotal1, long * p_blockTotal2;
long *p_longtemphost2;
int * p_inthost;

__device__ f64 * d_B, * p_eqns, * p_eqns2;
__device__ int * d_Ipiv512, *d_info;
__device__ f64 * p_RHS;
__device__ f64_vec2 * p_regrlc2;
__device__ f64 * p_regrlc_iz;
__device__ f64 * p_regrlc_ez;
__device__ int * p_Selectflag, *p_SelectflagNeut;

__device__ NTrates * NT_addition_tri_d2, *NT_addition_tri_d3, *NT_addition_rates_d_2, *NT_addition_rates_d_3;

__device__ f64 * p_accelgraph[12];
f64 * p_accelgraph_host[12];
f64 * p_Ohmsgraph_host[20];
__device__ f64 * p_Ohmsgraph[20];
__device__ f64 * p_eps_against_d_eps, *p_eps_against_deps2;
__device__ f64 * p_arelz_graph[12];
f64 * p_arelz_graph_host[12];
__device__ f64 * p_AreaMinor_cc, *p_sqrtfactor;
__device__ v4 * v4temparray, *zero_vec4, *p_vie_save;
__device__ f64_vec3 * p_MAR_ion3, *p_MAR_elec3, *p_MAR_neut3 , *p_MAR_neut2,
					* p_MAR_neut4, *p_MAR_ion4, *p_MAR_elec4;
__device__ f64_vec3 * p_MAR_ion_pressure_major_stored, *p_MAR_ion_visc_major_stored, *p_MAR_elec_pressure_major_stored, *p_MAR_elec_visc_major_stored, *p_MAR_elec_ionization_major_stored,
					*p_v_n_modified_k;
__device__ v4 * p_vie_k_stored, *p_vie_modified_k, *p_vie2;	
__device__ bool * p_pressureflag;
__device__ f64_vec3 * p_d_epsilon_by_d_beta_x, *p_d_epsilon_by_d_beta_y,
*p_d_epsilon_by_d_beta_z, *p_epsilon3, *v3temp, *zero_vec3;
__device__ f64 * p_place_contribs;
__device__ v4 * p_storeviscmove;
__device__ f64 * p_Residuals;
__device__ long * p_longtemp;
__device__ bool * p_bool;
__device__ f64 * p_regressor_n, *p_regressor_i, *p_regressor_e, *p_Effect_self_n, *p_Effect_self_i, *p_Effect_self_e,
*d_eps_by_dx_neigh_n, *d_eps_by_dx_neigh_i, *d_eps_by_dx_neigh_e;
__device__ T3 * p_store_T_move1, *p_store_T_move2;
NTrates * p_NTrates_host;
__device__ f64 * p_sqrtD_inv_n, *p_sqrtD_inv_i, *p_sqrtD_inv_e;
__device__ f64 * p_regressors;
f64 * p_sum_eps_deps_by_dbeta_x8_host;
__device__ f64_vec3 * p_tempvec3, *p_regressors3, *p_stored_move3, *p_stored_move3_2;
__device__ v4 * p_tempvec4, *p_regressors4, *p_stored_move4;
__device__ f64_vec3 * p_prev_move3;
__device__ f64 * p_SS, * p_epsilon_x, * p_epsilon_y, * p_epsilon_z, *p_d_eps_by_d_beta_x_, *p_d_eps_by_d_beta_y_,
			*p_d_eps_by_d_beta_z_;
f64_vec3 * p_tempvec3host;
f64 * p_SS_host;
f64_vec2 * p_tempvec2_host, *p_tempvec2_host2, *p_tempvec2_host3;
__device__ short * sz_who_vert_vert;
__device__ long * p_indicator;
__device__ f64 * p_Jacobian_list;
__device__ NTrates * p_store_NTFlux;
#define SQUASH_POINTS  24
__device__ f64 * p_matrix_blocks;
__device__ f64 * p_vector_blocks;
f64 * p_matrix_blocks_host, *p_vector_blocks_host;
f64 * p_sum_product_matrix_host;
__device__ Symmetric3 *p_sum_product_matrix3;
Symmetric3 * p_sum_product_matrix_host3;
f64_vec3 * p_eps_against_deps_host;
f64 * p_eps_against_d_eps_host;
v4 * p_tempvec4_host, * p_tempvec4_2_host;
__device__ f64_vec3 * p_eps_against_deps;
__device__ f64 * p_sum_product_matrix;
__device__ double * d_work;

#define p_slot1n p_Ap_n
#define p_slot1i p_Ap_i
#define p_slot1e p_Ap_e
#define p_slot2n p_d_eps_iz_by_d_beta_i
#define p_slot2i p_d_eps_iz_by_d_beta_e
#define p_slot2e p_d_eps_ez_by_d_beta_i

__device__ f64 * p_Tn, *p_Ti, *p_Te, *p_Ap_n, *p_Ap_i, *p_Ap_e, * p_NnTn, * p_NTi, * p_NTe,
			* stored_Az_move;

#define p_Tik p_NTi
#define p_Tek p_NTe
#define p_Tnk p_NnTn

// Don't forget made this union.

__device__ T3  *zero_array;
__device__ f64 * p_Ax;
bool * p_boolhost;
f64 * p_temphost1, *p_temphost2, *p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
long * p_longtemphost;
f64_vec2 * p_GradTe_host, *p_GradAz_host;
f64_vec3 * p_B_host, *p_MAR_ion_host, *p_MAR_elec_host, *p_MAR_ion_compare, *p_MAR_elec_compare,
*p_MAR_neut_host,*p_MAR_neut_compare;
__device__ nn *p_nn_ionrec_minor;
__device__ OhmsCoeffs * p_OhmsCoeffs;
OhmsCoeffs * p_OhmsCoeffs_host; // for display
__device__ f64 * p_Iz0, *p_sigma_Izz;
__device__ f64_vec3 * p_vn0;
__device__ v4 * p_v0;
__device__ nvals * p_one_over_n, *p_one_over_n2;
__device__ f64_vec3 * p_MAR_neut, *p_MAR_ion, *p_MAR_elec;
__device__ f64 * p_Az, *p_LapAz, *p_LapCoeffself, *p_Azdot0, *p_gamma, *p_LapJacobi,
*p_Jacobi_x, *p_epsilon, *p_LapAzNext,
*p_Integrated_div_v_overall,
*p_Div_v_neut, *p_Div_v, *p_Div_v_overall, *p_ROCAzdotduetoAdvection,
*p_ROCAzduetoAdvection, *p_AzNext,
*p_kappa_n,*p_kappa_i,*p_kappa_e,*p_nu_i,*p_nu_e;
__device__ bool * p_bFailed, *p_boolarray, * p_boolarray2, *p_boolarray_block;

__device__ f64 * p_epsilon_heat, *p_Jacobi_heat,
				*p_sum_eps_deps_by_dbeta_heat, *p_sum_depsbydbeta_sq_heat, *p_sum_eps_eps_heat;
f64  *p_sum_eps_deps_by_dbeta_host_heat, *p_sum_depsbydbeta_sq_host_heat, *p_sum_eps_eps_host_heat;

__device__ f64_vec4 * p_d_eps_by_dbetaJ_n_x4, *p_d_eps_by_dbetaJ_i_x4, *p_d_eps_by_dbetaJ_e_x4,
*p_d_eps_by_dbetaR_n_x4, *p_d_eps_by_dbetaR_i_x4, *p_d_eps_by_dbetaR_e_x4,
*p_sum_eps_deps_by_dbeta_J_x4, *p_sum_eps_deps_by_dbeta_R_x4;
__device__ f64 * p_sum_depsbydbeta_8x8;
f64 * p_sum_depsbydbeta_8x8_host;
f64_vec4 *p_sum_eps_deps_by_dbeta_J_x4_host, *p_sum_eps_deps_by_dbeta_R_x4_host;

__device__ species3 *p_nu_major;
__device__ f64_vec2 * p_GradAz, *p_GradTe;
__device__ ShardModel *p_n_shards, *p_n_shards_n;
__device__ NTrates *NT_addition_rates_d, *NT_addition_tri_d, *NT_addition_rates_d_temp2;
long numReverseJzTriangles;
__device__ f64 *p_sum_eps_deps_by_dbeta, *p_sum_depsbydbeta_sq, *p_sum_eps_eps;
f64  *p_sum_eps_deps_by_dbeta_host, *p_sum_depsbydbeta_sq_host, *p_sum_eps_eps_host;
__device__ char * p_was_vertex_rotated, *p_triPBClistaffected;
__device__ T3 * p_T_upwind_minor_and_putative_T;

__device__ f64 * p_d_eps_by_dbeta_n, *p_d_eps_by_dbeta_i, *p_d_eps_by_dbeta_e,
*p_d_eps_by_dbetaR_n, *p_d_eps_by_dbetaR_i, *p_d_eps_by_dbetaR_e,
*p_Jacobi_n, *p_Jacobi_i, *p_Jacobi_e, *p_epsilon_n, *p_epsilon_i, *p_epsilon_e,
*p_coeffself_n, *p_coeffself_i, *p_coeffself_e; // Are these fixed or changing with each iteration?

__device__ f64_tens3 * p_InvertedMatrix_i, *p_InvertedMatrix_e, *p_InvertedMatrix_n;
__device__ f64_vec3 * p_MAR_ion2, *p_MAR_elec2, * p_vJacobi_i, * p_vJacobi_e, *p_vJacobi_n,
	* p_d_eps_by_d_beta_i, *p_d_eps_by_d_beta_e;

__device__ f64_vec2 *p_d_epsxy_by_d_beta_i, *p_d_epsxy_by_d_beta_e;
__device__ f64 *p_d_eps_iz_by_d_beta_i, *p_d_eps_ez_by_d_beta_i,
			   *p_d_eps_iz_by_d_beta_e, *p_d_eps_ez_by_d_beta_e;

f64_vec3 * p_sum_vec_host;
__device__ NTrates * NT_addition_rates_d_temp, * store_heatcond_NTrates;
__device__ f64_vec2 * p_epsilon_xy;
__device__ f64 * p_epsilon_iz, *p_epsilon_ez,
*p_sum_eps_deps_by_dbeta_i, *p_sum_eps_deps_by_dbeta_e, *p_sum_depsbydbeta_i_times_i,
*p_sum_depsbydbeta_e_times_e, *p_sum_depsbydbeta_e_times_i;
f64 * p_sum_eps_deps_by_dbeta_i_host, * p_sum_eps_deps_by_dbeta_e_host, * p_sum_depsbydbeta_i_times_i_host,
*p_sum_depsbydbeta_e_times_e_host, *p_sum_depsbydbeta_e_times_i_host;

__device__ f64 *p_sum_eps_deps_by_dbeta_J, *p_sum_eps_deps_by_dbeta_R, *p_sum_depsbydbeta_J_times_J,
	*p_sum_depsbydbeta_R_times_R, *p_sum_depsbydbeta_J_times_R,
	*p_sum_eps_deps_by_dbeta_x8;
f64 * p_sum_eps_deps_by_dbeta_J_host, *p_sum_eps_deps_by_dbeta_R_host, *p_sum_depsbydbeta_J_times_J_host,
	*p_sum_depsbydbeta_R_times_R_host, *p_sum_depsbydbeta_J_times_R_host;
__device__ f64 * ionmomflux_eqns, *elecmomflux_eqns;

__device__ AAdot *p_AAdot_target, *p_AAdot_start;
__device__ f64_vec3 * p_v_n_target, *p_v_n_start;
__device__ v4 * p_vie_target, *p_vie_start;

int * iRing;
__device__ int * p_iRing;
bool * bSelected;
__device__ bool * p_selectflag;
short * p_equation_index_host;
__device__ short *p_equation_index;

TriMesh * pTriMesh;

f64 * temp_array_host;
f64 tempf64;
FILE * fp_traj;

void GosubAccelerate(long iSubcycles, f64 hsub, cuSyst * pX_use, cuSyst * pX_intermediate);

//f64 Tri_n_n_lists[NMINOR][6],Tri_n_lists[NMINOR][6];
// Not clear if I ended up using Tri_n_n_lists - but it was a workable way if not.

long * address;
f64 * f64address;
size_t uFree, uTotal;
extern real evaltime;
extern cuSyst cuSyst_host;

void SafeExit(long code);

//
//kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
//(p_epsilon_n, p_epsilon_i, p_epsilon_e,
//	p_temp1, p_temp2, p_temp3);
//Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
//SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//for (iTile = 0; iTile < numTilesMajorClever; iTile++)
//{
//	SS_n += p_temphost1[iTile];
//	SS_i += p_temphost2[iTile];
//	SS_e += p_temphost3[iTile];
//}

/*
void RunBackwardForHeat_BiCGstab(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESHOLD 0.33

	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	f64 dot2_n, dot2_i, dot2_e;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	printf("\nBiCGStab for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 SS_n, SS_i, SS_e, oldSS_n, oldSS_i, oldSS_e, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e;

	// seed: just set T to T_k.
	//kernelUnpacktoNT << < numTilesMajorClever, threadsPerTileMajorClever >> >
	//	(p_NnTn, p_NTi, p_NTe, p_T_k, 
	//		pX_use->p_AreaMajor,
	//		pX_use->p_n_major);
	//Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T_k);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");

	// 1. Compute epsilon:

	// epsilon = T_k+1 - T_k - (h/N)rates.NT;
	// epsilon = b - A T_k+1
	// Their Matrix A = -identity + (h/N) ROC NT due to heat flux
	// Their b = - T_k

	// Compute heat flux given p_T	
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	//
	//	kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
	//		(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
	//			pX_use->p_AreaMajor,
	//			pX_use->p_n_major);
	//	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,  // using vert indices

			pX_use->p_T_minor + BEGINNING_OF_CENTRAL, // not used!
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i, p_kappa_e,
			p_nu_i, p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	// Let's think about this carefully. If we could do it with the same data loaded each vertex for neigh list it would actually save time.
	// We rely on the fact that loading the vertex list data is contiguous fetch.
	// And we do store it for 3 species - so let's stick with all in 1.

	kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_Tn, p_Ti, p_Te,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			NT_addition_rates_d_temp // it's especially silly having a whole struct of 5 instead of 3 here.
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

	// Copy to r0hat
	cudaMemcpy(p_temp4, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_temp5, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_temp6, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	rho_prev_n = 1.0; alpha_n = 1.0; omega_n = 1.0;
	rho_prev_i = 1.0; alpha_i = 1.0; omega_i = 1.0;
	rho_prev_e = 1.0; alpha_e = 1.0; omega_e = 1.0;

	cudaMemset(p_Ap_n_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_n_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0
	cudaMemset(p_Ap_i_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_i_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0
	cudaMemset(p_Ap_e_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_e_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0
	 
	bool bContinue = true;
	iIteration = 1;
	do {
		//rho_n = dotproduct(p_temp4, p_epsilon_n); // r_array is for iIter-1
		//rho_i = dotproduct(p_temp5, p_epsilon_i); // r_array is for iIter-1
		//rho_e = dotproduct(p_temp6, p_epsilon_e); // r_array is for iIter-1
		dotproducts(p_temp4, p_epsilon_n,
			p_temp5, p_epsilon_i,
			p_temp6, p_epsilon_e,
			rho_n, rho_i, rho_e);
		beta_n = rho_n*alpha_n / (rho_prev_n*omega_n);
		beta_i = rho_i*alpha_i / (rho_prev_i*omega_i);
		beta_e = rho_e*alpha_e / (rho_prev_e*omega_e);
		UpdateRegressorBiCG(p_p_n_BiCG, p_epsilon_n, beta_n, omega_n, p_Ap_n_BiCG); // omega_i-1
		UpdateRegressorBiCG(p_p_i_BiCG, p_epsilon_i, beta_i, omega_i, p_Ap_i_BiCG); // omega_i-1
		UpdateRegressorBiCG(p_p_e_BiCG, p_epsilon_e, beta_e, omega_e, p_Ap_e_BiCG); // omega_i-1
				// this is now p_i but still v_i-1

		setequaltoAtimes(p_Ap_n_BiCG, p_Ap_i_BiCG, p_Ap_e_BiCG,
			p_p_n_BiCG, p_p_i_BiCG, p_p_e_BiCG,
			p_T_k, hsub, pX_use); // can we fill it in?
		dotproducts(p_temp4, p_Ap_n_BiCG,
			p_temp5, p_Ap_i_BiCG,
			p_temp6, p_Ap_e_BiCG,
			dot_n, dot_i, dot_e);
		alpha_n = rho_n / dot_n;
		alpha_i = rho_i / dot_i;
		alpha_e = rho_e / dot_e;

		// regressor=h, x is at i-1 
		LinearCombo(p_regressor_n, p_Tn, alpha_n, p_p_n_BiCG);
		LinearCombo(p_regressor_i, p_Ti, alpha_i, p_p_i_BiCG);
		LinearCombo(p_regressor_e, p_Te, alpha_e, p_p_e_BiCG);
		LinearCombo(p_s_n_BiCG, p_epsilon_n, -alpha_n, p_Ap_n_BiCG); // r_i-1 is in epsilon
		LinearCombo(p_s_i_BiCG, p_epsilon_i, -alpha_i, p_Ap_i_BiCG); // r_i-1 is in epsilon
		LinearCombo(p_s_e_BiCG, p_epsilon_e, -alpha_e, p_Ap_e_BiCG); // r_i-1 is in epsilon
		setequaltoAtimes(p_As_n, p_As_i, p_As_e,
			p_s_n_BiCG, p_s_i_BiCG, p_s_e_BiCG,
			p_T_k, hsub, pX_use);
		dotproducts(p_s_n_BiCG, p_As_n,
			p_s_i_BiCG, p_As_i,
			p_s_e_BiCG, p_As_e,
			omega_n, omega_i, omega_e);
		SumsOfSquares(p_As_n, p_As_i, p_As_e, SS_n, SS_i, SS_e);
		omega_n /= SS_n;
		omega_i /= SS_i;
		omega_e /= SS_e;
		LinearCombo(p_Tn, p_regressor_n, omega, p_s_BiCG);
		LinearCombo(p_epsilon_n, p_s_BiCG, -omega, p_As);
		SumsOfSquares(p_epsilon_n, p_epsilon_i, p_epsilon_e, SS_n, SS_i, SS_e);
		L2eps_n = sqrt(SS_n / NUMVERTICES);
		L2eps_i = sqrt(SS_i / NUMVERTICES);
		L2eps_e = sqrt(SS_e / NUMVERTICES);
		printf("L2eps %1.9E %1.9E %1.9E \n", L2eps_n, L2eps_i, L2eps_e);
		++iIteration;
	} while (iIteration < 200);

	while (1) getch();
}*/

bool TestDomainPosHost(f64_vec2 pos)
{
	return (
		(pos.x*pos.x + pos.y*pos.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
		&&
		(pos.x*pos.x + (pos.y - CATHODE_ROD_R_POSITION)*(pos.y - CATHODE_ROD_R_POSITION) > CATHODE_ROD_RADIUS*CATHODE_ROD_RADIUS)
		);
}

void SolveBackwardAzAdvanceCG(f64 hsub,
	f64 * pAz_k,
	f64 * pAzdot0, f64 * pgamma,
	f64 * p_Solution, 
	
	cuSyst * pX_use)// AreaMinor will be set in pX_use when we call GetLapMinor
{
	// The principal reason this routine stutters and doesn't work is that we would have to multiply by AreaMinor
	// to get equation symmetry. ?
	
	// I don't suppose we can use last move as a guess from which to generate epsilon and start off AzCG?
	// Running a regression with it is almost as cheap anyway.

	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 RSS, oldRSS, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e, RSS_n, RSS_i, RSS_e;
	int iTile;

#define p_regressor p_temp2
#define p_sqrthgamma_Az p_temp1

	cudaMemset(p_temp4, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_temp5, 0, sizeof(f64)*NMINOR);
	//
	//// Do this just in case what we were sent was deficient:
	//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
	//	pX_use->p_info,
	//	pX_use->p_tri_neigh_index,
	//	p_Solution);
	//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz"); // can leave this here but
	//// NEVER LOOK INTO FRILL -- can it be done? yep
 //
	// Get Area and Lap:
	kernelGetLap_minor_SYMMETRIC << < numTriTiles, threadsPerTileMinor >> >
		(pX_use->p_info,
			p_Solution,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz, // INTEGRATED
			p_AreaMinor_cc,
			false);
	Call(cudaThreadSynchronize(), "cudaTS GetLapminor");
	
	kernelDividebyroothgamma << < numTilesMinor, threadsPerTileMinor >> >
		(p_sqrthgamma_Az,
			p_sqrtfactor, // sqrt(h gamma / area) -- surely needed!			
			p_Solution, 
			hsub, pgamma, p_AreaMinor_cc); 
	// divide by sqrt h gamma over area !!		
	Call(cudaThreadSynchronize(), "cudaTS Divideroothgamma");

	//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
	//	pX_use->p_info,
	//	pX_use->p_tri_neigh_index,
	//	p_sqrthgamma_Az);
	//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
	//// reset frills because we will not have been able to divide by gamma in frills. But the value is understood.

	// 1. Compute epsilon:
	
	kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
		(hsub, pX_use->p_info, 
			p_Solution, // in Az units
			pAz_k, pAzdot0, pgamma,
			p_LapAz,
			p_epsilon,
			p_sqrtfactor, // needed!
			p_bFailed, false
			); // must include multiply epsilon by h gamma
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonAz");

//	kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
//		pX_use->p_info,
//		pX_use->p_tri_neigh_index,
//		p_epsilon);
//	Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

	// Does ResetFrills performed on epsilon make it symmetric or not? :
	// Frill values should be never used.

	// 2. Regressor = epsilon
	cudaMemcpy(p_regressor, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	
	// Take RSS :
	kernelAccumulateSumOfSquares1 << < numTilesMinor, threadsPerTileMinor>> >
		(p_epsilon, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	RSS = 0.0;
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (int iTile = 0; iTile < numTilesMinor; iTile++)
	{
		RSS += p_temphost3[iTile];
	}
	oldRSS = RSS; 
	long iIteration = 0;
	printf("iIteration %d : L2eps[sqrt(h gamma)Az] %1.11E  /n",
		iIteration, sqrt(RSS / (f64)NMINOR));
	
	bool bContinue = true;
	do {
		// remember -Ax is what appears in epsilon.
		
		kernelCreateAzbymultiplying << < numTilesMinor, threadsPerTileMinor >> >
			(p_Solution, p_regressor, p_sqrtfactor);
		Call(cudaThreadSynchronize(), "cudaTS CreateAzbymultiplying");
		// multiply by sqrt(h gamma / area) to give Az
//
//		kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
//			pX_use->p_info,
//			pX_use->p_tri_neigh_index,
//			p_Solution);
//		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

		// now careful : pass Azdot0 = 0, Az_k = 0 and we can get -sqrt(h gamma) Lap (h gamma (regressor))
		kernelGetLap_minor_SYMMETRIC << < numTriTiles, threadsPerTileMinor >> >
			(pX_use->p_info,
				p_Solution,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_LapAz,
				p_AreaMinor_cc,
				false);
		Call(cudaThreadSynchronize(), "cudaTS GetLapminor");
		kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
			(hsub, pX_use->p_info,
				p_Solution, // in Az units
				p_temp4, p_temp5, // zero, zero
				pgamma, 
				p_LapAz, // lap of Az(regressor)
				p_Ax,
				p_sqrtfactor,
				p_bFailed, false
				); // must include multiply epsilon by h gamma
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon Regressor");
		//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		//	pX_use->p_info,
		//	pX_use->p_tri_neigh_index,
		//	p_Ax);
		//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

		// epsilon = b-Ax so even though our own A includes a minus, we now negate
		NegateVector << <numTilesMinor, threadsPerTileMinor >> > (p_Ax);
		Call(cudaThreadSynchronize(), "cudaTS NegateVector");
		
		kernelAccumulateDotProduct << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressor, p_Ax, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProduct");
		f64 xdotAx = 0.0; 
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			xdotAx += p_temphost3[iTile];
		};

		f64 alpha = RSS / xdotAx;
		printf("alpha %1.9E ... ", alpha);

		VectorAddMultiple1 << < numTilesMinor, threadsPerTileMinor >> > (
			p_sqrthgamma_Az, alpha, p_regressor);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiple");		
		kernelCreateAzbymultiplying << < numTilesMinor, threadsPerTileMinor >> >
			(p_Solution, p_sqrthgamma_Az, p_sqrtfactor);
		Call(cudaThreadSynchronize(), "cudaTS CreateAzbymultiplying");
		//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		//	pX_use->p_info,
		//	pX_use->p_tri_neigh_index,
		//	p_Solution);
		//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
		// should be doing nothing...

		// Update Epsilon: or can simply recalculate eps = b - A newx
		// Sometimes would rather update epsilon completely:


		// Note: speedup to implement here, in comparing with R8LS.


		if (iIteration % 1 == 0) {

			kernelGetLap_minor_SYMMETRIC << < numTriTiles, threadsPerTileMinor >> >
				(pX_use->p_info,
					p_Solution,
					pX_use->p_izTri_vert,
					pX_use->p_izNeigh_TriMinor,
					pX_use->p_szPBCtri_vert,
					pX_use->p_szPBC_triminor,
					p_LapAz,
					p_AreaMinor_cc,
					false);
			Call(cudaThreadSynchronize(), "cudaTS GetLapminor");

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
			kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
				(hsub, pX_use->p_info,
					p_Solution, // in Az units
					pAz_k, p_Azdot0, 
					pgamma,
					p_LapAz,
					p_epsilon, 
					p_sqrtfactor,
					p_bFailed, true
					); // must include multiply epsilon by rt h gamma
			// but do test by dividing epsilon by rt h gamma; compare to Solution
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon eps");
			//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
			//	pX_use->p_info,
			//	pX_use->p_tri_neigh_index,
			//	p_epsilon);
			//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

			// Now test for convergence:
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
			bContinue = false;
			for (iTile = 0; iTile < numTilesMinor; iTile++)
				if (p_boolhost[iTile]) bContinue = true;
		//	if (bContinue == true) printf("failed tests\n");

		} else {
			bContinue = true;
			VectorAddMultiple1 << <numTilesMinor, threadsPerTileMinor >> >
				( p_epsilon, -alpha, p_Ax);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiple eps");
			// it should be true that we setted p_Ax[outer_frill] = 0
			p_boolhost[0] = true;
		};

		// Take RSS :
		kernelAccumulateSumOfSquares1 << < numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		RSS = 0.0;
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			RSS += p_temphost3[iTile];
		}
		f64 ratio = RSS / oldRSS;
		oldRSS = RSS;
		

		//VectorAddMultiple1 << <numTilesMinor, threadsPerTileMinor >> >
		//	(p_regressor, ratio, p_epsilon);

		//Whereas wikipedia thinks  it's x_k+1 = eps_k+1 + beta x_k.

		CreateConjugateRegressor << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressor, ratio, p_epsilon);
		Call(cudaThreadSynchronize(), "cudaTS CreateConjugateRegressor");

		//kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		//	pX_use->p_info,
		//	pX_use->p_tri_neigh_index,
		// 	p_regressor); // should be unneeded now since epsilon is frilled
		//Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
		// we affect the frills as well --- in case eps is not frilled, regressor frilled
		
		printf("iIteration %d L2eps[sqrt[h gamma] units] %1.10E \n", iIteration, sqrt(RSS / (real)NMINOR));
		iIteration++;
		 
		if (bContinue == false) printf("all tests ok\n");
		 
	} while ((iIteration < 4000) && (bContinue));
	 
	if (iIteration == 4000) { SafeExit(768); };
		
	// MAKE SURE WE REPOPULATE LapAz after calling this routine.
	// It is the integrated Lap here.
	// We could proceed just to divide it here, for luck.
	// AND in fact WE MUST USE SYMMETRIC WHEN WE DO THE ADVANCE !.


#undef p_regressor
#undef p_sqrthgamma_Az
}

 
void SolveBackwardAzAdvanceJ3LS(f64 hsub,
	f64 * pAz_k,
	f64 * p_Azdot0, f64 * p_gamma,
	f64 * p_AzNext, f64 * p_LapCoeffself,
	cuSyst * pX_use)
{  
	f64 L2eps;
	f64 beta[SQUASH_POINTS];
	Tensor3 mat;
	f64 RSS;
	int iTile;
	char buffer[256];
	int iIteration = 0;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	f64 matrix[SQUASH_POINTS*SQUASH_POINTS];
	f64 vector[SQUASH_POINTS];

	long iMax = 0;
	bool bContinue = true;
	 
//	printf("iIteration = %d\n", iIteration);
	// 1. Create regressor:
	// Careful with major vs minor + BEGINNING_OF_CENTRAL:

	//GlobalSuppressSuccessVerbosity = true;

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_AzNext);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");
	// if DIRICHLET, set outer ones to 0 just for sake of it.	 
	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_AzNext,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapAzNext
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");
	// Put 0 line through outer frill centroid radius.
	// Squash the outermost vertex cell.
	
	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
	kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
		(hsub, // ?
			pX_use->p_info,
			p_AzNext, pAz_k,
			p_Azdot0, p_gamma,
			p_LapCoeffself, p_LapAzNext,
			p_epsilon, p_Jacobi_x,
			//this->p_AAdot, // to use Az_dot_k for equation);
			p_bFailed
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
	// Do not recognize Jz outside 5cm
	// Do include self-effect looking at inner frill which is mirror
	// Do include self-effect looking at outer frill which is zero
	
	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");
	 
	f64 L4L2ratio = 0.0;

//	FILE * fpdbg = fopen("J3LS_2.txt", "w");
	L2eps = -1.0;
	bool bSpitOutErrorAfter = false;
	long iMax0;
	do
	{
		// Now we want to create another regressor, let it be called p_regressor_n
		// Let p_Jacobi_x act as AzNext

		if (((iIteration > 4) && (L4L2ratio > 10.0) && (iIteration % 2 == 0))
			|| ((iIteration > 1600) && (iIteration % 4 == 0))) // if things are messed up, try it anyway
		{
			
			printf("\nDoing the smash! iteration %d\n", iIteration);

			// Alternative: smoosh 24 points

			// find maximum
			cudaMemset(p_indicator, 0, sizeof(long)*NMINOR);
			int number_set = 0; // now we're going to have to number them as 1 through 24 .. be careful.
			do {
				kernelReturnMaximumInBlock << <numTilesMinor, threadsPerTileMinor >> > (
					p_epsilon,
					p_temp1, // max in block
					p_longtemp,
					p_indicator // long*NMINOR : if this point is already used, do not pick it up.
					);
				cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
				f64 maximum = 0.0;
				long iMax = 0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
				{
					if (p_temphost1[iTile] > maximum) {
						maximum = p_temphost1[iTile];
						iMax = p_longtemphost[iTile];
					};
				};
				long ii = number_set + 1;
				cudaMemcpy(&(p_indicator[iMax]), &ii, sizeof(long), cudaMemcpyHostToDevice);

				//if (number_set == 0)
				//{
				//	long * longaddress;
				//	Call(cudaGetSymbolAddress((void **)(&longaddress), MyMaxIndex),
				//		"cudaGetSymbolAddress((void **)(&longaddress), MyMaxIndex)");
				//	Call(cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice),
				//		"cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice)");
				//	f64 tempf64;
				//	cudaMemcpy(&tempf64, &(p_epsilon[iMax]), sizeof(f64), cudaMemcpyDeviceToHost);
				//	printf("\nError at iMax %1.14E\n\n", tempf64);
				//	bSpitOutErrorAfter = true;
				//	iMax0 = iMax;
				//};

				//printf("%d: %d  || ", ii, iMax);

				number_set++;
				// Just do this 24 times ... dumbest way possible but nvm.
				
				// Quicker way (develop later) :				
				// Just recruit neighbours until we get to 24?
				//if (iMax >= BEGINNING_OF_CENTRAL) {
				//	cudaMemcpy(p_izTri_host, &(pX_use->p_izTri_vert[MAXNEIGH*iMax]), sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
				//		// Only pick up neighbours if they are not already used.
				//		// Stop when we reach SMASH_POINTS
				//	{
				//		number_set++;
				//	}
				//} else {
				//	{
				// Only pick up neighbours if they are not already used.
				//		// Stop when we reach SMASH_POINTS
				//		number_set++;
				//	}
				//};
			} while (number_set < SQUASH_POINTS);
			//printf("\n");

			kernelComputeJacobianValues << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				
			//	p_AzNext,pAz_k,p_Azdot0,  // needed?
				p_gamma,   // needed?
				hsub,
				p_indicator,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_Jacobian_list // needs to be NMINOR * SQUASH_POINTS
				);
			Call(cudaThreadSynchronize(), "cudaTS CollectJacobian");
			
			AggregateSmashMatrix << <numTilesMinor * 2, threadsPerTileMajor >> > (
				p_Jacobian_list,
				p_epsilon,
				p_matrix_blocks,
				p_vector_blocks
				);
			Call(cudaThreadSynchronize(), "cudaTS CollectMatrix");

			cudaMemcpy(p_matrix_blocks_host, p_matrix_blocks,
				sizeof(f64)*SQUASH_POINTS*SQUASH_POINTS*numTilesMinor*2, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_vector_blocks_host, p_vector_blocks,
				sizeof(f64)*SQUASH_POINTS*numTilesMinor * 2, cudaMemcpyDeviceToHost);
			memset(matrix, 0, sizeof(f64)*SQUASH_POINTS*SQUASH_POINTS);
			memset(vector, 0, sizeof(f64)*SQUASH_POINTS);
			for (iTile = 0; iTile < numTilesMinor * 2; iTile++)
			{
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//		if (p_matrix_blocks_host[j * SQUASH_POINTS + j + iTile*SQUASH_POINTS*SQUASH_POINTS] != 0.0) 
			//			printf("iTile %d contrib to (%d, %d) %1.9E | \n", iTile,j,j, p_matrix_blocks_host[j*SQUASH_POINTS + j + iTile*SQUASH_POINTS*SQUASH_POINTS]);
				 
				for (int i = 0; i < SQUASH_POINTS; i++)
				{
					for (int j = 0; j < SQUASH_POINTS; j++)
						matrix[i*SQUASH_POINTS+j] += p_matrix_blocks_host[i*SQUASH_POINTS+j+iTile*SQUASH_POINTS*SQUASH_POINTS];
					
					// Note that the matrix is symmetric so i, j order doesn't matter anyway.
					vector[i] -= p_vector_blocks_host[i+iTile*SQUASH_POINTS];

					// INSERTED THE MINUS HERE.

				};
			}; 
		//	printf("\n");


			Matrix_real matLU;
			matLU.Invoke(SQUASH_POINTS);
			
			for (int i = 0; i < SQUASH_POINTS; i++)
				for (int j = 0; j < SQUASH_POINTS; j++)
					matLU.LU[i][j] = matrix[i*SQUASH_POINTS+j];

			matLU.LUdecomp();
			matLU.LUSolve(vector, beta); // solving Ax = b and it's (b, x).
			
			//	printf("===================\n");
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//		printf("%d : change %1.12E \n", j, beta[j]);
			//	printf("===================\n");

			CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, SQUASH_POINTS * sizeof(f64)));
			// proper name for the result.
			// But beta is the set of coefficients on a set of individual dummies
			kernelAddToAz << <numTilesMinor, threadsPerTileMinor >> > (
				p_indicator,
				p_AzNext
				);
			Call(cudaThreadSynchronize(), "cudaTS AddToAz");
			// Think we probably are missing a minus: did we include it in the RHS vector?

			kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info, pX_use->p_tri_neigh_index,
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");
/*
#ifdef LAPACKE
			lapack_int ipiv[SQUASH_POINTS];
			lapack_int Nrows = SQUASH_POINTS,
				Ncols = SQUASH_POINTS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = SQUASH_POINTS, info;

			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
				Nrows, 1, matrix,
				Ncols, ipiv, //sum_eps_deps_by_dbeta_vector
				vector, Nrhscols);

				
			// Check for the exact singularity :
			if (info > 0) {
				printf("The diagonal element of the triangular factor of A,\n");
				printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n\a");
				getch();
			} 	else {
			//	printf("LAPACKE_dgesv ran successfully.\n");
				memcpy(beta, vector, SQUASH_POINTS * sizeof(f64));

			//	printf("===================\n");
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//		printf("%d : change %1.12E \n", j, beta[j]);
			//	printf("===================\n");
				
				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, SQUASH_POINTS * sizeof(f64)));
				// proper name for the result.
				// But beta is the set of coefficients on a set of individual dummies
				kernelAddToAz << <numTilesMinor, threadsPerTileMinor >> > (
					p_indicator,
					p_AzNext
					);
				Call(cudaThreadSynchronize(), "cudaTS AddToAz");
				// Think we probably are missing a minus: did we include it in the RHS vector?

				kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
					pX_use->p_info, pX_use->p_tri_neigh_index,
					p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");
			}

			#endif
			*/

		} else {

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_Jacobi_x,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_LapJacobi
				);
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

			kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info,
				hsub,
				p_Jacobi_x,
				p_LapJacobi,
				p_LapCoeffself,
				p_gamma,
				p_regressor_n);
			Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
			// Look over it just to be sure what applies.

			kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info, pX_use->p_tri_neigh_index,
				p_regressor_n);
			Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_regressor_n,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_temp4
				);
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 2");

			//FILE * fpdbg = fopen("LapAz3.txt", "w");
			//cudaMemcpy(p_temphost1, p_LapAzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost2, p_LapJacobi, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost3, p_temp4, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost4, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost5, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//for (long iRow = 0; iRow < NMINOR; iRow++)
			//{
			//	fprintf(fpdbg, "%d LapAz %1.14E LapJacobi %1.14E temp4 %1.14E epsilon %1.14E Az %1.14E\n", iRow, p_temphost1[iRow], p_temphost2[iRow], p_temphost3[iRow], p_temphost4[iRow],
			//		p_temphost5[iRow]);
			//};
			//fclose(fpdbg);


			//			kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
			//				pX_use->p_info,
			//				hsub,
			//				p_regressor_n,
			//				p_temp4,
			//				p_LapCoeffself,
			//				p_gamma,
			//				p_regressor_i);
			//			Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
			//

			MultiplyVector << <numTilesMinor, threadsPerTileMinor >> >
				(p_Jacobi_x, p_epsilon, p_regressor_i);
			Call(cudaThreadSynchronize(), "cudaTS Multiply Jacobi*epsilon");

			// Wipe out regressor_i with epsilon: J2RLS:
			// Doesn't help.
			// cudaMemcpy(p_regressor_i, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

			kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info, pX_use->p_tri_neigh_index,
				p_regressor_i);
			Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_regressor_i,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_temp5
				); 
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

			// Okay ... now we need to do the routine that creates the matrix deps/dbeta_i deps/dbeta_j
			// and the vector against epsilon
			  
			kernelAccumulateMatrix << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info,
				hsub,
				p_epsilon,
				p_Jacobi_x,
				p_regressor_n,
				p_regressor_i,
				p_LapJacobi,
				p_temp4,
				p_temp5,
				p_gamma,

				p_temp1, // sum of matrices, in lots of 6
				p_eps_against_deps
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateMatrix");

			// Now take 6 sums
			f64 sum_mat[6];
			f64_vec3 sumvec(0.0, 0.0, 0.0);
			memset(sum_mat, 0, sizeof(f64) * 6);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64) * 6 * numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_vec_host, p_eps_against_deps, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{ 
				sum_mat[0] += p_temphost1[iTile * 6 + 0];
				sum_mat[1] += p_temphost1[iTile * 6 + 1];
				sum_mat[2] += p_temphost1[iTile * 6 + 2];
				sum_mat[3] += p_temphost1[iTile * 6 + 3];
				sum_mat[4] += p_temphost1[iTile * 6 + 4];
				sum_mat[5] += p_temphost1[iTile * 6 + 5];
				sumvec += p_sum_vec_host[iTile];
				 
				//printf("iTile %d sum_mat +=  %1.9E %1.9E %1.9E %1.9E %1.9E %1.9E sum_vec %1.9E \n",
				//	iTile, p_temphost1[iTile * 6 + 0],
				//	p_temphost1[iTile * 6 + 1],
				//	p_temphost1[iTile * 6 + 2],
				//	p_temphost1[iTile * 6 + 3],
				//	p_temphost1[iTile * 6 + 4],
				//	p_temphost1[iTile * 6 + 5],
				//	p_sum_vec_host[iTile]);
			};

		//	printf("sum_mat %1.9E %1.9E %1.9E %1.9E %1.9E %1.9E sum_vec %1.9E \n",
		//		sum_mat[0], sum_mat[1], sum_mat[2], sum_mat[3], sum_mat[4], sum_mat[5], sumvec);

			if (sumvec != sumvec){
				printf("sum_mat %1.9E %1.9E %1.9E %1.9E %1.9E %1.9E sum_vec %1.9E \n",
						sum_mat[0], sum_mat[1], sum_mat[2], sum_mat[3], sum_mat[4], sum_mat[5], sumvec);
				SafeExit(1195);
			}

			// Now populate symmetric matrix
			f64_tens3 mat, mat2;

			mat.xx = sum_mat[0];
			mat.xy = sum_mat[1];
			mat.xz = sum_mat[2];
			mat.yx = mat.xy;
			mat.yy = sum_mat[3];
			mat.yz = sum_mat[4];
			mat.zx = mat.xz;
			mat.zy = mat.yz;
			mat.zz = sum_mat[5];
			// debug:

	//		mat.yx = 0.0; mat.yy = 1.0; mat.yz = 0.0;
	//		mat.zx = 0.0; mat.zy = 0.0; mat.zz = 1.0;
	//		sumvec.y = 0.0;
	//		sumvec.z = 0.0;
	//
			mat.Inverse(mat2);
			//printf(
			//	" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
			//	" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
			//	" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
			//	mat.xx, mat.xy, mat.xz, sumvec.x,
			//	mat.yx, mat.yy, mat.yz, sumvec.y,
			//	mat.zx, mat.zy, mat.zz, sumvec.z);
			f64_vec3 product = mat2*sumvec;

			beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

			if (iIteration % 1 == 0) 
				printf("iIteration = %d L2eps %1.9E beta %1.8E %1.8E %1.8E \n", iIteration, L2eps, beta[0], beta[1], beta[2]);

			//	printf("Verify: \n");
			//	f64 z1 = mat.xx*beta[0] + mat.xy*beta[1] + mat.xz*beta[2];
			//	f64 z2 = mat.yx*beta[0] + mat.yy*beta[1] + mat.yz*beta[2];
			//	f64 z3 = mat.zx*beta[0] + mat.zy*beta[1] + mat.zz*beta[2];
			//	printf("z1 %1.14E sumvec.x %1.14E | z2 %1.14E sumvec.y %1.14E | z3 %1.14E sumvec.z %1.14E \n",
			//		z1, sumvec.x, z2, sumvec.y, z3, sumvec.z);

				// Since iterations can be MORE than under Jacobi, something went wrong.
				// Try running with matrix s.t. beta1 = beta2 = 0. Do we get more improvement ever or a different coefficient than Jacobi?
				// If we always do better than Jacobi we _could_ still end up with worse result due to different trajectory but this is unlikely 
				// to be the explanation so we should track it down.

			//	cudaMemcpy(p_temphost2, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

			kernelAddRegressors << <numTilesMinor, threadsPerTileMinor >> > (
				p_AzNext,
				beta[0], beta[1], beta[2],
				p_Jacobi_x,
				p_regressor_n,
				p_regressor_i
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelAddRegressors");
		};
		// should have no effect since we applied it to regressor(s).

		// Yet it does have an effect. Is this because initial AzNext wasn't frilled? Or because regressors are not?

		// Yes.

		// ok --- spit out Az
		//char buffer[255];
		//sprintf(buffer, "Az%d.txt", iIteration);
		//cudaMemcpy(p_temphost1, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//
		//FILE * jibble = fopen(buffer, "w");
		//for (int i = 0; i < NMINOR; i++)
		//	fprintf(jibble, "%d Az %1.14E Jac_added %1.14E \n",i, p_temphost1[i], p_temphost2[i]);
		//fclose(jibble);


	//	missing a ResetFrills. Though it follows linearly?


		// 1. Create regressor:
		// Careful with major vs minor + BEGINNING_OF_CENTRAL:

		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_AzNext,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAzNext
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

		//cudaMemcpy(p_temphost6, p_epsilon, NMINOR * sizeof(f64), cudaMemcpyDeviceToHost);

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
		kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
			(hsub, // ?
				pX_use->p_info,
				p_AzNext, pAz_k,
				p_Azdot0, p_gamma,
				p_LapCoeffself, p_LapAzNext,
				p_epsilon, p_Jacobi_x,
				//this->p_AAdot, // to use Az_dot_k for equation);
				p_bFailed
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
		// Check again.


		//// Now printf what's going on at the back:
		//SetConsoleTextAttribute(hConsole, 10);
		// 
		//f64 eps1, eps2, A1, A2, LapA1, LapA2, Azdot01, Azdot02;
		//cudaMemcpy(&eps1, &(p_epsilon[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&eps2, &(p_epsilon[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&A1, &(p_AzNext[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&A2, &(p_AzNext[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&LapA1, &(p_LapAzNext[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&LapA2, &(p_LapAzNext[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&Azdot01, &(p_Azdot0[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&Azdot02, &(p_Azdot0[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("%d: eps %1.9E Az %1.9E Lap %1.9E Azdot0 %1.9E\n",
		//	VERTCHOSEN, eps1, A1, LapA1, Azdot01);
		//printf("%d: eps %1.9E Az %1.9E Lap %1.9E Azdot0 %1.9E\n",
		//	VERTCHOSEN2, eps2, A2, LapA2, Azdot02); 
		//SetConsoleTextAttribute(hConsole, 15);


		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		RSS = 0.0; 
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
		L2eps = sqrt(RSS / (f64)NMINOR);
		if (iIteration % 1 == 0) printf("L2eps: %1.9E .. ", L2eps);

		kernelAccumulateSumOfQuarts << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		f64 RSQ = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSQ += p_temphost3[iTile];
		f64 L4eps = sqrt(sqrt(RSQ / (f64)NMINOR));
		if (iIteration % 10 == 0) printf("L4eps: %1.9E  ratio L4/L2 %1.9E \n", L4eps, L4eps / L2eps);
		L4L2ratio = L4eps / L2eps;

		//if (bSpitOutErrorAfter)
	//	{
//			f64 tempf64;
		//	cudaMemcpy(&tempf64, &(p_epsilon[iMax0]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("\nError at iMax0 %1.14E\n\n", tempf64);
//		}

		if (iIteration == 4001){ //(iIteration > 600) {
			
			// graphs:
			cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_Azdot0, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_regressor_n, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_regressor_i, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost6, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
						
			SetActiveWindow(hWnd);
			RefreshGraphs(X1, AZSOLVERGRAPHS);
			SetActiveWindow(hWnd);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			printf("done graph\n");
		};

		if (iIteration == 4001) {
			// let's print some stats

			cudaMemcpy(p_temphost3, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			// Find max, avg, SD
			long iMinor, iMax = 0;
			f64 max = 0.0;
			f64 sum = 0.0, sumsq = 0.0;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				if (fabs(p_temphost3[iMinor]) > max) {
					max = fabs(p_temphost3[iMinor]);
					iMax = iMinor;
				};
				sum += p_temphost3[iMinor];
				sumsq += p_temphost3[iMinor] * p_temphost3[iMinor];
			}
			structural info;
			f64 var = sumsq / (real)NMINOR - sum*sum / (real)(NMINOR*NMINOR);
			cudaMemcpy(&info, &(pX_use->p_info[iMax]), sizeof(structural), cudaMemcpyDeviceToHost);
			printf("Avg %1.9E Max %1.10E iMax %d flag %d pos %1.9E %1.9E SD %1.9E \n",
				sum / (real)NMINOR, p_temphost3[iMax], iMax, info.flag, info.pos.x, info.pos.y, sqrt(var));
			
			SafeExit(1397);
		}
	



	//	fprintf(fpdbg, "L2eps %1.14E beta %1.14E %1.14E %1.14E \n", L2eps, beta[0], beta[1], beta[2]);

		/*
		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost4, p_temp6, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

		f64 eps_predict;
		sprintf(buffer, "eps_vs_eps%d.txt", iIteration);
		FILE * jibble = fopen(buffer, "w");
		for (int i = 0; i < NMINOR; i++)
		{
			eps_predict = p_temphost6[i] + beta[0] * p_temphost2[i] + beta[1] * p_temphost3[i] + beta[2] * p_temphost4[i];
			fprintf(jibble, "%d eps_predict %1.14E epsilon %1.14E predicted dbyd %1.14E %1.14E %1.14E old_eps %1.14E \n", 
				i, eps_predict, p_temphost1[i],
				p_temphost2[i], p_temphost3[i], p_temphost4[i], p_temphost6[i]);
		}
		fclose(jibble);
		printf("\n\nFile saved\a\n\n");
		*/

		/*sprintf(buffer, "eps%d.txt", iIteration);
		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		jibble = fopen(buffer, "w");
		for (int i = 0; i < NMINOR; i++)
			fprintf(jibble, "%d eps %1.14E\n", i, p_temphost1[i]);
		fclose(jibble);*/

		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		bContinue = false;
		for (iTile = 0; ((p_boolhost[iTile] == 0) && (iTile < numTilesMinor)); iTile++);
		;
		if (iTile < numTilesMinor) {
//			printf("failed test\n");
			bContinue = true;
		};
		iIteration++;

	} while (bContinue);
//	fclose(fpdbg);

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}


void RegressionSeedAz(f64 const hsub,
	f64 * pAz_k,
	f64 * p_AzNext,
	f64 * p_x1, f64 * p_x2, f64 * p_Azdot0, 
	f64 * p_gamma,
	f64 * p_LapCoeffself, 
	cuSyst * pX_use)
{
	f64 L2eps;
	f64 beta[3];
	Tensor3 mat;
	f64 RSS;
	int iTile;
	char buffer[256];
	int iIteration = 0;

	cudaMemcpy(p_AzNext, pAz_k, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	// was this what we were missing?
	// yet it shouldn't have been far out anyway?


	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_x1);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills 1");


	// Create Epsilon for initial state:

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_AzNext,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapAzNext
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

	kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
		(hsub, // ?
			pX_use->p_info,
			p_AzNext, pAz_k,
			p_Azdot0, p_gamma,
			p_LapCoeffself, p_LapAzNext,
			p_epsilon, p_Jacobi_x,                     // try sticking with this Jacobi instead of doing it on move.
			//this->p_AAdot, // to use Az_dot_k for equation);
			p_bFailed
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
	//kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
	//	pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
	//Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

	// DEBUG:
	// cudaMemcpy(p_x1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	// removed it.
	
	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_x2);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills 2");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_x2,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_temp5
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap 1");


	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_x1);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills 2");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_x1,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_temp4
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap 2");

/*
	kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info,
		hsub,
		p_x1,
		p_temp4,
		p_LapCoeffself,
		p_gamma,
		p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
*/

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");
	
	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_Jacobi_x,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapJacobi
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 3");


	kernelAccumulateMatrix << <numTilesMinor, threadsPerTileMinor >> >(
		pX_use->p_info,
		hsub,
		p_epsilon,
		p_x1,       // used epsilon
		p_x2,       // difference of previous soln's
		p_Jacobi_x, // Jacobi of x1
		p_temp4, // Lap of x1
		p_temp5, // Lap of x2  // don't use temp6 ! It is x2!
		p_LapJacobi,
		p_gamma,
		p_temp1, // sum of matrices, in lots of 6
		p_eps_against_deps
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateMatrix");

	// Now take 6 sums
	f64 sum_mat[6];
	f64_vec3 sumvec(0.0, 0.0, 0.0);
	memset(sum_mat, 0, sizeof(f64) * 6);
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64) * 6 * numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_vec_host, p_eps_against_deps, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		sum_mat[0] += p_temphost1[iTile * 6 + 0];
		sum_mat[1] += p_temphost1[iTile * 6 + 1];
		sum_mat[2] += p_temphost1[iTile * 6 + 2];
		sum_mat[3] += p_temphost1[iTile * 6 + 3];
		sum_mat[4] += p_temphost1[iTile * 6 + 4];
		sum_mat[5] += p_temphost1[iTile * 6 + 5];
		sumvec += p_sum_vec_host[iTile];
	};

	// Now populate symmetric matrix
	f64_tens3 mat2;

	mat.xx = sum_mat[0];
	mat.xy = sum_mat[1];
	mat.xz = sum_mat[2];
	mat.yx = mat.xy;
	mat.yy = sum_mat[3];
	mat.yz = sum_mat[4];
	mat.zx = mat.xz;
	mat.zy = mat.yz;
	mat.zz = sum_mat[5];
	mat.Inverse(mat2);
	//printf(
	//	" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
	//	" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
	//	" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
	//	mat.xx, mat.xy, mat.xz, sumvec.x,
	//	mat.yx, mat.yy, mat.yz, sumvec.y,
	//	mat.zx, mat.zy, mat.zz, sumvec.z);
	f64_vec3 product = mat2*sumvec;

	beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

	printf("beta [x1 = temp6] %1.8E [x2 = move] %1.8E [x3 = Jacobi] %1.8E ", beta[0], beta[1], beta[2]);

	kernelAddRegressors << <numTilesMinor, threadsPerTileMinor >> >(
		p_AzNext,
		beta[0], beta[1], beta[2],
		p_x1,
		p_x2,
		p_Jacobi_x
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelAddRegressors");

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_AzNext);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills AzNext");

	// TESTING:
	if (GlobalSuppressSuccessVerbosity == false) {

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		RSS = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
		L2eps = sqrt(RSS / (f64)NMINOR);
		printf("L2eps: %1.9E \n", L2eps);


		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_AzNext,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAzNext
			//,pX_use->p_AreaMinor // populates it
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

		kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
			(hsub, // ?
				pX_use->p_info,
				p_AzNext, pAz_k,
				p_Azdot0, p_gamma,
				p_LapCoeffself, p_LapAzNext,
				p_epsilon, p_Jacobi_x,
				//this->p_AAdot, // to use Az_dot_k for equation);
				p_bFailed
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");


		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		RSS = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
		L2eps = sqrt(RSS / (f64)NMINOR);
		printf("L2eps: %1.9E \n", L2eps);

	};
	// Totally ineffective -- choosing small coefficients.
	// Can't believe these aren't useful regressors.


}


int RunBackwardForHeat_ConjugateGradient(
	T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use,
	bool bUseMask 
	)
{
#define UPLIFT_THRESHOLD 0.33
#define NO_EQUILIBRATE 
	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	f64 dot2_n, dot2_i, dot2_e;
	bool bProgress;
	f64 old_heatrate, new_heatrate, change_heatrate;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	//GlobalSuppressSuccessVerbosity = true;

	// Be sure to bring equilibrate back.


	bool btemp;
	f64 f64temp;

	//cudaMemcpy(&btemp, &(p_boolarray2[25587 + 2*NUMVERTICES]), sizeof(bool), cudaMemcpyDeviceToHost);
	//printf("25587 : %d \n", btemp ? 1 : 0);

	// Assume p_T contains seed.
#define NO_EQUILIBRATE


	printf("\nConjugate gradient for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 SS_n, SS_i, SS_e, oldSS_n, oldSS_i, oldSS_e, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e, RSS_n, RSS_i, RSS_e;

	// How to determine diagonal coefficient in our equations? In order to equilibrate.

	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_e, 0, sizeof(f64)*NUMVERTICES);

#ifndef NO_EQUILIBRATE
	kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			pX_use->p_AreaMajor,

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e // what exactly it calculates?
			); // used for Jacobi
	Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

	// In all honesty, we should only call this the once, not both here and in JLS repeatedly.

	// D = coeffself
	// We are going to need to make the following changes:
	//  : where sqrt(N)T_k applies, multiply by 1/sqrt(D_i)
	//  : Use sqrt(D_i)sqrt(Ni)T_i as the indt variable
	//     (when we divide by sqrt(D_i)sqrt(N_i) to obtain T,
	//     we can then use that to get the input to epsilon
	//  : But when we calc epsilon, multiply all by 1/sqrt(D_i)

	// So all in all it seems 1/sqrt(D_i) is the factor we would like to save.

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_n, p_sqrtD_inv_n);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf n");

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_i, p_sqrtD_inv_i);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf i");

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_e, p_sqrtD_inv_e);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf e");

	// seed: just set T to T_k.
	kernelUnpacktorootDN_T << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te, p_T,
			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e,
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoDNT");

	// Was all very well but now we wanted it to be root NT times sqrt(coeffself)

	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	// Unimportant that it fills in 0 for the masked values.

	kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te, 
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

#else
	
	kernelUnpacktorootNT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_NnTn, p_NTi, p_NTe, p_T,
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");


	// 1. Compute epsilon:

	// epsilon = T_k+1 - T_k - (h/N)rates.NT;
	// epsilon = b - A T_k+1
	// Their Matrix A = -identity + (h/N) ROC NT due to heat flux
	// Their b = - T_k

	// Compute heat flux given p_T	
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

	// Unimportant that it fills in 0 for the masked values.

	kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe, // divide by root N
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

#endif
	
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,  // using vert indices

			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i, p_kappa_e,
			p_nu_i, p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemcpy(&old_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);

	// Let's say we don't know what is in NT_rates_d_temp outside of mask, but it is unset.


	// Let's think about this carefully. If we could do it with the same data loaded each vertex for neigh list it would actually save time.
	// We rely on the fact that loading the vertex list data is contiguous fetch.
	// And we do store it for 3 species - so let's stick with all in 1.

#ifndef NO_EQUILIBRATE
	kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e, // outputs - ensure we have set 0 in mask 
			p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
			NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
			0,
			p_boolarray2,
			p_boolarray_block,
			bUseMask // NOTE THAT MOSTLY EPSILON ARE UNSET.
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

#else
	cudaMemcpy(&f64temp, &(p_epsilon_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\n\n25587 : p_epsilon_e[25587] %1.13E \n\n", f64temp);

	kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_NnTn, p_NTi, p_NTe,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
			0, // p_bFailed
			p_boolarray2,
			p_boolarray_block,
			bUseMask // calc eps = 0 if mask is on and maskbool = 0
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon"); // sets 0 outside of mask.
	
#endif

	// --==
	// p = eps
	cudaMemcpy(p_regressor_n, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_i, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_e, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);




	//cudaMemcpy(&f64temp, &(p_epsilon_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\n\n25587 : p_epsilon_e[25587] %1.13E \n\n", f64temp);



	// Take RSS :
	kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_temp1, p_temp2, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		SS_n += p_temphost1[iTile];
		SS_i += p_temphost2[iTile];
		SS_e += p_temphost3[iTile];

		//printf("iTile: %d p_temphost %1.10E %1.10E %1.10E \n", iTile, p_temphost1[iTile], p_temphost2[iTile], p_temphost3[iTile]);
		// Let's see if there is a specific tile where we find differences from one time to the next...
	}

	oldSS_n = SS_n; oldSS_i = SS_i; oldSS_e = SS_e;
	long iIteration = 0;

	if (bUseMask) {
		printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
			sqrt(SS_n * over_iEquations_n),
			sqrt(SS_i * over_iEquations_i),
			sqrt(SS_e * over_iEquations_e));
	} else {
		printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
			sqrt(SS_n / (f64)NUMVERTICES),
			sqrt(SS_i / (f64)NUMVERTICES),
			sqrt(SS_e / (f64)NUMVERTICES));
	};

	f64 Store_SS_e = SS_e;
	
	bool bContinue = true;
	do {

		// See if we can get rid of additional ROCWRTregressor routine.
		// Ap = -p + (h / N) flux(p);
		// remember -Ax is what appears in epsilon.

		// normally eps = T_k+1 - T_k - (h/N) flux(T_k+1)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
 
		// regressor represents the addition to NT, we need to pass T = regressor/N

#ifndef NO_EQUILIBRATE

		kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_regressor_n, p_regressor_i, p_regressor_e,
				pX_use->p_AreaMajor,
				pX_use->p_n_major,
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromDNT reg");

#else
		//cudaMemcpy(&f64temp, &(p_regressor_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\n\n25587 : p_regressor_e[25587] %1.13E \n\n", f64temp);

		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_regressor_n, p_regressor_i, p_regressor_e,
				pX_use->p_AreaMajor, // divide by root N!
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT reg");


		// This is 0 outside of mask... because we propose to add 0 outside of mask.

		// debug:
		// now let's copy off this :
		cudaMemcpy(p_temp5, p_Te, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// we'll test what happens to heat rate if we add alpha_e*p_temp5 to T.

		//cudaMemcpy(&f64temp, &(p_Te[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\n\n25587 : p_Te (reg) [25587] %1.13E \n\n", f64temp);
		

#endif
		// Note that we passed 0 in the masked cells so we are expecting this to fail when we look at them?
		// Well, those T are not going to change.

		printf("With values from regressors:\n");


		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES); // redundant
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,				 // values from regressors
				pX_use->p_B + BEGINNING_OF_CENTRAL, 
				p_kappa_n,
				p_kappa_i, p_kappa_e,
				p_nu_i, p_nu_e,
				NT_addition_rates_d_temp,          // this is zero in mask
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate 1");

#ifndef NO_EQUILIBRATE
		cudaMemset(zero_array, 0, sizeof(T3)*NUMVERTICES);
		kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_Ap_n, p_Ap_i, p_Ap_e, // the result
				p_regressor_n, p_regressor_i, p_regressor_e,
				zero_array, // 0 which we multiply by N...
				pX_use->p_AreaMajor,
				pX_use->p_n_major, // we load this only in order to multiply with 0
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
				NT_addition_rates_d_temp,
				0,
				p_boolarray2,
				p_boolarray_block,
				bUseMask); // for masked cells, everything = 0
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonHeat_Equilibrated");
#else
		cudaMemset(zero_array, 0, sizeof(T3)*NUMVERTICES);
		kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_Ap_n, p_Ap_i, p_Ap_e, // the result
				p_regressor_n, p_regressor_i, p_regressor_e,
				zero_array, // 0 which we multiply by N...
				pX_use->p_AreaMajor,
				pX_use->p_n_major, // we load this only in order to multiply with 0
				NT_addition_rates_d_temp,
				0,
				p_boolarray2,
				p_boolarray_block,
				bUseMask); // for masked cells, everything = 0
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonHeat");
#endif

		cudaMemcpy(&change_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		
		// I believe that we should be taking - [eps dot deps/dbeta] / [deps/dbeta dot deps/dbeta]
		/*
		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon_n, p_Ap_n, // it's -A really
				p_epsilon_i, p_Ap_i,
				p_epsilon_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}

		kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Ap_n, p_Ap_i, p_Ap_e, p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		dot2_n = 0.0; dot2_i = 0.0; dot2_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot2_n += p_temphost1[iTile];
			dot2_i += p_temphost2[iTile];
			dot2_e += p_temphost3[iTile];
		}

		alpha_n = -dot_n / dot2_n;
		alpha_i = -dot_i / dot2_i;
		alpha_e = -dot_e / dot2_e; // ?
		*/

		// Are  we missing a minus??? Yes .. true? epsilon = b-A NT and we just calculated epsilon from NT, b=0
		// ====================================================================================================
		
		NegateVectors << <numTilesMajorClever, threadsPerTileMajorClever >> > 
			(p_Ap_n, p_Ap_i, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS NegateVectors");

		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(	p_regressor_n, p_Ap_n,
				p_regressor_i, p_Ap_i,
				p_regressor_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3	);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");

		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}

		alpha_n = (dot_n != 0.0) ? (SS_n / dot_n) : 0.0;
		alpha_i = (dot_i != 0.0) ? (SS_i / dot_i) : 0.0;
		alpha_e = (dot_e != 0.0) ? (SS_e / dot_e) : 0.0;
				
		printf("alpha %1.8E %1.8E %1.8E SS %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E\n", alpha_n, alpha_i, alpha_e,
			SS_n, SS_i, SS_e, dot_n, dot_i, dot_e); 
				
		/*
		NegateVectors << <numTilesMajorClever, threadsPerTileMajorClever >> >(p_Ap_n, p_Ap_i, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS NegateVectors");

		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_epsilon_n,
				p_regressor_i, p_epsilon_i,
				p_regressor_e, p_epsilon_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}
		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_Ap_n,
				p_regressor_i, p_Ap_i,
				p_regressor_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot2_n = 0.0; dot2_i = 0.0; dot2_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot2_n += p_temphost1[iTile];
			dot2_i += p_temphost2[iTile];
			dot2_e += p_temphost3[iTile];
		}

		alpha_n = dot_n / dot2_n;
		alpha_i = dot_i / dot2_i;
		alpha_e = dot_e / dot2_e; // dot is so large that alpha is very small.

		printf("alpha %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E\n", alpha_n, alpha_i, alpha_e,
			dot_n, dot_i, dot_e, dot2_n, dot2_i, dot2_e);
		*/


		// DEBUG:
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
				pX_use->p_AreaMajor,
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");
		
		VectorAddMultiple1 << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_Te, alpha_e, p_temp5);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiple ");
	
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i, p_kappa_e,
				p_nu_i, p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//	cudaMemcpy(&new_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);

		// Compare:
	//	printf("Test adding to T: heatrate difference %1.14E - %1.14E = %1.14E\n",
	//		new_heatrate, old_heatrate, new_heatrate - old_heatrate);
	//	printf("Multiply: alpha_e %1.14E * changerate %1.14E = %1.14E\n",
	//		alpha_e, change_heatrate, alpha_e*change_heatrate);
		// Will be interesting.

		// Now overwrite T:

		// end of debug section
		 
		VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_NnTn, alpha_n, p_regressor_n,
			p_NTi, alpha_i, p_regressor_i,
			p_NTe, alpha_e, p_regressor_e);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiples");
		
		// Are we going to do this without using mask? 
		// What is in p_NTi in mask? Correct values hopefully.

#ifndef NO_EQUILIBRATE

		kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
				pX_use->p_AreaMajor,
				pX_use->p_n_major,
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
				); 
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromsqrtDN_T");
#else
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
				pX_use->p_AreaMajor,
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");
#endif
		// Update Epsilon: or can simply recalculate eps = b - A newx
		// Sometimes would rather update epsilon completely:
		if (iIteration % 1 == 0) {

			cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
			kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(
					pX_use->p_info,
					pX_use->p_izNeigh_vert,
					pX_use->p_szPBCneigh_vert,
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_cc,
					pX_use->p_n_major,
					p_Tn, p_Ti, p_Te,
					pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
					p_kappa_n,
					p_kappa_i, p_kappa_e,
					p_nu_i, p_nu_e,
					NT_addition_rates_d_temp,
					pX_use->p_AreaMajor,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//		cudaMemcpy(&new_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	//		
	//		f64 diff_heatrate = new_heatrate - old_heatrate;
	//		f64 predict_heatrate = alpha_e*change_heatrate;
			
	//		printf("VERTCHOSEN %d \n", VERTCHOSEN);
	//		printf("new %1.13E - old %1.13E = %1.14E \n", new_heatrate, old_heatrate, diff_heatrate);
	//		printf("alpha_e %1.14E * changerate %1.14E = predict %1.14E\n", alpha_e, change_heatrate, predict_heatrate);

	//		old_heatrate = new_heatrate;

#ifndef NO_EQUILIBRATE
			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon_n, p_epsilon_i, p_epsilon_e,
					p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
					NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
					p_bFailed,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");
#else
			
			// DEBUG:
			cudaMemcpy(p_temp1, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_temp2, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_temp3, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp1, -alpha_n, p_Ap_n,
				p_temp2, -alpha_i, p_Ap_i,
				p_temp3, -alpha_e, p_Ap_e
				);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiples 2");

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon_n, p_epsilon_i, p_epsilon_e,
					p_NnTn, p_NTi, p_NTe,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
					p_bFailed,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

			VectorCompareMax << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp2, 
				p_epsilon_i,
				p_longtemp, p_temp4 
				);
			Call(cudaThreadSynchronize(), "cudaTS CompareMax");
			cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 maxo = 0.0;
			long iWhich;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				if (p_temphost4[iTile] > maxo) {
					maxo = p_temphost4[iTile];
					iWhich = iTile;
				}
			};
			long iMaxVert = p_longtemphost[iWhich];
			printf(" ion  iMaxVert %d max %1.10E \n", iMaxVert, maxo);
		
			VectorCompareMax << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp3,
				p_epsilon_e,
				p_longtemp, p_temp4
				);
			Call(cudaThreadSynchronize(), "cudaTS CompareMax");
			cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			maxo = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				if (p_temphost4[iTile] > maxo) {
					maxo = p_temphost4[iTile];
					iWhich = iTile;
				}
			};
			iMaxVert = p_longtemphost[iWhich];
			printf(" elec iMaxVert %d max %1.10E \n", iMaxVert, maxo);
			
#endif
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);

			//// DEBUG:
			//cudaMemcpy(p_temphost4, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost5, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost6, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			//FILE * fp = fopen("debug0.txt", "a");
			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//	fprintf(fp, "iVertex %d epsilon_old %1.13E %1.13E %1.13E epsilon %1.13E %1.13E %1.13E \n",
			//		iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], 
			//		p_temphost4[iVertex], p_temphost5[iVertex], p_temphost6[iVertex]);
			//
			//cudaMemcpy(p_temphost1, p_Ap_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost2, p_Ap_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost3, p_Ap_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			//cudaMemcpy(p_NTrates_host, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(cuSyst_host.p_n_major, pX_use->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(cuSyst_host.p_AreaMajor, pX_use->p_AreaMajor, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//

			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//{
			//	fprintf(fp, "iVertex %d alpha %1.13E %1.13E %1.13E p_Ap %1.13E %1.13E %1.13E Nn %1.13E N %1.13E ddtNnTn %1.13E ddtNiTi %1.14E ddtNeTe %1.14E \n",
			//		iVertex, alpha_n, alpha_i, alpha_e,
			//		p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex],
			//		cuSyst_host.p_AreaMajor[iVertex]*cuSyst_host.p_n_major[iVertex].n_n, cuSyst_host.p_AreaMajor[iVertex] * cuSyst_host.p_n_major[iVertex].n,
			//		p_NTrates_host[iVertex].NnTn, p_NTrates_host[iVertex].NiTi, p_NTrates_host[iVertex].NeTe);
			//}
			//fclose(fp);
			//printf("file saved (append)\n");

		} else {
			bContinue = true;
			VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_epsilon_n, -alpha_n, p_Ap_n,
				p_epsilon_i, -alpha_i, p_Ap_i,
				p_epsilon_e, -alpha_e, p_Ap_e
				);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiples 2");
			p_boolhost[0] = true;
			// addition should be 0 in mask.

			// am I right that this gives different answer? we seem to drop erratically?


		};
		
		kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon_n, p_epsilon_i, p_epsilon_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		bContinue = false;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			SS_n += p_temphost1[iTile];
			SS_i += p_temphost2[iTile];
			SS_e += p_temphost3[iTile];
			if (p_boolhost[iTile] == true) bContinue = true;
		}
		
		ratio_n = (oldSS_n > 0.0) ? (SS_n / oldSS_n) : 0.0;
		ratio_i = (oldSS_i > 0.0) ? (SS_i / oldSS_i) : 0.0;
		ratio_e = (oldSS_e > 0.0) ? (SS_e / oldSS_e) : 0.0;

		// get it going first and then profile to see if we want to do more masking.

		if (bUseMask) {
			printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
				sqrt(SS_n * over_iEquations_n),
				sqrt(SS_i * over_iEquations_i),
				sqrt(SS_e * over_iEquations_e));
		}
		else {
			printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
				sqrt(SS_n / (f64)NUMVERTICES),
				sqrt(SS_i / (f64)NUMVERTICES),
				sqrt(SS_e / (f64)NUMVERTICES));
		};

		//printf("ratio %1.10E %1.10E %1.10E SS %1.10E %1.10E %1.10E\n", ratio_n, ratio_i, ratio_e,
		//	SS_n,SS_i,SS_e);

		kernelRegressorUpdate << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_regressor_i, p_regressor_e,
				p_epsilon_n, p_epsilon_i, p_epsilon_e,
				ratio_n, ratio_i, ratio_e,

				p_boolarray_block,
				bUseMask
				);
		Call(cudaThreadSynchronize(), "cudaTS RegressorUpdate");
		// regressor = epsilon + ratio*regressor; 

		bProgress = false;
		if ((oldSS_n > 0.0) && (sqrt(ratio_n) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;
		if ((oldSS_i > 0.0) && (sqrt(ratio_i) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;
		if ((oldSS_e > 0.0) && (sqrt(ratio_e) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;

		oldSS_n = SS_n;
		oldSS_i = SS_i;
		oldSS_e = SS_e;

		// Now calculate epsilon in original equations
	/*	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		if (iIteration % 4 == 0) {
			kernelCreateEpsilonHeatOriginalScaling << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_temp4, p_temp5, p_temp6,
					p_Tn, p_Ti, p_Te,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					NT_addition_rates_d_temp, // we were assuming this was populated for T but it sometimes wasn't.
					p_bFailed);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonOriginal");
			kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_temp4, p_temp5, p_temp6,
					p_temp1, p_temp2, p_temp3);
			Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
			RSS_n = 0.0; RSS_i = 0.0; RSS_e = 0.0;
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
			bContinue = false;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				RSS_n += p_temphost1[iTile];
				RSS_i += p_temphost2[iTile];
				RSS_e += p_temphost3[iTile];
				if (p_boolhost[iTile] == true) bContinue = true;
			} 
		} else {
			// NT addition rates was not populated -- skip this
			bContinue = true;
		}
		printf("original eqns: L2eps %1.12E %1.12E %1.12E \n", sqrt(RSS_n / (f64)NUMVERTICES),
			sqrt(RSS_i / (f64)NUMVERTICES), sqrt(RSS_e / (f64)NUMVERTICES));
			*/

		iIteration++;

		if (bContinue == false) printf("all tests ok\n");

		// Seems to present correct result yet gives incorrect figures for L2eps in between - I have no idea why this is.

		// set bContinue according to all species converging
	} while ((bContinue) &&
		((iIteration < ITERATIONS_BEFORE_SWITCH) || (bProgress))
		);
	
	GlobalSuppressSuccessVerbosity = true;

		//((sqrt(RSS_i / (f64)NUMVERTICES) > 1.0e-28) ||
		//(sqrt(RSS_e / (f64)NUMVERTICES) > 1.0e-28) ||
			//(sqrt(RSS_n / (f64)NUMVERTICES) > 1.0e-28)));
	
	// It is looking a lot like we could save a lot if we had
	// actually split it out by species. Clearly that didn't
	// occur to me at the time I did the routines but it is now obvious.
	// There should be 1 routine, called for each species. 
	// We are just reusing position - not worth it I think.

	// OKay how much of it is now proposing to use split out arrays? This is when we start doing this.
	// We can zip them back up into T3 struct afterwards.

	// seed: just set T to T_k.
	
	if ((bContinue == true) && (Store_SS_e < SS_e)) {
			printf("It got worse!\n");
	} else {
		kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated sqrt(DN)T.
		Call(cudaThreadSynchronize(), "cudaTS PackupT3");
	};

	//Was working before. But it's SO temperamental. Now it makes things worse, masked or unmasked.
		
	if (bContinue == true) return 1;
	return 0;
}

#define REGRESSORS 8

__device__ f64 * regressors;
__device__ f64 * p_coeffself;


int RunGeometricHeatConduction_NonlinearCG__singleregression(f64 * p_T_k, f64 * p_T, 
	f64 hsub, cuSyst * pX_use, bool bUseMask, f64 * p_kappa, f64 * p_nu,
	int iSpecies) 
{

	// This is an alternative method to the R8LS.
	// It seemed to run a bit slower.
	// Surprisingly enough.
	// Need to write about it.


	f64 L2eps, L2reg;
	bool bFailedTest, bContinue;
	Triangle * pTri;
	f64 tempf64;
	long iTile, i;

	int iIteration = 0;
	f64 beta_PR;
	f64 tempdebug;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
	// We certainly should run with previous move as regressor and do a line search from that before we do anything.
#define zerovec1 p_Residuals

	printf("RunGeometricHeatConduction_NonlinearCG:\n");

	CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES));
	
	// call that first and send species kappa.

	GlobalSuppressSuccessVerbosity = true;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	if (iSpecies == 0) SetConsoleTextAttribute(hConsole, 14);
	if (iSpecies == 1) SetConsoleTextAttribute(hConsole, 10);
	if (iSpecies == 2) SetConsoleTextAttribute(hConsole, 15); // 12= red 11=cyan




	// Don't forget to actually augment d/dt NT when we are done here.







	bool bContinue1 = true;
	do {
		// There should be a 0th step where we use previous moves.
		// Ideally store 2 moves, do multivariate regression to get direction;
		// line search for that direction.
		
		// Get epsilon for use in SD calc:

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
			
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,

			// scrap masking for now --- but bring it back intelligently???			
			p_boolarray2 + NUMVERTICES*iSpecies,
			p_boolarray_block,
			false, //bool bUseMask,
			
			// Just hope that our clever version will converge fast.
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
		kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h/N sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_T,
			p_T_k,
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn
			p_epsilon,
			p_bFailed,
			p_boolarray2 + NUMVERTICES*iSpecies,
			p_boolarray_block,
			bUseMask,
			iSpecies
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		if (bUseMask == 0) {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		} else {
			f64 over = over_iEquations_n;
			if (iSpecies == 1) over = over_iEquations_i;
			if (iSpecies == 2) over = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over);
		}
		printf(" L2eps %1.11E  :\n ", L2eps);
		f64 RSS1 = sum_eps_eps;


		if (L2eps < 1.0e-19) bContinue1 = false; // FOR NOW
		
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		//  1. Calculate steepest descent direction.
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#define p_large_array_maxneigh_by_numvertices  d_eps_by_dx_neigh_n
		kernelAccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T,
			p_epsilon,
			hsub,
			// we need to multiply by h/N in the target to get deps/dbeta at the vertex

			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
			pX_use->p_AreaMajor,	

			// scrap masking for now --- but bring it back intelligently??? ???
			p_boolarray2 + NUMVERTICES*iSpecies,
			p_boolarray_block,
			false, //bool bUseMask,
			iSpecies,
			p_large_array_maxneigh_by_numvertices,
			p_temp1 // effect self
		); 
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric");

		AddFromMyNeighbours << < numTilesMajor, threadsPerTileMajor >> >( // Also add + 1.0 on self, so + epsilon in other words.
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_large_array_maxneigh_by_numvertices, // needs to be MAXNEIGH*NUMVERTICES
			p_temp1, // effect self
			p_regressors,
			pX_use->p_izNeigh_vert,
			sz_who_vert_vert
		);
		Call(cudaThreadSynchronize(), "cudaTS AddFromMyNeighbours");
		
		// Let's do an empirical test and see whether this "p_regressors" contains predictive information.
	/*	f64 tempf64, dRSS_i, dRSS_ii;
		cudaMemcpy(&dRSS_i, &(p_regressors[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&dRSS_ii, &(p_regressors[VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
		f64 predicted = 1.0e-15*(dRSS_i + dRSS_ii);

		// Add 1e-14 to 2 vertices:
	//	cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);		
	//	tempf64 += 1.0e-14;
	//	cudaMemcpy(&(p_T[VERTCHOSEN]), &tempf64, sizeof(f64), cudaMemcpyHostToDevice);
		
		cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("T = %1.14E beforehand ; ", tempf64);
		tempf64 += 1.0e-15;
		printf("new T %1.14E \n\n", tempf64);
		cudaMemcpy(&(p_T[VERTCHOSEN2]), &tempf64, sizeof(f64), cudaMemcpyHostToDevice);
		

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T, pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
			NT_addition_rates_d_temp, pX_use->p_AreaMajor,		
			p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
		kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h/N sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,	pX_use->p_info + BEGINNING_OF_CENTRAL,	p_T, p_T_k,
			NT_addition_rates_d_temp,			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn
			p_epsilon,			p_bFailed,
			p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block,	bUseMask,
			iSpecies		);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];

		f64 RSS2 = sum_eps_eps;
		printf("RSS2-RSS1 %1.14E predicted %1.14E RSS1 %1.14E RSS2 %1.14E \n", RSS2 - RSS1, predicted, RSS1, RSS2);

		cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("T = %1.14E beforehand ; ", tempf64);
		tempf64 += 0.9e-14;
		printf("new T %1.14E \n\n", tempf64);
		cudaMemcpy(&(p_T[VERTCHOSEN2]), &tempf64, sizeof(f64), cudaMemcpyHostToDevice);


		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T, pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
			NT_addition_rates_d_temp, pX_use->p_AreaMajor,
			p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
		kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h/N sum kappa dot grad T)
			// x = -eps/coeffself
			hsub, pX_use->p_info + BEGINNING_OF_CENTRAL, p_T, p_T_k,
			NT_addition_rates_d_temp,			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn
			p_epsilon, p_bFailed,
			p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, bUseMask,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		
		predicted = 1.0e-14*(dRSS_i + dRSS_ii);
		RSS2 = sum_eps_eps;
		printf("RSS2-RSS1 %1.14E predicted[1e-15] %1.14E RSS1 %1.14E RSS2 %1.14E\n", RSS2 - RSS1, predicted, RSS1, RSS2);

		cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("T = %1.14E beforehand ; ", tempf64);
		tempf64 -= 1.0e-14;
		printf("new T %1.14E \n\n", tempf64);
		cudaMemcpy(&(p_T[VERTCHOSEN2]), &tempf64, sizeof(f64), cudaMemcpyHostToDevice);

		getch();
		getch();
		*/
		// 


		cudaMemcpy(p_regressors + NUMVERTICES*2, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// hold on to Steepest Descent!

		if (iIteration > 0) {
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			// 2. Calculate "beta_PR" and subtract previous regr if beta_PR > 0
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			
			kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS AccSum");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 sn_SS = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				sn_SS += p_sum_eps_eps_host[iTile];
						
			kernelAccumulateDotProduct << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors, p_regressors + NUMVERTICES, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS DotProduct");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 sn_dot_sprev = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				sn_dot_sprev += p_sum_eps_eps_host[iTile];

			kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors + NUMVERTICES, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS AccSum");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 sprev_SS = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				sprev_SS += p_sum_eps_eps_host[iTile];
			
			beta_PR = (sn_SS - sn_dot_sprev)/sprev_SS;

			f64 beta_FR = sn_SS / sprev_SS;
		//	printf("sn_SS %1.9E sn.sprev %1.9E diff %1.9E sprev_SS %1.9E beta_PR %1.9E beta_FR %1.9E\n",
		//		sn_SS, sn_dot_sprev, sn_SS-sn_dot_sprev, sprev_SS, beta_PR, beta_FR);
			
			// Correlations:
			// steepest descent vs previous regressor :
			
			kernelAccumulateDotProduct << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors, p_regressors + 3*NUMVERTICES, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS DotProduct");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 xn_dot_xprev = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				xn_dot_xprev += p_sum_eps_eps_host[iTile];

			kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors + 3*NUMVERTICES, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS AccSum");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 xprev_SS = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				xprev_SS += p_sum_eps_eps_host[iTile];

			f64 correl1 = xn_dot_xprev / sqrt(xprev_SS*sn_SS);
			printf("correl between SD and previous regressor: %1.10E\n", correl1);

			// steepest descent vs previous steepest descent:
			f64 correl2 = sn_dot_sprev / sqrt(sprev_SS*sn_SS);
			printf("correl between SD and previous SD: %1.10E\n", correl2);
						
			beta_PR = max(0.0, beta_PR);
			// CHANGED:
			if ((beta_PR > 0.0)) // && (iIteration % 12 != 0)) 
			{
				VectorAddMultiple1 << < numTilesMajorClever, threadsPerTileMajorClever >> >(
					p_regressors, -beta_PR, p_regressors + NUMVERTICES	);
				Call(cudaThreadSynchronize(), "cudaTS VectorAddMultiple1");
			};

			kernelAccumulateDotProduct << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors, p_regressors + 3 * NUMVERTICES, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS DotProduct");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			xn_dot_xprev = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				xn_dot_xprev += p_sum_eps_eps_host[iTile];
			kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_regressors, p_temp2);
			Call(cudaThreadSynchronize(), "cudaTS AccSum");
			cudaMemcpy(p_sum_eps_eps_host, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 xn_SS = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				xn_SS += p_sum_eps_eps_host[iTile];
			f64 correl3 = xn_dot_xprev / sqrt(xprev_SS*xn_SS);
			printf("correl between regressors: %1.10E\n", correl3);
		};
		// save steepest descent:
		cudaMemcpy(p_regressors + NUMVERTICES, p_regressors + 2*NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		// save regressor:
		cudaMemcpy(p_regressors + NUMVERTICES * 3, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		//  3. Line search for how much regressor to add: 
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		// . Do linear regression estimate for first proposal beta_1:
		kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T,
			hsub,
			p_regressors,
			pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
			p_d_eps_by_d_beta_x_, // d/dbeta of d(NT)/dt in this cell
			pX_use->p_AreaMajor,
			// scrap masking for now --- but bring it back intelligently??? ???
			p_boolarray2 + NUMVERTICES*iSpecies,
			p_boolarray_block,
			false, //bool bUseMask,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRateROC_wrt_regressor_1s");

		//// make ascii file:
		//char buff[255];
		//sprintf(buff, "Iter%d.txt", iIteration);
		//FILE * fp = fopen(buff, "w");
		//cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost2, p_d_eps_by_d_beta_x_, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost3, p_T, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//for (i = 0; i < NUMVERTICES; i++)
		//	fprintf(fp, "%d epsilon %1.12E dbydbeta %1.12E T %1.12E \n",
		//		i, p_temphost1[i], p_temphost2[i], p_temphost3[i]);
		//fclose(fp);

		// collect eps against d_eps and d_eps against itself:
		// This ought to work even though it's set up for minor cells:
		kernelAccumulateSummands2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon,
			p_d_eps_by_d_beta_x_, // will this fail?
			p_temp1, //p_sum_eps_d,
			p_temp2, //p_sum_d_d,
			p_temp3 // p_sum_eps_eps
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummands2");
		
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			
		f64 sum_eps_d = 0.0, sum_d_d = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++) {
			sum_eps_d += p_temphost1[iTile];
			sum_d_d += p_temphost2[iTile];
			sum_eps_eps += p_temphost3[iTile];
		//	printf("iTile %d %1.9E %1.9E %1.9E ", iTile, p_temphost1[iTile], p_temphost2[iTile], p_temphost3[iTile]);		
		};
		
		f64 beta_1 = -sum_eps_d / sum_d_d; // WHY NOT MINUS?		
		// strangely makes almost no difference + or -. Coeff large?

		f64 RSS0 = sum_eps_eps;
		f64 sum_eps_d_0 = sum_eps_d; 

		printf("eps_d %1.12E _d_d %1.12E sum_epssq %1.12E\n",
			sum_eps_d, sum_d_d, sum_eps_eps);

		if (sum_eps_d > 0.0) {
			printf("alert -- sum_eps_d > 0.0\n");
			getch();
		};
		
		printf("beta_1 %1.12E \n", beta_1);
		
		// Now make the move and make another assessment of derivative:

		// Bear in mind we did not find derivative of RSS -- that is sum [2 eps deps]
		// That is something however that we are trying to minimize.

#define p_Tcurr p_regressor_i
#define p_T0    p_regressor_e

		// STORE T:
		cudaMemcpy(p_T0, p_T, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
		f64 beta_prev = 0.0;
		f64 sum_eps_eps_prev = sum_eps_eps;
		f64 sum_eps_d_prev = sum_eps_d;   // what we already calc'd
		f64 beta_curr = beta_1;
		f64 sum_eps_d_curr, sum_d_d_curr, sum_eps_eps_curr;

		if (iSpecies > 0) {
			// For neutrals we do not need to do line search.

			printf("\n");

			bool bContinue = true;
			bool b_vanWijngaarden_Dekker_Brent = false;
			do {
				cudaMemcpy(&tempf64, &(p_regressors[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
				printf("beforehand p_regressors[%d] = %1.12E \n", VERTCHOSEN, tempf64);
				cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
				printf("beforehand p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

				AddLC << < numTilesMajorClever, threadsPerTileMajorClever >> >
					(p_Tcurr, p_T0, beta_curr, p_regressors); // Tcurr = T0 + beta_curr*regressors
				Call(cudaThreadSynchronize(), "cudaTS AddLC");
				
				printf("Added beta_curr = %1.9E times regressor to produce p_Tcurr \n", beta_curr);
				
				cudaMemcpy(&tempf64, &(p_Tcurr[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
				printf("now p_Tcurr[%d] = %1.12E \n", VERTCHOSEN, tempf64);

				// EvaluateRSSandDeriv(p_Tcurr) :
				// call to get heatflux and hence epsilon:

				cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
				kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
					pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
					p_Tcurr,
					pX_use->p_B + BEGINNING_OF_CENTRAL,
					p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
					NT_addition_rates_d_temp, pX_use->p_AreaMajor,
					// scrap masking for now --- but bring it back intelligently???			
					p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false, //bool bUseMask,
					iSpecies);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

				CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
				kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					// eps = T - (T_k +- h sum kappa dot grad T)
					hsub, pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_Tcurr, p_T_k,
					NT_addition_rates_d_temp,			// NEED N = n*AreaMajor
					pX_use->p_n_major, pX_use->p_AreaMajor,
					p_epsilon, p_bFailed,
					p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, bUseMask,
					iSpecies);
				Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

				//kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				//	(p_epsilon, p_sum_eps_eps);
				//Call(cudaThreadSynchronize(), "cudaTS AccSum");
				//cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				//f64 sum_eps_eps = 0.0;
				//for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				//	sum_eps_eps += p_sum_eps_eps_host[iTile];
				//RSScurr = sum_eps_eps;
				// Let's see whether sum_d_d is now small and we can stop.

				// Get dRSS/dbeta:

				kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
					pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
					p_Tcurr,
					hsub,
					p_regressors,
					pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
					p_d_eps_by_d_beta_x_, // d/dbeta of d(NT)/dt in this cell
					pX_use->p_AreaMajor,
					p_boolarray2 + NUMVERTICES*iSpecies, // scrap masking for now --- but bring it back intelligently??? ???
					p_boolarray_block,
					false, //bool bUseMask,
					iSpecies);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRateROC_wrt_regressor_1s");

				// This ought to work even though it's set up for minor cells:
				kernelAccumulateSummands2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon,
					p_d_eps_by_d_beta_x_, // will this fail?
					p_temp1, //p_sum_eps_d,
					p_temp2, //p_sum_d_d,
					p_temp3 // p_sum_eps_eps -- overwrites it
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummands2");
				cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				sum_eps_d_curr = 0.0; 
				sum_d_d_curr = 0.0;
				sum_eps_eps_curr = 0.0;
				for (iTile = 0; iTile < numTilesMajorClever; iTile++) {
					sum_eps_d_curr += p_temphost1[iTile];
					sum_d_d_curr += p_temphost2[iTile];
					sum_eps_eps_curr += p_temphost3[iTile];
				};

				printf("sum_eps_eps_curr %1.11E \n", sum_eps_eps_curr);

				// note: dRSS/dbeta = 2 sum eps*deps/dbeta
				if ((fabs(sum_eps_d_curr) < 0.0005*fabs(sum_eps_d_prev))
					|| ((fabs(beta_curr-beta_prev) < 0.001*fabs(beta_curr))
						&& (sum_eps_eps_curr < sum_eps_eps_prev)) )
				{					
					// break out:
					bContinue = false;

					printf("Set bContinue = false. |DSS2| %1.9E |DSS1| %1.9E fabs(b2-b1) %1.9E fabs(b1) %1.9E\n",
						fabs(sum_eps_d_curr), fabs(sum_eps_d_prev), fabs(beta_curr - beta_prev), fabs(beta_curr));

				} else {
						
					// Now there are a number of cases.
					// 1. Has dRSS/dbeta changed sign?
					if (sum_eps_d_curr > 0.0) { // prev should be < 0

						// root is bracketed so drop out to vWDB routine.
						b_vanWijngaarden_Dekker_Brent = true;

						printf("Pass now to van Wijngaarden-Dekker-Brent.\n");

					} else {

						if (sum_eps_eps_curr > sum_eps_eps_prev)
						{
							// RSS went up
							// There is a root in-between;
							// Simplest response is bisection; but bisecting here we could get caught in a loop with linear extrapolation and bisecting back.

							// Fit a quadratic to find where dRSS/dbeta went through 0.
							f64 f1 = 2.0*sum_eps_d_curr;
							f64 f0 = 2.0*sum_eps_d_prev;
							f64 overdelta = 1.0 / (0.5*(beta_curr - beta_prev));
							printf("f0 %1.14E f1 %1.14E beta_curr %1.14E beta_prev %1.8E RSS2 %1.14E RSS1 %1.14E\n",
								f0, f1, beta_curr, beta_prev, sum_eps_eps_curr, sum_eps_eps_prev);

							// In a nasty enough case we could still get stuck in a loop.
							// Maybe we should just ignore this possibility and concentrate on WDB for f'.
							// Too much code overall.
							
							f64 b = (f1 - f0) / (beta_curr - beta_prev);
							f64 over_a = 1.0 / (-0.75*((sum_eps_eps_curr - sum_eps_eps_prev)*overdelta - (f1 + f0))*overdelta*overdelta);
							f64 c = 0.75*(sum_eps_eps_curr - sum_eps_eps_prev)*overdelta - 0.25*(f0 + f1);
							f64 tempsqrt = sqrt(-c*over_a + 0.25*b*b*over_a*over_a);

							printf("b %1.14E over_a %1.14E c %1.14E tempsqrt %1.14E \n", b, over_a, c, tempsqrt);

							f64 beta_minus = 0.5*(beta_prev + beta_curr) - tempsqrt - 0.5*(b*over_a);
							f64 beta_plus = 0.5*(beta_prev + beta_curr) + tempsqrt - 0.5*(b*over_a);
							if ((beta_minus > beta_prev) && (beta_minus < beta_curr))
							{
								beta_curr = beta_minus;

								printf("Quadratic succeeded -- beta_minus = %1.9E \n", beta_minus);

							} else {
								if ((beta_plus > beta_prev) && (beta_plus < beta_curr))
								{
									beta_curr = beta_plus; // probably never happens

									printf("Quadratic succeeded -- beta_plus = %1.9E \n", beta_plus);
								} else {
									printf("Quadratic failed! beta_prev %1.9E minus %1.9E plus %1.9E curr %1.9E\n", 
										beta_prev, beta_minus, beta_plus, beta_curr);
									beta_curr = (beta_curr + beta_prev)*0.5;

									// Why & how did this fail?




								};
							};
						} else {

							// RSS has decreased, but did something weird happen?

							f64 beta_additional = -sum_eps_d_curr / sum_d_d_curr;
							f64 empiricalderiv = (sum_eps_eps_curr - sum_eps_eps_prev) / (beta_curr - beta_prev);

							printf("RSS decreased. beta_additional %1.11E empiricalderiv %1.11E \n", beta_additional, empiricalderiv);

							if ((sum_eps_d_curr > sum_eps_d_prev) ||
								(empiricalderiv > 2.0*sum_eps_d_prev) ||
								(empiricalderiv < 2.0*sum_eps_d_curr))
							{
								// Go again from curr.

								beta_prev = beta_curr;
								beta_curr = beta_curr + beta_additional;

								printf("Weirdness occurred, so use beta_curr to go again: add beta_additional. \n");

							} else {
								// Let's go extra bit:
								beta_prev = beta_curr;
								beta_curr += beta_additional*(beta_curr + beta_additional - 0.0) / (beta_1);
								// trying to get past the root by multiplying addition by amount that first linear estimate was apparently wrong.

								printf("expanded beta_additional to the tune of %1.11E \n",
									(beta_curr + beta_additional - 0.0) / (beta_1)	);
							};
						};
					};
				};
			} while ((bContinue == true) && (b_vanWijngaarden_Dekker_Brent == false));
			
			if (b_vanWijngaarden_Dekker_Brent) {

				bContinue = true;
				// The bracket ends are beta_prev and beta_curr,
				// and the functions have been calculated.

				int iter;
				f64 a = beta_curr, b = beta_prev, c = beta_prev, d, e, min1, min2;
				f64 fa = sum_eps_d_curr, fb = sum_eps_d_prev, fc, p, q, r, s, tol1, xm;
				fc = fb;
				while (bContinue) {
					
					// Code from Press Teukolsky et al:
				
					if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
						c = a; //Rename a, b, c and adjust bounding interval d
						fc = fa; 
						e = d = b - a;
					};
					if (fabs(fc) < fabs(fb)) {
						a = b;
						b = c;
						c = a;
						fa = fb;
						fb = fc;
						fc = fa;
					};
#define EPS  1.0e-14  // double-precision tolerance

					tol1 = 2.0*EPS*fabs(b) + 0.5*0.0008*fabs(sum_eps_d_0); // Convergence check.
					xm = 0.5*(c - b);
					if ((fabs(fb) < 0.0008*fabs(sum_eps_d_0))
						|| (fabs(b - c) < 0.001*fabs(b))) 						
						//if (fabs(xm) <= tol1 || fb == 0.0)
					{
						bContinue = false;
					}
					else {
						if ((fabs(e) >= tol1) && (fabs(fa) > fabs(fb))) {
							s = fb / fa; // Attempt inverse quadratic interpolation.
							if (a == c) {
								p = 2.0*xm*s;
								q = 1.0 - s;
							}
							else {
								q = fa / fc;
								r = fb / fc;
								p = s*(2.0*xm*q*(q - r) - (b - a)*(r - 1.0));
								q = (q - 1.0)*(r - 1.0)*(s - 1.0);
							}
							if (p > 0.0) q = -q; //Check whether in bounds.
							p = fabs(p);
							min1 = 3.0*xm*q - fabs(tol1*q);
							min2 = fabs(e*q);
							if (2.0*p < (min1 < min2 ? min1 : min2)) {
								e = d;// Accept interpolation.
								d = p / q;
							}
							else {
								d = xm; // Interpolation failed, use bisection.
								e = d;
							};
						}
						else {
							//Bounds decreasing too slowly, use bisection.
							d = xm; // distance
							e = d;
						};
						a = b;// Move last best guess to a.
						fa = fb;
						if (fabs(d) > tol1) //Evaluate new trial root.
							b += d;
						else {
							if (xm > 0.0) {
								b += tol1;
							} else {
								b -= tol1;
							};
							// b += SIGN(tol1, xm); // I think, tol1 with the sign of xm
						};
						// Here we must compute RSS' at beta = b.
						AddLC << < numTilesMajorClever, threadsPerTileMajorClever >> >
							(p_Tcurr, p_T0, b, p_regressors); // Tcurr = T0 + beta_curr*regressors
						Call(cudaThreadSynchronize(), "cudaTS AddLC");

						// EvaluateRSSandDeriv(p_Tcurr) :
						// call to get heatflux and hence epsilon:

						cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
						kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
							pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
							p_Tcurr,
							pX_use->p_B + BEGINNING_OF_CENTRAL,
							p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
							NT_addition_rates_d_temp, pX_use->p_AreaMajor,
							// scrap masking for now --- but bring it back intelligently???			
							p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false, //bool bUseMask,
							iSpecies);
						Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

						CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
						kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
							// eps = T - (T_k +- h sum kappa dot grad T)
							hsub, pX_use->p_info + BEGINNING_OF_CENTRAL,
							p_Tcurr, p_T_k,
							NT_addition_rates_d_temp,			// NEED N = n*AreaMajor
							pX_use->p_n_major, pX_use->p_AreaMajor,
							p_epsilon, p_bFailed,
							p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, bUseMask,
							iSpecies);
						Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

						//kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
						//	(p_epsilon, p_sum_eps_eps);
						//Call(cudaThreadSynchronize(), "cudaTS AccSum");
						//cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
						//f64 sum_eps_eps = 0.0;
						//for (iTile = 0; iTile < numTilesMajorClever; iTile++)
						//	sum_eps_eps += p_sum_eps_eps_host[iTile];
						//RSScurr = sum_eps_eps;
						// Let's see whether sum_d_d is now small and we can stop.

						// Get dRSS/dbeta:

						kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
							pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
							p_Tcurr,
							hsub,
							p_regressors,
							pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
							p_d_eps_by_d_beta_x_, // d/dbeta of d(NT)/dt in this cell
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*iSpecies, // scrap masking for now --- but bring it back intelligently??? ???
							p_boolarray_block,
							false, //bool bUseMask,
							iSpecies);
						Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRateROC_wrt_regressor_1s");

						// This ought to work even though it's set up for minor cells:
						kernelAccumulateSummands2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
							pX_use->p_info + BEGINNING_OF_CENTRAL,
							p_epsilon,
							p_d_eps_by_d_beta_x_, // will this fail?
							p_temp1, //p_sum_eps_d,
							p_temp2, //p_sum_d_d,
							p_temp3 // p_sum_eps_eps
							);
						Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummands2");
						cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
						cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
						cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
						//sum_eps_d_curr = 0.0;
						//sum_d_d_curr = 0.0;
						sum_eps_eps_curr = 0.0;
						fb = 0.0;
						for (iTile = 0; iTile < numTilesMajorClever; iTile++) {
							fb += p_temphost1[iTile];
//							sum_d_d_curr += p_temphost2[iTile];
							sum_eps_eps_curr += p_temphost3[iTile];
						};

						printf("vWDB : b = %1.11E RSS = 1.11E \n", b, sum_eps_eps_curr);

	//					fb = (*func)(b);
						
					};// while (bContinue);
					
				};
			};

			// beta_sec = beta_prev + (beta_curr - beta_prev)*(-sum_eps_d_prev) / (sum_eps_d_curr - sum_eps_d_prev);

			//bContinue = true;
			//beta_next = beta_curr;
			//RSS_next = sum_eps_eps_curr;
			//deriv_next = sum_eps_d_curr; // note no factor 2 was applied
			//beta_curr = beta_sec; // loop

			// ======================================
			// Calls to ADHR are all about equal cost, we need to minimize.
			// ==========================================================================

			cudaMemcpy(p_T, p_Tcurr, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
		} else {
			// Neutrals: just accept the linear move.
			// We have only calc'd beta, we have not computed RSS or T. !
			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("beforehand p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);

			AddLCtoT << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_Tcurr, p_T, beta_curr, p_regressors);
			Call(cudaThreadSynchronize(), "cudaTS AddLCtoT");

			cudaMemcpy(&tempf64, &(p_regressors[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("p_regressors[%d] = %1.12E \n", VERTCHOSEN, tempf64);

			printf("Added beta_curr = %1.9E times regressor to produce p_Tcurr \n", beta_curr);

			cudaMemcpy(p_T, p_Tcurr, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			
			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);

		};
		// save regressor:
		//cudaMemcpy(p_regressors + NUMVERTICES, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// We don't save the regressor itself, but the previous steepest descent direction.

		++iIteration;
	} while (bContinue1);

	return 0;
}

int RunGeometricHeatConduction_NonlinearCG(f64 * p_T_k, f64 * p_T,
	f64 hsub, cuSyst * pX_use, bool bUseMask, f64 * p_kappa, f64 * p_nu,
	int iSpecies)
{
	f64 L2eps, L2reg;
	bool bFailedTest, bContinue;
	Triangle * pTri;
	f64 tempf64;
	long iTile, i;

	int iIteration = 0;
	f64 beta_PR;
	f64 tempdebug;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
	// We certainly should run with previous move as regressor and do a line search from that before we do anything.
#define zerovec1 p_Residuals

	printf("RunGeometricHeatConduction_Nonlinear__SDx3:\n");

	CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
	CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
	CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES));

	// call that first and send species kappa.

	GlobalSuppressSuccessVerbosity = true;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	if (iSpecies == 0) SetConsoleTextAttribute(hConsole, 14);
	if (iSpecies == 1) SetConsoleTextAttribute(hConsole, 11);
	if (iSpecies == 2) SetConsoleTextAttribute(hConsole, 15); // 12= red 11=cyan

	

	printf("\n\n Don't forget to actually augment d/dt NT when we are done here.\n\n");
	


	f64 storeRHS[4];
	f64 storemat[4 * 4];
	f64 beta[4];


	bool bContinue1 = true;
	do {
		// There should be a 0th step where we use previous moves.
		// Ideally store 2 moves, do multivariate regression to get direction;
		// line search for that direction.
		
		// Store initial position:
		cudaMemcpy(p_T0, p_T, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	//	cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

	//	cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("BEFORE L2EPS CALC: p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
		  
		// Get epsilon for use in SD calc:
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu, 
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block,	false, 
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
		kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			hsub,pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_T, p_T_k, NT_addition_rates_d_temp,
			pX_use->p_n_major, pX_use->p_AreaMajor, 
			p_epsilon, p_bFailed,
			p_boolarray2 + NUMVERTICES*iSpecies,p_boolarray_block,bUseMask,
			iSpecies
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		
		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		if (bUseMask == 0) {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		} else {
			f64 over = over_iEquations_n;
			if (iSpecies == 1) over = over_iEquations_i;
			if (iSpecies == 2) over = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over);
		}
		printf(" L2eps %1.11E  :\n ", L2eps);
		f64 RSS1 = sum_eps_eps; // stored.
		
		if (L2eps > 1.0) getch();

		if (L2eps < 1.0e-19) bContinue1 = false; // FOR NOW

		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		//  1. Calculate new steepest descent direction.
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		// Shift memory out ; we could use pointers to cycle and avoid doing this!
		if (iIteration >= 3) 
			cudaMemcpy(p_regressors + NUMVERTICES * 3, p_regressors + 2*NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		if (iIteration >= 2) 
			cudaMemcpy(p_regressors + NUMVERTICES*2, p_regressors+NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		if (iIteration >= 1)
			cudaMemcpy(p_regressors + NUMVERTICES, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
#define p_large_array_maxneigh_by_numvertices  d_eps_by_dx_neigh_n
		kernelAccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
			p_T, p_epsilon,	hsub,
			pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.
			pX_use->p_AreaMajor,
			p_boolarray2 + NUMVERTICES*iSpecies,p_boolarray_block,false, 
			iSpecies,
			p_large_array_maxneigh_by_numvertices,
			p_temp1 // effect self
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric");
		AddFromMyNeighbours << < numTilesMajor, threadsPerTileMajor >> >( // Also add + 1.0 on self, so + epsilon in other words.
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_large_array_maxneigh_by_numvertices, // needs to be MAXNEIGH*NUMVERTICES
			p_temp1, // effect self
			p_regressors,
			pX_use->p_izNeigh_vert,
			sz_who_vert_vert
			);
		Call(cudaThreadSynchronize(), "cudaTS AddFromMyNeighbours");
		
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		// 2. Search in up-to-4-dimensional space using these regressor directions:
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		for (int ii = 0; ii <= min(iIteration, 3); ii++) {
			kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
				p_T, hsub,
				p_regressors + NUMVERTICES * ii,
				pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
				p_Ax + NUMVERTICES * ii,
				pX_use->p_AreaMajor,
				p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false, //bool bUseMask,
				iSpecies);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRateROC_wrt_regressor_1s");
		};
		// Notice that the rate of change is dependent on T position so we cannot store and reuse it between iterations.

		kernelAccumulateSummands_4x4 << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			p_epsilon,
			p_Ax,
			p_sum_eps_deps_by_dbeta_x8,
			p_sum_depsbydbeta_8x8
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands_4x4");

		// have 20 doubles per thread in shared. That's ok with 256 in tile : 40K ish. Check that shared is preferred
		
		cudaMemcpy(p_sum_eps_deps_by_dbeta_x8_host, p_sum_eps_deps_by_dbeta_x8, sizeof(f64) * 4 * numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 4 * 4 * numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_deps_by_dbeta_vector[4];

		double mat[4 * 4];
		memset(sum_eps_deps_by_dbeta_vector, 0, 4 * sizeof(f64));
		memset(mat, 0, 4 * 4 * sizeof(f64));
		sum_eps_eps = 0.0;
		int i, j;
		for (i = 0; i < 4; i++)
			for (j = 0; j < 4; j++)
				for (iTile = 0; iTile < numTilesMajorClever; iTile++)
					mat[i * 4 + j] += p_sum_depsbydbeta_8x8_host[iTile * 4 * 4 + i * 4 + j];
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			for (i = 0; i < 4; i++)
				sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile * 4 + i];
		 
		// Here ensure that unwanted rows are 0. First wipe out any that accumulated 0.
		// or are beyond #equations.
		for (i = 0; i < 4; i++)
		{
			if ((mat[i * 4 + i] == 0.0) || (i > iIteration))
			{
				memset(mat + i * 4, 0, sizeof(f64) * 4);
				mat[i * 4 + i] = 1.0;
				sum_eps_deps_by_dbeta_vector[i] = 0.0;
			};
		};
		// Note that if a colour was not relevant for volleys, that just covered it.

		memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64) * 4);
		memcpy(storemat, mat, sizeof(f64) * 4 * 4);
		
		Matrix_real matLU;
		matLU.Invoke(4);

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				matLU.LU[i][j] = mat[i * 4 + j];

		matLU.LUdecomp();
		matLU.LUSolve(sum_eps_deps_by_dbeta_vector, beta); // solving Ax = b and it's (b, x).

		/*
#ifdef LAPACKE
		lapack_int ipiv[4];
		lapack_int Nrows = 4,
			Ncols = 4,  // lda
			Nrhscols = 1, // ldb
			Nrhsrows = 4, info; // MUST SOLVE ALL 4 EVERY TIME. Matrix is not drawn up to have 2x2 elements in correct positions for solve 2x2.


		// Solve the equations A*X = B 
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
		// Check for the exact singularity :
		if (info != 0) {
			//	printf("The diagonal element of the triangular factor of A,\n");
			//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
			printf("the solution could not be computed.\n");
			getch();
		}
		else {
			memcpy(beta, sum_eps_deps_by_dbeta_vector, 4 * sizeof(f64));
		};
#endif
		*/

		printf(" ( %1.9E %1.9E %1.9E %1.9E ) ( beta0 )   ( %1.9E )\n",
			storemat[0], storemat[1], storemat[2], storemat[3], storeRHS[0]);
		printf(" ( %1.9E %1.9E %1.9E %1.9E ) ( beta0 ) = ( %1.9E )\n",
			storemat[4], storemat[5], storemat[6], storemat[7], storeRHS[1]);
		printf(" ( %1.9E %1.9E %1.9E %1.9E ) ( beta0 )   ( %1.9E )\n",
			storemat[8], storemat[9], storemat[10], storemat[11], storeRHS[2]);
		printf(" ( %1.9E %1.9E %1.9E %1.9E ) ( beta0 )   ( %1.9E )\n",
			storemat[12], storemat[13], storemat[14], storemat[15], storeRHS[3]);
		printf("\n beta : %1.9E %1.9E %1.9E %1.9E\n"
			"-----------------------------------------------------------------------------\n",
			beta[0], beta[1], beta[2], beta[3]);
		
		// How to decide 'magnitude of dRSS/dbeta_overall' at point 0 ?
		// d / dbeta_overall represents increasing all components proportionally.
		// ie coefficient = beta_overall * beta

		// This is only half the actual deriv = sum eps.deps 
		f64 dRSSbydbeta0 = storeRHS[0] * beta[0] + storeRHS[1] * beta[1] + storeRHS[2] * beta[2] + storeRHS[3] * beta[3];
#define p_regr_x p_Ap_n

		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, 4 * sizeof(f64)));

		// store directional regressor:
		kernelAddtoT_lc___ << < numTilesMajorClever, threadsPerTileMajorClever >> >(
			p_regr_x, zerovec1,
			p_regressors, // regressors
			4,  // number of regrs
			1.0 // multiply beta_n_c by this factor.
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAddtoT_lc___");
		
	//	cudaMemcpy(&tempf64, &(p_regressors[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("p_regressors[%d] = %1.12E \n", VERTCHOSEN, tempf64);
	//	cudaMemcpy(&tempf64, &(p_regr_x[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("p_regr_x[%d] = %1.12E \n", VERTCHOSEN, tempf64);


		if (iSpecies == 0) {
			// For neutrals we do not need to do line search.

		//	cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//	printf("beforehand p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);

			kernelAdd2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_T, p_T0,
				1.0, // multiply x by this factor.
				p_regr_x // regressor
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelAddtoT_lc___");

	//		cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
	//		cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);


			printf("p_T neutral computed using beta_curr = %1.9E \n", 1.0);

			// done.
		} else {

			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			//   3. Do a brief line search in the computed direction.
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			f64 RSS_prev = RSS1;
			f64 beta_prev = 0.0;
			f64 beta_curr = 1.0;
			f64 sum_eps_d_prev = dRSSbydbeta0, sum_eps_d_curr;
			f64 RSS_curr;

			bContinue = true;
			while (bContinue) {

	//			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("beforehand p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
	//			cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

				kernelAdd2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					p_T, p_T0,
					beta_curr, // multiply x by this factor.
					p_regr_x // regressor
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelAddtoT_lc___");

	//			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
	//			cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

				printf("p_T computed using beta_curr = %1.9E \n", beta_curr);

				// ===============================================================================
				// Compute new epsilon and RSS:
				// ===============================================================================
				// Note that if we stop here we could avoid recalc'ing at the start of the loop.

				cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
				kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
					p_T,
					pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
					NT_addition_rates_d_temp,
					pX_use->p_AreaMajor,
					p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false,
					iSpecies);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

				CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
				kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					hsub, pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_T, p_T_k, NT_addition_rates_d_temp,
					pX_use->p_n_major, pX_use->p_AreaMajor,
					p_epsilon, p_bFailed,
					p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, bUseMask,
					iSpecies
					);
				Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

				kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
					(p_epsilon, p_sum_eps_eps);
				Call(cudaThreadSynchronize(), "cudaTS AccSum");
				cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				f64 sum_eps_eps = 0.0;
				for (iTile = 0; iTile < numTilesMajorClever; iTile++)
					sum_eps_eps += p_sum_eps_eps_host[iTile];
				RSS_curr = sum_eps_eps;

				printf("RSS_curr = %1.10E L2eps = %1.10E \n", RSS_curr,
					sqrt(RSS_curr / (real)NUMVERTICES));

				// Now compute dRSS/dbeta at this new position.

				// The directional vector [x = sum beta regressor] should be first stored.
				// Now we can take deps/d beta_x at our present position.

				kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
					p_T, hsub,
					p_regr_x,
					pX_use->p_B + BEGINNING_OF_CENTRAL, p_kappa, p_nu,
					p_d_eps_by_d_beta_x_,
					pX_use->p_AreaMajor,
					p_boolarray2 + NUMVERTICES*iSpecies, p_boolarray_block, false, //bool bUseMask,
					iSpecies);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRateROC_wrt_regressor_1s LINESEARCH");

				// collect eps against d_eps and d_eps against itself:
				// This ought to work even though it's set up for minor cells:
				kernelAccumulateSummands2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon,
					p_d_eps_by_d_beta_x_, // will this fail?
					p_temp1, //p_sum_eps_d,
					p_temp2, //p_sum_d_d,
					p_temp3 // p_sum_eps_eps
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummands2");

				cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);

				f64 sum_eps_d = 0.0, sum_d_d = 0.0;
				sum_eps_eps = 0.0;
				for (iTile = 0; iTile < numTilesMajorClever; iTile++) {
					sum_eps_d += p_temphost1[iTile];
					sum_d_d += p_temphost2[iTile];
					sum_eps_eps += p_temphost3[iTile];
					//	printf("iTile %d %1.9E %1.9E %1.9E ", iTile, p_temphost1[iTile], p_temphost2[iTile], p_temphost3[iTile]);		
				};
				sum_eps_d_curr = sum_eps_d;
				f64 beta_1 = -sum_eps_d / sum_d_d;
				// beta_1 says how much more of this regressor to add.
				// Bear in mind the initial move said add 1.0 of it.

				printf("sum_eps_eps %1.12E \n", sum_eps_eps);

				if (fabs(beta_1) < 0.0001) {
					// good enough.
					beta_curr += beta_1;
		//			cudaMemcpy(&tempf64, &(p_regr_x[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//			printf("p_regr_x[%d] = %1.12E \n", VERTCHOSEN, tempf64);
		//			cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//			printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

					kernelAdd2 << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						p_T, p_T0,
						beta_curr, // multiply x by this factor.
						p_regr_x // regressor
						);
					Call(cudaThreadSynchronize(), "cudaTS kernelAddtoT_lc___");

		//			cudaMemcpy(&tempf64, &(p_T[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//			printf("p_T[%d] = %1.12E \n", VERTCHOSEN, tempf64);
		//			cudaMemcpy(&tempf64, &(p_T0[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//			printf("p_T0[%d] = %1.12E \n", VERTCHOSEN, tempf64);

					bContinue = false;
					printf("end of line search; beta_curr used: %1.9E\n", beta_curr);
				} else {

					if ((RSS_curr < RSS_prev) && (beta_1 > 0.0))// -0.5*fabs(beta_curr-beta_prev))) {
					{
						sum_eps_d_prev = sum_eps_d_curr;
						RSS_prev = RSS_curr;
						beta_prev = beta_curr;
						beta_curr += beta_1;

						// We need a special procedure if we allow walking backwards
					} else {
						
						// Either beta_1 says move backwards, but RSS has decreased;
						// so dRSS/dbeta has crossed 0.
						// Or RSS has increased even though beta_1 says still move forwards,
						// which implies derivative has to have crossed 0 twice.

						// Fit a quadratic to find where dRSS/dbeta went through 0 :
						f64 f1 = 2.0*sum_eps_d_curr;
						f64 f0 = 2.0*sum_eps_d_prev;
						f64 overdelta = 1.0 / (0.5*(beta_curr - beta_prev));
						//printf("f0 %1.14E f1 %1.14E beta_curr %1.14E beta_prev %1.8E RSS2 %1.14E RSS1 %1.14E\n",
						//	f0, f1, beta_curr, beta_prev, sum_eps_eps_curr, sum_eps_eps_prev);
						// In a nasty enough case we could still get stuck in a loop.
						// Maybe we should just ignore this possibility and concentrate on WDB for f'.
						// Too much code overall.
						f64 b = (f1 - f0) / (beta_curr - beta_prev);
						f64 over_a = 1.0 / (-0.75*((RSS_curr - RSS_prev)*overdelta - (f1 + f0))*overdelta*overdelta);
						f64 c = 0.75*(RSS_curr - RSS_prev)*overdelta - 0.25*(f0 + f1);
						f64 tempsqrt = sqrt(-c*over_a + 0.25*b*b*over_a*over_a);
						printf("b %1.14E over_a %1.14E c %1.14E tempsqrt %1.14E \n", b, over_a, c, tempsqrt);
						f64 beta_minus = 0.5*(beta_prev + beta_curr) - tempsqrt - 0.5*(b*over_a);
						f64 beta_plus = 0.5*(beta_prev + beta_curr) + tempsqrt - 0.5*(b*over_a);
						if ((beta_minus > beta_prev) && (beta_minus < beta_curr))
						{
							beta_curr = beta_minus;

							printf("Quadratic succeeded -- beta_minus = %1.9E \n", beta_minus);
						} else {
							if ((beta_plus > beta_prev) && (beta_plus < beta_curr))
							{
								beta_curr = beta_plus; // probably never happens

								printf("Quadratic succeeded -- beta_plus = %1.9E \n", beta_plus);
							}
							else {
								printf("Quadratic failed! beta_prev %1.9E minus %1.9E plus %1.9E curr %1.9E\n",
									beta_prev, beta_minus, beta_plus, beta_curr);
								beta_curr = (beta_curr + beta_prev)*0.5;
							};
						};
					};
				};
			};

		};
				
		printf("iteration %d complete \n", iIteration);

		++iIteration;
	} while (bContinue1);

	printf("\nconvergence achieved.\n");

	return 0;
}
 /*
int RunBwdJnLSForHeat(f64 * p_T_k, f64 * p_T, f64 hsub, cuSyst * pX_use, bool bUseMask,
	int species, f64 * p_kappa, f64 * p_nu) // not sure if we can pass device pointers or not
{
	// The idea: pick Jacobi for definiteness. 
	// First write without equilibration, then compare two versions.
	// Try n = 6, 12 regressors.
	Matrix_real sum_ROC_products;
	f64 sum_eps_deps_by_dbeta_vector[24];
	f64 beta[24];

	long pointless_dummy[8] = {19180, 28610, 28607, 32192, 32190, 32183, 28592};//19163 fails

	printf("\nJLS^ %d for heat: \n", REGRESSORS);
	//long iMinor;
	f64 L2eps, L2reg;
	bool bFailedTest, bContinue;
	Triangle * pTri;
	f64 tempf64; 
	long iTile, i;

	int iIteration = 0;

	f64 tempdebug;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
#define zerovec1 p_temp1

	CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*(REGRESSORS+1)));
	CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
	CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
	cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES);
	
	printf("iEquations[%d] %d\n", species, iEquations[species]);

	if (iEquations[species] <= REGRESSORS) {
		// solve with regressors that are Kronecker delta for each point.
		// Solution should be same as just solving equations directly.

		// CPU search for which elements and put them into a list.
		long equationindex[REGRESSORS];

		cudaMemcpy(p_boolhost, p_boolarray2 + species*NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
		
		long iCaret = 0;
		for (i = 0; i < NUMVERTICES; i++)
		{
			if (p_boolhost[i]) {
				equationindex[iCaret] = i;
		//		printf("eqnindex[%d] = %d\n", iCaret, i);
				iCaret++;
			};
		}
		if (iCaret != iEquations[species]) {
			printf("(iCaret != iEquations[species])\n");
			getch(); getch(); getch(); getch(); getch(); return 1000;
		} else {
		//	printf("iCaret %d iEquations[%d] %d \n", iCaret, species, iEquations[species]);
		}

		f64 one = 1.0;
		for (i = 0; i < iCaret; i++) {
			cudaMemcpy(p_regressors + i*NUMVERTICES + equationindex[i], &one, sizeof(f64), cudaMemcpyHostToDevice);
		};
		// Then we want to fall out of branch into creating Ax.
		
		// And we want to make sure we construct the matrix with ID & 0 RHS for the unused equations.
		
		// Solution should be exact but then we can let it fall out of loop naturally?

		// Leave this as done and just skip regressor creation.

	} else {
		// Else we probably, should be using volleys especially if there is a mask set.
		
		//WE want to look into this:
		kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				hsub,
				pX_use->p_info, // minor
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				pX_use->p_AreaMajor,
				 
				p_coeffself_n,
				p_coeffself_i,
				p_coeffself_e // what exactly it calculates?
				); // used for Jacobi
		Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

		p_coeffself = p_coeffself_n;
		if (species == 1) p_coeffself = p_coeffself_i;
		if (species == 2) p_coeffself = p_coeffself_e;

		cudaMemcpy(&tempdebug, p_coeffself + pointless_dummy[0], sizeof(f64), cudaMemcpyDeviceToHost);
		printf("coeffself[%d] = %1.14E\n", pointless_dummy[0], tempdebug);
	};
	 
	::GlobalSuppressSuccessVerbosity = true; 
	char buffer[256];
	 
	iIteration = 0;
	do {
		printf("\nspecies %d ITERATION %d : ", species, iIteration);
		// create epsilon, & Jacobi 0th regressor.

//		And we want to look into this:
	//	and all the similar routines. What did we do near insulator?!

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
		( 
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa,
			p_nu,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2 + NUMVERTICES*species, // FOR SPECIES NOW
			p_boolarray_block,
			bUseMask,
			species
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		 
		if (REGRESSORS < iEquations[species]) {
			// Note: most are near 1, a few are like 22 or 150.
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonAndJacobi_Heat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				// eps = T - (T_k +- h sum kappa dot grad T)
				// x = -eps/coeffself
				hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_T,
				p_T_k,
				NT_addition_rates_d_temp,
				// NEED N = n*AreaMajor
				pX_use->p_n_major, // got this
				pX_use->p_AreaMajor, // got this -> N, Nn

				p_coeffself,

				p_epsilon,
				p_regressors,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species,
				true // yes to eps in regressor
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
			// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0
		} else {
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				// eps = T - (T_k +- h sum kappa dot grad T)
				// x = -eps/coeffself
				hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_T,
				p_T_k,
				NT_addition_rates_d_temp,
				// NEED N = n*AreaMajor
				pX_use->p_n_major, // got this
				pX_use->p_AreaMajor, // got this -> N, Nn
				p_epsilon,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		};
		
		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> > 
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];		
		if (bUseMask == 0) {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		} else {
			f64 over = over_iEquations_n;
			if (species == 1) over = over_iEquations_i;
			if (species == 2) over = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over);
		}
		printf(" L2eps %1.11E  : ", L2eps);

		// Weird: it picks coefficients on normalized regressors that are all high,
		// yet it never reduces L2eps by much.
		// Is it worth understanding why that is?

		
		// graph:
		// draw a graph:
		
		//SetActiveWindow(hwndGraphics);

		//cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//	pdata->temp.x = p_temphost1[iVertex];
		//	pdata->temp.y = p_temphost1[iVertex];

		//	++pVertex;
		//	++pdata;
		//}

		//sprintf(buffer, "epsilon iteration %d", iIteration);
		//Graph[0].DrawSurface(buffer,
		//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//	false,
		//	GRAPH_EPSILON, pTriMesh);

		//cudaMemcpy(p_temphost1, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//	pdata->temp.x = p_temphost1[iVertex];
		//	pdata->temp.y = p_temphost1[iVertex];

		//	++pVertex;
		//	++pdata;
		//}
		//sprintf(buffer, "Jac0 iteration %d", iIteration);
		//Graph[1].DrawSurface(buffer,
		//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//	false,
		//	GRAPH_AZ, pTriMesh);
		//cudaMemcpy(p_temphost1, p_regressors + NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//
		//// temp5 is predicted difference
		//// temp6 is old epsilon
		//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp4, p_temp6, p_epsilon);
		//// temp4 = actual difference & it's right-left, so new eps-old eps
		//Call(cudaThreadSynchronize(), "subtractvector");
		//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp3, p_temp5, p_temp4);
		//// temp3 = predicted-actual
		//Call(cudaThreadSynchronize(), "subtractvector");


		//cudaMemcpy(p_temphost3, p_temp5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//	pdata->temp.x = p_temphost3[iVertex];
		//	pdata->temp.y = p_temphost3[iVertex];

		//	++pVertex;
		//	++pdata;
		//}
		//
		//Graph[2].DrawSurface("predicted difference",
		//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//	false,
		//	GRAPH_AZ, pTriMesh);

		////cudaMemcpy(p_temphost1, p_regressors + 2 * NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//   
		//cudaMemcpy(p_temphost1, p_temp3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//	pdata->temp.x = p_temphost1[iVertex];
		//	pdata->temp.y = p_temphost1[iVertex];

		//	++pVertex;
		//	++pdata;
		//}

		//Graph[3].DrawSurface("difference from pred",
		//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//	false,
		//	GRAPH_AZ, pTriMesh);

		//SetActiveWindow(hwndGraphics);
		//ShowWindow(hwndGraphics, SW_HIDE);
		//ShowWindow(hwndGraphics, SW_SHOW);
		//Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		//printf("done graphs\n\n");

		//getch();

		//cudaMemcpy(p_temp6, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		

		// Did epsilon now pass test? If so, skip to the end.

		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		} else {
			printf("bFailedTest false \n");
		};

		bContinue = bFailedTest; 
		if (bContinue) {

			// DEBUG:
			bool bUseVolleys = (iIteration % 2 == 0);
			//if (bUseMask == 0) bUseVolleys = !bUseVolleys; // start without volleys for unmasked.

			// To prepare volley regressors we only need 2 x Jacobi:
			if (iEquations[species] > REGRESSORS) {

				for (i = 1; ((i <= REGRESSORS) || ((bUseVolleys) && (i <= 2))); i++)
				{

					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + (i - 1)*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonAndJacobi_Heat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + (i - 1)*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_coeffself,
						p_Ax + (i - 1)*NUMVERTICES, // the Ax for i-1; they will thus be defined for 0 up to 7 
						p_regressors + i*NUMVERTICES, // we need extra space here to create the last one we never use.
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species,
						true // no to eps in regressor
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

#define DO_NOT_NORMALIZE_REGRESSORS
#ifndef DO_NOT_NORMALIZE_REGRESSORS

					kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(p_regressors + i*NUMVERTICES, p_sum_eps_eps);
					Call(cudaThreadSynchronize(), "cudaTS AccSum");
					cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
					f64 sum_eps_eps = 0.0;
					for (iTile = 0; iTile < numTilesMajorClever; iTile++)
						sum_eps_eps += p_sum_eps_eps_host[iTile];
					if (bUseMask == 0) {
						L2reg = sqrt(sum_eps_eps / (real)NUMVERTICES);
					}
					else {
						f64 over = over_iEquations_n;
						if (species == 1) over = over_iEquations_i;
						if (species == 2) over = over_iEquations_e;
						L2reg = sqrt(sum_eps_eps * over);
					};
					// Now in order to set L2reg = L2eps say, we want to multiply by
					// alpha = sqrt(L2eps/L2reg)
					f64 alpha = sqrt(L2eps / L2reg);
					if (L2reg == 0.0) alpha = 0.0;

					kernelMultiplyVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_regressors + i*NUMVERTICES, alpha);
					Call(cudaThreadSynchronize(), "cudaTS MultiplyVector");

#endif
				};
			};

			if ((iEquations[species] > REGRESSORS) && (bUseVolleys)) {
				// Now create volleys:
				kernelVolleyRegressors << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					p_regressors,
					NUMVERTICES,
					pX_use->p_iVolley
					);
				Call(cudaThreadSynchronize(), "cudaTS volley regressors");
			};

			if ((iEquations[species] <= REGRESSORS) || (bUseVolleys)) {
				cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS);
				// only first few columns actually needed it
				for (i = 0; ((i < iEquations[species]) && (i < REGRESSORS)); i++)
				{
					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + i*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + i*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_Ax + i*NUMVERTICES, // the output
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


				};

			};
			lapack_int ipiv[REGRESSORS];
			double mat[REGRESSORS*REGRESSORS];
			lapack_int Nrows = REGRESSORS,
				Ncols = REGRESSORS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = REGRESSORS, info;

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			// if we introduce skipping blocks, must change to MajorClever
			kernelAccumulateSummands7 << <numTilesMajor, threadsPerTileMajor >> > (
				p_epsilon,
				p_Ax, // be careful: what do we take minus?
				p_sum_eps_deps_by_dbeta_x8,
				p_sum_depsbydbeta_8x8  // not sure if we want to store 64 things in memory?
				);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
							// Better off running through multiple times and doing 4 saves. But it's optimization.
			Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands");
			// Say we store 24 doubles/thread. So 4x4?. We could multiply 2 sets of 4.
			// We are at 8 for now so let's stick with the 8-way routine.

			cudaMemcpy(p_sum_eps_deps_by_dbeta_x8_host, p_sum_eps_deps_by_dbeta_x8, sizeof(f64) * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * REGRESSORS * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);

			memset(sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64));
			memset(mat, 0, REGRESSORS*REGRESSORS * sizeof(f64));
			sum_eps_eps = 0.0;
			int i, j;

			//for (iTile = 0; iTile < numTilesMajor; iTile++) {
			//	printf("iTile %d : %1.9E %1.9E\n",iTile,
			//		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
			//		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
			//};

			for (i = 0; i < REGRESSORS; i++)
				for (j = 0; j < REGRESSORS; j++)
					for (iTile = 0; iTile < numTilesMajor; iTile++)
						mat[i*REGRESSORS + j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];

			for (iTile = 0; iTile < numTilesMajor; iTile++)
				for (i = 0; i < REGRESSORS; i++)
					sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
			// let's say they are in rows of 8 per tile.

	//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
	//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
	//		printf("\n");

		// Here ensure that unwanted rows are 0. First wipe out any that accumulated 0.
			// or are beyond #equations.
			for (i = 0; i < REGRESSORS; i++)
			{
				if ((mat[i*REGRESSORS + i] == 0.0) || (i >= iEquations[species]))
				{
					memset(mat + i*REGRESSORS, 0, sizeof(f64)*REGRESSORS);
					mat[i*REGRESSORS + i] = 1.0;
					sum_eps_deps_by_dbeta_vector[i] = 0.0;
				}
			}
			// Note that if a colour was not relevant for volleys, that just covered it.

			f64 storeRHS[REGRESSORS];
			f64 storemat[REGRESSORS*REGRESSORS];
			memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
			memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

			// * Need to test speed against our own LU method.

			//	printf("LAPACKE_dgesv Results\n");
			// Solve the equations A*X = B
			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
			// Check for the exact singularity :


			if (info > 0) {
			//	printf("The diagonal element of the triangular factor of A,\n");
			//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n");

				if (bUseVolleys) {
					// Try deleting every other regressor
					memcpy(mat, storemat, sizeof(f64)*REGRESSORS*REGRESSORS);
					memcpy(sum_eps_deps_by_dbeta_vector, storeRHS, sizeof(f64)*REGRESSORS);

					memset(mat + 8, 0, sizeof(f64) * 8);
					memset(mat + 3*8, 0, sizeof(f64) * 8);
					memset(mat + 5*8, 0, sizeof(f64) * 8);
					memset(mat + 7*8, 0, sizeof(f64) * 8);
					mat[1 * 8 + 1] = 1.0;
					mat[3 * 8 + 3] = 1.0;
					mat[5 * 8 + 5] = 1.0;
					mat[7 * 8 + 7] = 1.0;
					sum_eps_deps_by_dbeta_vector[1] = 0.0;
					sum_eps_deps_by_dbeta_vector[3] = 0.0;
					sum_eps_deps_by_dbeta_vector[5] = 0.0;
					sum_eps_deps_by_dbeta_vector[7] = 0.0;

			//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
			//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
			//		printf("\n");

			//		f64 storeRHS[REGRESSORS];
			//		f64 storemat[REGRESSORS*REGRESSORS];
					memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
					memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

					// * making sure that we had zeroed any unwanted colours already.

					printf("LAPACKE_dgesv Results (volleys for 1 regressor only) \n");
					// Solve the equations A*X = B
					info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
					// Check for the exact singularity :
					if (info > 0) {
						printf("still didn't work..\n");
						print_matrix("Entry Matrix A", Nrows, Ncols, storemat, Ncols);
						print_matrix("Right Hand Side", Nrows, Nrhscols, storeRHS, Nrhscols);

						while (1) getch();
					};

					// Do not know whether my own LU is faster than LAPACKE dgesv.
				}; // (bUseVolleys)
			};


		//	print_matrix("Solution",Nrows, 1, sum_eps_deps_by_dbeta_vector, Nrhscols);
		//	print_matrix("Details of LU factorization",Nrows,Ncols,mat, Ncols);
		//	print_int_vector("Pivot indices",Nrows, ipiv);
			//
			if (info == 0) {
				memcpy(beta, sum_eps_deps_by_dbeta_vector, REGRESSORS * sizeof(f64));

				//sum_ROC_products.Invoke(REGRESSORS); // does set to zero
				// oooh memset(&sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64)); // why & ????
				//sum_eps_eps = 0.0;
				//int i, j;

				////for (iTile = 0; iTile < numTilesMajor; iTile++) {
				////	printf("iTile %d : %1.9E %1.9E\n",iTile,
				////		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
				////		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
				////};

				//for (i = 0; i < REGRESSORS; i++)
				//	for (j = 0; j < REGRESSORS; j++)
				//		for (iTile = 0; iTile < numTilesMajor; iTile++) {
				//			sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];
				//		}
				//for (iTile = 0; iTile < numTilesMajor; iTile++)
				//	for (i = 0; i < REGRESSORS; i++)
				//		sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
				//// let's say they are in rows of 8 per tile.

				//for (i = 0; i < REGRESSORS; i++) {
				//	printf("{ ");
				//	for (j = 0; j < REGRESSORS; j++)
				//		printf(" %1.8E ", sum_ROC_products.LU[i][j]);
				//	printf(" } { beta%d } ", i);
				//	if (i == 3) { printf(" = "); }
				//	else { printf("   "); };
				//	printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
				//	// Or is it minus??
				//};

				//memset(beta, 0, sizeof(f64) * REGRESSORS);
				////if (L2eps > 1.0e-28) { // otherwise just STOP !
				//					   // 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.
				//					   // Test for a zero row:
				//
				//bool zero_present = false;
				//for (i = 0; i <  REGRESSORS; i++)
				//{
				//	f64 sum = 0.0;
				//	for (j = 0; j < REGRESSORS; j++)
				//		sum += sum_ROC_products.LU[i][j];
				//	if (sum == 0.0) zero_present = true;
				//};
				//if (zero_present == false) {
				//
				//	// DEBUG:
				//	printf("sum_ROC_products.LUdecomp() :");
				//	sum_ROC_products.LUdecomp();
				//	printf("done\n");
				//	printf("sum_ROC_products.LUSolve : ");
				//	sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
				//	printf("done\n");

				//} else {
				//	printf("zero row present -- gah\n");
				//};

				printf("\nbeta: ");
				for (i = 0; i < REGRESSORS; i++)
					printf(" %1.8E ", beta[i]);
				printf("\n");

				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

				// add lc to our T

				kernelAddtoT_lc << <numTilesMajor, threadsPerTileMajor >> > (
					p_T, p_regressors, REGRESSORS);
				Call(cudaThreadSynchronize(), "cudaTS AddtoT");
			};
						
			// store predicted difference:
			ScaleVector << <numTilesMajor, threadsPerTileMajor >> > (p_temp5, beta[0], p_Ax);
			Call(cudaThreadSynchronize(), "ScaleVector");

			iIteration++;

		}; // if (bContinue)
	} while (bContinue);

	// To test whether this is sane, we need to spit out typical element in 0th and 8th iterate.

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return 0;
}
*/
int RunBwdRnLSForHeat(f64 * p_T_k, f64 * p_T, f64 hsub,
				cuSyst * pX_use, bool bUseMask,
				int species, f64 * p_kappa, f64 * p_nu) // not sure if we can pass device pointers or not
	{
		// The idea: pick Jacobi for definiteness.
		// First write without equilibration, then compare two versions.
		// Try n = 6, 12 regressors.
		Matrix_real sum_ROC_products;
		f64 sum_eps_deps_by_dbeta_vector[24];
		f64 beta[24];
		int ctr = 0;
		printf("\nRLS%d for heat: \n", REGRESSORS);
		//long iMinor;
		f64 L2eps, L2reg;
		bool bFailedTest, bContinue;
		Triangle * pTri;
		f64 tempf64;
		long iTile, i;

		int iIteration = 0;

		f64 tempdebug;
		Vertex * pVertex;
		long iVertex;
		plasma_data * pdata;
	#define zerovec1 p_temp1

		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

		if (species == 0) SetConsoleTextAttribute(hConsole, 14);
		if (species == 1) SetConsoleTextAttribute(hConsole, 10);
		if (species == 2) SetConsoleTextAttribute(hConsole, 15); // 12= red 11=cyan


		// We enter this routine with p_regressors + 7*NUMVERTICES already set up as a historic regressor.

		// Hang on.
		CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*(REGRESSORS - 1)));

		// regressors+7*NUMVERTICES came from seed.		

		CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
		CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
		cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES);

		printf("iEquations[%d] %d\n", species, iEquations[species]);

		if (iEquations[species] <= REGRESSORS) {
			// solve with regressors that are Kronecker delta for each point.
			// Solution should be same as just solving equations directly.
			CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*(REGRESSORS)));

			// CPU search for which elements and put them into a list.
			long equationindex[REGRESSORS];

			cudaMemcpy(p_boolhost, p_boolarray2 + species*NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);

			long iCaret = 0;
			for (i = 0; i < NUMVERTICES; i++)
			{
				if (p_boolhost[i]) {
					equationindex[iCaret] = i;
					//		printf("eqnindex[%d] = %d\n", iCaret, i);
					iCaret++;
				};
			}
			if (iCaret != iEquations[species]) {
				printf("(iCaret != iEquations[species])\n");
				getch(); getch(); getch(); getch(); getch(); return 1000;
			}
			else {
				//	printf("iCaret %d iEquations[%d] %d \n", iCaret, species, iEquations[species]);
			}

			f64 one = 1.0;
			for (i = 0; i < iCaret; i++) {
				cudaMemcpy(p_regressors + i*NUMVERTICES + equationindex[i], &one, sizeof(f64), cudaMemcpyHostToDevice);
			};
			// Then we want to fall out of branch into creating Ax.

			// And we want to make sure we construct the matrix with ID & 0 RHS for the unused equations.

			// Solution should be exact but then we can let it fall out of loop naturally?

			// Leave this as done and just skip regressor creation.

		} else {

		};
		 
		char buffer[256];
		char o = '4';

		f64 RSS = 0.0;
		f64 oldRSS;
		bool bSlowProgress = false;

		iIteration = 0;
		do {
			printf("\nspecies %d ITERATION %d : ", species, iIteration);
			// create epsilon, & Jacobi 0th regressor.

			//		And we want to look into this:
			//	and all the similar routines. What did we do near insulator?!

			cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

			kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >(
				pX_use->p_info, pX_use->p_izNeigh_vert, pX_use->p_szPBCneigh_vert, pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert, pX_use->p_cc, pX_use->p_n_major,
				p_T,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa, p_nu,  // make sure we are passing what is calc'd for vertices.

				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor,

				// scrap masking for now --- but bring it back intelligently???
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				false, //bool bUseMask,

						// Just hope that our clever version will converge fast.
				species);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");


			if (REGRESSORS < iEquations[species]) {
				// Note: most are near 1, a few are like 22 or 150.
				CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
				kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					// eps = T - (T_k +- h sum kappa dot grad T)
					// x = -eps/coeffself
					hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_T, 
					p_T_k,
					NT_addition_rates_d_temp,
					// NEED N = n*AreaMajor
					pX_use->p_n_major, // got this
					pX_use->p_AreaMajor, // got this -> N, Nn

					p_epsilon,
					p_bFailed,
					p_boolarray2 + NUMVERTICES*species,
					p_boolarray_block,
					bUseMask,
					species 
					);
				Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
				// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0

				// get sum of squares:
				kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
					(p_epsilon, p_temp2);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateSumOfSquares1");
				cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
				oldRSS = RSS;
				for (i = 0; i < numTilesMajorClever; i++) RSS += p_temphost2[i];
				printf("RSS %1.10E", RSS);
				if (
				   ((RSS < oldRSS*1.000000000001) && (RSS > 0.99*oldRSS) && (oldRSS > 0.0))
				|| ((iIteration % 100 == 0) && (iIteration > 200))
					){
					bSlowProgress = true;
					ctr = 0;
				};


				// We don't want to get trapped by Jacobi increasing RSS

				// New way: 
				// Let's include Jacobi as first regressor.
				
				cudaMemcpy(p_regressors + NUMVERTICES, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);				
				// ? CallMAC(cudaMemset(p_Ax + (i - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES));
				
				kernelHeat_1species_geometric_coeffself << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					pX_use->p_info,
					pX_use->p_izNeigh_vert,
					pX_use->p_szPBCneigh_vert,
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_cc,
					pX_use->p_n_major,
					p_T,  // evaluation point
					hsub,
					pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
					p_kappa,
					p_nu,
					pX_use->p_AreaMajor,
					p_boolarray2 + NUMVERTICES*species,
					p_boolarray_block,
					bUseMask,
					species,
					p_Ap_e  // coeffself
					);						
				Call(cudaThreadSynchronize(), "cudaTS kernelHeat_1species_geometric_coeffself");

				kernelDivide << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					p_regressors,
					p_epsilon,
					p_Ap_e
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelDivide");

			} else {
				CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
				kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					// eps = T - (T_k +- h sum kappa dot grad T)
					// x = -eps/coeffselfhsub,
				
					hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_T,
					p_T_k,
					NT_addition_rates_d_temp,
					// NEED N = n*AreaMajor
					pX_use->p_n_major, // got this
					pX_use->p_AreaMajor, // got this -> N, Nn
					p_epsilon,
					p_bFailed,
					p_boolarray2 + NUMVERTICES*species,
					p_boolarray_block,
					bUseMask,
					species
					);
				Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
			};

			kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_epsilon, p_sum_eps_eps);
			Call(cudaThreadSynchronize(), "cudaTS AccSum");
			cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 sum_eps_eps = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				sum_eps_eps += p_sum_eps_eps_host[iTile];
			if (bUseMask == 0) {
				L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
			}
			else {
				f64 over = over_iEquations_n;
				if (species == 1) over = over_iEquations_i;
				if (species == 2) over = over_iEquations_e;
				L2eps = sqrt(sum_eps_eps * over);
			}
			printf(" L2eps %1.11E  : ", L2eps);

			// Weird: it picks coefficients on normalized regressors that are all high,
			// yet it never reduces L2eps by much.
			// Is it worth understanding why that is?


			// graph:
			// draw a graph:
					
			if (iIteration > 600) {

				SetActiveWindow(hwndGraphics);

				cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost1[iVertex];
					pdata->temp.y = p_temphost1[iVertex];
					++pVertex;
					++pdata;
				};
				sprintf(buffer, "epsilon, iteration %d", iIteration);
				Graph[0].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);

				cudaMemcpy(p_temphost1, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost1[iVertex];
					pdata->temp.y = p_temphost1[iVertex];
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "Jacobi, iteration %d", iIteration);
				Graph[1].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_AZ, pTriMesh);
				
				for (i = 2; i < 5; i++) {
					cudaMemcpy(p_temphost1, p_regressors + i*NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
					pVertex = pTriMesh->X;
					pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
					for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
					{
						pdata->temp.x = p_temphost1[iVertex];
						pdata->temp.y = p_temphost1[iVertex];
						++pVertex;
						++pdata;
					}
					sprintf(buffer, "regressor %d, iteration %d", i, iIteration);
					Graph[i + 1].DrawSurface(buffer,
						DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
						AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
						false,
						GRAPH_AZ, pTriMesh);
				};


				cudaMemcpy(p_temphost1, p_regressors + NUMVERTICES*7, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost1[iVertex];
					pdata->temp.y = p_temphost1[iVertex];
					++pVertex;
					++pdata;
				};
				sprintf(buffer, "regressor 7, iteration %d", iIteration);
				Graph[5].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_AZ, pTriMesh);
				//cudaMemcpy(p_temphost1, p_regressors + NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);



				//// temp5 is predicted difference
				//// temp6 is old epsilon
				//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > 
				//	(p_temp4, p_temp6, p_epsilon);
				//// temp4 = actual difference & it's right-left, so new eps-old eps
				//
				//Call(cudaThreadSynchronize(), "subtractvector");
				//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > 
				//	(p_temp3, p_temp5, p_temp4);
				//// temp3 = predicted-actual
				//Call(cudaThreadSynchronize(), "subtractvector");


				//cudaMemcpy(p_temphost3, p_temp5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

				//pVertex = pTriMesh->X;
				//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				//{
				//	pdata->temp.x = p_temphost3[iVertex];
				//	pdata->temp.y = p_temphost3[iVertex];

				//	++pVertex;
				//	++pdata;
				//}

				//Graph[2].DrawSurface("predicted difference",
				//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				//	false,
				//	GRAPH_AZ, pTriMesh);

				////cudaMemcpy(p_temphost1, p_regressors + 2 * NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

				//cudaMemcpy(p_temphost1, p_temp3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				//pVertex = pTriMesh->X;
				//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				//
				//long iMax = 0;
				//f64 maxi = 0.0;
				//
				//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				//{
				//	pdata->temp.x = p_temphost1[iVertex];
				//	pdata->temp.y = p_temphost1[iVertex];
				//	if (fabs(p_temphost1[iVertex]) > maxi) {
				//		maxi = fabs(p_temphost1[iVertex]);
				//		iMax = iVertex;
				//	}
				//	++pVertex;
				//	++pdata;
				//}

				//Graph[3].DrawSurface("difference from pred",
				//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				//	AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				//	false,
				//	GRAPH_AZ, pTriMesh);
				//printf("\nMax %1.10E was found at %d \n\n", maxi, iMax);


				SetActiveWindow(hwndGraphics);
				ShowWindow(hwndGraphics, SW_HIDE);
				ShowWindow(hwndGraphics, SW_SHOW);
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

				printf("done graphs\n\n");
				printf("1 = restrict regressors to regr 1\n");
				printf("2 = restrict regressors to regr 1 vertex 26107\n");
				printf("3 = restrict regressors to regr 1 vertex 26084\n");
				printf("4 = no restriction\n");
				do {
					o = getch();
				} while ((o != '1') && (o != '2') && (o != '3') && (o != '4'));

				if (o == '2') {
					f64 tempf64;
					cudaMemcpy(&tempf64, p_regressors + 26107, sizeof(f64), cudaMemcpyDeviceToHost);
					cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES);
					cudaMemcpy(p_regressors + 26107, &tempf64, sizeof(f64), cudaMemcpyHostToDevice);
				};
				if (o == '3') {
					f64 tempf64;
					cudaMemcpy(&tempf64, p_regressors + 26084, sizeof(f64), cudaMemcpyDeviceToHost);
					cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES);
					cudaMemcpy(p_regressors + 26084, &tempf64, sizeof(f64), cudaMemcpyHostToDevice);
					
				};

				cudaMemcpy(p_temp6, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			};

			// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		

			// Did epsilon now pass test? If so, skip to the end.

			bFailedTest = false;
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
				if (p_boolhost[iTile]) bFailedTest = true;
			if (bFailedTest) {
				printf("bFailedTest true \n");
			}
			else {
				printf("bFailedTest false \n");
			};

			bContinue = bFailedTest;
			if (bContinue) {

				bool bUseVolleys = (iIteration % 8 == 0);
				if (bSlowProgress) bUseVolleys = false;
										//if (bUseMask == 0) bUseVolleys = !bUseVolleys; // start without volleys for unmasked.

										// To prepare volley regressors we only need 2 x Jacobi:
				if (iEquations[species] > REGRESSORS) {

					// 0th is Jacobi
					// let regr 7 already be set also, to previous move.
					
					for (i = 0; ((i < REGRESSORS) || ((bUseVolleys) && (i < 2))) ; i++)
					{
						// Here we need to be careful.
						// ... we want to take deps/dbeta at the position T.

						// create depsilon/dbeta and Jacobi for this regressor
						CallMAC(cudaMemset(p_Ax + i*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES));
						kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >
							(
								pX_use->p_info,
								pX_use->p_izNeigh_vert,
								pX_use->p_szPBCneigh_vert,
								pX_use->p_izTri_vert,
								pX_use->p_szPBCtri_vert,
								pX_use->p_cc,
								pX_use->p_n_major,

								p_T,  // evaluation point
								hsub,
								p_regressors + i*NUMVERTICES, // proposed direction
								//p_regressors + (i - 1)*NUMVERTICES, // input as T
								pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
								p_kappa,
								p_nu,
								p_Ax + i*NUMVERTICES, // output = deps/dbeta
								pX_use->p_AreaMajor,
								p_boolarray2 + NUMVERTICES*species,
								p_boolarray_block,
								bUseMask,
								species);
						// used for epsilon (T)
						Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric ");

						//kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						//	// eps = T - (T_k +- h sum kappa dot grad T)
						//	// x = -eps/coeffself
						//	hsub,
						//	pX_use->p_info + BEGINNING_OF_CENTRAL,
						//	p_regressors + (i - 1)*NUMVERTICES, // input
						//	zerovec1,
						//	NT_addition_rates_d_temp,
						//	pX_use->p_n_major,
						//	pX_use->p_AreaMajor,
						//	p_Ax + (i - 1)*NUMVERTICES, // the Ax for i-1; they will thus be defined for 0 up to 7 
						//	0,
						//	p_boolarray2 + NUMVERTICES*species,
						//	p_boolarray_block,
						//	bUseMask,
						//	species
						//	);
						//Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

						// Note: we now have J and R as first two regressors from above.
						// Except in case that REGRESSORS > iEquations
						
						if ((bUseVolleys == false) && (i >= 1) && (i < REGRESSORS-2))
							// fill in REGRESSORS-2, and then make another pass which sets Ax for it,
							// and another pass which sets it for REGRESSORS-1 which is the prev move.
						{
							if (o < '4') {
								cudaMemset(p_regressors + (i+1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
							}
							else {
								//looks like a BUG

								// found bug: this said NMINOR.
								SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> >
									(p_regressors + (i + 1)*NUMVERTICES,
										p_Ax + i*NUMVERTICES, p_regressors + i*NUMVERTICES);
								Call(cudaThreadSynchronize(), "cudaTS Subtract");
								//cudaMemcpy(p_regressors + (i + 1)*NUMVERTICES,
								//	p_Ax + i*NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
								
							//#define DO_NOT_NORMALIZE_REGRESSORS
#ifndef DO_NOT_NORMALIZE_REGRESSORS

								kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
									(p_regressors + (i + 1)*NUMVERTICES, p_sum_eps_eps);
								Call(cudaThreadSynchronize(), "cudaTS AccSum");
								cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
								f64 sum_eps_eps = 0.0;
								for (iTile = 0; iTile < numTilesMajorClever; iTile++)
									sum_eps_eps += p_sum_eps_eps_host[iTile];


								// Early on in the simulation it is possible for the result to be zero throughout even though
								// we renormalized the vector. I'm not sure why this is -- just smoothing it enough times crushed whatever
								// impetus there was for heat conduction.
								// In this case set the regressor to 0 and when we get zeroes in the solution matrix we will set 1 on the diagonal.
								// ie set alpha = 0 if sum_eps^2 == 0.

								if (bUseMask == 0) {
									L2reg = sqrt(sum_eps_eps / (real)NUMVERTICES);
								}
								else {
									f64 over = over_iEquations_n;
									if (species == 1) over = over_iEquations_i;
									if (species == 2) over = over_iEquations_e;
									L2reg = sqrt(sum_eps_eps * over);
								};
								// Now in order to set L2reg = L2eps say, we want to multiply by
								// alpha = sqrt(L2eps/L2reg)
								f64 alpha;
								if (L2reg == 0.0) { alpha = 0.0; }
								else { alpha = sqrt(L2eps / L2reg); };

								//	printf("sum_eps_eps %1.9E  L2reg = %1.10E alpha = %1.10E \n", sum_eps_eps, L2reg, alpha);

								kernelMultiplyVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_regressors + (i + 1)*NUMVERTICES, alpha);
								Call(cudaThreadSynchronize(), "cudaTS MultiplyVector");
#endif
							};
						};
					};
				};

				if ((iEquations[species] > REGRESSORS) && (bUseVolleys)) {
					// Now create volleys:
					// Splits 2 into 8
					kernelVolleyRegressors << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						p_regressors + NUMVERTICES,
						NUMVERTICES,
						pX_use->p_iVolley
						);
					Call(cudaThreadSynchronize(), "cudaTS volley regressors");
				};

				if ((iEquations[species] <= REGRESSORS) || (bUseVolleys)) {
					cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS);
					// only first few columns actually needed it
					for (i = 0; ((i < iEquations[species]) && (i < REGRESSORS)); i++)
					{ 
						// create depsilon/dbeta and Jacobi for this regressor
						CallMAC(cudaMemset(p_Ax + i*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES));
						kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> >
							(
								pX_use->p_info,
								pX_use->p_izNeigh_vert,
								pX_use->p_szPBCneigh_vert,
								pX_use->p_izTri_vert,
								pX_use->p_szPBCtri_vert,
								pX_use->p_cc,
								pX_use->p_n_major,

								p_T,  // evaluation point
								hsub,
								p_regressors + i*NUMVERTICES, // proposed direction
																	//p_regressors + (i - 1)*NUMVERTICES, // input as T
								pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
								p_kappa,
								p_nu,
								p_Ax + i*NUMVERTICES, // output = deps/dbeta
								pX_use->p_AreaMajor,
								p_boolarray2 + NUMVERTICES*species,
								p_boolarray_block,
								bUseMask,
								species);
						// used for epsilon (T)
						Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric ");

					};

				};
				// We don't need to test for domain, we need to make sure the summands are zero otherwise.

				// if we introduce skipping blocks, must change to MajorClever
				kernelAccumulateSummands7 << <numTilesMajor, threadsPerTileMajor >> > (
					p_epsilon,
					p_Ax, // be careful: what do we take minus?
					p_sum_eps_deps_by_dbeta_x8,
					p_sum_depsbydbeta_8x8  // not sure if we want to store 64 things in memory?
					);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
								// Better off running through multiple times and doing 4 saves. But it's optimization.
				Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands");
				// Say we store 24 doubles/thread. So 4x4?. We could multiply 2 sets of 4.
				// We are at 8 for now so let's stick with the 8-way routine.

				cudaMemcpy(p_sum_eps_deps_by_dbeta_x8_host, p_sum_eps_deps_by_dbeta_x8, sizeof(f64) * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * REGRESSORS * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);

				double mat[REGRESSORS*REGRESSORS];
				memset(sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64));
				memset(mat, 0, REGRESSORS*REGRESSORS * sizeof(f64));
				sum_eps_eps = 0.0;
				int i, j;

				//for (iTile = 0; iTile < numTilesMajor; iTile++) {
				//	printf("iTile %d : %1.9E %1.9E\n",iTile,
				//		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
				//		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
				//};

				for (i = 0; i < REGRESSORS; i++)
					for (j = 0; j < REGRESSORS; j++)
						for (iTile = 0; iTile < numTilesMajor; iTile++)
							mat[i*REGRESSORS + j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];

				for (iTile = 0; iTile < numTilesMajor; iTile++)
					for (i = 0; i < REGRESSORS; i++) {
						sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
						if (sum_eps_deps_by_dbeta_vector[i] != sum_eps_deps_by_dbeta_vector[i]) printf("iTile %d : RHS %d = NaN\n", iTile, i);
					}
				// let's say they are in rows of 8 per tile.

				//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
				//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
				//		printf("\n");

				// Here ensure that unwanted rows are 0. First wipe out any that accumulated 0.
				// or are beyond #equations.
				for (i = 0; i < REGRESSORS; i++)
				{
					if ((mat[i*REGRESSORS + i] == 0.0) || (i >= iEquations[species]))
					{
						memset(mat + i*REGRESSORS, 0, sizeof(f64)*REGRESSORS);
						mat[i*REGRESSORS + i] = 1.0;
						sum_eps_deps_by_dbeta_vector[i] = 0.0;
					}
				}
				// Note that if a colour was not relevant for volleys, that just covered it.

				f64 storeRHS[REGRESSORS];
				f64 storemat[REGRESSORS*REGRESSORS];
				memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
				memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);
				if (!GlobalSuppressSuccessVerbosity) {
					for (i = 0; i < REGRESSORS; i++)
					{
						printf("( ");
						for (j = 0; j < REGRESSORS; j++)
							printf("%1.8E ", storemat[i*REGRESSORS + j]);
						printf(") (  )  =  ( %1.8E )\n", storeRHS[i]);
					};
					printf("\n");
				}
				Matrix_real matLU;
				matLU.Invoke(REGRESSORS);
				for (i = 0; i < REGRESSORS; i++)
					for (j = 0; j < REGRESSORS; j++)
						matLU.LU[i][j] = mat[i*REGRESSORS + j];
				if (!GlobalSuppressSuccessVerbosity) printf("Doing LUdecomp:\n");
				matLU.LUdecomp();
				if (!GlobalSuppressSuccessVerbosity) printf("Doing LUSolve:\n");
				matLU.LUSolve(sum_eps_deps_by_dbeta_vector, beta);

				if ((iIteration > 200) && (bSlowProgress))
				{
					// A punch of Jacobi
					memset(beta, 0, sizeof(f64) * 8);
					beta[0] = 0.8;
					SetConsoleTextAttribute(hConsole, 12);
					ctr++;
					if (ctr == 24) { // 16 * 0.8 = 12.8 Jacobi's.
						ctr = 0;
						bSlowProgress = false;
						SetConsoleTextAttribute(hConsole, 15);
					};
				}

				if (o == '1') {
					memset(beta, 0, sizeof(f64) * 8);
					beta[0] = 1.0;
				};
				if ((o == '2') || (o == '3')) {
					memset(beta, 0, sizeof(f64) * 8);
					beta[0] = 0.01;
				};

				printf("\nbeta: ");
				for (i = 0; i < REGRESSORS; i++)
					printf(" %1.8E ", beta[i]);
				printf("\n");

				
				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

				// add lc to our T

				kernelAddtoT_lc << <numTilesMajor, threadsPerTileMajor >> > (
					p_T, p_regressors, REGRESSORS); 
				// Also populates p_regressors + 7 * NUMVERTICES
				Call(cudaThreadSynchronize(), "cudaTS AddtoT");
				/*
#ifdef LAPACKE
				//It's actually a huge problem that we can't get it to run LAPACKE.
				// This ability to return that there is a singularity, is something we need to emulate.

				lapack_int ipiv[REGRESSORS];
				lapack_int Nrows = REGRESSORS,
					Ncols = REGRESSORS,  // lda
					Nrhscols = 1, // ldb
					Nrhsrows = REGRESSORS, info;

				// * Need to test speed against our own LU method.

				//	printf("LAPACKE_dgesv Results\n");
				// Solve the equations A*X = B
				info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
				// Check for the exact singularity :

				if (info > 0) {
					//	printf("The diagonal element of the triangular factor of A,\n");
					//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
					printf("the solution could not be computed.\n");

					if (bUseVolleys) {
						// Try deleting every other regressor
						memcpy(mat, storemat, sizeof(f64)*REGRESSORS*REGRESSORS);
						memcpy(sum_eps_deps_by_dbeta_vector, storeRHS, sizeof(f64)*REGRESSORS);

						memset(mat + 8, 0, sizeof(f64) * 8);
						memset(mat + 3 * 8, 0, sizeof(f64) * 8);
						memset(mat + 5 * 8, 0, sizeof(f64) * 8);
						memset(mat + 7 * 8, 0, sizeof(f64) * 8);
						mat[1 * 8 + 1] = 1.0;
						mat[3 * 8 + 3] = 1.0;
						mat[5 * 8 + 5] = 1.0;
						mat[7 * 8 + 7] = 1.0;
						sum_eps_deps_by_dbeta_vector[1] = 0.0;
						sum_eps_deps_by_dbeta_vector[3] = 0.0;
						sum_eps_deps_by_dbeta_vector[5] = 0.0;
						sum_eps_deps_by_dbeta_vector[7] = 0.0;

						//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
						//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
						//		printf("\n");

						//		f64 storeRHS[REGRESSORS];
						//		f64 storemat[REGRESSORS*REGRESSORS];
						memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
						memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

						// * making sure that we had zeroed any unwanted colours already.

						printf("LAPACKE_dgesv Results (volleys for 1 regressor only) \n");
						// Solve the equations A*X = B
						info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
						// Check for the exact singularity :
						if (info > 0) {
							printf("still didn't work..\n");
							print_matrix("Entry Matrix A", Nrows, Ncols, storemat, Ncols);
							print_matrix("Right Hand Side", Nrows, Nrhscols, storeRHS, Nrhscols);

							while (1) getch();
						};

						// Do not know whether my own LU is faster than LAPACKE dgesv.
					}; // (bUseVolleys)
				};
				

				//	print_matrix("Solution",Nrows, 1, sum_eps_deps_by_dbeta_vector, Nrhscols);
				//	print_matrix("Details of LU factorization",Nrows,Ncols,mat, Ncols);
				//	print_int_vector("Pivot indices",Nrows, ipiv);
				//
				if (info == 0) {
					memcpy(beta, sum_eps_deps_by_dbeta_vector, REGRESSORS * sizeof(f64));

					//sum_ROC_products.Invoke(REGRESSORS); // does set to zero
					// oooh memset(&sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64)); // why & ????
					//sum_eps_eps = 0.0;
					//int i, j;

					////for (iTile = 0; iTile < numTilesMajor; iTile++) {
					////	printf("iTile %d : %1.9E %1.9E\n",iTile,
					////		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
					////		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
					////};

					//for (i = 0; i < REGRESSORS; i++)
					//	for (j = 0; j < REGRESSORS; j++)
					//		for (iTile = 0; iTile < numTilesMajor; iTile++) {
					//			sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];
					//		}
					//for (iTile = 0; iTile < numTilesMajor; iTile++)
					//	for (i = 0; i < REGRESSORS; i++)
					//		sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
					//// let's say they are in rows of 8 per tile.

					//for (i = 0; i < REGRESSORS; i++) {
					//	printf("{ ");
					//	for (j = 0; j < REGRESSORS; j++)
					//		printf(" %1.8E ", sum_ROC_products.LU[i][j]);
					//	printf(" } { beta%d } ", i);
					//	if (i == 3) { printf(" = "); }
					//	else { printf("   "); };
					//	printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
					//	// Or is it minus??
					//};

					//memset(beta, 0, sizeof(f64) * REGRESSORS);
					////if (L2eps > 1.0e-28) { // otherwise just STOP !
					//					   // 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.
					//					   // Test for a zero row:
					//
					//bool zero_present = false;
					//for (i = 0; i <  REGRESSORS; i++)
					//{
					//	f64 sum = 0.0;
					//	for (j = 0; j < REGRESSORS; j++)
					//		sum += sum_ROC_products.LU[i][j];
					//	if (sum == 0.0) zero_present = true;
					//};
					//if (zero_present == false) {
					//
					//	// DEBUG:
					//	printf("sum_ROC_products.LUdecomp() :");
					//	sum_ROC_products.LUdecomp();
					//	printf("done\n");
					//	printf("sum_ROC_products.LUSolve : ");
					//	sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
					//	printf("done\n");

					//} else {
					//	printf("zero row present -- gah\n");
					//};

					printf("\nbeta: ");
					for (i = 0; i < REGRESSORS; i++)
						printf(" %1.8E ", beta[i]);
					printf("\n");

					CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

					// add lc to our T

					kernelAddtoT_lc << <numTilesMajor, threadsPerTileMajor >> > (
						p_T, p_regressors, REGRESSORS);
					Call(cudaThreadSynchronize(), "cudaTS AddtoT");
				};
#endif
*/
	// store predicted difference:
				ScaleVector << <numTilesMajor, threadsPerTileMajor >> > (p_temp5, beta[0], p_Ax);
				Call(cudaThreadSynchronize(), "ScaleVector");

				// predicted difference only works for regressor 1 only.
				
				iIteration++;

			} // if (bContinue) 
		} while (bContinue);

	// To test whether this is sane, we need to spit out typical element in 0th and 8th iterate.

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return 0;
}

/*
int RunBwdRnLSForHeatBackup(f64 * p_T_k, f64 * p_T, f64 hsub, cuSyst * pX_use, bool bUseMask,
	int species, f64 * p_kappa, f64 * p_nu) // not sure if we can pass device pointers or not
{
	// The idea: pick Jacobi for definiteness. 
	// First write without equilibration, then compare two versions.
	// Try n = 6, 12 regressors.
	Matrix_real sum_ROC_products;
	f64 sum_eps_deps_by_dbeta_vector[24];
	f64 beta[24];

	printf("\nRLS%d for heat: \n", REGRESSORS);
	//long iMinor;
	f64 L2eps, L2reg;
	bool bFailedTest, bContinue;
	Triangle * pTri;
	f64 tempf64;
	long iTile, i;

	int iIteration = 0;

	f64 tempdebug;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
#define zerovec1 p_temp1

	CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*(REGRESSORS + 1)));
	CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
	CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
	cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES);

	printf("iEquations[%d] %d\n", species, iEquations[species]);

	if (iEquations[species] <= REGRESSORS) {
		// solve with regressors that are Kronecker delta for each point.
		// Solution should be same as just solving equations directly.

		// CPU search for which elements and put them into a list.
		long equationindex[REGRESSORS];

		cudaMemcpy(p_boolhost, p_boolarray2 + species*NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);

		long iCaret = 0;
		for (i = 0; i < NUMVERTICES; i++)
		{
			if (p_boolhost[i]) {
				equationindex[iCaret] = i;
				//		printf("eqnindex[%d] = %d\n", iCaret, i);
				iCaret++;
			};
		}
		if (iCaret != iEquations[species]) {
			printf("(iCaret != iEquations[species])\n");
			getch(); getch(); getch(); getch(); getch(); return 1000;
		}
		else {
			//	printf("iCaret %d iEquations[%d] %d \n", iCaret, species, iEquations[species]);
		}

		f64 one = 1.0;
		for (i = 0; i < iCaret; i++) {
			cudaMemcpy(p_regressors + i*NUMVERTICES + equationindex[i], &one, sizeof(f64), cudaMemcpyHostToDevice);
		};
		// Then we want to fall out of branch into creating Ax.

		// And we want to make sure we construct the matrix with ID & 0 RHS for the unused equations.

		// Solution should be exact but then we can let it fall out of loop naturally?

		// Leave this as done and just skip regressor creation.

	}
	else {

	};

	::GlobalSuppressSuccessVerbosity = true;
	char buffer[256];

	iIteration = 0;
	do {
		printf("\nspecies %d ITERATION %d : ", species, iIteration);
		// create epsilon, & Jacobi 0th regressor.

		//		And we want to look into this:
		//	and all the similar routines. What did we do near insulator?!

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_T,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa,
				p_nu,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor,
				p_boolarray2 + NUMVERTICES*species, // FOR SPECIES NOW
				p_boolarray_block,
				bUseMask,
				species
				);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

		if (REGRESSORS < iEquations[species]) {
			// Note: most are near 1, a few are like 22 or 150.
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				// eps = T - (T_k +- h sum kappa dot grad T)
				// x = -eps/coeffself
				hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_T,
				p_T_k,
				NT_addition_rates_d_temp,
				// NEED N = n*AreaMajor
				pX_use->p_n_major, // got this
				pX_use->p_AreaMajor, // got this -> N, Nn

				p_epsilon,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
			// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0

			cudaMemcpy(p_regressors, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

		}
		else {
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				// eps = T - (T_k +- h sum kappa dot grad T)
				// x = -eps/coeffself
				hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_T,
				p_T_k,
				NT_addition_rates_d_temp,
				// NEED N = n*AreaMajor
				pX_use->p_n_major, // got this
				pX_use->p_AreaMajor, // got this -> N, Nn
				p_epsilon,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		};

		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		if (bUseMask == 0) {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		}
		else {
			f64 over = over_iEquations_n;
			if (species == 1) over = over_iEquations_i;
			if (species == 2) over = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over);
		}
		printf(" L2eps %1.11E  : ", L2eps);

		// Weird: it picks coefficients on normalized regressors that are all high,
		// yet it never reduces L2eps by much.
		// Is it worth understanding why that is?


		// graph:
		// draw a graph:
		
		//SetActiveWindow(hwndGraphics);

		//cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//pdata->temp.x = p_temphost1[iVertex];
		//pdata->temp.y = p_temphost1[iVertex];

		//++pVertex;
		//++pdata;
		//}

		//sprintf(buffer, "epsilon iteration %d", iIteration);
		//Graph[0].DrawSurface(buffer,
		//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//false,
		//GRAPH_EPSILON, pTriMesh);

		//cudaMemcpy(p_temphost1, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//pdata->temp.x = p_temphost1[iVertex];
		//pdata->temp.y = p_temphost1[iVertex];

		//++pVertex;
		//++pdata;
		//}
		//sprintf(buffer, "Jac0 iteration %d", iIteration);
		//Graph[1].DrawSurface(buffer,
		//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//false,
		//GRAPH_AZ, pTriMesh);
		//cudaMemcpy(p_temphost1, p_regressors + NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);


		//// temp5 is predicted difference
		//// temp6 is old epsilon
		//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp4, p_temp6, p_epsilon);
		//// temp4 = actual difference & it's right-left, so new eps-old eps
		//Call(cudaThreadSynchronize(), "subtractvector");
		//SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp3, p_temp5, p_temp4);
		//// temp3 = predicted-actual
		//Call(cudaThreadSynchronize(), "subtractvector");


		//cudaMemcpy(p_temphost3, p_temp5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//pdata->temp.x = p_temphost3[iVertex];
		//pdata->temp.y = p_temphost3[iVertex];

		//++pVertex;
		//++pdata;
		//}

		//Graph[2].DrawSurface("predicted difference",
		//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//false,
		//GRAPH_AZ, pTriMesh);

		////cudaMemcpy(p_temphost1, p_regressors + 2 * NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		//cudaMemcpy(p_temphost1, p_temp3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//pVertex = pTriMesh->X;
		//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		//{
		//pdata->temp.x = p_temphost1[iVertex];
		//pdata->temp.y = p_temphost1[iVertex];

		//++pVertex;
		//++pdata;
		//}

		//Graph[3].DrawSurface("difference from pred",
		//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		//false,
		//GRAPH_AZ, pTriMesh);

		//SetActiveWindow(hwndGraphics);
		//ShowWindow(hwndGraphics, SW_HIDE);
		//ShowWindow(hwndGraphics, SW_SHOW);
		//Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		//printf("done graphs\n\n");

		//getch();

		//cudaMemcpy(p_temp6, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);


		// Did epsilon now pass test? If so, skip to the end.

		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		}
		else {
			printf("bFailedTest false \n");
		};

		bContinue = bFailedTest;
		if (bContinue) {

			// DEBUG:
			bool bUseVolleys = false;//(iIteration % 2 == 0);
									 //if (bUseMask == 0) bUseVolleys = !bUseVolleys; // start without volleys for unmasked.

									 // To prepare volley regressors we only need 2 x Jacobi:
			if (iEquations[species] > REGRESSORS) {

				for (i = 1; ((i <= REGRESSORS) || ((bUseVolleys) && (i <= 2))); i++)
				{

					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + (i - 1)*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + (i - 1)*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_Ax + (i - 1)*NUMVERTICES, // the Ax for i-1; they will thus be defined for 0 up to 7 
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

					cudaMemcpy(p_regressors + i*NUMVERTICES,
						p_Ax + (i - 1)*NUMVERTICES, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);


					//#define DO_NOT_NORMALIZE_REGRESSORS
#ifndef DO_NOT_NORMALIZE_REGRESSORS

					kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(p_regressors + i*NUMVERTICES, p_sum_eps_eps);
					Call(cudaThreadSynchronize(), "cudaTS AccSum");
					cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
					f64 sum_eps_eps = 0.0;
					for (iTile = 0; iTile < numTilesMajorClever; iTile++)
						sum_eps_eps += p_sum_eps_eps_host[iTile];
					if (bUseMask == 0) {
						L2reg = sqrt(sum_eps_eps / (real)NUMVERTICES);
					}
					else {
						f64 over = over_iEquations_n;
						if (species == 1) over = over_iEquations_i;
						if (species == 2) over = over_iEquations_e;
						L2reg = sqrt(sum_eps_eps * over);
					};
					// Now in order to set L2reg = L2eps say, we want to multiply by
					// alpha = sqrt(L2eps/L2reg)
					f64 alpha = sqrt(L2eps / L2reg);
					if (L2reg == 0.0) alpha = 0.0;
					kernelMultiplyVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_regressors + i*NUMVERTICES, alpha);
					Call(cudaThreadSynchronize(), "cudaTS MultiplyVector");

#endif
				};
			};

			if ((iEquations[species] > REGRESSORS) && (bUseVolleys)) {
				// Now create volleys:
				kernelVolleyRegressors << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					p_regressors,
					NUMVERTICES,
					pX_use->p_iVolley
					);
				Call(cudaThreadSynchronize(), "cudaTS volley regressors");
			};

			if ((iEquations[species] <= REGRESSORS) || (bUseVolleys)) {
				cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS);
				// only first few columns actually needed it
				for (i = 0; ((i < iEquations[species]) && (i < REGRESSORS)); i++)
				{
					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + i*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + i*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_Ax + i*NUMVERTICES, // the output
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


				};

			};

			f64 storeRHS[REGRESSORS];
			f64 storemat[REGRESSORS*REGRESSORS];
			double mat[REGRESSORS*REGRESSORS];
			int info = 0;
			

			lapack_int ipiv[REGRESSORS];
			lapack_int Nrows = REGRESSORS,
				Ncols = REGRESSORS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = REGRESSORS, info;

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			// if we introduce skipping blocks, must change to MajorClever
			kernelAccumulateSummands7 << <numTilesMajor, threadsPerTileMajor >> > (
				p_epsilon,
				p_Ax, // be careful: what do we take minus?
				p_sum_eps_deps_by_dbeta_x8,
				p_sum_depsbydbeta_8x8  // not sure if we want to store 64 things in memory?
				);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
							// Better off running through multiple times and doing 4 saves. But it's optimization.
			Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands");
			// Say we store 24 doubles/thread. So 4x4?. We could multiply 2 sets of 4.
			// We are at 8 for now so let's stick with the 8-way routine.

			cudaMemcpy(p_sum_eps_deps_by_dbeta_x8_host, p_sum_eps_deps_by_dbeta_x8, sizeof(f64) * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * REGRESSORS * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);

			memset(sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64));
			memset(mat, 0, REGRESSORS*REGRESSORS * sizeof(f64));
			sum_eps_eps = 0.0;
			int i, j;

			//for (iTile = 0; iTile < numTilesMajor; iTile++) {
			//	printf("iTile %d : %1.9E %1.9E\n",iTile,
			//		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
			//		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
			//};

			for (i = 0; i < REGRESSORS; i++)
				for (j = 0; j < REGRESSORS; j++)
					for (iTile = 0; iTile < numTilesMajor; iTile++)
						mat[i*REGRESSORS + j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];

			for (iTile = 0; iTile < numTilesMajor; iTile++)
				for (i = 0; i < REGRESSORS; i++)
					sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
			// let's say they are in rows of 8 per tile.

			//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
			//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
			//		printf("\n");

			// Here ensure that unwanted rows are 0. First wipe out any that accumulated 0.
			// or are beyond #equations.
			for (i = 0; i < REGRESSORS; i++)
			{
				if ((mat[i*REGRESSORS + i] == 0.0) || (i >= iEquations[species]))
				{
					memset(mat + i*REGRESSORS, 0, sizeof(f64)*REGRESSORS);
					mat[i*REGRESSORS + i] = 1.0;
					sum_eps_deps_by_dbeta_vector[i] = 0.0;
				}
			}
			// Note that if a colour was not relevant for volleys, that just covered it.

			memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
			memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

			// * Need to test speed against our own LU method.

			//	printf("LAPACKE_dgesv Results\n");
			// Solve the equations A*X = B 
			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
			// Check for the exact singularity :

			if (info > 0) {
				//	printf("The diagonal element of the triangular factor of A,\n");
				//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n");

				if (bUseVolleys) {
					// Try deleting every other regressor
					memcpy(mat, storemat, sizeof(f64)*REGRESSORS*REGRESSORS);
					memcpy(sum_eps_deps_by_dbeta_vector, storeRHS, sizeof(f64)*REGRESSORS);

					memset(mat + 8, 0, sizeof(f64) * 8);
					memset(mat + 3 * 8, 0, sizeof(f64) * 8);
					memset(mat + 5 * 8, 0, sizeof(f64) * 8);
					memset(mat + 7 * 8, 0, sizeof(f64) * 8);
					mat[1 * 8 + 1] = 1.0;
					mat[3 * 8 + 3] = 1.0;
					mat[5 * 8 + 5] = 1.0;
					mat[7 * 8 + 7] = 1.0;
					sum_eps_deps_by_dbeta_vector[1] = 0.0;
					sum_eps_deps_by_dbeta_vector[3] = 0.0;
					sum_eps_deps_by_dbeta_vector[5] = 0.0;
					sum_eps_deps_by_dbeta_vector[7] = 0.0;

					//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
					//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
					//		printf("\n");

					//		f64 storeRHS[REGRESSORS];
					//		f64 storemat[REGRESSORS*REGRESSORS];
					memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
					memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

					// * making sure that we had zeroed any unwanted colours already.
#ifdef LAPACKE
					printf("LAPACKE_dgesv Results (volleys for 1 regressor only) \n");
					// Solve the equations A*X = B 
					info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
					// Check for the exact singularity :
					if (info > 0) {
						printf("still didn't work..\n");
						print_matrix("Entry Matrix A", Nrows, Ncols, storemat, Ncols);
						print_matrix("Right Hand Side", Nrows, Nrhscols, storeRHS, Nrhscols);

						while (1) getch();
					};
#endif

					// Do not know whether my own LU is faster than LAPACKE dgesv.
				}; // (bUseVolleys)
			};

			//	print_matrix("Solution",Nrows, 1, sum_eps_deps_by_dbeta_vector, Nrhscols);
			//	print_matrix("Details of LU factorization",Nrows,Ncols,mat, Ncols);
			//	print_int_vector("Pivot indices",Nrows, ipiv);
			//
			if (info == 0) {
				memcpy(beta, sum_eps_deps_by_dbeta_vector, REGRESSORS * sizeof(f64));

				//sum_ROC_products.Invoke(REGRESSORS); // does set to zero
				// oooh memset(&sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64)); // why & ????
				//sum_eps_eps = 0.0;
				//int i, j;

				////for (iTile = 0; iTile < numTilesMajor; iTile++) {
				////	printf("iTile %d : %1.9E %1.9E\n",iTile,
				////		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
				////		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
				////};

				//for (i = 0; i < REGRESSORS; i++)
				//	for (j = 0; j < REGRESSORS; j++)
				//		for (iTile = 0; iTile < numTilesMajor; iTile++) {
				//			sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];
				//		}
				//for (iTile = 0; iTile < numTilesMajor; iTile++)
				//	for (i = 0; i < REGRESSORS; i++)
				//		sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
				//// let's say they are in rows of 8 per tile.

				//for (i = 0; i < REGRESSORS; i++) {
				//	printf("{ ");
				//	for (j = 0; j < REGRESSORS; j++)
				//		printf(" %1.8E ", sum_ROC_products.LU[i][j]);
				//	printf(" } { beta%d } ", i);
				//	if (i == 3) { printf(" = "); }
				//	else { printf("   "); };
				//	printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
				//	// Or is it minus??
				//};

				//memset(beta, 0, sizeof(f64) * REGRESSORS);
				////if (L2eps > 1.0e-28) { // otherwise just STOP !
				//					   // 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.
				//					   // Test for a zero row:
				// 
				//bool zero_present = false;
				//for (i = 0; i <  REGRESSORS; i++)
				//{
				//	f64 sum = 0.0;
				//	for (j = 0; j < REGRESSORS; j++)
				//		sum += sum_ROC_products.LU[i][j];
				//	if (sum == 0.0) zero_present = true;
				//};
				//if (zero_present == false) {
				//	
				//	// DEBUG:
				//	printf("sum_ROC_products.LUdecomp() :");
				//	sum_ROC_products.LUdecomp();
				//	printf("done\n");
				//	printf("sum_ROC_products.LUSolve : ");
				//	sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
				//	printf("done\n");

				//} else {
				//	printf("zero row present -- gah\n");
				//};

				printf("\nbeta: ");
				for (i = 0; i < REGRESSORS; i++)
					printf(" %1.8E ", beta[i]);
				printf("\n");

				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

				// add lc to our T

				kernelAddtoT_lc << <numTilesMajor, threadsPerTileMajor >> > (
					p_T, p_regressors, REGRESSORS);
				Call(cudaThreadSynchronize(), "cudaTS AddtoT");
			};

			// store predicted difference:
			ScaleVector << <numTilesMajor, threadsPerTileMajor >> > (p_temp5, beta[0], p_Ax);
			Call(cudaThreadSynchronize(), "ScaleVector");

			iIteration++;

		}; // if (bContinue)
	} while (bContinue);

	// To test whether this is sane, we need to spit out typical element in 0th and 8th iterate.

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return 0;
}
*/

int RunBackwardJLSForHeat(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use,
	bool bUseMask)
{
#define UPLIFT_THRESHOLD 0.33
//	GlobalSuppressSuccessVerbosity = true;



	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;
	bool bProgress;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	// seed: just set T to T_k.
//	cudaMemcpy(p_T, p_T_k, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	
	// Assume we were passed the seed.

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T);
	Call(cudaThreadSynchronize(), "cudaTS unpack");
	cudaMemset(p_T, 0, sizeof(T3)*NUMVERTICES);
	// Same should apply for all solver routines: initial seed to be created by regression from previous solutions.

	printf("\nJRLS for heat: ");
	//long iMinor;
	f64 L2eps_n, L2eps_e, L2eps_i, L2eps_n_old, L2eps_i_old, L2eps_e_old;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	int iIteration = 0;

	CallMAC(cudaMemset(p_Jacobi_n, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_Jacobi_i, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_Jacobi_e, 0, sizeof(f64)*NUMVERTICES));

	CallMAC(cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES)); // NEED TO LOOK AND DO SAME IN CG ROUTINE.


	// Better if we would just load in epsilon and create Jacobi? No coeffself array, simpler.
	// Careful: if we find coeffself in d/dt NT then eps <- -(h/N) d/dtNT
	// so we need N as well.
	kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			pX_use->p_AreaMajor,

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e // what exactly it calculates?
			); // used for Jacobi
	Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

	
	do {
		printf("\nITERATION %d \n\n", iIteration);
		if ((iIteration >= 2000) && (iIteration % 1 == 0))
		{
			// draw a graph:

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_Jacobi_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_temphost1[iVertex];
				pdata->temp.y = p_temphost1[iVertex];

				if (fabs(p_temphost1[iVertex]) > 1.0e-14) {
					printf("iVertex %d epsilon %1.10E deps/dJac %1.9E ", iVertex, p_temphost1[iVertex], p_temphost2[iVertex]);						
					cudaMemcpy(&tempf64, &(p_Jacobi_e[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Jac %1.8E ", tempf64);
					cudaMemcpy(&tempf64, &(p_Te[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Te %1.8E\n", tempf64);
				};
				//if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
				//{

				//	} else {
				//		pdata->temp.x = 0.0;
				//		pdata->temp.y = 0.0;
				//	}
				++pVertex;
				++pdata;
			}

			Graph[0].DrawSurface("uuuu",
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			//pVertex = pTriMesh->X;
			//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//{
			//pdata->temp.x = p_temphost3[iVertex];
			//pdata->temp.y = p_temphost3[iVertex];
			//++pVertex;
			//++pdata;
			//}

			////overdraw:
			//Graph[0].DrawSurface("Jacobi_e",
			//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			//false,
			//GRAPH_OPTI, pTriMesh);
			
//			pVertex = pTriMesh->X;
//			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
//			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//			{
//				pdata->temp.x = p_temphost3[iVertex];
//				pdata->temp.y = p_temphost3[iVertex];
//				++pVertex;
//				++pdata;
//			}
//			Graph[2].DrawSurface("Jacobi",
//				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
//				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
//				false,
//				GRAPH_NINE, pTriMesh);
//
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			printf("done graph");
		}
		// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
		// 3. Calculate Jacobi: for each point, Jacobi = eps/(deps/dT)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d_temp, 
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask
				);
		// used for epsilon (T) 
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		    
		// Note: most are near 1, a few are like 22 or 150.
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor));
		kernelCreateEpsilonAndJacobi_Heat << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_Tn, p_Ti, p_Te,
			p_T_k,
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e, 
			 
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e,
			p_bFailed,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0

		
		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = T - (T_k + h dT/dt)
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.

		cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES); // important to zero in mask!
		
		CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		 
		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_T, // zerovec
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor 
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			// no check threshold
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

		// and eps as regressor:

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_epsilon_n, p_epsilon_i, p_epsilon_e, // We could easily use 2nd iterate of Jacobi instead. Most probably, profitably.
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_T, // T3 zerovec
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			// no check threshold
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		
		
		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.

		//		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
		//			pX_use->p_info,
		//			p_epsilon_n,
		//			p_d_eps_by_dbeta_n,
		//			p_sum_eps_deps_by_dbeta,
		//			p_sum_depsbydbeta_sq,
		//			p_sum_eps_eps);
		//		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands OOA");


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_n,

			p_d_eps_by_dbeta_n,
			p_d_eps_by_dbetaR_n,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		f64 tempf64 = (real)NUMVERTICES;

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_nJ = 0.0;
			beta_nR = 0.0;
		}
		else {
			beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		}
		L2eps_n_old = L2eps_n;
		L2eps_n = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\nfor neutral [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_nJ, beta_nR, L2eps_n);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,
			p_d_eps_by_dbeta_i,
			p_d_eps_by_dbetaR_i,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		 
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_iJ = 0.0; beta_iR = 0.0;
		}
		else {
			beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);

			printf("sum_eps_deps_by_dbeta_J %1.10E sum_depsbydbeta_J_times_J %1.10E \n", sum_eps_deps_by_dbeta_J, sum_depsbydbeta_J_times_J);

		};
		
		L2eps_i_old = L2eps_i;
		L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ION [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_iJ, beta_iR, L2eps_i);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			p_epsilon_e,
			p_d_eps_by_dbeta_e,
			p_d_eps_by_dbetaR_e,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");
		
		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		 
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_eJ = 0.0; beta_eR = 0.0;
		}
		else {
			beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		

//			if (((fabs(beta_eJ) < 0.1) && (fabs(beta_eR) < 0.05)) 
//				|| ((iIteration > 100) && (iIteration % 4 != 0))) {
//				beta_eJ = 0.25; beta_eR = 0.0;
//			}	
			// Sometimes snarls things up. Switch back to CG instead.

		};
		
		L2eps_e_old = L2eps_e;
		L2eps_e = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\nfor Electron [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

		// ======================================================================================================================
		
		// bringing back adding both at once WOULD be more efficient.

		VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
			p_Tn, beta_nJ, p_Jacobi_n,
			p_Ti, beta_iJ, p_Jacobi_i,
			p_Te, beta_eJ, p_Jacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

		VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
			p_Tn, beta_nR, p_epsilon_n,
			p_Ti, beta_iR, p_epsilon_i,
			p_Te, beta_eR, p_epsilon_e);		
		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___2");
//		kernelAddtoT << <numTilesMajor, threadsPerTileMajor >> > (
//			p_Tn, p_Ti, p_Te, beta_nJ, beta_nR, beta_iJ, beta_iR, beta_eJ, beta_eR,
//			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
//			p_epsilon_n, p_epsilon_i, p_epsilon_e);		
		//Call(cudaThreadSynchronize(), "cudaTS AddtoT ___");
	
		iIteration++;

		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajor; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		}
		else {
			printf("bFailedTest false \n");
		}
		
		f64 ratio;
		bProgress = false;
		if ((L2eps_e_old < 1.0e-30) && (L2eps_i_old < 1.0e-30) && (L2eps_n_old < 1.0e-30)) 
		{
			// good enough for no test
			bProgress = true;
		} else {
			if (L2eps_e_old >= 1.0e-30) {
				ratio = (L2eps_e / L2eps_e_old);
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_i_old >= 1.0e-30) {
				ratio = (L2eps_i / L2eps_i_old);
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_n_old >= 1.0e-30) {
				ratio = (L2eps_n / L2eps_n_old);
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
		};

	} while ((iIteration < 2)
		|| ( (bFailedTest)
		&& ((iIteration < ITERATIONS_BEFORE_SWITCH) || (bProgress))
			 ) );

	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated NT.
	Call(cudaThreadSynchronize(), "cudaTS packup");

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return (bFailedTest == false) ? 0 : 1;



	// Modify denominator of L2 to reflect number of equations.



}


void RegressionSeedTe__damaged(f64 hsub, T3 * p_move1, T3 * p_move2, T3 * p_T, T3 * p_T_k, 
	cuSyst * pX_use, bool bUseMask)
{

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;

	long iVertex;
	f64 L2eps_n, L2eps_e, L2eps_i;

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T);
	Call(cudaThreadSynchronize(), "cudaTS unpack");
	// unpack moves to scalars :

	if (bUseMask) {
		cudaMemset(p_slot1n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot1i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot1e, 0, sizeof(f64)*NUMVERTICES);
		kernelUnpackWithMask << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot1n, p_slot1i, p_slot1e, p_move1,
				p_boolarray2,
				p_boolarray_block);
		Call(cudaThreadSynchronize(), "cudaTS unpack move1");

		cudaMemset(p_slot2n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot2i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot2e, 0, sizeof(f64)*NUMVERTICES);
		kernelUnpackWithMask << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot2n, p_slot2i, p_slot2e, p_move2,
				p_boolarray2,
				p_boolarray_block	);
		Call(cudaThreadSynchronize(), "cudaTS unpack move2");

	} else {
		kernelUnpack<< < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot1n, p_slot1i, p_slot1e, p_move1	);
		Call(cudaThreadSynchronize(), "cudaTS unpack move1");

		kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot2n, p_slot2i, p_slot2e, p_move2);
		Call(cudaThreadSynchronize(), "cudaTS unpack move2");
	}

	// move2 never gets used.
	
	// Important to ensure regressors were 0 in mask.

	cudaMemset(p_T, 0, sizeof(T3)*NUMVERTICES); // use as zerovec
	kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			pX_use->p_AreaMajor,

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e // what exactly it calculates?
			
			); // used for Jacobi
	Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

	// . Create epsilon for p_Tn etc
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
			
	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_use->p_info,
		pX_use->p_izNeigh_vert,
		pX_use->p_szPBCneigh_vert,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_cc,
		pX_use->p_n_major,
		p_Tn,
		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_nu_i,
		NT_addition_rates_d_temp,
		pX_use->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		0);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate neut");
	
	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_use->p_info,
		pX_use->p_izNeigh_vert,
		pX_use->p_szPBCneigh_vert,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_cc,
		pX_use->p_n_major,
		p_Ti,
		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_i,
		p_nu_i,
		NT_addition_rates_d_temp,
		pX_use->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		1);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate ion");

	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_use->p_info,
		pX_use->p_izNeigh_vert,
		pX_use->p_szPBCneigh_vert,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_cc,
		pX_use->p_n_major,
		p_Te,
		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_e,
		p_nu_e,
		NT_addition_rates_d_temp,
		pX_use->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		2);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate elec");



	//
	//kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
	//	(
	//		pX_use->p_info,
	//		pX_use->p_izNeigh_vert,
	//		pX_use->p_szPBCneigh_vert,
	//		pX_use->p_izTri_vert,
	//		pX_use->p_szPBCtri_vert,
	//		pX_use->p_cc,
	//		pX_use->p_n_major,
	//		p_Tn, p_Ti, p_Te,
	//		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
	//		p_kappa_n,
	//		p_kappa_i,
	//		p_kappa_e,
	//		p_nu_i,
	//		p_nu_e,
	//		NT_addition_rates_d_temp,
	//		pX_use->p_AreaMajor,
	//		p_boolarray2,
	//		p_boolarray_block,
	//		bUseMask);
	//Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES); 
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_Tn, p_Ti, p_Te,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon_n, p_epsilon_i, p_epsilon_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");  
	
	// . Now pretend p_move1 is p_T and enter zerovec for T_k
	// That gives us d_eps_by_d_1

	cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES);

	CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_slot1n, p_slot1i, p_slot1e,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemset(p_slot2n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_slot2i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_slot2e, 0, sizeof(f64)*NUMVERTICES); 
		// because we are about to overwrite it

	kernelCreateEpsilonAndJacobi_Heat << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot1n, p_slot1i, p_slot1e, // regressor
		p_T, // zerovec
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_coeffself_n, p_coeffself_i, p_coeffself_e,
		p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e,
		p_slot2n, p_slot2i, p_slot2e,
		0,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 1");
	
	CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_slot2n, p_slot2i, p_slot2e,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	// used for epsilon (T)
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//// Note: most are near 1, a few are like 22 or 150.


	//cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
	//cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
	//cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES);
	// We have done this.
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		// x = -eps/coeffself
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot2n, p_slot2i, p_slot2e,
		p_T, // zerovec
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn

		p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 2");

	// We have a 2-regression programmed. Should we have a 3-regression programmed also?
	// Do each species in turn:
	
	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
		// We don't need to test for domain, we need to make sure the summands are zero otherwise.
		p_epsilon_n,
		p_d_eps_by_dbeta_n,
		p_d_eps_by_dbetaR_n,
		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_nJ = 0.0;
		beta_nR = 0.0;
	} else {
		beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	}

	if (bUseMask) {
		L2eps_n = sqrt(sum_eps_eps * over_iEquations_n);
	} else {
		L2eps_n = sqrt(sum_eps_eps / (real)NUMVERTICES);
	};
	printf("\n for neutral [ Beta1 %1.10E Beta2 %1.10E L2eps(old) %1.10E ] ", beta_nJ, beta_nR, L2eps_n);
	
	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
		// We don't need to test for domain, we need to make sure the summands are zero otherwise.
		p_epsilon_i,
		p_d_eps_by_dbeta_i,
		p_d_eps_by_dbetaR_i,
		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_iJ = 0.0; beta_iR = 0.0;
	}
	else {
		beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	};

	if (bUseMask) {
		L2eps_i = sqrt(sum_eps_eps * over_iEquations_i);
	} else {
		L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);
	};
	printf("\n for ION [ Beta1 %1.10E Beta2 %1.10E L2eps(old) %1.10E ] ", beta_iJ, beta_iR, L2eps_i);
//
//
//	FILE * dbgfile = fopen("debugsolve2.txt", "w");
//	cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_temphost3, p_d_eps_by_dbetaR_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_boolhost, p_boolarray2 + 2 * NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//	{
//		if ((p_temphost1[iVertex] != 0.0) || (p_temphost2[iVertex] != 0.0) || (p_temphost3[iVertex] != 0.0))
//		{
//			fprintf(dbgfile, "iVertex %d eps %1.14E depsbydbeta1 %1.14E depsbydbeta2 %1.14E bool %d\n",
//				iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], (p_boolhost[iVertex] ? 1 : 0));
//		}
//	}
//	fclose(dbgfile);
//	printf("dbgfile done\n");

	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
		p_epsilon_e,
		p_d_eps_by_dbeta_e,
		p_d_eps_by_dbetaR_e,

		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands elec");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_eJ = 0.0; beta_eR = 0.0;
	}
	else {
		beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	
		printf("sum_eps_depsJ %1.11E sum_eps_deps2 %1.11E sum_JJ %1.11E sum_RR %1.11E sum_JR %1.11E\n",
			sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R, sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R,
			sum_depsbydbeta_J_times_R);
	
	};

	if (bUseMask) {
		L2eps_e = sqrt(sum_eps_eps * over_iEquations_e);
	} else {
		L2eps_e = sqrt(sum_eps_eps / (f64)NUMVERTICES);
	}
	printf("\n for Electron [ Beta1 %1.14E Beta2 %1.14E L2eps(old) %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

	// ======================================================================================================================

	// bringing back adding both at once WOULD be more efficient.

	cudaMemcpy(&tempf64, &(p_Te[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("Te[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	VectorAddMultiple_masked << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		p_Tn, beta_nJ, p_slot1n,
		p_Ti, beta_iJ, p_slot1i,
		p_Te, beta_eJ, p_slot1e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask		
		);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

//	cudaMemcpy(&tempf64, &(p_slot1e[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//	printf("regressor[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	
	VectorAddMultiple_masked << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		p_Tn, beta_nR, p_slot2n,
		p_Ti, beta_iR, p_slot2i,
		p_Te, beta_eR, p_slot2e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___2");
	
	cudaMemcpy(&tempf64, &(p_slot2e[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("regressor[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	 
	cudaMemcpy(&tempf64, &(p_Te[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("Te[%d]:%1.9E\n", VERTCHOSEN, tempf64);

	// Test effect of additions:

	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");


	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES); // should be unnecessary calls.
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_Tn, p_Ti, p_Te,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon_n, p_epsilon_i, p_epsilon_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


	kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_temp1, p_temp2, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	f64 SS_n = 0.0, SS_i = 0.0, SS_e = 0.0;
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		SS_n += p_temphost1[iTile];
		SS_i += p_temphost2[iTile];
		SS_e += p_temphost3[iTile];
	}

	if (bUseMask) {
		L2eps_n = sqrt(SS_n * over_iEquations_n);
		L2eps_i = sqrt(SS_i * over_iEquations_i);
		L2eps_e = sqrt(SS_e * over_iEquations_e);
	}
	else {
		L2eps_n = sqrt(SS_n / (real)NUMVERTICES);
		L2eps_i = sqrt(SS_i / (real)NUMVERTICES);
		L2eps_e = sqrt(SS_e / (real)NUMVERTICES);
	};

	printf("L2eps_n %1.10E  L2eps_i %1.10E  L2eps_e %1.10E \n\n", L2eps_n, L2eps_i, L2eps_e);

	//dbgfile = fopen("debugsolve3.txt", "w");
	//cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost3, p_d_eps_by_dbetaR_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_boolhost, p_boolarray2 + 2 * NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	//{
	//	if ((p_temphost1[iVertex] != 0.0) || (p_temphost2[iVertex] != 0.0) || (p_temphost3[iVertex] != 0.0))
	//	{
	//		fprintf(dbgfile, "iVertex %d eps %1.14E depsbydbeta1 %1.14E depsbydbeta2 %1.14E bool %d\n",
	//			iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], (p_boolhost[iVertex] ? 1 : 0));
	//	}
	//}
	//fclose(dbgfile);
	printf("dbgfile done\n");

	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	
	Call(cudaThreadSynchronize(), "cudaTS packup");
	

	// Next stop: be careful what we are adding to what.
	// TO produce seed regressors.

}


void RegressionSeedT_1species(f64 hsub, f64 * p_move1,
	//f64 * p_move2, 
	f64 * p_T, f64 * p_T_k,
	cuSyst * pX_use, f64 * p_kappa, f64 * p_nu, int iSpecies, bool bUseMask)
{

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;

	long iVertex;
	f64 L2eps;

	f64 tempf64;
	cudaMemcpy(&tempf64, &(p_T_k[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("p_T_k[%d] = %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemcpy(p_slot1i, p_move1, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(p_slot2i, p_move2, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	// move2 never gets used.

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// Important to ensure regressors were 0 in mask.
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	cudaMemcpy(&tempf64, &(p_T_k[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("p_T_k[%d] = %1.10E \n", VERTCHOSEN, tempf64);

	
	kernelHeat_1species_geometric_coeffself << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		pX_use->p_info, // minor
		pX_use->p_izNeigh_vert,
		pX_use->p_szPBCneigh_vert,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_cc,
		pX_use->p_n_major,
		p_T,
		hsub,
		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa,
		p_nu,
		pX_use->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		iSpecies,
		p_coeffself_i
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelHeat_1species_geometric_coeffself");
	cudaMemcpy(&tempf64, &(p_T_k[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("p_T_k[%d] = %1.10E \n", VERTCHOSEN, tempf64);

	for (int iPass = 0; iPass < 2; iPass++) {
		// . Create epsilon for p_Tn etc
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa,
			p_nu,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate neut");
		cudaMemcpy(&tempf64, &(p_T_k[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("p_T_k[%d] = %1.10E \n", VERTCHOSEN, tempf64);


		cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot2i, 0, sizeof(f64)*NUMVERTICES);
		// because we are about to overwrite it
		cudaMemcpy(&tempf64, &(p_T_k[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("p_T_k[%d] = %1.10E \n", VERTCHOSEN, tempf64);

		kernelCreateEpsilonAndJacobi_Heat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_T,
			p_T_k,
			NT_addition_rates_d_temp,
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn
			p_coeffself_i,
			p_epsilon,
			p_slot2i, // Jacobi regressor
			0,
			p_boolarray2,
			p_boolarray_block,
			bUseMask,
			iSpecies,
			true
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 1 and Jacobi");

		cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);

		kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_T,
			hsub,
			p_slot1i,
			pX_use->p_B + BEGINNING_OF_CENTRAL,
			p_kappa,
			p_nu,

			p_d_eps_by_dbetaR_n,  // for move1

			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS Create d/dbeta for p_slot1i");

		kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_T,
			hsub,
			p_slot2i,
			pX_use->p_B + BEGINNING_OF_CENTRAL,
			p_kappa,
			p_nu,

			p_d_eps_by_dbeta_n,  // for Jacobi

			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask,
			iSpecies);
		Call(cudaThreadSynchronize(), "cudaTS Create d/dbeta for p_slot2i");

		// We have a 2-regression programmed. Should we have a 3-regression programmed also?
		// Do each species in turn:

		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon,
			p_d_eps_by_dbeta_n,
			p_d_eps_by_dbetaR_n,
			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_nJ = 0.0;
			beta_nR = 0.0;
		}
		else {
			beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		}
		f64 L2eps;
		if (bUseMask) {
			long over_iEquations;
			if (iSpecies == 0) over_iEquations = over_iEquations_n;
			if (iSpecies == 1) over_iEquations = over_iEquations_i;
			if (iSpecies == 2) over_iEquations = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over_iEquations);
		}
		else {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		};
		printf("\n [ Beta_Jacobi %1.10E Beta_Move %1.10E L2eps(old) %1.10E ] ", beta_nJ, beta_nR, L2eps);

		cudaMemcpy(p_Ap_e, p_T, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		kernelAdd_to_T_lc << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_T, beta_nR, p_slot1i, beta_nJ, p_slot2i);
		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

		SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressors + (REGRESSORS - 1)*NUMVERTICES, p_T, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS Subtract");
		
	};

	// It was surprisingly ineffective and chooses a low coefficient on Jacobi
	// Therefore let's go again with Jacobi reprised.


	// Test effect of additions:

	// . Create epsilon for p_Tn etc
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_use->p_info,
		pX_use->p_izNeigh_vert,
		pX_use->p_szPBCneigh_vert,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_cc,
		pX_use->p_n_major,
		p_T,
		pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa,
		p_nu,
		NT_addition_rates_d_temp,
		pX_use->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		iSpecies);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate neut");

	cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES);

	kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_T,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon,
		0,
		p_boolarray2,
		p_boolarray_block,
		bUseMask,
		iSpecies
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 1 and Jacobi");

	kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_epsilon, p_temp1);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	f64 SS_n = 0.0;
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		SS_n += p_temphost1[iTile];
	}

	if (bUseMask) {
		long over_iEquations;
		if (iSpecies == 0) over_iEquations = over_iEquations_n;
		if (iSpecies == 1) over_iEquations = over_iEquations_i;
		if (iSpecies == 2) over_iEquations = over_iEquations_e;
		L2eps = sqrt(SS_n * over_iEquations);
	}
	else {
		L2eps = sqrt(SS_n / (real)NUMVERTICES);
	};

	printf("L2eps %1.12E [after seed regression]\n", L2eps);



	// It was surprisingly ineffective and chooses a low coefficient on Jacobi
	// Let's go again with Jacobi reprised.
}


/*
void RunBackwardJLSForHeat_volleys(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESH 0.25

	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	// seed: just set T to T_k.
	cudaMemcpy(p_T, p_T_k, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	printf("\nJRLS for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	int iIteration = 0;
	do {
		printf("\nITERATION %d \n\n", iIteration);
		if ((iIteration >= 500) && (iIteration % 100 == 0))
		{
	 		// draw a graph:

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(cuSyst_host.p_iVolley, pX_use->p_iVolley, sizeof(char)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//	cudaMemcpy(p_temphost2, p_regressor_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_temphost1[iVertex];
				pdata->temp.y = p_temphost1[iVertex];

				if (p_temphost1[iVertex] > 1.0e-15) {
					printf("iVertex %d epsilon %1.10E iVolley %d ",
						iVertex, p_temphost1[iVertex],
						cuSyst_host.p_iVolley[iVertex]);
//
					cudaMemcpy(&tempf64vec4, &(p_d_eps_by_dbetaJ_e_x4[iVertex]), sizeof(f64_vec4), cudaMemcpyDeviceToHost);
					printf("deps/dJ %1.6E %1.6E %1.6E %1.6E ", tempf64vec4.x[0], tempf64vec4.x[1], tempf64vec4.x[2], tempf64vec4.x[3]);
//
					cudaMemcpy(&tempf64, &(p_Jacobi_e[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Jac %1.8E\n", tempf64);
				};
				//if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
				//{
					
			//	} else {
			//		pdata->temp.x = 0.0;
			//		pdata->temp.y = 0.0;
			//	}
				++pVertex;
				++pdata;
			}

			Graph[0].DrawSurface("epsilon",
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);


			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			printf("done graph");
		}
		// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
		// 3. Calculate Jacobi: for each point, Jacobi = eps/(deps/dT)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				hsub,
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_T, // using vert indices
				pX_use->p_T_minor + BEGINNING_OF_CENTRAL, // T_k+1/2 or T_k
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		 
		// Better if we would just load in epsilon and create Jacobi? No coeffself array, simpler.
		// Careful: if we find coeffself in d/dt NT then eps <- -(h/N) d/dtNT
		// so we need N as well.
		kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				hsub,
				pX_use->p_info, // minor
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_T, // using vert indices
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				pX_use->p_AreaMajor,

				p_coeffself_n,
				p_coeffself_i,
				p_coeffself_e, // what exactly it calculates?
				1.0); // used for Jacobi
		Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

		// Splitting up routines will probably turn out better although it's tempting to combine.

		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor);
		kernelCreateEpsilonAndJacobi_Heat << <numTilesMajor, threadsPerTileMajor >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_T,
			p_T_k,
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e,

			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e,
			p_bFailed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");



		// This bit is not yet modified per changing to use derivatives at sides.
		// Let's roll back to a JRLS -- simple?
		//

		//memset(d_eps_by_dx_neigh_n,0,sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(d_eps_by_dx_neigh_i, 0, sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(d_eps_by_dx_neigh_e, 0, sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(p_Effect_self_n, 0, sizeof(f64)*NUMVERTICES);
		//memset(p_Effect_self_i, 0, sizeof(f64)*NUMVERTICES);
		//memset(p_Effect_self_e, 0, sizeof(f64)*NUMVERTICES);
		//                   
		//kernelCalculateArray_ROCwrt_my_neighbours << <numTilesMajor, threadsPerTileMajor >> >(
		//	hsub,
		//	pX_use->p_info,
		//	pX_use->p_izNeigh_vert,
		//	pX_use->p_szPBCneigh_vert,
		//	pX_use->p_izTri_vert,
		//	pX_use->p_szPBCtri_vert,
		//	pX_use->p_cc,
		//	pX_use->p_n_major,
	 //	 	pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		//	p_kappa_n,
	 //		p_kappa_i,
		//	p_kappa_e,
	 //		p_nu_i,
		//	p_nu_e,
		//	pX_use->p_AreaMajor,
		//	  
		//	// Output:
		//	d_eps_by_dx_neigh_n, // save an array of MAXNEIGH f64 values at this location
		//	d_eps_by_dx_neigh_i,
	 //		d_eps_by_dx_neigh_e,
		//	p_Effect_self_n,
		//	p_Effect_self_i,
		//	p_Effect_self_e
		//);
		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateArray_ROCwrt_my_neighbours");

		//kernelCalculateOptimalMove<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		//	pX_use->p_info + BEGINNING_OF_CENTRAL,
		//	d_eps_by_dx_neigh_n,
		//	d_eps_by_dx_neigh_i,
		//	d_eps_by_dx_neigh_e,
		//	p_Effect_self_n,
		//	p_Effect_self_i,
		//	p_Effect_self_e,
		//	pX_use->p_izNeigh_vert,

		//	p_epsilon_n,
		//	p_epsilon_i,
		//	p_epsilon_e,
		//	// output:
		//	p_regressor_n,
		//	p_regressor_i,
		//	p_regressor_e
		//);
		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateOptimalMove");
		//
		//


		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = T - (T_k + h dT/dt)
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.
		
		cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);


		
		cudaMemset(p_d_eps_by_dbetaJ_n_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaJ_i_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaJ_e_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_n_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_i_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_e_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		
		kernelCalculateROCepsWRTregressorT_volleys << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			hsub,
			pX_use->p_info, // THIS WAS USED OK
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major, // got this
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,

			pX_use->p_AreaMajor, // got this -> N, Nn
			pX_use->p_iVolley,

			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e,

			p_d_eps_by_dbetaJ_n_x4,
			p_d_eps_by_dbetaJ_i_x4,
			p_d_eps_by_dbetaJ_e_x4  // 4 dimensional

			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");


		kernelCalculateROCepsWRTregressorT_volleys << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			hsub,
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major, // got this
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,

			pX_use->p_AreaMajor, // got this -> N, Nn
			pX_use->p_iVolley,

			p_epsilon_n,
			p_epsilon_i, // p_regressor_i,
			p_epsilon_e,
			p_d_eps_by_dbetaR_n_x4,
			p_d_eps_by_dbetaR_i_x4,
			p_d_eps_by_dbetaR_e_x4
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT Richardson");



		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_n,

			p_d_eps_by_dbetaJ_n_x4,
			p_d_eps_by_dbetaR_n_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
		// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_ROC_products.Invoke(8); // does set to zero
		
		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		//memset(&sum_ROC_products, 0, 8 * 8 * sizeof(f64));
		sum_eps_eps = 0.0;
		
		int i, j;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile*8*8+i*8+j];

		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for neutral [ L2eps %1.10E ] \n", L2eps);

		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_ROC_products.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		};

		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) { // otherwise just STOP !
			// 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.

			// Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_ROC_products.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:
				sum_ROC_products.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?
				// As long as we do not adjust kappa it is same, right? For each species.

				sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
			}
		} else {
			printf("beta === 0\n");
		};
		printf("\n beta: ");
		for (i = 0; i < 8; i++) 
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, 8 * sizeof(f64)));

		// ======================================================================================================================

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,

			p_d_eps_by_dbetaJ_i_x4,
			p_d_eps_by_dbetaR_i_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
									// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_products_i.Invoke(8); // does set to zero

		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		sum_eps_eps = 0.0;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_products_i.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile * 8 * 8 + i * 8 + j];

		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for ion [ L2eps %1.10E  ]\n ", L2eps);
		
		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_products_i.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		}; 
		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) { // otherwise just STOP !

		   // Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_products_i.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:
				sum_products_i.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?
				// As long as we do not adjust kappa it is same, right? For each species.
				sum_products_i.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
			}
		} else {
			printf("beta === 0\n");
		}
		printf("\n beta: ");
		for (i = 0; i < 8; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_i_c, beta, 8 * sizeof(f64)));


		// ======================================================================================================================

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_e,

			p_d_eps_by_dbetaJ_e_x4,
			p_d_eps_by_dbetaR_e_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
									// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands e");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		if (iIteration >= 500) {

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_Jacobi_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_regressor_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			cudaMemcpy(cuSyst_host.p_iVolley, pX_use->p_iVolley, sizeof(char)*NUMVERTICES, cudaMemcpyDeviceToHost);
			FILE * filedbg = fopen("debug1.txt", "w");
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				cudaMemcpy(&tempf64vec4, &(p_d_eps_by_dbetaJ_e_x4[iVertex]), sizeof(f64_vec4), cudaMemcpyDeviceToHost);

				fprintf(filedbg, "iVertex %d eps %1.12E iVolley %d Jacobi %1.9E opti %1.9E d_eps_by_dbeta_J0 %1.12E \n", iVertex, p_temphost1[iVertex], cuSyst_host.p_iVolley[iVertex],
					p_temphost2[iVertex], p_temphost3[iVertex], tempf64vec4.x[0]);
			};
			fclose(filedbg);
			printf("outputted to file\n");
		};

		sum_products_e.Invoke(8); // does set to zero

		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		sum_eps_eps = 0.0;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_products_e.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile * 8 * 8 + i * 8 + j];
		
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for elec [ L2eps %1.10E  ]\n ", L2eps);

		if (iIteration > 100) getch();

		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_products_e.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		};
		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) {

			// Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_products_e.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:

//				f64 storedbg[8][8];
//				memcpy(storedbg, sum_products_e.LU, sizeof(f64) * 8 * 8);
//
//				FILE * fpdebug = fopen("matrix_e_result.txt", "w");
//				fprintf(fpdebug, "\n");
//				for (i = 0; i < 8; i++)
//				{
//					for (j = 0; j < 8; j++)
//						fprintf(fpdebug, "%1.14E ", sum_products_e.LU[i][j]);
//
//					fprintf(fpdebug, "   |  %1.14E  \n", sum_eps_deps_by_dbeta_vector[i]);
//				};
				sum_products_e.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?

				// As long as we do not adjust kappa it is same, right? For each species.

				sum_products_e.LUSolve(sum_eps_deps_by_dbeta_vector, beta);

				// Compute test vector:
//				f64 result[8];
//				for (i = 0; i < 8; i++)
	//			{
	//				result[i] = 0.0;
	//				for (j = 0; j < 8; j++)
	//					result[i] += storedbg[i][j] * beta[j];
	//			}

		//		for (i = 0; i < 8; i++)
		//			fprintf(fpdebug, " beta %1.14E result %1.14E \n", beta[i], result[i]);
		//		fprintf(fpdebug, "\n");
		//		fclose(fpdebug); // Test
			};
		} else {
			printf("beta === 0\n");
		};

		printf("\n beta: ");
		for (i = 0; i < 8; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_e_c, beta, 8 * sizeof(f64)));

		kernelAddtoT_volleys << <numTilesMajor, threadsPerTileMajor >> > (
			p_T, pX_use->p_iVolley,
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_epsilon_n, p_epsilon_i, p_epsilon_e); // p_regressor_i

		/*
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_nJ = 0.0;
			beta_nR = 0.0;
		} else {
			beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		}
		printf("\n for neutral [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_nJ, beta_nR, L2eps_neut);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,
			p_d_eps_by_dbeta_i,
			p_d_eps_by_dbetaR_i,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_iJ = 0.0; beta_iR = 0.0;
		} else {
			beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		};
		L2eps_ion = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ION [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_iJ, beta_iR, L2eps_ion);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
			
			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			p_epsilon_e,
			p_d_eps_by_dbeta_e,
			p_d_eps_by_dbetaR_e,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_eJ = 0.0; beta_eR = 0.0;
		}
		else {
			beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		};
		L2eps_elec = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for Electron [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_eJ, beta_eR, L2eps_elec);
		
		*/



		/*

		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info,
			
				// WHOOPS

			p_epsilon_i,
			p_d_eps_by_dbeta_i,

			p_sum_eps_deps_by_dbeta,
			p_sum_depsbydbeta_sq,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1a");

		cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
			sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		if (sum_depsbydbeta_sq == 0.0) {
			beta_i = 1.0;
		} else {
			beta_i = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		};
		if (beta_i < UPLIFT_THRESH) {
			printf("beta_i %1.10E ", beta_i);
			beta_i = (UPLIFT_THRESH + beta_i) / (2.0 - UPLIFT_THRESH + beta_i); // Try to navigate space instead of getting stuck.
			printf("beta_i after uplift %1.10E \n", beta_i);
		}
		L2eps_ion = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ion [ %1.14E %1.14E ] ", beta_i, L2eps_ion);

		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info,

			p_epsilon_e,
			p_d_eps_by_dbeta_e,

			p_sum_eps_deps_by_dbeta,
			p_sum_depsbydbeta_sq,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1");

		cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
			sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		if (sum_depsbydbeta_sq == 0.0) {
			beta_e = 1.0;
		} else {
			beta_e = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		};
		if (beta_e < UPLIFT_THRESH) {
			printf("beta_e %1.10E ", beta_e);
			beta_e = (UPLIFT_THRESH + beta_e) / (2.0 - UPLIFT_THRESH + beta_e);
			printf("beta_e after uplift %1.10E\n", beta_e);
		}
		
		L2eps_elec = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for elec [ %1.14E %1.14E ] ", beta_e, L2eps_elec);
		*/
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		
/*
kernelAddtoT << <numTilesMajor, threadsPerTileMajor >> > (
			p_T, beta_nJ, beta_nR, beta_iJ, beta_iR, beta_eJ, beta_eR, 
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_epsilon_n, p_epsilon_i, p_epsilon_e);

		// For some reason, beta calculated was the opposite of the beta that was needed.
		// Don't know why... only explanation is that we are missing a - somewhere in deps/dbeta
		// That's probably what it is? Ought to verify.

		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___");
		*/
/*
		iIteration++;


		// Test that all temperatures will be > 0 and within 1% of putative value
		// WOULD IT NOT be better just to set them to their putative values? We don't know what heat flux that corresponds to?

		// cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor);
		//kernelTestEpsilon << <numTilesMajor, threadsPerTileMajor >> > (
		//	p_epsilon_n,
		//	p_epsilon_i,
		//	p_epsilon_e,
		//	p_T,
		//	p_bFailed // each thread can set to 1
		//	);
		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajor; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		} else {
			printf("bFailedTest false \n");
		}
	} while ((iIteration < NUM_BWD_ITERATIONS)// || (L2eps_elec > 1.0e-14) || (L2eps_ion > 1.0e-14) || (L2eps_neut > 1.0e-14)
			|| (bFailedTest));
		
}
*/

void RunBackwardJLSForViscosity(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use,
	v4 * p_initial_regressor, bool bHistory	)
// BE SURE ABOUT PARAMETER ORDER -- CHECK IT CHECK IT
{
	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.


	f64 beta_e, beta_i, beta_n;
	long iTile;

	// seed: just set T to T_k.
	// No --- assume we were sent the seed.
	cudaMemcpy(p_vie, p_vie_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	
	// oooh

	// JLS:

	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 beta, L2eps;

	// Do outside:
	//kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> >(
	//	pX_use->p_info,
	//	pX_use->p_n_minor,
	//	pX_use->p_T_minor,
	//	p_temp3,	//	p_temp4,	//	p_temp5,	//	p_temp1,	//	p_temp2,	//	p_temp6);
	//Call(cudaThreadSynchronize(), "cudaTS ita 1");

	int iIteration;

	kernelCalc_Matrices_for_Jacobi_Viscosity << < numTriTiles, threadsPerTileMinor >> >//SelfCoefficient
			(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, p_temp2, p_temp3, p_temp4,
			pX_use->p_B,
			pX_use->p_n_minor, // eps += -h/N * MAR; thus N features in self coefficient
			pX_use->p_AreaMinor,
			p_InvertedMatrix_i,
			p_InvertedMatrix_e
			); // don't forget +1 in self coefficient
	Call(cudaThreadSynchronize(), "cudaTS kernelCalc_Jacobi_for_Viscosity");
	
	iIteration = 0;
	bool bContinue = true;
	do 
	{
		// ***************************************************
		// Requires averaging of n,T to triangles first. & ita
		// ***************************************************

		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			p_vie,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion2, // just accumulates
			p_MAR_elec2,
			NT_addition_rates_d_temp, 
				// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");
		
		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_vie,
			p_vie_k,
			p_MAR_ion2, p_MAR_elec2,

			pX_use->p_n_minor,
			pX_use->p_AreaMinor,

			p_epsilon_xy, 
			p_epsilon_iz, 
			p_epsilon_ez,
			p_bFailed,
			p_Selectflag
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		int i;
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;

		if ((iIteration == 0) && (bHistory)) {
			kernelSplitIntoSeedRegressors << <numTilesMinor, threadsPerTileMinor >> >
				(	p_initial_regressor,
					p_vJacobi_i,
					p_vJacobi_e,
					p_epsilon_xy // use this to create 2nd regressor somehow.
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelSplitIntoSeedRegressors ");
		} else {

			kernelMultiply_Get_Jacobi_Visc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
				(
					pX_use->p_info,
					p_epsilon_xy, // input
					p_epsilon_iz, // input
					p_epsilon_ez, // input
					p_InvertedMatrix_i,
					p_InvertedMatrix_e,
					// output:
					p_vJacobi_i, // 3-vec array
					p_vJacobi_e  // 3-vec array	= InvertedMatrix epsilon
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");
		};
		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = v - (v_k + h [viscous effect] + h [known increment rate of v])
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.

		cudaMemset(p_d_epsxy_by_d_beta_i, 0, sizeof(f64_vec2)*NMINOR);
		cudaMemset(p_d_eps_iz_by_d_beta_i, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_d_eps_ez_by_d_beta_i, 0, sizeof(f64)*NMINOR);

		cudaMemset(p_d_epsxy_by_d_beta_e, 0, sizeof(f64_vec2)*NMINOR);
		cudaMemset(p_d_eps_iz_by_d_beta_e, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_d_eps_ez_by_d_beta_e, 0, sizeof(f64)*NMINOR);
		
		/*

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);

		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			
			p_regressor_1, // regressor 1 means add to vxy, viz, right?

			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion2, // just accumulates
			p_MAR_elec2,
			NT_addition_rates_d_temp,
			// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~ff");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_regressor_1, // vez=0
			zero_vie,
			p_MAR_ion2, p_MAR_elec2,

			pX_use->p_n_minor,
			pX_use->p_AreaMinor,

			p_d_epsilon_xy_dbeta1,
			p_d_epsilon_iz_dbeta1,
			p_d_epsilon_ez_dbeta1,
			b_Failed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon viscff");


		 
		*/

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		// Create suggested change from p_vJacobi_i

		kernelSet << <numTilesMinor, threadsPerTileMinor >> >(v4temparray, p_vJacobi_i, SPECIES_ION);
		Call(cudaThreadSynchronize(), "cudaTS kernelSet");

		// This is really dumb, we should split into more regressors.
		// But that means more calls to flow evaluation.

		// xy | z

		cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			v4temparray,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion3, // just accumulates
			p_MAR_elec3,
			NT_addition_rates_d_temp, // not used for anything --- hopefully??!
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v4temparray, // Jacobi regressor
			zero_vec4,
			p_MAR_ion3, 
			p_MAR_elec3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			// It affects all 4 errors.
			p_d_epsxy_by_d_beta_i,
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,
			p_bFailed, 
			p_Selectflag
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		kernelSet << <numTilesMinor, threadsPerTileMinor >> >(v4temparray, p_vJacobi_e, SPECIES_ELEC);
		Call(cudaThreadSynchronize(), "cudaTS kernelSet");


		cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			v4temparray,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion3,
			p_MAR_elec3,
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v4temparray, // Jacobi regressor
			zero_vec4,
			p_MAR_ion3,
			p_MAR_elec3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			// It affects all 4 errors.
			p_d_epsxy_by_d_beta_e,
			p_d_eps_iz_by_d_beta_e,
			p_d_eps_ez_by_d_beta_e,
			p_bFailed, // is it ok to junk this?
			p_Selectflag
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		//kernelCalculate_deps_WRT_beta_Visc << < numTriTiles, threadsPerTileMinor >> >(
		//	hsub,
		//	pX_use->p_info,
		//	pX_use->p_izTri_vert,
		//	pX_use->p_szPBCtri_vert,
		//	pX_use->p_izNeigh_TriMinor,
		//	pX_use->p_szPBC_triminor,

		//	p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		//	p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		//	p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		//	p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
		//	pX_use->p_B,

		//	pX_use->p_n_minor, // got this
		//	pX_use->p_AreaMinor, // got this -> N, Nn

		//	p_vJacobi_i, // 3-vec
		//	p_vJacobi_e, // 3-vec

		//	p_d_eps_by_d_beta_i,
		//	p_d_eps_by_d_beta_e
		//	);

		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");
		//

		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.


		kernelAccumulateSummands3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_xy,
			p_epsilon_iz,
			p_epsilon_ez,

			p_d_epsxy_by_d_beta_i,
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,
			p_d_epsxy_by_d_beta_e,
			p_d_eps_iz_by_d_beta_e,
			p_d_eps_ez_by_d_beta_e,
			
			// 6 outputs:
			p_sum_eps_deps_by_dbeta_i,
			p_sum_eps_deps_by_dbeta_e,
			p_sum_depsbydbeta_i_times_i,
			p_sum_depsbydbeta_e_times_e,
			p_sum_depsbydbeta_e_times_i,
			p_sum_eps_eps);

		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1aa");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_i_host, p_sum_eps_deps_by_dbeta_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_e_host, p_sum_eps_deps_by_dbeta_e, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_i_times_i_host, p_sum_depsbydbeta_i_times_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_e_times_e_host, p_sum_depsbydbeta_e_times_e, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_e_times_i_host, p_sum_depsbydbeta_e_times_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		
		f64 sum_eps_deps_by_dbeta_i = 0.0;
		f64 sum_eps_deps_by_dbeta_e = 0.0;
		f64 sum_depsbydbeta_i_times_i = 0.0;
		f64 sum_depsbydbeta_e_times_e = 0.0;
		f64 sum_depsbydbeta_e_times_i = 0.0;
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			sum_eps_deps_by_dbeta_i += p_sum_eps_deps_by_dbeta_i_host[iTile];
			sum_eps_deps_by_dbeta_e += p_sum_eps_deps_by_dbeta_e_host[iTile];
			sum_depsbydbeta_i_times_i += p_sum_depsbydbeta_i_times_i_host[iTile];
			sum_depsbydbeta_e_times_e += p_sum_depsbydbeta_e_times_e_host[iTile];
			sum_depsbydbeta_e_times_i += p_sum_depsbydbeta_e_times_i_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		
		if ((sum_eps_eps == 0.0) || ((sum_depsbydbeta_i_times_i*sum_depsbydbeta_e_times_e - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i) == 0.0))
			return;
		
		beta_i = -(sum_eps_deps_by_dbeta_i*sum_depsbydbeta_e_times_e - sum_eps_deps_by_dbeta_e*sum_depsbydbeta_e_times_i)/
				  (sum_depsbydbeta_i_times_i*sum_depsbydbeta_e_times_e - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i);
		beta_e = -(sum_eps_deps_by_dbeta_e*sum_depsbydbeta_i_times_i - sum_eps_deps_by_dbeta_i*sum_depsbydbeta_e_times_i)/
				  (sum_depsbydbeta_e_times_e*sum_depsbydbeta_i_times_i - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i);
		 
		L2eps = sqrt(sum_eps_eps / (real)NMINOR);
		if (iIteration % 10 == 0) printf("\nIteration %d visc: [ beta_i %1.14E beta_e %1.14E L2eps %1.14E ] ", iIteration, beta_i, beta_e, L2eps);
		
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAdd_to_v << <numTilesMinor, threadsPerTileMinor >> > (
			p_vie, beta_i, beta_e, p_vJacobi_i, p_vJacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS Addtov ___");
			
		
		 
		iIteration++;

	} while ((bContinue) && (iIteration < 1000));

	if (iIteration == 1000) {
		printf("\a");
		getch();
	}



	// Do after calling and recalc'ing MAR:

	//Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
	//	this->p_info,
	//	this->p_n_minor,
	//	this->p_tri_corner_index,
	//	this->p_AreaMajor, // populated?
	//	p_temp4);
	//Call(cudaThreadSynchronize(), "cudaTS Nsum");

	//kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
	//	this->p_info,
	//	this->p_izTri_vert,
	//	this->p_n_minor,
	//	this->p_AreaMajor,
	//	p_temp4,
	//	NT_addition_rates_d,
	//	NT_addition_tri_d
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS sum up heat 1");
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}


void RunBackwardR8LSForNeutralViscosity(f64_vec3 * p_v_n_k, f64_vec3 * p_v_n, f64 const hsub,
	cuSyst * pX_use
	//f64_vec3 * p_initial_regressor
	) {

	f64 beta[REGRESSORS];
	long iTile;
	f64_vec3 L2eps, Rsquared;
	// Function manages its historic seed move internally, as p_stored_move3.

	// What we are missing is a regression on last time.

	// Also sensible convergence criterion. 

	// Then could profile to see what is slowing things down so much.
	 

	// If it's heat or viscosity, with ita given we could form an lc of nearby points and it could well be faster.
	// It means some bus journeys to collect the doubles. Versus ... ?
	// Worth a try perhaps.
	// Especially for visc as it is complicated.


	// (soon add last move as first regressor)

	cudaMemcpy(p_v_n, p_v_n_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	 
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	long iMinor;
	int iIteration, i;
	f64_vec3 TSS, RSS;
	cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
	iIteration = 0;
	bool bContinue = true;

	// 1. Create residual epsilon for v_k
	cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_v_n,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_temp6, // ita
		p_temp5,
		p_MAR_neut2, // just accumulates
		NT_addition_rates_d_temp,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
	cudaMemset(p_epsilon_x, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_y, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_z, 0, sizeof(f64)*NMINOR);
	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
	kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_use->p_info,
		p_v_n,
		p_v_n_k,
		p_MAR_neut2,
		pX_use->p_n_minor,
		pX_use->p_AreaMinor,
		p_epsilon_x,
		p_epsilon_y,
		p_epsilon_z,
		p_bFailed,
		p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

	bContinue = false;
	cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
	if (i < numTilesMinor) bContinue = true;

	// Collect L2eps :
	RSS.x = 0.0;
	RSS.y = 0.0;
	RSS.z = 0.0;
	
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_x, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.x += p_SS_host[iTile];

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_y, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.y += p_SS_host[iTile];
	
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_z, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.z += p_SS_host[iTile];
	L2eps.x = sqrt(RSS.x / (real)NMINOR);
	L2eps.y = sqrt(RSS.y / (real)NMINOR);
	L2eps.z = sqrt(RSS.z / (real)NMINOR);

	printf("L2eps %1.8E %1.8E %1.8E\n", L2eps.x, L2eps.y, L2eps.z);

	if (L2eps.x + L2eps.y + L2eps.z == 0.0) goto labelNeutralBudgie;

	while(bContinue) {

		// 2. Create set of 7 or 8 regressors, starting with epsilon3 normalized,
		// and deps/dbeta for each one.
		// The 8th is usually either for the initial seed regressor (prev move) or comes from previous iteration

		CallMAC(cudaMemset(p_d_eps_by_d_beta_x_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		CallMAC(cudaMemset(p_d_eps_by_d_beta_y_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		CallMAC(cudaMemset(p_d_eps_by_d_beta_z_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		for (i = 0; i < REGRESSORS; i++)
		{
			// purpose of loop: define regressor & take d_eps_by_d_beta.

			// populate regressor i 
			// regressor 0 = p_epsilon3, normalized, since on iteration 0 this is simply h A vk

			if (i == 0) {
				AssembleVector3 <<<numTilesMinor, threadsPerTileMinor >>>(p_regressors3, p_epsilon_x,
					p_epsilon_y, p_epsilon_z);
				Call(cudaThreadSynchronize(), "AssembleVector3");
			} else {
				if ((i == REGRESSORS - 1) && ((iIteration > 0) || (iHistoryVN>0))) {
					cudaMemcpy(p_regressors3 + i*NMINOR, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
					// might as well used p_stored_move for both prev move and seed from previous call.
				} else {

					// Care to subtract the previous regressor to reduce colinearity? Yep.
					// d_eps was formed including (1-hA) regressor
					// so let's remove 1x regressor from it.
					SubtractVector3stuff << <numTilesMinor, threadsPerTileMinor >> >
						(p_regressors3 + i*NMINOR,
							p_d_eps_by_d_beta_x_ + (i - 1)*NMINOR,
							p_d_eps_by_d_beta_y_ + (i - 1)*NMINOR,
							p_d_eps_by_d_beta_z_ + (i - 1)*NMINOR,
							p_regressors3 + (i - 1)*NMINOR
							); // out.x = a.x-b.x
					Call(cudaThreadSynchronize(), "SubtractVector3stuff");
				};
			};
			// What???

			// Normalize regressor: divide by L2 norm.
			// ___________________________________________:
			kernelAccumulateSumOfSquares3 << <numTilesMinor, threadsPerTileMinor >> >
				(	p_regressors3 + i*NMINOR,
					p_tempvec3
					);
			Call(cudaThreadSynchronize(), "SS3");
			cudaMemcpy(p_tempvec3host, p_tempvec3, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
			f64_vec3 SS(0.0, 0.0, 0.0);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{
				SS += p_tempvec3host[iTile];
			}
			f64_vec3 L2regress;
			L2regress.x = sqrt(SS.x / (real)NMINOR);
			L2regress.y = sqrt(SS.y / (real)NMINOR);
			L2regress.z = sqrt(SS.z / (real)NMINOR);

		//	printf("got to here  -- L2regress %1.8E %1.8E %1.8E \n", L2regress.x, L2regress.y, L2regress.z);
			
			if (L2regress.x == 0.0) L2regress.x = 1.0;
			if (L2regress.y == 0.0) L2regress.y = 1.0;
			if (L2regress.z == 0.0) L2regress.z = 1.0;

			ScaleVector3 << <numTilesMinor, threadsPerTileMinor >> > (p_regressors3 + i*NMINOR, 
				1.0/L2regress.x, 1.0 / L2regress.y, 1.0 / L2regress.z);
			Call(cudaThreadSynchronize(), "ScaleVector3");

			// ============================================================
			cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
			kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_regressors3 + i*NMINOR,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,
				p_temp6, // ita
				p_temp5,
				p_MAR_neut2, // just accumulates
				NT_addition_rates_d_temp,
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
			kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_use->p_info,
				p_regressors3 + i*NMINOR,
				zero_vec3,
				p_MAR_neut2,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				p_d_eps_by_d_beta_x_ + i*NMINOR, 
				p_d_eps_by_d_beta_y_ + i*NMINOR, 
				p_d_eps_by_d_beta_z_ + i*NMINOR,  // This is assigning values as if epsilon can be changed in ins.
				0,
				p_SelectflagNeut
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon neut visc");
			
		};

		// Now on to determine coefficients.
		// We can solve for 3 separate sets of coefficients since each dimension was independent.
		// Then we add different amounts for x,y,z.
		// Then we test the overall move to see if the dot product with the consequent forward direction is positive.
		// Though given that dimensions are independent, there is certainly a case to treat the criterion independently.
		// ==============================================================================================================
		f64 * p_epsilon_, * p_deps_by_dbeta;
		int iDim;

		cudaMemset(p_stored_move3, 0, sizeof(f64_vec3)*NMINOR);
		for (iDim = 0; iDim < 3; iDim++)
		{
			if (iDim == 0)			{
				p_epsilon_ = p_epsilon_x;
				p_deps_by_dbeta = p_d_eps_by_d_beta_x_;
			};
			if (iDim == 1) {
				p_epsilon_ = p_epsilon_y;
				p_deps_by_dbeta = p_d_eps_by_d_beta_y_;
			};
			if (iDim == 2) {
				p_epsilon_ = p_epsilon_z;
				p_deps_by_dbeta = p_d_eps_by_d_beta_z_;
			};

			// Neue plan:
			// when we get here we have deps/dbeta for each dimension as separate arrays.
			// Only regressors take the form f64_vec3.
			
			cudaMemset(p_eps_against_d_eps, 0, sizeof(f64)*REGRESSORS * numTilesMinor);
			cudaMemset(p_sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS* numTilesMinor);
			// DIMENSION!!

			// We are going to want 8 beta for each dimension.

			// It's too much to store in shared memory so split product matrix sum into 3 calls:
			
			// MUST ENSURE THAT things are 0 away from domain cells!!
			kernelAccumulateSummandsNeutVisc2 << <numTilesMinor, threadsPerTileMinor/4 >> > (
				p_epsilon_,
				p_deps_by_dbeta,      // data for this dimension, 8 regressors
				p_eps_against_d_eps,  // 1x8 for each tile
				p_sum_product_matrix // this is 8x8 for each tile, for each dimension
				);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neutvisc2");
			
			cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_d_eps, sizeof(f64) * 8 * numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(f64) * 64 * numTilesMinor, cudaMemcpyDeviceToHost);

			// DIMENSION!!
			f64 eps_deps[REGRESSORS];
			f64 sum_product_matrix[REGRESSORS*REGRESSORS];
			memset(eps_deps, 0, sizeof(f64) * REGRESSORS);
			memset(sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{
				for (i = 0; i < REGRESSORS; i++)
					eps_deps[i] -= p_eps_against_d_eps_host[iTile *REGRESSORS + i];

				// Note minus, to get beta already negated.

				for (i = 0; i < REGRESSORS*REGRESSORS; i++)
					sum_product_matrix[i] += p_sum_product_matrix_host[iTile *REGRESSORS*REGRESSORS + i];
			};

			// Sometimes get 0 row and column at the end.
			// Try this:
			if ((sum_product_matrix[REGRESSORS*REGRESSORS - 1] == 0.0) && (eps_deps[REGRESSORS-1] == 0.0)) 
				sum_product_matrix[REGRESSORS*REGRESSORS - 1] = 1.0;
			if ((sum_product_matrix[REGRESSORS*(REGRESSORS-1) - 1] == 0.0) && (eps_deps[REGRESSORS - 2] == 0.0))
				sum_product_matrix[REGRESSORS*(REGRESSORS-1) - 1] = 1.0;
			if (!GlobalSuppressSuccessVerbosity)
			{
				printf("\n");
				for (i = 0; i < REGRESSORS; i++) {
					for (int j = 0; j < REGRESSORS; j++)
						printf("neut %1.9E ", sum_product_matrix[i*REGRESSORS + j]);
					printf(" |  %1.9E \n", eps_deps[i]);
				}
				printf("\n");
			};

			// Note that file 1041-Krylov.pdf claims that simple factorization for LS is an
			// unstable method and that is why the complications of GMRES are needed.
			// now we need the LAPACKE dgesv code to solve the 8x8 linear equation.
			f64 storeRHS[REGRESSORS];
			f64 storemat[REGRESSORS*REGRESSORS];
			memcpy(storeRHS, eps_deps, sizeof(f64)*REGRESSORS);
			memcpy(storemat, sum_product_matrix, sizeof(f64)*REGRESSORS*REGRESSORS);
						
			Matrix_real matLU;
			matLU.Invoke(REGRESSORS);
			for (i = 0; i < REGRESSORS; i++)
				for (int j = 0; j < REGRESSORS; j++)
					matLU.LU[i][j] = sum_product_matrix[i*REGRESSORS + j];
			matLU.LUdecomp();
			matLU.LUSolve(eps_deps, beta);
		
			printf("\nbeta: ");
			for (i = 0; i < REGRESSORS; i++)
				printf(" %1.8E ", beta[i]);
			printf("\n");

			CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));
			
			/*
#ifdef LAPACKE
			lapack_int ipiv[REGRESSORS];
			lapack_int Nrows = REGRESSORS,
				Ncols = REGRESSORS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = REGRESSORS, info;

			//	printf("LAPACKE_dgesv Results\n");
			// Solve the equations A*X = B 
			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 
				Nrows, 1, sum_product_matrix, Ncols, ipiv, eps_deps, Nrhscols);
			// Check for the exact singularity :
			
			if (info > 0) {
				//	printf("The diagonal element of the triangular factor of A,\n");
				//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed Neutral visc.\n");
				printf("press c\n");
				while (getch() != 'c');
			} else {
				if (info == 0) {
					memcpy(beta, eps_deps, REGRESSORS * sizeof(f64)); // that's where LAPACKE saves the result apparently.
				};
			}
#endif
*/

			// Now beta[8] is the set of coefficients for x
			// Move to the new value: add lc of regressors to proposal vector.
			
			CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

			printf("Iteration %d nvisc: [ beta ", iIteration);
			for (i = 0; i < REGRESSORS; i++) printf("%1.3E ", beta[i]);
			printf(" ]\n");

			AddLCtoVector3component << <numTilesMinor, threadsPerTileMinor >> > (p_v_n, p_regressors3, iDim,
				p_stored_move3);
			Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector3component");
		}; // next iDim
		
		// Finally, test whether the new values satisfy 'reasonable' criteria:
	
		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_temp6, // ita
			p_temp5,
			p_MAR_neut2, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
		cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			p_v_n,
			p_v_n_k,
			p_MAR_neut2,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_epsilon_x, // array of 3-vectors  -- overwrite
			p_epsilon_y, 
			p_epsilon_z,
			p_bFailed ,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");
		
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism

		// Collect L2eps and R^2:
		TSS = RSS;
		RSS.x = 0.0;
		RSS.y = 0.0;
		RSS.z = 0.0;
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_x, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.x += p_SS_host[iTile];		
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_y, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.y += p_SS_host[iTile];
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_z, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.z += p_SS_host[iTile];
		L2eps.x = sqrt(RSS.x / (real)NMINOR);
		L2eps.y = sqrt(RSS.y / (real)NMINOR);
		L2eps.z = sqrt(RSS.z / (real)NMINOR);

		// What is R^2? We should like to report it.
		Rsquared.x = (TSS.x - RSS.x) / TSS.x;
		Rsquared.y = (TSS.y - RSS.y) / TSS.y;
		Rsquared.z = (TSS.z - RSS.z) / TSS.z;
		
		Rsquared.x *= 100.0; Rsquared.y *= 100.0; Rsquared.z *= 100.0;
		printf("L2eps xyz %1.8E %1.8E %1.8E R^2 %2.3f%% %2.3f%% %2.3f%% bCont: %d\n",
			L2eps.x, L2eps.y, L2eps.z, Rsquared.x, Rsquared.y, Rsquared.z, (bContinue?1:0));
		/*

		1. We are getting that it can get worse. But I do not see MK 1 Eyeball what is wrong with the above code, so we need to see whether change in eps is as predicted.
		Is matrix solve same as on spreadsheet?
		 
		It could be we are operating below precision. 1e-9 on 1e+5 ? Yep sounds like it.
		What is convergence criterion?
			...

*/


		// Just to be clear, what do we mean by 'move direction', we're clear move up to here is proposal-Tk
		// hA is the direction from x_k+1 .. so why are we not just asking for epsilon > 0
		// There is more to this than meets the eye.
		
		// We would also like to do this above. Is there a cleverer way to rearrange code?
						
		// 2. A reasonable criterion for proximity to a sensible value. It only has to be within say 1e-4 relative.

		// Ideally would split into 3 loops for xyz but it's only neutral viscosity.
		// For the others the pattern has to be that we take eps.deps as a dot product over all eps including all 3 dimensions
		// So actually simpler.
		
		iIteration++;
	} // wend bContinue

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
labelNeutralBudgie:

	// Save move for next time:
	iHistoryVN++;
	SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, p_v_n, p_v_n_k);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");
	

}

void RunBackwardR8LSForViscosity(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use)
// BE SURE ABOUT PARAMETER ORDER -- CHECK IT CHECK IT
{
	// ***************************************************
	// Requires averaging of n,T to triangles first. & ita
	// ***************************************************

	static bool bHistory = false;

	f64 beta[REGRESSORS];
	f64 beta_e, beta_i, beta_n;
	long iTile;
	cudaMemcpy(p_vie, p_vie_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	long iMinor;
	f64 L2eps;
	int i;
	f64 TSS_xy, TSS_iz, TSS_ez, RSS_xy, RSS_iz, RSS_ez;
	f64 Rsquared_xy, Rsquared_iz, Rsquared_ez;

	cudaMemset(zero_vec4, 0, sizeof(v4)*NMINOR);
	int iIteration = 0;
	bool bContinue = true;

	// We'll go with 1 regressor vector x 8 ...
	// Complicated otherwise.		
	cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

		pX_use->p_info,
		p_vie,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,

		p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

		pX_use->p_B,
		p_MAR_ion2, // just accumulates
		p_MAR_elec2,
		NT_addition_rates_d_temp,
		// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");


	// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
	kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
		// eps = v - (v_k +- h [viscous effect])
		// x = -eps/coeffself
		hsub,
		pX_use->p_info,
		p_vie,
		p_vie_k,
		p_MAR_ion2, p_MAR_elec2,

		pX_use->p_n_minor,
		pX_use->p_AreaMinor,

		p_epsilon_xy,
		p_epsilon_iz,
		p_epsilon_ez,
		p_bFailed,
		p_Selectflag
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

	bContinue = false;
	cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
	if (i < numTilesMinor) bContinue = true;

	// Collect L2:
	RSS_xy = 0.0;
	RSS_iz = 0.0;
	RSS_ez = 0.0;

	kernelAccumulateSumOfSquares2vec << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_xy, p_SS);
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_xy += p_SS_host[iTile];
	f64 L2eps_xy = sqrt(RSS_xy / (real)NMINOR);
	printf("L2eps xy %1.8E ", L2eps_xy);

	if (L2eps_xy == 0.0) goto labelBudgerigar;

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_iz, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_iz += p_SS_host[iTile];
	f64 L2eps_iz = sqrt(RSS_iz / (real)NMINOR);
	printf("iz %1.8E ", L2eps_iz);

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_ez, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_ez += p_SS_host[iTile];
	f64 L2eps_ez = sqrt(RSS_ez / (real)NMINOR);
	printf("ez %1.8E \n", L2eps_ez);
    
	//// Debug: find maximum epsilons.
	//f64 maxio = 0.0;
	//int iMaxx = -1;
	//cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_tempvec2_host[iMinor].dot(p_tempvec2_host[iMinor]) > maxio) {
	//		iMaxx = iMinor;
	//		maxio = p_tempvec2_host[iMinor].dot(p_tempvec2_host[iMinor]);
	//	};
	//printf("eps_xy : maxio %1.8E at %d\n", sqrt(maxio), iMaxx);
	//
	//maxio = 0.0;
	//int iMaxiz = -1;
	//cudaMemcpy(p_temphost6, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_temphost6[iMinor] * p_temphost6[iMinor] > maxio) {
	//		iMaxiz = iMinor;
	//		maxio = p_temphost6[iMinor] * p_temphost6[iMinor];
	//	};
	//printf("eps_iz : maxio %1.8E at %d\n", sqrt(maxio), iMaxiz);
	//
	//maxio = 0.0;
	//int iMaxez = -1;
	//cudaMemcpy(p_temphost6, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_temphost6[iMinor] * p_temphost6[iMinor] > maxio) {
	//		iMaxez = iMinor;
	//		maxio = p_temphost6[iMinor] * p_temphost6[iMinor];
	//	};
	//printf("eps_ez : maxio %1.8E at %d\n", sqrt(maxio), iMaxez);
		
	f64 oldRSS = 0.0;

	// set epsilon vectors again at end of loop.		
	while (bContinue) {

		// 2. Create set of 7 or 8 regressors, starting with epsilon3 normalized,
		// and deps/dbeta for each one.
		// The 8th is usually either for the initial seed regressor (prev move) or comes from previous iteration

		cudaMemset(p_d_epsxy_by_d_beta_i, 0, sizeof(f64_vec2)*NMINOR*REGRESSORS);
		cudaMemset(p_d_eps_iz_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS);
		cudaMemset(p_d_eps_ez_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS);

		for (i = 0; i < REGRESSORS; i++)
		{
			if (i == 0) {
				AssembleVector4 << <numTilesMinor, threadsPerTileMinor >> > (
					p_regressors4,
					p_epsilon_xy,
					p_epsilon_iz, p_epsilon_ez);
				Call(cudaThreadSynchronize(), "AssembleVector4");
			}
			else {
				if ((i == REGRESSORS - 1) && ((iIteration > 0) || (bHistory))) {
					cudaMemcpy(p_regressors4 + i*NMINOR, p_stored_move4, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
					// might as well used p_stored_move for both prev move and seed from previous call.
				}
				else {

					// Care to subtract the previous regressor to reduce colinearity? Yep.
					// d_eps was formed including (1-hA) regressor
					// so let's remove 1x regressor from it.
					SubtractVector4stuff << <numTilesMinor, threadsPerTileMinor >> >
						(p_regressors4 + i*NMINOR,
							p_d_epsxy_by_d_beta_i + (i - 1)*NMINOR,
							p_d_eps_iz_by_d_beta_i + (i - 1)*NMINOR,
							p_d_eps_ez_by_d_beta_i + (i - 1)*NMINOR,
							p_regressors4 + (i - 1)*NMINOR
							); // out.x = a.x-b.x
					Call(cudaThreadSynchronize(), "SubtractVector4stuff");
				};
			};

			// Normalize regressor: divide by L2 norm.
			// ___________________________________________:
			kernelAccumulateSumOfSquares_4 << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors4 + i*NMINOR,
					p_SS
					);
			Call(cudaThreadSynchronize(), "SS4");

			f64 RSS = 0.0;
			cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++) {
				RSS += p_SS_host[iTile];
			}

			// It's one regressor so we cannot scale different dimensions differently.

			f64 L2regress = sqrt(RSS / (real)NMINOR);
			if (L2regress == 0.0) L2regress = 1.0;

			ScaleVector4 << <numTilesMinor, threadsPerTileMinor >> > (
				p_regressors4 + i*NMINOR, 1.0 / L2regress);
			Call(cudaThreadSynchronize(), "ScaleVector4");

			// ============================================================
			// Now calculate deps/dbeta for this regressor
			cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
			cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
			kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

				pX_use->p_info,
				p_regressors4 + i*NMINOR,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
				p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
				p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
				p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

				pX_use->p_B,
				p_MAR_ion2, // just accumulates
				p_MAR_elec2,
				NT_addition_rates_d_temp,
				// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

			// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
			kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				// x = -eps/coeffself
				hsub,
				pX_use->p_info,
				p_regressors4 + i*NMINOR,
				zero_vec4,
				p_MAR_ion2, p_MAR_elec2,

				pX_use->p_n_minor,
				pX_use->p_AreaMinor,

				p_d_epsxy_by_d_beta_i + i*NMINOR,
				p_d_eps_iz_by_d_beta_i + i*NMINOR,
				p_d_eps_ez_by_d_beta_i + i*NMINOR,
				0,
				p_Selectflag		
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");
		}; // next i

		cudaMemset(p_eps_against_deps, 0, sizeof(f64)*REGRESSORS * numTilesMinor);
		cudaMemset(p_sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS* numTilesMinor);
		kernelAccumulateSummandsVisc << <numTilesMinor, threadsPerTileMinor / 4 >> > (
			p_epsilon_xy, // 
			p_epsilon_iz,
			p_epsilon_ez,
			p_d_epsxy_by_d_beta_i, // f64_vec2
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,

			p_eps_against_d_eps,  // 1x8 for each tile
			p_sum_product_matrix // this is 8x8 for each tile
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands visc2");

		cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_d_eps, sizeof(f64) * REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(f64) * REGRESSORS*REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);

		f64 eps_deps[REGRESSORS];
		f64 sum_product_matrix[REGRESSORS*REGRESSORS];
		memset(eps_deps, 0, sizeof(f64) * REGRESSORS);
		memset(sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			for (i = 0; i < REGRESSORS; i++)
				eps_deps[i] -= p_eps_against_d_eps_host[iTile *REGRESSORS + i];

			// Note minus, to get beta already negated.

			for (i = 0; i < REGRESSORS*REGRESSORS; i++)
				sum_product_matrix[i] += p_sum_product_matrix_host[iTile *REGRESSORS*REGRESSORS + i];
		};
		// Try this:
		if ((sum_product_matrix[REGRESSORS*REGRESSORS - 1] == 0.0) && (eps_deps[REGRESSORS - 1] == 0.0))
			sum_product_matrix[REGRESSORS*REGRESSORS - 1] = 1.0;
		if ((sum_product_matrix[REGRESSORS*(REGRESSORS - 1) - 1] == 0.0) && (eps_deps[REGRESSORS - 2] == 0.0))
			sum_product_matrix[REGRESSORS*(REGRESSORS - 1) - 1] = 1.0;

		if (0)//!GlobalSuppressSuccessVerbosity)
		{
			printf("\n");
			for (i = 0; i < REGRESSORS; i++) {
				for (int j = 0; j < REGRESSORS; j++)
					printf("ei %1.9E ", sum_product_matrix[i*REGRESSORS + j]);
				printf(" |  %1.9E \n", eps_deps[i]);
			}
			printf("\n");
		}
		// Note that file 1041-Krylov.pdf claims that simple factorization for LS is an
		// unstable method and that is why the complications of GMRES are needed.
		
		// now we need the LAPACKE dgesv code to solve the 8x8 linear equation.
		f64 storeRHS[REGRESSORS];
		f64 storemat[REGRESSORS*REGRESSORS];
		memcpy(storeRHS, eps_deps, sizeof(f64)*REGRESSORS);
		memcpy(storemat, sum_product_matrix, sizeof(f64)*REGRESSORS*REGRESSORS);

		Matrix_real matLU;
		matLU.Invoke(REGRESSORS);
		for (i = 0; i < REGRESSORS; i++)
			for (int j = 0; j < REGRESSORS; j++)
				matLU.LU[i][j] = sum_product_matrix[i*REGRESSORS + j];
		matLU.LUdecomp();
		matLU.LUSolve(eps_deps, beta);

		printf("\nbeta: ");
		for (i = 0; i < REGRESSORS; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n");

		/*

#ifdef LAPACKE
		lapack_int ipiv[REGRESSORS];
		lapack_int Nrows = REGRESSORS,
			Ncols = REGRESSORS,  // lda
			Nrhscols = 1, // ldb
			Nrhsrows = REGRESSORS, info;

	//	for (i = 0; i < REGRESSORS; i++) {
	//		for (int j = 0; j < REGRESSORS; j++)
	//			printf("%1.8E ", sum_product_matrix[i*REGRESSORS + j]);
	//		printf(" ] [ %1.8E ]\n", eps_deps[i]);
	//	};
		


		//	printf("LAPACKE_dgesv Results\n");
		// Solve the equations A*X = B 
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
			Nrows, 1, sum_product_matrix, Ncols, ipiv, eps_deps, Nrhscols);
		// Check for the exact singularity :

		if (info > 0) {
			printf("The diagonal element of the triangular factor of A,\n");
			printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
			printf("the solution could not be computed.\n");
			printf("press c\n");
			while (getch() != 'c');
		}
		else {
			if (info == 0) {
				memcpy(beta, eps_deps, REGRESSORS * sizeof(f64)); // that's where LAPACKE saves the result apparently.
			};
		}
#endif
*/


		// OK this is a bit hmmm.
		// What I think we need to do is consider each dimensional move separately. 
		// This does mean more runs of evaluating the derivative.
		// Also let's consider carefully what R7 means in terms of .. Taylor series?


		// Debug:
		//printf("a = dummy1, s = dummy2, d = dummy3; other = continue\n");
		//char o = getch();
		//if ((o == 'a') || (o == 's') || (o == 'd') || (o == 'f') || (o == 'g') || (o == 'h') || (o == 'j') || (o == 'k') ) 
		//{
		//	memset(beta,0,sizeof(f64)*REGRESSORS);

		//	if (o == 'a') beta[0] = 100.0;
		//	if (o == 's') beta[1] = 100.0;
		//	if (o == 'd') beta[2] = 100.0;
		//	if (o == 'f') beta[3] = 100.0;
		//	if (o == 'g') beta[4] = 100.0;
		//	if (o == 'h') beta[5] = 100.0;
		//	if (o == 'j') beta[6] = 100.0;
		//	if (o == 'k') beta[7] = 100.0;
		//}
		//

		// Now beta[8] is the set of coefficients for x
		// Move to the new value: add lc of regressors to proposal vector.

		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

		printf("Iteration %d visc: [ beta ", iIteration);
		for (i = 0; i < REGRESSORS; i++) printf("%1.3E ", beta[i]);
		printf(" ]\n");
		
		// DEBUG:
	//	cudaMemcpy(p_tempvec4_host, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxx, p_tempvec4_host[iMaxx].vxy.x,
	//		p_tempvec4_host[iMaxx].vxy.y, p_tempvec4_host[iMaxx].viz, p_tempvec4_host[iMaxx].vez);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxiz, p_tempvec4_host[iMaxiz].vxy.x,
	//		p_tempvec4_host[iMaxiz].vxy.y, p_tempvec4_host[iMaxiz].viz, p_tempvec4_host[iMaxiz].vez);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxez, p_tempvec4_host[iMaxez].vxy.x,
	//		p_tempvec4_host[iMaxez].vxy.y, p_tempvec4_host[iMaxez].viz, p_tempvec4_host[iMaxez].vez);
	
		AddLCtoVector4component << <numTilesMinor, threadsPerTileMinor >> > 
			(p_vie, p_regressors4, p_stored_move4);
		Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector3component");

		// DEBUG: 
	//	cudaMemcpy(p_tempvec4_host, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxx, p_tempvec4_host[iMaxx].vxy.x,
	//		p_tempvec4_host[iMaxx].vxy.y, p_tempvec4_host[iMaxx].viz, p_tempvec4_host[iMaxx].vez);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxiz, p_tempvec4_host[iMaxiz].vxy.x,
	//		p_tempvec4_host[iMaxiz].vxy.y, p_tempvec4_host[iMaxiz].viz, p_tempvec4_host[iMaxiz].vez);
	//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxez, p_tempvec4_host[iMaxez].vxy.x,
	//		p_tempvec4_host[iMaxez].vxy.y, p_tempvec4_host[iMaxez].viz, p_tempvec4_host[iMaxez].vez);
		
		// ===========================================================================
		// Finally, test whether the new values satisfy 'reasonable' criteria:


		// Debug:
		//kernelCreatePredictionsDebug << <numTilesMinor, threadsPerTileMinor >> > (
		//	// eps = v - (v_k +- h [viscous effect])
		//	// x = -eps/coeffself
		//	hsub,
		//	pX_use->p_info,

		//	p_epsilon_xy,
		//	p_epsilon_iz,
		//	p_epsilon_ez,

		//	p_d_epsxy_by_d_beta_i, // f64_vec2
		//	p_d_eps_iz_by_d_beta_i,
		//	p_d_eps_ez_by_d_beta_i,

		//	p_GradAz,
		//	p_epsilon_i,
		//	p_epsilon_e
		//	);
		//Call(cudaThreadSynchronize(), "cudaTS Create predictions");




		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(
			pX_use->p_info,
			p_vie,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
			pX_use->p_B,
			p_MAR_ion2, // just accumulates
			p_MAR_elec2,
			NT_addition_rates_d_temp,
			// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_vie,
			p_vie_k,
			p_MAR_ion2, p_MAR_elec2,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_epsilon_xy,
			p_epsilon_iz,
			p_epsilon_ez,
			p_bFailed,
			p_Selectflag
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		// 
		//kernelCompare << <numTilesMinor, threadsPerTileMinor >> > (
		//	p_epsilon_xy,
		//	p_epsilon_iz,
		//	p_epsilon_ez,
		//	p_GradAz,
		//	p_epsilon_i,
		//	p_epsilon_e);
		//Call(cudaThreadSynchronize(), "cudaTS Compare");
		//printf("COMPARISON DONE");





		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism

		// Collect L2eps and R^2:
		TSS_xy = RSS_xy; TSS_iz = RSS_iz; TSS_ez = RSS_ez;
		RSS_xy = 0.0; RSS_iz = 0.0; RSS_ez = 0.0;
		kernelAccumulateSumOfSquares2vec << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_xy, p_SS);
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_xy += p_SS_host[iTile];
		L2eps_xy = sqrt(RSS_xy / (real)NMINOR);

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_iz, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_iz += p_SS_host[iTile];
		L2eps_iz = sqrt(RSS_iz / (real)NMINOR);

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_ez, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_ez += p_SS_host[iTile];
		L2eps_ez = sqrt(RSS_ez / (real)NMINOR);

		// What is R^2? We should like to report it.
		Rsquared_xy = (TSS_xy - RSS_xy) / TSS_xy;
		Rsquared_iz = (TSS_iz - RSS_iz) / TSS_iz;
		Rsquared_ez = (TSS_ez - RSS_ez) / TSS_ez;

		Rsquared_xy *= 100.0; Rsquared_iz *= 100.0; Rsquared_ez *= 100.0;

		if (L2eps_xy == 0.0) bContinue = false;
		printf("L2eps xy %1.8E iz %1.8E ez %1.8E TOTALRSS %1.10E \nR^2 %2.3f%% %2.3f%% %2.3f%% bCont: %d\n",
			L2eps_xy, L2eps_iz, L2eps_ez, 
			RSS_xy + RSS_iz + RSS_ez,			
			Rsquared_xy, Rsquared_iz, Rsquared_ez, (bContinue ? 1 : 0));

		// D E B U G :

		//if ((iIteration > 1) && (RSS_xy + RSS_iz + RSS_ez > oldRSS)) getch();
		oldRSS = RSS_xy + RSS_iz + RSS_ez;

		// yes it happens. L2 is not below precision either.
		



		//// Debug: find maximum epsilons:		
		//cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_xy : maxio %1.8E at %d\n", p_tempvec2_host[iMaxx].modulus(), iMaxx);
		//cudaMemcpy(p_temphost6, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_iz : maxio %1.8E at %d\n", p_temphost6[iMaxiz], iMaxiz);
		//cudaMemcpy(p_temphost6, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_ez : maxio %1.8E at %d\n", p_temphost6[iMaxez], iMaxez);


		iIteration++;
	};

labelBudgerigar:

	// Save move for next time:
	bHistory = true;
	SubtractVector4 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move4, p_vie, p_vie_k);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 4");
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

}


void RunBackwardJLSForNeutralViscosity(f64_vec3 * p_v_n_k, f64_vec3 * p_v_n, f64 const hsub,
	cuSyst * pX_use) {

	f64_vec3 beta;
	long iTile;
	f64_vec3 SS, L2eps3;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	//GlobalSuppressSuccessVerbosity = true;

	// (soon add last move as first regressor)

	cudaMemcpy(p_v_n, p_v_n_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	 
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 L2eps;
	int iIteration;
	f64_tens3 Y;
	f64_vec3 vec3;
	f64_tens3 inv;

	kernelCalc_Matrices_for_Jacobi_NeutralViscosity << < numTriTiles, threadsPerTileMinor >> >//SelfCoefficient
		(
			hsub,
			pX_use->p_info, // minor
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,
			pX_use->p_n_minor, // eps += -h/N * MAR; thus N features in self coefficient
			pX_use->p_AreaMinor,
			p_InvertedMatrix_n // Actually just scalings.
			); // don't forget +1 in self coefficient
	Call(cudaThreadSynchronize(), "cudaTS kernelCalc_Jacobi_for_Viscosity");


	// *. How to map to line between fwd and bwd?

	// Try : RK2 fwd is accepted if it is not overshooting. We could do fwd half-step.
	// Then we accept the RK2 fwd at positions where it's not overshooting
	// And then introduce bool to set where bwd solution will take place.
	// That has got to be quicker.


	iIteration = 0;
	bool bContinue = true;
	do
	{
		bool George;
		if (iIteration > 900) {
			George = true;
		} else {
			George = false;
		}
		bool * booladdress;
		Call(cudaGetSymbolAddress((void **)(&booladdress), bSwitch),
			"cudaGetSymbolAddress((void **)(&booladdress), bSwitch)");
		Call(cudaMemcpy(booladdress, &George, sizeof(bool), cudaMemcpyHostToDevice),
			"cudaMemcpy(longaddress, &George, sizeof(bool), cudaMemcpyHostToDevice)");
		//cudaMemcpyToSymbol(&bSwitch, &George, sizeof(bool));
		// Didn't work for some reason :/

		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		 
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_temp6, // ita
			p_temp5,
			p_MAR_neut2, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
		 
		cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_epsilon_x, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_epsilon_y, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_epsilon_z, 0, sizeof(f64)*NMINOR);
		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_v_n,
			p_v_n_k,
			p_MAR_neut2,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_epsilon_x,
			p_epsilon_y,
			p_epsilon_z,
			p_bFailed,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		int i;
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism


		GetMax<<<numTilesMinor, threadsPerTileMinor>>>(p_epsilon_y, p_longtemp, p_temp1);
		Call(cudaThreadSynchronize(), "cudaTS GetMax");
		cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		f64 maxi = 0.0;
		long iMax = -1;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			if (maxi < p_temphost1[iTile]) {
				maxi = p_temphost1[iTile];
				iMax = p_longtemphost[iTile];
			}
		};
		structural vimto;
		cudaMemcpy(&vimto, &(pX_use->p_info[iMax]), sizeof(structural), cudaMemcpyDeviceToHost);
		printf("imax %d max |eps y| = %1.14E flag %d pos %1.9E %1.9E\n", iMax, maxi, vimto.flag, vimto.pos.x, vimto.pos.y);


		cudaMemcpy(&tempf64, &(p_vJacobi_n[iMax].y), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("Jacobi[%d] = %1.10E \n", iMax, tempf64);
		cudaMemcpy(&tempf64, &(p_temp3_2[iMax].y), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("3rdreg[%d] = %1.10E \n", iMax, tempf64);


		GetMax << <numTilesMinor, threadsPerTileMinor >> >(p_epsilon_z, p_longtemp, p_temp1);
		Call(cudaThreadSynchronize(), "cudaTS GetMax");
		cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		maxi = 0.0;
		iMax = -1;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			if (maxi < p_temphost1[iTile]) {
				maxi = p_temphost1[iTile];
				iMax = p_longtemphost[iTile];
			}
		};
		cudaMemcpy(&vimto, &(pX_use->p_info[iMax]), sizeof(structural), cudaMemcpyDeviceToHost);
		printf("imax %d max |eps z| = %1.14E flag %d pos %1.9E %1.9E\n\n", iMax, maxi, vimto.flag, vimto.pos.x, vimto.pos.y);



		if ((iIteration > 0) || (iHistoryVN == 0))
		{
			// What we usually do:

			AssembleVector3 << <numTilesMinor, threadsPerTileMinor >> > (p_epsilon3, p_epsilon_x, p_epsilon_y, p_epsilon_z);
			Call(cudaThreadSynchronize(), "cudaTS AssembleVector3");
			// p_epsilon3 can be our first regressor.

			// if iIteration > 0 then p_temp3_2 is already set
			// otherwise:
			if (iIteration == 0) {
			
				// Try always using product of Richardson and Jacobi as 3rd regressor.

				kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
					(
						pX_use->p_info,
						p_epsilon3, // input
						p_InvertedMatrix_n,
						// output:
						p_vJacobi_n // 3-vec array
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

				SetProduct3 << <numTilesMinor, threadsPerTileMinor >> > (p_temp3_2, p_epsilon3, p_vJacobi_n);
				Call(cudaThreadSynchronize(), "cudaTS SetProduct3");
				SetConsoleTextAttribute(hConsole, 15);
			} else {

				// ================================
				// Typical case.
				// ================================

				// overwrite regressors:
				//if (iIteration % 2 == 1) {

				// Didn't help:
				//	// Overwrite 1st regressor with Jacobi:
				//	kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
				//		(
				//			pX_use->p_info,
				//			p_epsilon3, // input
				//			p_InvertedMatrix_n,
				//			// output:
				//			p_vJacobi_n // 3-vec array
				//			);
				//	Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

				//	cudaMemcpy(p_epsilon3, p_vJacobi_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
				//	SetConsoleTextAttribute(hConsole, 11);
				//} else {
				//	SetConsoleTextAttribute(hConsole, 15);
				//};

				// A eps: didn't help ?
				//cudaMemset(p_d_eps_by_d_beta_x_, 0, sizeof(f64)*NMINOR * 3);
				//cudaMemset(p_d_eps_by_d_beta_y_, 0, sizeof(f64)*NMINOR * 3);
				//cudaMemset(p_d_eps_by_d_beta_z_, 0, sizeof(f64)*NMINOR * 3);

				//cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
				//kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
				//		pX_use->p_info,
				//		p_epsilon3,
				//		pX_use->p_izTri_vert,
				//		pX_use->p_szPBCtri_vert,
				//		pX_use->p_izNeigh_TriMinor,
				//		pX_use->p_szPBC_triminor,

				//		p_temp6, // ita
				//		p_temp5,

				//		p_MAR_neut3, // just accumulates
				//		NT_addition_rates_d_temp,
				//		NT_addition_tri_d);
				//Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

				//cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
				//kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				//		// eps = v - (v_k +- h [viscous effect])
				//		hsub,
				//		pX_use->p_info,
				//		p_epsilon3, // Jacobi regressor or Richardson.
				//		zero_vec3,
				//		p_MAR_neut3,
				//		pX_use->p_n_minor,
				//		pX_use->p_AreaMinor,
				//		p_d_eps_by_d_beta_x_,
				//		p_d_eps_by_d_beta_y_,
				//		p_d_eps_by_d_beta_z_,
				//		0
				//		);
				//Call(cudaThreadSynchronize(), "cudaTS Create deps by dbeta Jacobi");
				//	
				//	// OVERWRITE:
				//AssembleVector3 << <numTilesMinor, threadsPerTileMinor >> > (p_vJacobi_n, p_d_eps_by_d_beta_x_,
				//	p_d_eps_by_d_beta_y_,
				//	p_d_eps_by_d_beta_z_);
				//Call(cudaThreadSynchronize(), "cudaTS AssembleVector3");
				//// Overwriting vJacobi_n because it's the 2nd regressor!

				//// Use: Jacobi, A*Jacobi, Jacobi(A*Jacobi)
				//// Or   epsilon, Aepsilon, Jacobi(A epsilon)

				//// Overwrite 1st regressor with Jacobi:
				//kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
				//	(  
				//		pX_use->p_info,
				//		p_vJacobi_n, // input
				//		p_InvertedMatrix_n,
				//		// output:
				//		p_temp3_2 // 3-vec array
				//		);
				//Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

				kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
					(
						pX_use->p_info,
						p_epsilon3, // input
						p_InvertedMatrix_n,
						// output:
						p_vJacobi_n // 3-vec array
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

				// Really didn't help!
			//	ResettoGeomAverage << <numTilesMinor, threadsPerTileMinor >> > (p_vJacobi_n, p_epsilon3);
				//SetProduct3 << <numTilesMinor, threadsPerTileMinor >> > (p_temp3_3, p_epsilon3, p_vJacobi_n);
			//	Call(cudaThreadSynchronize(), "cudaTS GeomAvg");				
				// we want to amplify the values where epsilon is higher -- otherwise Jac seems to live elsewhere.

				// .. and do not overwrite prev move as regressor.

				// will this help? :

				// The reason it doesn't help is that there are + and - regions of epsilon, and eps*eps/coeffself is like eps*eps, useless.

				//SetProduct3 << <numTilesMinor, threadsPerTileMinor >> > (p_temp3_2, p_epsilon3, p_vJacobi_n);
				//Call(cudaThreadSynchronize(), "cudaTS SetProduct3");
				SetConsoleTextAttribute(hConsole, 15);



				// Get deps/dbeta for Jacobi:
				/*
				cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
				kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

					pX_use->p_info,
					p_vJacobi_n,
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_izNeigh_TriMinor,
					pX_use->p_szPBC_triminor,

					p_temp6, // ita
					p_temp5,

					p_MAR_neut3, // just accumulates
					NT_addition_rates_d_temp,
					NT_addition_tri_d);
				Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

				cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
				kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
					// eps = v - (v_k +- h [viscous effect])
					hsub,
					pX_use->p_info,
					p_vJacobi_n, // Jacobi regressor
					zero_vec3,
					p_MAR_neut3,
					pX_use->p_n_minor,
					pX_use->p_AreaMinor,
					p_d_eps_by_d_beta_x_ + NMINOR,
					p_d_eps_by_d_beta_y_ + NMINOR,
					p_d_eps_by_d_beta_z_ + NMINOR,
					0
					);
				Call(cudaThreadSynchronize(), "cudaTS Create deps by dbeta Jacobi");

				AssembleVector3 << <numTilesMinor, threadsPerTileMinor >> >
					(p_temp3_3, p_d_eps_by_d_beta_x_ + NMINOR,
						p_d_eps_by_d_beta_y_ + NMINOR,
						p_d_eps_by_d_beta_z_ + NMINOR);
				Call(cudaThreadSynchronize(), "cudaTS AssembleVector3");

				kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
					(
						pX_use->p_info,
						p_temp3_3, // input
						p_InvertedMatrix_n,
						// output:
						p_temp3_2 // 3-vec array
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi II");
								*/

			}; // was it 0th iteration

		} else {
			//  ((iIteration == 0) && (iHistoryVN != 0))
			if (iHistoryVN > 1) {

				//1st regressor:
				cudaMemcpy(p_epsilon3, p_prev_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
				//2nd regressor:
				cudaMemcpy(p_vJacobi_n, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
				//3rd regressor:
				// Jacobi of the more recent move:

				kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
					(
						pX_use->p_info,
						p_stored_move3, // input
						p_InvertedMatrix_n,
						// output:
						p_temp3_2 // 3-vec array
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");
				SetConsoleTextAttribute(hConsole, 11);

				// Highly effective once it gets going, can knock the residuals down by 3 orders.
				// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			} else {
				// We've only got 1 preexisting move.

				AssembleVector3 << <numTilesMinor, threadsPerTileMinor >> > (p_epsilon3, p_epsilon_x, p_epsilon_y, p_epsilon_z);
				Call(cudaThreadSynchronize(), "cudaTS AssembleVector3");
				// p_epsilon3 can be our first regressor.
				
				kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
					(
						pX_use->p_info,
						p_epsilon3, // input
						p_InvertedMatrix_n,
						// output:
						p_vJacobi_n // 3-vec array
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

				cudaMemcpy(p_temp3_2, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
				
				SetConsoleTextAttribute(hConsole, 14);
			};
		};

		
		SetActiveWindow(hwndGraphics);
		cudaMemcpy(p_temphost1, p_epsilon_y + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		long iVertex;
		char buffer[256];
		Vertex * pVertex = pTriMesh->X;
		plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = p_temphost1[iVertex];

			++pVertex;
			++pdata;
		}

		sprintf(buffer, "epsilon_y iteration %d", iIteration);
		Graph[0].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

		cudaMemcpy(p_MAR_ion_host, p_v_n + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_tempvec3host, p_v_n_k + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToHost);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_MAR_ion_host[iVertex].y - p_tempvec3host[iVertex].y;
			pdata->temp.y = p_MAR_ion_host[iVertex].y - p_tempvec3host[iVertex].y;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "vydiff iteration %d", iIteration);
		Graph[1].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		cudaMemcpy(p_temphost1, p_epsilon_z + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = p_temphost1[iVertex];
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "epsilon_z iteration %d", iIteration);
		Graph[2].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		cudaMemcpy(p_tempvec3host, p_epsilon3 + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToHost);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec3host[iVertex].y;
			pdata->temp.y = p_tempvec3host[iVertex].y;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "regressor 1.y iteration %d", iIteration);
		Graph[3].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		cudaMemcpy(p_tempvec3host, p_vJacobi_n + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToHost);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec3host[iVertex].y;
			pdata->temp.y = p_tempvec3host[iVertex].y;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "regressor 2.y iteration %d", iIteration);
		Graph[4].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		cudaMemcpy(p_tempvec3host, p_temp3_2 + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToHost);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec3host[iVertex].y;
			pdata->temp.y = p_tempvec3host[iVertex].y;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "regressor 3.y iteration %d", iIteration);
		Graph[5].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);
		
		SetActiveWindow(hwndGraphics);
		ShowWindow(hwndGraphics, SW_HIDE);
		ShowWindow(hwndGraphics, SW_SHOW);
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		printf("done graphs\n\n");

		getch();

		
		// ________________________
		// Now get the derivatives:
		// ========================

		cudaMemset(p_d_eps_by_d_beta_x_, 0, sizeof(f64)*NMINOR*3);
		cudaMemset(p_d_eps_by_d_beta_y_, 0, sizeof(f64)*NMINOR*3);
		cudaMemset(p_d_eps_by_d_beta_z_, 0, sizeof(f64)*NMINOR*3);

//		kernelSetx << <numTilesMinor, threadsPerTileMinor >> >
//			(v3temp, p_vJacobi_n);
//		Call(cudaThreadSynchronize(), "cudaTS kernelSetx");
//
		// we are proposing to put a separate coefficient on the x-change
		// so we need to distinguish the effect of x-change [which affects eps x only]
		// but we can calc this at the same thing as d epsy / dbetay

		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_epsilon3,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			p_epsilon3, // Richardson regressor
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_eps_by_d_beta_x_,
			p_d_eps_by_d_beta_y_,
			p_d_eps_by_d_beta_z_,
			0,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create deps by dbeta Jacobi");


		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			p_vJacobi_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			p_vJacobi_n, // Jacobi regressor
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_eps_by_d_beta_x_ + NMINOR, 
			p_d_eps_by_d_beta_y_ + NMINOR, 
			p_d_eps_by_d_beta_z_ + NMINOR, 
			0,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create deps by dbeta Jacobi");
		

		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			p_temp3_2,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			p_temp3_2,
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_eps_by_d_beta_x_ + 2 * NMINOR,
			p_d_eps_by_d_beta_y_ + 2 * NMINOR,
			p_d_eps_by_d_beta_z_ + 2 * NMINOR,
			0,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		cudaMemset(p_eps_against_deps, 0, sizeof(f64_vec3) * numTilesMinor);
		cudaMemset(p_sum_product_matrix3, 0, sizeof(Symmetric3) * numTilesMinor);
		AccumulateSummandsScalars3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_x,
			p_d_eps_by_d_beta_x_,
			p_d_eps_by_d_beta_x_ + NMINOR,
			p_d_eps_by_d_beta_x_ + 2 * NMINOR,

			// 3+9+1 outputs:
			p_eps_against_deps,
			p_sum_product_matrix3, // Symmetric3 for the 3 regressors.
			p_sum_eps_eps
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummandsScalars3");

		cudaMemcpy(p_eps_against_deps_host, p_eps_against_deps, sizeof(f64) * 3 * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host3, p_sum_product_matrix3, sizeof(Symmetric3) * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		memset(&Y, 0, sizeof(f64_tens3));
		memset(&vec3, 0, sizeof(f64_vec3));
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			vec3.x += p_eps_against_deps_host[iTile].x;
			vec3.y += p_eps_against_deps_host[iTile].y;
			vec3.z += p_eps_against_deps_host[iTile].z;
			Y.xx += p_sum_product_matrix_host3[iTile].xx;
			Y.xy += p_sum_product_matrix_host3[iTile].xy;
			Y.xz += p_sum_product_matrix_host3[iTile].xz;
			Y.yy += p_sum_product_matrix_host3[iTile].yy;
			Y.yz += p_sum_product_matrix_host3[iTile].yz;
			Y.zz += p_sum_product_matrix_host3[iTile].zz;
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		Y.yx = Y.xy;
		Y.zx = Y.xz;
		Y.zy = Y.yz;
		Y.Inverse(inv);
		f64_vec3 beta_x = -(inv*vec3);
		SS.x = sum_eps_eps;

		cudaMemset(p_eps_against_deps, 0, sizeof(f64_vec3) * numTilesMinor);
		cudaMemset(p_sum_product_matrix3, 0, sizeof(Symmetric3) * numTilesMinor);
		AccumulateSummandsScalars3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_y,
			p_d_eps_by_d_beta_y_,
			p_d_eps_by_d_beta_y_ + NMINOR,
			p_d_eps_by_d_beta_y_ + 2 * NMINOR,

			// 3+9+1 outputs:
			p_eps_against_deps,
			p_sum_product_matrix3, // Symmetric3 for the 3 regressors.
			p_sum_eps_eps
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummandsScalars3");

		cudaMemcpy(p_eps_against_deps_host, p_eps_against_deps, sizeof(f64) * 3 * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host3, p_sum_product_matrix3, sizeof(Symmetric3) * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		memset(&Y, 0, sizeof(f64_tens3));
		memset(&vec3, 0, sizeof(f64_vec3));
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			vec3.x += p_eps_against_deps_host[iTile].x;
			vec3.y += p_eps_against_deps_host[iTile].y;
			vec3.z += p_eps_against_deps_host[iTile].z;
			Y.xx += p_sum_product_matrix_host3[iTile].xx;
			Y.xy += p_sum_product_matrix_host3[iTile].xy;
			Y.xz += p_sum_product_matrix_host3[iTile].xz;
			Y.yy += p_sum_product_matrix_host3[iTile].yy;
			Y.yz += p_sum_product_matrix_host3[iTile].yz;
			Y.zz += p_sum_product_matrix_host3[iTile].zz;
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		Y.yx = Y.xy;
		Y.zx = Y.xz;
		Y.zy = Y.yz;
		Y.Inverse(inv);
		f64_vec3 beta_y = -(inv*vec3);
		SS.y = sum_eps_eps;

		cudaMemset(p_eps_against_deps, 0, sizeof(f64_vec3) * numTilesMinor);
		cudaMemset(p_sum_product_matrix3, 0, sizeof(Symmetric3) * numTilesMinor);
		AccumulateSummandsScalars3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_z,
			p_d_eps_by_d_beta_z_,
			p_d_eps_by_d_beta_z_ + NMINOR,
			p_d_eps_by_d_beta_z_ + 2 * NMINOR,

			// 3+9+1 outputs:
			p_eps_against_deps,
			p_sum_product_matrix3, // Symmetric3 for the 3 regressors.
			p_sum_eps_eps
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateSummandsScalars3");

		cudaMemcpy(p_eps_against_deps_host, p_eps_against_deps, sizeof(f64) * 3 * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host3, p_sum_product_matrix3, sizeof(Symmetric3) * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		memset(&Y, 0, sizeof(f64_tens3));
		memset(&vec3, 0, sizeof(f64_vec3));
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			vec3.x += p_eps_against_deps_host[iTile].x;
			vec3.y += p_eps_against_deps_host[iTile].y;
			vec3.z += p_eps_against_deps_host[iTile].z;
			Y.xx += p_sum_product_matrix_host3[iTile].xx;
			Y.xy += p_sum_product_matrix_host3[iTile].xy;
			Y.xz += p_sum_product_matrix_host3[iTile].xz;
			Y.yy += p_sum_product_matrix_host3[iTile].yy;
			Y.yz += p_sum_product_matrix_host3[iTile].yz;
			Y.zz += p_sum_product_matrix_host3[iTile].zz;
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		Y.yx = Y.xy;
		Y.zx = Y.xz;
		Y.zy = Y.yz;
		Y.Inverse(inv);
		f64_vec3 beta_z = -(inv*vec3);
		SS.z = sum_eps_eps;
		printf("SS %1.8E %1.8E %1.8E\n", SS.x, SS.y, SS.z);
		L2eps3.x = sqrt(SS.x / (real)NMINOR);
		L2eps3.y = sqrt(SS.y / (real)NMINOR);
		L2eps3.z = sqrt(SS.z / (real)NMINOR);

		if (iIteration % 4 == 0) 
			printf("Iteration %d neutvisc: [ x beta %1.9E %1.9E %1.9E L2eps %1.9E ] \n"
				   "Iteration %d neutvisc: [ y beta %1.9E %1.9E %1.9E L2eps %1.9E ] \n"
				   "Iteration %d neutvisc: [ z beta %1.9E %1.9E %1.9E L2eps %1.9E ] \n"
				, iIteration, beta_x.x, beta_x.y, beta_x.z, L2eps3.x,
				iIteration, beta_y.x, beta_y.y, beta_y.z, L2eps3.y,
				iIteration, beta_z.x, beta_z.y, beta_z.z, L2eps3.z
				);

		 
		cudaMemcpy(p_temp3_1, p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
		kernelAddLC_3vec3 << <numTilesMinor, threadsPerTileMinor >> > (
			p_v_n, beta_x, beta_y, beta_z,
			p_epsilon3, p_vJacobi_n, p_temp3_2); // beta_z means coefficients for dimension z 
		Call(cudaThreadSynchronize(), "cudaTS Addtovn ___");

		SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >
			(p_temp3_2, p_v_n, p_temp3_1);// a-b
		Call(cudaThreadSynchronize(), "cudaTS subtractvec3");
		// let's us step back some of this move next time if we want.

		iIteration++;

	} while ((bContinue) && (iIteration < 3000));
	
	if (iIteration == 3000) printf("\n\n\nOH DEAR\n\n\n\a");

	// Save move for next time:
	iHistoryVN++;
	cudaMemcpy(p_prev_move3, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, p_v_n, p_v_n_k);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
	

}

void RunBackward8LSForNeutralViscosity_Geometric(f64_vec3 * p_v_n_k, f64_vec3 * p_v_n,
	f64 const hsub, cuSyst * pX_use)
// BE SURE ABOUT PARAMETER ORDER -- CHECK IT CHECK IT
{
	// ***************************************************
	// Requires averaging of n,T to triangles first. & ita
	// ***************************************************

	
	static int iHistoryNeutVisc = 0;

	f64 beta[REGRESSORS];
	f64 beta_e, beta_i, beta_n;
	long iTile;
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	long iMinor;
	f64 L2eps;
	int i, iMoveType;
	f64 TSS_xy, TSS_z, RSS_xy, RSS_z;
	f64 Rsquared_xy, Rsquared_z;

	cudaMemset(p_d_eps_ez_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS); // rubbish

	//cudaMemcpy(p_v_n, p_v_n_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemset(zero_vec4, 0, sizeof(v4)*NMINOR);
	int iIteration = 0;
	bool bContinue = true;

	Vector3Breakdown << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3_2, p_stored_move_neut_xy2, p_stored_move_neut_z2,
		p_SelectflagNeut);
	Call(cudaThreadSynchronize(), "cudaTS Vector3Breakdown");
	Vector3Breakdown << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, p_stored_move_neut_xy, p_stored_move_neut_z,
		p_SelectflagNeut);
	Call(cudaThreadSynchronize(), "cudaTS Vector3Breakdown");

	//careful about past regressors and where they affect.
	// Aim: split out z vs xy. === probably a failing strategy as it turns out?

	cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_v_n,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_ita_n, // ita
		p_nu_n,
		p_MAR_neut2, // just accumulates
		NT_addition_rates_d_2,
		NT_addition_tri_d2);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_epsilon_x, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_y, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_z, 0, sizeof(f64)*NMINOR);
	// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
	kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
		// eps = v - (v_k +- h [viscous effect])
		// x = -eps/coeffself
		hsub,
		pX_use->p_info,
		p_v_n,
		p_v_n_k,
		p_MAR_neut2,
		pX_use->p_n_minor,
		pX_use->p_AreaMinor,
		p_epsilon_x,
		p_epsilon_y,
		p_epsilon_z,
		p_bFailed,
		p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

	bContinue = false;
	cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
	if (i < numTilesMinor) bContinue = true;
	// primitive mechanism

	RSS_xy = 0.0;
	RSS_z = 0.0;
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_x, p_SS);
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_xy += p_SS_host[iTile];
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_y, p_SS);
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_xy += p_SS_host[iTile];
	f64 L2eps_xy = sqrt(RSS_xy / (real)NMINOR);
	printf("L2eps xy %1.8E ", L2eps_xy);

	if (L2eps_xy == 0.0) goto labelBudgerigar3;

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_z, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_z += p_SS_host[iTile];
	f64 L2eps_z = sqrt(RSS_z / (real)NMINOR);
	printf("nz %1.8E ", L2eps_z);

	f64 oldRSS = 0.0;

	AssembleVector2 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_xy, p_epsilon_x, p_epsilon_y);
	Call(cudaThreadSynchronize(), "cudaTS AssembleVector2");

	while (bContinue)
	{

		// . Get Jacobi inverse:
		// ======================
		CalculateCoeffself << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_vie,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_n,   // nT / nu ready to look up
			p_nu_n,   // nT / nu ready to look up
			pX_use->p_B,
			p__matrix_xy_i, // matrix ... 
			p__coeffself_iz, // we are being naughty and using the memory from ion
			p__xzyzzxzy_i, // xz yz zx zy == 0
			0,
			m_n_,
			1.0 / m_n_
			);
		Call(cudaThreadSynchronize(), "cudaTS CalculateCoeffself ion");
		 
		kernelCreateNeutralInverseCoeffself << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p__coeffself_iz,
			p__invcoeffself
			);
		Call(cudaThreadSynchronize(), "cudaTS CreateNeutralInverseCoeffself");


		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		// 2. Regressor creation code and get deps/dbeta
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		// Zero out to begin with. We will use the ith NMINOR array combination for each regr.
		cudaMemset(p_regressors2, 0, sizeof(f64_vec2)*NMINOR*REGRESSORS);
		cudaMemset(p_regressors_iz, 0, sizeof(f64)*NMINOR*REGRESSORS);
		
		cudaMemset(p_d_epsxy_by_d_beta_i, 0, sizeof(f64_vec2)*NMINOR*REGRESSORS);
		cudaMemset(p_d_eps_iz_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS);

		// Regr 1: eps_xy
		// Regr 2: eps_z
		// Regr 3: Jac xy
		// Regr 4: Jac z
		// Regr 5: deps xy/dbeta 3
		// Regr 6: deps z/dbeta 4
		// Regr 7: prev move xy
		// Regr 8: prev move z
		// [on 1st ever go, multiply]

		// These all come out almost same -- is something wrong with that?

		// New plan:

		// 0, 1: Jac xy, z
		// 2, 3: square
		// 4, 5: deps(xy, z)/dbeta 0,1 minus 0,1
		// 6, 7: stored move OR square of the above
		
		// Result:
		// That worked for a bit but it got stuck. Convergence not obtained.

		// Try this:

		// 0, 1: epsilon
		// 2, 3: deps(xy, z)/dbeta(0,1) minus (0,1)
		// 4, 5: deps(xy, z)/dbeta(2,3) minus (2,3)
		// 6, 7: previous move or another iteration of same.

		// Result:
		//   .  Convergence obtained in 147 steps.

		// These results do not really make sense in the light of the above. We basically swapped out square for d^2eps but the latter was 0...

		// Try:
		// 0, 1: epsilon
		// 2, 3: deps(xy, z)/dbeta(0,1) minus (0,1)
		// 4, 5: square of epsilon
		// 6, 7: previous move

		// Result:
		//   .  Convergence obtained in 241 steps.
		

		// Try:
		// 0, 1: epsilon
		// 2, 3: deps(xy, z)/dbeta(0,1) minus (0,1)
		// 4, 5: square of epsilon
		// 6 = previous to previous move (xyz)
		// 7 = previous move (xyz)
		
		// Result:
		//   .  Convergence obtained in 229 steps.

		// Try:
		// 0, 1: epsilon
		// 2, 3: deps(xy, z)/dbeta(0,1) minus (0,1)
		// 4, 5: deps(xy, z)/dbeta(2,3) minus (2,3)
		// 6 = previous to previous move (xyz)
		// 7 = previous move (xyz)
		
		// Result:
		//  .  Convergence obtained in 153 steps.



		//   .  We are seeing occasionally an increase in L2epsz : predictions how far out ?
		// It would take an investigation to see if that is caused by just the nonlinearity.



		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// Inventiveness called for !! !

		// .. Maybe we need extra search for coefficient
		// .. Maybe we need smash method.
		// Do what we have to do .


		cudaMemcpy(p_regressors2,
			p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_regressors_iz + NMINOR,
				   p_epsilon_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 0,
			1, 0); // XY_ONLY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 1,
			0, 1);// Z_ONLY

		Subtract_xy << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors2 + NMINOR * 2, p_d_epsxy_by_d_beta_i, p_regressors2);
		Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
		Subtract << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors_iz + NMINOR * 3, p_d_eps_iz_by_d_beta_i + NMINOR, p_regressors_iz + NMINOR);
		Call(cudaThreadSynchronize(), "cudaTS Subtract z");

		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 2,
			1, 0); // XY_ONLY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 3,
			0, 1);// Z_ONLY

		//cudaMemcpy(p_regressors2 + NMINOR * 4, p_regressors2, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
		//kernelSquare2 << <numTilesMinor, threadsPerTileMinor >> >(p_regressors2 + NMINOR * 4);
		//Call(cudaThreadSynchronize(), "cudaTS kernelsquare2");

		//cudaMemcpy(p_regressors_iz + NMINOR * 5, p_regressors_iz + NMINOR, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		//kernelSquare << <numTilesMinor, threadsPerTileMinor >> >(p_regressors_iz + NMINOR * 5);
		//Call(cudaThreadSynchronize(), "cudaTS kernelsquare");
//
		Subtract_xy << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i + 2*NMINOR, p_regressors2+ 2*NMINOR);
		Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
		Subtract << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors_iz + NMINOR * 5, p_d_eps_iz_by_d_beta_i + 3*NMINOR, p_regressors_iz + 3*NMINOR);
		Call(cudaThreadSynchronize(), "cudaTS Subtract z");

		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 4,
			1, 0);// XY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 5,
			0, 1); // Z


		/*


//		cudaMemcpy(p_regressors_iz + NMINOR,
//			p_epsilon_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

		kernelCreateJacobiRegressorNeutralxy2 << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors2 , //+NMINOR* 2,
				p_epsilon_x, p_epsilon_y,
				p__invcoeffself);
		Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
		
		kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors_iz + NMINOR, // *3
				p_epsilon_z,
				p__invcoeffself);
		Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

		cudaMemcpy(p_regressors2 + NMINOR * 2, p_regressors2, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
		kernelSquare2 << <numTilesMinor, threadsPerTileMinor >> >(p_regressors2 + NMINOR * 2);
		Call(cudaThreadSynchronize(), "cudaTS kernelsquare2");

		cudaMemcpy(p_regressors_iz + NMINOR * 3, p_regressors_iz + NMINOR, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		kernelSquare << <numTilesMinor, threadsPerTileMinor >> >(p_regressors_iz + NMINOR*3);
		Call(cudaThreadSynchronize(), "cudaTS kernelsquare");

		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 0,
			1, 0); // XY_ONLY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 1,
			0, 1);// Z_ONLY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 2,
			1, 0);// XY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 3,
			0, 1); // Z
		
		// Regressor 5: for J
		Subtract_xy << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i, p_regressors2);
		Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
		//cudaMemcpy(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i + NMINOR*2, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
		// deps/dbeta includes +1*regressor, so take -1*regressor

		// Regressor 6: for J
		Subtract << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressors_iz + NMINOR * 5, p_d_eps_iz_by_d_beta_i + NMINOR, p_regressors_iz + NMINOR);
		Call(cudaThreadSynchronize(), "cudaTS Subtract");

		//cudaMemcpy(p_regressors_iz + NMINOR * 5, p_d_eps_iz_by_d_beta_i + NMINOR*3, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 4,
			1, 0); // XY_ONLY
		SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 5,
			0, 1);// Z_ONLY
			*/

		if ((iHistoryNeutVisc > 1) || (iIteration > 1)) {

			cudaMemcpy(p_regressors2 + NMINOR * 7, p_stored_move_neut_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors_iz + NMINOR * 7, p_stored_move_neut_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(p_regressors2 + NMINOR * 6, p_stored_move_neut_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(p_regressors_iz + NMINOR * 7, p_stored_move_neut_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors2 + NMINOR * 6, p_stored_move_neut_xy2, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors_iz + NMINOR * 6, p_stored_move_neut_z2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

			// Should not be necessary -- we killed already above where we broke it down.

			KillOutsideRegion << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 6, p_regressors_iz + NMINOR * 6, p_SelectflagNeut);
			KillOutsideRegion << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 7, p_regressors_iz + NMINOR * 7, p_SelectflagNeut);
			Call(cudaThreadSynchronize(), "cudaTS KillOutsideRegion");


			SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 6,
				1, 1);
			SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 7,
				1, 1); 
		} else {
			if ((iHistoryNeutVisc == 1) || (iIteration == 1)){
				// If we have been through the code before then p_stored_move_neut_xy is now populated.
				// & If we have been through 1 iteration then p_stored_move_neut_xy is now populated.
				cudaMemcpy(p_regressors2 + NMINOR * 6, p_stored_move_neut_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
				cudaMemcpy(p_regressors_iz + NMINOR * 7, p_stored_move_neut_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
				SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 6,
					1, 0);// XY
				SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 7,
					0, 1); // Z
				
				KillOutsideRegion << <numTilesMinor, threadsPerTileMinor >> >
					(p_regressors2 + NMINOR * 6, p_regressors_iz + NMINOR*7, p_SelectflagNeut);
				Call(cudaThreadSynchronize(), "cudaTS KillOutsideRegion");


			} else {
				// Just use more deps/dbeta as regressors on the first ever go:
				//cudaMemcpy(p_regressors2 + NMINOR * 6, p_d_epsxy_by_d_beta_i + NMINOR * 4, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
				//cudaMemcpy(p_regressors_iz + NMINOR * 7, p_d_eps_iz_by_d_beta_i + NMINOR * 5, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

				cudaMemcpy(p_regressors2 + NMINOR * 6, p_regressors2 + NMINOR * 4, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
				kernelSquare2 << <numTilesMinor, threadsPerTileMinor >> > (p_regressors2 + NMINOR * 6);
				Call(cudaThreadSynchronize(), "cudaTS kernelsquare2");

				cudaMemcpy(p_regressors_iz + NMINOR * 7, p_regressors_iz + NMINOR * 5, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
				kernelSquare << <numTilesMinor, threadsPerTileMinor >> > (p_regressors_iz + NMINOR * 7);
				Call(cudaThreadSynchronize(), "cudaTS kernelsquare");

				SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 6,
					1, 0);// XY
				SubroutineComputeDbyDbetaNeutral(hsub, p_regressors2, p_regressors_iz, p_v_n, pX_use, 7,
					0, 1); // Z
			};
		};


		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		// 3. Solver section
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&

				
	//	cudaMemset(p_d_epsxy_by_d_beta_i+NMINOR, 0, sizeof(f64_vec2)*NMINOR*(REGRESSORS-1));
	//	cudaMemset(p_d_eps_iz_by_d_beta_i+NMINOR, 0, sizeof(f64)*NMINOR*(REGRESSORS-1));


		// 3a. Add up sum products

		cudaMemset(p_eps_against_deps, 0, sizeof(f64)*REGRESSORS * numTilesMinor);
		cudaMemset(p_sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS* numTilesMinor);
		cudaMemset(p_epsilon_ez, 0, sizeof(f64)*NMINOR);
		kernelAccumulateSummandsVisc << <numTilesMinor, threadsPerTileMinor / 4 >> > (
			p_epsilon_xy, // 
			p_epsilon_z,
			p_epsilon_ez, // rubbish
			p_d_epsxy_by_d_beta_i, // f64_vec2
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i, // rubbish

			p_eps_against_d_eps,  // 1x8 for each tile
			p_sum_product_matrix // this is 8x8 for each tile
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands visc2");


		// 3b. Solve eqns

		cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_d_eps, sizeof(f64) * REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(f64) * REGRESSORS*REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);

		f64 eps_deps[REGRESSORS];
		f64 sum_product_matrix[REGRESSORS*REGRESSORS];
		memset(eps_deps, 0, sizeof(f64) * REGRESSORS);
		memset(sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			for (i = 0; i < REGRESSORS; i++)
				eps_deps[i] -= p_eps_against_d_eps_host[iTile *REGRESSORS + i];

			// Note minus, to get beta already negated.

			for (i = 0; i < REGRESSORS*REGRESSORS; i++)
				sum_product_matrix[i] += p_sum_product_matrix_host[iTile *REGRESSORS*REGRESSORS + i];
		};
		// Try this:
		for (i = 0; i < REGRESSORS; i++)
			if ((sum_product_matrix[REGRESSORS*i + i] == 0.0) && (eps_deps[i] == 0.0))
				sum_product_matrix[REGRESSORS*i + i] = 1.0;

		if (!GlobalSuppressSuccessVerbosity)
		{
			printf("\n");
			for (i = 0; i < REGRESSORS; i++) {
				for (int j = 0; j < REGRESSORS; j++)
					printf("ei %1.9E ", sum_product_matrix[i*REGRESSORS + j]);
				printf(" |  %1.9E \n", eps_deps[i]);
			}
			printf("\n");
		}
		// Note that file 1041-Krylov.pdf claims that simple factorization for LS is an
		// unstable method and that is why the complications of GMRES are needed.

		// now we need the LAPACKE dgesv code to solve the 8x8 linear equation.
		f64 storeRHS[REGRESSORS];
		f64 storemat[REGRESSORS*REGRESSORS];
		memcpy(storeRHS, eps_deps, sizeof(f64)*REGRESSORS);
		memcpy(storemat, sum_product_matrix, sizeof(f64)*REGRESSORS*REGRESSORS);

		Matrix_real matLU;
		matLU.Invoke(REGRESSORS);
		for (i = 0; i < REGRESSORS; i++)
			for (int j = 0; j < REGRESSORS; j++)
				matLU.LU[i][j] = sum_product_matrix[i*REGRESSORS + j];
		matLU.LUdecomp();
		matLU.LUSolve(eps_deps, beta);

		printf("\nbeta: ");
		for (i = 0; i < REGRESSORS; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n");
		
		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

		printf("Iteration %d visc: [ beta ", iIteration);
		for (i = 0; i < REGRESSORS; i++) printf("%1.3E ", beta[i]);
		printf(" ]\n");
		//
		//f64 tempf64_2, tempf64_3, tempf64_4;
		//// prediction:
		//cudaMemcpy(&tempf64, &(p_epsilon_x[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&tempf64_2, &(p_epsilon_y[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("%d epsilon xy %1.8E %1.8E ", CHOSEN, tempf64, tempf64_2);

		//cudaMemcpy(&tempf64_3, &(p_d_epsxy_by_d_beta_i[CHOSEN].x), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&tempf64_4, &(p_d_epsxy_by_d_beta_i[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("deps / dbeta0 %1.8E %1.8E |", tempf64_3, tempf64_4);
		//
		//f64 predict1 = tempf64 + beta[0] * tempf64_3;
		//f64 predict2 = tempf64_2 + beta[0] * tempf64_4;
		//printf(" prediction %1.8E %1.8E \n", predict1, predict2);
		//
		// 3c. Add lc to soln vector.

		
		// Note that if it's the first ever iteration, this won't now get used, so this is okay:
		cudaMemcpy(p_stored_move_neut_xy2, p_stored_move_neut_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_stored_move_neut_z2, p_stored_move_neut_z, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		
		AddLCtoVector3 << <numTilesMinor, threadsPerTileMinor >> >
			(p_v_n, p_regressors2, p_regressors_iz, p_stored_move_neut_xy,
				p_stored_move_neut_z);
		Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector3component");

		//  Now compute epsilon and RSS
		// =============================

		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);

		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_n, // ita
			p_nu_n,
			p_MAR_neut2, // just accumulates
			NT_addition_rates_d_2,
			NT_addition_tri_d2);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

		cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_epsilon_x, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_epsilon_y, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_epsilon_z, 0, sizeof(f64)*NMINOR);
		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_v_n,
			p_v_n_k,
			p_MAR_neut2,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_epsilon_x,
			p_epsilon_y,
			p_epsilon_z,
			p_bFailed,
			p_SelectflagNeut
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		AssembleVector2 << <numTilesMinor, threadsPerTileMinor >> > (
			p_epsilon_xy, p_epsilon_x, p_epsilon_y);
		Call(cudaThreadSynchronize(), "cudaTS AssembleVector2");

	//	cudaMemcpy(&tempf64, &(p_epsilon_x[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(&tempf64_2, &(p_epsilon_y[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf(" Result %1.8E %1.8E \n\n", tempf64, tempf64_2);

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism

		// Collect L2eps and R^2:
		TSS_xy = RSS_xy; TSS_z = RSS_z; 
		RSS_xy = 0.0; RSS_z = 0.0; 
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_x, p_SS);
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_xy += p_SS_host[iTile];
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_y, p_SS);
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_xy += p_SS_host[iTile];
		L2eps_xy = sqrt(RSS_xy / (real)NMINOR);

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_z, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS_z += p_SS_host[iTile];
		L2eps_z = sqrt(RSS_z / (real)NMINOR);

		// What is R^2? We should like to report it.
		Rsquared_xy = (TSS_xy - RSS_xy) / TSS_xy;
		Rsquared_z = (TSS_z - RSS_z) / TSS_z;
		Rsquared_xy *= 100.0; Rsquared_z *= 100.0; 
		 
		if (L2eps_xy == 0.0) bContinue = false;
		printf("L2eps xy %1.8E z %1.8E TOTALRSS %1.10E \nR^2 %2.3f%% %2.3f%% bCont: %d\n",
			L2eps_xy, L2eps_z, 
			RSS_xy + RSS_z ,
			Rsquared_xy, Rsquared_z,(bContinue ? 1 : 0));

		// D E B U G :

		//if ((iIteration > 1) && (RSS_xy + RSS_iz + RSS_ez > oldRSS)) getch();
		oldRSS = RSS_xy + RSS_z ;

		++iIteration;		
	};	

labelBudgerigar3:

	// Save move for next time:
	iHistoryNeutVisc++;

	// This will no longer work because it's v_k_modified.

	//cudaMemcpy(p_stored_move3_2, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	//SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, p_v_n, p_v_n_k);
	//Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");
	//GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	FILE * fp = fopen("iter_neut.txt", "a");
	fprintf(fp, "iHistoryNeutVisc %d iIter = %d \n", iHistoryNeutVisc, iIteration);
	fclose(fp);
}

int Compare_f64_vec2(f64_vec2 * p1, f64_vec2 * p2, long N);
int Compare_n_shards(ShardModel * p1, ShardModel * p2, const cuSyst * p_cuSyst_host)
{

	f64 maxdiff = 0.0;
	f64 mindiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1; 
	long i;
	for (i = 0; i < NUMVERTICES; i++)
	{
		f64 diff = (p1[i].n_cent - p2[i].n_cent);
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }
		if (p1[i].n_cent != 0.0) {
			f64 reldiff = fabs(diff / p1[i].n_cent);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max  cent diff: %1.3E at %d : %1.12E %1.12E \n",
			maxdiff, iMax, p1[iMax].n_cent, p2[iMax].n_cent);
	}
	else {
		printf(" Max diff == zero \n");
	};
	if (iMin != -1) {
		printf(" Min diff: %1.3E at %d : %1.12E %1.12E \n",
			mindiff, iMin, p1[iMin].n_cent, p2[iMin].n_cent);
	}
	else {
		printf(" Min diff == zero \n");
	};
	if (iMaxRel != -1) {
		printf(" Max rel diff %1.3E at %d : %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel].n_cent, p2[iMaxRel].n_cent);
	}
	else {
		printf(" Max rel diff == zero \n");
	}

	maxdiff = 0.0;
	mindiff = 0.0;
	maxreldiff = 0.0;
	iMin = -1;
	iMax = -1;
	iMaxRel = -1;
	f64 diff;
	int j; f64 diff_;
	for (i = 0; i < NUMVERTICES; i++)
	{
		diff = 0.0;
		short neigh_len = p_cuSyst_host->p_info[i + BEGINNING_OF_CENTRAL].neigh_len;
		for (j = 0; j < neigh_len; j++)
		{
			diff_ = fabs(p1[i].n[j] - p2[i].n[j]);
			if (diff_ > diff) diff = diff_;
		}
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }
		if (p1[i].n_cent != 0.0) {
			f64 reldiff = fabs(diff / p1[i].n_cent);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max fabs [--] diff: %1.4E at %d ; n_cent = %1.10E \n",
			maxdiff, iMax,p1[iMax].n_cent);
	} else {
		printf(" Max diff == zero \n");
	};

	if (iMaxRel != -1) {
		printf(" Max rel diff %1.4E at %d ; n_cent =  %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel].n_cent, p2[iMaxRel].n_cent);
	} else {
		printf(" Max rel diff == zero \n");
	};
	return 0;
}
 
int Compare_f64(f64 * p1, f64 * p2, long N);
int Compare_NTrates(NTrates * p1, NTrates * p2)
{
	f64 temp1[NUMVERTICES], temp2[NUMVERTICES];
	long iVertex;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].N;
		temp2[iVertex] = p2[iVertex].N;
	}
	printf("N:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].Nn;
		temp2[iVertex] = p2[iVertex].Nn;
	}  
	printf("Nn:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NiTi;
		temp2[iVertex] = p2[iVertex].NiTi;
	}
	printf("NiTi:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NeTe;
		temp2[iVertex] = p2[iVertex].NeTe;
	}
	printf("NeTe:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NnTn;
		temp2[iVertex] = p2[iVertex].NnTn;
	}
	printf("NnTn:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	return 0;
}
int Compare_f64(f64 * p1, f64 * p2, long N)
{
	// Arithmetic difference:

	f64 maxdiff = 0.0;
	f64 mindiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diff = (p1[i] - p2[i]);
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }

		// Relative difference:
		if (p1[i] != 0.0) {
			f64 reldiff = fabs(diff / p1[i]);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff: %1.3E at %d : %1.12E %1.12E \n",
			maxdiff, iMax, p1[iMax], p2[iMax]);
	} else {
		printf(" Max diff == zero \n");
	};
	if (iMin != -1) {
		printf(" Min diff: %1.3E at %d : %1.12E %1.12E \n",
			mindiff, iMin, p1[iMin], p2[iMin]);
	} else {
		printf(" Min diff == zero \n");
	};
	if (iMaxRel != -1) {
		printf(" Max rel diff %1.3E at %d : %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel], p2[iMaxRel]);
	} else {
		printf(" Max rel diff == zero \n");
	}
	return 0;
}

int Compare_nvals(nvals * p1, nvals * p2, long N)
{
	// Arithmetic difference:
	f64 *n1 = (f64 *)malloc(N * sizeof(f64));
	f64 *n2 = (f64 *)malloc(N * sizeof(f64));
	if (n2 != 0) {
		long i;
		printf("n:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].n;
			n2[i] = p2[i].n;
		}
		Compare_f64(n1, n2, N);
		printf("n_n:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].n_n;
			n2[i] = p2[i].n_n;
		}
		Compare_f64(n1, n2, N);
		free(n1);
		free(n2);
		return 0;
	}
	else {
		printf("memory error.");
		return 1;
	}
}
int Compare_T3(T3 * p1, T3 * p2, long N)
{
	// Arithmetic difference:
	f64 *n1 = (f64 *)malloc(N * sizeof(f64));
	f64 *n2 = (f64 *)malloc(N * sizeof(f64));
	if (n2 != 0) {
		long i;
		printf("Tn:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Tn;
			n2[i] = p2[i].Tn;
		}
		Compare_f64(n1, n2, N);
		printf("Ti:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Ti;
			n2[i] = p2[i].Ti;
		}
		Compare_f64(n1, n2, N);
		printf("Te:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Te;
			n2[i] = p2[i].Te;
		}
		Compare_f64(n1, n2, N);
		free(n1);
		free(n2);
		return 0;
	}
	else {
		printf("memory error.");
		return 1;
	}
}
int Compare_structural(structural * p1, structural * p2, long N)
{
	f64_vec2 *pos1 = (f64_vec2 *)malloc(N * sizeof(f64_vec2));
	f64_vec2 *pos2 = (f64_vec2 *)malloc(N * sizeof(f64_vec2));
	
	if (pos2 != 0) {
		long i;
		printf("pos:\n");
		for (i = 0; i < N; i++)
		{
			pos1[i] = p1[i].pos;
			pos2[i] = p2[i].pos;
		}
		Compare_f64_vec2(pos1, pos2, N);
		
		free(pos1);
		free(pos2);
	}
	else {
		printf("memory error.");
		return 1;
	}
	bool bFailneighlen = false, bFailflag = false; // has_periodic: never used?
	long iFailflag_start, iFailneighlen_start, iFailflag, iFailneigh_len;
	long i;
	for (i = 0; i < N; i++)
	{
		if (p1[i].neigh_len != p2[i].neigh_len) {
			if (bFailneighlen == false) iFailneighlen_start = i;
			bFailneighlen = true; 
			iFailneigh_len = i;
		}
		if (p1[i].flag != p2[i].flag) {
			if (bFailflag == false) iFailflag_start = i;
			bFailflag = true; iFailflag = i;
		}
	}
	if (bFailneighlen) printf("Start of inconsistent neigh_len: %d end: %d\n",
		iFailneighlen_start, iFailneigh_len);
	if (bFailflag) printf("Start of inconsistent flag: %d end : %d\n",
		iFailflag, iFailflag_start);
}
 
int Compare_f64_vec2(f64_vec2 * p1, f64_vec2 * p2, long N)
{
	f64 maxdiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diffmod = sqrt((p1[i].x - p2[i].x)*(p1[i].x-p2[i].x)
						+ (p1[i].y - p2[i].y)*(p1[i].y - p2[i].y));
		if (diffmod > maxdiff) { maxdiff = diffmod; iMax = i; }
		
		// Relative difference:
		if ((p1[i].x != 0.0) || (p1[i].y != 0.0)) {
			f64 reldiff = diffmod / p1[i].modulus();
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E\n",
			maxdiff, iMax, p1[iMax].x, p2[iMax].x - p1[iMax].x, p1[iMax].y, p2[iMax].y - p1[iMax].y);
	} else {
		printf(" Max diff == zero. \n");
	}
	if (iMaxRel != -1) {
		printf(" Max rel diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E\n",
			maxreldiff, iMaxRel, p1[iMaxRel].x, p2[iMaxRel].x - p1[iMaxRel].x, p1[iMaxRel].y, p2[iMaxRel].y - p1[iMaxRel].y);
	} else {
		printf(" Max rel diff zero / not found. \n");
	}
	return 0;
}

int Compare_f64_vec3(f64_vec3 * p1, f64_vec3 * p2, long N)
{
	f64 maxdiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diffmod = sqrt((p1[i].x - p2[i].x)*(p1[i].x - p2[i].x)
			+ (p1[i].y - p2[i].y)*(p1[i].y - p2[i].y) + (p1[i].z - p2[i].z)*(p1[i].z - p2[i].z));
		if (diffmod > maxdiff) { maxdiff = diffmod; iMax = i; }

		// Relative difference:
		if ((p1[i].x != 0.0) || (p1[i].y != 0.0)) {
			f64 reldiff = diffmod / p1[i].modulus();
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E z %1.12E diff %1.3E\n",
			maxdiff, iMax, p1[iMax].x, p2[iMax].x - p1[iMax].x, p1[iMax].y, p2[iMax].y - p1[iMax].y,
			p1[iMax].z, p2[iMax].z - p1[iMax].z);
	}
	else {
		printf(" Max diff == zero. \n");
	}
	if (iMaxRel != -1) {
		printf(" Max rel diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E z %1.12E diff %1.3E\n",
			maxreldiff, iMaxRel, p1[iMaxRel].x, p2[iMaxRel].x - p1[iMaxRel].x, p1[iMaxRel].y, p2[iMaxRel].y - p1[iMaxRel].y,
			p1[iMaxRel].z, p2[iMaxRel].z - p1[iMaxRel].z);
	}
	else {
		printf(" Max rel diff zero / not found. \n");
	}
	return 0;
}
 
real GetIzPrescribed(real const t)
{
	
	static real const C_over_38e6PI = (PEAKCURRENT_STATCOULOMB*0.5 / (38.0e6*PEAKTIME))*cos(PIOVERPEAKTIME*0.5 / 19.0e6);
	
	real Iz = -PEAKCURRENT_STATCOULOMB * sin((t + ZCURRENTBASETIME) * 0.5* PIOVERPEAKTIME); // half pi / peaktime

	// Changed back to no osc.
	// Change back to osc :
#ifdef OSCILLATE_IZ																					
	real factor11 = 1.1*(1.0-exp(-t*1.0e10)); // exp(2) = 13.6% at the end of the first cycle.	
	Iz -= C_over_38e6PI*factor11*sin(PI*0.993 + 38.0e6*PI*t);
#endif
	
	//printf("\nGetIzPrescribed : t + ZCURRENTBASETIME = %1.5E : %1.12E\n", t + ZCURRENTBASETIME, Iz);
	return Iz;
}

long numStartZCurrentTriangles__, numEndZCurrentTriangles__;

void PerformCUDA_Invoke_Populate(
	cuSyst * pX_host, // populate in calling routine...
	long numVerts,
	f64 InnermostFrillCentroidRadius,
	f64 OutermostFrillCentroidRadius,
	long numStartZCurrentTriangles_,
	long numEndZCurrentTriangles_
)
{
	int i;
	GlobalSuppressSuccessVerbosity = false;

	numStartZCurrentTriangles__ = numStartZCurrentTriangles_;
	numEndZCurrentTriangles__ = numEndZCurrentTriangles_;

	long iVertex;
	cuSyst * pX1, *pX2, *pX_half;

	printf("sizeof(CHAR4): %zd \n"
		"sizeof(structural): %zd \n"
		"sizeof(LONG3): %zd \n"
		"sizeof(nn): %zd \n",
		sizeof(CHAR4), sizeof(structural), sizeof(LONG3), sizeof(nn));

	if (cuSyst1.bInvoked == false) {

		Call(cudaMemGetInfo(&uFree, &uTotal), "cudaMemGetInfo (&uFree,&uTotal)");
		printf("Before Invokes: uFree %d uTotal %d\n", uFree, uTotal);

		cuSyst1.Invoke();
		cuSyst2.Invoke();
		cuSyst3.Invoke();

		Call(cudaMemGetInfo(&uFree, &uTotal), "cudaMemGetInfo (&uFree,&uTotal)");
		printf("After Invokes: uFree %zu uTotal %zu\n", uFree, uTotal);
	}

	// Populate video constant memory:
	// ________________________________

	printf("got to here - BEFORE\n");

	long const tempint = CHOSEN;
	long * longaddress;
	Call(cudaGetSymbolAddress((void **)(&longaddress), lChosen),
		"cudaGetSymbolAddress((void **)(&longaddress), lChosen)");
	Call(cudaMemcpy(longaddress, &tempint, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &tempint, sizeof(long), cudaMemcpyHostToDevice)");

	printf("got to here - AFTER\n");

	//__constant__ f64 recomb_coeffs[32][3][5];
	//f64 recomb_coeffs_host[32][3][5];
	//__constant__ f64 ionize_coeffs[32][5][5];
	//f64 ionize_coeffs_host[32][5][5];
	//__constant__ f64 ionize_temps[32][10];
	//f64 ionize_temps_host[32][10];
	printf("fopen ionize coeffs\n");
	FILE * fp = fopen("ionize_coeffs.txt", "rt");

	if (fp == NULL) {
		printf("FILE OPEN FAILED - ionize_coeffs.txt is probably not found.\n");
		SafeExit(10865); // we are going to need this
	};

	rewind(fp);
	for (int iV = 0; iV < 32; iV++)
	{
		for (int j = 0; j < 10; j++)
			fscanf(fp, " %lf", &(ionize_temps_host[iV][j]));
			// check format specifier
		for (int iWhich = 0; iWhich < 5; iWhich++) 
			fscanf(fp, " %lf %lf %lf %lf %lf", &(ionize_coeffs_host[iV][iWhich][0]),
				&(ionize_coeffs_host[iV][iWhich][1]),
				&(ionize_coeffs_host[iV][iWhich][2]),
				&(ionize_coeffs_host[iV][iWhich][3]),
				&(ionize_coeffs_host[iV][iWhich][4]));		
	};
	fclose(fp);
	printf("fopen rec_coeffs\n");

	if (fp == NULL) {
		printf("FILE OPEN FAILED - rec_coeffs.txt is probably not found.\n");
		SafeExit(10888); // we are going to need this
	};

	fp = fopen("rec_coeffs.txt", "rt");
	rewind(fp);
	for (int iV = 0; iV < 32; iV++)
	{
		for (int iWhich = 0; iWhich < 3; iWhich++)
			fscanf(fp, " %lf %lf %lf %lf %lf", &(recomb_coeffs_host[iV][iWhich][0]),
				&(recomb_coeffs_host[iV][iWhich][1]),
				&(recomb_coeffs_host[iV][iWhich][2]),
				&(recomb_coeffs_host[iV][iWhich][3]),
				&(recomb_coeffs_host[iV][iWhich][4]));
	};
	fclose(fp);

	printf("ionize_temps[8][3] %1.14E \n", ionize_temps_host[8][3]);
	printf("ionize_coeffs[11][4][2] %1.14E \n", ionize_coeffs_host[11][4][2]);
	printf("recomb_coeffs[28][1][3] %1.14E \n", recomb_coeffs_host[28][1][3]);
	//getch(); // test what we loaded
	Call(cudaMemcpyToSymbol(ionize_temps, ionize_temps_host, 32*10 * sizeof(f64)), 
		"cudaMemcpyToSymbol(ionize_temps)");
	Call(cudaMemcpyToSymbol(ionize_coeffs, ionize_coeffs_host, 32 * 5*5 * sizeof(f64)),
		"cudaMemcpyToSymbol(ionize_coeffs)");
	Call(cudaMemcpyToSymbol(recomb_coeffs, recomb_coeffs_host, 32 * 3*5 * sizeof(f64)),
		"cudaMemcpyToSymbol(recomb_coeffs)");


	f64_tens2 anticlock2;
	anticlock2.xx = cos(FULLANGLE);
	anticlock2.xy = -sin(FULLANGLE);
	anticlock2.yx = sin(FULLANGLE);
	anticlock2.yy = cos(FULLANGLE);
	Tensor2 * T2address;
	Call(cudaGetSymbolAddress((void **)(&T2address), Anticlockwise_d),
		"cudaGetSymbolAddress((void **)(&T2address),Anticlockwise)");
	Call(cudaMemcpy(T2address, &anticlock2, sizeof(f64_tens2), cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &anticlock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");
	// Note that objects appearing in constant memory must have empty constructor & destructor.

	f64_tens2 clock2;
	clock2.xx = cos(FULLANGLE);
	clock2.xy = sin(FULLANGLE);
	clock2.yx = -sin(FULLANGLE);
	clock2.yy = cos(FULLANGLE);
	Call(cudaGetSymbolAddress((void **)(&T2address), Clockwise_d),
		"cudaGetSymbolAddress((void **)(&T2address),Clockwise)");
	Call(cudaMemcpy(T2address, &clock2, sizeof(f64_tens2), cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &clock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");

	Set_f64_constant(kB, kB_); 
	Set_f64_constant(c, c_);
	Set_f64_constant(q, q_);
	Set_f64_constant(m_e, m_e_);
	Set_f64_constant(m_ion, m_ion_);
	Set_f64_constant(m_i, m_ion_);
	Set_f64_constant(m_n, m_n_);
	Set_f64_constant(eoverm, eoverm_);
	Set_f64_constant(qoverM, qoverM_);
	Set_f64_constant(moverM, moverM_);
	Set_f64_constant(qovermc, eovermc_);
	Set_f64_constant(qoverMc, qoverMc_);
	Set_f64_constant(FOURPI_Q_OVER_C, FOUR_PI_Q_OVER_C_);
	Set_f64_constant(FOURPI_Q, FOUR_PI_Q_);
	Set_f64_constant(FOURPI_OVER_C, FOURPI_OVER_C_);
	f64 one_over_kB_ = 1.0 / kB_;
	f64 one_over_kB_cubed_ = 1.0 / (kB_*kB_*kB_);
	f64 kB_to_3halves_ = sqrt(kB_)*kB_;
	Set_f64_constant(one_over_kB, one_over_kB_);
	Set_f64_constant(one_over_kB_cubed, one_over_kB_cubed_);
	Set_f64_constant(kB_to_3halves, kB_to_3halves_);
	Set_f64_constant(NU_EI_FACTOR, NU_EI_FACTOR_);
	Set_f64_constant(nu_eiBarconst, nu_eiBarconst_);
	Set_f64_constant(Nu_ii_Factor, Nu_ii_Factor_);
	
	f64 M_i_over_in_ = m_i_ / (m_i_ + m_n_);
	f64 M_e_over_en_ = m_e_ / (m_e_ + m_n_);
	f64	M_n_over_ni_ = m_n_ / (m_i_ + m_n_);
	f64	M_n_over_ne_ = m_n_ / (m_e_ + m_n_);
	f64	M_en_ = m_e_ * m_n_ / ((m_e_ + m_n_)*(m_e_ + m_n_));
	f64	M_in_ = m_i_ * m_n_ / ((m_i_ + m_n_)*(m_i_ + m_n_));
	f64	M_ei_ = m_e_ * m_i_ / ((m_e_ + m_i_)*(m_e_ + m_i_));
	f64	m_en_ = m_e_ * m_n_ / (m_e_ + m_n_);
	f64	m_ei_ = m_e_ * m_i_ / (m_e_ + m_i_);
	Set_f64_constant(M_i_over_in, M_i_over_in_);
	Set_f64_constant(M_e_over_en, M_e_over_en_);// = m_e / (m_e + m_n);
	Set_f64_constant(M_n_over_ni, M_n_over_ni_);// = m_n / (m_i + m_n);
	Set_f64_constant(M_n_over_ne, M_n_over_ne_);// = m_n / (m_e + m_n);
	Set_f64_constant(M_en, M_en_);
	Set_f64_constant(M_in, M_in_);
	Set_f64_constant(M_ei, M_ei_);
	Set_f64_constant(m_en, m_en_);
	Set_f64_constant(m_ei, m_ei_);
	
	// We are seriously saying that the rate of heat transfer e-n and e-i is
	// basically affected by factor m_e/m_n --- model document says so. ...

	Set_f64_constant(over_m_e, over_m_e_);
	Set_f64_constant(over_m_i, over_m_i_);
	Set_f64_constant(over_m_n, over_m_n_);

	f64 over_sqrt_m_ion_ = 1.0 / sqrt(m_i_);
	f64 over_sqrt_m_e_ = 1.0 / sqrt(m_e_);
	f64 over_sqrt_m_neutral_ = 1.0 / sqrt(m_n_);
	Set_f64_constant(over_sqrt_m_ion, over_sqrt_m_ion_);
	Set_f64_constant(over_sqrt_m_e, over_sqrt_m_e_);
	Set_f64_constant(over_sqrt_m_neutral, over_sqrt_m_neutral_);

	Set_f64_constant(RELTHRESH_AZ_d, RELTHRESH_AZ);

	Set_f64_constant(FRILL_CENTROID_OUTER_RADIUS_d, OutermostFrillCentroidRadius);
	Set_f64_constant(FRILL_CENTROID_INNER_RADIUS_d, InnermostFrillCentroidRadius);
	//f64 UNIFORM_n_temp = UNIFORM_n;
	Set_f64_constant(UNIFORM_n_d, UNIFORM_n);

	Call(cudaGetSymbolAddress((void **)(&f64address), m_e ), 
			"cudaGetSymbolAddress((void **)(&f64address), m_e )");
	Call(cudaMemcpy( f64address, &m_e_, sizeof(f64),cudaMemcpyHostToDevice),
			"cudaMemcpy( f64address, &m_e_, sizeof(f64),cudaMemcpyHostToDevice) src dest");
						
	f64 value = 1.25;
	f64 value2 = 1.5;
	Call(cudaMemcpyToSymbol(billericay, &value, sizeof(f64)), "bill the bat.");
	Call(cudaGetSymbolAddress((void **)(&f64address), billericay),"billericay1");
	Call(cudaMemcpy(f64address, &value, sizeof(f64), cudaMemcpyHostToDevice),"can we");
	Call(cudaMemcpy(&value2, f64address, sizeof(f64), cudaMemcpyDeviceToHost),"fdfdf");
	printf("value2 = %f\n",value2); // = 1.25

	// So this stuff DOES work
	// But debugger gives incorrect reading of everything as 0

	Call(cudaGetSymbolAddress((void **)(&f64address), m_e), "m_e");
	Call(cudaMemcpy(&value2, f64address, sizeof(f64), cudaMemcpyDeviceToHost), "fdfdf");
	printf("value2 = %1.8E\n", value2); // = m_e.
	// This was a total runaround.

	// four_pi_over_c_ReverseJz, EzStrength_d; // set at the time
	numReverseJzTriangles = numEndZCurrentTriangles_ - numStartZCurrentTriangles_;

	Call(cudaGetSymbolAddress((void **)(&longaddress), numStartZCurrentTriangles),
		"cudaGetSymbolAddress((void **)(&longaddress), numStartZCurrentTriangles)");
	Call(cudaMemcpy(longaddress, &numStartZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &numStartZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice)");
	Call(cudaGetSymbolAddress((void **)(&longaddress), numEndZCurrentTriangles),
		"cudaGetSymbolAddress((void **)(&longaddress), numEndZCurrentTriangles)");
	Call(cudaMemcpy(longaddress, &numEndZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &numEndZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice)");
	// stored so we can check if it's a triangle that has reverse Jz
	Call(cudaGetSymbolAddress((void **)(&longaddress), NumInnerFrills_d),
		"cudaGetSymbolAddress((void **)(&longaddress), NumInnerFrills_d)");
	Call(cudaMemcpy(longaddress, &NumInnerFrills, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &NumInnerFrills, sizeof(long), cudaMemcpyHostToDevice)");
//	Call(cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d),
//		"cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d)");
//	Call(cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice),
//		"cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice)");
// Cannot be used: FirstOuterFrill is not reliable with retiling.

	Call(cudaMemcpyToSymbol(cross_T_vals_d, cross_T_vals, 10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_T_vals_d,cross_T_vals, 10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d, cross_s_vals_viscosity_ni,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d,cross_s_vals_viscosity_ni, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d, cross_s_vals_viscosity_nn,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d,cross_s_vals_viscosity_nn, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_MT_ni_d, cross_s_vals_momtrans_ni,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_MT_ni_d,cross_s_vals_momtrans_ni, \
		10*sizeof(f64))");

	long temp0 = 0;
	Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
		"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
	Call(cudaMemcpy(longaddress, &temp0, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &temp0, sizeof(long), cudaMemcpyHostToDevice)");



	printf("1. cudaMallocs for d/dt arrays and main data\n");
	// 1. More cudaMallocs for d/dt arrays and main data:
	// and aggregation arrays...
	// ____________________________________________________
	
	CallMAC(cudaMalloc((void **)&p_sum_product_matrix3, numTilesMinor * sizeof(Symmetric3)));

	CallMAC(cudaMalloc((void **)&p_storeviscmove, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vie_modified_k, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_v_n_modified_k, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p__invcoeffself_x, NMINOR * sizeof(f64_tens2)));
	CallMAC(cudaMalloc((void **)&p__invcoeffself_y, NMINOR * sizeof(f64_tens2)));

	CallMAC(cudaMalloc((void **)&p_eps_against_deps2, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_blockTotal1, NMINOR * sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_blockTotal2, NMINOR * sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion4, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec4, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_neut4, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p__xzyzzxzy_i, NMINOR * sizeof(double4)));
	CallMAC(cudaMalloc((void **)&p__xzyzzxzy_e, NMINOR * sizeof(double4)));
	CallMAC(cudaMalloc((void **)&p__matrix_xy_i, NMINOR * sizeof(f64_tens2)));
	CallMAC(cudaMalloc((void **)&p__matrix_xy_e, NMINOR * sizeof(f64_tens2)));
	CallMAC(cudaMalloc((void **)&p__invmatrix, NMINOR * sizeof(f64_tens2)));
	CallMAC(cudaMalloc((void **)&p__coeffself_iz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p__coeffself_ez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p__invcoeffselfviz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p__invcoeffselfvez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p__invcoeffself, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressors2, NMINOR * sizeof(f64_vec2)*REGRESSORS));
	CallMAC(cudaMalloc((void **)&p_regressors_iz, NMINOR * sizeof(f64)*REGRESSORS));
	CallMAC(cudaMalloc((void **)&p_regressors_ez, NMINOR * sizeof(f64)*REGRESSORS));
	CallMAC(cudaMalloc((void **)&p_dump, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_dump2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_stored_move_neut_xy, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_stored_move_neut_z, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_stored_move_neut_xy2, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_stored_move_neut_z2, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_iRing, NMINOR * sizeof(int)));

	CallMAC(cudaMalloc((void**)&ionmomflux_eqns, sizeof(double) * 3*3 * EQNS_TOTAL* EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&elecmomflux_eqns, sizeof(double) * 3*3 * EQNS_TOTAL* EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&d_B, sizeof(double) * 4*EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&d_Ipiv512, sizeof(int) * 4*EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&p_RHS, sizeof(f64) * 4 * EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&d_info, sizeof(int) * 4 * EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&p_eqns, sizeof(double) * 4 * EQNS_TOTAL * 4 * EQNS_TOTAL));
	CallMAC(cudaMalloc((void**)&p_eqns2, sizeof(double) * 4 * EQNS_TOTAL * 4 * EQNS_TOTAL));

	CallMAC(cudaMalloc((void**)&p_selectflag, sizeof(bool)*NMINOR));
	CallMAC(cudaMalloc((void**)&p_equation_index, sizeof(short) * NMINOR));
	
	CallMAC(cudaMalloc((void **)&NT_addition_tri_d2, 3* NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_tri_d3, 3* NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_2, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_3, NMINOR * sizeof(NTrates)));

	CallMAC(cudaMalloc((void **)&p_ita_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ita_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ita_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_nu_n, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&v4temparray, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&zero_vec4, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_neut3, NMINOR * sizeof(f64_vec3)));
	cudaMemset(zero_vec4, 0, sizeof(v4)*NMINOR);
	CallMAC(cudaMalloc((void **)&p_prev_move3, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_Jacobian_list, NMINOR * SQUASH_POINTS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_indicator, NMINOR * sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_AAdot_target, NMINOR * sizeof(AAdot)));
	CallMAC(cudaMalloc((void **)&p_AAdot_start, NMINOR * sizeof(AAdot)));
	CallMAC(cudaMalloc((void **)&p_v_n_target, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_v_n_start, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vie_target, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vie2, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vie_start, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vie_save, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_regrlc2, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_regrlc_iz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regrlc_ez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_eps_against_deps2, NMINOR * sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_Residuals, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_pressureflag, NUMVERTICES * sizeof(bool)));
	
	CallMAC(cudaMalloc((void **)&p_eps_against_deps, NMINOR * sizeof(f64_vec3)*REGRESSORS*3));
	CallMAC(cudaMalloc((void **)&p_eps_against_d_eps, NMINOR * sizeof(f64)*REGRESSORS * 3));
	
	CallMAC(cudaMalloc((void **)&p_sum_product_matrix, numTilesMinor * sizeof(f64)*REGRESSORS*REGRESSORS*3));

	CallMAC(cudaMalloc((void **)&p_MAR_ion_pressure_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_pressure_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion_visc_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_visc_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_ionization_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vie_k_stored, NUMVERTICES * sizeof(v4)));


	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_x, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_y, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_z, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_epsilon3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&v3temp, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&zero_vec3, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_place_contribs, NMINOR*6 * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_regressor_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_e, NMINOR * sizeof(f64))); // only need NUMVERTICES but we reused.

	CallMAC(cudaMalloc((void **)&p_sqrtfactor, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_AreaMinor_cc, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_store_NTFlux, NUMVERTICES * MAXNEIGH * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&sz_who_vert_vert, NUMVERTICES * MAXNEIGH * sizeof(short)));

	CallMAC(cudaMalloc((void **)&p_store_T_move1, NUMVERTICES * sizeof(T3)));
	CallMAC(cudaMalloc((void **)&p_store_T_move2, NUMVERTICES * sizeof(T3)));

	CallMAC(cudaMalloc((void **)&p_temp3_1, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_temp3_2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_temp3_3, NMINOR * sizeof(f64_vec3)));
	
	CallMAC(cudaMalloc((void **)&p_matrix_blocks, SQUASH_POINTS*SQUASH_POINTS * numTilesMinor*2 * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_vector_blocks, SQUASH_POINTS* numTilesMinor *2* sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_Tn, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ti, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Te, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NnTn, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NTi, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NTe, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_e, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&zero_array, NUMVERTICES * sizeof(T3)));

	CallMAC(cudaMalloc((void **)&p_regressors, NUMVERTICES * (REGRESSORS + 1) * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_x8, numTilesMinor * REGRESSORS * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_regressors4, NMINOR*REGRESSORS * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_stored_move4, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_tempvec4, NMINOR * sizeof(v4)));

	CallMAC(cudaMalloc((void **)&p_regressors3, NMINOR*REGRESSORS * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_tempvec3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_SS, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_x, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_y, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_z, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_stored_move3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_stored_move3_2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_x_, NMINOR*REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_y_, NMINOR*REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_z_, NMINOR*REGRESSORS * sizeof(f64)));



	CallMAC(cudaMalloc((void **)&p_Effect_self_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_e, NUMVERTICES * sizeof(f64)));
	 
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_e, NUMVERTICES * sizeof(f64)));
		
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_n, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_i, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_e, NUMVERTICES * MAXNEIGH * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_boolarray, 2*NUMVERTICES * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_boolarray2, 3 * NUMVERTICES * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_boolarray_block, numTilesMinor * sizeof(bool)));

	CallMAC(cudaMalloc((void **)&p_nu_major, NUMVERTICES * sizeof(species3)));
	CallMAC(cudaMalloc((void **)&p_was_vertex_rotated, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_triPBClistaffected, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_T_upwind_minor_and_putative_T, NMINOR * sizeof(T3)));
	
	CallMAC(cudaMalloc((void **)&p_v0, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vn0, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_sigma_Izz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_OhmsCoeffs, NMINOR * sizeof(OhmsCoeffs)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_heat, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_heat, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ax, NUMVERTICES *REGRESSORS * sizeof(f64))); // sometimes use as NMINOR
	 
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_bFailed, NMINOR * sizeof(bool)));
		
	CallMAC(cudaMalloc((void **)&p_Jacobi_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&stored_Az_move, NMINOR * sizeof(f64)));


	CallMAC(cudaMalloc((void **)&p_MAR_neut, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_Az, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_AzNext, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapAz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapAzNext, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapCoeffself, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapJacobi, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_x, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Azdot0, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_gamma, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Integrated_div_v_overall, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v_neut, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v_overall, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzdotduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_GradAz, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_GradTe, NMINOR * sizeof(f64_vec2)));

	CallMAC(cudaMalloc((void **)&p_one_over_n, NMINOR * sizeof(nvals)));
	CallMAC(cudaMalloc((void **)&p_one_over_n2, NMINOR * sizeof(nvals)));

	CallMAC(cudaMalloc((void **)&p_kappa_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_kappa_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_kappa_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_nu_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_nu_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_n_shards, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&p_n_shards_n, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_tri_d, 3* NMINOR * sizeof(NTrates)));
	 
	CallMAC(cudaMalloc((void **)&p_coeff_of_vez_upon_viz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_beta_ie_z, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_longtemp, NMINOR*2* sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_temp1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp6, NMINOR * sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_graphdata1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata6, NMINOR * sizeof(f64)));

	for (i = 0; i < 9; i++)
		CallMAC(cudaMalloc((void **)&p_Tgraph[i], NUMVERTICES * sizeof(f64)));
	for (i = 0; i < 12; i++)
		CallMAC(cudaMalloc((void **)&p_accelgraph[i], NUMVERTICES * sizeof(f64)));
	for (i = 0; i < 20; i++)
		CallMAC(cudaMalloc((void **)&p_Ohmsgraph[i], NUMVERTICES * sizeof(f64)));
	for (i = 0; i < 12; i++)
		CallMAC(cudaMalloc((void **)&p_arelz_graph[i], NUMVERTICES * sizeof(f64)));
	

	CallMAC(cudaMalloc((void **)&p_MAR_ion_temp_central, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_temp_central, NUMVERTICES * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_bool, NMINOR * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_denom_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_denom_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_n, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_i, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_e, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_neut2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_ROCMAR1, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_ROCMAR2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_ROCMAR3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_temp, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_temp2, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&p_epsilon_xy, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_epsilon_iz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_ez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_n, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_e, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_e, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_d_epsxy_by_d_beta_i, NMINOR * REGRESSORS * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_d_eps_iz_by_d_beta_i, NMINOR * REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_ez_by_d_beta_i, NMINOR * REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_epsxy_by_d_beta_e, NMINOR * REGRESSORS * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_d_eps_iz_by_d_beta_e, NMINOR * REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_ez_by_d_beta_e, NMINOR * REGRESSORS * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&store_heatcond_NTrates, NUMVERTICES * sizeof(NTrates)));

	CallMAC(cudaMalloc((void **)&p_Selectflag, NMINOR * sizeof(int)));
	CallMAC(cudaMalloc((void **)&p_SelectflagNeut, NMINOR * sizeof(int)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_i, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_e, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_i_times_i, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_e_times_e, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_e_times_i, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_J, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_R, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_J_times_J, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_R_times_R, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_J_times_R, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_n_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_i_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_e_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_n_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_i_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_e_x4, NMINOR * sizeof(f64_vec4)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_J_x4, numTilesMinor*sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_R_x4, numTilesMinor * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_8x8, numTilesMinor * sizeof(f64) *  REGRESSORS*REGRESSORS));
	
	p_eqns_host = (f64 *)malloc(sizeof(f64) * 4 * 4 * EQNS_TOTAL*EQNS_TOTAL);

	p_sum_product_matrix_host3 = (Symmetric3 *)malloc(numTilesMinor * sizeof(Symmetric3));

	p_matrix_blocks_host = (f64 *)malloc(SQUASH_POINTS*SQUASH_POINTS * numTilesMinor * 2 * sizeof(f64));
	p_vector_blocks_host = (f64 *)malloc(SQUASH_POINTS* numTilesMinor * 2 * sizeof(f64));

	p_tempvec3host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_tempvec4_host = (v4 *)malloc(NMINOR * sizeof(v4));
	p_tempvec4_2_host = (v4 *)malloc(NMINOR * sizeof(v4));
	p_tempvec2_host = (f64_vec2 *)malloc(NMINOR * REGRESSORS* sizeof(f64_vec2));
	p_tempvec2_host2 = (f64_vec2 *)malloc(NMINOR * REGRESSORS * sizeof(f64_vec2));
	p_tempvec2_host3 = (f64_vec2 *)malloc(NMINOR * REGRESSORS * sizeof(f64_vec2));

	p_SS_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_sum_product_matrix_host = (f64 *)malloc(numTilesMinor * sizeof(f64)*REGRESSORS*REGRESSORS*3);
	p_eps_against_deps_host = (f64_vec3 *)malloc(numTilesMinor * sizeof(f64_vec3)*REGRESSORS*3); //*3 is gratuitous in case we re-use.
	p_eps_against_d_eps_host = (f64 *)malloc(numTilesMinor*sizeof(f64)*REGRESSORS*3); //*3 is gratuitous

	p_sum_eps_deps_by_dbeta_J_x4_host = (f64_vec4 *) malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_eps_deps_by_dbeta_R_x4_host = (f64_vec4 *)malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_depsbydbeta_8x8_host = (f64 *)malloc(numTilesMinor * REGRESSORS*REGRESSORS * sizeof(f64));

	p_sum_eps_deps_by_dbeta_x8_host = (f64 *)malloc(numTilesMinor*REGRESSORS*sizeof(f64));
	p_GradTe_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_GradAz_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_B_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));

	p_longtemphost = (long *)malloc(NMINOR*2 * sizeof(long));
	p_temphost1 = (f64 *)malloc(NMINOR *REGRESSORS* sizeof(f64)); // changed for debugging
	p_temphost2 = (f64 *)malloc(NMINOR * REGRESSORS * sizeof(f64)); // changed for debugging
	p_temphost3 = (f64 *)malloc(NMINOR * REGRESSORS * sizeof(f64));
	p_temphost4 = (f64 *)malloc(NMINOR * REGRESSORS * sizeof(f64));
	p_temphost5 = (f64 *)malloc(NMINOR * REGRESSORS * sizeof(f64));
	p_temphost6 = (f64 *)malloc(NMINOR * REGRESSORS * sizeof(f64));

	p_graphdata1_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata2_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata3_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata4_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata5_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata6_host = (f64 *)malloc(NMINOR * sizeof(f64));

	for (i = 0; i < 9; i++)
		p_Tgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64));
	for (i = 0; i < 12; i++)
		p_accelgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64)); // 3.6 MB
	for (i = 0; i < 20; i++)
		p_Ohmsgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64)); 
	for (i = 0; i < 12; i++)
		p_arelz_graph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64));

	p_boolhost = (bool *)malloc(NMINOR * sizeof(bool));
	p_longtemphost2 = (long *)malloc(NMINOR * sizeof(long));
	p_sum_vec_host = (f64_vec3 *)malloc(numTilesMinor * sizeof(f64_vec3));
	p_inthost = (int *)malloc(NMINOR * sizeof(int));

	if (p_temphost6 == 0) { printf("p6 == 0"); }
	else { printf("p6 != 0"); };
	temp_array_host = (f64 *)malloc(NMINOR * sizeof(f64));

	p_NTrates_host = (NTrates *)malloc(NMINOR * sizeof(NTrates));

	p_OhmsCoeffs_host = (OhmsCoeffs *)malloc(NMINOR * sizeof(OhmsCoeffs));

	p_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_initial_host = (f64 *)malloc(numTilesMinor * sizeof(f64));

	p_sum_eps_deps_by_dbeta_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_sq_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_eps_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	// Cannot see that I have ever yet put in anywhere to free this memory.

	p_sum_eps_deps_by_dbeta_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_sq_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_eps_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	
	p_sum_eps_deps_by_dbeta_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_deps_by_dbeta_e_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_i_times_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_e_times_e_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_e_times_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));

	p_sum_eps_deps_by_dbeta_J_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_deps_by_dbeta_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_J_times_J_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_R_times_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_J_times_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	
	iRing = (int *)malloc(NMINOR * sizeof(int));
	bSelected = (bool *)malloc(NMINOR * sizeof(bool));
	p_equation_index_host = (short *)malloc(NMINOR * sizeof(short));

	printf("2. cudaMemcpy system state from host\n");

	// 2. cudaMemcpy system state from host: this happens always
	// __________________________________________________________
	 
	// Note that we do always need an intermediate system on the host because
	// cudaMemcpy is our MO.
	pX_host->SendToDevice(cuSyst1);
	cuSyst2.CopyStructuralDetailsFrom(cuSyst1);
	cuSyst3.CopyStructuralDetailsFrom(cuSyst1);
	// Any logic to this?
	// Why not make a separate object containing the things that stay the same between typical runs?
	// ie, what is a neighbour of what.
	// info contains both pos and flag so that's not constant under advection; only neigh lists are.
	 
	printf("Done main cudaMemcpy to video memory.\n");
	  
	// Set up kernel L1/shared:
	  
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); // default!
	 
	cudaFuncSetCacheConfig(kernelCreateShardModelOfDensities_And_SetMajorArea,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelAdvanceDensityAndTemperature,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelPopulateBackwardOhmsLaw,
		cudaFuncCachePreferL1); 
	cudaFuncSetCacheConfig(kernelCalculate_ita_visc, 
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelComputeJacobianValues,
		cudaFuncCachePreferL1);

	// check some others here for speed.


	pX1 = &cuSyst1;
	pX_half = &cuSyst2;
	pX2 = &cuSyst3;

	printf("during Invoke_Populate:\n");
	Call(cudaGetSymbolAddress((void **)(&f64address), m_i), "m_i");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_i = %1.10E \n", value);
	Call(cudaGetSymbolAddress((void **)(&f64address), m_e), "m_e");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_e = %1.10E \n", value);

	/*
	CallMAC(cudaMemset(p_summands, 0, sizeof(f64)*numTilesMinor));
	Kernel_GetZCurrent << <numTilesMinor, threadsPerTileMinor >> >(
	pX1->p_tri_perinfo,
	pX1->p_nT_ion_minor,
	pX1->p_nT_elec_minor,
	pX1->p_v_ion,
	pX1->p_v_elec, // Not clear if this should be nv or {n,v} yet - think.
	pX1->p_area_minor,
	p_summands
	);
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize GetZCurrent 1.");

	CallMAC(cudaMemcpy(p_summands_host, p_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost));
	Iz0 = 0.0;
	for (int ii = 0; ii < numTilesMinor; ii++)
	{
	Iz0 += p_summands_host[ii];
	};
	printf("Iz X1 before area calc %1.14E \n", Iz0); // == 0.0 since areas = 0
	*/

	//pX1->PerformCUDA_Advance(&pX2, &pX_half);

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}

// Sequence:
// _________
// 
// Once: call Invoke routine
// .. and send host system to device

// ... do advance step x 10     [1 advection cycle = 1e-11?]
// ... send back and display & save (1e-11 ... 5s per ns)
// ... do advance step x 10     [1 advection cycle = 1e-11?]
// ... send back and display & save
// ... 
// ... send back and display & save;

// ...               Re-Delaunerize (1e-10)
// ...         send to device

//
// Once: revoke all
 
long iHistoryAz;

void PerformCUDA_RunStepsAndReturnSystem(cuSyst * pX_host)
{
	float elapsedTime;
	static cuSyst * pX1 = &cuSyst1;
	static cuSyst * pX2 = &cuSyst3;    // remember which way round - though with an even number of steps it's given we get back
	static cuSyst * pX_half = &cuSyst2; 
	cuSyst * pXtemp;
	// So let's be careful here

	structural info;

//	cudaMemcpy(&info, &(cuSyst1.p_info[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
//	printf("%d info: flag %d tri_len %d  \n", VERTCHOSEN, info.flag, info.neigh_len);

	iHistoryAz = 0;

	bViscousHistory = false; // it might sometimes be true but hey
	
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	cudaEventSynchronize(start1);

	kernelSetPressureFlag << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info,
		pX1->p_izTri_vert,
		p_pressureflag
		); // 0 for those that are next to CROSSING_CATH
	Call(cudaThreadSynchronize(), "cudaTS kernelSetPressureFlag");

	kernelCreateWhoAmI_verts << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info + BEGINNING_OF_CENTRAL,
		pX1->p_izNeigh_vert,
		sz_who_vert_vert // array of MAXNEIGH shorts for each vertex.
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreateWhoAmI_verts");
	

	long iSubstep;
	int i;
	// Ultimately this 10 steps .. so 1e-11? .. can be 1 advective step.

	// B is set for pX_half and pX1. So take pX_half value and spit it to host.
	for (i = 0; i < 12; i++)
		cudaMemset(p_accelgraph[i], 0, sizeof(f64)*NUMVERTICES);
	for (i = 0; i < 12; i++)
		cudaMemset(p_arelz_graph[i], 0, sizeof(f64)*NUMVERTICES);

	pX_half->CopyStructuralDetailsFrom(*pX1);
	pX2->CopyStructuralDetailsFrom(*pX1);

	fp_dbg = fopen("dbg1.txt", "a");
	fp_trajectory = fopen("traj.txt", "a");
	
	for (int iRepeat = 0; iRepeat < ADVECT_STEPS_PER_GPU_VISIT; iRepeat++) 
	{	
		printf("Advection step:\n"); 
		bGlobalSaveTGraphs == true;
		pX1->PerformCUDA_AdvectionCompressionInstantaneous(TIMESTEP*(real)ADVECT_FREQUENCY, pX2, pX_half);

		// We need to smoosh the izTri etc data on to the pXhalf and new pX dest systems
		// as it doesn't update any other way and this data will be valid until renewed
		// in the dest system at the end of the step -- riiiiiiiight?

		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;
		 
		pX_half->CopyStructuralDetailsFrom(*pX1);
		pX2->CopyStructuralDetailsFrom(*pX1);	
		cudaMemcpy(pX2->p_AreaMinor, pX1->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX_half->p_AreaMinor, pX1->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
				
		bGlobalSaveTGraphs = false;
		for (iSubstep = 0; iSubstep < ADVECT_FREQUENCY; iSubstep++)
		{
			printf("\nSTEP %d\n-------------\n", iSubstep);
			printf("evaltime = %1.10E \n\n", evaltime);
			if (iSubstep == ADVECT_FREQUENCY-1) bGlobalSaveTGraphs = true;
			pX1->PerformCUDA_Advance_noadvect(pX2, pX_half);

			pXtemp = pX1;
			pX1 = pX2;
			pX2 = pXtemp;
			
			/*
					cudaMemcpy(pX_half->p_izTri_vert, pX1->p_izTri_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_izTri_vert, pX1->p_izTri_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_izNeigh_vert, pX1->p_izNeigh_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_izNeigh_vert, pX1->p_izNeigh_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_szPBCtri_vert, pX1->p_szPBCtri_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_szPBCtri_vert, pX1->p_szPBCtri_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_szPBCneigh_vert, pX1->p_szPBCneigh_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_szPBCneigh_vert, pX1->p_szPBCneigh_vert,
						NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_izNeigh_TriMinor, pX1->p_izNeigh_TriMinor,
						NUMTRIANGLES*6 * sizeof(long), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_izNeigh_TriMinor, pX1->p_izNeigh_TriMinor,
						NUMTRIANGLES * 6 * sizeof(long), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
						NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX2->p_szPBC_triminor, pX1->p_szPBC_triminor,
						NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);

					cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
						NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);
					cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
						NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);*/
		};

		
	}



	// After an even number of goes, pX1 = &cuSyst1 and this is where we ended up.
		
	fclose(fp_dbg);
	fclose(fp_trajectory);

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsedTime, start1, stop1);
	printf("Elapsed time for %d steps : %f ms\n", GPU_STEPS, elapsedTime);

	// update with the most recent B field since we did not update it properly after subcycle:
	cudaMemcpy(pX1->p_B, pX_half->p_B, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	// For graphing :
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

	printf("Graphing data passed: %d : Lap %1.9E ; %d : Lap %1.9E \n",
		VERTCHOSEN, temp_array_host[VERTCHOSEN + BEGINNING_OF_CENTRAL], VERTCHOSEN2, temp_array_host[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]);
	

	cudaMemcpy(p_OhmsCoeffs_host, p_OhmsCoeffs, sizeof(OhmsCoeffs)*NMINOR, cudaMemcpyDeviceToHost);
	
	pX1->SendToHost(*pX_host);
	
	// Now store for 1D graphs: temphost3 = n, temphost4 = vr, temphost5 = vez

	// This is where we think to fill in temphost1 = nu_ei_effective + nu_en_MT
	// temphost2 = nu_en_MT / temphost1.

	kernelPrepareNuGraphs << <numTilesMinor, threadsPerTileMinor >> > (
		pX1->p_info,
		pX1->p_n_minor,
		pX1->p_T_minor,
		p_temp1,
		p_temp2
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelPrepareNuGraphs");
	cudaMemcpy(p_temphost1 , p_temp1, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2 , p_temp2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);



	// Get some graph data:

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		TIMESTEP,
		pX1->p_info,
		pX1->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		pX1->p_n_major,		
		pX1->p_AreaMajor,

		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,

		pX1->p_vie + BEGINNING_OF_CENTRAL,
		pX1->p_v_n + BEGINNING_OF_CENTRAL,

		pX2->p_T_minor + BEGINNING_OF_CENTRAL,
		false
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelIonisationRates");

	kernelPrepareIonizationGraphs << <numTilesMajor, threadsPerTileMajor >> >(
		pX1->p_info + BEGINNING_OF_CENTRAL,
		pX1->p_n_major,
		pX1->p_AreaMajor,
		NT_addition_rates_d, // dN/dt, dNeTe/dt
		p_temp3_3, // --> d/dt v_e

		p_graphdata1, p_graphdata2, p_graphdata3, p_graphdata4, p_graphdata5, p_graphdata6
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelPrepareIonizationGraphs");
	cudaMemcpy(p_graphdata1_host, p_graphdata1, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata2_host, p_graphdata2, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata3_host, p_graphdata3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata4_host, p_graphdata4, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata5_host, p_graphdata5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata6_host, p_graphdata6, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	
	kernelCollectOhmsGraphs << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info + BEGINNING_OF_CENTRAL,
		
		p_MAR_ion_pressure_major_stored,
		p_MAR_ion_visc_major_stored,
		p_MAR_elec_pressure_major_stored,  // need to distinguish viscous from pressure part.
		p_MAR_elec_visc_major_stored,
		p_MAR_elec_ionization_major_stored,

		pX1->p_B + BEGINNING_OF_CENTRAL,

		p_vie_k_stored, // ALL MAJOR
		pX1->p_vie + BEGINNING_OF_CENTRAL, // k+1

		p_GradTe + BEGINNING_OF_CENTRAL, // stored?
		pX1->p_n_minor + BEGINNING_OF_CENTRAL,
		pX1->p_T_minor + BEGINNING_OF_CENTRAL,

		pX1->p_AAdot + BEGINNING_OF_CENTRAL,
		pX1->p_AreaMinor, // EXCEPT THIS ONE
		p_Ohmsgraph[0], // elastic effective frictional coefficient zz
		p_Ohmsgraph[1], // ionization effective frictional coefficient zz
		p_Ohmsgraph[2], // 2 is combined y pressure accel rate
		p_Ohmsgraph[3],// 3 is q/(M+m) Ez -- do we have
		p_Ohmsgraph[4], // 4 is thermal force accel
		
		p_Ohmsgraph[5], // T_zy
		p_Ohmsgraph[6], // T_zz

		p_Ohmsgraph[7], // T acting on pressure
		p_Ohmsgraph[8], // T acting on electromotive
		p_Ohmsgraph[9], // T acting on thermal force
		p_Ohmsgraph[10], // prediction vez-viz

		p_Ohmsgraph[11], // difference of prediction from vez_k
		p_Ohmsgraph[12], // progress towards eqm: need vez_k+1
		p_Ohmsgraph[13], // viscous acceleration of electrons and ions (z)
		p_Ohmsgraph[14], // Prediction of Jz
		p_Ohmsgraph[15], // sigma zy
		p_Ohmsgraph[16], // sigma zz
		p_Ohmsgraph[17], // sigma zz times electromotive 
		p_Ohmsgraph[18] // Difference of prediction from Jz predicted.
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCollectOhmsGraphs");

	for (i = 0; i < 9; i++)
		cudaMemcpy(p_Tgraph_host[i], p_Tgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (i = 0; i < 12; i++)
		cudaMemcpy(p_accelgraph_host[i], p_accelgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (i = 0; i < 20; i++)
		cudaMemcpy(p_Ohmsgraph_host[i], p_Ohmsgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (i = 0; i < 12; i++)
		cudaMemcpy(p_arelz_graph_host[i], p_arelz_graph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);


	f64 Integral_Azdotdot = 0.0;
	f64 Integral_fabsAzdotdot = 0.0;
	f64 Integral_Azdot = 0.0;
	f64 Integral_fabsAzdot = 0.0;
	for (long i = 0; i < NMINOR; i++)
	{
		p_temphost3[i] = pX_host->p_AAdot[i].Azdot;
		p_temphost5[i] = temp_array_host[i]; 
		p_temphost6[i] = -FOURPIOVERC_*q_*pX_host->p_n_minor[i].n*
			(pX_host->p_vie[i].viz - pX_host->p_vie[i].vez);
		p_temphost4[i] = c_*c_*(temp_array_host[i] - p_temphost6[i]);
		
		Integral_Azdotdot += p_temphost4[i] * pX_host->p_AreaMinor[i];
		Integral_fabsAzdotdot += fabs(p_temphost4[i] * pX_host->p_AreaMinor[i]);
		Integral_Azdot += p_temphost3[i] * pX_host->p_AreaMinor[i];
		Integral_fabsAzdot += fabs(p_temphost3[i] * pX_host->p_AreaMinor[i]);
	}
	printf("Integral Azdotdot %1.10E fabs %1.10E \n", Integral_Azdotdot, Integral_fabsAzdotdot);
	printf("Integral Azdot %1.10E fabs %1.10E \n", Integral_Azdot, Integral_fabsAzdot);
	
	// Here we go .. Azdotdot doesn't say anything sensible because we are missing the +Jz contrib.
	// Azdot however doesn't show near 0 either.
	// That is bad.
	// Next: We should find that the actual change in Azdot sums to zero.
	// We should also therefore be finding that Azdot sums to zero.
//
//	FILE * fpaccel = fopen("accel.txt", "w");
//	for (i = 0; i < NUMVERTICES; i++)
//		fprintf(fpaccel, "%d %1.10E \n", i, p_accelgraph_host[10][i]);
//	fclose(fpaccel);
//	printf("FP ACCEL PRINTED");
//	getch(); getch(); getch();
	 
}

f64_vec3 *vn_compare;
  
void DebugNaN(cuSyst * p_cuSyst)
{
	p_cuSyst->SendToHost(cuSyst_host);
	bool bSwitch = 0;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		if (cuSyst_host.p_AAdot[iMinor].Azdot != cuSyst_host.p_AAdot[iMinor].Azdot)
		{
			printf("Nan %d Azdot", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_AAdot[iMinor].Az != cuSyst_host.p_AAdot[iMinor].Az)
		{
			printf("Nan %d Az", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_n_minor[iMinor].n != cuSyst_host.p_n_minor[iMinor].n)
		{
			printf("Nan %d n", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_T_minor[iMinor].Te != cuSyst_host.p_T_minor[iMinor].Te)
		{ 
			printf("Nan %d Te", iMinor);
			bSwitch = 1;
		}  
		if (cuSyst_host.p_vie[iMinor].vez != cuSyst_host.p_vie[iMinor].vez)
		{ 
			printf("Nan %d vez", iMinor);
			bSwitch = 1;
		} 
		if (cuSyst_host.p_vie[iMinor].viz != cuSyst_host.p_vie[iMinor].viz)
		{
			printf("Nan %d viz", iMinor);
			bSwitch = 1;
		} 
		if ((cuSyst_host.p_T_minor[iMinor].Te < 0.0)
			|| (cuSyst_host.p_T_minor[iMinor].Te > 1.0e-8)) { // thermal velocity 3e9 = 0.1c
			printf("Te = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Te, iMinor);
			bSwitch = 1;
		}
		if ((cuSyst_host.p_T_minor[iMinor].Ti < 0.0)
			|| (cuSyst_host.p_T_minor[iMinor].Ti > 1.0e-7)) { // thermal velocity 2e8
			printf("Ti = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Ti, iMinor);
			bSwitch = 1;
		}
	}; 
	if (bSwitch) {
		SafeExit(11874);
	}
	else {
		printf("\nDebugNans OK\n");
	}
}



void cuSyst::PerformCUDA_AdvectionCompressionInstantaneous(//const 
	f64 const Timestep,
	cuSyst * pX_target,
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles, iVertex;
	FILE * fp;
	long iMinor, iSubstep;
	f64 Iz_prescribed;
	static long runs = 0;
	float elapsedTime;
	f64 Iz_k, Iz_prescribed_endtime;
	f64_vec2 temp_vec2;
	// DEBUG:
	f64 sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot;
	FILE * fp_2;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	cudaEvent_t start, stop, middle;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&middle);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

	SetConsoleTextAttribute(hConsole, 14);

#define USE_N_MAJOR_FOR_VERTEX 
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d vn.y %1.9E vy %1.9E", VERTCHOSEN, tempf64, tempb);

		nvals n;
		cudaMemcpy(&n, &(this->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
		printf("%d n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	}

	if (!GlobalSuppressSuccessVerbosity) {
		long izTri[MAXNEIGH];

		structural info;
		cudaMemcpy(&info, &(this->p_info[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
		printf("info: flag %d tri_len %d  \n", info.flag, info.neigh_len);

		cudaMemcpy(izTri, this->p_izTri_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d tri %d : %d\n", VERTCHOSEN, i, izTri[i]);

		long izNeigh[MAXNEIGH];
		cudaMemcpy(izNeigh, this->p_izNeigh_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d neigh %d : %d\n", VERTCHOSEN, i, izNeigh[i]);

		LONG3 cornerindex;
		cudaMemcpy(&cornerindex, this->p_tri_corner_index + izTri[0], sizeof(LONG3), cudaMemcpyDeviceToHost);

		for (i = 0; i < 3; i++)
			printf("%d corner 012 : %d %d %d\n", izTri[0], cornerindex.i1, cornerindex.i2, cornerindex.i3);

		nvals n2;
		cudaMemcpy(&n2, &(this->p_n_major[VERTCHOSEN]),
			sizeof(nvals), cudaMemcpyDeviceToHost);
		printf("evaltime %1.8E n %1.8E n_n %1.8E ", evaltime, n2.n, n2.n_n);

		f64_vec3 B;
		cudaMemcpy(&B, &(this->p_B[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("B %1.8E %1.8E %1.8E ", B.x, B.y, B.z);

		T3 T;
		cudaMemcpy(&T, &(this->p_T_minor[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(T3), cudaMemcpyDeviceToHost);
		printf("T %1.8E %1.8E %1.8E \n", T.Tn, T.Ti, T.Te);

		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("v_n [%d] %1.9E %1.9E %1.9E ", VERTCHOSEN, tempvec3.x, tempvec3.y, tempvec3.z);

		v4 tempv4;
		cudaMemcpy(&tempv4, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(v4), cudaMemcpyDeviceToHost);
		printf("v_xy [%d] %1.9E %1.9E viz %1.9E vez %1.9E ", VERTCHOSEN, tempv4.vxy.x,
			tempv4.vxy.y, tempv4.viz, tempv4.vez);

		f64 tempf64;
		cudaMemcpy(&tempf64, &(this->p_AAdot[VERTCHOSEN + BEGINNING_OF_CENTRAL].Azdot),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("Azdot %1.10E \n", tempf64);


	}
	//char ch;
	//cudaMemcpy(&ch, &(this->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("this holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_half->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_half holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_target->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_target holds char %d", ch);
	//getch();

	// Comes in with triangle centres on insulator: bug.
	// We need to set triangle positions before calling AreaMinorFluid.



	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false 
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelGet_AreaMinorFluid << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		p_pressureflag,
		this->p_AreaMinor // output
		);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");
	 
	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_n_major,
		this->p_n_minor,  // DESIRED VALUES
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		p_n_shards,
		p_n_shards_n,
		this->p_AreaMajor,
		false // USE CENTROIDS
		);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels this");
	 
	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");
	
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		Timestep // note this is used for determining distance.
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");

	// Infer tri velocities based on actual moves of verts:
	kernelCentroidVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> > (
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		this->p_v_overall_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles");

	// Move tri positions:
	kernelAdvectPositionsTris << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,
		this->p_info,
		pX_half->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");

	// Set AreaMinor:
	
	// Ideally we would reset triangle centroids first but this won't be a lot different.

	kernelGet_AreaMinorFluid << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		p_pressureflag,
		pX_half->p_AreaMinor // output
		);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");
	 
	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	cudaMemset(p_store_NTFlux, 0, sizeof(NTrates)*NUMVERTICES*MAXNEIGH);

	if (!DEFAULTSUPPRESSVERBOSITY)
	{		
		cudaMemcpy(&tempf64, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);		 
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	kernelAccumulateAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_vert,   // we never pick up position from it so don't need per flag?
		sz_who_vert_vert,

		this->p_n_major, // unused?
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_vie,
		this->p_v_overall_minor,
		
		p_n_shards,

		NT_addition_rates_d,
		p_Div_v,
		p_Integrated_div_v_overall,
		p_store_NTFlux
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");
	kernelAccumulateNeutralAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_vert,
		sz_who_vert_vert,
		 
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		 
		this->p_v_n,
		this->p_v_overall_minor,

		p_n_shards_n,

		NT_addition_rates_d,
		p_Div_v_neut,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRateNeutral");
	  
	kernelAddStoredNTFlux << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info + BEGINNING_OF_CENTRAL,
		p_store_NTFlux,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS AddStoredNTFlux");

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	DivideNeTe_by_N << <numTilesMajor, threadsPerTileMajor >> >(
		NT_addition_rates_d,
		this->p_AreaMajor,
		this->p_n_major,
		p_Tgraph[6]);
	Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");


	////////////////////////////////
	cudaMemset(
		NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_vie,
		this->p_v_overall_minor,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards,
		this->p_n_minor,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor");
	 
	kernelNeutral_momflux << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_v_n,
		p_n_shards_n,
		this->p_n_minor,

		this->p_v_overall_minor,
		p_MAR_neut,

		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_momflux");

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");
	 
	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");



	long i;
	

	if (!DEFAULTSUPPRESSVERBOSITY)
	{ 
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			this->p_info,
			this->p_vie,
			this->p_v_n,
			this->p_T_minor,
			this->p_tri_neigh_index,
			this->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

	//cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2); // don't know why this was here but it doesn't do anything.
	
	kernelAdvanceDensityAndTemperature_nosoak_etc << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,

		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T"); 

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	// DEBUG:
	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_half->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}
	 
	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,
		false  
		); 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");
	cudaMemcpy(pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_half->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)// based on pXhalf !!
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");
	
//	Need to figure that v lives on centroids because 1 we have viscosity and 2 
//	Az lives on centroids to avoid trouble.

	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(   
			0.5*Timestep,
			this->p_info,
			this->p_n_minor,    // multiply by old mass ..
			this->p_AreaMinor,    // multiply by old mass ..
			pX_half->p_n_minor, // divide by new mass ..
			pX_half->p_AreaMinor, // using a cutoff at the edge of the domain.
			
			this->p_vie,
			this->p_v_n,

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,
			 
			// outputs:
			pX_half->p_vie,
			pX_half->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelAccelerate_v_from_advection pX_half");

	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d this vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
	}
	 
	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			pX_half->p_info,
			pX_half->p_vie,
			pX_half->p_v_n,
			pX_half->p_T_minor,
			pX_half->p_tri_neigh_index,
			pX_half->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info,
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		Timestep
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities 22");
	   
	kernelCentroidVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_v_overall_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags // questionable wisdom of repeating info in 3 systems
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles 22");
	
	kernelAdvectPositionsTris << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		pX_target->p_info,
		pX_half->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris 22");


	kernelGet_AreaMinorFluid  << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBC_triminor,
		p_pressureflag,
		pX_target->p_AreaMinor // output
			);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");
		
	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	cudaMemset(p_store_NTFlux, 0, sizeof(NTrates)*NUMVERTICES*MAXNEIGH);

	kernelAccumulateAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_vert,
		sz_who_vert_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		 
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		p_n_shards,

		NT_addition_rates_d,
		p_Div_v,
		p_Integrated_div_v_overall,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRateNew pX_half");
	kernelAccumulateNeutralAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_vert,
		sz_who_vert_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_v_n,
		pX_half->p_v_overall_minor,

		p_n_shards_n,

		NT_addition_rates_d,
		p_Div_v_neut,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate Neutral NT pX_half");
	kernelAddStoredNTFlux << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		p_store_NTFlux,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS AddStoredNTFlux");

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));
	cudaMemset(
		NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);

	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards,
		pX_half->p_n_minor,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor pX_half");
	
	kernelNeutral_momflux << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		 
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		pX_half->p_v_n,
		p_n_shards_n,
		pX_half->p_n_minor,

		pX_half->p_v_overall_minor,
		p_MAR_neut,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_momflux pX_half");

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");

	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");


	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	    
	kernelAdvanceDensityAndTemperature_nosoak_etc << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		pX_half->p_n_major,  // ?
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,

		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,

		pX_target->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
		pX_target->p_cc,
		pX_target->p_tri_corner_index,
		pX_target->p_tri_periodic_corner_flags,
		false
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");
	cudaMemcpy(pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_target->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");
	 
	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner,
		p_one_over_n);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	cudaEventRecord(middle, 0);
	cudaEventSynchronize(middle); 

	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(
			Timestep, 
			this->p_info,
			
			this->p_n_minor,    // multiply by old mass ..
			this->p_AreaMinor,
			pX_target->p_n_minor, // divide by new mass ..
			pX_target->p_AreaMinor,

			this->p_vie,
			this->p_v_n, // v_k

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,

			// outputs:
			pX_target->p_vie,
			pX_target->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS Accelerate_v_from_advection");
	 
	 
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\n\n\n%d this vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_target->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_target->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_target vn.y %1.9E vy %1.9E\n\n\n", VERTCHOSEN, tempf64, tempb);
	}
	
	if (bGlobalSaveTGraphs)
	{
		Divide_diff_get_accel << <numTilesMajor, threadsPerTileMajor >> >(
			pX_target->p_vie + BEGINNING_OF_CENTRAL,
			this->p_vie + BEGINNING_OF_CENTRAL,
			Timestep,
			p_accelgraph[10]
		); 
		Call(cudaThreadSynchronize(), "cudaTS Divide_diff_get_accel");
	}
	// get grad Az, Azdot and anti-advect:
	 
	kernelAntiAdvect << <numTriTiles, threadsPerTileMinor >> >(
		Timestep,
		this->p_info, 
		
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		this->p_AAdot,
		pX_half->p_v_overall_minor, // speed of move of this point
		pX_target->p_AAdot // for output
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelAntiAdvect ");
	 
	kernelWrapVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_was_vertex_rotated); // B will be recalculated.
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapvertices ");

	// Here put a test of whether any did have to wrap around.

	cudaMemset(p_triPBClistaffected, 0, sizeof(char)*NUMVERTICES);
	kernelWrapTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_triPBClistaffected,
		pX_target->p_tri_periodic_corner_flags
		); // B will be recalculated.							   
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapTriangles ");

	kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_tri_neigh_index,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_tri_periodic_corner_flags,
		pX_target->p_tri_periodic_neigh_flags,
		pX_target->p_szPBC_triminor,
		p_triPBClistaffected
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor ");

	kernelReset_szPBCtri_vert << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_vert,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBCneigh_vert,
		p_triPBClistaffected);
	Call(cudaThreadSynchronize(), "cudaTS Reset for vert. ");

	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, Timestep);

	SetConsoleTextAttribute(hConsole, 13);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time advection : %f ms ", elapsedTime);
	SetConsoleTextAttribute(hConsole, 15);

	nvals n;
	cudaMemcpy(&n, &(pX_half->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_half n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_target->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_target n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64;
		cudaMemcpy(&tempf64, &(pX_target->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d v.y %1.9E \n", VERTCHOSEN, tempf64);
	}
	runs++;
}
 
void cuSyst::PerformCUDA_Advance_noadvect(//const 
	cuSyst * pX_target,
	//const 
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles, iVertex;
	f64 hsub, Timestep;
	FILE * fp;
	long iMinor, iSubstep;
	f64 Iz_prescribed;
	static long runs = 0;
	float elapsedTime;
	f64 Iz_k, Iz_prescribed_endtime;
	f64_vec2 temp_vec2;
	// DEBUG:
	f64 sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot;
	FILE * fp_2;
	static int iHistory = 0;
	static int iTemp = 0;

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	// Clearly I hadn't moved this up.
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	//Came in with it.
	//Did it ever actually use it before? 
	//This seems not to fit. We don't double the change, we get it for the first time.
	//That's actually more concerning than that we came in with it.
	
	cudaEvent_t start, stop, middle;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&middle);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

#define USE_N_MAJOR_FOR_VERTEX 

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);

	Timestep = TIMESTEP;
	// DEBUG:
	printf("\nDebugNaN this\n\n");
	DebugNaN(this);
	
	Test_Asinh << <1, 1 >> > ();
	printf("C asinh(0.0): %1.8E \n", asinh(0.0));
	Call(cudaThreadSynchronize(), "cudaTS asinh test");


	if (!GlobalSuppressSuccessVerbosity) {
		long izTri[MAXNEIGH];

		structural info;
		cudaMemcpy(&info, &(this->p_info[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
		printf("info: flag %d tri_len %d  \n", info.flag, info.neigh_len);

		cudaMemcpy(izTri, this->p_izTri_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d tri %d : %d\n", VERTCHOSEN, i, izTri[i]);

		long izNeigh[MAXNEIGH];
		cudaMemcpy(izNeigh, this->p_izNeigh_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d neigh %d : %d\n", VERTCHOSEN, i, izNeigh[i]);

		LONG3 cornerindex;
		cudaMemcpy(&cornerindex, this->p_tri_corner_index + izTri[0], sizeof(LONG3), cudaMemcpyDeviceToHost);

		for (i = 0; i < 3; i++)
			printf("%d corner 012 : %d %d %d\n", izTri[0], cornerindex.i1, cornerindex.i2, cornerindex.i3);

		nvals n2;
		cudaMemcpy(&n2, &(this->p_n_major[VERTCHOSEN]),
			sizeof(nvals), cudaMemcpyDeviceToHost);
		printf("evaltime %1.8E n %1.8E n_n %1.8E ", evaltime, n2.n, n2.n_n);

		f64_vec3 B;
		cudaMemcpy(&B, &(this->p_B[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("B %1.8E %1.8E %1.8E ", B.x, B.y, B.z);

		T3 T;
		cudaMemcpy(&T, &(this->p_T_minor[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(T3), cudaMemcpyDeviceToHost);
		printf("T %1.8E %1.8E %1.8E \n", T.Tn, T.Ti, T.Te);

		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("v_n [%d] %1.9E %1.9E %1.9E ", VERTCHOSEN, tempvec3.x, tempvec3.y, tempvec3.z);

		v4 tempv4;
		cudaMemcpy(&tempv4, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL]),
			sizeof(v4), cudaMemcpyDeviceToHost);
		printf("v_xy [%d] %1.9E %1.9E viz %1.9E vez %1.9E ", VERTCHOSEN, tempv4.vxy.x,
			tempv4.vxy.y, tempv4.viz, tempv4.vez);

		f64 tempf64;
		cudaMemcpy(&tempf64, &(this->p_AAdot[VERTCHOSEN + BEGINNING_OF_CENTRAL].Azdot),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("Azdot %1.10E \n", tempf64);		
	}

	// ```````````````````````````````````````````````````````````````
	//                        Thermal Pressure:
	// ```````````````````````````````````````````````````````````````

	// We are going to want n_shards and it is on centroids not circumcenters.
	// v lives on centroids .. as does Az.
	// Can anticipate problems if we tried to change that.

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false // true == calculate n and T on circumcenters instead of centroids
			  // We are using n_minor for things where we never load cc.
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// To match how we do it below we should really be adding in iterations of ShardModel and InferMinorDensity.

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_n_major,
		this->p_n_minor,  // DESIRED VALUES
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		p_n_shards,
		p_n_shards_n,
		this->p_AreaMajor,
		false // USE CENTROIDS --- pressure routine does not load in cc's
		);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels this");

	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n); // overwrites but it doesn't matter
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");

	// Used for ?

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> > (

		this->p_info,
		this->p_T_minor,
		this->p_AAdot,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,

		p_n_shards,				// this kernel is for i+e only
		this->p_n_minor,
		p_pressureflag,
		p_GradTe,
		p_GradAz,
		this->p_B // HERE THIS IS GETTING PoP'D 
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_CurlA_minor");

	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_T_minor,
		p_n_shards_n,
		this->p_n_minor,

		p_pressureflag,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure only : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure only : pMAR_neut.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
	};


	long i;
	SetConsoleTextAttribute(hConsole, 11);

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		// Report NnTn:
		SetConsoleTextAttribute(hConsole, 14);

		cudaMemcpy(&tempf64, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		//if (tempf64 <= 0.0) getch();

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NnTn rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NiTi rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}
	 
	// ``````````````````````````````````````````````````````````````````````
	//              Heat conduction:
	// ``````````````````````````````````````````````````````````````````````

	// Putting heat cond first because we chose to do it on cc.
	// That can be questioned but it goes well with longitudinal flow between vertices - see formula in kernel.
	// Not changing it right now, but worth a rethink if possible.
	// The aim of the following is to put T on cc.

	// values never used:
	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		true // true == calculate n and T on circumcenters instead of centroids
			  // We are using n_minor for things where we never load cc.
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
	// We do not need to construct shard model however. That doesn't affect T_minor.
	// n from average is good enough for kappa & nu I should think.
	
	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		// Report NnTn:
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	//kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> > (
	//	this->p_info,
	//	this->p_n_minor,
	//	this->p_T_minor,
	//	p_kappa_n,
	//	p_kappa_i,
	//	p_kappa_e,
	//	p_nu_i,
	//	p_nu_e		// define all these.
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS Calculate_kappa_nu(this)");
	
	kernelCalculate_kappa_nu_vertices << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e
	);
	Call(cudaThreadSynchronize(), "cudaTS Kappa on vertices");

	cudaMemcpy(pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

#define p_move1n p_sqrtD_inv_n
#define p_move1i p_sqrtD_inv_i
#define p_move1e p_sqrtD_inv_e

	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_move1n, p_move1i, p_move1e, p_store_T_move1); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack move");
	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, pX_half->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k+1 ");
	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tnk, p_Tik, p_Tek, this->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k");

	int iSuccess;
#define WOOLLY
#ifndef WOOLLY
	do {
		iSuccess = RunBackwardForHeat_ConjugateGradient(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this,
			false);
		if (iSuccess != 0) iSuccess = RunBackwardJLSForHeat(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this,
			false);		
	} while (iSuccess != 0);

#else
	
	iEquations[0] = NUMVERTICES;
	iEquations[1] = NUMVERTICES;
	iEquations[2] = NUMVERTICES;

	printf("NEUTRAL SOLVE:\n");
	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			0.5*Timestep,
			p_move1n,
			p_Tn,
			p_Tnk,
			this,
			p_kappa_n, p_nu_i,
			0,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess =

			RunBwdRnLSForHeat(
				p_Tnk, p_Tn,
				0.5*Timestep, this, false, 
				0, p_kappa_n, p_nu_i);
		//	
		//	RunBwdJnLSForHeat(p_Tnk, p_Tn,
		//	0.5*Timestep, this, false,
		//	0,
		//	p_kappa_n,
		//	p_nu_i);
		
	} while (iSuccess != 0);
//	cudaMemcpy(&tempf64, &(p_Tn[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&tempf64_2, &(p_Ti[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&tempf64_3, &(p_Te[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//
//	printf("%d T %1.10E %1.10E %1.10E \n", VERTCHOSEN, tempf64, tempf64_2, tempf64_3);

	printf("ION SOLVE:\n");
	
	//cudaMemcpy(&tempf64, &(p_kappa_i[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&tempf64_2, &(p_nu_i[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&tempf64_3, &(p_move1i[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);

	//printf("%d kappa_i %1.10E nu_i %1.10E move1i %1.10E \n", VERTCHOSEN, tempf64, tempf64_2, tempf64_3);

	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			0.5*Timestep,
			p_move1i,
			p_Ti,
			p_Tik,
			this,
			p_kappa_i, p_nu_i,
			1,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess = RunBwdRnLSForHeat(
			p_Tik, p_Ti,			
			0.5*Timestep, this, false, 1, p_kappa_i, p_nu_i
			);
		//iSuccess = RunBwdJnLSForHeat(p_Tik, p_Ti,
		//	0.5*Timestep, this, false,
		//	1,
		//	p_kappa_i,
		//	p_nu_i);
	} while (iSuccess != 0);
	
	printf("ELECTRON SOLVE:\n");
	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			0.5*Timestep,
			p_move1e,
			p_Te,
			p_Tek,
			this,
			p_kappa_e, p_nu_e,
			2,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess = RunBwdRnLSForHeat(
			p_Tek, p_Te,
			0.5*Timestep, this, false, 2,p_kappa_e, p_nu_e
			); 
		
	} while (iSuccess != 0);
	// Almost instant. Whereas NLCG (formula) takes 40s and NLCG(ours) takes nearly 4 minutes.

	kernelPackupT3 << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(pX_half->p_T_minor + BEGINNING_OF_CENTRAL, p_Tn, p_Ti, p_Te); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelPack k+1");

#endif
	
	// Store move:
	SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
		(p_store_T_move1,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL);
	Call(cudaThreadSynchronize(), "cudaTS subtractT3");
	SetConsoleTextAttribute(hConsole, 15);

	cudaMemcpy(&tempf64, &((pX_half->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("solved Te[%d]: %1.9E\n", VERTCHOSEN, tempf64);
	if (tempf64 < 0.0) {
		SafeExit(13223);
	}	
	//if ((DEBUGTE) && (tempf64 > 1.0e-11)) {
	//	printf("press f");
	//	while (getch() != 'f');
	//}
	
	// Now create dNT from T_k+1/2 ! :
	// ================================

	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			this->p_info,
			this->p_izNeigh_vert,
			this->p_szPBCneigh_vert,
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_cc,
			this->p_n_major,
			p_Tn, // T_k+1/2 just calculated
			this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_nu_i,
			NT_addition_rates_d,
			this->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			false,//bUseMask,
			0);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate neut");

	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		this->p_n_major,
		p_Ti, // T_k+1/2 just calculated
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_i,
		p_nu_i,
		NT_addition_rates_d,
		this->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		false,//bUseMask,
		1);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate ion");

	SetConsoleTextAttribute(hConsole, 14);
	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		this->p_n_major,
		p_Te, // T_k+1/2 just calculated
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_e,
		p_nu_e,
		NT_addition_rates_d,
		this->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		false,//bUseMask,
		2);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate elec");

	SetConsoleTextAttribute(hConsole, 15);

	// This isn't ideal ---- for this one we might like the old half move and the old
	// full move stored
	// but for the full move which is the more expensive one -- do we really want
	// half move and old move?

	// this did have area populated... within ins
	// dNT/dt = 0 here anyway!
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES * 2); // initially allow all flows good
//
	//int iPass = 0;
	//bool bContinue;
	//do {
	//	printf("iPass %d :\n", iPass);
	//	// reset NTrates:
	//	cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	//	kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
	//		this->p_info,
	//		this->p_izNeigh_vert,
	//		this->p_szPBCneigh_vert,
	//		this->p_izTri_vert,
	//		this->p_szPBCtri_vert,
	//		this->p_cc,
	//		this->p_n_major,
	//		  
	//		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
	//		p_boolarray, // array of which ones require longi flows
	//			 		 // 2 x NMAJOR
	//		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
	//		p_kappa_n,
	//		p_kappa_i, 
	//		p_kappa_e,
	//		p_nu_i, 
	//		p_nu_e,
	//		NT_addition_rates_d,
	//		this->p_AreaMajor,
	//		(iPass == 0) ? false : true,
	//		     
	//		p_boolarray2,
	//		p_boolarray_block,
	//		false);
	//	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
	//	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
	//	
	////	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	////	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		//	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
	//	kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
	//		0.5*Timestep,
	//		this->p_info,
	//		this->p_T_minor + BEGINNING_OF_CENTRAL,
	//		//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
	//		this->p_n_major,
	//		this->p_AreaMajor,
	//		NT_addition_rates_d,
	//		p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
	//					 // 2x NMAJOR
	//		p_bFailed, // did we hit any new negative T to add
	//		p_boolarray2,
	//		p_boolarray_block,
	//		false
	//		);
	//	Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");
		//	f64 Te;
	//	cudaMemcpy(&Te, &(this->p_T_minor[BEGINNING_OF_CENTRAL + VERTCHOSEN].Te), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("%d Te (this) %1.10E \n", VERTCHOSEN, Te);
	//	cudaMemcpy(&tempf64, &(pX_half->p_T_minor[BEGINNING_OF_CENTRAL + VERTCHOSEN].Te), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("%d Te (pX_half) %1.10E \n", VERTCHOSEN, tempf64);
		//	f64 area;
	//	cudaMemcpy(&area, &(this->p_AreaMajor[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("Area %1.10E ", area);
	//	f64 temp2;
	//	cudaMemcpy(&temp2, &(this->p_n_major[VERTCHOSEN].n), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("* n %1.10E = %1.10E \n", temp2, temp2*tempf64);
	//	f64 temp3;
	//	cudaMemcpy(&temp3, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("%d NT_addition_rates_d NeTe rate %1.10E \n", VERTCHOSEN, temp3);
	//	
	//	printf("predicted Te: %1.10E \n", Te + 0.5e-12*temp3 / (temp2*area));
		//	bContinue = false;
	//	cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	//	int i;
	//	for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
	//	if (i < numTilesMajorClever) bContinue = true;
	//	iPass++;
	//} while (bContinue);
	  
//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
//	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemcpy(store_heatcond_NTrates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);
	
	// Overwrite minor densities back again:
	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n); // overwrites but it doesn't matter
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d before ionization : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d before ionization : pMAR_neut.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
	};

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> > (
		0.5*Timestep,
		this->p_info, 
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		this->p_AreaMajor,
		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		0,0
		);  
	Call(cudaThreadSynchronize(), "cudaTS Ionisation");
	
	cudaMemcpy(&tempf64, &(p_temp3_3[VERTCHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_temp3_e.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);

	cudaMemcpy(&tempf64, &(p_temp3_1[VERTCHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_temp3_n.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);

	Collect_Ntotal_major << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMinor,
		p_temp1, // Ntotal major
		p_temp2  // Nntotal major 
		);
	Call(cudaThreadSynchronize(), "cudaTS Gather Ntotal");
	  
	Augment_dNv_minor << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		p_temp1,
		p_temp2,
		this->p_AreaMinor,
		p_temp3_1, p_temp3_2, p_temp3_3,
		p_MAR_neut,
		p_MAR_ion,
		p_MAR_elec);
	Call(cudaThreadSynchronize(), "cudaTS Augment_dNv_minor");
	

	
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		// DEBUG:
		cudaMemcpy(&tempf64, &(p_MAR_neut[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
		if (tempf64 != tempf64) {
			printf("NaN encountered: p_MAR_neut[%d].z", CHOSEN);
			getch();
			SafeExit(100000001);
		}
		// Report NnTn:
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(p_MAR_elec[VERTCHOSEN + BEGINNING_OF_CENTRAL].z), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nMAR_elec.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
	(
		this->p_info,
		this->p_vie,
		this->p_v_n,
		this->p_T_minor,
		this->p_tri_neigh_index,
		this->p_izNeigh_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS resetv");
	 
#ifdef PRECISE_VISCOSITY

	structural info;
	//cudaMemcpy(&info, &(this->p_info[CHOSEN]), sizeof(structural), cudaMemcpyDeviceToHost);
	//printf("\npX_use->p_info[%d].pos %1.10E %1.10E modulus %1.10E \n\n",
	//	CHOSEN, info.pos.x, info.pos.y, info.pos.modulus());

	//                                    Viscosity
	// ================================================================================
	 
	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor, this->p_T_minor,
		p_nu_i,	p_nu_e, p_nu_n,
		p_ita_i, p_ita_e, p_ita_n, 
		0, 0); 
	Call(cudaThreadSynchronize(), "cudaTS ita 1");
	
	// 1. Create time-derivative based on t_k (or t_k+1/2 later on); advance v putatively.
	
	cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d3, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_3, 0, sizeof(NTrates)*NUMVERTICES);

	// Let 3 = from forward move, 2= backward

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_vie,
		// For neutral it needs a different pointer.
		this->p_v_n, // not used
		this->p_izTri_vert,	this->p_szPBCtri_vert,	this->p_izNeigh_TriMinor,	this->p_szPBC_triminor,
		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		this->p_B,
		p_MAR_ion3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		1, m_ion_, 1.0 / m_ion_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_vie,
		// For neutral it needs a different pointer.
		this->p_v_n, // not used
		this->p_izTri_vert,	this->p_szPBCtri_vert,	this->p_izNeigh_TriMinor,	this->p_szPBC_triminor,
		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		this->p_B,
		p_MAR_elec3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

	NTrates NTratestemp;

	SetConsoleTextAttribute(hConsole, 13);
	printf("-------------------------------------------\n");
	SetConsoleTextAttribute(hConsole, 15);
	cudaMemcpy(&NTratestemp, &(NT_addition_rates_d_3[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
	printf("did initial forward move.\n");
	printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
	SetConsoleTextAttribute(hConsole, 13);
	printf("-------------------------------------------\n");
	SetConsoleTextAttribute(hConsole, 15);

	//printf("DEBUG: Look for negs!");
	//cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
	//kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
	//	NT_addition_rates_d_3, NT_addition_tri_d3,
	//	p_longtemp);
	//Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
	//cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	//long negs = 0;
	//for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
	//printf("at least %d negs\n", negs);
	//if (negs > 0) getch();


	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_v_n, 
		this->p_izTri_vert,	this->p_szPBCtri_vert,	this->p_izNeigh_TriMinor,	this->p_szPBC_triminor,
		p_ita_n, // ita
		p_nu_n,
		p_MAR_neut3, 
		NT_addition_rates_d_3,
		NT_addition_tri_d3);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	// These are the forward flows and heatings which we need to save, in the forward region.

	kernelPutativeAccel << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep*0.5,
		this->p_info,
		this->p_vie, 
		this->p_v_n,
		pX_half->p_vie,
		pX_half->p_v_n,
		this->p_n_minor, this->p_AreaMinor,
		p_MAR_ion3,
		p_MAR_elec3,
		p_MAR_neut3);
	Call(cudaThreadSynchronize(), "cudaTS PutativeAccel");
	

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// 2 . Take dv/dt at the destination (pX_half):
	// NOTE:
	// We could try including frictional part if that damps the explosive oscillations.  !
	// That is fair -- we include it in putative accel and can see if then there is a countersigned new dv/dt.

	cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_vie,
		pX_half->p_v_n, // not used
		this->p_izTri_vert,	this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_i,  p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		this->p_B,
		p_MAR_ion2, // accumulates
		NT_addition_rates_d_2, NT_addition_tri_d2, // throwaway
		1, m_ion_, 1.0 / m_ion_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_vie,
		pX_half->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_e,  p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		this->p_B,
		p_MAR_elec2, // accumulates
		NT_addition_rates_d_2, NT_addition_tri_d2,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_v_n,
		this->p_izTri_vert,	this->p_szPBCtri_vert,	this->p_izNeigh_TriMinor,	this->p_szPBC_triminor,
		p_ita_n,  p_nu_n,
		p_MAR_neut2, // just accumulates
		NT_addition_rates_d_2,
		NT_addition_tri_d2);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");


	//Okay hang on, if this is the case why do we have a neutral pointer in the ie routine. --- Pointless apparently.
	
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// 3.4. Make a test for an overshooting criterion; expand flag region around it.

	CallMAC(cudaMemset(p_Selectflag, 0, sizeof(int)*NMINOR));
	CallMAC(cudaMemset(p_SelectflagNeut, 0, sizeof(int)*NMINOR));
	kernelTest_Derivatives << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep*0.5,
		this->p_info,
		p_MAR_ion3,		p_MAR_elec3,		p_MAR_neut3,
		p_MAR_ion2,		p_MAR_elec2,		p_MAR_neut2,
		this->p_vie,
		this->p_v_n,
		this->p_n_minor, this->p_AreaMinor,
		p_Selectflag, p_SelectflagNeut // this is the output
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelTest_Derivatives");
	
	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		1);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");
	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		2);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");
	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		3);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");
	
	kernelCountNumberMoreThanZero<<<numTilesMinor, threadsPerTileMinor >> > (
		p_Selectflag, p_SelectflagNeut,
		p_blockTotal1,p_blockTotal2);
	cudaMemcpy(p_longtemphost, p_blockTotal1, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_longtemphost2, p_blockTotal2, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	long total1 = 0, total2 = 0;
	for (int iTile = 0; iTile < numTilesMinor; iTile++)
	{
		total1 += p_longtemphost[iTile];
		total2 += p_longtemphost2[iTile];
	};
	printf("Number selectflag>0 %d selectflagneut>0 %d \n\n",
		total1, total2);
	
	//cudaMemcpy(p_inthost, &(p_Selectflag[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(int), cudaMemcpyDeviceToHost);
	//printf("flag at %d = %d \n", VERTCHOSEN, p_inthost[0]);
	
	kernelZeroSelected << <numTilesMinor, threadsPerTileMinor >> > (
		p_MAR_ion3,
		p_MAR_elec3,
		p_MAR_neut3,
		NT_addition_rates_d_3, NT_addition_tri_d3,
		p_Selectflag, p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS ZeroSelected");
	
	// 5. Update variables going into backward solve:
	// where Selectflag == 0, v_k_modified = v_updated, but it isn't important since we don't look there for epsilon anyway
	// where Selectflag > 0, v_k_modified = v_k (+/-) the flows to only selectflag==0 neighbours. Times h/N.

	// But accumulate fixed flows into p_MAR_3:
	// 4 = bwd from fwd region

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species___fixedflows_only << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_vie,
		this->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		this->p_B,
		p_MAR_ion3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		1, m_ion_, 1.0 / m_ion_,
		p_Selectflag
		);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species___fixedflows_only
		<< <numTriTiles, threadsPerTileMinor >> > (
			this->p_info,
			this->p_vie, 
			this->p_v_n, // not used
			this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
			p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			this->p_B,
			p_MAR_elec3, // accumulates
			NT_addition_rates_d_3, NT_addition_tri_d3,
			2, m_e_, 1.0 / m_e_,
			p_Selectflag
			);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");
	
	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d_3[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("did fixed flows \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);

	};

	//printf("DEBUG: Look for negs!");
	//cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
	//kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
	//	NT_addition_rates_d_3, NT_addition_tri_d3,
	//	p_longtemp);
	//Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
	//cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	//negs = 0;
	//for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
	//printf("at least %d negs\n", negs);
	//if (negs > 0) getch();


	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric___fixedflows_only << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_v_n,

	//	Why was this half not this.
		
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_n, // ita
		p_nu_n,
		p_MAR_neut3, // just accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	// The following function is modelled on epsilon calc and must be updated with it.
	kernelCreate_v_k_modified_with_fixed_flows << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		this->p_info,
		p_vie_modified_k,
		p_v_n_modified_k,
		this->p_vie,
		this->p_v_n,
		pX_half->p_vie,
		pX_half->p_v_n,
		p_Selectflag,
		p_SelectflagNeut,
		p_MAR_ion3, p_MAR_elec3, p_MAR_neut3,
		this->p_n_minor,
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS Create v_k_modified");
	
	cudaMemcpy(pX_half->p_vie, p_vie_modified_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_v_n, p_v_n_modified_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	// This is to give us a good starting point for applying a seed that we created based on the move between systems.

	
	if (0) {
		f64 tempf64_2, tempf64_3;
		long izTri[12];
		cudaMemcpy(izTri, &(this->p_izTri_vert[MAXNEIGH*VERTCHOSEN]), sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		int ii;
		for (i = 0; i < 6; i++)
		{
			cudaMemcpy(&tempf64, &(pX_half->p_vie[izTri[i]].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&tempf64_2, &(this->p_vie[izTri[i]].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&ii, &(p_Selectflag[izTri[i]]), sizeof(int), cudaMemcpyDeviceToHost);
			printf("%d | Selectflag %d  vy_k %1.9E  vy_mod %1.9E \n",
				izTri[i], ii, tempf64_2, tempf64);
		};
		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempf64_2, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&ii, &(p_Selectflag[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(int), cudaMemcpyDeviceToHost);
		printf("%d | Selectflag %d  vy_k %1.9E  vy_mod %1.9E \n",
			VERTCHOSEN + BEGINNING_OF_CENTRAL, ii, tempf64_2, tempf64);

		printf("press q\n");
		while (getch() != 'q');
	};
	

	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_n_minor, // Now on centroids so need to have put it back
		pX_half->p_T_minor,
		p_nu_i,
		p_nu_e,
		p_nu_n,
		p_ita_i,
		p_ita_e,
		p_ita_n,
		p_Selectflag, p_SelectflagNeut);
	Call(cudaThreadSynchronize(), "cudaTS ita");

	if (0)//DEFAULTSUPPRESSVERBOSITY == false)
	{
	// NOW DO GRAPH
	printf("graph time\n");
	GlobalCutaway = false;
	//kernelSplit_vec2 << <numTilesMinor, threadsPerTileMinor >> > (p_epsilon_xy, p_nu_i, p_nu_e);
	//Call(cudaThreadSynchronize(), "cudaTS splitvec2");
	SetActiveWindow(hwndGraphics);
	cudaMemcpy(p_temphost1, p_ita_i + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_ita_e + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_nu_i + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost4, p_nu_e + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost5, p_nu_n + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost6, p_ita_n + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

	char buffer[256];
	Vertex * pVertex = pTriMesh->X;
	plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_temphost1[iVertex];
		pdata->temp.y = (p_temphost1[iVertex] == 0.0) ? 10.0 : 0.0;
		++pVertex;
		++pdata;
	}

	sprintf(buffer, "ita_ion");
	Graph[0].DrawSurface(buffer,
		DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		false,
		GRAPH_EPSILON, pTriMesh);

	pVertex = pTriMesh->X;
	pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_temphost2[iVertex];
		pdata->temp.y = (p_temphost2[iVertex] == 0.0) ? 10.0 : 0.0;
		++pVertex;
		++pdata;
	}
	sprintf(buffer, "ita_elec");
	Graph[1].DrawSurface(buffer,
		DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		false,
		GRAPH_EPSILON, pTriMesh);

	pVertex = pTriMesh->X;
	pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_temphost6[iVertex];
		pdata->temp.y = (p_temphost6[iVertex] == 0.0) ? 10.0 : 0.0;
		++pVertex;
		++pdata;
	}
	sprintf(buffer, "ita_neut");
	Graph[2].DrawSurface(buffer,
		DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
		SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
		false,
		GRAPH_EPSILON, pTriMesh);

	SetActiveWindow(hwndGraphics);
	ShowWindow(hwndGraphics, SW_HIDE);
	ShowWindow(hwndGraphics, SW_SHOW);
	Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

	printf("done graphs\npress o to continue\n");

	while (getch() != 'o');
	};

	// 7B. Handle the case that there are few bwd equations.
	// Options:
	//       We could increase the number to the # for a direct solve, go in Bwd Solve routine and do the direct solve.
	//       OR, we could arrange to make the direct solve work without so many equations. That would be ideally it.
	//       (Alternative Bwd solve routine I guess.)

	// DEBUG:
	cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NMINOR,
		cudaMemcpyDeviceToDevice);
	// stored it for difference

	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("what already exists \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d_temp2[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
	};

	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_3);
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT fwd");

	if (DEFAULTSUPPRESSVERBOSITY == false) {

		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("now added '3' \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);

	};

	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d_temp, NT_addition_rates_d_3);// FOR GRAPHING DEBUG
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT fwd"); // FOR GRAPHING DEBUG


	if (total1 != 0) {
		RunBackwardR8LSForViscosity_Geometric(
				p_vie_modified_k, pX_half->p_vie, 0.5*Timestep, this);
		// NTrates struct should not have existed at all.		

		// The second v arg is the one where we create the solution.
		// It should be set to forward values in the forward region --- CORRECT, done. 		
		// And we setted p_vie_modified_k to the same thing in the forward region, not to generate epsilon.
		// And setted both v to v_k in the bwd region so we can apply the seed.
	} else {
		// Make sure data is zeroed.
		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
	};

	/*printf("DEBUG: Look for negs!");
	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
	kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
		NT_addition_rates_d_2, NT_addition_tri_d2,
		p_longtemp);
	Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	negs = 0;
	for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
	printf("at least %d negs\n", negs);
	if (negs > 0) getch();*/


	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d_2[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("rates 2 from bwd solve \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
	};

	// NTrates2 will get smooshed so do addition here & save from tris
	cudaMemcpy(NT_addition_tri_d, NT_addition_tri_d2, sizeof(NTrates)*NUMVERTICES * 6, cudaMemcpyDeviceToDevice);
	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_2);
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT");
	
	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d_temp, NT_addition_rates_d_2);// FOR GRAPHING DEBUG
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT bwd");// FOR GRAPHING DEBUG


	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("added bwd solve into main rates \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);		
	};

/*
	printf("DEBUG: Look for negs!");
	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
	kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
		NT_addition_rates_d, NT_addition_tri_d,
		p_longtemp);
	Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	negs = 0;
	for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
	printf("at least %d negs\n", negs);
	if (negs > 0) getch();


*/
	if (total2 != 0) {
		RunBackward8LSForNeutralViscosity_Geometric
		(p_v_n_modified_k, pX_half->p_v_n, 0.5*Timestep, this);
	} else {
		// Make sure data is zeroed.
		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
	};
	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_2);  // the data from neutral bwd
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT neut bwd");

	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d_temp, NT_addition_rates_d_2);// FOR GRAPHING DEBUG
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT bwd neut");// FOR GRAPHING DEBUG


	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("did neutral stuff \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
	};

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp1, p_temp2);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 1");
	kernelTransmit_3x_HeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp1, p_temp2,
		NT_addition_rates_d, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd");
#else
	kernelCollect_Up_3x_HeatIntoVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_tri_corner_index, // to see which corner we are.
		NT_addition_rates_d, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd");
#endif

	// What we now understand: we accumulated various things into NT_addition_rates_d, which already had data.

	// So finding negatives is not appropriate.

	if (DEFAULTSUPPRESSVERBOSITY == false) {
		kernelSubtractNiTiCheckNeg << <numTilesMajor, threadsPerTileMajor >> >
			(NT_addition_rates_d_temp2, NT_addition_rates_d, p_temp1,
				this->p_n_minor + BEGINNING_OF_CENTRAL,
				this->p_AreaMajor);
		Call(cudaThreadSynchronize(), "cudaTS SubtractNiTi");
	};

	/*printf("\nDEBUG: Look for negs!\n");
	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
	kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
		NT_addition_rates_d, NT_addition_tri_d,
		p_longtemp);
	Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	negs = 0;
	for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
	printf("at least %d negs\n", negs);
	if (negs > 0) {
		getch(); getch(); getch();
	};*/


	 

	if (DEFAULTSUPPRESSVERBOSITY == false) {
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("transmitted 3x heat into verts \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		cudaMemcpy(&NTratestemp, &(NT_addition_rates_d_temp2[VERTCHOSEN]), sizeof(NTrates), cudaMemcpyDeviceToHost);
		printf("old \n");
		printf("%d NiTi : %1.10E\n", VERTCHOSEN, NTratestemp.NiTi);
		SetConsoleTextAttribute(hConsole, 13);
		printf("-------------------------------------------\n");
		SetConsoleTextAttribute(hConsole, 15);
	};

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	kernelTransmit_3x_HeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,// FOR GRAPHING DEBUG// FOR GRAPHING DEBUG// FOR GRAPHING DEBUG// FOR GRAPHING DEBUG
		this->p_AreaMajor,
		p_temp1, p_temp2,
		NT_addition_rates_d_temp, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd temp");
#else
	kernelCollect_Up_3x_HeatIntoVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_tri_corner_index, // to see which corner we are.
		NT_addition_rates_d_temp, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd");
#endif

	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_ion2, p_MAR_ion3);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_elec2, p_MAR_elec3);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_neut2, p_MAR_neut3);
	Call(cudaThreadSynchronize(), "cudaTS addMAR");
	
	// Get seed:
	kernelPutativeAccel << <numTilesMinor, threadsPerTileMinor >> >(
			Timestep*0.5,
			this->p_info,
			this->p_vie,
			this->p_v_n,
			pX_half->p_vie,
			pX_half->p_v_n,
			this->p_n_minor, this->p_AreaMinor,
			p_MAR_ion2,
			p_MAR_elec2,
			p_MAR_neut2);
	Call(cudaThreadSynchronize(), "cudaTS PutativeAccel");
	// Ensure that we put the whole move into storage, but that when we use it, we select only the solved-for cells.	
	Subtract_V4 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move4, this->p_vie, pX_half->p_vie);
	Call(cudaThreadSynchronize(), "cudaTS Subtract_V4");
	SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, this->p_v_n, pX_half->p_v_n);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");

	// NOTE: When we run whole step we save into p_stored_move3_2 instead.
	
	// Note that MAR and heating were already in use!
	// Now add the combined acceleration into the acceleration we use.
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_ion, p_MAR_ion2);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_elec, p_MAR_elec2);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_neut, p_MAR_neut2);
	Call(cudaThreadSynchronize(), "cudaTS addMAR");
	

	// --------------------------------------------------------
	// --------           DIAGNOSTIC GRAPHS            --------
	// --------------------------------------------------------

	if (DEFAULTSUPPRESSVERBOSITY == false);
	
	{
		/*
		printf("graph time ii\n");
		GlobalCutaway = false;
		//kernelSplit_vec2 << <numTilesMinor, threadsPerTileMinor >> > (p_epsilon_xy, p_nu_i, p_nu_e);
		//Call(cudaThreadSynchronize(), "cudaTS splitvec2");
		SetActiveWindow(hwndGraphics);
		cudaMemcpy(p_tempvec4_host, p_stored_move4 + BEGINNING_OF_CENTRAL, sizeof(v4)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_inthost, p_Selectflag + BEGINNING_OF_CENTRAL, sizeof(int)*NUMVERTICES, cudaMemcpyDeviceToHost);

		cudaMemcpy(p_NTrates_host, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

		DivideNiTiDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			this->p_AreaMajor,
			this->p_n_major,
			p_Tgraph[2]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNiTi_by_N");
		cudaMemcpy(p_temphost1, p_Tgraph[2], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		DivideNeTeDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			this->p_AreaMajor,
			this->p_n_major,
			p_Tgraph[2]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");
		cudaMemcpy(p_temphost2, p_Tgraph[2], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		Vertex * pVertex;
		plasma_data * pdata;

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec4_host[iVertex].vxy.x;
			pdata->temp.y = p_tempvec4_host[iVertex].vxy.y;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "change xy");
		Graph[0].DrawSurface(buffer,
			VELOCITY_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			VELOCITY_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
			false,
			GRAPH_EPSILON, pTriMesh);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec4_host[iVertex].viz;
			int ii = p_inthost[iVertex];
			pdata->temp.y = ((f64)ii)*2.0e-12;
			++pVertex;
			++pdata;
		};
		sprintf(buffer, "change viz");
		Graph[1].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_tempvec4_host[iVertex].vez;
			int ii = p_inthost[iVertex];
			pdata->temp.y = ((f64)ii)*2.0e-12;
			++pVertex;
			++pdata;
		};
		sprintf(buffer, "change vez");
		Graph[2].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

			
		// Now we want to collect d/dt Ti and d/dt Te from dividing by N.

		//##################
		// 1. Need to go up above and collect total change in NT.
		//##################

		// 2. Create a routine to divide by N and send changes --- is this a graphing routine that already is applied?
		// BE BRAVE
		// BE YOU
		// Don't be afraid!


		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			++pVertex;
			++pdata;
		};
		sprintf(buffer, "dTi/dt");
		Graph[3].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
			false,
			GRAPH_EPSILON, pTriMesh);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost2[iVertex];
			int ii = p_inthost[iVertex];
			pdata->temp.y = ((f64)ii)*2.0e-12;
			++pVertex;
			++pdata;
		};
		sprintf(buffer, "dTe/dt");
		Graph[4].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_NTrates_host[iVertex].NeTe;
			int ii = p_inthost[iVertex];
			pdata->temp.y = ((f64)ii)*2.0e-12;
			++pVertex;
			++pdata;
		};
		sprintf(buffer, "dNeTe/dt");
		Graph[5].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

		SetActiveWindow(hwndGraphics);
		ShowWindow(hwndGraphics, SW_HIDE);
		ShowWindow(hwndGraphics, SW_SHOW);
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		printf("done graphs\npress u to continue\n");

		while (getch() != 'u');
		
		// Maybe next add option to move around or fly around the graphs.

		// Could put multiple alongside each other on a graph. 1536.
		*/
	};
	
#endif

	// Well here's a thought.
	// We ARE expecting v to change when we do a backward viscosity.
	// Yet, we will find v off from its trajectory towards that point.
	// That's when we tune the viscous flow.


	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}

	kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> > (
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL, // for resistive htg
		this->p_v_n + BEGINNING_OF_CENTRAL, // fixed bug
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_B + BEGINNING_OF_CENTRAL		  
		);
	// Add in a test for T<0 !!!
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T _noadvect"); // vertex
	
	if (!DEFAULTSUPPRESSVERBOSITY) {
		printf("\nDebugNaN pX_half\n\n");
		DebugNaN(pX_half);
	};

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,
		false
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");
	cudaMemcpy(pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_half->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)// based on pXhalf !!
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");

	//	Iz_prescribed_starttime = GetIzPrescribed(evaltime); // because we are setting pX_half->v

	f64 store_evaltime = evaltime;
	// Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*Timestep); // because we are setting pX_half->v

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d ? : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d ? : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
	};

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs");

	cudaMemcpy(p_AAdot_start, this->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_vie_start, this->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_v_n_start, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	GosubAccelerate(SUBCYCLES/2,//iSubcycles, 
		(0.5*Timestep) / (real)(SUBCYCLES), // hsub
		pX_half, // pX_use
		this // pX_intermediate
	);

	//getch();

	cudaMemcpy(pX_half->p_AAdot, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_vie, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_v_n, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64;
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN2 + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", VERTCHOSEN2, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[CHOSEN1].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", CHOSEN1, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[CHOSEN2].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", CHOSEN2, tempf64);
	}
	
	long iTile;
	
	//kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
	//	0.5*Timestep,
	//	this->p_info,
	//	p_MAR_neut, p_MAR_ion, p_MAR_elec,
	//	this->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
	//	p_LapAz,
	//	p_GradAz,
	//	p_GradTe,
	//	this->p_n_minor,
	//	this->p_T_minor,
	//	this->p_vie,
	//	this->p_v_n,
	//	this->p_AAdot,
	//	this->p_AreaMinor,
 //  
	//	p_vn0,
	//	p_v0,
	//	p_OhmsCoeffs,
	//	p_Iz0_summands,
	//	p_sigma_Izz,
	//	p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
	//	true);
	//Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");

	//cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//
	//f64 Iz0 = 0.0;
	//f64 sigma_Izz = 0.0;
	//Iz_k = 0.0;
	//for (iTile = 0; iTile < numTilesMinor; iTile++)
	//{
	//	Iz0 += p_Iz0_summands_host[iTile];
	//	sigma_Izz += p_summands_host[iTile];
	//	Iz_k += p_temphost1[iTile];
	//	if ((Iz0 != Iz0) || (sigma_Izz != sigma_Izz)) printf("tile %d Iz0 %1.9E sigma_Izz %1.9E summands %1.9E %1.9E \n",
	//		iTile, Iz0, sigma_Izz, p_Iz0_summands_host[iTile], p_summands_host[iTile]);
	//	// Track down what cell causing NaN Iz0
	//};
	 
	//f64 Ez_strength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;
	//Set_f64_constant(Ez_strength, Ez_strength_);

	//f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	//Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

	//printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
	//	Iz_prescribed_endtime, Iz0, sigma_Izz);
	//printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);

	//// Update velocities and Azdot:
	//kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
	//	0.5*Timestep,
	//	pX_half->p_info,
	//	p_vn0,
	//	p_v0,
	//	p_OhmsCoeffs,
	//	pX_target->p_AAdot,  // why target? intermediate value
	//	pX_half->p_n_minor,
	//	this->p_AreaMinor,
	//	p_LapAz,
	//	pX_half->p_AAdot,
	//	pX_half->p_vie,
	//	pX_half->p_v_n
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// =====================

	SetConsoleTextAttribute(hConsole, 15);
	evaltime = store_evaltime;

//	kernelAdvanceAzEuler << <numTilesMinor, threadsPerTileMinor >> >
//		(0.5*h, this->p_AAdot, pX_half->p_AAdot, p_ROCAzduetoAdvection);
//	Call(cudaThreadSynchronize(), "cudaTS AdvanceAzEuler");
		
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	// Why are we doing this way? Consistent would be better.
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


	kernelResetFrillsAz_II << < numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_tri_neigh_index, pX_half->p_AAdot);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills I");

	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	if (0) {
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez pX_half [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}
	SetConsoleTextAttribute(hConsole, 15);
	
	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_T_minor,
		pX_half->p_AAdot,
		 
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards,				// this kernel is for i+e only
		pX_half->p_n_minor,

		p_pressureflag,
		p_GradTe,
		p_GradAz,

		pX_half->p_B
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect pX_half");
	

	// Copy it to pX_target because we haven't called pressure there yet:
	 
	cudaMemcpy(pX_target->p_AreaMinor, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	
	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_info,
		  
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,
		 
		pX_half->p_T_minor,
		p_n_shards_n,
		pX_half->p_n_minor,

		p_pressureflag,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure pX_half");

	if (bGlobalSaveTGraphs) {
		cudaMemcpy(p_MAR_ion_pressure_major_stored,
			p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_pressure_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		// Pre-store existing:
		cudaMemcpy(p_MAR_elec_ionization_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	};

	if (bGlobalSaveTGraphs) {
		DivideMAR_get_accel << <numTilesMajor, threadsPerTileMajor >> > (
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL,
			this->p_n_minor + BEGINNING_OF_CENTRAL,
			this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
			p_accelgraph[4],
			p_accelgraph[5]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideMAR_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
	}

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
	};

	// If we're going to wipe out n temporarily, let's store it to put back afterwards:
	cudaMemcpy(pX_target->p_n_minor, pX_half->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_T_minor, pX_half->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToDevice);

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,

		true // calculate n and T on circumcenters instead of centroids
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS Average_nTx cc");

	// Do not need cc shard model

	// Do need cc for the following THOUGH !!!

	// HEAT DIFFUSION:

	kernelCalculate_kappa_nu_vertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e
		);
	Call(cudaThreadSynchronize(), "cudaTS Kappa on vertices");

	// Simple edit, 5th November 2020: Turn off mask for now. :/

#define HEATBASESYST this
#define HEATSTEP Timestep

	cudaMemcpy(pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	
	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_move1n, p_move1i, p_move1e, p_store_T_move2); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack move");
	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, pX_target->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k+1 ");
	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tnk, p_Tik, p_Tek, HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k");

	iEquations[0] = NUMVERTICES;
	iEquations[1] = NUMVERTICES;
	iEquations[2] = NUMVERTICES;

	printf("NEUTRAL SOLVE:\n");
	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			HEATSTEP,
			p_move1n,
			p_Tn,
			p_Tnk,
			HEATBASESYST,
			p_kappa_n, p_nu_i,
			0,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess = RunBwdRnLSForHeat(
				p_Tnk, p_Tn,
				HEATSTEP, pX_half, false,
				0, p_kappa_n, p_nu_i);		

	} while (iSuccess != 0);


	printf("ION SOLVE:\n");
	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			HEATSTEP,
			p_move1i,
			p_Ti,
			p_Tik,
			HEATBASESYST,
			p_kappa_i, p_nu_i,
			1,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess = RunBwdRnLSForHeat(
			p_Tik, p_Ti,
			HEATSTEP, pX_half, false,
			1, p_kappa_i, p_nu_i);
	} while (iSuccess != 0);


	printf("ELECTRON SOLVE:\n");
	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedT_1species(
			HEATSTEP,
			p_move1e,
			p_Te,
			p_Tek,
			HEATBASESYST,
			p_kappa_e, p_nu_e,
			2,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("iHistory %d\n", iHistory);
		cudaMemset(p_regressors + (REGRESSORS - 1)*NUMVERTICES, 0, sizeof(f64)*NUMVERTICES);
	};
	do {
		iSuccess = RunBwdRnLSForHeat(
			p_Tek, p_Te,
			HEATSTEP, pX_half, false, 
			2, p_kappa_e, p_nu_e
		);
	} while (iSuccess != 0);
	
	kernelPackupT3 << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(pX_target->p_T_minor + BEGINNING_OF_CENTRAL, p_Tn, p_Ti, p_Te); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelPack k+1");

	// Store move:
	SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
		(p_store_T_move2,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL);
	Call(cudaThreadSynchronize(), "cudaTS subtractT3");
	SetConsoleTextAttribute(hConsole, 15);


	// Now create dNT from T_k+1/2 ! :
	// ================================

	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,
		pX_half->p_n_major,
		p_Tn, // T_k+1/2 just calculated
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_nu_i,
		NT_addition_rates_d,
		pX_half->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		false,//bUseMask,
		0);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate neut");

	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,
		pX_half->p_n_major,
		p_Ti, // T_k+1/2 just calculated
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_i,
		p_nu_i,
		NT_addition_rates_d,
		pX_half->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		false,//bUseMask,
		1);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate i");

	SetConsoleTextAttribute(hConsole, 14);
	kernelAccumulateDiffusiveHeatRate_1species_Geometric << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,
		pX_half->p_n_major,
		p_Te, // T_k+1/2 just calculated
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_e,
		p_nu_e,
		NT_addition_rates_d,
		pX_half->p_AreaMajor,
		p_boolarray2,
		p_boolarray_block,
		false,//bUseMask,
		2);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate e");

	SetConsoleTextAttribute(hConsole, 15);
	iHistory++;
	/*
	//kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> > (
	//	pX_half->p_info,
	//	pX_half->p_n_minor,
	//	pX_half->p_T_minor,

	//	p_kappa_n,
	//	p_kappa_i,
	//	p_kappa_e,
	//	p_nu_i,
	//	p_nu_e		// define all these.
	//	); 
	//Call(cudaThreadSynchronize(), "cudaTS Calculate_kappa_nu(pXhalf)");
	// 

	// not ready to try and change .. means changing the advance also ...

	
	// 8th Nov 2019. Experiment to see if we can sometimes get away with midpoint step,
	// ie use forward step here off of the half-step of backward.
	// Try stability test:

	// 1. Try midpoint step to pX_target->T_major
	// -- we can if necessary shield against negatives in the same way as we do for bwd.
	// If some T went negative anyway then we scrap the midpoint attempt.
	// 2. Run NTrates on new position. If something reversed sign and increased in magnitude, we scrap the midpoint attempt.
	// Need to think about that carefully in terms that we should produce NTrates overall in the same way.
	// So that requires the rollaround to get rid of negatives.
	
	// // a. Longitudinal. Calc putative T, Check for negative. If they exist, give up on this way entirely.
	// // b. Full with rollaround. 
	// // c. Advance to target system
	// // d. On target,  Calc putative T, check for negative. If they exist, give up on this way.
	// e. Full with rollaround to get rid of negative. What is timestep?

	// -- or,
	// i. just take the ROC we already calculated and use this. 
	// c. Advance to target system.
	// ii. Check for negative T. If they exist, forget it, do backward.
	// iii. on target system, use existing boolarray to calc new rates, full vs longi. 
	// iv. Now compare with the NTRates we used to get us here. If we reversed sign and 
	// it is greater magnitude now, then give up and do backward.

	// maybe just those places need to be backward? That could be a quick solve, set
	// most values to epsilon == 0 and value is sorted. Good project for a day.
	
	// It failed, 1 negative value.

	// New effort. Let boolarray store where we want to do a solve - expand it to 3*NVERTS.
	// Then we just need to execute carefully. 
	
	// 1a. Find how many negative T and switch on bool
	// 1b. Find how many switched over stability and switch on bool
	// 1c. Set their neighbours to on also.

	// . Main task in CG routine is Calc d/dt(NT) given T -- we need to skip for most
	// we can let it exist or just never use it
	// We then calc epsilon which we should leave at 0 and make sure any additive regressor
	// is 0 outside of the "on" cells.
	// 
	// Calc d/dtNT is easily the most expensive routine involved.
	// We should detect create block-level flags so that whole blocks can be switched off,
	// as well as individual flags, which will leave NTrates unamended and uncalculated.
	// It will have to load shared data for tiles that have points in, but that does not 
	// mean that we don't save anything by only running those points. We do.
	// 
	// It should also be a quicker solve if there are effectively only a smaller number
	// of equations --- which may not even connect.
	

	bool btemp;
	bool bBackward = false;
	// 1a. Find how many negative T and switch on bool
	cudaMemset(p_boolarray2, 0, sizeof(bool) * 3 * NUMVERTICES);
	// p_boolarray is still needed because it shows how to create midpoint step: use longi
	// and after we do backward longi solve we will need to decide whether those points are longi or full as in general.
	
	// This is a mistaken decision, we shouldn't be just looking at whether we get negative this way.
	// We should be doing it more simply: take our half-time system, recalculate heat flows, ask if we are getting negative.

	// I see the logic: our first half-step was actually a backward step.
	// That isn't a necessary way of doing it.
	
	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);





	don't need to do any of the following, just call geometric heat calc.'



		;

	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
	kernelCreatePutativeTandsave << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		HEATSTEP,
		pX_half->p_info,
		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_AreaMajor,
		store_heatcond_NTrates,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // THE FORWARD MOVE using stored NTrates
		p_boolarray2  // store here 1 if it is below 0
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeTandsave");
//	 
//	cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
//		cudaMemcpyDeviceToHost);
//	printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	// For safety let's check again properly: are there negative T or not?
	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajorClever);
	kernelReturnNumberNegativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelReturnNumberNegativeT");
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	long iNegativeT = 0;
	for (i = 0; (i < numTilesMajorClever); i++)
		iNegativeT += p_longtemphost[i];
	if (iNegativeT > 0) {
		printf("%d negatives!", iNegativeT);
		bBackward = true;
	}
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,				// We do heat from cc -- seriously?

		pX_half->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
		p_boolarray, // array of which ones require longi flows
			 		 // 2 x NMAJOR
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e,
		NT_addition_rates_d,
		pX_half->p_AreaMajor,
		true,
		p_boolarray2,
		p_boolarray_block,
		false // recalculate everything using pX_target->p_T_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate test");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajor);
	kernelCompareForStability_andSetFlag << <numTilesMajor, threadsPerTileMajor >> >
		(
			pX_half->p_info,
			store_heatcond_NTrates,
			NT_addition_rates_d,
			p_longtemp,
			p_boolarray2 // store 1 if it has reversed and amplified
			);
	Call(cudaThreadSynchronize(), "cudaTS CompareStability");

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);


	long iReversals = 0;
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajor; iTile++)
		iReversals += p_longtemphost[iTile];
	if (iReversals > 0)
	{
		printf("%d reversals! \n", iReversals);
		bBackward = true;
	};
	
	if (bBackward == false) {
		printf("midpoint accepted in entirety\n");
		// Are we done? CHECK:
		// We got NT_addition_rates_d
		
	//	The problem is we haven't ever set move1
		// first time we get here, set move1 = move2
		// if we are here for 2nd time, set iHistory = 1


		// There is a cleverer way. Before we set p_store_T_move2,
		// if iHistory == 0 then set p_store_T_move1 to it 
		// and if we are NOT setting it to just zero,
		// 

		if (iTemp == 0) {
			
			iTemp = 1;
			cudaMemcpy(p_store_T_move1, p_store_T_move2, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		} else {
			iHistory++; // ready to use move1, move2. move1 is older.
		};

	}
	else {
		//	
		//		kernelSetNeighboursBwd << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		//			pX_half->p_info,
		//			pX_half->p_izNeigh_vert,
		//			p_boolarray2);
		//		Call(cudaThreadSynchronize(), "cudaTS kernelSetNeighboursBwd");
		//
				// doesn't work yet.
//		cudaMemcpy(&btemp, &(p_boolarray2[NUMVERTICES*2 + VERTCHOSEN]), sizeof(bool), cudaMemcpyDeviceToHost);
//		SetConsoleTextAttribute(hConsole, 13);
//		printf("\nbool_e[%d] %d \n\n", VERTCHOSEN, (btemp ? 1 : 0));
//		SetConsoleTextAttribute(hConsole, 15);

		cudaMemset(p_boolarray_block, 0, sizeof(bool)*numTilesMajorClever);
		cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajorClever * 3);
		kernelSetBlockMaskFlag_CountEquations_reset_Tk << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_boolarray2,
			p_boolarray_block,
			p_longtemp,
			HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelSetBlockMaskFlag");
		
		iEquations[0] = 0;  iEquations[1] = 0; iEquations[2] = 0;
		cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever * 3, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			iEquations[0] += p_longtemphost[iTile * 3];
			iEquations[1] += p_longtemphost[iTile * 3 + 1];
			iEquations[2] += p_longtemphost[iTile * 3 + 2];
		};
		cudaMemcpy(&btemp, &(p_boolarray2[NUMVERTICES + 22351]), sizeof(bool), cudaMemcpyDeviceToHost);
		SetConsoleTextAttribute(hConsole, 13);
		printf("\nbool[%d] %d \n\n", NUMVERTICES + 22351, (btemp ? 1 : 0));
		SetConsoleTextAttribute(hConsole, 15);
		 
		// IMPORTANT:

		// Block flag can only be used when << < numTilesMajorClever, threadsPerTileMajorClever >> >

		over_iEquations_n = (iEquations[0] > 0) ? (1.0 / (f64)iEquations[0]) : 1.0;
		over_iEquations_i = (iEquations[1] > 0) ? (1.0 / (f64)iEquations[1]) : 1.0;
		over_iEquations_e = (iEquations[2] > 0) ? (1.0 / (f64)iEquations[2]) : 1.0;
		printf("iEquations %d %d %d \n", iEquations[0], iEquations[1], iEquations[2]);

		// Backward Euler step:
	//	 cudaMemcpy(pX_target->p_T_minor + BEGINNING_OF_CENTRAL, 
	//		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// Except we don't necessarily want to have done that.
		// ++ Keep the proper values in T_minor that we get from forward step. ++


		if (iHistory > 0) {

			// Hang on. We are solving here for the system to use for heat flows, is that correct?
			// Yep.

			// Then we can still use these flows along with the whole-length step
			// hsub is not a part of the flow, it's a flux.
			 
			SetConsoleTextAttribute(hConsole, 11);
			RegressionSeedTe(HEATSTEP,
				p_store_T_move1, p_store_T_move2,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);
			SetConsoleTextAttribute(hConsole, 10);
		};
		// We stored result in pX_target->p_T_minor?
		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);


		
		// If we want to change back to JRLS then we need to
		// get rid of re-pack as well, remember!


#define JnLS

#ifndef JnLS
		do {
			iSuccess = RunBackwardJLSForHeat(
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATSTEP,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);

			if (iSuccess != 0)  RunBackwardForHeat_ConjugateGradient(
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATSTEP,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);

			//iSuccess = RunBwdJnLSForHeat(p_Tnk, p_Tn, 
			//	HEATSTEP, pX_half, true,
			//	0, 
			//	p_kappa_n, 
			//	p_nu_i // not used
			//);

		} while (iSuccess != 0);
#endif

#ifdef JnLS
		kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, pX_target->p_T_minor + BEGINNING_OF_CENTRAL); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k+1 A");

		kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tnk, p_Tik, p_Tek, HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k");

		if (iEquations[0] > 0) {
			//printf("Neutral solve:\n"); // should have a way to skip if iEquations == 0
			do {
			
				iSuccess = RunBwdRnLSForHeat(p_Tnk, p_Tn, 
					HEATSTEP, pX_half, true,
					0, 
					p_kappa_n, 
					p_nu_i // not used
				);

			} while (iSuccess != 0);
			
		};
		if (iEquations[1] > 0) {
			printf("Ion solve:\n");
			do {
				iSuccess = RunBwdRnLSForHeat(p_Tik, p_Ti,
					HEATSTEP, pX_half, true,
					1,
					p_kappa_i,
					p_nu_i);
			} while (iSuccess != 0);
			
		};

		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		if (iEquations[2] > 0) {
			printf("Electron solve:\n");
			do {
				iSuccess = RunBwdRnLSForHeat(p_Tek, p_Te,
					HEATSTEP, pX_half, true,
					2,
					p_kappa_e,
					p_nu_e);
			} while (iSuccess != 0);
			
		};

//		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
//			cudaMemcpyDeviceToHost);
//		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		kernelPackupT3 << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(pX_target->p_T_minor + BEGINNING_OF_CENTRAL, p_Tn, p_Ti, p_Te); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelPack k+1");

//		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
	//		cudaMemcpyDeviceToHost);
		//printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);


#endif
		if (!DEFAULTSUPPRESSVERBOSITY)
		{
			SetConsoleTextAttribute(hConsole, 14);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNnTn rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
			SetConsoleTextAttribute(hConsole, 15);
		};

		SetConsoleTextAttribute(hConsole, 15);

		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
				(p_store_T_move1,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS subtractT3");

		iHistory++; // we have now been through this point.

		//debug:
		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("solved Te[%d]: %1.9E\n", VERTCHOSEN, tempf64);
		if (tempf64 < 0.0) {
			printf("4getch");  getch(); getch(); getch(); getch(); PerformCUDA_Revoke(); printf("end");
			while (1) getch();
		}

		// Something to know : we never zero "NT_addition_rates" in the routine.
		// So we need to do it outside.
		
		cudaMemcpy(NT_addition_rates_d_temp, store_heatcond_NTrates, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES * 2); // initially allow all flows good
		
		// This is the tricky bit --- we shouldn't allow all flows good if we did masked flow longi?
		// Basically for masked points we do NOT want to do any of this.
		// We want to accept the d/dt(NT) that we already went with.
		
		// pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		// & store_heatcond_NTrates
		// contain the relevant data.

		cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d_temp[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d_temp Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		// We need to basically wipe over NT_addition_rates_d_temp with 0 wherever it's an active cell.

		kernelSelectivelyZeroNTrates << <numTilesMajorClever, threadsPerTileMajorClever >> >(
			NT_addition_rates_d_temp,
			p_boolarray2
		);
		Call(cudaThreadSynchronize(), "cudaTS SelectivelyZeroRates");
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d_temp[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d_temp Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		bool bContinue;

		int iPass = 0;
		do {
			printf("iPass %d :\n", iPass);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			// reset NTrates:
			cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				pX_half->p_info,
				pX_half->p_izNeigh_vert,
				pX_half->p_szPBCneigh_vert,
				pX_half->p_izTri_vert,
				pX_half->p_szPBCtri_vert,
				pX_half->p_cc,

				pX_half->p_n_major,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
				p_boolarray, // array of which ones require longi flows
							       // 2 x NMAJOR
				pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d, // This will increase it!
				pX_half->p_AreaMajor,
				(iPass == 0) ? false : true,
				p_boolarray2,
				p_boolarray_block,
				true // assume p_boolarray2, p_boolarray_blocks have been set : do nothing for already-set values
				);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
			// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
	
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				HEATSTEP,
				pX_half->p_info,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
				pX_half->p_n_major,
				pX_half->p_AreaMajor,
				NT_addition_rates_d,
				p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
							 // 2x NMAJOR
				p_bFailed, // did we hit any new negative T to add
				p_boolarray2,
				p_boolarray_block,
				true // only bother with those we are solving for.
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");
		
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			bContinue = false;
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			int i;
			for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
			if (i < numTilesMajorClever) bContinue = true;
			iPass++;
		} while (bContinue);
	}
	*/

	if (!DEFAULTSUPPRESSVERBOSITY)
	{

		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	if (bGlobalSaveTGraphs) {
		// Store in Tgraph1, the conductive dT/dt
		DivideNeTe_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[0]
		);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	};

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
	// PUT BACK THE n,T MINOR FOR CENTROIDS THAT WE GOT FROM CALLING FOR SHARD MODEL:

	cudaMemcpy(pX_half->p_n_minor, pX_target->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_T_minor, pX_target->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToDevice);

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);

	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		HEATSTEP,
		pX_half->p_info,
		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		HEATBASESYST->p_n_major,
		pX_half->p_AreaMajor,
		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,

		HEATBASESYST->p_vie + BEGINNING_OF_CENTRAL,
		HEATBASESYST->p_v_n + BEGINNING_OF_CENTRAL,

		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		true
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	// PERIODIC NEGLECTED:
	Collect_Ntotal_major << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMinor,
		p_temp1, // Ntotal major
		p_temp2  // Nntotal major
		);
	Call(cudaThreadSynchronize(), "cudaTS Gather Ntotal");
	  
	Augment_dNv_minor << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		p_temp1,
		p_temp2,
		pX_half->p_AreaMinor,
		p_temp3_1, p_temp3_2, p_temp3_3,
		p_MAR_neut, 
		p_MAR_ion,
		p_MAR_elec); 
	Call(cudaThreadSynchronize(), "cudaTS Augment_dNv_minor");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post ionization : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post ionization : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
	};

	// DEBUG:
	cudaMemcpy(&tempf64, &(p_MAR_neut[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	if (tempf64 != tempf64) {
		printf("NaN encountered: p_MAR_neut[%d].z", CHOSEN);
		SafeExit(100000001);
	}
	

	if (bGlobalSaveTGraphs) {
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_elec_ionization_major_stored, p_MAR_elec + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss");

		// Pre-store existing:
		cudaMemcpy(p_MAR_ion_visc_major_stored,
			p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_visc_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	}
	if (bGlobalSaveTGraphs) {
		// Store in Tgraph2, the ionization dT/dt
		DivideNeTeDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[1]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);


		DivideMARDifference_get_accel_y << <numTilesMajor, threadsPerTileMajor >> > (
				p_MAR_ion + BEGINNING_OF_CENTRAL,
				p_MAR_elec + BEGINNING_OF_CENTRAL,
			p_MAR_ion_temp_central,
			p_MAR_elec_temp_central,
				this->p_n_minor + BEGINNING_OF_CENTRAL,
				this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
				p_accelgraph[9]
				);
		Call(cudaThreadSynchronize(), "cudaTS DivideMARdiff_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
				cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
				cudaMemcpyDeviceToDevice);
		
	};
	if (!DEFAULTSUPPRESSVERBOSITY)
	{

		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	//
	//cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("%d NiTi rate %1.10E \n", VERTCHOSEN, tempf64);
	//cudaMemcpy(&tempf64, &(p_MAR_elec[VERTCHOSEN + BEGINNING_OF_CENTRAL].z), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\nMAR_elec.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	//

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			pX_half->p_info,
			pX_half->p_vie,
			pX_half->p_v_n,
			pX_half->p_T_minor,
			pX_half->p_tri_neigh_index,
			pX_half->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

#ifdef PRECISE_VISCOSITY
	
	// . Calculate ROC given ita that we assign everywhere.
	// . If ROC is fast then include it
	
	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor, pX_half->p_T_minor,
		p_nu_i, p_nu_e, p_nu_n,
		p_ita_i, p_ita_e, p_ita_n,
		0,0); // No exclusions
	Call(cudaThreadSynchronize(), "cudaTS ita");

	// Method:
	// 1. Create time-derivative based on t_k (or t_k+1/2 later on); advance v putatively.

	cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d3, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_3, 0, sizeof(NTrates)*NUMVERTICES);
	
	// Here we are doing the RK2 step so we use pX_half->v

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_vie,
		// For neutral it needs a different pointer.
		pX_half->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_half->p_B,
		p_MAR_ion3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		1, m_ion_, 1.0 / m_ion_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_vie,
		// For neutral it needs a different pointer.
		pX_half->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_half->p_B,
		p_MAR_elec3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");
	
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_v_n,
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_n, // ita
		p_nu_n,
		p_MAR_neut3, // just accumulates
		NT_addition_rates_d_3,
		NT_addition_tri_d3);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	// These are the forward flows and heatings which we need to save, in the forward region.

	kernelPutativeAccel << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		pX_half->p_info,
		this->p_vie,
		this->p_v_n, // X_k
		pX_target->p_vie,
		pX_target->p_v_n,
		pX_half->p_n_minor, this->p_AreaMinor,
		p_MAR_ion3,
		p_MAR_elec3,
		p_MAR_neut3);
	Call(cudaThreadSynchronize(), "cudaTS PutativeAccel");


	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// 2 . Take dv/dt at the destination (pX_target):
	// NOTE:
	// We could try including frictional part if that damps the explosive oscillations.  !
	// That is fair -- we include it in putative accel and can see if then there is a countersigned new dv/dt.

	cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
	// 2 is the throwaway one.
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_target->p_vie,
		pX_target->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_half->p_B, 
		p_MAR_ion2, // accumulates
		NT_addition_rates_d_2, NT_addition_tri_d2,
		1, m_ion_, 1.0 / m_ion_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_target->p_vie,
		pX_target->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_half->p_B,
		p_MAR_elec2, // accumulates
		NT_addition_rates_d_2, NT_addition_tri_d2,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_target->p_v_n,
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_n, p_nu_n,
		p_MAR_neut2, // just accumulates
		NT_addition_rates_d_2,
		NT_addition_tri_d2);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// 3.4. Make a test for an overshooting criterion; expand flag region around it.

	CallMAC(cudaMemset(p_Selectflag, 0, sizeof(int)*NMINOR));
	CallMAC(cudaMemset(p_SelectflagNeut, 0, sizeof(int)*NMINOR));
	kernelTest_Derivatives << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		p_MAR_ion3,		p_MAR_elec3,		p_MAR_neut3,
		p_MAR_ion2,		p_MAR_elec2,		p_MAR_neut2,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_n_minor, this->p_AreaMinor,
		p_Selectflag, p_SelectflagNeut // this is the output
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelTest_Derivatives");

	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		1);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");
	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		2);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");
	kernelExpandSelectFlagIta << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_izTri_vert, this->p_izNeigh_TriMinor,
		p_Selectflag, p_SelectflagNeut,
		3);
	Call(cudaThreadSynchronize(), "cudaTS ExpandSelectFlagIta");

	kernelCountNumberMoreThanZero << <numTilesMinor, threadsPerTileMinor >> > (
		p_Selectflag, p_SelectflagNeut,
		p_blockTotal1, p_blockTotal2);
	cudaMemcpy(p_longtemphost, p_blockTotal1, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_longtemphost2, p_blockTotal2, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
	total1 = 0; total2 = 0;
	for (int iTile = 0; iTile < numTilesMinor; iTile++)
	{
		total1 += p_longtemphost[iTile];
		total2 += p_longtemphost2[iTile];
	};
	printf("Full step : Number selectflag>0 %d selectflagneut>0 %d \n\n",
		total1, total2);
	
	kernelZeroSelected << <numTilesMinor, threadsPerTileMinor >> > (
		p_MAR_ion3,
		p_MAR_elec3,
		p_MAR_neut3,
		NT_addition_rates_d_3, NT_addition_tri_d3,
		p_Selectflag, p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS ZeroSelected");

	// 5. Update variables going into backward solve:

	// where Selectflag == 0, v_k_modified = v_updated, but it isn't important since we don't look there for epsilon anyway
	// where Selectflag > 0, v_k_modified = v_k (+/-) the flows to only selectflag==0 neighbours. Times h/N.
	
	// But accumulate fixed flows into p_MAR_3:

	//kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric << <numTriTiles, threadsPerTileMinor >> > (
	//	pX_half->p_info,
	//	pX_half->p_v_n,
	//	this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
	//	p_ita_n, // ita
	//	p_nu_n,
	//	p_MAR_neut3, // just accumulates
	//	NT_addition_rates_d_3,
	//	NT_addition_tri_d3);
	//Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");




	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species___fixedflows_only << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_vie,
		pX_half->p_v_n, // not used
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_half->p_B,
		p_MAR_ion3, // accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		1, m_ion_, 1.0 / m_ion_,
		p_Selectflag
		);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");
	
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species___fixedflows_only
		<< <numTriTiles, threadsPerTileMinor >> > (
			this->p_info,
			pX_half->p_vie,
			pX_half->p_v_n, // not used
			this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
			p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			pX_half->p_B,
			p_MAR_elec3, // accumulates
			NT_addition_rates_d_3, NT_addition_tri_d3,
			2, m_e_, 1.0 / m_e_,
			p_Selectflag
			);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric___fixedflows_only << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		pX_half->p_v_n,
		this->p_izTri_vert, this->p_szPBCtri_vert, this->p_izNeigh_TriMinor, this->p_szPBC_triminor,
		p_ita_n, // ita
		p_nu_n,
		p_MAR_neut3, // just accumulates
		NT_addition_rates_d_3, NT_addition_tri_d3,
		p_SelectflagNeut
		);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

	// The following function is modelled on epsilon calc and must be updated with it.
	kernelCreate_v_k_modified_with_fixed_flows << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		this->p_info,
		p_vie_modified_k,
		p_v_n_modified_k,
		this->p_vie,
		this->p_v_n,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_Selectflag,
		p_SelectflagNeut,
		p_MAR_ion3, p_MAR_elec3, p_MAR_neut3,
		pX_half->p_n_minor,
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS Create v_k_modified");

	cudaMemcpy(pX_target->p_vie, p_vie_modified_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_v_n, p_v_n_modified_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	// This is to give us a good starting point for applying a seed that we created based on the move between systems.

	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_n_minor, // Now on centroids so need to have put it back
		pX_half->p_T_minor,
		p_nu_i,
		p_nu_e,
		p_nu_n,
		p_ita_i,
		p_ita_e,
		p_ita_n,
		p_Selectflag, p_SelectflagNeut);
	Call(cudaThreadSynchronize(), "cudaTS ita");

	if (0)//DEFAULTSUPPRESSVERBOSITY == false)
	{
		// NOW DO GRAPH
		GlobalCutaway = false;

		//kernelSplit_vec2 << <numTilesMinor, threadsPerTileMinor >> > (p_epsilon_xy, p_nu_i, p_nu_e);
		//Call(cudaThreadSynchronize(), "cudaTS splitvec2");

		SetActiveWindow(hwndGraphics);
		cudaMemcpy(p_temphost1, p_ita_i + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_ita_e + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_nu_i + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost4, p_nu_e + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost5, p_nu_n + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost6, p_ita_n + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		
		Vertex * pVertex;
		plasma_data * pdata;
		char buffer[2303];

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = (p_temphost1[iVertex] == 0.0) ? 10.0 : 0.0;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "ita_ion");
		Graph[0].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost2[iVertex];
			pdata->temp.y = (p_temphost2[iVertex] == 0.0) ? 10.0 : 0.0;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "ita_elec");
		Graph[1].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost6[iVertex];
			pdata->temp.y = (p_temphost6[iVertex] == 0.0) ? 10.0 : 0.0;
			++pVertex;
			++pdata;
		}
		sprintf(buffer, "ita_neut");
		Graph[2].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);
		SetActiveWindow(hwndGraphics);
		ShowWindow(hwndGraphics, SW_HIDE);
		ShowWindow(hwndGraphics, SW_SHOW);
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		printf("done graphs\npress o to continue\n");

		while (getch() != 'o');
	};

	GlobalCutaway = true;
	
	// DEBUG:
	cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NMINOR,
		cudaMemcpyDeviceToDevice);

	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_3);
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT fwd");

	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d_temp, NT_addition_rates_d_3);// FOR GRAPHING DEBUG
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT fwd"); // FOR GRAPHING DEBUG

	if (total1 != 0) {
		RunBackwardR8LSForViscosity_Geometric(
			p_vie_modified_k, pX_target->p_vie, Timestep, pX_half);
	} else {
		// Make sure data is zeroed.
		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
	};

	// NTrates2 will get smooshed so do addition here & save from tris
	cudaMemcpy(NT_addition_tri_d, NT_addition_tri_d2, sizeof(NTrates)*NUMVERTICES * 6, cudaMemcpyDeviceToDevice);
	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_2);
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT");

	if (total2 != 0) {
		RunBackward8LSForNeutralViscosity_Geometric(
			p_v_n_modified_k, pX_target->p_v_n, Timestep, pX_half);		
	} else {
		// Make sure data is zeroed.
		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
		cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
	};
	kernelAddNT << <numTilesMajor, threadsPerTileMajor >> >
		(NT_addition_rates_d, NT_addition_rates_d_2);  // the data from neutral bwd
	Call(cudaThreadSynchronize(), "cudaTS add dbydt NT neut bwd");


	// Why do we collect Nsum? Because we wanted to apportion per density?

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		pX_half->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp1, p_temp2);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 1");

	kernelTransmit_3x_HeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		pX_half->p_n_minor,
		this->p_AreaMajor,
		p_temp1, p_temp2,
		NT_addition_rates_d, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd");

#else

	kernelCollect_Up_3x_HeatIntoVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_tri_corner_index, // to see which corner we are.
		NT_addition_rates_d, // dest
		NT_addition_tri_d, // the data from i&e bwd 
		NT_addition_tri_d2, // the data from neutral bwd 
		NT_addition_tri_d3 // the forward data
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat neut bwd");


#endif

	// create putative move as pX_half? Using rates 2 and 3 ?
	// Yes.

	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_ion2, p_MAR_ion3);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_elec2, p_MAR_elec3);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_neut2, p_MAR_neut3);
	Call(cudaThreadSynchronize(), "cudaTS addMAR");

	// Get seed:
	kernelPutativeAccel << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		this->p_vie,
		this->p_v_n,
		pX_target->p_vie,
		pX_target->p_v_n,
		pX_half->p_n_minor, this->p_AreaMinor,
		p_MAR_ion2,
		p_MAR_elec2,
		p_MAR_neut2);
	Call(cudaThreadSynchronize(), "cudaTS PutativeAccel");
	// Ensure that we put the whole move into storage, but that when we use it, we select only the solved-for cells.	
	Subtract_V4 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move4, this->p_vie, pX_target->p_vie);
	Call(cudaThreadSynchronize(), "cudaTS Subtract_V4");
	SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3_2, this->p_v_n, pX_target->p_v_n);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");

	  // Now add the combined acceleration into the acceleration we use.
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_ion, p_MAR_ion2);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_elec, p_MAR_elec2);
	kernelAdd3 << <numTilesMinor, threadsPerTileMinor >> > (p_MAR_neut, p_MAR_neut2);
	Call(cudaThreadSynchronize(), "cudaTS addMAR");

	bViscousHistory = true; // We have now passed through this point. Use this move as regr for both parts of next move.


	//if (DEFAULTSUPPRESSVERBOSITY == false)
	if (0) {
		kernelSubtractNiTiCheckNeg << <numTilesMajor, threadsPerTileMajor >> >
			(NT_addition_rates_d_temp2, NT_addition_rates_d, p_temp1,
				this->p_n_minor + BEGINNING_OF_CENTRAL,
				this->p_AreaMajor);
		Call(cudaThreadSynchronize(), "cudaTS SubtractNiTi");
	};

	if (0) {
		printf("\nDEBUG: Look for negs!\n");
		cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMinor);
		kernelFindNegative_dTi << <numTilesMinor, threadsPerTileMinor >> > (
			NT_addition_rates_d, NT_addition_tri_d,
			p_longtemp);
		Call(cudaThreadSynchronize(), "cudaTS kernelFindNegative_dTi");
		cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
		long negs = 0;
		for (i = 0; i < numTilesMinor; i++) negs += p_longtemphost[i];
		printf("at least %d negs\n", negs);
	};
	

	if (bGlobalSaveTGraphs) {
		// Store in Tgraph3, the viscous dT/dt
		DivideNeTeDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[2]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
		DivideMARDifference_get_accel_y << <numTilesMajor, threadsPerTileMajor >> > (
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL,
			p_MAR_ion_temp_central,
			p_MAR_elec_temp_central,
			this->p_n_minor + BEGINNING_OF_CENTRAL,
			this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
			p_accelgraph[8] // viscosity y
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideMARdiff_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
	};
	// This must be where most runtime cost lies.
	// 2 ways to reduce: reduce frequency to 1e-12, introduce masking.

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NiTi rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

#endif 


	if (bGlobalSaveTGraphs) {
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_ion_visc_major_stored, p_MAR_ion + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss1");
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_elec_visc_major_stored, p_MAR_elec + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss2");

	}

	if (bGlobalSaveTGraphs == false) {
		kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> > (
			Timestep,
			this->p_info + BEGINNING_OF_CENTRAL,
			this->p_n_major,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			NT_addition_rates_d,
			pX_half->p_n_major,  // ?
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

			pX_half->p_vie + BEGINNING_OF_CENTRAL,
			pX_half->p_v_n + BEGINNING_OF_CENTRAL,

			this->p_AreaMajor,

			pX_target->p_n_major,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_B + BEGINNING_OF_CENTRAL
			); // do check for T<0
		Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");
	} else {
		kernelAdvanceDensityAndTemperature_noadvectioncompression_Copy << <numTilesMajor, threadsPerTileMajor >> > (
			Timestep, 
			this->p_info + BEGINNING_OF_CENTRAL,
			this->p_n_major,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			NT_addition_rates_d,
			pX_half->p_n_major,  // ?
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

			pX_half->p_vie + BEGINNING_OF_CENTRAL,
			pX_half->p_v_n + BEGINNING_OF_CENTRAL,

			this->p_AreaMajor,

			pX_target->p_n_major,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_B + BEGINNING_OF_CENTRAL,
			p_Tgraph[3], // resistive/fric
			p_Tgraph[4], // soak
			p_Tgraph[5],  // dTe/dt total
			p_Tgraph[7]   // dnTe/dt
			); // do check for T<0
		Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");
	};
	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
		pX_target->p_cc,
		pX_target->p_tri_corner_index,
		pX_target->p_tri_periodic_corner_flags,
		false
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");
	cudaMemcpy(pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// Fill in full shard-based n_minor for destination system, to match the others:

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_target->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	if (!DEFAULTSUPPRESSVERBOSITY) {
		printf("DebugNaN pX_target\n");
		DebugNaN(pX_target);
	}
	
	// ============================================================

	f64 starttime = evaltime;
	printf("run %d ", runs);
	cudaEventRecord(middle, 0);
	cudaEventSynchronize(middle);

	// BETTER:
	// Just make this the determinant of how long the overall timestep is;
	// Make supercycle: advection is not usually applied.

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs");

	//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTriTiles, cudaMemcpyDeviceToHost);

	//// It should be universally true that coeffself is negative. Higher self = more negative Lap.
	//f64 mincoeffself = 0.0;
	//long iMin = -1;
	//for (iTile = 0; iTile < numTriTiles; iTile++)
	//{
	//	if (p_temphost1[iTile] < mincoeffself) {
	//		mincoeffself = p_temphost1[iTile];
	//		iMin = p_longtemphost[iTile];
	//	}
	//	//	printf("iTile %d iMin %d cs %1.12E \n", iTile, p_longtemphost[iTile], p_temphost1[iTile]);
	//}

	//f64 h_sub_max = 1.0 / (c_*sqrt(fabs(mincoeffself))); // not strictly correct - 
	//													 // e.g. with this at 7.3e-14, using 1e-13 as substep works (10% backward) ;
	//													 // with it at 6.4e-14 it does not work. 
	//													 // So the inflation factor that you can get away with, isn't huge.
	//printf("\nMin coeffself %1.12E iMin %d 1.0/(c sqrt(-mcs)) %1.12E\n", mincoeffself, iMin,
	//	h_sub_max);
	//// Comes out with sensible values for max abs coeff ~~ delta squared?
	//// Add in factor to inflate Timestep when we want to play around.

	//// iSubcycles = (long)(Timestep / h_sub_max)+1;
	//if (Timestep > h_sub_max*2.0) // YES IT IS LESS THAN 1x h_sub_max . Now that seems bad. But we are doing bwd so .. ???
	//{
	//	printf("\nAlert! Timestep > 2.0 h_sub_max %1.11E %1.11E \a\n", Timestep, h_sub_max);
	//}
	//else {
	//	printf("Timestep %1.11E h_sub_max %1.11E \n", Timestep, h_sub_max);
	//}
	    
	      
	//if (runs % BWD_SUBCYCLE_FREQ == 0) {
	//	printf("backward!\n");
	//	iSubcycles /= BWD_STEP_RATIO; // some speedup this way
	//} else {
	//	iSubcycles *= FWD_STEP_FACTOR;
	//}

	// Don't do this stuff --- just make whole step shorter.
	 
//	iSubcycles = SUBCYCLES; // 10
//	hsub = Timestep / (real)iSubcycles;
	
	cudaMemcpy(p_AAdot_start, this->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_vie_start, this->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_v_n_start, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	if (bGlobalSaveTGraphs) {
		cudaMemcpy(p_vie_k_stored, this->p_vie + BEGINNING_OF_CENTRAL, sizeof(v4)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	}

	
	GosubAccelerate(SUBCYCLES,//iSubcycles, 
		Timestep / (real)SUBCYCLES, // hsub

		pX_target, // pX_use

		pX_half // pX_intermediate
	);
	 
	cudaMemcpy(pX_target->p_AAdot, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_vie, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_v_n, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
		
	if (bGlobalSaveTGraphs) {
		   
		MeasureAccelxy_and_JxB_and_soak << <numTilesMajor, threadsPerTileMajor >> >(
			pX_target->p_vie + BEGINNING_OF_CENTRAL,
			this->p_vie + BEGINNING_OF_CENTRAL,
			Timestep,
			p_GradAz + BEGINNING_OF_CENTRAL,
			pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_v_n + BEGINNING_OF_CENTRAL,
			pX_target->p_v_n + BEGINNING_OF_CENTRAL,
			p_accelgraph[0], 
			p_accelgraph[1], // accel xy
			p_accelgraph[2],
			p_accelgraph[3], // vxB accel xy
			p_accelgraph[11], // grad_y Az
			p_accelgraph[6]
		);  
		Call(cudaThreadSynchronize(), "cudaTS MeasureAccelxy ");

		
		MeasureAccelz << <numTilesMajor, threadsPerTileMajor >> > (
			pX_target->p_info + BEGINNING_OF_CENTRAL,
			p_vie_start + BEGINNING_OF_CENTRAL,
			pX_target->p_vie + BEGINNING_OF_CENTRAL,
			p_v_n_start + BEGINNING_OF_CENTRAL,
			pX_target->p_v_n + BEGINNING_OF_CENTRAL,
			Timestep / (real)SUBCYCLES,
			p_GradAz + BEGINNING_OF_CENTRAL, // for v x B component
			p_GradTe + BEGINNING_OF_CENTRAL,
			pX_target->p_AAdot + BEGINNING_OF_CENTRAL, 
			p_AAdot_start + BEGINNING_OF_CENTRAL,
			p_LapAz + BEGINNING_OF_CENTRAL,
			 
			pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
			 
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_B + BEGINNING_OF_CENTRAL, // ...
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL,
			p_MAR_neut + BEGINNING_OF_CENTRAL,
			 
			pX_half->p_AreaMinor + BEGINNING_OF_CENTRAL,
			 
			p_arelz_graph[0], // aez-aiz
			p_arelz_graph[1], // MAR_ion effect
			p_arelz_graph[2], // MAR_elec effect
			p_arelz_graph[3], // Electromotive
			p_arelz_graph[4], // Inductive electromotive (using Adot_k+1)
			p_arelz_graph[5], // v x B force term
			p_arelz_graph[6],  // thermal force term
			p_arelz_graph[7], // friction to neutrals (updated velocities) 
			p_arelz_graph[8], // friction e-i (updated velocities)
			p_arelz_graph[9], // sum
			p_arelz_graph[10] // error in arelz-sum : should be ~ 0 ; verification test.
			);
		Call(cudaThreadSynchronize(), "cudaTS MeasureAccel z ");

//		cudaMemcpy(&tempf64, &(p_vie_target[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("vez [%d]  = %1.13E \n", VERTCHOSEN, tempf64);
//		cudaMemcpy(&tempf64, &(p_arelz_graph[3][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("arelz_Ezext [%d]  = %1.13E \n", VERTCHOSEN, tempf64);
//		cudaMemcpy(&tempf64, &(p_arelz_graph[4][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("arelz_Azdot [%d]  = %1.13E \n", VERTCHOSEN, tempf64);
//		cudaMemcpy(&tempf64, &(p_arelz_graph[5][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("arelz_v x B [%d]  = %1.13E \n", VERTCHOSEN, tempf64);
//		
//		cudaMemcpy(&tempf64, &(p_arelz_graph[9][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("arelz [%d]  = %1.13E \n", VERTCHOSEN, tempf64);
		
	}
	 
	// spit out to story file.

	if ((bGlobalSaveTGraphs)) // 100 lines per ns = 2000 lines
	{
		FILE * biller = fopen(STORYFILE, "a");

		// spit timestamp

		fprintf(biller, "%d evaltime %1.14E ", VERTCHOSEN, evaltime);

		nvals n;
		v4 v;
		f64_vec3 v_n;
		AAdot tempAAdot;
		// . spit n,v all species
		cudaMemcpy(&n, pX_target->p_n_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN,
			sizeof(nvals), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v, pX_target->p_vie + BEGINNING_OF_CENTRAL + VERTCHOSEN,
			sizeof(v4), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v_n, pX_target->p_v_n + BEGINNING_OF_CENTRAL + VERTCHOSEN,
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempAAdot, pX_target->p_AAdot + BEGINNING_OF_CENTRAL + VERTCHOSEN,
			sizeof(AAdot), cudaMemcpyDeviceToHost);
		fprintf(biller, "n %1.14E n_n %1.14E v_n %1.14E %1.14E %1.14E v %1.14E %1.14E viz %1.14E vez %1.14E EzStrength %1.14E Azdot %1.14E ",
			n.n, n.n_n, v_n.x, v_n.y, v_n.z, v.vxy.x, v.vxy.y, v.viz, v.vez, EzStrength_, tempAAdot.Azdot);
		// . spit arelz terms
		for (i = 0; i <= 10; i++) {
			cudaMemcpy(&tempf64, &(p_arelz_graph[i][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			fprintf(biller, "arelz%d %1.14E ", i, tempf64);
		}
		// . spit axy terms
		for (i = 0; i <= 11; i++) {
			cudaMemcpy(&tempf64, &(p_accelgraph[i][VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
			fprintf(biller, "axy%d %1.14E ", i, tempf64);
		}
		fprintf(biller, "\n");

		fclose(biller);
		biller = fopen(STORYFILE2, "a");

		// spit timestamp

		fprintf(biller, "%d evaltime %1.14E ", VERTCHOSEN2, evaltime);

		// . spit n,v all species
		cudaMemcpy(&n, pX_target->p_n_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN2,
			sizeof(nvals), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v, pX_target->p_vie + BEGINNING_OF_CENTRAL + VERTCHOSEN2,
			sizeof(v4), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v_n, pX_target->p_v_n + BEGINNING_OF_CENTRAL + VERTCHOSEN2,
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempAAdot, pX_target->p_AAdot + BEGINNING_OF_CENTRAL + VERTCHOSEN2,
			sizeof(AAdot), cudaMemcpyDeviceToHost);
		fprintf(biller, "n %1.14E n_n %1.14E v_n %1.14E %1.14E %1.14E v %1.14E %1.14E viz %1.14E vez %1.14E EzStrength %1.14E Azdot %1.14E ",
			n.n, n.n_n, v_n.x, v_n.y, v_n.z, v.vxy.x, v.vxy.y, v.viz, v.vez, EzStrength_, tempAAdot.Azdot);
		// . spit arelz terms
		for (i = 0; i <= 10; i++) {
			cudaMemcpy(&tempf64, &(p_arelz_graph[i][VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
			fprintf(biller, "arelz%d %1.14E ", i, tempf64);
		}
		// . spit axy terms
		for (i = 0; i <= 11; i++) {
			cudaMemcpy(&tempf64, &(p_accelgraph[i][VERTCHOSEN2]), sizeof(f64), cudaMemcpyDeviceToHost);
			fprintf(biller, "axy%d %1.14E ", i, tempf64);
		}
		fprintf(biller, "\n");

		fclose(biller);
	}

	 
	SetConsoleTextAttribute(hConsole, 15);
	
	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, Timestep);
	  
	fp = fopen("elapsed_ii.txt", "a");
	SetConsoleTextAttribute(hConsole, 13);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms ", elapsedTime);
	fprintf(fp, "runs %d Elapsed time : %f ms ", runs, elapsedTime);
	cudaEventElapsedTime(&elapsedTime, start, middle);
	printf("of which pre subcycle was %f ms \n", elapsedTime);
	fprintf(fp, "of which pre subcycle was %f ms \n", elapsedTime);
	SetConsoleTextAttribute(hConsole, 15);
	fclose(fp);

	nvals n;
	cudaMemcpy(&n, &(this->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d this n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_half->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_half n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_target->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_target n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);


	runs++; 
}  
 
void GosubAccelerate(long iSubcycles, f64 hsub, cuSyst * pX_use, cuSyst * pX_intermediate)
{
	
	//GlobalSuppressSuccessVerbosity = true;
	// DEBUG !

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	//char ch;
	//cudaMemcpy(&ch, &(pX_use->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_use holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_intermediate->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_intermediate holds char %d", ch);
	//getch();
	

	for (int iSubstep = 0; iSubstep < iSubcycles; iSubstep++)
	{
		// I suggest a better alternative not yet tried:
		// .. Advance J first with putative Azdot, given LapAz est at halftime
		// .. Advance (bwd) Az with subcycle steps
		// .. Go again for J,Adot given integral over time of LapAz.
		// ........ but just doing this for now.

		kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
			p_AAdot_start, // A_k
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS PullAz");
		evaltime += hsub; // t_k+1

		f64 Iz_prescribed_endtime = GetIzPrescribed(evaltime); // APPLIED AT END TIME: we are determining
#ifndef AZCG

		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");
			
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info, 
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");
#else
		kernelGetLap_minor_SYMMETRIC << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz,
			p_AreaMinor_cc,
			true // divide by Area
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");
#endif
		
		//This needs to be called every substep? Surely not.
		// do profiling.
		 
		SetConsoleTextAttribute(hConsole, 13);
		kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_MAR_neut, p_MAR_ion, p_MAR_elec,
			pX_intermediate->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?

			p_LapAz,
			p_GradAz,
			p_GradTe,
			pX_use->p_n_minor,  // questionable
			pX_use->p_T_minor,
			   
			p_vie_start,
			p_v_n_start,
			p_AAdot_start,   // dimension these & fill in above.
			pX_intermediate->p_AreaMinor, // NOT POPULATED FOR PXTARGET -- yes it should be
			  
			p_vn0,
			p_v0,
			p_OhmsCoeffs,
			p_Iz0_summands,
			p_sigma_Izz,
			p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
			true);
		Call(cudaThreadSynchronize(), "cudaTS kernelPopulateBackwardOhmsLaw ");
		 
		cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		f64 Iz0 = 0.0;
		f64 Sigma_Izz = 0.0;
		f64 Iz_k = 0.0;
		long iBlock;
		for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
		{
			Iz0 += p_Iz0_summands_host[iBlock];
			Sigma_Izz += p_summands_host[iBlock];
			Iz_k += p_temphost1[iBlock];
		}
		EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
		if (EzStrength_ != EzStrength_) {
			printf("EzStrength_ != EzStrength_endIzpresc %1.10E Iz0 %1.10E Sigma_Izz %1.10E\n",
				Iz_prescribed_endtime, Iz0, Sigma_Izz); 
			SafeExit(15576);
		};

		Set_f64_constant(Ez_strength, EzStrength_);

		f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
		Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
		// Electrons travel from cathode to anode so Jz is down in filament,
		// up around anode.
		printf("Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n", Iz0, Sigma_Izz, EzStrength_);

		if ((EzStrength_ > 1.0e5) || (EzStrength_ < -1.0e6)){
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				printf("Block %d : Iz0 = %1.10E        ~~      ", iBlock, p_Iz0_summands_host[iBlock]);
				if (iBlock % 3 == 0) printf("\n");
			}
			SafeExit(15589);
		}

		kernelCreateLinearRelationshipBwd_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_OhmsCoeffs,
			p_v0,
			p_LapAz,  // used for cancelling .. 
			pX_use->p_n_minor,
			p_denom_e,
			p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z,

			p_AAdot_start,

			pX_intermediate->p_AreaMinor, // because not populated in PXTARGET
			p_Azdot0,
			p_gamma
			); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
		Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationshipBwd ");

		kernelCreateExplicitStepAz << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			p_Azdot0,
			p_gamma,
			p_LapAz, // we based this off of half-time Az. --not any more, time t_k
			p_temp6); // = h (Azdot0 + gamma*LapAz)
		Call(cudaThreadSynchronize(), "cudaTS 	kernelCreateExplicitStepAz");

		// set p_storeAz to some useful value on the very first step.

		if (iHistoryAz > 0)
		{
			if (GlobalSuppressSuccessVerbosity == false) printf("RegressionSeedAz: --------\n------------------\n");

			RegressionSeedAz(hsub, p_Az, p_AzNext, p_temp6, stored_Az_move, p_Azdot0, p_gamma, p_LapCoeffself, pX_use);
			// Idea: regress epsilon(Az) on p_temp6, stored_Az_move, Jacobi(stored_Az_move);
			// Update p_AzNext as the result.
			// .
			// Do moves really have a low correlation with each other?
			// Save & analyse correls.
			// .
			// Alternative way: regress on states t_k and t_k-1 rather than difference.
			// Result there?
			// Or do 2 historic states, then Richardson+JR+JJR, then etc.
		}
		else {
			
			if (GlobalSuppressSuccessVerbosity == false) printf("CreateSeedAz: --------\n------------------\n");

			kernelCreateSeedAz << <numTilesMinor, threadsPerTileMinor >> >
				(hsub, p_Az, p_Azdot0, p_gamma, p_LapAz, p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS createSeed");
		}

		SetConsoleTextAttribute(hConsole, 10);
#ifdef AZCG
		
		SolveBackwardAzAdvanceCG(hsub, p_Az, p_Azdot0, p_gamma,
			p_AzNext, pX_intermediate); // pX_target);		
		SubtractVector << <numTilesMinor, threadsPerTileMinor >> >
			(stored_Az_move, p_Az, p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS subtract");
		iHistoryAz++; // we have now been through this point.
		cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		kernelGetLap_minor_SYMMETRIC << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz,// it doesn't really make any difference which syst -- no vertices moving
			p_AreaMinor_cc,
			true
		);
		Call(cudaThreadSynchronize(), "cudaTS GetLap AzSYMMETRIC");

#else

		SolveBackwardAzAdvanceJ3LS(hsub, p_Az, p_Azdot0, p_gamma,
			p_AzNext, p_LapCoeffself, pX_intermediate); // pX_target);
		GlobalSuppressSuccessVerbosity = true;
		SubtractVector << <numTilesMinor, threadsPerTileMinor >> >
			(stored_Az_move, p_Az, p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS subtract");
		iHistoryAz++; // we have now been through this point.
		cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz// it doesn't really make any difference which syst -- no vertices moving
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");
#endif
		
	//	cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	f64 sum = 0.0; 
	//	f64 sumVT = 0.0, sumTV = 0.0, sumTT = 0.0;
	//	for (int iTile = 0; iTile < numTriTiles; iTile++) {
	//		sum += p_temphost4[iTile];
	//		sumVT += p_temphost1[iTile];
	//		sumTV += p_temphost2[iTile];
	//		sumTT += p_temphost3[iTile];
	//	};
	//	printf("sum = %1.14E \n", sum);
	//	printf("sumVT = %1.14E sumTV = %1.14E sum TT = %1.14E\n", sumVT, sumTV, sumTT);
	//	getch();
	//	SetConsoleTextAttribute(hConsole, 15);
		// Lap Az is now known, let's say.
		// So we are again going to call PopOhms Backward -- but this time we do not wish to save off stuff
		// except for the v(Ez) relationship.
		SetConsoleTextAttribute(hConsole, 15);
		kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_MAR_neut, p_MAR_ion, p_MAR_elec,
			pX_intermediate->p_B,
			p_LapAz,
			p_GradAz, // THIS WE OUGHT TO TWEEN AT LEAST
			p_GradTe,
			pX_use->p_n_minor,  // this is what is suspect -- dest n
			pX_use->p_T_minor,

			p_vie_start,
			p_v_n_start,
			p_AAdot_start, // not updated...

			pX_intermediate->p_AreaMinor, // pop'd? interp?

			p_vn0,
			p_v0,
			p_OhmsCoeffs,

			p_Iz0_summands,
			p_sigma_Izz,
			p_denom_i,
			p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
			false);
		Call(cudaThreadSynchronize(), "cudaTS PopBwdOhms II ");

		// Might as well recalculate Ez_strength again :
		// Iz already set for t+hsub.
		cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		Iz0 = 0.0;
		Sigma_Izz = 0.0;
		Iz_k = 0.0;
		for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
		{
			Iz0 += p_Iz0_summands_host[iBlock];
			Sigma_Izz += p_summands_host[iBlock];
		}
		EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
		Set_f64_constant(Ez_strength, EzStrength_);

		neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
		Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
		// Electrons travel from cathode to anode so Jz is down in filament,
		// up around anode.

		if (EzStrength_ != EzStrength_) {
			printf("EzStrength_ %1.10E Iz_prescribed %1.10E Iz0 %1.10E sigma_Izz %1.10E \n",
				EzStrength_, Iz_prescribed_endtime, Iz0, Sigma_Izz);
			SafeExit(15762);
		}

		kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
//			pX_use->p_tri_corner_index,
			p_vn0,
			p_v0,
			p_OhmsCoeffs,

			p_AAdot_start,
			//(iSubstep == iSubcycles - 1) ? pX_use->p_n_minor:pX_intermediate->p_n_minor,
			pX_use->p_n_minor, // NOT OKAY FOR IT TO NOT BE SAME n AS USED THROUGHOUT BY OHMS LAW
			pX_intermediate->p_AreaMinor,  // Still because pXuse Area still not populated
					   // We need to go back through, populate AreaMinor before we do all these things.
					   // Are we even going to be advecting points every step?
					   // Maybe make advection its own thing.
			p_LapAz,
			p_AAdot_target,
			p_vie_target,
			p_v_n_target

	//		p_temp1, // Jz-
	//		p_temp2, // Jz+
	//		p_temp3  // LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
		
		kernelAdvanceAzBwdEuler << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			p_AAdot_start,
			p_AAdot_target,
			p_ROCAzduetoAdvection, false);
		Call(cudaThreadSynchronize(), "cudaTS kernelAdvanceAzBwdEuler ");
		SetConsoleTextAttribute(hConsole, 14);
		if (!DEFAULTSUPPRESSVERBOSITY) {

			cudaMemcpy(&tempf64, &(p_vie_target[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nvez p_vie_target [%d] : %1.13E\n", VERTCHOSEN + BEGINNING_OF_CENTRAL, tempf64);
		}
		// Set up next go: 
		if (iSubstep < iSubcycles - 1) {
			cudaMemcpy(p_AAdot_start, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_vie_start, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_v_n_start, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
		};

		cudaMemcpy(&tempf64, &(p_vie_target[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("p_vie_start[%d].vez = %1.10E \n", VERTCHOSEN + BEGINNING_OF_CENTRAL, tempf64);
		SetConsoleTextAttribute(hConsole, 15);


		//105190 CVAA vez - 1.76403661228E+08 v0 - 1.76505992629E+08 Ez_strength - 1.94208907560254E+00 sigma - 5.26914045086209E+04
		//	Azdot 4.216272377E+10 components: k 4.100999623E+10 h_use*(c*c*p_LapAz) 4.883047049E+08 hc4piJ 6.644228360E+08
		//	n viz vez 2.08647047583313E+11 - 3.03196304146681E+05 - 1.76403661228195E+08
		//	p_vie_start[105190].vez = -1.7641836247E+08

		// Makes no sense whatosever.


	}; // substeps
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}

void SafeExit(long code)
{
	printf("SafeExit called with code %d \n", code);
	BOOL boolean = DestroyWindow(hWnd);
	printf("DestroyWindow returned %d [0 = fail]\n", (boolean ? 1 : 0));
	printf("press x to exit");
	while (getch() != 'x');
	exit(100);
}

void PerformCUDA_Revoke()
{

	GlobalSuppressSuccessVerbosity = true;
	CallMAC(cudaFree(p_sum_product_matrix3));
	CallMAC(cudaFree(p_storeviscmove));
	CallMAC(cudaFree(p_dump));
	CallMAC(cudaFree(p_dump2));
	CallMAC(cudaFree(p__matrix_xy_i));
	CallMAC(cudaFree(p__matrix_xy_e));
	CallMAC(cudaFree(p__xzyzzxzy_i));
	CallMAC(cudaFree(p__xzyzzxzy_e));
	CallMAC(cudaFree(p__invmatrix));
	CallMAC(cudaFree(p__coeffself_iz));
	CallMAC(cudaFree(p__coeffself_ez));
	CallMAC(cudaFree(p__invcoeffselfviz));
	CallMAC(cudaFree(p__invcoeffselfvez));
	CallMAC(cudaFree(p__invcoeffself));
	CallMAC(cudaFree(p_regressors2));
	CallMAC(cudaFree(p_regressors_iz));
	CallMAC(cudaFree(p_regressors_ez));

	CallMAC(cudaFree(p_temp3_1));
	CallMAC(cudaFree(p_temp3_2));
	CallMAC(cudaFree(p_temp3_3));
	CallMAC(cudaFree(p_regressors));
	CallMAC(cudaFree(d_eps_by_dx_neigh_n));
	CallMAC(cudaFree(d_eps_by_dx_neigh_i));
	CallMAC(cudaFree(d_eps_by_dx_neigh_e));
	CallMAC(cudaFree(p_regressor_n));
	CallMAC(cudaFree(p_regressor_i));
	CallMAC(cudaFree(p_regressor_e));
	CallMAC(cudaFree(p_ROCMAR1));
	CallMAC(cudaFree(p_ROCMAR2));
	CallMAC(cudaFree(p_ROCMAR3));
	CallMAC(cudaFree(p_Effect_self_n));
	CallMAC(cudaFree(p_Effect_self_i));
	CallMAC(cudaFree(p_Effect_self_e));
	CallMAC(cudaFree(p_boolarray));
	CallMAC(cudaFree(p_store_T_move1));
	CallMAC(cudaFree(p_store_T_move2));
	CallMAC(cudaFree(store_heatcond_NTrates));
	CallMAC(cudaFree(p_store_NTFlux)); 
	CallMAC(cudaFree(sz_who_vert_vert));
	CallMAC(cudaFree(p_T_upwind_minor_and_putative_T));
	CallMAC(cudaFree(p_bool));
	CallMAC(cudaFree(p_nu_major));
	CallMAC(cudaFree(p_was_vertex_rotated));
	CallMAC(cudaFree(p_triPBClistaffected));
	CallMAC(cudaFree(p_MAR_neut));
	CallMAC(cudaFree(p_MAR_ion));
	CallMAC(cudaFree(p_MAR_elec));
	CallMAC(cudaFree(p_v0));
	CallMAC(cudaFree(p_vn0));
	CallMAC(cudaFree(p_sigma_Izz));
	CallMAC(cudaFree(p_Iz0));
	CallMAC(cudaFree(p_OhmsCoeffs));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta));
	CallMAC(cudaFree(p_sum_depsbydbeta_sq));
	CallMAC(cudaFree(p_sum_eps_eps));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_heat));
	CallMAC(cudaFree(p_sum_depsbydbeta_sq_heat));
	CallMAC(cudaFree(p_sum_eps_eps_heat));
	CallMAC(cudaFree(p_bFailed));
	CallMAC(cudaFree(p_Ax));

	CallMAC(cudaFree(p_Jacobi_n));
	CallMAC(cudaFree(p_Jacobi_i));
	CallMAC(cudaFree(p_Jacobi_e));
	CallMAC(cudaFree(p_epsilon_n));
	CallMAC(cudaFree(p_epsilon_i));
	CallMAC(cudaFree(p_epsilon_e));
	CallMAC(cudaFree(p_coeffself_n));
	CallMAC(cudaFree(p_coeffself_i));
	CallMAC(cudaFree(p_coeffself_e));
	CallMAC(cudaFree(p_d_eps_by_dbeta_n));
	CallMAC(cudaFree(p_d_eps_by_dbeta_i));
	CallMAC(cudaFree(p_d_eps_by_dbeta_e));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_n));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_i));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_e));

	CallMAC(cudaFree(p_Az));
	CallMAC(cudaFree(p_AzNext));
	CallMAC(cudaFree(p_LapAz));
	CallMAC(cudaFree(p_LapAzNext));
	CallMAC(cudaFree(p_LapCoeffself));
	CallMAC(cudaFree(p_LapJacobi));
	CallMAC(cudaFree(p_Jacobi_x));
	CallMAC(cudaFree(p_epsilon));
	
	CallMAC(cudaFree(p_Jacobi_heat));
	CallMAC(cudaFree(p_epsilon_heat));

	CallMAC(cudaFree(p_Azdot0));
	CallMAC(cudaFree(p_gamma));
	CallMAC(cudaFree(p_Integrated_div_v_overall));
	CallMAC(cudaFree(p_Div_v_neut));
	CallMAC(cudaFree(p_Div_v));
	CallMAC(cudaFree(p_Div_v_overall));
	CallMAC(cudaFree(p_ROCAzdotduetoAdvection));
	CallMAC(cudaFree(p_ROCAzduetoAdvection));
	CallMAC(cudaFree(p_GradAz));
	CallMAC(cudaFree(p_GradTe));

	CallMAC(cudaFree(p_one_over_n));
	CallMAC(cudaFree(p_one_over_n2));

	CallMAC(cudaFree(p_kappa_n));
	CallMAC(cudaFree(p_kappa_i));
	CallMAC(cudaFree(p_kappa_e));
	CallMAC(cudaFree(p_nu_i));
	CallMAC(cudaFree(p_nu_e));
	
	CallMAC(cudaFree(p_n_shards));
	CallMAC(cudaFree(p_n_shards_n));
	CallMAC(cudaFree(NT_addition_rates_d));
	CallMAC(cudaFree(NT_addition_tri_d));
	CallMAC(cudaFree(p_denom_i));
	CallMAC(cudaFree(p_denom_e));
	CallMAC(cudaFree(p_temp1));
	CallMAC(cudaFree(p_temp2));
	CallMAC(cudaFree(p_temp3));
	CallMAC(cudaFree(p_temp4));
	CallMAC(cudaFree(p_coeff_of_vez_upon_viz));
	CallMAC(cudaFree(p_longtemp));

	CallMAC(cudaFree(p_graphdata1));
	CallMAC(cudaFree(p_graphdata2));
	CallMAC(cudaFree(p_graphdata3));
	CallMAC(cudaFree(p_graphdata4));
	CallMAC(cudaFree(p_graphdata5));
	CallMAC(cudaFree(p_graphdata6));
	for (int i = 0; i < 9; i++)
		CallMAC(cudaFree(p_Tgraph[i]));
	for (int i = 0; i < 12; i++)
		CallMAC(cudaFree(p_accelgraph[i]));
	for (int i = 0; i < 12; i++)
		CallMAC(cudaFree(p_arelz_graph[i]));
	for (int i = 0; i < 20; i++)
		CallMAC(cudaFree(p_Ohmsgraph[i]));

	CallMAC(cudaFree(p_MAR_ion_temp_central));
	CallMAC(cudaFree(p_MAR_elec_temp_central));

	CallMAC(cudaFree(p_InvertedMatrix_n));
	CallMAC(cudaFree(p_InvertedMatrix_i));
	CallMAC(cudaFree(p_InvertedMatrix_e));
	CallMAC(cudaFree(p_MAR_ion2));
	CallMAC(cudaFree(p_MAR_elec2));
	CallMAC(cudaFree(NT_addition_rates_d_temp));
	CallMAC(cudaFree(p_epsilon_xy));
	CallMAC(cudaFree(p_epsilon_iz));
	CallMAC(cudaFree(p_epsilon_ez));
	CallMAC(cudaFree(p_vJacobi_i));
	CallMAC(cudaFree(p_vJacobi_e));
	CallMAC(cudaFree(p_vJacobi_n));
	CallMAC(cudaFree(p_d_eps_by_d_beta_i));
	CallMAC(cudaFree(p_d_epsxy_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_iz_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_ez_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_by_d_beta_e));
	CallMAC(cudaFree(p_d_epsxy_by_d_beta_e));
	CallMAC(cudaFree(p_d_eps_iz_by_d_beta_e));
	CallMAC(cudaFree(p_d_eps_ez_by_d_beta_e));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_i));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_i_times_i));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_i));

	CallMAC(cudaFree(p_AreaMinor_cc));
	CallMAC(cudaFree(p_sqrtfactor));

	CallMAC(cudaFree(p_epsilon3));
	CallMAC(cudaFree(zero_vec3));
	CallMAC(cudaFree(v3temp));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_x));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_y));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_z));

	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_J));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_R));
	CallMAC(cudaFree(p_sum_depsbydbeta_J_times_J));
	CallMAC(cudaFree(p_sum_depsbydbeta_R_times_R));
	CallMAC(cudaFree(p_sum_depsbydbeta_J_times_R));

	CallMAC(cudaFree(p_d_eps_by_dbetaJ_n_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaJ_i_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaJ_e_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_n_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_i_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_e_x4));

	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_J_x4));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_R_x4));
	CallMAC(cudaFree(p_sum_depsbydbeta_8x8));
	
	CallMAC(cudaFree(stored_Az_move));
	
	CallMAC(cudaFree(p_Tn));
	CallMAC(cudaFree(p_Ti));
	CallMAC(cudaFree(p_Te));
	CallMAC(cudaFree(p_Ap_n));
	CallMAC(cudaFree(p_Ap_i));
	CallMAC(cudaFree(p_Ap_e));

	CallMAC(cudaFree(p_regressors3));
	CallMAC(cudaFree(p_regressors4));
	CallMAC(cudaFree(p_tempvec4));

	CallMAC(cudaFree(p_tempvec3));
	CallMAC(cudaFree(p_SS));
	CallMAC(cudaFree(p_epsilon_x));
	CallMAC(cudaFree(p_epsilon_y));
	CallMAC(cudaFree(p_epsilon_z));
	CallMAC(cudaFree(p_stored_move3));
	CallMAC(cudaFree(p_stored_move4));
	CallMAC(cudaFree(p_d_eps_by_d_beta_x_));
	CallMAC(cudaFree(p_d_eps_by_d_beta_y_));
	CallMAC(cudaFree(p_d_eps_by_d_beta_z_));

	CallMAC(cudaFree(p_boolarray2));
	CallMAC(cudaFree(p_boolarray_block));
	CallMAC(cudaFree(p_sqrtD_inv_n));
	CallMAC(cudaFree(p_sqrtD_inv_i));
	CallMAC(cudaFree(p_sqrtD_inv_e));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_x8));
	CallMAC(cudaFree(p_eps_against_deps));
	CallMAC(cudaFree(p_sum_product_matrix));

	CallMAC(cudaFree(p_AAdot_start));
	CallMAC(cudaFree(p_AAdot_target));
	CallMAC(cudaFree(p_v_n_start));
	CallMAC(cudaFree(p_vie_start));
	CallMAC(cudaFree(p_v_n_target));
	CallMAC(cudaFree(p_vie_target));
	CallMAC(cudaFree(p_prev_move3));

	free(p_sum_eps_deps_by_dbeta_J_x4_host);
	free(p_sum_eps_deps_by_dbeta_R_x4_host);
	free(p_sum_depsbydbeta_8x8_host);

	free(p_tempvec2_host);
	free(p_tempvec2_host2);
	free(p_tempvec2_host3);
	free(p_eps_against_deps_host);
	free(p_sum_product_matrix_host);
	free(p_sum_product_matrix_host3);
	free(p_sum_eps_deps_by_dbeta_x8_host);
	free(p_boolhost);
	free(p_longtemphost);
	free(temp_array_host);
	free(p_temphost1);
	free(p_temphost2);
	free(p_GradTe_host);
	free(p_GradAz_host);
	free(p_B_host);
	free(p_MAR_ion_host);
	free(p_MAR_elec_host);
	free(p_MAR_neut_host);
	free(p_MAR_ion_compare);
	free(p_MAR_elec_compare);
	free(p_MAR_neut_compare);
	free(p_OhmsCoeffs_host);
	free(p_NTrates_host);
	free(p_graphdata1_host);
	free(p_graphdata2_host);
	free(p_graphdata3_host);
	free(p_graphdata4_host);
	free(p_graphdata5_host);
	free(p_graphdata6_host);

	for (int i = 0; i < 9; i++)
		free(p_Tgraph_host[i]);
	for (int i = 0; i < 12; i++)
		free(p_accelgraph_host[i]);
	for (int i = 0; i < 12; i++)
		free(p_arelz_graph_host[i]);
	for (int i = 0; i < 20; i++)
		free(p_Ohmsgraph_host[i]);

	GlobalSuppressSuccessVerbosity = false;
	printf("revoke done\n");

}

void Setup_residual_array()
{
	cuSyst * pX = &cuSyst1; // lazy

	// Find the pre-existing values of LapAz + 4piq/c n(viz-vez) 
	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
		pX->p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX->p_info, // populated position... not neigh_len apparently
		p_Az,
		pX->p_izTri_vert,
		pX->p_izNeigh_TriMinor,
		pX->p_szPBCtri_vert,
		pX->p_szPBC_triminor,
		p_LapAz
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaa2");

	kernelPopulateResiduals << <numTilesMinor, threadsPerTileMinor >> > (
		p_LapAz,
		pX->p_n_minor, pX->p_vie, // is this the n for which the relationship holds for verts? *************
		p_Residuals
		);
	Call(cudaThreadSynchronize(), "cudaTS PopulateResiduals");

}

void Zap_the_back()
{
	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst1.p_info,
			cuSyst1.p_n_major,
			cuSyst1.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst2.p_info,
			cuSyst2.p_n_major,
			cuSyst2.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst3.p_info,
			cuSyst3.p_n_major,
			cuSyst3.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

}

void Go_visit_the_other_file()
{

	f64 LapAz, viz, vez, n, coeffself, Az;
	int iRepeat;
	//f64 epsilon[NMINOR], p_regressor[NMINOR];

	memset(p_temphost2, 0, sizeof(f64)*NMINOR); // epsilon
	memset(p_temphost1, 0, sizeof(f64)*NMINOR); // regressor
//
//	kernelSetZero << <numTriTiles, threadsPerTileMinor >> > (
//		p_LapCoeffself
//		);
//	Call(cudaThreadSynchronize(), "cudaTS setzero");

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> > (
		cuSyst1.p_info,
		cuSyst1.p_izTri_vert,
		cuSyst1.p_izNeigh_TriMinor,
		cuSyst1.p_szPBCtri_vert,
		cuSyst1.p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs x");
	// Illegal memory access encountered?!
	// But this stuff should be dimensioned.
	// A bug in what it does, not my fault, by the looks.
	long iIteration = 0;
	bool bContinue;
	
	// 1. Calculate Lap Az and coeffself Lap Az; including at our few points.

	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
		cuSyst1.p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	nvals nvals1, nvals2, nvals3;
	v4 v1, v2, v3;
	f64 resid, resid1, resid2, resid3;
	f64 LapAz1, LapAz2, LapAz3;

	// Now define p_temphost4 as the constant part of eps and p_temphost5 as "resid"

	// 2. For each of our points bring Lap Az, Jz and coeffself to CPU
	for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
	{
		if (flaglist[i]) {
			cudaMemcpy(&viz, &(cuSyst1.p_vie[i].viz), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&vez, &(cuSyst1.p_vie[i].vez), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&n, &(cuSyst1.p_n_minor[i].n), sizeof(f64), cudaMemcpyDeviceToHost);

			LONG3 cornerindex;
			cudaMemcpy(&cornerindex, &(cuSyst1.p_tri_corner_index[i]), sizeof(LONG3), cudaMemcpyDeviceToHost);

			cudaMemcpy(&resid1, &(p_Residuals[cornerindex.i1]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&resid2, &(p_Residuals[cornerindex.i2]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&resid3, &(p_Residuals[cornerindex.i3]), sizeof(f64), cudaMemcpyDeviceToHost);
			// cudaMemcpy(&resid, &(p_Residuals[i]), sizeof(f64), cudaMemcpyDeviceToHost);

			resid = 0.33333333*(resid1 + resid2 + resid3);
			p_temphost5[i] = 0.01*fabs(resid); // thresh

			// What is average of LapAz + 4piq/c n(viz-vez) ?
			// eps = -LapAz - 4pi/cJ
			// so aim for LapAz = - 4pi/cJ - eps

			p_temphost4[i] = -FOUR_PI_Q_OVER_C_*n*(viz - vez) - resid;
			
		};
	};
	
	f64 beta;
	do {
		bContinue = false;
		
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			cuSyst1.p_info, // populated position... not neigh_len apparently
			p_Az,
			cuSyst1.p_izTri_vert,
			cuSyst1.p_izNeigh_TriMinor,
			cuSyst1.p_szPBCtri_vert,
			cuSyst1.p_szPBC_triminor, 
			p_LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaa2");
		
		// 2. For each of our points bring Lap Az, Jz and coeffself to CPU
		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{
			if (flaglist[i]) {
				cudaMemcpy(&LapAz, &(p_LapAz[i]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&coeffself, &(p_LapCoeffself[i]), sizeof(f64), cudaMemcpyDeviceToHost);

				// 3. For each of our points, adjust Az per Jacobi:
				//printf("%d Az %1.11E LapAz %1.11E 4pi/c Jz %1.11E coeffself %1.9E resid %1.9E resid123 %1.9E %1.9E %1.9E ", i, Az, LapAz, FOUR_PI_Q_OVER_C_*n*(viz - vez),
				//	coeffself, resid, resid1, resid2, resid3);

				p_temphost2[i] = p_temphost4[i] - LapAz; // epsilon
				if (fabs(p_temphost2[i]) > p_temphost5[i]) bContinue = true;
				p_temphost1[i] = (p_temphost4[i] - LapAz) / coeffself; // Jacobi move
			};
		};

		cudaMemcpy(p_temp1, p_temphost1, sizeof(f64), cudaMemcpyHostToDevice);
		cudaMemset(p_temp2, 0, sizeof(f64)*NMINOR);
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				cuSyst1.p_info, // populated position... not neigh_len apparently
				p_temp1,
				cuSyst1.p_izTri_vert,
				cuSyst1.p_izNeigh_TriMinor,
				cuSyst1.p_szPBCtri_vert,
				cuSyst1.p_szPBC_triminor,
				p_temp2
				);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaaa2");

		cudaMemcpy(p_temphost3, p_temp2, sizeof(f64), cudaMemcpyDeviceToHost);
		 
		f64 sum_depsbydbeta_sq = 0.0;
		f64 sum_eps_depsbydbeta = 0.0;
		f64 sum_eps_eps = 0.0;
		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{ 
			if (flaglist[i]) {
				f64 d_eps_by_d_beta = -p_temphost3[i];

				sum_depsbydbeta_sq += d_eps_by_d_beta*d_eps_by_d_beta;
				sum_eps_depsbydbeta += p_temphost2[i] * d_eps_by_d_beta;
				sum_eps_eps += p_temphost2[i] * p_temphost2[i];
			};
		};

		beta = -sum_eps_depsbydbeta / sum_depsbydbeta_sq;

		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{ 
			if (flaglist[i]) {
				cudaMemcpy(&Az, &(p_Az[i]), sizeof(f64), cudaMemcpyDeviceToHost);

				Az += beta*p_temphost1[i];

				cudaMemcpy(&(p_Az[i]), &Az, sizeof(f64), cudaMemcpyHostToDevice);
				// Note that we didn't keep track of which system's which, so we need to set it in all of them.				
			};
		};
		iIteration++;
		printf("iteration %d sum_eps_eps %1.10E beta %1.10E\n", iIteration, sum_eps_eps, beta);
	} while (bContinue);

	for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
	{
		if (flaglist[i]) {
			cudaMemcpy(&Az, &(p_Az[i]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&(cuSyst1.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			cudaMemcpy(&(cuSyst2.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			cudaMemcpy(&(cuSyst3.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			// Note that we didn't keep track of which system's which, so we need to set it in all of them.
		};
	};
	
	printf("\n\nRecalc Ampere: underrelaxation Jacobi iterations: %d\n\n\n", iIteration);
	
	Beep(750, 150);

	// Problem: by default, __constant__ variables have file scope. Need special
	// compiler settings to do relocatable device code.
}

void inline SubroutineComputeDbyDbetaNeutral(
	f64 const hsub, f64_vec2 * p_regrxy, f64 * p_regriz, f64_vec3 * p_v_n, cuSyst * pX_use,
	int i,
	int iUsexy, int iUse_z)
{
	cudaMemset(p_ROCMAR1, 0, sizeof(f64_vec3)*NMINOR);

	if (iUsexy) {
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_xy << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_vie,
			p_v_n, // not used
			p_regrxy + i*NMINOR, 

			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_ita_n, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_nu_n, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			pX_use->p_B,

			p_ROCMAR1,
			0,
			m_n_,
			1.0 / m_n_
			);
		Call(cudaThreadSynchronize(), "cudaTS dbydbeta regr1 xy_n");
	};
	if (iUse_z) {
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_z << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_vie,
			p_v_n,
			p_regriz + i*NMINOR, // regressor 1

			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_ita_n, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_nu_n, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			pX_use->p_B,

			p_ROCMAR1, 0, m_n_, 1.0 / m_n_
			);
		Call(cudaThreadSynchronize(), "cudaTS  dbydbeta regr1 neutz");
	};
		 
	kernelComputeNeutralDEpsByDBeta << <numTilesMinor, threadsPerTileMinor >> >
		(
			hsub,
			pX_use->p_info,
			p_regrxy + i*NMINOR,
			p_regriz + i*NMINOR,
			p_ROCMAR1,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_epsxy_by_d_beta_i + i*NMINOR,
			p_d_eps_iz_by_d_beta_i + i*NMINOR
			);
	Call(cudaThreadSynchronize(), "cudaTS  dbydbeta ");

}
  
void inline SubroutineComputeDbyDbeta(
	f64 const hsub, f64_vec2 * p_regrxy, f64 * p_regriz, f64 * p_regrez, v4 * p_vie, cuSyst * pX_use, int i,
	int iUsexy, int iUseiz, int iUseez)
{
	f64 tempf64;
	cudaMemset(p_ROCMAR1, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_ROCMAR2, 0, sizeof(f64_vec3)*NMINOR);

	if (iUsexy) {

		// Note that ita =0 will make it find 0 for forward region.
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_xy << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n, // not used
			p_regrxy + i*NMINOR, 			 
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_i, p_nu_i, 
			pX_use->p_B,
			p_ROCMAR1,
			1, m_ion_, 1.0 / m_ion_
			);
		Call(cudaThreadSynchronize(), "cudaTS dbydbeta regr1 xy_i");
		//cudaMemcpy(&tempf64, &(p_ROCMAR1[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\nROCMAR1[%d].y %1.14E \n\n", CHOSEN, tempf64);
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_xy << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n, // not used
			p_regrxy + i*NMINOR, // regressor 1
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_e, p_nu_e, 
			pX_use->p_B,
			p_ROCMAR2, 2, m_e_, 1.0 / m_e_
			);
		Call(cudaThreadSynchronize(), "cudaTS  dbydbeta regr1 xy_e");
		//cudaMemcpy(&tempf64, &(p_ROCMAR2[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\nROCMAR2[%d].y %1.14E \n\n", CHOSEN, tempf64);
	}; 

	if (iUseiz) {
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_z << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n, // not used
			p_regriz + i*NMINOR, // regressor 1
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_i, p_nu_i, 
			pX_use->p_B,
			p_ROCMAR1, 1, m_i_, 1.0 / m_i_
			);
		Call(cudaThreadSynchronize(), "cudaTS  dbydbeta regr1 ionz");
	//	cudaMemcpy(&tempf64, &(p_ROCMAR2[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	//	printf("\nROCMAR2[%d].z %1.14E \n\n", CHOSEN, tempf64);
	}
	
	if (iUseez) {
		kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_z << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n, // not used
			p_regrez + i*NMINOR,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,			 
			p_ita_e, p_nu_e, 
			pX_use->p_B,			 
			p_ROCMAR2, 2, m_e_, 1.0 / m_e_
			);
		Call(cudaThreadSynchronize(), "cudaTS  dbydbeta regr1 elecz");

	//	cudaMemcpy(&tempf64, &(p_ROCMAR2[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);

//		printf("\nROCMAR2[%d].z %1.14E \n\n", CHOSEN, tempf64);

	} 
	//printf("iUse %d %d %d ; now entering CCDEPDB \n", iUsexy, iUseiz, iUseez);

	kernelComputeCombinedDEpsByDBeta << <numTilesMinor, threadsPerTileMinor >> >
	(
		hsub,
		pX_use->p_info,
		p_regrxy + i*NMINOR,
		p_regriz + i*NMINOR,
		p_regrez + i*NMINOR,
		p_ROCMAR1,
		p_ROCMAR2,
		pX_use->p_n_minor,
		pX_use->p_AreaMinor,
		p_d_epsxy_by_d_beta_i + i*NMINOR,
		p_d_eps_iz_by_d_beta_i + i*NMINOR,
		p_d_eps_ez_by_d_beta_i + i*NMINOR
	);
	Call(cudaThreadSynchronize(), "cudaTS  dbydbeta ");

	// Tuesday:
	// Let's compare an empirical estimate now.
	// Pick a cell for comparison -- CHOSEN will do.
		
	// Straight to comparing the MAR that we get ... and do this for ez.

	/*
	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_vie,
		pX_use->p_v_n, 
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_use->p_B,
		p_MAR_elec2, // accumulates
		NT_addition_rates_d_temp, NT_addition_tri_d,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

	cudaMemcpy(&tempf64, &(p_MAR_elec2[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("existing MAR[%d].z %1.14E \n", CHOSEN, tempf64);
	f64 MAR1 = tempf64;

	AddLittleBitORegressors << <numTilesMinor, threadsPerTileMinor >> >
		(1.0e-10, p_vie, p_regrxy + i*NMINOR,
		p_regriz + i*NMINOR,
		p_regrez + i*NMINOR);
	Call(cudaThreadSynchronize(), "cudaTS AddLittleBit");

	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_vie,
		pX_use->p_v_n,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		pX_use->p_B,
		p_MAR_elec2, // accumulates
		NT_addition_rates_d_temp, NT_addition_tri_d,
		2, m_e_, 1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");
	cudaMemcpy(&tempf64, &(p_MAR_elec2[CHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("updated MAR[%d].z %1.14E \n\n", CHOSEN, tempf64);
	f64 MAR2 = tempf64;
	
	printf("estimate of deriv = %1.14E  ;  ", (MAR2 - MAR1) * 1.0e10);
	f64 temp___;
	cudaMemcpy(&tempf64, &(p_regrez[CHOSEN + i*NMINOR]), sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(&temp___, &(p_vie[CHOSEN].vez), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("regr ez[%d] = %1.10E  vez %1.10E \n\n", CHOSEN, tempf64, temp___);

	AddLittleBitORegressors << <numTilesMinor, threadsPerTileMinor >> >
		(-1.0e-10, p_vie, p_regrxy + i*NMINOR,
			p_regriz + i*NMINOR,
			p_regrez + i*NMINOR);
	Call(cudaThreadSynchronize(), "cudaTS AddLittleBit");
	*/

}



void RunBackwardR8LSForViscosity_Geometric(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use)
// BE SURE ABOUT PARAMETER ORDER -- CHECK IT CHECK IT
{
	// ***************************************************
	// Requires averaging of n,T to triangles first. & ita
	// ***************************************************

#define ITERGRAPH 400

	static bool bHistoryVisc = false;
	FILE * dbgfile;

	f64_vec2 tempvec2;
	f64 tempf64_2;
	f64 beta[REGRESSORS];
	f64 beta_e, beta_i, beta_n;
	long iTile;
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	long iMinor;
	f64 L2eps;
	int i, iMoveType;
	f64 TSS_xy, TSS_iz, TSS_ez, RSS_xy, RSS_iz, RSS_ez;
	f64 Rsquared_xy, Rsquared_iz, Rsquared_ez;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	// Do not do it:
	//cudaMemcpy(p_vie, p_vie_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	
	cudaMemset(zero_vec4, 0, sizeof(v4)*NMINOR);
	int iIteration = 0;
	bool bContinue = true;

	bool bDebug = false;

	structural info;
//	cudaMemcpy(&info, &(pX_use->p_info[CHOSEN]), sizeof(structural), cudaMemcpyDeviceToHost);
//	printf("\npX_use->p_info[%d].pos %1.10E %1.10E modulus %1.10E \n\n",
//		CHOSEN, info.pos.x, info.pos.y, info.pos.modulus());

	// Aim: split out z vs xy.
	// Bad aim as it turned out.

	//=============================================================================

	cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
	cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
		
	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_vie,
		// For neutral it needs a different pointer.
		pX_use->p_v_n, // not used
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_ita_i, p_nu_i,
		pX_use->p_B,
		p_MAR_ion2, // accumulates
		NT_addition_rates_d_temp, NT_addition_tri_d2,
		1,		m_ion_,		1.0 / m_ion_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

	kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_vie,
		// For neutral it needs a different pointer.
		pX_use->p_v_n, // not used
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,

		p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up

		pX_use->p_B,
		p_MAR_elec2, // accumulates
		NT_addition_rates_d_2, NT_addition_tri_d2,
		2,		m_e_,		1.0 / m_e_);
	Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");
	
	// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
	kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
		// eps = v - (v_k +- h [viscous effect])
		// x = -eps/coeffself
		hsub,
		pX_use->p_info,
		p_vie,
		p_vie_k,
		p_MAR_ion2, p_MAR_elec2,

		pX_use->p_n_minor,
		pX_use->p_AreaMinor,

		p_epsilon_xy,
		p_epsilon_iz,
		p_epsilon_ez,
		p_bFailed,
		p_Selectflag
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

	// This stayed the same.

	bContinue = false;
	cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
	if (i < numTilesMinor) bContinue = true;

	// Collect L2:
	RSS_xy = 0.0;
	RSS_iz = 0.0;
	RSS_ez = 0.0;

	kernelAccumulateSumOfSquares2vec << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_xy, p_SS);
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_xy += p_SS_host[iTile];
	f64 L2eps_xy = sqrt(RSS_xy / (real)NMINOR);

	SetConsoleTextAttribute(hConsole, 14);

	printf("L2eps xy %1.8E ", L2eps_xy);

	if (L2eps_xy == 0.0) goto labelBudgerigar2;

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_iz, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_iz += p_SS_host[iTile];
	f64 L2eps_iz = sqrt(RSS_iz / (real)NMINOR);
	printf("iz %1.8E ", L2eps_iz);
	 
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> > (
		p_epsilon_ez, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS_ez += p_SS_host[iTile];
	f64 L2eps_ez = sqrt(RSS_ez / (real)NMINOR);
	printf("ez %1.8E \n", L2eps_ez);
	 
	TSS_ez = RSS_ez;
	TSS_iz = RSS_iz;
	TSS_xy = RSS_xy;
	SetConsoleTextAttribute(hConsole, 15);
	// This stayed the same.
	printf("bContinue: %d \n", bContinue ? 1 : 0);

	//// Debug: find maximum epsilons.
	//f64 maxio = 0.0;
	//int iMaxx = -1;
	//cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_tempvec2_host[iMinor].dot(p_tempvec2_host[iMinor]) > maxio) {
	//		iMaxx = iMinor;
	//		maxio = p_tempvec2_host[iMinor].dot(p_tempvec2_host[iMinor]);
	//	};
	//printf("eps_xy : maxio %1.8E at %d\n", sqrt(maxio), iMaxx);
	//
	//maxio = 0.0;
	//int iMaxiz = -1;
	//cudaMemcpy(p_temphost6, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_temphost6[iMinor] * p_temphost6[iMinor] > maxio) {
	//		iMaxiz = iMinor;
	//		maxio = p_temphost6[iMinor] * p_temphost6[iMinor];
	//	};
	//printf("eps_iz : maxio %1.8E at %d\n", sqrt(maxio), iMaxiz);
	//
	//maxio = 0.0;
	//int iMaxez = -1;
	//cudaMemcpy(p_temphost6, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//	if (p_temphost6[iMinor] * p_temphost6[iMinor] > maxio) {
	//		iMaxez = iMinor;
	//		maxio = p_temphost6[iMinor] * p_temphost6[iMinor];
	//	};
	//printf("eps_ez : maxio %1.8E at %d\n", sqrt(maxio), iMaxez);
	 
	f64 oldRSS = 0.0;
	f64 keep, tempf, keep2;
	// set epsilon vectors again at end of loop.		
	while (bContinue) {

		cudaMemcpy(&info, &(pX_use->p_info[CHOSEN]), sizeof(structural), cudaMemcpyDeviceToHost);
		printf("\npX_use->p_info[%d].pos %1.10E %1.10E modulus %1.10E \n\n",
			CHOSEN, info.pos.x, info.pos.y, info.pos.modulus());

		// . Get Jacobi inverse:
		// ======================
		CalculateCoeffself << <numTriTiles, threadsPerTileMinor >> >(
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_i,   // nT / nu ready to look up
			p_nu_i,   // nT / nu ready to look up
			pX_use->p_B,
			p__matrix_xy_i, // matrix ... 
			p__coeffself_iz,
			p__xzyzzxzy_i,
			1,			m_ion_,			1.0 / m_ion_
			);
		Call(cudaThreadSynchronize(), "cudaTS CalculateCoeffself ion");

		CalculateCoeffself << <numTriTiles, threadsPerTileMinor >> >(
			pX_use->p_info,
			p_vie,
			pX_use->p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_ita_e,   // nT / nu ready to look up
			p_nu_e,   // nT / nu ready to look up
			pX_use->p_B,
			p__matrix_xy_e, // matrix ... 
			p__coeffself_ez,
			p__xzyzzxzy_e,
			2,			m_e_,			1.0 / m_e_
			);
		Call(cudaThreadSynchronize(), "cudaTS CalculateCoeffself e");

		// Test what we got:
//		f64 tempf64;
//		cudaMemcpy(&tempf64, &(p__matrix_xy_i[VERTCHOSEN + BEGINNING_OF_CENTRAL].yy), sizeof(f64), cudaMemcpyDeviceToHost);
//		printf("\np__matrix_xy_i[%d + %d].yy %1.10E \n\n", VERTCHOSEN, BEGINNING_OF_CENTRAL, tempf64);

// Zero out to begin with. We will use the ith NMINOR array combination for each regr.
		cudaMemset(p_regressors2, 0, sizeof(f64_vec2)*NMINOR*REGRESSORS);
		cudaMemset(p_regressors_iz, 0, sizeof(f64)*NMINOR*REGRESSORS);
		cudaMemset(p_regressors_ez, 0, sizeof(f64)*NMINOR*REGRESSORS);

		// This function also populates 1st regressor...
		// but we're missing any bool for selected, which is inefficient
		
		// We might like to be able to run this with a selection only for ring. That's the best way to create 4DJ.
		// Harsh necessity.

		// Let's bring that in afterwards. Independent dims will have to do for now

		kernelCreateDByDBetaCoeffmatrix << <numTilesMinor, threadsPerTileMinor >> >(
			hsub,
			pX_use->p_info,
			p__matrix_xy_i,			p__matrix_xy_e,
			p__coeffself_iz,		p__coeffself_ez,  // d MAR / d v
			p__xzyzzxzy_i,			p__xzyzzxzy_e,
			pX_use->p_n_minor,		pX_use->p_AreaMinor,
			p_epsilon_xy,		p_epsilon_iz,		p_epsilon_ez,
			p_regressors2,		p_regressors_iz,		p_regressors_ez,
			p__invmatrix,		p__invcoeffselfviz,		p__invcoeffselfvez,
			p__invcoeffself_x,	p__invcoeffself_y // new variables
			);
		Call(cudaThreadSynchronize(), "cudaTS CreateDByDBetaCoeffmatrix");
		
		// 2. Create set of 7 or 8 regressors, starting with epsilon3 normalized,
		// and deps/dbeta for each one.
		// The 8th is usually either for the initial seed regressor (prev move) or comes from previous iteration

		cudaMemset(p_d_epsxy_by_d_beta_i, 0, sizeof(f64_vec2)*NMINOR*REGRESSORS);
		cudaMemset(p_d_eps_iz_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS);
		cudaMemset(p_d_eps_ez_by_d_beta_i, 0, sizeof(f64)*NMINOR*REGRESSORS);

		//
		//kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
		//	(p_regressors2,
		//		p_epsilon_xy,
		//		p__invmatrix);
		//Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");
		//kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
		//	(p_regressors_iz + NMINOR,
		//		p_epsilon_iz,
		//		p__invcoeffselfviz);
		//Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");
		//
		//kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
		//	(p_regressors_ez + NMINOR * 2,
		//		p_epsilon_ez,
		//		p__invcoeffselfvez);
		//Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
		//SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
		//	1, 0, 0); // XY_ONLY
		//SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
		//	0, 1, 0);// IZ_ONLY);
		//SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
		//	0, 0, 1);// EZ_ONLY);		
		// ---------------------------------------------------------------------------------------
		// Type 3 move:
		// Do everything:

		iMoveType = 3;
		if (L2eps_ez + L2eps_iz > 50.0*L2eps_xy) { iMoveType = 1; }
		else { if (L2eps_xy > 50.0*(L2eps_ez + L2eps_iz)) { iMoveType = 2; }; };
		printf("iMoveType %d : L2pes_ez + L2eps_iz %1.8E * 50 = %1.8E L2eps_xy %1.8E\n", iMoveType,
			L2eps_ez+L2eps_iz, 50*(L2eps_ez + L2eps_iz), L2eps_xy);

		if ((iIteration >= 32) && (iIteration % 4 == 0))
			iMoveType = 4;

		// debug:
		if (iIteration > ITERGRAPH) iMoveType = 4;
		
		
		if (0) {
			printf("Do you want to do a smash move : y / n:");
			char o;
			do {
				o = getch();
			} while ((o != 'y') && (o != 'n'));
			if (o == 'y') iMoveType = 4;
			printf("%c\n", o);
		}
			//&& (iIteration % 4 == 0)) 
		
		int Ipiv512[4 * EQNS_TOTAL];
		f64 LU[4 * 4 * EQNS_TOTAL];
		f64 result[4 * EQNS_TOTAL];

		f64 prediction;
		long iMinor, iMax = -1;
		f64 epssq, maxepssq;
		long izNeighMinor[MAXNEIGH];
		int ilengthprev;
		long izArrayPrev[2048];
		long izArray[2048];
		int whichRing, ilength, iEqns;
		long j, jMinor;
		short neigh_len;
		FILE * fp;
		// debug:
		long eqnlist[2048];

		cusolverDnHandle_t cusolverH = NULL;
		cudaStream_t stream = NULL;

		cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
		cudaError_t cudaStat1 = cudaSuccess;
		cudaError_t cudaStat2 = cudaSuccess;
		cudaError_t cudaStat3 = cudaSuccess;
		cudaError_t cudaStat4 = cudaSuccess;
		const int m = 4 * EQNS_TOTAL;
		const int lda = m;
		const int ldb = m;
		//			double A[lda*m] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0 };
		//			double B[m] = { 1.0, 2.0, 3.0 };
		//			double X[m]; /* X = A\B */
		//			double LU[lda*m]; /* L and U */
		//			int Ipiv[4*EQNS_TOTAL];      /* host copy of pivoting sequence */
		//			int info = 0;     /* host copy of error info */

		//double *d_A = NULL; /* device copy of A */
		//double *d_B = NULL; /* device copy of B */
		//int *d_Ipiv = NULL; /* pivoting sequence */
		//int *d_info = NULL; /* error info */
		int lwork = 0;     /* size of workspace */
		//double *d_work = NULL; /* device workspace for getrf */

		const int pivot_on = 1;

		cudaMemcpy(p_inthost, p_Selectflag, sizeof(int)*NMINOR, cudaMemcpyDeviceToHost);

		bool bUseUnselected;
		{
		Matrix_real matLU;
		switch (iMoveType)
		{
		case 4:
			// Smash move.
			printf("case 4\n");

			matLU.Invoke(4 * EQNS_TOTAL);

			//printf("Creating equation region.\n");
			// 1. Identify highest residual point
			cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

			maxepssq = 0.0;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				epssq = p_tempvec2_host[iMinor].dot(p_tempvec2_host[iMinor])
					+ p_temphost1[iMinor] * p_temphost1[iMinor]
					+ p_temphost2[iMinor] * p_temphost2[iMinor];
				if (epssq > maxepssq)
				{
					maxepssq = epssq;
					iMax = iMinor;
				};
			};

			long * longaddress;
			Call(cudaGetSymbolAddress((void **)(&longaddress), lChosen),
				"cudaGetSymbolAddress((void **)(&longaddress), lChosen)");
			Call(cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice),
				"cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice)");

			// 2. Spread out around it, add points to equation list (CPU?) 
			// (within domain only)
			printf("Add points to eqn list. iMax: %d \n", iMax);

			CallMAC(cudaMemcpy(cuSyst_host.p_info, pX_use->p_info, sizeof(structural)*NMINOR, cudaMemcpyDeviceToHost));

			memset(iRing, 0, sizeof(int)*NMINOR);
			memset(bSelected, 0, sizeof(bool)*NMINOR);

			ilengthprev = 1;
			izArrayPrev[0] = iMax;
			iRing[iMax] = 1; // ring 1 is innermost ring

			bSelected[iMax] = true;
			p_equation_index_host[iMax] = 0; // SHORT ARRAY
			iEqns = 1;
			eqnlist[0] = iMax;

			whichRing = 2;
			bUseUnselected = false;
			do {
				ilength = 0;

			label1:

				// each point in previous ring of additions:
				for (i = 0; ((i < ilengthprev) && (iEqns < EQNS_TOTAL) && (ilength < 2048)); i++)
				{
					//printf("i %d ilengthprev %d iEqns %d izArrayPrev[i] %d\n", i, ilengthprev, iEqns, izArrayPrev[i]);

					if (izArrayPrev[i] >= BEGINNING_OF_CENTRAL) {
						// get izNeighMinor
						long iVertex = izArrayPrev[i] - BEGINNING_OF_CENTRAL;
						cudaMemcpy(izNeighMinor, &(pX_use->p_izTri_vert[iVertex * MAXNEIGH]),
							sizeof(long) * MAXNEIGH, cudaMemcpyDeviceToHost);
						neigh_len = cuSyst_host.p_info[izArrayPrev[i]].neigh_len;
					}
					else {
						// get izTri as izNeighMinor
						cudaMemcpy(izNeighMinor, &(pX_use->p_izNeigh_TriMinor[izArrayPrev[i] * 6]),
							sizeof(long) * 6, cudaMemcpyDeviceToHost);
						neigh_len = 6;
					};

					for (j = 0; ((j < neigh_len) && (iEqns < EQNS_TOTAL) && (ilength < 2048)); j++)
					{
						jMinor = izNeighMinor[j];
						// For each neighbour:
						// . Is it within domain? && is it not already selected ?
						// Not every CROSSING_INS should be included -- correct?
						// So we need to do position test as well as check what sort
						// of flag.
						if (
							((cuSyst_host.p_info[jMinor].flag == DOMAIN_TRIANGLE)
								||
								(cuSyst_host.p_info[jMinor].flag == DOMAIN_VERTEX)
								||
								((cuSyst_host.p_info[jMinor].flag == CROSSING_INS) &&
									TestDomainPosHost(cuSyst_host.p_info[jMinor].pos))
								)
							&& (iRing[jMinor] == 0)
							&& ((p_inthost[jMinor] != 0) || bUseUnselected) // exclude unselected equations
							)
						{
							// . Add it to this ring
							// . Tell it that it is in this ring
							iRing[jMinor] = whichRing;
							izArray[ilength] = jMinor;
							ilength++;
							if (ilength == 2048) {
								printf("ilength == 2048\n");
							}
							//							iEqnArray[iEqns] = jMinor; // because we need a list of the equations?
								//						printf("iEqnArray[%d] %d \n", iEqns, iEqnArray[iEqns]);
														// Not sure we need such a list.
							bSelected[jMinor] = true;
							p_equation_index_host[jMinor] = iEqns;

							eqnlist[iEqns] = jMinor;

							iEqns++;
					//		printf("iRing %d ilength %d index %d : iEqns %d selectflag %d\n", 
					//					whichRing, ilength, jMinor, iEqns, p_inthost[jMinor]);
						};
					}; // next j in neighs
				}; // next i in ring prev
			//	printf("ring %d ilength %d iEqns %d\n", whichRing, ilength, iEqns);
				if ((ilength == 0) && (iEqns < EQNS_TOTAL)) {
					// go again, we ran out of selected points?
					bUseUnselected = true;
					goto label1;
				}
				++whichRing;
				ilengthprev = ilength;
				memcpy(izArrayPrev, izArray, sizeof(long)*ilength);
			} while (iEqns < EQNS_TOTAL);
			// let's say we make 1 extra ring outside those, in case we want it.

			printf("iEqns = %d/n", iEqns);
			// debug:
			/*
			cudaMemcpy(p_temphost1, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);

			for (i = 0; i < EQNS_TOTAL; i++)
			{
				printf("Eqn %d : %d : epsilon_iz %1.9E ez %1.9E x %1.9E y %1.9E\n",
					i, eqnlist[i], p_temphost1[eqnlist[i]], p_temphost2[eqnlist[i]], p_tempvec2_host[eqnlist[i]].x, p_tempvec2_host[eqnlist[i]].y);
			}
			getch();
			*/
			printf("Creating equations.\n");
			CallMAC(cudaMemcpy(p_selectflag, bSelected, sizeof(bool)*NMINOR, cudaMemcpyHostToDevice));
			CallMAC(cudaMemcpy(p_equation_index, p_equation_index_host, sizeof(short)*NMINOR, cudaMemcpyHostToDevice));

			//================================================
			// 3. Create equations:
			// 3a. Get ion mom flux coefficients
			// 3b. Get elec mom flux coefficients
			// 3c. Combine to create equations for solving.

			CallMAC(cudaMemset(ionmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
			CallMAC(cudaMemset(elecmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
				<< <numTriTiles, threadsPerTileMinor >> > (
					pX_use->p_info,
					p_vie,
					// For neutral it needs a different pointer.
					pX_use->p_v_n, // not used				
					// structure of this:
					//  
					ionmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.				
					p_selectflag, // whether it's in the smoosh to smash
					p_equation_index, // each one assigned an equation index				
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_izNeigh_TriMinor,
					pX_use->p_szPBC_triminor,
					p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
					p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
					pX_use->p_B,
					1, m_i_, 1.0 / m_i_
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
				<< <numTriTiles, threadsPerTileMinor >> > (
					pX_use->p_info,
					p_vie,
					pX_use->p_v_n, // not used
					elecmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.
					p_selectflag, // whether it's in the smoosh to smash
					p_equation_index, // each one assigned an equation index
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_izNeigh_TriMinor,
					pX_use->p_szPBC_triminor,
					p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
					p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
					pX_use->p_B,
					2, m_e_, 1.0 / m_e_
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

			CallMAC(cudaMemset(p_eqns, 0, sizeof(f64) * 4 * 4 * EQNS_TOTAL*EQNS_TOTAL));
			CallMAC(cudaMemset(p_RHS, 0, sizeof(f64) * 4 * EQNS_TOTAL));
			kernelCreateEquations << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_use->p_info,
				ionmomflux_eqns,
				elecmomflux_eqns,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_izTri_vert,
				p_equation_index, // each one assigned an equation index
				p_selectflag,
				p_eqns, // 4 x EQNS_TOTAL x 4 x EQNS_TOTAL
				p_epsilon_xy,
				p_epsilon_iz,
				p_epsilon_ez,
				p_RHS // --- just minus epsilon in an unbundled list.
				// maybe need _ on variables
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreateEquations");
	
cudaMemcpy(p_eqns_host, p_eqns, 4 * EQNS_TOTAL * 4 * EQNS_TOTAL * sizeof(f64),
	cudaMemcpyDeviceToHost);
cudaMemcpy(p_temphost5, p_RHS, 4*EQNS_TOTAL*sizeof(f64), cudaMemcpyDeviceToHost);

fp = fopen("eqnmatrix1.txt", "w");
for (i = 0; i < 4 * EQNS_TOTAL; i++) {
	for (j = 0; j < 4 * EQNS_TOTAL; j++)
	{
		fprintf(fp, "%1.14E ", p_eqns_host[i * 4 * EQNS_TOTAL + j]);
	};
	fprintf(fp, "  |  %1.14E \n", p_temphost5[i]);
}
fclose(fp);
cudaMemcpy(p_eqns_host, ionmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
	cudaMemcpyDeviceToHost);
fp = fopen("ionmatrix1.txt", "w");
for (i = 0; i < 3 * EQNS_TOTAL; i++) {
	for (j = 0; j < 3 * EQNS_TOTAL; j++)
	{
		fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
	};
	fprintf(fp, "\n");
}
fclose(fp);
cudaMemcpy(p_eqns_host, elecmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
	cudaMemcpyDeviceToHost);
fp = fopen("elecmatrix1.txt", "w");
for (i = 0; i < 3 * EQNS_TOTAL; i++) {
	for (j = 0; j < 3 * EQNS_TOTAL; j++)
	{
		fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
	};
	fprintf(fp, "\n");
}
fclose(fp);
printf("outputted elecmatrix.txt");

cudaMemoryTest();

			printf("send to host\n");

			cudaMemcpy(p_eqns_host, p_eqns, 4 * EQNS_TOTAL * 4 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_RHS, 4 * EQNS_TOTAL * sizeof(f64), cudaMemcpyDeviceToHost);

			printf("Solving equations.\n");
			for (int i = 0; i < 4 * EQNS_TOTAL; i++)
				memcpy(&(matLU.LU[i][0]), &(p_eqns_host[i * 4 * EQNS_TOTAL]), sizeof(f64) * 4 * EQNS_TOTAL);
			matLU.LUdecomp();
			matLU.LUSolve(p_temphost5, result); // solving Ax = b and it's (b, x).
			printf("Done LU. Dimension was %d.\n", 4 * EQNS_TOTAL);

			cudaMemcpy(p_RHS, result, sizeof(f64) * 4 * EQNS_TOTAL, cudaMemcpyHostToDevice);
		
			fp = fopen("soln1.txt", "w");
			for (i = 0; i < 4 * EQNS_TOTAL; i++)
			{
					fprintf(fp, "indx %d Soln %1.14E \n", matLU.indx[i], result[i]);
			}
			fclose(fp);
			printf("done soln.txt");
			/*

			// 4. Solve equations : cuSolver parallel solve...
			//====================================================
			
			// Spit out matrix eqn for purpose of query on forums:
			cudaMemcpy(p_eqns_host, p_eqns, 4 * EQNS_TOTAL * 4 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_RHS, 4 * EQNS_TOTAL * sizeof(f64), cudaMemcpyDeviceToHost);
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < m; j++)	printf("%1.3E  ", p_eqns_host[i + j*m]);
				printf(" | %1.4E\n", p_temphost5[i]);
			};
			printf("\n");

			if (pivot_on) {
				printf("pivot is on : compute P*A = L*U \n");
			} else {
				printf("pivot is off: compute A = L*U (not numerically stable)\n");
			};

			// step 1: create cusolver handle, bind a stream
			status = cusolverDnCreate(&cusolverH);
			if (CUSOLVER_STATUS_SUCCESS != status) { printf("cusolverDnCreate(&cusolverH) failed.\n"); getch(); }
			cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			if (cudaSuccess != cudaStat1) { printf("cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); failed.\n"); getch(); }
			status = cusolverDnSetStream(cusolverH, stream);
			if (CUSOLVER_STATUS_SUCCESS != status) {printf("cusolverDnSetStream(cusolverH, stream) failed.\n");	getch();}

			// step 2: 'copy A to device'
			// p_eqns is allocated on device, and so is p_RHS, d_Ipiv512, d_info
			// step 3: query working space of getrf
			//	status = cusolverDnDgetrf_bufferSize(
			//		cusolverH,	m,	m,	d_A,lda,&lwork);
			status = cusolverDnDgetrf_bufferSize(
				cusolverH,
				4 * EQNS_TOTAL,	4 * EQNS_TOTAL,
				p_eqns,
				4 * EQNS_TOTAL,	&lwork);
			if (CUSOLVER_STATUS_SUCCESS != status) {
				printf("cusolverDnDgetrf_bufferSize failed.\n"); getch();
			} else {
				printf("Success:cusolverDnDgetrf_bufferSize\n");
			};
			CallMAC(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

			// step 4: LU factorization
			if (pivot_on) {
				status = cusolverDnDgetrf(
					cusolverH,
					4 * EQNS_TOTAL,	4 * EQNS_TOTAL,
					p_eqns,	lda, d_work, d_Ipiv512,	d_info);
//				status = cusolverDnDgetrf(
//					cusolverH,	m,	m,	p_eqns, // 4*N*4*N
//					lda,d_work,	d_Ipiv512, // 128*4 = 512
//					d_info);
			} else {
				status = cusolverDnDgetrf(
					cusolverH,
					4*EQNS_TOTAL,	4 * EQNS_TOTAL,
					p_eqns,	lda,	d_work,	NULL,	d_info);
			}
			if (CUSOLVER_STATUS_SUCCESS != status) {
				printf("cusolverDnDgetrf failed.\n"); getch();
			} else {
				printf("cusolverDnDgetrf : CUSOLVER_STATUS_SUCCESS == status\n");
			}
			CallMAC(cudaDeviceSynchronize());

			int info;
			if (pivot_on) {
				CallMAC(cudaMemcpy(Ipiv512, d_Ipiv512, sizeof(int)*m, cudaMemcpyDeviceToHost));
			}
		//	cudaStat2 = cudaMemcpy(LU, p_eqns, sizeof(double)*4*4*EQNS_TOTAL, cudaMemcpyDeviceToHost);
			CallMAC(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
			if (0 > info) {
				printf("%d-th parameter is wrong \n", -info);
				while (1) getch();
			}
			if ((pivot_on)) {
				printf("pivoting sequence: (m = %d; lda = %d)\n", m, lda);
				for (int j = 0; j < 4*EQNS_TOTAL; j++) {
					printf("Ipiv[%d] = %d\n", j, Ipiv512[j]);
					//if (j % 4 == 0) printf("");
				};
			}

			while (1) getch();

			// step 5: solve A*X = B
			if (pivot_on) {
				status = cusolverDnDgetrs(
					cusolverH,	CUBLAS_OP_N,
					4*EQNS_TOTAL,
					1,
					p_eqns,	4 * EQNS_TOTAL,
					d_Ipiv512,	p_RHS,
					4*EQNS_TOTAL, d_info);
				//status = cusolverDnDgetrs(
				//	cusolverH,	CUBLAS_OP_N,	m,	1,
				//	d_A,	lda,	d_Ipiv,	d_B,	ldb,	d_info);
			} else {
				status = cusolverDnDgetrs(
					cusolverH, CUBLAS_OP_N,
					4*EQNS_TOTAL,
					1,
					p_eqns, 4*EQNS_TOTAL,
					NULL, p_RHS, // d_B
					4*EQNS_TOTAL, d_info);
			}
			CallMAC(cudaDeviceSynchronize());
			if (CUSOLVER_STATUS_SUCCESS != status) {
				printf("cusolverDnDgetrs failed.\n");
				getch();
			};

			CallMAC(cudaMemcpy(result, p_RHS, sizeof(double)*m, cudaMemcpyDeviceToHost));

			if (cusolverH) cusolverDnDestroy(cusolverH);
			if (stream) cudaStreamDestroy(stream);


			fp = fopen("soln.txt", "w");
			for (i = 0; i < 4 * EQNS_TOTAL; i++)
			{
				if (pivot_on) {
					fprintf(fp, "iPiv %d Soln %1.14E Soln[ipiv] %1.14E \n",
						Ipiv512[i] - 1, result[i], result[Ipiv512[i] - 1]);
				}
				else {
					fprintf(fp, " %d Soln %1.14E \n",
						i, result[i]);
				}
			}
			fclose(fp);

			*/
			// Test:
/*
			cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
			cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_vie,
				// For neutral it needs a different pointer.
				pX_use->p_v_n, // not used
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
				p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
				pX_use->p_B,
				p_MAR_ion2, // accumulates
				NT_addition_rates_d_temp, NT_addition_tri_d,
				1, m_ion_, 1.0 / m_ion_);
			Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");

			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_vie,
				// For neutral it needs a different pointer.
				pX_use->p_v_n, // not used
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
				p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up

				pX_use->p_B,
				p_MAR_elec2, // accumulates
				NT_addition_rates_d_temp, NT_addition_tri_d,
				2, m_e_, 1.0 / m_e_);
			Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

			cudaMemcpy(&tempf, &(p_MAR_elec2[iMax].y), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("p_MAR_elec2[%d].y = %1.10E\n",
				iMax, tempf);

			// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
			kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				// x = -eps/coeffself
				hsub,
				pX_use->p_info,
				p_vie,
				p_vie_k,
				p_MAR_ion2, p_MAR_elec2,

				pX_use->p_n_minor,
				pX_use->p_AreaMinor,

				p_epsilon_xy,
				p_epsilon_iz,
				p_epsilon_ez,
				p_bFailed
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

			*/
			// 2 further things to check, since it does not work.
			// A. Effect of own vy on py 0?
			// B. Effect of element 1 vy or vz on py 0.
			// Done element 3. Now it's element 1
			//printf("Deleting all elements except element 7\n");
			//keep = result[7];
			//cudaMemset(p_RHS, 0, sizeof(f64) * 4 * EQNS_TOTAL);
			//cudaMemcpy(p_RHS + 7, &keep, sizeof(f64), cudaMemcpyHostToDevice);						
			//cudaMemcpy(&tempf, &(p_eqns[4*EQNS_TOTAL*1 + 1]), sizeof(f64), cudaMemcpyDeviceToHost);
			/*
			// think p_eqns has been malformed.
			tempf = p_eqns_host[4 * EQNS_TOTAL + 7];
			prediction = 0.0;
			for (i = 0; i < 4 * EQNS_TOTAL; i++) // which variable to change...
			{
				printf("Predicted change in epsilon[1] given %d : += %1.12E * chg %1.12E = %1.12E\n",
					i, p_eqns_host[4 * EQNS_TOTAL + i], result[i], p_eqns_host[4 * EQNS_TOTAL + i] * result[i]);
				prediction += p_eqns_host[4 * EQNS_TOTAL + i] * result[i];
			};
			printf("Total prediction: %1.14E \n\n", prediction);

			// Maybe problem is that we are supposed to manually un-pivot.

			*/

			//printf("Predicted change in elec y flux = %1.10E * %1.10E = %1.10E\n&&&&&&&&&&&\n",
			//	keep, p_eqns_host[3 * EQNS_TOTAL * 1 + 1], keep* p_eqns_host[3 * EQNS_TOTAL * 1 + 1]);
			/*
			kernelAddSolution << <numTilesMinor, threadsPerTileMinor >> > (
				p_vie, p_selectflag, p_equation_index, p_RHS
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelAddSolution");
			*/
			printf("Regressors...\n");

			// 5. Regression: bleed coefficient applies to further out points..
			// Regressor 0 = solution * dummy within central region
			// Regressor 1,2 = solution * dummy at boundary of region
			// Regressor 3+ = Jacobi (change compensation) outside region
			// could work outwards in rings for the other regressors.
			// If that does not work, try just setting regressor 0 == 1.
			// ie force apply the change on interior, leave those points fixed
			// and apply regressors outside the solution domain before doing anything else
			// or set the others zero.

			//iRing[jMinor] = whichRing; // Don't see why this belongs here. If anything it's whichRing we want to set.

			// Assign different coefficient to last 2 rings.
			// Remember if whichRing is now at 11 then whichRing == 9 or 10 is where we put different coefficient.

			CallMAC(cudaMemcpy(p_iRing, iRing, sizeof(int)*NMINOR, cudaMemcpyHostToDevice));

			// Clear the 1st regressor because we set it above to be Jacobi.
			cudaMemset(p_regressors2, 0, sizeof(f64_vec2)*NMINOR*4);
			cudaMemset(p_regressors_iz, 0, sizeof(f64)*NMINOR*4);
			cudaMemset(p_regressors_ez, 0, sizeof(f64)*NMINOR*4);
			kernelPopulateRegressors_from_iRing_RHS << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2,
					p_regressors_iz,
					p_regressors_ez,
					p_regressors2 + NMINOR,
					p_regressors_iz + NMINOR,
					p_regressors_ez + NMINOR,
					p_regressors2 + NMINOR * 2,
					p_regressors_iz + NMINOR * 2,
					p_regressors_ez + NMINOR * 2,
					p_selectflag,
					p_equation_index,
					p_iRing,
					p_RHS,
					whichRing);
			Call(cudaThreadSynchronize(), "cudaTS PopulateRegressors");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				1, 1, 1);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				1, 1, 1);

			// For now let's do without Jacobi, to experiment.




			if (0){//iIteration == 401) {


				SetConsoleTextAttribute(hConsole, 14);
				kernelPutative_v_from_matrix << <numTilesMinor, threadsPerTileMinor >> > (
					p_ita_i, // if 0 can avoid loading regrs
					p_vie2, p_vie, p_regressors2, p_regressors_iz, p_regressors_ez);
				// Assume coeff on Jacobi is 0 and coeff on solve is +0.045
				Call(cudaThreadSynchronize(), "cudaTS kernelPutative_v_from matrix");
				SetConsoleTextAttribute(hConsole, 15);

				// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
				// Heun 3 : Compute new equation matrix in new memory space
				// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

				CallMAC(cudaMemset(ionmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
				CallMAC(cudaMemset(elecmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
				kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
					<< <numTriTiles, threadsPerTileMinor >> > (
						pX_use->p_info,
						p_vie2,
						pX_use->p_v_n, // not used				
						ionmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.				
						p_selectflag, // whether it's in the smoosh to smash
						p_equation_index, // each one assigned an equation index				
						pX_use->p_izTri_vert, pX_use->p_szPBCtri_vert, pX_use->p_izNeigh_TriMinor, pX_use->p_szPBC_triminor,
						p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
						p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
						pX_use->p_B,
						1, m_i_, 1.0 / m_i_
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

				kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
					<< <numTriTiles, threadsPerTileMinor >> > (
						pX_use->p_info,
						p_vie2,
						// For neutral it needs a different pointer.
						pX_use->p_v_n, // not used
						elecmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.
						p_selectflag, // whether it's in the smoosh to smash
						p_equation_index, // each one assigned an equation index
						pX_use->p_izTri_vert, pX_use->p_szPBCtri_vert, pX_use->p_izNeigh_TriMinor, pX_use->p_szPBC_triminor,
						p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
						p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
						pX_use->p_B,
						2, m_e_, 1.0 / m_e_
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

				printf("done 2nd viscous elecmomflux_eqns\n press g\n");
				while (getch() != 'g');

				CallMAC(cudaMemset(p_eqns, 0, sizeof(f64) * 4 * 4 * EQNS_TOTAL*EQNS_TOTAL));
				CallMAC(cudaMemset(p_RHS, 0, sizeof(f64) * 4 * EQNS_TOTAL));
				kernelCreateEquations << <numTilesMinor, threadsPerTileMinor >> > (
					hsub,
					pX_use->p_info,
					ionmomflux_eqns,
					elecmomflux_eqns,
					pX_use->p_n_minor,
					pX_use->p_AreaMinor,
					pX_use->p_izNeigh_TriMinor, pX_use->p_izTri_vert,
					p_equation_index, // each one assigned an equation index
					p_selectflag,
					p_eqns2, // 4 x EQNS_TOTAL x 4 x EQNS_TOTAL
					p_epsilon_xy,
					p_epsilon_iz,
					p_epsilon_ez,
					p_RHS   // resets p_RHS
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelCreateEquations");


				cudaMemcpy(p_eqns_host, ionmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
					cudaMemcpyDeviceToHost);
				fp = fopen("ionmatrix__2.txt", "w");
				for (i = 0; i < 3 * EQNS_TOTAL; i++) {
					for (j = 0; j < 3 * EQNS_TOTAL; j++)
					{
						fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
					};
					fprintf(fp, "\n");
				}
				fclose(fp);

				printf("spitted 2 matrices\n");
				while (1)getch();
			};
#define NOHEUN
#ifndef NOHEUN
			/*
			// We could certainly make kernelCreateDByDBetaCoeffmatrix work for specific places.
			// However for now it's this, independent:

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 3,
					p_d_eps_iz_by_d_beta_i + NMINOR * 2, // The change created by the outermost ring only -- this is not adequate because penultimate ring can be on the outside.
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");
			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 3,
					p_d_eps_ez_by_d_beta_i + NMINOR * 2,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
			kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 3,
					p_d_epsxy_by_d_beta_i + NMINOR * 2,
					p__invmatrix);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");
			// Solution --- put final and penultimate ring together in one regressor. Done.
			kernelZeroWithinRings << <numTilesMinor, threadsPerTileMinor >> > (
				p_regressors_ez + NMINOR * 3,
				p_selectflag				);
			Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
			kernelZeroWithinRings << <numTilesMinor, threadsPerTileMinor >> > (
				p_regressors_iz + NMINOR * 3,
				p_selectflag				);
			Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
			kernelZeroWithinRings2 << <numTilesMinor, threadsPerTileMinor >> > (
				p_regressors2 + NMINOR * 3,
				p_selectflag				);
			Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
			*/
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			// Heun 2 : Create putative v that involves applying matrix solve move with coefficient 1 on this and on Jacobi
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			 
			SetConsoleTextAttribute(hConsole, 14);
			kernelPutative_v_from_matrix << <numTilesMinor, threadsPerTileMinor >> > (
				p_ita_i, // if 0 can avoid loading regrs
				p_vie2, p_vie, p_regressors2, p_regressors_iz, p_regressors_ez);
			// Assume coeff on Jacobi is minus 1 and coeff on solve is +1
			Call(cudaThreadSynchronize(), "cudaTS kernelPutative_v_from matrix");
			SetConsoleTextAttribute(hConsole, 15);

			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			// Heun 3 : Compute new equation matrix in new memory space
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

			CallMAC(cudaMemset(ionmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
			CallMAC(cudaMemset(elecmomflux_eqns, 0, sizeof(f64) * 3 * 3 * EQNS_TOTAL*EQNS_TOTAL));
			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
				<< <numTriTiles, threadsPerTileMinor >> > (
					pX_use->p_info,
					p_vie2,
					pX_use->p_v_n, // not used				
					ionmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.				
					p_selectflag, // whether it's in the smoosh to smash
					p_equation_index, // each one assigned an equation index				
					pX_use->p_izTri_vert, pX_use->p_szPBCtri_vert, pX_use->p_izNeigh_TriMinor, pX_use->p_szPBC_triminor,
					p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
					p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
					pX_use->p_B,
					1, m_i_, 1.0 / m_i_
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
				<< <numTriTiles, threadsPerTileMinor >> > (
					pX_use->p_info,
					p_vie2,
					// For neutral it needs a different pointer.
					pX_use->p_v_n, // not used
					elecmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.
					p_selectflag, // whether it's in the smoosh to smash
					p_equation_index, // each one assigned an equation index
					pX_use->p_izTri_vert, pX_use->p_szPBCtri_vert, pX_use->p_izNeigh_TriMinor, pX_use->p_szPBC_triminor,
					p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
					p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
					pX_use->p_B,
					2, m_e_, 1.0 / m_e_
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

			printf("done 2nd viscous elecmomflux_eqns\n press g\n");
			while (getch() != 'g');

			CallMAC(cudaMemset(p_eqns, 0, sizeof(f64) * 4 * 4 * EQNS_TOTAL*EQNS_TOTAL));
			CallMAC(cudaMemset(p_RHS, 0, sizeof(f64) * 4 * EQNS_TOTAL));
			kernelCreateEquations << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_use->p_info,
				ionmomflux_eqns,
				elecmomflux_eqns,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				pX_use->p_izNeigh_TriMinor, pX_use->p_izTri_vert,
				p_equation_index, // each one assigned an equation index
				p_selectflag,
				p_eqns2, // 4 x EQNS_TOTAL x 4 x EQNS_TOTAL
				p_epsilon_xy,
				p_epsilon_iz,
				p_epsilon_ez,
				p_RHS   // resets p_RHS
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreateEquations");


			cudaMemcpy(p_eqns_host, ionmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			fp = fopen("ionmatrix__2.txt", "w");
			for (i = 0; i < 3 * EQNS_TOTAL; i++) {
				for (j = 0; j < 3 * EQNS_TOTAL; j++)
				{
					fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
				};
				fprintf(fp, "\n");
			}
			fclose(fp);

			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			// Heun 4 : Compute new equation matrix in new memory space
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

			kernelSimpleAverage << <4 * 4 * EQNS_TOTAL, EQNS_TOTAL >> > (p_eqns, p_eqns2); // updates p_eqns
			Call(cudaThreadSynchronize(), "cudaTS kernelSimpleAverage");
			
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			// Heun 5 : Run matrix solve again, on CPU
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

			cudaMemcpy(p_eqns_host, p_eqns, 4 * EQNS_TOTAL * 4 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_RHS, 4 * EQNS_TOTAL * sizeof(f64), cudaMemcpyDeviceToHost);

			fp = fopen("eqnmatrix2.txt", "w");
			for (i = 0; i < 4 * EQNS_TOTAL; i++) {
				for (j = 0; j < 4 * EQNS_TOTAL; j++)
				{
					fprintf(fp, "%1.14E ", p_eqns_host[i * 4 * EQNS_TOTAL + j]);
				};
				fprintf(fp, "  |  %1.14E \n", p_temphost5[i]);
			}
			fclose(fp);
			cudaMemcpy(p_eqns_host, ionmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			fp = fopen("ionmatrix2.txt", "w");
			for (i = 0; i < 3 * EQNS_TOTAL; i++) {
				for (j = 0; j < 3 * EQNS_TOTAL; j++)
				{
					fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
				};
				fprintf(fp, "\n");
			}
			fclose(fp);
			/*
			cudaMemcpy(p_eqns_host, elecmomflux_eqns, 3 * EQNS_TOTAL * 3 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			fp = fopen("elecmatrix2.txt", "w");
			for (i = 0; i < 3 * EQNS_TOTAL; i++) {
				for (j = 0; j < 3 * EQNS_TOTAL; j++)
				{
					fprintf(fp, "%1.14E ", p_eqns_host[i * 3 * EQNS_TOTAL + j]);
				};
				fprintf(fp, "\n");
			}
			fclose(fp);
			printf("outputted elecmatrix2.txt");

			cudaMemoryTest();*/
			printf("Solving equations, again.\n");
			cudaMemcpy(p_eqns_host, p_eqns, 4 * EQNS_TOTAL * 4 * EQNS_TOTAL * sizeof(f64),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_RHS, 4 * EQNS_TOTAL * sizeof(f64), cudaMemcpyDeviceToHost);

			for (int i = 0; i < 4 * EQNS_TOTAL; i++)
				memcpy(&(matLU.LU[i][0]), &(p_eqns_host[i * 4 * EQNS_TOTAL]), sizeof(f64) * 4 * EQNS_TOTAL);
			matLU.LUdecomp();
			matLU.LUSolve(p_temphost5, result); // solving Ax = b and it's (b, x).
			printf("Done LU. Dimension was %d.\n", 4 * EQNS_TOTAL);

			cudaMemcpy(p_RHS, result, sizeof(f64) * 4 * EQNS_TOTAL, cudaMemcpyHostToDevice);
			/*fp = fopen("soln2.txt", "w");
			for (i = 0; i < 4 * EQNS_TOTAL; i++)
			{
				fprintf(fp, "indx %d Soln %1.14E \n", matLU.indx[i], result[i]);
			}
			fclose(fp);*/
			printf("done soln.txt");

			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			// Heun 6a : Regressors within solve region again.
			// Heun 6b : K.O.self with regressors in rings concentric to the solve region 
			// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
			
			// Clear the preexisting regressors:
			cudaMemset(p_regressors2, 0, sizeof(f64_vec2)*NMINOR * 4);
			cudaMemset(p_regressors_iz, 0, sizeof(f64)*NMINOR * 4);
			cudaMemset(p_regressors_ez, 0, sizeof(f64)*NMINOR * 4);

			kernelPopulateRegressors_from_iRing_RHS << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2,
					p_regressors_iz,
					p_regressors_ez,
					p_regressors2 + NMINOR,
					p_regressors_iz + NMINOR,
					p_regressors_ez + NMINOR,
					p_regressors2 + NMINOR * 2,
					p_regressors_iz + NMINOR * 2,
					p_regressors_ez + NMINOR * 2,
					p_selectflag,
					p_equation_index,
					p_iRing,
					p_RHS,
					whichRing);
			Call(cudaThreadSynchronize(), "cudaTS PopulateRegressors");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				1, 1, 1);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				1, 1, 1);

			printf("Done Heun.\n");
#endif
			
			for (i = 3; i < REGRESSORS; i++)
			{
				kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
					(p_regressors_iz + NMINOR * i,
						p_d_eps_iz_by_d_beta_i + NMINOR * (i-1),
						p__invcoeffselfviz);
				Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");
				kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
					(p_regressors_ez + NMINOR * i,
						p_d_eps_ez_by_d_beta_i + NMINOR * (i-1),
						p__invcoeffselfvez);
				Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
				kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
					(p_regressors2 + NMINOR * i,
						p_d_epsxy_by_d_beta_i + NMINOR * (i-1),
						p__invmatrix);
				Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");
				kernelZeroWithinRings << <numTilesMinor, threadsPerTileMinor >> > (
					p_regressors_ez + NMINOR * i,
					p_selectflag
					);
				Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
				kernelZeroWithinRings << <numTilesMinor, threadsPerTileMinor >> > (
					p_regressors_iz + NMINOR * i,
					p_selectflag
					);
				Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
				kernelZeroWithinRings2 << <numTilesMinor, threadsPerTileMinor >> > (
					p_regressors2 + NMINOR * i,
					p_selectflag
					);
				Call(cudaThreadSynchronize(), "cudaTS ZeroWithin");
				SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, i,
					1, 1, 1);
			};
			
			//	Careful of autocorrelation. Is there a way to reduce it?
			/*
			// Now spit out regressors for the 4 values:
			fp = fopen("regrs.txt", "a");
			fprintf(fp, "\n");
			for (int iEqn = 0; iEqn < EQNS_TOTAL; iEqn++) {

				cudaMemcpy(&tempvec2, &(p_regressors2[eqnlist[iEqn]]), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64, &(p_regressors_iz[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64_2, &(p_regressors_ez[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);

				printf("iEqn %d : regr2 %1.8E %1.8E regr_iz %1.7E regr_ez %1.8E | ", eqnlist[iEqn], tempvec2.x, tempvec2.y, tempf64, tempf64_2);
				fprintf(fp, "iEqn %d : regr2 %1.14E %1.14E regr_iz %1.14E regr_ez %1.14E | ", eqnlist[iEqn], tempvec2.x, tempvec2.y, tempf64, tempf64_2);
				cudaMemcpy(&tempvec2, &(p_regressors2[eqnlist[iEqn]+NMINOR]), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64, &(p_regressors_iz[eqnlist[iEqn] + NMINOR]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64_2, &(p_regressors_ez[eqnlist[iEqn] + NMINOR]), sizeof(f64), cudaMemcpyDeviceToHost);

				printf("regr2 %1.8E %1.8E iz %1.8E ez %1.8E | ", tempvec2.x, tempvec2.y, tempf64, tempf64_2);
				fprintf(fp, "regr2 %1.14E %1.14E iz %1.14E ez %1.14E | ", tempvec2.x, tempvec2.y, tempf64, tempf64_2);

				cudaMemcpy(&tempvec2, &(p_regressors2[eqnlist[iEqn] + 2*NMINOR]), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64, &(p_regressors_iz[eqnlist[iEqn] + 2*NMINOR]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tempf64_2, &(p_regressors_ez[eqnlist[iEqn] + 2*NMINOR]), sizeof(f64), cudaMemcpyDeviceToHost);

				printf("regr2 %1.8E %1.8E iz %1.8E ez %1.8E | \n", tempvec2.x, tempvec2.y, tempf64, tempf64_2);
				fprintf(fp, "regr2 %1.14E %1.14E iz %1.14E ez %1.14E | \n", tempvec2.x, tempvec2.y, tempf64, tempf64_2);

			}
			fclose(fp);
			getch();
			printf("here6\n");*/

			break;

		case 3:
			// Mixture move

			// We now set the 0th regressor to be Jacobi xyz.
			// So think carefully -- there will be very high collinearity with the components.
			// One way is to split the 0th into 3 components.
			// Advantageous?

			cudaMemcpy(p_regressors_iz + NMINOR, p_regressors_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemset(p_regressors_iz, 0, sizeof(f64)*NMINOR);
			cudaMemcpy(p_regressors_ez + NMINOR * 2, p_regressors_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemset(p_regressors_ez, 0, sizeof(f64)*NMINOR);
			
			//kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
			//	(p_regressors2,
			//		p_epsilon_xy,
			//		p__invmatrix);
			//Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");

			//kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
			//	(p_regressors_iz + NMINOR,
			//		p_epsilon_iz,
			//		p__invcoeffselfviz);
			//Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			//kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
			//	(p_regressors_ez + NMINOR * 2,
			//		p_epsilon_ez,
			//		p__invcoeffselfvez);
			//Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

			// 4th one is all together Richardson:
			cudaMemcpy(p_regressors2 + NMINOR * 3,
				p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors_iz + NMINOR * 3,
				p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors_ez + NMINOR * 3,
				p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			// to do: add scaling?

			// Collect deps/dbeta for first 4 regressors:
			// ===================================================

		//	Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
		//		"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
		//	cudaMemset(longaddress, 1L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);

		//	Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
		//		"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
		//	cudaMemset(longaddress, 0L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub,p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				0, 1, 0);// IZ_ONLY);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				0, 0, 1);// EZ_ONLY);						
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 3,
				1, 1, 1); // all apply
				//1, 0, 1);
				
			// Regressor 5:
				
			Subtract_xy << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i, p_regressors2);
			Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
			Subtract << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 5, p_d_eps_iz_by_d_beta_i + NMINOR, p_regressors_iz + NMINOR);
			Call(cudaThreadSynchronize(), "cudaTS Subtract z");
			Subtract << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 5, p_d_eps_ez_by_d_beta_i + NMINOR*2, p_regressors_ez + NMINOR*2);
			Call(cudaThreadSynchronize(), "cudaTS Subtract z");

			//cudaMemcpy(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			// Regressor 6:
			//cudaMemcpy(p_regressors_iz + NMINOR * 5, p_d_eps_iz_by_d_beta_i + NMINOR, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(p_regressors_ez + NMINOR * 5, p_d_eps_ez_by_d_beta_i + NMINOR*2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 4,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
				0, 1, 1); // all apply
				
			// 7 = previous xy move, or historic xy move
			// 8 = previous iz and ez moves, or historic.

			if (iIteration == 0) {
				if (bHistoryVisc == 0) {
					// in this case let's start with just 2nd derivatives:
					cudaMemcpy(p_regressors2 + NMINOR * 6, p_d_epsxy_by_d_beta_i + 4 * NMINOR, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);

					cudaMemcpy(p_regressors_iz + NMINOR * 7, p_d_eps_iz_by_d_beta_i + 5 * NMINOR, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
					cudaMemcpy(p_regressors_ez + NMINOR * 7, p_d_eps_ez_by_d_beta_i + 5 * NMINOR, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

				} else {
					// Historic move split into xy, izez :
					SplitVector4 << <numTilesMinor, threadsPerTileMinor >> > (
						p_regressors2 + NMINOR * 6, p_regressors_iz + NMINOR * 7, p_regressors_ez + NMINOR * 7,
						p_stored_move4, p_Selectflag
						);
					Call(cudaThreadSynchronize(), "cudaTS SplitVector4");
				};
			} else {
				// Previous move split into xy, izez :
				SplitVector4 << < numTilesMinor, threadsPerTileMinor >> > (
					p_regressors2 + NMINOR * 6, p_regressors_iz + NMINOR * 7, p_regressors_ez + NMINOR * 7,
					p_stored_move4, p_Selectflag
					);
				Call(cudaThreadSynchronize(), "cudaTS SplitVector4");
			};
			
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 6,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 7,
				0, 1, 1); 
				
			break;
		case 1:
			// Primarily z move.

			// Leave the Jacobi regressor at 0th.

			// 0: Jacobi all
			// 1: allow this Jacobi viz:
			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR,
					p_epsilon_iz,
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			// 2: Richardson izez
			cudaMemcpy(p_regressors_iz + NMINOR * 2, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_regressors_ez + NMINOR * 2, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

			// 3: Jacobi vez:
			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 3,
					p_epsilon_ez,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
			 
			// Collect deps/dbeta for first 4 regressors:
			// ===================================================

	//		Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
	//			"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
	//		cudaMemset(longaddress, 1L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);

	//		Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
	//			"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
	//		cudaMemset(longaddress, 0L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				0, 1, 0);// IZ_ONLY);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				0, 1, 1);// EZ_ONLY);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 3,
				0, 0, 1); // all apply

			kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR*4,
					p_d_epsxy_by_d_beta_i + NMINOR,
					p__invmatrix);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");

			kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 5,
					p_d_epsxy_by_d_beta_i + NMINOR*3,
					p__invmatrix);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");
			 
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 4,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
				1, 0, 0); // all apply

			SplitVector4 << < numTilesMinor, threadsPerTileMinor >> > (
				p_d_epsxy_by_d_beta_e, // DISCARD
				p_regressors_iz + NMINOR * 6, p_regressors_ez + NMINOR * 7,
				p_stored_move4, p_Selectflag
				);
			Call(cudaThreadSynchronize(), "cudaTS SplitVector4");

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 6,
				0, 1, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 7,
				0, 0, 1); // all apply


			// If this method isn't effective then we need to swap out the 'correcting' xy regressors
			// or combine them into 1 and combine prev directions into 1
			// and use more derivatives in the regression.
			// That is almost certainly going to turn out to be a good idea.

			break;

		case 2:
			// Primarily xy move.

			// 0: Jxy
			// 1: Jx
			// 2: Jy
			// 3: eps xy
			// 4: deps/dbeta0 - Jxy
			// 5: deps/dbeta0 * eps_xy
			// 6: J_iz,ez(change due to beta 0)
			// 7: previous xy -- maybe we need more previous terms


			// OLD:

			// . 0: Jxy
			// . 1: eps xy
			// . 2: deps/dbeta0 - Jxy
			// . 3: multiply components deps/dbeta0 * eps
			// . 4: deps/dbeta1 - eps xy
			// . 6: J_iz,ez(change due to beta 0)
			// . prev prev everything?? --- need to bring 2x stored as an improvement experiment
			// for now:
			//   5: deps/dbeta4 - regr 4			
			// . 7: previous xy


			// Keep 0th = Jacobi 4D.

			// Bear in mind high collinearity here.
			// No value in having separate go like this:
			//kernelCreateJacobiRegressorxy << <numTilesMinor, threadsPerTileMinor >> >
			//	(p_regressors2 + NMINOR * 0,
			//		p_epsilon_xy,
			//		p__invmatrix);
			//Call(cudaThreadSynchronize(), "cudaTS Jacobi xy");
			
			kernelCreateJacobiRegressor_x << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 1,
					p_epsilon_xy,
					p__invcoeffself_x);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi x");
			
			kernelCreateJacobiRegressor_y << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors2 + NMINOR * 2,
					p_epsilon_xy,
					p__invcoeffself_y);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi y");
			
			cudaMemcpy(p_regressors2 + NMINOR * 3, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);

			Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
				"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
			cudaMemset(longaddress, 1L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);
			
			Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
				"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
			cudaMemset(longaddress, 0L, sizeof(long));

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 3,
				1, 0, 0);


			Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
				(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i + (0) * NMINOR, p_regressors2 + (0)*NMINOR);
			Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");

			Multiply_components_xy << <numTilesMinor, threadsPerTileMinor >> > // c=ba
				(p_regressors2 + NMINOR * 5, p_d_epsxy_by_d_beta_i + 0 * NMINOR, p_epsilon_xy);
			Call(cudaThreadSynchronize(), "cudaTS Multiply_components_xy");

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 4,
				1, 0, 0);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
				1, 0, 0);
			
			//Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
			//	(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i + 1 * NMINOR, p_regressors2 + 1*NMINOR);
			//Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");

			//SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 4,
			//	1, 0, 0);

			//Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
			//	(p_regressors2 + NMINOR * 5, p_d_epsxy_by_d_beta_i + 4 * NMINOR, p_regressors2 +4*NMINOR);
			//Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");

			//SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
			//	1, 0, 0);
			
			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 6,
					p_d_eps_iz_by_d_beta_i,
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 6,
					p_d_eps_ez_by_d_beta_i,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 6,
				0, 1, 1);

			/*
			
			cudaMemcpy(p_regressors2 + NMINOR * 0, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 0, 0);

			// Version 3:
			Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
				(p_regressors2 + NMINOR * 1, p_d_epsxy_by_d_beta_i + (0) * NMINOR, p_regressors2 + (0)*NMINOR);
			Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 1,
				1, 0, 0);

			Multiply_components_xy << <numTilesMinor, threadsPerTileMinor >> > // c=ba
				(p_regressors2 + NMINOR * 2, p_d_epsxy_by_d_beta_i + 0 * NMINOR, p_epsilon_xy);
			Call(cudaThreadSynchronize(), "cudaTS Multiply_components_xy");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 2,
				1, 0, 0);

			Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
				(p_regressors2 + NMINOR * 3, p_d_epsxy_by_d_beta_i + 1 * NMINOR, p_regressors2 + 1*NMINOR);
			Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 3,
				1, 0, 0);

			Multiply_components_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
				(p_regressors2 + NMINOR * 4, p_d_epsxy_by_d_beta_i + 1 * NMINOR, p_epsilon_xy);
			Call(cudaThreadSynchronize(), "cudaTS Multiply_xy");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 4,
				1, 0, 0);

			Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
				(p_regressors2 + NMINOR * 5, p_d_epsxy_by_d_beta_i + 3 * NMINOR, p_regressors2 + 3 * NMINOR);
			Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
				1, 0, 0);
			
			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 6,
					p_d_eps_iz_by_d_beta_i,
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 6,
					p_d_eps_ez_by_d_beta_i,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 6,
				0, 1, 1);



			// Version 2: 
			/*
			for (int jj = 1; jj < 5; jj++) {
				Subtract_xy << <numTilesMinor, threadsPerTileMinor >> > // c=b-a
					(p_regressors2 + NMINOR * jj, p_d_epsxy_by_d_beta_i + (jj-1) * NMINOR, p_regressors2 + (jj-1)*NMINOR);
				Call(cudaThreadSynchronize(), "cudaTS Subtract_xy");
				SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, jj,
					1, 0, 0);
			};
			
			// Did 1 thru 4.
			// 5 = Jacobi(iz, ez)
			// 6 = iz,ez Jacobi(deps0)
			// 7 = prev move (xyz)

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 5,
					p_epsilon_iz,
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 5,
					p_epsilon_ez,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_iz + NMINOR * 6,
					p_d_eps_iz_by_d_beta_i,
					p__invcoeffselfviz);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi iz");

			kernelCreateJacobiRegressorz << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors_ez + NMINOR * 6,
					p_d_eps_ez_by_d_beta_i,
					p__invcoeffselfvez);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");

			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 5,
				0, 1, 1);
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 6,
				0, 1, 1);
				*/


			SplitVector4 << < numTilesMinor, threadsPerTileMinor >> > (
				p_regressors2 + NMINOR*7,
				p_dump, p_dump2,
				p_stored_move4, p_Selectflag
				);
			Call(cudaThreadSynchronize(), "cudaTS Jacobi ez");
			
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 7,
				1, 0, 0); 
			
			break;
		}
		} // scoping brace for matLU

		 
			/*// Normalize regressor: divide by L2 norm.
			// ___________________________________________:
			kernelAccumulateSumOfSquares_4 << <numTilesMinor, threadsPerTileMinor >> >
				(p_regressors4 + i*NMINOR,
					p_SS
					);
			Call(cudaThreadSynchronize(), "SS4");

			f64 RSS = 0.0;
			cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++) {
				RSS += p_SS_host[iTile];
			}

			// It's one regressor so we cannot scale different dimensions differently.

			f64 L2regress = sqrt(RSS / (real)NMINOR);
			if (L2regress == 0.0) L2regress = 1.0;

			ScaleVector4 << <numTilesMinor, threadsPerTileMinor >> > (
				p_regressors4 + i*NMINOR, 1.0 / L2regress);
			Call(cudaThreadSynchronize(), "ScaleVector4");*/

		//	// ============================================================
		//	// Now calculate deps/dbeta for this regressor
		//	// ===============================================
		//
		//	cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		//	cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
		//	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		//	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		//	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
		//
		//		pX_use->p_info,
		//		p_regressors4 + i*NMINOR,
		//		pX_use->p_izTri_vert,
		//		pX_use->p_szPBCtri_vert,
		//		pX_use->p_izNeigh_TriMinor,
		//		pX_use->p_szPBC_triminor,
		//
		//		p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		//		p_ita_e, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		//		p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		//		p_nu_e, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
		//
		//		pX_use->p_B,
		//		p_MAR_ion2, // just accumulates
		//		p_MAR_elec2,
		//		NT_addition_rates_d_temp,
		//		// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
		//		NT_addition_tri_d);
		//	Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");
		//
		//	// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		//	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		//	kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
		//		// eps = v - (v_k +- h [viscous effect])
		//		// x = -eps/coeffself
		//		hsub,
		//		pX_use->p_info,
		//		p_regressors4 + i*NMINOR,
		//		zero_vec4,
		//		p_MAR_ion2, p_MAR_elec2,
		//		pX_use->p_n_minor,
		//		pX_use->p_AreaMinor,
		//		p_d_epsxy_by_d_beta_i + i*NMINOR,
		//		p_d_eps_iz_by_d_beta_i + i*NMINOR,
		//		p_d_eps_ez_by_d_beta_i + i*NMINOR,
		//		0
		//		);
		//	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");
		//}; // next i
		// All regressors now created, and deps/dbeta obtained.

		////	Regressors:
		//cudaMemcpy(p_tempvec2_host, p_regressors2, sizeof(f64_vec2)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost1, p_regressors_iz, sizeof(f64)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost2, p_regressors_ez, sizeof(f64)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);
		////  epsilon:
		//cudaMemcpy(p_tempvec2_host2, p_epsilon_xy, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost3, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost4, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		////  
		//cudaMemcpy(p_tempvec2_host3, p_d_epsxy_by_d_beta_i, sizeof(f64_vec2)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost5, p_d_eps_iz_by_d_beta_i, sizeof(f64)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost6, p_d_eps_ez_by_d_beta_i, sizeof(f64)*NMINOR*REGRESSORS, cudaMemcpyDeviceToHost);


		//dbgfile = fopen("debug.txt", "w");
		//for (long iMinor = 0; iMinor < NMINOR; iMinor++)
		//{
		//	fprintf(dbgfile, "eps %1.14E %1.14E %1.14E %1.14E | regr ", p_tempvec2_host2[iMinor].x, p_tempvec2_host2[iMinor].y,
		//		p_temphost3[iMinor], p_temphost4[iMinor]);
		//	for (int j = 0; j < REGRESSORS; j++)
		//	{
		//		fprintf(dbgfile, "%d : %1.14E %1.14E %1.14E %1.14E | ",j, p_tempvec2_host[iMinor + j*NMINOR].x, p_tempvec2_host[iMinor + j*NMINOR].y,
		//			p_temphost1[iMinor + j*NMINOR], p_temphost2[iMinor + j*NMINOR]);
		//	}
		//	fprintf(dbgfile, " deps/dbeta ");
		//	for (int j = 0; j < REGRESSORS; j++)
		//	{
		//		fprintf(dbgfile, "%d : %1.14E %1.14E %1.14E %1.14E | ", j, 
		//			p_tempvec2_host3[iMinor + j*NMINOR].x, p_tempvec2_host3[iMinor + j*NMINOR].y,
		//			p_temphost5[iMinor + j*NMINOR], p_temphost6[iMinor + j*NMINOR]);
		//	}
		//	fprintf(dbgfile, "\n");
		//};
		//fclose(dbgfile);

		if ((iIteration > ITERGRAPH) || (bDebug)) {

			structural info;
			cudaMemcpy(&info, pX_use->p_info + CHOSEN, sizeof(structural), cudaMemcpyDeviceToHost);
			printf("CHOSEN %d \n"
				"pos %1.9E %1.9E flag %d neigh_len %d \n\n", 
				CHOSEN,
				info.pos.x, info.pos.y, info.flag, info.neigh_len);
			
			//GlobalSuppressSuccessVerbosity = false;
			GlobalCutaway = false;

			// 1. Represent epsilon xy as xy velocity

			DivideVec2<<<numTilesMinor, threadsPerTileMinor>>>(p_d_epsxy_by_d_beta_e, p_epsilon_xy); // subtract b from a
			Call(cudaThreadSynchronize(), "cudaTS DivideVec2");
			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_e, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
			f64 maxx = 0.0, minx = 0.0, miny = 0.0, maxy = 0.0;
			long iMin = 0, iMax = 0, iMiy = 0, iMay = 0;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				if (p_tempvec2_host[iMinor].x > maxx) {
					maxx = p_tempvec2_host[iMinor].x;
					iMax = iMinor;
				};
				if (p_tempvec2_host[iMinor].x < minx) {
					minx = p_tempvec2_host[iMinor].x;
					iMin = iMinor;
				};
				if (p_tempvec2_host[iMinor].y > maxy) {
					maxy = p_tempvec2_host[iMinor].y;
					iMay = iMinor;
				};
				if (p_tempvec2_host[iMinor].y < miny) {
					miny = p_tempvec2_host[iMinor].y;
					iMiy = iMinor;
				};
			};
			printf("Results of prev move:\n");
			printf("Ratio over eps: maxx %1.9E at %d ; minx %1.9E at %d\n", maxx, iMax, minx, iMin);
			printf("Ratio over eps: maxy %1.9E at %d ; miny %1.9E at %d\n", maxy, iMay, miny, iMiy);

			// Do again for verts:

			iMin = 0; iMax = 0; iMiy = 0; iMay = 0;
			maxx = 0.0; minx = 0.0; maxy = 0.0; miny = 0.0;
			for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
			{
				if (p_tempvec2_host[iMinor].x > maxx) {
					maxx = p_tempvec2_host[iMinor].x;
					iMax = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].x < minx) {
					minx = p_tempvec2_host[iMinor].x;
					iMin = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].y > maxy) {
					maxy = p_tempvec2_host[iMinor].y;
					iMay = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].y < miny) {
					miny = p_tempvec2_host[iMinor].y;
					iMiy = iMinor-BEGINNING_OF_CENTRAL;
				};
			};
	
			printf("In verts: maxx %1.9E at %d ; minx %1.9E at %d\n", maxx, iMax, minx, iMin);
			printf("In verts: maxy %1.9E at %d ; miny %1.9E at %d\n", maxy, iMay, miny, iMiy);

			long iVertex;
			char buffer[256];
			Vertex * pVertex = pTriMesh->X;
			plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			maxy = 0.0;
			iMax = -1;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;

				if (pdata->temp.y*pdata->temp.y > maxy*maxy) {
					maxy = pdata->temp.y;
					iMax = iVertex;
				};

				++pVertex;
				++pdata;
			}
			sprintf(buffer, "epsilon_y_rat iter %d", iIteration);
			Graph[0].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("epsilon_y_rat maxy %1.9E at %d\n", maxy, iMax);




			SubtractVec2 << <numTilesMinor, threadsPerTileMinor >> >(p_d_epsxy_by_d_beta_e, p_epsilon_xy); // subtract b from a
			Call(cudaThreadSynchronize(), "cudaTS SubtractVec2");
			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_e, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
			maxx = 0.0; minx = 0.0; miny = 0.0; maxy = 0.0;
			iMin = 0; iMax = 0; iMiy = 0; iMay = 0;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				if (p_tempvec2_host[iMinor].x > maxx) {
					maxx = p_tempvec2_host[iMinor].x;
					iMax = iMinor;
				};
				if (p_tempvec2_host[iMinor].x < minx) {
					minx = p_tempvec2_host[iMinor].x;
					iMin = iMinor;
				};
				if (p_tempvec2_host[iMinor].y > maxy) {
					maxy = p_tempvec2_host[iMinor].y;
					iMay = iMinor;
				};
				if (p_tempvec2_host[iMinor].y < miny) {
					miny = p_tempvec2_host[iMinor].y;
					iMiy = iMinor;
				};
			};
			printf("Diff (getting worse/beyond): maxx %1.9E at %d ; minx %1.9E at %d\n", maxx, iMax, minx, iMin);
			printf("Diff (getting worse/beyond): maxy %1.9E at %d ; miny %1.9E at %d\n", maxy, iMay, miny, iMiy);

			// Do again for verts:

			iMin = 0; iMax = 0; iMiy = 0; iMay = 0;
			maxx = 0.0; minx = 0.0; maxy = 0.0; miny = 0.0;
			for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
			{
				if (p_tempvec2_host[iMinor].x > maxx) {
					maxx = p_tempvec2_host[iMinor].x;
					iMax = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].x < minx) {
					minx = p_tempvec2_host[iMinor].x;
					iMin = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].y > maxy) {
					maxy = p_tempvec2_host[iMinor].y;
					iMay = iMinor - BEGINNING_OF_CENTRAL;
				};
				if (p_tempvec2_host[iMinor].y < miny) {
					miny = p_tempvec2_host[iMinor].y;
					iMiy = iMinor - BEGINNING_OF_CENTRAL;
				};
			};

			printf("In verts: maxx %1.9E at %d ; minx %1.9E at %d\n", maxx, iMax, minx, iMin);
			printf("In verts: maxy %1.9E at %d ; miny %1.9E at %d\n", maxy, iMay, miny, iMiy);


			SetActiveWindow(hwndGraphics);
						
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			maxy = 0.0;
			iMax = -1;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;

				if (pdata->temp.y*pdata->temp.y > maxy*maxy) {
					maxy = pdata->temp.y;
					iMax = iVertex;
				};

				++pVertex;
				++pdata;
			}
			sprintf(buffer, "epsilon_y_diff iter %d", iIteration);
			Graph[1].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("epsilon_y_diff maxy %1.9E at %d\n", maxy, iMax);

			maxy = 0.0;
			iMax = -1;
			cudaMemcpy(p_tempvec2_host, p_epsilon_xy + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				if (pdata->temp.y*pdata->temp.y > maxy*maxy) {
					maxy = pdata->temp.y;
					iMax = iVertex;
				};
				++pVertex;
				++pdata;
			}						
			sprintf(buffer, "epsilon_y iter %d", iIteration);
			Graph[2].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("iMax = vertex %d eps.y = %1.10E \n", iMax, maxy);
			f64_vec2 tempvec2;
			cudaMemcpy(&tempvec2, &(pX_use->p_info[iMax + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
			printf("position %1.9E %1.9E \n", tempvec2.x, tempvec2.y);

			printf("remember this is before move.\n");

			/*cudaMemcpy(p_tempvec4_host, p_vie + BEGINNING_OF_CENTRAL, sizeof(v4)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_tempvec4_2_host, p_vie_k + BEGINNING_OF_CENTRAL, sizeof(v4)*NUMVERTICES, cudaMemcpyDeviceToHost);

			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec4_host[iVertex].vxy.x - p_tempvec4_2_host[iVertex].vxy.x;
				pdata->temp.y = p_tempvec4_host[iVertex].vxy.y - p_tempvec4_2_host[iVertex].vxy.y;
				++pVertex;
				++pdata;
			}
			sprintf(buffer, "v diff iter %d", iIteration);
			Graph[1].DrawSurface(buffer,
				VELOCITY_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				VELOCITY_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_AZ, pTriMesh);

			printf("vdiff at %d =  %1.9E\n", iMax, p_tempvec4_host[iMax].vxy.y - p_tempvec4_2_host[iMax].vxy.y);
			*/


			// Want: regressors x 3. Already putted epsilon
			// coeffself would be information?
			// Put Ae, AAe, AAAe --- this is informative
			// Show ratio: Ae = ROC epsilon / epsilon

			// That could be interesting.
			cudaMemcpy(p_inthost, p_Selectflag + BEGINNING_OF_CENTRAL, sizeof(int)*NUMVERTICES, cudaMemcpyDeviceToHost);
			
			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				++pVertex;
				++pdata;
			}
			sprintf(buffer, "depsy/d0 iter %d", iIteration);
			Graph[3].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			sprintf(buffer, "depsx/d0 iter %d", iIteration);
			Graph[4].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("depsx/d0 at %d = %1.9E \n", iMax, p_tempvec2_host[iMax].x);

			cudaMemcpy(p_tempvec2_host, p_regressors2 + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				++pVertex;
				++pdata;
			}
			sprintf(buffer, "regressor 0 iter %d", iIteration);
			Graph[5].DrawSurface(buffer,
				VELOCITY_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				VELOCITY_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
				false,
				GRAPH_AXY, pTriMesh);

			/*
			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i + NMINOR + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				++pVertex;
				++pdata;
			}
			sprintf(buffer, "deps/d1 iter %d", iIteration);
			Graph[3].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("deps/d1 at %d = %1.9E \n", iMax, p_tempvec2_host[iMax].y);
			

			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i + 2*NMINOR + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				++pVertex;
				++pdata;
			}
			sprintf(buffer, "deps/d2 iter %d", iIteration);
			Graph[4].DrawSurface(buffer,
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			printf("deps/d2 at %d = %1.9E \n", iMax, p_tempvec2_host[iMax].y);
			*/
//
//			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i + 2 * NMINOR + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//			pVertex = pTriMesh->X;
//			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
//			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//			{
//				pdata->temp.x = p_tempvec2_host[iVertex].x;
//				pdata->temp.y = p_tempvec2_host[iVertex].y;
//				++pVertex;
//				++pdata;
//			}
//			sprintf(buffer, "deps/d3 iter %d", iIteration);
//			Graph[4].DrawSurface(buffer,
//				VELOCITY_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
//				VELOCITY_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
//				false,
//				GRAPH_EPSILON, pTriMesh);
//
//			printf("deps/d3 at %d = %1.9E \n", iMax, p_tempvec2_host[iMax].y);
//

			cudaMemcpy(p_tempvec2_host, p_epsilon_xy + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_tempvec2_host[iVertex].x;
				pdata->temp.y = p_tempvec2_host[iVertex].y;
				++pVertex;
				++pdata;
			}
			/*cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			f64 vertexsum = 0.0;
			f64 maxy2 = 0.0, miny2 = 0.0;
			long iMax2 =0  , iMin2 = 0;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				// epsilon over depsilon
				// Try this: ROC of eps^2 in location. = 2*eps*deps

				pdata->temp.x *= p_tempvec2_host[iVertex].x;
				pdata->temp.y *= p_tempvec2_host[iVertex].y;
				vertexsum += pdata->temp.y;
				
				if (pdata->temp.y*pdata->temp.y > maxy2*maxy2) {
					maxy2 = pdata->temp.y;
					iMax2 = iVertex;
				};

				if (pdata->temp.y*pdata->temp.y < miny2*miny2) {
					miny2 = pdata->temp.y;
					iMin2 = iVertex;
				};
				
				pdata->temp.y = ((f64)p_inthost[iVertex])*2.0e-12;

				//if (fabs(pdata->temp.x) < 1.0e-6) {
				//	// doesn't matter
				//	pdata->temp.x = 1.0;
				//} else {
				//	
				//	// going to take deps/eps
				//	if (p_tempvec2_host[iVertex].x == 0.0) {
				//		// epsilon = 0; but we could be changing it
				//	}
				//	
				//	
				//	if (fabs(p_tempvec2_host[iVertex].x) < 0.001*fabs(pdata->temp.x)) {
				//		if (p_tempvec2_host[iVertex].x*pdata->temp.x < 0.0) {
				//			pdata->temp.x = -1000.0;
				//		} else {
				//			pdata->temp.x = 1000.0;
				//		}
				//	}
				//	else {
				//		pdata->temp.x /= p_tempvec2_host[iVertex].x;
				//	};
				//};
				//if (fabs(pdata->temp.y) < 1.0e-6) {
				//	pdata->temp.y = 1.0;
				//} else {
				//	if (fabs(p_tempvec2_host[iVertex].y) < 0.001*fabs()
				//	pdata->temp.y /= p_tempvec2_host[iVertex].y;
				//}
				++pVertex;
				++pdata;
			}

			//sprintf(buffer, "eps.x*deps.x iter %d", iIteration);
			//Graph[4].DrawSurface(buffer,
			//	DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			//	SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			//	false,
			//	GRAPH_EPSILON, pTriMesh);

//			sprintf(buffer, "eps.y*deps.y iter %d", iIteration);
//			Graph[5].DrawSurface(buffer,
//				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
//				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
//				false,
//				GRAPH_EPSILON, pTriMesh);

			printf("eps*deps/d0 y at %d = %1.9E \n\n", iMax, pTriMesh->pData[BEGINNING_OF_CENTRAL + iMax].temp.y);
			printf("vertexsum  = %1.10E \n", vertexsum);
			printf("iMax2 = %d max = %1.10E iMin2 = %d min = %1.10E \n", iMax2, maxy2, iMin2, miny2);
			// It chooses 0 coefficient anyway.

			cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
			
			cudaMemcpy(p_tempvec2_host + NMINOR, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
			iMax2 = 0; iMin2 = 0;
			maxy2 = 0.0; miny2 = 0.0;
			f64 epsdeps;
			f64 sum0 = 0.0, sumx = 0.0;
			maxx = 0.0; minx = 0.0;
			f64 epsdepsx;
			long iMaxx = 0, iMinx = 0;

			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				epsdeps = p_tempvec2_host[iMinor].y*p_tempvec2_host[iMinor + NMINOR].y;
				if (epsdeps < miny2) {
					miny2 = epsdeps;
					iMin2 = iMinor;
				};
				if (epsdeps > maxy2) {
					maxy2 = epsdeps;
					iMax2 = iMinor;
				};
				sum0 += epsdeps;

				epsdepsx = p_tempvec2_host[iMinor].x*p_tempvec2_host[iMinor + NMINOR].x;
				if (epsdepsx < minx) {
					minx = epsdepsx;
					iMinx = iMinor;
				};
				if (epsdepsx > maxx) {
					maxx = epsdepsx;
					iMaxx = iMinor;
				};
				sumx += epsdepsx;
			}
			printf("Over ALL epsdeps: iMax2 = %d , = %1.9E ; iMin2 = %d , = %1.9E \n",
				iMax2, maxy2, iMin2, miny2);
			printf("x : iMax %d = %1.9E , iMin %d = %1.9E \n", iMaxx, maxx, iMinx, minx);

			f64_vec2 xy;
			cudaMemcpy(&xy, &(pX_use->p_info[iMin2].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
			printf("pos of iMin2 %1.9E %1.9E \n", xy.x, xy.y);

			printf("sum0 = %1.12E \n", sum0);
			printf("sumx = %1.12E \n", sumx);
			*/
			//Also report max and min of epsilon y over all:

			cudaMemcpy(p_tempvec2_host + NMINOR, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);

			long iMaxy = 0, iMiny = 0;
			f64 epsmaxy = p_tempvec2_host[0 + NMINOR].y, 
				epsmaxx = p_tempvec2_host[0 + NMINOR].x, 
				epsminy = p_tempvec2_host[0 + NMINOR].y, 
				epsminx = p_tempvec2_host[0 + NMINOR].x;
			long iMaxx = 0, iMinx = 0;
			f64 eps1;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				eps1 = p_tempvec2_host[iMinor + NMINOR].y;
				if (eps1 < epsminy) {
					epsminy = eps1;
					iMiny = iMinor;
				};
				if (eps1 > epsmaxy) {
					epsmaxy = eps1;
					iMaxy = iMinor;
				};
				eps1 = p_tempvec2_host[iMinor + NMINOR].x;
				if (eps1 < epsminx) {
					epsminx = eps1;
					iMinx = iMinor;
				};
				if (eps1 > epsmaxx) {
					epsmaxx = eps1;
					iMaxx = iMinor;
				};				
			};
			printf("Epsilon x : overall min %d = %1.9E max %d = %1.9E \n"
				"Epsilon y: overall min %d = %1.9E max %d = %1.9E \n",
				iMinx, epsminx, iMaxx, epsmaxx,
				iMiny, epsminy, iMaxy, epsmaxy);
			
			SetActiveWindow(hwndGraphics);
			ShowWindow(hwndGraphics, SW_HIDE);
			ShowWindow(hwndGraphics, SW_SHOW);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			printf("iMoveType was %d \n", iMoveType);
			printf("done graphs\n\n");

			getch();

		}

		
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		// ===============================================
		// collect eps deps and sum-product matrix
		// ===============================================
						
		//if (iMoveType != 4) {
		
		cudaMemset(p_eps_against_d_eps, 0, sizeof(f64)*REGRESSORS * numTilesMinor);
		cudaMemset(p_sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS* numTilesMinor);
		kernelAccumulateSummandsVisc << <numTilesMinor, threadsPerTileMinor / 4 >> > (
			p_epsilon_xy, // 
			p_epsilon_iz, 
			p_epsilon_ez,
			p_d_epsxy_by_d_beta_i, // f64_vec2
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,

			p_eps_against_d_eps,  // 1x8 for each tile
			p_sum_product_matrix // this is 8x8 for each tile
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands visc2");

		cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_d_eps, sizeof(f64) * REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(f64) * REGRESSORS*REGRESSORS * numTilesMinor, cudaMemcpyDeviceToHost);
		 
		f64 eps_deps[REGRESSORS];
		f64 sum_product_matrix[REGRESSORS*REGRESSORS];
		memset(eps_deps, 0, sizeof(f64) * REGRESSORS);
		memset(sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			for (i = 0; i < REGRESSORS; i++)
				eps_deps[i] -= p_eps_against_d_eps_host[iTile *REGRESSORS + i];

			// Note minus, to get beta already negated.

			for (i = 0; i < REGRESSORS*REGRESSORS; i++)
				sum_product_matrix[i] += p_sum_product_matrix_host[iTile *REGRESSORS*REGRESSORS + i];
		};
		// Try this:
		for (i = 0; i < REGRESSORS; i++) 
			if ((sum_product_matrix[REGRESSORS*i + i] == 0.0) && (eps_deps[i] == 0.0))
				sum_product_matrix[REGRESSORS*i + i] = 1.0;
		

		if ((!GlobalSuppressSuccessVerbosity) || (iIteration >= ITERGRAPH))
		{
			printf("\n");
			for (i = 0; i < REGRESSORS; i++) {
				for (int j = 0; j < REGRESSORS; j++)
					printf("ei %1.9E ", sum_product_matrix[i*REGRESSORS + j]);
				printf(" |  %1.9E \n", eps_deps[i]);
			}
			printf("\n");
		}
		// Note that file 1041-Krylov.pdf claims that simple factorization for LS is an
		// unstable method and that is why the complications of GMRES are needed.

		// now we need the LAPACKE dgesv code to solve the 8x8 linear equation.
		f64 storeRHS[REGRESSORS];
		f64 storemat[REGRESSORS*REGRESSORS];
		memcpy(storeRHS, eps_deps, sizeof(f64)*REGRESSORS);
		memcpy(storemat, sum_product_matrix, sizeof(f64)*REGRESSORS*REGRESSORS);
		 
		Matrix_real matLU;
		matLU.Invoke(REGRESSORS);
		for (i = 0; i < REGRESSORS; i++)
			for (int j = 0; j < REGRESSORS; j++)
				matLU.LU[i][j] = sum_product_matrix[i*REGRESSORS + j];
		matLU.LUdecomp();
		matLU.LUSolve(eps_deps, beta);
		
		SetConsoleTextAttribute(hConsole, 13);
		printf("\nbeta: ");
		for (i = 0; i < REGRESSORS; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n");
		SetConsoleTextAttribute(hConsole, 15);

		char o;
		if ((iIteration > ITERGRAPH) || (bDebug)) {
			printf("reset beta? y/n\n");
			do {
				o = getch();
			} while ((o != 'y') && (o != 'n'));
			if (o == 'y') {
				beta[0] = 0.05;
				beta[1] = 0.05;
				beta[2] = 0.05;  // inaccurate at 1%? yes
				beta[3] = 0.0;    // inaccurate at 0.1%? not so much
				beta[4] = 0.0;
				beta[5] = 0.0;
				beta[6] = 0.0;
				beta[7] = 0.0;
			}

			// If it is inaccurate at 1% then does that mean reassessment at 1% gives accurate estimate?

		}
////
//		beta[0] = 0.0;
//		beta[1] = 0.0;
//		beta[4] = 0.0;
//		beta[5] = 0.0;
//		beta[6] = 0.0;
//		beta[7] = 0.0;
		

	//	beta[0] = -1.0;
	//	beta[1] = -1.0;
	//	beta[2] = -1.0;
		/*

		#ifdef LAPACKE
		lapack_int ipiv[REGRESSORS];
		lapack_int Nrows = REGRESSORS,
		Ncols = REGRESSORS,  // lda
		Nrhscols = 1, // ldb
		Nrhsrows = REGRESSORS, info;

		//	for (i = 0; i < REGRESSORS; i++) {
		//		for (int j = 0; j < REGRESSORS; j++)
		//			printf("%1.8E ", sum_product_matrix[i*REGRESSORS + j]);
		//		printf(" ] [ %1.8E ]\n", eps_deps[i]);
		//	};



		//	printf("LAPACKE_dgesv Results\n");
		// Solve the equations A*X = B
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
		Nrows, 1, sum_product_matrix, Ncols, ipiv, eps_deps, Nrhscols);
		// Check for the exact singularity :

		if (info > 0) {
		printf("The diagonal element of the triangular factor of A,\n");
		printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
		printf("the solution could not be computed.\n");
		printf("press c\n");
		while (getch() != 'c');
		}
		else {
		if (info == 0) {
		memcpy(beta, eps_deps, REGRESSORS * sizeof(f64)); // that's where LAPACKE saves the result apparently.
		};
		}
		#endif
		*/ 
		// Debug:
		//printf("a = dummy1, s = dummy2, d = dummy3; other = continue\n");
		//char o = getch();
		//if ((o == 'a') || (o == 's') || (o == 'd') || (o == 'f') || (o == 'g') || (o == 'h') || (o == 'j') || (o == 'k') ) 
		//{ 
		//	memset(beta,0,sizeof(f64)*REGRESSORS);
		//	if (o == 'a') beta[0] = 100.0;
		//	if (o == 's') beta[1] = 100.0;
		//	if (o == 'd') beta[2] = 100.0;
		//	if (o == 'f') beta[3] = 100.0;
		//	if (o == 'g') beta[4] = 100.0;
		//	if (o == 'h') beta[5] = 100.0;
		//	if (o == 'j') beta[6] = 100.0;
		//	if (o == 'k') beta[7] = 100.0;
		//}
		// 
			 
		// Now beta[8] is the set of coefficients for x
		// Move to the new value: add lc of regressors to proposal vector.
		 
		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

		printf("Iteration %d visc: [ beta ", iIteration);
		for (i = 0; i < REGRESSORS; i++) printf("%1.3E ", beta[i]);
		printf(" ]\n");
			 
		//// Predict change.
		//f64 deps_, epsy;
		//cudaMemcpy(&epsy, &((p_epsilon_xy + CHOSEN)->y), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("eps.y now at %d = %1.12E \n", CHOSEN, epsy);
		//f64 predict = epsy;
		//for (i = 0; i < REGRESSORS; i++) {
		//	cudaMemcpy(&deps_, &(p_d_epsxy_by_d_beta_i[CHOSEN + i*NMINOR].y), sizeof(f64), cudaMemcpyDeviceToHost);
		//	predict += deps_*beta[i];
		//	printf("[%d] + %1.9E * %1.9E = %1.9E  to give %1.12E\n", i, deps_, beta[i], deps_*beta[i], predict);
		//};
		//	
		// DEBUG:
		//	cudaMemcpy(p_tempvec4_host, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxx, p_tempvec4_host[iMaxx].vxy.x,
		//		p_tempvec4_host[iMaxx].vxy.y, p_tempvec4_host[iMaxx].viz, p_tempvec4_host[iMaxx].vez);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxiz, p_tempvec4_host[iMaxiz].vxy.x,
		//		p_tempvec4_host[iMaxiz].vxy.y, p_tempvec4_host[iMaxiz].viz, p_tempvec4_host[iMaxiz].vez);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxez, p_tempvec4_host[iMaxez].vxy.x,
		//		p_tempvec4_host[iMaxez].vxy.y, p_tempvec4_host[iMaxez].viz, p_tempvec4_host[iMaxez].vez);

		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		// Prepare for line search:

		f64 TotalRSS0 = TSS_xy + TSS_iz + TSS_ez;
		f64 dRSS0, dRSS1, TotalRSS1;
		f64 lambda = 1.0;

		// Bear in mind eps_deps[i] was defined with minus involved.
		// Be careful what we are doing here:			
		// we have 4 dimensions of epsilon.
		// However we only care about sum eps.deps which is d/dlambda[RSS].

		// 2. Compute dRSS/dlambda(0) 
		dRSS0 = 0.0;
		for (i = 0; i < REGRESSORS; i++)
			dRSS0 -= 2.0*beta[i] * eps_deps[i];
		// Test to make sure we got it negative:
		if (dRSS0 > 0.0) {
			printf("ERROR: dRSSbydlambda0 %1.9E \n\a", dRSS0);
			
		} else {
		//	printf("Success? dRSSbydlambda0 %1.9E \n", dRSS0);
		};

		cudaMemcpy(p_vie_save, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
		// could do more efficiently no doubt.
		AddLCtoVector4 << <numTilesMinor, threadsPerTileMinor >> >
			(p_vie, p_regressors2, p_regressors_iz, p_regressors_ez, p_stored_move4);		
		Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector4");

		if ((iMoveType == 4) && (iIteration >= ITERGRAPH))
		{
			v4 tempv4;
			for (int iEqn = 0; iEqn < EQNS_TOTAL; iEqn++) {
				cudaMemcpy(&tempv4, &(p_stored_move4[eqnlist[iEqn]]), sizeof(v4), cudaMemcpyDeviceToHost);
				printf("Eqn: %d %d : movedx %1.14E %1.14E %1.14E %1.14E \n",
					iEqn, eqnlist[iEqn], tempv4.vxy.x, tempv4.vxy.y, tempv4.viz, tempv4.vez);
			};
			getch();
		}

		// DEBUG:
		if (0) {
			SetConsoleTextAttribute(hConsole, 12);
			long * longaddress;
			Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
				"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
			cudaMemset(longaddress, 1L, sizeof(long));
			SubroutineComputeDbyDbeta(hsub, p_regressors2, p_regressors_iz, p_regressors_ez, p_vie, pX_use, 0,
				1, 1, 1);
			Call(cudaGetSymbolAddress((void **)(&longaddress), DebugFlag),
				"cudaGetSymbolAddress((void **)(&longaddress), DebugFlag)");
			cudaMemset(longaddress, 0L, sizeof(long));
			SetConsoleTextAttribute(hConsole, 15);
		}
		
		// DEBUG: 
		//	cudaMemcpy(p_tempvec4_host, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxx, p_tempvec4_host[iMaxx].vxy.x,
		//		p_tempvec4_host[iMaxx].vxy.y, p_tempvec4_host[iMaxx].viz, p_tempvec4_host[iMaxx].vez);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxiz, p_tempvec4_host[iMaxiz].vxy.x,
		//		p_tempvec4_host[iMaxiz].vxy.y, p_tempvec4_host[iMaxiz].viz, p_tempvec4_host[iMaxiz].vez);
		//	printf("vie_4 : %d : %1.9E %1.9E %1.9E %1.9E \n", iMaxez, p_tempvec4_host[iMaxez].vxy.x,
		//		p_tempvec4_host[iMaxez].vxy.y, p_tempvec4_host[iMaxez].viz, p_tempvec4_host[iMaxez].vez);

		// ===========================================================================
		// Finally, test whether the new values satisfy 'reasonable' criteria:
				
		/*
		// Debug:
		kernelCreatePredictionsDebug << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,

			p_epsilon_xy,
			p_epsilon_iz,
			p_epsilon_ez,

			p_d_epsxy_by_d_beta_i, // f64_vec2
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,

			p_GradAz,
			p_epsilon_i,
			p_epsilon_e
			);
		Call(cudaThreadSynchronize(), "cudaTS Create predictions");Reused Grad Az here which corrupted it.
		*/
		
		// . Store in a vector, the linear combination of regressors, ie the move unscaled.
		// Inner loop will rescale move to ensure it's a good stab at optimal.
		CreateLC4 << <numTilesMinor, threadsPerTileMinor >> >
			(p_regrlc2, p_regrlc_iz, p_regrlc_ez, p_regressors2, p_regressors_iz, p_regressors_ez);
		Call(cudaThreadSynchronize(), "cudaTS CreateLCv4");

		// DEBUG : 
		if ((iMoveType == 4) && (iIteration >= ITERGRAPH))
		{
			v4 tempv4;
			for (int iEqn = 0; iEqn < EQNS_TOTAL; iEqn++) {
				cudaMemcpy(&(tempv4.vxy), &(p_epsilon_xy[eqnlist[iEqn]]), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				cudaMemcpy(&(tempv4.viz), &(p_epsilon_iz[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&(tempv4.vez), &(p_epsilon_ez[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);
				printf("Eqn: %d %d : epsilon b4: %1.14E %1.14E %1.14E %1.14E \n",
					iEqn, eqnlist[iEqn], tempv4.vxy.x, tempv4.vxy.y, tempv4.viz, tempv4.vez);
			};
		}
		// Idea: until we cross 0, try some kind of secant or extend by 1.6* .
		// Once we cross 0, start on vWijngaarden-Dekker-Brent.

		bool b_vWDB = false;
		bool mflag = true;
		f64 a_Brent = 0.0, b_Brent, c_Brent, d_Brent = 0.0;
		f64 dRSSa = dRSS0, dRSSb, dRSSc, dRSStemp;
		bool bool_c_equals_a_or_b;

		// . Structure of what we need to do:
		// . After evaluating f(s)
		// .. determine new relevant interval and new s for evaluation.
		// 

		cudaMemcpy(p_d_epsxy_by_d_beta_e, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToDevice);

		bContinue = false;
		bool condition = true;
		bool bGiveUp = false;
		while (condition) {

			printf("Loop: lambda = %1.9E \n", lambda);

			// 1. Find RSS(lambda) and if not accepted, dRSS(lambda); call this dRSS1.

			cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);
			cudaMemset(NT_addition_tri_d2, 0, sizeof(NTrates)*NUMVERTICES * 6);
			cudaMemset(NT_addition_rates_d_2, 0, sizeof(NTrates)*NUMVERTICES);
			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_vie,	pX_use->p_v_n, 
				pX_use->p_izTri_vert,	pX_use->p_szPBCtri_vert,	pX_use->p_izNeigh_TriMinor,	pX_use->p_szPBC_triminor,
				p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
				p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
				pX_use->p_B,
				p_MAR_ion2, // accumulates
				NT_addition_rates_d_2, NT_addition_tri_d2,
				1, m_ion_, 1.0 / m_ion_);
			Call(cudaThreadSynchronize(), "cudaTS Create visc flux ion");
			kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_vie,	pX_use->p_v_n, // not used
				pX_use->p_izTri_vert,		pX_use->p_szPBCtri_vert,	pX_use->p_izNeigh_TriMinor,	pX_use->p_szPBC_triminor,
				p_ita_e, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
				p_nu_e, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
				pX_use->p_B,
				p_MAR_elec2, // accumulates
				NT_addition_rates_d_2, NT_addition_tri_d2,
				2, m_e_, 1.0 / m_e_);
			Call(cudaThreadSynchronize(), "cudaTS Create visc flux elec");

			// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
			kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				// x = -eps/coeffself
				hsub,   pX_use->p_info,
				p_vie,	p_vie_k,
				p_MAR_ion2, p_MAR_elec2,
				pX_use->p_n_minor,	pX_use->p_AreaMinor,
				p_epsilon_xy,	p_epsilon_iz,	p_epsilon_ez,
				p_bFailed,
				p_Selectflag
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
			if (i < numTilesMinor) bContinue = true;

			// Collect L2eps and R^2:
			RSS_xy = 0.0; RSS_iz = 0.0; RSS_ez = 0.0;
			kernelAccumulateSumOfSquares2vec << <numTilesMinor, threadsPerTileMinor >> >(
				p_epsilon_xy, p_SS);
			cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
				RSS_xy += p_SS_host[iTile];
			L2eps_xy = sqrt(RSS_xy / (real)NMINOR);
			 
			kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
				p_epsilon_iz, p_SS);
			Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
			cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
				RSS_iz += p_SS_host[iTile];
			L2eps_iz = sqrt(RSS_iz / (real)NMINOR);

			kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
				p_epsilon_ez, p_SS);
			Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
			cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
				RSS_ez += p_SS_host[iTile];
			L2eps_ez = sqrt(RSS_ez / (real)NMINOR);
			
			// What is R^2? We should like to report it.
			if (TSS_xy > 0.0) {
				Rsquared_xy = (TSS_xy - RSS_xy) / TSS_xy;
			} else {
				Rsquared_xy = 1.0;
			};
			if (TSS_iz > 0.0) {
				Rsquared_iz = (TSS_iz - RSS_iz) / TSS_iz;
			} else {
				Rsquared_iz = 1.0;
			}
			if (TSS_ez > 0.0) {
				Rsquared_ez = (TSS_ez - RSS_ez) / TSS_ez;
			} else {
				Rsquared_ez = 1.0;
			}
			Rsquared_xy *= 100.0; Rsquared_iz *= 100.0; Rsquared_ez *= 100.0;

			SetConsoleTextAttribute(hConsole, 14);
			printf("L2eps xy %1.8E iz %1.8E ez %1.8E TOTALRSS %1.10E \nR^2 %2.3f%% %2.3f%% %2.3f%% bCont: %d\n",
				L2eps_xy, L2eps_iz, L2eps_ez,
				RSS_xy + RSS_iz + RSS_ez,
				Rsquared_xy, Rsquared_iz, Rsquared_ez, (bContinue ? 1 : 0));
			SetConsoleTextAttribute(hConsole, 15);
		

			if ((iIteration > ITERGRAPH) || (bDebug)) {

				if (iMoveType == 4)
				{
					v4 tempv4;
					for (int iEqn = 0; iEqn < EQNS_TOTAL; iEqn++) {
						cudaMemcpy(&(tempv4.vxy), &(p_epsilon_xy[eqnlist[iEqn]]), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
						cudaMemcpy(&(tempv4.viz), &(p_epsilon_iz[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);
						cudaMemcpy(&(tempv4.vez), &(p_epsilon_ez[eqnlist[iEqn]]), sizeof(f64), cudaMemcpyDeviceToHost);
						printf("Eqn: %d %d : epsilon new %1.14E %1.14E %1.14E %1.14E \n",
							iEqn, eqnlist[iEqn], tempv4.vxy.x, tempv4.vxy.y, tempv4.viz, tempv4.vez);
					};
				}
				structural info;
				cudaMemcpy(&info, pX_use->p_info + CHOSEN, sizeof(structural), cudaMemcpyDeviceToHost);
				printf("Move done!");

				//GlobalSuppressSuccessVerbosity = false;
				GlobalCutaway = false;

				// 1. Represent epsilon xy as xy velocity
				/*
				cudaMemcpy(p_temphost1, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_temphost2, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
				v4 tempv4;
				for (i = 0; i < EQNS_TOTAL; i++)
				{
					cudaMemcpy(&tempv4, p_vie + eqnlist[i], sizeof(v4), cudaMemcpyDeviceToHost);

					printf("Eqn %d : %d : epsilon_iz %1.9E ez %1.9E x %1.9E y %1.9E %1.14E %1.14E %1.14E %1.8E\n",
						i, eqnlist[i], p_temphost1[eqnlist[i]], p_temphost2[eqnlist[i]], p_tempvec2_host[eqnlist[i]].x, p_tempvec2_host[eqnlist[i]].y,
						tempv4.vxy.x, tempv4.vxy.y, tempv4.viz, tempv4.vez
						);
				}
				kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux
					<< <numTriTiles, threadsPerTileMinor >> > (
						pX_use->p_info,
						p_vie,
						pX_use->p_v_n, // not used	
						ionmomflux_eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.				
						p_selectflag, // whether it's in the smoosh to smash
						p_equation_index, // each one assigned an equation index				
						pX_use->p_izTri_vert, pX_use->p_szPBCtri_vert, pX_use->p_izNeigh_TriMinor, pX_use->p_szPBC_triminor,
						p_ita_i, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
						p_nu_i, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
						pX_use->p_B,
						1, m_i_, 1.0 / m_i_
						);
				Call(cudaThreadSynchronize(), "cudaTS kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux");

				printf("press t\n");
				while (getch() != 't');*/

				SetActiveWindow(hwndGraphics);
				cudaMemcpy(p_tempvec2_host, p_epsilon_xy + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);

				long iVertex;
				char buffer[256];
				Vertex * pVertex = pTriMesh->X;
				plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				f64 maxy = 0.0;
				long iMax = -1;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_tempvec2_host[iVertex].x;
					pdata->temp.y = p_tempvec2_host[iVertex].y;

					if (pdata->temp.y*pdata->temp.y > maxy*maxy) {
						maxy = pdata->temp.y;
						iMax = iVertex;
					};
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "epsilon_y iter %d", iIteration);
				Graph[0].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
					AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);
				sprintf(buffer, "epsilon_x iter %d", iIteration);
				Graph[2].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
					false,
					GRAPH_EPSILON, pTriMesh);
				printf("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
					"iMax = vertex %d eps.y = %1.10E \n"
					"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n", iMax, maxy);

				f64_vec2 tempvec2;
				cudaMemcpy(&tempvec2, &(pX_use->p_info[iMax + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				printf("position %1.9E %1.9E \n", tempvec2.x, tempvec2.y);

				printf("iRing = %d\n", iRing[iMax + BEGINNING_OF_CENTRAL]);

				cudaMemcpy(p_inthost, p_Selectflag + BEGINNING_OF_CENTRAL, sizeof(int)*NUMVERTICES, cudaMemcpyDeviceToHost);

				printf("p_Selectflag = %d\n", p_inthost[iMax + BEGINNING_OF_CENTRAL]);

				cudaMemcpy(p_tempvec2_host, p_regrlc2 + BEGINNING_OF_CENTRAL, sizeof(f64_vec2)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_tempvec2_host[iVertex].x;
					pdata->temp.y = p_tempvec2_host[iVertex].y;
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "regrlc2 %d", iIteration);
				Graph[1].DrawSurface(buffer,
					VELOCITY_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					VELOCITY_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.x)),
					false,
					GRAPH_EPSILON, pTriMesh);

				cudaMemcpy(p_temphost1, p_regrlc_iz + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				int ii;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost1[iVertex];
					ii = p_inthost[iVertex];
					pdata->temp.y = ((f64)ii)*2.0e-12;
					if (iMoveType == 4)
						pdata->temp.y = ((f64)iRing[BEGINNING_OF_CENTRAL + iVertex])*1.0e-12;
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "regr_iz %d", iIteration);
				Graph[3].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);
				
				cudaMemcpy(p_temphost1, p_regrlc_ez + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost1[iVertex];
					ii = p_inthost[iVertex];
					pdata->temp.y = ((f64)ii)*2.0e-12;
					if (iMoveType == 4)
						pdata->temp.y = ((f64)iRing[BEGINNING_OF_CENTRAL + iVertex])*1.0e-12;
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "regr_ez %d", iIteration);
				Graph[4].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);
								
				cudaMemcpy(p_temphost2, p_epsilon_ez + BEGINNING_OF_CENTRAL, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
				pVertex = pTriMesh->X;
				pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					pdata->temp.x = p_temphost2[iVertex];
					ii = p_inthost[iVertex];
					pdata->temp.y = ((f64)ii)*2.0e-12;
					if (iMoveType == 4)
						pdata->temp.y = ((f64)iRing[BEGINNING_OF_CENTRAL + iVertex])*1.0e-12;
					++pVertex;
					++pdata;
				}
				sprintf(buffer, "eps_ez %d", iIteration);
				Graph[5].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);

				/*
				f64 vertexsum = 0.0;
				f64 maxy2 = 0.0, miny2 = 0.0;
				long iMax2 = 0, iMin2 = 0;
				for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
				{
					// epsilon over depsilon

					// Try this: ROC of eps^2 in location. = 2*eps*deps

					pdata->temp.x 

					pdata->temp.x *= p_tempvec2_host[iVertex].x;
					pdata->temp.y *= p_tempvec2_host[iVertex].y;
					vertexsum += pdata->temp.y;

					if (pdata->temp.y*pdata->temp.y > maxy2*maxy2) {
						maxy2 = pdata->temp.y;
						iMax2 = iVertex;
					};

					if (pdata->temp.y*pdata->temp.y < miny2*miny2) {
						miny2 = pdata->temp.y;
						iMin2 = iVertex;
					};

					pdata->temp.y = ((f64)p_inthost[iVertex])*2.0e-12;

					//if (fabs(pdata->temp.x) < 1.0e-6) {
					//	// doesn't matter
					//	pdata->temp.x = 1.0;
					//} else {
					//	
					//	// going to take deps/eps

					//	if (p_tempvec2_host[iVertex].x == 0.0) {
					//		// epsilon = 0; but we could be changing it
					//	}
					//	
					//	
					//	if (fabs(p_tempvec2_host[iVertex].x) < 0.001*fabs(pdata->temp.x)) {
					//		if (p_tempvec2_host[iVertex].x*pdata->temp.x < 0.0) {
					//			pdata->temp.x = -1000.0;
					//		} else {
					//			pdata->temp.x = 1000.0;
					//		}
					//	}
					//	else {
					//		pdata->temp.x /= p_tempvec2_host[iVertex].x;
					//	};
					//};
					//if (fabs(pdata->temp.y) < 1.0e-6) {
					//	pdata->temp.y = 1.0;
					//} else {
					//	if (fabs(p_tempvec2_host[iVertex].y) < 0.001*fabs()
					//	pdata->temp.y /= p_tempvec2_host[iVertex].y;
					//}
					++pVertex;
					++pdata;
				}

				sprintf(buffer, "eps.x*deps.x iter %d", iIteration);
				Graph[4].DrawSurface(buffer,
					DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
					SEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
					false,
					GRAPH_EPSILON, pTriMesh);

				//			sprintf(buffer, "eps.y*deps.y iter %d", iIteration);
				//			Graph[5].DrawSurface(buffer,
				//				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.y)),
				//				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				//				false,
				//				GRAPH_EPSILON, pTriMesh);

				printf("eps*deps/d0 y at %d = %1.9E \n\n", iMax, pTriMesh->pData[BEGINNING_OF_CENTRAL + iMax].temp.y);
				printf("vertexsum  = %1.10E \n", vertexsum);
				printf("iMax2 = %d max = %1.10E iMin2 = %d min = %1.10E \n", iMax2, maxy2, iMin2, miny2);
				// It chooses 0 coefficient anyway.

				cudaMemcpy(p_tempvec2_host, p_d_epsxy_by_d_beta_i, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_tempvec2_host + NMINOR, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);

				iMax2 = 0; iMin2 = 0;
				maxy2 = 0.0; miny2 = 0.0;
				f64 epsdeps;
				f64 sum0 = 0.0, sumx = 0.0, maxx = 0.0, minx = 0.0;
				f64 epsdepsx;
				long iMaxx = 0, iMinx = 0;

				for (iMinor = 0; iMinor < NMINOR; iMinor++)
				{
					epsdeps = p_tempvec2_host[iMinor].y*p_tempvec2_host[iMinor + NMINOR].y;
					if (epsdeps < miny2) {
						miny2 = epsdeps;
						iMin2 = iMinor;
					};
					if (epsdeps > maxy2) {
						maxy2 = epsdeps;
						iMax2 = iMinor;
					};
					sum0 += epsdeps;

					epsdepsx = p_tempvec2_host[iMinor].x*p_tempvec2_host[iMinor + NMINOR].x;
					if (epsdepsx < minx) {
						minx = epsdepsx;
						iMinx = iMinor;
					};
					if (epsdepsx > maxx) {
						maxx = epsdepsx;
						iMaxx = iMinor;
					};
					sumx += epsdepsx;
				}
				printf("Over ALL epsdeps: iMax2 = %d , = %1.9E ; iMin2 = %d , = %1.9E \n",
					iMax2, maxy2, iMin2, miny2);
				printf("x : iMax %d = %1.9E , iMin %d = %1.9E \n", iMaxx, maxx, iMinx, minx);

				f64_vec2 xy;
				cudaMemcpy(&xy, &(pX_use->p_info[iMin2].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
				printf("pos of iMin2 %1.9E %1.9E \n", xy.x, xy.y);

				printf("sum0 = %1.12E \n", sum0);
				printf("sumx = %1.12E \n", sumx);
				*/
				//Also report max and min of epsilon y over all:

				cudaMemcpy(p_tempvec2_host + NMINOR, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
				long iMaxy = 0, iMiny = 0, iMaxx = 0, iMinx = 0;
				f64 epsmaxy = p_tempvec2_host[0 + NMINOR].y,
					epsmaxx = p_tempvec2_host[0 + NMINOR].x,
					epsminy = p_tempvec2_host[0 + NMINOR].y,
					epsminx = p_tempvec2_host[0 + NMINOR].x;
				f64 eps1;
				for (iMinor = 0; iMinor < NMINOR; iMinor++)
				{
					eps1 = p_tempvec2_host[iMinor + NMINOR].y;
					if (eps1 < epsminy) {
						epsminy = eps1;
						iMiny = iMinor;
					};
					if (eps1 > epsmaxy) {
						epsmaxy = eps1;
						iMaxy = iMinor;
					};

					eps1 = p_tempvec2_host[iMinor + NMINOR].x;
					if (eps1 < epsminx) {
						epsminx = eps1;
						iMinx = iMinor;
					};
					if (eps1 > epsmaxx) {
						epsmaxx = eps1;
						iMaxx = iMinor;
					};
				};
				printf("Epsilon x : overall min %d = %1.9E max %d = %1.9E \n"
					"Epsilon y: overall min %d = %1.9E max %d = %1.9E \n",
					iMinx, epsminx, iMaxx, epsmaxx,
					iMiny, epsminy, iMaxy, epsmaxy);

				SetActiveWindow(hwndGraphics);
				ShowWindow(hwndGraphics, SW_HIDE);
				ShowWindow(hwndGraphics, SW_SHOW);
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

				printf("iMoveType was %d \n", iMoveType);
				printf("done graphs during loop\n\n");

				getch();

			}



			if (bGiveUp) {
				condition = false;
				//printf("condition == false and go to skip.\n");
				goto skip;
			}
			TotalRSS1 = RSS_xy + RSS_iz + RSS_ez;
			if ((TotalRSS1 < 0.6 * TotalRSS0) || (bContinue == false))  {
			
				// Accept move as is.
				condition = false;
			} else {

				printf("not accepted on RSS alone. TotalRSS01 %1.9E %1.9E\n", TotalRSS0, TotalRSS1);

				// We already got epsilon, RSS1.
				// We already accepted move if RSS was down * 0.81
				// 6. Compute deps/dlambda_1 & Accumulate eps*deps to get dRSS1
				// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				SubroutineComputeDbyDbeta(hsub, p_regrlc2, p_regrlc_iz, p_regrlc_ez, p_vie, pX_use, 0,
					1, 1, 1);

				// stores in same deps as before:
				cudaMemset(p_eps_against_deps2, 0, sizeof(f64) * numTilesMinor);
				kernelAccumulateSummandsProduct << <numTilesMinor, threadsPerTileMinor >> > (
					p_epsilon_xy, // 
					p_epsilon_iz,
					p_epsilon_ez,
					p_d_epsxy_by_d_beta_i, // f64_vec2
					p_d_eps_iz_by_d_beta_i,
					p_d_eps_ez_by_d_beta_i,
					p_eps_against_deps2  // 1x1 for each tile
					);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands visc2");
				cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_deps2, sizeof(f64) * numTilesMinor, cudaMemcpyDeviceToHost);
				f64 eps__deps = 0.0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
					eps__deps += p_eps_against_d_eps_host[iTile];
				dRSS1 = eps__deps*2.0;

				//===========================================================================

				// 7. Exit lambda loop if dRSS/dlambda is sufficiently reduced in magnitude

				if (
					((TotalRSS1 < TotalRSS0) && (fabs(dRSS1) < 0.1*fabs(dRSS0)))
					||
					((b_vWDB) && (fabs(a_Brent - b_Brent) < 0.00001) && (fabs(dRSS1) < fabs(dRSS0)))
					) {
					// good enough
					printf("lambda %1.9E accepted: dRSS0 %1.6E dRSS1 %1.6E\n", lambda, dRSS0, dRSS1);
					condition = false;
					goto skip;
				};

				// DEBUG:
				printf("RSS0 %1.10E RSS1 %1.10E dRSS0 %1.10E dRSS1 %1.10E \n",
					TotalRSS0, TotalRSS1, dRSS0, dRSS1);

				if (b_vWDB)
				{
					d_Brent = c_Brent;// (d is assigned for the first time here; it won't be used above on the first iteration because mflag is set)
					c_Brent = b_Brent;
					dRSSc = dRSSb;
					bool_c_equals_a_or_b = true;
					if (dRSSa*dRSS1 < 0.0)
					{
						b_Brent = lambda;
						dRSSb = dRSS1;
						bool_c_equals_a_or_b = false; 
					} else {
						a_Brent = lambda;
						dRSSa = dRSS1;
					};

					if (fabs(dRSSa) < fabs(dRSSb))
					{
						// swap(a, b))
						// careful.
						dRSStemp = dRSSb;
						tempf64 = b_Brent;
						b_Brent = a_Brent;
						dRSSb = dRSSa;
						a_Brent = tempf64;
						dRSSa = dRSStemp;
					};
						
					// Having created the new interval create s:

					if (bool_c_equals_a_or_b)
					{
						// secant
						lambda = b_Brent - dRSSb*(b_Brent - a_Brent) / (dRSSb - dRSSa);
					} else {
						// inverse quadratic:
						f64 denom = (dRSSa - dRSSb)*(dRSSa - dRSSc)*(dRSSb - dRSSc);
						lambda = (a_Brent*dRSSb*dRSSc*(dRSSb - dRSSc)
							+b_Brent*dRSSa*dRSSc*(dRSSc - dRSSa)
							+c_Brent*dRSSa*dRSSb*(dRSSa - dRSSb)
							)/ denom;
					};
					f64 delta = 1.0e-14*(fabs(b_Brent) + 1.0); // machine precision tolerance, see p.51 of
					// "Algorithms for Minimization Without Derivatives" by Richard P. Brent
					if ((!within(lambda,0.75*a_Brent+0.25*b_Brent,b_Brent) ) ||
						((mflag == true) && ((fabs(lambda-b_Brent) >=0.5*fabs(b_Brent-c_Brent)) || (fabs(b_Brent-c_Brent) < delta))) ||
						((mflag == false) && ((fabs(lambda-b_Brent) >= 0.5*fabs(c_Brent-d_Brent)) || (fabs(c_Brent-d_Brent) < delta)))
						)
					{
						// when things are getting close to within delta we always bisect
						// delta is probably something like a machine precision, in concept
						lambda = 0.5*(a_Brent + b_Brent);
						mflag = true;
					} else {
						mflag = false;
					};					

				} else {

					// Not Brent's method yet
					// (b_vWDB == false)
					if ((dRSS1 > 0.0)) {
						// We now bracketed the root and can start in on Brent's method.
						// Brent's method

						b_vWDB = true;
						// a is the more distant point
						a_Brent = 0.0;
						b_Brent = lambda;
						dRSSb = dRSS1;
						dRSSa = dRSS0;
						c_Brent = a_Brent;
						dRSSc = dRSSa;
						bool_c_equals_a_or_b = true;

						// c is somehow lagged to a and b -- accepting s is what triggers cnot equal a or b
						// Run secant method the first time:

						lambda = b_Brent - dRSSb*(b_Brent - a_Brent) / (dRSSb - dRSSa);
						f64 delta = 1.0e-14*(fabs(b_Brent) + 1.0); 
						// machine precision tolerance, see p.51 of R.P.Brent (1973)
																   
						if ((!within(lambda, 0.75*a_Brent + 0.25*b_Brent, b_Brent)) ||
							((fabs(lambda - b_Brent) >= 0.5*fabs(b_Brent - c_Brent)) || (fabs(b_Brent - c_Brent) < delta))
							)
						{
							lambda = 0.5*(a_Brent + b_Brent);
							mflag = true;
						} else {
							mflag = false;
						};
					} else {
						// Not bracketed the root yet.

						if (dRSS1 > dRSS0) {
							// Linearly extrapolate;
							// aim slightly beyond where we think the root is.
							lambda = lambda + (-dRSS0 / (dRSS1 - dRSS0))*lambda*1.01;
						} else {
							// Just multiply lambda by 1.6
							lambda *= 1.6;
						}
					}
				};
				
				if (condition) {
					// Now scale move per the new lambda ???!??!?!					
					// Create v = v_prev + lambda*move

					kernelCreate_v4 << <numTilesMinor, threadsPerTileMinor >> >
						(p_vie, p_vie_save, lambda, p_regrlc2, p_regrlc_iz, p_regrlc_ez);
					Call(cudaThreadSynchronize(), "cudaTS kernelCreate_v4");

				};
				
			}; // whether accept

		skip:

			if ((lambda < 0.1) && (iMoveType == 4) && (condition == false)) 
			{

				// If we think we are accepted lambda < 0.1 then think again.

				printf("((lambda < 0.1) && (iMoveType == 4)) so restore coeffs to {1/4,1/4,1/4,0,0,0,0,0}\n");
				// Go again:
				beta[0] = 0.25;
				beta[1] = 0.25;
				beta[2] = 0.25;
				beta[3] = 0.0;
				beta[4] = 0.0;
				beta[5] = 0.0;
				beta[6] = 0.0;
				beta[7] = 0.0;
				cudaMemcpy(beta_n_c, beta, sizeof(f64)*REGRESSORS, cudaMemcpyHostToDevice);

				cudaMemcpy(p_vie, p_vie_save, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
				AddLCtoVector4 << <numTilesMinor, threadsPerTileMinor >> >
					(p_vie, p_regressors2, p_regressors_iz, p_regressors_ez, p_stored_move4);
				Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector4");

				lambda = 1.0;
				bGiveUp = true;
				condition = true;
				// continue to one more loop to calculate epsilon.
			}; // end: first test for whether move was accepted due to decrease in RSS.
		}; // while (condition)
	
		TSS_xy = RSS_xy; TSS_iz = RSS_iz; TSS_ez = RSS_ez;
		oldRSS = RSS_xy + RSS_iz + RSS_ez;
		
		// Note that we do need RSS determined at the end of an iIteration loop.

		// Criterion may need some magic.		
		// yes it happens. L2 is not below precision either.
		
		//// Debug: find maximum epsilons:		
		//cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_xy : maxio %1.8E at %d\n", p_tempvec2_host[iMaxx].modulus(), iMaxx);
		//cudaMemcpy(p_temphost6, p_epsilon_iz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_iz : maxio %1.8E at %d\n", p_temphost6[iMaxiz], iMaxiz);
		//cudaMemcpy(p_temphost6, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//printf("eps_ez : maxio %1.8E at %d\n", p_temphost6[iMaxez], iMaxez);
		iIteration++; 
	};
	 
labelBudgerigar2:

	// Save move for next time:
	bHistoryVisc = true;
	//SubtractVector4 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move4, p_vie, p_vie_k);
	//Call(cudaThreadSynchronize(), "cudaTS subtract vector 4");
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	//Yep -- don't do this here. 
	// "vie_k" is modified vk, it's forward move in the forward region.
	
	
	// Important.
	// Maybe let it be bundled up as p_stored_move4.
	//getch();
	
	/*
	structural info, info2;
	v4 v4temp, v4temp2;
	cudaMemcpy(&info2, &(pX_use->p_info[VERTCHOSEN+BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
	cudaMemcpy(&v4temp, &(p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(v4), cudaMemcpyDeviceToHost);
	cudaMemcpy(&v4temp2, &(p_vie_k[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(v4), cudaMemcpyDeviceToHost);
	printf("%d %d v4 %1.11E %1.11E %1.11E %1.11E pos %1.11E %1.11E vk %1.11E %1.11E %1.11E %1.11E\n\n",
		VERTCHOSEN, VERTCHOSEN + BEGINNING_OF_CENTRAL, v4temp.vxy.x, v4temp.vxy.y, v4temp.viz, v4temp.vez, info2.pos.x, info2.pos.y,
		v4temp2.vxy.x, v4temp2.vxy.y, v4temp2.viz, v4temp2.vez);

	long izTri[MAXNEIGH];
	long izNeigh[MAXNEIGH];
	cudaMemcpy(izTri, &(pX_use->p_izTri_vert[VERTCHOSEN*MAXNEIGH]), sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
	cudaMemcpy(izNeigh, &(pX_use->p_izNeigh_vert[VERTCHOSEN*MAXNEIGH]), sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
	cudaMemcpy(&info, &(pX_use->p_info[VERTCHOSEN+BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
	for (i = 0; i < info.neigh_len; i++)
	{
		long iTri = izTri[i];
		cudaMemcpy(&info2, &(pX_use->p_info[iTri]), sizeof(structural), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v4temp, &(p_vie[iTri]), sizeof(v4), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v4temp2, &(p_vie_k[iTri]), sizeof(v4), cudaMemcpyDeviceToHost);
		printf("%d %d v4 %1.11E %1.11E %1.11E %1.11E pos %1.11E %1.11E vk %1.11E %1.11E %1.11E %1.11E\n",
			i, iTri, v4temp.vxy.x, v4temp.vxy.y, v4temp.viz, v4temp.vez, info2.pos.x, info2.pos.y, 
			v4temp2.vxy.x, v4temp2.vxy.y, v4temp2.viz, v4temp2.vez);
	};
	printf("\n");
	for (i = 0; i < info.neigh_len; i++)
	{
		long iNeigh = izNeigh[i];
		cudaMemcpy(&info2, &(pX_use->p_info[iNeigh+BEGINNING_OF_CENTRAL]), sizeof(structural), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v4temp, &(p_vie[iNeigh+BEGINNING_OF_CENTRAL]), sizeof(v4), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v4temp2, &(p_vie_k[iNeigh + BEGINNING_OF_CENTRAL]), sizeof(v4), cudaMemcpyDeviceToHost);
		printf("%d v %d v4 %1.11E %1.11E %1.11E %1.11E pos %1.11E %1.11E vk %1.11E %1.11E %1.11E %1.11E\n",
			i, iNeigh, v4temp.vxy.x, v4temp.vxy.y, v4temp.viz, v4temp.vez, info2.pos.x, info2.pos.y,
			v4temp2.vxy.x, v4temp2.vxy.y, v4temp2.viz, v4temp2.vez);
	};
	printf("\n\npress t\n");

	char o;
	do {
		o = getch();
	} while (o != 't');
	*/
	FILE * fp = fopen("iterations.txt", "a");
	fprintf(fp, "RBR8LSVisc iIter = %d \n", iIteration);
	fclose(fp);
}

 
#include "kernel.cu"
#include "little_kernels.cu"

// There must be a better way.
