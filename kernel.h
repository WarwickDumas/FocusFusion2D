#pragma once
#include "cuda_struct.h"

//__global__ void kernelCalculateOverallVelocitiesVertices(
//	structural * __restrict__ p_info_major,
//	v4 * __restrict__ p_vie_major,
//	f64_vec3 * __restrict__ p_v_n_major,
//	nvals * __restrict__ p_n_major,
//	f64_vec2 * __restrict__ p_v_overall_major);

__global__ void kernelCollect_Up_3x_HeatIntoVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	LONG3 * __restrict__ p_tri_corner_index, // to see which corner we are.

	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri1,
	NTrates * __restrict__ NT_addition_tri2,
	NTrates * __restrict__ NT_addition_tri3
);

__global__ void kernelSubtractNiTiCheckNeg
(NTrates * __restrict__ NTrates_init, NTrates * __restrict__ NTrates_final, 
	f64 * __restrict__ p_diff, nvals * __restrict__ p_n, f64 * __restrict__ p_AreaMajor);


__global__ void kernelFindNegative_dTi
(
	NTrates * NTrate,
	NTrates * NTtri,
	long * pl);


__global__ void kernelSimpleAverage
(f64 * __restrict__ p_update,
	f64 * __restrict__ p_avg);

__global__ void kernelPutative_v_from_matrix(
	f64 * __restrict__ p_ita, // if 0 can avoid loading regrs
	v4 * __restrict__ p_vie_2,
	v4 * __restrict__ p_vie_,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez);

__global__ void DivideNiTiDifference_by_N(
	NTrates * __restrict__ NT_addition_rates_initial,
	NTrates * __restrict__ NT_addition_rates_final,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_dTbydt);

__global__ void kernelSetPressureFlag(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	bool * __restrict__ bz_pressureflag
);
__global__ void kernelAddNT(NTrates * __restrict__ p_update, NTrates * __restrict__ p_addition);

__global__ void kernelAccumulateSummands_4x4(
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	// outputs:
	f64 * __restrict__ p_sum_eps_depsbydbeta_x4,
	f64 * __restrict__ p_sum_depsbydbeta__4x4
);

__global__ void kernelAdd2(
	f64 * __restrict__ p_result,
	f64 * __restrict__ p_src,
	f64 const beta_,
	f64 * __restrict__ p_addition
);

__global__ void kernelAddtoT_lc___(
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_T_src,
	f64 * __restrict__ p_addition,
	int const howmany,
	f64 const multiplier
);

__global__ void AddLC(
	f64 * __restrict__ p_result,
	f64 * __restrict__ p_summand1,
	f64 const betaa,
	f64 * __restrict__ p_summand2);

__global__ void kernelCalculate_kappa_nu_vertices(
	structural * __restrict__ p_info_major, // NB: MAJOR
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,

	f64 * __restrict__ p_kappa_n_major,
	f64 * __restrict__ p_kappa_i_major,
	f64 * __restrict__ p_kappa_e_major,
	f64 * __restrict__ p_nu_i_major,
	f64 * __restrict__ p_nu_e_major // CHECK DIMENSIONS IF USING BIG ARRAY
);

__global__ void AddLCtoT(f64 * __restrict__ p_T_,
	f64 * __restrict__ p_T_use,
	f64 const bet, f64 * __restrict__ p_add);

__global__ void kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_T,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa,
	f64 * __restrict__ p_nu,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,
	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,
	int species);

__global__ void kernelUnpackWithMask(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock
);

__global__ void VectorAddMultiple_masked(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock,
	bool const bUseMask);

__global__ void kernelAccumulateSummands7(
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	// outputs:
	f64 * __restrict__ p_sum_eps_depsbydbeta_x8,
	f64 * __restrict__ p_sum_depsbydbeta__8x8);

__global__ void kernelMultiplyVector(
	f64 * __restrict__ p_multiply,
	f64 const factor);

__global__ void kernelCreateEpsilonHeat_1species
(
	f64 const h_sub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_T,
	f64 * __restrict__ p_Tk, // T_k for substep
	NTrates * __restrict__ p_NTrates_diffusive,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	f64 * __restrict__ p__epsilon,
	bool * __restrict__ p_bFailedTest,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock,
	bool bUseMask,
	int species
);

__global__ void kernelVolleyRegressors(
	f64 * __restrict__ p_regress,
	long const Length,
	char * __restrict__ p_iVolley
);

__global__ void VectorCompareMax(
	f64 * __restrict__ p_comp1,
	f64 * __restrict__ p_comp2,
	long * __restrict__ p_iWhich,
	f64 * __restrict__ p_max
);

__global__ void kernelAdd_to_T_lc(
	f64 * __restrict__ p__T,
	f64 const beta1,
	f64 * __restrict__ p_add1,
	f64 const beta2,
	f64 * __restrict__ p_add2
);

__global__ void kernelHeat_1species_geometric_coeffself(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 const h_use,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies,

	f64 * __restrict__ p_effectself // hmmm
);

__global__ void kernelCreateEpsilonAndJacobi_Heat_1species
(
	f64 const h_sub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_T,
	f64 * p_Tk, // T_k for substep
	NTrates * __restrict__ p_NTrates_diffusive,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	f64 * __restrict__ p__coeffself,
	f64 * __restrict__ p__epsilon,
	f64 * __restrict__ p__Jacobi,
	bool * __restrict__ p_bFailedTest,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock,
	bool bUseMask,
	int species,
	bool bIncorporateEps
);

__global__ void kernelAddtoT_lc(
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_addition,
	int const howmany
);

__global__ void kernelMultiply_Get_Jacobi_Visc(
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_tens3 * __restrict__ p_Matrix_i,
	f64_tens3 * __restrict__ p_Matrix_e,
	f64_vec3 * __restrict__ p_Jacobi_ion,
	f64_vec3 * __restrict__ p_Jacobi_elec
);

__global__ void kernelCalc_Matrices_for_Jacobi_Viscosity(
	f64 const hsub,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_tens3 * __restrict__ p_matrix_i,
	f64_tens3 * __restrict__ p_matrix_e
);

__global__ void kernelDivide(
	f64 * __restrict__ p_regress,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_effectself
);

__global__ void
kernelAddSolution(
	v4 * __restrict__ p_v,
	bool * __restrict__ p_bSelected,
	short * __restrict__ p_eqn_index,
	f64 * __restrict__ p_solution);

__global__ void
kernelZeroWithinRings(
	f64 * __restrict__ p_operand,
	bool * __restrict__ p_not
);

__global__ void
kernelZeroWithinRings2(
	f64_vec2 * __restrict__ p_operand,
	bool * __restrict__ p_not
);

__global__ void
kernelPopulateRegressors_from_iRing_RHS
(f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	f64_vec2 * __restrict__ p_regr_2_1,
	f64 * __restrict__ p_regr_iz_1,
	f64 * __restrict__ p_regr_ez_1,
	f64_vec2 * __restrict__ p_regr_2_2,
	f64 * __restrict__ p_regr_iz_2,
	f64 * __restrict__ p_regr_ez_2,
	bool * __restrict__ p_selected,
	short * __restrict__ p_eqn_index,
	int * __restrict__ p_Ring,
	f64 * __restrict__ p_solution,
	int const whicRing);

__global__ void kernelCreateEquations(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64 * __restrict__ p_ionmomflux,
	f64 * __restrict__ p_elecmomflux,

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	long * __restrict__ p_Indexneigh_tri,
	long * __restrict__ p_izTri_vert,
	short * __restrict__ p_eqn_index, // each one assigned an equation index
	bool * __restrict__ p_bSelectFlag,
	f64 * __restrict__ eqns,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64 * __restrict__ pRHS
);


__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_effect_of_neighs_on_flux(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n,

	// we want to now have an array to store the effect of each selected on each selected residual

	f64 * __restrict__ eqns, // say it is 256*3*256*3. Actually pretty big then. Try 128*128*3*3.
	bool * __restrict__ p_selectflag, // whether it's in the smoosh to smash
	short * __restrict__ p_mapping_to_array,

	//We're going to need to store impact of each v on the gradient ..
	//then what ? Have to trace it through ?
	//Know ROC of each W wrt each one...
	//Know Pi(W)->ROC of each Pi wrt each.
	//And then e.g. for prev and next we have them again as opp ..

	// Best go through W Pi code multiple times -- loop -- to do for each vz value given ROCgrad.

	// Global access is the only way I think -- don't try to store an additional array of "how my
	// neighbours affect me" 
	// Just add to the value in the big matrix IF your neighbour is selected.
	// Only your own thread is adding to it so that's okay.

	//f64 * __restrict__ p__x, // regressor vz

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	//f64_vec3 * __restrict__ p_ROCMAR,
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s
);

__global__ void kernelCalculate_deps_WRT_beta_Visc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	nvals * __restrict__ p_n_minor, // got this
	f64 * __restrict__ p_AreaMinor, // got this -> N, Nn

	f64_vec3 * __restrict__ p_Jacobi_ion,
	f64_vec3 * __restrict__ p_Jacobi_elec,

	f64_vec3 * __restrict__ p_d_eps_by_d_beta_i_,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_e_
);

__global__ void kernelCreateEpsilon_Visc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_MAR_ion__,
	f64_vec3 * __restrict__ p_MAR_elec__,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_vec2 * __restrict__ p_epsilon_xy,
	f64 * __restrict__ p_epsilon_iz,
	f64 * __restrict__ p_epsilon_ez,
	bool * __restrict__ p_bFailedTest,
	int * __restrict__ p_Select
);

__global__ void kernelCalculateOverallVelocitiesVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major,

	ShardModel * __restrict__ p_shards_n,
	ShardModel * __restrict__ p_shards_n_n,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,
	f64 const h_full_adv
);

__global__ void kernelReturnMaximumInBlock
(
	f64 * __restrict__ p_f,
	f64 * __restrict__ p_outputmax,
	long * __restrict__ p_outputiMax,
	long * __restrict__ p_indic
);

__global__ void kernelAdvectPositionsVertex(
	f64 h_use,
	structural * __restrict__ p_info_src_major,
	structural * __restrict__ p_info_dest_major,
	f64_vec2 * __restrict__ p_v_overall_major,
	nvals * __restrict__ p_n_major,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCneigh_vert);


__global__ void kernelResetNeutralDensityOutsideRadius(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_major,
	nvals * __restrict__ p_n_minor
);

__global__ void kernelAddToAz(
	long * p_indicator,
	f64 * pAz
);
__global__ void MultiplyVector(
	f64 * __restrict__ multiply1,
	f64 * __restrict__ multiply2,
	f64 * __restrict__ output
);

__global__ void kernelAccumulateSumOfQuarts(
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_SS);

__global__ void kernelCentroidVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
);
__global__ void kernelCircumcenterVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
);

__global__ void AggregateSmashMatrix(
	f64 * __restrict__ p_Jacobianesque_list,
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_smash_matrix_block,
	f64 * __restrict__ p_smash_vector_block
);

__global__ void kernelComputeJacobianValues(

	structural * __restrict__ p_info,
	//	f64 * __restrict__ p_Aznext,
	//	f64 * __restrict__ p_Azk,
	//	f64 * __restrict__ pAzdot0,
	f64 * __restrict__ pgamma,
	f64 const h_use,
	long * __restrict__ p_indic,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_Jacobianesque_list);


__global__ void kernelCalculateROCepsWRTregressorT_volleys(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,

	char * __restrict__ p_iVolley,
	f64 * __restrict__ p_regressor_n,
	f64 * __restrict__ p_regressor_i,
	f64 * __restrict__ p_regressor_e,

	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_n_x4,
	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_i_x4,
	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_e_x4
);

__global__ void kernelCalculateArray_ROCwrt_my_neighbours(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,

	// Output:
	f64 * __restrict__ D_eps_by_dx_neigh_n, // save an array of MAXNEIGH f64 values at this location
	f64 * __restrict__ D_eps_by_dx_neigh_i,
	f64 * __restrict__ D_eps_by_dx_neigh_e,
	f64 * __restrict__ Effect_self_n,
	f64 * __restrict__ Effect_self_i,
	f64 * __restrict__ Effect_self_e
);

__global__ void kernelCalculateOptimalMove(
	structural * __restrict__ p_info_major,
	f64 * __restrict__ D_eps_by_dx_neigh_n,
	f64 * __restrict__ D_eps_by_dx_neigh_i,
	f64 * __restrict__ D_eps_by_dx_neigh_e,
	f64 * __restrict__ Effect_self_n,
	f64 * __restrict__ Effect_self_i,
	f64 * __restrict__ Effect_self_e,
	long * __restrict__ izNeigh_verts,

	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e,
	// output:
	f64 * __restrict__ p_regressor_n,
	f64 * __restrict__ p_regressor_i,
	f64 * __restrict__ p_regressor_e
);

__global__ void kernelAccumulateSummands6(
	f64 * __restrict__ p_epsilon,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaJ_x4,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaR_x4,

	// outputs:
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_J_x4,
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_R_x4,
	f64 * __restrict__ p_sum_depsbydbeta__8x8,  // do we want to store 64 things in memory? .. we don't.
	f64 * __restrict__ p_sum_eps_eps_
);

__global__ void kernelAddtoT_volleys(
	T3 * __restrict__ p_T_dest,
	//f64 beta_n[8],
	//f64 beta_i[8],
	//f64 beta_e[8], // copy arrays to constant memory ahead of time
	char * __restrict__ p_iVolley,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e
);

__global__ void kernelAdvectPositionsTris (
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_v_overall_minor);

__global__ void kernelAverage_n_T_x_to_tris  (
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_cc,

	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,

	bool bCalculateOnCircumcenters
	);

__global__ void kernelAddtoT(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 beta_nJ, f64 beta_nR,
	f64 beta_iJ, f64 beta_iR,
	f64 beta_eJ, f64 beta_eR,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e);

__global__ void kernelSetx(f64_vec3 * __restrict__ p_v1,
	f64_vec3 * __restrict__ p_src);

__global__ void kernelSety(f64_vec3 * __restrict__ p_v1,
	f64_vec3 * __restrict__ p_src);

__global__ void kernelSetz(f64_vec3 * __restrict__ p_v1,
	f64_vec3 * __restrict__ p_src);

__global__ void GetMax(
	f64 * __restrict__ p_comp1,
	long * __restrict__ p_iWhich,
	f64 * __restrict__ p_max
);

__global__ void kernelCreateJacobiRegressorxy
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_factoreps,
	f64_tens2 * __restrict__ p_factor2);


__global__ void SplitVector4(
	f64_vec2 * __restrict__ p_xy,
	f64 * __restrict__ p_z1,
	f64 * __restrict__ p_z2,
	v4 * __restrict__ p_v4,
	int * __restrict__ p_Select
);


__global__ void kernelExpandSelectFlagIta(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	int * __restrict__ p_iSelectFlag,
	int * __restrict__ p_iSelectflagNeut,
	int const number
);


__global__ void kernelAccumulateSummandsProduct(
	f64_vec2 * __restrict__ p_eps_xy, // 
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_vec2 * __restrict__ p_d_epsxy_by_d_beta, // f64_vec2
	f64 * __restrict__ p_d_eps_iz_by_d_beta,
	f64 * __restrict__ p_d_eps_ez_by_d_beta,
	// outputs:
	f64 * __restrict__ p_sum_eps_deps_
);

__global__ void kernelCreateJacobiRegressorz
(f64 * __restrict__ p_regr,
	f64 * __restrict__ p_factor1,
	f64 * __restrict__ p_factor2);

__global__ void kernelCreateJacobiRegressor_x
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_factor2);

__global__ void CreateLC4(
	f64_vec2 * __restrict__ p_regrlc2_,
	f64 * __restrict__ p_regrlc_iz_,
	f64 * __restrict__ p_regrlc_ez_,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez);

__global__ void kernelCreate_v4
(
	v4 * __restrict__ p_vie_,
	v4 * __restrict__ p_vie_save_,
	f64 const lambda_,
	f64_vec2 * __restrict__ p_regrlc2_,
	f64 * __restrict__ p_regrlc_iz_,
	f64 * __restrict__ p_regrlc_ez_);

__global__ void kernelCreateJacobiRegressor_y
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_factor2);

__global__ void CalculateCoeffself(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	// For neutral it needs a different pointer.
	f64_vec3 * __restrict__ p_v_n,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	Tensor2 * __restrict__ p__coeffself_xy, // matrix ... 
	f64 * __restrict__ p__coeffself_z,
	double4 * __restrict__ p__billabong, // xz, yz, zx, zy

										 // Note that we then need to add matrices and invert somehow, for xy.
										 // For z we just take 1/x .
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s
);


__global__ void Test_Asinh();
__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	// For neutral it needs a different pointer.
	f64_vec3 * __restrict__ p_v_n,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_MAR,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri,
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s);


__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_xy(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n,
	f64_vec2 * __restrict__ p__x, // regressor

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_ROCMAR,
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s
);// easy way to put it in constant memory;

__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species_dbydbeta_z(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n,
	f64 * __restrict__ p__x, // regressor vz

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_ROCMAR,
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s
);
__global__ void kernelCreateNeutralInverseCoeffself(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ p__coeffself_input,
	f64 * __restrict__ p__invcoeffself_output
);

__global__ void AddLCtoVector3
(f64_vec3 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_z,
	f64_vec2 * __restrict__ p_storemove_xy,
	f64 * __restrict__ p_storemove_z);

__global__ void AssembleVector2(
	f64_vec2 * __restrict__ p_vec2,
	f64 * __restrict__ p__x,
	f64 * __restrict__ p__y);

__global__ void AddLCtoVector4
(v4 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	v4 * __restrict__ p_storemove);

__global__ void kernelCreateJacobiRegressorNeutralxy
(f64_vec2 * __restrict__ p_out,
	f64_vec2 * __restrict__ p_in,
	f64 * __restrict__ p_invcoeffself);

__global__ void AddLittleBitORegressors(
	f64 const coeff,
	v4 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p__regrxy,
	f64 * __restrict__ p__regriz,
	f64 * __restrict__ p__regrez);

__global__ void kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_neut_minor,   //
	f64 * __restrict__ p_nu_neut_minor,   // 

	f64_vec3 * __restrict__ p_MAR_neut,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri);

__global__ void Multiply_components_xy // c=b-a
(f64_vec2 * __restrict__ p_result,
	f64_vec2 * __restrict__ p_multiplicand_1,
	f64_vec2 * __restrict__ p_multiplicand_2);

__global__ void kernelComputeCombinedDEpsByDBeta(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_regr_xy,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	f64_vec3 * __restrict__ p_MAR_ion2,
	f64_vec3 * __restrict__ p_MAR_elec2,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64_vec2 * __restrict__ p_Depsilon_xy,
	f64 * __restrict__ p_Depsilon_iz,
	f64 * __restrict__ p_Depsilon_ez
);

__global__ void kernelCreateDByDBetaCoeffmatrix(
	f64 const hsub,
	structural * __restrict__ p_info_minor,

	Tensor2 * __restrict__ p_matrix_xy_i,
	Tensor2 * __restrict__ p_matrix_xy_e,
	f64 * __restrict__ p_coeffself_iz,
	f64 * __restrict__ p_coeffself_ez,

	double4 * __restrict__ p_xzyzzxzy__i,
	double4 * __restrict__ p_xzyzzxzy__e,

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_vec2 * __restrict__ p_epsxy,
	f64 * __restrict__ p_epsiz,
	f64 * __restrict__ p_epsez,
	f64_vec2 * __restrict__ p_Jacxy,
	f64 * __restrict__ p_Jaciz,
	f64 * __restrict__ p_Jacez,
	
	f64_tens2 * __restrict__ p_invmatrix,
	f64 * __restrict__ p_invcoeffselfviz,
	f64 * __restrict__ p_invcoeffselfvez,
	f64 * __restrict__ p_invcoeffselfx,
	f64 * __restrict__ p_invcoeffselfy
);

__global__ void kernelCreatePredictionsDebug(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_epsxyold,
	f64 * __restrict__ p_epsizold,
	f64 * __restrict__ p_epsezold,
	f64_vec2 * __restrict__ p_d_epsxy_by_d,
	f64 * __restrict__ p_d_epsiz_by_d,
	f64 * __restrict__ p_d_epsez_by_d,
	f64_vec2 * __restrict__ p_predictxy,
	f64 * __restrict__ p_predictiz,
	f64 * __restrict__ p_predictez
);

__global__ void kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_SYMM(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_neut_minor,   //
	f64 * __restrict__ p_nu_neut_minor,   // 

	f64_vec3 * __restrict__ p_MAR_neut,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri);

__global__ void kernelAdd3(f64_vec3 * __restrict__ p_update, f64_vec3 * __restrict__ p_addition);

__global__ void kernelTransmit_3x_HeatToVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n,
	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri1,
	NTrates * __restrict__ NT_addition_tri2,
	NTrates * __restrict__ NT_addition_tri3
);

__global__ void kernelCalc_Matrices_for_Jacobi_NeutralViscosity_SYMM(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_tens3 * __restrict__ p_matrix_n
);

__global__ void Vector3Breakdown(
	f64_vec3 * __restrict__ p_input,
	f64_vec2 * __restrict__ p_outxy,
	f64 * __restrict__ p_outz,
	int * __restrict__ p_Select
);

__global__ void kernelCreateJacobiRegressorNeutralxy2 
(f64_vec2 * __restrict__ p_regr,
	f64 * __restrict__ p_factorepsx, f64 * __restrict__ p_factorepsy,
	f64 * __restrict__ p_factor2);

__global__ void Subtract(
	f64 * __restrict__ p_c,
	f64 * __restrict__ p_b,
	f64 * __restrict__ p_a
);

__global__ void Subtract_xy(
	f64_vec2 * __restrict__ p_c,
	f64_vec2 * __restrict__ p_b,
	f64_vec2 * __restrict__ p_a
);

__global__ void kernelSquare2
(f64_vec2 * __restrict__ p__x);

__global__ void kernelSquare
(f64 * __restrict__ p__x);

__global__ void kernelComputeNeutralDEpsByDBeta 
(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_regr_xy,
	f64 * __restrict__ p_regr_z,
	f64_vec3 * __restrict__ p_ROCMAR,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_z
	);


__global__ void kernelCompare(
	f64_vec2 * __restrict__ p_epsxy,
	f64 * __restrict__ p_epsiz,
	f64 * __restrict__ p_epsez,
	f64_vec2 * __restrict__ p_epsxyp,
	f64 * __restrict__ p_epsizp,
	f64 * __restrict__ p_epsezp,
	f64 * __restrict__ p_distance);

__global__ void SetProduct3(
	f64_vec3 * __restrict__p_result,
	f64_vec3 * __restrict__ p_mult1,
	f64_vec3 * __restrict__ p_mult2);


__global__ void kernelAddLC_3vec3
(f64_vec3 * __restrict__ p_vec,
	f64_vec3 coeffs_x, f64_vec3 coeffs_y, f64_vec3 coeffs_z,
	f64_vec3 * __restrict__ p_addition1,
	f64_vec3 * __restrict__ p_addition2,
	f64_vec3 * __restrict__ p_addition3
);

__global__ void ResettoGeomAverage(
	f64_vec3 * __restrict__ p_result,
	f64_vec3 * __restrict__ p_mult1);

__global__ void AccumulateSummandsScalars3(
	structural * __restrict__ p_info_minor,
	f64 * __restrict__ p__eps,
	f64 * __restrict__ p__deps_1,
	f64 * __restrict__ p__deps_2,
	f64 * __restrict__ p__deps_3,
	f64_vec3 * __restrict__ p_sum_eps_deps_,
	Symmetric3 * __restrict__ p_sum_product_matrix_,
	f64 * __restrict__ p_sum_eps_sq
);

__global__ void kernelMultiply_Get_Jacobi_NeutralVisc(
	structural * __restrict__ p_info,
	f64_vec3 * __restrict__ p_eps3,
	f64_tens3 * __restrict__ p_Matrix_n,
	f64_vec3 * __restrict__ p_Jacobi);

__global__ void kernelAdd_to_v(
	v4 * __restrict__ p_vie,
	f64 const beta_i, f64 const beta_e,
	f64_vec3 * __restrict__ p_vJacobi_ion,
	f64_vec3 * __restrict__ p_vJacobi_elec
);

__global__ void kernelAddLC_vec3
(f64_vec3 * __restrict__ p_vec,
	f64_vec3 coeff,
	f64_vec3 * __restrict__ p_addition);

__global__ void kernelCalc_Matrices_for_Jacobi_NeutralViscosity(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_tens3 * __restrict__ p_matrix_n
);

__global__ void SubtractVector3(f64_vec3 * __restrict__ p_output,
	f64_vec3 * __restrict__ p_a, f64_vec3 * __restrict__ p_b);

__global__ void kernelAccumulateSumOfSquares3(
	f64_vec3 * __restrict__ p_eps,
	f64_vec3 * __restrict__ p_SS);

__global__ void KillOutsideRegion(
	f64_vec2 * __restrict__ targ2,
	f64 * __restrict__ targ,
	int * __restrict__ p_Select);

__global__ void ScaleVector3(
	f64_vec3 * __restrict__ p_eps,
	f64 const factorx, f64 const factory, f64 const factorz);

__global__ void kernelAccumulateSummands2(
	structural * __restrict__ p_info,

	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,

	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps);

__global__ void kernelCreateEpsilon_NeutralVisc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	f64_vec3 * __restrict__ p_v_n_k,
	f64_vec3 * __restrict__ pMAR_neut,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ p_eps_x,
	f64 * __restrict__ p_eps_y,
	f64 * __restrict__ p_eps_z,
	bool * __restrict__ p_bFailedTest,
	int * __restrict__ p_Select
);

__global__ void kernelAccumulateSummandsNeutVisc(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_eps3,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_x_,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_y_,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_z_,
	// outputs:
	f64_vec3 * __restrict__ p_sum_eps_deps_,
	Symmetric3 * __restrict__ p_sum_product_matrix_,
	f64 * __restrict__ p_sum_eps_sq
);

__global__ void kernelAccumulateSummandsNeutVisc2(
	f64 * __restrict__ p_eps,  // 
	f64 * __restrict__ p_d_eps_by_d_beta,
	// outputs:
	f64 * __restrict__ p_sum_eps_deps_,  // 8 values for this block
	f64 * __restrict__ p_sum_product_matrix_
);

__global__ void kernelAccumulateSummands3(
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_vec2 * __restrict__ p_d_epsxy_by_d_beta_i_,
	f64 * __restrict__ p_d_eps_iz_by_d_beta_i_,
	f64 * __restrict__ p_d_eps_ez_by_d_beta_i_,

	f64_vec2 * __restrict__ p_d_epsxy_by_d_beta_e_,
	f64 * __restrict__ p_d_eps_iz_by_d_beta_e_,
	f64 * __restrict__ p_d_eps_ez_by_d_beta_e_,

	f64 * __restrict__ p_sum_eps_deps_by_dbeta_i_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_e_,
	f64 * __restrict__ p_sum_depsbydbeta_i_times_i_,
	f64 * __restrict__ p_sum_depsbydbeta_e_times_e_,
	f64 * __restrict__ p_sum_depsbydbeta_e_times_i_,
	f64 * __restrict__ p_sum_eps_sq
);

__global__ void SubtractT3(
	T3 * __restrict__ p_result,
	T3 * __restrict__ p_a, T3 * __restrict__ p_b);

__global__ void kernelAccumulateSummands4(

	// We don't need to test for domain, we need to make sure the summands are zero otherwise.
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_d_beta_J,
	f64 * __restrict__ p_d_eps_by_d_beta_R,

	f64 * __restrict__ p_sum_eps_deps_by_dbeta_J_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_J_,
	f64 * __restrict__ p_sum_depsbydbeta_R_times_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_R_,
	f64 * __restrict__ p_sum_eps_sq
);

__global__ void kernelDummy(
	f64 * __restrict__ p_d_eps_by_d_beta_n
);

__global__ void kernelCreateEpsilonHeat
(
	f64 const hsub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_NT_n,
	f64 * __restrict__ p_NT_i,
	f64 * __restrict__ p_NT_e,
	T3 * __restrict__ p_T_k,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	NTrates * __restrict__ NTadditionrates, // it's especially silly having a whole struct of 5 instead of 3 here.
	bool * __restrict__ p_b_Failed,
		bool * __restrict__ p_bMask3,
		bool * __restrict__ p_bMaskblock,
		bool bUseMask
);

__global__ void kernelCreateEpsilonAndJacobi_Heat
(
	f64 const h_sub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	T3 * p_Tk, // T_k for substep

			   // f64 * __restrict__ p_Azdot0,f64 * __restrict__ p_gamma, 
			   // corresponded to simple situation where Azdiff = h*(Azdot0+gamma Lap Az)

	NTrates * __restrict__ p_NTrates_diffusive,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p__coeffself_n, // what about dividing by N?
	f64 * __restrict__ p__coeffself_i,
	f64 * __restrict__ p__coeffself_e,
	f64 * __restrict__ p__epsilon_n,
	f64 * __restrict__ p__epsilon_i,
	f64 * __restrict__ p__epsilon_e,
	f64 * __restrict__ p__Jacobi_n,
	f64 * __restrict__ p__Jacobi_i,
	f64 * __restrict__ p__Jacobi_e,
	bool * __restrict__ p_bFailedTest,
	bool * __restrict__ p_bMask3,
	bool * __restrict__ p_bMaskblock,
	bool bUseMask
);

__global__ void kernelCreateWhoAmI_verts(
	structural * __restrict__ p_info_major,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_sz_who_am_I // array of MAXNEIGH shorts for each vertex.
);

__global__ void kernelAddStoredNTFlux(
	structural * __restrict__ p_info_major,
	NTrates * __restrict__ p_additional_array,
	NTrates * __restrict__ p_values_to_augment
);

__device__ void Augment_Jacobean(
	f64_tens3 * pJ,
	real Factor, //h_over (N m_i)
	f64_vec2 edge_normal,
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
);
__global__ void kernelAccumulateAdvectiveMassHeatRateNew(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,

	long * __restrict__ p_izNeigh_vert,

	short * __restrict__ p_who_am_I_to_my_neighbours,

	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,   // use T vertex itself to infer what T to use.

	v4 * __restrict__ p_vie_minor,
	// f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,
	//T3 * __restrict__ p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

	// ShardModel * __restrict__ p_n_shard_n_major,
	ShardModel * __restrict__ p_n_shard_major,

	NTrates * __restrict__ p_NTadditionrates,
	f64 * __restrict__ p_div_v,
	//	f64 * __restrict__ p_div_v_n, // write ion & electron routine only first; re-do as neutral.
	f64 * __restrict__ p_Integrated_div_v_overall,

	NTrates * __restrict__ p_store_flux
);

__global__ void kernelAccumulateNeutralAdvectiveMassHeatRateNew(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_who_am_I_to_my_neighbours,

	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,   // use T vertex itself to infer what T to use.

	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,
	ShardModel * __restrict__ p_n_shard_major,

	NTrates * __restrict__ p_NTadditionrates,
	f64 * __restrict__ p_div_v_n,
	NTrates * __restrict__ p_store_flux
);

__device__ void Augment_JacobeanNeutral(
	f64_tens3 * pJ,
	real Factor, //h_over (N m_i)
	f64_vec2 edge_normal,
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
);

__global__ void kernelPrepareNuGraphs(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,
	f64 * __restrict__ p_nu_e_MT,
	f64 * __restrict__ p_nu_en_MT
);

__global__ void kernelTileMaxMajor(
	f64 * __restrict__ p_z,
	f64 * __restrict__ p_max
);
__global__ void kernelCalculate_kappa_nu(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,
	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e
);

__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea_Debug(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_major,
	nvals * __restrict__ p_n_minor,

	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
	//	long * __restrict__ Tri_n_lists,
	//	long * __restrict__ Tri_n_n_lists	,
	f64 * __restrict__ p_AreaMajor,
	f64 * __restrict__ p_CPU_n_cent);


__global__ void kernelAccumulateDiffusiveHeatRateROC_wrt_T_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p___result, // d/dT of d(NT)/dt in this cell

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies);

__global__ void kernelAccumulateDiffusiveHeatRate_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies);

__global__ void kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 const h_use,
	f64 * __restrict__ p__x,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p___result, // d/dbeta of d(NT)/dt in this cell

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies);

__global__ void kernelAccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_epsilon,
	f64 const h_use,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies,

	f64 * __restrict__ p_array,
	f64 * __restrict__ p_effectself
);
__global__ void AddFromMyNeighbours(
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_array,
	f64 * __restrict__ p_arrayself,
	f64 * __restrict__ p_sum,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_who_am_I_to_you
);


__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_major,
	nvals * __restrict__ p_n_minor,

	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	f64_vec2 * __restrict__ p_cc,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
//	long * __restrict__ Tri_n_lists,
//	long * __restrict__ Tri_n_n_lists,
	f64 * __restrict__ p_AreaMajor,
	bool bUseCircumcenter);

__global__ void kernelSetBlockMaskFlag_CountEquations_reset_Tk(
	bool * __restrict__ p_bMask3,
	bool * __restrict__ p_bMaskBlock3,
	long * __restrict__ p_longblock3,
	T3 * __restrict__ p_T_k,
	T3 * __restrict__ p_T
);

__global__ void kernelSetNeighboursBwd(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izNeigh_vert,
	bool * __restrict__ p_bMask3);

__global__ void kernelInferMinorDensitiesFromShardModel(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_shards_n,
	LONG3 * __restrict__ p_tri_corner_index,
	LONG3 * __restrict__ p_who_am_I_to_corner,
	nvals * __restrict__ p_one_over_n
); 

__global__ void kernelSelectivelyZeroNTrates(
	NTrates * __restrict__ NTadditionrates,
	bool * __restrict__ pMaskbool3
);

__global__ void kernelCreateTfromNTbydividing_bysqrtDN(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_sqrtDNn_Tn,
	f64 * __restrict__ p_sqrtDN_Ti,
	f64 * __restrict__ p_sqrtDN_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_sqrtDinv_n, f64 * __restrict__ p_sqrtDinv_i, f64 * __restrict__ p_sqrtDinv_e
);


__global__ void kernelUnpacktorootDN_T(
	f64 * __restrict__ psqrtDNnTn,
	f64 * __restrict__ psqrtDNTi,
	f64 * __restrict__ psqrtDNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_D_n,
	f64 * __restrict__ p_D_i,
	f64 * __restrict__ p_D_e,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major);

__global__ void kernelCreateEpsilonHeat_Equilibrated
(
	f64 const hsub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_sqrtDNT_n,
	f64 * __restrict__ p_sqrtDNT_i,
	f64 * __restrict__ p_sqrtDNT_e,
	T3 * __restrict__ p_T_k,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_invsqrtD_n,
	f64 * __restrict__ p_invsqrtD_i,
	f64 * __restrict__ p_invsqrtD_e,
	NTrates * __restrict__ NTadditionrates, // it's especially silly having a whole struct of 5 instead of 3 here.
	bool * __restrict__ p_b_Failed,
	bool * __restrict__ p_bMask3,
	bool * __restrict__ p_bMaskblock,
	bool bUseMask
);

__global__ void kernelPowerminushalf
(f64 * __restrict__ p_input, f64 * __restrict__ p_output);

__global__ void kernelReturnNumberNegativeT(
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T,
	long * __restrict__ p_sum
);


__global__ void kernelPrepareIonizationGraphs(
	structural * __restrict__ p_info_major,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ ratesNT, // dN/dt, dNeTe/dt
	f64_vec3 * __restrict__ p_dNve, // --> d/dt v_e

	f64 * __restrict__ p_graph1, 
	f64 * __restrict__ p_graph2, 
	f64 * __restrict__ p_graph3, 
	f64 * __restrict__ p_graph4, 
	f64 * __restrict__ p_graph5, 
	f64 * __restrict__ p_graph6);


__global__ void kernelCompareForStability_andSetFlag(
	structural * __restrict__ p_info_minor,
	NTrates * __restrict__ p_NTrates1,
	NTrates * __restrict__ p_NTrates2,
	long * __restrict__ p_sum,
	bool * __restrict__ p_bMask3
);

__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n,
	T3 * __restrict__ p_T,
	species3 * __restrict__ p_nu);

__global__ void kernelCreatePutativeTandsave(
	f64 hsub,
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_k,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ NTadditionrates,
	T3 * __restrict__ p_T_dest,
	bool * bMask3
);


__global__ void kernelKillNeutral_v_OutsideRadius(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n
);

__global__ void kernelCreatePutativeT(
	f64 hsub,
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_k,
	// T3 * __restrict__ p_T_putative,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ NTadditionrates,

	bool * __restrict__ p_boolarray, // 2x NMAJOR
	bool * __restrict__ p_bFailedtest,
	bool * __restrict__ p_bMask3,
	bool * __restrict__ p_bMaskBlock, // do 1 for all species
	bool bUseMask
);

__global__ void kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	T3 * __restrict__ p_T_k,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor);

__global__ void kernelCreateEpsilonHeatOriginalScaling
(
	f64 const hsub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	T3 * __restrict__ p_T_k,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	NTrates * __restrict__ NTadditionrates,// it's especially silly having a whole struct of 5 instead of 3 here.
	bool * __restrict__ bTest
);

__global__ void kernelUnpacktorootNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major);


__global__ void kernelUnpack(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT);
__global__ void kernelUnpacktoNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major);
__global__ void NegateVectors(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_x2, f64 * __restrict__ p_x3);

__global__ void kernelAccumulateSumOfSquares(
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_SS_n,
	f64 * __restrict__ p_SS_i,
	f64 * __restrict__ p_SS_e);

__global__ void kernelRegressorUpdate
(
	f64 * __restrict__ p_x_n,
	f64 * __restrict__ p_x_i,
	f64 * __restrict__ p_x_e,
	f64 * __restrict__ p_a_n, f64 * __restrict__ p_a_i, f64 * __restrict__ p_a_e,
	f64 const ratio1, f64 const ratio2, f64 const ratio3,
	bool * __restrict__ p_bMaskBlock,
	bool bUseMask
	);

__global__ void VectorAddMultiple(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3);

__global__ void kernelPackupT3(
	T3 * __restrict__ p_T,
	f64 * __restrict__ p_Tn, f64 * __restrict__ p_Ti, f64 * __restrict__ p_Te);

__global__ void kernelAccumulateDotProducts(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_x2, f64 * __restrict__ p_y2,
	f64 * __restrict__ p_x3, f64 * __restrict__ p_y3,
	f64 * __restrict__ p_dot1,
	f64 * __restrict__ p_dot2,
	f64 * __restrict__ p_dot3);


__global__ void kernelCreateTfromNTbydividing(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_Nn_Tn,
	f64 * __restrict__ p_N_Ti,
	f64 * __restrict__ p_N_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major);

__global__ void kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_T_n, f64 * __restrict__ p_T_i, f64 * __restrict__ p_T_e,
//	T3 * __restrict__ p_T_k,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,

	bool * __restrict__ p_maskbool3,
	bool * __restrict__ p_maskblock,
	bool bUseMask
	
	);

__global__ void kernelAccumulateDiffusiveHeatRate_new_Full(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	bool * __restrict__ p_bool_longi,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,

	bool bCheckWhetherToDoctorUp,
	bool * __restrict__ p_maskbool3,
	bool * __restrict__ p_maskblock,
	bool bUseMask
);
 

__global__ void DivideVec2(f64_vec2 * __restrict__ p_update,
	f64_vec2 * __restrict__ p_apply);

__global__ void SubtractVec2(f64_vec2 * __restrict__ p_update,
	f64_vec2 * __restrict__ p_apply);

__global__ void Augment_dNv_minor(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	LONG3 * __restrict__ p_tricornerindex,
	f64 * __restrict__ p_temp_Ntotalmajor,
	f64 * __restrict__ p_temp_Nntotalmajor,
	f64 * __restrict__ p_AreaMinor,
	f64_vec3 * __restrict__ p_MAR_neut_major,
	f64_vec3 * __restrict__ p_MAR_ion_major,
	f64_vec3 * __restrict__ p_MAR_elec_major,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec);

__global__ void Collect_Ntotal_major(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_temp_Ntotalmajor,
	f64 * __restrict__ p_temp_Nntotalmajor)
	;

__global__ void kernelIonisationRates(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_major,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ NTadditionrates,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	v4 * __restrict__ p_v,
	f64_vec3 * __restrict__ p_v_n, 
	T3 * __restrict__ p_T_use_major,
	bool b_useTuse
);
//__global__ void kernelIonisationRates(
//	f64 const h_use,
//	structural * __restrict__ p_info_minor,
//	T3 * __restrict__ p_T_major,
//	nvals * __restrict__ p_n_major,
//	f64 * __restrict__ p_AreaMajor,
//	NTrates * __restrict__ NTadditionrates
//
//);

__global__ void kernelAccumulateDiffusiveHeatRateAndCalcIonisation(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_cc,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,
	species3 * __restrict__ p_nu_major,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor);

__global__ void kernelAdvanceDensityAndTemperature(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use,

	f64 * __restrict__ p_div_v_neut,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_Integrated_div_v_overall,
	f64 * __restrict__ p_AreaMajor, // hmm

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest
);
__global__ void kernelGetLap_minor__sum_detectcontribs(

	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_integralLapAz,
	f64 * __restrict__ p_integralVT,
	f64 * __restrict__ p_integralTV,
	f64 * __restrict__ p_integralTT,
	f64 * __restrict__ p_contriblist
);
__global__ void kernelGetLap_minor__sum_placecontribs(

	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_integralLapAz,
	f64 * __restrict__ p_integralVT,
	f64 * __restrict__ p_integralTV,
	f64 * __restrict__ p_integralTT,
	f64 * __restrict__ p_contriblist
);

__global__ void kernelSet(
	v4 * __restrict__ p_v4,
	f64_vec3 * __restrict__ p_src,
	int flag
);

__global__ void kernelGetLap_minor__sum(

	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_integralLapAz,
	f64 * __restrict__ p_integralVT,
	f64 * __restrict__ p_integralTV,
	f64 * __restrict__ p_integralTT
);

__global__ void kernelCalculateVelocityAndAzdot_noadvect__debugintegrate(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?
	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out,

	f64 * __restrict__ p_integ_Jz1,
	f64 * __restrict__ p_integ_Jz2,
	f64 * __restrict__ p_integ_LapAz
);

__global__ void kernelCalculateUpwindDensity_tris(
	structural * __restrict__ p_info_minor,
	ShardModel * __restrict__ p_n_shard_n_major,
	ShardModel * __restrict__ p_n_shard_major,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor_minor,
	LONG3 * __restrict__ p_tricornerindex,
	LONG3 * __restrict__ p_trineighindex,
	LONG3 * __restrict__ p_which_iTri_number_am_I,
	CHAR4 * __restrict__ p_szPBCneigh_tris, 
	nvals * __restrict__ p_n_upwind_minor ,// result
	T3 * __restrict__ p_T_minor,
	T3 * __restrict__ p_T_upwind_minor
);

__global__ void Estimate_Effect_on_Integral_Azdot_from_Jz_and_LapAz(
	f64 hstep,
	structural * __restrict__ p_info,
	nvals * __restrict__ p_nvals_k,
	nvals * __restrict__ p_nvals_use,
	v4 * __restrict__ p_vie_k,
	v4 * __restrict__ p_vie_kplus1,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz,

	AAdot * __restrict__ p_Azdot,

	f64 * __restrict__ p_tile1, // +ve Jz
	f64 * __restrict__ p_tile2, // -ve Jz
	f64 * __restrict__ p_tile3, // LapAz
	f64 * __restrict__ p_tile4, // integrate Azdot diff
	f64 * __restrict__ p_tile5,
	f64 * __restrict__ p_tile6
);

__global__ void kernelAccumulateAdvectiveMassHeatRate(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,

	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,

	nvals * __restrict__ p_n_upwind_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,
	T3 * __restrict__ p_T_upwind_minor, // may or may not overlap source: don't we only use from tris? so not overlap

	NTrates * __restrict__ p_NTadditionrates,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_div_v_n,
	f64 * __restrict__ p_Integrated_div_v_overall
	);

__global__ void kernelEstimateCurrent(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	v4 * __restrict__ p_vie,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Iz
);

__global__ void kernelPopulateOhmsLaw(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,

	nvals * __restrict__ p_n_minor_use,
	nvals * __restrict__ p_one_over_n,
	T3 * __restrict__ p_T_minor_use,
	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor, 
	f64 * __restrict__ ROCAzdotduetoAdvection,

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,
	
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,

	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave,
	bool const bUse_dest_n_for_Iz,
	nvals * __restrict__ p_n_dest_minor);

__global__ void kernelPopulateOhmsLaw_dbg2(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,

	nvals * __restrict__ p_n_minor_use,
	T3 * __restrict__ p_T_minor_use,
	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ ROCAzdotduetoAdvection,

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,

	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave,
	bool const bUse_dest_n_for_Iz,
	nvals * __restrict__ p_n_dest_minor,
	f64 * __restrict__ p_debug,
	f64 * __restrict__ p_debug2,
	f64 * __restrict__ p_debug3
		);

__global__ void SubtractVector(
	f64 * __restrict__ result,
	f64 * __restrict__ b,
	f64 * __restrict__ a);


__global__ void kernelCreateExplicitStepAz(
	f64 const hsub,
	f64 * __restrict__ pAzdot0,
	f64 * __restrict__ pgamma,
	f64 * __restrict__ pLapAz, // we based this off of half-time Az.
	f64 * __restrict__ p_result); // = h (Azdot0 + gamma*LapAz)


__global__ void kernelCreateEpsilon_Heat_for_Jacobi
(
	f64 const h_sub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	T3 * p_T_k, // T_k for substep

	NTrates * __restrict__ p_NTrates_diffusive,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e, 
	bool * __restrict__ p_bMask3,
	bool * __restrict__ p_bMaskblock,
	bool bUseMask
);

__global__ void kernelCalc_SelfCoefficient_for_HeatConduction
(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,
	nvals * __restrict__ p_n_major,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p_coeffself_n,
	f64 * __restrict__ p_coeffself_i,
	f64 * __restrict__ p_coeffself_e // outputs
	);

__global__ void kernelCreateEpsilon_NeutralVisc__Modified(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	f64_vec3 * __restrict__ p_v_n_k,
	f64_vec3 * __restrict__ pMAR_neut,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ p_eps_x,
	f64 * __restrict__ p_eps_y,
	f64 * __restrict__ p_eps_z,
	int * __restrict__ p_iFailedTest
);
__global__ void kernelCreateEpsilon_Visc__Modified(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_MAR_ion2,
	f64_vec3 * __restrict__ p_MAR_elec2,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_vec2 * __restrict__ p_epsilon_xy,
	f64 * __restrict__ p_epsilon_iz,
	f64 * __restrict__ p_epsilon_ez,
	int * __restrict__ p_iFailedTest
);

__global__ void kernelZeroSelected(
	f64_vec3 * __restrict__ p_MAR_i,
	f64_vec3 * __restrict__ p_MAR_e,
	f64_vec3 * __restrict__ p_MAR_n,
	NTrates * __restrict__ p_NTrates_vert,
	NTrates * __restrict__ p_NTrates_tri,
	int * __restrict__ p_Select,
	int * __restrict__ p_SelectNeut
);

__global__ void kernelCreate_v_k_modified_with_fixed_flows(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_modify_k,
	f64_vec3 * __restrict__ p_v_n_modify_k,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_v_n_k,

	v4 * __restrict__ p_v_updated,
	f64_vec3 * __restrict__ p_v_n_updated,
	int * __restrict__ p_Select,
	int * __restrict__ p_SelectNeut,

	f64_vec3 * __restrict__ p_MAR_ion_,
	f64_vec3 * __restrict__ p_MAR_elec_,
	f64_vec3 * __restrict__ p_MAR_neut_,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor);


__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species___fixedflows_only(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	// For neutral it needs a different pointer.
	f64_vec3 * __restrict__ p_v_n, // UNUSED!

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_MAR,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri,
	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s,
	int * __restrict__ p_Select
);

__global__ void kernelCreate_neutral_viscous_contrib_to_MAR_and_NT_Geometric___fixedflows_only(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_neut_minor,   //
	f64 * __restrict__ p_nu_neut_minor,   // 

	f64_vec3 * __restrict__ p_MAR_neut,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri,
	int * __restrict__ p_Select);


__global__ void kernelTest_Derivatives(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_ion__1,
	f64_vec3 * __restrict__ p_MAR_elec__1,
	f64_vec3 * __restrict__ p_MAR_neut__1,

	f64_vec3 * __restrict__ p_MAR_ion__2,
	f64_vec3 * __restrict__ p_MAR_elec__2,
	f64_vec3 * __restrict__ p_MAR_neut__2,

	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n, // for use.
	nvals * __restrict__ p_n,
	f64 * __restrict__ p_AreaMinor,

	int * __restrict__ p_iSelect,
	int * __restrict__ p_iSelectNeut // this is the output
);

__global__ void kernelCountNumberMoreThanZero(
	int * __restrict__ p_thing1,
	int * __restrict__ p_thing2,
	long * __restrict__ p_blocktot1,
	long * __restrict__ p_blocktot2);

__global__ void kernelPutativeAccel(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_v_n_k,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_vec3 * __restrict__ p_MAR_ion_,
	f64_vec3 * __restrict__ p_MAR_elec_,
	f64_vec3 * __restrict__ p_MAR_neut_
);

__global__ void kernelSplit_vec2(
	f64_vec2 * __restrict__ p_vec2,
	f64 * __restrict__ p_a,
	f64 * __restrict__ p_b);

__global__ void kernelCreate_neutral_viscous_contrib_to_MAR_and_NT(

	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_neut_minor,   //
	f64 * __restrict__ p_nu_neut_minor,   // 

	f64_vec3 * __restrict__ p_MAR_neut,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri);

__global__ void kernelCalculate_ita_visc(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,

	f64 * __restrict__ p_nu_ion_minor,
	f64 * __restrict__ p_nu_elec_minor,
	f64 * __restrict__ p_nu_nn_visc,
	f64 * __restrict__ p_ita_par_ion_minor,
	f64 * __restrict__ p_ita_par_elec_minor,
	f64 * __restrict__ p_ita_neutral_minor,

	int * __restrict__ p_iSelect,
	int * __restrict__ p_iSelectNeut);

__global__ void kernelTransmitHeatToVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n,
	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri);

__global__ void Collect_Nsum_at_tris(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	LONG3 * __restrict__ p_tricornerindex,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n);

__global__ void kernelCreate_viscous_contrib_to_MAR_and_NT(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,
	
	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri
);


__global__ void kernelCalculateVelocityAndAzdot_debug(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_intermediate,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out,
	bool * __restrict__ p_alertflag);

__global__ void kernelSetZero(
	f64 * __restrict__ data
);

__global__ void kernelCalculateVelocityAndAzdot_dbg2(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_intermediate,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out,
	f64 * __restrict__ p_debug,
	f64 * __restrict__ p_debug2);

__global__ void kernelCreateEpsilonAndJacobiDebug(
	f64 const h_use,
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_array_next,
	f64 * __restrict__ p_Az_array,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_Lap_Aznext,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi_x,
	AAdot * __restrict__ p_AAdot_k,
	bool * __restrict__ p_bFail);

__global__ void kernelCalculateVelocityAndAzdot(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?
	f64 * __restrict__ p_ROCAzdotantiadvect,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out);

__global__ void kernelAverage(
	f64 * __restrict__ p_update,
	f64 * __restrict__ p_input2);

__global__ void kernelAdvanceAzEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection);

__global__ void kernelUpdateAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
);
__global__ void kernelPopulateArrayAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
);
__global__ void kernelPushAzInto_dest(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
);
__global__ void kernelPullAzFromSyst(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
);
__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
);
__global__ void kernelCreateSeedPartOne(
	f64 const h_use,
	f64 * __restrict__ p_Az,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_AzNext
);

__global__ void kernelCreateSeedPartTwo(
	f64 const h_use,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext_update
);

__global__ void kernelResetFrillsAz(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az);

__global__ void kernelResetFrillsAz_II(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	AAdot * __restrict__ p_Az);

__global__ void kernelCreateLinearRelationshipBwd(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz,
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_k,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ ROCAzdotduetoAdvection
);

__global__ void kernelAdvanceAzBwdEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	bool const bUseROC);

__global__ void kernelPopulateBackwardOhmsLaw(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,
	nvals * __restrict__ p_n_minor_use,
	T3 * __restrict__ p_T_minor_use,
	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ ROCAzdotduetoAdvection,
	// Now going to need to go through and see this set 0 or sensible every time.

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	//AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave);

__global__ void Divide_diff_get_accel(
	v4 * __restrict__ p_vie_f,
	v4 * __restrict__ p_vie_i,
	f64 const h_use,
	f64 * __restrict__ p_output
);

__global__ void DivideMARDifference_get_accel_y(
	f64_vec3 * __restrict__ pMAR_ion,
	f64_vec3 * __restrict__ pMAR_elec,
	f64_vec3 * __restrict__ pMAR_ion_old,
	f64_vec3 * __restrict__ pMAR_elec_old,
	nvals * __restrict__ p_n,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_output_y
);
__global__ void DivideMAR_get_accel(
	f64_vec3 * __restrict__ pMAR_ion,
	f64_vec3 * __restrict__ pMAR_elec,
	nvals * __restrict__ p_n,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_output_x,
	f64 * __restrict__ p_output_y
);
__global__ void MeasureAccelz(
	structural * __restrict__ p_info,
	v4 * __restrict__ p_vie_initial,
	v4 * __restrict__ p_vie_final,
	f64_vec3 * __restrict__ p_v_nk,
	f64_vec3 * __restrict__ p_v_nkplus1,

	f64 const h_use, // substep
	f64_vec2 * __restrict__ pGradAz,
	f64_vec2 * __restrict__ pGradTe,
	AAdot * __restrict__ p_AAdot,
	AAdot * __restrict__ p_AAdot_k,
	f64 * __restrict__ pLapAz,

	nvals * __restrict__ p_n_central,
	T3 * __restrict__ p_T_central,
	f64_vec3 * __restrict__ p_B,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ p_arelz,
	f64 * __restrict__ p_MAR_ion_effect,
	f64 * __restrict__ p_MAR_elec_effect,
	f64 * __restrict__ p_Ezext_electromotive,
	f64 * __restrict__ p_inductive_electromotive,
	f64 * __restrict__ p_vxB,
	f64 * __restrict__ p_thermal_force_effect,
	f64 * __restrict__ p_friction_neutrals,
	f64 * __restrict__ p_friction_ei,
	f64 * __restrict__ p_sum_of_effects,
	f64 * __restrict__ p_difference
);

__global__ void MeasureAccelxy_and_JxB_and_soak(
	v4 * __restrict__ p_vie_final,
	v4 * __restrict__ p_vie_initial,
	f64 const h_use,
	f64_vec2 * __restrict__ pGradAz,
	nvals * __restrict__ p_n_central,
	T3 * __restrict__ p_T_central,
	f64_vec3 * __restrict__ p_v_nk,
	f64_vec3 * __restrict__ p_v_nkplus1,

	f64 * __restrict__ p_accel_x,
	f64 * __restrict__ p_accel_y,
	f64 * __restrict__ p_vxB_x,
	f64 * __restrict__ p_vxB_y,
	f64 * __restrict__ p_grad_y_Az,
	f64 * __restrict__ p_soak_y
);

__global__ void kernelReset_v_in_outer_frill_and_outermost
(
	structural * __restrict__ p_info,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	T3 * __restrict__ p_T_minor,
	LONG3 * __restrict__ trineighbourindex,
	long * __restrict__ p_izNeigh_vert
);

__global__ void kernelAddRegressors(
	f64 * __restrict__ p_AzNext,
	f64 const beta0, f64 const beta1, f64 const beta2,
	f64 * __restrict__ p_reg1,
	f64 * __restrict__ p_reg2,
	f64 * __restrict__ p_reg3
	);

__global__ void kernelAccumulateMatrix_debug(
	structural * __restrict__ p_info,
	f64 const h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_regressor1,
	f64 * __restrict__ p_regressor2,
	f64 * __restrict__ p_regressor3,
	f64 * __restrict__ p_LapReg1,
	f64 * __restrict__ p_LapReg2,
	f64 * __restrict__ p_LapReg3,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_deps_matrix,
	f64_vec3 * __restrict__ p_eps_against_deps,

	f64 * __restrict__ p_deps_1,
	f64 * __restrict__ p_deps_2,
	f64 * __restrict__ p_deps_3

);

__global__ void kernelAccumulateMatrix(
	structural * __restrict__ p_info,
	f64 const h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_regressor1,
	f64 * __restrict__ p_regressor2,
	f64 * __restrict__ p_regressor3,
	f64 * __restrict__ p_LapReg1,
	f64 * __restrict__ p_LapReg2,
	f64 * __restrict__ p_LapReg3,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_deps_matrix,
	f64_vec3 * __restrict__ p_eps_against_deps
	);

__global__ void kernelAccumulateSummandsVisc(
	f64_vec2 * __restrict__ p_eps_xy, // 
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_vec2 * __restrict__ p_d_epsxy_by_d_beta, // f64_vec2
	f64 * __restrict__ p_d_eps_iz_by_d_beta,
	f64 * __restrict__ p_d_eps_ez_by_d_beta,
	// outputs:
	f64 * __restrict__ p_sum_eps_deps_,  // 8 values for this block
	f64 * __restrict__ p_sum_product_matrix_
);

__global__ void SubtractVector4(v4 * __restrict__ p_output,
	v4 * __restrict__ p_a, v4 * __restrict__ p_b);

__global__ void AddLCtoVector4component(
	v4 * __restrict__ p_operand,
	v4 * __restrict__ p_regr,
	v4 * __restrict__ p_storemove);

__global__ void kernelCreate_further_regressor(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_regressor,
	f64 * __restrict__ p_Lap_regressor,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_regressor2);

__global__ void AssembleVector4(v4 * __restrict__ p_output,
	f64_vec2 * __restrict__ p_xy, f64 * __restrict__ p_iz, f64 * __restrict__ p_ez);

__global__ void kernelAccumulateSumOfSquares_4(
	v4 * __restrict__ p_eps,
	f64 * __restrict__ p_SS);

__global__ void SubtractVector4stuff(v4 * __restrict__ p_output,
	f64_vec2 * __restrict__ p_xy,
	f64 * __restrict__ p_iz,
	f64 * __restrict__ p_ez,
	v4 * __restrict__ p_to_subtract);

__global__ void NegateVector(
	f64 * __restrict__ p_x1);

__global__ void kernelAccumulateSumOfSquares2vec(
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_SS);

__global__ void ScaleVector4(
	v4 * __restrict__ p_eps,
	f64 const factor);

__global__ void kernelCreateAzbymultiplying(
	f64 * __restrict__ p_Az,
	f64 * __restrict__ p_scaledAz,
	f64 * __restrict__ p_factor
);

__global__ void kernelAccumulateSumOfSquares1(
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_SS);

__global__ void kernelAccumulateDotProduct(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_dot1);
__global__ void CreateConjugateRegressor(
	f64 * __restrict__ p_regr, f64 const betarat, f64 * __restrict__ p_eps);


__global__ void VectorAddMultiple1(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1);

__global__ void kernelDividebyroothgamma
(
	f64 * __restrict__ result,
	f64 * __restrict__ p__sqrtfactor,
	f64 * __restrict__ Az,
	f64 const hsub,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_Area
);

__global__ void kernelCreateSeedAz(
	f64 const h_use,
	f64 * __restrict__ p_Az_k,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext);

__global__ void kernelCreateEpsilonAndJacobi(
	f64 const h_use,
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_array_next,
	f64 * __restrict__ p_Az_array,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_Lap_Aznext,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi_x,
	bool * __restrict__ p_bFail);

__global__ void kernelAccumulateSummands(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi,
	f64 * __restrict__ p_LapJacobi,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps);


/*
__global__ void kernelGetLap_verts(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izNeighMinor,
	long * __restrict__ p_izTri,
	f64 * __restrict__ p_LapAz);
*/

__global__ void kernelCalculateVelocityAndAzdot_noadvect_SPIT(
	f64 h_use,
	structural * p_info_minor,
	LONG3 * p_tricornerindex,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out);

__global__ void kernelPopulateResiduals(
	f64 * __restrict__ pLapAz,
	nvals * __restrict__ p_n_minor,
	v4 * __restrict__ p_vie,
	f64 * __restrict__ p_residual
);

__global__ void kernelGetLapCoeffs(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf);

__global__ void kernelGetLapCoeffs_and_min(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_min_array,
	long * __restrict__ p_min_index);

__global__ void kernelGetLapCoeffs_and_min_DEBUG(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_min_array,
	long * __restrict__ p_min_index);


__global__ void kernelGetLap_minor_SYMMETRIC(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AreaMinor ,
	bool const bDivideByArea
);


__global__ void kernelGetLap_minor_debug(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,

	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_LapAz,

	f64 * __restrict__ p_Integratedconts_fromtri,
	f64 * __restrict__ p_Integratedconts_fromvert,
	f64 * __restrict__ p_Integratedconts_vert,
	
	f64 * __restrict__ p_AreaMinor);


__global__ void kernelCreateEpsilon_Az_CG(
	f64 const h_use,
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_plus,
	f64 * __restrict__ p_Az_k,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_Lap_Az,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p__sqrtfactor,
	bool * __restrict__ p_bFail,
	bool const bSaveFail);


__global__ void kernelGet_AreaMinorFluid(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,
	bool * __restrict__ bz_pressureflag,
	f64 * __restrict__ p_AreaMinor
);


__global__ void kernelGetLap_minor(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,

	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_LapAz

//	f64 * __restrict__ p_Integratedconts_fromtri,
//	f64 * __restrict__ p_Integratedconts_fromvert,
//	f64 * __restrict__ p_Integratedconts_vert,
	//f64 * __restrict__ p_AreaMinor
	);
// debug why it is that we get sum of Lap nonzero when we integrate against AreaMinor, yet sum here to small


__global__ void kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor(

	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_minor,
	AAdot * __restrict__ p_AAdot,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,

	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just so we can handle insulator

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,
	f64 * __restrict__ p_LapAz,

	f64 * __restrict__ ROCAzduetoAdvection,
	f64 * __restrict__ ROCAzdotduetoAdvection,
	f64_vec2 * __restrict__ p_v_overall_minor,

	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_AreaMinor
);

__global__ void kernelInterpolateVarsAndPositions(
	f64 ppn,
	structural * __restrict__ p_info1,
	structural * __restrict__ p_info2,
	nvals * __restrict__ p_n_minor1,
	nvals * __restrict__ p_n_minor2,
	T3 * __restrict__ p_T_minor1,
	T3 * __restrict__ p_T_minor2,
//	f64_vec3 * __restrict__ p_B1,
//	f64_vec3 * __restrict__ p_B2,

	structural * __restrict__ p_info_dest,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor
	//f64_vec3 * __restrict__ p_B
	);

__global__ void kernelCreate_momflux_minor(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor,
	NTrates * __restrict__ NT_addition_tri
);

__global__ void kernelCreateLinearRelationship(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz, 
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_intermediate,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma
);

__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	f64_vec3 * __restrict__ p_v_n,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut
);


__global__ void kernelWrapTriangles(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,

	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	char * __restrict__ p_triPBClistaffected,

	CHAR4 * __restrict__ p_tri_periodic_corner_flags);

__global__ void kernelWrapVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	char * __restrict__ p_was_vertex_rotated);


__global__ void kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_neigh_index,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,
	CHAR4 * __restrict__ p_tri_periodic_neigh_flags,
	char * __restrict__ p_szPBC_triminor,
	char * __restrict__ p_triPBClistaffected
);
__global__ void kernelReset_szPBCtri_vert(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri_vert,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCtri_vert,
	char * __restrict__ p_szPBCneigh_vert,
	char * __restrict__ p_triPBClistaffected
);

__global__ void kernelPopulateOhmsLaw_debug(
	f64 h_use,

	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,
	nvals * __restrict__ p_n_minor_use,
	T3 * __restrict__ p_T_minor_use,

	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ ROCAzdotduetoAdvection,
	// Now going to need to go through and see this set 0 or sensible every time.

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave,
	bool const bUse_dest_n_for_Iz,
	nvals * __restrict__ p_n_dest_minor,
	f64 * __restrict__ p_dvez_friction,
	f64 * __restrict__ p_dvez_Ez
);

__global__ void kernelAccumulateDiffusiveHeatRateAndCalcIonisation_debug(
	f64 const h_use,
	structural * __restrict__ p_info_major,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,
	species3 * __restrict__ p_nu_major,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p_dTedtconduction,
	f64 * __restrict__ p_dTedtionisation
);


__global__ void kernelAdvanceDensityAndTemperature_debug(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	// Think we see the mistake here: are these to be major or minor values?
	// Major, right? Check code:

	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use,

	f64 * __restrict__ p_div_v_neut,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_Integrated_div_v_overall,
	f64 * __restrict__ p_AreaMajor, // hmm

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest,

	f64 * __restrict__ p_ei,
	f64 * __restrict__ p_en
);

__global__ void kernelAntiAdvect(
	f64 const h_use,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	AAdot * __restrict__ p_AAdot,
	f64_vec2 * __restrict__ p_v_overall_minor,
	AAdot * __restrict__ p_AAdot_dest
);

__global__ void kernelAccelerate_v_from_advection
(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_k,    // multiply by old mass ..
	f64 * __restrict__ p_AreaMinor_k,
	nvals * __restrict__ p_n_plus, // divide by new mass ..
	f64 * __restrict__ p_AreaMinor_plus,

	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_v_n_k,

	f64_vec3 * __restrict__ p_MAR_neut, // these contain the mom flux due to advection.
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	
	// outputs:
	v4 * __restrict__ p_vie_dest,
	f64_vec3 * __restrict__ p_v_n_dest);


__global__ void kernelSplitIntoSeedRegressors
(
	v4 * __restrict__ p_move,
	f64_vec3 * __restrict__ p_regr_i,
	f64_vec3 * __restrict__ p_regr_e,
	f64_vec2 * __restrict__ p_epsxy
);

__global__ void Subtract_V4(
	v4 * __restrict__ p_result,
	v4 * __restrict__ p_a,
	v4 * __restrict__ p_b
);

__global__ void kernelAdvanceDensityAndTemperature_nosoak_etc(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use,

	f64 * __restrict__ p_div_v_neut,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_Integrated_div_v_overall,
	f64 * __restrict__ p_AreaMajor, // hmm

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest
);

__global__ void kernelNeutral_momflux(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_v_n_minor,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut,

	NTrates * __restrict__ NT_addition_tri
);


__global__ void DivideNeTe_by_N(
	NTrates * __restrict__ NT_rates,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_dTbydt);


__global__ void DivideNeTeDifference_by_N(
	NTrates * __restrict__ NT_addition_rates_initial,
	NTrates * __restrict__ NT_addition_rates_final,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_dTbydt);


__global__ void kernelAdvanceDensityAndTemperature_noadvectioncompression(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,

	f64_vec3 * __restrict__ p_v_n_use, 
	f64 * __restrict__ p_AreaMajor,

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest,

	f64_vec3 * __restrict__ p_B_major
);
__global__ void kernelAdvanceDensityAndTemperature_noadvectioncompression_Copy(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,
	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest,
	f64_vec3 * __restrict__ p_B_major,
	f64 * __restrict__ p_Tgraph_resistive,
	f64 * __restrict__ p_Tgraph_other,
	f64 * __restrict__ p_Tgraph_total,
	f64 * __restrict__ p_Tgraph_dNT
);

__global__ void kernelNeutral_pressure(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	bool * __restrict__ bz_pressureflag,

	f64_vec3 * __restrict__ p_MAR_neut
);
__global__ void kernelCreateLinearRelationshipBwd_noadvect(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz,
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_k,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma
);

__global__ void kernelPopulateBackwardOhmsLaw_noadvect(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,
	nvals * __restrict__ p_n_minor_use,
	T3 * __restrict__ p_T_minor_use,
	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	//AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave);

__global__ void kernelCalculateVelocityAndAzdot_noadvect(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out);

__global__ void Reversesubtract_vec3(
	f64_vec3 * __restrict__ p_reverse,
	f64_vec3 * __restrict__ p_augmented);

__global__ void ScaleVector(
	f64 * __restrict__ output,
	f64 const coeff,
	f64 * __restrict__ multiply
);

__global__ void AssembleVector3(f64_vec3 * __restrict__ p_output,
	f64 * __restrict__ p_x, f64 * __restrict__ p_y, f64 * __restrict__ p_z);


__global__ void AddLCtoVector3component(
	f64_vec3 * __restrict__ p_operand,
	f64_vec3 * __restrict__ p_regr,
	int iDim,
	f64_vec3 * __restrict__ p_storemove);

__global__ void SubtractVector3stuff(f64_vec3 * __restrict__ p_output,
	f64 * __restrict__ p_x, f64 * __restrict__ p_y, f64 * __restrict__ p_z,
	f64_vec3 * __restrict__ p_to_subtract);

__global__ void kernelCollectOhmsGraphs(
	structural * __restrict__ p_info_major,
	f64_vec3 * __restrict__ p_MAR_ion_pressure_major,
	f64_vec3 * __restrict__ p_MAR_ion_visc_major,
	f64_vec3 * __restrict__ p_MAR_elec_pressure_major,  // need to distinguish viscous from pressure part.
	f64_vec3 * __restrict__ p_MAR_elec_visc_major,
	f64_vec3 * __restrict__ p_MAR_elec_ionization_major,
	f64_vec3 * __restrict__ p_B_major,

	v4 * __restrict__ p_vie_k, // ALL MAJOR
	v4 * __restrict__ p_vie_kplus,

	f64_vec2 * __restrict__ p_GradTe_major,
	nvals * __restrict__ p_n_major_use,
	T3 * __restrict__ p_T_major_use,

	AAdot * __restrict__ p_AAdot_kplus,
	f64 * __restrict__ p_AreaMinor, // EXCEPT THIS ONE

	f64 * __restrict__ p_Ohmsgraph_0, // elastic effective frictional coefficient zz
	f64 * __restrict__ p_Ohmsgraph_1, // ionization effective frictional coefficient zz
	f64 * __restrict__ p_Ohmsgraph_2, // 2 is combined y pressure accel rate
	f64 * __restrict__ p_Ohmsgraph_3,// 3 is q/(M+m) Ez -- do we have
	f64 * __restrict__ p_Ohmsgraph_4, // 4 is thermal force accel

	f64 * __restrict__ p_Ohmsgraph_5, // T_zy
	f64 * __restrict__ p_Ohmsgraph_6, // T_zz

	f64 * __restrict__ p_Ohmsgraph_7, // T acting on pressure
	f64 * __restrict__ p_Ohmsgraph_8, // T acting on electromotive
	f64 * __restrict__ p_Ohmsgraph_9, // T acting on thermal force
	f64 * __restrict__ p_Ohmsgraph_10, // prediction vez-viz

	f64 * __restrict__ p_Ohmsgraph_11, // difference of prediction from vez_k
	f64 * __restrict__ p_Ohmsgraph_12, // progress towards eqm: need vez_k+1
	f64 * __restrict__ p_Ohmsgraph_13, // viscous acceleration of electrons and ions (z)
	f64 * __restrict__ p_Ohmsgraph_14, // Prediction of Jz
	f64 * __restrict__ p_Ohmsgraph_15, // sigma zy
	f64 * __restrict__ p_Ohmsgraph_16, // sigma zz
	f64 * __restrict__ p_Ohmsgraph_17, // sigma zz times electromotive 
	f64 * __restrict__ p_Ohmsgraph_18 // Difference of prediction from Jz predicted.
);

__global__ void kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect(

	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_minor,
	AAdot * __restrict__ p_AAdot,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just so we can handle insulator

	bool * __restrict__ bz_pressureflag,

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,

	f64_vec3 * __restrict__ p_B
);

