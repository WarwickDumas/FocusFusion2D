#include "kernel.h"
#include "cuda_struct.h"
#include "constant.h"
#include "FFxtubes.h"

#define BWDSIDET
#define LONGITUDINAL

// TO DO:
// Line 1420:
// Yes, very much a waste. The edge positions should be calculated from the vertex positions, we can
// load flags to determine if it is an insulator-crossing triangle and that is the proper way to handle that.

#define TEST_OVERALL_V (0) //index == 38799)

#define FOUR_PI 12.5663706143592


__device__ void Augment_JacobeanNeutral(
	f64_tens3 * pJ,
	real Factor, //h_over (N m_i)
	f64_vec2 edge_normal,
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
) {
	
		//Pi_zx = -ita_par*(gradviz.x);
		//Pi_zy = -ita_par*(gradviz.y);		
		//	visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
		// The z direction doesn't feature vx --- that is because dvx/dz == 0

		pJ->xx += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);
		
		pJ->yy += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);

		pJ->zz += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);	


	// We are storing a whole matrix when it's just a scalar. !!!

}

__device__ void Augment_Jacobean(
	f64_tens3 * pJ, 
	real Factor, //h_over (N m_i)
	f64_vec2 edge_normal, 
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
) {
	if ((VISCMAG == 0) || (omega.dot(omega) < 0.01*0.1*nu*nu))
	{
		// run unmagnetised case

		//Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
		//Pi_xy = -ita_par*(gradvx.y + gradvy.x);
		//Pi_yx = Pi_xy;
		//Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
		//Pi_zx = -ita_par*(gradviz.x);
		//Pi_zy = -ita_par*(gradviz.y);		
		//	visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);


		// eps_x = vx_k+1 - vx_k - h MAR.x / N
		pJ->xx += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(4.0*grad_vjdx_coeff_on_vj_self)
				) *edge_normal.x + (
					//Pi_xy
					-ita_par*(grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yx += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(-2.0*grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		// The z direction doesn't feature vx --- that is because dvx/dz == 0

		pJ->xy += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(-2.0*grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_xy
					-ita_par*(grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yy += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdx_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(4.0*grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->zz += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);

		// In this way we develop let's say the J matrix, J for Jacobean
		// Then we could wish to do LU decomp of 4x4 matrix J so that
		// Regressor = J^-1 epsilon[loaded]
		// But we'll do 2 x 3 x 3.
	} else {

		f64 omegamod;
		f64_vec3 unit_b, unit_perp, unit_Hall;
		{
			f64 omegasq = omega.dot(omega);
			omegamod = sqrt(omegasq);
			unit_b = omega / omegamod;
			unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
			unit_perp = unit_perp / unit_perp.modulus();
			unit_Hall = unit_b.cross(unit_perp); // Note sign.
		}

		//f64 momflux_b, momflux_perp, momflux_Hall;
		f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
		f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
		f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
		f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));

		f64_vec3mag mag_edge;
		mag_edge.b = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
		mag_edge.P = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
		mag_edge.H = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

		// ==================================================================

		// Our approach is going to be to populate the "Flux Jacobean".
		// Let F_P = PI_Pb edgenormal_b + PI_PP edgenormal_P + PI_PH edgenormal_H

		// *********************
		//  Accumulate dF_b/dvx 
		// *********************

		f64_tens3mag F;
		memset(&F, 0, sizeof(f64_tens3mag));
		f64 bdotPsi = unit_b.x*grad_vjdx_coeff_on_vj_self + unit_b.y*grad_vjdy_coeff_on_vj_self;
		f64 PdotPsi = unit_perp.x*grad_vjdx_coeff_on_vj_self + unit_perp.y*grad_vjdy_coeff_on_vj_self;
		f64 HdotPsi = unit_Hall.x*grad_vjdx_coeff_on_vj_self + unit_Hall.y*grad_vjdy_coeff_on_vj_self;

		f64 d_Pi_by_dvx;
		// how to use union? Can just put in and out of scope.
		// d_Pi_bb_by_dvx =
		d_Pi_by_dvx = -ita_par*THIRD*(4.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi);

		F.bx += d_Pi_by_dvx * mag_edge.b; // Pi_bb

		d_Pi_by_dvx = -ita_2*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi)
			- ita_4*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi);
		// Pi_bP

		F.bx += d_Pi_by_dvx * mag_edge.P; // Pi_bP
		F.Px += d_Pi_by_dvx * mag_edge.b; // Pi_Pb

		d_Pi_by_dvx = -(ita_2*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi) + ita_4*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi));
		// Pi_bH

		F.bx += d_Pi_by_dvx * mag_edge.H; // Pi_bH 
		F.Hx += d_Pi_by_dvx * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_PP

		F.Px += d_Pi_by_dvx * mag_edge.P;

		d_Pi_by_dvx = -ita_1*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi) + ita_3*(unit_perp.x*PdotPsi - unit_Hall.x*HdotPsi);
		// Pi_PH

		F.Px += d_Pi_by_dvx * mag_edge.H;
		F.Hx += d_Pi_by_dvx * mag_edge.P; // Pi_PH

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			+ ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_HH

		F.Hx += d_Pi_by_dvx*mag_edge.H;

		// That was the x column.
		// Repeat exact same thing again replacing .x ..
		// first get it sensible.
		f64 d_Pi_by_dvy;

		// d_Pi_bb_by_dvy =
		d_Pi_by_dvy = -ita_par*THIRD*(4.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi);

		F.by += d_Pi_by_dvy * mag_edge.b; // Pi_bb

		d_Pi_by_dvy = -ita_2*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi)
			- ita_4*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi);
		// Pi_bP

		F.by += d_Pi_by_dvy * mag_edge.P; // Pi_bP
		F.Py += d_Pi_by_dvy * mag_edge.b; // Pi_Pb

		d_Pi_by_dvy = -(ita_2*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi) + ita_4*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi));
		// Pi_bH

		F.by += d_Pi_by_dvy * mag_edge.H; // Pi_bH 
		F.Hy += d_Pi_by_dvy * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_PP

		F.Py += d_Pi_by_dvy * mag_edge.P;

		d_Pi_by_dvy = -ita_1*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi) + ita_3*(unit_perp.y*PdotPsi - unit_Hall.y*HdotPsi);
		// Pi_PH

		F.Py += d_Pi_by_dvy * mag_edge.H;
		F.Hy += d_Pi_by_dvy * mag_edge.P; // Pi_PH

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			+ ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_HH

		F.Hy += d_Pi_by_dvy*mag_edge.H;

		f64 d_Pi_by_dvz;

		// d_Pi_bb_by_dvz =
		d_Pi_by_dvz = -ita_par*THIRD*(4.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi);

		F.bz += d_Pi_by_dvz * mag_edge.b; // Pi_bb

		d_Pi_by_dvz = -ita_2*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi)
			- ita_4*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi);
		// Pi_bP

		F.bz += d_Pi_by_dvz * mag_edge.P; // Pi_bP
		F.Pz += d_Pi_by_dvz * mag_edge.b; // Pi_Pb

		d_Pi_by_dvz = -(ita_2*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi) + ita_4*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi));
		// Pi_bH

		F.bz += d_Pi_by_dvz * mag_edge.H; // Pi_bH 
		F.Hz += d_Pi_by_dvz * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_PP

		F.Pz += d_Pi_by_dvz * mag_edge.P;

		d_Pi_by_dvz = -ita_1*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi) + ita_3*(unit_perp.z*PdotPsi - unit_Hall.z*HdotPsi);
		// Pi_PH

		F.Pz += d_Pi_by_dvz * mag_edge.H;
		F.Hz += d_Pi_by_dvz * mag_edge.P; // Pi_PH

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			+ ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_HH

		F.Hz += d_Pi_by_dvz*mag_edge.H;

		// *************************
		//  Now use it to create J
		// *************************

		pJ->xx += Factor*(unit_b.x*F.bx + unit_perp.x*F.Px + unit_Hall.x*F.Hx);
		pJ->xy += Factor*(unit_b.x*F.by + unit_perp.x*F.Py + unit_Hall.x*F.Hy); // d eps x / d vy
		pJ->xz += Factor*(unit_b.x*F.bz + unit_perp.x*F.Pz + unit_Hall.x*F.Hz);

		pJ->yx += Factor*(unit_b.y*F.bx + unit_perp.y*F.Px + unit_Hall.y*F.Hx);
		pJ->yy += Factor*(unit_b.y*F.by + unit_perp.y*F.Py + unit_Hall.y*F.Hy);
		pJ->yz += Factor*(unit_b.y*F.bz + unit_perp.y*F.Pz + unit_Hall.y*F.Hz);

		pJ->zx += Factor*(unit_b.z*F.bx + unit_perp.z*F.Px + unit_Hall.z*F.Hx);
		pJ->zy += Factor*(unit_b.z*F.by + unit_perp.z*F.Py + unit_Hall.z*F.Hy);
		pJ->zz += Factor*(unit_b.z*F.bz + unit_perp.z*F.Pz + unit_Hall.z*F.Hz);

	}
}

__global__ void kernelUnpack(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	pTn[iVertex] = T.Tn;
	pTi[iVertex] = T.Ti;
	pTe[iVertex] = T.Te;
}

__global__ void kernelUnpackWithMask(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock
	)
{	
	if (p_bMaskblock[blockIdx.x] == 0) return;
	
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;

	T3 T = pT[iVertex];
	if (p_bMask[iVertex]) pTn[iVertex] = T.Tn;
	if (p_bMask[iVertex + NUMVERTICES]) pTi[iVertex] = T.Ti;
	if (p_bMask[iVertex + 2 * NUMVERTICES]) pTe[iVertex] = T.Te;
}


__global__ void kernelUnpacktorootDN_T(
	f64 * __restrict__ psqrtDNnTn,
	f64 * __restrict__ psqrtDNTi,
	f64 * __restrict__ psqrtDNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_D_n,
	f64 * __restrict__ p_D_i,
	f64 * __restrict__ p_D_e,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	psqrtDNnTn[iVertex] = T.Tn*sqrt(p_D_n[iVertex]*AreaMajor*n.n_n);
	psqrtDNTi[iVertex] = T.Ti*sqrt(p_D_i[iVertex]*AreaMajor*n.n);
	psqrtDNTe[iVertex] = T.Te*sqrt(p_D_e[iVertex]*AreaMajor*n.n);
}


__global__ void kernelUnpacktorootNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	pNnTn[iVertex] = T.Tn*sqrt(AreaMajor*n.n_n);
	pNTi[iVertex] = T.Ti*sqrt(AreaMajor*n.n);
	pNTe[iVertex] = T.Te*sqrt(AreaMajor*n.n);
}
__global__ void kernelUnpacktoNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	pNnTn[iVertex] = T.Tn*AreaMajor*n.n_n;
	pNTi[iVertex] = T.Ti*AreaMajor*n.n;
	pNTe[iVertex] = T.Te*AreaMajor*n.n;
}

__global__ void NegateVectors(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_x2, f64 * __restrict__ p_x3)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	p_x1[iVertex] = -p_x1[iVertex];
	p_x2[iVertex] = -p_x2[iVertex];
	p_x3[iVertex] = -p_x3[iVertex];
}

__global__ void NegateVector(
	f64 * __restrict__ p_x1)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	p_x1[iVertex] = -p_x1[iVertex];
}

__global__ void SubtractT3(
	T3 * __restrict__ p_result,
	T3 * __restrict__ p_a, T3 * __restrict__ p_b)
{
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	T3 result;
	T3 T_1 = p_a[index];
	T3 T_2 = p_b[index];
	result.Tn = T_1.Tn - T_2.Tn;
	result.Ti = T_1.Ti - T_2.Ti;
	result.Te = T_1.Te - T_2.Te;
	p_result[index] = result;
}

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
)
{
	__shared__ f64 sumdata_eps_i[threadsPerTileMinor];
	__shared__ f64 sumdata_eps_e[threadsPerTileMinor];
	__shared__ f64 sumdata_ii[threadsPerTileMinor];
	__shared__ f64 sumdata_ee[threadsPerTileMinor];
	__shared__ f64 sumdata_ei[threadsPerTileMinor];
	__shared__ f64 sumdata_ss[threadsPerTileMinor];

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;
	
	sumdata_eps_i[threadIdx.x] = 0.0;
	sumdata_eps_e[threadIdx.x] = 0.0;
	sumdata_ii[threadIdx.x] = 0.0;
	sumdata_ee[threadIdx.x] = 0.0;
	sumdata_ei[threadIdx.x] = 0.0;
	sumdata_ss[threadIdx.x] = 0.0;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64_vec2 eps_xy = p_eps_xy[iMinor];
		f64 eps_iz = p_eps_iz[iMinor];
		f64 eps_ez = p_eps_ez[iMinor];
		
		f64_vec2 depsxybydbeta_i = p_d_epsxy_by_d_beta_i_[iMinor];
		f64_vec2 depsxybydbeta_e = p_d_epsxy_by_d_beta_e_[iMinor];
		f64 deps_iz_bydbeta_i = p_d_eps_iz_by_d_beta_i_[iMinor];
		f64 deps_ez_bydbeta_i = p_d_eps_ez_by_d_beta_i_[iMinor];
		f64 deps_iz_bydbeta_e = p_d_eps_iz_by_d_beta_e_[iMinor];
		f64 deps_ez_bydbeta_e = p_d_eps_ez_by_d_beta_e_[iMinor];
		
		sumdata_eps_i[threadIdx.x] = depsxybydbeta_i.x * eps_xy.x
			+ depsxybydbeta_i.y*eps_xy.y + deps_iz_bydbeta_i*eps_iz + deps_ez_bydbeta_i*eps_ez;

		sumdata_eps_e[threadIdx.x] = depsxybydbeta_e.x * eps_xy.x
			+ depsxybydbeta_e.y*eps_xy.y + deps_iz_bydbeta_e*eps_iz + deps_ez_bydbeta_e*eps_ez;

		sumdata_ii[threadIdx.x] = depsxybydbeta_i.dot(depsxybydbeta_i) + deps_iz_bydbeta_i*deps_iz_bydbeta_i
								+ deps_ez_bydbeta_i*deps_ez_bydbeta_i;

		sumdata_ee[threadIdx.x] = depsxybydbeta_e.dot(depsxybydbeta_e) + deps_iz_bydbeta_e*deps_iz_bydbeta_e
								+ deps_ez_bydbeta_e*deps_ez_bydbeta_e;

		sumdata_ei[threadIdx.x] = depsxybydbeta_e.x*depsxybydbeta_i.x
								+ depsxybydbeta_e.y*depsxybydbeta_i.y
								+ deps_iz_bydbeta_i*deps_iz_bydbeta_e
								+ deps_ez_bydbeta_i*deps_ez_bydbeta_e; // NO z COMPONENT.

		sumdata_ss[threadIdx.x] = eps_xy.dot(eps_xy) + eps_iz*eps_iz + eps_ez*eps_ez;
	}
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata_eps_i[threadIdx.x] += sumdata_eps_i[threadIdx.x + k];
			sumdata_eps_e[threadIdx.x] += sumdata_eps_e[threadIdx.x + k];
			sumdata_ii[threadIdx.x] += sumdata_ii[threadIdx.x + k];
			sumdata_ee[threadIdx.x] += sumdata_ee[threadIdx.x + k];
			sumdata_ei[threadIdx.x] += sumdata_ei[threadIdx.x + k];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata_eps_i[threadIdx.x] += sumdata_eps_i[threadIdx.x + s - 1];
			sumdata_eps_e[threadIdx.x] += sumdata_eps_e[threadIdx.x + s - 1];
			sumdata_ii[threadIdx.x] += sumdata_ii[threadIdx.x + s - 1];
			sumdata_ee[threadIdx.x] += sumdata_ee[threadIdx.x + s - 1];
			sumdata_ei[threadIdx.x] += sumdata_ei[threadIdx.x + s - 1];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_deps_by_dbeta_i_[blockIdx.x] = sumdata_eps_i[0];
		p_sum_eps_deps_by_dbeta_e_[blockIdx.x] = sumdata_eps_e[0];
		p_sum_depsbydbeta_i_times_i_[blockIdx.x] = sumdata_ii[0];
		p_sum_depsbydbeta_e_times_e_[blockIdx.x] = sumdata_ee[0];
		p_sum_depsbydbeta_e_times_i_[blockIdx.x] = sumdata_ei[0];
		p_sum_eps_sq[blockIdx.x] = sumdata_ss[0];			
	};
}


__global__ void AccumulateSummandsScalars3(
	structural * __restrict__ p_info_minor,
	f64 * __restrict__ p__eps,
	f64 * __restrict__ p__deps_1,
	f64 * __restrict__ p__deps_2, 
	f64 * __restrict__ p__deps_3,
	f64_vec3 * __restrict__ p_sum_eps_deps_,
	Symmetric3 * __restrict__ p_sum_product_matrix_,
	f64 * __restrict__ p_sum_eps_sq
) {
	__shared__ f64 sumdata_eps_depsx[threadsPerTileMinor];
	__shared__ f64 sumdata_eps_depsy[threadsPerTileMinor];
	__shared__ f64 sumdata_eps_depsz[threadsPerTileMinor];
	__shared__ Symmetric3 sum_product[threadsPerTileMinor];
	__shared__ f64 sumdata_ss[threadsPerTileMinor];

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;

	sumdata_eps_depsx[threadIdx.x] = 0.0;
	sumdata_eps_depsy[threadIdx.x] = 0.0;
	sumdata_eps_depsz[threadIdx.x] = 0.0;
	memset(&(sum_product[threadIdx.x]), 0, sizeof(Symmetric3));
	sumdata_ss[threadIdx.x] = 0.0;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64 eps = p__eps[iMinor];
		f64 depsbydbeta_x = p__deps_1[iMinor];
		f64 depsbydbeta_y = p__deps_2[iMinor];
		f64 depsbydbeta_z = p__deps_3[iMinor];
		
		// sum over all values, of deps_i/dbetax deps_i/dbetay

		sumdata_eps_depsx[threadIdx.x] = depsbydbeta_x * eps;
		sumdata_eps_depsy[threadIdx.x] = depsbydbeta_y * eps;
		sumdata_eps_depsz[threadIdx.x] = depsbydbeta_z * eps;

		sum_product[threadIdx.x].xx = depsbydbeta_x *depsbydbeta_x;
		sum_product[threadIdx.x].xy = depsbydbeta_x *depsbydbeta_y;
		sum_product[threadIdx.x].xz = depsbydbeta_x *depsbydbeta_z;
		sum_product[threadIdx.x].yy = depsbydbeta_y *depsbydbeta_y;
		sum_product[threadIdx.x].yz = depsbydbeta_y *depsbydbeta_z;
		sum_product[threadIdx.x].zz = depsbydbeta_z *depsbydbeta_z;

		sumdata_ss[threadIdx.x] = eps*eps;
	}

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata_eps_depsx[threadIdx.x] += sumdata_eps_depsx[threadIdx.x + k];
			sumdata_eps_depsy[threadIdx.x] += sumdata_eps_depsy[threadIdx.x + k];
			sumdata_eps_depsz[threadIdx.x] += sumdata_eps_depsz[threadIdx.x + k];

			sum_product[threadIdx.x].xx += sum_product[threadIdx.x + k].xx;
			sum_product[threadIdx.x].xy += sum_product[threadIdx.x + k].xy;
			sum_product[threadIdx.x].xz += sum_product[threadIdx.x + k].xz;
			sum_product[threadIdx.x].yz += sum_product[threadIdx.x + k].yz;
			sum_product[threadIdx.x].yy += sum_product[threadIdx.x + k].yy;
			sum_product[threadIdx.x].zz += sum_product[threadIdx.x + k].zz;

			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata_eps_depsx[threadIdx.x] += sumdata_eps_depsx[threadIdx.x + s - 1];
			sumdata_eps_depsy[threadIdx.x] += sumdata_eps_depsy[threadIdx.x + s - 1];
			sumdata_eps_depsz[threadIdx.x] += sumdata_eps_depsz[threadIdx.x + s - 1];
			sum_product[threadIdx.x].xx += sum_product[threadIdx.x + s - 1].xx;
			sum_product[threadIdx.x].xy += sum_product[threadIdx.x + s - 1].xy;
			sum_product[threadIdx.x].xz += sum_product[threadIdx.x + s - 1].xz;
			sum_product[threadIdx.x].yz += sum_product[threadIdx.x + s - 1].yz;
			sum_product[threadIdx.x].yy += sum_product[threadIdx.x + s - 1].yy;
			sum_product[threadIdx.x].zz += sum_product[threadIdx.x + s - 1].zz;
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		f64_vec3 sum_eps_deps;
		sum_eps_deps.x = sumdata_eps_depsx[0];
		sum_eps_deps.y = sumdata_eps_depsy[0];
		sum_eps_deps.z = sumdata_eps_depsz[0];

		p_sum_eps_deps_[blockIdx.x] = sum_eps_deps;
		memcpy(&(p_sum_product_matrix_[blockIdx.x]), &(sum_product[0]), sizeof(Symmetric3));
		p_sum_eps_sq[blockIdx.x] = sumdata_ss[0];
	};
}

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
	)
	{
		__shared__ f64 sumdata_eps_depsx[threadsPerTileMinor];
		__shared__ f64 sumdata_eps_depsy[threadsPerTileMinor];
		__shared__ f64 sumdata_eps_depsz[threadsPerTileMinor];
		__shared__ Symmetric3 sum_product[threadsPerTileMinor];
		__shared__ f64 sumdata_ss[threadsPerTileMinor];

		long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;

		sumdata_eps_depsx[threadIdx.x] = 0.0;
		sumdata_eps_depsy[threadIdx.x] = 0.0;
		sumdata_eps_depsz[threadIdx.x] = 0.0;
		memset(&(sum_product[threadIdx.x]), 0, sizeof(Symmetric3));
		sumdata_ss[threadIdx.x] = 0.0;

		structural info = p_info_minor[iMinor];
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
		{
			f64_vec3 eps = p_eps3[iMinor];
			f64_vec3 depsbydbeta_x = p_d_eps_by_d_beta_x_[iMinor];
			f64_vec3 depsbydbeta_y = p_d_eps_by_d_beta_y_[iMinor];
			f64_vec3 depsbydbeta_z = p_d_eps_by_d_beta_z_[iMinor];


			// sum over all values, of deps_i/dbetax deps_i/dbetay

			sumdata_eps_depsx[threadIdx.x] = depsbydbeta_x.x * eps.x + depsbydbeta_x.y*eps.y + depsbydbeta_x.z*eps.z;
			sumdata_eps_depsy[threadIdx.x] = depsbydbeta_y.x * eps.x + depsbydbeta_y.y*eps.y + depsbydbeta_y.z*eps.z;
			sumdata_eps_depsz[threadIdx.x] = depsbydbeta_z.x * eps.x + depsbydbeta_z.y*eps.y + depsbydbeta_z.z*eps.z;

			sum_product[threadIdx.x].xx = depsbydbeta_x.x *depsbydbeta_x.x
				+ depsbydbeta_x.y *depsbydbeta_x.y
				+ depsbydbeta_x.z *depsbydbeta_x.z;
			sum_product[threadIdx.x].xy = depsbydbeta_x.x *depsbydbeta_y.x
				+ depsbydbeta_x.y *depsbydbeta_y.y
				+ depsbydbeta_x.z *depsbydbeta_y.z;
			sum_product[threadIdx.x].xz = depsbydbeta_x.x *depsbydbeta_z.x
				+ depsbydbeta_x.y *depsbydbeta_z.y
				+ depsbydbeta_x.z *depsbydbeta_z.z;
			sum_product[threadIdx.x].yy = depsbydbeta_y.x *depsbydbeta_y.x
				+ depsbydbeta_y.y *depsbydbeta_y.y
				+ depsbydbeta_y.z *depsbydbeta_y.z;
			sum_product[threadIdx.x].yz = depsbydbeta_y.x *depsbydbeta_z.x
				+ depsbydbeta_y.y *depsbydbeta_z.y
				+ depsbydbeta_y.z *depsbydbeta_z.z;
			sum_product[threadIdx.x].zz = depsbydbeta_z.x *depsbydbeta_z.x
				+ depsbydbeta_z.y *depsbydbeta_z.y
				+ depsbydbeta_z.z *depsbydbeta_z.z;

			sumdata_ss[threadIdx.x] = eps.dot(eps);
		}

		__syncthreads();

		int s = blockDim.x;
		int k = s / 2;

		while (s != 1) {
			if (threadIdx.x < k)
			{
				sumdata_eps_depsx[threadIdx.x] += sumdata_eps_depsx[threadIdx.x + k];
				sumdata_eps_depsy[threadIdx.x] += sumdata_eps_depsy[threadIdx.x + k];
				sumdata_eps_depsz[threadIdx.x] += sumdata_eps_depsz[threadIdx.x + k];

				sum_product[threadIdx.x].xx += sum_product[threadIdx.x + k].xx;
				sum_product[threadIdx.x].xy += sum_product[threadIdx.x + k].xy;
				sum_product[threadIdx.x].xz += sum_product[threadIdx.x + k].xz;
				sum_product[threadIdx.x].yz += sum_product[threadIdx.x + k].yz;
				sum_product[threadIdx.x].yy += sum_product[threadIdx.x + k].yy;
				sum_product[threadIdx.x].zz += sum_product[threadIdx.x + k].zz;

				sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
			};
			__syncthreads();

			// Modify for case blockdim not 2^n:
			if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
				sumdata_eps_depsx[threadIdx.x] += sumdata_eps_depsx[threadIdx.x + s - 1];
				sumdata_eps_depsy[threadIdx.x] += sumdata_eps_depsy[threadIdx.x + s - 1];
				sumdata_eps_depsz[threadIdx.x] += sumdata_eps_depsz[threadIdx.x + s - 1];
				sum_product[threadIdx.x].xx += sum_product[threadIdx.x + s - 1].xx;
				sum_product[threadIdx.x].xy += sum_product[threadIdx.x + s - 1].xy;
				sum_product[threadIdx.x].xz += sum_product[threadIdx.x + s - 1].xz;
				sum_product[threadIdx.x].yz += sum_product[threadIdx.x + s - 1].yz;
				sum_product[threadIdx.x].yy += sum_product[threadIdx.x + s - 1].yy;
				sum_product[threadIdx.x].zz += sum_product[threadIdx.x + s - 1].zz;
			};
			// In case k == 81, add [39] += [80]
			// Otherwise we only get to 39+40=79.
			s = k;
			k = s / 2;
			__syncthreads();
		};

		if (threadIdx.x == 0)
		{
			f64_vec3 sum_eps_deps;
			sum_eps_deps.x = sumdata_eps_depsx[0];
			sum_eps_deps.y = sumdata_eps_depsy[0];
			sum_eps_deps.z = sumdata_eps_depsz[0];

			p_sum_eps_deps_[blockIdx.x] = sum_eps_deps;
			memcpy(&(p_sum_product_matrix_[blockIdx.x]), &(sum_product[0]), sizeof(Symmetric3));
			p_sum_eps_sq[blockIdx.x] = sumdata_ss[0]; 			
		};
	}

__global__ void Subtract(
		f64 * __restrict__ p_c,
		f64 * __restrict__ p_b,
		f64 * __restrict__ p_a
	) {
		long const index = threadIdx.x + blockIdx.x * threadsPerTileMinor;
		p_c[index] = p_b[index] - p_a[index];
}

__global__ void Subtract_xy(
	f64_vec2 * __restrict__ p_c,
	f64_vec2 * __restrict__ p_b,
	f64_vec2 * __restrict__ p_a
) {
	long const index = threadIdx.x + blockIdx.x * threadsPerTileMinor;
	p_c[index] = p_b[index] - p_a[index];
}


__global__ void kernelAccumulateSummandsNeutVisc2(
		f64 * __restrict__ p_eps,  // 
		f64 * __restrict__ p_d_eps_by_d_beta,
		// outputs:
		f64 * __restrict__ p_sum_eps_deps_,  // 8 values for this block
		f64 * __restrict__ p_sum_product_matrix_
)
{
	__shared__ f64 sumdata_eps_deps[threadsPerTileMinor/4][REGRESSORS];
	__shared__ f64 sum_product[threadsPerTileMinor/4][REGRESSORS][REGRESSORS];
	// Call with threadsPerTileMinor/4

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	long const iMinor = threadIdx.x + blockIdx.x * threadsPerTileMinor;
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	f64 depsbydbeta[REGRESSORS], eps;
	int i, j;
	memset(&(sumdata_eps_deps[threadIdx.x]), 0, sizeof(f64)*REGRESSORS);
	memset(&(sum_product[threadIdx.x]), 0, sizeof(f64)*REGRESSORS*REGRESSORS);

		
	eps = p_eps[iMinor];
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		depsbydbeta[i] = p_d_eps_by_d_beta[iMinor + i*NMINOR];
	};
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		sumdata_eps_deps[threadIdx.x][i] = depsbydbeta[i] * eps;
		for (j = 0; j < REGRESSORS; j++)
			sum_product[threadIdx.x][i][j] = depsbydbeta[i] * depsbydbeta[j];
	};				
		

	eps = p_eps[iMinor + threadsPerTileMinor/4];
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		depsbydbeta[i] = p_d_eps_by_d_beta[iMinor + threadsPerTileMinor / 4 + i*NMINOR];
	};
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		sumdata_eps_deps[threadIdx.x][i] += depsbydbeta[i] * eps;
		for (j = 0; j < REGRESSORS; j++)
			sum_product[threadIdx.x][i][j] += depsbydbeta[i] * depsbydbeta[j];
	};

	eps = p_eps[iMinor + threadsPerTileMinor / 2];
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		depsbydbeta[i] = p_d_eps_by_d_beta[iMinor + threadsPerTileMinor / 2 + i*NMINOR];
	};
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		sumdata_eps_deps[threadIdx.x][i] += depsbydbeta[i] * eps;
		for (j = 0; j < REGRESSORS; j++)
			sum_product[threadIdx.x][i][j] += depsbydbeta[i] * depsbydbeta[j];
	};

	eps = p_eps[iMinor + 3*threadsPerTileMinor / 4];
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		depsbydbeta[i] = p_d_eps_by_d_beta[iMinor + 3*threadsPerTileMinor / 4 + i*NMINOR];
	};
#pragma unroll
	for (i = 0; i < REGRESSORS; i++)
	{
		sumdata_eps_deps[threadIdx.x][i] += depsbydbeta[i] * eps;
		for (j = 0; j < REGRESSORS; j++)
			sum_product[threadIdx.x][i][j] += depsbydbeta[i] * depsbydbeta[j];
	};



	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (i = 0; i < REGRESSORS; i++)
			{
				sumdata_eps_deps[threadIdx.x][i] += sumdata_eps_deps[threadIdx.x + k][i];
				for (j = 0; j < REGRESSORS; j++)
					sum_product[threadIdx.x][i][j] += sum_product[threadIdx.x + k][i][j];
			};				
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (i = 0; i < REGRESSORS; i++)
			{
				sumdata_eps_deps[threadIdx.x][i] += sumdata_eps_deps[threadIdx.x + s - 1][i];
				for (j = 0; j < REGRESSORS; j++)
					sum_product[threadIdx.x][i][j] += sum_product[threadIdx.x + s - 1][i][j];
			};				
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_deps_[blockIdx.x*REGRESSORS]), sumdata_eps_deps[0], sizeof(f64)*REGRESSORS);
		memcpy(&(p_sum_product_matrix_[blockIdx.x*REGRESSORS*REGRESSORS]), &(sum_product[0][0][0]), sizeof(f64)*REGRESSORS*REGRESSORS);
	};
}

__global__ void Vector3Breakdown(
	f64_vec3 * __restrict__ p_input,
	f64_vec2 * __restrict__ p_outxy,
	f64 * __restrict__ p_outz,
	int * __restrict__ p_Select
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec3 vec3;
	memset(&vec3, 0, sizeof(f64_vec3));
	if (p_Select[index] != 0) {
		vec3 = p_input[index];
	};
	p_outz[index] = vec3.z;
	f64_vec2 xy; xy.x = vec3.x; xy.y = vec3.y;
	p_outxy[index] = xy;
}

__global__ void AddLittleBitORegressors(
	f64 const coeff,
	v4 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p__regrxy,
	f64 * __restrict__ p__regriz,
	f64 * __restrict__ p__regrez)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	v4 operand = p_operand[index];

	operand.vxy += coeff*p__regrxy[index];
	operand.viz += coeff*p__regriz[index];
	operand.vez += coeff*p__regrez[index];

	if (index % 20000 == 0) printf("%d regrez %1.14E operand.vez %1.14E\n", index, p__regrez[index],
		operand.vez);

	p_operand[index] = operand;
}

__global__ void AddLCtoVector4component(
	v4 * __restrict__ p_operand,
	v4 * __restrict__ p_regr,
	v4 * __restrict__ p_storemove)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	v4 operand = p_operand[index];
	v4 old = operand;
	v4 move;
	int i, j;
	for (i = 0; i < REGRESSORS; i++)
	{
		operand.vxy += beta_n_c[i] * p_regr[index + i*NMINOR].vxy;
		operand.viz += beta_n_c[i] * p_regr[index + i*NMINOR].viz;
		operand.vez += beta_n_c[i] * p_regr[index + i*NMINOR].vez;
	//	if (index == CHOSEN)
	//		printf("operand.vez %d beta %1.8E regr %1.8E \n",
	//			operand.vez, beta_n_c[i], p_regr[index + i*NMINOR].vez);

	};
	move.vxy = operand.vxy - old.vxy;
	move.viz = operand.viz - old.viz;
	move.vez = operand.vez - old.vez;
	p_storemove[index] = move;
	p_operand[index] = operand;
}



__global__ void AddLCtoVector3component(
	f64_vec3 * __restrict__ p_operand,
	f64_vec3 * __restrict__ p_regr,
	int iDim,
	f64_vec3 * __restrict__ p_storemove)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 operand = p_operand[index];
	f64_vec3 old = operand;
	f64 move;
	int i, j;
	switch (iDim)
	{
	case 0:
		for (i = 0; i < REGRESSORS; i++)
			operand.x += beta_n_c[i] * p_regr[index+i*NMINOR].x;
		move = operand.x - old.x;
		p_storemove[index].x = move;
		break;
	case 1:
		for (i = 0; i < REGRESSORS; i++)
			operand.y += beta_n_c[i] * p_regr[index + i*NMINOR].y;
		move = operand.y - old.y;
		p_storemove[index].y = move;
		break;
	case 2:
		for (i = 0; i < REGRESSORS; i++)
			operand.z += beta_n_c[i] * p_regr[index + i*NMINOR].z;
		move = operand.z - old.z;
		p_storemove[index].z = move;
		break;
	}
	p_operand[index] = operand;
}

__global__ void kernelSetx(f64_vec3 * __restrict__ p_v1,
		f64_vec3 * __restrict__ p_src)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 src = p_src[index];
	f64_vec3 v(src.x, 0.0, 0.0);
	p_v1[index] = v;
}

__global__ void kernelSety(f64_vec3 * __restrict__ p_v1,
	f64_vec3 * __restrict__ p_src)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 src = p_src[index];
	f64_vec3 v(0.0, src.y, 0.0);
	p_v1[index] = v;
}
__global__ void kernelSetz(f64_vec3 * __restrict__ p_v1,
	f64_vec3 * __restrict__ p_src)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 src = p_src[index];
	f64_vec3 v(0.0, 0.0, src.z);
	p_v1[index] = v;
}
__global__ void kernelAddLC_vec3
	(f64_vec3 * __restrict__ p_vec,
		f64_vec3 coeff,
		f64_vec3 * __restrict__ p_addition)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 v = p_vec[index];
	f64_vec3 add = p_addition[index];
	v.x += coeff.x*add.x;
	v.y += coeff.y*add.y;
	v.z += coeff.z*add.z;
	p_vec[index] = v;
}

__global__ void ResettoGeomAverage(
	f64_vec3 * __restrict__ p_result,
	f64_vec3 * __restrict__ p_mult1)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 m1 = p_mult1[index];
	f64_vec3 result = p_result[index];
	result.x = m1.x*result.x;
	if (result.x > 0.0) result.x = sqrt(result.x);
	if (result.x < 0.0) result.x = -sqrt(-result.x);

	result.y = (m1.y*result.y);
	if (result.y > 0.0) result.y = sqrt(result.y);
	if (result.y < 0.0) result.y = -sqrt(-result.y);

	result.z = m1.z*result.z;
	if (result.z > 0.0) result.z = sqrt(result.z);
	if (result.z < 0.0) result.z = -sqrt(-result.z);

	p_result[index] = result;
}
__global__ void SetProduct3(
	f64_vec3 * __restrict__ p_result,
	f64_vec3 * __restrict__ p_mult1,
	f64_vec3 * __restrict__ p_mult2)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 m1 = p_mult1[index];
	f64_vec3 m2 = p_mult2[index];
	f64_vec3 result;
	result.x = m1.x*m2.x;
	result.y = m1.y*m2.y;
	result.z = m1.z*m2.z;
	p_result[index] = result;
}

__global__ void kernelAddLC_3vec3
(f64_vec3 * __restrict__ p_vec,
	f64_vec3 coeffs_x, 
	f64_vec3 coeffs_y, 
	f64_vec3 coeffs_z,
	f64_vec3 * __restrict__ p_addition1,
	f64_vec3 * __restrict__ p_addition2,
	f64_vec3 * __restrict__ p_addition3
	)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec3 v = p_vec[index];
	f64_vec3 add1 = p_addition1[index];
	f64_vec3 add2 = p_addition2[index];
	f64_vec3 add3 = p_addition3[index];
	v.x += coeffs_x.x*add1.x + coeffs_x.y*add2.x + coeffs_x.z*add3.x;
	v.y += coeffs_y.x*add1.y + coeffs_y.y*add2.y + coeffs_y.z*add3.y;
	v.z += coeffs_z.x*add1.z + coeffs_z.y*add2.z + coeffs_z.z*add3.z;
	p_vec[index] = v;
}

__global__ void kernelAccumulateSummands4(

	// We don't need to test for domain, we need to make sure the summands are zero otherwise.
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_d_beta_J_,
	f64 * __restrict__ p_d_eps_by_d_beta_R_,

	f64 * __restrict__ p_sum_eps_deps_by_dbeta_J_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_J_,
	f64 * __restrict__ p_sum_depsbydbeta_R_times_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_R_,
	f64 * __restrict__ p_sum_eps_sq
)
{
	__shared__ f64 sumdata_eps_J[threadsPerTileMajor];
	__shared__ f64 sumdata_eps_R[threadsPerTileMajor];
	__shared__ f64 sumdata_JJ[threadsPerTileMajor];
	__shared__ f64 sumdata_RR[threadsPerTileMajor];
	__shared__ f64 sumdata_JR[threadsPerTileMajor];
	__shared__ f64 sumdata_ss[threadsPerTileMajor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;

	sumdata_eps_J[threadIdx.x] = 0.0;
	sumdata_eps_R[threadIdx.x] = 0.0;
	sumdata_JJ[threadIdx.x] = 0.0;
	sumdata_RR[threadIdx.x] = 0.0;
	sumdata_JR[threadIdx.x] = 0.0;
	sumdata_ss[threadIdx.x] = 0.0;


	//structural info = p_info_minor[iMinor];
	//if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64 eps = p_epsilon[iVertex];
		f64 depsbydbeta_J = p_d_eps_by_d_beta_J_[iVertex];
		f64 depsbydbeta_R = p_d_eps_by_d_beta_R_[iVertex];

		sumdata_eps_J[threadIdx.x] = depsbydbeta_J * eps;
		sumdata_eps_R[threadIdx.x] = depsbydbeta_R * eps;
		sumdata_JJ[threadIdx.x] = depsbydbeta_J*depsbydbeta_J;
		sumdata_RR[threadIdx.x] = depsbydbeta_R*depsbydbeta_R;
		sumdata_JR[threadIdx.x] = depsbydbeta_J*depsbydbeta_R;
		sumdata_ss[threadIdx.x] = eps*eps;
	}

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata_eps_J[threadIdx.x] += sumdata_eps_J[threadIdx.x + k];
			sumdata_eps_R[threadIdx.x] += sumdata_eps_R[threadIdx.x + k];
			sumdata_JJ[threadIdx.x] += sumdata_JJ[threadIdx.x + k];
			sumdata_RR[threadIdx.x] += sumdata_RR[threadIdx.x + k];
			sumdata_JR[threadIdx.x] += sumdata_JR[threadIdx.x + k];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata_eps_J[threadIdx.x] += sumdata_eps_J[threadIdx.x + s - 1];
			sumdata_eps_R[threadIdx.x] += sumdata_eps_R[threadIdx.x + s - 1];
			sumdata_JJ[threadIdx.x] += sumdata_JJ[threadIdx.x + s - 1];
			sumdata_RR[threadIdx.x] += sumdata_RR[threadIdx.x + s - 1];
			sumdata_JR[threadIdx.x] += sumdata_JR[threadIdx.x + s - 1];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_deps_by_dbeta_J_[blockIdx.x] = sumdata_eps_J[0];
		p_sum_eps_deps_by_dbeta_R_[blockIdx.x] = sumdata_eps_R[0];
		p_sum_depsbydbeta_J_times_J_[blockIdx.x] = sumdata_JJ[0];
		p_sum_depsbydbeta_R_times_R_[blockIdx.x] = sumdata_RR[0];
		p_sum_depsbydbeta_J_times_R_[blockIdx.x] = sumdata_JR[0];
		p_sum_eps_sq[blockIdx.x] = sumdata_ss[0];
	};
}


__global__ void kernelAccumulateSummands7(
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	// outputs:
	f64 * __restrict__ p_sum_eps_depsbydbeta_x8,
	f64 * __restrict__ p_sum_depsbydbeta__8x8
) {
	__shared__ f64 sumdata[threadsPerTileMajor][24];
	// Row-major memory layout implies that this is a contiguous array for each thread.

	// We can have say 24 doubles in shared. We need to sum 64 + 8 + 1 = 73 things. 24*3 = 72. hah!
	// It would be nicer then if we just called this multiple times. But it has to be for distinct input data..
	// Note that given threadsPerTileMajor = 128 we could comfortably put 48 doubles in shared and still run something.

	// The inputs are only 9 doubles so we can have them.
	// We only need the upper matrix. 1 + 2 + 3 + 4 +5 +6+7+8+9 = 45
	// So we can do it in 2 goes.

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	f64 eps = p_epsilon[iVertex];
	f64 d_eps_by_d_beta[REGRESSORS];
	int i;
	for (i = 0; i < REGRESSORS; i++) {
		d_eps_by_d_beta[i] = p_d_eps_by_dbeta[iVertex + i*NUMVERTICES];
		if (d_eps_by_d_beta[i] != d_eps_by_d_beta[i]) printf("Alert : iVertex %d d/dbeta %d NaN\n", iVertex, i);
	};

	
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = eps*d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][REGRESSORS + i] = d_eps_by_d_beta[0] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][2 * REGRESSORS + i] = d_eps_by_d_beta[1] * d_eps_by_d_beta[i];


	// That was 24.

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_depsbydbeta_x8[blockIdx.x*REGRESSORS]), &(sumdata[0][0]), sizeof(f64) * 8);
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS]), &(sumdata[0][REGRESSORS]),
			2 * REGRESSORS * sizeof(f64));
	};

	__syncthreads();

#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = d_eps_by_d_beta[2] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + REGRESSORS] = d_eps_by_d_beta[3] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + 2 * REGRESSORS] = d_eps_by_d_beta[4] * d_eps_by_d_beta[i];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS + 2 * REGRESSORS]), &(sumdata[0][0]), 3 * REGRESSORS * sizeof(f64));
	};
	__syncthreads();

#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = d_eps_by_d_beta[5] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + REGRESSORS] = d_eps_by_d_beta[6] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + 2 * REGRESSORS] = d_eps_by_d_beta[7] * d_eps_by_d_beta[i];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS + 5 * REGRESSORS]),
			&(sumdata[0][0]), 3 * REGRESSORS * sizeof(f64));
	};

}


__global__ void kernelAccumulateSummands_4x4(
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	// outputs:
	f64 * __restrict__ p_sum_eps_depsbydbeta_x4,
	f64 * __restrict__ p_sum_depsbydbeta__4x4
) {
	__shared__ f64 sumdata[threadsPerTileMajorClever][20]; 
	// Row-major memory layout implies that this is a contiguous array for each thread.
	
	// We can have say 24 doubles in shared. We need to sum 64 + 8 + 1 = 73 things. 24*3 = 72. hah!
	// It would be nicer then if we just called this multiple times. But it has to be for distinct input data..
	// Note that given threadsPerTileMajor = 128 we could comfortably put 48 doubles in shared and still run something.

	// The inputs are only 9 doubles so we can have them.
	// We only need the upper matrix. 1 + 2 + 3 + 4 +5 +6+7+8+9 = 45
	// So we can do it in 2 goes.

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	f64 eps = p_epsilon[iVertex];
	f64 d_eps_by_d_beta[4]; 
	int i;
	for (i = 0; i < 4; i++) {
		d_eps_by_d_beta[i] = p_d_eps_by_dbeta[iVertex + i*NUMVERTICES];
	};	

#pragma unroll 
	for (i = 0; i < 4; i++)
		sumdata[threadIdx.x][i] = eps*d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < 4; i++)
		sumdata[threadIdx.x][4+i] = d_eps_by_d_beta[0] *d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < 4; i++)
		sumdata[threadIdx.x][8 + i] = d_eps_by_d_beta[1] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < 4; i++)
		sumdata[threadIdx.x][12 + i] = d_eps_by_d_beta[2] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < 4; i++)
		sumdata[threadIdx.x][16 + i] = d_eps_by_d_beta[3] * d_eps_by_d_beta[i];

	// That was 20. Repeated some!

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 20; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 20; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_depsbydbeta_x4[blockIdx.x*4]), &(sumdata[0][0]), sizeof(f64)*4);
		memcpy(&(p_sum_depsbydbeta__4x4[blockIdx.x *4 * 4]), &(sumdata[0][4]),
			4*4 * sizeof(f64));		
	};
		
}


__global__ void kernelAccumulateSummands6(
	f64 * __restrict__ p_epsilon,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaJ_x4,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaR_x4,

	// outputs:
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_J_x4,
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_R_x4,
	f64 * __restrict__ p_sum_depsbydbeta__8x8,  // do we want to store 64 things in memory? .. we don't.
	f64 * __restrict__ p_sum_eps_eps_
) {
	__shared__ f64 sumdata[threadsPerTileMajor][24];
	// Row-major memory layout implies that this is a contiguous array for each thread.

	// We can have say 24 doubles in shared. We need to sum 64 + 8 + 1 = 73 things. 24*3 = 72. hah!
	// It would be nicer then if we just called this multiple times. But it has to be for distinct input data..
	// Note that given threadsPerTileMajor = 128 we could comfortably put 48 doubles in shared and still run something.

	// The inputs are only 9 doubles so we can have them.
	// We only need the upper matrix. 1 + 2 + 3 + 4 +5 +6+7+8+9 = 45
	// So we can do it in 2 goes.

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	f64 eps = p_epsilon[iVertex];
	f64_vec4 d_eps_by_d_beta_J;
	f64_vec4 d_eps_by_d_beta_R;
	memcpy(&d_eps_by_d_beta_J, &(p_d_eps_by_dbetaJ_x4[iVertex]), sizeof(f64_vec4));
	memcpy(&d_eps_by_d_beta_R, &(p_d_eps_by_dbetaR_x4[iVertex]), sizeof(f64_vec4));

	sumdata[threadIdx.x][0] = eps*d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = eps*d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = eps*d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = eps*d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = eps*d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = eps*d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = eps*d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = eps*d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[3];

	// Can we fit the rest into 24? yes

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_depsbydbeta_J_x4[blockIdx.x]), &(sumdata[0][0]), sizeof(f64_vec4));
		memcpy(&(p_sum_eps_depsbydbeta_R_x4[blockIdx.x]), &(sumdata[0][4]), sizeof(f64_vec4));

		// Now careful - let's fill in one row at a time. 
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8]), &(sumdata[0][8]), 16 * sizeof(f64));

		if (sumdata[0][17] < 0.0) printf("blockIdx.x %d sumdata[0][17] %1.5E \n",
			blockIdx.x, sumdata[0][17]);

	};

	__syncthreads();

	sumdata[threadIdx.x][0] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[3];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8 + 8 + 8]), &(sumdata[0][0]), 24 * sizeof(f64));
	};
	__syncthreads();


	sumdata[threadIdx.x][0] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[3];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8 + 40]), &(sumdata[0][0]), 24 * sizeof(f64));
	};
	__syncthreads();

	sumdata[threadIdx.x][0] = eps*eps;

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata[threadIdx.x][0] += sumdata[threadIdx.x + k][0];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata[threadIdx.x][0] += sumdata[threadIdx.x + s - 1][0];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		p_sum_eps_eps_[blockIdx.x] = sumdata[0][0];

	};
}

/*__global__ void kernelCalculateOverallVelocitiesVertices_1(
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
	)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural const info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64_vec2 v_overall(0.0, 0.0);
	
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + threadIdx.x * 2 + threadsPerTileMinor*blockIdx.x, sizeof(structural) * 2);
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
	}
	long const EndMinor = threadsPerTileMinor*blockIdx.x + 2 * blockDim.x;

	__syncthreads();

	if (info.flag == DOMAIN_VERTEX)
	{
		v4 const vie = p_vie_major[iVertex];
		f64_vec3 const v_n = p_v_n_major[iVertex];
		nvals const n = p_n_major[iVertex];

		short tri_len = info.neigh_len;
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		memcpy(izTri, p_izTri + iVertex * MAXNEIGH, sizeof(long) * MAXNEIGH);
		memcpy(szPBC, p_szPBCtri_verts + iVertex*MAXNEIGH, sizeof(char)*MAXNEIGH);

		// Our own drift:

		v_overall = (vie.vxy*(m_e + m_i)*n.n +
			v_n.xypart()*m_n*n.n_n) /
			((m_e + m_i)*n.n + m_n*n.n_n);
		f64_vec2 v_adv = v_overall;

		if (TEST3) printf("%d vie.vxy %1.9E %1.9E v_n %1.9E %1.9E n %1.9E nn %1.9E\n"
			"v_overall %1.9E %1.9E\n",
			iVertex, vie.vxy.x, vie.vxy.y, v_n.x, v_n.y, n.n, n.n_n, v_overall.x, v_overall.y);

		// Now add in drift towards barycenter

		// 1. Work out where barycenter is, from n_shards
		// (we have to load for both n and n_n, and need to be careful about combining to find overall barycenter)
		f64_vec2 barycenter;
		ShardModel shards_n, shards_n_n;
		memcpy(&shards_n, p_shards_n + iVertex, sizeof(ShardModel));
		memcpy(&shards_n_n, p_shards_n_n + iVertex, sizeof(ShardModel));
		
		// Sum in triangles the integral of n against (x,y):
		// Sum in triangles the integral of n:
		f64_vec2 numer(0.0,0.0);
		f64 mass = 0.0, Areatot = 0.0;
		short inext, i;
		f64_vec2 pos0, pos1;
		f64 Area_tri;
		f64 wt0, wt1, wtcent;
		for (i = 0; i < tri_len; i++)
		{
			inext = i + 1; if (inext == tri_len) inext = 0;
			// Collect positions:
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				pos0 = shared_pos[izTri[i] - StartMinor];
			} else {
				pos0 = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise_d*pos0;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise_d*pos0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
			{
				pos1 = shared_pos[izTri[inext] - StartMinor];
			} else {
				pos1 = p_info_minor[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise_d*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise_d*pos1;

			// Get Area_tri:

			Area_tri = 0.5*fabs(pos0.x*info.pos.y + info.pos.x*pos1.y + pos1.x*pos0.y
				              - info.pos.x*pos0.y - pos1.x*info.pos.y - pos0.x*pos1.y);

#define MIN_n_DENSITY_FOR_MESH 2.5e16

			// 5e16 means 20 times fewer vertices through space than in the
			// initial mesh. 1e16 means 100 times fewer, spacing 10 times larger.
			// 2.5e16 is 40 times fewer, sqrt(40) ~ 6.5 times spacing.

			f64 ntot_i = shards_n.n[i] + shards_n_n.n[i];
			f64 ntot_next = shards_n.n[inext] + shards_n_n.n[inext];
			f64 ntot_cent = shards_n.n_cent + shards_n_n.n_cent;
			
			if (TEST3) printf("ntot_i,next,cent %1.8E %1.8E %1.8E pos.y %1.8E %1.8E %1.8E\n",
				ntot_i, ntot_next, ntot_cent, pos0.y, pos1.y, info.pos.y);

			if (ntot_i < MIN_n_DENSITY_FOR_MESH) ntot_i = MIN_n_DENSITY_FOR_MESH;
			if (ntot_next < MIN_n_DENSITY_FOR_MESH) ntot_next = MIN_n_DENSITY_FOR_MESH;
			if (ntot_cent < MIN_n_DENSITY_FOR_MESH) ntot_cent = MIN_n_DENSITY_FOR_MESH;


			wt0 = ntot_i / 12.0
				+ ntot_next / 24.0
				+ ntot_cent / 24.0;
			wt1 = ntot_i / 24.0
				+ ntot_next / 12.0
				+ ntot_cent / 24.0;
			wtcent = ntot_i / 24.0
				+ ntot_next / 24.0
				+ ntot_cent / 12.0;

			numer.x += 2.0*Area_tri*(pos0.x*wt0 + pos1.x*wt1 + info.pos.x*wtcent);
			numer.y += 2.0*Area_tri*(pos0.y*wt0 + pos1.y*wt1 + info.pos.y*wtcent);

			mass += THIRD*Area_tri*(ntot_i + ntot_next + ntot_cent);
			Areatot += Area_tri; 

		//	if (iVertex == VERTCHOSEN)
		//		printf("%d info.pos %1.9E %1.9E  pos0 %1.9E %1.9E pos1 %1.9E %1.9E\n"
		//			"Area_shard %1.10E mass_shard %1.10E shards_n %1.9E %1.9E %1.9E\n\n",
		//			iVertex,  info.pos.x, info.pos.y, pos0.x, pos0.y,pos1.x,pos1.y,
		//			Area_tri, THIRD*Area_tri*(shards_n.n[i] + shards_n_n.n[i] +
		//				shards_n.n[inext] + shards_n_n.n[inext] + shards_n.n_cent + shards_n_n.n_cent),
		//			shards_n.n_cent + shards_n_n.n_cent,
		//			shards_n.n[i] + shards_n_n.n[i],
		//			shards_n.n[inext] + shards_n_n.n[inext]
		//			);
		}

		// Divide one by the other to give the barycenter:
		barycenter = numer / mass;
						
		// Having great difficulty seeing why we get the result that we do.
		// How close are different points?

		// I think we want to work on optimizing the distance to neighbour relative to average density of this and neighbour.
		// That should control triangle area per triangle density.
		// Alternatively, we could directly look at which triangle centroid propels us away because the triangle is too small ...

		// Also: splint at absolute distance of 4 micron, for now.
				
		// 2. Drift towards it is proportional to normalized distance squared.
		// square root of vertcell area, to normalize dist from barycenter:

		f64_vec2 to_bary = barycenter - info.pos;
		
		f64 factor = max( 3.0 * to_bary.dot(to_bary) / Areatot, 0.8);
		// r = sqrt(Area)/pi and we take (delta/r)^2
		// uplift by 4/PI
		// Think this should have been max: at MOST go to where barycenter is. In fact at most go 0.8, let velocity drag you a bit from it.
		
		//f64 distance = to_bary.modulus();
		//f64_vec2 unit_to_bary = to_bary / distance;
		
		if ((TEST3) || (v_overall.dot(v_overall) > 1.0e19))  {
			printf("iVertex %d pos %1.9E %1.9E barycenter %1.9E %1.9E v_overall %1.9E %1.9E \n"
				"factor %1.9E sqrt(Areatot) %1.9E |to_bary| %1.9E \n",
				iVertex, info.pos.x, info.pos.y, barycenter.x, barycenter.y, v_overall.x, v_overall.y,
				factor, sqrt(Areatot), to_bary.modulus());
		}
		
		// v_overall += unit_to_bary * factor * 1.0e10;
		// We used area of vertcell rather than area with its neighbours at corners.
		// 1e10 > any v
		// but also not too big? If dist = 1e-3 then 1e10*1e-12 = 1e-2. Oh dear.
		// That is too big of a hop.
		// The "speed" should be proportional to actual distance.

		// We can use 1e10 if ||to_bary|| is 1e-2 or more.
		// We need to multiply by distance/1e-2

		v_overall += to_bary * factor / h_full_adv; 

		// So then we do want to cancel some of hv as an extra term in case v = ~1e8 and the cell is only 2 micron wide?
		// Hmm
		v_overall -= to_bary * factor * max(v_adv.dot(to_bary)/to_bary.dot(to_bary),0.0);
		// ie if we were moving in the direction of the barycenter anyway, we can knock that off.
		// Now let's look at /h in that.
		// let v_addition = to_bary*factor/h ; we knock off 
		// the component of v_addition that aligned with v_adv?
		// should be v_adv.dot(v_additional)/ v_adv.

		// Would do better to predict barycenter trajectory.

		// Want to knock OFF of additional, v_adv projected on to additional

		// Trouble is we need to ensure that when density is low we cling to barycenter even when there is a wind
		// blowing, we keep restoring.


		// Let's try a very simple way:
		// We want everything to live at its own barycenter.
		// Ignore the v.
		// Just move to the barycenter. 
		// Maximum speed 2e7. We probably won't see excess of that anywhere that density is considerable.

		// There is almost certainly a more elegant way that puts them together.
		// If v faces to_bary then we do nothing.
		
		if (TEST3) 
			printf("%d additional v %1.9E %1.9E cancelv addition %1.9E %1.9E\n\n", 
				iVertex, to_bary.x * factor / h_full_adv,
				to_bary.y * factor / h_full_adv,
				to_bary.x * factor * max(-v_adv.dot(to_bary) / to_bary.dot(to_bary), 0.0),
				to_bary.y * factor * max(-v_adv.dot(to_bary) / to_bary.dot(to_bary), 0.0)				
				);
		
		// To watch out for: overshooting because hv takes us towards barycenter
		
		// Hope this stuff works well because the equal masses takes a bit of doing.		
	};
	p_v_overall_major[iVertex] = v_overall;
}*/


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
)
{
#ifdef EULERIAN
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	f64_vec2 v_overall(0.0, 0.0);
	p_v_overall_major[iVertex] = v_overall;
#else
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural const info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64_vec2 v_overall(0.0, 0.0);

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + threadIdx.x * 2 + threadsPerTileMinor*blockIdx.x, sizeof(structural) * 2);
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
	}
	long const EndMinor = threadsPerTileMinor*blockIdx.x + 2 * blockDim.x;

	__syncthreads();

	if ((info.flag == DOMAIN_VERTEX) && (info.pos.dot(info.pos) < MESHMOVE_MAX_RADIUS*MESHMOVE_MAX_RADIUS))
	{
		// No velocity load.
	//	v4 const vie = p_vie_major[iVertex];
	//	f64_vec3 const v_n = p_v_n_major[iVertex];

	//	nvals const n = p_n_major[iVertex]; // not used
		short tri_len = info.neigh_len;
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		memcpy(izTri, p_izTri + iVertex * MAXNEIGH, sizeof(long) * MAXNEIGH);
		memcpy(szPBC, p_szPBCtri_verts + iVertex*MAXNEIGH, sizeof(char)*MAXNEIGH);

		//v_overall = (vie.vxy*(m_e + m_i)*n.n +
		//	v_n.xypart()*m_n*n.n_n) /
		//	((m_e + m_i)*n.n + m_n*n.n_n);
		//f64_vec2 v_adv = v_overall;
		//if (TEST3) printf("%d vie.vxy %1.9E %1.9E v_n %1.9E %1.9E n %1.9E nn %1.9E\n"
		//	"v_overall %1.9E %1.9E\n",
		//	iVertex, vie.vxy.x, vie.vxy.y, v_n.x, v_n.y, n.n, n.n_n, v_overall.x, v_overall.y);

		// Now add in drift towards barycenter

		// 1. Work out where barycenter is, from n_shards
		// (we have to load for both n and n_n, and need to be careful about combining to find overall barycenter)
		f64_vec2 barycenter;
		ShardModel shards_n, shards_n_n;
		memcpy(&shards_n, p_shards_n + iVertex, sizeof(ShardModel));
		memcpy(&shards_n_n, p_shards_n_n + iVertex, sizeof(ShardModel));

		// Sum in triangles the integral of n against (x,y):
		// Sum in triangles the integral of n:
		f64_vec2 numer(0.0, 0.0);
		f64 mass = 0.0, Areatot = 0.0;
		short inext, i;
		f64_vec2 pos0, pos1;
		f64 Area_tri;
		f64 wt0, wt1, wtcent;
		bool bZeroOut = false;
		for (i = 0; i < tri_len; i++)
		{
			inext = i + 1; if (inext == tri_len) inext = 0;
			// Collect positions:
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				pos0 = shared_pos[izTri[i] - StartMinor];
			} else {
				pos0 = p_info_minor[izTri[i]].pos;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise_d*pos0;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise_d*pos0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
			{
				pos1 = shared_pos[izTri[inext] - StartMinor];
			} else {
				pos1 = p_info_minor[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise_d*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise_d*pos1;

			if (pos0.x*pos0.x + (pos0.y - CATHODE_ROD_R_POSITION)*(pos0.y - CATHODE_ROD_R_POSITION) <
				CATHODE_ROD_RADIUS*CATHODE_ROD_RADIUS)
				bZeroOut = true;

			// Get Area_tri:
			Area_tri = 0.5*fabs(pos0.x*info.pos.y + info.pos.x*pos1.y + pos1.x*pos0.y
				- info.pos.x*pos0.y - pos1.x*info.pos.y - pos0.x*pos1.y);

#define MIN_n_DENSITY_FOR_MESH 2.5e16

			// 5e16 means 20 times fewer vertices through space than in the
			// initial mesh. 1e16 means 100 times fewer, spacing 10 times larger.
			// 2.5e16 is 40 times fewer, sqrt(40) ~ 6.5 times spacing.

			f64 ntot_i = shards_n.n[i] + shards_n_n.n[i];
			f64 ntot_next = shards_n.n[inext] + shards_n_n.n[inext];
			f64 ntot_cent = shards_n.n_cent + shards_n_n.n_cent;

			if (TEST3) printf("ntot_i,next,cent %1.8E %1.8E %1.8E pos.y %1.8E %1.8E %1.8E\n",
				ntot_i, ntot_next, ntot_cent, pos0.y, pos1.y, info.pos.y);

			if (ntot_i < MIN_n_DENSITY_FOR_MESH) ntot_i = MIN_n_DENSITY_FOR_MESH;
			if (ntot_next < MIN_n_DENSITY_FOR_MESH) ntot_next = MIN_n_DENSITY_FOR_MESH;
			if (ntot_cent < MIN_n_DENSITY_FOR_MESH) ntot_cent = MIN_n_DENSITY_FOR_MESH;
			
			wt0 = ntot_i / 12.0
				+ ntot_next / 24.0
				+ ntot_cent / 24.0;
			wt1 = ntot_i / 24.0
				+ ntot_next / 12.0
				+ ntot_cent / 24.0;
			wtcent = ntot_i / 24.0
				+ ntot_next / 24.0
				+ ntot_cent / 12.0;

			numer.x += 2.0*Area_tri*(pos0.x*wt0 + pos1.x*wt1 + info.pos.x*wtcent);
			numer.y += 2.0*Area_tri*(pos0.y*wt0 + pos1.y*wt1 + info.pos.y*wtcent);

			mass += THIRD*Area_tri*(ntot_i + ntot_next + ntot_cent);
			Areatot += Area_tri;
		}

		// Divide one by the other to give the barycenter:
		barycenter = numer / mass;

		f64_vec2 to_bary = barycenter - info.pos;

		f64 factor = max(4.0 * to_bary.dot(to_bary) / Areatot, 0.8);
		// We may move slowly if we are not far away.
		
		// For now, just zero out vertex motion near the cathode rod:
		if (bZeroOut) {
			v_overall.x = 0.0; v_overall.y = 0.0;
		} else {

			v_overall = to_bary * factor / h_full_adv;
			// Now reduce if we are over 2e7
			f64 ratio = sqrt(1 + v_overall.dot(v_overall) / 4.0e14);
			v_overall /= ratio;
			if ((TEST3) || (v_overall.dot(v_overall) > 1.0e19)) {
				printf("iVertex %d pos %1.9E %1.9E barycenter %1.9E %1.9E v_overall %1.9E %1.9E \n"
					"factor %1.9E sqrt(Areatot) %1.9E |to_bary| %1.9E \n",
					iVertex, info.pos.x, info.pos.y, barycenter.x, barycenter.y, v_overall.x, v_overall.y,
					factor, sqrt(Areatot), to_bary.modulus());
			};
		};
		
		// Let's try a very simple way:
		// We want everything to live at its own barycenter.
		// Ignore the v.
		// Just move to the barycenter. 
		// Maximum speed 2e7. We probably won't see excess of that anywhere that density is considerable.
		
		// Hope this stuff works well because the equal masses takes a bit of doing.		
	}

	p_v_overall_major[iVertex] = v_overall;
#endif
}

__global__ void kernelCentroidVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
)
{
	__shared__ f64_vec2 shared_v[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMajor];


	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_v[threadIdx.x] = p_overall_v_major[getindex];
		shared_pos[threadIdx.x] = p_info[BEGINNING_OF_CENTRAL + getindex].pos;
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();

	
	// Thoughts:
	// We want it to be the motion of the circumcenter .. but linear interpolation of v is probably good enough?
	// That won't work - consider right-angled.
	// Silver standard approach: empirical estimate of time-derivative of cc position.

	f64_vec2 v(0.0, 0.0);

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

		f64_vec2 poscorner0, poscorner1, poscorner2, vcorner0, vcorner1, vcorner2;

		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			poscorner0 = shared_pos[tri_corner_index.i1 - StartMajor];
			vcorner0 = shared_v[tri_corner_index.i1 - StartMajor];
		}
		else {
			poscorner0 = p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
			vcorner0 = p_overall_v_major[tri_corner_index.i1];
		};
		if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) {
			poscorner0 = Clockwise_d*poscorner0;
			vcorner0 = Clockwise_d*vcorner0;
		}
		if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner0 = Anticlockwise_d*poscorner0;
			vcorner0 = Anticlockwise_d*vcorner0;
		}

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			poscorner1 = shared_pos[tri_corner_index.i2 - StartMajor];
			vcorner1 = shared_v[tri_corner_index.i2 - StartMajor];
		}
		else {
			poscorner1 = p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
			vcorner1 = p_overall_v_major[tri_corner_index.i2];
		};
		if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) {
			poscorner1 = Clockwise_d*poscorner1;
			vcorner1 = Clockwise_d*vcorner1;
		}
		if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner1 = Anticlockwise_d*poscorner1;
			vcorner1 = Anticlockwise_d*vcorner1;
		}

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			poscorner2 = shared_pos[tri_corner_index.i3 - StartMajor];
			vcorner2 = shared_v[tri_corner_index.i3 - StartMajor];
		}
		else {
			poscorner2 = p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
			vcorner2 = p_overall_v_major[tri_corner_index.i3];
		};

		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) {
			poscorner2 = Clockwise_d*poscorner2;
			vcorner2 = Clockwise_d*vcorner2;
		}
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner2 = Anticlockwise_d*poscorner2;
			vcorner2 = Anticlockwise_d*vcorner2;
		}
		
		v = 0.3333333333333*(vcorner0 + vcorner1 + vcorner2);
		
		if (v.dot(v) > 1.0e19) {
			printf("iTri %d v %1.8E %1.8E : %d %d %d vcorner %1.8E %1.8E , %1.8E %1.8E , %1.8E %1.8E \n"
				, index,
				index, v.x, v.y, 
				tri_corner_index.i1, tri_corner_index.i2, tri_corner_index.i3,
				vcorner0.x, vcorner0.y, vcorner1.x, vcorner1.y, vcorner2.x, vcorner2.y);
		}

		if (info.flag == CROSSING_INS) {
			// This is the one place we should recognize, the 'centroid' is never going to move off the insulator.
			// This v_overall is used in mass & heat advection so should properly have vr set to 0 to show it is 0.
			// When we set AreaMajor we still project down to insulator -- -right ?!

			// The truth is that the triangle centre can have motion as it's not on the insulator.

			// Think through how this is used in momflux and in heat advection.

			//f64_vec2 rhat = info.pos / info.pos.modulus();
			f64_vec2 wouldbe_centroid = THIRD*(poscorner0 + poscorner1 + poscorner2);
			if (wouldbe_centroid.dot(wouldbe_centroid) < DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				v -= (v.dot(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos;
			
			// Important to know that only those below are projected up to it.

			// What about in the case that all were projected because we are doing major cells? WATCH OUT.

		};

	} else {
		// leave it == 0		
	};
	p_overall_v_minor[index] = v;
}

__global__ void Reversesubtract_vec3(
	f64_vec3 * __restrict__ p_reverse,
	f64_vec3 * __restrict__ p_augmented)
{
	long const index = threadIdx.x + blockIdx.x*blockDim.x;
	f64_vec3 rev = p_reverse[index];
	f64_vec3 aug = p_augmented[index];
	rev = aug - rev;
	p_reverse[index] = rev;
}

__global__ void kernelCircumcenterVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
)
{
	__shared__ f64_vec2 shared_v[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMajor];


	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_v[threadIdx.x] = p_overall_v_major[getindex];
		shared_pos[threadIdx.x] = p_info[BEGINNING_OF_CENTRAL + getindex].pos;
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();


	// Thoughts:
	// We want it to be the motion of the circumcenter .. but linear interpolation of v is probably good enough?
	// That won't work - consider right-angled.
	// Silver standard approach: empirical estimate of time-derivative of cc position.

	f64_vec2 v(0.0, 0.0);

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

		f64_vec2 poscorner0, poscorner1, poscorner2, vcorner0, vcorner1, vcorner2;

		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			poscorner0 = shared_pos[tri_corner_index.i1 - StartMajor];
			vcorner0 = shared_v[tri_corner_index.i1 - StartMajor];
		}
		else {
			poscorner0 = p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
			vcorner0 = p_overall_v_major[tri_corner_index.i1];
		};
		if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) {
			poscorner0 = Clockwise_d*poscorner0;
			vcorner0 = Clockwise_d*vcorner0;
		}
		if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner0 = Anticlockwise_d*poscorner0;
			vcorner0 = Anticlockwise_d*vcorner0;
		}

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			poscorner1 = shared_pos[tri_corner_index.i2 - StartMajor];
			vcorner1 = shared_v[tri_corner_index.i2 - StartMajor];
		}
		else {
			poscorner1 = p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
			vcorner1 = p_overall_v_major[tri_corner_index.i2];
		};
		if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) {
			poscorner1 = Clockwise_d*poscorner1;
			vcorner1 = Clockwise_d*vcorner1;
		}
		if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner1 = Anticlockwise_d*poscorner1;
			vcorner1 = Anticlockwise_d*vcorner1;
		}

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			poscorner2 = shared_pos[tri_corner_index.i3 - StartMajor];
			vcorner2 = shared_v[tri_corner_index.i3 - StartMajor];
		}
		else {
			poscorner2 = p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
			vcorner2 = p_overall_v_major[tri_corner_index.i3];
		};

		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) {
			poscorner2 = Clockwise_d*poscorner2;
			vcorner2 = Clockwise_d*vcorner2;
		}
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner2 = Anticlockwise_d*poscorner2;
			vcorner2 = Anticlockwise_d*vcorner2;
		}

		f64_vec2 pos;
		f64_vec2 Bb = poscorner1 - poscorner0;
		f64_vec2 C = poscorner2 - poscorner0;
		f64 D = 2.0*(Bb.x*C.y - Bb.y*C.x);
		f64 modB = Bb.x*Bb.x + Bb.y*Bb.y;
		f64 modC = C.x*C.x + C.y*C.y;
		pos.x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
		pos.y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;

		// choose step where h*(sqrt(sum of vcorner^2)) is 1e-9 cm
		f64 temp = sqrt(vcorner0.dot(vcorner0) + vcorner1.dot(vcorner1) + vcorner2.dot(vcorner2));
		f64 h_deriv = 1.0e-9 / temp;

		if (TEST_OVERALL_V)
			printf("iMinor %d poscorner0 %1.12E %1.12E | %1.12E %1.12E | %1.11E %1.11E \n",
				index, poscorner0.x, poscorner0.y,
				poscorner1.x, poscorner1.y,
				poscorner2.x, poscorner2.y
			);
		poscorner0 += h_deriv*vcorner0;
		poscorner1 += h_deriv*vcorner1;
		poscorner2 += h_deriv*vcorner2;

		if (TEST_OVERALL_V)
			printf("iMinor %d poscorner0 %1.12E %1.12E | %1.12E %1.12E | %1.11E %1.11E \n",
				index, poscorner0.x, poscorner0.y,
				poscorner1.x, poscorner1.y,
				poscorner2.x, poscorner2.y
			);
		f64_vec2 newpos;
		Bb = poscorner1 - poscorner0;
		C = poscorner2 - poscorner0;
		D = 2.0*(Bb.x*C.y - Bb.y*C.x);
		modB = Bb.x*Bb.x + Bb.y*Bb.y;
		modC = C.x*C.x + C.y*C.y;
		newpos.x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
		newpos.y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;

		if (info.flag == CROSSING_INS) {
			if (TEST_OVERALL_V)
				printf("iMinor %d info.flag %d :  %d %d %d \n"
					"v %1.8E %1.8E newpos %1.12E %1.12E pos %1.12E %1.12E\n",
					index, info.flag, tri_corner_index.i1, tri_corner_index.i2, tri_corner_index.i3,
					v.x, v.y, newpos.x, newpos.y, pos.x, pos.y);


			f64_vec2 pos2 = pos;
			pos2.project_to_radius(pos, DEVICE_RADIUS_INSULATOR_OUTER);
			pos2 = newpos;
			pos2.project_to_radius(newpos, DEVICE_RADIUS_INSULATOR_OUTER);
		};

		v = (newpos - pos) / h_deriv;


		if (TEST_OVERALL_V)
			printf("iMinor %d info.flag %d :  %d %d %d \n"
				"v %1.8E %1.8E newpos %1.12E %1.12E pos %1.12E %1.12E\n"
				"vcorner0 %1.8E %1.8E vcorner1 %1.8E %1.8E vcorner2 %1.8E %1.8E\n"
				"hderiv %1.9E \n",
				index, info.flag, tri_corner_index.i1, tri_corner_index.i2, tri_corner_index.i3,
				v.x, v.y, newpos.x, newpos.y, pos.x, pos.y,
				vcorner0.x, vcorner0.y, vcorner1.x, vcorner1.y, vcorner2.x, vcorner2.y,
				h_deriv);



		// Empirical estimate of derivative. Saves me messing about with taking derivative of circumcenter position.

		//if (index == 42940)	
		//	printf("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n"
		//		"pos %1.11E %1.11E newpos-pos %1.11E %1.11E hderiv %1.11E\n "
		//		"vcorner0 %1.11E %1.11E vcorner1 %1.11E %1.11E corner2 %1.11E %1.11E v %1.11E %1.11E\n"
		//		"poscorner0 %1.11E %1.11E poscorner1 %1.11E %1.11E poscorner2 %1.11E %1.11E \n"
		//		" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n",
		//		pos.x, pos.y, newpos.x - pos.x, newpos.y - pos.y, h_deriv,
		//		vcorner0.x, vcorner0.y, vcorner1.x, vcorner1.y, vcorner2.x, vcorner2.y, v.x, v.y,
		//		poscorner0.x, poscorner0.y, poscorner1.x, poscorner1.y, poscorner2.x, poscorner2.y);			


	}
	else {
		// leave it == 0		
	};
	p_overall_v_minor[index] = v;
}

__global__ void kernelAdvectPositionsVertex(
	f64 h_use,
	structural * __restrict__ p_info_src_major,
	structural * __restrict__ p_info_dest_major,
	f64_vec2 * __restrict__ p_v_overall_major,
	nvals * __restrict__ p_n_major,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCneigh_vert
	// what else is copied?
	// something we can easily copy over
	// with cudaMemcpy, even ahead of steps?
	// Is there a reason we cannot put into the above routine
	// with a split for "this is a vertex->just use its overall v"
)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; 
	structural info = p_info_src_major[iVertex];
	f64_vec2 overall_v = p_v_overall_major[iVertex];
	f64_vec2 oldpos = info.pos;
	info.pos += h_use*overall_v;

	// Now make correction
	long izNeigh[MAXNEIGH_d];
	char PBCneigh[MAXNEIGH_d];
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izNeigh, p_izNeigh_vert + iVertex*MAXNEIGH_d, sizeof(long)*MAXNEIGH_d);
		memcpy(PBCneigh, p_szPBCneigh_vert + iVertex*MAXNEIGH_d, sizeof(char)*MAXNEIGH_d);
		short i, iMost = -1, iLeast = -1;
		f64 most = 0.0 , least = 1.0e100;
		f64_vec2 leastpos, mostpos, diff;
		nvals n_neigh;
		nvals n_own;
		n_own = p_n_major[iVertex];
		structural infoneigh;
		f64 ratio;
		char buff[255];

		for (i = 0; i < info.neigh_len; i++)
		{ 
			n_neigh = p_n_major[izNeigh[i]];
			infoneigh = p_info_src_major[izNeigh[i]];
			if (infoneigh.flag == DOMAIN_VERTEX) {
				char PBC = PBCneigh[i];
				if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
				f64_vec2 diff = infoneigh.pos - info.pos;
				f64 deltasq = diff.dot(diff);
				ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
				// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
				if (ratio > most) { iMost = i; most = ratio; mostpos = infoneigh.pos; }
				if (ratio < least) { iLeast = i; least = ratio; leastpos = infoneigh.pos; }
			}
		};

		if (most > 2.0*least) {
			// we need to move in the direction away from the 'least' distant point
			// as long as it's improving most/least
		
			for (i = 0; i < info.neigh_len; i++)
			{
				n_neigh = p_n_major[izNeigh[i]];
				infoneigh = p_info_src_major[izNeigh[i]];
				if (infoneigh.flag == DOMAIN_VERTEX) {
					char PBC = PBCneigh[i];
					if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
					if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
					f64_vec2 diff = infoneigh.pos - info.pos;
					f64 deltasq = diff.dot(diff);
					ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
					// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
					printf("%d i %d izNeigh %d ratio %1.14E \n",iVertex, i, izNeigh[i], ratio);
				}
			}
						
			diff = info.pos-leastpos;
			// We want squared modulus of dist to equal half of most			
			f64_vec2 oldpos2 = info.pos;
			info.pos += diff*(sqrt(most/(2.0*least))-1.0);
			printf("%d: most %1.10E least %1.10E diff %1.9E %1.9E \noldpos %1.12E %1.12E old2pos %1.12E %1.12E info.pos %1.12E %1.12E\n",
				iVertex, most, least, diff.x, diff.y, oldpos.x, oldpos.y, oldpos2.x, oldpos2.y, info.pos.x, info.pos.y);

			for (i = 0; i < info.neigh_len; i++)
			{
				n_neigh = p_n_major[izNeigh[i]];
				infoneigh = p_info_src_major[izNeigh[i]];
				if (infoneigh.flag == DOMAIN_VERTEX) {
					char PBC = PBCneigh[i];
					if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
					if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
					f64_vec2 diff = infoneigh.pos - info.pos;
					f64 deltasq = diff.dot(diff);
					ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
					// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
					printf("%d i %d izNeigh %d ratio %1.14E \n",iVertex, i, izNeigh[i], ratio);
				}
			};
		};

//		least = 1.0e100;
//		iLeast = -1;
//		for (i = 0; i < info.neigh_len; i++)
//		{
//			n_neigh = p_n_major[izNeigh[i]];
//			infoneigh = p_info_src_major[izNeigh[i]];
//			char PBC = PBCneigh[izNeigh[i]];
//			if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise2*infoneigh.pos;
//			if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise2*infoneigh.pos;
//			f64_vec2 diff = infoneigh.pos - info.pos;
//			f64 deltasq = diff.dot(diff);			
//			if (deltasq < least) { iLeast = i; least = deltasq; leastpos = infoneigh.pos; }
//		}
//#define MINDIST 0.0003 // 3 micron
//
//		if (least < MINDIST) {
//
//		}

		overall_v = (info.pos-oldpos) / h_use;
		p_v_overall_major[iVertex] = overall_v;
	}
	p_info_dest_major[iVertex] = info;

}

// Run vertex first, then average v_overall to tris, then run this after.
__global__ void kernelAdvectPositionsTris(
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_v_overall_minor
	// what else is copied?
	// something we can easily copy over
	// with cudaMemcpy, even ahead of steps?
	// Is there a reason we cannot put into the above routine
	// with a split for "this is a vertex->just use its overall v"
)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	structural info = p_info_src[index];
	f64_vec2 overall_v = p_v_overall_minor[index];
	f64_vec2 oldpos = info.pos;
	info.pos += h_use*overall_v;
	if (index == VERTCHOSEN + BEGINNING_OF_CENTRAL)
		printf("iVertex %d oldpos %1.10E %1.10E info.pos %1.10E %1.10E overall_v %1.10E %1.10E h_use %1.10E\n",
			VERTCHOSEN, oldpos.x, oldpos.y, info.pos.x, info.pos.y, overall_v.x, overall_v.y, h_use);
	
	p_info_dest[index] = info;
}


__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info_major,
	nvals * __restrict__ p_n,
	T3 * __restrict__ p_T,
	species3 * __restrict__ p_nu
) {
	// Save nu_iHeart, nu_eHeart, nu_nn_visc.

	species3 nu;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // iVertex OF VERTEX
	structural info = p_info_major[iVertex];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		// We have not ruled out calculating traffic into outermost vertex cell - so this needs nu calculated in it.
		// (Does it actually try and receive traffic?)

		our_n = p_n[iVertex]; // never used again once we have kappa
		T = p_T[iVertex];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		sigma_visc *= ArtificialUpliftFactor_MT(our_n.n, our_n.n_n);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;


		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * max(MINIMUM_NU_EI_DENSITY, our_n.n) *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;
		
		if (TEST) printf("%d nu_en %1.9E nu_eiBar %1.9E nu_eHeart %1.9E\n",
			VERTCHOSEN, nu_en_visc, nu_eiBar, nu.e);

		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
		
		nu.n = 0.74*nu_ni_visc + 0.4*nu.n; // Rate to use in thermal conductivity.

		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);

		if ((TEST) ) {
			printf("@@@\nGPU %d: nu.i %1.12E |  %1.12E %1.12E %1.12E %1.12E n %1.10E n_n %1.10E Ti %1.10E\n"
				"sigma_visc %1.12E Ti %1.10E Tn %1.10E sqrt %1.10E\n", 
				
				iVertex, nu.i, nu_in_visc, nu_ii, nu_ni_visc, nu.n,
				our_n.n, our_n.n_n, T.Ti,
				sigma_visc, T.Ti, T.Tn, sqrt(T.Ti / m_i + T.Tn / m_n));
			printf("@@@\nGPU %d: nu.e %1.14E | nu_eiBar %1.14E our_n %1.14E lambda %1.14E over T^3/2 %1.14E nu_en_visc %1.14E\n",
				iVertex, nu.e, nu_eiBar, our_n.n, Get_lnLambda_d(our_n.n, T.Te), 1.0 / (T.Te*sqrt_T), nu_en_visc);
		}

		//  shared_n_over_nu[threadIdx.x].e = our_n.n / nu.e;
		//	shared_n_over_nu[threadIdx.x].i = our_n.n / nu.i;
		//	shared_n_over_nu[threadIdx.x].n = our_n.n_n / nu.n;
	}
	else {
		memset(&nu, 0, sizeof(species3));
	}

	p_nu[iVertex] = nu;
}

__global__ void kernelCalculate_kappa_nu(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,
	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e
)
{
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	species3 nu;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		our_n = p_n_minor[iMinor];
		T = p_T_minor[iMinor];
		
		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);

		nu.n = 0.74*nu_ni_visc + 0.4*nu.n; // Rate to use in thermal conductivity.
				
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);
		//  ita uses nu_ion =  0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar;
		// which again is only about half as much. BUT WE KNOW HE WORKS in DOUBLE Braginskii ??
		// says Vranjes -- so check that the rest of the formula does not compensate.

		// Would like consistent approach.
		// 1. We approached the heat flux directly per Golant. Where did it say what nu to use and how to add up?
		// Can we follow our own logic and then compare with what Zhdanov says?
		// 2. What about the limit of full ionization? Zero ionization? Does Zhdanov attain sensible limit?
		// Our version made sense.

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;

		// Comparison with Zhdanov's denominator for ita looks like this one overestimated
		// by a factor of something like 1.6?
		
		f64 kappa_n = NEUTRAL_KAPPA_FACTOR * our_n.n_n * T.Tn / (m_n * nu.n) ;
		f64 kappa_i = (20.0 / 9.0) * our_n.n*T.Ti / (m_i * nu.i);
		f64 kappa_e = 2.5*our_n.n*T.Te / (m_e * nu.e);
		
		if ((TESTTRI)) printf("kappa_e %1.9E our_n.n %1.9E Te %1.9E nu %1.9E\n",
			kappa_e, our_n.n, T.Te, nu.e);

		if (kappa_i != kappa_i) printf("Tri %d kappa_i = NaN T %1.9E %1.9E %1.9E n %1.9E %1.9E \n", iMinor,
			T.Tn, T.Ti, T.Te, our_n.n_n, our_n.n);
		p_kappa_n[iMinor] = kappa_n;
		p_kappa_i[iMinor] = kappa_i;
		p_kappa_e[iMinor] = kappa_e;
		p_nu_i[iMinor] = nu.i;
		p_nu_e[iMinor] = nu.e;

	};
}


__global__ void kernelCalculate_kappa_nu_vertices(
	structural * __restrict__ p_info_major, // NB: MAJOR
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,

	f64 * __restrict__ p_kappa_n_major,
	f64 * __restrict__ p_kappa_i_major,
	f64 * __restrict__ p_kappa_e_major,
	f64 * __restrict__ p_nu_i_major,
	f64 * __restrict__ p_nu_e_major // CHECK DIMENSIONS IF USING BIG ARRAY
)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	species3 nu;
	
	structural info = p_info_major[iVertex];
	if ((info.flag == DOMAIN_VERTEX))
	{
		our_n = p_n_major[iVertex];
		T = p_T_major[iVertex];

		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		
	//	sigma_visc *= ArtificialUpliftFactor(our_n.n, our_n.n_n);
	//	sigma_MT *= ArtificialUpliftFactor(our_n.n, our_n.n_n);

		
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);

		nu.n = 0.74*nu_ni_visc + 0.4*nu.n; // Rate to use in thermal conductivity.

		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);
		//  ita uses nu_ion =  0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar;
		// which again is only about half as much. BUT WE KNOW HE WORKS in DOUBLE Braginskii ??
		// says Vranjes -- so check that the rest of the formula does not compensate.


		// Would like consistent approach.
		// 1. We approached the heat flux directly per Golant. Where did it say what nu to use and how to add up?
		// Can we follow our own logic and then compare with what Zhdanov says?
		// 2. What about the limit of full ionization? Zero ionization? Does Zhdanov attain sensible limit?
		// Our version made sense.

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * 
			//max(MINIMUM_NU_EI_DENSITY,our_n.n) *
			our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;
		

		if (iVertex == VERTCHOSEN) {
			printf("iVertex %d nu.n %1.10E nu_i %1.10E nu_e %1.10E nu_eiBar %1.10E nu_ii %1.10E \nnu_eiBar over n %1.10E nu_ii over n %1.10E \n", 
				iVertex, nu.n, nu.i, nu.e, nu_eiBar, nu_ii, nu_eiBar/our_n.n,
				nu_ii/our_n.n);
		}

		// Comparison with Zhdanov's denominator for ita looks like this one overestimated
		// by a factor of something like 1.6?

		f64 kappa_n = NEUTRAL_KAPPA_FACTOR * our_n.n_n * T.Tn / (m_n * nu.n);
		f64 kappa_i = (20.0 / 9.0) * our_n.n*T.Ti / (m_i * nu.i);
		f64 kappa_e = 2.5*our_n.n*T.Te / (m_e * nu.e);

		if (iVertex == VERTCHOSEN) {
			printf("iVertex %d kappa_n %1.10E kappa_i %1.10E kappa_e %1.10E nn %1.10E n %1.10E Tn %1.10E Ti %1.10E Te %1.10E\n",
				iVertex, kappa_n, kappa_i, kappa_e, our_n.n_n, our_n.n, T.Tn, T.Ti, T.Te);
		};

		//if ((TESTKAPPA)) printf("kappa_e %1.9E our_n.n %1.9E Te %1.9E nu %1.9E\n",
		//	kappa_e, our_n.n, T.Te, nu.e);

		if (kappa_i != kappa_i) printf("iVertex %d kappa_i = NaN T %1.9E %1.9E %1.9E n %1.9E %1.9E \n", iVertex,
			T.Tn, T.Ti, T.Te, our_n.n_n, our_n.n);
		p_kappa_n_major[iVertex] = kappa_n;
		p_kappa_i_major[iVertex] = kappa_i;
		p_kappa_e_major[iVertex] = kappa_e;
		p_nu_i_major[iVertex] = nu.i;
		p_nu_e_major[iVertex] = nu.e;

	};
}


__global__ void kernelSetPressureFlag(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	bool * __restrict__ bz_pressureflag
) {
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	long izTri[MAXNEIGH];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	memcpy(izTri, p_izTri + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);

	bool bPressure = false;
	if (info.flag == DOMAIN_VERTEX) bPressure = true;	
#pragma unroll MAXNEIGH
	for (int i = 0; i < info.neigh_len; i++)
	{
		if (p_info_minor[izTri[i]].flag == CROSSING_CATH) bPressure = false;
	}
	bz_pressureflag[iVertex] = bPressure;
}

__global__ void kernelPrepareNuGraphs(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,
	f64 * __restrict__ p_nu_e_MT,
	f64 * __restrict__ p_nu_en_MT
)
{
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_MT;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	species3 nu;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)
		|| (info.flag == CROSSING_CATH))
	{
		our_n = p_n_minor[iMinor];
		T = p_T_minor[iMinor];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_MT = our_n.n_n * sigma_MT * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		
		p_nu_e_MT[iMinor] = nu_en_MT + nu_eiBar; // just doing this roughly, it may not correspond to what is in PopOhms
		p_nu_en_MT[iMinor] = nu_en_MT;

	} else {
		p_nu_e_MT[iMinor] = 0.0;
		p_nu_en_MT[iMinor] = 0.0;
	}
}

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
	f64 * __restrict__ p_graph6) {
	
	long const iVertex = threadIdx.x + blockIdx.x*blockDim.x;
	structural info = p_info_major[iVertex];

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		NTrates NT_rates = ratesNT[iVertex];
		nvals n_nn = p_n_major[iVertex];
		f64 dNvez = p_dNve[iVertex].z;
		f64 Area = p_AreaMajor[iVertex];

		p_graph1[iVertex] = NT_rates.N / Area; // dn/dt
		p_graph2[iVertex] = p_graph1[iVertex] / (n_nn.n);
		p_graph3[iVertex] = log10(n_nn.n);
		p_graph4[iVertex] = NT_rates.NeTe / (Area*n_nn.n); // dT/dt
		p_graph5[iVertex] = dNvez / (Area*n_nn.n); // dvez/dt
		p_graph6[iVertex] = n_nn.n / (n_nn.n + n_nn.n_n); // ionization fraction

	} else {
		p_graph1[iVertex] = 0.0;
		p_graph2[iVertex] = 0.0;
		p_graph3[iVertex] = 0.0;
		p_graph4[iVertex] = 0.0;
		p_graph5[iVertex] = 0.0;
		p_graph6[iVertex] = 0.0;
	};

}

__global__ void kernelKillNeutral_v_OutsideRadius(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n
)
{
	long const index = threadIdx.x + blockIdx.x*blockDim.x;
	structural info = p_info_minor[index];
	if (info.pos.x*info.pos.x + info.pos.y*info.pos.y >
		KILL_NEUTRAL_V_OUTSIDE_TEMP*KILL_NEUTRAL_V_OUTSIDE_TEMP
		)
		memset(&(p_v_n[index]), 0, sizeof(f64_vec3));
}

__global__ void kernelCreateTfromNTbydividing_bysqrtDN(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_sqrtDNn_Tn,
	f64 * __restrict__ p_sqrtDN_Ti,
	f64 * __restrict__ p_sqrtDN_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_sqrtDinv_n, f64 * __restrict__ p_sqrtDinv_i,f64 * __restrict__ p_sqrtDinv_e
)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;

	nvals n = p_n_major[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 sqrtDNnTn = p_sqrtDNn_Tn[iVertex];
	f64 sqrtDNTi = p_sqrtDN_Ti[iVertex];
	f64 sqrtDNTe = p_sqrtDN_Te[iVertex];
	f64 Tn, Ti, Te;
	if (n.n_n*AreaMajor == 0.0) {
		Tn = 0.0;
	} else {
		Tn = sqrtDNnTn *p_sqrtDinv_n[iVertex] / sqrt(AreaMajor*n.n_n);
	}
	p_T_n[iVertex] = Tn;

	if (Tn != Tn) printf("iVertex %d Tn %1.10E area %1.9E \n",
		iVertex, Tn, AreaMajor);

	if (n.n*AreaMajor == 0.0) {
		Ti = 0.0;
		Te = 0.0;
	} else {
		Ti = sqrtDNTi *p_sqrtDinv_i[iVertex] / sqrt(AreaMajor*n.n);
		Te = sqrtDNTe *p_sqrtDinv_e[iVertex] / sqrt(AreaMajor*n.n);
	}
	p_T_i[iVertex] = Ti;
	p_T_e[iVertex] = Te;
}

__global__ void kernelCreateTfromNTbydividing(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_Nn_Tn,
	f64 * __restrict__ p_N_Ti,
	f64 * __restrict__ p_N_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; 

	nvals n = p_n_major[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 NnTn = p_Nn_Tn[iVertex];
	f64 NTi = p_N_Ti[iVertex];
	f64 NTe = p_N_Te[iVertex];
	f64 Tn, Ti, Te;
	if (n.n_n*AreaMajor == 0.0) {
		Tn = 0.0;
	} else {
		Tn = NnTn / sqrt(AreaMajor*n.n_n);
	}
	p_T_n[iVertex] = Tn;
	
	if (Tn != Tn) printf("iVertex %d Tn %1.10E area %1.9E \n",
		iVertex, Tn, AreaMajor);

	if (n.n*AreaMajor == 0.0) {
		Ti = 0.0;
		Te = 0.0;
	} else {
		Ti = NTi / sqrt(AreaMajor*n.n);
		Te = NTe / sqrt(AreaMajor*n.n);
	}
	p_T_i[iVertex] = Ti;
	p_T_e[iVertex] = Te;
}


__global__ void kernelTileMaxMajor(
	f64 * __restrict__ p_z,
	f64 * __restrict__ p_max
) 
{
	__shared__ f64 shared_z[threadsPerTileMajorClever];

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	shared_z[threadIdx.x] = fabs(p_z[iVertex]);	
	
	__syncthreads();

//	if ((blockIdx.x == 0)) printf("iVertex %d threadIdx %d z %1.9E \n",
//		iVertex, threadIdx.x, shared_z[threadIdx.x]);

	int s = blockDim.x;
	int k = s / 2;
	while (s != 1) {
		if (threadIdx.x < k)
		{
			shared_z[threadIdx.x] = max(shared_z[threadIdx.x], shared_z[threadIdx.x + k]);
		//	if (blockIdx.x == 0) printf("s %d thread %d max %1.9E looked at %d : %1.9E\n", s, threadIdx.x, shared_z[threadIdx.x],
		//		threadIdx.x + k, shared_z[threadIdx.x + k]);
		};
		__syncthreads();
		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			shared_z[threadIdx.x] = max(shared_z[threadIdx.x], shared_z[threadIdx.x + s - 1]);
	//		if (blockIdx.x == 0) printf("EXTRA CODE: s %d thread %d max %1.9E \n", s, threadIdx.x, shared_z[threadIdx.x]);
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};
	if (threadIdx.x == 0)
	{
		p_max[blockIdx.x] = shared_z[0];

		if (blockIdx.x == 0) printf("block 0 max %1.10E \n", p_max[blockIdx.x]);
	}
}





__global__ void kernelEstimateCurrent(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	v4 * __restrict__ p_vie,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Iz
) {
	__shared__ f64 Izcell[numTilesMinor];

	// This is what we need rather than making PopOhmsLaw even more bloated.
	// Maybe we can look towards gradually moving some more content into this 1st routine.
	long iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	Izcell[threadIdx.x] = 0.0;

 	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == OUTERMOST) || (info.flag == CROSSING_INS))
	{
		nvals n_use = p_n_minor[iMinor];
		v4 vie = p_vie[iMinor];
		f64 AreaMinor = p_AreaMinor[iMinor];

		Izcell[threadIdx.x] = q*n_use.n*(vie.viz - vie.vez)*AreaMinor;
	}
	
	__syncthreads();
	
	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			Izcell[threadIdx.x] += Izcell[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			Izcell[threadIdx.x] += Izcell[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_Iz[blockIdx.x] = Izcell[0];
	}
}


__global__ void kernelAverage(
	f64 * __restrict__ p_update,
	f64 * __restrict__ p_input2)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_update[index] = 0.5*p_update[index] + 0.5*p_input2[index];
}

__global__ void kernelAdvanceAzEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection)
{

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	AAdot_use.Az += h_use*(AAdot_use.Azdot + ROCAz);
	p_AAdot_dest[index] = AAdot_use;

}

__global__ void kernelAdvanceAzBwdEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	bool const bUseROC)
{

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	AAdot AAdot_dest = p_AAdot_dest[index];
	if (bUseROC) {
		f64 ROCAz = p_ROCAzduetoAdvection[index];
		AAdot_use.Az += h_use*(AAdot_dest.Azdot + ROCAz);
	} else {
		AAdot_use.Az += h_use*(AAdot_dest.Azdot);
	}
	AAdot_use.Azdot = AAdot_dest.Azdot;
	p_AAdot_dest[index] = AAdot_use;
	// So we did not predict how Az would change due to ROCAz -- it's neglected when we solve for A(Adot(LapAz))
}

__global__ void kernelUpdateAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	p_Az[index] += h_use*(AAdot_use.Azdot + ROCAz);
} // perhaps exists a mathematical way to roll up ROC due to advection into our Azdot.

__global__ void kernelPopulateArrayAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	p_Az[index] = AAdot_use.Az + h_use*(AAdot_use.Azdot + ROCAz);
} // perhaps exists a mathematical way to roll up ROC due to advection into our Azdot.

__global__ void kernelPushAzInto_dest(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_AAdot[index].Az = p_Az[index];
} 
__global__ void kernelPullAzFromSyst(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_Az[index] = p_AAdot[index].Az;
}

__global__ void kernelAddtoT(
	T3 * __restrict__ p_T_dest,
	f64 beta_nJ, f64 beta_nR, 
	f64 beta_iJ, f64 beta_iR, 
	f64 beta_eJ, f64 beta_eR,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T = p_T_dest[index];
	T.Tn += beta_nJ*p_Jacobi_n[index] + beta_nR*p_epsilon_n[index];
	T.Ti += beta_iJ*p_Jacobi_i[index] + beta_iR*p_epsilon_i[index];
	T.Te += beta_eJ*p_Jacobi_e[index] + beta_eR*p_epsilon_e[index];

	// Testing to see if - makes it get closer instead of further away
	// It does indeed - but we can't explain why.
	
	p_T_dest[index] = T;
}

__global__ void kernelAddtoT_lc(
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_addition,
	int const howmany
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 T = p__T[iVertex];
	f64 oldT = T;
	for (int i = 0; i < howmany; i++)
		T += beta_n_c[i] * p_addition[i*NUMVERTICES+iVertex];
	p__T[iVertex] = T;
	p_addition[(howmany - 1)*NUMVERTICES + iVertex] = T - oldT;
}

__global__ void kernelAddtoT_lc___(
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_T_src,
	f64 * __restrict__ p_addition,
	int const howmany,
	f64 const multiplier
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 T = p_T_src[iVertex];
	for (int i = 0; i < howmany; i++)
		T += beta_n_c[i] * p_addition[i*NUMVERTICES + iVertex]*multiplier;
	p__T[iVertex] = T;

}
__global__ void kernelAdd_to_T_lc(
	f64 * __restrict__ p__T,
	f64 const beta1,
	f64 * __restrict__ p_add1,
	f64 const beta2,
	f64 * __restrict__ p_add2
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	p__T[iVertex] += beta1*p_add1[iVertex] + beta2*p_add2[iVertex];
}

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
) {
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T = p_T_dest[iVertex];
	char iVolley = p_iVolley[iVertex];
	switch (iVolley) {
	case 0:
		T.Tn += beta_n_c[0] * p_Jacobi_n[iVertex] + beta_n_c[4] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[0] * p_Jacobi_i[iVertex] + beta_i_c[4] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[0] * p_Jacobi_e[iVertex] + beta_e_c[4] * p_epsilon_e[iVertex];
		break;
	case 1:
		T.Tn += beta_n_c[1] * p_Jacobi_n[iVertex] + beta_n_c[5] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[1] * p_Jacobi_i[iVertex] + beta_i_c[5] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[1] * p_Jacobi_e[iVertex] + beta_e_c[5] * p_epsilon_e[iVertex];
		break;
	case 2:
		T.Tn += beta_n_c[2] * p_Jacobi_n[iVertex] + beta_n_c[6] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[2] * p_Jacobi_i[iVertex] + beta_i_c[6] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[2] * p_Jacobi_e[iVertex] + beta_e_c[6] * p_epsilon_e[iVertex];
		break;
	case 3:
		T.Tn += beta_n_c[3] * p_Jacobi_n[iVertex] + beta_n_c[7] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[3] * p_Jacobi_i[iVertex] + beta_i_c[7] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[3] * p_Jacobi_e[iVertex] + beta_e_c[7] * p_epsilon_e[iVertex];
		break;
	}
	p_T_dest[iVertex] = T;
}

__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_updated[index] += beta * p_added[index];
}

__global__ void kernelAdd2(
	f64 * __restrict__ p_result,
	f64 * __restrict__ p_src,
	f64 const beta_,
	f64 * __restrict__ p_addition
)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_result[index] = p_src[index] + beta_ * p_addition[index];
}


__global__ void kernelAdd_to_v(
	v4 * __restrict__ p_vie,
	f64 const beta_i, f64 const beta_e,
	f64_vec3 * __restrict__ p_vJacobi_ion,
	f64_vec3 * __restrict__ p_vJacobi_elec
)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 vie;
	memcpy(&vie, &(p_vie[index]), sizeof(v4));
	f64_vec3 vJ_ion = p_vJacobi_ion[index];
	f64_vec3 vJ_elec = p_vJacobi_elec[index];
	vie.vxy.x += beta_i*vJ_ion.x + beta_e*vJ_elec.x;
	vie.vxy.y += beta_i*vJ_ion.y + beta_e*vJ_elec.y;
	vie.viz += beta_i*vJ_ion.z;
	vie.vez += beta_e*vJ_elec.z;
	memcpy(&(p_vie[index]), &vie, sizeof(v4));
}


// Try resetting frills here and ignoring in calculation:
__global__ void kernelResetFrillsAz(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	
	if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		if (info.flag == INNER_FRILL)
		{
			p_Az[index] = p_Az[izNeigh.i1]; 
	//		printf("hello I'm a frill. Tri %d equal to tri %d : %1.10E \n",
	//			index, izNeigh.i1, p_Az[index]);

		} else {			
				
#if RADIALDECLINE
			f64 r = p_info[izNeigh.i1].pos.modulus();
		 	p_Az[index] = (r/ FRILL_CENTROID_OUTER_RADIUS_d)*p_Az[izNeigh.i1]; // should be something like 0.99*p_Az[izNeigh.i1]
			// Better if we store a constant called Outer_Frill_Factor to save a load and a division.
#else
			if (DIRICHLET == false)
			{
				p_Az[index] = p_Az[izNeigh.i1];
			}
			else {
				p_Az[index] = 0.0;
			}
#endif
#ifdef LOG_INCLINE
			// Alternative to try: put A = C ln r ...

			// The result is that it swung up and down vez with oscillations about every 0.7ns for a full cycle
			// We would rather not see this bucking phenomenon.

			f64 r = p_info[izNeigh.i1].pos.modulus();
			f64 lnr = log(r);
			f64 ln_ours = log(FRILL_CENTROID_OUTER_RADIUS_d);

			// ln_ours is a GREATER value.
			// ln r is increasing in r --- what gives?
			// want A'' = - r A' .
			// Let's just try it.
			p_Az[index] = (ln_ours / lnr)*p_Az[izNeigh.i1]; // GREATER
			
#endif		
		};
	};
}

__global__ void kernelAddToAz(
	long * p_indicator,
	f64 * pAz
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	long ind = p_indicator[index];
	if (ind > 0)
		pAz[index] += beta_n_c[ind - 1];

}
__global__ void kernelReturnMaximumInBlock
(
	f64 * __restrict__ p_f,
	f64 * __restrict__ p_outputmax,
	long * __restrict__ p_outputiMax,
	long * __restrict__ p_indic
) {
	__shared__ f64 shared_f[threadsPerTileMinor];
	__shared__ long iMax[threadsPerTileMinor];

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	shared_f[threadIdx.x] = fabs(p_f[index]);
	iMax[threadIdx.x] = index;
	if (p_indic[index] != 0) shared_f[threadIdx.x] = 0.0;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			if (shared_f[threadIdx.x] < shared_f[threadIdx.x + k])
			{
				shared_f[threadIdx.x] = shared_f[threadIdx.x + k];
				iMax[threadIdx.x] = iMax[threadIdx.x + k];
			}
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {

			if (shared_f[threadIdx.x] < shared_f[threadIdx.x + s - 1])
			{
				shared_f[threadIdx.x] = shared_f[threadIdx.x + s - 1];
				iMax[threadIdx.x] = iMax[threadIdx.x + s - 1];
			}
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_outputmax[blockIdx.x] = shared_f[threadIdx.x];
		p_outputiMax[blockIdx.x] = iMax[threadIdx.x];
	};

}



__global__ void kernelReset_v_in_outer_frill_and_outermost
(
	structural * __restrict__ p_info,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	T3 * __restrict__ p_T_minor,
	LONG3 * __restrict__ trineighbourindex,
	long * __restrict__ p_izNeigh_vert
	) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		p_vie[index] = p_vie[izNeigh.i1];
		p_v_n[index] = p_v_n[izNeigh.i1];
		p_T_minor[index] = p_T_minor[izNeigh.i1];
		// memcpy may be more efficient than operator =.

		// This is fine to have here since outer_frill v should never be used for anything really.

	}
	if ((info.flag == OUTERMOST))
	{
		// all we want to say at outermost is that vxy == 0.

		v4 vie = p_vie[index];
		f64_vec3 v_n = p_v_n[index];
		vie.vxy.x = 0.0;
		vie.vxy.y = 0.0;
		v_n.x = 0.0;
		v_n.y = 0.0;
		p_vie[index] = vie;
		p_v_n[index] = v_n;

		// No reason to do anything else special about here.
		// Make it like a wall.

		/*

		long izNeigh[MAXNEIGH_d];
		long iVertex = index - BEGINNING_OF_CENTRAL;
		memcpy(izNeigh, p_izNeigh_vert + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		v4 result, temp4;
		f64_vec3 v_n, temp3;
		T3 T, temp5;
		memset(&result, 0, sizeof(v4));
		memset(&v_n, 0, sizeof(f64_vec3));
		memset(&T, 0, sizeof(T3));
		long iDomain = 0;
		for (short i = 0; i < 4; i++)
		{
			structural infoneigh = p_info[izNeigh[i] + BEGINNING_OF_CENTRAL];
			if (infoneigh.flag == DOMAIN_VERTEX)
			{
				temp4 = p_vie[izNeigh[i] + BEGINNING_OF_CENTRAL];
				temp3 = p_v_n[izNeigh[i] + BEGINNING_OF_CENTRAL];
				temp5 = p_T_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				iDomain++;
				result.vxy += temp4.vxy;
				result.vez += temp4.vez;
				result.viz += temp4.viz;
				v_n += temp3;
				T.Tn += temp5.Tn;
				T.Ti += temp5.Ti;
				T.Te += temp5.Te;
			}
		}
		if (iDomain > 0) {
			f64 fac = 1.0 / (f64)iDomain;
			result.vxy *= fac;
			result.vez *= fac;
			result.viz *= fac;
			v_n *= fac;
			T.Tn *= fac;
			T.Ti *= fac;
			T.Te *= fac;
		}
		p_vie[index] = result;
		p_v_n[index] = v_n;
		p_T_minor[index] = T;
		*/
	}
}

// Try resetting frills here and ignoring in calculation:
__global__ void kernelResetFrillsAz_II(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	AAdot * __restrict__ p_Az)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		p_Az[index].Az = p_Az[izNeigh.i1].Az;
	}

//	need to change this if we want to use it for RADIALDECLINE.



}


__global__ void kernelCreateAzbymultiplying(
	f64 * __restrict__ p_Az,
	f64 * __restrict__ p_scaledAz,
	f64 * __restrict__ p_factor
)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;	
	p_Az[iMinor] = p_scaledAz[iMinor] * p_factor[iMinor];
}

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
	f64_vec3 * __restrict__ p_eps_against_deps)

{
	__shared__ f64 sum_mat[threadsPerTileMinor][6];
	__shared__ f64 sum_eps_deps[threadsPerTileMinor][3];

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 d_eps_by_d_beta1, d_eps_by_d_beta2, d_eps_by_d_beta3;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		d_eps_by_d_beta1 = 0.0;
		d_eps_by_d_beta2 = 0.0;
		d_eps_by_d_beta3 = 0.0; // eps here actually is 0.
	}
	else {
		d_eps_by_d_beta1 = (p_regressor1[index] - h_use * p_gamma[index] * p_LapReg1[index]);
		d_eps_by_d_beta2 = (p_regressor2[index] - h_use * p_gamma[index] * p_LapReg2[index]);
		d_eps_by_d_beta3 = (p_regressor3[index] - h_use * p_gamma[index] * p_LapReg3[index]);
	};
	sum_mat[threadIdx.x][0] = d_eps_by_d_beta1*d_eps_by_d_beta1;
	sum_mat[threadIdx.x][1] = d_eps_by_d_beta1*d_eps_by_d_beta2;
	sum_mat[threadIdx.x][2] = d_eps_by_d_beta1*d_eps_by_d_beta3;
	sum_mat[threadIdx.x][3] = d_eps_by_d_beta2*d_eps_by_d_beta2;
	sum_mat[threadIdx.x][4] = d_eps_by_d_beta2*d_eps_by_d_beta3;
	sum_mat[threadIdx.x][5] = d_eps_by_d_beta3*d_eps_by_d_beta3;
	f64 eps = p_epsilon[index];
	sum_eps_deps[threadIdx.x][0] = eps*d_eps_by_d_beta1;
	sum_eps_deps[threadIdx.x][1] = eps*d_eps_by_d_beta2;
	sum_eps_deps[threadIdx.x][2] = eps*d_eps_by_d_beta3;

	if (sum_eps_deps[threadIdx.x][0] != sum_eps_deps[threadIdx.x][0])
		printf("index %d sum_eps_deps[threadIdx.x][0] == NAN\n"
			"p_regressor1[index] %1.8E gamma %1.8E LapReg %1.8E \n",
			index, p_regressor1[index],
			p_gamma[index], p_LapReg1[index]);

	if (sum_eps_deps[threadIdx.x][1] != sum_eps_deps[threadIdx.x][1])
		printf("index %d sum_eps_deps[threadIdx.x][1] == NAN\n"
			"p_regressor2[index] %1.8E gamma %1.8E LapReg %1.8E \n",
			index, p_regressor2[index],
			p_gamma[index], p_LapReg2[index]);

	if (sum_eps_deps[threadIdx.x][2] != sum_eps_deps[threadIdx.x][2])
		printf("index %d sum_eps_deps[threadIdx.x][2] == NAN eps %1.10E d_eps_by_dbeta3 %1.10E\n"
			"p_regressor3[index] %1.8E gamma %1.8E LapReg %1.8E \n",
			index, eps, d_eps_by_d_beta3,
			p_regressor3[index],
			p_gamma[index], p_LapReg3[index]);

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
#pragma unroll
			for (int y = 0; y < 6; y++)
				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + k][y];
			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + k][0];
			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + k][1];
			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + k][2];

		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 6; y++)
				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + s - 1][y];
			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + s - 1][0];
			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + s - 1][1];
			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + s - 1][2];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_deps_matrix[6 * blockIdx.x]), sum_mat[0], sizeof(f64) * 6);
		f64_vec3 tempvec3;
		tempvec3.x = sum_eps_deps[0][0];
		tempvec3.y = sum_eps_deps[0][1];
		tempvec3.z = sum_eps_deps[0][2];

		memcpy(&p_eps_against_deps[blockIdx.x], &tempvec3, sizeof(f64_vec3));
	}
}

__global__ void GetMax(
		f64 * __restrict__ p_comp1,
		long * __restrict__ p_iWhich,
		f64 * __restrict__ p_max
	)
	{
		__shared__ f64 comp[threadsPerTileMinor];
		__shared__ long longarray[threadsPerTileMinor];

		long const iMinor = threadIdx.x + blockDim.x*blockIdx.x;
		comp[threadIdx.x] = fabs(p_comp1[iMinor]);
		longarray[threadIdx.x] = iMinor;
		__syncthreads();

		int s = blockDim.x;
		int k = s / 2;

		while (s != 1) {
			if (threadIdx.x < k)
			{
#pragma unroll

				if (comp[threadIdx.x] > comp[threadIdx.x + k])
				{
					// do nothing	
				}
				else {
					comp[threadIdx.x] = comp[threadIdx.x + k];
					longarray[threadIdx.x] = longarray[threadIdx.x + k];
				};

			};
			__syncthreads();

			// Modify for case blockdim not 2^n:
			if ((s % 2 == 1) && (threadIdx.x == k - 1)) {

				if (comp[threadIdx.x] > comp[threadIdx.x + s - 1])
				{
					// do nothing	
				}
				else {
					comp[threadIdx.x] = comp[threadIdx.x + s - 1];
					longarray[threadIdx.x] = longarray[threadIdx.x + s - 1];
				};
			};
			// In case k == 81, add [39] += [80]
			// Otherwise we only get to 39+40=79.
			s = k;
			k = s / 2;
			__syncthreads();
		};

		if (threadIdx.x == 0)
		{
			p_iWhich[blockIdx.x] = longarray[0];
			p_max[blockIdx.x] = comp[0];
		}
	}

__global__ void VectorCompareMax(
	f64 * __restrict__ p_comp1,
	f64 * __restrict__ p_comp2,
	long * __restrict__ p_iWhich,
	f64 * __restrict__ p_max
)
{
	__shared__ f64 diff[threadsPerTileMajorClever];
	__shared__ long longarray[threadsPerTileMajorClever];

	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;
	diff[threadIdx.x] = fabs(p_comp1[iVertex] - p_comp2[iVertex]);
	longarray[threadIdx.x] = iVertex;
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
#pragma unroll
			
			if (diff[threadIdx.x] > diff[threadIdx.x + k])
			{
				// do nothing	
			} else {
				diff[threadIdx.x] = diff[threadIdx.x + k];
				longarray[threadIdx.x] = longarray[threadIdx.x + k];
			};
			
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			
			if (diff[threadIdx.x] > diff[threadIdx.x + s-1])
			{
				// do nothing	
			} else {
				diff[threadIdx.x] = diff[threadIdx.x + s-1];
				longarray[threadIdx.x] = longarray[threadIdx.x + s-1];
			};
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_iWhich[blockIdx.x] = longarray[0];
		p_max[blockIdx.x] = diff[0];
	}
}

__global__ void AddLCtoT(f64 * __restrict__ p_T_,
	f64 * __restrict__ p_T_use,
	f64 const bet, f64 * __restrict__ p_add) {
	long const iVertex = blockIdx.x*blockDim.x + threadIdx.x;
	p_T_[iVertex] = p_T_use[iVertex] + p_add[iVertex] * bet;
}
__global__ void AddLC(
	f64 * __restrict__ p_result,
	f64 * __restrict__ p_summand1,
	f64 const betaa,
	f64 * __restrict__ p_summand2) 
{
	long const iVertex = blockIdx.x*blockDim.x + threadIdx.x;
	p_result[iVertex] = p_summand1[iVertex] + betaa*p_summand2[iVertex];
}



//
//__global__ void kernelAccumulateMatrix_debug(
//	structural * __restrict__ p_info,
//	f64 const h_use,
//	f64 * __restrict__ p_epsilon,
//	f64 * __restrict__ p_regressor1,
//	f64 * __restrict__ p_regressor2,
//	f64 * __restrict__ p_regressor3,
//	f64 * __restrict__ p_LapReg1,
//	f64 * __restrict__ p_LapReg2,
//	f64 * __restrict__ p_LapReg3,
//	f64 * __restrict__ p_gamma,
//	f64 * __restrict__ p_deps_matrix,
//	f64_vec3 * __restrict__ p_eps_against_deps,
//	
//	f64 * __restrict__ p_deps_1,
//	f64 * __restrict__ p_deps_2,
//	f64 * __restrict__ p_deps_3
//
//	)
//	
//{
//	__shared__ f64 sum_mat[threadsPerTileMinor][6];
//	__shared__ f64 sum_eps_deps[threadsPerTileMinor][3];
//	
//	long const index = blockDim.x*blockIdx.x + threadIdx.x;
//	f64 d_eps_by_d_beta1, d_eps_by_d_beta2, d_eps_by_d_beta3;
//	structural info = p_info[index];
//	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
//	{
//		d_eps_by_d_beta1 = 0.0; 
//		d_eps_by_d_beta2 = 0.0;
//		d_eps_by_d_beta3 = 0.0; // eps here actually is 0.
//	} else {
//		d_eps_by_d_beta1 = (p_regressor1[index] - h_use * p_gamma[index] * p_LapReg1[index]);
//		d_eps_by_d_beta2 = (p_regressor2[index] - h_use * p_gamma[index] * p_LapReg2[index]);
//		d_eps_by_d_beta3 = (p_regressor3[index] - h_use * p_gamma[index] * p_LapReg3[index]);
//	};
//	sum_mat[threadIdx.x][0] = d_eps_by_d_beta1*d_eps_by_d_beta1;
//	sum_mat[threadIdx.x][1] = d_eps_by_d_beta1*d_eps_by_d_beta2;
//	sum_mat[threadIdx.x][2] = d_eps_by_d_beta1*d_eps_by_d_beta3;
//	sum_mat[threadIdx.x][3] = d_eps_by_d_beta2*d_eps_by_d_beta2;
//	sum_mat[threadIdx.x][4] = d_eps_by_d_beta2*d_eps_by_d_beta3;
//	sum_mat[threadIdx.x][5] = d_eps_by_d_beta3*d_eps_by_d_beta3;
//	f64 eps = p_epsilon[index];
//	sum_eps_deps[threadIdx.x][0] = eps*d_eps_by_d_beta1;
//	sum_eps_deps[threadIdx.x][1] = eps*d_eps_by_d_beta2;
//	sum_eps_deps[threadIdx.x][2] = eps*d_eps_by_d_beta3;
//
//	p_deps_1[index] = d_eps_by_d_beta1;
//	p_deps_2[index] = d_eps_by_d_beta2;
//	p_deps_3[index] = d_eps_by_d_beta3;
//
//	__syncthreads();
//
//	int s = blockDim.x;
//	int k = s / 2;
//
//	while (s != 1) {
//		if (threadIdx.x < k)
//		{
//#pragma unroll
//			for (int y = 0; y < 6; y++)
//				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + k][y];
//			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + k][0];
//			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + k][1];
//			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + k][2];
//			
//		};
//		__syncthreads();
//
//		// Modify for case blockdim not 2^n:
//		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
//			for (int y = 0; y < 6; y++)
//				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + s - 1][y];
//			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + s - 1][0];
//			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + s - 1][1];
//			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + s - 1][2];			
//		};
//		// In case k == 81, add [39] += [80]
//		// Otherwise we only get to 39+40=79.
//		s = k;
//		k = s / 2;
//		__syncthreads();
//	};
//
//	if (threadIdx.x == 0)
//	{
//		memcpy(&(p_deps_matrix[6 * blockIdx.x]), sum_mat[0], sizeof(f64) * 6);
//		f64_vec3 tempvec3;
//		tempvec3.x = sum_eps_deps[0][0];
//		tempvec3.y = sum_eps_deps[0][1];
//		tempvec3.z = sum_eps_deps[0][2];
//		
//		memcpy(&p_eps_against_deps[blockIdx.x], &tempvec3, sizeof(f64_vec3));
//	}
//}

__global__ void kernelAddRegressors(
	f64 * __restrict__ p_AzNext,
	f64 const beta0, f64 const beta1, f64 const beta2,
	f64 * __restrict__ p_reg1,
	f64 * __restrict__ p_reg2,
	f64 * __restrict__ p_reg3)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	p_AzNext[iMinor] += beta0*p_reg1[iMinor] + beta1*p_reg2[iMinor] + beta2*p_reg3[iMinor];
}

__global__ void kernelAccumulateSummands(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi,
	f64 * __restrict__ p_LapJacobi,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMinor];
	__shared__ f64 sumdata2[threadsPerTileMinor];
	__shared__ f64 sumdata3[threadsPerTileMinor];

	f64 depsbydbeta;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
	}
	else {
//#ifdef MIDPT_A_AND_ACTUALLY_MIDPT_A_NOT_JUST_EFFECT_ON_AZDOT
//		depsbydbeta = (p_Jacobi[index] - 0.5*h_use * p_gamma[index] * p_LapJacobi[index]);
//#else
		depsbydbeta = (p_Jacobi[index] - h_use * p_gamma[index] * p_LapJacobi[index]);
//#endif
	};
	f64 eps = p_epsilon[index];
	sumdata1[threadIdx.x] = depsbydbeta * eps;
	sumdata2[threadIdx.x] = depsbydbeta * depsbydbeta;
	sumdata3[threadIdx.x] = eps * eps;

	__syncthreads();
	
	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};
	
	if (threadIdx.x == 0)
	{
		p_sum_eps_d[blockIdx.x] = sumdata1[0];
		p_sum_d_d[blockIdx.x] = sumdata2[0];
		p_sum_eps_eps[blockIdx.x] = sumdata3[0];
	}
}



__global__ void kernelAccumulateSumOfSquares(
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_SS_n,
	f64 * __restrict__ p_SS_i,
	f64 * __restrict__ p_SS_e)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	__shared__ f64 sumdata2[threadsPerTileMajorClever];
	__shared__ f64 sumdata3[threadsPerTileMajorClever];

	f64 epsilon_n = p_eps_n[index];
	f64 epsilon_i = p_eps_i[index];
	f64 epsilon_e = p_eps_e[index];

	sumdata1[threadIdx.x] = epsilon_n*epsilon_n;
	sumdata2[threadIdx.x] = epsilon_i*epsilon_i;
	sumdata3[threadIdx.x] = epsilon_e*epsilon_e;
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS_n[blockIdx.x] = sumdata1[0];
		p_SS_i[blockIdx.x] = sumdata2[0];
		p_SS_e[blockIdx.x] = sumdata3[0];
	}
}


__global__ void kernelAccumulateSumOfSquares1(
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];

	f64 epsilon_n = p_eps[index];

	sumdata1[threadIdx.x] = epsilon_n*epsilon_n;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}


__global__ void kernelAccumulateSumOfSquares2vec(
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];

	f64_vec2 epsilon_n = p_eps[index];

	sumdata1[threadIdx.x] = epsilon_n.dot(epsilon_n);

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}

__global__ void kernelAccumulateSumOfSquares_4(
	v4 * __restrict__ p_eps,
	f64 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];

	v4 eps = p_eps[index];

	sumdata1[threadIdx.x] =
		eps.vxy.dot(eps.vxy) + eps.viz*eps.viz + eps.vez*eps.vez;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}


__global__ void ScaleVector4(
	v4 * __restrict__ p_eps,
	f64 const factor)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 vec4 = p_eps[index];
	vec4.vxy *= factor;
	vec4.viz *= factor;
	vec4.vez *= factor;
	p_eps[index] = vec4;
}

__global__ void ScaleVector3(
	f64_vec3 * __restrict__ p_eps,
	f64 const factorx, f64 const factory, f64 const factorz)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec3 vec3 = p_eps[index];
	vec3.x *= factorx;
	vec3.y *= factory;
	vec3.z *= factorz;
	p_eps[index] = vec3;
}
	
__global__ void kernelAccumulateSumOfSquares3(
	f64_vec3 * __restrict__ p_eps,
	f64_vec3 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64_vec3 sumdata1[threadsPerTileMajorClever];

	f64_vec3 epsilon_n = p_eps[index];

	sumdata1[threadIdx.x].x = epsilon_n.x*epsilon_n.x;
	sumdata1[threadIdx.x].y = epsilon_n.y*epsilon_n.y;
	sumdata1[threadIdx.x].z = epsilon_n.z*epsilon_n.z;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}

__global__ void AssembleVector4(v4 * __restrict__ p_output,
	f64_vec2 * __restrict__ p_xy, f64 * __restrict__ p_iz, f64 * __restrict__ p_ez)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 vec4;
	vec4.vxy = p_xy[index];
	vec4.viz = p_iz[index];
	vec4.vez = p_ez[index];
	p_output[index] = vec4;
}



__global__ void AssembleVector3(f64_vec3 * __restrict__ p_output,
	f64 * __restrict__ p_x, f64 * __restrict__ p_y, f64 * __restrict__ p_z)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec3 vec3;
	vec3.x = p_x[index];
	vec3.y = p_y[index];
	vec3.z = p_z[index];
	p_output[index] = vec3;
}

__global__ void SubtractVector4(v4 * __restrict__ p_output,
	v4 * __restrict__ p_a, v4 * __restrict__ p_b)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 result;
	v4 a = p_a[index];
	v4 b = p_b[index];
	result.vxy = a.vxy - b.vxy;
	result.viz = a.viz - b.viz;
	result.vez = a.vez - b.vez;
	p_output[index] = result;
}


__global__ void SubtractVector3stuff(f64_vec3 * __restrict__ p_output,
	f64 * __restrict__ p_x, f64 * __restrict__ p_y, f64 * __restrict__ p_z,
	f64_vec3 * __restrict__ p_to_subtract)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec3 vec3;
	vec3.x = p_x[index];
	vec3.y = p_y[index];
	vec3.z = p_z[index];
	vec3 -= p_to_subtract[index];
	p_output[index] = vec3;
}
__global__ void SubtractVector4stuff(v4 * __restrict__ p_output,
	f64_vec2 * __restrict__ p_xy, 
	f64 * __restrict__ p_iz, 
	f64 * __restrict__ p_ez,
	v4 * __restrict__ p_to_subtract)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 vec4;
	vec4.vxy = p_xy[index];
	vec4.viz = p_iz[index];
	vec4.vez = p_ez[index];
	v4 st = p_to_subtract[index];
	vec4.vxy -= st.vxy;
	vec4.viz -= st.viz;
	vec4.vez -= st.vez;

	p_output[index] = vec4;
}



__global__ void SubtractVector3(f64_vec3 * __restrict__ p_output,
	f64_vec3 * __restrict__ p_a, f64_vec3 * __restrict__ p_b)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_output[index] = p_a[index] - p_b[index];
}

__global__ void MultiplyVector(
	f64 * __restrict__ multiply1,
	f64 * __restrict__ multiply2,
	f64 * __restrict__ output
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	output[index] = multiply1[index] * multiply2[index];
}

__global__ void ScaleVector(
	f64 * __restrict__ output,
	f64 const coeff,
	f64 * __restrict__ multiply
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	output[index] = multiply[index] * coeff;
}

__global__ void kernelAccumulateSumOfQuarts(
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];

	f64 epsilon_n = p_eps[index];

	sumdata1[threadIdx.x] = epsilon_n*epsilon_n*epsilon_n*epsilon_n;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}
__global__ void kernelAccumulateDotProducts(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_x2, f64 * __restrict__ p_y2,
	f64 * __restrict__ p_x3, f64 * __restrict__ p_y3,
	f64 * __restrict__ p_dot1,
	f64 * __restrict__ p_dot2,
	f64 * __restrict__ p_dot3)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	__shared__ f64 sumdata2[threadsPerTileMajorClever];
	__shared__ f64 sumdata3[threadsPerTileMajorClever];

	f64 x1 = p_x1[index];
	f64 x2 = p_x2[index];
	f64 x3 = p_x3[index];
	f64 y1 = p_y1[index];
	f64 y2 = p_y2[index];
	f64 y3 = p_y3[index];

	sumdata1[threadIdx.x] = x1*y1;
	sumdata2[threadIdx.x] = x2*y2;
	sumdata3[threadIdx.x] = x3*y3;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_dot1[blockIdx.x] = sumdata1[0];
		p_dot2[blockIdx.x] = sumdata2[0];
		p_dot3[blockIdx.x] = sumdata3[0];
	}
}

__global__ void kernelAccumulateDotProduct(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_dot1)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	
	f64 x1 = p_x1[index];
	f64 y1 = p_y1[index];
	
	sumdata1[threadIdx.x] = x1*y1;
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_dot1[blockIdx.x] = sumdata1[0];
	}
}

__global__ void VectorAddMultiple1(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	p_T1[iVertex] += alpha1*p_x1[iVertex];
}

__global__ void CreateConjugateRegressor(
	f64 * __restrict__ p_regr, f64 const betarat, f64 * __restrict__ p_eps)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 x = p_regr[index];
	x = betarat*x + p_eps[index];
	p_regr[index] = x;
}

__global__ void VectorAddMultiple(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	//if (iVertex == VERTCHOSEN) printf("%d Ti %1.10E ", iVertex, p_T2[iVertex]);
	
	p_T1[iVertex] += alpha1*p_x1[iVertex];
	p_T2[iVertex] += alpha2*p_x2[iVertex];
	p_T3[iVertex] += alpha3*p_x3[iVertex];

	//if (iVertex == VERTCHOSEN) printf("alpha2 %1.12E x2 %1.12E result %1.12E\n",
	//	alpha2, p_x2[iVertex], p_T2[iVertex]);
	
}

__global__ void VectorAddMultiple_masked(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock,
	bool const bUseMask)
{
	if ((bUseMask) && (p_bMaskblock[blockIdx.x] == 0)) return;

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	//if (iVertex == VERTCHOSEN) printf("%d Ti %1.10E ", iVertex, p_T2[iVertex]);
	if (bUseMask) {

		bool bMask[3];
		bMask[0] = p_bMask[iVertex];
		bMask[1] = p_bMask[iVertex + NUMVERTICES];
		bMask[2] = p_bMask[iVertex + NUMVERTICES*2];
		if (bMask[0]) p_T1[iVertex] += alpha1*p_x1[iVertex];
		if (bMask[1]) p_T2[iVertex] += alpha2*p_x2[iVertex];

		if (iVertex == VERTCHOSEN) printf("%d old T : %1.12E ", iVertex, p_T3[iVertex]);

		if (bMask[2]) p_T3[iVertex] += alpha3*p_x3[iVertex];

		if (iVertex == VERTCHOSEN) printf("alpha3 %1.12E x3 %1.12E new T %1.12E \n",
			alpha3, p_x3[iVertex], p_T3[iVertex]);

	} else {
		p_T1[iVertex] += alpha1*p_x1[iVertex];
		p_T2[iVertex] += alpha2*p_x2[iVertex];
		p_T3[iVertex] += alpha3*p_x3[iVertex];
	}
	//if (iVertex == VERTCHOSEN) printf("alpha2 %1.12E x2 %1.12E result %1.12E\n",
	//	alpha2, p_x2[iVertex], p_T2[iVertex]);

}
__global__ void kernelRegressorUpdate
(
	f64 * __restrict__ p_x_n,
	f64 * __restrict__ p_x_i,
	f64 * __restrict__ p_x_e,
	f64 * __restrict__ p_a_n, f64 * __restrict__ p_a_i, f64 * __restrict__ p_a_e,
	f64 const ratio1, f64 const ratio2, f64 const ratio3,
	bool * __restrict__ p_bMaskBlock,
	bool bUseMask
	)
{
	if ((bUseMask) && (p_bMaskBlock[blockIdx.x] == 0)) return;

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	
	f64 xn = p_x_n[iVertex];
	p_x_n[iVertex] = p_a_n[iVertex] + ratio1*xn;
	f64 xi = p_x_i[iVertex];
	p_x_i[iVertex] = p_a_i[iVertex] + ratio2*xi;
	f64 xe = p_x_e[iVertex];
	p_x_e[iVertex] = p_a_e[iVertex] + ratio3*xe;
}

__global__ void kernelPackupT3(
	T3 * __restrict__ p_T,
	f64 * __restrict__ p_Tn, f64 * __restrict__ p_Ti, f64 * __restrict__ p_Te)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T;
	T.Tn = p_Tn[iVertex];
	T.Ti = p_Ti[iVertex];
	T.Te = p_Te[iVertex];
	p_T[iVertex] = T;
}

__global__ void kernelAccumulateSummands2(
	structural * __restrict__ p_info,
	
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMinor];
	__shared__ f64 sumdata2[threadsPerTileMinor];
	__shared__ f64 sumdata3[threadsPerTileMinor];

	f64 eps = p_epsilon[index];
	f64 depsbydbeta = p_d_eps_by_dbeta[index];

	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
		depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
	// could rearrange to not have to do that or load info.

	sumdata1[threadIdx.x] = depsbydbeta * eps;
	sumdata2[threadIdx.x] = depsbydbeta * depsbydbeta;
	sumdata3[threadIdx.x] = eps * eps;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_d[blockIdx.x] = sumdata1[0];
		p_sum_d_d[blockIdx.x] = sumdata2[0];
		p_sum_eps_eps[blockIdx.x] = sumdata3[0];
	}
}


__global__ void kernelDividebyroothgamma
(
	f64 * __restrict__ result,
	f64 * __restrict__ p__sqrtfactor,
	f64 * __restrict__ Az,
	f64 const hsub,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_Area
)
{
	long const index = threadIdx.x + blockIdx.x*blockDim.x;
	f64 gamma = p_gamma[index];
	f64 area = p_Area[index];
	if (area == 0.0) {
		result[index] = 0.0;
		p__sqrtfactor[index] = 0.0;
	} else {
		f64 sqrtfactor = sqrt(hsub*gamma / area);
		p__sqrtfactor[index] = sqrtfactor;

		
		if (sqrtfactor != sqrtfactor) printf("%d sqrtfactor %1.8E hsub %1.8E gamma %1.8E area %1.8E \n",
			index, sqrtfactor, hsub, gamma, area);


		if (sqrtfactor == 0.0) {
			result[index] = 0.0;
		} else {
			result[index] = Az[index] / sqrtfactor;
		};
	};
}


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
	)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	structural info1 = p_info1[iMinor];
	structural info2 = p_info2[iMinor];
	structural info;
	f64 r = 1.0 - ppn;
	info.pos = r*info1.pos + ppn*info2.pos;
	info.flag = info1.flag;
	info.neigh_len = info1.neigh_len;
	p_info_dest[iMinor] = info;

	nvals nvals1 = p_n_minor1[iMinor];
	nvals nvals2 = p_n_minor2[iMinor];
	nvals nvals_dest;
	nvals_dest.n = r*nvals1.n + ppn*nvals2.n;
	nvals_dest.n_n = r*nvals1.n_n + ppn*nvals2.n_n;
	p_n_minor[iMinor] = nvals_dest;

	T3 T1 = p_T_minor1[iMinor];
	T3 T2 = p_T_minor2[iMinor];
	T3 T;
	T.Te = r*T1.Te + ppn*T2.Te;
	T.Ti = r*T1.Ti + ppn*T2.Ti;
	T.Tn = r*T1.Tn + ppn*T2.Tn;
	p_T_minor[iMinor] = T;

//	f64_vec3 B1 = p_B1[iMinor];
//	f64_vec3 B2 = p_B2[iMinor];
//	f64_vec3 B = r*B1 + ppn*B2;
//	p_B[iMinor] = B;
}

// Correct disposition of routines:
// --- union of T and [v + v_overall] -- uses n_shards --> pressure, momflux, grad Te
// --- union of T and [v + v_overall] -- uses n_n shards --> neutral pressure, neutral momflux
// --- Az,Azdot + v_overall -- runs for whole domain ---> Lap A, curl A, grad A, grad Adot, ROCAz, ROCAzdot
//    ^^ base off of GetLap_minor.

// Worst case number of vars:
// (4+2)*1.5+6.5 <-- because we use v_vertex. + 3 for positions. 
// What can we stick in L1? n_cent we could.
// We should be aiming a ratio 3:1 from shared:L1, if registers are small.
// For tris we are using n_shards from shared points.
// And it is for tris that we require vertex data v to be present.
// Idea: vertex code determines array of 12 relevant n and sticks them into shared.
// Only saved us 1 var. 9 + 6 + 3 = 18.
// Still there is premature optimization here -- none of this happens OFTEN.

__global__ void DivideVec2(f64_vec2 * __restrict__ p_update,
	f64_vec2 * __restrict__ p_apply, f64_vec2 * __restrict__ p_numer)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	f64_vec2 jill;
	f64_vec2 jill2 = p_apply[index];
	f64_vec2 numer = p_numer[index];

	jill.x = numer.x / (jill2.x + 1.0);
	jill.y = numer.y / (jill2.y + 1.0);

	p_update[index] = jill;
}

__global__ void Divide(f64 * __restrict__ p_outputz, f64 * __restrict__ p_denom, f64 * __restrict__ p_numer)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	p_outputz[index] = p_numer[index] / (p_denom[index] + 1.0);
}

__global__ void kernelDivideAdd3things(f64_vec2 * __restrict__ p_output2,
	f64_vec2 * __restrict__ p_denom, f64_vec2 * __restrict__ p_summand1,
	f64_vec2 * __restrict__ p_summand2, f64_vec2 * __restrict__ p_summand3,
	f64 * __restrict__ p_outputz, f64 * __restrict__ p_denomz, f64 * __restrict__ p_summandz1,
	f64 * __restrict__ p_summandz2, f64 * __restrict__ p_summandz3)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;

	f64_vec2 denom = p_denom[index];
	f64_vec2 numer = p_summand1[index] + p_summand2[index] + p_summand3[index];
	f64_vec2 output;
	output.x = numer.x / (denom.x + 1.0);
	output.y = numer.y / (denom.y + 1.0);
//	output.z = numer.z / (denom.z + 1.0);
	p_output2[index] = output;
	f64 denomz = p_denomz[index];
	f64 numerz = p_summandz1[index] + p_summandz2[index] + p_summandz3[index];
	p_outputz[index] = numerz / (denomz + 1.0);
}

__global__ void kernelAdd3things
(f64_vec2 * __restrict__ p_output2,
	f64_vec2 * __restrict__ p_summand1, f64_vec2 * __restrict__ p_summand2,
	f64_vec2 * __restrict__ p_summand3,
	f64 * __restrict__ p_outputz, f64 * __restrict__ p_summandz1,
	f64 * __restrict__ p_summandz2, f64 * __restrict__ p_summandz3)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	p_output2[index] = p_summand1[index] + p_summand2[index] + p_summand3[index];
	p_outputz[index] = p_summandz1[index] + p_summandz2[index] + p_summandz3[index];
}
	
	
__global__ void SubtractVec2(f64_vec2 * __restrict__ p_update,
	f64_vec2 * __restrict__ p_apply)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	f64_vec2 jill = p_update[index];
	f64_vec2 jill2 = p_apply[index];

	if (jill.x > 0.0) {
		if (jill2.x - jill.x < 0.0) {
			// moving the right way
			if (jill2.x*jill.x < 0.0) {
				jill.x = jill2.x; // distance out the other side of 0 = new value
			}
			else {
				jill.x = 0.0;
			};
		}
		else {
			jill.x = (jill2.x - jill.x);
		};
	} else {
		if (jill2.x - jill.x > 0.0) {
			// moving the right way
			if (jill2.x*jill.x < 0.0) {
				jill.x = jill2.x; // distance out the other side of 0 = new value
			} else {
				jill.x = 0.0;
			};
		} else {
			jill.x = (jill2.x - jill.x);
		};
	}
	
	if (jill.y > 0.0) {
		if (jill2.y - jill.y < 0.0) {
			// moving the right way
			if (jill2.y*jill.y < 0.0) {
				jill.y = jill2.y; // distance out the other side of 0 = new value
			}
			else {
				jill.y = 0.0;
			};
		}
		else {
			jill.y = (jill2.y - jill.y);
		};
	}
	else {
		if (jill2.y - jill.y > 0.0) {
			// moving the right way
			if (jill2.y*jill.y < 0.0) {
				jill.y = jill2.y; // distance out the other side of 0 = new value
			}
			else {
				jill.y = 0.0;
			};
		}
		else {
			jill.y = (jill2.y - jill.y);
		};
	}

	p_update[index] = jill;
}

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
	f64_vec3 * __restrict__ p_MAR_elec)
{
	long iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];

	if (info.flag == DOMAIN_TRIANGLE)
	{
		if (iMinor < BEGINNING_OF_CENTRAL)
		{
			LONG3 tricornerindex = p_tricornerindex[iMinor];
			nvals nminor = p_n_minor[iMinor];
			f64 areaminor = p_AreaMinor[iMinor];
			f64 Nhere = areaminor * nminor.n;
			f64 Nnhere =areaminor * nminor.n_n;
			f64 coeff1 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i1];
			f64 coeff2 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i2];
			f64 coeff3 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i3];
		
			// this may be dividing by 0 if the corner is not a domain vertex -- so for ease we stick to domain minors

			f64_vec3 add_i = p_MAR_ion_major[tricornerindex.i1] * coeff1
				+ p_MAR_ion_major[tricornerindex.i2] * coeff2
				+ p_MAR_ion_major[tricornerindex.i3] * coeff3;
			f64_vec3 add_e = p_MAR_elec_major[tricornerindex.i1] * coeff1
				+ p_MAR_elec_major[tricornerindex.i2] * coeff2
				+ p_MAR_elec_major[tricornerindex.i3] * coeff3;

			coeff1 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i1];
			coeff2 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i2];
			coeff3 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i3];

			// DEBUG:
			if (0)//iMinor == CHOSEN) 
				printf("%d Nntotal123 %1.9E %1.9E %1.9E MARneutz %1.8E %1.8E %1.8E \n",CHOSEN,
				p_temp_Nntotalmajor[tricornerindex.i1],
				p_temp_Nntotalmajor[tricornerindex.i2],
				p_temp_Nntotalmajor[tricornerindex.i3],
				p_MAR_neut_major[tricornerindex.i1].z,
				p_MAR_neut_major[tricornerindex.i2].z,
				p_MAR_neut_major[tricornerindex.i3].z
				);



			f64_vec3 add_n = p_MAR_neut_major[tricornerindex.i1] * coeff1
				+ p_MAR_neut_major[tricornerindex.i2] * coeff2
				+ p_MAR_neut_major[tricornerindex.i3] * coeff3;


		//	if (iMinor == 6100) 
		//		printf("p_MAR_neut_major[tricornerindex.i1].z %1.10E coeff1 %1.10E \n"
		//			"p_MAR_neut_major[tricornerindex.i2].z %1.10E coeff2 %1.10E \n"
		//			"p_MAR_neut_major[tricornerindex.i3].z %1.10E coeff3 %1.10E \n"
		//			"add_n.z %1.10E Nnhere %1.10E p_temp_Nntotalmajor[tricornerindex.i1] %1.10E  \n"
		//			"tricornerindex %d %d %d\n"
		//			,
		//			p_MAR_neut_major[tricornerindex.i1].z , coeff1,
		//			p_MAR_neut_major[tricornerindex.i2].z, coeff2,
		//			p_MAR_neut_major[tricornerindex.i3].z, coeff3,
		//			add_n.z, Nnhere,
		//			p_temp_Nntotalmajor[tricornerindex.i1],
		//			tricornerindex.i1, tricornerindex.i2, tricornerindex.i3
		//		);


			if (add_n.z != add_n.z) printf("NaN add_n.z %d\n", iMinor);
			if (add_i.z != add_i.z) printf("NaN add_i.z %d\n", iMinor);
			if (add_e.x != add_e.x) printf("NaN add_e.x %d\n", iMinor);

			p_MAR_neut[iMinor] += add_n;
			p_MAR_ion[iMinor] += add_i;
			p_MAR_elec[iMinor] += add_e;
		} else {
			nvals nminor = p_n_minor[iMinor];
			f64 Nhere = p_AreaMinor[iMinor] * nminor.n;
			long const iVertex = iMinor - BEGINNING_OF_CENTRAL;
			f64 coeff = Nhere / p_temp_Ntotalmajor[iVertex];
			f64_vec3 add_i = p_MAR_ion_major[iVertex] * coeff;
			f64_vec3 add_e = p_MAR_elec_major[iVertex] * coeff;

			if (TEST_IONIZE) {
				printf("iMinor %d coeff %1.8E add_e.z %1.8E MAR %1.9E Nhere %1.9E Ntotal %1.9E \n",
					iMinor, coeff, add_e.z, p_MAR_elec_major[iVertex].z, Nhere, p_temp_Ntotalmajor[iVertex]);
			}
			
			f64 Nnhere = p_AreaMinor[iMinor] * nminor.n_n;
			coeff = Nnhere / p_temp_Nntotalmajor[iMinor - BEGINNING_OF_CENTRAL];
			f64_vec3 add_n = p_MAR_neut_major[iMinor - BEGINNING_OF_CENTRAL] * coeff;

			if (add_n.z != add_n.z) printf("NaN add_n.z %d\n", iMinor);
			if (add_i.z != add_i.z) printf("NaN add_i.z %d\n", iMinor);
			if (add_e.x != add_e.x) printf("NaN add_e.x %d\n", iMinor);

			p_MAR_neut[iMinor] += add_n;
			p_MAR_ion[iMinor] += add_i;
			p_MAR_elec[iMinor] += add_e;
		};		
	};
}

#define RESET_NEUTRAL_N_OUTSIDE  4.6

__global__ void kernelResetNeutralDensityOutsideRadius(
		structural * __restrict__ p_info,
		nvals * __restrict__ p_n_major,
		nvals * __restrict__ p_n_minor
	) {
	long iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];
	if (iMinor > BEGINNING_OF_CENTRAL) {
		if (info.pos.dot(info.pos) > RESET_NEUTRAL_N_OUTSIDE*RESET_NEUTRAL_N_OUTSIDE)
			p_n_major[iMinor - BEGINNING_OF_CENTRAL].n_n = 1.0e18;
	}
	if (info.pos.dot(info.pos) > RESET_NEUTRAL_N_OUTSIDE*RESET_NEUTRAL_N_OUTSIDE)
		p_n_minor[iMinor].n_n = 1.0e18;

}

__global__ void DivideNeTeDifference_by_N(
		NTrates * __restrict__ NT_addition_rates_initial,
		NTrates * __restrict__ NT_addition_rates_final,
		f64 * __restrict__ p_AreaMajor,
		nvals * __restrict__ p_n_major,
		f64 * __restrict__ p_dTbydt)
{
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 diff = NT_addition_rates_final[iVertex].NeTe -
		NT_addition_rates_initial[iVertex].NeTe;
	f64 N = p_n_major[iVertex].n*p_AreaMajor[iVertex];
	
	if (N == 0) {
		p_dTbydt[iVertex] = 0.0;
	} else {
		p_dTbydt[iVertex] = diff / N;
	}
}

__global__ void DivideNiTiDifference_by_N(
	NTrates * __restrict__ NT_addition_rates_initial,
	NTrates * __restrict__ NT_addition_rates_final,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_dTbydt)
{
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 diff = NT_addition_rates_final[iVertex].NiTi -
		NT_addition_rates_initial[iVertex].NiTi;
	f64 N = p_n_major[iVertex].n*p_AreaMajor[iVertex];

	if (N == 0) {
		p_dTbydt[iVertex] = 0.0;
	}
	else {
		p_dTbydt[iVertex] = diff / N;
		if (iVertex == VERTCHOSEN) printf("%d final %1.10E initial %1.10E diff %1.10E N %1.8E dT %1.8E\n",
			VERTCHOSEN, NT_addition_rates_final[iVertex].NiTi,
			NT_addition_rates_initial[iVertex].NiTi, diff, N,
			p_dTbydt[iVertex]);

	}
}

__global__ void DivideNeTe_by_N(
	NTrates * __restrict__ NT_rates,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_dTbydt)
{
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 diff = NT_rates[iVertex].NeTe;
	f64 N = p_n_major[iVertex].n*p_AreaMajor[iVertex];
	if (N == 0) {
		p_dTbydt[iVertex] = 0.0;
	}
	else {
		p_dTbydt[iVertex] = diff / N;
	}
}


__global__ void DivideMAR_get_accel(
	f64_vec3 * __restrict__ pMAR_ion,
	f64_vec3 * __restrict__ pMAR_elec,
	nvals * __restrict__ p_n,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_output_x,
	f64 * __restrict__ p_output_y
) {
	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;

	f64_vec3 MAR_ion = pMAR_ion[iVertex];
	f64_vec3 MAR_elec = pMAR_elec[iVertex];

	f64 N = p_n[iVertex].n*p_AreaMinor[iVertex];
	if (N == 0.0) {
		p_output_x[iVertex] = 0.0;
		p_output_y[iVertex] = 0.0;
	} else {
		p_output_x[iVertex] = (m_i*MAR_ion.x + m_e*MAR_elec.x) /
			((m_i + m_e)*N);
		p_output_y[iVertex] = (m_i*MAR_ion.y + m_e*MAR_elec.y) /
			((m_i + m_e)*N);
	};
}


__global__ void Divide_diff_get_accel(
	v4 * __restrict__ p_vie_f,
	v4 * __restrict__ p_vie_i,
	f64 const h_use,
	f64 * __restrict__ p_output
) {
	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;

	p_output[iVertex] = (p_vie_f[iVertex].vxy.y -
		p_vie_i[iVertex].vxy.y) / h_use;
}


__global__ void DivideMARDifference_get_accel_y(
	f64_vec3 * __restrict__ pMAR_ion,
	f64_vec3 * __restrict__ pMAR_elec,
	f64_vec3 * __restrict__ pMAR_ion_old,
	f64_vec3 * __restrict__ pMAR_elec_old,
	nvals * __restrict__ p_n,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_output_y
) {
	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;

	f64_vec3 MAR_ion = pMAR_ion[iVertex];
	f64_vec3 MAR_elec = pMAR_elec[iVertex];
	f64_vec3 MAR_ion_old = pMAR_ion_old[iVertex];
	f64_vec3 MAR_elec_old = pMAR_elec_old[iVertex];

	f64 N = p_n[iVertex].n*p_AreaMinor[iVertex];
	if (N == 0.0) {
		p_output_y[iVertex] = 0.0;
	} else {
		p_output_y[iVertex] = (m_i*(MAR_ion.y - MAR_ion_old.y) + m_e*(MAR_elec.y - MAR_elec_old.y)) /
			((m_i + m_e)*N);
	};
}
__global__ void kernelCreateWhoAmI_verts(
	structural * __restrict__ p_info_major,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_sz_who_am_I // array of MAXNEIGH shorts for each vertex.
) {
	__shared__ long izNeigh[threadsPerTileMajor][MAXNEIGH]; // 12*long = 6 double < 8 double
	
	short result[MAXNEIGH];
	long izNeigh_neigh[MAXNEIGH];

	long const iVertex = blockIdx.x*blockDim.x + threadIdx.x;
	
	memcpy(izNeigh[threadIdx.x], p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);

	__syncthreads();

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	structural info = p_info_major[iVertex];
	//if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) return;

	// Oh ---- so what do outermost lists contain?
	// They do not know who they are to their neighbours.
	// Don't know why I even put in that skip -- totally unnecessary. Even innermosts can know who they are to their neighbours.

	// BEWARE THEN that info.neigh_len is not apparently accurate.

	if (iVertex == 19180) printf("%d info.neigh_len %d izNeigh[%d-1] %d\n", iVertex, 
		info.neigh_len, info.neigh_len-1, izNeigh[threadIdx.x][info.neigh_len-1]);

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) info.neigh_len--;

	memset(result, 0, sizeof(short)*MAXNEIGH);
	//int neigh_len = info.tri_len;
	//if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) neigh_len--;
	for (int i = 0; i < info.neigh_len; i++)
	{
		
		long iNeigh = izNeigh[threadIdx.x][i];
		long *p;
		if ((iNeigh >= StartMajor) && (iNeigh < EndMajor))
		{
			p = izNeigh[iNeigh - StartMajor];			
			if (iVertex == 19180) printf("shared p. %d %d %d \n",
				p[0], p[1], p[2]);
		} else {
			memcpy(izNeigh_neigh, p_izNeigh_vert + MAXNEIGH*iNeigh,	sizeof(long)*MAXNEIGH);
			p = izNeigh_neigh;

			if (iVertex == 19180) printf("non-shared p. %d %d %d \n",
				p[0], p[1], p[2]);
		};
		short j = 0;
		while ((*p != iVertex) && (j < MAXNEIGH+1)) {
			++p;
			++j;
			if (j >= MAXNEIGH) printf(" j = %d iVertex %d iNeigh %d\n", j, iVertex, iNeigh);
		};
		result[i] = j;
		if (iVertex == 19180) printf("i %d iVertex %d iNeigh %d result j %d",
			i, iVertex, iNeigh, j);
	};

	memcpy(p_sz_who_am_I + MAXNEIGH*iVertex, result, sizeof(short)*MAXNEIGH);

}

__global__ void kernelAddStoredNTFlux(
	structural * __restrict__ p_info_major,
	NTrates * __restrict__ p_additional_array,
	NTrates * __restrict__ p_values_to_augment
) {
	long const iVertex = blockIdx.x*blockDim.x + threadIdx.x;
	structural info = p_info_major[iVertex];
	if (info.flag != DOMAIN_VERTEX) return;

	NTrates additional; // 12*5 doubles = 60 = too many!!
	NTrates total;
	memcpy(&total, p_values_to_augment + iVertex, sizeof(NTrates));
	for (int i = 0; i < MAXNEIGH; i++) {
		memcpy(&additional, &(p_additional_array[MAXNEIGH*iVertex + i]),
			sizeof(NTrates));
		if (iVertex == VERTCHOSEN) printf("%d i= %d : existing %1.10E add %1.10E neut: %1.9E %1.9E\n",
			iVertex, i, total.N, additional.N, total.Nn, additional.Nn);
		total.N += additional.N;
		total.Nn += additional.Nn;
		total.NiTi += additional.NiTi;
		total.NnTn += additional.NnTn;
		total.NeTe += additional.NeTe;
	};
	memcpy(p_values_to_augment + iVertex, &total, sizeof(NTrates));
}

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
) {
	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;

	v4 vie_f = p_vie_final[iVertex];
	v4 vie_i = p_vie_initial[iVertex];
	f64_vec2 accel;
	p_accel_x[iVertex] = (vie_f.vxy.x - vie_i.vxy.x) / h_use;
	p_accel_y[iVertex] = (vie_f.vxy.y - vie_i.vxy.y) / h_use;

	f64_vec2 Grad_Az = pGradAz[iVertex];
	p_vxB_x[iVertex] = (q / (c*(m_i + m_e)))*Grad_Az.x*(vie_f.viz-vie_f.vez);
	p_vxB_y[iVertex] = (q / (c*(m_i + m_e)))*Grad_Az.y*(vie_f.viz-vie_f.vez);

	p_grad_y_Az[iVertex] = Grad_Az.y;

	nvals n_use = p_n_central[iVertex];
	T3 T = p_T_central[iVertex];
	f64_vec3 v_nk = p_v_nk[iVertex];
	f64_vec3 v_nkplus1 = p_v_nkplus1[iVertex];

	if ((n_use.n == 0.0) || (T.Te == 0.0)) {

		p_soak_y[iVertex] = 0.0;

	} else {
				
		f64 cross_section_times_thermal_en, cross_section_times_thermal_in;
		f64 ionneut_thermal, electron_thermal;
		f64 sqrt_Te = sqrt(T.Te);

		ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
		electron_thermal = sqrt_Te * over_sqrt_m_e;
		f64 s_in_MT, s_en_MT, s_en_visc, s_in_visc;

		Estimate_Ion_Neutral_Cross_sections_d(T.Ti*one_over_kB, &s_in_MT, &s_in_visc);
		Estimate_Ion_Neutral_Cross_sections_d(T.Te*one_over_kB, &s_en_MT, &s_en_visc);

		cross_section_times_thermal_en = s_en_MT * electron_thermal;
		cross_section_times_thermal_in = s_in_MT * ionneut_thermal;

		p_soak_y[iVertex] = (1.0 / (m_i + m_e))*
			(m_n*M_i_over_in*cross_section_times_thermal_in*n_use.n_n
				+ m_n * M_e_over_en*cross_section_times_thermal_en*n_use.n_n)*(v_nk.y - vie_f.vxy.y);
//
//		if (TESTACCEL) printf("p_soak_y rate %1.10E v_n.y %1.10E v.y %1.10E v_n diff: %1.10E\n", p_soak_y[iVertex],
//			v_n*/k.y, vie_f.vxy.y, v_nkplus1.y - v_nk.y);

		// Do we get no component of Upsilon that gives us a contribution from another dimension?
		// Not sure on that -- don't think we can push both species same way due to a difference in vz.

	};
}


__global__ void Collect_Ntotal_major(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_temp_Ntotalmajor,
	f64 * __restrict__ p_temp_Nntotalmajor)
{
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	long izTri[MAXNEIGH_d];
	short i;
	//if (iVertex == 3056) {
	//	printf("\niVertex %d info.flag %d info.pos %1.8E %1.8E \n\n", iVertex, info.flag,
	//		info.pos.x, info.pos.y);
	//}

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		nvals ncentral = p_n_minor[iVertex + BEGINNING_OF_CENTRAL];
		f64 areaminorhere = p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL];
		f64 sum_N = ncentral.n*areaminorhere;
		f64 sum_Nn = ncentral.n_n*areaminorhere;
		f64 areaminor;
		nvals nminor;
		for (i = 0; i < info.neigh_len; i++)
		{
			if (TEST_IONIZE)
				printf("iVertex %d i %d izTri[i] %d flag %d sum_N %1.9E\n", 
					iVertex ,i, izTri[i], p_info_minor[izTri[i]].flag, sum_N);
		
			if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE) // see above
			{
				nminor = p_n_minor[izTri[i]];
				areaminor = p_AreaMinor[izTri[i]];
				sum_N += 0.33333333333333*nminor.n*areaminor;
				sum_Nn += 0.33333333333333*nminor.n_n*areaminor;
				if (TEST_IONIZE)
					printf("nminor %1.9E areaminor %1.8E sum_N %1.8E \n", nminor.n, areaminor, sum_N);
			}
			
		};
		if (sum_Nn == 0.0) printf("iVertex %d Nn == 0 %d \n",iVertex, info.flag);
		p_temp_Ntotalmajor[iVertex] = sum_N;
		p_temp_Nntotalmajor[iVertex] = sum_Nn;
	};
}
// We could probably create a big speedup by having a set of blocks that index only the DOMAIN!


__global__ void Collect_Nsum_at_tris(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	LONG3 * __restrict__ p_tricornerindex,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n)
 {
	long iTri = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iTri];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		LONG3 tricornerindex = p_tricornerindex[iTri];
		p_Nsum[iTri] = p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n * p_AreaMajor[tricornerindex.i1]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n * p_AreaMajor[tricornerindex.i2]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n * p_AreaMajor[tricornerindex.i3];

		p_Nsum_n[iTri] = p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n_n * p_AreaMajor[tricornerindex.i1]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n_n * p_AreaMajor[tricornerindex.i2]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n_n * p_AreaMajor[tricornerindex.i3];

//		if (tricornerindex.i1 == VERTCHOSEN) printf("%d corner 1 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n, p_AreaMajor[tricornerindex.i1], p_Nsum[iTri]);
//		if (tricornerindex.i2 == VERTCHOSEN) printf("%d corner 2 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n, p_AreaMajor[tricornerindex.i2], p_Nsum[iTri]);
//		if (tricornerindex.i3 == VERTCHOSEN) printf("%d corner 3 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n, p_AreaMajor[tricornerindex.i3], p_Nsum[iTri]);

	} else {
		p_Nsum[iTri] = 1.0;
		p_Nsum_n[iTri] = 1.0;
	}
}

__global__ void kernelTransmitHeatToVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n,
	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri
) {
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iVertex + BEGINNING_OF_CENTRAL];	
	nvals n_use = p_n_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 N = n_use.n*AreaMajor;
	f64 Nn = n_use.n_n*AreaMajor;

	long izTri[MAXNEIGH_d];
	short i;
	f64 sum_NeTe = 0.0, sum_NiTi = 0.0, sum_NnTn = 0.0;
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		for (i = 0; i < info.neigh_len; i++)
		{
			sum_NiTi += (N / p_Nsum[izTri[i]])*	NT_addition_tri[izTri[i]].NiTi;
			sum_NeTe += (N / p_Nsum[izTri[i]])*	NT_addition_tri[izTri[i]].NeTe;
			sum_NnTn += (Nn / p_Nsum_n[izTri[i]])*	NT_addition_tri[izTri[i]].NnTn;
			// stabilize in the way we apportion heat out of triangle
			
		};
		NT_addition_rates[iVertex].NiTi += sum_NiTi;
		NT_addition_rates[iVertex].NeTe += sum_NeTe;
		NT_addition_rates[iVertex].NnTn += sum_NnTn;
		
	}

	// Idea: pre-store a value which is the sum of N at corners.
}

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
) {
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iVertex + BEGINNING_OF_CENTRAL];
	nvals n_use = p_n_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 N = n_use.n*AreaMajor;
	f64 Nn = n_use.n_n*AreaMajor;

	NTrates NTtri1, NTtri2, NTtri3, NTrat;

	long izTri[MAXNEIGH_d];
	short i;
	f64 sum_NeTe = 0.0, sum_NiTi = 0.0, sum_NnTn = 0.0;
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		for (i = 0; i < info.neigh_len; i++)
		{
			NTtri1 = NT_addition_tri1[izTri[i]];
			NTtri2 = NT_addition_tri2[izTri[i]];
			NTtri3 = NT_addition_tri3[izTri[i]];
			sum_NiTi += (N / p_Nsum[izTri[i]])*	(NTtri1.NiTi+NTtri2.NiTi+NTtri3.NiTi);
			sum_NeTe += (N / p_Nsum[izTri[i]])*	(NTtri1.NeTe+NTtri2.NeTe+NTtri3.NeTe);
			sum_NnTn += (Nn / p_Nsum_n[izTri[i]])*	(NTtri1.NnTn + NTtri2.NnTn + NTtri3.NnTn);
			// stabilize in the way we apportion heat out of triangle
		};
		NTrat = NT_addition_rates[iVertex];
		NTrat.NiTi += sum_NiTi;
		NTrat.NeTe += sum_NeTe;
		NTrat.NnTn += sum_NnTn;
		NT_addition_rates[iVertex] = NTrat;

		if (iVertex == VERTCHOSEN) printf("%d : NiTi %1.10E sum_from tris %1.10E \n",
			iVertex, NTrat.NiTi, sum_NiTi);

	};
	// Idea: pre-store a value which is the sum of N at corners.
}

__global__ void kernelCollect_Up_3x_HeatIntoVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	LONG3 * __restrict__ p_tri_corner_index, // to see which corner we are.
		
	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri1,
	NTrates * __restrict__ NT_addition_tri2,
	NTrates * __restrict__ NT_addition_tri3
	)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iVertex + BEGINNING_OF_CENTRAL];

	NTrates NTtri1, NTtri2, NTtri3, NTrat;

	long izTri[MAXNEIGH_d];
	short i;
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		for (i = 0; i < info.neigh_len; i++)
		{
			// load in the corner data
			LONG3 corners = p_tri_corner_index[izTri[i]];
			int iPlus = 0;
			if (corners.i2 == iVertex) iPlus = 1;
			if (corners.i3 == iVertex) iPlus = 2;
			
			NTtri1 = NT_addition_tri1[izTri[i]*3+iPlus]; // the tri's corner 0 = our vertex
			NTtri2 = NT_addition_tri1[izTri[i]*3+iPlus]; // the tri's corner 0 = our vertex
			NTtri3 = NT_addition_tri1[izTri[i]*3+iPlus]; // the tri's corner 0 = our vertex
			
			// stabilize in the way we apportion heat out of triangle
		};
		NTrat = NT_addition_rates[iVertex];
		NTrat.NiTi += NTtri1.NiTi + NTtri2.NiTi + NTtri3.NiTi;
		NTrat.NeTe += NTtri1.NeTe + NTtri2.NeTe + NTtri3.NeTe;
		NTrat.NnTn += NTtri1.NnTn + NTtri2.NnTn + NTtri3.NnTn;
		NT_addition_rates[iVertex] = NTrat;
	};		
}






// Not optimized: !!
#define FACTOR_HALL (1.0/0.96)
#define FACTOR_PERP (1.2/0.96)
#define DEBUGNANS

__global__ void kernelAdd3(f64_vec3 * __restrict__ p_update, f64_vec3 * __restrict__ p_addition)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec3 vec = p_update[index];
	f64_vec3 vec2 = p_addition[index];
	vec += vec2;
	p_update[index] = vec;
}

__global__ void kernelAddNT(NTrates * __restrict__ p_update, NTrates * __restrict__ p_addition)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	NTrates vec = p_update[index];
	NTrates vec2 = p_addition[index];
	vec.NiTi += vec2.NiTi;
	vec.NeTe += vec2.NeTe;
	vec.NnTn += vec2.NnTn;
	vec.N += vec2.N;
	vec.Nn += vec2.Nn;
	p_update[index] = vec;
}
__global__ void Subtract_V4(
	v4 * __restrict__ p_result,
	v4 * __restrict__ p_a,
	v4 * __restrict__ p_b
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 result;
	v4 a, b;
	a = p_a[index]; 
	b = p_b[index];
	result.vxy = b.vxy - a.vxy;
	result.viz = b.viz - a.viz;
	result.vez = b.vez - a.vez;
	p_result[index] = result;
}

__global__ void kernelSet(
	v4 * __restrict__ p_v4,
	f64_vec3 * __restrict__ p_src,
	int flag
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 v;
	f64_vec3 src = p_src[index];
	v.vxy.x = src.x;
	v.vxy.y = src.y;
	v.viz = 0.0;
	v.vez = 0.0;
	if (flag == SPECIES_ION) v.viz = src.z;
	if (flag == SPECIES_ELEC) v.vez = src.z;
	p_v4[index] = v;
}


__global__ void kernelMultiply_Get_Jacobi_Visc(
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_tens3 * __restrict__ p_Matrix_i,
	f64_tens3 * __restrict__ p_Matrix_e,
	f64_vec3 * __restrict__ p_Jacobi_ion,
	f64_vec3 * __restrict__ p_Jacobi_elec
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64_tens3 Matrix;
		memcpy(&Matrix, p_Matrix_i + iMinor, sizeof(f64_tens3));
		f64_vec3 Jacobi;
		f64_vec3 epsilon;
		memcpy(&epsilon, &(p_eps_xy[iMinor]), sizeof(f64_vec2));
		epsilon.z = p_eps_iz[iMinor];
		Jacobi = Matrix*epsilon;
		p_Jacobi_ion[iMinor] = Jacobi;

		memcpy(&Matrix, p_Matrix_e + iMinor, sizeof(f64_tens3));
		epsilon.z = p_eps_ez[iMinor];
		Jacobi = Matrix*epsilon;
		p_Jacobi_elec[iMinor] = Jacobi;

		// That simple.
	}
	else {
		// Jacobi = 0
		memset(&(p_Jacobi_ion[iMinor]), 0, sizeof(f64_vec3));
		memset(&(p_Jacobi_elec[iMinor]), 0, sizeof(f64_vec3));
	}
}


__global__ void kernelMultiply_Get_Jacobi_NeutralVisc(
	structural * __restrict__ p_info,
	f64_vec3 * __restrict__ p_eps3,
	f64_tens3 * __restrict__ p_Matrix_n,
	f64_vec3 * __restrict__ p_Jacobi ) 
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64_tens3 Matrix;
		memcpy(&Matrix, p_Matrix_n + iMinor, sizeof(f64_tens3));
		f64_vec3 Jacobi;
		f64_vec3 epsilon;
		memcpy(&epsilon, &(p_eps3[iMinor]), sizeof(f64_vec3));
		Jacobi = Matrix*epsilon;
		p_Jacobi[iMinor] = Jacobi;

	} else {
		// Jacobi = 0
		memset(&(p_Jacobi[iMinor]), 0, sizeof(f64_vec3));		
	}
}



__global__ void kernelCreateSeedPartOne(
	f64 const h_use,
	f64 * __restrict__ p_Az,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_AzNext
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext[iMinor] = p_Az[iMinor] + h_use*0.5*p_AAdot_use[iMinor].Azdot;
}

__global__ void kernelCreateSeedPartTwo(
	f64 const h_use,
	f64 * __restrict__ p_Azdot0, 
	f64 * __restrict__ p_gamma, 
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext_update
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext_update[iMinor] += 0.5*h_use* (p_Azdot0[iMinor]
		+ p_gamma[iMinor] * p_LapAz[iMinor]);
}

__global__ void SubtractVector(
	f64 * __restrict__ result,
	f64 * __restrict__ b,
	f64 * __restrict__ a) 
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	result[iMinor] = a[iMinor] - b[iMinor];
}

__global__ void kernelCreateSeedAz(
	f64 const h_use,
	f64 * __restrict__ p_Az_k,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext[iMinor] = p_Az_k[iMinor] + h_use*
		(p_Azdot0[iMinor] + p_gamma[iMinor] * p_LapAz[iMinor]);
	// This seed is suitable if we have no historic data
	// Given 3 points we can make a cubic extrap that should be better.
	// We could then record the proportion of where the solution lay (least squares) between
	// this seed and the cubic seed, see if that has a pattern, and if so, 
	// be recording it (weighted average 50:30:20 of the last 3), use that to provide a better seed still.
	// Detecting the LS optimal proportion depends on writing another bit of code. We could actually
	// just run a regression and record 2 coefficients. If one of them is negative that's not so funny.
	// We could even come up with a 3rd regressor such as Jz or Azdot_k.
}


__global__ void kernelWrapVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	char * __restrict__ p_was_vertex_rotated
) {
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	// I SEE NOW that I am borrowing long const from CPU which is only a backdoor.

	if (info.pos.x*(1.0 - 1.0e-13) > info.pos.y*GRADIENT_X_PER_Y) {
		info.pos = Anticlockwise_d*info.pos;

		v4 vie = p_vie[iVertex + BEGINNING_OF_CENTRAL];
		vie.vxy = Anticlockwise_d*vie.vxy;
		f64_vec3 v_n = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
		v_n = Anticlock_rotate3(v_n);
		p_vie[iVertex + BEGINNING_OF_CENTRAL] = vie;
		p_v_n[iVertex + BEGINNING_OF_CENTRAL] = v_n;
		p_info_minor[iVertex + BEGINNING_OF_CENTRAL] = info;
		
		// Now let's worry about rotating variables in all the triangles that become periodic.
		// Did we do that before in cpp file? Yes.
	
		// We probably need to set a flag on tris modified and launch later.
		// Violating gather-not-scatter. Save a char instead.

		// Also: reassess PBC lists for vertex.
		
		p_was_vertex_rotated[iVertex] = ROTATE_ME_ANTICLOCKWISE;
	};
	if (info.pos.x*(1.0 - 1.0e-13) < -info.pos.y*GRADIENT_X_PER_Y) {

		info.pos = Clockwise_d*info.pos;
		v4 vie = p_vie[iVertex + BEGINNING_OF_CENTRAL];
		vie.vxy = Clockwise_d*vie.vxy;
		f64_vec3 v_n = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
		v_n = Clockwise_rotate3(v_n);

		p_vie[iVertex + BEGINNING_OF_CENTRAL] = vie;
		p_v_n[iVertex + BEGINNING_OF_CENTRAL] = v_n;
		p_info_minor[iVertex + BEGINNING_OF_CENTRAL] = info;
		p_was_vertex_rotated[iVertex] = ROTATE_ME_CLOCKWISE;
	};	

	// Here we could add in some code to add up 1 for each wrapped vertex in the block
	// or just a bool whether any in the block wrapped.

}

__global__ void kernelWrapTriangles(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_corner_index, 
	char * __restrict__ p_was_vertex_rotated,

	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	char * __restrict__ p_triPBClistaffected,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
) {
	long iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info_tri = p_info_minor[iMinor];

	LONG3 cornerindex = p_tri_corner_index[iMinor];

	// Inefficient, no shared mem used:
	char flag0 = p_was_vertex_rotated[cornerindex.i1];
	char flag1 = p_was_vertex_rotated[cornerindex.i2];
	char flag2 = p_was_vertex_rotated[cornerindex.i3];

	if ((flag0 == 0) && (flag1 == 0) && (flag2 == 0))
	{
		// typical case: do nothing
	} else {

		// okay... it is near the PBC edge, because a vertex wrapped.

		// if all vertices are on left or right, it's not a periodic triangle.
		// We need to distinguish what happened: if on one side all the vertices are newly crossed over,
		// then it didn't used to be periodic but now it is. If that is the left side, we need to rotate tri data.
		// If all are now on right, we can rotate tri data to the right. It used to be periodic, guaranteed.

		structural info[3];
		info[0] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i1];
		info[1] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i2];
		info[2] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i3];

		// We are going to set this for the corners whether this tri rotates or not:
		p_triPBClistaffected[cornerindex.i1] = 1;
		p_triPBClistaffected[cornerindex.i2] = 1;
		p_triPBClistaffected[cornerindex.i3] = 1;

		if ((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
		{
			// All now on right => previously some were on left.

			if (TESTTRI) printf("%d All on right\n",iMinor);
			
			p_vie_minor[iMinor].vxy = Clockwise_d*p_vie_minor[iMinor].vxy;
			info_tri.pos = Clockwise_d*info_tri.pos;
			p_v_n_minor[iMinor] = Clockwise_rotate3(p_v_n_minor[iMinor]);
		} else {
			if (((info[0].pos.x > 0.0) || (flag0 == ROTATE_ME_ANTICLOCKWISE))
				&&
				((info[1].pos.x > 0.0) || (flag1 == ROTATE_ME_ANTICLOCKWISE))
				&&
				((info[2].pos.x > 0.0) || (flag2 == ROTATE_ME_ANTICLOCKWISE)))
			{
				// Logic here?
				// Iff all that are on the left are new, then for the first time we are periodic and need to rotate.
				if (TESTTRI) printf("%d Second condition\n", iMinor);

				p_vie_minor[iMinor].vxy = Anticlockwise_d*p_vie_minor[iMinor].vxy;
				info_tri.pos = Anticlockwise_d*info_tri.pos;
				p_v_n_minor[iMinor] = Anticlock_rotate3(p_v_n_minor[iMinor]);
			}
		}
		p_info_minor[iMinor] = info_tri;
		if (TESTTRI) printf("%d info_tri.pos %1.9E %1.9E \n", iMinor, info_tri.pos.x, info_tri.pos.y);

		// Now reassess periodic for corners:
		CHAR4 tri_per_corner_flags;
		memset(&tri_per_corner_flags, 0, sizeof(CHAR4));
		tri_per_corner_flags.flag = info_tri.flag;
		if (((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
			||
			((info[0].pos.x < 0.0) && (info[1].pos.x < 0.0) && (info[2].pos.x < 0.0)))
		{
			// 0 is correct -- triangles only ever rotate corners anticlockwise
			tri_per_corner_flags.per0 = 0;
			tri_per_corner_flags.per1 = 0;
			tri_per_corner_flags.per2 = 0;
			// this was a bug?
		} else {
			if (info[0].pos.x > 0.0) tri_per_corner_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
			if (info[1].pos.x > 0.0) tri_per_corner_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
			if (info[2].pos.x > 0.0) tri_per_corner_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
		}
		
		p_tri_periodic_corner_flags[iMinor] = tri_per_corner_flags;
		if (TESTTRI) printf("%d flags %d %d %d\n",
			iMinor, tri_per_corner_flags.per0, tri_per_corner_flags.per1, tri_per_corner_flags.per2);
	};
}

__global__ void kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_neigh_index,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,
	CHAR4 * __restrict__ p_tri_periodic_neigh_flags,
	char * __restrict__ p_szPBC_triminor,
	char * __restrict__ p_triPBClistaffected
	)
{
	CHAR4 tri_periodic_neigh_flags;

	long const iTri = blockDim.x*blockIdx.x + threadIdx.x;
	LONG3 cornerindex = p_tri_corner_index[iTri];

	// Inefficient, no shared mem used:
	char flag0 = p_triPBClistaffected[cornerindex.i1];
	char flag1 = p_triPBClistaffected[cornerindex.i2];
	char flag2 = p_triPBClistaffected[cornerindex.i3];

	//char flag0 = p_was_vertex_rotated[cornerindex.i1];
	
	//if (iTri == 92250) {
	//	printf("92250 flag012 %d %d %d --\n", flag0, flag1, flag2);
	//}

	if ((flag0 == 0) && (flag1 == 0) && (flag2 == 0))
	{
		// typical case: do nothing
		
	} else {
		// A neighbour tri had a vertex that wrapped.

		structural info = p_info_minor[iTri];
		LONG3 tri_neigh_index = p_tri_neigh_index[iTri];
		memset(&tri_periodic_neigh_flags, 0, sizeof(CHAR4));
		tri_periodic_neigh_flags.flag = info.flag;

		//if (iTri == 92250) {
		//	printf("92250 info.flag %d pos.x %1.8E \n", info.flag, info.pos.x);
		//}

		if (info.pos.x > 0.0) {

			CHAR4 test = p_tri_periodic_corner_flags[tri_neigh_index.i1];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per0 = ROTATE_ME_CLOCKWISE;

			test = p_tri_periodic_corner_flags[tri_neigh_index.i2];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per1 = ROTATE_ME_CLOCKWISE;

			test = p_tri_periodic_corner_flags[tri_neigh_index.i3];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per2 = ROTATE_ME_CLOCKWISE;

		}
		else {
			// if we are NOT periodic but on left, neighs are not rotated rel to us.
			// If we ARE periodic but neigh is not and neigh cent > 0.0 then it is rotated.

			CHAR4 ours = p_tri_periodic_corner_flags[iTri];

			//if (iTri == 92250) {
			//	printf("ours %d %d %d %d\n", ours.per0, ours.per1, ours.per2, ours.flag);
			//}

			if ((ours.per0 != 0) || (ours.per1 != 0) || (ours.per2 != 0)) // ours IS periodic
			{

				structural info0 = p_info_minor[tri_neigh_index.i1];
				structural info1 = p_info_minor[tri_neigh_index.i2];
				structural info2 = p_info_minor[tri_neigh_index.i3];

				if (info0.pos.x > 0.0) tri_periodic_neigh_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
				if (info1.pos.x > 0.0) tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
				if (info2.pos.x > 0.0) tri_periodic_neigh_flags.per2 = ROTATE_ME_ANTICLOCKWISE;

				//	if ((pTri->neighbours[1]->periodic == 0) && (pTri->neighbours[1]->cent.x > 0.0))
					//	tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;			
			

				if (iTri == 92250) {
					printf("tri_periodic_neigh_flags %d %d %d %1.8E %1.8E %1.8E\n",
						tri_periodic_neigh_flags.per0, tri_periodic_neigh_flags.per1,
						tri_periodic_neigh_flags.per2,
						info0.pos.x, info1.pos.x, info2.pos.x);
				}
			};

		};

		p_tri_periodic_neigh_flags[iTri] = tri_periodic_neigh_flags;

		// Set indexneigh periodic list for this tri:
		CHAR4 tri_periodic_corner_flags = p_tri_periodic_corner_flags[iTri];
		char szPBC_triminor[6];
		szPBC_triminor[0] = tri_periodic_corner_flags.per0;
		szPBC_triminor[1] = tri_periodic_neigh_flags.per2;
		szPBC_triminor[2] = tri_periodic_corner_flags.per1;
		szPBC_triminor[3] = tri_periodic_neigh_flags.per0;
		szPBC_triminor[4] = tri_periodic_corner_flags.per2;
		szPBC_triminor[5] = tri_periodic_neigh_flags.per1;
		memcpy(p_szPBC_triminor + 6 * iTri, szPBC_triminor, sizeof(char) * 6);

	}; // was a corner a corner of a tri that had a corner wrapped
}

__global__ void kernelReset_szPBCtri_vert( // would rather it say Update not Reset
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri_vert,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCtri_vert, 
	char * __restrict__ p_szPBCneigh_vert,
	char * __restrict__ p_triPBClistaffected
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	short i;

	structural info = p_info_minor[BEGINNING_OF_CENTRAL + iVertex];
	if (p_triPBClistaffected[iVertex] != 0) {
		char szPBCtri[MAXNEIGH];
		char szPBCneigh[MAXNEIGH];
		long izTri[MAXNEIGH];
		long izNeigh[MAXNEIGH];

		// Now reassess PBC lists for tris 
		memcpy(izTri, p_izTri_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
		structural infotri;
		if (info.pos.x > 0.0) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infotri = p_info_minor[izTri[i]];
				szPBCtri[i] = 0;
				if (infotri.pos.x < 0.0) szPBCtri[i] = ROTATE_ME_CLOCKWISE;

				if (TEST) printf("%d info.pos.x %1.9E RIGHT iTri %d : i %d infotri.pos.x %1.9E szPBCtri[i] %d\n",
					iVertex, info.pos.x, izTri[i], i, infotri.pos.x, (int)szPBCtri[i]);
			};
		} else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infotri = p_info_minor[izTri[i]];
				szPBCtri[i] = 0;
				if (infotri.pos.x > 0.0) szPBCtri[i] = ROTATE_ME_ANTICLOCKWISE;

				if (TEST) printf("%d info.pos.x %1.9E : i %d iTri %d infotri.pos.x %1.9E szPBCtri[i] %d\n",
					iVertex, info.pos.x, i, izTri[i], infotri.pos.x, (int)szPBCtri[i]);
			};
		};
		memcpy(p_szPBCtri_vert + MAXNEIGH*iVertex, szPBCtri, sizeof(char)*MAXNEIGH);

		// If a neighbour wrapped then we share a tri with it that will have given us the
		// PBC tri affected flag. 
		structural infoneigh;

		memcpy(izNeigh, p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);

		if (info.pos.x > 0.0) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x < 0.0) szPBCneigh[i] = ROTATE_ME_CLOCKWISE;
			};
		}
		else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x > 0.0) szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
			};
		};

		memcpy(p_szPBCneigh_vert + MAXNEIGH*iVertex, szPBCneigh, sizeof(char)*MAXNEIGH);

	} else {
		// no update
	}
	
	// Possibly could also argue that if triPBClistaffected == 0 then as it had no wrapping
	// triangle it cannot have a wrapping neighbour. Have to visualise to be sure.
}


