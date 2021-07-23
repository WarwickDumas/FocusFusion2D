
#define TESTHEAT (0)
#define TESTXYDERIVZVISCVERT (0) // (iVertex == VERTCHOSEN) && (DebugFlag))
#define TEST_EPSILON_Y (0)
#define TEST_EPSILON_X (0)//iVertex == VERTCHOSEN)
#define TEST_EPSILON_Y_IMINOR (0)//iMinor == lChosen)
#define TEST_EPSILON_X_MINOR (0)//iMinor == CHOSEN) // iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)

#define HTGPRINT (0) // (iVertex == VERTCHOSEN) && (iSpecies == 1))
#define HTGPRINT2 (0) // (iMinor == CHOSEN) && (iSpecies == 1))

#define TESTVISCCOEFFS (0) //(iMinor == CHOSEN))
#define TESTVISCCOEFF (0)
#define TESTVISCCOEFFX (0)//iMinor == CHOSEN)
#define TESTVISCCOEFFY (0) //(iVertex == VERTCHOSEN) && (iSpecies == 1))

#define TEST_VISC_VERT 1
#define TEST_EPSILON_Z_VERT 1
#define TEST_EPSILON_Z_MINOR 1
// with this 1, or 0, it failed umder TEST_VISC

#define TEST_VISC_VERT_deep (0)
#define TEST_VISC_MINOR 0
#define TEST_VISC_deep (0)//(iMinor == CHOSEN) && (iSpecies == 1))
#define DBG_ROC  (0) // (1)

#define DECAY_IN_VISC_EQNS  1
#define INS_INS_3POINT

#define ALLOWABLE_v_DRIFT_RATE 1.0e9
#define ALLOWABLE_v_DRIFT_RATE_ez 1.0e11

#define SQRTNV

// Note that we have not reconditioned the solver to aim to reduce epsilon in different dimensions differently. It is a bit iffy to have a solver
// that aims at a different criterion than the threshold applied.



//__device__ __forceinline__ v4 Intermediate_v(v4 v1, v4 v2)
//{
//	// In x,y : the angle is bisected and then the modulus is scaled to
//	// the geometric average of the v1 and v2 moduli.
//	// This is more aggressive than letting the square of the modulus be the average square of the moduli.
//	// e.g. |v1| = 1, |v2| = 3, then sqrt(1*1*0.5+3*3*0.5) = sqrt(5) ; sqrt(1*3) = sqrt(3).
//
//	v.vxy = v1.vxy + v2.vxy;
//	if (v.vxy.dot(v.vxy) != 0.0) {
//		f64 product_of_moduli = sqrt((v1.vxy.dot(v1.vxy))*(v2.vxy.dot(v2.vxy)));
//		f64 ratio = sqrt(product_of_moduli / (v.vxy.dot(v.vxy)));
//		v.vxy *= ratio;
//	};
//	if (v1.viz*v2.viz > 0.0) {
//		v.viz = sqrt(v1.viz*v2.viz);
//		if (v1.viz < 0.0) v.viz = -v.viz; // growth of a consistently negative v
//	} else {
//		// they were different signs!
//		v.viz = 0.5*(v1.viz + v2.viz); // Not sure what else to do.
//		// But no, because it needs to be continuous as one end goes through 0.
//
//
//	};
//
//	if (v1.vez*v2.vez > 0.0) {
//		v.vez = sqrt(v1.vez*v2.vez);
//		if (v1.vez < 0.0) v.vez = -v.vez; // growth of a consistently negative v
//	} else {
//
//	};
//}


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

	f64 * __restrict__ p_nu_in_MT_minor,
	f64 * __restrict__ p_nu_en_MT_minor, // 2 separate values? How to use?

	int * __restrict__ p_iSelect,
	int * __restrict__ p_iSelectNeut
)
{
	// Save nu_iHeart, nu_eHeart, nu_nn_visc.

	f64 TeV, sigma_en_MT, sigma_in_MT, sigma_en_visc, sigma_in_visc, sqrt_T, nu_en_visc, lnLambda;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii, nu_nn;
	nvals our_n;
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	structural info = p_info_minor[index];

	int Selected_ie = 0, Selected_neut = 0;

	if (p_iSelect != 0) {
		Selected_ie = p_iSelect[index];
		Selected_neut = p_iSelectNeut[index];
	}
	else {
		Selected_ie = 1;
		Selected_neut = 1;
	};

	if (((info.flag == DOMAIN_VERTEX) ||
		// (info.flag == OUTERMOST) ||
		(info.flag == DOMAIN_TRIANGLE) ||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		) &&
		((Selected_ie > 0) || (Selected_neut > 0))
		)
	{
		// We have not ruled out calculating traffic into outermost vertex cell - so this needs nu calculated in it.
		// (Does it actually try and receive traffic?)

		our_n = p_n_minor[index]; // never used again once we have kappa
		T = p_T_minor[index];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_en_MT, &sigma_en_visc);
		Estimate_Ion_Neutral_Cross_sections_d(T.Ti*one_over_kB, &sigma_in_MT, &sigma_in_visc);
		
		// commented.
		//		sigma_visc *= ArtificialUpliftFactor(our_n.n, our_n.n_n);
		//		sigma_MT *= ArtificialUpliftFactor(our_n.n, our_n.n_n);

		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_en_visc * sqrt_T * over_sqrt_m_e;
		lnLambda = Get_lnLambda_d(our_n.n, T.Te);
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves *
			//max(MINIMUM_NU_EI_DENSITY,our_n.n) *
			// do we want to enhance nu here or not?
			our_n.n*
			lnLambda / (T.Te*sqrt_T);

		//if ((our_n.n < LOW_THRESH_FOR_VISC_CALCS) || (T.Te < 1.1e-14)
		//	|| (info.pos.dot(info.pos) > VISCOSITY_MAX_RADIUS*VISCOSITY_MAX_RADIUS)) {
		if (Selected_ie == 0) {
			p_ita_par_elec_minor[index] = 0.0;
		}
		else {
			p_ita_par_elec_minor[index] =
				//0.73*our_n.n*T.Te / nu_eiBar; // ? Check it's not in eV in formulary
				0.5*our_n.n*T.Te / ((0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc);
			// This from Zhdanov chapter 7. Compare Braginskii.
			// 0.5/(0.3*0.87+0.6) = 0.58 not 0.73
		};
		f64 nu_elec_minor = (0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc;
		if (T.Te <= 0.0) printf("ita calc: Negative Te encountered iMinor = %d /n", index);

		if ((our_n.n < 1.0e11) && (nu_elec_minor < 5.0e13)) {
			// Inflate to reach 5e13 at n=1.0e10
			if (our_n.n < 1.0e10) {
				nu_elec_minor = 5.0e13;
			}
			else {
				// tween towards it
				f64 lambda = (1.0e11 - our_n.n) / (0.9e11);
				nu_elec_minor += lambda*(5.0e13 - nu_elec_minor);
			};
		}
		p_nu_elec_minor[index] = nu_elec_minor;

		//		if ((index == 85822) || (index == 24335))
		//			printf("\n###################################\nindex %d nu_e %1.14E ita %1.14E n %1.14E Te %1.14E \nnu_eiBar %1.14E nu_en_visc %1.14E\n\n",
		//				index, (0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc, p_ita_par_elec_minor[index],
		//				our_n.n, T.Te, nu_eiBar, nu_en_visc);		
		//nu_eHeart:
		//	nu.e = nu_en_visc + 1.87*nu_eiBar;
		// HOWEVER, WE SHOULD PROBABLY EXPECT THAT IN THE END OUR NU_eHEART IS RELEVANT TO OUTPUT HERE
		// TeV = T.Ti*one_over_kB;
		// Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call

		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_in_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor / (sqrt_T*T.Ti);


		if (Selected_ie == 0) {
			p_ita_par_ion_minor[index] = 0.0;
		}
		else {
			p_ita_par_ion_minor[index] = 0.5*our_n.n*T.Ti / (0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar);
		};
		//0.96*our_n.n*T.Ti / nu_ii; // Formulary
		f64 nu_ion_minor = 0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar;

		if ((our_n.n < 1.0e11) && (nu_ion_minor < 1.5e10)) {
			// Inflate to reach 5e13 at n=1.0e10
			if (our_n.n < 1.0e10) {
				nu_ion_minor = 1.5e10;
			}
			else {
				// tween towards it
				f64 lambda = (1.0e11 - our_n.n) / (0.9e11);
				nu_ion_minor += lambda*(1.5e10 - nu_elec_minor);
			};
		}
		p_nu_ion_minor[index] = nu_ion_minor;



		// Here copy-paste code from PopOhms:

		// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
		f64 sqrt_Te, ionneut_thermal, electron_thermal,
			cross_section_times_thermal_en, cross_section_times_thermal_in;
		sqrt_Te = sqrt(T.Te);
		ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
		electron_thermal = sqrt_Te * over_sqrt_m_e;
		
		//nu_ne_MT = s_en_MT * electron_thermal * n_use.n; // have to multiply by n_e for nu_ne_MT
		//nu_ni_MT = s_in_MT * ionneut_thermal * n_use.n;
		//nu_in_MT = s_in_MT * ionneut_thermal * n_use.n_n;
		//nu_en_MT = s_en_MT * electron_thermal * n_use.n_n;

		cross_section_times_thermal_en = sigma_en_MT * electron_thermal;
		cross_section_times_thermal_in = sigma_in_MT * ionneut_thermal;

		// ARTIFICIAL CHANGE TO STOP IONS SMEARING AWAY OFF OF NEUTRAL BACKGROUND:
		if (our_n.n_n > ARTIFICIAL_RELATIVE_THRESH *our_n.n) {
			cross_section_times_thermal_en *= our_n.n_n / (ARTIFICIAL_RELATIVE_THRESH*our_n.n);
			cross_section_times_thermal_in *= our_n.n_n / (ARTIFICIAL_RELATIVE_THRESH*our_n.n);
		};
		cross_section_times_thermal_en *= ArtificialUpliftFactor_MT(our_n.n, our_n.n_n);
		cross_section_times_thermal_in *= ArtificialUpliftFactor_MT(our_n.n, our_n.n_n); // returns factor 1.0 if n+nn > 1.0e14.

		// nu_en_MT = cross_section_times_thermal_en*n_use.n_n 
		// applies in accel with factor * m_n/(m_e+m_n)
		// nu_in_MT = cross_section_times_thermal_in*n_use.n_n
		// applies in accel with factor * m_n/(m_i+m_n)
		
		p_nu_in_MT_minor[index] = cross_section_times_thermal_in*our_n.n_n;
		p_nu_en_MT_minor[index] = cross_section_times_thermal_en*our_n.n_n;

		// Use in for xy & viz , and use en for vez.
		
		//-----=---------=-==-----------------==========-------------------==-----==--------------=----

		// I think we just need to knock out points where ita_par_ion_minor ends up low
		// rather than specifically low density.
		// Trouble is that will just give us nothing out of a shock front.
		// Maybe need to be more careful -- go 2 passes

		//nu_nn_visc:
		nu_nn = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
		f64 nu_n = 0.3*nu_nn + 0.4*nu_ni_visc;

		// This is set at 1e10
		//if ((our_n.n_n < LOW_THRESH_FOR_VISC_CALCS)
		//	|| (info.pos.dot(info.pos) > VISCOSITY_MAX_RADIUS*VISCOSITY_MAX_RADIUS)) {
		if (Selected_neut == 0) {
			p_ita_neutral_minor[index] = 0.0;
		}
		else {
			f64 ita = our_n.n_n*T.Tn / nu_n;
			// NEW THRESHOLD HERE TO STOP NEUTRAL VISC:
			if (T.Tn < 1.1e-14) ita = 0.0;
			p_ita_neutral_minor[index] = ita;
		};
		p_nu_nn_visc[index] = nu_n;
	}
	else {
		p_ita_par_elec_minor[index] = 0.0;
		p_nu_elec_minor[index] = 0.0;
		p_ita_par_ion_minor[index] = 0.0;
		p_nu_ion_minor[index] = 0.0;
		p_ita_neutral_minor[index] = 0.0;
		p_nu_nn_visc[index] = 0.0;
		// For CROSSING_CATH this is fine.
	}
}

// Historical record
// We have to reform heat cond so that we use min n but a good T throughout kappa
// We do try to use same n in numer and denom, including in ln Lambda.
// We have too many spills in the routine so we should make a separate ionisation routine
// and try to do species separately.



__global__ void kernelSplit_vec2(
	f64_vec2 * __restrict__ p_vec2,
	f64 * __restrict__ p_a,
	f64 * __restrict__ p_b)
{

	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 vec2 = p_vec2[iMinor];
	p_a[iMinor] = vec2.x;
	p_b[iMinor] = vec2.y;
}

__global__ void kernelSquare
(f64 * __restrict__ p__x)
{

	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64 x = p__x[iMinor];
	p__x[iMinor] = x*x;
}

__global__ void kernelSquare2
(f64_vec2 * __restrict__ p__x)
{

	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 x = p__x[iMinor];
	x.x = x.x*x.x;
	x.y = x.y*x.y;
	p__x[iMinor] = x;
}
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
	)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iMinor];
	f64_vec2 epsxy;
	f64 epsz = 0.0;
	memset(&epsxy, 0, sizeof(f64_vec2));
	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n_n;

		f64_vec2 regrxy = p_regr_xy[iMinor];
		f64 regrz = p_regr_z[iMinor];
		f64_vec3 ROCMAR = p_ROCMAR[iMinor];
		f64 fac;
		if (N == 0) {
			fac = 0.0; // ?
		} else {
			fac = hsub / N;
		};
		// epsilon = v_n - v_n_k - hsub*(MAR_neut) / N;
		// Rate of change:
		epsxy.x = regrxy.x - fac*ROCMAR.x;
		epsxy.y = regrxy.y - fac*ROCMAR.y;
		epsz = regrz - fac*ROCMAR.z;

		if (0) { //((iMinor > BEGINNING_OF_CENTRAL) && ((iMinor - BEGINNING_OF_CENTRAL) % 8000 == 0)) {
			printf("iVertex %d deps/dbeta %1.9E regrz %1.9E fac %1.9E ROCMAR.z %1.9E \n",
				iMinor - BEGINNING_OF_CENTRAL, epsz, regrz, fac, ROCMAR.z);
		}

	} else {
		// eps = 0
	};

	p_eps_xy[iMinor] = epsxy;
	p_eps_z[iMinor] = epsz;
}
__device__ __forceinline__ f64_vec2 ReconstructEdgeNormal(
	f64_vec2 prevpos, f64_vec2 ownpos, f64_vec2 nextpos, f64_vec2 opppos)
{
	f64_vec2 endpt0 = THIRD*(prevpos + ownpos + opppos);
	f64_vec2 endpt1 = THIRD*(nextpos + ownpos + opppos);
	
	// One of prev and next may be below ins.
	// If so, we want to restrict to the line between ownpos and opppos.
	/*

	// 13/06/21: Trouble with that is that for mag visc, we are not getting enough diffusion of x-momentum sideways. We need the sideways diffusion
	// to outweigh the radial antidiffusion.

	if (TestDomainPos(prevpos) == false) {

		// move endpt0 towards endpt1 until we reach the line from ownpos to opppos?

		// (x,y) = x0 + a (x1-x0)
		// &
		// (x - xself)*(y_opp-y_self) + (y-yself)*(x_self-x_opp) = 0

		// (x,y-xself,yself) is orthogonal to (-y,x) of opp-self.

		// (x0x + a (x1x-x0x) - xself)(y_opp-y_self) = (x0y + a (x1y-x0y) - yself)(x_opp-x_self)

		// a((x1x-x0x)(y_opp-y_self) - (x1y-x0y)(x_opp-x_self)) = 
		//    - (x0x-xself) (y_opp-y_self) + (x0y-yself) (x_opp-x_self)

		f64 alpha = ((endpt0.y - ownpos.y)*(opppos.x - ownpos.x) - (endpt0.x - ownpos.x)*(opppos.y - ownpos.y)) /
			((endpt1.x - endpt0.x)*(opppos.y - ownpos.y) - (endpt1.y - endpt0.y)*(opppos.x - ownpos.x));

		endpt0 += alpha*(endpt1 - endpt0);

	}
	else {
		if (TestDomainPos(nextpos) == false) {

			// swap 0,1 :
			f64 alpha = ((endpt1.y - ownpos.y)*(opppos.x - ownpos.x) - (endpt1.x - ownpos.x)*(opppos.y - ownpos.y)) /
				((endpt0.x - endpt1.x)*(opppos.y - ownpos.y) - (endpt0.y - endpt1.y)*(opppos.x - ownpos.x));

			endpt1 += alpha*(endpt0 - endpt1);
		};
	};

	*/

	f64 mod1 = endpt1.modulus();
	f64 mod0 = endpt0.modulus();

	if (mod0 < DEVICE_RADIUS_INSULATOR_OUTER)
	{
		// add the multiple of (endpt1-endpt0) that will get us to radius 3.44 -- ish will do
		if (mod1 == mod0) // error
		{
			endpt0 *= 3.44 / mod0;
			printf("\n\n3rror\n\n");
		}
		else {
			endpt0 += (endpt1 - endpt0)*(3.44 - mod0) / (mod1 - mod0);
		};
	};
	if (mod1 < DEVICE_RADIUS_INSULATOR_OUTER)
	{
		if (mod1 == mod0)
		{
			endpt1 *= 3.44 / mod1;
			printf("\n\n3rror\n\n");
		}
		else {
			endpt1 += (endpt0 - endpt1)*(3.44 - mod1) / (mod0 - mod1);
		};
	};
	f64_vec2 edge_normal;
	edge_normal.x = endpt1.y - endpt0.y;
	edge_normal.y = endpt0.x - endpt1.x;
	return edge_normal;
};
__device__ __forceinline__ f64_vec2 ReconstructEdgeNormalDebug(
	f64_vec2 prevpos, f64_vec2 ownpos, f64_vec2 nextpos, f64_vec2 opppos)
{
	f64_vec2 endpt0 = THIRD*(prevpos + ownpos + opppos);
	f64_vec2 endpt1 = THIRD*(nextpos + ownpos + opppos);


	// One of prev and next may be below ins.
	// If so, we want to restrict to the line between ownpos and opppos.
	/*

	// 13/06/21: Trouble with that is that for mag visc, we are not getting enough diffusion of x-momentum sideways. We need the sideways diffusion
	// to outweigh the radial antidiffusion.

	if (TestDomainPos(prevpos) == false) {

		// move endpt0 towards endpt1 until we reach the line from ownpos to opppos?

		// (x,y) = x0 + a (x1-x0)
		// &
		// (x - xself)*(y_opp-y_self) + (y-yself)*(x_self-x_opp) = 0

		// (x,y-xself,yself) is orthogonal to (-y,x) of opp-self.

		// (x0x + a (x1x-x0x) - xself)(y_opp-y_self) = (x0y + a (x1y-x0y) - yself)(x_opp-x_self)

		// a((x1x-x0x)(y_opp-y_self) - (x1y-x0y)(x_opp-x_self)) = 
		//    - (x0x-xself) (y_opp-y_self) + (x0y-yself) (x_opp-x_self)
		// Had it wrong before.

		f64 alpha = ((endpt0.y-ownpos.y)*(opppos.x - ownpos.x) - (endpt0.x-ownpos.x)*(opppos.y - ownpos.y)) /
			((endpt1.x - endpt0.x)*(opppos.y - ownpos.y) - (endpt1.y - endpt0.y)*(opppos.x - ownpos.x));

		endpt0 += alpha*(endpt1 - endpt0);

	}
	else {
		if (TestDomainPos(nextpos) == false) {

			// swap 0,1 :
			f64 alpha = ((endpt1.y-ownpos.y)*(opppos.x - ownpos.x) - (endpt1.x-ownpos.x)*(opppos.y - ownpos.y)) /
				((endpt0.x - endpt1.x)*(opppos.y - ownpos.y) - (endpt0.y - endpt1.y)*(opppos.x - ownpos.x));

			printf("endpt0 %1.10E %1.10E endpt1 %1.10E %1.10E alpha %1.10E \n",
				endpt0.x, endpt0.y, endpt1.x, endpt1.y, alpha);

			endpt1 += alpha*(endpt0 - endpt1);
		};
	};
	*/

	f64 mod1 = endpt1.modulus();
	f64 mod0 = endpt0.modulus();
	printf("mod0 %1.10E mod1 %1.10E\n", mod0, mod1);

	if (mod0 < DEVICE_RADIUS_INSULATOR_OUTER)
	{
		// add the multiple of (endpt1-endpt0) that will get us to radius 3.44 -- ish will do
		if (mod1 == mod0) // error
		{
			endpt0 *= 3.44 / mod0;
			printf("\n\n3rror\n\n");
		}
		else {
			endpt0 += (endpt1 - endpt0)*(3.44 - mod0) / (mod1 - mod0);
			printf("endpt0 += (endpt1 - endpt0)*(3.44 - mod0) / (mod1 - mod0);\n");
		};
	};
	if (mod1 < DEVICE_RADIUS_INSULATOR_OUTER)
	{
		if (mod1 == mod0)
		{
			endpt1 *= 3.44 / mod1;
			printf("\n\n3rror\n\n");
		}
		else {
			endpt1 += (endpt0 - endpt1)*(3.44 - mod1) / (mod0 - mod1);
			printf("endpt1 += (endpt0 - endpt1)*(3.44 - mod1) / (mod0 - mod1);\n");
		};
	};
	f64_vec2 edge_normal;
	edge_normal.x = endpt1.y - endpt0.y;
	edge_normal.y = endpt0.x - endpt1.x;
	return edge_normal;
};

__device__ __forceinline__ f64_vec2 GetGradient_3Point(
	f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 opppos, 
	f64 prev_v, f64 our_v, f64 opp_v)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:

	//f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	//f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	//f64 out_modulus = vec_out.modulus();

	////Apparently this sometimes comes out as zero.

	////That does not seem right.
	//if (out_modulus == 0.0) printf("ourpos %1.10E %1.10E ppn %1.10E opppos %1.10E %1.10E nextpos %1.10E %1.10E vec-out=0\n",
	//	ourpos.x, ourpos.y, proportion, opppos.x, opppos.y, nextpos.x, nextpos.y);


	//f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	//f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;
	//// deriv in the direction towards nextpos.

	f64 proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);

	f64_vec2 vec_out = (prevpos - (ourpos + proportion*(opppos - ourpos)));
	f64 lengthout = vec_out.modulus();
	
	f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 transversederiv = (prev_v - intermediatevalue) / lengthout;

	// Get longi derivative first :
	f64_vec2 gradv;
	gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	gradv.y = gradv.x * (opppos.y - ourpos.y);
	gradv.x *= (opppos.x - ourpos.x);

	transversederiv /= lengthout; // no longer deriv -- done this way to minimize divides.
	gradv.x += transversederiv*vec_out.x;
	gradv.y += transversederiv*vec_out.y;

	return gradv;
}


__device__ __forceinline__ f64_vec2 GetGradient(
	f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, 
	f64_vec2 opppos, f64 prev_v, f64 our_v, f64 next_v, f64 opp_v)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:
	
	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();
	
	//Apparently this sometimes comes out as zero.

	//That does not seem right.
	if (out_modulus == 0.0) printf("ourpos %1.10E %1.10E ppn %1.10E opppos %1.10E %1.10E nextpos %1.10E %1.10E vec-out=0\n",
		ourpos.x, ourpos.y, proportion, opppos.x, opppos.y, nextpos.x, nextpos.y);


	f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;	
	// deriv in the direction towards nextpos.

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();
	
	intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 transversederivprev = -(prev_v - intermediatevalue) / lengthout;

	// Special formula now found:
	
	// . First need to rescale the two things to have a sum of squares that is 400.
	// . Now take sinh of the average of asinh's
	// . Now scale back again.
	// . This way the shape of the curve is not units-dependent.
	
	f64 transversederiv;
	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);
	if (root == 0.0) { transversederiv = 0.0; } 
	else {
		f64 overroot = 1.0 / root;
		transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
			+ asinh(20.0*overroot*transversederivnext)));

		if (transversederiv != transversederiv) printf("xversederiv NaN y1 %1.8E y2 %1.8E overroot %1.8E "
			"asinh %1.7E %1.8E sinh %1.8E root %1.8E \n",
			transversederivprev, transversederivnext, overroot, asinh(20.0*overroot*transversederivprev),
			asinh(20.0*overroot*transversederivnext), sinh(0.5*(asinh(20.0*overroot*transversederivprev)
				+ asinh(20.0*overroot*transversederivnext))), root);
	};
	
	// Get longi derivative first :
	f64_vec2 gradv;
	gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	gradv.y = gradv.x * (opppos.y - ourpos.y);
	gradv.x *= (opppos.x - ourpos.x);

	transversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
	gradv.x += transversederiv*vec_out.x;
	gradv.y += transversederiv*vec_out.y;

	// DEBUG:
	if (out_modulus == 0) printf("Out_modulus == 0\n");
	if (gradv.x != gradv.x) printf("opppos %1.8E %1.8E ourpos %1.8E %1.8E vec_out.x %1.8E \n",
		opppos.x, opppos.y, ourpos.x, ourpos.y, vec_out.x);

	return gradv;
}

__device__ __forceinline__ f64_vec2 GetGradientDebug(
	f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos,
	f64_vec2 opppos, f64 prev_v, f64 our_v, f64 next_v, f64 opp_v)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:

	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	//Apparently this sometimes comes out as zero.

	//That does not seem right.
	if (out_modulus == 0.0) printf("ourpos %1.10E %1.10E ppn %1.10E opppos %1.10E %1.10E nextpos %1.10E %1.10E vec-out=0\n",
		ourpos.x, ourpos.y, proportion, opppos.x, opppos.y, nextpos.x, nextpos.y);


	f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;
	// deriv in the direction towards nextpos.

	printf("next_v %1.12E intervalue %1.12E transversederivnext %1.12E\n", 
		next_v, 
		intermediatevalue,
		transversederivnext);


	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();

	intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 transversederivprev = -(prev_v - intermediatevalue) / lengthout;

	printf("prev_v %1.12E intervalue %1.12E transversederivprev %1.12E\n", 
		prev_v,
		intermediatevalue, 
		transversederivprev);

	// Special formula now found:

	// . First need to rescale the two things to have a sum of squares that is 400.
	// . Now take sinh of the average of asinh's
	// . Now scale back again.
	// . This way the shape of the curve is not units-dependent.

	f64 transversederiv;
	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);
	if (root == 0.0) { transversederiv = 0.0; 
		printf("root == 0.0");
	}
	else {
		f64 overroot = 1.0 / root;
		transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
			+ asinh(20.0*overroot*transversederivnext)));

		printf("root %1.12E overroot %1.12E asinh prev next %1.12E %1.12E xversederiv %1.12E\n",
			root,overroot,
			asinh(20.0*overroot*transversederivprev),
			asinh(20.0*overroot*transversederivnext),
			transversederiv
		);

		if (transversederiv != transversederiv) printf("xversederiv NaN y1 %1.8E y2 %1.8E overroot %1.8E "
			"asinh %1.7E %1.8E sinh %1.8E root %1.8E \n",
			transversederivprev, transversederivnext, overroot, asinh(20.0*overroot*transversederivprev),
			asinh(20.0*overroot*transversederivnext), sinh(0.5*(asinh(20.0*overroot*transversederivprev)
				+ asinh(20.0*overroot*transversederivnext))), root);
	};

	// Get longi derivative first :
	f64_vec2 gradv;
	gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	gradv.y = gradv.x * (opppos.y - ourpos.y);
	gradv.x *= (opppos.x - ourpos.x);

	printf("longi deriv %1.12E %1.12E \n", gradv.x, gradv.y);

	transversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
	gradv.x += transversederiv*vec_out.x;
	gradv.y += transversederiv*vec_out.y;
	
	// DEBUG:
	if (out_modulus == 0) printf("Out_modulus == 0\n");
	if (gradv.x != gradv.x) printf("opppos %1.8E %1.8E ourpos %1.8E %1.8E vec_out.x %1.8E \n",
		opppos.x, opppos.y, ourpos.x, ourpos.y, vec_out.x);
	
	printf("total deriv %1.12E %1.12E vec_out %1.9E %1.9E out_modulus %1.9E\n", gradv.x, gradv.y, vec_out.x, vec_out.y,
		out_modulus);

	return gradv;
}

//__device__ __forceinline__ f64_vec2 GetSelfCoeffForGradient(
//	f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos) 
//	// .x is coeff for dvx/dx on vx , .y is coeff for dvx/dy on vx
//{
//	// I think we should neglect the effect of self on transverse deriv.
//	// Both for simplicity, efficiency and because it should generally end up small. 
//
//	// Get longi derivative first :
//	f64_vec2 gradv_coeff_vq;
//	gradv_coeff_vq.x = (-1.0) / (opppos - ourpos).dot(opppos - ourpos);
//	gradv_coeff_vq.y = gradv_coeff_vq.x * (opppos.y - ourpos.y);
//	gradv_coeff_vq.x *= (opppos.x - ourpos.x);
//
//	// here
//	// d grad_x v_by_dself = (-1)*(opppos.x - ourpos.x)/(opppos - ourpos).dot(opppos - ourpos);
//	// d grad_y v_by_dself = (-1)*(opppos.y - ourpos.y)/(opppos - ourpos).dot(opppos - ourpos);
//
//	return gradv_coeff_vq;
//}
   // Is it dependent on more than just position?

__global__ void Test_Asinh()
{
	f64 test = asinh(0.0);
	printf("asinh(0.0) = %1.10E \n", test);
}

__device__ __forceinline__ f64_vec2
GetSelfEffectOnGradient_3Point(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 opp_v
)
{
	// For code brevity, could just call the d/dbeta function and tell it a[self] = 1, a[neighs] = 0.
	
	//f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	//f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	//f64 out_modulus = vec_out.modulus();

	//f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	//f64 ROCintermediatevalue = 1.0 - proportion;
	//f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;
	//f64 ROCnext = -ROCintermediatevalue / out_modulus;
	// deriv in the direction towards nextpos.

	f64 proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (prevpos - (ourpos + proportion*(opppos - ourpos)));
	f64 lengthout = vec_out.modulus();

	f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 ROCintermediatevalue = 1.0 - proportion;
	f64 transversederiv = (prev_v - intermediatevalue) / lengthout;
	f64 ROCtransversederiv = -ROCintermediatevalue / lengthout;

	// Get longi derivative first :
	f64_vec2 ROCgradv;

	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.x = -1.0 / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.y = ROCgradv.x * (opppos.y - ourpos.y);
	ROCgradv.x *= (opppos.x - ourpos.x);

	ROCtransversederiv /= lengthout; // no longer deriv -- done this way to minimize divides.
	ROCgradv.x += ROCtransversederiv*vec_out.x;
	ROCgradv.y += ROCtransversederiv*vec_out.y;

	return ROCgradv;
}


__device__ __forceinline__ f64_vec2
GetSelfEffectOnGradient(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
	)
{
	// For code brevity, could just call the d/dbeta function and tell it a[self] = 1, a[neighs] = 0.


	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	 f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 ROCintermediatevalue = 1.0 - proportion;
	 f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;
	f64 ROCnext = -ROCintermediatevalue / out_modulus;
	// deriv in the direction towards nextpos.

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();

	 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	ROCintermediatevalue = 1.0 - proportion;
	 f64 transversederivprev = -(prev_v - intermediatevalue) / lengthout;
	f64 ROCprev = ROCintermediatevalue / lengthout;
	
	f64 transversederiv;
	
	// Because this is v-dependent, we have to recalculate every time we want to generate Jacobi.
	// Oh jolly dear.

	// No other choice really --- can't do by halves.

	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);

	f64 ROCtransversederiv;

	if (root == 0.0) {

		f64 vlocal = sqrt(0.25*(our_v*our_v + prev_v*prev_v + next_v*next_v + opp_v*opp_v));
		f64 smallstep = 2.0e-10 + 2.0e-10*vlocal;
		
		//f64 smallstep = 1.0e-6; // 1e-7 above FP unless v > 1.0e7 .. watch out
		bool bCont = false;
		int iIterate = 0;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			root = smallstep*sqrt(ROCprev*ROCprev + ROCnext*ROCnext);

			// confused why it's this? Because root == 0 means that
			// transversederivprev == transversederivnext == 0

			if (root == 0.0) {
				ROCtransversederiv = 0.0;
			} else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*ROCprev*smallstep)
						+ asinh(20.0*overroot*ROCnext*smallstep)));
				f64 d_by_dbeta_transversederiv2 = transversederiv_stepped / smallstep;

				smallstep *= 0.5;
				root *= 0.5;
				overroot *= 2.0;
				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*ROCprev*smallstep)
						+ asinh(20.0*overroot*ROCnext*smallstep)));
				f64 d_by_dbeta_transversederiv1 = transversederiv_stepped / smallstep;

				if ((d_by_dbeta_transversederiv2 > 1.01*d_by_dbeta_transversederiv1) ||
					(d_by_dbeta_transversederiv2 < 0.99*d_by_dbeta_transversederiv1)) {
					// go again
					bCont = true;
				} else {
					ROCtransversederiv = d_by_dbeta_transversederiv1
						+ d_by_dbeta_transversederiv1 - d_by_dbeta_transversederiv2;
					// let's assume derivative changes linearly.
					// Q2 = Q(step*0.5) = 0.25*step*step*f'' + 0.5*step*f'
					// Q1 = Q(step*0.25) = 0.25*0.25*step*step*f'' + 0.25*step*f'
					// Q(0) = ?
					bCont = false;
				};
			};
		} while ((bCont) && (iIterate < 100));
		if (iIterate == 100) printf("\niIterate==100 dbydbetatransversederiv = %1.9E \n\n", ROCtransversederiv);
		
	} else {

		f64 overroot = 1.0 / root;
		f64 y1 = transversederivprev;
		f64 y2 = transversederivnext;

		f64 avgasinh = 0.5*(asinh(20.0*y1*overroot) + asinh(20.0*y2*overroot));
		f64 ratio = (y1*ROCprev + y2*ROCnext) / (y1*y1 + y2*y2);

		ROCtransversederiv = (y1*ROCprev + y2*ROCnext)*0.05*overroot*sinh(avgasinh)
			+ root*0.05*cosh(avgasinh)*10.0*
			((ROCprev - y1*ratio) / sqrt(401.0*y1*y1 + y2*y2)
				+ (ROCnext - y2*ratio) / sqrt(y1*y1 + 401.0*y2*y2));


	}

	// Get longi derivative first :
	f64_vec2 ROCgradv;
	
	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.x = -1.0 / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.y = ROCgradv.x * (opppos.y - ourpos.y);
	ROCgradv.x *= (opppos.x - ourpos.x);

	ROCtransversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
	ROCgradv.x += ROCtransversederiv*vec_out.x;
	ROCgradv.y += ROCtransversederiv*vec_out.y;

	// DEBUG:
	if (out_modulus == 0) printf("Out_modulus == 0\n");
	
	return ROCgradv;
}

__device__ __forceinline__ f64_vec2
GetSelfEffectOnGradientLongitudinal(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
)
{
	// For code brevity, could just call the d/dbeta function and tell it a[self] = 1, a[neighs] = 0.

	// Get longi derivative first :
	f64_vec2 ROCgradv;

	//gradv.x = (opp_vx - our_vx)*(opppos-ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.x = -1.0 / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.y = ROCgradv.x * (opppos.y - ourpos.y);
	ROCgradv.x *= (opppos.x - ourpos.x);

	return ROCgradv;
}

__device__ __forceinline__ f64_vec2
GetSelfEffectOnGradientDebug(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
)
{
	// For code brevity, could just call the d/dbeta function and tell it a[self] = 1, a[neighs] = 0.


	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	f64 intermediatevalue = our_v + (opp_v - our_v)*proportion;
	f64 ROCintermediatevalue = 1.0 - proportion;
	f64 transversederivnext = (next_v - intermediatevalue) / out_modulus;
	f64 ROCnext = -ROCintermediatevalue / out_modulus;
	// deriv in the direction towards nextpos.

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();

	intermediatevalue = our_v + (opp_v - our_v)*proportion;
	ROCintermediatevalue = 1.0 - proportion;
	f64 transversederivprev = -(prev_v - intermediatevalue) / lengthout;
	f64 ROCprev = ROCintermediatevalue / lengthout;

	f64 transversederiv;

	// Because this is v-dependent, we have to recalculate every time we want to generate Jacobi.
	// Oh jolly dear.

	// No other choice really --- can't do by halves.

	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);

	f64 ROCtransversederiv;

	printf("xverse prev next %1.10E %1.10E root %1.10E ROCprev %1.10E ROCnext %1.10E",
		transversederivprev, transversederivnext, root, ROCprev, ROCnext);

	if (root == 0.0) {

		f64 vlocal = sqrt(0.25*(our_v*our_v + prev_v*prev_v + next_v*next_v + opp_v*opp_v));
		f64 smallstep = 2.0e-10 + 2.0e-10*vlocal;

		//f64 smallstep = 1.0e-6; // 1e-7 above FP unless v > 1.0e7 .. watch out
		bool bCont = false;
		int iIterate = 0;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			root = smallstep*sqrt(ROCprev*ROCprev + ROCnext*ROCnext);

			// confused why it's this? Because root == 0 means that
			// transversederivprev == transversederivnext == 0

			if (root == 0.0) {
				ROCtransversederiv = 0.0;
			}
			else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*ROCprev*smallstep)
						+ asinh(20.0*overroot*ROCnext*smallstep)));
				f64 d_by_dbeta_transversederiv2 = transversederiv_stepped / smallstep;

				smallstep *= 0.5;
				root *= 0.5;
				overroot *= 2.0;
				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*ROCprev*smallstep)
						+ asinh(20.0*overroot*ROCnext*smallstep)));
				f64 d_by_dbeta_transversederiv1 = transversederiv_stepped / smallstep;

				printf("estimated d/dbeta xverse 1 2 : %1.10E %1.10E \n", d_by_dbeta_transversederiv1, d_by_dbeta_transversederiv2);
				printf("looks to me this routine isn't up to date with other similar routines??\n");
				printf("smallstep %1.10E root %1.9E \n", smallstep, root);

				if ((d_by_dbeta_transversederiv2 > 1.01*d_by_dbeta_transversederiv1) ||
					(d_by_dbeta_transversederiv2 < 0.99*d_by_dbeta_transversederiv1)) {
					// go again
					bCont = true;
				}
				else {
					ROCtransversederiv = d_by_dbeta_transversederiv1
						+ d_by_dbeta_transversederiv1 - d_by_dbeta_transversederiv2;
					// let's assume derivative changes linearly.
					// Q2 = Q(step*0.5) = 0.25*step*step*f'' + 0.5*step*f'
					// Q1 = Q(step*0.25) = 0.25*0.25*step*step*f'' + 0.25*step*f'
					// Q(0) = ?
					bCont = false;
				};
			};
		} while ((bCont) && (iIterate < 100));
		if (iIterate == 100) printf("\niIterate==100 dbydbetatransversederiv = %1.9E \n\n", ROCtransversederiv);

	}
	else {

		f64 overroot = 1.0 / root;
		f64 y1 = transversederivprev;
		f64 y2 = transversederivnext;

		f64 avgasinh = 0.5*(asinh(20.0*y1*overroot) + asinh(20.0*y2*overroot));
		f64 ratio = (y1*ROCprev + y2*ROCnext) / (y1*y1 + y2*y2);

		ROCtransversederiv = 
			(y1*ROCprev + y2*ROCnext)*0.05*overroot*sinh(avgasinh)
			+ root*0.05*cosh(avgasinh)*10.0*
			(     (ROCprev - y1*ratio) / sqrt(401.0*y1*y1 + y2*y2)
				+ (ROCnext - y2*ratio) / sqrt(y1*y1 + 401.0*y2*y2)     );

		printf("ROCtraversederiv %1.10E \n", ROCtransversederiv);

		// Are we ignoring the change in root ????????????????????????????????????????????

	}

	// Get longi derivative first :
	f64_vec2 ROCgradv;

	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.x = -1.0 / (opppos - ourpos).dot(opppos - ourpos);
	ROCgradv.y = ROCgradv.x * (opppos.y - ourpos.y);
	ROCgradv.x *= (opppos.x - ourpos.x);

	printf("ROCgradv : longitudinal %1.10E %1.10E ", ROCgradv.x, ROCgradv.y);

	ROCtransversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
	ROCgradv.x += ROCtransversederiv*vec_out.x;
	ROCgradv.y += ROCtransversederiv*vec_out.y;

	printf("ROCgradv : transverse contrib %1.10E %1.10E ", ROCtransversederiv*vec_out.x, ROCtransversederiv*vec_out.y);

	// DEBUG:
	if (out_modulus == 0) printf("Out_modulus == 0\n");

	return ROCgradv;
}



__device__ __forceinline__ f64_vec2 GetGradientDBydBeta_3Point(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 opp_v,
	f64 prev_a, f64 our_a, f64 opp_a // adding beta*a to v
)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:

	//f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	//f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	//f64 out_modulus = vec_out.modulus();

	//f64 transversederivnext = (next_v - (our_v + (opp_v - our_v)*proportion)) / out_modulus;
	//// deriv in the direction towards nextpos.
	//f64 d_by_dbeta_next = (next_a - (our_a + (opp_a - our_a)*proportion)) / out_modulus;

	f64 proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (prevpos - (ourpos + proportion*(opppos - ourpos)));
	f64 lengthout = vec_out.modulus();

	f64 transversederiv = (prev_v - (our_v + (opp_v - our_v)*proportion)) / lengthout;
	f64 d_by_dbeta_transversederiv = (prev_a - (our_a + (opp_a - our_a)*proportion)) / lengthout;

	// Get longi derivative first :
	f64_vec2 d_by_dbeta_gradv;
	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	//gradv.y = gradv.x * (opppos.y - ourpos.y);
	//gradv.x *= (opppos.x - ourpos.x);
	d_by_dbeta_gradv.x = (opp_a - our_a) / (opppos - ourpos).dot(opppos - ourpos);
	d_by_dbeta_gradv.y = d_by_dbeta_gradv.x * (opppos.y - ourpos.y);
	d_by_dbeta_gradv.x *= (opppos.x - ourpos.x);

	d_by_dbeta_transversederiv /= lengthout; // no longer deriv -- done this way to minimize divides.
											   //gradv.x += transversederiv*vec_out.x;
											   //gradv.y += transversederiv*vec_out.y;
	d_by_dbeta_gradv.x += d_by_dbeta_transversederiv*vec_out.x;
	d_by_dbeta_gradv.y += d_by_dbeta_transversederiv*vec_out.y;
	return d_by_dbeta_gradv;
}


__device__ __forceinline__ f64_vec2 GetGradientDBydBeta(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 next_v, f64 opp_v,
	f64 prev_a, f64 our_a, f64 next_a, f64 opp_a // adding beta*a to v
	)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:

	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	f64 transversederivnext = (next_v - (our_v + (opp_v - our_v)*proportion))/out_modulus;
	// deriv in the direction towards nextpos.
	f64 d_by_dbeta_next = (next_a - (our_a + (opp_a - our_a)*proportion))/out_modulus;

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();

	f64 transversederivprev = -(prev_v - (our_v + (opp_v - our_v)*proportion)) /lengthout;
	f64 d_by_dbeta_prev = -(prev_a - (our_a + (opp_a - our_a)*proportion)) /lengthout;

	// transversederiv = asinh(0.5*(sinh(prev)+sinh(next)))

	// (asinh x)' =  1/ sqrt(1+x*x)
	// d/dx sinh x = cosh x

	// Empirical derivative might well be faster.
	f64 d_by_dbeta_transversederiv;
	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);
	
	if (root == 0.0) {
	// 0 may be wrong? 
	// Be careful what rescaling is doing in the case both operands are very small.

	// Try empirical estimate in this case.

		f64 vlocal = sqrt(0.25*(our_v*our_v + prev_v*prev_v + next_v*next_v + opp_v*opp_v));
		f64 smallstep = 1.0e-7 + 1.0e-7*vlocal;
		
		// Wrong idea to think we can tune smallstep to get accuracy. -- ?
		// In this case of starting from transversederiv == 0, we might be able to.

		bool bCont = false;
		int iIterate = 0;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			root = smallstep*sqrt(d_by_dbeta_prev*d_by_dbeta_prev + d_by_dbeta_next*d_by_dbeta_next);
			
			if (root == 0.0) {
				d_by_dbeta_transversederiv = 0.0;
			} else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev*smallstep)
						+ asinh(20.0*overroot*d_by_dbeta_next*smallstep)));
				f64 d_by_dbeta_transversederiv2 = transversederiv_stepped / smallstep;

				smallstep *= 0.5;
				root *= 0.5;
				overroot *= 2.0;
				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev*smallstep)
						+ asinh(20.0*overroot*d_by_dbeta_next*smallstep)));
				f64 d_by_dbeta_transversederiv1 = transversederiv_stepped / smallstep;

				if ((d_by_dbeta_transversederiv2 > 1.01*d_by_dbeta_transversederiv1) ||
					(d_by_dbeta_transversederiv2 < 0.99*d_by_dbeta_transversederiv1)) {
					// go again
					bCont = true;
				} else {
					d_by_dbeta_transversederiv = d_by_dbeta_transversederiv1
						+ d_by_dbeta_transversederiv1 - d_by_dbeta_transversederiv2;
					// let's assume derivative changes linearly.
					// Q2 = Q(step*0.5) = 0.25*step*step*f'' + 0.5*step*f'
					// Q1 = Q(step*0.25) = 0.25*0.25*step*step*f'' + 0.25*step*f'
					// Q(0) = ?
					bCont = false;
				};
			};
		} while ((bCont) && (iIterate < 100));
		if (iIterate == 100) printf("\niIterate==100 dbydbetatransversederiv = %1.9E \n\n",d_by_dbeta_transversederiv);
	} else {
		f64 overroot = 1.0 / root;
		f64 y1 = transversederivprev;
		f64 y2 = transversederivnext;

		f64 avgasinh = 0.5*(asinh(20.0*y1*overroot) + asinh(20.0*y2*overroot));
		f64 ratio = (y1*d_by_dbeta_prev + y2*d_by_dbeta_next) / (y1*y1 + y2*y2);

		d_by_dbeta_transversederiv = (y1*d_by_dbeta_prev + y2*d_by_dbeta_next)*0.05*overroot*
			sinh(avgasinh) + root*0.05*cosh(avgasinh)*10.0*
			((d_by_dbeta_prev - y1*ratio) / sqrt(401.0*y1*y1 + y2*y2)
				+ (d_by_dbeta_next - y2*ratio) / sqrt(y1*y1 + 401.0*y2*y2));
		
		//transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
		//	+ asinh(20.0*overroot*transversederivnext)));

		
	};

	// Get longi derivative first :
	f64_vec2 d_by_dbeta_gradv;
	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	//gradv.y = gradv.x * (opppos.y - ourpos.y);
	//gradv.x *= (opppos.x - ourpos.x);
	d_by_dbeta_gradv.x = (opp_a - our_a) / (opppos - ourpos).dot(opppos - ourpos);
	d_by_dbeta_gradv.y = d_by_dbeta_gradv.x * (opppos.y - ourpos.y);
	d_by_dbeta_gradv.x *= (opppos.x - ourpos.x);
	
	d_by_dbeta_transversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
	//gradv.x += transversederiv*vec_out.x;
	//gradv.y += transversederiv*vec_out.y;
	d_by_dbeta_gradv.x += d_by_dbeta_transversederiv*vec_out.x;
	d_by_dbeta_gradv.y += d_by_dbeta_transversederiv*vec_out.y;
	return d_by_dbeta_gradv;
}

__device__ __forceinline__ f64_vec2 GetGradientDBydBetaDebug(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v, f64 our_v, f64 next_v, f64 opp_v,
	f64 prev_a, f64 our_a, f64 next_a, f64 opp_a // adding beta*a to v
)
{
	int iIterate;
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	// Consider the transverse derivative:

	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	f64 transversederivnext = (next_v - (our_v + (opp_v - our_v)*proportion)) / out_modulus;
	// deriv in the direction towards nextpos.
	f64 d_by_dbeta_next = (next_a - (our_a + (opp_a - our_a)*proportion)) / out_modulus;

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();

	f64 transversederivprev = -(prev_v - (our_v + (opp_v - our_v)*proportion)) / lengthout;
	f64 d_by_dbeta_prev = -(prev_a - (our_a + (opp_a - our_a)*proportion)) / lengthout;

	// transversederiv = asinh(0.5*(sinh(prev)+sinh(next)))

	// (asinh x)' =  1/ sqrt(1+x*x)
	// d/dx sinh x = cosh x

	// Empirical derivative might well be faster.
	f64 d_by_dbeta_transversederiv;
	f64 root = sqrt(transversederivprev*transversederivprev +
		transversederivnext*transversederivnext);
	printf("root %1.12E \n", root);
	if (root == 0.0) {
		// 0 may be wrong? 
		// Be careful what rescaling is doing in the case both operands are very small.

		// Try empirical estimate in this case.
		f64 vlocal = sqrt(0.25*(our_v*our_v + prev_v*prev_v + next_v*next_v + opp_v*opp_v));
		f64 smallstep = 1.0e-7 + 1.0e-7*vlocal;

		bool bCont = false;
		iIterate = 0;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			root = smallstep*sqrt(d_by_dbeta_prev*d_by_dbeta_prev + d_by_dbeta_next*d_by_dbeta_next);

			if (root == 0.0) {
				d_by_dbeta_transversederiv = 0.0;

				printf("iIterate %d : root == 0 ", iIterate);
			}
			else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev*smallstep)
						+ asinh(20.0*overroot*d_by_dbeta_next*smallstep)));
				f64 d_by_dbeta_transversederiv2 = transversederiv_stepped / smallstep;
				
				printf("iIterate %d : root %1.9E step %1.4E ", iIterate, root, smallstep);

				smallstep *= 0.5;
				root *= 0.5;
				overroot *= 2.0;
				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev)
						+ asinh(20.0*overroot*d_by_dbeta_next)));
				f64 d_by_dbeta_transversederiv1 = transversederiv_stepped / smallstep;

				printf("emp xverse12 %1.12E %1.12E \n",
					d_by_dbeta_transversederiv1, d_by_dbeta_transversederiv2);

				if ((d_by_dbeta_transversederiv2 > 1.01*d_by_dbeta_transversederiv1) ||
					(d_by_dbeta_transversederiv2 < 0.99*d_by_dbeta_transversederiv1)) {
					// go again
					bCont = true;
				}
				else {
					d_by_dbeta_transversederiv = d_by_dbeta_transversederiv1
						+ d_by_dbeta_transversederiv1 - d_by_dbeta_transversederiv2;
					// let's assume derivative changes linearly.
					// Q2 = Q(step*0.5) = 0.25*step*step*f'' + 0.5*step*f'
					// Q1 = Q(step*0.25) = 0.25*0.25*step*step*f'' + 0.25*step*f'
					// Q(0) = ?

				};
			};
		} while ((bCont) && (iIterate < 100));
	}
	else {
		f64 overroot = 1.0 / root;
		//f64 transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
		//	+ asinh(20.0*overroot*transversederivnext)));
		/*
		f64 smallstep = 1.0e-7;
		iIterate = 0;
		bool bCont = false;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			f64 putativeprev = transversederivprev + smallstep*d_by_dbeta_prev;
			f64 putativenext = transversederivnext + smallstep*d_by_dbeta_next;
			root = sqrt(putativeprev*putativeprev+putativenext*putativenext);
			if (root == 0.0) {
				d_by_dbeta_transversederiv = 0.0;
				printf("root == 0.0\n");
				bCont = false;
			}
			else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped;
				f64 transversederiv_stepped2 =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*putativeprev)
						+ asinh(20.0*overroot*putativenext)));
				f64 d_by_dbeta_transversederiv2 = (transversederiv_stepped2-transversederiv) / smallstep;

				smallstep *= 0.5;

				putativeprev = transversederivprev + smallstep*d_by_dbeta_prev;
				putativenext = transversederivnext + smallstep*d_by_dbeta_next;
				root = sqrt(putativeprev*putativeprev + putativenext*putativenext);
				overroot = 1.0 / root;

				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*putativeprev)
						+ asinh(20.0*overroot*putativenext)));
				f64 d_by_dbeta_transversederiv1 = (transversederiv_stepped-transversederiv) / smallstep;

				printf("empirical, smallstep %1.9E d/db xverse12 %1.9E %1.9E putativeprevnext %1.10E %1.10E\n estderivs %1.14E %1.14E %1.14E\n",
					smallstep, d_by_dbeta_transversederiv1, d_by_dbeta_transversederiv2,
					putativeprev, putativenext,
					transversederiv, transversederiv_stepped, transversederiv_stepped2
					);

				if ((d_by_dbeta_transversederiv2 > 1.01*d_by_dbeta_transversederiv1) ||
					(d_by_dbeta_transversederiv2 < 0.99*d_by_dbeta_transversederiv1)) {
					// go again
					bCont = true;
				}
				else {
					d_by_dbeta_transversederiv = d_by_dbeta_transversederiv1
						+ d_by_dbeta_transversederiv1 - d_by_dbeta_transversederiv2;
					// let's assume derivative changes linearly.
					// Q2 = Q(step*0.5) = 0.25*step*step*f'' + 0.5*step*f'
					// Q1 = Q(step*0.25) = 0.25*0.25*step*step*f'' + 0.25*step*f'
					// Q(0) = ?
					bCont = false;
				};
			};
		} while ((bCont) && (iIterate < 20));

		printf("empirical d/db xverse %1.13E \n", d_by_dbeta_transversederiv);*/
		// Note this sometimes comes out with 0 as an answer -- hopefully in the case
		// of the 'initially derivs == 0' case when it's used in anger, it won't.

		overroot = 1.0 / root;
		f64 y1 = transversederivprev;
		f64 y2 = transversederivnext;

		f64 avgasinh = 0.5*(asinh(20.0*y1*overroot) + asinh(20.0*y2*overroot));
		f64 ratio = (y1*d_by_dbeta_prev + y2*d_by_dbeta_next) / (y1*y1 + y2*y2);
		printf("overroot %1.9E avgasinh %1.9E ratio %1.9E \n", overroot, avgasinh, ratio);
		d_by_dbeta_transversederiv = (y1*d_by_dbeta_prev + y2*d_by_dbeta_next)*0.05*overroot*
			sinh(avgasinh) + root*0.05*cosh(avgasinh)*10.0*
			((d_by_dbeta_prev - y1*ratio) / sqrt(401.0*y1*y1 + y2*y2)
				+ (d_by_dbeta_next - y2*ratio) / sqrt(y1*y1 + 401.0*y2*y2));
		printf("answer %1.13E sinh(avg) %1.10E cosh(avg) %1.10E\n---------\n", d_by_dbeta_transversederiv,
			sinh(avgasinh), cosh(avgasinh));

		//transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
		//	+ asinh(20.0*overroot*transversederivnext)));

	};

	// Get longi derivative first :
	f64_vec2 d_by_dbeta_gradv;
	//gradv.x = (opp_v - our_v) / (opppos - ourpos).dot(opppos - ourpos);
	//gradv.y = gradv.x * (opppos.y - ourpos.y);
	//gradv.x *= (opppos.x - ourpos.x);
	d_by_dbeta_gradv.x = (opp_a - our_a) / (opppos - ourpos).dot(opppos - ourpos);
	d_by_dbeta_gradv.y = d_by_dbeta_gradv.x * (opppos.y - ourpos.y);
	d_by_dbeta_gradv.x *= (opppos.x - ourpos.x);

	printf("longitudinal component %1.10E %1.10E\n", d_by_dbeta_gradv.x, d_by_dbeta_gradv.y);

	d_by_dbeta_transversederiv /= out_modulus; // no longer deriv -- done this way to minimize divides.
											   //gradv.x += transversederiv*vec_out.x;
											   //gradv.y += transversederiv*vec_out.y;
	d_by_dbeta_gradv.x += d_by_dbeta_transversederiv*vec_out.x;
	d_by_dbeta_gradv.y += d_by_dbeta_transversederiv*vec_out.y;
	printf("tranverse component %1.10E %1.10E\n", d_by_dbeta_transversederiv*vec_out.x, d_by_dbeta_transversederiv*vec_out.y);
	return d_by_dbeta_gradv;
}



/*
__device__ void GetGradient3(f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
	f64 prev_v1, f64 our_v1, f64 next_v1, f64 opp_v1, f64_vec2 * p_result1,
	f64 prev_v2, f64 our_v2, f64 next_v2, f64 opp_v2, f64_vec2 * p_result2,
	f64 prev_v3, f64 our_v3, f64 next_v3, f64 opp_v3, f64_vec2 * p_result3	
	)
{
	// New method that can make sense for velocities:
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	// Consider the transverse derivative:

	f64 proportion = (nextpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64_vec2 vec_out = (nextpos - (ourpos + proportion*(opppos - ourpos)));
	f64 out_modulus = vec_out.modulus();

	f64 intermediatevalue = our_v1 + (opp_v1 - our_v1)*proportion;
	f64 transversederivnext1 = (next_v1 - intermediatevalue) / out_modulus;
	intermediatevalue = our_v2 + (opp_v2 - our_v2)*proportion;
	f64 transversederivnext2 = (next_v2 - intermediatevalue) / out_modulus;
	intermediatevalue = our_v3 + (opp_v3 - our_v3)*proportion;
	f64 transversederivnext3 = (next_v3 - intermediatevalue) / out_modulus;

	// deriv in the direction towards nextpos.

	proportion = (prevpos - ourpos).dot(opppos - ourpos) / (opppos - ourpos).dot(opppos - ourpos);
	f64 lengthout = (prevpos - (ourpos + proportion*(opppos - ourpos))).modulus();
	
	intermediatevalue = our_v1 + (opp_v1 - our_v1)*proportion;
	f64 transversederivprev1 = -(prev_v1 - intermediatevalue) / lengthout;
	intermediatevalue = our_v2 + (opp_v2 - our_v2)*proportion;
	f64 transversederivprev2 = -(prev_v2 - intermediatevalue) / lengthout;
	intermediatevalue = our_v3 + (opp_v3 - our_v3)*proportion;
	f64 transversederivprev3 = -(prev_v3 - intermediatevalue) / lengthout;

	// Special formula now found:
	f64 transversederiv = asinh(0.5*(sinh(transversederivprev1) + sinh(transversederivnext1)));
	
	// Get longi derivative :
	p_result1->x = (opp_v1 - our_v1) / (opppos - ourpos).dot(opppos - ourpos);
	p_result1->y = p_result1->x * ((opppos.y - ourpos.y));
	p_result1->x *= (opppos.x - ourpos.x);
	transversederiv /= out_modulus; // it is no longer the deriv
	p_result1->x += transversederiv*vec_out.x;
	p_result1->y += transversederiv*vec_out.y;

	// Special formula now found:
	transversederiv = asinh(0.5*(sinh(transversederivprev2) + sinh(transversederivnext2)));

	// Get longi derivative :
	p_result2->x = (opp_v2 - our_v2) / (opppos - ourpos).dot(opppos - ourpos);
	p_result2->y = p_result2->x * ((opppos.y - ourpos.y));
	p_result2->x *= (opppos.x - ourpos.x);
	transversederiv /= out_modulus; // it is no longer the deriv
	p_result2->x += transversederiv*vec_out.x;
	p_result2->y += transversederiv*vec_out.y;

	// Special formula now found:
	transversederiv = asinh(0.5*(sinh(transversederivprev3) + sinh(transversederivnext3)));

	// Get longi derivative :
	p_result3->x = (opp_v3 - our_v3) / (opppos - ourpos).dot(opppos - ourpos);
	p_result3->y = p_result3->x * ((opppos.y - ourpos.y));
	p_result3->x *= (opppos.x - ourpos.x);
	transversederiv /= out_modulus; // it is no longer the deriv
	p_result3->x += transversederiv*vec_out.x;
	p_result3->y += transversederiv*vec_out.y;	
}
*/



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

	int iSpecies)
{	
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																		 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
	// Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
	// But it means we are not being consistent with our definition of a cell?
	// Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];      

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.
							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.
							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.
							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));		
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	} else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64 x_out, x_anti, x_clock;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	//NTrates ourrates;      // +5
	//f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	//f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 result = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)) )
	{
		// Need this, we are adding on to existing d/dt N,NT :
		//memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
			pIndexNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
			pPBCNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
			szPBCtri_verts + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
			// Now do Tn:

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = shared_pos_verts[indexneigh - StartMajor];
			T_clock = shared_T[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_clock = info2.pos;
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};

		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};

		if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
		{
			endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
		}
		else {
#ifdef CENTROID_HEATCONDUCTION
			endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
			endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
		}
		PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
		if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

//		if (T_clock == 0.0) {
//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
//		};
//		Mimic


#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			{
				short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
			}
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
				// Now let's see
				// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

			if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
			{
				endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_anti = p_info_minor[izTri[iNeigh]].pos;
				// we should switch back to centroids!!
#else
				endpt_anti = p_cc[izTri[iNeigh]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

			edge_normal.x = (endpt_anti.y - endpt_clock.y);
			edge_normal.y = (endpt_clock.x - endpt_anti.x);
			
			// SMARTY:
			if (TestDomainPos(pos_out))
			{				
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
					//	ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
					//		(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						result += TWOTHIRDS * kappa_par *(-1.0)*(edgelen / (pos_out - info.pos).modulus());
						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}					
				} else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);
					
					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
					//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;


						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						//if ((T_out > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						//}
						//else {
						//	sqrt_Tout_Tanti = 0.0;
						//}
						//if ((T_out > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						//}
						//else {
						//	sqrt_Tout_Tclock = 0.0;
						//}

						//if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						//}
						//else {
						//	sqrt_Tours_Tanti = 0.0;
						//}
						//if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						//}
						//else {
						//	sqrt_Tours_Tclock = 0.0;
						//}
					
						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];
						
						// Simplify:

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						if ((T_anti > 0.0) && (T_clock > 0.0)) {

							coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
							coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

							//grad_T.x = 0.25*(
							//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							//	);

							f64 sqrt_Tanti, sqrt_Tclock;
							if (T_anti > 0.0) {
								sqrt_Tanti = sqrt(T_anti);
							}
							else {
								sqrt_Tanti = 0.0;
							};
							if (T_clock > 0.0) {
								sqrt_Tclock = sqrt(T_clock);
							}
							else {
								sqrt_Tclock = 0.0;
							};

							coeffsqrt_grad_T.x = 0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
								+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
								) / Area_hex;

							coeffsqrt_grad_T.y = -0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
								+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
								) / Area_hex;

							// Isotropic part:
							f64 result_coeff_self = TWOTHIRDS * kappa_par *(
								nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								(-1.0)*(edgelen / delta_out)
								+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								;
							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								;

							f64 over_sqrt_T_ours;
							if (shared_T[threadIdx.x] > 0.0) {
								over_sqrt_T_ours = 1.0 / sqrt(shared_T[threadIdx.x]);
							}
							else {
								over_sqrt_T_ours = 0.0; // if shared_T wasn't > 0 then all sqrt terms involving it were evaluated 0.
							}

							result += result_coeff_self + 0.5*result_coeff_sqrt*over_sqrt_T_ours;
						} else {

							coeffself_grad_T = -(pos_out - pos_ours) / (pos_out - pos_ours).dot(pos_out - pos_ours);
							f64 result_coeff_self = TWOTHIRDS * kappa_par *(
								nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								(-1.0)*(edgelen / delta_out)
								+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								;
							result += result_coeff_self;

						};
						// Let's be careful ---- if we ARE dealing with a value below zero, all the sqrt come out as 0,
						// so the contribution of 0 to deps/dT is correct.
					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				// Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
		}; // next iNeigh
				
	}; // DOMAIN vertex active in mask

	// Turned out to be stupid having a struct called NTrates. We just want to modify one scalar at a time.
	
	p___result[iVertex] = result;

	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}


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

	int iSpecies)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	NTrates ourrates;      // +5
						   //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						   //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		// Need this, we are adding on to existing d/dt N,NT :
		memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
			pIndexNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
			pPBCNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
			szPBCtri_verts + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
		// Now do Tn:

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = shared_pos_verts[indexneigh - StartMajor];
			T_clock = shared_T[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_clock = info2.pos;
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};

		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};

		if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
		{
			endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
		}
		else {
#ifdef CENTROID_HEATCONDUCTION
			endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
			endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
		}
		PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
		if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic


#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			{
				short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
			}
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
			// Now let's see
			// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

			if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
			{
				endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_anti = p_info_minor[izTri[iNeigh]].pos;
				// we should switch back to centroids!!
#else
				endpt_anti = p_cc[izTri[iNeigh]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

			edge_normal.x = (endpt_anti.y - endpt_clock.y);
			edge_normal.y = (endpt_clock.x - endpt_anti.x);

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

//						if (0) printf("%d %d kappa_par %1.10E edgelen %1.10E delta %1.10E T %1.10E \n"
//							"T_out %1.14E contrib %1.14E flux coefficient on T_out %1.14E\n",
//							iVertex, indexneigh, kappa_par, edgelen, (pos_out - info.pos).modulus(), shared_T[threadIdx.x], T_out,
//							TWOTHIRDS * kappa_par * edgelen *
//							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus(),
//							TWOTHIRDS * kappa_par * edgelen / (pos_out - info.pos).modulus()	);
						
						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						if ((T_out > 0.0) && (T_anti > 0.0)) {
							sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						} else {
							sqrt_Tout_Tanti = 0.0;
						};
						if ((T_out > 0.0) && (T_clock > 0.0)) {
							sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						}			else {
							sqrt_Tout_Tclock = 0.0;
						};
						if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
							sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						}		else {
							sqrt_Tours_Tanti = 0.0;
						};
						if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
							sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						}			else {
							sqrt_Tours_Tclock = 0.0;
						};
						//grad_T.x = 0.5*(T_out + sqrt_Tout_Tanti)*0.5*(pos_anti.y - pos_out.y)
						//	+ 0.5*(sqrt_Tours_Tanti + sqrt_Tout_Tanti)*
						//	//(0.5*(pos_ours.y + pos_anti.y) - 0.5*(pos_out.y + pos_anti.y))
						//	0.5*(pos_ours.y - pos_out.y)
						//	+ 0.5*(sqrt_Tours_Tanti + T_ours)*0.5*(pos_ours.y - pos_anti.y)
						//	+ 0.5*(sqrt_Tours_Tclock + T_ours)*0.5*(pos_clock.y - pos_ours.y)
						//	+ 0.5*(sqrt_Tours_Tclock + sqrt_Tout_Tclock)*
						//	0.5*(pos_out.y - pos_ours.y)
						//	+ 0.5*(sqrt_Tout_Tclock + T_out)*0.5*(pos_out.y - pos_clock.y);

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];
						
						if ((T_anti == 0.0) || (T_clock == 0.0)) {

							grad_T = (T_out - shared_T[threadIdx.x])*(pos_out - info.pos) /
								(pos_out - info.pos).dot(pos_out - info.pos);
						} else {
							// Simplify:
							grad_T.x = 0.25*(
								(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
								+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
								+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
								+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
								);
							// could simplify further to just take coeff on each T value.

							grad_T.y = -0.25*(
								(T_out + sqrt_Tout_Tanti)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.x - pos_out.x)
								+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.x - pos_anti.x)
								+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.x - pos_ours.x)
								+ (sqrt_Tout_Tclock + T_out)*(pos_out.x - pos_clock.x)
								);

							// Integrate 1 : = integral of df/dx for f(x,y) = x.
							//		Area_hex = 0.5*(pos_out.x + 0.5*(pos_out.x+pos_anti.x))*0.5*(pos_anti.y - pos_out.y)
							//			+ 0.5*(0.5*(pos_out.x+pos_anti.x) + 0.5*(pos_ours.x+pos_anti.x))*0.5*(pos_ours.y - pos_out.y)
							//			+ 0.5*(0.5*(pos_ours.x+pos_anti.x) + pos_ours.x)*0.5*(pos_ours.y - pos_anti.y)
							//			+ 0.5*(0.5*(pos_ours.x+pos_clock.x) + pos_ours.x)*0.5*(pos_clock.y - pos_ours.y)
							//			+ 0.5*(0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x+pos_clock.x))*0.5*(pos_out.y - pos_ours.y)
							//			+ 0.5*(0.5*(pos_out.x + pos_clock.x) + pos_out.x)*0.5*(pos_out.y - pos_clock.y);

							// Simplify:
							f64 Area_hex = 0.25*(
								(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
								+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
								+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
								+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
								+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
								+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
								);

							grad_T.x /= Area_hex;
							grad_T.y /= Area_hex;

						};

						if (iSpecies == 1) {

							// Isotropic part:
							ourrates.NiTi += TWOTHIRDS * kappa_par *(
								nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								+
								(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								;
						}
						else {
							ourrates.NeTe += TWOTHIRDS * kappa_par *(
								nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								+
								(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								; // same thing
#if TESTHEAT
								printf("%d %d iSpecies %d contrib %1.10E T %1.9E T_out %1.9E Tanti %1.9E Tclock %1.9E\n"
									"kappa_par %1.9E nu %1.9E omega %1.9E %1.9E grad_T %1.10E %1.10E \n"
									"omega.dotxy(grad_T) %1.9E omega.dotxy(edge_normal) %1.9E Bdiffusive term %1.9E isotropic %1.9E\n",
									iVertex, iNeigh, iSpecies,
									TWOTHIRDS * kappa_par *(
										nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
										+
										(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
										) / (nu * nu + omega.dot(omega)),

									shared_T[threadIdx.x], T_out, T_anti, T_clock, kappa_par, nu, omega.x, omega.y,
									grad_T.x, grad_T.y, omega.dotxy(grad_T), omega.dotxy(edge_normal),
									TWOTHIRDS * kappa_par *(
										(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
										) / (nu * nu + omega.dot(omega)),
									TWOTHIRDS * kappa_par *(
										nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
										) / (nu * nu + omega.dot(omega))
								);
#endif
						};

						

					}
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	   // Turned out to be stupid having a struct called NTrates. We just want to modify one scalar at a time.

	memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}

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

	int iSpecies)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double
	bool bMask;
	f64 result;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	//NTrates ourrates;      // +5
						   //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						   //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		// Need this, we are adding on to existing d/dt N,NT :
	//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		our_x = p__x[iVertex];

		memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
			pIndexNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
			pPBCNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
			szPBCtri_verts + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
		// Now do Tn:

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = shared_pos_verts[indexneigh - StartMajor];
			T_clock = shared_T[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_clock = info2.pos;
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		x_out = p__x[indexneigh];

		if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
		{
			endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
		}
		else {
#ifdef CENTROID_HEATCONDUCTION
			endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
			endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
		}
		PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
		if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic


#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			{
				short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
			}
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
			// Now let's see
			// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

			if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
			{
				endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_anti = p_info_minor[izTri[iNeigh]].pos;
				// we should switch back to centroids!!
#else
				endpt_anti = p_cc[izTri[iNeigh]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

			edge_normal.x = (endpt_anti.y - endpt_clock.y);
			edge_normal.y = (endpt_clock.x - endpt_anti.x);

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
							(-1.0) / (pos_out - info.pos).modulus();
						d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				} else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						} else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						//if ((T_out > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						//}
						//else {
						//	sqrt_Tout_Tanti = 0.0;
						//}
						//if ((T_out > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						//}
						//else {
						//	sqrt_Tout_Tclock = 0.0;
						//}

						//if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						//}
						//else {
						//	sqrt_Tours_Tanti = 0.0;
						//}
						//if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						//}
						//else {
						//	sqrt_Tours_Tclock = 0.0;
						//}

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
						coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
						if (T_anti > 0.0) {
							sqrt_Tanti = sqrt(T_anti);
						}
						else {
							sqrt_Tanti = 0.0;
						};
						if (T_clock > 0.0) {
							sqrt_Tclock = sqrt(T_clock);
						}
						else {
							sqrt_Tclock = 0.0;
						};
						if (shared_T[threadIdx.x] > 0.0) {
							sqrt_Tours = sqrt(shared_T[threadIdx.x]);
						}
						else {
							sqrt_Tours = 0.0;
						};
						if (T_out > 0.0) {
							sqrt_Tout = sqrt(T_out);
						}
						else {
							sqrt_Tout = 0.0;
						};

						coeffsqrt_grad_T.x = 0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
							+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
							+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
							) / Area_hex;

						coeffsqrt_grad_T.y = -0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
							+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
							+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
							) / Area_hex;

						f64 result_coeff_self;

						if ((T_anti == 0.0) || (T_clock == 0.0)) {

							//grad_T = (T_out - shared_T[threadIdx.x])*(pos_out - info.pos) /
							//	(pos_out - info.pos).dot(pos_out - info.pos);
							coeffself_grad_T =
								-(pos_out - info.pos) / (pos_out - info.pos).dot(pos_out - info.pos);

						};

						result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0) && (T_clock > 0.0)) {		
											
							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));								
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.
												
						d_by_dbeta += our_x*result;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						// coeff on power 1:
						f64_vec2 ROC_grad_wrt_T_out;

						if ((T_anti > 0.0) && (T_clock > 0.0)) {

							ROC_grad_wrt_T_out.x = 0.25*(pos_anti.y - pos_clock.y) / Area_hex;
							ROC_grad_wrt_T_out.y = 0.25*(pos_clock.x - pos_anti.x) / Area_hex;
						} else {
							ROC_grad_wrt_T_out =
								(pos_out - info.pos) / (pos_out - info.pos).dot(pos_out - info.pos);

						}
						// stick to format from above :

						d_by_dbeta += x_out* TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(1.0)*(edgelen / delta_out)
							+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;
						
						if ((T_out > 0.0) && (T_anti > 0.0) && (T_clock > 0.0)) {
							
							f64_vec2 coeff_grad_wrt_sqrt_T_out;
							coeff_grad_wrt_sqrt_T_out.x = 0.25*(
								(sqrt_Tanti)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ (sqrt_Tclock)*(pos_out.y - pos_clock.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_out.y = -0.25*(
								(sqrt_Tanti)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ (sqrt_Tclock)*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_out))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_out*0.5*result_coeff_sqrt / sqrt(T_out);
						};

						// T_anti:
						if ((T_anti > 0.0) && (T_clock > 0.0)) {
							f64_vec2 coeff_grad_wrt_sqrt_T_anti;
							coeff_grad_wrt_sqrt_T_anti.x = 0.25*(
								(sqrt_Tout)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_ours.y - pos_anti.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_anti.y = -0.25*(
								(sqrt_Tout)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_ours.x - pos_anti.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_anti))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_anti*0.5*result_coeff_sqrt / sqrt(T_anti);
						};

						if ((T_anti > 0.0) && (T_clock > 0.0)) {

							//grad_T.x = 0.25*(
							//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							//	);
							f64_vec2 coeff_grad_wrt_sqrt_T_clock;
							coeff_grad_wrt_sqrt_T_clock.x = 0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_clock.y - pos_ours.y)
								+ sqrt_Tout*(pos_out.y - pos_clock.y)
								) / Area_hex;
							coeff_grad_wrt_sqrt_T_clock.y = -0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_clock.x - pos_ours.x)
								+ sqrt_Tout*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_clock))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_clock*0.5*result_coeff_sqrt / sqrt(T_clock);
						};

					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
		}; // next iNeigh

		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		} else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		};

		result = -d_by_dbeta*(h_use / N) + our_x;

	//	if (result != result) printf("iVertex %d NaN result. d/dbeta %1.10E N %1.8E our_x %1.8E \n",
	//		iVertex, d_by_dbeta, N, our_x);


	} else { // was it DOMAIN vertex active in mask
		result = 0.0;
	};
	
	p___result[iVertex] = result;
}


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
	)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;
	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	f64 fzArray[MAXNEIGH_d];  // { deps/dx_j * eps }
	f64 effectself; 
		
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  //NTrates ourrates;      // +5
						  //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						  //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		memset(fzArray, 0, sizeof(f64)*MAXNEIGH_d);

		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		} else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		}
		f64 epsilon = p_epsilon[iVertex];
		f64 our_fac = -2.0*epsilon* h_use / N; // factor for change in epsilon^2
		// But we missed out the effect of changing T on epsilon directly ! ...

		// need to add 1.0
		effectself = epsilon*2.0; // change in own epsilon by changing T is +1.0 for eps = T_k+1-T_k-hF
#if TESTHEAT
		printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);
#endif

		// Need this, we are adding on to existing d/dt N,NT :
		//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
			pIndexNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
			pPBCNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
			szPBCtri_verts + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
		// Now do Tn:

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = shared_pos_verts[indexneigh - StartMajor];
			T_clock = shared_T[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_clock = info2.pos;
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		//x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		//x_out = p__x[indexneigh];

		if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
		{
			endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
		}
		else {
#ifdef CENTROID_HEATCONDUCTION
			endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
			endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
		}
		PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
		if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic
		short iPrev = info.neigh_len - 1;

#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			
			short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
			
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			//x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
			// Now let's see
			// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

			if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
			{
				endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_anti = p_info_minor[izTri[iNeigh]].pos;
				// we should switch back to centroids!!
#else
				endpt_anti = p_cc[izTri[iNeigh]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

			edge_normal.x = (endpt_anti.y - endpt_clock.y);
			edge_normal.y = (endpt_clock.x - endpt_anti.x);

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					} else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

//						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
	//						(-1.0) / (pos_out - info.pos).modulus();
		//				d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
			//				(1.0) / (pos_out - info.pos).modulus();

						f64 temp = TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						fzArray[iNeigh] += temp*our_fac;
						effectself -= temp*our_fac;
#if (TESTHEAT) 
							printf("iVertex %d indexneigh %d temp %1.14E our_fac %1.14E iNeigh %d temp*our_fac %1.14E \n",
								iVertex, indexneigh, temp, our_fac, iNeigh, temp*our_fac);
#endif
						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
						if ((T_anti > 0.0) && (T_clock > 0.0)) {

							coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
							coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

							if (T_anti > 0.0) {
								sqrt_Tanti = sqrt(T_anti);
							}
							else {
								sqrt_Tanti = 0.0;
							};
							if (T_clock > 0.0) {
								sqrt_Tclock = sqrt(T_clock);
							}
							else {
								sqrt_Tclock = 0.0;
							};
							if (shared_T[threadIdx.x] > 0.0) {
								sqrt_Tours = sqrt(shared_T[threadIdx.x]);
							}
							else {
								sqrt_Tours = 0.0;
							};
							if (T_out > 0.0) {
								sqrt_Tout = sqrt(T_out);
							}
							else {
								sqrt_Tout = 0.0;
							};

							coeffsqrt_grad_T.x = 0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
								+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
								) / Area_hex;

							coeffsqrt_grad_T.y = -0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
								+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
								) / Area_hex;

						} else {

							coeffself_grad_T = -(pos_out - pos_ours) / (pos_out - pos_ours).dot(pos_out - pos_ours);

							coeffsqrt_grad_T.x = 0.0;
							coeffsqrt_grad_T.y = 0.0;
						}
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0) && (T_clock > 0.0)) {

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.

						//d_by_dbeta += our_x*result;

						effectself += result*our_fac;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						// coeff on power 1:
						f64_vec2 ROC_grad_wrt_T_out;

						if ((T_anti > 0.0) && (T_clock > 0.0)) {
							ROC_grad_wrt_T_out.x = 0.25*(pos_anti.y - pos_clock.y) / Area_hex;
							ROC_grad_wrt_T_out.y = 0.25*(pos_clock.x - pos_anti.x) / Area_hex;
						} else {
							ROC_grad_wrt_T_out = (pos_out - pos_ours) / (pos_out - pos_ours).dot(pos_out - pos_ours);
						}
						// stick to format from above :

						// Isotropic part:
						//d_by_dbeta += x_out* TWOTHIRDS * kappa_par *(
						//	nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
						//	(1.0)*(edgelen / delta_out)
						//	+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
						//	) / (nu * nu + omega.dot(omega))
						//	;

						fzArray[iNeigh] += our_fac*TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(1.0)*(edgelen / delta_out)
							+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						if ((T_anti > 0.0) && (T_clock > 0.0)) {
							if (T_out > 0.0) {
								f64_vec2 coeff_grad_wrt_sqrt_T_out;
								coeff_grad_wrt_sqrt_T_out.x = 0.25*(
									(sqrt_Tanti)*(pos_anti.y - pos_out.y)
									+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
									+ (sqrt_Tclock)*(pos_out.y - pos_clock.y)
									) / Area_hex;

								coeff_grad_wrt_sqrt_T_out.y = -0.25*(
									(sqrt_Tanti)*(pos_anti.x - pos_out.x)
									+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
									+ (sqrt_Tclock)*(pos_out.x - pos_clock.x)
									) / Area_hex;

								f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
									(omega.dotxy(coeff_grad_wrt_sqrt_T_out))*(omega.dotxy(edge_normal))
									) / (nu * nu + omega.dot(omega));
								//d_by_dbeta += x_out*0.5*result_coeff_sqrt / sqrt(T_out);

								fzArray[iNeigh] += our_fac*0.5*result_coeff_sqrt / sqrt(T_out);

							};

							// T_anti:
							if (T_anti > 0.0) {
								f64_vec2 coeff_grad_wrt_sqrt_T_anti;
								coeff_grad_wrt_sqrt_T_anti.x = 0.25*(
									(sqrt_Tout)*(pos_anti.y - pos_out.y)
									+ (sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
									+ sqrt_Tours*(pos_ours.y - pos_anti.y)
									) / Area_hex;

								coeff_grad_wrt_sqrt_T_anti.y = -0.25*(
									(sqrt_Tout)*(pos_anti.x - pos_out.x)
									+ (sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
									+ sqrt_Tours*(pos_ours.x - pos_anti.x)
									) / Area_hex;

								f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
									(omega.dotxy(coeff_grad_wrt_sqrt_T_anti))*(omega.dotxy(edge_normal))
									) / (nu * nu + omega.dot(omega));

								//d_by_dbeta += x_anti*0.5*result_coeff_sqrt / sqrt(T_anti);

								fzArray[iNext] += our_fac*0.5*result_coeff_sqrt / sqrt(T_anti);
							};

							if (T_clock > 0.0) {

								//grad_T.x = 0.25*(
								//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
								//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
								//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
								//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
								//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
								//	);
								f64_vec2 coeff_grad_wrt_sqrt_T_clock;
								coeff_grad_wrt_sqrt_T_clock.x = 0.25*(
									-(sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
									+ sqrt_Tours*(pos_clock.y - pos_ours.y)
									+ sqrt_Tout*(pos_out.y - pos_clock.y)
									) / Area_hex;
								coeff_grad_wrt_sqrt_T_clock.y = -0.25*(
									-(sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
									+ sqrt_Tours*(pos_clock.x - pos_ours.x)
									+ sqrt_Tout*(pos_out.x - pos_clock.x)
									) / Area_hex;

								f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
									(omega.dotxy(coeff_grad_wrt_sqrt_T_clock))*(omega.dotxy(edge_normal))
									) / (nu * nu + omega.dot(omega));
								//d_by_dbeta += x_clock*0.5*result_coeff_sqrt / sqrt(T_clock);

								fzArray[iPrev] += our_fac*0.5*result_coeff_sqrt / sqrt(T_clock);
							};
						};
					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
			iPrev = iNeigh;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	memcpy(p_array + iVertex*MAXNEIGH_d, fzArray, sizeof(f64)*MAXNEIGH_d);
	p_effectself[iVertex] = effectself;
	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}


__global__ void kernelDivide(
	f64 * __restrict__ p_regress,
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_effectself
) {
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;

	f64 bink = p_effectself[iVertex];
	if (bink != 0.0) {
		p_regress[iVertex] = -p_eps[iVertex] / bink;
	} else {
		p_regress[iVertex] = 0.0;
	};
}


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
)
{
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;
	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	f64 effectself;

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  //NTrates ourrates;      // +5
						  //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						  //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		
		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		}
		else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		}
		f64 our_fac = - h_use / N; // factor for change in epsilon^2											 
		effectself = 1.0; // change in own epsilon by changing T is +1.0 for eps = T_k+1-T_k-hF
#if TESTHEAT
		printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);
#endif

		// Need this, we are adding on to existing d/dt N,NT :
		//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
			pIndexNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
			pPBCNeigh + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
			szPBCtri_verts + MAXNEIGH_d * iVertex,
			MAXNEIGH_d * sizeof(char));
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
		// Now do Tn:

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = shared_pos_verts[indexneigh - StartMajor];
			T_clock = shared_T[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_clock = info2.pos;
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		//x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		//x_out = p__x[indexneigh];

		if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
		{
			endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
		}
		else {
#ifdef CENTROID_HEATCONDUCTION
			endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
			endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
		}
		PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
		if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic
		short iPrev = info.neigh_len - 1;

#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{

			short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];

			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			//x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
			// Now let's see
			// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

			if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
			{
				endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_anti = p_info_minor[izTri[iNeigh]].pos;
				// we should switch back to centroids!!
#else
				endpt_anti = p_cc[izTri[iNeigh]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

			edge_normal.x = (endpt_anti.y - endpt_clock.y);
			edge_normal.y = (endpt_clock.x - endpt_anti.x);

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						//						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
						//						(-1.0) / (pos_out - info.pos).modulus();
						//				d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
						//				(1.0) / (pos_out - info.pos).modulus();

						f64 temp = TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						effectself -= temp*our_fac;
						
						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				} else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						if ((T_anti > 0.0) && (T_clock > 0.0))
						{
							coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
							coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

							f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
							if (T_anti > 0.0) {
								sqrt_Tanti = sqrt(T_anti);
							}
							else {
								sqrt_Tanti = 0.0;
							};
							if (T_clock > 0.0) {
								sqrt_Tclock = sqrt(T_clock);
							}
							else {
								sqrt_Tclock = 0.0;
							};
							if (shared_T[threadIdx.x] > 0.0) {
								sqrt_Tours = sqrt(shared_T[threadIdx.x]);
							}
							else {
								sqrt_Tours = 0.0;
							};
							if (T_out > 0.0) {
								sqrt_Tout = sqrt(T_out);
							}
							else {
								sqrt_Tout = 0.0;
							};

							coeffsqrt_grad_T.x = 0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
								+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
								) / Area_hex;

							coeffsqrt_grad_T.y = -0.25*(
								(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
								+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
								) / Area_hex;

						} else {

							coeffself_grad_T = -(pos_out - pos_ours) / (pos_out - pos_ours).dot(pos_out - pos_ours);

							coeffsqrt_grad_T.x = 0.0;
							coeffsqrt_grad_T.y = 0.0;
						};
						
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if (shared_T[threadIdx.x] > 0.0)  {

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.
						//d_by_dbeta += our_x*result;

						effectself += result*our_fac;
					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
			iPrev = iNeigh;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	p_effectself[iVertex] = effectself;
	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}

__global__ void AddFromMyNeighbours(
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_array,
	f64 * __restrict__ p_arrayself,
	f64 * __restrict__ p_sum,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_who_am_I_to_you
) {
	//This requires who_am_I to always be well updated.;
	__shared__ f64 pArray[threadsPerTileMajor*MAXNEIGH_d];
	// We can actually fit 24 doubles/thread at 256 threads, 48K - so this is actually running 2 tiles at once.
	__shared__ short who_am_I[threadsPerTileMajor*MAXNEIGH_d];
	// think about memory balance => this got shared

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	memcpy(pArray + threadIdx.x*MAXNEIGH_d, p_array + iVertex*MAXNEIGH_d, sizeof(f64)*MAXNEIGH_d);

	__syncthreads();

	long indexneigh[MAXNEIGH_d];

	memcpy(indexneigh, p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
	memcpy(who_am_I + MAXNEIGH_d*threadIdx.x, p_who_am_I_to_you + MAXNEIGH*iVertex, sizeof(short)*MAXNEIGH);

	structural info = p_info_major[iVertex];
	f64 sum = 0.0;
	if (info.flag == DOMAIN_VERTEX) {
		sum = p_arrayself[iVertex]; // 2.0*epsilon _self * deps/dself
		
	//	if ((iVertex == VERTCHOSEN2))
	//		printf("%d sum %1.9E \n", iVertex, sum);

		for (int i = 0; i < info.neigh_len; i++)
		{
			short iWhich = who_am_I[threadIdx.x*MAXNEIGH_d + i];
			long iNeigh = indexneigh[i];
			if ((iNeigh >= StartMajor) && (iNeigh < EndMajor)) {

				sum += pArray[(iNeigh - StartMajor)*MAXNEIGH_d + iWhich];
			//	if ((iVertex == VERTCHOSEN2))
			//		printf("iVertex %d i %d iNeigh %d iWhich %d p_Array[ ] %1.14E sum %1.9E \n", iVertex, i, iNeigh, iWhich,
			//			pArray[(iNeigh - StartMajor)*MAXNEIGH_d + iWhich], sum);

			} else {

				sum += p_array[iNeigh*MAXNEIGH_d + iWhich];
			//	if ((iVertex == VERTCHOSEN2))
			//		printf("iVertex %d i %d iNeigh %d iWhich %d p_array[] %1.14E sum %1.9E \n", iVertex, i, iNeigh, iWhich,
			//			p_array[iNeigh*MAXNEIGH_d + iWhich] , sum);
			};

		};

	}
	p_sum[iVertex] = -sum; // added up eps_j deps_j/dx_i
	// put in minus for steepest descent instead of ascent.

}




__global__ void
// __launch_bounds__(128) -- manual says that if max is less than 1 block, kernel launch will fail. Too bad huh.
kernelCreate_viscous_contrib_to_MAR_and_NT_Geometric_1species(

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
	f64 const over_m_s) 
{

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//                                                                                                //
	//                                   WATCH   OUT                                                  //
	//                                                                                                //
	////////////////////////////////////////////////////////////////////////////////////////////////////

	// A copy of this routine is the ___fixedflows_only routine and any changes made here must be reflected there.





	__shared__ f64_vec3 shared_v[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	__shared__ f64_vec3 shared_v_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

		// 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
		// Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	{
		v4 temp = p_vie_minor[iMinor];
		shared_v[threadIdx.x].x = temp.vxy.x;
		shared_v[threadIdx.x].y = temp.vxy.y;
		if (iSpecies == 1) {
			shared_v[threadIdx.x].z = temp.viz;
		} else {
			shared_v[threadIdx.x].z = temp.vez;
		};
	}
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			v4 temp;
			memcpy(&temp, &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_v_verts[threadIdx.x].x = temp.vxy.x;
			shared_v_verts[threadIdx.x].y = temp.vxy.y;
			if (iSpecies == 1) {
				shared_v_verts[threadIdx.x].z = temp.viz;
			} else {
				shared_v_verts[threadIdx.x].z = temp.vez;
			};
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_v_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&ownrates_visc, 0, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))			
			// ita_par is set to 0 for forward region so including bool for selected eqn isn't the concern it seemed.						
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));
			
			f64_vec3 opp_v, prev_v, next_v;
			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_v[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				v4 temp = p_vie_minor[izTri[iprev]];
				prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y;
				if (iSpecies == 1) { prev_v.z = temp.viz; }
				else { prev_v.z = temp.vez; };
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);// = Clockwise3_d*prev_v;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_v[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				v4 temp = p_vie_minor[izTri[i]];
				opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y;
				if (iSpecies == 1) { opp_v.z = temp.viz; }
				else { opp_v.z = temp.vez; };
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}			

#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext == tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len-1;

				f64_vec2 gradvx, gradvy, gradvz;
				f64_vec3 htg_diff;
				
				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 
				
				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!
				
				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_v[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
#if TEST_VISC_VERT
					if ((iVertex == VERTCHOSEN) && (iSpecies == 2)) 
						printf("inext %d  izTri[inext] %d tri_len %d nextpos %1.9E %1.9E\n", inext, izTri[inext], tri_len,
						nextpos.x, nextpos.y);
#endif
				}
				else {
					v4 temp = p_vie_minor[izTri[inext]];
					next_v.x = temp.vxy.x; next_v.y = temp.vxy.y;
					if (iSpecies == 1) { next_v.z = temp.viz; }
					else { next_v.z = temp.vez; };
					nextpos = p_info_minor[izTri[inext]].pos;
#if TEST_VISC_VERT
					if ((iVertex == VERTCHOSEN) && (iSpecies == 2)) 
						printf("inext %d II izTri[inext] %d tri_len %d\n", inext, izTri[inext], tri_len);
#endif
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				f64_vec3 omega_c;
				{
					f64_vec2 opp_B;
					f64 ita_theirs, nu_theirs;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						ita_theirs = shared_ita_par[izTri[i] - StartMinor];
						nu_theirs = shared_nu[izTri[i] - StartMinor];
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						ita_theirs = p_ita_parallel_minor[izTri[i]];
						nu_theirs = p_nu_minor[izTri[i]];
					};

					// GEOMETRIC ITA:
					ita_par = sqrt(shared_ita_par_verts[threadIdx.x] * ita_theirs);

					if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					// Arithmetic average nu is larger so helps omega/nu to be smaller, making life easier.
					nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); 
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.
				
				bool bLongi = false;
				if (ita_par > 0.0)
				{
					
#ifdef INS_INS_3POINT

					if (TestDomainPos(prevpos) == false) {

#if TEST_VISC_VERT
						if ((iVertex == VERTCHOSEN) && (iSpecies == 2))
							printf("(TestDomainPos(prevpos) == false) vx ours next opp %1.10E %1.10E %1.10E\n",
								shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);
#endif

						gradvx = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
						);
						gradvy = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
						);
						gradvz = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
						);

					} else {
						if (TestDomainPos(nextpos) == false) {
#if TEST_VISC_VERT
							if ((iVertex == VERTCHOSEN) && (iSpecies == 2))
								printf("(TestDomainPos(nextpos) == false) vx prev ours opp %1.10E %1.10E %1.10E\n",
									prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x);
#endif
							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v_verts[threadIdx.x].y, opp_v.y
							);
							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v_verts[threadIdx.x].z, opp_v.z
							);
						} else {
#if TEST_VISC_VERT
							if ((iVertex == VERTCHOSEN) && (iSpecies == 2)) 
								printf("standard edge izTri[i] %d ; vy prev ours next opp %1.10E %1.10E %1.10E %1.10E\n",
									izTri[i], prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
#endif
								
							gradvx = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
							);
							
						};
					};
#if TEST_VISC_VERT
					if ((iVertex == VERTCHOSEN) && (iSpecies == 2))  
						printf("%d %d %d vz psno %1.12E %1.12E %1.12E %1.12E gradvz %1.12E %1.12E\n"
						"prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n",
						//"gradvx %1.12E %1.12E gradvz %1.12E %1.12E vz %1.9E %1.9E %1.9E %1.9E\n"
						iVertex, i, izTri[i],
						prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z, gradvz.x, gradvz.y,
						prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y
						//gradvx.x, gradvx.y, 
						//gradvz.x, gradvz.y,
						//prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z,
						//prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
						);
#endif
					
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false)) bLongi = true;
#else
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						gradvx = (opp_v.x - shared_v_verts[threadIdx.x].x)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						gradvy = (opp_v.y - shared_v_verts[threadIdx.x].y)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						gradvz = (opp_v.z - shared_v_verts[threadIdx.x].z)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);

					} else {
						
						if (TESTIONVERTVISC) {
							printf("%d i inext %d %d izTri %d %d ourpos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E opp %1.8E %1.8E \n",
								iVertex, i, inext, izTri[i], izTri[inext],
								info.pos.x, info.pos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y, opppos.x, opppos.y);
						}

						gradvx = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
						);
						gradvy = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
						);
						gradvz = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
						);
						// Could switch to the 3 in one function that handles all 3. in one.
					}
					// No Area_hex gathered for us.
#endif

//					if (TESTIONVERTVISC) printf(
//						"iVertex %d  \n"
//						"our_v.x next prev opp %1.8E %1.8E %1.8E %1.8E gradvx %1.8E %1.8E\n"
//						"our_v.y next prev opp %1.8E %1.8E %1.8E %1.8E gradvy %1.8E %1.8E\n"
//						"our_v.z next prev opp %1.8E %1.8E %1.8E %1.8E gradvz %1.8E %1.8E\n"
//						"info.pos %1.8E %1.8E opppos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E\n",
//						iVertex,
//						shared_vie_verts[threadIdx.x].vxy.x, next_v.vxy.x, prev_v.vxy.x, opp_v.vxy.x,
//						gradvx.x, gradvx.y,
//						shared_vie_verts[threadIdx.x].vxy.y, next_v.vxy.y, prev_v.vxy.y, opp_v.vxy.y,
//						gradvy.x, gradvy.y,
//						shared_vie_verts[threadIdx.x].viz, next_v.viz, prev_v.viz, opp_v.viz,
//						gradvz.x, gradvz.y,
//						info.pos.x, info.pos.y, opppos.x, opppos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y);

					htg_diff = shared_v_verts[threadIdx.x] - opp_v;
				}
//
//				if ((TEST_EPSILON_Y) || (TEST_EPSILON_X)) {
//					printf("%d %d %d gradvx %1.9E %1.9E gradvy %1.9E %1.9E gradvz %1.9E %1.9E \n"
//						"prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n"
//						"prev_vx %1.12E our_vx %1.12E nextvx %1.12E oppvx %1.12E \n"
//						"prev_vy %1.12E our_vy %1.12E nextvy %1.12E oppvy %1.12E \n",
//						iVertex, i, izTri[i], gradvx.x, gradvx.y, gradvy.x, gradvy.y, gradvz.x, gradvz.y,
//						prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
//						prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
//						prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
//				}
				
				if (ita_par > 0.0) {

					// possibly it always merges this to the block above.


					if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
					{
						// run unmagnetised case
						
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
						if (bLongi) edge_normal = ReconstructEdgeNormal(
							prevpos, info.pos, nextpos, opppos
						);

						// visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);


						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_s*ita_par*( THIRD*(4.0*gradvx.x - 2.0*gradvy.y)*edge_normal.x
													+ (gradvx.y + gradvy.x)*edge_normal.y);
						visc_contrib.y = over_m_s*ita_par*((gradvx.y + gradvy.x)*edge_normal.x
													+ THIRD*(4.0*gradvy.y - 2.0*gradvx.x)*edge_normal.y);
						//-over_m_s*(-ita_par*(gradvx.y + gradvy.x)*edge_normal.x
						//	- ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x)*edge_normal.y);

						//visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);
						visc_contrib.z = over_m_s*ita_par*(gradvz.x*edge_normal.x + gradvz.y*edge_normal.y);

						ownrates_visc += visc_contrib;
						visc_htg += -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib));

						//if (HTGPRINT) printf("Htg %d i %d %d : %1.9E htg_diff %1.7E %1.7E %1.7E vctb %1.7E %1.7E %1.7E\n",
						//	iVertex, i, izTri[i], -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib)),
						//	htg_diff.x, htg_diff.y, htg_diff.z, visc_contrib.x, visc_contrib.y, visc_contrib.z);
#if TEST_VISC_VERT
						
						// check result is the same when this printf is added:

						if ((iVertex == VERTCHOSEN) && (iSpecies == 2))
						{
							//f64 Pi_zx, Pi_zy;
							f64 Pi_xx, Pi_xy; f64 Pi_yx, Pi_yy, Pi_zx, Pi_zy;

							Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
							Pi_xy = -ita_par*(gradvx.y + gradvy.x);
							Pi_yx = -ita_par*(gradvx.y + gradvy.x); // maybe Pi_yx = Pi_xy was doing something : it replaced Pi_yx with Pi_xy to get rid of var?
							Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
							Pi_zx = -ita_par*(gradvz.x);
							Pi_zy = -ita_par*(gradvz.y);
							printf("unmag %d : %d : ita %1.8E gradvx %1.8E %1.8E Pi_xy %1.9E Pi_zx %1.9E Pi_zy %1.9E\n"
								"contrib.x %1.8E yz %1.10E %1.10E edgenml %1.8E %1.8E gradvz %1.12E %1.12E omega.x %1.8E nu %1.8E\n"
								"------------\n",
								iVertex, izTri[i], ita_par, gradvx.x, gradvx.y, Pi_xy, Pi_zx, Pi_zy,
								visc_contrib.x, visc_contrib.y, visc_contrib.z, edge_normal.x, edge_normal.y, gradvz.x, gradvz.y, omega_c.x, nu);
							
							
							// This (with Pi) puts us back to 8e-5.

							// Interesting this has a big effect despite the if clause.
							// But it should not have any at all.
							// So, 8e-5 ... may well be the correct one.
							// It happens when we have printf regardless of what it says above.

							// CRAZY

							// Getted rid of Pi : same result because spit out gradvz??
							// It's 2e-4 if we get rid of Pi from here.

							// Incomprehensible.
							
							//I don't even know which version is correct.

							//There are limited choices of what we can actually do.
							//We cannot ultimately leave printf in

							//But if we leave it w/o Pi , then we can put printf in and out.
							//	Why does Pi do anything?

						};


#endif

						// So we are saying if edge_normal.x > 0 and gradviz.x > 0
						// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					} else {

						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; 
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
							f64 omegamod;
							{
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);

								if (bLongi) edge_normal = ReconstructEdgeNormal(
									prevpos, info.pos, nextpos, opppos
								);
								f64 omegasq = omega_c.dot(omega_c);
								omegamod = sqrt(omegasq);
								unit_b = omega_c / omegamod;
								unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
								unit_perp = unit_perp / unit_perp.modulus();
								unit_Hall = unit_b.cross(unit_perp); // Note sign.

								 // Since B is in the plane, it's saying we picked perp in the plane, H = z.
								// CHECK THAT
								// It probably doesn't matter what direction perp is. As long as it is in the plane normal to b.
								// Formulary tells us if b is z then perp is x, H is y. So we can freely rotate x,y in the normal plane
								// and this will still hold.
								// Let's verify if we got it : 
								// z cross x = (0, 1, 0) --- tick.

								// From the perspective of the +- ita3 and ita4, all that matter is the relative orientation of P and H.
							}
							{
								f64_vec3 intermed;

								// use: d vb / da = b transpose [ dvi/dxj ] a
								// Prototypical element: a.x b.y dvy/dx
								// b.x a.y dvx/dy
								intermed.x = unit_b.dotxy(gradvx);
								intermed.y = unit_b.dotxy(gradvy);
								intermed.z = unit_b.dotxy(gradvz);
								{
									f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

									dvb_by_db = unit_b.dot(intermed);
									dvperp_by_db = unit_perp.dot(intermed);
									dvHall_by_db = unit_Hall.dot(intermed);

									W_bb += 4.0*THIRD*dvb_by_db;
									W_bP += dvperp_by_db;
									W_bH += dvHall_by_db;
									W_PP -= 2.0*THIRD*dvb_by_db;
									W_HH -= 2.0*THIRD*dvb_by_db;
									 
#if TEST_VISC_VERT_deep 
									if (0)
										printf("dvb/db %1.9E dvperp/db %1.9E dvH/db %1.9E W_PP %1.9E W_HH %1.9E\n",
											dvb_by_db, dvperp_by_db, dvHall_by_db, W_PP, W_HH);
#endif
								}
								{
									f64 dvb_by_dperp, dvperp_by_dperp,
										dvHall_by_dperp;
									// Optimize by getting rid of different labels.

									intermed.x = unit_perp.dotxy(gradvx);
									intermed.y = unit_perp.dotxy(gradvy);
									intermed.z = unit_perp.dotxy(gradvz);

									dvb_by_dperp = unit_b.dot(intermed);
									dvperp_by_dperp = unit_perp.dot(intermed);
									dvHall_by_dperp = unit_Hall.dot(intermed);

									W_bb -= 2.0*THIRD*dvperp_by_dperp;
									W_PP += 4.0*THIRD*dvperp_by_dperp;
									W_HH -= 2.0*THIRD*dvperp_by_dperp;
									W_bP += dvb_by_dperp;
									W_PH += dvHall_by_dperp;

#if (TEST_VISC_VERT_deep) //if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
										printf("dvb/dP %1.9E dvperp/dP %1.9E dvH/dP %1.9E W_PP %1.9E W_HH %1.9E \n",
											dvb_by_dperp, dvperp_by_dperp, dvHall_by_dperp, W_PP, W_HH);
#endif
								}
								{
									f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;
									// basically, all should be 0
									
									intermed.x = unit_Hall.dotxy(gradvx);
									intermed.y = unit_Hall.dotxy(gradvy);
									intermed.z = unit_Hall.dotxy(gradvz);

									dvb_by_dHall = unit_b.dot(intermed);
									dvperp_by_dHall = unit_perp.dot(intermed);
									dvHall_by_dHall = unit_Hall.dot(intermed);

									W_bb -= 2.0*THIRD*dvHall_by_dHall;
									W_PP -= 2.0*THIRD*dvHall_by_dHall;
									W_HH += 4.0*THIRD*dvHall_by_dHall;
									W_bH += dvb_by_dHall;
									W_PH += dvperp_by_dHall;
#if (TEST_VISC_VERT_deep) //if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
										printf("dvb/dH %1.9E dvperp/dH %1.9E dvH/dH %1.9E W_PP %1.9E W_HH %1.9E \n",
											dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall, W_PP,W_HH);
#endif

								}
							}							
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
								// W_HH = 0
								//if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
#if (TEST_VISC_VERT_deep)
									printf("ita_1 %1.9E par %1.8E W_bb %1.9E W_PP %1.9E W_HH %1.9E Pi_bb %1.9E Pi_PP %1.9E\n",
										ita_1, ita_par, W_bb, W_PP, W_HH, Pi_b_b, Pi_P_P);
#endif
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;

#if (TEST_VISC_VERT_deep)//if ((TEST_EPSILON_Y) && (iSpecies == 1))
									printf("ita_2 %1.9E W_bP %1.9E Pi_Pb %1.9E \n",
										ita_2, W_bP, Pi_P_b);
#endif
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
#if (TEST_VISC_VERT_deep) //if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
									printf("ita_3 %1.9E W_PH %1.9E Pi_PP %1.9E \n",
										ita_3, W_PH, Pi_P_P);
#endif
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;

#if (TEST_VISC_VERT_deep) // if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
									printf("ita_4 %1.9E W_bH %1.9E Pi_Pb %1.9E \n",
										ita_4, W_bH, Pi_P_b);
#endif
								Pi_H_b += ita_4*W_bP;
							}
						} // scope W

						// All we want left over at this point is Pi .. and unit_b

						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							// Most efficient way: compute mom flux in magnetic coords
							f64_vec3 mag_edge;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);
#if (TEST_VISC_VERT_deep)//(((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1))
								printf("edgenml %1.12E %1.12E\n", edge_normal.x, edge_normal.y);
#endif
							if (bLongi) {
#if (TEST_VISC_VERT_deep) 
									edge_normal = ReconstructEdgeNormalDebug(
										prevpos, info.pos, nextpos, opppos);
#else
									edge_normal = ReconstructEdgeNormal(
										prevpos, info.pos, nextpos, opppos);
#endif
							};
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

#if TEST_VISC_VERT_deep
							if (0)
								printf("mag_edge %1.10E %1.10E %1.10E  unit_b.x %1.8E perp.y %1.8E H.z %1.8E edgenml %1.12E %1.12E\n", mag_edge.x, mag_edge.y, mag_edge.z,
									unit_b.x, unit_perp.y, unit_Hall.z, edge_normal.x, edge_normal.y);
#endif

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
						}

						// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
						// is the flow of p_x dotted with the edge_normal
						// ownrates will be divided by N to give dv/dt
						// m N dvx/dt = integral div momflux_x
						// Therefore divide here just by m

						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
						visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
						visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
												
						ownrates_visc += visc_contrib;
						visc_htg += -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib));

					//	if (HTGPRINT) printf("Htg %d i %d %d : %1.9E htg_diff %1.7E %1.7E %1.7E vctb %1.7E %1.7E %1.7E\n",
					//		iVertex, i, izTri[i], -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib)),
					//		htg_diff.x, htg_diff.y, htg_diff.z, visc_contrib.x, visc_contrib.y, visc_contrib.z);

						//if (TESTXYDERIVZVISCVERT) {
						//	printf("%d visc_ctbz %1.9E over_m_s %1.8E mf b P H %1.9E %1.9E %1.9E b.z P.z H.z %1.8E %1.8E %1.8E \n",
						//		iVertex, visc_contrib.z, over_m_s, momflux_b, momflux_perp, momflux_Hall, unit_b.z, unit_perp.z, unit_Hall.z);
						//	printf("Pi bb Pb Hb PP HP HH %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E \n",
						//		Pi_b_b, Pi_P_b, Pi_H_b, Pi_P_P, Pi_H_P, Pi_H_H);
						//}

						//if (((TEST_EPSILON_Y) || (TEST_EPSILON_X)) && (iSpecies == 1)) 
#if TEST_VISC_VERT
						if ((iVertex == VERTCHOSEN) && (iSpecies == 2)) {
							printf("iVertex %d %d tri %d species %d ita_par %1.9E \n"
								"omega %1.9E %1.9E %1.9E nu %1.11E ourpos %1.8E %1.8E \n"
								"unit_b %1.8E %1.8E %1.8E unit_perp %1.8E %1.7E %1.7E unit_Hall %1.6E %1.6E %1.6E\n",
								iVertex, i, izTri[i], iSpecies, ita_par,
								omega_c.x, omega_c.y, omega_c.z, nu, info.pos.x, info.pos.y,
								unit_b.x, unit_b.y, unit_b.z, unit_perp.x, unit_perp.y, unit_perp.z, unit_Hall.x, unit_Hall.y, unit_Hall.z);
							printf(
								"%d Pi_b_b %1.9E Pi_P_b %1.9E Pi_P_P %1.9E Pi_H_b %1.9E Pi_H_P %1.9E Pi_H_H %1.9E\n"
								"momflux b %1.9E perp %1.9E Hall %1.9E visc_contrib %1.9E %1.9E %1.9E \n"
								"---------------------\n",
								i, Pi_b_b, Pi_P_b, Pi_P_P, Pi_H_b, Pi_H_P, Pi_H_H,
								momflux_b, momflux_perp, momflux_Hall,
								visc_contrib.x, visc_contrib.y, visc_contrib.z);
						};
#endif
					};
				}; // was ita_par == 0

				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
				// Just leaving these but they won't do anything :
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;	
			}; // next i

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(p_MAR + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(f64_vec3));

			if (iSpecies == 1) {
				p_NT_addition_rate[iVertex].NiTi += visc_htg;
			} else {
				p_NT_addition_rate[iVertex].NeTe += visc_htg;
			}

#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iVertex %d NaN ownrates.x\n", iVertex);
			if (ownrates.y != ownrates.y)
				printf("iVertex %d NaN ownrates.y\n", iVertex);
			if (ownrates.z != ownrates.z)
				printf("iVertex %d NaN ownrates.z\n", iVertex);
			if (visc_htg != visc_htg) printf("iVertex %d NAN VISC HTG\n", iVertex);
#endif
		} else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memset(&ownrates_visc, 0, sizeof(f64_vec3));

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	visc_htg = 0.0;
#else
	f64 visc_htg0, visc_htg1, visc_htg2;
	visc_htg0 = 0.0; visc_htg1 = 0.0; visc_htg2 = 0.0;
#endif
	{
		long izNeighMinor[6];
		char szPBC[6];

	//	if (TESTVISC) printf("\niMinor = %d ; info.flag %d ; ita_par %1.9E \n\n", iMinor, info.flag, shared_ita_par[threadIdx.x]);

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
			
			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			f64_vec3 prev_v, opp_v, next_v;
			f64_vec2 prevpos, nextpos, opppos;

			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_v[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			} else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					v4 temp = p_vie_minor[izNeighMinor[iprev]];
					prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y; 
					if (iSpecies == 1) {
						prev_v.z = temp.viz;
					} else {
						prev_v.z = temp.vez;
					};
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_v[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_v_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					v4 temp = p_vie_minor[izNeighMinor[i]];
					opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						opp_v.z = temp.viz;
					} else {
						opp_v.z = temp.vez;
					};
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}
			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				if (TESTIONVISC) printf("start loop %d: ownrates.x %1.9E", i, ownrates_visc.x);
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_v[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				} else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_v_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						v4 temp = p_vie_minor[izNeighMinor[inext]];
						next_v.x = temp.vxy.x; next_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							next_v.z = temp.viz;
						} else {
							next_v.z = temp.vez;
						};
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				bool bUsableSide = true;
				{
					f64 nu_theirs, ita_theirs;
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
						ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];						
					} else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];	
						} else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							nu_theirs = p_nu_minor[izNeighMinor[i]];
							ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
						}
					}
					// GEOMETRIC ITA:
					if (ita_theirs == 0.0) bUsableSide = false;
					ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);
					if (shared_ita_par[threadIdx.x] * ita_theirs < 0.0) printf("iMinor %d Alert: %1.10E \n", iMinor, shared_ita_par[threadIdx.x] * ita_theirs);
					
						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// ins-ins triangle traffic:

				bool bLongi = false;

#ifdef INS_INS_NONE
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
#else
//#ifdef INS_INS_LONGI
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bLongi = true;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
				// Use this in this case as flag for reconstructing edge normal to where
				// it only stays within triangle!
//#else
				// do it later
//#endif
#endif

				f64_vec2 gradvx, gradvy, gradvz;
				f64_vec3 htg_diff;

				//if (0)//(iMinor == CHOSEN) || (izNeighMinor[i] == CHOSEN))
				//	printf("%d %d %d : using centroids at corners (why?) edge: %1.10E %1.10E , %1.10E %1.10E\n"
				//	"gradvz %1.10E %1.10E\n", iMinor, i, izNeighMinor[i],
				//	THIRD*(prevpos.x + opppos.x + info.pos.x), THIRD*(prevpos.y + opppos.y + info.pos.y),
				//	THIRD*(nextpos.x + opppos.x + info.pos.x), THIRD*(nextpos.y + opppos.y + info.pos.y));

				if (bUsableSide)
				{
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false) 
					{
						gradvx = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v[threadIdx.x].x, next_v.x, opp_v.x
						);
						gradvy = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v[threadIdx.x].y, next_v.y, opp_v.y
						);

						gradvz = GetGradient_3Point(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							shared_v[threadIdx.x].z, next_v.z, opp_v.z
						);
						


					} else {
						if (TestDomainPos(nextpos) == false)
						{
							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v[threadIdx.x].x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v[threadIdx.x].y, opp_v.y
							);
							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v[threadIdx.x].z, opp_v.z
							);
							
						} else {

							if (0) //(iMinor == CHOSEN) && (i == 0)) 
							{
								gradvx = GetGradientDebug(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x
								);
							} else {
								gradvx = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x
								);
							};
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
							);							
						};
					};
#else
					
					
					if (bLongi)
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						gradvx = (opp_v.x - shared_v[threadIdx.x].x)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						gradvy = (opp_v.y - shared_v[threadIdx.x].y)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						gradvz = (opp_v.z - shared_v[threadIdx.x].z)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);

					} else {
						gradvx = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x
						);
						gradvy = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y
						);
						gradvz = GetGradient(
							//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
							prevpos, info.pos, nextpos, opppos,
							//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
							prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
						);
					}	
					if (TEST_EPSILON_X_MINOR) printf("%d %d %d : vx prev own anti opp %1.10E %1.10E %1.10E %1.10E\n"
						"gradvx %1.10E %1.10E\n", iMinor, i, izNeighMinor[i],
						prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x, gradvx.x, gradvx.y);

#endif

#if TEST_VISC_MINOR
					if ((iMinor == CHOSEN) && (iSpecies == 2))

						printf("%d %d %d "
						"prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n"
						"gradvx %1.12E %1.12E vx psno %1.10E %1.10E %1.10E %1.10E\n"
						"gradvy %1.12E %1.12E vy psno %1.10E %1.10E %1.10E %1.10E\n"
						"gradvz %1.12E %1.12E vz psno %1.10E %1.10E %1.10E %1.10E\n",
						iMinor, i, izNeighMinor[i],
						prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
						gradvx.x, gradvx.y, prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
						gradvy.x, gradvy.y, prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
						gradvz.x, gradvz.y, prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
					);
#endif
				
#ifdef INS_INS_NONE
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {
							// just set it to 0.
							bUsableSide = false;
							gradvz.x = 0.0;
							gradvz.y = 0.0;
							gradvx.x = 0.0;
							gradvx.y = 0.0;
							gradvy.x = 0.0;
							gradvy.y = 0.0;
						};
					};
#endif

					htg_diff = shared_v[threadIdx.x] - opp_v;
					
				} else {
					//if (TESTIONVISC) printf("side not usable: %d", i);
				};

				if (bUsableSide) {
					if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
					{
						// run unmagnetised case
						f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy;// Pi_zx, Pi_zy;

						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
						Pi_yx = -ita_par*(gradvx.y + gradvy.x);
						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
						//Pi_zx = -ita_par*(gradvz.x);
						//Pi_zy = -ita_par*(gradvz.y);
						
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  
												
						if (bLongi) {
							// move any edge_normal endpoints that are below the insulator,
							// until they are above the insulator.
							edge_normal = ReconstructEdgeNormal(
								prevpos, info.pos, nextpos, opppos);
						};

						f64_vec3 visc_contrib;
						visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						visc_contrib.z = over_m_s*ita_par*(gradvz.x*edge_normal.x + gradvz.y*edge_normal.y);

						// With ita_par it gives the 6 result. With Pi it reverts to the 5 result. Unless we printf.

						// Now trying this vers without optimization: 
						// 6e-5, 2.36e10
						// Now try with Pi and no opti.
						
						// same
						// Now turn it back on:


						ownrates_visc += visc_contrib;
#if TEST_VISC_MINOR
						if ((iMinor == CHOSEN) && (iSpecies == 2))
							printf("%d %d %d unmag visc_contrib.z %1.10E Pi_zx %1.10E Pi_zy %1.10E edgenml %1.8E %1.8E\n",
								iMinor, i, izNeighMinor[i], visc_contrib.z, -ita_par*gradvz.x*edge_normal.x, -ita_par*gradvz.y*edge_normal.y, edge_normal.x, edge_normal.y);
															// This is the guilty one -- on its own it makes the difference.

															// Let's see what happens if we comment part of it out.
															// outcome as if do nothing.

															// OK, try adding in this:
//						if (iMinor == CHOSEN) printf("visc_contrib.z %1.10E Pi_zx %1.10E Pi_zy %1.10E\n",
	//						visc_contrib.z, Pi_zx, Pi_zy);
						// yes, changed.

						// Now moved it to after 'ownrates_visc += visc_contrib'
						// can't see what diff this should make. <-- Apparently none
						
						//if (iMinor == CHOSEN) printf("Pi_zx %1.10E Pi_zy %1.10E\n",	Pi_zx, Pi_zy);
						// yes, still changed.

						// Try this:
						//printf("ssssssssjjjjjjjj opppos %1.10E %1.10E \n", opppos.x, opppos.y);
						// results as without.
						
						// Results as without:
						//if (iMinor == CHOSEN) printf("ssssssssjjjjjjjj visc_contrib.z %1.10E opppos %1.10E %1.10E \n", visc_contrib.z, opppos.x, opppos.y);
						
						// Results in the changed output:
						//if (iMinor == CHOSEN) printf("Pi_zx %1.10E \n", Pi_zx);

						// So it must be optimizing out the variable?

						// Now try editing it out myself.
						// Result is still bloody changed!

						// Now move variables down here:
						//
						//if (0){// iMinor == CHOSEN) {
						//	f64 Pi_zx, Pi_zy;
						//	Pi_zx = -ita_par*(gradvz.x);
						//	Pi_zy = -ita_par*(gradvz.y);
						//	printf("Pi_zx %1.10E \n", Pi_zx);
						//};
						// yes still changed.
						// Now try removing this.
						// yes still changed.

						// Now try changing it back to Pi_zx,zy above.
						// Yes, reverted.
#endif
						if (i % 2 == 0) {
							// vertex : heat collected by vertex
							// those all summed to +, for now...
							
					//		if ((iMinor == CHOSEN) && (iSpecies == 1)) printf("Htg %d i %d %d\n",
					//			iMinor, i, izNeighMinor[i]);

						} else {

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES

							visc_htg += -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
#else
							f64 htg_addn = -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
							if (i == 1) // opposite 2
							{
								visc_htg0 += 0.5*htg_addn;
								visc_htg1 += 0.5*htg_addn;
							}
							else {
								if (i == 3) {
									visc_htg1 += 0.5*htg_addn;
									visc_htg2 += 0.5*htg_addn;
								}
								else {
									// i == 5:
									visc_htg0 += 0.5*htg_addn;
									visc_htg2 += 0.5*htg_addn;
								};
							};
#endif

							//if (HTGPRINT2) printf("Htg %d i %d %d : %1.9E htg_diff %1.7E %1.7E %1.7E vctb %1.7E %1.7E %1.7E\n",
							//	iMinor, i, izNeighMinor[i], -THIRD*(m_s)*(htg_diff.dot(visc_contrib)),
							//	htg_diff.x, htg_diff.y, htg_diff.z, visc_contrib.x, visc_contrib.y, visc_contrib.z);

							//if (0)//((iSpecies == 1) && (-THIRD*(m_s)*(htg_diff.dot(visc_contrib)) < -1.0))
							//{
							//	printf("COOLING: %d %d %d visc_htg += %1.9E htg_diff %1.8E %1.8E %1.8E vctb %1.8E %1.8E %1.8E\n",
							//		iMinor, i, izNeighMinor[i], -THIRD*(m_s)*(htg_diff.dot(visc_contrib)),
							//		htg_diff.x, htg_diff.y, htg_diff.z,
							//		visc_contrib.x, visc_contrib.y, visc_contrib.z);
							//};
						};

						// So we are saying if edge_normal.x > 0 and gradviz.x > 0
						// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					} else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

							if (bLongi) {
								// move any edge_normal endpoints that are below the insulator,
								// until they are above the insulator.
								edge_normal = ReconstructEdgeNormal(
									prevpos, info.pos, nextpos, opppos
								);
							};
							f64 omegasq = omega_c.dot(omega_c);
							omegamod = sqrt(omegasq);
							unit_b = omega_c / omegamod;
							unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
							unit_perp = unit_perp / unit_perp.modulus();
							unit_Hall = unit_b.cross(unit_perp); // Note sign.
																 // store omegamod instead.
						}
						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
							{
								f64_vec3 intermed;

								// use: d vb / da = b transpose [ dvi/dxj ] a
								// Prototypical element: a.x b.y dvy/dx
								// b.x a.y dvx/dy

								intermed.x = unit_b.dotxy(gradvx);
								intermed.y = unit_b.dotxy(gradvy);
								intermed.z = unit_b.dotxy(gradvz);
								{
									f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

									dvb_by_db = unit_b.dot(intermed);
									dvperp_by_db = unit_perp.dot(intermed);
									dvHall_by_db = unit_Hall.dot(intermed);

									W_bb += 4.0*THIRD*dvb_by_db;
									W_bP += dvperp_by_db;
									W_bH += dvHall_by_db;
									W_PP -= 2.0*THIRD*dvb_by_db;
									W_HH -= 2.0*THIRD*dvb_by_db;
#if TEST_VISC_MINOR
									if (0)//iMinor == CHOSEN) 
										printf("%d %d : dvb/db %1.10E dvP/db %1.10E dvH/db %1.10E --> \n"
										"Wbb %1.9E WbP %1.9E WbH %1.9E WPP %1.9E WHH %1.9E \n",
										iMinor, i, dvb_by_db, dvperp_by_db, dvHall_by_db, W_bb, W_bP, W_bH, W_PP, W_HH);
#endif
								}
								{
									f64 dvb_by_dperp, dvperp_by_dperp,
										dvHall_by_dperp;
									// Optimize by getting rid of different labels.

									intermed.x = unit_perp.dotxy(gradvx);
									intermed.y = unit_perp.dotxy(gradvy);
									intermed.z = unit_perp.dotxy(gradvz);

									dvb_by_dperp = unit_b.dot(intermed);
									dvperp_by_dperp = unit_perp.dot(intermed);
									dvHall_by_dperp = unit_Hall.dot(intermed);

									W_bb -= 2.0*THIRD*dvperp_by_dperp;
									W_PP += 4.0*THIRD*dvperp_by_dperp;
									W_HH -= 2.0*THIRD*dvperp_by_dperp;
									W_bP += dvb_by_dperp;
									W_PH += dvHall_by_dperp;

#if TEST_VISC_MINOR
									if (0)//((iMinor == CHOSEN) && (iSpecies == 2))
										printf("%d %d : dvb/dP %1.10E dvP/dP %1.10E dvH/dP %1.10E --> \n"
										"Wbb %1.9E WbP %1.9E WPP %1.9E WHH %1.9E WPH %1.9E\n",
										iMinor, i, dvb_by_dperp, dvperp_by_dperp, dvHall_by_dperp, 
										W_bb, W_bP, W_PP, W_HH, W_PH);
#endif
								}
								{
									f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

									intermed.x = unit_Hall.dotxy(gradvx);
									intermed.y = unit_Hall.dotxy(gradvy);
									intermed.z = unit_Hall.dotxy(gradvz);

									dvb_by_dHall = unit_b.dot(intermed);
									dvperp_by_dHall = unit_perp.dot(intermed);
									dvHall_by_dHall = unit_Hall.dot(intermed);

									W_bb -= 2.0*THIRD*dvHall_by_dHall;
									W_PP -= 2.0*THIRD*dvHall_by_dHall;
									W_HH += 4.0*THIRD*dvHall_by_dHall;
									W_bH += dvb_by_dHall;
									W_PH += dvperp_by_dHall;

								}
							}						
							
							f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
							Pi_b_b += -ita_par*W_bb;
							Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
							Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
							Pi_H_P += -ita_1*W_PH;							
							
							f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_2*W_bP;
							Pi_H_b += -ita_2*W_bH;
														
							f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
							Pi_P_P -= ita_3*W_PH;
							Pi_H_H += ita_3*W_PH;
							Pi_H_P += 0.5*ita_3*(W_PP - W_HH);							
							
							f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_4*W_bH;
							Pi_H_b += ita_4*W_bP;							

#if TEST_VISC_deep
								printf(
									"%d %d : ita_par %1.9E ita1 %1.9E ita2 %1.9E ita3 %1.9E ita4 %1.9E\n"
									"W_bb %1.12E W_bP %1.12E W_PP %1.9E W_HH %1.9E W_bH %1.12E W_PH %1.12E\n"
									"Pi bb %1.9E PP %1.9E Pb %1.9E HH %1.9E Hb %1.9E HP %1.9E\n",
									iMinor, i,
									ita_par, ita_1, ita_2, ita_3, ita_4,
									W_bb, W_bP, W_PP, W_HH, W_bH, W_PH,
									Pi_b_b, Pi_P_P, Pi_P_b, Pi_H_H, Pi_H_b, Pi_H_P);
#endif
						}
						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;							
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

							if (bLongi) {
								edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
							};

							// Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);

#if TEST_VISC_MINOR
							if (0) printf("%d %d %d mag_edge.bPH %1.8E %1.8E %1.8E momflux.bPH %1.9E %1.9E %1.9E\n",
								iMinor, i, izNeighMinor[i], mag_edge.x, mag_edge.y, mag_edge.z,
								momflux_b, momflux_perp, momflux_Hall);
#endif
						}

						// Time to double-check carefully the signs.
						// Pi was defined with - on dv/dx and we then dot that with the edge_normal, so giving + if we are higher than outside.

						// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
						// is the flow of p_x dotted with the edge_normal
						// ownrates will be divided by N to give dv/dt
						// m N dvx/dt = integral div momflux_x
						// Therefore divide here just by m
						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
						visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
						visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

#if TEST_VISC_MINOR
						if ((iMinor == CHOSEN) && (iSpecies == 2))
							printf("%d %d : mag visc_ctb xyz %1.10E %1.10E %1.10E | unit_b %1.9E %1.9E unit_P %1.9E %1.9E\n"
								"=================================+++++++++++++++\n",
							iMinor, i, visc_contrib.x, visc_contrib.y, visc_contrib.z, unit_b.x, unit_b.y, unit_perp.x, unit_perp.y);
#endif
						
						ownrates_visc += visc_contrib;
						if (i % 2 != 0) // not vertex
						{

							// Label visc_htg for the corner OPPOSITE

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES

							visc_htg += -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
							if (TESTIONVISC)
								printf("%d %d visc_htg %1.10E\n", iMinor, i, -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib)));

#else
							f64 htg_addn = -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));

							if (i == 1) // opposite 2
							{
								visc_htg0 += 0.5*htg_addn;
								visc_htg1 += 0.5*htg_addn;
							} else {
								if (i == 3) {
									visc_htg1 += 0.5*htg_addn;
									visc_htg2 += 0.5*htg_addn;
								} else {
									// i == 5:
									visc_htg0 += 0.5*htg_addn;
									visc_htg2 += 0.5*htg_addn;
								};
							}
#endif

						///if (HTGPRINT2) printf("Htg %d i %d %d : %1.9E htg_diff %1.7E %1.7E %1.7E vctb %1.7E %1.7E %1.7E\n",
						//		iMinor, i, izNeighMinor[i], -THIRD*(m_s)*(htg_diff.dot(visc_contrib)),
						//		htg_diff.x, htg_diff.y, htg_diff.z, visc_contrib.x, visc_contrib.y, visc_contrib.z);

							//if (0) // (iSpecies == 1) && (-THIRD*(m_s)*(htg_diff.dot(visc_contrib)) < -1.0))
							//{
							//	printf("COOLING: %d %d %d visc_htg += %1.9E htg_diff %1.8E %1.8E %1.8E vctb %1.8E %1.8E %1.8E\n",
							//		iMinor, i, izNeighMinor[i], -THIRD*(m_s)*(htg_diff.dot(visc_contrib)),
							//		htg_diff.x, htg_diff.y, htg_diff.z,
							//		visc_contrib.x, visc_contrib.y, visc_contrib.z);
							//};
						} else {

							//if (HTGPRINT2) printf("Htg %d i %d %d\n",
							//	iMinor, i, izNeighMinor[i]);

						};
					};
				}; // bUsableSide

				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR[iMinor]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(&(p_MAR[iMinor]), &(ownrates), sizeof(f64_vec3));

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
			if (iSpecies == 1) {
				p_NT_addition_tri[iMinor].NiTi += visc_htg;
			} else {
				p_NT_addition_tri[iMinor].NeTe += visc_htg;
			}
#else
		
			if (iSpecies == 1) {
				p_NT_addition_tri[iMinor * 3 + 0].NiTi += visc_htg0;
				p_NT_addition_tri[iMinor * 3 + 1].NiTi += visc_htg1;
				p_NT_addition_tri[iMinor * 3 + 2].NiTi += visc_htg2;

			} else {
				p_NT_addition_tri[iMinor * 3 + 0].NeTe += visc_htg0;
				p_NT_addition_tri[iMinor * 3 + 1].NeTe += visc_htg1;
				p_NT_addition_tri[iMinor * 3 + 2].NeTe += visc_htg2;
			}

#endif

			// We will have to round this up into the vertex heat afterwards.


#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iMinor %d NaN ownrates.x\n", iMinor);
			if (ownrates.y != ownrates.y)
				printf("iMinor %d NaN ownrates.y\n", iMinor);
			if (ownrates.z != ownrates.z)
				printf("iMinor %d NaN ownrates.z\n", iMinor);

			if (visc_htg != visc_htg) printf("iMinor %d NAN VISC HTG\n", iMinor);
#endif

			// We do best by taking each boundary, considering how
			// much heat to add for each one.

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope
}


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
	) 
{
	// Purpose of routine
	// Be like the visc contrib routine but only apply contribs if we are at p_Select = 1 and
	// only if the neigh is at p_Select = 0.


	__shared__ f64_vec3 shared_v[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	__shared__ f64_vec3 shared_v_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	{
		v4 temp = p_vie_minor[iMinor];
		shared_v[threadIdx.x].x = temp.vxy.x;
		shared_v[threadIdx.x].y = temp.vxy.y;
		if (iSpecies == 1) {
			shared_v[threadIdx.x].z = temp.viz;
		}
		else {
			shared_v[threadIdx.x].z = temp.vez;
		};
	}
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	if (0)//iMinor == lChosen)
		printf("\n\n iMinor %d B %1.10E %1.10E v %1.10E %1.10E %1.10E nu %1.9E \n\n\n",
			iMinor, shared_B[threadIdx.x].x, shared_B[threadIdx.x].y,
			shared_v[threadIdx.x].x, shared_v[threadIdx.x].y, shared_v[threadIdx.x].z,
			shared_nu[threadIdx.x]);

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			v4 temp;
			memcpy(&temp, &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_v_verts[threadIdx.x].x = temp.vxy.x;
			shared_v_verts[threadIdx.x].y = temp.vxy.y;
			if (iSpecies == 1) {
				shared_v_verts[threadIdx.x].z = temp.viz;
			} else {
				shared_v_verts[threadIdx.x].z = temp.vez;
			};
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		} else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_v_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&ownrates_visc, 0, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0)
			&& (p_Select[iVertex + BEGINNING_OF_CENTRAL] != 0)
			)
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_vec3 opp_v, prev_v, next_v;
			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_v[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				v4 temp = p_vie_minor[izTri[iprev]];
				prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y;
				if (iSpecies == 1) { prev_v.z = temp.viz; }
				else { prev_v.z = temp.vez; };
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);// = Clockwise3_d*prev_v;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_v[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				v4 temp = p_vie_minor[izTri[i]];
				opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y;
				if (iSpecies == 1) { opp_v.z = temp.viz; }
				else { opp_v.z = temp.vez; };
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}


#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext == tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

				f64_vec2 gradvx, gradvy, gradvz;
				f64_vec3 htg_diff;

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_v[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
					//if ((TEST_VISC_VERT)) printf("inext %d  izTri[inext] %d tri_len %d nextpos %1.9E %1.9E\n", inext, izTri[inext], tri_len,
					//	nextpos.x, nextpos.y);
				}
				else {
					v4 temp = p_vie_minor[izTri[inext]];
					next_v.x = temp.vxy.x; next_v.y = temp.vxy.y;
					if (iSpecies == 1) { next_v.z = temp.viz; }
					else { next_v.z = temp.vez; };
					nextpos = p_info_minor[izTri[inext]].pos;
					//if ((TEST_VISC_VERT)) printf("inext %d II izTri[inext] %d tri_len %d\n", inext, izTri[inext], tri_len);
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				if (p_Select[izTri[i]] == 0) {

					f64_vec3 omega_c;
					{
						f64_vec2 opp_B;
						f64 ita_theirs, nu_theirs;
						if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
						{
							opp_B = shared_B[izTri[i] - StartMinor];
							ita_theirs = shared_ita_par[izTri[i] - StartMinor];
							nu_theirs = shared_nu[izTri[i] - StartMinor];
						}
						else {
							opp_B = p_B_minor[izTri[i]].xypart();
							ita_theirs = p_ita_parallel_minor[izTri[i]];
							nu_theirs = p_nu_minor[izTri[i]];
						};
						// GEOMETRIC ITA:
						ita_par = sqrt(ita_theirs*shared_ita_par_verts[threadIdx.x]);

						if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opp_B = Clockwise_d*opp_B;
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opp_B = Anticlockwise_d*opp_B;
						}
						nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
						omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
						if (iSpecies == 1) omega_c *= qoverMc;
						if (iSpecies == 2) omega_c *= qovermc;
					} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

					bool bLongi = false;

					if (ita_par > 0.0)
					{

#ifdef INS_INS_3POINT

						if (TestDomainPos(prevpos) == false) {

						//	if (TEST_VISC_VERT)
						//		printf("(TestDomainPos(prevpos) == false) vx ours next opp %1.10E %1.10E %1.10E\n",
						//			shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);

							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
							);

						} else {
							if (TestDomainPos(nextpos) == false) {

						//		if (TEST_VISC_VERT)
						//			printf("(TestDomainPos(nextpos) == false) vx prev ours opp %1.10E %1.10E %1.10E\n",
						//				prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x);

								gradvx = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x
								);
								gradvy = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_verts[threadIdx.x].y, opp_v.y
								);
								gradvz = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_verts[threadIdx.x].z, opp_v.z
								);
							} else {

						//		if (TEST_VISC_VERT)
						//			printf("standard edge ; vx prev ours next opp %1.10E %1.10E %1.10E %1.10E\n",
						//				prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);

								gradvx = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
								);
								gradvy = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
								);
								gradvz = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
								);
							};
						};

						//if (TEST_VISC_VERT) printf("%d %d %d vy psno %1.12E %1.12E %1.12E %1.12E gradvy %1.12E %1.12E\n"
						//	"prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n"
						//	"gradvx %1.12E %1.12E gradvz %1.12E %1.12E vz %1.8E %1.8E %1.8E %1.8E\n"
						//	"vx %1.10E %1.10E %1.10E %1.10E \n",

						//	iVertex, i, izTri[i],
						//	prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y, gradvy.x, gradvy.y,
						//	prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
						//	gradvx.x, gradvx.y, gradvz.x, gradvz.y,
						//	prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z,
						//	prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
						//);

						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false)) bLongi = true;
#else
						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						{
							// One of the sides is dipped under the insulator -- set transverse deriv to 0.
							// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

							gradvx = (opp_v.x - shared_v_verts[threadIdx.x].x)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvy = (opp_v.y - shared_v_verts[threadIdx.x].y)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvz = (opp_v.z - shared_v_verts[threadIdx.x].z)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);

						}
						else {

							if (TESTIONVERTVISC) {
								printf("%d i inext %d %d izTri %d %d ourpos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E opp %1.8E %1.8E \n",
									iVertex, i, inext, izTri[i], izTri[inext],
									info.pos.x, info.pos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y, opppos.x, opppos.y);
							}

							gradvx = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z
							);
							// Could switch to the 3 in one function that handles all 3. in one.
					}
						// No Area_hex gathered for us.
#endif

					//					if (TESTIONVERTVISC) printf(
					//						"iVertex %d  \n"
					//						"our_v.x next prev opp %1.8E %1.8E %1.8E %1.8E gradvx %1.8E %1.8E\n"
					//						"our_v.y next prev opp %1.8E %1.8E %1.8E %1.8E gradvy %1.8E %1.8E\n"
					//						"our_v.z next prev opp %1.8E %1.8E %1.8E %1.8E gradvz %1.8E %1.8E\n"
					//						"info.pos %1.8E %1.8E opppos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E\n",
					//						iVertex,
					//						shared_vie_verts[threadIdx.x].vxy.x, next_v.vxy.x, prev_v.vxy.x, opp_v.vxy.x,
					//						gradvx.x, gradvx.y,
					//						shared_vie_verts[threadIdx.x].vxy.y, next_v.vxy.y, prev_v.vxy.y, opp_v.vxy.y,
					//						gradvy.x, gradvy.y,
					//						shared_vie_verts[threadIdx.x].viz, next_v.viz, prev_v.viz, opp_v.viz,
					//						gradvz.x, gradvz.y,
					//						info.pos.x, info.pos.y, opppos.x, opppos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y);

						htg_diff = shared_v_verts[threadIdx.x] - opp_v;
					}

					
					if (ita_par > 0.0) {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

							Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
							Pi_xy = -ita_par*(gradvx.y + gradvy.x);
							Pi_yx = Pi_xy;
							Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
							Pi_zx = -ita_par*(gradvz.x);
							Pi_zy = -ita_par*(gradvz.y);

							f64_vec3 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);

							if (bLongi) edge_normal = ReconstructEdgeNormal(
								prevpos, info.pos, nextpos, opppos
							);

							visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
							visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

							ownrates_visc += visc_contrib;
							visc_htg += -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib));

							// So we are saying if edge_normal.x > 0 and gradviz.x > 0
							// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
						}
						else {

							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);

									if (bLongi) edge_normal = ReconstructEdgeNormal(
										prevpos, info.pos, nextpos, opppos
									);
									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.

																		 // Since B is in the plane, it's saying we picked perp in the plane, H = z.
																		 // CHECK THAT
																		 // It probably doesn't matter what direction perp is. As long as it is in the plane normal to b.
																		 // Formulary tells us if b is z then perp is x, H is y. So we can freely rotate x,y in the normal plane
																		 // and this will still hold.
																		 // Let's verify if we got it : 
																		 // z cross x = (0, 1, 0) --- tick.

																		 // From the perspective of the +- ita3 and ita4, all that matter is the relative orientation of P and H.
								}
								{
									f64_vec3 intermed;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									intermed.x = unit_b.dotxy(gradvx);
									intermed.y = unit_b.dotxy(gradvy);
									intermed.z = unit_b.dotxy(gradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.dot(intermed);
										dvperp_by_db = unit_perp.dot(intermed);
										dvHall_by_db = unit_Hall.dot(intermed);

										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;

									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermed.x = unit_perp.dotxy(gradvx);
										intermed.y = unit_perp.dotxy(gradvy);
										intermed.z = unit_perp.dotxy(gradvz);

										dvb_by_dperp = unit_b.dot(intermed);
										dvperp_by_dperp = unit_perp.dot(intermed);
										dvHall_by_dperp = unit_Hall.dot(intermed);

										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;

									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;
										// basically, all should be 0

										intermed.x = unit_Hall.dotxy(gradvx);
										intermed.y = unit_Hall.dotxy(gradvy);
										intermed.z = unit_Hall.dotxy(gradvz);

										dvb_by_dHall = unit_b.dot(intermed);
										dvperp_by_dHall = unit_perp.dot(intermed);
										dvHall_by_dHall = unit_Hall.dot(intermed);

										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;
										
									}
								}
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

									Pi_b_b += -ita_par*W_bb;
									Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									Pi_H_P += -ita_1*W_PH;
									// W_HH = 0
									
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_2*W_bP;
									Pi_H_b += -ita_2*W_bH;
									
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									Pi_P_P -= ita_3*W_PH;
									Pi_H_H += ita_3*W_PH;
									Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
									
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_4*W_bH;															
									Pi_H_b += ita_4*W_bP;
								}
							} // scope W

							  // All we want left over at this point is Pi .. and unit_b

							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);

								if (bLongi) edge_normal = ReconstructEdgeNormal(
									prevpos, info.pos, nextpos, opppos
								);
								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
							}

							// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
							// is the flow of p_x dotted with the edge_normal
							// ownrates will be divided by N to give dv/dt
							// m N dvx/dt = integral div momflux_x
							// Therefore divide here just by m

							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

							ownrates_visc += visc_contrib;
							visc_htg += -TWOTHIRDS*(m_s)*(htg_diff.dot(visc_contrib));
//
//							if (TEST_VISC_VERT)
//							{
//								printf("iVertex %d %d tri %d species %d ita_par %1.9E \n"
//									"omega %1.9E %1.9E %1.9E nu %1.11E ourpos %1.8E %1.8E \n"
//									"unit_b %1.11E %1.11E %1.11E unit_perp %1.11E %1.11E %1.11E unit_Hall %1.8E %1.8E %1.9E\n",
//									iVertex, i, izTri[i], iSpecies, ita_par,
//									omega_c.x, omega_c.y, omega_c.z, nu, info.pos.x, info.pos.y,
//									unit_b.x, unit_b.y, unit_b.z, unit_perp.x, unit_perp.y, unit_perp.z, unit_Hall.x, unit_Hall.y, unit_Hall.z);
//								printf(
//									"%d Pi_b_b %1.9E Pi_P_b %1.9E Pi_P_P %1.9E Pi_H_b %1.9E Pi_H_P %1.9E Pi_H_H %1.9E\n"
//									"momflux b %1.9E perp %1.9E Hall %1.9E visc_contrib %1.9E %1.9E %1.9E \n"
//									"---------------------\n",
//									i, Pi_b_b, Pi_P_b, Pi_P_P, Pi_H_b, Pi_H_P, Pi_H_H,
//									momflux_b, momflux_perp, momflux_Hall,
//									visc_contrib.x, visc_contrib.y, visc_contrib.z);
//							};
						};
					}; // was ita_par == 0
				}; // was it p_Select == 0 

				   // v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
				   // Just leaving these but they won't do anything :
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			//	if (TEST) 
			//		printf("%d ion ownrates %1.8E %1.8E %1.8E ownrates_visc %1.8E %1.8E %1.8E our_v %1.8E %1.8E %1.8E\n",
			//		iVertex, ownrates.x, ownrates.y, ownrates.z, ownrates_visc.x, ownrates_visc.y, ownrates_visc.z, our_v.vxy.x, our_v.vxy.y, our_v.viz);
			ownrates += ownrates_visc;
			memcpy(p_MAR + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(f64_vec3));

			if (iSpecies == 1) {
				p_NT_addition_rate[iVertex].NiTi += visc_htg;
			} else {
				p_NT_addition_rate[iVertex].NeTe += visc_htg;
			}

#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iVertex %d NaN ownrates.x\n", iVertex);
			if (ownrates.y != ownrates.y)
				printf("iVertex %d NaN ownrates.y\n", iVertex);
			if (ownrates.z != ownrates.z)
				printf("iVertex %d NaN ownrates.z\n", iVertex);
			if (visc_htg != visc_htg) printf("iVertex %d NAN VISC HTG\n", iVertex);
#endif
		}
		else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memset(&ownrates_visc, 0, sizeof(f64_vec3));
#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	visc_htg = 0.0;
#else
	f64 visc_htg0, visc_htg1, visc_htg2;
	visc_htg0 = 0.0; visc_htg1 = 0.0; visc_htg2 = 0.0;
#endif
	{
		long izNeighMinor[6];
		char szPBC[6];

		//if (TESTVISC) printf("\niMinor = %d ; info.flag %d ; ita_par %1.9E \n\n", iMinor, info.flag, shared_ita_par[threadIdx.x]);

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)
			&& (p_Select[iMinor] != 0)
			) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			f64_vec3 prev_v, opp_v, next_v;
			f64_vec2 prevpos, nextpos, opppos;

			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_v[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					v4 temp = p_vie_minor[izNeighMinor[iprev]];
					prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						prev_v.z = temp.viz;
					}
					else {
						prev_v.z = temp.vez;
					};
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_v[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_v_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					v4 temp = p_vie_minor[izNeighMinor[i]];
					opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						opp_v.z = temp.viz;
					}
					else {
						opp_v.z = temp.vez;
					};
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}
			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				//if (TESTIONVISC) printf("start loop %d: ownrates.x %1.9E", i, ownrates_visc.x);
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_v[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_v_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						v4 temp = p_vie_minor[izNeighMinor[inext]];
						next_v.x = temp.vxy.x; next_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							next_v.z = temp.viz;
						}
						else {
							next_v.z = temp.vez;
						};
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				if (p_Select[izNeighMinor[i]] == 0) {

					bool bUsableSide = true;
					{
						f64 nu_theirs, ita_theirs;
						f64_vec2 opp_B(0.0, 0.0);
						// newly uncommented:
						if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
						{
							opp_B = shared_B[izNeighMinor[i] - StartMinor];
							nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
							ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];
						}
						else {
							if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							}
							else {
								opp_B = p_B_minor[izNeighMinor[i]].xypart();
								nu_theirs = p_nu_minor[izNeighMinor[i]];
								ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
							}
						}
						// GEOMETRIC ITA:
						if (ita_theirs == 0.0) bUsableSide = false;
						ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);

						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opp_B = Clockwise_d*opp_B;
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opp_B = Anticlockwise_d*opp_B;
						}
						nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
						omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
						if (iSpecies == 1) omega_c *= qoverMc;
						if (iSpecies == 2) omega_c *= qovermc;
					};

					// ins-ins triangle traffic:

					bool bLongi = false;

#ifdef INS_INS_NONE
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS)
							bUsableSide = 0;
					}
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						bLongi = true;
#else
					//#ifdef INS_INS_LONGI
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS)
							bLongi = true;
					}
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						bLongi = true;
					// Use this in this case as flag for reconstructing edge normal to where
					// it only stays within triangle!
					//#else
					// do it later
					//#endif
#endif

					f64_vec2 gradvx, gradvy, gradvz;
					f64_vec3 htg_diff;
									
					if (bUsableSide)
					{
#ifdef INS_INS_3POINT
						if (TestDomainPos(prevpos) == false)
						{
							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v[threadIdx.x].y, next_v.y, opp_v.y
							);

							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v[threadIdx.x].z, next_v.z, opp_v.z
							);
						}
						else {
							if (TestDomainPos(nextpos) == false)
							{
								gradvx = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v[threadIdx.x].x, opp_v.x
								);
								gradvy = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v[threadIdx.x].y, opp_v.y
								);
								gradvz = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v[threadIdx.x].z, opp_v.z
								);
								
							}
							else {

								gradvx = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x
								);
								gradvy = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y
								);
								gradvz = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
								);
								
							};
						};


						//if (TEST_VISC) printf("%d %d %d prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n"
						//	"gradvz %1.12E %1.12E vz psno %1.10E %1.10E %1.10E %1.10E\n"
						//	"gradvy %1.12E %1.12E vy psno %1.10E %1.10E %1.10E %1.10E\n"
						//	"gradvx %1.12E %1.12E vx psno %1.10E %1.10E %1.10E %1.10E\n",
						//	iMinor, i, izNeighMinor[i],
						//	prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
						//	gradvx.x, gradvx.y, prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
						//	gradvy.x, gradvy.y, prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
						//	gradvz.x, gradvz.y, prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
						//);
#else


						if (bLongi)
						{
							// One of the sides is dipped under the insulator -- set transverse deriv to 0.
							// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

							gradvx = (opp_v.x - shared_v[threadIdx.x].x)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvy = (opp_v.y - shared_v[threadIdx.x].y)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvz = (opp_v.z - shared_v[threadIdx.x].z)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);

						}
						else {
							gradvx = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z
							);
						}
						if (TEST_EPSILON_X_MINOR) printf("%d %d %d : vx prev own anti opp %1.10E %1.10E %1.10E %1.10E\n"
							"gradvx %1.10E %1.10E\n", iMinor, i, izNeighMinor[i],
							prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x, gradvx.x, gradvx.y);

#endif


#ifdef INS_INS_NONE
						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if (flag == CROSSING_INS) {
								// just set it to 0.
								bUsableSide = false;
								gradvz.x = 0.0;
								gradvz.y = 0.0;
								gradvx.x = 0.0;
								gradvx.y = 0.0;
								gradvy.x = 0.0;
								gradvy.y = 0.0;
						};
						};
#endif

						htg_diff = shared_v[threadIdx.x] - opp_v;

					}
					else {
						//if (TESTIONVISC) printf("side not usable: %d", i);
					};

					if (bUsableSide) {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

							Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
							Pi_xy = -ita_par*(gradvx.y + gradvy.x);
							Pi_yx = Pi_xy;
							Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
							Pi_zx = -ita_par*(gradvz.x);
							Pi_zy = -ita_par*(gradvz.y);

							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);

							if (bLongi) {
								// move any edge_normal endpoints that are below the insulator,
								// until they are above the insulator.
								edge_normal = ReconstructEdgeNormal(
									prevpos, info.pos, nextpos, opppos
								);
							};

							f64_vec3 visc_contrib;
							visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
							visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						//	if (TEST_EPSILON_X_MINOR)
						//		printf("%d %d %d unmag visc_contrib.z %1.10E Pi_zx %1.10E Pi_zy %1.10E edgenml %1.8E %1.8E\n",
						//			iMinor, i, izNeighMinor[i], visc_contrib.x, Pi_xx, Pi_xy, edge_normal.x, edge_normal.y);

							ownrates_visc += visc_contrib;

							if (i % 2 == 0) {
								// vertex : heat collected by vertex
							}
							else {

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES

								visc_htg += -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
#else
								f64 htg_addn = -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
								if (i == 1) // opposite 2
								{
									visc_htg0 += 0.5*htg_addn;
									visc_htg1 += 0.5*htg_addn;
								}
								else {
									if (i == 3) {
										visc_htg1 += 0.5*htg_addn;
										visc_htg2 += 0.5*htg_addn;
									}
									else {
										// i == 5:
										visc_htg0 += 0.5*htg_addn;
										visc_htg2 += 0.5*htg_addn;
									};
								};
#endif
							}
							// So we are saying if edge_normal.x > 0 and gradviz.x > 0
							// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					}
						else {
							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 omegamod;
							{
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

								if (bLongi) {
									// move any edge_normal endpoints that are below the insulator,
									// until they are above the insulator.
									edge_normal = ReconstructEdgeNormal(
										prevpos, info.pos, nextpos, opppos
									);
								};
								f64 omegasq = omega_c.dot(omega_c);
								omegamod = sqrt(omegasq);
								unit_b = omega_c / omegamod;
								unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
								unit_perp = unit_perp / unit_perp.modulus();
								unit_Hall = unit_b.cross(unit_perp); // Note sign.
																	 // store omegamod instead.
							}
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
								{
									f64_vec3 intermed;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy

									intermed.x = unit_b.dotxy(gradvx);
									intermed.y = unit_b.dotxy(gradvy);
									intermed.z = unit_b.dotxy(gradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.dot(intermed);
										dvperp_by_db = unit_perp.dot(intermed);
										dvHall_by_db = unit_Hall.dot(intermed);

										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;
									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermed.x = unit_perp.dotxy(gradvx);
										intermed.y = unit_perp.dotxy(gradvy);
										intermed.z = unit_perp.dotxy(gradvz);

										dvb_by_dperp = unit_b.dot(intermed);
										dvperp_by_dperp = unit_perp.dot(intermed);
										dvHall_by_dperp = unit_Hall.dot(intermed);

										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;

									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

										intermed.x = unit_Hall.dotxy(gradvx);
										intermed.y = unit_Hall.dotxy(gradvy);
										intermed.z = unit_Hall.dotxy(gradvz);

										dvb_by_dHall = unit_b.dot(intermed);
										dvperp_by_dHall = unit_perp.dot(intermed);
										dvHall_by_dHall = unit_Hall.dot(intermed);

										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;

									}
								}

								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;

								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;

								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);

								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;
								Pi_H_b += ita_4*W_bP;

							}
							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

								if (bLongi) {
									edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
								};

								// Most efficient way: compute mom flux in magnetic coords
								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);

							}

							// Time to double-check carefully the signs.
							// Pi was defined with - on dv/dx and we then dot that with the edge_normal, so giving + if we are higher than outside.

							// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
							// is the flow of p_x dotted with the edge_normal
							// ownrates will be divided by N to give dv/dt
							// m N dvx/dt = integral div momflux_x
							// Therefore divide here just by m
							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
												
							ownrates_visc += visc_contrib;
							if (i % 2 != 0) // not vertex
							{

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES

								visc_htg += -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
								if (TESTIONVISC)
									printf("%d %d visc_htg %1.10E\n", iMinor, i, -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib)));

#else
								f64 htg_addn = -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));

								if (i == 1) // opposite 2
								{
									visc_htg0 += 0.5*htg_addn;
									visc_htg1 += 0.5*htg_addn;
								}
								else {
									if (i == 3) {
										visc_htg1 += 0.5*htg_addn;
										visc_htg2 += 0.5*htg_addn;
									}
									else {
										// i == 5:
										visc_htg0 += 0.5*htg_addn;
										visc_htg2 += 0.5*htg_addn;
									};
								};
#endif
							};
						};
					}; // bUsableSide

				};// p_Select[izNeighMinor[i]] == 0

				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR[iMinor]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(&(p_MAR[iMinor]), &(ownrates), sizeof(f64_vec3));
			
#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
			if (iSpecies == 1) {
				p_NT_addition_tri[iMinor].NiTi += visc_htg;
			} else {
				p_NT_addition_tri[iMinor].NeTe += visc_htg;
			}
#else
			
			if (iSpecies == 1) {
				p_NT_addition_tri[iMinor * 3 + 0].NiTi += visc_htg0;
				p_NT_addition_tri[iMinor * 3 + 1].NiTi += visc_htg1;
				p_NT_addition_tri[iMinor * 3 + 2].NiTi += visc_htg2;

			} else {
				p_NT_addition_tri[iMinor * 3 + 0].NeTe += visc_htg0;
				p_NT_addition_tri[iMinor * 3 + 1].NeTe += visc_htg1;
				p_NT_addition_tri[iMinor * 3 + 2].NeTe += visc_htg2;
			}

#endif
			
#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iMinor %d NaN ownrates.x\n", iMinor);
			if (ownrates.y != ownrates.y)
				printf("iMinor %d NaN ownrates.y\n", iMinor);
			if (ownrates.z != ownrates.z)
				printf("iMinor %d NaN ownrates.z\n", iMinor);

			if (visc_htg != visc_htg) printf("iMinor %d NAN VISC HTG\n", iMinor);
#endif
			// We do best by taking each boundary, considering how
			// much heat to add for each one.

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		};
	} // scope
}

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
) // easy way to put it in constant memory
{
	__shared__ f64_vec3 shared_v[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];
	__shared__ f64_vec2 shared_regr[threadsPerTileMinor];

	__shared__ f64_vec3 shared_v_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_regr_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	if (iSpecies > 0) {
		v4 temp = p_vie_minor[iMinor];
		shared_v[threadIdx.x].x = temp.vxy.x; 
		shared_v[threadIdx.x].y = temp.vxy.y;
		if (iSpecies == 1) {
			shared_v[threadIdx.x].z = temp.viz;
		} else {
			shared_v[threadIdx.x].z = temp.vez;
		};
	} else {
		shared_v[threadIdx.x] = p_v_n[iMinor];
	}
	shared_regr[threadIdx.x] = p__x[iMinor];
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			if (iSpecies > 0) {
				v4 temp;
				memcpy(&(temp), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
				shared_v_verts[threadIdx.x].x = temp.vxy.x;
				shared_v_verts[threadIdx.x].y = temp.vxy.y;
				if (iSpecies == 1) {
					shared_v_verts[threadIdx.x].z = temp.viz;
				} else {
					shared_v_verts[threadIdx.x].z = temp.vez;
				};
			} else {
				shared_v_verts[threadIdx.x] = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
			}
						
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_regr_verts[threadIdx.x] = p__x[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_v_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			memset(&(shared_regr_verts[threadIdx.x]), 0, sizeof(f64_vec2));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memcpy(&ownrates_visc, p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL,sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_vec3 opp_v, prev_v, next_v;
			f64_vec2 opp_x, prev_x, next_x;
			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_v[izTri[iprev] - StartMinor];
				prev_x = shared_regr[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				if (iSpecies > 0) {
					v4 temp = p_vie_minor[izTri[iprev]];
					prev_v.x = temp.vxy.x;
					prev_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						prev_v.z = temp.viz;
					} else {
						prev_v.z = temp.vez;
					};

					// we'd have done better with 2 separate v vectors as it turns out.
				} else {
					prev_v = p_v_n[izTri[iprev]];
				}
				prev_x = p__x[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
				prev_x = Clockwise_d*prev_x;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
				prev_x = Anticlockwise_d*prev_x;
			}
			
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_v[izTri[i] - StartMinor];
				
				opppos = shared_pos[izTri[i] - StartMinor];
				opp_x = shared_regr[izTri[i] - StartMinor];
			} else {
				if (iSpecies > 0) {
					v4 temp = p_vie_minor[izTri[i]];
					opp_v.x = temp.vxy.x;
					opp_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						opp_v.z = temp.viz;
					} else {
						opp_v.z = temp.vez;
					};
				} else {
					opp_v = p_v_n[izTri[i]];
				};
				opppos = p_info_minor[izTri[i]].pos;
				opp_x = p__x[izTri[i]];
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
				opp_x = Clockwise_d*opp_x;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
				opp_x = Anticlockwise_d*opp_x;
			}

#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext == tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

				f64_vec2 ROCgradvx, ROCgradvy;

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!
#if (TESTXYDERIVZVISCVERT) 
				printf("i %d inext %d \n", i, inext);
#endif

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_v[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
					next_x = shared_regr[izTri[inext] - StartMinor];
				}
				else {
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izTri[inext]];
						next_v.x = temp.vxy.x;
						next_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							next_v.z = temp.viz;
						} else {
							next_v.z = temp.vez;
						};
					} else {
						next_v = p_v_n[izTri[inext]];
					}
					nextpos = p_info_minor[izTri[inext]].pos;
					next_x = p__x[izTri[inext]];
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
					next_x = Clockwise_d*next_x;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
					next_x = Anticlockwise_d*next_x;
				}

				f64_vec3 omega_c;
				{
					f64_vec2 opp_B;
					f64 ita_theirs, nu_theirs;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						ita_theirs = shared_ita_par[izTri[i] - StartMinor];
						nu_theirs = shared_nu[izTri[i] - StartMinor];
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						ita_theirs = p_ita_parallel_minor[izTri[i]];
						nu_theirs = p_nu_minor[izTri[i]];
					};

					ita_par = sqrt(shared_ita_par_verts[threadIdx.x] * ita_theirs);

					if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.
				bool bLongi = false;
				if (ita_par > 0.0)
				{
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false) {
						
#if (TESTXYDERIVZVISCVERT)
							printf("(TestDomainPos(prevpos) == false) vx ours next opp %1.10E %1.10E %1.10E regr %1.9E %1.9E %1.9E\n",
								shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
								shared_regr_verts[threadIdx.x].x, next_x.x, opp_x.x);
#endif
						ROCgradvx =
							GetGradientDBydBeta_3Point(
								info.pos, nextpos, opppos,
								shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
								shared_regr_verts[threadIdx.x].x, next_x.x, opp_x.x
							);
						ROCgradvy =
							GetGradientDBydBeta_3Point(
								info.pos, nextpos, opppos,
								shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
								shared_regr_verts[threadIdx.x].y, next_x.y, opp_x.y
							);

					} else {
						if (TestDomainPos(nextpos) == false) {
							

#if (TESTXYDERIVZVISCVERT)
								printf("(TestDomainPos(nextpos) == false) vx prev ours opp %1.10E %1.10E %1.10E regr %1.9E %1.9E %1.9E\n",
									prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x,
									prev_x.x, shared_regr_verts[threadIdx.x].x, opp_x.x);
#endif
							ROCgradvx =
								GetGradientDBydBeta_3Point(
									prevpos, info.pos, opppos,
									prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x,
									prev_x.x, shared_regr_verts[threadIdx.x].x, opp_x.x
								);
							ROCgradvy =
								GetGradientDBydBeta_3Point(
									prevpos, info.pos, opppos,
									prev_v.y, shared_v_verts[threadIdx.x].y, opp_v.y,
									prev_x.y, shared_regr_verts[threadIdx.x].y, opp_x.y
								);

						} else {

#if (TESTXYDERIVZVISCVERT)
								printf("standard edge; %1.10E %1.10E %1.10E %1.10E regr %1.8E %1.8E %1.8E %1.8E\n"
									"prevpos info.pos nextpos opppos %1.10E %1.10E , %1.10E %1.10E , %1.10E %1.10E , %1.10E %1.10E \n",
									prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
									prev_x.x, shared_regr_verts[threadIdx.x].x, next_x.x, opp_x.x,
									prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y
								);
#endif
							ROCgradvx =
								GetGradientDBydBeta(
									prevpos, info.pos, nextpos, opppos,
									prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
									prev_x.x, shared_regr_verts[threadIdx.x].x, next_x.x, opp_x.x
								);
							ROCgradvy =
								GetGradientDBydBeta(
									prevpos, info.pos, nextpos, opppos,
									prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
									prev_x.y, shared_regr_verts[threadIdx.x].y, next_x.y, opp_x.y
								);
						};
					};
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false)) 
						bLongi = true;

#else
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						ROCgradvx = (opp_x.x - shared_regr_verts[threadIdx.x].x)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						ROCgradvy = (opp_x.y - shared_regr_verts[threadIdx.x].y)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
					}
					else {

						
						ROCgradvx =
							GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x,	opp_v.x,
								prev_x.x, shared_regr_verts[threadIdx.x].x, next_x.x, opp_x.x
							);
						ROCgradvy = 
							GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y,	opp_v.y,
								prev_x.y, shared_regr_verts[threadIdx.x].y, next_x.y, opp_x.y
							);
					};
#endif
				}

#if (TESTXYDERIVZVISCVERT) 
					printf("%d %d %d ROCgradvx %1.9E %1.9E ROCgradvy %1.9E %1.9E ita_par %1.9E\n",
						iVertex, i, izTri[i], ROCgradvx.x, ROCgradvx.y, ROCgradvy.x, ROCgradvy.y, ita_par);
#endif

				if (ita_par > 0.0) {

					if (iSpecies == 0) {
						f64_vec3 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
						if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
						
						visc_contrib.x = over_m_n*(ita_par*ROCgradvx.dot(edge_normal)); // if we are looking at higher vz looking out, go up.
						visc_contrib.y = over_m_n*(ita_par*ROCgradvy.dot(edge_normal));
						visc_contrib.z = 0.0;

						ownrates_visc += visc_contrib;
					} else {

						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy;// , Pi_zx, Pi_zy;

							Pi_xx = -ita_par*THIRD*(4.0*ROCgradvx.x - 2.0*ROCgradvy.y);
							Pi_xy = -ita_par*(ROCgradvx.y + ROCgradvy.x);
							Pi_yx = Pi_xy;
							Pi_yy = -ita_par*THIRD*(4.0*ROCgradvy.y - 2.0*ROCgradvx.x);
							//Pi_zx = 0.0; // -ita_par*(gradvz.x);
							//Pi_zy = 0.0; // -ita_par*(gradvz.y);

							f64_vec3 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);
							if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

							visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
							visc_contrib.z = 0.0;// -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

							ownrates_visc += visc_contrib;
						}
						else {

							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);
									if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.

										// Since B is in the plane, it's saying we picked perp in the plane, H = z.
										// CHECK THAT
										// It probably doesn't matter what direction perp is. As long as it is in the plane normal to b.
										// Formulary tells us if b is z then perp is x, H is y. So we can freely rotate x,y in the normal plane
										// and this will still hold.
										// Let's verify if we got it : 
										// z cross x = (0, 1, 0) --- tick.

										// From the perspective of the +- ita3 and ita4, all that matter is the relative orientation of P and H.
								}
								{
									f64_vec2 intermed;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									intermed.x = unit_b.dotxy(ROCgradvx);
									intermed.y = unit_b.dotxy(ROCgradvy);
									//intermed.z = unit_b.dotxy(gradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.dotxy(intermed);
										dvperp_by_db = unit_perp.dotxy(intermed);
										dvHall_by_db = unit_Hall.dotxy(intermed); // 0
										// This actually is always 0 because vHall isn't changing.

#if ((TESTXYDERIVZVISCVERT)) 
											printf("%d %d : dvb/db %1.9E dvP/db %1.9E \n",
												iVertex, i, dvb_by_db, dvperp_by_db);
#endif
										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;
									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermed.x = unit_perp.dotxy(ROCgradvx);
										intermed.y = unit_perp.dotxy(ROCgradvy);
										//intermed.z = unit_perp.dotxy(gradvz);

										dvb_by_dperp = unit_b.dotxy(intermed);
										dvperp_by_dperp = unit_perp.dotxy(intermed);
										dvHall_by_dperp = unit_Hall.dotxy(intermed); // 0

#if ((TESTXYDERIVZVISCVERT))
											printf("%d %d : dvb/dP %1.9E dvP/dP %1.9E \n",
												iVertex, i, dvb_by_dperp, dvperp_by_dperp);
#endif

										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;
									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

										intermed.x = unit_Hall.dotxy(ROCgradvx);  // 0
										intermed.y = unit_Hall.dotxy(ROCgradvy);  // 0

										dvb_by_dHall = unit_b.dotxy(intermed);
										dvperp_by_dHall = unit_perp.dotxy(intermed);
										dvHall_by_dHall = unit_Hall.dotxy(intermed);

										// all 0
										
										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;
									}
								}

#if ((TESTXYDERIVZVISCVERT))
									printf("%d %d : W_bb %1.10E W_bP %1.10E W_PP %1.10E W_bH %1.10E W_PH %1.10E W_HH %1.10E\n",
										iVertex, i, W_bb, W_bP, W_PP, W_bH, W_PH, W_HH);
#endif
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

									Pi_b_b += -ita_par*W_bb;
									Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									Pi_H_P += -ita_1*W_PH;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_2*W_bP;
									Pi_H_b += -ita_2*W_bH;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									Pi_P_P -= ita_3*W_PH;
									Pi_H_H += ita_3*W_PH;
									Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_4*W_bH;
									Pi_H_b += ita_4*W_bP;
								}
							} // scope W

							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);
								if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0

#if ((TESTXYDERIVZVISCVERT)) 
									printf("%d %d : mag_Edge %1.9E %1.9E %1.9E edge_normal %1.9E %1.9E bLongi %d b.y %1.7E perp.y %1.7E H.y %1.5E\n"
										"prevpos %1.8E %1.8E info.pos %1.8E %1.8E nextpos %1.8E %1.8E opppos %1.8E %1.8E\n",
										iVertex, i, mag_edge.x, mag_edge.y, mag_edge.z, edge_normal.x, edge_normal.y,
										((bLongi) ? 1 : 0), unit_b.y, unit_perp.y, unit_Hall.y,
										prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y
										);
#endif

								// So basically, Pi_H_H is not only 0, it is never used.

								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
							}

							// ROC :
							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

#if ((TESTXYDERIVZVISCVERT)) 
								printf("%d %d %d : 1/m_s %1.6E mf bPH %1.9E %1.9E %1.9E PibbPbHbPPHPHH %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E\n"
									"visc_ctb %1.9E %1.9E %1.9E \n==============================\n",
									iVertex, i, izTri[i], over_m_s, momflux_b, momflux_perp, momflux_Hall,								
									Pi_b_b, Pi_P_b,  Pi_P_P, Pi_H_b, Pi_H_P, Pi_H_H,
									visc_contrib.x, visc_contrib.y, visc_contrib.z);
#endif

							ownrates_visc += visc_contrib;

						}
					};
				}; // was ita_par == 0
				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				// Just leaving these but they won't do anything :
				prevpos = opppos;
				prev_v = opp_v;
				prev_x = opp_x;
				opppos = nextpos;
				opp_v = next_v;
				opp_x = next_x;
			}; // next i

			// Update:

#if ((TESTXYDERIVZVISCVERT)) 
				printf("%d : ownrates_visc %1.10E %1.10E %1.10E\n",
					iVertex, ownrates_visc.x, ownrates_visc.y, ownrates_visc.z);
#endif

			memcpy(p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, &ownrates_visc, sizeof(f64_vec3));
		} else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memcpy(&(ownrates_visc), &(p_ROCMAR[iMinor]), sizeof(f64_vec3));
	{
		long izNeighMinor[6];
		char szPBC[6];

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			f64_vec3 prev_v, opp_v, next_v;
			f64_vec2 prev_x, opp_x, next_x;
			f64_vec2 prevpos, nextpos, opppos;

			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{				
				prev_v = shared_v[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_x = shared_regr[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prev_v = shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prev_x = shared_regr_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izNeighMinor[iprev]];
						prev_v.x = temp.vxy.x;
						prev_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							prev_v.z = temp.viz;
						} else {
							prev_v.z = temp.vez;
						};
					} else {
						prev_v = p_v_n[izNeighMinor[iprev]];
					}
					prev_x = p__x[izNeighMinor[iprev]];
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
				prev_x = Clockwise_d*prev_x;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
				prev_x = Anticlockwise_d*prev_x;
			}

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opp_v = shared_v[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_x = shared_regr[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opp_v = shared_v_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opp_x = shared_regr_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izNeighMinor[i]];
						opp_v.x = temp.vxy.x;
						opp_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							opp_v.z = temp.viz;
						} else {
							opp_v.z = temp.vez;
						};
					} else {
						opp_v = p_v_n[izNeighMinor[i]];
					}
					opp_x = p__x[izNeighMinor[i]];
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
				opp_x = Clockwise_d*opp_x;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
				opp_x = Anticlockwise_d*opp_x;
			}

			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					next_v = shared_v[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_x = shared_regr[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						next_v = shared_v_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						next_x = shared_regr_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						if (iSpecies > 0) {
							v4 temp = p_vie_minor[izNeighMinor[inext]];
							next_v.x = temp.vxy.x;
							next_v.y = temp.vxy.y;
							if (iSpecies == 1) {
								next_v.z = temp.viz;
							}
							else {
								next_v.z = temp.vez;
							};
						} else {
							next_v = p_v_n[izNeighMinor[inext]];
						}
						next_x = p__x[izNeighMinor[inext]];
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
					next_x = Clockwise_d*next_x;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
					next_x = Anticlockwise_d*next_x;
				}


				bool bUsableSide = true;
				{
					f64 nu_theirs, ita_theirs;
					f64_vec2 opp_B(0.0, 0.0);
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
						ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							nu_theirs = p_nu_minor[izNeighMinor[i]];
							ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
						}
					}
					// GEOMETRIC ITA:
					if (ita_theirs == 0.0) bUsableSide = false;
					ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// ins-ins triangle traffic:

				bool bLongi = false;

#ifdef INS_INS_NONE
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
#else 
				//#ifdef INS_INS_LONGI
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bLongi = true;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
				// Use this in this case as flag for reconstructing edge normal to where
				// it only stays within triangle!
				//#else


				//#endif
#endif
				f64_vec2 ROCgradvx, ROCgradvy;
				if (bUsableSide)
				{
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false)
					{
						ROCgradvx = GetGradientDBydBeta_3Point(
							info.pos, nextpos, opppos,
							shared_v[threadIdx.x].x, next_v.x, opp_v.x,
							shared_regr[threadIdx.x].x, next_x.x, opp_x.x
						);
						ROCgradvy = GetGradientDBydBeta_3Point(
							info.pos, nextpos, opppos,
							shared_v[threadIdx.x].y, next_v.y, opp_v.y,
							shared_regr[threadIdx.x].y, next_x.y, opp_x.y
						);
					} else {
						if (TestDomainPos(nextpos) == false)
						{
							ROCgradvx = GetGradientDBydBeta_3Point(
								prevpos, info.pos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, opp_v.x,
								prev_x.x, shared_regr[threadIdx.x].x, opp_x.x
							);
							ROCgradvy = GetGradientDBydBeta_3Point(
								prevpos, info.pos, opppos,
								prev_v.y, shared_v[threadIdx.x].y, opp_v.y,
								prev_x.y, shared_regr[threadIdx.x].y, opp_x.y
							);
						} else {
							ROCgradvx = GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								prev_x.x, shared_regr[threadIdx.x].x, next_x.x, opp_x.x
							);
							ROCgradvy = GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
								prev_x.y, shared_regr[threadIdx.x].y, next_x.y, opp_x.y
							);
						}
					}
#else
					if (bLongi)
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						ROCgradvx = (opp_x.x - shared_regr[threadIdx.x].x)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						ROCgradvy = (opp_x.y - shared_regr[threadIdx.x].y)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
					} else {

						if (TEST_EPSILON_X_MINOR) {
						/*	printf("%d vx %1.8E %1.8E %1.8E %1.8E x.x %1.8E %1.8E %1.8E %1.8E\n",
								CHOSEN, prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								prev_x.x, shared_regr[threadIdx.x].x, next_x.x, opp_x.x);
							
							ROCgradvx = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								prev_x.x, 0.0,0.0,0.0
							);
							printf("ROCgradvx (prev,0,0,0) = %1.10E %1.10E\n", ROCgradvx.x, ROCgradvx.y);

							ROCgradvx = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								0.0, shared_regr[threadIdx.x].x, 0.0, 0.0
							);
							
							printf("ROCgradvx (0,self,0,0) = %1.10E %1.10E\n", ROCgradvx.x, ROCgradvx.y);
							
							ROCgradvx = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								0.0, 0.0, next_x.x, 0.0
							);
							
							printf("ROCgradvx (0,0,next,0) = %1.10E %1.10E\n", ROCgradvx.x, ROCgradvx.y);
							
							ROCgradvx = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								0.0, 0.0, 0.0, opp_x.x
							);

							printf("ROCgradvx (0,0,0,opp) = %1.10E %1.10E\n", ROCgradvx.x, ROCgradvx.y);
							*/
							ROCgradvx = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								prev_x.x, shared_regr[threadIdx.x].x, next_x.x, opp_x.x
							);

						} else {
							ROCgradvx = GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								prev_x.x, shared_regr[threadIdx.x].x, next_x.x, opp_x.x
							);
						};
						
						ROCgradvy = GetGradientDBydBeta(
							prevpos, info.pos, nextpos, opppos,
							prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
							prev_x.y, shared_regr[threadIdx.x].y, next_x.y, opp_x.y
						);						
					}
#endif
#ifdef INS_INS_NONE
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {
							// just set it to 0.
							bUsableSide = false;
							ROCgradvx.x = 0.0;
							ROCgradvx.y = 0.0;
							ROCgradvy.x = 0.0;
							ROCgradvy.y = 0.0;
							if (TEST_EPSILON_X_MINOR) {
								printf("ROCgradvx was set to zero.\n");
							}
						};
					};
#endif
				};

				if (bUsableSide) {
					if (iSpecies == 0) {
						f64_vec3 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

						if (bLongi) {
							edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
						};
						visc_contrib.x = over_m_n*(ita_par*ROCgradvx.dot(edge_normal)); // if we are looking at higher vz looking out, go up.
						visc_contrib.y = over_m_n*(ita_par*ROCgradvy.dot(edge_normal));
						visc_contrib.z = 0.0;

						ownrates_visc += visc_contrib;
					}
					else {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy;// Pi_zx, Pi_zy == 0;

							Pi_xx = -ita_par*THIRD*(4.0*ROCgradvx.x - 2.0*ROCgradvy.y);
							Pi_xy = -ita_par*(ROCgradvx.y + ROCgradvy.x);
							Pi_yx = Pi_xy;
							Pi_yy = -ita_par*THIRD*(4.0*ROCgradvy.y - 2.0*ROCgradvx.x);

							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);

							if (bLongi) {
								edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
							};
								
							f64_vec3 visc_contrib;
							visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
							visc_contrib.z = 0.0;// -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

							ownrates_visc += visc_contrib;

							// So we are saying if edge_normal.x > 0 and gradviz.x > 0
							// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
						}
						else {
							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);

									if (bLongi) {
										edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
									};

									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.
								}
								{
									f64_vec2 intermed;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									intermed.x = unit_b.dotxy(ROCgradvx);
									intermed.y = unit_b.dotxy(ROCgradvy);
									//intermed.z = unit_b.dotxy(gradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.dotxy(intermed);
										dvperp_by_db = unit_perp.dotxy(intermed);
										dvHall_by_db = unit_Hall.dotxy(intermed); // 0
																				  // This actually is always 0 because vHall isn't changing.

										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;
									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermed.x = unit_perp.dotxy(ROCgradvx);
										intermed.y = unit_perp.dotxy(ROCgradvy);
										//intermed.z = unit_perp.dotxy(gradvz);

										dvb_by_dperp = unit_b.dotxy(intermed);
										dvperp_by_dperp = unit_perp.dotxy(intermed);
										dvHall_by_dperp = unit_Hall.dotxy(intermed); // 0

										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;
									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

										intermed.x = unit_Hall.dotxy(ROCgradvx);  // 0
										intermed.y = unit_Hall.dotxy(ROCgradvy);  // 0

										dvb_by_dHall = unit_b.dotxy(intermed);
										dvperp_by_dHall = unit_perp.dotxy(intermed);
										dvHall_by_dHall = unit_Hall.dotxy(intermed);

										// all 0

										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;
									}
								}
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

									Pi_b_b += -ita_par*W_bb;
									Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									Pi_H_P += -ita_1*W_PH;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_2*W_bP;
									Pi_H_b += -ita_2*W_bH;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									Pi_P_P -= ita_3*W_PH;
									Pi_H_H += ita_3*W_PH;
									Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_4*W_bH;
									Pi_H_b += ita_4*W_bP;
								}
							} // scope W

							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);

								if (bLongi) {
									edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
								};

								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0
													// So basically, Pi_H_H is not only 0, it is never used.

								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
							}

							// ROC :
							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

							ownrates_visc += visc_contrib;

						};
					};
				}; // bUsableSide

				prevpos = opppos;
				prev_v = opp_v;
				prev_x = opp_x;
				opppos = nextpos;
				opp_v = next_v;
				opp_x = next_x;
			};

			// UPDATE:
			memcpy(&(p_ROCMAR[iMinor]), &(ownrates_visc), sizeof(f64_vec3));

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
			memset(&(p_ROCMAR[iMinor]), 0, sizeof(f64_vec3));
		};
	} // scope
}


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
) // easy way to put it in constant memory
{
	//__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];
	__shared__ f64 shared_regr[threadsPerTileMinor];

	//__shared__ v4 shared_vie_verts[threadsPerTileMajor]; // if we could drop 3*1.5 that would be a plus for sure.
	__shared__ f64 shared_regr_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

	__shared__ f64 shared_vz[threadsPerTileMinor];
	__shared__ f64 shared_vz_verts[threadsPerTileMajor];

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	if (iSpecies > 0){
		v4 v = p_vie_minor[iMinor];

		if (iSpecies == 1) {
			shared_vz[threadIdx.x] = v.viz;
		} else {
			shared_vz[threadIdx.x] = v.vez;
		}
	} else {
		shared_vz[threadIdx.x] = p_v_n[iMinor].z;
	}

	shared_regr[threadIdx.x] = p__x[iMinor];
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.
	
	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			if (iSpecies > 0) {
				v4 v;
				memcpy(&(v), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
				if (iSpecies == 1) {
					shared_vz_verts[threadIdx.x] = v.viz;
				} else {
					shared_vz_verts[threadIdx.x] = v.vez;
				}
			} else {
				shared_vz_verts[threadIdx.x] = p_v_n[iVertex + BEGINNING_OF_CENTRAL].z;
			};

			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_regr_verts[threadIdx.x] = p__x[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.

		} else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_vz_verts[threadIdx.x]), 0, sizeof(f64));
			memset(&(shared_regr_verts[threadIdx.x]), 0, sizeof(f64));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		//memset(&ownrates_visc, 0, sizeof(f64_vec3));
		memcpy(&ownrates_visc, p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64 opp_v, prev_v, next_v; // Hey ... do we even need/use vxy ?
			f64 opp_x, prev_x, next_x;
			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vz[izTri[iprev] - StartMinor];
				prev_x = shared_regr[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				if (iSpecies == 0) {
					prev_v = p_v_n[izTri[iprev]].z;
				} else {
					if (iSpecies == 1) {
						prev_v = p_vie_minor[izTri[iprev]].viz;
					}
					else {
						prev_v = p_vie_minor[izTri[iprev]].vez;
					}
				};
				prev_x = p__x[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vz[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
				opp_x = shared_regr[izTri[i] - StartMinor];
			}
			else {
				if (iSpecies == 0) {
					opp_v = p_v_n[izTri[i]].z;
				} else {
					if (iSpecies == 1) {
						opp_v = p_vie_minor[izTri[i]].viz;
					}
					else {
						opp_v = p_vie_minor[izTri[i]].vez;
					}
				};
				opppos = p_info_minor[izTri[i]].pos;
				opp_x = p__x[izTri[i]];
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext == tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

				f64_vec2 ROCgradvz;

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vz[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
					next_x = shared_regr[izTri[inext] - StartMinor];
				} else {
					if (iSpecies == 0) {
						next_v = p_v_n[izTri[inext]].z;
					} else {
						if (iSpecies == 1) {
							next_v = p_vie_minor[izTri[inext]].viz;
						}
						else {
							next_v = p_vie_minor[izTri[inext]].vez;
						}
					};
					nextpos = p_info_minor[izTri[inext]].pos;
					next_x = p__x[izTri[inext]];
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				f64_vec3 omega_c;
				{
					f64_vec2 opp_B;
					f64 ita_theirs, nu_theirs;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						ita_theirs = shared_ita_par[izTri[i] - StartMinor];
						nu_theirs = shared_nu[izTri[i] - StartMinor];
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						ita_theirs = p_ita_parallel_minor[izTri[i]];
						nu_theirs = p_nu_minor[izTri[i]];
					};
					ita_par = sqrt(shared_ita_par_verts[threadIdx.x] * ita_theirs);

					if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				bool bLongi = false;
				if (ita_par > 0.0)
				{
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false)
					{
						ROCgradvz =
							GetGradientDBydBeta_3Point(
								info.pos, nextpos, opppos,
								shared_vz_verts[threadIdx.x], next_v, opp_v,
								shared_regr_verts[threadIdx.x], next_x, opp_x
							);
					} else {
						if (TestDomainPos(nextpos) == false)
						{
							ROCgradvz =
								GetGradientDBydBeta_3Point(
									prevpos, info.pos, opppos,
									prev_v, shared_vz_verts[threadIdx.x], opp_v,
									prev_x, shared_regr_verts[threadIdx.x], opp_x
								);

						} else {

							ROCgradvz =
								GetGradientDBydBeta(
									prevpos, info.pos, nextpos, opppos,
									prev_v, shared_vz_verts[threadIdx.x], next_v, opp_v,
									prev_x, shared_regr_verts[threadIdx.x], next_x, opp_x
								);
						};
					}

					if (TESTXYDERIVZVISCVERT) {
						printf("%d %d %d ROCgradvz %1.9E %1.9E \n",
							iVertex, i, izTri[i], ROCgradvz.x, ROCgradvz.y);
					};
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						bLongi = true;
#else
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						ROCgradvz = (opp_x - shared_regr_verts[threadIdx.x])*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						
					} else {

						ROCgradvz =
							GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v, shared_vz_verts[threadIdx.x], next_v, opp_v,
								prev_x, shared_regr_verts[threadIdx.x], next_x, opp_x
							);
					};
#endif
				};

				if (ita_par > 0.0) {

					if (iSpecies == 0) {
						f64 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
						if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

						visc_contrib = over_m_n*(ita_par*ROCgradvz.dot(edge_normal));

						ownrates_visc.z += visc_contrib;
					}
					else {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case (actually same as neutral ... )
							f64 Pi_zx, Pi_zy;

							//Pi_xx = -ita_par*THIRD*(4.0*ROCgradvx.x - 2.0*ROCgradvy.y);
							//Pi_xy = -ita_par*(ROCgradvx.y + ROCgradvy.x);
							//Pi_yx = Pi_xy;
							//Pi_yy = -ita_par*THIRD*(4.0*ROCgradvy.y - 2.0*ROCgradvx.x);
							Pi_zx = -ita_par*(ROCgradvz.x);
							Pi_zy = -ita_par*(ROCgradvz.y);

							f64 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);
							if (bLongi) 
								edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
							
							//visc_contrib.x = 0.0;
							//visc_contrib.y = 0.0;
							visc_contrib = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

							ownrates_visc.z += visc_contrib;
						}
						else {

							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);
									if (bLongi)
										edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.
								}
								{
									f64 intermedz;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									intermedz = unit_b.dotxy(ROCgradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.z*(intermedz);       // 0
										dvperp_by_db = unit_perp.z*(intermedz); // 0
										dvHall_by_db = unit_Hall.z*(intermedz);

										if ((TESTXYDERIVZVISCVERT))
											printf("%d %d : dvb/db %1.9E dvP/db %1.9E \n",
												iVertex, i, dvb_by_db, dvperp_by_db);

										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;     // nonzero
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;
									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermedz = unit_perp.dotxy(ROCgradvz);
										//intermed.z = unit_perp.dotxy(gradvz);

										dvb_by_dperp = unit_b.z*(intermedz);
										dvperp_by_dperp = unit_perp.z*(intermedz);
										dvHall_by_dperp = unit_Hall.z*(intermedz); // 0
#if ((TESTXYDERIVZVISCVERT))
											printf("%d %d : dvb/dP %1.9E dvP/dP %1.9E \n",
												iVertex, i, dvb_by_dperp, dvperp_by_dperp);
#endif
										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;
									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

										intermedz = unit_Hall.dotxy(ROCgradvz);  // 0

										// These are all always 0 ! Because nothing changes in z direction.

										dvb_by_dHall = unit_b.z*(intermedz);
										dvperp_by_dHall = unit_perp.z*(intermedz);
										dvHall_by_dHall = unit_Hall.z*(intermedz);

										// all 0

										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;
									}
								}

#if ((TESTXYDERIVZVISCVERT))
									printf("%d %d : W_bb %1.10E W_bP %1.10E W_PP %1.10E W_bH %1.10E W_PH %1.10E W_HH %1.10E\n",
										iVertex, i, W_bb, W_bP, W_PP, W_bH, W_PH, W_HH);
#endif
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

									Pi_b_b += -ita_par*W_bb;
									Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									Pi_H_P += -ita_1*W_PH;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_2*W_bP;
									Pi_H_b += -ita_2*W_bH;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									Pi_P_P -= ita_3*W_PH;
									Pi_H_H += ita_3*W_PH;
									Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_4*W_bH;
									Pi_H_b += ita_4*W_bP;
								}
							} // scope W

							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);
								if (bLongi)
									edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0

								// So basically, Pi_H_H is not only 0, it is never used.

#if ((TESTXYDERIVZVISCVERT)) 
									printf("%d %d : mag_Edge %1.9E %1.9E %1.9E edge_normal %1.9E %1.9E bLongi %d \n"
										"prevpos %1.8E %1.8E info.pos %1.8E %1.8E nextpos %1.8E %1.8E opppos %1.8E %1.8E\n",
										iVertex, i, mag_edge.x, mag_edge.y, mag_edge.z, edge_normal.x, edge_normal.y,
										((bLongi) ? 1 : 0),
										prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y
									);
#endif
								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
							}

							// ROC :
							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

#if ((TESTXYDERIVZVISCVERT)) 
								printf("%d %d %d : 1/m_s %1.6E mf bPH %1.9E %1.9E %1.9E PibbPbPPHbHPHH %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E\n"
									"visc_ctb %1.9E %1.9E %1.9E \n==============================\n",
									iVertex, i, izTri[i], over_m_s, momflux_b, momflux_perp, momflux_Hall,
									Pi_b_b, Pi_P_b,Pi_P_P, Pi_H_b, Pi_H_P, Pi_H_H,
									visc_contrib.x, visc_contrib.y, visc_contrib.z);
#endif
							ownrates_visc += visc_contrib;

						};
					};
				}; // was ita_par == 0
				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				// Just leaving these but they won't do anything :
				prevpos = opppos;
				prev_v = opp_v;
				prev_x = opp_x;
				opppos = nextpos;
				opp_v = next_v;
				opp_x = next_x;
			}; // next i

			// UPDATE
			memcpy(p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, &ownrates_visc, sizeof(f64_vec3));

#if ((TESTXYDERIVZVISCVERT)) 
				printf("%d ownrates_visc (from z) %1.10E %1.10E %1.10E \n",iVertex, ownrates_visc.x, ownrates_visc.y, ownrates_visc.z);
#endif

		} else {			
			memset(p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, 0, sizeof(f64_vec3));
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];

	memcpy(&(ownrates_visc), &(p_ROCMAR[iMinor]), sizeof(f64_vec3));
	{
		long izNeighMinor[6];
		char szPBC[6];

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			f64 prev_v, opp_v, next_v;
			f64 prev_x, opp_x, next_x;
			f64_vec2 prevpos, nextpos, opppos;

			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prev_v = shared_vz[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_x = shared_regr[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prev_v = shared_vz_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prev_x = shared_regr_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;

					if (iSpecies == 0) {
						prev_v = p_v_n[izNeighMinor[iprev]].z;
					} else {
						if (iSpecies == 1) {
							prev_v = p_vie_minor[izNeighMinor[iprev]].viz;
						} else {
							prev_v = p_vie_minor[izNeighMinor[iprev]].vez;
						};
					};
					prev_x = p__x[izNeighMinor[iprev]];
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opp_v = shared_vz[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_x = shared_regr[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opp_v = shared_vz_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opp_x = shared_regr_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					if (iSpecies == 0) {
						opp_v = p_v_n[izNeighMinor[i]].z;
					} else {
						if (iSpecies == 1) {
							opp_v = p_vie_minor[izNeighMinor[i]].viz;
						} else {
							opp_v = p_vie_minor[izNeighMinor[i]].vez;
						};
					};
					opp_x = p__x[izNeighMinor[i]];
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}
			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					next_v = shared_vz[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_x = shared_regr[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						next_v = shared_vz_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						next_x = shared_regr_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						if (iSpecies == 0) {
							next_v = p_v_n[izNeighMinor[inext]].z;
						} else {
							if (iSpecies == 1) {
								next_v = p_vie_minor[izNeighMinor[inext]].viz;
							}
							else {
								next_v = p_vie_minor[izNeighMinor[inext]].vez;
							}
						};
						next_x = p__x[izNeighMinor[inext]];
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}


				bool bUsableSide = true;
				{
					f64 nu_theirs, ita_theirs;
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
						ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							nu_theirs = p_nu_minor[izNeighMinor[i]];
							ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
						}
					}
					// GEOMETRIC ITA:
					if (ita_theirs == 0.0) bUsableSide = false;
					ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// ins-ins triangle traffic:

				bool bLongi = false;

#ifdef INS_INS_NONE
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
#else 
//#ifdef INS_INS_LONGI
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bLongi = true;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
				// Use this in this case as flag for reconstructing edge normal to where
				// it only stays within triangle!
//#else
				
				
//#endif
#endif

				f64_vec2 ROCgradvz;
				if (bUsableSide)
				{
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false)
					{
						ROCgradvz = GetGradientDBydBeta_3Point(
							info.pos, nextpos, opppos,
							shared_vz[threadIdx.x], next_v, opp_v,
							shared_regr[threadIdx.x], next_x, opp_x
						);

					} else {
						if (TestDomainPos(nextpos) == false) {
							ROCgradvz = GetGradientDBydBeta_3Point(
								prevpos, info.pos, opppos,
								prev_v, shared_vz[threadIdx.x], opp_v,
								prev_x, shared_regr[threadIdx.x], opp_x
							);
						} else {
							
							ROCgradvz = GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v, shared_vz[threadIdx.x], next_v, opp_v,
								prev_x, shared_regr[threadIdx.x], next_x, opp_x
							);
						};
					};

#else
					if (bLongi)
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.
						if (0)//iMinor == CHOSEN) 
							printf("longi. x %1.8E %1.8E\n", opp_x, shared_regr[threadIdx.x]);

						ROCgradvz = (opp_x - shared_regr[threadIdx.x])*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
					} else {

						if (0)//(iMinor == CHOSEN)) 
						{
							printf("%d v %1.8E %1.8E %1.8E %1.8E x %1.8E %1.8E %1.8E %1.8E\n",
								CHOSEN, prev_v, shared_vz[threadIdx.x], next_v, opp_v,
								prev_x, shared_regr[threadIdx.x], next_x, opp_x);
							ROCgradvz = GetGradientDBydBetaDebug(
								prevpos, info.pos, nextpos, opppos,
								prev_v, shared_vz[threadIdx.x], next_v, opp_v,
								prev_x, shared_regr[threadIdx.x], next_x, opp_x
							);
						} else {

							ROCgradvz = GetGradientDBydBeta(
								prevpos, info.pos, nextpos, opppos,
								prev_v, shared_vz[threadIdx.x], next_v, opp_v,
								prev_x, shared_regr[threadIdx.x], next_x, opp_x
							);
						};
					};
#endif

#ifdef INS_INS_NONE
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {
							// just set it to 0.
							bUsableSide = false;
							ROCgradvz.x = 0.0;
							ROCgradvz.y = 0.0;
						};
					};
#endif
				};


			//	if (0)//iMinor == CHOSEN) 
			//		printf("%d ROCgradvz %1.8E %1.8E bUsableSide %d\n",
			//		CHOSEN, ROCgradvz.x, ROCgradvz.y, bUsableSide ? 1 : 0);


				if (bUsableSide) {
					if (iSpecies == 0) {
						f64 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

						if (bLongi) {
							edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

						//	This is where we need to think harder in all routines ---
						//	in the case that nextdoor is CROSSING_INS, we want to stop at edge of triangle
						
							// use bLongi as flag.
						};

						visc_contrib = over_m_n*(ita_par*ROCgradvz.dot(edge_normal));

						ownrates_visc.z += visc_contrib;
					} else {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							f64 Pi_zx, Pi_zy;
							Pi_zx = -ita_par*(ROCgradvz.x);
							Pi_zy = -ita_par*(ROCgradvz.y);

							f64 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);

							if (bLongi) {
								edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
							};
							//visc_contrib.x = 0.0;
							//visc_contrib.y = 0.0;
							visc_contrib = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						//	if (0)//iMinor == CHOSEN)
						//		printf("CHOSEN unmag visc_contrib.z %1.8E \n", visc_contrib);

							ownrates_visc.z += visc_contrib;

							// So we are saying if edge_normal.x > 0 and gradviz.x > 0
							// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
						}
						else {
							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
							{
								f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);

									if (bLongi) {
										edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
									};
									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.
								}
								{
									f64 intermedz;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									intermedz = unit_b.dotxy(ROCgradvz);
									{
										f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										dvb_by_db = unit_b.z*(intermedz);
										dvperp_by_db = unit_perp.z*(intermedz);
										dvHall_by_db = unit_Hall.z*(intermedz); // 0
																				  // This actually is always 0 because vHall isn't changing.
										W_bb += 4.0*THIRD*dvb_by_db;
										W_bP += dvperp_by_db;
										W_bH += dvHall_by_db;
										W_PP -= 2.0*THIRD*dvb_by_db;
										W_HH -= 2.0*THIRD*dvb_by_db;
									}
									{
										f64 dvb_by_dperp, dvperp_by_dperp,
											dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										intermedz = unit_perp.dotxy(ROCgradvz);

										dvb_by_dperp = unit_b.z*(intermedz);
										dvperp_by_dperp = unit_perp.z*(intermedz);
										dvHall_by_dperp = unit_Hall.z*(intermedz); // 0

										W_bb -= 2.0*THIRD*dvperp_by_dperp;
										W_PP += 4.0*THIRD*dvperp_by_dperp;
										W_HH -= 2.0*THIRD*dvperp_by_dperp;
										W_bP += dvb_by_dperp;
										W_PH += dvHall_by_dperp;
									}
									{
										f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

										intermedz = unit_Hall.dotxy(ROCgradvz); // 0

										dvb_by_dHall = unit_b.z*(intermedz);
										dvperp_by_dHall = unit_perp.z*(intermedz);
										dvHall_by_dHall = unit_Hall.z*(intermedz);

										W_bb -= 2.0*THIRD*dvHall_by_dHall;
										W_PP -= 2.0*THIRD*dvHall_by_dHall;
										W_HH += 4.0*THIRD*dvHall_by_dHall;
										W_bH += dvb_by_dHall;
										W_PH += dvperp_by_dHall;
									}
								}
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

									Pi_b_b += -ita_par*W_bb;
									Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									Pi_H_P += -ita_1*W_PH;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_2*W_bP;
									Pi_H_b += -ita_2*W_bH;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									Pi_P_P -= ita_3*W_PH;
									Pi_H_H += ita_3*W_PH;
									Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									Pi_P_b += -ita_4*W_bH;
									Pi_H_b += ita_4*W_bP;
								}
							} // scope W

							f64 momflux_b, momflux_perp, momflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);

								if (bLongi) {
									edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
								};

								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0

								momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
							}

							// ROC :
							f64_vec3 visc_contrib;
							visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

						//	if ((0) && (iMinor == CHOSEN)) printf("CHOSEN mag visc_contrib.z %1.8E \n", visc_contrib.z);

							ownrates_visc += visc_contrib;
						};
					};
				}; // bUsableSide

				prevpos = opppos;
				prev_v = opp_v;
				prev_x = opp_x;
				opppos = nextpos;
				opp_v = next_v;
				opp_x = next_x;
			};

			//if (iMinor == CHOSEN) printf("CHOSEN ROCMAR.z %1.8E \n", ownrates_visc.z);

			// UPDATE:
			memcpy(&(p_ROCMAR[iMinor]), &(ownrates_visc), sizeof(f64_vec3));

		} else {
			memset(&(p_ROCMAR[iMinor]), 0, sizeof(f64_vec3));
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope
}

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
) // easy way to put it in constant memory
{
	//__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input

	__shared__ f64_vec3 shared_v[threadsPerTileMinor]; 
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];
//	__shared__ f64 shared_regr[threadsPerTileMinor];

	//__shared__ v4 shared_vie_verts[threadsPerTileMajor]; // if we could drop 3*1.5 that would be a plus for sure.
//	__shared__ f64 shared_regr_verts[threadsPerTileMajor];
	__shared__ f64_vec3 shared_v_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

	//__shared__ f64 shared_vz[threadsPerTileMinor];
	//__shared__ f64 shared_vz_verts[threadsPerTileMajor];


	// 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
	// Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	if (iSpecies > 0) {

		v4 temp = p_vie_minor[iMinor];
		shared_v[threadIdx.x].x = temp.vxy.x;
		shared_v[threadIdx.x].y = temp.vxy.y;
		if (iSpecies == 1) {
			shared_v[threadIdx.x].z = temp.viz;
		} else {
			shared_v[threadIdx.x].z = temp.vez;
		};
	} else {
		shared_v[threadIdx.x] = p_v_n[iMinor];
	};

	//shared_regr[threadIdx.x] = p__x[iMinor];
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];


	//if (0)//iMinor == lChosen) 
	//	printf("\n\n iMinor %d B %1.10E %1.10E v %1.10E %1.10E %1.10E nu %1.9E \n\n\n",
	//	iMinor, shared_B[threadIdx.x].x, shared_B[threadIdx.x].y,
	//	shared_v[threadIdx.x].x, shared_v[threadIdx.x].y, shared_v[threadIdx.x].z,
	//	shared_nu[threadIdx.x]);


	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			if (iSpecies > 0) {
				v4 temp;
				memcpy(&(temp), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
				shared_v_verts[threadIdx.x].x = temp.vxy.x;
				shared_v_verts[threadIdx.x].y = temp.vxy.y; 
				if (iSpecies == 1) {
					shared_v_verts[threadIdx.x].z = temp.viz;
				}
				else {
					shared_v_verts[threadIdx.x].z = temp.vez;
				};				
			} else {
				shared_v_verts[threadIdx.x] = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
			}

			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
	//		shared_regr_verts[threadIdx.x] = p__x[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.

		} else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_v_verts[threadIdx.x]), 0, sizeof(f64_vec3));

	//		memset(&(shared_regr_verts[threadIdx.x]), 0, sizeof(f64));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};
	int iDimension;

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		//memset(&ownrates_visc, 0, sizeof(f64_vec3));
		//memcpy(&ownrates_visc, p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))
		{
			//	f64 * __restrict__ p_array, // say it is 256x256. 
			//	bool * __restrict__ p_selectflag, // whether it's in the smoosh to smash
			//short * __restrict__ p_mapping_to_array,

			bool bSelected = p_selectflag[iVertex + BEGINNING_OF_CENTRAL];
			if (bSelected) {

				short iEqnOurs, iEqnTheirs;

				iEqnOurs = p_mapping_to_array[iVertex + BEGINNING_OF_CENTRAL];
				// We are losing energy if there is viscosity into OUTERMOST.

				memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
				memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

				f64_vec3 opp_v, prev_v, next_v;
			//	f64 opp_x, prev_x, next_x;
				f64_vec2 opppos, prevpos, nextpos;
				// ideally we might want to leave position out of the loop so that we can avoid reloading it.

				short i = 0;
				short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
				short inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
				{
					prev_v = shared_v[izTri[iprev] - StartMinor];
					//	prev_x = shared_regr[izTri[iprev] - StartMinor];
					prevpos = shared_pos[izTri[iprev] - StartMinor];
				} else {
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izTri[iprev]];
						prev_v.x = temp.vxy.x;
						prev_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							prev_v.z = temp.viz;
						} else {
							prev_v.z = temp.vez;
						};
						// we'd have done better with 2 separate v vectors as it turns out.
					} else {
						prev_v = p_v_n[izTri[iprev]];
					}
					//			prev_x = p__x[izTri[iprev]];
					prevpos = p_info_minor[izTri[iprev]].pos;
				};
				if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
					prevpos = Clockwise_d*prevpos;
					RotateClockwise(prev_v);
				};
				if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
					prevpos = Anticlockwise_d*prevpos;
					RotateAnticlockwise(prev_v);
				};

				if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
				{
					opp_v = shared_v[izTri[i] - StartMinor];
					opppos = shared_pos[izTri[i] - StartMinor];
					//			opp_x = shared_regr[izTri[i] - StartMinor];
				} else {
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izTri[i]];
						opp_v.x = temp.vxy.x;
						opp_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							opp_v.z = temp.viz;
						} else {
							opp_v.z = temp.vez;
						};
					} else {
						opp_v = p_v_n[izTri[i]];
					};
					opppos = p_info_minor[izTri[i]].pos;
					//			opp_x = p__x[izTri[i]];
				}
				if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
					opppos = Clockwise_d*opppos;
					RotateClockwise(opp_v);
				}
				if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
					opppos = Anticlockwise_d*opppos;
					RotateAnticlockwise(opp_v);
				}

#pragma unroll 
				for (i = 0; i < tri_len; i++)
				{
					// Tri 0 is anticlockwise of neighbour 0, we think
					inext = i + 1; if (inext == tri_len) inext = 0;
					iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

					// Order of calculations may help things to go out/into scope at the right times so careful with that.
					//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
					// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

					// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
					// That seems counter-intuitive??
					// Oh and the positions too!

					if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
					{
						next_v = shared_v[izTri[inext] - StartMinor];
						nextpos = shared_pos[izTri[inext] - StartMinor];
						//			next_x = shared_regr[izTri[inext] - StartMinor];
					}
					else {
						if (iSpecies > 0) {
							v4 temp = p_vie_minor[izTri[inext]];
							next_v.x = temp.vxy.x;
							next_v.y = temp.vxy.y;
							if (iSpecies == 1) {
								next_v.z = temp.viz;
							} else {
								next_v.z = temp.vez;
							};
						} else {
							next_v = p_v_n[izTri[inext]];
						};
						nextpos = p_info_minor[izTri[inext]].pos;
						//			next_x = p__x[izTri[inext]];
					}
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextpos = Clockwise_d*nextpos;
						RotateClockwise(next_v);
					}
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
						nextpos = Anticlockwise_d*nextpos;
						RotateAnticlockwise(next_v);
					}

					f64_vec3 omega_c;
					{
						f64_vec2 opp_B;
						f64 ita_theirs, nu_theirs;
						if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
						{
							opp_B = shared_B[izTri[i] - StartMinor];
							ita_theirs = shared_ita_par[izTri[i] - StartMinor];
							nu_theirs = shared_nu[izTri[i] - StartMinor];
						}
						else {
							opp_B = p_B_minor[izTri[i]].xypart();
							ita_theirs = p_ita_parallel_minor[izTri[i]];
							nu_theirs = p_nu_minor[izTri[i]];
						};

						ita_par = sqrt(shared_ita_par_verts[threadIdx.x] * ita_theirs);

						if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opp_B = Clockwise_d*opp_B;
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opp_B = Anticlockwise_d*opp_B;
						}
						nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
						omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
						if (iSpecies == 1) omega_c *= qoverMc;
						if (iSpecies == 2) omega_c *= qovermc;
					} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.
					

					f64 prev_x, opp_x, next_x, self_x;
					f64_vec2 ROCgradvx, ROCgradvy, ROCgradvz;

					for (iDimension = 0; iDimension < 3; iDimension++)
					{
						for (int iWhich = 0; iWhich < 4; iWhich++) {
							prev_x = 0.0;
							opp_x = 0.0;
							next_x = 0.0;
							self_x = 0.0;
							long iTri;
							if (iWhich == 0) {
								prev_x = 1.0;
								iTri = izTri[iprev];
								//iEqnTheirs = p_mapping_to_array[izTri[iprev]];
							}
							if (iWhich == 1) {
								opp_x = 1.0;
								iTri = izTri[i];
							};
							if (iWhich == 2) {
								next_x = 1.0;
								iTri = izTri[inext];
							};
							if (iWhich == 3) self_x = 1.0;

							// Optimize later.

							// is iWhich's point selected? If not, pass over this loop.
							if (((iWhich < 3) && (p_selectflag[iTri]))
								||
								(iWhich == 3))
							{
								if (iWhich < 3) {
									iEqnTheirs = p_mapping_to_array[iTri];
								}
								else {
									iEqnTheirs = iEqnOurs;
								};

								bool bLongi = false;
								ROCgradvx.x = 0.0; ROCgradvx.y = 0.0;
								ROCgradvy.x = 0.0; ROCgradvy.y = 0.0;
								ROCgradvz.x = 0.0; ROCgradvz.y = 0.0;

								if (ita_par > 0.0)
								{
#ifdef INS_INS_3POINT
									if (TestDomainPos(prevpos) == false)
									{
#if DBG_ROC
										if (DBG_ROC && (iVertex == VERTCHOSEN) && (iDimension == 1) && (iEqnTheirs == 1)) {
											printf("TestDomainPos(prevpos) == false !\n");
										};
#endif
										if (iDimension == 0)
											ROCgradvx =
											GetGradientDBydBeta_3Point(
												info.pos, nextpos, opppos,
												shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
												self_x, next_x, opp_x
											);
										if (iDimension == 1) 
											ROCgradvy =
											GetGradientDBydBeta_3Point(
												info.pos, nextpos, opppos,
												shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
												self_x, next_x, opp_x
											);						



										if (iDimension == 2)
											ROCgradvz =
											GetGradientDBydBeta_3Point(
												info.pos, nextpos, opppos,
												shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z,
												self_x, next_x, opp_x
											);
									} else {
										if (TestDomainPos(nextpos) == false)
										{
											if (DBG_ROC && (iVertex == VERTCHOSEN) && (iDimension == 1) && (iEqnTheirs == 1)) {
												printf("TestDomainPos(nextpos) == false !\n");
											};

											if (iDimension == 0)
												ROCgradvx =
												GetGradientDBydBeta_3Point(
													prevpos, info.pos, opppos,
													prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x,
													prev_x, self_x, opp_x
												);
											if (iDimension == 1)
											ROCgradvy =
												GetGradientDBydBeta_3Point(
													prevpos, info.pos, opppos,
													prev_v.y, shared_v_verts[threadIdx.x].y, opp_v.y,
													prev_x, self_x, opp_x
												); 
											if (iDimension == 2)
											ROCgradvz =
												GetGradientDBydBeta_3Point(
													prevpos, info.pos, opppos,
													prev_v.z, shared_v_verts[threadIdx.x].z, opp_v.z,
													prev_x, self_x, opp_x
												);
										} else {
											
											if (iDimension == 0)
												ROCgradvx =
												GetGradientDBydBeta(
													prevpos, info.pos, nextpos, opppos,
													prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
													prev_x, self_x, next_x, opp_x
												);
											if (iDimension == 1) {
#if DGB_ROC
												if (DBG_ROC && (iVertex == VERTCHOSEN) && (iDimension == 1) && (iEqnTheirs == 1))
												{

													ROCgradvy =
														GetGradientDBydBetaDebug(
															prevpos, info.pos, nextpos, opppos,
															prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
															prev_x, self_x, next_x, opp_x
														);

													printf("Both sides domain! %d %d vy psno %1.10E %1.10E %1.10E %1.10E \n"
														"prev_x, self_x, next_x, opp_x %1.4E %1.4E %1.4E %1.4E ROCgradvy %1.12E %1.12E \n",
														iVertex, i,
														prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
														prev_x, self_x, next_x, opp_x,
														ROCgradvy.x, ROCgradvy.y);
												};
#else
												
													ROCgradvy =
														GetGradientDBydBeta(
															prevpos, info.pos, nextpos, opppos,
															prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y,
															prev_x, self_x, next_x, opp_x
														);
#endif
											};
											if (iDimension == 2)
												ROCgradvz =
												GetGradientDBydBeta(
													prevpos, info.pos, nextpos, opppos,
													prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z,
													prev_x, self_x, next_x, opp_x
												);
										};
									}

									if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
										bLongi = true;
#else
									if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
									{
										// One of the sides is dipped under the insulator -- set transverse deriv to 0.
										// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

										ROCgradvz = (opp_x - shared_regr_verts[threadIdx.x])*(opppos - info.pos) /
											(opppos - info.pos).dot(opppos - info.pos);

									}
									else {

										ROCgradvz =
											GetGradientDBydBeta(
												prevpos, info.pos, nextpos, opppos,
												prev_v, shared_vz_verts[threadIdx.x], next_v, opp_v,
												prev_x, self_x, next_x, opp_x
											);
									};
#endif
								};

								if (ita_par > 0.0) {

									if (iSpecies == 0) {
										f64_vec3 visc_contrib;
										f64_vec2 edge_normal;
										edge_normal.x = THIRD * (nextpos.y - prevpos.y);
										edge_normal.y = THIRD * (prevpos.x - nextpos.x);
										if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

										visc_contrib.x = over_m_n*(ita_par*ROCgradvx.dot(edge_normal)); // if we are looking at higher vz looking out, go up.
										visc_contrib.y = over_m_n*(ita_par*ROCgradvy.dot(edge_normal));
										visc_contrib.z = over_m_n*(ita_par*ROCgradvz.dot(edge_normal));

										// x -> x:
										if (iDimension == 0)
											eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.x; 
										// y -> y:
										if (iDimension == 1)
											eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.y; 
										// z -> z:
										if (iDimension == 2)
											eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2] += visc_contrib.z; 
									}
									else {
										if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
										{
											// run unmagnetised case (actually same as neutral ... )
											f64 Pi_zx, Pi_zy, Pi_xx, Pi_xy, Pi_yx, Pi_yy;

											Pi_xx = -ita_par*THIRD*(4.0*ROCgradvx.x - 2.0*ROCgradvy.y);
											Pi_xy = -ita_par*(ROCgradvx.y + ROCgradvy.x);
											Pi_yx = Pi_xy;
											Pi_yy = -ita_par*THIRD*(4.0*ROCgradvy.y - 2.0*ROCgradvx.x);
											Pi_zx = -ita_par*(ROCgradvz.x);
											Pi_zy = -ita_par*(ROCgradvz.y);

											f64_vec3 visc_contrib;
											f64_vec2 edge_normal;
											edge_normal.x = THIRD * (nextpos.y - prevpos.y);
											edge_normal.y = THIRD * (prevpos.x - nextpos.x);
											if (bLongi)
												edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

											visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
											visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
											visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);


											// OK let's be careful how we're going to do this.
											// In extremis.
											// Whole 3-vector affects 3-vector of residuals.
											// We are considering here the chance in z and how that affects residual xyz.
											// Therefore let's renumber.
											// eqns[iEqnOurs*3] = row of coeffs for our x residual
											// eqns[iEqnOurs*3+1] = row of coeffs for our y residual
											// eqns[iEqnOurs*3+2] = row of coeffs for our z residual

											// Note that x and y affected each other, but we don't nec have an effect from z

											if (iDimension < 2) {
												if (iDimension == 0) {
													// x->x
													eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.z;
													// x->y
													eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.z;
												} else {
													// y->x
													eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.z;
													// y->y
													eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.z;
												};
											} else {
												eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2] += visc_contrib.z;
											};
										}
										else {

											f64_vec3 unit_b, unit_perp, unit_Hall;
											f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
											{
												f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
												// these have to be alive at same time as 9 x partials
												// but we can make do with 3x partials
												// 2. Now get partials in magnetic coordinates 
												f64 omegamod;
												{
													f64_vec2 edge_normal;
													edge_normal.x = THIRD * (nextpos.y - prevpos.y);
													edge_normal.y = THIRD * (prevpos.x - nextpos.x);
													if (bLongi)
														edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

													f64 omegasq = omega_c.dot(omega_c);
													omegamod = sqrt(omegasq);
													unit_b = omega_c / omegamod;
													unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
													unit_perp = unit_perp / unit_perp.modulus();
													unit_Hall = unit_b.cross(unit_perp); // Note sign.
												}
												{
													f64_vec3 intermed;

													// use: d vb / da = b transpose [ dvi/dxj ] a
													// Prototypical element: a.x b.y dvy/dx
													// b.x a.y dvx/dy

													intermed.x = unit_b.dotxy(ROCgradvx);
													intermed.y = unit_b.dotxy(ROCgradvy);
													intermed.z = unit_b.dotxy(ROCgradvz); // == 0
													{
														f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

														dvb_by_db = unit_b.dot(intermed);
														dvperp_by_db = unit_perp.dot(intermed);
														dvHall_by_db = unit_Hall.dot(intermed); // == 0
														
														W_bb += 4.0*THIRD*dvb_by_db;
														W_bP += dvperp_by_db;
														W_bH += dvHall_by_db;     // nonzero
														W_PP -= 2.0*THIRD*dvb_by_db;
														W_HH -= 2.0*THIRD*dvb_by_db;
													}
													{
														f64 dvb_by_dperp, dvperp_by_dperp,
															dvHall_by_dperp;
														// Optimize by getting rid of different labels.

														intermed.x = unit_perp.dotxy(ROCgradvx);
														intermed.y = unit_perp.dotxy(ROCgradvy);
														intermed.z = unit_perp.dotxy(ROCgradvz);
														//intermed.z = unit_perp.dotxy(gradvz);

														dvb_by_dperp = unit_b.dot(intermed);
														dvperp_by_dperp = unit_perp.dot(intermed);
														dvHall_by_dperp = unit_Hall.dot(intermed); // 0

														W_bb -= 2.0*THIRD*dvperp_by_dperp;
														W_PP += 4.0*THIRD*dvperp_by_dperp;
														W_HH -= 2.0*THIRD*dvperp_by_dperp;
														W_bP += dvb_by_dperp;
														W_PH += dvHall_by_dperp;
													}
													{
														f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

														intermed.x = unit_Hall.dotxy(ROCgradvx);
														intermed.y = unit_Hall.dotxy(ROCgradvy);
														intermed.z = unit_Hall.dotxy(ROCgradvz);  // 0
																								 // These are all always 0 ! Because nothing changes in z direction.
														dvb_by_dHall = unit_b.dot(intermed);
														dvperp_by_dHall = unit_perp.dot(intermed);
														dvHall_by_dHall = unit_Hall.dot(intermed);
														// all 0

														W_bb -= 2.0*THIRD*dvHall_by_dHall;
														W_PP -= 2.0*THIRD*dvHall_by_dHall;
														W_HH += 4.0*THIRD*dvHall_by_dHall;
														W_bH += dvb_by_dHall;
														W_PH += dvperp_by_dHall;
													}
												}
												{
													f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

													Pi_b_b += -ita_par*W_bb;
													Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
													Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
													Pi_H_P += -ita_1*W_PH;
												}
												{
													f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
													Pi_P_b += -ita_2*W_bP;
													Pi_H_b += -ita_2*W_bH;
												}
												{
													f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
													Pi_P_P -= ita_3*W_PH;
													Pi_H_H += ita_3*W_PH;
													Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
												}
												{
													f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
													Pi_P_b += -ita_4*W_bH;
													Pi_H_b += ita_4*W_bP;
												}
											} // scope W

											f64 momflux_b, momflux_perp, momflux_Hall;
											{
												// Most efficient way: compute mom flux in magnetic coords
												f64_vec3 mag_edge;
												f64_vec2 edge_normal;
												edge_normal.x = THIRD * (nextpos.y - prevpos.y);
												edge_normal.y = THIRD * (prevpos.x - nextpos.x);
												if (bLongi)
													edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

												mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
												mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
												mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0

																													// So basically, Pi_H_H is not only 0, it is never used.

												momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
												momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
												momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
											}

											// ROC :
											f64_vec3 visc_contrib;
											visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
											visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
											visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
#if TESTVISCCOEFFY
											if ((TESTVISCCOEFFY) && (iDimension == 1) && (iEqnTheirs == 1)) 
												printf("Dimension %d iOurs %d iTheirs %d contribxyz %1.12E %1.12E %1.12E\n",
												iDimension, iEqnOurs, iEqnTheirs, visc_contrib.x, visc_contrib.y, visc_contrib.z);
#endif
											// Dimension 1 = change in y in EqnTheirs, effect on xyz of EqnOurs.
											// Probably won't need to analyze this.

											eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.x;
											eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.y; // contrib vz -> eps_y
											eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.z;

											//if (0) // (iVertex == VERTCHOSEN)) 
											//{
											//	printf("iEqn %d theirs %d izTri %d which %d Dim %d cont to x %1.10E  pop %d \n",
											//		iEqnOurs, iEqnTheirs, izTri[i], iWhich, iDimension, visc_contrib.x,
											//		(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension
											//		);
											//}

										};
									};
								};  // was ita_par == 0
							};
						}; // iWhich : prev, opp, next, self.
					}; // iDimension: x, y, z.
					prevpos = opppos;
					prev_v = opp_v;
					prev_x = opp_x;
					opppos = nextpos;
					opp_v = next_v;
					opp_x = next_x;
				}; // next i				
			}; // bSelected for this vertex			
		} else {
		//	memset(p_ROCMAR + iVertex + BEGINNING_OF_CENTRAL, 0, sizeof(f64_vec3));
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...
	info = p_info_minor[iMinor];	
	{
		long izNeighMinor[6];
		char szPBC[6];

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			bool bSelected = p_selectflag[iMinor];
			if (bSelected) {
				long iEqnOurs, iEqnTheirs;

				iEqnOurs = p_mapping_to_array[iMinor];

				memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
				memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
			
				short i = 0;
				short inext = i + 1; if (inext > 5) inext = 0;
				short iprev = i - 1; if (iprev < 0) iprev = 5;
				f64_vec3 prev_v, opp_v, next_v;
			//	f64 prev_x, opp_x, next_x;
				f64_vec2 prevpos, nextpos, opppos;

				if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
				{
					prev_v = shared_v[izNeighMinor[iprev] - StartMinor];
					prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				//	prev_x = shared_regr[izNeighMinor[iprev] - StartMinor];
				} else {
					if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						prev_v = shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
						prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					//	prev_x = shared_regr_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						prevpos = p_info_minor[izNeighMinor[iprev]].pos;

						if (iSpecies > 0) {
							v4 temp = p_vie_minor[izNeighMinor[iprev]];
							prev_v.x = temp.vxy.x;
							prev_v.y = temp.vxy.y;
							if (iSpecies == 1) {
								prev_v.z = temp.viz;
							}
							else {
								prev_v.z = temp.vez;
							};
						}
						else {
							prev_v = p_v_n[izNeighMinor[iprev]];
						}
					///	prev_x = p__x[izNeighMinor[iprev]];
					};
				};
				if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
					prevpos = Clockwise_d*prevpos;
					RotateClockwise(prev_v);
				}
				if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
					prevpos = Anticlockwise_d*prevpos;
					RotateAnticlockwise(prev_v);
				}

				if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
				{
					opp_v = shared_v[izNeighMinor[i] - StartMinor];
					opppos = shared_pos[izNeighMinor[i] - StartMinor];
				//	opp_x = shared_regr[izNeighMinor[i] - StartMinor];
				}
				else {
					if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						opp_v = shared_v_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
						opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				//		opp_x = shared_regr_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						opppos = p_info_minor[izNeighMinor[i]].pos;
						if (iSpecies > 0) {
							v4 temp = p_vie_minor[izNeighMinor[i]];
							opp_v.x = temp.vxy.x;
							opp_v.y = temp.vxy.y;
							if (iSpecies == 1) {
								opp_v.z = temp.viz;
							} else {
								opp_v.z = temp.vez;
							};
						}
						else {
							opp_v = p_v_n[izNeighMinor[i]];
						}
				//		opp_x = p__x[izNeighMinor[i]];
					};
				};
				if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
					opppos = Clockwise_d*opppos;
					RotateClockwise(opp_v);
				}
				if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
					opppos = Anticlockwise_d*opppos;
					RotateAnticlockwise(opp_v);
				}
				f64_vec3 omega_c;
				// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
				for (short i = 0; i < 6; i++)
				{
					inext = i + 1; if (inext > 5) inext = 0;
					iprev = i - 1; if (iprev < 0) iprev = 5;

					if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
					{
						next_v = shared_v[izNeighMinor[inext] - StartMinor];
						nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				//		next_x = shared_regr[izNeighMinor[inext] - StartMinor];
					}
					else {
						if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							next_v = shared_v_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
							nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
				//			next_x = shared_regr_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						}
						else {
							nextpos = p_info_minor[izNeighMinor[inext]].pos;
							if (iSpecies > 0) {
								v4 temp = p_vie_minor[izNeighMinor[inext]];
								next_v.x = temp.vxy.x;
								next_v.y = temp.vxy.y;
								if (iSpecies == 1) {
									next_v.z = temp.viz;
								}
								else {
									next_v.z = temp.vez;
								};
							}
							else {
								next_v = p_v_n[izNeighMinor[inext]];
							}
				//			next_x = p__x[izNeighMinor[inext]];
						};
					};
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextpos = Clockwise_d*nextpos;
						RotateClockwise(next_v);
					}
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
						nextpos = Anticlockwise_d*nextpos;
						RotateAnticlockwise(next_v);
					}


					bool bUsableSide = true;
					{
						f64 nu_theirs, ita_theirs;
						f64_vec2 opp_B(0.0, 0.0);
						// newly uncommented:
						if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
						{
							opp_B = shared_B[izNeighMinor[i] - StartMinor];
							nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
							ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];
						}
						else {
							if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							}
							else {
								opp_B = p_B_minor[izNeighMinor[i]].xypart();
								nu_theirs = p_nu_minor[izNeighMinor[i]];
								ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
							}
						}
						// GEOMETRIC ITA:
						if (ita_theirs == 0.0) bUsableSide = false;
						ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);

						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opp_B = Clockwise_d*opp_B;
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opp_B = Anticlockwise_d*opp_B;
						}
						nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
						omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
						if (iSpecies == 1) omega_c *= qoverMc;
						if (iSpecies == 2) omega_c *= qovermc;
					}


					f64 prev_x, opp_x, next_x, self_x;
					for (iDimension = 0; iDimension < 3; iDimension++) {
						for (int iWhich = 0; iWhich < 4; iWhich++) {
							prev_x = 0.0;
							opp_x = 0.0;
							next_x = 0.0;
							self_x = 0.0;
							long iTri;
							if (iWhich == 0) {
								prev_x = 1.0;
								iTri = izNeighMinor[iprev];
								//iEqnTheirs = p_mapping_to_array[izTri[iprev]];
							}
							if (iWhich == 1) {
								opp_x = 1.0;
								iTri = izNeighMinor[i];
							};
							if (iWhich == 2) {
								next_x = 1.0;
								iTri = izNeighMinor[inext];
							};
							if (iWhich == 3) self_x = 1.0;

							// Optimize later.

							// ins-ins triangle traffic:

							// is iWhich's point selected? If not, pass over this loop.
							if (((iWhich < 3) && (p_selectflag[iTri]))
								||
								(iWhich == 3))
							{
								if (iWhich < 3) {
									iEqnTheirs = p_mapping_to_array[iTri];
								}
								else {
									iEqnTheirs = iEqnOurs;
								};

								bool bLongi = false;

#ifdef INS_INS_NONE
								if (info.flag == CROSSING_INS) {
									char flag = p_info_minor[izNeighMinor[i]].flag;
									if (flag == CROSSING_INS)
										bUsableSide = 0;
								}
								if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
									bLongi = true;
#else 
								//#ifdef INS_INS_LONGI
								if (info.flag == CROSSING_INS) {
									char flag = p_info_minor[izNeighMinor[i]].flag;
									if (flag == CROSSING_INS)
										bLongi = true;
								}
								if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
									bLongi = true;
								// Use this in this case as flag for reconstructing edge normal to where
								// it only stays within triangle!
								//#else


								//#endif
#endif

								f64_vec2 ROCgradvx(0.0,0.0), ROCgradvy(0.0,0.0), ROCgradvz(0.0,0.0);
								if (bUsableSide)
								{
#ifdef INS_INS_3POINT
									if (TestDomainPos(prevpos) == false)
									{
										if (iDimension == 0)
											ROCgradvx = GetGradientDBydBeta_3Point(
											info.pos, nextpos, opppos,
											shared_v[threadIdx.x].x, next_v.x, opp_v.x,
											self_x, next_x, opp_x
										);
										if (iDimension == 1)
										ROCgradvy = GetGradientDBydBeta_3Point(
											info.pos, nextpos, opppos,
											shared_v[threadIdx.x].y, next_v.y, opp_v.y,
											self_x, next_x, opp_x
										);
										
										if (iDimension == 2) {
											ROCgradvz = GetGradientDBydBeta_3Point(
												info.pos, nextpos, opppos,
												shared_v[threadIdx.x].z, next_v.z, opp_v.z,
												self_x, next_x, opp_x
											);
										};
#if TESTVISCCOEFF
										if (TESTVISCCOEFF //&& (iWhich == 3)
											&& (iDimension == 0)) {
											printf("iMinor %d i %d %d : v selfnextopp %1.9E %1.9E %1.9E x %1.2E %1.2E %1.2E ROCgradvx %1.12E %1.12E\n", 
												iMinor, i, izNeighMinor[i],
												shared_v[threadIdx.x].x, next_v.x, opp_v.x,
												self_x, next_x, opp_x,
												ROCgradvx.x, ROCgradvx.y
											);
										}
										if (TESTVISCCOEFF //&& (iWhich == 3)
											&& (iDimension == 1)) {
											printf("iMinor %d i %d %d : vy selfnextopp %1.9E %1.9E %1.9E x %1.2E %1.2E %1.2E ROCgradvy %1.12E %1.12E\n",
												iMinor, i, izNeighMinor[i],
												shared_v[threadIdx.x].y, next_v.y, opp_v.y,
												self_x, next_x, opp_x,
												ROCgradvy.x, ROCgradvy.y
											);
										}
#endif
									} else {
										if (TestDomainPos(nextpos) == false) {
											if (iDimension == 0)
											ROCgradvx = GetGradientDBydBeta_3Point(
												prevpos, info.pos, opppos,
												prev_v.x, shared_v[threadIdx.x].x, opp_v.x, prev_x, self_x, opp_x
											);
											if (iDimension == 1)
											ROCgradvy = GetGradientDBydBeta_3Point(
												prevpos, info.pos, opppos,
												prev_v.y, shared_v[threadIdx.x].y, opp_v.y, prev_x, self_x, opp_x
											);
											if (iDimension == 2)
											ROCgradvz = GetGradientDBydBeta_3Point(
												prevpos, info.pos, opppos,
												prev_v.z, shared_v[threadIdx.x].z, opp_v.z,
												prev_x, self_x, opp_x
											);
#if TESTVISCCOEFF
											if (TESTVISCCOEFF //&& (iWhich == 3)
												&& (iDimension == 0)) {
												printf("iMinor %d i %d %d : v prevselfopp %1.9E %1.9E %1.9E x %1.2E %1.2E %1.2E ROCgradvx %1.12E %1.12E\n",
													iMinor, i, izNeighMinor[i],
													prev_v.x, shared_v[threadIdx.x].x, opp_v.x,
													prev_x, self_x, opp_x,
													ROCgradvx.x, ROCgradvx.y
												);
											}

											if (TESTVISCCOEFF //&& (iWhich == 3)
												&& (iDimension == 1)) {
												printf("iMinor %d i %d %d : vy prevselfopp %1.9E %1.9E %1.9E x %1.2E %1.2E %1.2E ROCgradvy %1.12E %1.12E\n",
													iMinor, i, izNeighMinor[i],
													prev_v.y, shared_v[threadIdx.x].y, opp_v.y,
													prev_x, self_x, opp_x,
													ROCgradvy.x, ROCgradvy.y
												);
											}
#endif
										} else {
											if (iDimension == 0) {
												
													ROCgradvx = GetGradientDBydBeta(
														prevpos, info.pos, nextpos, opppos,
														prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
														prev_x, self_x, next_x, opp_x
													);
											}
											if (iDimension == 1)

													ROCgradvy = GetGradientDBydBeta(
														prevpos, info.pos, nextpos, opppos,
														prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
														prev_x, self_x, next_x, opp_x
													);
											if (iDimension == 2)
												ROCgradvz = GetGradientDBydBeta(
													prevpos, info.pos, nextpos, opppos,
													prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z,
													prev_x, self_x, next_x, opp_x
												);
#if TESTVISCCOEFF																				
											if (TESTVISCCOEFF //&& (iWhich == 3)
												&& (iDimension == 1)) {
												printf("iMinor %d i %d %d : v psno %1.9E %1.9E %1.9E %1.9E x %1.2E %1.2E %1.2E %1.2E ROCgradvy %1.12E %1.12E\n",
													iMinor, i, izNeighMinor[i],
													prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y,
													prev_x, self_x, next_x, opp_x,
													ROCgradvy.x, ROCgradvy.y
												);
											};
#endif
										};
									};

#else
									if (bLongi)
									{
										// One of the sides is dipped under the insulator -- set transverse deriv to 0.
										// Bear in mind we are looking from a vertex into a tri, it can be ins tri.
										if (0)//iMinor == CHOSEN) 
											printf("longi. x %1.8E %1.8E\n", opp_x, shared_regr[threadIdx.x]);

										ROCgradvz = (opp_x - self_x)*(opppos - info.pos) /
											(opppos - info.pos).dot(opppos - info.pos);
									}
									else {

										if (0)//(iMinor == CHOSEN)) 
										{
											printf("%d v %1.8E %1.8E %1.8E %1.8E x %1.8E %1.8E %1.8E %1.8E\n",
												CHOSEN, prev_v, shared_vz[threadIdx.x], next_v, opp_v,
												prev_x, self_x, next_x, opp_x);
											ROCgradvz = GetGradientDBydBetaDebug(
												prevpos, info.pos, nextpos, opppos,
												prev_v, shared_vz[threadIdx.x], next_v, opp_v,
												prev_x, self_x, next_x, opp_x
											);
										}
										else {

											ROCgradvz = GetGradientDBydBeta(
												prevpos, info.pos, nextpos, opppos,
												prev_v, shared_vz[threadIdx.x], next_v, opp_v,
												prev_x, self_x, next_x, opp_x
											);
										};
									};
#endif

#ifdef INS_INS_NONE
									if (info.flag == CROSSING_INS) {
										char flag = p_info_minor[izNeighMinor[i]].flag;
										if (flag == CROSSING_INS) {
											// just set it to 0.
											bUsableSide = false;
											ROCgradvz.x = 0.0;
											ROCgradvz.y = 0.0;
										};
									};
#endif
								};


#if (TESTVISCCOEFFS)
									printf("%d ROCgradvz %1.8E %1.8E bUsableSide %d\n",
										iMinor, ROCgradvz.x, ROCgradvz.y, bUsableSide ? 1 : 0);
#endif

								if (bUsableSide) {
									if (iSpecies == 0) {
										f64_vec3 visc_contrib;
										f64_vec2 edge_normal;
										edge_normal.x = THIRD * (nextpos.y - prevpos.y);
										edge_normal.y = THIRD * (prevpos.x - nextpos.x);

										if (bLongi) {
											edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

											//	This is where we need to think harder in all routines ---
											//	in the case that nextdoor is CROSSING_INS, we want to stop at edge of triangle

											// use bLongi as flag.
										};

										visc_contrib.x = over_m_n*(ita_par*ROCgradvx.dot(edge_normal)); // if we are looking at higher vz looking out, go up.
										visc_contrib.y = over_m_n*(ita_par*ROCgradvy.dot(edge_normal));
										visc_contrib.z = over_m_n*(ita_par*ROCgradvz.dot(edge_normal));
										if (iDimension == 0)
											eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.x;
										if (iDimension == 1)
											eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.y;
										if (iDimension == 2)
											eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2] += visc_contrib.z; // +2,+2 --> vz to eps_z
									//	if (iMinor == 38295) printf("38295: iEqnOurs %d iEqnTheirs %d filling in %d with %1.8E \n",
								//			iEqnOurs, iEqnTheirs, (iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2, visc_contribz);

									} else {
										if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
										{
											// run unmagnetised case
											f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;
											Pi_xx = -ita_par*THIRD*(4.0*ROCgradvx.x - 2.0*ROCgradvy.y);
											Pi_xy = -ita_par*(ROCgradvx.y + ROCgradvy.x);
											Pi_yx = Pi_xy;
											Pi_yy = -ita_par*THIRD*(4.0*ROCgradvy.y - 2.0*ROCgradvx.x);
											Pi_zx = -ita_par*(ROCgradvz.x);
											Pi_zy = -ita_par*(ROCgradvz.y);

											f64_vec2 edge_normal;
											edge_normal.x = THIRD * (nextpos.y - prevpos.y);
											edge_normal.y = THIRD * (prevpos.x - nextpos.x);

											if (bLongi) {
												edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
											};
											f64_vec3 visc_contrib;
											visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
											visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
											visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

#if (TESTVISCCOEFFS)
												printf("CHOSEN unmag visc_contrib.z %1.8E \n", visc_contrib.z);
#endif
											if (iDimension == 0) {
												eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.x;
												eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 0] += visc_contrib.y;
											};
											if (iDimension == 1) {
												eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.x;
												eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 1] += visc_contrib.y;
											};
											if (iDimension == 2)
												eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2] += visc_contrib.z;

											//if (TESTVISCCOEFFS) printf("38295: iEqnOurs %d iEqnTheirs %d filling in %d with %1.8E \n",
											//	iEqnOurs, iEqnTheirs, (iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2, visc_contribz);

											// So we are saying if edge_normal.x > 0 and gradviz.x > 0
											// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
										}
										else {
											f64_vec3 unit_b, unit_perp, unit_Hall;
											f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
											{
												f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0;
												// these have to be alive at same time as 9 x partials
												// but we can make do with 3x partials
												// 2. Now get partials in magnetic coordinates 
												f64 omegamod;
												{
													f64_vec2 edge_normal;
													edge_normal.x = THIRD * (nextpos.y - prevpos.y);
													edge_normal.y = THIRD * (prevpos.x - nextpos.x);

													if (bLongi) {
														edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
													};
													f64 omegasq = omega_c.dot(omega_c);
													omegamod = sqrt(omegasq);
													unit_b = omega_c / omegamod;
													unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
													unit_perp = unit_perp / unit_perp.modulus();
													unit_Hall = unit_b.cross(unit_perp); // Note sign.
												}
												{
													f64_vec3 intermed;

													// use: d vb / da = b transpose [ dvi/dxj ] a
													// Prototypical element: a.x b.y dvy/dx
													// b.x a.y dvx/dy
													intermed.x = unit_b.dotxy(ROCgradvx);
													intermed.y = unit_b.dotxy(ROCgradvy);
													intermed.z = unit_b.dotxy(ROCgradvz);
													{
														f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

														dvb_by_db = unit_b.dot(intermed);
														dvperp_by_db = unit_perp.dot(intermed);
														dvHall_by_db = unit_Hall.dot(intermed); // 0
																								// This actually is always 0 because vHall isn't changing.
														W_bb += 4.0*THIRD*dvb_by_db;
														W_bP += dvperp_by_db;
														W_bH += dvHall_by_db;
														W_PP -= 2.0*THIRD*dvb_by_db;
														W_HH -= 2.0*THIRD*dvb_by_db;

													//	if (0) printf("ROC dvb/db %1.9E  dvperp/db %1.9E  dvHall/db %1.9E\n",
													//		dvb_by_db, dvperp_by_db, dvHall_by_db);
													}
													{
														f64 dvb_by_dperp, dvperp_by_dperp,
															dvHall_by_dperp;
														// Optimize by getting rid of different labels.

														intermed.x = unit_perp.dotxy(ROCgradvx);
														intermed.y = unit_perp.dotxy(ROCgradvy);
														intermed.z = unit_perp.dotxy(ROCgradvz);

														dvb_by_dperp = unit_b.dot(intermed);
														dvperp_by_dperp = unit_perp.dot(intermed);
														dvHall_by_dperp = unit_Hall.dot(intermed); // 0

														//if (0) printf("dvb/dP %1.9E  dvperp/dP %1.9E  dvHall/dP %1.9E\n",
														//	dvb_by_dperp, dvperp_by_dperp, dvHall_by_dperp);
														W_bb -= 2.0*THIRD*dvperp_by_dperp;
														W_PP += 4.0*THIRD*dvperp_by_dperp;
														W_HH -= 2.0*THIRD*dvperp_by_dperp;
														W_bP += dvb_by_dperp;
														W_PH += dvHall_by_dperp;
													}
													{
														f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

														intermed.x = unit_Hall.dotxy(ROCgradvx);
														intermed.y = unit_Hall.dotxy(ROCgradvy);
														intermed.z = unit_Hall.dotxy(ROCgradvz); // 0

														dvb_by_dHall = unit_b.dot(intermed);
														dvperp_by_dHall = unit_perp.dot(intermed);
														dvHall_by_dHall = unit_Hall.dot(intermed);

														//if (0) //TESTVISCCOEFFS) 
														//	printf("dvb/dH %1.9E  dvperp/dH %1.9E  dvHall/dH %1.9E\n",
														//	dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall);

														W_bb -= 2.0*THIRD*dvHall_by_dHall;
														W_PP -= 2.0*THIRD*dvHall_by_dHall;
														W_HH += 4.0*THIRD*dvHall_by_dHall;
														W_bH += dvb_by_dHall;
														W_PH += dvperp_by_dHall;
													}
												}

												// can bring back braces but are they worth it!

												f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

												Pi_b_b += -ita_par*W_bb;
												Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
												Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
												Pi_H_P += -ita_1*W_PH;

												f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
												Pi_P_b += -ita_2*W_bP;
												Pi_H_b += -ita_2*W_bH;

												f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
												Pi_P_P -= ita_3*W_PH;
												Pi_H_H += ita_3*W_PH;
												Pi_H_P += 0.5*ita_3*(W_PP - W_HH);

												f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
												Pi_P_b += -ita_4*W_bH;
												Pi_H_b += ita_4*W_bP;

												//if (0) // TESTVISCCOEFFS) 
												//{
												//	printf(
												//		"ita_par %1.9E ita1 %1.9E ita2 %1.9E ita3 %1.9E ita4 %1.9E\n"
												//		"W_bb %1.9E W_bP %1.9E W_PP %1.9E W_HH %1.9E W_bH %1.9E W_PH %1.9E\n"
												//		"Pi bb %1.9E PP %1.9E Pb %1.9E HH %1.9E Hb %1.9E HP %1.9E\n",
												//		ita_par, ita_1, ita_2, ita_3, ita_4,
												//		W_bb, W_bP, W_PP, W_HH, W_bH, W_PH,
												//		Pi_b_b, Pi_P_P, Pi_P_b, Pi_H_H, Pi_H_b, Pi_H_P);
												//};
											} // scope W

											f64 momflux_b, momflux_perp, momflux_Hall;
											{
												// Most efficient way: compute mom flux in magnetic coords
												f64_vec3 mag_edge;
												f64_vec2 edge_normal;
												edge_normal.x = THIRD * (nextpos.y - prevpos.y);
												edge_normal.y = THIRD * (prevpos.x - nextpos.x);

												if (bLongi) {
													edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
												};

												mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
												mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
												mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // 0

												momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
												momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
												momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);

												//if (0) //TESTVISCCOEFFS) 
												//	printf("mag_edge bPH %1.9E %1.9E %1.9E momflux bPH %1.9E %1.9E %1.9E\n"
												//	"mf_b = -Pibb edge_b-PiPb edge_P ; mf_H = -Pi_Hb edge_b-Pi_HP edge_P\n"
												//	,
												//	mag_edge.x, mag_edge.y, mag_edge.z, momflux_b, momflux_perp, momflux_Hall);
											}

											// ROC :
											f64_vec3 visc_contrib;
											visc_contrib.x = over_m_s*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
											visc_contrib.y = over_m_s*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
											visc_contrib.z = over_m_s*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

											// Mostly want to know own impact on self.

											eqns[(iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.x;
											eqns[(iEqnOurs * 3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.y; // contrib vz -> eps_y
											eqns[(iEqnOurs * 3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension] += visc_contrib.z;
											
											// For some reason we are not obtaining 
										//	if ((iEqnOurs * 3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + iDimension == 97) {
										//		printf("Element 97 augmented: eqns[97] %1.10E : added visc_contrib.x %1.10E\n",
										//			eqns[97], visc_contrib.x);}

											
											//if (iMinor == 38295)
											//{
											//	printf("38295: iEqnOurs %d iEqnTheirs %d filling in %d with %1.8E (x) \n",
											//		iEqnOurs, iEqnTheirs, (iEqnOurs*3 + 0) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2, visc_contrib.x);
											//	printf("38295: iEqnOurs %d iEqnTheirs %d filling in %d with %1.8E (y) \n",
											//		iEqnOurs, iEqnTheirs, (iEqnOurs*3 + 1) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2, visc_contrib.y); 
											//	printf("38295: iEqnOurs %d iEqnTheirs %d filling in %d with %1.8E (z) \n",
											//			iEqnOurs, iEqnTheirs, (iEqnOurs*3 + 2) * 3 * EQNS_TOTAL + iEqnTheirs * 3 + 2, visc_contrib.z);
											//};

											// Missing a *3
											// This way of organizing really sucked --- but ultimately we have to spit out a matrix, that's the problem.
											// Would be better if we could have had matrix of vectors and written a 3-vector here.
										};
									};

								}; // bUsableSide
							}; // point for iWhich had selectflag
						}; // next iWhich
					}; // next iDimension
					prevpos = opppos;
					prev_v = opp_v;
					prev_x = opp_x;
					opppos = nextpos;
					opp_v = next_v;
					opp_x = next_x;
				}; // next i

			} // bSelected this point

			//if (iMinor == CHOSEN) printf("CHOSEN ROCMAR.z %1.8E \n", ownrates_visc.z);

			// UPDATE:
		//	memcpy(&(p_ROCMAR[iMinor]), &(ownrates_visc), sizeof(f64_vec3));

		} else {
		//	memset(&(p_ROCMAR[iMinor]), 0, sizeof(f64_vec3));
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope
}

__global__ void kernelSimpleAverage
(f64 * __restrict__ p_update,
	f64 * __restrict__ p_avg)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;
	f64 s = p_update[index];
	s = 0.5*s + 0.5*p_avg[index];
	p_update[index] = s;
}

__global__ void kernelPutative_v_from_matrix(
	f64 * __restrict__ p_ita, // if 0 can avoid loading regrs
	v4 * __restrict__ p_vie_2, 
	v4 * __restrict__ p_vie_, 
	f64_vec2 * __restrict__ p_regr2, 
	f64 * __restrict__ p_regr_iz, 
	f64 * __restrict__ p_regr_ez)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	if (p_ita[iMinor] > 0.0) {

		// it's +1 the first 3, -1 the next one. Pretty sure!
		// check coeffs though.
		// 	;
		v4 temp;
		memcpy(&temp, &(p_vie_[iMinor]), sizeof(v4));
		
		temp.vxy += 0.045*(p_regr2[iMinor] + p_regr2[iMinor + NMINOR] + p_regr2[iMinor + 2 * NMINOR]);// -p_regr2[iMinor + 3 * NMINOR];
		temp.viz += 0.045*(p_regr_iz[iMinor] + p_regr_iz[iMinor + NMINOR] + p_regr_iz[iMinor + 2 * NMINOR]);// -p_regr_iz[iMinor + 3 * NMINOR];
		temp.vez += 0.045*(p_regr_ez[iMinor] + p_regr_ez[iMinor + NMINOR] + p_regr_ez[iMinor + 2 * NMINOR]);// -p_regr_ez[iMinor + 3 * NMINOR];
		
//		printf("iMinor %d v_new %1.10E %1.10E %1.10E %1.10E v_old %1.10E %1.10E %1.10E %1.10E p_regrx %1.10E %1.10E %1.10E\n",
//			iMinor, temp.vxy.x, temp.vxy.y, temp.viz, temp.vez, p_vie_[iMinor].vxy.x, p_vie_[iMinor].vxy.y,
//			p_vie_[iMinor].viz, p_vie_[iMinor].vez, p_regr2[iMinor].x, p_regr2[iMinor + NMINOR].x, p_regr2[iMinor + 2 * NMINOR].x);
		// +++-

		memcpy(&(p_vie_2[iMinor]), &temp, sizeof(v4));
	} else {
		
		memcpy(&(p_vie_2[iMinor]), &(p_vie_[iMinor]), sizeof(v4));
	};
}

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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	short iOurEqn;
	long iNeigh;

	// There is some chance that a vector3 lives in registry not local memory?
	// Google doesn't reveal whether it does or not!
	// Possibly this is something making all the code slow : 
	// How many of these vector3 are not in registers??

	// If we have only 16K per 256 threads, that's only 8 doubles per thread. Watch out.
	// Need to do with vectors instead. Are they allocated in registers?

	// eps = v - v_k - h MAR / N

	bool bSelect = p_bSelectFlag[iMinor];
	short neigh_len;

	if (bSelect)
	{
		structural info = p_info_minor[iMinor];
		
		// Try using CUDA's built in struct types:
		double3 ionfluxcoeff3;
		double3 elecfluxcoeff3;
		double4 coeff; // careful as we also have to load izTri or whichever.

		long izNeigh[MAXNEIGH];
		//structural info = p_info_minor[iMinor];
		//if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		//	|| (info.flag == CROSSING_INS)) // ?
		//	note this is different .. what to do?

			// we want to say, 
			// eps = v_own - constant(not important, goes on RHS)
			// - hsub (m_i/(mi+me))*ionflux

		iOurEqn = p_eqn_index[iMinor];
		f64_vec2 epsxy = p_eps_xy[iMinor];
		double4 temp;
		temp.x = -epsxy.x;// , sizeof(f64) * 2);
		temp.y = -epsxy.y;
		temp.z = -p_eps_iz[iMinor];
		temp.w = -p_eps_ez[iMinor];
		
		memcpy(&(pRHS[iOurEqn * 4]), &temp, sizeof(f64) * 4);
		
		// we only want to do this for coeffs in the neighbour list..
		// too many elements otherwise

		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;
		f64 factor_i = -hsub*(m_ion / ((m_ion + m_e)*N));
		f64 factor_e = -hsub*(m_e / ((m_ion + m_e)*N));
		f64 minus_hsub_over_N = -hsub / N;
		short iTheirIndex;
		//	epsilon.vxy = vie.vxy - vie_k.vxy
		//		- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
		//		((m_ion + m_e)*N));
		//	epsilon.viz = vie.viz - vie_k.viz - hsub*(MAR_ion.z / N);
		//	epsilon.vez = vie.vez - vie_k.vez - hsub*(MAR_elec.z / N);

			// Can we explain why it's 3 and not 4 ? -- oops
			// Our own effect:
			// contribution to x residual:
		memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3) + 3 * iOurEqn]),
			sizeof(double3)); // half a bus!
		// 3*EQNS_TOTAL*(iOurEqn*3) is the start of row of our epsilon_x.
		memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3) + 3 * iOurEqn]),
			sizeof(double3));

		//Think carefully about meanings. We propose a change in vy. This changes
		//both elec px flux and ion px flux, which in turn can affect this px residual.

		coeff.x = factor_i*ionfluxcoeff3.x + factor_e*elecfluxcoeff3.x;
		coeff.y = factor_i*ionfluxcoeff3.y + factor_e*elecfluxcoeff3.y;

		//We propose a change in viz. This affects ion pz flux only.
		coeff.z = factor_i*ionfluxcoeff3.z;
		coeff.w = factor_e*elecfluxcoeff3.z;

		//Here add 1 or something;
		
		coeff.x += 1.0;
		// Note that whereas ion flux was 3 dimensions contributing to 3 dimensions of flux,
		// the equations are 4 dimensions contributing to 4 dimensions of residual.

#ifdef SQRTNV
		f64 sqrtN = sqrt(N);
		coeff.x *= sqrtN;
		coeff.y *= sqrtN;
		coeff.z *= sqrtN;
		coeff.w *= sqrtN;

		// d eps_v / dv was given;
		// We want d (sqrtN eps_v / dv) because we still make a proposed regressor for moving v.

#endif
		memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4) + 4 * iOurEqn]), &coeff, sizeof(double4));

		// Contribs to y residual:
		memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 1) + 3 * iOurEqn]),
			sizeof(double3)); // half a bus!
		memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3+1) + 3 * iOurEqn]),
			sizeof(double3));
		coeff.x = factor_i*ionfluxcoeff3.x + factor_e*elecfluxcoeff3.x;
		coeff.y = factor_i*ionfluxcoeff3.y + factor_e*elecfluxcoeff3.y;
		coeff.z = factor_i*ionfluxcoeff3.z;
		coeff.w = factor_e*elecfluxcoeff3.z;
		//Here add 1 :
		coeff.y += 1.0;

#ifdef SQRTNV
		coeff.x *= sqrtN;
		coeff.y *= sqrtN;
		coeff.z *= sqrtN;
		coeff.w *= sqrtN;
#endif
		// Note that whereas ion flux was 3 dimensions contributing to 3 dimensions of flux,
		// the equations are 4 dimensions contributing to 4 dimensions of residual.
		memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 1) + 4 * iOurEqn]), &coeff, sizeof(double4));

		//if (TEST_EPSILON_Y_IMINOR) printf("%d y->y coeff.y %1.14E ionfluxcoeff3.y %1.14E elecfluxcoeff3.y %1.13E factor_i %1.11E factor_e %1.11E\n",
		//	iMinor, coeff.y, ionfluxcoeff3.y, elecfluxcoeff3.y, 
		//	factor_i, factor_e
		//	//minus_hsub_over_N, 4 * EQNS_TOTAL*(iOurEqn * 4 + 1) + 4 * iOurEqn + 1
		//);
		//if (TEST_EPSILON_Y_IMINOR) printf("%d ez->y coeff.w %1.14E elecfluxcoeff3.z %1.13E minus_hsub_over_N %1.11E %d\n",
		//	iMinor, coeff.w, elecfluxcoeff3.z, minus_hsub_over_N, 4 * EQNS_TOTAL*(iOurEqn * 4 + 1) + 4 * iOurEqn + 3
		//);

		// Contribs to iz residual:
		memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 2) + 3 * iOurEqn]),
			sizeof(double3)); // half a bus!
		// MAR_ion_z is whose derivative is populated in ionfluxcoeff3 WRT vx
		coeff.x = minus_hsub_over_N*ionfluxcoeff3.x;
		coeff.y = minus_hsub_over_N*ionfluxcoeff3.y;
		coeff.z = minus_hsub_over_N*ionfluxcoeff3.z; 
		// z component is viz -> eps iz , there is no vez->eps iz
		coeff.w = 0.0;
		//Here add 1 or something;
		coeff.z += 1.0;
#ifdef SQRTNV
		coeff.x *= sqrtN;
		coeff.y *= sqrtN;
		coeff.z *= sqrtN;
		coeff.w *= sqrtN;
#endif
		// Note that whereas ion flux was 3 dimensions contributing to 3 dimensions of flux,
		// the equations are 4 dimensions contributing to 4 dimensions of residual.
		memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 2) + 4 * iOurEqn]), &coeff, sizeof(double4));

		// Contribs to ez residual:
		memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3+2) + 3 * iOurEqn]),
			sizeof(double3));
		coeff.x = minus_hsub_over_N*elecfluxcoeff3.x;
		coeff.y = minus_hsub_over_N*elecfluxcoeff3.y;
		coeff.z = 0.0;
		coeff.w = minus_hsub_over_N*elecfluxcoeff3.z; // z component is viz -> eps iz , there is no vez->eps iz
													 //Here add 1 or something;
		coeff.w += 1.0;

#ifdef SQRTNV
		coeff.x *= sqrtN;
		coeff.y *= sqrtN;
		coeff.z *= sqrtN;
		coeff.w *= sqrtN;
#endif

		// Note that whereas ion flux was 3 dimensions contributing to 3 dimensions of flux,
		// the equations are 4 dimensions contributing to 4 dimensions of residual.
		memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 3) + 4 * iOurEqn]), &coeff, sizeof(double4));

		if (iMinor < BEGINNING_OF_CENTRAL) {
			memcpy(izNeigh, p_Indexneigh_tri + 6 * iMinor, sizeof(long) * 6);
			neigh_len = 6;
		} else {
			memcpy(izNeigh, p_izTri_vert + MAXNEIGH*(iMinor - BEGINNING_OF_CENTRAL), sizeof(long)*MAXNEIGH);
			neigh_len = info.neigh_len;
		};

		// Contributions to residual i from izNeigh[j] :
		for (int j = 0; j < neigh_len; j++)
		{
			iNeigh = izNeigh[j];
			if (p_bSelectFlag[iNeigh]) {

				// Note: there are 3 rows and 3 columns per cell.
				iTheirIndex = p_eqn_index[iNeigh];

				// contribution to x residual:
				memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3) + 3 * iTheirIndex]),
					sizeof(double3)); // half a bus!
				memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3) + 3 * iTheirIndex]),
					sizeof(double3));
				coeff.x = factor_i*ionfluxcoeff3.x + factor_e*elecfluxcoeff3.x;
				coeff.y = factor_i*ionfluxcoeff3.y + factor_e*elecfluxcoeff3.y;
				coeff.z = factor_i*ionfluxcoeff3.z;
				coeff.w = factor_e*elecfluxcoeff3.z;

#ifdef SQRTNV
				coeff.x *= sqrtN;
				coeff.y *= sqrtN;
				coeff.z *= sqrtN;
				coeff.w *= sqrtN;
#endif
				memcpy(&(eqns[4 * EQNS_TOTAL*iOurEqn*4 + 4 * iTheirIndex]), &coeff, sizeof(double4));

				// The y residual:
				memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 1) + 3 * iTheirIndex]),
					sizeof(double3)); // half a bus!
				memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 1) + 3 * iTheirIndex]),
					sizeof(double3));
				coeff.x = factor_i*ionfluxcoeff3.x + factor_e*elecfluxcoeff3.x;
				coeff.y = factor_i*ionfluxcoeff3.y + factor_e*elecfluxcoeff3.y;
				coeff.z = factor_i*ionfluxcoeff3.z ;
				coeff.w = factor_e*elecfluxcoeff3.z;

#ifdef SQRTNV
				coeff.x *= sqrtN;
				coeff.y *= sqrtN;
				coeff.z *= sqrtN;
				coeff.w *= sqrtN;
#endif
				memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 1) + 4 * iTheirIndex]), &coeff, sizeof(double4));


				memcpy(&ionfluxcoeff3, &(p_ionmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 2) + 3 * iTheirIndex]),
					sizeof(double3)); // half a bus!
				coeff.x = minus_hsub_over_N*ionfluxcoeff3.x;
				coeff.y = minus_hsub_over_N*ionfluxcoeff3.y;
				coeff.z = minus_hsub_over_N*ionfluxcoeff3.z;
				coeff.w = 0.0;

#ifdef SQRTNV
				coeff.x *= sqrtN;
				coeff.y *= sqrtN;
				coeff.z *= sqrtN;
				coeff.w *= sqrtN;
#endif
				memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 2) + 4 * iTheirIndex]), &coeff, sizeof(double4));
				
				memcpy(&elecfluxcoeff3, &(p_elecmomflux[3 * EQNS_TOTAL*(iOurEqn*3 + 2) + 3 * iTheirIndex]),
					sizeof(double3));
				coeff.x = minus_hsub_over_N*elecfluxcoeff3.x;
				coeff.y = minus_hsub_over_N*elecfluxcoeff3.y;
				coeff.z = 0.0;
				coeff.w = minus_hsub_over_N*elecfluxcoeff3.z;

#ifdef SQRTNV
				coeff.x *= sqrtN;
				coeff.y *= sqrtN;
				coeff.z *= sqrtN;
				coeff.w *= sqrtN;
#endif
				memcpy(&(eqns[4 * EQNS_TOTAL*(iOurEqn*4 + 3) + 4 * iTheirIndex]), &coeff, sizeof(double4));

			} // was this neigh selected.
		}; // next j
	}; // bSelect
//
//
//	Alternative: could do least squares for our own and surrounding
//		by including more eqns ..
//		think then we always might as well populate coeffs;
//				 the choice is, do we want to leave outer coeffs = 0
//					 just because they affect things outside
//
//			I think stick with the plan where we've got a square matrix.
//			And post-hoc a regression.
//
		
} 

__global__ void Richardson_Divide_by_sqrtsqrtN(
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iMinor];
	f64_vec2 regr2;
	f64 regr_iz, regr_ez;
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| ((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		f64 oversqrtN = 1.0 / sqrt(sqrt(p_n_minor[iMinor].n*p_AreaMinor[iMinor]));
		regr2 = p_eps_xy[iMinor] * oversqrtN;
		regr_iz = p_eps_iz[iMinor] * oversqrtN;
		regr_ez = p_eps_ez[iMinor] * oversqrtN;

	}
	else {
		regr2.x = 0.0;
		regr2.y = 0.0;
		regr_iz = 0.0;
		regr_ez = 0.0;
	};
	p_regr2[iMinor] = regr2;
	p_regr_iz[iMinor] = regr_iz;
	p_regr_ez[iMinor] = regr_ez;
}


__global__ void Richardson_Divide_by_sqrtN(
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	nvals * __restrict__ p_n_minor, 
	f64 * __restrict__ p_AreaMinor
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iMinor];
	f64_vec2 regr2;
	f64 regr_iz, regr_ez;
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| ((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		f64 oversqrtN = 1.0/sqrt(p_n_minor[iMinor].n*p_AreaMinor[iMinor]);
		regr2 = p_eps_xy[iMinor] * oversqrtN;
		regr_iz = p_eps_iz[iMinor] * oversqrtN;
		regr_ez = p_eps_ez[iMinor] * oversqrtN;
		
	} else {
		regr2.x = 0.0;
		regr2.y = 0.0;
		regr_iz = 0.0;
		regr_ez = 0.0;
	};
	p_regr2[iMinor] = regr2;
	p_regr_iz[iMinor] = regr_iz;
	p_regr_ez[iMinor] = regr_ez;
}


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
) // easy way to put it in constant memory
{

	__shared__ f64_vec3 shared_v[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	__shared__ f64_vec3 shared_v_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Might as well add to shared then, if there are spills (surely there are?)

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_tens2 deriv_xy;
	f64 deriv_z;
	f64 deriv_3_xz,
		deriv_3_yz,
		deriv_3_zx,
		deriv_3_zy;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	if (iSpecies > 0) 
	{
		v4 temp = p_vie_minor[iMinor];
		shared_v[threadIdx.x].x = temp.vxy.x;
		shared_v[threadIdx.x].y = temp.vxy.y;
		if (iSpecies == 1) {
			shared_v[threadIdx.x].z = temp.viz;
		}
		else {
			shared_v[threadIdx.x].z = temp.vez;
		};
	} else {

		shared_v[threadIdx.x] = p_v_n[iMinor];
	}
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{

			if (iSpecies > 0) {
				v4 temp;
				memcpy(&temp, &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
				shared_v_verts[threadIdx.x].x = temp.vxy.x;
				shared_v_verts[threadIdx.x].y = temp.vxy.y;
				if (iSpecies == 1) {
					shared_v_verts[threadIdx.x].z = temp.viz;
				}
				else {
					shared_v_verts[threadIdx.x].z = temp.vez;
				};
				shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
				shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
				// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
				// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
				// OK just do that.
			} else {
				shared_v_verts[threadIdx.x] = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
			};
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_v_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&deriv_xy, 0, sizeof(f64_tens2));
		deriv_z = 0.0;
		deriv_3_xz = 0.0;
		deriv_3_yz = 0.0;
		deriv_3_zx = 0.0;
		deriv_3_zy = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_vec3 opp_v, prev_v, next_v;
			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;


			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_v[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				if (iSpecies > 0) {
					v4 temp = p_vie_minor[izTri[iprev]];
					prev_v.x = temp.vxy.x;
					prev_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						prev_v.z = temp.viz;
					} else {
						prev_v.z = temp.vez;
					};

					// we'd have done better with 2 separate v vectors as it turns out.
				} else {
					prev_v = p_v_n[izTri[iprev]];
				}
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);// = Clockwise3_d*prev_v;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_v[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				if (iSpecies > 0) {
					v4 temp = p_vie_minor[izTri[i]];
					opp_v.x = temp.vxy.x;
					opp_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						opp_v.z = temp.viz;
					} else {
						opp_v.z = temp.vez;
					};
				} else {
					opp_v = p_v_n[izTri[i]];
				}
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}


#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext == tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;


				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_v[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					if (iSpecies > 0) {
						v4 temp = p_vie_minor[izTri[inext]];
						next_v.x = temp.vxy.x;
						next_v.y = temp.vxy.y;
						if (iSpecies == 1) {
							next_v.z = temp.viz;
						} else {
							next_v.z = temp.vez;
						};
					} else {
						next_v = p_v_n[izTri[inext]];
					}
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				f64_vec3 omega_c;
				{
					f64_vec2 opp_B;
					f64 ita_theirs, nu_theirs;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						ita_theirs = shared_ita_par[izTri[i] - StartMinor];
						nu_theirs = shared_nu[izTri[i] - StartMinor];
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						ita_theirs = p_ita_parallel_minor[izTri[i]];
						nu_theirs = p_nu_minor[izTri[i]];
					};

					ita_par = sqrt(shared_ita_par_verts[threadIdx.x] * ita_theirs); 

					if (shared_ita_par_verts[threadIdx.x] * ita_theirs < 0.0) printf("Alert: %1.9E i %d iVertex %d \n", shared_ita_par_verts[threadIdx.x] * ita_theirs, i, iVertex);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu_verts[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.


				f64_vec2 coeffgradvx_by_vx, coeffgradvy_by_vy, coeffgradvz_by_vz; 
				// component j = rate of change of dvq/dj wrt change in vq_self

				bool bLongi = false;
				if (ita_par > 0.0) {

					// We take the derivative considering only the longitudinal component.
					// The effect of self v on the transverse component should be small.
//					coeffgradvq_by_vq = (-1.0)*(opppos - info.pos) /
//						(opppos - info.pos).dot(opppos - info.pos);

					
#ifdef INS_INS_3POINT
					if (TestDomainPos(prevpos) == false) {
						coeffgradvx_by_vx = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);
						coeffgradvy_by_vy = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z);

					} else {
						if (TestDomainPos(nextpos) == false) {
							coeffgradvx_by_vx = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.x, shared_v_verts[threadIdx.x].x, opp_v.x);
							coeffgradvy_by_vy = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.y, shared_v_verts[threadIdx.x].y, opp_v.y);
							coeffgradvz_by_vz = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.z, shared_v_verts[threadIdx.x].z, opp_v.z);
						} else {
							
							coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);
							coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
							coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z);
						}

					};
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						bLongi = true;
#else
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{

						coeffgradvx_by_vx = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x);
						coeffgradvy_by_vy = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z);
					} else {
						coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
							prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x);
						coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
							prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
							prev_v.z, shared_v_verts[threadIdx.x].z, next_v.z, opp_v.z);
					};

#endif

#if (TEST_EPSILON_X) 
						printf("iVertex %d coeffgradvx_by_vx %1.10E %1.10E \n", iVertex, coeffgradvx_by_vx.x,
						coeffgradvx_by_vx.y);
#endif
					if (iSpecies == 0) {

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
						if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
						
						f64 visc_contrib_x = over_m_n*ita_par*(coeffgradvx_by_vx.dot(edge_normal));
						f64 visc_contrib_y = over_m_n*ita_par*(coeffgradvy_by_vy.dot(edge_normal));
						f64 visc_contrib_z = over_m_n*ita_par*(coeffgradvz_by_vz.dot(edge_normal));
						// same for all dimensions
						
						deriv_xy.xx += visc_contrib_x;
						deriv_xy.yy += visc_contrib_y;
						deriv_z += visc_contrib_z;
						
					} else {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							//f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;
	//
	//						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
	//						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
	//						Pi_yx = Pi_xy;
	//						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
	//						Pi_zx = -ita_par*(gradvz.x);
	//						Pi_zy = -ita_par*(gradvz.y);

							// We now want to make Pi = dPi/d v_self ... 
							// Only vx affects gradvx and only vy affects gradvy



					//		Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
					//		Pi_xy = -ita_par*(gradvx.y + gradvy.x);
					//		Pi_yx = Pi_xy;
					//		Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
					//		Pi_zx = -ita_par*(gradvz.x);
					//		Pi_zy = -ita_par*(gradvz.y);


							f64 dPixx_bydvx = -ita_par*THIRD*4.0*coeffgradvx_by_vx.x;
							f64 dPixx_bydvy = ita_par*THIRD*2.0*coeffgradvy_by_vy.y;
							f64 dPixy_bydvx = -ita_par*coeffgradvx_by_vx.y;
							f64 dPixy_bydvy = -ita_par*coeffgradvy_by_vy.x;
							f64 dPiyy_bydvx = ita_par*THIRD*2.0*coeffgradvx_by_vx.x;
							f64 dPiyy_bydvy = -ita_par*THIRD*4.0*coeffgradvy_by_vy.y;

							f64 dPizx_bydvz = -ita_par*coeffgradvz_by_vz.x;
							f64 dPizy_bydvz = -ita_par*coeffgradvz_by_vz.y;

							// We are going to want d [ ROC x y ] / d v xy
							// we discard any effect of vz on ROCxy
							// In this unmagnetized version we can say that Pi_zx, Pi_zy is what contributes to visc_contrib.z.
							
							f64_tens2 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);
							if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
							//						visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
												//	visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);

							visc_contrib.xx = -over_m_s*(dPixx_bydvx*edge_normal.x + dPixy_bydvx*edge_normal.y);
							visc_contrib.xy = -over_m_s*(dPixx_bydvy*edge_normal.x + dPixy_bydvy*edge_normal.y);
							visc_contrib.yx = -over_m_s*(dPixy_bydvx*edge_normal.x + dPiyy_bydvx*edge_normal.y);
							visc_contrib.yy = -over_m_s*(dPixy_bydvy*edge_normal.x + dPiyy_bydvy*edge_normal.y);

							f64 visc_contrib_z = -over_m_s*(dPizx_bydvz*edge_normal.x + dPizy_bydvz*edge_normal.y);

							deriv_xy += visc_contrib;
							deriv_z += visc_contrib_z;

						}
						else {

							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64_vec3 dPibb_by_dv(0.0, 0.0, 0.0),
								dPiPb_by_dv(0.0, 0.0, 0.0),
								dPiPP_by_dv(0.0, 0.0, 0.0),
								dPiHb_by_dv(0.0, 0.0, 0.0),
								dPiHP_by_dv(0.0, 0.0, 0.0),
								dPiHH_by_dv(0.0, 0.0, 0.0);
							{
								f64_vec3 dW_bb_by_dv(0.0, 0.0, 0.0),
									dW_bP_by_dv(0.0, 0.0, 0.0),
									dW_bH_by_dv(0.0, 0.0, 0.0),
									dW_PP_by_dv(0.0, 0.0, 0.0),
									dW_PH_by_dv(0.0, 0.0, 0.0),
									dW_HH_by_dv(0.0, 0.0, 0.0);
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod, d_intermed_x_by_dvx, d_intermed_y_by_dvy, d_intermed_z_by_dvz;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);
									if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.

								}
								{
									//	f64_vec3 intermed;

										// use: d vb / da = b transpose [ dvi/dxj ] a
										// Prototypical element: a.x b.y dvy/dx
										// b.x a.y dvx/dy
										//intermed.x = unit_b.dotxy(gradvx);
										//intermed.y = unit_b.dotxy(gradvy);
										//intermed.z = unit_b.dotxy(gradvz);

									{
										d_intermed_x_by_dvx = unit_b.dotxy(coeffgradvx_by_vx); 
										d_intermed_y_by_dvy = unit_b.dotxy(coeffgradvy_by_vy); 
										d_intermed_z_by_dvz = unit_b.dotxy(coeffgradvz_by_vz); 

										//	f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

											//dvb_by_db = unit_b.dot(intermed);
											//dvperp_by_db = unit_perp.dot(intermed);
											//dvHall_by_db = unit_Hall.dot(intermed);

											//dvb_by_db_by_dx = unit_b.x*d_intermed_j_by_dvj;
											//dvb_by_db_by_dy = unit_b.y*d_intermed_j_by_dvj;
											//dvb_by_db_by_dz = unit_b.z*d_intermed_j_by_dvj;

											//dvperp_by_db_by_dx = unit_perp.x*d_intermed_j_by_dvj;

											// coeffgradvq_by_vq is the derivative of gradvx wrt x

											// dvb_by_db is ROC wrt v .. b ? 

										//	W_bb += 4.0*THIRD*dvb_by_db;
										//	W_bP += dvperp_by_db;
										//	W_bH += dvHall_by_db;
										//	W_PP -= 2.0*THIRD*dvb_by_db;
										//	W_HH -= 2.0*THIRD*dvb_by_db;

									//	d__dvb_by_db__by_dvx = unit_b.x*d_intermed_x_by_dvx;
									//	d__dvb_by_db__by_dvy = unit_b.y*d_intermed_y_by_dvy;
									//	d__dvb_by_db__by_dvz = unit_b.z*d_intermed_z_by_dvz;

										dW_bb_by_dv.x += 4.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y += 4.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z += 4.0*THIRD*unit_b.z*d_intermed_z_by_dvz;

										dW_bP_by_dv.x += unit_perp.x*d_intermed_x_by_dvx;
										dW_bP_by_dv.y += unit_perp.y*d_intermed_y_by_dvy;
										dW_bP_by_dv.z += unit_perp.z*d_intermed_z_by_dvz;

										dW_bH_by_dv.x += unit_Hall.x*d_intermed_x_by_dvx;
										dW_bH_by_dv.y += unit_Hall.y*d_intermed_y_by_dvy;
										dW_bH_by_dv.z += unit_Hall.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x -= 2.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y -= 2.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z -= 2.0*THIRD*unit_b.z*d_intermed_z_by_dvz;

										dW_HH_by_dv.x -= 2.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y -= 2.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z -= 2.0*THIRD*unit_b.z*d_intermed_z_by_dvz;
																				
										//dW_bH_by_dv += unit_Hall*d_intermed_j_by_dvj;
										//dW_PP_by_dv -= 2.0*THIRD*unit_b*d_intermed_j_by_dvj;
										//dW_HH_by_dv -= 2.0*THIRD*unit_b*d_intermed_j_by_dvj;
										
//
//										if (TEST_EPSILON_Y) printf("Contrib to dWbb/dv from xyz %1.9 %1.9E %1.9E d_inter_j %1.10E\n",
//											4.0*THIRD*unit_b.x*d_intermed_j_by_dvj,
//											4.0*THIRD*unit_b.y*d_intermed_j_by_dvj,
//											4.0*THIRD*unit_b.z*d_intermed_j_by_dvj,
//											d_intermed_j_by_dvj
//											);
									}
									{
										//		f64 dvb_by_dperp, dvperp_by_dperp,
										//			dvHall_by_dperp;
												// Optimize by getting rid of different labels.

										//		intermed.x = unit_perp.dotxy(gradvx);
										//		intermed.y = unit_perp.dotxy(gradvy);
										//		intermed.z = unit_perp.dotxy(gradvz);

										d_intermed_x_by_dvx = unit_perp.dotxy(coeffgradvx_by_vx);
										d_intermed_y_by_dvy = unit_perp.dotxy(coeffgradvy_by_vy);
										d_intermed_z_by_dvz = unit_perp.dotxy(coeffgradvz_by_vz);
										//d_intermed_j_by_dvj = unit_perp.dotxy(coeffgradvq_by_vq);

										//		dvb_by_dperp = unit_b.dot(intermed);
										//		dvperp_by_dperp = unit_perp.dot(intermed);
										//		dvHall_by_dperp = unit_Hall.dot(intermed);

										//		W_bb -= 2.0*THIRD*dvperp_by_dperp;
										//		W_PP += 4.0*THIRD*dvperp_by_dperp;
										//		W_HH -= 2.0*THIRD*dvperp_by_dperp;
										//		W_bP += dvb_by_dperp;
										//		W_PH += dvHall_by_dperp;

										dW_bb_by_dv.x -= 2.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y -= 2.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z -= 2.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x += 4.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y += 4.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z += 4.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;
																				
										dW_HH_by_dv.x -= 2.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y -= 2.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z -= 2.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;
																				
										dW_bP_by_dv.x += unit_b.x*d_intermed_x_by_dvx;
										dW_bP_by_dv.y += unit_b.y*d_intermed_y_by_dvy;
										dW_bP_by_dv.z += unit_b.z*d_intermed_z_by_dvz;
																				
										dW_PH_by_dv.x += unit_Hall.x*d_intermed_x_by_dvx;
										dW_PH_by_dv.y += unit_Hall.y*d_intermed_y_by_dvy;
										dW_PH_by_dv.z += unit_Hall.z*d_intermed_z_by_dvz;

										//if (TEST_EPSILON_Y) printf("Contrib2 to dWPP/dv from xyz %1.9 %1.9E %1.9E d_inter_j %1.10E = perp.dgradvj/vj\n",
										//	4.0*THIRD*unit_perp.x*d_intermed_j_by_dvj,
										//	4.0*THIRD*unit_perp.y*d_intermed_j_by_dvj,
										//	4.0*THIRD*unit_perp.z*d_intermed_j_by_dvj,
										//	d_intermed_j_by_dvj
										//);
									}
									{
										//f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;
										//// basically, all should be 0

										//intermed.x = unit_Hall.dotxy(gradvx);
										//intermed.y = unit_Hall.dotxy(gradvy);
										//intermed.z = unit_Hall.dotxy(gradvz);

										//dvb_by_dHall = unit_b.dot(intermed);
										//dvperp_by_dHall = unit_perp.dot(intermed);
										//dvHall_by_dHall = unit_Hall.dot(intermed);

										d_intermed_x_by_dvx = unit_Hall.dotxy(coeffgradvx_by_vx); // 0
										d_intermed_y_by_dvy = unit_Hall.dotxy(coeffgradvy_by_vy);
										d_intermed_z_by_dvz = unit_Hall.dotxy(coeffgradvz_by_vz);

									//	W_bb -= 2.0*THIRD*dvHall_by_dHall;
									//	W_PP -= 2.0*THIRD*dvHall_by_dHall;
									//	W_HH += 4.0*THIRD*dvHall_by_dHall;
									//	W_bH += dvb_by_dHall;
									//	W_PH += dvperp_by_dHall;

										dW_bb_by_dv.x -= 2.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y -= 2.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z -= 2.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x -= 2.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y -= 2.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z -= 2.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_HH_by_dv.x += 4.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y += 4.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z += 4.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_bH_by_dv.x += unit_b.x*d_intermed_x_by_dvx;
										dW_bH_by_dv.y += unit_b.y*d_intermed_y_by_dvy;
										dW_bH_by_dv.z += unit_b.z*d_intermed_z_by_dvz;

										dW_PH_by_dv.x += unit_perp.x*d_intermed_x_by_dvx;
										dW_PH_by_dv.y += unit_perp.y*d_intermed_y_by_dvy;
										dW_PH_by_dv.z += unit_perp.z*d_intermed_z_by_dvz;
									}
								}
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
									//
									//								Pi_b_b += -ita_par*W_bb;
									//								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									//								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									//								Pi_H_P += -ita_1*W_PH;
																	// W_HH = 0

									dPibb_by_dv += -ita_par*dW_bb_by_dv;
									dPiPP_by_dv += -0.5*(ita_par + ita_1)*dW_PP_by_dv - 0.5*(ita_par - ita_1)*dW_HH_by_dv;
									dPiHH_by_dv += -0.5*(ita_par + ita_1)*dW_HH_by_dv - 0.5*(ita_par - ita_1)*dW_PP_by_dv;
									dPiHP_by_dv += -ita_1*dW_PH_by_dv;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									//Pi_P_b += -ita_2*W_bP;
									//Pi_H_b += -ita_2*W_bH;

									dPiPb_by_dv += -ita_2*dW_bP_by_dv;
									dPiHb_by_dv += -ita_2*dW_bH_by_dv;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									//Pi_P_P -= ita_3*W_PH;
									//Pi_H_H += ita_3*W_PH;
									//Pi_H_P += 0.5*ita_3*(W_PP - W_HH);

									dPiPP_by_dv -= ita_3*dW_PH_by_dv;
									dPiHH_by_dv += ita_3*dW_PH_by_dv;
									dPiHP_by_dv += 0.5*ita_3*(dW_PP_by_dv - dW_HH_by_dv);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									//Pi_P_b += -ita_4*W_bH;
									//Pi_H_b += ita_4*W_bP;

									dPiPb_by_dv += -ita_4*dW_bH_by_dv;
									dPiHb_by_dv += ita_4*dW_bP_by_dv;
								}

								//if (iVertex == VERTCHOSEN) {
								//	// 2 components only:
								//	printf("%d %d dW_bb_by_dv %1.10E %1.10E | dW_bP_by_dv %1.10E %1.10E | dW_bH_by_dv %1.10E %1.10E \n"
								//		" |   dW_PP_by_dv %1.10E %1.10E | dW_PH_by_dv %1.10E %1.10E | dW_HH_by_dv %1.10E %1.10E \n",
								//		iVertex, i, dW_bb_by_dv.x, dW_bb_by_dv.y, dW_bP_by_dv.x, dW_bP_by_dv.y, dW_bH_by_dv.x, dW_bH_by_dv.y,
								//		dW_PP_by_dv.x, dW_PP_by_dv.y, dW_PH_by_dv.x, dW_PH_by_dv.y, dW_HH_by_dv.x, dW_HH_by_dv.y);
								//	printf("%d %d dPibb_by_dv %1.10E %1.10E | dPiPb_by_dv %1.10E %1.10E | dPiHb_by_dv %1.10E %1.10E \n"
								//		" |  dPiPP_by_dv %1.10E %1.10E | dPiHP_by_dv %1.10E %1.10E | dPiHH_by_dv %1.10E %1.10E \n",
								//		iVertex, i, dPibb_by_dv.x, dPibb_by_dv.y, dPiPb_by_dv.x, dPiPb_by_dv.y, dPiHb_by_dv.x, dPiHb_by_dv.y,
								//		dPiPP_by_dv.x, dPiPP_by_dv.y, dPiHP_by_dv.x, dPiHP_by_dv.y, dPiHH_by_dv.x, dPiHH_by_dv.y);
								//}

							} // scope W

							  // All we want left over at this point is Pi .. and unit_b

							f64_vec3 Dmomflux_b, Dmomflux_perp, Dmomflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);
								if (bLongi) edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);
								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

								//momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								//momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								//momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
								Dmomflux_b =    -(dPibb_by_dv*mag_edge.x + dPiPb_by_dv*mag_edge.y + dPiHb_by_dv*mag_edge.z);
								Dmomflux_perp = -(dPiPb_by_dv*mag_edge.x + dPiPP_by_dv*mag_edge.y + dPiHP_by_dv*mag_edge.z);
								Dmomflux_Hall = -(dPiHb_by_dv*mag_edge.x + dPiHP_by_dv*mag_edge.y + dPiHH_by_dv*mag_edge.z);
							}

							// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
							// is the flow of p_x dotted with the edge_normal
							// ownrates will be divided by N to give dv/dt
							// m N dvx/dt = integral div momflux_x
							// Therefore divide here just by m

							f64_vec3 Dvisc_contrib;// , Dvisc_contriby, Dvisc_contribz;
						//	Dvisc_contribx = over_m_s*(unit_b.x*Dmomflux_b + unit_perp.x*Dmomflux_perp + unit_Hall.x*Dmomflux_Hall);
							Dvisc_contrib = over_m_s*(unit_b.x*Dmomflux_b + unit_perp.x*Dmomflux_perp + unit_Hall.x*Dmomflux_Hall);
							//	Dvisc_contriby = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							//	Dvisc_contribz = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);

								// only the elements xx, xy, yx, yy, zz will be kept.
							deriv_xy.xx += Dvisc_contrib.x;
							deriv_xy.xy += Dvisc_contrib.y;

							deriv_3_xz += Dvisc_contrib.z;

							Dvisc_contrib = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							deriv_xy.yx += Dvisc_contrib.x;
							deriv_xy.yy += Dvisc_contrib.y;

							deriv_3_yz += Dvisc_contrib.z;

							Dvisc_contrib = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);
							deriv_z += Dvisc_contrib.z;

							deriv_3_zx += Dvisc_contrib.x;
							deriv_3_zy += Dvisc_contrib.y;
							
							// We aim to save the whole matrix -- yes it is needful.
							
						}; // whether unmagnetized
					}; // is it for neutrals
				}; // was ita_par == 0

				   // v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i

#if (TEST_EPSILON_X)
				printf("\n%d CalculateCoeffself : deriv_xy= xx %1.9E xy %1.9E yx %1.9E yy %1.9E \n\n",
				iVertex, deriv_xy.xx, deriv_xy.xy, deriv_xy.yx, deriv_xy.yy);
#endif

			memcpy(&(p__coeffself_xy[iVertex + BEGINNING_OF_CENTRAL]), &deriv_xy, sizeof(f64_tens2));
			p__coeffself_z[iVertex + BEGINNING_OF_CENTRAL] = deriv_z;

			double4 billabong;
			billabong.x = deriv_3_xz; 
			billabong.y = deriv_3_yz;
			billabong.z = deriv_3_zx;
			billabong.w = deriv_3_zy;

			memcpy(&(p__billabong[iVertex + BEGINNING_OF_CENTRAL]), &billabong, sizeof(double4));
			// maybe there are quicker ways.
			// just save xz, yz, zx, zy as 4 more things.

		} else {
			// NOT domain vertex: Do nothing			
			memset(&(p__coeffself_xy[iVertex + BEGINNING_OF_CENTRAL]), 0, sizeof(f64_tens2));
			p__coeffself_z[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
			memset(&(p__billabong[iVertex + BEGINNING_OF_CENTRAL]), 0, sizeof(double4));

		}; // whether domain vertex

	}; // if do vertex at all

	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memset(&deriv_xy, 0, sizeof(f64_tens2));
	deriv_z = 0.0;
	deriv_3_xz = 0.0;
	deriv_3_yz = 0.0;
	deriv_3_zx = 0.0;
	deriv_3_zy = 0.0;

	{
		long izNeighMinor[6];
		char szPBC[6];
		
		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			f64_vec3 prev_v, opp_v, next_v;
			f64_vec2 prevpos, nextpos, opppos;

			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_v[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			} else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					v4 temp = p_vie_minor[izNeighMinor[iprev]];
					prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y;
					if (iSpecies == 1) {
						prev_v.z = temp.viz;
					} else {
						prev_v.z = temp.vez;
					}; 
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_v[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_v_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					v4 temp = p_vie_minor[izNeighMinor[i]];
					opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y; 
					if (iSpecies == 1) {
						opp_v.z = temp.viz;
					} else {
						opp_v.z = temp.vez;
					};
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}
			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_v[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_v_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						v4 temp = p_vie_minor[izNeighMinor[inext]];
						next_v.x = temp.vxy.x; next_v.y = temp.vxy.y; 
						if (iSpecies == 1) {
							next_v.z = temp.viz;
						} else {
							next_v.z = temp.vez;
						};
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}


				bool bUsableSide = true;
				{
					f64 nu_theirs, ita_theirs;
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						nu_theirs = shared_nu[izNeighMinor[i] - StartMinor];
						ita_theirs = shared_ita_par[izNeighMinor[i] - StartMinor];
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							nu_theirs = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							ita_theirs = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							nu_theirs = p_nu_minor[izNeighMinor[i]];
							ita_theirs = p_ita_parallel_minor[izNeighMinor[i]];
						}
					}
					// GEOMETRIC ITA:
					if (ita_theirs == 0.0) bUsableSide = false;
					ita_par = sqrt(shared_ita_par[threadIdx.x] * ita_theirs);

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					nu = 0.5*(nu_theirs + shared_nu[threadIdx.x]);
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// ins-ins triangle traffic:

				bool bLongi = false;

#ifdef INS_INS_NONE
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
#else 
				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bLongi = true;
				}
				if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					bLongi = true;
#endif

				f64_vec2 coeffgradvx_by_vx, coeffgradvy_by_vy, coeffgradvz_by_vz;
				// component j = rate of change of dvq/dj wrt change in vq_self

				if (bUsableSide) {

					// We take the derivative considering only the longitudinal component.
					// The effect of self v on the transverse component should be small.
					//					coeffgradvq_by_vq = (-1.0)*(opppos - info.pos) /
					//						(opppos - info.pos).dot(opppos - info.pos);

#ifdef INS_INS_3POINT

					if (TestDomainPos(prevpos) == false) {

						coeffgradvx_by_vx = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v[threadIdx.x].x, next_v.x, opp_v.x);
						coeffgradvy_by_vy = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradient_3Point(info.pos, nextpos, opppos,
							shared_v[threadIdx.x].z, next_v.z, opp_v.z);

#if (TEST_EPSILON_X_MINOR) 
							printf("prev was under. info next opp %1.9E %1.9E , %1.9E %1.9E , %1.9E %1.9E \nv %1.9E %1.9E %1.9E coeffgradvxbyvx %1.10E %1.10E \n",
								info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
								shared_v[threadIdx.x].x, next_v.x, opp_v.x,
								coeffgradvx_by_vx.x, coeffgradvx_by_vx.y);
#endif

					} else {
						if (TestDomainPos(nextpos) == false) {

							coeffgradvx_by_vx = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, opp_v.x);
							coeffgradvy_by_vy = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.y, shared_v[threadIdx.x].y, opp_v.y);
							coeffgradvz_by_vz = GetSelfEffectOnGradient_3Point(prevpos, info.pos, opppos,
								prev_v.z, shared_v[threadIdx.x].z, opp_v.z);

#if (TEST_EPSILON_X_MINOR) 
								printf("next was under. prev info opp %1.9E %1.9E , %1.9E %1.9E , %1.9E %1.9E \nv %1.9E %1.9E %1.9E coeffgradvxbyvx %1.10E %1.10E \n",
									prevpos.x, prevpos.y, info.pos.x, info.pos.y, opppos.x, opppos.y,
									prev_v.x, shared_v[threadIdx.x].x, opp_v.x,
									coeffgradvx_by_vx.x, coeffgradvx_by_vx.y);
#endif

						} else {

							coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x);
							coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y);
							coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z);

#if (TEST_EPSILON_X_MINOR) 
								printf("%d %d prev info next opp %1.9E %1.9E , %1.9E %1.9E , %1.9E %1.9E , %1.9E %1.9E\nv %1.9E %1.9E %1.9E %1.9E coeffgradvxbyvx %1.10E %1.10E \n",
									iMinor, i, prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
									prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x,
									coeffgradvx_by_vx.x, coeffgradvx_by_vx.y);
#endif
						};
					};

#else
					if (bLongi) {
						
						coeffgradvx_by_vx = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x);
						coeffgradvy_by_vy = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradientLongitudinal(prevpos, info.pos, nextpos, opppos,
							prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z);
						
					} else {

						coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
							prev_v.y, shared_v[threadIdx.x].y, next_v.y, opp_v.y);
						coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
							prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z);
						
#if (TEST_EPSILON_X_MINOR)
						
							coeffgradvx_by_vx = GetSelfEffectOnGradientDebug(prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x);

							printf("iMinor %d coeffgradvx_by_vx %1.10E %1.10E | coeffgradvy_by_vy %1.10E %1.10E\npos prev ours anti opp %1.8E %1.8E , %1.8E %1.8E , %1.8E %1.8E , %1.8E %1.8E\n", iMinor, coeffgradvx_by_vx.x,
								coeffgradvx_by_vx.y, coeffgradvy_by_vy.x, coeffgradvy_by_vy.y,
								prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y);
#else 

							coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
								prev_v.x, shared_v[threadIdx.x].x, next_v.x, opp_v.x);
#endif
					};
#endif

					if (iSpecies == 0) {

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

						if (bLongi) {
							// move any edge_normal endpoints that are below the insulator,
							// until they are above the insulator.
							edge_normal = ReconstructEdgeNormal(
								prevpos, info.pos, nextpos, opppos
							);
						};
						f64 visc_contrib_x = over_m_n*ita_par*(coeffgradvx_by_vx.dot(edge_normal));
						f64 visc_contrib_y = over_m_n*ita_par*(coeffgradvy_by_vy.dot(edge_normal));
						f64 visc_contrib_z = over_m_n*ita_par*(coeffgradvz_by_vz.dot(edge_normal));
						// same for all dimensions
						
						deriv_xy.xx += visc_contrib_x;
						deriv_xy.yy += visc_contrib_y;
						deriv_z += visc_contrib_z;

					}
					else {
						if ((VISCMAG == 0) || (omega_c.dot(omega_c) < 0.01*0.1*nu*nu))
						{
							// run unmagnetised case
							//f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;
							//
							//						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
							//						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
							//						Pi_yx = Pi_xy;
							//						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
							//						Pi_zx = -ita_par*(gradvz.x);
							//						Pi_zy = -ita_par*(gradvz.y);

							// We now want to make Pi = dPi/d v_self ... 
							// Only vx affects gradvx and only vy affects gradvy



							//		Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
							//		Pi_xy = -ita_par*(gradvx.y + gradvy.x);
							//		Pi_yx = Pi_xy;
							//		Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
							//		Pi_zx = -ita_par*(gradvz.x);
							//		Pi_zy = -ita_par*(gradvz.y);


							f64 dPixx_bydvx = -ita_par*THIRD*4.0*coeffgradvx_by_vx.x;
							f64 dPixx_bydvy = ita_par*THIRD*2.0*coeffgradvy_by_vy.y;
							f64 dPixy_bydvx = -ita_par*coeffgradvx_by_vx.y;
							f64 dPixy_bydvy = -ita_par*coeffgradvy_by_vy.x;
							f64 dPiyy_bydvx = ita_par*THIRD*2.0*coeffgradvx_by_vx.x;
							f64 dPiyy_bydvy = -ita_par*THIRD*4.0*coeffgradvy_by_vy.y;

							f64 dPizx_bydvz = -ita_par*coeffgradvz_by_vz.x;
							f64 dPizy_bydvz = -ita_par*coeffgradvz_by_vz.y;

							// We are going to want d [ ROC x y ] / d v xy
							// we discard any effect of vz on ROCxy
							// In this unmagnetized version we can say that Pi_zx, Pi_zy is what contributes to visc_contrib.z.
							
							f64_tens2 visc_contrib;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);

							if (bLongi) {
								// move any edge_normal endpoints that are below the insulator,
								// until they are above the insulator.
								edge_normal = ReconstructEdgeNormal(
									prevpos, info.pos, nextpos, opppos
								);
							};
							//						visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							//	visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);

							visc_contrib.xx = -over_m_s*(dPixx_bydvx*edge_normal.x + dPixy_bydvx*edge_normal.y);
							visc_contrib.xy = -over_m_s*(dPixx_bydvy*edge_normal.x + dPixy_bydvy*edge_normal.y);
							visc_contrib.yx = -over_m_s*(dPixy_bydvx*edge_normal.x + dPiyy_bydvx*edge_normal.y);
							visc_contrib.yy = -over_m_s*(dPixy_bydvy*edge_normal.x + dPiyy_bydvy*edge_normal.y);

							f64 visc_contrib_z = -over_m_s*(dPizx_bydvz*edge_normal.x + dPizy_bydvz*edge_normal.y);

							deriv_xy += visc_contrib;
							deriv_z += visc_contrib_z;

#if (TEST_EPSILON_X_MINOR)
							printf("UNMAG : visccontrib xx xy yx yy %1.10E %1.10E %1.10E %1.10E \n",
								visc_contrib.xx, visc_contrib.xy, visc_contrib.yx, visc_contrib.yy);
#endif
						} else {

							f64_vec3 unit_b, unit_perp, unit_Hall;
							f64_vec3 dPibb_by_dv(0.0, 0.0, 0.0),
								dPiPb_by_dv(0.0, 0.0, 0.0),
								dPiPP_by_dv(0.0, 0.0, 0.0),
								dPiHb_by_dv(0.0, 0.0, 0.0),
								dPiHP_by_dv(0.0, 0.0, 0.0),
								dPiHH_by_dv(0.0, 0.0, 0.0);
							{
								f64_vec3 dW_bb_by_dv(0.0, 0.0, 0.0),
									dW_bP_by_dv(0.0, 0.0, 0.0),
									dW_bH_by_dv(0.0, 0.0, 0.0),
									dW_PP_by_dv(0.0, 0.0, 0.0),
									dW_PH_by_dv(0.0, 0.0, 0.0),
									dW_HH_by_dv(0.0, 0.0, 0.0);
								// these have to be alive at same time as 9 x partials
								// but we can make do with 3x partials
								// 2. Now get partials in magnetic coordinates 
								f64 omegamod, d_intermed_x_by_dvx, d_intermed_y_by_dvy, d_intermed_z_by_dvz;
								{
									f64_vec2 edge_normal;
									edge_normal.x = THIRD * (nextpos.y - prevpos.y);
									edge_normal.y = THIRD * (prevpos.x - nextpos.x);

									if (bLongi) {
										// move any edge_normal endpoints that are below the insulator,
										// until they are above the insulator.
										edge_normal = ReconstructEdgeNormal(
											prevpos, info.pos, nextpos, opppos
										);
									};
									f64 omegasq = omega_c.dot(omega_c);
									omegamod = sqrt(omegasq);
									unit_b = omega_c / omegamod;
									unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
									unit_perp = unit_perp / unit_perp.modulus();
									unit_Hall = unit_b.cross(unit_perp); // Note sign.

								}
								{
									//	f64_vec3 intermed;

									// use: d vb / da = b transpose [ dvi/dxj ] a
									// Prototypical element: a.x b.y dvy/dx
									// b.x a.y dvx/dy
									//intermed.x = unit_b.dotxy(gradvx);
									//intermed.y = unit_b.dotxy(gradvy);
									//intermed.z = unit_b.dotxy(gradvz);

									{
										d_intermed_x_by_dvx = unit_b.dotxy(coeffgradvx_by_vx);
										d_intermed_y_by_dvy = unit_b.dotxy(coeffgradvy_by_vy);
										d_intermed_z_by_dvz = unit_b.dotxy(coeffgradvz_by_vz);

										//	f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

										//dvb_by_db = unit_b.dot(intermed);
										//dvperp_by_db = unit_perp.dot(intermed);
										//dvHall_by_db = unit_Hall.dot(intermed);

										//dvb_by_db_by_dx = unit_b.x*d_intermed_j_by_dvj;
										//dvb_by_db_by_dy = unit_b.y*d_intermed_j_by_dvj;
										//dvb_by_db_by_dz = unit_b.z*d_intermed_j_by_dvj;

										//dvperp_by_db_by_dx = unit_perp.x*d_intermed_j_by_dvj;

										// coeffgradvq_by_vq is the derivative of gradvx wrt x

										// dvb_by_db is ROC wrt v .. b ? 

										//	W_bb += 4.0*THIRD*dvb_by_db;
										//	W_bP += dvperp_by_db;
										//	W_bH += dvHall_by_db;
										//	W_PP -= 2.0*THIRD*dvb_by_db;
										//	W_HH -= 2.0*THIRD*dvb_by_db;

									//	d__dvb_by_db__by_dvx = unit_b.x*d_intermed_x_by_dvx;
									//	d__dvb_by_db__by_dvy = unit_b.y*d_intermed_y_by_dvy;
									//	d__dvb_by_db__by_dvz = unit_b.z*d_intermed_z_by_dvz;

										dW_bb_by_dv.x += 4.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y += 4.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z += 4.0*THIRD*unit_b.z*d_intermed_z_by_dvz;

										dW_bP_by_dv.x += unit_perp.x*d_intermed_x_by_dvx;
										dW_bP_by_dv.y += unit_perp.y*d_intermed_y_by_dvy;
										dW_bP_by_dv.z += unit_perp.z*d_intermed_z_by_dvz;

										dW_bH_by_dv.x += unit_Hall.x*d_intermed_x_by_dvx;
										dW_bH_by_dv.y += unit_Hall.y*d_intermed_y_by_dvy;
										dW_bH_by_dv.z += unit_Hall.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x -= 2.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y -= 2.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z -= 2.0*THIRD*unit_b.z*d_intermed_z_by_dvz;

										dW_HH_by_dv.x -= 2.0*THIRD*unit_b.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y -= 2.0*THIRD*unit_b.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z -= 2.0*THIRD*unit_b.z*d_intermed_z_by_dvz;

										//dW_bH_by_dv += unit_Hall*d_intermed_j_by_dvj;
										//dW_PP_by_dv -= 2.0*THIRD*unit_b*d_intermed_j_by_dvj;
										//dW_HH_by_dv -= 2.0*THIRD*unit_b*d_intermed_j_by_dvj;

										//
										//										if (TEST_EPSILON_Y) printf("Contrib to dWbb/dv from xyz %1.9 %1.9E %1.9E d_inter_j %1.10E\n",
										//											4.0*THIRD*unit_b.x*d_intermed_j_by_dvj,
										//											4.0*THIRD*unit_b.y*d_intermed_j_by_dvj,
										//											4.0*THIRD*unit_b.z*d_intermed_j_by_dvj,
										//											d_intermed_j_by_dvj
										//											);
									}
									{
										//		f64 dvb_by_dperp, dvperp_by_dperp,
										//			dvHall_by_dperp;
										// Optimize by getting rid of different labels.

										//		intermed.x = unit_perp.dotxy(gradvx);
										//		intermed.y = unit_perp.dotxy(gradvy);
										//		intermed.z = unit_perp.dotxy(gradvz);

										d_intermed_x_by_dvx = unit_perp.dotxy(coeffgradvx_by_vx);
										d_intermed_y_by_dvy = unit_perp.dotxy(coeffgradvy_by_vy);
										d_intermed_z_by_dvz = unit_perp.dotxy(coeffgradvz_by_vz);
										//d_intermed_j_by_dvj = unit_perp.dotxy(coeffgradvq_by_vq);

										//		dvb_by_dperp = unit_b.dot(intermed);
										//		dvperp_by_dperp = unit_perp.dot(intermed);
										//		dvHall_by_dperp = unit_Hall.dot(intermed);

										//		W_bb -= 2.0*THIRD*dvperp_by_dperp;
										//		W_PP += 4.0*THIRD*dvperp_by_dperp;
										//		W_HH -= 2.0*THIRD*dvperp_by_dperp;
										//		W_bP += dvb_by_dperp;
										//		W_PH += dvHall_by_dperp;

										dW_bb_by_dv.x -= 2.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y -= 2.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z -= 2.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x += 4.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y += 4.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z += 4.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;

										dW_HH_by_dv.x -= 2.0*THIRD*unit_perp.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y -= 2.0*THIRD*unit_perp.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z -= 2.0*THIRD*unit_perp.z*d_intermed_z_by_dvz;

										dW_bP_by_dv.x += unit_b.x*d_intermed_x_by_dvx;
										dW_bP_by_dv.y += unit_b.y*d_intermed_y_by_dvy;
										dW_bP_by_dv.z += unit_b.z*d_intermed_z_by_dvz;

										dW_PH_by_dv.x += unit_Hall.x*d_intermed_x_by_dvx;
										dW_PH_by_dv.y += unit_Hall.y*d_intermed_y_by_dvy;
										dW_PH_by_dv.z += unit_Hall.z*d_intermed_z_by_dvz;

										//if (TEST_EPSILON_Y) printf("Contrib2 to dWPP/dv from xyz %1.9 %1.9E %1.9E d_inter_j %1.10E = perp.dgradvj/vj\n",
										//	4.0*THIRD*unit_perp.x*d_intermed_j_by_dvj,
										//	4.0*THIRD*unit_perp.y*d_intermed_j_by_dvj,
										//	4.0*THIRD*unit_perp.z*d_intermed_j_by_dvj,
										//	d_intermed_j_by_dvj
										//);
									}
									{
										//f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;
										//// basically, all should be 0

										//intermed.x = unit_Hall.dotxy(gradvx);
										//intermed.y = unit_Hall.dotxy(gradvy);
										//intermed.z = unit_Hall.dotxy(gradvz);

										//dvb_by_dHall = unit_b.dot(intermed);
										//dvperp_by_dHall = unit_perp.dot(intermed);
										//dvHall_by_dHall = unit_Hall.dot(intermed);

										d_intermed_x_by_dvx = unit_Hall.dotxy(coeffgradvx_by_vx); // 0
										d_intermed_y_by_dvy = unit_Hall.dotxy(coeffgradvy_by_vy);
										d_intermed_z_by_dvz = unit_Hall.dotxy(coeffgradvz_by_vz);

										//	W_bb -= 2.0*THIRD*dvHall_by_dHall;
										//	W_PP -= 2.0*THIRD*dvHall_by_dHall;
										//	W_HH += 4.0*THIRD*dvHall_by_dHall;
										//	W_bH += dvb_by_dHall;
										//	W_PH += dvperp_by_dHall;

										dW_bb_by_dv.x -= 2.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_bb_by_dv.y -= 2.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_bb_by_dv.z -= 2.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_PP_by_dv.x -= 2.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_PP_by_dv.y -= 2.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_PP_by_dv.z -= 2.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_HH_by_dv.x += 4.0*THIRD*unit_Hall.x*d_intermed_x_by_dvx;
										dW_HH_by_dv.y += 4.0*THIRD*unit_Hall.y*d_intermed_y_by_dvy;
										dW_HH_by_dv.z += 4.0*THIRD*unit_Hall.z*d_intermed_z_by_dvz;

										dW_bH_by_dv.x += unit_b.x*d_intermed_x_by_dvx;
										dW_bH_by_dv.y += unit_b.y*d_intermed_y_by_dvy;
										dW_bH_by_dv.z += unit_b.z*d_intermed_z_by_dvz;

										dW_PH_by_dv.x += unit_perp.x*d_intermed_x_by_dvx;
										dW_PH_by_dv.y += unit_perp.y*d_intermed_y_by_dvy;
										dW_PH_by_dv.z += unit_perp.z*d_intermed_z_by_dvz;
									}
								}
								{
									f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
									//
									//								Pi_b_b += -ita_par*W_bb;
									//								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
									//								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
									//								Pi_H_P += -ita_1*W_PH;
									// W_HH = 0

									dPibb_by_dv += -ita_par*dW_bb_by_dv;
									dPiPP_by_dv += -0.5*(ita_par + ita_1)*dW_PP_by_dv - 0.5*(ita_par - ita_1)*dW_HH_by_dv;
									dPiHH_by_dv += -0.5*(ita_par + ita_1)*dW_HH_by_dv - 0.5*(ita_par - ita_1)*dW_PP_by_dv;
									dPiHP_by_dv += -ita_1*dW_PH_by_dv;
								}
								{
									f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
									//Pi_P_b += -ita_2*W_bP;
									//Pi_H_b += -ita_2*W_bH;

									dPiPb_by_dv += -ita_2*dW_bP_by_dv;
									dPiHb_by_dv += -ita_2*dW_bH_by_dv;
								}
								{
									f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
									//Pi_P_P -= ita_3*W_PH;
									//Pi_H_H += ita_3*W_PH;
									//Pi_H_P += 0.5*ita_3*(W_PP - W_HH);

									dPiPP_by_dv -= ita_3*dW_PH_by_dv;
									dPiHH_by_dv += ita_3*dW_PH_by_dv;
									dPiHP_by_dv += 0.5*ita_3*(dW_PP_by_dv - dW_HH_by_dv);
								}
								{
									f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
									//Pi_P_b += -ita_4*W_bH;
									//Pi_H_b += ita_4*W_bP;

									dPiPb_by_dv += -ita_4*dW_bH_by_dv;
									dPiHb_by_dv += ita_4*dW_bP_by_dv;
								}

								//if (iVertex == VERTCHOSEN) {
								//	// 2 components only:
								//	printf("%d %d dW_bb_by_dv %1.10E %1.10E | dW_bP_by_dv %1.10E %1.10E | dW_bH_by_dv %1.10E %1.10E \n"
								//		" |   dW_PP_by_dv %1.10E %1.10E | dW_PH_by_dv %1.10E %1.10E | dW_HH_by_dv %1.10E %1.10E \n",
								//		iVertex, i, dW_bb_by_dv.x, dW_bb_by_dv.y, dW_bP_by_dv.x, dW_bP_by_dv.y, dW_bH_by_dv.x, dW_bH_by_dv.y,
								//		dW_PP_by_dv.x, dW_PP_by_dv.y, dW_PH_by_dv.x, dW_PH_by_dv.y, dW_HH_by_dv.x, dW_HH_by_dv.y);
								//	printf("%d %d dPibb_by_dv %1.10E %1.10E | dPiPb_by_dv %1.10E %1.10E | dPiHb_by_dv %1.10E %1.10E \n"
								//		" |  dPiPP_by_dv %1.10E %1.10E | dPiHP_by_dv %1.10E %1.10E | dPiHH_by_dv %1.10E %1.10E \n",
								//		iVertex, i, dPibb_by_dv.x, dPibb_by_dv.y, dPiPb_by_dv.x, dPiPb_by_dv.y, dPiHb_by_dv.x, dPiHb_by_dv.y,
								//		dPiPP_by_dv.x, dPiPP_by_dv.y, dPiHP_by_dv.x, dPiHP_by_dv.y, dPiHH_by_dv.x, dPiHH_by_dv.y);
								//}

							} // scope W

							  // All we want left over at this point is Pi .. and unit_b

							f64_vec3 Dmomflux_b, Dmomflux_perp, Dmomflux_Hall;
							{
								// Most efficient way: compute mom flux in magnetic coords
								f64_vec3 mag_edge;
								f64_vec2 edge_normal;
								edge_normal.x = THIRD * (nextpos.y - prevpos.y);
								edge_normal.y = THIRD * (prevpos.x - nextpos.x);

								if (bLongi) {
									// move any edge_normal endpoints that are below the insulator,
									// until they are above the insulator.
									edge_normal = ReconstructEdgeNormal(
										prevpos, info.pos, nextpos, opppos
									);
								};
								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

								//momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								//momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								//momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
								Dmomflux_b = -(dPibb_by_dv*mag_edge.x + dPiPb_by_dv*mag_edge.y + dPiHb_by_dv*mag_edge.z);
								Dmomflux_perp = -(dPiPb_by_dv*mag_edge.x + dPiPP_by_dv*mag_edge.y + dPiHP_by_dv*mag_edge.z);
								Dmomflux_Hall = -(dPiHb_by_dv*mag_edge.x + dPiHP_by_dv*mag_edge.y + dPiHH_by_dv*mag_edge.z);

#if (TEST_EPSILON_X_MINOR) 
								printf("MAG : Dmomflux_b %1.9E %1.9E %1.9E \n"
									"dPibb_by_dv %1.8E %1.8E %1.8E mag_edge.x %1.9E\n"
									"dPiPb_by_dv %1.8E %1.8E %1.8E mag_edge.y %1.9E\n"
									"Dmomflux_perp %1.9E %1.9E %1.9E \n"
									"dPiPP_by_dv %1.8E %1.8E %1.8E mag_edge.y %1.9E\n",
									Dmomflux_b.x, Dmomflux_b.y, Dmomflux_b.z, dPibb_by_dv.x, dPibb_by_dv.y, dPibb_by_dv.z, mag_edge.x,
									dPiPb_by_dv.x, dPiPb_by_dv.y, dPiPb_by_dv.z, mag_edge.y,
									Dmomflux_perp.x, Dmomflux_perp.y, Dmomflux_perp.z, dPiPP_by_dv.x, dPiPP_by_dv.y, dPiPP_by_dv.z, mag_edge.y
									);
#endif
							}

							// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
							// is the flow of p_x dotted with the edge_normal
							// ownrates will be divided by N to give dv/dt
							// m N dvx/dt = integral div momflux_x
							// Therefore divide here just by m

							f64_vec3 Dvisc_contrib;// , Dvisc_contriby, Dvisc_contribz;
												   //	Dvisc_contribx = over_m_s*(unit_b.x*Dmomflux_b + unit_perp.x*Dmomflux_perp + unit_Hall.x*Dmomflux_Hall);
							Dvisc_contrib = over_m_s*(unit_b.x*Dmomflux_b + unit_perp.x*Dmomflux_perp + unit_Hall.x*Dmomflux_Hall);
							//	Dvisc_contriby = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							//	Dvisc_contribz = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);
#if (TEST_EPSILON_X_MINOR) 
							printf("Dvisc_contrib.x (-> xx) %1.8E ; unit_b.x*Dmomflux_b.x %1.8E unit_perp.x*Dmomflux_perp.x %1.8E \n",
								Dvisc_contrib.x, unit_b.x*Dmomflux_b.x, unit_perp.x*Dmomflux_perp.x);
#endif
							// only the elements xx, xy, yx, yy, zz will be kept.
							deriv_xy.xx += Dvisc_contrib.x;
							deriv_xy.xy += Dvisc_contrib.y;

							deriv_3_xz += Dvisc_contrib.z;

							Dvisc_contrib = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							deriv_xy.yx += Dvisc_contrib.x;
							deriv_xy.yy += Dvisc_contrib.y;

							deriv_3_yz += Dvisc_contrib.z;

							Dvisc_contrib = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);
							deriv_z += Dvisc_contrib.z;

							deriv_3_zx += Dvisc_contrib.x;
							deriv_3_zy += Dvisc_contrib.y;
							
						}; // whether unmagnetized
					}; // is it for neutrals
				}; // bUsableSide

				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				// Just leaving these but they won't do anything :

				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i


			memcpy(&(p__coeffself_xy[iMinor]), &deriv_xy, sizeof(f64_tens2));
			p__coeffself_z[iMinor] = deriv_z;
			double4 billabong;
			billabong.x = deriv_3_xz;
			billabong.y = deriv_3_yz;
			billabong.z = deriv_3_zx;
			billabong.w = deriv_3_zy;

			memcpy(&(p__billabong[iMinor]), &billabong, sizeof(double4));

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
			memset(&(p__coeffself_xy[iMinor]), 0, sizeof(f64_tens2));
			p__coeffself_z[iMinor] = 0.0;
			memset(&(p__billabong[iMinor]), 0, sizeof(double4));

		}
	} // scope

}


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
	f64 * __restrict__ p_nu_in_MT_minor,
	f64 * __restrict__ p_nu_en_MT_minor,
	f64_vec2 * __restrict__ p_Depsilon_xy,
	f64 * __restrict__ p_Depsilon_iz,
	f64 * __restrict__ p_Depsilon_ez
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	v4 Depsilon;
	memset(&Depsilon, 0, sizeof(v4));
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| ((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)// ?
	{
		f64_vec2 regrxy = p_regr_xy[iMinor];
		f64 regriz = p_regr_iz[iMinor];
		f64 regrez = p_regr_ez[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion2[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec2[iMinor];
		nvals n_use = p_n_minor[iMinor];
		f64 N = p_AreaMinor[iMinor] * n_use.n;

#ifndef SQRTNV
		Depsilon.vxy = regrxy 
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		Depsilon.viz = regriz - hsub*(MAR_ion.z / N);
		Depsilon.vez = regrez - hsub*(MAR_elec.z / N);
#else
		f64 sqrtN = sqrt(N);

#if DECAY_IN_VISC_EQNS
		// Now introduce decay factor.

		// We want to anticipate that v will decay.

		// Therefore take

		// v_k+1 target = (v_k + h/N MAR(k+1) - vnk)/(1+h nu mn/(ms+mn) nn/ntot)+vnk
		f64 nu_in_MT = p_nu_in_MT_minor[iMinor];
		f64 nu_en_MT = p_nu_en_MT_minor[iMinor];

		f64 ratio_nn_ntot = n_use.n_n / (n_use.n_n + n_use.n);

		Depsilon.vez = sqrtN*
			(regrez - (hsub*(MAR_elec.z) / N) /
			(1.0 + hsub*nu_en_MT*ratio_nn_ntot)
				);


		f64_vec2 putative = hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		Depsilon.vxy = sqrtN*(
			(regrxy - (putative) /
			(1.0 + hsub*nu_in_MT*ratio_nn_ntot)
				));
		Depsilon.viz = sqrtN*
			(regriz - ( hsub*MAR_ion.z / N ) /
			(1.0 + hsub*nu_in_MT*ratio_nn_ntot)
				);

#else
		Depsilon.vxy = sqrtN*regrxy
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*sqrtN));
		Depsilon.viz = sqrtN*regriz - hsub*(MAR_ion.z / sqrtN);
		Depsilon.vez = sqrtN*regrez - hsub*(MAR_elec.z / sqrtN);
#endif
#endif
		//if ((iMinor == CHOSEN) && (DebugFlag))
		//	printf("\nkernelCCdepsbydbeta %d regr.y %1.10E Depsilon.vxy.y %1.10E MAR_ion.y %1.10E MAR_elec.y %1.10E hsub/N %1.9E\n", CHOSEN, regrxy.y, Depsilon.vxy.y,
		//		MAR_ion.y, MAR_elec.y, hsub/N);
		//if (0) //iMinor == CHOSEN) 
		//	printf("kernelCCdepsbydbeta %d regrez %1.12E Depsilon.vez %1.12E MAR_elec.z %1.12E hsub %1.8E N %1.12E\n",
		//	CHOSEN, regrez, Depsilon.vez, MAR_elec.z, hsub, N);

	} else {
		// epsilon = 0
	};
	p_Depsilon_xy[iMinor] = Depsilon.vxy;
	p_Depsilon_iz[iMinor] = Depsilon.viz;
	p_Depsilon_ez[iMinor] = Depsilon.vez;
}

// not beta
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
	f64 * __restrict__ p_nu_in_MT_minor,
	f64 * __restrict__ p_nu_en_MT_minor,

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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	
	Tensor2 invmatrix;
	memset(&invmatrix, 0, sizeof(Tensor2));
	f64 invcoeffselfviz = 0.0;
	f64 invcoeffselfvez = 0.0;
	f64 invcoeffself_x = 0.0;
	f64 invcoeffself_y = 0.0;

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{

		nvals n_use = p_n_minor[iMinor];
		f64 N = p_AreaMinor[iMinor] * n_use.n;
		
		Tensor2 mat, matxy1, matxy2;
		memset(&mat, 0, sizeof(Tensor2));
		memcpy(&matxy1, &(p_matrix_xy_i[iMinor]), sizeof(Tensor2));
		memcpy(&matxy2, &(p_matrix_xy_e[iMinor]), sizeof(Tensor2));


		// These are the rates of change of MAR given a change in v.
		// Stop to consider this.

		// Are we still going to be adding to v? Might as well as long as 
		// we are using regressors weighted.

		// Careful we do not double-correct.
		// Jacobi regressor: we want to look for 

//
//		if ((TEST_EPSILON_Y_IMINOR) || (TEST_EPSILON_X_MINOR))
//			printf("\n%d mat1 (i) xx %1.9E xy %1.9E yx %1.9E yy %1.9E \nmat2 (e) %1.9E %1.9E %1.9E %1.9E\nhsub %1.11E N %1.11E\n",
//			iMinor, matxy1.xx, matxy1.xy, matxy1.yx, matxy1.yy,
//			matxy2.xx, matxy2.xy, matxy2.yx, matxy2.yy,
//			hsub, N);

		// Think this through carefully.

#ifndef SQRTNV
						
		mat.xx = 1.0;
		mat.yy = 1.0;
		mat.xx += -hsub*((matxy1.xx*m_ion + matxy2.xx*m_e) / ((m_ion + m_e)*N)); 
		mat.xy += -hsub*((matxy1.xy*m_ion + matxy2.xy*m_e) / ((m_ion + m_e)*N));
		mat.yx += -hsub*((matxy1.yx*m_ion + matxy2.yx*m_e) / ((m_ion + m_e)*N));
		mat.yy += -hsub*((matxy1.yy*m_ion + matxy2.yy*m_e) / ((m_ion + m_e)*N));
#else
		// Are we proposing a change in v or sqrt(N) v? v.
		// We actually leave Jacobi unchanged and should take this shortcut.
		// Factor of N^1/2 in epsilon and in the matrix, cancels out.

		mat.xx = 1.0;
		mat.yy = 1.0;

		f64 factor = hsub / ((m_ion + m_e)*N);
#if DECAY_IN_VISC_EQNS

		f64 nu_in_MT = p_nu_in_MT_minor[iMinor];
		f64 nu_en_MT = p_nu_en_MT_minor[iMinor];
		f64 ratio_nn_ntot = n_use.n_n / (n_use.n_n + n_use.n);

		factor /= 1.0 + hsub*nu_in_MT*ratio_nn_ntot;

#endif

		mat.xx += -factor*((matxy1.xx*m_ion + matxy2.xx*m_e));
		mat.xy += -factor*((matxy1.xy*m_ion + matxy2.xy*m_e));
		mat.yx += -factor*((matxy1.yx*m_ion + matxy2.yx*m_e));
		mat.yy += -factor*((matxy1.yy*m_ion + matxy2.yy*m_e));

#endif
		mat.Inverse(invmatrix);




		//if (((TEST_EPSILON_Y_IMINOR) || (TEST_EPSILON_X_MINOR))
		//	||
		//	(iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL))
		//	printf("\n%d mat xx %1.9E xy %1.9E yx %1.9E yy %1.9E \ninvmatrix %1.9E %1.9E %1.9E %1.9E\n\n",
		//		iMinor, mat.xx, mat.xy, mat.yx, mat.yy,
		//		invmatrix.xx, invmatrix.xy, invmatrix.yx, invmatrix.yy);

		// Think carefully.
		// There are 4 things to set and 4 residual epsilon.

		double4 xzyz_i = p_xzyzzxzy__i[iMinor];
		double4 xzyz_e = p_xzyzzxzy__e[iMinor];
		// xz, yz, zx = z, zy = w.

		//if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
		//	// Let's repeat calculation on spreadsheet and see if we get same inverse:

		//	printf("xzyz_i  xz %1.13E yz %1.13E zx %1.14E zy %1.14E\n",
		//		xzyz_i.x, xzyz_i.y, xzyz_i.z, xzyz_i.w);
		//	printf("xzyz_e  xz %1.13E yz %1.13E zx %1.14E zy %1.14E\n",
		//		xzyz_e.x, xzyz_e.y, xzyz_e.z, xzyz_e.w);
		//};

		// Create Jacobi regressor directly:
		// First substitute :
		
		// zxi rx + zyi ry + zzi riz = eps_iz
		// zxe rx + zye ry + zze rez = eps_ez

		// riz = (eps_iz - zxi rx - zyi ry) / zzi;
		// rez = (eps_ez - zxe rx - zye ry) / zze;

		// Therefore:

		// Maybe need to write equations out. mat is meant to involve -hsub/N * 
		// but why do we have the sharing m_i/(m_e+m_i).
		// Meaning of it is vxy effect of viz? vxy is affected by momentum contribution from each species
		f64 hsuboverN = hsub / N;
#if DECAY_IN_VISC_EQNS

		hsuboverN /= 1.0 + hsub*nu_in_MT*ratio_nn_ntot;

#endif
		// So xz means what? px that we get from viz.
		f64 zzi = 1.0 - hsuboverN*p_coeffself_iz[iMinor];
		xzyz_i.x *= -hsuboverN;
		xzyz_i.y *= -hsuboverN;
		xzyz_i.z *= -hsuboverN;
		xzyz_i.w *= -hsuboverN;

#if DECAY_IN_VISC_EQNS
		hsuboverN = hsub / (N + N*hsub*nu_en_MT*ratio_nn_ntot);
		
#endif

		f64 zze = 1.0 - hsuboverN*p_coeffself_ez[iMinor];
		
		xzyz_e.x *= -hsuboverN;
		xzyz_e.y *= -hsuboverN;
		xzyz_e.z *= -hsuboverN;
		xzyz_e.w *= -hsuboverN;

		//if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
		//	// Let's repeat calculation on spreadsheet and see if we get same inverse:

		//	printf("xzyz_i  xz %1.13E yz %1.13E zx %1.14E zy %1.14E zz %1.14E\n",
		//		xzyz_i.x, xzyz_i.y, xzyz_i.z, xzyz_i.w, zzi);
		//	printf("xzyz_e  xz %1.13E yz %1.13E zx %1.14E zy %1.14E zz %1.14E\n",
		//		xzyz_e.x, xzyz_e.y, xzyz_e.z, xzyz_e.w, zze);
		//};

		// zxi now means effect of vxchange on iz residual.
		// (m_i/(m_i+m_e)) xzi is effect of vizchange on vx residual
		
		// vizchange satisfies 
		// zxi vxchange + zyi vychange + zzi vizchange = eps_iz
		// vizchange = (eps_iz - zxi vxchange - zyi vychange) / zzi;
		// vezchange = (eps_ez - zxe vxchange - zye vychange) / zze;

		f64 p_ion = m_ion / (m_ion + m_e);
		f64 p_e = m_e / (m_ion + m_e);

		mat.xx += ( p_ion*(xzyz_i.x // xz
			*(-xzyz_i.z // zx
				/ zzi))
					+ p_e*(xzyz_e.x*(-xzyz_e.z / zze))
					);
		mat.xy += (p_ion*(xzyz_i.x*(-xzyz_i.w / zzi))
			+ p_e*(xzyz_e.x*(-xzyz_e.w / zze))
			);
		mat.yx += (p_ion*(xzyz_i.y*(-xzyz_i.z / zzi))
			+ p_e*(xzyz_e.y*(-xzyz_e.z / zze))
			);
		mat.yy += (p_ion*(xzyz_i.y*(-xzyz_i.w / zzi))
			+ p_e*(xzyz_e.y*(-xzyz_e.w / zze))
			);

	//	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
	//		printf("mat updated xy %1.13E xy %1.13E yx %1.14E yy %1.14E p_ion %1.14E p_e %1.14E\n",
	//			mat.xx, mat.xy, mat.yx, mat.yy, p_ion, p_e);


		// Modify RHS:
		// ... + xzyz_i.x*(m_ion/(m_i+m_e))*(eps_iz/zzi) = eps_x

		f64_vec2 eps_xy = p_epsxy[iMinor];
		f64 eps_iz = p_epsiz[iMinor];
		f64 eps_ez = p_epsez[iMinor];

#ifdef SQRTNV
		// To make it match we have to divide the loaded epsilon by sqrt N.

		f64 sqrtN = sqrt(N);
		eps_xy /= sqrtN;
		eps_iz /= sqrtN;
		eps_ez /= sqrtN;
		// Because our matrix, for rate of change of epsilon, would be missing a factor sqrt(N).
#endif
	
		eps_xy.x -= xzyz_i.x*(p_ion)*(eps_iz / zzi) + xzyz_e.x*(p_e)*(eps_ez / zze);
		eps_xy.y -= xzyz_i.y*(p_ion)*(eps_iz / zzi) + xzyz_e.y*(p_e)*(eps_ez / zze);
		
		f64_tens2 invmatrix2;
		mat.Inverse(invmatrix2);
		f64_vec2 regrxy;
		regrxy.x = invmatrix2.xx*eps_xy.x + invmatrix2.xy*eps_xy.y;
		regrxy.y = invmatrix2.yx*eps_xy.x + invmatrix2.yy*eps_xy.y;
		p_Jaciz[iMinor] = (eps_iz - xzyz_i.z*regrxy.x - xzyz_i.w*regrxy.y) / zzi;
		p_Jacez[iMinor] = (eps_ez - xzyz_e.z*regrxy.x - xzyz_e.w*regrxy.y) / zze;
		p_Jacxy[iMinor] = regrxy;

	//	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
	//		// Let's repeat calculation on spreadsheet and see if we get same inverse:			
	//		printf("Jac xy %1.13E %1.13E iz %1.14E ez %1.14E\n\n",
	//			regrxy.x, regrxy.y, p_Jaciz[iMinor], p_Jacez[iMinor]);			
	//	};

		invcoeffselfviz = 1.0 / zzi; // rate of change of epsilon iz wrt viz
		invcoeffselfvez = 1.0 / zze;
		invcoeffself_x = 1.0 / mat.xx;
		invcoeffself_y = 1.0 / mat.yy;

#ifdef SQRTNV

		invmatrix *= sqrtN;
		invcoeffselfviz *= sqrtN;
		invcoeffselfvez *= sqrtN;
		invcoeffself_x *= sqrtN;
		invcoeffself_y *= sqrtN;

		// What we save off, is for eps[Nv]'s rate of change. This way we can enter such as deps[Nv]/dbeta as numerator.

#endif
	} else {
		// epsilon = 0
	};
	
	memcpy(&(p_invmatrix[iMinor]), &invmatrix, sizeof(Tensor2));
	p_invcoeffselfviz[iMinor] = invcoeffselfviz;
	p_invcoeffselfvez[iMinor] = invcoeffselfvez;
	p_invcoeffselfx[iMinor] = invcoeffself_x;
	p_invcoeffselfy[iMinor] = invcoeffself_y;
}
 


__global__ void kernelCreateDByDBetaCoeffmatrix2(
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
	f64 * __restrict__ p_Jacez

) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];

	Tensor2 invmatrix;
	memset(&invmatrix, 0, sizeof(Tensor2));
	f64 invcoeffselfviz = 0.0;
	f64 invcoeffselfvez = 0.0;
	f64 invcoeffself_x = 0.0;
	f64 invcoeffself_y = 0.0;

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;

		Tensor2 mat, matxy1, matxy2;
		memset(&mat, 0, sizeof(Tensor2));
		memcpy(&matxy1, &(p_matrix_xy_i[iMinor]), sizeof(Tensor2));
		memcpy(&matxy2, &(p_matrix_xy_e[iMinor]), sizeof(Tensor2));


		// These are the rates of change of MAR given a change in v.
		// Stop to consider this.

		// Are we still going to be adding to v? Might as well as long as 
		// we are using regressors weighted.

		// Careful we do not double-correct.
		// Jacobi regressor: we want to look for 

		//
		//		if ((TEST_EPSILON_Y_IMINOR) || (TEST_EPSILON_X_MINOR))
		//			printf("\n%d mat1 (i) xx %1.9E xy %1.9E yx %1.9E yy %1.9E \nmat2 (e) %1.9E %1.9E %1.9E %1.9E\nhsub %1.11E N %1.11E\n",
		//			iMinor, matxy1.xx, matxy1.xy, matxy1.yx, matxy1.yy,
		//			matxy2.xx, matxy2.xy, matxy2.yx, matxy2.yy,
		//			hsub, N);

		// Think this through carefully.

#ifndef SQRTNV


		mat.xx = 1.0;
		mat.yy = 1.0;
		mat.xx += -hsub*((matxy1.xx*m_ion + matxy2.xx*m_e) / ((m_ion + m_e)*N));
		mat.xy += -hsub*((matxy1.xy*m_ion + matxy2.xy*m_e) / ((m_ion + m_e)*N));
		mat.yx += -hsub*((matxy1.yx*m_ion + matxy2.yx*m_e) / ((m_ion + m_e)*N));
		mat.yy += -hsub*((matxy1.yy*m_ion + matxy2.yy*m_e) / ((m_ion + m_e)*N));
#else
		// Are we proposing a change in v or sqrt(N) v? v.
		// We actually leave Jacobi unchanged and should take this shortcut.
		// Factor of N^1/2 in epsilon and in the matrix, cancels out.

		mat.xx = 1.0;
		mat.yy = 1.0;
		mat.xx += -hsub*((matxy1.xx*m_ion + matxy2.xx*m_e) / ((m_ion + m_e)*N));
		mat.xy += -hsub*((matxy1.xy*m_ion + matxy2.xy*m_e) / ((m_ion + m_e)*N));
		mat.yx += -hsub*((matxy1.yx*m_ion + matxy2.yx*m_e) / ((m_ion + m_e)*N));
		mat.yy += -hsub*((matxy1.yy*m_ion + matxy2.yy*m_e) / ((m_ion + m_e)*N));

#endif
		mat.Inverse(invmatrix);

		//if (((TEST_EPSILON_Y_IMINOR) || (TEST_EPSILON_X_MINOR))
		//	||
		//	(iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL))
		//	printf("\n%d mat xx %1.9E xy %1.9E yx %1.9E yy %1.9E \ninvmatrix %1.9E %1.9E %1.9E %1.9E\n\n",
		//		iMinor, mat.xx, mat.xy, mat.yx, mat.yy,
		//		invmatrix.xx, invmatrix.xy, invmatrix.yx, invmatrix.yy);

		// Think carefully.
		// There are 4 things to set and 4 residual epsilon.

		double4 xzyz_i = p_xzyzzxzy__i[iMinor];
		double4 xzyz_e = p_xzyzzxzy__e[iMinor];
		// xz, yz, zx = z, zy = w.

		//if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
		//	// Let's repeat calculation on spreadsheet and see if we get same inverse:

		//	printf("xzyz_i  xz %1.13E yz %1.13E zx %1.14E zy %1.14E\n",
		//		xzyz_i.x, xzyz_i.y, xzyz_i.z, xzyz_i.w);
		//	printf("xzyz_e  xz %1.13E yz %1.13E zx %1.14E zy %1.14E\n",
		//		xzyz_e.x, xzyz_e.y, xzyz_e.z, xzyz_e.w);
		//};

		// Create Jacobi regressor directly:
		// First substitute :

		// zxi rx + zyi ry + zzi riz = eps_iz
		// zxe rx + zye ry + zze rez = eps_ez

		// riz = (eps_iz - zxi rx - zyi ry) / zzi;
		// rez = (eps_ez - zxe rx - zye ry) / zze;

		// Therefore:

		// Maybe need to write equations out. mat is meant to involve -hsub/N * 
		// but why do we have the sharing m_i/(m_e+m_i).
		// Meaning of it is vxy effect of viz? vxy is affected by momentum contribution from each species
		f64 hsuboverN = hsub / N;
		// So xz means what? px that we get from viz.
		f64 zzi = 1.0 - hsuboverN*p_coeffself_iz[iMinor];
		f64 zze = 1.0 - hsuboverN*p_coeffself_ez[iMinor];

		xzyz_i.x *= -hsuboverN;
		xzyz_i.y *= -hsuboverN;
		xzyz_i.z *= -hsuboverN;
		xzyz_i.w *= -hsuboverN;
		xzyz_e.x *= -hsuboverN;
		xzyz_e.y *= -hsuboverN;
		xzyz_e.z *= -hsuboverN;
		xzyz_e.w *= -hsuboverN;

		//if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
		//	// Let's repeat calculation on spreadsheet and see if we get same inverse:

		//	printf("xzyz_i  xz %1.13E yz %1.13E zx %1.14E zy %1.14E zz %1.14E\n",
		//		xzyz_i.x, xzyz_i.y, xzyz_i.z, xzyz_i.w, zzi);
		//	printf("xzyz_e  xz %1.13E yz %1.13E zx %1.14E zy %1.14E zz %1.14E\n",
		//		xzyz_e.x, xzyz_e.y, xzyz_e.z, xzyz_e.w, zze);
		//};

		// zxi now means effect of vxchange on iz residual.
		// (m_i/(m_i+m_e)) xzi is effect of vizchange on vx residual

		// vizchange satisfies 
		// zxi vxchange + zyi vychange + zzi vizchange = eps_iz
		// vizchange = (eps_iz - zxi vxchange - zyi vychange) / zzi;
		// vezchange = (eps_ez - zxe vxchange - zye vychange) / zze;

		f64 p_ion = m_ion / (m_ion + m_e);
		f64 p_e = m_e / (m_ion + m_e);

		mat.xx += (p_ion*(xzyz_i.x // xz
			*(-xzyz_i.z // zx
				/ zzi))
			+ p_e*(xzyz_e.x*(-xzyz_e.z / zze))
			);
		mat.xy += (p_ion*(xzyz_i.x*(-xzyz_i.w / zzi))
			+ p_e*(xzyz_e.x*(-xzyz_e.w / zze))
			);
		mat.yx += (p_ion*(xzyz_i.y*(-xzyz_i.z / zzi))
			+ p_e*(xzyz_e.y*(-xzyz_e.z / zze))
			);
		mat.yy += (p_ion*(xzyz_i.y*(-xzyz_i.w / zzi))
			+ p_e*(xzyz_e.y*(-xzyz_e.w / zze))
			);

		//	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
		//		printf("mat updated xy %1.13E xy %1.13E yx %1.14E yy %1.14E p_ion %1.14E p_e %1.14E\n",
		//			mat.xx, mat.xy, mat.yx, mat.yy, p_ion, p_e);


		// Modify RHS:
		// ... + xzyz_i.x*(m_ion/(m_i+m_e))*(eps_iz/zzi) = eps_x

		f64_vec2 eps_xy = p_epsxy[iMinor];
		f64 eps_iz = p_epsiz[iMinor];
		f64 eps_ez = p_epsez[iMinor];

#ifdef SQRTNV
		// To make it match we have to divide the loaded epsilon by sqrt N.

		f64 sqrtN = sqrt(N);
		eps_xy /= sqrtN;
		eps_iz /= sqrtN;
		eps_ez /= sqrtN;
		// Because our matrix, for rate of change of epsilon, would be missing a factor sqrt(N).

#endif
		
		eps_xy.x -= xzyz_i.x*(p_ion)*(eps_iz / zzi) + xzyz_e.x*(p_e)*(eps_ez / zze);
		eps_xy.y -= xzyz_i.y*(p_ion)*(eps_iz / zzi) + xzyz_e.y*(p_e)*(eps_ez / zze);
		
		f64_tens2 invmatrix2;
		mat.Inverse(invmatrix2);
		f64_vec2 regrxy;
		regrxy.x = invmatrix2.xx*eps_xy.x + invmatrix2.xy*eps_xy.y;
		regrxy.y = invmatrix2.yx*eps_xy.x + invmatrix2.yy*eps_xy.y;
		p_Jaciz[iMinor] = (eps_iz - xzyz_i.z*regrxy.x - xzyz_i.w*regrxy.y) / zzi;
		p_Jacez[iMinor] = (eps_ez - xzyz_e.z*regrxy.x - xzyz_e.w*regrxy.y) / zze;
		p_Jacxy[iMinor] = regrxy;

		// What we save off, is for eps[Nv]'s rate of change. This way we can enter such as deps[Nv]/dbeta as numerator.

	} else {
		// epsilon = 0
	};
}



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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{
		f64_vec2 epsilon_xy = p_epsxyold[iMinor];
		f64 epsilon_iz = p_epsizold[iMinor];
		f64 epsilon_ez = p_epsezold[iMinor];

	//	if (iMinor==CHOSEN) printf("Predict: %d epsold x y %1.12E %1.12E ", iMinor, epsilon_xy.x, epsilon_xy.y);
		
		for (int i = 0; i < REGRESSORS; i++)
		{
			epsilon_xy += beta_n_c[i] * p_d_epsxy_by_d[iMinor + i*NMINOR];
			epsilon_iz += beta_n_c[i] * p_d_epsiz_by_d[iMinor + i*NMINOR];
			epsilon_ez += beta_n_c[i] * p_d_epsez_by_d[iMinor + i*NMINOR];
		//	if (iMinor == CHOSEN) {
		//	printf("epsilon_xy %1.12E %1.12E beta %1.9E deps %1.9E \n", epsilon_xy.x, epsilon_xy.y,
		//		beta_n_c[i], p_d_epsxy_by_d[iMinor + i*NMINOR].y);
		//	};
		}

		p_predictxy[iMinor] = epsilon_xy;
		p_predictiz[iMinor] = epsilon_iz;
		p_predictez[iMinor] = epsilon_ez;


	}
	else {
		p_predictxy[iMinor].x = 0.0;
		p_predictxy[iMinor].y = 0.0;
		p_predictiz[iMinor] = 0.0;
		p_predictez[iMinor] = 0.0;
	};
}


__global__ void kernelCreateNeutralInverseCoeffself(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ p__coeffself_input,
	f64 * __restrict__ p__invcoeffself_output
	)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];

	f64 invcoeffself = 0.0;

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;

		// Think this through carefully.

		invcoeffself = 1.0 / (1.0 - hsub*(p__coeffself_input[iMinor] / N)); // rate of change of epsilon iz wrt viz
		
		if (0) { //((iMinor > BEGINNING_OF_CENTRAL) && ((iMinor-BEGINNING_OF_CENTRAL) % 8000 == 0)) {
			printf("iVertex %d coeffself %1.9E = 1- h/N %1.9E ; h/N = %1.8E\n",
				iMinor-BEGINNING_OF_CENTRAL, 1.0 - hsub*(p__coeffself_input[iMinor] / N),
				p__coeffself_input[iMinor], hsub/N);
		}

	} else {
		// epsilon = 0
	};

	p__invcoeffself_output[iMinor] = invcoeffself;	
}


__global__ void kernelCreateJacobiRegressorz
(f64 * __restrict__ p_regr,
	f64 * __restrict__ p_factor1,
	f64 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	p_regr[iMinor] = p_factor1[iMinor] * p_factor2[iMinor];
//	if (0) { //((iMinor > BEGINNING_OF_CENTRAL) && ((iMinor - BEGINNING_OF_CENTRAL) % 8000 == 0)) {
//		printf("iVertex %d regr %1.10E factor1 %1.9E factor2 %1.9E \n",
//			iMinor-BEGINNING_OF_CENTRAL, p_regr[iMinor], p_factor1[iMinor], p_factor2[iMinor]);
//	}

}


__global__ void kernelCreateJacobiRegressorxy
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_factoreps,
	f64_tens2 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	p_regr[iMinor] = p_factor2[iMinor] * p_factoreps[iMinor];
//	if (0) // iMinor == CHOSEN)
//		printf("CreateJacobiRegressorxy : %d : xx %1.10E xy %1.10E yx %1.10E yy %1.10E eps.xy %1.10E %1.10E\n"
//			"Jacobi regressor %1.13E %1.13E \n", iMinor,
//			p_factor2[iMinor].xx, p_factor2[iMinor].xy, 
//			p_factor2[iMinor].yx, p_factor2[iMinor].yy, p_factoreps[iMinor].x, p_factoreps[iMinor].y,
//			p_regr[iMinor].x, p_regr[iMinor].y);

}

__global__ void kernelCreateJacobiRegressor_x
(	f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 vec2;
	vec2.x = p_factor2[iMinor] * p_eps[iMinor].x;
	vec2.y = 0.0;
	p_regr[iMinor] = vec2;
}

__global__ void kernelCreateJacobiRegressor_y
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_eps,
	f64 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 vec2;
	vec2.y = p_factor2[iMinor] * p_eps[iMinor].y;
	vec2.x = 0.0;
	p_regr[iMinor] = vec2;
}

__global__ void kernelCreateJacobiRegressorNeutralxy2 
(f64_vec2 * __restrict__ p_regr,
	f64 * __restrict__ p_factorepsx,
	f64 * __restrict__ p_factorepsy,
	f64 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 regr2;
	regr2.x = p_factorepsx[iMinor]; 
	regr2.y = p_factorepsy[iMinor]; 
	p_regr[iMinor] = p_factor2[iMinor]*regr2;
}

__global__ void AssembleVector2(
	f64_vec2 * __restrict__ p_vec2,
	f64 * __restrict__ p__x,
	f64 * __restrict__ p__y)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 vec2;
	vec2.x = p__x[iMinor];
	vec2.y = p__y[iMinor];
	p_vec2[iMinor] = vec2;
}

__global__ void kernelCreateJacobiRegressorNeutralxy
( f64_vec2 * __restrict__ p_out,
	f64_vec2 * __restrict__ p_in,
	f64 * __restrict__ p_invcoeffself)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 vec2 = p_in[iMinor] * p_invcoeffself[iMinor];
	p_out[iMinor] = vec2;
}

__global__ void AddLCtoVector4
(v4 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	v4 * __restrict__ p_storemove)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;

	v4 operand = p_operand[index];
	v4 old = operand;
	v4 move;
	int i, j;
	for (i = 0; i < REGRESSORS; i++)
	{
		operand.vxy += beta_n_c[i] * p_regr2[index + i*NMINOR];
		operand.viz += beta_n_c[i] * p_regr_iz[index + i*NMINOR];
		operand.vez += beta_n_c[i] * p_regr_ez[index + i*NMINOR];		
	};
	move.vxy = operand.vxy - old.vxy;
	move.viz = operand.viz - old.viz;
	move.vez = operand.vez - old.vez;
	p_storemove[index] = move;
	p_operand[index] = operand;
}

__global__ void AddLCtoVector3
(f64_vec3 * __restrict__ p_operand,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_z,
	f64_vec2 * __restrict__ p_storemove_xy,
	f64 * __restrict__ p_storemove_z)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;

	f64_vec3 operand = p_operand[index];
	f64_vec3 old = operand;
	f64_vec2 movexy;
	int i, j;
	for (i = 0; i < REGRESSORS; i++)
	{
		f64_vec2 temp2 = p_regr2[index + i*NMINOR];
		operand.x += beta_n_c[i] * temp2.x;
		operand.y += beta_n_c[i] * temp2.y;
		operand.z += beta_n_c[i] * p_regr_z[index + i*NMINOR];
	};
	movexy.x = operand.x - old.x;
	movexy.y = operand.y - old.y;
	p_storemove_xy[index] = movexy;
	p_storemove_z[index] = operand.z - old.z;
	p_operand[index] = operand;
}
__global__ void Multiply_components_xy // c=b-a
(f64_vec2 * __restrict__ p_result,
	f64_vec2 * __restrict__ p_multiplicand_1,
	f64_vec2 * __restrict__ p_multiplicand_2)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;

	f64_vec2 result;
	f64_vec2 in1 = p_multiplicand_1[index];
	f64_vec2 in2 = p_multiplicand_2[index];
	result.x = in1.x*in2.x;
	result.y = in1.y*in2.y;
	p_result[index] = result;
}
__global__ void
kernelPopulateRegressors_from_iRing_RHS
(f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez,
	f64_vec2 * __restrict__ p_regr2_1,
	f64 * __restrict__ p_regr_iz_1,
	f64 * __restrict__ p_regr_ez_1,
	f64_vec2 * __restrict__ p_regr2_2,
	f64 * __restrict__ p_regr_iz_2,
	f64 * __restrict__ p_regr_ez_2,
	bool * __restrict__ p_selected,
	short * __restrict__ p_eqn_index,
	int * __restrict__ p_Ring,
	f64 * __restrict__ p_solution,
	int const whicRing)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;

	if (p_selected[index]) {
		int ring = p_Ring[index];
		short eqnindex = p_eqn_index[index];
		double4 jillium;
		memcpy(&jillium, &(p_solution[4*eqnindex]), sizeof(f64) * 4);

		// Important:
		// regressor 2 corresponds to whichRing-1, whichRing-2
		
		if (ring < whicRing - 3)
		{
			p_regr2[index].x = jillium.x;
			p_regr2[index].y = jillium.y;
			p_regr_iz[index] = jillium.z;
			p_regr_ez[index] = jillium.w; // don't get them wrong way round.
		} else {
			if (ring == whicRing - 3) // whichRing-1 and whichRing-2 are the last two rings.
			{
				p_regr2_1[index].x = jillium.x;
				p_regr2_1[index].y = jillium.y;
				p_regr_iz_1[index] = jillium.z;
				p_regr_ez_1[index] = jillium.w;
			} else {
				p_regr2_2[index].x = jillium.x; // whichRing-1, whichRing-2
				p_regr2_2[index].y = jillium.y;
				p_regr_iz_2[index] = jillium.z;
				p_regr_ez_2[index] = jillium.w;
			};
		};
	} else {
		// leave them zero
	};
}

__global__ void
kernelAddSolution(
	v4 * __restrict__ p_v,
	bool * __restrict__ p_bSelected,
	short * __restrict__ p_eqn_index,
	f64 * __restrict__ p_solution)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;

	if (p_bSelected[index])
	{
		v4 v = p_v[index];
		short eqn = p_eqn_index[index];
		printf("%d : before %1.9E %1.9E %1.9E %1.9E eqn_index %d\n", index,
			v.vxy.x, v.vxy.y, v.viz, v.vez, eqn);
		double4 soln;
		memcpy(&soln, p_solution + 4 * eqn, sizeof(double4));
		printf("%d : add %1.9E %1.9E %1.9E %1.9E -- array x %1.9E iz %1.9E ez %1.9E\n",
			index, soln.x, soln.y, soln.z, soln.w,
			p_solution[4*eqn], p_solution[4 * eqn + 2], p_solution[4 * eqn+3]);
		v.vxy.x += soln.x;
		v.vxy.y += soln.y;
		v.viz += soln.z;
		v.vez += soln.w;
		p_v[index] = v;
	};
}

__global__ void
kernelZeroWithinRings(
	f64 * __restrict__ p_operand,
	bool * __restrict__ p_not
)
{
	long const index = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_not[index]) p_operand[index] = 0.0;
}

__global__ void
kernelZeroWithinRings2(
	f64_vec2 * __restrict__ p_operand,
	bool * __restrict__ p_not
)
{
	const f64_vec2 zerovec(0.0, 0.0);
	long const index = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_not[index]) p_operand[index] = zerovec;
}


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
	)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];	
	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		v4 vie;
		v4 vie_k = p_vie_k[iMinor];
		f64_vec3 v_n_k = p_v_n_k[iMinor];
		f64_vec3 v_n;
		f64_vec3 MAR_ion = p_MAR_ion_[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec_[iMinor];
		f64_vec3 MAR_neut = p_MAR_neut_[iMinor];
		f64 area = p_AreaMinor[iMinor];
		nvals nn = p_n_minor[iMinor];
		f64 N = area * nn.n;
		f64 Nn = area * nn.n_n;
		vie.vxy = vie_k.vxy + hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		vie.viz = vie_k.viz + hsub*(MAR_ion.z / N);
		vie.vez = vie_k.vez + hsub*(MAR_elec.z / N);

//		if ((iMinor == 52177) || (iMinor == 52179) 
	//		|| (iMinor == 26089 + BEGINNING_OF_CENTRAL) || (iMinor == 52178)
	//		|| (iMinor == 52147) || (iMinor == 52150)) printf("%d n %1.8E N %1.8E Nneut %1.8E \n", iMinor, nn.n, N, Nn);

		v_n = v_n_k + hsub*MAR_neut / Nn;
		p_v_n[iMinor] = v_n;
		p_vie[iMinor] = vie;
	} else {
		p_v_n[iMinor] = p_v_n_k[iMinor];
		p_vie[iMinor] = p_vie_k[iMinor];
	};
}


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
	)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iMinor];

	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		v4 vie_k = p_vie[iMinor];
		f64_vec3 MAR_ion__1 = p_MAR_ion__1[iMinor];
		f64_vec3 MAR_elec__1 = p_MAR_elec__1[iMinor];
		f64_vec3 MAR_ion__2 = p_MAR_ion__2[iMinor];
		f64_vec3 MAR_elec__2 = p_MAR_elec__2[iMinor];
		f64 area = p_AreaMinor[iMinor];
		nvals nn = p_n[iMinor];
		f64 N = area*nn.n;
		f64 Nn = area*nn.n_n;

		f64_vec2 hROCxy1N = hsub*((MAR_ion__1.xypart()*m_ion + MAR_elec__1.xypart()*m_e) /
			((m_ion + m_e)));
		f64 hROCiz1N = hsub*(MAR_ion__1.z );
		f64 hROCez1N = hsub*(MAR_elec__1.z);

		f64_vec2 hROCxy2N = hsub*((MAR_ion__2.xypart()*m_ion + MAR_elec__2.xypart()*m_e) /
			((m_ion + m_e)));
		f64 hROCiz2N = hsub*(MAR_ion__2.z );
		f64 hROCez2N = hsub*(MAR_elec__2.z );
		
		bool problem = 0;
		//if (ROCxy1.dot(ROCxy2) < -0.5*ROCxy1.dot(ROCxy1)) {
		//	// retreated over half of the distance covered in direction 1.			
		//	distance = fabs(ROCxy1.dot(ROCxy2) / modulus1);
		//	absthreshold = 
		//	if (distance > 0.5*modulus1 + absthreshold) problem = true;			
		//}
		f64 distance, absthreshold;

		if (hROCxy1N.dot(hROCxy2N) < -0.5*hROCxy1N.dot(hROCxy1N)) {
			// retreated over half of the distance covered in direction 1.			
			f64 modulus1 = hROCxy1N.modulus();
			//distance = fabs(hROCxy1N.dot(hROCxy2N) / modulus1);
			distance = fabs(hROCxy1N.dot(hROCxy2N));

			f64 Nmodvxy = N*vie_k.vxy.modulus();
			absthreshold = 1.0e-9 * (Nmodvxy + 1.0e2*N);
			//if (distance > 0.5*modulus1 + absthreshold) problem = true;
			if (distance > 0.5*modulus1*modulus1 + absthreshold*modulus1) problem = true;
			// Got rid of division.
		}

		// What we want : hypothetical forward move |h*ROC2| < |1/2 h*ROC1 + 1e-9 (mod v + 100)|
		// Multiply all by N:
		// |h*ROC2 N| < |1/2 h*ROC1 N + 1e-9 (N mod v  + 100 N)|
		
		if (hROCiz2N*hROCiz1N < -0.5*hROCiz1N*hROCiz1N) {
			distance = fabs(hROCiz2N);
			f64 Nmodviz = N*fabs(vie_k.viz);
			absthreshold = 1.0e-9 * (Nmodviz + 1.0e2*N);  // 
			if (distance > 0.5*fabs(hROCiz1N) + absthreshold) problem = true;
		}
		
		if (hROCez2N*hROCez1N < -0.5*hROCez1N*hROCez1N) {
			distance = fabs(hROCez2N);
			f64 Nmodvez = N*fabs(vie_k.vez);
			absthreshold = 1.0e-9 * (Nmodvez + 1.0e4*N);  // much higher threshold for elec			
			if (distance > 0.5*fabs(hROCez1N) + absthreshold) problem = true;
		}
		
		int writeval = 0;
		if (problem) writeval = 1;
		p_iSelect[iMinor] = writeval;


		// Now neutral:
		f64_vec3 MAR_neut__1 = p_MAR_neut__1[iMinor];
		f64_vec3 MAR_neut__2 = p_MAR_neut__2[iMinor];
		f64_vec3 v_n_k = p_v_n[iMinor];

		problem = false;
		//f64 hROCnx1Nn = hsub*(MAR_neut__1.x);

		// Maybe we should allow xy to be dot product 
		// Doesn't it mean we can end up circling and getting higher though? without 1 move ever turning
		// back on itself?
		// Let's not go for that. Each dimension has to keep going same way.

		//if (hROCnx2Nn*hROCnx1Nn < -0.5*hROCnx1N*hROCnx1N) {
		if (MAR_neut__2.x*MAR_neut__1.x < -0.5*MAR_neut__1.x*MAR_neut__1.x) {
			//distance = fabs(hROCnx2N);
			distance = hsub*fabs(MAR_neut__2.x); 
			f64 Nnmodvx = Nn*fabs(v_n_k.x);
			absthreshold = 1.0e-9 * (Nnmodvx + 1.0e2*Nn);  // 
			if (distance > 0.5*hsub*fabs(MAR_neut__1.x) + absthreshold) problem = true;
		}
		if (MAR_neut__2.y*MAR_neut__1.y < -0.5*MAR_neut__1.y*MAR_neut__1.y) {
			//distance = fabs(hROCny2N);
			distance = hsub*fabs(MAR_neut__2.y);
			f64 Nnmodvy = Nn*fabs(v_n_k.y);
			absthreshold = 1.0e-9 * (Nnmodvy + 1.0e2*Nn);  // 
			if (distance > 0.5*hsub*fabs(MAR_neut__1.y) + absthreshold) problem = true;
		}
		if (MAR_neut__2.z*MAR_neut__1.z < -0.5*MAR_neut__1.z*MAR_neut__1.z) {
			//distance = fabs(hROCnz2N);
			distance = hsub*fabs(MAR_neut__2.z);
			f64 Nnmodvz = Nn*fabs(v_n_k.z);
			absthreshold = 1.0e-9 * (Nnmodvz + 1.0e2*Nn);  // 
			if (distance > 0.5*hsub*fabs(MAR_neut__1.z) + absthreshold) problem = true;
		}

		writeval = 0;
		if (problem) writeval = 1;
		p_iSelectNeut[iMinor] = writeval;
		


	} else {
		p_iSelect[iMinor] = 0;
		p_iSelectNeut[iMinor] = 0;
	}
	
}


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
	int * __restrict__ p_Select)
{
	__shared__ f64_vec3 shared_v_n[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	__shared__ f64_vec3 shared_v_n_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // There is room for some more double in shared per thread.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_v_n[threadIdx.x] = p_v_n_minor[iMinor];
	shared_ita_par[threadIdx.x] = p_ita_neut_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_neut_minor[iMinor];

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			memcpy(&(shared_v_n_verts[threadIdx.x]), &(p_v_n_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = p_ita_neut_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_neut_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.			
			memset(&(shared_v_n_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	// How shall we arrange to do v_n, which is isotropic? Handle this first...
	// Is the v_n coefficient negligible? Check.

	// We actually have to think how to handle the x-y dimension. PopOhms will handle it.

	// We can re-use some shared data -- such as pos and B -- to do both ions and electrons
	// But they use different ita_par and different vez, viz. 
	// Often we don't need to do magnetised ion viscosity when we do magnetised electron.


	if (threadIdx.x < threadsPerTileMajor) {
		memset(&ownrates_visc, 0, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

										// JUST TO GET IT TO RUN: LIMIT OURSELVES TO RADIUS 4.9 : 
										// !
		if ((info.flag == DOMAIN_VERTEX)
			// && (info.pos.modulus() < 4.9) -- if we have this then need in d/dbeta also.
			&& (shared_ita_par_verts[threadIdx.x] > 0.0)
			&& (p_Select[iVertex + BEGINNING_OF_CENTRAL] != 0)			
			)
			//|| (info.flag == OUTERMOST)) 
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

#pragma unroll 
			for (short i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				if (p_Select[izTri[i]] == 0)
				{
					// must be looking into an unselected point
					{
						if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
						{
							if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
							{
								ita_par = shared_ita_par_verts[threadIdx.x];
								nu = shared_nu_verts[threadIdx.x];
							}
							else {
								ita_par = shared_ita_par[izTri[i] - StartMinor];
								nu = shared_nu[izTri[i] - StartMinor];
							};
						}
						else {
							f64 ita_theirs = p_ita_neut_minor[izTri[i]];
							f64 nu_theirs = p_nu_neut_minor[izTri[i]];
							if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
								ita_par = shared_ita_par_verts[threadIdx.x];
								nu = shared_nu_verts[threadIdx.x];
							}
							else {
								ita_par = ita_theirs;
								nu = nu_theirs;
							};

							// I understand why we are still doing minimum ita at the wall but we would ideally like to stop.

						};
					} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

					f64_vec2 gradvx, gradvy, gradvz;
					f64_vec3 htg_diff;
					f64_vec2 edge_normal;

					if (ita_par > 0.0) // note it was the minimum taken.
					{
						f64_vec3 opp_v, prev_v, next_v;
						f64_vec2 opppos, prevpos, nextpos;
						// ideally we might want to leave position out of the loop so that we can avoid reloading it.

						short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
						short inext = i + 1; if (inext >= tri_len) inext = 0;

						if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
						{
							prev_v = shared_v_n[izTri[iprev] - StartMinor];
							prevpos = shared_pos[izTri[iprev] - StartMinor];
						}
						else {
							prev_v = p_v_n_minor[izTri[iprev]];
							prevpos = p_info_minor[izTri[iprev]].pos;
						}
						if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
							prevpos = Clockwise_d*prevpos;
							RotateClockwise(prev_v);
						}
						if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
							prevpos = Anticlockwise_d*prevpos;
							RotateAnticlockwise(prev_v);
						}

						if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
						{
							opp_v = shared_v_n[izTri[i] - StartMinor];
							opppos = shared_pos[izTri[i] - StartMinor];

							//		if (iVertex == VERTCHOSEN) printf("opp_v %1.9E izTri[i] %d \n", opp_v.x, izTri[i]);

						}
						else {
							opp_v = p_v_n_minor[izTri[i]];
							opppos = p_info_minor[izTri[i]].pos;

							//	if (iVertex == VERTCHOSEN) printf("opp_v %1.9E v_n_minor izTri[i] %d \n", opp_v.x, izTri[i]);
						}
						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opppos = Clockwise_d*opppos;
							RotateClockwise(opp_v);
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opppos = Anticlockwise_d*opppos;
							RotateAnticlockwise(opp_v);
						}

						if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
						{
							next_v = shared_v_n[izTri[inext] - StartMinor];
							nextpos = shared_pos[izTri[inext] - StartMinor];
						}
						else {
							next_v = p_v_n_minor[izTri[inext]];
							nextpos = p_info_minor[izTri[inext]].pos;
						}
						if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
							nextpos = Clockwise_d*nextpos;
							RotateClockwise(next_v);
						}
						if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
							nextpos = Anticlockwise_d*nextpos;
							RotateAnticlockwise(next_v);
						}

						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
							edge_normal = ReconstructEdgeNormal(prevpos, info.pos, nextpos, opppos);

#ifdef INS_INS_3POINT
						if (TestDomainPos(prevpos) == false) {

							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n_verts[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n_verts[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n_verts[threadIdx.x].z, next_v.z, opp_v.z
							);

						}
						else {
							if (TestDomainPos(nextpos) == false) {

								gradvx = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_n_verts[threadIdx.x].x, opp_v.x
								);
								gradvy = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_n_verts[threadIdx.x].y, opp_v.y
								);
								gradvz = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_n_verts[threadIdx.x].z, opp_v.z
								);

							}
							else {
								gradvx = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_n_verts[threadIdx.x].x, next_v.x, opp_v.x
								);
								gradvy = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_n_verts[threadIdx.x].y, next_v.y, opp_v.y
								);
								gradvz = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_n_verts[threadIdx.x].z, next_v.z, opp_v.z
								);
							};
						};

#else
						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						{
							// One of the sides is dipped under the insulator -- set transverse deriv to 0.
							// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

							gradvx = (opp_v.x - shared_v_n_verts[threadIdx.x].x)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvy = (opp_v.y - shared_v_n_verts[threadIdx.x].y)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvz = (opp_v.z - shared_v_n_verts[threadIdx.x].z)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);

						}
						else {
							gradvx = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v_n_verts[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v_n_verts[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v_n_verts[threadIdx.x].z, next_v.z, opp_v.z
							);

							// Could switch to the 3 in one function that handles all 3. in one.
						};
						// Simplify:
#endif
						htg_diff.x = shared_v_n_verts[threadIdx.x].x - opp_v.x;
						htg_diff.y = shared_v_n_verts[threadIdx.x].y - opp_v.y;
						htg_diff.z = shared_v_n_verts[threadIdx.x].z - opp_v.z;

						if (TESTNEUTVISC)
							printf("============================\nNeutral viscosity %d tri %d ita_par %1.10E\n"
								"v %1.9E %1.9E %1.9E  opp_v %1.9E %1.9E %1.9E\n"
								"gradvx %1.9E %1.9E gradvy %1.9E %1.9E gradvz %1.9E %1.9E \n"
								"ourpos %1.8E %1.8E prevpos %1.8E %1.8E opppos %1.8E %1.8E nextpos %1.8E %1.8E edge_nor %1.9E %1.9E\n"
								,
								iVertex, izTri[i], ita_par,
								shared_v_n_verts[threadIdx.x].x, shared_v_n_verts[threadIdx.x].y,
								shared_v_n_verts[threadIdx.x].z, opp_v.x, opp_v.y, opp_v.z,
								gradvx.x, gradvx.y, gradvy.x, gradvy.y, gradvz.x, gradvz.y,
								info.pos.x, info.pos.y, prevpos.x, prevpos.y, opppos.x, opppos.y, nextpos.x, nextpos.y,
								edge_normal.x, edge_normal.y);
					}

					// Order of calculations may help things to go out/into scope at the right times so careful with that.

					// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

					if (ita_par > 0.0)
					{
						// For neutral fluid viscosity does not involve dimensional transfers.

						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_n*(ita_par*gradvx.dot(edge_normal)); // if we are looking at higher vz looking out, go up.
						visc_contrib.y = over_m_n*(ita_par*gradvy.dot(edge_normal));
						visc_contrib.z = over_m_n*(ita_par*gradvz.dot(edge_normal));

						//		if (iVertex == VERTCHOSEN) {
						//			printf("visc_contrib %1.9E %1.9E %1.9E  ita %1.10E \n",
						//				visc_contrib.x, visc_contrib.y, visc_contrib.z, ita_par);
						//		}

						ownrates_visc += visc_contrib;
						visc_htg += -THIRD*m_n*(htg_diff.dot(visc_contrib));

						if (TESTNEUTVISC)
							printf("htg_diff %1.9E %1.9E %1.9E visc_contrib %1.9E %1.9E %1.9E visc_htg %1.10E\n"
								,
								htg_diff.x, htg_diff.y, htg_diff.z, visc_contrib.x, visc_contrib.y, visc_contrib.z,
								visc_htg
							);

					}

					// MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
					// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
				
				}; // whether looking into unselected point

			}; // next i

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR_neut[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(p_MAR_neut + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(f64_vec3));

			p_NT_addition_rate[iVertex].NnTn += visc_htg;
			if (TESTNEUTVISC) {
				printf("%d : cumulative d/dt NnTn %1.10E \n", iVertex, p_NT_addition_rate[iVertex].NnTn);
			};
#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iVertex %d NaN ownrates.x\n", iVertex);
			if (ownrates.y != ownrates.y)
				printf("iVertex %d NaN ownrates.y\n", iVertex);
			if (ownrates.z != ownrates.z)
				printf("iVertex %d NaN ownrates.z\n", iVertex);
			if (visc_htg != visc_htg) printf("iVertex %d NAN VISC HTG\n", iVertex);
#endif
		}
		else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];

	// memcpy(&(ownrates), &(p_MAR_ion[iMinor]), sizeof(f64_vec3));
	memset(&ownrates_visc, 0, sizeof(f64_vec3));
	
#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
	visc_htg = 0.0;
#else
	f64 visc_htg0, visc_htg1, visc_htg2;
	visc_htg0 = 0.0;
	visc_htg1 = 0.0;
	visc_htg2 = 0.0;

#endif

	{
		long izNeighMinor[6];
		char szPBC[6];

		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
			// && (info.pos.modulus() < 4.9) // if we have this then we have to have it in d/dbeta routine also.
			
			// TestDomainPos is not needed as ita is always set to 0 if it does not pass TestDomainPos anyway.

			&& (p_Select[iMinor] != 0)

			&& (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				bool bUsableSide = true;
				if (p_Select[izNeighMinor[i]] == 0)
				{
					{


						if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
						{
							if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							}
							else {
								ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
								nu = shared_nu[izNeighMinor[i] - StartMinor];
							};

							if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
						}
						else {
							if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
								{
									ita_par = shared_ita_par[threadIdx.x];
									nu = shared_nu[threadIdx.x];
								}
								else {
									ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
									nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								};
								if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
							}
							else {
								f64 ita_par_opp = p_ita_neut_minor[izNeighMinor[i]];
								f64 nu_theirs = p_nu_neut_minor[izNeighMinor[i]];
								if (shared_ita_par[threadIdx.x] < ita_par_opp) {
									ita_par = shared_ita_par[threadIdx.x];
									nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
								}
								else {
									ita_par = ita_par_opp;
									nu = nu_theirs;
								}
								if (ita_par_opp == 0.0) bUsableSide = false;
							};
						}
					}
					// basically bUsableSide here just depends on whether min(ita, ita_opp) == 0.

					bool bLongi = false;
#ifdef INS_INS_NONE
					// Get rid of ins-ins triangle traffic:
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS)
							bUsableSide = 0;
					}
					//		if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					//			bLongi = true;
					// have to put it below
#else
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS)
							bLongi = true;
					}
#endif


					f64_vec2 gradvx, gradvy, gradvz;
					f64_vec2 edge_normal;
					f64_vec3 htg_diff;

					if (bUsableSide)
					{
						short inext = i + 1; if (inext > 5) inext = 0;
						short iprev = i - 1; if (iprev < 0) iprev = 5;
						f64_vec3 prev_v, opp_v, next_v;
						f64_vec2 prevpos, nextpos, opppos;

						if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
						{
							memcpy(&prev_v, &(shared_v_n[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
							prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
						}
						else {
							if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								memcpy(&prev_v, &(shared_v_n_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
								prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
							}
							else {
								prevpos = p_info_minor[izNeighMinor[iprev]].pos;
								memcpy(&prev_v, &(p_v_n_minor[izNeighMinor[iprev]]), sizeof(f64_vec3));

							};
						};
						if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
							prevpos = Clockwise_d*prevpos;
							RotateClockwise(prev_v);
						};
						if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
							prevpos = Anticlockwise_d*prevpos;
							RotateAnticlockwise(prev_v);
						};

						if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
						{
							memcpy(&opp_v, &(shared_v_n[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
							opppos = shared_pos[izNeighMinor[i] - StartMinor];
						}
						else {
							if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								memcpy(&opp_v, &(shared_v_n_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
								opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
							}
							else {
								opppos = p_info_minor[izNeighMinor[i]].pos;
								memcpy(&opp_v, &(p_v_n_minor[izNeighMinor[i]]), sizeof(f64_vec3));
							};
						};
						if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
							opppos = Clockwise_d*opppos;
							RotateClockwise(opp_v);
						}
						if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
							opppos = Anticlockwise_d*opppos;
							RotateAnticlockwise(opp_v);
						}

						if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
						{
							memcpy(&next_v, &(shared_v_n[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
							nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
						}
						else {
							if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
								(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
							{
								memcpy(&next_v, &(shared_v_n_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
								nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
							}
							else {
								nextpos = p_info_minor[izNeighMinor[inext]].pos;
								memcpy(&next_v, &(p_v_n_minor[izNeighMinor[inext]]), sizeof(f64_vec3));
							};
						};
						if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
							nextpos = Clockwise_d*nextpos;
							RotateClockwise(next_v);
						};
						if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
							nextpos = Anticlockwise_d*nextpos;
							RotateAnticlockwise(next_v);
						};

						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
							bLongi = true;
#ifdef INS_INS_3POINT
						if (TestDomainPos(prevpos) == false) {

							gradvx = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient_3Point(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								shared_v_n[threadIdx.x].z, next_v.z, opp_v.z
							);

						}
						else {
							if (TestDomainPos(nextpos) == false) {

								gradvx = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_n[threadIdx.x].x, opp_v.x
								);
								gradvy = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_n[threadIdx.x].y, opp_v.y
								);
								gradvz = GetGradient_3Point(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_n[threadIdx.x].z, opp_v.z
								);

							}
							else {

								gradvx = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.x, shared_v_n[threadIdx.x].x, next_v.x, opp_v.x
								);
								gradvy = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.y, shared_v_n[threadIdx.x].y, next_v.y, opp_v.y
								);
								gradvz = GetGradient(
									//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
									prevpos, info.pos, nextpos, opppos,
									//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
									prev_v.z, shared_v_n[threadIdx.x].z, next_v.z, opp_v.z
								);

							};
						};
#else
						if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
						{
							// One of the sides is dipped under the insulator -- set transverse deriv to 0.
							// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

							gradvx = (opp_v.x - shared_v_n[threadIdx.x].x)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvy = (opp_v.y - shared_v_n[threadIdx.x].y)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);
							gradvz = (opp_v.z - shared_v_n[threadIdx.x].z)*(opppos - info.pos) /
								(opppos - info.pos).dot(opppos - info.pos);

						}
						else {
							gradvx = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.x, shared_v_n[threadIdx.x].x, next_v.x, opp_v.x
							);
							gradvy = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.y, shared_v_n[threadIdx.x].y, next_v.y, opp_v.y
							);
							gradvz = GetGradient(
								//f64_vec2 prevpos, f64_vec2 ourpos, f64_vec2 nextpos, f64_vec2 opppos,
								prevpos, info.pos, nextpos, opppos,
								//f64 prev_v, f64 our_v, f64 next_v, f64 opp_v
								prev_v.z, shared_v_n[threadIdx.x].z, next_v.z, opp_v.z
							);
						}
#endif
#ifdef INS_INS_NONE
						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if (flag == CROSSING_INS) {
								// just set it to 0.
								bUsableSide = false;
								gradvz.x = 0.0;
								gradvz.y = 0.0;
								gradvx.x = 0.0;
								gradvx.y = 0.0;
								gradvy.x = 0.0;
								gradvy.y = 0.0;
							};
						};
#endif

						htg_diff = shared_v_n[threadIdx.x] - opp_v;

						if (TESTNEUTVISC2) {
							printf("%d i %d prev_v %1.10E our_v %1.10E opp_v %1.10E next_v %1.10E\n",
								iMinor, i, prev_v.y, shared_v_n[threadIdx.x].y, opp_v.y, next_v.y);
						};

						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors


																		  //					if (iMinor == CHOSEN) printf("============================\nNeutral viscosity %d %d\n"
																		  //							"v.x %1.9E  opp_v.x %1.9E prev_v.x %1.9E next_v.x %1.9E\n"
																		  //							"ourpos %1.9E %1.9E \n"
																		  //							"prevpos %1.9E %1.9E \n"
																		  //							"opppos %1.9E %1.9E \n"
																		  //							"nextpos %1.9E %1.9E \n"
																		  //							"gradvx %1.9E %1.9E gradvy %1.9E %1.9E edge_nor %1.9E %1.9E\n",
																		  //							iMinor, izNeighMinor[i],
																		  //							shared_v_n[threadIdx.x].x, opp_v.x, prev_v.x, next_v.x,
																		  //							info.pos.x, info.pos.y, prevpos.x, prevpos.y, opppos.x, opppos.y, nextpos.x, nextpos.y,
																		  //							gradvx.x, gradvx.y, gradvy.x, gradvy.y, edge_normal.x, edge_normal.y);
																		  //

						if (bLongi) {
							// move any edge_normal endpoints that are below the insulator,
							// until they are above the insulator.
							edge_normal = ReconstructEdgeNormal(
								prevpos, info.pos, nextpos, opppos
							);
						};

					};
					if (bUsableSide) {

						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_n*ita_par*gradvx.dot(edge_normal);
						visc_contrib.y = over_m_n*ita_par*gradvy.dot(edge_normal);
						visc_contrib.z = over_m_n*ita_par*gradvz.dot(edge_normal);

						// Set to 0 any that are pushing momentum uphill. For neutral this is unphysical.
						//	if (visc_contrib.x*htg_diff.x > 0.0) visc_contrib.x = 0.0;
						// Can't do it because it'll ruin backward solve.

						ownrates_visc += visc_contrib;

						if (TESTNEUTVISC2) {
							printf("%d i %d contrib.y %1.10E gradvy %1.10E %1.10E edge_nml %1.9E %1.9E ita %1.8E /m_n %1.8E cumu %1.9E\n",
								iMinor, i, visc_contrib.y, gradvy.x, gradvy.y, edge_normal.x, edge_normal.y, ita_par, over_m_n, ownrates_visc.y);
						};

						if (i % 2 == 0) {
							// vertex : heat collected by vertex
						}
						else {
							
#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
							visc_htg += -THIRD*m_ion*(htg_diff.dot(visc_contrib));
#else
							f64 heat_addn = -THIRD*m_ion*(htg_diff.dot(visc_contrib));
							if (i == 1) {
								visc_htg0 += 0.5*heat_addn;
								visc_htg1 += 0.5*heat_addn;
							} else {
								if (i == 3) {
									visc_htg1 += 0.5*heat_addn;
									visc_htg2 += 0.5*heat_addn;
								} else {
									visc_htg0 += 0.5*heat_addn;
									visc_htg2 += 0.5*heat_addn;
								};
							};
#endif
						};
					}; // bUsableSide
				}; // whether looking into unselected.
			};// next

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR_neut[iMinor]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(&(p_MAR_neut[iMinor]), &(ownrates), sizeof(f64_vec3));

#ifdef COLLECT_VISC_HTG_IN_TRIANGLES
			p_NT_addition_tri[iMinor].NnTn += visc_htg;
#else

			p_NT_addition_tri[iMinor * 3 + 0].NnTn += visc_htg0;
			p_NT_addition_tri[iMinor * 3 + 1].NnTn += visc_htg1;
			p_NT_addition_tri[iMinor * 3 + 2].NnTn += visc_htg2;

#endif
			

			// We will have to round this up into the vertex heat afterwards.

#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iMinor %d NaN ownrates.x\n", iMinor);
			if (ownrates.y != ownrates.y)
				printf("iMinor %d NaN ownrates.y\n", iMinor);
			if (ownrates.z != ownrates.z)
				printf("iMinor %d NaN ownrates.z\n", iMinor);

			if (visc_htg != visc_htg) printf("iMinor %d NAN VISC HTG\n", iMinor);
#endif

			// We do best by taking each boundary, considering how
			// much heat to add for each one.

		}
		else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope

}

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
	f64 * __restrict__ p_AreaMinor)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	// but we want eps to be SMALLER
	// eps = v - (vk + hsomeofMAR) - h MARremaining/N
	// vk + hMAR/N is what v has to aim for. Makes sense of signs.

	structural info = p_info_minor[iMinor];
	
	if ((
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		))
	{
		v4 vie = p_vie_k[iMinor];
		f64_vec3 v_n = p_v_n_k[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion_[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec_[iMinor];
		f64_vec3 MAR_neut = p_MAR_neut_[iMinor];
		f64 area = p_AreaMinor[iMinor];
		nvals nn = p_n_minor[iMinor];
		f64 N = area * nn.n;
		f64 Nn = area * nn.n_n;

		vie.vxy += hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		vie.viz += hsub*(MAR_ion.z / N);
		vie.vez += + hsub*(MAR_elec.z / N);

		v_n += hsub*MAR_neut / Nn;

		if (p_Select[iMinor] == 0) vie = p_v_updated[iMinor];
		if (p_SelectNeut[iMinor] == 0) v_n = p_v_n_updated[iMinor];
		// That is so that when we assess epsilon we should find it to be 0: v_half - v_updated + 0 = 0.
		// Alternatively we could have left v_half at v_k. But we didn't.

		p_v_n_modify_k[iMinor] = v_n;
		p_vie_modify_k[iMinor] = vie;
	} else {

		v4 vie = p_vie_k[iMinor];
		f64_vec3 v_n = p_v_n_k[iMinor];
		if (p_Select[iMinor] == 0) vie = p_v_updated[iMinor];
		if (p_SelectNeut[iMinor] == 0) v_n = p_v_n_updated[iMinor];
		p_v_n_modify_k[iMinor] = v_n;
		p_vie_modify_k[iMinor] = vie;
	};	
}

__global__ void kernelCreateEpsilon_Visc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_v_n_k,
	f64_vec3 * __restrict__ p_MAR_ion__,
	f64_vec3 * __restrict__ p_MAR_elec__,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_nu_in_MT_minor,
	f64 * __restrict__ p_nu_en_MT_minor,

	f64_vec2 * __restrict__ p_epsilon_xy,
	f64 * __restrict__ p_epsilon_iz,
	f64 * __restrict__ p_epsilon_ez,
	bool * __restrict__ p_bFailedTest,
	int * __restrict__ p_Select
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	if (p_Select[iMinor] == 0) return;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
		
	v4 epsilon;
	memset(&epsilon, 0, sizeof(v4));

	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		v4 vie = p_vie[iMinor];
		v4 vie_k = p_vie_k[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion__[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec__[iMinor];
		nvals n_use = p_n_minor[iMinor];
		f64 N = p_AreaMinor[iMinor] * n_use.n;

#ifndef SQRTNV
		epsilon.vxy = vie.vxy - vie_k.vxy
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		epsilon.viz = vie.viz - vie_k.viz - hsub*(MAR_ion.z / N);
		epsilon.vez = vie.vez - vie_k.vez - hsub*(MAR_elec.z / N);
#else

		f64 sqrtN = sqrt(N);

#if DECAY_IN_VISC_EQNS
		// Now introduce decay factor.

		// We want to anticipate that v will decay.

		// Therefore take
		f64_vec3 v_n_k = p_v_n_k[iMinor];
		// v_k+1 target = (v_k + h/N MAR(k+1) - vnk)/(1+h nu mn/(ms+mn) nn/ntot)+vnk
		f64 nu_in_MT = p_nu_in_MT_minor[iMinor];
		f64 nu_en_MT = p_nu_en_MT_minor[iMinor];

		f64 ratio_nn_ntot = n_use.n_n / (n_use.n_n + n_use.n);

		epsilon.vez = sqrtN*
			(vie.vez - v_n_k.z - (vie_k.vez + hsub*(MAR_elec.z) / N - v_n_k.z) /
								(1.0 + hsub*nu_en_MT*ratio_nn_ntot)
			);
		f64_vec2 putative = vie_k.vxy + hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		epsilon.vxy = sqrtN*
			(vie.vxy - v_n_k.xypart() - (putative - v_n_k.xypart()) /
									(1.0 + hsub*nu_in_MT*ratio_nn_ntot)
				);
		epsilon.viz = sqrtN*
			(vie.viz - v_n_k.z - (vie_k.viz + hsub*MAR_ion.z / N - v_n_k.z) /
								(1.0 + hsub*nu_in_MT*ratio_nn_ntot)
				);
#else

		epsilon.vxy = sqrtN*(vie.vxy - vie_k.vxy
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N)));
		epsilon.viz = sqrtN*(vie.viz - vie_k.viz - hsub*(MAR_ion.z / N));
		epsilon.vez = sqrtN*(vie.vez - vie_k.vez - hsub*(MAR_elec.z / N));

#endif
#endif

	//	if (TEST_EPSILON_X_MINOR)
	//		printf("%d epsilon.vx %1.14E vie.vx %1.14E vie_k.vx %1.14E hsub/N %1.14E\nMAR_ion.x %1.12E MAR_elec.x %1.10E avg'd %1.10E \n-------------\n",
	//			iMinor, epsilon.vxy.x, vie.vxy.x, vie_k.vxy.x, hsub / N,
	//			MAR_ion.x, MAR_elec.x,
	//			(MAR_ion.x*m_ion + MAR_elec.x*m_e) / (m_ion + m_e)
	//		);
	//	if ((TEST_EPSILON_X_MINOR) || (TEST_EPSILON_Y_IMINOR))
	//		printf("%d epsilon.vy %1.14E vie.vy %1.14E vie_k.vy %1.14E hsub/N %1.14E\n"
	//			"factor_i %1.11E factor_e %1.11E MAR_ion.y %1.14E MAR_elec.y %1.14E avg'd %1.14E \n-------------\n",
	//			iMinor, epsilon.vxy.y, vie.vxy.y, vie_k.vxy.y, hsub / N,
	//			-hsub*(m_ion / ((m_ion + m_e)*N)),
	//			-hsub*(m_e / ((m_ion + m_e)*N)),
	//			MAR_ion.y, MAR_elec.y,
	//			(MAR_ion.y*m_ion + MAR_elec.y*m_e) / (m_ion + m_e)
	//		);

#if TEST_EPSILON_Z_VERT
		if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) {
			printf("%d epsilon.ez %1.14E vie.vez %1.14E vie_k.vez %1.14E hsub/N %1.14E N %1.9E "
				"MAR_elec.z %1.12E\n----------\n",
				iMinor, epsilon.vez, vie.vez, vie_k.vez, hsub / N, N,
				MAR_elec.z);

		}
#endif
#if TEST_EPSILON_Z_MINOR
		if (iMinor == CHOSEN) {
			printf("%d epsilon.ez %1.14E vie.vez %1.14E vie_k.vez %1.14E hsub/N %1.14E N %1.8E "
				"MAR_elec.z %1.12E\n----------\n",
				iMinor, epsilon.vez, vie.vez, vie_k.vez, hsub / N, N,
				MAR_elec.z);
				
		}
#endif
		
		if (p_bFailedTest != 0) {
			// Stability criterion:
			// We need to know whether it's still heading the same way, in EACH dimension.
			// For general viscosity we just ask if the dot product of next forward move, with our move, is positive.

			// The onward move is hA v_n . 
			bool bFail = false;
			// if ((v_n.x > v_n_k.x) && (MAR_neut.x < 0.0)) bFail = true; // heading wrong way
			//			if (
			//				  (vie.vxy.x - vie_k.vxy.x)*(m_ion*MAR_ion.x + m_e*MAR_elec.x)
			//				+ (vie.vxy.y - vie_k.vxy.y)*(m_ion*MAR_ion.y + m_e*MAR_elec.y)				
			//					< 0.0) bFail = true;
			//			
			//			if ((vie.viz - vie_k.viz)*MAR_ion.z < 0.0) bFail = true;
			//			if ((vie.vez - vie_k.vez)*MAR_elec.z < 0.0) bFail = true;
			//

			// What means that this is sensibly close to the trajectory?
			// Say we look back and it comes from a v_k that is 1% out
			// ** h should govern how big of an error we can make**

			// Allow a 1% error away from accurate backward move.
			// Allow a small independent drift.

			// after 1.0e-7 it's gained spuriously 1e2 cm/s. But not really because it's still a zero-sum when we do flows.

			f64_vec3 mix = (m_ion*MAR_ion + m_e*MAR_elec) / (m_ion + m_e);
			f64 epsmixz = (m_ion*epsilon.viz + m_e*epsilon.vez) / (m_ion + m_e);
			//if (epsilon.vxy.dot(epsilon.vxy) > (0.0001*(hsub*hsub / (N*N))*(mix.dotxy(mix))
			//	+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub)) bFail = true;

			// New test:
			// Trouble is if we redefine the test then we have to redefine epsilon, which affects every routine.
#ifndef SQRTNV
			if (epsilon.vxy.x*epsilon.vxy.x > 0.0001*(hsub*hsub / (N*N))*mix.x*mix.x
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.vxy.y*epsilon.vxy.y > 0.0001*(hsub*hsub / (N*N))*mix.y*mix.y
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;

			if (epsilon.viz*epsilon.viz > 0.0001*(hsub*hsub / (N*N))*MAR_ion.z*MAR_ion.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.vez*epsilon.vez > 0.0001*(hsub*hsub / (N*N))*MAR_elec.z*MAR_elec.z
				+ ALLOWABLE_v_DRIFT_RATE_ez*hsub*ALLOWABLE_v_DRIFT_RATE_ez*hsub) bFail = true;


			if (epsmixz*epsmixz > 0.00001*(hsub*hsub / (N*N))*mix.z*mix.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.vxy.dot(epsilon.vxy) > 0.00001*(hsub*hsub / (N*N))*mix.dotxy(mix)
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
#else

#define RELPPN 1.0e-7
#define FACTOR_C 1.0e12
			if (epsilon.vxy.x*epsilon.vxy.x > RELPPN*hsub*hsub*mix.x*mix.x/N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;
			if (epsilon.vxy.y*epsilon.vxy.y > RELPPN*hsub*hsub*mix.y*mix.y/N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;

			// whoa having to use braincells, could we just have put difference in sqrt N * 

			if (epsilon.viz*epsilon.viz > RELPPN*hsub*hsub*MAR_ion.z*MAR_ion.z / N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;
			if (epsilon.vez*epsilon.vez > RELPPN*hsub*hsub*MAR_elec.z*MAR_elec.z/N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;
			
			// we multiplied target by sqrtN*sqrtN.

			if (epsmixz*epsmixz > RELPPN*hsub*hsub*mix.z*mix.z/N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;
			if (epsilon.vxy.dot(epsilon.vxy) > RELPPN*hsub*hsub*mix.dotxy(mix)/N
				+ FACTOR_C*hsub*FACTOR_C*hsub) bFail = true;
			
#endif
			// We allow a 0.1-1% deviation from trajectory over time.

			// Previous threshold: 1e-14*(v^2 + 1e4) . absolute part 1e-10 vs
			// 1e(18-24).
			// REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(v_n.dot(v_n) + 1.0e4))))

			if (bFail) p_bFailedTest[blockIdx.x] = true;
		};

	} else {
		// epsilon = 0
	}; 
	p_epsilon_xy[iMinor] = epsilon.vxy;
	p_epsilon_iz[iMinor] = epsilon.viz;
	p_epsilon_ez[iMinor] = epsilon.vez;
}

/*
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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	v4 epsilon;
	memset(&epsilon, 0, sizeof(v4));

	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		v4 vie = p_vie[iMinor];
		v4 vie_k = p_vie_k[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion2[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec2[iMinor];
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;

		epsilon.vxy = vie.vxy - vie_k.vxy
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		epsilon.viz = vie.viz - vie_k.viz - hsub*(MAR_ion.z / N);
		epsilon.vez = vie.vez - vie_k.vez - hsub*(MAR_elec.z / N);
		
		if ((epsilon.vxy.x != epsilon.vxy.x) || (epsilon.vxy.y != epsilon.vxy.y))
			printf("%d epsilon x y %1.8E %1.8E\n",
				iMinor, epsilon.vxy.x, epsilon.vxy.y);

		if (p_iFailedTest != 0) {
			// Stability criterion:
			// We need to know whether it's still heading the same way, in EACH dimension.
			// For general viscosity we just ask if the dot product of next forward move, with our move, is positive.

			// The onward move is hA v_n . 
			bool bFail = false;
			// if ((v_n.x > v_n_k.x) && (MAR_neut.x < 0.0)) bFail = true; // heading wrong way
			//			if (
			//				  (vie.vxy.x - vie_k.vxy.x)*(m_ion*MAR_ion.x + m_e*MAR_elec.x)
			//				+ (vie.vxy.y - vie_k.vxy.y)*(m_ion*MAR_ion.y + m_e*MAR_elec.y)				
			//					< 0.0) bFail = true;
			//			
			//			if ((vie.viz - vie_k.viz)*MAR_ion.z < 0.0) bFail = true;
			//			if ((vie.vez - vie_k.vez)*MAR_elec.z < 0.0) bFail = true;
			//

			// What means that this is sensibly close to the trajectory?
			// Say we look back and it comes from a v_k that is 1% out
			// ** h should govern how big of an error we can make**

			// Allow a 1% error away from accurate backward move.
			// Allow a small independent drift.

			// after 1.0e-7 it's gained spuriously 1e2 cm/s. But not really because it's still a zero-sum when we do flows.

			f64_vec3 mix = (m_ion*MAR_ion + m_e*MAR_elec) / (m_ion + m_e);
			f64 epsmixz = (m_ion*epsilon.viz + m_e*epsilon.vez) / (m_ion + m_e);
			//if (epsilon.vxy.dot(epsilon.vxy) > (0.0001*(hsub*hsub / (N*N))*(mix.dotxy(mix))
			//	+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub)) bFail = true;

			if (epsilon.vxy.x*epsilon.vxy.x > 0.0001*(hsub*hsub / (N*N))*mix.x*mix.x
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) {
				bFail = true;

				epsilon.vxy.x = 1.0;
			} else {
				epsilon.vxy.x = 0.0;
			}
			if (epsilon.vxy.y*epsilon.vxy.y > 0.0001*(hsub*hsub / (N*N))*mix.y*mix.y
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) {
				bFail = true;
				epsilon.vxy.y = 1.0;
			} else {
				epsilon.vxy.y = 0.0;
			}

			if (epsilon.viz*epsilon.viz > 0.0001*(hsub*hsub / (N*N))*MAR_ion.z*MAR_ion.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) {
				bFail = true;
				epsilon.viz = 1.0;
			} else {
				epsilon.viz = 0.0;
			};

			if (epsilon.vez*epsilon.vez > 0.0001*(hsub*hsub / (N*N))*MAR_elec.z*MAR_elec.z
				+ ALLOWABLE_v_DRIFT_RATE_ez*hsub*ALLOWABLE_v_DRIFT_RATE_ez*hsub) {
				bFail = true;
				epsilon.vez = 1.0;
			} else {
				epsilon.vez = 0.0;
			};
			

			if (epsmixz*epsmixz > 0.00001*(hsub*hsub / (N*N))*mix.z*mix.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.vxy.dot(epsilon.vxy) > 0.00001*(hsub*hsub / (N*N))*mix.dotxy(mix)
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;



			// We allow a 0.1-1% deviation from trajectory over time.

			// Previous threshold: 1e-14*(v^2 + 1e4) . absolute part 1e-10 vs
			// 1e(18-24).
			// REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(v_n.dot(v_n) + 1.0e4))))

			if (bFail) p_iFailedTest[iMinor] = 1;
		};


		//	if ((p_bFailedTest != 0) && 
		//		((epsilon.vxy.dot(epsilon.vxy) > REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(vie.vxy.dot(vie.vxy) + 1.0e4))				
		//			||				
		//			(epsilon.viz*epsilon.viz > REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(vie.viz*vie.viz + 1.0e4))
		//			||
		//			(epsilon.vez*epsilon.vez > REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(vie.vez*vie.vez + 1.0e8))
		//			))
		//	{
		//		p_bFailedTest[blockIdx.x] = true;
		//	}

	}
	else {
		// epsilon = 0
	};
	p_epsilon_xy[iMinor] = epsilon.vxy;
	p_epsilon_iz[iMinor] = epsilon.viz;
	p_epsilon_ez[iMinor] = epsilon.vez;
	
}*/

__global__ void KillOutsideRegion(
	f64_vec2 * __restrict__ targ2,
	f64 * __restrict__ targ,
	int * __restrict__ p_Select)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	f64_vec2 const zerovec2(0.0, 0.0);

	int s = p_Select[iMinor];
	if (s == 0) {
		targ2[iMinor] = zerovec2;
		targ[iMinor] = 0.0;
	};
}


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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_Select[iMinor] == 0) return;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	f64_vec3 epsilon;
	memset(&epsilon, 0, sizeof(f64_vec3));
	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		f64_vec3 v_n = p_v_n[iMinor];
		f64_vec3 v_n_k = p_v_n_k[iMinor];
		f64_vec3 MAR_neut = pMAR_neut[iMinor];
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n_n;

		epsilon = v_n - v_n_k - hsub*(MAR_neut) / N;

		if (p_bFailedTest != 0)
		{
			// Stability criterion:
			// We need to know whether it's still heading the same way, in EACH dimension.
			// For general viscosity we just ask if the dot product of next forward move, with our move, is positive.

			// The onward move is hA v_n . 
			bool bFail = false;
			// if ((v_n.x > v_n_k.x) && (MAR_neut.x < 0.0)) bFail = true; // heading wrong way
			//	if ((v_n.x - v_n_k.x)*MAR_neut.x < 0.0) bFail = true;
			//	if ((v_n.y - v_n_k.y)*MAR_neut.y < 0.0) bFail = true;
			//	if ((v_n.z - v_n_k.z)*MAR_neut.z < 0.0) bFail = true;

			// What means that this is sensibly close to the trajectory?
			// Say we look back and it comes from a v_k that is 1% out
			// ** h should govern how big of an error we can make**

			// Allow a 1% error away from accurate backward move.
			// Allow a small independent drift.

			// after 1.0e-7 it's gained spuriously 1e2 cm/s. But not really because it's still a zero-sum when we do flows.

			if (epsilon.x*epsilon.x > 0.0001*(hsub*hsub / (N*N))*MAR_neut.x*MAR_neut.x
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.y*epsilon.y > 0.0001*(hsub*hsub / (N*N))*MAR_neut.y*MAR_neut.y
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.z*epsilon.z > 0.0001*(hsub*hsub / (N*N))*MAR_neut.z*MAR_neut.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;

			if (epsilon.dot(epsilon) > (0.00001*(hsub*hsub / (N*N))*MAR_neut.dot(MAR_neut)
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub)) bFail = true;

			if ((bFail) && (bSwitch)) printf("%d eps %1.8E %1.8E %1.8E h/N MAR %1.8E %1.8E %1.8E Thr %1.8E\n",
				iMinor, epsilon.x, epsilon.y, epsilon.z,
				(hsub / N)*MAR_neut.x, (hsub / N)*MAR_neut.y, (hsub / N)*MAR_neut.z,
				hsub*ALLOWABLE_v_DRIFT_RATE);

			// 1e9*1e-12 = 1e-3 so how can we possibly have L2eps at order 1e-4.


			// Previous threshold: 1e-14*(v^2 + 1e4) . absolute part 1e-10 vs
			// 1e(18-24).
			// REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(v_n.dot(v_n) + 1.0e4))))

			//		if (bFail) printf("%d v_n %1.14E %1.14E %1.14E vk %1.14E %1.14E %1.14E MAR %1.14E %1.14E %1.14E hsub/N %1.14E\n"
			//			"eps %1.14E %1.14E %1.14E move %1.14E %1.14E %1.14E\n",
			//			iMinor, v_n.x, v_n.y, v_n.z, v_n_k.x, v_n_k.y, v_n_k.z, MAR_neut.x, MAR_neut.y, MAR_neut.z, hsub/N, 
			//			epsilon.x, epsilon.y, epsilon.z, v_n.x - v_n_k.x, v_n.y - v_n_k.y, v_n.z - v_n_k.z
			//			);

			if (bFail) p_bFailedTest[blockIdx.x] = true;
		};
	}
	else {
		// epsilon = 0
	};
	p_eps_x[iMinor] = epsilon.x;
	p_eps_y[iMinor] = epsilon.y;
	p_eps_z[iMinor] = epsilon.z;
}


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
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	f64_vec3 epsilon;
	memset(&epsilon, 0, sizeof(f64_vec3));
	if (
		(info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		||
		((info.flag == CROSSING_INS) && (TestDomainPos(info.pos)))
		)
	{
		f64_vec3 v_n = p_v_n[iMinor];
		f64_vec3 v_n_k = p_v_n_k[iMinor];
		f64_vec3 MAR_neut = pMAR_neut[iMinor];
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n_n;

		epsilon = v_n - v_n_k - hsub*(MAR_neut) / N;

		if (p_iFailedTest != 0)
		{
			// Stability criterion:
			// We need to know whether it's still heading the same way, in EACH dimension.
			// For general viscosity we just ask if the dot product of next forward move, with our move, is positive.

			// The onward move is hA v_n . 
			bool bFail = false;
			// if ((v_n.x > v_n_k.x) && (MAR_neut.x < 0.0)) bFail = true; // heading wrong way
			//	if ((v_n.x - v_n_k.x)*MAR_neut.x < 0.0) bFail = true;
			//	if ((v_n.y - v_n_k.y)*MAR_neut.y < 0.0) bFail = true;
			//	if ((v_n.z - v_n_k.z)*MAR_neut.z < 0.0) bFail = true;

			// What means that this is sensibly close to the trajectory?
			// Say we look back and it comes from a v_k that is 1% out
			// ** h should govern how big of an error we can make**

			// Allow a 1% error away from accurate backward move.
			// Allow a small independent drift.

			// after 1.0e-7 it's gained spuriously 1e2 cm/s. But not really because it's still a zero-sum when we do flows.

			if (epsilon.x*epsilon.x > 0.0001*(hsub*hsub / (N*N))*MAR_neut.x*MAR_neut.x
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.y*epsilon.y > 0.0001*(hsub*hsub / (N*N))*MAR_neut.y*MAR_neut.y
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;
			if (epsilon.z*epsilon.z > 0.0001*(hsub*hsub / (N*N))*MAR_neut.z*MAR_neut.z
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub) bFail = true;

			if (epsilon.dot(epsilon) > (0.00001*(hsub*hsub / (N*N))*MAR_neut.dot(MAR_neut)
				+ ALLOWABLE_v_DRIFT_RATE*hsub*ALLOWABLE_v_DRIFT_RATE*hsub)) bFail = true;

			//if ((bFail) && (bSwitch)) printf("%d eps %1.8E %1.8E %1.8E h/N MAR %1.8E %1.8E %1.8E Thr %1.8E\n",
			//	iMinor, epsilon.x, epsilon.y, epsilon.z,
			//	(hsub / N)*MAR_neut.x, (hsub / N)*MAR_neut.y, (hsub / N)*MAR_neut.z,
			//	hsub*ALLOWABLE_v_DRIFT_RATE);

			// 1e9*1e-12 = 1e-3 so how can we possibly have L2eps at order 1e-4.
			
			// Previous threshold: 1e-14*(v^2 + 1e4) . absolute part 1e-10 vs
			// 1e(18-24).
			// REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(v_n.dot(v_n) + 1.0e4))))

			//		if (bFail) printf("%d v_n %1.14E %1.14E %1.14E vk %1.14E %1.14E %1.14E MAR %1.14E %1.14E %1.14E hsub/N %1.14E\n"
			//			"eps %1.14E %1.14E %1.14E move %1.14E %1.14E %1.14E\n",
			//			iMinor, v_n.x, v_n.y, v_n.z, v_n_k.x, v_n_k.y, v_n_k.z, MAR_neut.x, MAR_neut.y, MAR_neut.z, hsub/N, 
			//			epsilon.x, epsilon.y, epsilon.z, v_n.x - v_n_k.x, v_n.y - v_n_k.y, v_n.z - v_n_k.z
			//			);

			if (bFail) p_iFailedTest[iMinor] = 1;
		};
	} else {
		// epsilon = 0
	};
	p_eps_x[iMinor] = epsilon.x;
	p_eps_y[iMinor] = epsilon.y;
	p_eps_z[iMinor] = epsilon.z;
}

__global__ void kernelCreateEpsilon_NeutralVisc__TestStability(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	f64_vec3 * __restrict__ p_v_n_k,
	f64_vec3 * __restrict__ pMAR_neut,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_vec3 * __restrict__ p_eps3,
	bool * __restrict__ p_bFailedTest
) {
	/*long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	f64_vec3 epsilon;
	memset(&epsilon, 0, sizeof(f64_vec3));
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
	|| (info.flag == CROSSING_INS)) // ?
	{
	f64_vec3 v_n = p_v_n[iMinor];
	f64_vec3 v_n_k = p_v_n_k[iMinor];
	f64_vec3 MAR_neut = pMAR_neut[iMinor];
	f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n_n;

	f64_vec3 suggested_move = p_forward_move[iMinor];

	epsilon = v_n - v_n_k - hsub*(MAR_neut) / N;    // v_n-v_n_k is the move we made, h *MAR_neut/N is the move implied.
	// suppose we also load in the initial suggested move.


	bFailed = false;
	if ((epsilon.dot(epsilon) < REL_THRESHOLD_VISC*REL_THRESHOLD_VISC*(v_n.dot(v_n) + 1.0e4))) {
	// it's ok - do nothing
	}
	else {
	// overshooting test:
	if (
	(((v_n.x - v_n_k.x) > 0.0) && ((hsub*MAR_neut.x / N) < -(v_n.x - v_n_k.x)))
	||
	(((v_n.x - v_n_k.x) <= 0.0) && ((hsub*MAR_neut.x / N) > -(v_n.x - v_n_k.x)))
	)
	{
	// it more than reverses itself, once we evaluate MAR_neut at v_n.


	// overshooting:
	bFailed = true;

	}
	else {
	// not overshooting:
	// is it between fwd & bwd?!

	// Suppose for the suggested move, Xk + F(Xinit) - Xinit < 0: using X gives us a result to the left of X.
	// Whereas for bwd, Xk + F(Xbwd) - Xbwd = 0.

	// Too far (beyond bwd): Xk + F(X) - X > 0
	// Not far enough: Xk + F(X) - X < Xk + F(Xinit)-Xinit

	if (
	((initialimageminusinitial.x < 0.0) && (image - X > 0.0))
	||
	((initialimageminusinitial.x > 0.0) && (image - X < 0.0))
	)
	{
	// beyond bwd:
	bFailed = true; // move back towards bwd?
	}
	else {
	//					if (((initialimageminusinitial.x < 0.0) && (image - X < initialimageminusinitial.x))
	//						||
	//						((initialimageminusinitial.x > 0.0) && (image - X > initialimageminusinitial.x))
	//						)
	// ended up the wrong side of fwd, God knows how, but this is no good.


	// yeah this isn't the right test.

	// The right test is the overshooting one, for which we have to
	// run and evaluate

	if (((F(X) < 0.0) && (Xk + F(X) + F(Xk+F(X)) > Xk))
	||
	((F(X) > 0.0) && (Xk + F(X) + F(Xk + F(X)) < Xk)))
	{
	bFailed = true; // still overshooting; move towards bwd
	}
	}
	}
	}

	if ((p_bFailedTest != 0) &&	(bFailed))
	{
	p_bFailedTest[blockIdx.x] = true;
	}

	}
	else {
	// epsilon = 0
	};
	p_eps3[iMinor] = epsilon;*/
}


__global__ void kernelCountNumberMoreThanZero(
	int * __restrict__ p_thing1,
	int * __restrict__ p_thing2,
	long * __restrict__ p_blocktot1,
	long * __restrict__ p_blocktot2)
{
	__shared__ int add1[threadsPerTileMinor];
	__shared__ int add2[threadsPerTileMinor];

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	int thing1 = p_thing1[index];
	int thing2 = p_thing2[index];

	if (thing1 > 0) {
		add1[threadIdx.x] = 1;
	} else {
		add1[threadIdx.x] = 0;
	};
	if (thing2 > 0) {
		add2[threadIdx.x] = 1;
	} else {
		add2[threadIdx.x] = 0;
	};

	// Add:

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			add1[threadIdx.x] += add1[threadIdx.x + k];			
			add2[threadIdx.x] += add2[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			add1[threadIdx.x] += add1[threadIdx.x + s - 1];
			add2[threadIdx.x] += add2[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_blocktot1[blockIdx.x] = (long)add1[0];
		p_blocktot2[blockIdx.x] = (long)add2[0];
	};

}

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
)
{
	//__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input
	// Not used, right? Nothing nonlinear?
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	//__shared__ v4 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Thus putting some stuff in shared may speed up if there are spills.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64_vec2 opppos, prevpos, nextpos;
	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_ita_par[threadIdx.x] = p_ita_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			//	memcpy(&(shared_vie_verts[threadIdx.x]), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = p_ita_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			//	memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (info.pos.modulus() < 4.5)
			&& (shared_ita_par_verts[threadIdx.x] > 0.0))
			//|| (info.flag == OUTERMOST)) // !!!!!!!!!!!!!!!!
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;
			// d_eps_z_by_d_viz = 1.0;  // Note that eps includes v_k+1


#pragma unroll 
			for (short i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think

				{
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
						{
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izTri[i] - StartMinor];
							nu = shared_nu[izTri[i] - StartMinor];
						};
					}
					else {
						f64 ita_theirs = p_ita_minor[izTri[i]];
						f64 nu_theirs = p_nu_minor[izTri[i]];
						if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = ita_theirs;
							nu = nu_theirs;
						};

						// I understand why we are still doing minimum ita at the wall but we would ideally like to stop.

					};
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				f64_vec2 gradvx, gradvy, gradvz;
				f64_vec2 edge_normal;

				if (ita_par > 0.0) // note it was the minimum taken.
				{
					f64_vec2 opppos, prevpos, nextpos;
					// ideally we might want to leave position out of the loop so that we can avoid reloading it.

					short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
					short inext = i + 1; if (inext >= tri_len) inext = 0;

					if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
					{
						prevpos = shared_pos[izTri[iprev] - StartMinor];
					}
					else {
						prevpos = p_info_minor[izTri[iprev]].pos;
					}
					if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
						prevpos = Clockwise_d*prevpos;
					}
					if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
						prevpos = Anticlockwise_d*prevpos;
					}

					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opppos = shared_pos[izTri[i] - StartMinor];

					}
					else {
						opppos = p_info_minor[izTri[i]].pos;

						//	if (iVertex == VERTCHOSEN) printf("opp_v %1.9E v_n_minor izTri[i] %d \n", opp_v.x, izTri[i]);
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opppos = Clockwise_d*opppos;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opppos = Anticlockwise_d*opppos;
					}

					if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
					{
						nextpos = shared_pos[izTri[inext] - StartMinor];
					}
					else {
						nextpos = p_info_minor[izTri[inext]].pos;
					}
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextpos = Clockwise_d*nextpos;
					}
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
						nextpos = Anticlockwise_d*nextpos;
					}

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					edge_normal.x = THIRD * (nextpos.y - prevpos.y);
					edge_normal.y = THIRD * (prevpos.x - nextpos.x);

					f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
					f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;

					// For neutral fluid viscosity does not involve dimensional transfers.

					f64_vec3 zero3(0.0, 0.0, 0.0);

					Augment_JacobeanNeutral(&J,
						hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n_n*p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] * m_n),
						edge_normal, ita_par, nu, zero3,
						grad_vjdx_coeff_on_vj_self,
						grad_vjdy_coeff_on_vj_self
					);
				}
			};

			f64_tens3 result;
			J.Inverse(result);

			memcpy(&(p_matrix_n[iVertex + BEGINNING_OF_CENTRAL]), &result, sizeof(f64_tens3));
			// inverted it so that we are ready to put Jacobi = result.eps

		}
		else {
			// NOT domain vertex: Do nothing			

			// NOTE: We did not include OUTERMOST. Justification / effect ??
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];

	//if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	{
		long izNeighMinor[6];
		char szPBC[6];

		// JUST TO GET IT TO RUN:
		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) &&
			(info.pos.modulus() < 4.9) && (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;

			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				bool bUsableSide = true;
				{
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							}
							else {
								ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							};
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							f64 ita_par_opp = p_ita_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							}
							else {
								ita_par = ita_par_opp;
								nu = nu_theirs;
							}
							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
				}

				f64_vec2 edge_normal;
				f64_vec3 htg_diff;

				if (bUsableSide)
				{
					short inext = i + 1; if (inext > 5) inext = 0;
					short iprev = i - 1; if (iprev < 0) iprev = 5;
					f64_vec2 prevpos, nextpos, opppos;

					if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
					{
						prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
					}
					else {
						if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
						}
						else {
							prevpos = p_info_minor[izNeighMinor[iprev]].pos;
						};
					};
					if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
						prevpos = Clockwise_d*prevpos;
					};
					if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
						prevpos = Anticlockwise_d*prevpos;
					};

					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opppos = shared_pos[izNeighMinor[i] - StartMinor];
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
						}
						else {
							opppos = p_info_minor[izNeighMinor[i]].pos;
						};
					};
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opppos = Clockwise_d*opppos;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opppos = Anticlockwise_d*opppos;
					}

					if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
					{
						nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					}
					else {
						if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						}
						else {
							nextpos = p_info_minor[izNeighMinor[inext]].pos;
						};
					};
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextpos = Clockwise_d*nextpos;
					}
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
						nextpos = Anticlockwise_d*nextpos;
					}

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
					f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;

					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {

							bUsableSide = 0;
							grad_vjdx_coeff_on_vj_self = 0.0;
							grad_vjdy_coeff_on_vj_self = 0.0;
							/*

							f64 prev_vx = p_v_n_minor[izNeighMinor[iprev]].x;
							if (prev_vx == 0.0) // prev is in the insulator.
							{
							// do like the above but it goes (ours, next, opp) somehow?

							f64 area_triangle = 0.5*(
							(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
							+ (opppos.x + info.pos.x)*(opppos.y - info.pos.y)
							+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y));

							//gradvx.x = 0.5*(
							//	(shared_v_n[threadIdx.x].x + next_v.x)*(info.pos.y - nextpos.y)
							//	+ (opp_v.x + shared_v_n[threadIdx.x].x)*(opppos.y - info.pos.y)
							//	+ (next_v.x + opp_v.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
							//	) / area_triangle;

							grad_vjdx_coeff_on_vj_self = 0.5*(opppos.y - nextpos.y) / area_triangle;
							grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - opppos.x) / area_triangle;

							} else {
							f64 next_vx = p_v_n_minor[izNeighMinor[inext]].x;
							if (next_vx == 0.0) // next is in the insulator
							{
							f64 area_triangle = 0.5*(
							(prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
							+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
							+ (info.pos.x + opppos.x)*(info.pos.y - opppos.y)
							);

							grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - opppos.y) / area_triangle;
							grad_vjdy_coeff_on_vj_self = 0.5*(opppos.x - prevpos.x) / area_triangle;

							} else {
							//printf("\n\n\nDid not make sense! Alert RING-TAILED LEMUR. iMinor %d iNiegh %d \n"
							//	"izNeighMinor[inext] %d izNeighMinor[iprev] %d flag %d %d \n"
							//	"prev_v.x %1.8E next_v.x %1.8E \n"
							//	"\n\n\a", iMinor,
							//	izNeighMinor[i],
							//	izNeighMinor[inext], izNeighMinor[iprev], p_info_minor[izNeighMinor[inext]].flag,
							//	p_info_minor[izNeighMinor[iprev]].flag, prev_v.x, next_v.x);
							};
							};
							*/
						};
					};

					edge_normal.x = THIRD * (nextpos.y - prevpos.y);
					edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

					f64_vec3 zero3(0.0, 0.0, 0.0);
					Augment_Jacobean(&J,
						hsub / (p_n_minor[iMinor].n_n * p_AreaMinor[iMinor] * m_n),
						edge_normal, ita_par, nu, zero3,
						grad_vjdx_coeff_on_vj_self,
						grad_vjdy_coeff_on_vj_self
					);

				}; // bUsableSide
			};

			f64_tens3 result;
			J.Inverse(result);
			memcpy(&(p_matrix_n[iMinor]), &result, sizeof(f64_tens3));

		}
		else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope

}

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
)
{
	//__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input
	// Not used, right? Nothing nonlinear?
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	//__shared__ v4 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Thus putting some stuff in shared may speed up if there are spills.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64_vec2 opppos, prevpos, nextpos;
	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_ita_par[threadIdx.x] = p_ita_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			//	memcpy(&(shared_vie_verts[threadIdx.x]), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = p_ita_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			//	memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	f64_tens3 result;
	f64_vec2 cc0, cc1;
	if (threadIdx.x < threadsPerTileMajor) {

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		f64_tens3 result;
		if ((info.flag == DOMAIN_VERTEX) && (info.pos.modulus() < 4.5)
			&& (shared_ita_par_verts[threadIdx.x] > 0.0))
			//|| (info.flag == OUTERMOST)) // !!!!!!!!!!!!!!!!
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			short i = 0;
			f64_vec2 opppos, prevpos, nextpos;
			short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
			short inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opppos = shared_pos[izTri[i] - StartMinor];

			} else {
				opppos = p_info_minor[izTri[i]].pos;

				//	if (iVertex == VERTCHOSEN) printf("opp_v %1.9E v_n_minor izTri[i] %d \n", opp_v.x, izTri[i]);
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			CalculateCircumcenter(&cc0, info.pos, opppos, prevpos);


			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;
			// d_eps_z_by_d_viz = 1.0;  // Note that eps includes v_k+1


#pragma unroll 
			for (short i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				{
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
						{
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izTri[i] - StartMinor];
							nu = shared_nu[izTri[i] - StartMinor];
						};
					}
					else {
						f64 ita_theirs = p_ita_minor[izTri[i]];
						f64 nu_theirs = p_nu_minor[izTri[i]];
						if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = ita_theirs;
							nu = nu_theirs;
						};

						// I understand why we are still doing minimum ita at the wall but we would ideally like to stop.

					};
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				short iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;
				short inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				CalculateCircumcenter(&cc1, opppos, info.pos, nextpos);

				if (ita_par > 0.0) // note it was the minimum taken.
				{
					// ideally we might want to leave position out of the loop so that we can avoid reloading it.

					f64 visc_contrib_x_coeff_on_vx_self =
						over_m_n*ita_par*(-1.0 / (opppos - info.pos).modulus())*(cc1 - cc0).modulus();
					f64 contrib_to_Jxx =
						-(hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n_n*p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL]))
						* visc_contrib_x_coeff_on_vx_self;
					J.xx += contrib_to_Jxx;
					J.yy += contrib_to_Jxx;
					J.zz += contrib_to_Jxx;

					// For neutral fluid viscosity does not involve dimensional transfers.

					//Augment_JacobeanNeutral(&J,
					//	hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n_n*p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] * m_n),
					//	edge_normal, ita_par, nu, zero3,
					//	grad_vjdx_coeff_on_vj_self,
					//	grad_vjdy_coeff_on_vj_self					);

					// 
					//pJ->yy += Factor*
					//	((
					//		// Pi_zx
					//		-ita_par*grad_vjdx_coeff_on_vj_self
					//		)*edge_normal.x + (
					//			// Pi_zy
					//			-ita_par*grad_vjdy_coeff_on_vj_self
					//			)*edge_normal.y);
				}

				cc0 = cc1;
				prevpos = opppos;
				opppos = nextpos;
			}; // next i

			J.Inverse(result);

			// inverted it so that we are ready to put Jacobi = result.eps

		}
		else {

			memset(&result, 0, sizeof(f64_tens3));
			result.xx = 1.0;
			result.yy = 1.0;
			result.zz = 1.0;
			// NOTE: We did not include OUTERMOST. Justification / effect ??
		};
		memcpy(&(p_matrix_n[iVertex + BEGINNING_OF_CENTRAL]), &result, sizeof(f64_tens3));

	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];

	//if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	{
		long izNeighMinor[6];
		char szPBC[6];

		// JUST TO GET IT TO RUN:
		if (((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) &&
			(info.pos.modulus() < 4.9) && (shared_ita_par[threadIdx.x] > 0.0)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short i = 0;
			short inext = i + 1; if (inext > 5) inext = 0;
			short iprev = i - 1; if (iprev < 0) iprev = 5;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			};
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			};

			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			CalculateCircumcenter(&cc0, info.pos, opppos, prevpos);

			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;

			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;
				bool bUsableSide = true;
				{
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							}
							else {
								ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							};
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							f64 ita_par_opp = p_ita_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							}
							else {
								ita_par = ita_par_opp;
								nu = nu_theirs;
							}
							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
				}

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				CalculateCircumcenter(&cc1, info.pos, nextpos, opppos);
				if (bUsableSide)
				{

					f64 visc_contrib_x_coeff_on_vx_self =
						over_m_n*ita_par*(-1.0 / (opppos - info.pos).modulus())*(cc1 - cc0).modulus();
					f64 contrib_to_Jxx =
						-(hsub / (p_n_minor[iMinor].n_n*p_AreaMinor[iMinor]))
						* visc_contrib_x_coeff_on_vx_self;
					J.xx += contrib_to_Jxx;
					J.yy += contrib_to_Jxx;
					J.zz += contrib_to_Jxx;

				}; // bUsableSide

				cc0 = cc1;
				prevpos = opppos;
				opppos = nextpos;

			}; // next i

			J.Inverse(result);

		}
		else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
			memset(&result, 0, sizeof(f64_tens3));
			result.xx = 1.0;
			result.yy = 1.0;
			result.zz = 1.0;
		}
	} // scope
	memcpy(&(p_matrix_n[iMinor]), &result, sizeof(f64_tens3));

}

__global__ void kernelCreate_v4 (
	v4 * __restrict__ p_vie_,
	v4 * __restrict__ p_vie_save_,
	f64 const lambda_,
	f64_vec2 * __restrict__ p_regrlc2_,
	f64 * __restrict__ p_regrlc_iz_,
	f64 * __restrict__ p_regrlc_ez_)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	v4 vie = p_vie_save_[iMinor];
	f64_vec2 lc2 = p_regrlc2_[iMinor];
	f64 lciz = p_regrlc_iz_[iMinor];
	f64 lcez = p_regrlc_ez_[iMinor];	

	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) printf("iMinor %d: vez %1.12E lambda %1.9E lcez %1.9E\n",
		iMinor, vie.vez, lambda_, lcez);

	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) printf("iMinor %d: vy %1.13E lambda %1.9E lcy %1.9E\n",
		iMinor, vie.vxy.y, lambda_, lc2.y);
	vie.vxy += lambda_*lc2;
	vie.viz += lambda_*lciz;
	vie.vez += lambda_*lcez;

	if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) printf("iMinor %d: vy %1.13E \n",
		iMinor, vie.vxy.y);
	
	p_vie_[iMinor] = vie;
}

__global__ void CreateLC4(
	f64_vec2 * __restrict__ p_regrlc2_,
	f64 * __restrict__ p_regrlc_iz_,
	f64 * __restrict__ p_regrlc_ez_,
	f64_vec2 * __restrict__ p_regr2,
	f64 * __restrict__ p_regr_iz,
	f64 * __restrict__ p_regr_ez)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	f64_vec2 xy, lcxy(0.0,0.0);
	f64 iz, ez;
	f64 lciz = 0.0, lcez =0.0;
	
	for (int i = 0; i < REGRESSORS; i++)
	{
		xy = p_regr2[NMINOR*i + iMinor];
iz = p_regr_iz[NMINOR*i + iMinor];
ez = p_regr_ez[NMINOR*i + iMinor];
lcxy += beta_n_c[i] * xy;
lciz += beta_n_c[i] * iz;
lcez += beta_n_c[i] * ez;

//		if (iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) printf("iMinor %d: ez %1.8E beta[i] %1.8E lcez %1.8E\n",
//			iMinor, ez, beta_n_c[i], lcez);
	};
	p_regrlc2_[iMinor] = lcxy;
	p_regrlc_iz_[iMinor] = lciz;
	p_regrlc_ez_[iMinor] = lcez;
}


__global__ void kernelZeroSelected(
	f64_vec3 * __restrict__ p_MAR_i,
	f64_vec3 * __restrict__ p_MAR_e,
	f64_vec3 * __restrict__ p_MAR_n,
	NTrates * __restrict__ p_NTrates_vert,
	NTrates * __restrict__ p_NTrates_tri,
	int * __restrict__ p_Select,
	int * __restrict__ p_SelectNeut
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	if (p_Select[index] != 0) {
		memset(&(p_MAR_i[index]), 0, sizeof(f64_vec3));
		memset(&(p_MAR_e[index]), 0, sizeof(f64_vec3));
		if (index >= BEGINNING_OF_CENTRAL) {
			p_NTrates_vert[index - BEGINNING_OF_CENTRAL].NiTi = 0.0;
			p_NTrates_vert[index - BEGINNING_OF_CENTRAL].NeTe = 0.0;
		}
		else {
			p_NTrates_tri[index].NiTi = 0.0;
			p_NTrates_tri[index].NeTe = 0.0;
		};
	};
	if (p_SelectNeut[index] != 0) {
		memset(&(p_MAR_n[index]), 0, sizeof(f64_vec3));
		if (index >= BEGINNING_OF_CENTRAL) {
			p_NTrates_vert[index - BEGINNING_OF_CENTRAL].NnTn = 0.0;
		}
		else {
#ifdef COLLECT_VISC_HTG_IN_TRIANGLES

			p_NTrates_tri[index].NnTn = 0.0;
#else
			p_NTrates_tri[index * 3].NnTn = 0.0;
			p_NTrates_tri[index * 3 + 1].NnTn = 0.0;
			p_NTrates_tri[index * 3 + 2].NnTn = 0.0;
#endif
		};
	};
}


__global__ void kernelFindNegative_dTi
(
	NTrates * NTrate,
	NTrates * NTtri,
	long * pl)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	NTrates test;
	if (index >= BEGINNING_OF_CENTRAL) {
		memcpy(&test, &(NTrate[index - BEGINNING_OF_CENTRAL]), sizeof(NTrates));
	}
	else {
		memcpy(&test, &(NTtri[index]), sizeof(NTrates));
	};

	if (test.NiTi < 0.0) {
		printf("index %d (%d) NiTi %1.8E\n", index, index - BEGINNING_OF_CENTRAL,
			test.NiTi);
		pl[blockIdx.x]++;
	};

}


__global__ void kernelSubtractNiTiCheckNeg
(NTrates * __restrict__ NTrates_init, NTrates * __restrict__ NTrates_final, f64 * __restrict__ p_diff, nvals * __restrict__ p_n, f64 * __restrict__ p_AreaMajor)
{
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	f64 diff = NTrates_final[index].NiTi - NTrates_init[index].NiTi;
	nvals nnn = p_n[index];
	f64 Area = p_AreaMajor[index];
	if (diff < 0.0) printf("%d : NiTi diff %1.12E Tchg %1.10E\n", index, diff, diff / (nnn.n*Area));
	p_diff[index] = diff;
}


__global__ void kernelCreateIta_over_nM(
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,
	f64 * __restrict__ p_kappa_n,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_Area,
	f64 * __restrict__ result_i,
	f64 * __restrict__ result_e,
	f64 * __restrict__ result_n,
	bool const DivideBy_m_s
) {
	long const index = blockIdx.x*blockDim.x + threadIdx.x;
	nvals NN = p_n_major[index];
	f64 K_i = p_kappa_i[index];
	f64 K_e = p_kappa_e[index];
	f64 K_n = p_kappa_n[index];
	f64 Area = p_Area[index];
	if (DivideBy_m_s) {
		if (Area*NN.n == 0.0) {
			result_i[index] = -1000.0;
			result_e[index] = -1000.0;
		} else {
			result_i[index] = K_i / (m_ion*Area*NN.n);
			result_e[index] = K_e / (m_e*Area*NN.n);
		};
		if (Area*NN.n_n == 0.0) {
			result_n[index] = -1000.0;
		} else {
			result_n[index] = K_n / (m_n*Area*NN.n_n);
		};
	} else {
		if (Area*NN.n == 0.0) {
			result_i[index] = -1000.0;
			result_e[index] = -1000.0;
		}
		else {
			result_i[index] = K_i / (Area*NN.n);
			result_e[index] = K_e / (Area*NN.n);
		};
		if (Area*NN.n_n == 0.0) {
			result_n[index] = -1000.0;
		}
		else {
			result_n[index] = K_n / (Area*NN.n_n);
		};
	}
}