
// Device routines that can be #included by the kernels file.
#include "cuda_struct.h"
#include "kernel.h"
 

#ifdef __CUDACC__
__device__ __forceinline__ f64 GetEzShape(f64 r) {
	return 1.0 - 1.0 / (1.0 + exp(-24.0*(r - 4.32)));
	// return 1.0 - 1.0 / (1.0 + exp(-16.0*(r - 4.2))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}
#else
f64 inline GetEzShape_(f64 r) {
	return 1.0 - 1.0 / (1.0 + exp(-16.0*(r - 4.2))); // At 4.0cm it is 96% as strong as at tooth. 4.2 50%. At 4.4 it is 4%.
}
#endif

__device__ __forceinline__ f64 Get_lnLambda_ion_d(f64 n_ion,f64 T_ion)
{
	// Assume static f64 const is no good in kernel.

	f64 factor, lnLambda_sq;
	f64 Tion_eV3 = T_ion*T_ion*T_ion*one_over_kB_cubed;
	f64 lnLambda = 23.0 - 0.5*log(n_ion/Tion_eV3); 

	// floor at 2.0:
	lnLambda_sq = lnLambda*lnLambda;
	factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
	lnLambda += 2.0/factor;

	return lnLambda;
}		

__device__ __forceinline__ f64 Get_lnLambda_d(real n_e,real T_e)
{
	real lnLambda, factor, lnLambda_sq, lnLambda1, lnLambda2;

	real Te_eV = T_e*one_over_kB;
	real Te_eV2 = Te_eV*Te_eV;
	real Te_eV3 = Te_eV*Te_eV2;

	if (n_e*Te_eV3 > 0.0) {
		
		lnLambda1 = 23.0 - 0.5*log(n_e/Te_eV3);
		lnLambda2 = 24.0 - 0.5*log(n_e/Te_eV2);
		// smooth between the two:
		factor = 2.0*fabs(Te_eV-10.0)*(Te_eV-10.0)/(1.0+4.0*(Te_eV-10.0)*(Te_eV-10.0));
		lnLambda = lnLambda1*(0.5-factor)+lnLambda2*(0.5+factor);
		
		// floor at 2 just in case, but it should not get near:
		lnLambda_sq = lnLambda*lnLambda;
		factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
		lnLambda += 2.0/factor;

		// Golant p.40 warns that it becomes invalid when an electron gyroradius is less than a Debye radius.
		// That is something to worry about if  B/400 > n^1/2 , so looks not a big concern.

		// There is also a quantum ceiling. It will not be anywhere near. At n=1e20, 0.5eV, the ceiling is only down to 29; it requires cold dense conditions to apply.

		if (lnLambda < 2.0) lnLambda = 2.0; // deal with negative inputs

	} else {
		lnLambda = 20.0;
	};
	return lnLambda;
}		


__device__ __forceinline__ f64_vec2 Anticlock_rotate2(const f64_vec2 arg)
{
	f64_vec2 result;
	result.x = Anticlockwise_d.xx*arg.x+Anticlockwise_d.xy*arg.y;
	result.y = Anticlockwise_d.yx*arg.x+Anticlockwise_d.yy*arg.y;
	return result;
}
__device__ __forceinline__ f64_vec2 Clockwise_rotate2(const f64_vec2 arg)
{
	f64_vec2 result;
	result.x = Clockwise_d.xx*arg.x+Clockwise_d.xy*arg.y;
	result.y = Clockwise_d.yx*arg.x+Clockwise_d.yy*arg.y;
	return result;
}

__device__ __forceinline__ f64_vec3 Anticlock_rotate3(const f64_vec3 arg)
{
	f64_vec3 result;
	result.x = Anticlockwise_d.xx*arg.x+Anticlockwise_d.xy*arg.y;
	result.y = Anticlockwise_d.yx*arg.x+Anticlockwise_d.yy*arg.y;
	result.z = arg.z;
	return result;
}
__device__ __forceinline__ f64_vec3 Clockwise_rotate3(const f64_vec3 arg)
{
	f64_vec3 result;
	result.x = Clockwise_d.xx*arg.x+Clockwise_d.xy*arg.y;
	result.y = Clockwise_d.yx*arg.x+Clockwise_d.yy*arg.y;
	result.z = arg.z;
	return result;
}

__device__  __forceinline__ void Estimate_Ion_Neutral_Cross_sections_d(real T, // call with T in electronVolts
	real * p_sigma_in_MT,
	real * p_sigma_in_visc)
{
	if (T > cross_T_vals_d[9]) {
		*p_sigma_in_MT = cross_s_vals_MT_ni_d[9];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni_d[9];
		return;
	}
	if (T < cross_T_vals_d[0]) {
		*p_sigma_in_MT = cross_s_vals_MT_ni_d[0];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni_d[0];
		return;
	}
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;

	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			}
			else {
				i = 8;
			};
		}
		else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			}
			else {
				i = 6;
			};
		};
	}
	else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			}
			else {
				i = 4;
			};
		}
		else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			}
			else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				}
				else {
					i = 1;
				};
			};
		};
	};
	// T lies between i-1,i
	real ppn = (T - cross_T_vals_d[i - 1]) / (cross_T_vals_d[i] - cross_T_vals_d[i - 1]);

	*p_sigma_in_MT = ppn * cross_s_vals_MT_ni_d[i] + (1.0 - ppn)*cross_s_vals_MT_ni_d[i - 1];
	*p_sigma_in_visc = ppn * cross_s_vals_viscosity_ni_d[i] + (1.0 - ppn)*cross_s_vals_viscosity_ni_d[i - 1];
	return;
}

__device__ __forceinline__ f64 Estimate_Neutral_MT_Cross_section_d(f64 T)
{
	// CALL WITH T IN eV

	if (T > cross_T_vals_d[9]) return cross_s_vals_MT_ni_d[9];		
	if (T < cross_T_vals_d[0]) return cross_s_vals_MT_ni_d[0];
	
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 
	
	// T lies between i-1,i
	real ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_MT_ni_d[i] + (1.0-ppn)*cross_s_vals_MT_ni_d[i-1];

}

__device__ __forceinline__ f64 Estimate_Neutral_Neutral_Viscosity_Cross_section_d(f64 T) 
{
	// call with T in electronVolts
	
	if (T > cross_T_vals_d[9]) return cross_s_vals_viscosity_nn_d[9];
	if (T < cross_T_vals_d[0]) return cross_s_vals_viscosity_nn_d[0];

	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 

	// T lies between i-1,i
	real ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_viscosity_nn_d[i] + (1.0-ppn)*cross_s_vals_viscosity_nn_d[i-1];
}

__device__ __forceinline__ f64 Estimate_Ion_Neutral_Viscosity_Cross_section(f64 T)
{
	if (T > cross_T_vals_d[9]) return cross_s_vals_viscosity_ni_d[9];		
	if (T < cross_T_vals_d[0]) return cross_s_vals_viscosity_ni_d[0];
	
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 
	
	// T lies between i-1,i
	real ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_viscosity_ni_d[i] + (1.0-ppn)*cross_s_vals_viscosity_ni_d[i-1];
}


__device__ __forceinline__ f64 Calculate_Kappa_Neutral(f64 n_i, f64 T_i, f64 n_n, f64 T_n)
{
	// NOTE:
	// It involves sqrt and we could easily find a way to calculate only once.
		
	if (n_n == 0.0) return 0.0;

	f64 s_in_visc, s_nn_visc;

	s_in_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_i*one_over_kB);
	s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(T_n*one_over_kB);

	// Oh. So there's another two we have to port.
	// Yet for ion eta it's so different, apparently.
	
	f64 ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
	f64	nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
	f64	nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);
	f64	nu_nheart = 0.75*nu_ni_visc + 0.25*nu_nn_visc;
	f64 kappa_n = NEUTRAL_KAPPA_FACTOR*n_n*T_n/(m_n*nu_nheart);
	// NEUTRAL_KAPPA_FACTOR should be in constant.h
	// e-n does not feature.
	return kappa_n;
}


__device__ __forceinline__ void Get_kappa_parallels_and_nu_hearts
				(real n_n,real T_n,real n_i,real T_i,real n_e,real T_e,
				f64 * pkappa_neut, f64 * pnu_nheart, 
				f64 * pkappa_ion_par, f64 * pnu_iheart,
				f64 * pkappa_e_par, f64 * pnu_eheart,
				f64 * pratio)
{
	f64 s_in_visc, s_nn_visc, s_en_visc;

	f64 ionneut_thermal, 
		nu_ni_visc, nu_nn_visc, nu_nheart,
		nu_in_visc, nu_en_visc, nu_ii, nu_iheart, nu_eheart,
		sqrt_Te, electron_thermal, nu_eiBar;
	 
	f64 lnLambda = Get_lnLambda_ion_d(n_i,T_i);

	ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
	sqrt_Te = sqrt(T_e);
	
	s_in_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_i*one_over_kB);
	s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(T_n*one_over_kB);
	
	nu_in_visc = n_n*s_in_visc*ionneut_thermal;
	nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);
	nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
	
	nu_ii = Nu_ii_Factor*kB_to_3halves*n_i*lnLambda/(T_i*sqrt(T_i));

	nu_iheart = 0.75*nu_in_visc
			+ 0.8*nu_ii-0.25*nu_in_visc*nu_ni_visc/(3.0*nu_ni_visc+nu_nn_visc);
	*pkappa_ion_par = 2.5*n_i*T_i/(m_ion*(nu_iheart));
	*pnu_iheart = nu_iheart;

	s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_e*one_over_kB);
	electron_thermal = (sqrt_Te*over_sqrt_m_e);
	
	lnLambda = Get_lnLambda_d(n_e,T_e);
	
	nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
	nu_en_visc = n_n*s_en_visc*electron_thermal;
	nu_eheart = 1.87*nu_eiBar + nu_en_visc;
	*pnu_eheart = nu_eheart;
	*pkappa_e_par =  2.5*n_e*T_e/(m_e*nu_eheart);

	// Store ratio for thermoelectric use:
	*pratio = nu_eiBar/nu_eheart;


	if (n_n == 0.0){
		*pkappa_neut = 0.0;
	} else {

		nu_nheart = 0.75*nu_ni_visc + 0.25*nu_nn_visc;
		*pkappa_neut = NEUTRAL_KAPPA_FACTOR*n_n*T_n/(m_n*nu_nheart);
		*pnu_nheart = nu_nheart;
		// NEUTRAL_KAPPA_FACTOR should be in constant.h
		// e-n does not feature.
	};
	 
}
__device__ __forceinline__ void RotateClockwise(f64_vec3 & v)
{
	f64 temp = Clockwise_d.xx*v.x + Clockwise_d.xy*v.y;
	v.y = Clockwise_d.yx*v.x + Clockwise_d.yy*v.y;
	v.x = temp;
}
__device__ __forceinline__ void RotateAnticlockwise(f64_vec3 & v)
{
	f64 temp = Anticlockwise_d.xx*v.x + Anticlockwise_d.xy*v.y;
	v.y = Anticlockwise_d.yx*v.x + Anticlockwise_d.yy*v.y;
	v.x = temp;
}

__device__ __forceinline__ f64_vec2 GetRadiusIntercept(f64_vec2 x1,f64_vec2 x2,f64 r)
{
	// where we meet radius r on the line passing through u0 and u1?
	f64_vec2 result;
	
	f64 den = (x2.x-x1.x)*(x2.x-x1.x) + (x2.y - x1.y)*(x2.y - x1.y) ;
	f64 a = (x1.x * (x2.x-x1.x) + x1.y * (x2.y-x1.y) ) / den;
	// (t + a)^2 - a^2 = (  c^2 - x1.x^2 - x1.y^2  )/den
	f64 root = sqrt( (r*r- x1.x*x1.x - x1.y*x1.y)/den + a*a ) ;
	f64 t1 = root - a;
	f64 t2 = -root - a;
	
	// since this is a sufficient condition to satisfy the circle, this probably means that
	// the other solution is on the other side of the circle.
	// Which root is within x1, x2 ? Remember x2 would be t = 1.

	if (t1 > 1.0) 
	{
		if ((t2 < 0.0) || (t2 > 1.0))
		{	
			// This usually means one of the points actually is on the curve.
			f64 dist1 = min(fabs(t1-1.0),fabs(t1));
			f64 dist2 = min(fabs(t2-1.0),fabs(t2));
			if (dist1 < dist2)
			{
				// use t1				
				result.x = x1.x + t1*(x2.x-x1.x);
				result.y = x1.y + t1*(x2.y-x1.y);
		//		printf("t1@@");
			} else {
				// use t2				
				result.x = x1.x + t2*(x2.x-x1.x);
				result.y = x1.y + t2*(x2.y-x1.y);
		//		printf("t2@@");
			};
		} else {		
			// use t2:		
			result.x = x1.x + t2*(x2.x-x1.x);
			result.y = x1.y + t2*(x2.y-x1.y);
		//	printf("t2~");
		};
	} else {
		result.x = x1.x + t1*(x2.x-x1.x);
		result.y = x1.y + t1*(x2.y-x1.y);	
		//printf("t1~");
	};

	// For some reason this is only hitting the radius to single precision.

	// printf to compare difference between achieved radius and r.
	
	//if ((result.x < -0.145) && (result.x > -0.155))
	//{
	//	f64 achieve = result.modulus();
	//	printf("ach %1.12E r %1.2f t1 %1.10E \nx %1.12E y %1.12E\n",achieve,r,t1,result.x,result.y);
	//}

	// So what do we do?

	// We could boost back but there seem to be bigger problems thereafter.

	// Ideally we'd go through and compare and see, is it t1 that is a bit wrong here?
	// 

	return result;
}

