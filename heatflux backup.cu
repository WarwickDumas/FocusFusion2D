
#define TESTHEAT (0)
#define TESTXYDERIVZVISCVERT (0)
#define TEST_EPSILON_Y (iVertex == VERTCHOSEN)

//
//
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
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
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
//
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
		f64 smallstep = 2.0e-10 + 2.0e-10*vlocal;

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
	printf("root %1.10E \n", root);
	if (root == 0.0) {
		// 0 may be wrong? 
		// Be careful what rescaling is doing in the case both operands are very small.

		// Try empirical estimate in this case.

		f64 smallstep = 1.0e-7;
		bool bCont = false;
		iIterate = 0;
		do {
			iIterate++;
			smallstep *= 0.1;
			bCont = false;
			root = smallstep*sqrt(d_by_dbeta_prev*d_by_dbeta_prev + d_by_dbeta_next*d_by_dbeta_next);
			if (root == 0.0) {
				d_by_dbeta_transversederiv = 0.0;
			}
			else {
				f64 overroot = 1.0 / root;
				f64 transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev*smallstep)
						+ asinh(20.0*overroot*d_by_dbeta_next*smallstep)));
				f64 d_by_dbeta_transversederiv2 = transversederiv_stepped / smallstep;
				
				smallstep *= 0.5;
				root *= 0.5;
				overroot *= 2.0;
				transversederiv_stepped =
					root*0.05*sinh(0.5*(asinh(20.0*overroot*d_by_dbeta_prev)
						+ asinh(20.0*overroot*d_by_dbeta_next)));
				f64 d_by_dbeta_transversederiv1 = transversederiv_stepped / smallstep;

				printf("empirical, xverse12 %1.9E %1.9E \n",
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
		} while ((bCont) && (iIterate < 20));
	}
	else {
		f64 overroot = 1.0 / root;
		f64 transversederiv = root*0.05*sinh(0.5*(asinh(20.0*overroot*transversederivprev)
			+ asinh(20.0*overroot*transversederivnext)));

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

		printf("empirical d/db xverse %1.13E \n", d_by_dbeta_transversederiv);


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
	printf("tranverse    component %1.10E %1.10E\n", d_by_dbeta_transversederiv*vec_out.x, d_by_dbeta_transversederiv*vec_out.y);
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

						if (0) printf("%d %d kappa_par %1.10E edgelen %1.10E delta %1.10E T %1.10E \n"
							"T_out %1.14E contrib %1.14E flux coefficient on T_out %1.14E\n",
							iVertex, indexneigh, kappa_par, edgelen, (pos_out - info.pos).modulus(), shared_T[threadIdx.x], T_out,
							TWOTHIRDS * kappa_par * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus(),
							TWOTHIRDS * kappa_par * edgelen / (pos_out - info.pos).modulus()	);
						
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
							if (TESTHEAT)
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

		if (result != result) printf("iVertex %d NaN result. d/dbeta %1.10E N %1.8E our_x %1.8E \n",
			iVertex, d_by_dbeta, N, our_x);


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

		if (TESTHEAT) printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);

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
						if (TESTHEAT) {
							printf("iVertex %d indexneigh %d temp %1.14E our_fac %1.14E iNeigh %d temp*our_fac %1.14E \n",
								iVertex, indexneigh, temp, our_fac, iNeigh, temp*our_fac);
						}
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

		if (TESTHEAT) printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);

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
	f64 const over_m_s) // easy way to put it in constant memory
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
			v4 temp;
			memcpy(&temp, &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_v_verts[threadIdx.x].x = temp.vxy.x;
			shared_v_verts[threadIdx.x].y = temp.vxy.y;
			shared_v_verts[threadIdx.x].z = (iSpecies == 1) ? temp.viz : temp.vez;
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
				}
				else {
					v4 temp = p_vie_minor[izTri[inext]];
					next_v.x = temp.vxy.x; next_v.y = temp.vxy.y;
					if (iSpecies == 1) { next_v.z = temp.viz; }
					else { next_v.z = temp.vez; };
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

					if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = ita_theirs;
						nu = nu_theirs;
					};

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				if (ita_par > 0.0)
				{
					
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

				if (TEST_EPSILON_Y) {
					printf("%d %d %d gradvx %1.9E %1.9E gradvy %1.9E %1.9E gradvz %1.9E %1.9E \n"
						"prevpos %1.10E %1.10E info.pos %1.10E %1.10E nextpos %1.10E %1.10E opppos %1.10E %1.10E\n"
						"prev_vx %1.12E our_vx %1.12E nextvx %1.12E oppvx %1.12E \n"
						"prev_vy %1.12E our_vy %1.12E nextvy %1.12E oppvy %1.12E \n",
						iVertex, i, izTri[i], gradvx.x, gradvx.y, gradvy.x, gradvy.y, gradvz.x, gradvz.y,
						prevpos.x, prevpos.y, info.pos.x, info.pos.y, nextpos.x, nextpos.y, opppos.x, opppos.y,
						prev_v.x, shared_v_verts[threadIdx.x].x, next_v.x, opp_v.x,
						prev_v.y, shared_v_verts[threadIdx.x].y, next_v.y, opp_v.y);
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

						visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						ownrates_visc += visc_contrib;
						visc_htg += -THIRD*(m_s)*(htg_diff.dot(visc_contrib));

						if (TEST_EPSILON_Y) {
							printf("unmag %d : %d : ita %1.8E gradvx %1.8E %1.8E Pi_xx %1.8E Pi_xy %1.8E contrib.x %1.8E edgenml %1.8E %1.8E\n"
								"------------\n",
								iVertex, izTri[i], ita_par, gradvx.x, gradvx.y, Pi_xx, Pi_xy, visc_contrib.x, edge_normal.x, edge_normal.y);

						}

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

									if ((TEST_EPSILON_Y) && (iSpecies == 1))
										printf("dvb/db %1.9E dvperp/db %1.9E dvH/db %1.9E \n",
											dvb_by_db, dvperp_by_db, dvHall_by_db);

									if (TESTIONVERTVISC)
										printf("dvperp_by_db %1.8E W_bP %1.8E \n", dvperp_by_db, W_bP);
									if (TESTIONVERTVISC)
										printf("dvHall_by_db %1.8E W_bH %1.8E \n", dvHall_by_db, W_bH);
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

									if ((TEST_EPSILON_Y) && (iSpecies == 1))
										printf("dvb/dP %1.9E dvperp/dP %1.9E dvH/dP %1.9E \n",
											dvb_by_dperp, dvperp_by_dperp, dvHall_by_dperp);

									if (TESTIONVERTVISC)
										printf("dvb_by_dperp %1.8E W_bP %1.8E \n",
											dvb_by_dperp, W_bP);
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

									if ((TEST_EPSILON_Y) && (iSpecies == 1))
										printf("dvb/dH %1.9E dvperp/dH %1.9E dvH/dH %1.9E \n",
											dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall);

									W_bb -= 2.0*THIRD*dvHall_by_dHall;
									W_PP -= 2.0*THIRD*dvHall_by_dHall;
									W_HH += 4.0*THIRD*dvHall_by_dHall;
									W_bH += dvb_by_dHall;
									W_PH += dvperp_by_dHall;
									if (TESTIONVERTVISC)
										printf("dvb_by_dHall %1.8E W_bH %1.8E \n",
											dvb_by_dHall, W_bH);
								}
							}							
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
								// W_HH = 0
								if ((TEST_EPSILON_Y) && (iSpecies == 1)) 
									printf("ita_1 %1.9E par %1.8E W_bb %1.9E W_PP %1.9E W_HH %1.9E Pi_bb %1.9E Pi_PP %1.9E\n",
										ita_1, ita_par, W_bb, W_PP, W_HH, Pi_b_b, Pi_P_P);
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;

								if (TESTIONVERTVISC)
									printf(" -ita_2 %1.8E W_bP %1.8E contrib %1.8E Pi_P_b %1.8E \n",
										-ita_2, W_bP, -ita_2*W_bP, Pi_P_b);

								if ((TEST_EPSILON_Y) && (iSpecies == 1))
									printf("ita_2 %1.9E W_bP %1.9E Pi_Pb %1.9E \n",
										ita_2, W_bP, Pi_P_b);
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
								if ((TEST_EPSILON_Y) && (iSpecies == 1))
									printf("ita_3 %1.9E W_PH %1.9E Pi_PP %1.9E \n",
										ita_3, W_PH, Pi_P_P);
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;

								if (TESTIONVERTVISC)
									printf(" -ita_4 %1.8E W_bH %1.8E contrib %1.8E Pi_P_b %1.8E nu %1.8E omega %1.8E \n",
										-ita_4, W_bH, -ita_4*W_bH, Pi_P_b, nu, omegamod);

								if ((TEST_EPSILON_Y) && (iSpecies == 1))
									printf("ita_4 %1.9E W_bH %1.9E Pi_Pb %1.9E \n",
										ita_4, W_bH, Pi_P_b);
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

							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;
							if ((TEST_EPSILON_Y) && (iSpecies == 1)) {
								printf("mag_edge %1.10E %1.10E %1.10E \n", mag_edge.x, mag_edge.y, mag_edge.z);
							};

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
						

						if (TESTXYDERIVZVISCVERT) {
							printf("%d visc_ctbz %1.9E over_m_s %1.8E mf b P H %1.9E %1.9E %1.9E b.z P.z H.z %1.8E %1.8E %1.8E \n",
								iVertex, visc_contrib.z, over_m_s, momflux_b, momflux_perp, momflux_Hall, unit_b.z, unit_perp.z, unit_Hall.z);
							printf("Pi bb Pb Hb PP HP HH %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E \n",
								Pi_b_b, Pi_P_b, Pi_H_b, Pi_P_P, Pi_H_P, Pi_H_H);
						}

						if ((TEST_EPSILON_Y) && (iSpecies == 1)) {
							printf("iVertex %d tri %d species %d ita_par %1.9E gradvx %1.8E %1.8E\n"
								"omega %1.9E %1.9E %1.9E nu %1.11E ourpos %1.8E %1.8E \n"
								"unit_b %1.11E %1.11E %1.11E unit_perp %1.11E %1.11E %1.11E unit_Hall %1.8E %1.8E %1.11E\n",
								iVertex, izTri[i], iSpecies, ita_par, gradvx.x, gradvx.y,
								omega_c.x, omega_c.y, omega_c.z, nu, info.pos.x, info.pos.y,
								unit_b.x, unit_b.y, unit_b.z, unit_perp.x, unit_perp.y, unit_perp.z, unit_Hall.x, unit_Hall.y, unit_Hall.z);
							printf(
								"Pi_b_b %1.11E Pi_P_b %1.11E Pi_P_P %1.11E Pi_H_b %1.11E Pi_H_P %1.11E Pi_H_H %1.11E\n"
								"momflux b %1.11E perp %1.11E Hall %1.11E visc_contrib %1.9E %1.9E %1.9E \n",
								Pi_b_b, Pi_P_b, Pi_P_P, Pi_H_b, Pi_H_P, Pi_H_H,
								momflux_b, momflux_perp, momflux_Hall,
								visc_contrib.x, visc_contrib.y, visc_contrib.z);
							
						}
						
					}
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
		} else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memset(&ownrates_visc, 0, sizeof(f64_vec3));
	visc_htg = 0.0;
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
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					v4 temp = p_vie_minor[izNeighMinor[iprev]];
					prev_v.x = temp.vxy.x; prev_v.y = temp.vxy.y; prev_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
					opp_v.x = temp.vxy.x; opp_v.y = temp.vxy.y; opp_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
						next_v.x = temp.vxy.x; next_v.y = temp.vxy.y; next_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						// USEFUL:
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
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
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							} else {
								ita_par = ita_par_opp;
								nu = nu_theirs;
							}

							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// Get rid of ins-ins triangle traffic:

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}

				f64_vec2 gradvx, gradvy, gradvz;
				f64_vec3 htg_diff;

				if (bUsableSide)
				{
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
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

					if (iMinor == CHOSEN) printf("%d %d %d vz prev own anti opp %1.10E %1.10E %1.10E %1.10E "
						"gradvz %1.10E %1.10E\n", iMinor, i, izNeighMinor[i],
						prev_v.z, shared_v[threadIdx.x].z, next_v.z, opp_v.z, gradvz.x, gradvz.y);

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

					htg_diff = shared_v[threadIdx.x] - opp_v;
					
				} else {
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

						f64_vec3 visc_contrib;
						visc_contrib.x = -over_m_s*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						visc_contrib.y = -over_m_s*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						visc_contrib.z = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						if (iMinor == CHOSEN) printf("%d %d %d unmag visc_contrib.z %1.10E Pi_zx %1.10E Pi_zy %1.10E edgenml %1.8E %1.8E\n",
							iMinor, i, izNeighMinor[i], visc_contrib.z, Pi_zx, Pi_zy, edge_normal.x, edge_normal.y);

						ownrates_visc += visc_contrib;

						if (i % 2 == 0) {
							// vertex : heat collected by vertex
						} else {
							visc_htg += -THIRD*(m_s)*(htg_diff.dot(visc_contrib));
						}

						// So we are saying if edge_normal.x > 0 and gradviz.x > 0
						// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					} else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

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

							{
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
							}
						}
						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;

							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

							// Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);

							if (iMinor == CHOSEN) printf("%d %d %d mag_edge %1.8E %1.8E %1.8E Pi_b_b %1.10E Pi_H_P %1.8E Pi_H_b %1.8E \n",
								iMinor, i, izNeighMinor[i], mag_edge.x, mag_edge.y, mag_edge.z, Pi_b_b, Pi_H_P, Pi_H_b);

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

						if (iMinor == CHOSEN) printf("%d %d %d mag visc_ctb.z %1.10E b.z %1.10E momf_b %1.10E "
							"perp.z %1.10E momf_perp %1.10E unit_H.z %1.8E mf_H %1.9E\n",
							iMinor, i, izNeighMinor[i], visc_contrib.z, unit_b.z, momflux_b, unit_perp.z, momflux_perp, unit_Hall.z, momflux_Hall);


						ownrates_visc += visc_contrib;
						if (i % 2 != 0) // not vertex
						{
							visc_htg += -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib));
							if (TESTIONVISC)
								printf("%d %d visc_htg %1.10E\n", iMinor, i, -THIRD*((iSpecies == 1) ? m_ion : m_e)*(htg_diff.dot(visc_contrib)));
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

			if (iSpecies == 1) {
				p_NT_addition_tri[iMinor].NiTi += visc_htg;
			} else {
				p_NT_addition_tri[iMinor].NeTe += visc_htg;
			}

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
		shared_v[threadIdx.x].z = (iSpecies == 1) ? temp.viz : temp.vez;
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
				shared_v_verts[threadIdx.x].z = (iSpecies == 1) ? temp.viz : temp.vez;
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
					prev_v.z = (iSpecies == 1) ? temp.viz : temp.vez;

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
					opp_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
				} else {
					opp_v = p_v_n[izTri[i]];
				}
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
						next_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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

					if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = ita_theirs;
						nu = nu_theirs;
					};

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				if (ita_par > 0.0)
				{

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
					}
				}

				if (TESTXYDERIVZVISCVERT) {
					printf("%d %d %d ROCgradvx %1.9E %1.9E ROCgradvy %1.9E %1.9E\n",
						iVertex, i, izTri[i], ROCgradvx.x, ROCgradvx.y, ROCgradvy.x, ROCgradvy.y);
				};

				if (ita_par > 0.0) {

					if (iSpecies == 0) {
						f64_vec3 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
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

							if (TESTXYDERIVZVISCVERT) {
								printf("%d visc_ctbz %1.9E over_m_s %1.8E mf b P H %1.9E %1.9E %1.9E b.z P.z H.z %1.8E %1.8E %1.8E \n",
									iVertex, visc_contrib.z, over_m_s, momflux_b, momflux_perp, momflux_Hall, unit_b.z, unit_perp.z, unit_Hall.z);
								printf("Pi bb Pb Hb PP HP HH %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E \n",
									Pi_b_b, Pi_P_b, Pi_H_b, Pi_P_P, Pi_H_P, Pi_H_H);
							}

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
						prev_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
						opp_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
							next_v.z = (iSpecies == 1) ? temp.viz : temp.vez;
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
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						// USEFUL:
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					} else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
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
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_minor[izNeighMinor[i]];
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
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// Get rid of ins-ins triangle traffic:

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}

				f64_vec2 ROCgradvx, ROCgradvy;
				if (bUsableSide)
				{
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.

						ROCgradvx = (opp_x.x - shared_regr[threadIdx.x].x)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
						ROCgradvy = (opp_x.y - shared_regr[threadIdx.x].y)*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
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
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {
							// just set it to 0.
							bUsableSide = false;
							ROCgradvx.x = 0.0;
							ROCgradvx.y = 0.0;
							ROCgradvy.x = 0.0;
							ROCgradvy.y = 0.0;
						};
					};
				};

				if (bUsableSide) {
					if (iSpecies == 0) {
						f64_vec3 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
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

					if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = ita_theirs;
						nu = nu_theirs;
					};

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				if (ita_par > 0.0)
				{

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
				};

				if (ita_par > 0.0) {

					if (iSpecies == 0) {
						f64 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
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
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						// USEFUL:
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					} else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
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
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_minor[izNeighMinor[i]];
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
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// Get rid of ins-ins triangle traffic:

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}

				f64_vec2 ROCgradvz;
				if (bUsableSide)
				{
					if ((TestDomainPos(prevpos) == false) || (TestDomainPos(nextpos) == false))
					{
						// One of the sides is dipped under the insulator -- set transverse deriv to 0.
						// Bear in mind we are looking from a vertex into a tri, it can be ins tri.
						if (iMinor == CHOSEN) printf("longi. x %1.8E %1.8E\n", opp_x, shared_regr[threadIdx.x]);

						ROCgradvz = (opp_x - shared_regr[threadIdx.x])*(opppos - info.pos) /
							(opppos - info.pos).dot(opppos - info.pos);
					} else {

						if ((iMinor == CHOSEN)) {
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
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if (flag == CROSSING_INS) {
							// just set it to 0.
							bUsableSide = false;
							ROCgradvz.x = 0.0;
							ROCgradvz.y = 0.0;
						};
					};
				};


				if (iMinor == CHOSEN) printf("%d ROCgradvz %1.8E %1.8E bUsableSide %d\n",
					CHOSEN, ROCgradvz.x, ROCgradvz.y, bUsableSide ? 1 : 0);


				if (bUsableSide) {
					if (iSpecies == 0) {
						f64 visc_contrib;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
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

							//visc_contrib.x = 0.0;
							//visc_contrib.y = 0.0;
							visc_contrib = -over_m_s*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

							if (iMinor == CHOSEN) printf("CHOSEN unmag visc_contrib.z %1.8E \n", visc_contrib);

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

							if ((0) && (iMinor == CHOSEN)) printf("CHOSEN mag visc_contrib.z %1.8E \n", visc_contrib.z);

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

			if (iMinor == CHOSEN) printf("CHOSEN ROCMAR.z %1.8E \n", ownrates_visc.z);

			// UPDATE:
			memcpy(&(p_ROCMAR[iMinor]), &(ownrates_visc), sizeof(f64_vec3));

		} else {
			memset(&(p_ROCMAR[iMinor]), 0, sizeof(f64_vec3));
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope
}

__global__ void CalculateCoeffself(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	Tensor2 * __restrict__ p__coeffself_xy, // matrix ... 
	f64 * __restrict__ p__coeffself_z,

	// Note that we then need to add matrices and invert somehow, for xy.
	// For z we just take 1/x .

	int const iSpecies,
	f64 const m_s,
	f64 const over_m_s

	// we want to know coeffself 


) // easy way to put it in constant memory
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

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

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
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
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&deriv_xy, 0, sizeof(f64_tens2));
		deriv_z = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX) && (shared_ita_par_verts[threadIdx.x] > 0.0))
		{
			// We are losing energy if there is viscosity into OUTERMOST.

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_vec2 opppos, prevpos, nextpos;
			// ideally we might want to leave position out of the loop so that we can avoid reloading it.

			short i = 0;
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


				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 

				// It seems that I think it's worth having the velocities as 3 x v4 objects limited scope even if we keep reloading from global
				// That seems counter-intuitive??
				// Oh and the positions too!

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

					if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = ita_theirs;
						nu = nu_theirs;
					};

					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.


				f64_vec2 coeffgradvx_by_vx, coeffgradvy_by_vy, coeffgradvz_by_vz; 
				// component j = rate of change of dvq/dj wrt change in vq_self

				if (ita_par > 0.0) {

					// We take the derivative considering only the longitudinal component.
					// The effect of self v on the transverse component should be small.
//					coeffgradvq_by_vq = (-1.0)*(opppos - info.pos) /
//						(opppos - info.pos).dot(opppos - info.pos);

					coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.x,our_v.x,next_v.x,opp_v.x);
					coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.y, our_v.y, next_v.y, opp_v.y);
					coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.z, our_v.z, next_v.z, opp_v.z);


					if (iVertex == VERTCHOSEN) printf("iVertex %d coeffgradvx_by_vx %1.10E %1.10E \n", iVertex, coeffgradvx_by_vx.x,
						coeffgradvx_by_vx.y);

					if (iSpecies == 0) {

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

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

							Dvisc_contrib = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							deriv_xy.yx += Dvisc_contrib.x;
							deriv_xy.yy += Dvisc_contrib.y;

							Dvisc_contrib = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);
							deriv_z += Dvisc_contrib.z;

						}; // whether unmagnetized
					}; // is it for neutrals
				}; // was ita_par == 0

				   // v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				prevpos = opppos;
				opppos = nextpos;
			}; // next i

			if (0) { //(iVertex % 8000 == 0) {
				printf("iVertex %d deriv_z %1.10E \n", iVertex, deriv_z);
			};
			if (iVertex == VERTCHOSEN) printf("\n%d coeff_self xx %1.9E xy %1.9E yx %1.9E yy %1.9E \n\n",
				iVertex, deriv_xy.xx, deriv_xy.xy, deriv_xy.yx, deriv_xy.yy);

			memcpy(&(p__coeffself_xy[iVertex + BEGINNING_OF_CENTRAL]), &deriv_xy, sizeof(f64_tens2));
			p__coeffself_z[iVertex + BEGINNING_OF_CENTRAL] = deriv_z;

		}
		else {
			// NOT domain vertex: Do nothing			
			memset(&(p__coeffself_xy[iVertex + BEGINNING_OF_CENTRAL]), 0, sizeof(f64_tens2));
			p__coeffself_z[iVertex + BEGINNING_OF_CENTRAL] = 0.0;

		}; // whether domain vertex

	}; // if do vertex at all

	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];
	memset(&deriv_xy, 0, sizeof(f64_tens2));
	deriv_z = 0.0;
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
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
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
			f64_vec3 omega_c;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (short i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

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

				bool bUsableSide = true;
				{
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						// USEFUL:
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
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
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_minor[izNeighMinor[i]];
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
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_c = 0.5*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
					if (iSpecies == 1) omega_c *= qoverMc;
					if (iSpecies == 2) omega_c *= qovermc;
				}

				// Get rid of ins-ins triangle traffic:

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if (flag == CROSSING_INS)
						bUsableSide = 0;
				}

				f64_vec2 coeffgradvx_by_vx, coeffgradvy_by_vy, coeffgradvz_by_vz;
				// component j = rate of change of dvq/dj wrt change in vq_self

				if (bUsableSide) {

					// We take the derivative considering only the longitudinal component.
					// The effect of self v on the transverse component should be small.
					//					coeffgradvq_by_vq = (-1.0)*(opppos - info.pos) /
					//						(opppos - info.pos).dot(opppos - info.pos);

					coeffgradvx_by_vx = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.x, our_v.x, next_v.x, opp_v.x);
					coeffgradvy_by_vy = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.y, our_v.y, next_v.y, opp_v.y);
					coeffgradvz_by_vz = GetSelfEffectOnGradient(prevpos, info.pos, nextpos, opppos,
						prev_v.z, our_v.z, next_v.z, opp_v.z);


					if (iVertex == VERTCHOSEN) printf("iVertex %d coeffgradvx_by_vx %1.10E %1.10E \n", iVertex, coeffgradvx_by_vx.x,
						coeffgradvx_by_vx.y);

					if (iSpecies == 0) {

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

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

								mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
								mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
								mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

								//momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
								//momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
								//momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
								Dmomflux_b = -(dPibb_by_dv*mag_edge.x + dPiPb_by_dv*mag_edge.y + dPiHb_by_dv*mag_edge.z);
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

							Dvisc_contrib = over_m_s*(unit_b.y*Dmomflux_b + unit_perp.y*Dmomflux_perp + unit_Hall.y*Dmomflux_Hall);
							deriv_xy.yx += Dvisc_contrib.x;
							deriv_xy.yy += Dvisc_contrib.y;

							Dvisc_contrib = over_m_s*(unit_b.z*Dmomflux_b + unit_perp.z*Dmomflux_perp + unit_Hall.z*Dmomflux_Hall);
							deriv_z += Dvisc_contrib.z;

						}; // whether unmagnetized
					}; // is it for neutrals
				}; // bUsableSide

				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				// Just leaving these but they won't do anything :
				prevpos = opppos;
				opppos = nextpos;
			}; // next i


			memcpy(&(p__coeffself_xy[iMinor]), &deriv_xy, sizeof(f64_tens2));
			p__coeffself_z[iMinor] = deriv_z;

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
			memset(&(p__coeffself_xy[iMinor]), 0, sizeof(f64_tens2));
			p__coeffself_z[iMinor] = 0.0;

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
		|| (info.flag == CROSSING_INS)) // ?
	{
		f64_vec2 regrxy = p_regr_xy[iMinor];
		f64 regriz = p_regr_iz[iMinor];
		f64 regrez = p_regr_ez[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion2[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec2[iMinor];
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;

		Depsilon.vxy = regrxy 
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		Depsilon.viz = regriz - hsub*(MAR_ion.z / N);
		Depsilon.vez = regrez - hsub*(MAR_elec.z / N);

		if (iMinor == CHOSEN)
			printf("kernelCCdepsbydbeta %d regr.y %1.8E Depsilon.vxy.y %1.8E MAR_ion.y %1.8E MAR_elec.y %1.8E\n", CHOSEN, regrxy.y, Depsilon.vxy.y,
				MAR_ion.y, MAR_elec.y);
		if (0) //iMinor == CHOSEN) 
			printf("kernelCCdepsbydbeta %d regrez %1.12E Depsilon.vez %1.12E MAR_elec.z %1.12E hsub %1.8E N %1.12E\n",
			CHOSEN, regrez, Depsilon.vez, MAR_elec.z, hsub, N);

	}
	else {
		// epsilon = 0
	};
	p_Depsilon_xy[iMinor] = Depsilon.vxy;
	p_Depsilon_iz[iMinor] = Depsilon.viz;
	p_Depsilon_ez[iMinor] = Depsilon.vez;
	if (0) //iMinor == CHOSEN) 
		printf("sent -- %d Depsilon.viz %1.8E \n", CHOSEN, Depsilon.viz);

}

// not beta
__global__ void kernelCreateDByDBetaCoeffmatrix(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	
	Tensor2 * __restrict__ p_matrix_xy_i,
	Tensor2 * __restrict__ p_matrix_xy_e,
	f64 * __restrict__ p_coeffself_iz,
	f64 * __restrict__ p_coeffself_ez,

	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_tens2 * __restrict__ p_invmatrix,
	f64 * __restrict__ p_invcoeffselfviz,
	f64 * __restrict__ p_invcoeffselfvez
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	
	Tensor2 invmatrix;
	memset(&invmatrix, 0, sizeof(Tensor2));
	f64 invcoeffselfviz = 0.0;
	f64 invcoeffselfvez = 0.0;

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;
		
		Tensor2 mat, matxy1, matxy2;
		memset(&mat, 0, sizeof(Tensor2));
		memcpy(&matxy1, &(p_matrix_xy_i[iMinor]), sizeof(Tensor2));
		memcpy(&matxy2, &(p_matrix_xy_e[iMinor]), sizeof(Tensor2));

		if (iMinor == CHOSEN) printf("\n%d mat1 xx %1.9E xy %1.9E yx %1.9E yy %1.9E \nmat2 %1.9E %1.9E %1.9E %1.9E\nhsub %1.11E N %1.11E\n",
			iMinor, matxy1.xx, matxy1.xy, matxy1.yx, matxy1.yy,
			matxy2.xx, matxy2.xy, matxy2.yx, matxy2.yy,
			hsub, N);

		// Think this through carefully.

		mat.xx = 1.0;
		mat.yy = 1.0;
		mat.xx += -hsub*((matxy1.xx*m_ion + matxy2.xx*m_e) /
			((m_ion + m_e)*N)); 
		mat.xy += -hsub*((matxy1.xy*m_ion + matxy2.xy*m_e) /
			((m_ion + m_e)*N));
		mat.yx += -hsub*((matxy1.yx*m_ion + matxy2.yx*m_e) /
			((m_ion + m_e)*N));
		mat.yy += -hsub*((matxy1.yy*m_ion + matxy2.yy*m_e) /
			((m_ion + m_e)*N));
		
		mat.Inverse(invmatrix);

		if (iMinor == CHOSEN) printf("\n%d mat xx %1.9E xy %1.9E yx %1.9E yy %1.9E \ninvmatrix %1.9E %1.9E %1.9E %1.9E\n\n",
			iMinor, mat.xx, mat.xy, mat.yx, mat.yy,
			invmatrix.xx, invmatrix.xy, invmatrix.yx, invmatrix.yy);

		invcoeffselfviz = 1.0 / (1.0 - hsub*(p_coeffself_iz[iMinor] / N)); // rate of change of epsilon iz wrt viz
		invcoeffselfvez = 1.0 / (1.0 - hsub*(p_coeffself_ez[iMinor] / N));

	}
	else {
		// epsilon = 0
	};

	memcpy(&(p_invmatrix[iMinor]), &invmatrix, sizeof(Tensor2));
	p_invcoeffselfviz[iMinor] = invcoeffselfviz;
	p_invcoeffselfvez[iMinor] = invcoeffselfvez;
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
	if (0) { //((iMinor > BEGINNING_OF_CENTRAL) && ((iMinor - BEGINNING_OF_CENTRAL) % 8000 == 0)) {
		printf("iVertex %d regr %1.10E factor1 %1.9E factor2 %1.9E \n",
			iMinor-BEGINNING_OF_CENTRAL, p_regr[iMinor], p_factor1[iMinor], p_factor2[iMinor]);
	}
}

__global__ void kernelCreateJacobiRegressorxy
(f64_vec2 * __restrict__ p_regr,
	f64_vec2 * __restrict__ p_factoreps,
	f64_tens2 * __restrict__ p_factor2)
{
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;
	if (iMinor == CHOSEN) printf("CreateJacobiRegressorxy : %d : xx %1.10E xy %1.10E yx %1.10E yy %1.10E eps.xy %1.10E %1.10E", iMinor,
		p_factor2[iMinor].xx, p_factor2[iMinor].xy, p_factor2[iMinor].yx, p_factor2[iMinor].yy, p_factoreps[iMinor].x, p_factoreps[iMinor].y);
	p_regr[iMinor] = p_factor2[iMinor] * p_factoreps[iMinor];
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
