
#define A_DOT_RETARDED

// Version 0.95:
// Always choose the 3 closest neighbours except for a selected point.
// Well this isn't as big of a difference perhaps as we think.
// Problem eventually and may need to generate Del tris sooner or later.
// No special tweaks about which points to select and indeed we might select fewer again.
// Could go back to spacing 2. 


#ifdef __CUDACC__
__device__ __forceinline__ f64 GetEzShape(f64 r) {
	return 1.0-1.0/(1.0+exp(-16.0*(r-4.2))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}
#else
f64 inline GetEzShape_(f64 r) {
	return 1.0-1.0/(1.0+exp(-16.0*(r-4.2))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}
#endif
// repeat this in cpp file


//#include "systdata.cpp"


// Version 0.94

// Detect duplicate influence lists and expand to 3 points. Etc.


// Version 0.91

// Introduce a switch to make us do the effect of A on Gauss consistently with grad phi on Gauss.

// We then will proceed to do a programmatic test: what about each time if we try raising Ax to reduce the
// persistent peak Amp-x error? At what number vertex is it located?

// Depending on the results of that, we plot our next move. It _should_ be possible to do this solve??
// We know it works except for Gauss & Amp-xy together.


bool const bConsistentChargeflow = false; // false: effect of A on Gauss more like effect of A on Ampere.
									// true: effect of A on Gauss more like effect of phi on Gauss.
bool const bRescaleGauss = true;
bool const bNeumannPhi = false;
bool const bDisconnectAxy = false;
bool const bDisconnectPhi = false;
bool const bDiscAxyFromGauss = false; // for true, change to bConsistentChargeFlow
bool const bDiscPhiFromAmpere = false;

int const COEFFS_OF_GEOMETRIC_NEIGH = 1;
int const PLANAR_WEIGHTS = 2;
int const COEFFS_OF_CONNECTED = 3;
int const COEFFS_OF_CONNECTED_CORNERS = 4;
int const iWeightSwitch = PLANAR_WEIGHTS;

// Initially keep it simple: try just 2 switches: 1 is for phi, the other for A and sigma, all circumstances.
int const OPPOSING_BETA = 0; // only slightly fairer than 50/50 but almost as likely to succeed.
int const CORNERS_BETA = 1;   // the original version
int const SIMPSON_BETA = 2;      // This is the fairest.
int const FORCE_FIFTY_FIFTY = 3; // This is the most likely to succeed??
int const FORCE_TWELTHS = 4;  // 1/12, 5/12, 5/12, 1/12 -- slightly more likely to succeed than SIMPSON_BETA ??

int iEdgeAvg_for_Gauss = SIMPSON_BETA; 
// A and sigma
// .. and phi. Let's try just one way for everything.
// In each of these cases we need to consider the 5 cases above.

bool bGlobalSpitFlag,bGlobalInitial;
real store_coeff_self;

extern char * report_time(int action);
extern real GetPossiblyPeriodicDist(Vector2 & vec1, Vector2 & vec2);
extern real GetPossiblyPeriodicDistSq(Vector2 & vec1, Vector2 & vec2);

int globaldebugswitch = 0;

class CalculateAccelsClass
{
public:
	// exists only to do a calculation repeatedly from some stored data

	Vector3 omega_ce, omega_ci;
	Tensor3 omega_ci_cross;
	
	real nu_eiBar, nu_eHeart, nu_ieBar, 
			nu_en_MT, nu_in_MT, nu_ne_MT, nu_ni_MT,
			n_i, n_n, n_e;
			
	real heat_transfer_rate_in,heat_transfer_rate_ni,
		heat_transfer_rate_en,heat_transfer_rate_ne,
		heat_transfer_rate_ei,heat_transfer_rate_ie;
	
	//Vector3 ROC_v_ion_due_to_Rie;
		//a_ion_pressure_and_E_accel,
		//a_neut_pressure,
		//a_ion_pressure;
	// Don't think we are using any of these. a_pressure lives in Vertex.

	Vector3 vrel_e, ROC_v_ion_thermal_force; 
	Tensor3 Upsilon_nu_eHeart;
	Tensor3 Rie_thermal_force_matrix;
	real fric_dTe_by_dt_ei;
	Tensor3 Rie_friction_force_matrix;
	Tensor3 Ratio_times_Upsilon_eHeart;	
	real StoredEz;
	bool bNeutrals;

	// EASIER WAY:
	// Let's just stick simple Ohm's Law v_e(v_i,v_n)
	// in CalculateCoefficients.

	Vector3 Unreduced_v_e_0;
	Tensor3 Unreduced_beta_e_ion;
	Tensor3 Unreduced_beta_e_neut;
	Tensor3 Unreduced_sigma_e;

	//Vector3 SimpleOhms_ve0;
	//Tensor3 SimpleOhms_beta_neut, SimpleOhms_beta_ion; // xy = effect of viy on vex
	// Don't understand this yet...

//	Vector3 Reduced_v_e_0;
//	Tensor3 Ohms_sigma; // reduced Ohm's Law
	// Is this calculated by this routine at all and if so, why?
	// Reduced equation requires ion stuff as input so I do not understand right now what
	// it was doing here.

	CalculateAccelsClass(){};

	void CalculateCoefficients(Vertex  *pVertex)
	{

		static Tensor3 const ID3x3 (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
		static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
		static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
		static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
		static real const kB_to_3halves = sqrt(kB)*kB;
		static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
		static real const over_sqrt_m_e = 1.0/sqrt(m_e);
		static real const qoverMc = q/(m_ion*c);
		static real const qovermc = q/(m_e*c);
		real const NU_EI_FACTOR = 1.0/(3.44e5);
		static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);
		real area, det;
		real T_ion, T_n, T_e, sqrt_Te, ionneut_thermal, electron_thermal,
			lnLambda, s_in_MT, s_in_visc, s_en_MT,s_en_visc,
			nu_en_visc;
		
		// Inputs: AreaCell, B, a_pressure_elec

		// nu_eibar, nu_in, nu_en
		// ======================
		// Get nu_eiBar
		// Get nu_en, nu_in, nu_ni, nu_ne, nu_eHeart
	
		StoredEz = pVertex->E.z;

		area = pVertex->AreaCell;
		n_i = pVertex->Ion.mass/area;
		n_e = pVertex->Elec.mass/area;
		n_n = pVertex->Neut.mass/area;

		if (pVertex->Ion.mass > 0.0) {
			T_ion = pVertex->Ion.heat/pVertex->Ion.mass;   
		} else {
			T_ion = 0.0;
		};
		if (pVertex->Neut.mass > 0.0) {
			T_n = pVertex->Neut.heat/pVertex->Neut.mass;
		} else {
			T_n = 0.0;
		};
		if (pVertex->Elec.mass > 0.0) {
			T_e = pVertex->Elec.heat/pVertex->Elec.mass;
			sqrt_Te = sqrt(T_e);
		} else {
			T_e = 0.0;
			sqrt_Te = 0.0;
		};
		
		ionneut_thermal = sqrt(T_ion/m_ion+T_n/m_n); // hopefully not sqrt(0)
		electron_thermal = sqrt_Te*over_sqrt_m_e; // possibly == 0

		lnLambda = Get_lnLambda(n_i,T_e); // anything strange in there?

		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &s_in_MT, &s_in_visc);
		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&s_en_MT,&s_en_visc);
		// To use combined temperature looks to be more intelligent -- rel temp GZSB(6.55) for ion, neutral at least.
		
		if (T_e != 0.0) {
			nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
		} else {
			nu_eiBar = 0.0;
		};
		
		nu_ieBar = nu_eiBar; // always same when n_e=n_i
		
		nu_en_MT = n_n*s_en_MT*electron_thermal;
		nu_in_MT = n_n*s_in_MT*ionneut_thermal;
		nu_ne_MT = n_e*s_en_MT*electron_thermal;
		nu_ni_MT = n_i*s_in_MT*ionneut_thermal;
		
		nu_en_visc = n_n*s_en_visc*electron_thermal; 
		
		// those should all be fine though may == 0
		
		nu_eHeart = 1.87*nu_eiBar + nu_en_visc; // note, used visc
				 
		heat_transfer_rate_in = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
										*nu_in_MT; // ratio nu_in/nu_ni = n_n/n_i
		heat_transfer_rate_ni = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
										*nu_ni_MT;
		heat_transfer_rate_ne = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
										*nu_ne_MT;
		heat_transfer_rate_en = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
										*nu_en_MT;
		heat_transfer_rate_ei = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
										*nu_eiBar;
		heat_transfer_rate_ie = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
										*nu_ieBar;
		
		// OK that bit is clear and as expected.
		// So what is the difference when we transfer NT ?

		// (n_n/n_i) transfer_rate_ni = transfer_rate_in


		omega_ce = qovermc*pVertex->B;
		omega_ci = qoverMc*pVertex->B; // note: if ion acceleration stage, we could if we wanted work out B at k+1 first.
		omega_ci_cross.MakeCross(omega_ci);

		// NOTE: Uses GradTe so it better exist.

		// Populate Upsilon(nu_eHeart):
		real nu = nu_eHeart;
		Vector3 omega = omega_ce;

		det = nu*nu + omega.dot(omega);

		// (nu - omega x ) ^-1 :
		Upsilon_nu_eHeart.xx = nu*nu+omega.x*omega.x;
		Upsilon_nu_eHeart.xy = -nu*omega.z + omega.x*omega.y;
		Upsilon_nu_eHeart.xz = nu*omega.y + omega.x*omega.z;
		Upsilon_nu_eHeart.yx = nu*omega.z + omega.x*omega.y;
		Upsilon_nu_eHeart.yy = nu*nu + omega.y*omega.y;
		Upsilon_nu_eHeart.yz = -nu*omega.x + omega.y*omega.z;
		Upsilon_nu_eHeart.zx = -nu*omega.y + omega.z*omega.x;
		Upsilon_nu_eHeart.zy = nu*omega.x + omega.y*omega.z;
		Upsilon_nu_eHeart.zz = nu*nu + omega.z*omega.z;
		
		Upsilon_nu_eHeart = Upsilon_nu_eHeart/det;
	
		if (nu_eHeart > 0.0) {
			Ratio_times_Upsilon_eHeart = (nu_eiBar/nu_eHeart)*Upsilon_nu_eHeart;
		} else {
			ZeroMemory(&Ratio_times_Upsilon_eHeart,sizeof(Tensor3));
		};

		// Unreduced law v_e(v_i,v_n,E) :
		
		Tensor3 omega_ce_cross;
		omega_ce_cross.MakeCross(omega_ce);
		Tensor3 chi_inv = 
			  (ID3x3-0.9*this->Ratio_times_Upsilon_eHeart)*this->nu_eiBar
			+ (m_n/(m_e+m_n))*this->nu_en_MT*ID3x3 
			- omega_ce_cross;
		Tensor3 chi = chi_inv.Inverse();
		
		Unreduced_v_e_0 = chi*(Make3(pVertex->a_pressure_elec,0.0) - 1.5/m_e*
			(this->Ratio_times_Upsilon_eHeart*Make3(pVertex->GradTe,0.0)));
		
		// Note: Grad Te USED
		
	//	SimpleOhms_vez0 = (-qoverm*StoredEz 
	//		-(1.5/m_e)*((this->Ratio_times_Upsilon_eHeart*pTri->GradTe).z)
	//		)/chi; 
		// Where did this get used?? In Stage 3?? Careful on that.
		
		Unreduced_beta_e_ion = chi*(ID3x3-0.9*this->Ratio_times_Upsilon_eHeart)*this->nu_eiBar;
		Unreduced_beta_e_neut = chi*(m_n/(m_e+m_n))*this->nu_en_MT;
		
		Unreduced_sigma_e = chi*(-qoverm)*ID3x3;

		// Components for acceleration ions:		
		Rie_friction_force_matrix = 
			nu_ieBar*(m_e/m_i)*(ID3x3-0.9*Ratio_times_Upsilon_eHeart);		
		Rie_thermal_force_matrix = (1.5/m_i)*Ratio_times_Upsilon_eHeart;
		// We multiply by + GradTe for ions
		//ROC_v_ion_due_to_Rie = 
		//		  Rie_thermal_force_matrix * pTri->GradTe
		//		  + R_ie_friction_force_matrix*(ve-vi);
		ROC_v_ion_thermal_force = Rie_thermal_force_matrix*Make3(pVertex->GradTe,0.0);
		
	}

	// The following is called when doing Stage II's feint acceleration
	// to reduce an Ohm's Law, and create the v_k+1 to be used for displacement.
	void inline Populate_Acceleration_Coefficients(real H[6][6], 
												   real a0[6], 
												   real a1[6], 
												   real a_sigma1[6][3], 
												   Vector3 E_k,
												   Vertex * pVertex)
	{
		real factor;
		Vector3 Effect_ion_ve0, Effect_of_Ek;
		Tensor3 Effect_ion_ion, Effect_ion_neut,Effect_of_E_via_e;

		ZeroMemory(H,sizeof(real)*6*6);
		ZeroMemory(a0, sizeof(real)*6);
		ZeroMemory(a1, sizeof(real)*6);
		ZeroMemory(a_sigma1,sizeof(real)*6*3);

		 // Pressure:
		a0[0] += pVertex->a_pressure_ion.x + this->ROC_v_ion_thermal_force.x;
		a0[1] += pVertex->a_pressure_ion.y + this->ROC_v_ion_thermal_force.y;
		a0[2] += this->ROC_v_ion_thermal_force.z;
		// neutral pressure lower down.

		// Let's face it, we have not been very careful about whether these forces
		// apply at time t_k ; we don't have the opportunity to do that at Stage II.
		// Bit concerning?
		// #@#@#@#@#@#@#@#@
		
		// magnetic Lorentz for ions:
		H[0][1] += omega_ci.z;
		H[0][2] -= omega_ci.y;
		H[1][0] -= omega_ci.z;
		H[1][2] += omega_ci.x;
		H[2][0] += omega_ci.y;
		H[2][1] -= omega_ci.x;
		// Again, cannot look forward to use B_k+1 at all. (That makes a nonlinear solver.)

		H[0][0] -= Rie_friction_force_matrix.xx;
		H[0][1] -= Rie_friction_force_matrix.xy;
		H[0][2] -= Rie_friction_force_matrix.xy;
		H[1][0] -= Rie_friction_force_matrix.yx;
		H[1][1] -= Rie_friction_force_matrix.yy;
		H[1][2] -= Rie_friction_force_matrix.yz;
		H[2][0] -= Rie_friction_force_matrix.zx;
		H[2][1] -= Rie_friction_force_matrix.zy;
		H[2][2] -= Rie_friction_force_matrix.zz;

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Effect of v_e :
		// v_e = Unreduced_v_e_0 + Unreduced_beta_e_ion v_i
		//			+ Unreduced_beta_e_neut v_n + Unreduced_sigma_e E
		// a_ion += Rie_friction_force_matrix v_e
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		Effect_ion_ve0 = Rie_friction_force_matrix*Unreduced_v_e_0; // matrix * vec3
		a0[0] += Effect_ion_ve0.x;
		a0[1] += Effect_ion_ve0.y;
		a0[2] += Effect_ion_ve0.z;

		Effect_ion_ion = Rie_friction_force_matrix * Unreduced_beta_e_ion; // multiply matrices

		H[0][0] += Effect_ion_ion.xx;
		H[0][1] += Effect_ion_ion.xy;
		H[0][2] += Effect_ion_ion.xz;
		H[1][0] += Effect_ion_ion.yx;
		H[1][1] += Effect_ion_ion.yy;
		H[1][2] += Effect_ion_ion.yz;
		H[2][0] += Effect_ion_ion.zx;
		H[2][1] += Effect_ion_ion.zy;
		H[2][2] += Effect_ion_ion.zz;
		
		if (bNeutrals) {
			
			Effect_ion_neut =  Rie_friction_force_matrix * Unreduced_beta_e_neut; 

			H[0][3] += Effect_ion_neut.xx;
			H[0][4] += Effect_ion_neut.xy;
			H[0][5] += Effect_ion_neut.xz;
			H[1][3] += Effect_ion_neut.yx;
			H[1][4] += Effect_ion_neut.yy;
			H[1][5] += Effect_ion_neut.yz;
			H[2][3] += Effect_ion_neut.zx;
			H[2][4] += Effect_ion_neut.zy;
			H[2][5] += Effect_ion_neut.zz;

			// Direct effect of neutral on ion:

			factor = this->nu_in_MT*(m_n/(m_ion+m_n));
			H[0][0] -= factor;
			H[0][3] += factor;
			H[1][1] -= factor;
			H[1][4] += factor;
			H[2][2] -= factor;
			H[2][5] += factor;

			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			// SHOULD we be working in velocity and not momentum here?? We work in mom for everything else, almost, right?
			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


			// Neutral acceleration:
			// _____________________

			a0[3] += pVertex->a_pressure_neut_or_overall.x;
			a0[4] += pVertex->a_pressure_neut_or_overall.y;
			
			// i-n friction:

			factor = this->nu_ni_MT*(m_ion/(m_ion+m_n));
			H[3][0] += factor;
			H[3][3] -= factor;
			H[4][1] += factor;
			H[4][4] -= factor;
			H[5][2] += factor;
			H[5][5] -= factor;

			// e-n friction:
			factor = (m_e/(m_e+m_n))*this->nu_ne_MT;

			// - factor * vn:
			H[3][3] -= factor;
			H[4][4] -= factor;
			H[5][5] -= factor;
			
			// v_e_k+1 = v_e_0 + beta_e_ion v_i + beta_e_n v_n + sigma_e E

			a0[3] += factor*Unreduced_v_e_0.x;
			a0[4] += factor*Unreduced_v_e_0.y;
			a0[5] += factor*Unreduced_v_e_0.z;

			H[3][0] += factor*Unreduced_beta_e_ion.xx;
			H[3][1] += factor*Unreduced_beta_e_ion.xy;
			H[3][2] += factor*Unreduced_beta_e_ion.xz;
			H[4][0] += factor*Unreduced_beta_e_ion.yx;
			H[4][1] += factor*Unreduced_beta_e_ion.yy;
			H[4][2] += factor*Unreduced_beta_e_ion.yz;
			H[5][0] += factor*Unreduced_beta_e_ion.zx;
			H[5][1] += factor*Unreduced_beta_e_ion.zy;
			H[5][2] += factor*Unreduced_beta_e_ion.zz;

			H[3][3] += factor*Unreduced_beta_e_neut.xx;
			H[3][4] += factor*Unreduced_beta_e_neut.xy;
			H[3][5] += factor*Unreduced_beta_e_neut.xz;
			H[4][3] += factor*Unreduced_beta_e_neut.yx;
			H[4][4] += factor*Unreduced_beta_e_neut.yy;
			H[4][5] += factor*Unreduced_beta_e_neut.yz;
			H[5][3] += factor*Unreduced_beta_e_neut.zx;
			H[5][4] += factor*Unreduced_beta_e_neut.zy;
			H[5][5] += factor*Unreduced_beta_e_neut.zz;

			// do E effect via e's at the end
		};



		// Here start working on end-of-step values:

		
		a1[0] = a0[0];
		a1[1] = a0[1];
		a1[2] = a0[2];
		
		a0[0] += qoverM*E_k.x;
		a0[1] += qoverM*E_k.y;
		a0[2] += qoverM*E_k.z;
		a_sigma1[0][0] += qoverM;
		a_sigma1[1][1] += qoverM;
		a_sigma1[2][2] += qoverM; // We need full a_sigma because we have the effect
								 // via electrons to worry about.

		Effect_of_E_via_e = Rie_friction_force_matrix * Unreduced_sigma_e; // multiply matrices
		Effect_of_Ek = Effect_of_E_via_e*E_k;
		a0[0] += Effect_of_Ek.x;
		a0[1] += Effect_of_Ek.y;
		a0[2] += Effect_of_Ek.z;
		a_sigma1[0][0] += Effect_of_E_via_e.xx;
		a_sigma1[0][1] += Effect_of_E_via_e.xy;
		a_sigma1[0][2] += Effect_of_E_via_e.xz;
		a_sigma1[1][0] += Effect_of_E_via_e.yx;
		a_sigma1[1][1] += Effect_of_E_via_e.yy;
		a_sigma1[1][2] += Effect_of_E_via_e.yz;
		a_sigma1[2][0] += Effect_of_E_via_e.zx;
		a_sigma1[2][1] += Effect_of_E_via_e.zy;
		a_sigma1[2][2] += Effect_of_E_via_e.zz;

		if (bNeutrals) {
			
			factor = (m_e/(m_e+m_n))*this->nu_ne_MT;
			Effect_of_E_via_e = factor*Unreduced_sigma_e;
			Effect_of_Ek = Effect_of_E_via_e*E_k;
			a0[3] += Effect_of_Ek.x;
			a0[4] += Effect_of_Ek.y;
			a0[5] += Effect_of_Ek.z;
			a_sigma1[3][0] += Effect_of_E_via_e.xx;
			a_sigma1[3][1] += Effect_of_E_via_e.xy;
			a_sigma1[3][2] += Effect_of_E_via_e.xz;
			a_sigma1[4][0] += Effect_of_E_via_e.yx;
			a_sigma1[4][1] += Effect_of_E_via_e.yy;
			a_sigma1[4][2] += Effect_of_E_via_e.yz;
			a_sigma1[5][0] += Effect_of_E_via_e.zx;
			a_sigma1[5][1] += Effect_of_E_via_e.zy;
			a_sigma1[5][2] += Effect_of_E_via_e.zz;
		};

	} // end function

};

void TriMesh::ComputeOhmsLaw()
{
	// Uses pVertex->E as E_k in the ion acceleration => ion displacement

	// So we really need to have hold of that.
	// (Will also use trapezoidal E for ion acceleration in Stage III.)

	// !@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@


	// creates Ohm's for v_k+1 for 3 species. This enables species displacement,
	// trapezoidal for heavy species, and enables to calculate charge density in solver.
	// We re-do acceleration properly having done that part.
	// (v_e - avg of vi,vn) becomes settled, so the reduced Ohm's Law is perhaps the most influential part.
	

	static Tensor3 const ID3x3 (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
	static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB)*kB;
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const qoverMc = q/(m_ion*c);
	static real const qovermc = q/(m_e*c);

	real const NU_EI_FACTOR = 1.0/(3.44e5);
	static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
	// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);
	// To find earlier code go back to simulation.cpp version 1.1

	// The reason we cannot just apply solution to linear DEs is that
	// we want to do heating, which is nonlinear. Heating does not affect
	// acceleration, but we have to have so many waypoints for
	// % change in v, that there is no point in using actual
	// solution instead of cheap implicit method RK[-2].

	// Instructions for how to do differently are in serious.lyx

	CalculateAccelsClass Y;

	Vector3 v_ion_k, v_neut_k, electron_pressure_accel;
	real T_ion_k, T_neut_k, T_e_k,t;
	
	real H[6][6];
	real Hsq[6][6];
	Matrix_real LHS;
	
	real v[6], vnext[6];
	real RHS[6];
	real a0[6],a[6],a1[6], a_sigma1[6][3];

	real RHS_additional[6];
	int i,j, iStep;
	real hSub;
	real RHS_sigma_x[6], RHS_sigma_y[6],RHS_sigma_z[6],
		sigma_x[6], sigma_y[6],sigma_z[6],
		sigmanextx[6], sigmanexty[6],sigmanextz[6],
		RHS_Ex[6],RHS_Ey[6],RHS_Ez[6];
	
	// code == COMPUTE_SIGMA
	
	// = Predict v_heavy_k+1 using present vrel, (in effect, vrel_k+1=vrel_k) ;
	// use this to create vrel_k+1 as linear function of E_k+1

	// pTri->vrel0 : vrel = vrel0 + sigma*E
	// The idea is to create a definite relationship that will then be actually
	// enforced to give vrel_k+1.
		// Because this appears here, we ought to do some heat conduction before COMPUTE_SIGMA call :
	Vertex * pVertex;
	long iVertex;
	real lambda, lambdahalf;

	this->GetGradTeOnVertices(); // ready to get thermal force

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{

		if (iVertex == 43600) {
			iVertex = iVertex;
		};

		Y.CalculateCoefficients(pVertex); // get things like nu, based on present Te
			
		v_ion_k = pVertex->Ion.mom/pVertex->Ion.mass; // half-time values
		T_ion_k = pVertex->Ion.heat/pVertex->Ion.mass;
		T_e_k = pVertex->Elec.heat/pVertex->Elec.mass;
		//v_e_z = pTri->elec.mom.z/pTri->elec.mass;
		
		if (pVertex->Neut.mass != 0.0) {
			v_neut_k = pVertex->Neut.mom/pVertex->Neut.mass;
			T_neut_k = pVertex->Neut.heat/pVertex->Neut.mass;
			Y.bNeutrals = true;
		} else {
			ZeroMemory(&v_neut_k,sizeof(Vector3));
			T_neut_k = 1.0;
			Y.bNeutrals = false;
		}
		
		v[0] = v_ion_k.x;
		v[1] = v_ion_k.y;
		v[2] = v_ion_k.z;
		v[3] = v_neut_k.x;
		v[4] = v_neut_k.y;
		v[5] = v_neut_k.z;
		ZeroMemory(sigma_x,sizeof(real)*6);
		ZeroMemory(sigma_y,sizeof(real)*6);
		ZeroMemory(sigma_z,sizeof(real)*6); // effect of Ex,Ey,Ez on 2x3 directions
		
		int const NUMSTEPS = 4;
		hSub = h*0.25; 
		
		LHS.Invoke(6);

		Y.Populate_Acceleration_Coefficients(H, a0, a1, a_sigma1, pVertex->E,pVertex); // pVertex gives a_pressure
		// a and H change from a0 to a1, H0 to H1 as E moves from Ek to E_k+1 ; 
		// a_sigma1 is for E_k+1 

		// Might as well let v_e(E) at each intermediate time
		// Pretty sure we are doing no evaluations at t_k so v_e_k doesn't matter.

		// . Revisit Bwd Improved method: eval times?

			// E = tween from E_k, E_k+1 unknown.
			// v_e = v_e_0 + beta v_ion + chi (-q/m) (1-lambda) E_k + chi (-q/m) lambda E_unknown
			// . Get a0(t) <-- initially comes from v(E_k)
			// a_sigma comes through that latter and,
			// a_i += q/M (lambda E_k+1 + (1-lambda) E_k)
			// 
			// Admitting this means that we have to do a decomp for every substep because the LHS
			// matrix of self-effects is different every substep. Hard cheese! This procedure is not the main expense!!
			
		for (i = 0; i < 6; i++)
		for (j = 0; j < 6; j++) 
		{
				// element i j of H^2 ?
				// multiply row i and column j of H
			Hsq[i][j] = H[i][0]*H[0][j] + H[i][1]*H[1][j] + H[i][2]*H[2][j]
					  + H[i][3]*H[3][j] + H[i][4]*H[4][j] + H[i][5]*H[5][j];
			LHS.LU[i][j] = ((i == j)?1.0:0.0) - hSub*H[i][j] + (hSub*hSub*0.5)*Hsq[i][j] ;
		}
		LHS.LUdecomp();
		
		for (iStep = 0; iStep < NUMSTEPS; iStep++)
		{
			lambda = hSub*((real)(iStep+1))/h;
			lambdahalf = hSub*(((real)iStep)+0.5)/h;
			
			for (i = 0; i < 6; i++)
			{
				a[i] = (1.0-lambda) * a0[i] + lambda * a1[i];
			};
			//a_sigma = lambda * a_sigma1;
			// look over Bwd Improved Euler for eval times - get to know really well !

			for (i = 0; i < 6; i++)
			{
				// """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
				// X_k+1 = (1-hF+hhFF/2)^(-1) (X_k + h a0_k+1/2 - hh/2 F a0_k+1)
				// """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

				RHS_additional[i] = hSub*( (1.0-lambdahalf)*a0[i]+lambdahalf*a1[i] )
						- hSub*hSub*0.5*
					(H[i][0]*a[0] + H[i][1]*a[1]
					+ H[i][2]*a[2] + H[i][3]*a[3]
					+ H[i][4]*a[4] + H[i][5]*a[5]); 

				RHS_sigma_x[i] = lambdahalf*hSub*a_sigma1[i][0] - lambda*hSub*hSub*0.5*
					(H[i][0]*a_sigma1[0][0]
					+ H[i][1]*a_sigma1[1][0]
					+ H[i][2]*a_sigma1[2][0]
					+ H[i][3]*a_sigma1[3][0]
					+ H[i][4]*a_sigma1[4][0]
					+ H[i][5]*a_sigma1[5][0]);
				RHS_sigma_y[i] = lambdahalf*hSub*a_sigma1[i][1] - lambda*hSub*hSub*0.5*
					(H[i][0]*a_sigma1[0][1]
					+ H[i][1]*a_sigma1[1][1]
					+ H[i][2]*a_sigma1[2][1]
					+ H[i][3]*a_sigma1[3][1]
					+ H[i][4]*a_sigma1[4][1]
					+ H[i][5]*a_sigma1[5][1]);
				RHS_sigma_z[i] = lambdahalf*hSub*a_sigma1[i][2] - lambda*hSub*hSub*0.5*
					(H[i][0]*a_sigma1[0][2]
					+ H[i][1]*a_sigma1[1][2]
					+ H[i][2]*a_sigma1[2][2]
					+ H[i][3]*a_sigma1[3][2]
					+ H[i][4]*a_sigma1[4][2]
					+ H[i][5]*a_sigma1[5][2]);
					
				RHS[i] = v[i] + RHS_additional[i];
				RHS_Ex[i] = sigma_x[i] + RHS_sigma_x[i];
				RHS_Ey[i] = sigma_y[i] + RHS_sigma_y[i];
				RHS_Ez[i] = sigma_z[i] + RHS_sigma_z[i];
			};
			
			LHS.LUSolve(RHS,vnext);
			LHS.LUSolve(RHS_Ex,sigmanextx);
			LHS.LUSolve(RHS_Ey,sigmanexty);
			LHS.LUSolve(RHS_Ez,sigmanextz);
				
			for (i = 0; i < 6; i++)
			{
				v[i] = vnext[i]; // v represents v0
				sigma_x[i] = sigmanextx[i];
				sigma_y[i] = sigmanexty[i];
				sigma_z[i] = sigmanextz[i];
			}
		};
		
		// Ionic Ohm's Law, t_k+1 :

		pVertex->v_i_0.x = v[0];
		pVertex->v_i_0.y = v[1];
		pVertex->v_i_0.z = v[2];
		pVertex->sigma_i.xx = sigma_x[0];
		pVertex->sigma_i.xy = sigma_y[0];
		pVertex->sigma_i.xz = sigma_z[0]; // sigma_z = effect of E_z
		pVertex->sigma_i.yx = sigma_x[1];
		pVertex->sigma_i.yy = sigma_y[1];
		pVertex->sigma_i.yz = sigma_z[1]; // sigma_z = effect of E_z
		pVertex->sigma_i.zx = sigma_x[2];
		pVertex->sigma_i.zy = sigma_y[2];
		pVertex->sigma_i.zz = sigma_z[2]; // sigma_z = effect of E_z
		
		// will need law for v_n:
		pVertex->v_n_0.x = v[3];     
		pVertex->v_n_0.y = v[4];
		pVertex->v_n_0.z = v[5];
		pVertex->sigma_n.xx = sigma_x[3];
		pVertex->sigma_n.xy = sigma_y[3];
		pVertex->sigma_n.xz = sigma_z[3];
		pVertex->sigma_n.yx = sigma_x[4];
		pVertex->sigma_n.yy = sigma_y[4];
		pVertex->sigma_n.yz = sigma_z[4];
		pVertex->sigma_n.zx = sigma_x[5];
		pVertex->sigma_n.zy = sigma_y[5];
		pVertex->sigma_n.zz = sigma_z[5];

		// If we want to drop these from vertex there is a way to save 2 tensors: work this out again following knowledge
		// of E_k+1 and do displacement that way ; need only to store sigma_J instead. Some care needed on that.
		
		// Do this simple and memory-inefficient way for now.
		// Estimating 144 doubles per vertex come from solver; 60 from other; 36 from these Ohm's Laws. Maybe can save 10 or 20 other; 
		// can save none from solver and probably solver needs more (6 x 4 x 4 = 96 but want room for > 6 neighs ...).

		// Now create the reduced Ohm's Law for v_e :
		// __________________________________________
		
		// Substitute in for v_i, v_n to give reduced law;
		// and add what is already in the unreduced law:
		

		// ????????????????????????????????????????????????????????????????????
		// This is producing nonsense (on debug) for both v_i_0 and v_e_0.
		// ????????????????????????????????????????????????????????????????????


		pVertex->v_e_0 = Y.Unreduced_v_e_0
							+ Y.Unreduced_beta_e_ion*pVertex->v_i_0
							+ Y.Unreduced_beta_e_neut*pVertex->v_n_0;
		pVertex->sigma_e = Y.Unreduced_sigma_e
							+ Y.Unreduced_beta_e_ion*pVertex->sigma_i
							+ Y.Unreduced_beta_e_neut*pVertex->sigma_n;

		// Note that sigma_n is for v_n_k+1 whereas displacement is to be generated trapezoidally.
		// sigma_n could be significant when there are not many neutrals left (not that an error then matters a lot necessarily).
		++pVertex;
	};
}

void TriMesh::EstimateInitialOhms_zz()
{
	// This will create v_e_0, sigma_e, v_i_0, sigma_i, v_n_0, sigma_n, with vxy = 0 
	Vertex * pVertex;
	long iVertex;
	CalculateAccelsClass Y;

	printf("call Set_nT .. \n");
	this->Set_nT_and_Get_Pressure(SPECIES_ELECTRON); // it will be unused but it should be pop'd.
	
	printf("done\n");

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		memset(&(pVertex->v_i_0),0,sizeof(Vector3));
		memset(&(pVertex->v_e_0),0,sizeof(Vector3));
		memset(&(pVertex->v_n_0),0,sizeof(Vector3));
		memset(&(pVertex->sigma_i),0,sizeof(Tensor3));
		memset(&(pVertex->sigma_e),0,sizeof(Tensor3));
		memset(&(pVertex->sigma_n),0,sizeof(Tensor3));
		memset(&(pVertex->xdotdot),0,sizeof(Vector2));

		if (pVertex->flags == DOMAIN_VERTEX) {
			

   			Y.CalculateCoefficients(pVertex); // get things like nu, based on present Te
			// Now assume v_n_z = 0.0
			// m_i v_i_z + m_e v_e_z = 0.0
		
			real factor = 1.0/(1.0 + Y.Unreduced_beta_e_ion.zz*m_e/m_i); // assume have +beta(-me/mi)ve and move to LHS
			pVertex->v_e_0.z = factor*Y.Unreduced_v_e_0.z;
			pVertex->sigma_e.zz = factor*Y.Unreduced_sigma_e.zz;

			if (iVertex == 16416){
				printf ("16416: sigma_e_zz %1.9E ( %1.5E )\n",pVertex->sigma_e.zz,pVertex->v_e_0.z);
				printf ("Y.nu_eiBar %1.8E nu_en %1.8E\n",Y.nu_eiBar,Y.nu_en_MT);
			}
			pVertex->v_i_0.z = -(m_e/m_i)*pVertex->v_e_0.z;
			pVertex->sigma_i.zz = -(m_e/m_i)*pVertex->sigma_e.zz;
			
			// NEW EFFORT:
			pVertex->sigma_e.xx = factor*Y.Unreduced_sigma_e.xx;
			pVertex->sigma_e.yy = factor*Y.Unreduced_sigma_e.yy; // ???

			// use xdotdot to store some graph info:
			pVertex->xdotdot.x = pVertex->Elec.mass*pVertex->sigma_e.zz / pVertex->AreaCell; // put in a graph of this.
		};

		// Anything else needed for solve to run?
		++pVertex;
	};
}

void TriMesh::CreateODECoefficientsAz(bool bInitial)
{
	FILE * fp;

	long iVertex;
	Vertex * pVertex;
	real edgelen_proj;
	Vector2 extranormal, rhat, Exy_over_phi;
	bool bAnode, bAnodePrev, bAnode_i,bAnodeNext;
	real factor;
	Tensor3 sigma___;
	
	// PLAN:
	// In use, we load a set of coefficients into memory and apply them how needful ...
	// Grad neighbour i data;  eps += beta . neighdata

	// That is what we prefer, to saying, each equation at a time, with ITS set of all neighbour
	// coefficients, scurrying off to each neighbour in turn.

	long tri_len, neigh_len;
	long izTri[128];
	long izNeigh[128];
	Vertex * pNeigh, *pNeighPrev, *pNeighNext;
	real dist1, dist2;
	int iprev, inext;
	real Lapcoeff[MAXNEIGH];
	real Lapcoeff_self = 0.0;
	memset(Lapcoeff,0,sizeof(real)*MAXNEIGH);
	Vector2 u[4];
	ConvexPolygon cp;
	Triangle * pTri1,*pTri2, *pTri;
	int i;
	real beta_2_self, beta_2_1, beta_2_2, beta_1_self, beta_1_3, beta_1_2,
		beta_3_self, beta_3_2;
	real shoelace;
	Vector2 grad[4], cent1, cent2, edge_normal, edgenorm_use[4], applied_displacement_edge,
		contig_disp[4];
	Vector2 vi_trap, ve_trap;
	Vector3 vi[4],ve[4];
	real ionflow_out, elecflow_out, Ndiff0, area;
	Tensor3 sigma__I, sigma_e_n_e, sigma_i_n_i, Matrix0,Matrix1,Matrix2,sigma_J_use;
	int par[3],rotated[4];
	//Vector2 edge_normal;
	int iii;
	Tensor3 sigma_J[MAXNEIGH];
	real n_i,n_e;
	Vector2 uvert[3];
	real beta[3];
	real betasum;
	real wt[4],wt_domain[4], wt_domain_sum;
	real Lapcontrib_self,Lapcontrib_inext,Lapcontrib_i,Lapcontrib_iprev;
	Vector2 gradcontrib_self, gradcontrib_inext,gradcontrib_i,gradcontrib_iprev;
	Vector3 nv_i_k,nv_e_k,nvi[4],nve[4];
	Vector2 nvi_trap,nve_trap;
	real effect_of_Ax_contig, effect_of_Ay_contig;

	static real const SIXTH = 1.0/6.0;
	static real const FOURPI = 4.0*PI;

	real rsq, r;

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->temp2.x = 1.0; // for k+1/2 --- ?
		pVertex->Ez_coeff_on_phi_anode = GetEzShape_(pVertex->pos.modulus());
		++pVertex;
	};

	this->Iz_prescribed = GetIzPrescribed(evaltime);
	
	real ReverseJzPerVertcell = -Iz_prescribed/(real)(numEndZCurrentRow-numStartZCurrentRow+1);
	
	// Now we have to think about this.
	// Solver wants to use Jz defined on vertices. We therefore do need numStartZCurrentRow.
	// It will not be in quite the right place.

	Epsilon_Iz_constant = -((qd_or_d)Iz_prescribed)*(qd_or_d)FOURPI_OVER_C; // eps_Iz = -Iz_presc/c + Iz_domain/c
	Epsilon_Iz_coeff_On_PhiAnode = 0.0; 
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Zero them first because for [IZ] we contribute to neighbour's coefficient.
	
		//memset(pVertex->coeff_self,0,NUM_EQNS_2*NUM_AFFECTORS_2*sizeof(real));
		//memset(pVertex->coeff,0,MAXNEIGH*NUM_EQNS_1*NUM_AFFECTORS_1*sizeof(real));
		pVertex->ZeroCoefficients();
		++pVertex;
	};
	
	real coeff_Aplus_for_Adot, coeff_Ak_for_Adot, coeff_Adot;
	real const ch = c*h;
	
#ifndef A_DOT_RETARDED

	// version where Adot means Adot_k
	// A-dot = (2.0 (A_k+1-A_k)/h) - Adot_k
	coeff_Aplus_for_Adot = 2.0;
	coeff_Ak_for_Adot = -2.0;
	coeff_Adot = -1.0;
#else
	// version where Adot means Adot_k-1/2
	// A-dot = (1.5 (A_k+1-A_k)/h) - 0.5 Adot_{k-1/2}
	coeff_Aplus_for_Adot = 1.5;
	coeff_Ak_for_Adot = -1.5;
	coeff_Adot = -0.5;
#endif
	if (bInitial) {
		coeff_Aplus_for_Adot = GlobalIzElasticity*h;
		coeff_Ak_for_Adot = 0.0;
		coeff_Adot = 0.0;
	};
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		
		// Quadratic assumes that the vertical charge accumulation rate
		// is homogeneous in z, which is of course, very inaccurate.
		// Neglecting dA/dz also which is unjustified.

		// ==========================================================

		tri_len = pVertex->GetTriIndexArray(izTri);
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		
		Vertex * pVertex0, *pVertex1,*pVertex2;
		// equal except at edges of domain.
		// . First make ConvexPolygon for this cell:
		// and set up sigma_J for each tri centroid.

				// ************************************************************************************************************
				// Centroid vs vertex position:
				// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				// It has to be all one way or the other.
				// The problem with putting all on vertex position is that if there is a long cell (abutting ins?)
				// with the vertex at one end, then using n,v,T and their derived quantities based on N/area,NT/N,
				// we are getting values distorted if we put them at the vertex and not the centroid of the vcentered cell.
			
				// So now I am going to write this assuming that n,v,T apply at the centroid.

				// ON THE OTHER HAND. When we do a vertex *move* we apply v at vertex. Ho hum.
				// We are going to compare species rel v "at the vertex" with _VERTEX_ displacement. Correct?
				// So to get rel displacement at tri centres.

				// If we want to do on centroids, we have to come in with a quantity saved that says "applied displacement of centroid"
				// How to get that though? MAYBE we have to average displacement from vertices, instead** <-- yes
				// => get relative displacement on tri centroid from separately, average displacement, average species velocity.
				// Don't ever know if insulator stuffs will work OK. Pretending v lives on vertex instead of centroid can only make
				// matters worse.
				
				// 99.9% of the time, centroid =~= vertex position. But not _ALL_ the time, if we are stretching away from ins.

				// ************************************************************************************************************
				// Use vertex centroids => fewer doubts.
				// But the already-applied displacement is not known at vertex centroid, only tri centroid, from actual vertex.
				// ************************************************************************************************************
		
		// New addition:

		if (iVertex == 11702) {
			iVertex= iVertex;
		};
		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			n_i = pVertex->Ion.mass/pVertex->AreaCell;
			n_e = pVertex->Elec.mass/pVertex->AreaCell;
			sigma__I = q*(pVertex->Ion.mass*pVertex->sigma_i - pVertex->Elec.mass*pVertex->sigma_e); 
			// do we understand that J needs to be integrated?
			// I = sigma__I E
			
			// 0b. Eps_Amp += 4pi/c J(E == 0)
			Vector3 defaultI = FOUR_PI_Q_OVER_C*(
								pVertex->Ion.mass*pVertex->v_i_0 - pVertex->Elec.mass*pVertex->v_e_0);
			
			pVertex->coeff_self[AMPZ][UNITY] += defaultI.z;
			Epsilon_Iz_constant += defaultI.z; // same as AMPZ
			// So eps_Iz is scaled by 4pi/c.
			
			// Now let's think about the effect on z charge flow:
			
			// Main component: difference of flows = 
			// h(4piq) n_e sigma_ve_zz (-grad_z phi) [top - bottom] AreaCell
			// because electrons leaving => + residual
			// divide all residuals by PLANEHEIGHT -->
			// MINUS h(4piq) Ne sigma_ve_zz (ddphi/dzdz)
			
			// ** We ignore ion dvz/dz for now **
		};

		// For Az we do not set == 0 in outermost cell but inside cathode wall.
		// Makes any sense? Not sure why.
		// Once cathode rod is added then Az == 0 is at the outer domain edge.
		// Have to anchor Az properly.

		cp.Clear();			
		/*if (pVertex->flags == OUTERMOST) {
			// before and after, if wanted:
			pTri = T + izTri[0];
			Vector2 centre = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
			Vector2 proj;
			centre.project_to_radius(&proj,pVertex->pos.modulus());
			cp.add(proj);
		}*/
			
		for (i = 0; i < tri_len; i++)
		{
			pTri = T + izTri[i];
			cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
			// Note for outermost/innermost it's triangle centres forming a triangle -- not a house.
		};

		for (i = 0; i < neigh_len; i++)
		{
			iprev = i-1; if (iprev < 0) iprev = neigh_len-1;
			inext = i+1; if (inext == neigh_len) inext = 0;
			pNeigh = X + izNeigh[i];
			pNeighPrev = X + izNeigh[iprev];
			pNeighNext = X + izNeigh[inext];

			// Now [VERIFY : ]
			// neighbour 0 is clockwise edge of tri 0; neighbour 1 is anticlockwise.
			// So looking towards neighbour 1, we have tri 0 and 1 either side.
			pTri1 = T + izTri[iprev];
			pTri2 = T + izTri[i];

			// Now get coeffs for grad over this edge.

			// If the edge is at the edge of domain,
			if (((pNeigh->flags == OUTERMOST) ||
				 (pNeigh->flags == INNERMOST)) &&
				 (pNeigh->flags == pVertex->flags))
			{
				// side to side on these: do nothing

				// Now consider carefully: otherwise, do we pick up the right
				// corners of cp otherwise? 
				// Yes, only if we have sorted the outermost cell's tri and neigh
				// index lists correctly.

			} else {
				// usual case:
				bAnodePrev = false;
				bAnode_i = false;
				bAnodeNext = false;
				if (pNeighPrev->pos.x*pNeighPrev->pos.x+pNeighPrev->pos.y*pNeighPrev->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnodePrev = true;
				if (pNeigh->pos.x*pNeigh->pos.x+pNeigh->pos.y*pNeigh->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnode_i = true;
				if (pNeighNext->pos.x*pNeighNext->pos.x+pNeighNext->pos.y*pNeighNext->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnodeNext = true;
								
				// Draw line between two triangle centroids:
				cent1 = cp.coord[iprev];//pTri1->GetContiguousCentroid__assuming_centroids_set(pVertex);
				cent2 = cp.coord[i];//pTri2->GetContiguousCentroid__assuming_centroids_set(pVertex);
				
				edge_normal.x = cent2.y-cent1.y;
				edge_normal.y = cent1.x-cent2.x; // to face outwards
				// all to be contiguous to own self...
				
				// Get contiguous quadrilateral positions, and displacements, using v centroid:			
				u[0] = pVertex->pos;
				u[1] = pNeighNext->pos;
				u[2] = pNeigh->pos;
				u[3] = pNeighPrev->pos;
				rotated[1] = 0; rotated[2] = 0; rotated[3] = 0;

				// Centroid vs position I guess... at least makes it hard to debug.
				
				// HMM. What to do about that?
				// Use position here?

				for (iii = 1; iii < 4; iii++)
					edgenorm_use[iii] = edge_normal;
				
				if ((pTri1->periodic != 0) || (pTri2->periodic != 0)) {
					if (u[0].x < 0.0) {
						if (u[1].x > 0.0) {
							u[1] = Anticlockwise*u[1];
							edgenorm_use[1] = Clockwise*edge_normal;
							rotated[1] = 1;
						};
						if (u[2].x > 0.0) {
							u[2] = Anticlockwise*u[2];
							edgenorm_use[2] = Clockwise*edge_normal;
							rotated[2] = 1;
						};
						if (u[3].x > 0.0) {
							u[3] = Anticlockwise*u[3];
							edgenorm_use[3] = Clockwise*edge_normal;
							rotated[3] = 1;
						};
					} else {
						if (u[1].x < 0.0) {
							u[1] = Clockwise*u[1];
							edgenorm_use[1] = Anticlockwise*edge_normal;
							rotated[1] = -1;
						};
						if (u[2].x < 0.0) {
							u[2] = Clockwise*u[2];
							edgenorm_use[2] = Anticlockwise*edge_normal;
							rotated[2] = -1;
						};
						if (u[3].x < 0.0) {
							u[3] = Clockwise*u[3];
							edgenorm_use[3] = Anticlockwise*edge_normal;
							rotated[3] = -1; // Leave this here:
							// Axy will need to be rotated.
						};
					};
				};
					
				// obtain (without regard for re-orienting)
				// Anticlockwise tri centroid: beta_2_self, beta_2_1, beta_2_2; // pTri2 is by point 1
				// Clockwise tri centroid: beta_1_self, beta_1_3, beta_1_2; // pTri1 is by point 3
				// This is the contiguous averaging from vertex centroids to tri centroids:
			
				shoelace = u[0].x*(u[1].y-u[3].y)
					     + u[1].x*(u[2].y-u[0].y)
						 + u[2].x*(u[3].y-u[1].y)
						 + u[3].x*(u[0].y-u[2].y);
				
				grad[0].x = (u[1].y-u[3].y)/shoelace; // the effect of 0 on grad.x
				grad[0].y = (u[3].x-u[1].x)/shoelace;
				// Anticlockwise:
				grad[1].x = (u[2].y-u[0].y)/shoelace;
				grad[1].y = (u[0].x-u[2].x)/shoelace;
				// Out:
				grad[2].x = (u[3].y-u[1].y)/shoelace;
				grad[2].y = (u[1].x-u[3].x)/shoelace;
				// Clockwise:
				grad[3].x = (u[0].y-u[2].y)/shoelace;
				grad[3].y = (u[2].x-u[0].x)/shoelace;

				Lapcontrib_self = grad[0].dot(edge_normal);
				Lapcontrib_inext = grad[1].dot(edge_normal);
				Lapcontrib_i = grad[2].dot(edge_normal);
				Lapcontrib_iprev = grad[3].dot(edge_normal);

				// We are not going to need to get grad.
				// We only need Lap Az + 4pi/c Jz = 0; Jz = sigma_zz (Ez_circuit - Adotz/c)

				if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
					
					// Project this edge to circle about 0 and take length:
					r = pVertex->centroid.modulus();
					Vector2 rhat = pVertex->centroid/r;
					Vector2 thetahat; thetahat.y = -rhat.x; thetahat.x = rhat.y;
					edgelen_proj = fabs(thetahat.dot(u[3]-u[1]));
					// This r,edgelen_proj should be stored.
				};

				// ___________________________________________________________________
				// 5. Eps_Amp = Lap A
				// ___________________________________________________________________
				
				pVertex->coeff_self[AMPZ][AZ] += Lapcontrib_self;
				pVertex->coeff[iprev].co[AMPZ][AZ] += Lapcontrib_iprev;
				pVertex->coeff[inext].co[AMPZ][AZ] += Lapcontrib_inext;
				pVertex->coeff[i].co[AMPZ][AZ] += Lapcontrib_i;
				
				if (pVertex->flags == OUTERMOST) {
					// Also need here to do Az looking down to 0 within cathode wall, just to anchor it:						
					// outward derivative = (-A)/(DOR-r);
					pVertex->coeff_self[AMPZ][AZ] += edgelen_proj*(-1.0/(DOMAIN_OUTER_RADIUS-r));	
				};
			
			};	// whether side-to-side on outermost/innermost vertex -- neglected apparently
		}; // next neighbour
		

		if (iVertex == 16000) {
			//FILE * file = fopen("16000coeffs.txt","w");
			//for (int iNeigh = 0 ; iNeigh < neigh_len; iNeigh++)
			//{
			//	fprintf(file,"%d %1.12E \n",
			//		izNeigh[iNeigh],pVertex->coeff[iNeigh].co[AMPZ][AZ]);

			//};
			//fprintf(file,"16000 %1.12E \n",pVertex->coeff_self[AMPZ][AZ]);
			store_coeff_self = pVertex->coeff_self[AMPZ][AZ];
			//fclose(file);
		}



		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			// ___________________________________________________________________
			// 7a. Eps_Amp += 4pi/c J(A self)
			// 7b. Eps_Amp += 4pi/c J(Ez = tunefac*shape)
			// ___________________________________________________________________

			// Ax --> Ex --> Jxyz --> epsAmp_xyz
			// epsAmpx <-- Jx <-- Exyz <-- Axyz [same dimensions]
			
			// How does this go?

			Vector3 effect3 = FOUR_PI_OVER_C*( sigma__I * (-coeff_Ak_for_Adot*pVertex->A/ch - coeff_Adot* pVertex->Adot/c));
			
			pVertex->coeff_self[AMPZ][AZ] += FOUR_PI_OVER_C*( sigma__I.zz * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPZ][UNITY] += effect3.z;
			pVertex->coeff_self[AMPZ][EZTUNING] += FOUR_PI_OVER_C * sigma__I.zz * pVertex->Ez_coeff_on_phi_anode;
			
			pVertex->coeff_self[IZ][AZ] += FOUR_PI_OVER_C*( sigma__I.zz * (-coeff_Aplus_for_Adot/ch)); // sames
			Epsilon_Iz_constant += effect3.z;
			Epsilon_Iz_coeff_On_PhiAnode += FOUR_PI_OVER_C* sigma__I.zz * pVertex->Ez_coeff_on_phi_anode;
			// Epsilon_Iz = 4pi/c(-IzPrescribed + Iz_domain)

			// _____________________________

			// Consider there's an alternative approach to get at A-dot if this way empirically fails
			// [ test it, or both, against (A_k+2-A_k)/2h ]
			// ie
			// instead of storing Adot estimated, store past Adot = (A_k-A_k-1)/h ; then take
			// Adot_k+1 = (A_k+1-A_k)/h + 0.5 (A_k+1-2A_k+A_k-1)/h^2
		};
		//
		//if (iVertex == 16000) {
		//	FILE * file = fopen("16000coeffs.txt","a");
		//	fprintf(file,"16000 with Jz effect %1.12E \n",pVertex->coeff_self[AMPZ][AZ]);
		//	fclose(file);
		//}

		// Reverse current:
		if ((iVertex >= numStartZCurrentRow) && (iVertex <= numEndZCurrentRow ))
		{
			pVertex->coeff_self[AMPZ][UNITY] += FOUR_PI_OVER_C*ReverseJzPerVertcell;
		}				
		memset(&(pVertex->epsilon),0,sizeof(real)*4); // debug: use epsilon to store d[sum eps Gauss]/d?.
		++pVertex;
	};
	
}

void TriMesh::CreateODECoefficients(bool bInitial)
{
	FILE * fp;
	// It is assumed that the A values found initially are A_k.

	// To find Laplacian we can take Lap = Div Grad ;
	// Div Grad f = sum of outward Grad f along sides ;
	// Grad f on side is found from quadrilateral of f values.

	// Remember to rotate Axy when a vertex is across PBC.

	// 2 ways:

	// 1. phi_neigh => E_cell => v_e_cell => rho here and in neighs
	// 2. phi_quad => E_edge => v_e_edge => rho here and in neighs

	// The 2nd way has the technical advantage that we do not depend on
	// cells that are further out.

	// It's a GOOD thing if phi difference across an edge drives charge across
	// the edge. It's not good to change phi here and affect flows out of neighs.

	// The other version is perhaps more ambitious in that we can get at a 
	// non-bwd electron displacement, more readily. But otherwise problematic.

	// ___

	// Cell->edge in general: 1/3(avg side neighs) + 2/3 (avg (in + out))
	// but not -- because we use vertex centroids not positions.

	// ___
	
	// End results of solve: v_e - (avg of v_i, v_n) ; advected n_e,n_i,n_n; A, phi.

	// Assume Ohm's Laws are already generated in the form
	// v_i = v_i_0 + sigma_i E, v_e = v_e_0 + sigma_e E .

	// Allow for shape of chEz field.

	long iVertex;
	Vertex * pVertex;
	real edgelen_proj;
	Vector2 extranormal, rhat, Exy_over_phi;
	bool bAnode, bAnodePrev, bAnode_i,bAnodeNext;
	real factor;
	Tensor3 sigma___;
	
	// PLAN:
	// In use, we load a set of coefficients into memory and apply them how needful ...
	// Grad neighbour i data;  eps += beta . neighdata

	// That is what we prefer, to saying, each equation at a time, with ITS set of all neighbour
	// coefficients, scurrying off to each neighbour in turn.

	long tri_len, neigh_len;
	long izTri[128];
	long izNeigh[128];
	Vertex * pNeigh, *pNeighPrev, *pNeighNext;
	real dist1, dist2;
	int iprev, inext;
	real Lapcoeff[MAXNEIGH];
	real Lapcoeff_self = 0.0;
	memset(Lapcoeff,0,sizeof(real)*MAXNEIGH);
	Vector2 u[4];
	ConvexPolygon cp;
	Triangle * pTri1,*pTri2, *pTri;
	int i;
	real beta_2_self, beta_2_1, beta_2_2, beta_1_self, beta_1_3, beta_1_2,
		beta_3_self, beta_3_2;
	real shoelace;
	Vector2 grad[4], cent1, cent2, edge_normal, edgenorm_use[4], applied_displacement_edge,
		contig_disp[4];
	Vector2 vi_trap, ve_trap;
	Vector3 vi[4],ve[4];
	real ionflow_out, elecflow_out, Ndiff0, area;
	Tensor3 sigma__I, sigma_e_n_e, sigma_i_n_i, Matrix0,Matrix1,Matrix2,sigma_J_use;
	int par[3],rotated[4];
	//Vector2 edge_normal;
	int iii;
	Tensor3 sigma_J[MAXNEIGH];
	real n_i,n_e;
	Vector2 uvert[3];
	real beta[3];
	real betasum;
	real wt[4],wt_domain[4], wt_domain_sum;
	real Lapcontrib_self,Lapcontrib_inext,Lapcontrib_i,Lapcontrib_iprev;
	Vector2 gradcontrib_self, gradcontrib_inext,gradcontrib_i,gradcontrib_iprev;
	Vector3 nv_i_k,nv_e_k,nvi[4],nve[4];
	Vector2 nvi_trap,nve_trap;
	real effect_of_Ax_contig, effect_of_Ay_contig;

	static real const SIXTH = 1.0/6.0;
	static real const FOURPI = 4.0*PI;

	real rsq, r;

	static int const iReport = 11383;

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->temp2.x = 1.0; // for k+1/2 --- ?
		pVertex->temp2.y = 1.0; // shape of Ez field vacuum solution.

		// scaling parameter multiplies this to give Ez
		rsq = pVertex->pos.y*pVertex->pos.y + pVertex->pos.x*pVertex->pos.x;
		if (rsq > 3.8*3.8) {
			r = sqrt(rsq);
			if (r < 4.6) {
				pVertex->temp2.y = 1.0-(r-3.8)/(4.6-3.8);
			} else {
				pVertex->temp2.y = 0.0;
			}
		}
		// This will be used in case bNeumannPhi = 1 ; otherwise
		// there needs to be some other description of both dphi/dz
		// above and dphi/dz below (or phi'').
		// =================================================================

		// Calculate linear formula for Ez and ddphi/dzdz based on 
		// quadratic interpolation of 3 phi points:

		r = pVertex->centroid.modulus();
		
		// . bottom phi = phi(z0) = phi0 :

		// from 2.7 to 3.44, 
		if (r < REVERSE_ZCURRENT_RADIUS) {
			// not relevant calcs within anode:
			
			pVertex->ddphidzdz_coeff_on_phi = 0.0;
			pVertex->ddphidzdz_coeff_on_phi_anode = 0.0;

			pVertex->Ez_coeff_on_phi = 0.0;
			pVertex->Ez_coeff_on_phi_anode = 0.0;

		} else {

			real ztop = ZTOP;
			real phitop_ppn_of_phi_anode = 1.0-(r-REVERSE_ZCURRENT_RADIUS)/
						(DOMAIN_OUTER_RADIUS - REVERSE_ZCURRENT_RADIUS); // linear
			real z0, phi0_ppn_of_phi_anode, z;


			if (r < DEVICE_RADIUS_INSULATOR_OUTER) 
			{
				z0 = EFFECTIVE_PLATEAU_HEIGHT; // can't see any point sloping upwards??

				// ah careful -- phi0 needs to feed through a coefficient on phi_anode
				phi0_ppn_of_phi_anode = 1.0-(r-REVERSE_ZCURRENT_RADIUS)
								/(0.5*(DEVICE_RADIUS_INSULATOR_OUTER+DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE)
											- REVERSE_ZCURRENT_RADIUS);
			} else {
				// we headed to phi_edge:
				real phi_edge = 1.0-(DEVICE_RADIUS_INSULATOR_OUTER-REVERSE_ZCURRENT_RADIUS)
								/(0.5*(DEVICE_RADIUS_INSULATOR_OUTER+DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE)
											- REVERSE_ZCURRENT_RADIUS); // ~= 0.1172
								
				real gradient_inside = 1.0/(0.5*(DEVICE_RADIUS_INSULATOR_OUTER+DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE)
											- REVERSE_ZCURRENT_RADIUS); // ~= 1.3793
				real maxsinr = DEVICE_RADIUS_INSULATOR_OUTER + PI*0.5/(gradient_inside/phi_edge); // ~= 3.565 cm

				if (r < maxsinr) {
					// smooth transition from linear slope to 0 slope via sine wave:
					phi0_ppn_of_phi_anode = phi_edge*(1.0-sin( (r-DEVICE_RADIUS_INSULATOR_OUTER)*
															(gradient_inside/phi_edge)	
														) );
				} else {
					phi0_ppn_of_phi_anode = 0.0;
				};

				// Logit for height of base:
				z0 = EFFECTIVE_PLATEAU_HEIGHT/(1.0 + exp(EFFECTIVE_LOGIT_COEFFICIENT*(r-EFFECTIVE_LOGIT_CENTRE)));

			};
			// Let's hope this works.
			// Generally we expect for 0 charge we are linearly between top and bottom.
			// But, that means due to our strange specification of what that implies,
			// that means we have set up places where d/dr of that is not zero.
			// Probably charge has to accumulate near there to cancel what phi'' would
			// exist.

			z = PLANE_Z;

			pVertex->ddphidzdz_coeff_on_phi = (2.0/(ztop-z))*(-1.0/(z-z0));
			pVertex->ddphidzdz_coeff_on_phi_anode = 
				phi0_ppn_of_phi_anode*
				(2.0/(ztop-z))*(-1.0/(ztop-z0)+1.0/(z-z0)) +
				phitop_ppn_of_phi_anode*
				(2.0/(ztop-z))*(1.0/(ztop-z0));

			real dphidz_coeff_on_phi = 
				((ztop-z)-(z-z0))/((z-z0)*(ztop-z));
			real dphidz_coeff_on_phitop = (z-z0)/((ztop-z)*(ztop-z0));
			real dphidz_coeff_on_phi_anode = 
				-phi0_ppn_of_phi_anode*dphidz_coeff_on_phi
				-phi0_ppn_of_phi_anode*dphidz_coeff_on_phitop
				+phitop_ppn_of_phi_anode*dphidz_coeff_on_phitop;
			// Ez = -dphi/dz so now need to negate:
			pVertex->Ez_coeff_on_phi = -dphidz_coeff_on_phi;
			pVertex->Ez_coeff_on_phi_anode = -dphidz_coeff_on_phi_anode;

			if (iVertex == iReport) {
				fp = fopen("report_Ez.txt","w");
				fprintf(fp,"phi'' coeff on phi %1.14E %1.14E \n"
					"Ez %1.14E %1.14E \n"
					"z0 z ztop %1.14E %1.14E %1.14E"
					"phi ppn %1.14E %1.14E\n",
					pVertex->ddphidzdz_coeff_on_phi,
					pVertex->ddphidzdz_coeff_on_phi_anode,
					pVertex->Ez_coeff_on_phi,
					pVertex->Ez_coeff_on_phi_anode,
					z0,z,ztop,
					phi0_ppn_of_phi_anode,phitop_ppn_of_phi_anode
					);
				fclose(fp);
			};
		};

		++pVertex;
	};
	// What about initial case? Will that now need phi? Apparently yes.
	// There is no longer a monolithic "set Ez" that can be applied to get
	// current.
	// But what chargeflow do we apply for initial ? Need to examine that.

	// Simple BCs:

	// phi at outer edge fits Gauss to avoid charge accumulation there;
	// phi = 0 on innermost cells to normalize; note that 3D would supply a solution here
	// and set phi within electrodes.

	// Az at inner edge can neglect inward look; this means it is constant inwards.
	// A at outer edge = 0 to normalize; otherwise there is a universal contribution to E inductive.
	
	// Axy at inner edge ought to be decreasing towards zero radially.
	// this is the bit to consider.
	// Just set the innermost ones to be radially declined from the average of
	// the two neighbours outside.

	this->Iz_prescribed = GetIzPrescribed(evaltime);
	
	real ReverseJzPerVertcell = -Iz_prescribed/(real)(numEndZCurrentRow-numStartZCurrentRow+1);

	Epsilon_Iz_constant = -((qd_or_d)Iz_prescribed)*(qd_or_d)FOURPI_OVER_C; // eps_Iz = -Iz_presc/c + Iz_domain/c
	Epsilon_Iz_coeff_On_PhiAnode = 0.0; 

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Zero them first because for [IZ] we contribute to neighbour's coefficient.

		//memset(pVertex->coeff_self,0,NUM_EQNS_2*NUM_AFFECTORS_2*sizeof(real));
		//memset(pVertex->coeff,0,MAXNEIGH*NUM_EQNS_1*NUM_AFFECTORS_1*sizeof(real));
		pVertex->ZeroCoefficients();
		++pVertex;
	};

	real coeff_Aplus_for_Adot, coeff_Ak_for_Adot, coeff_Adot;
	real const ch = c*h;

#ifndef A_DOT_RETARDED

	// version where Adot means Adot_k
	// A-dot = (2.0 (A_k+1-A_k)/h) - Adot_k
	coeff_Aplus_for_Adot = 2.0;
	coeff_Ak_for_Adot = -2.0;
	coeff_Adot = -1.0;
#else
	// version where Adot means Adot_k-1/2
	// A-dot = (1.5 (A_k+1-A_k)/h) - 0.5 Adot_{k-1/2}
	coeff_Aplus_for_Adot = 1.5;
	coeff_Ak_for_Adot = -1.5;
	coeff_Adot = -0.5;
#endif
	if (bInitial) {
		coeff_Aplus_for_Adot = GlobalIzElasticity*h;
		coeff_Ak_for_Adot = 0.0;
		coeff_Adot = 0.0;
	};
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		
		// Quadratic assumes that the vertical charge accumulation rate
		// is homogeneous in z, which is of course, very inaccurate.
		// Neglecting dA/dz also which is unjustified.

		// ==========================================================

		tri_len = pVertex->GetTriIndexArray(izTri);
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		
		Vertex * pVertex0, *pVertex1,*pVertex2;
		// equal except at edges of domain.
		// . First make ConvexPolygon for this cell:
		// and set up sigma_J for each tri centroid.

				// ************************************************************************************************************
				// Centroid vs vertex position:
				// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				// It has to be all one way or the other.
				// The problem with putting all on vertex position is that if there is a long cell (abutting ins?)
				// with the vertex at one end, then using n,v,T and their derived quantities based on N/area,NT/N,
				// we are getting values distorted if we put them at the vertex and not the centroid of the vcentered cell.
			
				// So now I am going to write this assuming that n,v,T apply at the centroid.

				// ON THE OTHER HAND. When we do a vertex *move* we apply v at vertex. Ho hum.
				// We are going to compare species rel v "at the vertex" with _VERTEX_ displacement. Correct?
				// So to get rel displacement at tri centres.

				// If we want to do on centroids, we have to come in with a quantity saved that says "applied displacement of centroid"
				// How to get that though? MAYBE we have to average displacement from vertices, instead** <-- yes
				// => get relative displacement on tri centroid from separately, average displacement, average species velocity.
				// Don't ever know if insulator stuffs will work OK. Pretending v lives on vertex instead of centroid can only make
				// matters worse.
				
				// 99.9% of the time, centroid =~= vertex position. But not _ALL_ the time, if we are stretching away from ins.

				// ************************************************************************************************************
				// Use vertex centroids => fewer doubts.
				// But the already-applied displacement is not known at vertex centroid, only tri centroid, from actual vertex.
				// ************************************************************************************************************
		
		// New addition:

		pVertex->coeff_self[GAUSS][PHI] += pVertex->ddphidzdz_coeff_on_phi*pVertex->AreaCell;
		// CAREFUL - What is area that applies here? cp.area?
		pVertex->coeff_self[GAUSS][PHI_ANODE] += pVertex->ddphidzdz_coeff_on_phi_anode*pVertex->AreaCell;
		// ditto
		if (iVertex == iReport) {
			fp = fopen("report.txt","w");
			fprintf(fp,"VERTEX %d \n"
						"ddphidzdz: [GAUSS][PHI] %1.14E [GAUSS][PHI_ANODE] %1.14E \n"
						"pVertex->ddphidzdz_coeff_on_phi %1.14E %1.14E pVertex->AreaCell %1.14E \n",
						iVertex,
						pVertex->ddphidzdz_coeff_on_phi*pVertex->AreaCell,
						pVertex->ddphidzdz_coeff_on_phi_anode*pVertex->AreaCell,
						pVertex->ddphidzdz_coeff_on_phi,pVertex->ddphidzdz_coeff_on_phi_anode,
						pVertex->AreaCell);
		};

		// Where is corresponding charge flow?
		// **



		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			n_i = pVertex->Ion.mass/pVertex->AreaCell;
			n_e = pVertex->Elec.mass/pVertex->AreaCell;
			sigma__I = q*(pVertex->Ion.mass*pVertex->sigma_i - pVertex->Elec.mass*pVertex->sigma_e); 
			// do we understand that J needs to be integrated?
			// I = sigma__I E
			
			Ndiff0 = pVertex->Ion.mass - pVertex->Elec.mass;
			pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*Ndiff0;
			
			// 0b. Eps_Amp += 4pi/c J(E == 0)
			Vector3 defaultI = FOUR_PI_Q_OVER_C*(
								pVertex->Ion.mass*pVertex->v_i_0 - pVertex->Elec.mass*pVertex->v_e_0);
			
			pVertex->coeff_self[AMPX][UNITY] += defaultI.x;
			pVertex->coeff_self[AMPY][UNITY] += defaultI.y;
			pVertex->coeff_self[AMPZ][UNITY] += defaultI.z;
			Epsilon_Iz_constant += defaultI.z; // same as AMPZ
			
			// Now let's think about the effect on z charge flow:
			
			// Main component: difference of flows = 
			// h(4piq) n_e sigma_ve_zz (-grad_z phi) [top - bottom] AreaCell
			// because electrons leaving => + residual
			// divide all residuals by PLANEHEIGHT -->
			// MINUS h(4piq) Ne sigma_ve_zz (ddphi/dzdz)
			
			// ** We ignore ion dvz/dz for now **
			
			pVertex->coeff_self[GAUSS][PHI] += - pVertex->ddphidzdz_coeff_on_phi*
				h*FOUR_PI_Q*pVertex->Elec.mass*pVertex->sigma_e.zz;
			pVertex->coeff_self[GAUSS][PHI_ANODE] += - pVertex->ddphidzdz_coeff_on_phi_anode*
				h*FOUR_PI_Q*pVertex->Elec.mass*pVertex->sigma_e.zz;
			
			if (iVertex == iReport) {
				fprintf(fp, "chargeflow_z [GAUSS][PHI] %1.14E [GAUSS][PHI_ANODE] %1.14E \n"
							"pVertex->Elec.mass %1.14E pVertex->sigma_e.zz %1.14E h4piq %1.14E \n",
							- pVertex->ddphidzdz_coeff_on_phi*
							h*FOUR_PI_Q*pVertex->Elec.mass*pVertex->sigma_e.zz,
							 - pVertex->ddphidzdz_coeff_on_phi_anode*
								h*FOUR_PI_Q*pVertex->Elec.mass*pVertex->sigma_e.zz,
								pVertex->Elec.mass, pVertex->sigma_e.zz, h*FOUR_PI_Q);
			}
		};
		
		bAnode = 0;
		if ((pVertex->flags == INNERMOST) || 
			(pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS))
		{
			pVertex->coeff_self[GAUSS][PHI] = -1.0; // set phi = phi_anode here.
			pVertex->coeff_self[GAUSS][PHI_ANODE] = 1.0; // phi-phi_anode = 0
			bAnode = true;
		}

		// For Az we do not set == 0 in outermost cell but inside cathode wall.
		// Makes any sense? Not sure why.
		// Once cathode rod is added then Az == 0 is at the outer domain edge.
		// Have to anchor Az properly.

		cp.Clear();			
		/*if (pVertex->flags == OUTERMOST) {
			// before and after, if wanted:
			pTri = T + izTri[0];
			Vector2 centre = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
			Vector2 proj;
			centre.project_to_radius(&proj,pVertex->pos.modulus());
			cp.add(proj);
		}*/
			
		for (i = 0; i < tri_len; i++)
		{
			pTri = T + izTri[i];
			cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
			// Note for outermost/innermost it's triangle centres forming a triangle -- not a house.
		};

		for (i = 0; i < neigh_len; i++)
		{
			iprev = i-1; if (iprev < 0) iprev = neigh_len-1;
			inext = i+1; if (inext == neigh_len) inext = 0;
			pNeigh = X + izNeigh[i];
			pNeighPrev = X + izNeigh[iprev];
			pNeighNext = X + izNeigh[inext];

			// Now [VERIFY : ]
			// neighbour 0 is clockwise edge of tri 0; neighbour 1 is anticlockwise.
			// So looking towards neighbour 1, we have tri 0 and 1 either side.
			pTri1 = T + izTri[iprev];
			pTri2 = T + izTri[i];

			// Now get coeffs for grad over this edge.

			// If the edge is at the edge of domain,
			if (((pNeigh->flags == OUTERMOST) ||
				 (pNeigh->flags == INNERMOST)) &&
				 (pNeigh->flags == pVertex->flags))
			{
				// side to side on these: do nothing

				// Now consider carefully: otherwise, do we pick up the right
				// corners of cp otherwise? 
				// Yes, only if we have sorted the outermost cell's tri and neigh
				// index lists correctly.

			} else {
				// usual case:
				bAnodePrev = false;
				bAnode_i = false;
				bAnodeNext = false;
				if (pNeighPrev->pos.x*pNeighPrev->pos.x+pNeighPrev->pos.y*pNeighPrev->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnodePrev = true;
				if (pNeigh->pos.x*pNeigh->pos.x+pNeigh->pos.y*pNeigh->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnode_i = true;
				if (pNeighNext->pos.x*pNeighNext->pos.x+pNeighNext->pos.y*pNeighNext->pos.y < REVERSE_ZCURRENT_RADIUS*REVERSE_ZCURRENT_RADIUS)
					bAnodeNext = true;
								
				// Draw line between two triangle centroids:
				cent1 = cp.coord[iprev];//pTri1->GetContiguousCentroid__assuming_centroids_set(pVertex);
				cent2 = cp.coord[i];//pTri2->GetContiguousCentroid__assuming_centroids_set(pVertex);
				
				edge_normal.x = cent2.y-cent1.y;
				edge_normal.y = cent1.x-cent2.x; // to face outwards
				// all to be contiguous to own self...
				
				// Get contiguous quadrilateral positions, and displacements, using v centroid:			
				u[0] = pVertex->centroid;
				u[1] = pNeighNext->centroid;
				u[2] = pNeigh->centroid;
				u[3] = pNeighPrev->centroid;
				rotated[1] = 0; rotated[2] = 0; rotated[3] = 0;

				contig_disp[0] = pVertex->ApplDisp;
				contig_disp[1] = pNeighNext->ApplDisp;
				contig_disp[2] = pNeigh->ApplDisp;
				contig_disp[3] = pNeighPrev->ApplDisp;

				for (iii = 1; iii < 4; iii++)
					edgenorm_use[iii] = edge_normal;
				
				if ((pTri1->periodic != 0) || (pTri2->periodic != 0)) {
					if (u[0].x < 0.0) {
						if (u[1].x > 0.0) {
							u[1] = Anticlockwise*u[1];
							contig_disp[1] = Anticlockwise*contig_disp[1];
							edgenorm_use[1] = Clockwise*edge_normal;
							rotated[1] = 1;
						};
						if (u[2].x > 0.0) {
							u[2] = Anticlockwise*u[2];
							contig_disp[2] = Anticlockwise*contig_disp[2];
							edgenorm_use[2] = Clockwise*edge_normal;
							rotated[2] = 1;
						};
						if (u[3].x > 0.0) {
							u[3] = Anticlockwise*u[3];
							contig_disp[3] = Anticlockwise*contig_disp[3];
							edgenorm_use[3] = Clockwise*edge_normal;
							rotated[3] = 1;
						};
					} else {
						if (u[1].x < 0.0) {
							u[1] = Clockwise*u[1];
							contig_disp[1] = Clockwise*contig_disp[1];
							edgenorm_use[1] = Anticlockwise*edge_normal;
							rotated[1] = -1;
						};
						if (u[2].x < 0.0) {
							u[2] = Clockwise*u[2];
							contig_disp[2] = Clockwise*contig_disp[2];
							edgenorm_use[2] = Anticlockwise*edge_normal;
							rotated[2] = -1;
						};
						if (u[3].x < 0.0) {
							u[3] = Clockwise*u[3];
							contig_disp[3] = Clockwise*contig_disp[3];
							edgenorm_use[3] = Anticlockwise*edge_normal;
							rotated[3] = -1; // Leave this here:
							// Axy will need to be rotated.
						};
					};
				};
									

				if ((iEdgeAvg_for_Gauss == OPPOSING_BETA) ||
					(iEdgeAvg_for_Gauss == CORNERS_BETA) ||
					(iEdgeAvg_for_Gauss == SIMPSON_BETA) )
				{
					
					GetInterpolationCoefficients(beta,cent2.x,cent2.y,u[0],u[1],u[2]);
					beta_2_self = beta[0];
					beta_2_1 = beta[1];
					beta_2_2 = beta[2];

					GetInterpolationCoefficients(beta,cent1.x,cent1.y,u[0],u[3],u[2]);
					beta_1_self = beta[0];
					beta_1_3 = beta[1];
					beta_1_2 = beta[2];
					
					Vector2 centre_edge = 0.5*(cent1+cent2);
					dist1 = (u[0]-centre_edge).modulus();
					dist2 = (u[2]-centre_edge).modulus();

					beta_3_self = dist2/(dist1+dist2);
					beta_3_2 = 1.0-beta_3_self;

					if (iEdgeAvg_for_Gauss == CORNERS_BETA) {
						wt[0] = 0.5*(beta_1_self+beta_2_self);
						wt[1] = 0.5*beta_2_1; // corresp pNeighNext
						wt[2] = 0.5*(beta_1_2+beta_2_2);
						wt[3] = 0.5*beta_1_3; // corresp pNeighPrev
					};
					if (iEdgeAvg_for_Gauss == SIMPSON_BETA) {
						wt[0] = 0.25*(beta_1_self+beta_2_self)+0.5*beta_3_self;
						wt[1] = 0.25*beta_2_1; // corresp pNeighNext
						wt[2] = 0.25*(beta_1_2+beta_2_2)+0.5*beta_3_2;
						wt[3] = 0.25*beta_1_3; // corresp pNeighPrev
					};
					if (iEdgeAvg_for_Gauss == OPPOSING_BETA) {
						wt[0] = beta_3_self;
						wt[1] = 0.0;
						wt[2] = beta_3_2;
						wt[3] = 0.0;
					};
				} else {
					if (iEdgeAvg_for_Gauss == FORCE_FIFTY_FIFTY) {
						wt[0] = 0.5;
						wt[1] = 0.0;
						wt[2] = 0.5;
						wt[3] = 0.0;
					};
					if (iEdgeAvg_for_Gauss == FORCE_TWELTHS) {
						wt[0] = 5.0/12.0;
						wt[1] = 1.0/12.0;
						wt[2] = 5.0/12.0;
						wt[3] = 1.0/12.0;
					};
				};
				wt_domain[0] = wt[0];
				wt_domain[1] = wt[1];
				wt_domain[2] = wt[2];
				wt_domain[3] = wt[3];
				
				// handle insulator:
				if (pVertex->flags == DOMAIN_VERTEX) {
					if (pNeighPrev->flags == INNER_VERTEX) wt_domain[3] = 0.0;
					if (pNeighNext->flags == INNER_VERTEX) wt_domain[1] = 0.0;
					if (pNeigh->flags == INNER_VERTEX) wt_domain[2] = 0.0;
					wt_domain_sum = wt_domain[0]+wt_domain[1]+wt_domain[2]+wt_domain[3];
					wt_domain[0] /= wt_domain_sum;
					wt_domain[1] /= wt_domain_sum;
					wt_domain[2] /= wt_domain_sum;
					wt_domain[3] /= wt_domain_sum;
					// Total is meant to be 1, right?
				};

				applied_displacement_edge = wt[0]*contig_disp[0] + wt[2]*contig_disp[2] 
											+ wt[1]*contig_disp[1] + wt[3]*contig_disp[3];
				// Am I assuming a typical interior edge here?
				// ###########################################
				// Presumably ApplDisp was set to zero for inner vertex?
			

				// obtain (without regard for re-orienting)
				// Anticlockwise tri centroid: beta_2_self, beta_2_1, beta_2_2; // pTri2 is by point 1
				// Clockwise tri centroid: beta_1_self, beta_1_3, beta_1_2; // pTri1 is by point 3
				// This is the contiguous averaging from vertex centroids to tri centroids:
			
			// if phi is to be set on electrode surface,
			// [that's OK because we DO have a charge flow from the electrode]
			// we just have a set of vertices _just within_ , where we set phi. That should do it: phi
			// difference at edge of cell (electrode surface) is considered to generate inflow.
			
			// What if the same applies: set phi within ins, to a fixed value.
			// we can then have multiple solutions? NO - think it's okay.
			
			// What did we do about setting A overall change? = 0 at outer edge? Why not inner?
			// Inner *should* be about equal... but outer represents rest of world, so constant.
			// Same might reasonably apply to phi then ....
			// Why not? we get small charge accumulation at the back? phi is set there to eject
			// charge...
			
				shoelace = u[0].x*(u[1].y-u[3].y)
					     + u[1].x*(u[2].y-u[0].y)
						 + u[2].x*(u[3].y-u[1].y)
						 + u[3].x*(u[0].y-u[2].y);
				
				grad[0].x = (u[1].y-u[3].y)/shoelace; // the effect of 0 on grad.x
				grad[0].y = (u[3].x-u[1].x)/shoelace;
				// Anticlockwise:
				grad[1].x = (u[2].y-u[0].y)/shoelace;
				grad[1].y = (u[0].x-u[2].x)/shoelace;
				// Out:
				grad[2].x = (u[3].y-u[1].y)/shoelace;
				grad[2].y = (u[1].x-u[3].x)/shoelace;
				// Clockwise:
				grad[3].x = (u[0].y-u[2].y)/shoelace;
				grad[3].y = (u[2].x-u[0].x)/shoelace;

				Lapcontrib_self = grad[0].dot(edge_normal);
				Lapcontrib_inext = grad[1].dot(edge_normal);
				Lapcontrib_i = grad[2].dot(edge_normal);
				Lapcontrib_iprev = grad[3].dot(edge_normal);

				// Now how do we assess grad phi on the vertcell?
				// Bear in mind we want to deal with the set of tri centroids and
				// probably averaging on the tri centroids.
				
				// We are going to average phi along the edge and multiply
				// by (coord[i].y-coord[iprev].y) to get dT/dx
				// bearing in mind we are always going anticlockwise around cp.

				//edge_normal.x = cent2.y-cent1.y;
				//edge_normal.y = cent1.x-cent2.x; // to face outwards

				// Only used in one place: getting grad phi, to affect Ampere.
				// We want integral of grad phi, to give sigma_I grad_phi = integrated current, correct?
				gradcontrib_self = wt[0]*edge_normal/pVertex->AreaCell;
				gradcontrib_inext = wt[1]*edge_normal/pVertex->AreaCell;
				gradcontrib_i = wt[2]*edge_normal/pVertex->AreaCell;
				gradcontrib_iprev = wt[3]*edge_normal/pVertex->AreaCell;

				if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
					
					// Project this edge to circle about 0 and take length:
					r = pVertex->centroid.modulus();
					Vector2 rhat = pVertex->centroid/r;
					Vector2 thetahat; thetahat.y = -rhat.x; thetahat.x = rhat.y;
					edgelen_proj = fabs(thetahat.dot(u[3]-u[1]));
					// This r,edgelen_proj should be stored.
					if (pVertex->flags == INNERMOST) {
						extranormal = -edgelen_proj*rhat;
					} else {
						extranormal = edgelen_proj*rhat;
					};
				};						

				// 7 EFFECTS:
				// ==========

				// Gauss: -Div E + 4 pi rho = 0

				// 0a. Eps_Gauss += 4 pi rho0 (n_i,n_e existing if v==0)
				// 0b. Eps_Amp += 4pi/c I(E == 0) ; Eps_Iz += 4pi/c Iz also.

				// 1. Eps_Gauss += Lap phi
				// 2. Eps_Gauss += Div [dA/dt]/c

				// 3. Eps_Gauss += 4 pi rho[v nearby[A nearby, Ez_ext]]
				// 4. Eps_Gauss += 4 pi rho [phi nearby -> E_edge -> J_edge*h]

				// Ampere: Lap A + 4pi/c J = 0

				// 5. Eps_Amp = +Lap A
				// 6. Eps_Amp += 4pi/c J(- grad phi) ;    Eps_Iz += 4pi/c Jz also
				// 7. Eps_Amp += 4pi/c J(A self,Ez_ext) ; Eps_Iz += 4pi/c Jz also
				
				// _____________________________________________________________________________________________
				// 1. Eps_Gauss += Lap phi
				// ________________________________________
				// This gives integral div grad == integral Lap.

				//	For every vertex except those within electrodes:
						
				if (bAnode == false) {
					// try this: if looking inside anode, replace with [PHI_ANODE]
					
					pVertex->coeff_self[GAUSS][PHI] += Lapcontrib_self;					
					if (bAnodePrev == false) {
						pVertex->coeff[iprev].co[GAUSS][PHI] += Lapcontrib_iprev;
					} else {
						pVertex->coeff_self[GAUSS][PHI_ANODE] += Lapcontrib_iprev;
					};
					if (bAnode_i == false) {
						pVertex->coeff[i].co[GAUSS][PHI] += Lapcontrib_i;
					} else {
						pVertex->coeff_self[GAUSS][PHI_ANODE] += Lapcontrib_i;
					};
					if (bAnodeNext == false) {
						pVertex->coeff[inext].co[GAUSS][PHI] += Lapcontrib_inext;
					} else {
						pVertex->coeff_self[GAUSS][PHI_ANODE] += Lapcontrib_inext;
					};
					
					// Makes sense even near the outer edge? We ruled out side-to-side on outermost & innermost.
					// phi = 0 within cathode and our outermost is looking into cathode.
					if (pVertex->flags == OUTERMOST) {
						// -phi/dist is outward radial gradient of phi
						// +Lap phi comes from grad phi dot edgenormal
						pVertex->coeff_self[GAUSS][PHI] += (-1.0/(DOMAIN_OUTER_RADIUS-r))*edgelen_proj;
					}

			if (iVertex == iReport) {
				fprintf(fp, "Lap phi vs neigh %d : [GAUSS][PHI] %1.14E \n"
							"iprev %1.14E edge_normal %1.14E %1.14E grad[0] %1.14E %1.14E \n",
							izNeigh[i], Lapcontrib_self, 
							Lapcontrib_iprev, edge_normal.x,edge_normal.y, grad[0].x,grad[0].y);
			}
					// _____________________________________________________________________________________________
					// 2. Eps_Gauss += Div [dA/dt]/ch
					// ________________________________________

					// div Adot = sum (Adot_edge dot edge_normal_outward)
					
					// divcoeff_self is _not necessarily_ net zero because we do not
					// use simple average at tri centroid.
					
					pVertex->coeff[iprev].co[GAUSS][AX] += coeff_Aplus_for_Adot * wt[3] * edgenorm_use[3].x/ch; 
					pVertex->coeff[iprev].co[GAUSS][AY] += coeff_Aplus_for_Adot * wt[3] * edgenorm_use[3].y/ch; 
					pVertex->coeff_self[GAUSS][UNITY] += wt[3]*
											(coeff_Ak_for_Adot*pNeighPrev->A.dotxy(edgenorm_use[3])/ch
											 +coeff_Adot*pNeighPrev->Adot.dotxy(edgenorm_use[3])/c);
					// because we add + the divergence of A-dot.
					
					pVertex->coeff[inext].co[GAUSS][AX] += coeff_Aplus_for_Adot *wt[1]* edgenorm_use[1].x/ch; // 3/6 = 1/2
					pVertex->coeff[inext].co[GAUSS][AY] += coeff_Aplus_for_Adot *wt[1]* edgenorm_use[1].y/ch;
					pVertex->coeff_self[GAUSS][UNITY] += wt[1]*(
											(coeff_Ak_for_Adot*pNeighNext->A.dotxy(edgenorm_use[1]))/ch
											+coeff_Adot*pNeighNext->Adot.dotxy(edgenorm_use[1])/c); 
					
					pVertex->coeff[i].co[GAUSS][AX] +=  coeff_Aplus_for_Adot*wt[2]*edgenorm_use[2].x/ch; 
					pVertex->coeff[i].co[GAUSS][AY] +=  coeff_Aplus_for_Adot*wt[2]*edgenorm_use[2].y/ch;
					pVertex->coeff_self[GAUSS][UNITY] += wt[2]*
											(coeff_Ak_for_Adot*pNeigh->A.dotxy(edgenorm_use[2])/ch
											+coeff_Adot*pNeigh->Adot.dotxy(edgenorm_use[2])/c);
					
					// Include self-contribution just in case this does not net to zero:
					// Not sure if it should or not, since not simple averaging at tris.
					pVertex->coeff_self[GAUSS][AX] += coeff_Aplus_for_Adot*wt[0]*edge_normal.x/ch;
					pVertex->coeff_self[GAUSS][AY] +=  coeff_Aplus_for_Adot*wt[0]*edge_normal.y/ch;
					pVertex->coeff_self[GAUSS][UNITY] += wt[0]*
											(coeff_Ak_for_Adot*pVertex->A.dotxy(edge_normal)/ch
											+coeff_Adot*pVertex->Adot.dotxy(edge_normal)/c);
						
			if (iVertex == iReport) {
				fprintf(fp, "coeff on neigh %d for Div_A-dot: [GAUSS][AX] %1.14E \n",
							coeff_Aplus_for_Adot*wt[0]*edge_normal.x/ch);
			}
					if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
					
						// Assume Adot declines inward just like A.
						
						if (pVertex->flags == INNERMOST) {
							factor = INNER_A_BOUNDARY/r;
						} else {
							factor = r/DOMAIN_OUTER_RADIUS;
						};
							
						pVertex->coeff_self[GAUSS][AX] += 
							factor*coeff_Aplus_for_Adot*extranormal.x/ch;
						pVertex->coeff_self[GAUSS][AY] += 
							factor*coeff_Aplus_for_Adot*extranormal.y/ch;
						pVertex->coeff_self[GAUSS][UNITY] += factor*
							(coeff_Ak_for_Adot*pVertex->A.dotxy(extranormal)/ch
							+coeff_Adot*pVertex->Adot.dotxy(extranormal)/c);
					};// whether innermost || outermost					

				}; // bAnode == false

				// The rest of epsGauss only applies for a domain vertex not looking into insulator.
				// However it also needs to apply at the outer edge because for now, the outer edge is
				// a cathode wall.

				if (((pVertex->flags == DOMAIN_VERTEX)||(pVertex->flags==OUTERMOST))
					&&
					((pTri1->u8domain_flag == DOMAIN_TRIANGLE) || (pTri2->u8domain_flag == DOMAIN_TRIANGLE)))
				{
					// _____________________________________________________________________________________________
					
					// 3.a. Eps_Gauss += 4 pi rho (v0)
					// _____________________________
		
					// Now we have to decide how much leaves ordinarily:
					
					// For ions we do not need it to be bwd.
					// . Go above and get applied displacement on tri centroid.
					// contiguous with this vertcell.
					
					// So this is about averaging from centroid to tri cent and comparing.
					// Amount of normal flow = (h v - appl displ)_avg_along_edge dot edge_normal
					memset(&nvi_trap,0,sizeof(nvi_trap));
					memset(&nve_trap,0,sizeof(nve_trap));

					nv_i_k = pVertex->Ion.mom/pVertex->AreaCell;
					nv_e_k = pVertex->Elec.mom/pVertex->AreaCell;// not used...
					nvi[0] = 0.5*(nv_i_k + (pVertex->Ion.mass/pVertex->AreaCell)*pVertex->v_i_0); // assume have v_i_k+1 = v_i_0 + sigma E_k+1
					nve[0] = //0.5*(nv_e_k + 
							(pVertex->Elec.mass/pVertex->AreaCell)*pVertex->v_e_0;
				
					nvi_trap += wt_domain[0]*nvi[0].xypart();
					nve_trap += wt_domain[0]*nve[0].xypart();

					// Stick with working out eqns for v anyway.

					// We need n_i and n_e in here. We would like to use the values
					// averaged at each tri centroid? It makes no big difference: the bottom line is
					// that high n can have a high v and pull out all the mass in the low n cell.
					// The problem case is always that the high n cell is heading away.
					// Assume the downwind cell is a large low-density cell.
					// which is not expanding ... 
					// The trouble with upwind minmod is that it's not linear for these equations.

					// WOULD GREATLY PREFER UPWIND MINMOD.
					// ... but stick with this method just for now...

					// Preferring trapezoidal ion displacement.
					// Having different treatment of i and e, is something that prevents
					// us storing one sigma_J tensor.

					// assuming DOMAIN VERTEX already:
					if ((pNeighPrev->flags == DOMAIN_VERTEX) || (pNeighPrev->flags == OUTERMOST)) {
						nv_i_k = pNeighPrev->Ion.mom/pNeighPrev->AreaCell; 
						nv_e_k = pNeighPrev->Elec.mom/pNeighPrev->AreaCell; 
						nvi[3] = 0.5*(nv_i_k + (pNeighPrev->Ion.mass/pNeighPrev->AreaCell)*pNeighPrev->v_i_0);
						nve[3] = (pNeighPrev->Elec.mass/pNeighPrev->AreaCell)*pNeighPrev->v_e_0;
						if (rotated[3] == 0) {
						} else {
							if (rotated[3] == 1) {
								// need to rotate v anticlockwise;  
								nvi[3] = Anticlockwise3*nvi[3];
								nve[3] = Anticlockwise3*nve[3];
							} else {
								nvi[3] = Clockwise3*nvi[3];
								nve[3] = Clockwise3*nve[3];
							};
						};
						nvi_trap += wt_domain[3]*nvi[3].xypart();
						nve_trap += wt_domain[3]*nve[3].xypart();
					};

					if ((pNeighNext->flags == DOMAIN_VERTEX) || (pNeighNext->flags == OUTERMOST)) {
						nv_i_k = pNeighNext->Ion.mom/pNeighNext->AreaCell;
						nv_e_k = pNeighNext->Elec.mom/pNeighNext->AreaCell;
						nvi[1] = 0.5*(nv_i_k + (pNeighNext->Ion.mass/pNeighNext->AreaCell)*pNeighNext->v_i_0);
						nve[1] = (pNeighNext->Elec.mass/pNeighNext->AreaCell)*pNeighNext->v_e_0;
						if (rotated[1] == 0) {
						} else {
							if (rotated[1] == 1) {
								nvi[1] = Anticlockwise3*nvi[1];
								nve[1] = Anticlockwise3*nve[1];
							} else {
								nvi[1] = Clockwise3*nvi[1];
								nve[1] = Clockwise3*nve[1];
							};
						};
						nvi_trap += wt_domain[1]*nvi[1].xypart();
						nve_trap += wt_domain[1]*nve[1].xypart();
					};

					if ((pNeigh->flags == DOMAIN_VERTEX) || (pNeigh->flags == OUTERMOST)) {
						nv_i_k = pNeigh->Ion.mom/pNeigh->AreaCell;
						nv_e_k = pNeigh->Elec.mom/pNeigh->AreaCell;
						nvi[2] = 0.5*(nv_i_k + (pNeigh->Ion.mass/pNeigh->AreaCell)*pNeigh->v_i_0);
						nve[2] = (pNeigh->Elec.mass/pNeigh->AreaCell)*pNeigh->v_e_0;
						if (rotated[2] == 0) {
						} else {
							if (rotated[2] == 1) {
								nvi[2] = Anticlockwise3*nvi[2];
								nve[2] = Anticlockwise3*nve[2];
							} else {
								nvi[2] = Clockwise3*nvi[2];
								nve[2] = Clockwise3*nve[2];
							};
						};
						// Made vi[],ve[] all contiguous to self . 
						
						nvi_trap += wt_domain[2]*nvi[2].xypart();
						nve_trap += wt_domain[2]*nve[2].xypart();
					};

					// NOTE: you can't add 0*undefined and hope it has no effect.
					//nvi_trap = (wt_domain[0]*nvi[0] + wt_domain[1]*nvi[1] + wt_domain[2]*nvi[2] + wt_domain[3]*nvi[3]).xypart();
					ionflow_out = (h*nvi_trap - applied_displacement_edge).dot(edge_normal);
					//nve_trap = (wt_domain[0]*nve[0] + wt_domain[1]*nve[1] + wt_domain[2]*nve[2] + wt_domain[3]*nve[3]).xypart();
					elecflow_out = (h*nve_trap- applied_displacement_edge).dot(edge_normal);

					pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(elecflow_out-ionflow_out);

			if (iVertex == iReport) {
				fprintf(fp, "chargeflow from v0: [GAUSS][UNITY] %1.14E \n",
							FOUR_PI_Q*(elecflow_out-ionflow_out));
			}
					if (pVertex->flags == OUTERMOST) {
						// projected outer edge:
						elecflow_out = (h*nve[0]).dot(extranormal);
						pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(elecflow_out);
					}

					// __________________________________________________________________
					// 3b. Eps_Gauss += 4 pi rho[v nearby[A nearby]]
					// 3c. Effect of TUNEFAC->Ez
					// __________________________________________________________________

					// v <-- E so v_neigh is taken to be affected by dA/dt = 2(A -Ak)/ch - Adot_k/c
					
					// E in self affects vself via sigma_e ; vself affects
					// d_edge_normal += h*wt[0] vself dot edge_normal
					// effect of Ax_k+1 on (ve-vi).x : -3/2 (sigma_e.xx-sigma_i.xx)/ch
					// effect on (ve-vi).y : -3/2(sigma_e.yx - sigma_i.yx)*/ch

					// Multiply all by 4pi q
				
					if (bConsistentChargeflow == false) {

						n_e = pVertex->Elec.mass/pVertex->AreaCell;
						n_i = pVertex->Ion.mass/pVertex->AreaCell;
						pVertex->coeff_self[GAUSS][AZ] += FOUR_PI_Q*
								h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*(-coeff_Aplus_for_Adot/ch)
										+ edge_normal.y*(n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*(-coeff_Aplus_for_Adot/ch));

						//pVertex->coeff_self[GAUSS][TUNEFAC] += FOUR_PI_Q*h*wt_domain[0]* pVertex->temp2.y*
						//	  ( (n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*edge_normal.x
						//	+   (n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*edge_normal.y);
						
						// Phi and phi_anode now determine Ez
						pVertex->coeff_self[GAUSS][PHI] += FOUR_PI_Q*h*wt_domain[0]*(pVertex->Ez_coeff_on_phi)*
							  ( (n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*edge_normal.x
							+   (n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*edge_normal.y);
						pVertex->coeff_self[GAUSS][PHI_ANODE] += FOUR_PI_Q*h*wt_domain[0]*(pVertex->Ez_coeff_on_phi_anode)*
							  ( (n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*edge_normal.x
							+   (n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*edge_normal.y);
						

						pVertex->coeff_self[GAUSS][AX] += FOUR_PI_Q*
								h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xx-0.5*n_i*pVertex->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
										+ edge_normal.y*(n_e*pVertex->sigma_e.yx-0.5*n_i*pVertex->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch) );
						pVertex->coeff_self[GAUSS][AY] += FOUR_PI_Q*
								h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xy-0.5*n_i*pVertex->sigma_i.xy)*(-coeff_Aplus_for_Adot/ch)
										+ edge_normal.y*(n_e*pVertex->sigma_e.yy-0.5*n_i*pVertex->sigma_i.yy)*(-coeff_Aplus_for_Adot/ch) );
						
						pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(h*wt_domain[0]/c)*
								((n_e*pVertex->sigma_e-0.5*n_i*pVertex->sigma_i)*
							   (-coeff_Ak_for_Adot*pVertex->A/h - coeff_Adot*pVertex->Adot)).dotxy(edge_normal);
						
						
			if (iVertex == iReport) {
				fprintf(fp, "xy chargeflow due to phi->Ez: [GAUSS][PHI] %1.14E [PIH_ANODE] %1.14E \n"
					"pVertex->Ez_coeff_on_phi %1.14E pVertex->Ez_coeff_on_phi_anode %1.14E \n",
					FOUR_PI_Q*h*wt_domain[0]*(pVertex->Ez_coeff_on_phi)*
							  ( (n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*edge_normal.x
							+   (n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*edge_normal.y),
					FOUR_PI_Q*h*wt_domain[0]*(pVertex->Ez_coeff_on_phi_anode)*
							  ( (n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*edge_normal.x
							+   (n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*edge_normal.y),
							pVertex->Ez_coeff_on_phi,pVertex->Ez_coeff_on_phi_anode);
				fprintf(fp,"[GAUSS][AZ] %1.14E \n",FOUR_PI_Q*
								h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xz-0.5*n_i*pVertex->sigma_i.xz)*(-coeff_Aplus_for_Adot/ch)
										+ edge_normal.y*(n_e*pVertex->sigma_e.yz-0.5*n_i*pVertex->sigma_i.yz)*(-coeff_Aplus_for_Adot/ch)));
				fprintf(fp,"[GAUSS][AX] %1.14E \n",FOUR_PI_Q*
								h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xx-0.5*n_i*pVertex->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
										+ edge_normal.y*(n_e*pVertex->sigma_e.yx-0.5*n_i*pVertex->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch) ));
			}
						// Careful of contiguity for other neighs: easiest if we rotate edge_normal
						// coeff_Aplus_for_Adot, coeff_Ak_for_Adot, coeff_Adot are defined as coeffs to create dA/dt. E += -dA/dt /c
						
						// to be contiguous with the values they will be getting. Then we can work
						// entirely in their own coordinates otherwise.
						
						if (pVertex->flags == OUTERMOST) { // Look into cathode wall.

							// self gets full weight for outer edge: Az-dot (edge) = Az-dot (vertcell)

							// Assuming only electron inwards flow for now.
							sigma___ = ( pVertex->sigma_e*pVertex->Elec.mass)/pVertex->AreaCell; 

							pVertex->coeff_self[GAUSS][AZ] += FOUR_PI_Q*(-coeff_Aplus_for_Adot/ch)*h*
											( extranormal.x*(sigma___.xz) + extranormal.y*(sigma___.yz));
							pVertex->coeff_self[GAUSS][PHI] += FOUR_PI_Q*h*(pVertex->Ez_coeff_on_phi)*
											( (sigma___.xz)*extranormal.x+ (sigma___.yz)*extranormal.y);
							pVertex->coeff_self[GAUSS][PHI_ANODE] += FOUR_PI_Q*h*(pVertex->Ez_coeff_on_phi_anode)*
												( (sigma___.xz)*extranormal.x + (sigma___.yz)*extranormal.y);

							pVertex->coeff_self[GAUSS][AX] += FOUR_PI_Q*h*(-coeff_Aplus_for_Adot/ch)*
												( extranormal.x*(sigma___.xx) + extranormal.y*(sigma___.yx) );
							pVertex->coeff_self[GAUSS][AY] += FOUR_PI_Q*h*(-coeff_Aplus_for_Adot/ch)*
												( extranormal.x*(sigma___.xy)+ extranormal.y*(sigma___.yy));
										
							pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*h*
											((sigma___)*(-(coeff_Ak_for_Adot/ch)*pVertex->A - (coeff_Adot/c)*pVertex->Adot)).dotxy(extranormal);
						};

						if ((pNeighPrev->flags == DOMAIN_VERTEX) || (pNeighPrev->flags == OUTERMOST))
						{
							n_e = pNeighPrev->Elec.mass/pNeighPrev->AreaCell;
							n_i = pNeighPrev->Ion.mass/pNeighPrev->AreaCell;

							//if ((bDiscAxyFromGauss == false) && (bInitial == false)) {
								// Change of heart: we need Axy initially.
								// Perhaps 
							pVertex->coeff[iprev].co[GAUSS][AX] += FOUR_PI_Q*
									h*wt_domain[3]*( edgenorm_use[3].x*(n_e*pNeighPrev->sigma_e.xx-0.5*n_i*pNeighPrev->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[3].y*(n_e*pNeighPrev->sigma_e.yx-0.5*n_i*pNeighPrev->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch));
							pVertex->coeff[iprev].co[GAUSS][AY] += FOUR_PI_Q*
									h*wt_domain[3]*( edgenorm_use[3].x*(n_e*pNeighPrev->sigma_e.xy-0.5*n_i*pNeighPrev->sigma_i.xy)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[3].y*(n_e*pNeighPrev->sigma_e.yy-0.5*n_i*pNeighPrev->sigma_i.yy)*(-coeff_Aplus_for_Adot/ch));
							
							pVertex->coeff[iprev].co[GAUSS][AZ] += FOUR_PI_Q*
									h*wt_domain[3]*( edgenorm_use[3].x*(n_e*pNeighPrev->sigma_e.xz-0.5*n_i*pNeighPrev->sigma_i.xz)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[3].y*(n_e*pNeighPrev->sigma_e.yz-0.5*n_i*pNeighPrev->sigma_i.yz)*(-coeff_Aplus_for_Adot/ch));
							
							pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(h*wt_domain[3]/c)*
								((n_e*pNeighPrev->sigma_e-0.5*n_i*pNeighPrev->sigma_i)*
								 (-coeff_Ak_for_Adot*pNeighPrev->A/h -coeff_Adot*pNeighPrev->Adot)).dotxy(edgenorm_use[3]);

							//pVertex->coeff_self[GAUSS][TUNEFAC] += FOUR_PI_Q*h*wt_domain[3]* pNeighPrev->temp2.y*
							//	 ((n_e*pNeighPrev->sigma_e.xz-0.5*n_i*pNeighPrev->sigma_i.xz)*edgenorm_use[3].x
							//	+ (n_e*pNeighPrev->sigma_e.yz-0.5*n_i*pNeighPrev->sigma_i.yz)*edgenorm_use[3].y);
							// changing Ez in neighbour changes chargeflow through edge:
							pVertex->coeff[iprev].co[GAUSS][PHI] += FOUR_PI_Q*h*wt_domain[3]* pNeighPrev->Ez_coeff_on_phi*
								 ((n_e*pNeighPrev->sigma_e.xz-0.5*n_i*pNeighPrev->sigma_i.xz)*edgenorm_use[3].x
								+ (n_e*pNeighPrev->sigma_e.yz-0.5*n_i*pNeighPrev->sigma_i.yz)*edgenorm_use[3].y);
							pVertex->coeff_self[GAUSS][PHI_ANODE] += FOUR_PI_Q*h*wt_domain[3]* pNeighPrev->Ez_coeff_on_phi_anode*
								 ((n_e*pNeighPrev->sigma_e.xz-0.5*n_i*pNeighPrev->sigma_i.xz)*edgenorm_use[3].x
								+ (n_e*pNeighPrev->sigma_e.yz-0.5*n_i*pNeighPrev->sigma_i.yz)*edgenorm_use[3].y);
							
						};

						if ((pNeighNext->flags == DOMAIN_VERTEX) || (pNeighNext->flags == OUTERMOST))
						{
							n_e = pNeighNext->Elec.mass/pNeighNext->AreaCell;
							n_i = pNeighNext->Ion.mass/pNeighNext->AreaCell;
							pVertex->coeff[inext].co[GAUSS][AX] += FOUR_PI_Q*
									h*wt_domain[1]*( 
											  edgenorm_use[1].x*(n_e*pNeighNext->sigma_e.xx-0.5*n_i*pNeighNext->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[1].y*(n_e*pNeighNext->sigma_e.yx-0.5*n_i*pNeighNext->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch));
							pVertex->coeff[inext].co[GAUSS][AY] += FOUR_PI_Q*
									h*wt_domain[1]*( edgenorm_use[1].x*(n_e*pNeighNext->sigma_e.xy-0.5*n_i*pNeighNext->sigma_i.xy)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[1].y*(n_e*pNeighNext->sigma_e.yy-0.5*n_i*pNeighNext->sigma_i.yy)*(-coeff_Aplus_for_Adot/ch));
							pVertex->coeff[inext].co[GAUSS][AZ] += FOUR_PI_Q*
									h*wt_domain[1]*( edgenorm_use[1].x*(n_e*pNeighNext->sigma_e.xz-0.5*n_i*pNeighNext->sigma_i.xz)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[1].y*(n_e*pNeighNext->sigma_e.yz-0.5*n_i*pNeighNext->sigma_i.yz)*(-coeff_Aplus_for_Adot/ch));
							
							pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(h*wt_domain[1]/c)*
								((n_e*pNeighNext->sigma_e - 0.5*n_i*pNeighNext->sigma_i)*
												   (-coeff_Ak_for_Adot*pNeighNext->A/h -coeff_Adot* pNeighNext->Adot)).dotxy(edgenorm_use[1]);
							
						//	pVertex->coeff_self[GAUSS][TUNEFAC] += FOUR_PI_Q*h*wt_domain[1]* pNeighNext->temp2.y*
						//		 ((n_e*pNeighNext->sigma_e.xz-0.5*n_i*pNeighNext->sigma_i.xz)*edgenorm_use[1].x
						//		+ (n_e*pNeighNext->sigma_e.yz-0.5*n_i*pNeighNext->sigma_i.yz)*edgenorm_use[1].y);
							
							pVertex->coeff[inext].co[GAUSS][PHI] += FOUR_PI_Q*h*wt_domain[1]* pNeighNext->temp2.y*
								 ((n_e*pNeighNext->sigma_e.xz-0.5*n_i*pNeighNext->sigma_i.xz)*edgenorm_use[1].x
								+ (n_e*pNeighNext->sigma_e.yz-0.5*n_i*pNeighNext->sigma_i.yz)*edgenorm_use[1].y);								
							pVertex->coeff_self[GAUSS][PHI_ANODE] += FOUR_PI_Q*h*wt_domain[1]* pNeighNext->temp2.y*
								 ((n_e*pNeighNext->sigma_e.xz-0.5*n_i*pNeighNext->sigma_i.xz)*edgenorm_use[1].x
								+ (n_e*pNeighNext->sigma_e.yz-0.5*n_i*pNeighNext->sigma_i.yz)*edgenorm_use[1].y);
						};

						if ((pNeigh->flags == DOMAIN_VERTEX) || (pNeigh->flags == OUTERMOST))
						{
							n_e = pNeigh->Elec.mass/pNeigh->AreaCell;
							n_i = pNeigh->Ion.mass/pNeigh->AreaCell;
							pVertex->coeff[i].co[GAUSS][AX] += FOUR_PI_Q*
									h*wt_domain[2]*( edgenorm_use[2].x*(n_e*pNeigh->sigma_e.xx-0.5*n_i*pNeigh->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[2].y*(n_e*pNeigh->sigma_e.yx-0.5*n_i*pNeigh->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch));
							pVertex->coeff[i].co[GAUSS][AY] += FOUR_PI_Q*
									h*wt_domain[2]*( edgenorm_use[2].x*(n_e*pNeigh->sigma_e.xy-0.5*n_i*pNeigh->sigma_i.xy)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[2].y*(n_e*pNeigh->sigma_e.yy-0.5*n_i*pNeigh->sigma_i.yy)*(-coeff_Aplus_for_Adot/ch));
							pVertex->coeff[i].co[GAUSS][AZ] += FOUR_PI_Q*
									h*wt_domain[2]*( edgenorm_use[2].x*(n_e*pNeigh->sigma_e.xz-0.5*n_i*pNeigh->sigma_i.xz)*(-coeff_Aplus_for_Adot/ch)
											+ edgenorm_use[2].y*(n_e*pNeigh->sigma_e.yz-0.5*n_i*pNeigh->sigma_i.yz)*(-coeff_Aplus_for_Adot/ch));
							
							pVertex->coeff_self[GAUSS][UNITY] += FOUR_PI_Q*(h*wt_domain[2]/c)*
								((n_e*pNeigh->sigma_e - 0.5*n_i*pNeigh->sigma_i)*
												   (-coeff_Ak_for_Adot*pNeigh->A/h -coeff_Adot* pNeigh->Adot)).dotxy(edgenorm_use[2]);
														
							pVertex->coeff[i].co[GAUSS][PHI] += FOUR_PI_Q*h*wt_domain[2]* pNeigh->Ez_coeff_on_phi*
								 ((n_e*pNeigh->sigma_e.xz-0.5*n_i*pNeigh->sigma_i.xz)*edgenorm_use[2].x
								+ (n_e*pNeigh->sigma_e.yz-0.5*n_i*pNeigh->sigma_i.yz)*edgenorm_use[2].y);
							pVertex->coeff_self[GAUSS][PHI_ANODE] += FOUR_PI_Q*h*wt_domain[2]* pNeigh->Ez_coeff_on_phi_anode*
								 ((n_e*pNeigh->sigma_e.xz-0.5*n_i*pNeigh->sigma_i.xz)*edgenorm_use[2].x
								+ (n_e*pNeigh->sigma_e.yz-0.5*n_i*pNeigh->sigma_i.yz)*edgenorm_use[2].y);
						};
					}; // (bConsistentChargeflow == false) 

					// ___________________________________________________________________
					// 4. Eps_Gauss += 4 pi rho [phi nearby -> E_edge -> J_edge*h]
					// and the 3b parts, in the case of consistent charge flow.
					// ___________________________________________________________________

					// "we stored sigma_J for all tri centroids"
					//sigma_J_use = 0.5*(sigma_J[i] + sigma_J[iprev]);
					
					// Instead recalculate sigma_J_use here and scrap sigma_J above.
					// First obtain the sigma_J for each of the 4 points, rotated to be contiguous.

					sigma_J[0] = q*(0.5*pVertex->sigma_i*pVertex->Ion.mass
										- pVertex->sigma_e*pVertex->Elec.mass)/pVertex->AreaCell;
					sigma_J[1] = q*(0.5*pNeighNext->sigma_i*pNeighNext->Ion.mass
										- pNeighNext->sigma_e*pNeighNext->Elec.mass)/pNeighNext->AreaCell;
					sigma_J[2] = q*(0.5*pNeigh->sigma_i*pNeigh->Ion.mass 
										- pNeigh->sigma_e*pNeigh->Elec.mass)/pNeigh->AreaCell;
					sigma_J[3] = q*(0.5*pNeighPrev->sigma_i*pNeighPrev->Ion.mass
										- pNeighPrev->sigma_e*pNeighPrev->Elec.mass)/pNeighPrev->AreaCell;

						// Let's say sigma belongs to clockwise side but
						// E is calc'd on anticlockwise side; want J also anticlockwise.
						// J = Anticlockwise(sigma(Clockwise E))
						// Therefore take
						//if (par[0] == 1) 
						//	Matrix0 = Anticlockwise3*Matrix0*Clockwise3;
					
					if (rotated[1] == 1) 	// sigma belongs on clockwise side.
						sigma_J[1] = Anticlockwise3*sigma_J[1]*Clockwise3;							
					if (rotated[1] == -1)	 
						sigma_J[1] = Clockwise3*sigma_J[1]*Anticlockwise3;							
					if (rotated[2] == 1) 
						sigma_J[2] = Anticlockwise3*sigma_J[2]*Clockwise3;	
					if (rotated[2] == -1) 
						sigma_J[2] = Clockwise3*sigma_J[2]*Anticlockwise3;							
					if (rotated[3] == 1) 
						sigma_J[3] = Anticlockwise3*sigma_J[3]*Clockwise3;	
					if (rotated[3] == -1) 
						sigma_J[3] = Clockwise3*sigma_J[3]*Anticlockwise3;	
					
					sigma_J_use = wt_domain[0]*sigma_J[0] + wt_domain[1]*sigma_J[1]
								+ wt_domain[2]*sigma_J[2] + wt_domain[3]*sigma_J[3];
																					
					if ((pNeighPrev->flags == INNER_VERTEX) || (pNeighNext->flags == INNER_VERTEX)) {
						// azimuthal at insulator
						// Suppose that grad comes only from this point and neighbour.
						// Gradient is considered to run along vector between u[0] and u[2].
						
						// slope = (f[0]-f[2])/||u[0]-u[2]|| 
						// gradient = slope. (u[0]-u[2])/||u[0]-u[2]|| 
						
						// Then when we move from u[2] to u[0]: f += (u[0]-u[2]).gradient = (f[0]-f[2])
						
						real modsq;
						Vector2 grad0,grad2;
						modsq = (u[0].x-u[2].x)*(u[0].x-u[2].x) + (u[0].y-u[2].y)*(u[0].y-u[2].y);
						grad0.x = (u[0].x-u[2].x)/modsq;
						grad0.y = (u[0].y-u[2].y)/modsq;
						grad2.x = -grad0.x; 
						grad2.y = -grad0.y;
						
						pVertex->coeff_self[GAUSS][PHI] += FOURPI*h*(grad0.x*(sigma_J_use.xx*edge_normal.x
																			+ sigma_J_use.yx*edge_normal.y)
																	+ grad0.y*(sigma_J_use.xy*edge_normal.x
																			+ sigma_J_use.yy*edge_normal.y));
						pVertex->coeff[i].co[GAUSS][PHI] += FOURPI*h*(grad2.x*(sigma_J_use.xx*edge_normal.x
																			+ sigma_J_use.yx*edge_normal.y)
																    + grad2.y*(sigma_J_use.xy*edge_normal.x
																			+ sigma_J_use.yy*edge_normal.y));
						// grad phi -> -E -> dot with outward -> get + inflow
						
						//pVertex->coeff_self[GAUSS][AX] += FOUR_PI_Q*
						//		h*wt_domain[0]*( edge_normal.x*(n_e*pVertex->sigma_e.xx-0.5*n_i*pVertex->sigma_i.xx)*(-coeff_Aplus_for_Adot/ch)
						//				+ edge_normal.y*(n_e*pVertex->sigma_e.yx-0.5*n_i*pVertex->sigma_i.yx)*(-coeff_Aplus_for_Adot/ch) );
						
					} else {

						pVertex->coeff_self[GAUSS][PHI] += FOURPI*h*(grad[0].x*(sigma_J_use.xx*edge_normal.x
																	+ sigma_J_use.yx*edge_normal.y)
																  + grad[0].y*(sigma_J_use.xy*edge_normal.x
																	+ sigma_J_use.yy*edge_normal.y));
						// Plus because: J is going outward here, but we have E = -grad phi.
						pVertex->coeff[iprev].co[GAUSS][PHI] += FOURPI*h*(grad[3].x*(sigma_J_use.xx*edge_normal.x
																	+ sigma_J_use.yx*edge_normal.y)
																  + grad[3].y*(sigma_J_use.xy*edge_normal.x
																	+ sigma_J_use.yy*edge_normal.y));

						pVertex->coeff[i].co[GAUSS][PHI] += FOURPI*h*(grad[2].x*(sigma_J_use.xx*edge_normal.x
																	+ sigma_J_use.yx*edge_normal.y)
																  + grad[2].y*(sigma_J_use.xy*edge_normal.x
																	+ sigma_J_use.yy*edge_normal.y));

						pVertex->coeff[inext].co[GAUSS][PHI] += FOURPI*h*(grad[1].x*(sigma_J_use.xx*edge_normal.x
																	+ sigma_J_use.yx*edge_normal.y)
																  + grad[1].y*(sigma_J_use.xy*edge_normal.x
																			+ sigma_J_use.yy*edge_normal.y));
						
			if (iVertex == iReport) {
				fprintf(fp,"chargeflow [%d] due to phi [GAUSS][PHI] %1.14E \n"
					"sigma_J_use_xy %1.14E %1.14E %1.14E %1.14E \n",
										izNeigh[i],	FOURPI*h*(grad[0].x*(sigma_J_use.xx*edge_normal.x
																	+ sigma_J_use.yx*edge_normal.y)
																  + grad[0].y*(sigma_J_use.xy*edge_normal.x
																	+ sigma_J_use.yy*edge_normal.y)),
																	sigma_J_use.xx,sigma_J_use.xy,
																	sigma_J_use.yx,sigma_J_use.yy);
			};
						if (pVertex->flags == OUTERMOST) {
							// in this case the projected edge is concerned with phi here vs phi==0 in 'cathode' :
							
							// Assume grad is in radial outward direction and
							// radial outward phi gradient = (0-phi)/(dist to back) since back is actually edge of cathode.
							// Does that make any sense? 
							
							// radial_gradient = -phi/(DOMAIN_OUTER_RADIUS-r);
							// E = -radial_gradient * rhat;
							rhat = pVertex->centroid/r;
							Exy_over_phi = rhat/(DOMAIN_OUTER_RADIUS-r);
							// Now Jxy = sigma E and we dot this with the edge normal for the projected edge.
							// only take electron flows:
							sigma___ = q*( pVertex->sigma_e*pVertex->Elec.mass)/pVertex->AreaCell; 
							// Beware: this is electron velocity 
							
							pVertex->coeff_self[GAUSS][PHI] += FOURPI*h*(
												(sigma___.xx*Exy_over_phi.x + sigma___.xy*Exy_over_phi.y)*extranormal.x + 
												(sigma___.yx*Exy_over_phi.x + sigma___.yy*Exy_over_phi.y)*extranormal.y);
							// Note that _outward_ electron motion (v dot extranormal) would increase eps_Gauss.


						}
					} ;
				
					if ((bConsistentChargeflow == true)) {
						// Effect of Adot on chargeflow -- again
					
						pVertex->coeff_self[GAUSS][AZ] += FOURPI*h*wt_domain[0]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
						// Note: Adot/c -> -E -> -J -> dot with outward to get +d/dt rho.
						// coeff_Aplus_for_Adot/ch -> -E -> dot with outward -> get + inflow
						
						//pVertex->coeff_self[GAUSS][TUNEFAC] -= FOURPI*h*wt_domain[0]* pVertex->temp2.y*
						//					( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );						
						pVertex->coeff_self[GAUSS][PHI] -= FOURPI*h*wt_domain[0]* pVertex->Ez_coeff_on_phi*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );						
						pVertex->coeff_self[GAUSS][PHI_ANODE] -= FOURPI*h*wt_domain[0]* pVertex->Ez_coeff_on_phi_anode*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
											
			if (iVertex == iReport) {
				fprintf(fp,"chargeflow from Ez : pVertex->coeff_self[GAUSS][PHI] %1.14E %1.14E\n"
					"pVertex->Ez_coeff_on_phi %1.14E \n",
					pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[GAUSS][PHI_ANODE],pVertex->Ez_coeff_on_phi);
			};
						pVertex->coeff_self[GAUSS][AX] += FOURPI*h*wt_domain[0]*(coeff_Aplus_for_Adot/ch)*
												( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
						pVertex->coeff_self[GAUSS][AY] += FOURPI*h*wt_domain[0]*(coeff_Aplus_for_Adot/ch)*
												( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
									
						pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[0]*
											(( sigma_J_use* ( (coeff_Ak_for_Adot/ch)*pVertex->A 
															+ (coeff_Adot/c)*(pVertex->Adot))   )
													.dotxy(edge_normal));
						
						if (pVertex->flags == OUTERMOST) {
							// only take electron flows:
							sigma___ = q*( pVertex->sigma_e*pVertex->Elec.mass)/pVertex->AreaCell; 
							// qnv

							// Assume Az-dot(edge) = Az-dot(vertex). -coeff*?/ch = -Adot/c -> E.
							// sigma___ E gives q n_e v_e . Dot this with outward normal and we get + to eps_Gauss.
							// Opposite of signs above.
														
							pVertex->coeff_self[GAUSS][AZ] += FOURPI*h*(-coeff_Aplus_for_Adot/ch)*
											( sigma___.xz*extranormal.x + sigma___.yz*extranormal.y );
							pVertex->coeff_self[GAUSS][AX] += FOURPI*h*(-coeff_Aplus_for_Adot/ch)*
												( sigma___.xx*extranormal.x + sigma___.yx*extranormal.y);
							pVertex->coeff_self[GAUSS][AY] += FOURPI*h*(-coeff_Aplus_for_Adot/ch)*
												( sigma___.xy*extranormal.x + sigma___.yy*extranormal.y);
							pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*
											(( sigma___* ( (-coeff_Ak_for_Adot/ch)*pVertex->A 
															+ (-coeff_Adot/c)*(pVertex->Adot))   )
													.dotxy(edge_normal));

							pVertex->coeff_self[GAUSS][PHI] += FOURPI*h*pVertex->Ez_coeff_on_phi*
											( sigma___.xz*extranormal.x + sigma___.yz*edge_normal.y );						
							pVertex->coeff_self[GAUSS][PHI_ANODE] += FOURPI*h*pVertex->Ez_coeff_on_phi_anode*
											( sigma___.xz*edge_normal.x + sigma___.yz*edge_normal.y );
													
						};

						// Let's be careful now about periodic, for the others.
						// sigma_J_use applies to the contiguous image of A.

						if ((pNeighNext->flags == DOMAIN_VERTEX) || (pNeighNext->flags == OUTERMOST))
						{
							pVertex->coeff[inext].co[GAUSS][AZ] += FOURPI*h*wt_domain[1]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							// Note: Adot/c -> -E -> -J -> dot with outward to get +d/dt rho.
		
							pVertex->coeff[inext].co[GAUSS][PHI] -= FOURPI*h*wt_domain[1]* pNeighNext->Ez_coeff_on_phi*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							
							pVertex->coeff_self[GAUSS][PHI_ANODE] -= FOURPI*h*wt_domain[1]* pNeighNext->Ez_coeff_on_phi_anode*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							
							if (rotated[1] == 0) {
								pVertex->coeff[inext].co[GAUSS][AX] += FOURPI*h*wt_domain[1]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								pVertex->coeff[inext].co[GAUSS][AY] += FOURPI*h*wt_domain[1]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
				
								pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[1]*
											(( sigma_J_use* ( (coeff_Ak_for_Adot/ch)*pNeighNext->A + (coeff_Adot/c)*(pNeighNext->Adot)))
												.dotxy(edge_normal));

							} else {
								// rotated[1] == 1 => would have to pull A anticlockwise to be contig.
								
								effect_of_Ax_contig = FOURPI*h*wt_domain[1]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								effect_of_Ay_contig = FOURPI*h*wt_domain[1]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
								if (rotated[1] == 1) {
									pVertex->coeff[inext].co[GAUSS][AX] += Anticlockwise.xx*effect_of_Ax_contig
																		+ Anticlockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[inext].co[GAUSS][AY] += Anticlockwise.xy*effect_of_Ax_contig
																		+ Anticlockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[1]*
											(( sigma_J_use* ( Anticlockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeighNext->A + (coeff_Adot/c)*(pNeighNext->Adot))))
												.dotxy(edge_normal));
								} else {
									pVertex->coeff[inext].co[GAUSS][AX] += Clockwise.xx*effect_of_Ax_contig
																		+ Clockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[inext].co[GAUSS][AY] += Clockwise.xy*effect_of_Ax_contig
																		+ Clockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[1]*
											(( sigma_J_use* ( Clockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeighNext->A + (coeff_Adot/c)*(pNeighNext->Adot))))
												.dotxy(edge_normal));
								};
								
							};

						};
						if ((pNeigh->flags == DOMAIN_VERTEX) || (pNeigh->flags == OUTERMOST))
						{
							pVertex->coeff[i].co[GAUSS][AZ] += FOURPI*h*wt_domain[2]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							// Note: Adot/c -> -E -> -J -> dot with outward to get +d/dt rho.
							
							pVertex->coeff[i].co[GAUSS][PHI] -= FOURPI*h*wt_domain[2]* pNeigh->Ez_coeff_on_phi*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );

							pVertex->coeff_self[GAUSS][PHI_ANODE] -= FOURPI*h*wt_domain[2]* pNeigh->Ez_coeff_on_phi_anode*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );

							
							
							if (rotated[2] == 0) {
								pVertex->coeff[i].co[GAUSS][AX] += FOURPI*h*wt_domain[2]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								pVertex->coeff[i].co[GAUSS][AY] += FOURPI*h*wt_domain[2]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);

								pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[2]*
											(( sigma_J_use* ( (coeff_Ak_for_Adot/ch)*pNeigh->A + (coeff_Adot/c)*(pNeigh->Adot)))
												.dotxy(edge_normal));
							} else {
								effect_of_Ax_contig = FOURPI*h*wt_domain[2]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								effect_of_Ay_contig = FOURPI*h*wt_domain[2]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
								if (rotated[2] == 1) {
									pVertex->coeff[i].co[GAUSS][AX] += Anticlockwise.xx*effect_of_Ax_contig
																		+ Anticlockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[i].co[GAUSS][AY] += Anticlockwise.xy*effect_of_Ax_contig
																		+ Anticlockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[2]*
											(( sigma_J_use* ( Anticlockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeigh->A + (coeff_Adot/c)*(pNeigh->Adot))))
												.dotxy(edge_normal));

								} else {
									pVertex->coeff[i].co[GAUSS][AX] += Clockwise.xx*effect_of_Ax_contig
																		+ Clockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[i].co[GAUSS][AY] += Clockwise.xy*effect_of_Ax_contig
																		+ Clockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[2]*
											(( sigma_J_use* ( Clockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeigh->A + (coeff_Adot/c)*(pNeigh->Adot))))
												.dotxy(edge_normal));
								};
							};
							
						};
						if ((pNeighPrev->flags == DOMAIN_VERTEX) || (pNeighPrev->flags == OUTERMOST))
						{
							pVertex->coeff[iprev].co[GAUSS][AZ] += FOURPI*h*wt_domain[3]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							
							pVertex->coeff[iprev].co[GAUSS][PHI] -= FOURPI*h*wt_domain[3]* pNeighPrev->Ez_coeff_on_phi*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							pVertex->coeff_self[GAUSS][PHI_ANODE] -= FOURPI*h*wt_domain[3]* pNeighPrev->Ez_coeff_on_phi_anode*
											( sigma_J_use.xz*edge_normal.x + sigma_J_use.yz*edge_normal.y );
							
							if (rotated[3] == 0) {
								pVertex->coeff[iprev].co[GAUSS][AX] += FOURPI*h*wt_domain[3]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								pVertex->coeff[iprev].co[GAUSS][AY] += FOURPI*h*wt_domain[3]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
								pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[3]*
											(( sigma_J_use* ( (coeff_Ak_for_Adot/ch)*pNeighPrev->A + (coeff_Adot/c)*(pNeighPrev->Adot)))
												.dotxy(edge_normal));
							} else {
								effect_of_Ax_contig = FOURPI*h*wt_domain[3]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xx*edge_normal.x + sigma_J_use.yx*edge_normal.y);
								effect_of_Ay_contig = FOURPI*h*wt_domain[3]*(coeff_Aplus_for_Adot/ch)*
											( sigma_J_use.xy*edge_normal.x + sigma_J_use.yy*edge_normal.y);
								if (rotated[3] == 1) {
									pVertex->coeff[iprev].co[GAUSS][AX] += Anticlockwise.xx*effect_of_Ax_contig
																		+ Anticlockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[iprev].co[GAUSS][AY] += Anticlockwise.xy*effect_of_Ax_contig
																		+ Anticlockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[3]*
											(( sigma_J_use* ( Anticlockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeighPrev->A + (coeff_Adot/c)*(pNeighPrev->Adot))))
												.dotxy(edge_normal));
								} else {
									pVertex->coeff[iprev].co[GAUSS][AX] += Clockwise.xx*effect_of_Ax_contig
																		+ Clockwise.yx*effect_of_Ay_contig;
									pVertex->coeff[iprev].co[GAUSS][AY] += Clockwise.xy*effect_of_Ax_contig
																		+ Clockwise.yy*effect_of_Ay_contig;
									pVertex->coeff_self[GAUSS][UNITY] += FOURPI*h*wt_domain[3]*
											(( sigma_J_use* ( Clockwise3*(
														(coeff_Ak_for_Adot/ch)*pNeighPrev->A + (coeff_Adot/c)*(pNeighPrev->Adot))))
												.dotxy(edge_normal));
								};
							};
							
						}; // ((pNeighPrev->flags == DOMAIN_VERTEX) || (pNeighPrev->flags == OUTERMOST))
					}; // bConsistentChargeflow
				};	// whether domain vertex && one of the triangles is domain triangle.
				
				// {{____________________________________________________________________________________}}

				// ___________________________________________________________________
				// 5. Eps_Amp = Lap A
				// watch out for rotating Axy.
				// ___________________________________________________________________
				
				pVertex->coeff_self[AMPZ][AZ] += Lapcontrib_self;
				pVertex->coeff[iprev].co[AMPZ][AZ] += Lapcontrib_iprev;
				pVertex->coeff[inext].co[AMPZ][AZ] += Lapcontrib_inext;
				pVertex->coeff[i].co[AMPZ][AZ] += Lapcontrib_i;
				
				pVertex->coeff_self[AMPX][AX] += Lapcontrib_self;
				pVertex->coeff_self[AMPY][AY] += Lapcontrib_self;

				if (pVertex->flags == INNERMOST) {
					// Set Ax,Ay to radially decline inwards:

					// Contribution to integrated Laplacian = edge normal dot grad at edge
					// = edge length dot normal gradient; normal gradient = -A_vertex /r_vertex.
					
					pVertex->coeff_self[AMPX][AX] += edgelen_proj*(-1.0/r);
					pVertex->coeff_self[AMPY][AY] += edgelen_proj*(-1.0/r);
				};
				if (pVertex->flags == OUTERMOST) {
					// A declines with 1/r. Therefore?
					// A = C/r
					// dA/dr = -C/r^2 
					// C = pVertex->A*r 
					// dA/dr(edge) = - pVertex->A*r/(r_edge*r_edge)
					pVertex->coeff_self[AMPX][AX] += edgelen_proj*(-1.0*r/(DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS));
					pVertex->coeff_self[AMPY][AY] += edgelen_proj*(-1.0*r/(DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS));
					
					if (iVertex == 27381) {
						iVertex = iVertex;
					}
					// Also need here to do Az looking down to 0 within cathode wall, just to anchor it:						
					// outward derivative = (-A)/(DOR-r);
					pVertex->coeff_self[AMPZ][AZ] += edgelen_proj*(-1.0/(DOMAIN_OUTER_RADIUS-r));						
				};
			
				
				if (rotated[3] == 0) {
					pVertex->coeff[iprev].co[AMPX][AX] += Lapcontrib_iprev;
					pVertex->coeff[iprev].co[AMPY][AY] += Lapcontrib_iprev;
				} else {
					if (rotated[3] == 1) {
						// it is clockwise; A affects with Anticlockwise*A
						pVertex->coeff[iprev].co[AMPX][AX] += Lapcontrib_iprev*Anticlockwise.xx;
						pVertex->coeff[iprev].co[AMPX][AY] += Lapcontrib_iprev*Anticlockwise.xy;
						pVertex->coeff[iprev].co[AMPY][AX] += Lapcontrib_iprev*Anticlockwise.yx;
						pVertex->coeff[iprev].co[AMPY][AY] += Lapcontrib_iprev*Anticlockwise.yy;														 
					} else {
						pVertex->coeff[iprev].co[AMPX][AX] += Lapcontrib_iprev*Clockwise.xx;
						pVertex->coeff[iprev].co[AMPX][AY] += Lapcontrib_iprev*Clockwise.xy;
						pVertex->coeff[iprev].co[AMPY][AX] += Lapcontrib_iprev*Clockwise.yx;
						pVertex->coeff[iprev].co[AMPY][AY] += Lapcontrib_iprev*Clockwise.yy;														 
					};
				};
				
				if (rotated[1] == 0) {
					pVertex->coeff[inext].co[AMPX][AX] += Lapcontrib_inext;
					pVertex->coeff[inext].co[AMPY][AY] += Lapcontrib_inext;
				} else {
					if (rotated[1] == 1) {
						// it is clockwise; A affects with Anticlockwise*A
						pVertex->coeff[inext].co[AMPX][AX] += Lapcontrib_inext*Anticlockwise.xx;
						pVertex->coeff[inext].co[AMPX][AY] += Lapcontrib_inext*Anticlockwise.xy;
						pVertex->coeff[inext].co[AMPY][AX] += Lapcontrib_inext*Anticlockwise.yx;
						pVertex->coeff[inext].co[AMPY][AY] += Lapcontrib_inext*Anticlockwise.yy;														 
					} else {
						pVertex->coeff[inext].co[AMPX][AX] += Lapcontrib_inext*Clockwise.xx;
						pVertex->coeff[inext].co[AMPX][AY] += Lapcontrib_inext*Clockwise.xy;
						pVertex->coeff[inext].co[AMPY][AX] += Lapcontrib_inext*Clockwise.yx;
						pVertex->coeff[inext].co[AMPY][AY] += Lapcontrib_inext*Clockwise.yy;														 
					};
				};

				if (rotated[2] == 0) {
					pVertex->coeff[i].co[AMPX][AX] += Lapcontrib_i;
					pVertex->coeff[i].co[AMPY][AY] += Lapcontrib_i;
				} else {
					if (rotated[2] == 1) {
						// it is clockwise; A affects with Anticlockwise*A
						pVertex->coeff[i].co[AMPX][AX] += Lapcontrib_i*Anticlockwise.xx;
						pVertex->coeff[i].co[AMPX][AY] += Lapcontrib_i*Anticlockwise.xy;
						pVertex->coeff[i].co[AMPY][AX] += Lapcontrib_i*Anticlockwise.yx;
						pVertex->coeff[i].co[AMPY][AY] += Lapcontrib_i*Anticlockwise.yy;														 
					} else {
						pVertex->coeff[i].co[AMPX][AX] += Lapcontrib_i*Clockwise.xx;
						pVertex->coeff[i].co[AMPX][AY] += Lapcontrib_i*Clockwise.xy;
						pVertex->coeff[i].co[AMPY][AX] += Lapcontrib_i*Clockwise.yx;
						pVertex->coeff[i].co[AMPY][AY] += Lapcontrib_i*Clockwise.yy;														 
					};
				};
						
				// ___________________________________________________________________
				// 6. Eps_Amp += 4pi/c I(- grad phi) [domain vertex]
				// ___________________________________________________________________

				if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST)) {
					// Each neighbour phi contributes to 'grad phi' estimate in the vertcell region.
					// We are meant to integrate over the vertcell to give the equation.
					// This is accounted for by sigma__I.
					// In the case of outermost we need to consider the projected edge where we look at phi==0
					
					pVertex->coeff_self[AMPX][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.xx)*( -gradcontrib_self.x)
															+ (sigma__I.xy)*( -gradcontrib_self.y));
					pVertex->coeff[iprev].co[AMPX][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.xx)*( -gradcontrib_iprev.x)
															+ (sigma__I.xy)*( -gradcontrib_iprev.y));
					pVertex->coeff[i].co[AMPX][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.xx)*( -gradcontrib_i.x)
															+ (sigma__I.xy)*( -gradcontrib_i.y));
					pVertex->coeff[inext].co[AMPX][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.xx)*( -gradcontrib_inext.x)
															+ (sigma__I.xy)*( -gradcontrib_inext.y));
						
					pVertex->coeff_self[AMPY][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.yx)*( -gradcontrib_self.x)
															+ (sigma__I.yy)*( -gradcontrib_self.y));
					pVertex->coeff[iprev].co[AMPY][PHI] += FOUR_PI_OVER_C*
														( (sigma__I.yx)*( -gradcontrib_iprev.x)
															+ (sigma__I.yy)*( -gradcontrib_iprev.y));
					pVertex->coeff[i].co[AMPY][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.yx)*( -gradcontrib_i.x)
															+ (sigma__I.yy)*( -gradcontrib_i.y));
					pVertex->coeff[inext].co[AMPY][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.yx)*( -gradcontrib_inext.x)
															+ (sigma__I.yy)*( -gradcontrib_inext.y));
					
					pVertex->coeff_self[AMPZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_self.x)
															+ (sigma__I.zy)*( -gradcontrib_self.y));
					pVertex->coeff[iprev].co[AMPZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_iprev.x)
															+ (sigma__I.zy)*( -gradcontrib_iprev.y));
					pVertex->coeff[i].co[AMPZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_i.x)
															+ (sigma__I.zy)*( -gradcontrib_i.y));
					pVertex->coeff[inext].co[AMPZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_inext.x)
															+ (sigma__I.zy)*( -gradcontrib_inext.y));
						
					// contribution to Eps Iz is same as for Ampz 

					// But if we want to put it on pNeigh then we have to have zeroed coefficients initially.
					// We do want to.
						
					pVertex->coeff_self[IZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_self.x)
															+ (sigma__I.zy)*( -gradcontrib_self.y));
					pNeighPrev->coeff_self[IZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_iprev.x)
															+ (sigma__I.zy)*( -gradcontrib_iprev.y)); // = Ampz contrib
					pNeigh->coeff_self[IZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_i.x)
															+ (sigma__I.zy)*( -gradcontrib_i.y)); // = Ampz contrib
					pNeighNext->coeff_self[IZ][PHI] += FOUR_PI_OVER_C*
															( (sigma__I.zx)*( -gradcontrib_inext.x)
															+ (sigma__I.zy)*( -gradcontrib_inext.y)); // = Ampz contrib

					// Now let's be careful: we said at insulator that we ignore looking azimuthally, or give
					// special treatment -- phi cannot affect from within ins and neither does Adot.
					// How are we treating effect on current in cell?
					//			*

					if (pVertex->flags == OUTERMOST) {
						// gradcontrib_self = wt[0]*edge_normal/pVertex->AreaCell;
						// gradcontrib_inext = wt[1]*edge_normal/pVertex->AreaCell;
						// Basically average phi on to the edge, multiply this by edge_normal and integrate to get
						// integral of grad.

						// But phi==0 along the edge at the cathode wall, so for now it looks like do nothing there.
						// How reasonable does that seem?
					};
				}; // whether domain vertex for including grad phi -> current
			};	// whether side-to-side on outermost/innermost vertex -- neglected apparently
		}; // next neighbour
		
			if (iVertex == iReport) {
				fclose(fp);
			};

		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			// ___________________________________________________________________
			// 7a. Eps_Amp += 4pi/c J(A self)
			// 7b. Eps_Amp += 4pi/c J(Ez = tunefac*shape)
			// ___________________________________________________________________

			// Ax --> Ex --> Jxyz --> epsAmp_xyz
			// epsAmpx <-- Jx <-- Exyz <-- Axyz [same dimensions]
			
			Vector3 effect3 = FOUR_PI_OVER_C*( sigma__I * (-coeff_Ak_for_Adot*pVertex->A/ch -coeff_Adot* pVertex->Adot/c));
			
			pVertex->coeff_self[AMPX][AX] += FOUR_PI_OVER_C*( sigma__I.xx * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPX][AY] += FOUR_PI_OVER_C*( sigma__I.xy * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPX][AZ] += FOUR_PI_OVER_C*( sigma__I.xz * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPX][UNITY] += effect3.x;
			pVertex->coeff_self[AMPX][PHI] += FOUR_PI_OVER_C* sigma__I.xz * pVertex->Ez_coeff_on_phi;
			pVertex->coeff_self[AMPX][PHI_ANODE] += FOUR_PI_OVER_C* sigma__I.xz * pVertex->Ez_coeff_on_phi_anode;
			
			pVertex->coeff_self[AMPY][AX] += FOUR_PI_OVER_C*( sigma__I.yx * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPY][AY] += FOUR_PI_OVER_C*( sigma__I.yy * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPY][AZ] += FOUR_PI_OVER_C*( sigma__I.yz * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPY][UNITY] += effect3.y;
			pVertex->coeff_self[AMPY][PHI] += FOUR_PI_OVER_C* sigma__I.yz * pVertex->Ez_coeff_on_phi;
			pVertex->coeff_self[AMPY][PHI_ANODE] += FOUR_PI_OVER_C* sigma__I.yz * pVertex->Ez_coeff_on_phi_anode;
			
			pVertex->coeff_self[AMPZ][AX] += FOUR_PI_OVER_C*( sigma__I.zx * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPZ][AY] += FOUR_PI_OVER_C*( sigma__I.zy * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPZ][AZ] += FOUR_PI_OVER_C*( sigma__I.zz * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[AMPZ][UNITY] += effect3.z;
			pVertex->coeff_self[AMPZ][PHI] += FOUR_PI_OVER_C* sigma__I.zz * pVertex->Ez_coeff_on_phi;
			pVertex->coeff_self[AMPZ][PHI_ANODE] += FOUR_PI_OVER_C* sigma__I.zz * pVertex->Ez_coeff_on_phi_anode;
			
			pVertex->coeff_self[IZ][AX] += FOUR_PI_OVER_C*( sigma__I.zx * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[IZ][AY] += FOUR_PI_OVER_C*( sigma__I.zy * (-coeff_Aplus_for_Adot/ch));
			pVertex->coeff_self[IZ][AZ] += FOUR_PI_OVER_C*( sigma__I.zz * (-coeff_Aplus_for_Adot/ch)); // sames
			Epsilon_Iz_constant += effect3.z;
			pVertex->coeff_self[IZ][PHI] += FOUR_PI_OVER_C* sigma__I.zz * pVertex->Ez_coeff_on_phi;
			Epsilon_Iz_coeff_On_PhiAnode += FOUR_PI_OVER_C* sigma__I.zz * pVertex->Ez_coeff_on_phi_anode;
			// Epsilon_Iz = 4pi/c(-IzPrescribed + Iz_domain)

			// _____________________________

			// Consider there's an alternative approach to get at A-dot if this way empirically fails
			// [ test it, or both, against (A_k+2-A_k)/2h ]
			// ie
			// instead of storing Adot estimated, store past Adot = (A_k-A_k-1)/h ; then take
			// Adot_k+1 = (A_k+1-A_k)/h + 0.5 (A_k+1-2A_k+A_k-1)/h^2

			// This might well be preferable - I'll have to think about it.
		}; 
		// Reverse current:
		if ((iVertex >= numStartZCurrentRow) && (iVertex <= numEndZCurrentRow ))
		{
			pVertex->coeff_self[AMPZ][UNITY] += FOUR_PI_OVER_C*ReverseJzPerVertcell;
		}		

		memset(&(pVertex->epsilon),0,sizeof(real)*4); // debug: use epsilon to store d[sum eps Gauss]/d?.

		if (bDisconnectAxy) {
			pVertex->coeff_self[AMPX][PHI] = 0.0;
			pVertex->coeff_self[AMPX][AX] = 1.0;
			pVertex->coeff_self[AMPX][AY] = 0.0;
			pVertex->coeff_self[AMPX][AZ] = 0.0;
			pVertex->coeff_self[AMPX][UNITY] = 0.0;
			pVertex->coeff_self[AMPX][PHI_ANODE] = 0.0;

			pVertex->coeff_self[AMPY][PHI] = 0.0;
			pVertex->coeff_self[AMPY][AX] = 0.0;
			pVertex->coeff_self[AMPY][AY] = 1.0;
			pVertex->coeff_self[AMPY][AZ] = 0.0;
			pVertex->coeff_self[AMPY][UNITY] = 0.0;
			pVertex->coeff_self[AMPY][PHI_ANODE] = 0.0;

			pVertex->coeff_self[GAUSS][AX] = 0.0;
			pVertex->coeff_self[GAUSS][AY] = 0.0;
			pVertex->coeff_self[AMPZ][AX] = 0.0;
			pVertex->coeff_self[AMPZ][AY] = 0.0;
			pVertex->coeff_self[IZ][AX] = 0.0;
			pVertex->coeff_self[IZ][AY] = 0.0;

			for (i = 0; i < neigh_len; i++)
			{
				pVertex->coeff[i].co[AMPX][PHI] = 0.0;
				pVertex->coeff[i].co[AMPX][AX] = 0.0;
				pVertex->coeff[i].co[AMPX][AY] = 0.0;
				pVertex->coeff[i].co[AMPX][AZ] = 0.0;
				pVertex->coeff[i].co[AMPY][PHI] = 0.0;
				pVertex->coeff[i].co[AMPY][AX] = 0.0;
				pVertex->coeff[i].co[AMPY][AY] = 0.0;
				pVertex->coeff[i].co[AMPY][AZ] = 0.0;

				pVertex->coeff[i].co[GAUSS][AX] = 0.0;
				pVertex->coeff[i].co[GAUSS][AY] = 0.0;
				pVertex->coeff[i].co[AMPZ][AX] = 0.0;
				pVertex->coeff[i].co[AMPZ][AY] = 0.0;
			};
		};
		if (bDisconnectPhi) {
			pVertex->coeff_self[GAUSS][PHI] = 1.0;
			pVertex->coeff_self[GAUSS][AX] = 0.0;
			pVertex->coeff_self[GAUSS][AY] = 0.0;
			pVertex->coeff_self[GAUSS][AZ] = 0.0;
			pVertex->coeff_self[GAUSS][UNITY] = 0.0;
			pVertex->coeff_self[GAUSS][PHI_ANODE] = 0.0;

			pVertex->coeff_self[AMPX][PHI] = 0.0;
			pVertex->coeff_self[AMPY][PHI] = 0.0;
			pVertex->coeff_self[AMPZ][PHI] = 0.0;

			pVertex->coeff_self[IZ][PHI] = 0.0;

			for (i = 0; i < neigh_len; i++)
			{
				pVertex->coeff[i].co[GAUSS][PHI] = 0.0;
				pVertex->coeff[i].co[GAUSS][AX] = 0.0;
				pVertex->coeff[i].co[GAUSS][AY] = 0.0;
				pVertex->coeff[i].co[GAUSS][AZ] = 0.0;
				
				pVertex->coeff[i].co[AMPX][PHI] = 0.0;
				pVertex->coeff[i].co[AMPY][PHI] = 0.0;
				pVertex->coeff[i].co[AMPZ][PHI] = 0.0;
			};
		};

		++pVertex;
	};


	// RESCALING:
	if (bRescaleGauss) {

		real delta = (X[1].pos-X[0].pos).modulus();
		real rescale = delta/(c*h); // 0.1/ch
		real overall = 0.008;

		printf("rescale = %1.8E \n",rescale);

		fp = fopen("reportresc.txt.","a");
		fprintf(fp,"RESCALE %1.14E\n",rescale);
		fclose(fp);
	
		pVertex = X;
		for (iVertex = 0;iVertex < numVertices; iVertex++)
		{
			
			pVertex->coeff_self[GAUSS][0] *= rescale;
			pVertex->coeff_self[GAUSS][1] *= rescale;
			pVertex->coeff_self[GAUSS][2] *= rescale;
			pVertex->coeff_self[GAUSS][3] *= rescale;
			pVertex->coeff_self[GAUSS][4] *= rescale;
			pVertex->coeff_self[GAUSS][5] *= rescale;
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i = 0; i < neigh_len; i++)
			{
				pVertex->coeff[i].co[GAUSS][0] *= rescale;
				pVertex->coeff[i].co[GAUSS][1] *= rescale;
				pVertex->coeff[i].co[GAUSS][2] *= rescale;
				pVertex->coeff[i].co[GAUSS][3] *= rescale;
			};

			for (int iEqn = 0; iEqn < 4; iEqn++)
			{
				for (i =0; i < 6; i++)
					pVertex->coeff_self[iEqn][i] *= overall;
				for (i = 0; i < neigh_len; i++)
				{
					pVertex->coeff[i].co[iEqn][0] *= overall;
					pVertex->coeff[i].co[iEqn][1] *= overall;
					pVertex->coeff[i].co[iEqn][2] *= overall;
					pVertex->coeff[i].co[iEqn][3] *= overall;
				};				
			}; // shrank all epsilon. IZ equation remains same.

			++pVertex;
		};		
	};

	// DEBUGGING:
	real phicoeff, Axcoeff, Aycoeff, Azcoeff,
		total, total2;
	long neigh_len2, izNeigh2[256], ii;
	FILE * file;
/*
	file = fopen("coeffs2.txt","w");
	pVertex = X;
	for (iVertex = 0;iVertex < numVertices; iVertex++)
	{
		if ((pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y > 3.4*3.4)
			&& (pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y > 3.55*3.55))
		{
		
		fprintf(file,"%d %d %1.12E %1.12E | ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
		// Gauss equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[GAUSS][AY],
			pVertex->coeff_self[GAUSS][AZ],pVertex->coeff_self[GAUSS][UNITY],pVertex->coeff_self[GAUSS][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[GAUSS][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[GAUSS][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[GAUSS][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[GAUSS][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[GAUSS][AX])
			+ fabs(pVertex->coeff_self[GAUSS][AY]) + fabs(pVertex->coeff_self[GAUSS][AZ]);
		total2 = total + fabs(pVertex->coeff_self[GAUSS][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total);
		fprintf(file,"\n");
		
		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-x equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPX][PHI],pVertex->coeff_self[AMPX][AX],pVertex->coeff_self[AMPX][AY],
			pVertex->coeff_self[AMPX][AZ],pVertex->coeff_self[AMPX][UNITY],pVertex->coeff_self[AMPX][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPX][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPX][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPX][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPX][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPX][PHI])
			+ fabs(pVertex->coeff_self[AMPX][AY]) + fabs(pVertex->coeff_self[AMPX][AZ]);
		total2 = total + fabs(pVertex->coeff_self[AMPX][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPX][AX]/total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPX][AX]/total);
		fprintf(file,"\n");
		
		
		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-y equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPY][PHI],pVertex->coeff_self[AMPY][AX],pVertex->coeff_self[AMPY][AY],
			pVertex->coeff_self[AMPY][AZ],pVertex->coeff_self[AMPY][UNITY],pVertex->coeff_self[AMPY][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPY][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPY][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPY][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPY][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPY][PHI])
			+ fabs(pVertex->coeff_self[AMPY][AX]) + fabs(pVertex->coeff_self[AMPY][AZ]);
		total2 = total + fabs(pVertex->coeff_self[AMPY][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPY][AY]/total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPY][AY]/total);
		fprintf(file,"\n");
		
		
		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-z equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPZ][PHI],pVertex->coeff_self[AMPZ][AX],pVertex->coeff_self[AMPZ][AY],
			pVertex->coeff_self[AMPZ][AZ],pVertex->coeff_self[AMPZ][UNITY],pVertex->coeff_self[AMPZ][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPZ][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPZ][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPZ][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPZ][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPZ][PHI])
			+ fabs(pVertex->coeff_self[AMPZ][AX]) + fabs(pVertex->coeff_self[AMPZ][AY]);
		total2 = total + fabs(pVertex->coeff_self[AMPZ][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total);
		fprintf(file,"\n");
		
		
		
		fprintf(file,"%d %d  phi effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[AMPX][PHI],pVertex->coeff_self[AMPY][PHI],
			pVertex->coeff_self[AMPZ][PHI]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = X + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][PHI]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][PHI]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][PHI]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][PHI]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[AMPX][PHI])
			+ fabs(pVertex->coeff_self[AMPY][PHI])
			+ fabs(pVertex->coeff_self[AMPZ][PHI]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total);

		fprintf(file,"\n");
			

		// 

		fprintf(file,"%d %d  Ax effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[AMPX][AX],pVertex->coeff_self[AMPY][AX],
			pVertex->coeff_self[AMPZ][AX]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = X + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AX]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AX]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AX]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AX]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AX])
			+ fabs(pVertex->coeff_self[AMPY][AX])
			+ fabs(pVertex->coeff_self[AMPZ][AX]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPX][AX]/total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPX][AX]/total);
		fprintf(file,"\n");

		fprintf(file,"%d %d  Ay effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AY],pVertex->coeff_self[AMPX][AY],pVertex->coeff_self[AMPY][AY],
			pVertex->coeff_self[AMPZ][AY]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = X + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AY]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AY]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AY]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AY]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AY])
			+ fabs(pVertex->coeff_self[AMPX][AY])
			+ fabs(pVertex->coeff_self[AMPZ][AY]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPY][AY]/total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPY][AY]/total);
		fprintf(file,"\n");

		fprintf(file,"%d %d  Az effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AZ],pVertex->coeff_self[AMPX][AZ],pVertex->coeff_self[AMPY][AZ],
			pVertex->coeff_self[AMPZ][AZ]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = X + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AZ]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AZ]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AZ]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AZ])
			+ fabs(pVertex->coeff_self[AMPX][AZ])
			+ fabs(pVertex->coeff_self[AMPY][AZ]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total);
		fprintf(file,"\n\n");

		}; // whether to print vertex

		++pVertex;
	};
	fclose(file);
*/
	
}

void TriMesh::CreateAuxiliarySubmeshes(bool bJustAz)
{

	Vertex * pVertex, *pNeigh,*pNeigh2 ,*pAux, *pTest,*pTest0,*pIntermed,*pVertq;
	long iVertex, iNeigh,iNeigh2, neigh_len, neigh_len2;
	long izNeigh[128],izNeigh2[128];
	int iLevel, i, ii;
	long numFine, tri_len, izTri[128];
	Vertex * Xarray;
	Triangle * pTri,*pAuxTri,*pAuxTriNeigh, *Tarray;
	long numInnermost, numLastRow;
	real * rrarray;
	long * index, *curtain;
	long iLeft, iRight, iIntermed, curtain_length, iIndex,iTri;
	bool bPeriodic, bContinue, bFoundAnother;
	Vector2 u[3], leftpos, rightpos, intermedpos, to_NE, to_NW, 
		rhat, thetahat,to_intermed,qpos,vec_cc_q, to_left, to_right;
	real pdistsq;

	// DEBUG:
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->Temp.x = 0.0;
		pVertex->Temp.y = 0.0;
		pVertex->Temp.z = 0.0;
		pVertex->temp2.x = 0.0;
		++pVertex;
	};


	for (iLevel = 0; iLevel < NUM_COARSE_LEVELS; iLevel++)
	{
		printf("\n\nLevel %d\n\n",iLevel);

		numAuxVertices[iLevel] = 0;
		// Set a flag, then count how many.
		if (iLevel == 0) {
			Xarray = X;
			Tarray = T;
			numFine = numVertices;
			numInnermost = numInnermostRow;
			numLastRow = numOutermostRow;
		} else {
			Xarray = AuxX[iLevel-1];
			Tarray = AuxT[iLevel-1];
			// iLevel is the one we are filling in.
			numFine = numAuxVertices[iLevel-1];
			numInnermost = numInnermostRowAux[iLevel-1];
			numLastRow = numLastRowAux[iLevel-1];
		};		
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine;iVertex++)
		{
			// All vertices not just domain.
			pVertex->iVolley = 0;
			pVertex->iScratch = -1;
			pVertex->ClearCoarseIndexList();
			pVertex++;
		}
		// iVolley == 0 : unused
		// iVolley == 1 : selected
		// iVolley == 2 : within 2 of a selection (add selected index to list of affectors) and so cannot be selected itself.

#define SPACING1
#ifdef SPACING1

		// We are finding that with spacing 2 we can have really a lot of + and - cancelling from Lvl 2 (solved) down to
		// Lvl 1 (large errors which do not seem to instantly annihilate).
		// Perhaps by choosing spacing 1 we can make it simpler.
		
		long numat[128];
		memset(numat,0,sizeof(long)*128);
		numLastRowAux[iLevel] = 0;
		numInnermostRowAux[iLevel] = 0;
		long iStart, iEnd;
		// More efficient I think to do bottom row FIRST -- top row then bottom row then the rest.
		// Is there a way to put an outer loop so that same code exists in the middle.

		printf("1. Assign pVertex->iVolley\n");

		numat[0] = numFine;
		int iPass = 0;
		int Looking_for = 0, high_watermark = 0;		
		do {		
			if (iPass == 0) {
				// First put them in the last row:
				iStart = numFine-numLastRow; iEnd = numFine;
			} else {
				if (iPass == 1) {
					iStart = 0; iEnd = numInnermost;
				} else {
					iStart = 0; iEnd = numFine;
				};
			};
			pVertex = Xarray + iStart;
			for (iVertex = iStart; ((iVertex < iEnd) && (numat[Looking_for] > 0)); iVertex++)
			{
				
				if ((pVertex->iVolley == Looking_for) // this means on 1st pass we will move on to 0 - 
					// not desired behaviour. Therefore do what? Allow it to change looking_for to 3, then it finds them all.
					||
					((iPass < 2) && (pVertex->iVolley != 2)) // select alternate in rows 0 and N-1.
					)
				{
					numat[pVertex->iVolley]--; // used up one
					pVertex->iVolley = 1;
					numat[1]++;
					pVertex->iScratch = iVertex;
					pVertex->ClearCoarseIndexList();
					pVertex->AddUniqueCoarse(iVertex);
					numAuxVertices[iLevel]++;
					if (iVertex < numInnermost) numInnermostRowAux[iLevel]++;
					if (iVertex >= numFine-numLastRow) numLastRowAux[iLevel]++;

					// Set neighbours to not be selected:			
					if (iLevel == 0) {
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
					} else {
						neigh_len = pVertex->GetAuxNeighIndexArray(izNeigh); // geometric neighbours only
					}
					for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						numat[pNeigh->iVolley]--;
						pNeigh->iVolley = 2;
						numat[2]++;
						pNeigh->iScratch = iVertex;
						pNeigh->AddUniqueCoarse(iVertex);
					};
					
					// Deal with extended:
					for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						if (iLevel == 0) {
							neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
						} else {
							neigh_len2 = pNeigh->GetAuxNeighIndexArray(izNeigh2);
						}
						for (i = 0; i < neigh_len2; i++)
						{
							pNeigh2 = Xarray + izNeigh2[i];
							if (pNeigh2->iScratch != iVertex) {
								pNeigh2->iScratch = iVertex;
								if (pNeigh2->iVolley >= 3) {
									numat[pNeigh2->iVolley]--;
									pNeigh2->iVolley++; 
									numat[pNeigh2->iVolley]++;
									// Note that if we create such a point, it is viable for selection
									// until a later action dictates otherwise.
									if ((pNeigh2->iVolley > Looking_for)) {
										Looking_for = pNeigh2->iVolley; // select from these now until all gone.
									};
									// We are always looking_for the highest # of overlaps yet created,
									// until those go down to 0.
								};
								if (pNeigh2->iVolley == 0) {
									pNeigh2->iVolley = 3; 
									numat[0]--;
									numat[3]++;
									if (pNeigh2->iVolley > Looking_for)
										Looking_for = pNeigh2->iVolley; // select from these now until all gone.
								};
							};
						}; // next i
					}; // next iNeigh
				};
				++pVertex;
			};
			
			if (Looking_for > high_watermark) high_watermark = Looking_for;
			if (Looking_for >= 3) {
				while (numat[Looking_for] == 0) Looking_for--; // we'll never fall through 2
				if (Looking_for < 3) Looking_for = 0;
			}				
			
			iPass++;
			printf("iPass %d placed %d \n",iPass,numAuxVertices[iLevel]); // now stuck in a loop.
			
			// NOW DO GRAPH OF AUX MESH:
			if ((iLevel >= 1) && (iPass % 20 == 0)) {
				Graph[4].mhTech = Graph[4].mFX->GetTechniqueByName("MeshTech");
				Graph[4].SetDataWithColourAux(*this,iLevel-1,FLAG_COLOUR_MESH,FLAG_FLAT_MESH,0,0,
										   numAuxTriangles[iLevel-1]);
				Graph[4].RenderAux("",0,this,iLevel-1); 
				Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
			
				printf("Looking_for = %d\n",Looking_for);				
			}

		} while (numat[1] + numat[2] < numFine); // everything adjacent to a selected 

		printf("iPass %d numat: ",iPass);
		for (i = 0; i <= high_watermark; i++) 
			printf("%d %d ; ",i,numat[i]);
		printf("\n");

		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		// NOW UPGRADE ALL COARSE_LEN=1 TO LVL 2

		long numNeighs, numUpgraded = 0;
		pVertex = Xarray ;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if (pVertex->iVolley == 2) 
			{
				if (iLevel == 0) {
					neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				} else {
					neigh_len = pVertex->GetAuxNeighIndexArray(izNeigh); // geometric neighbours only
				}
				numNeighs = 0;
				for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
				{
					pNeigh = Xarray + izNeigh[iNeigh];
					if (pNeigh->iVolley == 1) numNeighs++;
				};
				if (numNeighs < 2) {

					numat[pVertex->iVolley]--; // used up one
					pVertex->iVolley = 1;
					numat[1]++;
					pVertex->iScratch = iVertex;
					pVertex->ClearCoarseIndexList();
					pVertex->AddUniqueCoarse(iVertex);
					numAuxVertices[iLevel]++;
					if (iVertex < numInnermost) numInnermostRowAux[iLevel]++;
					if (iVertex >= numFine-numLastRow) numLastRowAux[iLevel]++;

					// Set neighbours to not be selected:			
					for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						if (pNeigh->iVolley == 2) {
							pNeigh->AddUniqueCoarse(iVertex);
						};
					};
					numUpgraded++;
				}
				
				// What about if we just take the 3 highest coefficients ?
				// Maybe not geometrical __enough__
			}
			++pVertex;
		};
		printf("num upgraded = %d \n\n",numUpgraded);
		
		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		if (iLevel == 1) {
			
			// DOPPELTCHECKEN:

			long num[100];
			memset(num,0,sizeof(long)*100);
			pVertex = AuxX[0];
			for (iVertex = 0; iVertex < numAuxVertices[0]; iVertex++)
			{
				num[pVertex->iVolley]++;
				++pVertex;
			}
			printf("num[0] %d [1] %d [2] %d [3] %d [4] %d [5] %d [6] %d [7] %d [8] %d \n",
				num[0],num[1],num[2],num[3],num[4],num[5],num[6],num[7],num[8]);

		}
		// NOW DO GRAPH OF AUX MESH:
		if (iLevel >= 1) {
			Graph[4].mhTech = Graph[4].mFX->GetTechniqueByName("MeshTech");
			Graph[4].SetDataWithColourAux(*this,iLevel-1,FLAG_COLOUR_MESH,FLAG_FLAT_MESH,0,0,
									   numAuxTriangles[iLevel-1]);
			Graph[4].RenderAux("",1,this,iLevel-1); 
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
		
			printf("Looking_for = %d\nPress t\n",Looking_for);
		}

		// I think that will work: everything ends up either 2 or 1.
		// Now we have to ask: do we want to do anything about fact that some are spaced 2 apart?
		// Ptic near back row?
		// Let's assume not.
		
		/////////////////////////////////////////////////////////////////////////////////////////////////

		// Now change iVolley ==2 to everything else, avoiding adjacency:
		numVolleys[iLevel] = 0;  // iLevel is level we are populating, but here used for fine level, ie would be iLevel+1 index in array.
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if (pVertex->iVolley == 2) {
				// what's next door?
				neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				pVertex->iVolley = 1;
				bool bFound;
				do {
					pVertex->iVolley++;
					bFound = 0;
					for (i = 0; i < neigh_len; i++)
					{
						if ((izNeigh[i] < iVertex) && ((Xarray+izNeigh[i])->iVolley == pVertex->iVolley))
						{
							bFound = true;
							i = neigh_len; // skip out
						}
					}					
					// We considered only points already established, ie those with izNeigh[i] < iVertex.
				} while (bFound);
				if (pVertex->iVolley > numVolleys[iLevel]) numVolleys[iLevel] = pVertex->iVolley;
				numat[2]--;
				numat[pVertex->iVolley]++;
			}
			++pVertex;
		};

		printf("numVolleys lvl %d = %d \n",iLevel,numVolleys[iLevel]);
		
		// We cannot rule out that 0's still remain?
		printf("Volleys numat: ");
		
		if (iLevel >= 1) {
			for (i = 0; i < numVolleys[iLevel]; i++) 
				printf("%d %d ; ",i,numat[i]);
			printf("\nPress d\n");

			Graph[4].RenderAux("",1,this,iLevel-1); 
			char o;
		};
#else

		// 0. To get going, we can start off the innermost row: space every 3.
		

		numInnermostRowAux[iLevel] = 0;

		pVertex = Xarray;	
		for (iVertex = 0; iVertex < numInnermost; iVertex+=3)
		{
			if (pVertex->iVolley == 0) {
				pVertex->iVolley = 1; //selected
				pVertex->AddUniqueCoarse(iVertex);
				numAuxVertices[iLevel]++;
				numInnermostRowAux[iLevel]++;
				neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
				{
					pNeigh = Xarray + izNeigh[iNeigh];
					pNeigh->iVolley = 2;
					pNeigh->AddUniqueCoarse(iVertex);
					neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
					for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
					{
						pNeigh2 = Xarray + izNeigh2[iNeigh2];
						if (pNeigh2->iVolley != 1) {
							pNeigh2->iVolley = 2;
							pNeigh2->AddUniqueCoarse(iVertex);
						};
					};
				};
			};
			pVertex += 3;
		}
		
		// multiple passes:
		bool bFound, bFound2ndpass, bSelect;
		long store;
		do {
			bFound2ndpass = false;
			do {
				bFound = false;
				pVertex = Xarray;
				for (iVertex = 0; iVertex < numFine; iVertex++)
				{
					if (pVertex->iVolley == 0) { 
						// we wish to move to where we are adjacent to at least 2 different selected's regions.
						// Check neighbours:
						bSelect = false;
						store = -1;
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
						{
							pNeigh = Xarray + izNeigh[iNeigh];
							if (pNeigh->coarse_len >= 2) {
								bSelect = true; // can't imagine what it's like; for certain this won't be perfect
							} else {
								if (pNeigh->coarse_len == 1) {
									if (store == -1) store = pNeigh->iCoarseIndex[0];
								} else {
									if (store != pNeigh->iCoarseIndex[0]) bSelect = true;
								};
							};
						};
						
						if (bSelect) {
							pVertex->iVolley = 1;
							pVertex->AddUniqueCoarse(iVertex);
							numAuxVertices[iLevel]++;
							for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
							{
								pNeigh = Xarray + izNeigh[iNeigh];
								pNeigh->iVolley = 2;
								pNeigh->AddUniqueCoarse(iVertex);
								neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
								for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
								{
									pNeigh2 = Xarray + izNeigh2[iNeigh2];
									if (pNeigh2->iVolley != 1) // == 1 should be impossible anyway!
									{
										pNeigh2->iVolley = 2;
										pNeigh2->AddUniqueCoarse(iVertex);
									};
								};
							};
							bFound = true;
						};
					};
					++pVertex;
				};
			} while (bFound);

			// 1B. Now select any that are left with iVolley = 0. 1 pass then if found, return to above.

			pVertex = Xarray;
			for (iVertex = 0; iVertex < numFine; iVertex++)
			{
				if (pVertex->iVolley == 0) { 
					pVertex->iVolley = 1;
					pVertex->AddUniqueCoarse(iVertex);
					numAuxVertices[iLevel]++;

					neigh_len = pVertex->GetNeighIndexArray(izNeigh);
					for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						if (pNeigh->iVolley != 1) { // should never anyway...
							pNeigh->iVolley = 2;
							pNeigh->AddUniqueCoarse(iVertex);
						};
						neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
						for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
						{
							pNeigh2 = Xarray + izNeigh2[iNeigh2];
							if (pNeigh2->iVolley != 1) {
								pNeigh2->iVolley = 2;
								pNeigh2->AddUniqueCoarse(iVertex);
							};
						};
					};
					bFound2ndpass = true;

					iVertex = numFine; // break out - otherwise outer loop is pointless.
				};
				++pVertex;
			};

		} while (bFound2ndpass); // if finding some that didn't fit our favourite pattern, now go back and look again for favourite.

		// Finally: if you have only 1 coarse vertex but you are not a selected point,
		// make you into a selected point. Otherwise at the back we get some that go 
		// adding up to make low errors in the level above - but don't seem to easily annihilate
		// - perhaps because the level above dominates.

		printf("iLevel %d upgraded: \n",iLevel);
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if ((pVertex->iVolley != 1) && (pVertex->coarse_len < 2))
			{ 
				pVertex->iVolley = 1;
				pVertex->ClearCoarseIndexList(); // just map to self
				pVertex->AddUniqueCoarse(iVertex);
				numAuxVertices[iLevel]++;
				printf(" %d",iVertex);
				
				neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				for (iNeigh=0; iNeigh < neigh_len; iNeigh++)
				{
					pNeigh = Xarray + izNeigh[iNeigh];
					if (pNeigh->iVolley != 1) { // should never anyway...
						pNeigh->iVolley = 2;
						pNeigh->AddUniqueCoarse(iVertex);
					};
					neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
					for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
					{
						pNeigh2 = Xarray + izNeigh2[iNeigh2];
						if (pNeigh2->iVolley != 1) {
							pNeigh2->iVolley = 2;
							pNeigh2->AddUniqueCoarse(iVertex);
						};
					};
				};
			};
			++pVertex;
		};
		printf("\n \n");
#endif

		// 2. Create AuxTriangles:
		// The data that is needed...

		printf("2.Create AuxTriangles\n");

		// Create vertices and sort them according to radius.
// Create auxiliary vertex array : all we really will want is neighbour list, coefficient, solution and epsilon
		// Also need position so that we can create weights for higher levels.
		// So we don't need to do anything else much.
		
		AuxX[iLevel] = new Vertex[numAuxVertices[iLevel]];
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			pAux->iLevel = iLevel; // know what level you are on.
			++pAux;
		}
		
		// . Set position;
		// . Change fine-level coarse indices to index this array:
		//     first set for selected, ie iVolley == 1 : this is the fine->coarse mapping index
		//     second set for everything else via that.
		
		// What is needed for Galerkin? We can follow out: each fine relationship, knows where to affect coarse epsilon...
		// so nothing.
		long iCaret = 0;
		pVertex = Xarray;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if (pVertex->iVolley == 1) {
				// selected:
				pAux->pos = pVertex->pos;
				pAux->centroid = pVertex->centroid;
				pVertex->iCoarseIndex[0] = iCaret;
				++iCaret;
				++pAux;
			};
			++pVertex;
		};
		
#define AUX_TRIS
#ifdef AUX_TRIS
		
		
		rrarray = new real[numAuxVertices[iLevel]];
		index = new long[numAuxVertices[iLevel]];
		
		Vector2 temp2;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			temp2 = pAux->pos; // get into habit of setting local data instead of dereferencing
			rrarray[iVertex] = temp2.x*temp2.x+temp2.y*temp2.y;
			index[iVertex] = iVertex;
			++pAux;
		}
	
		QuickSort(index,rrarray,0,numAuxVertices[iLevel]-1); // inclusive limits

		// now presumably index[0] contains index of furthest-in point.

		// Dimension AuxTriangle array:
		numAuxTriangles[iLevel] = 
			(numAuxVertices[iLevel]+numInnermostRowAux[iLevel]+numLastRowAux[iLevel])*2;
		// that ought to be enough?
		AuxT[iLevel] = new Triangle[numAuxTriangles[iLevel]];
		// remember to delete this
		printf("numAuxTriangles = %d.\n",numAuxTriangles[iLevel]);

		if (AuxT[iLevel] == 0) {
			printf("Memory allocation error.\n");
			getch();
		};

		numTrianglesAuxAllocated[iLevel] = numAuxTriangles[iLevel];
		numAuxTriangles[iLevel] = 0; // add as we call SetAuxTri
		
		// Add points incrementally and maintain curtain.
		
		curtain = new long[numAuxVertices[iLevel]]; // why not
		curtain_length = numInnermostRowAux[iLevel]; 
		for (iVertex = 0; iVertex < numInnermostRowAux[iLevel]; iVertex++)
			curtain[iVertex] = iVertex; // note: not from index[]
		// keep this iVertex: it should be at correct position in array.

		for (; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			pAux = AuxX[iLevel]+index[iVertex];
			
			// Make triangles from this point to points in the curtain.
			// 90 degrees out is reasonable, but always make sure it's convex??
			
			// ie what? do not want 2 points of curtain in line for given ray.
			// So all points on curtain must remain in azimuthal order.
			
			// .Do a search of curtain[] to find azimuthally in which interval
			// this point lies:
			// Note we need to know which interval contains PB.
			// Let's make sure that is always the interval to 0.

			real gradient = pAux->pos.x/pAux->pos.y;
			pTest = AuxX[iLevel]+curtain[curtain_length-1];
			pTest0 = AuxX[iLevel]+curtain[0];
			if ((gradient >= pTest->pos.x/pTest->pos.y) ||
				(gradient <= pTest0->pos.x/pTest0->pos.y)) // unhandled exception
			{
				iLeft = curtain_length-1;
				iRight = 0;				
				bPeriodic = true; // left is to left of PB, right is to right of PB.
			} else {
				iRight = 0;
				do {
					++iRight;
					pTest = AuxX[iLevel]+curtain[iRight];
				} while (gradient > pTest->pos.x/pTest->pos.y);
				iLeft = iRight-1;
				bPeriodic = false;
			};
			
			// .Connect to the sides of this interval; update curtain
						
			pTri = SetAuxTri(iLevel,curtain[iLeft],curtain[iRight],index[iVertex]);
			if (bPeriodic) 
			{
				pTri->periodic = ((pAux->pos.x < 0.0)?1:0) + 1;
			} else {
				pTri->periodic = 0;
			};
			
			if ((bPeriodic == false) || (pAux->pos.x < 0.0)) {
				//curtain.Insert(index[iVertex],iRight);
				memmove(curtain+iRight+1,curtain+iRight,sizeof(long)*(curtain_length-iRight));
				curtain[iRight] = index[iVertex];
				iIndex = iRight;
				curtain_length++;
				// so if pos.x < 0 but periodic = true, move whole thing and slot this in at 0
				// because < 0 means it appears on the left of the array.
			} else {
				curtain[curtain_length] = index[iVertex];
				iIndex = curtain_length;
				curtain_length++;
			}; 
			
			
			// Now what will happen: 
			// If left point crossed PB then when we delete it, we are always looking
			// just across boundary to intermediate = n-1

			// If OUR point is n-1 then we are unconcerned about PB here.
			// We are adding tris and none of them have periodic flag set.

			// If we start off away from the boundary then we could delete points down to 0
			// and then we set bAcrossPB when our point becomes point 0.
			// From that point onwards, intermediate = n-1, and all further tris have per flag set.
			// If OUR index = 1 then we only rotate the farther left point.

			// So we can drop bools and just look at our index in curtain.

			// /////////////////////////////////////////////////////////////////////////////////////////
			// Do further connections each direction as long as there are more points within 45 degrees:
			
			do {
				iIntermed = iIndex-1; if (iIntermed == -1) iIntermed = curtain_length-1;
				iLeft = iIntermed-1; if (iLeft == -1) iLeft = curtain_length-1;
					
				// Is this point within 45 degrees, streaming back; 
				// (alternatively, will it form an acute triangle?)
				pTest = AuxX[iLevel]+curtain[iLeft];
				leftpos = pTest->pos;
				if (iIndex <= 1) leftpos = Anticlockwise*leftpos; // contiguous
				
				pIntermed = AuxX[iLevel]+curtain[iIntermed];
				intermedpos = pIntermed->pos;
				if (iIndex == 0) 
					intermedpos = Anticlockwise*intermedpos;

				to_NE = pAux->pos - leftpos;
				rhat = pAux->pos;//pAux->pos.modulus(); // sqrt
				thetahat.x = rhat.y; // clockwise
				thetahat.y = -rhat.x;
				if (to_NE.dot(thetahat) < to_NE.dot(rhat)) // both > 0
				{
					bFoundAnother = true;					
				} else {
					// We could still include it, for an acute triangle formed?
					// How to detect?
					// Vector dot product looking out from intermediate to 2 others,
					// should be > 0
					to_left = leftpos - intermedpos;
					to_right = pAux->pos - intermedpos;
					bFoundAnother =  (to_left.dot(to_right) > 0.0);
				};
				if (bFoundAnother) {
					// Want to include it. Do a test that we are not cutting off a corner:
					
					to_intermed = intermedpos - pAux->pos;
					to_NW.x = -to_NE.y;
					to_NW.y = to_NE.x;
					if (to_intermed.dot(to_NW) > 0.0) {
						// above the line between the two others.
						bFoundAnother = false;
					}					
				}
				if (bFoundAnother) {
					pTri = SetAuxTri(iLevel,curtain[iLeft],curtain[iIntermed],index[iVertex]);
					pTri->periodic = 0;
					if (iIndex == 0) pTri->periodic = 1;
					if (iIndex == 1) pTri->periodic = 2;
					
					// remove point from curtain:
					if (iIndex == 0) {
						// do nothing: iIntermed is last point
					} else {
						memmove(curtain+iIntermed,curtain+iIndex,sizeof(long)*(curtain_length-iIndex));
						iIndex--; // it moved to left.
					};
					curtain_length--;
				};
			} while (bFoundAnother);
			
			// Now right hand side:
			
			do {
				iIntermed = iIndex+1; if (iIntermed == curtain_length) iIntermed = 0;
				iRight = iIntermed+1; if (iRight == curtain_length) iRight = 0;
					
				// Is this point within 45 degrees, streaming back; 
				// alternatively, will it form an acute triangle?
				pTest = AuxX[iLevel]+curtain[iRight];
				rightpos = pTest->pos;
				if (iIndex >= curtain_length-2) rightpos = Clockwise*rightpos; // contiguous
				
				pIntermed = AuxX[iLevel]+curtain[iIntermed];
				intermedpos = pIntermed->pos;
				if (iIndex == curtain_length-1) 
					intermedpos = Clockwise*intermedpos;

				to_NW = pAux->pos - rightpos;
				rhat = pAux->pos;//pAux->pos.modulus(); // sqrt
				thetahat.x = -rhat.y;
				thetahat.y = rhat.x; // anticlockwise
				if (to_NW.dot(thetahat) < to_NW.dot(rhat)) // both > 0
				{
					bFoundAnother = true;					
				} else {
					// We could still include it, for an acute triangle formed.
					to_left = pAux->pos-intermedpos;
					to_right = rightpos-intermedpos;
					bFoundAnother = (to_left.dot(to_right) > 0.0);
				}
				if (bFoundAnother) {
					to_intermed = intermedpos - pAux->pos;
					to_NE.x = to_NW.y;
					to_NE.y = -to_NW.x;
					if (to_intermed.dot(to_NE) > 0.0) {
						// above the line between the two others.
						bFoundAnother = false;
					}
				}
				if (bFoundAnother) {
					pTri=SetAuxTri(iLevel,index[iVertex],curtain[iIntermed],curtain[iRight]);
					pTri->periodic = 0;
					if (iIndex == curtain_length-1) pTri->periodic = 2;
					if (iIndex == curtain_length-2) pTri->periodic = 1;

					// remove point from curtain and wind back around:

					if (iRight != 0) {
						memmove(curtain+iIntermed,curtain+iRight,
								sizeof(long)*(curtain_length-iRight));
					};
					// If iIntermed == 5, iRight == 6 then this works
					// if iIntermed == 0, it works
					// if iIntermed == max-1, it's not right

					// ordinarily this point that got extinguished is to the right of us
					// in the array. But if we are at the last point, it can be at 0.
					if (iIndex == curtain_length-1) {
						iIndex--; // 0 point was removed.
					};
					curtain_length--;
				};

			} while (bFoundAnother);
			
			++pAux;
		}
		
		// Note that it's impossible to connect to a point at its own 
		// radius by looking back at 45 degrees, so what do we do when
		// this is no longer happening?
		// What will it look like in the end?
		// A sawtooth and both final rows are in the curtain. 
		// There can be other points too, because we did not backfill.
		// So we have to add another row of triangles:

		bool bContinue;
		do {
			iIndex = 0;
			bContinue = false;
			while (iIndex < curtain_length) {
				
				// If we destroy to the right of i, keep iIndex where it is
				// Question is whether we can draw a triangle here and find that it
				// is facing the right way.

				iIntermed = iIndex+1; if (iIntermed == curtain_length) iIntermed = 0;
				iRight = iIntermed+1; if (iRight == curtain_length) iRight = 0;

				pAux = AuxX[iLevel] + curtain[iIndex];
				pTest = AuxX[iLevel] + curtain[iRight];
				rightpos = pTest->pos;
				if (iIndex >= curtain_length-2) rightpos = Clockwise*rightpos;

				pIntermed = AuxX[iLevel] + curtain[iIntermed];
				intermedpos = pIntermed->pos;
				if (iIndex == curtain_length-1) intermedpos = Clockwise*intermedpos;

				to_NW = pAux->pos - rightpos;
				to_intermed = intermedpos - pAux->pos;
				to_NE.x = to_NW.y;
				to_NE.y = -to_NW.x;
				if (to_intermed.dot(to_NE) < 0.0) {
					// It lies below the line.					
					pTri = SetAuxTri(iLevel,curtain[iIndex],curtain[iIntermed],curtain[iRight]);
					pTri->periodic = 0;
					if (iIndex == curtain_length-1) pTri->periodic = 2;
					if (iIndex == curtain_length-2) pTri->periodic = 1;

					if (iRight != 0) {
						memmove(curtain+iIntermed,curtain+iRight,
								sizeof(long)*(curtain_length-iRight));
					};
					// ordinarily this point that got extinguished is to the right of us
					// in the array. But if we are at the last point, it can be at 0.
					if (iIndex == curtain_length-1) {
						iIndex--; // 0 point was removed.
					};
					curtain_length--;

					bContinue = true;
				} else {
					// If no tri, move on.
					iIndex++;
				};
			};
		} while (bContinue);
		
		// Perhaps want to check here for debugging that every remaining point
		// is Outermost and every Outermost point is accounted for.
		if (curtain_length != numLastRowAux[iLevel]) {
			printf("\n\ncurtain_length error.\n\n");
		}
		
		// HERE CREATE TRIANGLE NEIGHBOUR LISTS OF 3:
		for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
		{
			pTri = AuxT[iLevel] + iTri;
			pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri,iLevel);
			pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri,iLevel);
			pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri,iLevel);
		};
		
		Vertex * pOpp;
		Vector2 u3, edge_normal, vec1, vec2, vec_along;
		int iprev, inext;

		for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
		{
			pTri = AuxT[iLevel] + iTri;
			// you share the off-vertices.
			pTri->MapLeftIfNecessary(u[0],u[1],u[2]);
			for (iNeigh = 0; iNeigh < 3; iNeigh++)
			{
				if (pTri->neighbours[iNeigh] != pTri) {
					// 1,2 shared
					iprev = iNeigh-1; if (iprev == -1) iprev = 2;
					inext = iNeigh+1; if (inext == 3) inext = 0;
					pOpp = pTri->neighbours[iNeigh]->ReturnUnsharedVertex(pTri);

					// make contiguous:
					u3 = pOpp->pos;
					if (pTri->periodic == 0) {
						if (pTri->neighbours[iNeigh]->periodic == 0) {
						} else {
							if ((u3.x > 0.0) && (pTri->cornerptr[0]->pos.x < 0.0))
								u3 = Anticlockwise*u3;
							if ((u3.x < 0.0) && (pTri->cornerptr[0]->pos.x > 0.0))
								u3 = Clockwise*u3;
						};
					} else {
						// pTri periodic
						if (u3.x > 0.0) u3 = Anticlockwise*u3;
					};
					
					vec1 = u[iNeigh]-u[iprev];
					vec2 = u3-u[iprev];
					vec_along = u[inext]-u[iprev];
					edge_normal.x = vec_along.y;
					edge_normal.y = -vec_along.x;
					if ((vec1.dot(edge_normal))*(vec2.dot(edge_normal)) > 0.0) {
						// overlap
						printf("overlap\n");
						getch();          // nothing detected
					};
				}
			};			
		};

		delete[] rrarray;
		delete[] index;

		printf("3.Auxiliary Delaunay flips.\n");
		

		// _____________________________
		// 3. Auxiliary Delaunay flips
		// _____________________________

		// This depends on computing circumcenter and seeing if
		// a neighbour point looking outwards, is closer than a corner.
		long numFlips;
		Vector2 cc;
		iPass = 0;
		do {
			numFlips = 0;
			pAuxTri = AuxT[iLevel];
			for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
			{
				// Try against each of its neighbours:
				for (iNeigh = 0; iNeigh < 3; iNeigh++)
				{
					if (pAuxTri->neighbours[iNeigh] != pAuxTri) {
						// edge exists, not edge of domain:
						pAuxTri->CalculateCircumcenter(cc, &pdistsq);
						
						// Find point q that is the unshared point of neighbour tri:
						pAuxTriNeigh = pAuxTri->neighbours[iNeigh]; // using triangle pointers
						pVertq = pAuxTriNeigh->ReturnUnsharedVertex(pAuxTri);
						qpos = pVertq->pos;
												
						// Make contiguous to pAuxTri:
						if (pAuxTri->periodic == 0) {
							// in this case p does not need rotating.
							if (pAuxTriNeigh->periodic == 0) {
								// q is OK also
							} else {
								// if neigh tri is periodic and this one not,
								// that guarantees that q is across PB. ?
								// But not which direction.

								// Mistake to use cc as indicator of our tri pos
								// because it can be
								// focused far on the other side of the domain.

								if ((pAuxTri->cornerptr[0]->pos.x > 0.0) && (qpos.x < 0.0))
								{
									qpos = Clockwise*qpos;
								} else {
									if ((pAuxTri->cornerptr[0]->pos.x < 0.0) && (qpos.x > 0.0)) {
										qpos = Anticlockwise*qpos;
									} else {
										printf("situation unexpected.\n");
										getch();
									};
								};
							};
						} else {
						//	if (p.x > 0.0) p = Anticlockwise*p;
							if (qpos.x > 0.0) qpos = Anticlockwise*qpos;
						};
						
						//vec_cc_p = p-cc;
						vec_cc_q = qpos-cc;
						if (//vec_cc_p.x*vec_cc_p.x + vec_cc_p.y*vec_cc_p.y >
							pdistsq*0.99999999999 > vec_cc_q.x*vec_cc_q.x + vec_cc_q.y*vec_cc_q.y) 
						{
							// Flip ... the new triangle goes {q,p1,corner[iNeigh]}
							// and {q,p2,corner[iNeigh]}
							
							printf("%d %d : Vertq = %d qdistsq pdistsq %1.13E %1.13E\n",
								pAuxTri-AuxT[0],pAuxTriNeigh-AuxT[0],
								pVertq-AuxX[0],
								vec_cc_q.x*vec_cc_q.x + vec_cc_q.y*vec_cc_q.y,
								pdistsq);

							// Maintaining what:  
							// periodic flag, corner list, neighbour list apparently
							// coarse vertex tri index list
							// 
							Flip(pAuxTri, pAuxTriNeigh, iLevel); // bool for whether aux

							numFlips++;
						}
					}
				}
				++pAuxTri;
			}
			printf("iPass %d numFlips %d \n",iPass, numFlips);
			
			for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
			{
				pTri = AuxT[iLevel] + iTri;
				// you share the off-vertices.
				pTri->MapLeftIfNecessary(u[0],u[1],u[2]);
				for (iNeigh = 0; iNeigh < 3; iNeigh++)
				{
					if (pTri->neighbours[iNeigh] != pTri) {
						// 1,2 shared
						iprev = iNeigh-1; if (iprev == -1) iprev = 2;
						inext = iNeigh+1; if (inext == 3) inext = 0;
						pOpp = pTri->neighbours[iNeigh]->ReturnUnsharedVertex(pTri);

						// make contiguous:
						u3 = pOpp->pos;
						if (pTri->periodic == 0) {
							if (pTri->neighbours[iNeigh]->periodic == 0) {
							} else {
								if ((u3.x > 0.0) && (pTri->cornerptr[0]->pos.x < 0.0))
									u3 = Anticlockwise*u3;
								if ((u3.x < 0.0) && (pTri->cornerptr[0]->pos.x > 0.0))
									u3 = Clockwise*u3;
							};
						} else {
							// pTri periodic
							if (u3.x > 0.0) u3 = Anticlockwise*u3;
						};
						
						vec1 = u[iNeigh]-u[iprev];
						vec2 = u3-u[iprev];
						vec_along = u[inext]-u[iprev];
						edge_normal.x = vec_along.y;
						edge_normal.y = -vec_along.x;
						if ((vec1.dot(edge_normal))*(vec2.dot(edge_normal)) > 0.0) {
							// overlap
							printf("overlap\n");
							getch();          // nothing detected
						};
					}
				};			
			};

			iPass++;
		} while (numFlips > 0);

		printf("numAuxTriangles[%d] %d\n",iLevel,numAuxTriangles[iLevel]);


		// Now create, for reference in creating the next level, "auxneigh" lists
		// which will allow us to know what is actually geometrically next to something else.

		pVertex = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			pVertex->ClearAuxNeighs();
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				pTri = AuxT[iLevel]+izTri[i];
				if (pTri->cornerptr[0] != pVertex) pVertex->AddNeighbourIfNecessaryAux(pTri->cornerptr[0]-AuxX[iLevel]);
				if (pTri->cornerptr[1] != pVertex) pVertex->AddNeighbourIfNecessaryAux(pTri->cornerptr[1]-AuxX[iLevel]);
				if (pTri->cornerptr[2] != pVertex) pVertex->AddNeighbourIfNecessaryAux(pTri->cornerptr[2]-AuxX[iLevel]);
			} // any sequence.
			++pVertex;
		}

		printf("4. Create coarse weights.\n");

		// ( Destroy any connections with weight < 1% ? )

		// This requires being able to determine which coarse triangle we are within,
		// which requires testing whether we belong to each AuxTriangle of neighbouring selected.
		
		// Therefore we need : 
		// : AuxTriangle::corner index
		// : AuxTriangle::neighbour index
		//						which we should maintain during Delaunay flips.
		// : Each coarse point needs an index list of AuxTriangles
		// : AuxTriangle::Periodic flag
		
		// Each fine point determines which triangle it inhabits; then it is easy
		// (take into account PB!) to find the interpolation coefficients.
		
		pAuxTri = AuxT[iLevel];
		for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
		{
			pAuxTri->RecalculateEdgeNormalVectors(false);
			++pAuxTri;
		}

		real coefflocal[4][4];
		Vertex * pVertSel;
		long iIndex;
		pVertex = Xarray;
		bool bDefault;
		real weightsum[4];
		int dbgctr = 0;

		for (iVertex = 0; iVertex < numFine; iVertex++)
		{	
			if (pVertex->iVolley != 1) {
				// pick a neighbour that is selected and try all its triangles:
	
				// PLANAR_WEIGHTS -- let's try this. 1/dist or 1/distsq also valid.

				// Change back to single, symmetric weight.
				//



				//bDefault = false;
				//	
				//if (iWeightSwitch == COEFFS_OF_GEOMETRIC_NEIGH)
				//{
				//	
				//	pVertex->ClearCoarseIndexList();
				//	// check geometric neighs:
				//	// if iLevel == -1, they are just the neighs.

				//	if (iLevel == 0) {
				//		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				//	} else {
				//		neigh_len = pVertex->GetAuxNeighIndexArray(izNeigh);
				//	};
				//	tri_len = pVertex->GetTriIndexArray(izTri);

				//	memset(weightsum,0,sizeof(real)*4);
				//	for (i = 0; i < neigh_len; i++)
				//	{
				//		pNeigh = Xarray + izNeigh[i];
				//		if (pNeigh->iVolley == 1) 
				//		{
				//			pVertex->GetCoefficients(&(coefflocal[0][0]), i);
				//			pVertex->AddUniqueCoarse(pNeigh->iCoarseIndex[0]);
				//			iIndex = pVertex->coarse_len-1;
				//			pVertex->wt_var[iIndex][PHI] = coefflocal[GAUSS][PHI]; 
				//			// phi should be used to govern Gauss since A_self is unimportant.
				//			// but for now we only look at the impact of phi_coarse not A_coarse.
				//			pVertex->wt_var[iIndex][AX] = coefflocal[AMPX][AX];
				//			pVertex->wt_var[iIndex][AY] = coefflocal[AMPY][AY];
				//			pVertex->wt_var[iIndex][AZ] = coefflocal[AMPZ][AZ];

				//			pVertex->PBC_uplink[iIndex] = 0;

				//			// "coarse eps_Gauss" = sum of [d eps_fine / dphi_centre]
				//			pVertex->wt_eps[iIndex][GAUSS][GAUSS] = coefflocal[GAUSS][PHI]; // coeff here on neigh phi = +ve
				//			pVertex->wt_eps[iIndex][GAUSS][AMPX] = coefflocal[AMPX][PHI];
				//			pVertex->wt_eps[iIndex][GAUSS][AMPY] = coefflocal[AMPY][PHI];
				//			pVertex->wt_eps[iIndex][GAUSS][AMPZ] = coefflocal[AMPZ][PHI];

				//			pVertex->wt_eps[iIndex][AMPX][GAUSS] = coefflocal[GAUSS][AX];
				//			pVertex->wt_eps[iIndex][AMPX][AMPX] = coefflocal[AMPX][AX];
				//			pVertex->wt_eps[iIndex][AMPX][AMPY] = coefflocal[AMPY][AX];
				//			pVertex->wt_eps[iIndex][AMPX][AMPZ] = coefflocal[AMPZ][AX];

				//			pVertex->wt_eps[iIndex][AMPY][GAUSS] = coefflocal[GAUSS][AY];
				//			pVertex->wt_eps[iIndex][AMPY][AMPX] = coefflocal[AMPX][AY];
				//			pVertex->wt_eps[iIndex][AMPY][AMPY] = coefflocal[AMPY][AY];
				//			pVertex->wt_eps[iIndex][AMPY][AMPZ] = coefflocal[AMPZ][AY];

				//			pVertex->wt_eps[iIndex][AMPZ][GAUSS] = coefflocal[GAUSS][AZ];
				//			pVertex->wt_eps[iIndex][AMPZ][AMPX] = coefflocal[AMPX][AZ];
				//			pVertex->wt_eps[iIndex][AMPZ][AMPY] = coefflocal[AMPY][AZ];
				//			pVertex->wt_eps[iIndex][AMPZ][AMPZ] = coefflocal[AMPZ][AZ];
				//			// if we do this way, go down and remove minus below.
		
				//			// Okay ... we don't know that coefficients are symmetric?
				//			// Shall we lookup the coefficient that neigh puts on phi here??

				//			if ((pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y)
				//			 && (pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y))
				//			{
				//				// Search and see if the triangle containing them both is periodic.
				//				// If so we need to set the PBC_uplink flag.
				//				// Hopefully geometric neighbours give no wrap-around of connection.

				//				int iWhich = -1;
				//				do {
				//					iWhich++;
				//					pTri = Tarray + izTri[iWhich];
				//				} while (pTri->has_vertex(pNeigh) == false);

				//				if (pTri->periodic == 0) {
				//					pVertex->PBC_uplink[iIndex] = 0;
				//				} else {
				//					pVertex->PBC_uplink[iIndex] = ANTICLOCKWISE; 
				//					// eps_fine would rotate ANTI to apply to eps_coarse.

				//					// Now in this case let's be clear: we will be saying
				//					// eps_coarse += weight[][AY] * (Anticlockwise.yx * Ax + Anticlockwise.yy*Ay)

				//					//pVertex->weight[iIndex][AMPX] = coefflocal[AMPX][AX]/Anticlockwise.yy;
				//					// the true coefficient on Ax contiguous will be that...
				//					//pVertex->weight[iIndex][AMPY] = coefflocal[AMPY][AY]/Anticlockwise.yy;
				//					
				//					// or take hypotenuse:
				//					pVertex->wt_var[iIndex][AMPX] = 
				//						sqrt(coefflocal[AMPX][AX]*coefflocal[AMPX][AX] + coefflocal[AMPX][AY]*coefflocal[AMPX][AY]);
				//					pVertex->wt_var[iIndex][AMPY] = 
				//						sqrt(coefflocal[AMPY][AX]*coefflocal[AMPY][AX] + coefflocal[AMPY][AY]*coefflocal[AMPY][AY]);
				//					
				//	//pCoarseVertex->coeff_self[AMPY][UNITY] += Anticlockwise.yx*wt[AMPY]*pAux->epsilon[AMPX]
				//	//					 + Anticlockwise.yy*wt[AMPY]*pAux->epsilon[AMPY];
				//				};
				//			};
				//			if ((pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y) 
				//				&& (pNeigh->pos.x/pNeigh->pos.y > 0.5*GRADIENT_X_PER_Y))
				//			{
				//				// Search and see if the triangle containing them both is periodic.
				//				int iWhich = -1;
				//				do {
				//					iWhich++;
				//					pTri = Tarray + izTri[iWhich];
				//				} while (pTri->has_vertex(pNeigh) == false);

				//				if (pTri->periodic == 0) {
				//					pVertex->PBC_uplink[iIndex] = 0;
				//				} else {
				//					pVertex->PBC_uplink[iIndex] = CLOCKWISE;
				//					
				//					pVertex->wt_var[iIndex][AMPX] = 
				//						sqrt(coefflocal[AMPX][AX]*coefflocal[AMPX][AX] + coefflocal[AMPX][AY]*coefflocal[AMPX][AY]);
				//					pVertex->wt_var[iIndex][AMPY] = 
				//						sqrt(coefflocal[AMPY][AX]*coefflocal[AMPY][AX] + coefflocal[AMPY][AY]*coefflocal[AMPY][AY]);
				//				};
				//			};
				//			weightsum[PHI] += pVertex->wt_var[iIndex][PHI];
				//			weightsum[AX] += pVertex->wt_var[iIndex][AX];
				//			weightsum[AY] += pVertex->wt_var[iIndex][AY];
				//			weightsum[AZ] += pVertex->wt_var[iIndex][AZ];
				//		};
				//		// I just can't see a good way to approach even this PB issue via 4x4 matrix. 
				//		// Therefore we have to stick with vector and indication of where PBC applies.

				//	}; // next i 

				//	for (ii = 0; ii < pVertex->coarse_len; ii++)
				//	{						
				//		if (weightsum[PHI] != 0.0) 
				//			pVertex->wt_var[ii][PHI] /= weightsum[PHI]; 
				//		// Sometimes weightsum comes out as 0
				//		// How can that be?

				//		if ( weightsum[AX] != 0.0) 
				//			pVertex->wt_var[ii][AX] /= weightsum[AX];
				//		if ( weightsum[AY] != 0.0)
				//			pVertex->wt_var[ii][AY] /= weightsum[AY];
				//		if (weightsum[AZ] != 0.0)
				//			pVertex->wt_var[ii][AZ] /= weightsum[AZ];
				//	};

				//	if (pVertex->coarse_len < 2) {
				//		//bDefault = true;
				//		// ? we still default to planar from tri in case that there is only 1 selected connected.

				//		// Try a different way:
				//		// Just connect to non-geometric neighbours.
				//		// We should above have ensured that we have some.

				//		// The geometric one still has priority.



				//		// Hmm I could be gettin confused.

				//		// Recall that having just 2 neighbours involved can lead to dups anyway
				//		// And who likes dups.

				//		// Maybe better to always get to 3. Although it could mean deepening.
				//		// Try various things.
				//		
				//	};
				//}; // iWeightSwitch
				if (iWeightSwitch == PLANAR_WEIGHTS)// || (bDefault))
				{
					real beta[3];

					pVertex->ClearCoarseIndexList();
					// Using coarse vertex:
					pVertSel = AuxX[iLevel] + (Xarray + pVertex->iCoarseIndex[0])->iCoarseIndex[0];
					tri_len = pVertSel->GetTriIndexArray(izTri);
					i = 0;
					while ((i < tri_len) && 
						((AuxT[iLevel]+izTri[i])->ContainsPoint(pVertex->pos.x,pVertex->pos.y) == 0))
						i++;
					if (i == tri_len) {

					//	printf("iLevel %d izTri[0] %d x y %1.9E %1.9E r %1.9E %1.5E %1.5E\n",iLevel,izTri[0],
					//		pVertex->pos.x,pVertex->pos.y,pVertex->pos.modulus(),
					//		pVertex->pos.modulus()-DOMAIN_OUTER_RADIUS,
					//		INNER_A_BOUNDARY-pVertex->pos.modulus());

					//	getch();
						
				//		if (izTri[0] == 18862) {
				//			dbgctr++;
				//			if (dbgctr == 2) globaldebugswitch = 1;
				//		}
						pTri = ReturnPointerToTriangleContainingPoint(
										AuxT[iLevel]+izTri[0],  // seed for beginning triangle search
										pVertex->pos.x,pVertex->pos.y); 
						
						// Whether this works depends on if it is within one of them.
						// It might be outside in edge of domain.
						// In that case?
					
					} else {
						pTri = AuxT[iLevel]+izTri[i];
					};
					// Now do a test? we should be able to find every iCoarseIndex from adjacent points, 
					// as corners of the triangle. Correct? No.
					// Therefore we cannot assume that it belongs to the first coarse point we tried either.

					// From default of the other approaches or because iWeightSwitch == PLANAR_WEIGHTS

					pVertex->coarse_len = 3;
					pVertex->iCoarseIndex[0] = pTri->cornerptr[0]-AuxX[iLevel];
					pVertex->iCoarseIndex[1] = pTri->cornerptr[1]-AuxX[iLevel];
					pVertex->iCoarseIndex[2] = pTri->cornerptr[2]-AuxX[iLevel];

					pTri->MapLeftIfNecessary(u[0],u[1],u[2]);
					if ((pVertex->pos.x > 0.0) && (pTri->periodic != 0))
					{
						u[0] = Clockwise*u[0]; u[1] = Clockwise*u[1]; u[2] = Clockwise*u[2];
					};
					
					if (pTri->periodic == 0) {
						pVertex->PBC_uplink[0] = 0;
						pVertex->PBC_uplink[1] = 0;
						pVertex->PBC_uplink[2] = 0;
					} else {
						if (pVertex->pos.x < 0.0) {
							// if ((pNeigh->pos.x > 0.0) && (pVertex->pos.x < 0.0)) PBC = CLOCKWISE;
							if (pTri->cornerptr[0]->pos.x < 0.0) {
								pVertex->PBC_uplink[0] = 0;
							} else {
								pVertex->PBC_uplink[0] = CLOCKWISE; // apparently, that our point rotates clockwise to apply.
								// Let's check that's how it is used.
							};
							if (pTri->cornerptr[1]->pos.x < 0.0) {
								pVertex->PBC_uplink[1] = 0;
							} else {
								pVertex->PBC_uplink[1] = CLOCKWISE; // apparently, that our point rotates clockwise to apply.
								// Let's check that's how it is used.
							};
							if (pTri->cornerptr[2]->pos.x < 0.0) {
								pVertex->PBC_uplink[2] = 0;
							} else {
								pVertex->PBC_uplink[2] = CLOCKWISE; // apparently, that our point rotates clockwise to apply.
								// Let's check that's how it is used.
							};
						} else {
							if (pTri->cornerptr[0]->pos.x > 0.0) {
								pVertex->PBC_uplink[0] = 0;
							} else {
								pVertex->PBC_uplink[0] = ANTICLOCKWISE;
							};
							if (pTri->cornerptr[1]->pos.x > 0.0) {
								pVertex->PBC_uplink[1] = 0;
							} else {
								pVertex->PBC_uplink[1] = ANTICLOCKWISE;
							};
							if (pTri->cornerptr[2]->pos.x > 0.0) {
								pVertex->PBC_uplink[2] = 0;
							} else {
								pVertex->PBC_uplink[2] = ANTICLOCKWISE;
							};
						};
					};
					// Planar coefficients:
					::GetInterpolationCoefficients(beta, pVertex->pos.x,pVertex->pos.y,u[0],u[1],u[2]);

					// Note: these sum to 1.
#ifndef FUNKYWEIGHTS
					// Same wt_vars for all equations:
					pVertex->weight[0] = beta[0];
					pVertex->weight[1] = beta[1];
					pVertex->weight[2] = beta[2];
#else
					pVertex->wt_var[0][0] = beta[0];
					pVertex->wt_var[0][1] = beta[0];
					pVertex->wt_var[0][2] = beta[0];
					pVertex->wt_var[0][3] = beta[0];
					pVertex->wt_var[1][0] = beta[1];
					pVertex->wt_var[1][1] = beta[1];
					pVertex->wt_var[1][2] = beta[1];
					pVertex->wt_var[1][3] = beta[1];					
					pVertex->wt_var[2][0] = beta[2];
					pVertex->wt_var[2][1] = beta[2];
					pVertex->wt_var[2][2] = beta[2];
					pVertex->wt_var[2][3] = beta[2];
#endif

				};

			} else {
				// coarse index?? Already changed. ? -- CHECK?
				pVertex->PBC_uplink[0] = 0;

#ifndef FUNKYWEIGHTS
				pVertex->weight[0] = 1.0;
				
#else
				// pVertex->iVolley == 1
				pVertex->wt_var[0][0] = 1.0;
				pVertex->wt_var[0][1] = 1.0;
				pVertex->wt_var[0][2] = 1.0;
				pVertex->wt_var[0][3] = 1.0;
				
				// "coarse Gauss" = "sum [(rate of change wrt phi) * eps]"
				pVertex->wt_eps[0][GAUSS][GAUSS] = -pVertex->coeff_self[GAUSS][PHI]; // -ve: coeff here on own phi
				pVertex->wt_eps[0][GAUSS][AMPX] = -pVertex->coeff_self[AMPX][PHI];
				pVertex->wt_eps[0][GAUSS][AMPY] = -pVertex->coeff_self[AMPY][PHI];
				pVertex->wt_eps[0][GAUSS][AMPZ] = -pVertex->coeff_self[AMPZ][PHI];
				
				pVertex->wt_eps[0][AMPX][GAUSS] = -pVertex->coeff_self[GAUSS][AX]; 
				pVertex->wt_eps[0][AMPX][AMPX] = -pVertex->coeff_self[AMPX][AX];
				pVertex->wt_eps[0][AMPX][AMPY] = -pVertex->coeff_self[AMPY][AX];
				pVertex->wt_eps[0][AMPX][AMPZ] = -pVertex->coeff_self[AMPZ][AX];

				pVertex->wt_eps[0][AMPY][GAUSS] = -pVertex->coeff_self[GAUSS][AY]; 
				pVertex->wt_eps[0][AMPY][AMPX] = -pVertex->coeff_self[AMPX][AY];
				pVertex->wt_eps[0][AMPY][AMPY] = -pVertex->coeff_self[AMPY][AY];
				pVertex->wt_eps[0][AMPY][AMPZ] = -pVertex->coeff_self[AMPZ][AY];

				pVertex->wt_eps[0][AMPZ][GAUSS] = -pVertex->coeff_self[GAUSS][AZ]; 
				pVertex->wt_eps[0][AMPZ][AMPX] = -pVertex->coeff_self[AMPX][AZ];
				pVertex->wt_eps[0][AMPZ][AMPY] = -pVertex->coeff_self[AMPY][AZ];
				pVertex->wt_eps[0][AMPZ][AMPZ] = -pVertex->coeff_self[AMPZ][AZ];
				// If we change to neigh coeff on here, above, then moot whether sign changes here.
#endif

			}

			// Now we need to set up the 4x4 matrices wt_eps.
			// We shall assume that they involve the same influence/d points as wt_var.

			// To begin with let's try:

			// wt_eps to look like coefficients, unreweighted.
			// minus on coeff_self.
			
			// OKAY
			// Not sure what is going on

			// Try rebalancing all coarse epsilon so that they come from a 
			// total weight 1 on fine epsilon:

			++pVertex;
		}
#ifdef FUNKYWEIGHTS

		real total[4];
		long iCoarse;
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			total[0] = 0.0;
			total[1] = 0.0;
			total[2] = 0.0;
			total[3] = 0.0;
			// How to find contributors? Are they all neighbours of the fine point that gave rise to it?
			// OK, let's scroll through fine points, find those with iVolley == 1.
			if (pVertex->coarse_len == 1) {
				iCoarse = pVertex->iCoarseIndex[0];
				i = 0;
				total[0] += pVertex->wt_eps[i][0][0] + pVertex->wt_eps[i][0][1] + 
							  + pVertex->wt_eps[i][0][2] + pVertex->wt_eps[i][0][3];
				total[1] += pVertex->wt_eps[i][1][0] + pVertex->wt_eps[i][1][1] + 
							  + pVertex->wt_eps[i][1][2] + pVertex->wt_eps[i][1][3];
				total[2] += pVertex->wt_eps[i][2][0] + pVertex->wt_eps[i][2][1] + 
							  + pVertex->wt_eps[i][2][2] + pVertex->wt_eps[i][2][3];
				total[3] += pVertex->wt_eps[i][3][0] + pVertex->wt_eps[i][3][1] + 
							  + pVertex->wt_eps[i][3][2] + pVertex->wt_eps[i][3][3];
				// Search neighbour list for points with our coarse vertex in their list. 
				neigh_len = pVertex->GetNeighIndexArray(izNeigh);
				for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
				{
					pNeigh = Xarray + izNeigh[iNeigh];
					for (i = 0; i < pNeigh->coarse_len; i++)
						if (pNeigh->iCoarseIndex[i] == iCoarse) {
							// do something:
							
							total[0] += pNeigh->wt_eps[i][0][0] + pNeigh->wt_eps[i][0][1] + 
									  + pNeigh->wt_eps[i][0][2] + pNeigh->wt_eps[i][0][3];
							total[1] += pNeigh->wt_eps[i][1][0] + pNeigh->wt_eps[i][1][1] + 
									  + pNeigh->wt_eps[i][1][2] + pNeigh->wt_eps[i][1][3];
							total[2] += pNeigh->wt_eps[i][2][0] + pNeigh->wt_eps[i][2][1] + 
									  + pNeigh->wt_eps[i][2][2] + pNeigh->wt_eps[i][2][3];
							total[3] += pNeigh->wt_eps[i][3][0] + pNeigh->wt_eps[i][3][1] + 
									  + pNeigh->wt_eps[i][3][2] + pNeigh->wt_eps[i][3][3];
						};
				};

				// Now go again and divide them, including our own.

				pVertex->wt_eps[0][0][0] /= total[0];
				pVertex->wt_eps[0][0][1] /= total[0];
				pVertex->wt_eps[0][0][2] /= total[0];
				pVertex->wt_eps[0][0][3] /= total[0];
				pVertex->wt_eps[0][1][0] /= total[1];
				pVertex->wt_eps[0][1][1] /= total[1];
				pVertex->wt_eps[0][1][2] /= total[1];
				pVertex->wt_eps[0][1][3] /= total[1];
				pVertex->wt_eps[0][2][0] /= total[2];
				pVertex->wt_eps[0][2][1] /= total[2];
				pVertex->wt_eps[0][2][2] /= total[2];
				pVertex->wt_eps[0][2][3] /= total[2];
				pVertex->wt_eps[0][3][0] /= total[3];
				pVertex->wt_eps[0][3][1] /= total[3];
				pVertex->wt_eps[0][3][2] /= total[3];
				pVertex->wt_eps[0][3][3] /= total[3];
				
				for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
				{
					pNeigh = Xarray + izNeigh[iNeigh];
					for (i = 0; i < pNeigh->coarse_len; i++)
						if (pNeigh->iCoarseIndex[i] == iCoarse) {
							// do something:
							pNeigh->wt_eps[i][0][0] /= total[0];
							pNeigh->wt_eps[i][0][1] /= total[0];
							pNeigh->wt_eps[i][0][2] /= total[0];
							pNeigh->wt_eps[i][0][3] /= total[0];
							pNeigh->wt_eps[i][1][0] /= total[1];
							pNeigh->wt_eps[i][1][1] /= total[1];
							pNeigh->wt_eps[i][1][2] /= total[1];
							pNeigh->wt_eps[i][1][3] /= total[1];
							pNeigh->wt_eps[i][2][0] /= total[2];
							pNeigh->wt_eps[i][2][1] /= total[2];
							pNeigh->wt_eps[i][2][2] /= total[2];
							pNeigh->wt_eps[i][2][3] /= total[2];
							pNeigh->wt_eps[i][3][0] /= total[3];
							pNeigh->wt_eps[i][3][1] /= total[3];
							pNeigh->wt_eps[i][3][2] /= total[3];
							pNeigh->wt_eps[i][3][3] /= total[3];
						};
				};
				
			};
			++pVertex;
		};
#endif

#else


		
		real weight[12],weightsum;
		int iMin;
		real d, minweight;
		Vertex * pCoarseVertex,*pAux;
		long iVertex;

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if (pVertex->coarse_len == 1) {
				pVertex->weight[0] = 1.0;
			} else {

				weightsum = 0.0;
				for (i = 0; i < pVertex->coarse_len; i++)
				{
					pCoarseVertex = Xarray + pVertex->GetCoarseIndex(i);
					if (QUADWEIGHTS) {
						d = GetPossiblyPeriodicDistSq(pCoarseVertex->centroid,pVertex->centroid);
					} else {
						d = GetPossiblyPeriodicDist(pCoarseVertex->centroid,pVertex->centroid);
					};

					// simple weights: 1 / distance
					weight[i] = 1.0/d;
					if (d == 0.0) {
						printf("!d == 0.0!");
					}
					// I think we'd prefer planar linear weights, esp if there are 3 influences -
					// but if there are 2 perhaps we should allow to be influenced by 3 and do Del tris.

					weightsum += weight[i];
				}
				for (i = 0; i < pVertex->coarse_len; i++)
					weight[i] /= weightsum; // all now < 1.0
				
				// keep biggest COARSE_LIMIT weights...:
				while ( pVertex->coarse_len > COARSE_LIMIT)
				{
					iMin = -1;
					minweight = 10.0;
					for (i = 0; i < pVertex->coarse_len; i++)
					{
						if (weight[i] < minweight) {
							minweight = weight[i];
							iMin = i;
						}
					}
					for (i = iMin; i < pVertex->coarse_len-1; i++)
						weight[i] = weight[i+1];
					pVertex->RemoveCoarseIndexIfExists(pVertex->GetCoarseIndex(iMin));
				};				
				weightsum = 0.0;
				for (i = 0; i < pVertex->coarse_len; i++)
					weightsum += weight[i];
				for (i = 0; i < pVertex->coarse_len; i++)
					weight[i] /= weightsum; 
				
				// USUALLY the amount of weights should be 1, 2 or 3, in rough proportions 1:4:1 .
				// There is an argument for restricting to 3 or 6 weights to reduce network deepening.
				// But be sure that a vertex always affects its own immediate neighbours.
				
				for (i = 0; i < pVertex->coarse_len; i++)
					pVertex->weight[i] = weight[i];
			};
			++pVertex;
		};

		// --------------------------------------------------



		char PBC,PBC2;
		long coarse_len, tri_len, tri_len2, izTri2[128],
			izTri[128];
		Triangle * pTri,*pTri2;

		// PBC_uplink flag:
		if (iLevel == 0) {
			pVertex = Xarray;
			for (iVertex= 0; iVertex < numFine; iVertex++)
			{
				if (pVertex->iVolley != 1) {
					// for selected ones it would make no sense.
					
					// Go through neighs.
					// If one of these IS a coarse link, can decide PBC from that
					// Otherwise look for it in 2nd neighbours; what is PBC then?
					
					coarse_len = pVertex->coarse_len;
					neigh_len = pVertex->GetNeighIndexArray(izNeigh);
					tri_len = pVertex->GetTriIndexArray(izTri);
					
					memset(pVertex->PBC_uplink,'a',coarse_len);
					// Set character 10 so we can go back over.
					
					for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						// First, what is relative rotation of this neighbour?
						// Should find that tri 0 has neighbour 0;
						// At edge of memory, have to stick with tri 3 for neighbour 4.
						
						if (iNeigh < tri_len) pTri = T + izTri[iNeigh];
						
						if (pTri->periodic == 0) {
							PBC = 0;
						} else {
							PBC = 0;
							if ((pNeigh->pos.x > 0.0) && (pVertex->pos.x < 0.0)) PBC = CLOCKWISE;
							if ((pNeigh->pos.x < 0.0) && (pVertex->pos.x > 0.0)) PBC = ANTICLOCKWISE;
						};
						
						// Is it a coarse link?
						for (i = 0; i < coarse_len; i++)
						{
							if (pVertex->iCoarseIndex[i] == izNeigh[iNeigh]) 
								// here iCoarseIndex does not yet index into the level above.
							{
								pVertex->PBC_uplink[i] = PBC; // the neighbour relative to this.
								i = coarse_len; // break out
							};
						};

#ifndef SPACING1
						// Is it on the way to a coarse link?
						// Check neighbours:
						neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
						tri_len2 = pNeigh->GetTriIndexArray(izTri2);
						for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
						{
							pNeigh2 = Xarray + izNeigh2[iNeigh2];
							
							if (iNeigh2 < tri_len2) pTri2 = T + izTri2[iNeigh2];
							if (pTri2->periodic == 0) {
								PBC2 = 0;
							} else {
								PBC2 = 0;
								if ((pNeigh2->pos.x > 0.0) && (pNeigh->pos.x < 0.0)) PBC2 = CLOCKWISE;
								if ((pNeigh2->pos.x < 0.0) && (pNeigh->pos.x > 0.0)) PBC2 = ANTICLOCKWISE;
							};
							// Amalgamate PBC,PBC2 to get relation to pVertex.
							// According to:
							//   PBC   PBC2   result
							//     0      0      0
							//     0      1      1
							//    -1      1      0
							// so, PBC + PBC2
							
							for (i = 0; i < coarse_len; i++)
							{
								if (pVertex->iCoarseIndex[i] == izNeigh2[iNeigh2]) 
								{
									pVertex->PBC_uplink[i] = PBC + PBC2;
									i = coarse_len; // break out
								};
							};
						}; // next iNeigh2
#endif
					}; // next iNeigh

					for (i = 0; i < coarse_len; i++)
					{
						if (pVertex->PBC_uplink[i] == 'a') {
							// try to guess
							pNeigh = Xarray + pVertex->iCoarseIndex[i];
							pVertex->PBC_uplink[i] = 0;
							if ((pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y)
								&&
								(pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y))
								pVertex->PBC_uplink[i] = ANTICLOCKWISE;
							if ((pNeigh->pos.x/pNeigh->pos.y > 0.5*GRADIENT_X_PER_Y)
								&&
								(pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y))
								pVertex->PBC_uplink[i] = CLOCKWISE;
						};
					};

				} else {
					// A selected vertex.
					pVertex->PBC_uplink[0] = 0;
				}
				++pVertex;
			};
		} else {
		
			// iLevel > 0. Still need to set pVertex->PBC_link at finer level
			pVertex = Xarray;
			for (iVertex= 0; iVertex < numFine; iVertex++)
			{
				if (pVertex->iVolley != 1) {
					// for selected ones it would make no sense.

					// Go through neighs.
					// If one of these IS a coarse link, can decide PBC from that
					// Otherwise look for it in 2nd neighbours; what is PBC then?

					coarse_len = pVertex->coarse_len;
					neigh_len = pVertex->GetNeighIndexArray(izNeigh);
					
					memset(pVertex->PBC_uplink,'a',coarse_len);

					for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
					{
						pNeigh = Xarray + izNeigh[iNeigh];
						PBC = -pVertex->PBC[iNeigh];
						// pVertex relative to <-> pNeigh rel to pVertex is what we want

						// Is it a coarse link?
						for (i = 0; i < coarse_len; i++)
						{
							if (pVertex->iCoarseIndex[i] == izNeigh[iNeigh]) 
							{
								pVertex->PBC_uplink[i] = PBC; // the neighbour relative to this.
								i = coarse_len; // break out
							};
						};
#ifndef SPACING1
						// Is it on the way to a coarse link? Check neighbours:
						neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);						
						for (iNeigh2 = 0; iNeigh2 < neigh_len2; iNeigh2++)
						{
							pNeigh2 = Xarray + izNeigh2[iNeigh2];
							PBC2 = -pNeigh->PBC[iNeigh2]; // pNeigh2 rel to pNeigh
							
							for (i = 0; i < coarse_len; i++)
							{
								if (pVertex->iCoarseIndex[i] == izNeigh2[iNeigh2]) 
								{
									pVertex->PBC_uplink[i] = PBC + PBC2;
									i = coarse_len; // break out
								};
							};
						}; // next iNeigh2
#endif
					}; // next iNeigh
					
					for (i = 0; i < coarse_len; i++)
					{
						if (pVertex->PBC_uplink[i] == 'a') {
							// try to guess
							pNeigh = Xarray + pVertex->iCoarseIndex[i];
							pVertex->PBC_uplink[i] = 0;
							if ((pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y)
								&&
								(pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y))
								pVertex->PBC_uplink[i] = ANTICLOCKWISE;
							if ((pNeigh->pos.x/pNeigh->pos.y > 0.5*GRADIENT_X_PER_Y)
								&&
								(pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y))
								pVertex->PBC_uplink[i] = CLOCKWISE;
						};
					};
				} else {
					// A selected vertex.
					pVertex->PBC_uplink[0] = 0; // not rotated relative to itself
				}
				++pVertex;
			};
		};		

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			if (pVertex->iVolley != 1) {
				for (i = 0; i < pVertex->coarse_len; i++)
				{
					pVertex->iCoarseIndex[i] = (Xarray + pVertex->iCoarseIndex[i])->iCoarseIndex[0];
				};
			};
			++pVertex;
		};
#endif

		printf("5. Set aux neighs and Galerkin \n");

		// Create neighbour list during Galerkin coefficient assignment... :	
		//if (bJustAz) {
		//	Set_AuxNeighs_And_GalerkinCoefficientsAz(iLevel);// . Clear coarse neighs as part of this.
		//} else {
			Set_AuxNeighs_And_GalerkinCoefficients(iLevel);// . Clear coarse neighs as part of this.
		//};


		// MUST delete[] auxiliary vertex arrays at the end of solve.
		// Or, if we want, switch to realloc - but careful of those coefficient arrays, so delete is better and future-proof.

		//if (iLevel == 0) {
		//	FILE * file = fopen("weights.txt","w");

		//	printf("doing file output...");

		//	long iStart = 8400, iEnd = 12500;
		//	pVertex = X+iStart;
		//	for (iVertex = iStart; iVertex < iEnd; iVertex++)
		//	{
		//		fprintf(file,"%d %d %1.10E %1.10E | %d | ",
		//			iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y,pVertex->iVolley);
		//		for (int iEqn = 0; iEqn < 4; iEqn++)
		//		{
		//			for (i = 0; i < pVertex->coarse_len; i++)
		//				fprintf(file," %d %1.12E ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][iEqn]);
		//			fprintf(file," | ");
		//		};
		//		fprintf(file,"\n");
		//		pVertex++;
		//	}
		//	fclose(file);
		//	printf("done \n"); 
		//};
		//// Announce: Number in level and average # neighbours

		long maxneighs = 0;
		real Averageneighs = 0.0;
		pVertex = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			Averageneighs += (real)neigh_len;
			if (maxneighs < neigh_len) maxneighs = neigh_len;
			++pVertex;
		};
		Averageneighs /= (real)(numAuxVertices[iLevel]);

		printf("\nLevel %d numAuxVertices %d Avg#neighs %f maxneighs %d\n",
			iLevel,numAuxVertices[iLevel],Averageneighs,maxneighs);

		// For the finer level, what is distribution of # coarse influences ?

		long num[MAXCOARSE+1];
		memset(num,0,sizeof(long)*(MAXCOARSE+1));
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numFine; iVertex++)
		{
			num[pVertex->coarse_len]++;
			if (pVertex->coarse_len == 0) 
				printf("iVertex %d coarse_len==0 iVolley %d \n",iVertex,pVertex->iVolley);
			++pVertex;
		};
		printf("Lvl %d: 0: %d 1: %d 2: %d 3: %d 4: %d 5: %d 6: %d 7: %d 8: %d 9: %d 10: %d 11: %d 12: %d\n",
			iLevel-1, num[0], num[1],num[2],num[3],num[4],num[5],num[6],num[7],num[8],num[9],num[10],num[11],num[12]);

	}; // next iLevel
	printf("NumAuxVertices: [%d] ",numVertices);
	for (iLevel = 0; iLevel < NUM_COARSE_LEVELS; iLevel++)
	{
		printf("%d ",numAuxVertices[iLevel]);
	}
	printf("\nany key\n");
	getch();		
}

void TriMesh::Set_AuxNeighs_And_GalerkinCoefficients(int iLevel)
{
	long numFine;
	Vertex * Xarray, * pVertex, * pNeigh, *pNeigh2;
	char PBC,PBC2;
	int coarse_len,i;
	Triangle * pTri, * pTri2;
	long izNeigh[128],izNeigh2[128],neigh_len,neigh_len2,tri_len,tri_len2,
		izTri[128],izTri2[128],iNeigh, iVertex, iNeigh2;

	char rotatedest,rotatesrc;
	int iAffectedIndex, iAffectorIndex;
	real coefflocal[4][4];
	int iEqn, iVar;
	real factor[4];
	Vertex * pAffected, *pAffector;
	long iIndex;
	char rotateintermed;
	

	// Set a flag, then count how many.
	if (iLevel == 0) {
		Xarray = X;
		numFine = numVertices;
	} else {
		Xarray = AuxX[iLevel-1];
		// iLevel is the one we are filling in.
		numFine = numAuxVertices[iLevel-1];
	};
	
	// Idea is very simple: sweep through fine vertices and accumulate the coefficients.
	
	// . Clear all coefficient arrays and neighbour arrays:
	// [cf finest coeff routine]
	// ________________________________
	
	pVertex = AuxX[iLevel];
	for (iVertex= 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		pVertex->ClearNeighs();
		//pVertex->ZeroCoefficients(); // ClearNeighs does it
		pVertex->contrib_to_phi_avg = 0.0;
		++pVertex;
	};
	// Where we add neighbours in init, post-Flip, etc: do we know to clear neighs?
	
	
	// On fine level [if finest], pre-compute the rotations to the coarser vertices.
	
	
	// Need to have a way to know which neighbour links are periodic in auxiliary meshes.
	// As we add each new neighbour for auxiliary level in what follows, add a flag for periodic relation.
	
	// And for now just add an array of chars to Vertex object. = PBC
	
	
	// But, we do need to recognise that coarse levels have a different requirement for coefficient
	// array length: the way we end up doing it, there can be some deepening.
	// Handle coefficients how? We are going to do memcpy to actually access, correct?
	// Dynamic array may be best.
	// Will need to know how we are going to manage with GPU if we don't know how many rows deep for a
	// rectangular array containing coefficient[][] object columns.
	// There is no way around deepening.
	
	
	// Accumulate coefficients:
	// ________________________
	
	pVertex = Xarray;
	for (iVertex= 0; iVertex < numFine; iVertex++)
	{
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		tri_len = pVertex->GetTriIndexArray(izTri);
		
		for (iAffectedIndex = 0; iAffectedIndex < pVertex->coarse_len; iAffectedIndex++)
		{
			// Do we rotate the effect between eps fine and eps coarse?
			pAffected = AuxX[iLevel] + pVertex->iCoarseIndex[iAffectedIndex];
			rotatedest = pVertex->PBC_uplink[iAffectedIndex]; 
			// ANTI if Affected is anticlockwise.

			//if (iLevel == 0) {
			//	if (iVertex < numInnermostRow) 
			//		pAffected->contrib_to_phi_avg += pVertex->weight[iAffectedIndex];
			//} else {
			//	pAffected->contrib_to_phi_avg += pVertex->contrib_to_phi_avg*pVertex->weight[iAffectedIndex];
			//}
			
			// pVertex->coeff_self generates coarse-level connections:
			// _______________________________________________________
#ifdef FUNKYWEIGHTS
			Accumulate_coeffself_unary(
				pAffected,
				pVertex->wt_var[iAffectedIndex], // not sure this is right...
				pVertex->wt_eps[iAffectedIndex], // destination weight
				pVertex->coeff_self,
				rotatedest
				);  // Note that in fact, the [UNITY] coefficient gets wiped out.
#else
			Accumulate_coeffself_unary(
				pAffected,
				pVertex->weight[iAffectedIndex],
				pVertex->coeff_self,
				rotatedest
				);  // Note that in fact, the [UNITY] coefficient gets wiped out.
#endif

			for (iAffectorIndex = 0; iAffectorIndex < pVertex->coarse_len; iAffectorIndex++) 
			{
				pAffector = AuxX[iLevel] + pVertex->iCoarseIndex[iAffectorIndex];
				rotatesrc = -pVertex->PBC_uplink[iAffectorIndex];
				// so that rotation from src to dest is rotatesrc + rotatedest
				
				// Here for vector of weights to eps_coarse but we will want matrix.
				//factor[0] = pVertex->weight[iAffectorIndex][0]*pVertex->weight[iAffectedIndex][0];
				//factor[1] = pVertex->weight[iAffectorIndex][1]*pVertex->weight[iAffectedIndex][1];
				//factor[2] = pVertex->weight[iAffectorIndex][2]*pVertex->weight[iAffectedIndex][2];
				//factor[3] = pVertex->weight[iAffectorIndex][3]*pVertex->weight[iAffectedIndex][3];
				
				//factor = pVertex->weight[iAffectorIndex]*pVertex->weight[iAffectedIndex];
				
				for (iEqn = 0; iEqn < 4; iEqn++)
					memcpy(coefflocal[iEqn],pVertex->coeff_self[iEqn],sizeof(real)*NUM_AFFECTORS_1);
				
				if (pAffected == pAffector) {
					AccumulateCoefficients(
						pAffected, 
						-1,
			//			pVertex->wt_var[iAffectorIndex],
			//			pVertex->wt_eps[iAffectedIndex],
			pVertex->weight[iAffectorIndex],
			pVertex->weight[iAffectedIndex],
						coefflocal,
						rotatesrc,rotatedest,
						// for debug:
						iVertex, // index of phi_fine
						iVertex // index of intermediate
						); 
					
					// debug:
					

				} else {
					iIndex = pAffected->AddNeighbourIfNecessary(
						pVertex->iCoarseIndex[iAffectorIndex], rotatesrc + rotatedest);
					// when we write this, ensure if it exists, per flag is verified equal
					
					AccumulateCoefficients(
						pAffected, //pAffector, 
						//pVertex->iCoarseIndex[iAffectedIndex],
						iIndex,//pVertex->iCoarseIndex[iAffectorIndex],
					//	pVertex->wt_var[iAffectorIndex],
					//	pVertex->wt_eps[iAffectedIndex],
						pVertex->weight[iAffectorIndex],
						pVertex->weight[iAffectedIndex],
						coefflocal,
						rotatesrc,rotatedest,
						
						// for debug:
						iVertex, // index of phi_fine
						iVertex
						); 
				};
			};

			// rest of coefficients, ie pVertex affected by some pNeigh:
			// _________________________________________________________

			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pNeigh = Xarray + izNeigh[iNeigh];
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);

				for (iAffectorIndex = 0; iAffectorIndex < pNeigh->coarse_len; iAffectorIndex++) 
				{
					pAffector = AuxX[iLevel] + pNeigh->iCoarseIndex[iAffectorIndex];
					rotatesrc = -pNeigh->PBC_uplink[iAffectorIndex];
					// Now we need to decide if pVertex is rotated relative to pNeigh

					//factor = pNeigh->weight[iAffectorIndex]*pVertex->weight[iAffectedIndex];
				
					if (iLevel == 0) {

						if (iNeigh < tri_len) pTri = T + izTri[iNeigh]; // otherwise keep same
					
						if (pTri->periodic == 0) {
							rotateintermed = 0;
						} else {
							rotateintermed = 0;
							if ((pNeigh->pos.x > 0.0) && (pVertex->pos.x < 0.0)) rotateintermed = ANTICLOCKWISE;
							if ((pNeigh->pos.x < 0.0) && (pVertex->pos.x > 0.0)) rotateintermed = CLOCKWISE;
						};
						// ## pVertex is the destination and we note whether destination is anticlockwise. ##
						
					} else {						
						rotateintermed = pVertex->PBC[iNeigh];
						// rotation of fine eps rel to fine A vertex						
					}
						
					if (pAffected == pAffector)
					{
						AccumulateCoefficients(
							pAffected,	-1,
					//		pNeigh->wt_var[iAffectorIndex],
					//		pVertex->wt_eps[iAffectedIndex],

					pNeigh->weight[iAffectorIndex],
					pVertex->weight[iAffectedIndex],
							coefflocal,
							rotatesrc,rotatedest,
							
							pNeigh-Xarray,
							iVertex); 
					} else {
												
						iIndex = pAffected->AddNeighbourIfNecessary(
									pNeigh->iCoarseIndex[iAffectorIndex],
									rotatesrc + rotateintermed + rotatedest);
						
						AccumulateCoefficients(
							pAffected, //pAffector, 
									//pVertex->iCoarseIndex[iAffectedIndex],
							iIndex,//pNeigh->iCoarseIndex[iAffectorIndex],
							pNeigh->weight[iAffectorIndex],
							pVertex->weight[iAffectedIndex],
//							pNeigh->wt_var[iAffectorIndex],
//							pVertex->wt_eps[iAffectedIndex],
							coefflocal,
							rotatesrc,rotatedest,
							
							pNeigh-Xarray,
							iVertex); 
					};
					// OUR rotatesrc means rotation of fine A pt rel to coarse A pt;
					// OUR rotatedest means rotation of coarse eps pt rel to fine eps pt.
				};
			};
		};

		++pVertex;
	};


	// Debug output:

/*	real phicoeff, Axcoeff, Aycoeff, Azcoeff,
	total, total2;
	long  ii;
	char buffer[128];
	FILE * file;
	
	sprintf(buffer,"coeff_lvl%d.txt",iLevel);
	file = fopen(buffer,"w");
	pVertex = AuxX[iLevel];
	for (iVertex = 0;iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		fprintf(file,"%d %d %1.12E %1.12E | ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
		// Gauss equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[GAUSS][AY],
			pVertex->coeff_self[GAUSS][AZ],pVertex->coeff_self[GAUSS][UNITY],pVertex->coeff_self[GAUSS][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[GAUSS][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[GAUSS][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[GAUSS][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[GAUSS][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[GAUSS][AX])
			+ fabs(pVertex->coeff_self[GAUSS][AY]) + fabs(pVertex->coeff_self[GAUSS][AZ]);
		total2 = total + fabs(pVertex->coeff_self[GAUSS][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total);
		fprintf(file,"\n");
		

		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-x equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPX][PHI],pVertex->coeff_self[AMPX][AX],pVertex->coeff_self[AMPX][AY],
			pVertex->coeff_self[AMPX][AZ],pVertex->coeff_self[AMPX][UNITY],pVertex->coeff_self[AMPX][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPX][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPX][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPX][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPX][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPX][PHI])
			+ fabs(pVertex->coeff_self[AMPX][AY]) + fabs(pVertex->coeff_self[AMPX][AZ]);
		total2 = total + fabs(pVertex->coeff_self[AMPX][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPX][AX]/total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPX][AX]/total);
		fprintf(file,"\n");
		
		
		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-y equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPY][PHI],pVertex->coeff_self[AMPY][AX],pVertex->coeff_self[AMPY][AY],
			pVertex->coeff_self[AMPY][AZ],pVertex->coeff_self[AMPY][UNITY],pVertex->coeff_self[AMPY][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPY][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPY][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPY][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPY][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPY][PHI])
			+ fabs(pVertex->coeff_self[AMPY][AX]) + fabs(pVertex->coeff_self[AMPY][AZ]);
		total2 = total + fabs(pVertex->coeff_self[AMPY][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPY][AY]/total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPY][AY]/total);
		fprintf(file,"\n");
		
		
		fprintf(file,"%d %d --- --- | ",iVertex,pVertex->flags);
		// Amp-z equation coeffself:
		fprintf(file," %1.12E %1.12E %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[AMPZ][PHI],pVertex->coeff_self[AMPZ][AX],pVertex->coeff_self[AMPZ][AY],
			pVertex->coeff_self[AMPZ][AZ],pVertex->coeff_self[AMPZ][UNITY],pVertex->coeff_self[AMPZ][PHI_ANODE]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			phicoeff += fabs(pVertex->coeff[i].co[AMPZ][PHI]);
			Axcoeff += fabs(pVertex->coeff[i].co[AMPZ][AX]);
			Aycoeff += fabs(pVertex->coeff[i].co[AMPZ][AY]);
			Azcoeff += fabs(pVertex->coeff[i].co[AMPZ][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has row diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff + fabs(pVertex->coeff_self[AMPZ][PHI])
				 + fabs(pVertex->coeff_self[AMPZ][AX]) + fabs(pVertex->coeff_self[AMPZ][AY]);
		total2 = total + fabs(pVertex->coeff_self[AMPZ][PHI_ANODE]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total2) fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total) fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total);
		fprintf(file,"\n");
		
		
		
		fprintf(file,"%d %d  phi effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[AMPX][PHI],pVertex->coeff_self[AMPY][PHI],
			pVertex->coeff_self[AMPZ][PHI]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = AuxX[iLevel] + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while ((ii < neigh_len2) && (izNeigh2[ii] != iVertex))  ii++; // why is this throwing an error?
			if (ii == neigh_len2) {
				ii = ii;
			}
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][PHI]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][PHI]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][PHI]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][PHI]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[AMPX][PHI])
			+ fabs(pVertex->coeff_self[AMPY][PHI])
			+ fabs(pVertex->coeff_self[AMPZ][PHI]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total2);
		if (fabs(pVertex->coeff_self[GAUSS][PHI]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[GAUSS][PHI]/total);

		fprintf(file,"\n");
			

		// 

		fprintf(file,"%d %d  Ax effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[AMPX][AX],pVertex->coeff_self[AMPY][AX],
			pVertex->coeff_self[AMPZ][AX]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = AuxX[iLevel] + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AX]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AX]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AX]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AX]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AX])
			+ fabs(pVertex->coeff_self[AMPY][AX])
			+ fabs(pVertex->coeff_self[AMPZ][AX]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPX][AX]/total2);
		if (fabs(pVertex->coeff_self[AMPX][AX]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPX][AX]/total);
		fprintf(file,"\n");

		fprintf(file,"%d %d  Ay effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AY],pVertex->coeff_self[AMPX][AY],pVertex->coeff_self[AMPY][AY],
			pVertex->coeff_self[AMPZ][AY]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = AuxX[iLevel] + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AY]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AY]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AY]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AY]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AY])
			+ fabs(pVertex->coeff_self[AMPX][AY])
			+ fabs(pVertex->coeff_self[AMPZ][AY]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPY][AY]/total2);
		if (fabs(pVertex->coeff_self[AMPY][AY]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPY][AY]/total);
		fprintf(file,"\n");

		fprintf(file,"%d %d  Az effect | ",iVertex,pVertex->flags);
		fprintf(file," %1.12E %1.12E %1.12E %1.12E | ",
			pVertex->coeff_self[GAUSS][AZ],pVertex->coeff_self[AMPX][AZ],pVertex->coeff_self[AMPY][AZ],
			pVertex->coeff_self[AMPZ][AZ]);
		// Add up coefficients on other things: other phi, other Ax, etc
		phicoeff = 0.0;
		Axcoeff = 0.0;
		Aycoeff = 0.0;
		Azcoeff = 0.0;
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = AuxX[iLevel] + izNeigh[i];
			
			neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
			ii = 0;
			while (izNeigh2[ii] != iVertex) ii++;
				
			phicoeff += fabs(pNeigh->coeff[ii].co[GAUSS][AZ]);
			Axcoeff += fabs(pNeigh->coeff[ii].co[AMPX][AZ]);
			Aycoeff += fabs(pNeigh->coeff[ii].co[AMPY][AZ]);
			Azcoeff += fabs(pNeigh->coeff[ii].co[AMPZ][AZ]);
		};
		fprintf(file," %1.12E %1.12E %1.12E %1.12E |",phicoeff,Axcoeff,Aycoeff,Azcoeff);
		// Now decide if it has column diagonal dominance.
		total = phicoeff + Axcoeff + Aycoeff + Azcoeff
			+ fabs(pVertex->coeff_self[GAUSS][AZ])
			+ fabs(pVertex->coeff_self[AMPX][AZ])
			+ fabs(pVertex->coeff_self[AMPY][AZ]);
		total2 = total + fabs(pVertex->coeff_self[IZ][PHI]);
		fprintf(file," %1.12E %1.12E ",total,total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total2) 
			fprintf(file," vstotal2 %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total2);
		if (fabs(pVertex->coeff_self[AMPZ][AZ]) < total) 
			fprintf(file," vstotal %1.12E ",pVertex->coeff_self[AMPZ][AZ]/total);
		fprintf(file,"\n\n");

		++pVertex;
	};
	fclose(file);
	*/
}


void TriMesh::Accumulate_coeffself_unary(
	Vertex * pAffected, //long iAffected,
	real weight,
	//real wtsrc[4],real wtdest[4][4], 
	real coeff_self[NUM_EQNS_2][NUM_AFFECTORS_2], 
	char rotatedest)
{
	int iEqn;

	// Treating case that rotatedest does not apply and wtdest is [4][4].
#ifdef FUNKYWEIGHTS
	for (iEqn = 0; iEqn < 4; iEqn++) 
	{
		pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn][GAUSS]*coeff_self[GAUSS][UNITY]
											+ wtdest[iEqn][AMPX]*coeff_self[AMPX][UNITY]
											+ wtdest[iEqn][AMPY]*coeff_self[AMPY][UNITY]
											+ wtdest[iEqn][AMPZ]*coeff_self[AMPZ][UNITY];
		// pointless anyway - it gets overwritten?

											
		pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn][GAUSS]*coeff_self[GAUSS][PHI_ANODE]
												+ wtdest[iEqn][AMPX]*coeff_self[AMPX][PHI_ANODE]
												+ wtdest[iEqn][AMPY]*coeff_self[AMPY][PHI_ANODE]
												+ wtdest[iEqn][AMPZ]*coeff_self[AMPZ][PHI_ANODE];

	};
#else

	iEqn = AMPZ;
	pAffected->coeff_self[iEqn][UNITY] += weight*coeff_self[iEqn][UNITY];
	pAffected->coeff_self[iEqn][EZTUNING] += weight*coeff_self[iEqn][EZTUNING];
	

#endif

	// older:
/*
	for (iEqn = 0; iEqn < 4; iEqn++) 
	{
		if ((rotatedest == 0) || (iEqn == GAUSS) || (iEqn == AMPZ)) {
			pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn]*coeff_self[iEqn][UNITY];
			pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn]*coeff_self[iEqn][PHI_ANODE];
		} else {
			if (iEqn == AMPX) {
				// Ampere x epsilon
				if (rotatedest == CLOCKWISE) {
					pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn]*(
						Clockwise.xx*coeff_self[AMPX][UNITY]
					+	Clockwise.xy*coeff_self[AMPY][UNITY]);
					pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn]*(
						Clockwise.xx*coeff_self[AMPX][PHI_ANODE]
					+	Clockwise.xy*coeff_self[AMPY][PHI_ANODE]);
				} else {					
					pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn]*(
						Anticlockwise.xx*coeff_self[AMPX][UNITY]
					+	Anticlockwise.xy*coeff_self[AMPY][UNITY]);
					pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn]*(
						Anticlockwise.xx*coeff_self[AMPX][PHI_ANODE]
					+	Anticlockwise.xy*coeff_self[AMPY][PHI_ANODE]);
				};
			} else {
				// Ampere y epsilon
				if (rotatedest == CLOCKWISE) {
					pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn]*(
						Clockwise.yx*coeff_self[AMPX][UNITY]
					+	Clockwise.yy*coeff_self[AMPY][UNITY]);
					pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn]*(
						Clockwise.yx*coeff_self[AMPX][PHI_ANODE]
					+	Clockwise.yy*coeff_self[AMPY][PHI_ANODE]);
				} else {
					pAffected->coeff_self[iEqn][UNITY] += wtdest[iEqn]*(
						Anticlockwise.yx*coeff_self[AMPX][UNITY]
					+	Anticlockwise.yy*coeff_self[AMPY][UNITY]);
					pAffected->coeff_self[iEqn][PHI_ANODE] += wtdest[iEqn]*(
						Anticlockwise.yx*coeff_self[AMPX][PHI_ANODE]
					+	Anticlockwise.yy*coeff_self[AMPY][PHI_ANODE]);
				};
			};
		};		
	};
*/

	// IZ eqn:

	// Changing A at coarse point affects A at fine point which contributes to Eps_Iz,
	// and we put this effect into coeff_self at the coarse point.
	
	// SO HERE IT'S ACTUALLY wtsrc NOT wtdest.
	// We used "wtdest" for now as long as they are actually the same.
	
	// CUTTED OUT COEFFS FOR JUST AMPZ,AZ

	//if (0)
	//	pAffected->coeff_self[IZ][PHI] += wtsrc[PHI]*coeff_self[IZ][PHI];
	
	pAffected->coeff_self[IZ][AZ] += weight*coeff_self[IZ][AZ];

	if (0) {
/*
	if (rotatedest == 0) {
		pAffected->coeff_self[IZ][AX] += wtsrc[AX]*coeff_self[IZ][AX];
		pAffected->coeff_self[IZ][AY] += wtsrc[AY]*coeff_self[IZ][AY];
	} else {
		if (rotatedest == CLOCKWISE) // note: for this situation, rotatedest = -rotatesrc
		{

			pAffected->coeff_self[IZ][AX] += 
				wtsrc[AX]*(Anticlockwise.xx*coeff_self[IZ][AX]
							+ Anticlockwise.yx*coeff_self[IZ][AY]);
				// Note how this is used.
			pAffected->coeff_self[IZ][AY] += 
				wtsrc[AY]*(Anticlockwise.yy*coeff_self[IZ][AY]
							+ Anticlockwise.xy*coeff_self[IZ][AX]);
		} else {
			pAffected->coeff_self[IZ][AX] += 
				wtsrc[AX]*(Clockwise.xx*coeff_self[IZ][AX]
							+ Clockwise.yx*coeff_self[IZ][AY]);
			pAffected->coeff_self[IZ][AY] += 
				wtsrc[AY]*(Clockwise.yy*coeff_self[IZ][AY]
							+ Clockwise.xy*coeff_self[IZ][AX]);
		};
	};
	*/
	};

}

void TriMesh::AccumulateCoefficients(
			Vertex * pAffected, //pAffector, 
					//pVertex->iCoarseIndex[iAffectedIndex],
			int iNeigh,//pVertex->iCoarseIndex[iAffectorIndex],
			//real wtsrc[4], real wtdest[4][4], 
			real wtsrc, real wtdest,
			real coeff[4][4],
			char rotatesrc,char rotatedest,
			
			long iFinePhi, long iIntermediate 
			)
{
	// OUR rotatesrc means rotation of fine A pt rel to coarse A pt;
	// OUR rotatedest means rotation of coarse eps pt rel to fine eps pt.

	// Important: if need to retrograde this function, look in files from 28th October.

	int iEqn;
	real coeff_addition[4][4];
	real effect_on_fine_eps[4][4];
	memset(&(coeff_addition[0][0]),0,sizeof(real)*4*4);
	memset(&(effect_on_fine_eps[0][0]),0,sizeof(real)*4*4);

	// GET RID OF THEM, THIS IS FOR AMPZ.
	if(0) {
/*	coeff_addition[GAUSS][PHI] += wtsrc[PHI]*(wtdest[GAUSS][GAUSS]*coeff[GAUSS][PHI]
											+ wtdest[GAUSS][AMPX]*coeff[AMPX][PHI]
											+ wtdest[GAUSS][AMPY]*coeff[AMPY][PHI]
											+ wtdest[GAUSS][AMPZ]*coeff[AMPZ][PHI]);
	
	coeff_addition[AMPX][PHI] += wtsrc[PHI]*( wtdest[AMPX][GAUSS]*coeff[GAUSS][PHI]
										    + wtdest[AMPX][AMPX]*coeff[AMPX][PHI]
											+ wtdest[AMPX][AMPY]*coeff[AMPY][PHI]
											+ wtdest[AMPX][AMPZ]*coeff[AMPZ][PHI]);

	coeff_addition[AMPY][PHI] += wtsrc[PHI]*( wtdest[AMPY][GAUSS]*coeff[GAUSS][PHI]
											+ wtdest[AMPY][AMPX]*coeff[AMPX][PHI]
											+ wtdest[AMPY][AMPY]*coeff[AMPY][PHI]
											+ wtdest[AMPY][AMPZ]*coeff[AMPZ][PHI]);

	coeff_addition[AMPZ][PHI] += wtsrc[PHI]*( wtdest[AMPZ][GAUSS]*coeff[GAUSS][PHI]
											+ wtdest[AMPZ][AMPX]*coeff[AMPX][PHI]
											+ wtdest[AMPZ][AMPY]*coeff[AMPY][PHI]
											+ wtdest[AMPZ][AMPZ]*coeff[AMPZ][PHI]);

	
	coeff_addition[GAUSS][AZ] += wtsrc[AZ]*(wtdest[GAUSS][GAUSS]*coeff[GAUSS][AZ]
											+ wtdest[GAUSS][AMPX]*coeff[AMPX][AZ]
											+ wtdest[GAUSS][AMPY]*coeff[AMPY][AZ]
											+ wtdest[GAUSS][AMPZ]*coeff[AMPZ][AZ]);
	
	coeff_addition[AMPX][AZ] += wtsrc[AZ]*( wtdest[AMPX][GAUSS]*coeff[GAUSS][AZ]
										    + wtdest[AMPX][AMPX]*coeff[AMPX][AZ]
											+ wtdest[AMPX][AMPY]*coeff[AMPY][AZ]
											+ wtdest[AMPX][AMPZ]*coeff[AMPZ][AZ]);
	
	coeff_addition[AMPY][AZ] += wtsrc[AZ]*( wtdest[AMPY][GAUSS]*coeff[GAUSS][AZ]
											+ wtdest[AMPY][AMPX]*coeff[AMPX][AZ]
											+ wtdest[AMPY][AMPY]*coeff[AMPY][AZ]
											+ wtdest[AMPY][AMPZ]*coeff[AMPZ][AZ]);
*/
	}; // if (0)

#ifdef FUNKYWEIGHTS
	coeff_addition[AMPZ][AZ] += wtsrc[AZ]*( wtdest[AMPZ][GAUSS]*coeff[GAUSS][AZ]
											+ wtdest[AMPZ][AMPX]*coeff[AMPX][AZ]
											+ wtdest[AMPZ][AMPY]*coeff[AMPY][AZ]
											+ wtdest[AMPZ][AMPZ]*coeff[AMPZ][AZ]);
#else
	coeff_addition[AMPZ][AZ] += wtsrc*( wtdest*coeff[AMPZ][AZ]);
#endif
										
	// Let's track contributions to GAUSS for [0][3529].
	
//	coeff_addition[AMPZ][PHI] += wtsrc[PHI]*wtdest[AMPZ]*coeff[AMPZ][PHI];
//	coeff_addition[GAUSS][PHI] += wtsrc[PHI]*wtdest[GAUSS][GAUSS]
//	coeff_addition[AMPZ][AZ] += wtsrc[AZ]*wtdest[AMPZ]*coeff[AMPZ][AZ];

	if (0) {
	if (rotatesrc == 0) {
		/*coeff_addition[GAUSS][AX] += wtsrc[AX]*(wtdest[GAUSS][GAUSS]*coeff[GAUSS][AX]
											+ wtdest[GAUSS][AMPX]*coeff[AMPX][AX]
											+ wtdest[GAUSS][AMPY]*coeff[AMPY][AX]
											+ wtdest[GAUSS][AMPZ]*coeff[AMPZ][AX]);
	
		coeff_addition[AMPX][AX] += wtsrc[AX]*( wtdest[AMPX][GAUSS]*coeff[GAUSS][AX]
										    + wtdest[AMPX][AMPX]*coeff[AMPX][AX]
											+ wtdest[AMPX][AMPY]*coeff[AMPY][AX]
											+ wtdest[AMPX][AMPZ]*coeff[AMPZ][AX]);
	
		coeff_addition[AMPY][AX] += wtsrc[AX]*( wtdest[AMPY][GAUSS]*coeff[GAUSS][AX]
											+ wtdest[AMPY][AMPX]*coeff[AMPX][AX]
											+ wtdest[AMPY][AMPY]*coeff[AMPY][AX]
											+ wtdest[AMPY][AMPZ]*coeff[AMPZ][AX]);
	
		coeff_addition[AMPZ][AX] += wtsrc[AX]*( wtdest[AMPZ][GAUSS]*coeff[GAUSS][AX]
											+ wtdest[AMPZ][AMPX]*coeff[AMPX][AX]
											+ wtdest[AMPZ][AMPY]*coeff[AMPY][AX]
											+ wtdest[AMPZ][AMPZ]*coeff[AMPZ][AX]);
		// =====
		
		coeff_addition[GAUSS][AY] += wtsrc[AY]*(wtdest[GAUSS][GAUSS]*coeff[GAUSS][AY]
											+ wtdest[GAUSS][AMPX]*coeff[AMPX][AY]
											+ wtdest[GAUSS][AMPY]*coeff[AMPY][AY]
											+ wtdest[GAUSS][AMPZ]*coeff[AMPZ][AY]);
		
		coeff_addition[AMPX][AY] += wtsrc[AY]*( wtdest[AMPX][GAUSS]*coeff[GAUSS][AY]
										    + wtdest[AMPX][AMPX]*coeff[AMPX][AY]
											+ wtdest[AMPX][AMPY]*coeff[AMPY][AY]
											+ wtdest[AMPX][AMPZ]*coeff[AMPZ][AY]);
	
		coeff_addition[AMPY][AY] += wtsrc[AY]*( wtdest[AMPY][GAUSS]*coeff[GAUSS][AY]
											+ wtdest[AMPY][AMPX]*coeff[AMPX][AY]
											+ wtdest[AMPY][AMPY]*coeff[AMPY][AY]
											+ wtdest[AMPY][AMPZ]*coeff[AMPZ][AY]);
	
		coeff_addition[AMPZ][AY] += wtsrc[AY]*( wtdest[AMPZ][GAUSS]*coeff[GAUSS][AY]
											+ wtdest[AMPZ][AMPX]*coeff[AMPX][AY]
											+ wtdest[AMPZ][AMPY]*coeff[AMPY][AY]
											+ wtdest[AMPZ][AMPZ]*coeff[AMPZ][AY]);

		//coeff_addition[GAUSS][AX] += wtdest[GAUSS]*wtsrc[AX]*coeff[GAUSS][AX];
		//coeff_addition[GAUSS][AY] += wtdest[GAUSS]*wtsrc[AY]*coeff[GAUSS][AY];
		//coeff_addition[AMPZ][AX] += wtdest[AMPZ]*wtsrc[AX]*coeff[AMPZ][AX];
		//coeff_addition[AMPZ][AY] += wtdest[AMPZ]*wtsrc[AY]*coeff[AMPZ][AY];
		//		
		//effect_on_fine_eps[AMPX][AX] += coeff[AMPX][AX];
		//effect_on_fine_eps[AMPX][AY] += coeff[AMPX][AY];
		//effect_on_fine_eps[AMPY][AX] += coeff[AMPY][AX];
		//effect_on_fine_eps[AMPY][AY] += coeff[AMPY][AY];		
	} else {
		if (rotatesrc == ANTICLOCKWISE) {
			// A src point was clockwise from affected fine point
			// Fine A = Anticlockwise * Coarse A
			// Changing Coarse Ax means Fine Ay += Anticlockwise.yx * change.

			//coeff_addition[GAUSS][AX] += wtdest[GAUSS]*wtsrc[AX]*(
			//	Anticlockwise.xx*coeff[GAUSS][AX] + Anticlockwise.yx*coeff[GAUSS][AY]);

			// wtsrc[AX] is the coefficient on the rotated Ax = (Anticlockwise.xx*Ax + Anticlockwise.xy*Ay) -- correct?
			// -**- check that.

			// It works like this : (Ax,Ay) -> rotated -> wtsrc[AX] applies to x component to give Ax.
			// Ax -> Anti.xx Ax -> Ax fine -> eps_Gauss fine
			// Ay -> Anti.xy Ay ->[wtsrc Ax] Ax fine -> eps_fine
			// Okay but we ask what happens from changing coarse Ax.
			// Ax -> Anti.yx Ax ->[wtsrc Ay] Ay fine -> eps_fine

			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				coeff_addition[iEqn][AX] += wtdest[iEqn][GAUSS]*
					(wtsrc[AX]*Anticlockwise.xx*coeff[GAUSS][AX] + wtsrc[AY]*Anticlockwise.yx*coeff[GAUSS][AY])
												 + wtdest[iEqn][AMPX]*
					(wtsrc[AX]*Anticlockwise.xx*coeff[AMPX][AX] + wtsrc[AY]*Anticlockwise.yx*coeff[AMPX][AY])
												+ wtdest[iEqn][AMPY]*
					(wtsrc[AX]*Anticlockwise.xx*coeff[AMPY][AX] + wtsrc[AY]*Anticlockwise.yx*coeff[AMPY][AY])
												+ wtdest[iEqn][AMPZ]*
					(wtsrc[AX]*Anticlockwise.xx*coeff[AMPZ][AX] + wtsrc[AY]*Anticlockwise.yx*coeff[AMPZ][AY]);

				coeff_addition[iEqn][AY] += wtdest[iEqn][GAUSS]*
					(wtsrc[AX]*Anticlockwise.xy*coeff[GAUSS][AX] + wtsrc[AY]*Anticlockwise.yy*coeff[GAUSS][AY])
												+ wtdest[iEqn][AMPX]*
					(wtsrc[AX]*Anticlockwise.xy*coeff[AMPX][AX] + wtsrc[AY]*Anticlockwise.yy*coeff[AMPX][AY])
												+ wtdest[iEqn][AMPY]*
					(wtsrc[AX]*Anticlockwise.xy*coeff[AMPY][AX] + wtsrc[AY]*Anticlockwise.yy*coeff[AMPY][AY])
												+ wtdest[iEqn][AMPZ]*
					(wtsrc[AX]*Anticlockwise.xy*coeff[AMPZ][AX] + wtsrc[AY]*Anticlockwise.yy*coeff[AMPZ][AY]);
			};
			// Quite obviously could do the confab more efficiently.
			
		} else {
			
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				coeff_addition[iEqn][AX] += wtdest[iEqn][GAUSS]*
					(wtsrc[AX]*Clockwise.xx*coeff[GAUSS][AX] + wtsrc[AY]*Clockwise.yx*coeff[GAUSS][AY])
												 + wtdest[iEqn][AMPX]*
					(wtsrc[AX]*Clockwise.xx*coeff[AMPX][AX] + wtsrc[AY]*Clockwise.yx*coeff[AMPX][AY])
												+ wtdest[iEqn][AMPY]*
					(wtsrc[AX]*Clockwise.xx*coeff[AMPY][AX] + wtsrc[AY]*Clockwise.yx*coeff[AMPY][AY])
												+ wtdest[iEqn][AMPZ]*
					(wtsrc[AX]*Clockwise.xx*coeff[AMPZ][AX] + wtsrc[AY]*Clockwise.yx*coeff[AMPZ][AY]);

				coeff_addition[iEqn][AY] += wtdest[iEqn][GAUSS]*
					(wtsrc[AX]*Clockwise.xy*coeff[GAUSS][AX] + wtsrc[AY]*Clockwise.yy*coeff[GAUSS][AY])
												+ wtdest[iEqn][AMPX]*
					(wtsrc[AX]*Clockwise.xy*coeff[AMPX][AX] + wtsrc[AY]*Clockwise.yy*coeff[AMPX][AY])
												+ wtdest[iEqn][AMPY]*
					(wtsrc[AX]*Clockwise.xy*coeff[AMPY][AX] + wtsrc[AY]*Clockwise.yy*coeff[AMPY][AY])
												+ wtdest[iEqn][AMPZ]*
					(wtsrc[AX]*Clockwise.xy*coeff[AMPZ][AX] + wtsrc[AY]*Clockwise.yy*coeff[AMPZ][AY]);
			};
		};*/
	};

	}; // if (0)

	/*
	if (rotatedest == 0) {
		coeff_addition[AMPX][PHI] += wtdest[AMPX]*wtsrc[PHI]*coeff[AMPX][PHI];
		coeff_addition[AMPY][PHI] += wtdest[AMPY]*wtsrc[PHI]*coeff[AMPY][PHI];
		coeff_addition[AMPX][AZ] += wtdest[AMPX]*wtsrc[AZ]*coeff[AMPX][AZ];
		coeff_addition[AMPY][AZ] += wtdest[AMPY]*wtsrc[AZ]*coeff[AMPY][AZ];

		coeff_addition[AMPX][AX] += wtdest[AMPX]*wtsrc[AX]*effect_on_fine_eps[AMPX][AX];
		coeff_addition[AMPX][AY] += wtdest[AMPX]*wtsrc[AY]*effect_on_fine_eps[AMPX][AY];
		coeff_addition[AMPY][AX] += wtdest[AMPY]*wtsrc[AX]*effect_on_fine_eps[AMPY][AX];
		coeff_addition[AMPY][AY] += wtdest[AMPY]*wtsrc[AY]*effect_on_fine_eps[AMPY][AY];
	} else {
		if (rotatedest == ANTICLOCKWISE) {
			// coarse eps is Anticlockwise from fine eps.
			// coeff[AMPX][PHI] is the contribution to fine eps x. Contrib to coarse eps x comes through fine x and fine y
			coeff_addition[AMPX][PHI] += wtdest[AMPX]*wtsrc[PHI]*(
				Anticlockwise.xx*coeff[AMPX][PHI] + Anticlockwise.xy*coeff[AMPY][PHI]);
			coeff_addition[AMPY][PHI] += wtdest[AMPY]*wtsrc[PHI]*(
				Anticlockwise.yx*coeff[AMPX][PHI] + Anticlockwise.yy*coeff[AMPY][PHI]);
			coeff_addition[AMPX][AZ] += wtdest[AMPX]*wtsrc[AZ]*(
				Anticlockwise.xx*coeff[AMPX][AZ] + Anticlockwise.xy*coeff[AMPY][AZ]);
			coeff_addition[AMPY][AZ] += wtdest[AMPY]*wtsrc[AZ]*(
				Anticlockwise.yx*coeff[AMPX][AZ] + Anticlockwise.yy*coeff[AMPY][AZ]);

			coeff_addition[AMPX][AX] += wtdest[AMPX]*wtsrc[AX]*(
				Anticlockwise.xx*effect_on_fine_eps[AMPX][AX] + Anticlockwise.xy*effect_on_fine_eps[AMPY][AX]);
			coeff_addition[AMPY][AX] += wtdest[AMPY]*wtsrc[AX]*(
				Anticlockwise.yx*effect_on_fine_eps[AMPX][AX] + Anticlockwise.yy*effect_on_fine_eps[AMPY][AX]);
			coeff_addition[AMPX][AY] += wtdest[AMPX]*wtsrc[AY]*(
				Anticlockwise.xx*effect_on_fine_eps[AMPX][AY] + Anticlockwise.xy*effect_on_fine_eps[AMPY][AY]);
			coeff_addition[AMPY][AY] += wtdest[AMPY]*wtsrc[AY]*(
				Anticlockwise.yx*effect_on_fine_eps[AMPX][AY] + Anticlockwise.yy*effect_on_fine_eps[AMPY][AY]);

		} else {
			coeff_addition[AMPX][PHI] += wtdest[AMPX]*wtsrc[PHI]*(
				Clockwise.xx*coeff[AMPX][PHI] + Clockwise.xy*coeff[AMPY][PHI]);
			coeff_addition[AMPY][PHI] += wtdest[AMPY]*wtsrc[PHI]*(
				Clockwise.yx*coeff[AMPX][PHI] + Clockwise.yy*coeff[AMPY][PHI]);
			coeff_addition[AMPX][AZ] += wtdest[AMPX]*wtsrc[AZ]*(
				Clockwise.xx*coeff[AMPX][AZ] + Clockwise.xy*coeff[AMPY][AZ]);
			coeff_addition[AMPY][AZ] += wtdest[AMPY]*wtsrc[AZ]*(
				Clockwise.yx*coeff[AMPX][AZ] + Clockwise.yy*coeff[AMPY][AZ]);
						
			coeff_addition[AMPX][AX] += wtdest[AMPX]*wtsrc[AX]*(
				Clockwise.xx*effect_on_fine_eps[AMPX][AX] + Clockwise.xy*effect_on_fine_eps[AMPY][AX]);
			coeff_addition[AMPY][AX] += wtdest[AMPY]*wtsrc[AX]*(
				Clockwise.yx*effect_on_fine_eps[AMPX][AX] + Clockwise.yy*effect_on_fine_eps[AMPY][AX]);
			coeff_addition[AMPX][AY] += wtdest[AMPX]*wtsrc[AY]*(
				Clockwise.xx*effect_on_fine_eps[AMPX][AY] + Clockwise.xy*effect_on_fine_eps[AMPY][AY]);
			coeff_addition[AMPY][AY] += wtdest[AMPY]*wtsrc[AY]*(
				Clockwise.yx*effect_on_fine_eps[AMPX][AY] + Clockwise.yy*effect_on_fine_eps[AMPY][AY]);
		};
	};
*/

	pAffected->AddToCoefficients(iNeigh,coeff_addition);
	// iNeigh sometimes == -1
}

void TriMesh::SpitOutGauss()//long iStart, long iEnd)
{
	static long ctr = 0;
	char buffer[256];
	FILE * fp;
	long iStart, iEnd, i;
	real rr;
	long neigh_len, iVertex, izNeigh[128];
	Vertex * pNeigh;
	Vertex * pVertex;

	printf("SpitOutGauss %d\n",ctr);
	sprintf(buffer,"gauss_%d.txt",ctr);
	fp = fopen(buffer,"w");
	pVertex = X;
	iStart = 0;
	do 
	{
		iStart++;
		pVertex++;
		rr = pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y;
	} while (rr < 3.4*3.4);

	iEnd = iStart;
	do 
	{
		iEnd++;
		pVertex++;
		rr = pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y;
	} while (rr < 3.5*3.5);

	real coefflocal[4][4];
	pVertex = X + iStart;
	for (iVertex = iStart; iVertex < iEnd; iVertex++)
	{
		fprintf(fp,"%d %1.10E %1.10E | %1.14E %1.14E %1.14E %1.14E | "
					"%1.14E %1.14E %1.14E %1.14E | ",
			iVertex,pVertex->pos.x,pVertex->pos.y,pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
			pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3]);
		fprintf(fp,"%1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
			pVertex->coeff_self[0][0],
			pVertex->coeff_self[0][1],pVertex->coeff_self[0][2],
			pVertex->coeff_self[0][3],pVertex->coeff_self[0][4],pVertex->coeff_self[0][5]);
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		
		for (i = 0; i < neigh_len; i++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),i);
			//pNeigh = X + izNeigh[i];
			fprintf(fp,"%d %1.14E %1.14E %1.14E %1.14E ; ",
				izNeigh[i],coefflocal[0][0],coefflocal[0][1],coefflocal[0][2],coefflocal[0][3]);
		};
		fprintf(fp,"\n");
		++pVertex;
	}
	fclose(fp);
	ctr++;
}


void TriMesh::Solve_Az( real const time_back_for_Adot_if_initial )
{
	Vertex * pVertex;
	long iVertex;
	char o;
	int i, iVolley, iEqn, iLevel;
	real RSS, sumeps, RSSArray[5],Eeps_array[5],RSS_Absolute_array[5],
		L2abs,thresh,variance,SD,Iz_attained;
	bool bSucceedIz;
	FILE *fp;
	char buffer[256];
	HANDLE hConsole;
	real Iz_diff;
	static int ctr = 0;
	real Iz_effect_of_phi, Iz_effect_of_Az, Izcoeff_phianode;
	int counter1 = 0;
	bool bMultimesh = true;
		
	bGlobalSpitFlag = false;
	bGlobalInitial = true;
	
	printf("CreateODECoefficients(true);...");
	CreateODECoefficientsAz(true);
		
	printf("done\n");

	// Seed:
	this->EzTuning = 0.0;

	// This is what we need to be setting,
	// and we need equivalent on every aux level.

	pVertex= X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		memset(&(pVertex->A),0,sizeof(Vector3));
		++pVertex;
	};
	// this wiped out the existing, A_k.
		
	printf("CreateAuxiliarySubmeshes() and Galerkin coefficients: ");
	this->CreateAuxiliarySubmeshes(true); // change this to just do simple, 1 symmetric weight, instead of dozens asymmetric.
		
	printf("Time elapsed %s \n ",report_time(1));
		
	report_time(0);	
	real * storeAz;
	real store_before_epsIz;
	long iIteration = 0;
	int graphswitch = 1;
	bool bRefreshLUCoeffs = 1;
	//int const ITERATIONS_PER_LEVEL = 4;
	int iFinestIteration = 0;
	int times_up[NUM_COARSE_LEVELS];
	bool Finest_condition_known_satisfied = 0;
	iLevel = -1;
	
	memset(times_up,0,sizeof(int)*NUM_COARSE_LEVELS);
	
	while (Finest_condition_known_satisfied == 0)
	{		
		hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hConsole, 13);
		printf("\nLevel %d. iIteration %d\n",iLevel,iIteration);
		SetConsoleTextAttribute(hConsole, 15);
		
		// DEBUG:
		// Start of a go: 
		/*
		sprintf(buffer,"Lvl%d_%d.txt",iLevel,iIteration);
		fp = fopen(buffer,"w");
		if (iLevel == -1) {
			this->CalculateEpsilons();
		} else {
			this->CalculateEpsilonAux4(iLevel);
		};
		
		if (iLevel >= 0) {
			fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
			pVertex = AuxX[iLevel];
			for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
			{
				fprintf(fp,"%d %1.14E %d ",iVertex,pVertex->epsilon[3],pVertex->coarse_len);	
				for (i = 0; i < pVertex->coarse_len; i++)
					fprintf(fp,"%d %1.6E ",pVertex->iCoarseIndex[i],pVertex->weight[i]);
				fprintf(fp,"\n");
				++pVertex;
			};
		} else {
			fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz);
			pVertex = X;
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
				fprintf(fp,"%d %1.14E %d ",iVertex,pVertex->epsilon[3],pVertex->coarse_len);	
				for (i = 0; i < pVertex->coarse_len; i++)
					fprintf(fp,"%d %1.6E ",pVertex->iCoarseIndex[i],pVertex->weight[i]);
				fprintf(fp,"\n");
				++pVertex;
			};
		}
		fclose(fp);*/

		// ==========================================

		// DEBUG:
//#define DBG_0
#ifdef DBG_0

			IterationsJRLS(-1,40);

#else

		bGlobalSpitFlag = ((iIteration % 10 == 0)
						|| (iLevel >= NUM_COARSE_LEVELS-2));
				
		if (iLevel == NUM_COARSE_LEVELS-1)
		{
			printf("RunLU...\n");
			RunLU_Az(iLevel,bRefreshLUCoeffs);
			bRefreshLUCoeffs = 0;			
		} else {
			if ((iLevel == -1) || (times_up[iLevel] == 0))
			{
				printf("iLevel = %d; numVolleys = %d \n",iLevel,numVolleys[iLevel+1]);
				if (iLevel == -1) {
					CalculateEpsilonsAz();
				} else {
					CalculateEpsilonAuxAz(iLevel);
				};
				
				IterationsJRLS_Az(iLevel,4);			
				
			} else {
				IterationsJRLS_Az(iLevel,ITERATIONS_at_levels[iLevel+1]);		
				
				CalculateEpsilonAuxAz(iLevel);
				/*for (iVolley = 0; iVolley < numVolleys[iLevel+1]; iVolley++)
				{
					printf("got to here --- iVolley = %d\n",iVolley);
					GaussSeidel(iLevel,iVolley);
				};*/
			};
			
			if (iLevel == -1) {
				// debug:
				// spit out some data for phi, Gauss equation, eps. First 2000 points.
				
			} else {
				// debug:
				// What is phi_anode and how does it affect
				printf("phi_anode at lvl %d = %1.9E\n",iLevel,this->PhiAnode_aux[iLevel].x[0]);
			};
		};
#endif

		if ((iLevel == -1) && (iIteration % 1 == 2)) {
			
			if (iIteration % 1 == 0) {
				printf("0= Gauss graphs,1= Amp xy,2= Amp z,3= eps:");
				o = '2';// (char)getch();
				if (o == 'f') {
					// switch to finest
					bMultimesh = 0;
					printf("f\n");
				};
				if (o == 'm') {
					bMultimesh = true;
					printf("m\n");
				};
				if (o == '0') {
					printf("0");
					// Create Exy (from all sources) and rho..
					
					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.x = 0.0; pVertex->E.y = 0.0;
						pVertex->temp2.x = 0.0; // use Temp for J. temp2.y is shape of Ez field.
						++pVertex;
					};
					graphswitch = 0;
				}
				if (o == '1') {
					printf("1");

					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.x = 0.0; pVertex->E.y = 0.0;
						memset(&(pVertex->Temp),0,sizeof(Vector3)); // J
						
						++pVertex;
					};
					graphswitch = 1;
				}
				if (o == '2') {
					printf("2");
					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.z = 0.0; 
						memset(&(pVertex->Temp),0,sizeof(Vector3)); // J
						
						++pVertex;
					};
					graphswitch = 2;
				}
				if (o == '3') {
					printf("3");
					graphswitch = 3;				
				}
				printf("\n");
			};

			switch(graphswitch) {
				case 0:
						
					Graph[0].DrawSurface("phi",
					  DATA_HEIGHT,(real *)(&(this->X[0].phi)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].phi)),
					  true, 
					  GRAPH_PHI, this);
					Graph[1].DrawSurface("Exy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].E)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].E)),
					  true,
					  GRAPH_EXY, this);
					Graph[2].DrawSurface("eps_Gauss",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[0])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[0])),
					  true,
					  GRAPH_EPS0, this);
					Graph[3].DrawSurface("rho",
					  DATA_HEIGHT,(real *)(&(this->X[0].temp2.x)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].temp2.x)),
					  true,
					  GRAPH_RHO, this);		
					break;
				case 1:
					
					Graph[0].DrawSurface("Axy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].A.x)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].A.x)),
					  true, 
					  GRAPH_AXY, this);
					Graph[1].DrawSurface("eps_Ax",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS1, this);
					Graph[2].DrawSurface("eps_Axy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS12, this);
					Graph[3].DrawSurface("Ax",
					  DATA_HEIGHT,(real *)(&(this->X[0].A.x)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].A.x)),
					  true,
					  GRAPH_AX, this);		
					break;
				case 2:
					Graph[0].DrawSurface("Az",
					  DATA_HEIGHT,(real *)(&(this->X[0].A.z)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].A.z)),
					  true, 
					  GRAPH_AZ, this);
					Graph[1].DrawSurface("Exy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].E)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].E)),
					  true,
					  GRAPH_EXY, this);
					Graph[2].DrawSurface("eps_z",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[3])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[3])),
					  true,
					  GRAPH_EPS3, this);
					Graph[3].DrawSurface("Jz",
					  DATA_HEIGHT,(real *)(&(this->X[0].Temp.z)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].Temp.z)),
					  true,
					  GRAPH_JZ, this);
					break;
				case 3:
					Graph[0].DrawSurface("eps_Gauss",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[0])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[0])),
					  true, 
					  GRAPH_AZ, this);
					Graph[1].DrawSurface("eps_Ampx",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS1, this);
					Graph[2].DrawSurface("eps_Ampz",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[3])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[3])),
					  true,
					  GRAPH_EPS3, this);
					Graph[3].DrawSurface("eps_Ampy",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[2])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[2])),
					  true,
					  GRAPH_EPS2, this);
				break;
			};
			
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
			
			 // whether finest level, to do graphs
		};


		if (iLevel == -1) {

			// Test whether finest condition satisfied:
			// ========================================
			
			CalculateEpsilonsAz();		
			// Survey RSS from each equation:
			if (iIteration % 4 == 0) {
				printf("sumeps ");
			};
			iEqn = AZ;
			{
				RSS = 0.0;
				sumeps = 0.0;
				pVertex = X;
				for (iVertex =0; iVertex < numVertices; iVertex++)
				{
					RSS += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					sumeps += pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSSArray[iEqn] = RSS;
				Eeps_array[iEqn] = sumeps/(real)numVertices;

				if (iIteration % 4 == 0) {
					printf("%1.6E ",sumeps);
				};
			};
			RSSArray[4] = this->Epsilon_Iz*this->Epsilon_Iz; // error in 4pi/c Iz
			if (iIteration % 4 == 0) {
				printf("\n");
			};	
			printf("RSS Ampz %1.5E Iz %1.5E\n",
					RSSArray[3],RSSArray[4]);
			
			// What do we want? Look at other file - see how to do FP comparison.
			
			CalculateEpsilonsAbsoluteAz(RSS_Absolute_array);
			
			Finest_condition_known_satisfied = true;
			iEqn = AMPZ;
			{
				if (RSSArray[iEqn] != 0.0)
				{
					variance = RSSArray[iEqn]/(real)numVertices - Eeps_array[iEqn]*Eeps_array[iEqn];
					SD = sqrt(variance);						
					
					L2abs = sqrt(RSS_Absolute_array[iEqn]/(real)numVertices);
					thresh = ODE_FPRATIO*L2abs + ODE_ABSTHRESHOLD[iEqn];
					
					printf("iEqn %d  thresh %1.8E  SD %1.8E \n",iEqn,thresh,SD);
					if (SD > thresh) Finest_condition_known_satisfied = false;				
					// Think about this: should be doing Axy jointly.				
				};
			};

			// RSS_Absolute_array[IZ] is a positive number, Iz_prescribed is a negative number.

			Iz_attained = RSS_Absolute_array[IZ]; // involves 4pi/c factor
			thresh = fabs( ODE_IZRELTOL*Iz_prescribed*FOURPI_OVER_C );
			
			Iz_diff = fabs(Iz_attained - FOURPI_OVER_C*Iz_prescribed); 
			
			printf("4pi/c Iz_presc %1.6E attain %1.6E diff %1.6E thresh %1.6E \n",
				Iz_prescribed*FOURPI_OVER_C, RSS_Absolute_array[IZ],
				Iz_diff, thresh);
			bSucceedIz = (Iz_diff < thresh);
			if (bSucceedIz == false) Finest_condition_known_satisfied = false;
			
			if (Finest_condition_known_satisfied == false)
			{

				if (bMultimesh) {
					Lift_to_coarse_eps_Az(0);
					iLevel = 0;
					times_up[0] = 0;
				};
				
				// Test descend/ascend: done
				
				//fp = fopen("ascend.txt","w");
				//this->CalculateEpsilonAux4(0);
				//fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[0]);
				//pVertex = AuxX[0];
				//for (iVertex = 0; iVertex < numAuxVertices[0]; iVertex++)
				//{
				//	fprintf(fp,"%d %1.14E \n",iVertex,pVertex->epsilon[3]);	
				//	++pVertex;
				//};
				//fclose(fp);

				//this->Affect_vars_finer(iLevel);	
				//this->CalculateEpsilons();
				//Lift_to_coarse_eps(0); // nothing happened so this isn't even a Galerkin test.

				//fp = fopen("reascend.txt","w");
				//this->CalculateEpsilonAux4(0);
				//fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[0]);
				//pVertex = AuxX[0];
				//for (iVertex = 0; iVertex < numAuxVertices[0]; iVertex++)
				//{
				//	fprintf(fp,"%d %1.14E \n",iVertex,pVertex->epsilon[3]);	
				//	++pVertex;
				//};
				//fclose(fp);
				
			};

		} else {
			// not finest level

			this->CalculateEpsilonAuxAz(iLevel);
			// Collect RSS:
			
			memset(RSSArray,0,sizeof(real)*4);
			sumeps = 0.0;
			pVertex = AuxX[iLevel];
			for (iVertex =0; iVertex < numAuxVertices[iLevel]; iVertex++)
			{
				RSSArray[3] += pVertex->epsilon[3]*pVertex->epsilon[3];
				++pVertex;
			};
			printf("Level %d RSS %1.5E\n",
				iLevel,RSSArray[3]);

			if ((iLevel == NUM_COARSE_LEVELS-1)
				|| (times_up[iLevel] >= CYCLEPEAKS)) // consider changing this to 3, or running longer at Lvl 1 intermediately.
			{
			
				// descend:
				// Do a Galerkin test instead:
/*
				sprintf(buffer,"lvl%d_b4desc_II_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					fprintf(fp,"iVertex %d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E coarse_len %d \n",
						iVertex,pVertex->pos.x,pVertex->pos.y,
						pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
						pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
						pVertex->coarse_len);
					++pVertex;
				};
				fclose(fp);
*/
				store_before_epsIz = Epsilon_Iz_aux[iLevel];
				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					pVertex->Temp.z = pVertex->epsilon[3]; // store it
					++pVertex;
				};
				iLevel--;
/*
				if (iLevel >= 0) {
					// spit out what's here before descend:

					sprintf(buffer,"lvl%d_predesc_%d.txt",iLevel,counter1);
					fp = fopen(buffer,"w");

					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
					pVertex = AuxX[iLevel];
					for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E %d ",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->coarse_len);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						++pVertex;
					};
					fclose(fp);
				} else {
					// only spit out where relevant.

					sprintf(buffer,"lvl%d_predesc_%d.txt",iLevel,counter1);
					fp = fopen(buffer,"w");
					FILE * fp2 = fopen("vals_b4.txt","w");
					pVertex = X + 13000;
					for (iVertex = 13000; iVertex < 15000; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E %d ",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->coarse_len);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][0],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						
						
						if ((pVertex->Temp.x != 0.0) ||
							(pVertex->Temp.y != 0.0)) 	
						{
							fprintf(fp2,"iVertex %d coeffs %1.14E %1.14E %1.14E %1.14E vals %1.14E %1.14E %1.14E %1.14E  \n",
								iVertex,pVertex->Temp.x,pVertex->Temp.y,pVertex->Temp.z,pVertex->temp2.x,
								pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z);
						}

						++pVertex;
					};
					fclose(fp);
					fclose(fp2);
				};*/

				this->Affect_vars_finer_Az(iLevel);	

				if (iLevel == -1) {
					this->CalculateEpsilonsAz();
					printf("After descend: Eps_Iz[-1]: %1.14E \n",Epsilon_Iz);
				} else {
					this->CalculateEpsilonAuxAz(iLevel);
					printf("After descend: Eps_Iz[%d]: %1.14E \n",iLevel,Epsilon_Iz_aux[iLevel]);
				};

				/*
				sprintf(buffer,"lvl%d_desc_II_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				if (iLevel >= 0) {
					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
					pVertex = AuxX[iLevel];
					for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E | %d |",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->iVolley);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][GAUSS],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						++pVertex;
					};
				} else {
					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz);

					FILE * fp2 = fopen("vals_after.txt","w");
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						fprintf(fp,"%d %1.10E %1.10E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E | %d | ",iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->iVolley);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"%d %1.10E ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][GAUSS]);
						fprintf(fp,"\n");

						if ((pVertex->Temp.x != 0.0) ||
							(pVertex->Temp.y != 0.0)) 	
						{
							fprintf(fp2,"iVertex %d coeffs %1.14E %1.14E %1.14E %1.14E vals %1.14E %1.14E %1.14E %1.14E  \n",
								iVertex,pVertex->Temp.x,pVertex->Temp.y,pVertex->Temp.z,pVertex->temp2.x,
								pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z);
						}
						++pVertex;
					};
					fclose(fp2);
				}
				fclose(fp);*/


				Lift_to_coarse_eps_Az(iLevel+1);
				iLevel++;
				counter1++;
				sprintf(buffer,"lvl%d_GalerkinII_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				this->CalculateEpsilonAuxAz(iLevel);
				fprintf(fp,"Eps_Iz: %1.14E Before: %1.14E \n",Epsilon_Iz_aux[iLevel],store_before_epsIz);
				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					fprintf(fp,"%d  %1.14E before: %1.14E\n",
						iVertex,
						pVertex->epsilon[3],pVertex->Temp.z);	
					++pVertex;
				}; 
				fclose(fp);

				// go back down:
				iLevel--;

			} else {

				// NOTICE THE FOLLOWING ASSUMES WE HAVE DONE CALC EPS AT OUR OWN LEVEL!
				// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	//			// Ascend:
				times_up[iLevel]++;
				Lift_to_coarse_eps_Az(iLevel+1);
				iLevel++;
				times_up[iLevel] = 0;
	//			// If we arrive on a level from below, set times_up to 0
			};
		};		
		iIteration++;
	}; // while finest not satisfied

	printf("Convergence criteria satisfied. Time elapsed %s ",report_time(1));

	// The post-solve:
	// Use same method as in coefficients to do the ion and electron displacement.
	// To do neutral displacement can do any way; use pVertex->sigma_n.

	if (true) {

		Vector3 A_past;

		// Create Vertex::E
		// Create Jz
		// Create A-dot _k-1/2

		//real factor = 1.0/(1.0 + Y.Unreduced_beta_e_ion.zz*m_e/m_i); // assume have +beta(-me/mi)ve and move to LHS
		//pVertex->v_e_0.z = factor*Y.Unreduced_v_e_0.z;
		//pVertex->sigma_e.zz = factor*Y.Unreduced_sigma_e.zz;
		//pVertex->v_i_0.z = -(m_e/m_i)*pVertex->v_e_0.z;
		//pVertex->sigma_i.zz = -(m_e/m_i)*pVertex->sigma_e.zz;

		// We'd rather create A-dot-z right now.

		long izNeigh[128];
		long neigh_len;
		//FILE * fp = fopen("outputpostsolve.txt","w");
		pVertex = X;
		f64 IzAttained = 0.0;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->E.x = 0.0;
			pVertex->E.y = 0.0;

			f64 EzShape = GetEzShape_(pVertex->pos.modulus());
			pVertex->E.z = // assumptions? = -Adot/c + Ez_ext
				-(GlobalIzElasticity*pVertex->A.z)/c + (EzTuning*EzShape).x[0];

			pVertex->Ion.mom.z = pVertex->Ion.mass*(pVertex->v_i_0.z + pVertex->sigma_i.zz*pVertex->E.z);
			pVertex->Elec.mom.z = pVertex->Elec.mass*(pVertex->v_e_0.z + pVertex->sigma_e.zz*pVertex->E.z);

			if (iVertex == 16000) printf("\n\n16000: elec.mom.z %1.8E \n\n",pVertex->Elec.mom.z);

			// now go back in time to make Adot in the past: time_back_for_Adot_if_initial

			A_past = pVertex->A*(1.0-time_back_for_Adot_if_initial*GlobalIzElasticity);
			pVertex->Adot = A_past*GlobalIzElasticity;
			// Just pass 0 time back when we call this func.

			if (iVertex == 20000) {
				printf("20000: Adot.z = %1.9E\n",pVertex->Adot.z);
				printf("20000: Adot.z = %1.9E\n",pVertex->Adot.z);
				getch();
			}
			//pVertex->phidot = 0.0;
			pVertex->phi = 0.0;
			

			// Do file output
			/*
			fprintf(fp,"%d %d %1.14E %1.14E cent %1.14E %1.14E Ez %1.14E Az %1.14E temp2.y %1.14E TuneFac %1.14E "
				"Ion.vz %1.14E Elec.vz %1.14E n %1.14E sigma_i %1.14E sigma_e %1.14E ionmomz %1.14E elecmomz %1.14E ",
				iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y,pVertex->centroid.x,pVertex->centroid.y,
				pVertex->E.z,pVertex->A.z,pVertex->temp2.y,PhiAnode.x[0],
				pVertex->Ion.mom.z/pVertex->Ion.mass,pVertex->Elec.mom.z/pVertex->Elec.mass,
				pVertex->Elec.mass/pVertex->AreaCell,pVertex->sigma_i.zz,pVertex->sigma_e.zz,
				pVertex->Ion.mom.z,pVertex->Elec.mom.z);
			fprintf(fp,"coeff_self %1.14E | %1.14E %1.14E %1.14E | %1.14E %1.14E ",
				pVertex->coeff_self[AMPZ][PHI],pVertex->coeff_self[AMPZ][AX],
				pVertex->coeff_self[AMPZ][AY],pVertex->coeff_self[AMPZ][AZ],
				pVertex->coeff_self[AMPZ][UNITY],pVertex->coeff_self[AMPZ][PHI_ANODE]);
			
			fprintf(fp,"coeffs_on_Az  ");
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i = 0; i < neigh_len; i++)
			{
				fprintf(fp,"%d %1.14E ",izNeigh[i],pVertex->coeff[i].co[AMPZ][AZ]);
			}
			fprintf(fp,"\n");*/

			IzAttained += q*(pVertex->Ion.mom.z-pVertex->Elec.mom.z);
			++pVertex;
		};

		printf("EzTuning: %1.10E IzAttained: %1.10E \n",EzTuning.x[0],IzAttained);

		// Now report what we got for Lap Az and Jz at 16000:

		f64 LapAz = store_coeff_self*X[16000].A.z;
		neigh_len = X[16000].GetNeighIndexArray(izNeigh);
		for (int iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			LapAz += X[16000].coeff[iNeigh].co[AMPZ][AZ]*((X+izNeigh[iNeigh])->A.z);
		}
		printf("LapAz[16000] integrated %1.12E \n4pi/c Jz[16000] integrated %1.12E \n",
			LapAz,(4.0*PI/c_)*q*(X[16000].Ion.mom.z-X[16000].Elec.mom.z));

		printf("Az[16000] %1.5E Azdot[16000] %1.5E GIzE %1.5E\n",
			X[16000].A.z,X[16000].Adot.z,GlobalIzElasticity);
		// Az should be same sign as Jz surely. Check equations?

		f64 IzPred = (X[16000].coeff_self[AZ][AZ]-store_coeff_self)*X[16000].A.z
					+ X[16000].coeff_self[AZ][EZTUNING]*EzTuning.x[0]
					+ X[16000].coeff_self[AZ][UNITY];
		printf("IzPred %1.12E \n",IzPred);
		printf("coeff EZTUNING %1.12E effect %1.12E\n",
			X[16000].coeff_self[AZ][EZTUNING],
			FOUR_PI_OVER_C_*GetEzShape_(X[16000].pos.modulus())*q*(
			X[16000].Ion.mass*X[16000].sigma_i.zz  - X[16000].Elec.mass*X[16000].sigma_e.zz ));

		printf("Epsilon %1.14E \n",X[16000].epsilon[3]);
		CalculateEpsilonsAz();

		printf("Epsilon %1.14E \n",X[16000].epsilon[3]);
		

		//fclose(fp);
		//printf("done file output.\n");

	} else {
	
		// Create Vertex::E


		// ...
	}

	// delete[] to match new Vertex :
	for (i = 0; i < NUM_COARSE_LEVELS; i++)
		delete[] AuxX[i];
}




void TriMesh::Solve_A_phi(bool const bInitial, real const time_back_for_Adot_if_initial)
{
	/*
	Vertex * pVertex;
	long iVertex;
	char o;
	int i, iVolley, iEqn, iLevel;
	real RSS, sumeps, RSSArray[5],Eeps_array[5],RSS_Absolute_array[5],
		L2abs,thresh,variance,SD,Iz_attained;
	bool bSucceedIz;
	FILE *fp;
	char buffer[256];
	HANDLE hConsole;
	real Iz_diff;
	static int ctr = 0;
	real Iz_effect_of_phi, Iz_effect_of_Az, Izcoeff_phianode;
	int counter1 = 0;
	bool bMultimesh = true;
		
	bGlobalSpitFlag = !bInitial;
	bGlobalInitial = bInitial;
	
	printf("CreateODECoefficients(bInitial);\n");
	CreateODECoefficients(bInitial);
	
	
	printf("done\n");
	// Seed:
	this->PhiAnode = 0.0;
	pVertex= X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->phi = 0.0; // not used...
		memset(&(pVertex->A),0,sizeof(Vector3));
		++pVertex;
	};
	// this wiped out the existing, A_k.
	
	//CreateSeed(); // will want to pass an old mesh
	
	
	printf("CreateAuxiliarySubmeshes() and Galerkin coefficients: ");
	this->CreateAuxiliarySubmeshes(false);
	printf("Time elapsed %s \n ",report_time(1));
	
	
	//CreateMultimeshCoefficients(false); // DO NOT CONSTRUCT iCoarseTriangle...
	// ???!!?!?
	
	report_time(0);
	
	real * storeAz;
	real store_before_epsIz;
	long iIteration = 0;
	int graphswitch = 1;
	bool bRefreshLUCoeffs = 1;
	//int const ITERATIONS_PER_LEVEL = 4;
	int iFinestIteration = 0;
	int times_up[NUM_COARSE_LEVELS];
	bool Finest_condition_known_satisfied = 0;
	iLevel = -1;
	
	memset(times_up,0,sizeof(int)*NUM_COARSE_LEVELS);
	
	while (Finest_condition_known_satisfied == 0)
	{
		
		hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hConsole, 13);
		printf("\nLevel %d. iIteration %d\n",iLevel,iIteration);
		SetConsoleTextAttribute(hConsole, 15);
		
		// DEBUG:
		// Start of a go: 
		/*
		sprintf(buffer,"Lvl%d_%d.txt",iLevel,iIteration);
		fp = fopen(buffer,"w");
		if (iLevel == -1) {
			this->CalculateEpsilons();
		} else {
			this->CalculateEpsilonAux4(iLevel);
		};
		
		if (iLevel >= 0) {
			fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
			pVertex = AuxX[iLevel];
			for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
			{
				fprintf(fp,"%d %1.14E %d ",iVertex,pVertex->epsilon[3],pVertex->coarse_len);	
				for (i = 0; i < pVertex->coarse_len; i++)
					fprintf(fp,"%d %1.6E ",pVertex->iCoarseIndex[i],pVertex->weight[i]);
				fprintf(fp,"\n");
				++pVertex;
			};
		} else {
			fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz);
			pVertex = X;
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
				fprintf(fp,"%d %1.14E %d ",iVertex,pVertex->epsilon[3],pVertex->coarse_len);	
				for (i = 0; i < pVertex->coarse_len; i++)
					fprintf(fp,"%d %1.6E ",pVertex->iCoarseIndex[i],pVertex->weight[i]);
				fprintf(fp,"\n");
				++pVertex;
			};
		}
		fclose(fp);*/
/*
		// ==========================================

		// DEBUG:
//#define DBG_0
#ifdef DBG_0

			IterationsJRLS(-1,40);

#else

		bGlobalSpitFlag = ((iIteration % 10 == 0)
						|| (iLevel >= NUM_COARSE_LEVELS-2));
		
		// Test the neighbours of 3529.
		/*
		long neigh_len, izNeigh[128];
		Vertex * pNeigh;
		neigh_len = AuxX[0][3529].GetNeighIndexArray(izNeigh);
		FILE * fp3 = fopen("testfile3529.txt","w");
		
		Lift_to_coarse_eps(0);	this->CalculateEpsilonAux4(0);
		fprintf(fp3,"initially %1.14E \n",AuxX[0][3529].epsilon[GAUSS]);
		iLevel = -1;
		
		for (i = 0; i < neigh_len; i++)
		{
			Lift_to_coarse_eps(0);
			// Change Ay value at a neighbour
			pNeigh = AuxX[0]+izNeigh[i];
			pNeigh->A.y = 10.0;
			Affect_vars_finer(-1);
			this->CalculateEpsilons();
			Lift_to_coarse_eps(0);
			this->CalculateEpsilonAux4(0);
			// What is epsilon at 3529? What is coefficient on this neighbour phi?
			// Spit both out.
			fprintf(fp3,"%d %1.14E %1.14E \n",
				izNeigh[i], AuxX[0][3529].epsilon[GAUSS], AuxX[0][3529].coeff[i].co[0][AY]*10.0);
			// Restore the status quo ante.
			iLevel = -1;
			pVertex = X + 13000;
			for (iVertex = 13000; iVertex < 15000; iVertex++)
			{
				pVertex->A.x = 0.0;
				pVertex->A.y = 0.0;
				pVertex->A.z = 0.0;
				pVertex->phi = 0.0;
				++pVertex;
			};
		}; 
		fclose(fp3);
		// there is another possibility: a wrong coefficient on itself
		printf("all done");
		while (1) getch();
		*/
/*
		if (iLevel == NUM_COARSE_LEVELS-1)
		{
			
			printf("RunLU...\n");
			RunLU(iLevel,bRefreshLUCoeffs);
			bRefreshLUCoeffs = 0;
			
		} else {
			if ((iLevel == -1) || (times_up[iLevel] == 0))
			{
				
				printf("iLevel = %d; numVolleys = %d \n",iLevel,numVolleys[0]);
				if (iLevel == -1) {
					CalculateEpsilons();
				} else {
					CalculateEpsilonAux4(iLevel);
				};
				
				//SpitOutGauss();
				for (iVolley = 1; iVolley <= 3; iVolley++)//numVolleys[iLevel+1]; iVolley++)
				{
					printf("got to here --- iVolley = %d\n",iVolley);
					GaussSeidel(iLevel,iVolley);
				//	SpitOutGauss();
				};
				IterationsJRLS(iLevel,4);			
				
				// so it IS doing GS on centres first. That cannot overcome the big errors.
				
			} else {
				IterationsJRLS(iLevel,ITERATIONS_at_levels[iLevel+1]);		
				
				CalculateEpsilonAux4(iLevel);
				/*for (iVolley = 0; iVolley < numVolleys[iLevel+1]; iVolley++)
				{
					printf("got to here --- iVolley = %d\n",iVolley);
					GaussSeidel(iLevel,iVolley);
				};*/
/*			};
			
			if (iLevel == -1) {
				// debug:
				
				// spit out some data for phi, Gauss equation, eps. First 2000 points.
				
				printf("output file ...");
				
				// ___________________________________________________________
				
				// What is phi_anode and how does it affect z-current?
				// Go back n create coefficient on it: sum of Iz[phi_anode].
				// Compute net effect of phi on Iz: sum of Iz[phi]*phi.
				// Likewise net effect of Az on Iz: sum of Iz[Az]*Az
				
				Iz_effect_of_phi = 0.0; Iz_effect_of_Az = 0.0;
				//Izcoeff_phianode = 0.0;
				
				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					Iz_effect_of_phi += pVertex->coeff_self[IZ][PHI]*pVertex->phi;
					Iz_effect_of_Az += pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
				//	Izcoeff_phianode += pVertex->coeff_self[IZ][PHI_ANODE]; // ??
					++pVertex;
				}
				real Iz = Epsilon_Iz - this->Epsilon_Iz_constant.x[0];
				printf("4pi/c Iz %1.10E \n",Iz);
				printf("Izcoeff[phi_anode] %1.8E phi_anode %1.8E Izcontrib %1.8E\n",
					this->Epsilon_Iz_coeff_On_PhiAnode.x[0], this->PhiAnode.x[0], (Epsilon_Iz_coeff_On_PhiAnode*PhiAnode).x[0]);
				printf("Iz effect of phi: %1.8E  effect of Az: %1.8E \n\n",
					Iz_effect_of_phi, Iz_effect_of_Az);
				// Doing this will explain how come we can have apparently correct current with phi_anode 3 times greater,
				// which is in itself a mystery.
				
				ctr++;
				sprintf(buffer,"Gauss__%d.txt",ctr);
				/*fp = fopen(buffer,"w");
				fprintf(fp,"4pi/c Iz %1.10E \n",Iz);
				fprintf(fp,"Izcoeff[phi_anode] %1.9E phi_anode %1.9E\n Izcontrib %1.9E\n",
					this->Epsilon_Iz_coeff_On_PhiAnode.x[0], PhiAnode.x[0], (Epsilon_Iz_coeff_On_PhiAnode*PhiAnode).x[0]);
				fprintf(fp,"Iz effect of phi: %1.8E  effect of Az: %1.8E \n\n",
					Iz_effect_of_phi, Iz_effect_of_Az);
				long neigh_len, izNeigh[100];

				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					real rr = pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y;
					//if ((rr > 3.44*3.44) && (rr < 3.9*3.9))
					{
						fprintf(fp,"%d %d %1.14E %1.14E | phi %1.14E Az %1.14E | eps_Gau %1.14E | on_phi %1.14E on_Az %1.14E on_1 %1.14E on_phi_anode %1.14E | ",
							iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y, pVertex->phi,pVertex->A.z,pVertex->epsilon[GAUSS],
							pVertex->coeff_self[GAUSS][PHI],pVertex->coeff_self[GAUSS][AZ],pVertex->coeff_self[GAUSS][UNITY],
							pVertex->coeff_self[GAUSS][PHI_ANODE]);
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i =0 ; i < neigh_len; i++)
							fprintf(fp,"%d %1.14E ",izNeigh[i],pVertex->coeff[i].co[GAUSS][PHI]);
						fprintf(fp,"\n");
					};
					++pVertex;
				}
				fclose(fp);
				sprintf(buffer,"Az__%d.txt",ctr);
				fp = fopen(buffer,"w");
				fprintf(fp,"4pi/c Iz %1.10E \n",Iz);
				fprintf(fp,"Izcoeff[phi_anode] %1.9E phi_anode %1.9E\n Izcontrib %1.9E\n",
					this->Epsilon_Iz_coeff_On_PhiAnode.x[0], PhiAnode.x[0], (Epsilon_Iz_coeff_On_PhiAnode*PhiAnode).x[0]);
				fprintf(fp,"Iz effect of phi: %1.8E  effect of Az: %1.8E \n\n",
					Iz_effect_of_phi, Iz_effect_of_Az);
			//	long neigh_len, izNeigh[100];

				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					real rr = pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y;
					//if ((rr > 3.44*3.44) && (rr < 3.9*3.9))
					{
						fprintf(fp,"%d %d %1.14E %1.14E | phi %1.14E Az %1.14E | eps_Gau %1.14E | on_phi %1.14E on_Az %1.14E on_1 %1.14E on_phi_anode %1.14E | ",
							iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y, pVertex->phi,pVertex->A.z,pVertex->epsilon[AZ],
							pVertex->coeff_self[AZ][PHI],pVertex->coeff_self[AZ][AZ],pVertex->coeff_self[AZ][UNITY],
							pVertex->coeff_self[AZ][PHI_ANODE]);
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i =0 ; i < neigh_len; i++)
							fprintf(fp,"%d %1.14E ",izNeigh[i],pVertex->coeff[i].co[AZ][AZ]);
						fprintf(fp,"\n");
					};
					++pVertex;
				}
				fclose(fp);
				printf("done\n");*/
/*
				// We will look and see what it is about the 'solution' that creates the largest error.
				
			} else {
				// debug:
				// What is phi_anode and how does it affect
				
				printf("phi_anode at lvl %d = %1.9E\n",iLevel,this->PhiAnode_aux[iLevel].x[0]);
				
			};
		};
#endif

		if ((iLevel == -1) && (iIteration % 1 == 0)) {
			
			if (iIteration % 1 == 0) {
				printf("0= Gauss graphs,1= Amp xy,2= Amp z,3= eps:");
				o = (char)getch();
				if (o == 'f') {
					// switch to finest
					bMultimesh = 0;
					printf("f\n");
				};
				if (o == 'm') {
					bMultimesh = true;
					printf("m\n");
				};
				if (o == '0') {
					printf("0");
					// Create Exy (from all sources) and rho..
					
					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.x = 0.0; pVertex->E.y = 0.0;
						pVertex->temp2.x = 0.0; // use Temp for J. temp2.y is shape of Ez field.
						++pVertex;
					};
					graphswitch = 0;
				}
				if (o == '1') {
					printf("1");

					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.x = 0.0; pVertex->E.y = 0.0;
						memset(&(pVertex->Temp),0,sizeof(Vector3)); // J
						
						++pVertex;
					};
					graphswitch = 1;
				}
				if (o == '2') {
					printf("2");
					// dummy values for now:
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						pVertex->E.z = 0.0; 
						memset(&(pVertex->Temp),0,sizeof(Vector3)); // J
						
						++pVertex;
					};
					graphswitch = 2;
				}
				if (o == '3') {
					printf("3");
					graphswitch = 3;				
				}
				printf("\n");
			};

			switch(graphswitch) {
				case 0:
						
					Graph[0].DrawSurface("phi",
					  DATA_HEIGHT,(real *)(&(this->X[0].phi)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].phi)),
					  true, 
					  GRAPH_PHI, this);
					Graph[1].DrawSurface("Exy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].E)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].E)),
					  true,
					  GRAPH_EXY, this);
					Graph[2].DrawSurface("eps_Gauss",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[0])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[0])),
					  true,
					  GRAPH_EPS0, this);
					Graph[3].DrawSurface("rho",
					  DATA_HEIGHT,(real *)(&(this->X[0].temp2.x)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].temp2.x)),
					  true,
					  GRAPH_RHO, this);		
					break;
				case 1:
					
					Graph[0].DrawSurface("Axy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].A.x)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].A.x)),
					  true, 
					  GRAPH_AXY, this);
					Graph[1].DrawSurface("eps_Ax",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS1, this);
					Graph[2].DrawSurface("eps_Axy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS12, this);
					Graph[3].DrawSurface("Ax",
					  DATA_HEIGHT,(real *)(&(this->X[0].A.x)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].A.x)),
					  true,
					  GRAPH_AX, this);		
					break;
				case 2:
					Graph[0].DrawSurface("Az",
					  DATA_HEIGHT,(real *)(&(this->X[0].A.z)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].A.z)),
					  true, 
					  GRAPH_AZ, this);
					Graph[1].DrawSurface("Exy",
					  VELOCITY_HEIGHT,(real *)(&(this->X[0].E)),
					  VELOCITY_COLOUR,(real *)(&(this->X[0].E)),
					  true,
					  GRAPH_EXY, this);
					Graph[2].DrawSurface("eps_z",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[3])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[3])),
					  true,
					  GRAPH_EPS3, this);
					Graph[3].DrawSurface("Jz",
					  DATA_HEIGHT,(real *)(&(this->X[0].Temp.z)),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].Temp.z)),
					  true,
					  GRAPH_JZ, this);
					break;
				case 3:
					Graph[0].DrawSurface("eps_Gauss",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[0])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[0])),
					  true, 
					  GRAPH_AZ, this);
					Graph[1].DrawSurface("eps_Ampx",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[1])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[1])),
					  true,
					  GRAPH_EPS1, this);
					Graph[2].DrawSurface("eps_Ampz",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[3])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[3])),
					  true,
					  GRAPH_EPS3, this);
					Graph[3].DrawSurface("eps_Ampy",
					  DATA_HEIGHT,(real *)(&(this->X[0].epsilon[2])),
					  AZSEGUE_COLOUR,(real *)(&(this->X[0].epsilon[2])),
					  true,
					  GRAPH_EPS2, this);
				break;
			};
			
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
			
			 // whether finest level, to do graphs
		};


		if (iLevel == -1) {

			// Test whether finest condition satisfied:
			// ========================================
			
			CalculateEpsilons();		
			// Survey RSS from each equation:
			if (iIteration % 4 == 0) {
				printf("sumeps ");
			};
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				RSS = 0.0;
				sumeps = 0.0;
				pVertex = X;
				for (iVertex =0; iVertex < numVertices; iVertex++)
				{
					RSS += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					sumeps += pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSSArray[iEqn] = RSS;
				Eeps_array[iEqn] = sumeps/(real)numVertices;

				if (iIteration % 4 == 0) {
					printf("%1.6E ",sumeps);
				};
			};
			RSSArray[4] = this->Epsilon_Iz*this->Epsilon_Iz; // error in 4pi/c Iz
			if (iIteration % 4 == 0) {
				printf("\n");
			};	
			printf("RSS Gau %1.5E Amp %1.5E %1.5E %1.5E Iz %1.5E\n",
					RSSArray[0],RSSArray[1],RSSArray[2],RSSArray[3],RSSArray[4]);
			
			// What do we want? Look at other file - see how to do FP comparison.
			
			CalculateEpsilonsAbsolute(RSS_Absolute_array);
			
			Finest_condition_known_satisfied = true;
			for (iEqn = 0; iEqn < NUMCELLEQNS; iEqn++)
			{
				if (RSSArray[iEqn] != 0.0)
				{
					variance = RSSArray[iEqn]/(real)numVertices - Eeps_array[iEqn]*Eeps_array[iEqn];
					SD = sqrt(variance);						
					
					L2abs = sqrt(RSS_Absolute_array[iEqn]/(real)numVertices);
					thresh = ODE_FPRATIO*L2abs + ODE_ABSTHRESHOLD[iEqn];
					
					printf("iEqn %d  thresh %1.8E  SD %1.8E \n",iEqn,thresh,SD);
					if (SD > thresh) Finest_condition_known_satisfied = false;				
					// Think about this: should be doing Axy jointly.				
				};
			};

			// RSS_Absolute_array[IZ] is a positive number, Iz_prescribed is a negative number.

			Iz_attained = RSS_Absolute_array[IZ]; // involves 4pi/c factor
			thresh = fabs( ODE_IZRELTOL*Iz_prescribed*FOURPI_OVER_C );
			
			Iz_diff = fabs(Iz_attained - FOURPI_OVER_C*Iz_prescribed); 
			
			printf("4pi/c Iz_presc %1.6E attain %1.6E diff %1.6E thresh %1.6E \n",
				Iz_prescribed*FOURPI_OVER_C, RSS_Absolute_array[IZ],
				Iz_diff, thresh);
			bSucceedIz = (Iz_diff < thresh);
			if (bSucceedIz == false) Finest_condition_known_satisfied = false;
			
			if (Finest_condition_known_satisfied == false)
			{

// DEBUG:
#ifndef DBG_0
				if (bMultimesh) {
					Lift_to_coarse_eps(0);
					iLevel = 0;
					times_up[0] = 0;

					printf("AuxX[0][3529].coeff_self[0][UNITY] %1.14E \n",
						AuxX[0][3529].coeff_self[0][UNITY]);
					getch();
					getch();
				};
#endif
				
				
				// Test descend/ascend: done
				
				//fp = fopen("ascend.txt","w");
				//this->CalculateEpsilonAux4(0);
				//fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[0]);
				//pVertex = AuxX[0];
				//for (iVertex = 0; iVertex < numAuxVertices[0]; iVertex++)
				//{
				//	fprintf(fp,"%d %1.14E \n",iVertex,pVertex->epsilon[3]);	
				//	++pVertex;
				//};
				//fclose(fp);

				//this->Affect_vars_finer(iLevel);	
				//this->CalculateEpsilons();
				//Lift_to_coarse_eps(0); // nothing happened so this isn't even a Galerkin test.

				//fp = fopen("reascend.txt","w");
				//this->CalculateEpsilonAux4(0);
				//fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[0]);
				//pVertex = AuxX[0];
				//for (iVertex = 0; iVertex < numAuxVertices[0]; iVertex++)
				//{
				//	fprintf(fp,"%d %1.14E \n",iVertex,pVertex->epsilon[3]);	
				//	++pVertex;
				//};
				//fclose(fp);
				
			};

		} else {
			// not finest level

			this->CalculateEpsilonAux4(iLevel);
			// Collect RSS:
			
			memset(RSSArray,0,sizeof(real)*4);
			sumeps = 0.0;
			pVertex = AuxX[iLevel];
			for (iVertex =0; iVertex < numAuxVertices[iLevel]; iVertex++)
			{
				RSSArray[0] += pVertex->epsilon[0]*pVertex->epsilon[0];
				RSSArray[1] += pVertex->epsilon[1]*pVertex->epsilon[1];
				RSSArray[2] += pVertex->epsilon[2]*pVertex->epsilon[2];
				RSSArray[3] += pVertex->epsilon[3]*pVertex->epsilon[3];
				++pVertex;
			};
			printf("Level %d RSS %1.5E %1.5E %1.5E %1.5E\n",
				iLevel,RSSArray[0],RSSArray[1],RSSArray[2],RSSArray[3]);

			if ((iLevel == NUM_COARSE_LEVELS-1)
				|| (times_up[iLevel] >= CYCLEPEAKS)) // consider changing this to 3, or running longer at Lvl 1 intermediately.
			{
			
				// descend:
				// Do a Galerkin test instead:

				sprintf(buffer,"lvl%d_b4desc_II_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					fprintf(fp,"iVertex %d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E coarse_len %d \n",
						iVertex,pVertex->pos.x,pVertex->pos.y,
						pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
						pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
						pVertex->coarse_len);
					++pVertex;
				};
				fclose(fp);

				store_before_epsIz = Epsilon_Iz_aux[iLevel];
				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					pVertex->a_pressure_neut_or_overall.x = pVertex->epsilon[0];
					pVertex->Temp.x = pVertex->epsilon[1];
					pVertex->Temp.y = pVertex->epsilon[2];
					pVertex->Temp.z = pVertex->epsilon[3]; // store it
					++pVertex;
				};

				iLevel--;

				if (iLevel >= 0) {
					// spit out what's here before descend:

					sprintf(buffer,"lvl%d_predesc_%d.txt",iLevel,counter1);
					fp = fopen(buffer,"w");

					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
					pVertex = AuxX[iLevel];
					for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E %d ",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->coarse_len);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						++pVertex;
					};
					fclose(fp);
				} else {
					// only spit out where relevant.

					sprintf(buffer,"lvl%d_predesc_%d.txt",iLevel,counter1);
					fp = fopen(buffer,"w");
					FILE * fp2 = fopen("vals_b4.txt","w");
					pVertex = X + 13000;
					for (iVertex = 13000; iVertex < 15000; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E %d ",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->coarse_len);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][0],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						
						
						if ((pVertex->Temp.x != 0.0) ||
							(pVertex->Temp.y != 0.0)) 	
						{
							fprintf(fp2,"iVertex %d coeffs %1.14E %1.14E %1.14E %1.14E vals %1.14E %1.14E %1.14E %1.14E  \n",
								iVertex,pVertex->Temp.x,pVertex->Temp.y,pVertex->Temp.z,pVertex->temp2.x,
								pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z);
						}

						++pVertex;
					};
					fclose(fp);
					fclose(fp2);
				};

				this->Affect_vars_finer(iLevel);	

				if (iLevel == -1) {
					this->CalculateEpsilons();
					printf("After descend: Eps_Iz[-1]: %1.14E \n",Epsilon_Iz);
				} else {
					this->CalculateEpsilonAux4(iLevel);
					printf("After descend: Eps_Iz[%d]: %1.14E \n",iLevel,Epsilon_Iz_aux[iLevel]);
				};

				sprintf(buffer,"lvl%d_desc_II_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				if (iLevel >= 0) {
					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz_aux[iLevel]);
					pVertex = AuxX[iLevel];
					for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
					{
						fprintf(fp,"%d %1.14E %1.14E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E | %d |",
							iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->iVolley);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"| %d %1.6E %d ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][GAUSS],pVertex->PBC_uplink[i]);
						fprintf(fp,"\n");
						++pVertex;
					};
				} else {
					fprintf(fp,"Eps_Iz: %1.14E \n",Epsilon_Iz);

					FILE * fp2 = fopen("vals_after.txt","w");
					pVertex = X;
					for (iVertex = 0; iVertex < numVertices; iVertex++)
					{
						fprintf(fp,"%d %1.10E %1.10E phi %1.14E A %1.14E %1.14E %1.14E eps %1.14E %1.14E %1.14E %1.14E | %d | ",iVertex,
							pVertex->pos.x,pVertex->pos.y,
							pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
							pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
							pVertex->iVolley);	
						for (i = 0; i < pVertex->coarse_len; i++)
							fprintf(fp,"%d %1.10E ",pVertex->iCoarseIndex[i],pVertex->wt_var[i][GAUSS]);
						fprintf(fp,"\n");

						if ((pVertex->Temp.x != 0.0) ||
							(pVertex->Temp.y != 0.0)) 	
						{
							fprintf(fp2,"iVertex %d coeffs %1.14E %1.14E %1.14E %1.14E vals %1.14E %1.14E %1.14E %1.14E  \n",
								iVertex,pVertex->Temp.x,pVertex->Temp.y,pVertex->Temp.z,pVertex->temp2.x,
								pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z);
						}
						++pVertex;
					};
					fclose(fp2);
				}
				fclose(fp);


				Lift_to_coarse_eps(iLevel+1);
				iLevel++;
				counter1++;
				sprintf(buffer,"lvl%d_GalerkinII_%d.txt",iLevel,counter1);
				fp = fopen(buffer,"w");

				this->CalculateEpsilonAux4(iLevel);
				fprintf(fp,"Eps_Iz: %1.14E Before: %1.14E \n",Epsilon_Iz_aux[iLevel],store_before_epsIz);
				pVertex = AuxX[iLevel];
				for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
				{
					fprintf(fp,"%d %1.14E %1.14E %1.14E %1.14E before: %1.14E %1.14E %1.14E %1.14E\n",
						iVertex,
						pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
						pVertex->a_pressure_neut_or_overall.x,pVertex->Temp.x,pVertex->Temp.y,pVertex->Temp.z);	
					++pVertex;
				}; 
				fclose(fp);

				// go back down:
				iLevel--;

				if (iLevel == -1) {
					printf("all done.\n");
					while (1) getch();
				};
			} else {

				// NOTICE THE FOLLOWING ASSUMES WE HAVE DONE CALC EPS AT OUR OWN LEVEL!
				// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	//			// Ascend:
				times_up[iLevel]++;
				Lift_to_coarse_eps(iLevel+1);
				iLevel++;
				times_up[iLevel] = 0;
	//			// If we arrive on a level from below, set times_up to 0
			};
		};		
		iIteration++;
	}; // while finest not satisfied

	printf("Convergence criteria satisfied. Time elapsed %s ",report_time(1));

	// The post-solve:
	// Use same method as in coefficients to do the ion and electron displacement.
	// To do neutral displacement can do any way; use pVertex->sigma_n.

	if (!bInitial) {
		printf("DONE");
		getch();getch();getch();
	}

	if (bInitial) {

		Vector3 A_past;

		// Create Vertex::E
		// Create Jz
		// Create A-dot _k-1/2

		//real factor = 1.0/(1.0 + Y.Unreduced_beta_e_ion.zz*m_e/m_i); // assume have +beta(-me/mi)ve and move to LHS
		//pVertex->v_e_0.z = factor*Y.Unreduced_v_e_0.z;
		//pVertex->sigma_e.zz = factor*Y.Unreduced_sigma_e.zz;
		//pVertex->v_i_0.z = -(m_e/m_i)*pVertex->v_e_0.z;
		//pVertex->sigma_i.zz = -(m_e/m_i)*pVertex->sigma_e.zz;
		long izNeigh[128];
		long neigh_len;
		FILE * fp = fopen("outputpostsolve.txt","w");
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->E.x = 0.0;
			pVertex->E.y = 0.0;
			pVertex->E.z = // assumptions? = -Adot/c + Ez_ext
				-(GlobalIzElasticity*pVertex->A.z)/c + (PhiAnode*pVertex->Ez_coeff_on_phi_anode).x[0]
												     + pVertex->phi*pVertex->Ez_coeff_on_phi;

			pVertex->Ion.mom = pVertex->Ion.mass*(pVertex->v_i_0 + pVertex->sigma_i*pVertex->E);
			pVertex->Elec.mom = pVertex->Elec.mass*(pVertex->v_e_0 + pVertex->sigma_e*pVertex->E);

			// now go back in time to make Adot in the past: time_back_for_Adot_if_initial

			A_past = pVertex->A*(1.0-time_back_for_Adot_if_initial*GlobalIzElasticity);
			pVertex->Adot = A_past*GlobalIzElasticity;

			// Do file output
			
			fprintf(fp,"%d %d %1.14E %1.14E cent %1.14E %1.14E Ez %1.14E Az %1.14E temp2.y %1.14E TuneFac %1.14E "
				"Ion.vz %1.14E Elec.vz %1.14E n %1.14E sigma_i %1.14E sigma_e %1.14E ionmomz %1.14E elecmomz %1.14E ",
				iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y,pVertex->centroid.x,pVertex->centroid.y,
				pVertex->E.z,pVertex->A.z,pVertex->temp2.y,PhiAnode.x[0],
				pVertex->Ion.mom.z/pVertex->Ion.mass,pVertex->Elec.mom.z/pVertex->Elec.mass,
				pVertex->Elec.mass/pVertex->AreaCell,pVertex->sigma_i.zz,pVertex->sigma_e.zz,
				pVertex->Ion.mom.z,pVertex->Elec.mom.z);
			fprintf(fp,"coeff_self %1.14E | %1.14E %1.14E %1.14E | %1.14E %1.14E ",
				pVertex->coeff_self[AMPZ][PHI],pVertex->coeff_self[AMPZ][AX],
				pVertex->coeff_self[AMPZ][AY],pVertex->coeff_self[AMPZ][AZ],
				pVertex->coeff_self[AMPZ][UNITY],pVertex->coeff_self[AMPZ][PHI_ANODE]);
			
			fprintf(fp,"coeffs_on_Az  ");
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i = 0; i < neigh_len; i++)
			{
				fprintf(fp,"%d %1.14E ",izNeigh[i],pVertex->coeff[i].co[AMPZ][AZ]);
			}
			fprintf(fp,"\n");
			++pVertex;
		};
		fclose(fp);
		printf("done file output.\n");

	} else {
		
		// Create Vertex::E


		// ...
	}

	// Species relative displacement has to give rise to extra comp htg / expansion cooling.

	// When do we create B ? 
	// In stage III, is it worth tweening from Bk since we still have it? For the good that will do?

	// delete[] to match new Vertex :
	for (i = 0; i < NUM_COARSE_LEVELS; i++)
		delete[] AuxX[i];*/

}


void TriMesh::CalculateEpsilons()
{
	// Each vertex, have 4 epsilon to calculate.
	// Also accumulate error in Iz equation.
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn;
	long izNeigh[128];
	int neigh_len;
	Vector3 A;
	real phi;
	int iNeigh;
	real coefflocal[4][4];
	real coeffself[5][6];
	real pVertexphi;
	Vector3 pVertexA;

	Epsilon_Iz = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_PhiAnode*PhiAnode).x[0]; 
	real f64PhiAnode = PhiAnode.x[0];

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// coeff_self first:
		memcpy (&(coeffself[0][0]),pVertex->coeff_self,sizeof(real)*5*6);
		pVertexphi = pVertex->phi;
		pVertexA = pVertex->A;
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			pVertex->epsilon[iEqn] = coeffself[iEqn][UNITY]
							 + coeffself[iEqn][PHI_ANODE]*f64PhiAnode // !!
							 + coeffself[iEqn][PHI]*pVertexphi
							 + coeffself[iEqn][AX]*pVertexA.x
							 + coeffself[iEqn][AY]*pVertexA.y
							 + coeffself[iEqn][AZ]*pVertexA.z;
			// DEBUG:
//
//			if (isnan(pVertex->epsilon[iEqn])) {
//			
//				printf("iVertex %d iEqn %d failed at self :\n",iVertex,iEqn);
//				printf("pVertex->coeff_self[iEqn]: %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E \n",
//					pVertex->coeff_self[iEqn][0],
//					pVertex->coeff_self[iEqn][1],
//					pVertex->coeff_self[iEqn][2],
//					pVertex->coeff_self[iEqn][3],
//					pVertex->coeff_self[iEqn][4],
//					pVertex->coeff_self[iEqn][5]);
//				printf("phi A %1.14E %1.14E %1.14E %1.14E \n",
//					pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z);
//				printf("f64TuneFac %1.14E \n",f64TuneFac);
//					getch();
//			}
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = X + izNeigh[iNeigh];
			phi = pNeigh->phi;
			A = pNeigh->A;		
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				pVertex->epsilon[iEqn] += coefflocal[iEqn][PHI]*phi
										+ coefflocal[iEqn][AX]*A.x
										+ coefflocal[iEqn][AY]*A.y
										+ coefflocal[iEqn][AZ]*A.z;

				// DEBUG:

				//if (isnan(pVertex->epsilon[iEqn])) {
				//	printf("iVertex %d iEqn %d iNeigh %d \n",iVertex,iEqn,iNeigh);
				//	printf("coefflocal %1.14E %1.14E %1.14E %1.14E\n",coefflocal[iEqn][PHI],
				//		coefflocal[iEqn][AX],coefflocal[iEqn][AY],coefflocal[iEqn][AZ]);
				//	printf("phi A %1.14E %1.14E %1.14E %1.14E\n",phi,A.x,A.y,A.z);
				//	getch();
				//}
			};
		};

		Epsilon_Iz += //pVertex->coeff_self[IZ][UNITY] +
					// pVertex->coeff_self[IZ][TUNEFAC]*f64TuneFac +
					 pVertex->coeff_self[IZ][PHI]*pVertex->phi +
					 pVertex->coeff_self[IZ][AX]*pVertex->A.x +
					 pVertex->coeff_self[IZ][AY]*pVertex->A.y +
					 pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
		
		// Let's be careful:
		// did we concatenate all EpsIz coefficients to refer from the affecting vertex?

		// TuneFac * temp2.y is Ez_ext but temp2.y is already appearing in the coefficient.
		
		++pVertex;
	};	
}
void TriMesh::CalculateEpsilonsAz()
{
	// Each vertex, have 4 epsilon to calculate.
	// Also accumulate error in Iz equation.
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn;
	long izNeigh[128];
	int neigh_len;
	Vector3 A;
	real phi;
	int iNeigh;
	real coefflocal[4][4];
	real coeffself[5][6];
	real pVertexphi;
	Vector3 pVertexA;

	Epsilon_Iz = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_PhiAnode*EzTuning).x[0]; 
	real f64EzTuning = EzTuning.x[0]; // apparently that's going to be dd_real
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// coeff_self first:
		memcpy (&(coeffself[0][0]),pVertex->coeff_self,sizeof(real)*5*6);
		pVertexA = pVertex->A;
		iEqn = AMPZ;
		{
			pVertex->epsilon[iEqn] = coeffself[iEqn][UNITY]
							 + coeffself[iEqn][EZTUNING]*f64EzTuning
							 + coeffself[iEqn][AZ]*pVertexA.z;
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = X + izNeigh[iNeigh];
			A = pNeigh->A;		
			iEqn = AMPZ;
			{
				pVertex->epsilon[iEqn] += coefflocal[iEqn][AZ]*A.z;
			};
		};

		Epsilon_Iz += pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
		
		// Let's be careful:
		// did we concatenate all EpsIz coefficients to refer from the affecting vertex?
		// TuneFac * temp2.y is Ez_ext but temp2.y is already appearing in the coefficient.
		
		++pVertex;
	};	
}



void TriMesh::CalculateEpsilonsAbsolute(real RSS_Absolute_array[4])
{
	// Each vertex, have 4 epsilon to calculate.
	// Also accumulate error in Iz equation.
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn;
	long izNeigh[128];
	int neigh_len,iNeigh;
	Vector3 A;
	real phi;
	real coefflocal[4][4];
	real epsilon[5];
	memset(RSS_Absolute_array,0,sizeof(real)*5);

	//Epsilon_Iz = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_TuningFactor*TuneFac).x[0]; 
	RSS_Absolute_array[4] = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_PhiAnode*PhiAnode).x[0]; 
	real f64PhiAnode = PhiAnode.x[0];
		
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// coeff_self first:
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			epsilon[iEqn] = fabs(pVertex->coeff_self[iEqn][UNITY])
							 + fabs(pVertex->coeff_self[iEqn][PHI_ANODE]*f64PhiAnode)
							 + fabs(pVertex->coeff_self[iEqn][PHI]*pVertex->phi)
							 + fabs(pVertex->coeff_self[iEqn][AX]*pVertex->A.x)
							 + fabs(pVertex->coeff_self[iEqn][AY]*pVertex->A.y)
							 + fabs(pVertex->coeff_self[iEqn][AZ]*pVertex->A.z);
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = X + izNeigh[iNeigh];
			phi = pNeigh->phi;
			A = pNeigh->A;		
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				epsilon[iEqn] += fabs(coefflocal[iEqn][PHI]*phi)
										+ fabs(coefflocal[iEqn][AX]*A.x)
										+ fabs(coefflocal[iEqn][AY]*A.y)
										+ fabs(coefflocal[iEqn][AZ]*A.z);
			};
		};

		RSS_Absolute_array[4] += 
					// pVertex->coeff_self[IZ][UNITY] +
					 // pVertex->coeff_self[IZ][TUNEFAC]*f64TuneFac +  // these should == 0
					 pVertex->coeff_self[IZ][PHI]*pVertex->phi +
					 pVertex->coeff_self[IZ][AX]*pVertex->A.x +
					 pVertex->coeff_self[IZ][AY]*pVertex->A.y +
					 pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
		
		RSS_Absolute_array[0] += epsilon[0]*epsilon[0];
		RSS_Absolute_array[1] += epsilon[1]*epsilon[1];
		RSS_Absolute_array[2] += epsilon[2]*epsilon[2];
		RSS_Absolute_array[3] += epsilon[3]*epsilon[3];
		
		// RSS_Absolute_array[4] will == Iz_attained.
		// Let's go back n check: we are adding coeff each vertex per TUNEFAC but
		// that's not right is it.
		++pVertex;
	};	
	// RSS_Absolute_array[4]: create as for epsilonIz but add + Iz_prescribed 4pi/c

	RSS_Absolute_array[4] += FOURPI_OVER_C*this->Iz_prescribed; // = 4pi/c Iz_attained

}

void TriMesh::CalculateEpsilonsAbsoluteAz(real RSS_Absolute_array[4])
{
	// Each vertex, have 4 epsilon to calculate.
	// Also accumulate error in Iz equation.
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn;
	long izNeigh[128];
	int neigh_len,iNeigh;
	Vector3 A;
	real phi;
	real coefflocal[4][4];
	real epsilon[5];
	memset(RSS_Absolute_array,0,sizeof(real)*5);

	//Epsilon_Iz = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_TuningFactor*TuneFac).x[0]; 
	RSS_Absolute_array[4] = (Epsilon_Iz_constant + Epsilon_Iz_coeff_On_PhiAnode*EzTuning).x[0]; 
	real f64EzTuning = EzTuning.x[0];
		
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// coeff_self first:
		iEqn = AMPZ;
		{
			epsilon[iEqn] = fabs(pVertex->coeff_self[iEqn][UNITY])
							 + fabs(pVertex->coeff_self[iEqn][EZTUNING]*f64EzTuning)
							 + fabs(pVertex->coeff_self[iEqn][AZ]*pVertex->A.z);
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = X + izNeigh[iNeigh];
			A = pNeigh->A;		
			iEqn = AMPZ;
			{
				epsilon[iEqn] += fabs(coefflocal[iEqn][AZ]*A.z);
			};
		};

		RSS_Absolute_array[4] += 
					// pVertex->coeff_self[IZ][UNITY] +
					 // pVertex->coeff_self[IZ][TUNEFAC]*f64TuneFac +  // these should == 0
					 pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
		
		RSS_Absolute_array[3] += epsilon[3]*epsilon[3];
		
		// RSS_Absolute_array[4] will == Iz_attained.
		// Let's go back n check: we are adding coeff each vertex per TUNEFAC but
		// that's not right is it.
		++pVertex;
	};	
	// RSS_Absolute_array[4]: create as for epsilonIz but add + Iz_prescribed 4pi/c

	RSS_Absolute_array[4] += FOURPI_OVER_C*this->Iz_prescribed; // = 4pi/c Iz_attained x
}

void TriMesh::IterationsJRLS(int iLevel, int iterations)
{
	real ROC_0_3,ROC_1_3,ROC_ext;
	real * pRelevant;
	real ratio;
	int iEdge;
	static int const NUMREGRESStimesNUMCELLEQNS = NUMREGRESS*NUMCELLEQNS;
	Triangle * pTri,*pTri2;
	long iTri;
	long iIteration, iVariable, iEqn, iRegrVec, iGamma, iGamma1, iGamma2, iRegrVec1, iRegrVec2;
	long i,j, iVertex,len, i1, i2;
	real sum_Az,sum_phi,avg_Az,avg_phi;
	real old_RSS;
	Vertex * pVertex, *pNeigh, *Xarray;
	real residual, coeffsum, SumEps;
	int iCorner;
	real TempMES[3], tempgamma[3];
	real GlobalStoreBeta, GlobalEpsilonExist,maxeps, dbyd1;
	long iMax,numUse;	
	real overL2JA,overL2Jv,overL2RA,overL2Rv;

	real static const overch = 1.0/(c*h);	
	real const Change_chEz = 1.0;

	real SSreg[NUMREGRESS][NUMCELLEQNS];
	real L2reg[NUMREGRESS][NUMCELLEQNS];

	real RSStot;
	real predict_eps_Iz =0.0;

	Matrix Summary, Storemat, Tempsumm;	

	qd_or_d debugLU[GAMMANUM][GAMMANUM];
	qd_or_d minus_eps_sums[GAMMANUM], gamma[GAMMANUM];
	
	real BetaX[NUM_EQNS_1][NUMREGRESS][NUM_AFFECTORS_1];
	real BetaX_ext[NUM_EQNS_1]; // The 9th regressor
	// The extra equation:
	qd_or_d BetaX_for_Iz[NUMREGRESS][NUM_AFFECTORS_1];
	qd_or_d BetaX_for_Iz_ext;

	real data[NUM_AFFECTORS_1];
	real coefflocal[NUM_EQNS_1][NUM_AFFECTORS_1];
	real coeffself[NUM_EQNS_2][NUM_AFFECTORS_2]; // 2 > 1

	char buffer[256];
	static int counter = 0;

	real beta,sum_beta_sq,sum_eps_beta;
	long iNeigh, neigh_len;
	long izNeigh[128];
	long iVar, iWhich;

	Matrix_real LU4;
	real b[4],x[4];

//#define DEBUGJRLS

	LU4.Invoke(4);
	Summary.Invoke(GAMMANUM);
	Tempsumm.Invoke(3);
	Storemat.Invoke(GAMMANUM); // debugging

	HANDLE  hConsole;
    int k;
	real RSS[4],RSSpred[4];
	FILE * fp;
	

	fp = fopen("RSSnew__.txt","a");
	
	fprintf(fp,"------------------\n");

	// DEBUG:
	
//	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel];
		numUse = numAuxVertices[iLevel];
	};
	
	// The point of putting this loop inside is to retain any calcs and reduce overheads.
	for (iIteration = 0; iIteration < iterations; iIteration++)
	{
		if (iLevel == -1) {
			CalculateEpsilons();
		} else {
			CalculateEpsilonAux4(iLevel);
		};
		SpitOutGauss();

#ifdef DEBUGJRLS

		// DEBUG:
		// Survey RSS from each equation:
		RSStot = 0.0;

		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			RSS[iEqn] = 0.0;
			RSSpred[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				RSSpred[iEqn] += pVertex->predict_eps[iEqn]*pVertex->predict_eps[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};

		printf("RSS[]: %1.9E %1.9E %1.9E %1.9E\npred : %1.9E %1.9E %1.9E %1.9E \nepsIz %1.9E : %1.9E\n",
			RSS[0],RSS[1],RSS[2],RSS[3],		
			RSSpred[0],RSSpred[1],RSSpred[2],RSSpred[3],
			((iLevel >= 0)?Epsilon_Iz_aux[iLevel]:Epsilon_Iz),predict_eps_Iz);
		printf("RSStot %1.9E\n",RSStot);
#else
		RSStot = 0.0;
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			RSS[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};
#endif
		if ((iIteration > 0) && (RSStot > old_RSS*1.0000000000001))
		{
			printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
				old_RSS,RSStot);
			getch();
		}

		fprintf(fp,"RSS[]: %1.9E %1.9E %1.9E %1.9E EpsIz: %1.9E \n",
				RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		printf("RSS[%d]: %1.5E %1.5E %1.5E %1.5E EpsIz: %1.5E\n",iLevel,
				RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		old_RSS = RSStot;

		// 1. Define regressor vectors
		
		ZeroMemory(SSreg, sizeof(real)*NUMCELLEQNS*NUMREGRESS);
		real SSextra = 0.0;
		

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// Jacobi:
			// =======
			
			if ((pVertex->coeff_self[GAUSS][PHI_ANODE] == 0.0) || (pVertex->coeff_self[GAUSS][PHI] == 0.0))			
			{
				pVertex->circuit_regressor = 0.0;
			} else {
				// create "circuit field" regressor by changing phi to cancel out (for epsilon Gauss) a change of 1 in phi_anode.
				pVertex->circuit_regressor = -pVertex->coeff_self[GAUSS][PHI_ANODE]/pVertex->coeff_self[GAUSS][PHI];
				// should we be altering whole vector (A,phi) to cancel effect on all equations?
				// That will need to happen.
			};
			
			// Jacobi for 4 equations: Solve to make epsilon = 0 for these 4.

			// Beta_00 changephi + Beta_01 changeAx + Beta_02 changeAy + Beta_03 (changeAz) + epsilon_exist_0 = 0
			// 4x4 matrix: LU I guess.

			//LU4.LU[0][0] = pVertex->coeff_self[GAUSS][PHI];
			//LU4.LU[0][1] = pVertex->coeff_self[GAUSS][
			memcpy(LU4.LU[0],pVertex->coeff_self[GAUSS],sizeof(real)*4);
			memcpy(LU4.LU[1],pVertex->coeff_self[AMPX],sizeof(real)*4);
			memcpy(LU4.LU[2],pVertex->coeff_self[AMPY],sizeof(real)*4);
			memcpy(LU4.LU[3],pVertex->coeff_self[AMPZ],sizeof(real)*4);
			b[0] = -pVertex->epsilon[GAUSS];
			b[1] = -pVertex->epsilon[AMPX];  
			b[2] = -pVertex->epsilon[AMPY];
			b[3] = -pVertex->epsilon[AMPZ];
			
			if (LU4.LUdecomp() == 1) {
				printf("LU4.LUdecomp() failed.\n");
				getch();
			}

			// beta.change = -epsilon
			LU4.LUSolve(b,pVertex->regressor[JACOBI_REGR]);
			//memcpy(pVertex->regressor[JACOBI_REGR], x[0];
			// DEBUG! :
			if (  (_isnan(pVertex->regressor[JACOBI_REGR][0]))
				|| (_isnan(pVertex->regressor[JACOBI_REGR][1]))
				|| (_isnan(pVertex->regressor[JACOBI_REGR][2]))
				|| (_isnan(pVertex->regressor[JACOBI_REGR][3]))
				) {
				printf("NaN!"); getch();
			};
		//	printf("%d %1.6E %1.6E %1.6E %1.6E \n",iVertex,
		//		pVertex->regressor[JACOBI_REGR][0],
		//		pVertex->regressor[JACOBI_REGR][1],
		//		pVertex->regressor[JACOBI_REGR][2],
		//		pVertex->regressor[JACOBI_REGR][3]);

			memcpy(pVertex->regressor[RICHARDSON_REGR],pVertex->epsilon,sizeof(real)*NUM_AFFECTORS_1);
			
			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
				{
					SSreg[iRegrVec][iVar] += pVertex->regressor[iRegrVec][iVar] * 
											 pVertex->regressor[iRegrVec][iVar];
					if (!(SSreg[iRegrVec][iVar] == SSreg[iRegrVec][iVar])) {
						iVar = iVar;
					}
				};
			
			++pVertex;
		};
		
		// Normalize regressors:
		for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
			for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
				L2reg[iRegrVec][iVar] = sqrt(SSreg[iRegrVec][iVar]/(real)numUse);
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
			{
				 // do not divide by L2(var itself) which may be == 0
				{
					for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
						if (L2reg[iRegrVec][iVar] != 0.0)
							pVertex->regressor[iRegrVec][iVar] /= L2reg[iRegrVec][iVar];
				};
			};
			++pVertex;
		};
		
		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		
		for (i = 0; i < GAMMANUM; i++)
			memset(Summary.LU[i],0,sizeof(qd_or_d)*GAMMANUM);
		memset(minus_eps_sums,0,sizeof(qd_or_d)*GAMMANUM);
		
		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; 
		
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// The idea is to collect [Beta_ki dot x_i] [Beta_kj dot x_j] ,
			// so first collect [Beta_ki dot x_i],
			// for each eps k [k over all eqns at this vertex], each i [i over all regressors]
			
			// Each epsilon^2 gets an equal shout in sums and dot products here.
			
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
			};
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
				{
					for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
					{
						memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
						BetaX[iEqn][iRegrVec][0] += coefflocal[iEqn][0]*data[0];
						BetaX[iEqn][iRegrVec][1] += coefflocal[iEqn][1]*data[1];
						BetaX[iEqn][iRegrVec][2] += coefflocal[iEqn][2]*data[2];
						BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];
					};
					
					BetaX_ext[iEqn] += coefflocal[iEqn][0]*pNeigh->circuit_regressor;
				};				
			};
			
			for (iEqn = 0; iEqn < NUMCELLEQNS; iEqn++)
			{
				for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
				for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
				{
					i1 = iRegrVec1*NUMCELLEQNS+iGamma1;
					for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
					for (iGamma2 = 0; iGamma2 < NUMCELLEQNS; iGamma2++)
					{
						i2 = iRegrVec2*NUMCELLEQNS+iGamma2;
						Summary.LU[i1][i2] +=
							BetaX[iEqn][iRegrVec1][iGamma1]*BetaX[iEqn][iRegrVec2][iGamma2];
					};
				
					Summary.LU[i1][NUMREGRESStimesNUMCELLEQNS] += BetaX[iEqn][iRegrVec1][iGamma1]*BetaX_ext[iEqn];
					Summary.LU[NUMREGRESStimesNUMCELLEQNS][i1] += BetaX[iEqn][iRegrVec1][iGamma1]*BetaX_ext[iEqn];
					minus_eps_sums[i1] -= BetaX[iEqn][iRegrVec1][iGamma1]*pVertex->epsilon[iEqn];
				};
				Summary.LU[NUMREGRESStimesNUMCELLEQNS][NUMREGRESStimesNUMCELLEQNS] += 
																BetaX_ext[iEqn]*BetaX_ext[iEqn];
				minus_eps_sums[NUMREGRESStimesNUMCELLEQNS] -= BetaX_ext[iEqn]*pVertex->epsilon[iEqn];
				// Every epsilon should appear in every epsilon sum;
				// In sum j, each epsilon appears against the change in it due to regressor j.

			}; // next iEqn = next epsilon from this location
						
			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				BetaX_for_Iz[iRegrVec1][iGamma1] +=
						coeffself[IZ][iGamma1]*pVertex->regressor[iRegrVec1][iGamma1];
			};
			BetaX_for_Iz_ext += coeffself[IZ][PHI] * pVertex->circuit_regressor;

			++pVertex;
		};
		
		
		// still same regardless of level ...
		
		// Add to summary:
		for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				i1 = iRegrVec1*NUMCELLEQNS+iGamma1;
				for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
					for (iGamma2 = 0; iGamma2 < NUMCELLEQNS; iGamma2++)
						Summary.LU[i1][iRegrVec2*NUMCELLEQNS+iGamma2] +=
							BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz[iRegrVec2][iGamma2];

				Summary.LU[i1][NUMREGRESStimesNUMCELLEQNS] += BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz_ext;
				Summary.LU[NUMREGRESStimesNUMCELLEQNS][i1] += BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz_ext;
				minus_eps_sums[i1] -= BetaX_for_Iz[iRegrVec1][iGamma1]*
														((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
			};
		
		Summary.LU[NUMREGRESStimesNUMCELLEQNS][NUMREGRESStimesNUMCELLEQNS] += BetaX_for_Iz_ext*BetaX_for_Iz_ext;
		minus_eps_sums[NUMREGRESStimesNUMCELLEQNS] -= BetaX_for_Iz_ext*((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		
		bool bNonzero;
		for (i = 0; i < GAMMANUM; i++)
		{
			bNonzero = 0;
			for (j = 0; j < GAMMANUM; j++){
				if (Summary.LU[i][j] != 0.0)
					bNonzero = true;
				debugLU[i][j] = Summary.LU[i][j];
			};
			if (bNonzero == 0) {
				Summary.LU[i][i] = 1.0;
		//		printf("{ %d }",i);
			};
		};
		//printf("\n");
#ifdef DEBUGJRLS
		for (i = 0; i < GAMMANUM; i++)
		{
			for (j = 0; j < GAMMANUM; j++)
				printf("%1.2E ",Summary.LU[i][j].x[0]);
			printf("\n");
		}
		printf("\n");
		
		// debug:
		Storemat.CopyFrom(Summary);
#endif
		
		if (Summary.LUdecomp() == 1) {
			printf("LUdecomp failed.\n");
			getch();
		}
		Summary.LUSolve(minus_eps_sums, gamma); // solve matrix equation

#ifdef DEBUGJRLS
		printf("gam %1.3E %1.3E %1.3E %1.3E \n", gamma[0].x[0], gamma[1].x[0], gamma[2].x[0], gamma[3].x[0]);
		printf("--- %1.3E %1.3E %1.3E %1.3E %1.3E \n", gamma[4].x[0], gamma[5].x[0], gamma[6].x[0], gamma[7].x[0],gamma[8].x[0]);
		
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		// DEBUG:
		// Make predictions:
		real ROC_ext_comp3 = 0.0, ROC_ext_comp7 = 0.0, ROC_ext_comp8 = 0.0,
			ROC_ext_comp, ROC_ext_summ3 = 0.0, ROC_ext_summ7 = 0.0, ROC_ext_summ8 = 0.0,
			ROC_ext_summ, ROC_ext_eps = 0.0;
		ROC_0_3 = 0.0;
		ROC_1_3 = 0.0;
		ROC_ext = 0.0;
		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		memset(&(BetaX_for_Iz_ext),0,sizeof(qd_or_d));

		// always same coefficient:
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; 
		// still same regardless of level ...

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
				// External field affects only via coeff_self				
				
								 
				// BUT can't see where prediction uses neighbour effect of circuit_regressor on neighbour phi
			
			    // That may account for why we get different on that.
			};
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
					for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
					{
						BetaX[iEqn][iRegrVec][0] += coefflocal[iEqn][0]*data[0];
						BetaX[iEqn][iRegrVec][1] += coefflocal[iEqn][1]*data[1];
						BetaX[iEqn][iRegrVec][2] += coefflocal[iEqn][2]*data[2];
						BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];
					};
					BetaX_ext[iEqn] += coefflocal[iEqn][0]*pNeigh->circuit_regressor;
				};
			};


			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
				pVertex->predict_eps[iEqn] = pVertex->epsilon[iEqn] +
					(  BetaX[iEqn][0][0]*gamma[0] + BetaX[iEqn][1][0]*gamma[4]
				     + BetaX[iEqn][0][1]*gamma[1] + BetaX[iEqn][1][1]*gamma[5]
					 + BetaX[iEqn][0][2]*gamma[2] + BetaX[iEqn][1][2]*gamma[6]
					 + BetaX[iEqn][0][3]*gamma[3] + BetaX[iEqn][1][3]*gamma[7]
					  + BetaX_ext[iEqn]*gamma[8]).x[0];
		    // Note that RSS 0 is increasing. So we absolutely should investigate whether
			// Gauss errors are as predicted. But we have to allow that RSS 0 could increase.
			// ie we have to check the required equations
			// d RSS[all eqns included] / d gamma_i = 0

		    

			//pVertex->predict_eps[1] = 0.0;//pVertex->epsilon[1] +
				//(BetaX[1][0][3]*gamma[3] + BetaX[1][1][3]*gamma[7] + BetaX_ext[1]*gamma[8]).x[0];
			//pVertex->predict_eps[2] = 0.0;//pVertex->epsilon[2] +
				//(BetaX[2][0][3]*gamma[3] + BetaX[2][1][3]*gamma[7] + BetaX_ext[2]*gamma[8]).x[0];
			//pVertex->predict_eps[3] = 0.0;//pVertex->epsilon[3] +
				//(BetaX[3][0][3]*gamma[3] + BetaX[3][1][3]*gamma[7] + BetaX_ext[3]*gamma[8]).x[0];

			// 3,7,8 = Az, Ez_ext.


			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				BetaX_for_Iz[iRegrVec1][iGamma1] +=
						coeffself[IZ][iGamma1]*pVertex->regressor[iRegrVec1][iGamma1];
			};
			
		// Test:
		// Are we predicting that each row: sum [ALL eps . deps/dgamma_i] = 0 ?

			ROC_0_3 += pVertex->predict_eps[3]*BetaX[3][0][3];
			ROC_1_3 += pVertex->predict_eps[3]*BetaX[3][1][3];
			ROC_ext += pVertex->predict_eps[3]*BetaX_ext[3];

			// ROC_ext failed to be 0.
			// Other ways to compute ROC_ext, morphing back to our formulation above:
			// First, as above:

			ROC_ext_comp3 += (BetaX[3][0][3]*gamma[3]*BetaX_ext[3]).x[0];
			ROC_ext_comp7 += (BetaX[3][1][3]*gamma[7]*BetaX_ext[3]).x[0];
			// We should find that we can then remove gamma out and get back the initial summary matrix
			// If not, we hunt down which product-sum term is different.
			ROC_ext_comp8 += (BetaX_ext[3]*gamma[8]*BetaX_ext[3]).x[0];
			ROC_ext_eps += pVertex->epsilon[3]*BetaX_ext[3];
			
			ROC_ext_summ3 += BetaX[3][0][3]*BetaX_ext[3];
			ROC_ext_summ7 += BetaX[3][1][3]*BetaX_ext[3];
			ROC_ext_summ8 += BetaX_ext[3]*BetaX_ext[3];
			
			++pVertex;
		};

		
		predict_eps_Iz = Epsilon_Iz;
		if (iLevel >= 0) predict_eps_Iz = Epsilon_Iz_aux[iLevel];

		ROC_ext_eps += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		predict_eps_Iz += (BetaX_for_Iz[0][3]*gamma[3] + BetaX_for_Iz[1][3]*gamma[7] + BetaX_for_Iz_ext*gamma[8]).x[0];
		
		printf("predict_eps_Iz %1.9E\n",predict_eps_Iz);

		ROC_0_3 += (predict_eps_Iz*BetaX_for_Iz[0][3]).x[0];
		ROC_1_3 += (predict_eps_Iz*BetaX_for_Iz[1][3]).x[0];
		ROC_ext += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		ROC_ext_comp3 += (BetaX_for_Iz[0][3]*gamma[3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp7 += (BetaX_for_Iz[1][3]*gamma[7]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp8 += (BetaX_for_Iz_ext*gamma[8]*BetaX_for_Iz_ext).x[0];
		
		ROC_ext_summ3 += (BetaX_for_Iz[0][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ7 += (BetaX_for_Iz[1][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ8 += (BetaX_for_Iz_ext*BetaX_for_Iz_ext).x[0];
			
		// Generate ROCext predictions from comp and summ:
		
		ROC_ext_comp = ROC_ext_comp3+ROC_ext_comp7+ROC_ext_comp8 +ROC_ext_eps;

		printf("COMP prediction: %1.10E\n",ROC_ext_comp);

		// Compare summ with summary matrix original elements:

		ROC_ext_summ = (ROC_ext_summ3*gamma[3]+ROC_ext_summ7*gamma[7]+ROC_ext_summ8*gamma[8] +
						ROC_ext_eps).x[0];

		printf("SUMM prediction: %1.10E\n",ROC_ext_summ);

		printf("Summ %1.10E %1.10E %1.10E %1.10E \n",
			ROC_ext_summ3,ROC_ext_summ7,ROC_ext_summ8, ROC_ext_eps);
		printf("Orig %1.10E %1.10E %1.10E %1.10E \n",
			Storemat.LU[8][3].x[0],Storemat.LU[8][7].x[0],Storemat.LU[8][8].x[0],
			minus_eps_sums[8].x[0]);

		printf("\nROC_0_3 %1.9E ROC_1_3 %1.9E ROC_ext %1.9E \n\n",
			ROC_0_3,ROC_1_3,ROC_ext);

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#endif
		
		// Make additions to variables:

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			pVertex->phi += (gamma[0]*pVertex->regressor[0][PHI]
						   + gamma[4]*pVertex->regressor[1][PHI]
						   + gamma[8]*pVertex->circuit_regressor).x[0];
			pVertex->A.x += (gamma[1]*pVertex->regressor[0][AX]
						   + gamma[5]*pVertex->regressor[1][AX]).x[0];
			pVertex->A.y += (gamma[2]*pVertex->regressor[0][AY]
						   + gamma[6]*pVertex->regressor[1][AY]).x[0];
			pVertex->A.z += (gamma[3]*pVertex->regressor[0][AZ]
						   + gamma[7]*pVertex->regressor[1][AZ]).x[0];
			++pVertex;
		};
		if (iLevel == -1) {
			this->PhiAnode += (gamma[NUMREGRESStimesNUMCELLEQNS]);//.x[0];
		} else {
			PhiAnode_aux[iLevel] += (gamma[NUMREGRESStimesNUMCELLEQNS]);//.x[0];
		};

	}; // next iteration
	
	// DEBUG:
	if (0)//bGlobalSpitFlag) 
	{
		
		counter++;
		if (((counter-1) % 4 == 0) || (iLevel >= NUM_COARSE_LEVELS-2)) {

		sprintf(buffer,"JRLS_Gauss_%d_lvl%d.txt",counter,iLevel);
		FILE * fp1 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Axy_%d_lvl%d.txt",counter,iLevel);
		FILE * fp2 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Ax_%d_lvl%d.txt",counter,iLevel);
		FILE * fp3 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Ay_%d_lvl%d.txt",counter,iLevel);
		FILE * fp4 = fopen(buffer,"w");

		//
		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	// Store existing epsilon...
		//	pVertex->xdot.x = pVertex->epsilon[GAUSS];
		//	
		//	pVertex->v_n_0.x = pVertex->epsilon[AX];
		//	pVertex->v_n_0.y = pVertex->epsilon[AY];
		//	pVertex->v_n_0.z = pVertex->epsilon[AZ];

		//	++pVertex;
		//};

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	//pVertex->phi += 9.0*(gamma[0]*pVertex->regressor[0][PHI]).x[0];
		//	pVertex->A.x += 9.0*(gamma[1]*pVertex->regressor[0][AX]).x[0];
		//	pVertex->A.y += 9.0*(gamma[2]*pVertex->regressor[0][AY]).x[0];
		//	//pVertex->A.z += 9.0*(gamma[3]*pVertex->regressor[0][AZ]).x[0];

		//	++pVertex;
		//};

		//CalculateEpsilons();

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	// Store epsilon for 10x change...
		//	pVertex->xdotdot.x = pVertex->epsilon[AX];
		//	pVertex->xdotdot.y = pVertex->epsilon[AY];
		//	++pVertex;
		//};

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	//pVertex->phi -= 9.0*(gamma[0]*pVertex->regressor[0][PHI]).x[0];
		//	pVertex->A.x -= 9.0*(gamma[1]*pVertex->regressor[0][AX]).x[0];
		//	pVertex->A.y -= 9.0*(gamma[2]*pVertex->regressor[0][AY]).x[0];
		//	//pVertex->A.z -= 9.0*(gamma[3]*pVertex->regressor[0][AZ]).x[0];
		//	++pVertex;
		//};

	if (iLevel == -1) {
		CalculateEpsilons();
	} else {
		CalculateEpsilonAux4(iLevel);
	};
	

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// Info to spit out: coeff_self, coeffs on neighs for Gauss
			// phi regressors and coefficients
			// Old & new epsilons
			
			fprintf(fp1,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp1,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp1,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp1,"| coeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[GAUSS][PHI],
				pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[GAUSS][AY],pVertex->coeff_self[GAUSS][AZ],
				pVertex->coeff_self[GAUSS][UNITY],pVertex->coeff_self[GAUSS][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp1,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[GAUSS][PHI],(X+izNeigh[i])->phi);
			fprintf(fp1,"\n");
			
			fprintf(fp2,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp2,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp2,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp2,"| Axcoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AX][PHI],
				pVertex->coeff_self[AX][AX],pVertex->coeff_self[AX][AY],pVertex->coeff_self[AX][AZ],
				pVertex->coeff_self[AX][UNITY],pVertex->coeff_self[AX][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp2,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AX][AX],(X+izNeigh[i])->A.x);
			fprintf(fp2,"\n");
			
			fprintf(fp3,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp3,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp3,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp3,"| Axcoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AX][PHI],
				pVertex->coeff_self[AX][AX],pVertex->coeff_self[AX][AY],pVertex->coeff_self[AX][AZ],
				pVertex->coeff_self[AX][UNITY],pVertex->coeff_self[AX][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp3,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AX][AX],(X+izNeigh[i])->A.x);
			fprintf(fp3,"\n");
			
			fprintf(fp2,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp2,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp2,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp2,"| Aycoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AY][PHI],
				pVertex->coeff_self[AY][AX],pVertex->coeff_self[AY][AY],pVertex->coeff_self[AY][AZ],
				pVertex->coeff_self[AY][UNITY],pVertex->coeff_self[AY][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp2,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AY][AY],(X+izNeigh[i])->A.y);
			fprintf(fp2,"\n");
			
			fprintf(fp4,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp4,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp4,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp4,"| Aycoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AY][PHI],
				pVertex->coeff_self[AY][AX],pVertex->coeff_self[AY][AY],pVertex->coeff_self[AY][AZ],
				pVertex->coeff_self[AY][UNITY],pVertex->coeff_self[AY][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp4,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AY][AY],(X+izNeigh[i])->A.y);
			fprintf(fp4,"\n");
			
			++pVertex;
		};
		
		// What happens if coefficient gamma[0] is multiplied by ten? (Put it back after?)
		fclose(fp1);
		fclose(fp2);
		fclose(fp3);
		fclose(fp4);
		};
	}

	// DEBUG:
	if (iLevel == -1) {
		CalculateEpsilons();
	} else {
		CalculateEpsilonAux4(iLevel);
	};
	
	// Survey RSS from each equation:
	RSStot = 0.0;
	for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
	{
		RSS[iEqn] = 0.0;
		pVertex = Xarray;
		for (iVertex =0; iVertex < numUse; iVertex++)
		{
			RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
			++pVertex;
		};
		RSStot += RSS[iEqn];
	};
	if (iLevel == -1) {
		RSStot += Epsilon_Iz*Epsilon_Iz;
	} else {
		RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	};

	if ((RSStot > old_RSS*1.0000000000001))
	{
		printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
			old_RSS,RSStot);
		getch();
	}

	fprintf(fp,"RSS[]: %1.9E %1.9E %1.9E %1.9E EpsIz: %1.9E \n\n",
			RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
	fclose(fp);
	

}

void TriMesh::IterationsJRLS_individual_equations(int iLevel, int iterations)
{
	real ROC_0_3,ROC_1_3,ROC_ext;
	real * pRelevant;
	real ratio;
	int iEdge;
	static int const NUMREGRESStimesNUMCELLEQNS = NUMREGRESS*NUMCELLEQNS;
	Triangle * pTri,*pTri2;
	long iTri;
	long iIteration, iVariable, iEqn, iRegrVec, iGamma, iGamma1, iGamma2, iRegrVec1, iRegrVec2;
	long i,j, iVertex,len, i1, i2;
	real sum_Az,sum_phi,avg_Az,avg_phi;
	real old_RSS;
	Vertex * pVertex, *pNeigh, *Xarray;
	real residual, coeffsum, SumEps;
	int iCorner;
	real TempMES[3], tempgamma[3];
	real GlobalStoreBeta, GlobalEpsilonExist,maxeps, dbyd1;
	long iMax,numUse;	
	real overL2JA,overL2Jv,overL2RA,overL2Rv;

	real static const overch = 1.0/(c*h);	
	real const Change_chEz = 1.0;

	real SSreg[NUMREGRESS][NUMCELLEQNS];
	real L2reg[NUMREGRESS][NUMCELLEQNS];

	real RSStot;
	real predict_eps_Iz =0.0;

	Matrix Summary, Storemat, Tempsumm;	

	qd_or_d debugLU[GAMMANUM][GAMMANUM];
	qd_or_d minus_eps_sums[GAMMANUM], gamma[GAMMANUM];
	
	real BetaX[NUM_EQNS_1][NUMREGRESS][NUM_AFFECTORS_1];
	real BetaX_ext[NUM_EQNS_1]; // The 9th regressor
	// The extra equation:
	qd_or_d BetaX_for_Iz[NUMREGRESS][NUM_AFFECTORS_1];
	qd_or_d BetaX_for_Iz_ext;

	real data[NUM_AFFECTORS_1];
	real coefflocal[NUM_EQNS_1][NUM_AFFECTORS_1];
	real coeffself[NUM_EQNS_2][NUM_AFFECTORS_2]; // 2 > 1

	char buffer[256];
	static int counter = 0;

	real beta,sum_beta_sq,sum_eps_beta;
	long iNeigh, neigh_len;
	long izNeigh[128];
	long iVar, iWhich;


	Summary.Invoke(GAMMANUM);
	Tempsumm.Invoke(3);
	Storemat.Invoke(GAMMANUM); // debugging

	HANDLE hConsole;
    int k;
	real RSS[4],RSSpred[4];
	FILE * fp;
	
	// DEBUG:
	
		if (iLevel == 3) {
			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			fprintf(fp3,"JRLS called.\n");
			fclose(fp3);
		};

	fp = fopen("RSSnew__.txt","a");
	
	fprintf(fp,"--------------------\nRun JRLS Level %d \n",iLevel);


//	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel];
		numUse = numAuxVertices[iLevel];
	};
	
	// The point of putting this loop inside is to retain any calcs and reduce overheads.
	for (iIteration = 0; iIteration < iterations; iIteration++)
	{
		if (iLevel == -1) {
			CalculateEpsilons();
		} else {
			CalculateEpsilonAux4(iLevel);
		};

#ifdef DEBUGJRLS

		// DEBUG:
		// Survey RSS from each equation:
		RSStot = 0.0;

		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			RSS[iEqn] = 0.0;
			RSSpred[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				RSSpred[iEqn] += pVertex->predict_eps[iEqn]*pVertex->predict_eps[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};

		printf("RSS[]: %1.9E %1.9E %1.9E %1.9E\npred : %1.9E %1.9E %1.9E %1.9E \nepsIz %1.9E : %1.9E\n",
			RSS[0],RSS[1],RSS[2],RSS[3],		
			RSSpred[0],RSSpred[1],RSSpred[2],RSSpred[3],
			((iLevel >= 0)?Epsilon_Iz_aux[iLevel]:Epsilon_Iz),predict_eps_Iz);
		printf("RSStot %1.9E\n",RSStot);
#else
		RSStot = 0.0;
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			RSS[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};
#endif
		if ((iIteration > 0) && (RSStot > old_RSS*1.0000000000001))
		{
			printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
				old_RSS,RSStot);
			getch();
		}

		fprintf(fp,"RSS[]: %1.9E %1.9E %1.9E %1.9E EpsIz: %1.9E \n",
						RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		
		printf("RSS[%d]: %1.5E %1.5E %1.5E %1.5E EpsIz: %1.5E\n",iLevel,
				RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		
		old_RSS = RSStot;

		//fprintf(fp,"iLevel %d  RSS_Ampz %1.14E  Eps_Iz %1.14E  RSSTot %1.14E \n",
		//	iLevel, RSS[3], (iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel], RSStot);

		// 1. Define regressor vectors
		
		ZeroMemory(SSreg, sizeof(real)*NUMCELLEQNS*NUMREGRESS);
		real SSextra = 0.0;
		
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// Jacobi:
			// =======
			
			if ((pVertex->coeff_self[GAUSS][PHI_ANODE] == 0.0) || (pVertex->coeff_self[GAUSS][PHI] == 0.0))			
			{
				pVertex->circuit_regressor = 0.0;
			} else {
				// create "circuit field" regressor by changing phi to cancel out (for epsilon Gauss) a change of 1 in phi_anode.
				pVertex->circuit_regressor = -pVertex->coeff_self[GAUSS][PHI_ANODE]/pVertex->coeff_self[GAUSS][PHI];
			};

			if (pVertex->coeff_self[GAUSS][PHI] == 0.0) {
				pVertex->regressor[JACOBI_REGR][PHI] = 0.0;
			} else {
				pVertex->regressor[JACOBI_REGR][PHI] = pVertex->epsilon[GAUSS] / pVertex->coeff_self[GAUSS][PHI]; 
			};
			if (pVertex->coeff_self[AX][AX] == 0.0) {
				pVertex->regressor[JACOBI_REGR][AX] = 0.0;
			} else {
				pVertex->regressor[JACOBI_REGR][AX] = pVertex->epsilon[AX] / pVertex->coeff_self[AX][AX]; 
			};
			if (pVertex->coeff_self[AY][AY] == 0.0) {
				pVertex->regressor[JACOBI_REGR][AY] = 0.0;
			} else {
				pVertex->regressor[JACOBI_REGR][AY] = pVertex->epsilon[AY] / pVertex->coeff_self[AY][AY]; 
			};
			if (pVertex->coeff_self[AZ][AZ] == 0.0)
			{
				pVertex->regressor[JACOBI_REGR][AZ] = 0.0;
			} else {
				pVertex->regressor[JACOBI_REGR][AZ] = pVertex->epsilon[AZ] / pVertex->coeff_self[AZ][AZ]; 
			};

			memcpy(pVertex->regressor[RICHARDSON_REGR],pVertex->epsilon,sizeof(real)*NUM_AFFECTORS_1);

			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
				{
					SSreg[iRegrVec][iVar] += pVertex->regressor[iRegrVec][iVar] * 
											 pVertex->regressor[iRegrVec][iVar];
					if (!(SSreg[iRegrVec][iVar] == SSreg[iRegrVec][iVar])) {
						iVar = iVar;
					}
				}

				// DEBUG:
		//	if (isnan(pVertex->regressor[0][0])) {
		//		printf("S regr %1.14E iVertex %d eps %1.14E coeffself[0][0] %1.14E \n",pVertex->regressor[0][0],iVertex,pVertex->epsilon[GAUSS],
		//			pVertex->coeff_self[GAUSS][PHI]);

		//		getch();
		//	}

			++pVertex;
		};

		// Normalize regressors:
		for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
			for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
				L2reg[iRegrVec][iVar] = sqrt(SSreg[iRegrVec][iVar]/(real)numUse);
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
			{
				 // do not divide by L2(var itself) which may be == 0
				{
					for (iVar = 0; iVar < NUM_AFFECTORS_1; iVar++)
						if (L2reg[iRegrVec][iVar] != 0.0)
							pVertex->regressor[iRegrVec][iVar] /= L2reg[iRegrVec][iVar];
				};
			};
			++pVertex;
		};

		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


		for (i = 0; i < GAMMANUM; i++)
			memset(Summary.LU[i],0,sizeof(qd_or_d)*GAMMANUM);
		memset(minus_eps_sums,0,sizeof(qd_or_d)*GAMMANUM);

		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; 
		
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// The idea is to collect [Beta_ki dot x_i] [Beta_kj dot x_j] ,
			// so first collect [Beta_ki dot x_i],
			// for each eps k [k over all eqns at this vertex], each i [i over all regressors]

			// Each epsilon^2 gets an equal shout in sums and dot products here.
			
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
			};
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
					for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
					{
						BetaX[iEqn][iRegrVec][0] += coefflocal[iEqn][0]*data[0];
						BetaX[iEqn][iRegrVec][1] += coefflocal[iEqn][1]*data[1];
						BetaX[iEqn][iRegrVec][2] += coefflocal[iEqn][2]*data[2];
						BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];

						// NEW ADDITION 09/10/16
						BetaX_ext[iEqn] += coefflocal[iEqn][0]*pNeigh->circuit_regressor;
					};
				};
			};
			
			for (iEqn = 0; iEqn < NUMCELLEQNS; iEqn++)
			{
				for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
				for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
				{
					i1 = iRegrVec1*NUMCELLEQNS+iGamma1;
					for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
					for (iGamma2 = 0; iGamma2 < NUMCELLEQNS; iGamma2++)
					{
						i2 = iRegrVec2*NUMCELLEQNS+iGamma2;
						Summary.LU[i1][i2] +=
							BetaX[iEqn][iRegrVec1][iGamma1]*BetaX[iEqn][iRegrVec2][iGamma2];
					};
				
					Summary.LU[i1][NUMREGRESStimesNUMCELLEQNS] += BetaX[iEqn][iRegrVec1][iGamma1]*BetaX_ext[iEqn];
					Summary.LU[NUMREGRESStimesNUMCELLEQNS][i1] += BetaX[iEqn][iRegrVec1][iGamma1]*BetaX_ext[iEqn];
					minus_eps_sums[i1] -= BetaX[iEqn][iRegrVec1][iGamma1]*pVertex->epsilon[iEqn];
				};
				Summary.LU[NUMREGRESStimesNUMCELLEQNS][NUMREGRESStimesNUMCELLEQNS] += 
																BetaX_ext[iEqn]*BetaX_ext[iEqn];
				minus_eps_sums[NUMREGRESStimesNUMCELLEQNS] -= BetaX_ext[iEqn]*pVertex->epsilon[iEqn];
				// Every epsilon should appear in every epsilon sum;
				// In sum j, each epsilon appears against the change in it due to regressor j.

			}; // next iEqn = next epsilon from this location
						
			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				BetaX_for_Iz[iRegrVec1][iGamma1] +=
						coeffself[IZ][iGamma1]*pVertex->regressor[iRegrVec1][iGamma1];
			};
			BetaX_for_Iz_ext += coeffself[IZ][PHI] * pVertex->circuit_regressor;

			++pVertex;
		};
		
		
		// still same regardless of level ...
		
		// Add to summary:
		for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				i1 = iRegrVec1*NUMCELLEQNS+iGamma1;
				for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
					for (iGamma2 = 0; iGamma2 < NUMCELLEQNS; iGamma2++)
						Summary.LU[i1][iRegrVec2*NUMCELLEQNS+iGamma2] +=
							BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz[iRegrVec2][iGamma2];

				Summary.LU[i1][NUMREGRESStimesNUMCELLEQNS] += BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz_ext;
				Summary.LU[NUMREGRESStimesNUMCELLEQNS][i1] += BetaX_for_Iz[iRegrVec1][iGamma1]*BetaX_for_Iz_ext;
				minus_eps_sums[i1] -= BetaX_for_Iz[iRegrVec1][iGamma1]*
														((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
			};

		Summary.LU[NUMREGRESStimesNUMCELLEQNS][NUMREGRESStimesNUMCELLEQNS] += BetaX_for_Iz_ext*BetaX_for_Iz_ext;
		minus_eps_sums[NUMREGRESStimesNUMCELLEQNS] -= BetaX_for_Iz_ext*((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);

		bool bNonzero;
		for (i = 0; i < GAMMANUM; i++)
		{
			bNonzero = 0;
			for (j = 0; j < GAMMANUM; j++){
				if (Summary.LU[i][j] != 0.0)
					bNonzero = true;
				debugLU[i][j] = Summary.LU[i][j];
			};
			if (bNonzero == 0) {
				Summary.LU[i][i] = 1.0;
		//		printf("{ %d }",i);
			};
		};
		//printf("\n");
#ifdef DEBUGJRLS
		for (i = 0; i < GAMMANUM; i++)
		{
			for (j = 0; j < GAMMANUM; j++)
				printf("%1.2E ",Summary.LU[i][j].x[0]);
			printf("\n");
		}
		printf("\n");

		// debug:
		Storemat.CopyFrom(Summary);
#endif
		
		if (Summary.LUdecomp() == 1) {
			printf("LUdecomp failed.\n");
			getch();
		}
		Summary.LUSolve(minus_eps_sums, gamma); // solve matrix equation

#ifdef DEBUGJRLS
		printf("gam %1.3E %1.3E %1.3E %1.3E \n", gamma[0].x[0], gamma[1].x[0], gamma[2].x[0], gamma[3].x[0]);
		printf("--- %1.3E %1.3E %1.3E %1.3E %1.3E \n", gamma[4].x[0], gamma[5].x[0], gamma[6].x[0], gamma[7].x[0],gamma[8].x[0]);
		
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		// DEBUG:
		// Make predictions:
		real ROC_ext_comp3 = 0.0, ROC_ext_comp7 = 0.0, ROC_ext_comp8 = 0.0,
			ROC_ext_comp, ROC_ext_summ3 = 0.0, ROC_ext_summ7 = 0.0, ROC_ext_summ8 = 0.0,
			ROC_ext_summ, ROC_ext_eps = 0.0;
		ROC_0_3 = 0.0;
		ROC_1_3 = 0.0;
		ROC_ext = 0.0;
		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		memset(&(BetaX_for_Iz_ext),0,sizeof(qd_or_d));

		// always same coefficient:
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; 
		// still same regardless of level ...

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
				// External field affects only via coeff_self				
				// pVertex->coeff_self[GAUSS][TUNEFAC] += FOUR_PI_Q*h*wt_domain[2]* pNeigh->temp2.y*
				//				 (n_e*pNeigh->sigma_e.xz-0.5*n_i*pNeigh->sigma_i.xz)*edgenorm_use[2].x
				// ...
				// We propose to change Ez by gamma_ext*temp2.y at each location.
				// This means the effect of gamma_ext on epsilon[iEqn] is coeffself[iEqn][TUNEFAC].

			};
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
					for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
					{
						BetaX[iEqn][iRegrVec][0] += coefflocal[iEqn][0]*data[0];
						BetaX[iEqn][iRegrVec][1] += coefflocal[iEqn][1]*data[1];
						BetaX[iEqn][iRegrVec][2] += coefflocal[iEqn][2]*data[2];
						BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];
					};
				};
			};

			pVertex->predict_eps[0] = pVertex->epsilon[0] +
				(BetaX[0][0][3]*gamma[3] + BetaX[0][1][3]*gamma[7] + BetaX_ext[0]*gamma[8]).x[0];
			pVertex->predict_eps[1] = pVertex->epsilon[1] +
				(BetaX[1][0][3]*gamma[3] + BetaX[1][1][3]*gamma[7] + BetaX_ext[1]*gamma[8]).x[0];
			pVertex->predict_eps[2] = pVertex->epsilon[2] +
				(BetaX[2][0][3]*gamma[3] + BetaX[2][1][3]*gamma[7] + BetaX_ext[2]*gamma[8]).x[0];
			pVertex->predict_eps[3] = pVertex->epsilon[3] +
				(BetaX[3][0][3]*gamma[3] + BetaX[3][1][3]*gamma[7] + BetaX_ext[3]*gamma[8]).x[0];

			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				BetaX_for_Iz[iRegrVec1][iGamma1] +=
						coeffself[IZ][iGamma1]*pVertex->regressor[iRegrVec1][iGamma1];
			};
			
		// Test:
		// Are we predicting that each row: sum [ALL eps . deps/dgamma_i] = 0 ?

			ROC_0_3 += pVertex->predict_eps[3]*BetaX[3][0][3];
			ROC_1_3 += pVertex->predict_eps[3]*BetaX[3][1][3];
			ROC_ext += pVertex->predict_eps[3]*BetaX_ext[3];

			// ROC_ext failed to be 0.
			// Other ways to compute ROC_ext, morphing back to our formulation above:
			// First, as above:

			ROC_ext_comp3 += (BetaX[3][0][3]*gamma[3]*BetaX_ext[3]).x[0];
			ROC_ext_comp7 += (BetaX[3][1][3]*gamma[7]*BetaX_ext[3]).x[0];
			// We should find that we can then remove gamma out and get back the initial summary matrix
			// If not, we hunt down which product-sum term is different.
			ROC_ext_comp8 += (BetaX_ext[3]*gamma[8]*BetaX_ext[3]).x[0];
			ROC_ext_eps += pVertex->epsilon[3]*BetaX_ext[3];
			
			ROC_ext_summ3 += BetaX[3][0][3]*BetaX_ext[3];
			ROC_ext_summ7 += BetaX[3][1][3]*BetaX_ext[3];
			ROC_ext_summ8 += BetaX_ext[3]*BetaX_ext[3];
			
			++pVertex;
		};

		
		predict_eps_Iz = Epsilon_Iz;
		if (iLevel >= 0) predict_eps_Iz = Epsilon_Iz_aux[iLevel];

		ROC_ext_eps += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		predict_eps_Iz += (BetaX_for_Iz[0][3]*gamma[3] + BetaX_for_Iz[1][3]*gamma[7] + BetaX_for_Iz_ext*gamma[8]).x[0];
		
		printf("predict_eps_Iz %1.9E\n",predict_eps_Iz);

		ROC_0_3 += (predict_eps_Iz*BetaX_for_Iz[0][3]).x[0];
		ROC_1_3 += (predict_eps_Iz*BetaX_for_Iz[1][3]).x[0];
		ROC_ext += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		ROC_ext_comp3 += (BetaX_for_Iz[0][3]*gamma[3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp7 += (BetaX_for_Iz[1][3]*gamma[7]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp8 += (BetaX_for_Iz_ext*gamma[8]*BetaX_for_Iz_ext).x[0];
		
		ROC_ext_summ3 += (BetaX_for_Iz[0][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ7 += (BetaX_for_Iz[1][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ8 += (BetaX_for_Iz_ext*BetaX_for_Iz_ext).x[0];
			
		// Generate ROCext predictions from comp and summ:
		
		ROC_ext_comp = ROC_ext_comp3+ROC_ext_comp7+ROC_ext_comp8 +ROC_ext_eps;

		printf("COMP prediction: %1.10E\n",ROC_ext_comp);

		// Compare summ with summary matrix original elements:

		ROC_ext_summ = (ROC_ext_summ3*gamma[3]+ROC_ext_summ7*gamma[7]+ROC_ext_summ8*gamma[8] +
						ROC_ext_eps).x[0];

		printf("SUMM prediction: %1.10E\n",ROC_ext_summ);

		printf("Summ %1.10E %1.10E %1.10E %1.10E \n",
			ROC_ext_summ3,ROC_ext_summ7,ROC_ext_summ8, ROC_ext_eps);
		printf("Orig %1.10E %1.10E %1.10E %1.10E \n",
			Storemat.LU[8][3].x[0],Storemat.LU[8][7].x[0],Storemat.LU[8][8].x[0],
			minus_eps_sums[8].x[0]);

		printf("\nROC_0_3 %1.9E ROC_1_3 %1.9E ROC_ext %1.9E \n\n",
			ROC_0_3,ROC_1_3,ROC_ext);

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#endif
		
		// Make additions to variables:
		// ____________________________

// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"Before add: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);
		};
*/

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			pVertex->phi += (gamma[0]*pVertex->regressor[0][PHI]
						   + gamma[4]*pVertex->regressor[1][PHI]
						   + gamma[8]*pVertex->circuit_regressor).x[0];
			pVertex->A.x += (gamma[1]*pVertex->regressor[0][AX]
						   + gamma[5]*pVertex->regressor[1][AX]).x[0];
			pVertex->A.y += (gamma[2]*pVertex->regressor[0][AY]
						   + gamma[6]*pVertex->regressor[1][AY]).x[0];
			pVertex->A.z += (gamma[3]*pVertex->regressor[0][AZ]
						   + gamma[7]*pVertex->regressor[1][AZ]).x[0];
			++pVertex;
		};
		if (iLevel == -1) {
			this->PhiAnode += (gamma[NUMREGRESStimesNUMCELLEQNS]);//.x[0];
		} else {
			PhiAnode_aux[iLevel] += (gamma[NUMREGRESStimesNUMCELLEQNS]);//.x[0];
		};

// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"Before resetphi: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);

		};*/
// ////////////////////////////////////////////////
		// Reset to get avg Az = 0 and avg phi = 0 at the back.
/*
		real sum_phi,avg_phi;

		sum_phi = 0.0;
		pVertex = Xarray;
		if (iLevel == -1) {
			for (iVertex = 0; iVertex < numInnermostRow; iVertex++)
			{
				sum_phi += pVertex->phi;
				++pVertex;
			};			
			avg_phi = sum_phi/(real)numInnermostRow;
		} else {
			for (iVertex = 0; iVertex < numInnermostRowAux[iLevel]; iVertex++)
			{
				sum_phi += pVertex->phi;
				++pVertex;
			};			
			avg_phi = sum_phi/(real)numInnermostRowAux[iLevel];
		};

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			pVertex->phi -= avg_phi;
			++pVertex;
		};
*/
		// Can no longer adjust level of phi, nor should we need to.

		// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"After resetphi: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n\n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);

		};*/

		// Is this affecting the Gauss errors? We should verify since it just decided to break.

		// Maybe it affects total Axy errors -- that is actually possible.
		// Let's look at how sum of epsilon Ampere depends on changing the level of phi.
		// It's clear it SHOULD NOT affect anything -- and changing all values at Lvl 0 implies
		// a change of all values at Lvl 1. 
		// 



	//	sum_Az = 0.0;
	//	pTri = T + StartAvgRowTri; 
		// how long will last row of tris remain inviolate? doesn't matter a whole lot though if it's a little bit wrong.
	//	for (iTri = StartAvgRowTri; iTri < numTriangles; iTri++)
	//	{
	//		sum_Az += pTri->A.z;
	//		++pTri;
	//	};
	//	avg_Az = sum_Az/(real)(numTriangles-StartAvgRowTri);
		
		// Note that nothing was being done for pTri->phi anyway.

		// When we move Az we also move chEzExt:
//		pTri = T;
//		for (iTri = 0; iTri < numTriangles; iTri++)
//		{
//			pTri->A.z -= avg_Az;
//			++pTri;
//		};
//		chEzExt -= avg_Az; // when Az goes up, chEz should go up, as this keeps E the same.

	}; // next iteration
	
	// DEBUG:
	if (0)//bGlobalSpitFlag) 
	{
		
		counter++;
		if (((counter-1) % 4 == 0) || (iLevel >= NUM_COARSE_LEVELS-2)) {

		sprintf(buffer,"JRLS_Gauss_%d_lvl%d.txt",counter,iLevel);
		FILE * fp1 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Axy_%d_lvl%d.txt",counter,iLevel);
		FILE * fp2 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Ax_%d_lvl%d.txt",counter,iLevel);
		FILE * fp3 = fopen(buffer,"w");
		sprintf(buffer,"JRLS_Ay_%d_lvl%d.txt",counter,iLevel);
		FILE * fp4 = fopen(buffer,"w");

		//
		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	// Store existing epsilon...
		//	pVertex->xdot.x = pVertex->epsilon[GAUSS];
		//	
		//	pVertex->v_n_0.x = pVertex->epsilon[AX];
		//	pVertex->v_n_0.y = pVertex->epsilon[AY];
		//	pVertex->v_n_0.z = pVertex->epsilon[AZ];

		//	++pVertex;
		//};

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	//pVertex->phi += 9.0*(gamma[0]*pVertex->regressor[0][PHI]).x[0];
		//	pVertex->A.x += 9.0*(gamma[1]*pVertex->regressor[0][AX]).x[0];
		//	pVertex->A.y += 9.0*(gamma[2]*pVertex->regressor[0][AY]).x[0];
		//	//pVertex->A.z += 9.0*(gamma[3]*pVertex->regressor[0][AZ]).x[0];

		//	++pVertex;
		//};

		//CalculateEpsilons();

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	// Store epsilon for 10x change...
		//	pVertex->xdotdot.x = pVertex->epsilon[AX];
		//	pVertex->xdotdot.y = pVertex->epsilon[AY];
		//	++pVertex;
		//};

		//pVertex = Xarray;
		//for (iVertex = 0; iVertex < numUse; iVertex++)
		//{
		//	//pVertex->phi -= 9.0*(gamma[0]*pVertex->regressor[0][PHI]).x[0];
		//	pVertex->A.x -= 9.0*(gamma[1]*pVertex->regressor[0][AX]).x[0];
		//	pVertex->A.y -= 9.0*(gamma[2]*pVertex->regressor[0][AY]).x[0];
		//	//pVertex->A.z -= 9.0*(gamma[3]*pVertex->regressor[0][AZ]).x[0];
		//	++pVertex;
		//};

	if (iLevel == -1) {
		CalculateEpsilons();
	} else {
		CalculateEpsilonAux4(iLevel);
	};
	

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// Info to spit out: coeff_self, coeffs on neighs for Gauss
			// phi regressors and coefficients
			// Old & new epsilons
			
			fprintf(fp1,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp1,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp1,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp1,"| coeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[GAUSS][PHI],
				pVertex->coeff_self[GAUSS][AX],pVertex->coeff_self[GAUSS][AY],pVertex->coeff_self[GAUSS][AZ],
				pVertex->coeff_self[GAUSS][UNITY],pVertex->coeff_self[GAUSS][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp1,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[GAUSS][PHI],(X+izNeigh[i])->phi);
			fprintf(fp1,"\n");
			
			fprintf(fp2,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp2,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp2,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp2,"| Axcoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AX][PHI],
				pVertex->coeff_self[AX][AX],pVertex->coeff_self[AX][AY],pVertex->coeff_self[AX][AZ],
				pVertex->coeff_self[AX][UNITY],pVertex->coeff_self[AX][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp2,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AX][AX],(X+izNeigh[i])->A.x);
			fprintf(fp2,"\n");
			
			fprintf(fp3,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp3,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp3,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp3,"| Axcoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AX][PHI],
				pVertex->coeff_self[AX][AX],pVertex->coeff_self[AX][AY],pVertex->coeff_self[AX][AZ],
				pVertex->coeff_self[AX][UNITY],pVertex->coeff_self[AX][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp3,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AX][AX],(X+izNeigh[i])->A.x);
			fprintf(fp3,"\n");
			
			fprintf(fp2,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp2,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp2,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp2,"| Aycoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AY][PHI],
				pVertex->coeff_self[AY][AX],pVertex->coeff_self[AY][AY],pVertex->coeff_self[AY][AZ],
				pVertex->coeff_self[AY][UNITY],pVertex->coeff_self[AY][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp2,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AY][AY],(X+izNeigh[i])->A.y);
			fprintf(fp2,"\n");
			
			fprintf(fp4,"%d %d %1.14E %1.14E ",iVertex,pVertex->flags,pVertex->pos.x,pVertex->pos.y);
			fprintf(fp4,"| phi %1.14E A %1.14E %1.14E %1.14E tunefac %1.14E ",pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,PhiAnode.x[0]);
			fprintf(fp4,"| eps %1.14E %1.14E %1.14E %1.14E ",
				pVertex->epsilon[GAUSS],pVertex->epsilon[AX],pVertex->epsilon[AMPY],pVertex->epsilon[AMPZ]);
			fprintf(fp4,"| Aycoeffself %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E | ",
				pVertex->coeff_self[AY][PHI],
				pVertex->coeff_self[AY][AX],pVertex->coeff_self[AY][AY],pVertex->coeff_self[AY][AZ],
				pVertex->coeff_self[AY][UNITY],pVertex->coeff_self[AY][PHI_ANODE]);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (i =0; i < neigh_len; i++)
				fprintf(fp4,"%d %1.14E %1.14E ",izNeigh[i],pVertex->coeff[i].co[AY][AY],(X+izNeigh[i])->A.y);
			fprintf(fp4,"\n");
			
			++pVertex;
		};
		
		// What happens if coefficient gamma[0] is multiplied by ten? (Put it back after?)
		fclose(fp1);
		fclose(fp2);
		fclose(fp3);
		fclose(fp4);
		};
	}

	// DEBUG:
	if (iLevel == -1) {
		CalculateEpsilons();
	} else {
		CalculateEpsilonAux4(iLevel);
	};
	
	// Survey RSS from each equation:
	RSStot = 0.0;
	for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
	{
		RSS[iEqn] = 0.0;
		pVertex = Xarray;
		for (iVertex =0; iVertex < numUse; iVertex++)
		{
			RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
			++pVertex;
		};
		RSStot += RSS[iEqn];
	};
	if (iLevel == -1) {
		RSStot += Epsilon_Iz*Epsilon_Iz;
	} else {
		RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	};

	if ((RSStot > old_RSS*1.0000000000001))
	{
		printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
			old_RSS,RSStot);
		getch();
	}

	fprintf(fp,"RSS[]: %1.9E %1.9E %1.9E %1.9E EpsIz: %1.9E \n\n",
			RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
	fclose(fp);
	

}


void TriMesh::IterationsJRLS_Az(int iLevel, int iterations)
{
	real ROC_0_3,ROC_1_3,ROC_ext;
	real * pRelevant;
	real ratio;
	int iEdge;
	static int const NUMREGRESStimesNUMCELLEQNS = NUMREGRESS*NUMCELLEQNS;
	Triangle * pTri,*pTri2;
	long iTri;
	long iIteration, iVariable, iEqn, iRegrVec, iGamma, iGamma1, iGamma2, iRegrVec1, iRegrVec2;
	long i,j, iVertex,len, i1, i2;
	real sum_Az,sum_phi,avg_Az,avg_phi;
	real old_RSS = 1.0e10;
	Vertex * pVertex, *pNeigh, *Xarray;
	real residual, coeffsum, SumEps;
	int iCorner;
	real TempMES[3], tempgamma[3];
	real GlobalStoreBeta, GlobalEpsilonExist,maxeps, dbyd1;
	long iMax,numUse;	
	real overL2JA,overL2Jv,overL2RA,overL2Rv;

	real static const overch = 1.0/(c*h);	
	real const Change_chEz = 1.0;

	real SSreg[NUMREGRESS][NUMCELLEQNS];
	real L2reg[NUMREGRESS][NUMCELLEQNS];

	real RSStot;
	real predict_eps_Iz =0.0;

	Matrix Summary, Storemat, Tempsumm;	

	qd_or_d debugLU[GAMMANUM][GAMMANUM];
	qd_or_d minus_eps_sums[GAMMANUM], gamma[GAMMANUM];
	
	real BetaX[NUM_EQNS_1][NUMREGRESS][NUM_AFFECTORS_1];
	real BetaX_ext[NUM_EQNS_1]; // The 9th regressor
	// The extra equation:
	qd_or_d BetaX_for_Iz[NUMREGRESS][NUM_AFFECTORS_1];
	qd_or_d BetaX_for_Iz_ext;

	real data[NUM_AFFECTORS_1];
	real coefflocal[NUM_EQNS_1][NUM_AFFECTORS_1];
	real coeffself[NUM_EQNS_2][NUM_AFFECTORS_2]; // 2 > 1

	char buffer[256];
	static int counter = 0;

	real beta,sum_beta_sq,sum_eps_beta;
	long iNeigh, neigh_len;
	long izNeigh[128];
	long iVar, iWhich;

	Summary.Invoke(GAMMANUM);
	Tempsumm.Invoke(3);
	Storemat.Invoke(GAMMANUM); // debugging

	HANDLE hConsole;
    int k;
	real RSS[4],RSSpred[4];
	

//	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel];
		numUse = numAuxVertices[iLevel];
	};
	
	// The point of putting this loop inside is to retain any calcs and reduce overheads.
	for (iIteration = 0; iIteration < iterations; iIteration++)
	{
		if (iLevel == -1) {
			CalculateEpsilonsAz();
		} else {
			CalculateEpsilonAuxAz(iLevel);
		};

#ifdef DEBUGJRLS

		// DEBUG:
		// Survey RSS from each equation:
		RSStot = 0.0;

		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			RSS[iEqn] = 0.0;
			RSSpred[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				RSSpred[iEqn] += pVertex->predict_eps[iEqn]*pVertex->predict_eps[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};

		printf("RSS[]: %1.9E %1.9E %1.9E %1.9E\npred : %1.9E %1.9E %1.9E %1.9E \nepsIz %1.9E : %1.9E\n",
			RSS[0],RSS[1],RSS[2],RSS[3],		
			RSSpred[0],RSSpred[1],RSSpred[2],RSSpred[3],
			((iLevel >= 0)?Epsilon_Iz_aux[iLevel]:Epsilon_Iz),predict_eps_Iz);
		printf("RSStot %1.9E\n",RSStot);
#endif
		RSStot = 0.0;
		int iEqn = AMPZ;
		{
			RSS[iEqn] = 0.0;
			pVertex = Xarray;
			for (iVertex =0; iVertex < numUse; iVertex++)
			{
				RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
				++pVertex;
			};
			RSStot += RSS[iEqn];
		};
		if (iLevel == -1) {
			RSStot += Epsilon_Iz*Epsilon_Iz;
		} else {
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		};

		if ((iIteration > 0) && (RSStot > old_RSS*1.0000000000001))
		{
			printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
				old_RSS,RSStot);
			getch();
		}

		printf("RSS[%d]: %1.5E EpsIz: %1.5E\n",iLevel,
				RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		
		old_RSS = RSStot;

		//fprintf(fp,"iLevel %d  RSS_Ampz %1.14E  Eps_Iz %1.14E  RSSTot %1.14E \n",
		//	iLevel, RSS[3], (iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel], RSStot);

		// 1. Define regressor vectors
		
		ZeroMemory(SSreg, sizeof(real)*NUMCELLEQNS*NUMREGRESS);
		real SSextra = 0.0;
		
		// was OK with comment here  -?
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// Jacobi:
			// =======
			
			pVertex->circuit_regressor = 0.0;
			
			if (pVertex->coeff_self[AZ][AZ] == 0.0)
			{
				pVertex->regressor[JACOBI_REGR][AZ] = 0.0;
			} else {
				pVertex->regressor[JACOBI_REGR][AZ] = pVertex->epsilon[AZ] / pVertex->coeff_self[AZ][AZ]; 
			};

			memcpy(pVertex->regressor[RICHARDSON_REGR],pVertex->epsilon,sizeof(real)*NUM_AFFECTORS_1);

			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					iVar = AZ;
					SSreg[iRegrVec][iVar] += pVertex->regressor[iRegrVec][iVar] * 
											 pVertex->regressor[iRegrVec][iVar];
					if (!(SSreg[iRegrVec][iVar] == SSreg[iRegrVec][iVar])) {
						iVar = iVar;
					}
				}
			++pVertex;
		};


		// Normalize regressors:
		for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
		{
			iVar = AZ;
			L2reg[iRegrVec][iVar] = sqrt(SSreg[iRegrVec][iVar]/(real)numUse);
		}

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
			{
				 // do not divide by L2(var itself) which may be == 0
				{
					iVar = AZ;
					if (L2reg[iRegrVec][iVar] != 0.0)
						pVertex->regressor[iRegrVec][iVar] /= L2reg[iRegrVec][iVar];
				};
			};
			++pVertex;
		};

		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		for (i = 0; i < GAMMANUM; i++)
			memset(Summary.LU[i],0,sizeof(qd_or_d)*GAMMANUM);
		memset(minus_eps_sums,0,sizeof(qd_or_d)*GAMMANUM);

		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; // that should be nonzero
		
		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			// The idea is to collect [Beta_ki dot x_i] [Beta_kj dot x_j] ,
			// so first collect [Beta_ki dot x_i],
			// for each eps k [k over all eqns at this vertex], each i [i over all regressors]

			// Each epsilon^2 gets an equal shout in sums and dot products here.
			
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			iEqn = AMPZ;
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
			};
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
					iEqn = AMPZ;
					BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];
					
				};
			};
			
			iEqn = AMPZ;
			{
				for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
				{
					i1 = iRegrVec1;
					for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
					{
						i2 = iRegrVec2;
						Summary.LU[i1][i2] +=
							BetaX[AMPZ][iRegrVec1][3]*BetaX[AMPZ][iRegrVec2][3];
						// ?
					};

					Summary.LU[i1][2] += BetaX[iEqn][iRegrVec1][3]*BetaX_ext[iEqn];
					Summary.LU[2][i1] += BetaX[iEqn][iRegrVec1][3]*BetaX_ext[iEqn];
					minus_eps_sums[i1] -= BetaX[iEqn][iRegrVec1][3]*pVertex->epsilon[iEqn];
				};
				Summary.LU[2][2] +=  BetaX_ext[iEqn]*BetaX_ext[iEqn];
				minus_eps_sums[2] -= BetaX_ext[iEqn]*pVertex->epsilon[iEqn];
				// Every epsilon should appear in every epsilon sum;
				// In sum j, each epsilon appears against the change in it due to regressor j.

			}; // next iEqn = next epsilon from this location
						
			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			{
				BetaX_for_Iz[iRegrVec1][3] +=
						coeffself[IZ][3]*pVertex->regressor[iRegrVec1][3];
			};
			
			++pVertex;
		};
		
		
		// still same regardless of level ...
		
		// Add to summary:
		for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
		{
			i1 = iRegrVec1;
			for (iRegrVec2 = 0; iRegrVec2 < NUMREGRESS; iRegrVec2++)
				Summary.LU[i1][iRegrVec2] +=
						BetaX_for_Iz[iRegrVec1][3]*BetaX_for_Iz[iRegrVec2][3];
			
			Summary.LU[i1][2]  += BetaX_for_Iz[iRegrVec1][3]*BetaX_for_Iz_ext;
			Summary.LU[2][i1]  += BetaX_for_Iz[iRegrVec1][3]*BetaX_for_Iz_ext;
			minus_eps_sums[i1] -= BetaX_for_Iz[iRegrVec1][3]*
										((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		};

		Summary.LU[2][2]  += BetaX_for_Iz_ext*BetaX_for_Iz_ext;
		minus_eps_sums[2] -= BetaX_for_Iz_ext*((iLevel == -1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);

		bool bNonzero;
		for (i = 0; i < GAMMANUM; i++)
		{
			bNonzero = 0;
			for (j = 0; j < GAMMANUM; j++){
				if (Summary.LU[i][j] != 0.0)
					bNonzero = true;
				debugLU[i][j] = Summary.LU[i][j];
			};
			if (bNonzero == 0) {
				Summary.LU[i][i] = 1.0;
				printf("{ %d }",i);
			};
		};
		//printf("\n");

		printf("Summary.LU\n");
		for (i = 0; i < 3; i++)
		{
			for (j = 0; j < 3; j++)
				printf("%1.3E ",Summary.LU[i][j].x[0]);
			printf("| %1.4E \n",minus_eps_sums[i].x[0]);
		}
		printf("\n");

		// debug:
	//	Storemat.CopyFrom(Summary);

		if (Summary.LUdecomp() == 1) {
			printf("LUdecomp failed.\n");
			getch();
		}
		Summary.LUSolve(minus_eps_sums, gamma); // solve matrix equation

		printf("soln gamma %1.4E %1.4E %1.4E \n",
			gamma[0].x[0],gamma[1].x[0],gamma[2].x[0]);

#ifdef DEBUGJRLS
		printf("gam %1.3E %1.3E %1.3E %1.3E \n", gamma[0].x[0], gamma[1].x[0], gamma[2].x[0], gamma[3].x[0]);
		printf("--- %1.3E %1.3E %1.3E %1.3E %1.3E \n", gamma[4].x[0], gamma[5].x[0], gamma[6].x[0], gamma[7].x[0],gamma[8].x[0]);
		
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		// DEBUG:
		// Make predictions:
		real ROC_ext_comp3 = 0.0, ROC_ext_comp7 = 0.0, ROC_ext_comp8 = 0.0,
			ROC_ext_comp, ROC_ext_summ3 = 0.0, ROC_ext_summ7 = 0.0, ROC_ext_summ8 = 0.0,
			ROC_ext_summ, ROC_ext_eps = 0.0;
		ROC_0_3 = 0.0;
		ROC_1_3 = 0.0;
		ROC_ext = 0.0;
		memset(BetaX_for_Iz,0,sizeof(qd_or_d)*NUMREGRESS*NUM_AFFECTORS_1);
		memset(&(BetaX_for_Iz_ext),0,sizeof(qd_or_d));

		// always same coefficient:
		BetaX_for_Iz_ext = Epsilon_Iz_coeff_On_PhiAnode; 
		// still same regardless of level ...

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			memset(BetaX,0,sizeof(real)*NUM_EQNS_1*NUMREGRESS*NUM_AFFECTORS_1);
			memset(BetaX_ext,0,sizeof(real)*NUM_EQNS_1);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pVertex->regressor[iRegrVec],sizeof(real)*4); // phi,A
					BetaX[iEqn][iRegrVec][0] += coeffself[iEqn][0] * data[0];						
					BetaX[iEqn][iRegrVec][1] += coeffself[iEqn][1] * data[1];						
					BetaX[iEqn][iRegrVec][2] += coeffself[iEqn][2] * data[2];						
					BetaX[iEqn][iRegrVec][3] += coeffself[iEqn][3] * data[3];						
				};
				BetaX_ext[iEqn] += coeffself[iEqn][PHI_ANODE]
								 + coeffself[iEqn][0]*pVertex->circuit_regressor;
				// External field affects only via coeff_self				
				// pVertex->coeff_self[GAUSS][TUNEFAC] += FOUR_PI_Q*h*wt_domain[2]* pNeigh->temp2.y*
				//				 (n_e*pNeigh->sigma_e.xz-0.5*n_i*pNeigh->sigma_i.xz)*edgenorm_use[2].x
				// ...
				// We propose to change Ez by gamma_ext*temp2.y at each location.
				// This means the effect of gamma_ext on epsilon[iEqn] is coeffself[iEqn][TUNEFAC].

			};
			
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
			{
				pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
				pNeigh = Xarray + izNeigh[iNeigh];
				for (iRegrVec = 0; iRegrVec < NUMREGRESS; iRegrVec++)
				{
					memcpy(data, pNeigh->regressor[iRegrVec],sizeof(real)*4); // phi,A
					for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
					{
						BetaX[iEqn][iRegrVec][0] += coefflocal[iEqn][0]*data[0];
						BetaX[iEqn][iRegrVec][1] += coefflocal[iEqn][1]*data[1];
						BetaX[iEqn][iRegrVec][2] += coefflocal[iEqn][2]*data[2];
						BetaX[iEqn][iRegrVec][3] += coefflocal[iEqn][3]*data[3];
					};
				};
			};

			pVertex->predict_eps[0] = pVertex->epsilon[0] +
				(BetaX[0][0][3]*gamma[3] + BetaX[0][1][3]*gamma[7] + BetaX_ext[0]*gamma[8]).x[0];
			pVertex->predict_eps[1] = pVertex->epsilon[1] +
				(BetaX[1][0][3]*gamma[3] + BetaX[1][1][3]*gamma[7] + BetaX_ext[1]*gamma[8]).x[0];
			pVertex->predict_eps[2] = pVertex->epsilon[2] +
				(BetaX[2][0][3]*gamma[3] + BetaX[2][1][3]*gamma[7] + BetaX_ext[2]*gamma[8]).x[0];
			pVertex->predict_eps[3] = pVertex->epsilon[3] +
				(BetaX[3][0][3]*gamma[3] + BetaX[3][1][3]*gamma[7] + BetaX_ext[3]*gamma[8]).x[0];

			// Contribute towards Iz epsilon:
			for (iRegrVec1 = 0; iRegrVec1 < NUMREGRESS; iRegrVec1++)
			for (iGamma1 = 0; iGamma1 < NUMCELLEQNS; iGamma1++)
			{
				BetaX_for_Iz[iRegrVec1][iGamma1] +=
						coeffself[IZ][iGamma1]*pVertex->regressor[iRegrVec1][iGamma1];
			};
			
		// Test:
		// Are we predicting that each row: sum [ALL eps . deps/dgamma_i] = 0 ?

			ROC_0_3 += pVertex->predict_eps[3]*BetaX[3][0][3];
			ROC_1_3 += pVertex->predict_eps[3]*BetaX[3][1][3];
			ROC_ext += pVertex->predict_eps[3]*BetaX_ext[3];

			// ROC_ext failed to be 0.
			// Other ways to compute ROC_ext, morphing back to our formulation above:
			// First, as above:

			ROC_ext_comp3 += (BetaX[3][0][3]*gamma[3]*BetaX_ext[3]).x[0];
			ROC_ext_comp7 += (BetaX[3][1][3]*gamma[7]*BetaX_ext[3]).x[0];
			// We should find that we can then remove gamma out and get back the initial summary matrix
			// If not, we hunt down which product-sum term is different.
			ROC_ext_comp8 += (BetaX_ext[3]*gamma[8]*BetaX_ext[3]).x[0];
			ROC_ext_eps += pVertex->epsilon[3]*BetaX_ext[3];
			
			ROC_ext_summ3 += BetaX[3][0][3]*BetaX_ext[3];
			ROC_ext_summ7 += BetaX[3][1][3]*BetaX_ext[3];
			ROC_ext_summ8 += BetaX_ext[3]*BetaX_ext[3];
			
			++pVertex;
		};

		
		predict_eps_Iz = Epsilon_Iz;
		if (iLevel >= 0) predict_eps_Iz = Epsilon_Iz_aux[iLevel];

		ROC_ext_eps += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		predict_eps_Iz += (BetaX_for_Iz[0][3]*gamma[3] + BetaX_for_Iz[1][3]*gamma[7] + BetaX_for_Iz_ext*gamma[8]).x[0];
		
		printf("predict_eps_Iz %1.9E\n",predict_eps_Iz);

		ROC_0_3 += (predict_eps_Iz*BetaX_for_Iz[0][3]).x[0];
		ROC_1_3 += (predict_eps_Iz*BetaX_for_Iz[1][3]).x[0];
		ROC_ext += (predict_eps_Iz*BetaX_for_Iz_ext).x[0];

		ROC_ext_comp3 += (BetaX_for_Iz[0][3]*gamma[3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp7 += (BetaX_for_Iz[1][3]*gamma[7]*BetaX_for_Iz_ext).x[0];
		ROC_ext_comp8 += (BetaX_for_Iz_ext*gamma[8]*BetaX_for_Iz_ext).x[0];
		
		ROC_ext_summ3 += (BetaX_for_Iz[0][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ7 += (BetaX_for_Iz[1][3]*BetaX_for_Iz_ext).x[0];
		ROC_ext_summ8 += (BetaX_for_Iz_ext*BetaX_for_Iz_ext).x[0];
			
		// Generate ROCext predictions from comp and summ:
		
		ROC_ext_comp = ROC_ext_comp3+ROC_ext_comp7+ROC_ext_comp8 +ROC_ext_eps;

		printf("COMP prediction: %1.10E\n",ROC_ext_comp);

		// Compare summ with summary matrix original elements:

		ROC_ext_summ = (ROC_ext_summ3*gamma[3]+ROC_ext_summ7*gamma[7]+ROC_ext_summ8*gamma[8] +
						ROC_ext_eps).x[0];

		printf("SUMM prediction: %1.10E\n",ROC_ext_summ);

		printf("Summ %1.10E %1.10E %1.10E %1.10E \n",
			ROC_ext_summ3,ROC_ext_summ7,ROC_ext_summ8, ROC_ext_eps);
		printf("Orig %1.10E %1.10E %1.10E %1.10E \n",
			Storemat.LU[8][3].x[0],Storemat.LU[8][7].x[0],Storemat.LU[8][8].x[0],
			minus_eps_sums[8].x[0]);

		printf("\nROC_0_3 %1.9E ROC_1_3 %1.9E ROC_ext %1.9E \n\n",
			ROC_0_3,ROC_1_3,ROC_ext);

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#endif
		
		// Make additions to variables:
		// ____________________________

// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"Before add: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);
		};
*/

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			pVertex->A.z += (gamma[0]*pVertex->regressor[0][AZ]
						   + gamma[1]*pVertex->regressor[1][AZ]).x[0];
			++pVertex;
		};
		if (iLevel == -1) {
			this->EzTuning += (gamma[2]);//.x[0];
		} else {
			PhiAnode_aux[iLevel] += (gamma[2]);//.x[0];
			printf("gamma[2].x[0] %1.4E PhiAnode_aux[iLevel] %1.4E \n",
				gamma[2].x[0],PhiAnode_aux[iLevel].x[0]);
		};



// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"Before resetphi: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);

		};*/
// ////////////////////////////////////////////////
		// Reset to get avg Az = 0 and avg phi = 0 at the back.
/*
		real sum_phi,avg_phi;

		sum_phi = 0.0;
		pVertex = Xarray;
		if (iLevel == -1) {
			for (iVertex = 0; iVertex < numInnermostRow; iVertex++)
			{
				sum_phi += pVertex->phi;
				++pVertex;
			};			
			avg_phi = sum_phi/(real)numInnermostRow;
		} else {
			for (iVertex = 0; iVertex < numInnermostRowAux[iLevel]; iVertex++)
			{
				sum_phi += pVertex->phi;
				++pVertex;
			};			
			avg_phi = sum_phi/(real)numInnermostRowAux[iLevel];
		};

		pVertex = Xarray;
		for (iVertex = 0; iVertex < numUse; iVertex++)
		{
			pVertex->phi -= avg_phi;
			++pVertex;
		};
*/
		// Can no longer adjust level of phi, nor should we need to.

		// DEBUG:
		/*if (iLevel == 3) {
			CalculateEpsilonAux4(iLevel);

			FILE * fp3 = fopen("lvl3_JRLS.txt","a");
			// Survey RSS from each equation:
			RSStot = 0.0;
			for (iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				RSS[iEqn] = 0.0;
				pVertex = Xarray;
				for (iVertex =0; iVertex < numUse; iVertex++)
				{
					RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
					++pVertex;
				};
				RSStot += RSS[iEqn];
			};
			RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
			
			fprintf(fp3,"After resetphi: %1.12E %1.12E %1.12E %1.12E eps_Iz %1.12E RSStot %1.12E \n\n",
				RSS[0],RSS[1],RSS[2],RSS[3],Epsilon_Iz_aux[iLevel],RSStot);
			fclose(fp3);

		};*/

		// Is this affecting the Gauss errors? We should verify since it just decided to break.

		// Maybe it affects total Axy errors -- that is actually possible.
		// Let's look at how sum of epsilon Ampere depends on changing the level of phi.
		// It's clear it SHOULD NOT affect anything -- and changing all values at Lvl 0 implies
		// a change of all values at Lvl 1. 
		// 



	//	sum_Az = 0.0;
	//	pTri = T + StartAvgRowTri; 
		// how long will last row of tris remain inviolate? doesn't matter a whole lot though if it's a little bit wrong.
	//	for (iTri = StartAvgRowTri; iTri < numTriangles; iTri++)
	//	{
	//		sum_Az += pTri->A.z;
	//		++pTri;
	//	};
	//	avg_Az = sum_Az/(real)(numTriangles-StartAvgRowTri);
		
		// Note that nothing was being done for pTri->phi anyway.

		// When we move Az we also move chEzExt:
//		pTri = T;
//		for (iTri = 0; iTri < numTriangles; iTri++)
//		{
//			pTri->A.z -= avg_Az;
//			++pTri;
//		};
//		chEzExt -= avg_Az; // when Az goes up, chEz should go up, as this keeps E the same.

	}; // next iteration
	
	
	// DEBUG:
	if (iLevel == -1) {
		CalculateEpsilonsAz();
	} else {
		CalculateEpsilonAuxAz(iLevel);
	};
	
	// Survey RSS from each equation:
	RSStot = 0.0;
	iEqn = AMPZ;
	{
		RSS[iEqn] = 0.0;
		pVertex = Xarray;
		for (iVertex =0; iVertex < numUse; iVertex++)
		{
			RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
			++pVertex;
		};
		RSStot += RSS[iEqn];
	};
	if (iLevel == -1) {
		RSStot += Epsilon_Iz*Epsilon_Iz;
	} else {
		RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	};

	if ((RSStot > old_RSS*1.0000000000001))
	{
		printf("error RSS > old_RSS \a \nOLD_RSS %1.14E RSS %1.14E\n============+===============\n",
			old_RSS,RSStot);
		getch();
	}
}


void TriMesh::GaussSeidel(int iLevel, int iVolley) {
	// adjust one value set at a time for those belonging to iVolley

	long izNeigh[128], neigh_len, i, iVertex, numUse,
		neigh_len2,ii,izNeigh2[128];
	Vertex * Xarray, * pVertex, * pNeigh;
	real sum_beta_eps,sum_beta_beta;
	real coefflocal[NUM_EQNS_1][NUM_AFFECTORS_1];
	real coeffself[NUM_EQNS_2][NUM_AFFECTORS_2]; // 2 > 1
	int iVar,iEqn;
	real RSStot, RSS[4];

	printf("Gauss-Seidel Level %d Volley %d :\n",iLevel,iVolley);
	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel];
		numUse = numAuxVertices[iLevel];
	};
	
	// Should we keep epsilon updated? Yes, I think so.
	
	// Calculation is:
	// Sum [deps/dphi] eps_new = 0.
	// = Sum [deps/dphi] [eps_old + [deps/dphi] change ]
	// change = -sum[(deps/dphi) eps_old]/sum[ (deps/dphi)^2 ]
	
	// Have to take into account all eqns. Really should change 4
	// vars at once but let's not bother.

	real minusepssum[4];
	real sumproduct[4][4];
	real localeps[4];
	Matrix_real LU4;
	LU4.Invoke(4);
	real x[4];
	
	pVertex = Xarray;
	for (iVertex = 0; iVertex < numUse; iVertex++)
	{
		if (pVertex->iVolley == iVolley) {
			
			// Solve 4 equations:
			// sum epsilon_new beta_var_0 = 0
			// sum epsilon_new beta_var_1 = 0
			// ...
			
			//   <=>

			// sum [(eps_exist + beta_var_0 change_phi + beta_var_1 change_Ax ) beta_var_0] = 0
			
	// (sum[beta_var_0^2] sum[beta_var_0 beta_var_1]) [ change phi ] = -sum [eps_exist beta_var_0]
	// (                                            ) [  change Ax ] = -sum [eps_exist beta_var_1]
	
			// So we need to collect those sum products.

			memset(sumproduct,0,sizeof(real)*4*4);
			memset(minusepssum,0,sizeof(real)*4);
				
			// neighs are what is affected.
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			memcpy(coeffself,pVertex->coeff_self,sizeof(real)*5*6);
			memcpy(localeps,pVertex->epsilon,sizeof(real)*4);
			
			for (iVar = 0; iVar < 4; iVar++)
			{
				minusepssum[iVar] -= coeffself[0][iVar]*localeps[0]
							 + coeffself[1][iVar]*localeps[1]
							 + coeffself[2][iVar]*localeps[2]
							 + coeffself[3][iVar]*localeps[3];

			    sumproduct[0][iVar] += coeffself[0][iVar]*coeffself[0][0]
									+  coeffself[1][iVar]*coeffself[1][0]
									+  coeffself[2][iVar]*coeffself[2][0]
									+  coeffself[3][iVar]*coeffself[3][0];
				sumproduct[1][iVar] += coeffself[0][iVar]*coeffself[0][1]
									+  coeffself[1][iVar]*coeffself[1][1]
									+  coeffself[2][iVar]*coeffself[2][1]
									+  coeffself[3][iVar]*coeffself[3][1];
				sumproduct[2][iVar] += coeffself[0][iVar]*coeffself[0][2]
									+  coeffself[1][iVar]*coeffself[1][2]
									+  coeffself[2][iVar]*coeffself[2][2]
									+  coeffself[3][iVar]*coeffself[3][2];
				sumproduct[3][iVar] += coeffself[0][iVar]*coeffself[0][3]
									+  coeffself[1][iVar]*coeffself[1][3]
									+  coeffself[2][iVar]*coeffself[2][3]
									+  coeffself[3][iVar]*coeffself[3][3];
			};
			
			for (i = 0; i < neigh_len; i++)
			{
				pNeigh = Xarray + izNeigh[i];
				neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
				ii = 0; while (izNeigh2[ii] != iVertex) ii++;
				pNeigh->GetCoefficients(&(coefflocal[0][0]),ii);
				memcpy(localeps,pNeigh->epsilon,sizeof(real)*4);
			
				for (iVar = 0; iVar < 4; iVar++)
				{
					
					minusepssum[iVar] -= coefflocal[0][iVar]*localeps[0]
								   + coefflocal[1][iVar]*localeps[1]
								   + coefflocal[2][iVar]*localeps[2]
								   + coefflocal[3][iVar]*localeps[3];
					sumproduct[0][iVar] += coefflocal[0][iVar]*coefflocal[0][0]
										+  coefflocal[1][iVar]*coefflocal[1][0]
										+  coefflocal[2][iVar]*coefflocal[2][0]
										+  coefflocal[3][iVar]*coefflocal[3][0];
					sumproduct[1][iVar] += coefflocal[0][iVar]*coefflocal[0][1]
										+  coefflocal[1][iVar]*coefflocal[1][1]
										+  coefflocal[2][iVar]*coefflocal[2][1]
										+  coefflocal[3][iVar]*coefflocal[3][1];
					sumproduct[2][iVar] += coefflocal[0][iVar]*coefflocal[0][2]
										+  coefflocal[1][iVar]*coefflocal[1][2]
										+  coefflocal[2][iVar]*coefflocal[2][2]
										+  coefflocal[3][iVar]*coefflocal[3][2];
					sumproduct[3][iVar] += coefflocal[0][iVar]*coefflocal[0][3]
										+  coefflocal[1][iVar]*coefflocal[1][3]
										+  coefflocal[2][iVar]*coefflocal[2][3]
										+  coefflocal[3][iVar]*coefflocal[3][3];
				};				
			};
			
			memcpy(LU4.LU[0],sumproduct[0],sizeof(real)*4);
			memcpy(LU4.LU[1],sumproduct[1],sizeof(real)*4);
			memcpy(LU4.LU[2],sumproduct[2],sizeof(real)*4);
			memcpy(LU4.LU[3],sumproduct[3],sizeof(real)*4);
			
			LU4.LUdecomp();
			LU4.LUSolve(minusepssum,x);
			pVertex->phi += x[0];
			pVertex->A.x += x[1];
			pVertex->A.y += x[2];
			pVertex->A.z += x[3];

			
			// Now update all epsilon affected:
			RecalculateEpsilonVertex(iVertex,iLevel);
			for (i = 0; i < neigh_len; i++)
			{
				//pNeigh = Xarray + izNeigh[i];
				//pNeigh->epsilon[0] += 
				// We could almost certainly go faster by not actually recalculating
				// but just augmenting for the change added.
				RecalculateEpsilonVertex(izNeigh[i],iLevel);
			};
		}
		++pVertex;
	};
	
	if (iLevel == -1) {
		this->CalculateEpsilons();
	} else {
		this->CalculateEpsilonAux4(iLevel);
	};
	// Get eps_Iz, apart from anything else.

	// Will likely want to report RSS at each iVolley, for the experiment.
	RSStot = 0.0;
	for (iEqn = 0; iEqn < 4; iEqn++)
	{
		RSS[iEqn] = 0.0;
		pVertex = Xarray;
		for (iVertex =0; iVertex < numUse; iVertex++)
		{
			RSS[iEqn] += pVertex->epsilon[iEqn]*pVertex->epsilon[iEqn];
			++pVertex;
		};
		RSStot += RSS[iEqn];
	};
	if (iLevel == -1) {
		RSStot += Epsilon_Iz*Epsilon_Iz;
	} else {
		RSStot += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	};
		
	printf("RSS[%d]: %1.5E %1.5E %1.5E %1.5E EpsIz: %1.5E\n",iLevel,
		RSS[0],RSS[1],RSS[2],RSS[3],(iLevel==-1)?Epsilon_Iz:Epsilon_Iz_aux[iLevel]);
		
}


void TriMesh::CalculateEpsilonAux4(int iLevel)
{
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn;
	long izNeigh[128];
	int neigh_len;
	Vector3 A;
	real phi;
	int iNeigh;
	real coefflocal[4][4];

	Epsilon_Iz_aux[iLevel] = (Epsilon_Iz_Default[iLevel]
							+ Epsilon_Iz_coeff_On_PhiAnode*PhiAnode_aux[iLevel]).x[0]; 

	real f64PhiAnode = PhiAnode_aux[iLevel].x[0];

	pVertex = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		// coeff_self first:
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			pVertex->epsilon[iEqn] = pVertex->coeff_self[iEqn][UNITY]
							 + pVertex->coeff_self[iEqn][PHI_ANODE]*f64PhiAnode
							 + pVertex->coeff_self[iEqn][PHI]*pVertex->phi
							 + pVertex->coeff_self[iEqn][AX]*pVertex->A.x
							 + pVertex->coeff_self[iEqn][AY]*pVertex->A.y
							 + pVertex->coeff_self[iEqn][AZ]*pVertex->A.z;
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = AuxX[iLevel] + izNeigh[iNeigh];
			phi = pNeigh->phi;
			A = pNeigh->A;		
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				pVertex->epsilon[iEqn] += coefflocal[iEqn][PHI]*phi
										+ coefflocal[iEqn][AX]*A.x
										+ coefflocal[iEqn][AY]*A.y
										+ coefflocal[iEqn][AZ]*A.z;
			};
		};

		Epsilon_Iz_aux[iLevel] += //pVertex->coeff_self[IZ][UNITY] +
					// pVertex->coeff_self[IZ][TUNEFAC]*f64TuneFac +
					 pVertex->coeff_self[IZ][PHI]*pVertex->phi +
					 pVertex->coeff_self[IZ][AX]*pVertex->A.x +
					 pVertex->coeff_self[IZ][AY]*pVertex->A.y +
					 pVertex->coeff_self[IZ][AZ]*pVertex->A.z;
		
		++pVertex;
	};	
}
void TriMesh::CalculateEpsilonAuxAz(int iLevel)
{
	Vertex * pVertex, *pNeigh;
	long iVertex;
	int iEqn, neigh_len, iNeigh;
	long izNeigh[128];
	Vector3 A;
	real phi;
	real coefflocal[4][4];
	Epsilon_Iz_aux[iLevel] = (Epsilon_Iz_Default[iLevel]
							+ Epsilon_Iz_coeff_On_PhiAnode*PhiAnode_aux[iLevel]).x[0]; 
	real f64EzTuning = PhiAnode_aux[iLevel].x[0];
	pVertex = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		// coeff_self first:
		iEqn = AMPZ;
		{
			pVertex->epsilon[iEqn] = pVertex->coeff_self[iEqn][UNITY]
							 + pVertex->coeff_self[iEqn][PHI_ANODE]*f64EzTuning
							 + pVertex->coeff_self[iEqn][AZ]*pVertex->A.z;
		};
		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
		{
			pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
			pNeigh = AuxX[iLevel] + izNeigh[iNeigh];
			A = pNeigh->A;		
			iEqn = AMPZ;
			{
				pVertex->epsilon[iEqn] += coefflocal[iEqn][AZ]*A.z;
			};
		};
		Epsilon_Iz_aux[iLevel] += pVertex->coeff_self[IZ][AZ]*pVertex->A.z;		
		++pVertex;
	};	
}

void TriMesh::RecalculateEpsilonVertex(long iVertex,int iLevel)
{
	real coeffself[5][6];
	long neigh_len, iNeigh,iEqn, izNeigh[128];
	real phi,pVertexphi;
	Vector3 A,pVertexA;
	real coefflocal[4][4];
	Vertex * pVertex, *pNeigh, *Xarray;
	real f64PhiAnode;

	if (iLevel == -1) {
		Xarray = X;
		f64PhiAnode = PhiAnode.x[0];
	} else {
		Xarray = AuxX[iLevel];
		f64PhiAnode = PhiAnode_aux[iLevel].x[0];
	};

	pVertex = Xarray + iVertex;
	// coeff_self first:
	memcpy (&(coeffself[0][0]),pVertex->coeff_self,sizeof(real)*5*6);
	pVertexphi = pVertex->phi;
	pVertexA = pVertex->A;
	for (iEqn = 0; iEqn < 4; iEqn++)
	{
		pVertex->epsilon[iEqn] = coeffself[iEqn][UNITY]
						 + coeffself[iEqn][PHI_ANODE]*f64PhiAnode // !!
						 + coeffself[iEqn][PHI]*pVertexphi
						 + coeffself[iEqn][AX]*pVertexA.x
						 + coeffself[iEqn][AY]*pVertexA.y
						 + coeffself[iEqn][AZ]*pVertexA.z;
	};
	neigh_len = pVertex->GetNeighIndexArray(izNeigh);
	for (iNeigh = 0; iNeigh < neigh_len; iNeigh++)
	{
		pVertex->GetCoefficients(&(coefflocal[0][0]),iNeigh);
		pNeigh = Xarray + izNeigh[iNeigh];
		phi = pNeigh->phi;
		A = pNeigh->A;		
		for (iEqn = 0; iEqn < 4; iEqn++)
		{
			pVertex->epsilon[iEqn] += coefflocal[iEqn][PHI]*phi
									+ coefflocal[iEqn][AX]*A.x
									+ coefflocal[iEqn][AY]*A.y
									+ coefflocal[iEqn][AZ]*A.z;
		};
	};

	// could speed up by dropping this routine and just updating epsilon per
	// the thing that has changed!!

}

void TriMesh::Lift_to_coarse_eps(int iLevel)
{/*
	// iLevel is the level where epsilon is to be populated.
	// Populate ->Temp as the eps we have when coarse A = 0. 
	long iVertex, iEqn;
	Vertex * pAux, *pCoarseVertex, *Xarray;
	Vertex * pVertex;
	real wt[4][4];
	long iCoarse, i, numUse;
	
	// Can we assume pTri->cc is populated when we get here? Yes.
	
//	bZeroSeed = true;
	Epsilon_Iz_aux[iLevel] = 0.0;
	PhiAnode_aux[iLevel] = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		memset(&(pAux->epsilon),0,sizeof(pAux->epsilon));
		memset(&(pAux->A),0,sizeof(Vector3));
		//memset(&(pAux->v_e),0,sizeof(Vector3));
		memset(&(pAux->phi),0,sizeof(real));
		
		// What we are going to set up:
		for (iEqn = 0; iEqn < 5; iEqn++)
			pAux->coeff_self[iEqn][UNITY] = 0.0;
		// So this actually wipes out any "Galerkin" coefficient setting with [UNITY]...
		++pAux;
	};


	if (iLevel == 0) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel-1]; 
		numUse = numAuxVertices[iLevel-1];
	};

	int iInput;
	// Populate from level below

	char buffer[256];
	sprintf(buffer,"ascend%d.txt",iLevel);
	FILE * fp = fopen(buffer,"w");

	pAux = Xarray;
	for (iVertex = 0; iVertex < numUse; iVertex++)
	{
		// contribute where relevant.
		// Will this way work?
		// Each fine point knows a list of coarse indices and weights
		
		fprintf(fp,"%d | ",iVertex);
				
		for (i = 0; i < pAux->coarse_len; i++)
		{
			iCoarse = pAux->GetCoarseIndex(i);
			pCoarseVertex = AuxX[iLevel] + iCoarse;
			memcpy(&(wt[0][0]),pAux->wt_eps[i],sizeof(real)*4*4);
			
			// Vector weight version:
			
			//if (pAux->PBC_uplink[i] == 0) {
			for (iEqn = 0; iEqn < 4; iEqn++)
			for (iInput = 0; iInput < 4; iInput++)
				pCoarseVertex->coeff_self[iEqn][UNITY] += wt[iEqn][iInput]*pAux->epsilon[iInput];

			if (i < 3) {
				fprintf(fp,"%d %1.8E %1.8E %1.8E : ",iCoarse,wt[0][0],wt[0][1],wt[0][3]);
				fprintf(fp," %1.8E | ",
										pCoarseVertex->coeff_self[GAUSS][UNITY]);
			};

			
		};
		fprintf(fp,"\n");
		++pAux;
	};			
	fclose(fp);
	
	// HOW TO DO THE IZ EQUATION. :

// At finest we take :
// Epsilon_Iz = Epsilon_Iz_constant + 
//              Epsilon_Iz_coeff_on_chEz_ext * chEzExt
//			  + [IZ]coefficient*self+neigh data at each point.
// 
// Aux: chEz_aux will be the addition to chEzExt.
// Post epsilon_Iz up to the next level 
// Epsilon_Iz_coarse = Epsilon_Iz + 
//		[IZ]coefficient*additional data at each point
//		+ Epsilon_Iz_coeff_on_chEz_ext * chEz_additional
	if (iLevel == 0) {
		Epsilon_Iz_Default[iLevel] = Epsilon_Iz; 
	} else {
		// We do want to calculate Iz as it exists with A=E=0 at next level.
		Epsilon_Iz_Default[iLevel] = Epsilon_Iz_aux[iLevel-1]; 
	};*/

}
void TriMesh::Lift_to_coarse_eps_Az(int iLevel)
{
	// iLevel is the level where epsilon is to be populated.
	// Populate ->Temp as the eps we have when coarse A = 0. 
	long iVertex, iEqn;
	Vertex * pAux, *pCoarseVertex, *Xarray;
	Vertex * pVertex;
	long iCoarse, i, numUse;
	
	// Can we assume pTri->cc is populated when we get here? Yes.
	
//	bZeroSeed = true;
	Epsilon_Iz_aux[iLevel] = 0.0;
	PhiAnode_aux[iLevel] = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		memset(&(pAux->epsilon),0,sizeof(pAux->epsilon));
		memset(&(pAux->A),0,sizeof(Vector3));
		//memset(&(pAux->v_e),0,sizeof(Vector3));
		memset(&(pAux->phi),0,sizeof(real));
		
		// What we are going to set up:
		for (iEqn = 0; iEqn < 5; iEqn++)
			pAux->coeff_self[iEqn][UNITY] = 0.0;
		// So this actually wipes out any "Galerkin" coefficient setting with [UNITY]...
		++pAux;
	};

	if (iLevel == 0) {
		Xarray = X;
		numUse = numVertices;
	} else {
		Xarray = AuxX[iLevel-1]; 
		numUse = numAuxVertices[iLevel-1];
	};

	int iInput;
	// Populate from level below

	pAux = Xarray;
	for (iVertex = 0; iVertex < numUse; iVertex++)
	{
		// contribute where relevant.
		// Will this way work?
		// Each fine point knows a list of coarse indices and weights
		
		for (i = 0; i < pAux->coarse_len; i++)
		{
			iCoarse = pAux->GetCoarseIndex(i);
			pCoarseVertex = AuxX[iLevel] + iCoarse;
			
			pCoarseVertex->coeff_self[AMPZ][UNITY] += pAux->weight[i]*pAux->epsilon[AMPZ];
		};
		++pAux;
	};

	// HOW TO DO THE IZ EQUATION. :

// At finest we take :
// Epsilon_Iz = Epsilon_Iz_constant + 
//              Epsilon_Iz_coeff_on_chEz_ext * chEzExt
//			  + [IZ]coefficient*self+neigh data at each point.
// 
// Aux: chEz_aux will be the addition to chEzExt.
// Post epsilon_Iz up to the next level 
// Epsilon_Iz_coarse = Epsilon_Iz + 
//		[IZ]coefficient*additional data at each point
//		+ Epsilon_Iz_coeff_on_chEz_ext * chEz_additional
	if (iLevel == 0) {
		Epsilon_Iz_Default[iLevel] = Epsilon_Iz; 
	} else {
		// We do want to calculate Iz as it exists with A=E=0 at next level.
		Epsilon_Iz_Default[iLevel] = Epsilon_Iz_aux[iLevel-1]; 
	};
}

void TriMesh::Affect_vars_finer(int iLevel)
{/*
	Vertex * pVertex, *Xarray, *pCoarseVertex;
	int i;
	real wt[4];
	long numUse;
	long iVertex, iTri;

	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
		this->PhiAnode += this->PhiAnode_aux[iLevel+1];
	} else {
		Xarray = AuxX[iLevel]; 
		numUse = numAuxVertices[iLevel];
		this->PhiAnode_aux[iLevel] += this->PhiAnode_aux[iLevel+1];
	};

	Vector3 temp3;
	pVertex = Xarray;
	for (iVertex = 0; iVertex < numUse; iVertex++)
	{
		for (i = 0; i < pVertex->coarse_len; i++)
		{
			pCoarseVertex = AuxX[iLevel+1] + pVertex->iCoarseIndex[i];

			if ((pVertex->iCoarseIndex[i] == 3496) && (iVertex == 14106))
			{
				i = i;
			};

			memcpy(wt,pVertex->wt_var[i],sizeof(real)*4);

			pVertex->phi += wt[PHI]*pCoarseVertex->phi;
			if (pVertex->PBC_uplink[i] == 0) {
				pVertex->A.x += wt[AX]*pCoarseVertex->A.x;
				pVertex->A.y += wt[AY]*pCoarseVertex->A.y;
				pVertex->A.z += wt[AZ]*pCoarseVertex->A.z;
			} else {
				if (pVertex->PBC_uplink[i] == ANTICLOCKWISE) {
					// in this case we have to rotate clockwise to affect the finer vertex.
					temp3 = Clockwise3*pCoarseVertex->A;
				} else {
					temp3 = Anticlockwise3*pCoarseVertex->A;
				};
				pVertex->A.x += wt[AX]*temp3.x;
				pVertex->A.y += wt[AY]*temp3.y;
				pVertex->A.z += wt[AZ]*temp3.z;
			};
			
			if (pCoarseVertex->A.z != 0.0) {
			//	printf("iVert %d effec %1.11E <- %d ;=> %1.11E\n",iVertex,wt*pCoarseVertex->A.z,pVertex->iCoarseIndex[i],
			//		wt*pCoarseVertex->A.z*pVertex->coeff_self[IZ][AZ]);
				
			};
		};
		++pVertex;
	};*/

}

void TriMesh::Affect_vars_finer_Az(int iLevel)
{
	Vertex * pVertex, *Xarray, *pCoarseVertex;
	int i;
	real wt[4];
	long numUse;
	long iVertex, iTri;

	if (iLevel == -1) {
		Xarray = X;
		numUse = numVertices;
		this->EzTuning += this->PhiAnode_aux[iLevel+1];
	} else {
		Xarray = AuxX[iLevel]; 
		numUse = numAuxVertices[iLevel];
		this->PhiAnode_aux[iLevel] += this->PhiAnode_aux[iLevel+1];
	};
	
	pVertex = Xarray;
	for (iVertex = 0; iVertex < numUse; iVertex++)
	{
		for (i = 0; i < pVertex->coarse_len; i++)
		{
			pCoarseVertex = AuxX[iLevel+1] + pVertex->iCoarseIndex[i];

			pVertex->A.z += pVertex->weight[i]*pCoarseVertex->A.z;			
		};
		++pVertex;
	};
}


void TriMesh::RunLU(int const iLevel, bool bRefreshCoeff)
{
	static Matrix eqns; // presumably static variable in method is like element of object ??
	// 300 x 300 matrix => 90000 x 8 bytes = 720kB
	// That said - object does actually contain Matrix Coarsest which we do not use apparently.

	qd_or_d * epsilon_original;
	qd_or_d * epsilon1, *phi, * epsilon2, * epsilon2_small, *epsilon_reduced;

	dd_real eps;
		
	int jRow, j;
	qd_or_d Beta_ij;
	Vertex * pAux, *pAux2;
	long iAux, i, iVar, iEqn;
	real tempval;
	char buffer[245];
	static int counter = 0;
	FILE * fp2, * fp;
	real RSS;
	long iVertex;
	int const NUMBER_OVERDETERMINED_START = numAuxVertices[iLevel]; 
	// out of about 75... 25 per cm sq => 6 in a row at back??
	
	if (bNeumannPhi == false) {
		// I am going to scrap everything extra, just assume that the matrix gives well-defined solutions
		// by virtue of Dirichlet BCs at the finest level.
	};
	
	counter++;
	static Matrix Original, Debug1, Debug2,Copy2,
		Debug2_small, Debug2_original, Debug2_small_original;	
	long const numEqns = NUMCELLEQNS*numAuxVertices[iLevel] + 1;
	
	CalculateEpsilonAux4(iLevel);
	RSS = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		RSS +=	pAux->epsilon[0]*pAux->epsilon[0] +
				pAux->epsilon[1]*pAux->epsilon[1] +
				pAux->epsilon[2]*pAux->epsilon[2] +
				pAux->epsilon[3]*pAux->epsilon[3];
		++pAux;
	}
	RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	
	//printf("counter = %d ; RSS before LU: %1.14E EpsIz %1.14E \n",counter, RSS,
	//	Epsilon_Iz_aux[iLevel]);

//	fp = fopen("RSS.txt","a");
//	fprintf(fp,"iLevel %d  LU : RSSTot %1.14E \n",iLevel,RSS);
//	fclose(fp);
	
	real old_RSS = RSS;
	
	sprintf(buffer,"pre%d.txt",counter);
	fp = fopen(buffer,"w");
	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{
		fprintf(fp,"%d %1.14E %1.14E %1.14E %1.14E | %1.14E %1.14E %1.14E %1.14E | ",
			iAux,pAux->epsilon[0],pAux->epsilon[1],pAux->epsilon[2],pAux->epsilon[3],
			pAux->phi,pAux->A.x,pAux->A.y,pAux->A.z);

		fprintf(fp,"%1.14E %1.14E \n",this->Epsilon_Iz_aux[iLevel],this->PhiAnode_aux[iLevel]);
		++pAux;
	}
	fclose(fp);
	
	printf("old RSS = %1.14E \n",old_RSS);

	if (bRefreshCoeff)
	{
		printf("creating LU ... ");
		// Fill in matrix 'eqns'

		// Gauss; Ampx; Ampy; Ampz; each cell
		// final one = Iz
		// So index is iAux*4+iVar

		if(eqns.Invoke(numEqns)) // does nothing if number of eqns unchanged.
		{
			printf("Memory allocation error eqns.Invoke \n");
		} else {
			if (Original.Invoke(numEqns))
				printf("Memory allocation error Original.Invoke \n");
			Debug1.Invoke(numEqns);
			Debug2.Invoke(numEqns);
			Copy2.Invoke(numEqns);
			Debug2_original.Invoke(numEqns);
			Debug2_small.Invoke(numAuxVertices[iLevel]);
			Debug2_small_original.Invoke(numAuxVertices[iLevel]);
		}

		//ZeroMemory(eqns.LU,sizeof(real)*numEqns*numEqns);
		// That assertion is not legit. And, it is now done as part of Invoke.

		long neigh_len, izNeigh[128];
		real coefflocal[4][4];
		int row;

		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				int row = iAux*4+iEqn;
				for (iVar = 0; iVar < 4; iVar++)
					eqns.LU[row][iAux*4+iVar] = pAux->coeff_self[iEqn][iVar];				
				// Consider the effects on this epsilon from chEz_aux, the final variable:
				eqns.LU[row][numEqns-1] = pAux->coeff_self[iEqn][PHI_ANODE];
			};
			neigh_len = pAux->GetNeighIndexArray(izNeigh);
			for (i = 0; i < neigh_len; i++)
			{
				pAux->GetCoefficients(&(coefflocal[0][0]),i);
				for (iEqn = 0; iEqn < 4; iEqn++)
				{
					row = iAux*4+iEqn;
					eqns.LU[row][izNeigh[i]*4+0] = coefflocal[iEqn][0];
					eqns.LU[row][izNeigh[i]*4+1] = coefflocal[iEqn][1];
					eqns.LU[row][izNeigh[i]*4+2] = coefflocal[iEqn][2];
					eqns.LU[row][izNeigh[i]*4+3] = coefflocal[iEqn][3];
				};
			};

			// not considered IZ eqn yet.
			// iVar also not yet counted unity, chEz.
			for (iEqn = 0; iEqn < 4; iEqn++)
			{
				row = iAux*4+iEqn;
				if (eqns.LU[row][iAux*4]+eqns.LU[row][iAux*4+1]+eqns.LU[row][iAux*4+2]+eqns.LU[row][iAux*4+3] == 0.0) 
					{ eqns.LU[row][row] = 1.0; } // assume in this case that it's an unpopulated row & epsilon
			};

			// Now it remains to consider the effects on Iz from this point and its neighbours:
			eqns.LU[numEqns-1][iAux*4] = pAux->coeff_self[IZ][PHI];
			eqns.LU[numEqns-1][iAux*4+1] = pAux->coeff_self[IZ][AX];
			eqns.LU[numEqns-1][iAux*4+2] = pAux->coeff_self[IZ][AY];
			eqns.LU[numEqns-1][iAux*4+3] = pAux->coeff_self[IZ][AZ];
			++pAux;
		}
		eqns.LU[numEqns-1][numEqns-1] = this->Epsilon_Iz_coeff_On_PhiAnode;

		Original.CopyFrom(eqns);	

		if (bNeumannPhi) {

			// replace with overdetermined equations:
			// every phi equation is replaced by least squares taking into account
			// its impact on phi_avg.

			// To create the RHS vector we will have to store the original matrix:
			Original.CopyFrom(eqns);			
			Debug1.CopyFrom(eqns);

			// Equation j becomes d [ sum eps^2 ]/ d x_j == 0

			// ie sum_i [ Beta_ij Beta_i.x ] == sum_i [b_i Beta_ij]
			// where Beta_ij here is ___the coefficient on x_j in equation i___, ie row i column j of our original matrix
			
			// Original version: manipulate Gauss equations only.
			/*
			pAux = AuxX[iLevel];
			for (jRow = 0; jRow < numEqns-1; jRow += 4) {
				memset(eqns.LU[jRow],0,sizeof(qd_or_d)*numEqns); // the row to replace
				
				if (jRow == 0) {
					
		// Different Idea:
		// Try setting one value to 0, the rest LS all eqns.
					eqns.LU[0][0] = 1.0; 

				} else {
				
					for (i = 0; i < numEqns-1; i+=4)
					{
						Beta_ij = Original.LU[i][jRow];  // d eps_i / d phi_j					
						// Changed here to i+=4 : optimise over Gauss equations. Other eqns catered for by other variables.
						
						// Wiped out whole row then we added the original rows.
						// This is not obviously wrong. Why do coeffs on other variables add to exactly zero.
						if (Beta_ij != 0.0)
						{
							// add Beta_ij times row i
							for (j = 0; j < numEqns; j++)
								eqns.LU[jRow][j] += Beta_ij*Original.LU[i][j];	
							// Here carrying out dd-precision calculations.
						};
					};		
				};

				//// final LS eqn to include in all: sum of contrib_to_phi_avg == 0
				////pAux = AuxX[iLevel] + jRow/4;
				//Beta_ij = 1.0;//pAux->contrib_to_phi_avg;
				//if (Beta_ij != 0.0)
				//{
				//	pAux2 = AuxX[iLevel];
				//	for (j = 0; j < numEqns-1; j+=4)
				//	{
				//		eqns.LU[jRow][j] += Beta_ij*1.0;//pAux2->contrib_to_phi_avg;
				//		++pAux2;
				//	};
				//};
				++pAux;
			};		*/

			// New version: replace every single equation with LS.
/*
			for (jRow = 0; jRow < numEqns; jRow ++) {
				memset(eqns.LU[jRow],0,sizeof(qd_or_d)*numEqns); // the row to replace
				for (i = 0; i < numEqns; i++) // the rows we combine
				{
					Beta_ij = Original.LU[i][jRow];  // d eps_i / d phi_j					
					if (Beta_ij != 0.0)
					{
						// add Beta_ij times row i
						for (j = 0; j < numEqns; j++)
							eqns.LU[jRow][j] += Beta_ij*Original.LU[i][j];					
						// Here carrying out dd-precision calculations.
					};
				};
				
				// final LS eqn to include where phi contributes: sum of contrib_to_phi_avg == 0
				if ((jRow % 4 == 0) && (jRow < numEqns-1)) {
					Beta_ij = 1.0;//pAux->contrib_to_phi_avg;
					//pAux2 = AuxX[iLevel];
					for (j = 0; j < numEqns-1; j+=4)
					{
						eqns.LU[jRow][j] += Beta_ij*1.0;//pAux2->contrib_to_phi_avg;
						//++pAux2;
					};					
				};

			};		
*/
			Debug2.CopyFrom(eqns);

		}; // if (bNeumannPhi)

		Copy2.CopyFrom(eqns);

		eqns.LUdecomp(); 		
		printf("Done refresh coeff + decomp\n");
	};

	qd_or_d * epsilon = new qd_or_d[numEqns];
	qd_or_d * x = new qd_or_d[numEqns];
	dd_real * dbg = new dd_real[numEqns];
	ZeroMemory(epsilon,sizeof(qd_or_d)*numEqns);

	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{
		// Trying to set it equal to the error that has to be cancelled.
		epsilon[iAux*4] = -pAux->coeff_self[GAUSS][UNITY];
		epsilon[iAux*4+1] = -pAux->coeff_self[AMPX][UNITY];
		epsilon[iAux*4+2] = -pAux->coeff_self[AMPY][UNITY];
		epsilon[iAux*4+3] = -pAux->coeff_self[AMPZ][UNITY];
		// pAux->epsilon has been calculated but let's stick with what
		// it was calculated from.		
		// setting dd_real equal to double
		++pAux;
	}
	epsilon[numEqns-1] = -this->Epsilon_Iz_Default[iLevel]; //-this->Epsilon_Iz_aux[iLevel];
	
	if (bNeumannPhi) {		
		epsilon_original = new qd_or_d[numEqns];
		memcpy(epsilon_original,epsilon,sizeof(qd_or_d)*numEqns);
		epsilon1 = new qd_or_d[numEqns];
		memcpy(epsilon1,epsilon,sizeof(qd_or_d)*numEqns);
		epsilon2 = new qd_or_d[numEqns];
		epsilon2_small = new qd_or_d[numAuxVertices[iLevel]];
		epsilon_reduced = new qd_or_d[numAuxVertices[iLevel]];
		phi = new qd_or_d[numAuxVertices[iLevel]];

		// ie sum_i [ Beta_ij Beta_i.x ] == sum_i [b_i Beta_ij]
		// where Beta_ij here is ___the coefficient on x_j in equation i___, ie row i column j of our original matrix

		// To create the RHS vector we will have to store the original matrix.
		/*for (jRow = 0; jRow < numEqns-1; jRow+=4)
		{
			epsilon[jRow] = 0.0;		
			// THIS MUST MATCH WHAT APPEARS ABOVE: SAME EPS COUNTED HERE AS EQNS ADDED FOR LS ABOVE...
			for (i = 0; i < numEqns-1; i+=4)
				epsilon[jRow] += Original.LU[i][jRow]* epsilon_original[i];
			// dd_real manipulations here
		};*/
/*
		// Different Idea:
		// Try setting one value to 0, the rest LS all eqns.
		epsilon[0] = 0.0; 
		for (jRow = 4; jRow < numEqns-1; jRow+=4)
		{
			epsilon[jRow] = 0.0;		
			// THIS MUST MATCH WHAT APPEARS ABOVE: SAME EPS COUNTED HERE AS EQNS ADDED FOR LS ABOVE...
			for (i = 0; i < numEqns-1; i+=4)
				epsilon[jRow] += Original.LU[i][jRow]* epsilon_original[i];
			// dd_real manipulations here
		};*/


		/*
		// LS all eqns:

		for (jRow = 0; jRow < numEqns; jRow++)
		{
			epsilon[jRow] = 0.0;		
			// THIS MUST MATCH WHAT APPEARS ABOVE: SAME EPS COUNTED HERE AS EQNS ADDED FOR LS ABOVE...
			for (i = 0; i < numEqns; i++)
				epsilon[jRow] += Original.LU[i][jRow]* epsilon_original[i];
			// dd_real manipulations here
		};*/

		memcpy(epsilon2,epsilon,sizeof(qd_or_d)*numEqns);
	};

	eqns.LUSolve(epsilon, x);

	dd_real RSS_ = 0.0;
	for (i = 0; i < numEqns; i++)
	{
		// calculate error in original equation:
		eps = -epsilon[i];
		for (j = 0; j < numEqns; j++)
			eps += Original.LU[i][j]*x[j];
		RSS_ += eps*eps;
	};
	printf("RSS dd_real %1.14E \n",RSS_.x[0]);

/*	sprintf(buffer,"LUsoln_LS_subtract.txt");
	fp2 = fopen(buffer,"w");
	dd_real eps,RSS_ = 0.0,
		RSS_Gauss = 0.0, RSS_orig = 0.0, RSS_origGauss = 0.0;
	for (i = 0; i < numEqns; i++)
	{
		for (j = 0; j < numEqns; j++)
			fprintf(fp2,"%1.14E ",Original.LU[i][j].x[0]);

		// calculate error in original equation:
		eps = -epsilon_original[i];
		for (j = 0; j < numEqns; j++)
			eps += Original.LU[i][j]*x[j];
		RSS_orig += eps*eps;
		if ((i %4 == 0) && (i < numEqns-1)) RSS_origGauss += eps*eps;
		
		fprintf(fp2,"# %1.14E # %1.14E # %1.14E \n",
			epsilon_original[i].x[0],x[i].x[0],eps.x[0]);
	};

	fprintf(fp2,"\n\n");
	for (i = 0; i < numEqns; i++)
	{
		for (j = 0; j < numEqns; j++)
			fprintf(fp2,"%1.14E ",Debug2.LU[i][j].x[0]);
		
		eps = -epsilon[i];
		for (j = 0; j < numEqns; j++)
			eps += Copy2.LU[i][j]*x[j];
		RSS_ += eps*eps;

		fprintf(fp2,"# %1.14E # %1.14E # %1.14E \n",
			epsilon[i].x[0],x[i].x[0],eps.x[0]);
	};
	fclose(fp2);

	printf("RSS for LS  %1.8E Gauss %1.8E ; \nfor original %1.8E Gauss %1.8E \n",RSS_.x[0],
		RSS_Gauss.x[0],RSS_orig.x[0],RSS_origGauss.x[0]);
	*/
	// -----------------------------------------------

	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{		
		pAux->phi = x[iAux*4].x[0];  // - subtract;
		pAux->A.x = x[iAux*4+1].x[0];
		pAux->A.y = x[iAux*4+2].x[0];
		pAux->A.z = x[iAux*4+3].x[0];		
		++pAux;
	}
	PhiAnode_aux[iLevel] = x[numEqns-1];
		
	CalculateEpsilonAux4(iLevel);
	RSS = 0.0;
	real RSSGauss = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		RSS += pAux->epsilon[0]*pAux->epsilon[0] +
			pAux->epsilon[1]*pAux->epsilon[1] +
			pAux->epsilon[2]*pAux->epsilon[2] +
			pAux->epsilon[3]*pAux->epsilon[3];
		RSSGauss += pAux->epsilon[0]*pAux->epsilon[0];
		++pAux;
	}
	RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	printf("dp RSS after LU, no subtract done: %1.14E EpsIz %1.14E",RSS,Epsilon_Iz_aux[iLevel]);
	printf("RSSGauss: %1.12E \n",RSSGauss);
	// ======================================================================
/*
	dd_real avg_phi = 0.0;
	for (i = 0; i < numEqns-1; i+=4)
	{
		avg_phi += x[i];
	};
	avg_phi /= (dd_real)(numAuxVertices[iLevel]);
	for (i = 0; i < numEqns-1; i+=4)
	{
		x[i] -= avg_phi;
	};	
	
	printf("subtracted %1.12E \n",avg_phi.x[0]);

	fp = fopen("LUeqns_soln.txt","w");
	RSS_ = 0.0;
	for (i = 0; i < numEqns; i++)
	{
		// We should find some way here of assessing Ax - b.
		eps = -epsilon[i];
		for (j = 0; j < numEqns; j++)
			eps += Copy2.LU[i][j]*x[j];
		for (j = 0; j < numEqns; j++)
			fprintf(fp, " %1.14E ",Original.LU[i][j].x[0]);
		fprintf(fp," : %d x %1.14E target %1.14E error %1.14E\n",i,x[i].x[0],epsilon_original[i].x[0],eps.x[0]);
		RSS_ += eps*eps;
	};
	fclose(fp);
	printf("RSS for LS matrix AFTER subtraction %1.12E \n",RSS_.x[0]);
	
	real contributed_phi_sum = 0.0;
	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{		
		pAux->phi = x[iAux*4].x[0];  // - subtract;
		pAux->A.x = x[iAux*4+1].x[0];
		pAux->A.y = x[iAux*4+2].x[0];
		pAux->A.z = x[iAux*4+3].x[0];	
		// debug:
		contributed_phi_sum += pAux->phi;
		++pAux;
	}
	PhiAnode_aux[iLevel] = x[numEqns-1];
		
	// debug:
	memcpy(dbg,x,sizeof(dd_real)*numEqns);

	CalculateEpsilonAux4(iLevel);
	RSS = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		RSS += pAux->epsilon[0]*pAux->epsilon[0] +
			pAux->epsilon[1]*pAux->epsilon[1] +
			pAux->epsilon[2]*pAux->epsilon[2] +
			pAux->epsilon[3]*pAux->epsilon[3];
		++pAux;
	}
	RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	printf("dp RSS after subtraction: %1.14E \n",RSS);
	*/
	Vertex * pVertex;
	sprintf(buffer,"LUresult%d.txt",counter);
	fp = fopen(buffer,"w");
	pVertex = AuxX[iLevel];
	fprintf(fp,"PhiAnode %1.12E Iz err: %1.12E \n",PhiAnode_aux[iLevel].x[0], Epsilon_Iz_aux[iLevel]);
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		fprintf(fp,"%d %1.12E %1.12E contrib %1.12E | %1.12E %1.12E %1.12E %1.12E eps %1.12E %1.12E %1.12E %1.12E initial %1.12E %1.12E %1.12E %1.12E \n",
			iVertex, pVertex->pos.x,pVertex->pos.y,pVertex->contrib_to_phi_avg,
			pVertex->phi,pVertex->A.x,pVertex->A.y,pVertex->A.z,
			pVertex->epsilon[0],pVertex->epsilon[1],pVertex->epsilon[2],pVertex->epsilon[3],
			pVertex->coeff_self[GAUSS][UNITY],
			pVertex->coeff_self[AX][UNITY],
			pVertex->coeff_self[AY][UNITY],
			pVertex->coeff_self[AZ][UNITY]);
		++pVertex;
	};
	fclose(fp);
	

	int ii,jj;
	real product;

	if (0) {//bRefreshCoeff) {

		// Now programmatically, make a matrix : epsilon_j depsilon_j/dx_i 
	/*	fp = fopen("LUtest_derivs.txt","w");
		for (i = 0; i < numAuxVertices[iLevel]; i++)
		{
			for (ii = 0; ii < 4; ii++)
			{
				pAux = AuxX[iLevel];
				for (j = 0; j < numAuxVertices[iLevel]; j++)
				{
					for (jj = 0; jj < 4; jj++)
					{
						product = (Original.LU[j*4+jj][i*4+ii]).x[0]*pAux->epsilon[jj];
						fprintf(fp," %1.14E ",product);
					};				
					++pAux;
				};
				// Now got to put last 2 epsilons: Iz and contrib_to_phi_avg.
				fprintf(fp," %1.14E ",Epsilon_Iz_aux[iLevel]*(Original.LU[numEqns-1][i*4+ii].x[0]));

				if (ii == 0) {
					fprintf(fp," %1.14E ",//contributed_phi_sum*AuxX[iLevel][i].contrib_to_phi_avg);
											contributed_phi_sum);
				} else {
					fprintf(fp," 0.0 ");
				};

				// All the rows should sum to zero as (d/dx_i)sum[eps^2] = 0
				fprintf(fp,"\n");
			};
		};
		fclose(fp);*/
	};

	if (RSS > old_RSS)
	{
		printf("\n\n\n\n*\n*\n\aRSS > old_RSS: LU FAILED\n\n*\n*\n\nrun 128*JRLS instead:");
		// Put in a break here if we want to debug this.

		getch();

		// revert to the zero seed that was better, to do iterations:
		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			pAux->phi = 0.0;
			memset(&(pAux->A),0,sizeof(Vector3));
			++pAux;
		};
		PhiAnode_aux[iLevel] = 0.0;
		IterationsJRLS(iLevel, 128);
	};

	if (0) {// bRefreshCoeff) {

		// DEBUG: Go again.
		// We have supposedly removed the influence of phi on Az, Iz.
		// Step 1. Set equations to get Ampere & Iz using A & chEz.
		// For this exercise we can set phi = 0.

		printf("\nStep 1.\n");

		// This comes out wrong even when the full matrix is right. 
		// What gives?


		// we copied to Debug1, epsilon1;

		// We basically want to knock out Gauss and put phi = 0.
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			for (j = 0; j < numEqns; j++)
				Debug1.LU[iAux*4][j] = 0.0;
			Debug1.LU[iAux*4][iAux*4] = 1.0;
			epsilon1[iAux*4] = 0.0;
		}         
		
		sprintf(buffer,"LUdebug1_%d.txt",counter);
		fp = fopen(buffer,"w");
		for (i = 0; i < numEqns; i++)
		{
			for (int j = 0; j < numEqns; j++)
			{
				fprintf(fp,"%1.13E ",Debug1.LU[i][j].x[0]);
			}
			fprintf(fp,"| %1.13E \n",epsilon1[i].x[0]);
		};
		fclose(fp);

		Debug1.LUdecomp();
		Debug1.LUSolve(epsilon1,x);

		// Spit out solution : should be getting same A and chEz as above if Ampere does not have phi

		sprintf(buffer,"LUdebug1soln.txt");
		fp = fopen(buffer,"w");
		for (i = 0; i < numEqns; i++)
		{
			fprintf(fp,"%1.14E \n",x[i].x[0]);
		} 

		// Now assess Ampere & Iz errors:
		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{		
			pAux->phi = x[iAux*4].x[0];  // - subtract;
			pAux->A.x = x[iAux*4+1].x[0];
			pAux->A.y = x[iAux*4+2].x[0];
			pAux->A.z = x[iAux*4+3].x[0];
			++pAux;
		}
		PhiAnode_aux[iLevel] = x[numEqns-1];
		// Why this doesn't work after the 1st time even on 
		// the Az-only pass?
			
		CalculateEpsilonAux4(iLevel);
		RSS = 0.0;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			RSS +=
				pAux->epsilon[1]*pAux->epsilon[1] +
				pAux->epsilon[2]*pAux->epsilon[2] +
				pAux->epsilon[3]*pAux->epsilon[3];
			++pAux;
		}
		RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
		printf("Ampere & Iz RSS after Step 1: %1.14E EpsIz %1.14E",RSS,Epsilon_Iz_aux[iLevel]);

		// Step 2. This time set A & chEz to their solved values. Try to set phi to achieve Gauss.
		printf("Step 2.\n");

		// This is the bit that failed. So let's go again:
		// Make a smaller matrix and a set of calculated constants. 
		// Why is it we can't get there?
		
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			for (j = 0; j < numEqns; j++)
			{
				Debug2.LU[iAux*4+1][j] = 0.0;
				Debug2.LU[iAux*4+2][j] = 0.0;
				Debug2.LU[iAux*4+3][j] = 0.0;
			}
			Debug2.LU[iAux*4+1][iAux*4+1] = 1.0;
			Debug2.LU[iAux*4+2][iAux*4+2] = 1.0;
			Debug2.LU[iAux*4+3][iAux*4+3] = 1.0;
			
			epsilon2[iAux*4+1] = x[iAux*4+1];
			epsilon2[iAux*4+2] = x[iAux*4+2];
			epsilon2[iAux*4+3] = x[iAux*4+3];
		}
		for (j = 0; j < numEqns; j++)
			Debug2.LU[numEqns-1][j] = 0.0;
		Debug2.LU[numEqns-1][numEqns-1] = 1.0;
		epsilon2[numEqns-1] = x[numEqns-1];
				
		Debug2_original.CopyFrom(Debug2);
		
		Debug2.LUdecomp();
		Debug2.LUSolve(epsilon2,x);
		
		fp = fopen("Debug2_original.txt","w");
		for (i = 0; i < numEqns; i++)
		{
			for (j = 0; j < numEqns; j++)
				fprintf(fp,"%1.14E ",Debug2_original.LU[i][j].x[0]);
			fprintf(fp," #  %1.14E # %1.14E \n",x[i].x[0],epsilon2[i].x[0]);
		};
		fclose(fp);

		// Check what RSS we are getting at dd_real precision:

		dd_real RSS_ = 0.0;
		dd_real RSS_Gauss = 0.0;
		for (i = 0; i < numEqns; i++)
		{
			// Ax = b ? Take Ax-b
			eps = -epsilon2[i];
			for (j = 0; j < numEqns; j++)
			{
				eps += Debug2_original.LU[i][j]*x[j];
			}
			RSS_ += eps*eps;
			if ((i % 4 == 0) && ( i < numEqns-1))
				RSS_Gauss += eps*eps;
		}
		printf("dd_real RSS %1.14E Gauss %1.14E\n",RSS_.x[0],RSS_Gauss.x[0]);

		// Says that it did solve it.

		// Now let's test ??
		
		RSS_ = 0.0;
		RSS_Gauss = 0.0;
		for (i = 0; i < numEqns; i++)
		{
			// Ax = b ? Take Ax-b
			eps = -epsilon_original[i];
			for (j = 0; j < numEqns; j++)
			{
				eps += Original.LU[i][j]*x[j]; // Original was copied off before LS.
			}
			RSS_ += eps*eps;
			if ((i % 4 == 0) && ( i < numEqns-1))
				RSS_Gauss += eps*eps;
		}
		printf("From Original: RSS %1.14E Gauss %1.14E\n",RSS_.x[0],RSS_Gauss.x[0]);


		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			pAux->phi = x[iAux*4].x[0];  // - subtract;
			pAux->A.x = x[iAux*4+1].x[0];
			pAux->A.y = x[iAux*4+2].x[0];
			pAux->A.z = x[iAux*4+3].x[0];		
			
			++pAux;
		}
		PhiAnode_aux[iLevel] = x[numEqns-1];
		
		CalculateEpsilonAux4(iLevel);
		RSS = 0.0;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			RSS += pAux->epsilon[0]*pAux->epsilon[0];
			++pAux;
		}
		
		printf("Gauss dp RSS after Step 2 (no subtract): %1.14E \n",RSS);
		
		// Go again:
		// subtract A_ A from epsilon vector.
		// We have to do this and THEN do LS manipulation.
		
		for (i = 0; i < numAuxVertices[iLevel]; i++)
		{
			epsilon_reduced[i] = epsilon_original[i*4]; // not LS

			// reducing:
			for (j = 0; j < numEqns; j++) {
				if ((j % 4 != 0) || (j == numEqns-1))
					epsilon_reduced[i] -= Original.LU[i*4][j]*x[j]; // not LS, times given x
			};

			// reducing matrix that is already Gauss LS
			for (j = 0; j < numAuxVertices[iLevel]; j++)
				Debug2_small.LU[i][j] = Debug2_original.LU[i*4][j*4];
		};

		// LS manip:
		for (jRow = 0; jRow < numAuxVertices[iLevel]; jRow++)
		{
			epsilon2_small[jRow] = 0.0;
			for (i = 0; i < numAuxVertices[iLevel]; i++)
			{
				Beta_ij = Original.LU[i*4][jRow*4];
				epsilon2_small[jRow] += Beta_ij*epsilon_reduced[i];
			};
			// Are we doing something bad somewhere with reusing epsilon?
		};

		Debug2_small_original.CopyFrom(Debug2_small);

		Debug2_small.LUdecomp();
		Debug2_small.LUSolve(epsilon2_small,phi);

		// Assess errors:
		RSS_ = 0.0;
		for (i = 0; i < numAuxVertices[iLevel]; i++)
		{
			// Ax = b ? Take Ax-b
			eps = -epsilon2_small[i];
			for (j = 0; j < numAuxVertices[iLevel]; j++)
			{
				eps += Debug2_small_original.LU[i][j]*phi[j];
			}
			RSS_ += eps*eps;
		}
		printf("dd_real Debug2_small RSS %1.14E \n",RSS_.x[0]);
	
		FILE * jillio = fopen("debug2s.txt","w");
		for (i = 0; i < numAuxVertices[iLevel]; i++)
		{
			for (j = 0; j < numAuxVertices[iLevel]; j++)
				fprintf(jillio,"%1.14E ",Debug2_small_original.LU[i][j].x[0]);
			fprintf(jillio," # %1.14E # %1.14E # %1.14E \n",
				phi[i].x[0],epsilon2_small[i].x[0],epsilon_reduced[i].x[0]);
		};
		fclose(jillio);
		
		for (i = 0; i < numAuxVertices[iLevel]; i+=10)
		{
			printf("%d Before %1.5E ",i,x[i*4].x[0]);
			x[i*4] = phi[i];
			printf("After %1.5E \n ",phi[i].x[0]);
		};

		RSS_ = 0.0;
		RSS_Gauss = 0.0;
		fp = fopen("reduced_error.txt","w");
		for (i = 0; i < numEqns; i++)
		{
			// Ax = b ? Take Ax-b
			eps = -epsilon_original[i];
			for (j = 0; j < numEqns; j++)
			{
				eps += Original.LU[i][j]*x[j]; // Original was copied off before LS.
			}
			RSS_ += eps*eps;
			if ((i % 4 == 0) && ( i < numEqns-1))
				RSS_Gauss += eps*eps;
			fprintf(fp,"%1.14E %1.14E \n",x[i].x[0],eps.x[0]);
		}
		fclose(fp);
		printf("From Original: RSS %1.14E Gauss %1.14E\n",RSS_.x[0],RSS_Gauss.x[0]);

		getch();

		// Tidy up:

		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			pAux->phi = dbg[iAux*4].x[0];  // - subtract;
			pAux->A.x = dbg[iAux*4+1].x[0];
			pAux->A.y = dbg[iAux*4+2].x[0];
			pAux->A.z = dbg[iAux*4+3].x[0];		
			++pAux;
		}
		PhiAnode_aux[iLevel] = dbg[numEqns-1];
	};
	
	// Which of these steps fails to find an exact solution?
	
	delete[] epsilon;
	delete[] x;
	if (bNeumannPhi)
	{
		delete[] epsilon_original;
		delete[] epsilon1;
		delete[] epsilon2;
		delete[] epsilon2_small;
		delete[] phi;
	}
}


void TriMesh::RunLU_Az(int const iLevel, bool bRefreshCoeff)
{
	static Matrix eqns; // presumably static variable in method is like element of object ??
	// 300 x 300 matrix => 90000 x 8 bytes = 720kB
	// That said - object does actually contain Matrix Coarsest which we do not use apparently.

	qd_or_d * epsilon_original;
	qd_or_d * epsilon1, *phi, * epsilon2, * epsilon2_small, *epsilon_reduced;

	dd_real eps;
		
	int jRow, j;
	qd_or_d Beta_ij;
	Vertex * pAux, *pAux2;
	long iAux, i, iVar, iEqn;
	real tempval;
	char buffer[245];
	static int counter = 0;
	FILE * fp2, * fp;
	real RSS;
	long iVertex;
	int const NUMBER_OVERDETERMINED_START = numAuxVertices[iLevel]; 
	// out of about 75... 25 per cm sq => 6 in a row at back??
	
	counter++;
	static Matrix Original, Debug1, Debug2,Copy2,
		Debug2_small, Debug2_original, Debug2_small_original;	

	long const numEqns = numAuxVertices[iLevel] + 1;
	
	CalculateEpsilonAuxAz(iLevel);
	RSS = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		RSS +=	pAux->epsilon[3]*pAux->epsilon[3];
		++pAux;
	}
	RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	
	real old_RSS = RSS;
	
	printf("old RSS = %1.14E \n",old_RSS);

	if (bRefreshCoeff)
	{
		printf("creating LU ... ");
		// Fill in matrix 'eqns'
			
		// Gauss; Ampx; Ampy; Ampz; each cell
		// final one = Iz
		// So index is iAux*4+iVar
		
		if(eqns.Invoke(numEqns)) // does nothing if number of eqns unchanged.
		{
			printf("Memory allocation error eqns.Invoke \n");
		} else {
			if (Original.Invoke(numEqns))
				printf("Memory allocation error Original.Invoke \n");
			Debug1.Invoke(numEqns);
			Debug2.Invoke(numEqns);
			Copy2.Invoke(numEqns);
			Debug2_original.Invoke(numEqns);
			Debug2_small.Invoke(numAuxVertices[iLevel]);
			Debug2_small_original.Invoke(numAuxVertices[iLevel]);
		}
		
		//ZeroMemory(eqns.LU,sizeof(real)*numEqns*numEqns);
		// That assertion is not legit. And, it is now done as part of Invoke.
		
		long neigh_len, izNeigh[128];
		real coefflocal[4][4];
		int row;
		
		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			
			{
				int row = iAux;
				
				eqns.LU[row][iAux] = pAux->coeff_self[AZ][AZ];				
				// Consider the effects on this epsilon from chEz_aux, the final variable:
				eqns.LU[row][numEqns-1] = pAux->coeff_self[AZ][PHI_ANODE];
			};
			neigh_len = pAux->GetNeighIndexArray(izNeigh);
			for (i = 0; i < neigh_len; i++)
			{
				pAux->GetCoefficients(&(coefflocal[0][0]),i);
				{
					row = iAux;
					eqns.LU[row][izNeigh[i]] = coefflocal[AMPZ][AZ];					
				};
			};

			// not considered IZ eqn yet.
			// iVar also not yet counted unity, chEz.
			{
				row = iAux;
				if (eqns.LU[row][row] == 0.0) 
					{ eqns.LU[row][row] = 1.0; } // assume in this case that it's an unpopulated row & epsilon
			};

			// Now it remains to consider the effects on Iz from this point and its neighbours:
		
			eqns.LU[numEqns-1][iAux] = pAux->coeff_self[IZ][AZ];
			++pAux;
		}
		eqns.LU[numEqns-1][numEqns-1] = this->Epsilon_Iz_coeff_On_PhiAnode;

		Original.CopyFrom(eqns);	

		eqns.LUdecomp(); 		
		printf("Done refresh coeff + decomp\n");
	};

	qd_or_d * epsilon = new qd_or_d[numEqns];
	qd_or_d * x = new qd_or_d[numEqns];
	dd_real * dbg = new dd_real[numEqns];
	ZeroMemory(epsilon,sizeof(qd_or_d)*numEqns);

	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{
		// Trying to set it equal to the error that has to be cancelled.
		epsilon[iAux] = -pAux->coeff_self[AMPZ][UNITY];
		++pAux;
	}
	epsilon[numEqns-1] = -this->Epsilon_Iz_Default[iLevel]; //-this->Epsilon_Iz_aux[iLevel];
	
	eqns.LUSolve(epsilon, x);

	dd_real RSS_ = 0.0;
	for (i = 0; i < numEqns; i++)
	{
		// calculate error in original equation:
		eps = -epsilon[i];
		for (j = 0; j < numEqns; j++)
			eps += Original.LU[i][j]*x[j];
		RSS_ += eps*eps;
	};
	printf("RSS dd_real %1.14E \n",RSS_.x[0]);

	// -----------------------------------------------

	pAux = AuxX[iLevel];
	for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
	{		
		pAux->A.z = x[iAux].x[0];		
		++pAux;
	}
	PhiAnode_aux[iLevel] = x[numEqns-1];
		
	CalculateEpsilonAuxAz(iLevel);
	RSS = 0.0;
	real RSSGauss = 0.0;
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		RSS += pAux->epsilon[3]*pAux->epsilon[3];
		++pAux;
	}
	RSS += Epsilon_Iz_aux[iLevel]*Epsilon_Iz_aux[iLevel];
	printf("dp RSS after LU, no subtract done: %1.14E EpsIz %1.14E",RSS,Epsilon_Iz_aux[iLevel]);
	
	if (RSS > old_RSS)
	{
		printf("\n\n\n\n*\n*\n\aRSS > old_RSS: LU FAILED\n\n*\n*\n\nrun 128*JRLS instead:");
		// Put in a break here if we want to debug this.

		getch();

		// revert to the zero seed that was better, to do iterations:
		pAux = AuxX[iLevel];
		for (iAux = 0; iAux < numAuxVertices[iLevel]; iAux++)
		{
			pAux->phi = 0.0;
			memset(&(pAux->A),0,sizeof(Vector3));
			++pAux;
		};
		PhiAnode_aux[iLevel] = 0.0;
		IterationsJRLS_Az(iLevel, 128);
	};
	
	delete[] epsilon;
	delete[] x;
	
}
