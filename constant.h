
#ifndef constant_h
#define constant_h

#define real double


// decided to stick with this
// but move math.h functions out of header file
// into "constant.cpp"


real const zero = 0.0;
real const unity = 1.0;
real const two = 2.0;
real const twothirds = 2.0/3.0;
real const THIRTEENSIX = 13.6;
#define TWOTHIRDS  0.666666666666667
real const FOURTHIRDS = 4.0/3.0;
real const TWONINTHS = 2.0/9.0;
real const NINTH = 1.0/9.0;
#define THIRD     0.333333333333333

real const PI = 3.14159265358979323846;
real const PI2 = 2.0*PI;     
real const PI2inv = unity/PI2; 

real const sC_ = 2997924580.0;    // Coulomb to statCoulomb
real const kB_ = 1.60217646e-12;  // erg per eV

real const c_ = 29979245800.0 ;   // speed of light in vacuum (cm/s)
real const Z_ = unity ;           // number of charges in ion species
real const e_ = 4.8e-10  ;        // charge of electrons (statcoul)
real const q_ = 4.8e-10  ;        // ion charge [unsigned]
real const m_e_ = 9.10953e-28;    // electron mass in grams

// NOTE CHANGE : 3 x mass :

real const m_i_ = 3.0*3.67291e-24;    // deuteron mass in grams

real const m_ion_ = m_i_;
real const m_neutral_ = m_e_ + m_ion_;
real const m_n_ = m_neutral_;

#define m_i_over_m_n 0.999917333695   // Naughty - #define here yet not for m_e
#define m_e_over_m_n 8.26663049989847e-5
#define over_m_e 1.09775147565242e+27
#define TWO_PI_OVER_C 2.09584502195168e-10
#define M_PI 3.141592653589793

// To use #define we have to make sure the expression never appears within any other.
// Need a terminating & perhaps initial character such as C for const. 
// Or universally use CAPITALS for define, ensure no #define signature appears within another,
// and ensure that we only use smalls for variables.

//real const OVER_SQRT_2PI = 1.0/sqrt(two*PI);
// no sqrt in .h

real const PIOVER32 = PI/32.0;

real const eoverm_  = e_/m_e_;//5.26920708313162e+17 ;         // e/me, electron statcoul/gram
real const qoverM_  = q_/m_ion_;//130686567326725.0     ;        // q/mi, ion statcoul/gram
real const moverM_  = m_e_/m_ion_;//0.000248019417845795    ;      // electron to ion mass ratio
real const qoverm_  = q_/m_e_;

real const eovermc_ = eoverm_/c_;
real const qoverMc_ = qoverM_/c_;
real const FOUR_PI_Q_OVER_C_ = 4.0*PI*q_/c_;
real const FOUR_PI_Q_ = 4.0*PI*q_;
real const FOURPI_OVER_C_ = 4.0*PI/c_;
real const FOURPIOVERC_ = 4.0*PI/c_;
real const FOUR_PI_OVER_C_ = FOURPIOVERC_;

// Having done this, it will certainly cause issues when we are
// messing around with Az.

real const NU_EI_FACTOR_ = 1.0/(3.44e5);
//real const NU_II_FACTOR = 1.0/(sqrt(2.0)*2.09e7);
// sqrt in .h file == bad

real const nu_eiBarconst_ = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

#define NEUTRAL_KAPPA_FACTOR   10.0

#define ION_KAPPA_FACTOR = (20.0/9.0)
//; // 10/3 times 2/3
#define ELECTRON_KAPPA_FACTOR 2.6666666666667
// 5/2 times 2/3

#define ALPHA_ION        0.96
//; // this factor appears in the viscosity coefficient
#define ALPHA_e          0.73
#define ALPHA_ELECTRON   0.73

#define E0     13.6


real const cross_T_vals[10] = {0.1,0.501187,1.0,1.99526,3.16228,5.01187,7.94328,12.5893,19.9526,31.6228};

// momentum-transfer cross section data from http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
real const cross_s_vals_momtrans_ni[10] = {
	1.210e-14,1.020e-14,9.784e-15,9.076e-15,8.589e-15,8.115e-15,7.653e-15,7.207e-15,6.776e-15,6.351e-15};
// distinguishable particles:
	//4.408e-15,2.213e-15,1.666e-15,7.625e-16,4.685e-16,2.961e-16,1.878e-16,1.192e-16,7.442e-17,4.083e-17};

// viscosity cross section data from  http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
real const cross_s_vals_viscosity_ni[10] = {
	4.904e-15,3.023e-15,2.673e-15,1.891e-15,1.203e-15,7.582e-16,4.891e-16,3.185e-16,2.030e-16,1.223e-16};
// viscosity cross section data from http://www-cfadc.phy.ornl.gov/elastic/dd0/tel.html
real const cross_s_vals_viscosity_nn[10] = {
	1.753e-15,1.179e-15,9.030e-16,7.650e-16,6.316e-16,4.278e-16,2.685e-16,1.641e-16,9.609e-17,5.550e-17};

#endif
