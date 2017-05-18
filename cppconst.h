
#ifndef cppconst_h
#define cppconst_h

real const sC = 2997924580.0;    // Coulomb to statCoulomb
real const kB = 1.60217646e-12;  // erg per eV

real const c = 29979245800.0 ;   // speed of light in vacuum (cm/s)
real const Z = unity ;           // number of charges in ion species
real const e = 4.8e-10  ;        // charge of electrons (statcoul)
real const q = 4.8e-10  ;        // ion charge [unsigned]
real const m_e = 9.10953e-28;    // electron mass in grams

// NOTE CHANGE : 3 x mass :

real const m_i = 3.0*3.67291e-24;    // deuteron mass in grams

real const m_ion = m_i;
real const m_neutral = m_e + m_ion;
real const m_n = m_neutral;

real const eoverm  = e/m_e;//5.26920708313162e+17 ;         // e/me, electron statcoul/gram
real const qoverM  = q/m_ion;//130686567326725.0     ;        // q/mi, ion statcoul/gram
real const moverM  = m_e/m_ion;//0.000248019417845795    ;      // electron to ion mass ratio
real const qoverm  = q/m_e;

real const eovermc = eoverm/c;
real const qoverMc = qoverM/c;
real const FOUR_PI_Q_OVER_C = 4.0*PI*q/c;
real const FOUR_PI_Q = 4.0*PI*q;
real const FOURPI_OVER_C = 4.0*PI/c;
real const FOURPIOVERC = 4.0*PI/c;
real const FOUR_PI_OVER_C = FOURPIOVERC;

// Having done this, it will certainly cause issues when we are
// messing around with Az.

real const NU_EI_FACTOR = 1.0/(3.44e5);
//real const NU_II_FACTOR = 1.0/(sqrt(2.0)*2.09e7);
// sqrt in .h file == bad

real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

#endif
