#ifndef FLAGS_H
#define FLAGS_H

char const ANTICLOCKWISE = -1;
char const CLOCKWISE = 1;

#define NEEDS_ANTI       1
#define NEEDS_CLOCK      -1

int const DOMAIN_VERTEX = 0;
int const DOMAIN_TRIANGLE = 0;
int const DOMAIN_MINOR = 0;

int const CROSSING_INS = 1;

int const INNER_VERTEX = 3;
int const OUT_OF_DOMAIN = 3;

// note that "< 4" is a test that appears hardcoded sometimes
int const INNERMOST_EDGE = 4;
int const INNERMOST = 4;
int const CONCAVE_EDGE_VERTEX = 4;

int const CONVEX_EDGE_VERTEX = 5;
int const OUTERMOST = 5;
int const IMMOVABLE_OUTER = 5;

int const REVERSE_JZ_TRI = 6;

int const INNER_FRILL = 7;
int const OUTER_FRILL = 8;

// Note that we should find we basically never hit the insulator
// As that would entail compressing the material too much
// between the insulator and the vertex.
// However we can populate a full tri mesh including tris
// that cross the insulator.
// We then put the vertex-centred cell corners along the insulator.

// The following does not really belong in this file:

// equations for solver:
int const NUMCELLEQNS = 4; // Leave this at 4 but it really is 1
int const GAUSS = 0, AMPX = 1, AMPY = 2, AMPZ = 3, IZ = 4;
// affector index:
int const PHI = 0, AX = 1, AY = 2, AZ = 3, UNITY = 4, PHI_ANODE = 5;
int const NUMREGRESS = 2;
int const JACOBI_REGR = 0, RICHARDSON_REGR = 1;
int const GAMMANUM = NUMREGRESS*NUMCELLEQNS+1;  // ??
int const EZTUNING = 5;

#define MASS_ONLY   0
#define ALL_VARS    1

#define ELEC_X   4
#define ELEC_Y   5
#define ION_X    2
#define ION_Y    3
#define NEUT_X   0
#define NEUT_Y   1

#define DEPENDS_ON_SELF      0
#define DEPENDS_ON_NONE      0
#define DEPENDS_ON_DIRECT    1
#define DEPENDS_ON_INDIRECT  2

	int const KAPPA_ION = 0;
	int const KAPPA_E = 1;
	int const KAPPA_NEUT = 2;
	int const NU_IHEART = 3;
	int const NU_EHEART = 4;
	int const NU_NHEART = 5;

int const FEINT = 1;
int const REAL = 2;

#define SPECIES_NEUT         0
#define SPECIES_NEUTRAL      0
#define SPECIES_ION          1
#define SPECIES_ELECTRON     2
#define SPECIES_ELEC         2
#define OVERALL              8


// graph codes (not pruned) :

#define JZAZBXYEZ                        4
#define JXYAXYBZEXY               5
#define JXYAXYBZ_FIXED            6
#define SIGMA_E_J                 7
#define JXY_A2XY_RHOXY   8
#define EXY_RHO_PHI_JXY		9
#define VI_VE								10
#define GRAPH_PHI		11
#define GRAPH_EXY		12
#define GRAPH_NONE		13
#define GRAPH_PHI_BIRTHRATE		14
#define GRAPH_RHO			15
#define GRAPH_RHO_OLD			16
#define GRAPH_EPSILON   17
#define GRAPH_EPS_1D   17
#define GRAPH_EPS0		17
#define GRAPH_vexy        18
#define GRAPH_JXY           19
#define GRAPH_EPSILON_Z  20
#define GRAPH_EPS_Z  20
#define GRAPH_EPS3   20
#define GRAPH_AXY  21
#define GRAPH_AZ	22
#define GRAPH_BXY	23
#define GRAPH_BZ	24
#define GRAPH_JZ     25
#define GRAPH_FLAT_WIRE_MESH 26
#define GRAPH_EPSILON_XY   27
#define GRAPH_EPS_XY   27
#define GRAPH_EPS12    27
#define GRAPH_NEUTRAL_N 28
#define GRAPH_NEUTRAL_V 29
#define GRAPH_NEUTRAL_T 30
#define GRAPH_NEUT_N 28
#define GRAPH_NEUT_V 29
#define GRAPH_NEUT_T 30
#define GRAPH_ION_N 31
#define GRAPH_ION_T 32
#define GRAPH_ION_V 33
#define GRAPH_ELECTRON_N 34
#define GRAPH_ELECTRON_T 35
#define GRAPH_ELECTRON_V 36
#define GRAPH_ELEC_N 34
#define GRAPH_ELEC_T 35
#define GRAPH_ELEC_V 36
#define GRAPH_VEZ 37
#define GRAPH_TESTCURLBZ 38
#define GRAPH_TESTCURLBXY 39
#define JZAZBXYTEST                        40
#define JXYAXYBZTEST                        41
#define TOTAL                               42
#define GRAPH_TOTAL_N							43
#define GRAPH_TOTAL_V							44
#define GRAPH_TOTAL_T							45
#define GRAPH_TOTAL_N_II						46
#define GRAPH_EZ							47
#define SIGMA_VEZ_J_E					49
#define GRAPH_E							50
#define GRAPH_J							51
#define GRAPH_SIGMA						52
#define GRAPH_REG						53
#define GRAPH_REG_XY					54
#define GRAPH_REG2    55
#define GRAPH_REG2_XY 56
#define GRAPH_AX      57
#define GRAPH_AY      58
#define GRAPH_EPS_X   59
#define GRAPH_EPS1    59
#define GRAPH_EPSILON_X  59
#define GRAPH_EPS_Y      60
#define GRAPH_EPS2    60
#define GRAPH_EPSILON_Y  60
#define GRAPH_VE0		61
#define GRAPH_VI0		62
#define GRAPH_JZ_TEMP   63
#define GRAPH_JXY_TEMP  64
#define GRAPH_JX_TEMP   65
#define GRAPH_JY_TEMP   66
#define GRAPH_RHO_STORED 67
#define GRAPH_MAGACCEL 68
#define GRAPH_BETA     69

#define GRAPH_ACCEL_TOTAL 70
#define GRAPH_ACCEL_ION  71
#define GRAPH_ML_TOTAL  72
#define GRAPH_ML_ION   73
#define ACCELS 74

#define OHM  75
#define JCOMPONENTS 76
#define GRAPH_VIZ   77
#define GRAPH_AIZ   78
#define GRAPH_SIGMA_E 79
#define GRAPH_VE0Z   80
#define VE0			81
#define GRAPH_SIGMATEMP 82


// Number of vars used to produce heights:
int const FLAG_FLAT_MESH = 0;
int const FLAG_DATA_HEIGHT = 1;
int const FLAG_VELOCITY_HEIGHT = 2;
int const FLAG_VEC3_HEIGHT = 3;

int const FLAT_MESH = 0;
int const DATA_HEIGHT = 1;
int const VELOCITY_HEIGHT = 2;
int const VEC3_HEIGHT = 3;

int const FLAG_COLOUR_MESH = 0;
int const FLAG_SEGUE_COLOUR = 1;
int const FLAG_VELOCITY_COLOUR = 2;
int const FLAG_CURRENT_COLOUR = 3;
int const FLAG_AZSEGUE_COLOUR = 4;
int const FLAG_IONISE_COLOUR = 5;

int const COLOUR_MESH = 0;
int const SEGUE_COLOUR = 1;
int const VELOCITY_COLOUR = 2;
int const CURRENT_COLOUR = 3;
int const AZSEGUE_COLOUR = 4;
int const IONISE_COLOUR = 5;
#endif