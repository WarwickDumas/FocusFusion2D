#include "FFxtubes.h"
#include "bandlu.h" // must not include a cpp file here!

#include <conio.h>
// no - don't need to be calling getch() in a header file !

#include "flags.h"
#include "vector_tensor.cu"
#include <windows.h> // contains max, min macros

#include <d3dx9.h>
//#include <dinput.h>
#include <dxerr.h>
#include "d3d.h"

#include "cuda_struct.h"

#ifndef mesh_h
#define mesh_h

#define CP_MAX  64
#define FLAG_CODE_LEFT    1
#define FLAG_CODE_RIGHT  2
#define FLAG_CODE_BOTTOM 3
#define FLAG_CODE_TOP  4
#define FLAG_CODE_LEFTTOP 5
#define FLAG_CODE_LEFTBOT 6
#define FLAG_CODE_RIGHTTOP 5
#define FLAG_CODE_RIGHTBOT 6


// DO NOT WANT THE SMART ARRAY CLASSES IN NVCC.
// MOVED HERE.

class smartreal
{
public:
	static const int ALLOC = 8;

	real * ptr;
	short len, alloclen;

	smartreal();

	void clear();
	
	int ReDim(int length);

	void add(real x);

	~smartreal();
};

class smartlong
{
public:
	static const int ALLOC = 8;

	long * ptr;
	short len, alloclen;
	// experiment with carets:
//	long * ptrlast;

	smartlong();

	void clear();
	void remove_if_exists(long what);

	void remove(long what);

	void IncreaseDim();
	void add(long what);

	void add_at_element(long what,long iInsert);

	void copyfrom(smartlong & src);

	bool contains(long what);

	long FindIndex(long what);
	void add_unique(long what);

	void remove_element( long iWhich );
	int remove_elements( long iStart, long iHowmany);
	~smartlong();
};



class Proto {
public:
	// empty class
};


class ConvexPolygon
{
public:
	// This seems no good now.
	// We need a dynamic array.

	Vector2 coord[CP_MAX]; // change according to max desired
	int status[CP_MAX];
	int numCoords;

	ConvexPolygon(const Vector2 & x1,const Vector2 & x2,const Vector2 & x3);
	ConvexPolygon();
	
	void SetTri(const Vector2 & x1,const Vector2 & x2, const Vector2 & x3);
	void CreateClockwiseImage(const ConvexPolygon & cpSrc) ;
	void CreateAnticlockwiseImage(const ConvexPolygon & cpSrc);
	
	void inline ConvexPolygon::Clear()
	{
		numCoords = 0;
	}

	void inline ConvexPolygon::add(real x,real y)
	{
		numCoords++;
		coord[numCoords-1].x = x;
		coord[numCoords-1].y = y;
	}

	void inline ConvexPolygon::add(Vector2 & u)
	{
		numCoords++;
		coord[numCoords-1] = u;
	}

	bool ClipAgainstHalfplane(const Vector2 & r1, const Vector2 & r2, const Vector2 & r3);
	void CopyFrom(ConvexPolygon & cp);
	real FindTriangleIntersectionArea(Vector2 & r1, Vector2 & r2, Vector2 & r3);
	real FindQuadrilateralIntersectionArea(Vector2 & r1, Vector2 & r2, Vector2 & r3, Vector2 & r4);
	real GetArea();
	void ConvexPolygon::GetCentre(Vector2 & centre);
	bool ConvexPolygon::GetIntersectionWithTriangle(ConvexPolygon * pPoly,Vector2 & r1, Vector2 & r2, Vector2 & r3);
	bool ConvexPolygon::GetIntersectionWithPolygon(ConvexPolygon * pPoly, ConvexPolygon * pClip);
	
	void ConvexPolygon::Integrate_Planes(Vector2 & r1, Vector2 & r2, Vector2 & r3,
										real yvals1[],
										real yvals2[],
										real yvals3[],	
										real results[],
										long N_planes);
	void ConvexPolygon::IntegrateMass(Vector2 & r1, Vector2 & r2, Vector2 & r3,
									real yvals1, real yvals2, real yvals3, real * pResult);
	//real GetSideLength(int side);
	real GetPrecedingSideLength(int side);
	real GetSucceedingSideLength(int side);
	
	Vector3 ConvexPolygon::Get_curl2D_from_anticlockwise_array(Vector3 A[]);
	Vector2 ConvexPolygon::Get_grad_from_anticlockwise_array(real Te[]);
	Vector2 ConvexPolygon::Get_Integral_grad_from_anticlockwise_array(real Te[]);

	void ConvexPolygon::Get_Bxy_From_Az(real Az_array[], real * pBx,real * pBy);
	
	Vector2 ConvexPolygon::CalculateBarycenter();
	real ConvexPolygon::minmod(real n[], // output array
					  real ndesire[], real N, 
					  Vector2 central );
};





class fluidnvT
{
public:
	real n,T;
	Vector3 v;

	void Interpolate ( fluidnvT * pvv1,  fluidnvT * pvv2,
							Vector2 & pos1, Vector2 & pos2, Vector2 & ourpos);
};
// Can't think of a good reason we want that.


class macroscopic
{
public:
	real mass, heat;
	Vector3 mom;   
	// store macroscopic conserved quantities for doing triangle collisions
	friend macroscopic operator* (const real hh,const macroscopic &vars);
};



/*class AuxVertex // for inner mesh and coarser levels
{
public:
/*	real x,y;

	Vector3 Temp; // always 0 except for Jz where the reverse current flows
	// ?
	
	real epsilon[NUMCELLEQNS]; // store epsilon for 7 eqns
	
	real phi;
	Vector3 A, v_e; // for phi solver.


	// Note that this is to be used in fixed equilateral meshes.
	long iNeighbours[MAXNEIGH]; 
		// What is iNeighbours used for?
		// Search for location of point from finer level maybe.
		// No, that is triangles I'd think.
	long iTriangles[MAXNEIGH];
	long iCoarseTriangle;	 // index into the coarser mesh above for doing multimesh; also use as scratch if need be.
	real weight[3];

	long iVolley; 

	long iIndicator; // for prosecuting searches

	short tri_len, neigh_len;
	short flags; 

	//real coefficients[8][7][MAXNEIGH]; 
	
	Coefficientry co;
	real coeff_self[NUMCELLEQNSPLUS1][NUMCELLEQNSPLUS2];


	//Coefficientry co2, co3, co4;
	//real coeff_self2[NUMCELLEQNSPLUS1][NUMCELLEQNSPLUS2];
	//real coeff_self4[NUMCELLEQNSPLUS1][NUMCELLEQNSPLUS2];


	real regressor[NUMREGRESS][NUMCELLEQNS];    // add to phi, A, v_e
	Vector3 extra_regressor; // add to chEz

	real contrib_to_Az_avg, contrib_to_phi_avg;

	// eqn 8 is IZ and represents effect on Iz eqn rather than its own epsilon
//	smartreal coeff_extra;
//	smartlong index_extra;

	// We now store all coefficient information in a 
	// Coefficientry object. Except coeff_self.

	real sum_eps_beta[NUMCELLEQNS];
	real sum_beta_sq[NUMCELLEQNS]; // one for phi,Ax,Ay,Az


	// Probably not used:
	unsigned char has_periodic_interval;   // does its interval for joining domain mesh, cross PB?
	unsigned char  has_periodic;	  // whether connects periodically to any vertices - inner or domain
	unsigned char  look_out_for_periodic; // if yes, then if x < 0 we rotate anticlockwise if cont is to right of 2pi/32 clockwise from this point; if x< 0, vice versa.
	real gradient;

	// Remember there are relatively few of the coarse vertices.
	// If this prevents parallelisation, it's small comfort however.


	// DEBUG:

	//Tensor2 coeff_debug[320];
	//Tensor2 coeff_debug2[320];
	//long index_debug[320];
	//long index_debug2[320];
	//long debug_len;
	//long debug_neighlen;
	//Vector2 debug_implied_eps,debug_implied_eps2,debug_implied_eps3;
	

	// in the case of a coarse mesh, coeff_extra is used for those
	// contributions that are from non-neighbours in coarse mesh
	// and those that have to be rotated. index_extra signals
	// the rotation of the contribution.
	
	real scratch;
	real SSweights;

	// need some storage for coefficients applying to domain vertices:
	// 
		


	// eps = Temp + coeff_self * Aself + coefficients . A values
	// new A += -eps/coeff_self
	//
	// or just, with another schema,
	// new A value = coefficients . A values + Temp   
	//

	AuxVertex();

	void addtri(long iTri);

	void remove_tri(long iTri);

	void add_neigh(long iNeigh);

	int add_neigh_unique(long iNeigh);

	void PopulatePosition(Vector2 & result);

	void AuxVertex::periodic_image(Vector2 & result, int side, int do_it);

	int Save(FILE * fp, void * pTriArray2);
	
	int Load(FILE * fp, void * pTriArray2);
	*/

//};

/*
	AuxVertex * cornerptr[3]; // store indices so that we can relate to either
	// inner vertex or to an actual vertex.
	
	AuxTriangle * neighbours[3]; 
	int periodic; // number of points that are Clockwise wrapped relative to the others
	// note that BYTE == unsigned char and creates problems with testing for decrement below zero
	BYTE flags; 
	Vector2 cc;
	real area; 
	Vector2 edge_normal[3];

	void SetTriangleVertex(int which, AuxVertex * pInner);
	void AuxTriangle::ReturnCentre(Vector2 * p_cc, AuxVertex * pVertex);

	void AuxTriangle::Set(AuxVertex * p1, AuxVertex * p2, AuxVertex * p3);
	void AuxTriangle::Set(AuxVertex * p1, AuxVertex * p2, AuxVertex * p3, long iTri);
	void AuxTriangle::Reset(AuxVertex * p1, AuxVertex * p2, AuxVertex * p3, long iTri);

	void AuxTriangle::GuessPeriodic(void);
	void AuxTriangle::CalculateCircumcenter(Vector2 & cc, real * pdistsq);

	bool inline AuxTriangle::has_vertex(AuxVertex * pVertex);
	void AuxTriangle::MapLeft(Vector2 & u0, Vector2 & u1, Vector2 & u2);

	bool AuxTriangle::ContainsPoint(real x, real y);
	int AuxTriangle::TestAgainstEdge(real x,real y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  AuxTriangle ** ppNeigh);

	AuxVertex * AuxTriangle::ReturnUnsharedVertex(AuxTriangle * pTri2, int * pwhich = 0);

	bool AuxTriangle::TestDelaunay(AuxVertex * pAux);

	AuxTriangle::AuxTriangle();
	AuxTriangle::~AuxTriangle();

	void AuxTriangle::PopulatePositions(Vector2 & u0, Vector2 & u1, Vector2 & u2);

	int AuxTriangle::GetLeftmostIndex();
	int AuxTriangle::GetRightmostIndex();

	void AuxTriangle::ReturnCircumcenter(Vector2 & u, AuxVertex * pVertex);

	void AuxTriangle::RecalculateEdgeNormalVectors(bool normalise);
	*/


struct fluid_NvT
{
	real N[3];
	Vector3 Nv[3];
	real NT[3]; 
};

struct fluid_nvT
{
	real n[3];
	Vector3 nv[3];
	real nT[3]; 
	// heat density -> nT/n is temp density
	// but heat density is better since we want
	// usually to integrate and contribute NT.
	void inline Interpolate(real beta[3],
							fluid_nvT Z[3])
							// 3 corners
	{
		n[0] = beta[0]*Z[0].n[0] + 
			   beta[1]*Z[1].n[0] +
			   beta[2]*Z[2].n[0];
		n[1] = beta[0]*Z[0].n[1] + 
			   beta[1]*Z[1].n[1] +
			   beta[2]*Z[2].n[1];
		n[2] = beta[0]*Z[0].n[2] + 
			   beta[1]*Z[1].n[2] +
			   beta[2]*Z[2].n[2];
		nv[0] = beta[0]*Z[0].nv[0] +
			    beta[1]*Z[1].nv[0] +
				beta[2]*Z[2].nv[0];
		nv[1] = beta[0]*Z[0].nv[1] +
				beta[1]*Z[1].nv[1] +
				beta[2]*Z[2].nv[1];
		nv[2] = beta[0]*Z[0].nv[2] +
				beta[1]*Z[1].nv[2] +
				beta[2]*Z[2].nv[2];
		nT[0] = beta[0]*Z[0].nT[0] +
				beta[1]*Z[1].nT[0] +
				beta[2]*Z[2].nT[0];
		nT[1] = beta[0]*Z[0].nT[1] +
				beta[1]*Z[1].nT[1] +
				beta[2]*Z[2].nT[1];
		nT[2] = beta[0]*Z[0].nT[2] +
				beta[1]*Z[1].nT[2] +
				beta[2]*Z[2].nT[2];
	}
	fluid_nvT Clockwise() const;
	fluid_nvT Anticlockwise() const;
	
};


//int const MAXNEIGH = 36; // Allow that each level may increase network connection depth.
// Keep neigh, coeff handling OO for the Vertex class so that we can change to heap if/when needed.
int const MAXCOARSE = 8; // Generally should be 2 or 1
// Perhaps we should restrict to 6.
int const AUXNEIGHMAX = 14; // geometric neighbours list
// We need to find out why it is finding 14. :-(


struct ShardData
{
	fluid_nvT fluidnvT[MAXNEIGH];	
	ConvexPolygon cp;
	// Coords must match the values array.
	Vector2 central;
	fluid_nvT cdata;
	long len;
	// ConvexPolygon has 64 coords static, at present,
	// so be really careful about dimensioning a lot
	// of these objects.
};

int const NUM_EQNS_1 = 4; // 1 less because Iz eqn not affected by neighbours
int const NUM_AFFECTORS_1 = 4; // A,phi in neighbours
int const NUM_EQNS_2 = 5;   // + Iz eqn
int const NUM_AFFECTORS_2 = 6; // + UNITY, chEzExt

struct Coeff_array
{
	real co[NUM_EQNS_1][NUM_AFFECTORS_1]; 
	// Think we will have to switch to dynamic allocation.	
};

class Vertex
{
private:
	
	// Moved neighbour data to private because I do not want them accessed
	// directly.
	// We need a way to access that is consistent: does not need to be
	// altered whenever a change is made to the nature of storage;
	// and it makes sense to populate a local static array of integers,
	// both for debugging and hopefully for speed.

	long izTri[MAXNEIGH];
	long izNeigh[MAXNEIGH]; // index the neighbouring vertices
	long tri_len,neigh_len;
	long Auxneigh_len, izNeighAux[AUXNEIGHMAX];
	// Do OO coefficient handling so that we can adapt later if necessary.
	// memcpy sends Coeff_array contents to a local object when we want to use fast.

public:
	
	Vector2 pos; 
	macroscopic Ion,Elec,Neut; // mass, heat, mom
	Vector3 A, E, B, Adot; // It's important to have a stored estimate of Adot.
	// This is either the estimate at the same timeslice or at half a timeslice before (empirically)
	real phi; 

	// Let's think about this - do we need a special A_k storage in order to create that information? Economise.

	long flags;	// Changed to long because of the way it is used by multigrid routine. 
	bool has_periodic; 

	// Here was the storage data. 
	// ---------------------------

	// To recalculate:

	real AreaCell; // Area of Voronoi, or QV, or whatever used.
	Vector2 GradTe;

	// scratch data:
	real n,T;
	Vector3 v, Temp; 
	Vector2 temp2, centroid;

	// other:
	Vector2 a_pressure_ion, a_pressure_neut_or_overall, a_pressure_elec; 
	// adding for now. See later if we can get rid of. 
	
	long iVolley; // also used for showing if it is selected to submesh 
	long iScratch; // can be used to index coarse vertex above in submesh
	long iIndicator; // used in searches
	// better look again at automatic submesh routine: how it worked?

	Vector2 AdvectedPosition0; // or Displacement_default
	Vector2 AdvectedPosition; // for solution of move. *We could just use pos in dest mesh. But means changing some code.*
	Vector2 xdot, xdotdot; // just temporary - should find suitable scratch data
	Vector2 ApplDisp;

	// Not clear yet if we will need for Stage III:
	Vector2 PressureMomAdditionRate[3]; // in terms of particle mass ; try to reuse smth else

	// Ohm's Law:
	Vector3 v_e_0, v_i_0, v_n_0;
	Tensor3 sigma_e, sigma_i, sigma_n;
	// We assume that v_e adapts instantaneously, but the other two "laws" are for stepping to t_k+1
	// so that we can do the species relative displacement ready for Stage III.
	// Maybe we can find a way to economise on all this memory use. Remember though there are 100 coefficients. 6 x 4 x 4 = 96.

	// +36 doubles
	
	// ODE solving:
	// _____________

	real epsilon[NUM_EQNS_2];
	real coeff_self[NUM_EQNS_2][NUM_AFFECTORS_2];
	//real coeff[MAXNEIGH][NUM_EQNS_1][NUM_AFFECTORS_1]; // we can collate effects on Iz via neighs...

	// 16 x 4 x 4 = 256 doubles -> 2048 kB
	Coeff_array coeff[MAXNEIGH];
	// Probably want a dynamic array for the 1st [].
	// Either we have just a flatpack of reals (which could be OK),
	// or, use struct with [NUM_EQNS_1][NUM_AFFECTORS_1].
	// Neither of these allows us to access just as pVertex->coeff[1][0][0].
	
	real regressor[2][4]; 
	real circuit_regressor;
	// Here, another 5 + 30 + 8 = 43 doubles.

	// the following look to be unused (JRLSX only?):
	//real sum_eps_beta[4];
	//real sum_beta_sq[4]; // one for phi,Ax,Ay,Az   // ???
	
	// Multimesh:
	//real weight[3]; // 3 corners of supertriangle for which this vertex has a weight.
	//long iCoarseTriangle; // need to know which coarse vertex is nearest, ie which Voronoi cell inhabited
	
	long iCoarseIndex[MAXCOARSE]; // usually should be 1,2,3 coarse_len in ratio 1:4:1.

	real weight[3];
#ifdef FUNKYWEIGHTS
	real wt_var[MAXCOARSE][4]; // maybe one day need matrix; get vector for now.
	real wt_eps[MAXCOARSE][4][4]; // recruit to eps_eqn0 from all eqns.
#endif

	char PBC_uplink[MAXCOARSE]; // status of coarser point relative to our point
	int coarse_len; // how many affectors & affected at coarser level.
	
	char PBC[MAXNEIGH]; // Only intending to populate for auxiliary.
	// NOTE it is orientation of THIS POINT relative to neighbour
	
	char iLevel; // what level is this vertex on?
	
	real contrib_to_phi_avg; // so that phi level can be changed on higher levels.
	// Is this needed?
	// Each phi += [lc with total 1]*phi
	// So if we change every coarse phi equally we are changing phi's level.
	// ??
	// Yet adding to 1 of them a bit more than another could mean
	// that we added +1 to 5 phi's fine and -1 to 3 phi's fine.
	// ??
	// phi_avg probably best put only in beginning row --
	// so it's moot.
	
	real Ez_coeff_on_phi, Ez_coeff_on_phi_anode;
	
	// temp:
	real ddphidzdz_coeff_on_phi,ddphidzdz_coeff_on_phi_anode;
	// DEBUG:
	real predict_eps[NUM_EQNS_1]; 
	
	
	// I think we may need to now rethink coeff[MAXNEIGH] given that we are doing coefficients for
	// vertex object.
	
	// **: Can we inherit an object that has more coeff defined? 

	// Heap coefficients:
	// Finest ClearCoefficients should also redimension them according to neigh_len;
	// for auxiliary we use a different way to clear and add coefficients.



	inline Vertex() {
		iLevel = -1;
		this->ClearTris();
		this->ClearNeighs();
		this->ClearCoarseIndexList();
		ClearAuxNeighs();
		memset(&(sigma_i),0,sizeof(Tensor3));
		memset(&(sigma_e),0,sizeof(Tensor3));
		memset(&(sigma_n),0,sizeof(Tensor3));
	}

	long inline GiveMeAnIndex() const
	{
		return izTri[0];
	}
	long inline GetCoarseIndex(int i)
	{
		return iCoarseIndex[i];
	}
		
	void inline Vertex::CopyDataFrom(Vertex * pSrc)
	{
		A = pSrc->A;
		this->Adot = pSrc->Adot;
		this->B = pSrc->B;
		this->E = pSrc->E;
		this->Ion = pSrc->Ion;
		this->Neut = pSrc->Neut;
		this->Elec = pSrc->Elec;
		this->phi = pSrc->phi;
		this->pos = pSrc->pos;
		// can same class access object's private data?
	}

	void inline Vertex::CopyLists(Vertex * pSrc)
	{
		flags = pSrc->flags;
		has_periodic = pSrc->has_periodic;
		memcpy(izTri,pSrc->izTri,sizeof(long)*MAXNEIGH);
		memcpy(izNeigh,pSrc->izNeigh,sizeof(long)*MAXNEIGH);
		tri_len = pSrc->tri_len;
		neigh_len = pSrc->neigh_len;
		// can same class access object's private data?
	}

	long inline Vertex::GetNeighIndexArray(long * arr)
	{
		memcpy(arr,izNeigh,sizeof(long)*neigh_len);

		return neigh_len;
	}

	long inline Vertex::GetAuxNeighIndexArray(long * arr)
	{
		memcpy(arr,izNeighAux,sizeof(long)*Auxneigh_len);

		return Auxneigh_len;
	}

	void inline Vertex::SetNeighIndexArray(long * arr, long len)
	{
		if (len > MAXNEIGH) {
			printf("neighs > MAXNEIGH. [1] \n");
			getch();
			return;
		};
		memcpy(izNeigh,arr,sizeof(long)*len);
		neigh_len = len;
	} // never called?

	void inline Vertex::GetCoefficients(real * coefflocal, int iNeigh) // ?
	{
		if (iNeigh >= neigh_len) {
			printf("error iNeigh >= len Vertex::GetCoefficients\n");
			return;
		}
		memcpy(coefflocal, &(coeff[iNeigh].co[0][0]), sizeof(real)*NUM_AFFECTORS_1*NUM_EQNS_1);
	}

	long inline Vertex::GetTriIndexArray(long * arr)
	{
		memcpy(arr,izTri, sizeof(long)*tri_len);

		return tri_len;
	}
	void inline Vertex::SetTriIndexArray(long * arr, long len)
	{
		if (len > MAXNEIGH) {
			printf("tris > MAXNEIGH. [1]");
			getch();
			return;
		}
		memcpy(izTri,arr, sizeof(long)*len);
		tri_len = len;
	}
	void inline ClearTris()
	{
		tri_len = 0;
	}
	void inline ClearNeighs()
	{
		neigh_len = 0;
		ZeroCoefficients();
	}
	void inline ClearAuxNeighs()
	{
		Auxneigh_len = 0;
	}

	void inline ZeroCoefficients()
	{
		memset(coeff_self,0,NUM_EQNS_2*NUM_AFFECTORS_2*sizeof(real));
		for (int i = 0; i < MAXNEIGH; i++)
		{
			memset(&(coeff[i].co[0][0]),0,sizeof(real)*NUM_EQNS_1*NUM_AFFECTORS_1);
		}
	}
	void inline ClearCoarseIndexList()
	{
		coarse_len = 0;
	}
	void inline AddUniqueCoarse(long iVertex)
	{
		int i;
		if (coarse_len < MAXCOARSE) {
			for (i = 0; i < coarse_len; i++)
				if (iCoarseIndex[i] == iVertex) return;
		} else {
			printf("more than MAXCOARSE coarse indices added.\n");
			getch();
		}
		iCoarseIndex[coarse_len] = iVertex;
		coarse_len++;
	}
	bool inline Vertex::RemoveCoarseIndexIfExists(long index)
	{
		int i = 0;
		while ((i < coarse_len) && (iCoarseIndex[i] != index)) ++i;
		if (i == coarse_len) return false;
		coarse_len--;
		memmove(iCoarseIndex+i,iCoarseIndex+i+1,sizeof(long)*(coarse_len-i));
		return true;
	}
	void inline Vertex::AddToCoefficients(int iNeigh,real coeff_addition[4][4])
	{
		int const maxi = NUM_EQNS_1*NUM_AFFECTORS_1;
		real *pf64 = &(coeff_addition[0][0]);
		
		if (iNeigh == -1) {
			// add to coeff_self
			real *ptr;
			for (int iEqn = 0; iEqn < NUM_EQNS_1; iEqn++)
			{
				ptr = &(coeff_self[iEqn][0]);
				for (int i = 0; i < NUM_AFFECTORS_1; i++)
				{
					*ptr += *pf64;
					++ptr;
					++pf64;
				};
			};
		} else {
			real *ptr = &(coeff[iNeigh].co[0][0]);
			real *pf64 = &(coeff_addition[0][0]);
			for (int i = 0; i < maxi; i++)
			{
				*ptr += *pf64;
				++ptr;
				++pf64;
			};
		};
	}

	int Vertex::AddNeighbourIfNecessaryAux (long iNeigh)
	{
		// this is for adding to the 'geometrically connected' array for auxiliary vertices.
		int i;
		for (i = 0; i < Auxneigh_len; i++)
		{
			if (izNeighAux[i] == iNeigh) return i;
		};
		if (Auxneigh_len == AUXNEIGHMAX) {
			printf("too many aux neighs\n");
			getch();
			return -1;
		};
		izNeighAux[Auxneigh_len] = iNeigh;
		Auxneigh_len++;
		return Auxneigh_len-1;
	}



	int Vertex::AddNeighbourIfNecessary(
					long iNeigh, char rotate_here_rel_to_there)
	{
		int i;
		for (i = 0; i < neigh_len; i++)
		{
			if (izNeigh[i] == iNeigh) {
				if (rotate_here_rel_to_there == PBC[i]) // CHECK HOW PBC USED: here_rel_to_there?
					return i;
				if (iLevel < NUM_COARSE_LEVELS-1) {
					printf("rotation not in agreement. iNeigh %d rotate %d %d",
						iNeigh,rotate_here_rel_to_there,PBC[i]);
					// This warning comes up if we have 2 routes across PBC or not to get
					// to the affected point.

					// However in terms of coefficients that doesn't matter.
					// It is used for PBC_uplink but not in this version.

				//	getch();
					// For coarsest level we do not need PBC to be populated, so we can accept
					// that some influences come from one way and some from another.
					// (In fact, where is PBC flag ever used? Not sure - only for creating PBC_uplink?)
				}
				return i;
			};
		};
		AddNeighbourIndex(iNeigh);
		
		if (iLevel < NUM_COARSE_LEVELS-1) {
			PBC[i] = rotate_here_rel_to_there;
		} else {
			PBC[i] = 8;
		}
		return i;
	}

	void inline Vertex::AddNeighbourIndex(long index)
	{
		if (neigh_len < MAXNEIGH) {
			izNeigh[neigh_len] = index;
			memset(&(coeff[neigh_len].co[0][0]),0,sizeof(real)*NUM_EQNS_1*NUM_AFFECTORS_1);

			neigh_len++;			
		}  else {
			printf("error: neigh_len >= MAXNEIGH and attempted add neigh\n");
		};
	}
	// Only the stupid edge points make us need a separate neigh_len,tri_len.
	bool inline RemoveNeighIndexIfExists(long index)
	{
		int i = 0;
		while ((i < neigh_len) && (izTri[i] != index)) ++i;
		if (i == neigh_len) return false;
		neigh_len--;
		memmove(izNeigh+i,izNeigh+i+1,sizeof(long)*(neigh_len-i));
		return true;
	}
	void inline Vertex::AddTriIndex(long index)
	{
		if (tri_len < MAXNEIGH) {
			izTri[tri_len] = index;
			tri_len++;
		}  else {
			printf("error: tri_len >= MAXNEIGH and attempted add tri\n");
		};
	}
	bool inline Vertex::RemoveTriIndexIfExists(long index)
	{
		int i = 0;
		while ((i < tri_len) && (izTri[i] != index)) ++i;
		if (i == tri_len) return false;
		tri_len--;
		memmove(izTri+i,izTri+i+1,sizeof(long)*(tri_len-i));
		return true;
	}

	int Save(FILE * fp);
	int Load(FILE * fp);
	
	Vector2 Vertex::PopulateContiguousPosition__Guesswork(Vertex * pVertex);
};


class Triangle 
{
public:
	// Storing data:

	Vertex * cornerptr[3];    // keep this way for ease of continuity.
	Triangle * neighbours[3]; // do we want this? let's keep.
	int periodic; // number of points that are Clockwise wrapped relative to the others
	// note that BYTE == unsigned char and creates problems with testing for decrement below zero
	unsigned char u8domain_flag;  
	long indicator; // for storing whether triangle has been traversed in intersection search etc
	
	// Recalculated data:
	// Can recalculate normalised or unnormalised. To make use of it, see that it points outwards.
	Vector2 edge_normal[3]; // [1] means for edge with 0,2. // unwanted?
	Vector2 cent; 
//	Vector3 B;	// if A will be on vertices, B is naturally defined where?
	// Best plan will be to put A on edges and triangle centroids to get B_vertex.
	// We can also create B_edge by taking a quadrilateral of A.
	
	real temp_f64, area, nT, ROC_nT; // Maybe we need NT for doing pressures.

	Triangle()	;	
	
	// change most of these to inline and put them here:

	void Triangle::PopulatePositions(Vector2 & u0, Vector2 & u1, Vector2 & u2) const
	{
		u0 = cornerptr[0]->pos;
		u1 = cornerptr[1]->pos;
		u2 = cornerptr[2]->pos;
	};

	void Triangle::MapLeftIfNecessary(Vector2 & u0, Vector2 & u1, Vector2 & u2) const;
	real Triangle::GetDomainIntersectionArea(bool bUseOwnCoords, Vector2 u[3]) const;
	real Triangle::GetDomainIntersectionAreaROC(Vector2 u[3],int iWhichMove,Vector2 ROC);

	void Triangle::CreateCoordinates_rel_to_vertex(Vertex * pVertex,Vector2 & u1, Vector2 & u2, Vector2 & u3);
	bool inline Triangle::has_vertex(Vertex * pVertex)
	{
		return ((cornerptr[0] == pVertex) || (cornerptr[1] == pVertex) || (cornerptr[2] == pVertex));
	}

	int Triangle::GetCentreOfIntersectionWithInsulator(Vector2 & result);

	void Triangle::CalculateCircumcenter(Vector2 & cc, real * pdistsq);
	Vector2 RecalculateCentroid(real InnermostFrillCentroidRadius,real OutermostFrillCentroidRadius);
	Vector2 Triangle::GetContiguousCent_AssumingCentroidsSet(Vertex * pVertex);
	real Triangle::ReturnAngle(Vertex * pVertex);

	Vector3 GetAAvg() const;
	//void Triangle::GenerateContiguousCentroid(Vector2 * pCentre, Triangle * pContig);

	void GuessPeriodic(void);
	void inline Triangle::GetParity(int parity[3]) const
	{
		parity[0] = (cornerptr[0]->pos.x > 0.0)?1:0;
		parity[1] = (cornerptr[1]->pos.x > 0.0)?1:0;
		parity[2] = (cornerptr[2]->pos.x > 0.0)?1:0;
	}

	int inline Triangle::GetLeftmostIndex() const
	{
		// Note: we could put an argument for returning the one with leftmost gradient x/y
		int c1 = 0;
		if (cornerptr[1]->pos.x/cornerptr[1]->pos.y < cornerptr[0]->pos.x/cornerptr[0]->pos.y)
			c1 = 1;
		if (cornerptr[2]->pos.x/cornerptr[2]->pos.y < cornerptr[c1]->pos.x/cornerptr[c1]->pos.y)
			c1 = 2;
		return c1;
	}

	int inline Triangle::GetRightmostIndex() const
	{
		int c1 = 0;
		if (cornerptr[1]->pos.x/cornerptr[1]->pos.y > cornerptr[0]->pos.x/cornerptr[0]->pos.y)
			c1 = 1;
		if (cornerptr[2]->pos.x/cornerptr[2]->pos.y > cornerptr[c1]->pos.x/cornerptr[c1]->pos.y)
			c1 = 2;
		return c1;
	}

	int Triangle::GetCornerIndex(Vertex * pVertex);
	Vertex * GetOtherBaseVertex(Vertex * pVertex);
	int FindNeighbour(Triangle * pTri);

	void Triangle::RecalculateEdgeNormalVectors(bool normalise);
	real Triangle::ReturnNormalDist(Vertex * pOppVert);
	void Triangle::GetEdgeLengths(real edge_length[]);

	real Triangle::GetWeight(Vertex * pVertex); // ?

	void Triangle::Return_grad_Area(Vertex *pVertex, real * p_dA_by_dx, real * p_dA_by_dy);

	int Triangle::Save(FILE * fp,Vertex * pVertArray, Triangle *pTriArray);
	int Triangle::Load(FILE * fp, Vertex * pVertArray, Triangle * pTriArray);
	// saving and loading needed or not??

	real Triangle::GetArea(void) const
	{
		Vector2 u[3];
		MapLeftIfNecessary(u[0],u[1],u[2]);
		// Use shoelace formula not Heron:
		real area =  0.5*fabs(
					  u[0].x*u[1].y - u[1].x*u[0].y
					+ u[1].x*u[2].y - u[2].x*u[1].y
					+ u[2].x*u[0].y - u[0].x*u[2].y );
		return area;
		// Heron:
		//		d1 = distance(u0.x,u0.y,u1.x,u1.y);
		//		d2 = distance(u1.x,u1.y,u2.x,u2.y);
		//		d3 = distance(u0.x,u0.y,u2.x,u2.y);
		//	real Z = (d1+d2+d3)*0.5;
		//	real Heron = sqrt(Z*(Z-d1)*(Z-d2)*(Z-d3));
	}
	
	bool ContainsPoint(real x, real y); // requires transverse vectors to be set
	bool ContainsPointInterior (Vertex * pVert); // calls the above function after checking for equality with cornerptr
	
	// We are going to want to see if points lie within what? A vertex-centred cell?

	void IncrementPeriodic(void);
	void DecrementPeriodic(void);
	
	//char InferRelativeWrapping(Vertex * pVert, Vertex * pVertDisco);

	Vertex * Triangle::ReturnOtherSharedVertex(Triangle * pTri,Vertex * pVertex);
	Vertex * ReturnUnsharedVertex(Triangle * pTri2, int * pwhich = 0);
		
	//void Triangle::SetTriangleVertex(int which, Vertex * pVert);
		
	int TestAgainstEdge(real x, real y, 
							int c1, // the "start" of the relevant edge
							int other, // the point opposite the relevant edge
							Triangle ** ppNeigh);

	real Triangle::GetPossiblyPeriodicDistCentres(Triangle * pTri, int * prela);

	bool TestAgainstEdges(real x,real y, Triangle ** ppNeigh);
	bool TestAgainstEdges(float x,float y, Triangle ** ppNeigh);
	
	void Triangle::ReturnPositionOtherSharedVertex_conts_tranche(Triangle * pTri, Vertex * pVert, Vector2 * pResult);

	void Triangle::CalculateIntersectionArea(Triangle & tri2); 
	
	char Triangle::InferRelativeWrapping(Vertex * pVert, Vertex * pVertDisco);
};

//class AuxTriangle : public Triangle // note: cannot do inheritance in CUDA very well?
	// DO NOT WANT. Use Triangles now.
//{
//public:
//}; // probably do not want aux at all.
// EXCEPT  to not accidentally use X instead of AuxX[0].

/*
class ROCArray 
{
public:
	Vector3 * dv_ion_bydt, * dv_neut_bydt;
	real * heatrate_neut, * heatrate_ion; // for viscous heating

	long numTris;
	bool bInvoked;

	ROCArray();
	~ROCArray();

	//ROCArray(long numTriangles);
	int Invoke(long numTriangles);

	void CopyFrom(ROCArray * );
	void Zero();
	
	void Extrapolate( // populates values in self
			ROCArray * pROCflux0, ROCArray * pROCflux1, 
			real h_used, 
			ROCArray * pROCbase, real h_extrap);
	
	void InterpolateBack(ROCArray * pROCbase, real const h_attempt, real const h_old);

	real ROCArray::Estimate_d2dt2_and_get_timestep_for_constraint
			(ROCArray * pROC1, ROCArray * pROC2, real hstep, 
			real error_ppn_max // maximum ppnl error in approx of function
			);
	
	void ROCArray::SetLinear(ROCArray * pROCflux0, real hstep, ROCArray * dfdt, real h2, ROCArray * d2);
	void ROCArray::SetLinear(ROCArray * pROCflux0, real hstep, ROCArray * dfdt);
	void ROCArray::Get_dfdt(ROCArray * pROCflux0, ROCArray * pROCflux1, real hstep);
	void ROCArray::GetAvgSq(real * pneut, real * pion);
	
	void ROCArray::lambdablend(Speciesvector *plambda, ROCArray * pROCflux0, ROCArray * pROCflux1);

	void ROCArray::FileOutput(char filename[]);

};

// prefer array of struct.

class ROCHeat 
{
public:
	real d_NiTi_bydt[cellslength];
	real d_NnTn_bydt[cellslength];
	real d_NeTe_bydt[cellslength];

	long numTris;

	ROCHeat();
	ROCHeat(long numTriangles);

	void CopyFrom(ROCHeat * );
	void Zero();
	
	void Extrapolate( // populates values in self
			ROCHeat * pROCflux0, ROCHeat * pROCflux1, 
			real h_used, 
			ROCHeat * pROCbase, real h_extrap);
	
	void ROCHeat::SetLinear(ROCHeat * pROCflux0, real hstep, ROCHeat * dfdt, real h2, ROCHeat * d2);
	void ROCHeat::SetLinear(ROCHeat * pROCflux0, real hstep, ROCHeat * dfdt);
	void ROCHeat::Get_dfdt(ROCHeat * pROCflux0, ROCHeat * pROCflux1, real hstep);
	void ROCHeat::GetAvgSq(real * pneut, real * pion, real * pelec);
	
	void ROCHeat::lambdablend(Speciesvector *plambda, ROCHeat * pROCflux0, ROCHeat * pROCflux1);

};
class dTarray
{
public:
	real dTe_by_dt[cellslength];

	long numTris;

	dTarray();
	dTarray(long numTriangles);

	void Zero();
	
};
// prefer array of struct.
*/



class TriMesh
{
public:
	
	Vertex * X;                        
	Triangle * T;
	Vertex * Xdomain; // pointer to start of movable vertices.
	// Or for tri-data-based, pointer to start of insulator vertices.
	
	long numVertices, numTriangles, numTrianglesAllocated,
		numRows, numInnerVertices, numDomainVertices;
	long numInnermostRow, numOutermostRow, numLastRowAux[NUM_COARSE_LEVELS];
	// This stuff still applies. Domain vertices never leave the domain.

	// Vertex-based:
	// Full Delaunay on whole thing.
	// Let tri cells cross the insulator;
	// vertex cells are bounded at the edge by moving centroids to the centre of the intersection between cell and insulator.
	
	real Outermost_r_achieved; // outermost on finest level
	real Innermost_r_achieved; // innermost on finest level
	real InnermostFrillCentroidRadius, OutermostFrillCentroidRadius;
	
	real sum_of_coefficients;
	// The following used for the Iz equation in ODE solve:
		
	long numStartZCurrentRow, numEndZCurrentRow; // for verts

	long numReverseJzTris; // now put tris on edge of anode since offset

	long StartAvgRow;
	Matrix Coarsest;
	Matrix LUphi;
	real scratchval;

	dd_real EzTuning; // ?!
	
	Vertex * AuxX[NUM_COARSE_LEVELS];  
	Triangle * AuxT[NUM_COARSE_LEVELS]; 

	long numAuxVertices[NUM_COARSE_LEVELS];
	long numAuxTriangles[NUM_COARSE_LEVELS];
	long numTrianglesAuxAllocated[NUM_COARSE_LEVELS];
	long numRowsAux[NUM_COARSE_LEVELS];
	long numInnermostRowAux[NUM_COARSE_LEVELS];

	real Iz_prescribed, Epsilon_Iz, Epsilon_Iz_aux[NUM_COARSE_LEVELS];;
	// hmm, why not dd_real?

	qd_or_d Epsilon_Iz_coeff_On_PhiAnode;
	//Epsilon_Iz_coeff_On_TuningFactor;
	// Ez = TuneFac*pVertex->temp2.y
	// Making qd_or_d to be consistent with the rest of 'gamma' calculation.

	qd_or_d PhiAnode, PhiAnode_aux[NUM_COARSE_LEVELS]; // Auxiliary addition to Eext.
	qd_or_d Epsilon_Iz_constant, Epsilon_Iz_Default[NUM_COARSE_LEVELS];
	// plays same role as Epsilon_Iz_constant

	// match data types for finest.

	int numVolleysAux[NUM_COARSE_LEVELS];
	long numNonBackAuxVertices[NUM_COARSE_LEVELS];
	
//	smartlong immovable;

	// Used in DestroyOverlaps:
	smartlong Disconnected;  // for vertex indices
	smartlong TriangleHeap; // INDICES of scraped triangles.

	//smartvp TriangleHeap;    // pointers to scrapped triangles -- still need? indexes would be better tho?
	smartlong ScratchSearchTris;  // for doing radar search for vertex within triangle
		
	// used in initialisation functions:
	short numRow[10000]; // >10000 rows is not likely
	//short numRowInner[10000];
		
	long numVolleys[NUM_COARSE_LEVELS+1];

	// screw pinch only:
	real OuterRadiusAttained;
	real OuterRadius[NUM_COARSE_LEVELS];

	TriMesh();
	~TriMesh();
	
	// For now, get rid of most member functions and keep only those that we know we shall use.

	// New additions:

	void TriMesh::SetTriangleVertex(int iWhichCorner, Triangle * pTri, Vertex * pVertex);
	void TriMesh::Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	int TriMesh::SeekVertexInsideTriangle(Vertex * v1,
									  Vertex * v2,
									  Vertex * v3,          // up to 4 points to check for
									  Vertex * v4,           // in order of preference;
									  Triangle * pSeedTri,    // Seed triangle to begin radar
									  Vertex ** ppReturnVert,  // address for returning guilty vertex
									  Triangle ** ppReturnTri); // address for returning triangle that contains

	void TriMesh::ShiftVertexPositionsEquanimity();

	// *******************************************************************
	// *****																	  *****
	// *********            Initialisation functions:            *********
	// *******************************************************************
	
	// to go in MeshUtil.cpp:
	int TriMesh::CreateEquilateralAuxMesh(int iLevel); // don't do it -- create on the fly we said supposedly.
	int TriMesh::CreateEquilateralAuxMeshScrewPinch(int iLevel);
	void TriMesh::InitialiseScrewPinch();
	void TriMesh::InitialPopulateScrewPinch(void); // it will need some clever rethink.

	int Initialise(int token);                             
	// the token allows us to identify the object in giving error messages.

	real TriMesh::SolveConsistentTemperature(real n, real n_n);
	void TriMesh::InitialPopulate(void);  // call this once initial positions are obtained.
	void InitialisePeriodic(void);			  

	// *******************************************************************
	// *****                                                         *****
	// *********       Mesh Maintenance / utility functions:     *********
	// *******************************************************************
	
	void TriMesh::CreateTilingAndResequence(TriMesh * pDestMesh);

	// in MeshUtil.cpp:
	int Load(char * filename);
	int Save(char * filename);
	int SaveText(char * filename);

	void TriMesh::SetTriangle(Triangle * pTri, Vertex * pV1, Vertex * pV2, Vertex * pV3);	// not to be called with impunity

	Triangle * TriMesh::SetAuxTri(int iLevel, long iVertex1, long iVertex2, long iVertex3);

		// in mesh.cpp:
//	int SeekVertexInsideTriangle(Vertex * v1,
//									  Vertex * v2,
//									  Vertex * v3,          // up to 4 points to check for
//									  Vertex * v4,           // in order of preference;
//									  Triangle * pSeedTri,    // Seed triangle to begin radar
//									  Vertex ** ppReturnVert,   // address to return guilty vertex
//									  Triangle ** ppReturnTri); // address for returning triangle that contains
		// Not clear that ever gets used. ?

	// void TriMesh::ShiftVertexPositionsEquanimity(); // no
	
	void TriMesh::ResetTriangleNeighbours(Triangle * pTri); 

	// this is for if first, triangles are sorted anticlockwise: (??)
	void TriMesh::RefreshVertexNeighboursOfVerticesOrdered(void) ; // ??
		
	void TriMesh::RefreshHasPeriodic(); // see if any triangles had by a vertex are periodic triangles.

	void TriMesh::RecalculateEdgeNormals(bool bNormalise);
	
	void TriMesh::RefreshHasPeriodicAux(int iLevel);

//	Vertex * TriMesh::Search_for_iVolley_equals (Vertex * pSeed,int value);
//	AuxVertex * TriMesh::Search_for_iVolley_equals (AuxVertex * pSeed,int value, int iLevel);
//	void TriMesh::RefreshIndexlists(); // did what?	
	//AuxTriangle * TriMesh::ReturnPointerToOtherSharedTriangleAux(
	//	AuxVertex * pVert,
	//	AuxVertex * pOther,
	//	AuxTriangle * p_not_this_one,
	//	int iLevel);	
	//void TriMesh::Recalculate_Kappa_NuHeart_and_Ratio (Vertex * pVertex, int species);
	// don't think we want such a function.

	// Search functions, in MeshUtil.cpp //was basics.cpp:

	Triangle * ReturnPointerToTriangleContainingPoint(
				Triangle * pTri,              // seed for beginning triangle search
				real x, real y	); 

	void TriMesh::SearchIntersectionsForPolygon(ConvexPolygon & cp,Triangle * pTri, 
							real coefficient, macroscopic * pVars, int varcode, real area);
	
	Triangle * ReturnPointerToOtherSharedTriangle(
		Vertex * pVert,
		Vertex * pOther,
		Triangle * p_not_this_one, int iLevel = -1);

	//AuxTriangle * TriMesh::GetAuxTriangleContaining(AuxVertex * pAux1,
	//						   AuxVertex * pAux2,
	//						   int iLevel);

	//long TriMesh::SearchForAuxTriangleContainingPoint(real x, real y, int iLevel); // version for equilateral aux mesh
	//long TriMesh::SearchForAuxTriangleContainingPoint(real x, real y, 
	//											  int iLevel,
	//									AuxTriangle * pTriSeed);

	bool TriMesh::FindOtherNeigh(Vertex * pVertex1, Vertex * pVertex2, Vertex * pVertNot, Vertex ** ppOtherNeigh);
	
	Vertex * TriMesh::Search_for_iVolley_equals (Vertex * pSeed,int value);

	void TriMesh::CreateVolleys(int separation) ;
	void TriMesh::CreateTriangleVolleys();

	// Mesh maintenance proper:
	// ________________________
	// in mesh.cpp :

	int DestroyOverlaps(int max_attempts);
	int ExamineNeighbourAndDisconnectIfNeeded(Triangle * pTri, int opp, int c1);
	int Disconnect(Vertex * pVertDisco, Triangle *pTriContain); // return index of next point to disconnect, or -1 on total success
	int FullDisconnect(Vertex * pVertDisco, Triangle *pTriContain); // return number disconnected
	void ReconnectLastPointInDiscoArray(void);
	void TriMesh::DebugTestWrongNumberTrisPerEdge(void);
	bool TriMesh::DebugTestForOverlaps();

	void TriMesh::CheckDoubleEdges(int numTris);

	void Redelaunerize(bool exhaustion, bool bReplace);	
	void Flip(Triangle * pTri1, Triangle * pTri2, int iLevel);//, int flag = 0);
	//void TriMesh::Flip(AuxTriangle *pTri1, AuxTriangle * pTri2, int iLevel);
	// we will need something similar if we do construct Delaunay at each level.

	bool TriMesh::DebugDetectDuplicateNeighbourInList(Vertex * pVertex);

	real TriMesh::SwimVertices(TriMesh * pSrcMesh, real coefficient, real * pAcceptance);
	void TriMesh::SwimMesh(TriMesh * pSrcMesh);
	
	void TriMesh::CopyMesh(TriMesh * pDestMesh);
	void TriMesh::SurveyCellMassStats(real * pAvgMass, real * pMassSD, real * pMinMass, real * pMaxMass, int * piMin);
	

	
	// *******************************************************************
	// *****                                                         *****
	//// *********              Simulation functions:              *********
	// *******************************************************************
	
	
	void TriMesh::Set_nT_and_Get_Pressure(int species);

	void TriMesh::InterpolateAFrom(TriMesh * pSrcMesh);
	void TriMesh::CreateShards(Vertex * pVertex, ShardData & shard_data);
	void TriMesh::GiveAndTake(ShardData & shard_data, Vertex * pVDest,Vertex * pVSrc);
	void Advance(TriMesh *); // does whole step by calling some of the following.
	int TriMesh::AdvancePinch(TriMesh * pDestMesh);
	void TriMesh::RecalculateCentroid(Vertex * pVertex); // needed access to tris !

	//void TriMesh::MakePressureAccelData(int code);

	void TriMesh::RestoreSpeciesTotals(TriMesh * pSrc);
	void TriMesh::ReportIonElectronMass();

	void TriMesh::GetBFromA();

	void TriMesh::GetGradTeOnVertices();

	void TriMesh::GetCurlBcOver4Pi();

	//void TriMesh::SearchIntersectionsOfTriangle(Triangle * pTri, Vector2 & x0, Vector2 & x1, Vector2 & x2,
	//				real mass, real heat, Vector3 & mom, int species, real area, bool src_periodic_flag);
	
	void TriMesh::CollectFunctionals();
	real TriMesh::Report_Min_nTotal_Tri();
	void TriMesh::ReportTextOutput1DData(TriMesh * pDestMesh);
		
	void TriMesh::ZeroCellData();
	void TriMesh::RepopulateCells(TriMesh * pDestMesh, int code);

	int TriMesh::DebugCheckTemperaturesPositive();


	// Evolution:
	// __________
/*
	void TriMesh::Collect_maxh4_bEndpt(
		DTArray * pROCflux_old, DTArray * pROCflux_now,
		DTArray * pROCflux_output,	DTArray * pROCflux_predict,
		f64 hstep_old,f64 hstep,
		f64 * pmaxh4_running, bool * pbEndptFluxEffectAgree);

	void TriMesh::Collect_maxh4_bEndpt_VA(
				ROCArray * pROCflux_old,
				ROCArray * pROCflux_now,
				ROCArray * pROCflux_output,
				ROCArray * pROCflux_predict,
				f64 hstep_old, f64 hstep,
				f64 * pmaxh4_running, bool * pbEndptFluxEffectAgree);

	void TriMesh::Setup_Data_for_CUDA(
							fluid3BE * pCelldata_host
						//	structural * pCellinfo_host, 
						//	vertinfo * pVertinfo_host,
						//	vertdata * pVertdata_host
							);
*/
	void TriMesh::PopulateSystdata_from_this(
							Systdata * pSystdata
							);

	void TriMesh::Populate_this_fromSystdata(
							Systdata * pSystdata
							);
	

#ifdef CPU
	void TriMesh::ViscosityAndAcceleration(real hsub);
	void TriMesh::ComputeMomFlux(ROCArray * pOutput);
	void TriMesh::InternalAcceleration(real hsub, ROCArray * pROCinitial, ROCArray * pROCfinal,
		Heat_storage * pHeat_storage);
	void TriMesh::IonisationAndHeat(real hsub, ROCHeat * pdTinitial, ROCHeat * pdTfinal,
		Momentum_storage * pMomentum_storage,
		real h_full);
	void TriMesh::HeatRoutine(real hsub);
	void TriMesh::ComputeHeatFlux(ROCHeat * pROC);
#endif

	void TriMesh::GetGradTe(Vertex * pVertex);

	real TriMesh::GetOutwardHeatFlux ( Triangle * pTri, Triangle * pTriDest, int iEdge, int species );

	//void TriMesh::AccelerateIons_or_ComputeOhmsLawForRelativeVelocity(Triangle * pTri, int code);

	// Advection-related
	// __________________

	void TriMesh::CreateFeintHeavyDisplacement(Triangle * pTri,
		                        Vector3 * p_v_ion_k, Vector3 * p_v_neut_k);
	void TriMesh::CreateMeshDisplacement_zero_future_pressure();

	void TriMesh::ZeroVertexPositions();
	void TriMesh::SolveForAdvectedPositions(TriMesh * pDestMesh);

	void TriMesh::PlaceAdvected_Triplanar_Conservative_IntoNewMesh(
													int which_species,
													TriMesh * pDestMesh,
													 int code,
													 int bDoCompressiveHeating);
	void TriMesh::AverageVertexPositionsAndInterpolate(TriMesh * pSrcMesh, bool bInterpolatePositions);
	
	void TriMesh::SendMacroscopicPolygon(ConvexPolygon & cp,Triangle * pTriSeed, bool src_periodic,
									 real coefficient,macroscopic * pVars, int varcode);

	void TriMesh::ApplyVertexMoves(int which_species,TriMesh * pDestMesh);

	real TriMesh::SendAllMacroscopicPlanarTriangle(ConvexPolygon & cp,Triangle * pTriSeed, int src_periodic,
									 fluidnvT * pvertvars0, fluidnvT * pvertvars1, fluidnvT * pvertvars2,
									 int species, int code);

	void TriMesh::FinishAdvectingMesh(TriMesh * pDestMesh);

	// Solver-related
	//________________
	
	void TriMesh::ExtractData_PerformJRLS();

	real inline TriMesh::GetIzPrescribed(real const t);

	void TriMesh::ComputeOhmsLaw();
	
	void TriMesh::RecalculateEpsilonVertex(long iVertex, int iLevel);

	void TriMesh::SpitOutGauss();

	void TriMesh::GaussSeidel(int iLevel, int iVolley);
	void TriMesh::IterationsJRLS(int iLevel, int iterations);
	void TriMesh::IterationsJRLS_individual_equations(int iLevel, int iterations);
	void TriMesh::RunLU(int const iLevel, bool bRefreshCoeff);

	void TriMesh::IterationsJRLS_Az(int iLevel, int iterations);
	void TriMesh::RunLU_Az(int const iLevel, bool bRefreshCoeff);
	void TriMesh::CalculateEpsilons();
	void TriMesh::CalculateEpsilonsAbsolute(real RSS_Absolute_array[4]);
	void TriMesh::CalculateEpsilonAux4(int iLevel);

	void TriMesh::CalculateEpsilonsAz();
	void TriMesh::CalculateEpsilonsAbsoluteAz(real RSS_Absolute_array[4]);
	void TriMesh::CalculateEpsilonAuxAz(int iLevel);

	//void TriMesh::CreateSeed(TriMesh * pMesh_with_A_k); // ?

	void TriMesh::Solve_A_phi(bool const bInitial, real const time_back_for_Adot_if_initial = 0.0);

	void TriMesh::Solve_Az( real const time_back_for_Adot_if_initial );

	//void TriMesh::Calculate_Epsilons_Gauss_Ampere(void);
	//void TriMesh::RunIterations(int iLevel, long iterations);	
	//void TriMesh::RunIterations4(int iLevel, long iterations);	
		
	void TriMesh::Lift_to_coarse_eps(int iLevel);
	void TriMesh::Affect_vars_finer(int iLevel);
	void TriMesh::Lift_to_coarse_eps_Az(int iLevel);
	void TriMesh::Affect_vars_finer_Az(int iLevel);
	
	void TriMesh::EstimateInitialOhms_zz(void);
	//void TriMesh::EstimateInitialEandJz(void);

	void TriMesh::CreateAuxiliarySubmeshes(bool);
	void TriMesh::Set_AuxNeighs_And_GalerkinCoefficients(int iLevel);
	void TriMesh::Set_AuxNeighs_And_GalerkinCoefficientsAz(int iLevel);
	void TriMesh::AccumulateCoefficients(
			Vertex * pAffected, //pAffector, 
					//pVertex->iCoarseIndex[iAffectedIndex],
			int iNeigh,//pVertex->iCoarseIndex[iAffectorIndex],
			real wtsrc, real wtdest, real coeff[4][4],
			char rotatesrc,char rotatedest,
			
			// for debug:
			long iFinePhi, long iIntermediate);
	void TriMesh::Accumulate_coeffself_unary(
		Vertex * pAffected, //long iAffected,
		real,
		real coeff_self[NUM_EQNS_2][NUM_AFFECTORS_2], 
		char rotatedest);

	// void TriMesh::CreateMultimeshCoefficients(bool bConstruct_iCoarseTriangle);
	void TriMesh::CreateODECoefficients( bool bInitial);
	void TriMesh::CreateODECoefficientsAz( bool bInitial);
	
	
	// *******************************************************************
	// *****																			 *****
	// *********              Graphics functions:                *********
	// *******************************************************************

	void TriMesh::GosubMakeGraphs(int GraphDisplaySwitch, TriMesh * pMesh_with_A_k);

	void TriMesh::SetVerticesAndIndicesAux(int iLevel,
									   VertexPNT3 * vertices,
										  DWORD * indices,
									long numVerticesMax,
									long numTrianglesMax,
									int colourflag,
									int heightflag,
									int offset_data,			// how far data is from start of Vertex, for real*
									int offset_vcolour,
									float zeroplane, float yscale,
									int NTris = 0)		;
	//void TriMesh::Setup_DivE_rho_phi_eps_Graphing();	

	//void TriMesh::SetupJ_A2_rho_Graphing();
	real TriMesh::ReturnMaximumData(int offset);
	void TriMesh::ReturnMaxMinData(int offset, real * pmax, real * pmin, bool bDisplayInner) const;
	void TriMesh::ReturnL5Data(int offset, real * pmax, real * pmin, bool bDisplayInner) const;
	void TriMesh::ReturnMaxMinDataAux(int iLevel, int offset, real * pmax, real * pmin);
	
	void TriMesh::Setup_J() ;
	void TriMesh::Reset_vertex_nvT(int species) ;

	long TriMesh::GetVertsRightOfCutawayLine_Sorted(long VertexIndexArray[],
										real radiusArray[]) const;

	real TriMesh::ReturnMaximumDataAux(int iLevel, int offset);

	void TriMesh::SetupAccelGraphs();
	//void TriMesh::SetupJBEGraphing();
	//void TriMesh::SetupSigmaJEGraphing();
	//void TriMesh::SetupOhmsEzGraphing();
	//void TriMesh::SetupAizJzGraphing();
	void TriMesh::CalculateTotalGraphingData();

	real TriMesh::ReturnMaximumVelocity(int offset_v, bool bDisplayInner) const;
	real TriMesh::ReturnL4_Velocity(int offset_v, bool bDisplayInner) const;
	real TriMesh::ReturnMaximum3DMagnitude(int offset_v, bool bDisplayInner) const;
	real TriMesh::ReturnL4_3DMagnitude(int offset_v, bool bDisplayInner) const;
	real TriMesh::ReturnMaximumVelocityAux(int iLevel, int offset);
	//void TriMesh::SetupJGraphing();			// set A2 to J at vertex

	//void TriMesh::SetupJ_E_rho_Graphing();

	long GetNumVerticesGraphics(void);
	
	long TriMesh::GetNumVerticesGraphicsAux(int iLevel);

	long TriMesh::GetNumKeyVerticesGraphics(long * pnumTrianglesKey) const;

	void TriMesh::SetVerticesKeyButton(VertexPNT3 * vertices, DWORD * indices, real maximum_v, int colourflag) const;

	void SetVerticesAndIndices(VertexPNT3 * vertices[],        // better to do in the other class...
							        DWORD * indices[], // let's hope this means an array of pointers
							long const numVerticesMax[], long const numTrianglesMax[], // pass it the integer counts so that it can test for overrun
		                    long numVerticesUsed[], long numTrianglesUsed[],
							int colourflag,
							int heightflag, int offset, int v_offset, float zeroplane, float yscale,
							bool boolDisplayInnerMesh) const;	
};

bool PerformCUDA_JRLS (real * pBeta, real * pBetaIz, 
					   long * pIndex, 
					   long nbetamax, long Ncells,
					   real * pAz,real chEzExt,
					   real Epsilon_Iz_constant,
						real Epsilon_Iz_coeff_On_chEz_ext,
						real * return_chEzExt,
						long StartAvgRowTri,
						long iterations);
	

//
//class Store_v_array
//{
//public:
//	Vector3 v_ion[cellslength];
//	Vector3 v_neut[cellslength];
//
//	// Storing for vertices also:
//	Vector3 v_ion_vert[vertlen];
//	Vector3 v_neut_vert[vertlen];
//
//	void store(TriMesh * pX);
//
//	void Writeback(TriMesh * pX);
//	
//	real Findmaxtimestep_given_dv(Store_v_array * pStore_v_1, long numTris, real hstep, real PPNMAX);
//};
//
//
//int const ITERATIONS_MAX = 40000; // applies if SolveForA called with no argument.



#endif