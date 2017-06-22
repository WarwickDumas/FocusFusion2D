#ifndef BASICSCPP
#define BASICSCPP

#include "mesh.h"
#include "globals.h"

#include "vector_tensor.cu"
//#include "vector_tensor.cpp"
// For manipulations of triangles and vertices.

smartlong GlobalVertexScratchList;

extern real FRILL_CENTROID_OUTER_RADIUS;
extern real FRILL_CENTROID_INNER_RADIUS;

//real const minimum_pressure_SD_at_1e18_sq = minimum_pressure_SD_at_1e18*minimum_pressure_SD_at_1e18;
//real const min_variance_heat = min_SD_heat*min_SD_heat;

Tensor2 const Anticlockwise(cos(FULLANGLE),-sin(FULLANGLE),sin(FULLANGLE),cos(FULLANGLE));
Tensor2 const Clockwise(cos(FULLANGLE),sin(FULLANGLE),-sin(FULLANGLE),cos(FULLANGLE));
Tensor3 const Anticlockwise3 (cos(FULLANGLE),-sin(FULLANGLE), 0.0,
					sin(FULLANGLE),cos(FULLANGLE), 0.0,
					0.0, 0.0, 1.0);
Tensor3 const Clockwise3 (cos(FULLANGLE),sin(FULLANGLE), 0.0,
					-sin(FULLANGLE),cos(FULLANGLE), 0.0,
					0.0, 0.0, 1.0);
Tensor2 const HalfAnticlockwise (cos(HALFANGLE),-sin(HALFANGLE),sin(HALFANGLE),cos(HALFANGLE));
Tensor2 const HalfClockwise(cos(HALFANGLE),sin(HALFANGLE),-sin(HALFANGLE),cos(HALFANGLE));

real modelled_n;
void ConvexPolygon::CreateClockwiseImage(const ConvexPolygon & cpSrc) 
{
	numCoords = cpSrc.numCoords;
	for (int i = 0; i < numCoords; i++)
		coord[i] = Clockwise*cpSrc.coord[i];
}
void ConvexPolygon::CreateAnticlockwiseImage(const ConvexPolygon & cpSrc) 
{
	numCoords = cpSrc.numCoords;
	for (int i = 0; i < numCoords; i++)
		coord[i] = Anticlockwise*cpSrc.coord[i];	
}

fluid_nvT fluid_nvT::Clockwise() const
	{
		fluid_nvT result;
		memcpy(&(result.n),&(n),sizeof(real)*3);
		memcpy(&(result.nT),&(nT),sizeof(real)*3);
		result.nv[0] = Clockwise3*nv[0];
		result.nv[1] = Clockwise3*nv[1];
		result.nv[2] = Clockwise3*nv[2];
		return result;
	}
fluid_nvT fluid_nvT::Anticlockwise() const
	{
		fluid_nvT result;
		memcpy(&(result.n),&(n),sizeof(real)*3);
		memcpy(&(result.nT),&(nT),sizeof(real)*3);
		result.nv[0] = Anticlockwise3*nv[0];
		result.nv[1] = Anticlockwise3*nv[1];
		result.nv[2] = Anticlockwise3*nv[2];
		return result;
	}
void fluidnvT::Interpolate ( fluidnvT* pvv1, fluidnvT * pvv2,
							Vector2 & pos1, Vector2 & pos2, Vector2 & ourpos)
	{
		// want to take dist1/(dist1 + dist2)
		//  (dist1/(dist1+dist2)) = dist1/dist2 / (1 + dist1/dist2)
//		real ratio = sqrt( 
//			((pos1.x-ourpos.x)*(pos1.x-ourpos.x)+(pos1.y-ourpos.y)*(pos1.y-ourpos.y))/
//			((pos2.x-ourpos.x)*(pos2.x-ourpos.x)+(pos2.y-ourpos.y)*(pos2.y-ourpos.y)));
		// this is too dangerous - maybe ourpos == pos2
	
		real dist1 = sqrt((pos1.x-ourpos.x)*(pos1.x-ourpos.x)+(pos1.y-ourpos.y)*(pos1.y-ourpos.y));
		real dist2 = sqrt((pos2.x-ourpos.x)*(pos2.x-ourpos.x)+(pos2.y-ourpos.y)*(pos2.y-ourpos.y));
		real ppn = dist1/(dist1+dist2); 
		real minus = 1.0-ppn;
		n = ppn*pvv2->n + minus*pvv1->n;
		T = ppn*pvv2->T + minus*pvv1->T;
		v = ppn*pvv2->v + minus*pvv1->v;
	}

void GetInterpolationCoefficients( real beta[3],
							real x, real y,
							Vector2 pos0, Vector2 pos1, Vector2 pos2)
{
	// idea is to form a plane that passes through z0,z1,z2.

	// so firstly if we lie on a line between 0 and 1, we know what that is;
	// then we have some gradient in the direction normal to that which is determined by y2

	//relative = pos-pos0;
	//along01 = relative.dot(pos1-pos0)/(pos1-pos0).modulus(); 
	//// by being clever we should be able to avoid the square root since have z0 + (z1-z0)/(pos1-pos0).modulus()
	//perp.x = pos0.y-pos1.y;
	//perp.y = pos1.x-pos0.x;
	//away = relative.dot(perp)/perp.modulus();

	//pos2along01 = (pos2 - pos0).dot(pos1-pos0)/(pos1-pos0).modulus();
	//pos2away = (pos2-pos0).dot(perp)/perp.modulus();

	//real z_ = z0 + pos2along01*(z1-z0)/(pos1-pos0).modulus();
	//gradient_away = (z2-z_)/pos2away;

	//real z = z0 + along01*((z1-z0)/(pos1-pos0).modulus()) + away*gradient_away;
	//*pResult = z;


	// fast version:

	Vector2 pos(x,y);
	Vector2 perp;
	real ratio;//, coeff_on_z0, coeff_on_z1, coeff_on_z2;
	Vector2 relative = pos-pos0;
	Vector2 rel1 = pos1-pos0;
	Vector2 rel2 = pos2-pos0;
	real mod01sq = rel1.dot(rel1);
	real along01_over_mod01 = relative.dot(rel1)/mod01sq;
	real pos2along01_over_mod01 = rel2.dot(rel1)/mod01sq;
	//real z_expect = z0 + pos2along01_over_mod01*(z1-z0);
	//gradient_away = (z2-z_expect)*(perp.modulus()/((pos2-pos0).dot(perp)));
	//away_times_gradient_away = (z2-z_expect)*relative.dot(perp)/((pos2-pos0).dot(perp));
	//real z = z0 + along01_over_mod01*((z1-z0)) + away_times_gradient_away;

	// can we work out coefficients actually on z0,z1,z2 because then can do faster in 2D,3D. :
	
	perp.x = -rel1.y;
	perp.y = rel1.x;
	ratio = relative.dot(perp)/(rel2.dot(perp));
	
	//beta[0] = 1.0 - along01_over_mod01 - ratio + ratio*pos2along01_over_mod01;
	beta[1] =         along01_over_mod01             - ratio*pos2along01_over_mod01;
	beta[2] =                              ratio;
	beta[0] = 1.0 - beta[1] - beta[2];
	
	//*pResult = coeff_on_z0*z0 + coeff_on_z1*z1 + coeff_on_z2*z2;
}
void TriMesh::RecalculateCentroid(Vertex * pVertex)
	{
		// ASSUMES TRI CENTROIDS SET.

		long tri_len, i;
		Triangle * pTri;
		ConvexPolygon cp;
		long izTri[128];
		Vector2 tri_cent;

		if ((pVertex->flags != INNERMOST) && (pVertex->flags != CONVEX_EDGE_VERTEX))
		{
			cp.Clear();
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				tri_cent = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
				cp.add(tri_cent);
			}
			pVertex->centroid = cp.CalculateBarycenter();			
		} else {
			pVertex->centroid = pVertex->pos; // at edge of memory
		};
	}
	
Vector3 Triangle::GetAAvg() const
{
	Vector3 A;
	if (u8domain_flag != CROSSING_INS)
	{
		if (periodic == 0)
		{	A = (cornerptr[0]->A + cornerptr[1]->A + cornerptr[2]->A)/3.0;
			return A;
		};
		int par[3];
		GetParity(par);
		A.x = 0.0; A.y = 0.0; A.z = 0.0;
		if (par[0] == 0){
			A += cornerptr[0]->A;
		} else {
			A += Anticlockwise3*cornerptr[0]->A;
		};
		if (par[1] == 0) {
			A += cornerptr[1]->A;
		} else {
			A += Anticlockwise3*cornerptr[1]->A;
		};
		if (par[2] == 0) {
			A += cornerptr[2]->A;
		} else {
			A += Anticlockwise3*cornerptr[2]->A;
		};
		A /= 3.0;
		return A;
	}
	// In the insulator crossing case the used position is shifted from
	// the centroid down to the insulator.
	
	real beta[3];
	GetInterpolationCoefficients(beta, cent.x, cent.y,
						cornerptr[0]->pos,
						cornerptr[1]->pos,
						cornerptr[2]->pos);
	if (periodic == 0) {
		A = beta[0]*cornerptr[0]->A + beta[1]*cornerptr[1]->A + beta[2]*cornerptr[2]->A;
		return A;
	};
	int par[3];
	GetParity(par);
	A.x = 0.0; A.y = 0.0; A.z = 0.0;
	if (par[0] == 0){
		A += beta[0]*cornerptr[0]->A;
	} else {
		A += beta[0]*(Anticlockwise3*cornerptr[0]->A);
	};
	if (par[1] == 0) {
		A += beta[1]*cornerptr[1]->A;
	} else {
		A += beta[1]*(Anticlockwise3*cornerptr[1]->A);
	};
	if (par[2] == 0) {
		A += beta[2]*cornerptr[2]->A;
	} else {
		A += beta[2]*(Anticlockwise3*cornerptr[2]->A);
	};
	return A;
}


/*macroscopic macroscopic::operator* (const real hh,const macroscopic &vars)
	{
		macroscopic cv;
		cv.mass = hh*vars.mass;
		cv.heat = hh*vars.heat;
		cv.mom = hh*vars.mom;
		return cv;
	}
*/
/*
bool inline AuxTriangle::has_vertex(AuxVertex * pVertex)
{
	return ((cornerptr[0] == pVertex) || (cornerptr[1] == pVertex) || (cornerptr[2] == pVertex));
};

AuxVertex::AuxVertex() {
		flags = 0;
		tri_len = 0;
		neigh_len = 0;
	};

	void AuxVertex::addtri(long iTri)
	{
		iTriangles[tri_len] = iTri;
		tri_len++;
		if (tri_len > MAXNEIGH) {
			printf("\n\ntri_len > MAXNEIGH. stop.\n\n");
			getch();
  			tri_len = tri_len;
		};
		if (tri_len > 8) 
		{
			tri_len = tri_len;
		}
	};

	void AuxVertex::remove_tri(long iTri)
	{
		long iWhich = 0;
		while ((iWhich < tri_len)
			&& (iTriangles[iWhich] != iTri)) iWhich++;
		if (iWhich == tri_len)
		{
			iWhich = iWhich;
		}

		memmove(iTriangles+iWhich,iTriangles+iWhich+1,sizeof(long)*(tri_len-iWhich-1));
		tri_len--;
		
	};

	void AuxVertex::add_neigh(long iNeigh)
	{
		if (neigh_len == MAXNEIGH){
			printf("Had to stop: too many neighs in Aux mesh.\n");
			getch();
			return;
		}
		iNeighbours[neigh_len] = iNeigh;
		neigh_len++;
	};

	int AuxVertex::add_neigh_unique(long iNeigh)
	{
		int i;
		for (i = 0; i < neigh_len; i++)
			if (iNeighbours[i] == iNeigh) return 0;
		if (neigh_len == MAXNEIGH){
			printf("Had to stop: too many neighs in Aux mesh.\n");
			getch();
			return 2;
		}
		iNeighbours[neigh_len] = iNeigh;
		neigh_len++;
		return 1;
	};

	//void coeff_add(long iVertex, real beta)
	//{
	//	coeff_extra.add(beta);
	//	coeff_self -= beta;
	//	index_extra.add(iVertex);

	//};

	void AuxVertex::PopulatePosition(Vector2 & result)
	{
		result.x = x; result.y = y;
	}

	AuxTriangle::AuxTriangle() {
		flags = DOMAIN_TRIANGLE;
		periodic = 0;
	}
	AuxTriangle::~AuxTriangle() {}

	void AuxTriangle::PopulatePositions(Vector2 & u0, Vector2 & u1, Vector2 & u2)
	{
		cornerptr[0]->PopulatePosition(u0);
		cornerptr[1]->PopulatePosition(u1);
		cornerptr[2]->PopulatePosition(u2);
	};

	int AuxTriangle::GetLeftmostIndex()
	{
		// Note: we could put an argument for returning the one with leftmost gradient x/y
		int c1 = 1;
		if (cornerptr[2]->pos.x/cornerptr[2]->pos.y < cornerptr[1]->pos.x/cornerptr[1]->pos.y)
			c1 = 2;
		if (cornerptr[0]->pos.y != 0.0) {
			if (cornerptr[0]->pos.x/cornerptr[0]->pos.y < cornerptr[c1]->x/cornerptr[c1]->y)
				c1 = 0;
		};
		return c1;
	}

	int AuxTriangle::GetRightmostIndex()
	{
		int c1 = 1;
		if (cornerptr[2]->pos.x/cornerptr[2]->pos.y > cornerptr[1]->pos.x/cornerptr[1]->pos.y)
			c1 = 2;
		if (cornerptr[0]->pos.y != 0.0) {
			if (cornerptr[0]->pos.x/cornerptr[0]->pos.y > cornerptr[c1]->x/cornerptr[c1]->y)
				c1 = 0;
		};
		return c1;
	}
*/


smartlong::smartlong()
	{
		ptr = NULL;
		len = 0;
		alloclen = 0;
	};

void smartlong::clear()
	{
		if (ptr != NULL) free(ptr);
		ptr = NULL;
		len = 0;
		alloclen = 0;
	};

void smartlong::remove_if_exists(long what)
	{
		if (len == 0) return;
		
		long * look = ptr;
		long * ptrlast = ptr+len-1;

		for (look = ptr; look <= ptrlast; ++look)
		{
			if (*look == what) {
				for (; look < ptrlast; ++look)
					*look = look[1];
				len--;
				return;
			}
		};
	}

void smartlong::remove(long what)
	{
		// DEBUG VERSION:
		//
		if (len == 0) return;
		
		//long * look = ptr;
		//while (*look != what) ++look;
		//long * ptrlast = ptr+len-1;
		//for (; look < ptrlast; ++look)
		//	*look = look[1];
		//len--;		
		
		long * look = ptr;
		long * ptrlast = ptr+len-1;

		while ((look <= ptrlast) && (*look != what)) ++look;
		if (look > ptrlast) 
		{
			printf("!!!");
			getch();
		};
		for (; look < ptrlast; ++look)
			*look = look[1];
		len--;		
	}

void smartlong::IncreaseDim()
	{
		ptr = (long *)realloc(ptr,sizeof(long)*(alloclen+ALLOC));
		if (ptr == 0) 
		{
			printf("smartlong memory alloc failed!!!\n");	getch();
			len = len;
		};
		alloclen = alloclen+ALLOC;
	}

void smartlong::add(long what)
	{
		// make another function to only add unique....
		len++;
		if (len >= alloclen) IncreaseDim();
		
		ptr[len-1] = what;
	};

void smartlong::add_at_element(long what,long iInsert)
	{
		len++;
		if (len >= alloclen) IncreaseDim();
		
		memmove(ptr+iInsert+1,ptr+iInsert,sizeof(long)*(len-iInsert-1)); // new len ...

		ptr[iInsert] = what;
	}

void smartlong::copyfrom(smartlong & src)
	{
		clear();
		for (int i = 0; i < src.len; i++)
			add(src.ptr[i]);
	}

bool smartlong::contains(long what)
	{
		if (len == 0) return false;
		long * look = ptr;
		long * ptrlast = ptr+len-1;
		for (; look <= ptrlast; ++look)
			if (*look == what) return true;
		return false;
	};

long smartlong::FindIndex(long what)
	{
		long * look = ptr;
		long * ptrafter = ptr+len;
		while ( look < ptrafter )
		{
			if (*look == what) return look-ptr;
			++look;
		};
		return -1;
	}

void smartlong::add_unique(long what)
	{
		long * look = ptr;
		for (long k = 0; k < len; k++)
		{
			if (*look == what) return;
			++look;
		};
		// Still here => it was not already in the array.
		add(what);
	}

void smartlong::remove_element( long iWhich )
	{
		if (iWhich >= len) {
			iWhich = iWhich;
		}
		long * look = ptr+iWhich;
		memmove(look, look+1,sizeof(long)*(len-iWhich-1));
		// if len == 4: 0 1 2 3, delete element 2 -> copy 1 element.
		len--;		
		if (len <= 0) {
			printf("Pls don't use remove_element to delte all elemetns. \n");
			getch();
		};
	}

int smartlong::remove_elements( long iStart, long iHowmany)
	{
		int iReturn;
		if (iStart+iHowmany > len) {
			
			memmove(ptr,ptr+(iHowmany+iStart-len),sizeof(long)*(len-iHowmany));
			len-= iHowmany;
			
			iReturn = 0;
		} else {
			long * look = ptr+iStart;
			memmove(look,look+iHowmany,sizeof(long)*(len-iStart-iHowmany));
			len -= iHowmany;
			iReturn = iStart;
		};
		
		if (len <= 0) {
				printf("Pls don't use remove_elements to delte all elemetns. \n");
				getch();
		};
		return iReturn;
		// check this over again.		
	}

smartlong::~smartlong()
	{
		if (ptr != NULL) free(ptr);
	}


Triangle::Triangle()	
	{
		indicator = 0;
	}
	
	
	
	int Triangle::FindNeighbour(Triangle * pTri)
	{
		if (pTri == neighbours[0]) return 0;
		if (pTri == neighbours[1]) return 1;
		if (pTri == neighbours[2]) return 2;
		return -1;
	}
	
	void Triangle::IncrementPeriodic(void)
	{
		++periodic;
		if (periodic == 3) periodic = 0;
	}
	void Triangle::DecrementPeriodic(void)
	{
		--periodic;
		if (periodic < 0) periodic = 2;
	}
	
	void TriMesh::SetTriangleVertex(int iWhichCorner, Triangle * pTri, Vertex * pVertex)
	{
		pTri->cornerptr[iWhichCorner] = pVertex;
		pVertex->AddTriIndex(pTri-T);
	}


bool Triangle::ContainsPointInterior (Vertex * pVert)
{
	if (cornerptr[0] == pVert) return false;
	if (cornerptr[1] == pVert) return false;
	if (cornerptr[2] == pVert) return false;

	return ContainsPoint(pVert->pos.x,pVert->pos.y);
}

// Helper function:
void GetIntercept(const Vector2 & a1,const Vector2 & b1, const Vector2 & a2, const Vector2 & b2,
								Vector2 * pIntercept)
{
	// where does (a1 -> b1) cross (a2 -> b2) ?

	real t1 = ((a1.x-a2.x)*(b2.y-a2.y)-(b2.x-a2.x)*(a1.y-a2.y))/
			((a1.x-b1.x)*(b2.y-a2.y)-(b2.x-a2.x)*(a1.y-b1.y));

	pIntercept->x = a1.x + t1*(b1.x-a1.x);
	pIntercept->y = a1.y + t1*(b1.y-a1.y);
}

/*real GetPossiblyPeriodicDist(Vertex * pVert1, Vertex * pVert2)
{
	real dist1sq,dist2sq,dist3sq,mindistsq;
	Vector2 uL,uR;
	
	uL = Anticlockwise*pVert1->pos;
	uR = Clockwise*pVert1->pos;

	dist1sq = (pVert2->pos.x-uL.x)*(pVert2->pos.x-uL.x)+(pVert2->pos.y-uL.y)*(pVert2->pos.y-uL.y);
	dist2sq = (pVert2->pos.x-pVert1->pos.x)*(pVert2->pos.x-pVert1->pos.x)+(pVert2->pos.y-pVert1->pos.y)*(pVert2->pos.y-pVert1->pos.y);
	dist3sq = (pVert2->pos.x-uR.x)*(pVert2->pos.x-uR.x)+(pVert2->pos.y-uR.y)*(pVert2->pos.y-uR.y);
	
	mindistsq = min(dist1sq,min(dist2sq,dist3sq));
	return sqrt(mindistsq);
}*/ // use GetPossiblyPeriodicDist(pVert1->pos,pVert2->pos);

real CalculateAngle(real x, real y)
{
	static const real TWOPI = 2.0*PI;
	real angle = atan2(y,x);
	if (angle < 0.0) angle += TWOPI;

#ifdef DEBUG
	if (((x > 0.0) && (y > 0.0)) && ((angle > PI*0.5) || (angle < 0.0)))
	{
		x = x;
	};
	if (((x < 0.0) && (y > 0.0)) && ((angle > PI) || (angle < PI*0.5)))
	{
		 x = x;
	};
	if (((x < 0.0) && (y < 0.0)) && ((angle > PI*1.5) || (angle < PI)))
	{
		 x= x;
	};
	if (((x > 0.0) && (y < 0.0)) && ((angle > PI*2.0) || (angle < PI*1.5)))
	{
		 x= x;
	};
#endif
	return angle;
}

real GetPossiblyPeriodicDist(Vector2 & vec1, Vector2 & vec2)
{
	real dist1sq,dist2sq,dist3sq,mindistsq;
	Vector2 uL,uR;
	uL = Anticlockwise*vec1;
	uR = Clockwise*vec1;
	dist1sq = (vec2.x-uL.x)*(vec2.x-uL.x)+(vec2.y-uL.y)*(vec2.y-uL.y);
	dist2sq = (vec2.x-vec1.x)*(vec2.x-vec1.x)+(vec2.y-vec1.y)*(vec2.y-vec1.y);
	dist3sq = (vec2.x-uR.x)*(vec2.x-uR.x)+(vec2.y-uR.y)*(vec2.y-uR.y);
	
	mindistsq = min(dist1sq,min(dist2sq,dist3sq));
	return sqrt(mindistsq);
}
real GetPossiblyPeriodicDistSq(Vector2 & vec1, Vector2 & vec2)
{
	real dist1sq,dist2sq,dist3sq,mindistsq;
	Vector2 uL,uR;
	uL = Anticlockwise*vec1;
	uR = Clockwise*vec1;
	dist1sq = (vec2.x-uL.x)*(vec2.x-uL.x)+(vec2.y-uL.y)*(vec2.y-uL.y);
	dist2sq = (vec2.x-vec1.x)*(vec2.x-vec1.x)+(vec2.y-vec1.y)*(vec2.y-vec1.y);
	dist3sq = (vec2.x-uR.x)*(vec2.x-uR.x)+(vec2.y-uR.y)*(vec2.y-uR.y);
	
	mindistsq = min(dist1sq,min(dist2sq,dist3sq));
	return (mindistsq);
}
/*real GetPossiblyPeriodicDistSq(Vertex * pVert1, Vector2 & u)
{
	real dist1sq,dist2sq,dist3sq,mindistsq;
	Vector2 uL,uR;

	pVert1->periodic_image(uL,0,1);
	pVert1->periodic_image(uR,1,1);
	dist1sq = (u.x-uL.x)*(u.x-uL.x)+(u.y-uL.y)*(u.y-uL.y);
	dist2sq = (u.x-pVert1->pos.x)*(u.x-pVert1->pos.x)+(u.y-pVert1->pos.y)*(u.y-pVert1->pos.y);
	dist3sq = (u.x-uR.x)*(u.x-uR.x)+(u.y-uR.y)*(u.y-uR.y);
	
	mindistsq = min(dist1sq,min(dist2sq,dist3sq));
	return mindistsq;
}

real GetPossiblyPeriodicDistSq(real x1, real y1, real x2, real y2)
{
	real dist1sq,dist2sq,dist3sq,mindistsq;
	Vector2 uL,uR;

	Vector2 u1(x1,y1), u2(x2,y2);
	uL = Anticlockwise*u1;
	uR = Clockwise*u1;
	dist1sq = (uL.x-u2.x)*(uL.x-u2.x)+(u2.y-uL.y)*(u2.y-uL.y);
	dist2sq = (u1.x-u2.x)*(u1.x-u2.x)+(u1.y-u2.y)*(u1.y-u2.y);
	dist3sq = (u2.x-uR.x)*(u2.x-uR.x)+(u2.y-uR.y)*(u2.y-uR.y);
	
	mindistsq = min(dist1sq,min(dist2sq,dist3sq));
	return mindistsq;
}

real GetPossiblyPeriodicDistAcrossTriangle(Triangle * pTri,int which)
{
	int i1,i2;
	real linex,liney,modulus,dist1x,dist1y;
		// to avoid periodic woes, if it's periodic then we map to left(?) and then
		// call again for our temporary triangle
		// (Make sure any pointers internal to Triangle are reset before it goes out of scope!)

	if (pTri->periodic == 1)
	{
		// one point clockwise wrapped
		// unwrap...
		Triangle Tri2;
		Vertex Tempvert;
		Vector2 u;

		i1 = pTri->GetLeftmostIndex(); 
		for (int i = 0; i < 3; i++)
		{
			if (i == i1)
			{
				pTri->cornerptr[i1]->periodic_image(u,1,1);
				Tempvert.x = u.x;
				Tempvert.y = u.y;
				Tri2.cornerptr[i] = &Tempvert;
			} else {
				Tri2.cornerptr[i] = pTri->cornerptr[i];
			};
		};
		Tri2.periodic = 0;

		return GetPossiblyPeriodicDistAcrossTriangle(&Tri2,which);
	};
	if (pTri->periodic == 2)
	{
		// one point not clockwise wrapped
		Triangle Tri2;
		Vertex Tempvert;
		Vector2 u;
		
		i1 = pTri->GetRightmostIndex(); 
		for (int i = 0; i < 3; i++)
		{
			if (i == i1)
			{
				pTri->cornerptr[i1]->periodic_image(u,0,1);
				Tempvert.x = u.x;
				Tempvert.y = u.y;
				Tri2.cornerptr[i] = &Tempvert;
			} else {
				Tri2.cornerptr[i] = pTri->cornerptr[i];
			};
		};
		Tri2.periodic = 0;

		return GetPossiblyPeriodicDistAcrossTriangle(&Tri2,which);
	};


	//// distance across from cornerptr[which]

	//if ((pTri->flags == TRIFLAG_LOWWEDGE) || (pTri->flags == TRIFLAG_HIGHWEDGE))
	//{
	//	// Assume which == 0 or which == 1

	//	i1 = 1-which;
	//	
	//	linex = pTri->cornerptr[i1]->x;
	//	liney = pTri->cornerptr[i1]->y;
	//	
	//	modulus = sqrt(linex*linex+liney*liney);
	//	linex /= modulus;
	//	liney /= modulus;

	//	dist1x = pTri->cornerptr[which]->x-pTri->cornerptr[i1]->x;
	//	dist1y = pTri->cornerptr[which]->y-pTri->cornerptr[i1]->y;
	//		
	//	// project on to (liney,-linex)

	//	return fabs(dist1x*liney - dist1y*linex);
	//};

	// Triangle...
		
	i1 = which+1;
	i2 = which+2;
	if (i1 == 3) i1 = 0;
	if (i2 > 2) i2 -= 3;

	linex = pTri->cornerptr[i1]->x-pTri->cornerptr[i2]->x;
	liney = pTri->cornerptr[i1]->y-pTri->cornerptr[i2]->y;

	modulus = sqrt(linex*linex+liney*liney);
	linex /= modulus;
	liney /= modulus;

	// we want the distance to that line...

	dist1x = pTri->cornerptr[which]->x-pTri->cornerptr[i1]->x;
	dist1y = pTri->cornerptr[which]->y-pTri->cornerptr[i1]->y;
	
	// project on to (liney,-linex)

	return fabs(dist1x*liney - dist1y*linex);

}


real GetSqDistance_SetGlobalFlagNeedPeriodicImage(Vertex * pVertSrc, Vertex * pVert2)
{
	real distx = pVertSrc->x-pVert2->pos.x;
	real disty = pVertSrc->y-pVert2->pos.y;
	if (GlobalPeriodicSearch)
	{
		// in this case we check for Clockwise and anti-Clockwise rotations
		Vector2 u_anti;
		Vector2 u_clock;

		pVertSrc->periodic_image(u_anti,0,1); // Anticlockwise
		pVertSrc->periodic_image(u_clock,1,1);
	
		real distx_anti,disty_anti,distx_clock,disty_clock;
		distx_anti = u_anti.x-pVert2->pos.x;
		disty_anti = u_anti.y-pVert2->pos.y;
		distx_clock = u_clock.x-pVert2->pos.x;
		disty_clock = u_clock.y-pVert2->pos.y;

		real distsqanti = distx_anti*distx_anti+disty_anti*disty_anti;
		real distsq0 = distx*distx+disty*disty;
		real distsqclock = distx_clock*distx_clock+disty_clock*disty_clock;
		
		// If we find that Clockwise is nearest, set a flag on VertSrc
		// If we find that Anticlockwise is nearest, set a flag on VertSrc
			
		if (distsq0 < distsqanti)
		{
			if (distsq0 < distsqclock) {
				return distsq0;
			} else {
				//pVertSrc->flags |= VERTFLAGS_INFLUENCE_Anticlockwise;
				GlobalFlagNeedPeriodicImage = true;
				return distsqclock;
			};
		} else {
			if (distsqanti < distsqclock) {
				//pVertSrc->flags |= VERTFLAGS_INFLUENCE_CLOCKWISE;
				GlobalFlagNeedPeriodicImage = true;
				return distsqanti;
			} else { 
				//pVertSrc->flags |= VERTFLAGS_INFLUENCE_Anticlockwise; // this vertex was Clockwise rotated.
				GlobalFlagNeedPeriodicImage = true;
				return distsqclock;
			};
		};
	};
	return distx*distx+disty*disty;
}

/*real GetSqDistance_SetPerInfluenceFlagOnVertex_Full(Vertex * pVertSrc, Vertex * pVert2, real * pRetDistx, real * pRetDisty)
{
	real distx = pVertSrc->x-pVert2->pos.x;
	real disty = pVertSrc->y-pVert2->pos.y;
	if (GlobalPeriodicSearch)
	{
		// in this case we check for Clockwise and anti-Clockwise rotations
		Vector2 u_anti;
		Vector2 u_clock;

		pVertSrc->periodic_image(u_anti,0,1); // Anticlockwise
		pVertSrc->periodic_image(u_clock,1,1);
	
		real distx_anti,disty_anti,distx_clock,disty_clock;
		distx_anti = u_anti.x-pVert2->pos.x;
		disty_anti = u_anti.y-pVert2->pos.y;
		distx_clock = u_clock.x-pVert2->pos.x;
		disty_clock = u_clock.y-pVert2->pos.y;

		real distsqanti = distx_anti*distx_anti+disty_anti*disty_anti;
		real distsq0 = distx*distx+disty*disty;
		real distsqclock = distx_clock*distx_clock+disty_clock*disty_clock;
		
		// If we find that Clockwise is nearest, set a flag on VertSrc
		// If we find that Anticlockwise is nearest, set a flag on VertSrc
			
		if (distsq0 < distsqanti)
		{
			if (distsq0 < distsqclock) {
				*pRetDistx = distx;
				*pRetDisty = disty;
				return distsq0;
			} else {
				pVertSrc->flags |= VERTFLAGS_INFLUENCE_Anticlockwise;
				*pRetDistx = distx_clock;
				*pRetDisty = disty_clock;
				return distsqclock;
			};
		} else {
			if (distsqanti < distsqclock) {
				pVertSrc->flags |= VERTFLAGS_INFLUENCE_CLOCKWISE;
				*pRetDistx = distx_anti;
				*pRetDisty = disty_anti;
				return distsqanti;
			} else { 
				pVertSrc->flags |= VERTFLAGS_INFLUENCE_Anticlockwise; // this vertex was Clockwise rotated.
				*pRetDistx = distx_clock;
				*pRetDisty = disty_clock;
				return distsqclock;
			};
		};
	};
	*pRetDistx = distx;
	*pRetDisty = disty;
	return distx*distx+disty*disty;
}
*/

int sgn(real x)
{
	if (x > 0.0) return 1;
	return -1;
};

int GetNumberSharedVertices(Triangle & tri1, Triangle & tri2)
{
	int match;
	int matches = 0;
	for (int i = 0; i < 3; i++)
	{
		match = 0;
		for (int j = 0; j < 3; j++)
			if (tri1.cornerptr[i] == tri2.cornerptr[j])
				match = 1;
		if (match == 1) matches++;
	};
	return matches;
}

real Triangle::ReturnAngle(Vertex * pVertex)
{
	Vector2 v1,v2,u[3];
	real dotproduct_over_moduli,weight;
	static const real TWOPI = 2.0*PI;

	MapLeftIfNecessary(u[0],u[1],u[2]);

	if (pVertex == cornerptr[0]) {
		v1 = u[1]-u[0];
		v2 = u[2]-u[0];				
	} else {
		if (pVertex == cornerptr[1]) {
			v1 = u[0]-u[1];
			v2 = u[2]-u[1];
		} else {
			v1 = u[0]-u[2];
			v2 = u[1]-u[2];
		};
	};

	dotproduct_over_moduli = (v1.x*v2.x+v1.y*v2.y)/
				sqrt((v1.x*v1.x+v1.y*v1.y)*(v2.x*v2.x+v2.y*v2.y));

	weight = acos(dotproduct_over_moduli)/TWOPI;
	return weight;
}


Vector2 Triangle::RecalculateCentroid()
{
	Vector2 u[3];
	MapLeftIfNecessary(u[0],u[1],u[2]);
	cent = (u[0]+u[1]+u[2])/3.0;

	if (u8domain_flag == CROSSING_INS)
	{
		// Modify the centre to be the centre of the intersection of insulator
		GetCentreOfIntersectionWithInsulator(cent);
	}
	if (u8domain_flag == OUTER_FRILL) {
		Vector2 temp = 0.5*(u[0]+u[1]); // ? compare to GPU
		temp.project_to_radius(cent, FRILL_CENTROID_OUTER_RADIUS);
	};
	if (u8domain_flag == INNER_FRILL) {
		Vector2 temp = 0.5*(u[0]+u[1]); // ? compare to GPU
		temp.project_to_radius(cent, FRILL_CENTROID_INNER_RADIUS);
	};
	return cent;
}


Vector2 Triangle::GetContiguousCent_AssumingCentroidsSet(Vertex * pVertex)
{
	if (periodic == 0)	return cent;
	// It is assumed that pVertex is one of the corners.
	if (pVertex->pos.x < 0.0) return cent;
	return Clockwise*cent;
}

void TriMesh::Recalculate_TriCentroids_VertexCellAreas_And_Centroids()
{
	ConvexPolygon cp;
	Triangle * pTri;
	Vertex * pVertex;
	long iVertex, iTri;
	Vector2 u;
	int i;

	// 1. Reset triangle centroids.

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateCentroid();
		++pTri; // this seems like it should still work if we have not wrapped any vertex that moved, even if tri no longer periodic in truth but some pts outside tranche
	};

	// 2. Reset vertex cell areas.

	long izTri[128];
	long tri_len;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		tri_len = pVertex->GetTriIndexArray(izTri);
		cp.Clear();

		if ((pVertex->flags == CONCAVE_EDGE_VERTEX) ||
			(pVertex->flags == CONVEX_EDGE_VERTEX) )
		{
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				u = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
				if (u.x*u.x+u.y*u.y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
					u.project_to_radius(u,INNER_A_BOUNDARY);
				if (u.x*u.x+u.y*u.y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
					u.project_to_radius(u,DOMAIN_OUTER_RADIUS);
				cp.add(u);				
			};
			
			/*
			// Project to a radius ...
			pTri = T + izTri[0];
			u = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
			if (pVertex->flags == INNERMOST) {
				u.project_to_radius(u,INNER_A_BOUNDARY);
			} else {
				u.project_to_radius(u,DOMAIN_OUTER_RADIUS);
			};
			cp.add(u);
			
			// Outermost should project to DOMAIN_OUTER_RADIUS.
			// what if innermost should project also?
			// Innermost is on INNERMOST_A_BOUNDARY.
			
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
			};
			
			u = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
			if (pVertex->flags == INNERMOST) {
				u.project_to_radius(u,INNER_A_BOUNDARY);
			} else {
				u.project_to_radius(u,DOMAIN_OUTER_RADIUS);
			};
			cp.add(u);
			*/
		} else {
		
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
			};
		};

		pVertex->AreaCell = cp.GetArea();
		pVertex->centroid = cp.CalculateBarycenter();

		
		//if (iVertex == 36685) {
		//	printf("vertex %d flag %d \n",iVertex,pVertex->flags);
		//	for (i = 0; i < cp.numCoords; i++)
		//		printf("%1.5E %1.5E ... %1.5E \n",cp.coord[i].x,cp.coord[i].y,
		//									cp.coord[i].modulus());
		//	printf("\n\n");
		//	
		//	for (i = 0; i < tri_len; i++)
		//	{
		//		pTri = T + izTri[i];
		//		u = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
		//		printf("%1.5E %1.5E ... %1.5E \n",u.x,u.y,u.modulus());			
		//	};

		//	getch();
		//}
		//
		++pVertex;
	};
}


void Triangle::MapLeftIfNecessary(Vector2 & u0, Vector2 & u1, Vector2 & u2) const
{	
	PopulatePositions(u0,u1,u2);
	
	if (periodic == 1)
	{
		int o1 = GetLeftmostIndex(); 
		if (o1 != 0) u0 = Anticlockwise*u0;
		if (o1 != 1) u1 = Anticlockwise*u1;
		if (o1 != 2) u2 = Anticlockwise*u2;
		return;
	};
	if (periodic == 2)
	{
		int o1 = GetRightmostIndex();
		if (o1 == 0) u0 = Anticlockwise*u0;
		if (o1 == 1) u1 = Anticlockwise*u1;
		if (o1 == 2) u2 = Anticlockwise*u2;	
		return;
	};
}

real Triangle::GetDomainIntersectionAreaROC(Vector2 u[3],int iWhichMove,Vector2 ROC)
{
	// Call once for each moving corner to get total ROC area.
	// ROC is the rate of change of position u[iWhichMove] which is in the domain.
	bool bDomain[3];
	int iDomain, iWhich, iWhich1, iWhich2;
	Vector2 intercept1, intercept2, ROCintercept1, ROCintercept2,
		dArea_by_d_top, dArea_by_d1,dArea_by_d2;
	real shoelace;

	bDomain[0] = (cornerptr[0]->flags == DOMAIN_VERTEX)?1:0;
	bDomain[1] = (cornerptr[1]->flags == DOMAIN_VERTEX)?1:0;
	bDomain[2] = (cornerptr[2]->flags == DOMAIN_VERTEX)?1:0;

	iDomain = bDomain[0]+bDomain[1]+bDomain[2];

	if (iDomain == 1) {
		if (bDomain[iWhichMove] != 1) {
			printf("dodginesse\n");
			getch();
		} 

		iWhich = 0; while (bDomain[iWhich] == 0) iWhich++;
		iWhich1 = iWhich-1; if (iWhich1 == -1) iWhich1 = 2;
		iWhich2 = iWhich+1; if (iWhich2 == 3) iWhich2 = 0;

		GetInsulatorIntercept(&intercept1,u[iWhich1],u[iWhich]);
		GetInsulatorIntercept(&intercept2,u[iWhich2],u[iWhich]);

		Get_ROC_InsulatorIntercept(&ROCintercept1,u[iWhich1],u[iWhich],ROC);
		Get_ROC_InsulatorIntercept(&ROCintercept2,u[iWhich2],u[iWhich],ROC);

		// cp.GetArea contents:

		// for (i = 0; i < numCoords-1; i++)
		//		area += coord[i].x*coord[i+1].y - coord[i+1].x*coord[i].y;
		// area += coord[i].x*coord[0].y - coord[0].x*coord[i].y;
		// return fabs(area*0.5);

		// ROCArea = sum_i[top & intercepts] dArea/dx_i . dx_i/dt

		// establish which way round shoelace is positive:
		shoelace = u[iWhich].x*u[iWhich2].y - u[iWhich2].x*u[iWhich].y
					  + u[iWhich2].x*u[iWhich1].y - u[iWhich1].x*u[iWhich2].y
					  + u[iWhich1].x*u[iWhich].y - u[iWhich].x*u[iWhich1].y;
		real sign = 1.0;
		if (shoelace < 0.0) sign = -1.0;
		// area = 0.5*sign* that shoelace.
		
		dArea_by_d_top.x = 0.5*sign*(u[iWhich2].y - u[iWhich1].y);
		dArea_by_d_top.y = 0.5*sign*(u[iWhich1].x - u[iWhich2].x);
		dArea_by_d1.x = 0.5*sign*(u[iWhich].y-u[iWhich2].y);
		dArea_by_d1.y = 0.5*sign*(u[iWhich2].x-u[iWhich].x);
		dArea_by_d2.x = 0.5*sign*(u[iWhich1].y-u[iWhich].y);
		dArea_by_d2.y = 0.5*sign*(u[iWhich].x-u[iWhich1].x);

		real answer = dArea_by_d1.dot(ROCintercept1) + dArea_by_d2.dot(ROCintercept2)
						+ dArea_by_d_top.dot(ROC);
		return answer;
	};

	// We consider one corner moving at a time. 

	iWhich = 0; while (bDomain[iWhich] == 1) iWhich++;
	iWhich1 = 0; while (bDomain[iWhich1] == 0) iWhich1++;
	iWhich2 = iWhich1+1; while (bDomain[iWhich2] == 0) iWhich2++;

	GetInsulatorIntercept(&intercept1,u[iWhich1],u[iWhich]);
	GetInsulatorIntercept(&intercept2,u[iWhich2],u[iWhich]);

	if (iWhichMove == iWhich1) {

		Get_ROC_InsulatorIntercept(&ROCintercept1,u[iWhich],u[iWhich1],ROC);
		
		shoelace = u[iWhich1].x*intercept1.y - intercept1.x*u[iWhich1].y
			     + intercept1.x*intercept2.y - intercept2.x*intercept1.y
				 + intercept2.x*u[iWhich2].y - u[iWhich2].x*intercept2.y
				 + u[iWhich2].x*u[iWhich1].y - u[iWhich1].x*u[iWhich2].y;
		real sign = 1.0;
		if (shoelace < 0.0) sign = -1.0;

		dArea_by_d_top.x = 0.5*sign*(intercept1.y - u[iWhich2].y);
		dArea_by_d_top.y = 0.5*sign*(u[iWhich2].x - intercept1.x);
		dArea_by_d1.x = 0.5*sign*(intercept2.y - u[iWhich1].x);
		dArea_by_d1.y = 0.5*sign*(u[iWhich1].y - intercept2.y);

		real answer = dArea_by_d_top.dot(ROC) + dArea_by_d1.dot(ROCintercept1);
		return answer;
	} else {

		Get_ROC_InsulatorIntercept(&ROCintercept2,u[iWhich],u[iWhich2],ROC);

		shoelace = u[iWhich1].x*intercept1.y - intercept1.x*u[iWhich1].y
			     + intercept1.x*intercept2.y - intercept2.x*intercept1.y
				 + intercept2.x*u[iWhich2].y - u[iWhich2].x*intercept2.y
				 + u[iWhich2].x*u[iWhich1].y - u[iWhich1].x*u[iWhich2].y;
		real sign = 1.0;
		if (shoelace < 0.0) sign = -1.0;

		dArea_by_d_top.x = 0.5*sign*(u[iWhich1].y - intercept2.y);
		dArea_by_d_top.y = 0.5*sign*(intercept2.x - u[iWhich2].x);
		dArea_by_d2.x = 0.5*sign*(u[iWhich2].y - intercept1.y);
		dArea_by_d2.y = 0.5*sign*(intercept1.x - u[iWhich2].x);
		
		real answer = dArea_by_d_top.dot(ROC) + dArea_by_d2.dot(ROCintercept2);
		return answer;
	};
}
real Triangle::GetDomainIntersectionArea(bool bUseOwnCoords, Vector2 u[3]) const
{
	ConvexPolygon cp;
	int iDomain, iWhich, iWhich1, iWhich2;
	int bDomain[3];
	Vector2 intercept1, intercept2;

	if (u8domain_flag == OUT_OF_DOMAIN) return 0.0;
	if (u8domain_flag == DOMAIN_TRIANGLE) return this->GetArea();

	if (bUseOwnCoords) MapLeftIfNecessary(u[0],u[1],u[2]); // This gives for the original triangle.

	bDomain[0] = (cornerptr[0]->flags == DOMAIN_VERTEX)?1:0;
	bDomain[1] = (cornerptr[1]->flags == DOMAIN_VERTEX)?1:0;
	bDomain[2] = (cornerptr[2]->flags == DOMAIN_VERTEX)?1:0;

	iDomain = bDomain[0]+bDomain[1]+bDomain[2];

	if (iDomain == 1) {
		iWhich = 0; while (bDomain[iWhich] == 0) iWhich++;
		iWhich1 = 0; while (bDomain[iWhich1] == 1) iWhich1++;
		iWhich2 = iWhich1+1; while (bDomain[iWhich2] == 1) iWhich2++;

		GetInsulatorIntercept(&intercept1,u[iWhich1],u[iWhich]);
		GetInsulatorIntercept(&intercept2,u[iWhich2],u[iWhich]);

		cp.Clear();
		cp.add(intercept1);
		cp.add(u[iWhich]);
		cp.add(intercept2);

		return cp.GetArea();
	};
	if (iDomain != 2) {
		printf("Error in GetDomainIntersectionArea.\n");
		return 0.0;
	};

	iWhich = 0; while (bDomain[iWhich] == 1) iWhich++;
	iWhich1 = 0; while (bDomain[iWhich1] == 0) iWhich1++;
	iWhich2 = iWhich1+1; while (bDomain[iWhich2] == 0) iWhich2++;
	// iWhich1 shall go next to intercept1 in the sequence.

	GetInsulatorIntercept(&intercept1,u[iWhich1],u[iWhich]);
	GetInsulatorIntercept(&intercept2,u[iWhich2],u[iWhich]);

	cp.Clear();
	cp.add(intercept1);
	cp.add(u[iWhich1]);
	cp.add(u[iWhich2]);
	cp.add(intercept2);

	return cp.GetArea();
}

void Triangle::GuessPeriodic(void)
{
	real ratio0,ratio1,ratio2,gradient;
	
	ratio0 = cornerptr[0]->pos.x/cornerptr[0]->pos.y;
	ratio1 = cornerptr[1]->pos.x/cornerptr[1]->pos.y;
	ratio2 = cornerptr[2]->pos.x/cornerptr[2]->pos.y;
	gradient = GRADIENT_X_PER_Y/2.0;

	periodic = 0;

	if (ratio0 > gradient)
	{
		// number periodic is the number of others that are < -GRADIENT_X_PER_Y/3.0	
		if (ratio1 < -gradient)
			++periodic;
		if (ratio2 < -gradient)
			++periodic;
	} else {
		if (ratio1 > gradient)
		{
			if (ratio0 < -gradient)
				++periodic;
			if (ratio2 < -gradient)
				++periodic;
		} else {
			if (ratio2 > gradient)
			{
				if (ratio0 < -gradient)
					++periodic;
				if (ratio1 < -gradient)
					++periodic;
			};
		};
	};	
}

void Triangle::RecalculateEdgeNormalVectors(bool normalise)
{
	int iPrev,iNext;
	Vector2 u[3];

	MapLeftIfNecessary(u[0],u[1],u[2]);

	for (int i = 0; i < 3; i++)
	{
		iPrev = i-1; if (iPrev < 0) iPrev = 2;
		iNext = i+1; if (iNext > 2) iNext = 0;			
		edge_normal[i].x = u[iNext].y-u[iPrev].y;
		edge_normal[i].y = u[iPrev].x-u[iNext].x;
		if (edge_normal[i].dot(u[i]-u[iPrev]) > 0.0)
		{
			// facing the wrong way - should face away from u[i]
			edge_normal[i].x = -edge_normal[i].x;
			edge_normal[i].y = -edge_normal[i].y;
		};
		if (normalise) edge_normal[i].Normalise();
		// NOTE: if normalise == false then the length of edge_normal is the side length -- quite convenient
	};
	// Same code will work even if looking out of the domain.
}


// Better if we make this part of some prototypical base class.
// OR, AuxTriangles just are Triangles. Why not?
/*void AuxTriangle::GuessPeriodic(void)
{
	real ratio0,ratio1,ratio2,gradient;
		
	ratio0 = cornerptr[0]->pos.x/cornerptr[0]->pos.y;
	ratio1 = cornerptr[1]->pos.x/cornerptr[1]->pos.y;
	ratio2 = cornerptr[2]->pos.x/cornerptr[2]->pos.y;
	gradient = GRADIENT_X_PER_Y/2.0;

	periodic = 0;

	if (ratio0 > gradient)
	{
			// number periodic is the number of others that are < -GRADIENT_X_PER_Y/3.0	
		if (ratio1 < -gradient)
			++periodic;
		if (ratio2 < -gradient)
			++periodic;
	} else {
		if (ratio1 > gradient)
		{
			if (ratio0 < -gradient)
				++periodic;
			if (ratio2 < -gradient)
				++periodic;
		} else {
			if (ratio2 > gradient)
			{
				if (ratio0 < -gradient)
					++periodic;
				if (ratio1 < -gradient)
					++periodic;
			};
		};
	};
}


void AuxTriangle::RecalculateEdgeNormalVectors(bool normalise)
{
	// copy of function below
	// !
	int iPrev,iNext;
	Vector2 u[3];

	if (periodic == 0)
	{
		this->PopulatePositions(u[0],u[1],u[2]);
	} else {
		this->MapLeft(u[0],u[1],u[2]);
	};

	for (int i = 0; i < 3; i++)
	{
		iPrev = i-1; if (iPrev < 0) iPrev = 2;
		iNext = i+1; if (iNext > 2) iNext = 0;			
		edge_normal[i].x = u[iNext].y-u[iPrev].y;
		edge_normal[i].y = u[iPrev].x-u[iNext].x;
		if (edge_normal[i].dot(u[i]-u[iPrev]) > 0.0)
		{
			// facing the wrong way - should face away from u[i]
			edge_normal[i].x = -edge_normal[i].x;
			edge_normal[i].y = -edge_normal[i].y;
		};

		if (normalise) edge_normal[i].Normalise();
		// NOTE: if normalise == false then the length of edge_normal is the side length -- quite convenient
	};
	// Same code will work even if looking out of the domain.
}
*/



// unnecessary as far as I know:

/*void AuxTriangle::RecalculateEdgeNormalVectors(bool normalise)
{
	// in CUDA version we will only use edge_normal and get rid of transvec stuff
	
	int iPrev,iNext;
	Vector2 u[3];

	if (periodic == 0)
	{
		this->PopulatePositions(u[0],u[1],u[2]);
	} else {
		this->MapLeft(u[0],u[1],u[2]);
	};

	for (int i = 0; i < 3; i++)
	{
		iPrev = i-1; if (iPrev < 0) iPrev = 2;
		iNext = i+1; if (iNext > 2) iNext = 0;
			
		edge_normal[i].x = u[iNext].y-u[iPrev].y;
		edge_normal[i].y = u[iPrev].x-u[iNext].x;

		if (edge_normal[i].dot(u[i]-u[iPrev]) > 0.0)
		{
			// facing the wrong way - should face away from u[i]
			edge_normal[i].x = -edge_normal[i].x;
			edge_normal[i].y = -edge_normal[i].y;
		};

		if (normalise) edge_normal[i].Normalise();
		// NOTE: if normalise == false then the length of edge_normal is the side length -- quite convenient
	};
	// Same code will work even if looking out of the domain.
}

*/
/*real Triangle::GetShortArea()
{
	Vector2 u0,u1,u2;
	Vector2 u0dash, u1dash;
	real u0mod, u1mod;

	if (flags != 2)
	{
		printf("bad call.\n");
		getch();
	};

	// place u0dash, u1dash at projected coordinates NOTIONAL_DISTANCE further out.

	if (periodic == 0) {
		PopulatePositions(u0,u1,u2);
	} else {
		MapLeft(u0,u1,u2);
	};

	u0mod = u0.modulus();
	u1mod = u1.modulus();
	u0dash = ((u0mod+NOTIONAL_DISTANCE)/u0mod)*u0;
	u1dash = ((u0mod+NOTIONAL_DISTANCE)/u1mod)*u1;

	// shoelace:
	
	//return 0.5*fabs( u0.x*u1.y - u1.x*u0.y
	//						+ u1.x*u2.y - u2.x*u1.y
	//						+ u2.x*u0.y - u0.x*u2.y);

	return 0.5*fabs( u0.x*u0dash.y - u0dash.x*u0.y
							 + u0dash.x*u1dash.y - u1dash.x*u0dash.y
							 + u1dash.x*u1.y - u1.x*u1dash.y
							 + u1.x*u0.y - u0.x*u1.y);
};*/

/*real Triangle::GetNormalDistance(int opp)
{
/*	// Requires that transvec be already set correctly.
	real diffx,diffy;
	// This is only for the non-periodic case -- we need to allow for periodic
	real transmod = sqrt(transvecx[opp]*transvecx[opp]+transvecy[opp]*transvecy[opp]);
	real transhatx = transvecx[opp]/transmod;
	real transhaty = transvecy[opp]/transmod;


	// But how to know if transhat faces in or out??


	// we're not guaranteed this is same as transhat start.
	int which = opp-1;
	if (which < 0) which = 2;


	if (periodic == 0)
	{
		
		// difference dot with trans hat vector = normal distance
		diffx = cornerptr[opp]->x-cornerptr[which]->x;
		diffy = cornerptr[opp]->y-cornerptr[which]->y;

	} else {

		// note that transvec is calculated by moving all to the left.
		
		// difference dot with trans hat vector = normal distance
		Vector2 v_opp, v_which;		

		cornerptr[opp]->periodic_image(v_opp,0);
		cornerptr[which]->periodic_image(v_which,0);

		diffx = v_opp.x-v_which.x;
		diffy = v_opp.y-v_which.y;
	};

	return diffx*transhatx + diffy*transhaty;*/

	// Better to do differently:

		// Pythagoras: 
	
	// side1^2 = x^2 + y^2
	// side2^2 = (a-x)^2 + y^2
	// we want y, we do not know x, we know a & side1,side2.
	
	// side2^2 - side1^2 = a^2 - 2ax
	
	// x = (s2^2 - s1^2 - a^2)/(-2a)
	// y = sqrt(s1^2 - x^2)


	// PERIODIC CASE?
	// Normal distance may need to be given correctly!!!


	// ->proceed to populate 3 Vector2's based on flags, periodic and
	// then do the Pythagoras normal distance calculation.

/*
	Vector2 u1,u2,uO;

	if (flags == 0)
	{
	
		int c1 = opp-1;
		if (c1 < 0) c1 = 2;
		int c2 = opp+1;
		if (c2 > 2) c2 = 0;

		if (periodic == 0)
		{
			cornerptr[c1]->PopulatePosition(u1);
			cornerptr[c2]->PopulatePosition(u2);
			cornerptr[opp]->PopulatePosition(uO);
		} else {
			cornerptr[c1]->periodic_image(u1,0);
			cornerptr[c2]->periodic_image(u2,0);
			cornerptr[opp]->periodic_image(uO,0);
		};
	} else {
		if (flags == 1)
		{
			// low wedge

			// in this case, normal distance is construed relative to
			// the sides and top 

			// X Don't do -->>>  just assume periodic to make code shorter.
			
			if (periodic > 0)
			{
				
				if (opp == 0)
				{
					// side goes from 1 to ins

					cornerptr[1]->periodic_image(u1,0);
					cornerptr[1]->project_to_ins_periodic(u2,0);
					cornerptr[0]->periodic_image(uO,0);

				} else {
					if (opp == 1)
					{
						// side goes from 0 to ins

						cornerptr[0]->periodic_image(u1,0);
						cornerptr[0]->project_to_ins_periodic(u2,0);
						cornerptr[1]->periodic_image(uO,0);
					} else {

						// side goes from 0 to 1

						cornerptr[0]->periodic_image(u1,0);
						cornerptr[1]->periodic_image(u2,0);
						cornerptr[0]->project_to_ins_periodic(uO,0);
					};
				};
			} else {
				if (opp == 0)
				{
					// side goes from 1 to ins

					cornerptr[1]->PopulatePosition(u1);
					cornerptr[1]->project_to_ins(u2);
					cornerptr[0]->PopulatePosition(uO);

				} else {
					if (opp == 1)
					{
						// side goes from 0 to ins

						cornerptr[0]->PopulatePosition(u1);
						cornerptr[0]->project_to_ins(u2);
						cornerptr[1]->PopulatePosition(uO);
					} else {

						// side goes from 0 to 1

						cornerptr[0]->PopulatePosition(u1);
						cornerptr[1]->PopulatePosition(u2);
						cornerptr[0]->project_to_ins(uO);
					};
				};
			};
		} else {
			if (periodic > 0)
			{
				// high wedge
				if (opp == 0)
				{
					// side goes from 1 to ins

					cornerptr[1]->periodic_image(u1,0);
					cornerptr[1]->project_to_100cm_periodic(u2,0);
					cornerptr[0]->periodic_image(uO,0);

				
				} else {
					if (opp == 1)
					{
						// side goes from 0 to ins

						cornerptr[0]->periodic_image(u1,0);
						cornerptr[0]->project_to_100cm_periodic(u2,0);
						cornerptr[1]->periodic_image(uO,0);
					} else {

						// side goes from 0 to 1

						cornerptr[0]->periodic_image(u1,0);
						cornerptr[1]->periodic_image(u2,0);
						cornerptr[0]->project_to_100cm_periodic(uO,0);
					};
				};
			} else {
				// high wedge
				if (opp == 0)
				{
					// side goes from 1 to ins

					cornerptr[1]->PopulatePosition(u1);
					cornerptr[1]->project_to_100cm(u2);
					cornerptr[0]->PopulatePosition(uO);

				
				} else {
					if (opp == 1)
					{
						// side goes from 0 to ins

						cornerptr[0]->PopulatePosition(u1);
						cornerptr[0]->project_to_100cm(u2);
						cornerptr[1]->PopulatePosition(uO);
					} else {

						// side goes from 0 to 1

						cornerptr[0]->PopulatePosition(u1);
						cornerptr[1]->PopulatePosition(u2);
						cornerptr[0]->project_to_100cm(uO);
					};
				};
			};
		};
	};

	
	real dist1sq = (u1.x-uO.x)*(u1.x-uO.x)+(u1.y-uO.y)*(u1.y-uO.y);
	real dist2sq = (u2.x-uO.x)*(u2.x-uO.x)+(u2.y-uO.y)*(u2.y-uO.y);
	real distasq = (u1.x-u2.x)*(u1.x-u2.x)+(u1.y-u2.y)*(u1.y-u2.y);;
		
	real x = (dist2sq - dist1sq - distasq)/(-2.0*sqrt(distasq));
	real y = sqrt(dist1sq - x*x);
	return y;
	
}*/
	
/*real Vertex::CalculateVoronoiArea()
{
	// Voronoi area is found assuming that ... circumcenters were already calculated.
	// (!)
	// and stored as pTri->numerator_x,y
	Triangle * pTri;
	ConvexPolygon cp;
	Vector2 circumcenter, cc;
	real theta;
	int i,j,k;
	real angle[100];
	int index[100];
	Proto * tempptr[100];
	
	if (triangles.len >= 100)
	{
		printf("\ncannot do it - static array not big enough\n");
		getch();
	};

	// First got to sort the triangles:
	for (i = 0; i < triangles.len; i++)
	{
		pTri = (Triangle *)(triangles.ptr[i]);
		
		pTri->ReturnCentre(&cc,this); 
		// For now do this lazy and inefficient trig way:
		theta = CalculateAngle(cc.x-x,cc.y-y);		
		j = 0;
		while ((j < i) && (theta > angle[j])) j++; // if i == 1 then we can only move up to place 1, since we have 1 element already
		if (j < i) {
			// move the rest of them forward in the list:
			for (k = i; k > j ; k--)
			{
				index[k] = index[k-1];
				angle[k] = angle[k-1];
			};
		}
		angle[j] = theta;
		index[j] = i;
	};			
	for (i = 0; i < triangles.len; i++)
		tempptr[i] = triangles.ptr[index[i]];
	for (i = 0; i < triangles.len; i++)
		triangles.ptr[i] = tempptr[i];			


	for (i = 0; i < triangles.len; i++)
	{
		pTri = (Triangle *)(triangles.ptr[i]);
		circumcenter.x = pTri->numerator_x;
		circumcenter.y = pTri->numerator_y;

		if ((pTri->periodic > 0) && (x > 0.0))
		{
			// for a per tri, the circumcenter is found for left image.
			// if our point on the right, circumcenter will need to be mapped over.
			circumcenter = Clockwise*circumcenter;
		};
		cp.add(circumcenter);
	};

	return cp.GetArea();
}

*/
void Triangle::ReturnPositionOtherSharedVertex_conts_tranche(Triangle * pTri, Vertex * pVert, Vector2 * pResult)
{
	// First find the common vertex that is not pVert :
	int iShared;

	if (pVert == cornerptr[0])
	{
		if (   (pTri->cornerptr[0] == cornerptr[1])
			|| (pTri->cornerptr[1] == cornerptr[1])
			|| (pTri->cornerptr[2] == cornerptr[1]) )
		{
			//cornerptr[1] is it
			iShared = 1;
		} else {
			iShared = 2; 
			
			// DEBUG:
			if ((pTri->cornerptr[0] != cornerptr[2])
			&& (pTri->cornerptr[1] != cornerptr[2])
			&& (pTri->cornerptr[2] != cornerptr[2]) )
			{
				printf("!JDdewjiw!\n"); getch();
			};		
		};
	} else {
		if (pVert == cornerptr[1]) 
		{
			if ((pTri->cornerptr[0] == cornerptr[0])
			|| (pTri->cornerptr[1] == cornerptr[0])
			|| (pTri->cornerptr[2] == cornerptr[0]) )
			{
				iShared = 0;
			} else {
				iShared = 2; 

				// DEBUG:
				if (   (pTri->cornerptr[0] != cornerptr[2])
				&& (pTri->cornerptr[1] != cornerptr[2])
				&& (pTri->cornerptr[2] != cornerptr[2]) )
				{
					printf("!JDdewjiw!\n"); getch();
				};		
			};
		} else {
			// pVert == cornerptr[2]
			if (   (pTri->cornerptr[0] == cornerptr[0])
			|| (pTri->cornerptr[1] == cornerptr[0])
			|| (pTri->cornerptr[2] == cornerptr[0]) )
			{
				iShared = 0;
			} else {
				iShared = 1; 

				// DEBUG:
				if (   (pTri->cornerptr[0] != cornerptr[1])
				&& (pTri->cornerptr[1] != cornerptr[1])
				&& (pTri->cornerptr[2] != cornerptr[1]) )
				{
					printf("!JDdewjiw!\n"); getch();
				};		
			};
		};
	};

	// Populate position with same wrapping as pVert .
	// If the vertex is INS_VERT or HIGH_VERT -- as it may be -- then populate by projection.

	//if (cornerptr[iShared] == INS_VERT)
	//{
	//	pVert->project_to_ins(*pResult);
	//} else {
	//	if (cornerptr[iShared] == HIGH_VERT)
	//	{
	//		pVert->project_to_radius(*pResult, HIGH_WEDGE_OUTER_RADIUS);
	//	} else {
	if (periodic == 0)
	{
		// usual case:
		*pResult = cornerptr[iShared]->pos;
	} else {
		// periodic; not at inner or outer boundary
		*pResult = cornerptr[iShared]->pos;

		int iVert = 0;
		while (pVert != cornerptr[iVert]) iVert++;
		if (periodic == 1)
		{
			int o = GetLeftmostIndex();
			// o is the wrapped point
			if (o == iVert)
			{
				// want to wrap the other point anticlockwise:				
				*pResult = Anticlockwise*(*pResult);
			} else {
				if (o == iShared) {
					// unwrap the other point to clockwise:
					*pResult = Clockwise*(*pResult);					
				};
			};
		} else {
			int o = GetRightmostIndex();
			if (o == iVert)
			{
				// unwrap the other point clockwise:
				*pResult = Clockwise*(*pResult);	
			} else {
				if (o == iShared) 
				{
					// wrap it anticlockwise:
					*pResult = Anticlockwise*(*pResult);
				};
			};
		};
	};
	//	};
	//};
	//	

}

Vertex * Triangle::ReturnOtherSharedVertex(Triangle * pTri,Vertex * pVertex)
{
	if ((cornerptr[0] != pVertex) && (pTri->has_vertex(cornerptr[0]))) return cornerptr[0];
	if ((cornerptr[1] != pVertex) && (pTri->has_vertex(cornerptr[1]))) return cornerptr[1];
	return cornerptr[2];
}
/*
AuxVertex * AuxTriangle::ReturnUnsharedVertex(AuxTriangle * pTri2, int * pwhich)
{
	
	if (   (pTri2->cornerptr[0] == cornerptr[0])
			|| (pTri2->cornerptr[1] == cornerptr[0])
			|| (pTri2->cornerptr[2] == cornerptr[0]) )
	{
		// it's not vertices[0]
		if (   (pTri2->cornerptr[0] == cornerptr[1])
			|| (pTri2->cornerptr[1] == cornerptr[1])
			|| (pTri2->cornerptr[2] == cornerptr[1]) )
		{
			if (pwhich != 0) *pwhich = 2;
			return cornerptr[2];  // which might well be 0 
		} else {
			if (pwhich != 0) *pwhich = 1;
			return cornerptr[1];
		};
	} else {
		if (pwhich != 0) *pwhich = 0;
		return cornerptr[0];
	};
}
*/

Vertex * Triangle::ReturnUnsharedVertex(Triangle * pTri2, int * pwhich) // pwhich = 0
{
	// test each one in turn.

	//if (flags == 0)
	//{

	// in case of wedge, we do want to return 0 if it is not sharing either of the top ones.

	if (   (pTri2->cornerptr[0] == cornerptr[0])
			|| (pTri2->cornerptr[1] == cornerptr[0])
			|| (pTri2->cornerptr[2] == cornerptr[0]) )
	{
		// it's not vertices[0]

		if (   (pTri2->cornerptr[0] == cornerptr[1])
			|| (pTri2->cornerptr[1] == cornerptr[1])
			|| (pTri2->cornerptr[2] == cornerptr[1]) )
		{
			if (pwhich != 0) *pwhich = 2;
			return cornerptr[2];  // which might well be 0 
		} else {
			if (pwhich != 0) *pwhich = 1;
			return cornerptr[1];
		};
	} else {
		if (pwhich != 0) *pwhich = 0;
		return cornerptr[0];
	};
	//} else {
	//	// Only test vertex 0 and 1.

	//	if (   (pTri2->cornerptr[0] == cornerptr[0])
	//		|| (pTri2->cornerptr[1] == cornerptr[0])
	//		|| (pTri2->cornerptr[2] == cornerptr[0]) )
	//	{
	//		// it's not vertices[0]

	//		if (pwhich != 0) *pwhich = 1;
	//		return cornerptr[1];
	//		
	//	} else {
	//		
	//		if (pwhich != 0) *pwhich = 0;
	//		return cornerptr[0];
	//	};
	//};
}


//int Triangle::DecodeSign(int other)
//{
//	switch(other)
//	{
//	case 0:
//		return ((sign_other_dot_transvec & TRIFLAG_SIGN0) > 0)?1:-1;
//	case 1:
//		return ((sign_other_dot_transvec & TRIFLAG_SIGN1) > 0)?1:-1;
//	case 2:
//		return ((sign_other_dot_transvec & TRIFLAG_SIGN2) > 0)?1:-1;
//	}
//	return 0; // just to suppress warning
//}

// double-precision overload:


// NEW TESTAGAINSTEDGE FUNCTIONS:

/*
int Triangle::TestAgainstEdge(float x,float y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  Triangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

	Vector2 u1;//, u2, uO;
	bool outside;
//	Vector2 edge;
	Vector2 transverse;
	//long Tindex;
	real x_dot_transverse;//,other_dot_transverse;
	
	if (periodic == 0)
	{
		cornerptr[c1]->PopulatePosition(u1);
	} else {
		// ensure c1 is mapped to left if need be:
		if (periodic == 1)
		{
			int iMapped = GetLeftmostIndex();
			if (iMapped == c1)
			{
				cornerptr[c1]->PopulatePosition(u1);
			} else {
				cornerptr[c1]->periodic_image(u1,0,1); 
			};
		} else {
			int iUnmapped = GetRightmostIndex();
			if (iUnmapped == c1)
			{
				cornerptr[c1]->periodic_image(u1,0,1); 
			} else {
				cornerptr[c1]->PopulatePosition(u1);
			};
		};
	};

	x_dot_transverse = (x-u1.x)*transvecx[other] + (y-u1.y)*transvecy[other];
		
	outside = (sgn(x_dot_transverse) == DecodeSign(other))?0:1;

	if (outside)
	{
		*ppNeigh = neighbours[other]; // neighbours is now always a valid value.

		return 1;
	};
	return 0;
}
	*/

int Triangle::TestAgainstEdge(real x,real y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  Triangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

	Vector2 u1;//, u2, uO;
	bool outside;
	Vector2 transverse;
	real x_dot_transverse;
	
	u1 = cornerptr[c1]->pos;
	if (periodic == 0)
	{
	} else {
		// ensure c1 is mapped to left if need be:
		if (periodic == 1)
		{
			int iMapped = GetLeftmostIndex();
			if (iMapped != c1)
				u1 = Anticlockwise*u1;			
		} else {
			int iUnmapped = GetRightmostIndex();
			if (iUnmapped == c1)
				u1 = Anticlockwise*u1;
		};
	};

	// Seems this routine assumes that we can test against a left-mapped periodic triangle.

	x_dot_transverse = (x-u1.x)*edge_normal[other].x + (y-u1.y)*edge_normal[other].y;
		
	outside = (x_dot_transverse > 0.0)?1:0;	// edge_normal points outside

	if (outside)
	{
		*ppNeigh = neighbours[other]; // neighbours is now always a valid value.
		if ((neighbours[other]->u8domain_flag == OUTER_FRILL) ||
			(neighbours[other]->u8domain_flag == INNER_FRILL))
			*ppNeigh = this;
		return 1;
	};
	return 0;
}
/*
int Triangle::TestAgainstEdge(real x,real y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  Triangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

		Vector2 u1;//, u2, uO;
	bool outside;
//	Vector2 edge;
	Vector2 transverse;
//	long Tindex;
	real x_dot_transverse;//,other_dot_transverse;
	
	if (periodic == 0)
	{
		cornerptr[c1]->PopulatePosition(u1);
	} else {
		// ensure c1 is mapped to left if need be:
		if (periodic == 1)
		{
			int iMapped = GetLeftmostIndex();
			if (iMapped == c1)
			{
				cornerptr[c1]->PopulatePosition(u1);
			} else {
				cornerptr[c1]->periodic_image(u1,0,1); 
			};
		} else {
			int iUnmapped = GetRightmostIndex();
			if (iUnmapped == c1)
			{
				cornerptr[c1]->periodic_image(u1,0,1); 
			} else {
				cornerptr[c1]->PopulatePosition(u1);
			};
		};
	};

	x_dot_transverse = (x-u1.x)*transvecx[other] + (y-u1.y)*transvecy[other];
		
	outside = (sgn(x_dot_transverse) == DecodeSign(other))?0:1;

	if (outside)
	{
		*ppNeigh = neighbours[other]; // neighbours is now always a valid value.

		return 1;
	};
	return 0;
}
*/

/*



int Triangle::TestAgainstEdge(float x,float y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  Triangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

	Vector2 u1;//, u2, uO;
	bool outside;
//	Vector2 edge;
	Vector2 transverse;
	long Tindex;
	real x_dot_transverse,other_dot_transverse;
	
	
	if (periodic == 0)
	{
		cornerptr[c1]->PopulatePosition(u1);
		//pTri->cornerptr[c2]->PopulatePosition(u2);
		//pTri->cornerptr[other]->PopulatePosition(uO);

		// test whether we are on the inside side of edge [0]<->[1]:
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
			// transverse:
//		transverse.x = (u1.y-u2.y);//-edge.y;
//		transverse.y = (u2.x-u1.x);//edge.x; 
//		x_dot_transverse = (x-u1.x)*transverse.x + (y-u1.y)*transverse.y;

		x_dot_transverse = (x-u1.x)*transvecx[other] + (y-u1.y)*transvecy[other];
		
		// now take the vector headed to the other point, to compare:
		//edge.x = uO.x-u1.x;
		//edge.y = uO.y-u1.y;

//		other_dot_transverse = //edge.x*transverse.x + edge.y*transverse.y;
//							(uO.x-u1.x)*transverse.x + (uO.y-u1.y)*transverse.y;

		// we're outside the triangle in the case that these are not the same sign.
		outside = (sgn(x_dot_transverse) == DecodeSign(other))?0:1;

		if (outside)
		{
			// This should work OK for wedges, I think. [2] should send us north.

			*ppNeigh = neighbours[other]; // neighbours is now always a valid value.

	//		Tindex = neighbours[other];
	//		if (Tindex >= 0)
	//		{
	//			*piNeigh = Tindex;
	//			return 1;
	//		} else {
				// if that is the only boundary it's outside then stick with this triangle...
	//			*piNeigh = -2; // this triangle not known
				// -1 indicates outside hopefully the upper boundary
	//			return 0;
	//		};
			return 1;
		};
		return 0;
	} else {
		int side = (x > 0.0f)?1:0;
		Vector2 u2,uO;
		int c2 = other-1; if (c2 == -1) c2 = 2;

		cornerptr[c1]->periodic_image(u1,side);
		cornerptr[c2]->periodic_image(u2,side);
		cornerptr[other]->periodic_image(uO,side);
			
		// test whether we are on the inside side of edge [0]<->[1]:
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
			// the vector that runs along this edge:
			// this is why Vertex should have contained a Float2

			// transverse:
		transverse.x = (u1.y-u2.y);//-edge.y;
		transverse.y = (u2.x-u1.x);//edge.x; 
		x_dot_transverse = (x-u1.x)*transverse.x + (y-u1.y)*transverse.y;
		// now take the vector headed to the other point, to compare:
		//edge.x = uO.x-u1.x;
		//edge.y = uO.y-u1.y;
		other_dot_transverse = //edge.x*transverse.x + edge.y*transverse.y;
							(uO.x-u1.x)*transverse.x + (uO.y-u1.y)*transverse.y;
		// we're outside the triangle in the case that these are not the same sign.
		outside = (sgn(x_dot_transverse) == sgn(other_dot_transverse))?0:1;

		if (outside)
		{
			*ppNeigh = neighbours[other];
			return 1;

			////Tindex = pTri->neighbours[other];
			////if (Tindex >= 0)
			////{
			////	*piNeigh = Tindex;
			////	return 1;
			////} else {
			////	// if that is the only boundary it's outside then stick with this triangle...
			////	*piNeigh = -2; // this triangle not known
			////	// -1 indicates outside hopefully the upper boundary

			////	// Thing is, that means we're allowing this triangle's domain to be a spray
			////	// Oh well to that.

			////	return 0;
			////};
		};
		return 0;
	};
}

int Triangle::TestAgainstEdge(real x,real y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  Triangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

	Vector2 u1;//, u2, uO;
	bool outside;
//	Vector2 edge;
	Vector2 transverse;
	long Tindex;
	real x_dot_transverse,other_dot_transverse;

	if (periodic == 0)
	{
		cornerptr[c1]->PopulatePosition(u1);
		
		x_dot_transverse = (x-u1.x)*transvecx[other] + (y-u1.y)*transvecy[other];
		
		//outside = (sgn(x_dot_transverse) == DecodeSign(other))?false:true;

	} else {
		// But should we even be doing this? 
		
		// Note that the recorded transverse sign is set up by using everything mapped to left side.
		// That should be fine?

//		int side = (x > 0.0)?1:0;
		Vector2 u2,uO;

		// In periodic case, we have set up transvec etc for the left-mapped cell.
		// Therefore we are willing to compare multiple periodic images of (x,y) to this cell.
		// If x > 0 however then we might as well just compare the rotated (x,y); if x < 0 just compare (x,y)

		cornerptr[c1]->periodic_image(u1,0);

		Vector2 u;
			
		if (x > 0)
		{
			Vertex temp;
			temp.x = x;
			temp.y = y;
			temp.periodic_image(u,0); 
		} else {
			u.x = x;
			u.y = y;
		};

		x_dot_transverse = (u.x-u1.x)*transvecx[other] + (u.y-u1.y)*transvecy[other];

//		int c2 = 0; while ((c2 == other) || (c2 == c1)) c2++;

//		cornerptr[c1]->periodic_image(u1,side);
//		cornerptr[c2]->periodic_image(u2,side);
		// Nothing to stop c2 from being 0 here .

		// The following brazen function call gives an error because cornerptr[2] might be ==0.
		//cornerptr[other]->periodic_image(uO,side);
		
		// test whether we are on the inside side of edge [0]<->[1]:
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
			// transverse:
//		transverse.x = (u1.y-u2.y);//-edge.y;
//		transverse.y = (u2.x-u1.x);//edge.x; 
//		x_dot_transverse = (x-u1.x)*transverse.x + (y-u1.y)*transverse.y;
//		other_dot_transverse = //edge.x*transverse.x + edge.y*transverse.y;
//							(uO.x-u1.x)*transverse.x + (uO.y-u1.y)*transverse.y;
//		outside = (sgn(x_dot_transverse) == sgn(other_dot_transverse))?0:1;

	};

	
	outside = (sgn(x_dot_transverse) == DecodeSign(other))?false:true;

	if (outside)
	{
		*ppNeigh = neighbours[other]; // neighbours is now always a valid value.
		return 1;
	};
	return 0;
}


*/


Triangle * TriMesh::ReturnPointerToOtherSharedTriangle(
		Vertex * pVert,
		Vertex * pOther,
		Triangle * p_not_this_one, int iLevel)
{

	// Ask what tri contains pVert and pOther that is not p_not_this_one
	// Usually used for setting neighbours.

	long tri_len, izTri[128];
	Triangle *pTri,*Tarray;
	long iNot;

	if (iLevel == -1) {
		Tarray = T;
	} else {
		Tarray = AuxT[iLevel];
	};
	iNot = p_not_this_one-Tarray;

	tri_len = pVert->GetTriIndexArray(izTri);
	for (int i = 0; i < tri_len; i++)
	{
		if (izTri[i] != iNot)
		{
			pTri = Tarray + izTri[i];
			// does this triangle also contain otherpoint?
			if (   (pTri->cornerptr[0] == pOther)
				|| (pTri->cornerptr[1] == pOther)
				|| (pTri->cornerptr[2] == pOther) ) // note that since cornerptr[2] == 0 for wedge, this should not cause a problem.
				return pTri;
		}; 
	};
	
	// got here -> no triangle found

	return p_not_this_one;
	// Return itself instead.
}


/*
AuxTriangle * TriMesh::ReturnPointerToOtherSharedTriangle(
		AuxVertex * pVert,
		AuxVertex * pOther,
		AuxTriangle * p_not_this_one)
{

	// Ask what tri contains pVert and pOther that is not p_not_this_one
	// Usually used for setting neighbours.

	// Note that if pOther == 0 then the question is just,
	// what tri contains pVert and is a wedge -
	// in this case p_not_this_one should also be a wedge so that
	// the answer can be unique.
	
	// If pOther == 0 then of course, cornerptr[2] == 0 will test for wedgeness anyway.

//	AuxTriangle ** ptr = (Triangle **)pVert->triangles.ptr;

	AuxTriangle * ptr;
	for (int i = 0; i < pVert->tri_len; i++)
	{
		ptr = InnerT + pVert->iTriangles[i];
		if (ptr != p_not_this_one)
		{
			// does this triangle also contain otherpoint?
			if (   (ptr->cornerptr[0] == pOther)
				|| (ptr->cornerptr[1] == pOther)
				|| (ptr->cornerptr[2] == pOther) ) // note that since cornerptr[2] == 0 for wedge, this should not cause a problem.
				return ptr;
		}; 
	};
	
	return 0;
}*/
/*
AuxTriangle * TriMesh::ReturnPointerToOtherSharedTriangleAux(
		AuxVertex * pVert,
		AuxVertex * pOther,
		AuxTriangle * p_not_this_one,
		int iLevel)
{

	// Ask what tri contains pVert and pOther that is not p_not_this_one
	// Usually used for setting neighbours.

	AuxTriangle * ptr;
	for (int i = 0; i < pVert->tri_len; i++)
	{
		ptr = AuxT[iLevel] + pVert->iTriangles[i];
		if (ptr != p_not_this_one)
		{
			// does this triangle also contain otherpoint?
			if (   (ptr->cornerptr[0] == pOther)
				|| (ptr->cornerptr[1] == pOther)
				|| (ptr->cornerptr[2] == pOther) ) // note that since cornerptr[2] == 0 for wedge, this should not cause a problem.
				return ptr;
		}; 
	};
	
	return p_not_this_one; // default if another neighbour sharing the edge not found.
}
*/

/*Triangle * TriMesh::SearchCornerptr(long index0, long index1, long index2, Triangle * pTriSeed)
{
	// Is pTriSeed, it?
	int test[3];

	Triangle * pTri = pTriSeed;

	test[0] = pTri->cornerptr[0]-X;
	test[1] = pTri->cornerptr[1]-X;
	test[2] = pTri->cornerptr[2]-X;
	
	if (((test[0] == index0) || (test[1] == index0) || (test[2] == index0))
		&&
		((test[0] == index1) || (test[1] == index1) || (test[2] == index1))
		&&
		((test[0] == index2) || (test[1] == index2) || (test[2] == index2)))
		return pTri;
	
	// No? Then search all tris of X+index0

	Vertex * pVertex = X + index0;
	int i;
	for (i = 0; i < pVertex->triangles.len; i++)
	{
		pTri = (Triangle *)(pVertex->triangles.ptr[i]);

		test[0] = pTri->cornerptr[0]-X;
		test[1] = pTri->cornerptr[1]-X;
		test[2] = pTri->cornerptr[2]-X;
		
		if (((test[0] == index0) || (test[1] == index0) || (test[2] == index0))
			&&
			((test[0] == index1) || (test[1] == index1) || (test[2] == index1))
			&&
			((test[0] == index2) || (test[1] == index2) || (test[2] == index2)))
			return pTri;
	}

	return 0; // triangle with these 3 did not exist!
}


int AuxTriangle::TestAgainstEdge(real x,real y, 
							int c1, // the "start" of the relevant edge
							  int other, // the point opposite the relevant edge
							  AuxTriangle ** ppNeigh)
{
	// returns 1 in the event that (x,y) is outside the triangle.

	Vector2 u1;//, u2, uO;
	bool outside;
	Vector2 transverse;
	real x_dot_transverse;
	
	if (periodic == 0)
	{
		cornerptr[c1]->PopulatePosition(u1);
	} else {
		// ensure c1 is mapped to left if need be:
		if (periodic == 1)
		{
			int iMapped = GetLeftmostIndex();
			if (iMapped == c1)
			{
				cornerptr[c1]->PopulatePosition(u1);
			} else {
				cornerptr[c1]->periodic_image(u1,0,1); 
			};
		} else {
			int iUnmapped = GetRightmostIndex();
			if (iUnmapped == c1)
			{
				cornerptr[c1]->periodic_image(u1,0,1); 
			} else {
				cornerptr[c1]->PopulatePosition(u1);
			};
		};
	};

	x_dot_transverse = (x-u1.x)*edge_normal[other] .x+ (y-u1.y)*edge_normal[other].y;
	outside = (x_dot_transverse > 0.0)?1:0;	// edge_normal points outside
	//(sgn(x_dot_transverse) == DecodeSign(other))?0:1;

	if (outside)
	{
		*ppNeigh = neighbours[other]; // neighbours is now always a valid value.

		return 1;
	};
	return 0;
}

bool AuxTriangle::ContainsPoint(real x, real y)
{
	// Note that this is returning true in the case that it is 
	// only outside on the side where neighbours[i] == this.
	// That is a very liberal test for being inside.
	// We ought to probably put a limit on what can be considered azimuthally to belong to this one.	
	AuxTriangle * pNeigh;

	// always test for our triangle unmapped:
	int out = 0;
	if (TestAgainstEdge(x,y,
								0,       // edge corner
								 2,       // the opposite point
								 &pNeigh    // neighbour in this direction, if it's outside this way
								 ) && (neighbours[2] != this))
	{
		out = 1;
	} else {
		if (TestAgainstEdge(x,y, 1, 0, &pNeigh) && (neighbours[0] != this))
		{
			out = 1;
		} else {
			if (TestAgainstEdge(x,y, 0, 1, &pNeigh) && (neighbours[1] != this))
				out = 1;
		};
	};
	
	if (periodic == 0) return (1-out);
	if (out == 0) return (1-out); // found point already in left interpreted tri
	
	// if periodic > 0, we want to also test RH
	int out2 = 0;
	real destx,desty;
	// map point Anticlockwise to represent mapping triangle Clockwise:
	destx = Anticlockwise.xx*x + Anticlockwise.xy*y;
	desty = Anticlockwise.yx*x + Anticlockwise.yy*y;
	
	if ((TestAgainstEdge(destx,desty,0,2,&pNeigh)) && (neighbours[2] != this))
		return false;
	if ((TestAgainstEdge(destx,desty, 1, 0, &pNeigh)) && (neighbours[0] != this))
		return false;
	if ((TestAgainstEdge(destx,desty, 0, 1, &pNeigh)) && (neighbours[1] != this))
		return false;
	return true;		
	
	// Note that this is returning true in the case that it is 
	// only outside on the side where neighbours[i] == this.
	// That is a very liberal test for being inside.
	// We ought to probably put a limit on what can be considered azimuthally to belong to this one.	
}


*/

bool Triangle::ContainsPoint(real x, real y)
{
	Triangle * pNeigh;

	// always test for our triangle unmapped:
	int out = 0;
	if(TestAgainstEdge(x,y,
								0,       // edge corner
								 2,       // the opposite point
								 &pNeigh    // neighbour in this direction, if it's outside this way
								 ))
	{
		out = 1;
	} else {
		if(TestAgainstEdge(x,y, 1, 0, &pNeigh))
		{
			out = 1;
		} else {
			if(TestAgainstEdge(x,y, 0, 1, &pNeigh)) 
				out = 1;
		};
	};

	if (periodic == 0) return (1-out);
	if (out == 0) return (1-out); // found point already in left interpreted tri

	// if periodic > 0, we want to also test RH
	int out2 = 0;
	real destx,desty;
	// map point Anticlockwise to represent mapping triangle Clockwise:
	destx = Anticlockwise.xx*x + Anticlockwise.xy*y;
	desty = Anticlockwise.yx*x + Anticlockwise.yy*y;

	if(TestAgainstEdge(destx,desty,0,2,&pNeigh))
	{
		out2 = 1;
	} else {
		if(TestAgainstEdge(destx,desty, 1, 0, &pNeigh))
		{
			out2 = 1;
		} else {
			if(TestAgainstEdge(destx,desty, 0, 1, &pNeigh)) // do not send vertex 2 here in case it does not exist.
				out2 = 1;
		};
	};
	return (1-out2);
}

// same as above function basically but now periodic lives only on left
bool Triangle::TestAgainstEdges(real x,real y, Triangle ** ppNeigh)
{
	// If an edge triangle, we require it give preference to a neighbour other than itself. 
//	static real const FP_FUZZY_THRESH_LARGE = 1.0e-8;
	// ^^  more sophistication called for. !


	// Two things to concern with it seems:
	
	// If point is outside edge of memory, it should test positive that it is
	// NOT in this triangle... there are no holes in the domain though.
	// Return self in this case. ??
	// Then look at calls to this function to see how to handle that.
	// Maybe change function return type to int, return a flag for Out Of Memory Domain.

	// However we must FIRST test against the other edges. This rules out that
	// the point in question belongs to the memory domain at all.

	if ((periodic != 0) && (x > 0.0)) // in this case test anticlock image of x, which had to be within domain tranche to begin with.
	{
		real newx = Anticlockwise.xx*x+Anticlockwise.xy*y;
		real newy = Anticlockwise.yx*x+Anticlockwise.yy*y;
		x = newx; y = newy;

		// Prioritize looking left.
		if ((cornerptr[2]->pos.x > 0.0) && (cornerptr[1]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		if ((cornerptr[0]->pos.x > 0.0) && (cornerptr[2]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		if ((cornerptr[1]->pos.x > 0.0) && (cornerptr[0]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
	
		// Second favourite: stay within periodic.
		if ((cornerptr[2]->pos.x > 0.0) || (cornerptr[1]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		if ((cornerptr[0]->pos.x > 0.0) || (cornerptr[2]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		if ((cornerptr[1]->pos.x > 0.0) || (cornerptr[0]->pos.x > 0.0))
			if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
	
		// Got here: we'll have to exit to the left side of domain then.
		if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
	}; 
	// Idea: We need to prioritize NOT CROSSING to the x < 0 side if the target x > 0.

	// New attempt:

	if ((neighbours[0]->u8domain_flag == OUTER_FRILL) || (neighbours[0]->u8domain_flag == INNER_FRILL)
		|| (neighbours[0] == this))
	{
		// In this case:
		// prioritize neighbours[1] and [2].

		if(TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		if(TestAgainstEdge(x,y,	0,        // edge corner
								2,        // the opposite point - ie which edge
								ppNeigh)) // neighbour in this direction, if it's outside this way
			return 1;

		if(TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		return 0; // inside *this
	}
	if ((neighbours[1]->u8domain_flag == OUTER_FRILL) || (neighbours[1]->u8domain_flag == INNER_FRILL)
		|| (neighbours[1] == this))
	{
		if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		return 0;
	}
	// Just changed the order of tests.

	/*
	if (neighbours[0] == this) {
		
		if(TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		if(TestAgainstEdge(x,y,	0,       // edge corner
								 2,       // the opposite point - ie which edge
									 ppNeigh))    // neighbour in this direction, if it's outside this way
			return 1;

		if(TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		return 0; // inside *this
	}
	if (neighbours[1] == this) {
		if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
		if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
		return 0;
	}
	*/
	
	if ((periodic != 0) && (x > 0.0))
	
	if (TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
	if (TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
	if (TestAgainstEdge(x,y, 0, 2, ppNeigh)) return 1;
	
	return 0;
	


	// .
	// .

	// If point is within insulator, we should return the correct triangle.
	// But how that is handled, in the case of placement?


	//if (u8EdgeFlag > 0)
	//{
	//	int c1,c2;
	//	// Find not base corner:
	//	int iBase = 0; while (cornerptr[iBase]->flags > 3) iBase++;

	//	c1 = iBase+1; if (c1 == 3) c1 = 0;
	//	c2 = c1+1; if (c2 == 3) c2 = 0;

	//	if(TestAgainstEdge(x,y,c1,c2,       // the opposite point
	//								 ppNeigh ))   // neighbour in this direction, if it's outside this way
	//		return 1;

	//	if (TestAgainstEdge(x,y,c2,c1,       // the opposite point
	//								 ppNeigh ))   // neighbour in this direction, if it's outside this way
	//		return 1;

	//	// finally test against inner edge .... but maybe wish to return 0
	//	if (TestAgainstEdge(x,y,c1,iBase,       // the opposite point
	//								 ppNeigh ))   // neighbour in this direction, if it's outside this way
	//	{
	//		// ask if it is azimuthally compatible. If it is, we can return 0.
	//		Vector2 u[3];
	//		PopulatePositions(u[0],u[1],u[2]);
	//		if (periodic) {
	//			if (x > 0.0) { MapRight(u[0],u[1],u[2]); } else {MapLeft(u[0],u[1],u[2]); };
	//		};
	//		real grad1 = u[c1].x/u[c1].y; real grad2 = u[c2].x/u[c2].y;
	//		real grad = x/y;

	//		// realise we can afford to be quite liberal here
	//		// It failed to test as outside either of the other two sides
	//		// The only options: we are nowhere near, or this really is it.
	//		// If we are nowhere near , the following comparator will be quite large, so:
	//		if ((grad-grad1)*(grad-grad2) <= FP_FUZZY_THRESH_LARGE) return 0;
	//		return 1; 
	//		// outside azimuthally and on edge of domain therefore returning 1 with *ppNeigh as itself.
	//	}
	//	return 0;

	//} else {
	//	if(TestAgainstEdge(x,y,
	//								0,       // edge corner
	//								 2,       // the opposite point
	//								 ppNeigh    // neighbour in this direction, if it's outside this way
	//								 ))
	//		return 1;
	//	if(TestAgainstEdge(x,y, 1, 0, ppNeigh)) return 1;
	//	if(TestAgainstEdge(x,y, 0, 1, ppNeigh)) return 1;
	//	return 0;
	//};
}			

// same as above function basically
/*bool Triangle::TestAgainstEdges(float x,float y, Triangle ** ppNeigh)
{
	int out = 0;
	if(TestAgainstEdge(x,y,
								0,       // edge corner
								 2,       // the opposite point
								 ppNeigh    // neighbour in this direction, if it's outside this way
								 ))
	{
		out = 1;
	} else {
		if(TestAgainstEdge(x,y, 1, 0, ppNeigh))
		{
			out = 1;
		} else {
		if(TestAgainstEdge(x,y, 0, 1, ppNeigh)) // do not send vertex 2 here in case it does not exist.
			out = 1;
		};
	};
	return out;	
}			
*/

// Now here we implement routines to calculate triangle intersections:


#define DYDX   0
#define DXDY   1

void GetIntersection(Vector2 * result,const Vector2 & x0,real gradient,int flagdydx, Vector2 & a, Vector2 & b)
{
	real x,y;
	// where is line a->b cut by the line that starts at start and has gradient gradient,

	// DEBUG:
	if (!(_finite(a.x) && _finite(a.y) && _finite(b.x) && _finite(b.y)))
	{
		a.x = a.x;
	}

	if (flagdydx == DYDX)
	{
		// on first line, 
		// x = x0.x + t
		// y = x0.y + t dy/dx
		// x - y/ dy/dx = x0.x - x0.y / dy/dx

		// y = dy/dx(x - x0.x) + x0.y 

		// on second line
		// x = a.x + t(b.x-a.x)
		// y = a.y + t(b.y-a.y)

		// (b.x-a.x)y - x(b.y-a.y) = (b.x-a.x)a.y - (b.y-a.y)a.x
		
		// For both to be true simultaneously?

		// (b.x-a.x)(dy/dx(x - x0.x) + x0.y ) - x(b.y-a.y) = (b.x-a.x)a.y - (b.y-a.y)a.x

		// ((b.x - a.x)dy/dx - (b.y-a.y) ) x =  (b.x-a.x)a.y - (b.y-a.y)a.x + (b.x-a.x)dy/dx x0.x -  (b.x-a.x)x0.y
		
		// looks too complicated?

		x = ((b.x-a.x)*(a.y + gradient*x0.x - x0.y) - (b.y-a.y)*a.x)/
			((b.x - a.x)*gradient - (b.y-a.y));
		y = x0.y + gradient*(x - x0.x);

		// DEBUG:

		// Test that (x,y) is actually a solution:


	} else {

		// x = dx/dy (y- x0.y) + x0.x

		// (b.x-a.x)y - x(b.y-a.y) = (b.x-a.x)a.y - (b.y-a.y)a.x

		// (b.x-a.x)y - (dx/dy (y- x0.y) + x0.x)(b.y-a.y) = (b.x-a.x)a.y - (b.y-a.y)a.x

		// (b.x-a.x - dx/dy (b.y-a.y)) y = (b.x-a.x)a.y - (b.y-a.y)a.x + (x0.x - dx/dy x0.y) (b.y-a.y)

		y = ((b.x-a.x)*a.y + (x0.x - gradient*x0.y - a.x)*(b.y-a.y))/
			(b.x-a.x - gradient * (b.y-a.y));
		x = gradient*(y-x0.y) + x0.x;

	};
	result->x = x;
	result->y = y;
}

void Triangle::CalculateCircumcenter(Vector2 & cc, real * pdistsq)
{
	Vector2 Bb,C,b,c,a;
	
	MapLeftIfNecessary(a,b,c);
	
	Bb = b-a;
	C = c-a;		
	real D = 2.0*(Bb.x*C.y-Bb.y*C.x);
	real modB = Bb.x*Bb.x+Bb.y*Bb.y;
	real modC = C.x*C.x+C.y*C.y;
	cc.x = (C.y*modB-Bb.y*modC)/D + a.x;
	cc.y = (Bb.x*modC-C.x*modB)/D + a.y;

	*pdistsq = (a.x-cc.x)*(a.x-cc.x)+(a.y-cc.y)*(a.y-cc.y); // why?
}

void GetInsulatorIntercept(Vector2 *result, const Vector2 & x1, const Vector2 & x2)
{
	// find where line x1->x2 crosses r = DEVICE_RADIUS_INSULATOR_OUTER

	// x = x1.x + t(x2.x-x1.x)
	// y = x1.y + t(x2.y-x1.y)
	// x^2+y^2 = c^2

	// (x1.x + t(x2.x-x1.x))^2 + (x1.y + t(x2.y-x1.y))^2 = c^2

	// or, y = x1.y + dy/dx (x-x1.x)

	// x^2 + (x1.y - dy/dx x1.x + dy/dx x)^2 = c^2


	// (x1.x + t(x2.x-x1.x))^2 + (x1.y + t(x2.y-x1.y))^2 = c^2

	// t^2 ( (x2.x-x1.x)^2 + (x2.y - x1.y)^2 ) + 2t (x1.x (x2.x-x1.x) + x1.y (x2.y-x1.y) )
	//    + x1.x^2 + x1.y^2 = c^2
	// t^2 + 2t ( -- ) / (-- ) = (c^2 - x1.x^2 - x1.y^2)/ (-- )
	
	real den = (x2.x-x1.x)*(x2.x-x1.x) + (x2.y - x1.y)*(x2.y - x1.y) ;
	real a = (x1.x * (x2.x-x1.x) + x1.y * (x2.y-x1.y) ) / den;

	// (t + a)^2 - a^2 = (  c^2 - x1.x^2 - x1.y^2  )/den
	
	real root = sqrt( (DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER
							- x1.x*x1.x - x1.y*x1.y)/den + a*a ) ;
	
	real t1 = root - a;
	real t2 = -root - a;
	
	// since this is a sufficient condition to satisfy the circle, this probably means that
	// the other solution is on the other side of the circle.
	
	// Which root is within x1, x2 ? Remember x2 would be t = 1.

	if (t1 > 1.0) 
	{
		if ((t2 < 0.0) || (t2 > 1.0))
		{	
			// This usually means one of the points actually is on the curve.

			real dist1 = min(fabs(t1-1.0),fabs(t1));
			real dist2 = min(fabs(t2-1.0),fabs(t2));

			if (dist1 < dist2)
			{
				// use t1				
				if (dist1 > 0.00000001)
				{
					printf("\n\nError.\n"); 
					getch();
				};				
				result->x = x1.x + t1*(x2.x-x1.x);
				result->y = x1.y + t1*(x2.y-x1.y);
			} else {
				// use t2				
				if (dist2 > 0.00000001)
				{
					printf("\n\nError.\n"); 
					printf("t1 = %1.10E , \nt2 = %1.10E , \nx1.x= %1.10E ,\nx1.y = %1.10E ,\nx2.x = %1.10E ,\nx2.y = %1.10E\n",
								t1,t2,x1.x,x1.y,x2.x,x2.y);
					getch();
				};				
				result->x = x1.x + t2*(x2.x-x1.x);
				result->y = x1.y + t2*(x2.y-x1.y);
			};
		} else {		
			// use t2:		
			result->x = x1.x + t2*(x2.x-x1.x);
			result->y = x1.y + t2*(x2.y-x1.y);
		};
	} else {
		if (t1 < -1.0e-13) 
		{	
			printf("\n\nError.KL\n"); 
			printf("t1 = %1.10E , \nt2 = %1.10E , \nx1.x= %1.10E ,\nx1.y = %1.10E ,\nx2.x = %1.10E ,\nx2.y = %1.10E\n",
				t1,t2,x1.x,x1.y,x2.x,x2.y);
			getch(); 
		};
		result->x = x1.x + t1*(x2.x-x1.x);
		result->y = x1.y + t1*(x2.y-x1.y);		
	};
#ifdef DEBUG
	if (result->x*result->x + result->y*result->y > 1.000001*DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
	{
		result = result;
	};
	if (result->y < 0.0)
	{
		result = result;
	};
#endif
}
void Get_ROC_InsulatorIntercept(Vector2 * pROCintercept1,
								Vector2 lower , Vector2 moving,Vector2 ROC)
{
	// A rough estimate might do.	
	// Moving away directly from lower does not change the intercept;
	// moving perpendicularly does.

	// Do empirically:
	Vector2 interceptplus, interceptminus;

	real length = (moving-lower).modulus();
	real ROClength = ROC.modulus();
	real ROCfactor = length*0.0001/ROClength;
	Vector2 ROCmove = ROC*ROCfactor;
	Vector2 plus = moving + ROCmove;
	Vector2 minus = moving - ROCmove;

	GetInsulatorIntercept(&interceptplus, lower, plus);
	GetInsulatorIntercept(&interceptminus, lower, minus);

	Vector2 derivative = (interceptplus-interceptminus)/(2.0*ROCfactor);
	*pROCintercept1 = derivative;
}

int Triangle::GetCentreOfIntersectionWithInsulator(Vector2 & cc)
{

	// where this triangle crosses r=3.44,
	// we want to return the middle of that arc.

	// 3 lines; should give 2 intercepts of 3.44
	// If not, failed.

	real azimuth01,azimuth12,azimuth02;
	real r0sq, r1sq, r2sq, Rsq;
	int number_of_intercepts;
	real angle;
	Vector2 u[3];
	Vector2 intercept;

	MapLeftIfNecessary(u[0],u[1],u[2]);

	r0sq = u[0].dot(u[0]);
	r1sq = u[1].dot(u[1]);
	r2sq = u[2].dot(u[2]);
	Rsq = DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER;

	number_of_intercepts = 0;
	azimuth01 = 0.0;
	azimuth12 = 0.0;
	azimuth02 = 0.0;

	// 0-1:

	if ((r0sq-Rsq)*(r1sq-Rsq) < 0.0)
	{
		number_of_intercepts++;

		GetInsulatorIntercept(&intercept,u[0],u[1]);
		azimuth01 = atan2(intercept.y,intercept.x);
	};
	// 1-2:

	if ((r2sq-Rsq)*(r1sq-Rsq) < 0.0)
	{
		number_of_intercepts++;

		GetInsulatorIntercept(&intercept,u[1],u[2]);
		azimuth12 = atan2(intercept.y,intercept.x);
	};
	// 0-2:

	if ((r2sq-Rsq)*(r0sq-Rsq) < 0.0)
	{
		number_of_intercepts++;

		GetInsulatorIntercept(&intercept,u[0],u[2]);
		azimuth02 = atan2(intercept.y,intercept.x);
	};

	if (number_of_intercepts != 2) 
	{
		printf("intercept fail\n"); getch();

		return 1;
	};

	angle = 0.5*(azimuth01+azimuth12+azimuth02);
	//if (angle < -HALFANGLE+PI*0.5) angle += FULLANGLE;
	//if (angle > HALFANGLE+PI*0.5) angle -= FULLANGLE; // not wanted
	
	// If this is periodic triangle then allow that angle is contiguous with tri, not part of canonical tranche

	cc.x = cos(angle)*DEVICE_RADIUS_INSULATOR_OUTER;
	cc.y = sin(angle)*DEVICE_RADIUS_INSULATOR_OUTER;

	// defend against errors:
	// Is cc within triangle?
	
	// This is a fairly unnecessary way of doing it.
	// Here's a better one:
	// take linear average then project. That requires sqrt not atan.

	return 0;
}

/*void AuxTriangle::CalculateCircumcenter(Vector2 & cc, real * pdistsq)
{
	Vector2 Bb,C,b,c,a;
	Vector2 basea,baseb;
	
	if (periodic > 0)
	{
		// map everything to left hand side.
		MapLeft(a,b,c);
	} else {
		PopulatePositions(a,b,c);
	};
	Bb = b-a;
	C = c-a;		
	real D = 2.0*(Bb.x*C.y-Bb.y*C.x);
	real modB = Bb.x*Bb.x+Bb.y*Bb.y;
	real modC = C.x*C.x+C.y*C.y;
	cc.x = (C.y*modB-Bb.y*modC)/D + a.x;
	cc.y = (Bb.x*modC-C.x*modB)/D + a.y;

	if (pdistsq != 0)
		*pdistsq = (a.x-cc.x)*(a.x-cc.x)+(a.y-cc.y)*(a.y-cc.y); 
}





*/

	void ConvexPolygon::SetTri(const Vector2 & x1,const Vector2 & x2, const Vector2 & x3)
	{
		numCoords = 3;
		coord[0] = x1;
		coord[1] = x2;
		coord[2] = x3;
	}
	ConvexPolygon::ConvexPolygon(const Vector2 & x1,const Vector2 & x2,const Vector2 & x3)
	{
		SetTri(x1, x2, x3);
	}

	ConvexPolygon::ConvexPolygon()
	{
		numCoords = 0;
	}

	void ConvexPolygon::Get_Bxy_From_Az(real Az_array[], real * pBx,real * pBy)
	{
		// Assume we have coords that are sorted anticlockwise

		int i, inext;
		real Bx = 0, By = 0;
		real Integral_x, Integral_y;
		for (i = 0; i < numCoords; i++)
		{
			inext = i+1; if (inext == numCoords) inext = 0;
			Integral_x = (Az_array[i] + Az_array[inext])
							*(coord[inext].x-coord[i].x);
			Integral_y = (Az_array[i] + Az_array[inext])
							*(coord[inext].y-coord[i].y);
			Bx += Integral_x;
			By += Integral_y;
		}
		
		real area = this->GetArea();
		*pBx = 0.5*Bx/area;
		*pBy = 0.5*By/area;
	}

	Vector3 ConvexPolygon::Get_curl2D_from_anticlockwise_array(Vector3 A[])
	{
		// Assuming we have coords that are sorted anticlockwise

		int i, inext;
		Vector3 B;
		memset(&B,0,sizeof(Vector3));
		
		real Integral_x, Integral_y, Integral_z;
		for (i = 0; i < numCoords; i++)
		{
			inext = i+1; if (inext == numCoords) inext = 0;
			Integral_x = (A[i].z + A[inext].z)
							*(coord[inext].x-coord[i].x); // [anticlockwise]--> -dAz/dy
			Integral_y = (A[i].z + A[inext].z)
							*(coord[inext].y-coord[i].y); // [anticlockwise]--> dAz/dx
			Integral_z = (A[i].y+A[inext].y)
							*(coord[inext].y-coord[i].y) // [anticlockwise] --> dAy/dx 
						+ (A[i].x+A[inext].x)
						    *(coord[inext].x-coord[i].x);  // --> -dAx/dy
						 			
			B.x += Integral_x;
			B.y += Integral_y;
			B.z += Integral_z;
		}
		
		real area = this->GetArea();
		B *= 0.5/area;
		B.z += BZ_CONSTANT;
		
		return B;
	}
	Vector2 ConvexPolygon::Get_Integral_grad_from_anticlockwise_array(real Te[])
	{
		Vector2 grad;
		int i, inext;
		memset(&grad,0,sizeof(Vector2));

		real Integral_x, Integral_y;
		for (i = 0; i < numCoords; i++)
		{
			inext = i+1; if (inext == numCoords) inext = 0;
			Integral_x = 0.5*(Te[i] + Te[inext])
							*(coord[inext].y-coord[i].y); // [anticlockwise]--> dTe/dx
			Integral_y = 0.5*(Te[i] + Te[inext])
							*(coord[i].x-coord[inext].x); // [anticlockwise]--> dTe/dy
			grad.x += Integral_x;
			grad.y += Integral_y;
		}
		// ---> Compare to GradTe formula on tris.

		return grad;
	}
	Vector2 ConvexPolygon::Get_grad_from_anticlockwise_array(real Te[])
	{
		Vector2 grad;
		grad = Get_Integral_grad_from_anticlockwise_array(Te);
		real area = this->GetArea();
		grad /= area;
		return grad;
	}
	Vector2 ConvexPolygon::CalculateBarycenter()
	{
		// Assume we have coords that are sorted anticlockwise or clockwise
		Vector2 u;
		int i, inext;
		real Integral_x, Integral_y, shoelace;
		Integral_x = 0.0;
		Integral_y = 0.0;
		shoelace = 0.0;
		real lace;
		for (i = 0; i < numCoords; i++)
		{
			inext = i+1; if (inext == numCoords) inext = 0;

			lace = (coord[i].x * coord[inext].y - coord[inext].x*coord[i].y);
			Integral_x += (coord[i].x + coord[inext].x)*lace;
			Integral_y += (coord[i].y + coord[inext].y)*lace;
			shoelace += lace;
		};

		u.x = THIRD*Integral_x/shoelace;
		u.y = THIRD*Integral_y/shoelace;
		return u;
	}



			// Now, we will make sure that we enter the points in anticlockwise
			// order. 

			// Make polygon method: return_Bxy_from_Az( set of anticlockwise values)
		



	// Note: cases that apply:
	// when we introduce the first corner, it may cut the LH and bottom sides of the "house".
	// It cannot cut off the bottom entirely, I don't think. 
	
	// The second one may cut the RH edge and either the bottom or the LH diagonal.

	// ^^^ 1. Get some confidence that this is true.

	// %%% then we can proceed to say? First look at LH then at RH. We want to end up with 
	// 1. Edge length in each of 5 directions: maintain this as we go?
	// 2. Voronoi ConvexPolygon so that we can take intersections with wedges.
	


	
/*
	bool VoronoiPolygon::ClipAgainstHalfplane_Update_SideIndexList(const Vector2 & r1, const Vector2 & r2, const Vector2 & r3, int flag_new)
	{
		// Similar to the standard function except here, we will maintain for each vertex a number that indicates, for the side to anticlockwise
		// what flag that side corresponded to.
		// When we clip a vertex, the new vertex on the right takes up the rightmost clipped vertex's index
		// The new vertex on the left takes up the flag that was passed.
		
		// We can then look back (in the caller) and see which sides still exist and how long they are.
		// Less clear what we can do about overlaps with other wedges ... maybe ignore carefully.

		// first le's cut n paste:


		bool intersect;
		bool above_is_inside;
		real compare;
		int first, last;
		bool setfirst;
		real gradient ;
		int flag, pullback, i, post_last, pre_first;
		Vector2 cross1, cross2;
		
		static const real EPS = 5.0e-14;

		// Now we have to be able to deal with degenerate cases.
		// =====================================

		// We assign each point a status: INSIDE the clip region, NEAR the clip boundary, or OUTSIDE the clip region.

		// When some consecutive points are found to be OUTSIDE the clip region, we also
		// remove any consecutive points that are NEAR the clip boundary.
		// This ensures hopefully that intersections are only taken towards points that are away from the clip boundary.

		// If no points figure as actually OUTSIDE (further outside than EPS) then clipping is skipped.

		// If no points figure as actually INSIDE (further inside than EPS) then we return no intersection.

#define INSIDE     0
#define OUTSIDE    1
#define NEARBY       2
		// There is no point distinguishing near inside and near outside as this will be unreliable due to rounding anyway.
		
		Vector2 direction = r2-r1;
		// r1 + alpha.direction is the line
		
		if (direction.x*direction.x > direction.y*direction.y) 
			// defend against case that it's basically vertical
		{
			flag = DYDX;
			// determine whether r3 is above line:
		
			gradient = direction.y/direction.x;
			above_is_inside = (r3.y > r1.y + (r3.x-r1.x)*gradient) ;
			// Hope we didn't get passed r3 that is on the line r1-r2 . If so it's an unfair call and we have to change the caller.
			
			// Now ask which of our existing coordinates is in/near/out:			
			intersect = false;
			setfirst = false;
			for (int i = 0; i < numCoords; i++)
			{
				compare = r1.y + (coord[i].x-r1.x)*gradient;
				
				//if (above)
				//{
				//	is_above[i] = (coord[i].y > r1.y + (coord[i].x-r1.x)*gradient - EPS);
				//} else {
				//	is_above[i] = (coord[i].y > r1.y + (coord[i].x-r1.x)*gradient + EPS);
				//};
				
				if (above_is_inside)
				{
					if (coord[i].y > compare + EPS) {
						status[i] = INSIDE;
						intersect = true;
					} else {
						if (coord[i].y < compare - EPS) {
							status[i] = OUTSIDE;
							first = i;
							setfirst = true;
						} else {
							status[i] = NEARBY;
						};
					};
				} else {
					if (coord[i].y > compare + EPS) {
						status[i] = OUTSIDE;
						first = i;
						setfirst = true;
					} else {
						if (coord[i].y < compare - EPS) {
							status[i] = INSIDE;
							intersect = true;
						} else {
							status[i] = NEARBY;
						};
					};
				};
			//		last = i; // ordinarily this means we stored the first and last scrappable vertex
					// but beware that it won't work if the scrapped section crossed 0
			};
			
		} else {
			// line was more vertical than horizontal so take gradient x per y

			flag = DXDY;

			gradient = direction.x/direction.y;
			above_is_inside = (r3.x > r1.x + (r3.y-r1.y)*gradient); // true if r3 is to right

			// Now ask which of our existing coordinates is above
			
			intersect = false;
			setfirst = false;
			for (int i = 0; i < numCoords; i++)
			{
				compare = r1.x + (coord[i].y-r1.y)*gradient;
				
				if (above_is_inside)
				{
					if (coord[i].x > compare + EPS) {
						status[i] = INSIDE;
						intersect = true;
					} else {
						if (coord[i].x < compare - EPS) {
							status[i] = OUTSIDE;
							first = i;
							setfirst = true;
						} else {
							status[i] = NEARBY;
						};
					};
				} else {
					if (coord[i].x > compare + EPS) {
						status[i] = OUTSIDE;
						first = i;
						setfirst = true;
					} else {
						if (coord[i].x < compare - EPS) {
							status[i] = INSIDE;
							intersect = true;
						} else {
							status[i] = NEARBY;
						};
					};
				};
			};

		};

		if (intersect == false)
		{
			printf("error - Voronoi cell eclipsed.\n");
			// no intersection of halfplane and existing polygon
			return false;		
		};
		if (setfirst == false) 
		{
			// no clipping applies
			return true;
		};

		// If we get here, some polygon vertices were (properly) clipped and some were (properly) not.

		// OK now scrap those that did not intersect, if any, and replace them with the points where the lines are intersected

		// Let's get the first point before our OUTSIDE subset that is not INSIDE.

		while (status[first] != INSIDE)
		{
			first--;
			if (first < 0) first = numCoords-1;
		};
		pre_first = first;
		first++;
		if (first == numCoords) first = 0; // the first clipped vertex
		
		GetIntersection(&cross1,r1,gradient,flag,coord[pre_first],coord[first]); 
		
		// Now move forward to last scrappable.

		last = first;
		while (status[last] != INSIDE)
		{
			last++;
			if (last == numCoords) last = 0;
		};
		post_last = last;
		last--;
		if (last < 0) last = numCoords-1; // the last clipped vertex
		
		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[post_last]); 

		// now repopulate the array:
		// cases:
		if (last >= first)
		{
			// easy cases:
			if (last == first)
			{
				// exactly one vertex is clipped.
				
				// debug check: Never scrap a circumcenter:
				if (last <= max_index_no_scrap) {
					printf("circumcenter scrapped - not cool\n");
					last = last;
				};


				//	*	// When we clip a vertex, the new vertex on the right takes up the rightmost clipped vertex's index
				//	*	// The new vertex on the left takes up the flag that was passed.

				// array gets longer - need to move elements outwards first
				for (i = numCoords-1; i >= last+1; i--)
				{
					coord[i+1] = coord[i];
					edge_flag[i+1] = edge_flag[i];
				}
				numCoords++;
				coord[first] = cross1;
				coord[first+1] = cross2; 			
				// The anticlockwise added point now indexes the rest of the existing anticlock side:
				edge_flag[first+1] = edge_flag[first];
				// The clockwise added point indexes our new side:
				edge_flag[first] = flag_new;

			} else {
				
				if (first <= max_index_no_scrap) {
					printf("circumcenter scrapped - not cool - first %d max_index_no_scrap %d \n",first,max_index_no_scrap);
					first = first;
					getch();
				};


				edge_flag[first+1] = edge_flag[last];
				edge_flag[first] = flag_new;

				// last > first so we may need to pull some elements backwards
				pullback = last-first-1; // last == first +1 => pullback == 0
				for (i = last+1; i < numCoords; i++)
				{
					coord[i-pullback] = coord[i];
					edge_flag[i-pullback] = edge_flag[i];
				};
				numCoords -= pullback;
				coord[first] = cross1;
				coord[first+1] = cross2;
			};
		} else {
			// scrappable subset crosses 0
			// post_last is the first element that is INSIDE
				
			// debug check: Never scrap a circumcenter:
			if (this->max_index_no_scrap >= 0)
			{
				printf("circumcenter scrapped - not cool - first %d last %d post_last %d \n", first, last, post_last);

				numCoords = numCoords;
				getch();
			};
			
			// Move elements back to 0; if post_last = 4, first = 6 then there are 2 such elements + 2 new ones
			
			i = first-post_last;
			edge_flag[i+1] = edge_flag[last]; // last should be last scrapped vertex, > 0
			for (i =0; i < first-post_last; i++)
			{
				coord[i] = coord[i+post_last];
				edge_flag[i] = edge_flag[i+post_last];
			};
			coord[i] = cross1;
			coord[i+1] = cross2;
			edge_flag[i] = flag_new;
			numCoords = i+2;
		};

		return true;
	}
*/

	real ConvexPolygon::GetSucceedingSideLength(int side)
	{
		int next = side+1;
		if (next == numCoords) next = 0;
		return sqrt(
			(coord[next].x-coord[side].x)*(coord[next].x-coord[side].x)+
			(coord[next].y-coord[side].y)*(coord[next].y-coord[side].y));
	}

	real ConvexPolygon::GetPrecedingSideLength(int side)
	{
		int prev = side-1;
		if (prev == -1) prev = numCoords-1;
		return sqrt(
			(coord[prev].x-coord[side].x)*(coord[prev].x-coord[side].x)+
			(coord[prev].y-coord[side].y)*(coord[prev].y-coord[side].y));
	}

	bool ConvexPolygon::ClipAgainstHalfplane(const Vector2 & r1, const Vector2 & r2, const Vector2 & r3)
	{

		// The reason this way is not succeeding: basically we can create two equal points due to 
		// clipping a vertex that is on the boundary and replacing with 2

		bool intersect;
		bool above_is_inside;
		real compare;
		int first, last;
		bool setfirst;
		real gradient ;
		int flag, pullback, i, post_last, pre_first;
		Vector2 cross1, cross2;
		
		static const real EPS = 5.0e-14;

		// Now we have to be able to deal with degenerate cases.
		// =====================================

		// We assign each point a status: INSIDE the clip region, NEAR the clip boundary, or OUTSIDE the clip region.

		// When some consecutive points are found to be OUTSIDE the clip region, we also
		// remove any consecutive points that are NEAR the clip boundary.
		// This ensures hopefully that intersections are only taken towards points that are away from the clip boundary.

		// If no points figure as actually OUTSIDE (further outside than EPS) then clipping is skipped.

		// If no points figure as actually INSIDE (further inside than EPS) then we return no intersection.

#define INSIDE     0
#define OUTSIDE    1
#define NEARBY       2
		// There is no point distinguishing near inside and near outside as this will be unreliable due to rounding anyway.
		
		Vector2 direction = r2-r1;
		// r1 + alpha.direction is the line
		
		if (direction.x*direction.x > direction.y*direction.y) 
			// defend against case that it's basically vertical
		{
			flag = DYDX;
			// determine whether r3 is above line:
		
			gradient = direction.y/direction.x;
			above_is_inside = (r3.y > r1.y + (r3.x-r1.x)*gradient) ;
			// Hope we didn't get passed r3 that is on the line r1-r2 . If so it's an unfair call and we have to change the caller.
			
			// Now ask which of our existing coordinates is in/near/out:			
			intersect = false;
			setfirst = false;
			for (int i = 0; i < numCoords; i++)
			{
				compare = r1.y + (coord[i].x-r1.x)*gradient;
				
				//if (above)
				//{
				//	is_above[i] = (coord[i].y > r1.y + (coord[i].x-r1.x)*gradient - EPS);
				//} else {
				//	is_above[i] = (coord[i].y > r1.y + (coord[i].x-r1.x)*gradient + EPS);
				//};
				
				if (above_is_inside)
				{
					if (coord[i].y > compare + EPS) {
						status[i] = INSIDE;
						intersect = true;
					} else {
						if (coord[i].y < compare - EPS) {
							status[i] = OUTSIDE;
							first = i;
							setfirst = true;
						} else {
							status[i] = NEARBY;
						};
					};
				} else {
					if (coord[i].y > compare + EPS) {
						status[i] = OUTSIDE;
						first = i;
						setfirst = true;
					} else {
						if (coord[i].y < compare - EPS) {
							status[i] = INSIDE;
							intersect = true;
						} else {
							status[i] = NEARBY;
						};
					};
				};
			//		last = i; // ordinarily this means we stored the first and last scrappable vertex
					// but beware that it won't work if the scrapped section crossed 0
			};
			
		} else {
			// line was more vertical than horizontal so take gradient x per y

			flag = DXDY;

			gradient = direction.x/direction.y;
			above_is_inside = (r3.x > r1.x + (r3.y-r1.y)*gradient); // true if r3 is to right

			// Now ask which of our existing coordinates is above
			
			intersect = false;
			setfirst = false;
			for (int i = 0; i < numCoords; i++)
			{
				compare = r1.x + (coord[i].y-r1.y)*gradient;
				
				if (above_is_inside)
				{
					if (coord[i].x > compare + EPS) {
						status[i] = INSIDE;
						intersect = true;
					} else {
						if (coord[i].x < compare - EPS) {
							status[i] = OUTSIDE;
							first = i;
							setfirst = true;
						} else {
							status[i] = NEARBY;
						};
					};
				} else {
					if (coord[i].x > compare + EPS) {
						status[i] = OUTSIDE;
						first = i;
						setfirst = true;
					} else {
						if (coord[i].x < compare - EPS) {
							status[i] = INSIDE;
							intersect = true;
						} else {
							status[i] = NEARBY;
						};
					};
				};
			};

		};

		if (intersect == false) return false;		// nothing clipped
		if (setfirst == false) return true;			// all to be clipped - no change to ConvexPolygon object

		// If we get here, some polygon vertices were (properly) clipped and some were (properly) not.

		// OK now scrap those that did not intersect, if any, and replace them with the points where the lines are intersected

		// Let's get the first point before our OUTSIDE subset that is not INSIDE.

		while (status[first] != INSIDE)
		{
			first--;
			if (first < 0) first = numCoords-1;
		};
		pre_first = first;
		first++;
		if (first == numCoords) first = 0;
		
		GetIntersection(&cross1,r1,gradient,flag,coord[pre_first],coord[first]); 

		// Now move forward to last scrappable.

		last = first;
		while (status[last] != INSIDE)
		{
			last++;
			if (last == numCoords) last = 0;
		};
		post_last = last;
		last--;
		if (last < 0) last = numCoords-1;
		
		if (!_finite(coord[last].y)) {
			printf("bad ness. %1.8E %1.8E %1.8E  %1.8E %1.8E %1.8E \n",r1.x,r1.y,r2.x,r2.y,r3.x,r3.y);
			getch();			
		};
		
		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[post_last]); // this was passed a.y == #INF
		
		// now repopulate the array:
		// cases:
		if (last >= first)
		{
			// easy cases:
			if (last == first)
			{
				// array gets longer - need to move elements outwards first
				for (i = numCoords-1; i >= last+1; i--)
					coord[i+1] = coord[i];
				numCoords++;
				coord[first] = cross1;
				coord[first+1] = cross2; 			
			} else {
				// last > first so we may need to pull some elements backwards
				pullback = last-first-1; // last == first +1 => pullback == 0
				for (i = last+1; i < numCoords; i++)
					coord[i-pullback] = coord[i];
				numCoords -= pullback;
				coord[first] = cross1;
				coord[first+1] = cross2;
			};
		} else {
			// scrappable subset crosses 0
			// post_last is the first element that is INSIDE
			
			// Move elements back to 0; if post_last = 4, first = 6 then there are 2 such elements + 2 new ones
			for (i =0; i < first-post_last; i++)
				coord[i] = coord[i+post_last];
			coord[i] = cross1;
			coord[i+1] = cross2;
			numCoords = i+2;
		};
		
		return true;
		

		//// we know it goes from first to last
		//// except in the case that first == 0 for which we have to go again

		//if (first > 0)
		//{
		//	// get the coordinates where it crosses
		//	
		//	GetIntersection(&cross1,r1,gradient,flag,coord[first-1],coord[first]); 
		//	if (last < numCoords-1)
		//	{
		//		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[last+1]);
		//	} else {
		//		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[0]);
		//	};
		//	if (last == first)
		//	{
		//		// in this case we need to bop points forward, there are more coords in total
		//		for (i = numCoords-1; i >= last+1; i--)
		//			coord[i+1] = coord[i];
		//		numCoords++;
		//		coord[first] = cross1;
		//		coord[first+1] = cross2; 					
		//	} else {
		//		// in this case we may need to pull points backward in the array
		//		pullback = last-first-1; // last-first == 1 => 0

		//		if (pullback > 0)
		//			for (i = last+1; i < numCoords; i++)
		//				coord[i-pullback] = coord[i];
		//		coord[first] = cross1;
		//		coord[first+1] = cross2;
		//		numCoords -= pullback;
		//	};
		//	
		//} else {
		//	// have to handle this special case that point 0 was not in: seek again to find the ends of the interval of scrappable vertices
		//	// work backwards from the end
		//	first = numCoords;
		//	while (is_above[first-1] != above) first--;
		//	if (first == numCoords) first = 0;
		//	last = 0;
		//	while (is_above[last+1] != above) last++;

		//	if (first == 0)
		//	{
		//		GetIntersection(&cross1,r1,gradient,flag,coord[numCoords-1],coord[0]); 
		//		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[last+1]); 
		//		
		//		// Now do exactly as above.
		//		// COPY-PASTE:

		//		if (last == first)
		//		{
		//			// in this case we need to bop points forward, there are more coords in total
		//			for (i = numCoords-1; i >= last+1; i--)
		//				coord[i+1] = coord[i];
		//			numCoords++;
		//			coord[first] = cross1;
		//			coord[first+1] = cross2; 					
		//		} else {
		//			// in this case we may need to pull points backward in the array
		//			pullback = last-first-1; // last-first = 1 -> 0

		//			if (pullback > 0)
		//				for (i = last+1; i < numCoords; i++)
		//					coord[i-pullback] = coord[i];
		//			coord[first] = cross1;
		//			coord[first+1] = cross2;
		//			numCoords -= pullback;
		//		};

		//	} else {
		//		
		//		// all those from first onwards are considered destroyed...
		//		GetIntersection(&cross1,r1,gradient,flag,coord[first-1],coord[first]); 
		//		GetIntersection(&cross2,r1,gradient,flag,coord[last],coord[last+1]); 
		//		
		//		coord[0] = cross2; 
		//		// remove up until last
		//		pullback = last; // if last == 0 then we don't need to move anything
		//		// if last == 1 then we move element 2 to element 1
		//		for (i = last+1; i < numCoords; i++)
		//			coord[i-pullback] = coord[i];
		//		coord[first-pullback] = cross1;
		//		numCoords = first-pullback+1;
		//	};
		//};
		//
		//return true;

	}


	void ConvexPolygon::CopyFrom(ConvexPolygon & cp)
	{
		numCoords = cp.numCoords;
		for (int i = 0; i < numCoords; i++)
			coord[i] = cp.coord[i];
	}

	void ConvexPolygon::GetCentre(Vector2 & centre)
	{
		centre.x = 0.0;
		centre.y = 0.0;
		for (int i = 0; i < numCoords; i++)
		{
			centre.x += coord[i].x;
			centre.y += coord[i].y;
		}
		centre.x /= (real)numCoords;
		centre.y /= (real)numCoords;
	}

	real ConvexPolygon::FindTriangleIntersectionArea(Vector2 & r1, Vector2 & r2, Vector2 & r3)
	{
		ConvexPolygon cp;
		cp.CopyFrom(*this);

		if (!cp.ClipAgainstHalfplane(r1,r2,r3)) return 0.0;
		if (!cp.ClipAgainstHalfplane(r1,r3,r2)) return 0.0;
		if (!cp.ClipAgainstHalfplane(r2,r3,r1)) return 0.0;
		return cp.GetArea();
	}

	bool ConvexPolygon::GetIntersectionWithTriangle(ConvexPolygon * pPoly,Vector2 & r1, Vector2 & r2, Vector2 & r3)
	{
		pPoly->CopyFrom(*this);

		if (!pPoly->ClipAgainstHalfplane(r1,r2,r3)) return false;
		if (!pPoly->ClipAgainstHalfplane(r1,r3,r2)) return false;
		if (!pPoly->ClipAgainstHalfplane(r2,r3,r1)) return false;
		
		return true;
	}
	
	bool ConvexPolygon::GetIntersectionWithPolygon(ConvexPolygon * pPoly, // target
																						ConvexPolygon * pClip)// clip this against that.
	{
		int i, inext, inext2;

		pPoly->CopyFrom(*this);
		
		// convex polygon: if we take any edge then we should be fine to supply any other point as being on the "in" side of that edge..

		for (i = 0; i < pClip->numCoords; i++)
		{
			inext = i+1; if (inext == pClip->numCoords) inext = 0;
			inext2 = inext+1; if (inext2 == pClip->numCoords) inext2 = 0;
			if (!pPoly->ClipAgainstHalfplane(pClip->coord[i],pClip->coord[inext],pClip->coord[inext2])) return false;
		};
		
		return true;
	}
/*
	real ConvexPolygon::IntegratePlane(Vector2 & r1, Vector2 & r2, Vector2 & r3,
																	real y1, real y2, real y3)
	{
		// Procedure: 
		// evaluate planar variable at all corners
		// chop up this into triangles
		// assume average attained by plane on each triangle
		// take sum of multiplying average by area of triangle
		real y[CP_MAX];


		// make tri-aligned coordinates:

		Vector2 x12 = r2 - r1;
		real dist12 = x12.modulus();
		x12.x /= dist12;
		x12.y /= dist12;
		Vector2 x12perp;
		x12perp.x = x12.y;
		x12perp.y = -x12.x;
		
		dbyd12 = (y2-y1)/dist12;
		Vector2 x13 = r3-r1;
		real x13_12 = x13.x*x12.x + x13.y*x12.y;
		real x13_perp = x13.x*x12perp.x + x13.y*x12perp.y;
		//Vector2 position = r1 + x13_12*x12;
		real ypos = y1 + dbyd12*x13_12;
		real dbydperp = (y3-ypos)/x13_perp;
		Vector2 relpos;
		// Now plane is
		// y1 + dbyd12*((x-r1) dot x12) + dbydperp*((x-r1) dot x12perp)
		
		// evaluate planar variable at all corners
		for (int i = 0; i < numCoords; i++)
		{
			relpos = coord[i] - r1;
			y[i] = y1 + dbyd12*(relpos.x*x12.x + relpos.y*x12.y) + dbydperp*(relpos.x*x12perp.x+relpos.y*x12perp.y);
		};

		// chop up this into triangles
		// assume average attained by plane on each triangle
		// take sum of multiplying average by area of triangle

		// pick point 0 and make tris
		// we know the points should always be ordered
		real average;
		ConvexPolygon cpTri;
		real sum = 0.0;
		for (int i = 2; i < numCoords; i++)
		{
			average = (y[0] + y[i-1] + y[i])*THIRD;
			cpTri.clear();
			cpTri.add(coord[0]);
			cpTri.add(coord[i-1]);
			cpTri.add(coord[i]);
			area = cpTri.GetArea();
			sum += average*area;
		};
		// case of 3 coords: 0,1,2
		// case of 4 coords: 0,1,2  0,2,3

		return sum;
	}*/
	void ConvexPolygon::IntegrateMass(Vector2 & r1, Vector2 & r2, Vector2 & r3,
									real yvals1, real yvals2, real yvals3, real * pResult)
	{
		real y[CP_MAX];

		Vector2 relpos;
		Vector2 x12perp,x12;
		real dist12;
		real ypos, dbyd12, dbydperp;
		
		// make tri-aligned coordinates:
		x12 = r2 - r1;
		dist12 = x12.modulus();
		x12.x /= dist12;
		x12.y /= dist12;
		x12perp.x = x12.y;
		x12perp.y = -x12.x;
		
		Vector2 x13 = r3-r1;
		// dot products to give lengths:
		real x13_12 = x13.x*x12.x + x13.y*x12.y;
		real x13_perp = x13.x*x12perp.x + x13.y*x12perp.y;
		
		dbyd12 = (yvals2-yvals1)/dist12;
		ypos = yvals1 + dbyd12*x13_12;
		dbydperp = (yvals3-ypos)/x13_perp;

		for (int i = 0; i < numCoords; i++)
		{
			relpos = coord[i] - r1;
			y[i] = yvals1 + dbyd12*(relpos.x*x12.x + relpos.y*x12.y) + dbydperp*(relpos.x*x12perp.x+relpos.y*x12perp.y);
		};

		*pResult = 0.0;
		// pick point 0 and make tris
		// we know the points should always be ordered
		real average,area;
		ConvexPolygon cpTri;
		for (int i = 2; i < numCoords; i++)
		{
			cpTri.Clear();
			cpTri.add(coord[0]);
			cpTri.add(coord[i-1]);
			cpTri.add(coord[i]);
			area = cpTri.GetArea();
			average = (y[0] + y[i-1] + y[i])*THIRD;			
			*pResult += average*area;
		};
	}
						
	void ConvexPolygon::Integrate_Planes(Vector2 & r1, Vector2 & r2, Vector2 & r3,
										real yvals1[],
										real yvals2[],
										real yvals3[],	
										real results[],
										long N_planes)
	{
		// Procedure: 
		// evaluate planar variable at all corners
		// chop up this into triangles
		// assume average attained by plane on each triangle
		// take sum of multiplying average by area of triangle


		// So what are we assuming here? That the polygon to integrate over is
		// a subset of the triangle??

		real y[CP_MAX][15]; // max 15 planes
		Vector2 relpos;
		Vector2 x12perp,x12;
		real dist12;
		real ypos, dbyd12[15], dbydperp[15];
		
		// make tri-aligned coordinates:
		x12 = r2 - r1;
		dist12 = x12.modulus();

		x12.x /= dist12;
		x12.y /= dist12;
		x12perp.x = x12.y;
		x12perp.y = -x12.x;
		
		// So x12 is a unit vector

		Vector2 x13 = r3-r1;
		// dot products to give lengths:
		real x13_12 = x13.x*x12.x + x13.y*x12.y;
		real x13_perp = x13.x*x12perp.x + x13.y*x12perp.y;
		
		// So x13_12 is projection of x13 in direction 12

		for (int j = 0; j < N_planes; j++)
		{
			dbyd12[j] = (yvals2[j]-yvals1[j])/dist12;
			ypos = yvals1[j] + dbyd12[j]*x13_12; // value at point along 12 as far as x13 projected
			dbydperp[j] = (yvals3[j]-ypos)/x13_perp;
		};
		// Now plane is
		// y1 + dbyd12*((x-r1) dot x12) + dbydperp*((x-r1) dot x12perp)
		
		// evaluate planar variable at all corners
		for (int i = 0; i < numCoords; i++)
		{
			relpos = coord[i] - r1;
			for (int j = 0; j < N_planes; j++)
				y[i][j] = yvals1[j] + dbyd12[j]*(relpos.x*x12.x + relpos.y*x12.y) + dbydperp[j]*(relpos.x*x12perp.x+relpos.y*x12perp.y);
		};

		// chop up this into triangles
		// assume average attained by plane on each triangle
		// take sum of multiplying average by area of triangle

		for (int j = 0; j < N_planes; j++)
			results[j] = 0.0;
		// pick point 0 and make tris
		// we know the points should always be ordered
		real average,area;
		ConvexPolygon cpTri;
		for (int i = 2; i < numCoords; i++)
		{
			cpTri.Clear();
			cpTri.add(coord[0]);
			cpTri.add(coord[i-1]);
			cpTri.add(coord[i]);
			area = cpTri.GetArea();
			for (int j = 0; j < N_planes; j++)
			{
				average = (y[0][j] + y[i-1][j] + y[i][j])*THIRD;			
				results[j] += average*area;
			};
		};
		// case of 3 coords: 0,1,2
		// case of 4 coords: 0,1,2  0,2,3
	}




	real ConvexPolygon::FindQuadrilateralIntersectionArea(Vector2 & r1, Vector2 & r2, Vector2 & r3, Vector2 & r4)
	{
		ConvexPolygon cp;
		cp.CopyFrom(*this);

		if (!cp.ClipAgainstHalfplane(r1,r2,r3)) return 0.0;
		if (!cp.ClipAgainstHalfplane(r2,r3,r4)) return 0.0;
		if (!cp.ClipAgainstHalfplane(r3,r4,r1)) return 0.0;
		if (!cp.ClipAgainstHalfplane(r1,r4,r2)) return 0.0;
		return cp.GetArea();
	}
	real ConvexPolygon::GetArea()
	{
		// shoelace formula as we should use elsewhere also.
		if (numCoords == 0) return 0.0;
		real area = 0.0;
		int i;
		for (i = 0; i < numCoords-1; i++)
		{
			area += coord[i].x*coord[i+1].y - coord[i+1].x*coord[i].y;
		};
		area += coord[i].x*coord[0].y - coord[0].x*coord[i].y;
		if (area < 0.0)
			return -area*0.5;
		return area*0.5;
	}

real CalculateTriangleIntersectionArea(Vector2 & x1, Vector2 & x2, Vector2 & x3,
													          Vector2 & r1, Vector2 & r2, Vector2 & r3)
{
	// Get stack overflow and it appears here ?!!

	// Sometimes this way fails.

	// Spurious point gets added when we clip against a plane that meets the boundary.

	// This invalidates the shoelace area formula which relies on convexity.


	//bool boolIntersection_exists;
	ConvexPolygon cp (x1,x2,x3);

	// Clip against half plane created by r1 to r2 in the direction of r3

	if (!cp.ClipAgainstHalfplane(r1,r2, r3)) // returns true if intersection existed
		return 0.0;

	if (!cp.ClipAgainstHalfplane(r2,r3,r1))
		return 0.0;

	if (!cp.ClipAgainstHalfplane(r1,r3,r2))
		return 0.0;

	return cp.GetArea(); // shoelace formula as we should use for triangle also.

 	// The way sketched out below may be faster so leave it in and try it after.
}


int Triangle::GetCornerIndex(Vertex * pVertex)
{
	if (cornerptr[0] == pVertex) return 0;
	if (cornerptr[1] == pVertex) return 1;
	return 2;
}



real Triangle::ReturnNormalDist(Vertex * pOppVert)
{
	Vector2 u[3];
	real dist;
	MapLeftIfNecessary(u[0],u[1],u[2]);
	
	if (pOppVert == cornerptr[0])
	{
		dist = edge_normal[0].dot(u[1]-u[0]);
		return dist;
	};
	if (pOppVert == cornerptr[1])
	{
		dist = edge_normal[1].dot(u[0]-u[1]);
		return dist;
	};
	dist = edge_normal[2].dot(u[0]-u[2]);
	return dist;
}

Vector2 CreateOutwardNormal(real x1, real y1,
					real x2, real y2,
					real x, real y)
{
	// (x,y) is on the "inside"
	Vector2 normal;

	normal.x = y2-y1;
	normal.y = x1-x2;
	if (normal.x*(x-x1)+normal.y*(y-y1) > 0.0)
	{
		// case: normal points towards (x,y) from (x1,y1)
		normal.x = -normal.x;
		normal.y = -normal.y;
	}

	return normal;
}

Vertex * TriMesh::Search_for_iVolley_equals (Vertex * pSeed,int value)
{
	if (pSeed->iVolley == value) return pSeed; // should not happen though
	pSeed->iIndicator = 1; // searched
	smartlong searched;
	searched.add(pSeed-X);

	Vertex * pNeigh,*pVertex,*pReturn;
	int iNeigh, i;
	// work outwards: look at neighbours
	// set indicator for search? Need smth like this.

	// Idea: Repeatedly take element from searched, scroll down it;
	// if neighbours are not already searched, search them and add to
	// bottom of the list.		

	long neigh_len;
	long izNeighs[128];

	long iSearchCaret = 0;
	int not_found = true;
	do
	{
		pVertex = X + searched.ptr[iSearchCaret];

		neigh_len = pVertex->GetNeighIndexArray(izNeighs);

		for (i = 0; i < neigh_len; i++)
		{
			pNeigh = X + izNeighs[i];
			if (pNeigh->iIndicator == 0) {
				if (pNeigh->iVolley == value) 
				{
					not_found = false;
					pReturn = pNeigh;
					break; // where this takes us out to, not sure, but hopefully doesn't matter
				}
				pNeigh->iIndicator = 1;
				searched.add(pNeigh-X);
			}
		}
		iSearchCaret++;
	} while (not_found);

	// Need to restore iIndicator == 0 at the end of a search....
	for (i = 0; i < searched.len; i++)
		(X + searched.ptr[i])->iIndicator = 0;
	
	return pReturn;
}


/*AuxVertex * TriMesh::Search_for_iVolley_equals (AuxVertex * pSeed,int value, int iLevel)
{
	if (pSeed->iVolley == value) return pSeed; // should not happen though
	pSeed->iIndicator = 1; // searched
	smartlong searched;
	searched.add(pSeed-AuxX[iLevel]);

	AuxVertex * pNeigh,*pAux,*pReturn;
	int iNeigh, i;
	// work outwards: look at neighbours
	// set indicator for search? Need smth like this.

	// Idea: Repeatedly take element from searched, scroll down it;
	// if neighbours are not already searched, search them and add to
	// bottom of the list.		

	long iSearchCaret = 0;
	int not_found = true;
	do
	{
		pAux = AuxX[iLevel] + searched.ptr[iSearchCaret];
		for (i = 0; i < pAux->neigh_len; i++)
		{
			pNeigh = AuxX[iLevel] + pAux->iNeighbours[i];
			if (pNeigh->iIndicator == 0) {
				if (pNeigh->iVolley == value) 
				{
					not_found = false;
					pReturn = pNeigh;
					break; // where this takes us out to, not sure, but hopefully doesn't matter
				}
				pNeigh->iIndicator = 1;
				searched.add(pNeigh-AuxX[iLevel]);
			}
		}
		iSearchCaret++;
	} while (not_found);

	// Need to restore iIndicator == 0 at the end of a search....
	for (i = 0; i < searched.len; i++)
		(AuxX[iLevel] + searched.ptr[i])->iIndicator = 0;
	
	return pReturn;
}*/
/*
void AuxTriangle::Set(AuxVertex * p1, AuxVertex * p2, AuxVertex * p3, long iTri)
{
	cornerptr[0] = p1;
	cornerptr[1] = p2;
	cornerptr[2] = p3;
	
	// Also calc circumcentre because we may be about to use it:
	Vector2 Bb,C,b,c,a;
	Vector2 basea,baseb;
	
	if (periodic > 0)
	{
		// map everything to left hand side.
		MapLeft(a,b,c);
	} else {
		PopulatePositions(a,b,c);
	};
	Bb = b-a;
	C = c-a;		
	real D = 2.0*(Bb.x*C.y-Bb.y*C.x);
	real modB = Bb.x*Bb.x+Bb.y*Bb.y;
	real modC = C.x*C.x+C.y*C.y;
	cc.x = (C.y*modB-Bb.y*modC)/D + a.x;
	cc.y = (Bb.x*modC-C.x*modB)/D + a.y;

	p1->addtri(iTri);
	p2->addtri(iTri);
	p3->addtri(iTri);
	
}

void AuxTriangle::Reset(AuxVertex * p1, AuxVertex * p2, AuxVertex * p3, long iTri)
{
	// First delete from existing vertex lists
	cornerptr[0]->remove_tri(iTri);
	cornerptr[1]->remove_tri(iTri);
	cornerptr[2]->remove_tri(iTri);

	Set(p1,p2,p3,iTri);	
}

AuxTriangle * TriMesh::GetAuxTriangleContaining(AuxVertex * pAux1,
							   AuxVertex * pAux2,
							   int iLevel)
{
	int i;
	AuxTriangle * pITri;
	for (i = 0; i < pAux1->tri_len; i++)
	{
		pITri = AuxT[iLevel]+pAux1->iTriangles[i];
		if ((pITri->cornerptr[0] == pAux2) ||
			(pITri->cornerptr[1] == pAux2) ||
			(pITri->cornerptr[2] == pAux2))
		{
			return pITri;
		}
	}
	return 0;
}

bool AuxTriangle::TestDelaunay(AuxVertex * pAux)
{
	real qdistsq = (pAux->x-cc.x)*(pAux->x-cc.x)+(pAux->y-cc.y)*(pAux->y-cc.y);
	real pdistsq = (cornerptr[0]->pos.x-cc.x)*(cornerptr[0]->pos.x-cc.x)
				+ (cornerptr[0]->pos.y-cc.y)*(cornerptr[0]->pos.y-cc.y);
	return (qdistsq < pdistsq);
	// return 1 if q is within circumcircle
}


/*
real Triangle::CalculateIntersectionArea(Vector2 & x1, Vector2 & x2, Vector2 & x3,
										                          Vector2 & r1, Vector2 & r2, Vector2 & r3)
{
	// Note that this routine works strictly on the actual given coordinates
	// So make sure any periodic mapping is done beforehand.

	// First let's establish a series of points that split up columns, starting with the leftmost point we have got.

	// a. Put each triangle into left-to-right order
	Vector2 temp,temp2;

	if (x2.x > x1.x )
	{
		if (x3.x > x2.x) 
		{
			// nothing to do
		} else {
			if (x3.x < x1.x)
			{
				// order is x3, x1, x2
				temp.x = x1.x;
				temp.y = x1.y;
				x1.x = x3.x;
				x1.y = x3.y;
				x3.x = x2.x;
				x3.y = x2.y;
				x2.x = temp.x;
				x2.y = temp.y;
			} else {
				// order is x1, x3, x2
				temp.x = x2.x;
				temp.y = x2.y;
				x2.x = x3.x;
				x2.y = x3.y;
				x3.x = temp.x;
				x3.y = temp.y;
			};
		};
	} else {
		if (x3.x > x1.x)
		{
			// order is x2 x1 x3
			temp.x = x1.x;
			temp.y = x1.y;
			x1.x = x2.x;
			x1.y = x2.y;
			x2.x = temp.x;
			x2.y = temp.y;
		} else {
			if (x3.x > x2.x)
			{
				// order is x2 x3 x1
				temp.x = x1.x;
				temp.y = x1.y;
				x1.x = x2.x;
				x1.y = x2.y;
				x2.x = x3.x;
				x2.y = x3.y;
				x3.x = temp.x;
				x3.y = temp.y;
			} else {
				// order is x3 x2 x1
				temp.x = x1.x;
				temp.y = x1.y;
				x1.x = x3.x;
				x1.y = x3.y;
				x3.x = temp.x;
				x3.y = temp.y;
			};
		};
	};
	
	// b. Where is the leftmost point? 
	// No point starting until both triangles are started.

	// Write this out in a longwinded way unless and until we think of a clever way.

	real area, gradient_r12, gradient_r23, gradient_x12, gradient_x23;

	area = 0.0;

	gradient_r12 = (r2.y-r1.y)/(r2.x-r1.x);
	gradient_r13 = (r3.y-r1.y)/(r3.x-r1.x);
	gradient_x12 = (x2.y-x1.y)/(x2.x-x1.x);
	gradient_x13 = (x3.y-x1.y)/(x3.x-x1.x);

	if (x1.x < r1.x)
	{
		// start at r1.x
		if (x2.x < r2.x)
		{
			// x1 r1 x2
			// going from r1.x to x2.x
			area += ColumnIntersection(
									// column x-values start and finish:
									r1.x,x2.x,
									// y-values:
									r1.y,			// left top for r
									r1.y,			// left bot for r
									r1.y + gradient_r12*(x2.x-r1.x),//(r2.y-r1.y)*(x2.x-r1.x)/(r2.x-r1.x),   // right top for r
									r1.y + gradient_r13*(x2.x-r1.x),//(r3.y-r1.y)*(x2.x-r1.x)/(r3.x-r1.x),   // right bot for r
									x1.y + gradient_x12*(r1.x-x1.x),//(x2.y-x1.y)*(r1.x-x1.x)/(x2.x-x1.x), // left top for x
									x1.y + gradient_x13*(r1.x-x1.x),//(x3.y-x1.y)*(r1.x-x1.x)/(x3.x-x1.x), // left bot for x
									x2.y,    // right top for x
									x1.y + gradient_x13*(x2.x-x1.x)//(x3.y-x1.y)*(x2.x-x1.x)/(x3.x-x1.x) // right bot for x
									);		
			if (r2.x < x3.x)
			{
				// going from x2.x to r2.x

				area += ColumnIntersection(
									// column x-values start and finish:
									x2.x,r2.x,
									// y-values:
									r1.y+gradient_r12*(x2.x-r1.x),//(r2.y-r1.y)*(x2.x-r1.x)/(r2.x-r1.x),			// left top for r
									r1.y+gradient_r13*(x2.x-r1.x),//(r3.y-r1.y)*(x2.x-r1.x)/(r3.x-r1.x),			// left bot for r
									r2.y,   // right top for r
									r1.y + gradient_r13*(r2.x-r1.x),//(r3.y-r1.y)*(r2.x-r1.x)/(r3.x-r1.x),   // right bot for r
									x2.y, // left top for x
									x1.y + gradient_x13*(x2.x-x1.x),//(x3.y-x1.y)*(x2.x-x1.x)/(x3.x-x1.x), // left bot for x
									x2.y + gradient_x23*(r2.x-x2.x),//(x3.y-x2.y)*(r2.x-x2.x)/(x3.x-x2.x),    // right top for x
									x1.y + gradient_x13*(r2.x-x1.x)//(x3.y-x1.y)*(r2.x-x1.x)/(x3.x-x1.x) // right bot for x
									);		

				if (x3.x < r3.x)
				{
					// the final one goes from r2.x to x3.x

					area += ColumnIntersection(
										r2.x,x3.x,
										// y-values:
										r2.y,							// left via r2
										r1.y+gradient_r13*(r2.x-r1.x),//(r3.y-r1.y)*(r2.x-r1.x)/(r3.x-r1.x), // left, r1->r3
										r2.y + gradient_r23*(x3.x-r2.x),//(r3.y-r2.y)*(x3.x-r2.x)/(r3.x-r2.x), // right via r2,
										r1.y + gradient_r13*(x3.x-r1.x),//(r3.y-r1.y)*(x3.x-r1.x)/(r3.x-r1.x), // right, r1->r3,
										
										x2.y + gradient_x23*(r2.x-x2.x),//(x3.y-x2.y)*(r2.x-x2.x)/(x3.x-x2.x),
										x1.y + gradient_x13*(r2.x-x1.x),//(x3.y-x1.y)*(r2.x-x1.x)/(x3.x-x1.x),
										x3.y,
										x3.y
										);
				} else {
					// the final one goes from r2.x to r3.x

					area += ColumnIntersection(
										r2.x,r3.x,
										// y-values:
										r2.y,
										r1.y+gradient_13*(r2.x-r1.x),//(r3.y-r1.y)*(r2.x-r1.x)/(r3.x-r1.x), // left, r1->r3
										r3.y,
										r3.y,
										x2.y + gradient_x23*(r2.x-x2.x),//(x3.y-x2.y)*(r2.x-x2.x)/(x3.x-x2.x),
										x1.y + gradient_x13*(r2.x-x1.x),//(x3.y-x1.y)*(r2.x-x1.x)/(x3.x-x1.x),
										x2.y + gradient_x23*(r3.x-x2.x),//(x3.y-x2.y)*(r3.x-x2.x)/(x3.x-x2.x),
										x1.y + gradient_x13*(r3.x-x1.x)//(x3.y-x1.y)*(r3.x-x1.x)/(x3.x-x1.x)
										);										
				};
			} else {
				// the final one from x2.x to x3.x
				// x1 r1 x2 x3
				// some efficiency savings to be made if we calculate gradient_s before calling these functions.
				area += ColumnIntersection(
								x2.x,x3.x,
								r1.y+gradient_r12*(x2.x-r1.x),
								r1.y+gradient_r13*(x2.x-r1.x),
								r1.y+gradient_r12*(x3.x-r1.x),
								r1.y+gradient_r13*(x3.x-r1.x),
								x2.y,
								x1.y+gradient_x13*(x2.x-x1.x),
								x3.y,
								x3.y);
			};
		} else {
			// x1 r1 r2 
			// going from r1.x to r2.x

			area += ColumnIntersection(
									// column x-values start and finish:
									r1.x,r2.x,
									// y-values:
									r1.y,			// left bot for r
									r1.y,			// left top for r
									r2.y,			 // right top for r
									r1.y + gradient_r13*(r2.x-r1.x),   // right bot for r
									
									x1.y + gradient_x12*(r1.x-x1.x), // left top for x
									x1.y + gradient_x13*(r1.x-x1.x), // left bot for x
									x1.y + gradient_x12*(r2.x-x1.x), // left top for x
									x1.y + gradient_x13*(r2.x-x1.x), // left bot for x
									);
			if (x2.x < r3.x)
			{
				// x1 r1 r2 x2

				// going from r2.x to x2.x

				area += ColumnIntersection(
									r2.x,x2.x,
									// y-values:
									r2.y,
									r1.y+gradient_r13*(r2.x-r1.x),
									r2.y+gradient_r23*(x2.x-r2.x),
									r1.y+gradient_r13*(x2.x-r1.x),
									
									x1.y+gradient_x12*(r2.x-x1.x),
									x1.y+gradient_x13*(r2.x-x1.x),
									x2.y,
									x1.y+gradient_x13*(x2.x-x1.x)
									);
									


				if (r3.x < x3.x)
				{
					// the final one from x2.x to r3.x
					
					// x1 r1 r2 x2 r3

					area += ColumnIntersection(
									x2.x,r3.x,
									
									r2.y + gradient_r23*(x2.x-r2.x),
									r1.y + gradient_r13*(x2.x-r1.x),
									r3.y,
									r3.y,

									x2.y,
									x1.y+gradient_x13*(x2.x-x1.x),
									x2.y+gradient_x23*(r3.x-x2.x),
									x1.y+gradient_x13*(r3.x-x1.x)
									);
				} else {
					// the final one from x2.x to x3.x
					
					// x1 r1 r2 x2 x3
					area += ColumnIntersection(
									x2.x,x3.x,

									r2.y + gradient_r23*(x2.x-r2.x),
									r1.y + gradient_r13*(x2.x-r1.x),
									r2.y + gradient_r23*(x3.x-r2.x),
									r1.y + gradient_r13*(x3.x-r1.x),

									x2.y,
									x1.y + gradient_x13*(x2.x-x1.x),
									x3.y,
									x3.y);
									
				};
			} else {
				// the final one from r2.x to r3.x

				// x1 r1 r2 r3
				area += ColumnIntersection(
					r2.x,r3.x,

					r2.y,
					r1.y + gradient_r13*(r2.x-r1.x),
					r3.y,
					r3.y,

					x1.y + gradient_r12*(r2.x-x1.x),
					x1.y + gradient_r13*(r2.x-x1.x),
					x1.y + gradient_r12*(r3.x-x1.x),
					x1.y + gradient_r13*(r3.x-x1.x)
					);
			};
		};
	} else {
		// r1  x1
		if (x2.x < r2.x)
		{
			// going from x1.x to x2.x

		} else {
			// r1 x1 r2 
			// going from x1.x to r2.x

			




		}
	};

	return area;
}

real ColumnIntersection( real x1, real x2,
						real y_a_1_left,
						real y_a_2_left,
						real y_a_1_right,
						real y_a_2_right,

						real y_b_1_left,
						real y_b_2_left,
						real y_b_1_right,
						real y_b_2_right)
{
	// Is it possible for x1->x2->x3 and x1->x3 to cross in this region?
	
	// take average to determine which of 1,2 is higher for a
	real b_bvg,a_avg, y_a_top_left,y_a_top_right,y_a_bot_left,y_a_bot_right,
		y_b_top_left,y_b_top_right,y_b_bot_left,y_b_bot_right;

	real width = x2-x1;
	a_avg1 = y_a_1_left + y_a_1_right;
	a_avg2 = y_a_2_left + y_a_2_right;		// can avoid this by keeping global flag for whether gradient12 > gradient13 for first triangle
	if (a_avg2 > a_avg1)
	{
		y_a_top_left = y_a_2_left; y_a_top_right = y_a_2_right;
		y_a_bot_left = y_a_1_left; y_a_bot_right = y_a_1_right;
	} else {
		y_a_top_left = y_a_1_left; y_a_top_right = y_a_1_right;
		y_a_bot_left = y_a_2_left; y_a_bot_right = y_a_2_right;
	};

	b_bvg1 = y_b_1_left + y_b_1_right;
	b_bvg2 = y_b_2_left + y_b_2_right;
	if (b_bvg2 > b_bvg1)
	{
		y_b_top_left = y_b_2_left; y_b_top_right = y_b_2_right;
		y_b_bot_left = y_b_1_left; y_b_bot_right = y_b_1_right;
	} else {
		y_b_top_left = y_b_1_left; y_b_top_right = y_b_1_right;
		y_b_bot_left = y_b_2_left; y_b_bot_right = y_b_2_right;
	};

	// Now consider some cases
	
	// We are going to kind of clip a against the two half planes given by b

	// Distinguish 3 possible locations for y_b_top_left rel to a, do cases for where y_b_top right is,
	// then go from there with y_b_bot

	// But do this in an order so that non-intersections can be detected quickly.

	total = 0.0;

		a_bot_gradient = (y_a_bot_right-y_a_bot_left)/width;
		a_top_gradient = (y_a_top_right-y_a_top_left)/width;
		b_bot_gradient = (y_b_bot_right-y_b_bot_left)/width;
		b_top_gradient = (y_b_top_right-y_b_top_left)/width;


	if (y_b_top_left < y_a_bot_left)
	{
		if (y_b_top_right < y_a_bot_right) return 0.0; // CASE I

		if (y_b_top_right < y_a_top_right) {
			// CASE II: b top cuts a bot upwards
			
			total = triangle area()
			x = (b

			if (y_b_bot_right > y_a_bot_right) 
			{
				// remove a triangle
				total -= triangle area()

			};
			return total;

		} else {
			// CASE III: b top cuts both lines of a from below
			
			if (
		};
	} else {
		if (y_b_top_left > y_a_top_left)
		{
			// b top started above a
			if (y_b_top_right > y_a_top_right)
			{
				// CASE IV: b top had no effect on a
			} else {
				if (y_b_top_right > y_a_bot_right)
				{
					// CASE V: b top cuts a top downwards

				} else {
					// CASE VI: b top cuts both lines of a, downwards

				};
			};
		} else {
			// b_top started in between
			
			if (y_b_top_right > y_a_top_right)
			{
				// CASE VII: b top starts inside, goes up and outside
			} else {
				if (y_b_top_right > y_a_bot_right)
				{
					// CASE VIII: b top starts inside, stays inside;

				} else {
					// CASE XI: b top starts inside, goes below; only a triangle is then relevant

				};
			};
		};
	};


	if (y_b_bot_left > y_a_top_left)
	{
		// First test for no intersection:
		if (y_b_bot_right > y_a_top_right) return 0.0;
				
		// therefore, at some point b_bot crosses a_top.
		// until that point, we have no intersection.
		
		// difference of gradients (b bot' - a top') * x + original difference (b bot - a top) = 0
		// x = - original difference / difference of gradients

		a_bot_gradient = (y_a_bot_right-y_a_bot_left)/width;
		a_top_gradient = (y_a_top_right-y_a_top_left)/width;
		b_bot_gradient = (y_b_bot_right-y_b_bot_left)/width;
		b_top_gradient = (y_b_top_right-y_b_top_left)/width;
		
		x = (y_a_top_left-y_b_bot_left)/(b_bot_gradient - a_top_gradient);
		
		// Now from this point onwards what can happen?


		x = (y_b_bot_left-y_a_top_left)/
			

	} else {
		if ((y_b_top_left < y_a_bot_left) && (y_b_top_right < y_a_bot_right)) return 0.0;

	};

	
	// OK so there is some intersection
	
		
}
*/



#endif
