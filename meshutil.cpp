#ifndef MESHUTILCPP
#define MESHUTILCPP

#define DEFINEexp  exp

#include "headers.h"

real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;
// include here only:

#include "mesh.cpp" // will include "basics.cpp"
#include "surfacegraph_tri.cpp"
#include "cppconst.h"

//		This file to contain 4 types of functions:

//		$$$   	Search functions
//		$$$   	Initialisation functions
//		$$$   	Save / Load / Output functions
//		$$$   	Graphics functions

bool GlobalFlagNeedPeriodicImage, GlobalFlagNeedReflectedImage;

//int nesting;
long GlobalVerticesInRange;
long GlobalTrianglesInfluencing;
bool GlobalPeriodicSearch;

real InitialIonDensity(real x, real y)
{
	static const real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0);
	static const real XCENTRE1 = -XCENTRE2;
	static const real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0);
	static const real YCENTRE_ROT = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0+PI/16.0);
	static const real XCENTRE3 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0+PI/16.0);
	static const real XCENTRE4 = -XCENTRE3;

	real n,xdist,ydist;
	xdist = x-XCENTRE1;
	ydist = y-YCENTRE;
	n = UNIFORM_n;
	n += FILAMENT_OWN_PEAK_n*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALnSD*INITIALnSD));
	xdist = x-XCENTRE2;
	n += FILAMENT_OWN_PEAK_n*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALnSD*INITIALnSD));
	xdist = x-XCENTRE3;
	ydist = y-YCENTRE_ROT;
	n += FILAMENT_OWN_PEAK_n*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALnSD*INITIALnSD));
	xdist = x-XCENTRE4;
	n += FILAMENT_OWN_PEAK_n*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALnSD*INITIALnSD));
	return n;
}		
real InitialIonDensityScrewPinch(real x,real y)
{
	y -= 3.61; // origin

	real n = N0_centre_SP*exp(-(x*x+y*y)/(a_nSD*a_nSD)) + n_UNIFORM_SP;
	return n;
}

real InitialTemperature(real x, real y)
{
	static const real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0);
	static const real XCENTRE1 = -XCENTRE2;
	static const real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0);
	static const real YCENTRE_ROT = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0+PI/16.0);
	static const real XCENTRE3 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0+PI/16.0);
	static const real XCENTRE4 = -XCENTRE3;

	real T,xdist,ydist;
	xdist = x-XCENTRE1;
	ydist = y-YCENTRE;
	T = UNIFORM_T;
	T += FILAMENT_OWN_PEAK_T*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALTSD*INITIALTSD));
	xdist = x-XCENTRE2;
	T += FILAMENT_OWN_PEAK_T*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALTSD*INITIALTSD));
	xdist = x-XCENTRE3;
	ydist = y-YCENTRE_ROT;
	T += FILAMENT_OWN_PEAK_T*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALTSD*INITIALTSD));
	xdist = x-XCENTRE4;
	T += FILAMENT_OWN_PEAK_T*DEFINEexp(-(xdist*xdist+ydist*ydist)/(2.0*INITIALTSD*INITIALTSD));
	return T;
}		

real InitialNeutralDensity(real x, real y)
{
	return 1.0e18 - InitialIonDensity(x, y);
}				

void TriMesh::InitialisePeriodic(void)
{
	Triangle * pTri = T;
	for (long i = 0; i < numTriangles; i++)
	{
		pTri->GuessPeriodic();
		++pTri;
	};
}

/*void AuxTriangle::SetTriangleVertex(int which, AuxVertex * pInner)
{
	cornerptr[which] = pInner;
	pInner->addtri(this-GlobalInnerT);

	// Well this is just no good.
}

*/

/*void Triangle::GetBaseMedian(Vector2 * pResult, Vertex * pContig)
	{
#ifdef VERTBASED
		// replaced by
		//this->GetCentreOfIntersectionWithInsulator(pResult);
		*pResult = GetContiguousCC(pContig);
#else
		
		pResult->x = 0.0; pResult->y = 0.0;			
		if (periodic == 0) {
			// just a case of identifying which vertices are base
			if (cornerptr[0]->flags == 3) { pResult->x += 0.5*cornerptr[0]->x; pResult->y += 0.5*cornerptr[0]->y; };
			if (cornerptr[1]->flags == 3) { pResult->x += 0.5*cornerptr[1]->x; pResult->y += 0.5*cornerptr[1]->y; };
			if (cornerptr[2]->flags == 3) { pResult->x += 0.5*cornerptr[2]->x; pResult->y += 0.5*cornerptr[2]->y; };
		} else {
			Vector2 u0,u1,u2;
			MapLeft(u0,u1,u2);
			if (cornerptr[0]->flags == 3) { *pResult += 0.5*u0; };
			if (cornerptr[1]->flags == 3) { *pResult += 0.5*u1; };
			if (cornerptr[2]->flags == 3) { *pResult += 0.5*u2; };
			if (pContig->x > 0.0) *pResult = Clockwise*(*pResult);
		};
#endif
	}

void Triangle::GetOuterMedian(Vector2 * pResult, Vertex * pContig)
	{
#ifdef VERTBASED
// replaced by
		
		// ?
		// pressure has to be felt from somewhere, might as well be ??

		*pResult = GetContiguousCC(pContig);
#else
	//	if (this->u8EdgeFlag != CONVEX_EDGE_TRI)
	//	{
	//		printf("oh dear -- GetOuterMedian called for wrong tri"); getch();
	//	};
		// u8EdgeFlag no longer valid.

		pResult->x = 0.0; pResult->y = 0.0;			
		if (periodic == 0) {
			// just a case of identifying which vertices are base
			if (cornerptr[0]->flags == 4) { pResult->x += 0.5*cornerptr[0]->x; pResult->y += 0.5*cornerptr[0]->y; };
			if (cornerptr[1]->flags == 4) { pResult->x += 0.5*cornerptr[1]->x; pResult->y += 0.5*cornerptr[1]->y; };
			if (cornerptr[2]->flags == 4) { pResult->x += 0.5*cornerptr[2]->x; pResult->y += 0.5*cornerptr[2]->y; };
		} else {
			Vector2 u0,u1,u2;
			MapLeft(u0,u1,u2);
			if (cornerptr[0]->flags == 4) { *pResult += 0.5*u0; };
			if (cornerptr[1]->flags == 4) { *pResult += 0.5*u1; };
			if (cornerptr[2]->flags == 4) { *pResult += 0.5*u2; };
			if (pContig->x > 0.0) *pResult = Clockwise*(*pResult);
		};
#endif
	}
*/

/*void TriMesh::InitialiseScrewPinch()
{
	
	static real const TWOOVERSQRT3 = 2.0/sqrt(3.0);
	static real const OVERSQRT3 = 1.0/sqrt(3.0);
	static real const SQRT3OVER2 = sqrt(3.0)/2.0;

	// can't be equilateral mesh because it's fully ionised.

	// so we do need to start with cells proportional to the density.

	// set radii accordingly to smth.

	// May have to move vertices to get same into all.

	// Think we have to assume that we do need an initial nonuniform density.
	// Can take some liberties with density per tri further out. Take maximum spacing.

	long iRow, i, prev_N_points, N_points;
	real angle;
	static real const TWOPI = 2.0*PI;

	long iVertex, iNext, iStartCircle ,	iStartInner,iVertexInside,
				iNextCircle,iVertexPrev,iTri,iprev,iNextInner;
	real newr2;
	Tensor2 rotate;
	Triangle * pTri;
	Vertex * pVertex;
	
	if (1)
	{
		// do radial:


		// 1. choose radius for initial hexagon, based on estimated
		// "mass per tri on average".

		// Let's try this way.

		// Integral of Gaussian = 1;
		// Gaussian = 1/(2 pi a^2) exp(-r^2 / a^2)
		// n = N0_centre exp(-r^2 / a^2) 

		real TotalMass = N0_centre_SP* PI*a_nSD*a_nSD;
		// hang on: this isn't with x^2/2 sigma^2 in exponent -- careful what it integrates to:
		// 2 sigma^2 = a^2 ; integral of exp(-) is 2 pi sigma^2 = pi a^2
		
		long TotalTrisAim = NUMBER_OF_VERTICES_AIMED*2;
		real AverageMass = TotalMass/((real)TotalTrisAim);

		// Now decide the radius for these innermost tris;
		// Assume density here is close to the max.

		real DesiredArea = AverageMass/N0_centre_SP;
		Vector2 firstpoint, origin;

		real target, twovar, r1, r2, argument, lnarg, relerr, r_spacing, InnerDelta;
		
		// mass = n0*area ; area = mass/n0
		// Area of equi triangle = 0.5*0.866*delta^2 
		InnerDelta = sqrt(DesiredArea/(0.5*0.866));


		for (int iPass = 0; iPass < 2; iPass++)
		{

			origin.x = 0.0; origin.y = 3.61;
			Vector2 temp2(InnerDelta,0.0);
			firstpoint = temp2;

			if (iPass == 1) {
				X[0].x = origin.x; X[0].y = origin.y; // where we place the filament.
				
				angle = TWOPI/6.0;
				Tensor2 rotate(cos(angle),-sin(angle),sin(angle),cos(angle));
				for (i = 1; i <= 6; i++)
				{
					X[i].x = temp2.x+origin.x;X[i].y = temp2.y+origin.y;
					temp2 = rotate*temp2;
				};
			};
			numRow[0] = 1;
			numRow[1] = 6;
			
			// 2. Now assume that at next radius we have spacing corresp to n(radius)
			// How much mass is then in tris, given that there are 2 for each new point?

			// Aiming for average mass in each tri ... until spacing reaches 0.04 ReturnRadius

			// Bear in mind we do not want to run out of vertices - we will get to a point
			// where they are at maximum spacing. Maybe have to go back over, do a count.

			// Simpler: assume radial spacing is 0.866 of theta spacing;
			// therefore as r increases we create fewer triangles


			// The mass integrated between two radii r1 and r2 is ?

			// -a^2 exp(-r^2/a^2)  -/-/-/->  2r exp(-r^2/a^2)

			// Integrate [N0 exp(-r^2/a^2)]dx dy  = N0 2pi[r exp(-r^2/a^2)]dr  =  a^2 N0 pi ( exp(-r1^2/a^2) - exp(-r2^2/a^2) )
			
			// Meanwhile, number of cells?
			// Assume spacing is 0.866 times theta spacing. Circumference 2 pi r2. Number of cells = 2 (2 pi r2) 0.866/(r2-r1)
			
			// So want ( exp(-r1^2/a^2) - exp(-r2^2/a^2) ) (r2-r1)/ r2 = 4 0.866 AverageMass / (a^2 N0)
			
			prev_N_points = 6;
			iRow = 2;
			iVertex = numRow[0]+numRow[1];
			bool bReachedMaximum = false;
			
			real f;
			real r1part, f_inner, r2low, r2high;

			r1 = InnerDelta;
			target = 4.0*0.866*AverageMass/(a_nSD*a_nSD*N0_centre_SP);
			twovar = a_nSD*a_nSD;
					
			// rewrite:
			// So want ( exp(-r1^2/a^2) - exp(-r2^2/a^2) ) (r2-r1)/ r2 = 4 0.866 AverageMass / (a^2 N0)
			
			// Try staircase method (can reverse if need be) :
			// Didn't work, or explode either.

			do { // circles:
				if (bReachedMaximum)
				{
					r_spacing = maximum_spacing*0.866;
					r2 = r1+r_spacing;
				} else {

					// Go again: bisection method ... 

					// 0 = ( exp(-r1^2/a^2) - exp(-r2^2/a^2) ) (r2-r1) - 4 r2 0.866 AverageMass / (a^2 N0)

					// when r2 gets large ... r2 ( exp(-r1^2/a^2) - 4*0.866 stuff)
					// when r2 == r1 ... again indeterminate.

					// it could be that this is large positive or large negative depending. 
					
					r1part = exp(-r1*r1/(a_nSD*a_nSD));

					r2 = r1+InnerDelta*0.5;
					f_inner = ( r1part - exp(-r2*r2/(a_nSD*a_nSD)) ) *(r2-r1) - r2*target;
					printf("r2 %1.5E f %1.5E \n",r2,f_inner);
					r2 = r1+maximum_spacing;
					f = ( r1part - exp(-r2*r2/(a_nSD*a_nSD)) ) *(r2-r1) - r2*target;
					printf("r2 %1.5E f %1.5E \n",r2,f);

					// seems to go from negative to positive, at least initially.
					if (f < 0.0) {
						// switch to maximum spacing

						r_spacing = r2-r1;
						bReachedMaximum = true;

					} else {

						if (f_inner > 0.0) {
							printf("error - positive f_inner");
							getch();
						} else {
							// bisection method

							r2high = r2;
							r2low = r1+InnerDelta;

							do {
								r2 = (r2high+r2low)*0.5;
								f = ( r1part - exp(-r2*r2/(a_nSD*a_nSD)) ) *(r2-r1) - r2*target;
								if (f > 0.0) {
									r2high = r2;
								} else {
									r2low = r2;
								};

							} while (r2high-r2low > 1.0e-10);
						};
						
						r_spacing = r2-r1;

						printf("solved distance = %1.6E (InnerDelta %1.6E )\n",
							r_spacing,InnerDelta);
					};

				}; // whether reached maximum spacing


				// first point: take half the previous angle that was away from its first point.
				// Don't know if that's always best or not.
				real desired_theta_spacing = r_spacing/0.866;
				N_points = (long)( (TWOPI*r2)/desired_theta_spacing );
				
				firstpoint = (firstpoint)*(r2/r1);
				
				if (iPass == 1) {
					angle = PI/(real)(N_points); // half (new) angle
					rotate.xx = cos(angle);
					rotate.xy = -sin(angle);
					rotate.yx = sin(angle);
					rotate.yy = cos(angle);
					
					firstpoint = rotate*firstpoint;
					
					angle = TWOPI/(real)(N_points);
					rotate.xx = cos(angle);
					rotate.xy = -sin(angle);
					rotate.yx = sin(angle);
					rotate.yy = cos(angle);
				
					temp2 = firstpoint;
					for (i = 0; i < N_points; i++)
					{
						X[iVertex].x = temp2.x+X[0].x;X[iVertex].y = temp2.y+X[0].y;
						temp2 = rotate*temp2;
						iVertex++;
					}
				} else {
					numRow[iRow] = N_points;
					iVertex += numRow[iRow];
				};

				r1 = r2;
				prev_N_points = N_points;
				iRow++;
				
			} while (r2 < OUTER_RADIUS);

			OuterRadiusAttained = r2;
			printf("OuterRadiusAttained %1.10E OUTER_RADIUS %1.10E \n",
				OuterRadiusAttained,OUTER_RADIUS);

			numRows = iRow;

			if (iPass == 0) {
				numVertices = iVertex;
				X = new Vertex[numVertices];
				if (X == 0){
					printf("allocation error: X \n"); getch();
					return;
				} else {
					INT64 address1 = (INT64)(&(X[0]));
					INT64 address2 = (INT64)(&(X[numVertices]));
					INT64 difference = address2-address1;
					INT64 est = numVertices*sizeof(Vertex);		
					printf(	"Allocated %d vertices. sizeof(Vertex) = %d .\n"
							"NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
								numVertices,sizeof(Vertex),est,difference);
				};
			};
		}; // next iPass :
		// first pass establishes number of verts for dim; second pass assigns positions.
			
		
		iTri = 0;
		for (int iPass = 0; iPass < 2; iPass++)
		{
			// Now assign triangles: track azimuthally between each pair of circles.
			
			iVertexInside = 0;
			pTri = T;
			for (i = 1; i <= 6; i++) 
			{
				// place triangle to left of point i:
				iprev = i-1; if (iprev == 0) iprev = numRow[1];

				iVertex = i; iVertexPrev = iprev;
				if (iPass == 1) {
					pTri->SetTriangleVertex(0,X + iVertexInside);
					pTri->SetTriangleVertex(1,X + iVertexPrev);
					pTri->SetTriangleVertex(2,X + iVertex);
					++pTri;
				} else {
					iTri++;
				};				
			}
			iVertex = i; // 7
			iRow = 1;
			iVertexInside = iVertex-numRow[iRow]; // 1
			for (iRow = 2; iRow < numRows; iRow++) // the outer circle is row iRow
			{
				// Start with triangle that has the first point from the previous circle, the first and last points of this.
				iStartCircle = iVertex;
				iStartInner = iVertexInside;
				iNextCircle = iVertex + numRow[iRow];
				iVertexPrev = iVertex + numRow[iRow]-1;
				if (iPass == 1) {					
					pTri->SetTriangleVertex(0,X + iVertexInside);
					pTri->SetTriangleVertex(1,X + iVertexPrev);
					pTri->SetTriangleVertex(2,X + iVertex);
					++pTri;
				} else {
					iTri++;
				};

				// Now, we walk along the circle that has the least azimuthal distance, until we get to a triangle that features
				// again both the last and previousfirst.

				while (iVertex != iVertexPrev)
				{
					iNextInner = iVertexInside + 1; if (iNextInner == iStartCircle) iNextInner = iStartInner;
					iNext = iVertex + 1; if (iNext == iNextCircle) iNext = iStartCircle;
					
					real anglenext_inner = atan2(X[iNextInner].y-origin.y,X[iNextInner].x-origin.x);
					if (anglenext_inner <= 0.0) anglenext_inner += TWOPI;
					real anglenext_outer = atan2(X[iNext].y-origin.y,X[iNext].x-origin.x);
					if (anglenext_outer <= 0.0) anglenext_outer += TWOPI;

					// To compare angles, put in similar quadrant:
					if ((anglenext_inner > 1.5*PI) || (anglenext_outer > 1.5*PI))
					{
						if (anglenext_inner < 0.5*PI) anglenext_inner += TWOPI;
						if (anglenext_outer < 0.5*PI) anglenext_outer += TWOPI;		
					};

					if (anglenext_inner < anglenext_outer) 
					{
						// add triangle, base on inside
			
						if (iPass == 1) {
							pTri->SetTriangleVertex(0,X + iVertex);
							pTri->SetTriangleVertex(1,X + iVertexInside);
							pTri->SetTriangleVertex(2,X + iNextInner);
							++pTri;
						} else {
							++iTri;
						};
						iVertexInside++;
						if (iVertexInside == iStartCircle) iVertexInside = iStartInner;

					} else {
						// add triangle, base on outside

						if (iPass == 1) {
							pTri->SetTriangleVertex(0,X + iVertexInside);
							pTri->SetTriangleVertex(1,X + iVertex);
							pTri->SetTriangleVertex(2,X + iNext);
							++pTri;
						} else {
							++iTri;
						};
						iVertex++;
						//if (iVertex == iNextCircle) iVertex = iStartCircle; // unnecessary
					};
				};
				// when we reach iVertex == end of circle iVertexPrev, that will happen before we reach iStartInner 
				// but after we reach the inner one previous to that. In other words, what remains
				// is to connect iVertexPrev, iStartInner and the one previous to that.

				if (iPass == 1) {
					pTri->SetTriangleVertex(0,X + iStartInner);
					pTri->SetTriangleVertex(1,X + iVertexPrev);
					pTri->SetTriangleVertex(2,X + iVertexInside);

					if (iVertexInside + 1 != iStartCircle) {
						printf("eweireijr");
						getch();
					};
					++pTri;
				} else {
					++iTri;
				};

				// Now set up next tri row:
				iVertex++;
				iVertexInside = iStartCircle;
			};	

			if (iPass == 0) {
				numTriangles = iTri;

				T = new Triangle[numTriangles];
				if (T == 0){
					printf("allocation error: T \n"); getch();
					return;
				} else {
					INT64 address1 = (INT64)(&(T[0]));
					INT64 address2 = (INT64)(&(T[numTriangles]));
					INT64 difference = address2-address1;
					INT64 est = numTriangles*sizeof(Triangle);		
					printf(	"Allocated %d triangles. sizeof(Triangle) = %d .\n"
							"NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
								numTriangles,sizeof(Triangle),est,difference);
				};
			};

		}; // iPass

	} else {

		// . Equilateral mesh: it will get automatically rebalanced.

		// Initial "scale length a" ( = n SD) will be 0.04

		// Make mesh that is MESHWIDTH wide and this is the diameter of the return current + a small amount.

		// Coordinates from - to + horizontally, 0 to max vertically.

		static real const NUMBER_OF_VERTICES = NUMBER_OF_VERTICES_AIMED;
		
		// number in square that is x by x, given delta: x^2/(0.866 delta^2)
		// 36000/x^2 = 1/ 0.866 delta^2
		// x^2/(0.866*36000) = delta^2

		// Trouble is: if we have say a = 0.04 cm initially and 1 cm radius out, then only
		// about 1/200 of the area is of interest.
		// We could change the radius of the return current to 10 initial SD and see what happens, 
		// but that still means only 3% (so think 1000 points) describe the filament initially.
	
		// Let's show what happens both ways.



	}

	this->StartAvgRowTri = 0; // for taking Az avg: just set avg over all space = 0


	// 5. Populate triangle neighbour list
	// ____________________________________

	// Can now use vertex triangle membership arrays to do this.


	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->flags = 0;
		// outermost row:
		if (iVertex >= numVertices-numRow[numRows-1]) pVertex->flags = CONVEX_EDGE_VERTEX;
		++pVertex;
	};

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
 		pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri);
		pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri);
		pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri);
		// This function now modified to return itself if there is no other shared tri.
	
		pTri->periodic = 0;
		pTri->u8domain_flag = DOMAIN_TRIANGLE;
		pTri->area = pTri->GetArea();
		pTri->CalculateCentroid(&(pTri->cc));
		pTri->RecalculateEdgeNormalVectors(false); 
		
		pTri++;		
	};

	numDomainVertices = numVertices;
	Xdomain = X;

	RefreshVertexNeighboursOfVerticesOrdered(); 

	InitialPopulateScrewPinch(); 

	this->Redelaunerize(true);
	this->RefreshVertexNeighboursOfVerticesOrdered();

	InitialPopulateScrewPinch(); 


	//==============================

	// Now do same for each coarse 'equilateral' mesh:

	for (i = 0; i < NUM_COARSE_LEVELS; i++)
	{
		CreateEquilateralAuxMeshScrewPinch(i);		// hmm
	};

	this->Coarsest.Invoke(numAuxVertices[NUM_COARSE_LEVELS-1]+1);
	this->LUphi.Invoke(numAuxVertices[NUM_COARSE_LEVELS-1]);

	// Will have to think about one day:
	// Should auxiliary meshes move with the mesh, or not?
	// Maybe if they did it would work better. The number of elements can still be small...
	// Especially if we insist on doing W-cycles instead of control loop.

	return;
}
*/

int TriMesh::Initialise(int token)
{
	real x,y,rr,r;
	long i,iRow,iTri;
	real spacing, theta_spacing, theta, r_spacing;
	long numRowprev;
	bool stop;
	long iVertex,iVertexLow,iVertexFirst,iVertexNextFirst,iVertexLowFirst;
	bool inner_stop;
	real xdist,ydist;
	real vertex_density_per_cm_sq;	
	Vertex * vert, *pVertex;
	Triangle * tri, *pTri;

	real R_aim, r_use1, r_use2;
	long numRow1, numRow2, numRowPrev;

	real r_row[1024];

	static real const TWOOVERSQRT3 = 2.0/sqrt(3.0);
	static real const OVERSQRT3 = 1.0/sqrt(3.0);
	static real const SQRT3OVER2 = sqrt(3.0)/2.0;

	real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0);
	real XCENTRE1 = -XCENTRE2;
	real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0);

	// New approach:

	// We will achieve the exact number of vertices so that it's easier in CUDA.
	// We then need to go and create frills when we do tri mesh.

	// Nvertices = Nrows(Nrows-1)(pi/(8 sqrt 3))((R2+R1)/(R2-R1))

	//real const R2 = DOMAIN_OUTER_RADIUS;
	//real const R1 = INNER_A_BOUNDARY;
	//Numrows = (int)(0.5 + sqrt(0.25 + ((real)NUMBER_OF_VERTICES_AIMED)*(R2-R1)*8.0*SQRT3/(PI*(R2+R1)));
	//// Chose too few, so the triangles will be longer and thinner than otherwise.
	//// Maybe Nrow = 200.99 so we ought to go to the nearer # rows.

	//// azimuthal:
	//spacing = ((real)Numrows)*PI_OVER_16*(R2+R1)/(real)NUMBER_OF_VERTICES_AIMED;

	// Oh dear --- we missed a trick: a row has to be put on REVERSE_ZCURRENT_RADIUS and rows have to be
	// halfway before and after insulator.
	
	// Try again. Stick with the present approach, bump up the numbers in each row.
	
	// Now:
	// * create numRow array
	// * find if we are over or under number of vertices, and add more into appropriate rows, or delete.
	
		
	// 1. Do a dry run to determine how many vertices we will try to actually use.
	// ____________________________________________________________________________
	
	real TotalArea = PI*(DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS-INNER_A_BOUNDARY*INNER_A_BOUNDARY)/16.0;
	real NUM_VERTICES_PER_CM_SQ = ((real)NUMBER_OF_VERTICES_AIMED)/TotalArea;

	iRow = 0;
	// PREVIOUS VERSION:
//	spacing = sqrt((1.0/(real)NUM_VERTICES_PER_CM_SQ)*TWOOVERSQRT3); 
//	r_spacing = spacing*SQRT3OVER2; // equilateral triangles

	// ---------------------------------------------------------------

	// Expect [0.5*(R1+R2)*(pi/8)/r_spacing]*[(R2-R1)/spacing + 1] = num of vertices obtained
	// Note (R1+R2)(R2-R1) = (R2*R2-R1*R1)

	// OLD, wrong:
//	r_spacing = sqrt(3.0/16.0 + TotalArea*SQRT3OVER2/(real)NUMBER_OF_VERTICES_AIMED) - SQRT3OVER2*0.5;

	f64 temp = 0.5*(PI/16.0)*(DOMAIN_OUTER_RADIUS+INNER_A_BOUNDARY)/(real)NUMBER_OF_VERTICES_AIMED;
	r_spacing = sqrt(temp*temp + SQRT3OVER2*TotalArea/(real)NUMBER_OF_VERTICES_AIMED) + temp;

	spacing = r_spacing/SQRT3OVER2;
		
	// Expecting then? Number of rows ~ sqrt((num_verts_aimed)/TotalArea
	
	// First fill in rows from INNER_A_BOUNDARY TO REVERSE_ZCURRENT_RADIUS;
	// then add more rows to get us near to the insulator; stop at 1/2 distance from it;
	// then add the domain vertices, again starting from 1/2 a row above the insulator.
	
	numRow1 = (int)((REVERSE_ZCURRENT_RADIUS-INNER_A_BOUNDARY)/r_spacing); // too few - squeeze out
	if ((REVERSE_ZCURRENT_RADIUS-INNER_A_BOUNDARY)/r_spacing -(real)numRow1 > 0.5) numRow1++; // squeeze in instead
	r_use1 = (REVERSE_ZCURRENT_RADIUS-INNER_A_BOUNDARY)/(real)numRow1;
	
	numVertices = 0;

	r = INNER_A_BOUNDARY;
	FRILL_CENTROID_INNER_RADIUS = r - r_use1*0.5;

	numRowprev = (int)(FULLANGLE*r/spacing)+1;
	for (iRow = 0; iRow <= numRow1; iRow++)
	{
		if (FULLANGLE/(real)numRowprev > spacing/r) 
		{
			numRow[iRow] = numRowprev+1; 
			// note that r_spacing < spacing and fullangle < 45 degrees, so 1 is enough
		} else {
			numRow[iRow] = numRowprev;
		};
		numRowprev = numRow[iRow];
		numVertices += numRow[iRow];		
		r_row[iRow] = r; // for temporary use
		r += r_use1;
	}; // numRow1 is the row at REVERSE_ZCURRENT_RADIUS
	
	numEndZCurrentRow = numVertices-1; // the previous one.
	numStartZCurrentRow = numVertices-numRow[numRow1];
	// THIS WILL HAVE TO BE CHANGED IF WE CHANGE THE NUMBER IN EACH ROW.
	// MAYBE ONLY ADJUST DOMAIN.
	
	// this gets us to REVERSE_ZCURRENT_RADIUS where we put the current.
	R_aim = DEVICE_RADIUS_INSULATOR_OUTER - 0.5*r_spacing;
	numRow2 = (int)((R_aim-REVERSE_ZCURRENT_RADIUS)/r_spacing);
	if (((R_aim-REVERSE_ZCURRENT_RADIUS)/r_spacing) - (real)numRow2 > 0.5) numRow2++;
	
	// Now stretch them:
	r_use2 = (DEVICE_RADIUS_INSULATOR_OUTER-REVERSE_ZCURRENT_RADIUS)/
				(((real)numRow2)+0.5);
	r += r_use2-r_use1; 
	for (; iRow <= numRow1+numRow2; iRow++)
	{
		if (FULLANGLE/(real)numRowprev > spacing/r) 
		{
			numRow[iRow] = numRowprev+1; 
			// note that r_spacing < spacing and fullangle < 45 degrees, so 1 is enough
		} else {
			numRow[iRow] = numRowprev;
		};
		numRowprev = numRow[iRow];		
		numVertices += numRow[iRow];
		
		r_row[iRow] = r;
		r += r_use2;
	}; // numRow1+numRow2 is the last row inside the ins.
	
	// Domain rows:
//	spacing = sqrt((1.0/(real)NUM_VERTICES_PER_CM_SQ)*TWOOVERSQRT3); 
//	r_spacing = spacing*SQRT3OVER2; // equilateral triangles
	// Why did we ever do that????
	
	// A sensible idea: refresh r_spacing from here on out to try to fit the remaining number of vertices.
	// (We could do it every row.)
	
	r = DEVICE_RADIUS_INSULATOR_OUTER + r_spacing*0.5;
	
	int numUse = (int)((DOMAIN_OUTER_RADIUS-r)/r_spacing); // excludes 0th row
	real r_use3 = (DOMAIN_OUTER_RADIUS-r)/numUse;
	
	for (i = 0; //r < DOMAIN_OUTER_RADIUS; r += r_use3)
				i <= numUse; i++)
	{
		if (FULLANGLE/(real)numRowprev > spacing/r) 
		{
			numRow[iRow] = numRowprev+1; // note that r_spacing < spacing and fullangle < 45 degrees, so 1 is enough
		} else {
			numRow[iRow] = numRowprev;
		};
		numRowprev = numRow[iRow];		
		numVertices += numRow[iRow];	
		
		r_row[iRow] = r;
		r += r_use3;		
		iRow++;
	};
	
	numRows = iRow; 	
	Outermost_r_achieved = r-r_use3; // should now be DOMAIN_OUTER_RADIUS. 	
	FRILL_CENTROID_OUTER_RADIUS = r - r_use3*0.5;
	// Used for Lap A calculating from A_frill but not for major area calc.
	
	// This is giving disagreement of areas: major areas think they
	// include out to this centroid whereas minor areas take frill area = 0.
	
	// Consider what is best to do about that.
	

	// ##################################

	// Now go over and increment / decrement each row to try to get the exact number of vertices.
	// printf("Outermost_r_achieved %1.10E r_row[0] %1.10E\n",r-r_use3,r_row[0]);
	
	printf("numVertices %d NUM_AIMED %d ... \n",numVertices, NUMBER_OF_VERTICES_AIMED);
	
	// Clever recoding would avoid going through and doing all these divides:
	while (numVertices > NUMBER_OF_VERTICES_AIMED) {
		// change numRow from numRow1+1 onward.
		f64 density, highdens = 0.0;
		long iHigh = 0;
		for (i = numRow1+1; i < numRows; i++) // if we adjust more-inner rows, then we'd have to change StartZCurrentRow
		{
			density = ((real)numRow[i])/r_row[i];
			if (density > highdens) {
				highdens = density;
				iHigh = i;
			}
		}
		numRow[iHigh]--;
		numVertices--;
		printf("iHigh %d numRow[iHigh] %d \n",iHigh,numRow[iHigh]);
	};
	
	while (numVertices < NUMBER_OF_VERTICES_AIMED) {
		f64 density, lowdens = 1.0e100;
		long iLow = 0;
		for (i = numRow1+1; i < numRows; i++)
		{
			density = ((real)numRow[i])/r_row[i];
			if (density < lowdens) {
				lowdens = density;
				iLow = i;				
			}
		}
		numRow[iLow]++;
		numVertices++;
		printf("iLow %d numRow[iLow] %d \n",iLow,numRow[iLow]);
	};
	
	// ##################################################
	
	numInnermostRow = numRow[0]; // store
	numOutermostRow = numRow[numRows-1]; // store
	
	// PREVIOUS VERS:
	//numTrianglesAllocated = (long)(2.02*(real)(numVertices+numRows
	//	+ numRow[0] + numRow[numRows-1])); 	
	if (numVertices != NUMBER_OF_VERTICES_AIMED) {
		printf("error: numVertices %d NUM_AIMED %d \n",
			numVertices, NUMBER_OF_VERTICES_AIMED);
		getch();
	}
	
	numTrianglesAllocated = 2*numVertices;
	
	// Try to ALLOCATE:	
	X = new Vertex[numVertices];
	if (X == 0){
		printf("allocation error: X in object %d \n", token); getch();
		return 1;
	} else {
		INT64 address1 = (INT64)(&(X[0]));
		INT64 address2 = (INT64)(&(X[numVertices]));
		INT64 difference = address2-address1;
		INT64 est = numVertices*sizeof(Vertex);		
		printf("Allocated %d vertices. sizeof(Vertex) = %d .\n"
			      "NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
			numVertices,sizeof(Vertex),est,difference);
	};
	int ttt = sizeof(Triangle);
	INT64 howmuch = ((INT64)ttt)*numTrianglesAllocated;

	T = new Triangle[numTrianglesAllocated]; 
	if (T == 0) {
		printf("allocation error: T in object %d \n",token); getch();
		return 2;
	} else {
		INT64 address1 = (INT64)(&(T[0]));
		INT64 address2 = (INT64)(&(T[numTrianglesAllocated]));
		INT64 difference = address2-address1;
		INT64 est = numTrianglesAllocated*sizeof(Triangle);
		printf("Allocated %d triangles. sizeof(Triangle) = %d .\n"
			      "NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
			numTrianglesAllocated,sizeof(Triangle),est,difference);
	};
	
	// _____________________________________________________________________________
	// Now place the vertices: 
	// _____________________________________________________________________________
	
	iVertex = 0;
	vert = &(X[0]);
	r = INNER_A_BOUNDARY;
	Innermost_r_achieved = r;
	numInnerVertices = 0;
	numDomainVertices = 0;

	for (iRow = 0; iRow < numRows; iRow++)
	{
		printf("iRow %d numRow %d numRow1+numRow2 %d numRows %d \n",iRow,numRow[iRow],numRow1+numRow2,numRows);

		if (iRow == numRows-1) StartAvgRow = iVertex;

		theta_spacing = FULLANGLE/(real)numRow[iRow];
		theta = -HALFANGLE + ((iRow % 2 == 1)?(0.5*theta_spacing):(0.01*theta_spacing));
		// pls don't put exactly on boundary - just asking for trouble!!!
		// Offsetting at left side guarantees there is at least one place that consecutive rows are antiphased.
		
		for (i = 0; i < numRow[iRow]; i++) 
		{
			x = -r*cos(theta+PI*0.5);
			y = r*sin(theta+PI*0.5);
			
			vert->pos.x = x;
			vert->pos.y = y;
			
			if (iRow == 0) {
				vert->flags = CONCAVE_EDGE_VERTEX; // == INNERMOST
			} else {
				if (iRow == numRows-1)
				{
					vert->flags = CONVEX_EDGE_VERTEX; // == OUTERMOST
				} else {
					if (iRow <= numRow1+numRow2) {
						vert->flags = INNER_VERTEX;
					} else {
						vert->flags = DOMAIN_VERTEX; 	
					};
				};
			};
			// Rather than creating a flag for reverse z current, just
			// remember which row and where it starts and ends.
			
			++vert;
			++iVertex;		
			if (iRow <= numRow1+numRow2) {
				numInnerVertices++;
			} else {
				numDomainVertices++;
			};
			theta += theta_spacing;
		};

		if (iRow < numRow1) {
			r += r_use1;
		} else {
			if (iRow < numRow1+numRow2) // one of the inner rows, not the last one
			{
				r += r_use2;
			} else {
				if (iRow == numRow1+numRow2) {
					// last inner row 
					r = DEVICE_RADIUS_INSULATOR_OUTER+0.5*r_spacing; 
					Xdomain = vert; // next point is first of domain vertices.
				} else {
					r += r_use3;
				};
			};
		};
	};
	printf("X[36800].modulus %1.10E \n",X[36800].pos.modulus());

	getch();

	if (iVertex != numVertices) {
		printf("summat wrong. iVertex %d numVertices %d \n",iVertex,numVertices);
			getch();
		};

	// ************************************************************************************
	// Now proceed to create triangles between all the rows of vertices.
	// Need to have algorithm to walk along the row above until time to walk below.
	// ************************************************************************************
	
	iRow = 0;
	iVertexLowFirst = 0;
	int top_circled, bot_circled, top_advance;
	real gradient1, gradient2, anglenext, anglenextlow,
		anglelow, angle;
	Vertex * pNext, *pNextLow;
	iTri = 0;
	pTri = T;

	// First let's put lowest frills, which we could not avoid.
	int iNext;
	iVertex = 0;
	for (iTri = 0; iTri < numRow[0]; iTri++)
	{
		// corners 0,1 of a frill must be actual vertices.
		SetTriangleVertex(0,pTri,X + iVertex);
		iVertex++;
		if (iVertex == numRow[0]) iVertex = 0;
		SetTriangleVertex(1,pTri,X + iVertex);
	//	SetTriangleVertex(2,pTri,X + iVertex); // not used, hopefully
		pTri->cornerptr[2] = pTri->cornerptr[1];
		pTri->u8domain_flag = INNER_FRILL;
		++pTri;
	}
	
	for (iRow = 0; iRow < numRows-1; iRow++)
		// why -1 ? Because this is the bottom of the row. 
	{	
		iVertexFirst = iVertexLowFirst+numRow[iRow];
		iVertexNextFirst = iVertexFirst+numRow[iRow+1];
		
		iVertex = iVertexFirst;
		iVertexLow = iVertexLowFirst;

		top_circled = 0;
		bot_circled = 0;
		while ((bot_circled == 0) || (top_circled == 0))
		{
			
			if (iTri >= numTrianglesAllocated)
			{	printf("Ran out of triangles!\n""iVertex = %d, iTri = %d, iRow = %d. NumRows = %d\n"					
				"numTrianglesAllocated = %d""\nAny key\n", 
				iVertex,iTri,iRow,numRows, numTrianglesAllocated);
				getch(); return 1;
			};

			gradient1 = (X+iVertexLow)->pos.x/(X+iVertexLow)->pos.y;
			gradient2 = (X+iVertex)->pos.x/(X+iVertex)->pos.y;

			top_advance = (gradient1 > gradient2)?1:0;
			// Advance top iff lower one is to right of it.
			// That might not be what we want.

			// We might prefer to choose which one based on whether the new point will be 
			// closer azimuthally to the other (existing) point.
			
			// Still we can easily screw ourselves up here.
			// Try putting the following checks __first__ :

			if (bot_circled){
				top_advance = 1; // keep going and join up circle
			} else {
				if (top_circled) {
					top_advance = 0;
				} else {
					// Neither circle is completed so I guess it is all right to look forward
					// If this is the last point in the row then we should look forward to the first 
					// of this row.
					
					// Then we can compare azimuths and see which advance is more desirable to form
					// this triangle.
					if (iVertex < iVertexNextFirst-1) {
						pNext = X+iVertex+1;
						anglenext = atan2(pNext->pos.y,pNext->pos.x);
					} else {
						pNext = X+iVertexFirst;
						anglenext = atan2(pNext->pos.y, pNext->pos.x)-FULLANGLE; // make contiguous
					};
					if (iVertexLow < iVertexFirst-1) {
						pNextLow = X+iVertexLow+1;
						anglenextlow = atan2(pNextLow->pos.y,pNextLow->pos.x);
					} else {
						pNextLow = X+iVertexLowFirst;
						anglenextlow = atan2(pNextLow->pos.y,pNextLow->pos.x)-FULLANGLE;
					};
					angle = atan2((X+iVertex)->pos.y,(X+iVertex)->pos.x);
					anglelow = atan2((X+iVertexLow)->pos.y,(X+iVertexLow)->pos.x);
					
					if (angle < anglelow) {
						if (anglenext < anglenextlow){
							top_advance = 0;
							// I think all inequalities were wrong way round here.
						} else {
							// not a no-brainer:
							// short distance is at top. See whether next top point belongs to the existing
							// base point or the one following.
							if (anglelow-anglenext < angle-anglenextlow)
							{
								// those should be the positive way round.
								top_advance = 1; // less distance from here to next top
							} else {
								top_advance = 0;
							};
						};
					} else {
						if (anglenextlow < anglenext) {
							top_advance = 1;
						} else {
							// not a no-brainer:
							if (anglelow-anglenext < angle-anglenextlow)
							{
								top_advance = 1;
							} else {
								top_advance = 0;
							};
						};
					};
				};
			};

			if (top_advance) {
				// check what this routine says:
				SetTriangleVertex(2,pTri,X + iVertexLow);
				SetTriangleVertex(0,pTri,X + iVertex);

				++iVertex; 
				
				if (iVertex == iVertexNextFirst)
				{
					iVertex = iVertexFirst; // scroll back to start of upper row
					top_circled = 1;
				};
				SetTriangleVertex(1,pTri,X + iVertex);			
					
			} else {
				// make a triangle with base on the floor:									
				SetTriangleVertex(2,pTri,X + iVertex);
				SetTriangleVertex(0,pTri,X + iVertexLow);
				++iVertexLow; 
				if (iVertexLow == iVertexFirst)
				{
					iVertexLow = iVertexLowFirst; // scroll back to start of upper row
					bot_circled = 1;
				};
				SetTriangleVertex(1,pTri,X + iVertexLow);
			};
			
			// Triangle flags:

			if (iRow < numRow1+numRow2)
			{
				pTri->u8domain_flag = OUT_OF_DOMAIN;
			} else {
				if (iRow == numRow1+numRow2)
				{
					pTri->u8domain_flag = CROSSING_INS;
				} else {
					pTri->u8domain_flag = DOMAIN_TRIANGLE;
				};
			};
			
			++pTri;
			++iTri;
		}; // end while
		
		iVertexLowFirst = iVertexFirst; // move to next row and everything else hangs on just this.
	}; // next iRow

	// Now go around again for outer frills:
	iVertex = iVertexLowFirst;
	for (int iExtra = 0; iExtra < numRow[numRows-1]; iExtra++)
	{
		SetTriangleVertex(0,pTri,X + iVertex);
		iVertex++;
		if (iVertex == numVertices) iVertex = iVertexLowFirst;
		SetTriangleVertex(1,pTri,X + iVertex);
	//	SetTriangleVertex(2,pTri,X + iVertex);
		pTri->cornerptr[2] = pTri->cornerptr[1];
		
		pTri->u8domain_flag = OUTER_FRILL;
		++pTri;
		++iTri;
	}

	numTriangles = iTri;
	
	printf("Triangles used %d, Triangles allocated %d \n",numTriangles,numTrianglesAllocated);
	if (numTriangles != numTrianglesAllocated) {getch();getch();};
	
	// 5. Populate triangle neighbour list
	// ____________________________________
	
	// Can now use vertex triangle membership arrays to do this.
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if ((pTri->u8domain_flag == INNER_FRILL) || (pTri->u8domain_flag == OUTER_FRILL))
		{
			pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri);
			pTri->neighbours[0] = pTri->neighbours[2];
			pTri->neighbours[1] = pTri->neighbours[2];
		} else {
	 		pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri);
			pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri);
			pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri);
		};
		// This function now modified to return itself if there is no other shared tri.
		pTri++;	
	};
	
	InitialisePeriodic(); // for initial mesh, position-based way is fine
	
 	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->area = pTri->GetArea();
		// pTri->RecalculateEdgeNormalVectors(true); 
		// True needed for the following routine:		
		// pTri->CalculateCircumcenterAndForceToInterior(pTri->cc);
		
		// WE DO NOT USE CIRCUMCENTERS ... REPEAT ...
		// This is because we think the motion of a cell corner should
		// be based on linear v, not move as the focus of a lens.
		
		pTri->RecalculateCentroid(); // for ins-crossing tri, lies in centre of insulator intersection.
		
		// Usually want false:
		pTri->RecalculateEdgeNormalVectors(false); 
		++pTri;
	};
	
	// assumes u8EdgeFlag populated: 
	RefreshVertexNeighboursOfVerticesOrdered();  // -- ?
	this->Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	InitialPopulate(); 

	// Vert-based:
	// We want to flag all those that are connecting domain vertices to inner vertices.

	//==========================================================================================

	// Now do same for each coarse 'equilateral' mesh:
	//for (i = 0; i < NUM_COARSE_LEVELS; i++)
	//{
	//	CreateEquilateralAuxMesh(i);		
	//};

#ifdef DIRICHLET_LU
	// Don't do Dirichlet LU as it gives completely different answers.
	// Instead have to accept that a constant may be added to solution.?
	this->Coarsest.Invoke(numNonBackAuxVertices[NUM_COARSE_LEVELS-1]+1);
	this->Coarsestphi.Invoke(numAuxVertices[NUM_COARSE_LEVELS-1]+1);
#else

	this->Coarsest.Invoke(numAuxVertices[NUM_COARSE_LEVELS-1]+1);
	this->LUphi.Invoke(numAuxVertices[NUM_COARSE_LEVELS-1]);

#endif
	return 0;
} 
/*int TriMesh::CreateEquilateralAuxMeshScrewPinch(int iLevel)
{
	// For now start with always equilateral aux mesh, no following fine mesh.

	
	static real const TWOOVERSQRT3 = 2.0/sqrt(3.0);
	static real const OVERSQRT3 = 1.0/sqrt(3.0);
	static real const SQRT3OVER2 = sqrt(3.0)/2.0;

	long iRow, i, prev_N_points, N_points;
	real theta;
	static real const TWOPI = 2.0*PI;

	long iVertex, iNext, iStartCircle ,	iStartInner,iVertexInside,
				iNextCircle,iVertexPrev,iTri,iprev,iNextInner;
	real newr2;
	Tensor2 rotate;

		AuxTriangle *pTri;
		AuxVertex *pAux;
		Vector2 firstpoint, origin;

	if (1)
	{
		// circular domain version
		
		real target, twovar, r1, r2, argument, lnarg, relerr, r_spacing, InnerDelta;

		// Edge length: given delta, number of verts will be
		// sum [2 pi (r=k delta 0.866) / delta] for k = 0 until k delta 0.866 == R
		// Therefore the kth circle contributes 0.866*2pi k, say.
		// kmax = OUTER_RADIUS/0.866delta 
		// sum to k = k(k+1)/2; therefore get k(k+1) 0.866 pi verts
		// =(roughly)= (pi/0.866) R^2/delta^2
		// therefore delta^2 # = (pi/0.866) R^2 
		// therefore delta = sqrt ((pi/0.866) R^2 / #)

		real EdgeLen = sqrt((PI/0.866)*OUTER_RADIUS*OUTER_RADIUS/
			NUM_AUX_VERTS[iLevel]);

		InnerDelta = EdgeLen;
		r_AuxSpacing[iLevel] = EdgeLen; // for use in searches

		long numAuxRow[10000];

		for (int iPass = 0; iPass < 2; iPass++)
		{

			origin.x = 0.0; origin.y = 3.61;
			Vector2 temp2(InnerDelta,0.0);
			firstpoint = temp2;

			if (iPass == 1) {
				AuxX[iLevel][0].x = origin.x; 
				AuxX[iLevel][0].y = origin.y; // where we place the filament.
				
				theta = TWOPI/6.0;
				Tensor2 rotate(cos(theta),-sin(theta),sin(theta),cos(theta));
				for (i = 1; i <= 6; i++)
				{
					AuxX[iLevel][i].x = temp2.x+origin.x;
					AuxX[iLevel][i].y = temp2.y+origin.y;
					temp2 = rotate*temp2;
				};
			};
			numAuxRow[0] = 1;
			numAuxRow[1] = 6;
			
			prev_N_points = 6;
			iRow = 2;
			iVertex = numAuxRow[0]+numAuxRow[1];
			real r_outer;

			r1 = InnerDelta;
					
			if (iLevel == 0) {
				r_outer = OuterRadiusAttained + maximum_spacing; // ensure no vertex pokes out.
			} else {
				r_outer = OuterRadius[iLevel-1] + this->r_AuxSpacing[iLevel-1];
			}
			do { // circles:
				r_spacing = EdgeLen*0.866;
				r2 = r1+r_spacing;
				
				// first point: take half the previous angle that was away from its first point.
				
				N_points = (long)( (TWOPI*r2)/EdgeLen );
				
				firstpoint = (firstpoint)*(r2/r1);
				
				if (iPass == 1) {
					theta = PI/(real)(N_points); // half (new) angle
					rotate.xx = cos(theta);
					rotate.xy = -sin(theta);
					rotate.yx = sin(theta);
					rotate.yy = cos(theta);
					
					firstpoint = rotate*firstpoint;
					
					theta = TWOPI/(real)(N_points);
					rotate.xx = cos(theta);
					rotate.xy = -sin(theta);
					rotate.yx = sin(theta);
					rotate.yy = cos(theta);
				
					temp2 = firstpoint;
					for (i = 0; i < N_points; i++)
					{
						AuxX[iLevel][iVertex].x = temp2.x+origin.x;
						AuxX[iLevel][iVertex].y = temp2.y+origin.y;
						temp2 = rotate*temp2;
						iVertex++;
					}
				} else {
					numAuxRow[iRow] = N_points;
					iVertex += numAuxRow[iRow];
				};

				r1 = r2;
				prev_N_points = N_points;
				iRow++;
				
			} while (r2 < r_outer);

			OuterRadius[iLevel] = r2;
			printf("Lvl %d OuterRadius %1.10E \n",iLevel,OuterRadius[iLevel]); 
			numRowsAux[iLevel] = iRow;

			if (iPass == 0) {
				numAuxVertices[iLevel] = iVertex;
				AuxX[iLevel] = new AuxVertex[numAuxVertices[iLevel]];
				if (AuxX[iLevel] == 0){
					printf("allocation error: AuxX[%d] \n",iLevel); getch();
					return 100;
				} else {
					printf("Level %d Aimed %d Allocated %d \n",iLevel,NUM_AUX_VERTS[iLevel],numAuxVertices[iLevel]);
				};
			};
		}; // next iPass :
		// first pass establishes number of verts for dim; second pass assigns positions.
		
		
		NumTrisInRow[iLevel][0] = 0;
		
		iTri = 0;
		for (int iPass = 0; iPass < 2; iPass++)
		{
			// Now assign triangles: track azimuthally between each pair of circles.
			
			iVertexInside = 0;
			pTri = AuxT[iLevel];
			TriStartRow[iLevel][0] = 0;
			TriStartRow[iLevel][1] = 0;
			for (i = 1; i <= 6; i++) 
			{
				// place triangle to left of point i:
				iprev = i-1; if (iprev == 0) iprev = numAuxRow[1];

				iVertex = i; iVertexPrev = iprev;
				if (iPass == 1) {
					pTri->SetTriangleVertex(0,AuxX[iLevel] + iVertexInside);
					pTri->SetTriangleVertex(1,AuxX[iLevel] + iVertexPrev);
					pTri->SetTriangleVertex(2,AuxX[iLevel] + iVertex);
					++pTri;
				} else {
					iTri++;
				};				
			}
			iVertex = i; // 7
			iRow = 1;
			NumTrisInRow[iLevel][1]=6;
			iVertexInside = iVertex-numAuxRow[iRow]; // 1
			for (iRow = 2; iRow < numRowsAux[iLevel]; iRow++) // the outer circle is row iRow
			{
				
				
				// Start with triangle that has the first point from the previous circle, the first and last points of this.
				iStartCircle = iVertex;
				iStartInner = iVertexInside;
				iNextCircle = iVertex + numAuxRow[iRow];
				iVertexPrev = iVertex + numAuxRow[iRow]-1;
				if (iPass == 1) {					
					pTri->SetTriangleVertex(0,AuxX[iLevel] + iVertexInside);
					pTri->SetTriangleVertex(1,AuxX[iLevel] + iVertexPrev);
					pTri->SetTriangleVertex(2,AuxX[iLevel] + iVertex);
					++pTri;
				} else {
					NumTrisInRow[iLevel][iRow] = 0;

					TriStartRow[iLevel][iRow] = iTri;
					iTri++;					
					NumTrisInRow[iLevel][iRow]++;
				};

				// Now, we walk along the circle that has the least azimuthal distance, until we get to a triangle that features
				// again both the last and previousfirst.

				while (iVertex != iVertexPrev)
				{
					iNextInner = iVertexInside + 1; if (iNextInner == iStartCircle) iNextInner = iStartInner;
					iNext = iVertex + 1; if (iNext == iNextCircle) iNext = iStartCircle;
					
					real anglenext_inner = atan2(AuxX[iLevel][iNextInner].y-origin.y,AuxX[iLevel][iNextInner].x-origin.x);
					if (anglenext_inner <= 0.0) anglenext_inner += TWOPI;
					real anglenext_outer = atan2(AuxX[iLevel][iNext].y-origin.y,AuxX[iLevel][iNext].x-origin.x);
					if (anglenext_outer <= 0.0) anglenext_outer += TWOPI;

					// To compare angles, put in similar quadrant:
					if ((anglenext_inner > 1.5*PI) || (anglenext_outer > 1.5*PI))
					{
						if (anglenext_inner < 0.5*PI) anglenext_inner += TWOPI;
						if (anglenext_outer < 0.5*PI) anglenext_outer += TWOPI;		
					};

					if (anglenext_inner < anglenext_outer) 
					{
						// add triangle, base on inside
			
						if (iPass == 1) {
							pTri->SetTriangleVertex(0,AuxX[iLevel] + iVertex);
							pTri->SetTriangleVertex(1,AuxX[iLevel] + iVertexInside);
							pTri->SetTriangleVertex(2,AuxX[iLevel] + iNextInner);
							++pTri;
						} else {
							++iTri;
							NumTrisInRow[iLevel][iRow]++;
						};
						iVertexInside++;
						if (iVertexInside == iStartCircle) iVertexInside = iStartInner;

					} else {
						// add triangle, base on outside

						if (iPass == 1) {
							pTri->SetTriangleVertex(0,AuxX[iLevel] + iVertexInside);
							pTri->SetTriangleVertex(1,AuxX[iLevel] + iVertex);
							pTri->SetTriangleVertex(2,AuxX[iLevel] + iNext);
							++pTri;
						} else {
							++iTri;
							NumTrisInRow[iLevel][iRow]++;
						};
						iVertex++;
						//if (iVertex == iNextCircle) iVertex = iStartCircle; // unnecessary
					};
				};
				// when we reach iVertex == end of circle iVertexPrev, that will happen before we reach iStartInner 
				// but after we reach the inner one previous to that. In other words, what remains
				// is to connect iVertexPrev, iStartInner and the one previous to that.

				if (iPass == 1) {
					pTri->SetTriangleVertex(0,AuxX[iLevel] + iStartInner);
					pTri->SetTriangleVertex(1,AuxX[iLevel] + iVertexPrev);
					pTri->SetTriangleVertex(2,AuxX[iLevel] + iVertexInside);

					if (iVertexInside + 1 != iStartCircle) {
						printf("eweireijr");
						getch();
					};
					++pTri;
				} else {
					++iTri;
					NumTrisInRow[iLevel][iRow]++;
				};

				// Now set up next tri row:
				iVertex++;
				iVertexInside = iStartCircle;
			};	

			if (iPass == 0) {
				numAuxTriangles[iLevel] = iTri;

				AuxT[iLevel] = new AuxTriangle[numAuxTriangles[iLevel]];
				if (AuxT[iLevel] == 0){
					printf("allocation error: AuxT[%d] \n",iLevel); getch();
					return 1000;
				} else {
					printf("Level %d: allocated %d \n",iLevel,numAuxTriangles[iLevel]);
				};
				
				GlobalInnerT = AuxT[iLevel]; // set up pointer to parent for AuxTriangle objects

			};
		}; // iPass

	} else {
		// square patch version


	};
	
	
	// What next? set up tri neighbours
	AuxTriangle *pITri;
	
	pITri = AuxT[iLevel]; // was +1 to avoid centre disk
	for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
	{
		pITri->neighbours[0] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[2],pITri->cornerptr[1],pITri,iLevel);
 		pITri->neighbours[1] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[0],pITri->cornerptr[2],pITri,iLevel);
		pITri->neighbours[2] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[0],pITri->cornerptr[1],pITri,iLevel);
		
		pITri++;
	};
	
	AuxT[iLevel][0].periodic = 0;
	pITri = AuxT[iLevel];
	for (long i = 0; i < numAuxTriangles[iLevel]; i++)
	{
		pITri->periodic = 0;
		pITri->RecalculateEdgeNormalVectors(false);
		++pITri;
	};

	// Do we need to set vertex neighbours, given what we are going to do with it?
	// Yes because we will have coefficients from neighbours affecting this one's triangles.
	
	// Use exact same kind of thinking as in the main mesh case.
	// ie
	// a. Shuffle triangle list
	// b. Get hold of a starting point and go around triangles
	
	AuxTriangle *pITriPrev;
	AuxVertex * pVertPrev;
	long iCaret;
	long j,k;
	Vector2 cc;
	real angle[9];
	long index[9], tempind[9];
	
	pAux = AuxX[iLevel]; 
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{	
		for (i = 0; i < pAux->tri_len; i++)
		{
			pITri = AuxT[iLevel] + pAux->iTriangles[i];			
			pITri->ReturnCentre(&cc,pAux); 
			theta = CalculateAngle(cc.x-pAux->x,cc.y-pAux->y);		
			j = 0;
			while ((j < i) && (theta > angle[j])) j++; 
// if i == 1 then we can only move up to place 1, since we have 1 element already
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
		for (i = 0; i < pAux->tri_len; i++)
			tempind[i] = pAux->iTriangles[index[i]];
		for (i = 0; i < pAux->tri_len; i++)
			pAux->iTriangles[i] = tempind[i];			
		
		++pAux;
	};

	// Now going to refresh and sort the vertex neighbours.

	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{		
		//pInner->neighbours.clear();

		// decide for triangle 0 which is the most clockwise other vertex
		pITri = AuxT[iLevel] + pAux->iTriangles[0];
		pITriPrev = AuxT[iLevel] + pAux->iTriangles[pAux->tri_len-1];

		if ((pAux->flags == CONVEX_EDGE_VERTEX)) {
			
			if (pITri->cornerptr[0] == pAux) {
				if ( pITri->cornerptr[2]->flags == pAux->flags ) {
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1]; // The one that was added - actually the clockwise one.
				} else {
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2];
				};
			} else {
				if (pITri->cornerptr[1] == pAux) {
					if ( pITri->cornerptr[2]->flags == pAux->flags ) {
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					};
				} else {
					if ( pITri->cornerptr[1]->flags == pAux->flags ) {
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					};
				};
			};
		} else {
			
			// vertex from within the mesh:

			if (pITri->cornerptr[0] == pAux) {
				if ( pITriPrev->has_vertex(pITri->cornerptr[1]) ) {
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2]; // The one that was added - actually the clockwise one.
				} else {
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1];
				};
			} else {
				if (pITri->cornerptr[1] == pAux) {
					if (pITriPrev->has_vertex(pITri->cornerptr[0]) ) {
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				} else {
					if (pITriPrev->has_vertex(pITri->cornerptr[0]) ) {
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				};
			};
		};
	
		// Now do the rest of them:
		for (int i = 1; i < pAux->tri_len; i++)
		{
			pITri = AuxT[iLevel] + pAux->iTriangles[i];
			
			// whichever point is neither pAux nor pVertPrev
			if (pITri->cornerptr[0] == pAux) 
			{
				if (pITri->cornerptr[1] == pVertPrev)
				{
					pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2];
				} else {
					pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1];
				};
			} else {
				if (pITri->cornerptr[1] == pAux)
				{
					if (pITri->cornerptr[0] == pVertPrev)
					{
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				} else {
					if (pITri->cornerptr[0] == pVertPrev)
					{
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				};
			};
		};
		++pAux;
	};

	pAux = AuxX[iLevel];
	pITri = AuxT[iLevel];
	
	for (long iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		memset(&(pAux->A),0,sizeof(Vector3));
		memset(&(pAux->epsilon),0,sizeof(pAux->epsilon));
		pAux->phi = 0.0;
		pAux++;
	};
	
	// 2. Set up mapping on the level below

	AuxVertex * pAux0,*pAux1,*pAux2;
	Vector2 u0,u1,u2,u;
	real dist0sq, dist1sq, dist2sq, sum, area0, area1, area2;

	if (iLevel == 0) {
	// do in code
	} else {
		// Set up on SlimVertices of the level below
		//pAux = AuxX[iLevel-1];
		//pAux->iCoarseTriangle = 0;
		//pAux->weight[0] = 1.0;   // centre, if it ever got used
		//pAux->weight[1] = 0.0;  
		//pAux->weight[2] = 0.0;  
		 
		pAux = AuxX[iLevel-1];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel-1]; iVertex++)
		{
			pAux->iCoarseTriangle = SearchForAuxTriangleContainingPoint(pAux->x,pAux->y, iLevel);

			pITri = AuxT[iLevel] + pAux->iCoarseTriangle;
			pAux0 = pITri->cornerptr[0];
			pAux1 = pITri->cornerptr[1];
			pAux2 = pITri->cornerptr[2];
			if (pITri->periodic == 0) {
				pITri->PopulatePositions(u0,u1,u2);
			} else {
				pITri->MapLeft(u0,u1,u2);
			};
			pAux->PopulatePosition(u);
			if ((pITri->periodic > 0) && (u.x > 0.0))
				u = Anticlockwise*u; 

			dist0sq = (u0-u).dot(u0-u);
			dist1sq = (u1-u).dot(u1-u);
			dist2sq = (u2-u).dot(u2-u);
			
			//if (dist0sq < dist1sq) {
			//	if (dist0sq < dist2sq) {
			//		pAux->iCoarseVertex = pAux0-AuxX[iLevel];
			//	} else {
			//		pAux->iCoarseVertex = pAux2-AuxX[iLevel];
			//	};
			//} else {
			//	if (dist1sq < dist2sq) {
			//		pAux->iCoarseVertex = pAux1-AuxX[iLevel];
			//	} else {
			//		pAux->iCoarseVertex = pAux2-AuxX[iLevel];
			//	};
			//};

			area0 = fabs((u2-u).dot(pITri->edge_normal[0]));
			area1 = fabs((u2-u).dot(pITri->edge_normal[1]));
			area2 = fabs((u0-u).dot(pITri->edge_normal[2]));
				
#ifndef QUADWEIGHTS
			sum = area0 + area1 + area2;
			pAux->weight[0] = area0/sum;  // trying linear interp first
			pAux->weight[1] = area1/sum;
			pAux->weight[2] = area2/sum;
#else
			sum = area0*area0 + area1*area1 + area2*area2;
			pAux->weight[0] = area0*area0/sum;  
			pAux->weight[1] = area1*area1/sum;
			pAux->weight[2] = area2*area2/sum;
#endif		
			++pAux;
		};
	};
 

	long iVolley;
	bool found;
	AuxVertex * pNeighAux;

	// Set up volleys - this should be done ahead of time truly.
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		pAux->iVolley = 0;
		++pAux;
	}
	iVolley = 0;
	do {
		found = false;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			if ( pAux->iVolley == iVolley )
			{
				found = true;
				// tell all neighbours they cannot be in this volley:
				for (i = 0; i < pAux->neigh_len ; i++) // actual geometric neighbours only --- see how we fare
				{
					pNeighAux = AuxX[iLevel]+pAux->iNeighbours[i];
					if (pNeighAux->iVolley == iVolley) pNeighAux->iVolley++;
				};
			};
			++pAux;
		};
		iVolley++;
	} while (found);
	numVolleysAux[iLevel] = iVolley;
	printf("iLevel %d numVolleys = %d \n",iLevel,numVolleys);


	return 0;
}


int TriMesh::CreateEquilateralAuxMesh(int iLevel)
{
	// besides setting up the equilateral structure of verts and tris,
	// we must assign values for the level below ... for finest level do for inner mesh only.
	// For the domain we have to reassign every go.
	AuxVertex * pAux0, * pAux1, * pAux2;
	Vector2 u0,u1,u2,u;
	real dist0sq, dist1sq, dist2sq, area0, area1, area2, sum;
		real x,y,rr,r;
	long i,iRow,iTri;
	real spacing, theta_spacing, theta, r_spacing;
	long numRowprev;
	bool stop;
	long iVertex,iVertexLow,iVertexFirst,iVertexNextFirst,iVertexLowFirst;
	bool inner_stop;
	real xdist,ydist;
	real vertex_density_per_cm_sq;	
	Vertex * vert, *pVertex;
	Triangle * tri, *pTri;
	AuxVertex * pInner, * pAux;
	AuxTriangle * pITri;
	Vector2 utemp;

	long numRowAux[10000]; // local to this routine
	long startrow[10000];


	static real const TWOOVERSQRT3 = 2.0/sqrt(3.0);
	static real const OVERSQRT3 = 1.0/sqrt(3.0);
	static real const SQRT3OVER2 = sqrt(3.0)/2.0;
	
	// 1. Set up mesh (see preceding routine)
	
	spacing = sqrt((1.0/((real)(NUM_AUX_VERTS_PER_CM_SQ[iLevel])))*TWOOVERSQRT3); 
	r_spacing = spacing*SQRT3OVER2; // equilateral triangles

	// we will modify it: stretch it out to ensure lowest row is inside of the finer lowest row

	iRow = 0;
	r = Outermost_r_achieved + r_spacing*0.05; // start outside with whole domain enclosed within.
	// Careful: do not wish to evaluate that a finer point lies outside all of our triangles.
	
	// Work inwards because we don't know where the number of points gets very small and don't really want to have to visualise that.
	r_AuxOuter[iLevel] = r;
	
	// It turns out it was the wrong way round!! Too bad. Now we'll just stretch downwards.

	numRowprev = (long)(r * FULLANGLE / spacing); // 2 pi r is circumference
	numAuxVertices[iLevel] = 0; 

	// Okay, 
	// We want to go further in than the level below. But that is addressed by a stretch, which follows.
	for (; ((r > INNER_A_BOUNDARY) && (numRowprev >= 5)); r -= r_spacing) 
	{		
		if (FULLANGLE/(real)numRowprev < spacing/r)
		{
			numRowAux[iRow] = numRowprev-1;
		} else {
			numRowAux[iRow] = numRowprev;
		};
		// Must now avoid positioning according to theta_spacing -- there could be too many!
		
		numRowprev = numRowAux[iRow];
		numAuxVertices[iLevel] += numRowAux[iRow];
		++iRow;
	};
	numRowsAux[iLevel] = iRow; 
	numTrianglesAuxAllocated[iLevel] = (long)(2.02*(real)(numAuxVertices[iLevel]) );
	
	r += r_spacing; // the lowest r, that would be.
	
	r_AuxInner[iLevel] = r;

	// stretch inwards:

	if (iLevel == 0)
	{
		if (r > Innermost_r_achieved)
		{
			// stretch down:	
			r_AuxInner[iLevel] = Innermost_r_achieved-0.25*r_spacing;
			r_spacing = (r_AuxOuter[iLevel]-r_AuxInner[iLevel])/(real)(numRowsAux[iLevel]-1);
			// note, -1 because if we had 4 rows of vertices then the space is divided into 3.
		} else {
		};
	} else {
		if (r > r_AuxInner[iLevel-1])
		{
			// stretch down:
			r_AuxInner[iLevel] = r_AuxInner[iLevel-1]-0.25*r_spacing;
			r_spacing = (r_AuxOuter[iLevel]-r_AuxInner[iLevel])/(real)(numRowsAux[iLevel]-1);
		} else {
			r_AuxInner[iLevel] = r;
		};
	};
	r_AuxSpacing[iLevel] = r_spacing;
	
	// Still a bit slapdash but at least we are no longer stretching > 1 row of aux cells
	// within the innermost row above. Getting zero for coefficients on aux run is no good.
	// Haven't assured beyond a doubt that there is no finer vertex not within a coarser tri
	// so we should put in a test for that. Maybe it will come up not found when we go to assign iScratch.

	
	printf("Level %d Inner radius %1.4E number %d \n",iLevel,r_AuxInner[iLevel],
		numRowAux[iRow-1]);
	
	//numAuxVertices[iLevel]++; // allow for central position

	// Try to allocate:
	AuxX[iLevel] = new AuxVertex[numAuxVertices[iLevel] ];
	if (AuxX[iLevel] == 0) {
		printf("allocation error: AuxX %d", iLevel); getch();
		return 1;
	} else {
		INT64 address1 = (INT64)(AuxX[iLevel]); // are we sure this works? AuxX[iLevel] 
		INT64 address2 = (INT64)(&(AuxX[iLevel][numAuxVertices[iLevel]]));
		INT64 difference = address2-address1;
		INT64 est = numAuxVertices[iLevel]*sizeof(AuxVertex);
		
		printf("Allocated %d aux vertices. sizeof(AuxVertex) = %d .\n"
			      "NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
			numAuxVertices[iLevel],sizeof(AuxVertex),est,difference);
	};
	
	AuxT[iLevel] = new AuxTriangle[numTrianglesAuxAllocated[iLevel]]; 
	if (AuxT[iLevel] == 0) {
		printf("allocation error: AuxT \n"); getch();
		return 2;
	} else {
		INT64 address1 = (INT64)(&(AuxT[iLevel][0]));
		INT64 address2 = (INT64)(&(AuxT[iLevel][numTrianglesAuxAllocated[iLevel]]));
		INT64 difference = address2-address1;
		INT64 est = numTrianglesAuxAllocated[iLevel]*sizeof(AuxTriangle);
		printf("Allocated %d inner triangles. sizeof(AuxTriangle) = %d .\n"
			      "NUM*sizeof = %I64d. Allocated %I64d bytes.\n",
			numTrianglesAuxAllocated[iLevel],sizeof(AuxTriangle),est,difference);
	};
	
	GlobalInnerT = AuxT[iLevel]; // set up pointer to parent for AuxTriangle objects

	//pAux = AuxX[iLevel];
	//pAux->x = 0.0;
	//pAux->y = 0.0;
	//pAux->flags = 0; // central vertex

	pAux = AuxX[iLevel];//+1
	r = r_AuxInner[iLevel];

	numNonBackAuxVertices[iLevel] = 0; // count these

	for (iRow = numRowsAux[iLevel]-1; iRow >= 0; iRow--)			// define them from the bottom up
	{
		theta_spacing = FULLANGLE/(real)numRowAux[iRow];
		// now since we ensured that we have an integer division of the row
		// it follows that we can be safe from PB funny triangles if we do this:
		theta = -HALFANGLE + ((iRow % 2 == 1)?(0.5*theta_spacing):(0.01*theta_spacing));
		
		startrow[iRow] = pAux-AuxX[iLevel];
			
		for (i = 0; i < numRowAux[iRow]; i++)  // NOTE: do not want so many in non-offset row that points are identified
		{
			pAux->x = -r*cos(theta+PI*0.5);
			pAux->y = r*sin(theta+PI*0.5);			
			pAux->flags = 0;
			if (iRow == 0) {
				pAux->flags = 4;
			} else {
				numNonBackAuxVertices[iLevel]++;
			};
			if (iRow == numRowsAux[iLevel]-1) pAux->flags = 3; // base where we assume A constant inside.
			// Not sure we even wish to flag these any more - behaviour can be as typical auxiliary vertex.
			theta += theta_spacing;
			
			if (pAux->x < 0.0) {
				utemp.x = HalfClockwise.xx*pAux->x+HalfClockwise.xy*pAux->y;
				utemp.y = HalfClockwise.yx*pAux->x+HalfClockwise.yy*pAux->y;
			} else {
				utemp.x = HalfAnticlockwise.xx*pAux->x+HalfAnticlockwise.xy*pAux->y;
				utemp.y = HalfAnticlockwise.yx*pAux->x+HalfAnticlockwise.yy*pAux->y;
			};
			pAux->gradient = utemp.x/utemp.y;

			++pAux;	
		};
		r += r_spacing;
	};
	
	int top_circled, bot_circled, top_advance;
	real gradient1, gradient2;

	// Create triangles between central point and innermost row:
			
	pITri = AuxT[iLevel];
	iRow = numRowsAux[iLevel]-1; // innermost contains first vertices but is _last row_

	//NumTrisInRow[iLevel][iRow] = 0;
	//TriStartRow[iLevel][iRow] = 0;
	//// First row is of triangles going in to the central vertex 0
	//iVertexFirst = 1;
	//iVertexNextFirst = iVertexFirst + numRowAux[iRow];
	//top_circled = 0;

	//while (top_circled == 0)
	//{
	//	pITri->flags = 16;
	//	pITri->SetTriangleVertex(0,AuxX[iLevel]);
	//	pITri->SetTriangleVertex(1,AuxX[iLevel] + iVertexFirst);
	//	++iVertexFirst;
	//	if (iVertexFirst == iVertexNextFirst)
	//	{
	//		pITri->SetTriangleVertex(2,AuxX[iLevel] + 1); // tells triangle and vertex.
	//		pITri->periodic = 1; // interpret centre as on right => tri centre is returned to the right, becomes first neighbour ... 
	//		top_circled = 1;
	//	} else {
	//		pITri->SetTriangleVertex(2,AuxX[iLevel] + iVertexFirst); // tells triangle and vertex.
	//		pITri->periodic = 0;
	//	};
	//	++pITri;
	//	++NumTrisInRow[iLevel][iRow];
	//};
	
	//pITri = AuxT[iLevel];
	//pITri->cornerptr[0] = AuxX[iLevel];
	//pITri->cornerptr[1] = AuxX[iLevel];
	//pITri->cornerptr[2] = AuxX[iLevel];
	
	pITri = AuxT[iLevel];
	//iRow = numRowsAux[iLevel]-1; // innermost contains first vertices but is _last row_

	iVertexLowFirst = 0;//1	
	iTri = 0;//NumTrisInRow[iLevel][iRow]; 
		
	// New plan:
	// unified regardless of row evenness
	// . check which is further anticlockwise (gradient) out of lower and upper
	// (note that once we cross back to the start (across PBC) that always counts as further on)
	// . add triangle facing either up to down
	// . check for reaching the initial pair of points
	
	for (iRow = numRowsAux[iLevel]-2; iRow >= 0; iRow--)
	{
		// iVertexFirst = first in this row
		// iVertexLowFirst = first in row below
		// iVertexNextFirst = first in row above
		// iRow = upper (outer) row
		
		NumTrisInRow[iLevel][iRow] = 0;
		TriStartRow[iLevel][iRow] = iTri;
		
		iVertexFirst = iVertexLowFirst+numRowAux[iRow+1];
		// iRow+1 is the row below!
		iVertexNextFirst = iVertexFirst+numRowAux[iRow];
		
		iVertex = iVertexFirst;
		iVertexLow = iVertexLowFirst;
		
		top_circled = 0;
		bot_circled = 0;
		while ((bot_circled == 0) || (top_circled == 0))
		{
			gradient1 = (AuxX[iLevel]+iVertexLow)->x/(AuxX[iLevel]+iVertexLow)->y;
			gradient2 = (AuxX[iLevel]+iVertex)->x/(AuxX[iLevel]+iVertex)->y;
			top_advance = (gradient1 > gradient2)?1:0;
			if (bot_circled) top_advance = 1;
			if (top_circled) top_advance = 0;
			
			if (top_advance) {
				// make a triangle with base on the top
				
				pITri->flags = 0;
				if (iRow == 0) pITri->flags = 24;
				if (iRow == numRowsAux[iLevel]-2) pITri->flags = 2;
				
				// Note: this is what uses GlobalInnerT
				pITri->SetTriangleVertex(2,AuxX[iLevel] + iVertexLow);
				pITri->SetTriangleVertex(0,AuxX[iLevel] + iVertex);
				++iVertex; 
				if (iVertex == iVertexNextFirst)
				{
					iVertex = iVertexFirst; // scroll back to start of upper row
					top_circled = 1;
				};
				pITri->SetTriangleVertex(1,AuxX[iLevel]+iVertex);			
				
			} else {
				// make a triangle with base on the floor
				
				pITri->flags = 0;
				if (iRow == 0) pITri->flags = 8;
				if (iRow == numRowsAux[iLevel]-2) pITri->flags = 6; // iRow is along top of this tri row.
				
				pITri->SetTriangleVertex(2,AuxX[iLevel] + iVertex);
				pITri->SetTriangleVertex(0,AuxX[iLevel] + iVertexLow);
				++iVertexLow; 
				if (iVertexLow == iVertexFirst)
				{
					iVertexLow = iVertexLowFirst; // scroll back to start of upper row
					bot_circled = 1;
				};
				pITri->SetTriangleVertex(1,AuxX[iLevel] + iVertexLow);
			};
			++pITri;
			
			++iTri;
			if (iTri >= numTrianglesAuxAllocated[iLevel])
			{	printf("Ran out of triangles!\n""iVertex = %d, iTri = %d, iRow = %d. NumRows = %d\n""numTrianglesAllocated = %d""\nAny key\n", 
						iVertex,iTri,iRow,numRowsAux[iLevel], numTrianglesAuxAllocated[iLevel]);getch(); return 1;
			};
			
			++NumTrisInRow[iLevel][iRow];
					
		}; // end while
		iVertexLowFirst=iVertexFirst;
		
	}; // next iRow
	
	numAuxTriangles[iLevel] = iTri;
	
	printf("Aux triangles used %d, allocated %d \n",numAuxTriangles[iLevel],numTrianglesAuxAllocated[iLevel]);
	
	// What next? set up tri neighbours
	
	pITri = AuxT[iLevel]; // was +1 to avoid centre disk
	for (iTri = 0; iTri < numAuxTriangles[iLevel]; iTri++)
	{
		pITri->neighbours[0] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[2],pITri->cornerptr[1],pITri,iLevel);
 		pITri->neighbours[1] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[0],pITri->cornerptr[2],pITri,iLevel);
		
		// What flags are set above?
		
		if ((pITri->flags != 24) && (pITri->flags != 6))
			// old: topmost triangle only, has no neighbour[2]. Bottom row does have neighbours!
		{
			pITri->neighbours[2] = ReturnPointerToOtherSharedTriangleAux(pITri->cornerptr[0],pITri->cornerptr[1],pITri,iLevel);
		} else {
			pITri->neighbours[2] = pITri;
		};
		pITri++;
	};
	
	// edit InitialisePeriodic to cater to these triangles?
	AuxT[iLevel][0].periodic = 0;
	pITri = AuxT[iLevel];
	for (long i = 0; i < numRowAux[numRowsAux[iLevel]-1]; i++)
	{
		pITri->GuessPeriodic(); // do not call that for any triangle that has the (0,0) point.
		pITri->RecalculateEdgeNormalVectors(false);
		++pITri;
	};
	for (long i = numRowAux[numRowsAux[iLevel]-1]; i < numAuxTriangles[iLevel]; i++)
	{	
		pITri->GuessPeriodic(); // do not call that for any triangle that has the (0,0) point.
		pITri->RecalculateEdgeNormalVectors(false);
		++pITri;
	};
	
	// Do we need to set vertex neighbours, given what we are going to do with it?
	// Yes because we will have coefficients from neighbours affecting this one's triangles.
	
	// Use exact same kind of thinking as in the main mesh case.
	// ie
	// a. Shuffle triangle list
	// b. Get hold of a starting point and go around triangles
	
	AuxTriangle *pITriPrev;
	AuxVertex * pVertPrev;
	long iCaret;
	long j,k;
	Vector2 cc;
	real angle[9];
	long index[9], tempind[9];
	
	pAux = AuxX[iLevel]; // do for centre vertex
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{	
		for (i = 0; i < pAux->tri_len; i++)
		{
			pITri = AuxT[iLevel] + pAux->iTriangles[i];			
			// need to write:
			pITri->ReturnCentre(&cc,pAux); // if a periodic triangle, centre given must be for same tranche as pAux.
			// For zero point we want it on the right. So set that triangle to periodic = 1, and the zero point will be interpreted as on the right.
			theta = CalculateAngle(cc.x-pAux->x,cc.y-pAux->y);		
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
		for (i = 0; i < pAux->tri_len; i++)
			tempind[i] = pAux->iTriangles[index[i]];
		for (i = 0; i < pAux->tri_len; i++)
			pAux->iTriangles[i] = tempind[i];			
		
		++pAux;
	};

	// Now going to refresh and sort the vertex neighbours.

	pAux = AuxX[iLevel]; // what happens for central point?

	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{		
		//pInner->neighbours.clear();

		// decide for triangle 0 which is the most clockwise other vertex
		pITri = AuxT[iLevel] + pAux->iTriangles[0];
		pITriPrev = AuxT[iLevel] + pAux->iTriangles[pAux->tri_len-1];

		if ((pAux->flags == 4) || (pAux->flags == 3)) {
			// should find that this pITri is either flags == 6 or flags == 24
			if ((pITri->flags != 6) && (pITri->flags != 24))
			{
				printf("not right flag on pITri %d\n",pITri-AuxT[iLevel]); getch();
			};

			if (pITri->cornerptr[0] == pAux) {
				if ( pITri->cornerptr[2]->flags == pAux->flags ) {
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1]; // The one that was added - actually the clockwise one.
				} else {
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2];
				};
			} else {
				if (pITri->cornerptr[1] == pAux) {
					if ( pITri->cornerptr[2]->flags == pAux->flags ) {
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					};
				} else {
					if ( pITri->cornerptr[1]->flags == pAux->flags ) {
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					};
				};
			};
		} else {
			
			// vertex from within the mesh:

			if (pITri->cornerptr[0] == pAux) {
				if ( pITriPrev->has_vertex(pITri->cornerptr[1]) ) {
					pAux->add_neigh(pITri->cornerptr[2]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2]; // The one that was added - actually the clockwise one.
				} else {
					pAux->add_neigh(pITri->cornerptr[1]- AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1];
				};
			} else {
				if (pITri->cornerptr[1] == pAux) {
					if (pITriPrev->has_vertex(pITri->cornerptr[0]) ) {
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				} else {
					if (pITriPrev->has_vertex(pITri->cornerptr[0]) ) {
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				};
			};
		};
	
		// Now do the rest of them:
		for (int i = 1; i < pAux->tri_len; i++)
		{
			pITri = AuxT[iLevel] + pAux->iTriangles[i];
			
			// whichever point is neither pAux nor pVertPrev
			if (pITri->cornerptr[0] == pAux) 
			{
				if (pITri->cornerptr[1] == pVertPrev)
				{
					pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[2];
				} else {
					pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
					pVertPrev = pITri->cornerptr[1];
				};
			} else {
				if (pITri->cornerptr[1] == pAux)
				{
					if (pITri->cornerptr[0] == pVertPrev)
					{
						pAux->add_neigh(pITri->cornerptr[2]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[2];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				} else {
					if (pITri->cornerptr[0] == pVertPrev)
					{
						pAux->add_neigh(pITri->cornerptr[1]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[1];
					} else {
						pAux->add_neigh(pITri->cornerptr[0]-AuxX[iLevel]);
						pVertPrev = pITri->cornerptr[0];
					};
				};
			};
		};
		++pAux;
	};

	pAux = AuxX[iLevel];
	pITri = AuxT[iLevel];
	
	for (long iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		memset(&(pAux->A),0,sizeof(Vector3));
		memset(&(pAux->epsilon),0,sizeof(pAux->epsilon));
		pAux->phi = 0.0;
		pAux++;
	};
	
	// 2. Set up mapping on the level below

	if (iLevel == 0) {
	
		// Do nothing -- we will re-do at runtime

		// We can increase speed somewhat by having it remembered for
		// inner vertices-- but do not do this at first.

//
//		real r0 = r_AuxOuter[iLevel];//DOMAIN_OUTER_RADIUS+0.2*r_spacing;
//
//		//pInner = InnerX;
//		//pInner->iCoarseTriangle = 0;
//		//pInner->weight[0] = 1.0;
//		//pInner->weight[1] = 0.0;
//		//pInner->weight[2] = 0.0;
//
//		pInner = InnerX;
//		pITri = AuxT[iLevel];
//		for (iVertex = 0; iVertex < numInnerVertices; iVertex++)
//		{
//			printf("iVertex %d pInner->x %f pInner->y %f \n",iVertex,pInner->x,pInner->y);
//				
//			pInner->iCoarseTriangle = SearchForAuxTriangleContainingPoint(pInner->x,pInner->y, iLevel);	
//			
//			// Use:
//				// r_Outer[iLevel]
//				// NumTrisInRow[iLevel][iRow]
//				// this will get us a pretty good seed triangle;
//				// then do non-directed search
//
//			// Now search for which of the three coarse corner vertices is the nearest.
//			
//			pITri = AuxT[0] + pInner->iCoarseTriangle;
//			pAux0 = pITri->cornerptr[0];
//			pAux1 = pITri->cornerptr[1];
//			pAux2 = pITri->cornerptr[2];
//			if (pITri->periodic == 0) {
//				pITri->PopulatePositions(u0,u1,u2);
//			} else {
//				pITri->MapLeft(u0,u1,u2);
//			};
//			pInner->PopulatePosition(u);
//			if ((pITri->periodic > 0) && (u.x > 0.0))
//				u = Anticlockwise*u; 
//
//			dist0sq = (u0-u).dot(u0-u);
//			dist1sq = (u1-u).dot(u1-u);
//			dist2sq = (u2-u).dot(u2-u);
//		
//		//	if (dist0sq < dist1sq) {
//		//		if (dist0sq < dist2sq) {
//		//			pInner->iCoarseVertex = pAux0-AuxX[0];
//		//		} else {
//		//			pInner->iCoarseVertex = pAux2-AuxX[0];
//		//		};
//		//	} else {
//		//		if (dist1sq < dist2sq) {
//		//			pInner->iCoarseVertex = pAux1-AuxX[0];
//		//		} else {
//		//			pInner->iCoarseVertex = pAux2-AuxX[0];
//		//		};
//		//	};
//			
//			area0 = fabs((u2-u).dot(pITri->edge_normal[0]));
//			area1 = fabs((u2-u).dot(pITri->edge_normal[1]));
//			area2 = fabs((u0-u).dot(pITri->edge_normal[2]));
//			
//#ifndef QUADWEIGHTS
//				sum = area0 + area1 + area2;
//				pInner->weight[0] = area0/sum;  // trying linear interp first
//				pInner->weight[1] = area1/sum;
//				pInner->weight[2] = area2/sum;
//#else
//				sum = area0*area0 + area1*area1 + area2*area2;
//				pInner->weight[0] = area0*area0/sum;  
//				pInner->weight[1] = area1*area1/sum;
//				pInner->weight[2] = area2*area2/sum;
//#endif
//			++pInner;
//		};
	} else {
		// Set up on SlimVertices of the level below
		//pAux = AuxX[iLevel-1];
		//pAux->iCoarseTriangle = 0;
		//pAux->weight[0] = 1.0;   // centre, if it ever got used
		//pAux->weight[1] = 0.0;  
		//pAux->weight[2] = 0.0;  
		 
		pAux = AuxX[iLevel-1];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel-1]; iVertex++)
		{
			pAux->iCoarseTriangle = SearchForAuxTriangleContainingPoint(pAux->x,pAux->y, iLevel);

			pITri = AuxT[iLevel] + pAux->iCoarseTriangle;
			pAux0 = pITri->cornerptr[0];
			pAux1 = pITri->cornerptr[1];
			pAux2 = pITri->cornerptr[2];
			if (pITri->periodic == 0) {
				pITri->PopulatePositions(u0,u1,u2);
			} else {
				pITri->MapLeft(u0,u1,u2);
			};
			pAux->PopulatePosition(u);
			if ((pITri->periodic > 0) && (u.x > 0.0))
				u = Anticlockwise*u; 

			dist0sq = (u0-u).dot(u0-u);
			dist1sq = (u1-u).dot(u1-u);
			dist2sq = (u2-u).dot(u2-u);
			
			//if (dist0sq < dist1sq) {
			//	if (dist0sq < dist2sq) {
			//		pAux->iCoarseVertex = pAux0-AuxX[iLevel];
			//	} else {
			//		pAux->iCoarseVertex = pAux2-AuxX[iLevel];
			//	};
			//} else {
			//	if (dist1sq < dist2sq) {
			//		pAux->iCoarseVertex = pAux1-AuxX[iLevel];
			//	} else {
			//		pAux->iCoarseVertex = pAux2-AuxX[iLevel];
			//	};
			//};

			area0 = fabs((u2-u).dot(pITri->edge_normal[0]));
			area1 = fabs((u2-u).dot(pITri->edge_normal[1]));
			area2 = fabs((u0-u).dot(pITri->edge_normal[2]));
				
#ifndef QUADWEIGHTS
			sum = area0 + area1 + area2;
			pAux->weight[0] = area0/sum;  // trying linear interp first
			pAux->weight[1] = area1/sum;
			pAux->weight[2] = area2/sum;
#else
			sum = area0*area0 + area1*area1 + area2*area2;
			pAux->weight[0] = area0*area0/sum;  
			pAux->weight[1] = area1*area1/sum;
			pAux->weight[2] = area2*area2/sum;
#endif		
			++pAux;
		};
	};
 

	long iVolley;
	bool found;
	AuxVertex * pNeighAux;

	// Set up volleys - this should be done ahead of time truly.
	pAux = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		pAux->iVolley = 0;
		++pAux;
	}
	iVolley = 0;
	do {
		found = false;
		pAux = AuxX[iLevel];
		for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
		{
			if ( pAux->iVolley == iVolley )
			{
				found = true;
				// tell all neighbours they cannot be in this volley:
				for (i = 0; i < pAux->neigh_len ; i++) // actual geometric neighbours only --- see how we fare
				{
					pNeighAux = AuxX[iLevel]+pAux->iNeighbours[i];
					if (pNeighAux->iVolley == iVolley) pNeighAux->iVolley++;
				};
			};
			++pAux;
		};
		iVolley++;
	} while (found);
	numVolleysAux[iLevel] = iVolley;
	printf("iLevel %d numVolleys = %d \n",iLevel,numVolleys);


	return 0;
}

long TriMesh::SearchForAuxTriangleContainingPoint(real x, real y, int iLevel)
{
	// to return the triangle index

	//// (r_Outer[iLevel]-r) / r_spacing[iLevel] is number of vertex rows down from top r
	//// suppose that the triangle row beneath row i is given tristartrow[i]

	static real const angleright = atan2(1.0,GRADIENT_X_PER_Y);

	long iTri;
	AuxTriangle * pITri;
	
	if (bScrewPinch) {
		real r = sqrt(x*x+(y-SP_CENTRE_Y)*(y-SP_CENTRE_Y));

		long iRow = (long)(r / r_AuxSpacing[iLevel]);
		if (iRow > numRowsAux[iLevel]-1) iRow = numRowsAux[iLevel]-1;

		iTri = TriStartRow[iLevel][iRow];
		
		pITri = &(AuxT[iLevel][iTri]);
	
		if (iRow > 2) {
			real angle1 = atan2(pITri->cornerptr[0]->y-SP_CENTRE_Y,pITri->cornerptr[0]->x);
			//if (angle1 < 0.0) angle1 += 2.0*PI;
			real angle = atan2(y-SP_CENTRE_Y,x);
			//if (angle < 0.0) angle += 2.0*PI;
			// now both between 0 and 2PI
			real diff = angle-angle1;
			if (diff < 0.0) diff += 2.0*PI; // anticlockwise only

			iTri += (long)((diff/(2.0*PI)) * (real)NumTrisInRow[iLevel][iRow]);
			if (iTri >= numAuxTriangles[iLevel]) iTri = numAuxTriangles[iLevel]-1;
		};
		
	} else {

		real r = sqrt(x*x+y*y);
		long iRow = (long)((r_AuxOuter[iLevel]-r) / r_AuxSpacing[iLevel]);
		if (r < r_AuxInner[iLevel]) iRow = numRowsAux[iLevel]-1; // last, innermost row against centre.
	

		real angle = atan2(y,x);
	
		real ppn = 1.0-((angle-angleright)/(FULLANGLE));
		iTri = TriStartRow[iLevel][iRow] + (long)(ppn*((real)NumTrisInRow[iLevel][iRow]));
		if (iTri - TriStartRow[iLevel][iRow] > NumTrisInRow[iLevel][iRow]-1) iTri = TriStartRow[iLevel][iRow];

	};

	pITri = &(AuxT[iLevel][iTri]);
	
	// Now do non-directed search...

	if (pITri->ContainsPoint(x,y)) return iTri;
	
	ScratchSearchTris.clear();            
	ScratchSearchTris.add(pITri);
	long checkcaret = 1; // the next unchecked element
	// add neighbours to list:
	ScratchSearchTris.add_unique(pITri->neighbours[0]);
	ScratchSearchTris.add_unique(pITri->neighbours[1]);
	ScratchSearchTris.add_unique(pITri->neighbours[2]);

	do
	{
		// for each unchecked triangle: check it; if it's not a hit, add all its neighbours (unique add) to list.
		// make sure we did not run out of neighbours:
		if (checkcaret >= ScratchSearchTris.len) {
			printf("Did not find enough triangle neighbours in radar search! \n");
			getch();
		};

		pITri = (AuxTriangle *)ScratchSearchTris.ptr[checkcaret];
		if (pITri->ContainsPoint(x,y)) return pITri-AuxT[iLevel];
		
		// if we're still here, it did not contain. Add all its neighbours to look in:
		ScratchSearchTris.add_unique(pITri->neighbours[0]);
		ScratchSearchTris.add_unique(pITri->neighbours[1]);
		ScratchSearchTris.add_unique(pITri->neighbours[2]);
		
		checkcaret++;

	} while (checkcaret < 10000); // arbitrary cutoff

	return -1;
}

long TriMesh::SearchForAuxTriangleContainingPoint(real x, real y, 
												  int iLevel,
										AuxTriangle * pTriSeed)
{
	// The above function cannot work unless equilateral aux mesh.

	static real const angleright = atan2(1.0,GRADIENT_X_PER_Y);

	long iTri;
	AuxTriangle * pITri;
	
	pITri = pTriSeed;
	
	// Now do non-directed search...

	if (pITri->ContainsPoint(x,y)) return pTriSeed-AuxT[iLevel];
	
	ScratchSearchTris.clear();            
	ScratchSearchTris.add(pITri);
	long checkcaret = 1; // the next unchecked element
	// add neighbours to list:
	ScratchSearchTris.add_unique(pITri->neighbours[0]);
	ScratchSearchTris.add_unique(pITri->neighbours[1]);
	ScratchSearchTris.add_unique(pITri->neighbours[2]);

	do
	{
		// for each unchecked triangle: check it; if it's not a hit, add all its neighbours (unique add) to list.
		// make sure we did not run out of neighbours:
		if (checkcaret >= ScratchSearchTris.len) {
			printf("Did not find enough triangle neighbours in radar search! \n");
			getch();
		};

		pITri = (AuxTriangle *)ScratchSearchTris.ptr[checkcaret];
		if (pITri->ContainsPoint(x,y)){
			return pITri-AuxT[iLevel];
		}
		
		if ((pITri->neighbours[0] == pITri) ||
			(pITri->neighbours[1] == pITri) ||
			(pITri->neighbours[2] == pITri))
		{
			// debug
			pITri = pITri;
		}

		// if we're still here, it did not contain. Add all its neighbours to look in:
		ScratchSearchTris.add_unique(pITri->neighbours[0]);
		ScratchSearchTris.add_unique(pITri->neighbours[1]);
		ScratchSearchTris.add_unique(pITri->neighbours[2]);
		
		checkcaret++;

	} while (checkcaret < 10000); // arbitrary cutoff

	return -1;
}

*/
/*long TriMesh::SearchInnerInterval(real theta)
{
	static long index = 0;
	
	// move index forward and back until we find angle theta to be within.

	// Remember we stored pVertex->phi as the angle on the right.
	AuxVertex * pLeftOfInner, *pInner;
	AuxVertex * pTopFirstInner = InnerX + numInnerVertices - numRowInner[0];
	AuxVertex * pLastInner = InnerX + numInnerVertices - 1;

	// BEAR IN MIND, < MEANS TO THE RIGHT OF.
	// THINK THIS THROUGH AGAIN.

	// ALSO CORRECT WHERE THESE ARE ASSIGNED.


	// First move to left until either 
	// it is found to be > leftmost phi (the one for the PB interval), in which case we are done, 
	// or, it is at least 1 interval to the right (as regarded within tranche); or in PB is at far right...
	// then move to right 

	// can we just short-circuit this? They are actually equally spaced azimuthally except for PB!!
	// theta_spacing should be stored.

	// First rule out being to left of pTopFirstInner->phi?
	if (pTopFirstInner->phi < theta){ // less than means to right of...
		// Note that TopFirstInner->phi does not appear on the right.
		if (pLastInner->phi < theta) {
			// this happened either because theta on left and pLastInner->phi on right,
			// or both on left in which case return numRowInner[0]-1;

			// one of these is the leftmost phi
	
			if (pLastInner->phi > pTopFirstInner->phi)
				// MORE THAN MEANS LEFTMOST
			{
				return numRowInner[0]-1; // last one was periodic, contains theta
			} else {
				return 0; 
			};
		} else {
			// theta > than top first inner, < lastinner->phi
			// in this case the latter must have appeared on the left.
			return 0;
		};
	};

	// Now we know theta > pTopFirstInner and generally after this point they are distant by 
	// TopInnerThetaSpacing

	index = (long)((fabs(theta-pTopFirstInner->phi))/TopInnerThetaSpacing)+1;
	if (index == numRowInner[0]) return 0; // in this case, went beyond final marker and back to periodic ...

	// Debug checks:
	pLeftOfInner = pTopFirstInner + index-1;
	pInner = pTopFirstInner + index;
	if (theta > pLeftOfInner->phi) {
		printf("error:(theta > pLeftOfInner->phi) \n");
		getch();
	};
	if ((theta < pInner->phi) && (pInner->has_periodic_interval == 0))
	 {
		printf("error:(theta < pInner->phi) \n");
		getch();
	};
	// if it has periodic interval then to be to the right of the previous ->phi is enough.
	
	return index;
}
*/
/*
void TriMesh::SetVerticesAndIndices(VertexPNT3 * vertices[],			// better to do in the other class...
									 DWORD * indices,
									long numVerticesMax,
									int colourflag,
									int heightflag,
									int offset_data,			// how far data is from start of Vertex, for real*
									int offset_species,
									real zeroplane,
									real yscale)		
{	

	VertexPNT3 * pPNT = vertices;
	DWORD * pIndex = indices;
	
	AuxVertex * pVertex = InnerX;
	AuxTriangle * pTri;
	Vector2 position;
	real data0,data1,data2;

	real x0,x1;
	Vector2 tempvec;
	AuxVertex tempvert;
	D3DXVECTOR3 vec1, vec2;
	D3DXVECTOR3 *pVecAnticlock, *pVecClock;
	AuxVertex *pVert1,*pVert2;
	D3DXVECTOR3 normalnext;
	int quad1,quad2;//quad3,quad4;
	real grad1,grad2;
	real * ptr;
	int ii,i;
	float vx,vy,one,two;

	long numBaseVerts = 0; 
	long numTopVerts = 0; 
	
	pTri = InnerT;
	// Central vertex not used:
	pPNT->pos.x = 0.0f;
	pPNT->pos.y = 0.0f;
	pPNT->pos.z = 0.0f;
	pPNT++;
	++pVertex;

	for (long i = 1; i < numInnerVertices; ++i)
	{
		pPNT->pos.x = ((float)(pVertex->x))*xzscale;
		pPNT->pos.z = ((float)(pVertex->y))*xzscale;
		switch (heightflag)
		{
		case FLAG_FLAT_MESH:
			pPNT->pos.y = 0.0f;
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;   
			pPNT->normal.z = 0.0f;

			break;
		case FLAG_DATA_HEIGHT:
			pPNT->pos.y = zeroplane+(float)(*(((real *)pVertex)+offset_data))*yscale;
			break;

		case FLAG_VELOCITY_HEIGHT:
			ptr = (real *)pVertex + offset_data; // note: we have to be passing offset_data rel to slim vertex !
			vx = (float)(*ptr);
			++ptr;
			vy = (float)(*ptr);
			pPNT->pos.y = zeroplane+sqrt(vx*vx+vy*vy)*yscale;
			break;				
		};

		if (heightflag != FLAG_FLAT_MESH)
		{
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 0.0f;
			pPNT->normal.z = 0.0f;

			pTri = InnerT + pVertex->iTriangles[0];
			pVert1 = pTri->cornerptr[0];
			if (pVert1 == pVertex) pVert1 = pTri->cornerptr[1];

			for (ii = 0; ii < pVertex->tri_len; ii++)
			{
				pTri = InnerT + pVertex->iTriangles[ii];
				
				if (pTri->flags == 0)
				{
					if (pTri->cornerptr[0] == pVertex){
						pVert1 = pTri->cornerptr[1];
						pVert2 = pTri->cornerptr[2];
					} else {
						pVert1 = pTri->cornerptr[0];
						if (pTri->cornerptr[1] == pVertex) {
							pVert2 = pTri->cornerptr[2];
						} else {
							pVert2 = pTri->cornerptr[1];
						};
					};
					vec1.x = pVert1->x-pVertex->x;
					vec2.x = pVert2->x-pVertex->x;

					switch(heightflag)
					{
					case FLAG_DATA_HEIGHT:

						data1 = *(((real *)pVert1) + offset_data);
						data2 = *(((real *)pVert1) + offset_data);
						data0 = *(((real *)pVertex) + offset_data);

						vec1.y = zeroplane+(float)((data1-data0))*yscale;
						vec2.y = zeroplane+(float)((data2-data0))*yscale; // got to set Factor somewhere
						break;
					case FLAG_VELOCITY_HEIGHT:
						ptr = (real *)pVert1 + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						one = zeroplane+sqrt(vx*vx+vy*vy)*yscale;
						
						ptr = (real *)pVert2 + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						two = zeroplane+sqrt(vx*vx+vy*vy)*yscale;

						vec1.y = (float)(one-pPNT->pos.y);
						vec2.y = (float)(two-pPNT->pos.y);
						break;
					};
					vec1.z = pVert1->y-pVertex->y;
					vec2.z = pVert2->y-pVertex->y;

					// decide quadrant and gradient for each direction

					if (vec1.x > 0.0){
						if (vec1.z > 0.0){
							quad1 = 0;
						} else {
							quad1 = 3;
						};
					} else {
						if (vec1.z > 0.0){
							quad1 = 1;
						} else {
							quad1 = 2;
						};
					};
					if (vec2.x > 0.0){						
						if (vec2.z > 0.0){
							quad2 = 0;
						} else {
							quad2 = 3;
						};
					} else {
						if (vec2.z > 0.0){
							quad2 = 1;
						} else {
							quad2 = 2;
						};
					};

					// cases:
					// quadrant is +1 or -3 => Anticlockwise
					// quadrant is same => check gradients
					// quadrant is opposite => check against negated gradient

					if ((quad2 == quad1+1) || (quad2 == quad1-3))
					{
						pVecAnticlock = &vec2;
						pVecClock = &vec1;
					} else {
						if ((quad1 == quad2+1) || (quad1 == quad2-3))
						{
							pVecAnticlock = &vec1;
							pVecClock = &vec2;
						} else {
							
							grad1 = vec1.z/vec1.x;
							grad2 = vec2.z/vec2.x;

							// CHECK:
							if (quad1 == quad2) {
								
								// Higher gradient is then always Anticlockwise

								if (grad1 > grad2) {
									pVecAnticlock = &vec1;
									pVecClock = &vec2;
								} else {
									pVecAnticlock = &vec2;
									pVecClock = &vec1;
								};
								
							} else {
								// opposite quadrants:
								// Lower gradient grad2 means Anticlockwise 2
								// regardless of which quadrants

								if (grad2 < grad1) {
									pVecAnticlock = &vec2;
									pVecClock = &vec1;
								} else {
									pVecAnticlock = &vec1;
									pVecClock = &vec2;
								};
							};
						};
					};
					
					//D3DXVec3Cross(&(pPNT->normal),&vec1,&vec2);
					D3DXVec3Cross(&normalnext,pVecAnticlock,pVecClock);
					pPNT->normal.x += normalnext.x;
					pPNT->normal.y += normalnext.y;
					pPNT->normal.z += normalnext.z;
					// better if we took average with more.
					
				};
			};

		};


		if (colourflag == FLAG_COLOUR_MESH)
		{
			pPNT->tex0.x = 0.0f;
			pPNT->tex0.y = 1.0f;
		} else {
			if  (colourflag == FLAG_AZSEGUE_COLOUR)
			{
				// This is mightily strange ??
				// need to go over shader behaviour.

				pPNT->tex0.x = (float)(*(((real *)pVertex) + offset_data));
			};
			if (colourflag == FLAG_VELOCITY_COLOUR)
			{
				pPNT->tex0.x =(float)(*(((real *)pVertex) + offset_data));
				pPNT->tex0.y =(float)(*(((real *)pVertex) + offset_data+1));
				pPNT->tex0.z =(float)(*(((real *)pVertex) + offset_data+2));
			};
		};
		//SetVertexColour(pPNT,pVertex,colourflag,offset_species);
		
		//pPNT->tex0.x = 0.0f;
		//pPNT->tex0.y = 1.0f;	// for now do this - just to make this mesh a different colour
		
		++pPNT;
		++pVertex;
	};


	pTri = InnerT;
	long totalVertices = numInnerVertices;
	Vector2 temp;
	
	// Note that we need to add more vertices to do periodic triangles.
	
	for (i = 0; i < numInnerTriangles; ++i)
	{
		if (pTri->periodic == 0)
		{
			*pIndex = (DWORD)(pTri->cornerptr[0]-InnerX);//vertices[0];
			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[1]-InnerX);
			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[2]-InnerX);
			++pIndex;
		} else {
			// periodic:
			// Display one side? Both sides?
			
			// add additional mapped vertices.

			for (int j = 0; j < 3; j++)
			{
				if (pTri->cornerptr[j]->x < 0.0)
				{
					*pIndex = (DWORD)(pTri->cornerptr[j]-InnerX);
					++pIndex;
				} else {
					pVertex = pTri->cornerptr[j];

					pTri->cornerptr[j]->periodic_image(temp,0,1);
					pPNT->pos.x = ((float)(temp.x))*xzscale;
					pPNT->pos.z = ((float)(temp.y))*xzscale;
					pPNT->pos.y = zeroplane+(float)((*(((real *)pVertex) + offset_data))*yscale);

					if (heightflag == FLAG_VELOCITY_HEIGHT)
					{
						ptr = ((real *)pVertex) + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						pPNT->pos.y = zeroplane+sqrt(vx*vx+vy*vy)*yscale;
					};

					pPNT->normal.x = 0.0f;
					pPNT->normal.y = 1.0f;     // well, now we shall have a reason to do otherwise...
					pPNT->normal.z = 0.0f;
					
					if (colourflag == FLAG_COLOUR_MESH)
					{
						pPNT->tex0.x = 0.0f;
						pPNT->tex0.y = 1.0f;
					} else {
						if  (colourflag == FLAG_AZSEGUE_COLOUR)
						{
							pPNT->tex0.x = (float)((*(((real *)pVertex) + offset_data)));
						};
						if (colourflag == FLAG_VELOCITY_COLOUR)
						{
							pPNT->tex0.x =(float)((*(((real *)pVertex) + offset_data)));
							pPNT->tex0.y =(float)((*(((real *)pVertex) + offset_data+1)));
							pPNT->tex0.z =(float)((*(((real *)pVertex) + offset_data+2)));
						};
					};		
					//SetVertexColour(pPNT,pVertex,colourflag,offset_species);

					++pPNT;
					*pIndex = totalVertices;
					++pIndex;
					++totalVertices;
					if (totalVertices >= numVerticesMax)
					{
						printf("problems. vertex array exceeded"); getch();
					};
				};
			};
		};
		++pTri;		
		// well, the bad news is that we cannot rely on the vertices being in a given order;
		// are we set to CULL_NONE? yes...
	};	
}*/
void TriMesh::SetVerticesAndIndicesAux(int iLevel,
									   VertexPNT3 * vertices,
										  DWORD * indices,
									long numVerticesMax,
									long numTrianglesMax,
									int colourflag,
									int heightflag,
									int offset_data,			// how far data is from start of Vertex, for real*
									int offset_vcolour,
									float zeroplane,
									float yscale,
									int NTris)		
{	
	VertexPNT3 * pPNT = vertices;
	DWORD * pIndex = indices;
	
	Vertex * pVertex = AuxX[iLevel];
	Triangle * pTri;
	Vector2 position;
	real data0,data1,data2;

	real x0,x1;
	Vector2 tempvec;
	Vertex tempvert;
	D3DXVECTOR3 vec1, vec2;
	D3DXVECTOR3 *pVecAnticlock, *pVecClock;
	Vertex *pVert1,*pVert2;
	D3DXVECTOR3 normalnext;
	int quad1,quad2;//quad3,quad4;
	real grad1,grad2;
	real * ptr;
	int ii,i;
	float vx,vy,one,two;
	long tri_len, izTri[128];

	long numBaseVerts = 0; 
	long numTopVerts = 0; 
	
	pTri = AuxT[iLevel];

	for (long i = 0; i < numAuxVertices[iLevel]; ++i)
	{
		pPNT->pos.x = ((float)(pVertex->pos.x))*xzscale;
		pPNT->pos.z = ((float)(pVertex->pos.y))*xzscale;

		switch (heightflag)
		{
		case FLAG_FLAT_MESH:
			pPNT->pos.y = 0.0f;
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;   
			pPNT->normal.z = 0.0f;

			break;
		case FLAG_DATA_HEIGHT:
			pPNT->pos.y = zeroplane + (float)(*(((real *)pVertex)+offset_data))*yscale;
			break;

		case FLAG_VELOCITY_HEIGHT:
			ptr = (real *)pVertex + offset_data; // note: we have to be passing offset_data rel to slim vertex !
			vx = (float)(*ptr);
			++ptr;
			vy = (float)(*ptr);
			pPNT->pos.y = zeroplane + sqrt(vx*vx+vy*vy)*yscale; // zeroplane should be == 0.0
			break;				
		};

		if (heightflag != FLAG_FLAT_MESH)
		{
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 0.0f;
			pPNT->normal.z = 0.0f;

			tri_len = pVertex->GetTriIndexArray(izTri);
				
			for (ii = 0; ii < tri_len; ii++)
			{
				pTri = AuxT[iLevel] + izTri[ii];
				
				if (pTri->cornerptr[0] == pVertex){
					pVert1 = pTri->cornerptr[1];
					pVert2 = pTri->cornerptr[2];
				} else {
					pVert1 = pTri->cornerptr[0];
					if (pTri->cornerptr[1] == pVertex) {
						pVert2 = pTri->cornerptr[2];
					} else {
						pVert2 = pTri->cornerptr[1];
					};
				};
				vec1.x = pVert1->pos.x-pVertex->pos.x;
				vec2.x = pVert2->pos.x-pVertex->pos.x;

				switch(heightflag)
				{
				case FLAG_DATA_HEIGHT:

					data1 = *(((real *)pVert1) + offset_data);
					data2 = *(((real *)pVert2) + offset_data);
					data0 = *(((real *)pVertex) + offset_data);

					vec1.y = zeroplane + (float)((data1-data0))*yscale;
					vec2.y = zeroplane + (float)((data2-data0))*yscale; 
					break;
				case FLAG_VELOCITY_HEIGHT:
					ptr = (real *)pVert1 + offset_data;
					vx = (float)(*ptr);
					++ptr;
					vy = (float)(*ptr);
					one = zeroplane + sqrt(vx*vx+vy*vy)*yscale;
					
					ptr = (real *)pVert2 + offset_data;
					vx = (float)(*ptr);
					++ptr;
					vy = (float)(*ptr);
					two = zeroplane + sqrt(vx*vx+vy*vy)*yscale;

					vec1.y = (float)(one-pPNT->pos.y);
					vec2.y = (float)(two-pPNT->pos.y);
					break;
				};
				vec1.z = pVert1->pos.y-pVertex->pos.y;
				vec2.z = pVert2->pos.y-pVertex->pos.y;

				// decide quadrant and gradient for each direction

				if (vec1.x > 0.0){
					if (vec1.z > 0.0){
						quad1 = 0;
					} else {
						quad1 = 3;
					};
				} else {
					if (vec1.z > 0.0){
						quad1 = 1;
					} else {
						quad1 = 2;
					};
				};
				if (vec2.x > 0.0){						
					if (vec2.z > 0.0){
						quad2 = 0;
					} else {
						quad2 = 3;
					};
				} else {
					if (vec2.z > 0.0){
						quad2 = 1;
					} else {
						quad2 = 2;
					};
				};

				// cases:
				// quadrant is +1 or -3 => Anticlockwise
				// quadrant is same => check gradients
				// quadrant is opposite => check against negated gradient

				if ((quad2 == quad1+1) || (quad2 == quad1-3))
				{
					pVecAnticlock = &vec2;
					pVecClock = &vec1;
				} else {
					if ((quad1 == quad2+1) || (quad1 == quad2-3))
					{
						pVecAnticlock = &vec1;
						pVecClock = &vec2;
					} else {
						
						grad1 = vec1.z/vec1.x;
						grad2 = vec2.z/vec2.x;

						// CHECK:
						if (quad1 == quad2) {
							
							// Higher gradient is then always Anticlockwise

							if (grad1 > grad2) {
								pVecAnticlock = &vec1;
								pVecClock = &vec2;
							} else {
								pVecAnticlock = &vec2;
								pVecClock = &vec1;
							};
							
						} else {
							// opposite quadrants:
							// Lower gradient grad2 means Anticlockwise 2
							// regardless of which quadrants

							if (grad2 < grad1) {
								pVecAnticlock = &vec2;
								pVecClock = &vec1;
							} else {
								pVecAnticlock = &vec1;
								pVecClock = &vec2;
							};
						};
					};
				};
				
				
				//D3DXVec3Cross(&(pPNT->normal),&vec1,&vec2);
				D3DXVec3Cross(&normalnext,pVecAnticlock,pVecClock);
				pPNT->normal.x += normalnext.x;
				pPNT->normal.y += normalnext.y;
				pPNT->normal.z += normalnext.z;
				
			};
		};

		if (colourflag == FLAG_COLOUR_MESH)
		{
			if (pVertex->iVolley == 0) {
				pPNT->tex0.x = 0.0f;
				pPNT->tex0.y = 0.0f;
				pPNT->tex0.z = 0.0f; // RGB
			}
			if (pVertex->iVolley == 1) {
				pPNT->tex0.x = 0.0f;
				pPNT->tex0.y = 0.0f;
				pPNT->tex0.y = 1.0f;
			}
			if (pVertex->iVolley == 2) {
				pPNT->tex0.x = 0.3f;
				pPNT->tex0.y = 0.3f;
				pPNT->tex0.z = 0.3f;
			}
			if (pVertex->iVolley == 3) {
				pPNT->tex0.x = 1.0f;
				pPNT->tex0.y = 0.0f;
				pPNT->tex0.z = 0.8f;
			}
			if (pVertex->iVolley == 4) {
				pPNT->tex0.x = 0.5f;
				pPNT->tex0.y = 0.6f;
				pPNT->tex0.z = 0.0f;
			}
			if (pVertex->iVolley == 5) {
				pPNT->tex0.x = 0.2f;
				pPNT->tex0.y = 1.0f;
				pPNT->tex0.z = 0.2f;
			}
			if (pVertex->iVolley > 5) {
				pPNT->tex0.x = 0.0f;
				pPNT->tex0.y = 1.0f;
				pPNT->tex0.z = 1.0f;
			}

		} else {
			if  (colourflag == FLAG_AZSEGUE_COLOUR)
			{
				ptr = ((real *)pVertex) + offset_vcolour;
				pPNT->tex0.x = (float)(*ptr);
			};
			if ((colourflag == FLAG_VELOCITY_COLOUR) || (colourflag == FLAG_CURRENT_COLOUR))
			{
				ptr = ((real *)pVertex) + offset_vcolour;
				pPNT->tex0.x = (float)(*ptr);
				++ptr;
				pPNT->tex0.y = (float)(*ptr);
				++ptr;
				pPNT->tex0.z = (float)(*ptr);
				
			};
		};
		//SetVertexColour(pPNT,pVertex,colourflag,offset_species);
	
		//pPNT->tex0.x = 0.0f;
		//pPNT->tex0.y = 1.0f;	// for now do this - just to make this mesh a different colour

		++pPNT;
		++pVertex;
	};


	pTri = AuxT[iLevel];// do not display central disk
	long totalVertices = numAuxVertices[iLevel];
	Vector2 temp;

	// Note that we need to add more vertices to do periodic triangles.
	
	long howmany;
	if (NTris == 0){
		howmany = numAuxTriangles[iLevel];
	} else {
		howmany = NTris;
	}

	for (i = 0; i < howmany; ++i)
	{
		if (pTri->periodic == 0)
		{

			*pIndex = (DWORD)(pTri->cornerptr[0]-AuxX[iLevel]);//vertices[0];

			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[1]-AuxX[iLevel]);
			
			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[2]-AuxX[iLevel]);
			
			++pIndex;

		} else {
			// periodic:
			// Display one side
			
			// add additional mapped vertices....

			
			for (int j = 0; j < 3; j++)
			{
				if (pTri->cornerptr[j]->pos.x < 0.0)
				{
					*pIndex = (DWORD)(pTri->cornerptr[j]-AuxX[iLevel]);
					++pIndex;
				} else {
					pVertex = pTri->cornerptr[j];

					temp = Anticlockwise*pVertex->pos;
					//pTri->cornerptr[j]->periodic_image(temp,0,1);
					pPNT->pos.x = ((float)(temp.x))*xzscale;
					pPNT->pos.z = ((float)(temp.y))*xzscale;
					// altered:
					pPNT->pos.y = zeroplane + (float)((*(((real *)pVertex) + offset_data)))*yscale;

					if (heightflag == FLAG_VELOCITY_HEIGHT)
					{
						ptr = ((real *)pVertex) + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						pPNT->pos.y =zeroplane+ (sqrt(vx*vx+vy*vy))*yscale;
					};

					pPNT->normal.x = 0.0f;
					pPNT->normal.y = 1.0f;     // well, now we shall have a reason to do otherwise...
					pPNT->normal.z = 0.0f;
					
					if (colourflag == FLAG_COLOUR_MESH)
					{
						pPNT->tex0.x = 0.0f;
						pPNT->tex0.y = 1.0f;
					} else {
						if  (colourflag == FLAG_AZSEGUE_COLOUR)
						{
							pPNT->tex0.x = zeroplane + (float)(*(((real *)pVertex) + offset_vcolour));
						};
						if ((colourflag == FLAG_VELOCITY_COLOUR) || (colourflag == FLAG_CURRENT_COLOUR))
						{
							pPNT->tex0.x = zeroplane + (float)((*(((real *)pVertex) + offset_vcolour)));
							pPNT->tex0.y = zeroplane + (float)((*(((real *)pVertex) + offset_vcolour+1)));
							pPNT->tex0.z = zeroplane + (float)((*(((real *)pVertex) + offset_vcolour+2)));
						};
					};		
					//SetVertexColour(pPNT,pVertex,colourflag,offset_species);

					++pPNT;
					*pIndex = totalVertices;
					++pIndex;
					++totalVertices;
					if (totalVertices > numVerticesMax)
					{
						printf("problems. vertex array exceeded"); getch();
					};
				};
			};
		};
		++pTri;		
		// well, the bad news is that we cannot rely on the vertices being in a given order;
		// are we set to CULL_NONE? yes...
	};	



	if ( (NTris > 0) && (0) ){
	
		
		// move some end vertices to near start vertices
		pPNT =vertices+numAuxVertices[iLevel]-100;
		pVertex = AuxX[iLevel];
		for (i = 0; i < 50; i++)
		{
			pPNT->pos.x = ((float)(pVertex->pos.x)+2.0e-4)*xzscale;
			pPNT->pos.y = 0.0f;
			pPNT->pos.z = ((float)(pVertex->pos.y)+2.0e-4)*xzscale;
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;   
			pPNT->normal.z = 0.0f;
			++pPNT;

			pPNT->pos.x = ((float)(pVertex->pos.x)-2.0e-4)*xzscale;
			pPNT->pos.y = 0.0f;
			pPNT->pos.z = ((float)(pVertex->pos.y)+2.0e-4)*xzscale;
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;   
			pPNT->normal.z = 0.0f;
			++pPNT;
			
			++pVertex;
		}
		for (i = 0; i < 50; ++i)
		{
			// draw triangles around vertices
			*pIndex = (DWORD)(i);
			++pIndex;
			*pIndex = (DWORD)(numAuxVertices[iLevel]-100+i*2);
			++pIndex;
			*pIndex = (DWORD)(numAuxVertices[iLevel]-99+i*2);
			++pIndex;
		}
	}

//	printf("NTris done %d + 50\n",NTris);

}

real TriMesh::SolveConsistentTemperature(real n, real n_n)
{
	// Given two densities, what T makes the net rate of ionisation equal to zero?
	
	// start with T that is small because although it's unlikely there are multiple equilibria,
	// given the choice we will take the coldest.

	real TeV = 3.0*4.0e-14/5.0e-12; // initial T[eV]
	real temp, dFbydT, TeV_NR,expfrac;
	bool not_converged = true;
	// ionisation threshold: 0.1% to ionise in 1e-6 s
	static real const F_THRESHOLD = 1.0e3; 
	static real const SQRT2 = sqrt(2.0);
	static real const SQRTHALF = 1.0/SQRT2;
	
	//static real const E0 = 13.6; // eV
	// Use secant bisection method since I'm not absolutely 100% that Newton-Raphson
	// will not go mad.

	real sqrtTeV, TeVsq, TeV45, F, Left, Right, FLeft, FRight;

	//printf("n %1.10E n_n %1.10E \n",n,n_n);

	sqrtTeV = sqrt(TeV);
	TeVsq = TeV*TeV;
	TeV45 = TeVsq*TeVsq*sqrtTeV;
	F = n_n*(1.0e-5*sqrtTeV/(6.0*E0+TeV))*exp(-E0/TeV) 
				-	2.7e-13*n/sqrtTeV - 8.75e-27*n*n/(TeV45); // RATE: we divided by n
		
	if (F > 0.0) {
		do { // decrease temp while net ionisation
			Right = TeV;
			FRight = F;
			TeV *= 0.5;
			sqrtTeV *= SQRTHALF;
			TeVsq = TeV*TeV;
			TeV45 = TeVsq*TeVsq*sqrtTeV;
			F = n_n*(1.0e-5*sqrtTeV/(6.0*E0+TeV))*exp(-E0/TeV) 
				-	2.7e-13*n/sqrtTeV - 8.75e-27*n*n/(TeV45);
		
		//	printf("Tright %1.10E F %1.10E \n",TeV,F);
		} while (F > 0.0);

		Left = TeV;
		FLeft = F;
	} else {
		do {
			Left = TeV;
			FLeft = F;
			TeV *= 2.0;
			sqrtTeV *= SQRT2;
			TeVsq = TeV*TeV;
			TeV45 = TeVsq*TeVsq*sqrtTeV;
			F = n_n*(1.0e-5*sqrtTeV/(6.0*E0+TeV))*exp(-E0/TeV) 
					-	2.7e-13*n/sqrtTeV - 8.75e-27*n*n/(TeV45);
			
		//	printf("Tleft %1.10E F %1.10E \n",TeV,F);
			if (TeV > 12.0) {
				// summat strange
				TeV = TeV;
				getch();
			}
		} while (F < 0.0); // increase temp until net ionisation

		Right = TeV;
		FRight = F;
	};
	
	TeV = (Left*FRight-Right*FLeft)/(FRight-FLeft);

	// Now we bounded TeV by an interval and will not let NR go outside this interval
	// Moreover we can make the interval smaller every time we do a move, and if NR says go outside, 
	// we take secant instead.

	do
	{		
		sqrtTeV = sqrt(TeV);
		TeVsq = TeV*TeV;
		TeV45 = TeVsq*TeVsq*sqrtTeV;
		
		expfrac = exp(-E0/TeV);
		temp = 1.0e-5*sqrtTeV/(6.0*E0+TeV);
		F = n_n*temp*expfrac
				-	2.7e-13*n/sqrtTeV - 8.75e-27*n*n/(TeV45);

		// Note that TeV is always within the interval so now we might as well modify it.
		if (F < 0.0) { // TeV too low
			Left = TeV;
			FLeft = F;
		} else {
			Right = TeV;
			FRight = F;
		};


		dFbydT = expfrac*temp*n_n*( 0.5/TeV - 1.0/(6.0*E0+TeV) + E0/(TeV*TeV) )
						+ 0.5*2.7e-13*n/(TeV*sqrtTeV) + 4.5*8.75e-27*n/(TeV45*TeV);
		
		TeV_NR = TeV - F/dFbydT;

		if ((TeV_NR > Right) || (TeV_NR < Left) ){
			// no good, take secant instead
			
			TeV = (Left*FRight-Right*FLeft)/(FRight-FLeft);

		} else {
			TeV = TeV_NR;
		};
	//	printf("T %1.10E oldF %1.10E \n",TeV,F);
	} while (fabs(F) > F_THRESHOLD);
	
	//printf("Converged: T %1.10E F %1.10E \n",TeV,F);
		
	return TeV*kB;
}
//
//void TriMesh::InitialPopulateScrewPinch(void)
//{
//	TriMesh * pOtherMesh;
//
//	Vertex * pVert = X;
//	Triangle * pTri = T;
//	real n_ion,T_ion,n_neut,T_neut;
//
//	for (long iVertex = 0; iVertex < numVertices; iVertex++)
//	{
//
//		pVert->ion.T = SCREWPINCH_INITIAL_T;
//		pVert->neut.T = SCREWPINCH_INITIAL_T; // That won't work !
//		pVert->elec.T = SCREWPINCH_INITIAL_T;
//
//		pVert->ion.n = InitialIonDensityScrewPinch(pVert->x,pVert->y);
//		pVert->elec.n = pVert->ion.n;
//		pVert->neut.n = 0.0;
//		
//		memset(&(pVert->ion.v),0,sizeof(Vector3));
//		memset(&(pVert->neut.v),0,sizeof(Vector3));
//		memset(&(pVert->elec.v),0,sizeof(Vector3));
//		
//		memset(&(pVert->A),0,sizeof(Vector3));
//		memset(&(pVert->B),0,sizeof(Vector3));
//		memset(&(pVert->E),0,sizeof(Vector3));
//		pVert->phi = 0.0; 
//		pVert++;
//	};
//
//	real totalmass = 0.0;
//	pTri = T;
//	for (long iTri = 0; iTri < numTriangles; iTri++)
//	{
//		// set triangle values of n,T to average of vertex values!
//		// consequently infer mass, heat.
//
//		n_ion = (pTri->cornerptr[0]->ion.n + pTri->cornerptr[1]->ion.n + pTri->cornerptr[2]->ion.n)/3.0; 
//		T_ion = (pTri->cornerptr[0]->ion.T + pTri->cornerptr[1]->ion.T + pTri->cornerptr[2]->ion.T)/3.0;
//		n_neut = (pTri->cornerptr[0]->neut.n + pTri->cornerptr[1]->neut.n + pTri->cornerptr[2]->neut.n)/3.0;
//		T_neut = (pTri->cornerptr[0]->neut.T + pTri->cornerptr[1]->neut.T + pTri->cornerptr[2]->neut.T)/3.0;
//		
//		pTri->ion.mass = n_ion*(pTri->GetArea());
//		pTri->ion.heat = T_ion*(pTri->ion.mass);
//		pTri->neut.mass = 0.0;
//		pTri->neut.heat = 0.0;
//		pTri->elec.mass = pTri->ion.mass;
//		pTri->elec.heat = pTri->ion.heat;
//		
//		memset(&(pTri->ion.mom),0,sizeof(Vector3));
//		memset(&(pTri->neut.mom),0,sizeof(Vector3));
//		memset(&(pTri->elec.mom),0,sizeof(Vector3));
//		
//		totalmass += pTri->ion.mass + pTri->neut.mass;
//
//		memset(&(pTri->A),0,sizeof(Vector3));
//		memset(&(pTri->B),0,sizeof(Vector3));
//		memset(&(pTri->E),0,sizeof(Vector3));
//		
//		++pTri;
//	};
//		
//	this->RecalculateVertexVariables();
//}



void TriMesh::InitialPopulate(void)
{
	// Do this once vertices have reached their initial positions.

	// Set values of n, vx, vy, T for this species.

	TriMesh * pOtherMesh;
	
	Vertex * pVert = X;
	Triangle * pTri = T;
	real n_ion,T_ion,n_neut,T_neut;

	this->Recalculate_TriCentroids_VertexCellAreas_And_Centroids();

	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{
	//	pVert->ion.T = pVert->ion.n*T_INITIAL_CENTRE/n_INITIAL_CENTRE
	//			       + UNIFORM_T_EXTRA;

		// commented.

		n_ion = InitialIonDensity(pVert->pos.x,pVert->pos.y);
		n_neut = InitialNeutralDensity(pVert->pos.x,pVert->pos.y);

		pVert->Ion.mass = n_ion*pVert->AreaCell;
		pVert->Neut.mass = n_neut*pVert->AreaCell;
		pVert->Elec.mass = n_ion*pVert->AreaCell;

		//pVert->neut.T = NEUTRAL_INITIAL_T; // That won't work !
		
		//pVert->ion.T = this->SolveConsistentTemperature(pVert->ion.n,pVert->neut.n);
		T_ion = InitialTemperature(pVert->pos.x,pVert->pos.y);
		pVert->Ion.heat = pVert->Ion.mass*T_ion;
		pVert->Elec.heat = pVert->Elec.mass*T_ion;
		pVert->Neut.heat = pVert->Neut.mass*T_ion; // was wrong before.
		
		memset(&(pVert->Ion.mom),0,sizeof(Vector3));
		memset(&(pVert->Neut.mom),0,sizeof(Vector3));
		memset(&(pVert->Elec.mom),0,sizeof(Vector3));
		
		memset(&(pVert->A),0,sizeof(Vector3));
		memset(&(pVert->Adot),0,sizeof(Vector3));
		memset(&(pVert->B),0,sizeof(Vector3));
		memset(&(pVert->E),0,sizeof(Vector3));
		pVert->phi = 0.0; 

		memset(&(pVert->epsilon),0,sizeof(real)*4);
		
		pVert++;
	};

	pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->area = pTri->GetArea();
		++pTri;
	};
}




// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// Now we get to SEARCH type of functions
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

//
//void TriMesh::RefreshVertexNeighboursOfVertices(void) 
//{
//
//	// Note that since this uses indices, it will only be valid
//	// until we next make any change to the mesh.
// Can ionisation make mesh changes? !! No - vertices are for the bulk here.
//	
//	// ************************************
//	
//	Triangle * pTri;
//	long iVert;
//	Vertex * pVert = X;
//
//	for (iVert = 0; iVert < numVertices; iVert++)
//	{
//		pVert->neighbours.clear();
//
//		for (int i = 0; i < pVert->triangles.len; i++)
//		{
//			pTri = (Triangle *)(pVert->triangles.ptr[i]);
//
//			// subtract pointers to give an integer
//
//			if (pTri->cornerptr[0] != pVert)// don't add itself to itself
//				pVert->neighbours.add_unique((long)(pTri->cornerptr[0]-X));
//			if (pTri->cornerptr[1] != pVert)// don't add itself to itself
//				pVert->neighbours.add_unique((long)(pTri->cornerptr[1]-X));
//			if (pTri->cornerptr[2] != pVert)// don't add itself to itself
//			{
//				if (!IfZero(pTri->cornerptr[2])) // do not add nonsense if the neighbour is a wedge!
//				{
//					pVert->neighbours.add_unique((long)(pTri->cornerptr[2]-X));
//				}
//			};
//
//		};
//		++pVert;
//	};
//}*/

void TriMesh::RefreshVertexNeighboursOfVerticesOrdered(void) 
{
	// Triangle list must already be anticlockwise sorted for each vertex.
	// ^^ Therefore do that here.
	
	// We should only need to call this function once we have altered a mesh.
		
	printf("Start RVNOVO ... ");

	int EdgeFlag;
	Triangle * pTri,*pTriPrev;
	long iVertex, iCaret;
	long i,j,k;
	Vector2 cent;
	Vertex * pVertex, * pVertPrev, *relevant,*Xwhich;
	real theta, angle[100];
	long index[100];
	long tempint[100];
	long izTri[128],tri_len;

	// 1. Sort triangle lists anticlockwise!!!
	// ________________________________

	memset(angle,0,sizeof(real)*100);
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{					
		tri_len = pVertex->GetTriIndexArray(izTri);
		if (tri_len >= 100)
		{
			printf("\ncannot do it - static array not big enough\n");
			getch();
		}
		
		for (i = 0; i < tri_len; i++)
		{
			pTri = T+izTri[i];
			
			cent = pTri->GetContiguousCent_AssumingCentroidsSet(pVertex);
			theta = CalculateAngle(cent.x-pVertex->pos.x,cent.y-pVertex->pos.y);		
			
			// debug:
			if (_isnan(theta) ) {
				printf("theta nan! iVertex %d i %d \n",iVertex,i);
				getch();
			};

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
		for (i = 0; i < tri_len; i++)
			tempint[i] = izTri[index[i]];
		// And now we come to have problems.
		pVertex->SetTriIndexArray(tempint,tri_len);
		pVertex->GetTriIndexArray(izTri);
		//for (i = 0; i < tri_len; i++)
		//	pVertex->izTri[i] = tempint[i];			

		// Make sure the 0th triangle to be the most clockwise one if we are at edge.
		// .. CalculateAngle should generally work, but just in case a centre can come out below 0 angle.
		if ((pVertex->flags == CONCAVE_EDGE_VERTEX) || (pVertex->flags == CONVEX_EDGE_VERTEX))				
		{
			// Find one with neighbour == itself
			// Choose the last such one to be the first element

			// Specification calls for:
			// The 0th tri should be one that is not a frill itself
			// but is as anticlockwise as it can go and not be a frill.
			
			// Hopefully we got the correct ordering from the above.

			// Find tri:
			i = -1;
			// 1. Move clockwise until we get to a frill
			do {
				i++;
				pTri = T + izTri[i];
			} while ((pTri->u8domain_flag != OUTER_FRILL) && (pTri->u8domain_flag != INNER_FRILL));
			
			// 2. Carry on until we get to not a frill
			do {
				i++; if (i == tri_len) i = 0;
				pTri = T + izTri[i];
			} while ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL));
			
			// 3. Go and check what centroid will have been applied for a frill. Match GPU.
			// We assigned it FRILL_CENTROID_OUTER_RADIUS
			/*
			// old:
			int wend = 0;
			do 
			{
				wend= 0;
				pTri = T + izTri[i];
				if ((pTri->neighbours[0] == pTri) || 
					(pTri->neighbours[1] == pTri) ||
					(pTri->neighbours[2] == pTri))
				{
					wend = 1;
				} else {
					i++;
				}
			} while (wend == 0);
			int i1 = i;
			i++; // this will not have been the final element.
			wend = 0;
			do 
			{
				wend= 0;
				pTri = T + izTri[i];
				if ((pTri->neighbours[0] == pTri) || 
					(pTri->neighbours[1] == pTri) ||
					(pTri->neighbours[2] == pTri))
				{
					wend = 1;
				} else {
					i++;
				}
			} while (wend == 0);
			if ((i == tri_len-1) && (i1 == 0)) i = 0;			*/

			// Now rotate list:
			iCaret = i;
			for (i = 0; i < tri_len; i++)
				tempint[i] = izTri[i];
			for (i = 0; i < tri_len; i++)
			{
				izTri[i] = tempint[iCaret];
				iCaret++;
				if (iCaret == tri_len) iCaret = 0;
			};
			pVertex->SetTriIndexArray(izTri,tri_len);
		};
		++pVertex;
	};

	// Now going to refresh and sort the vertex neighbours.
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{		
		// Decide for triangle 0 which is the most clockwise other vertex
		pVertex->ClearNeighs(); 
		tri_len = pVertex->GetTriIndexArray(izTri);
		int trimax;

		pTri = T + izTri[0];
		pTriPrev = T + izTri[tri_len-1];
		
		// periodic makes testing angles awkward, so :
		// For domain vertex: just see which vertex also belongs to previous triangle.
		// For boundary vertex: just test which neighbour vertex is also on the boundary.

		if ((pVertex->flags == OUTERMOST) || (pVertex->flags == INNERMOST)) {
			
			// 25/05/17: New behaviour is different.
			
			// We only want to add an anticlockwise point for tris 0,...,tri_len-2
		/*	
			trimax = tri_len; // each further triangle adds a further neighbour.
			if (pTri->cornerptr[0] == pVertex) {
				if ( pTri->cornerptr[2]->flags == pVertex->flags ) {
					pVertex->AddNeighbourIndex(pTri->cornerptr[2]- X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[1]- X);
					pVertPrev = pTri->cornerptr[1]; 
				} else {
					pVertex->AddNeighbourIndex(pTri->cornerptr[1]- X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[2]- X);
					pVertPrev = pTri->cornerptr[2];
				};
			} else {
				if (pTri->cornerptr[1] == pVertex) {
					if ( pTri->cornerptr[2]->flags == pVertex->flags ) {
						pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertPrev = pTri->cornerptr[0];
					} else {
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
						pVertPrev = pTri->cornerptr[2];
					};
				} else {
					if ( pTri->cornerptr[1]->flags == pVertex->flags ) {
						pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertPrev = pTri->cornerptr[0];
					} else {
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
						pVertPrev = pTri->cornerptr[1];
					};
				};
			};*/

			trimax = tri_len-2;
			// And now will execute the standard code below.

		} else {
			// domain vertex: adds the more anticlockwise one only.
			// WHY??? Probably because I wanted to make this code simple.
			// So then Tri 0 will contain neighs 0 and N-1, not 0 and 1.
			// But what we want is to be CONSISTENT. So we want to add 0 first
			// Make it have 0 and 1 in first triangle.

			trimax = tri_len-1; // below, only add a point for each triangle up to the last one.
		};

		if (pTri->cornerptr[0] == pVertex) {
			if ( pTriPrev->has_vertex(pTri->cornerptr[1]) ) {
				pVertex->AddNeighbourIndex(pTri->cornerptr[1]- X);
				pVertex->AddNeighbourIndex(pTri->cornerptr[2]- X);
				pVertPrev = pTri->cornerptr[2];
			} else {
				pVertex->AddNeighbourIndex(pTri->cornerptr[2]- X);
				pVertex->AddNeighbourIndex(pTri->cornerptr[1]- X);
				pVertPrev = pTri->cornerptr[1];
			};
		} else {
			if (pTri->cornerptr[1] == pVertex) {
				if (pTriPrev->has_vertex(pTri->cornerptr[0]) ) {
					pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
					pVertPrev = pTri->cornerptr[2];
				} else {
					pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
					pVertPrev = pTri->cornerptr[0];
				};
			} else {
				if (pTriPrev->has_vertex(pTri->cornerptr[0]) ) {
					pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
					pVertPrev = pTri->cornerptr[1];
				} else {
					pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
					pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
					pVertPrev = pTri->cornerptr[0];
				};
			};
		};
		

		for (int i = 1; i < trimax; i++)
		{
			pTri = T+izTri[i];
			
			// whichever point is neither pVertex nor pVertPrev
			if (pTri->cornerptr[0] == pVertex) 
			{
				if (pTri->cornerptr[1] == pVertPrev)
				{
					pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
					pVertPrev = pTri->cornerptr[2];
				} else {
					pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
					pVertPrev = pTri->cornerptr[1];
				};
			} else {
				if (pTri->cornerptr[1] == pVertex)
				{
					if (pTri->cornerptr[0] == pVertPrev)
					{
						pVertex->AddNeighbourIndex(pTri->cornerptr[2]-X);
						pVertPrev = pTri->cornerptr[2];
					} else {
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertPrev = pTri->cornerptr[0];
					};
				} else {
					if (pTri->cornerptr[0] == pVertPrev)
					{
						pVertex->AddNeighbourIndex(pTri->cornerptr[1]-X);
						pVertPrev = pTri->cornerptr[1];
					} else {
						pVertex->AddNeighbourIndex(pTri->cornerptr[0]-X);
						pVertPrev = pTri->cornerptr[0];
					};
				};
			};
		};
		
		//DebugDetectDuplicateNeighbourInList(pVertex);
		
		++pVertex;
	};	
	
}


Triangle * TriMesh::ReturnPointerToTriangleContainingPoint(
				Triangle * pTri,             // seed for beginning triangle search
				real x, real y
				)
{

	// Note that if we are on a join, then floating point rounding can cause an infinite loop.
	// Now added a break-out clause. Therefore sometimes it will give the wrong result in such cases!

	// Note that if there are overlaps remaining, this function will often fail:
	// It relies on neighbour array actually taking us outside the triangle instead of folding us back.
	
	// And if we are just underneath the apparent edge of a triangle? Hmm. 
	// More risk is at outer edge: triangle goes inside outer arc.

	// For now special behaviour is only for edge of simulation space -- there is no 
	// special behaviour near insulator -- bear in mind.

	int i;
	bool test;
//	bool changed;
	Triangle * pNeigh;
	real destx,desty,newdestx,newdesty;
	static real const RIGHTGRAD = GRADIENT_X_PER_Y*0.5;
	static real const LEFTGRAD = -RIGHTGRAD;
	int Tranche, unsh_ind;
	long izTriPrev[128], tri_len_prev;

	Triangle * pPrevious = 0;

	static const real theta = 2.0*PI/16.0;
	static const Tensor2 anticlock(cos(theta),-sin(theta),sin(theta),cos(theta));
	static const Tensor2 clock(cos(theta),sin(theta),-sin(theta),cos(theta));
	
	//Idea:

	// We start "in" an image tranche. -1 = to left; 0 = ours; 1 = to right
	// This is represented by making the destination be mapped to left or right

	// If we are going to be in {-1,0} then clearly per tris are mapped to left; at this point
	// we are in Tranche 0.
	// If we are going to be in {0,1} then clearly per tris are mapped to right; at this point
	// we are in Tranche 0.

	// Initially determine what tranche we are in:
	//
	//if (pTri->periodic == 0)
	//{
	//	
	//	real grad1 = pTri->cornerptr[0]->x/pTri->cornerptr[0]->y;
	//	real grad2 = pTri->cornerptr[1]->x/pTri->cornerptr[1]->y;
	//	real grad3 = pTri->cornerptr[2]->x/pTri->cornerptr[2]->y;

	//	// Usual case: put ourselves in Tranche 0.
	//	Tranche = 0;
	//	// If the destination x/y < LEFTGRAD but our initial tri is fully x/y > RIGHTGRAD then we can say we shall be in 
	//	// Tranche -1
	//	if ((x/y < LEFTGRAD) && (grad1 > RIGHTGRAD) && (grad2 > RIGHTGRAD) && (grad3 > RIGHTGRAD))
	//		Tranche = -1;
	//	if ((x/y > RIGHTGRAD) && (grad1 < LEFTGRAD) && (grad2 < LEFTGRAD) && (grad3 < LEFTGRAD))
	//		Tranche = 1;
	//	
	//} else {
	//	
	//	// Starting in a per_tri so we have to first deal with this:
	//		// if a per_tri, decide which side it is going to live on.
	//		// Tranche == 0 <=> left per_tri
	//		// Tranche == 1 <=> right per_tri  == left of tranche 1
	//	if (x > 0.0){
	//		Tranche = 1;
	//	} else {
	//		Tranche = 0;
	//	};	
	//};

	// SCRAP TRANCHE SYSTEM.

	// New way:

	while (x/y < -GRADIENT_X_PER_Y) {
		newdestx = clock.xx*x + clock.xy*y; // move to right since we are in left tranche
		newdesty = clock.yx*x + clock.yy*y;
		x = newdestx;
		y = newdesty;
	};
	while (x/y > GRADIENT_X_PER_Y) {
		newdestx = anticlock.xx*x + anticlock.xy*y;
		newdesty = anticlock.yx*x + anticlock.yy*y;
		x = newdestx;
		y = newdesty;
	};

	// main body of function, all cases:

	long iterationes = 0;

	do
	{
		test = pTri->TestAgainstEdges(x,y,&pNeigh);    // neighbour in the direction it transgressed

		// Now TAE must be aware that it may be a periodic tri.

		/*

		if (Tranche == 0)
		{
		
		} else {
			if (Tranche < 0)
			{
				destx = x; desty = y;
				for (i = 0; i < -Tranche; i++)
				{
					newdestx = clock.xx*destx + clock.xy*desty; // move to right since we are in left tranche
					newdesty = clock.yx*destx + clock.yy*desty;
					destx = newdestx;
					desty = newdesty;
				};
			} else {
				destx = x; desty = y;
				for (i = 0; i < Tranche; i++)
				{
					newdestx = anticlock.xx*destx + anticlock.xy*desty;
					newdesty = anticlock.yx*destx + anticlock.yy*desty;
					destx = newdestx;
					desty = newdesty;
				};
			};
			test = pTri->TestAgainstEdges(destx,desty,&pNeigh);
		};*/

		if (test) {

			// Note TAE behaviour: if we return 1 but pNeigh==pTri then this means
			// we are out of an edge cell on the domain edge only[, and azimuthally not in there <-- not any more].
			// In that case :
			if (pNeigh == pTri) {

				printf("#");

				return pTri;
				// Think about it: 
				// if it's outside outwards, that means we must be next to this tri.
				// If it's on the inner side, nothing should turn out to be outside.

				// THIS HAS TO CHANGE IF WE PUT IN CATHODE ROD.


				Triangle * pTri2;
				Triangle * pStore = pTri;

				// Find azimuthally correct edge cell .

				real grad = x/y;
				int BaseFlag, c1, c2, iApex, iWhich;
				real grad1, grad2;
				Vector2 u[3];
				int found, use;				

				Vertex * pVertPrev = pTri->cornerptr[0];
				if (pTri->cornerptr[0]->flags < 4) 
					pVertPrev = pTri->cornerptr[1];
				// Grabbed a corner that is flags >= 4

				if (pVertPrev->flags < 4) {
					printf("error Garibaldi"); 
					// Error because 2 of the cornerptr should have been >= 4.
					getch();
				};

				BaseFlag = pVertPrev->flags;
				found = 0;
				long iiii = 0;
				do {
					iiii++;

					//1. get corners c1,c2 on edge:
					if (pTri->cornerptr[0]->flags != BaseFlag) {
						c1 = 1; c2 = 2; iApex = 0;
					};
					if (pTri->cornerptr[1]->flags != BaseFlag) {
						c1 = 0; c2 = 2; iApex = 1;
					};
					if (pTri->cornerptr[2]->flags != BaseFlag) {
						c1 = 0; c2 = 1; iApex = 2;
					};

					// compare grad to c1,c2 grads
					pTri->MapLeftIfNecessary(u[0],u[1],u[2]);
					if ((pTri->periodic > 0) && (x > 0.0))
					{
						u[0] = Clockwise*u[0]; u[1] = Clockwise*u[1]; u[2] = Clockwise*u[2];
					};
					grad1 = u[c1].x/u[c1].y;
					grad2 = u[c2].x/u[c2].y;
	
					if ((grad-grad1)*(grad-grad2) <= 0.0)
					{
						found = 1;
					} else {
						// set new pVertPrev:
						if (pTri->cornerptr[c1] == pVertPrev)
						{	use = c2;	} else { use = c1 ;};
						pVertPrev = pTri->cornerptr[use];
					
						// Now find next base triangle: belongs to new pVertPrev but is not this tri

						tri_len_prev = pVertPrev->GetTriIndexArray(izTriPrev);

						if (pTri-T == izTriPrev[0]) {
							iWhich = tri_len_prev-1;
						} else {
							iWhich = 0; // Rely on ordered triangles at vertex: base tris are at 0 and N-1.
						};
						
						pTri = T + izTriPrev[iWhich];

						if (iiii > 20000) {
							printf("error: went around too many times.\n");
							getch();
							iiii = iiii;
						};
					};
				} while (found == 0);


				test = 0; // found the one to stick with.
				
				// This approach will need to be modified if there is a hole in the domain.

			}  else {
			
				// pNeigh != pTri

				// continue and do nothing.


				//// check if pNeigh is periodic
				//// if so , are we moving to the next tranche?
				//if (pNeigh->periodic > 0)
				//{
				//	if (pTri->periodic == 0) {
				//		// GOING FROM non-periodic to periodic triangle:
				//		if (pNeigh->periodic == 1) {
				//			// one point goes Clockwise, so we entered from left side
				//			Tranche++;
				//		};// else {
				//			// one point goes anticlock, so we entered from right side;
				//			// in this case nothing happens, we are still in same tranche;
				//			// per tri is always interpreted as on the left of its tranche.
				//		//};
				//	};
				//} else {
				//	// have we left a tranche? Leave tranche by moving anticlock from per tri:
				//	if (pTri->periodic == 1) {
				//		Tranche--; 		// must have moved anticlock
				//	};
				//};
			};

			if (pPrevious == pNeigh) test = 0; // Break out if we are sending back to the previous triangle!
			pPrevious = pTri;
			pTri = pNeigh;
		};

		++iterationes;
		Tranche = 0;
		if ((iterationes >= 10000)) {
			printf("iters 10000 x %1.14E y %1.14E pTri-T %d \n%1.14E %1.14E %1.14E \nTranche %d  %1.14E %1.14E %d %d %d \n",
				x,y,pTri-T,
				pTri->cornerptr[0]->pos.x/pTri->cornerptr[0]->pos.y,
				pTri->cornerptr[1]->pos.x/pTri->cornerptr[1]->pos.y,
				pTri->cornerptr[2]->pos.x/pTri->cornerptr[2]->pos.y,
				Tranche,x/y,GRADIENT_X_PER_Y,
				pTri->cornerptr[0]->flags,
				pTri->cornerptr[1]->flags,
				pTri->cornerptr[2]->flags);
			if (iterationes % 10000 == 0) getch();
		}
	} while (test != 0); 
	
	return pTri;
}


/*void TriMesh::SearchVertexTree_And_Populate_GlobalVertexScratchList__Wedges_as_subtriangles(
												Vertex * pVertTest,
												Triangle * pTriSrc,
												//long iSrcVertex,
												real Tmax,
												int iCode)
{
	real radius;
	real dist_src_to_dest_sq,rhosq,distx,disty,exponent;
	Symmetric2 kappasum;
	real rr;
	int indexlist;
	real dist_src_to_dest,dist_refl_to_dest,hdest,SDsq;
	Vertex vert_temp0, vert_temp1, vert_temp2;
	real factor;
	static real const INSULATOR_RADIUS_SQ = DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER;
	real rr0,rr1,rr2;

	long index = (long)(pVertTest-X);
	if (GlobalVertexScratchList.contains(index)) return; // already been here
		
	if (pVertTest->flags & VERTFLAGS_HAS_PERIODIC_TRI) GlobalPeriodicSearch = true;
	
	// Note: code Triangle::flags == 3 means subtriangle of wedge, search from verts 0 and 1
	// Note: code Triangle::flags == 4 means subtriangle of wedge, search from vertex 0 only

	switch(iCode)
	{
	case R_I_L_PRESSURE:
			
		SDsq = pVertTest->sickle_sq + h*h*Tmax/m_ion;   // Note that sometimes it should be m_neutral
		
		dist_src_to_dest_sq = GetSqDistance_SetGlobalFlagNeedPeriodicImage(pTriSrc->cornerptr[0],pVertTest);
		if (pTriSrc->flags < 4)
		{
			dist_src_to_dest_sq = min(dist_src_to_dest_sq,
			                                GetSqDistance_SetGlobalFlagNeedPeriodicImage(pTriSrc->cornerptr[1],pVertTest));
		};
		if (pTriSrc->flags == 0)
			dist_src_to_dest_sq = min(dist_src_to_dest_sq,
			                                GetSqDistance_SetGlobalFlagNeedPeriodicImage(pTriSrc->cornerptr[2],pVertTest));
		
		// DEBUG !
		// **********************
#ifdef DEBUG_CRAZY
		if (dist_src_to_dest_sq > MAX_SD_CUTOFF_SQ_PRESSURE*4.0*SDsq) 
			return;
#else
		if (dist_src_to_dest_sq > MAX_SD_CUTOFF_SQ_PRESSURE*SDsq) 
			return;
#endif
		// Remember we should be liberal with contributions -- it's not only being CLOSER to another image
		// that should make us apply it. The question is whether the other image is in range of the dest at all. 

		// In practice, looking across boundary for targets should be good enough.
		break;
	/*case R_I_L_HEAT:

		SDsq = pVertTest->hdest_heat*(pVertTest->kappa+pVertSrc->kappa)/pVertSrc->n;
		MAX_SD_CUTOFF_SQ = MAX_SD_CUTOFF_SQ_HEAT;
		dist_src_to_dest_sq = GetSqDistance_SetPerInfluenceFlagOnVertex(pVertSrc,pVertTest);
		if (dist_src_to_dest_sq > MAX_SD_CUTOFF_SQ*SDsq) 
			return;

		break;
	case R_I_L_Te:
		
		// Note that it is quite possible to _only_ be in range under kappasum
		// and not 2 kappa_src, 2 kappa_dest. But we could happily allow all 3.

		dist_src_to_dest_sq = GetSqDistance_SetPerInfluenceFlagOnVertex_Full(pVertSrc,pVertTest,&distx,&disty);

		kappasum.xx = pVertSrc->kappa_e.xx + pVertTest->kappa_e.xx;
		kappasum.yy = pVertSrc->kappa_e.yy + pVertTest->kappa_e.yy;
		kappasum.xy = pVertSrc->kappa_e.xy + pVertTest->kappa_e.xy;

		rhosq = kappasum.xy*kappasum.xy/(kappasum.xx*kappasum.yy);
		// in place of distsq / var
		// we take :		
		exponent = pVertSrc->n*(distx*distx/kappasum.xx + disty*disty/kappasum.yy
			- 2.0*distx*disty*kappasum.xy/(kappasum.xx*kappasum.yy))
			/
			(pVertTest->hdest_heat*(1.0-rhosq));
		
		if (exponent > MAX_SD_CUTOFF_SQ_HEAT)
			return;

		// for now: (used below for reflection)
		MAX_SD_CUTOFF_SQ = MAX_SD_CUTOFF_SQ_HEAT;
		SDsq = pVertTest->hdest_heat*kappasum.yy/pVertSrc->n;
		break;

	case R_I_L_ELECTRON_PRESSURE:

		MAX_SD_CUTOFF_SQ = MAX_SD_CUTOFF_SQ_PRESSURE;
		SDsq = pVertTest->sickle_sq + pVertTest->reduction*h*h*pVertSrc->T/m_e;

		dist_src_to_dest_sq = GetSqDistance_SetPerInfluenceFlagOnVertex(pVertSrc,pVertTest);
		if (dist_src_to_dest_sq > MAX_SD_CUTOFF_SQ*SDsq) 
			return;
		break;*/
/*	
}
	
	// it's in, and hitherto not counted.
	
	GlobalVerticesInRange++;
	GlobalVertexScratchList.add(index); // index is this vertex.
	
	// ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
	// What we should do: at least make sure for heat that two sets are 
	// self-consistent by storing an influence list per vertex, and therefore
	// storing 
	//             vertex_influence[iSrcVert].add(iDestVert)
	//             vertex_influence[iDestVert].add_unique(iSrcVert)
	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// This of course isn't the way we might want to go - it doesn't allow
	//  independent construction of lists so might require a more complicated structure
	
	// contiguous domain <= continuous kappa



	// Now see whether the reflected src vertex is also a worthwhile source. If it is, we need
	// to set the VERTFLAGS_INFLUENCE_REFLECTED flag for this source vertex pVertSrc.

	// Preliminary check: this is easy to calculate...

	if ((pTriSrc->flags == 0) && (GlobalFlagNeedReflectedImage == false))
		// if it wasn't pTriSrc->flags == 0 then we don't need to change the GlobalFlagNeedReflectedImage flag
	{
		rr0 = pTriSrc->cornerptr[0]->x*pTriSrc->cornerptr[0]->x+pTriSrc->cornerptr[0]->y*pTriSrc->cornerptr[0]->y;
		rr1 = pTriSrc->cornerptr[1]->x*pTriSrc->cornerptr[1]->x+pTriSrc->cornerptr[1]->y*pTriSrc->cornerptr[1]->y;
		rr2 = pTriSrc->cornerptr[2]->x*pTriSrc->cornerptr[2]->x+pTriSrc->cornerptr[2]->y*pTriSrc->cornerptr[2]->y;
		rr = min(min(rr0,rr1),rr2);
		// It is obvious that the following cannot be true if rr - INSULATOR_RADIUS_SQ > MAX_SD_CUTOFF_SQ*SDsq; 
		// test that in advance for efficiency.
		if (rr - INSULATOR_RADIUS_SQ + dist_src_to_dest_sq < MAX_SD_CUTOFF_SQ_PRESSURE*SDsq)
		{
			// Okay then, it's worth testing whether the reflected pTriSrc is in range of pVertTest. 

			// Reflect our source triangle:
			
			factor = INSULATOR_RADIUS_SQ/rr;
			vert_temp0.x = pTriSrc->cornerptr[0]->x*factor;
			vert_temp0.y = pTriSrc->cornerptr[0]->y*factor; 
			vert_temp1.x = pTriSrc->cornerptr[1]->x*factor;
			vert_temp1.y = pTriSrc->cornerptr[1]->y*factor; 
			vert_temp2.x = pTriSrc->cornerptr[2]->x*factor;
			vert_temp2.y = pTriSrc->cornerptr[2]->y*factor; 
			
			// how to set up triangle list? set pointer equal to that of the source!
	//		vert_temp.triangles.ptr = pVertSrc->triangles.ptr;
	//		vert_temp.triangles.len = pVertSrc->triangles.len;

			// Don't get it - why that is needed? It's not.

			dist_src_to_dest_sq = GetSqDistance_SetGlobalFlagNeedPeriodicImage(&vert_temp0,pVertTest);
			dist_src_to_dest_sq = min(dist_src_to_dest_sq,
			                                GetSqDistance_SetGlobalFlagNeedPeriodicImage(&vert_temp1,pVertTest));
			dist_src_to_dest_sq = min(dist_src_to_dest_sq,
			                                GetSqDistance_SetGlobalFlagNeedPeriodicImage(&vert_temp2,pVertTest));
			
			// This will now do our test for us as regards periodic images:
			
			if (dist_src_to_dest_sq < MAX_SD_CUTOFF_SQ_PRESSURE*SDsq)
				GlobalFlagNeedReflectedImage = true;
			
			//pVertSrc->flags |= VERTFLAGS_INFLUENCE_REFLECTED; // set flag for reflection of tri's near this vertex

			// Note that destruction of vert_temp will also destroy pVertTest->triangles.ptr
			// if vert_temp thinks that pVertTest->triangles.ptr belongs to it.
	//		vert_temp.triangles.ptr = 0;
	//		vert_temp.triangles.len = 0;
		};
	};
	
	// Recursively call this function for each neighbour, same source:
	for (int i = 0; i < pVertTest->neighbours.len; i++)
		SearchVertexTree_And_Populate_GlobalVertexScratchList__Wedges_as_subtriangles(
												X+pVertTest->neighbours.ptr[i],
												pTriSrc,
												Tmax,
												iCode);

}
*/


long TriMesh::GetNumVerticesGraphics(void)
{
	long numVerts = numVertices;
	Triangle * pTri = &(T[0]);
	for (long i = 0; i < numTriangles; ++i)
	{
		if (pTri->periodic) numVerts += 2;
		//if (pTri->flags > 0) numVerts += 2;
		++pTri;
	};

	return numVerts;	
}

long TriMesh::GetNumVerticesGraphicsAux(int iLevel)
{
/*	long numVerts = numAuxVertices[iLevel];
	AuxTriangle * pTri = AuxT[iLevel];
	for (long i = 0; i < numAuxTriangles[iLevel]; ++i)
	{
		if (pTri->periodic) numVerts += 2;
		//if (pTri->flags > 0) numVerts += 2;
		++pTri;
	};
	return numVerts;
*/
	// How hum. Without triangles at levels, how do we propose to do graphs?
	// Ah.

	// Is that worth a rethink?
	return numAuxVertices[iLevel];
}
// we don't want AuxTriangle any more. Contestable whether we should want AuxVertex.

void SetVertexColour(VertexPNT3 * pPNT, Vertex * pVertex, int colourflag, int offset_data, int offset_vcolour)
{
	real vx;
	static real * ptr;
	
	static real const RR = (3.61+3.61-3.44)*(3.61+3.61-3.44);
	//pvars = (vertvars *)(ptr + offset_species);

		switch (colourflag)
		{
		case FLAG_COLOUR_MESH:
			pPNT->tex0.x = 0.0f;
			pPNT->tex0.y = 0.0f;
//			if (immovable.contains(i)) pPNT->tex0.x = 300.0f;
//			if (i % 50 == 0) pPNT->tex0.y = 1000.0f;	
			break;
		case FLAG_VELOCITY_COLOUR:
			
			ptr = ((real *)pVertex) + offset_vcolour;
			pPNT->tex0.x = (float)(*ptr);
			++ptr;
			pPNT->tex0.y = (float)(*ptr);
			++ptr;
			break;
		case FLAG_CURRENT_COLOUR:
				
			ptr = ((real *)pVertex) + offset_vcolour;
			pPNT->tex0.x = (float)(*ptr);
			++ptr;
			pPNT->tex0.y = (float)(*ptr);
			++ptr;
			pPNT->tex0.z = (float)(*ptr);
			//if (pVertex->x*pVertex->x+pVertex->y*pVertex->y > RR)
			//{
		//		pPNT->tex0.z = 100.0;
		//	} else {
		//		pPNT->tex0.z = 0.0;		// use for debugging
		//	};
			++ptr;
			
			break;

		case FLAG_SEGUE_COLOUR:
		case FLAG_AZSEGUE_COLOUR:
			
			ptr = ((real *)pVertex) + offset_data;
			pPNT->tex0.x = (float)(*ptr);

			break;
		case FLAG_IONISE_COLOUR:

			ptr = ((real *)pVertex) + offset_vcolour;
			pPNT->tex0.x = (float)(*ptr);
			ptr++;
			pPNT->tex0.y = (float)(*ptr);

			break;
		};
}

	
int const NUMRINGS = 20;
int const NUMANGLES = 100;

long TriMesh::GetNumKeyVerticesGraphics(long * pnumTrianglesKey) const
{
	// Do a dry run to check how many tris

	long iNumTris = NUMANGLES; // 100 points round first ring means 100 triangles.

	for (long iRing = 1; iRing < NUMRINGS; iRing++)
	{
			iNumTris += 2*NUMANGLES;
	};

	*pnumTrianglesKey = iNumTris;
	return 1 + NUMRINGS*NUMANGLES;
}


void TriMesh::SetVerticesKeyButton(VertexPNT3 * vertices, DWORD * indices, real maximum_v,int colourflag) const
{
	// FLAG_VELOCITY_COLOUR vs FLAG_CURRENT_COLOUR
	VertexPNT3 * pPNT = vertices;
	DWORD * pIndex = indices;
	long iRing,iAngle;
	real KeyRadius = 0.1; // 0.1 cm
	real KeyCentre_x = 0.0;
	real KeyCentre_y = 3.25;
	real r,theta,x,y,z,vmag;
	int iOuter,iInner,iPrevOuter,iNext;
	long totalVertices = 0;
	long triCount = 0;

	real RSTEP = KeyRadius/(real)(NUMRINGS);  
	real THETASTEP = 2.0*PI/(real)(NUMANGLES);
	x = KeyCentre_x;
	z = KeyCentre_y;
	y = 0.0f;
	pPNT->pos.x = ((float)x)*xzscale;
	pPNT->pos.z = ((float)z)*xzscale;
	pPNT->pos.y = 0.0f;
	
	pPNT->normal.x = 0.0f;
	pPNT->normal.y = 1.0f;
	pPNT->normal.z = 0.0f;
	pPNT->tex0.x = 0.0f;
	pPNT->tex0.y = 0.0f;  // set velocity to zero; use velocity shader to interpret.
	pPNT->tex0.z = maximum_v; // used if 3D colour flag
	++pPNT;
	++totalVertices;

	if (maximum_v == 0.0) maximum_v = 1.0;

	//printf("Key button maximum v %1.9E ",maximum_v);

	for (iRing = 0; iRing < NUMRINGS; iRing++)
	{
		for (iAngle = 0; iAngle < NUMANGLES; iAngle++)
		{
			r = RSTEP*(real)(iRing+1);           // 1/20 to 20/20
			theta = THETASTEP*(real)iAngle + 0.5*(real)(iRing % 2); // 0/100 to 99/100
			x = KeyCentre_x + r*cos(theta);
			z = KeyCentre_y + r*sin(theta);
			
			pPNT->pos.x = ((float)x)*xzscale;
			pPNT->pos.z = ((float)z)*xzscale;
			pPNT->pos.y = 0.0f;

			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;
			pPNT->normal.z = 0.0f;
			// Now the job is just to set the 'velocity'

			if (colourflag == FLAG_VELOCITY_COLOUR) {
				vmag = maximum_v*r/KeyRadius;
				pPNT->tex0.x = vmag*cos(theta);
				pPNT->tex0.y = vmag*sin(theta);
			} else {
				r = r/KeyRadius; 
				//pPNT->tex0.z = maximum_v*sqrt(1.0f-r*r);
				pPNT->tex0.z = maximum_v*sqrt(1.0f-r*r);
				// now ensure x*x+y*y = maximum_v^2- z*z = max_v^2 (1 - (1 - rr))
				// = r*r*maximum_v^2
				pPNT->tex0.x = r*maximum_v*(cos(theta));
				pPNT->tex0.y = r*maximum_v*(sin(theta));
				// actually same formula for x and y as in 2D v case.
			};
			++pPNT; 
			++totalVertices;
		};
	};
	--pPNT;
//	printf("tex0.x tex0.y tex0.z %1.9E %1.9E %1.9E \n",(real)(pPNT->tex0.x),(real)(pPNT->tex0.y),(real)(pPNT->tex0.z));
	++pPNT;

	// Now set up the triangles.

	// Start at innermost circle:

	iRing = 0;
	iNext = 2;
	iOuter = 1;
	triCount = 0;
	for (iAngle = 0; iAngle < NUMANGLES; iAngle++)
	{
		*pIndex = 0;
		++pIndex;
		*pIndex = iNext;
		++pIndex;
		*pIndex = iOuter; // the point below left hand side
		++pIndex;++triCount;
		iOuter = iNext;
		iNext++;
		if (iNext == NUMANGLES+1) iNext = 1;
	};
	
	iInner = 1;
	for (iRing = 1; iRing < NUMRINGS; iRing++)
	{
		for (iAngle = 0; iAngle < NUMANGLES; iAngle++)
		{
			if (iRing % 2 == 1)
			{
				// add 2 triangles:
				*pIndex = iInner;
				++pIndex;
				iOuter = iInner + NUMANGLES;
				*pIndex = iOuter; // 1, 101, 102
				++pIndex;
				iOuter++;
				if (iOuter > (iRing+1)*NUMANGLES) iOuter -= NUMANGLES; // 201 goes back to 101
				*pIndex = iOuter;
				++pIndex;++triCount;

				*pIndex = iInner;
				++pIndex;
				*pIndex = iOuter;
				++pIndex;
				iInner++;
				*pIndex = iInner;
				++pIndex;++triCount;

			} else {
				// add 2 triangles:
				*pIndex = iInner;
				++pIndex;
				iOuter = iInner + NUMANGLES;
				iPrevOuter = iOuter-1;
				if (iPrevOuter == iRing*NUMANGLES) iPrevOuter += NUMANGLES; // 200 forward to 300
				*pIndex = iPrevOuter;
				++pIndex;
				*pIndex = iOuter;
				++pIndex;++triCount;

				*pIndex = iInner;
				++pIndex;
				*pIndex = iOuter;
				++pIndex;
				iInner++;
				*pIndex = iInner;
				++pIndex;
				++triCount;
			};
		};
	};
	//numVerticesUsed[0] = totalVertices;
	//numTrianglesUsed[0] = triCount;
}

/*void FixedMesh::SetVerticesAndIndices(VertexPNT3 * vertices,			
							        DWORD * indices,	
									long numVerticesMax,
									int colourflag,
									int heightflag,
									int offset_data,			// how far data is from start of Vertex, for real*
									int offset_species,
									real heightscale)		
{
	
	VertexPNT3 * pPNT = vertices;
	DWORD * pIndex = indices;
	
	Vertex * pVertex = &(X[0]);
	Triangle * pTri;
	Vector2 position;
	real data0,data1,data2;

	real x0,x1;
	Vector2 tempvec;
	Vertex tempvert;
	D3DXVECTOR3 vec1, vec2;
	D3DXVECTOR3 *pVecAnticlock, *pVecClock;
	Vertex *pVert1,*pVert2;
	D3DXVECTOR3 normalnext;
	int quad1,quad2;//quad3,quad4;
	real grad1,grad2;
	real * ptr;
	int ii,i;
	float vx,vy,one,two;

	// numVertices = number of actual vertices
	// numBaseVerts = number of additional fake vertices on insulator
	long numBaseVerts = 0; // cannot assume numRow[0] is fresh... count wedges instead
	long numTopVerts = 0; // cannot assume numRow[0] is fresh... count wedges instead
	
	pTri = T;

	for (long i = 0; i < numVertices; ++i)
	{
		pPNT->pos.x = ((float)(pVertex->x))*xzscale;
		pPNT->pos.z = ((float)(pVertex->y))*xzscale;
		switch (heightflag)
		{
		case FLAG_FLAT_MESH:
			pPNT->pos.y = 0.0f;
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;   
			pPNT->normal.z = 0.0f;

			break;
		case FLAG_DATA_HEIGHT:
			pPNT->pos.y = (float)(*(((real *)pVertex)+offset_data))*heightscale;
			break;

		case FLAG_VELOCITY_HEIGHT:
			ptr = (real *)pVertex + offset_data;
			vx = (float)(*ptr);
			++ptr;
			vy = (float)(*ptr);
			pPNT->pos.y = sqrt(vx*vx+vy*vy)*(float)heightscale;

			break;				
		};

		if (heightflag != FLAG_FLAT_MESH)
		{
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 0.0f;
			pPNT->normal.z = 0.0f;

			pTri = (Triangle *)(pVertex->triangles.ptr[0]);
			pVert1 = pTri->cornerptr[0];
			if (pVert1 == pVertex) pVert1 = pTri->cornerptr[1];


				// there will be one corner that is equal to neither
				// pVertex (the centre) nor pVert1
				// If this corner is nonzero, use as pVert2
				// then set pVert1 = pVert2
				// If it is == INS_VERT or HIGH_VERT, we want
				// to carry pVert1, move on, and this time
				// seek the vertex that is neither 0 nor pVertex
				// that is then used as pVert2, then pVert1=pSeek=pVert2
				
				// Reason this won't work: the triangles are not in sequence.
				// Is there an efficient way to get circle of vertices??
				// Worth a thought. :/


			// Alternative way: (wedges not counted at all) take each tri,
			// decide which thing is Clockwise; this is first operand.
			// This always works: cannot have angles > 180 ? creates shared edge


			for (ii = 0; ii < pVertex->triangles.len; ii++)
			{
				pTri = (Triangle *)(pVertex->triangles.ptr[ii]);
				
				if (pTri->flags == 0)
				{
					if (pTri->cornerptr[0] == pVertex){
						pVert1 = pTri->cornerptr[1];
						pVert2 = pTri->cornerptr[2];
					} else {
						pVert1 = pTri->cornerptr[0];
						if (pTri->cornerptr[1] == pVertex) {
							pVert2 = pTri->cornerptr[2];
						} else {
							pVert2 = pTri->cornerptr[1];
						};
					};
					vec1.x = pVert1->x-pVertex->x;
					vec2.x = pVert2->x-pVertex->x;

					switch(heightflag)
					{
					case FLAG_DATA_HEIGHT:

						data1 = *(((real *)pVert1) + offset_data);
						data2 = *(((real *)pVert1) + offset_data);
						data0 = *(((real *)pVertex) + offset_data);

						vec1.y = (float)((data1-data0)*heightscale);
						vec2.y = (float)((data2-data0)*heightscale); // got to set Factor somewhere
						break;
					//case FLAG_HEIGHT_T:

					//	vec1.y = (float)((pVert1->T - pVertex->T)*Tfactor);
					//	vec2.y = (float)((pVert2->T - pVertex->T)*Tfactor);
					//	break;
					case FLAG_VELOCITY_HEIGHT:
						ptr = (real *)pVert1 + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						one = sqrt(vx*vx+vy*vy)*(float)heightscale;
						
						ptr = (real *)pVert2 + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						two = sqrt(vx*vx+vy*vy)*(float)heightscale;

						vec1.y = (float)(one-pPNT->pos.y);
						vec2.y = (float)(two-pPNT->pos.y);
						break;
					};
					vec1.z = pVert1->y-pVertex->y;
					vec2.z = pVert2->y-pVertex->y;

					// decide quadrant and gradient for each direction

					if (vec1.x > 0.0){
						if (vec1.z > 0.0){
							quad1 = 0;
						} else {
							quad1 = 3;
						};
					} else {
						if (vec1.z > 0.0){
							quad1 = 1;
						} else {
							quad1 = 2;
						};
					};
					if (vec2.x > 0.0){						
						if (vec2.z > 0.0){
							quad2 = 0;
						} else {
							quad2 = 3;
						};
					} else {
						if (vec2.z > 0.0){
							quad2 = 1;
						} else {
							quad2 = 2;
						};
					};

					// cases:
					// quadrant is +1 or -3 => Anticlockwise
					// quadrant is same => check gradients
					// quadrant is opposite => check against negated gradient

					if ((quad2 == quad1+1) || (quad2 == quad1-3))
					{
						pVecAnticlock = &vec2;
						pVecClock = &vec1;
					} else {
						if ((quad1 == quad2+1) || (quad1 == quad2-3))
						{
							pVecAnticlock = &vec1;
							pVecClock = &vec2;
						} else {
							
							grad1 = vec1.z/vec1.x;
							grad2 = vec2.z/vec2.x;

							// CHECK:
							if (quad1 == quad2) {
								
								// Higher gradient is then always Anticlockwise

								if (grad1 > grad2) {
									pVecAnticlock = &vec1;
									pVecClock = &vec2;
								} else {
									pVecAnticlock = &vec2;
									pVecClock = &vec1;
								};
								
							} else {
								// opposite quadrants:
								// Lower gradient grad2 means Anticlockwise 2
								// regardless of which quadrants

								if (grad2 < grad1) {
									pVecAnticlock = &vec2;
									pVecClock = &vec1;
								} else {
									pVecAnticlock = &vec1;
									pVecClock = &vec2;
								};
							};
						};
					};
					
					
					//D3DXVec3Cross(&(pPNT->normal),&vec1,&vec2);
					D3DXVec3Cross(&normalnext,pVecAnticlock,pVecClock);
					pPNT->normal.x += normalnext.x;
					pPNT->normal.y += normalnext.y;
					pPNT->normal.z += normalnext.z;
					
				};
			};

		};
		
		if (colourflag == FLAG_COLOUR_MESH)
		{
			pPNT->tex0.x = 0.0f;
			pPNT->tex0.y = 1.0f;
		} else {
			if  (colourflag == FLAG_AZSEGUE_COLOUR)
			{
				pPNT->tex0.x = (float)((*(((real *)pVertex) + offset_data))*heightscale);
			};
			if (colourflag == FLAG_VELOCITY_COLOUR)
			{
				pPNT->tex0.x =(float)((*(((real *)pVertex) + offset_data)));
				pPNT->tex0.y =(float)((*(((real *)pVertex) + offset_data+1)));
				pPNT->tex0.z =(float)((*(((real *)pVertex) + offset_data+2)));
			};
		};
		//SetVertexColour(pPNT,pVertex,colourflag,offset_species);
	
		//pPNT->tex0.x = 0.0f;
		//pPNT->tex0.y = 1.0f;	// for now do this - just to make this mesh a different colour

		++pPNT;
		++pVertex;
	};
	
	
	pTri = T;
	long totalVertices = numVertices;
	Vector2 temp;

	// Note that we need to add more vertices to do periodic triangles.
	
	for (i = 0; i < numTriangles; ++i)
	{
		if (pTri->periodic == 0)
		{

			*pIndex = (DWORD)(pTri->cornerptr[0]-X);//vertices[0];

			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[1]-X);
			
			++pIndex;
			*pIndex = (DWORD)(pTri->cornerptr[2]-X);
			
			++pIndex;

		} else {
			// periodic:
			// Display one side? Both sides?
			
			// add additional mapped vertices.

			for (int j = 0; j < 3; j++)
			{
				if (pTri->cornerptr[j]->x < 0.0)
				{
					*pIndex = (DWORD)(pTri->cornerptr[j]-X);
					++pIndex;
				} else {

					pVertex = pTri->cornerptr[j];

					pTri->cornerptr[j]->periodic_image(temp,0);
					pPNT->pos.x = ((float)(temp.x))*xzscale;
					pPNT->pos.z = ((float)(temp.y))*xzscale;
					// altered:
					pPNT->pos.y = (float)((*(((real *)pVertex) + offset_data))*heightscale);

					if (heightflag == FLAG_VELOCITY_HEIGHT)
					{
						ptr = ((real *)pVertex) + offset_data;
						vx = (float)(*ptr);
						++ptr;
						vy = (float)(*ptr);
						pPNT->pos.y = sqrt(vx*vx+vy*vy)*(float)heightscale;
					};

					pPNT->normal.x = 0.0f;
					pPNT->normal.y = 1.0f;     // well, now we shall have a reason to do otherwise...
					pPNT->normal.z = 0.0f;
					
					if (colourflag == FLAG_COLOUR_MESH)
					{
						pPNT->tex0.x = 0.0f;
						pPNT->tex0.y = 1.0f;
					} else {
						if  (colourflag == FLAG_AZSEGUE_COLOUR)
						{
							pPNT->tex0.x = (float)((*(((real *)pVertex) + offset_data))*heightscale);
						};
						if (colourflag == FLAG_VELOCITY_COLOUR)
						{
							pPNT->tex0.x =(float)((*(((real *)pVertex) + offset_data)));
							pPNT->tex0.y =(float)((*(((real *)pVertex) + offset_data+1)));
							pPNT->tex0.z =(float)((*(((real *)pVertex) + offset_data+2)));
						};
					};		
					//SetVertexColour(pPNT,pVertex,colourflag,offset_species);

					++pPNT;
					*pIndex = totalVertices;
					++pIndex;
					++totalVertices;
					if (totalVertices >= numVerticesMax)
					{
						printf("problems. vertex array exceeded"); getch();
					};
				};
			};
		};
		++pTri;		
		// well, the bad news is that we cannot rely on the vertices being in a given order;
		// are we set to CULL_NONE? yes...
	};	
}
*/
void TriMesh::SetVerticesAndIndices(VertexPNT3 * vertices[],			// better to do in the other class...
							        DWORD * indices[],				// let's hope this means an array of pointers
									long const numVerticesMax[], 
									long const numTrianglesMax[],	// pass it the integer counts so that it can test for overrun
					                long numVerticesUsed[], 
									long numTrianglesUsed[],
									int colourflag,
									int heightflag,
									int offset_data,			// how far data is from start of Vertex, for real*
									int offset_vcolour,
									float zeroplane, float yscale,
									bool boolDisplayInnerMesh)		const
{
	// we use vertices[0], indices[0] for now

	// FLAG_DATA_HEIGHT/FLAG_VELOCITY_HEIGHT - requires heightvar or vxcolourvar
	// FLAG_VELOCITY_COLOUR/FLAG_SEGUE_COLOUR - requires Tcolourvar or vxcolourvar
	
	// Populate the INDEX 1 array not the INDEX 0 one which is for key button and that kind of thing.
	// Note that instead of Direct2D we can make segue bar by using the straight-to-screen coords(?)

	real shoelace;
	Vector2 u[3];

	VertexPNT3 * pPNT = vertices[1];
	DWORD * pIndex = indices[1];
	DWORD * pIndexTransparent = indices[2];

	Vertex * pVertex = &(X[0]);
	Triangle * pTri;
	Vector2 position;
	real data0,data1,data2, vz;

	real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*sin(PI/32.0);
	real XCENTRE1 = 0.0;//-XCENTRE2;
	real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*cos(PI/32.0);

	real x0,x1;
	Vector2 tempvec;
	Vertex tempvert;
	D3DXVECTOR3 vec1, vec2;
	D3DXVECTOR3 *pVecAnticlock, *pVecClock;
	Vertex *pVert1,*pVert2;
	D3DXVECTOR3 normalnext;
	int quad1,quad2;//quad3,quad4;
	real grad1,grad2;
	real doublevx, doublevy, doublevz; // avoid fp overflow so easily from float
	float v;
	real * ptr;
	int ii;
	float vx,vy,one,two;
	Vertex * pBegin;

	long numVerticesUse;
	long neigh_len;
	long izNeighs[128];


	if (boolDisplayInnerMesh) {
		pVertex = X; 
		pBegin = X;
		numVerticesUse = numVertices;
	} else {
		pVertex = Xdomain;
		pBegin = Xdomain;
		numVerticesUse = numDomainVertices;
	};


	// debug:
	printf("%f \n",pVertex->pos.modulus());
	

	for (long i = 0; i < numVerticesUse; ++i)
	{
		pPNT->pos.x = ((float)(pVertex->centroid.x))*xzscale;
		pPNT->pos.z = ((float)(pVertex->centroid.y))*xzscale;
		switch (heightflag)
		{
		case FLAG_FLAT_MESH:
			pPNT->pos.y = 0.0f;//(float)(pVertex->n*nfactor);
			
			// set normal: how exactly?

			// clearly it points up but since we have that funny "negate normal" code,
			// make it point down?:
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 1.0f;     // well, now we shall have a reason to do otherwise...
			pPNT->normal.z = 0.0f;
			break;
		case FLAG_DATA_HEIGHT:
			pPNT->pos.y = zeroplane + (float)(*(((real *)pVertex)+offset_data))*yscale;
			break;

		//case FLAG_HEIGHT_T:
		//	pPNT->pos.y = (float)(pVertex->T*Tfactor);
		//	break;

/*
			for (ii = 0; ii < pVertex->triangles.len; ii++)
			{
				pTri = (Triangle *)(pVertex->triangles.ptr[ii]);
				if (pTri->flags == 0)
				{
					if (pTri->cornerptr[0] == pVertex){
						pVert1 = pTri->cornerptr[1];
						pVert2 = pTri->cornerptr[2];
					} else {
						pVert1 = pTri->cornerptr[0];
						if (pTri->cornerptr[1] == pVertex) {
							pVert2 = pTri->cornerptr[2];
						} else {
							pVert2 = pTri->cornerptr[1];
						};
					};
					vec1.x = pVert1->x-pVertex->x;
					vec2.x = pVert2->x-pVertex->x;
					vec1.y = pVert1->y-pVertex->y;
					vec2.y = pVert2->y-pVertex->y;
					//D3DXVec3Cross(&(pPNT->normal),&vec1,&vec2);
					D3DXVec3Cross(&normalnext,&vec1,&vec2);
					pPNT->normal.x += normalnext.x;
					pPNT->normal.y += normalnext.y;
					pPNT->normal.z += normalnext.z;
				};
			};*/

			// That was not good enough because we need to account that
			// maybe corner sequence is different for different triangles.
			break;

		case FLAG_VELOCITY_HEIGHT:
			ptr = (real *)pVertex + offset_data;
			doublevx = (*ptr);
			++ptr;
			doublevy = (*ptr);
			pPNT->pos.y = zeroplane + (float)(sqrt(doublevx*doublevx+doublevy*doublevy))*yscale; // zeroplane should be == 0.0

		/*	pTri = (Triangle *)(pVertex->triangles.ptr[0]);
			if (pTri->flags > 0)
			{
				pTri = (Triangle *)(pVertex->triangles.ptr[1]);
				if (pTri->flags > 0)
					pTri = (Triangle *)(pVertex->triangles.ptr[2]);
			};
			
			if (pTri->cornerptr[0] == pVertex){
				pVert1 = pTri->cornerptr[1];
				pVert2 = pTri->cornerptr[2];
			} else {
				pVert1 = pTri->cornerptr[0];
				if (pTri->cornerptr[1] == pVertex) {
					pVert2 = pTri->cornerptr[2];
				} else {
					pVert2 = pTri->cornerptr[1];
				};
			};
		*/	
			break;
		case FLAG_VEC3_HEIGHT:
			ptr = (real *)pVertex + offset_data;
			doublevx = (*ptr);
			++ptr;
			doublevy = (*ptr);
			++ptr;
			doublevz = (*ptr);
			pPNT->pos.y = zeroplane + 
				(float)(sqrt(doublevx*doublevx+doublevy*doublevy+doublevz*doublevz))*yscale;
			break;
		};

		if (heightflag != FLAG_FLAT_MESH)
		{
			// Now we gotta set the normal.
			// Technically should take average of x-products
			//	Can we pass to shader?
			// For now, just take 1 triangle
			// Hopefully DX will make its own thread
			pPNT->normal.x = 0.0f;
			pPNT->normal.y = 0.0f;
			pPNT->normal.z = 0.0f;

			neigh_len = pVertex->GetNeighIndexArray(izNeighs);
			int iinext;
			for (ii = 0; ii < neigh_len; ii++)
			{
				iinext = ii+1;
				if (iinext == neigh_len) iinext = 0;

				pVert1 = X+izNeighs[ii];
				pVert2 = X+izNeighs[iinext];

				vec1.x = pVert1->centroid.x-pVertex->centroid.x;
				vec2.x = pVert2->centroid.x-pVertex->centroid.x;

				vec1.z = pVert1->centroid.y-pVertex->centroid.y;
				vec2.z = pVert2->centroid.y-pVertex->centroid.y;

				switch(heightflag)
				{
				case FLAG_DATA_HEIGHT:

					data1 = *(((real *)pVert1) + offset_data);
					data2 = *(((real *)pVert2) + offset_data);
					data0 = *(((real *)pVertex) + offset_data);

					vec1.y = zeroplane + (float)((data1-data0)*(real)yscale);
					vec2.y = zeroplane + (float)((data2-data0)*(real)yscale);  
					// does putting in brackets get rid of FP overflow? Adding (real)?

					break;
				case FLAG_VELOCITY_HEIGHT:
					ptr = (real *)pVert1 + offset_data;
					
					doublevx = (*ptr);
					++ptr;
					doublevy = (*ptr);
				
					v = (float)sqrt(doublevx*doublevx+doublevy*doublevy)*yscale;
					// avoids a crash to use double if v^2 > 1e38 

					if (!_finite(v)) {
						printf("problems vx vy \n"); 
						getch();
						 doublevx = doublevx;
					}

					one = zeroplane + v*yscale;
					
					ptr = (real *)pVert2 + offset_data;
					doublevx = (*ptr);
					++ptr;
					doublevy = (*ptr);
					two = zeroplane + (float)(sqrt(doublevx*doublevx+doublevy*doublevy))*yscale;

					// crashed in here; presumably v not finite

					vec1.y = (float)(one-pPNT->pos.y);
					vec2.y = (float)(two-pPNT->pos.y);
					break;

				case FLAG_VEC3_HEIGHT:
					
					ptr = (real *)pVert1 + offset_data;
					doublevx = (float)(*ptr);
					++ptr;
					doublevy = (float)(*ptr);
					++ptr;
					doublevz = (float)(*ptr);
					one = zeroplane + (float)(
						sqrt(doublevx*doublevx+doublevy*doublevy+doublevz*doublevz))
									*yscale;
					
					ptr = (real *)pVert2 + offset_data;
					doublevx = (float)(*ptr);
					++ptr;
					doublevy = (float)(*ptr);
					++ptr;
					doublevz = (float)(*ptr);
					two = zeroplane + (float)(
						sqrt(doublevx*doublevx+doublevy*doublevy+doublevz*doublevz))
									*yscale;
					
					vec1.y = (float)(one-pPNT->pos.y);
					vec2.y = (float)(two-pPNT->pos.y);
					
					break;
				};

				// decide quadrant and gradient for each direction

				if (vec1.x > 0.0){
					if (vec1.z > 0.0){
						quad1 = 0;
					} else {
						quad1 = 3;
					};
				} else {
					if (vec1.z > 0.0){
						quad1 = 1;
					} else {
						quad1 = 2;
					};
				};
				if (vec2.x > 0.0){						
					if (vec2.z > 0.0){
						quad2 = 0;
					} else {
						quad2 = 3;
					};
				} else {
					if (vec2.z > 0.0){
						quad2 = 1;
					} else {
						quad2 = 2;
					};
				};

				// cases:
				// quadrant is +1 or -3 => Anticlockwise
				// quadrant is same => check gradients
				// quadrant is opposite => check against negated gradient

				if ((quad2 == quad1+1) || (quad2 == quad1-3))
				{
					pVecAnticlock = &vec2;
					pVecClock = &vec1;
				} else {
					if ((quad1 == quad2+1) || (quad1 == quad2-3))
					{
						pVecAnticlock = &vec1;
						pVecClock = &vec2;
					} else {
						
						grad1 = vec1.z/vec1.x;
						grad2 = vec2.z/vec2.x;

						// CHECK:
						if (quad1 == quad2) {
							
							// Higher gradient is then always Anticlockwise

							if (grad1 > grad2) {
								pVecAnticlock = &vec1;
								pVecClock = &vec2;
							} else {
								pVecAnticlock = &vec2;
								pVecClock = &vec1;
							};
							
						} else {
							// opposite quadrants:
							// Lower gradient grad2 means Anticlockwise 2
							// regardless of which quadrants

							if (grad2 < grad1) {
								pVecAnticlock = &vec2;
								pVecClock = &vec1;
							} else {
								pVecAnticlock = &vec1;
								pVecClock = &vec2;
							};
						};
					};
				};
				
				//D3DXVec3Cross(&(pPNT->normal),&vec1,&vec2);
				D3DXVec3Cross(&normalnext,pVecAnticlock,pVecClock);
				pPNT->normal.x += normalnext.x;
				pPNT->normal.y += normalnext.y;
				pPNT->normal.z += normalnext.z;
			};

		};

		SetVertexColour(pPNT,pVertex,colourflag,offset_data,offset_vcolour);
	
		++pPNT;
		++pVertex;
	};
	
	long totalVertices = numVerticesUse;
	
	if (totalVertices > numVerticesMax[1])
	{
		printf("problems! vertex array exceeded"); getch();
	}; // surely > not >=
	
	long triCount = 0;
	long triTransparentCount = 0;
	Vector2 temp;
	long i = 0;
	
	pTri = T;
	for (i = 0; i < numTriangles; ++i) 
	{		
		if ((boolDisplayInnerMesh) || (pTri->u8domain_flag == DOMAIN_TRIANGLE))
		{
			pTri->PopulatePositions(u[0],u[1],u[2]);

			if (	(GlobalCutaway == false) || 
					(	(u[0].x/u[0].y > -GRADIENT_X_PER_Y*0.5)
					&&	(u[1].x/u[1].y > -GRADIENT_X_PER_Y*0.5)
					&&	(u[2].x/u[2].y > -GRADIENT_X_PER_Y*0.5)
					)
					)
			{
				if (pTri->periodic == 0)
				{
					// ensure that we enter clockwise coordinates:

					shoelace = u[0].x*u[1].y - u[1].x*u[0].y + u[1].x*u[2].y - u[2].x*u[1].y
						+ u[2].x*u[0].y - u[0].x*u[2].y;
					if (shoelace < 0.0) {

						*pIndex = (DWORD)(pTri->cornerptr[0]-pBegin);

						++pIndex;
						*pIndex = (DWORD)(pTri->cornerptr[1]-pBegin);
						
						++pIndex;
						*pIndex = (DWORD)(pTri->cornerptr[2]-pBegin);
						
						++pIndex;

						//fprintf(debugfile,"%d %d %d  %f %f  %f %f  %f %f \n",pTri->cornerptr[0]-X,pTri->cornerptr[1]-X,pTri->cornerptr[2]-X,
						//	pTri->cornerptr[0]->x,pTri->cornerptr[0]->y,
						//	pTri->cornerptr[1]->x,pTri->cornerptr[1]->y,
						//	pTri->cornerptr[2]->x,pTri->cornerptr[2]->y);

						//
						
						// This alone, with CULL_CCW, produces alternate rows of triangles.
						// The rows face with points inwards.

						// Of course we do not know the structure of our vertex list.

					} else {
						// swap 1 and 0 to produce clockwise.
						*pIndex = (DWORD)(pTri->cornerptr[1]-pBegin);
						++pIndex;
						*pIndex = (DWORD)(pTri->cornerptr[0]-pBegin);						
						++pIndex;
						*pIndex = (DWORD)(pTri->cornerptr[2]-pBegin);						
						++pIndex;
					};

				} else {
					// periodic:

					// Add additional mapped vertices.

					for (int j = 0; j < 3; j++)
					{
						if (pTri->cornerptr[j]->pos.x < 0.0)
						{
							*pIndex = (DWORD)(pTri->cornerptr[j]-pBegin);
							++pIndex;
						} else {
							temp = Anticlockwise*pTri->cornerptr[j]->centroid;
							
							pPNT->pos.x = ((float)(temp.x))*xzscale;
							pPNT->pos.z = ((float)(temp.y))*xzscale;
							//pPNT->pos.y = 0.0f;
							pVertex = pTri->cornerptr[j];
							switch (heightflag)
							{
							case FLAG_DATA_HEIGHT:
								pPNT->pos.y = zeroplane + (float)(*(((real *)pVertex)+offset_data))*yscale;
								break;
							case FLAG_VELOCITY_HEIGHT:
								ptr = (real *)pVertex + offset_data;
								vx = (float)(*ptr);
								++ptr;
								vy = (float)(*ptr);
								pPNT->pos.y = zeroplane + sqrt(vx*vx+vy*vy)*yscale;
								break;
							case FLAG_VEC3_HEIGHT:
								ptr = (real *)pVertex + offset_data;
								vx = (float)(*ptr);
								++ptr;
								vy = (float)(*ptr);
								++ptr;
								vz = (float)(*ptr);
								pPNT->pos.y = zeroplane + sqrt(vx*vx+vy*vy+vz*vz)*yscale;
								break;
							default:
								pPNT->pos.y = 0.0f;					
								break;
							};
							pPNT->normal.x = 0.0f;
							pPNT->normal.y = 1.0f;     // well, now we shall have a reason to do otherwise...
							pPNT->normal.z = 0.0f;
							pPNT->tex0.x = 0.0f;
							pPNT->tex0.y = 0.0f;
							++pPNT;
							*pIndex = totalVertices;
							++pIndex;
							++totalVertices;
							if (totalVertices > numVerticesMax[1])
							{
								printf("problems. vertex array exceeded"); getch();
							};
						};
					};
				};

				++triCount;

			} else {

				if (pTri->periodic == 0) // otherwise don't bother
				{
					// ensure that we enter clockwise coordinates:

					shoelace = u[0].x*u[1].y - u[1].x*u[0].y + u[1].x*u[2].y - u[2].x*u[1].y
						+ u[2].x*u[0].y - u[0].x*u[2].y;
					if (shoelace < 0.0) {

						*pIndexTransparent = (DWORD)(pTri->cornerptr[0]-pBegin);
						++pIndexTransparent;
						*pIndexTransparent = (DWORD)(pTri->cornerptr[1]-pBegin);					
						++pIndexTransparent;
						*pIndexTransparent = (DWORD)(pTri->cornerptr[2]-pBegin);						
						++pIndexTransparent;
						// This alone, with CULL_CCW, produces alternate rows of triangles.
						// The rows face with points inwards.

						// Of course we do not know the structure of our vertex list.

					} else {
						// swap 1 and 0 to produce clockwise.
						*pIndexTransparent = (DWORD)(pTri->cornerptr[1]-pBegin);
						++pIndexTransparent;
						*pIndexTransparent = (DWORD)(pTri->cornerptr[0]-pBegin);				
						++pIndexTransparent;
						*pIndexTransparent = (DWORD)(pTri->cornerptr[2]-pBegin);						
						++pIndexTransparent;
					};
					++triTransparentCount;
				};
			};
		};



		++pTri;
		
	};


	numVerticesUsed[1] = totalVertices;
	numTrianglesUsed[1] = triCount;
	numTrianglesUsed[2] = triTransparentCount;

//	fprintf(debugfile,"%d %d \n",numVerticesUsed[1],numTrianglesUsed[1]);
//	fclose(debugfile);

}

void QuickSort (long VertexIndexArray[], real radiusArray[],
		long lo, long hi)
{
	real temp;
	long tempint,p;

	if (lo < hi) {
		// Pick a pivot element
		
		long q = (lo+hi)/2;
		// move it to a position p where it has partitioned the sublist.
		
		real pivotValue = radiusArray[q];

		// swap with element hi:
		temp = radiusArray[hi];
		radiusArray[hi] = pivotValue;
		radiusArray[q] = temp;
		tempint = VertexIndexArray[hi];
		VertexIndexArray[hi] = VertexIndexArray[q];
		VertexIndexArray[q] = tempint;

		int storeIndex = lo;
		for (int i = lo; i < hi; i++)
		{
			if (radiusArray[i] < pivotValue)
			{
				// swap i and storeIndex
				
				temp = radiusArray[i];
				radiusArray[i] = radiusArray[storeIndex];
				radiusArray[storeIndex] = temp;
				tempint = VertexIndexArray[i];
				VertexIndexArray[i] = VertexIndexArray[storeIndex];
				VertexIndexArray[storeIndex] = tempint;
				
				storeIndex++;
			};
		};
		// swap storeIndex and hi
		temp = radiusArray[hi];
		radiusArray[hi] = radiusArray[ storeIndex ];
		radiusArray[ storeIndex ] = temp;
		tempint = VertexIndexArray[hi];
		VertexIndexArray[hi] = VertexIndexArray[storeIndex];
		VertexIndexArray[storeIndex] = tempint;
				
		p = storeIndex;
			
		
		QuickSort (VertexIndexArray, radiusArray,
					lo, p-1);
		QuickSort (VertexIndexArray, radiusArray,
					p+1, hi);
	};
}


//


long TriMesh::GetVertsRightOfCutawayLine_Sorted(long * VertexIndexArray,
										real * radiusArray) const
{
	Vertex * pNeigh;
	Vertex * pVertex;
	long iVertex;
	long iCaret = 0;
	long neigh_len;
	long izNeighs[128];

	pVertex = X;

	if (radiusArray == NULL) {
		printf("ADJSA???");
		getch();
	}

	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// We want this one if and only if, it is to the right of the 
		// cutaway line but has neighbours to the left of it.
		
		if ((pVertex->pos.x/pVertex->pos.y > -0.5*GRADIENT_X_PER_Y)
			&& (pVertex->pos.x < 0.0)) {

			neigh_len = pVertex->GetNeighIndexArray(izNeighs);
			for (int i = 0; i < neigh_len; i++)
			{
				pNeigh = X + izNeighs[i];
				if (pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y) {
					// count this vertex
					VertexIndexArray[iCaret] = iVertex;
					radiusArray[iCaret] = pVertex->pos.modulus();
					iCaret++;
					i = 100000; // skip out
				};
			};
		};

		++pVertex;
	};

	// Now sort them!
	// No way round this.


	QuickSort (VertexIndexArray, radiusArray,
		0,iCaret-1); // lowest and highest elements to be sorted
			
	return iCaret;
}




real TriMesh::ReturnMaximumData(int offset)
{
	Vertex *pVert;
	pVert = X;
	real maxA2z = 0.0;
	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		if (*((real *)pVert + offset) > maxA2z) maxA2z = *((real *)pVert + offset);		
		++pVert;
	};
	if (maxA2z == 0.0) return 1.0; // do not return 0
	return maxA2z;
}

void TriMesh::ReturnL5Data(int offset, real * pmax, real * pmin, bool bDisplayInner) const
{
	Vertex *pVert;
	long numplus = 0, numminus = 0;
	// if there are no negatives, min on graph is zero anyway.
	real value, valuesq, value5;
	long numUse;

	// Idea:
	// 1. Discount observations within (neg mean/2, + mean/2)
	// 2. Apply L5 to what remains.

	if (bDisplayInner) {
		numUse = numVertices;
		pVert = X;
	} else {
		numUse = numDomainVertices;
		pVert = Xdomain;
	};

	// get +,- mean:

	real negmean = 0.0, posmean = 0.0;
	long negnum = 0, posnum = 0;
	for (long iVert = 0; iVert < numUse; iVert++)
	{
		value = *((real *)pVert + offset);
	
		if (value < 0.0) {
			negmean+=value;
			negnum++;
		}
		if (value > 0.0) {
			posmean+=value;
			posnum++;
		};
		++pVert;
	};
	if (negnum > 0)
		negmean /= (real)negnum;
	if (posnum > 0)
		posmean /= (real)posnum;
	
	// create cutoffs:
	negmean *= 0.6;
	posmean *= 0.6;

	// If cutoffs result in few observations selected (< 10?)
	// then go again with less cutoff???
	
	real S5_plus = 0.0, S5_minus = 0.0; 
	if (bDisplayInner) {
		pVert = X;
	} else {
		pVert = Xdomain;
	};
	for (long iVert = 0; iVert < numUse; iVert++)
	{
		value = *((real *)pVert + offset);
		valuesq = value*value;
		value5 = valuesq*valuesq*value;

		if (value > posmean){
			S5_plus += value5;
			numplus++;
		};
		if (value < negmean){
			S5_minus += value5;
			numminus++;
		};

		++pVert;
	};

	printf("numplus %d ",numplus);

	if (numplus > 10)
	{
		*pmax = pow(S5_plus/(real)numplus,1.0/5.0);
	} else {
		if (numplus > 0) {
			// this way failed: only a few above cutoff..
			// Try again with L7, values above posmean excluded.
			
			real S7_plus = 0.0; 
			numplus = 0;
			if (bDisplayInner) {
				pVert = X;
			} else {
				pVert = Xdomain;
			};
			for (long iVert = 0; iVert < numUse; iVert++)
			{
				value = *((real *)pVert + offset);
				valuesq = value*value;
				value5 = valuesq*valuesq*valuesq*value;
				
				if ((value < posmean) && (value > 0.0)){
					S7_plus += value5;
					numplus++;
				};
				++pVert;
			};
			printf("| numplus %d S7 %1.5E",numplus,S7_plus);
			if (numplus == 0) {*pmax = 0.0;} else {
				*pmax = pow(S7_plus/(real)numplus,1.0/7.0);
			};
		} else {
			*pmax = 0.0;
		};
	};

	if (numminus > 10)
	{
		*pmin = -pow(fabs(S5_minus)/(real)numminus,1.0/5.0);
	} else {
		if (numminus > 0) {

			real S7_minus = 0.0; 
			numminus = 0;
			if (bDisplayInner) {
				pVert = X;
			} else {
				pVert = Xdomain;
			};
			for (long iVert = 0; iVert < numUse; iVert++)
			{
				value = *((real *)pVert + offset);
				valuesq = value*value;
				value5 = valuesq*valuesq*valuesq*value;
				
				if ((value > negmean) && (value < 0.0)){
					S7_minus += value5;
					numminus++;
				};
				++pVert;
			};
			// Beware: numminus can still be 0
			if (numminus == 0) {*pmin = 0.0;} else {
				*pmin = -pow(-S7_minus/(real)numminus,1.0/7.0);
			};
		} else {
			*pmin = 0.0;
		};
	};
		
	printf("L5+ %1.8E L5- %1.8E\n",*pmax,*pmin);

	// Aesthetics:
	*pmax *= 1.11;
	*pmin *= 1.11;

//	printf("value at 11830: %1.8E \n",*((real *)(X+11830)+offset));
}

void TriMesh::ReturnMaxMinData(int offset, real * pmax, real * pmin, bool bDisplayInner) const
{
	Vertex *pVert;
	real maxA2z = -1.0e100;
	real minA2z = 1.0e100;

	long iVertMax = 0;
	long iVertMin = 0;

	if (bDisplayInner) {

		pVert = X;
		for (long iVert = 0; iVert < numVertices; iVert++)
		{
			if (*((real *)pVert + offset) > maxA2z) 
			{
				maxA2z = *((real *)pVert + offset);		
				iVertMax = iVert;
			};
			if (*((real *)pVert + offset) < minA2z) 
			{
				minA2z = *((real *)pVert + offset);		
				iVertMin = iVert;
			};
			++pVert;
		};
	} else {
		
		// debug:
		printf("Xdomain-X %d offset %d \n",
			(long)(Xdomain-X),offset);

		pVert = Xdomain;
		for (long iVert = 0; iVert < numDomainVertices; iVert++)
		{
	//		if (offset == 71) printf("iVert %d maxA2z %1.14E value %1.14E ",iVert+(Xdomain-X),maxA2z,*((real *)pVert + offset) );
			// "Floating point invalid operation" offset 71 code 25
			if (*((real *)pVert + offset) > maxA2z) 
			{
				maxA2z = *((real *)pVert + offset);		
				iVertMax = iVert+(Xdomain-X);
			};
			if (*((real *)pVert + offset) < minA2z) 
			{
				minA2z = *((real *)pVert + offset);		
				iVertMin = iVert+(Xdomain-X);
			};
			++pVert;
	//		if (offset == 71) printf(".\n");
		};
	};
	*pmax = maxA2z;
	*pmin = minA2z;
	printf("Max %1.8E found at vertex %d \n",*pmax,iVertMax);
	printf("Min %1.8E found at vertex %d \n",*pmin,iVertMin);
//	printf("value at 11830: %1.8E \n",*((real *)(X+11830)+offset));
}

void TriMesh::ReturnMaxMinDataAux(int iLevel, int offset, real * pmax, real * pmin)
{
	Vertex *pVert;
	pVert = AuxX[iLevel];
	real maxA2z = -1.0e100;
	real minA2z = 1.0e100;
	for (long iVert = 0; iVert < numAuxVertices[iLevel]; iVert++)
	{
		if (*((real *)pVert + offset) > maxA2z) maxA2z = *((real *)pVert + offset);		if (*((real *)pVert + offset) < minA2z) minA2z = *((real *)pVert + offset);		
		++pVert;
	};
	if (maxA2z == 0.0) maxA2z = 1.0; // do not return 0
	
	*pmax = maxA2z;
	*pmin = minA2z;
}
// Think carefully on this.
/*
void TriMesh::ReturnInnerMaxMinData(int offset, real * pmax, real * pmin)
{
	AuxVertex *pVert;
	pVert = InnerX;
	real maxA2z = -1.0e100;
	real minA2z = 1.0e100;
	for (long iVert = 0; iVert < numInnerVertices; iVert++)
	{
		if (*((real *)pVert + offset) > maxA2z) maxA2z = *((real *)pVert + offset);		
		if (*((real *)pVert + offset) < minA2z) minA2z = *((real *)pVert + offset);		
		++pVert;
	};
	*pmax = maxA2z;
	*pmin = minA2z;
}*/

real TriMesh::ReturnMaximumDataAux(int iLevel, int offset)
{
	Vertex *pVert;
	pVert = AuxX[iLevel];
	real maxA2z = 0.0;
	for (long iVert = 0; iVert < numAuxVertices[iLevel]; iVert++)
	{
		if (*((real *)pVert + offset) > maxA2z) maxA2z = *((real *)pVert + offset);		
		++pVert;
	};
	if (maxA2z == 0.0) return 1.0; // do not return 0
	return maxA2z;
}

real TriMesh::ReturnMaximumVelocityAux(int iLevel, int offset)
{
	Vertex *pVert;
	real * ptr;
	real maxvsq, vx, vy, vsq;
	pVert = AuxX[iLevel];
	maxvsq = 0.0;
	for (long iVert = 0; iVert < numAuxVertices[iLevel]; iVert++)
	{
		ptr = ((real *)pVert) + offset;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		vsq = vx*vx+vy*vy;
		if (vsq > maxvsq) maxvsq = vsq;		
		++pVert;
	};

	if (maxvsq == 0.0) return 1.0; // do not return 0
	return sqrt(maxvsq);
}
real TriMesh::ReturnL4_Velocity(int offset_v, bool bDisplayInner) const
{
	real * ptr;
	Vertex *pVert;
	real vx,vy;
	pVert = X;
	real vsq;
	real L2sum = 0.0;
	long num = 0;
	long nmax = numVertices;
	if (bDisplayInner == 0) {
		pVert = Xdomain;
		nmax = numDomainVertices;
	}

	// a. Take L2 and create cutoff for counting data
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;
		
		vsq = vx*vx+vy*vy;
		L2sum += vsq;
		if (vsq > 0.0) num++;
		
		++pVert;
	};
	if (num == 0) return 0.0;
	real L2sq = L2sum/(real)num;
	L2sq *= 0.25; // create cutoff

	// b. Take L6 to get _near_ the maximum.

	real L6sum = 0.0;
	num = 0;
	pVert = X;
	if (bDisplayInner == 0) {
		pVert = Xdomain;
	}
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;		
		vsq = vx*vx+vy*vy;
		if (vsq > L2sq) {
			L6sum += vsq*vsq*vsq;
			num++;
		};
		++pVert;
	};
	if (L6sum == 0.0) return 1.0; // do not return 0

	return 1.1*pow(L6sum/(real)num,1.0/6.0);
}
real TriMesh::ReturnMaximumVelocity(int offset_v, bool bDisplayInner) const
{
	real * ptr;
	Vertex *pVert;
	real vx,vy;
	pVert = X;
	real vsq;
	real maxvsq = 0.0;

	long nmax = numVertices;
	if (bDisplayInner == 0) {
		pVert = Xdomain;
		nmax = numDomainVertices;
	}
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;
		
		vsq = vx*vx+vy*vy;
		if (vsq > maxvsq) maxvsq = vsq;

		++pVert;
	};

	if (maxvsq == 0.0) return 1.0; // do not return 0

	return sqrt(maxvsq);
}
real TriMesh::ReturnMaximum3DMagnitude(int offset_v,bool bDisplayInner) const
{
	real * ptr;
	Vertex *pVert;
	real vx,vy,vz;
	pVert = X;
	real maxvsq = 0.0;
	real vsq;
	long nmax = numVertices;
	if (bDisplayInner == 0) {
		pVert = Xdomain;
		nmax = numDomainVertices;
	}
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;
		vz = *ptr;
		
		vsq = vx*vx+vy*vy+vz*vz;
		if (vsq > maxvsq) maxvsq = vsq;

		++pVert;
	};

	if (maxvsq == 0.0) return 1.0; // do not return 0

	return sqrt(maxvsq);
}
real TriMesh::ReturnL4_3DMagnitude(int offset_v,bool bDisplayInner) const
{
	real * ptr;
	Vertex *pVert;
	real vx,vy,vz;
	pVert = X;
	real vsq;
	real L2sum = 0.0;
	long nmax = numVertices;
	long num = 0;

	if (bDisplayInner == 0) {
		pVert = Xdomain;
		nmax = numDomainVertices;
	}
	
	// a. Take L2 and create cutoff for counting data
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;
		vz = *ptr;
		
		vsq = vx*vx+vy*vy+vz*vz;
		L2sum += vsq;
		if (vsq > 0.0) num++;
		
		++pVert;
	};	
	if (num == 0) return 0.0;
	real L2sq = L2sum/(real)num;

	L2sq *= 0.25; // create cutoff

	pVert = X;
	if (bDisplayInner == 0) {
		pVert = Xdomain;
	}
	real L6sum = 0.0;
	num = 0;
	for (long iVert = 0; iVert < nmax; iVert++)
	{
		ptr = ((real *)pVert) + offset_v;
		vx = *ptr;
		++ptr;
		vy = *ptr;
		++ptr;
		vz = *ptr;		
		vsq = vx*vx+vy*vy+vz*vz;
		if (vsq > L2sq){
			L6sum += vsq*vsq*vsq;
			num++;
		};
		++pVert;
	};
	if (L6sum == 0.0) return 1.0; // do not return 0

	return 1.1*pow(L6sum/(real)num,1.0/6.0);
}
void TriMesh::ResetTriangleNeighbours(Triangle * pTri)
{
	// Same caveat as before: we have not updated what vertex is at the outer edge...
	// Probably need whole different code for disconnecting outer vertex - is that enough? maybe not



	// This extra bit was not necessary: in default of finding answer, 
	// RPTOST does return the tri itself, that it was given.


	//// Should work fine in case of ins base triangles however.
	//if (pTri->u8EdgeFlag > 0) {
	//	if (pTri->cornerptr[2]->flags >= 4) {
	//		pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri);
	//	} else {
	//		pTri->neighbours[2] = pTri;
	//	};
	//	if (pTri->cornerptr[0]->flags >= 4) {
	//		pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri);
	//	} else {
	//		pTri->neighbours[0] = pTri;
	//	};
	//	if (pTri->cornerptr[1]->flags >= 4) {
	//		pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri);	
	//	} else {
	//		pTri->neighbours[1] = pTri;
	//	};
	//	return;
	//};

	pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri);
	pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri);
	pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri);
}
			
/*void TriMesh::SetEz()
{
	Triangle * pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
	//	pTri->A.z = 1000000.0;
		++pTri;
	}
	Vertex * pVertex = X;
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{
	//	pVertex->A.z = 1000000.0;
		++pVertex;
	};
}
*/
int Vertex::Save(FILE * fp)
{
	// For each vertex: 
	// save position, triangles array, vector A, flags

	long iTri;

	if (fwrite(&pos,sizeof(Vector2),1,fp) != 1) return 1;	
	if (fwrite(&centroid,sizeof(Vector2),1,fp) != 1) return 1;	
	if (fwrite(&Neut,sizeof(macroscopic),1,fp) != 1) return 100;
	if (fwrite(&Ion,sizeof(macroscopic),1,fp) != 1) return 100;
	if (fwrite(&Elec,sizeof(macroscopic),1,fp) != 1) return 100;
	if (fwrite(&phi,sizeof(real),1,fp) != 1) return 110;
	if (fwrite(&A,sizeof(Vector3),1,fp) != 1) return 111;
	if (fwrite(&B,sizeof(Vector3),1,fp) != 1) return 112;
	if (fwrite(&E,sizeof(Vector3),1,fp) != 1) return 113;
	if (fwrite(&Adot,sizeof(Vector3),1,fp) != 1) return 114;

	if (fwrite(&(tri_len),sizeof(long),1,fp) != 1) return 101;
	if (fwrite(&(neigh_len),sizeof(long),1,fp) != 1) return 102;

	if (fwrite(izTri,sizeof(long),tri_len,fp) != tri_len) return 1000;
	if (fwrite(izNeigh,sizeof(long),neigh_len,fp) != neigh_len) return 1000;
		
	if (fwrite(&flags,sizeof(long),1,fp) != 1) return 1; 
	if (fwrite(&has_periodic,sizeof(bool),1,fp) != 1) return 1; 

	return 0;
}
int Vertex::Load(FILE * fp)
	{
		if (fread(&pos,sizeof(Vector2),1,fp) != 1) return 1;
		if (fread(&centroid,sizeof(Vector2),1,fp) != 1) return 1;
		if (fread(&Neut,sizeof(macroscopic),1,fp) != 1) return 1;
		if (fread(&Ion,sizeof(macroscopic),1,fp) != 1) return 1;
		if (fread(&Elec,sizeof(macroscopic),1,fp) != 1) return 1;
		if (fread(&phi,sizeof(real),1,fp) != 1) return 1;
		if (fread(&A,sizeof(Vector3),1,fp) != 1) return 1;
		if (fread(&B,sizeof(Vector3),1,fp) != 1) return 1;
		if (fread(&E,sizeof(Vector3),1,fp) != 1) return 1;
		if (fread(&Adot,sizeof(Vector3),1,fp) != 1) return 1;

		if (fread(&(tri_len),sizeof(long),1,fp) != 1) return 101;
		if (fread(&(neigh_len),sizeof(long),1,fp) != 1) return 102;

		if (fread(izTri,sizeof(long),tri_len,fp) != tri_len) return 1000;
		if (fread(izNeigh,sizeof(long),neigh_len,fp) != neigh_len) return 1000;
		
		if (fread(&flags,sizeof(long),1,fp) != 1) return 1;
		if (fread(&has_periodic,sizeof(bool),1,fp) != 1) return 1;

		return 0;
	}

	int Triangle::Save(FILE * fp,Vertex * pVertArray, Triangle *pTriArray)
	{
		
	// For each triangle:
	// all data; indices in place of cornerptr and neighbour pointers; 
	// periodic, flags, contribflags?
		long iVert1,iVert2,iVert3,iNeigh0,iNeigh1,iNeigh2;
		
		iVert1 = cornerptr[0]-pVertArray;
		iVert2 = cornerptr[1]-pVertArray;
		iVert3 = cornerptr[2]-pVertArray;

		if (fwrite(&iVert1,sizeof(long),1,fp) != 1) return 1;
		if (fwrite(&iVert2,sizeof(long),1,fp) != 1) return 1;
		if (fwrite(&iVert3,sizeof(long),1,fp) != 1) return 1;

		iNeigh0 = neighbours[0] - pTriArray;
		iNeigh1 = neighbours[1] - pTriArray;
		iNeigh2 = neighbours[2] - pTriArray;
		if (fwrite(&iNeigh0,sizeof(long),1,fp) != 1) return 1;
		if (fwrite(&iNeigh1,sizeof(long),1,fp) != 1) return 1;
		if (fwrite(&iNeigh2,sizeof(long),1,fp) != 1) return 1;

		if (fwrite(&periodic,sizeof(int),1,fp) != 1) return 1;
		if (fwrite(&u8domain_flag,sizeof(BYTE),1,fp) != 1) return 1;

		return 0;
	}

	int Triangle::Load(FILE * fp, Vertex * pVertArray, Triangle * pTriArray)
	{
		long iVert1,iVert2,iVert3,iNeigh0,iNeigh1,iNeigh2;

		if (fread(&iVert1,sizeof(long),1,fp) != 1) return 1;
		if (fread(&iVert2,sizeof(long),1,fp) != 1) return 1;
		if (fread(&iVert3,sizeof(long),1,fp) != 1) return 1;

		cornerptr[0] = pVertArray + iVert1;
		cornerptr[1] = pVertArray + iVert2;
		cornerptr[2] = pVertArray + iVert3;
		
		if (fread(&iNeigh0,sizeof(long),1,fp) != 1) return 1;
		if (fread(&iNeigh1,sizeof(long),1,fp) != 1) return 1;
		if (fread(&iNeigh2,sizeof(long),1,fp) != 1) return 1;
		
		neighbours[0] = pTriArray + iNeigh0;
		neighbours[1] = pTriArray + iNeigh1;
		neighbours[2] = pTriArray + iNeigh2;

		if (fread(&periodic,sizeof(int),1,fp) != 1) return 1;	
		if (fread(&u8domain_flag,sizeof(BYTE),1,fp) != 1) return 1;
	
		return 0;
	}
int TriMesh::Save(char * filename)
{
	printf("\nsaving..\n");

	FILE * fp = fopen(filename,"wb");
	if (fp == NULL) { 
		printf("\nfile error %s\n",filename);
		return 1000001;
	};

	long version = 700;
	fwrite(&version,sizeof(long),1,fp);

	fwrite(&GlobalStepsCounter,sizeof(long),1,fp);
	fwrite(&evaltime,sizeof(real),1,fp);

	fwrite(&numVertices,sizeof(long),1,fp);
	fwrite(&numDomainVertices,sizeof(long),1,fp);
	fwrite(&numTriangles,sizeof(long),1,fp);
	fwrite(&numRows,sizeof(long),1,fp); // don't ask me why
	// also need to know that we have the correct mapping to coarse defined so save size of next level:
	// fwrite(&numAuxVertices[0],sizeof(long),1,fp);
	
	// if those details will all match, hopefully the rest are guessed correctly from initialisation.
	
	Vertex * pVert = X;
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{
		int retcode = pVert->Save(fp);
		if (retcode != 0) {
			printf("vertex save fail!\niVertex %d code %d\n",iVertex,retcode);
			getch();
			return 102102;
		};
		++pVert;
	};
	// No more inner vertices, the transition makes ODE solver too complex.

	Triangle * pTri = T;	
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		int retcode = pTri->Save(fp,X,T);
		if (retcode != 0) {
			printf("Triangle save fail!\niTri %d : %d\n",iTri,retcode);
			getch();
			return 102103;
		};	
		++pTri;
	};
	fclose(fp);
	printf("\nSystem saved successfully.\n");
	return 0;
}


int TriMesh::Load(char * filename)
{
	long read, file_version, file_numVertices, file_numTriangles, file_numRows,
		file_numDomainVertices;
	printf("\nloading..\n");
	FILE * fp = fopen(filename,"rb");
	if (fp == NULL) { printf("\nfile open error\n"); return 1000001;};

	read = fread(&file_version,sizeof(long),1,fp);
	if (read != 1) {printf("3rror\nread = %d",read); return 1;};

	if (file_version != 700) {
		printf("\nfile version = %d not 700; exiting\n",file_version);
		return -1200;
	};

	read = fread(&GlobalStepsCounter,sizeof(long),1,fp);
	if (read != 1) {printf("3rror 0\n"); return 1;};
	read = fread(&evaltime,sizeof(real),1,fp);
	if (read != 1) {printf("3rror 1\n"); return 11;};

	read = fread(&file_numVertices,sizeof(long),1,fp);
	if (read != 1) {printf("3rror 2\n"); return 12;};
	read = fread(&file_numDomainVertices,sizeof(long),1,fp);
	if (read != 1) {printf("3rror 2b\n"); return 13;};
	read = fread(&file_numTriangles,sizeof(long),1,fp);
	if (read != 1) {printf("3rror 3\n"); return 14;};
	read = fread(&file_numRows,sizeof(long),1,fp);
	if (read != 1) {printf("3rror 4\n"); return 15;};
//	read = fread(&file_numAuxVertices0,sizeof(long),1,fp);
//	if (read != 1) {printf("3rror 5\n"); return 1;};

	if ((numVertices != file_numVertices) || 
		(numDomainVertices != file_numDomainVertices) ||
		(numTriangles != file_numTriangles)	)
	{
		// refuse to load if # of vertices not same or inner mesh different.
		printf("Load failed - \n %d %d %d %d %d %d %d \n",
			numVertices,numDomainVertices,numTriangles,numAuxVertices[0],
			file_numVertices,file_numDomainVertices,file_numTriangles);
		// The reason we should test aux vertex number is because
		// we should be redefining coarse mapping if the number
		// is different!

		return 2000;
	}
	if (numTrianglesAllocated < file_numTriangles)
	{
		printf("\nLoad failed because numTrianglesAllocated < file_numTriangles\n\n");
		return 1000000;
		// Note that the number of triangles should be constant during the simulation.
	};
	
	// For each vertex: 
	// position, triangles array, vector A, flags

	Vertex * pVert = X;
	for (int iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVert->Load(fp)) {
			printf("vertex load fail!\n%d out of %d \n",iVertex,numVertices);
			getch();
			return 102101;
		};
		++pVert;
	};	
	// For each triangle:
	// all data; indices in place of cornerptr and neighbour pointers; 
	// periodic, flags, contribflags?

	Triangle * pTri = T;	
	for (int iTri = 0; iTri < numTriangles; iTri++)
	{
		if (pTri->Load(fp,X,T)) {
			printf("Triangle load fail!\n%d\n",iTri);
			getch();
			return 102104;
		};	
		++pTri;
	};	
	fclose(fp);

	// Now leap into setting up recalc'd data:

	real rdum;
	pTri = T;
	for (int iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->area = pTri->GetArea();
		pTri->RecalculateEdgeNormalVectors(false); 
		pTri->RecalculateCentroid();
		++pTri;
	};
	RefreshVertexNeighboursOfVerticesOrdered(); // does use tri centroids

	Recalculate_TriCentroids_VertexCellAreas_And_Centroids();

	GetBFromA(); 

	printf("\nSystem loaded successfully.\n");	
	return 0;
}

int TriMesh::SaveText(char * filename)
{
	FILE * fp = fopen(filename,"w");
	printf("\n\nfunctionality doesn't exist yet\n");
	fclose(fp);
	return 1;
}
void TriMesh::CreateTilingAndResequence(TriMesh * pDestMesh) {
	// First set Vertex::iVolley to which tile.
	// Then set Triangle::indicator to which tile. Rule that
	// if triangle has 2 corners within same tile, it belongs to the corresponding tri tile.
	// Each tile should have the right number of elements.
	// This then allows us to create a resequence, which we can put in
	
	// We then want to do what? Create another TriMesh object into which we pour the
	// resequenced data, updating all neighbour index etc etc.
	// ---------
	
	// 1. Create vertex sequence.
	
	// When a point is chosen for this block, we identify the point with most
	// connections to the block and include that next.
	// And we make this block contiguous to previous ...
	// To create bottom row simultaneously will prevent just spreading out horizontally.
	
	// Let's suppose we start by creating all these horizontal blocks, knowing how wide we
	// want them to be.
	// Then, suppose we follow on to each of these? Might not always be able to.
	// Also a block *might* get trapped in --- in that case the next point has to find where to go.

	// We probably MIGHT like the rows of blocks not to end up offset. Or worse, with some blocks squatter than others.
	// Well, it doesn't really matter if it's offset, so we could actually go with that.
		
	// ie look to place first point of run where multiple previous blocks meet.
	
	// *. Make next point of tile, where there are the greatest number of connections.
	// Given same # of connections to more points, use earliest point of tile.
	// Or -- using earliest point, choose those with greatest number of connections -- to this tile or to all?

	//int spacing = (int)(sqrt((real)threadsPerTileMajor));
	// If anything, LESS wide than it wants to be.
	//int numTilesOnGo = this->numInnermostRow/spacing;


	int numTilesOnGo = 12;
	int spacing = numInnermostRow/numTilesOnGo;

	printf("numInnermostRow %d tile width %d remainder %d \n",
		numInnermostRow,spacing,numInnermostRow % spacing);
	
	// Distribute remainder through tiles! :
	int Tileskips_between_remainder_points = numTilesOnGo/(numInnermostRow % spacing);
	
	printf("Tileskips between remainder +1's %d remainder %d\n",
		Tileskips_between_remainder_points, 
		numInnermostRow % Tileskips_between_remainder_points);
		
	// Note: Using greatest # connections AVOIDS accidental holes.
	// But so does using the full neighbour-based spread. ??
	// What if blocks meet... and we never get to do neighbours of those points.

	// ** What if we end block halfway through a neighbour list. Then we want to
	// have chosen wisely which points to use from that list.

	// Take array for resequence:
	long index[numTilesMajor][threadsPerTileMajor]; 
	long caret[numTilesMajor]; // checked up to caret, caret is the first one that can have a free neighbour.
	long iVertex, i, iTile;
	// Do we want to do this? Well, 
	// we can write on the vertices their search flags.
	// Then just hold on to the index of the earliest point for which we did not
	// exhaust the expansion to neighbours.
	// So we expand around each point -- remembering initial seq is random ...
	
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices;iVertex++)
	{
		pVertex->iVolley = -1; // which tile
		pVertex->iIndicator = 10000000; // link distance from initial seeds
		++pVertex;
	}
	
	memset(index,0,sizeof(long)*numTilesMajor*threadsPerTileMajor);
	int ctr = 0;

	long izNeigh[128], izNeigh2[128], izTri[128];
	Vertex * pCaret, * pNeigh, * pNeighNeigh;
	bool bTilesRemaining = true;
	int neigh_len, neigh_len2;
	long iTileStart = 0;
	long iThread, ii;
	i = 0; // let PBC go through 1st tile.
	iTile = 0;
	for (i = 0; i <= numInnermostRow-spacing;i += spacing)
	{
		index[iTile][0] = i;
		X[i].iVolley = iTile;
		X[i].iIndicator = 0;
		neigh_len = X[i].GetNeighIndexArray(izNeigh);
		for (ii = 0; ii < neigh_len; ii++)
		{
			pNeigh = X + izNeigh[ii];
			pNeigh->iIndicator = 1;
		}
		printf("i %d ; ",i);
		if (iTile % Tileskips_between_remainder_points == 0)
		{
			ctr++;
			if (ctr <= numInnermostRow % spacing) 
				i++; // extra move on 1
		};
		iTile++;
	}
	printf("\n");
	printf("final distance %d spacing %d \n",numInnermostRow-i+spacing, spacing);
	
	// iTile is now number of tiles along innermost row.
	numTilesOnGo = iTile;
	
	// What if we seeded whole thing: seeds kind of repel other seeds??
	// Sounds tricky ... how to know where to place seeds?
	// Carry on this way instead.
	
	// Check for greatest connections amongst the neighbours of the latest point ... vs...
	// check for greatest connections from all so far. 
	
	// Alternative: geometric sweep. Move sweepline radius out and attribute all within into some block.
	// According to existing neighbours. Sparser areas will make narrower blocks? Happens with the other way also.
	// If the sparseness is created by pushing to left and right then it's not so.
	// If the sparseness is created by upward stretch, block is stretched narrow.
	// Cannot see way to avoid.
	
	// Sweepline has its own problems. Makes assignment dependent on small variations in radius to determine
	// what else is already within sweepline.
	
	// . When we add a point, look at its neighbours: does one of them now have more connections than the 
	// otherwise next point??

	// Hmm

	// Start with neighs that have 3 connections or more, then neighs that have 2 connections.
	// This has to be updated __for_the_neighs__ as we go.
	// Then we move on to ... the one of the new 

	// I think we do need to store a list of the points in each tile so that we can work through them.
	// Refresh it when we get to the end of the tile.
	long lowestradius;
	int iWhich;
	memset(caret,0,sizeof(long)*numTilesMajor);
	// storing lowest neigh index for tile won't work:
	// can be altered if point is used up by another tile.
	
	while (iTileStart < numTilesMajor - numTilesOnGo) {
					/*
		// Debug output:
		FILE * fp;
		if (iTileStart == 280) {
			fp = fopen("tiles3.txt","a");
			for (iTile = 250; iTile < 280; iTile++)
			{
					fprintf(fp,"%d I x y   ",iTile);
			};
			fprintf(fp,"\n");
			
			for (iThread = 0; iThread < threadsPerTileMajor; iThread++)
			{
				for (iTile = 250; iTile < 280; iTile++)
				{
					fprintf(fp,"%d %d %1.10E %1.10E   ",
						iTile,index[iTile][iThread],(X + index[iTile][iThread])->pos.x,(X + index[iTile][iThread])->pos.y);
				};
				fprintf(fp,"\n");
			};

				fprintf(fp,"\n");
			for (long iVertex = 0; iVertex < numVertices; iVertex++)
			{
				pVertex = X + iVertex;
				if (pVertex->iVolley == -1) {
					fprintf(fp,"%d %1.10E %1.10E %d \n",iVertex,pVertex->pos.x,pVertex->pos.y,pVertex->iIndicator);
				}
			};

				fprintf(fp,"\n");
			fclose(fp);			
		};
*/
		
		for (iThread = 1; iThread <= threadsPerTileMajor; iThread++)
		{
			printf("\n iTileStart %d -- ",iTileStart);
			// <= because after the last point, we will set up the 1st point of the next tile.
			for (iTile = iTileStart; ((iTile < iTileStart + numTilesOnGo) &&
					 ((iThread < threadsPerTileMajor) || (iTile + numTilesOnGo + numTilesOnGo < numTilesMajor)))
								; iTile++)
			{
				printf("%d-",iTile);
				// Add 1 to all tiles on the go:
				
				// For this tile, we want to accumulate one further point.
				// We always choose a neighbour of the 'caret' that was the first point added
				// that still has neighbours unallocated.
				// As such, this is a neighbour-neighbour expansion. We do not search for the
				// unclaimed point that has the max number of connections.
				// However, we do search within the neighbours of the caret point, to find which
				// has the max connections to this tile.
				// If there is a tie, choose the earliest index -- although this is meaningless.
				
				if (caret[iTile] < iThread) {
					
					// We will choose one of the unallocated neighs of the tile that has the
					// lowest link radius to the base row.
					// Within those, choose one which has the max connections to this tile.
					// Within those, choose the earliest.

					// =====================================================================
					
					// First increment caret to where there will be any unallocated neighbours.
					
					int maxconnects, connects, numUnused;
					int unallocated = 0;
					// We may have wiped out all connections of caret, by the actions of other tiles if not this one.
					// Move forward in the list until we have unallocated neighbours or until we run out of list.
					do {	
						pVertex = X + index[iTile][caret[iTile]];
						if (pVertex->iVolley != iTile) {
							printf("(pVertex->iVolley != iTile) \n"); 
							getch();
						}; // defensive test
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i = 0; ((i < neigh_len) && (unallocated == 0)); i++)
						{
							pNeigh = X + izNeigh[i];
							if (pNeigh->iVolley == -1) 
								unallocated++;
						};
						
						if (unallocated == 0) {
							caret[iTile]++;
							if (caret[iTile] >= iThread) {
								// This means we are going to have to find a different point from which to
								// start more allocation to this tile.							
								// This case is going to be a problem even if we just placed the last point.
								printf("This is bad news! Tile %d covered over.",iTile);
								getch();							
							};
						};
					} while (unallocated == 0);
					
					lowestradius = 10000000;
										
					for (iWhich = caret[iTile]; iWhich < iThread; iWhich++)
					{
						pVertex = X + index[iTile][iWhich];
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i = 0; i < neigh_len; i++)
						{
							pNeigh = X + izNeigh[i];
							if ((pNeigh->iVolley == -1) &&
								(pNeigh->iIndicator < lowestradius))
								lowestradius = pNeigh->iIndicator;
						};
					};
					
					maxconnects = 0;
					for (iWhich = caret[iTile]; iWhich < iThread; iWhich++)
					{
						pVertex = X + index[iTile][iWhich];
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i = 0; i < neigh_len; i++)
						{
							pNeigh = X + izNeigh[i];
							if ((pNeigh->iVolley == -1) &&
								(pNeigh->iIndicator == lowestradius))
							{
								// Count connections:							
								neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
								connects = 0;
								for (ii = 0; ii < neigh_len2; ii++)
								{
									pNeighNeigh = X + izNeigh2[ii];
									if (pNeighNeigh->iVolley == iTile) connects++;								
								};
								if (connects > maxconnects) maxconnects = connects;
							};
						};
					};
					
					iWhich = caret[iTile];
					bool bNotFound = true;
					do {
						pVertex = X + index[iTile][iWhich];
						neigh_len = pVertex->GetNeighIndexArray(izNeigh);
						for (i = 0; ((i < neigh_len) && (bNotFound)); i++)
						{
							pNeigh = X + izNeigh[i];
							if ((pNeigh->iVolley == -1) &&
								(pNeigh->iIndicator == lowestradius))
							{
								// Count connections:							
								neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
								connects = 0;
								for (ii = 0; ii < neigh_len2; ii++)
								{
									pNeighNeigh = X + izNeigh2[ii];
									if (pNeighNeigh->iVolley == iTile) connects++;								
								};
								if (connects == maxconnects) bNotFound = false;
							};
						};
						iWhich++;
					} while (bNotFound);
					
					// pNeigh is the one to change. (Use pNeigh itself):

					if (iThread < threadsPerTileMajor) {

						pNeigh->iVolley = iTile;
						index[iTile][iThread] = pNeigh-X; // that will always work
					} else {
						// Note: we only ran this in case that
						// (iTile + numTilesOnGo < numTilesMajor - numTilesOnGo)

						pNeigh->iVolley = iTile + numTilesOnGo;
						index[iTile + numTilesOnGo][0] = pNeigh-X;
						// Okay, a problem: number of tiles is not divisible by numTilesOnGo?
						// We need to change that I expect.

						// It would be easier now if it was divisible.
						
					};
					neigh_len2 = pNeigh->GetNeighIndexArray(izNeigh2);
					for (ii = 0; ii < neigh_len2; ii++)
					{
						pNeighNeigh = X + izNeigh2[ii];
						if (pNeighNeigh->iIndicator > lowestradius + 1) 
							pNeighNeigh->iIndicator = lowestradius + 1;
					};

					// Note that we did not maintain caret[iTile] any more -- maybe it ran out of points, maybe not.
				} else {
					// (caret[iTile] >= iThread)
					
					// The tile points with unused neighbours have been exhausted.
					
					// We should follow a procedure that is similar to what we do when we first
					// start off a tile. 
					
					// If this is first point of tile, prefer we go to the anticlockwise side 
					// of the "previous" tile, numTilesOnGo less.
					// So find one of those points; move anticlockwise while on edge as much
					// as we can.
					
					// If we got the whole tile covered up by accident, that's a problem!
					// SO DO WHAT?
					// Mine our way out from the most anticlockwise point??
					// This won't be good anyway!
					// Better to start on the next tile when we run out of one.
									
					// We accumulate one point at a time so all run out together.
					// But this could still result in something being covered over?
					// That sounds nuts. Don't worry about efficiency in this unlikely case.
					
					// How about we just fail in that case.
					
					printf("Failed because tile covered over: iTile %d iThread %d numTilesOnGo %d iTileStart %d \n",
							iTile, iThread, numTilesOnGo, iTileStart);
					getch();
					
				}; // if (caret[iTile] < iThread)
			}; // next iThread
		}; // next iTile

		iTileStart += numTilesOnGo;
		if (numTilesMajor - iTileStart < numTilesOnGo)
			numTilesOnGo = numTilesMajor - iTileStart;
		
	}; // while (iTileStart < numTilesMajor) 
		
	// Now we have to do the last row. Do an azimuthal sweep.

	// Sort points.

	long thetaindex[2048];
	real thetaarray[2048];
	pVertex = X;
	i = 0;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->iVolley == -1) {
			thetaindex[i] = iVertex;
			thetaarray[i] = pVertex->pos.x/pVertex->pos.y;
			i++;
		};
		pVertex++;
	}
	long numUnallocated = i;
	QuickSort (thetaindex, thetaarray, 0, numUnallocated-1);
	printf("number left to fill tiles: %d unallocated %d \n",
		numTilesOnGo*threadsPerTileMajor, numUnallocated);
	if (numTilesOnGo*threadsPerTileMajor != numUnallocated)
	{
		printf("unequal.");
		getch();
	};
	i = 0;
	printf("iTile = %d",iTile);
	for (iTile = numTilesMajor-numTilesOnGo;iTile < numTilesMajor;iTile++)
	{
		for (iThread = 0; iThread < threadsPerTileMajor; iThread++)
		{
			index[iTile][iThread] = thetaindex[i];
			(X + thetaindex[i])->iVolley = iTile;
			i++;
		};
	};
	
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	for (iThread = 0; iThread < threadsPerTileMajor; iThread++)
	{
		pVertex = X + index[iTile][iThread];
		pVertex->iIndicator = iTile*threadsPerTileMajor + iThread;
	}
	
	printf("Reseq : vertex mapping done.\n");
	
	// ===============================================================================	
	// -------------------------------------------------------------------------------
	// So we can proceed by: 
	
	// Now comes the next: assign triangles into blocks also. :-(
	// More fiddlinesse :
	
	// First preference if triangle has 2 points within a vertex tile then assign it there.
	// We then try assigning 3-separate triangles to the tile that has the least assigned.
	// Do we end up with an unassignable triangle? In that case we have to see what?
	// Keep a record of how many 'negotiables' were attributed to each tile.
	// Set the unassignable to the one with the most negotiables.
	// Find one of the others and reassign it to a different tile.
	// If they all had 0 other negotiables, we fail.
	
	long tris_assigned[numTriTiles];
	long negotiables[numTriTiles];
	memset(tris_assigned,0,sizeof(long)*numTriTiles);
	memset(negotiables,0,sizeof(long)*numTriTiles);
	long iTri, iVolley0, iVolley1, iVolley2;
	Triangle * pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		iVolley0 = pTri->cornerptr[0]->iVolley;
		iVolley1 = pTri->cornerptr[1]->iVolley;
		iVolley2 = pTri->cornerptr[2]->iVolley;
		
		if ((iVolley0 == iVolley1) || (iVolley0 == iVolley2)) {
			pTri->indicator = iVolley0; // only 1 int available in triangle
			tris_assigned[iVolley0]++;
			if (tris_assigned[iVolley0] > threadsPerTileMinor) {
				printf("oh no - too many tris allocated on 1st pass to tri tile %d\n",
					iVolley0);
				getch();
			};
		} else {
			if (iVolley1 == iVolley2)
			{
				pTri->indicator = iVolley1;
				tris_assigned[iVolley1]++;
				if (tris_assigned[iVolley1] > threadsPerTileMinor) {
					printf("oh no - too many tris allocated on 1st pass to tri tile %d\n",
						iVolley1);
					getch();
				};
			} else {
				pTri->indicator = -1;
				negotiables[iVolley0]++;
				negotiables[iVolley1]++;
				negotiables[iVolley2]++;
			};
		};
		++pTri;
	}
	bool bThirdPassNeeded = false;
	long num0,num1,num2, nego0,nego1,nego2;
	
	// If tris_assigned + negotiables == threadsPerTileMinor then we
	// should start by assigning those...
	// If 1 more, we can go along assigning those, and so on.
	// Okay, that didn't work.


	// Going with a domain decomposition type of idea instead. For simplicity
	// let us take individual rows. We want to first try to assign the right number
	// of negotiables to each row. 
	// This may eventually break down but hopefully should be good enough initially.

	// Once we have them in a row, if we do not manage to get it right on a first pass,
	// and once shifting to immediate neighbours is exhausted, we can move extra negotiables
	// clockwise as long as there are joins. 
	// It does depend on having adequate shared negotiables for every pair in the row.
	// That is something to aim for in deciding which to keep for this row -- or, we might find
	// that it's not even possible to create a correct configuration.
		
	// Rows in seq since 1st row needs to claim what it can.

	long nego_2feet[numTriTiles]; // number of tris with 2 feet in this row, accessible to this tile.

	long numAttainedRow, numNeededForRow, iRow, numNegotRow;
	iTileStart = 0;
	int b0,b1,b2;
	for (iRow = 0; iRow < numTilesMajor/numTilesOnGo; iRow++)
	{
		// Claim ones for this row...
		Triangle * pTri = T;
		numNeededForRow = threadsPerTileMinor * numTilesOnGo;
		numAttainedRow = 0;
		numNegotRow = 0;
		for (iTile = iTileStart; iTile < iTileStart + numTilesOnGo; iTile++)
		{
			numAttainedRow += tris_assigned[iTile];			
		};
	
		memset(nego_2feet,0,sizeof(long)*numTriTiles); // scrap all the other rows - doesn't matter
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			// Collect nego_2feet so we attribute these as evenly as possible to the tiles in the row.
			if (pTri->indicator == -1)
			{
				iVolley0 = pTri->cornerptr[0]->iVolley;
				iVolley1 = pTri->cornerptr[1]->iVolley;
				iVolley2 = pTri->cornerptr[2]->iVolley;
				b0 = ((iVolley0 >= iTileStart) && (iVolley0 < iTileStart + numTilesOnGo))?1:0;
				b1 = ((iVolley1 >= iTileStart) && (iVolley1 < iTileStart + numTilesOnGo))?1:0;
				b2 = ((iVolley2 >= iTileStart) && (iVolley2 < iTileStart + numTilesOnGo))?1:0;
				if  (b0 + b1 + b2 >= 2) {
					if (b0 != 0) nego_2feet[iVolley0]++;
					if (b1 != 0) nego_2feet[iVolley1]++;
					if (b2 != 0) nego_2feet[iVolley2]++;
				};					
			};
			++pTri;
		};

		// Firstly: look for ones with 2 feet in this row:

		bool bEncounterAnything = 0;
		do { // go back to searching for inequality once chosen on an equality.
		
			// Exhaustive passes of INEQUALITY :
			bEncounterAnything = 0;
			bool bEncounteredTri = 0;
			
			do {
				bEncounteredTri = 0;
				pTri = T;
				for (iTri = 0; ((iTri < numTriangles) && (numNegotRow + numAttainedRow < numNeededForRow)); iTri++)
				{
					if (pTri->indicator == -1)
					{
						iVolley0 = pTri->cornerptr[0]->iVolley;
						iVolley1 = pTri->cornerptr[1]->iVolley;
						iVolley2 = pTri->cornerptr[2]->iVolley;
						b0 = ((iVolley0 >= iTileStart) && (iVolley0 < iTileStart + numTilesOnGo))?1:0;
						b1 = ((iVolley1 >= iTileStart) && (iVolley1 < iTileStart + numTilesOnGo))?1:0;
						b2 = ((iVolley2 >= iTileStart) && (iVolley2 < iTileStart + numTilesOnGo))?1:0;
						if  (b0 + b1 + b2 >= 2) {
							// assign to this row
							// but which tri?
							num0 = tris_assigned[iVolley0] + nego_2feet[iVolley0];
							num1 = tris_assigned[iVolley1] + nego_2feet[iVolley1];
							num2 = tris_assigned[iVolley2] + nego_2feet[iVolley2];
							
							if (b0+ b1+ b2 == 3) {
								if ((num0 < num1) && (num0 < num2)) {
									tris_assigned[iVolley0]++;
									pTri->indicator = iVolley0;
								} else {
									if (num1 < num2) {
										tris_assigned[iVolley1]++;
										pTri->indicator = iVolley1;
									} else {
										if (num2 < num1) {
											tris_assigned[iVolley2]++;
											pTri->indicator = iVolley2;
										};
									};
								};
							} else {
								if (b0 + b1 == 2) {
									if (num0 < num1) {
										tris_assigned[iVolley0]++;
										pTri->indicator = iVolley0;
									} else {
										if (num1 < num0) {
											tris_assigned[iVolley1]++;
											pTri->indicator = iVolley1;
										};
									};
								} else {
									if (b0 + b2 == 2) {
										if (num0 < num2) {
											tris_assigned[iVolley0]++;
											pTri->indicator = iVolley0;
										} else {
											if (num2 < num0) {
												tris_assigned[iVolley2]++;
												pTri->indicator = iVolley2;
											};
										};
									} else {
										if (num1 < num2) {
											tris_assigned[iVolley1]++;
											pTri->indicator = iVolley1;
										} else {
											if (num2 < num1) {
												tris_assigned[iVolley2]++;
												pTri->indicator = iVolley2;
											};
										};
									};
								};
							};
							if (pTri->indicator != -1) {
								nego_2feet[iVolley0]--;
								nego_2feet[iVolley1]--;
								nego_2feet[iVolley2]--;	
								negotiables[iVolley0]--;
								negotiables[iVolley1]--;
								negotiables[iVolley2]--;
								numNegotRow++;
								bEncounteredTri = true;
							};
							printf("iTri %d b012 %d %d %d : %d %d %d num %d %d %d : Assign to %d\n",
								iTri,b0,b1,b2,iVolley0,iVolley1,iVolley2,num0,num1,num2,pTri->indicator);
							
						}; // 2 feet in
					}	// unallocated	
					++pTri;
				}; // next tri
			} while (bEncounteredTri);
			if (bEncounteredTri == true) bEncounterAnything = true;
				
			// equality pass:
			bool bTookAnAction = 0;
			pTri = T;
			for (iTri = 0; ((iTri < numTriangles) && (numNegotRow + numAttainedRow < numNeededForRow)
				&& (bTookAnAction == 0)); // Added an extra cond: stop if assigned 1 triangle.
				iTri++)
			{
				if (pTri->indicator == -1)
				{
					iVolley0 = pTri->cornerptr[0]->iVolley;
					iVolley1 = pTri->cornerptr[1]->iVolley;
					iVolley2 = pTri->cornerptr[2]->iVolley;
					b0 = ((iVolley0 >= iTileStart) && (iVolley0 < iTileStart + numTilesOnGo))?1:0;
					b1 = ((iVolley1 >= iTileStart) && (iVolley1 < iTileStart + numTilesOnGo))?1:0;
					b2 = ((iVolley2 >= iTileStart) && (iVolley2 < iTileStart + numTilesOnGo))?1:0;
					if  (b0 + b1 + b2 >= 2) {
						// assign to this row
						// but which tri?
						num0 = tris_assigned[iVolley0] + nego_2feet[iVolley0];
						num1 = tris_assigned[iVolley1] + nego_2feet[iVolley1];
						num2 = tris_assigned[iVolley2] + nego_2feet[iVolley2];
						
						if (b0+ b1+ b2 == 3) {
							if ((num0 <= num1) && (num0 <= num2)) {
								tris_assigned[iVolley0]++;
								pTri->indicator = iVolley0;
							} else {
								if (num1 <= num2) {
									tris_assigned[iVolley1]++;
									pTri->indicator = iVolley1;
								} else {
									tris_assigned[iVolley2]++;
									pTri->indicator = iVolley2;
								};
							};
						} else {
							if (b0 + b1 == 2) {
								if (num0 <= num1) {
									tris_assigned[iVolley0]++;
									pTri->indicator = iVolley0;
								} else {
									tris_assigned[iVolley1]++;
									pTri->indicator = iVolley1;
								};
							} else {
								if (b0 + b2 == 2) {
									if (num0 <= num2) {
										tris_assigned[iVolley0]++;
										pTri->indicator = iVolley0;
									} else {
										tris_assigned[iVolley2]++;
										pTri->indicator = iVolley2;
									};
								} else {
									if (num1 <= num2) {
										tris_assigned[iVolley1]++;
										pTri->indicator = iVolley1;
									} else {
										tris_assigned[iVolley2]++;
										pTri->indicator = iVolley2;
									};
								};
							};
						};
						nego_2feet[iVolley0]--;
						nego_2feet[iVolley1]--;
						nego_2feet[iVolley2]--;	
						negotiables[iVolley0]--;
						negotiables[iVolley1]--;
						negotiables[iVolley2]--;
						printf("iTri %d b012 %d %d %d : %d %d %d num %d %d %d : Assign to %d\n",
							iTri,b0,b1,b2,iVolley0,iVolley1,iVolley2,num0,num1,num2,pTri->indicator);
						numNegotRow++;
						bEncounterAnything = true;
						bTookAnAction = true;
					};
				}			
				++pTri;
			};
		} while ((bEncounterAnything) && (numNegotRow + numAttainedRow < numNeededForRow));
		
		// If that did not assign all the ones to this row that we could have wanted,
		// we need to go again and bring in some more.
		bool bUsedOnly2Feet;
		if (numNegotRow + numAttainedRow < numNeededForRow) {
			printf("iRow %d didn't fill with 2-in triangles.\n",iRow);
			bUsedOnly2Feet = false;
		} else {
			printf("iRow %d done2ft numNegotRow %d numAttainedRow %d numNeeded %d\n",iRow,
				numNegotRow,numAttainedRow,numNeededForRow);
			bUsedOnly2Feet = true;
		};
		
		// Now we are going to have to use ones with only 1 corner in this row of tiles.
		// However, find the ones for which the other tris have the highest minimum number of T+N.
		long highmin, numAssigned;
		
		while (numNegotRow + numAttainedRow < numNeededForRow) {
			highmin = 0;
			for (iTri = 0; ((iTri < numTriangles) && (numNegotRow + numAttainedRow < numNeededForRow)); iTri++)
			{
				if (pTri->indicator == -1)
				{
					iVolley0 = pTri->cornerptr[0]->iVolley;
					iVolley1 = pTri->cornerptr[1]->iVolley;
					iVolley2 = pTri->cornerptr[2]->iVolley;
					b0 = ((iVolley0 >= iTileStart) && (iVolley0 < iTileStart + numTilesOnGo))?1:0;
					b1 = ((iVolley1 >= iTileStart) && (iVolley1 < iTileStart + numTilesOnGo))?1:0;
					b2 = ((iVolley2 >= iTileStart) && (iVolley2 < iTileStart + numTilesOnGo))?1:0;
					if  (b0 + b1 + b2 >= 1) {
						num0 = tris_assigned[iVolley0] + negotiables[iVolley0];
						num1 = tris_assigned[iVolley1] + negotiables[iVolley1];
						num2 = tris_assigned[iVolley2] + negotiables[iVolley2];
					
						// assign to this row?
						// Collect highest minimum
						if (b0 + b1 == 0) {
							if (min(num0,num1) > highmin) highmin = min(num0,num1);
							// There were more available here, signifying a better place to steal tri to 'our' row of tiles.
						};
						if (b0 + b2 == 0) {
							if (min(num0,num2) > highmin) highmin = min(num0,num2);
						};
						if (b1 + b2 == 0) {
							if (min(num1,num2) > highmin) highmin = min(num1,num2);
						};			
					};
				}; // if unallocated	
				++pTri;
			}; // next iTri
			
			numAssigned = 0;
			// Now repeat with highmin used as signal to assign.
			for (iTri = 0; ((iTri < numTriangles) && (numNegotRow + numAttainedRow < numNeededForRow)); iTri++)
			{
				if (pTri->indicator == -1)
				{
					iVolley0 = pTri->cornerptr[0]->iVolley;
					iVolley1 = pTri->cornerptr[1]->iVolley;
					iVolley2 = pTri->cornerptr[2]->iVolley;
					b0 = ((iVolley0 >= iTileStart) && (iVolley0 < iTileStart + numTilesOnGo))?1:0;
					b1 = ((iVolley1 >= iTileStart) && (iVolley1 < iTileStart + numTilesOnGo))?1:0;
					b2 = ((iVolley2 >= iTileStart) && (iVolley2 < iTileStart + numTilesOnGo))?1:0;
					if  (b0 + b1 + b2 >= 1) {
						// assign to this row?
						// Collect highest minimum
						if (((b0 + b1 == 0) && (min(num0,num1) == highmin))
							||
						    ((b0 + b2 == 0) && (min(num0,num2) == highmin))
						    ||
						    ((b1 + b2 == 0) && (min(num1,num2) == highmin)))
						{
							negotiables[iVolley0]--;
							negotiables[iVolley1]--;
							negotiables[iVolley2]--;
							if (b0 != 0){
								tris_assigned[iVolley0]++;
								pTri->indicator = iVolley0;
							};
							if (b1 != 0) {
								tris_assigned[iVolley1]++;
								pTri->indicator = iVolley1;
							};
							if (b2 != 0) {
								tris_assigned[iVolley2]++;
								pTri->indicator = iVolley2;
							};
							numNegotRow++;
							numAssigned++;
						};
					};
					// Of course we are giving symmetrical treatment above and below
					// this row -- which is probably not the correct thing to do.
				}; // if unallocated	
				++pTri;
			}; // next iTri
			
			printf("Assigned %d to row %d at highmin %d \n",numAssigned,iRow,highmin);
		}; // wend

		// Now within this row it's possible that we assigned too few to some triangles and too many
		// to others -- so we've got to try and shuttle them along to get it right.
		
		// First go for next-door transfers... start with ones that can give/take only from
		// one side.


		// Need to put this in.

		bool bChangesOnThisPass, bFound;
		int inext, iprev, iprevprev, inextnext;
		do {
			bChangesOnThisPass = false;
		for (iTile = iTileStart; iTile < iTileStart + numTilesOnGo; iTile++)
		{
			iprev = iTile-1;
			if (iprev < iTileStart) iprev += numTilesOnGo;
			inext = iTile+1;
			if (inext == iTileStart + numTilesOnGo) inext = iTileStart;
			iprevprev = iprev-1;
			if (iprevprev < iTileStart) iprevprev += numTilesOnGo;
			inextnext = inext+1;
			if (inextnext == iTileStart + numTilesOnGo) inextnext = iTileStart;

			bFound = true;
			while ( (bFound == true) &&
					(tris_assigned[iTile] > threadsPerTileMinor)
				&& (tris_assigned[iprev] < threadsPerTileMinor)
				&& ((tris_assigned[inext] >= threadsPerTileMinor)
						||
					(tris_assigned[iprevprev] <= threadsPerTileMinor))
				// either we can only give to 1 place or it can only get 
				// from 1 place.
				)
			{
				bFound = false;
				// Seeing if we can reassign something from here to previous tile.
				// switch over ... prefer to switch one that is
				// 3 separate feet, so do this pass first.
				pTri = T;
				for (iTri = 0; (  (iTri < numTriangles) &&
					(tris_assigned[iTile] > threadsPerTileMinor) && 
					(tris_assigned[iprev] < threadsPerTileMinor)
									) ; iTri++)
				{
					if (pTri->indicator == iTile) {
						iVolley0 = pTri->cornerptr[0]->iVolley;
						iVolley1 = pTri->cornerptr[1]->iVolley;
						iVolley2 = pTri->cornerptr[2]->iVolley;
						if ( ( ((iVolley0 != iTile) && (iVolley0 != iprev)) ||
							   ((iVolley1 != iTile) && (iVolley1 != iprev)) ||
							   ((iVolley2 != iTile) && (iVolley2 != iprev)) )
						      &&
						   ((iVolley0 == iprev)	||
							(iVolley1 == iprev) ||
							(iVolley2 == iprev)) ) 
						{
							tris_assigned[iTile]--;
							tris_assigned[iprev]++;
							pTri->indicator = iprev;
							bFound = true;
							bChangesOnThisPass = true;
						};						
					};
					++pTri;
				};
				// Possible we found none to use...
			};

			// Back the other way:
			bFound = true;
			while ( (bFound == true) &&
					(tris_assigned[iTile] > threadsPerTileMinor)
				&& (tris_assigned[inext] < threadsPerTileMinor)
				&& ((tris_assigned[iprev] >= threadsPerTileMinor)
						||
					(tris_assigned[inextnext] <= threadsPerTileMinor))
				// Either we can only give to 1 place or it can only get from 1 place.
				)
			{
				bFound = false;
				// Seeing if we can reassign something from here to previous tile.
				// switch over ... prefer to switch one that is
				// 3 separate feet, so do this pass first.
				pTri = T;
				for (iTri = 0; (  (iTri < numTriangles) &&
					(tris_assigned[iTile] > threadsPerTileMinor) && 
					(tris_assigned[inext] < threadsPerTileMinor)
									) ; iTri++)
				{
					if (pTri->indicator == iTile) {
						iVolley0 = pTri->cornerptr[0]->iVolley;
						iVolley1 = pTri->cornerptr[1]->iVolley;
						iVolley2 = pTri->cornerptr[2]->iVolley;
						if ( ( ((iVolley0 != iTile) && (iVolley0 != iprev)) ||
							   ((iVolley1 != iTile) && (iVolley1 != iprev)) ||
							   ((iVolley2 != iTile) && (iVolley2 != iprev)) )
						      &&
						   ((iVolley0 == iprev)	||
							(iVolley1 == iprev) ||
							(iVolley2 == iprev)) ) 
						{
							tris_assigned[iTile]--;
							tris_assigned[inext]++;
							pTri->indicator = inext;
							bFound = true;
							bChangesOnThisPass = true;
						};						
					};
					++pTri;
				};
				// Possible we found none to use...
			};
			
			// Do this test for each tile pair and then go around for more 
			// passes ... working on the end of any chain. 
			
		}; // next iTile
		} while (bChangesOnThisPass); // next pass	
		
		// Now unless we were unlucky about available corners, 
		// a chain can end only in 2+ or 2- if it has both + and -.
		
		// Let's just do a test and see if it ever fails at this point, to get going initially:
		bool bFindWrong, bTookAnAction;
		do {
			bFindWrong = false;
			for (iTile = iTileStart; iTile < iTileStart + numTilesOnGo; iTile++)
			{
				if (tris_assigned[iTile] != threadsPerTileMinor)
				{
					printf("Problem: iRow %d iTileStart %d iTile %d \n",
						iRow,iTileStart,iTile);
					for (i = iTileStart; i < iTileStart + numTilesOnGo; i++)
						printf("%d ",tris_assigned[i]);
					printf("\n");
					bFindWrong = true;
					if (bUsedOnly2Feet == false) {
					// Then we expect something a bit stranger...
					printf("weirdness12345.\n");
					getch();
				}
				};
				// Yes it does fail sometimes.
				// So we have to improve : try finding which tile has TOO MANY, move that
				// clockwise until we encounter TOO FEW.
				
				inext = iTile+1;
				if (inext == iTileStart + numTilesOnGo) inext = iTileStart;
				
		
				if ((tris_assigned[iTile] > threadsPerTileMinor) && (tris_assigned[inext] < tris_assigned[iTile]))
				{
					bTookAnAction = 0;
					pTri = T;
					for (iTri = 0; ((iTri < numTriangles) && (bTookAnAction == 0)); iTri++)
					{
						if (pTri->indicator == iTile)
						{
				
							// does it have a foot in next tile and in another?
							iVolley0 = pTri->cornerptr[0]->iVolley;
							iVolley1 = pTri->cornerptr[1]->iVolley;
							iVolley2 = pTri->cornerptr[2]->iVolley;
							if (((iVolley0 == inext) || (iVolley1 == inext) || (iVolley2 == inext))
								&&
								(
							((iVolley0 >= iTileStart+numTilesOnGo) || (iVolley0 < iTileStart))
								||
							((iVolley1 >= iTileStart+numTilesOnGo) || (iVolley1 < iTileStart))
								||
							((iVolley2 >= iTileStart+numTilesOnGo) || (iVolley2 < iTileStart))
							) ) {
								// This is one to switch across.
								bTookAnAction = true;
								
								pTri->indicator = inext;
								tris_assigned[iTile]--;
								tris_assigned[inext]++;

							}
						};
						++pTri;
					}; // next iTri
					if (bTookAnAction == 0) {
						printf("Deeper problem -- cannot move an extra peg clockwise\n");
						getch();
					};
				};	// is this a tile to alter		
			};		// next iTile
		} while (bFindWrong);

		iTileStart += numTilesOnGo;
	} // next iRow
	
	// Now since we did within each row, it should be done...
	
	// Old attempt:
	/*
	
	FILE * fp;

	bool bEncountered;
	long numUnassigned, numAssigned;
	int addition = 0;
	int iPass = 0;
	do {
		bEncountered = false;
		numUnassigned = 0;
		numAssigned = 0;
		
		
		
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			if (pTri->indicator == -1) {
				numUnassigned++;
				
				// 3 separate vertex tiles
				iVolley0 = pTri->cornerptr[0]->iVolley;
				iVolley1 = pTri->cornerptr[1]->iVolley;
				iVolley2 = pTri->cornerptr[2]->iVolley;
				num0 = tris_assigned[iVolley0] + negotiables[iVolley0];
				num1 = tris_assigned[iVolley1] + negotiables[iVolley1];
				num2 = tris_assigned[iVolley2] + negotiables[iVolley2];
				// Sort : A. By tris_assigned: fewer assigned => assign more
				//        B. By negotiables: fewer negotiables => assign to that one
				
				if ((num0 < threadsPerTileMinor) ||
					(num1 < threadsPerTileMinor) ||
					(num2 < threadsPerTileMinor))
				{
					num0 = num0;
				}
				
				// Sometimes num1 is already now less than tPTM+addition but num0== it.
				// In that case clearly we wanted to assign to the one with less.
				bool bFound = false;
				for (int add_used = 0; ((add_used < addition) && (bFound == false)); add_used++)
				{
					if (num0 == threadsPerTileMinor + add_used) {
						
						pTri->indicator = iVolley0;
						tris_assigned[iVolley0]++;
						negotiables[iVolley0]--;					
						
						if (num1 <= threadsPerTileMinor) printf("Problem: iVolley0 %d iVolley1 %d \n",iVolley0,iVolley1);
						if (num2 <= threadsPerTileMinor) printf("Problem: iVolley0 %d iVolley2 %d \n",iVolley0,iVolley2);
						
						negotiables[iVolley1]--;
						negotiables[iVolley2]--;
						bEncountered = true;
						bFound = true;
						numAssigned++;
					} else {
						if (num1 == threadsPerTileMinor + add_used) {
							pTri->indicator = iVolley1;
							tris_assigned[iVolley1]++;
							negotiables[iVolley1]--;
							
							if (num0 <= threadsPerTileMinor)printf("Problem: iVolley1 %d iVolley0 %d \n",iVolley1,iVolley0);
							if (num2 <= threadsPerTileMinor)printf("Problem: iVolley1 %d iVolley2 %d \n",iVolley1,iVolley2);
							
							negotiables[iVolley0]--;
							negotiables[iVolley2]--;
							bEncountered = true;
							bFound = true;							
							numAssigned++;
						} else {
							if (num2 == threadsPerTileMinor + add_used) {
								pTri->indicator = iVolley2;
								tris_assigned[iVolley2]++;
								negotiables[iVolley2]--;
								
								if (num0 <= threadsPerTileMinor)printf("Problem: iVolley2 %d iVolley0 %d \n",iVolley2,iVolley0);
								if (num1 <= threadsPerTileMinor)printf("Problem: iVolley2 %d iVolley1 %d \n",iVolley2,iVolley1);
								
								negotiables[iVolley0]--;
								negotiables[iVolley1]--;
								bEncountered = true;
								bFound = true;
								numAssigned++;
							};
						};
					};
				};
			};
			++pTri;
		}; // next iTri
		
		// Run at 0 addition until none encountered
		// Run at 1 --> if encounter, go back to 0; if not, go on to 2
		// Run at 2 --> if encounter, go back to 0; 
		if (bEncountered == true) {
			addition = 0;
		} else {
			addition++; // In fact it just keeps going up and up with none encountered.
		};

		fp = fopen("tris_assigned.txt","w");
		for (iTile = 0; iTile < numTriTiles; iTile++)
			fprintf(fp,"%d : assigned %d negotiable %d \n",iTile,tris_assigned[iTile],negotiables[iTile]);
		fclose(fp);

		printf("Pass %d addition %d ; unassigned %d of which %d assigned\n", iPass, addition, numUnassigned,
			numAssigned);
		getch();
		// Gets stuck in a loop with about 3000 unassigned, 0 being assigned.

		iPass++;
	} while (numUnassigned > 0);
	*/
	printf("Tri resequence done...\n");
	getch();
		// Now turn 'volley' information into a sequence.
	// WE CHANGE WHAT indicator MEANS :
	
	memset(tris_assigned,0,sizeof(long)*numTriTiles);
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		int iWhich = pTri->indicator;
		pTri->indicator = iWhich*threadsPerTileMinor + tris_assigned[iWhich];
		tris_assigned[iWhich]++;
		++pTri;
	};
	
	// We then can set the new sequence and try to resequence the triangles, and affect the
	// triangle index lists.
	//  - Create 2nd system:
	
	pVertex = X;
	Vertex * pVertdest;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertdest = pDestMesh->X + pVertex->iIndicator;
		
		pVertdest->CopyDataFrom(pVertex);
		pVertdest->CopyLists(pVertex);
		// but now we need to change them:
		
		// Only neighbour list is relevant so far...
		pVertdest->ClearNeighs();
		int neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		for (i = 0; i < neigh_len; i++)
		{
			pVertdest->AddNeighbourIndex((X + izNeigh[i])->iIndicator);
		};
		
		pVertdest->ClearTris();
		// Only neighbour list is relevant so far...
		int tri_len = pVertex->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			pVertdest->AddTriIndex((T + izTri[i])->indicator);
		};
		
		// Careful about other stuff.
	}
	// How to fill in lists? Need to remap: if old list said 25, new list says i[25]
	// For neighbours.
		
	Triangle * pTriDest;
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTriDest = pDestMesh->T + pTri->indicator;
		// 
		pTriDest->u8domain_flag = pTri->u8domain_flag;
		pTriDest->area = pTri->area;
		pTriDest->cent = pTri->cent;
		pTriDest->periodic = pTri->periodic;
		memcpy(pTriDest->edge_normal,pTri->edge_normal,sizeof(Vector2)*3);
		
		// Neighbour list, cornerptr list, what else to rewrite?
		
		pTriDest->neighbours[0] = pDestMesh->T + pTri->neighbours[0]->indicator;
		pTriDest->neighbours[1] = pDestMesh->T + pTri->neighbours[1]->indicator;
		pTriDest->neighbours[2] = pDestMesh->T + pTri->neighbours[2]->indicator;

		pTriDest->cornerptr[0] = pDestMesh->X + pTri->cornerptr[0]->iIndicator;
		pTriDest->cornerptr[1] = pDestMesh->X + pTri->cornerptr[1]->iIndicator;
		pTriDest->cornerptr[2] = pDestMesh->X + pTri->cornerptr[2]->iIndicator;
		
		
		
		
		++pTri;
	}
	
	printf("Tiling & resequencing done.\n");
	
	
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	//  - Create Systdata to match it
	//  - When we come back from GPU: 
	//    -- we want to refresh this post-Delaunay flips.
	//    -- can a flip trigger a local swap of indices to move smth between blocks?
	
	//    -- We want that to operate on Systdata not just System, because System does
	//       not have data on triangles.
	
	//   -- (On the other hand there is an argument that since System has objects for
	//      vertex and triangle, it will be easier to avoid making mistakes if we did for System as well.)
} 
// This is something needed for sure.

#endif
