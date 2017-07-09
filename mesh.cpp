
#include "basics.cpp"

#define TRIS_TO_TRIS             0
#define TRI_AND_WEDGE_TO_WEDGES  1
#define WEDGES_TO_WEDGE_AND_TRI  2

extern long steps_remaining;

int GlobalWedgeSwitch; 
smartlong GlobalAffectedTriIndexList;


	TriMesh::TriMesh()
	{
		X = NULL;
		Xdomain = NULL;
		T = NULL;
	//	InnerX = NULL;
	//	InnerT = NULL;

		numVertices = 0;
		numDomainVertices = 0;
	//	numInnerVertices = 0;
	//	numInnerTriangles = 0;

		numTriangles = 0;
		numTrianglesAllocated = 0;
		numRows = 0;

		numInnerVertices = 0;

		ZeroMemory(AuxX,sizeof(Vertex *)*NUM_COARSE_LEVELS);
	//	ZeroMemory(AuxT,sizeof(AuxVertex *)*NUM_COARSE_LEVELS);

		//numVertsLow = 0;
		//numVertsOuter = 0;
	//	insulator_verts = 0;
	//	outer_verts = 0;
	};

	TriMesh::~TriMesh()
	{
		if (X != NULL) delete [] X;
		X = NULL;
		if (T != NULL) delete [] T;
		T = NULL;
		//if (InnerX != NULL) delete [] InnerX;
		//if (InnerT != NULL) delete [] InnerT;
		
		//numInnerVertices = 0;
		//numInnerTriangles = 0;
		numVertices = 0;
		numTriangles = 0;
		numTrianglesAllocated = 0;
		
		Disconnected.clear();
		TriangleHeap.clear();
		ScratchSearchTris.clear();
	};


real inline distsq(Vector2 const u1, Vector2 const u2)
{
	return (u1.x-u2.x)*(u1.x-u2.x)+(u1.y-u2.y)*(u1.y-u2.y);
};

real GetCos(const Vector2 & u1, const Vector2 & centre, const Vector2 & u2)
{
	Vector2 diff1 = u1-centre;
	Vector2 diff2 = u2-centre;

	// a.b = |a| |b| cos theta

	real a_dot_b = diff1.x*diff2.x+diff1.y*diff2.y;
	real costheta = a_dot_b/sqrt((diff1.x*diff1.x+diff1.y*diff1.y)*(diff2.x*diff2.x+diff2.y*diff2.y));
	return costheta;
};

real GetCos(const Vector2 & v1, const Vector2 & v2)
{
	real costheta = v1.dot(v2)/sqrt(
		(v1.x*v1.x+v1.y*v1.y)*(v2.x*v2.x+v2.y*v2.y));
	// for this version, send the 2 difference vectors.
	return costheta;
}

Vector2 Vertex::PopulateContiguousPosition__Guesswork(Vertex * pVertex)
	{
		Vector2 u = pos;
		if ((pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y)
			&& ( u.x/u.y > 0.5*GRADIENT_X_PER_Y))
			u = Anticlockwise*u; 	
		if ((pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y)
			&& ( u.x/u.y < -0.5*GRADIENT_X_PER_Y))
			u = Clockwise*u;
		return u;
	}





/*
void Triangle::Set(Vertex * pV1, Vertex * pV2, Vertex * pV3)
{
	// set the cornerptr to these values and tell vertices to add this triangle also.

	// Set Triangle::flags based on Vertex::flags
	// and Triangle::periodic based on Vertex::iScratch which records relative wrapping apparently.

	cornerptr[0] = pV1;
	cornerptr[1] = pV2;
	cornerptr[2] = pV3;
	pV1->triangles.add(this);
	pV2->triangles.add(this);
	pV3->triangles.add(this);
	
	periodic = cornerptr[0]->iScratch + cornerptr[1]->iScratch + cornerptr[2]->iScratch;
	while (periodic >= 3) periodic -= 3;
	while (periodic < 0) periodic += 3;

}*/	// Do not have this: with integer arrays need to use TriMesh object.


char Triangle::InferRelativeWrapping(Vertex * pVert, Vertex * pVertDisco)
{
	if (periodic == 0) return 0;

	int c1;

	if (periodic == 1)
	{
		c1 = GetLeftmostIndex();
		if (cornerptr[c1] == pVert) {
			// pVert is wrapped relative to the others, Clockwise
			return 1;
		};
		if (cornerptr[c1] == pVertDisco) {
			return -1;
		};

		return 0; // the other vertex was wrapped only
	};
	
	// got here => periodic == 2
	
	c1 = GetRightmostIndex();
	if (cornerptr[c1] == pVert) {
		// pVert is wrapped Anticlockwise
		return -1;
	};
	if (cornerptr[c1] == pVertDisco) return 1;
	return 0;

}


int TriMesh::SeekVertexInsideTriangle(Vertex * v1,
									  Vertex * v2,
									  Vertex * v3,          // up to 4 points to check for
									  Vertex * v4,           // in order of preference;
									  Triangle * pSeedTri,    // Seed triangle to begin radar
									  Vertex ** ppReturnVert,  // address for returning guilty vertex
									  Triangle ** ppReturnTri) // address for returning triangle that contains
{
	// non-directed radar outwards from pSeedTri to find if a triangle contains
	// one of the listed vertices;
	// return 0,1,2,3 for which one it was; 
	// fill in *ppReturnVert with that vertex address.
	
	// return -1 if no vertex in tri interior found

	// Note that if something is a vertex of a triangle it cannot be considered to be within that triangle.
	
	Vertex * pVert[4];
	int i;
	Triangle *pTri;

	pVert[0] = v1;
	pVert[1] = v2;
	pVert[2] = v3;
	pVert[3] = v4;

	for (i = 0; i < 4; i++)
	if (pSeedTri->ContainsPointInterior(pVert[i]))
	{
		*ppReturnVert = pVert[i];
		*ppReturnTri = pSeedTri;
		return i;
	};

	ScratchSearchTris.clear();
	ScratchSearchTris.add(pSeedTri-T);
	long checkcaret = 1; // the next unchecked element
	// add neighbours to list:
	ScratchSearchTris.add_unique(pSeedTri->neighbours[0]-T);
	ScratchSearchTris.add_unique(pSeedTri->neighbours[1]-T);
	ScratchSearchTris.add_unique(pSeedTri->neighbours[2]-T);

	do
	{
		// for each unchecked triangle: check it; if it's not a hit, add all its neighbours (unique add) to list.
		// make sure we did not run out of neighbours:

		if (checkcaret >= ScratchSearchTris.len) {
			// 3rror
			printf("Did not find enough triangle neighbours in radar search! \n");
			getch();
		};

		pTri = T+ScratchSearchTris.ptr[checkcaret];
		
		for (i = 0; i < 4; i++)
		if (pTri->ContainsPointInterior(pVert[i]))
		{
			*ppReturnVert = pVert[i];
			*ppReturnTri = pTri;
			return i;
		};
		
		// if we're still here, it did not contain one of them. Add all its neighbours to look in:
		ScratchSearchTris.add_unique(pTri->neighbours[0]-T);
		ScratchSearchTris.add_unique(pTri->neighbours[1]-T);
		ScratchSearchTris.add_unique(pTri->neighbours[2]-T);		
		checkcaret++;
	} while (checkcaret < 2000); // arbitrary cutoff

	return -1;	
}

int TriMesh::DestroyOverlaps(int max_attempts)
{
	// Let's just search all triangles and see if we can find shared edges.
	
	// Can assume neighbours[0] is the triangle that shares edge [vertex1, vertex2]
	
	long iTri;
//	int j;
	Triangle * pTri;//, * pTri2;
	//Vertex * pvert, *q2;
	//Vector2 u;
	//bool test;
	long actions = 0;
	int attempts = 0;
	//int q2_cornerind;
	long iNeigh = 0; // not used?
//	Triangle *pNeigh; // not useful

	// FIRST DISCONNECT VERTICES UNTIL NO MORE OVERLAPS EXIST.

	TriangleHeap.clear();
	Disconnected.clear();

	do
	{
		actions = 0;
		pTri = T;// &(T[0]);
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			// Note that inside ExamineNeighbour, we do a test for if pTri has already been scrapped.
	
			// Look at each neighbour to make a pair and see if edge is shared.
			// If it is, disconnect some vertex and reconnect.

			if (pTri->neighbours[0] != pTri) // do not do comparison at the edge of the domain!
				actions += ExamineNeighbourAndDisconnectIfNeeded(pTri,0,1);
			if (pTri->neighbours[1] != pTri)
				actions += ExamineNeighbourAndDisconnectIfNeeded(pTri,1,0);
			if (pTri->neighbours[2] != pTri)			
				actions += ExamineNeighbourAndDisconnectIfNeeded(pTri,2,0);
			 
			++pTri;
		};
		//printf("TriMesh::DestroyOverlaps Actions taken: %d \n",actions);
		++attempts;
		
	} while ((actions > 0) && (attempts < max_attempts));


	printf("TriMesh::DestroyOverlaps disconnected: %d \n",Disconnected.len);
	//if (Disconnected.len > 0) getch();

	if (attempts == max_attempts)
	{
		printf("****************************\n");
		printf("Max attempts to destroy overlaps by disconnecting points exceeded.\n");
		getch();
	};

	if (TriangleHeap.len != 2*Disconnected.len)
	{ 
		printf("****************************\n");
		printf("Wrong amount of triangles and disco points accumulated:\n %d tris, %d points\n",
			TriangleHeap.len,Disconnected.len);
		getch();
	};

	// Now we have a heap of disconnected points and unused triangles.

	// NOW RECONNECT POINTS IN SAFETY

	while (Disconnected.len > 0)
	{
		// Reconnect 
		ReconnectLastPointInDiscoArray(); // this will also remove it from array and remove tri's from their array
	};
	
	// Replenish vertex neighbour array in untangled mesh:
	// have checked that it does not need to be intermediately kept refreshed above.
	if (attempts > 1) this->RefreshVertexNeighboursOfVerticesOrdered();
	
	return attempts-1;
	
}


	
int TriMesh::ExamineNeighbourAndDisconnectIfNeeded(Triangle * pTri, int opp, int c1)
{
	// c1 is on the edge we check
	// opp is opposite the edge we check
	// If pTri is a wedge, opp will not ==2.

	Triangle * pTri2, *pTriContain;
	Vertex * q2;
	int q2_cornerind; // not used for anything
	Vertex * pVertUse, * pVertRogue;
	Vector2 u,unsh2;
	Triangle *pNeigh; // not used for anything
	bool test;
	int which;
	int iMapped,iUnmapped;
	
	if (TriangleHeap.contains(pTri-T)) // can end up calling this function for scrapped triangle -- do nothing
		return 0;
	
	pTri2 = pTri->neighbours[opp];
	q2 = pTri2->ReturnUnsharedVertex(pTri,&q2_cornerind);
	
	// If they're both wedges, q2_cornerind will be 0 or 1
	
	if (pTri2->periodic == 0)
	{
		if (pTri->periodic == 0)
		{
			test = pTri->TestAgainstEdge(
						 q2->pos.x,q2->pos.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );
		} else {
			// pTri is periodic but pTri2 is not.
			// This means the shared edge is inside;
			
			// pTri->periodic == 1 means shared edge on right so map
			// q2 to compare with pTri on the left because we map pTri fully to the left.
			
			// pTri->periodic == 2 means q2 is already on the left and we make sure
			// we map pTri->c1 to the left if it's needing to be mapped.

			if (pTri->periodic == 1)
			{
				unsh2 = Anticlockwise*q2->pos;
				test = pTri->TestAgainstEdge(
						 unsh2.x,unsh2.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );
			} else {
				// We edited to ensure that c1 point does get mapped to left.
				test = pTri->TestAgainstEdge(
						 q2->pos.x,q2->pos.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );
			};
		};

	} else {

		// pTri2->periodic > 0

		if (pTri->periodic == 0)
		{

			// pTri2 is periodic but pTri is not.
			// This means shared edge is inside.
			// pTri2->periodic == 1 => that one is q2 and so it must be mapped
			// to the RIGHT to be with pTri
			// pTri2->periodic == 2 => q2 is on the right but pTri is on the left
			if (pTri2->periodic == 1)
			{
				unsh2 = Clockwise*q2->pos;						
			} else {
				unsh2 = Anticlockwise*q2->pos;
			};
			test = pTri->TestAgainstEdge(
					 unsh2.x,unsh2.y, 
					 c1,          // edge corner
					 opp,         // the opposite point
					 &pNeigh      // not used
					 );
		} else {
			// since pTri is periodic, comparison is done on the left
			// is q2 on the left?
			if (pTri2->periodic == 1)
			{
				iMapped = pTri2->GetLeftmostIndex();
				if (iMapped == q2_cornerind) {
					// already mapped
					test = pTri->TestAgainstEdge(
						 q2->pos.x,q2->pos.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );
				} else {
					unsh2 = Anticlockwise*q2->pos;

					test = pTri->TestAgainstEdge(
						 unsh2.x,unsh2.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );	
				};
			} else {
				iUnmapped = pTri2->GetRightmostIndex();
				if (iUnmapped == q2_cornerind) {
					unsh2 = Anticlockwise*q2->pos;

					test = pTri->TestAgainstEdge(
						 unsh2.x,unsh2.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );	
				} else {
					test = pTri->TestAgainstEdge(
						 q2->pos.x,q2->pos.y, 
						 c1,          // edge corner
						 opp,         // the opposite point
						 &pNeigh      // not used
						 );
				};
			};					
		};
	};
	// still within tri vs tri

	if (!test)
	{
		int c2 = 0;
		while ((c2 == c1) || (c2 == opp)) c2++;
		which = SeekVertexInsideTriangle(
			pTri->cornerptr[opp],
			q2,
			pTri->cornerptr[c1],   // up to 4 points to check for
			pTri->cornerptr[c2],	// in order of preference;
			pTri,				// Seed triangle to begin radar
			&pVertRogue,      // address for returning guilty vertex
			&pTriContain);

		if (which == -1) // found no vertex within a triangle. Strange?
		{
			printf("\nOverlap %d %d No vert within tri\n",pTri-T,pTri2-T);
			
			// kill the further one

			real dist1sq = distsq(pTri->cornerptr[opp]->pos,pTri->cornerptr[c1]->pos)
				          + distsq(pTri->cornerptr[opp]->pos,pTri->cornerptr[c2]->pos);

			real dist2sq = distsq(q2->pos,pTri->cornerptr[c1]->pos) + distsq(q2->pos,pTri->cornerptr[c2]->pos);
			
			if (dist1sq > dist2sq)
			{
				FullDisconnect(pTri->cornerptr[opp],pTri2);
			} else {
				FullDisconnect(q2,pTri);
			};
			
			return 1;
		};

		if ((pVertRogue == pTri->cornerptr[opp]) ||
			(pVertRogue == q2))
		{
			// an unshared point --
			// choose the closer of shared points for pVertUse
			real dist1sq = distsq(pVertRogue->pos,pTri->cornerptr[c1]->pos);
			real dist2sq = distsq(pVertRogue->pos,pTri->cornerptr[c2]->pos);
			if (dist1sq < dist2sq)
			{
				pVertUse = pTri->cornerptr[c1];
			} else {
				pVertUse = pTri->cornerptr[c2];
			};
		} else {
			// a shared point
			// choose the other shared point (for now at least)
			if (pVertRogue == pTri->cornerptr[c1])
			{
				pVertUse = pTri->cornerptr[c2];
			} else {
				pVertUse = pTri->cornerptr[c1];
			};
		};
		if (pVertRogue == q2)
		{
			FullDisconnect(pVertRogue,pTriContain);				
		} else {
			FullDisconnect(pVertRogue,pTriContain);
		};
		
		return 1; // action taken
	} else {
		return 0;
	};

}


bool TriMesh::FindOtherNeigh(Vertex * pVertex1, Vertex * pVertex2, Vertex * pVertNot, Vertex ** ppOtherNeigh)
{
	// found = FindOtherNeigh(pCaret, pNext1, pVertDisco, &pOtherNeigh);

	// pVertex1,2 should have a shared triangle with pVertNot on one side and another
	// vertex on the other, which we are to find.
	// If it is not this case, return false.
	bool thistri;
	int iCorner, i;
	Triangle * pTri;
	long tri_len, izTri[128];

	tri_len = pVertex1->GetTriIndexArray(izTri);

	for (i = 0; i < tri_len; i++)
	{
		pTri = T + izTri[i];
		thistri = false;
		for (iCorner = 0; iCorner < 3; iCorner++)
			if (pTri->cornerptr[iCorner] == pVertex2) thistri = true;
		for (iCorner = 0; iCorner < 3; iCorner++)
			if (pTri->cornerptr[iCorner] == pVertNot) thistri = false;
		if (thistri == true) {
			iCorner = 0;
			while ((pTri->cornerptr[iCorner] == pVertex1) || (pTri->cornerptr[iCorner] == pVertex2)) 
				iCorner++;
			*ppOtherNeigh = pTri->cornerptr[iCorner];
			return true;
		};
	};
	return false;
}

int TriMesh::FullDisconnect(Vertex *pVertDisco, Triangle *pTriContain)
{
	long iRet;
	long iDisconnected = 0;
	do
	{
		iRet = Disconnect(pVertDisco, pTriContain);
		if (iRet >= 0) pVertDisco = X + iRet; 
		// if Disconnect returns a value >=0  then it is the index of 
		// another vertex that needs disconnecting.
		iDisconnected++;
	} while (iRet > 0);
	return iDisconnected;
}

int TriMesh::Disconnect(Vertex *pVertDisco, Triangle *pTriContain)
{
	// pTriContain is the triangle that we will leave in its list when it's disconnected.
	// ================================================================

	int returnvalue = -1;

	if(pVertDisco->pos.x != pVertDisco->pos.x)
	{
		printf("\n\nIndeterminate value for x. Time to stop\n");
		steps_remaining = 0;
		getch();
		return -2;
	}
	//if (pVertDisco->flags == 3) {
	//	// should never happen
	//	printf("tried to disconnect insulator vertex %d \n",pVertDisco-X);
	//	getch();
	//}
	// For flags == 4, we want to worry about changing flags of triangles and vertices.


	// We no longer accept that in general, we can have more than two triangles 
	// that use one edge (pVertDisco,pVertUse). Consequently we do not worry about that.
	// If that happens, it indicates somewhere we allowed the "two tri per edge" rule
	// to be violated, and we should not be allowing that.

	// Therefore we have exactly 2 triangles with both pVertDisco and pVertUse; 
	// we remove these, and the rest we reconfigure to substitute pVertUse for pVertDisco
	// ^^^^^ --- X this is not correct. We go around the circle making new triangles in a spiral.***

	// Note that we can never have 2 points with only 3 neighs adjoining as long as
	// we insist every edge has 0 or 2 triangles.
	// Therefore we can avoid creating orphans by disconnecting all 3-neighbour points to begin with.

	// If we get rid of pVertUse then we switch to another vertex that pVertUse had in common with pVertDisco.
	// How to choose which one? Bear in mind, there must always be 2 - we do not allow complete overlap tri's
	// to come into being. We choose the other corner of pDestroy.

	// Meanwhile, this function is called when one of the 4 points for a shared edge belongs to
	// a triangle itself -- is that for the best? We did decide on that and it seemed to generally find one.
	// Otherwise kill all four?
	// For otherwise it's not clear which to disconnect, as it may not be one of the two unsharing points.
	
	// In case that pVertUse == 0, it's still the case that pVertDisco should belong to exactly 2 wedges.

	Vertex *pNext, *pVert, *pCaret;
	bool Used[1024];  
	Triangle * pNextTri;
	bool VertUseIsClockwise = 0;
	bool VertUseIsAnticlockwise = 0;
	int i,c1;
	Triangle *pTri, *pTriUse;
	smartlong hitlist;
	long iCaret;		
	Vector2 uCaret, uNext2, uNext1, uOther, perp;
	int jj;
	smartlong workingverts;
	smartlong nextverts;
	int tri_index;
//	long unviolated;
	static int passes = 0;
	int foundyet = 0;

	smartlong vert_ind;

	long izTrisDisco[128];
	long tri_len_disco = pVertDisco->GetTriIndexArray(izTrisDisco);

	//smartvp & tris = pVertDisco->triangles;
	//long * izTrisDisco; // pVertDisco->izTri;
	// Hoping to manipulate it with this pointer -- careful.

	int debug_extras = 0;
	// Spit out information used to draw graphs for debugging

	char filename[1024];
	FILE * file;

	// old comment:

	// We USED TO wish to replace pVertDisco with pVertUse in all triangles that have it in.
	// If the triangle already contains pVertUse it is scrapped and added to the 
	// TriangleHeap array.
	// Update: triangles list for neighbouring vertices
	//         triangle::cornerptr obviously
	//         triangle data such as periodic, flags etc
	//         triangle::neighbour lists
	// 
	// Finally we add pVertDisco to the Disconnected array and keep a note that
	// gives some idea of what triangle it used to inhabit (for reconnecting);
	
	// Preliminary 1:
	// Try to establish that neighbouring vertices form 1 circle as a graph.
	// _______________________________________________________
	
	if (tri_len_disco >= 1024) {printf("\n\ntris.len>1024 - something is about to break down\n");getch();};
	for (i = 0; i < tri_len_disco; i++)
		Used[i] = 0;
	int old_tri_len = tri_len_disco; // CAREFUL
		
	// vert_ind is the array of neighbours that are in order that they are linked by triangles.
	// We cannot do anything as easy as just taking angles here. We take the mesh structure as the guide for the circle.
	// Add to vert_ind the two corners from izTrisDisco[0] that are not == pVertDisco:
	
	if (pVertDisco->flags < 4) // Not edge
	{
		
		int first = 1;
		pTri = T + izTrisDisco[0];
		for (jj = 0; jj < 3; jj++)
		{
			if (pTri->cornerptr[jj] != pVertDisco)
			{
				if (first) {
					vert_ind.add(pTri->cornerptr[jj]-X);
					// Set periodic -1,0,+1 wrapping relative to pVertDisco ; is used by Triangle::Set to set triangle periodic flag
					pTri->cornerptr[jj]->iScratch = pTri->InferRelativeWrapping(pTri->cornerptr[jj],pVertDisco); 
					
					first = 0;
				} else {
					pNext = pTri->cornerptr[jj]; // we'll add this one when we find it in another tri.				
				};
			};
		};
		Used[0] = 1;
					
		for (int ii = 0; ii < old_tri_len-1; ii++)
		{
			// look for a match for pNext amongst the remaining triangles of pVertDisco

			int found = 0;
			for (i = 0; i < old_tri_len; i++)
			{
				if (Used[i] == 0)
				{
					// does this tri contain the point pNext?
					pTri = T + izTrisDisco[i];
					if ((pTri->cornerptr[0] == pNext) ||
						(pTri->cornerptr[1] == pNext) ||
						(pTri->cornerptr[2] == pNext) )
					{
						found = 1;
						pNextTri = pTri;
						Used[i] = 1;
					};
				};
			};
			if (found == 0)
			{
				// no pNext ??!?!!
				// => not 1 circular seq of neigh verts. Investigate further
				printf("Oh dear - non-circular sequence of neighbour vertices.\n");
				getch();
			};
			
			// Now found a match for this vertex; add it to the sequence
			vert_ind.add(pNext-X);
			pNext->iScratch = pNextTri->InferRelativeWrapping(pNext,pVertDisco);
			
			// Determine next thing to search for:
			jj = 0;
			while ((jj < 3) && 
				((pNextTri->cornerptr[jj] == pVertDisco) || (pNextTri->cornerptr[jj] == pNext)))
				jj++;
			// Note that we treat 0 as a valid member of the vertex list, that should appear at most once.
			if (jj == 3) { printf("\n3rror!\njj == 3\n"); getch(); }; // check that this is tri is remotely sensible.
			
			pNext = pNextTri->cornerptr[jj]; 
		};
		
		pTri = T + izTrisDisco[0]; // purpose?

		if (
			(pNext != X + vert_ind.ptr[0]) && 
			((pNext != 0) || (vert_ind.ptr[0] != -1)))
		{
			printf("\n3rror!\nCircle of neigh verts did not connect.\n"); getch();
		};
		// we already added that one at the start of the list.

		// Note this means vert_ind.len == tris.len. Verify that it is so:
		if (vert_ind.len != old_tri_len) {
			printf("vert_ind.len != old_tri_len\n"); getch();
		};
	} else {
		// EDGE VERTEX - CIRCLE WILL NOT CONNECT, AS WE INTEND THINGS.
		
		printf("Trying to call Disconnect for an edge vertex.\n");
		getch();

		int foundbase = 0;
		int iFirsttri = 0;
		do {
			if (iFirsttri == old_tri_len) {
				printf("failed to find even 1 base vertex by another.\n"); getch();
				return -23939;
			};
			
			pTri = T + izTrisDisco[iFirsttri]; 
			for (jj = 0; jj < 3; jj++)
			{
				if (pTri->cornerptr[jj] != pVertDisco)
				{
					if (pTri->cornerptr[jj]->flags >= 4) {
						vert_ind.add(pTri->cornerptr[jj]-X);
						// Set periodic -1,0,+1 wrapping relative to pVertDisco ; is used by Triangle::Set to set triangle periodic flag
						pTri->cornerptr[jj]->iScratch = pTri->InferRelativeWrapping(pTri->cornerptr[jj],pVertDisco); 
						foundbase = 1;
					} else {
						pNext = pTri->cornerptr[jj]; // we'll add this one when we find it in another tri.				
					};
				};
			};
			iFirsttri++;
			
		} while (foundbase = 0);
		iFirsttri--;
		Used[iFirsttri] = 1;
					
		for (int ii = 0; ii < old_tri_len-1; ii++) // ii index not used.
		{
			// look repeatedly for a match for pNext amongst the remaining triangles of pVertDisco
			int found = 0;
			for (i = 0; i < old_tri_len; i++)
			{
				if (Used[i] == 0)
				{
					// does this tri contain the point pNext?
					pTri = T+izTrisDisco[i];
					if ((pTri->cornerptr[0] == pNext) ||
						(pTri->cornerptr[1] == pNext) ||
						(pTri->cornerptr[2] == pNext) )
					{
						found = 1;
						pNextTri = pTri;
						Used[i] = 1;
					};
				};
			};

			if (found == 0)
			{
				// no pNext ??!?!! => not 1 circular seq of neigh verts. Investigate further
				printf("Oh dear - non-circular sequence of neighbour vertices (edge vert).\n");
				getch();
			};
			
			// Now found a match for this vertex; add it to the sequence
			vert_ind.add(pNext-X);
			pNext->iScratch = pNextTri->InferRelativeWrapping(pNext,pVertDisco);
			
			// Determine next thing to search for:
			jj = 0;
			while ((jj < 3) && 
				((pNextTri->cornerptr[jj] == pVertDisco) || (pNextTri->cornerptr[jj] == pNext)))
				jj++;
			// Note that we treat 0 as a valid member of the vertex list, that should appear at most once.
			if (jj == 3) { printf("\n3rror!\njj == 3\n"); getch(); }; // check that this is tri is remotely sensible.
			
			pNext = pNextTri->cornerptr[jj]; 
		};
		
		if (vert_ind.len != old_tri_len+1) {
			printf("vert_ind.len != tris.len+1\n"); getch();
		};
	};
	
	// No guarantee that we are in a position to run the standard "anticlockwise sort" routine, things could be in a complete tangle.
	// That is why we had to do the above.
	
	//=============================================================================

	// We used to delete neighs with 3 neighbours. 
	// The point was to create a convex set of remaining neighbours. 
	// But it just doesn't prevent having a non-convex set of neighbours. Having 3-ers probably does almost guarantee the set is not convex,
	// unless other things are also wrong. But that is all we can say.

	//=============================================================================

	// We might as well use the first tris.len-2 triangles for filling the space.
	// tris.len-1 in case of an edge vertex.

	// &&&&&&&&&&&&&&& &&&&&&&&&&&&&&& &&&&&&&&&&&&&&&
	// To scrap triangles, we just want to remove them from Vertex::Triangle lists:
	for (int ii = 0; ii < vert_ind.len; ii++)
	{
		pVert = X + vert_ind.ptr[ii];
		if (pVert != pVertDisco) // do not touch tri list of disconnection point !!!!
		{
			// remove all pVertDisco's triangles from pVert if they are there:
			for (i = 0; i < old_tri_len; i++)
				pVert->RemoveTriIndexIfExists(izTrisDisco[i]);
		};
	};  
	
	// &&&&&&&&&&&&&&& &&&&&&&&&&&&&&& &&&&&&&&&&&&&&&
	// Make triangles to fill the space.

	long iStart;
	long iEnd;
	long iPrevEnd,iNext1,iNext2;
	Vertex * pNext1, * pNext2;
	
	// copy verts_ind over workingverts:
	workingverts.copyfrom(vert_ind);
	tri_index = 0;
	
	// we will always make a triangle from iCaret to iCaret+1, iCaret+2
	// then move iCaret to iCaret+2, until we can do no more triangles;
	// adding the 0th,2nd,4th,etc points to the next circle.
	
	// Should be guaranteed that each edge has only 2 sharing. If we check that it's convex to cut out the intermediate point,
	// well I don't know but I think we should stick to just that! Disconnect more points if it still has a problem!
	// But maybe if we are careful here then we end up not creating "a new overlap" ?

	// Meh. Come back and try adding back that test later if we have problems. For now we just test whether we are cutting 
	// the intermediate point inside. 
	
	// nextverts is the array of uncovered points that will be used to then fill in the next, inner concentric set of tri's.

	Triangle * pTriTest;
	int bad;
	int tris_added;
	int always_onward;
	int num_edge;
	Vertex * pVertex, *pOtherNeigh;
	long iNext;
	bool found;
	int cautious = 1;
	long tri_len, izTri[128];

	bool bRanSet = 0;

	iStart = 0;
	while (workingverts.len >= 3)
	{
		nextverts.clear();
	
		// We do not want the LAST triangle to be faced with a triple of insulator points. === How to avoid? ===				
		// The way it can come about is if the previous triangle linked the first and last base points.
		// Such triangles are bad! We want to block off a base point not the last domain point -- so always
		// start our circle before the end of the base points, if they exist.
	
		num_edge = 0;
		for (i = 0; i < workingverts.len; i++)
		{
			pVertex = X+workingverts.ptr[i];
			if (pVertex->flags >= 4) num_edge++;
		};
		if (num_edge > 2) {
			// pedal backwards until the *next* vertex is an the edge
			do
			{
				iStart--;
				if (iStart < 0) iStart = workingverts.len-1;
				iNext = iStart+1; if (iNext == workingverts.len) iNext = 0;
				pVertex = X+workingverts.ptr[iNext];
			} while (pVertex->flags < 4);
		};
		

		iCaret = iStart;
		
		// Do we always add anything that iCaret passes through? :
		//	nextverts.add(workingverts.ptr[iCaret]);
		// No - because we might find iStart is covered over by final triangle;
		// Instead add points at end of triangle placement, knowing it is not covered over.
		
		iNext1 = iCaret+1;if (iNext1 >= workingverts.len) iNext1 = 0;
		iNext2 = iNext1+1;if (iNext2 >= workingverts.len) iNext2 = 0;
		
		tris_added = 0; // for checking we do not get infinite loop
		
		// inner loop: add triangles around edge of a circle
		//===================================================
		do 
		{

			// test this triangle: is it one that we want to use?
			pCaret = X+workingverts.ptr[iCaret];
			pNext1 = X+workingverts.ptr[iNext1];
			pNext2 = X+workingverts.ptr[iNext2];
			
			if (workingverts.len > 3)
			{	
				// We have to decide whether (pCaret,pNext1,pNext2) is a valid triangle to place.
				// Now because pVertDisco may have been moved somewhere daft,
				// it is no good testing whether pNext1 is same side of pCaret--pNext2 as pVertDisco.
				
				bad = 0;
				// Disallow connecting 3 edge vertices:
				if ((pCaret->flags >= 4) && (pNext1->flags >= 4) && (pNext2->flags >= 4)) 
				{
					bad = 1;
				} else {
					// OK so we have 3 sides we have to test.
					// pCaret-pNext1 exists ; one of its neighs was pVertDisco; we do not know yet which
					// side the other one was on. We propose a new neighbour pNext2.

					found = FindOtherNeigh(pCaret,pNext1, pVertDisco, &pOtherNeigh);
					if (found == 0) {
						// This will happen routinely, because pCaret and pNext1 were probably
						// not together in the initial list of vertices.

						printf("error -- had a side with only one triangle. Shouldna happened?");
						found = found;
						bad = 1;
					};
					
					if (pOtherNeigh == pNext2) {
						// concave shape then probably.
						bad = 1;
					} else {
						// pOtherNeigh same side as pNext2 ?
						// how to make sure we use contiguous image?
						// Make all contiguous using pNext1->has_periodic. pNext1 has all the others as neighbours.
						uCaret=pCaret->pos;
						uNext2=pNext2->pos;
						uNext1=pNext1->pos;
						uOther=pOtherNeigh->pos;
						if (pNext1->has_periodic) {
							if (uCaret.x > 0.0) uCaret = Anticlockwise*uCaret;
							if (uNext2.x > 0.0) uNext2 = Anticlockwise*uNext2;
							if (uNext1.x > 0.0) uNext1 = Anticlockwise*uNext1;
							if (uOther.x > 0.0) uOther = Anticlockwise*uOther;
						};
						perp.x = uNext1.y-uCaret.y; perp.y = uCaret.x-uNext1.x;
						if ((uNext2-uNext1).dot(perp) * (uOther-uNext1).dot(perp) > 0.0){
							if (cautious) {
								bad = 1;
							} else {
								returnvalue = pOtherNeigh-X;// disconnect that index next.
							};
						};
					};

					// pNext1-pNext2 exists. We propose a new neighbour pCaret.
					found = FindOtherNeigh(pNext1,pNext2, pVertDisco, &pOtherNeigh);
					if (found == 0) {
						printf("error -- had a side with only one triangle. Shouldna happened?");
						found = found;
						bad = 1;
					};
					if (pOtherNeigh==pCaret) {
						bad = 1;
					} else {
						// Again contiguous image is based on pNext1.
						uCaret=pCaret->pos;
						uNext2=pNext2->pos;
						uNext1=pNext1->pos;
						uOther=pOtherNeigh->pos;
						if (pNext1->has_periodic) {
							if (uCaret.x > 0.0) uCaret = Anticlockwise*uCaret;
							if (uNext2.x > 0.0) uNext2 = Anticlockwise*uNext2;
							if (uNext1.x > 0.0) uNext1 = Anticlockwise*uNext1;
							if (uOther.x > 0.0) uOther = Anticlockwise*uOther;
						};
						perp.x = uNext1.y-uNext2.y; perp.y = uNext2.x-uNext1.x;
						if ((uCaret-uNext1).dot(perp) * (uOther-uNext1).dot(perp) > 0.0){
							if (cautious) {
								bad = 1;
							} else {
								returnvalue = pOtherNeigh-X; // disconnect that index next.
							};
						};
					};
					
					// pCaret-pNext2 may not already exist.  
					// If it does then we'd be creating a 3x shared edge because it's guaranteed that two triangles
					// have pCaret<->pNext2 as an edge, and neither of them was one that we are deleting.
					// (Most of the time if it does exist, it just means that there is a concave part to the shape here.)
					// Creating 3x shared edge is not a problem that will be fixable with more mesh maintenance so we do not do that under any circumstances.

					tri_len = pCaret->GetTriIndexArray(izTri);
					for (int iiii = 0; iiii < tri_len; iiii++)
					{
						pTriTest = T+izTri[iiii];
						for (int jj = 0; jj < 3; jj++)
							if (pTriTest->cornerptr[jj] == pNext2)
								bad = 1;
					};

				}; // whether a base triple
			} else {
				// exactly 3 points - ie only triangle in final circle
				bad = 0; // have no choice to accept 3 points
				
				// Note that since these points were not linked in original circle they certainly can already be
				// connected elsewhere, even though in this circle they are adjacent ('linked')
				// If that happens, it could be that a different choice of triangulation might have avoided the problem
				// We are making a choice in using nearby neighbours and starting from a particular point.
				// Maybe we do need to reinstitute deleting any 3-neighbour neighbours to begin with.

				if ((pCaret->flags >= 4) && (pNext1->flags >= 4) && (pNext2->flags >= 4)) 
				{
					printf("Disconnect failed - left with a triple of edge points left over.\n");
					bad = 1;
					iStart = iStart;
				};
			};
			
			always_onward = 0;			
			if (iCaret == iStart)
			{
				// place iEnd at the place we should stop testing triangles
				if (bad == 0)
				{
					iEnd = iStart; // stop when we reach the point before, which would reach forward to one no longer existent, or at this point.
					nextverts.add(workingverts.ptr[iStart]); // make sure though that we do not add it twice
				} else {
					// make sure we do test for tri at (prevstart, (start), start+1), given modulo of 2
					iEnd = iStart+1; if (iEnd >= workingverts.len) iEnd = 0;
					always_onward = 1; // make sure it doesn't stop before it gets past the beginning.
					// If we did not put a triangle at start, iPrevEnd will be iStart and so we stop when we get back to iStart or beyond.
				};				
				iPrevEnd = iEnd-1; if (iPrevEnd < 0) iPrevEnd = workingverts.len-1; 
				// We will stop if we reach iCaret == iPrevEnd or iEnd.

				// Every point should be either added to nextverts or killed out of the set.
			};

			if (bad == 0)
			{
				pTriUse = T+izTrisDisco[tri_index];
				//pTriUse = (Triangle *)(tris.ptr[tri_index]);
				tri_index++;
				SetTriangle(pTriUse,pCaret,pNext1,pNext2); // will use iScratch to set pTriUse->periodic and vertex flags to set pTriUse->flags

				// Trouble with this:
				// we expect triangle arrays to be anticlockwise sorted.
				// * *** *** *** *** *** *** *** *** *** *** *** *** *** **
				bRanSet = 1; // to remind to fix later

				pCaret = pNext2;
				iCaret = iNext2;
				iNext1+=2;if (iNext1 >= workingverts.len) iNext1 -= workingverts.len;
				iNext2+=2;if (iNext2 >= workingverts.len) iNext2 -= workingverts.len;

				tris_added++;
			} else {
				// Advance 1.
				
				pCaret = pNext1;
				iCaret = iNext1;
				iNext1 = iNext2;
				pNext1 = pNext2;
				iNext2++;
				if (iNext2 >= workingverts.len) iNext2 -= workingverts.len;			
				// what on earth the point of advancing only 2 out of 3 pointers.
			};

			// iCaret is now the next point that is not covered over by a triangle ;
			// add the iCaret vertex, before we test it for a forward-looking triangle next time:
			if (iCaret == iStart) {
				nextverts.add_unique(workingverts.ptr[iCaret]); // we may have 2 triangles meeting at iStart; add it only once
			} else {
				if (iCaret != iEnd) 
					nextverts.add(workingverts.ptr[iCaret]); // if we just moved iEnd to iStart, we do add it the first time.
			};

		} while (((iCaret != iPrevEnd) && (iCaret != iEnd)) || (always_onward == 1));
		// We have tested triangles starting from iPrevEnd, or, have a triangle at iEnd
		
		if (tris_added == 0) {
			// failed circle
			if (cautious) {
				printf("warning -- disconnect creating more shared edges.  ");
				cautious = 0;
				// go around again and set returnvalue to index of vertex we might wish to disconnect.
			} else {
				printf("Disconnect failed. No way to connect decent circle. iVertDisco %d \n",pVertDisco-X);
				getch();
				iCaret = iCaret;
			};
		} else {
			// succeeded circle
			workingverts.copyfrom(nextverts);					
			// can't see a point in moving iStart unless it is > workingverts.len now. oh well
			iStart++;
			if (iStart >= workingverts.len) iStart = 0;
		};
	};

	if (pVertDisco->flags < 4)
	{
		if (tri_index != vert_ind.len-2) {
			printf("tri_index error %d %d \n",tri_index,vert_ind.len);
			getch();
		};
	} else {
		if (tri_index != vert_ind.len-1) {
			printf("tri_index error edge %d %d \n",tri_index,vert_ind.len);
			getch();
		};
	};
	
	// Now go through and set transvec
	for (i = 0; i < tri_index; i++)
	{
		pTriUse = T+izTrisDisco[i];
		// Important to do periodic before RecalculateUnnormalisedEdgeNormalVectors.
		// Flags and periodic were set in the Triangle::Set routine.
		pTriUse->RecalculateEdgeNormalVectors(false);
		GlobalAffectedTriIndexList.add(pTriUse-T); // remember list of those that altered(?)
	};
	
	// Now reset Triangle::neighbours for all tri's of the affected vertices.
	
	for (i = 0; i < vert_ind.len; i++)
	{
		pVert = X + vert_ind.ptr[i];
		// each of your triangles needs to reset its neighbours:
		
		tri_len = pVert->GetTriIndexArray(izTri);
		for (int ii = 0; ii < tri_len; ii++)
		{
			pTri = T+izTri[ii];
			ResetTriangleNeighbours(pTri);
		};
	};

	//
	// If an edge vertex being disconnected, will have 1 triangle left over not 2.
	// 

	// Last two triangles get put on scrapheap
	//=========================================

	// Now punch a hole in the remaining 2 triangles to mark them as scrapped, and
	// list our point as disconnected:
	
	//pTriUse = (Triangle *)(tris.ptr[tri_index]);
	pTriUse = T+izTrisDisco[tri_index];
	pTriUse->cornerptr[0] = 0;
	TriangleHeap.add(pTriUse-T);
	GlobalAffectedTriIndexList.add(pTriUse-T); // remember list of those that altered(?)
	tri_index++;
	pTriUse = T + izTrisDisco[tri_index];
	// Scrapped the idea of shuffling triangles past numTriangles. Have fixed total number.
	pTriUse->cornerptr[0] = 0;
	TriangleHeap.add(pTriUse-T);
	GlobalAffectedTriIndexList.add(pTriUse-T); // remember list of those that altered(?)

	long index = (long)(pVertDisco-X); 
	Disconnected.add(index);
	// Note we did not change flag on pVertDisco -- it can remember it is a base point if it gets reconnected in situ.
	
	// store some information in pVertDisco: what triangle should be used to look for it?
	//pVertDisco->triangles.clear();
	//pVertDisco->triangles.add(pTriContain);
	pVertDisco->ClearTris();
	pVertDisco->AddTriIndex(pTriContain-T);

	// What about its neighbours? Will they be cleared on reconnect?
	// Exmaine it.
	// WHAT ABOUT THE OPINION OF NEIGHBOURS ON WHETHER THIS ONE IS A NEIGHBOUR.
	// PROBLEMOS.
	
	workingverts.clear();
	nextverts.clear(); // just in case going out of scope left a memory leak.

	printf("Done disconnect %d \n",index);

#ifndef RELEASE
	//
	//sprintf(filename,"endII%d.txt",passes);
	//file = fopen(filename,"w");
	//spit_out_point_triangles(pVertUse,file);	
	//spit_out_point_triangles(pVertDefault,file);
	//fclose(file);
	//
	//DebugTestForLinkedScrapTris();
	//DebugTestForVertexInOnlyTwoTriangles();
	//DebugTestWrongNumberTrisPerEdge();
	//DebugTestNumberOfWedgeRings();
#endif

	passes++;
	if (bRanSet) {
		printf("unordered tri lists! Need to fix program! vertex neighs were unset also\n"
			"we run RVNOVO but intermediately what happened?"
			"and does RVNOVO know to look for disconnected vertex?");
		getch();
		RefreshVertexNeighboursOfVerticesOrdered();
	}

	return returnvalue;
}



void TriMesh::SetTriangle(Triangle * pTri, Vertex * pV1, Vertex * pV2, Vertex * pV3)
{
	// set the cornerptr to these values and tell vertices to add this triangle also.

	// Set Triangle::flags based on Vertex::flags
	// and Triangle::periodic based on Vertex::iScratch which records relative wrapping apparently.

	pTri->cornerptr[0] = pV1;
	pTri->cornerptr[1] = pV2;
	pTri->cornerptr[2] = pV3;
	pV1->AddTriIndex(pTri-T);
	pV2->AddTriIndex(pTri-T);
	pV3->AddTriIndex(pTri-T);
	
	pTri->periodic = pTri->cornerptr[0]->iScratch
		+ pTri->cornerptr[1]->iScratch + pTri->cornerptr[2]->iScratch;
	while (pTri->periodic >= 3) pTri->periodic -= 3;
	while (pTri->periodic < 0) pTri->periodic += 3;

}

Triangle * TriMesh::SetAuxTri(int iLevel, long iVertex1, long iVertex2, long iVertex3)
{
	Triangle * pTri = AuxT[iLevel] + numAuxTriangles[iLevel];
	long index = numAuxTriangles[iLevel];
	pTri->cornerptr[0] = AuxX[iLevel]+iVertex1;
	pTri->cornerptr[1] = AuxX[iLevel]+iVertex2;
	pTri->cornerptr[2] = AuxX[iLevel]+iVertex3;

	// Now what?
	pTri->cornerptr[0]->AddTriIndex(index);
	pTri->cornerptr[1]->AddTriIndex(index);
	pTri->cornerptr[2]->AddTriIndex(index);

	numAuxTriangles[iLevel]++;
	if (numAuxTriangles[iLevel] >= numTrianglesAuxAllocated[iLevel])
	{
		printf("got to end of allocated aux tris!"); getch();
	};
	return pTri;
}

void TriMesh::ReconnectLastPointInDiscoArray()
{
	// We want to reconnect the vertex indexed by Disconnected.ptr[0]
	// We use two triangles from TriangleHeap.
	// The vertex includes one thing in its triangles array, that gives us the place to seek its location.

	// To update: Triangle::cornerptr (obviously)
	//            Vertex::triangles
	//            Triangle::transvec,periodic,flag
	//            Triangle::neighbours

	real grad,gradleft,gradmid,gradright;
	int i,ii,iWhich;
	int additional, valueone, valuetwo;
	long lendisc = Disconnected.len;
	Vertex * pVert = X + Disconnected.ptr[lendisc-1];
	Vertex * pVertexTemp, *pVertPrev;

	long tri_len, izTri[128];
	tri_len = pVert->GetTriIndexArray(izTri);
	long izTriTemp[128];

	Triangle * pSeedTri = T+izTri[0];
	// stored record of where to look for it
	
	Triangle * pTri, *pTri2, *pTriCopy, *pTriNeigh;
	int BaseFlag;
	real grad1, grad2;
	int found,c1,c2,iApex,use;
	Vector2 u[3];

	if ((pSeedTri->u8domain_flag == INNER_FRILL) || 
		(pSeedTri->u8domain_flag == OUTER_FRILL))
	{
		pSeedTri = pSeedTri->neighbours[0];
	}

	// Get hold of two spare triangles:
	long lentris = TriangleHeap.len;
	Triangle * pTri_extra1 = T+TriangleHeap.ptr[lentris-1];
	Triangle * pTri_extra2 = T+TriangleHeap.ptr[lentris-2];
	if (lentris < 2) {
		printf("fatal error, lentris < 2\n"); getch(); return;
	}

	if (pVert->flags < 3) { // MAGIC NUMBERS 
	
		// Check something:
		if (pSeedTri->cornerptr[0] == 0)
		{
			// The nearby triangle got scrapped unfortunately, so restart from triangle 0:
			pSeedTri = T;
			while (pSeedTri->cornerptr[0] == 0)
				pSeedTri++;
		};
		// Following function should work fine because overlaps have been eliminated before we call Reconnect:
		pTri = ReturnPointerToTriangleContainingPoint(
					pSeedTri, pVert->pos.x, pVert->pos.y);
	
		// pTri now is what contains pVert.

		// We subdivide it into 3 triangles and pVert becomes the new cornerptr[2]

		// For periodic we see if pTri was periodic.
		// If it is, we have to decide which points pVert is on same side as.
		if (pTri->periodic > 0)
		{
			if (pTri->periodic == 1)
			{
				// one point is to Clockwise. Which one?
				int c1 = pTri->GetLeftmostIndex();
				int c3 = pTri->GetRightmostIndex();
				c2 = 0; while ((c2 == c1) || (c2 == c3)) c2++;
				//int c2 = pTri->GetxMidIndex();
				// Now we want to know where pVert lies in this.
				
				// Right of leftmost unwrapped is one case;
				// Left of wrapped point is another;
				// Otherwise something has failed.

				grad = pVert->pos.x/pVert->pos.y;
				gradleft = pTri->cornerptr[c1]->pos.x/pTri->cornerptr[c1]->pos.y;
				gradmid = pTri->cornerptr[c2]->pos.x/pTri->cornerptr[c2]->pos.y;

				if (grad > gradmid)
				{
					// In this case 2 triangles have periodic == 1
					// and 1 triangle, formed without c1, has periodic == 0				
					additional = 0;
				} else {
					if (grad < gradleft)
					{
						// In this case 2 triangles have periodic == 2
						// and 1 triangle, formed without c1, has periodic == 1					
						additional = 1;
					} else {
						printf("\n\nReconnection point not in its triangle. (periodic issue)\n");
						getch();
					};
				};
				if (c1 == 0)
				{
					pTri_extra1->periodic = 1 + additional;
					pTri_extra2->periodic = additional;
					pTri->periodic = 1 + additional;
				};
				if (c1 == 1)
				{
					pTri_extra1->periodic = additional;
					pTri_extra2->periodic = 1 + additional;
					pTri->periodic = 1 + additional;
				};
				if (c1 == 2)
				{
					pTri_extra1->periodic = 1 + additional;
					pTri_extra2->periodic = 1 + additional;
					pTri->periodic = additional;
				};
			} else {
				// one point is to Anticlockwise. Which one?
				int c1 = pTri->GetRightmostIndex();
				int c3 = pTri->GetLeftmostIndex();
				c2 = 0; while ((c2 == c1) || (c2 == c3)) c2++;

				grad = pVert->pos.x/pVert->pos.y;
				gradright = pTri->cornerptr[c1]->pos.x/pTri->cornerptr[c1]->pos.y;
				gradmid = pTri->cornerptr[c2]->pos.x/pTri->cornerptr[c2]->pos.y;

				// Now we want to know where pVert lies in this.
				
				// Left of rightmost unwrapped is one case;
				// Right of wrapped point is another;
				// Otherwise something has failed.
				if (grad < gradmid)
				{
					// In this case 2 triangles have periodic == 2
					// and 1 triangle, formed without c1, has periodic == 0
					valuetwo = 2;
					valueone = 0;
				} else {
					if (grad > gradright)
					{
						// In this case 2 triangles have periodic == 1
						// and 1 triangle, formed without c1, has periodic == 2
						valuetwo = 1;
						valueone = 2;
					} else {
						printf("(anticlock) Reconnection point not in its triangle.\n");
						getch();
					};
				};				
				if (c1 == 0)
				{
					pTri_extra1->periodic = valuetwo;
					pTri_extra2->periodic = valueone;
					pTri->periodic = valuetwo;
				};
				if (c1 == 1)
				{
					pTri_extra1->periodic = valueone;
					pTri_extra2->periodic = valuetwo;
					pTri->periodic = valuetwo;
				};
				if (c1 == 2)
				{
					pTri_extra1->periodic = valuetwo;
					pTri_extra2->periodic = valuetwo;
					pTri->periodic = valueone;
				};
			};
		} else {
			// pTri not periodic at all :
			pTri_extra1->periodic = 0;
			pTri_extra2->periodic = 0;
		};

		// Update Triangle::flags ..
		// u8domain_flag
		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
			pTri_extra1->u8domain_flag = DOMAIN_TRIANGLE;
			pTri_extra2->u8domain_flag = DOMAIN_TRIANGLE;
		} else {
			if (pTri->u8domain_flag == OUT_OF_DOMAIN) {
				pTri_extra1->u8domain_flag = OUT_OF_DOMAIN;
				pTri_extra2->u8domain_flag = OUT_OF_DOMAIN;
			} else {
				// CROSSING_INS
				
				// New pTri will go between cornerptr 0 and 1 and the new vertex
				// pTri_extra1 will have cornerptr 0 and 2
				// pTri_extra2 will have cornerptr 1 and 2
				pTri_extra1->u8domain_flag = CROSSING_INS;
				if ((pTri->cornerptr[0]->flags == DOMAIN_VERTEX) &&
					(pTri->cornerptr[2]->flags == DOMAIN_VERTEX))
					pTri_extra1->u8domain_flag = DOMAIN_TRIANGLE;
				pTri_extra2->u8domain_flag = CROSSING_INS;
				if ((pTri->cornerptr[1]->flags == DOMAIN_VERTEX) &&
					(pTri->cornerptr[2]->flags == DOMAIN_VERTEX))
					pTri_extra2->u8domain_flag = DOMAIN_TRIANGLE;
				pTri->u8domain_flag = CROSSING_INS;
				if ((pTri->cornerptr[0]->flags == DOMAIN_VERTEX) &&
					(pTri->cornerptr[1]->flags == DOMAIN_VERTEX))
					pTri->u8domain_flag = DOMAIN_TRIANGLE;
			};
		};


		pTri->cornerptr[2]->RemoveTriIndexIfExists(pTri-T);
		pTri->cornerptr[2]->AddTriIndex(pTri_extra1-T);
		pTri->cornerptr[2]->AddTriIndex(pTri_extra2-T);

		pTri->cornerptr[0]->AddTriIndex(pTri_extra1-T);
		pTri->cornerptr[1]->AddTriIndex(pTri_extra2-T);

		// Now pVert:
		pVert->ClearTris();
		pVert->AddTriIndex(pTri-T);
		pVert->AddTriIndex(pTri_extra1-T);
		pVert->AddTriIndex(pTri_extra2-T);

		// Now (finally) change verts:

		pTri_extra1->cornerptr[0] = pTri->cornerptr[0];
		pTri_extra1->cornerptr[1] = pVert;
		pTri_extra1->cornerptr[2] = pTri->cornerptr[2]; 

		pTri_extra2->cornerptr[0] = pVert;
		pTri_extra2->cornerptr[1] = pTri->cornerptr[1];
		pTri_extra2->cornerptr[2] = pTri->cornerptr[2]; 

		pTri->cornerptr[2] = pVert;

		// Triangle::edge_normal :

		pTri->RecalculateEdgeNormalVectors(false);
		pTri_extra1->RecalculateEdgeNormalVectors(false);
		pTri_extra2->RecalculateEdgeNormalVectors(false);

		// Finally Triangle::neighbours (requires Vertex::triangles) :
		// First reset neighbours of our 3 triangles:
		ResetTriangleNeighbours(pTri);
		ResetTriangleNeighbours(pTri_extra1);
		ResetTriangleNeighbours(pTri_extra2);	

		// Now look at each of those neighbours and reset THEIR neighbour lists also:

		if (pVert->GetTriIndexArray(izTriTemp) != 3)
		{
			printf("shucks! TriMesh::ReconnectLastPointInDiscoArray  tris.len != 3 !\n");
			getch();
		};

	} else {

		// %%%%%%%%%%%%%%%%%%%%%%%%
		// reconnecting edge vertex
		// %%%%%%%%%%%%%%%%%%%%%%%%

		// Chances are this NEVER happens.
		// Edges of memory should remain inviolate.
		// But be careful for tri-based!!

		// Actually it seems I do not know what behaviour I want in this case.
		// We can identify azimuthally which triangle is split up by introducing
		// this edge vertex. Then what? We are supposed to use two triangles
		// by putting it in. 
		// It probably created 2 extra triangles when disconnected!

		// comment out whole thing for now
		printf("Trying to reconnect edge vertex but code has been commented out.\n");
		getch();


		////if (pVert->flags == 3) {BaseFlag = 6;} else {BaseFlag = 24;};
		//grad = pVert->x/pVert->y;
		//// Find base triangle where it is azimuthally within corners

		//if ( (pSeedTri->flags == pVert->flags) && (pSeedTri->cornerptr[0] != 0) )
		//{
		//	pTri = pSeedTri;
		//} else {
		//	pTri = T;
		//	while (pTri->flags != pVert->flags) pTri++;
		//};

		//pVertPrev = pTri->cornerptr[0];
		//if (pTri->cornerptr[0]->flags != pVert->flags) pVertPrev = pTri->cornerptr[1];
		//
		//found = 0;
		//do {
		//	//1. get corners c1,c2 on edge:
		//	if (pTri->cornerptr[0]->flags != pVert->flags) {
		//		c1 = 1; c2 = 2; iApex = 0;
		//	};
		//	if (pTri->cornerptr[1]->flags != pVert->flags) {
		//		c1 = 0; c2 = 2; iApex = 1;
		//	};
		//	if (pTri->cornerptr[2]->flags != pVert->flags) {
		//		c1 = 0; c2 = 1; iApex = 2;
		//	};
		//	if (pTri->periodic == 0) {
		//		grad1= pTri->cornerptr[c1]->x/pTri->cornerptr[c1]->y;
		//		grad2 = pTri->cornerptr[c2]->x/pTri->cornerptr[c2]->y;				
		//	} else {
		//		if (pVert->x > 0.0) {
		//			pTri->MapRight(u[0],u[1],u[2]);
		//		} else {
		//			pTri->MapLeft(u[0],u[1],u[2]);
		//		};
		//		grad1 = u[c1].x/u[c1].y;
		//		grad2 = u[c2].x/u[c2].y;
		//	};
		//	if ((grad-grad1)*(grad-grad2) <= 0.0) // quick fix .... is == enough?
		//	{
		//		found = 1;
		//	} else {
		//		if (pTri->cornerptr[c1] == pVertPrev)
		//		{	use = c2;	} else { use = c1 ;};
		//		pVertPrev = pTri->cornerptr[use];
		//		// Now find next base triangle: belongs to new pVertPrev but is not this tri
		//		iWhich = -1;
		//		for (i = 0; i < pVertPrev->triangles.len; i++)
		//		{
		//			pTri2 = (Triangle *)(pVertPrev->triangles.ptr[i]);
		//			if ((pTri2 != pTri) && (pTri2->flags == pTri->flags))
		//				iWhich = i;
		//		};				
		//		if (iWhich == -1) {	printf("error iWhich == -1\n"); getch();	};
		//		pTri = (Triangle *)(pVertPrev->triangles.ptr[iWhich]);
		//	};
		//}	while (found == 0);

		//// Now we seek to place pVert on the edge of pTri

		//// Subdivide it into 2 triangles. pTri_extra1 becomes one;
		//// pTri_extra2 will be consigned to spare triangles.

		//// seek numEdgeVerts++ to see same code elsewhere.
		//pTriCopy = T+numTriangles-1;
		//memcpy(pTri_extra2,pTriCopy,sizeof(Triangle));
		//
		//// 2. all the other triangles and vertices that looked at pTriCopy have to now be looking at its new location.
		//for (i = 0; i < 3; i++)
		//{
		//	pTriNeigh = pTri->neighbours[i];
		//	for (int ii = 0; ii < 3; ii++)
		//	{
		//		if (pTriNeigh->neighbours[ii] == pTriCopy) pTriNeigh->neighbours[ii] = pTri_extra2;
		//	};
		//};
		//for (i = 0; i < 3; i++)
		//{
		//	pVertexTemp = pTriCopy->cornerptr[i]; // pTri or pTriCopy doesn't matter - copied over cornerptr already.
		//	for (int ii = 0; ii < pVertexTemp->triangles.len; ii++)
		//	{
		//		pTri2 = (Triangle *)(pVertexTemp->triangles.ptr[ii]);
		//		if (pTri2 == pTriCopy) pVertexTemp->triangles.ptr[ii] = pTri_extra2;
		//	};
		//};
		//
		//// assign spare triangle index and change counts:
		////pVert->iTriSpare = numTriangles-1;		
		//// INDEX NEVER USED FOR ANYTHING - JUST STACK
		//this->numTriangles--;
		//this->numEdgeVerts++;
		//

		//// To rewrite so that we don't have spares but secret tris underneath, would be
		//// possible but I'm not sure there would be a point.
		//
		//if (pTri->periodic > 0) {
		//	if (pVert->x > 0.0) { // unwrapped
		//		if (pTri->cornerptr[iApex]->x > 0.0) // unwrapped
		//		{
		//			pTri->periodic = 0;
		//			pTri_extra1->periodic = 1;
		//		} else {
		//			pTri->periodic = 1;
		//			pTri_extra1->periodic = 2;
		//		};
		//	} else {
		//		if (pTri->cornerptr[iApex]->x > 0.0) // unwrapped
		//		{
		//			pTri->periodic = 1;
		//			pTri_extra1->periodic = 2;
		//		} else {
		//			pTri->periodic = 2;
		//			pTri_extra1->periodic = 0;
		//		};
		//	};
		//} else {
		//	// pTri was not periodic:
		//	pTri_extra1->periodic = 0;
		//};
		// 
		//pTri_extra1->flags = pVert->flags;
		//pTri->flags = pVert->flags; 
		//// no others change since their cornerptrs do not change.

		//// Deal with Vertex::triangle
		//// CHANGE c2 TO NEW VERTEX
		//pTri->cornerptr[c2]->triangles.remove(pTri);
		//pTri->cornerptr[c2]->triangles.add(pTri_extra1);
		//
		//pTri->cornerptr[iApex]->triangles.add(pTri_extra1);
		//
		//pVert->triangles.clear();
		//pVert->triangles.add(pTri);
		//pVert->triangles.add(pTri_extra1);
		//
		//// Now Triangle::cornerptr

		//pTri_extra1->cornerptr[0] = pTri->cornerptr[iApex];
		//pTri_extra1->cornerptr[1] = pVert;
		//pTri_extra1->cornerptr[2] = pTri->cornerptr[c2]; 

		//pTri->cornerptr[c2] = pVert;

		//// Triangle::edge_normal :

		//pTri->RecalculateEdgeNormalVectors(false);
		//pTri_extra1->RecalculateEdgeNormalVectors(false);
		//
		//// Finally Triangle::neighbours (requires Vertex::triangles) :
		//ResetTriangleNeighbours(pTri);
		//ResetTriangleNeighbours(pTri_extra1);
		//
		//// Now look at each of those neighbours and reset THEIR neighbour lists also:

		//if (pVert->triangles.len != 2)
		//{
		//	printf("shucks! TriMesh::ReconnectLastPointInDiscoArray  tris.len !=2 !\n");
		//	getch();
		//};
	};

	
	tri_len = pVert->GetTriIndexArray(izTri);
	for (i = 0; i < tri_len; i++) // tris.len should equal 3 !!!!!!
	{
		pTri = T+izTri[i];
					
		ResetTriangleNeighbours(pTri->neighbours[0]);
		ResetTriangleNeighbours(pTri->neighbours[1]);
		ResetTriangleNeighbours(pTri->neighbours[2]);
	};
	
	// Remove triangles from scrapheap:
	// --------------------------------------
	Disconnected.remove(Disconnected.ptr[lendisc-1]);
	TriangleHeap.remove(pTri_extra1-T);
	TriangleHeap.remove(pTri_extra2-T);

	printf("reconnection tests");
	//DebugTestForVertexInOnlyTwoTriangles();
	DebugTestWrongNumberTrisPerEdge();

	// So then it IS missing any 
	// neighbours -> clear
	// neighbours -> add

	// Is it in RVNOVO?

	// In OUR version, we need vertex neighbours to be maintained.

	// ** That applies to Delaunay flips also.**

	// ??
}


bool TriMesh::DebugTestForOverlaps()
{
	Triangle * pTri,* pNeigh;
	Vertex * pOpp;
	long iTri;
	int iNeigh, iWhich, iprev;
	Vector2 u_1[3], u_2[3], diff_ours, diff_ther;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		for (iNeigh = 0; iNeigh<3; iNeigh++)
		{
			pNeigh = pTri->neighbours[iNeigh];
			if (pNeigh != pTri) {
				
				pTri->MapLeftIfNecessary(u_1[0],u_1[1],u_1[2]);
				pNeigh->MapLeftIfNecessary(u_2[0],u_2[1],u_2[2]);
						
				if ((pTri->periodic == 0) && (pNeigh->periodic > 0)
					&& (pTri->cornerptr[0]->pos.x > 0.0) )
				{
					u_2[0] = Clockwise*u_2[0];
					u_2[1] = Clockwise*u_2[1];
					u_2[2] = Clockwise*u_2[2];
				};
				if ((pNeigh->periodic == 0) && (pTri->periodic > 0)
				   && (pNeigh->cornerptr[0]->pos.x > 0.0))
				{
					u_2[0] = Anticlockwise*u_2[0];
					u_2[1] = Anticlockwise*u_2[1];
					u_2[2] = Anticlockwise*u_2[2];
				};

				pOpp = pNeigh->ReturnUnsharedVertex(pTri, &iWhich);
				// difference from a shared vertex:
				iprev = iNeigh-1; if (iprev == -1) iprev = 2;
				diff_ther = u_2[iWhich]-u_1[iprev];
				diff_ours = u_1[iNeigh]-u_1[iprev];
			
				if ( (diff_ours.dot(pTri->edge_normal[iNeigh]))
					*(diff_ther.dot(pTri->edge_normal[iNeigh])) > 0.0)
				{
					// same sign; error
					printf("overlap created by skips; should not be possible.\niTri %d ",iTri);
					iTri = iTri;
					getch();
					return true;
				};
			};

		};
	};
	return false;
}


void TriMesh::Redelaunerize(bool exhaustion, bool bReplace)
{
	long iTri2;
	Triangle * pTri2, * pTri;
	Vertex * pVertq;
	Vector2 cc;
	real pdistsq;
	real qdistx,qdisty;//,pdistx,pdisty;
	long flips;
	int side;
	Vertex tempV;
	Vector2 nearest;
	int c1,c2;
	int iprev, inext;
	Vertex * pVertex1, * pVertex2, *pUnsh;
	ShardData shard_data1, shard_data2;
	Vector2 corner1,corner2,centre,projected, diff1, diff2;
	bool to_flip,perflag;
	Triangle newtri;

	static real const REL_TOLERANCE = 1.0e-11; 
	// 5e-13 does not seem to avoid back-and-forward flips due to rounding.

	long flip_tri_to_tri = 0;
	long highflip_tri_to_wedge = 0;
	long highflip_wedge_to_tri = 0;
	long lowflip_tri_to_wedge = 0;
	long lowflip_wedge_to_tri = 0;

	static real const COS60 = 0.5;

	// whereas cos 90 = 0

	printf("start of redelaunerize");
	DebugTestWrongNumberTrisPerEdge();

	printf(" D ");

	long totalflips = 0;
	// if exhaustion == true, carry on to exhaustion; otherwise do 1 pass.
	do
	{
		flips = 0;
		pTri = T;
		for (long iTri = 0; iTri < numTriangles; ++iTri)
		{

			// Do not play at outer edge of memory: (fluid replace is not designed to work there)

			if ((pTri->cornerptr[0]->flags == OUTERMOST) ||
				(pTri->cornerptr[1]->flags == OUTERMOST) ||
				(pTri->cornerptr[2]->flags == OUTERMOST) ) 
			{
				// do nothing
			} else {

				// calculate circumcenter first....
				pTri->CalculateCircumcenter(cc, &pdistsq);
					
				for (int iNeigh = 0; iNeigh < 3; iNeigh++)
				{
					pTri2 = pTri->neighbours[iNeigh];
					
					if (	(pTri2 != pTri) 
						&& ((pTri2->u8domain_flag != OUT_OF_DOMAIN) || (pTri->u8domain_flag != OUT_OF_DOMAIN)) 
						&& (pTri2->cornerptr[0]->flags != OUTERMOST)
						&& (pTri2->cornerptr[1]->flags != OUTERMOST)
						&& (pTri2->cornerptr[2]->flags != OUTERMOST)
						)
					{
						pVertq = pTri2->ReturnUnsharedVertex(pTri);
							// we compare two triangles
							
						// need to consider another case:
						// one of the neighbours is across PBC
						// in which case we should map q to same side as p

						real qdistsq = GetPossiblyPeriodicDistSq(pVertq->pos,cc); // less elegant but should still work.

						if (qdistsq < pdistsq-pdistsq*REL_TOLERANCE)
						{
							++flips;
							++flip_tri_to_tri;

							if (bReplace) {
								// pTri2 is neighbours[iNeigh]
								// so pTri->cornerptr[iNeigh] is unshared.
								
								// Get shared vertices:
								inext = iNeigh+1; if (inext == 3) inext = 0;
								iprev = inext+1; if (iprev == 3) iprev = 0;
								pVertex1 = pTri->cornerptr[inext];
								pVertex2 = pTri->cornerptr[iprev];
								// These will be the giving vertcells. IF domain.
								if (pVertex1->flags == DOMAIN_VERTEX)
									CreateShards(pVertex1,shard_data1);  // seems to be assuming an interior vertex.
								if (pVertex2->flags == DOMAIN_VERTEX)
									CreateShards(pVertex2,shard_data2);
								pUnsh = pTri->cornerptr[iNeigh];
								
								// store indices (in shard array) of
								// corners where cell of pVertex1 meets pUnsh,pVertq...
								
								// That means finding common tri that is not
								// one of the flipping ones.
								
								// There could be confusion though if only 4 tris on vertex
								// - maybe it's better not to try to be clever but to just
								// intersect every shard with both destinations.
							}
							
							Flip(pTri, pTri2,-1); 
							
							if (bReplace) {
								// take from pVertex1 whatever we send to pUnsh:
								if ((pUnsh->flags == DOMAIN_VERTEX) && (pVertex1->flags == DOMAIN_VERTEX))
									GiveAndTake(shard_data1,pUnsh,pVertex1);
								if ((pVertq->flags == DOMAIN_VERTEX) && (pVertex1->flags == DOMAIN_VERTEX))
									GiveAndTake(shard_data1,pVertq,pVertex1);
								if ((pUnsh->flags == DOMAIN_VERTEX) && (pVertex2->flags == DOMAIN_VERTEX))
									GiveAndTake(shard_data2,pUnsh,pVertex2);
								if ((pVertq->flags == DOMAIN_VERTEX) && (pVertex2->flags == DOMAIN_VERTEX))
									GiveAndTake(shard_data2,pVertq,pVertex2);
							};
							
							iNeigh = 4; // Skip out of loop.
						};
					}; // whether an edge worth looking through, vs, going off the domain
				}; // next neighbour
			};
			++pTri;
		};

		printf("Flips: %d \n", flips);
	// Vertex::flags does not need to be changed as a result of this routine.
//#ifndef RELEASE
		if (flips > 0) DebugTestWrongNumberTrisPerEdge();
//#endif
		totalflips += flips;
	} while ((exhaustion == true) && (flips > 0));

	printf("ended Redelaunerize. totalflips: %d \n",totalflips);
	
	// Checked that
	// 1. vertex neighs do not need to be maintained during Delaunay routines.
	// 2. Flip does actually maintain lists, but not order of tris or vertices.
	// Therefore, we maintain afterwards:
	if (totalflips > 0) {
		this->RefreshVertexNeighboursOfVerticesOrdered();
	}
}

void TriMesh::GiveAndTake(ShardData & shard_data, Vertex * pVDest,Vertex * pVSrc)
{
	// ASSUME TRI CENTROIDS SET

	// intersect shards with pVDest, take away results from pVDest.
	// Should not get zero.
	// No motion going on when this is called, to cause heating/cooling - only flips.

	// Can assume shard_data has same rotation as pVSrc -- so iff we have to rotate
	// to intersect with pVDest, then rotate back when we subtract.

	int i,inext;
	bool bIntersected;
	ConvexPolygon cptri,cptri2, cpIntersection, cpDest;
	long izTri[128], tri_len;
	Triangle * pTri;
	fluid_nvT vals0,vals1,vals2;
	fluid_NvT integrals;


	// First create ConvexPolygon for pVDest. We have to streamline this...
	tri_len = pVDest->GetTriIndexArray(izTri);
	cpDest.Clear();
	for (i = 0; i < tri_len; i++)
	{
		pTri = T + izTri[i];
		cpDest.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVDest));
	};

	for (i = 0; i < shard_data.len; i++)
	{
		inext = i+1; if (inext == shard_data.len) inext = 0;
		cptri.Clear();
		cptri.add(shard_data.central);
		cptri.add(shard_data.cp.coord[i]);
		cptri.add(shard_data.cp.coord[inext]);

		bIntersected = cpDest.GetIntersectionWithTriangle(
			&cpIntersection, cptri.coord[0],cptri.coord[1],cptri.coord[2]);
		
		if (bIntersected) {
			
			cpIntersection.Integrate_Planes(
							cptri.coord[0],cptri.coord[1],cptri.coord[2],
							(real *)(&(shard_data.cdata)), // pass object as array of reals
							(real *)(&(shard_data.fluidnvT[i])),
							(real *)(&(shard_data.fluidnvT[inext])),
							(real *)(&integrals),// output - this is of form {N,NT,Nv} x 3.
							15);
			// Give, and take:
			pVDest->Neut.mass += integrals.N[0];
			pVSrc->Neut.mass -= integrals.N[0];
			pVDest->Neut.heat += integrals.NT[0];
			pVSrc->Neut.heat -= integrals.NT[0];
			pVDest->Neut.mom += integrals.Nv[0];
			pVSrc->Neut.mom -= integrals.Nv[0];
			pVDest->Ion.mass += integrals.N[1];
			pVSrc->Ion.mass -= integrals.N[1];
			pVDest->Ion.heat += integrals.NT[1];
			pVSrc->Ion.heat -= integrals.NT[1];
			pVDest->Ion.mom += integrals.Nv[1];
			pVSrc->Ion.mom -= integrals.Nv[1];
			pVDest->Elec.mass += integrals.N[2];
			pVSrc->Elec.mass -= integrals.N[2];
			pVDest->Elec.heat += integrals.NT[2];
			pVSrc->Elec.heat -= integrals.NT[2];
			pVDest->Elec.mom += integrals.Nv[2];
			pVSrc->Elec.mom -= integrals.Nv[2];

		} else {

			// Let's be careful: we could be heading either way to another tranche.

			if ((pVDest->pos.x > 0.0) && (shard_data.central.x < 0.0))
			{
				// Make sure whether rotating shard_data to right might hit it:
				cptri2.CreateClockwiseImage(cptri);

				bIntersected = cpDest.GetIntersectionWithTriangle(
					&cpIntersection, cptri2.coord[0],
					cptri2.coord[1],
					cptri2.coord[2]);
				if (bIntersected) {

					vals0 = shard_data.cdata.Clockwise();
					vals1 = shard_data.fluidnvT[i].Clockwise();
					vals2 = shard_data.fluidnvT[inext].Clockwise();

					cpIntersection.Integrate_Planes(
							cptri2.coord[0],
							cptri2.coord[1],
							cptri2.coord[2],
							(real *)(&vals0),
							(real *)(&vals1),
							(real *)(&vals2),
							(real *)(&integrals),// output
							15);
					// Give, and take:
					pVDest->Neut.mass += integrals.N[0];
					pVSrc->Neut.mass -= integrals.N[0];
					pVDest->Neut.heat += integrals.NT[0];
					pVSrc->Neut.heat -= integrals.NT[0];
					pVDest->Neut.mom += integrals.Nv[0];
					pVSrc->Neut.mom -= Anticlockwise3*integrals.Nv[0];
					pVDest->Ion.mass += integrals.N[1];
					pVSrc->Ion.mass -= integrals.N[1];
					pVDest->Ion.heat += integrals.NT[1];
					pVSrc->Ion.heat -= integrals.NT[1];
					pVDest->Ion.mom += integrals.Nv[1];
					pVSrc->Ion.mom -= Anticlockwise3*integrals.Nv[1];
					pVDest->Elec.mass += integrals.N[2];
					pVSrc->Elec.mass -= integrals.N[2];
					pVDest->Elec.heat += integrals.NT[2];
					pVSrc->Elec.heat -= integrals.NT[2];
					pVDest->Elec.mom += integrals.Nv[2];
					pVSrc->Elec.mom -= Anticlockwise3*integrals.Nv[2];
					
				};
				
			};
			if ((pVDest->pos.x < 0.0) && (shard_data.central.x > 0.0))
			{
				// Try rotating shard_data anticlockwise
				cptri2.CreateAnticlockwiseImage(cptri);
				
				bIntersected = cpDest.GetIntersectionWithTriangle(
					&cpIntersection, cptri2.coord[0],
					cptri2.coord[1],
					cptri2.coord[2]);
				if (bIntersected) {

					vals0 = shard_data.cdata.Anticlockwise();
					vals1 = shard_data.fluidnvT[i].Anticlockwise();
					vals2 = shard_data.fluidnvT[inext].Anticlockwise();

					cpIntersection.Integrate_Planes(
							cptri2.coord[0],cptri2.coord[1],cptri2.coord[2],
							(real *)(&vals0),
							(real *)(&vals1),
							(real *)(&vals2),
							(real *)(&integrals),// output
							15);

					// Give, and take:
					pVDest->Neut.mass += integrals.N[0];
					pVSrc->Neut.mass -= integrals.N[0];
					pVDest->Neut.heat += integrals.NT[0];
					pVSrc->Neut.heat -= integrals.NT[0];
					pVDest->Neut.mom += integrals.Nv[0];
					pVSrc->Neut.mom -= Clockwise3*integrals.Nv[0];
					pVDest->Ion.mass += integrals.N[1];
					pVSrc->Ion.mass -= integrals.N[1];
					pVDest->Ion.heat += integrals.NT[1];
					pVSrc->Ion.heat -= integrals.NT[1];
					pVDest->Ion.mom += integrals.Nv[1];
					pVSrc->Ion.mom -= Clockwise3*integrals.Nv[1];
					pVDest->Elec.mass += integrals.N[2];
					pVSrc->Elec.mass -= integrals.N[2];
					pVDest->Elec.heat += integrals.NT[2];
					pVSrc->Elec.heat -= integrals.NT[2];
					pVDest->Elec.mom += integrals.Nv[2];
					pVSrc->Elec.mom -= Clockwise3*integrals.Nv[2];
				};
			};
		};		
	};
}

void TriMesh::CreateShards(Vertex * pVertex, ShardData & shard_data)
{
	long izTri[128];
	Vector2 u[3];
	ShardData temp;
	fluid_nvT fluids[3];
	int tri_len, i;
	Triangle * pTri;
	int parity[3];
	real beta[3];
	bool found;
	real coeffremain;

	// ASSUMES TRI CENTROIDS POPULATED
	// DOES NOT ASSUME VERTEX CENTROIDS POPULATED
	// ASSUMES VERTEX AREACELL POPULATED

	// Maybe we should chalk up what things to actually MAINTAIN.

	// ****&&&&****&&&&****&&&&****&&&&****&&&&****&&&&****&&&&****&&&&****

	// Vertex centroid should be maintained, and removed from this routine.



	
	shard_data.cp.Clear();

	// 1A. Add tri centroids to shard_data.cp
	tri_len = pVertex->GetTriIndexArray(izTri);
	shard_data.len = tri_len;

	for (i = 0; i < tri_len; i++)
	{
		pTri = T + izTri[i];

		if ((pTri->periodic) && (pVertex->pos.x > 0.0))
		{
			shard_data.cp.add(Clockwise*pTri->cent);
		} else {
			shard_data.cp.add(pTri->cent);
		};
		
		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
			// set these first...
			// To set desired values at pTri's centroid (WHICH IS ASSUMED POPULATED)
			// need to ensure vertex centroids are populated:
			for (int iCorner = 0; iCorner < 3; iCorner++)
				RecalculateCentroid(pTri->cornerptr[iCorner]);

			u[0] = pTri->cornerptr[0]->centroid;
			u[1] = pTri->cornerptr[1]->centroid;
			u[2] = pTri->cornerptr[2]->centroid;
			if (pTri->periodic == 0) {
			} else {
				pTri->GetParity(parity); // 1 = clockwise side
				if (parity[0]) u[0] = Anticlockwise*u[0];
				if (parity[1]) u[1] = Anticlockwise*u[1];
				if (parity[2]) u[2] = Anticlockwise*u[2];			
			};		

			for (int iCorner = 0; iCorner < 3; iCorner++)
			{
				fluids[iCorner].n[0] = pTri->cornerptr[iCorner]->Neut.mass/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nv[0] = pTri->cornerptr[iCorner]->Neut.mom/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nT[0] = pTri->cornerptr[iCorner]->Neut.heat/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].n[1] = pTri->cornerptr[iCorner]->Ion.mass/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nv[1] = pTri->cornerptr[iCorner]->Ion.mom/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nT[1] = pTri->cornerptr[iCorner]->Ion.heat/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].n[2] = pTri->cornerptr[iCorner]->Elec.mass/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nv[2] = pTri->cornerptr[iCorner]->Elec.mom/pTri->cornerptr[iCorner]->AreaCell;
				fluids[iCorner].nT[2] = pTri->cornerptr[iCorner]->Elec.heat/pTri->cornerptr[iCorner]->AreaCell;
			};
			
			if (pTri->periodic == 0) {
				// do nothing to it.
			} else {
				// Have to rotate some momentum vectors to supply desired value contig to pTri,
				// as well as rotating positions u.				
				pTri->GetParity(parity); // 1 = clockwise side				
				for (int iCorner = 0; iCorner < 3; iCorner++)
				{
					if (parity[iCorner]) {
						fluids[iCorner].nv[0] = Anticlockwise3*fluids[iCorner].nv[0];
						fluids[iCorner].nv[1] = Anticlockwise3*fluids[iCorner].nv[1];
						fluids[iCorner].nv[2] = Anticlockwise3*fluids[iCorner].nv[2];
					};
				};
			};		
			GetInterpolationCoefficients(beta,pTri->cent.x,pTri->cent.y,u[0],u[1],u[2]);
			temp.fluidnvT[i].Interpolate(beta,fluids);
			
			// This made it contiguous for pTri.
			
		} else {
			// Not domain tri: do nothing; go again below:
		}
	};

	// 1B. Create desired values n,nT,nv at each CROSSING_INS tri centroid:
	long index[128];
	Triangle * pDomainTri;
	int numIntermed = 0;
	int iWhich, iNeigh, ii;
	real dist1, dist2, wt1, wt2, wtsum;
	int index_edge1 = -1, index_edge2 = -1;
	for (i = 0; i < tri_len; i++)
	{
		pTri = T + izTri[i];

		// To set desired values at pTri's centroid (WHICH IS ASSUMED POPULATED)
		// need to ensure vertex centroids are populated:
		if (pTri->u8domain_flag != DOMAIN_TRIANGLE) {
			if (pTri->u8domain_flag == CROSSING_INS) {

				// does it have a neighbour in the domain: if so, find it in the existing list of values.
				// It will be either i+1 (anticlock from here) or i-1 (clockwise from here).
				
				// Go again and set any that are left over; make a list of them here.
				
				if ((pTri->neighbours[0]->u8domain_flag != DOMAIN_TRIANGLE)
					&&
					(pTri->neighbours[1]->u8domain_flag != DOMAIN_TRIANGLE)
					&&
					(pTri->neighbours[2]->u8domain_flag != DOMAIN_TRIANGLE))
				{
					index[numIntermed] = i;
					memset(&(temp.fluidnvT[i]),0,sizeof(fluid_nvT)); // avoid crash later
					numIntermed++;
				} else {
					// keep note of index:
					if (index_edge1 == -1) {
						index_edge1 = i;
					} else {
						index_edge2 = i;
					};

					// Find domain neighbour in our list:
	
					iWhich = 0;
					while (pTri->neighbours[iWhich]->u8domain_flag != DOMAIN_TRIANGLE) iWhich++;
					iNeigh = pTri->neighbours[iWhich]-T;
					ii = 0; 
					while ((izTri[ii] != iNeigh) && (ii < tri_len)) ii++;
					if (ii == tri_len) {
						printf("error348231\n");
						getch();
					};
					
					// Now use two shared corners and the centre of this domain neighbour to create a plane;
					// the plane infers us the desirable value of n, nT, nv at our pTri centroid.
					// We can use Interpolate of course.

					pDomainTri = T + iNeigh;
					
					// First do exactly as before ... some of the data may be nonsense but it will be overwritten.
					u[0] = pTri->cornerptr[0]->centroid;
					u[1] = pTri->cornerptr[1]->centroid;
					u[2] = pTri->cornerptr[2]->centroid;
					if (pTri->periodic == 0) {
					} else {
						pTri->GetParity(parity); // 1 = clockwise side
						if (parity[0]) u[0] = Anticlockwise*u[0];
						if (parity[1]) u[1] = Anticlockwise*u[1];
						if (parity[2]) u[2] = Anticlockwise*u[2];			
					};
					for (int iCorner = 0; iCorner < 3; iCorner++)
					{
						fluids[iCorner].n[0] = pTri->cornerptr[iCorner]->Neut.mass/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nv[0] = pTri->cornerptr[iCorner]->Neut.mom/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nT[0] = pTri->cornerptr[iCorner]->Neut.heat/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].n[1] = pTri->cornerptr[iCorner]->Ion.mass/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nv[1] = pTri->cornerptr[iCorner]->Ion.mom/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nT[1] = pTri->cornerptr[iCorner]->Ion.heat/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].n[2] = pTri->cornerptr[iCorner]->Elec.mass/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nv[2] = pTri->cornerptr[iCorner]->Elec.mom/pTri->cornerptr[iCorner]->AreaCell;
						fluids[iCorner].nT[2] = pTri->cornerptr[iCorner]->Elec.heat/pTri->cornerptr[iCorner]->AreaCell;
					};					
					if (pTri->periodic != 0) 
					{
						// Have to rotate some momentum vectors to supply desired value contig to pTri,
						// as well as rotating positions u.				
						pTri->GetParity(parity); // 1 = clockwise side				
						for (int iCorner = 0; iCorner < 3; iCorner++)
						{
							if (parity[iCorner]) {
								fluids[iCorner].nv[0] = Anticlockwise3*fluids[iCorner].nv[0];
								fluids[iCorner].nv[1] = Anticlockwise3*fluids[iCorner].nv[1];
								fluids[iCorner].nv[2] = Anticlockwise3*fluids[iCorner].nv[2];
							};
						};
					};		

					// Now overwrite the bottom corner of this tri with the centre of the tri above it.
					u[iWhich] = pDomainTri->cent;
					fluids[iWhich] = temp.fluidnvT[ii];
					// These should already be contiguous for pTri, right? Actually might not be.
					if ( ((pDomainTri->periodic != 0) && (pTri->periodic == 0))
						||
						 ((pDomainTri->periodic == 0) && (pTri->periodic != 0)) )
					{
						printf("error ... not coded for this.\n");
						getch();
					}					
					GetInterpolationCoefficients(beta,pTri->cent.x,pTri->cent.y,u[0],u[1],u[2]);
					temp.fluidnvT[i].Interpolate(beta,fluids);										
				};				
			} else {
				printf("error: pTri->u8domain_flag = %d \n",pTri->u8domain_flag);
				getch();
			};
		};
	};

	for (i = 0; i < tri_len; i++)
	{
		pTri = T + izTri[i];
		// BETTER TO DO THIS LAST. MAKE EACH TRI'S fluidnvT contiguous TO ITS OWN TRI FIRST.
		// NOW MAKE IT CONTIGUOUS TO pVertex
		if ((pTri->periodic) && (pVertex->pos.x > 0.0))
		{
			temp.fluidnvT[i].nv[0] = Clockwise3*temp.fluidnvT[i].nv[0];
			temp.fluidnvT[i].nv[1] = Clockwise3*temp.fluidnvT[i].nv[1];
			temp.fluidnvT[i].nv[2] = Clockwise3*temp.fluidnvT[i].nv[2];
		};
	};
	
	// Now all are contiguous to pVertex, finally:
	if (numIntermed > 0) {
		// Fill in intermediate values along the insulator by just interpolating left to right between
		// the guard triangles index_edge1, index_edge2 :

		for (i = 0; i < numIntermed; i++)
		{
			iWhich = index[i];
			pTri = T + izTri[iWhich];
			dist1 = GetPossiblyPeriodicDist(shard_data.cp.coord[i],shard_data.cp.coord[index_edge1]);
			dist2 = GetPossiblyPeriodicDist(shard_data.cp.coord[i],shard_data.cp.coord[index_edge2]);
			wt1 = 1.0/dist1;
			wt2 = 1.0/dist2;
			wtsum = wt1+wt2;
			beta[0] = wt1/wtsum;
			beta[1] = wt2/wtsum;
			beta[2] = 0.0;
			fluids[0] = temp.fluidnvT[index_edge1];
			fluids[1] = temp.fluidnvT[index_edge2];
			fluids[2] = temp.fluidnvT[index_edge1]; // doesn't matter what
			temp.fluidnvT[i].Interpolate(beta,fluids);
		};
	};
	// Hopefully that's it.

	
	// ===
	// Is minmod even beneficial much? Why is it better than setting the desired edge values
	// and then just setting the centre, generally.
	// Think it's beneficial if there is a cliff...

	// MINMOD IS GOOD.

	// This has to be done for each of 15 surfaces (??)
	// Pretty much.

	//shard_data.Minmod(temp);

	real y[128], result[128];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].n[0];// temp.fluidnvT are the desired values.
	shard_data.cp.minmod(result,y,pVertex->Neut.mass,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].n[0] = result[i];
	
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nT[0];
	shard_data.cp.minmod(result,y,pVertex->Neut.heat,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nT[0] = result[i];
	
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[0].x;
	shard_data.cp.minmod(result,y,pVertex->Neut.mom.x,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[0].x = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[0].y;
	shard_data.cp.minmod(result,y,pVertex->Neut.mom.y,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[0].y = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[0].z;
	shard_data.cp.minmod(result,y,pVertex->Neut.mom.z,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[0].z = result[i];

		for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].n[1];// temp.fluidnvT are the desired values.
	shard_data.cp.minmod(result,y,pVertex->Ion.mass,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].n[1] = result[i];
	
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nT[1];
	shard_data.cp.minmod(result,y,pVertex->Ion.heat,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nT[1] = result[i];
	
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[1].x;
	shard_data.cp.minmod(result,y,pVertex->Ion.mom.x,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[1].x = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[1].y;
	shard_data.cp.minmod(result,y,pVertex->Ion.mom.y,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[1].y = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[1].z;
	shard_data.cp.minmod(result,y,pVertex->Ion.mom.z,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[1].z = result[i];

	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].n[2];// temp.fluidnvT are the desired values.
	shard_data.cp.minmod(result,y,pVertex->Elec.mass,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].n[2] = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nT[2];
	shard_data.cp.minmod(result,y,pVertex->Elec.heat,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nT[2] = result[i];
	
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[2].x;
	shard_data.cp.minmod(result,y,pVertex->Elec.mom.x,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[2].x = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[2].y;
	shard_data.cp.minmod(result,y,pVertex->Elec.mom.y,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[2].y = result[i];
	for (i = 0; i < tri_len; i++)
		y[i] = temp.fluidnvT[i].nv[2].z;
	shard_data.cp.minmod(result,y,pVertex->Elec.mom.z,pVertex->pos);
	for (i = 0; i < tri_len; i++)
		shard_data.fluidnvT[i].nv[2].z = result[i];

	// Decided I don't like the way round that data is being stored...
	
	// Write minmod routine; change around afterwards.

}

real ConvexPolygon::minmod(real n[], // output array
					  real ndesire[], real N, 
					  Vector2 central )
{

	// 2. Decide whether we can attain these values and get N,NT,Nv by setting
	// n,nT,nv at vertex to be between highest and lowest corner desired value.
	
	// If not, do not move a lower corner than n_avg, even lower, to attain a lower mass
	// than we can get by putting the centre on a par with the lowest.
	// instead, attain the desired low values then see how much mass then there is available to
	// push somewhat up to the high corners, in sequence.

	// Note: if n_avg < all desired corners, that is a sign we have to default to say 
	// constant-in-cell.
	real coeff[128];
	bool fixed[128];
	ConvexPolygon cptri;
	real tri_area, coeffcent, N0, n_C,n_acceptable,N_attained, coeffremain;
	real low_n,high_n;
	int i,inext;
	bool found;
		

	low_n = ndesire[0];
	high_n = ndesire[0];
	i = 0;
	while (i < numCoords) {
		if (low_n > ndesire[i]) low_n = ndesire[i];
		if (high_n < ndesire[i]) high_n = ndesire[i];
		++i;
	};
	
	real area = GetArea();
	real n_avg = N/area;
	if ((n_avg > high_n) || (n_avg < low_n)){
		// above/below all of them: minmod says give up and set constant
		for (i = 0;i < numCoords; i++)
			n[i] = n_avg;
		return n_avg; // n_C 
	};
	
	// Now see if we can set n_avg to a value that achieves ndesire and N.
	// work up a coefficient on n_C as well as what ndesire is giving us.

	// We assign to each corner a coefficient to make life easier.
	memset(coeff,0,sizeof(real)*128);
	coeffcent = 0.0;
	N0 = 0.0;
	for (i = 0; i < numCoords; i++)
	{
		inext = i+1; if (inext == numCoords) inext = 0;
		
		cptri.Clear();
		cptri.add(coord[i]);
		cptri.add(coord[inext]);
		cptri.add(central);
		tri_area = cptri.GetArea();
		
		N0 += tri_area*THIRD*(ndesire[i]+ndesire[inext]);
		coeff[i] += tri_area*THIRD;
		coeff[inext] += tri_area*THIRD;
		coeffcent += tri_area*THIRD;
	};
	
	real n_C_need = (N-N0)/coeffcent;
	if ((n_C_need > low_n) && (n_C_need < high_n))
	{
		// accept:
		for (i = 0;i < numCoords; i++)
			n[i] = ndesire[i];
		return n_C_need;  // hopefully this is frequently the case.
	};
	
	memset(fixed,0,sizeof(bool)*128);
			
	if (n_C_need < low_n) {
		// the mass is low. So for those less than n_avg let's fix
		// them in place, and fix n_C = low_n.
		// Then we'll see how high we can go.		
		n_C = low_n;
		n_acceptable = (N - coeffcent*n_C )/(area - THIRD*area);
		// area-THIRD*area = sum of other coeffs, and of course
		// coeffcent = THIRD*area
		// n_acceptable > N/area since N=area*n_avg > area*low_n.
		
		// We accept things that are less than this 'max average', and
		// let that increase the threshold; go again until
		// the time we do not find any new lower items ;				
		do {
			found = 0;	
			coeffremain = 0.0;
			N_attained = coeffcent*low_n;
			for (i = 0; i < numCoords; i++)
			{
				if (fixed[i] == 0) {
					if (ndesire[i] < n_acceptable) {
						// yes, use ndesire[i] ...
						fixed[i] = true;
						n[i] = ndesire[i];
						N_attained += n[i]*coeff[i];
						found = true;
					} else {
						coeffremain += coeff[i];
					};
				} else {
					N_attained += n[i]*coeff[i];
				};
			};
			if (found != 0) {
				n_acceptable = (N - N_attained)/coeffremain;
				// The value to which we have to set the remaining
				// n values.
			};						
		} while (found != 0);
		// Now we should set the remaining values to n_acceptable
		// which is less than ndesire[i] in all those cases.
		for (i = 0; i < numCoords; i++)
		{
			if (fixed[i] == 0) n[i] = n_acceptable;
		};
		return n_C;
		
	} else {
		n_C = high_n;
		n_acceptable = (N - coeffcent*n_C)/(area - THIRD*area);
		do {
			found = 0;	
			coeffremain = 0.0;
			N_attained = coeffcent*high_n;
			for (i = 0; i < numCoords; i++)
			{
				if (fixed[i] == 0) {
					if (ndesire[i] > n_acceptable) {
						// yes, use ndesire[i] ...
						fixed[i] = true;
						n[i] = ndesire[i];
						N_attained += n[i]*coeff[i];
						found = true;
					} else {
						coeffremain += coeff[i];
					};
				} else {
					N_attained += n[i]*coeff[i];
				};
			};
			if (found!= 0) {
				n_acceptable = (N - N_attained)/coeffremain;
			};						
		} while (found != 0);
		
		for (i = 0; i < numCoords; i++)
		{
			if (fixed[i] == 0) n[i] = n_acceptable;
		};
		return n_C;
	};
}

/*void TriMesh::Flip(AuxTriangle *pTri1, AuxTriangle * pTri2, int iLevel)
{

	int iClockwise, iAnticlockwise;
	AuxVertex * pVertex_unshared1, * pVertex_unshared2;
	int which, other_index_1, other_index_2;

	pVertex_unshared1 = pTri1->ReturnUnsharedVertex(pTri2, &which);
	pVertex_unshared2 = pTri2->ReturnUnsharedVertex(pTri1);
		
	if (which == 0) {
		other_index_1 = 1;
		other_index_2 = 2;
	} else {
		if (which == 1) {
			other_index_1 = 0; 
			other_index_2 = 2;
		} else {
			other_index_1 = 0;
			other_index_2 = 1;
		};
	};


	// Use iIndicator then wipe it clean.


	// First we set up periodic "scratch" data on the 4 vertices:
	if (pTri1->periodic == 0)
	{
		if (pTri2->periodic == 0)
		{
			pVertex_unshared1->iIndicator = 0;
			pVertex_unshared2->iIndicator = 0;
			pTri1->cornerptr[other_index_2]->iIndicator = 0;
			pTri1->cornerptr[other_index_1]->iIndicator = 0;
		} else {
			if (pTri2->periodic == 1)
			{
				// Should be the unshared point across PBC
				iClockwise = pTri2->GetLeftmostIndex();
				if (pTri2->cornerptr[iClockwise] != pVertex_unshared2)
				{
					// not possible
					printf("\n\n\nerror periodic :1: \n\n"); getch();
					pTri2 = pTri2;
				};
				pVertex_unshared1->iIndicator = 0;
				pVertex_unshared2->iIndicator = 1;
				pTri1->cornerptr[other_index_2]->iIndicator = 0;
				pTri1->cornerptr[other_index_1]->iIndicator = 0;
			} else {
				
				// pTri2->periodic == 2

				// Should be the two shared points across PBC
				iAnticlockwise = pTri2->GetRightmostIndex();
				if (pTri2->cornerptr[iAnticlockwise] != pVertex_unshared2)
				{
					// not possible
					printf("\n\n\nerror periodic\n\n"); getch();
					pTri2 = pTri2;
				};
				pVertex_unshared1->iIndicator = 1;
				pVertex_unshared2->iIndicator = 0;
				pTri1->cornerptr[other_index_2]->iIndicator = 1;
				pTri1->cornerptr[other_index_1]->iIndicator = 1;

			};				
		};
	} else {
		// use tri 1 to define the wrapping initially
		if (pTri1->periodic == 1)
		{
			// which one is across PBC?
			iClockwise = pTri1->GetLeftmostIndex();
			pTri1->cornerptr[0]->iIndicator = (iClockwise == 0)?1:0;
			pTri1->cornerptr[1]->iIndicator = (iClockwise == 1)?1:0;
			pTri1->cornerptr[2]->iIndicator = (iClockwise == 2)?1:0;

			if (pTri1->cornerptr[iClockwise] == pVertex_unshared1)
			{
				// the two shared points are not wrapped
				if (pTri2->periodic == 0) {
					pVertex_unshared2->iIndicator = 0;
				} else {
					if (pTri2->periodic == 1) {
						pVertex_unshared2->iIndicator = 1;
					} else {
						printf("\n\n\nerror periodic J\n\n"); getch();
						pTri2 = pTri2;
					};
				};
			} else {
				// one of the two shared points is wrapped
				if (pTri2->periodic == 0) { 
					printf("\n\n\nerror periodic KK\n\n"); getch();
					pTri2 = pTri2;
				};
				if (pTri2->periodic == 1) {
					// only the shared point was mapped
					pVertex_unshared2->iIndicator = 0;
				} else {
					pVertex_unshared2->iIndicator = 1;
				};
			};
		} else {
			// pTri1->periodic == 2
			iAnticlockwise = pTri1->GetRightmostIndex();
			pTri1->cornerptr[0]->iIndicator = (iAnticlockwise == 0)?0:1;
			pTri1->cornerptr[1]->iIndicator = (iAnticlockwise == 1)?0:1;
			pTri1->cornerptr[2]->iIndicator = (iAnticlockwise == 2)?0:1;
			
			if (pTri1->cornerptr[iAnticlockwise] == pVertex_unshared1)
			{
				// the two shared points are wrapped
				if (pTri2->periodic == 0) {
					pVertex_unshared2->iIndicator = 1;
				} else {
					if (pTri2->periodic == 1) {
						printf("\n\n\nerror periodic kljkl\n\n"); getch();
						pTri2 = pTri2;
					} else {
						pVertex_unshared2->iIndicator = 0;
					};
				};
			} else {
				// one of the two shared points is wrapped relative to the other
				if (pTri2->periodic == 0) {
					printf("\n\n\nerror periodic 223\n\n"); getch();
					pTri2 = pTri2;
				} else {
					if (pTri2->periodic == 1) {
						pVertex_unshared2->iIndicator = 0;
					} else {
						pVertex_unshared2->iIndicator = 1;
					};
				};					
			};
		};
	};


	// Now we need to tell the vertices affected about the triangles that overlap them:
	
	// First do Vertex::triangles
	// ==========================

	// For pVertex_unshared1, we now connect to the triangle of our pair that was not there before
	// For pVertex_unshared2, likewise
	
	pVertex_unshared1->addtri(pTri2-AuxT[iLevel]);
	pVertex_unshared2->addtri(pTri1-AuxT[iLevel]);

	// For old pTri1->vertices[other_index_2] we delete iTri1
	// For old pTri1->vertices[other_index_1] we delete iTri2
	
	pTri1->cornerptr[other_index_2]->remove_tri(pTri1-AuxT[iLevel]);  
	pTri1->cornerptr[other_index_1]->remove_tri(pTri2-AuxT[iLevel]);
	
	// Main assignment: Triangle::vertices,cornerptr
	// =============================================
	
	pTri2->cornerptr[0] = pVertex_unshared1;
	pTri2->cornerptr[1] = pVertex_unshared2;
	pTri2->cornerptr[2] = pTri1->cornerptr[other_index_2];
	
	pTri1->cornerptr[2] = pTri1->cornerptr[other_index_1];
	pTri1->cornerptr[0] = pVertex_unshared1;
	pTri1->cornerptr[1] = pVertex_unshared2;
	
	
	pTri1->periodic = pTri1->cornerptr[0]->iIndicator + 
					  pTri1->cornerptr[1]->iIndicator + 
					  pTri1->cornerptr[2]->iIndicator;
	while (pTri1->periodic >= 3) pTri1->periodic -= 3;
	pTri2->periodic = pTri2->cornerptr[0]->iIndicator + 
					  pTri2->cornerptr[1]->iIndicator + 
					  pTri2->cornerptr[2]->iIndicator;
	while (pTri2->periodic >= 3) pTri2->periodic -= 3;
	pTri1->cornerptr[0]->iIndicator = 0;
	pTri1->cornerptr[1]->iIndicator = 0;
	pTri1->cornerptr[2]->iIndicator = 0;
	pTri2->cornerptr[0]->iIndicator = 0;
	pTri2->cornerptr[1]->iIndicator = 0;
	pTri2->cornerptr[2]->iIndicator = 0;

	// Triangle::neighbours
	// =====================
	
	// six triangles affected in triangle-triangle case.
	// first get hold of them.
	smartvp neighs;
	AuxTriangle * pTri;
	
	neighs.add(pTri1->neighbours[0]);
	neighs.add(pTri1->neighbours[1]);
	neighs.add(pTri1->neighbours[2]);
	neighs.add(pTri2->neighbours[0]);
	neighs.add(pTri2->neighbours[1]);
	neighs.add(pTri2->neighbours[2]); 
	
	for (int i = 0; i < 6; i++)
	{
		pTri = (AuxTriangle *)neighs.ptr[i];				
		pTri->neighbours[2] = ReturnPointerToOtherSharedTriangleAux(pTri->cornerptr[0],pTri->cornerptr[1],pTri,iLevel);
		pTri->neighbours[0] = ReturnPointerToOtherSharedTriangleAux(pTri->cornerptr[1],pTri->cornerptr[2],pTri,iLevel);
		pTri->neighbours[1] = ReturnPointerToOtherSharedTriangleAux(pTri->cornerptr[0],pTri->cornerptr[2],pTri,iLevel);	
	};
	
}


*/

void TriMesh::Flip(Triangle * pTri1, Triangle * pTri2, int iLevel)
{
	// We alter the triangles:
	
	// Of the points they share, allocate one to each in the new setup; the remaining 2 points are now shared:
	// ===============================
	int which, other_index_1,other_index_2;
	int iClockwise,iAnticlockwise;
	int num_ins, num_out;
	int num_in_domain, BaseFlag;
	Triangle *pTri;
	Vertex * pVertex_unshared1, * pVertex_unshared2;

	if (iLevel == -1) {
		GlobalAffectedTriIndexList.add(pTri1-T); 
		GlobalAffectedTriIndexList.add(pTri2-T);

		printf("iTri1 %d iTri2 %d \n",pTri1-T,pTri2-T);
	};

	pVertex_unshared1 = pTri1->ReturnUnsharedVertex(pTri2, &which);
	pVertex_unshared2 = pTri2->ReturnUnsharedVertex(pTri1);
	
	// New tri 1: pVertex_unshared1, pVertex_unshared2, another tri1 point
	// New tri 2: pVertex_unshared1, pVertex_unshared2, remaining tri1 point
	
	if (which == 0) {
		other_index_1 = 1;
		other_index_2 = 2;
	} else {
		if (which == 1) {
			other_index_1 = 0; 
			other_index_2 = 2;
		} else {
			other_index_1 = 0;
			other_index_2 = 1;
		};
	};

	// First we set up periodic "scratch" data on the 4 vertices:
	if (pTri1->periodic == 0)
	{
		if (pTri2->periodic == 0)
		{
			pVertex_unshared1->iScratch = 0;
			pVertex_unshared2->iScratch = 0;
			pTri1->cornerptr[other_index_2]->iScratch = 0;
			pTri1->cornerptr[other_index_1]->iScratch = 0;
		} else {
			if (pTri2->periodic == 1)
			{
				// Should be the unshared point across PBC
				iClockwise = pTri2->GetLeftmostIndex();
				if (pTri2->cornerptr[iClockwise] != pVertex_unshared2)
				{
					// not possible
					printf("\n\n\nerror periodic\n\n"); getch();
					pTri2 = pTri2;
				};
				pVertex_unshared1->iScratch = 0;
				pVertex_unshared2->iScratch = 1;
				pTri1->cornerptr[other_index_2]->iScratch = 0;
				pTri1->cornerptr[other_index_1]->iScratch = 0;
			} else {				
				// pTri2->periodic == 2

				// Should be the two shared points across PBC
				iAnticlockwise = pTri2->GetRightmostIndex();
				if (pTri2->cornerptr[iAnticlockwise] != pVertex_unshared2)
				{
					// not possible
					printf("\n\n\nerror periodic\n\n"); getch();
					pTri2 = pTri2;
				};
				pVertex_unshared1->iScratch = 1;
				pVertex_unshared2->iScratch = 0;
				pTri1->cornerptr[other_index_2]->iScratch = 1;
				pTri1->cornerptr[other_index_1]->iScratch = 1;
			};				
		};
	} else {
		// use tri 1 to define the wrapping initially
		if (pTri1->periodic == 1)
		{
			// which one is across PBC?
			iClockwise = pTri1->GetLeftmostIndex();
			pTri1->cornerptr[0]->iScratch = (iClockwise == 0)?1:0;
			pTri1->cornerptr[1]->iScratch = (iClockwise == 1)?1:0;
			pTri1->cornerptr[2]->iScratch = (iClockwise == 2)?1:0;

			if (pTri1->cornerptr[iClockwise] == pVertex_unshared1)
			{
				// the two shared points are not wrapped
				if (pTri2->periodic == 0) {
					pVertex_unshared2->iScratch = 0;
				} else {
					if (pTri2->periodic == 1) {
						pVertex_unshared2->iScratch = 1;
					} else {
						printf("\n\n\nerror periodic\n\n"); getch();
						pTri2 = pTri2;
					};
				};
			} else {
				// one of the two shared points is wrapped
				if (pTri2->periodic == 0) { 
					printf("\n\n\nerror periodic\n\n"); getch();
					pTri2 = pTri2;
				};
				if (pTri2->periodic == 1) {
					// only the shared point was mapped
					pVertex_unshared2->iScratch = 0;
				} else {
					pVertex_unshared2->iScratch = 1;
				};
			};
		} else {
			// pTri1->periodic == 2
			iAnticlockwise = pTri1->GetRightmostIndex();
			pTri1->cornerptr[0]->iScratch = (iAnticlockwise == 0)?0:1;
			pTri1->cornerptr[1]->iScratch = (iAnticlockwise == 1)?0:1;
			pTri1->cornerptr[2]->iScratch = (iAnticlockwise == 2)?0:1;
			
			if (pTri1->cornerptr[iAnticlockwise] == pVertex_unshared1)
			{
				// the two shared points are wrapped
				if (pTri2->periodic == 0) {
					pVertex_unshared2->iScratch = 1;
				} else {
					if (pTri2->periodic == 1) {
						printf("\n\n\nerror periodic\n\n"); getch();
						pTri2 = pTri2;
					} else {
						pVertex_unshared2->iScratch = 0;
					};
				};
			} else {
				// one of the two shared points is wrapped relative to the other
				if (pTri2->periodic == 0) {
					printf("\n\n\nerror periodic\n\n"); getch();

					//  We get here.



					pTri2 = pTri2;
				} else {
					if (pTri2->periodic == 1) {
						pVertex_unshared2->iScratch = 0;
					} else {
						pVertex_unshared2->iScratch = 1;
					};
				};					
			};
		};
	};


	// Now we need to tell the vertices affected about the triangles that overlap them:
	
	// First do Vertex::triangles
	// ==========================

	// For pVertex_unshared1, we now connect to the triangle of our pair that was not there before
	// For pVertex_unshared2, likewise
	
	if (iLevel == -1) {

		pVertex_unshared1->AddTriIndex(pTri2-T);
		pVertex_unshared2->AddTriIndex(pTri1-T);

		// For old pTri1->vertices[other_index_2] we delete iTri1
		// For old pTri1->vertices[other_index_1] we delete iTri2
	
		pTri1->cornerptr[other_index_2]->RemoveTriIndexIfExists(pTri1-T);  
		pTri1->cornerptr[other_index_1]->RemoveTriIndexIfExists(pTri2-T);
	
		// Vertices also want to know what vertices are nearby:
		// Vertex::neighbours
		// ===============================

		// Why this was commented? Perhaps it spoils something after ???
		// Is it right?

		// The shared ones are no longer mutually connected
		pTri1->cornerptr[other_index_2]->RemoveNeighIndexIfExists(pTri1->cornerptr[other_index_1]-X);
		pTri1->cornerptr[other_index_1]->RemoveNeighIndexIfExists(pTri1->cornerptr[other_index_2]-X);
		// The unshared ones now are connected
		pVertex_unshared1->AddNeighbourIndex(pVertex_unshared2-X);
		pVertex_unshared2->AddNeighbourIndex(pVertex_unshared1-X);
	} else {

		pVertex_unshared1->AddTriIndex(pTri2-AuxT[iLevel]);
		pVertex_unshared2->AddTriIndex(pTri1-AuxT[iLevel]);
		pTri1->cornerptr[other_index_2]->RemoveTriIndexIfExists(pTri1-AuxT[iLevel]);  
		pTri1->cornerptr[other_index_1]->RemoveTriIndexIfExists(pTri2-AuxT[iLevel]);
		
		// We do not play with neighbour lists for auxiliary: these document where coefficients apply.
	};

	// We have to use remove_if_exists because this func is sometimes called when
	// neighbour lists are NOT populated.
	// However, if it is called with them populated, we'd prefer to maintain them
	// -- although the ORDER OF THE VERTEX LIST IS NOT MAINTAINED HERE.
	// And neither is the ORDER OF THE VERTEX TRIANGLE LIST.


	// Main assignment: Triangle::vertices,cornerptr
	// =============================================
	
	//pTri2->vertices[0] = pVertex_unshared1->index;
	pTri2->cornerptr[0] = pVertex_unshared1;
	//pTri2->vertices[1] = pVertex_unshared2->index;
	pTri2->cornerptr[1] = pVertex_unshared2;
	//pTri2->vertices[2] = pTri1->vertices[other_index_2];
	pTri2->cornerptr[2] = pTri1->cornerptr[other_index_2];
	
	//pTri1->vertices[2] = pTri1->vertices[other_index_1];
	pTri1->cornerptr[2] = pTri1->cornerptr[other_index_1];
	//pTri1->vertices[0] = pVertex_unshared1->index;
	pTri1->cornerptr[0] = pVertex_unshared1;
	//pTri1->vertices[1] = pVertex_unshared2->index;
	pTri1->cornerptr[1] = pVertex_unshared2;

	pTri1->periodic = pTri1->cornerptr[0]->iScratch + 
					  pTri1->cornerptr[1]->iScratch + 
					  pTri1->cornerptr[2]->iScratch;
	while (pTri1->periodic >= 3) pTri1->periodic -= 3;
	pTri2->periodic = pTri2->cornerptr[0]->iScratch + 
					  pTri2->cornerptr[1]->iScratch + 
					  pTri2->cornerptr[2]->iScratch;
	while (pTri2->periodic >= 3) pTri2->periodic -= 3;
	
	if (iLevel == -1) {
		// Note that the periodic flag is used in the following:
		pTri1->RecalculateEdgeNormalVectors(false);
		pTri2->RecalculateEdgeNormalVectors(false);
	 // these are not used during Redelaunerize so for aux we can set them after.

		// Set new Triangle::flags:
		// We treat Vertex::flags as valid.

		if ((pTri1->u8domain_flag == 0) && (pTri2->u8domain_flag == 0))
		{
			// do nothing

			// This is the only case that ever actually applies for tri data.
			// For VERTBASED we affect the whole mesh.

		} else {
			if ((pTri1->u8domain_flag == OUT_OF_DOMAIN) && (pTri2->u8domain_flag == OUT_OF_DOMAIN))
			{
				// do nothing
			}  else {

				// This fixed 180116 :
				num_in_domain =	((pTri1->cornerptr[0]->flags == 0)?1:0) + 
								((pTri1->cornerptr[1]->flags == 0)?1:0) + 
								((pTri1->cornerptr[2]->flags == 0)?1:0); // 0 == plasma domain
				// Not counting edge vertices here -- 
				// unlikely that u8domain_flag is differing if near those.
				pTri1->u8domain_flag = CROSSING_INS;
				if (num_in_domain == 3) pTri1->u8domain_flag = DOMAIN_TRIANGLE;
				if (num_in_domain == 0) pTri1->u8domain_flag = OUT_OF_DOMAIN;
				num_in_domain =	((pTri2->cornerptr[0]->flags == 0)?1:0) + 
								((pTri2->cornerptr[1]->flags == 0)?1:0) + 
								((pTri2->cornerptr[2]->flags == 0)?1:0);
				// Not counting edge vertices here -- 
				// unlikely that u8domain_flag is not making it obvious if near those.
				pTri2->u8domain_flag = CROSSING_INS;
				if (num_in_domain == 3) pTri2->u8domain_flag = DOMAIN_TRIANGLE;
				if (num_in_domain == 0) pTri2->u8domain_flag = OUT_OF_DOMAIN;
		
			};
		};
	};

	// Triangle::neighbours: only handles neighbours of neighbours!! But perhaps these included.

	Triangle * neighs[6];

	neighs[0] = pTri1->neighbours[0];
	neighs[1] = pTri1->neighbours[1];
	neighs[2] = pTri1->neighbours[2];
	neighs[3] = pTri2->neighbours[0];
	neighs[4] = pTri2->neighbours[1];
	neighs[5] = pTri2->neighbours[2];
	
	for (int i = 0; i < 6; i++)
	{
		pTri = neighs[i];
		pTri->neighbours[2] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[1],pTri,iLevel);
		pTri->neighbours[0] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[1],pTri->cornerptr[2],pTri,iLevel);
		pTri->neighbours[1] = ReturnPointerToOtherSharedTriangle(pTri->cornerptr[0],pTri->cornerptr[2],pTri,iLevel);
	};
}


void TriMesh::DebugTestWrongNumberTrisPerEdge(void)
{
	Vertex * pVert, *pVert2;
	Triangle * pTri;
	int count,count2;
	smartlong neighs;
	long tri_len, izTri[128];

	for (long i = 0; i < numVertices; i++)
	{
		if (!(Disconnected.contains(i)))
		{
			pVert = X + i;
			// for each thing contained in neighbouring triangles...
			
			neighs.clear();

			tri_len = pVert->GetTriIndexArray(izTri);
			for (int ii = 0; ii < tri_len; ii++)
			{
				pTri = T + izTri[ii];

				for (int jj = 0; jj < 3; jj++)
					if (pTri->cornerptr[jj] != pVert)
						neighs.add_unique(pTri->cornerptr[jj]-X);
			};

			for (int iii = 0; iii < neighs.len; iii++)
			{
				pVert2 = X+neighs.ptr[iii];//MakePtr(neighs,iii);
				// count how many tris have pVert2:

				count = 0;
				for (int ii = 0; ii < tri_len; ii++)
				{
					pTri = T+izTri[ii];
					for (int jj = 0; jj < 3; jj++)
						if (pTri->cornerptr[jj] == pVert2)
							count++;
				};
				if ((count != 2) && (count != 0))
				{
					if (count == 1) {
						// could be this edge is looking off the edge of the domain.
						if ((pVert->flags == pVert2->flags) && (pVert->flags != DOMAIN_VERTEX))
						{
							// give it a pass
						} else {

							printf("Value %d appears %d times at vertex %d. flags %d %d \n",neighs.ptr[iii],count,pVert-X,pVert->flags,pVert2->flags);
							count = count;
							getch();
						};
					} else {
						printf("Value %d appears %d times at vertex %d.\n",neighs.ptr[iii],count,pVert-X);
						count = count;
						getch();
					};
					count = count;
				};
			};
			
			
			// count how many tri's have a 0
			// should be exactly 2 or 0
			// same should hold for every other value.
			/*count = 0;
			count2 = 0;
			for (int ii = 0; ii < pVert->triangles.len; ii++)
			{
				pTri = (Triangle *)pVert->triangles.ptr[ii];
				if (pTri->cornerptr[2] == INS_VERT) count++;
				if (pTri->cornerptr[2] == HIGH_VERT) count2++;
			}
			if ((count != 0) && (count != 2))
			{
				printf("detected %d low wedges at one vertex.\n",count);
				getch();
				count = count;
			};
			if ((count2 != 0) && (count2 != 2))
			{
				printf("detected %d high wedges at one vertex.\n",count);
				getch();
				count = count;
			};
			*/
		};
	};
}


void TriMesh::ShiftVertexPositionsEquanimity()
{
	// Now we want to move each domain interior vertex
	// towards the centre of its polygon.
	// Only move if it is more than 25% of the sq root
	// of polygon area distant from the centre.
	// Do not move neighbours at the same time.

	Vertex * pVertex, *pNeigh, *pNeigh2;
	long iVertex, i, inext;
	ConvexPolygon cp;
	Vector2 u, u2, sum, avg, direction, cand, oldpos;
	real distsq, area, r, to_move, newdist, maxratio, dist, maxdist;
	bool bNo;
	long izNeighs[128];
	long neigh_len;

	real maxrat[3] = {0.38,0.3,0.24};

	for (int iLoop = 0; iLoop < 3; iLoop++)
	{
		maxratio = maxrat[iLoop];
		// make the comb finer over 3 loop repeats.

		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->iIndicator = 0;
			++pVertex;
		};

		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX)
			{
				// check if any neighbours modified already:
				bNo = false;

				neigh_len = pVertex->GetNeighIndexArray(izNeighs);
				for (i = 0; i < neigh_len; i++)
				{
					pNeigh = X + izNeighs[i];
					if (pNeigh->iIndicator == 1) bNo = true;
				}

				if (bNo == false) {

					cp.Clear();
					sum.x = 0.0; sum.y = 0.0;
					for (i = 0; i < neigh_len; i++)
					{
						pNeigh = X + izNeighs[i];
						u.x = pNeigh->pos.x; 
						u.y = pNeigh->pos.y;
						if ((pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y) && (pNeigh->pos.x/pNeigh->pos.y > 0.5*GRADIENT_X_PER_Y))
							u = Anticlockwise*u;

						if ((pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y) && (pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y))
							u = Clockwise*u;
					
						cp.add(u);
						sum += u;
					};
					avg = sum/(real)neigh_len;
					distsq = (avg.x-pVertex->pos.x)*(avg.x-pVertex->pos.x)
						+ (avg.y-pVertex->pos.y)*(avg.y-pVertex->pos.y);
					
					area = cp.GetArea();

					// criterion: if dist > 0.2*sqrt(area)

					if (distsq > maxratio*maxratio*area) {
						dist = sqrt(distsq);
						maxdist = 0.18*sqrt(area);
						newdist = maxdist*dist/(dist + maxdist);
						// -> maxdist as dist -> infinity
						// -> dist as dist << maxdist
						
						to_move = dist-newdist;
						direction.x = avg.x-pVertex->pos.x;
						direction.y = avg.y-pVertex->pos.y;
						r = direction.modulus();
						direction.x /= r;
						direction.y /= r;
												
						cand.x = pVertex->pos.x + direction.x*to_move;
						cand.y = pVertex->pos.y + direction.y*to_move;
												
						// avoid wrapping:
						if ((cand.x/cand.y < -GRADIENT_X_PER_Y) || 
							(cand.x/cand.y > GRADIENT_X_PER_Y))
						{
							// do nothing
						} else {

							oldpos.x = pVertex->pos.x;
							oldpos.y = pVertex->pos.y;
							pVertex->pos.x = cand.x;
							pVertex->pos.y = cand.y;
							pVertex->iIndicator = 1;
									
							for (int j = 0; j < neigh_len; j++)
							{
								inext = j+1; if (inext == neigh_len) inext = 0;
								pNeigh = X + izNeighs[j];
								if (pNeigh->iIndicator == 1) bNo = true;
								pNeigh2 = X + izNeighs[inext];

								u.x = pNeigh->pos.x; 
								u.y = pNeigh->pos.y;
								if ((pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y) && (pNeigh->pos.x/pNeigh->pos.y > 0.5*GRADIENT_X_PER_Y))
									u = Anticlockwise*u;
								if ((pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y) && (pNeigh->pos.x/pNeigh->pos.y < -0.5*GRADIENT_X_PER_Y))
									u = Clockwise*u;
						
								u2.x = pNeigh->pos.x; 
								u2.y = pNeigh->pos.y;
								if ((pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y) && (pNeigh2->pos.x/pNeigh2->pos.y > 0.5*GRADIENT_X_PER_Y))
									u2 = Anticlockwise*u2;
								if ((pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y) && (pNeigh2->pos.x/pNeigh2->pos.y < -0.5*GRADIENT_X_PER_Y))
									u2 = Clockwise*u2;
							
								// now rotate u-u2 90 degrees

								direction.x = u2.y-u.y;
								direction.y = u.x-u2.x;

								// is dot product with vector to new pos same sign as to old pos?

								if (
									(direction.x*(pVertex->pos.x-u.x) + direction.y*(pVertex->pos.y-u.y))*(direction.dot(oldpos-u)) < 0.0    )
								{
									printf("\n\nalert");
									getch();
									j = j;
								};
							}; // next j

						};
					};

				}; // whether neighbour modified
			};
			++pVertex;
		};
	};
	
	// clean up indicator
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iIndicator = 0;
		++pVertex;
	};

	this->DebugTestWrongNumberTrisPerEdge();
}