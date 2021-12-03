
#include "mesh.h"
#include "globals.h"
#include "headers.h"
#include "FFxtubes.h"
#include "cuda_struct.h"
//#include "cppconst.h"

#define CHOSEN 52233
#define VERT1 14639
#define VERT2 14645
// these should explain something we hope.
extern HWND hWnd;
extern HWND hwndGraphics;
extern D3D Direct3D;
extern void RefreshGraphs(TriMesh & X, const int iGraphsFlag);


f64 sigma_tiles[1024]; 
const int TriFiles[12] = { 29414, 29413, 29412, 29108, 29109, 29110,
						29442, 29441, 29440, 29136, 29137, 29138 };
const int Chosens[7] = { 25454, 86529, 25453, 86381, 25455, 86530, 25750 };
f64 LapCoeffself[NMINOR];
f64 LapCoefftri[NUMTRIANGLES][6];
f64 LapCoeffvert[NUMVERTICES][MAXNEIGH];
f64 Azdot0[NMINOR], gamma[NMINOR];
f64 Lap_Jacobi[NMINOR];
f64 Jacobi_x[NMINOR];
f64 epsilon[NMINOR];
f64 Lap_Aznext[NMINOR];

real GlobalIzElasticity;


extern f64_vec2 RotateClosest(f64_vec2 pos, f64_vec2 prox);

f64 inline GetEzShape__(f64 r) {
	return 1.0 - 1.0 / (1.0 + exp(-24.0*(r - 4.32))); // At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
}  

//class nvT_data {
//public:
//	f64_vec3 v_e, v_i, v_n; // these live on minors
//	f64 T_e, T_i, T_n, n, n_n; // deliberately choosing just n so that we cannot put unpopulated n_e,n_i
//	// maybe we'll wanna change it
//	// These have to be averaged here -- n should come from shard mass integral divided by area;
//	// T can be simple-averaged
//
//	f64 Az, Azdot; // these live on minors
//
//};
//struct nTlist {
//	f64 nT_n[MAXNEIGH]; // find nT on spoke pointing towards each neighbour
//	f64 nT_i[MAXNEIGH];
//	f64 nT_e[MAXNEIGH];
//}; // 48 doubles


three_vec3 AdditionalMomRates[NMINOR];
f64_vec2 p_v[NMINOR];
f64_vec2 GradAz[NMINOR];
f64_vec2 GradTeArray[NMINOR];
f64 LapAzArray[NMINOR];
NTrates NTadditionrates[NUMVERTICES];
f64 p_div_v_neut[NUMVERTICES];
f64 p_div_v[NUMVERTICES];
f64 Integrated_Div_v_overall[NUMVERTICES];
f64 ROCAzdotduetoAdvection[NMINOR];
f64 ROCAzduetoAdvection[NMINOR];
f64 Az_array[NMINOR], Az_array_next[NMINOR];

ShardModel n_shards[NUMVERTICES], n_shards_n[NUMVERTICES];
f64 Tri_n_n_lists[NMINOR][6];
f64 Tri_n_lists[NMINOR][6];

// footprint of static? (MAXNEIGH*2+5 = 33 say)*NVERT + (15*NMINOR) = say 2500000 doubles = 20 000 000 bytes

real inline TriMesh::GetIzPrescribed(real const t)
{

	real Iz = -PEAKCURRENT_STATCOULOMB * sin((t + ZCURRENTBASETIME) * 0.5*PIOVERPEAKTIME); // t/peaktime * pi/2

	printf("\nGetIzPrescribed called with t+ZCURRENTBASETIME = %1.5E : %1.5E\n", t + ZCURRENTBASETIME, Iz);
	
	return Iz;
}
//
//
//if (bScrewPinch) {
//		return IZ_SCREW_PINCH;
//	} else {
//		real Iz = -PEAKCURRENT_STATCOULOMB * sin ((t + ZCURRENTBASETIME) * PIOVERPEAKTIME );
//		return Iz;
//	};


void TriMesh::SwimMesh(TriMesh * pSrcMesh)
{
	long izTri[MAXNEIGH];

	printf("SwimMesh in simulation.cu . \n");

	real acceptance, mass_avg, mass_SD, mass_min, mass_max, move, coefficient;

	FILE * swimfile = fopen("swim.txt", "a");

	// coefficient is the (adaptive) proportion of the steps we try to make....
	// why is it that most of our moves start getting rejected, I do not know
	coefficient = 0.5;

	double GlobalMaxVertexRadiusSq = 0.0;
	Vertex * pVertex = X;
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y > GlobalMaxVertexRadiusSq)
			GlobalMaxVertexRadiusSq = pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y;
		++pVertex;
	};

	long izNeigh[MAXNEIGH];
	//pVertex = X + 40910;
	//short neigh_len = pVertex->GetNeighIndexArray(izNeigh);
	//for (int iii = 0; iii < neigh_len; iii++)
	//{
	//	printf("iVertex 40910 neigh %d pos %1.9E %1.9E \n",
	//		izNeigh[iii], (X + izNeigh[iii])->pos.x, (X + izNeigh[iii])->pos.y);
	//};
	//getch();

	Triangle * pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateCentroid(this->InnermostFrillCentroidRadius,
			this->OutermostFrillCentroidRadius);
		//if (pTri->has_vertex(X + 40910)) {
		//	printf("SWIM iTri %d cent %1.9E %1.9E %d | ", iTri, pTri->cent.x, pTri->cent.y,
		//		pTri->u8domain_flag);
		//	Vector2 u[3];
		//	pTri->MapLeftIfNecessary(u[0], u[1], u[2]);
		//	printf(" %1.9E %1.9E |  %1.9E %1.9E |  %1.9E %1.9E | \n",
		//		u[0].x, u[0].y, u[1].x, u[1].y, u[2].x, u[2].y);
		//}
		++pTri; // this seems like it should still work if we have not wrapped any vertex that moved, even if tri no longer periodic in truth but some pts outside tranche

	};

	// puzzle why moves are rarely accepted so try something more empirical even if it takes a little longer to run.
	for (int j = 0; j < 10; j++) // 3 goes of 1 squeeze, 1 further go
	{
		fprintf(swimfile, "\n\nGSC: %d\n", GlobalStepsCounter);

		printf("%d ", j);
		fprintf(swimfile, "Swim %d  ", j);
				
		for (int jj = j; jj < 18; jj++)
		{
			move = this->SwimVertices(pSrcMesh, coefficient, &acceptance);
			printf(" L2 of moves: %1.6E  Squeeze: %1.6E   Acceptance rate: %.2f%%\n", move, coefficient, acceptance*100.0);
		};

		fprintf(swimfile, " L2 of moves: %1.6E  Squeeze: %1.6E   Acceptance rate: %.2f%%\n", move, coefficient, acceptance*100.0);
		//if ((acceptance > 0.1) && (move < 1.0e-7)) break; // stop if wasting time

		// Re-integrate density. ?
		printf("About to re-integrate. \n");
		// If need be, check here about the polygon surrounding a vertex in pX.
		// Is it something to do with cent vs cc ??
		//
		//pTri = T;
		//for (long iTri = 0; iTri < numTriangles; iTri++)
		//{		
		//	pTri->RecalculateCentroid(this->InnermostFrillCentroidRadius,
		//		this->OutermostFrillCentroidRadius);
		//	if (pTri->has_vertex(X + 4000)) {
		//		printf("iTri %d cent %1.9E %1.9E %d | ", iTri, pTri->cent.x, pTri->cent.y,
		//			pTri->u8domain_flag);
		//		Vector2 u[3];
		//		pTri->MapLeftIfNecessary(u[0], u[1], u[2]);
		//		printf(" %1.9E %1.9E |  %1.9E %1.9E |  %1.9E %1.9E | \n", 
		//			u[0].x, u[0].y, u[1].x, u[1].y, u[2].x, u[2].y);
		//	}
		//	++pTri; // this seems like it should still work if we have not wrapped any vertex that moved, even if tri no longer periodic in truth but some pts outside tranche
		//};

		this->Integrate_using_iScratch(pSrcMesh, ((j < 9)?false:true)); // AFFECTS ONLY pData, NO STRUCTURAL IMPACT
		
		SetActiveWindow(hwndGraphics);
		ShowWindow(hwndGraphics, SW_HIDE);
		ShowWindow(hwndGraphics, SW_SHOW);
		RefreshGraphs(*this, SPECIES_ION);
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
		UpdateWindow(hwndGraphics);

		// sometimes move came out 0 indicating acceptance rates had fallen very low ; ..
		// redundant code:
		if (acceptance == 0.0)
		{
			coefficient *= 0.05;
		} else {
			if ((acceptance < 0.5) && (j % 3 == 2))
			{
				if (acceptance > 0.025) {
					coefficient *= acceptance / 0.5; // Note:quite aggressive - could put sqrt
				} else {
					// too low...
					coefficient *= 0.05;
				};
			};
		};
	};

	fclose(swimfile);

	SetupMajorPBCTriArrays(); // they've been destroyed by resequencing.
	Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	Create4Volleys(); // destroyed in resequencing.

	// Now refresh v avg in tris because we created v on verts only.
	pTri = T;
	f64_vec2 vxy, vxysum;
	f64 viz, vez, vezsum, vizsum, Az, Azsum, Azdotsum, Azdot;
	f64_vec3 v_n, vnsum;
	int npts;
	for (long iTri = 0; iTri < BEGINNING_OF_CENTRAL; iTri++)
	{
		npts = 0;
		vnsum.x = 0.0; vnsum.y = 0.0; vnsum.z = 0.0;
		vxysum.x = 0.0; vxysum.y = 0.0;
		vizsum = 0.0; vezsum = 0.0;

		Azsum = 0.0; Azdotsum = 0.0;
		for (int i = 0; i < 3; i++)
		{
			long iVertex = pTri->cornerptr[i]-X;
			if (pData[iVertex + BEGINNING_OF_CENTRAL].n > 0.0) {
				vnsum += pData[iVertex + BEGINNING_OF_CENTRAL].v_n;
				vxysum += pData[iVertex + BEGINNING_OF_CENTRAL].vxy;
				vizsum += pData[iVertex + BEGINNING_OF_CENTRAL].viz;
				vezsum += pData[iVertex + BEGINNING_OF_CENTRAL].vez;

				Azsum += pData[iVertex + BEGINNING_OF_CENTRAL].Az;
				Azdotsum += pData[iVertex + BEGINNING_OF_CENTRAL].Azdot;
				npts++;
			};
		}
		if (npts == 0) {
			vxy = vxysum;
			v_n = vnsum;
			vez = vezsum;
			viz = vizsum;
			Az = pSrcMesh->pData[iTri].Az;
			Azdot = pSrcMesh->pData[iTri].Azdot;
		} else {
			f64 over = 1.0 / (real)npts;
			vxy = vxysum*over;
			v_n = vnsum*over;
			vez = vezsum*over;
			viz = vizsum*over;
			Az = Azsum*over;
			Azdot = Azdotsum*over;
		};
		pData[iTri].vxy = vxy;
		pData[iTri].vez = vez;
		pData[iTri].viz = viz;
		pData[iTri].v_n = v_n;
		pData[iTri].Az = Az;
		pData[iTri].Azdot = Azdot;
				
		++pTri;
	};

	//Ultimately it could be done better, but we aren't going to do that.

	//1. Fix vtri -- where is vn?
	//2. Fix Atri -- need to integrate Az.
//	3. Check it runs sim step without resprinkle.
//	4. Ask what is different with resprinkle.

}


real TriMesh::SwimVertices(TriMesh * pSrcMesh, real coefficient, real * pAcceptance)
{
	// First let's try moving towards barycenters to sort out the messiness.
	long izTri[MAXNEIGH];
	long izNeigh[MAXNEIGH];
	f64 sqrtn[MAXNEIGH];
	short tri_len, neigh_len, i , inext;
	ConvexPolygon cp;
	Vertex * pVertex = X;
	Vertex * pNeigh, *pNeigh2;
	f64_vec2 pos1, pos2, Pretend_barycenter;
	f64 n1, n2;
	Triangle * pTri,*pTri2;
	long iVertex;
	f64 leng, moverat;
	f64_vec2 Direction;
	f64 SumSq = 0.0;
	long NumSum = 0;
	
	pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateCentroid(this->InnermostFrillCentroidRadius,
			this->OutermostFrillCentroidRadius);

		if (iTri == 52332) printf("Tri 52332 flag %d \n"
			"pos %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E\n", 
			pTri->u8domain_flag,
			pTri->cornerptr[0]->pos.x, pTri->cornerptr[0]->pos.y,
			pTri->cornerptr[1]->pos.x, pTri->cornerptr[1]->pos.y,
			pTri->cornerptr[2]->pos.x, pTri->cornerptr[2]->pos.y
			);

		// Makes no allowance for CROSSING_CATH but we don't care much.
		// Also GPU does not actually make different centroid in that case
		// -- don't know why not but it's deliberate.
		
		//if (pTri->has_vertex(X + 4000)) {
		//	printf("iTri %d cent %1.9E %1.9E %d %d | ", iTri, pTri->cent.x, pTri->cent.y,
		//		pTri->u8domain_flag, pTri->periodic);
		//	Vector2 u[3];
		//	pTri->MapLeftIfNecessary(u[0], u[1], u[2]);
		//	printf(" %1.9E %1.9E |  %1.9E %1.9E |  %1.9E %1.9E | %d %d %d\n",
		//		u[0].x, u[0].y, u[1].x, u[1].y, u[2].x, u[2].y,
		//		pTri->cornerptr[0] - X,
		//		pTri->cornerptr[1] - X,
		//		pTri->cornerptr[2] - X
		//	);
		//}

		// Recompute n based on domain corners only:

		int npts = 0, i1;
		f64_vec2 vec2;
		f64 nsum = 0.0, nnsum = 0.0;
		switch (pTri->u8domain_flag) {
		case DOMAIN_TRIANGLE:
			if (iTri == 52332) printf("52332 Domain triangle %d %1.8E %d %1.8E %d %1.8E\n",
				pTri->cornerptr[0] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[0] - X].n_n,
				pTri->cornerptr[1] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[1] - X].n_n,
				pTri->cornerptr[2] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[2] - X].n_n
			);

			pData[iTri].n = THIRD*(pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[0] - X].n
				+ pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[1] - X].n
				+ pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[2] - X].n);
			pData[iTri].n_n = THIRD*(pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[0] - X].n_n
				+ pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[1] - X].n_n
				+ pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[2] - X].n_n);
			break;

		case CROSSING_INS:
			if (iTri == 52332) printf("52332 Ins triangle %d %1.8E %d %1.8E %d %1.8E\n"
				"pos %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E\n",
				pTri->cornerptr[0] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[0] - X].n_n,
				pTri->cornerptr[1] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[1] - X].n_n,
				pTri->cornerptr[2] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[2] - X].n_n,
				pTri->cornerptr[0]->pos.x, pTri->cornerptr[0]->pos.y,
				pTri->cornerptr[1]->pos.x, pTri->cornerptr[1]->pos.y,
				pTri->cornerptr[2]->pos.x, pTri->cornerptr[2]->pos.y
			);

			for (i1 = 0; i1 < 3; i1++)
			{
				vec2 = pTri->cornerptr[i1]->pos;
				if (vec2.dot(vec2) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					nsum += pData[pTri->cornerptr[i1] - X + BEGINNING_OF_CENTRAL].n;
					nnsum += pData[pTri->cornerptr[i1] - X + BEGINNING_OF_CENTRAL].n_n;
					++npts;
				};
			}
			if (npts == 0) {
				printf("%d npts = 0 CROSSING_INS\n", iTri);
			}
			else {
				pData[iTri].n = nsum / (real)npts;
				pData[iTri].n_n = nnsum / (real)npts;
			};
			break;
		case CROSSING_CATH:
			if (iTri == 52332) printf("52332 Cath triangle %d %1.8E %d %1.8E %d %1.8E\n"
				"pos %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E\n",
				pTri->cornerptr[0] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[0] - X].n_n,
				pTri->cornerptr[1] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[1] - X].n_n,
				pTri->cornerptr[2] - X, pData[BEGINNING_OF_CENTRAL + pTri->cornerptr[2] - X].n_n,
				pTri->cornerptr[0]->pos.x, pTri->cornerptr[0]->pos.y,
				pTri->cornerptr[1]->pos.x, pTri->cornerptr[1]->pos.y,
				pTri->cornerptr[2]->pos.x, pTri->cornerptr[2]->pos.y
			);
			for (i1 = 0; i1 < 3; i1++)
			{
				vec2 = pTri->cornerptr[i1]->pos;
				vec2.y -= CATHODE_ROD_R_POSITION; // vec2 is vector from rod centre
				if (vec2.dot(vec2) > CATHODE_ROD_RADIUS*CATHODE_ROD_RADIUS)
				{
					nsum += pData[pTri->cornerptr[i1] - X + BEGINNING_OF_CENTRAL].n;
					nnsum += pData[pTri->cornerptr[i1] - X + BEGINNING_OF_CENTRAL].n_n;
					++npts;
				};
			}
			if (npts == 0) {
				printf("%d npts = 0 CROSSING_CATH\n", iTri);
			}
			else {
				pData[iTri].n = nsum / (real)npts;
				pData[iTri].n_n = nnsum / (real)npts;
			};
			break;
		default:
			pData[iTri].n = 0.0;
			pData[iTri].n_n = 0.0;
		};

		++pTri; // this seems like it should still work if we have not wrapped any vertex that moved, even if tri no longer periodic in truth but some pts outside tranche
	};
#define TEST50 0  // (iVertex == 13798))
	


	// We are doing based on izNeigh instead of izTri
	// This is no good. For one thing we can get sucked beneath the periodic boundary.
	// For another thing we need to define n in ins_crossing triangles in the appropriate way, not use out of domain values of n.

	// Conclusion : DO BASED ON izTri !!
//  Do based on izTri !!
	// _________________________

	f64_vec2 Uniform_barycenter, Direction2;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (iVertex % 1000 == 0) printf("%d ", iVertex);
		if ((pVertex->flags == DOMAIN_VERTEX) &&
			(pVertex->pos.dot(pVertex->pos) < START_SPREADING_OUT_RADIUS*START_SPREADING_OUT_RADIUS))
		{		
			// . Populate polygon about vertex:

			tri_len = pVertex->GetTriIndexArray(izTri);
			f64_vec2 numer, numer2;
			f64 denom, denom2;
			numer.x = 0.0; numer.y = 0.0; denom = 0.0; denom2 = 0.0; numer2.x = 0.0; numer2.y = 0.0;
			for (i = 0; i < tri_len; i++) {
				inext = i + 1; if (inext == tri_len) inext = 0;
				pTri = &(T[izTri[i]]);
				pTri2 = &(T[izTri[inext]]);
				pos1 = RotateClosest(pTri->cent, pVertex->pos);
				pos2 = RotateClosest(pTri2->cent, pVertex->pos);
				n1 = pData[izTri[i]].n + pData[izTri[i]].n_n;
				n2 = pData[izTri[inext]].n + pData[izTri[inext]].n_n;

				cp.Clear();
				cp.add(pos1);
				cp.add(pos2);
				cp.add(pVertex->pos);

				f64 weight1 = cp.GetArea();
				f64 argument = (n1 + n2 + pData[iVertex + BEGINNING_OF_CENTRAL].n
					+ pData[iVertex + BEGINNING_OF_CENTRAL].n_n);
				if ((argument < 0.0) || (argument == 0.0)) printf("Vertex %d argument %1.9E\n"
					"n1 %1.9E pData[%d].n %1.9E pData[%d].nn %1.9E \n"
					"n2 %1.9E pData[%d].n %1.9E pData[%d].nn %1.9E \n"
					"pData[%d+BOC].n %1.9E pData[%d+BOC].nn %1.9E \n",
					iVertex, argument,
					n1, izTri[i], pData[izTri[i]].n, izTri[i], pData[izTri[i]].n_n,
					n2, izTri[inext], pData[izTri[inext]].n, izTri[inext], pData[izTri[inext]].n_n,
					iVertex, pData[iVertex + BEGINNING_OF_CENTRAL].n, iVertex, pData[iVertex + BEGINNING_OF_CENTRAL].n_n
				);
				
				f64 weight = weight1*sqrt(argument)*THIRD;
				denom += weight;
				numer += weight*THIRD*(pos1 + pos2 + pVertex->pos);

				denom2 += weight1; // uniform barycenter
				numer2 += weight1*THIRD*(pos1 + pos2 + pVertex->pos);
/*

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);

			f64_vec2 numer, numer2;
			f64 denom, denom2;
			numer.x = 0.0; numer.y = 0.0; denom = 0.0; denom2 = 0.0; numer2.x = 0.0; numer2.y = 0.0;
			bool bFound = false;
			for (i = 0; i < neigh_len; i++) {
				inext = i + 1; if (inext == neigh_len) inext = 0;
				pNeigh = &(X[izNeigh[i]]);
				pNeigh2 = &(X[izNeigh[inext]]);
				pos1 = RotateClosest(pNeigh->pos, pVertex->pos);
				pos2 = RotateClosest(pNeigh2->pos, pVertex->pos);
				n1 = pData[izNeigh[i] + BEGINNING_OF_CENTRAL].n
					+ pData[izNeigh[i] + BEGINNING_OF_CENTRAL].n_n;
				n2 = pData[izNeigh[inext] + BEGINNING_OF_CENTRAL].n
					+ pData[izNeigh[inext] + BEGINNING_OF_CENTRAL].n_n;

				if (TEST50) printf("izNeigh[%d] %d pNeigh->pos %1.9E %1.9E our pos %1.9E %1.9E n1 n2 %1.9E %1.9E\n",
					i, izNeigh[i], pNeigh->pos.x, pNeigh->pos.y, pVertex->pos.x, pVertex->pos.y, n1, n2);

				// Suspect this one did not get populated properly by integration.

				cp.Clear();
				cp.add(pos1);
				cp.add(pos2);
				cp.add(pVertex->pos);
				
				if (TEST50)  printf("n1 %1.8E n2 %1.8E data n %1.9E %1.9E pos1 pos2 %1.9E %1.9E %1.9E %1.9E\n",
					n1, n2, pData[iVertex + BEGINNING_OF_CENTRAL].n, pData[iVertex + BEGINNING_OF_CENTRAL].n_n,
					pos1.x, pos1.y, pos2.x, pos2.y);
				// why is barycenter bad?
				
				f64 weight1 = cp.GetArea();

				// Vertex 13798 argument -3.905548138E+03

				f64 argument = (n1 + n2 + pData[iVertex + BEGINNING_OF_CENTRAL].n
										+ pData[iVertex + BEGINNING_OF_CENTRAL].n_n);
				if ((argument < 0.0) || (argument == 0.0)) printf("Vertex %d argument %1.9E\n"
					"n1 %1.9E pData[%d+BOC].n %1.9E pData[%d+BOC].nn %1.9E \n"
					"n2 %1.9E pData[%d+BOC].n %1.9E pData[%d+BOC].nn %1.9E \n"
					"pData[%d+BOC].n %1.9E pData[%d+BOC].nn %1.9E \n",
					iVertex, argument, 
					n1, izNeigh[i], pData[izNeigh[i] + BEGINNING_OF_CENTRAL].n, izNeigh[i], pData[izNeigh[i] + BEGINNING_OF_CENTRAL].n_n,
					n2, izNeigh[inext], pData[izNeigh[inext] + BEGINNING_OF_CENTRAL].n, izNeigh[inext], pData[izNeigh[inext] + BEGINNING_OF_CENTRAL].n_n,
					iVertex, pData[iVertex + BEGINNING_OF_CENTRAL].n, iVertex, pData[iVertex + BEGINNING_OF_CENTRAL].n_n
					);



				f64 weight = weight1*sqrt(argument)*THIRD;
				denom += weight;
				numer += weight*THIRD*(pos1 + pos2 + pVertex->pos);

				denom2 += weight1; // uniform barycenter
				numer2 += weight1*THIRD*(pos1 + pos2 + pVertex->pos);

				if (TEST50)  printf("weight %1.8E numer.x contrib %1.8E \n",
					weight, weight*THIRD*((pos1 + pos2 + pVertex->pos).x));
					*/			
			};
			if (TEST50)
				printf("numer %1.9E %1.9E denom %1.9E numer2 %1.9E %1.9E denom2 %1.9E\n", numer.x, numer.y, denom,
					numer2.x, numer2.y, denom2);
			if (denom == 0.0) printf("About to divide by zero denom: %d %1.8E %1.8E\n", iVertex, pVertex->pos.x, pVertex->pos.y);
			if (denom2 == 0.0) printf("About to divide by zero denom2: %d %1.8E %1.8E\n", iVertex, pVertex->pos.x, pVertex->pos.y);
			Pretend_barycenter = numer / denom;
			Uniform_barycenter = numer2 / denom2;

			//    If we were over 0.03 = 300 micron from a neighbour point, track towards it instead:			
			// 3. Fix why it is again coming out with zero in some of these middle places. Again.
			
			Direction = Pretend_barycenter - pVertex->pos;
			leng = Direction.modulus();
			
			if (TEST50)  printf("\nleng %1.9E Direction %1.9E %1.9E Barycenter %1.9E %1.9E pos %1.9E %1.9E\n",
				leng, Direction.x, Direction.y,
				Pretend_barycenter.x, Pretend_barycenter.y,
				pVertex->pos.x, pVertex->pos.y
				);
			if (leng <= 0.0) printf("About to divide by <=zero leng %d\n", iVertex);

			moverat = min(sqrt(0.1 * 0.005/leng),1.0); // cm						

			// Should this be max or min ?????????????
			
			Direction *= moverat*coefficient;

			f64_vec2 newpos = pVertex->pos + Direction;

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			bool bFound = false;
			for (i = 0; i < neigh_len; i++) {
				inext = i + 1; if (inext == neigh_len) inext = 0;
				pNeigh = &(X[izNeigh[i]]);
				pNeigh2 = &(X[izNeigh[inext]]);
				pos1 = RotateClosest(pNeigh->pos, pVertex->pos);
				pos2 = RotateClosest(pNeigh2->pos, pVertex->pos);
				if ((((pos1 - pVertex->pos).dot(pos1 - pVertex->pos)) > 0.03*0.03)
					||
					(((pos2 - pVertex->pos).dot(pos2 - pVertex->pos)) > 0.03*0.03))
					bFound = true;
			};

			f64 maxdistsq = 0.0;
			int iMax = -1;
			if (bFound) {
				//for (i = 0; i < neigh_len; i++) {
				//	pNeigh = &(X[izNeigh[i]]);
				//	pos1 = RotateClosest(pNeigh->pos, newpos);
				//	if (((pos1 - newpos).dot(pos1 - newpos)) > maxdistsq)
				//	{
				//		maxdistsq = ((pos1 - newpos).dot(pos1 - newpos));
				//		iMax = i;
				//	};
				//};
				//pos1 = RotateClosest(X[izNeigh[iMax]].pos, newpos);
				//Direction2 = pos1 - newpos;
				//// Cancel first part of move if it took us away from new farthest point:
				//if (Direction.dot(Direction2) < 0.0) {
				//	Direction.x = 0.0; Direction.y = 0.0;
				//	newpos = pVertex->pos;
				//}
				// Now move from newpos, add Direction, towards uniform barycenter
				//Direction2 = (Uniform_barycenter - newpos)*coefficient;
				//Direction += Direction2;

				Direction = Uniform_barycenter - pVertex->pos;
				leng = Direction.modulus();
				if (leng <= 0.0) printf("About to divide by <=zero leng %d\n", iVertex);
				moverat = min(sqrt(0.1 * 0.005 / leng), 1.0); // cm			
				Direction *= moverat;
			};
			
			pVertex->pos += Direction;	
			if (pVertex->pos.y == 0.0) printf("About to divide by zero pVertex->pos.y %d\n", iVertex);

			if (pVertex->pos.x / pVertex->pos.y > GRADIENT_X_PER_Y)
			{
				// went off RH side
				printf("R");
				f64 newx, newy;
				tri_len = pVertex->GetTriIndexArray(izTri);
				for (i = 0; i < tri_len; i++)
				{
					// if triangle is periodic then we need to map other vertices to living nearby....
					pTri = T + izTri[i];
					pTri->IncrementPeriodic();
				};
				newx = Anticlockwise.xx*pVertex->pos.x + Anticlockwise.xy*pVertex->pos.y;
				newy = Anticlockwise.yx*pVertex->pos.x + Anticlockwise.yy*pVertex->pos.y;
				pVertex->pos.x = newx;
				pVertex->pos.y = newy;
			};

			if (pVertex->pos.x / pVertex->pos.y < -GRADIENT_X_PER_Y)
			{
				printf("L");
				f64 newx, newy;
				tri_len = pVertex->GetTriIndexArray(izTri);
				for (i = 0; i < tri_len; i++)
				{
					// if triangle is periodic then we need to map other vertices to living nearby....
					pTri = T + izTri[i];
					pTri->DecrementPeriodic();
				};
				newx = Clockwise.xx*pVertex->pos.x + Clockwise.xy*pVertex->pos.y;
				newy = Clockwise.yx*pVertex->pos.x + Clockwise.yy*pVertex->pos.y;
				pVertex->pos.x = newx;
				pVertex->pos.y = newy;
			};

			SumSq += Direction.dot(Direction);

		//	if (Direction.modulus() > 0.03) {
		//		printf("Crezzy move: %d Direction %1.9E %1.9E Pretend_barycenter %1.9E %1.9E pos %1.9E %1.9E\n",
		//			iVertex, Direction.x, Direction.y,
		//			Pretend_barycenter.x, Pretend_barycenter.y,
		//			pVertex->pos.x, pVertex->pos.y);
		//		getch();
		//	}

			NumSum++;
		};

		++pVertex;
	};
	
	f64 L2 = sqrt(SumSq / (real)NumSum);

	// Now recalculate triangle centroids.

	this->Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	printf("L2 %1.7E\n", L2);
	printf("About to re-delaunerize.\n"); 
	pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateCentroid(this->InnermostFrillCentroidRadius,
			this->OutermostFrillCentroidRadius);
		//if (pTri->has_vertex(X + 4000)) {
		//	printf("iTri %d cent %1.9E %1.9E %d %d | ", iTri, pTri->cent.x, pTri->cent.y,
		//		pTri->u8domain_flag, pTri->periodic);
		//	Vector2 u[3];
		//	pTri->MapLeftIfNecessary(u[0], u[1], u[2]);
		//	printf(" %1.9E %1.9E |  %1.9E %1.9E |  %1.9E %1.9E | %d %d %d\n",
		//		u[0].x, u[0].y, u[1].x, u[1].y, u[2].x, u[2].y,
		//		pTri->cornerptr[0]-X,
		//		pTri->cornerptr[1] - X,
		//		pTri->cornerptr[2] - X
		//		);
		//}
		++pTri; // this seems like it should still work if we have not wrapped any vertex that moved, even if tri no longer periodic in truth but some pts outside tranche
	};
	// Redelaunerize:	
	this->Redelaunerize(false, true);
	this->RefreshVertexNeighboursOfVerticesOrdered();
	// What is not populated properly after Redelaun? BIG QUESTION. Need to set it out in orderly fashion.

	*pAcceptance = 1.0;
		
	//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	

	/*
	// Redistribute vertices towards where we get more equal masses in cells.

	// We must have already positioned vertices in pDestMesh -- position same as existing initially;
	// *this has the existing mesh and values that we keep until we are finished with iterating.

	// ( We should then swap pointers: DestMesh is then the current system. )


	//		1. Swim vertices according to distribution of mass seen on dest mesh;
	//		2. Renew mass distribution for dest mesh by doing advectioncompression with 0 velocity 
	// Then loop.

	// When we are ready to stop, we want those filled in values for dest mesh.

	// Here's one thought.
	// Suppose we pick a time when we are doing a move anyway.
	// ...
	//					we created the new mesh
	//				we do advectioncompression for two species
	//				that lands us with mass
	//					let's say we see that vertices can swim profitably
	//				we move them, zero the cells and do the advectioncompression again
	//					is there a criterion to see that it was an improvement?

	// OK so perhaps we want two functions:
	// 1. Establish whether we can make gains
	// 2. Make the moves

	// Remember that we want to FinishAdvectingMesh when we reposition the vertices
	// Might want to look over that also.



	// try to decide a more optimal positioning for this point based on equalising the masses of triangles...
	// So we want density to be inversely proportional to area.

	// If we assume that the density at this vertex is what we add or take away from a triangle, that is 
	// probably not madly wrong.


	// Well here is an idea ... assume density in triangles stays fairly constant ... this will be a small move ...
	// or assume that it's somewhere between the two ... 
	// ... either actually might fail if just one of the neighbours is very tall.
	// in that case we need to be assuming that the edge density of the tall cell is what we add when we
	// move the boundaries.

	// Could do the following way: 
	// We need to pick a direction and a magnitude for changing vertex position.
	// Pick the direction that optimises the sum of squared differences from equal mass -- ??

	// Can't remember which way worked best.

	// Maybe we should just consider moving towards average of neighbours --- is it true that if cells
	// are maldistributed then there is always something that can do this?

	// However it might sometimes be, if behaviour locally is very bad, so be careful.


	// OK let's do this -- let's consider just moving towards neighbour average, which makes the grid more equilateral.


	// rate of change of triangle area can be found by dotting this move direction with direction perp to other edge

	// If we assume density in each triangle is given (this may have proven to be a bad assumption before)
	// then we can try to minimize sum of squared masses ?

	// What we did before was take Sum(mass - avgmass)^2 as objective function and estimated grad empirically,
	// then changed magnitude - to settle on one s.t. both initial value is worse and halving magnitude is also worse.


	// So here is what we should do: only move lone vertices: set a flag to each neighbour that it has to be in the next volley.
	// This way we can tell if each move has been an improvement, if we remember the previous objective function at each vertex.

	// We're spoilt for choice: can indicate which volley with Vertex::iScratch or Vertex::flags

	// Vertex::e_pm_Heat_denominator can be the objective function stored for the initial position
	// Vertex::eP_Viscous_denominator_x, eP_Viscous_denominator_y can be the stored initial position

	// Vertex::IonP_Viscous_numerator_x, IonP_Viscous_numerator_y is the position we are leapfrogging
	// Vertex::ion_pm_Heat_numerator can be the objective function stored for this position

	// Vertex::iScratch can be the volley to which it is assigned.
	// (and we repeat until everything is assigned status -1 meaning done).
	// Vertex::flags can be which way we are heading -- 0 means more change, 1 means less.

	// in this way we bisect to get an improvement:
	// if the new value is worse than e_pm_Heat_denominator then we go smaller until it is better
	// and keep going smaller until it stops getting better
	// if the new value is better, we can try going larger; when it stops being better we stop and accept previous

	// Algorithm: 
	// If masses are already within ~8% of each other, it is not worth moving.
	//	If they are somewhat different, do a move:

	//					create 2 first guesses 20% apart --- the magnitude to pick may be based on a number of things,
	// but in particular we might solve the linear equation for d/dt (sum of squared masses) = 0.
	// Note that we also get our grad Objective by assuming that the change is adding (and subtracting) at rate n_vertex.

	// Let's say we take that, reduce it somewhat if necessary according to practical constraints,
	// then consider that vs 80% of that.

	// Now one guess will be better than the other;
	// we walk another 10% that way
	// and continue as long as it keeps getting better.
	// If we end up with a quite small move then stop bothering.
	// When we come to a guess that is worse, we go back to the previous one Viscous_numerator.

	// If that is no better than the original, we fail and stay where we were; if it is better, we accept it.

	// ... so, we need to store which way we are going; for this try Vertex::flags

	// We also need to store the direction we are heading in -- make this Pressure_numerator
	// -- since we assume we are not doing this as part of an advance.

	// it would be far better to therefore _NOT_ do this re-jig as part of advection
	// There will be a few stubborn points where we do the re-mapping many times -- for these,
	// we want to just re-create masses repeatedly using the triangles locally, not re-doing the whole system.

	// ...Seems that it is high time we created a function that returns the triplanar model for a tri
	// or the quadriplanar model for a wedge.
	// We will need it again for smoothings.
	// We will need to do a zero-velocity advection to place triangles of mass and mom on to this new mesh.
	// Wish to do it for particular sets of triangles at a time.

	// Bite the leather.






	// New plan.

	// Do populate masses from source mesh each time.

	// Each volley:
	//				Populate masses for initial position from source mesh;
	//				Calculate objective functions and store them; label neighbour vertices to next volley;
	//					store old positions and create new ones based on grad objective function (store grad and magnitude)
	//				Populate masses again from source mesh;
	//					Create another guess of position: store objective function and our first guess
	//				Populate masses again from source mesh;
	//					Calculate objective functions for 3rd time; now accept the best guess of the 3.

	// so we have 3 populates per volley; we may have 4 volleys I expect. But it could be 5. 
	// .... This is a fairly expensive procedure to run even 1 go of. 


	// REMEMBER TO CALL VERTEXNEIGHBOURSOFVERTICES BEFORE WE EMBARK ON THIS SWIMVERTICES BUSINESS
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	long * pIndex;
	Vertex * pVertex;
	long iVertex, iTri;
	real length1, length2;
	Vector2 average_pos, changevector;
	int i;
	Triangle * pTri;
	int iVolley = 0;
	bool found_vertices_this_volley;
	real objective, weight, area, minlen, magnitude, mass;
	Vector2 grad, putative;
	real d_mass_by_dx, d_area_by_dx, d_mass_by_dy, d_area_by_dy;
	Vector2 u0, u1, u2, uO;
	real sum_mass_times_rate_of_change, sum_squared_rates_of_change, sum_squared_move_length,
		d_mass_per_unit_grad, newx, newy;
	int triangles_len;
	bool crush_radial_move;
	Triangle * pDummyTri;
	int which, c1;

	real xdist, ydist, dist, graddotnormal, max, original_dot, original_dist, new_position_dot;
	Vector2 edgenormal, rhat, mingrad, from_here, to1, to2, u;
	Vertex * pNeigh1, *pNeigh2;
	real d_minmass_by_dt, d_mass_by_dt, crossover, minmass,
		normaldistance, tmax;
	int iMin;
	long attempted, accepted;
	attempted = 0; accepted = 0;

#ifdef VERTBASED

	// Have to work out what to do

	return 0;
	// wish I hadn't done this change.

#else
	pVertex = X;
	
	sum_squared_move_length = 0.0; // a crude way to gauge how much impression this call of SwimVertices makes
	short neigh_len;
	long izNeigh[MAXNEIGH];

	for (iVolley = 0; iVolley < 4; iVolley++)
	{
		
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			pTri->RecalculateEdgeNormalVectors(true); // these are used below
			++pTri;
		};
		//	Populate masses for initial position from source mesh;
		printf("`");
		pSrcMesh->RepopulateCells(this, MASS_ONLY); // this will need attention.
		printf(".");

		//	Calculate objective functions and store them; label neighbour vertices to next volley;
		//	store old positions and create new ones based on grad objective function (store grad and magnitude)
		
		long izTri[MAXNEIGH];
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if (pVertex->iVolley == iVolley)
			{
				objective = 0.0;
				grad.x = 0.0; grad.y = 0.0;
				sum_squared_rates_of_change = 0.0;
				sum_mass_times_rate_of_change = 0.0;

				triangles_len = pVertex->GetTriIndexArray(izTri);
				pData[iVertex + BEGINNING_OF_CENTRAL].n = 0.0;
				pData[iVertex + BEGINNING_OF_CENTRAL].n_n = 0.0;
				for (i = 0; i < triangles_len; i++)
				{
					// Get objective function:
					//pTri = (Triangle *)(pVertex->triangles.ptr[i]);
					iTri = izTri[i];

					T.area : populated ? Where ? It exists apparently.And would have to be kept well populated.;

					area = pTri->GetArea();
					mass = (pData[iTri].n + pData[iTri].n_n)*area;
					objective += mass*mass;

					// Recalculate pVertex->ion.n and pVertex->neut.n
					weight = pTri->ReturnAngle(pVertex); // takes acct of periodic & wedge cases.
					pVertex->ion.n += weight*(pData[iTri].n);
					pVertex->neut.n += weight*(pTri->neut.mass) / area; // really only want the total of course
																							// (( Is this how vertex n is calculated elsewhere? ))
				};

				Okay, so this is an older way of approaching it. n vertex is what has to be calculated as an average.
				Our way, we need to rethink what happens when points move around.

					What happens to vertex n?

					Get it rite.


					.. We want to move towards centre of mass of shard model.
					.. Only actually move if figure of merit is getting better.
					.. Another way to look at it : want to minimize the sum of squared FOM's. Therefore take gradient of this objective function
					.. with respect to the position. 
					.. But do not move on the vector that takes you away from COM & do not move towards COM on the vector away from objective function improvement.

					That's quite clever.







				pVertex->e_pm_Heat_denominator = objective; // store for initial positions
				pVertex->eP_Viscous_denominator_x = pVertex->pos.x;
				pVertex->eP_Viscous_denominator_y = pVertex->pos.y; // store initial positions

																// Now collect contributions to grad Area, and normalise:
				iMin = 0; minmass = 1.0e100;
				mingrad.x = 0.0; mingrad.y = 0.0;
				for (i = 0; i < triangles_len; i++)
				{
					pTri = (Triangle *)(pVertex->triangles.ptr[i]);
					pTri->Return_grad_Area(pVertex, &d_area_by_dx, &d_area_by_dy); 		// contiguous with pVertex
					mass = (pTri->ion.mass + pTri->neut.mass);

					// Now estimate gradient of objective:
					// d/dx sum of squares = 2 sum (mass )(d/dx mass )
					// To find change of area, dot x with the vector that is normal to the other side						
					// In this whole function we need to map periodically everything to be on same side as pVertex.
					// THAT is rather crucial isn't it 

					d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
					d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;

					if (mass < minmass) {
						minmass = mass;
						iMin = i;
						mingrad.x = d_mass_by_dx;
						mingrad.y = d_mass_by_dy;
					};
					grad.x += mass*d_mass_by_dx*2.0; // gradient of mass*mass ... 
					grad.y += mass*d_mass_by_dy*2.0;
				};
				grad.Normalise();

				// Now in place of grad we want our intended move direction.
				if (pVertex->flags >= 3) {
					// delete radial component:
					rhat.x = pVertex->pos.x; rhat.y = pVertex->pos.y; rhat.Normalise();
					grad -= rhat*(grad.dot(rhat));
				};

				// Test: if grad is making minimum tri smaller that is bad
				// Bear in mind we expect to head along negative of grad to REDUCE objective function
				if (grad.x*mingrad.x + grad.y*mingrad.y > 0.0)
				{
					// heading against mingrad - no good.
					// we already set pVertex->eP_Viscous_denominator_x = pVertex->pos.x

					pVertex->IonP_Viscous_numerator_x = pVertex->pos.x;
					pVertex->IonP_Viscous_numerator_y = pVertex->pos.y; // store unwrapped first guess positions

																	// would be good to record number of times we hit this branch.
				}
				else {

					// IN EDGE CASE, I think we should be only mooting such moves in the first place
					// We can still take gradient of mass*mass, 2D, but then consider moving azimuthally.

					// Decide where to place a guess of a better position:
					// How to find d/dt sum of squares = 0? Modelling area as linear function of progress in this direction,
					// Magnitude :
					//						t = - sum (dA/dt ^2) / sum (A * dA/dt)

					// not sure about that??

					// Now get sums

					// This bit was not commented:
					//for (i = 0; i <  triangles_len; i++)
					//{
					//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);
					//	
					//	mass = (pTri->ion.mass + pTri->neut.mass);
					//	pTri->Return_grad_Area(pVertex,&d_area_by_dx,&d_area_by_dy); 			
					//	d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
					//	d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;
					//	
					//	d_mass_per_unit_grad = d_mass_by_dx*grad.x + d_mass_by_dy*grad.y; // for a move in direction of grad.
					//	sum_squared_rates_of_change += d_mass_per_unit_grad*d_mass_per_unit_grad;
					//	sum_mass_times_rate_of_change += d_mass_per_unit_grad*mass;
					//};
					//
					//magnitude = - sum_mass_times_rate_of_change / sum_squared_rates_of_change;

					// THAT SEEMS LIKE A BOLD PLAN !

					// Alternative idea
					// ______________
					// They are all changing at different rates
					// Stop when one that is moving down becomes the minimum one???
					// How to do?
					// maybe just stop when whichever ones are moving downwards, reach past original average ?
					// might be one close to average moving down. 

					// See when the down-movers cross over the one coming up from least mass.
					// Stop when it crosses over one. Is there a scenario where that is bad? Think it looks pretty good.

					// assume we head in direction MINUS grad
					d_minmass_by_dt = -(mingrad.dot(grad)); // > 0

					magnitude = 1.0e100;
					for (i = 0; i < triangles_len; i++)
					{
						pTri = (Triangle *)(pVertex->triangles.ptr[i]);

						mass = (pTri->ion.mass + pTri->neut.mass);
						pTri->Return_grad_Area(pVertex, &d_area_by_dx, &d_area_by_dy);
						d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
						d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;

						d_mass_by_dt = -d_mass_by_dx*grad.x - d_mass_by_dy*grad.y;
						// ROC for a move in direction of minus grad, we think.

						if (d_mass_by_dt < 0.0) {
							// find crossing time; take t = min(t,crossover of this one with min mass)
							crossover = (mass - minmass) / (d_minmass_by_dt - d_mass_by_dt);
							magnitude = min(magnitude, crossover);
						};
					};

					// Just measure, in the first place, the normal distance across a triangle, and travel at most 0.33 of this.
					// Or perhaps just pick the ones where the motion is making that normal shorter - yes.

					if (pVertex->flags < 3) {
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);

							which = 0; c1 = 1;
							if (pVertex == pTri->cornerptr[1]) { which = 1; c1 = 0; };
							if (pVertex == pTri->cornerptr[2]) which = 2;
							pTri->cornerptr[c1]->PopulatePosition(u1);
							edgenormal = pTri->edge_normal[which]; // We called with (true) so is already normalised
																	// this faces across the triangle.

							if (pTri->periodic > 0)
							{
								if (pVertex->pos.x < 0.0) {
									if (u1.x > 0.0) u1 = Anticlockwise*u1;
								}
								else {
									edgenormal = Clockwise*edgenormal; // make contig with our vertex
									if (u1.x < 0.0) u1 = Clockwise*u1;
								};
							};

							if (grad.dot(edgenormal) < 0.0) {
								// only care if we are making the distance shorter by moving in direction _minus grad_
								from_here.x = u1.x - pVertex->pos.x;
								from_here.y = u1.y - pVertex->pos.y;
								normaldistance = from_here.dot(edgenormal); // >0
																			// -grad dot edgenormal is the rate of progress in reducing normal distance, dot product of normalised vectors
								tmax = -0.33*(normaldistance / (grad.dot(edgenormal))); // >0
								magnitude = min(tmax, magnitude);
							};
						};
					}
					else {
						// pVertex->flags >= 3 : test against base neighbour distance only
						pNeigh2 = X + pVertex->neighbours.ptr[0];
						pNeigh1 = X + pVertex->neighbours.ptr[pVertex->neighbours.len - 1];
						if ((pNeigh2->flags != pVertex->flags) || (pNeigh1->flags != pVertex->flags))
						{
							printf("\nDid we fail to call RefreshNeighboursOfVerticesOrdered?\n");
							getch();
						};
						// we will only be moving towards one of them. Which one?
						if (pVertex->has_periodic) {
							pNeigh1->PopulatePosition(u1);
							pNeigh2->PopulatePosition(u2);
							if (pVertex->pos.x > 0.0) {
								if (pNeigh1->x < 0.0) u1 = Clockwise*u1;
								if (pNeigh2->x < 0.0) u2 = Clockwise*u2;
							}
							else {
								if (pNeigh1->x > 0.0) u1 = Anticlockwise*u1;
								if (pNeigh2->x > 0.0) u2 = Anticlockwise*u2;
							};
							to1.x = u1.x - pVertex->pos.x; to1.y = u1.y - pVertex->pos.y;
							to2.x = u2.x - pVertex->pos.x; to2.y = u2.y - pVertex->pos.y;
						}
						else {
							to1.x = pNeigh1->x - pVertex->pos.x; to1.y = pNeigh1->y - pVertex->pos.y;
							to2.x = pNeigh2->x - pVertex->pos.x; to2.y = pNeigh2->y - pVertex->pos.y;
						};
						if (grad.dot(to1)*grad.dot(to2) > 0.0)
						{
							printf("summat WEIRD - heading towards/away from both edge neighs \n");
							to1 = to1;
						};
						if (grad.dot(to1) < 0.0) { // bear in mind we move along minus grad
							tmax = -0.33*(to1.dot(to1) / (grad.dot(to1))); // >0
							magnitude = min(tmax, magnitude);
						}
						if (grad.dot(to2) < 0.0) {
							tmax = -0.33*(to2.dot(to2) / (grad.dot(to2))); // >0
							magnitude = min(tmax, magnitude);
						};
					};

					magnitude *= coefficient;	//	Adaptive coefficient. Mysteriously goes small.

					if (magnitude < 0.0)
					{
						//  do a warning and try using 0.15 the nearest neighbour length
						printf("	\t magnitude negative -- ");
					};

					pVertex->pos.x = pVertex->pos.x - magnitude*grad.x;
					pVertex->pos.y = pVertex->pos.y - magnitude*grad.y;

					if (pVertex->flags == 3)
					{
						pVertex->project_to_ins(u);
						pVertex->pos.x = u.x; pVertex->pos.y = u.y;
					}
					if (pVertex->flags == 4)
					{
						pVertex->project_to_radius(u, Outermost_r_achieved);
						pVertex->pos.x = u.x; pVertex->pos.y = u.y;
					}

					// The following code was too complicated and so was replaced by simply comparing to distances across triangles.


					//// Now we make it at most the nearest neighbour distance.
					//minlen = 1.0; // 1 cm - improbably large
					//for (i = 0; i < triangles_len; i++)
					//{
					//	// if triangle is periodic then we need to map other vertices to living nearby....
					//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);
					//	if (pTri->periodic == 0)
					//	{
					//		pTri->PopulatePositions(u0,u1,u2);								
					//	} else {
					//		// periodic triangle								
					//		if (pVertex->pos.x < 0.0) // bit slapdash but hey, unreasonable for periodic tris to cross centre.
					//		{
					//			pTri->MapLeft(u0,u1,u2);
					//		} else {
					//			pTri->MapRight(u0,u1,u2);
					//		};								
					//	};
					//	
					//	if (pVertex == pTri->cornerptr[0])
					//	{
					//		length1 = (u0-u2).modulus();
					//		length2 = (u0-u1).modulus();
					//	} else {
					//		if (pVertex == pTri->cornerptr[1])
					//		{
					//			length1 = (u1-u0).modulus();
					//			length2 = (u1-u2).modulus();
					//		} else {
					//			length1 = (u2-u0).modulus();
					//			length2 = (u2-u1).modulus();
					//		};
					//	};
					//	minlen = min(minlen,min(length1,length2));
					//}; // Done this way because we need contiguous neighbour image which is harder to get from neighbour array. Okay.
					//
					//// We actually keep it down to 33% of the distance to a neighbour :
					//if (magnitude > 0.33*minlen) magnitude = 0.33*minlen;
					//
					//if (magnitude < 0.0)// && (crush_radial_move == false))
					//{
					//	//  do a warning and try using 0.15 the nearest neighbour length
					//	printf("	\t magnitude negative -- ");
					//	magnitude = 0.15*minlen;
					//};

					//// We do not push out beyond the outermost radius: (hopefully unnecessary check)
					//if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y > r_Outermost*r_Outermost)
					//{
					//	real factor = sqrt((pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y) / r_Outermost*r_Outermost);
					//	pVertex->pos.x /= factor;
					//	pVertex->pos.y /= factor;
					//};

					//// Now also verify that this move is not taking us outside the adjacent cells.
					//// This seems pointless: if we did not move more than 33% distance to nearest neighbour then
					//// how can we possibly have exited cells? Perhaps if triangle is extremely flat for some reason. :/

					//// Simpler way then: do not move more than a fraction of normal distance in a triangle!




					//// Use for debug only :

					//magnitude = - magnitude; // old way: coeff on grad not minus grad
					//for (i = 0; i < triangles_len; i++)
					//{
					//	// if triangle is periodic then we need to map other vertices to living nearby....
					//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);

					//	which = 0; c1 = 1;
					//	if (pVertex == pTri->cornerptr[1]) { which = 1; c1 = 0;};
					//	if (pVertex == pTri->cornerptr[2]) which = 2;

					//	// Note that transvec exist for triangle that is mapped left.
					//	if (pTri->periodic == 0) 
					//	{
					//		if (pTri->TestAgainstEdge(pVertex->pos.x,pVertex->pos.y, c1, which, &pDummyTri))
					//		{
					//			// outside!
					//			// how far is it to the edge then?
					//			// Take original position dot normalized tranverse vector

					//			printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

					//			edgenormal = pTri->edge_normal[which]; // DID WE DO NORMALISE TRUE?
					//			edgenormal.Normalise();
					//			xdist = pVertex->eP_Viscous_denominator_x - pTri->cornerptr[c1]->x;
					//			ydist = pVertex->eP_Viscous_denominator_y - pTri->cornerptr[c1]->y;
					//			dist = xdist*edgenormal.x + ydist*edgenormal.y; // may be + or -
					//			// That is the normal distance across the triangle.

					//			// We want to know what multiple of -grad
					//			graddotnormal = grad.x*edgenormal.x+grad.y*edgenormal.y; // dot product of normalized vectors
					//			max = -fabs(dist/graddotnormal);
					//			if (max < magnitude)
					//			{
					//				// error
					//				printf("\nshouldn't be here .. max < magnitude \n");
					//				getch();
					//			} else {
					//				magnitude = max*0.33;
					//				pVertex->pos.x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
					//				pVertex->pos.y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;
					//				if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y > r_Outermost*r_Outermost)
					//				{
					//					printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
					//					// ultimate default:
					//					pVertex->pos.x = pVertex->eP_Viscous_denominator_x;
					//					pVertex->pos.y = pVertex->eP_Viscous_denominator_y;
					//				};
					//				//should be domain interior verts here only.
					//			};
					//		};
					//	} else {
					//		// periodic triangle:
					//		
					//		if (pVertex->pos.x > 0.0)
					//		{
					//			//x_on_left = Anticlockwise.xx*pVertex->pos.x + Anticlockwise.xy*pVertex->pos.y;
					//			//y_on_left = Anticlockwise.yx*pVertex->pos.x + Anticlockwise.yy*pVertex->pos.y;

					//			if (which == 0) pTri->MapRight(uO,u0,u1);
					//			if (which == 1) pTri->MapRight(u0,uO,u1);
					//			if (which == 2) pTri->MapRight(u0,u1,uO);

					//			edgenormal.x = u1.y-u0.y;
					//			edgenormal.y = u0.x-u1.x;
					//			
					//				// want to assess whether pVertex->pos.x,y on same side as pVertex->Viscous_denominator
					//			original_dot = (pVertex->eP_Viscous_denominator_x - u0.x)*edgenormal.x
					//							 + (pVertex->eP_Viscous_denominator_y - u0.y)*edgenormal.y;

					//			new_position_dot = (pVertex->pos.x - u0.x)*edgenormal.x
					//							+ (pVertex->pos.y - u0.y)*edgenormal.y;
					//			if (new_position_dot*original_dot < 0.0)
					//			{
					//				
					//				printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

					//				// not same side of edge.										
					//				edgenormal.Normalise();										
					//				original_dist = (pVertex->eP_Viscous_denominator_x - u0.x)*edgenormal.x
					//											  + (pVertex->eP_Viscous_denominator_y - u0.y)*edgenormal.y;
					//				graddotnormal = grad.x*edgenormal.x + grad.y*edgenormal.y;
					//				
					//				max = -fabs(original_dist/graddotnormal);
					//				if (max < magnitude)
					//				{
					//					// error
					//					printf("\nshouldn't be here .. max < magnitude \n");
					//					getch();
					//				} else {
					//					magnitude = max*0.33;
					//					pVertex->pos.x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
					//					pVertex->pos.y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;
					//					if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y > GlobalMaxVertexRadiusSq)
					//					{
					//						printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
					//						// ultimate default:
					//						pVertex->pos.x = pVertex->eP_Viscous_denominator_x;
					//						pVertex->pos.y = pVertex->eP_Viscous_denominator_y;
					//					};
					//				};
					//			};
					//		} else {
					//			// x is on left so it should be easier
					//			if (pTri->TestAgainstEdge(pVertex->pos.x,pVertex->pos.y, c1, which, &pDummyTri))
					//			{
					//				// outside!
					//				// how far is it to the edge then?
					//				// Take original position dot normalized tranverse vector

					//				printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

					//				if (which == 0) pTri->MapLeft(uO,u0,u1);
					//				if (which == 1) pTri->MapLeft(u0,uO,u1);
					//				if (which == 2) pTri->MapLeft(u0,u1,uO); // may be wedge or tri

					//				edgenormal = pTri->edge_normal[which];
					//				//edgenormal.Normalise();
					//				xdist = pVertex->eP_Viscous_denominator_x - u0.x;
					//				ydist = pVertex->eP_Viscous_denominator_y - u0.y;
					//				original_dist = xdist*edgenormal.x + ydist*edgenormal.y; // may be + or -

					//				graddotnormal = grad.x*edgenormal.x+grad.y*edgenormal.y; // dot product of normalized vectors
					//				// this is how far we travel in direction jim for 1 unit of grad - that's one interpretation
					//			
					//				max = -fabs(original_dist/graddotnormal);

					//				if (max < magnitude)
					//				{
					//					// error
					//					printf("\nshouldn't be here .. max < magnitude \n");
					//					getch();
					//				} else {
					//					magnitude = max*0.33;

					//					pVertex->pos.x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
					//					pVertex->pos.y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;

					//					if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y > GlobalMaxVertexRadiusSq)
					//					{
					//						printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
					//						// ultimate default:
					//						pVertex->pos.x = pVertex->eP_Viscous_denominator_x;
					//						pVertex->pos.y = pVertex->eP_Viscous_denominator_y;
					//					};
					//				};
					//			};
					//		};
					//	};
					//};

					// Bear in mind, this may be across PB so ReturnPointerToTriangle would fail.

					// In the case it crossed PB, we ought to update periodicity of triangles...					
					// Wrap (x,y) also -- but do not wrap the stored version - this allows us to take an average
					pVertex->IonP_Viscous_numerator_x = pVertex->pos.x;
					pVertex->IonP_Viscous_numerator_y = pVertex->pos.y; // store unwrapped first guess positions
																	// ( used for counting up variance and doing periodic tests)

					if (pVertex->pos.x / pVertex->pos.y > GRADIENT_X_PER_Y)
					{
						// went off RH side						
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->IncrementPeriodic();
						};
						newx = Anticlockwise.xx*pVertex->pos.x + Anticlockwise.xy*pVertex->pos.y;
						newy = Anticlockwise.yx*pVertex->pos.x + Anticlockwise.yy*pVertex->pos.y;
						pVertex->pos.x = newx;
						pVertex->pos.y = newy;
					};

					if (pVertex->pos.x / pVertex->pos.y < -GRADIENT_X_PER_Y)
					{
						// went off LH side
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->DecrementPeriodic();
						};
						newx = Clockwise.xx*pVertex->pos.x + Clockwise.xy*pVertex->pos.y;
						newy = Clockwise.yx*pVertex->pos.x + Clockwise.yy*pVertex->pos.y;
						pVertex->pos.x = newx;
						pVertex->pos.y = newy;
					};

					// DEBUG:
					if (pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y < 11.833599999)
					{
						pVertex->pos.x = pVertex->pos.x;
						// absolutely should not be able to happen
						printf("\nTarnation! point swam inside ins! \n");
						getch();
					};

				}; // whether against minimum triangle area grad

			}; // whether iVolley

			++pVertex;
		};

		//		Populate masses again from source mesh;
		// Note that we need to update transvec in order to place points and thus triangles into mesh

		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			pTri->RecalculateEdgeNormalVectors(false);
			pTri++;
		};
		printf("`");
		pSrcMesh->RepopulateCells(this, MASS_ONLY);
		printf(".");

		// We just test whether we gained an improvement in the objective function, and either accept this or not.

		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if (pVertex->iVolley == iVolley)
			{
				objective = 0.0;
				triangles_len = pVertex->triangles.len;
				for (i = 0; i < pVertex->triangles.len; i++)
				{
					pTri = (Triangle *)(pVertex->triangles.ptr[i]);
					objective += (pTri->ion.mass + pTri->neut.mass)*(pTri->ion.mass + pTri->neut.mass);
				};
				//pVertex->ion_pm_Heat_numerator = objective; // store for first guess

				// Decide whether to accept move:
				if (objective < pVertex->e_pm_Heat_denominator)
				{
					// improved
					// (x,y) already set so that's it

					// but in recording move, need to still remember it may have been wrapped across PB
					// so use the unwrapped coords:						
					sum_squared_move_length += (pVertex->IonP_Viscous_numerator_x - pVertex->eP_Viscous_denominator_x)*(pVertex->IonP_Viscous_numerator_x - pVertex->eP_Viscous_denominator_x)
						+ (pVertex->IonP_Viscous_numerator_y - pVertex->eP_Viscous_denominator_y)*(pVertex->IonP_Viscous_numerator_y - pVertex->eP_Viscous_denominator_y);

					accepted++;
					attempted++;
				}
				else {
					// revert to original position
					attempted++;
					pVertex->pos.x = pVertex->eP_Viscous_denominator_x;
					pVertex->pos.y = pVertex->eP_Viscous_denominator_y;
					// twist back any periodic changes:
					if (pVertex->IonP_Viscous_numerator_x / pVertex->IonP_Viscous_numerator_y > GRADIENT_X_PER_Y)
					{
						// in this case we applied increment periodic
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->DecrementPeriodic();
						};
					};
					if (pVertex->IonP_Viscous_numerator_x / pVertex->IonP_Viscous_numerator_y < -GRADIENT_X_PER_Y)
					{
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->IncrementPeriodic();
						};
					};
				};
			};
			++pVertex;
		};


			/*
			// Due to PB, the following became too complicated to be viable !!


			//		Populate masses again from source mesh;

			PopulateMasses(pDestMesh);

			//		Calculate objective functions for 3rd time; now accept the best guess of the 3.

			pVertex = pDestMesh->X;
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
			if (pVertex->iScratch == iVolley)
			{
			objective = 0.0;
			for (i = 0; i < pVertex->triangles.len; i++)
			{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			objective += (pTri->ion.mass + pTri->neut.mass)*(pTri->ion.mass + pTri->neut.mass);
			};

			if (objective < pVertex->e_pm_Heat_denominator)
			{
			if (objective < pVertex->ion_pm_Heat_numerator)
			{
			// second guess is best
			// don't need to make further changes
			} else {
			// first guess is best
			pVertex->pos.x = pVertex->IonP_Viscous_numerator_x;
			pVertex->pos.y = pVertex->IonP_Viscous_numerator_y;
			};
			} else {
			if (pVertex->ion_pm_Heat_numerator < pVertex->e_pm_Heat_denominator)
			{
			// first guess is best
			pVertex->pos.x = pVertex->IonP_Viscous_numerator_x;
			pVertex->pos.y = pVertex->IonP_Viscous_numerator_y;
			} else {
			// failed: both guesses were exprovements.
			pVertex->pos.x = pVertex->eP_Viscous_denominator_x;
			pVertex->pos.y = pVertex->eP_Viscous_denominator_y;
			};
			};

			// Again, if we have crossed PB relative to presently existing (x,y) then we have to update triangles' periodicity
			// But now we have a difficult problem:

			// We already wrapped points :
			// confusing but I think we can miss something here. Suppose we end up reverting to original. It
			// may be on the other side from (x,y) which is a wrapped position.


			// This whole thing is too difficult

			// Let's just make one attempted move and take it or leave it ! (   :-(   )


			if (pVertex->pos.x/pVertex->pos.y > GRADIENT_X_PER_Y)
			{
			// went off RH side
			for (i = 0; i < triangles_len; i++)
			{
			// if triangle is periodic then we need to map other vertices to living nearby....
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			pTri->IncrementPeriodic();
			};
			newx = Anticlockwise.xx*pVertex->pos.x+Anticlockwise.xy*pVertex->pos.y;
			newy = Anticlockwise.yx*pVertex->pos.x+Anticlockwise.yy*pVertex->pos.y;
			pVertex->pos.x = newx;
			pVertex->pos.y = newy;
			};

			if (pVertex->pos.x/pVertex->pos.y < -GRADIENT_X_PER_Y)
			{
			for (i = 0; i < triangles_len; i++)
			{
			// if triangle is periodic then we need to map other vertices to living nearby....
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			pTri->DecrementPeriodic();
			};
			newx = Clockwise.xx*pVertex->pos.x+Clockwise.xy*pVertex->pos.y;
			newy = Clockwise.yx*pVertex->pos.x+Clockwise.yy*pVertex->pos.y;
			pVertex->pos.x = newx;
			pVertex->pos.y = newy;
			};

			sum_squared_move_length += (pVertex->pos.x - pVertex->eP_Viscous_denominator_x)*(pVertex->pos.x - pVertex->eP_Viscous_denominator_x)
			+ (pVertex->pos.y - pVertex->eP_Viscous_denominator_y)*(pVertex->pos.y - pVertex->eP_Viscous_denominator_y);

			}; // whether (pVertex->iScratch == iVolley)
			++pVertex;
			};*/
/*};

	// One round of SwimVertices will attempt to move every vertex once.

	*pAcceptance = ((real)accepted) / ((real)attempted);
	return sqrt(sum_squared_move_length / ((real)(numVertices)));
#endif
*/
	return L2;
}


real inline Get_lnLambda(real n_e, real T_e)
{
	real lnLambda, factor, lnLambda_sq, lnLambda1, lnLambda2;

	static real const one_over_kB = 1.0 / kB_;

	real Te_eV = T_e * one_over_kB;
	real Te_eV2 = Te_eV * Te_eV;
	real Te_eV3 = Te_eV * Te_eV2;

	if (n_e*Te_eV3 > 0.0) {

		lnLambda1 = 23.0 - 0.5*log(n_e / Te_eV3);
		lnLambda2 = 24.0 - 0.5*log(n_e / Te_eV2);
		// smooth between the two:
		factor = 2.0*fabs(Te_eV - 10.0)*(Te_eV - 10.0) / (1.0 + 4.0*(Te_eV - 10.0)*(Te_eV - 10.0));
		lnLambda = lnLambda1 * (0.5 - factor) + lnLambda2 * (0.5 + factor);

		// floor at 2 just in case, but it should not get near:
		lnLambda_sq = lnLambda * lnLambda;
		factor = 1.0 + 0.5*lnLambda + 0.25*lnLambda_sq + 0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
		lnLambda += 2.0 / factor;

		// Golant p.40 warns that it becomes invalid when an electron gyroradius is less than a Debye radius. That is something to worry about if  B/400 > n^1/2 , so looks not a big concern.

		// There is also a quantum ceiling. It will not be anywhere near. At n=1e20, 0.5eV, the ceiling is only down to 29; it requires cold dense conditions to apply.

	}
	else {
		lnLambda = 20.0;
	};
	//if (GlobalDebugRecordIndicator)
	//	Globaldebugdata.lnLambda = lnLambda;
	return lnLambda;
}


real inline Get_lnLambda_ion(real n_ion, real T_ion)
{
	static real const one_over_kB = 1.0 / kB_; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0 / (kB_*kB_*kB_); // multiply by this to convert to eV

	real factor, lnLambda_sq;

	real Tion_eV3 = T_ion * T_ion*T_ion*one_over_kB_cubed;

	real lnLambda = 23.0 - 0.5*log(n_ion / Tion_eV3);

	// floor at 2:
	lnLambda_sq = lnLambda * lnLambda;
	factor = 1.0 + 0.5*lnLambda + 0.25*lnLambda_sq + 0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
	lnLambda += 2.0 / factor;

	return lnLambda;
}
real Estimate_Neutral_Neutral_Viscosity_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_viscosity_nn[9];
	if (T < cross_T_vals[0]) return cross_s_vals_viscosity_nn[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T - cross_T_vals[i - 1]) / (cross_T_vals[i] - cross_T_vals[i - 1]);
	return ppn * cross_s_vals_viscosity_nn[i] + (1.0 - ppn)*cross_s_vals_viscosity_nn[i - 1];
}

void Estimate_Ion_Neutral_Cross_sections(real T, // call with T in electronVolts
	real * p_sigma_in_MT,
	real * p_sigma_in_visc)
{
	if (T > cross_T_vals[9]) {
		*p_sigma_in_MT = cross_s_vals_momtrans_ni[9];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni[9];
		return;
	}
	if (T < cross_T_vals[0]) {
		*p_sigma_in_MT = cross_s_vals_momtrans_ni[0];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni[0];
		return;
	}
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T - cross_T_vals[i - 1]) / (cross_T_vals[i] - cross_T_vals[i - 1]);

	*p_sigma_in_MT = ppn * cross_s_vals_momtrans_ni[i] + (1.0 - ppn)*cross_s_vals_momtrans_ni[i - 1];
	*p_sigma_in_visc = ppn * cross_s_vals_viscosity_ni[i] + (1.0 - ppn)*cross_s_vals_viscosity_ni[i - 1];
	return;
}

real Estimate_Ion_Neutral_MT_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_momtrans_ni[9];
	if (T < cross_T_vals[0]) return cross_s_vals_momtrans_ni[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T - cross_T_vals[i - 1]) / (cross_T_vals[i] - cross_T_vals[i - 1]);
	return ppn * cross_s_vals_momtrans_ni[i] + (1.0 - ppn)*cross_s_vals_momtrans_ni[i - 1];
}

void TriMesh::AdvectPositions_CopyTris(f64 h_use, TriMesh * pDestMesh, f64_vec2 * p_v)
{
	// Inputs:
	// 
	// 

	// Outputs:
	// pDestTri->periodic, pDestTri->neighbours, pDestTri->cornerptr
	// pDestMesh->TriMinorNeighLists, pDestMesh->MajorTriPBC, pDestMesh->TriMinorPBCLists

	Triangle * pTri = T;
	Triangle * pDestTri = pDestMesh->T;
	long iVertex, iTri, tri_len, i;
	long izTri[MAXNEIGH];
	char PBC[6];
	
	for (iTri = 0; iTri < NUMTRIANGLES; iTri++)
	{
		pDestTri->cornerptr[0] = pDestMesh->X + (pTri->cornerptr[0] - X);
		pDestTri->cornerptr[1] = pDestMesh->X + (pTri->cornerptr[1] - X);
		pDestTri->cornerptr[2] = pDestMesh->X + (pTri->cornerptr[2] - X);
		pDestTri->periodic = pTri->periodic;
		pDestTri->neighbours[0] = pDestMesh->T + (pTri->neighbours[0] - T);
		pDestTri->neighbours[1] = pDestMesh->T + (pTri->neighbours[1] - T);
		pDestTri->neighbours[2] = pDestMesh->T + (pTri->neighbours[2] - T);
		++pTri;
		++pDestTri;
	}

	memcpy(pDestMesh->TriMinorNeighLists, TriMinorNeighLists, sizeof(long) * 6 * NUMTRIANGLES);
	memcpy(pDestMesh->MajorTriPBC, MajorTriPBC, sizeof(char)*MAXNEIGH*NUMVERTICES);
	memcpy(pDestMesh->TriMinorPBCLists, TriMinorPBCLists, sizeof(char) * 6 * NUMTRIANGLES);
	// Hmm
	// Guess tri minors are qualitatively different to vertex minors in having 6 neighs.

	Vertex * pVertex = X;
	Vertex * pVertDest = pDestMesh->X;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pVertDest->pos = pVertex->pos + h_use * p_v[iVertex + BEGINNING_OF_CENTRAL];

		if (0) {// iVertex == 11588) {
			printf("\npVertDest->pos %1.9E %1.9E pVertex->pos %1.9E %1.9E h_use %1.9E v %1.9E %1.9E\n\n",
				pVertDest->pos.x, pVertDest->pos.y, pVertex->pos.x, pVertex->pos.y, h_use, p_v[iVertex + BEGINNING_OF_CENTRAL].x,
				p_v[iVertex + BEGINNING_OF_CENTRAL].y);
		}

		pVertDest->CopyLists(pVertex);
		// we could pretty much do memcpy of vertex
		// WE ARE NOT GOING TO WRAP IN THIS ROUTINE.

		++pVertex;
		++pVertDest;
	};

	// Each tri in dest mesh now wants to set its centroid, for the sake of argument.

	// Note that we calculate minor areas in another routine.
	// So go through logic including populating them.
}

void TriMesh::Wrap()
{
	// Detect if vertices need to be wrapped back into domain, and if so,
	// update PBC lists and wrap v for the minors affected.
	plasma_data data;
	long iVertex;
	long tri_len, i;
	long izTri[MAXNEIGH];
	Triangle * pTri;
	long iTri;
	char PBC[6];


	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if ((pVertex->pos.x / pVertex->pos.y < -GRADIENT_X_PER_Y-1.0e-13)
			||
			(pVertex->pos.x / pVertex->pos.y > GRADIENT_X_PER_Y + 1.0e-13))
			// in this case wrap and alert triangles of PB change and update PB lists for vertex neighs, tris.
		{// we never really use fact that vertex is exactly within domain, so use additional threshold to avoid fp flipping.
			memcpy(&data, pData+iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			int flag;
			if (pVertex->pos.x < 0.0) {
				pVertex->pos = Clockwise * pVertex->pos;
				data.pos = Clockwise * data.pos;
				data.vxy = Clockwise * data.vxy;
				data.v_n = Clockwise3 * data.v_n;
				data.B = Clockwise3 * data.B;
				flag = 0;
			} else {
				pVertex->pos = Anticlockwise * pVertex->pos;
				data.pos = Anticlockwise * data.pos;
				data.vxy = Anticlockwise * data.vxy;
				data.v_n = Anticlockwise3 * data.v_n;
				// No Axy to rotate.
				data.B = Anticlockwise3 * data.B;
				flag = 1;
			};

			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				if (flag == 0)
				{
					pTri->DecrementPeriodic();
				} else {
					pTri->IncrementPeriodic();
				}
				// Putting this here:
				pTri->RecalculateCentroid();

				// Refresh pTri minor PB list:
				memset(PBC, 0, sizeof(char) * 6);
				if (pTri->periodic == 0) {
					if ((pTri->neighbours[2]->periodic != 0) && (pTri->cent.x > 0.0))
						PBC[1] = ROTATE_ME_CLOCKWISE;
					if ((pTri->neighbours[0]->periodic != 0) && (pTri->cent.x > 0.0))
						PBC[3] = ROTATE_ME_CLOCKWISE;
					if ((pTri->neighbours[1]->periodic != 0) && (pTri->cent.x > 0.0))
						PBC[5] = ROTATE_ME_CLOCKWISE;
				}
				else {
					// we are periodic tri; bit of guesswork for now:
					if (pTri->cornerptr[0]->pos.x > 0.0)
						PBC[0] = ROTATE_ME_ANTICLOCKWISE;
					if (pTri->neighbours[2]->cent.x > 0.0)
						PBC[1] = ROTATE_ME_ANTICLOCKWISE;
					if (pTri->cornerptr[1]->pos.x > 0.0)
						PBC[2] = ROTATE_ME_ANTICLOCKWISE;
					if (pTri->neighbours[0]->cent.x > 0.0)
						PBC[3] = ROTATE_ME_ANTICLOCKWISE;
					if (pTri->cornerptr[2]->pos.x > 0.0)
						PBC[4] = ROTATE_ME_ANTICLOCKWISE;
					if (pTri->neighbours[1]->cent.x > 0.0)
						PBC[5] = ROTATE_ME_ANTICLOCKWISE;
				};
				memcpy(TriMinorPBCLists + izTri[i], PBC, sizeof(char) * 6);
			};

			// Refresh own PB list for tris:
			memset(MajorTriPBC[iVertex], 0, sizeof(char)*MAXNEIGH);
			for (iTri = 0; iTri < tri_len; iTri++)
			{
				pTri = T + izTri[iTri];
				if (pTri->periodic != 0) {
					// If it was 0 then necessarily this corner does not see the centroid rotated.
					if (pVertex->pos.x > 0.0) {
						// It sees tri as anticlockwise rotated
						MajorTriPBC[iVertex][iTri] = ROTATE_ME_CLOCKWISE;
					};
						// else tri centroid will ALWAYS live on the left so do nothing. If our vertex is on left side then all tris are unrotated.
				};
			};

			memcpy(pData + iVertex + BEGINNING_OF_CENTRAL, &data, sizeof(plasma_data));
		};
		++pVertex;
	};

}

void TriMesh::SetupMajorPBCTriArrays()
{
	// each vertex, create a list of PBC = ROTATE_ME_CLOCKWISE or ROTATE_ME_ANTICLOCKWISE or 0
	// for each triangle in list.
	long iVertex, iTri;
	long izTri[MAXNEIGH];
	long tri_len;
	Triangle * pTri;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		tri_len = pVertex->GetTriIndexArray(izTri);
		// Now the job is to populate MajorTriPBC[iVertex]
		memset(MajorTriPBC[iVertex], 0, sizeof(char)*MAXNEIGH);

		for (iTri = 0; iTri < tri_len; iTri++)
		{
			pTri = T + izTri[iTri];
			if (pTri->periodic != 0) {
				// If it was 0 then necessarily this corner does not see the centroid rotated.
				if (pVertex->pos.x > 0.0) {
					// It sees tri as anticlockwise rotated
					MajorTriPBC[iVertex][iTri] = ROTATE_ME_CLOCKWISE;
				}
				else {
					// Tri centroid will ALWAYS live on the left so do nothing. If our vertex is on left side then all tris are unrotated.
				};
			};
		};

		pData[iVertex + BEGINNING_OF_CENTRAL].pos = pVertex->pos; // ??????????????????????????????????
		++pVertex;
	};
}

void TriMesh::EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists()
{
	EnsureAnticlockwiseTriangleCornerSequences();

	// RUN ONLY AFTER WE GOT ANTICLOCKWISE TRI CORNERS...:
	Triangle * pTri = T;
	long iTri;
	long izNeigh[6];
	char PBC[6];

	for (iTri = 0; iTri < NUMTRIANGLES; iTri++)
	{
		// 1st in list is corner 0
		izNeigh[0] = (pTri->cornerptr[0] - X) + BEGINNING_OF_CENTRAL;
		izNeigh[1] = pTri->neighbours[2] - T;
		izNeigh[2] = (pTri->cornerptr[1] - X) + BEGINNING_OF_CENTRAL;
		izNeigh[3] = pTri->neighbours[0] - T;
		izNeigh[4] = (pTri->cornerptr[2] - X) + BEGINNING_OF_CENTRAL;
		izNeigh[5] = pTri->neighbours[1] - T;
		memset(PBC, 0, sizeof(char) * 6);
		if (pTri->periodic == 0) {
			if ((pTri->neighbours[2]->periodic != 0) && (pTri->cent.x > 0.0)) 
				PBC[1] = ROTATE_ME_CLOCKWISE;
			if ((pTri->neighbours[0]->periodic != 0) && (pTri->cent.x > 0.0))
				PBC[3] = ROTATE_ME_CLOCKWISE;
			if ((pTri->neighbours[1]->periodic != 0) && (pTri->cent.x > 0.0)) 
				PBC[5] = ROTATE_ME_CLOCKWISE;
		} else {
			// we are periodic tri; bit of guesswork for now:
			if (pTri->cornerptr[0]->pos.x > 0.0) 
				PBC[0] = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->neighbours[2]->cent.x > 0.0) 
				PBC[1] = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[1]->pos.x > 0.0) 
				PBC[2] = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->neighbours[0]->cent.x > 0.0)
				PBC[3] = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[2]->pos.x > 0.0) 
				PBC[4] = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->neighbours[1]->cent.x > 0.0) 
				PBC[5] = ROTATE_ME_ANTICLOCKWISE;
		};
		
	//	if (iTri == 92250) {
	//		printf("neigh %d %d %d PBC1 %d PBC3 %d PBC5 %d periodic %d pos.x %1.9E : %1.9E %1.9E %1.9E\n",
	//			pTri->neighbours[2] - T, pTri->neighbours[0] - T, pTri->neighbours[1] - T,
	//			PBC[1], PBC[3], PBC[5], pTri->periodic, pTri->cent.x, pTri->neighbours[2]->cent.x,
	//			pTri->neighbours[0]->cent.x, pTri->neighbours[1]->cent.x);
	//		getch();
	//	};

		memcpy(TriMinorNeighLists[iTri],izNeigh, sizeof(long) * 6);
		memcpy(TriMinorPBCLists[iTri], PBC, sizeof(char) * 6);

		pData[iTri].pos = pTri->cent; // ????????????????????????????????????
		++pTri;
	}

}


/*void TriMesh::CalculateIonisationRates(f64 h_use, NTrates NTaddition_array[])
{
	NTrates rates;

	plasma_data * pdata = pData + BEGINNING_OF_CENTRAL;
	plasma_data ourdata;

	for (iVertex = 0; iVertex < BEGINNING_OF_CENTRAL; iVertex++)
	{
		memcpy(&ourdata, pData + iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));

		f64 TeV = ourdata.Te / kB;
		f64 sqrtT = sqrt(TeV);

		f64 temp = 1.0e-5*exp(-13.6 / TeV)/ (13.6*(6.0*13.6 + TeV));
		// Let h n n_n S be the ionising amount,
		// h n S is the proportion of neutrals! Make sure we do not run out!
		f64 hnS = (h_use*ourdata.n*TeV*temp)/
			(sqrtT+h_use* ourdata.n_n*ourdata.n*temp*SIXTH*13.6);
		f64 ionise_rate = AreaMajor * ourdata.n_n*hnS / (h_use*(1 + hnS));
				// ionise_amt / h

		rates.N = ionise_rate;
		rates.Nn = -ionise_rate;

		// Let nR be the recombining amount, R is the proportion.
		
		f64 Ttothe5point5 = sqrtT * TeV * TeV*TeV * TeV*TeV;
		f64 hR = h_use*(ourdata.n * ourdata.n*8.75e-27*TeV) /
			(Ttothe5point5 + h_use*2.25*TWOTHIRDS*13.6*ourdata.n*ourdata.n*8.75e-27);

		f64 recomb_rate = AreaMajor * ourdata.n * hR / h_use; // could reasonably again take hR/(1+hR) for n_k+1
		rates.N -= recomb_rate;
		rates.Nn += recomb_rate;
		
		rates.NeTe += -TWOTHIRDS*13.6*kB*rates.N + 0.5*ourdata.Tn*ionise_rate;
		rates.NiTi += 0.5*ourdata.Tn*ionise_rate;
		rates.NnTn += (ourdata.Te + ourdata.Ti)*recomb_rate;

		// Maybe this is wrong:
		// do we want to add it instead of replacing?
		memcpy(NTaddition_array + iVertex, &rates, sizeof(NTrates));
		
		// WE HAVE NEGLECTED THE TRANSFER OF MOMENTUM BETWEEN SPECIES AND
		// SO WE ALSO NEGLECTED THE CORRESPONDING HEATING RATE.

		// Seems we insisted on doing change in N which requires knowing area.
		++pdata;
	}
}*/

void TriMesh::Advance(TriMesh * pDestMesh, TriMesh * pHalfMesh)
{
	long iMinor, iSubstep;
	// pre-run: 

	
	// NOTE: For GPU we actually changed things so that we do not use shards except for
	// advection rates.
		

	// These should be done every time we alter trimesh to change what is a neighbour of what:
	// EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	// SetupMajorPBCTriArrays();
	// Otherwise let them be copied --- grr we do need 1 PBC array per TriMesh ??
	// Or shall we just assume we delaunerise and update it then - yes that works just as well.

	// FOR NOW:
	// Let n_central = n_cent in shard model (not quite right)
	// Let n_tri = simple average of what it is being assigned by shard models.
	//	pHalfMesh->Infer_velocity_from_mass_and_momentum(); // super careful: conserving mom?
	// can see how if we wanted to store n_minor we could be using a really good shard-integrated
	// estimate both times. This is not where effort should be going right now.
	// 
	// OK... the bad news ... to get ALL masses to add up, in computing v to conserve
	// momentum ... we have to acknowledge that n_central is not the same as n_major, or#
	// not quite the same.
	// The relevant n_cent is basically what we stored in the shard model.
	// But we haven't arranged storage for n_cent in plasma_data and still retain vertex cell density.
	// We have to choose whether in Accelerate we'll load in n_shards.cent for vertex n. We should ideally.

	// -----------------------------------------------
#define USE_N_MAJOR

	static int runs = 0;
	CalculateOverallVelocities(p_v); // vertices first, then average to tris
	memset(pHalfMesh->pData, 0, sizeof(plasma_data)*NMINOR);
	AdvectPositions_CopyTris(0.5*TIMESTEP,pHalfMesh, p_v); 

	Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // call before CreateShardModel 
	CreateShardModelOfDensities_And_SetMajorArea();// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists

#ifndef USE_N_MAJOR
	InferMinorDensitiesFromShardModel(); 
#endif
	// together with n_cent they may not be mass-consistent but they are our best est.
	// Really we need to keep separate n_cent and n_major.

	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);	
	memset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);
	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);

	this->AccumulateAdvectiveMassHeatRateOld(p_v, NTadditionrates);

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	this->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);
	this->Add_ViscousMomentumFluxRates(AdditionalMomRates); // should also add to NTadditionrates.
	this->AccumulateDiffusiveHeatRateAndCalcIonisation(0.5*h,NTadditionrates); // Wants minor n,T and B
	
	this->AdvanceDensityAndTemperature(0.5*h, this, pHalfMesh, NTadditionrates);// p_div_v_neut, p_div_v);
	pHalfMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); 
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// We are going to want to introduce 2nd degree approx to get n,T desired on tris.
	// Now let's set up the accel move to half-time which will provide us input of v to the full n,T move.
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
	pHalfMesh->CreateShardModelOfDensities_And_SetMajorArea();
#ifndef USE_N_MAJOR
	pHalfMesh->InferMinorDensitiesFromShardModel(); // (At the moment just repopulating tri minor n.)
#endif

	// Get suitable v to use for resistive heating:
	this->Accelerate2018(0.5*h, this, pHalfMesh, evaltime + 0.5*h, false, true); 
	// current is attained at end of step

	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	pHalfMesh->CalculateOverallVelocities(p_v); // vertices first, then average to tris
	memset(pDestMesh->pData, 0, sizeof(plasma_data)*NMINOR);
	AdvectPositions_CopyTris(h, pDestMesh, p_v);
	
	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);
	pHalfMesh->AccumulateAdvectiveMassHeatRateOld(p_v, NTadditionrates);

	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);
	this->AntiAdvectAzAndAdvance(0.5*h, this, GradAz, pHalfMesh);
	pHalfMesh->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);

	pHalfMesh->Add_ViscousMomentumFluxRates(AdditionalMomRates);
	// Where is B populated? on pHalfMesh

	
	pHalfMesh->AccumulateDiffusiveHeatRateAndCalcIonisation(h, NTadditionrates); // Wants minor n,T and B
	// Problem: tries to use pHalfMesh->B which has not been populated.
	// Discovered an error on CPU.

	AdvanceDensityAndTemperature(h, pHalfMesh, pDestMesh, NTadditionrates);
	pDestMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // UPGRADE TO 2ND DEGREE
	
#ifndef USE_N_MAJOR
	pDestMesh->CreateShardModelOfDensities_And_SetMajorArea();
	pDestMesh->InferMinorDensitiesFromShardModel();
#endif

	// The thing is that we didn't calculate v on pDestMesh ...
	// Quicker for now is just to use dv/dt from half-time.
	// Ultimately we may revisit and find a way to take for pDestMesh and to tween at least the aTP.

	// Note: Recalc visc is expensive but roughly speaking we know the rate of momentum diffusion
	// is similar to that for heat conduction.

	// Now ready to do acceleration substeps:
	f64 starttime = evaltime;

	if (runs % 10 == 0)
	{
		// BACKWARD STEPS:

		pHalfMesh->GetLapCoeffs(); 

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			InterpolateVarsAndPositions(pHalfMesh, pDestMesh, (evaltime - starttime) / TIMESTEP);
			//pHalfMesh->GetLapFromCoeffs(Az_array, LapAzArray);
			
			pHalfMesh->GetLap(Az_array, LapAzArray);
			
			if (iSubstep == 0) {

				this->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, true,
					(iSubstep == SUBCYCLES-1));
			// true == bButReportCoefficientOnLapAz --- create Azdot0, gamma
			// true also implies DO NOT SAVE data_1 into DestMesh
				
			// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
			// Now we will wanna create each eqn for Az with coeffs on neighbour values.
			// So we need a func called "GetLapCoefficients".

			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]
		
			// Seed:
				for (iMinor = 0; iMinor < NMINOR; iMinor++)
					Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*this->pData[iMinor].Azdot + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
				pHalfMesh->JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
				memcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR);
				pHalfMesh->GetLap(Az_array, LapAzArray);
				this->Accelerate2018(SUBSTEP, pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false,
					(iSubstep == SUBCYCLES-1)); // Lap Az now given.

			} else {
				
				pDestMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, true,
					(iSubstep == SUBCYCLES-1)
					);
				Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*pDestMesh->pData[iMinor].Azdot + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
				
				pHalfMesh->JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
				
				memcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR);
				pHalfMesh->GetLap(Az_array, LapAzArray);
				pDestMesh->Accelerate2018(SUBSTEP, pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false,
					(iSubstep == SUBCYCLES - 1));
				
				// Oops

				// This is not receiving "this" so that it can interpolate n.
				// Can we interpolate from pHalfMesh????????????????????????

				// Better if we change to go back and forth?

				// pHalfMesh->n should be pretty jolly accurate to the n we ought to use.
				// Can we set it just to use that?

				// We should try n keep things simple. Using wrong values isn't simpler.



			};
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
				pDestMesh->pData[iMinor].Az = Az_array[iMinor];
			
			evaltime += 0.5*SUBSTEP;
			// more advanced implicit could be possible and effective.
		}
	} else {
		Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			InterpolateVarsAndPositions(pHalfMesh, pDestMesh, (evaltime - starttime) / TIMESTEP);
			// let n,T,x be interpolated on to pHalfMesh. B remains what we populated there.

			// ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / TIMESTEP);
			// Have a look how AMR is created.
			pHalfMesh->GetLap(Az_array, LapAzArray); // pHalfMesh has the positions to take Lap.
			/*
			f64 maxLap = -1.0e100;
			f64 minLap = 1.0e100;
			for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
			{
				if (LapAzArray[iMinor] > maxLap) maxLap = LapAzArray[iMinor];
				if (LapAzArray[iMinor] < minLap) minLap = LapAzArray[iMinor];
			}
			printf("substep %d Lap max %1.8E Lap min %1.8E tris\n", iSubstep, maxLap, minLap);
			maxLap = -1.0e100;
			minLap = 1.0e100;
			for (; iMinor < NMINOR; iMinor++)
			{
				if (LapAzArray[iMinor] > maxLap) maxLap = LapAzArray[iMinor];
				if (LapAzArray[iMinor] < minLap) minLap = LapAzArray[iMinor];
			}
			printf("substep %d Lap max %1.8E Lap min %1.8E verts\n", iSubstep, maxLap, minLap);
			*/
			if (iSubstep == 0) {
				this->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP,false,
					(iSubstep==SUBCYCLES-1)
					);

				// So we have a problem with this. We are using n from pDestMesh for the end
				// of the substep, but in fact it's for the end of the whole timestep.
				// What gives?

				// We would like to send n_1 = an interpolation between *this->n and *pDestMesh->n
				// We are sending *this so we can interpolate if we send a coefficient.
				

			}
			else {
				// Thereafter just update pDestMesh since we can discard the old values of v, Adot.
				pDestMesh->Accelerate2018(SUBSTEP, //ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP,false,
					(iSubstep == SUBCYCLES - 1)
					);
				// pHalfMesh is pUseMesh - tick, it contains B
			};

			if (iSubstep < SUBCYCLES - 1) {
				pDestMesh->AdvanceAz(SUBSTEP, ROCAzduetoAdvection, Az_array);
			}
			else {
				pDestMesh->FinalStepAz(SUBSTEP*0.5, ROCAzduetoAdvection, pDestMesh, Az_array); // just catch up to the final time
			};
			evaltime += 0.5*SUBSTEP;

			// FOR GRAPHING ONLY:
			if (iSubstep < SUBCYCLES - 1) {
				for (iMinor = 0; iMinor < NMINOR; iMinor++)
				{
					pDestMesh->pData[iMinor].Az = Az_array[iMinor];
				};
			};
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				pDestMesh->pData[iMinor].temp.x = LapAzArray[iMinor];
			};

			//RefreshGraphs(*pDestMesh, 10000); // sends data to graphs AND renders them
			//Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			//InvalidateRect(hWnd, 0, 0);
			//UpdateWindow(hWnd);
			printf("substep %d evaltime %1.5E \n", iSubstep, evaltime);
			//getch();

			// Think carefully. The ROC for n includes the advection of the mesh rel to the fluid.
			// The ROC for v should, likewise.
			// We have ignored anti-advection for A and Adot : correct? 
			// But they BOTH SHOULD APPLY.
			// The move is VERY small so we could do both in 1 fell swoop and be using Adotz in a different true
			// location ... 
			// I prefer for right now to do the anti-advection throughout.
			// What about floating-point? nvm - just a choice for now
		};
	};
	printf("evaltime %1.5E \n", evaltime); 
	printf("-----------------\n");
	
	//this->AntiAdvectAzAndAdvance(h, pHalfMesh, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pHalfMesh->AntiAdvectAzAndAdvance(h*0.5, pDestMesh, GradAz, pDestMesh);
	
	pDestMesh->Wrap();
	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, h);
	
	// For graphing Lap Az:
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		pDestMesh->pData[iMinor].temp.x = LapAzArray[iMinor];
	};

	//FILE * fp;
	char buffer[256];
	plasma_data data;
	sprintf(buffer, "StatsNEW.txt");
	//fp = fopen(buffer, "a");
	int i;
	Vertex * pVertex;
	long tri_len, j;
	long izTri[MAXNEIGH];
	/*
	fprintf(fp, "Step %d\n", runs);
	for (i = 0; i < 14; i++)
	{
		if (i == 12) {
			iMinor = BEGINNING_OF_CENTRAL + VERT1;
		}
		else {
			if (i == 13) {
				iMinor = BEGINNING_OF_CENTRAL + VERT2;
			}
			else {
				iMinor = TriFiles[i];
			}
		}	
		memcpy(&data, pDestMesh->pData + iMinor, sizeof(plasma_data));

		// Think no - think want a different way round, put all in 1 file.
		fprintf(fp, "%d x %1.14E y %1.14E r %1.14E n %1.14E vez %1.14E Az %1.14E Azdot %1.14E Lap_A %1.14E",
			iMinor, data.pos.x, data.pos.y, data.pos.modulus(), data.n, data.vez, data.Az, data.Azdot, data.temp.x);

		if (i >= 12) {
			pVertex = X + iMinor - BEGINNING_OF_CENTRAL;
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (j = 0; j < tri_len; j++)
			{
				fprintf(fp, " %d ", izTri[j]);
			}
		}
		fprintf(fp, "\n");

	}
	fprintf(fp, "\n\n");
	fclose(fp);
	*/
	runs++;
}


void TriMesh::JLS_for_Az_bwdstep(int iterations, f64 h_use)
{
	int iIteration;
	
	// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
	// Now we will wanna create each eqn for Az with coeffs on neighbour values.
	// So we need a func called "GetLapCoefficients".

	// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
	// Calculate regressor x_Jacobi from eps/coeff_on_A_i
	// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
	// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]

	// populate Az_array with k+1 values

	// INPUTS:
	// Az_array[] = previous Az
	// LapCoeffself[]
	// gamma[], Azdot0[]
	
	// OUTPUTS:
	// Az_array_next[]
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 beta, L2eps;
	Triangle * pTri;
	// 1. Create regressor:
	for (iIteration = 0; iIteration < iterations; iIteration++)
	{
		GetLap(Az_array_next, Lap_Aznext);
		pTri = T;
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			// Here it immediately turns out we do need GetLapCoeff.
			if ((iMinor < BEGINNING_OF_CENTRAL) &&
				((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)))
			{
				epsilon[iMinor] = Lap_Aznext[iMinor];
				Jacobi_x[iMinor] = -epsilon[iMinor] / LapCoeffself[iMinor];
			} else {
				epsilon[iMinor] = Az_array_next[iMinor] - h_use * gamma[iMinor] * Lap_Aznext[iMinor] - Az_array[iMinor] - h_use * Azdot0[iMinor];
				Jacobi_x[iMinor] = -epsilon[iMinor] / (1.0 - h_use * gamma[iMinor] * LapCoeffself[iMinor]);
			};
			
			++pTri;
		};
		GetLap(Jacobi_x, Lap_Jacobi);

		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		pTri = T;
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			if ((iMinor < BEGINNING_OF_CENTRAL) &&
				((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)))
			{
				depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
			} else {
				depsbydbeta = (Jacobi_x[iMinor] - h_use * gamma[iMinor] * Lap_Jacobi[iMinor]);
			};
			sum_eps_deps_by_dbeta += epsilon[iMinor] * depsbydbeta;
			sum_depsbydbeta_sq += depsbydbeta * depsbydbeta;
			sum_eps_eps += epsilon[iMinor] * epsilon[iMinor];
			++pTri;
		};
		beta = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		L2eps = sqrt(sum_eps_eps / (real)NMINOR);
		printf(" CPU[ beta %1.14E L2eps %1.14E ] ", beta, L2eps);

		/*
		FILE * fp = fopen("regress.txt", "a");
		fprintf(fp, "\n\n");
		fprintf(fp, "index epsilon depsbydbeta regressor Azdot0 gamma LapJacobi Aznext Lap_Aznext Azk\n");
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			fprintf(fp, "%d %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E \n",
					iMinor, epsilon[iMinor],
					(Jacobi_x[iMinor] - h_use * gamma[iMinor] * Lap_Jacobi[iMinor]),
					Jacobi_x[iMinor],
					Azdot0[iMinor],
					gamma[iMinor],
					Lap_Jacobi[iMinor],
					Az_array_next[iMinor],
					Lap_Aznext[iMinor],
					Az_array[iMinor]
				);
		};
		fclose(fp);
		*/

		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			Az_array_next[iMinor] += beta * Jacobi_x[iMinor];
		};

		// Try resetting frills here and ignoring in calculation:
		pTri = T;
		for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
		{
			if ((pTri->u8domain_flag == INNER_FRILL) ||
				(pTri->u8domain_flag == OUTER_FRILL))
				Az_array_next[iMinor] = Az_array_next[pTri->neighbours[0] - T];
			++pTri;
		};
	};

	printf("\n\n");
	pTri = T;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		if ((pTri->u8domain_flag == INNER_FRILL) ||
			(pTri->u8domain_flag == OUTER_FRILL))
			Az_array_next[iMinor] = Az_array_next[pTri->neighbours[0] - T];
		++pTri;
	};

}
void TriMesh::InterpolateVarsAndPositions(TriMesh * pTargetMesh, TriMesh * pEndMesh, f64 ppn)
{
	long iMinor;
	plasma_data * pdestdata, *pstartdata, *penddata;
	plasma_data localstart, localend, local;
	pstartdata = pData;
	penddata = pEndMesh->pData;
	pdestdata = pTargetMesh->pData;
	real oneminus = 1.0 - ppn;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		memcpy(&localstart, pstartdata, sizeof(plasma_data));
		memcpy(&localend, penddata, sizeof(plasma_data));
		memcpy(&local, pdestdata, sizeof(plasma_data)); // avoid overwriting other data by accident

		local.n = ppn * localend.n + oneminus * localstart.n;
		local.n_n = ppn * localend.n_n + oneminus * localstart.n_n;
		local.pos = ppn * localend.pos + oneminus * localstart.pos;
		local.Ti = ppn * localend.Ti + oneminus * localstart.Ti;
		local.Te = ppn * localend.Te + oneminus * localstart.Te;
		local.Tn = ppn * localend.Tn + oneminus * localstart.Tn;

		memcpy(pdestdata,&local, sizeof(plasma_data));
		++pstartdata;
		++penddata;
		++pdestdata;
	};

}

void TriMesh::Add_ViscousMomentumFluxRates(three_vec3 * AdditionalMomRates) {} // 0 for now

void TriMesh::AntiAdvectAzAndAdvance(f64 h_use, TriMesh * pUseMesh, f64_vec2 Grad_Az[NMINOR], TriMesh * pDestMesh)
{
	// Inputs:
	// GradAz
	// AreaMinorArray
	// pUseMesh data.Azdot
	//   pDestMesh minor pos & minor pos

	// Outputs:
	// pDestMesh->pdata[iMinor].Az

	long iMinor;
	plasma_data data, usedata;
	f64_vec2 grad_Az, move;
	f64 Az;

	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		//memcpy(&data, pData + iMinor, sizeof(plasma_data));

		// Behaviour at edges: 
		// we can advance A on innermost vertices knowing that we took contributions to LapAz only from the domain side.
		// See how/if we are doing that. ****
		// Same for outermost.

		// Are we being careful about how we deal with ins tris when we advance Azdot with current, weakly ?
		// Think we are not. ****

		Az = (pData + iMinor)->Az;
		grad_Az = Grad_Az[iMinor] ;
		move = pDestMesh->pData[iMinor].pos - pData[iMinor].pos;

		Az += h_use * (pUseMesh->pData + iMinor)->Azdot + move.dot(grad_Az);
		// We want Az at the new position so it has changed by grad Az . move

		pDestMesh->pData[iMinor].Az = Az;
		// Hopefully for frills nothing that mad happens.
	};
	Triangle * pTri = T;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		// Set all frills Az to equal the adjacent Az.
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL))
		{
			pDestMesh->pData[iMinor].Az = pDestMesh->pData[pTri->neighbours[0] - T].Az;
		};
		++pTri;
	};
}



void TriMesh::EnsureAnticlockwiseTriangleCornerSequences()
{
	Triangle * pTri = T;
	f64_vec2 u0, u1, u2;
	f64 Area;
	long iTri;
	Triangle * store_neigh;
	Vertex * store_ptr;
	f64_vec2 store_edge;
	for (iTri = 0; iTri < NUMTRIANGLES; iTri++)
	{
		pTri->MapLeftIfNecessary(u0, u1, u2);
		// Test if anticlockwise:
		
		Area = 0.5*((u0.x + u1.x)*(u1.y - u0.y) + (u1.x + u2.x)*(u2.y - u1.y) + (u2.x + u0.x)*(u0.y - u2.y));
		// the x-component of outward normal is y_anti-y_clock
		
		// If not, we want to switch over cornerptr and this means also switch over neighbour indices
		if (Area < 0.0) {
			store_ptr = pTri->cornerptr[1];
			pTri->cornerptr[1] = pTri->cornerptr[2];
			pTri->cornerptr[2] = store_ptr;
			store_neigh = pTri->neighbours[1];
			pTri->neighbours[1] = pTri->neighbours[2];
			pTri->neighbours[2] = store_neigh;
			// do edge_normal vector in case we ever use that still
			store_edge = pTri->edge_normal[1];
			pTri->edge_normal[1] = pTri->edge_normal[2];
			pTri->edge_normal[2] = store_edge;
		};

		++pTri;
	}; 
	// MUST call before we do anything and every time we mess with the mesh.

}


void TriMesh::CalcUpwindDensity_on_tris(f64 * p_n_upwind, f64 * p_nn_upwind, f64_vec2 * p_v_overall_tris)
{
	long iTri, iVertex;
	Triangle * pTri = T;
	plasma_data data;
	long izTri[MAXNEIGH];
	Vertex * pVertex;
	short tri_len;
	f64 n0, n1, n2, nn0, nn1, nn2, dot0, dot1, dot2;
	f64_vec2 edge_normal0, edge_normal1, edge_normal2;
	nvals n_upwind;

	for (iTri = 0; iTri < NUMTRIANGLES; iTri++)
	{
		if ((pTri->u8domain_flag == DOMAIN_TRIANGLE) || (pTri->u8domain_flag == CROSSING_INS))
		{
			memcpy(&data, &(pData[iTri]), sizeof(plasma_data));

			iVertex = pTri->cornerptr[0] - X;
			pVertex = pTri->cornerptr[0];
			tri_len = pVertex->GetTriIndexArray(izTri);
			int i = 0;
			while (izTri[i] != iTri) i++;
			n0 = n_shards[iVertex].n[i];
			nn0 = n_shards_n[iVertex].n[i];

			iVertex = pTri->cornerptr[1] - X;
			pVertex = pTri->cornerptr[1];
			tri_len = pVertex->GetTriIndexArray(izTri);
			i = 0;
			while (izTri[i] != iTri) i++;
			n1 = n_shards[iVertex].n[i];
			nn1 = n_shards_n[iVertex].n[i];

			iVertex = pTri->cornerptr[2] - X;
			pVertex = pTri->cornerptr[2];
			tri_len = pVertex->GetTriIndexArray(izTri);
			i = 0;
			while (izTri[i] != iTri) i++;
			n2 = n_shards[iVertex].n[i];
			nn2 = n_shards_n[iVertex].n[i];

			f64_vec2 relv = data.vxy - p_v_overall_tris[iTri];

			if (pTri->u8domain_flag == CROSSING_INS) {
				int number_within = (n0 > 0.0) ? 1 : 0 + (n1 > 0.0) ? 1 : 0 + (n2 > 0.0) ? 1 : 0;
				if (number_within == 1) {
					n_upwind.n = n0 + n1 + n2;
					n_upwind.n_n = nn0 + nn1 + nn2;
				}
				else {
					// quick way not upwind:
					n_upwind.n = 0.5*( n0 + n1 + n2);
					n_upwind.n_n = 0.5*( nn0 + nn1 + nn2);
				};
			} else {
				// Get contiguous neighbour cent:
				f64_vec2 pos0 = pData[pTri->neighbours[0] - T].pos;
				if (pTri->periodic == 0) {
					if ((pTri->neighbours[0]->periodic != 0) && (data.pos.x > 0.0))
						pos0 = Clockwise*pos0;
				}
				else {
					if ((pTri->neighbours[0]->periodic == 0) && (pos0.x > 0.0))
						pos0 = Anticlockwise*pos0;
				}

				f64_vec2 pos1 = pData[pTri->neighbours[1] - T].pos;
				if (pTri->periodic == 0) {
					if ((pTri->neighbours[1]->periodic != 0) && (data.pos.x > 0.0))
						pos1 = Clockwise*pos1;
				}
				else {
					if ((pTri->neighbours[1]->periodic == 0) && (pos1.x > 0.0))
						pos1 = Anticlockwise*pos1;
				}

				f64_vec2 pos2 = pData[pTri->neighbours[2] - T].pos;
				if (pTri->periodic == 0) {
					if ((pTri->neighbours[2]->periodic != 0) && (data.pos.x > 0.0))
						pos2 = Clockwise*pos2;
				}
				else {
					if ((pTri->neighbours[2]->periodic == 0) && (pos2.x > 0.0))
						pos2 = Anticlockwise*pos2;
				}

				edge_normal0.x = pos0.y - data.pos.y;
				edge_normal0.y = data.pos.x - pos0.x;
				edge_normal1.x = pos1.y - data.pos.y;
				edge_normal1.y = data.pos.x - pos1.x;
				edge_normal2.x = pos2.y - data.pos.y;
				edge_normal2.y = data.pos.x - pos2.x;

				dot0 = relv.dot(edge_normal0);
				dot1 = relv.dot(edge_normal1);
				dot2 = relv.dot(edge_normal2);

		//		if (iTri == CHOSEN) printf("%d CPU: nn0 %1.14E nn1 %1.14E nn2 %1.14E \n"
		//			"dot0 %1.14E dot1 %1.14E dot2 %1.14E \n"
		//			"relv %1.14E %1.14E \n"
		//			"edge_normal0 %1.14E %1.14E \n",
		//			CHOSEN, nn0, nn1, nn2, dot0, dot1, dot2, relv.x, relv.y, edge_normal0.x, edge_normal0.y);
				
				f64 numerator = 0.0;
				if (dot0*dot0 + dot1*dot1 + dot2*dot2 == 0.0) {
					n_upwind.n = THIRD*(n0 + n1 + n2);
					n_upwind.n_n = THIRD*(nn0 + nn1 + nn2);
		//			if (iTri == CHOSEN) printf("Got to here. CPU. n = %1.14E \n", n_upwind.n);
				}
				else {
					n_upwind.n = 0.0;
					n_upwind.n_n = 0.0;
					if (dot0 > 0.0) {
						n_upwind.n += dot0*n2;
					} else {
						dot0 = -dot0;
						n_upwind.n += dot0*n1;
					};
					if (dot1 > 0.0) {
						n_upwind.n += dot1*n0;
					} else {
						dot1 = -dot1;
						n_upwind.n += dot1*n2;
					};
					if (dot2 > 0.0) {
						n_upwind.n += dot2*n1;
					} else {
						dot2 = -dot2;
						n_upwind.n += dot2*n0;
					};
					n_upwind.n /= (dot0 + dot1 + dot2);
					
					relv = data.v_n.xypart() - p_v_overall_tris[iTri];

					dot0 = relv.dot(edge_normal0);
					dot1 = relv.dot(edge_normal1);
					dot2 = relv.dot(edge_normal2);

					if (dot0 > 0.0) {
						n_upwind.n_n += dot0*nn2;
					}
					else {
						dot0 = -dot0;
						n_upwind.n_n += dot0*nn1;
					};
					if (dot1 > 0.0) {
						n_upwind.n_n += dot1*nn0;
					}
					else {
						dot1 = -dot1;
						n_upwind.n_n += dot1*nn2;
					};
					if (dot2 > 0.0) {
						n_upwind.n_n += dot2*nn1;
					}
					else {
						dot2 = -dot2;
						n_upwind.n_n += dot2*nn0;
					};
					
					n_upwind.n_n /= (dot0 + dot1 + dot2);
					
				}
			};
			p_n_upwind[iTri] = n_upwind.n;
			p_nn_upwind[iTri] = n_upwind.n_n;

		} else {
			p_n_upwind[iTri] = 0.0;
			p_nn_upwind[iTri] = 0.0;
		};
		++pTri;
	};

}


void TriMesh::InferMinorDensitiesFromShardModel()
{
	// Inputs:
	// n_shards

	// Outputs:
	// pdata->n for tri minors only

	// Would love to use n_cent as the n in Accelerate but
	// we don't have separate storage. Would have to fetch sep??

	// first set the tri minor densities to 0
	long iMinor, iVertex, i;
	long izTri[MAXNEIGH];
	long tri_len;
	plasma_data * ptridata;
	Triangle * pTri;
	Vertex * pVertex = X;
	plasma_data * pdata = pData;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		pdata->n = 0.0;
		pdata->n_n = 0.0;
		++pdata;
	}

	// now accumulate them as 1/3 simple average of assigned value in shard models :

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX) {
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				ptridata = pData + izTri[i];
				pTri = T + izTri[i];
				if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
					ptridata->n += THIRD * n_shards[iVertex].n[i];
					ptridata->n_n += THIRD * n_shards_n[iVertex].n[i];
					
				} else {
					if (pTri->u8domain_flag == CROSSING_INS) {
						int iAbove = 0;
						f64_vec2 pos0 = pTri->cornerptr[0]->pos;
						f64_vec2 pos1 = pTri->cornerptr[1]->pos;
						f64_vec2 pos2 = pTri->cornerptr[2]->pos;
						
						if (pos0.x*pos0.x + pos0.y*pos0.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
							iAbove++;
						if (pos1.x*pos1.x + pos1.y*pos1.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
							iAbove++;
						if (pos2.x*pos2.x + pos2.y*pos2.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
							iAbove++;
						
						ptridata->n += n_shards[iVertex].n[i]/(f64)iAbove;
						ptridata->n_n += n_shards_n[iVertex].n[i]/(f64)iAbove;
						
					}
				}

			};
		};
		
		//pdata->n = n_shards[iVertex].n_cent;
		//pdata->n_n = n_shards_n[iVertex].n_cent;

		// ^^ in GPU case we did set this.

		++pVertex;
		++pdata;
	};	


}
/*void TriMesh::Infer_velocity_from_mass_and_momentum(ShardModel n_shards_n[NUMVERTICES],
	ShardModel n_shards[NUMVERTICES])
{
	// The assigned "minor densities" are in triangle n and in n_cent
	// The assigned "Nv" is in velocity storage space
	plasma_data data;
	long iMinor;

	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		memcpy(&data, pData + iMinor, sizeof(plasma_data));

		AreaMinor = ?
		data.v_n /= (data.n_n*AreaMinor);
		data.vxy /= (data.n*AreaMinor);
		data.viz /= (data.n*AreaMinor);
		data.vez /= (data.n*AreaMinor);

		memcpy(pData + iMinor, &data, sizeof(plasma_data));
	};
	// Yeah I am not so sure about all this way of doing it. *** Are we sure we 
	// are not changing if the advective flux is negligible? THIS IS THE KEY QUESTION.

	long iVertex = 0;
	for (; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, pData + iMinor, sizeof(plasma_data));
		f64 n_n = n_shards_n[iVertex].n_cent;
		f64 n = n_shards[iVertex].n_cent;

		AreaMinor = ?

		data.v_n /= (n_n*AreaMinor);
		data.vxy /= (n*AreaMinor);
		data.viz /= (n*AreaMinor);
		data.vez /= (n*AreaMinor);

		memcpy(pData + iMinor, &data, sizeof(plasma_data));
		++iVertex;
	};

}*/
/*ShardModel n_shards_n[NUMVERTICES], ShardModel n_shards[NUMVERTICES],
	AdditionalMomRates)
{
	// Temporarily store it as velocity

	// How to handle ionisation and ionisation effect on momentum, is what we'll come to.

	// Assume Nv = AreaMinor.n.v
	plasma_data data;
	three_vec3 Nv_rates;
	long iMinor;

	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		// Triangles: assume n is what we left on the triangle centre.

		memcpy(&data, pData + iMinor, sizeof(plasma_data));
		memcpy(&Nv_rates, AdditionalMomRates + iMinor, sizeof(three_vec3));
		AreaMinor = ?
		data.v_n = AreaMinor * data.n_n*data.v_n; // Nv
		data.vxy = AreaMinor * data.n*data.vxy;
		data.viz = AreaMinor * data.n*data.viz;
		data.vez = AreaMinor * data.n*data.vez;

		data.v_n += Nv_rates.neut*h_use;
		data.vxy += Nv_rates.ion.xypart()*h_use;
		data.viz += Nv_rates.ion.z*h_use;
		data.vez += Nv_rates.elec.z*h_use;
		// A bit iffy: we have storage for vxy joint
		// but what we have sent is momentum for electrons (vxy ^2) and ions (vxy ^2)
		
		data.n = 0.0;
		data.n_n = 0.0; // don't ask me why

		// It would surely be more good if we could just accumulate density here at the same time
		// and therefore divide the twain!!

		// But bear in mind that n_central != n_major

		memcpy(pHalfMesh->pData + iMinor, &data, sizeof(plasma_data));
	};

	long iVertex = 0;
	for (; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, pData + iMinor, sizeof(plasma_data));
		memcpy(&Nv_rates, AdditionalMomRates + iMinor, sizeof(three_vec3));
		
		f64 n = n_shards[iVertex].n_cent;
		f64 n_n = n_shards_n[iVertex].n_cent;
		AreaMinor = 
		data.v_n = AreaMinor * n_n*data.v_n; // Nv
		data.vxy = AreaMinor * n*data.vxy;
		data.viz = AreaMinor * n*data.viz;
		data.vez = AreaMinor * n*data.vez;

		data.v_n += Nv_rates.neut*h_use;
		data.vxy += Nv_rates.ion.xypart()*h_use;
		data.viz += Nv_rates.ion.z*h_use;
		data.vez += Nv_rates.elec.z*h_use;
		// A bit iffy: we have storage for vxy joint
		// but what we have sent is momentum for electrons (vxy ^2) and ions (vxy ^2)

		data.n = 0.0;
		data.n_n = 0.0; // don't ask me why

		// It would surely be more good if we could just accumulate density here at the same time
		// and therefore divide the twain!!

		// But bear in mind that n_central != n_major

		memcpy(pHalfMesh->pData + iMinor, &data, sizeof(plasma_data));
		++iVertex;
		
	}
}*/

void TriMesh::AdvanceDensityAndTemperature(f64 h_use, 
	TriMesh * pUseMesh, TriMesh * pDestMesh,
	NTrates NTadditionrate[NUMVERTICES])
{
	// Inputs:
			// pVertex->AreaCell should be POPULATED
	// Div_v x3, 
	// NTadditionrate [check how used these arrays: meanings!]
	// data n,v,T from src mesh
	// pVertex->AreaCell

	// Outputs:
	// pHalfMesh->pData+iMinor  [n,T populated; v=0]
	//

	long iVertex, iMinor;
	plasma_data ourdata, data, usedata;
	NTrates AdditionNT, newdata;
	f64 Div_v, Div_v_n, factor, factor_neut, Div_v_overall_integrated;
	Vertex * pVertex = X;
	Vertex * pVertDest = pDestMesh->X;
	static real const one_over_kB = 1.0 / kB_; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0 / (kB_*kB_*kB_); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB_)*kB_;
	static real const over_sqrt_m_e_ = 1.0 / sqrt(m_e_);
	f64 const M_en = m_e_ * m_n_ / ((m_e_ + m_n_)*(m_e_ + m_n_));
	f64 const M_in = m_i_ * m_n_ / ((m_i_ + m_n_)*(m_i_ + m_n_));
	f64 const M_ei = m_e_ * m_i_ / ((m_e_ + m_i_)*(m_e_ + m_i_));

	f64 nu_ne_MT, nu_ni_MT, nu_in_MT, nu_en_MT, nu_ie, nu_ei;

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		iMinor = iVertex + BEGINNING_OF_CENTRAL;

		if (pVertex->flags == DOMAIN_VERTEX) {

			//memcpy(&data, pHalfMesh->pData + iMinor, sizeof(plasma_data));
			memcpy(&ourdata, pData + iMinor, sizeof(plasma_data));
			memcpy(&usedata, pUseMesh->pData + iMinor, sizeof(plasma_data)); 

			memset(&data, 0, sizeof(plasma_data));
			memcpy(&AdditionNT, NTadditionrate + iVertex, sizeof(NTrates));
			Div_v = p_div_v[iVertex];
			Div_v_n = p_div_v_neut[iVertex];
			Div_v_overall_integrated = Integrated_Div_v_overall[iVertex];

			// pVertex->AreaCell should be POPULATED
			// and so is Div_v_overall
			// New area ~= Area + h integral div_v 

			newdata.N = ourdata.n*pVertex->AreaCell + h_use * AdditionNT.N;
			newdata.Nn = ourdata.n_n*pVertex->AreaCell + h_use * AdditionNT.Nn;
			data.n = newdata.N / (pVertex->AreaCell + h_use*Div_v_overall_integrated); // Do have to worry whether advection steps are too frequent.
			data.n_n = newdata.Nn / (pVertex->AreaCell + h_use*Div_v_overall_integrated); // What could do differently: know ROC area as well as mass flux through walls

			if (iVertex == CHOSEN) printf("CPU %d nsrc %1.14E N_n %1.14E AreaMajor %1.14E hAdditionN_n %1.14E \n"
				"dest_nn %1.14E newdata.Nn %1.14E Area_used %1.14E \n\n", iVertex,
				ourdata.n_n,
				ourdata.n_n*pVertex->AreaCell, pVertex->AreaCell, h_use * AdditionNT.Nn,
				data.n_n, newdata.Nn, pVertex->AreaCell + h_use*Div_v_overall_integrated);

			// roughly right ; maybe there are improvements.

			// --------------------------------------------------------------------------------------------
			// Simple way of doing area ratio for exponential growth of T: 
			// (1/(1+h div v)) -- v outward grows the area so must be + here. 

			// Compressive heating:
			// USE 1 iteration of Halley's method for cube root:
			// cu_root Q =~~= x0(x0^3+2Q)/(2x0^3+Q) .. for us x0 = 1, Q is (1+eps)^-2
			// Thus (1+2(1+eps)^-2)/(2+(1+eps)^-2)
			// Multiply through by (1+eps)^2:
			// ((1+eps)^2+2)/(1+2*(1+eps)^2) .. well of course it is
			// eps = h div v

			// Way to get reasonable answer without re-doing equations:
			// Take power -1/3 and multiply once before interspecies and once after.

			factor = (3.0 + h_use * Div_v) /
				(3.0 + 2.0* h_use * Div_v);
			factor_neut = (3.0 + h_use * Div_v_n) /
				(3.0 + 2.0*h_use * Div_v_n);
			// gives (1+ h div v)^(-1/3), roughly

			// Alternate version: 
			// factor = pow(pVertex->AreaCell / pVertDest->AreaCell, 2.0 / 3.0);
			// pVertDest->Ion.heat = pVertex->Ion.heat*factor;
			// but the actual law is with 5/3 
			// Comp htg dT/dt = -2/3 T div v_fluid 
			// factor (1/(1+h div v))^(2/3) --> that's same

			newdata.NnTn = ourdata.n_n*pVertex->AreaCell*ourdata.Tn*factor_neut + h_use * AdditionNT.NnTn;
			newdata.NiTi = ourdata.n*pVertex->AreaCell*ourdata.Ti*factor + h_use * AdditionNT.NiTi;
			newdata.NeTe = ourdata.n*pVertex->AreaCell*ourdata.Te*factor + h_use * AdditionNT.NeTe;  // 

			if (iVertex == CHOSEN) {
	//			printf("CPU h AdditionNT.NeTe %1.14E\n",
	//				h_use * AdditionNT.NeTe);
	//			printf("CPU %d inc factor: NnTn %1.14E NeTe %1.14E \n", CHOSEN, newdata.NnTn, newdata.NeTe);
	//			printf("n %1.14E Area %1.14E Te_k %1.14E factor %1.14E \n",
	//				ourdata.n, pVertex->AreaCell, ourdata.Te, factor);					
			}
			// We have initial NT
			// We have addition amount for NT
			// We have factor that applies
			// We can apply inter-species transfer
			// Finally divide by N to give new T

			// Cognate to accelerate routine collision freq calc code:
			{
				// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
				f64 sqrt_Te, ionneut_thermal, electron_thermal, lnLambda, s_in_MT, s_en_MT, s_en_visc;

				sqrt_Te = sqrt(usedata.Te); // should be "usedata"
				ionneut_thermal = sqrt(usedata.Ti / m_i_ + usedata.Tn / m_n_); // hopefully not sqrt(0)
				electron_thermal = sqrt_Te * over_sqrt_m_e_;
				lnLambda = Get_lnLambda(usedata.n, usedata.Te);

				s_in_MT = Estimate_Ion_Neutral_MT_Cross_section(usedata.Ti*one_over_kB);
				Estimate_Ion_Neutral_Cross_sections(usedata.Te*one_over_kB, // call with T in electronVolts
					&s_en_MT,
					&s_en_visc);
				//s_en_MT = Estimate_Ion_Neutral_MT_Cross_section(usedata.Te*one_over_kB);
				//s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(usedata.Te*one_over_kB);
				// Need nu_ne etc to be defined:
				nu_ne_MT = s_en_MT * usedata.n * electron_thermal; // have to multiply by n_e for nu_ne_MT
				nu_ni_MT = s_in_MT * usedata.n * ionneut_thermal;
				nu_en_MT = s_en_MT * usedata.n_n*electron_thermal;
	//			if (iVertex == CHOSEN) printf("CPU nu_en_MT components %1.8E %1.8E %1.8E\n",
	//				s_en_MT, usedata.n_n, electron_thermal);
				nu_in_MT = s_in_MT * usedata.n_n*ionneut_thermal;

				nu_ei = nu_eiBarconst_ * kB_to_3halves*usedata.n*lnLambda / 
									(usedata.Te*sqrt_Te);
				nu_ie = nu_ei;
			//	nu_eHeart = 1.87*nu_eiBar + data_k.n_n*s_en_visc*electron_thermal;
			}
			// For now doing velocity-independent resistive heating.
			// Because although we have a magnetic correction Upsilon_zz involved, we ignored it
			// since we are also squashing the effect of velocity-dependent collisions on vx and vy (which
			// would produce a current in the plane) and this squashing should create heat, which
			// maybe means it adds up to velo independent amount of heating. 

			newdata.NeTe += h_use*(pVertex->AreaCell*TWOTHIRDS*nu_en_MT*m_en_*(
				  (usedata.v_n.x-usedata.vxy.x)*(usedata.v_n.x - usedata.vxy.x)
				+ (usedata.v_n.y - usedata.vxy.y)*(usedata.v_n.y - usedata.vxy.y)
				+ (usedata.v_n.z - usedata.vez)*(usedata.v_n.z - usedata.vez))

				+ pVertex->AreaCell*TWOTHIRDS*nu_ei*m_ei_*(usedata.vez-usedata.viz)*(usedata.vez-usedata.viz));
			
			newdata.NiTi += h_use*(pVertex->AreaCell*TWOTHIRDS*nu_in_MT*M_in*m_n_*(
				  (usedata.v_n.x - usedata.vxy.x)*(usedata.v_n.x - usedata.vxy.x)
				+ (usedata.v_n.y - usedata.vxy.y)*(usedata.v_n.y - usedata.vxy.y)
				+ (usedata.v_n.z - usedata.viz)*(usedata.v_n.z - usedata.viz)));

			newdata.NnTn += h_use*(pVertex->AreaCell*TWOTHIRDS*nu_ni_MT*M_in*m_i_*(
				(usedata.v_n.x - usedata.vxy.x)*(usedata.v_n.x - usedata.vxy.x)
				+ (usedata.v_n.y - usedata.vxy.y)*(usedata.v_n.y - usedata.vxy.y)
				+ (usedata.v_n.z - usedata.viz)*(usedata.v_n.z - usedata.viz)));
//			if (iVertex == CHOSEN) {
//				printf("CPU %d : NnTn %1.14E NeTe %1.14E \n", CHOSEN, newdata.NnTn, newdata.NeTe);
//				printf("e-n %1.14E e-i %1.14E \n",
//					h_use*(pVertex->AreaCell*TWOTHIRDS*nu_en_MT*m_en_*(
//					(usedata.v_n.x - usedata.vxy.x)*(usedata.v_n.x - usedata.vxy.x)
//						+ (usedata.v_n.y - usedata.vxy.y)*(usedata.v_n.y - usedata.vxy.y)
//						+ (usedata.v_n.z - usedata.vez)*(usedata.v_n.z - usedata.vez))),
//					h_use*(pVertex->AreaCell*TWOTHIRDS*nu_ei*m_ei_*
//					(usedata.vez - usedata.viz)*(usedata.vez - usedata.viz))
//				);
//			}

			f64_tens3 tens3, LHS, inverted;
			f64_vec3 RHS, NT;
			// x = neutral
			// y = ion
			// z = elec
			tens3.xx = -M_en * nu_ne_MT - M_in * nu_ni_MT;
			tens3.xy = M_in * nu_in_MT;
			tens3.xz = M_en * nu_en_MT;
			tens3.yx = M_in * nu_ni_MT;
			tens3.yy = -M_in * nu_in_MT - M_ei * nu_ie;
			tens3.yz = M_ei * nu_ei;
			tens3.zx = M_en * nu_ne_MT;
			tens3.zy = M_ei * nu_ie;
			tens3.zz = -M_en * nu_en_MT - M_ei * nu_ei;
			// This is for NT
			LHS.xx = 1.0 - h_use * tens3.xx;
			LHS.xy = -h_use * tens3.xy;
			LHS.xz = -h_use * tens3.xz;
			LHS.yx = -h_use * tens3.yx;
			LHS.yy = 1.0 - h_use * tens3.yy;
			LHS.yz = -h_use * tens3.yz;
			LHS.zx = -h_use * tens3.zx;
			LHS.zy = -h_use * tens3.zy;
			LHS.zz = 1.0 -h_use * tens3.zz;
			LHS.Inverse(inverted);

			RHS.x = newdata.NnTn - h_use * (nu_ni_MT*M_in + nu_ne_MT * M_en)*newdata.NnTn
				+ h_use * nu_in_MT*M_in*newdata.NiTi + h_use * nu_en_MT*M_en*newdata.NeTe;
			RHS.y = newdata.NiTi - h_use * (nu_in_MT*M_in + nu_ie * M_ei)*newdata.NiTi
				+ h_use * nu_ni_MT*M_in*newdata.NnTn + h_use * nu_ei*M_ei*newdata.NeTe;
			RHS.z = newdata.NeTe - h_use * (nu_en_MT*M_en + nu_ei * M_ei)*newdata.NeTe
				+ h_use * nu_ie*M_ei*newdata.NiTi + h_use * nu_ne_MT*M_en*newdata.NnTn;

			NT = inverted * RHS;
			newdata.NnTn = NT.x;
			newdata.NiTi = NT.y;
			newdata.NeTe = NT.z;
			
	//		if (iVertex == CHOSEN) {
	//			printf("CPU LHS | \n %1.14E %1.14E %1.14E |\n %1.14E %1.14E %1.14E |  \n %1.14E %1.14E %1.14E | \n",
	//				LHS.xx, LHS.xy, LHS.xz, LHS.yx, LHS.yy, LHS.yz, LHS.zx, LHS.zy, LHS.zz);
	//			printf("CPU inverted | RHS \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n",
	//				inverted.xx, inverted.xy, inverted.xz, RHS.x, inverted.yx, inverted.yy, inverted.yz, RHS.y, inverted.zx, inverted.zy, inverted.zz, RHS.z);
	//			printf("CPU %d : NnTn %1.14E  NeTe %1.14E \n", CHOSEN, newdata.NnTn, newdata.NeTe);
	//			printf("CPU nu_en_MT %1.14E\n", nu_en_MT);
	//		};
			// Multiply all by "factor" regardless of how it turns out.

			// Whether we are being sensible at such small timesteps
			// we need to question and rethink.

			data.Tn = newdata.NnTn* factor_neut / newdata.Nn;
			data.Ti = newdata.NiTi* factor / newdata.N;
			data.Te = newdata.NeTe* factor / newdata.N;

//			if (iVertex == CHOSEN)
//				printf("CPU %d : Te %1.14E factor %1.14E newdata.N %1.14E\n",
//					CHOSEN, data.Te, factor, newdata.N);

			if (data.Te < 0.0) {
				iVertex = iVertex;
			}
			memcpy(pDestMesh->pData + iMinor, &data, sizeof(plasma_data));
		} else {

		    // Not DOMAIN_VERTEX :
			if (pVertex->flags == OUTERMOST) {
				memcpy(pDestMesh->pData + iMinor, pData + iMinor, sizeof(plasma_data));
			} else {
				(pDestMesh->pData + iMinor)->Te = 0.0;
				(pDestMesh->pData + iMinor)->Ti = 0.0;
				(pDestMesh->pData + iMinor)->Tn = 0.0;
				(pDestMesh->pData + iMinor)->n = 0.0;
				(pDestMesh->pData + iMinor)->n_n = 0.0;
			}
			// pass information forward in case this is outermost edge.
		};
		
		++pVertex;
		++pVertDest;
	};
	// The job here was only to populate for vertices. What to put elsewhere --
	// check always whether data is populated.
}


void TriMesh::CalculateOverallVelocities(f64_vec2 p_v[]) {
	
	// Inputs:
	// data n,v,T
	// basic Triangle data

	// Outputs:
	// p_v (parameter)
	
	long iMinor, index0, index1, index2;
	Triangle * pTri = T;
	plasma_data data;
	Vertex * pVertex = X;
	for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, pData + iMinor, sizeof(plasma_data));
		
		if (pVertex->flags == DOMAIN_VERTEX) {
			p_v[iMinor] = ((m_i_ + m_e_)*data.n*data.vxy +
				m_n_ * data.n_n*data.v_n.xypart()) /
				(data.n*(m_i_ + m_e_) + data.n_n*m_n_);
		//	if (iMinor - BEGINNING_OF_CENTRAL == 11588) {
		//		printf("\n11588 vxy %1.9E %1.9E  v_n %1.9E %1.9E \n",
		//			data.vxy.x, data.vxy.y, data.v_n.x, data.v_n.y);
		//	}
		} else {
			p_v[iMinor] = f64_vec2(0.0, 0.0); // Non-domain vertices do not move
		};
		++pVertex;
	};

	// advection of tri centroids : 
	for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
	{
		index0 = pTri->cornerptr[0] - X;
		index1 = pTri->cornerptr[1] - X;
		index2 = pTri->cornerptr[2] - X;

		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {

			f64_vec2 v0, v1, v2;
			v0 = p_v[BEGINNING_OF_CENTRAL + index0]; // This had a bug: these are vertex indices!
			v1 = p_v[BEGINNING_OF_CENTRAL + index1];
			v2 = p_v[BEGINNING_OF_CENTRAL + index2];
			if (pTri->periodic != 0) {
				if (pTri->cornerptr[0]->pos.x > 0.0) v0 = Anticlockwise * v0;
				if (pTri->cornerptr[1]->pos.x > 0.0) v1 = Anticlockwise * v1;
				if (pTri->cornerptr[2]->pos.x > 0.0) v2 = Anticlockwise * v2;
			}				
			// Continue to see periodic boundary tri as living on left regardless of centroid.
			// Idea is: Centroid ALWAYS lives on left even if it's outside domain.
			
			p_v[iMinor] = THIRD * (v0 + v1 + v2);
			
		} else {
			if (pTri->u8domain_flag == CROSSING_INS) {
				// We site the triangle data on the insulator,
				// so certainly its radial advection is zero ....

				f64_vec2 v0, v1, v2;
				v0 = p_v[BEGINNING_OF_CENTRAL + index0];
				v1 = p_v[BEGINNING_OF_CENTRAL + index1];
				v2 = p_v[BEGINNING_OF_CENTRAL + index2];

				if (pTri->periodic != 0) {
					if (pTri->cornerptr[0]->pos.x > 0.0) v0 = Anticlockwise * v0;
					if (pTri->cornerptr[1]->pos.x > 0.0) v1 = Anticlockwise * v1;
					if (pTri->cornerptr[2]->pos.x > 0.0) v2 = Anticlockwise * v2;
				}
				
				p_v[iMinor] = THIRD * (v0 + v1 + v2); // Note: still 1/3
				// It is equal to 1/3 avg, projected to ins.				
				// So if we are moving 2 points to the right, it only moves 2/3 as much.

				// Now remove the radial component:
				f64_vec2 r = pTri->cent;
				//rhat = r / r.modulus();
				//p_v[iMinor] -= rhat.dot(p_v[iMinor])*rhat;
				p_v[iMinor] = p_v[iMinor] - r*(r.dot(p_v[iMinor]) / (r.x*r.x + r.y*r.y));
				
			} else {
				p_v[iMinor] = f64_vec2(0.0, 0.0); // Applies to frills, ins tris.
			};
		};
		++pTri;
	};
}

void TriMesh::Average_n_T_to_tris_and_calc_centroids_and_minorpos()
{
	// Inputs:
	// vertex n,T

	// Outputs:
	// tri n,T
	// recalculated tri centroid
	// minor pos

	long iTri;
	f64 n0, n1, n2;
	plasma_data data0, data1, data2, tridata;

	Vertex * pVertex = X;
	plasma_data * pdata = pData + BEGINNING_OF_CENTRAL;
	for (long iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
	{
		pdata->pos = pVertex->pos;
		++pVertex;
		++pdata;
	};
	Triangle * pTri = T;
	for (iTri = 0; iTri < NUMTRIANGLES; iTri++)
	{
		memcpy(&data0, pData + (pTri->cornerptr[0] - X) + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
		memcpy(&data1, pData + (pTri->cornerptr[1] - X) + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
		memcpy(&data2, pData + (pTri->cornerptr[2] - X) + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
		memcpy(&tridata, pData + iTri, sizeof(plasma_data));
		// gather not scatter
		
		n0 = data0.n;
		n1 = data1.n;
		n2 = data2.n;
		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
			tridata.n = THIRD * (n0 + n1 + n2);
			tridata.n_n = THIRD * (data0.n_n + data1.n_n + data2.n_n);
			tridata.Te = THIRD * (data0.Te + data1.Te + data2.Te);
			tridata.Ti = THIRD * (data0.Ti + data1.Ti + data2.Ti);
			tridata.Tn = THIRD * (data0.Tn + data1.Tn + data2.Tn);
		} else {
			// Insulator tris:
			if (pTri->u8domain_flag == CROSSING_INS)
			{
				int iAbove = 0;
				tridata.n = 0.0; tridata.n_n = 0.0; tridata.Te = 0.0;
				tridata.Ti = 0.0; tridata.Tn = 0.0;

				if (data0.pos.x*data0.pos.x + data0.pos.y*data0.pos.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					iAbove++;
					tridata.n += n0; tridata.n_n += data0.n_n;
					tridata.Tn += data0.Tn; tridata.Ti += data0.Ti; tridata.Te += data0.Te;
				}
				if (data1.pos.x*data1.pos.x + data1.pos.y*data1.pos.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					iAbove++;
					tridata.n += n1; tridata.n_n += data1.n_n;
					tridata.Tn += data1.Tn; tridata.Ti += data1.Ti; tridata.Te += data1.Te;
				}
				if (data2.pos.x*data2.pos.x + data2.pos.y*data2.pos.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					iAbove++;
					tridata.n += n2; tridata.n_n += data2.n_n;
					tridata.Tn += data2.Tn; tridata.Ti += data2.Ti; tridata.Te += data2.Te;
				}
				f64 divide = 1.0 / (f64)iAbove;
				tridata.n *= divide;
				tridata.n_n *= divide;
				tridata.Te *= divide;
				tridata.Ti *= divide;
				tridata.Tn *= divide;

				// WE ASSUME n = T = 0 in ins, do we manage to maintain this?

				// Evidently not. *****************  **************  **************

			} else {
				tridata.n = 0.0;
				tridata.n_n = 0.0;
				tridata.Te = 0.0;
				tridata.Ti = 0.0;
				tridata.Tn = 0.0;
			};
		};
		pTri->RecalculateCentroid(InnermostFrillCentroidRadius,OutermostFrillCentroidRadius);
		tridata.pos = pTri->cent;
		
		memcpy(pData + iTri, &tridata, sizeof(plasma_data));

		if (tridata.Te < 0.0) {
			iTri = iTri;
		}
		++pTri;
	};
}

void TriMesh::AccumulateDiffusiveHeatRateAndCalcIonisation(f64 h_use, NTrates NTadditionrates[NUMVERTICES])
{
	// Inputs:
	// 

	// Outputs:
	// NTadditionrates augmented

	// Do tri input just to get it running, because we do not have stored the PB list for neighs.
	// But that is insane, so we will want to change this.

	// It will make much more of a headache on GPU.
	// The other option is to combine it with the advective mass & heat rates which already use minor info.
	static real const kB_to_3halves = sqrt(kB_)*kB_;
	static real const one_over_kB = 1.0 / kB_;
	static real const over_sqrt_m_e_ = 1.0 / sqrt(m_e_);

	Vertex * pVertex = X;
	long iVertex;
	long izNeigh[MAXNEIGH];
	long izTri[MAXNEIGH];
	char szPBCtri[MAXNEIGH];
	long neigh_len, tri_len;
	plasma_data tridata1, tridata2, neighdata, ourdata, prevdata, nextdata;
	int i, iprev;
	NTrates ourrates, yourrates;
	f64 AreaMajor;
	f64_vec2 prev_overall_v, here_overall_v, vxy, edge_normal, motion_edge;
	f64_vec3 v_n;
	f64 massflux, heatflux;
	f64 nu_eHeart1, nu_eHeart2, nu_eiBar, nu_in_visc, nu_ni_visc, nu_nn_visc1, nu_ii,
		nu_iHeart1, nu_iHeart2, nu_nn_visc2, nu_iHeart, nu_nn_visc,
		Area_quadrilateral, kappa_parallel, nu_eHeart;
	f64_vec2 integrated_grad_Te, integrated_grad_Ti, integrated_grad_Tn, gradTe, gradTi, gradTn,
		kappa_grad_T;
	f64_vec3 omega;
	Tensor2 kappa;
	int inext;
	static real const SQRT2 = sqrt(2.0);

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			AreaMajor = 0.0;

			memcpy(&ourdata, pData + iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

			f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;

			TeV = ourdata.Te * one_over_kB;
			Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc);
			sqrt_T = sqrt(ourdata.Te);
			nu_en_visc = ourdata.n_n * sigma_visc*sqrt_T * over_sqrt_m_e_;
			nu_eiBar = nu_eiBarconst_ * kB_to_3halves*ourdata.n*Get_lnLambda(ourdata.n, ourdata.Te) / (ourdata.Te*sqrt_T);
			nu_eHeart1 = (nu_en_visc + 1.87*nu_eiBar);

			TeV = ourdata.Ti*one_over_kB;
			Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc); // could easily save one call
			sqrt_T = sqrt(ourdata.Ti); // again not that hard to save one call
			nu_in_visc = ourdata.n_n * sigma_visc*sqrt(ourdata.Ti / m_i_ + ourdata.Tn / m_n_);
			nu_ni_visc = nu_in_visc * (ourdata.n / ourdata.n_n);
			nu_nn_visc1 = ourdata.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section(ourdata.Tn / kB_)
				* sqrt(ourdata.Tn / m_n_);
			nu_ii = ourdata.n*kB_to_3halves*Get_lnLambda_ion(ourdata.n, ourdata.Ti)*Nu_ii_Factor_ / (sqrt_T*ourdata.Ti);
			nu_iHeart1 = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu_nn_visc1);

	//		if (iVertex == CHOSEN) printf("CPU nu_i components %1.12E %1.12E %1.12E %1.12E n %1.10E n_n %1.10E Ti %1.10E \n"
	//			"sigma_visc %1.12E Ti %1.10E Tn %1.10E sqrt %1.10E\n",
	//			nu_in_visc, nu_ii, nu_ni_visc, nu_nn_visc1,
	//			ourdata.n, ourdata.n_n, ourdata.Ti,
	//			sigma_visc, ourdata.Ti, ourdata.Tn, sqrt(ourdata.Ti / m_i_ + ourdata.Tn / m_n_));

			// That's for our own.


			// The idea is simply to look at each edge. We take n from shard model on the upwind side. T is averaged, 
			// for now do simple average instead of honeycomb. 
			// v comes from the two triangles.

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			tri_len = pVertex->GetTriIndexArray(izTri);
			// and as usual we assume tri 0 has neigh 0 on the clockwise side of it.
	//		memcpy(szPBCtri, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);

	//		memcpy(&tridata1, pData + izTri[tri_len - 1], sizeof(plasma_data));
			iprev = neigh_len - 1;
	//		if (szPBCtri[iprev] == ROTATE_ME_CLOCKWISE) {
	//			tridata1.B = Clockwise3 * tridata1.B; // what we use from minors
	//			tridata1.pos = Clockwise * tridata1.pos;
	//		};
	//		if (szPBCtri[iprev] == ROTATE_ME_ANTICLOCKWISE) {
	//			tridata1.B = Anticlockwise3 * tridata1.B;
	//			tridata1.pos = Anticlockwise * tridata1.pos;
	//		};

	
	// Get T x3:
			memcpy(&prevdata, pData + BEGINNING_OF_CENTRAL + izNeigh[iprev], sizeof(plasma_data));
			memcpy(&neighdata, pData + izNeigh[0] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));

			if ((pX->T + izTri[iprev])->periodic == 0) {
				// do nothing: neighbour must be contiguous
			}
			else {
				if ((prevdata.pos.x > 0.0) && (ourdata.pos.x < 0.0))
				{
					//szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
					prevdata.B = Anticlockwise3*prevdata.B;
					prevdata.pos = Anticlockwise*prevdata.pos;
				}
				if ((prevdata.pos.x < 0.0) && (ourdata.pos.x > 0.0))
				{
					prevdata.B = Clockwise3*prevdata.B;
					prevdata.pos = Clockwise*prevdata.pos;
				}
			};
			if (prevdata.Te == 0.0) {
				prevdata.Te = 0.5*(ourdata.Te + neighdata.Te);
				prevdata.Ti = 0.5*(ourdata.Ti + neighdata.Ti);
				prevdata.Tn = 0.5*(ourdata.Tn + neighdata.Tn);
			}
			
			if ((pX->T + izTri[0])->periodic == 0) {
				// do nothing: neighbour must be contiguous
			}
			else {
				if ((neighdata.pos.x > 0.0) && (ourdata.pos.x < 0.0))
				{
					//szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
					neighdata.B = Anticlockwise3*neighdata.B;
					neighdata.pos = Anticlockwise*neighdata.pos;
				}
				if ((neighdata.pos.x < 0.0) && (ourdata.pos.x > 0.0))
				{
					neighdata.B = Clockwise3*neighdata.B;
					neighdata.pos = Clockwise*neighdata.pos;
				}
			};
		//	if (iVertex == CHOSEN) printf("neigh pos %1.14E %1.14E\n", neighdata.pos.x, neighdata.pos.y);

			// If we are doing this, we don't really need to load from tridata as well
			// We could get B, n and pos from neighbours.
			// It is guaranteed that pos is the average.

			// ^^ no it is not, in an INS_CROSSING tri. What did I do on GPU?

			// what is the name of the PB array for neighbours?
			// It seems we do not store it. ?? DASH IT
						 
			// nu depends log-linearly on n so why are we averaging n, we could
			// just average nu???
			// OK this was a stupid way round??
			// We actually use: nu_eHeart, nu_iHeart, nu_nn_visc. Is that correct?
			// B is mostly smooth and flattish.

			for (i = 0; i < neigh_len; i++)
			{
				// First get nu for neighdata:

				if (neighdata.pos.x*neighdata.pos.x+ neighdata.pos.y*neighdata.pos.y <
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER) {
					nu_eHeart2 = 0.0;
					nu_iHeart2 = 0.0;
					nu_nn_visc2 = 0.0;
					// And we will not do traffic
				} else {
					TeV = neighdata.Te * one_over_kB;
					Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc);
					sqrt_T = sqrt(neighdata.Te);
					nu_en_visc = neighdata.n_n * sigma_visc*sqrt_T * over_sqrt_m_e_;
					nu_eiBar = nu_eiBarconst_ * kB_to_3halves*neighdata.n*Get_lnLambda(neighdata.n, neighdata.Te) / (neighdata.Te*sqrt_T);
					nu_eHeart2 = (nu_en_visc + 1.87*nu_eiBar);

					TeV = neighdata.Ti*one_over_kB;
					Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc); // could easily save one call
					sqrt_T = sqrt(neighdata.Ti); // again not that hard to save one call
					nu_in_visc = neighdata.n_n * sigma_visc*sqrt(neighdata.Ti / m_i_ + neighdata.Tn / m_n_);
					nu_ni_visc = nu_in_visc * (neighdata.n / neighdata.n_n);
					nu_nn_visc2 = neighdata.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section(neighdata.Tn / kB_)
						* sqrt(neighdata.Tn / m_n_);
					nu_ii = neighdata.n*kB_to_3halves*Get_lnLambda_ion(neighdata.n, neighdata.Ti) *Nu_ii_Factor_ / (sqrt_T*neighdata.Ti);

					nu_iHeart2 = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu_nn_visc2);
				};
				//

				inext = i + 1; if (inext == neigh_len) inext = 0;
				//memcpy(&tridata2, pData + izTri[i], sizeof(plasma_data));
				memcpy(&nextdata, pData + izNeigh[inext] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
				
				if ((pX->T + izTri[inext])->periodic == 0) {
					// do nothing: neighbour must be contiguous
				} else {
					if ((nextdata.pos.x > 0.0) && (ourdata.pos.x < 0.0))
					{
						//szPBCnext[i] = ROTATE_ME_ANTICLOCKWISE;
						nextdata.B = Anticlockwise3*nextdata.B;
						nextdata.pos = Anticlockwise*nextdata.pos;
					}
					if ((nextdata.pos.x < 0.0) && (ourdata.pos.x > 0.0))
					{
						nextdata.B = Clockwise3*nextdata.B;
						nextdata.pos = Clockwise*nextdata.pos;
					}
				};
				if (iVertex == CHOSEN) printf("nextpos %1.12E %1.12E\n", nextdata.pos.x, nextdata.pos.y);

				// Handle case that nextdata is inside ins:
				if (nextdata.Te == 0.0)
				{
					nextdata.Te = 0.5*(ourdata.Te + neighdata.Te);
					nextdata.Ti = 0.5*(ourdata.Ti + neighdata.Ti);
					nextdata.Tn = 0.5*(ourdata.Tn + neighdata.Tn);
				};
				
				edge_normal.x = THIRD*(nextdata.pos.y - prevdata.pos.y);
				edge_normal.y = THIRD*(prevdata.pos.x - nextdata.pos.x); // nextdata = pos_anti
				
				// Living with the fact that we did not match the triangle centre if it's CROSSING_INS

				AreaMajor += 0.5*edge_normal.x*THIRD*(prevdata.pos.x + 2.0*ourdata.pos.x + 2.0*neighdata.pos.x + nextdata.pos.x); // NOT TO BE USED - match GPU

				// Get grad T on quadrilateral:
				// Might as well load NEIGHBOUR T for these gradients.

				integrated_grad_Te.x = 0.5*(
					(ourdata.Te + nextdata.Te)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Te + ourdata.Te)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Te + prevdata.Te)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Te + neighdata.Te)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Te.y = -0.5*( // notice minus
					(ourdata.Te + nextdata.Te)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Te + ourdata.Te)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Te + prevdata.Te)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Te + neighdata.Te)*(nextdata.pos.x - neighdata.pos.x)
					);

				integrated_grad_Ti.x = 0.5*(
					(ourdata.Ti + nextdata.Ti)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Ti + ourdata.Ti)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Ti + prevdata.Ti)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Ti + neighdata.Ti)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Ti.y = -0.5*( // notice minus
					(ourdata.Ti + nextdata.Ti)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Ti + ourdata.Ti)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Ti + prevdata.Ti)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Ti + neighdata.Ti)*(nextdata.pos.x - neighdata.pos.x)
					);
				integrated_grad_Tn.x = 0.5*(
					(ourdata.Tn + nextdata.Tn)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Tn + ourdata.Tn)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Tn + prevdata.Tn)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Tn + neighdata.Tn)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Tn.y = -0.5*( // notice minus
					(ourdata.Tn + nextdata.Tn)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Tn + ourdata.Tn)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Tn + prevdata.Tn)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Tn + neighdata.Tn)*(nextdata.pos.x - neighdata.pos.x)
					);
				Area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.pos.x + prevdata.pos.x)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + neighdata.pos.x)*(nextdata.pos.y - neighdata.pos.y)
					);
				
				gradTe = integrated_grad_Te / Area_quadrilateral;
				gradTi = integrated_grad_Ti / Area_quadrilateral;
				gradTn = integrated_grad_Tn / Area_quadrilateral;

			//	if (iVertex == CHOSEN) printf("CPU our_T %1.14E anti %1.14E opp %1.14E next %1.14E\n",
			//		ourdata.Te, prevdata.Te, neighdata.Te, nextdata.Te);

				// =================
				// Electrons:

				// nu_eHeart:
				// n WILL BE CANCELLED from kappa_parallel and basically from kappa until omega ~ nu
				// We can perfectly easily use n,T,B from the major data;
				// and thus avoid loading from minors.
				// We only did not do already because need to make PB char list !!!

				// Notice that n divides through as we have nu beneath.
				// So we'd rather average kappa_parallel than average n in its numerator.

				if (nu_eHeart2 != 0.0) {
					kappa_parallel = 2.5*
						0.5*(ourdata.n / nu_eHeart1 + neighdata.n / nu_eHeart2)
						* 0.5*(ourdata.Te + neighdata.Te)
						* over_m_e_;

		//			if (iVertex == CHOSEN) printf("kappa_par_e %1.14E nu_eHeart 1,2 %1.14E %1.14E n %1.14E %1.14E Te %1.14E %1.14E \n",
		//				kappa_parallel, nu_eHeart1, nu_eHeart2, ourdata.n, neighdata.n, ourdata.Te, neighdata.Te);

				} else {
					kappa_parallel = 0.0;
				};
				nu_eHeart = 0.5*(nu_eHeart1 + nu_eHeart2);

				omega = eovermc_ * 0.5*(ourdata.B + neighdata.B);
			//	if (iVertex == CHOSEN) 
			//		printf("CPU our B %1.10E %1.10E %1.10E neighB %1.10E %1.10E %1.10E\n",
			//			ourdata.B.x, ourdata.B.y, ourdata.B.z, neighdata.B.x, neighdata.B.y, neighdata.B.z);

				f64 omega_sq = omega.dot(omega);
				kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
				kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

				kappa_grad_T.x = kappa.xx*gradTe.x + kappa.xy*gradTe.y;
				kappa_grad_T.y = kappa.yx*gradTe.x + kappa.yy*gradTe.y;

				// if the outward gradient of T is positive, inwardheatflux is positive.
				ourrates.NeTe += TWOTHIRDS * kappa_grad_T.dot(edge_normal);
				
				if ((0) && (iVertex == CHOSEN)) {
					printf("CPU NeTe %d %d : contrib %1.14E kappa_par %1.14E edge_nor %1.14E %1.14E\n"
						"omega %1.14E %1.14E %1.14E \nnu_eHeart %1.14E grad_T %1.14E %1.14E\n"
						"kappa %1.14E %1.14E %1.14E %1.14E \n",
						iVertex, izNeigh[i],
						TWOTHIRDS * kappa_grad_T.dot(edge_normal),
						kappa_parallel, edge_normal.x, edge_normal.y,
						omega.x, omega.y, omega.z, nu_eHeart, gradTe.x, gradTe.y,
						kappa.xx, kappa.xy, kappa.yx, kappa.yy  
					);
				}

				// ------------------------------------------------------------------------
				// Ion:

				if (nu_iHeart2 != 0.0) {
					kappa_parallel = (20.0 / 9.0)*
						0.5*(ourdata.n / nu_iHeart1 + neighdata.n / nu_iHeart2)
						* 0.5*(ourdata.Ti + neighdata.Ti)
						/ m_ion_;
				} else {
					kappa_parallel = 0.0;
				};

				nu_iHeart = 0.5*(nu_iHeart1 + nu_iHeart2);
				omega = qoverMc_ * 0.5*(ourdata.B + neighdata.B);

				// We could arrange to just store nu_iHeart as we go around.

				kappa.xx = kappa_parallel * (nu_iHeart*nu_iHeart + omega.x*omega.x) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.xy = kappa_parallel * (omega.x*omega.y + nu_iHeart * omega.z) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.yx = kappa_parallel * (omega.x*omega.y - nu_iHeart * omega.z) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.yy = kappa_parallel * (omega.y*omega.y + nu_iHeart * nu_iHeart) / (nu_iHeart * nu_iHeart + omega_sq);

				kappa_grad_T.x = kappa.xx*gradTi.x + kappa.xy*gradTi.y;
				kappa_grad_T.y = kappa.yx*gradTi.x + kappa.yy*gradTi.y;

				ourrates.NiTi += TWOTHIRDS * kappa_grad_T.dot(edge_normal);

			//	if (iVertex == CHOSEN) {
			//		printf("CPU %d : %d NiTicontrib %1.9E kappa_par %1.9E nu %1.12E omega %1.9E %1.9E %1.9E "
			//			"gradTi %1.10E %1.10E edge_normal %1.9E %1.9E\nOWN nu: %1.12E nu_iHeart2 %1.12E\n",
			//			CHOSEN,izNeigh[i], TWOTHIRDS * kappa_grad_T.dot(edge_normal),
			//			kappa_parallel, nu_iHeart, omega.x, omega.y, omega.z,
			//			gradTi.x, gradTi.y,edge_normal.x,edge_normal.y,
			//			nu_iHeart1,nu_iHeart2);
			//	}

				// ----------------------------------------------------------------------
				// Neutral:

				// HERE IS WHERE IT'S A POINT OF SOME DOUBT: 10 or far less?
				if (nu_nn_visc2 != 0.0) {
					kappa_parallel = 10.0*0.5*
						(ourdata.n_n / nu_nn_visc1 + neighdata.n_n / nu_nn_visc2)
						* 0.5*(ourdata.Tn + neighdata.Tn)
						* over_m_n_;
				} else {
					kappa_parallel = 0.0;
				};
				kappa_grad_T.x = kappa_parallel * gradTn.x;
				kappa_grad_T.y = kappa_parallel * gradTn.y;

				ourrates.NnTn += TWOTHIRDS * kappa_grad_T.dot(edge_normal);

				// increments:
				memcpy(&prevdata, &neighdata, sizeof(plasma_data)); // ditto
				memcpy(&neighdata, &nextdata, sizeof(plasma_data)); // ditto

				iprev = i;
			};

			// now add IONISATION:

			TeV = ourdata.Te / kB_;
			f64 sqrtT = sqrt(TeV);

			f64 temp = 1.0e-5*exp(-13.6 / TeV) / (13.6*(6.0*13.6 + TeV));
			// Let h n n_n S be the ionising amount,
			// h n S is the proportion of neutrals! Make sure we do not run out!
			f64 hnS = (h_use*ourdata.n*TeV*temp) /
				(sqrtT + h_use * ourdata.n_n*ourdata.n*temp*SIXTH*13.6);
			f64 ionise_rate = AreaMajor * ourdata.n_n*hnS / (h_use*(1 + hnS));
			// ionise_amt / h

			ourrates.N += ionise_rate;
			ourrates.Nn += -ionise_rate;

			if (iVertex == CHOSEN) {
				printf("CPU iVertex %d : ourrates.N %1.14E ionise_rate %1.14E \n"
					"hnS %1.14E AreaMajor %1.14E TeV %1.14E \n",
					iVertex, ourrates.N, ionise_rate, hnS, AreaMajor, TeV);
			}

			// Let nR be the recombining amount, R is the proportion.

			AreaMajor = pVertex->AreaCell; // to match GPU
			
			f64 Ttothe5point5 = sqrtT * TeV * TeV*TeV * TeV*TeV;
			f64 hR = h_use * (ourdata.n * ourdata.n*8.75e-27*TeV) /
				(Ttothe5point5 + h_use * 2.25*TWOTHIRDS*13.6*ourdata.n*ourdata.n*8.75e-27);

			f64 recomb_rate = AreaMajor * ourdata.n * hR / h_use; // could reasonably again take hR/(1+hR) for n_k+1
			ourrates.N -= recomb_rate;
			ourrates.Nn += recomb_rate;

			if (iVertex == CHOSEN) {
				printf("CPU iVertex %d : ourrates.N %1.14E recomb_rate %1.14E \n"
					"Ttothe5point5 %1.14E hR %1.14E \n", iVertex,
					ourrates.N, recomb_rate, Ttothe5point5, hR);
				printf("CPU iVertex %d : NeTe %1.14E\n", iVertex, ourrates.NeTe);
			};

			ourrates.NeTe += -TWOTHIRDS * 13.6*kB_*ourrates.N + 0.5*ourdata.Tn*ionise_rate;
			ourrates.NiTi += 0.5*ourdata.Tn*ionise_rate;
			ourrates.NnTn += (ourdata.Te + ourdata.Ti)*recomb_rate;

			if (iVertex == CHOSEN) printf("CPU iVertex %d : ionisation %1.14E NeTe %1.14E\n", iVertex,
				-TWOTHIRDS * 13.6*kB_*ourrates.N + 0.5*ourdata.Tn*ionise_rate,
				ourrates.NeTe);
	
			memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));

		}
		else {
			// Not DOMAIN_VERTEX
			// Ignore flux into edge of outermost vertex I guess.

		}
		++pVertex;
	};
}

void TriMesh::AccumulateDiffusiveHeatRateAndCalcIonisationOld(f64 h_use, NTrates NTadditionrates[NUMVERTICES])
{
	// Inputs:
	// 

	// Outputs:
	// NTadditionrates augmented

	// Do tri input just to get it running, because we do not have stored the PB list for neighs.
	// But that is insane, so we will want to change this.

	// It will make much more of a headache on GPU.
	// The other option is to combine it with the advective mass & heat rates which already use minor info.
	static real const kB_to_3halves = sqrt(kB_)*kB_;
	static real const one_over_kB = 1.0 / kB_;
	static real const over_sqrt_m_e_ = 1.0 / sqrt(m_e_);

	Vertex * pVertex = X;
	long iVertex;
	long izNeigh[MAXNEIGH];
	long izTri[MAXNEIGH];
	char szPBCtri[MAXNEIGH];
	long neigh_len, tri_len;
	plasma_data tridata1, tridata2, neighdata, ourdata, prevdata, nextdata;
	int i, iprev;
	NTrates ourrates, yourrates;
	f64 AreaMajor;
	f64_vec2 prev_overall_v, here_overall_v, vxy, edge_normal, motion_edge;
	f64_vec3 v_n;
	f64 massflux, heatflux;
	f64 nu_eHeart1, nu_eHeart2, nu_eiBar, nu_in_visc, nu_ni_visc, nu_nn_visc1, nu_ii,
		nu_iHeart1, nu_iHeart2, nu_nn_visc2, nu_iHeart, nu_nn_visc,
		Area_quadrilateral, kappa_parallel, nu_eHeart;
	f64_vec2 integrated_grad_Te, integrated_grad_Ti, integrated_grad_Tn, gradTe, gradTi, gradTn,
		kappa_grad_T;
	f64_vec3 omega;
	Tensor2 kappa;
	int inext;
	static real const SQRT2 = sqrt(2.0);

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			AreaMajor = 0.0;

			memcpy(&ourdata, pData + iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

			// The idea is simply to look at each edge. We take n from shard model on the upwind side. T is averaged, 
			// for now do simple average instead of honeycomb. 
			// v comes from the two triangles.

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			tri_len = pVertex->GetTriIndexArray(izTri);
			// and as usual we assume tri 0 has neigh 0 on the clockwise side of it.
			memcpy(szPBCtri, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);

			memcpy(&tridata1, pData + izTri[tri_len - 1], sizeof(plasma_data));
			iprev = neigh_len - 1;
			if (szPBCtri[iprev] == ROTATE_ME_CLOCKWISE) {
				tridata1.B = Clockwise3 * tridata1.B; // what we use from minors
				tridata1.pos = Clockwise * tridata1.pos;
			};
			if (szPBCtri[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				tridata1.B = Anticlockwise3 * tridata1.B;
				tridata1.pos = Anticlockwise * tridata1.pos;
			};
			// Get T x3:
			memcpy(&prevdata, pData + BEGINNING_OF_CENTRAL + izNeigh[iprev], sizeof(plasma_data));
			memcpy(&neighdata, pData + izNeigh[0] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			// If we are doing this, we don't really need to load from tridata as well
			// We could get B, n and pos from neighbours.
			// It is guaranteed that pos is the average.
			// what is the name of the PB array for neighbours?
			
			// It seems we do not store it. ??
			// DASH IT

			f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;

			TeV = tridata1.Te * one_over_kB;
			Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc);
			sqrt_T = sqrt(tridata1.Te);
			nu_en_visc = tridata1.n_n * sigma_visc*sqrt_T * over_sqrt_m_e_;
			nu_eiBar = nu_eiBarconst_ * kB_to_3halves*tridata1.n*Get_lnLambda(tridata1.n, tridata1.Te) / (tridata1.Te*sqrt_T);
			nu_eHeart1 = (nu_en_visc + 1.87*nu_eiBar);
			
			TeV = tridata1.Ti*one_over_kB;
			Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc); // could easily save one call
			sqrt_T = sqrt(tridata1.Ti); // again not that hard to save one call
			nu_in_visc = tridata1.n_n * sigma_visc*sqrt(tridata1.Ti / m_i_ + tridata1.Tn / m_n_);
			nu_ni_visc = nu_in_visc * (tridata1.n / tridata1.n_n);
			nu_nn_visc1 = tridata1.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section(tridata1.Tn / kB_)
				* sqrt(tridata1.Tn / m_n_);
			nu_ii = tridata1.n*kB_to_3halves*Get_lnLambda_ion(tridata1.n, tridata1.Ti) / (2.07e7*SQRT2*sqrt_T*tridata1.Ti);

			nu_iHeart1 = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu_nn_visc1);

			// Why did I go and do it this way? 
			// nu depends log-linearly on n so why are we averaging n, we could
			// just average nu???

			// OK this was a stupid way round??
			// We actually use: nu_eHeart, nu_iHeart, nu_nn_visc. Is that correct?
			// B is mostly smooth and flattish.


			for (i = 0; i < neigh_len; i++)
			{
				inext = i + 1; if (inext == neigh_len) inext = 0;
				memcpy(&tridata2, pData + izTri[i], sizeof(plasma_data));
				memcpy(&nextdata, pData + izNeigh[inext] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
				
				if (szPBCtri[i] == ROTATE_ME_CLOCKWISE) {
					tridata2.B = Clockwise3 * tridata2.B;
					tridata2.pos = Clockwise * tridata2.pos;
				};
				if (szPBCtri[i] == ROTATE_ME_ANTICLOCKWISE) {
					tridata2.B = Anticlockwise3 * tridata2.B;
					tridata2.pos = Anticlockwise * tridata2.pos;
				};

				edge_normal.x = tridata2.pos.y - tridata1.pos.y;
				edge_normal.y = tridata1.pos.x - tridata2.pos.x;
	
				AreaMajor += 0.5*edge_normal.x*(tridata1.pos.x + tridata2.pos.x);

				// Get grad T on quadrilateral:
				// Might as well load NEIGHBOUR T for these gradients.

				integrated_grad_Te.x = 0.5*(
					(ourdata.Te + nextdata.Te)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Te + ourdata.Te)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Te + prevdata.Te)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Te + neighdata.Te)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Te.y = -0.5*( // notice minus
					(ourdata.Te + nextdata.Te)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Te + ourdata.Te)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Te + prevdata.Te)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Te + neighdata.Te)*(nextdata.pos.x - neighdata.pos.x)
					);

				integrated_grad_Ti.x = 0.5*(
					(ourdata.Ti + nextdata.Ti)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Ti + ourdata.Ti)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Ti + prevdata.Ti)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Ti + neighdata.Ti)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Ti.y = -0.5*( // notice minus
					(ourdata.Ti + nextdata.Ti)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Ti + ourdata.Ti)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Ti + prevdata.Ti)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Ti + neighdata.Ti)*(nextdata.pos.x - neighdata.pos.x)
					);
				integrated_grad_Tn.x = 0.5*(
					(ourdata.Tn + nextdata.Tn)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.Tn + ourdata.Tn)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.Tn + prevdata.Tn)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.Tn + neighdata.Tn)*(nextdata.pos.y - neighdata.pos.y)
					);
				integrated_grad_Tn.y = -0.5*( // notice minus
					(ourdata.Tn + nextdata.Tn)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevdata.Tn + ourdata.Tn)*(prevdata.pos.x - ourdata.pos.x)
					+ (neighdata.Tn + prevdata.Tn)*(neighdata.pos.x - prevdata.pos.x)
					+ (nextdata.Tn + neighdata.Tn)*(nextdata.pos.x - neighdata.pos.x)
					);
				Area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (neighdata.pos.x + prevdata.pos.x)*(neighdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + neighdata.pos.x)*(nextdata.pos.y - neighdata.pos.y)
					);

				gradTe = integrated_grad_Te / Area_quadrilateral;
				gradTi = integrated_grad_Ti / Area_quadrilateral;
				gradTn = integrated_grad_Tn / Area_quadrilateral;

				// =================
				// Electrons:

				// nu_eHeart:
				// n WILL BE CANCELLED from kappa_parallel and basically from kappa until omega ~ nu
				// We can perfectly easily use n,T,B from the major data;
				// and thus avoid loading from minors.
				// We only did not do already because need to make PB char list !!!
				
				//Te = THIRD * (ourdata.Te + nextdata.Te + neighdata.Te);
				TeV = tridata2.Te*one_over_kB;
				Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc); // could easily save one call
				sqrt_T = sqrt(tridata2.Te); // again not that hard to save one call
				nu_en_visc = tridata2.n_n * sigma_visc*sqrt_T*over_sqrt_m_e_;
				// could store nu_eHeart, kappa_parallel from previous pass through loop of course.

				nu_eiBar = nu_eiBarconst_ * kB_to_3halves*tridata2.n*Get_lnLambda(tridata2.n, tridata2.Te) / (tridata2.Te*sqrt_T);
				nu_eHeart2 = (nu_en_visc + 1.87*nu_eiBar);
				
				// Notice that n divides through as we have nu beneath.
				// So we'd rather average kappa_parallel than average n
				// in its numerator.
				kappa_parallel = 2.5*0.5*(
					  tridata1.n*tridata1.Te/ (m_e_*nu_eHeart1) 
					+ tridata2.n*tridata2.Te/ (m_e_*nu_eHeart2));
				nu_eHeart = 0.5*(nu_eHeart1 + nu_eHeart2);

				omega = eovermc_ * 0.5*(tridata1.B+tridata2.B);
				f64 omega_sq = omega.dot(omega);
				kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart +omega.x*omega.x) / (nu_eHeart * nu_eHeart +omega_sq);
				kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

				kappa_grad_T.x = kappa.xx*gradTe.x + kappa.xy*gradTe.y;
				kappa_grad_T.y = kappa.yx*gradTe.x + kappa.yy*gradTe.y;

				// if the outward gradient of T is positive, inwardheatflux is positive.
				ourrates.NeTe += TWOTHIRDS * kappa_grad_T.dot(edge_normal);

				// ------------------------------------------------------------------------
				// Ion:
				
				TeV = tridata2.Ti*one_over_kB;
				Estimate_Ion_Neutral_Cross_sections(TeV, &sigma_MT, &sigma_visc); // could easily save one call
				sqrt_T = sqrt(tridata2.Ti); // again not that hard to save one call
				nu_in_visc = tridata2.n_n * sigma_visc*sqrt(tridata2.Ti / m_i_ + tridata2.Tn / m_n_);
				nu_ni_visc = nu_in_visc * (tridata2.n / tridata2.n_n);
				nu_nn_visc2 = tridata2.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section(tridata2.Tn *one_over_kB)
					* sqrt(tridata2.Tn / m_n_);
				nu_ii = tridata2.n*kB_to_3halves*Get_lnLambda_ion(tridata2.n, tridata2.Ti) / (2.07e7*SQRT2*sqrt_T*tridata2.Ti);
				
				nu_iHeart2 = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu_nn_visc2);
				
				kappa_parallel = 2.5*0.5*(
					tridata1.n*tridata1.Te / (m_i_*nu_iHeart1)
					+ tridata2.n*tridata2.Te/ (m_i_*nu_iHeart2));
				nu_iHeart = 0.5*(nu_iHeart1 + nu_iHeart2);
				omega = qoverMc_ * 0.5*(tridata1.B + tridata2.B);

				// We could arrange to just store nu_iHeart as we go around.

				kappa.xx = kappa_parallel * (nu_iHeart*nu_iHeart + omega.x*omega.x) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.xy = kappa_parallel * (omega.x*omega.y + nu_iHeart * omega.z) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.yx = kappa_parallel * (omega.x*omega.y - nu_iHeart * omega.z) / (nu_iHeart * nu_iHeart + omega_sq);
				kappa.yy = kappa_parallel * (omega.y*omega.y + nu_iHeart * nu_iHeart) / (nu_iHeart * nu_iHeart + omega_sq);

				kappa_grad_T.x = kappa.xx*gradTi.x + kappa.xy*gradTi.y;
				kappa_grad_T.y = kappa.yx*gradTi.x + kappa.yy*gradTi.y;

				ourrates.NiTi += TWOTHIRDS * kappa_grad_T.dot(edge_normal);

				// ----------------------------------------------------------------------
				// Neutral:

				// HERE IS WHERE IT'S A POINT OF SOME DOUBT: 10 or far less?
				kappa_parallel = 10.0*0.5*(tridata1.n_n*tridata1.Tn / (m_n_*nu_nn_visc1)
										+ tridata2.n_n * tridata2.Tn / (m_n_*nu_nn_visc2));

				kappa_grad_T.x = kappa_parallel * gradTn.x;
				kappa_grad_T.y = kappa_parallel * gradTn.y;
				
				ourrates.NnTn += TWOTHIRDS * kappa_grad_T.dot(edge_normal);

				// increments:
				memcpy(&tridata1, &tridata2, sizeof(plasma_data)); // Could save time with pointer switch
				memcpy(&prevdata, &neighdata, sizeof(plasma_data)); // ditto
				memcpy(&neighdata, &nextdata, sizeof(plasma_data)); // ditto
				
				// Let's see if we can avoid the tridata one and avoid tridata entirely.
				nu_eHeart1 = nu_eHeart2;
				nu_iHeart1 = nu_iHeart2;
				nu_nn_visc1 = nu_nn_visc2;
				iprev = i;
			};

			// now add IONISATION:
			
			TeV = ourdata.Te / kB_;
			f64 sqrtT = sqrt(TeV);

			f64 temp = 1.0e-5*exp(-13.6 / TeV) / (13.6*(6.0*13.6 + TeV));
			// Let h n n_n S be the ionising amount,
			// h n S is the proportion of neutrals! Make sure we do not run out!
			f64 hnS = (h_use*ourdata.n*TeV*temp) /
				(sqrtT + h_use * ourdata.n_n*ourdata.n*temp*SIXTH*13.6);
			f64 ionise_rate = AreaMajor * ourdata.n_n*hnS / (h_use*(1 + hnS));
			// ionise_amt / h

			ourrates.N += ionise_rate;
			ourrates.Nn += -ionise_rate;

	//		if (iVertex == CHOSEN) {
	//			printf("CPU iVertex %d : ourrates.N %1.14E ionise_rate %1.14E \n"
	//				"hnS %1.14E AreaMajor %1.14E TeV %1.14E \n",
	//				iVertex, ourrates.N, ionise_rate, hnS, AreaMajor, TeV);
	//		}

			// Let nR be the recombining amount, R is the proportion.

			f64 Ttothe5point5 = sqrtT * TeV * TeV*TeV * TeV*TeV;
			f64 hR = h_use * (ourdata.n * ourdata.n*8.75e-27*TeV) /
				(Ttothe5point5 + h_use * 2.25*TWOTHIRDS*13.6*ourdata.n*ourdata.n*8.75e-27);

			f64 recomb_rate = AreaMajor * ourdata.n * hR / h_use; // could reasonably again take hR/(1+hR) for n_k+1
			ourrates.N -= recomb_rate;
			ourrates.Nn += recomb_rate;

		//	if (iVertex == CHOSEN) {
		//		printf("CPU iVertex %d : ourrates.N %1.14E recomb_rate %1.14E \n"
		//			"Ttothe5point5 %1.14E hR %1.14E \n", iVertex,
		//			ourrates.N, recomb_rate, Ttothe5point5, hR);
		//	};

			ourrates.NeTe += -TWOTHIRDS * 13.6*kB_*ourrates.N + 0.5*ourdata.Tn*ionise_rate;
			ourrates.NiTi += 0.5*ourdata.Tn*ionise_rate;
			ourrates.NnTn += (ourdata.Te + ourdata.Ti)*recomb_rate;
			
			memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
			
		} else {
			// Not DOMAIN_VERTEX
			// Ignore flux into edge of outermost vertex I guess.

		}
		++pVertex;
	};
}

/*void Vertex::CreateMajorPolygon(Triangle * T, ConvexPolygon & cp)
{
	cp.Clear();
	long izTri[128];
	long neigh_len = this->GetTriIndexArray(izTri);
	int i;
	Triangle * pTri;
	f64_vec2 cente;
	for (i = 0; i < neigh_len; i++)
	{	
		pTri = T + izTri[i];
		cente = pTri->cent; 

		if (pTri->periodic != 0) {
			if (this->pos.x > 0.0) cente = Clockwise * cente;
		} // SO ASSUMING HERE THAT PERIODIC TRI CENTROID IS ALLLWAAAYYYYSSS PLACED ON LEFT.

		cp.add(cente);
	}
}
*/
void TriMesh::CreateShardModelOfDensities_And_SetMajorArea()
{
	// Inputs:
	// n
	// pTri->cent
	// izTri
	// pTri->periodic
	// pVertex->pos

	// Outputs:
	// pVertex->AreaCell
	// n_shards[iVertex]
	// Tri_n_n_lists[izTri[i]][o1 * 2] <--- 0 if not set by domain vertex
		
	// CALL AVERAGE OF n TO TRIANGLES - SIMPLE AVERAGE - BEFORE WE BEGIN
	// MUST ALSO POPULATE pVertex->AreaCell with major cell area
	f64 ndesire_n[MAXNEIGH], ndesire[MAXNEIGH];
	ConvexPolygon cp;
	long izTri[128];
	long iVertex;
	int iNeigh, tri_len;
	f64 N_n, N, interpolated_n, interpolated_n_n;
	long i,inext,o1,o2;
	Triangle * pTri;
	Vertex * pVertex;
	
	memset(Tri_n_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);
	memset(Tri_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);

	pVertex = X;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX) {
			cp.Clear();
			tri_len = pVertex->GetTriIndexArray(izTri);
			f64_vec2 cente;
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				cente = pTri->cent;
				if (pTri->periodic != 0) {
					if (pVertex->pos.x > 0.0) cente = Clockwise * cente;
				} // SO ASSUMING HERE THAT PERIODIC TRI CENTROID IS ALLLWAAAYYYYSSS PLACED ON LEFT.
				cp.add(cente);
			} // triangle centroids are corners
			pVertex->AreaCell = cp.GetArea();

			for (iNeigh = 0; iNeigh < tri_len; iNeigh++)
			{
				ndesire_n[iNeigh] = pData[izTri[iNeigh]].n_n; // insert apparent triangle average
				ndesire[iNeigh] = pData[izTri[iNeigh]].n;
		//		if (iVertex == CHOSEN) printf("CPU CHOSEN %d ndesire %1.14E \n", iNeigh, ndesire[iNeigh]);
			};
			N_n = pData[BEGINNING_OF_CENTRAL + iVertex].n_n*pVertex->AreaCell;
			N = pData[BEGINNING_OF_CENTRAL + iVertex].n*pVertex->AreaCell;

			if (iVertex == CHOSEN) {
				iVertex = iVertex;
			}
			n_shards_n[iVertex].n_cent = cp.minmod(n_shards_n[iVertex].n, ndesire_n, N_n, pVertex->pos);
			n_shards[iVertex].n_cent = cp.minmod(n_shards[iVertex].n, ndesire, N, pVertex->pos);

	//		if (iVertex == CHOSEN) {
	//			printf("Conclusions:\n");
	//			for (iNeigh = 0; iNeigh < tri_len; iNeigh++)
	//				printf("CPU : %1.14E\n", n_shards[iVertex].n[iNeigh]);
	//			printf("CPU n_cent = %1.14E\n", n_shards[iVertex].n_cent);
	//		};

			for (i = 0; i < cp.numCoords; i++)
			{
				// for 2 triangles each corner:

				// first check which number corner this vertex is
				// make sure we enter them in order that goes anticlockwise for the 
				// Then we need to make izMinorNeigh match this somehow

				// Let's say izMinorNeigh goes [across corner 0, across edge 2, corner 1, edge 0, corner 2, edge 1]
				// We want 0,1 to be the values corresp corner 0.

				// shard value 0 is in tri 0. We look at each pair of shard values in turn to interpolate.

				inext = i + 1; if (inext == cp.numCoords) inext = 0;

				interpolated_n = THIRD * (n_shards[iVertex].n[i] + n_shards[iVertex].n[inext] + n_shards[iVertex].n_cent);
				interpolated_n_n = THIRD * (n_shards_n[iVertex].n[i] + n_shards_n[iVertex].n[inext] + n_shards_n[iVertex].n_cent);
				// contribute to tris i and inext:
				o1 = (T + izTri[i])->GetCornerIndex(X + iVertex);
				o2 = (T + izTri[inext])->GetCornerIndex(X + iVertex);
				
				// Now careful which one's which:

				// inext sees this point as more anticlockwise.

				Tri_n_lists[izTri[inext]][o2 * 2 + 1] = interpolated_n;
				Tri_n_lists[izTri[i]][o1 * 2] = interpolated_n;

				Tri_n_n_lists[izTri[inext]][o2 * 2 + 1] = interpolated_n_n;
				Tri_n_n_lists[izTri[i]][o1 * 2] = interpolated_n_n;
			};
		} else {

			// NOT A DOMAIN VERTEX 
			memset(&(n_shards_n[iVertex]), 0, sizeof(ShardModel));
			memset(&(n_shards[iVertex]), 0, sizeof(ShardModel));
	//		if (iVertex == CHOSEN) printf("CPU : %d was not a domain vertex\n", CHOSEN);
		}
		
		
		++pVertex;
	};

}


void TriMesh::AccumulateAdvectiveMassHeatRate(
	f64_vec2 p_overall_v[NMINOR],
	//	ShardModel n_shards_n[NUMVERTICES],
	//	ShardModel n_shards[NUMVERTICES],
	// outputs:
	NTrates AdditionalNT[NUMVERTICES],
	f64 * p_n_upwind,
	f64 * p_nn_upwind
	//	f64 p_div_v_neut[NUMVERTICES], 
	//	f64 p_div_v[NUMVERTICES]
)
{
	// Inputs:
	// n_shards[iVertex]
	// data n,v,T
	//  MajorTriPBC[iVertex]
	// p_overall_v
	// tridata.pos 

	// Outputs:
	// AdditionalNT[izNeigh[i]]
	// p_div_v, p_div_v_neut, Integrated_Div_v_overall

	Vertex * pVertex = X;
	long iVertex;
	long izNeigh[MAXNEIGH];
	long izTri[MAXNEIGH];
	char szPBCtri[MAXNEIGH];
	long neigh_len, tri_len;
	plasma_data tridata1, tridata2, neighdata, ourdata;
	int i, iprev;
	NTrates ourrates;
	f64 Div_v_neut, Div_v, n, Integral_div_v_overall, Area;
	f64_vec2 prev_overall_v, next_overall_v, vxy, edge_normal, motion_edge;
	f64_vec3 v_n;
	f64 massflux, heatflux;
	f64 n_upwind_prev, n_upwind_next, nn_upwind_prev, nn_upwind_next;
	
	// Ensure that if not defined due to domain vertex, the following will be 0:
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(Integrated_Div_v_overall, 0, sizeof(f64)*NUMVERTICES);

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			Div_v_neut = 0.0;
			Div_v = 0.0;
			Integral_div_v_overall = 0.0;
			Area = 0.0;

			// On GPU we would try to do gather not scatter by holding shardmodels in shared memory?
			// 48K shared memory. 128 bytes per shardmodel. Room for 48*8 if no other memory. ~ 384. 
			// Yet the typical shard model list is only 6. Is there nothing we can do more cleverly than using 16?
			// In case that a vertex has over say 10 neighbours we could have a special flag.
			// neigh 9 carries flag -1 and neigh 10 carries index of extra list. Complicated. But it could run a while with < 12.
			// 12*8 = 96 bytes / shardmodel. 48*1024/96 = 512 shardmodels in 48KB.
			// All to save a random access to set "yourrates" increment. Well, is it atomic.
			memcpy(&ourdata, pData + iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			memcpy(&ourrates, AdditionalNT + iVertex, sizeof(NTrates));

			// The idea is simply to look at each edge. We take n from shard model on the upwind side. 
			// T is averaged, for now do simple average instead of honeycomb. 
			// v comes from the two triangles.

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			tri_len = pVertex->GetTriIndexArray(izTri);
			// and as usual we assume tri 0 has neigh 0 on the clockwise side of it.

			memcpy(szPBCtri, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);

			//	pVertex->GetTriPBCArray(szPBCtri);
			// SHOULD BE WHAT APPLIES FOR GETTING MINOR PBC : IT'S SAME ARRAY

			memcpy(&tridata1, pData + izTri[tri_len - 1], sizeof(plasma_data));
			iprev = neigh_len - 1;
			prev_overall_v = p_overall_v[izTri[iprev]];
			if (szPBCtri[iprev] == ROTATE_ME_CLOCKWISE) {
				tridata1.v_n = Clockwise3 * tridata1.v_n;
				tridata1.vxy = Clockwise * tridata1.vxy;
				prev_overall_v = Clockwise * prev_overall_v;
				tridata1.pos = Clockwise * tridata1.pos;
			};
			if (szPBCtri[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				tridata1.v_n = Anticlockwise3 * tridata1.v_n;
				tridata1.vxy = Anticlockwise * tridata1.vxy;
				prev_overall_v = Anticlockwise * prev_overall_v;
				tridata1.pos = Anticlockwise * tridata1.pos;
			};
			n_upwind_prev = p_n_upwind[izTri[tri_len - 1]];
			nn_upwind_prev = p_nn_upwind[izTri[tri_len - 1]];

			for (i = 0; i < neigh_len; i++)
			{
				memcpy(&tridata2, pData + izTri[i], sizeof(plasma_data));
				memcpy(&neighdata, pData + izNeigh[i] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
				next_overall_v = p_overall_v[izTri[i]];
				n_upwind_next = p_n_upwind[izTri[i]];
				nn_upwind_next = p_nn_upwind[izTri[i]];

				if (szPBCtri[i] == ROTATE_ME_CLOCKWISE) {
					tridata2.v_n = Clockwise3 * tridata2.v_n;
					tridata2.vxy = Clockwise * tridata2.vxy;
					next_overall_v = Clockwise * next_overall_v;
					tridata2.pos = Clockwise * tridata2.pos;
				};
				if (szPBCtri[i] == ROTATE_ME_ANTICLOCKWISE) {
					tridata2.v_n = Anticlockwise3 * tridata2.v_n;
					tridata2.vxy = Anticlockwise * tridata2.vxy;
					next_overall_v = Anticlockwise * next_overall_v;
					tridata2.pos = Anticlockwise * tridata2.pos;
				};

				edge_normal.x = tridata2.pos.y - tridata1.pos.y;
				edge_normal.y = tridata1.pos.x - tridata2.pos.x;
				// We don't need to have cp at all actually. We loaded positions in other routine. ?

				Area += 0.5*edge_normal.x*(tridata1.pos.x + tridata2.pos.x);
				Div_v_neut += 0.5*(tridata1.v_n + tridata2.v_n).dotxy(edge_normal);
				Div_v += 0.5*(tridata1.vxy + tridata2.vxy).dot(edge_normal);
				Integral_div_v_overall += 0.5*(prev_overall_v + next_overall_v).dot(edge_normal); // Average outward velocity of edge...
				
				ourrates.N -= 0.5*(n_upwind_prev*(tridata1.vxy - prev_overall_v)
					+ n_upwind_next*(tridata2.vxy - next_overall_v)).dot(edge_normal);
				ourrates.Nn -= 0.5*(nn_upwind_prev*(tridata1.v_n.xypart() - prev_overall_v)
					+ nn_upwind_next*(tridata2.v_n.xypart() - next_overall_v)).dot(edge_normal);
				ourrates.NiTi -= 0.5*(n_upwind_prev*tridata1.Ti*(tridata1.vxy - prev_overall_v)
					+ n_upwind_next*tridata2.Ti*(tridata2.vxy - next_overall_v)).dot(edge_normal);
				ourrates.NeTe -= 0.5*(n_upwind_prev*tridata1.Te*(tridata1.vxy - prev_overall_v)
					+ n_upwind_next*tridata2.Te*(tridata2.vxy - next_overall_v)).dot(edge_normal);				
				ourrates.NnTn -= 0.5*(nn_upwind_prev*tridata1.Tn*(tridata1.v_n.xypart() - prev_overall_v)
					+ nn_upwind_next*tridata2.Tn*(tridata2.v_n.xypart() - next_overall_v)).dot(edge_normal);
				
		//		if (iVertex == CHOSEN) printf("CPU %d i %d contrib %1.14E \n"
		//			"nn_upwind_prev %1.14E nn_upwind_next %1.14E \n"
		//			"tridata1.v_n %1.14E %1.14E prev_overall_v %1.14E %1.14E \n"
		//			"tridata2.v_n %1.14E %1.14E next_overall_v %1.14E %1.14E \n"
		//			"edge_normal %1.14E %1.14E \n---------------------------------------\n",
		//			CHOSEN, i,
		//			0.5*(nn_upwind_prev*(tridata1.v_n.xypart() - prev_overall_v)
		//				+ nn_upwind_next*(tridata2.v_n.xypart() - next_overall_v)).dot(edge_normal),
		//			nn_upwind_prev, nn_upwind_next, tridata1.v_n.x, tridata1.v_n.y,
		//			prev_overall_v.x, prev_overall_v.y, tridata2.v_n.x, tridata2.v_n.y,
		//			next_overall_v.x, next_overall_v.y, 
		//			edge_normal.x, edge_normal.y
		//			);


		//		if (iVertex == CHOSEN) printf("CPU %d : contrib %1.10E n_upwind %1.9E %1.9E\n"
		//			"vnext %1.9E %1.9E overall %1.9E %1.9E edge_normal %1.8E %1.8E \n", CHOSEN, 
		//			-0.5*(n_upwind_prev*(tridata1.vxy - prev_overall_v)	+ n_upwind_next*(tridata2.vxy - next_overall_v)).dot(edge_normal),
		//			n_upwind_prev, n_upwind_next,
		//			tridata2.vxy.x, tridata2.vxy.y, next_overall_v.x, next_overall_v.y, edge_normal.x, edge_normal.y);

				memcpy(&tridata1, &tridata2, sizeof(plasma_data)); // Could save time with pointer switch
				prev_overall_v = next_overall_v;
				n_upwind_prev = n_upwind_next;
				nn_upwind_prev = nn_upwind_next;
				iprev = i;
			};
			if (iVertex == CHOSEN) printf("CPU %d : ourrates.Nn %1.10E \n", CHOSEN, ourrates.Nn);
			memcpy(AdditionalNT + iVertex, &ourrates, sizeof(NTrates));
			p_div_v[iVertex] = Div_v / Area;
			p_div_v_neut[iVertex] = Div_v_neut / Area;
			::Integrated_Div_v_overall[iVertex] = Integral_div_v_overall;
		}
		else {
			// Not DOMAIN_VERTEX
			// Ignore flux into edge of outermost vertex I guess.

		}
		++pVertex;
	};
}
void TriMesh::AccumulateAdvectiveMassHeatRateOld(
	f64_vec2 p_overall_v[NMINOR], 
//	ShardModel n_shards_n[NUMVERTICES],
//	ShardModel n_shards[NUMVERTICES],
	// outputs:
	NTrates AdditionalNT[NUMVERTICES]
//	f64 p_div_v_neut[NUMVERTICES], 
//	f64 p_div_v[NUMVERTICES]
)
{

	// Inputs:
	// n_shards[iVertex]
	// data n,v,T
	//  MajorTriPBC[iVertex]
	// p_overall_v
	// tridata.pos 

	// Outputs:
	// AdditionalNT[izNeigh[i]]
	// p_div_v, p_div_v_neut, Integrated_Div_v_overall

	Vertex * pVertex = X;
	long iVertex;
	long izNeigh[MAXNEIGH];
	long izTri[MAXNEIGH];
	char szPBCtri[MAXNEIGH];
	long neigh_len, tri_len;
	ShardModel n_shards_n_, n_shards_;
	plasma_data tridata1, tridata2, neighdata, ourdata;
	int i, iprev;
	NTrates ourrates, yourrates;
	f64 Div_v_neut, Div_v, n, Integral_div_v_overall,Area;
	f64_vec2 prev_overall_v, here_overall_v, vxy, edge_normal, motion_edge;
	f64_vec3 v_n;
	f64 massflux, heatflux;

	// Ensure that if not defined due to domain vertex, the following will be 0:
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(Integrated_Div_v_overall, 0, sizeof(f64)*NUMVERTICES);

	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			Div_v_neut = 0.0;
			Div_v = 0.0;
			Integral_div_v_overall = 0.0;
			Area = 0.0;

			memcpy(&n_shards_n_, n_shards_n + iVertex, sizeof(ShardModel));
			memcpy(&n_shards_, n_shards + iVertex, sizeof(ShardModel));
			// On GPU we would try to do gather not scatter by holding shardmodels in shared memory?
			// 48K shared memory. 128 bytes per shardmodel. Room for 48*8 if no other memory. ~ 384. 
			// Yet the typical shard model list is only 6. Is there nothing we can do more cleverly than using 16?
			// In case that a vertex has over say 10 neighbours we could have a special flag.
			// neigh 9 carries flag -1 and neigh 10 carries index of extra list. Complicated. But it could run a while with < 12.
			// 12*8 = 96 bytes / shardmodel. 48*1024/96 = 512 shardmodels in 48KB.
			// All to save a random access to set "yourrates" increment. Well, is it atomic.
			memcpy(&ourdata, pData + iVertex + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
			memcpy(&ourrates, AdditionalNT + iVertex, sizeof(NTrates));
			
			// The idea is simply to look at each edge. We take n from shard model on the upwind side. 
			// T is averaged, for now do simple average instead of honeycomb. 
			// v comes from the two triangles.

			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			tri_len = pVertex->GetTriIndexArray(izTri);
			// and as usual we assume tri 0 has neigh 0 on the clockwise side of it.
			
			memcpy(szPBCtri, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);
			
		//	pVertex->GetTriPBCArray(szPBCtri);
			// SHOULD BE WHAT APPLIES FOR GETTING MINOR PBC : IT'S SAME ARRAY

			memcpy(&tridata1, pData + izTri[tri_len - 1], sizeof(plasma_data));
			iprev = neigh_len - 1;
			prev_overall_v = p_overall_v[izTri[iprev]];
			if (szPBCtri[iprev] == ROTATE_ME_CLOCKWISE) {
				tridata1.v_n = Clockwise3 * tridata1.v_n;
				tridata1.vxy = Clockwise * tridata1.vxy;
				prev_overall_v = Clockwise * prev_overall_v;
				tridata1.pos = Clockwise * tridata1.pos;
			};
			if (szPBCtri[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				tridata1.v_n = Anticlockwise3 * tridata1.v_n;
				tridata1.vxy = Anticlockwise * tridata1.vxy;
				prev_overall_v = Anticlockwise * prev_overall_v;
				tridata1.pos = Anticlockwise * tridata1.pos;
			};

			for (i = 0; i < neigh_len; i++)
			{
				memcpy(&tridata2, pData + izTri[i], sizeof(plasma_data)); 
				memcpy(&neighdata, pData + izNeigh[i] + BEGINNING_OF_CENTRAL, sizeof(plasma_data));
				memcpy(&yourrates, AdditionalNT + izNeigh[i], sizeof(NTrates));
				here_overall_v = p_overall_v[izTri[i]];

				if (szPBCtri[i] == ROTATE_ME_CLOCKWISE) {
					tridata2.v_n = Clockwise3 * tridata2.v_n;
					tridata2.vxy = Clockwise * tridata2.vxy;
					here_overall_v = Clockwise * here_overall_v;
					tridata2.pos = Clockwise * tridata2.pos;
				};
				if (szPBCtri[i] == ROTATE_ME_ANTICLOCKWISE) {
					tridata2.v_n = Anticlockwise3 * tridata2.v_n;
					tridata2.vxy = Anticlockwise * tridata2.vxy;
					here_overall_v = Anticlockwise * here_overall_v;
					tridata2.pos = Anticlockwise * tridata2.pos;
				};

				v_n = 0.5*(tridata1.v_n + tridata2.v_n);
				vxy = 0.5*(tridata1.vxy + tridata2.vxy);
				
				edge_normal.x = tridata2.pos.y - tridata1.pos.y;
				edge_normal.y = tridata1.pos.x - tridata2.pos.x;
				// We don't need to have cp at all actually. We loaded positions in other routine. ?

				motion_edge = 0.5*(prev_overall_v + here_overall_v);

				Area += 0.5*edge_normal.x*(tridata1.pos.x + tridata2.pos.x);
				Div_v_neut += v_n.dotxy(edge_normal);
				Div_v += vxy.dot(edge_normal);
				Integral_div_v_overall += motion_edge.dot(edge_normal); // Average outward velocity of edge...

				f64 relvnormal = (v_n.xypart() - motion_edge).dot(edge_normal);
				if (relvnormal > 0.0)
				{
					// Upwind n:
					n = 0.5*(n_shards_n_.n[iprev] + n_shards_n_.n[i]);
					// looking at neigh 0, tri -1 and tri 0 are there.
					massflux =n*relvnormal;
					heatflux = n * 0.5*(ourdata.Tn + neighdata.Tn)*relvnormal;
					ourrates.Nn -= massflux;
					ourrates.NnTn -= heatflux;
					yourrates.Nn += massflux;
					yourrates.NnTn += heatflux;
				};

				relvnormal = (vxy - motion_edge).dot(edge_normal);
	//			if (iVertex == CHOSEN) printf("advect CPU %d : i %d relvnorm %1.9E ", iVertex,i,relvnormal); 
	// Note if relvnormal < 0 then n,massflux wasn't set
				if (relvnormal > 0.0)
				{
					n = 0.5*(n_shards_.n[iprev] + n_shards_.n[i]);					
					massflux = n*relvnormal;
					heatflux = n * 0.5*(ourdata.Ti + neighdata.Ti)*relvnormal;
					ourrates.N -= massflux;
					ourrates.NiTi -= heatflux;
					yourrates.N += massflux;
					yourrates.NiTi += heatflux;
					heatflux = n * 0.5*(ourdata.Te + neighdata.Te)*relvnormal;
					ourrates.NeTe -= heatflux;
					yourrates.NeTe += heatflux;

			//		if (iVertex == CHOSEN) printf("n %1.9E massflux %1.9E ", n, massflux);
				};
			//	if (iVertex == CHOSEN) printf("\n");
				memcpy(AdditionalNT + izNeigh[i], &yourrates, sizeof(NTrates)); // upwind...
				// increments:
				memcpy(&tridata1, &tridata2, sizeof(plasma_data)); // Could save time with pointer switch
				prev_overall_v = here_overall_v;
				iprev = i;
			};
			memcpy(AdditionalNT + iVertex, &ourrates, sizeof(NTrates));
			p_div_v[iVertex] = Div_v/Area;
			p_div_v_neut[iVertex] = Div_v_neut/Area;
			::Integrated_Div_v_overall[iVertex] = Integral_div_v_overall;
		} else {
			// Not DOMAIN_VERTEX
			// Ignore flux into edge of outermost vertex I guess.

		}
		++pVertex;
	};
}

/*void TriMesh::AccumulateAdvectiveMomRate(
	f64_vec2 p_overall_v[NMINOR],
	ShardModel n_shards_n[NUMVERTICES], ShardModel n_shards[NUMVERTICES],
	three_vec3 AdditionRateNv[NMINOR])
{
	// Look at each edge betweeen minor cells and counting once,
	// add the boundary flux of momentum to both sides.
	// n comes from shard model, v from simple average (of opposing)
	// Done for each species. 
	// Advective v is net of overall v.

	Triangle * pTri;
	long izTri[128], izNeigh[128];
	long tri_len, neigh_len;
	int i;
	f64 n, n_n, relvnormal, n_n_1, n_n_2, n1, n2, viz, vez;
	f64_vec2 motion_edge, edge_normal, ourpos, overall_v_ours, vxy;
	f64_vec3 outward_flux, v_n;
	Vertex * pVertex = X;
	plasma_data data, oppdata;
	long iMinor, iDestTri;
	three_vec3 ownrates, opprates;
	ShardModel n_shards_n_, n_shards_;

	long iVertex = 0;
	for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
	{
		
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			memcpy(&n_shards_n_, n_shards_n + iVertex, sizeof(ShardModel));
			memcpy(&n_shards_, n_shards + iVertex, sizeof(ShardModel));

			pVertex->CreateMajorPolygon(T, cp); // Assess afterwards: better to store cp and create once?? Creating requires many random accesses.
			overall_v_ours = p_overall_v[iMinor];
			memcpy(&data, pData + iMinor, sizeof(plasma_data));
			ourpos = pVertex->pos;
			memcpy(&ownrates, &(AdditionRateNv[iMinor]), sizeof(three_vec3));

			tri_len = pVertex->GetTriIndexArray(izTri);
			neigh_len = pVertex->GetNeighIndexArray(izNeigh);
			iprev = tri_len - 1;

			i = 0;
			neighpos1 = X[izNeigh[0]].pos; // store instead
			endpt1 = THIRD * (ourpos + ourpos + neighpos1);
			n_n_1 = Interpolate
			(endpt1,
				ourpos, cp.coord[iprev], cp.coord[i],
				n_shards_n_.n_cent,
				n_shards_n_.n[iprev],
				n_shards_n_.n[i]);;// find by interpolation
			n1 = Interpolate
			(endpt1,
				ourpos, cp.coord[iprev], cp.coord[i],
				n_shards_.n_cent,
				n_shards_.n[iprev],
				n_shards_.n[i]);;// find by interpolation

			for (i = 0; i < tri_len; i++)
			{
				inext = i + 1; if (inext == tri_len) inext = 0;

				iDestTri = izTri[i];
				//neighpos1 = X[izNeigh[i]].pos; // store instead
				neighpos2 = X[izNeigh[inext]].pos;
				endpt2 = THIRD * (ourpos + ourpos + neighpos2);
				
				n_n_2 = Interpolate
				    (endpt2,
					ourpos, cp.coord[i], cp.coord[inext],
					n_shards_n_.n_cent,
					n_shards_n_.n[i],
					n_shards_n_.n[inext]);
				n2 = Interpolate
				    (endpt2,
					ourpos, cp.coord[i], cp.coord[inext],
					n_shards_.n_cent,
					n_shards_.n[i],
					n_shards_.n[inext]);

				memcpy(&oppdata, &(pData[iDestTri]), sizeof(plasma_data));

				n_n = 0.5*(n_n_1 + n_n_2);
				n = 0.5*(n1 + n2);
				v_n = 0.5*(data.v_n + oppdata.v_n); // Not yet using honeycomb average (!)(!)
				vxy = 0.5*(data.vxy + oppdata.vxy);
				vez = 0.5*(data.vez + oppdata.vez);
				viz = 0.5*(data.viz + oppdata.viz);
				motion_edge = 0.5*(overall_v_ours + p_overall_v[iDestTri]); // this is actually correct:
				// edge v = 0.5* ( 1/3 b + 2/3 a + 1/3 c + 2/3 a ) = 0.5* ( avg + a )

				// We could consider just its own contribution, and scatter. Maybe scatter is not ideal? We could keeep
				// same data in shared on GPU for both vertex and tri minors to run at once ... they should be set at once? 

				edge_normal.x = endpt2.y - endpt1.y;
				edge_normal.y = endpt1.x - endpt2.x;

				memcpy(&opprates, &(AdditionRateNv[iDestTri]), sizeof(three_vec3));

				relvnormal = (v_n.xypart() - motion_edge).dot(edge_normal);
				outward_flux = n_n * relvnormal*v_n;
				ownrates.neut -= outward_flux;
				opprates.neut += outward_flux;

				relvnormal = (vxy - motion_edge).dot(edge_normal);
				outward_flux = n * relvnormal* Make3(vxy, viz);
				ownrates.ion -= outward_flux;
				opprates.ion += outward_flux;

				outward_flux = n * relvnormal*Make3(vxy,vez);
				ownrates.elec -= outward_flux;
				opprates.elec += outward_flux;
				
				// Now look at the minor edge between iDestTri and the next tri along

				iTriNext = izTri[inext];

				edgenormal.x = THIRD * (ourpos.y - neighpos2.y);
				edgenormal.y = THIRD * (neighpos2.x - ourpos.x);

				memcpy(&tridata2, &(pData[iTriNext]), sizeof(plasma_data));
				// not doing this as efficiently as it could be, nvm
				memcpy(&neighdata, &(pData[izNeigh[inext] + BEGINNING_OF_CENTRAL ]), sizeof(plasma_data));

				// motion_edge? average of velocity v
				motion_edge = 0.5*(overall_v_ours + p_overall_v[izNeigh[inext]]);

				// honeycomb:
				v_n = SIXTH * (2.0*tridata2.v_n + owndata.v_n + 2.0*oppdata.v_n + neighdata.v_n);
				vxy = SIXTH * (2.0*tridata2.vxy + owndata.vxy + 2.0*oppdata.vxy + neighdata.vxy);
				vez = SIXTH * (2.0*tridata2.vez + owndata.vez + 2.0*oppdata.vez + neighdata.vez);
				viz = SIXTH * (2.0*tridata2.viz + owndata.viz + 2.0*oppdata.viz + neighdata.viz);
			
				// Going to need to rotate each of these to be contig with US
				// so where does that leave flow? Also having to rotate the edge_normal
				// doing everything in OUR orientation. -- Bad news
				// Go again and do differently!!


				relvnormal = (v_n.xypart() - motion_edge).dot(edge_normal);
				outward_flux = 0.5*n_n_2 * relvnormal*v_n;
				opprates.neut -= outward_flux;
				AdditionRateNv[iTriNext].neut += outward_flux;

				relvnormal = (vxy - motion_edge).dot(edge_normal);
				outward_flux = 0.5*n2 * relvnormal*Make3(vxy,viz);
				opprates.ion -= outward_flux;
				AdditionRateNv[iTriNext].ion += outward_flux;
				outward_flux = 0.5*n2 * relvnormal*Make3(vxy, vez);
				opprates.elec -= outward_flux;
				AdditionRateNv[iTriNext].elec += outward_flux;

				memcpy(&(AdditionRateNv[iDestTri]), &opprates, sizeof(three_vec3));

				// Think carefully how to do "scatter" with GPU vers! Maybe no scatter.
				iprev = i;
				neighpos1 = neighpos2;
				endpt1 = endpt2; 
				n_n_1 = n_n_2;
				n1 = n2;
			}

			memcpy(&(AdditionRateNv[iMinor]), &ownrates, sizeof(three_vec3));

		} else {
			// not a domain vertex --> ?

			// CAREFUL : WHAT HAPPENS AT THE BACK? We have to sort that out.


		}
		pVertex++;
		iVertex++;
	};
	// Now we need to do minor-to-minor. Here check that we only look UP to destination index.

	//pTri = T;
	//for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
	//{
	//	iprev = 2;
	//	for (i = 0; i < 3; i++)
	//	{
	//		inext = i + 1; if (i == 2) inext = 0;

	//		if (pTri->neighbours[i] - T > pTri - T) {
	//			// remember corner 0 opposite edge 0
	//			iDestTri = pTri->neighbours[i] - T;
	//			// we don't know clockwise/anticlockwise order when we come to integrate grad of smth,
	//			// but we can test that edge_normal faces away from the other corner of tri.
	//			pCorner0 = pTri->cornerptr[iprev];
	//			pCorner1 = pTri->cornerptr[inext];
	//			cornerpos0 = pCorner0->pos;
	//			cornerpos1 = pCorner1->pos;

	//			memcpy(&oppdata, &(pData[iDestTri]), sizeof(plasma_data));

	//			endpt0 = THIRD * (cornerpos0 + cornerpos0 + cornerpos1);
	//			endpt1 = THIRD * (cornerpos0 + cornerpos1 + cornerpos1);
	//			edge_normal.x = endpt0.y - endpt1.y;
	//			edge_normal.y = endpt1.x - endpt0.x;
	//			if (edge_normal.dot(pTri->cornerptr[i]->pos - cornerpos0) > 0.0) {
	//				// faces into triangle
	//				edge_normal = -edge_normal;
	//			};

	//		};
	//		iprev = i;
	//	};
	//	++pTri;
	//};
}*/

void TriMesh::Create_A_from_advance(f64 hstep, f64 ROCAzduetoAdvection[], f64 Az_array[])
{
	long iMinor;
	plasma_data * pdata = pData;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		Az_array[iMinor] = pdata->Az + hstep * ROCAzduetoAdvection[iMinor] + hstep * pdata->Azdot;
		++pdata;
	};

	Triangle * pTri = T;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		// Set all frills Az to equal the adjacent Az.
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL))
		{
			Az_array[iMinor] = Az_array[pTri->neighbours[0] - T];
		};
		++pTri;
	};
}
void TriMesh::AdvanceAz(f64 hstep, f64 ROCAzduetoAdvection[], f64 Az_array[]) {
	long iMinor;
	plasma_data * pdata = pData;
	f64 maxAz = -1.0e100;
	f64 minAz = 1.0e100;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		Az_array[iMinor] += hstep * ROCAzduetoAdvection[iMinor] + hstep * pdata->Azdot;

		++pdata;
		
	};
	Triangle * pTri = T;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		// Set all frills Az to equal the adjacent Az.
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL))
		{
			Az_array[iMinor] = Az_array[pTri->neighbours[0] - T];
		};
		++pTri;
	};

}

void TriMesh::FinalStepAz(f64 hstep, f64 ROCAzduetoAdvection[], TriMesh * pDestMesh, f64 Az_array[]) {
	long iMinor;
	plasma_data * pdata = pData;
	plasma_data * pdestdata = pDestMesh->pData;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		pdestdata->Az = Az_array[iMinor] + hstep * ROCAzduetoAdvection[iMinor] + hstep * pdata->Azdot;
		++pdata;
		++pdestdata;
	};
	
	Triangle * pTri = T;
	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
	{
		// Set all frills Az to equal the adjacent Az.
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL))
		{
			pDestMesh->pData[iMinor].Az = pDestMesh->pData[pTri->neighbours[0] - T].Az;
		};
		++pTri;
	};
}

void TriMesh::Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(
	f64_vec2 p_overall_v[NMINOR],
	//ShardModel n_shards_n[NUMVERTICES], 
	//ShardModel n_shards[NUMVERTICES],
	three_vec3 AdditionRateNv[NMINOR])
{


	// Inputs:
	// data inc Az, pos 
	// MajorTriPBC
	// n_shards
	// p_overall_v
	// Tri_n_lists
	// TriMinorNeighLists
	// TriMinorPBCLists

	// Outputs:
	// IntegratedGradAz[iMinor]
	// LapAz
	// AreaMinorArray
	// GradTeArray
	// pData[iMinor].B
	// AdditionRateNv
	
	FILE * fp;
	Triangle * pTri;
	long iVertex, iMinor;
	int i, iprev, inext;
	long neigh_len, tri_len;
	f64 beta[3]; 
	f64_vec3 v_n0, v_n1;
	f64_vec2 vxy0, vxy1;
	f64 vez0, viz0, vez1, viz1;

	long izNeigh[MAXNEIGH], izTri[MAXNEIGH];
	f64 n_n0, n_n1, n0, n1, Tn0, Tn1, Ti0, Ti1, Te0, Te1, nextAz, relvnormal,Az_edge,
		Azdot_edge,area_quadrilateral;
	f64_vec2 motion_edge, edge_normal, //ourpos, 
		overall_v_ours, vxy;
	f64_vec2 endpt0, endpt1;
	f64_vec3 outward_flux, v_n;
	plasma_data ourdata, prevdata, oppdata, nextdata; // neighdata0, neighdata1;
	Vertex * pVertex = X;
//	ConvexPolygon cp;
	f64_vec2 Our_integral_curl_Az, Our_integral_grad_Te,
		Our_integral_grad_Azdot, Our_integral_grad_Az, motion_edge0, motion_edge1;
	f64 Our_integral_Lap_Az, prevAz, oppAz;
	ShardModel n_shards_n_, n_shards_;
	three_vec3 ownrates, opprates;
	char szPBC[MAXNEIGH];
	f64 AreaMinor;
	f64_vec2 integ_grad_Az;
	//// Now tris:
	//
	long izNeighMinor[6];
	f64 n_array[6];
	f64 n_n_array[6]; 
	char buffer[256];

	// Note that n_shards should perhaps be updated less frequently than we find ourselves here -- (?)
					  // In any case for sanity, we need some way of passing n_shards to where triangles can reach it in some shape or form#
					  // Memory is not at a premium. Creating lists for each tri allows us to access here this way.
					  // The alternative is to create an index into n_shards lists for each tri. 
					  // That way is still fine since on GPU we load into shared memory the n values -- we just need to have stored them.
					  // But this way seems cleaner. So how to create?
					  // 
#ifdef LAPFILE
	sprintf(buffer, "Lap74784_%d.txt", GlobalStepsCounter);
	FILE * fp1 = fopen(buffer, "w");
	sprintf(buffer, "Lap74816_%d.txt", GlobalStepsCounter);
	FILE * fp2 = fopen(buffer, "w");
	sprintf(buffer, "Lap2256_%d.txt", GlobalStepsCounter);
	FILE * fp3 = fopen(buffer, "w");
	sprintf(buffer, "Lap2292_%d.txt", GlobalStepsCounter);
	FILE * fp4 = fopen(buffer, "w");
	if ((fp1 == 0) || (fp2 == 0) || (fp3 == 0) || (fp4 == 0)) {
		printf("\n\nfile open failed.");
		while(1) getch();			
	}
#endif
	long const index[4] = { 74784, 74816, 2256, 2292 };

	iMinor = BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		// Need to pick out OUTERMOST/INNERMOST vertex as these are
		// having Az set after the event. Doesn't matter much what Lap Az we assign 
		// but probably should skip it to save crashes.
		
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Azdot.x = 0.0;
		Our_integral_grad_Azdot.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		AreaMinor = 0.0;
		tri_len = pVertex->GetTriIndexArray(izTri);
		
		iprev = tri_len - 1;
		memcpy(szPBC, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH); 
		memcpy(&ourdata, pData + iMinor, sizeof(plasma_data));
		
		i = 0; iprev = tri_len - 1;

		memcpy(&prevdata, pData + izTri[tri_len - 1], sizeof(plasma_data)); // all vertices
		if (szPBC[iprev] != 0) {
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevdata.v_n = Clockwise3 * prevdata.v_n;
				prevdata.vxy = Clockwise * prevdata.vxy;
				prevdata.pos = Clockwise * prevdata.pos;
			}
			else {
				prevdata.v_n = Anticlockwise3 * prevdata.v_n;
				prevdata.vxy = Anticlockwise * prevdata.vxy;
				prevdata.pos = Anticlockwise * prevdata.pos;
			}
		};
		//if (iVertex == 73841 - BEGINNING_OF_CENTRAL) {
		//	printf("CPU: prevdata.pos %1.9E %1.9E szPBC[iprev] %d \n",
		//		prevdata.pos.x, prevdata.pos.y, (int)szPBC[iprev]);
		//}
		
	//	if (iVertex == 11588) {
	//		printf("izTri[len-1] %d  prevdata.v_n.y %1.8E \n", izTri[tri_len-1], prevdata.v_n.y);
	//	}
		memcpy(&oppdata, &(pData[izTri[0]]), sizeof(plasma_data));
		if (szPBC[i] != 0) {
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				oppdata.v_n = Clockwise3 * oppdata.v_n;
				oppdata.vxy = Clockwise * oppdata.vxy;
				oppdata.pos = Clockwise * oppdata.pos; // let it be one object for now
			}
			else {
				oppdata.v_n = Anticlockwise3 * oppdata.v_n;
				oppdata.vxy = Anticlockwise * oppdata.vxy;
				oppdata.pos = Anticlockwise * oppdata.pos;
			}
		};
		
//if (iVertex == 11588) {
	//		printf("izTri[0] %d  oppdata.v_n.y %1.8E \n", izTri[0], oppdata.v_n.y);
//}
		endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

		overall_v_ours.x = 0.0;
		overall_v_ours.y = 0.0;
		if ((iMinor == CHOSEN) && (0)) {
			printf("CPU ourdata.Te %1.9E oppdata.Te %1.9E izTri[0] %d\n",
				ourdata.Te, oppdata.Te, izTri[0]);
		}
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			memcpy(&n_shards_n_, n_shards_n + iVertex, sizeof(ShardModel));
			memcpy(&n_shards_, n_shards + iVertex, sizeof(ShardModel));
			
		//	pVertex->CreateMajorPolygon(T, cp);
			// Think we can manage without it
			// Just go through & collect tri minor centroids
			// and make contiguous as we must also make their v contiguous

			overall_v_ours = p_overall_v[iMinor];
			memset(&ownrates, 0, sizeof(three_vec3));

			GetInterpolationCoefficients(beta, endpt0.x, endpt0.y,
				ourdata.pos, prevdata.pos, oppdata.pos);
			n_n0 = n_shards_n_.n_cent * beta[0] +
				   n_shards_n_.n[iprev] * beta[1] +
				   n_shards_n_.n[i] * beta[2];
			n0 = n_shards_.n_cent * beta[0] +
				 n_shards_.n[iprev] * beta[1] +
				 n_shards_.n[i] * beta[2];

			motion_edge0 = THIRD * (overall_v_ours + p_overall_v[izTri[iprev]] + p_overall_v[izTri[i]]);

			prevAz = prevdata.Az;
			oppAz = oppdata.Az;
			for (i = 0; i < tri_len; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.v_n = Clockwise3 * nextdata.v_n;
						nextdata.vxy = Clockwise * nextdata.vxy;
						nextdata.pos = Clockwise * nextdata.pos;
						// Yes, move position into minor data object
						// But maybe we'll move Az and Azdot out.
						// Hmmm -- still need position when we do Az only.
						// So that ought to be a separate flatpack ultimately.
					} else {
						nextdata.v_n = Anticlockwise3 * nextdata.v_n;
						nextdata.vxy = Anticlockwise * nextdata.vxy;
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};
				nextAz = nextdata.Az;
			
				// Infer T at endpts
				// I think we'd better assume we have T on each minor at this point.
				// ???***???***???***???***???***???***???***???***???***???

				Tn0 = THIRD * (prevdata.Tn + ourdata.Tn + oppdata.Tn);
				Tn1 = THIRD * (nextdata.Tn + ourdata.Tn + oppdata.Tn);
				Te0 = THIRD * (prevdata.Te + ourdata.Te + oppdata.Te);
				Te1 = THIRD * (nextdata.Te + ourdata.Te + oppdata.Te);
				Ti0 = THIRD * (prevdata.Ti + ourdata.Ti + oppdata.Ti);
				Ti1 = THIRD * (nextdata.Ti + ourdata.Ti + oppdata.Ti);
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

				GetInterpolationCoefficients(beta, endpt1.x, endpt1.y,
					ourdata.pos, nextdata.pos, oppdata.pos);
				n_n1 = n_shards_n_.n_cent*beta[0] +
					   n_shards_n_.n[inext] * beta[1] +
					   n_shards_n_.n[i]*beta[2];

				n1 = n_shards_.n_cent*beta[0] +
					n_shards_.n[inext] * beta[1] +
					n_shards_.n[i] * beta[2];
					
				// What to do with it, if we create nT at both ends of this edge?
				// Affect integral grad nT at this minor and the opposing one, surely.

				// On OUR side, these points go anticlockwise, so we take
				// integral grad nT += 0.5*(nT1+nT2)*((y2-y1) , (x1-x2))
				// But for THEIR side, it's the OPPOSITE direction.
				// (y2-y1,x1-x2) is also called edge_normal
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

		//		memcpy(&opprates, &(AdditionRateNv[iDestTri]), sizeof(three_vec3));

				//integral_grad_nT_n += 0.5*(n_n0*Tn0 + n_n1 * Tn1)*edge_normal;
				ownrates.neut -= Make3(0.5*(n_n0*Tn0 + n_n1 * Tn1)*over_m_n_*edge_normal, 0.0);
				ownrates.ion -= Make3(0.5*(n0*Ti0 + n1 * Ti1)*over_m_i_*edge_normal, 0.0);
				ownrates.elec -= Make3(0.5*(n0*Te0 + n1 * Te1)*over_m_e_*edge_normal, 0.0);
				
				// Addition rate Nv receives MINUS integral grad / m_s and the contrib to integral grad is MINUS because clockwise
		//		opprates.neut += Make3(0.5*(n_n0*Tn0 + n_n1 * Tn1)*over_m_n_*edge_normal, 0.0);
			//	opprates.ion += Make3(0.5*(n0*Ti0 + n1 * Ti1)*over_m_i_*edge_normal, 0.0);
		//		opprates.elec += Make3(0.5*(n0*Te0 + n1 * Te1)*over_m_e_*edge_normal, 0.0);

				// While averaging nT has its appeal, we end up working with
				// T_edge and n_edge because of minor look.
				// We see here how nT would have its advantages for sure

				// OK another way: get T as actual average at each minor edge endpoint.
				// So we want not to look up neigh data but instead just
				// look at tri data before and after. Maybe this saves a load.
				
				// ____________________________________________________________
				
				motion_edge1 = THIRD * (overall_v_ours + p_overall_v[izTri[inext]] + p_overall_v[izTri[i]]);
				v_n0 = THIRD * (ourdata.v_n + prevdata.v_n + oppdata.v_n);
				v_n1 = THIRD * (ourdata.v_n + oppdata.v_n + nextdata.v_n);
				vxy0 = THIRD * (ourdata.vxy + prevdata.vxy + oppdata.vxy);
				vxy1 = THIRD * (ourdata.vxy + oppdata.vxy + nextdata.vxy);
				vez0 = THIRD * (ourdata.vez + oppdata.vez + prevdata.vez);
				viz0 = THIRD * (ourdata.viz + oppdata.viz + prevdata.viz);
				vez1 = THIRD * (ourdata.vez + oppdata.vez + nextdata.vez);
				viz1 = THIRD * (ourdata.viz + oppdata.viz + nextdata.viz);
				
				relvnormal = 0.5*(v_n0.xypart() + v_n1.xypart() - motion_edge1 - motion_edge0).dot(edge_normal);
				ownrates.neut -= 0.5*relvnormal*
					(n_n0 * (v_n0 - ourdata.v_n)
						+ n_n1 * (v_n1 - ourdata.v_n));
				
				if ((iMinor == CHOSEN) && (0)) {
					printf("CPU %d neutadvectcontrib %1.12E  relvnormal %1.12E"
						"n_n0 %1.12E n_n1 %1.12E v0.y %1.12E v0-v.y %1.12E v1-v %1.12E  v_n.y %1.12E\n",
						CHOSEN,
						-0.5*relvnormal*
						(n_n0 * (v_n0.y - ourdata.v_n.y)
							+ n_n1 * (v_n1.y - ourdata.v_n.y)),
						relvnormal, 
						n_n0, n_n1, v_n0.y, v_n0.y - ourdata.v_n.y, v_n1.y - ourdata.v_n.y, ourdata.v_n.y);
				}

				relvnormal = 0.5*(vxy0 + vxy1 - motion_edge1 - motion_edge0).dot(edge_normal);
				ownrates.ion -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - ourdata.vxy, viz0 - ourdata.viz))
						+ n1 * (Make3(vxy1 - ourdata.vxy, viz1 - ourdata.viz)));
				ownrates.elec -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - ourdata.vxy, vez0 - ourdata.vez))
						+ n1 * (Make3(vxy1 - ourdata.vxy, vez1 - ourdata.vez)));

				// ______________________________________________________
				
				//// whether the v that is leaving is greater than our v ..
				//// Formula:
				//// dv/dt = (d(Nv)/dt - dN/dt v) / N
				//// We include the divide by N when we enter the accel routine.
				//
				//relvnormal = (vxy - motion_edge).dot(edge_normal);
				//ownrates.ion -= n * relvnormal* (Make3(vxy - ourdata.vxy, viz - ourdata.viz);
				////opprates.ion += same;

				//ownrates.elec -= n * relvnormal* (Make3(vxy - ourdata.vxy, vez - ourdata.vez);
			
				// ______________________________________________________-

				Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz); 
				Azdot_edge = SIXTH * (2.0*ourdata.Azdot + 2.0*oppdata.Azdot +
					prevdata.Azdot + nextdata.Azdot);
				Our_integral_grad_Azdot += Azdot_edge * edge_normal;

				Our_integral_grad_Az += Az_edge * edge_normal;				
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
			//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA

				f64 T_edge = SIXTH * (2.0*ourdata.Te + 2.0*oppdata.Te + prevdata.Te + nextdata.Te);
				Our_integral_grad_Te += T_edge * edge_normal;

				if (iMinor == CHOSEN)
					//printf("CPU %d GradTe contrib %1.14E %1.14E our Te %1.14E opp %1.14E next %1.14E prev %1.14E edgenormal %1.14E %1.14E\n", iMinor,
					//	SIXTH * (2.0*ourdata.Te + 2.0*oppdata.Te + prevdata.Te + nextdata.Te)*edge_normal.x,
					//	SIXTH * (2.0*ourdata.Te + 2.0*oppdata.Te + prevdata.Te + nextdata.Te)*edge_normal.y,
					//	ourdata.Te, oppdata.Te, nextdata.Te, prevdata.Te, edge_normal.x, edge_normal.y
					//);
					printf("CPU Az_edge %1.14E oppAz %1.14E endpt1 %1.14E %1.14E Integ_curl %1.14E %1.14E \n",
						Az_edge, oppAz, endpt1.x, endpt1.y, Our_integral_curl_Az.x, Our_integral_curl_Az.y
					);

				 // Integral of grad f is sum of edge values dot edge normal unnormalized
				 // Integral dx of grad x for area
				integ_grad_Az.x = 0.5*(
					(ourdata.Az + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourdata.Az)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourdata.Az + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourdata.Az)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);
			
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				
				// I daresay if we wanted to get really confused we could proceed
				// by expanding the dot product into integ_grad and inserting expressions
				// for edge_normal, rearrange to get some kind of havoc.
#ifdef LAPFILE
				if ((iMinor == index[0]) || 
					(iMinor == index[1]) ||
					(iMinor == index[2]) ||
					(iMinor == index[3]) )
				{
					fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;
					
					fprintf(fp, "area_quadrilateral %1.14E ourdata.pos %1.14E %1.14E "
						"prevdata.pos %1.14E %1.14E oppdata.pos %1.14E %1.14E "
						"nextdata.pos %1.14E %1.14E grad_Az %1.14E %1.14E "
						"edge_normal %1.14E %1.14E dot %1.14E Our_integral_Lap_Az %1.14E AreaMinor %1.14E lookat %d "
						"Aours %1.14E Aprev %1.14E Aopp %1.14E Anext %1.14E \n",
						area_quadrilateral, ourdata.pos.x, ourdata.pos.y,
						prevdata.pos.x, prevdata.pos.y, oppdata.pos.x, oppdata.pos.y,
						nextdata.pos.x, nextdata.pos.y, grad_Az.x, grad_Az.y,
						edge_normal.x, edge_normal.y, grad_Az.dot(edge_normal),
						Our_integral_Lap_Az, AreaMinor, izTri[i],
						ourdata.Az,prevdata.Az, oppdata.Az, nextdata.Az);
				};
#endif

			//	if (iVertex == 23323) {
				//	printf("AreaMinor %1.8E \n",
			//			AreaMinor);
			//	}
				// ===================================================================
				// ===================================================================
				
/*
				// scatter is bad on GPU!

				// Now look at the minor edge between iDestTri and the next tri along

				iTriNext = izTri[inext];
				edgenormal.x = THIRD * (ourpos.y - neighpos1.y);
				edgenormal.y = THIRD * (neighpos1.x - ourpos.x);
				// Let's think carefully about this.
				// we do know which sign this should be.

				contrib = 0.5*n_n_1 *0.5*(neighdata1.Tn + ourdata.Tn)*over_m_n_*edge_normal;
				// looking outwards, we take MINUS ...
				opprates.neut -= Make3(contrib, 0.0);
				AdditionRateNv[iTriNext].neut += Make3(contrib, 0.0);

				contrib = 0.5*n1 *0.5*(neighdata1.Te + ourdata.Te)*over_m_e_*edge_normal;
				opprates.elec -= Make3(contrib, 0.0);
				AdditionRateNv[iTriNext].elec += Make3(contrib, 0.0);

				contrib = 0.5*n1 *0.5*(neighdata1.Ti + ourdata.Ti)*over_m_i_*edge_normal;
				opprates.ion -= Make3(contrib, 0.0);
				AdditionRateNv[iTriNext].ion += Make3(contrib, 0.0);

				// This accounts for "our" half of the edge between iDestTri and iTriNext



				memcpy(&(AdditionRateNv[iDestTri]), &opprates, sizeof(three_vec3));



				Az_edge = SIXTH * (2.0*oppAz + 2.0*nextAz + ourdata.Az + neighdata1.Az);
				integral_grad_Az[iDestTri] += 0.5*Az_edge * edge_normal;
				integral_grad_Az[iTriNext] -= 0.5*Az_edge * edge_normal;

				// INTEGRATED:
				pData[iDestTri].B += Az_edge * THIRD*(ourpos - neighpos1);
				pData[iTriNext].B -= Az_edge * THIRD*(ourpos - neighpos1);// INTEGRATED
				// Maybe more proper to have grad,curl,Lap A in object we can write to in 1 go;
				// We can divide by area on the fly. 

				// Now we know to get grad we have to take the integral of grad divided by the integral of 1.
				integ_grad_Az.x = 0.5*(
					(nextAz + neighdata1.Az)*(cp.coord[inext].y - neighpos1.y)
					+ (ourdata.Az + nextAz)*(ourpos.y - cp.coord[inext].y)
					+ (oppAz + ourdata.Az)*(cp.coord[i].y - ourpos.y)
					+ (neighdata1.Az + oppAz)*(neighpos1.y - cp.coord[i].y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(nextAz + neighdata1.Az)*(cp.coord[inext].x - neighpos1.x)
					+ (ourdata.Az + nextAz)*(ourpos.x - cp.coord[inext].x)
					+ (oppAz + ourdata.Az)*(cp.coord[i].x - ourpos.x)
					+ (neighdata1.Az + oppAz)*(neighpos1.x - cp.coord[i].x)
					);
				area_quadrilateral = 0.5*(
					(cp.coord[inext].x + neighpos1.x)*(cp.coord[inext].y - neighpos1.y)
					+ (ourpos.x + cp.coord[inext].x)*(ourpos.y - cp.coord[inext].y)
					+ (cp.coord[i].x + ourpos.x)*(cp.coord[i].y - ourpos.y)
					+ (neighpos1.x + cp.coord[i].x)*(neighpos1.y - cp.coord[i].y)
					);
				grad_Az = integ_grad_Az / area_quadrilateral;

				integral_Lap_Az[iDestTri] += 0.5*grad_Az.dot(edge_normal);
				integral_Lap_Az[iTriNext] -= 0.5*grad_Az.dot(edge_normal);

				// You can imagine instead having array of neighbours for each minor.
				*/
				++iprev;
				motion_edge0 = motion_edge1;
				endpt0 = endpt1;
				n_n0 = n_n1;
				n0 = n1;
				memcpy(&prevdata, &oppdata, sizeof(plasma_data)); // assuming local memcpy is fast but pointer arith could be faster
				memcpy(&oppdata, &nextdata, sizeof(plasma_data));
				prevAz = oppAz;
				oppAz = nextAz;
			};
			GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			ROCAzduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Az / AreaMinor);
			ROCAzdotduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Azdot / AreaMinor);

			LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
			GradTeArray[iMinor] = Our_integral_grad_Te / AreaMinor;
			if (iMinor == CHOSEN) printf("Our_integral_grad_Te %1.14E AreaMinor %1.14E\n\n",
				Our_integral_grad_Te, AreaMinor);

			pData[iMinor].B = Make3(Our_integral_curl_Az/AreaMinor,BZ_CONSTANT);
			AreaMinorArray[iMinor] = AreaMinor;
			memcpy(&(AdditionRateNv[iMinor]), &ownrates, sizeof(three_vec3));
			
			pData[iMinor].temp.x = Our_integral_Lap_Az / AreaMinor;
#ifdef LAPFILE
			if ((iMinor == index[0]) ||
				(iMinor == index[1]) ||
				(iMinor == index[2]) ||
				(iMinor == index[3]))
			{
				fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;

				fprintf(fp, "\nWritten %1.14E\n\n", Our_integral_Lap_Az/ AreaMinor);
			};
#endif
		} else {

			f64_vec2 projendpt0, projendpt1;

			prevAz = prevdata.Az;
			oppAz = oppdata.Az;

			int istart = 0;
			int iend = tri_len;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				istart = 0;
				iend = tri_len - 2;
				// think about this.
				// if it went       3      4
				//                    2 1 0
				// we are going to do edges facing 0,1,2 as 5 = tri_len

				// Given that 0 is the first non-frill, we don't need to do the following?
				//prevAz = (pData + izTri[0])->Az;
				//oppAz = (pData + izTri[1])->Az;
				//endpt0 = THIRD * ((pData + izTri[0])->pos + (pData + izTri[1])->pos + ourdata.pos);

				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				} else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = istart; i < iend; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;
				
				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					} else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};
				nextAz = nextdata.Az;
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);
				// (y2-y1,x1-x2) is also called edge_normal
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// Integral dx of grad x for area
				integ_grad_Az.x = 0.5*(
					(ourdata.Az + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourdata.Az)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourdata.Az + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourdata.Az)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				// This is no good - we didn't compute B for vertices within insulator!
				// Just set to 0 for OUTERMOST and in ins -- we never use the latter and the former is not well-defined due to 2D cross-section.
				
#ifdef LAPFILE
				if ((iMinor == index[0]) ||
					(iMinor == index[1]) ||
					(iMinor == index[2]) ||
					(iMinor == index[3]))
				{
					fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;


					fprintf(fp, "lookat %d area_quadrilateral %1.14E ourdata.pos %1.14E %1.14E "
						"prevdata.pos %1.14E %1.14E oppdata.pos %1.14E %1.14E "
						"nextdata.pos %1.14E %1.14E grad_Az %1.14E %1.14E "
						"edge_normal %1.14E %1.14E dot %1.14E Our_integral_Lap_Az %1.14E AreaMinor %1.14E "
						"Aours %1.14E Aprev %1.14E Aopp %1.14E Anext %1.14E \n",
						izTri[i], area_quadrilateral, ourdata.pos.x, ourdata.pos.y,
						prevdata.pos.x, prevdata.pos.y, oppdata.pos.x, oppdata.pos.y,
						nextdata.pos.x, nextdata.pos.y, grad_Az.x, grad_Az.y,
						edge_normal.x, edge_normal.y, grad_Az.dot(edge_normal),
						Our_integral_Lap_Az, AreaMinor,
						ourdata.Az, prevAz, oppAz, nextAz);
				};
#endif

				// I daresay if we wanted to get really confused we could proceed
				// by expanding the dot product into integ_grad and inserting expressions
				// for edge_normal, rearrange to get some kind of havoc.
				// ===================================================================

				++iprev;
				endpt0 = endpt1;
				prevAz = oppAz;
				oppAz = nextAz;
				prevdata.pos = oppdata.pos;
				oppdata.pos = nextdata.pos;
			};

			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				// Now add on the final sides to give area:

				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.

				if (pVertex->flags == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius); 
				} else {
					endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;

				// line between out-projected points
			};

			LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
			ROCAzduetoAdvection[iMinor] = 0.0;
			ROCAzdotduetoAdvection[iMinor] = 0.0;
			GradTeArray[iMinor] = Vector2(0.0, 0.0);

			// Within insulator we don't care about grad A or curl A, only Lap A
			AreaMinorArray[iMinor] = AreaMinor;

			pData[iMinor].temp.x = Our_integral_Lap_Az / AreaMinor;

#ifdef LAPFILE
			if ((iMinor == index[0]) ||
				(iMinor == index[1]) ||
				(iMinor == index[2]) ||
				(iMinor == index[3]))
			{
				fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;

				fprintf(fp, "\nWritten %1.14E\n\n", Our_integral_Lap_Az / AreaMinor);
			};
#endif

			// Not enough: We run AntiAdvectAz for all minors so we want GradAz=0 to be set outside domain
			memset(GradAz + iMinor, 0, sizeof(f64_vec2));
			ROCAzdotduetoAdvection[iMinor] = 0.0;
			ROCAzduetoAdvection[iMinor] = 0.0;

			pData[iMinor].B = Vector3(0.0, 0.0, BZ_CONSTANT);

		}; // was it DOMAIN_VERTEX

		++pVertex;
		++iMinor;
	};
	
	pTri = T;
	for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
	{

		if ((iMinor == index[0]) ||
			(iMinor == index[1]) ||
			(iMinor == index[2]) ||
			(iMinor == index[3]))
		{
			iMinor = iMinor;
		};

		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)) {
			// do nothing

			Our_integral_curl_Az.x = 0.0;
			Our_integral_curl_Az.y = 0.0;
			Our_integral_grad_Az.x = 0.0;
			Our_integral_grad_Az.y = 0.0;
			Our_integral_grad_Azdot.x = 0.0;
			Our_integral_grad_Azdot.y = 0.0;
			Our_integral_grad_Te.x = 0.0;
			Our_integral_grad_Te.y = 0.0;
			Our_integral_Lap_Az = 0.0;
			AreaMinor = 1.0e-12;

			GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			ROCAzduetoAdvection[iMinor] = 0.0;
			ROCAzdotduetoAdvection[iMinor] = 0.0;
			
			LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
			pData[iMinor].B = Make3(Our_integral_curl_Az /AreaMinor, BZ_CONSTANT); 
			GradTeArray[iMinor] = Our_integral_grad_Te;
			AreaMinorArray[iMinor] = AreaMinor;
			memset(&(AdditionRateNv[iMinor]), 0, sizeof(three_vec3));

			pData[iMinor].temp.x = Our_integral_Lap_Az / AreaMinor;
		} else {
			memcpy(izNeighMinor, TriMinorNeighLists[iMinor], sizeof(long) * 6);
			memcpy(szPBC, TriMinorPBCLists[iMinor], sizeof(char) * 6);

			Our_integral_curl_Az.x = 0.0;
			Our_integral_curl_Az.y = 0.0;
			Our_integral_grad_Azdot.x = 0.0;
			Our_integral_grad_Azdot.y = 0.0;
			Our_integral_grad_Az.x = 0.0;
			Our_integral_grad_Az.y = 0.0;
			Our_integral_grad_Te.x = 0.0;
			Our_integral_grad_Te.y = 0.0;
			Our_integral_Lap_Az = 0.0;
			AreaMinor = 0.0;

			iprev = 5; i = 0;
			memcpy(&ourdata, pData + iMinor, sizeof(plasma_data));
			memcpy(&prevdata, pData + izNeighMinor[iprev], sizeof(plasma_data));
			if (szPBC[iprev] != 0) {
				if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
					prevdata.v_n = Clockwise3 * prevdata.v_n;
					prevdata.vxy = Clockwise * prevdata.vxy;
					prevdata.pos = Clockwise * prevdata.pos;
				}
				else {
					prevdata.v_n = Anticlockwise3 * prevdata.v_n;
					prevdata.vxy = Anticlockwise * prevdata.vxy;
					prevdata.pos = Anticlockwise * prevdata.pos;
				}
			};
			memcpy(&oppdata, pData + izNeighMinor[0], sizeof(plasma_data));
			if (szPBC[i] != 0) {
				if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
					oppdata.v_n = Clockwise3 * oppdata.v_n;
					oppdata.vxy = Clockwise * oppdata.vxy;
					oppdata.pos = Clockwise * oppdata.pos;
				}
				else {
					oppdata.v_n = Anticlockwise3 * oppdata.v_n;
					oppdata.vxy = Anticlockwise * oppdata.vxy;
					oppdata.pos = Anticlockwise * oppdata.pos;
				}
			};
			prevAz = prevdata.Az;
			oppAz = oppdata.Az;
			overall_v_ours.x = 0.0;
			overall_v_ours.y = 0.0;
			if ((pTri->u8domain_flag == DOMAIN_TRIANGLE) || (pTri->u8domain_flag == CROSSING_INS))
			{
				// MAKING THEM THE SAME ...
				// THIS MEANS WE NEED TO EnsUrE THAT WITHIN InsuLATOR, n=v=T=v_overall=0
				// especially n in the array.

				// Lower down, reset accel_TP radial component to 0 ...

				memcpy(n_array, Tri_n_lists[iMinor], sizeof(f64) * 6);
				memcpy(n_n_array, Tri_n_n_lists[iMinor], sizeof(f64) * 6);

				overall_v_ours = p_overall_v[iMinor];
				memset(&ownrates, 0, sizeof(three_vec3));
				
				motion_edge0 = THIRD * (overall_v_ours + p_overall_v[izNeighMinor[iprev]] + p_overall_v[izNeighMinor[i]]);
				
				for (i = 0; i < 6; i++)
				{
					inext = i + 1; if (inext > 5) inext = 0;
				
					// assume n[0] is clockwise point of corner 0
					// neigh array goes [corner 0][neigh 2][corner1][neigh0][corner2][neigh1]

					memcpy(&nextdata, pData + izNeighMinor[inext], sizeof(plasma_data));
					// Make contiguous:
					if (szPBC[inext] != 0) {
						if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
							nextdata.v_n = Clockwise3 * nextdata.v_n;
							nextdata.vxy = Clockwise * nextdata.vxy;
							nextdata.pos = Clockwise * nextdata.pos;
							// Yes, move position into minor data object
							// But maybe we'll move Az and Azdot out.
							// Hmmm -- still need position when we do Az only.
							// So that ought to be a separate flatpack ultimately.
						} else {
							nextdata.v_n = Anticlockwise3 * nextdata.v_n;
							nextdata.vxy = Anticlockwise * nextdata.vxy;
							nextdata.pos = Anticlockwise * nextdata.pos;
						}
					};
				
					nextAz = nextdata.Az;

					// n_array[i] and n_array[inext] are the two n values
					Tn0 = THIRD * (prevdata.Tn + ourdata.Tn + oppdata.Tn);
					Tn1 = THIRD * (nextdata.Tn + ourdata.Tn + oppdata.Tn);
					Te0 = THIRD* (prevdata.Te + ourdata.Te + oppdata.Te);
					Te1 = THIRD * (nextdata.Te + ourdata.Te + oppdata.Te);
					Ti0 = THIRD * (prevdata.Ti + ourdata.Ti + oppdata.Ti);
					Ti1 = THIRD * (nextdata.Ti + ourdata.Ti + oppdata.Ti);

					// New definition of endpoint of minor edge:
				
					endpt0 = THIRD * (prevdata.pos + ourdata.pos + oppdata.pos);
					endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);
				
					edge_normal.x = endpt1.y - endpt0.y;
					edge_normal.y = endpt0.x - endpt1.x;

					ownrates.neut -= Make3(0.5*(n_n_array[i]*Tn0 + n_n_array[inext] * Tn1)*over_m_n_*edge_normal, 0.0);
					ownrates.ion -= Make3(0.5*(n_array[i]*Ti0 + n_array[inext] * Ti1)*over_m_i_*edge_normal, 0.0);
					ownrates.elec -= Make3(0.5*(n_array[i]*Te0 + n_array[inext] * Te1)*over_m_e_*edge_normal, 0.0);
					

					if ((iMinor == 73061) && (0)) printf("CPU 73061 neutralpressure contrib %1.14E %1.14E \n"
						"n0 n1 %1.14E %1.14E T0 T1 %1.14E %1.14E \n",
						-0.5*(n_n_array[i] * Tn0 + n_n_array[inext] * Tn1)*over_m_n_*edge_normal.x,
						-0.5*(n_n_array[i] * Tn0 + n_n_array[inext] * Tn1)*over_m_n_*edge_normal.y,
						n_n_array[i], n_n_array[inext], Tn0, Tn1);


					/*
					if ((iMinor == CHOSEN) && (0)) {
						printf("CPU %d : ownrates.ion.x %1.10E contrib %1.11E n_array[i] %1.10E n_array[inext] %1.10E Ti0 %1.11E Ti1 %1.11E edge_normal.x %1.10E \n",
							CHOSEN, ownrates.ion.x,
							-0.5*(n_array[i] * Ti0 + n_array[inext] * Ti1)*over_m_i_*edge_normal.x,
							n_array[i], n_array[inext], Ti0, Ti1, edge_normal.x);
					}*/

				//	if ((iMinor == CHOSEN) && (0)) {
				//		printf("CPU %d ownrates.neut %1.11E %1.11E contrib %1.11E %1.11E n0 %1.11E n1 %1.11E T0 %1.11E T1 %1.11E edge_normal %1.11E %1.11E \n",
				//			CHOSEN, ownrates.neut.x, ownrates.neut.y,
				//			-0.5*(n_n_array[i] * Tn0 + n_n_array[inext] * Tn1)*over_m_n_*edge_normal.x,
				//			-0.5*(n_n_array[i] * Tn0 + n_n_array[inext] * Tn1)*over_m_n_*edge_normal.y, 
				//			n_n_array[i], n_n_array[inext], Tn0, Tn1, edge_normal.x, edge_normal.y);
				//	}

		//			if (iMinor == 23323) {
		//				printf("aTP 23323 ownrates.neut.y %1.8E n_n_array[i] %1.8E Tn0 %1.8E edge_normal.y %1.8E \n",
		//					ownrates.neut.y, n_n_array[i], Tn0, edge_normal.y);
		//			}
					//motion_edge = 0.5*(overall_v_ours + p_overall_v[izNeighMinor[i]]);
					// VERIFY: NOW THAT WE SAID EACH ENDPOINT IS AVERAGE FROM 3, IS THIS SO?
					// NO. Instead:

					// ____________________________________________________________

					motion_edge1 = THIRD * (overall_v_ours + p_overall_v[izNeighMinor[inext]] + p_overall_v[izNeighMinor[i]]);
					v_n0 = THIRD * (ourdata.v_n + prevdata.v_n + oppdata.v_n);
					v_n1 = THIRD * (ourdata.v_n + oppdata.v_n + nextdata.v_n);
					vxy0 = THIRD * (ourdata.vxy + prevdata.vxy + oppdata.vxy);
					vxy1 = THIRD * (ourdata.vxy + oppdata.vxy + nextdata.vxy);
					vez0 = THIRD * (ourdata.vez + oppdata.vez + prevdata.vez);
					viz0 = THIRD * (ourdata.viz + oppdata.viz + prevdata.viz);
					vez1 = THIRD * (ourdata.vez + oppdata.vez + nextdata.vez);
					viz1 = THIRD * (ourdata.viz + oppdata.viz + nextdata.viz);
					
					// consistent with how we did pressure, let's take n0v0 + n1v1
					// in which case we might as well go the whole hog and keep overall v at each end also
					// Maybe prefer to put .... nv_edge is 0.5 nvrel + 0.5 nvrel .. 
					// or, v rel is given and we average nv+nv 
					// Not sure I ptic like how I've done it here. Don't see that it matters.
					relvnormal = 0.5*(v_n0.xypart()+v_n1.xypart()-motion_edge1 - motion_edge0).dot(edge_normal);
					ownrates.neut -= 0.5*relvnormal*
						(n_n_array[i] * (v_n0 - ourdata.v_n)
							+ n_n_array[inext] * (v_n1 - ourdata.v_n));


					if ((iMinor == CHOSEN) && (0)) {
						printf("CPU %d neutraladvectcontrib.y %1.12E relvnormal %1.12E n0 n1 %1.12E %1.12E v0 v1 %1.12E %1.12E our_v %1.12E \n",
							CHOSEN,
							0.5*relvnormal* (n_n_array[i] *(v_n0.y - ourdata.v_n.y) + n_n_array[inext] * (v_n1.y - ourdata.v_n.y)),
							relvnormal,
							n_n_array[i], n_n_array[inext], v_n0.y, v_n1.y, ourdata.v_n.y
						);
					};

					relvnormal = 0.5*(vxy0 + vxy1 - motion_edge1 - motion_edge0).dot(edge_normal);
					ownrates.ion -= 0.5*relvnormal*
						(n_array[i] * (Make3(vxy0,viz0) - Make3(ourdata.vxy,ourdata.viz))
							+ n_array[inext] * (Make3(vxy1,viz1) - Make3(ourdata.vxy, ourdata.viz)));
					ownrates.elec -= 0.5*relvnormal*
						(n_array[i] * (Make3(vxy0,vez0) - Make3(ourdata.vxy, ourdata.vez))
							+ n_array[inext] * (Make3(vxy1,vez1) - Make3(ourdata.vxy, ourdata.vez)));

					//if ((iMinor == CHOSEN) && (0)) {
					//	printf("CPUadvective %d ownrates.ion.x %1.10E contrib %1.11E relvnormal %1.10E n_array[i] %1.12E n_array[inext] %1.12E vxy0.x %1.12E vxy1.x %1.12E ourvx %1.12E \n",
					//		CHOSEN,
					//		ownrates.ion.x, 
					//		-0.5*relvnormal*
					//		(n_array[i] * (vxy0.x - ourdata.vxy.x) + n_array[inext] * (vxy1.x - ourdata.vxy.x)),
					//		relvnormal,
					//		n_array[i], n_array[inext], vxy0.x, vxy1.x, ourdata.vxy.x);
					//};

					// ______________________________________________________-
					
					Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
					Azdot_edge = SIXTH * (2.0*ourdata.Azdot + 2.0*oppdata.Azdot +
						prevdata.Azdot + nextdata.Azdot);
					Our_integral_grad_Azdot += Azdot_edge * edge_normal;
					Our_integral_grad_Az += Az_edge * edge_normal;
					Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
					
					f64 T_edge = SIXTH * (2.0*ourdata.Te + 2.0*oppdata.Te + prevdata.Te + nextdata.Te);
					Our_integral_grad_Te += T_edge * edge_normal;

//					if ((iMinor == CHOSEN) && (0)) {
//						printf("CPU %d : ourintegralgradTe %1.9E %1.9E contrib %1.9E %1.9E T_edge %1.9E edgenormal %1.9E %1.9E\n",
//							CHOSEN,
//							Our_integral_grad_Te.x, Our_integral_grad_Te.y,
//							T_edge * edge_normal.x, T_edge * edge_normal.y,
//							T_edge, edge_normal.x, edge_normal.y);
//						printf("Ourdata.Te %1.9E oppdata.Te %1.9E prevdata.Te %1.9E nextdata.Te %1.9E\n",
//							ourdata.Te, oppdata.Te, prevdata.Te, nextdata.Te);
//						
//					}

					// Integral of grad f is sum of edge values dot edge normal unnormalized
					// Integral dx of grad x for area
					integ_grad_Az.x = 0.5*(
						(ourdata.Az + nextAz)*(ourdata.pos.y - nextdata.pos.y)
						+ (prevAz + ourdata.Az)*(prevdata.pos.y - ourdata.pos.y)
						+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
						+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
						);
					integ_grad_Az.y = -0.5*( // notice minus
						(ourdata.Az + nextAz)*(ourdata.pos.x - nextdata.pos.x)
						+ (prevAz + ourdata.Az)*(prevdata.pos.x - ourdata.pos.x)
						+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
						+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
						);
					area_quadrilateral = 0.5*(
						(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
						+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
						+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
						+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
						);
					f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
					Our_integral_Lap_Az += grad_Az.dot(edge_normal);

					AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;				
//
//					if (iMinor == CHOSEN) printf("CPU AreaMinor %d : %1.14E from += %1.14E : endpt0.x %1.14E endpt1.x %1.14E edge_normal.x %1.14E\n"
//						"endpt1.y endpt0.y, %1.14E %1.14E \n",
//						iMinor, AreaMinor, (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x,
//						endpt0.x, endpt1.x, edge_normal.x,
//						endpt1.y, endpt0.y);
//
#ifdef LAPFILE
					if ((iMinor == index[0]) ||
						(iMinor == index[1]) ||
						(iMinor == index[2]) ||
						(iMinor == index[3]))
					{
						fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;

					
						fprintf(fp, "izNeighMinor %d area_quadrilateral %1.14E ourdata.pos %1.14E %1.14E "
							"prevdata.pos %1.14E %1.14E oppdata.pos %1.14E %1.14E "
							"nextdata.pos %1.14E %1.14E grad_Az %1.14E %1.14E "
							"edge_normal %1.14E %1.14E dot %1.14E Our_integral_Lap_Az %1.14E AreaMinor %1.14E  "
							"Aours %1.14E Aprev %1.14E Aopp %1.14E Anext %1.14E \n",
							izNeighMinor[i],area_quadrilateral, ourdata.pos.x, ourdata.pos.y,
							prevdata.pos.x, prevdata.pos.y, oppdata.pos.x, oppdata.pos.y,
							nextdata.pos.x, nextdata.pos.y, grad_Az.x, grad_Az.y,
							edge_normal.x, edge_normal.y, grad_Az.dot(edge_normal),
							Our_integral_Lap_Az, AreaMinor, 
							ourdata.Az, prevdata.Az, oppdata.Az, nextdata.Az);
					};
#endif

					endpt0 = endpt1;
					prevAz = oppAz;
					oppAz = nextAz;
					memcpy(&prevdata, &oppdata, sizeof(plasma_data));
					memcpy(&oppdata, &nextdata, sizeof(plasma_data));
					motion_edge0 = motion_edge1;
					// There is an even quicker way which is to rotate pointers. No memcpy needed.
					// Is there the concept of a pointer to local data? Not experienced to know what it does.
				};

				if (pTri->u8domain_flag == CROSSING_INS) {
					// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
					f64_vec2 rhat = ourdata.pos / ourdata.pos.modulus();
					ownrates.neut -= Make3(
						(ownrates.neut.dotxy(ourdata.pos)/(ourdata.pos.x*ourdata.pos.x+ourdata.pos.y*ourdata.pos.y))*ourdata.pos,0.0);
					ownrates.ion -= Make3(
						(ownrates.ion.dotxy(ourdata.pos) / (ourdata.pos.x*ourdata.pos.x + ourdata.pos.y*ourdata.pos.y))*ourdata.pos, 0.0);
					ownrates.elec -= Make3(
						(ownrates.elec.dotxy(ourdata.pos) / (ourdata.pos.x*ourdata.pos.x + ourdata.pos.y*ourdata.pos.y))*ourdata.pos, 0.0);

					// I think we do need to make v_r = 0. It's common sense that it IS 0
					// since we site our v_r estimate on the insulator. Since it is sited there,
					// it is used for traffic into the insulator by n,nT unless we pick out
					// insulator-abutting cells on purpose.

					// However, we then should make an energy correction -- at least if
					// momentum is coming into this minor cell and being destroyed.
				};

				GradAz[iMinor] = Our_integral_grad_Az/AreaMinor;
				LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
				ROCAzduetoAdvection[iMinor] = (Our_integral_grad_Az / AreaMinor).dot(overall_v_ours);
				ROCAzdotduetoAdvection[iMinor] = (Our_integral_grad_Azdot / AreaMinor).dot(overall_v_ours);
				pData[iMinor].B = Make3(Our_integral_curl_Az/AreaMinor,BZ_CONSTANT); 
				GradTeArray[iMinor] = Our_integral_grad_Te / AreaMinor;
				AreaMinorArray[iMinor] = AreaMinor;
				memcpy(&(AdditionRateNv[iMinor]), &ownrates, sizeof(three_vec3));

				pData[iMinor].temp.x = Our_integral_Lap_Az / AreaMinor;

				/*
				if ((iMinor == 29109) || (iMinor == 29137))
				{
					fp = fp4; if (iMinor == 29109) fp = fp3;
					fprintf(fp, "\nWritten %1.14E\n\n", Our_integral_Lap_Az / AreaMinor);
					fclose(fp);
				};*/

			}
			else {

				// Not frill, crossing or domain

				for (i = 0; i < 6; i++)
				{
					inext = i + 1; if (inext > 5) inext = 0;

					// assume n[0] is clockwise point of corner 0
					// neigh array goes [corner 0][neigh 2][corner1][neigh0][corner2][neigh1]

					memcpy(&nextdata, pData + izNeighMinor[inext], sizeof(plasma_data));
					// Make contiguous:
					if (szPBC[inext] != 0) {
						if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
							nextdata.pos = Clockwise * nextdata.pos;
						}
						else {
							nextdata.pos = Anticlockwise * nextdata.pos;
						}
					};

					nextAz = nextdata.Az;

					// New definition of endpoint of minor edge:

					endpt0 = THIRD * (prevdata.pos + ourdata.pos + oppdata.pos);
					endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

					edge_normal.x = endpt1.y - endpt0.y;
					edge_normal.y = endpt0.x - endpt1.x;

					// ______________________________________________________-

					Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
					Our_integral_grad_Az += Az_edge * edge_normal;

					Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

						// Integral of grad f is sum of edge values dot edge normal unnormalized
						// Integral dx of grad x for area
					integ_grad_Az.x = 0.5*(
						(ourdata.Az + nextAz)*(ourdata.pos.y - nextdata.pos.y)
						+ (prevAz + ourdata.Az)*(prevdata.pos.y - ourdata.pos.y)
						+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
						+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
						);
					integ_grad_Az.y = -0.5*( // notice minus
						(ourdata.Az + nextAz)*(ourdata.pos.x - nextdata.pos.x)
						+ (prevAz + ourdata.Az)*(prevdata.pos.x - ourdata.pos.x)
						+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
						+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
						);
					area_quadrilateral = 0.5*(
						(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
						+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
						+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
						+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
						);
					f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
					Our_integral_Lap_Az += grad_Az.dot(edge_normal);

					AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
					
#ifdef LAPFILE
					if ((iMinor == index[0]) ||
						(iMinor == index[1]) ||
						(iMinor == index[2]) ||
						(iMinor == index[3]))
					{
						fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;


						fprintf(fp, "izNeighMinor %d area_quadrilateral %1.14E ourdata.pos %1.14E %1.14E "
							"prevdata.pos %1.14E %1.14E oppdata.pos %1.14E %1.14E "
							"nextdata.pos %1.14E %1.14E grad_Az %1.14E %1.14E "
							"edge_normal %1.14E %1.14E dot %1.14E Our_integral_Lap_Az %1.14E AreaMinor %1.14E "
							"Aours %1.14E Aprev %1.14E Aopp %1.14E Anext %1.14E \n",
							izNeighMinor[i], area_quadrilateral, ourdata.pos.x, ourdata.pos.y,
							prevdata.pos.x, prevdata.pos.y, oppdata.pos.x, oppdata.pos.y,
							nextdata.pos.x, nextdata.pos.y, grad_Az.x, grad_Az.y,
							edge_normal.x, edge_normal.y, grad_Az.dot(edge_normal),
							Our_integral_Lap_Az, AreaMinor, 
							ourdata.Az, prevdata.Az, oppdata.Az, nextdata.Az);
					};
#endif

					endpt0 = endpt1;
					prevAz = oppAz;
					oppAz = nextAz;
					memcpy(&prevdata, &oppdata, sizeof(plasma_data));
					memcpy(&oppdata, &nextdata, sizeof(plasma_data));
					motion_edge0 = motion_edge1;
					// There is an even quicker way which is to rotate pointers. No memcpy needed.
					// Is there the concept of a pointer to local data? Not experienced to know what it does.
				};
				GradAz[iMinor] = Our_integral_grad_Az/AreaMinor;
				ROCAzduetoAdvection[iMinor] = 0.0;
				ROCAzdotduetoAdvection[iMinor] = 0.0;
				LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
				pData[iMinor].B = Make3(Our_integral_curl_Az/AreaMinor, BZ_CONSTANT); 
				AreaMinorArray[iMinor] = AreaMinor;

				pData[iMinor].temp.x = Our_integral_Lap_Az / AreaMinor;
#ifdef LAPFILE
				if ((iMinor == index[0]) ||
					(iMinor == index[1]) ||
					(iMinor == index[2]) ||
					(iMinor == index[3]))
				{
					fp = fp1; if (iMinor == index[1]) fp = fp2; if (iMinor == index[2]) fp = fp3; if (iMinor == index[3]) fp = fp4;



					fprintf(fp, "Written %1.14E \n\n", Our_integral_Lap_Az / AreaMinor);
				};
#endif

			}; // end if: not a frill

				// how to handle each variable for each minor type:

				// Az : at innermost, let Az-dot be 0 for all time and likewise at the outer bdry.
				// we need to include REVERSE JZ when we do Az advance : 
				// We should be assuming Az DIES AWAY TOWARDS 0 NOT that it iS ZERO
				// solution to Laplace ??? radial decline probably 
				// Logic: when we look within domain for A outisde (or on outermost) we
				// assume there that it's radially declined from the Az domain value.
				// This is achieved by advancing the rest and then setting the boundary.
				// That ought to work.

				// grad Az is thus available using outermost Az values which are set that way.
				// curl, grad and Lap Az are not calculated for outermost vertices.
			
				// Think: We should estimate Lap Az near inner/outer boundary by ignoring the edges
				// that look inward / outward : the assumption is it's constant there so contrib to Lap = 0
				// WILL ULTIMATELY NEED TO CHECK THIS
			

				// There is not a special treatment of triangles next to outermost, as far as this goes.

				// PRESSURE:
				// As normal in triangles by outermost, outermost vertex no calc.
				// No calc within insulator; crossing_tri has to think carefully:
			// We can pretty much treat pressure as zero in a crossing tri without
			// any real consequence, but might rather let it be just azimuthal.
			// .. Only makes sense to collect nT on edges facing domain vertex or domain tri
			// .. then set radial component to zero.

			// Momentum flux:
			// We don't need anything to flow into outermost so we have to
			// stop it flowing OUT to outermost.
			// Alternatively we can let it go live there. The vertices can even
			// swarm outwards. There's no reason why not?
			// There's no benefit in that.
			// Run for outermost? No can do, frills have undefined values.
			// So we'd prefer to detect and cancel flux looking into outermost vertex.
			
			// What about at the insulator?
			// We are guaranteed vertex minors do not touch the insulator.
			// Crossing tris: 
			// Ignore edges that are not facing domain;
			// v_r is necessarily 0 -- how to do this best?
			// inward v will flow into it. 
			// We can just let it have v_r without any consequences? No -- let it be 0.
			// 'v here doesn't do anything so it's not that importnat if we let v 
			// flow here. It can diffuse back again... ?'

			// How, for creating shard model, do we decide n_desired?
			// For the time being just set it equal to the average n from the corners
			// We do want a better way eventually, allowing it to fit a quadratic to
			// the local data and n' = 0.

			// """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		};
		

		++pTri;
	};
#ifdef LAPFILE
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
#endif

}

void TriMesh::GetLap(real Az_array[], real LapAz_array[])
{

	FILE * fp;
	Triangle * pTri;
	long iVertex, iMinor;
	int i, iprev, inext;
	long neigh_len, tri_len;

	long izNeigh[MAXNEIGH], izTri[MAXNEIGH];
	f64_vec2 edge_normal;
	f64_vec2 endpt0, endpt1;
	f64_vec3 outward_flux, v_n;
	plasma_data ourdata, prevdata, oppdata, nextdata; // neighdata0, neighdata1;
	Vertex * pVertex = X;
	f64_vec2 Our_integral_curl_Az, Our_integral_grad_Te, Our_integral_grad_Az, motion_edge0, motion_edge1;
	f64 Our_integral_Lap_Az, prevAz, oppAz, nextAz, area_quadrilateral;
	char szPBC[MAXNEIGH];
	f64 AreaMinor;
	f64_vec2 integ_grad_Az;
	//// Now tris:
	//
	long izNeighMinor[6];
	f64 n_array[6];
	f64 n_n_array[6];
	char buffer[256];

	// Note that n_shards should perhaps be updated less frequently than we find ourselves here -- (?)
	// In any case for sanity, we need some way of passing n_shards to where triangles can reach it in some shape or form#
	// Memory is not at a premium. Creating lists for each tri allows us to access here this way.
	// The alternative is to create an index into n_shards lists for each tri. 
	// That way is still fine since on GPU we load into shared memory the n values -- we just need to have stored them.
	// But this way seems cleaner. So how to create?
	// 
	f64 ourAz;

	iMinor = BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		Our_integral_Lap_Az = 0.0;
		AreaMinor = 0.0;
		tri_len = pVertex->GetTriIndexArray(izTri);
		iprev = tri_len - 1;
		memcpy(szPBC, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);
		memcpy(&ourdata, pData + iMinor, sizeof(plasma_data)); // used only for position ?!

		ourAz = Az_array[iMinor];
		i = 0; iprev = tri_len - 1;

		memcpy(&prevdata, pData + izTri[tri_len - 1], sizeof(plasma_data));
		if (szPBC[iprev] != 0) {
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevdata.pos = Clockwise * prevdata.pos;
			}
			else {
				prevdata.pos = Anticlockwise * prevdata.pos;
			}
		};

		memcpy(&oppdata, &(pData[izTri[0]]), sizeof(plasma_data));
		if (szPBC[i] != 0) {
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				oppdata.pos = Clockwise * oppdata.pos; // let it be one object for now
			}
			else {
				oppdata.pos = Anticlockwise * oppdata.pos;
			}
		};

		endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

		if (pVertex->flags == DOMAIN_VERTEX)
		{
			prevAz = Az_array[izTri[tri_len - 1]];
			oppAz = Az_array[izTri[0]];
			for (i = 0; i < tri_len; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
				nextAz = Az_array[izTri[inext]];

				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					}
					else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};

				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
				//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourAz)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourAz)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				++iprev;

				motion_edge0 = motion_edge1;
				endpt0 = endpt1;

				prevdata.pos = oppdata.pos;
				oppdata.pos = nextdata.pos;
				prevAz = oppAz;
				oppAz = nextAz;
			};
			LapAz_array[iMinor] = Our_integral_Lap_Az / AreaMinor;
			if (LapAz_array[iMinor] < -1.0e-4) {
				iMinor = iMinor;
			}
		}
		else {
			f64_vec2 projendpt0, projendpt1;

			int istart = 0;
			int iend = tri_len;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				istart = 0;
				iend = tri_len - 2;
				// think about this.
				// if it went       3      4
				//                    2 1 0
				// we are going to do edges facing 0,1,2 as 5 = tri_len

				// Given that 0 is the first non-frill, we don't need to do the following?
				//prevAz = (pData + izTri[0])->Az;
				//oppAz = (pData + izTri[1])->Az;
				//endpt0 = THIRD * ((pData + izTri[0])->pos + (pData + izTri[1])->pos + ourdata.pos);

				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};
			prevAz = Az_array[izTri[tri_len - 1]];
			oppAz = Az_array[izTri[istart]];

			for (i = istart; i < iend; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;
				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
				nextAz = Az_array[izTri[inext]];
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					}
					else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);
				// (y2-y1,x1-x2) is also called edge_normal
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// Integral dx of grad x for area
				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourAz)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourAz)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				// I daresay if we wanted to get really confused we could proceed
				// by expanding the dot product into integ_grad and inserting expressions
				// for edge_normal, rearrange to get some kind of havoc.

				++iprev;
				endpt0 = endpt1;
				prevAz = oppAz;
				oppAz = nextAz;
				prevdata.pos = oppdata.pos;
				oppdata.pos = nextdata.pos;
			};

			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				// Now add on the final sides to give area:

				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.

				if (pVertex->flags == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius);
				}
				else {
					endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};

			LapAz_array[iMinor] = Our_integral_Lap_Az / AreaMinor;
		}; // was it DOMAIN_VERTEX

		++pVertex;
		++iMinor;
	};


	pTri = T;
	for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
	{
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)) {
			// do nothing

			Our_integral_Lap_Az = Az_array[pTri->neighbours[0]-T]-Az_array[iMinor];
			AreaMinor = 1.0e-12;

			LapAz_array[iMinor] = Our_integral_Lap_Az;

	//		if (iMinor == 0) printf("Az_array_neigh %1.14E Az_array %1.14E LapAz[0] %1.14E Neigh0 %d \n",
	//			Az_array[pTri->neighbours[0] - T], Az_array[iMinor], Our_integral_Lap_Az,
	//			pTri->neighbours[0] - T);
		}
		else {
			memcpy(izNeighMinor, TriMinorNeighLists[iMinor], sizeof(long) * 6);
			memcpy(szPBC, TriMinorPBCLists[iMinor], sizeof(char) * 6);
			
			Our_integral_Lap_Az = 0.0;
			AreaMinor = 0.0;

			iprev = 5; i = 0;
			memcpy(&ourdata, pData + iMinor, sizeof(plasma_data));
			ourAz = Az_array[iMinor];
			memcpy(&prevdata, pData + izNeighMinor[iprev], sizeof(plasma_data));
			if (szPBC[iprev] != 0) {
				if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
					prevdata.pos = Clockwise * prevdata.pos;
				}
				else {
					prevdata.pos = Anticlockwise * prevdata.pos;
				}
			};
			memcpy(&oppdata, pData + izNeighMinor[0], sizeof(plasma_data));
			if (szPBC[i] != 0) {
				if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
					oppdata.pos = Clockwise * oppdata.pos;
				}
				else {
					oppdata.pos = Anticlockwise * oppdata.pos;
				}
			};
			prevAz = Az_array[izNeighMinor[iprev]];
			oppAz = Az_array[izNeighMinor[0]];
	//		if (iMinor == CHOSEN) printf("\n%d Az %1.14E\n", iMinor, ourAz);
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

	//			if (iMinor == CHOSEN) printf("%d Az %1.14E\n", izNeighMinor[i], oppAz);

				memcpy(&nextdata, pData + izNeighMinor[inext], sizeof(plasma_data));
				// Make contiguous:
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					}
					else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					}
				};

				nextAz = Az_array[izNeighMinor[inext]];

				// New definition of endpoint of minor edge:

				endpt0 = THIRD * (prevdata.pos + ourdata.pos + oppdata.pos);
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourAz)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourAz)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);



	//			if (iMinor == CHOSEN) printf("ourdata.pos %1.14E %1.14E oppdata.pos %1.14E %1.14E gradAz %1.14E %1.14E edge_normal %1.14E %1.14E \ncontrib %1.14E Area_quadrilateral %1.14E \n",
	//				ourdata.pos.x, ourdata.pos.y, oppdata.pos.x, oppdata.pos.y,
	//				grad_Az.x, grad_Az.y, edge_normal.x, edge_normal.y,
	//				grad_Az.dot(edge_normal),
	//				area_quadrilateral);

	//			if (iMinor == CHOSEN) printf("GradAz.x comps: %1.14E %1.14E %1.14E %1.14E\n"
	//				"%1.14E %1.14E %1.14E %1.14E\n",
	//				ourAz, prevAz, oppAz, nextAz,
	//				(ourdata.pos.y - nextdata.pos.y) + (prevdata.pos.y - ourdata.pos.y),
	//				(prevdata.pos.y - ourdata.pos.y) + (oppdata.pos.y - prevdata.pos.y),
	//				(oppdata.pos.y - prevdata.pos.y) + (nextdata.pos.y - oppdata.pos.y),
	//				(nextdata.pos.y - oppdata.pos.y) + (ourdata.pos.y - nextdata.pos.y)
	//			);
				

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;
				prevAz = oppAz;
				oppAz = nextAz;
				memcpy(&prevdata, &oppdata, sizeof(plasma_data));
				memcpy(&oppdata, &nextdata, sizeof(plasma_data));
				motion_edge0 = motion_edge1;
				// There is an even quicker way which is to rotate pointers. No memcpy needed.
				// Is there the concept of a pointer to local data? Not experienced to know what it does.
			};

			if (pTri->u8domain_flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.
			};

			LapAz_array[iMinor] = Our_integral_Lap_Az / AreaMinor;
	//		if (iMinor == CHOSEN) printf("LapAz %1.14E integralLap %1.14E AreaMinor %1.14E\n",
	//			LapAz_array[iMinor], Our_integral_Lap_Az, AreaMinor);
		};

		++pTri;
	};
}

void TriMesh::GetLapCoeffs()
{	
	FILE * fp;
	Triangle * pTri;
	long iVertex, iMinor;
	int i, iprev, inext;
	long neigh_len, tri_len;
	
	long izNeigh[MAXNEIGH], izTri[MAXNEIGH];
	f64_vec2 edge_normal;
	f64_vec2 endpt0, endpt1;
	f64_vec3 outward_flux, v_n;
	plasma_data ourdata, prevdata, oppdata, nextdata; // neighdata0, neighdata1;
	Vertex * pVertex = X;
	f64 area_quadrilateral;
	char szPBC[MAXNEIGH];
	f64 AreaMinor;
	//// Now tris:
	long izNeighMinor[6];
	char buffer[256];
		
	iMinor = BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		AreaMinor = 0.0;
		tri_len = pVertex->GetTriIndexArray(izTri);
		iprev = tri_len - 1;
		memcpy(szPBC, MajorTriPBC[iVertex], sizeof(char)*MAXNEIGH);
		memcpy(&ourdata, pData + iMinor, sizeof(plasma_data)); // used only for position ?!

		memset(LapCoeffvert[iVertex], 0, sizeof(f64)*MAXNEIGH);
		LapCoeffself[iMinor] = 0.0;

		i = 0; iprev = tri_len - 1;

		memcpy(&prevdata, pData + izTri[tri_len - 1], sizeof(plasma_data));
		if (szPBC[iprev] != 0) {
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevdata.pos = Clockwise * prevdata.pos;
			} else {
				prevdata.pos = Anticlockwise * prevdata.pos;
			}
		};

		memcpy(&oppdata, &(pData[izTri[0]]), sizeof(plasma_data));
		if (szPBC[i] != 0) {
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				oppdata.pos = Clockwise * oppdata.pos; // let it be one object for now
			} else {
				oppdata.pos = Anticlockwise * oppdata.pos;
			}
		};

		endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

		if (pVertex->flags == DOMAIN_VERTEX)
		{
			for (i = 0; i < tri_len; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
			
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					} else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};
				
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);

				// . set 0 at start
				// . divide by areaminor at end

				LapCoeffself[iMinor] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(prevdata.pos.y - nextdata.pos.y) - edge_normal.y*(prevdata.pos.x - nextdata.pos.x));
				LapCoeffvert[iVertex][iprev] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(oppdata.pos.y - ourdata.pos.y) - edge_normal.y*(oppdata.pos.x-ourdata.pos.x));
				LapCoeffvert[iVertex][i] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(nextdata.pos.y - prevdata.pos.y) - edge_normal.y*(nextdata.pos.x - prevdata.pos.x));
				LapCoeffvert[iVertex][inext] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(ourdata.pos.y - oppdata.pos.y) - edge_normal.y*(ourdata.pos.x - oppdata.pos.x));
				/*
				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevAz + ourAz)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppAz + prevAz)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextAz + oppAz)*(nextdata.pos.y - oppdata.pos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(ourdata.pos.x - nextdata.pos.x)
					+ (prevAz + ourAz)*(prevdata.pos.x - ourdata.pos.x)
					+ (oppAz + prevAz)*(oppdata.pos.x - prevdata.pos.x)
					+ (nextAz + oppAz)*(nextdata.pos.x - oppdata.pos.x)
					);
				f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += grad_Az.dot(edge_normal);
				*/
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				
				++iprev;
				
				endpt0 = endpt1;
				
				prevdata.pos = oppdata.pos;
				oppdata.pos = nextdata.pos;
				
			};
			LapCoeffself[iMinor] /= AreaMinor;
			for (i = 0; i < tri_len; i++)
				LapCoeffvert[iVertex][i] /= AreaMinor;

			//LapAz_array[iMinor] = Our_integral_Lap_Az/AreaMinor;			
		} else {
			f64_vec2 projendpt0, projendpt1;

			int istart = 0;
			int iend = tri_len;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				istart = 0;
				iend = tri_len - 2;
				// think about this.
				// if it went       3      4
				//                    2 1 0
				// we are going to do edges facing 0,1,2 as 5 = tri_len

				// Given that 0 is the first non-frill, we don't need to do the following?
				//prevAz = (pData + izTri[0])->Az;
				//oppAz = (pData + izTri[1])->Az;
				//endpt0 = THIRD * ((pData + izTri[0])->pos + (pData + izTri[1])->pos + ourdata.pos);

				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				} else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};
			
			for (i = istart; i < iend; i++)
			{
				// Idea to create n at 1/3 out towards neighbour .. shard model defines n at tri centrnT_loids
				// Can infer n by interpolation within triangle.
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = tri_len - 1;

				memcpy(&nextdata, &(pData[izTri[inext]]), sizeof(plasma_data));
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					} else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					};
				};
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);
				// (y2-y1,x1-x2) is also called edge_normal
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;
								
				// Integral dx of grad x for area
				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				LapCoeffself[iMinor] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(prevdata.pos.y - nextdata.pos.y) - edge_normal.y*(prevdata.pos.x - nextdata.pos.x));
				LapCoeffvert[iVertex][iprev] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(oppdata.pos.y - ourdata.pos.y) - edge_normal.y*(oppdata.pos.x - ourdata.pos.x));
				LapCoeffvert[iVertex][i] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(nextdata.pos.y - prevdata.pos.y) - edge_normal.y*(nextdata.pos.x - prevdata.pos.x));
				LapCoeffvert[iVertex][inext] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(ourdata.pos.y - oppdata.pos.y) - edge_normal.y*(ourdata.pos.x - oppdata.pos.x));
			
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				// I daresay if we wanted to get really confused we could proceed
				// by expanding the dot product into integ_grad and inserting expressions
				// for edge_normal, rearrange to get some kind of havoc.

				++iprev;
				endpt0 = endpt1;
				prevdata.pos = oppdata.pos;
				oppdata.pos = nextdata.pos;
			};

			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				// Now add on the final sides to give area:

				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.

				if (pVertex->flags == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius);
				}
				else {
					endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};
			LapCoeffself[iMinor] /= AreaMinor;
			for (i = 0; i < tri_len; i++)
				LapCoeffvert[iVertex][i] /= AreaMinor;
		}; // was it DOMAIN_VERTEX

		++pVertex;
		++iMinor;
	};


	pTri = T;
	for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
	{
		memset(LapCoefftri[iMinor], 0, sizeof(f64) * 6);
		LapCoeffself[iMinor] = 0.0;
		
		if ((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)) {
			LapCoeffself[iMinor] = -1.0;
			LapCoefftri[iMinor][3] = 1.0; // neighbour 0

		} else {
			memcpy(izNeighMinor, TriMinorNeighLists[iMinor], sizeof(long) * 6);
			memcpy(szPBC, TriMinorPBCLists[iMinor], sizeof(char) * 6);

			AreaMinor = 0.0;

			iprev = 5; i = 0;
			memcpy(&ourdata, pData + iMinor, sizeof(plasma_data));
			memcpy(&prevdata, pData + izNeighMinor[iprev], sizeof(plasma_data));
			if (szPBC[iprev] != 0) {
				if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
					prevdata.pos = Clockwise * prevdata.pos;
				} else {
					prevdata.pos = Anticlockwise * prevdata.pos;
				}
			};
			memcpy(&oppdata, pData + izNeighMinor[0], sizeof(plasma_data));
			if (szPBC[i] != 0) {
				if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
					oppdata.pos = Clockwise * oppdata.pos;
				} else {
					oppdata.pos = Anticlockwise * oppdata.pos;
				}
			};
			
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				iprev = i - 1; if (iprev < 0) iprev = 5;

				memcpy(&nextdata, pData + izNeighMinor[inext], sizeof(plasma_data));
				// Make contiguous:
				if (szPBC[inext] != 0) {
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
						nextdata.pos = Clockwise * nextdata.pos;
					} else {
						nextdata.pos = Anticlockwise * nextdata.pos;
					}
				};
				// New definition of endpoint of minor edge:

				endpt0 = THIRD * (prevdata.pos + ourdata.pos + oppdata.pos);
				endpt1 = THIRD * (nextdata.pos + ourdata.pos + oppdata.pos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

			//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
			//	Our_integral_grad_Az += Az_edge * edge_normal;
			//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

				area_quadrilateral = 0.5*(
					(ourdata.pos.x + nextdata.pos.x)*(ourdata.pos.y - nextdata.pos.y)
					+ (prevdata.pos.x + ourdata.pos.x)*(prevdata.pos.y - ourdata.pos.y)
					+ (oppdata.pos.x + prevdata.pos.x)*(oppdata.pos.y - prevdata.pos.y)
					+ (nextdata.pos.x + oppdata.pos.x)*(nextdata.pos.y - oppdata.pos.y)
					);
				
				LapCoeffself[iMinor] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(prevdata.pos.y - nextdata.pos.y)
						- edge_normal.y*(prevdata.pos.x - nextdata.pos.x));
				LapCoefftri[iMinor][iprev] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(oppdata.pos.y - ourdata.pos.y)
						- edge_normal.y*(oppdata.pos.x - ourdata.pos.x));
				LapCoefftri[iMinor][i] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(nextdata.pos.y - prevdata.pos.y)
						- edge_normal.y*(nextdata.pos.x - prevdata.pos.x));
				LapCoefftri[iMinor][inext] += (0.5 / area_quadrilateral)*
					(edge_normal.x*(ourdata.pos.y - oppdata.pos.y)
						- edge_normal.y*(ourdata.pos.x - oppdata.pos.x));
				
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;
				memcpy(&prevdata, &oppdata, sizeof(plasma_data));
				memcpy(&oppdata, &nextdata, sizeof(plasma_data));
				
				// There is an even quicker way which is to rotate pointers. No memcpy needed.
				// Is there the concept of a pointer to local data? Not experienced to know what it does.
			};

			if (pTri->u8domain_flag == CROSSING_INS) {};

			LapCoeffself[iMinor] /= AreaMinor;
			for (i = 0; i < 6; i++)
				LapCoefftri[iMinor][i] /= AreaMinor;
						
		};

		++pTri;
	};
}

void TriMesh::Accelerate2018(f64 h_use, TriMesh * pUseMesh, TriMesh * pDestMesh, f64 evaltime_plus, bool bFeint, bool bUse_n_dest_for_Iz)
{
	
	// Populated inputs:

	// pUseMesh->AreaMinorArray
	// pData+iMinor, pUseMesh->pData+iMinor, 
	// inc // data_use.B
	// data_use.pos
	// AdditionalMomRates <<<<<<<<<<<<<<<<< how used : Nv ?
	// IntegratedGradAz,LapAz, GradTeArray
	// data_1.n for hitting current

	// Outputs:
	// data_1.v
	// data_1.Azdot 
	f64 viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az;

	static real const one_over_kB = 1.0 / kB_; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0 / (kB_*kB_*kB_); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB_)*kB_;
	static real const over_sqrt_m_e_ = 1.0 / sqrt(m_e_);

	Vertex * pVertex = X;
	Triangle * pTri = T;
	// v and A exist in minor cells, so both triangles and vertices.
	// Let's assume then that each vertex contains v and each triangle contains v

	// But when we go to display, we're stuck with just vertex values! eek!
	plasma_data data_k, data_use,data_1; // mostly midpoint step but half-time data can come from data_use
	three_vec3 MomAddRate;

	f64 Iz0 = 0.0, SigmaIzz = 0.0;
	f64_vec3 vn0, omega;
	f64 beta_ne, beta_ni, nu_ne_MT, nu_ni_MT, Lap_Az, nu_in_MT, nu_en_MT, beta_ie_z, sigma_i_zz, sigma_e_zz, Jz0;
	f64_vec2 beta_xy_z, grad_Az, vxy0, gradTe;
	f64 Ez_strength = 0.0;

	// debug:
	f64 vez0_component_vezk, vez0_component_Azdot, vez0_component_Azdotantiadvect,
		vez0_component_Lap_Az, vez0_component_fric, vez0_component_total, integral_n;

	//FILE * fp_individual_sigma_block_340 = fopen("individualCPU.txt", "w");

	f64 M_in = m_i_ / (m_i_ + m_n_);
	f64 M_en = m_e_ / (m_e_ + m_n_);
	f64 M_ni = m_n_ / (m_i_ + m_n_);
	f64 M_ne = m_n_ / (m_e_ + m_n_);
	long iMinor;
	int iPass;
	f64 vez0, viz0, nu_eiBar,nu_eHeart,denom, ROCAzdot_antiadvect;
	char buffer[128];
	f64 Iz_prescribed = GetIzPrescribed(evaltime_plus); // pass time t_k+1

	memset(sigma_tiles, 0, numTilesMinor * sizeof(f64));

	/*FILE * fptri[12];
	FILE * fp1 = fopen("Accel14631.txt", "a");
	FILE * fp2 = fopen("Accel14645.txt", "a");
	fprintf(fp1, "\n\n");
	fprintf(fp2, "\n\n");

	for (int i = 0; i < 12; i++)
	{
		sprintf(buffer, "Accel%d.txt",TriFiles[i]);
		fptri[i] = fopen(buffer, "a");
		fprintf(fptri[i], "\n\n");
	}
	
	FILE * fp;
	*/
	f64 minJz0,maxJz0;
	int maxJzindex;
	for (iPass = 0; iPass < 2; iPass++) {

		Iz0 = 0.0;
		minJz0 = 0.0;
		maxJz0 = 0.0;
		maxJzindex = 0;
		vez0_component_vezk = 0.0;
		vez0_component_Azdot = 0.0;
		vez0_component_Azdotantiadvect = 0.0;
		vez0_component_Lap_Az = 0.0;
		vez0_component_fric = 0.0;
		vez0_component_total = 0.0;
		integral_n = 0.0;

		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			if (
				((iMinor < BEGINNING_OF_CENTRAL) && (T[iMinor].u8domain_flag == DOMAIN_TRIANGLE))
				|| 
				((iMinor >= BEGINNING_OF_CENTRAL) && (X[iMinor - BEGINNING_OF_CENTRAL].flags == DOMAIN_VERTEX))
				)
				// What other types of cell?
			{
				memcpy(&data_k, pData+iMinor, sizeof(plasma_data));
				memcpy(&data_use, pUseMesh->pData + iMinor, sizeof(plasma_data));
				memcpy(&data_1, pDestMesh->pData+iMinor, sizeof(plasma_data));

				memcpy(&MomAddRate, AdditionalMomRates + iMinor, sizeof(three_vec3));
				// CHECK IT MEANS Nv
				
				f64 AreaMinor = pUseMesh->AreaMinorArray[iMinor]; // assumed populated
				
				// Where in memory does n,T,A,B come from?
				// Basically let's put arrays of data objects. 
				// length of array is NMINOR
				// Data includes v,Az,Azdot, n, T (these will be needed!), B

				vn0 = data_k.v_n;

				if (((iMinor == 73061) && (0))) {
					printf("CPU %d vn_k %1.9E %1.9E \n", iMinor, vn0.x, vn0.y);
				}

				vn0.x += h_use * (MomAddRate.neut.x / (data_use.n_n*AreaMinor));
				vn0.y += h_use * (MomAddRate.neut.y / (data_use.n_n*AreaMinor)); 
							// MomAddRate is addition rate for Nv. Divide by N.

				if (((iMinor == 73061) && (0))) {
					printf("CPU %d vn0 with pressure %1.9E %1.9E \n", iMinor, vn0.x, vn0.y);
				}
				{
					// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
					f64 sqrt_Te, ionneut_thermal, electron_thermal,
						lnLambda, s_in_MT, s_en_MT, s_en_visc;

					sqrt_Te = sqrt(data_use.Te);
					ionneut_thermal = sqrt(data_use.Ti / m_i_ + data_use.Tn / m_n_); // hopefully not sqrt(0)
					electron_thermal = sqrt_Te * over_sqrt_m_e_;
					lnLambda = Get_lnLambda(data_use.n, data_use.Te);

					s_in_MT = ::Estimate_Ion_Neutral_MT_Cross_section(data_use.Ti*one_over_kB);
					::Estimate_Ion_Neutral_Cross_sections(data_use.Te*one_over_kB, &s_en_MT, &s_en_visc);
					
					// Need nu_ne etc to be defined:
					nu_ne_MT = s_en_MT * electron_thermal * data_use.n; // have to multiply by n_e for nu_ne_MT
					nu_ni_MT = s_in_MT * ionneut_thermal * data_use.n;
					nu_in_MT = s_in_MT * ionneut_thermal * data_use.n_n;
					nu_en_MT = s_en_MT * electron_thermal * data_use.n_n;
					nu_eiBar = nu_eiBarconst_ * kB_to_3halves*data_use.n*lnLambda / (data_use.Te*sqrt_Te);
					nu_eHeart = 1.87*nu_eiBar + data_use.n_n*s_en_visc*electron_thermal;

//					if ((iMinor == CHOSEN) && (0)) printf("CPU: s_in_MT %1.14E ionneut_thermal %1.14E s_en_MT %1.14E \n",
	//					s_in_MT, ionneut_thermal, s_en_MT);

				}

				f64_vec3 v_e_k = Make3(data_k.vxy, data_k.vez);

				if (iMinor == CHOSEN) printf("CPU %d : vek %1.14E %1.14E %1.14E \n", iMinor,
					v_e_k.x, v_e_k.y, v_e_k.z);

				f64_vec3 v_i_k = Make3(data_k.vxy, data_k.viz);
				vn0 += -0.5*h_use*(M_en)*nu_ne_MT*(data_k.v_n - v_e_k)
					- 0.5*h_use*(M_in)*nu_ni_MT*(data_k.v_n - v_i_k);
				denom = 1.0 + h_use * 0.5*(M_en)*nu_ne_MT + 0.5*h_use*(M_in)*nu_ni_MT;

				if (((iMinor == 73061) && (0))) {
					printf("CPU %d after friction vn0 %1.9E %1.9E \n", iMinor, vn0.x, vn0.y);
				}

				vn0 /= denom; // It is now the REDUCED value
				
				if (((iMinor == 73061) && (0))) {
					printf("CPU %d after divide: vn0 %1.9E %1.9E denom %1.14E \n", iMinor, vn0.x, vn0.y, denom);
				}

				beta_ne = 0.5*h_use*(M_en)*nu_ne_MT / denom;
				beta_ni = 0.5*h_use*(M_in)*nu_ni_MT / denom;

				// Now we do vexy:
				
				grad_Az = GradAz[iMinor];
				Lap_Az = LapAzArray[iMinor];
				gradTe = GradTeArray[iMinor];
				ROCAzdot_antiadvect = ROCAzduetoAdvection[iMinor];

				vxy0 = data_k.vxy
					+ h_use * ((m_e_*MomAddRate.elec.xypart() + m_i_*MomAddRate.ion.xypart())
						/ (data_use.n*(m_i_ + m_e_)*AreaMinor));

				if ((iMinor == CHOSEN) && (0)) {
				//	printf("CPU %d vxy0 %1.14E %1.14E \n", CHOSEN, vxy0.x, vxy0.y);
			//		printf("CPU %d data_k %1.10E %1.10E MomAddRate_elec %1.10E %1.10E \n", CHOSEN, 
				//		data_k.vxy.x,data_k.vxy.y, MomAddRate.elec.x,MomAddRate.elec.y);
			//		printf("CPU %d n %1.10E AreaMinor %1.10E MAR_ion %1.10E %1.10E \n", CHOSEN, data_use.n, AreaMinor,
			//			MomAddRate.ion.x, MomAddRate.ion.y);
				}

				vxy0 += -h_use * (q_/ (2.0*c_*(m_i_ + m_e_)))*(data_k.vez - data_k.viz)*grad_Az
					- (h_use / (2.0*(m_i_ + m_e_)))*(m_n_*M_in*nu_in_MT + m_n_ * M_en*nu_en_MT)*
					(data_k.vxy - data_k.v_n.xypart() - vn0.xypart());

				denom = 1.0 + (h_use / (2.0*(m_i_ + m_e_)))*(m_n_*M_in*nu_in_MT + m_n_ * M_en*nu_en_MT)*(1.0 - beta_ne - beta_ni);
				vxy0 /= denom;
				beta_xy_z = (h_use * q_ / (2.0*c_*(m_i_ + m_e_)*denom)) * grad_Az;

				if ((iMinor == CHOSEN) && (0)) {
		//			printf("CPU %d v0.vxy %1.14E %1.14E denom %1.14E \n", CHOSEN, vxy0.x, vxy0.y, denom);
		//			printf("nu_in_MT %1.14E nu_en_MT %1.14E beta_ne %1.14E \n", nu_in_MT, nu_en_MT, beta_ne);
				}
				omega = eovermc_ * data_use.B; // Perhaps we'd rather B was stored separately

				f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + omega.z*omega.z) /
					(nu_eHeart*(nu_eHeart*nu_eHeart + omega.x*omega.x + omega.y*omega.y + omega.z*omega.z)));

				if ((iPass == 0) || (bFeint == false))
				{
					viz0 = data_k.viz

						+ h_use * MomAddRate.ion.z / (data_use.n*AreaMinor)

						- 0.5*h_use*qoverMc_*(2.0*data_k.Azdot
							+ h_use * ROCAzdot_antiadvect + h_use *c_*c_*(Lap_Az + TWO_PI_OVER_C * q_*data_use.n*(data_k.viz - data_k.vez)))
						- 0.5*h_use*qoverMc_*(data_k.vxy + vxy0).dot(grad_Az);
				} else {
					viz0 = data_k.viz

						+ h_use * MomAddRate.ion.z / (data_use.n*AreaMinor)

						- 0.5*h_use*qoverMc_*(2.0*data_k.Azdot
							+ h_use * ROCAzdot_antiadvect + h_use *c_*c_*( TWO_PI_OVER_C * q_*data_use.n*(data_k.viz - data_k.vez)))
						- 0.5*h_use*qoverMc_*(data_k.vxy + vxy0).dot(grad_Az);
				};
		//		if ((iMinor == CHOSEN)) {
		//			printf("viz0 I. %1.14E \n", viz0);
		//			printf("contribs were: k+MAR %1.14E \n    Azdotk %1.14E \n   ROC %1.14E \n   JviaLap %1.14E \n   Lorenzmag %1.14E \n",
		//			data_k.viz + h_use * MomAddRate.ion.z / (data_use.n*AreaMinor),
		//				-0.5*h_use*qoverMc_*(2.0*data_k.Azdot),
		//				-0.5*h_use*qoverMc_*h_use * ROCAzdot_antiadvect,
		//				-0.5*h_use*qoverMc_*h_use *c_*c_*(TWO_PI_OVER_C * q_*data_use.n*(data_k.viz - data_k.vez)),
		//				-0.5*h_use*qoverMc_*(data_k.vxy + vxy0).dot(grad_Az)
		//				);
		//			printf("viz0 due to LapAz if iPass==0 || bFeint==false: %1.14E iPass %d\n",
		//				-0.5*h_use*qoverMc_*h_use *c_*c_*Lap_Az, iPass);
		//		}

				viz0 +=
					1.5*h_use*nu_eiBar*((omega.x*omega.z - nu_eHeart * omega.y)*gradTe.x +
					(omega.y*omega.z + nu_eHeart * omega.x)*gradTe.y) /
						(m_i_*nu_eHeart*(nu_eHeart*nu_eHeart + omega.x*omega.x + omega.y*omega.y + omega.z*omega.z));
	//			if ((iMinor == CHOSEN)) printf("viz0 with thermal force %1.14E \n", viz0);
				
				// Insert ionization effect... // ????????????????????????????????????

				viz0 += -h_use * 0.5*M_ni*nu_in_MT *(data_k.viz - data_k.v_n.z - vn0.z)
					+ h_use * 0.5*(moverM_)*nu_ei_effective*(data_k.vez - data_k.viz);
			
		//		if ((iMinor == CHOSEN)) printf("viz0 contrib i-n %1.14E e-i %1.14E. viz0 %1.14E \n",
		//			-h_use * 0.5*M_ni*nu_in_MT *(data_k.viz - data_k.v_n.z - vn0.z),
		//			h_use * 0.5*(moverM_)*nu_ei_effective*(data_k.vez - data_k.viz),
		//			viz0);
				
				denom = 1.0 + h_use * h_use*PI*qoverM_*q_*data_use.n + h_use * 0.5*qoverMc_*(grad_Az.dot(beta_xy_z)) +
					h_use * 0.5*M_ni*nu_in_MT*(1.0 - beta_ni) + h_use * 0.5*moverM_*nu_ei_effective;

				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc_*h_use*c_*c_ / denom;

				viz0 /= denom;

	//			if ((iMinor == CHOSEN)) printf("viz0 divided %1.14E denom %1.14E \n", viz0, denom);

				sigma_i_zz = h_use * qoverM_ / denom;
				beta_ie_z = (h_use*h_use*PI*qoverM_*q_*data_use.n
					+ 0.5*h_use*qoverMc_*(grad_Az.dot(beta_xy_z))
					+ h_use * 0.5*M_ni*nu_in_MT*beta_ne
					+ h_use * 0.5*moverM_*nu_ei_effective) / denom;

				if ((iPass == 0) || (bFeint == false))
				{
					vez0 = data_k.vez

						+ h_use * MomAddRate.elec.z / (data_use.n*AreaMinor)

						+ h_use * 0.5*eovermc_*(2.0*data_k.Azdot
							+ h_use * ROCAzdot_antiadvect
							+ h_use *c_*c_*(Lap_Az
								+ TWO_PI_OVER_C * q_*data_use.n*(data_k.viz + viz0 - data_k.vez)))

						+ 0.5*h_use*eovermc_*(data_k.vxy + vxy0 + viz0 * beta_xy_z).dot(grad_Az);
					

					if ((iMinor == CHOSEN)) {
						printf("iPass %d vez_k %1.14E vez0 %1.14E contribs\n"
							"mom add rate %1.14E\n"
							"Azdot term %1.14E \n"
							"Lap Az term %1.14E \n"
							"J via Azdot %1.14E \n"
							"Lorenzmag %1.14E \n",
							iPass, data_k.vez, vez0,
							h_use * MomAddRate.elec.z / (data_use.n*AreaMinor),
							h_use * 0.5*eovermc_*2.0*data_k.Azdot,
							h_use *0.5*eovermc_*h_use *c_*c_*Lap_Az,
							h_use *0.5*eovermc_*h_use *c_*c_*TWO_PI_OVER_C * q_*data_use.n*(data_k.viz + viz0 - data_k.vez),
							0.5*h_use*eovermc_*(data_k.vxy + vxy0 + viz0 * beta_xy_z).dot(grad_Az)
						);
						printf("viz0 %1.14E \n", viz0);
					}
				} else {
					vez0 = data_k.vez

						+ h_use * MomAddRate.elec.z / (data_use.n*AreaMinor)

						+ h_use * 0.5*eovermc_*(2.0*data_k.Azdot
							+ h_use * ROCAzdot_antiadvect
							+ h_use *c_*c_*(
								TWO_PI_OVER_C * q_*data_use.n*(data_k.viz + viz0 - data_k.vez)))

						+ 0.5*h_use*eovermc_*(data_k.vxy + vxy0 + viz0 * beta_xy_z).dot(grad_Az);

//					vez0_coeff_on_Lap_Az = h_use * h_use*0.5*eovermc_*c_*c_; // always same, do not need
					if ((iMinor == CHOSEN)) {
						printf("iPass %d vez_k %1.14E vez0 now %1.14E contribs\n MomRate %1.14E \n"
							"Azdot contrib %1.14E\nantiadvect %1.14E \n"
							"JviaAdotz %1.14E Lorenzmag %1.14E \n",
							iPass, data_k.vez, vez0,
							h_use * MomAddRate.elec.z / (data_use.n*AreaMinor),
							h_use * 0.5*eovermc_*2.0*data_k.Azdot,
							h_use * 0.5*eovermc_*h_use * ROCAzdot_antiadvect,
							h_use *0.5*eovermc_*h_use *c_*c_*TWO_PI_OVER_C * q_*data_use.n*(data_k.viz + viz0 - data_k.vez),
							0.5*h_use*eovermc_*(data_k.vxy + vxy0 + viz0 * beta_xy_z).dot(grad_Az)
						);
						printf("viz0 %1.14E \n", viz0);
					}					// No Lap Az contribution
				};				
				vez0 -=
					1.5*h_use*nu_eiBar*((omega.x*omega.z - nu_eHeart * omega.y)*gradTe.x +
					(omega.y*omega.z + nu_eHeart * omega.x)*gradTe.y) /
						(m_e_*nu_eHeart*(nu_eHeart*nu_eHeart + omega.x*omega.x + omega.y*omega.y + omega.z*omega.z));
				// could store this from above and put opposite -- dividing by m_e_ instead of m_i_
				vez0 += -0.5*h_use*M_ne*nu_en_MT*(data_k.vez - data_k.v_n.z - vn0.z - beta_ni * viz0)
					- 0.5*h_use*nu_ei_effective*(data_k.vez - data_k.viz - viz0);
				
				if ((iMinor == CHOSEN)) {
					printf("thermal force contrib to vez %1.14E\n"
						"nu_eiBar %1.14E nu_eHeart %1.14E\n"
						"gradTe %1.14E %1.14E \n"
						"omega %1.14E %1.14E %1.14E \n",
						-1.5*h_use*nu_eiBar*((omega.x*omega.z - nu_eHeart * omega.y)*gradTe.x +
						(omega.y*omega.z + nu_eHeart * omega.x)*gradTe.y) /
							(m_e_*nu_eHeart*(nu_eHeart*nu_eHeart + omega.x*omega.x + omega.y*omega.y + omega.z*omega.z)),
						nu_eiBar, nu_eHeart,
						gradTe.x, gradTe.y,
						omega.x, omega.y, omega.z);
					printf("e-n %1.14E i-n %1.14E  viz0 %1.14E multiplier %1.14E\n",
						-0.5*h_use*M_ne*nu_en_MT*(data_k.vez - data_k.v_n.z - vn0.z - beta_ni * viz0),
						-0.5*h_use*nu_ei_effective*(data_k.vez - data_k.viz - viz0), viz0,
						-0.5*h_use*nu_ei_effective);
				}			
				denom = 1.0 + (h_use*h_use*PI*q_*qoverm_*data_use.n
					+ 0.5*h_use*eovermc_*(grad_Az.dot(beta_xy_z)))*(1.0 - beta_ie_z)
					+ 0.5*h_use*M_ne*nu_en_MT*(1.0 - beta_ne - beta_ni * beta_ie_z)
					+ 0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z);
				
				// where is vez0 /= denom? Bug? below.						
				
				vez0_coeff_on_Lap_Az = h_use * h_use*0.5*eovermc_* c_*c_; // always same, divide by denom in a minute
				vez0_coeff_on_Lap_Az += (
					h_use * 0.5*eovermc_*h_use *c_*c_*TWO_PI_OVER_C * q_*data_use.n
					 + 0.5*h_use*eovermc_*(beta_xy_z.dot(grad_Az))
					+ 0.5*h_use*M_ne*nu_en_MT*beta_ni
					+ 0.5*h_use*nu_ei_effective) // coeff on viz in vez
											* viz0_coeff_on_Lap_Az; // knock-on effect
				// 100% this is right?

				// We are missing divide by denom there.					

				if ((iMinor == CHOSEN) && (0)) printf("vez0_coeff_on_Lap_Az undivided %1.14E : cc_contrib %1.14E , coeff_on_viz_in_vez %1.12E viz0_coeff %1.14E \n",
					vez0_coeff_on_Lap_Az, h_use * h_use*0.5*eovermc_* c_*c_,
					(h_use * 0.5*eovermc_*h_use *c_*c_*TWO_PI_OVER_C * q_*data_use.n
						+ 0.5*h_use*eovermc_*(beta_xy_z.dot(grad_Az))
						+ 0.5*h_use*M_ne*nu_en_MT*beta_ni
						+ 0.5*h_use*nu_ei_effective),
					viz0_coeff_on_Lap_Az);

				sigma_e_zz = (-h_use * qoverm_ + h_use * h_use*PI*q_*qoverm_*data_use.n*sigma_i_zz
					+ h_use * 0.5*eovermc_*(grad_Az.dot(beta_xy_z))*sigma_i_zz
					+ 0.5*h_use*M_ne*nu_en_MT*beta_ni*sigma_i_zz
					+ 0.5*h_use*nu_ei_effective*sigma_i_zz)
					/ denom;

				bool bFile = false;
				//for (int i = 0; i < 12; i++) {
				//	if (iMinor == TriFiles[i]) {
				//		fp = fptri[i];
				//		bFile = true;
				//	};
				//}
				
				if (bFile && (iPass == 1))
				{
					/*
					fprintf(fp, "pos %1.14E %1.14E\nvez0 %1.12E data_k %1.12E momcomp %1.12E Azcomp %1.12E gradAzFX %1.12E gradTeFX %1.12E \n"
						"nu_enFX %1.12E nu_eiFX %1.12E denom %1.12E denom_n_comp %1.12E \n",
						data_k.pos.x,data_k.pos.y,
						vez0, data_k.vez, h_use * MomAddRate.elec.z / (data_use.n*AreaMinor),
						h_use * 0.5*eovermc_*(2.0*data_k.Azdot + h_use *c_*c_*(Lap_Az
							+ TWO_PI_OVER_C * q_*data_use.n*(data_k.viz + viz0 - data_k.vez))),
						+0.5*h_use*eovermc_*(data_k.vxy + vxy0 + viz0 * beta_xy_z).dot(grad_Az),
						-1.5*h_use*nu_eiBar*((omega.x*omega.z - nu_eHeart * omega.y)*gradTe.x +
						(omega.y*omega.z + nu_eHeart * omega.x)*gradTe.y) /
							(m_e_*nu_eHeart*(nu_eHeart*nu_eHeart + omega.x*omega.x + omega.y*omega.y + omega.z*omega.z)),
						-0.5*h_use*M_ne*nu_en_MT*(data_k.vez - data_k.v_n.z - vn0.z - beta_ni * viz0),
						-0.5*h_use*nu_ei_effective*(data_k.vez - data_k.viz - viz0),
						denom,
						h_use*h_use*PI*q_*qoverm_*data_use.n);
					fprintf(fp, "heovermc_ Azdot %1.12E hh0.5qcoverm Lap Az %1.12E hhqq_pi_over_m n (viz-vez) %1.12E \n",
						h_use*eovermc_*data_k.Azdot,
						h_use*h_use*0.5*qoverm_*c_*Lap_Az,
						h_use*h_use*q_*qoverm_*PI*(data_k.viz + viz0 - data_k.vez)*data_use.n);
					fprintf(fp, "data_k.Azdot %1.12E Lap_Az %1.12E viz-vez %1.12E n %1.12E \n", data_k.Azdot, Lap_Az, (data_k.viz + viz0 - data_k.vez), data_use.n);
					fprintf(fp, "sigma_e_zz %1.12E -hq/m %1.12E sigma_iFX1 %1.12E gradAzFX %1.12E sigma_iFX2 %1.12E \n",
						sigma_e_zz, -h_use * qoverm_,
						h_use * h_use*PI*q_*qoverm_*data_use.n*sigma_i_zz,
						h_use * 0.5*eovermc_*(grad_Az.dot(beta_xy_z))*sigma_i_zz,
						0.5*h_use*M_ne*nu_en_MT*beta_ni*sigma_i_zz
						+ 0.5*h_use*nu_ei_effective*sigma_i_zz);
					*/
				};
				
				if ((iMinor == CHOSEN)) 
					printf("CPU %d vez0 before dividing %1.14E iPass %d\n", CHOSEN, vez0, iPass);

				vez0 /= denom;
				vez0_coeff_on_Lap_Az /= denom;

				if ((iMinor == CHOSEN) && (0))
					printf("vez0_coeff divided %1.14E coeff_viz_vez_divided %1.14E \n", vez0_coeff_on_Lap_Az,
					(h_use *c_*c_*TWO_PI_OVER_C * q_*data_use.n
						+ 0.5*h_use*eovermc_*(beta_xy_z.dot(grad_Az))
						+ 0.5*h_use*M_ne*nu_en_MT*beta_ni
						+ 0.5*h_use*nu_ei_effective)/denom
						);

				if ((iMinor == CHOSEN)) printf("vez0 divided %1.14E LapAzcontrib(not applied) %1.14E over denom %1.14E \n"
					"Lap Az %1.14E factor %1.14E \n"
					, vez0,
					-0.5*h_use*eovermc_*h_use *c_*c_*Lap_Az,
					-0.5*h_use*eovermc_*h_use *c_*c_*Lap_Az/denom,
					Lap_Az, -0.5*h_use*eovermc_*h_use *c_*c_
					);

				if ((iMinor == CHOSEN) && (0)) {
					printf("~~~~~~~~~~~\nCPU %d vez0(divided, no lap contrib on 2nd pass) %1.14E denom %1.14E bFeint %d \n", CHOSEN, vez0, denom,
						bFeint ? 1 : 0);
					printf("sigma_e_zz %1.14E \n", sigma_e_zz);
				}


#define NODEBUGJZ0
#ifndef NODEBUGJZ0
				vez0_component_vezk -= q * data_use.n*AreaMinor*data_k.vez;
				vez0_component_Azdot -= q * data_use.n*AreaMinor*
					(h_use*eovermc_*data_k.Azdot);
				vez0_component_Azdotantiadvect -= q * data_use.n*AreaMinor*
					(h_use*0.5*eovermc_*(h_use*ROCAzdot_antiadvect));
				vez0_component_Lap_Az -= q * data_use.n*AreaMinor*
					(h_use*0.5*eovermc_*(h_use*c_*c*Lap_Az));
				vez0_component_fric -= q * data_use.n*AreaMinor*
					(-0.5*h_use*M_ne*nu_en_MT*(data_k.vez - data_k.v_n.z - vn0.z - beta_ni * viz0)
						- 0.5*h_use*nu_ei_effective*(data_k.vez - data_k.viz - viz0));
				vez0_component_total -= q_*data_use.n*AreaMinor*vez0;
				integral_n += data_use.n*AreaMinor;
#endif
				
				// Now update viz(Ez):
				viz0 += beta_ie_z * vez0;
				sigma_i_zz += beta_ie_z * sigma_e_zz;

				viz0_coeff_on_Lap_Az += beta_ie_z*vez0_coeff_on_Lap_Az;

	//			if (iMinor == CHOSEN) printf("vez0 %1.14E beta_ie_z %1.14E contribtoviz0 %1.14E\n",
	//				vez0, beta_ie_z, beta_ie_z * vez0);
	//			// did beta_ie_z ever get divided by  denom_i???

				// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez

				f64 EzShape = GetEzShape__(data_use.pos.modulus());
				sigma_i_zz *= EzShape;
				sigma_e_zz *= EzShape;
				f64 sigma_zz;
				if (iPass == 0) {
					
					// Now calculate relationship Jz(Ez) :
					if (bUse_n_dest_for_Iz) {
						Jz0 = q_* data_1.n*(viz0 - vez0);
						//(0.5*(viz_k-vez_k)+ 0.5* (viz0 - vez0));
						sigma_zz = //0.5*q_* data_use.n*(sigma_i_zz - sigma_e_zz);
							q_* data_1.n*(sigma_i_zz - sigma_e_zz);

						if ((iMinor == CHOSEN) && (0)) 
							printf("data_1.n %1.12E sigma_zz %1.12E AreaMinor %1.12E\n\n",
								data_1.n, sigma_zz, AreaMinor);
					} else {
						Jz0 = q_* data_use.n*(viz0 - vez0);
						//(0.5*(viz_k-vez_k)+ 0.5* (viz0 - vez0));
						sigma_zz = //0.5*q_* data_use.n*(sigma_i_zz - sigma_e_zz);
							q_* data_use.n*(sigma_i_zz - sigma_e_zz);
						if ((0) && (iMinor == CHOSEN))
							printf("data_use.n %1.12E sigma_zz %1.12E AreaMinor %1.12E\n\n",
								data_use.n, sigma_zz, AreaMinor);
					}

					// Now ask self if this is about to go majorly wrong, if we change to do half-time.
					// Will Ez be set so that it ends up oscillating?
					// Why allow that possibility?
					// Can't we just stick with the idea that we set Ez for the end of the step,
					// which is obviously the most stable?

					// Is there any reason that when it matters, on the subcycle, that we can't use the correct data_1.n?
					// Come to that, can we not load that anyway?
					// ==============================================================
					// Fudge: use n_half unless it is the last step of the subcycle.
					

					Iz0 += Jz0 * AreaMinor; // Think dest mesh area minor has not been populated so usemesh is best we can do
					SigmaIzz += sigma_zz * AreaMinor;

					long iTile = iMinor / threadsPerTileMinor;

					sigma_tiles[iTile] += sigma_zz*AreaMinor;
				//	if (iTile == 340) fprintf(fp_individual_sigma_block_340,
				//		"%d: %1.14E %1.14E \n", iMinor, sigma_zz, sigma_zz*AreaMinor);

					if ((iMinor == CHOSEN) && (0)) {
						printf("CPU %d: Iz0 %1.12E  sigma_zz %1.12E \n\n",
							CHOSEN, Iz0, sigma_zz*AreaMinor);
					};

				} else {
					data_1.vez = vez0 + sigma_e_zz * Ez_strength;
					data_1.viz = viz0 + sigma_i_zz * Ez_strength;
					//if ((iMinor == CHOSEN)) printf("data_1.vez %1.14E vez0 %1.14E sigma_e_zz %1.14E Ez_strength %1.14E\n================\n",
					//	data_1.vez, vez0, sigma_e_zz, Ez_strength);
				//	if ((iMinor == CHOSEN)) printf("viz0 %1.14E data_1.viz %1.14E sigma_i_zz %1.14E Ez %1.14E\n================\n",
				//		viz0, data_1.viz, sigma_i_zz, Ez_strength);

					if (bFeint == false) {
						data_1.vxy = vxy0 + beta_xy_z * (data_1.viz - data_1.vez);
						data_1.v_n = vn0;
						data_1.v_n.x += (beta_ne + beta_ni)*data_1.vxy.x;
						data_1.v_n.y += (beta_ne + beta_ni)*data_1.vxy.y;
						data_1.v_n.z += beta_ne * data_1.vez + beta_ni * data_1.viz;

						if ((iMinor == 73061) && (0)) printf("CPU: 73061 vn0 %1.14E %1.14E \n"
							"beta_ne %1.14E beta_ni %1.14E \n"
							"vxy %1.14E %1.14E \n"
							"vn.xy %1.14E %1.14E \n",
							vn0.x, vn0.y, beta_ne, beta_ni, data_1.vxy.x, data_1.vxy.y,
							data_1.v_n.x, data_1.v_n.y);

						data_1.Azdot = data_k.Azdot
							+ h_use * ROCAzdot_antiadvect + h_use *c_*c_*(Lap_Az +
								0.5*FOURPI_OVER_C_ * q_*data_use.n*(data_k.viz + data_1.viz
									- data_k.vez - data_1.vez));

						if ((iMinor == CHOSEN) && (0)) printf("\ndata_k.Azdot %1.14E antiadvect %1.14E\n"
							"Lapcontrib %1.14E  contrib %1.14E  n_use.n %1.14E \n"
							"vizk %1.14E viz %1.14E vezk %1.14E vez %1.14E\n\n",
							data_k.Azdot,
							h_use * ROCAzdot_antiadvect,
							h_use *c_*c_*Lap_Az,
							h_use *c_*c_*(
								0.5*FOURPI_OVER_C_ * q_*data_use.n*(data_k.viz + data_1.viz
									- data_k.vez - data_1.vez)),
							data_use.n, 
							data_k.viz, data_1.viz, data_k.vez, data_1.vez);

					//	if (iMinor == CHOSEN) printf("viz used for Azdot: %1.14E\n\n", data_1.viz);

						if (bUse_n_dest_for_Iz) {
							Iz0 += q_* data_1.n*(data_1.viz - data_1.vez)*AreaMinor;
						} else {
							Iz0 += q_* data_use.n*(data_1.viz - data_1.vez)*AreaMinor;
						}

						memcpy(pDestMesh->pData + iMinor, &data_1, sizeof(plasma_data));
					} else {
						Azdot0[iMinor] = data_k.Azdot
							+ h_use * ROCAzdot_antiadvect + h_use *c_*c_*(
								0.5*FOURPI_OVER_C_ * q_*data_use.n*(data_k.viz + data_1.viz
									- data_k.vez - data_1.vez));
						gamma[iMinor] = h_use *c_*c_*(1.0 + 0.5*FOURPI_OVER_C_ * q_*data_use.n*
							(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));
						 
						// Incorrect. 08/01/19
						// We ignored the knock-on effect of Lap_Az on vez via viz, when we set up vez0_coeff_on_Lap_Az.
						
						if ((iMinor == CHOSEN) && (0)) printf("\n@@@@@@@@@@@@@@@@@@@@\nCPU data_k.Azdot %1.14E antiadvect %1.14E\n"
							"contrib %1.14E  n_use.n %1.14E \n"
							"vizk %1.14E viz %1.14E vezk %1.14E vez %1.14E\n\n",
							data_k.Azdot,
							h_use * ROCAzdot_antiadvect,
							h_use *c_*c_*(
								0.5*FOURPI_OVER_C_ * q_*data_use.n*(data_k.viz + data_1.viz
									- data_k.vez - data_1.vez)),
							data_use.n,
							data_k.viz, data_1.viz, data_k.vez, data_1.vez);

						if ((iMinor == CHOSEN) && (0)) {
							printf("CPU Azdot0 %1.14E gamma %1.14E \n", Azdot0[iMinor], gamma[iMinor]);
							printf("gamma components: n %1.14E viz_coeff_on_Lap %1.14E vez_coeff_on_Lap %1.14E",
								data_use.n, viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az);
						};
					}

					// This does not seem to come out equal to what it says on the graph.
					// Seeing Jz = -2e16 and Azdot consistent with this, yet
					// seeing Iz attained = -1e9. The well radius is about 0.1cm so 0.01*pi*(-2e16) = -6e14 roughly.
					Jz0 = q_* data_use.n*(data_1.viz-data_1.vez);
					if (Jz0 < minJz0) minJz0 = Jz0;
		//			if ((iMinor > BEGINNING_OF_CENTRAL + 14630) 
			//			&& (iMinor < BEGINNING_OF_CENTRAL + 14650)) {
				//		printf("Area minor: %1.8E Jz %1.8E contribIz %1.8E\n",
					//		AreaMinor, q_* data_1.n*(data_1.viz - data_1.vez), q_* data_1.n*(data_1.viz - data_1.vez)*AreaMinor);
					//};
					if (Jz0 > maxJz0) {
						maxJz0 = Jz0;
						maxJzindex = iMinor;
					};

					if (bFile)
					{
//						fprintf(fp, "vez %1.14E viz %1.14E Azdot %1.14E n %1.14E \n", data_1.vez, data_1.viz, data_1.Azdot, data_1.n);
					};
					// Now we want to do A's advance?
					// We did in another routine...										
				}
			} else {
				// Non-domain triangle or vertex
				// ==============================

				// Need to decide whether crossing_ins triangle will experience same accel routine as the rest?
				// I think yes so go and add it above??

				// We said v_r = 0 necessarily to avoid sending mass into ins.
				// So how is that achieved there? What about energy loss?
				// Need to determine a good way. Given what v_r in tri represents. We construe it to be AT the ins edge so 
				// ...

				if ((iMinor < BEGINNING_OF_CENTRAL) && ((T[iMinor].u8domain_flag == OUTER_FRILL) || 
					(T[iMinor].u8domain_flag == INNER_FRILL)))
				{
					// Set Az equal to neighbour in every case, after Accelerate routine.
									
					if (iPass > 0) {
						(pDestMesh->pData + iMinor)->Azdot = 0.0;
						Azdot0[iMinor] = 0.0;
						gamma[iMinor] = 0.0;
					}
				}
				else {				
		//			if ((iMinor >= BEGINNING_OF_CENTRAL) && (X[iMinor - BEGINNING_OF_CENTRAL].flags == OUTERMOST))
			//		{
						// ??

						// Evolve here normally as for vertex in ins

						// We will assume for now that Az is flat going outwards. ?
						
						// Alternative is to set Azdot = 0 here and let Az be fixed here for all time;
						// a bit like assuming that whatever Lap Az was here observed, it can be neglected as there
						// is a radial slope to cancel it on the outside. That seems less good.

				//	} else {
//						if ((iMinor >= BEGINNING_OF_CENTRAL) && (X[iMinor - BEGINNING_OF_CENTRAL].flags == INNERMOST))
	//					{

							// Innermost vertex Az evolution:
							// EITHER: Assume no evolution since Lap Az = Jz = 0
							// or
							// assume flat looking inwards and therefore Lap Az is whatever it appears due to that

							// If we allow contributions to Lap Az above then we have to ask how that will have come out.


							// Correct way:
							// Evolve here normally as for vertex in ins





					//	} else {
							// A triangle or vertex in the insulator or anode.
							// Determine the amount of ReverseJz arc that intersects our cell's polygon:

							// Let's make it go right through the middle of a triangle row for simplicity.

							f64 Jz = 0.0;
							if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
							{
								// Azdotdot = c^2 (Lap Az + 4pi/c Jz)

								// ASSUME we are fed Iz_prescribed.
								Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);
								Jz /= pUseMesh->AreaMinorArray[iMinor]; // Iz would come from multiplying back by area and adding.

								/*
								f64 r0, r1, r2, r;
								f64_vec2 u0, u1, u2, intersect1, intersect2;

								pTri->MapLeftIfNecessary(u0, u1, u2);

								r0 = u0.modulus();
								r1 = u1.modulus();
								r2 = u2.modulus();
								r = DEVICE_RADIUS_ANODE;
								if (r0 > r) {
									if (r1 > r) {
										// No intersection r0-r1
										if (r2 > r) {
											printf("error all points outside anode\n");
											getch();
										} else {
											// r2 is the low one
											intersect1 = (u0 * (r - r2) + u2 * (r0 - r)) / (r0 - r2);
											intersect2 = (u1 * (r - r2) + u2 * (r1 - r)) / (r1 - r2);
										}
									} else {
										// r1 < r
										intersect1 = (u0*(r - r1) + u1 * (r0 - r)) / (r0 - r1);
										if (r2 > r) { // r1 is the low one
											intersect2 = (u2*(r - r1) + u1 * (r2 - r)) / (r2 - r1);
										} else { // r0 is the high one
											intersect2 = (u0*(r - r2) + u2 * (r0 - r)) / (r0 - r2);
										};
									};									
								} else {
									// r0 < r
									if (r1 > r) {
										intersect1 = (u0*(r - r1) + u1 * (r0 - r)) / (r0 - r1);
										if (r2 > r) {
											intersect2 = (u0*(r - r2) + u2 * (r0 - r)) / (r0 - r2);
										} else {
											// r2 < r, r1 > r
											intersect2 = (u2*(r - r1) + u1 * (r2 - r)) / (r2 - r1);
										};
									} else {
										// r1 < r, r0 < r
										if (r2 > r) { // r2 is the high one
											intersect1 = (u0 * (r - r2) + u2 * (r0 - r)) / (r0 - r2);
											intersect2 = (u1 * (r - r2) + u2 * (r1 - r)) / (r1 - r2);
										} else {
											printf("error - all points inside anode\n"); getch();
										};
									};
								};

								// Calculate line length:
								f64 intersecting_length = (intersect2 - intersect1).modulus();
								
								// Probably the easiest way is to get these lengths before time; maybe even save them
								// but sqrt is cheaper than a random access so don't worry about that.
								// But do add them up to make TotalArc and save that --- that seems like a good idea.
								
								f64 TotalArc = FULLANGLE * DEVICE_RADIUS_ANODE;
								f64 Izhere = -Iz_prescribed * intersecting_length / TotalArc;

								Jz = Izhere / AreaMinorArray[iMinor];
								*/
								// There is a shortcut:?
								// We know Jz along the curve, and this is a point on a curve -- make the centroids
								// live on the curve.
								// Well it doesn't work so very well. Jz depends on the WIDTH of that arc.
								// which is undefined. DOuble the indefinitely small width, halve Jz there.
								// Rather we probably need the AVERAGE Jz in the cell in order to advance the average Az in the cell.

								// So we DO need to apportion Iz to the cells.
								// Then we divide Iz_cell area to give average Jz and advance A "weakly".								
							};
							
							if ((iPass > 0) && (bFeint == false))
								pDestMesh->pData[iMinor].Azdot = pData[iMinor].Azdot
									// + h_use * ROCAzdot_antiadvect // == 0
									+ h_use * (c_*c_*LapAzArray[iMinor] + 4.0*PI*c_*Jz);

							Azdot0[iMinor] = pData[iMinor].Azdot + h_use * 4.0*PI*c_*Jz;
							gamma[iMinor] = h_use *c_*c_;
					//	};
				//	};					
				};
				
			};
		};

		// .Collect Jz = Jz0 + sigma_zz Ez_strength on each minor cell
		// .Estimate Ez
		// sigma_zz should include EzShape for this minor cell
		if (iPass == 0) {
			Ez_strength = (Iz_prescribed - Iz0) / SigmaIzz;

//			FILE * fp_ = fopen("sigma_tile.txt", "w");
//			for (long i = 0; i < numTilesMinor; i++)
//				fprintf(fp_, "%d sigma_tile %1.14E \n", i, sigma_tiles[i]);
//			fclose(fp_);

			printf("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
				"CPU Iz0 %1.14E Iz_presc %1.14E diff %1.6E SigmaIzz %1.13E Ez_strength %1.14E \n\n",
				Iz0,Iz_prescribed,Iz_prescribed-Iz0,SigmaIzz,Ez_strength);
		/*	printf("Iz0 components: \n"
				"vezk %1.9E \n"
				"Azdot %1.9E \n"
				"Azdotantiadvect %1.9E \n"
				"Lap_Az %1.9E \n"
				"fric %1.9E \n"
				"total vez %1.9E \n"
				"integral n = %1.9E \n\n",
				vez0_component_vezk,
				vez0_component_Azdot,
				vez0_component_Azdotantiadvect,
				vez0_component_Lap_Az,
				vez0_component_fric,
				vez0_component_total,
				integral_n);*/
		} else {
	//		printf("\nSecond pass: Iz attained %1.14E \n", Iz0);
	//		printf("Min Jz0 : %1.9E Max Jz0: %1.9E \n", minJz0, maxJz0);
	//		printf("max Jz found at %d\n", maxJzindex);
		}
	};
	/*for (int i = 0; i < 12; i++)
		fclose(fptri[i]);
	fclose(fp1);
	fclose(fp2);*/
	//fclose(fp_individual_sigma_block_340);
}


bool inline in_domain(Vector2 u)
{
	return (u.x*u.x+u.y*u.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER);
}
/*void TriMesh::Advance(TriMesh * pDestMesh)
{
	this->
	printf("CreateMeshDisplacement_zero_future_pressure();\n");

	CreateMeshDisplacement_zero_future_pressure(); // creates default displacement
	// and sets up nT = pTri->temp_f64
	// For now, in a cheap and dirty way.

	printf("SolveForAdvectedPositions(pDestMesh);\n");

	SolveForAdvectedPositions(pDestMesh);
	// creates data on dest mesh inc. comp htg

	// Idea for a sequence:

	// 1. Advect to new vertex positions
	// 2. Copy over triangle lists. 
	
	// . (De-tangle overlaps... how to then act?)

	// . Transfer data and apply compressive heating
	// (requires a call to pDestMesh->RecalculateVertexCellAreas()
	//  how does that play into wrapping over PBC?)
	// it works as long as we are either unwrapped, or wrapped with updated periodic flag.
	// . Initially, pTriDest->periodic = pTri->periodic
	
	// . Wrap around PBC. (Rotate Nv if already placed. pTri->DecrementPeriodic)

	// That is to be all included in the above routine.


	// . Interpolate to get A : AFTER wrap. 
	// It seems more natural to interpolate within domain only.
	// Small chance: we might be outside tranche and not be within a tri that is considered periodic
	// if we do it unwrapped?

	printf("pDestMesh->InterpolateA And Adot From(this);\n");

	pDestMesh->InterpolateAFrom(this); // hmm

	printf("pDestMesh->Redelaunerize(true,true);\n");

	pDestMesh->Redelaunerize(true, // 'to exhaustion'
							 true // try to replace fluids
							 );

	printf("pDestMesh->GetBFromA();\n");

	pDestMesh->GetBFromA();

	// Now we have to replace data:
	// For each flip, we have to decide what amount has transferred between vertices.

	// Stage II:

	printf("pDestMesh->Set_nT_and_Get_Pressure(SPECIES_ION);\n");

	pDestMesh->Set_nT_and_Get_Pressure(SPECIES_ION);
	pDestMesh->Set_nT_and_Get_Pressure(SPECIES_ELEC);
	pDestMesh->Set_nT_and_Get_Pressure(SPECIES_NEUT);

	printf("pDestMesh->ComputeOhmsLaw();\n");

	pDestMesh->ComputeOhmsLaw();

	printf("pDestMesh->Solve_A_phi(false);\n");
	printf("any key: \n"); getch();

	pDestMesh->Solve_A_phi(false); // not initial solve

	printf("done.stop.\n");
	getch();

	// Stage III:

	//pDestMesh->Evolve();
	
	// species relative advection,
	// heat and momentum diffusion,
	// ionisation and heating.
	
	
	
	// ===
	
	// We need to start from saying what planes apply on each shard.
	// Should we do it as we go?
	// Is there any advantage to doing otherwise?
	// In 2D we should avoid cycle of flips.

	// In a flip, 2 vertex-centered cells are strictly growing and
	// 2 are strictly decreasing.
	// Therefore we can 
	//   . attribute from the giving cells to the taking cells
	//   . conclude the totals for both by adding / subtracting from original.
	// 
	// The taking cells are the ones that are originally opposing.
	// ie they are the unshared vertices of the tris that will flip.
	// The shared vertices are going to lose mass, and the shards that are
	// concerned are the 3 ones involving one of the tri centroids of the flip tris.

	
	// 2. B. 3 shards, we wish to apportion into overlap with 3 known polygons,
	// for each of 2 sides; and integrate planes in each case.
	
	// 2. C. Subtract to see what's left over in the losing vertcells; add to give total in gaining vertcells.




	// Be careful about flips of ins-crossing triangles!!
	// (Consider afterwards)
	// ---------------------
	

	
	
	// ______________________________________________________________________________
	// Try storing old-time pressure at vertices (the prev mesh ones?). Then we can
	// assume the transition to newer pressure takes place 
	// gradually, when we are doing the evolution.
	// Have to be careful: the old cell region was different before Delaunay flip.	
}
*/

/*void TriMesh::InterpolateAFrom(TriMesh * pSrcMesh)
{	
	// We can readily do interpolation in triangles.
	// A values live on vertices so we just make planes with 3 of them.
	long iVertex;
	Vertex * pVertex, * pVertSrc;
	real beta[3];
	Triangle * pTri;
	Vector2 u[3];
	long iWhich, iTri;
		
	pVertex = X;
	pVertSrc = pSrcMesh->X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if ((pVertex->flags == INNERMOST) ||
			(pVertex->flags == OUTERMOST))
		{
			pVertex->A = pVertSrc->A;
			pVertex->Adot = pVertSrc->Adot;
		} else {
			
			Triangle * pSeedTri = pSrcMesh->T + pVertSrc->GiveMeAnIndex();
		
			if ((pSeedTri->u8domain_flag == INNER_FRILL) || 
				(pSeedTri->u8domain_flag == OUTER_FRILL))
			{
				pSeedTri = pSeedTri->neighbours[0];
			}
			pTri = pSrcMesh->ReturnPointerToTriangleContainingPoint(
				pSeedTri,
				pVertex->pos.x,pVertex->pos.y
				); 
			// presumably this works best when pos is actually
			// inside the tranche. So we call InterpolateA for wrapped mesh.
			
			if (pTri->periodic == 0) {
				
				GetInterpolationCoefficients(beta, 
							pVertex->pos.x,	pVertex->pos.y,
							pTri->cornerptr[0]->pos,
							pTri->cornerptr[1]->pos,
							pTri->cornerptr[2]->pos);

				pVertex->A =  beta[0]*pTri->cornerptr[0]->A
							+ beta[1]*pTri->cornerptr[1]->A
							+ beta[2]*pTri->cornerptr[2]->A;
								
				pVertex->Adot = beta[0]*pTri->cornerptr[0]->Adot
							+ beta[1]*pTri->cornerptr[1]->Adot
							+ beta[2]*pTri->cornerptr[2]->Adot;
			} else {
				// We apply this routine to WRAPPED MESH.
				// So if it inhabits a periodic src tri,
				// we can go by its own x-position to tell which side for A.
		
				pTri->MapLeftIfNecessary(u[0],u[1],u[2]);
				
				int par[3];
				pTri->GetParity(par);
				memset(&(pVertex->A),0,sizeof(Vector3));
				memset(&(pVertex->Adot),0,sizeof(Vector3));
				if (pVertex->pos.x > 0.0) {					
					GetInterpolationCoefficients(beta, 
							pVertex->pos.x,	pVertex->pos.y,
							Clockwise*u[0],
							Clockwise*u[1],
							Clockwise*u[2]);
					// equivalently we could just rotate anticlockwise our pos to get
					// at beta.

					if (par[0] == 0){
						pVertex->A += beta[0]*(Clockwise3*pTri->cornerptr[0]->A);
						pVertex->Adot += beta[0]*(Clockwise3*pTri->cornerptr[0]->Adot);
					} else {
						pVertex->A += beta[0]*(pTri->cornerptr[0]->A);
						pVertex->Adot += beta[0]*(pTri->cornerptr[0]->Adot);
					};	
					if (par[1] == 0) {
						pVertex->A += beta[1]*(Clockwise3*pTri->cornerptr[1]->A);
						pVertex->Adot += beta[1]*(Clockwise3*pTri->cornerptr[1]->Adot);
					} else {
						pVertex->A += beta[1]*(pTri->cornerptr[1]->A);
						pVertex->Adot += beta[1]*(pTri->cornerptr[1]->Adot);
					};
					if (par[2] == 0) {
						pVertex->A += beta[2]*(Clockwise3*pTri->cornerptr[2]->A);
						pVertex->Adot += beta[2]*(Clockwise3*pTri->cornerptr[2]->Adot);
					} else {
						pVertex->A += beta[2]*pTri->cornerptr[2]->A;
						pVertex->Adot += beta[2]*pTri->cornerptr[2]->Adot;
					};
				} else {
					GetInterpolationCoefficients(beta, 
							pVertex->pos.x,	pVertex->pos.y,
							u[0],u[1],u[2]);
					if (par[0] == 0){
						pVertex->A += beta[0]*pTri->cornerptr[0]->A;
						pVertex->Adot += beta[0]*pTri->cornerptr[0]->Adot;
					} else {
						pVertex->A += beta[0]*(Anticlockwise3*pTri->cornerptr[0]->A);
						pVertex->Adot += beta[0]*(Anticlockwise3*pTri->cornerptr[0]->Adot);
					};
					if (par[1] == 0) {
						pVertex->A += beta[1]*pTri->cornerptr[1]->A;
						pVertex->Adot += beta[1]*pTri->cornerptr[1]->Adot;
					} else {
						pVertex->A += beta[1]*(Anticlockwise3*pTri->cornerptr[1]->A);
						pVertex->Adot += beta[1]*(Anticlockwise3*pTri->cornerptr[1]->Adot);
					};
					if (par[2] == 0) {
						pVertex->A += beta[2]*pTri->cornerptr[2]->A;
						pVertex->Adot += beta[2]*pTri->cornerptr[2]->Adot;
					} else {
						pVertex->A += beta[2]*(Anticlockwise3*pTri->cornerptr[2]->A);
						pVertex->Adot += beta[2]*(Anticlockwise3*pTri->cornerptr[2]->Adot);
					};
				};
			};
		};
		++pVertex;
		++pVertSrc;
	};
}*/
/*

void TriMesh::GetBFromA()
{
	// This routine will set B on vertices based on A on vertices.
	
	// To set B at an edge we can use the quadrilateral of A values
	// near that edge. That is not what we are about to do, but we 
	// could make a subroutine to get curl of A around a ConvexPolygon.
	// And that seems very wise.
	
	ConvexPolygon cp;
	Triangle * pTri;
	Vertex * pVertex;
	long iVertex, iTri;
	Vector2 u;
	int i;
	long izTri[128];
	Vector3 A[128];	
	long tri_len;

	// Reset triangle centroids:
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateCentroid(this->InnermostFrillCentroidRadius,
			this->OutermostFrillCentroidRadius);
		++pTri;
	};

	// For each vertex create a ConvexPolygon of centroids
	// and a list of A-values:
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		tri_len = pVertex->GetTriIndexArray(izTri);
		cp.Clear();
		// Special allowance for outer edge is only to be that
		// we add on this point itself as the last one.		
		for (i = 0; i < tri_len; i++)
		{
			pTri = T + izTri[i];
			cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
			A[i] = pTri->GetAAvg(); // Note this assumes the data is in pVertex->A
			if ((pTri->periodic) && (pVertex->pos.x > 0.0))
				A[i] = Clockwise3*A[i];
		};		
		if ((pVertex->flags == CONCAVE_EDGE_VERTEX) ||
			(pVertex->flags == CONVEX_EDGE_VERTEX) )
		{
			cp.add(pVertex->pos);
			A[i] = pVertex->A;
		};		
		// Estimate the average by integration of curl A :		
		pVertex->B = cp.Get_curl2D_from_anticlockwise_array(A);		
		++pVertex;
	};
}

void TriMesh::GetBFromA_minors()
{
	// How to get integrated curl:
	// 	Integral_x = (A[i].z + A[inext].z)
	//               *(coord[inext].x - coord[i].x); // [anticlockwise]--> -dAz/dy
	//  Integral_y = (A[i].z + A[inext].z)
	//           	 *(coord[inext].y - coord[i].y); // [anticlockwise]--> dAz/dx

	// Form Az at minor corner by interpolation between minors.
	// 
	pData[iMinor].Bx = 
	pData[iMinor].Bz = BZ_CONSTANT;
}
*/
/*
void TriMesh::GetGradTeOnVertices()
{
	long iTri;
	Triangle * pTri;
	real beta[3];
	// Assign T to triangles: what is best fit?
	// To do it properly: 
	// Do minmod then match up where they do not match.
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {

			GetInterpolationCoefficients(beta, pTri->cent.x,pTri->cent.y,
				pTri->cornerptr[0]->centroid,
				pTri->cornerptr[1]->centroid,
				pTri->cornerptr[2]->centroid);
				pTri->temp_f64 = beta[0]*(pTri->cornerptr[0]->Elec.heat/pTri->cornerptr[0]->Elec.mass)
							   + beta[1]*(pTri->cornerptr[1]->Elec.heat/pTri->cornerptr[1]->Elec.mass)
							   + beta[2]*(pTri->cornerptr[2]->Elec.heat/pTri->cornerptr[2]->Elec.mass);
		} else {
			pTri->temp_f64 = 0.0;
			// the value will not be used.
		};
		++pTri;
	};

	// On GPU how to do Grad Te: each tri collects info from 3 places, =>
	// 3 x random access;
	// contributes to Grad Te at 3 corners. Can have 1 thread per 1 tri.
	// Same with vertex A -> vertex B.
	ConvexPolygon cp;
	long tri_len, i, iVertex;
	long izTri[128];
	real Te[128];	
	Vertex * pVertex = X;
	bool bDone = false;
	for (iVertex= 0; iVertex < numVertices; iVertex++)
	{
		if ((pVertex->flags == DOMAIN_VERTEX) ||
			(pVertex->flags == CONVEX_EDGE_VERTEX))
		{
			tri_len = pVertex->GetTriIndexArray(izTri);

			// Get centroid polygon:
			cp.Clear();			
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
					cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
					Te[i] = pTri->temp_f64; // Note this assumes the data is in pVertex->A
				} else {
					// Insulator tri: do not include, but instead put in the centre point:
					if (bDone == false)
					{
						bDone = true;
						cp.add(pVertex->centroid);
						Te[i] = pVertex->Elec.heat/pVertex->Elec.mass;
					};
				};
			};
			// As in code for A->B :
			// Special allowance for outer edge is only to be that
			// we add on this point itself as the last one.
			if (pVertex->flags == CONVEX_EDGE_VERTEX)
			{
				cp.add(pVertex->pos); 
				Te[i] = pVertex->Elec.heat/pVertex->Elec.mass;
			};
			// But remember heat lives in the whole "house-shaped" cell.
			// We could choose otherwise but let's not.
			// So, AreaCell routine contains house shape? 

			pVertex->GradTe = cp.Get_grad_from_anticlockwise_array(Te);
		} else {
			pVertex->GradTe.x = 0.0; pVertex->GradTe.y = 0.0;
		};
		++pVertex;
	};
}
*/
//
//class CalculateAccelsClass
//{
//public:
//	// exists only to do a calculation repeatedly from some stored data
//
//	Vector3 omega_ce, omega_ci;
//	Tensor3 omega_ci_cross;
//	
//	real nu_eiBar, nu_eHeart, nu_ieBar, 
//			nu_en_MT, nu_in_MT, 
//			nu_ne_MT, nu_ni_MT,
//			n_i, n_n, n_e;
//			
//	real heat_transfer_rate_in,heat_transfer_rate_ni,
//		 heat_transfer_rate_en,heat_transfer_rate_ne,
//		 heat_transfer_rate_ei,heat_transfer_rate_ie;
//	
//	Vector3 a_neut_pressure,
//			a_ion_pressure,
//			ROC_v_ion_due_to_Rie,
//			ROC_v_ion_thermal_force; 
//
//	Tensor3 Upsilon_nu_eHeart;
//	Tensor3 Rie_thermal_force_matrix;
//	Tensor3 Rie_friction_force_matrix;
//	Tensor3 Ratio_times_Upsilon_eHeart;
//	
//	real fric_dTe_by_dt_ei;
//		
//	real StoredEz;
//
//	bool bNeutrals;
//
//	// EASIER WAY:
//	// Let's just stick simple Ohm's Law v_e(v_i,v_n)
//	// in CalculateCoefficients.
//
//	real SimpleOhms_vez0, SimpleOhms_beta_neutz, Ohms_vez0, Ohms_sigma;
//	Vector3 SimpleOhms_beta_ion;
//
//	CalculateAccelsClass(){};
//
//	void CalculateCoefficients(Vertex * pVertex)
//	{
//		// NOTE: Uses GradTe so it better exist.
//		
//		static Tensor3 const ID3x3 (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
//		static real const TWOTHIRDSqsq_= 2.0*q_*q/3.0;
//		static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
//		static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
//		static real const kB_to_3halves = sqrt(kB)*kB;
//		static real const over_sqrt_m_ion = 1.0/sqrt(m_i_);
//		static real const over_sqrt_m_e_ = 1.0/sqrt(m_e_);
//		static real const qoverMc_ = q/(m_i_*c);
//		static real const eovermc_ = q/(m_e_*c);
//		static real const NU_EI_FACTOR = 1.0/(3.44e5);
//		static real const nu_eiBarconst_ = //(4.0/3.0)*sqrt(2.0*PI/m_e_)*q_*q_*q_*q_;
//		// don't know in what units but it IS exactly what we already had - see Formulary
//									1.0/(3.44e5);
//
//		real area, det;
//		real T_ion, T_n, T_e, sqrt_Te, ionneut_thermal, electron_thermal,
//			lnLambda, s_in_MT, s_in_visc, s_en_MT,s_en_visc,
//			nu_en_visc;
//		//Vector3 const E, Vector3 const vrel_e, real * const scratch
//		// The first thing we need to do is collect
//	
//		// nu_eibar, nu_in, nu_en
//		// ======================
//
//		// Get nu_eiBar
//		// Get nu_en, nu_in, nu_ni, nu_ne, nu_eHeart
//	
//		StoredEz = pVertex->E.z; 
//
//		area = pVertex->area;
//		n_i = pVertex->Ion.mass/area;
//		n_e = pVertex->Elec.mass/area;
//		n_n = pVertex->Neut.mass/area;
//
//		if (pVertex->Ion.mass > 0.0) {
//			T_Ion = pVertex->Ion.heat/pVertex->Ion.mass;   
//		} else {
//			T_Ion = 0.0;
//		};
//		if (pVertex->Neut.mass > 0.0) {
//			T_n = pVertex->Neut.heat/pVertex->Neut.mass;
//		} else {
//			T_n = 0.0;
//		};
//		if (pVertex->Elec.mass > 0.0) {
//			T_e = pVertex->Elec.heat/pVertex->Elec.mass;
//			sqrt_Te = sqrt(T_e);
//		} else {
//			T_e = 0.0;
//			sqrt_Te = 0.0;
//		};
//		
//		ionneut_thermal = sqrt(T_ion/m_i_+T_n/m_n_); // hopefully not sqrt(0)
//		electron_thermal = sqrt_Te*over_sqrt_m_e_; // possibly == 0
//
//		lnLambda = Get_lnLambda(n_i,T_e); // anything strange in there?
//
//		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &s_in_MT, &s_in_visc);
//		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&s_en_MT,&s_en_visc);
//		// To use combined temperature looks to be more intelligent -- rel temp GZSB(6.55) for ion, neutral at least.
//		
//		if (T_e != 0.0) {
//			nu_eiBar = nu_eiBarconst_*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
//		} else {
//			nu_eiBar = 0.0;
//		};
//		
//		nu_ieBar = nu_eiBar; // always same when n_e=n_i
//
//		nu_en_MT = n_n*s_en_MT*electron_thermal;
//		nu_in_MT = n_n*s_in_MT*ionneut_thermal;
//		nu_ne_MT = n_e*s_en_MT*electron_thermal;
//		nu_ni_MT = n_i*s_in_MT*ionneut_thermal;
//		
//		nu_en_visc = n_n*s_en_visc*electron_thermal; 
//		
//		// those should all be fine though may == 0
//		
//		nu_eHeart = 1.87*nu_eiBar + nu_en_visc; // note, used visc
//				 
//		heat_transfer_rate_in = (2.0*m_i_*m_n_/((m_i_+m_n_)*(m_i_+m_n_)))
//										*nu_in_MT; // ratio nu_in/nu_ni = n_n/n_i
//		heat_transfer_rate_ni = (2.0*m_i_*m_n_/((m_i_+m_n_)*(m_i_+m_n_)))
//										*nu_ni_MT;
//		heat_transfer_rate_ne = (2.0*m_e_*m_n_/((m_e_+m_n_)*(m_e_+m_n_)))
//										*nu_ne_MT;
//		heat_transfer_rate_en = (2.0*m_e_*m_n_/((m_e_+m_n_)*(m_e_+m_n_)))
//										*nu_en_MT;
//		heat_transfer_rate_ei = (2.0*m_e_*m_i_/((m_e_+m_i_)*(m_e_+m_i_)))
//										*nu_eiBar;
//		heat_transfer_rate_ie = (2.0*m_e_*m_i_/((m_e_+m_i_)*(m_e_+m_i_)))
//										*nu_ieBar;
//		
//		// OK that bit is clear and as expected.
//		// So what is the difference when we transfer NT ?
//
//		// (n_n/n_i) transfer_rate_ni = transfer_rate_in
//		
//		omega_ce = eovermc_*pTri->B;
//		omega_ci = qoverMc_*pTri->B; // note: if ion acceleration stage, we could if we wanted work out B at k+1 first.
//		omega_ci_cross.MakeCross(omega_ci);
//		
//		// Populate Upsilon(nu_eHeart):
//		real nu = nu_eHeart;
//		Vector3 omega = omega_ce;
//		det = nu*nu + omega.dot(omega);
//
//		// (nu - omega x ) ^-1 :
//		Upsilon_nu_eHeart.xx = nu*nu+omega.x*omega.x;
//		Upsilon_nu_eHeart.xy = -nu*omega.z + omega.x*omega.y;
//		Upsilon_nu_eHeart.xz = nu*omega.y + omega.x*omega.z;
//		Upsilon_nu_eHeart.yx = nu*omega.z + omega.x*omega.y;
//		Upsilon_nu_eHeart.yy = nu*nu + omega.y*omega.y;
//		Upsilon_nu_eHeart.yz = -nu*omega.x + omega.y*omega.z;
//		Upsilon_nu_eHeart.zx = -nu*omega.y + omega.z*omega.x;
//		Upsilon_nu_eHeart.zy = nu*omega.x + omega.y*omega.z;
//		Upsilon_nu_eHeart.zz = nu*nu + omega.z*omega.z;
//		
//		Upsilon_nu_eHeart = Upsilon_nu_eHeart/det;
//	
//		if (nu_eHeart > 0.0) {
//			Ratio_times_Upsilon_eHeart = (nu_eiBar/nu_eHeart)*Upsilon_nu_eHeart;
//		} else {
//			ZeroMemory(&Ratio_times_Upsilon_eHeart,sizeof(Tensor3));
//		};
//
//		Rie_friction_force_matrix = 
//			nu_ieBar*(m_e_/m_i_)*(ID3x3-0.9*Ratio_times_Upsilon_eHeart);
//		// multiply by (v_e-v_i) for ions
//
//		Rie_thermal_force_matrix = 
//			((1.5/m_i_)*(nu_ieBar/nu_eHeart)*Upsilon_nu_eHeart);
//		// We multiply by +GradTe for ions
//		
//		ZeroMemory(&vrel_e,sizeof(Vector3));
//		if (pVertex->Elec.mass > 0.0) 
//			vrel_e = pVertex->Elec.mom/pVertex->Elec.mass
//			         - pVertex->Ion.mom/pVertex->Ion.mass;	
//// !!!!!!!!!!!!!!!!!! Note bene.
//		
//		ROC_v_ion_thermal_force = Rie_thermal_force_matrix * Make3(pVertex->GradTe,0.0)
//
//		ROC_v_ion_due_to_Rie =  ROC_v_ion_thermal_force
//		                         + R_ie_friction_force_matrix*vrel_e;
//
//		// ===
//
//		if (pTri->ion.mass != 0.0) {
//			
//			a_ion_pressure.x = pTri->dNv_pressure.ion.x/pTri->ion.mass;
//			a_ion_pressure.y = pTri->dNv_pressure.ion.y/pTri->ion.mass;
//			a_ion_pressure.z = 0.0;
//
//			Vector3 a_elec_pressure;
//
//			a_elec_pressure.x = pTri->dNv_pressure.elec.x/pTri->elec.mass;
//			a_elec_pressure.y = pTri->dNv_pressure.elec.y/pTri->elec.mass;
//			a_elec_pressure.z = 0.0;
//
//			// MELD THE TWO ACCELS 
//			a_ion_pressure = (m_i_*a_ion_pressure + m_e_*a_elec_pressure)/(m_i_ + m_e_);
//
//			// ^ pTri->E will no longer exist. 
//			// It's pTri->Ez only, which will be from Az, chEz_ext.
//			
//		} else {
//			//ZeroMemory(&a_ion_pressure_and_E_accel, sizeof(Vector3));
//			ZeroMemory(&a_ion_pressure, sizeof(Vector3));
//		};
//
//		if (pTri->neut.mass > 0.0) {
//			a_neut_pressure.x = pTri->dNv_pressure.neut.x/pTri->neut.mass;
//			a_neut_pressure.y = pTri->dNv_pressure.neut.y/pTri->neut.mass;
//			a_neut_pressure.z = 0.0;
//		} else {
//			ZeroMemory(&a_neut_pressure, sizeof(Vector3));
//		};
//		// All of these things, do not change, because we do not change E, vrel, pressure.
////		
////
////		fric_dTe_by_dt_ei = 0.0;
////	// The fact is that if x-y current is dropped, you get just a scalar Ohm's Law.
////
////		// This illustrates why we need to go back to having current.
////
////		// Roll on the next version...
////
////		real chi = (m_n_/(m_e_+m_n_))*this->nu_en_MT
////			+ (1.0-0.9*this->Ratio_times_Upsilon_eHeart.zz)*this->nu_eiBar;
////
////		SimpleOhms_vez0 = (-qoverm_*StoredEz 
////			-(1.5/m_e_)*((this->Ratio_times_Upsilon_eHeart*pTri->GradTe).z)
////			)/chi; 
////
////		Ohms_vez0 = (
////			-(1.5/m_e_)*((this->Ratio_times_Upsilon_eHeart*pTri->GradTe).z)
////			)/chi; 
////		// Where is thermal pressure force? Doesn't exist in z dimension.
////
////		Ohms_sigma = -qoverm_/chi;
////
////		SimpleOhms_beta_ion.x = -this->omega_ce.y/chi;
////		SimpleOhms_beta_ion.y = this->omega_ce.x/chi;
////		SimpleOhms_beta_ion.z = ((1.0-0.9*this->Ratio_times_Upsilon_eHeart.zz)*this->nu_eiBar)/chi;
////
////		SimpleOhms_beta_neutz = (m_n_/(m_e_+m_n_))*this->nu_en_MT/chi;
////
////		// 03/04/16
//
//	}
//
//	void inline Populate_Acceleration_Coefficients_no_pressure(real H[6][6], real a0[6]) 
//	{
//		real factor;
//
//		 ZeroMemory(H,sizeof(real)*6*6);
//		 ZeroMemory(a0, sizeof(real)*6);
//
//		 // Pressure not applied.
//
//		 a0[2] += this->StoredEz*qoverM_; // StoredEz: populated or not?
//		
//		 // magnetic Lorentz for ions:
//		// a_ion.z -= omega_ci.x*v_ion.y-omega_ci.y*v_ion.x;
//		 H[2][0] += omega_ci.y;
//		 H[2][1] -= omega_ci.x;
//
//		// species combining gives:
//		//a_ion.x -= (m_i_/(m_e_+m_i_))*
//		//				omega_ci.y*(v_ion.z-v_e.z); 
//		 // v_e is something we are passed.
//		//a_ion.y += (m_i_/(m_e_+m_i_))*
//		//				omega_ci.x*(v_ion.z-v_e.z);
//		 H[0][2] -= (m_i_/(m_e_+m_i_))*omega_ci.y;
//		 H[1][2] += (m_i_/(m_e_+m_i_))*omega_ci.x;
//
//		 //a0[0] += (m_i_/(m_e_+m_i_))*omega_ci.y*v_e_z;
//		 factor = (m_i_/(m_e_+m_i_))*omega_ci.y;
//		 a0[0] += factor*this->SimpleOhms_vez0;
//		 H[0][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[0][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[0][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[0][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 
//		 //a0[1] -= (m_i_/(m_e_+m_i_))*omega_ci.x*v_e_z;
//		 factor = -(m_i_/(m_e_+m_i_))*omega_ci.x;
//		 a0[1] += factor*this->SimpleOhms_vez0;
//		 H[1][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[1][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[1][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[1][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 // e-i Friction:
//		 a0[2] += ROC_v_ion_thermal_force.z;
//		 a0[2] += Rie_friction_force_matrix.zz*this->SimpleOhms_vez0;
//		 H[2][0] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.x;
//		 H[2][1] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.y;
//		 H[2][2] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.z;
//		 H[2][5] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_neutz;
//
//		 H[2][2] -= Rie_friction_force_matrix.zz; // v_i term
//		
//		 if (bNeutrals) {
//			
//			// i-n, e-n friction:
//			
//			real Combined =	(m_e_*m_n_/((m_e_+m_i_)*(m_e_+m_n_)))*nu_en_MT +
//						(m_i_*m_n_/((m_e_+m_i_)*(m_i_+m_n_)))*nu_in_MT  ;
//
//			//a_ion.x += Combined*(v_neut.x - v_ion.x);
//			H[0][3] += Combined;
//			H[0][0] -= Combined;
//			H[1][4] += Combined;
//			H[1][1] -= Combined;
//			
//			//a_ion.z += (m_n_/(m_i_+m_n_))*nu_in_MT*(v_neut.z - v_ion.z);
//			H[2][2] -= (m_n_/(m_i_+m_n_))*nu_in_MT;
//			H[2][5] += (m_n_/(m_i_+m_n_))*nu_in_MT;
//
//			factor = (m_i_/(m_i_+m_n_))*nu_ni_MT;
//			//a_neut += (m_i_/(m_i_+m_n_))*nu_ni_MT*(v_ion - v_neut)
//			 //   + (m_e_/(m_n_+m_e_))*nu_ne_MT*(v_e - v_neut);
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][2] += factor;
//			H[5][5] -= factor;
//			factor = (m_e_/(m_n_+m_e_))*nu_ne_MT;
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][5] -= factor;
//			
//			//a0[5] += factor*v_e_z;
//			a0[5] += factor*SimpleOhms_vez0;
//			H[5][0] += factor*this->SimpleOhms_beta_ion.x;
//			H[5][1] += factor*this->SimpleOhms_beta_ion.y;
//			H[5][2] += factor*this->SimpleOhms_beta_ion.z;
//			H[5][5] += factor*this->SimpleOhms_beta_neutz;
//		} else {
//			H[3][3] = 1.0;
//			H[4][4] = 1.0;
//			H[5][5] = 1.0;
////			fric_heat_energy_rate_in = 0.0;
////			fric_heat_energy_rate_en_over_ne = 0.0;
//		}
//	} // almost exact same as above routine.
//
//
//		void inline Populate_Acceleration_Coefficients_no_pressure(real H[6][6], real a0[6]) 
//	{
//		real factor;
//
//		 ZeroMemory(H,sizeof(real)*6*6);
//		 ZeroMemory(a0, sizeof(real)*6);
//
//		 // Pressure not applied.
//
//		 a0[2] += this->StoredEz*qoverM_; // StoredEz: populated or not?
//		
//		 // magnetic Lorentz for ions:
//		// a_ion.z -= omega_ci.x*v_ion.y-omega_ci.y*v_ion.x;
//		 H[2][0] += omega_ci.y;
//		 H[2][1] -= omega_ci.x;
//
//		// species combining gives:
//		//a_ion.x -= (m_i_/(m_e_+m_i_))*
//		//				omega_ci.y*(v_ion.z-v_e.z); 
//		 // v_e is something we are passed.
//		//a_ion.y += (m_i_/(m_e_+m_i_))*
//		//				omega_ci.x*(v_ion.z-v_e.z);
//		 H[0][2] -= (m_i_/(m_e_+m_i_))*omega_ci.y;
//		 H[1][2] += (m_i_/(m_e_+m_i_))*omega_ci.x;
//
//		 //a0[0] += (m_i_/(m_e_+m_i_))*omega_ci.y*v_e_z;
//		 factor = (m_i_/(m_e_+m_i_))*omega_ci.y;
//		 a0[0] += factor*this->SimpleOhms_vez0;
//		 H[0][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[0][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[0][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[0][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 
//		 //a0[1] -= (m_i_/(m_e_+m_i_))*omega_ci.x*v_e_z;
//		 factor = -(m_i_/(m_e_+m_i_))*omega_ci.x;
//		 a0[1] += factor*this->SimpleOhms_vez0;
//		 H[1][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[1][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[1][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[1][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 // e-i Friction:
//		 a0[2] += ROC_v_ion_thermal_force.z;
//		 a0[2] += Rie_friction_force_matrix.zz*this->SimpleOhms_vez0;
//		 H[2][0] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.x;
//		 H[2][1] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.y;
//		 H[2][2] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_ion.z;
//		 H[2][5] += Rie_friction_force_matrix.zz*this->SimpleOhms_beta_neutz;
//
//		 H[2][2] -= Rie_friction_force_matrix.zz; // v_i term
//		
//		 if (bNeutrals) {
//			
//			// i-n, e-n friction:
//			
//			real Combined =	(m_e_*m_n_/((m_e_+m_i_)*(m_e_+m_n_)))*nu_en_MT +
//						(m_i_*m_n_/((m_e_+m_i_)*(m_i_+m_n_)))*nu_in_MT  ;
//
//			//a_ion.x += Combined*(v_neut.x - v_ion.x);
//			H[0][3] += Combined;
//			H[0][0] -= Combined;
//			H[1][4] += Combined;
//			H[1][1] -= Combined;
//			
//			//a_ion.z += (m_n_/(m_i_+m_n_))*nu_in_MT*(v_neut.z - v_ion.z);
//			H[2][2] -= (m_n_/(m_i_+m_n_))*nu_in_MT;
//			H[2][5] += (m_n_/(m_i_+m_n_))*nu_in_MT;
//
//			factor = (m_i_/(m_i_+m_n_))*nu_ni_MT;
//			//a_neut += (m_i_/(m_i_+m_n_))*nu_ni_MT*(v_ion - v_neut)
//			 //   + (m_e_/(m_n_+m_e_))*nu_ne_MT*(v_e - v_neut);
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][2] += factor;
//			H[5][5] -= factor;
//			factor = (m_e_/(m_n_+m_e_))*nu_ne_MT;
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][5] -= factor;
//			
//			//a0[5] += factor*v_e_z;
//			a0[5] += factor*SimpleOhms_vez0;
//			H[5][0] += factor*this->SimpleOhms_beta_ion.x;
//			H[5][1] += factor*this->SimpleOhms_beta_ion.y;
//			H[5][2] += factor*this->SimpleOhms_beta_ion.z;
//			H[5][5] += factor*this->SimpleOhms_beta_neutz;
//		} else {
//			H[3][3] = 1.0;
//			H[4][4] = 1.0;
//			H[5][5] = 1.0;
////			fric_heat_energy_rate_in = 0.0;
////			fric_heat_energy_rate_en_over_ne = 0.0;
//		}
//	} // almost exact same as with-pressure routine.
//
//};
/*
void TriMesh::Set_nT_and_Get_Pressure(int species)
{
	// Must have populated: 
	//  Vertex::AreaCell
	//  anticlockwise tri & neigh index arrays
	//  Triangle::cent
	
	Vector2 tri_cent, sum, u[3];
	ConvexPolygon cp;
	int par[3];
	Triangle * pTri;
	long iTri, iVertex;
	Vertex * pVertex;
	real beta[3];

	
	// Input: AreaCell

	// What it is actually going to involve.

	// 1. Compute nT on triangles. Get thermal pressure at vertices.

	// see pVertex->GetGradTeOnVertices for that.

	// PLAN to :

	// a. Set centroid of each vertex polygon
	// We'll assume nT = NT/Area at each vertcell centroid.
	// b. Compute domain tri centroid nT = as found from plane.
	// c. Compute insulator nT.

	// For these we now try to attain NT_vertcell.
	// If we have 2 vertcells we aim for the average using both. Can assume in a region
	// near the edge of them, it is near the average of both.
	// If we are below just 1, fill these in afterwards and solve to minimize
	// side-to-side curvature, hit the NT for the cell. ?
	// There may be a quick way without doing any solving.

	// d. Go again? : consider what NT_vertcell is being achieved...
	// This only becomes really necessary if we care _where_ pressure applies.
	// Otherwise the most important thing is just that we get NT_vertcell right at the ins.
	// Correct?


	sum.x = 0.0;
	sum.y = 0.0;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		RecalculateCentroid(pVertex); // MAINTAIN INSTEAD.

		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST)) {
			if (species == OVERALL) {
				pVertex->temp2.x = (pVertex->Elec.heat +pVertex->Ion.heat + pVertex->Neut.heat)/
							pVertex->AreaCell;
			} else {
				if (species == SPECIES_ION) 
					pVertex->temp2.x = pVertex->Ion.heat/pVertex->AreaCell;
				if (species == SPECIES_ELEC) 
					pVertex->temp2.x = pVertex->Elec.heat/pVertex->AreaCell;
				if (species == SPECIES_NEUT) 
					pVertex->temp2.x = pVertex->Neut.heat/pVertex->AreaCell;
			};
		} else {
			pVertex->temp2.x = 0.0; // scratch data for nT_overall
		};
		++pVertex;
	};

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
			
			// First create contiguous images of the 
			// vertcell centroids:

			u[0] = pTri->cornerptr[0]->centroid;
			u[1] = pTri->cornerptr[1]->centroid;
			u[2] = pTri->cornerptr[2]->centroid;

			if (pTri->periodic) {
				if (u[0].x > 0.0) u[0] = Anticlockwise*u[0];	
				if (u[1].x > 0.0) u[1] = Anticlockwise*u[1];
				if (u[2].x > 0.0) u[2] = Anticlockwise*u[2];
			};
			
			// Now create interp coefficients,
			// and decide on nT at this tri centroid.
			
			GetInterpolationCoefficients(beta,pTri->cent.x,pTri->cent.y,
				u[0],u[1],u[2]);

			// nT = NT/area
			pTri->temp_f64 =  beta[0]*pTri->cornerptr[0]->temp2.x
							+ beta[1]*pTri->cornerptr[1]->temp2.x
							+ beta[2]*pTri->cornerptr[2]->temp2.x;
		} else {
			pTri->temp_f64 = 0.0;
			// Use this later, in case of 1 vertex above ins.
		};
		
		++pTri;
	};

	// Now set for those which have 2 corners above.
	// ie, the edges of the houses.
	int iDomain, iWhich1, iWhich2, iWhichInner;
	Vertex* pVert1, *pVert2;
	real TotalNT, RemainArea, TotalArea, RemainNT;
	long tri_len, izTri[128], i, inext;
	Triangle * pTri1, *pTri2, *pTriNext;
	ConvexPolygon shard;
	real avg_nT_shard,area_shard, sumcoeffs;
	Vector2 cent1,cent2;
	real avg_nT_desired, nT_top, NT_found, avg_nT, NT_left;
	long index[128];
	real coeffs[128];
	int nIns;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if (pTri->u8domain_flag == CROSSING_INS) {
			iDomain = 0;
			if (pTri->cornerptr[0]->flags == DOMAIN_VERTEX) iDomain++;
			if (pTri->cornerptr[1]->flags == DOMAIN_VERTEX) iDomain++;
			if (pTri->cornerptr[2]->flags == DOMAIN_VERTEX) iDomain++;

			if (iDomain == 2) {
				// Now let's take in 
				// * average nT required in the 2 vertcells
				// * known nT_tri_cent values within domain
				// * assume nT = nT_avg at polygon centroids.

				// A. Take out NT and area for the shards then known.
				// B. Of the remaining NT, apportion the desired amt to
				// the middle region: {base, polygon centroids, tri cent shared}
				// C. Take out NT and area for a triangle between the polygon cents and shared cent.
				// D. Now we have a quadrilateral and desired NT. Make all the
				// base equal to our chosen value; thus determine it.
				iWhich1 = 0;
				while (pTri->cornerptr[iWhich1]->flags != DOMAIN_VERTEX) iWhich1++;
				iWhich2 = iWhich1+1;
				while (pTri->cornerptr[iWhich2]->flags != DOMAIN_VERTEX) iWhich2++;
				iWhichInner = 0;
				while (pTri->cornerptr[iWhichInner]->flags == DOMAIN_VERTEX) iWhichInner++;
				
				pVert1 = pTri->cornerptr[iWhich1];
				pVert2 = pTri->cornerptr[iWhich2];

				TotalArea = pVert1->AreaCell + pVert2->AreaCell;
				TotalNT = pVert1->temp2.x*pVert1->AreaCell + pVert2->temp2.x*pVert2->AreaCell;

				RemainArea = TotalArea;
				RemainNT = TotalNT;

				// Now subtract for shards known, pVert1:

				// . cycle through to the first ins tri then the first domain tri;

				tri_len = pVert1->GetTriIndexArray(izTri);
				i = 0;
				while ((T + izTri[i])->u8domain_flag == DOMAIN_TRIANGLE) i++;
				while ((T + izTri[i])->u8domain_flag == CROSSING_INS)
				{
					i++;
					if (i == tri_len) i = 0;
				};

				// Now this is part of the first shard

				// add shards until we reach an ins tri (unknown nT)
				do {
					inext = i+1; if (inext == tri_len) inext = 0;
					pTri1 = T + izTri[i];
					pTri2 = T + izTri[inext];
					if (pTri2->u8domain_flag == DOMAIN_TRIANGLE) {
						shard.Clear();
						cent1 = pTri1->GetContiguousCent_AssumingCentroidsSet(pVert1);
						shard.add(cent1);
						cent2 = pTri2->GetContiguousCent_AssumingCentroidsSet(pVert1);
						shard.add(cent2);
						shard.add(pVert1->centroid);
						avg_nT_shard = THIRD*(pTri1->temp_f64 + pTri2->temp_f64 + pVert1->temp2.x);
						area_shard = shard.GetArea();
						RemainArea -= area_shard;
						RemainNT -= avg_nT_shard*area_shard;

					};
					i++;
				} while (pTri2->u8domain_flag == DOMAIN_TRIANGLE);
				
				// . same for pVert2 :
				
				tri_len = pVert2->GetTriIndexArray(izTri);
				i = 0;
				while ((T + izTri[i])->u8domain_flag == DOMAIN_TRIANGLE) i++;
				while ((T + izTri[i])->u8domain_flag == CROSSING_INS)
				{
					i++;
					if (i == tri_len) i = 0;
				};
				do {
					inext = i+1; if (inext == tri_len) inext = 0;
					pTri1 = T + izTri[i];
					pTri2 = T + izTri[inext];
					if (pTri2->u8domain_flag == DOMAIN_TRIANGLE) {
						shard.Clear();
						cent1 = pTri1->GetContiguousCent_AssumingCentroidsSet(pVert2);
						shard.add(cent1);
						cent2 = pTri2->GetContiguousCent_AssumingCentroidsSet(pVert2);
						shard.add(cent2);
						shard.add(pVert2->centroid);
						avg_nT_shard = THIRD*(pTri1->temp_f64 + pTri2->temp_f64 + pVert2->temp2.x);
						area_shard = shard.GetArea();
						RemainArea -= area_shard;
						RemainNT -= avg_nT_shard*area_shard;
					};
					i++;
				} while (pTri2->u8domain_flag == DOMAIN_TRIANGLE);

				// We have a frieze-shape but we can't deal with
				// further neighbours to make it smaller.
				// Just take away a triangle that we do know about:

				pTri2 = pTri->neighbours[iWhichInner];
				avg_nT_shard = THIRD*(pTri2->temp_f64 + pVert1->temp2.x + pVert2->temp2.x);
				shard.Clear();
				if (pTri2->periodic == 0) {
					shard.add(pVert1->centroid);
					shard.add(pVert2->centroid);
					shard.add(pTri2->cent);
				} else {
					u[0] = pVert1->centroid;
					if (u[0].x > 0.0) u[0] = Anticlockwise*u[0];
					u[1] = pVert2->centroid;
					if (u[1].x > 0.0) u[1] = Anticlockwise*u[1];
					u[2] = pTri2->cent;
					if (u[2].x > 0.0) u[2] = Anticlockwise*u[2];
				};
				area_shard = shard.GetArea();
				RemainArea -= area_shard;
				RemainNT -= avg_nT_shard*area_shard;
				
				avg_nT_desired = RemainNT/RemainArea;
				// do not sketch a shape but just assume the base values
				// have to balance the centroid values:
				nT_top = 0.5*(pVert1->temp2.x + pVert2->temp2.x);
				pTri->temp_f64 = 2.0*avg_nT_desired-nT_top;	
				if (pTri->temp_f64 < 0.0) {
					printf("Error - nT < 0\n");
					getch();
				}
			}
		}; // CROSSING_INS
		++pTri;
	}

	// Now set for those which have 1 corner above, for each vertcell that contains.
	// USUALLY there is at most 1 per vertcell but there could be any number.

	// Add up known shards. This will leave some unknown ones.
	// Know what the values contribute to NT estimate--> know what lc must equal. 
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			// .Do a search: does it possess insulator triangles?
			// .If so, does it possess ones unique to this vertex alone?

			tri_len = pVertex->GetTriIndexArray(izTri);

			nIns = 0;

			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];

				if (pTri->u8domain_flag == CROSSING_INS) {
					if (
						  ((pTri->cornerptr[0]->flags == DOMAIN_VERTEX)?1:0)
						+ ((pTri->cornerptr[1]->flags == DOMAIN_VERTEX)?1:0)
						+ ((pTri->cornerptr[2]->flags == DOMAIN_VERTEX)?1:0) == 1)
					{
						index[nIns] = i;
						nIns++;
					};
				}
			}

			if (nIns > 0) {

				// part i:
				
				// .Work out what we are getting from the known shards, 
			
				memset(coeffs,0,sizeof(real)*128);

				NT_found = 0.0;
				for (i = 0; i < tri_len; i++)
				{
					inext = i+1;
					if (inext == tri_len) inext = 0;
					pTri = T + izTri[i];
					pTriNext = T + izTri[inext];

					//if ((pTri->temp_f64 != 0.0) && (pTriNext->temp_f64 != 0.0))
					{
						shard.Clear();
						shard.add(pVertex->centroid);
						shard.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
						shard.add(pTriNext->GetContiguousCent_AssumingCentroidsSet(pVertex));
						avg_nT = THIRD*(pVertex->temp2.x + pTri->temp_f64 + pTriNext->temp_f64);
						NT_found += avg_nT*shard.GetArea();
						if (pTri->temp_f64 == 0.0) {
							coeffs[i] += shard.GetArea()*THIRD;
							// amt that will be added to NT by changing this value.
						};
						if (pTriNext->temp_f64 == 0.0) {
							coeffs[inext] += shard.GetArea()*THIRD;
						};
					}
				};
				
				// Now we have a certain amount of NT left to account for.
				NT_left = pVertex->AreaCell*pVertex->temp2.x - NT_found;
				// and we have coeffs for each index that had 1 vertex above.

				// part ii:
				// Put them in a line. Alter the slope and intercept of that line
				// so that we
				// .. achieve NT
				// .. minimize the squared distance to the 2 neighbours that we just set.

				if (nIns == 1) {
					// straightforward:
					(T + izTri[index[0]])->temp_f64 = NT_left/coeffs[index[0]];
				} else {
						// we want: coeffs dot values = NT_left
						// and: to minimize the squared distance between each point and
						// its neighbour on the insulator.

						// Every run of "1 vertex above" must be bounded by "2 vertex above"
						// including this vertex.
						// Let's assume that if we pick a centre value between these
						// 2 neighbours, all the other points will take values
						// -- can be Bezier -- as we shift across, we move from the
						// line left-'centre' to 'centre'-right.
						// We aggregate coefficients and then can solve for center
						// value, then can put in values of nT for all.

						// Hold on. Does this make any difference to anything?
						// Probably not a lot? It does change the left-right pressure.


					// LEAVE IT FOR NOW: all = average
					// -------------------------------

					if (NT_left < 0.0) {
						printf("nT < 0; error\n");
						getch();
					};

					sumcoeffs = 0.0;
					for (int ii = 0; ii < nIns; ii++)
					{
						sumcoeffs += coeffs[ii];
					};
					for (int ii = 0; ii < nIns; ii++)
					{
						(T+izTri[index[ii]])->temp_f64 = NT_left/sumcoeffs;
					}
					// Set all to same value, that attains NT. Oh well.
					
				};
			};

		};
		++pVertex;
	}
	
	printf("Done nT\n");


	// That was a big deal. It had a lot of thought go into it.

	// Now going to create pressures:
	// //////////////////////////////

	// On GPU how to do : each tri collects info from 3 places, =>
	// 3 x random access; use shared mem.

	real nT[128];
	real totalvertexmass_grams;

	pVertex = X;
	for (iVertex= 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			tri_len = pVertex->GetTriIndexArray(izTri);

			// Get centroid polygon:
			cp.Clear();			
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				cp.add(pTri->GetContiguousCent_AssumingCentroidsSet(pVertex));
				nT[i] = pTri->temp_f64; 
			};
			
			// But remember heat lives in the whole "house-shaped" cell.
			// We could choose otherwise but let's not.
			// So, AreaCell routine contains house shape? 

			if (species == OVERALL) {
				totalvertexmass_grams = m_n_*pVertex->Neut.mass + m_i_*pVertex->Ion.mass + m_e_*pVertex->Elec.mass;
				pVertex->a_pressure_neut_or_overall = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/
					(totalvertexmass_grams);
			};
			if (species == SPECIES_ION) 
				pVertex->a_pressure_ion = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_i_*pVertex->Ion.mass);
			if (species == SPECIES_NEUT)
				pVertex->a_pressure_neut_or_overall = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_n_*pVertex->Neut.mass);
			if (species == SPECIES_ELEC)
				pVertex->a_pressure_elec = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_e_*pVertex->Elec.mass);

			// dividing integral of grad by integral of density.			

		} else {
			
			pVertex->a_pressure_ion.x = 0.0;
			pVertex->a_pressure_ion.y = 0.0;
			pVertex->a_pressure_elec.x = 0.0;
			pVertex->a_pressure_elec.y = 0.0;
			pVertex->a_pressure_neut_or_overall.x = 0.0;
			pVertex->a_pressure_neut_or_overall.y = 0.0;
		};

		++pVertex;
	}
}

void TriMesh::CreateMeshDisplacement_zero_future_pressure() 
{
	// Must have populated: 
	//  Vertex::AreaCell
	//  anticlockwise tri & neigh index arrays
	//  Triangle::cent

	// Creates pVertex->AdvectedPosition0

	Vector2 tri_cent, sum, u[3];
	Triangle * pTri;
	long iTri, iVertex;
	Vertex * pVertex;
	Vector3 JcrossB_contribution;
	real totalvertexmass_grams;

	// Inputs: AreaCell. pVertex->B
	

	//GetGradTeOnVertices(); // where used?
	
	Set_nT_and_Get_Pressure(OVERALL);
			
	pVertex = X;
	for (iVertex= 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->flags == DOMAIN_VERTEX)
		{
			totalvertexmass_grams = m_n_*pVertex->Neut.mass + m_i_*pVertex->Ion.mass + m_e_*pVertex->Elec.mass;
			
			JcrossB_contribution =  h*h*0.5*q_*((pVertex->Ion.mom-pVertex->Elec.mom).cross(pVertex->B))/
						(c_*totalvertexmass_grams);
			
			pVertex->AdvectedPosition0 = pVertex->pos
				
				+ h*((m_n_*pVertex->Neut.mom +m_i_*pVertex->Ion.mom + m_e_*pVertex->Elec.mom).xypart())/
				    totalvertexmass_grams;
				
				+ h*h*0.25*pVertex->a_pressure_neut_or_overall 
				
				+ JcrossB_contribution.xypart();
			
		} else {
			pVertex->AdvectedPosition0 = pVertex->pos;
		};
		
		++pVertex;
	}
	
	// Thus, we populated pVertex->AdvectedPosition0 AND pTri->temp_f64 = nT.

	

	



	//pVertex = X;
	//for (iVertex= 0; iVertex < numVertices; iVertex++)
	//{
	//	Y.CalculateCoefficients(pVertex); // get things like nu, based on present Te
	//	Y.Populate_Acceleration_Coefficients_no_pressure(H,a0);

	//	// Assumption: accel = a0 + Hv


	//	n_n = pVertex->Neut.mass/pVertex->AreaCell;
	//	v_n_k = pVertex->Neut.mom/pVertex->Neut.mass;
	//	T_n_k = pVertex->Neut.heat/pVertex->Neut.mass;

	//	n_ion = pVertex->Ion.mass/pVertex->AreaCell;
	//	v_ion_k = pVertex->Ion.mom/pVertex->Ion.mass;
	//	T_ion_k = pVertex->Ion.heat/pVertex->Ion.mass;
	//	
	//	T_e_k = pVertex->Elec.heat/pVertex->Elec.mass;

	//	v[0] = v_ion_k.x;
	//	v[1] = v_ion_k.y;
	//	v[2] = v_ion_k.z;
	//	v[3] = v_neut_k.x;
	//	v[4] = v_neut_k.y;
	//	v[5] = v_neut_k.z;

	//	memset(&(pVertex->AdvectedPosition0),&(pVertex->pos),sizeof(Vector2));
	//	ZeroMemory(&(pVertex->effect_on_overall_dis_aiTP),sizeof(Tensor2));
	//	ZeroMemory(&(pVertex->effect_on_overall_dis_anTP),sizeof(Tensor2)); // effect on overall displacement.

	//	// We will store 4 6-vectors that represent the effect of e.g. a_iTPx on v[0-5].
	//	
	//	real v_effect_a_iTPx[6], v_effect_a_iTPy[6], v_effect_a_nTPx[6], v_effect_a_nTPy[6];
	//	memset(v_effect_a_iTPx,0,6*sizeof(f64));
	//	memset(v_effect_a_iTPy,0,6*sizeof(f64));
	//	memset(v_effect_a_nTPx,0,6*sizeof(f64));
	//	memset(v_effect_a_nTPy,0,6*sizeof(f64));
	//	
	//	// Now let's be more decent: start using trapezoidal pressure effect....
	//	// ?
	//	// Leave as room for improvement...
	//			
	//	// Set up matrices:
	//	for (i = 0; i < 6; i++)
	//	{
	//		for (j = 0; j < 6; j++) 
	//			// Hsq_= multiply row i and column j of H
	//			Hsq[i][j] = H[i][0]*H[0][j] + H[i][1]*H[1][j] + H[i][2]*H[2][j]
	//				  + H[i][3]*H[3][j] + H[i][4]*H[4][j] + H[i][5]*H[5][j];
	//	};
	//	LHS.Invoke(6);
	//	for (i = 0; i < 6; i++)
	//		for (j = 0; j < 6; j++)
	//			LHS.LU[i][j] = ((i == j)?1.0:0.0) - hSub*H[i][j] + (hSub*hSub*0.5)*Hsq[i][j] ;
	//	LHS.LUdecomp();		
	//	
	//	for (i = 0; i < 6; i++)
	//	{
	//		RHS_additional[i] = hSub*a0[i] - hSub*hSub*0.5*
	//							(H[i][0]*a0[0] + 
	//							 H[i][1]*a0[1] +
	//							 H[i][2]*a0[2] +
	//							 H[i][3]*a0[3] +
	//							 H[i][4]*a0[4] +
	//							 H[i][5]*a0[5]);

	//		// Each of the pressure effect vectors evolves according to the same sort of eqn.
	//		RHS_additional_iTPx[i] = ((i == 0)?hSub:0.0) - hSub*hSub*0.5*H[i][0];
	//		RHS_additional_iTPy[i] = ((i == 1)?hSub:0.0) - hSub*hSub*0.5*H[i][1];
	//		RHS_additional_nTPx[i] = ((i == 3)?hSub:0.0) - hSub*hSub*0.5*H[i][3];
	//		RHS_additional_nTPy[i] = ((i == 4)?hSub:0.0) - hSub*hSub*0.5*H[i][4];
	//	}

	//	for (int iStep = 0; iStep < numSubsteps; iStep++)
	//	{
	//		// Evolve equation for v(1,a_TP):
	//		for (i = 0; i < 6; i++)
	//		{
	//			RHS[i] = v[i] + RHS_additional[i];

	//			RHS_va_iTPx[i] = v_effect_a_iTPx[i] + RHS_additional_iTPx[i];
	//			RHS_va_iTPy[i] = v_effect_a_iTPy[i] + RHS_additional_iTPy[i];
	//			RHS_va_nTPx[i] = v_effect_a_nTPx[i] + RHS_additional_nTPx[i];
	//			RHS_va_nTPy[i] = v_effect_a_nTPy[i] + RHS_additional_nTPy[i];
	//		}
	//		LHS.LUSolve(RHS,vnext);
	//		LHS.LUSolve(RHS_va_iTPx, v_effect_a_iTPx);
	//		LHS.LUSolve(RHS_va_iTPy, v_effect_a_iTPy);
	//		LHS.LUSolve(RHS_va_nTPx, v_effect_a_nTPx);
	//		LHS.LUSolve(RHS_va_nTPy, v_effect_a_nTPy);
	//	
	//		// It is of course just silly to be taking "Backward" instead of "trapezoidal" pressure.
	//		
	//		// Now increment the terms in the displacement equation:

	//		pVertex->AdvectedPosition0.x += 0.5*hSub*
	//			((v[0]+vnext[0])*n_ion+(v[3]+vnext[3])*n_n)
	//			/(n_ion+n_n);
	//		pVertex->AdvectedPosition0.y += 0.5*hSub*
	//			((v[1]+vnext[1])*n_ion+(v[4]+vnext[4])*n_n)
	//			/(n_ion+n_n);
	//		
	//		factor = 0.5*hSub;
	//		if (iStep < numSubsteps-1) factor = hSub;

	//		pVertex->effect_on_overall_dis_aiTP.xx += factor*(v_effect_a_iTPx[0]*n_ion
	//		  								         	+ v_effect_a_iTPx[3]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_aiTP.xy += factor*(v_effect_a_iTPy[0]*n_ion
	//											         	+ v_effect_a_iTPy[3]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_aiTP.yx += factor*(v_effect_a_iTPx[1]*n_ion
	//											         	+ v_effect_a_iTPx[4]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_aiTP.yy += factor*(v_effect_a_iTPy[1]*n_ion
	//											         	+ v_effect_a_iTPy[4]*n_n)/
	//														(n_ion+n_n);

	//		pVertex->effect_on_overall_dis_anTP.xx += factor*(v_effect_a_nTPx[0]*n_ion
	//		  								         	+ v_effect_a_nTPx[3]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_anTP.xy += factor*(v_effect_a_nTPy[0]*n_ion
	//											         	+ v_effect_a_nTPy[3]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_anTP.yx += factor*(v_effect_a_nTPx[1]*n_ion
	//											         	+ v_effect_a_nTPx[4]*n_n)/
	//														(n_ion+n_n);
	//		pVertex->effect_on_overall_dis_anTP.yy += factor*(v_effect_a_nTPy[1]*n_ion
	//											         	+ v_effect_a_nTPy[4]*n_n)/
	//														(n_ion+n_n);
	//		// Ready for next step:
	//		memcpy(v,vnext,sizeof(real)*6);
	//	};

	//	++pVertex;
	//};
}
*/

// 1. Go over the following routine.
// We understand now we are going to just want to use
// disp = h v_k_overall + h^2/2 (J x B)_k/c[ms ns] 
//        - h^2/4 [ grad[sum NT]/ sum[ms Ns] (_k + _k+1)]
//
// [sum NT] _k+1 = [sum NT]_k * (Area_old/Area_new)^(5/3)

// Not sure that is meaningful but nvm.

// 1b. Do we want to set up NT_tri with minmod etc?
// In general we have to be able to assess pressure the best we can.

// 2. Go over preparation of this eqn. We should not need to call CalculateCoefficients at all.
// 3. Handle compressive heating.
// How to be handling relative species advection compared to bulk?

// 4. Re-delaunerise: implies we can do placement calcs following Delaunay flip.
// Maybe more than that, if untangling is ever called.

// II 1. Set up Ohm's Law as before: feint ion v (E,v_e) gives unreduced equations
// and these can be then reduced to ionic and electronic Ohm's Law.
// II 2. We somehow take that forward to the solver.
// II 3. We come out with, let's say, v_e - (weighted average of v_i,v_n), set for k+1.
// III. Then Stage III we do evolution with this (v_e - v) given. Or, apply Ohm's Law all over again - take your pick.

/*
void TriMesh::SolveForAdvectedPositions(TriMesh * pDestMesh)  // populate advected position data for each vertex using AdvectedPosition0 and Pressure_effect.
{
	
	// Approach to solving:
	// Not sure about Jacobi since we can imagine that one point alone would move very slowly given that it soon feels pressure back from its surroundings.
	// Within reason / the right circumstances, we actually believe that just iterating will do it: pressure pushes us towards where 
	// the equilibrium lies.
	// But look in terms of the eqm of  xdot = -x + (xk + d0(h)) + F(hh) a
	// and using xdotdot we can have a second-order step for that system;
	// if something is going haywire or moving too near its surrounding polygon, we slow down the system trajectory timestep.
	// We have to calculate both a(x) and for a second-order step, a-dot due to xdot. 
	
	real area_original;
	real htraj = 0.5; 
	static real const MAXhtraj = 0.5;
	real guess_h, value_new, twoarea_new, twoarea, area_roc_over_area, factor;
	Vector2 temp2, to_existing;
	real compare;
	Vertex * pVertex, *pOther;
	Vector2 acceleration, momrate;
	Triangle * pTri1,* pTri2, *pTri;
	int i, inext, numDomain, iWhich1, iWhich2;
	Vector2 u[3], U[3], ROC[3];
	Vector2 ROCcc1, cc1, ROCcc2, cc2, ROCmomrate, ROC_accel, ROCu_ins;
	real area, ROCarea, ROCvalue, Average_nT, ROCAverage_nT, value;
	long iTri, iVertex;
	Vector2 u_ins, rhat;
	int iWhich;
	bool broken, not_converged_enough;
	long tri_len, izTri[128];

	static real const FIVETHIRDS = THIRD*5.0;
	
	long iIterationsConvergence = 0;

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->AdvectedPosition = pVertex->pos;
		++pVertex;
	};
		
	// Loop to solve for positions:
	do
	{
		// Get putative areas and nT_cell:
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			u[0] = pTri->cornerptr[0]->AdvectedPosition;
			u[1] = pTri->cornerptr[1]->AdvectedPosition;
			u[2] = pTri->cornerptr[2]->AdvectedPosition;
			if (pTri->periodic > 0) {
				// Map to left - we may deal with precomputed centre, on ins?
				if (pTri->periodic == 1) {
					i = pTri->GetLeftmostIndex();
					if (i != 0) u[0] = Anticlockwise*u[0];
					if (i != 1) u[1] = Anticlockwise*u[1];
					if (i != 2) u[2] = Anticlockwise*u[2];
				} else {
					i = pTri->GetRightmostIndex();
					u[i] = Anticlockwise*u[i];
				};
			};
			if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
				// Note problem: a triangle can stop or start being periodic as part of the planned advection.
				// What to do about it?
				// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.
				area = fabs(	0.5*(	u[0].x*u[1].y - u[1].x*u[0].y
									+	u[1].x*u[2].y - u[2].x*u[1].y
									+	u[2].x*u[0].y - u[0].x*u[2].y	));
				// Does need a branch or fabs because we do not know which way is clockwise.
				// heat/orig (Area now/ orig)^(-5/3) = heat (Area_k ^2/3 / Area_now ^5/3)
				
				pTri->nT = pTri->temp_f64*pow((pTri->area/area),FIVETHIRDS);
				// . temp_f64 should store original sum of n_s T_s
				// . pow is expensive - so we did this first for each triangle.
				
			} else {
				
				if (pTri->u8domain_flag == CROSSING_INS) {
					// In this case split out 2 further cases.
					// Bear in mind nothing ever crosses the insulator.
					// If there is 1 point above the ins, use the triangle formed with the ins.
					// If there are 2 points above the ins, use a triangle formed with 2 corners and "centroid".
					
					area_original = pTri->GetDomainIntersectionArea(true, u); // recalculate every time 
					area = pTri->GetDomainIntersectionArea(false, u);
					
					pTri->nT = pTri->temp_f64*pow((area_original/area),FIVETHIRDS);					
				} else {
					// not domain tri:
					pTri->nT = 0.0;
				};
			};
			++pTri;			
		};
		
		real RSS = 0.0;
		
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			// Next value is found from:
			// _________________________

			// 1. calculate acceleration at system present position

			momrate.x = 0.0; momrate.y = 0.0;
			
			if (pVertex->flags == DOMAIN_VERTEX)
			{		
				tri_len = pVertex->GetTriIndexArray(izTri);

				for (i = 0; i < tri_len; i++)
				{
					inext = i+1; if (inext == tri_len) inext = 0;
					pTri1 = T + izTri[i];
					pTri2 = T + izTri[inext];
					cc1 = pTri1->GetContiguousCent_AssumingCentroidsSet(pVertex);
					cc2 = pTri2->GetContiguousCent_AssumingCentroidsSet(pVertex);
					
					Average_nT = 0.5*(pTri1->nT + pTri2->nT);// had to precompute these - expensive.
					momrate.x -= (cc2.y-cc1.y)*Average_nT;
					momrate.y -= (cc1.x-cc2.x)*Average_nT; 
					// To get integral [-grad nT]
				}; 
			//	acceleration = momrate / pVertex->Polygon_mass; // Polygonmass = 1/3 total mass of cells, in our number units * m_species

				// That's the rub:
				// acceleration = momrate / pVertex->Polygon_mass;
				// = -grad[sum_s ns Ts]/(sum_s m_s Ns).
				
				acceleration = momrate/(pVertex->Ion.mass*m_i_ + pVertex->Neut.mass*m_n_ + pVertex->Elec.mass*m_e_);
				
				// 2. calculate position rate of change: xdot
				// ---------------------------------------------------------------
				// seek eqm of  xdot = -x + (xk + d0(h)) + F(hh) a :
				pVertex->xdot = 
					(pVertex->AdvectedPosition0 - pVertex->AdvectedPosition)
					//+ pVertex->Pressure_a_effect_dis[species]*acceleration;
					+ (h*h*0.25)*acceleration;
				// Note that we should subsume (h*h*0.25)*(-grad(nT_all)_k/ NT_all) into AdvectedPosition0.
			} else {
				// vertex that is not to move, whether outermost or inner:
				pVertex->xdot.x = 0.0; pVertex->xdot.y = 0.0;
			}
			RSS += pVertex->xdot.dot(pVertex->xdot);
			++pVertex;
		};

		real L2 = sqrt(RSS/(real)numVertices); 
		printf("\nL2 of residual: %1.12E \n",L2);
		
		// Now want to get xdotdot.

		// 3. First each cell calculates how fast its new area is changing according to xdot:
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			
			// Calculate rate of change of area and hence, ROC_nT_ion:
			u[0] = pTri->cornerptr[0]->AdvectedPosition;
			ROC[0] = pTri->cornerptr[0]->xdot;
			u[1] = pTri->cornerptr[1]->AdvectedPosition;
			ROC[1] = pTri->cornerptr[1]->xdot;
			u[2] = pTri->cornerptr[2]->AdvectedPosition;
			ROC[2] = pTri->cornerptr[2]->xdot;
			if (pTri->periodic > 0) {
				if (pTri->periodic == 1) {
					// rotate (original) leftmost point to right	
					i = pTri->GetLeftmostIndex();
					u[i] = Clockwise*u[i];
					ROC[i] = Clockwise*ROC[i]; // from the point of view of this per triangle, how it's moving
				} else {
					i = pTri->GetRightmostIndex(); // wrap the unwrapped point:
					u[i] = Anticlockwise*u[i];
					ROC[i] = Anticlockwise*ROC[i];
				};
			};


			if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {

				// Note problem: a triangle can stop or start being periodic as part of the planned advection.
				// What to do about it?
				// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.

				value = 0.5*( u[0].x*u[1].y - u[1].x*u[0].y
							+ u[1].x*u[2].y - u[2].x*u[1].y
							+ u[2].x*u[0].y - u[0].x*u[2].y);

				ROCvalue = 0.5*( 
					ROC[0].x*u[1].y + u[0].x*ROC[1].y - ROC[1].x*u[0].y - u[1].x*ROC[0].y
					+ ROC[1].x*u[2].y + u[1].x*ROC[2].y - ROC[2].x*u[1].y - u[2].x*ROC[1].y
					+ ROC[2].x*u[0].y + u[2].x*ROC[0].y - ROC[0].x*u[2].y - u[0].x*ROC[2].y);

				if (value > 0) {
					area = value;
					ROCarea = ROCvalue;
				} else {
					area = -value;
					ROCarea = -ROCvalue; 
				};
			// Note that change of sign compared to initial during a move is unexpected --
			// that indicates a triangle was flipped, and
			// we should have rejected any such attempted move and never got here.
				
				//pTri->nT_ion = pTri->ion.heat*pow(pTri->area/area),FIVETHIRDS)/pTri->area;
				
				pTri->ROC_nT = ROCarea*(-FIVETHIRDS)*pTri->nT / area; // f '(g(x))g'(x)
			} else {
				if (pTri->u8domain_flag == CROSSING_INS) {					
				//	area_original = pTri->GetDomainIntersectionArea(true, u); // recalculate every time 
					area = pTri->GetDomainIntersectionArea(false, u);
				//	pTri->nT = pTri->temp_f64*pow((area_original/area),FIVETHIRDS);					
					
					// What is ROCarea?
					// There may be 2 domain vertices moving, or only 1.
					
					numDomain = ((pTri->cornerptr[0]->flags == DOMAIN_VERTEX)?1:0) +
								((pTri->cornerptr[1]->flags == DOMAIN_VERTEX)?1:0) + 
								((pTri->cornerptr[2]->flags == DOMAIN_VERTEX)?1:0);
										
					if (numDomain == 1) {
						iWhich = 0; while (pTri->cornerptr[iWhich]->flags != DOMAIN_VERTEX) iWhich++;
						ROCarea = pTri->GetDomainIntersectionAreaROC(u,iWhich,ROC[iWhich]);

					} else {
						// debug check:
						if (numDomain != 2) {printf("error 102828\n");getch();}

						iWhich1 = 0; while (pTri->cornerptr[iWhich1]->flags != DOMAIN_VERTEX) iWhich1++;
						iWhich2 = iWhich1+1; while (pTri->cornerptr[iWhich2]->flags != DOMAIN_VERTEX) iWhich2++;
						ROCarea = pTri->GetDomainIntersectionAreaROC(u,iWhich1,ROC[iWhich1])
								+ pTri->GetDomainIntersectionAreaROC(u,iWhich2,ROC[iWhich2]);
					};					
					
					pTri->ROC_nT = ROCarea*(-FIVETHIRDS)*pTri->nT / area; // f '(g(x))g'(x)
				} else {
					// out of domain
					pTri->ROC_nT = 0.0;
				};
			};
			++pTri;
		};
		
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			// 4. ROC acceleration:

			// We have to know the combined effect on pressure here from the effects of moving all of, this point and all the neighbours
			// Area is changing, but also the centroid coordinates are changing.
						
			ROCmomrate.x = 0.0; ROCmomrate.y = 0.0;

			if (pVertex->flags == DOMAIN_VERTEX)
			{
				tri_len = pVertex->GetTriIndexArray(izTri);
				for (i = 0; i < tri_len; i++)
				{
					inext = i+1; if (inext == tri_len) inext = 0;
					pTri1 = T + izTri[i];
					pTri2 = T + izTri[inext];
					cc1 = pTri1->GetContiguousCent_AssumingCentroidsSet(pVertex);
					cc2 = pTri2->GetContiguousCent_AssumingCentroidsSet(pVertex);
					
					ROC[0] = pTri1->cornerptr[0]->xdot;
					ROC[1] = pTri1->cornerptr[1]->xdot;
					ROC[2] = pTri1->cornerptr[2]->xdot;
					if (pTri1->periodic > 0)
					{
						// important that it be relative to our vertex where acceleration is to be found! ..
						// wrapping status of corners is still per the original data
						if (pTri1->periodic == 1)
						{
							iWhich = pTri1->GetLeftmostIndex();
							if (pVertex->pos.x > 0.0) {
								// bring that periodic one back
								ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri1->periodic == 2
							iWhich = pTri1->GetRightmostIndex();
							if (pVertex->pos.x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc1 = THIRD*(ROC[0] + ROC[1] + ROC[2]);
		
					// same for pTri2 .... 
					ROC[0] = pTri2->cornerptr[0]->xdot;
					ROC[1] = pTri2->cornerptr[1]->xdot;
					ROC[2] = pTri2->cornerptr[2]->xdot;					
					if (pTri2->periodic > 0)
					{
						if (pTri2->periodic == 1)
						{
							iWhich = pTri2->GetLeftmostIndex();
							if (pVertex->pos.x > 0.0) {
								// bring that periodic one back
								ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri2->periodic == 2
							iWhich = pTri2->GetRightmostIndex();
							if (pVertex->pos.x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc2 = THIRD*(ROC[0] + ROC[1] + ROC[2]);

					// ROC_nT_ion calculated first to save on pow.
					ROCAverage_nT = 0.5*(pTri1->ROC_nT + pTri2->ROC_nT);
					Average_nT = 0.5*(pTri1->nT + pTri2->nT);
						
					//momrate.x -= (cc2.y-cc1.y)*Average_nT_ion/m_i_;
					ROCmomrate.x -= (ROCcc2.y-ROCcc1.y)*Average_nT
											+ (cc2.y-cc1.y)*ROCAverage_nT;
					ROCmomrate.y -= (ROCcc1.x-ROCcc2.x)*Average_nT
											+ (cc1.x-cc2.x)*ROCAverage_nT;
				};
				ROC_accel = ROCmomrate / (pVertex->Ion.mass*m_i_+pVertex->Neut.mass*m_n_+pVertex->Elec.mass*m_e_);
			
				// 5. ROC xdot = xdotdot:
				//pVertex->xdot = (pVertex->AdvectedPosition0 - pVertex->AdvectedPosition)
				//					+ h*h*0.25*acceleration;
				
				pVertex->xdotdot =  (h*h*0.25) * ROC_accel - pVertex->xdot;
			
				// 6. Now set putative coordinates:
				pVertex->temp2 = pVertex->AdvectedPosition + htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
				
				if ((pVertex->temp2.x-pVertex->pos.x > 2.0e-4) || (pVertex->temp2.x-pVertex->pos.x < -2.0e-4)) {
					i = i;
				};
				if ((pVertex->temp2.x > 4.0) || (pVertex->temp2.x < -4.0)) {
					i = i;
				};
			
			} else {

				// not DOMAIN_VERTEX

				pVertex->xdotdot.x = 0.0; pVertex->xdotdot.y = 0.0;
				pVertex->temp2 = pVertex->pos;

			};
				
			// Given that we estimate rate of change of accel, can we estimate there is a point where
			// the equation is actually achieved??
			// Probably not since accel probably heads off to the side as we progress.
			
			// xdot gets small as we get near but what happens to xdotdot?
			// xdot is small so area change is small and so xdotdot is also small?
				
			++pVertex;
		};	
		

		// Now test if that step failed: did something get too near to its surroundings too fast, for instance?
		// _________________________________________________________________________________________
		long iTriWorst, iType;
		real shoelace, shoe_new,guesshere;
		int broken_iterations = 0;
		do 
		{
			broken = false;
			guess_h = htraj;
			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				if (pTri->u8domain_flag == DOMAIN_TRIANGLE) {
					
					// Test shoelace every triangle. See if flipped and/or if area has _diminished_ by too great a factor.
					u[0] = pTri->cornerptr[0]->AdvectedPosition;
					u[1] = pTri->cornerptr[1]->AdvectedPosition;
					u[2] = pTri->cornerptr[2]->AdvectedPosition;
					
					U[0] = pTri->cornerptr[0]->temp2;
					U[1] = pTri->cornerptr[1]->temp2;
					U[2] = pTri->cornerptr[2]->temp2;
					
					if (pTri->periodic > 0) {
						if (pTri->periodic == 1) {
							// rotate (original) leftmost point to right	
							i = pTri->GetLeftmostIndex();
							u[i] = Clockwise*u[i];
							U[i] = Clockwise*U[i];
						} else {
							i = pTri->GetRightmostIndex();
							u[i] = Anticlockwise*u[i];
							U[i] = Anticlockwise*U[i];
						};
					};
					// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.
					value = (	u[0].x*u[1].y - u[1].x*u[0].y
										+	u[1].x*u[2].y - u[2].x*u[1].y
										+	u[2].x*u[0].y - u[0].x*u[2].y	);
					value_new = (U[0].x*U[1].y - U[1].x*U[0].y
										+	U[1].x*U[2].y - U[2].x*U[1].y
										+	U[2].x*U[0].y - U[0].x*U[2].y	);
					if (value_new*value < 0.0) {
						broken = true;
						if (htraj*0.2 < guess_h) {
							guess_h = htraj*0.2;
							iTriWorst = iTri;
							iType = 1;
							shoelace = value;
							shoe_new = value_new;
						}
					} else {
						twoarea = fabs(value);
						twoarea_new = fabs(value_new);
						if (twoarea_new < 0.4*twoarea) {
							broken = true;
							guesshere = htraj*(twoarea_new/(0.4*twoarea));
							if (guesshere < guess_h) {
								guess_h = guesshere;
								iTriWorst = iTri;
								iType = 2;
								shoelace = twoarea;
								shoe_new = twoarea_new;
							}
						};
					};

				} else {

					if (pTri->u8domain_flag == CROSSING_INS) {
						// Just check that any vertex has not crossed insulator.

						u[0] = pTri->cornerptr[0]->AdvectedPosition;
						u[1] = pTri->cornerptr[1]->AdvectedPosition;
						u[2] = pTri->cornerptr[2]->AdvectedPosition;
					
						U[0] = pTri->cornerptr[0]->temp2;
						U[1] = pTri->cornerptr[1]->temp2;
						U[2] = pTri->cornerptr[2]->temp2;
					
						if ((in_domain(u[0]) && !in_domain(U[0])) ||
							(in_domain(u[1]) && !in_domain(U[1])) ||
							(in_domain(u[2]) && !in_domain(U[2])))
						{
							broken = true;
							guesshere = htraj*0.5;
							if (guesshere < guess_h) {
								guess_h = guesshere;
								iTriWorst = iTri;
								iType = 3;
								shoelace = 0.0;
								shoe_new = 0.0;
							};
						};
					};
				};
				++pTri;			
			};


			if (broken) {
				printf("Iteration %d shortening: htraj %1.5E ; guess_h %1.5E\n",broken_iterations,
htraj,guess_h);
				printf("broken because tri %d, type %d: shoelace %1.8E %1.8E\n",iTriWorst,
					iType,shoelace, shoe_new);
				
				pTri = T + iTriWorst;
				for (i = 0; i < 3; i++)
				{
					pVertex = pTri->cornerptr[i];
					printf("%d xy %1.5E %1.5E adv %1.5E %1.5E temp2 %1.5E %1.5E | \n",
						pVertex-X, pVertex->pos.x, pVertex->pos.y, pVertex->AdvectedPosition.x,
						pVertex->AdvectedPosition.y, pVertex->temp2.x,pVertex->temp2.y);
				}
				
				guess_h *= 0.99;
		//		ratio = guess_h/htraj;
				htraj = guess_h;
				printf("htraj= %1.4E ",htraj);
				// Now, tween halfway back to the existing system position ?
				// No: because we used a quadratic not linear model of how position evolves !!
				// Instead we have xdotdot stored for every vertex and we use it.

				//	pVertex->temp2 = pVertex->AdvectedPosition[species] + htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
						
				real maxxdot = 0.0;
				real maxxdotdot = 0.0;
				long imax1, imax2;
				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					if (fabs(pVertex->xdot.x) > maxxdot) {
						maxxdot = fabs(pVertex->xdot.x);
						imax1 = iVertex;
					};
					if (fabs(pVertex->xdotdot.x) > maxxdotdot) {
						maxxdotdot = fabs(pVertex->xdotdot.x);
						imax2 = iVertex;
					};
					++pVertex;
				}
				printf("Max xdot: iVertex %d  %1.10E xdotdot: iVertex %d  %1.10E \n",
						imax1, maxxdot, imax2, maxxdotdot);
				printf("Max move: %1.6E \n",htraj*maxxdot+0.5*maxxdotdot*htraj*htraj);
				
				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					if (pVertex->flags == DOMAIN_VERTEX)
					{
						pVertex->temp2 = pVertex->AdvectedPosition
									+ htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
					} else {
						pVertex->temp2 = pVertex->AdvectedPosition;
					};
					++pVertex;
				};	

				broken_iterations++;
			};
		} while (broken); 
				
		// If no problems and htraj < some max, increase htraj back to get us to our solution faster ..
		//  set attempt flag : don't attempt again if it fails but shorter step works! know if we are
		//  heading up or down of timestep.
		
		if ((broken_iterations == 0) && (htraj < MAXhtraj))
		{
			htraj *= 1.6;
			if (htraj > MAXhtraj) htraj = MAXhtraj;
		};
			
		// Now accept temporary values ... 
		// ______________________________
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->AdvectedPosition = pVertex->temp2;
			++pVertex;
		};	
		printf("====================\nused htraj = %1.5E\n==============\n",htraj);

		// Test for convergence: is everything fairly close to converged?
		// _________________________________________________________
		
		// Planned xdot should all have been small compared to the move from pVertex->x,y. 
		// Let's say we should go 99.9% of the way?
		// Also small compared to dist to neighbour - for sure
		// Is that enough on its own?
		// Preferably would say that xdot stayed small this move!
		// Can we say smth about xdotdot ??
		// That xdot is not going to explode in magnitude in time 1 say ?
		
		if (broken_iterations == 0)
		{
			not_converged_enough = false;

			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				// Test whether "rate of area change" - the amt it is modelled linearly as going to change
				// during the progress to the implied system position - is > a fraction of new area.

				// pTri->ROC_nT_ion = ROCarea*(-FIVETHIRDS)*pTri->nT_ion / area; // f '(g(x))g'(x)
				if (pTri->u8domain_flag == DOMAIN_TRIANGLE)
				{
					area_roc_over_area = fabs(pTri->ROC_nT/(FIVETHIRDS*pTri->nT));
					// ?
					if (area_roc_over_area > 0.01) {
						not_converged_enough = true;
						printf("area not cvgd at tri %d \n",iTri);
						break;
					};
				};
				++pTri;
			};
			if (not_converged_enough == false)
			{
				pVertex = Xdomain;
				for (iVertex = 0; iVertex < numDomainVertices; iVertex++)
				{
					//xdot is the distance that was seen towards the implied target
					compare = max(pVertex->xdot.modulus(),(pVertex->xdot+pVertex->xdotdot).modulus());
					//neighbour distance at new position is what counts for that:

					//neighdist = 0.0;
					//for (i = 0; i < pVertex->neighbours.len; i++)
					//{
					//	pNeigh = X + pVertex->neighbours.ptr[i];
					//	dist = GetPossiblyPeriodicDist(pVertex->temp2,pNeigh->temp2);
					//	neighdist = max(neighdist,dist);
					//}; // that is super slow.
					//// faster way: take sqrt(area) of each neighbouring triangle, stored. ?

					// Better one: we worked out ROCarea. Let's demand |ROCarea| < 0.01*area.

					to_existing = pVertex->AdvectedPosition - pVertex->pos;
										
					// note that if it's stuck to the bottom, to_existing could be near zero. So then
					// we are testing whether it tried to move by 5e-9 cm. That is pretty small.

					if (compare > 5.0e-9 + 0.001*to_existing.modulus())
					{
						not_converged_enough = true;
						printf("xdot not cvgd at iVertex %d compare %1.8E \n",iVertex+(Xdomain-X), compare);
						break;
					};
					++pVertex;
				};				
			};
		};

		iIterationsConvergence++;
	} while (not_converged_enough);
	
	printf("SolveForAdvectedPositions converged in %d iterations.\n",iIterationsConvergence);
	// At least having got these positions, there is nothing further to do before placing cells on to the new bulk mesh.

	// Now affect dest mesh:
	Vertex * pVertDest = pDestMesh->X;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertDest->ApplDisp = pVertex->AdvectedPosition-pVertex->pos;
		pVertDest->pos = pVertex->AdvectedPosition;
		// Transfer all triangles, lists across as they are:
		pVertDest->CopyLists(pVertex);
		++pVertDest;
		++pVertex;
	};

	pTri = T;
	Triangle * pTriDest = pDestMesh->T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// The only things to set:
		// Vertex * cornerptr[3];    // keep this way for ease of continuity.
		// Triangle * neighbours[3]; // do we want this? let's keep.
		// int periodic; // number of points that are Clockwise wrapped relative to the others
		// unsigned char u8domain_flag;  

		pTriDest->cornerptr[0] = pDestMesh->X+(pTri->cornerptr[0]-X);
		pTriDest->cornerptr[1] = pDestMesh->X+(pTri->cornerptr[1]-X);
		pTriDest->cornerptr[2] = pDestMesh->X+(pTri->cornerptr[2]-X);
		pTriDest->neighbours[0] = pDestMesh->T + (pTri->neighbours[0]-T);
		pTriDest->neighbours[1] = pDestMesh->T + (pTri->neighbours[1]-T);
		pTriDest->neighbours[2] = pDestMesh->T + (pTri->neighbours[2]-T);
		pTriDest->periodic = pTri->periodic;
		pTriDest->u8domain_flag = pTri->u8domain_flag;
		// Only a Delaunay flip can change tri flag.

		++pTriDest;
		++pTri;
	};
	// Now transfer data across and apply compressive heating:

	pDestMesh->Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	// This can work on unwrapped points as long as periodic flag on tri has
	// not been disturbed.
		
	pVertDest = pDestMesh->X;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		factor = pow(pVertex->AreaCell/pVertDest->AreaCell,2.0/3.0);
		// Alternative: can we compute through estimated energy conservation?

		memcpy(&(pVertDest->Ion),&(pVertex->Ion),sizeof(macroscopic));
		memcpy(&(pVertDest->Neut),&(pVertex->Neut),sizeof(macroscopic));
		memcpy(&(pVertDest->Elec),&(pVertex->Elec),sizeof(macroscopic));
	
		pVertDest->Ion.heat = pVertex->Ion.heat*factor;
		pVertDest->Elec.heat = pVertex->Elec.heat*factor;
		pVertDest->Neut.heat = pVertex->Neut.heat*factor;

		++pVertDest;
		++pVertex;
	};
	// sending to unwrapped; if we wrapped first we'd have to somehow know
	// what needed to be rotated.

	// NOW WRAP AROUND PBC inc rotate Nv and sort out tri periodic flag:

	pVertex = pDestMesh->X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->pos.x/pVertex->pos.y > GRADIENT_X_PER_Y) {
			pVertex->pos = Anticlockwise*pVertex->pos;
			pVertex->centroid = Anticlockwise*pVertex->centroid; // maintains?
			pVertex->Ion.mom = Anticlockwise3*pVertex->Ion.mom;
			pVertex->Elec.mom = Anticlockwise3*pVertex->Elec.mom;
			pVertex->Neut.mom = Anticlockwise3*pVertex->Neut.mom;
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				pTri->IncrementPeriodic(); // crossed to right of PB
			};
		};
		if (pVertex->pos.x/pVertex->pos.y < -GRADIENT_X_PER_Y) {
			pVertex->pos = Clockwise*pVertex->pos;
			pVertex->centroid = Clockwise*pVertex->centroid; // maintains?
			pVertex->Ion.mom = Clockwise3*pVertex->Ion.mom;
			pVertex->Elec.mom = Clockwise3*pVertex->Elec.mom;
			pVertex->Neut.mom = Clockwise3*pVertex->Neut.mom;
			tri_len = pVertex->GetTriIndexArray(izTri);
			for (i = 0; i < tri_len; i++)
			{
				pTri = T + izTri[i];
				pTri->DecrementPeriodic(); // periodic is number over to right
			};
		};
		++pVertex;
	};

	// That ruined tri centroids, if they existed.

    pDestMesh->Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
};
*/
// #include "solver.cpp"

