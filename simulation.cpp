
#include "mesh.h"
#include "globals.h"
#include "headers.h"
#include "FFxtubes.h"
#include "cppconst.h"

real GlobalIzElasticity;



bool inline in_domain(Vector2 u)
{
	return (u.x*u.x+u.y*u.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER);
}
void TriMesh::Advance(TriMesh * pDestMesh)
{

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

real inline TriMesh::GetIzPrescribed(real const t)
{
	
	real Iz = -PEAKCURRENT_STATCOULOMB * sin ((t + ZCURRENTBASETIME) * PIOVERPEAKTIME );

	printf("\nGetIzPrescribed called with t+ZCURRENTBASETIME = %1.5E : %1.5E\n", t + ZCURRENTBASETIME, Iz);
	getch();

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

real inline Get_lnLambda(real n_e,real T_e)
{
	real lnLambda, factor, lnLambda_sq, lnLambda1, lnLambda2;

	static real const one_over_kB = 1.0/kB;
	
	real Te_eV = T_e*one_over_kB;
	real Te_eV2 = Te_eV*Te_eV;
	real Te_eV3 = Te_eV*Te_eV2;

	if (n_e*Te_eV3 > 0.0) {
		
		lnLambda1 = 23.0 - 0.5*log(n_e/Te_eV3);
		lnLambda2 = 24.0 - 0.5*log(n_e/Te_eV2);
		// smooth between the two:
		factor = 2.0*fabs(Te_eV-10.0)*(Te_eV-10.0)/(1.0+4.0*(Te_eV-10.0)*(Te_eV-10.0));
		lnLambda = lnLambda1*(0.5-factor)+lnLambda2*(0.5+factor);
		
		// floor at 2 just in case, but it should not get near:
		lnLambda_sq = lnLambda*lnLambda;
		factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
		lnLambda += 2.0/factor;

		// Golant p.40 warns that it becomes invalid when an electron gyroradius is less than a Debye radius. That is something to worry about if  B/400 > n^1/2 , so looks not a big concern.

		// There is also a quantum ceiling. It will not be anywhere near. At n=1e20, 0.5eV, the ceiling is only down to 29; it requires cold dense conditions to apply.

	} else {
		lnLambda = 20.0;
	};
	//if (GlobalDebugRecordIndicator)
	//	Globaldebugdata.lnLambda = lnLambda;
	return lnLambda;
}		


real inline Get_lnLambda_ion(real n_ion,real T_ion)
{
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	
	real factor, lnLambda_sq;

	real Tion_eV3 = T_ion*T_ion*T_ion*one_over_kB_cubed;
	
	real lnLambda = 23.0 - 0.5*log(n_ion/Tion_eV3);
	
	// floor at 2:
	lnLambda_sq = lnLambda*lnLambda;
	factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
	lnLambda += 2.0/factor;

	return lnLambda;
}		
real Estimate_Neutral_Neutral_Viscosity_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_viscosity_nn[9];
	if (T < cross_T_vals[0]) return cross_s_vals_viscosity_nn[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);
	return ppn*cross_s_vals_viscosity_nn[i] + (1.0-ppn)*cross_s_vals_viscosity_nn[i-1];
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
	if (T < cross_T_vals[0]){
		*p_sigma_in_MT = cross_s_vals_momtrans_ni[0];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni[0];
		return;
	}
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);

	*p_sigma_in_MT = ppn*cross_s_vals_momtrans_ni[i] + (1.0-ppn)*cross_s_vals_momtrans_ni[i-1];
	*p_sigma_in_visc = ppn*cross_s_vals_viscosity_ni[i] + (1.0-ppn)*cross_s_vals_viscosity_ni[i-1];
	return;
}

real Estimate_Ion_Neutral_MomentumTransfer_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_momtrans_ni[9];
	if (T < cross_T_vals[0]) return cross_s_vals_momtrans_ni[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);
	return ppn*cross_s_vals_momtrans_ni[i] + (1.0-ppn)*cross_s_vals_momtrans_ni[i-1];
}
void TriMesh::InterpolateAFrom(TriMesh * pSrcMesh)
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
			pTri = pSrcMesh->ReturnPointerToTriangleContainingPoint(
				pSrcMesh->T + pVertSrc->GiveMeAnIndex(),
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
}


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
		pTri->RecalculateCentroid();   // THIS SHOULD BE UNNECESSARY
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
	}
}

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
//		static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
//		static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
//		static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
//		static real const kB_to_3halves = sqrt(kB)*kB;
//		static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
//		static real const over_sqrt_m_e = 1.0/sqrt(m_e);
//		static real const qoverMc = q/(m_ion*c);
//		static real const qovermc = q/(m_e*c);
//		static real const NU_EI_FACTOR = 1.0/(3.44e5);
//		static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
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
//		ionneut_thermal = sqrt(T_ion/m_ion+T_n/m_n); // hopefully not sqrt(0)
//		electron_thermal = sqrt_Te*over_sqrt_m_e; // possibly == 0
//
//		lnLambda = Get_lnLambda(n_i,T_e); // anything strange in there?
//
//		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &s_in_MT, &s_in_visc);
//		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&s_en_MT,&s_en_visc);
//		// To use combined temperature looks to be more intelligent -- rel temp GZSB(6.55) for ion, neutral at least.
//		
//		if (T_e != 0.0) {
//			nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
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
//		heat_transfer_rate_in = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
//										*nu_in_MT; // ratio nu_in/nu_ni = n_n/n_i
//		heat_transfer_rate_ni = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
//										*nu_ni_MT;
//		heat_transfer_rate_ne = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
//										*nu_ne_MT;
//		heat_transfer_rate_en = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
//										*nu_en_MT;
//		heat_transfer_rate_ei = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
//										*nu_eiBar;
//		heat_transfer_rate_ie = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
//										*nu_ieBar;
//		
//		// OK that bit is clear and as expected.
//		// So what is the difference when we transfer NT ?
//
//		// (n_n/n_i) transfer_rate_ni = transfer_rate_in
//		
//		omega_ce = qovermc*pTri->B;
//		omega_ci = qoverMc*pTri->B; // note: if ion acceleration stage, we could if we wanted work out B at k+1 first.
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
//			nu_ieBar*(m_e/m_i)*(ID3x3-0.9*Ratio_times_Upsilon_eHeart);
//		// multiply by (v_e-v_i) for ions
//
//		Rie_thermal_force_matrix = 
//			((1.5/m_i)*(nu_ieBar/nu_eHeart)*Upsilon_nu_eHeart);
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
//			a_ion_pressure = (m_ion*a_ion_pressure + m_e*a_elec_pressure)/(m_ion + m_e);
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
////		real chi = (m_n/(m_e+m_n))*this->nu_en_MT
////			+ (1.0-0.9*this->Ratio_times_Upsilon_eHeart.zz)*this->nu_eiBar;
////
////		SimpleOhms_vez0 = (-qoverm*StoredEz 
////			-(1.5/m_e)*((this->Ratio_times_Upsilon_eHeart*pTri->GradTe).z)
////			)/chi; 
////
////		Ohms_vez0 = (
////			-(1.5/m_e)*((this->Ratio_times_Upsilon_eHeart*pTri->GradTe).z)
////			)/chi; 
////		// Where is thermal pressure force? Doesn't exist in z dimension.
////
////		Ohms_sigma = -qoverm/chi;
////
////		SimpleOhms_beta_ion.x = -this->omega_ce.y/chi;
////		SimpleOhms_beta_ion.y = this->omega_ce.x/chi;
////		SimpleOhms_beta_ion.z = ((1.0-0.9*this->Ratio_times_Upsilon_eHeart.zz)*this->nu_eiBar)/chi;
////
////		SimpleOhms_beta_neutz = (m_n/(m_e+m_n))*this->nu_en_MT/chi;
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
//		 a0[2] += this->StoredEz*qoverM; // StoredEz: populated or not?
//		
//		 // magnetic Lorentz for ions:
//		// a_ion.z -= omega_ci.x*v_ion.y-omega_ci.y*v_ion.x;
//		 H[2][0] += omega_ci.y;
//		 H[2][1] -= omega_ci.x;
//
//		// species combining gives:
//		//a_ion.x -= (m_ion/(m_e+m_ion))*
//		//				omega_ci.y*(v_ion.z-v_e.z); 
//		 // v_e is something we are passed.
//		//a_ion.y += (m_ion/(m_e+m_ion))*
//		//				omega_ci.x*(v_ion.z-v_e.z);
//		 H[0][2] -= (m_i/(m_e+m_i))*omega_ci.y;
//		 H[1][2] += (m_i/(m_e+m_i))*omega_ci.x;
//
//		 //a0[0] += (m_i/(m_e+m_i))*omega_ci.y*v_e_z;
//		 factor = (m_i/(m_e+m_i))*omega_ci.y;
//		 a0[0] += factor*this->SimpleOhms_vez0;
//		 H[0][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[0][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[0][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[0][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 
//		 //a0[1] -= (m_i/(m_e+m_i))*omega_ci.x*v_e_z;
//		 factor = -(m_i/(m_e+m_i))*omega_ci.x;
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
//			real Combined =	(m_e*m_n/((m_e+m_i)*(m_e+m_n)))*nu_en_MT +
//						(m_i*m_n/((m_e+m_i)*(m_i+m_n)))*nu_in_MT  ;
//
//			//a_ion.x += Combined*(v_neut.x - v_ion.x);
//			H[0][3] += Combined;
//			H[0][0] -= Combined;
//			H[1][4] += Combined;
//			H[1][1] -= Combined;
//			
//			//a_ion.z += (m_n/(m_i+m_n))*nu_in_MT*(v_neut.z - v_ion.z);
//			H[2][2] -= (m_n/(m_i+m_n))*nu_in_MT;
//			H[2][5] += (m_n/(m_i+m_n))*nu_in_MT;
//
//			factor = (m_i/(m_i+m_n))*nu_ni_MT;
//			//a_neut += (m_i/(m_i+m_n))*nu_ni_MT*(v_ion - v_neut)
//			 //   + (m_e/(m_n+m_e))*nu_ne_MT*(v_e - v_neut);
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][2] += factor;
//			H[5][5] -= factor;
//			factor = (m_e/(m_n+m_e))*nu_ne_MT;
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
//		 a0[2] += this->StoredEz*qoverM; // StoredEz: populated or not?
//		
//		 // magnetic Lorentz for ions:
//		// a_ion.z -= omega_ci.x*v_ion.y-omega_ci.y*v_ion.x;
//		 H[2][0] += omega_ci.y;
//		 H[2][1] -= omega_ci.x;
//
//		// species combining gives:
//		//a_ion.x -= (m_ion/(m_e+m_ion))*
//		//				omega_ci.y*(v_ion.z-v_e.z); 
//		 // v_e is something we are passed.
//		//a_ion.y += (m_ion/(m_e+m_ion))*
//		//				omega_ci.x*(v_ion.z-v_e.z);
//		 H[0][2] -= (m_i/(m_e+m_i))*omega_ci.y;
//		 H[1][2] += (m_i/(m_e+m_i))*omega_ci.x;
//
//		 //a0[0] += (m_i/(m_e+m_i))*omega_ci.y*v_e_z;
//		 factor = (m_i/(m_e+m_i))*omega_ci.y;
//		 a0[0] += factor*this->SimpleOhms_vez0;
//		 H[0][0] += factor*this->SimpleOhms_beta_ion.x;
//		 H[0][1] += factor*this->SimpleOhms_beta_ion.y;
//		 H[0][2] += factor*this->SimpleOhms_beta_ion.z;
//		 H[0][5] += factor*this->SimpleOhms_beta_neutz;
//
//		 
//		 //a0[1] -= (m_i/(m_e+m_i))*omega_ci.x*v_e_z;
//		 factor = -(m_i/(m_e+m_i))*omega_ci.x;
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
//			real Combined =	(m_e*m_n/((m_e+m_i)*(m_e+m_n)))*nu_en_MT +
//						(m_i*m_n/((m_e+m_i)*(m_i+m_n)))*nu_in_MT  ;
//
//			//a_ion.x += Combined*(v_neut.x - v_ion.x);
//			H[0][3] += Combined;
//			H[0][0] -= Combined;
//			H[1][4] += Combined;
//			H[1][1] -= Combined;
//			
//			//a_ion.z += (m_n/(m_i+m_n))*nu_in_MT*(v_neut.z - v_ion.z);
//			H[2][2] -= (m_n/(m_i+m_n))*nu_in_MT;
//			H[2][5] += (m_n/(m_i+m_n))*nu_in_MT;
//
//			factor = (m_i/(m_i+m_n))*nu_ni_MT;
//			//a_neut += (m_i/(m_i+m_n))*nu_ni_MT*(v_ion - v_neut)
//			 //   + (m_e/(m_n+m_e))*nu_ne_MT*(v_e - v_neut);
//			H[3][0] += factor;
//			H[3][3] -= factor;
//			H[4][1] += factor;
//			H[4][4] -= factor;
//			H[5][2] += factor;
//			H[5][5] -= factor;
//			factor = (m_e/(m_n+m_e))*nu_ne_MT;
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
				totalvertexmass_grams = m_n*pVertex->Neut.mass + m_ion*pVertex->Ion.mass + m_e*pVertex->Elec.mass;
				pVertex->a_pressure_neut_or_overall = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/
					(totalvertexmass_grams);
			};
			if (species == SPECIES_ION) 
				pVertex->a_pressure_ion = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_ion*pVertex->Ion.mass);
			if (species == SPECIES_NEUT)
				pVertex->a_pressure_neut_or_overall = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_n*pVertex->Neut.mass);
			if (species == SPECIES_ELEC)
				pVertex->a_pressure_elec = -cp.Get_Integral_grad_from_anticlockwise_array(nT)/(m_e*pVertex->Elec.mass);

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
			totalvertexmass_grams = m_n*pVertex->Neut.mass + m_ion*pVertex->Ion.mass + m_e*pVertex->Elec.mass;
			
			JcrossB_contribution =  h*h*0.5*q*((pVertex->Ion.mom-pVertex->Elec.mom).cross(pVertex->B))/
						(c*totalvertexmass_grams);
			
			pVertex->AdvectedPosition0 = pVertex->pos
				
				+ h*((m_n*pVertex->Neut.mom +m_ion*pVertex->Ion.mom + m_e*pVertex->Elec.mom).xypart())/
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
	//			// Hsq = multiply row i and column j of H
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
				
				acceleration = momrate/(pVertex->Ion.mass*m_ion + pVertex->Neut.mass*m_n + pVertex->Elec.mass*m_e);
				
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
						
					//momrate.x -= (cc2.y-cc1.y)*Average_nT_ion/m_ion;
					ROCmomrate.x -= (ROCcc2.y-ROCcc1.y)*Average_nT
											+ (cc2.y-cc1.y)*ROCAverage_nT;
					ROCmomrate.y -= (ROCcc1.x-ROCcc2.x)*Average_nT
											+ (cc1.x-cc2.x)*ROCAverage_nT;
				};
				ROC_accel = ROCmomrate / (pVertex->Ion.mass*m_ion+pVertex->Neut.mass*m_n+pVertex->Elec.mass*m_e);
			
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

#include "solver.cpp"

