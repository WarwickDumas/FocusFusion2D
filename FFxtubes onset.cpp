
// DropJxy.cpp, qd cpp files, avi_utils.cpp

// meshutil.cpp, mesh.cpp, basics.cpp, surfacegraph_tri.cpp, vector_tensor.cpp

// simulation.cpp, solver.cpp, vetted.cpp, accelerate.cpp, advance.cpp

#include "headers.h"

// #include some cpp files, which should appear here only :

#include "qd/src/fpu.cpp"
#include "qd/src/dd_const.cpp"
#include "qd/src/dd_real.cpp"
#include "qd/src/bits.cpp"
#include "qd/src/c_dd.cpp"
#include "qd/src/util.cpp"

#include "avi_utils.cpp"     // for making .avi

#include "cppconst.h"

// Global variables:
// =================

float xzscale;

bool bCullNone = false;
bool bGlobalsave = false;
int GlobalSwitchBox = 0;
int iGlobalScratch;

real GlobalHeightScale;

int GlobalSpeciesToGraph = 1;
int GlobalWhichLabels = 0;
bool GlobalRenderLabels = false;
int GlobalColoursPlanView = 0;
bool GlobalBothSystemsInUse;

bool GlobalCutaway = true;

unsigned int cw; // control word for floating point hardware exception hiding

// Simulation globals :
TriMesh * pX, * pXnew;
TriMesh X1, X2;
long steps_remaining, GlobalStepsCounter;
real evaltime, h;  

extern real GlobalIzElasticity;
FILE * massfile,* maxfile;

// Global Variables:
HINSTANCE hInst;   // current instance

// window vars:
HWND hWnd;
WNDCLASSEX wcex;
TCHAR szTitle[1024];					// The title bar text
TCHAR szWindowClass[1024];			// the main window class name

char Functionalfilename[1024];
int GlobalGraphSetting[5];
surfacegraph Graph[5]; // why was it 5? // 5th one can be whole thing.

float Historic_max[100][HISTORY]; // if max is falling, use historic maximum for graph.
float Historic_min[100][HISTORY];
int Historic_powermax[200];
int Historic_powermin[200]; // just store previous value only.

bool boolGlobalHistory, GlobalboolDisplayMeshWireframe;
D3DXVECTOR3 GlobalEye,GlobalLookat,GlobalPlanEye,GlobalPlanEye2,GlobalPlanLookat,
						GlobalPlanLookat2,	GlobalEye2,GlobalLookat2;

// avi file -oriented variables
int const NUMAVI = 1;
HAVI hAvi[NUMAVI];
int const GraphFlags[8] = {SPECIES_NEUTRAL,SPECIES_ION,SPECIES_ELEC,TOTAL,
							    JZAZBXYEZ, JXYAXYBZEXY, EXY_RHO_PHI_JXY, SIGMA_E_J}; 

char szAvi[9][128] = {"Plan","Neut","Ion","Elec","Total",
							"JzAzBxyEz","JxyAxyBzExy","ExyRhoPhiJxy","SigmaEJ"};

AVICOMPRESSOPTIONS opts; 
int counter;
HBITMAP surfbit, dib;
HDC surfdc, dibdc;	
LPVOID lpvBits;
BITMAPINFO bitmapinfo;

IDirect3DSurface9* p_backbuffer_surface;

Systdata Systdata_host;

//=======================================================
// Declarations of functions included in this source file
void RefreshGraphs(const TriMesh & X, const int iGraphsFlag);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	SetupBox(HWND, UINT, WPARAM, LPARAM);
	
void TriMesh::CalculateTotalGraphingData() 
{
	long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			pVertex->n = (pVertex->Neut.mass + pVertex->Ion.mass)/pVertex->AreaCell;
			pVertex->v = (m_n*pVertex->Neut.mom + m_ion*pVertex->Ion.mom + m_e*pVertex->Elec.mom)/
				(m_n*pVertex->Neut.mass+m_ion*pVertex->Ion.mass+m_e*pVertex->Elec.mass);
			pVertex->T = (pVertex->Neut.heat + pVertex->Ion.heat + pVertex->Elec.heat)/
						(pVertex->Neut.mass + pVertex->Ion.mass + pVertex->Elec.mass);
			pVertex->Temp.x = pVertex->Ion.mass/(pVertex->Neut.mass + pVertex->Ion.mass);
		};
		++pVertex;
	}
}
	
void TriMesh::Setup_J() 
{
	long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
			pVertex->Temp = q*(pVertex->Ion.mom-pVertex->Elec.mom)/pVertex->AreaCell;
		} else {
			memset(&(pVertex->Temp),0,sizeof(Vector3));
		}
		++pVertex;
	}
}
void TriMesh::Reset_vertex_nvT(int species) 
{
	long iVertex;
	Vertex * pVertex = X;
	switch (species)
	{
	case SPECIES_NEUT:

		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pVertex->n = pVertex->Neut.mass/pVertex->AreaCell;
				pVertex->v = pVertex->Neut.mom/pVertex->Neut.mass;
				pVertex->T = pVertex->Neut.heat/pVertex->Neut.mass;
			} else {
				pVertex->n = 0.0;
				memset(&(pVertex->v),0,sizeof(Vector3));
				pVertex->T= 0.0;
			};
			++pVertex;
		}
		break;
	case SPECIES_ION:

		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pVertex->n = pVertex->Ion.mass/pVertex->AreaCell;
				pVertex->v = pVertex->Ion.mom/pVertex->Ion.mass;
				pVertex->T = pVertex->Ion.heat/pVertex->Ion.mass;
			} else {
				pVertex->n = 0.0;
				memset(&(pVertex->v),0,sizeof(Vector3));
				pVertex->T= 0.0;
			};
			++pVertex;
		}
		break;
	case SPECIES_ELEC:

		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pVertex->n = pVertex->Elec.mass/pVertex->AreaCell;
				pVertex->v = pVertex->Elec.mom/pVertex->Elec.mass;
				pVertex->T = pVertex->Elec.heat/pVertex->Elec.mass;
			} else {
				pVertex->n = 0.0;
				memset(&(pVertex->v),0,sizeof(Vector3));
				pVertex->T= 0.0;
			};
			++pVertex;
		}
		break;
	}
}


char * report_time(int action)
{
	/* action = 0: reset ; action = 1: report */
	/* requires timebuffy to be defined as char[255] globally */

	static char timebuffer[255];
	static clock_t start;
	double timenow;
	long ops;

	if (action == 0) 
		{
		start = clock();
		}
	else
		{
		timenow = ((double)(clock()-start)/(double)CLOCKS_PER_SEC);
		ops = (long)(clock()-start);
		/* create a null-terminated string */
		sprintf (timebuffer, "%6.4f sec.",timenow);
		};
	return &(timebuffer[0]);	
};


void surfacegraph::DrawSurface(char szname[],
				   const int heightflag,
				   const real * var_ptr_0,
				   const int colourflag,
				   const real * var_ptr_c,
				   const bool bDisplayInner,
				   const int code, // graph code, to pass to called routines - sometimes useful
				   const TriMesh * pX // for passing to SetDataWithColour and Render
										// and for working out offsets
				   )
{
	// replaced CreateSurfaceGraphs.
	// I think this is about the right balance.

	char buff[256];
	real * temprealptr = (real *)(pX->X);
	long offset = var_ptr_0 - temprealptr;
	long offset_c = var_ptr_c - temprealptr;
	
	// Does shader always go with colour type?? yes I think.
	switch(colourflag) {
		case VELOCITY_COLOUR:
			this->mhTech = mFX->GetTechniqueByName("VelociTech");
			break;
		case SEGUE_COLOUR:
			this->mhTech = mFX->GetTechniqueByName("SegueTech");
			break;
		case CURRENT_COLOUR:
			this->mhTech = mFX->GetTechniqueByName("XYZTech");
			break;
		case AZSEGUE_COLOUR:			
			mhTech = mFX->GetTechniqueByName("AzSegueTech");
			break;
		case IONISE_COLOUR:			
			mhTech = mFX->GetTechniqueByName("IoniseTech");
			break;
	};
	
	// Usual settings:
		//if (GlobalGraphSetting[i] != GRAPH_NONE) {

	this->boolDisplayMainMesh = true;
	this->boolDisplayMeshWireframe = GlobalboolDisplayMeshWireframe;
	this->boolClearZBufferBeforeWireframe = false;
	// Or try setting true and CULL_CCW to see if this stops it showing "the back of the wireframe"
	this->SetEyeAndLookat(GlobalEye, GlobalLookat);
	this->boolDisplayScales = true;
		
	this->boolDisplayInnerMesh = bDisplayInner; 

	// work out whether to display key button:
	if (((colourflag == FLAG_VELOCITY_COLOUR) || (colourflag == FLAG_CURRENT_COLOUR))
		&& (bDisplayInner == 0))
	{
		this->boolDisplayKeyButton = true;
	} else {
		this->boolDisplayKeyButton = false;
	};
		//int const FLAG_COLOUR_MESH = 0;
		//int const FLAG_SEGUE_COLOUR = 1;
		//int const FLAG_VELOCITY_COLOUR = 2;
		//int const FLAG_CURRENT_COLOUR = 3;
		//int const FLAG_AZSEGUE_COLOUR = 4;
		//int const FLAG_IONISE_COLOUR = 5;

	this->SetDataWithColour(*pX,
			colourflag,	heightflag, // apparently it's that way round
				offset,offset_c, 
			    code);

	if (this->bDisplayTimestamp) {
		sprintf(buff, "%4.3f ns", evaltime*1.0e9);
		this->Render(szname,false,pX,buff);	
	} else {
		this->Render(szname,false,pX);
	};
}

// Here we make a function that we can call to tidy up graph calling code:


void PlanViewGraphs1(TriMesh & X)// only not const because of such as Reset_vertex_nvT
{
	Vertex * pVertex;
	long iVertex;
	X.Reset_vertex_nvT(SPECIES_ELECTRON);
	D3DXVECTOR3 PlanEye(5.4f,3.2f,70.3f);
	for (int i = 0; i < 4; i++)
	{
		Graph[i].SetEyePlan(PlanEye);
		Graph[i].boolDisplayMeshWireframe = false;
		Graph[i].boolClearZBufferBeforeWireframe = true;
		Graph[i].boolDisplayMainMesh = true;
		Graph[i].boolDisplayInnerMesh = false;
		Graph[i].boolDisplayScales = false;
	};

	int offset_v_e = (real *)(&(X.X[0].v)) -(real *)(&(X.X[0]));
	int offset_T_e = (real *)(&(X.X[0].T)) -(real *)(&(X.X[0]));

	int offset_phi = (real *)(&(X.X[0].phi)) -(real *)(&(X.X[0]));
	int offset_ndiff = (real *)(&(X.X[0].Temp.x)) -(real *)(&(X.X[0]));

	pVertex = X.X;
	for (iVertex = 0; iVertex < X.numVertices; iVertex++)
	{
		pVertex->Temp.x = (pVertex->Ion.mass-pVertex->Elec.mass)/pVertex->AreaCell;
		++pVertex;
	}
	// phi:	
	Graph[0].mhTech = Graph[0].mFX->GetTechniqueByName("AzSegueTech");
	// n_e-n_i:
	Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("AzSegueTech");
	// v_e:
	Graph[1].mhTech = Graph[1].mFX->GetTechniqueByName("VelociTech");
	// Te:
	Graph[3].mhTech = Graph[3].mFX->GetTechniqueByName("SegueTech");

	GlobalWhichLabels = 4; // numbers
	
	Graph[0].SetDataWithColour(X,
		FLAG_AZSEGUE_COLOUR,FLAG_FLAT_MESH,
		offset_phi, offset_phi,
			GRAPH_FLAT_WIRE_MESH);// colourmax is being set.
	Graph[0].colourmax = 2.5; 
	Graph[0].Render("phi", true, &X);  
	
	GlobalWhichLabels = 5; // numbers
	Graph[2].SetDataWithColour(X,
		FLAG_AZSEGUE_COLOUR,FLAG_FLAT_MESH,
		offset_ndiff, offset_ndiff,
			GRAPH_FLAT_WIRE_MESH);
	// Where to set colour scale params??
	Graph[2].colourmax = 1.0e14; 
	Graph[2].Render("ni-ne", true, &X);  
	
	GlobalWhichLabels = 3; 

	Graph[1].SetDataWithColour(X,
		FLAG_VELOCITY_COLOUR,FLAG_FLAT_MESH,
		offset_v_e,offset_v_e,
		GRAPH_FLAT_WIRE_MESH); // ?
	Graph[1].colourmax = 2.5e8; 
	Graph[1].Render("ve",true,&X);

	GlobalWhichLabels = 2;
	// Labels will very shortly be upgraded.

	Graph[3].SetDataWithColour(X,
		FLAG_SEGUE_COLOUR,FLAG_FLAT_MESH,
		offset_T_e, offset_T_e,
		GRAPH_FLAT_WIRE_MESH);
	//Graph[3].colourmax = 4.0e-11;  // colourmax should be left == 1 for T, for whatever reason.
	Graph[3].Render("Te",true,&X);
	// For now use "Render" -- but create an adjunct routine for adding the
	// polygon edges and other details such as velocity arrows..

	// Let's see if we can get this to work before continuing.

	//Graph[3].DrawSurface("[n_s T_s]/[n_s]",
	//  DATA_HEIGHT,(real *)(&(X.X[0].T)),
	//  SEGUE_COLOUR,(real *)(&(X.X[0].T)),
	//  false, 
	//  GRAPH_TOTAL_T, &X);
	//	
	// Graph 2, in case of species graphs:
			
	//if (GlobalColoursPlanView == 0)
	// nothing
	//Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("MeshTech");
	//Graph[2].SetDataWithColour(X,FLAG_COLOUR_MESH,FLAG_FLAT_MESH,0,0,
	//								GRAPH_FLAT_WIRE_MESH);
	//Graph[2].Render(buff, GlobalRenderLabels, &X);  

	// Tell SDWC not to mess with colourmax if it's a flat mesh.

}


void RefreshGraphs(TriMesh & X, // only not const because of such as Reset_vertex_nvT
				   const int iGraphsFlag)
{
	Vertex * pVertex;
	long iVertex;

	switch(iGraphsFlag) {
		case SPECIES_NEUTRAL:

			X.Reset_vertex_nvT(SPECIES_NEUTRAL);

			Graph[0].DrawSurface("Neutral n",
			  DATA_HEIGHT,(real *)(&(X.X[0].n)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_NEUT_N, &X);
			Graph[1].DrawSurface("Neutral v",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].v)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_NEUT_V, &X);
			Graph[3].TickRescaling = 1.0/kB;
			Graph[3].DrawSurface("Neutral T",
			  DATA_HEIGHT,(real *)(&(X.X[0].T)),
			  SEGUE_COLOUR,(real *)(&(X.X[0].T)),
			  false, // no inner mesh display
			  GRAPH_NEUT_T, &X);
			Graph[3].TickRescaling = 1.0;
			// How to handle Graph[2] ?
			
			break;
		case SPECIES_ION:

			X.Reset_vertex_nvT(SPECIES_ION);

			Graph[3].TickRescaling = 1.0/kB;
			Graph[3].DrawSurface("Ion T",
			  DATA_HEIGHT,(real *)(&(X.X[0].T)),
			  SEGUE_COLOUR,(real *)(&(X.X[0].T)),
			  false, // no inner mesh display
			  GRAPH_ION_T, &X);
			Graph[3].TickRescaling = 1.0;

			// labels only appear on first 1 called.

			Graph[0].DrawSurface("Ion n",
			  DATA_HEIGHT,(real *)(&(X.X[0].n)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_ION_T, &X);
			Graph[1].DrawSurface("Ion v",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].v)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_ION_V, &X);

			break;

		case SPECIES_ELEC:

			X.Reset_vertex_nvT(SPECIES_ELEC);

			Graph[0].DrawSurface("Elec n",
			  DATA_HEIGHT,(real *)(&(X.X[0].n)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_ELEC_T, &X);
			Graph[1].DrawSurface("Elec v",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].v)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_ELEC_V, &X);
			Graph[3].TickRescaling = 1.0/kB;
			Graph[3].DrawSurface("Neutral T",
			  DATA_HEIGHT,(real *)(&(X.X[0].T)),
			  SEGUE_COLOUR,(real *)(&(X.X[0].T)),
			  false, // no inner mesh display
			  GRAPH_ELEC_T, &X);
			Graph[3].TickRescaling = 1.0;
			break;

		// In other cases, (and even for the above),
		// here is a good place to call the 
		// setup routines for temp variables.

		case JZAZBXYEZ:

			X.Setup_J(); // the others can already exist.

			Graph[0].DrawSurface("Az",
			  DATA_HEIGHT,(real *)(&(X.X[0].A.z)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].A.z)),
			  true, 
			  GRAPH_AZ, &X);
			Graph[1].DrawSurface("Bxy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].B)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].B)),
			  false, // no inner mesh display: ??
			  GRAPH_BXY, &X);
			Graph[2].DrawSurface("Ez",
			  DATA_HEIGHT,(real *)(&(X.X[0].E.z)),
			  FLAG_SEGUE_COLOUR,(real *)(&(X.X[0].E.z)),
			  false, // ??
			  GRAPH_EZ, &X);
			Graph[3].DrawSurface("Jz",
			  DATA_HEIGHT,(real *)(&(X.X[0].Temp.z)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].Temp.z)),
			  false, // no inner mesh display.
			  GRAPH_JZ, &X);
			break;

		case JXYAXYBZEXY:

			X.Setup_J(); // the others can already exist.

			Graph[0].DrawSurface("Axy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].A.x)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].A.x)),
			  true, 
			  GRAPH_AXY, &X);
			Graph[1].DrawSurface("Bz",
			  DATA_HEIGHT,(real *)(&(X.X[0].B.z)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].B.z)),
			  false, // no inner mesh display: ??
			  GRAPH_BZ, &X);
			Graph[2].DrawSurface("Exy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].E)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].E)),
			  false, 
			  GRAPH_EXY, &X);
			Graph[3].DrawSurface("Jxy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].Temp.x)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].Temp.x)),
			  false, // no inner mesh display.
			  GRAPH_JXY, &X);
			break;
		
		case EXY_RHO_PHI_JXY:
			// create rho on pVertex->temp2.x ... 
			
			pVertex= pX->X;
			for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
			{
				if (pVertex->flags == DOMAIN_VERTEX) {
					pVertex->temp2.x = q*(pVertex->Ion.mass-pVertex->Elec.mass)/pVertex->AreaCell;
				} else {
					pVertex->temp2.x = 0.0;
				};
				++pVertex;
			}

			Graph[0].DrawSurface("phi",
			  DATA_HEIGHT,(real *)(&(X.X[0].phi)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].phi)),
			  false,
			  GRAPH_PHI, &X);
			Graph[1].DrawSurface("Jxy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].Temp)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].Temp)),
			  false, // no inner mesh display: ??
			  GRAPH_JXY, &X);
			Graph[2].DrawSurface("Exy",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].E)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].E)),
			  false, 
			  GRAPH_EXY, &X);
			Graph[3].DrawSurface("rho",
			  DATA_HEIGHT,(real *)(&(X.X[0].temp2.x)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].temp2.x)),
			  false, // no inner mesh display.
			  GRAPH_RHO, &X);
			break;

		case SIGMA_E_J:
			
			X.Setup_J(); // the others can already exist.

			Graph[0].DrawSurface("sigma_e_zz",
				DATA_HEIGHT,(real *)(&(X.X[0].sigma_e.zz)),
				AZSEGUE_COLOUR,(real *)(&(X.X[0].sigma_e.zz)),
			  true, 
			  GRAPH_SIGMA_E, &X);
			//Graph[1].DrawSurface("v_e_0.z",
			//	DATA_HEIGHT,(real *)(&(X.X[0].v_e_0.z)),
			//	AZSEGUE_COLOUR,(real *)(&(X.X[0].v_e_0.z)),
			  //false, // no inner mesh display: ??
			 // GRAPH_VE0Z, &X);
			Graph[1].DrawSurface("nsigma",
				DATA_HEIGHT,(real *)(&(X.X[0].xdotdot.x)),
				AZSEGUE_COLOUR,(real *)(&(X.X[0].xdotdot.x)),
				true, GRAPH_SIGMATEMP, &X);
			Graph[2].DrawSurface("Ez",
			  DATA_HEIGHT,(real *)(&(X.X[0].E.z)),
			  FLAG_AZSEGUE_COLOUR,(real *)(&(X.X[0].E.z)), // how to make SEGUE_COLOUR work?
			  false, // ??
			  GRAPH_EZ, &X);
			Graph[3].DrawSurface("Jz",
			  DATA_HEIGHT,(real *)(&(X.X[0].Temp.z)),
			  AZSEGUE_COLOUR,(real *)(&(X.X[0].Temp.z)),
			  false, // no inner mesh display.
			  GRAPH_JZ, &X);
			break;


		case TOTAL:

			// In this case we have to create data,
			// as we go.

			// Best put it here so we can see where
			// data is being populated.

			X.CalculateTotalGraphingData();

			// ought to change this to use variables n,v,T !

			Graph[0].DrawSurface("n_n+n_ion",
			  DATA_HEIGHT,(real *)(&(X.X[0].n)),
			  IONISE_COLOUR,(real *)(&(X.X[0].Temp.y)),
			  false,
			  GRAPH_TOTAL_N, &X);
			Graph[1].DrawSurface("[n_s v_s m_s]/[n_s m_s]",
			  VELOCITY_HEIGHT,(real *)(&(X.X[0].v)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, // no inner mesh display
			  GRAPH_TOTAL_V, &X);
			Graph[2].DrawSurface("n_n+n_ion",
			  DATA_HEIGHT,(real *)(&(X.X[0].n)),
			  VELOCITY_COLOUR,(real *)(&(X.X[0].v)),
			  false, 
			  GRAPH_TOTAL_N_II, &X);
			Graph[3].TickRescaling = 1.0/kB;
			Graph[3].DrawSurface("[n_s T_s]/[n_s]",
			  DATA_HEIGHT,(real *)(&(X.X[0].T)),
			  SEGUE_COLOUR,(real *)(&(X.X[0].T)),
			  false, 
			  GRAPH_TOTAL_T, &X);
			Graph[3].TickRescaling = 1.0;
		break;
		
	};

	// Graph 2, in case of species graphs:
	
	char buff[256];
	sprintf(buff, "%4.3f ns", evaltime*1.0e9);
			
	switch(iGraphsFlag) {
		case SPECIES_NEUTRAL:
		case SPECIES_ION:
		case SPECIES_ELEC:
		
			Graph[2].SetEyePlan(GlobalPlanEye);
			Graph[2].boolDisplayMeshWireframe = true;
			Graph[2].boolClearZBufferBeforeWireframe = true;
			Graph[2].boolDisplayMainMesh = true;
			Graph[2].boolDisplayInnerMesh = false;
			Graph[2].boolDisplayScales = false;

			if (GlobalColoursPlanView == 0)
			{
				// nothing
				Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("MeshTech");
				Graph[2].SetDataWithColour(X,FLAG_COLOUR_MESH,FLAG_FLAT_MESH,0,0,
																GRAPH_FLAT_WIRE_MESH);
				Graph[2].Render(buff, GlobalRenderLabels, &X);  

			} else {
				// Tell SDWC not to mess with colourmax if it's a flat mesh.

				X.Reset_vertex_nvT(SPECIES_ION);

				int offset_v_ion = (real *)(&(X.X[0].v)) -(real *)(&(X.X[0]));
				int offset_T_ion = (real *)(&(X.X[0].T)) -(real *)(&(X.X[0]));

				if (GlobalColoursPlanView == 1)
				{
					// velocity
					Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("VelociTech");
					Graph[2].colourmax = Graph[0].colourmax; // match colours
					
					Graph[2].SetDataWithColour(X,FLAG_VELOCITY_COLOUR,FLAG_FLAT_MESH,offset_v_ion,offset_v_ion,
						GRAPH_FLAT_WIRE_MESH);
					Graph[2].Render(buff, GlobalRenderLabels, &X);  
				} else {
					// temperature
					Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("SegueTech");
					// SegueVS should take maximum as a parameter;
					// at least for colours we should prefer an absolute scale for T
					// Is it ever used for anything else? Not so far? eps?
					
					Graph[2].SetDataWithColour(X,FLAG_SEGUE_COLOUR,FLAG_FLAT_MESH,offset_T_ion, offset_T_ion,
						GRAPH_FLAT_WIRE_MESH);
					Graph[2].Render(buff, GlobalRenderLabels, &X);  
				};
			};
		break;
	}
}


int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPTSTR    lpCmdLine,
                     int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);
	unsigned int old_cw;

	// with X86 defined 0 in config.h, this will do nothing:
//	fpu_fix_start(&old_cw);

	char szInitialFilenameAvi[512];
	MSG msg;
	HDC hdc;
//	HACCEL hAccelTable;
	real x,y,temp;
	int i,j;
	float a1,a2,a3,a4;
	HWND hwndConsole;
	FILE * fp;
	extern char Functionalfilename[1024];
	
	hInst = hInstance; 
	
	cw = 0;
	_controlfp_s(&cw, 0, 0); // zeroes out control word. 
    _controlfp_s(0, _EM_INEXACT | _EM_UNDERFLOW, _MCW_EM); 
	// hide only "inexact" and underflow FP exception
	// Why underflow?

	AllocConsole(); // for development purposes
	freopen( "CONOUT$", "wb", stdout );
	
	SetConsoleTitle("2D 1/16 annulus DPF simulation");
	Sleep(40);
	hwndConsole = FindWindow(NULL,"2D 1/16 annulus DPF simulation");
	MoveWindow(hwndConsole,0,0,SCREEN_WIDTH-VIDEO_WIDTH-10,SCREEN_HEIGHT-30,TRUE);

	h = TIMESTEP;
	evaltime = 0.0; // gets updated before advance
	
	memset(Historic_powermax, 0,200*sizeof(int));
	memset(Historic_powermin, 0,200*sizeof(int));

	ZeroMemory(Historic_max,100*HISTORY*sizeof(float));
	ZeroMemory(Historic_min,100*HISTORY*sizeof(float));

	Vector2 x1,x2,x3,r1,r2,r3;
	
	x1.x = 0.0; x1.y = 0.0;
	x2.x = 1.0; x2.y = 1.0;
	x3.x = 2.0; x3.y = 0.0;
	r1.x = 0.0; r1.y = 0.0;
	r2.x = 0.0; r2.y = 1.0;
	r3.x = 1.0; r3.y = 0.0; 
	// so far, appears to work

	real area = CalculateTriangleIntersectionArea(x1, x2, x3, r1, r2, r3);

	area = CalculateTriangleIntersectionArea(r1, r2, r3,x1, x2, x3);
	// what would really give confidence would be if we finish our columns method
	// and verify that they give the same answers.

	GlobalStepsCounter = 0; steps_remaining = 0;

	// Note : atanf gives values in 1st and 4th quadrant - but remember atan2
		
	// Find next available filename for functional output:
	int filetag = 0;
	do {
		filetag++;
		sprintf(Functionalfilename,FUNCTIONALFILE_START "%03d.txt",filetag);
	} while ((_access( Functionalfilename, 0 )) != -1 );
  	//if( (_access( "ACCESS.C", 0 )) != -1 )
   //{
   //   printf( "File ACCESS.C exists\n" );
   //   /* Check for write permission */
   //   if( (_access( "ACCESS.C", 2 )) != -1 )
   //      printf( "File ACCESS.C has write permission\n" );
   //}
	printf("\n\nopening %s \n",Functionalfilename);
	fp = fopen(Functionalfilename,"w");
	if (fp == 0) {
		 printf("error with %s \n",Functionalfilename);
		 getch();
	} else {
		 printf("opened %s \n",Functionalfilename);
	};
	fprintf(fp,"GSC evaltime Area neut.N ion.N elec.N neut.r ion.r elec.r SDneut.r SDion.r SDelec.r "
			 " neut.vr neut.vth neut.vz  ion.vr ion.vth ion.vz elec.vr elec.vth elec.vz neut.heat ion.heat elec.heat neut.T ion.T elec.T "
			 " neut.mnvv/3 ion.mnvv/3 elec.mnvv/3 elec.force(vxB)r within3.6 elec.Bth EE BB Heatings and dT changes - see code \n");
	fclose(fp);


	report_time(0);
	printf("Initialising meshes...\n");
	
	if (bScrewPinch) {
		//X1.InitialiseScrewPinch();
		//X2.InitialiseScrewPinch();
	} else {
		X1.Initialise(1); // Set evaltime first
		
		X2.Initialise(2);
		
	}

	pX = &X1;
	pXnew = &X2;

	// A,phi seeding relies on both system values

	GlobalBothSystemsInUse = 0;

	//if (SYSTEM_INITIALISE_VS_LOAD == LOAD)
	//{
	//	if (pX->Load(AUTO_DATA_LOAD_FILENAME))
	//	{
	//		printf("File error auto-loading %s",AUTO_DATA_LOAD_FILENAME);
	//		getch();
	//		PostQuitMessage(2000);
	//	} else {
	//		pXnew->Load(AUTO_DATA_LOAD_FILENAME);
	//		// lazy way! Do differently.			
	//	};
	//};

	printf(report_time(1));
	printf("\n");
	report_time(0);

	// Window setup
	LoadString(hInstance, IDS_APP_TITLE, szTitle, 1024);
	LoadString(hInstance, IDC_F2DVALS, szWindowClass, 1024);
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= LoadIcon(hInstance, MAKEINTRESOURCE(IDI_F2DVALS));
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName	= MAKEINTRESOURCE(IDR_MENU1);
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
	RegisterClassEx(&wcex);
	
	printf("SCREEN_WIDTH %d VIDEO_WIDTH %d VIDEO_HEIGHT %d \n",
		SCREEN_WIDTH,VIDEO_WIDTH,VIDEO_HEIGHT);


	hWnd = CreateWindowEx(NULL,szWindowClass, szTitle, WS_BORDER | WS_POPUP,
      SCREEN_WIDTH-VIDEO_WIDTH-5, 0, VIDEO_WIDTH+5, VIDEO_HEIGHT+20, NULL, NULL, hInstance, NULL);
	if (!hWnd) return 3000;
		// This fails, although dimensions are sensible.
	
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
	
	xzscale = 2.0/0.1; // very zoomed in. Now what?
	

	DXChk(Direct3D.Initialise(hWnd,hInstance,VIDEO_WIDTH,VIDEO_HEIGHT));
	
	// With Field Of View = PI/4 used this:

	GlobalEye.x = 0.0f;
	GlobalEye.y = 12.4f;  //7.2f;
	GlobalEye.z = -18.0f+2.5*xzscale;//DEVICE_RADIUS_INSULATOR_OUTER*xzscale;//-17.8f+

	GlobalLookat.x = 0.4f;
	GlobalLookat.y = 3.0f;
	GlobalLookat.z = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*xzscale; 

	GlobalPlanEye.x = 0.0f;
	GlobalPlanEye.y = 35.0f;
	GlobalPlanEye.z = (3.44+4.1)*0.5*xzscale; 

	GlobalPlanEye2.x = -0.1f;
	GlobalPlanEye2.y = 19.5f;
	GlobalPlanEye2.z = 2.8*xzscale; 

	GlobalPlanLookat.x = GlobalPlanEye.x;
	GlobalPlanLookat.y = 0.0f;
	GlobalPlanLookat.z = GlobalPlanEye.z+0.0001;

	GlobalPlanLookat2.x = GlobalPlanEye2.x;
	GlobalPlanLookat2.y = 0.0f;
	GlobalPlanLookat2.z = GlobalPlanEye2.z+0.0001; 

	if (DXChk(	Graph[3].InitialiseWithoutBuffers(GRAPH_WIDTH,GRAPH_HEIGHT,GRAPH_WIDTH,GRAPH_HEIGHT,GlobalEye,GlobalLookat) ) +
		DXChk(	Graph[3].InitialiseBuffers(X1) ) 
		) 
			{ PostQuitMessage(203); };

	if (DXChk(	Graph[2].InitialiseWithoutBuffers(GRAPH_WIDTH,0,GRAPH_WIDTH,GRAPH_HEIGHT,GlobalPlanEye,GlobalPlanLookat) ) +
		DXChk(	Graph[2].InitialiseBuffers(X1) ) 
		) 
			{ PostQuitMessage(202); };

	if (DXChk(	Graph[1].InitialiseWithoutBuffers(0,GRAPH_HEIGHT,GRAPH_WIDTH,GRAPH_HEIGHT,GlobalEye,GlobalLookat) ) +
		DXChk(	Graph[1].InitialiseBuffers(X1) ) 
		) 
			{ PostQuitMessage(201); };

	if (DXChk(	Graph[0].InitialiseWithoutBuffers(0,0,GRAPH_WIDTH,GRAPH_HEIGHT,GlobalEye, GlobalLookat) ) +
		DXChk(	Graph[0].InitialiseBuffers(X1) ) 
		) 
			{ PostQuitMessage(200); };
	
	if (DXChk(	Graph[4].InitialiseWithoutBuffers(0,0,GRAPH_WIDTH*2,GRAPH_HEIGHT*2,GlobalPlanEye,GlobalPlanLookat) ) +
		DXChk(	Graph[4].InitialiseBuffers(X1) ) 
		) 
			{ PostQuitMessage(204); };

	Graph[4].SetEyePlan(GlobalPlanEye);
	Graph[4].boolDisplayMeshWireframe = true;
	Graph[4].boolClearZBufferBeforeWireframe = true;
	Graph[4].boolDisplayMainMesh = true;
	Graph[4].boolDisplayInnerMesh = false;
	Graph[4].boolDisplayScales = false;

//	if (DXChk(  Graph[4].InitialiseWithoutBuffers(0,0,2*GRAPH_WIDTH,
//		2*GRAPH_HEIGHT,GlobalPlanEye2,GlobalPlanLookat2) ) +
//		DXChk(	Graph[4].InitialiseBuffers(X1) ) 
//		) 
//			{ PostQuitMessage(208); };
	// ?? -- for doing 1 graph in whole space. ?!

	Graph[0].bDisplayTimestamp = false;
	Graph[1].bDisplayTimestamp = false;
	Graph[2].bDisplayTimestamp = true;
	Graph[3].bDisplayTimestamp = false;

	Direct3D.pd3dDevice->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO, &p_backbuffer_surface);
	
	if(DXChk(p_backbuffer_surface->GetDC(&surfdc),1000))
					MessageBox(NULL,"GetDC failed","oh dear",MB_OK);
	
	surfbit = CreateCompatibleBitmap(surfdc,VIDEO_WIDTH,VIDEO_HEIGHT); // EXTRAHEIGHT = 90
	SelectObject(surfdc,surfbit);
	dibdc = CreateCompatibleDC(surfdc);
 
	long VideoWidth = VIDEO_WIDTH;
	long VideoHeight = VIDEO_HEIGHT;

	// pasted here just to set up format:
	bitmapinfo.bmiHeader.biSize = sizeof(BITMAPINFO);
	bitmapinfo.bmiHeader.biWidth = VideoWidth;
	bitmapinfo.bmiHeader.biHeight = VideoHeight;
	bitmapinfo.bmiHeader.biPlanes = 1;
	bitmapinfo.bmiHeader.biBitCount = 24;
	bitmapinfo.bmiHeader.biCompression = BI_RGB; // uncompressed  
	bitmapinfo.bmiHeader.biSizeImage = bitmapinfo.bmiHeader.biHeight;
	bitmapinfo.bmiHeader.biXPelsPerMeter = 3000;
	bitmapinfo.bmiHeader.biYPelsPerMeter = 3000;
	bitmapinfo.bmiHeader.biClrUsed = 0;
	bitmapinfo.bmiHeader.biClrImportant = 0;
	bitmapinfo.bmiColors->rgbBlue = 0;
	bitmapinfo.bmiColors->rgbRed = 0;
	bitmapinfo.bmiColors->rgbGreen = 0;
	bitmapinfo.bmiColors->rgbReserved = 0;
	// dimension DIB and set up pointer to bits
	dib = CreateDIBSection(dibdc, &bitmapinfo, DIB_RGB_COLORS, &lpvBits, NULL, 0);
	SelectObject(dibdc, dib);

	BitBlt(dibdc,0,0,VIDEO_WIDTH,VIDEO_HEIGHT,surfdc,0,0,SRCCOPY);

	for (i = 0 ; i < NUMAVI; i++)
	{
		sprintf(szInitialFilenameAvi,"%s%s_%s",FOLDER,szAvi[i],INITIALAVI);
		hAvi[i] = CreateAvi(szInitialFilenameAvi, AVIFRAMEPERIOD, NULL);
	};

	// 1000/25 = 40
	ZeroMemory(&opts,sizeof(opts));
	opts.fccHandler=mmioFOURCC('D','I','B',' ');//('d','i','v','x');
	opts.dwFlags = 8;
	
	for (i = 0 ; i < NUMAVI; i++)
		SetAviVideoCompression(hAvi[i],dib,&opts,false,hWnd); // always run this for every avi file but can
	// call with false as long as we know opts contains valid information. 
	
	counter = 0;
	
	//ReleaseDC(hWnd,surfdc);
	p_backbuffer_surface->ReleaseDC(surfdc);

	GlobalCutaway = true;

	RefreshGraphs(*pX,GlobalSpeciesToGraph);
	ZeroMemory( &msg, sizeof( msg ) );

	// Main message loop:
	while( msg.message != WM_QUIT )
    {
        if( PeekMessage( &msg, NULL, 0U, 0U, PM_REMOVE ) )
        {
			TranslateMessage( &msg );
            DispatchMessage( &msg );
        } else {
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
		};
	};
	
	UnregisterClass(szWindowClass, wcex.hInstance );
	fclose( stdout );
	FreeConsole();

//	fpu_fix_end(&old_cw);
	return (int) msg.wParam;
}	


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int iAntiskips;
	int wmId, wmEvent;
	int i,j;
	PAINTSTRUCT ps;
	HDC hdc;
	real time_back_for_Adot;
	TriMesh * temp;
	FILE * file, *fp;
	int maxeerr, count,iMin;
	char buf1000[1024];
	int attempts;
	real store_h;
	char ch;
	int failed;
	RECT rect;
	Vertex * pVertex;
	real TotalArea, TotalCharge;
	Triangle * pTri,*pTriSrc;
	long iVertex;
	real mass_avg, mass_SD, mass_min, mass_max;
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name
	char szFilter[1000]; // buffer for file filter
	char szfilter[256];

	switch (message)
	{
	case WM_CREATE:
		
		// Don't ever try doing initialisation here;
		// That should be done manually from the menus.		
		RefreshGraphs(*pX,GlobalSpeciesToGraph); 
		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
		
		break;
	case WM_COMMAND:
		wmId    = LOWORD(wParam);
		wmEvent = HIWORD(wParam);

		// Ensure that display menu items are consecutive IDs.

		if ((wmId >= ID_DISPLAY_NEUT) && 
			(wmId < ID_DISPLAY_NEUT + NUMAVI))
		{
			//if (wmId < ID_DISPLAY_NEUT + 4) 
			GlobalSpeciesToGraph = wmId;

			i = wmId-ID_DISPLAY_NEUT;
			printf("\nGraph: %d %s",i,szAvi[i]);
			RefreshGraphs(*pX,i); 
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
			return 0;
		}
		
		// Parse the menu selections:
		switch (wmId)
		{
		case ID_HELP_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case ID_FILE_EXIT:
			DestroyWindow(hWnd);
			break;
		case ID_FILE_SAVECAMERA:
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0cam\0*.CAM\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;
			
			if (GetSaveFileName(&ofn)==TRUE) 
			{
				printf("\nsaving camera...");
				fp = fopen(ofn.lpstrFile,"wt");
				if (fp == 0) {
					printf("save failed.\n");
				} else {
					fprintf(fp,"%f %f %f ",GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);	
					fprintf(fp,"%f %f %f ",GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
					fprintf(fp,"%f %f %f ",GlobalEye.x,GlobalEye.y,GlobalEye.z);
					fprintf(fp,"%f %f %f ",GlobalPlanLookat.x,GlobalPlanLookat.y,GlobalPlanLookat.z);
					fclose(fp);
					printf("done\n");			
				};
			} else {
				printf("there was an issue\n");
			};
			break;
		case ID_FILE_LOADCAMERA:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0*.cam\0*.Cam\0\0", 21); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0*.Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
				// Display the Open dialog box. 
			if (GetOpenFileName(&ofn)==TRUE) 
			{
				printf("\nloading camera...");
				fp = fopen(ofn.lpstrFile,"rt"); 
				if (fp == 0) {
					printf("failed.\n");
				} else {
					rewind(fp);
					fscanf(fp,"%f %f %f ",&(GlobalPlanEye.x),&(GlobalPlanEye.y),&(GlobalPlanEye.z));	
					fscanf(fp,"%f %f %f ",&(GlobalLookat.x),&(GlobalLookat.y),&(GlobalLookat.z));
					fscanf(fp,"%f %f %f ",&(GlobalEye.x),&(GlobalEye.y),&(GlobalEye.z));
					fscanf(fp,"%f %f %f ",&(GlobalPlanLookat.x),&(GlobalPlanLookat.y),&(GlobalPlanLookat.z));
					fclose(fp);
				};
				RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
				Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
						
			} else {
				printf("file error camera\n");
			};
			break;
		case ID_FILE_SAVEBINARY:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0*.dat\0*.Dat\0\0", 21); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;
			// Display the Open dialog box. 
			if (GetSaveFileName(&ofn)==TRUE) 
			{
				printf("\nsaving system...");
				pX->Save(ofn.lpstrFile);
				printf("done\n");			
			} else {
				printf("there was an issue\n");
			};
			break;
		case ID_FILE_SAVETEXT:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			//
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			//strcpy(szFilter,"All\0*.*\0Text\0*.TXT\0");
			memcpy(szfilter, "All\0*.*\0Dat\0*.DAT\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;
		// Display the Open dialog box. 
			if (GetSaveFileName(&ofn)==TRUE) 
			{
				printf("\nsaving system...");
				pX->SaveText(ofn.lpstrFile);
				printf("done\n");
			} else {
				printf("there was an issue\n");
			};
		break;
		case ID_FILE_LOAD:
			
			// Initialize OPENFILENAME:
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			//
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			//strcpy(szFilter, "All\0*.*\0Dat\0*.DAT\0\0");
			memcpy(szfilter, "All\0*.*\0Dat\0*.DAT\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

			// Display the Open dialog box. 
			if (GetOpenFileName(&ofn)==TRUE) 
			{
				printf("\nloading system...");
			
				X1.Load(ofn.lpstrFile);
				printf("done\n");
				
				RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
				Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
				GlobalBothSystemsInUse = 0;

			} else {
				printf("file error loadsyst\n");
			};
			break;

		case ID_RUN_SIMULATIONSTEPS:

			GlobalSwitchBox = 0;
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG1), hWnd, SetupBox);
			// that will not return with steps_remaining unset.
			
			if (steps_remaining > 0) 
				SetTimer(hWnd, 1, 10, NULL); // 10 millisecond delay
			
			break;
		
		case ID_RUN_STOP:

			steps_remaining = 0;
			break;

		case ID_INITIALISE_IONISATIONSTEPS:
			/*
			GlobalSwitchBox = 0;
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG1), hWnd, SetupBox);
			// that will not return with steps_remaining unset.
			
			
	Celldata_host = new fluid3BE[pX->numTriangles];
	Cellinfo_host = new structural[pX->numTriangles];
	Vertinfo_host = new vertinfo[pX->numTriangles];
	Vertdata_host = new vertdata[pX->numTriangles];

		pX->Setup_Data_for_CUDA(
							Celldata_host, // pointers to receive data
							//Vertdata_host,
							Cellinfo_host, 
							Vertinfo_host,
							Vertdata_host
							) ;
		
		printf("passed data to arrays.\n");

		PerformCUDA_Ionisation_Only(
			Celldata_host, Cellinfo_host, Vertinfo_host ,
			Vertdata_host, pX->numTriangles, pX->numVertices,
		    1.0e-13, 100); 
		// 100 repeats of 1e-13 ionisation.

		pX->Suck_Data_from_CUDA(Celldata_host);
		
		printf("retrieved data from arrays.\n");

		pX->RecalculateVertexVariables(); // needed to affect graphics

		RefreshGraphs(*pX); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );

	delete[] Celldata_host;
	delete[] Cellinfo_host;
	delete[] Vertinfo_host;
	delete[] Vertdata_host;
*/
			break;

		case ID_INITIALISE_RUNPOTENTIALSSOLVER:
			
			// Inputs:

			// Need to take pressure and grad Te before we call 
			// to setup matrices.

			printf("pX->GetGradTeOnVertices(); \n");
			pX->GetGradTeOnVertices(); // shouldn't have to belong here.

			pX->Iz_prescribed = pX->GetIzPrescribed(evaltime);
			
			printf("pX->EstimateInitialOhms_zz(); \n");
			pX->EstimateInitialOhms_zz(); // set up (zz) Ohm's Law for initial solve.
			
			GlobalIzElasticity = PIOVERPEAKTIME / tan ((evaltime + ZCURRENTBASETIME) * PIOVERPEAKTIME );
			store_h = h;

			printf("GlobalIzElasticity %1.10E \n",GlobalIzElasticity);
			getch();

			printf("pX->Solve_A_phi(true); \n");
			
			h = 1.0/c; // for chargeflows

			time_back_for_Adot = 0.0; //store_h*0.5; // Assume we want to set A-dot for half a simulation timestep into the past.
			printf("Time where Adot was set: %1.5E in the past",time_back_for_Adot);
			pX->Solve_Az(time_back_for_Adot); 
			h = store_h;
			pX->GetBFromA(); // MAKE SURE THEN THAT WE ARE PASSING IT.
			
			if (pX == &X1) {
				printf("\n\n did for X1\n");
			} else {
				printf("\n\n did for X2\n");
			};

			RefreshGraphs(*pX, JZAZBXYEZ);
			//GlobalSpeciesToGraph = JZAZBXYEZ;
			Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
					
			
/*
			pX->CreateVertexAveragingCoefficients(); // used in visc coeff routine and ODE

			pX->MakePressureAccelData(SEPARATE_ELECTRON_AND_ION);
			pTri = pX->T;
			for (iTri = 0; iTri < pX->numTriangles; iTri++)
			{
				pTri->RecalculateEdgeNormalVectors(false);
				pX->GetGradTe(pTri);
				++pTri;
			};			
			
			// Bear in mind we should define e,i params as 0 outside domain.
			pX->EstimateInitialSigma_etc();
			// defines v_e and v_i parameters
		
			
			pX->Iz_prescribed = pX->GetIzPrescribed(evaltime);
		
			if (wmId == ID_INITIALISE_RUNASOLVER)
			{
				if (bScrewPinch) {
					
					store_h = h;
					h = 100.0;
					printf("h effective for initial solve = %1.12E \n",h);			
					pX->Solve_AzEz(0);
					h = store_h;
					
				} else {

					GlobalIzElasticity = PIOVERPEAKTIME / tan ((evaltime + ZCURRENTBASETIME) * PIOVERPEAKTIME );
					store_h = h;
					h = 1.0/GlobalIzElasticity;
					printf("h effective for initial solve = %1.12E \n",h);			
					pX->Solve_AzEz(0);
					h = store_h;
				};
			} else {
	
				//h = 1.0e-8; // try that.

				pX->Solve_AzEz(pXnew);	
			};

			// ??
*/
			break;


		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;
		

	case WM_TIMER:
					
		if (wParam == 1)		
		{
			KillTimer(hWnd, 1);
			
			// Run 1 step:
			if (steps_remaining > 0)
			{
				//HeapValidate(GetProcessHeap(),0,NULL);//printf("heap validated.\n"); //getch();
				
				GlobalStepsCounter++;
				failed = 0;
			
				if (pX == &X1) {
					printf("Setup from X1 ");
				} else {
					printf("Setup from X2 ");
				};
				printf("\n\n16000: elec.mom.z %1.8E \n\n",pX->X[16000].Elec.mom.z);

#ifndef NOCUDA
				Systdata_host.InvokeHost(pX->numVertices);
				pX->Setup_Data_for_CUDA( &Systdata_host ) ;
				
				PerformCUDA_Advance (
					&Systdata_host, 
					pX->numVertices,
					1e-13, 
					10,
					pX->numStartZCurrentRow,
					pX->numEndZCurrentRow,
					&Systdata_host,
					evaltime 
				);
				evaltime += 1.0e-13;
				printf("evaltime %1.8E\n",evaltime);
#endif
				pX->Suck_Data_from_CUDA(&Systdata_host);
				printf("That's it.\n");
				
				// End of test.
				//		pX->Advance(pXnew);

				if (!failed)
				{
					// Only sucking back to pX.

			//		temp = pX;
			//		pX = pXnew;
			//		pXnew = temp;
				
					steps_remaining--;
					//evaltime += h;	// this is done during the step
					
					printf("Done step: %d   ||   Remaining this run: %d\n\n",GlobalStepsCounter,steps_remaining); 
												
					if (GlobalStepsCounter % GRAPHICS_FREQUENCY == 0)
					{
						// make video frames:
						//for (i = 0; i < NUMAVI; i++)
						i = 0;
						{
							::PlanViewGraphs1(*pX);

							//RefreshGraphs(*pX,GraphFlags[i]); // sends data to graphs AND renders them
							Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
						
							if(DXChk(p_backbuffer_surface->GetDC(&surfdc),100))
								MessageBox(NULL,"GetDC failed","oh dear",MB_OK);
							//SelectObject(surfdc,surfbit);
							BitBlt(dibdc,0,0,VIDEO_WIDTH,VIDEO_HEIGHT,surfdc,0,0,SRCCOPY);
							p_backbuffer_surface->ReleaseDC(surfdc);
							AddAviFrame(hAvi[i], dib);
						};								
					};
					
			//		RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
			//		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
					printf("%s\n",report_time(1));	
				};
/*
				if ((GlobalStepsCounter % SWIM_FREQUENCY == 0)) 
					// Swim involves ~9x8 advections so we may end up spending more time than on steps??
				{	
					count = 0;					
					do
					{
						pX->CopyMesh(pXnew);
						pXnew->SwimMesh(pX); // Note that SwimMesh does not call redelaunerize

						// will need to look carefully into mechanics of what is being populated by what.

						pXnew->RefreshVertexNeighboursOfVerticesOrdered();
						pXnew->Redelaunerize(true);
						pXnew->RefreshVertexNeighboursOfVerticesOrdered();
						
						pX->RepopulateCells(pXnew,ALL_VARS); 
						pXnew->AverageVertexPositionsAndInterpolate(pX,false);
						
						// Az (A and phi) handling is crucial.
						// Every time we rotate systems we should be interpolating Az.
						// We should interpolate B as well as A since it is not repopulated on vertices.

						// Flip systems
						temp = pX;
						pX = pXnew;
						pXnew = temp;
				
						pX->CreateVertexAveragingCoefficients();	
						pX->RecalculateVertexVariables();
					
						pX->SurveyCellMassStats(&mass_avg,&mass_SD,&mass_min,&mass_max,&iMin);
						printf("\nmass avg: %1.10E   SD: %1.10E\n min: %1.10E  max: %1.10E iMin: %d \n",
							mass_avg, mass_SD, mass_min, mass_max, iMin);

						count++;
					} while (((mass_SD > 0.2*mass_avg) || (mass_min < 0.2*mass_avg)) && (count < SWIM_COUNT));
					
					RefreshGraphs(*pX); // sends data to graphs AND renders them
					Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );

				}; // whether % SWIM_FREQ == 0
*/
				if (steps_remaining > 0){
					SetTimer(hWnd,1,DELAY_MILLISECS,NULL);
					printf("Waiting %d milliseconds to allow user input.\n",DELAY_MILLISECS);
				} else {
					printf("Run completed.\n");
					
					sprintf(buf1000, "autosave%d.dat", GlobalStepsCounter);
					pX->Save(buf1000); 
					printf("saved as %s\n",buf1000);
				};			
			} else {
				printf("steps_remaining = 0.\n");
			};
			
			if (GlobalStepsCounter % (AVI_FILE_PINCHOFF_FREQUENCY * GRAPHICS_FREQUENCY) == 0)
			{
				for (i = 0; i < NUMAVI; i++)
				{
					// now have to pinch out avi file and make a new one
					sprintf(buf1000, "%s%s_%d.avi", FOLDER, szAvi[i], GlobalStepsCounter);
					CloseAvi(hAvi[i]);
					hAvi[i] = CreateAvi(buf1000, AVIFRAMEPERIOD, NULL);
					SetAviVideoCompression(hAvi[i],dib,&opts,false,hWnd); 
				};
			};
			
			// Auto-save system:
			if (GlobalStepsCounter % DATA_SAVE_FREQUENCY == 0)
			{
				// now have to save data

				sprintf(buf1000, DATAFILENAME "%d.dat", GlobalStepsCounter);
				pX->Save(buf1000); 
				// Do regular text save as well: ?

			} else {
				pX->Save(AUTOSAVE_FILENAME); 
				pX->SaveText("autosave.txt");
			};
		};
		break;
		
	case WM_KEYDOWN:
		
		switch(wParam)
		{
		case 'W':
			GlobalEye.z += 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;
		case 'S':
			GlobalEye.z -= 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;
		case 'A':
			GlobalEye.x -= 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;
		case 'D':
			GlobalEye.x += 0.8f;		
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;
		case 'E':
			GlobalEye.y += 0.8f;		
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;
		case 'C':
			GlobalEye.y -= 0.8f;		
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z);
			break;

		case 'V':
			GlobalLookat.z -= 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z,GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'R':
			GlobalLookat.z += 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z,GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'F':
			GlobalLookat.x -= 0.4f;		
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z,GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'G':
			GlobalLookat.x += 0.4f;		
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z,GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'T':
			GlobalLookat.y += 0.4f;		
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'B':
			GlobalLookat.y -= 0.4f;		
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			break;
		case 'Q':
			GlobalCutaway = !GlobalCutaway;
			break;
		case 'Y':
		case '<':
			GlobalEye.x = -10.4; GlobalEye.y = 16.4; GlobalEye.z = 44.0;
			GlobalLookat.x = -3.6; GlobalLookat.y = 3.0; GlobalLookat.z = 72.2;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x,GlobalEye.y,GlobalEye.z,GlobalLookat.x,GlobalLookat.y,GlobalLookat.z);
			
			GlobalPlanEye.x = 7.1; GlobalPlanEye.y = 11.5; GlobalPlanEye.z = 71.35;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);

			break;
		case '_':
		case '-':
		case '>':
			GlobalPlanEye.x = 7.0; GlobalPlanEye.y = 14.0; GlobalPlanEye.z = 71.0;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);
			break;

		case 'U':
			GlobalPlanEye.z += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);
			break;
		case 'J':
			GlobalPlanEye.z -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);
			break;
		case 'H':
			GlobalPlanEye.x -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);			
			break;
		case 'K':
			GlobalPlanEye.x += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);			
			break;
		case 'I':
			GlobalPlanEye.y *= 1.25f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);			
			break;
		case 'M':
			GlobalPlanEye.y *= 0.8f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x,GlobalPlanEye.y,GlobalPlanEye.z);
			break;
		case 'N':	
			GlobalboolDisplayMeshWireframe = !GlobalboolDisplayMeshWireframe;
			//Graph1.boolDisplayMeshWireframe = (!(Graph1.boolDisplayMeshWireframe));
			break;
		case '9':
			GlobalRenderLabels = false;
			break;
		case '5':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 0;// iTri
			break;
		case '8':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 1;//T
			break;
		case '7':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 2;//v
			break;
		case '6':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 3;	//n
			break;
		case '1':
			GlobalColoursPlanView = 1;//v
			break;
		case '4':
			GlobalColoursPlanView = 0;//nothing
			break;
		case '2':
			GlobalColoursPlanView = 2;//T
			break;
		case '0':
			steps_remaining = 0;
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
							
		};
		
		PlanViewGraphs1(*pX);

		//RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
			
		break;
	case WM_PAINT:
		// Not sure, do we want to do this?
		//RefreshGraphs(*pX); // sends data to graphs AND renders them
		GetUpdateRect(hWnd, &rect, FALSE);
		Direct3D.pd3dDevice->Present( &rect, &rect, NULL, NULL );
		ValidateRect(hWnd, NULL);
		break;
	case WM_DESTROY:
		DeleteObject(dib);
		DeleteDC(dibdc);
		for (i = 0; i < NUMAVI; i++)
			CloseAvi(hAvi[i]);

		_controlfp_s(0, cw, _MCW_EM); // Line A
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}


// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}


INT_PTR CALLBACK SetupBox(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	char buffer[2048];
	char string[1024];
	real newh;
	
	switch (message)
	{
	case WM_INITDIALOG:
		sprintf(buffer,"New h? (present = %1.10E)",h);
		if (GlobalSwitchBox)
			SetDlgItemText(hDlg,IDC_STATIC,buffer);
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK)
		{
			// try to read data from edit control:
			GetDlgItemText(hDlg,IDC_EDIT1,buffer,2048);
			if (GlobalSwitchBox == 0)
			{
				// 
				steps_remaining = atoi(buffer);
				if (steps_remaining >= 0)
				{
					EndDialog(hDlg, LOWORD(wParam));
				} else {
					MessageBox(NULL,"incorrect value","Enter a nonnegative integer.",MB_OK);
				};
			} else {
				newh = atof(buffer);
				if (newh > 0.0)
				{
					EndDialog(hDlg, LOWORD(wParam));
					sprintf(string,"h = %1.10E\n",newh);
					h = newh;
					MessageBox(NULL,string,"New value of h",MB_OK);
				} else {
					MessageBox(NULL,"no good","Negative h entered",MB_OK);
				};
			};
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}


#ifndef NOCUDA
void TriMesh::Setup_Data_for_CUDA( Systdata * pSystdata )
{
	// The idea is just to repackage TriMesh data into the desired format.
	
	// FOR NOW -- not sure if we initialised or what so who cares
	// just to get speed test.


	memset(pSystdata->p_phi,0,sizeof(f64)*numVertices);
	memset(pSystdata->p_phidot,0,sizeof(f64)*numVertices);
	memset(pSystdata->p_A,0,sizeof(f64_vec3)*numVertices);
	memset(pSystdata->p_Adot,0,sizeof(f64_vec3)*numVertices);

	pSystdata->EzTuning = this->EzTuning.x[0];

	long izNeigh[128];
	long i;
	long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Writing on phi,A
		pSystdata->p_phi[iVertex] = pVertex->phi;
		pSystdata->p_A[iVertex] = pVertex->A;
		
		pSystdata->p_Adot[iVertex] = pVertex->Adot;
		pSystdata->p_B[iVertex] = pVertex->B;

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		// phidot IS MISSING from Vertex class !!!!
		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		pSystdata->p_nT_neut[iVertex].n = pVertex->Neut.mass/pVertex->AreaCell;
		pSystdata->p_nT_neut[iVertex].T = pVertex->Neut.heat/pVertex->Neut.mass;
		pSystdata->p_v_neut[iVertex] = pVertex->Neut.mom/pVertex->Neut.mass;

		pSystdata->p_nT_ion[iVertex].n = pVertex->Ion.mass/pVertex->AreaCell;
		pSystdata->p_nT_ion[iVertex].T = pVertex->Ion.heat/pVertex->Ion.mass;
		pSystdata->p_v_ion[iVertex] = pVertex->Ion.mom/pVertex->Ion.mass;
		
		pSystdata->p_nT_elec[iVertex].n = pVertex->Elec.mass/pVertex->AreaCell;
		pSystdata->p_nT_elec[iVertex].T = pVertex->Elec.heat/pVertex->Elec.mass;
		pSystdata->p_v_elec[iVertex] = pVertex->Elec.mom/pVertex->Elec.mass;
		
		pSystdata->p_area[iVertex] = pVertex->AreaCell;
		if (pVertex->AreaCell == 0.0) {
			printf(" area=0 %d ",iVertex);
		}

		long neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		pSystdata->p_info[iVertex].flag = (short)(pVertex->flags);
		pSystdata->p_info[iVertex].neigh_len = (short)neigh_len;
		pSystdata->p_info[iVertex].pos = pVertex->pos;

		memcpy(pSystdata->pIndexNeigh + MAXNEIGH_d*iVertex,izNeigh,sizeof(long)*neigh_len);

		if (neigh_len > MAXNEIGH_d) {
			printf("Too many neighbours iVertex %d : %d \n",iVertex,neigh_len);
			getch();
		};

		for (i = 0; i < neigh_len; i++)
		{
			// For each neighbour assess its periodic status relative to this.
			
			// A simple test will do initially:
			Vertex * pNeigh = X + izNeigh[i];
			char PBC = 0;
			if ((pNeigh->pos.x/pNeigh->pos.y > GRADIENT_X_PER_Y*0.5)
				&&
				(pVertex->pos.x/pVertex->pos.y < -0.5*GRADIENT_X_PER_Y))
			{
				PBC = ANTICLOCKWISE; 
				// " ANTICLOCKWISE means neigh needs to be rotated anticlockwise "
			};
			if ((pNeigh->pos.x/pNeigh->pos.y < -GRADIENT_X_PER_Y*0.5)
				&&
				(pVertex->pos.x/pVertex->pos.y > 0.5*GRADIENT_X_PER_Y))
			{
				PBC = CLOCKWISE;
			};
			pSystdata->pPBCneigh[MAXNEIGH_d*iVertex + i] = PBC;
			
			if (iVertex == 14336) {				
				printf("14336 i %d %d PBC %d pos %1.4E %1.4E \n",
					i,izNeigh[i],(int)PBC,pNeigh->pos.x,pNeigh->pos.y);
			}
		}

		// And that's all it wanted.

		++pVertex;
	};
	
}
void TriMesh::Suck_Data_from_CUDA( Systdata * pSystdata )
{
	this->EzTuning.x[0] = pSystdata->EzTuning;

	long izNeigh[128];
	long i;
	long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Writing on phi,A
		pVertex->phi = pSystdata->p_phi[iVertex];
		pVertex->A = pSystdata->p_A[iVertex];
		
		pVertex->Adot = pSystdata->p_Adot[iVertex];
		pVertex->B = pSystdata->p_B[iVertex];

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//      phidot IS MISSING from Vertex class !!!!
		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		// Can we create graphics directly from Systdata object? Haven't finalised structs yet.

		pVertex->Neut.mass = pSystdata->p_nT_neut[iVertex].n*pSystdata->p_area[iVertex];
		pVertex->Neut.heat = pVertex->Neut.mass*pSystdata->p_nT_neut[iVertex].T;
		pVertex->Neut.mom = pVertex->Neut.mass*pSystdata->p_v_neut[iVertex];

		// CAN we do that?

		pVertex->Ion.mass = pSystdata->p_nT_ion[iVertex].n*pSystdata->p_area[iVertex];
		pVertex->Ion.heat = pVertex->Ion.mass*pSystdata->p_nT_ion[iVertex].T;
		pVertex->Ion.mom = pVertex->Ion.mass*pSystdata->p_v_ion[iVertex];
		
		pVertex->Elec.mass = pSystdata->p_nT_elec[iVertex].n*pSystdata->p_area[iVertex];
		pVertex->Elec.heat = pVertex->Elec.mass*pSystdata->p_nT_elec[iVertex].T;
		pVertex->Elec.mom = pVertex->Elec.mass*pSystdata->p_v_elec[iVertex];
		
		if (iVertex == 13202) {

			printf("diff of mass %1.5E \n",
				(pVertex->Ion.mass - pVertex->Elec.mass));

		};

		pVertex->AreaCell = pSystdata->p_area[iVertex];
		if (pVertex->AreaCell == 0.0) {
			printf(" area=0 %d ",iVertex);
		}

		pVertex->pos = pSystdata->p_info[iVertex].pos;

		++pVertex;
	};
	
}
#endif
