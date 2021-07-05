#define real double
#define f64 double
 
#define HISTORY										4
     
#include <stdlib.h>
#include <stdio.h>
#include "lapacke.h"
  
// we must find out what causes graphics crash during SPECIES_ION

/* Auxiliary routines prototypes */
extern void print_matrix(char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda);
extern void print_int_vector(char* desc, lapack_int n, lapack_int* a);
 
extern void Go_visit_the_other_file();
extern void Setup_residual_array();
      
#include "headers.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <windows.h>
#include "resource.h"
#include "flags.h"
#include "FFxtubes.h"
//#include "cppconst.h"
#include "cuda_struct.h"
#include "constant.h"
#include "d3d.h"    
#include <d3dx9.h> 
#include <dxerr.h>
#include <commdlg.h>    // probably used by avi_utils
#include "surfacegraph_tri.h"
//#include "avi_utils.cpp"     // for making .avi
#include "kernel.h"

#include <mfapi.h>
#include <mfidl.h>
#include <Mfreadwrite.h>
#include <mferror.h>
#include <iostream>
#include <shlwapi.h>
#include <combaseapi.h>

#pragma comment(lib, "mfreadwrite")
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfuuid")

template <class T> void SafeRelease(T **ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = NULL;
	}
}


//=======================================================
// Declarations of functions:
   
void RefreshGraphs(TriMesh & X, const int iGraphsFlag);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	SetupBox(HWND, UINT, WPARAM, LPARAM);
extern f64 GetEzShape__(f64 r);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern void Zap_the_back();

extern f64 * temp_array_host;
extern OhmsCoeffs * p_OhmsCoeffs_host;
extern f64 * p_graphdata1_host,* p_graphdata2_host,* p_graphdata3_host, *p_graphdata4_host, *p_graphdata5_host, *p_graphdata6_host;
extern f64 * p_Tgraph_host[9];
extern f64 * p_accelgraph_host[12];
extern f64 * p_Ohmsgraph_host[20];

extern f64 * p_arelz_graph_host[12];

// Global variables:
// =================
//extern f64_vec3 * p_B_host;
extern f64 EzStrength_;
extern cuSyst cuSyst1, cuSyst2, cuSyst3;
extern D3D Direct3D;
extern f64 * p_temphost1, *p_temphost2,
*p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
 
extern __device__ f64 * p_LapCoeffself;
extern __device__ f64 * p_temp1;
extern __device__ long * p_longtemp;
extern __device__ f64 * p_Az, *p_LapAz;


float xzscale;

bool bCullNone = false;
bool bGlobalsave = false;
int GlobalSwitchBox = 0;
int iGlobalScratch;

real GlobalHeightScale;

int GlobalSpeciesToGraph = SPECIES_ION;
int GlobalWhichLabels = 0;
bool GlobalRenderLabels = false;
int GlobalColoursPlanView = 0;
bool GlobalBothSystemsInUse;

bool GlobalCutaway = true;

unsigned int cw; // control word for floating point hardware exception hiding

TriMesh * pX, *pXnew;
TriMesh X1, X2, X3, X4;
cuSyst cuSyst_host, cuSyst_host2;

D3DXVECTOR3 GlobalEye, GlobalLookat, GlobalPlanEye, GlobalPlanEye2, GlobalPlanLookat,
GlobalPlanLookat2, GlobalEye2, GlobalLookat2;

D3DXVECTOR3 newEye;
D3DXVECTOR3 newLookat;

IDirect3DSurface9* p_backbuffer_surface;

long steps_remaining, GlobalStepsCounter, steps_remaining_CPU;
real evaltime, h;

extern real GlobalIzElasticity;
FILE * massfile, *maxfile;

// Global Variables:
HINSTANCE hInst;   // current instance
				   // window vars:
HWND hWnd, hwndGraphics;
WNDCLASSEX wcex;
TCHAR szTitle[1024];					// The title bar text
TCHAR szWindowClass[1024];			// the main window class name

char Functionalfilename[1024];
int GlobalGraphSetting[8]; 
surfacegraph Graph[8]; // why was it 5? // 5th one can be whole thing.

float Historic_max[512][HISTORY]; // if max is falling, use historic maximum for graph.
float Historic_min[512][HISTORY];
int Historic_powermax[512];
int Historic_powermin[512]; // just store previous value only.

bool flaglist[NMINOR];

bool boolGlobalHistory, GlobalboolDisplayMeshWireframe;

// avi file -oriented variables
int const NUMAVI = 9;
//HAVI hAvi[NUMAVI + 1]; // does it work without OHMSLAW? //  OHMSLAW,
int const GraphFlags[NUMAVI] = { SPECIES_ION, OVERALL, JZAZBXYEZ, ONE_D, IONIZEGRAPH,
				DTGRAPH, ACCELGRAPHS, OHMS2, ARELZ};

WCHAR szmp4[NUMAVI][128] = { L"Elec",L"Total",L"JzAzBxy",L"Test", 
L"Ionize", L"dT", L"Accel",	L"Ohms", L"arelz"};

//AVICOMPRESSOPTIONS opts;
int counter;
HBITMAP surfbit, dib;
HDC surfdc, dibdc;
LPVOID lpvBits;
BITMAPINFO bitmapinfo;

DWORD dwBits[VIDEO_HEIGHT*VIDEO_WIDTH];

f64 graphdata[20][10000]; 
f64 graph_r[10000];
int numgraphs = 4;
int num_graph_data_points = 10000;
f64 maximum[20];
f64 truemax[20]; 

extern TriMesh * pTriMesh;


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
		timenow = ((double)(clock() - start) / (double)CLOCKS_PER_SEC);
		ops = (long)(clock() - start);
		/* create a null-terminated string */
		sprintf(timebuffer, "%6.4f sec.", timenow);
	};
	return &(timebuffer[0]);
};

f64 GetTriangleArea(f64_vec2 pos0, f64_vec2 pos1, f64_vec2 pos2)
{
	f64 area = 0.5*((pos0.x + pos1.x)*(pos1.y - pos0.y) + (pos1.x + pos2.x)*(pos2.y - pos1.y)
		+ (pos0.x + pos2.x)*(pos0.y - pos2.y));
	return fabs(area);
}


// Format constants
//const UINT32 VIDEO_WIDTH = 640;
//const UINT32 VIDEO_HEIGHT = 480;
const UINT32 VIDEO_FPS = 5;
const UINT64 VIDEO_FRAME_DURATION = 10 * 1000 * 1000 / VIDEO_FPS; // ?
const UINT32 VIDEO_BIT_RATE = 1048768;
const UINT32 VIDEO_PELS = VIDEO_WIDTH * VIDEO_HEIGHT;
const UINT32 VIDEO_FRAME_COUNT = 5;
// with 50 frames per nanosecond and 30 nanoseconds in file, it's 1500
// But to begin let's say 5?
const GUID   VIDEO_INPUT_FORMAT = MFVideoFormat_RGB24;

HRESULT InitializeSinkWriter(
	IMFSinkWriter **ppWriter, 
	DWORD *pStreamIndex, 
	LPCWSTR szFilename)
{
	*ppWriter = NULL;
	*pStreamIndex = NULL;

	IMFSinkWriter   *pSinkWriter = NULL;
	IMFMediaType    *pMediaTypeOut = NULL;
	IMFMediaType    *pMediaTypeIn = NULL;
	DWORD           streamIndex;
	
	HRESULT hr = MFCreateSinkWriterFromURL(szFilename, NULL, NULL, &pSinkWriter);

	// Set the output media type.
	if (SUCCEEDED(hr)) 
		hr = MFCreateMediaType(&pMediaTypeOut);	
	if (SUCCEEDED(hr))
		hr = pMediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeOut->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
	// whereas webcam capture sample says WMMEDIASUBTYPE_I420
	
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeOut->SetUINT32(MF_MT_AVG_BITRATE, VIDEO_BIT_RATE);
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeSize(pMediaTypeOut, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);
	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeRatio(pMediaTypeOut, MF_MT_FRAME_RATE, VIDEO_FPS, 1);
	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeRatio(pMediaTypeOut, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
	if (SUCCEEDED(hr)) 
		hr = pSinkWriter->AddStream(pMediaTypeOut, &streamIndex);
	
	// Set the input media type.
	if (SUCCEEDED(hr)) 
		hr = MFCreateMediaType(&pMediaTypeIn);	
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeIn->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);	
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeIn->SetGUID(MF_MT_SUBTYPE, VIDEO_INPUT_FORMAT);
	if (SUCCEEDED(hr)) 
		hr = pMediaTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive); 
		
	// should that be 0 ? 

	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeSize(pMediaTypeIn, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);
	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeRatio(pMediaTypeIn, MF_MT_FRAME_RATE, VIDEO_FPS, 1);
	if (SUCCEEDED(hr)) 
		hr = MFSetAttributeRatio(pMediaTypeIn, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
	if (SUCCEEDED(hr)) 
		hr = pSinkWriter->SetInputMediaType(streamIndex, pMediaTypeIn, NULL);
	
	// Tell the sink writer to start accepting data.
	if (SUCCEEDED(hr)) {
		hr = pSinkWriter->BeginWriting();
	}

	// Return the pointer to the caller.
	if (SUCCEEDED(hr)) {
		*ppWriter = pSinkWriter;
		(*ppWriter)->AddRef();
		*pStreamIndex = streamIndex;
	}

	SafeRelease(&pSinkWriter);
	SafeRelease(&pMediaTypeOut);
	SafeRelease(&pMediaTypeIn);
	return hr;
}

HRESULT WriteFrame(
	IMFSinkWriter *pWriter,
	DWORD streamIndex,
	const LONGLONG& rtStart        // Time stamp.
)
{
	IMFSample *pSample = NULL;
	IMFMediaBuffer *pBuffer = NULL;

	// SHOULD THIS BE 3 * ?

	const LONG cbWidth = 3 * VIDEO_WIDTH; // 4 bytes --- why?

	// so cbWidth is width in bytes

	const DWORD cbBuffer = cbWidth * VIDEO_HEIGHT;
	BYTE *pData = NULL;
	// Create a new memory buffer.
	HRESULT hr = MFCreateMemoryBuffer(cbBuffer, &pBuffer);
	// Lock the buffer and copy the video frame to the buffer.
	if (SUCCEEDED(hr))
		hr = pBuffer->Lock(&pData, NULL, NULL);
	
	if (SUCCEEDED(hr))
		hr = MFCopyImage(
			pData,                      // Destination buffer.
			cbWidth,                    // Destination stride.
			(BYTE *)lpvBits,//(BYTE*)videoFrameBuffer,    // First row in source image.
			cbWidth,                    // Source stride.
			cbWidth,                    // Image width in bytes.
			//I added x 3
			VIDEO_HEIGHT                // Image height in pixels.
		);
	
	if (pBuffer) pBuffer->Unlock();
	
	// Set the data length of the buffer.
	if (SUCCEEDED(hr))
		hr = pBuffer->SetCurrentLength(cbBuffer);

	// Create a media sample and add the buffer to the sample.
	if (SUCCEEDED(hr))
		hr = MFCreateSample(&pSample);
	if (SUCCEEDED(hr))
		hr = pSample->AddBuffer(pBuffer);
	// Set the time stamp and the duration.
	if (SUCCEEDED(hr))
		hr = pSample->SetSampleTime(rtStart);
	if (SUCCEEDED(hr))
		hr = pSample->SetSampleDuration(VIDEO_FRAME_DURATION);

	// Send the sample to the Sink Writer.
	if (SUCCEEDED(hr))
		hr = pWriter->WriteSample(streamIndex, pSample);

	SafeRelease(&pSample);
	SafeRelease(&pBuffer);
	return hr;
}


void TriMesh::CalculateTotalGraphingData()
{
	/*long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
	if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
	{
	pVertex->n = (pVertex->Neut.mass + pVertex->Ion.mass) / pVertex->AreaCell;
	pVertex->v = (m_n*pVertex->Neut.mom + m_ion * pVertex->Ion.mom + m_e * pVertex->Elec.mom) /
	(m_n*pVertex->Neut.mass + m_ion * pVertex->Ion.mass + m_e * pVertex->Elec.mass);
	pVertex->T = (pVertex->Neut.heat + pVertex->Ion.heat + pVertex->Elec.heat) /
	(pVertex->Neut.mass + pVertex->Ion.mass + pVertex->Elec.mass);
	pVertex->Temp.x = pVertex->Ion.mass / (pVertex->Neut.mass + pVertex->Ion.mass);
	};
	++pVertex;
	}*/
}

void TriMesh::Setup_J()
{
	/*long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
	if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
	{
	pVertex->Temp = q * (pVertex->Ion.mom - pVertex->Elec.mom) / pVertex->AreaCell;
	}
	else {
	memset(&(pVertex->Temp), 0, sizeof(Vector3));
	}
	++pVertex;
	}*/
}

void surfacegraph::DrawSurface(const char * szname,
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
	real * temprealptr = (real *)(pX->pData);
	long offset = var_ptr_0 - temprealptr;
	long offset_c = var_ptr_c - temprealptr;

	// Does shader always go with colour type?? yes I think.
	switch (colourflag) {
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
	case PPN_COLOUR:
		mhTech = mFX->GetTechniqueByName("ProportionTech"); // 1 = blue
		break;
	};

	// Usual settings:
	//if (GlobalGraphSetting[i] != GRAPH_NONE) {

	this->boolDisplayShadow = true;
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
	}
	else {
		this->boolDisplayKeyButton = false;
	};
	//int const FLAG_COLOUR_MESH = 0;
	//int const FLAG_SEGUE_COLOUR = 1;
	//int const FLAG_VELOCITY_COLOUR = 2;
	//int const FLAG_CURRENT_COLOUR = 3;
	//int const FLAG_AZSEGUE_COLOUR = 4;
	//int const FLAG_IONISE_COLOUR = 5;

	this->SetDataWithColour(*pX,
		colourflag, heightflag, // apparently it's that way round
		offset, offset_c,
		code);

	printf("DrawSurface code %d : calling Render:\n", code);
	if (this->bDisplayTimestamp) {
		sprintf(buff, "%6.2f ns", evaltime*1.0e9);
		this->Render(szname, false, pX, buff);
	}
	else {
		this->Render(szname, false, pX);
	};

}

void Draw1Dgraph(int iWhichGraph, int flag)
{
	float const MAXX = 11.0f;
	float const MAXY = 6.0f;
	float const YADJUST = -2.8f;

	char graphname[4][128] = { "Azdot","Azdotdot","Lap Az","-4pi/c Jz" };
	char Tgraphname[9][128] = { "conduction","ionization","viscosity","frictional","interspecies","dTe/dt total","compressive" ,
		"DnT","undefined" };
	char accelgraphname[9][128] = { "dvy/dt total", "v x B", "pressure", "neutral soak","viscosity", "ionization", "advection","grad_y Az" };
	char Ohmsgraphname[20][128] = { "elastic effective fric coeff", "ionization effective fric coeff",
	"thermal pressure y", "electromotive aez-aiz", "thermal force aiz-aez", "v-response T_zy", "v-response T_zz",
		"T_zy * thermal pressure y", "T_zz * electromotive", "T_zz * thermal force", "Predicted vez-viz",
	"Difference: prediction-vrelzk","vrelzk progress",
	"viscous aez-aiz","Predicted Jz","Conductivity sigma_zy","Conductivity sigma_zz",
	"sigma_zz * -electromotive", "Difference: Jz prediction-Jz","$$$" };

	char arelzgraphname[12][128] = { "arelz", "MAR_ion contribution", "MAR_elec contribution",
	    "Ez_ext effect", "dAz/dt effect","v x B effect", "thermal force effect", "friction to neutrals",
		"friction_ei", "sum of effects", "difference (error)"};


	char buffer[256];
	float x, y, z;
	float zeroplane = 0.0f;
	D3DXMATRIXA16 matWorld;
	vertex1 linedata[10000];
	vertex1 linedata2[12];
	int iGraph;

	D3DCOLOR colourlist[20];
	char namelist[20][256];
	bool bAlternating[20];
	int numgraphs;
	memset(bAlternating, 0, sizeof(bool) * 20);
	if (flag == ONE_D) {
		numgraphs = 4;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i],"%s   : graph max&min = +- %1.3E", 
				graphname[i], maximum[i]);		
		colourlist[0] = 0xff000000;
		colourlist[1] = 0xff0022ff;
		colourlist[2] = 0xffff0055;
		colourlist[3] = 0xff22ff00;
	};
	if (flag == DTGRAPH) {
		numgraphs = 8;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i], "%s  : graph max&min = +- %1.3E",
				Tgraphname[i], maximum[i]);
		colourlist[0] = 0xffffaa00; // conduction: orange red
		colourlist[1] = 0xff0000ff; // ionization: royal blue
		colourlist[2] = 0xff009999; // viscosity: aqua
		colourlist[3] = 0xffd500ff; // resistive: heliotrope
		colourlist[4] = 0xff00ff00; // soak: green
		colourlist[5] = 0xff000000; // total
		colourlist[6] = 0xff906545; // compressive: brown
	};
	if (flag == ACCELGRAPHS)
	{
		numgraphs = 8;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i], "%s   : graph max&min = +- %1.3E",
				accelgraphname[i], maximum[i]);
		colourlist[0] = 0xff000000; // total: 
		colourlist[1] = 0xffd500ff; // vxB: heliotrope
		colourlist[2] = 0xffff2200; // pressure: red
		colourlist[3] = 0xff00ff33; // soak:   use green
		colourlist[4] = 0xff009999; // viscosity: aqua
		colourlist[5] = 0xff0000ff; // ionization : royal blue
		colourlist[6] = 0xff906545; // advection :  brown
		colourlist[7] = 0xffeecd00; // grady_Az : olive?
	}

	if (flag == ARELZ)
	{
		numgraphs = 11;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i], "%s   : graph max&min = +- %1.3E",
				arelzgraphname[i], maximum[i]);
		colourlist[0] = 0xff000000; // total: 
		colourlist[1] = 0xff009999; // ion visc : aqua
		colourlist[2] = 0xffeecd00; // elec visc : olive
		colourlist[3] = 0xffdada66; // electromotive
		colourlist[4] = 0xff4400ff; // inductive electromotive: indigo
		colourlist[5] = 0xffd500ff; // vxB: heliotrope
		colourlist[6] = 0xffff7700; // "thermal force effect"
		colourlist[7] = 0xff00ff33; // neutral soak :green
		colourlist[8] = 0xff00aa00; // dkgreen e-i friction
		colourlist[9] = 0xffff2299; // pink : sum
		colourlist[10] = 0xff000011; // navy 
		bAlternating[10] = true;		
	}

	if (flag == OHMS2) {
		numgraphs = 11;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i], "%s :grmax+- %1.3E own|max| %1.3E",
				Ohmsgraphname[i], maximum[i], truemax[i]);
		
		colourlist[0] = 0xffcc0033; // elastic fric coeff: maroon
		colourlist[1] = 0xff00aa00; // dkgreen ionization fric coeff
		colourlist[2] = 0xffff0000; // pressure: red
		colourlist[3] = 0xffda00ff; // electromotive: violet
		colourlist[4] = 0xffff7700; // thermal force: orange
		colourlist[5] = 0xff00aadd; bAlternating[5] = true; // Tzy
		colourlist[6] = 0xff0000ff; // Tzz
		colourlist[7] = 0xffff55aa; bAlternating[7] = true;
		colourlist[8] = 0xffda00ff; bAlternating[8] = true;
		colourlist[9] = 0xffffaa00; bAlternating[9] = true;
		colourlist[10] = 0xff000000;
	};

	if (flag == OHMSLAW) {
		numgraphs = 9;
		for (int i = 0; i < numgraphs; i++)
			sprintf(namelist[i], "%s :grmax+- %1.3E own|max| %1.3E",
				Ohmsgraphname[i+10], maximum[i+10], truemax[i+10]);
		colourlist[0] = 0xff000000;
		colourlist[1] = 0xffff3333; // red: difference
		colourlist[2] = 0xffaadd00; // lime yellow: progress
		colourlist[3] = 0xff009999; // viscosity: aqua
		colourlist[4] = 0xffd500ff; // prediction Jz
		colourlist[5] = 0xff00bb33; // green conductivity
		colourlist[6] = 0xff0022ff; // blue conductivity
		colourlist[7] = 0xffda00ff; bAlternating[7] = true; // sigma_zz Ez
		colourlist[8] = 0xff666666; // difference of prediction
	}

	f64 rmax = GRAPH1D_MAXR;
	if (flag == ONE_D) rmax = DOMAIN_OUTER_RADIUS;
	f64 rmin = DEVICE_RADIUS_INSULATOR_OUTER - 0.01;
	if (flag == ONE_D) rmin = INNER_A_BOUNDARY;


	Graph[iWhichGraph].SetEyeAndLookat(newEye, newLookat); // sets matView not matProj
	printf("Eye %f %f %f\n", newEye.x, newEye.y, newEye.z);
	Direct3D.pd3dDevice->SetViewport(&(Graph[iWhichGraph].vp));

	D3DXMatrixIdentity(&matWorld);
	//D3DXMatrixIdentity(&Graph[6].matProj); // ???????????????
	Direct3D.pd3dDevice->SetTransform(D3DTS_WORLD, &matWorld);
	Direct3D.pd3dDevice->SetTransform(D3DTS_VIEW, &(Graph[iWhichGraph].matView));
	Direct3D.pd3dDevice->SetTransform(D3DTS_PROJECTION, &(Graph[iWhichGraph].matProj));

	Direct3D.pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
		D3DCOLOR_XRGB(250, 255, 250), 1.0f, 0);

	if (SUCCEEDED(Direct3D.pd3dDevice->BeginScene()))
	{
		Direct3D.pd3dDevice->SetFVF(point_fvf);

		real theta = -HALFANGLE;
		real r = 3.44;

		linedata[0].x = -MAXX;
		linedata[0].z = 3.44*xzscale;
		linedata[0].y = YADJUST;
		linedata[0].colour = 0xff888888; // grey

		linedata[1].x = -linedata[0].x;
		linedata[1].y = YADJUST;
		linedata[1].z = linedata[0].z;
		linedata[1].colour = linedata[0].colour;

		Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, 1, linedata, sizeof(vertex1));

			//Graph[6].RenderLabel2(buffer,  // text
			//	MAXX*0.66f + 1.2f*(float)iGraph,
			//	0.0f,
		for (iGraph = 0; iGraph < numgraphs; iGraph++)
		{
			if (iGraph < 6) {
				linedata[0].x = -MAXX;
				linedata[0].z = 3.44*xzscale;
				linedata[0].y = MAXY + 4.0f - 0.9f*(float)iGraph;				
			} else {
				linedata[0].x = 0.8f;
				linedata[0].z = 3.44*xzscale;
				linedata[0].y = MAXY + 4.0f - 0.9f*(float)(iGraph-6);
			}
			linedata[1].x = linedata[0].x + 0.5f;
			linedata[1].y = linedata[0].y;
			linedata[1].z = linedata[0].z;
			linedata[2].x = linedata[0].x + 1.0f;
			linedata[2].y = linedata[0].y;
			linedata[2].z = linedata[0].z;
			linedata[0].colour = colourlist[iGraph];
			linedata[1].colour = linedata[0].colour;
			linedata[2].colour = linedata[0].colour;
			if (bAlternating[iGraph]) linedata[1].colour = 0xffffffff;

			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, 2, linedata, sizeof(vertex1));
			
			Graph[iWhichGraph].RenderLabel2(namelist[iGraph], linedata[2].x + 0.1f, linedata[1].y - 0.3f, linedata[1].z, 0,0xff000000, true);

			
			int asdf;
			
			if (flag != OHMSLAW) {
				for (asdf = 0; asdf < num_graph_data_points; asdf++)
				{
					linedata[asdf].x = (float)(MAXX - 2.0*MAXX*((graph_r[asdf] - rmin) /
						(rmax - rmin)));

					// map 0 to 0.0f, maximum[iGraph] to MAXY and -maximum[iGraph] to MINY
					// Decide on graph scales maximum[] in preceding bit of code
					linedata[asdf].y = YADJUST + (float)(MAXY*graphdata[iGraph][asdf] / maximum[iGraph]);
					linedata[asdf].z = 3.44f*xzscale;
					linedata[asdf].colour = colourlist[iGraph];
					if ((bAlternating[iGraph]) && (asdf % 3 == 1)) linedata[asdf].colour = 0xffffffff;
				};
				Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, num_graph_data_points - 1, linedata, sizeof(vertex1));
			} else {
				for (asdf = 0; asdf < num_graph_data_points; asdf++)
				{
					linedata[asdf].x = (float)(MAXX - 2.0*MAXX*((graph_r[asdf] - rmin) /
						(rmax - rmin)));
					// map 0 to 0.0f, maximum[iGraph] to MAXY and -maximum[iGraph] to MINY
					// Decide on graph scales maximum[] in preceding bit of code
					linedata[asdf].y = YADJUST + (float)(MAXY*graphdata[iGraph+10][asdf] / maximum[iGraph+10]);
					linedata[asdf].z = 3.44f*xzscale;
					linedata[asdf].colour = colourlist[iGraph];
					if ((bAlternating[iGraph]) && (asdf % 3 == 1)) linedata[asdf].colour = 0xffffffff;
				};
				Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, num_graph_data_points - 1, linedata, sizeof(vertex1));

			};
			//sprintf(buffer, "%2.2E", maximum[iGraph]);
			//Graph[6].RenderLabel2(buffer,  // text
			//	MAXX*0.66f + 1.2f*(float)iGraph,
			//	MAXY,
			//	linedata[0].z, 0, linedata[0].colour);
			//sprintf(buffer
			//	linedata[0].z, 0, linedata[0].colour);
			//sprintf(buffer, "-%2.2E", maximum[iGraph]);
			//Graph[6].RenderLabel2(buffer,  // text
			//	MAXX*0.66f + 1.2f*(float)iGraph,
			//	-MAXY,
			//	linedata[0].z, 0, linedata[0].colour);
		};
		
		// Vertical lines:
		for (int i = 0; i < 9; i++)
		{
			x = 0.16*(-r*xzscale + 2.0*r*xzscale*(((real)i) / 8.0));
			z = 3.44*xzscale;// (float)(cos(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;

			linedata[0].x = x; linedata[0].z = z;
			linedata[1].x = x; linedata[1].z = z;
			linedata[0].colour = 0xff220011;
			linedata[1].colour = 0xff220011;
			linedata[0].y = -6.8f + YADJUST;// GRAPHIC_MIN_Y - 1.0f;  
			linedata[1].y = YADJUST + (((i == 0) || (i == 8)) ? 6.0f : 0.0f);// GRAPHIC_MAX_Y + 2.5f;

			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, 1, linedata, sizeof(vertex1));

			sprintf(buffer, "%5.2f", rmin + (1.0 - ((real)i) / 8.0)*(rmax - rmin));
			Graph[iWhichGraph].RenderLabel2(buffer,  // text
				linedata[0].x,
				YADJUST - 7.6f,
				linedata[0].z, 0);

		};
		//DXChk(mFX->SetValue(mhEyePos, &Eye, sizeof(D3DXVECTOR3)));

		linedata[0].x = -0.16*r*xzscale;
		linedata[0].y = YADJUST;
		linedata[0].z = 3.44*xzscale;
		linedata[0].colour = 0xff000000; // 

		linedata[1].x = 0.16*r*xzscale;
		linedata[1].y = YADJUST;
		linedata[1].z = linedata[0].z;
		linedata[1].colour = linedata[0].colour;

		Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP, 1, linedata, sizeof(vertex1));

		Direct3D.pd3dDevice->EndScene();
	}
	else {
		printf("BeginScene failed!\n\n");
		getch();
	}
}

void Create1DGraphingData(TriMesh * pX, bool bTdata = false, bool bAcceldata = false,
	bool bOhmsData = false, bool b_arelz_data = false)
{
	// Takes p_temphost3,4,5,6 and turns them into graphdata[iGraph=0,1,2,3][]

	Vertex * pVertex, * pVert2;
	f64_vec2 pos, pos0, pos1, pos2;
	f64 dist0, dist1, dist2, wt0, wt1, wt2, wttotal, y0, y1, y2;
	int iGraph, asdf, iWhich, iCorner, tri_len, i;
	bool has_more, has_less, has_grad;
	Triangle * pTri;
	long izTri[MAXNEIGH];

	long VertexIndexArray[10000];

	num_graph_data_points = pX->GetVertsRightOfCutawayLine_Sorted(VertexIndexArray, graph_r, true);
	
	printf("Xebeques furious\n Number of points %d\n", num_graph_data_points);

	memset(maximum, 0, sizeof(f64) * 20);

	// Method used in Render routine looks quite reasonable: find tri that crosses cutaway,
	// use some kind of interp on tri. But we need to use values from p_temphost array not a graph position.

	for (asdf = 0; asdf < num_graph_data_points; asdf++)
	{
	//	if (asdf % 10 == 0) printf("<");
	//	printf("%d ; ", VertexIndexArray[asdf]);
		pVertex = pX->X + VertexIndexArray[asdf];

		// We want the tri directly to the left of it, through which (-1,0) passes.
		// 1.Get these vertex indices
		// which tri contains a point which is further and a point less far?

		real rr = pVertex->pos.x*pVertex->pos.x + pVertex->pos.y*pVertex->pos.y;
		iWhich = -1;
		tri_len = pVertex->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			pTri = pX->T + izTri[i];
			has_more = false; has_less = false; has_grad = false;
			for (iCorner = 0; iCorner < 3; iCorner++)
			{
				pVert2 = pTri->cornerptr[iCorner];
				if (pVert2 != pVertex)
				{
					if (pVert2->pos.x*pVert2->pos.x + pVert2->pos.y*pVert2->pos.y > rr)
					{
						has_more = true;
					}
					else {
						has_less = true;
					};
				};
				if (pVert2->pos.x / pVert2->pos.y < pVertex->pos.x / pVertex->pos.y)
					has_grad = true;
			};

			if (has_more && has_less && has_grad)
			{
				iWhich = i;
			}
		};

		if (iWhich == -1) {// give up, do nothing} 
			printf("gave up. %d \n", VertexIndexArray[asdf]);
			graphdata[0][asdf] = 0.0;
			graphdata[1][asdf] = 0.0;
			graphdata[2][asdf] = 0.0;
			graphdata[3][asdf] = 0.0;
		} else {
			pTri = pX->T + izTri[iWhich];
			while ((pTri->u8domain_flag != DOMAIN_TRIANGLE) && (iWhich >= 0)) {
				pTri = pX->T + izTri[iWhich];
				iWhich--;
			};
			iWhich++;

			// we are needing to adjust graph_r and interp graphdata

			pos.y = pVertex->pos.y;
			pos.x = pVertex->pos.y*CUTAWAYANGLE; // can leave graph_r undisturbed

			pos0 = pTri->cornerptr[0]->pos;
			pos1 = pTri->cornerptr[1]->pos;
			pos2 = pTri->cornerptr[2]->pos;

			// if one sits at the CUTAWAYANGLE then we can get dist == 0.

			dist0 = sqrt((pos0 - pos).dot(pos0 - pos));
			dist1 = sqrt((pos1 - pos).dot(pos1 - pos));
			dist2 = sqrt((pos2 - pos).dot(pos2 - pos));
			
			if (dist0 == 0.0) {
				wt0 = 1.0; wt1 = 0.0; wt2 = 0.0;
			}
			else {
				if (dist1 == 0.0) {
					wt0 = 0.0; wt1 = 1.0; wt2 = 0.0;
				}
				else {
					if (dist2 == 0.0) {
						wt0 = 0.0; wt1 = 0.0; wt2 = 1.0;
					} else {

						wt0 = 1.0f / dist0;
						wt1 = 1.0f / dist1;
						wt2 = 1.0f / dist2;
						wttotal = wt0 + wt1 + wt2;
						wt0 /= wttotal;
						wt1 /= wttotal;
						wt2 /= wttotal;
						// Not a great way it has to be said.
					}
				}
			}


			if ((bTdata == false) && (bAcceldata == false) && (bOhmsData == false)
				&& (b_arelz_data == false)) {
				y0 = p_temphost3[(pTri->cornerptr[0] - pX->X) + BEGINNING_OF_CENTRAL];
				y1 = p_temphost3[(pTri->cornerptr[1] - pX->X) + BEGINNING_OF_CENTRAL];
				y2 = p_temphost3[(pTri->cornerptr[2] - pX->X) + BEGINNING_OF_CENTRAL];
				graphdata[0][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
				if (fabs(graphdata[0][asdf]) > maximum[0]) maximum[0] = fabs(graphdata[0][asdf]);

				if (numgraphs > 1) {
					y0 = p_temphost4[(pTri->cornerptr[0] - pX->X) + BEGINNING_OF_CENTRAL];
					y1 = p_temphost4[(pTri->cornerptr[1] - pX->X) + BEGINNING_OF_CENTRAL];
					y2 = p_temphost4[(pTri->cornerptr[2] - pX->X) + BEGINNING_OF_CENTRAL];
					graphdata[1][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
					if (fabs(graphdata[1][asdf]) > maximum[1]) maximum[1] = fabs(graphdata[1][asdf]);
				};

				if (numgraphs > 2) {
					y0 = p_temphost5[(pTri->cornerptr[0] - pX->X) + BEGINNING_OF_CENTRAL];
					y1 = p_temphost5[(pTri->cornerptr[1] - pX->X) + BEGINNING_OF_CENTRAL];
					y2 = p_temphost5[(pTri->cornerptr[2] - pX->X) + BEGINNING_OF_CENTRAL];
					graphdata[2][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
					if (fabs(graphdata[2][asdf]) > maximum[2]) maximum[2] = fabs(graphdata[2][asdf]);
				};
				if (numgraphs > 3) {
					y0 = p_temphost6[(pTri->cornerptr[0] - pX->X) + BEGINNING_OF_CENTRAL];
					y1 = p_temphost6[(pTri->cornerptr[1] - pX->X) + BEGINNING_OF_CENTRAL];
					y2 = p_temphost6[(pTri->cornerptr[2] - pX->X) + BEGINNING_OF_CENTRAL];
					graphdata[3][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
					if (fabs(graphdata[3][asdf]) > maximum[3]) maximum[3] = fabs(graphdata[3][asdf]);
				}
			} else {
				// go through from 0 = conduction to 5 = dTe/dt itself
				// we have missed out compressive...

				if (bTdata) {
					for (int j = 0; j < 8; j++)
					{
						y0 = p_Tgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_Tgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_Tgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[j][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
						if (fabs(graphdata[j][asdf]) > maximum[j]) maximum[j] = fabs(graphdata[j][asdf]);
					}
				} else {
					if (bAcceldata) {
						int j;
						j = 1; // total
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[0][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
						
						j = 3; // vxB
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[1][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
						j = 5; // pressure
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[2][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
						j = 6; // neutral soak
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[3][asdf] = wt0*y0 + wt1*y1 + wt2*y2;

						j = 8; // viscosity
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[4][asdf] = wt0*y0 + wt1*y1 + wt2*y2;

						j = 9; // ionization
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[5][asdf] = wt0*y0 + wt1*y1 + wt2*y2;

						j = 10; // advection
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[6][asdf] = wt0*y0 + wt1*y1 + wt2*y2;

						// works if comment here

					//	printf("%d ", asdf);
						for (int j = 0; j < 7; j++)
						{ 
							//		printf("%d", j);
							if (fabs(graphdata[j][asdf]) > maximum[0]) {
								maximum[0] = fabs(graphdata[j][asdf]);
								//			printf("maximum %1.9E\n", maximum[0]);
							}
						}

						// does it work if comment here? no

						j = 11; // grad_y Az
						y0 = p_accelgraph_host[j][(pTri->cornerptr[0] - pX->X)];
						y1 = p_accelgraph_host[j][(pTri->cornerptr[1] - pX->X)];
						y2 = p_accelgraph_host[j][(pTri->cornerptr[2] - pX->X)];
						graphdata[7][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
						if (fabs(graphdata[7][asdf]) > maximum[7]) maximum[7] = fabs(graphdata[7][asdf]);
					} else {
						if (bOhmsData) {
							int j;
							for (j = 0; j < 19; j++) {
								y0 = p_Ohmsgraph_host[j][(pTri->cornerptr[0] - pX->X)];
								y1 = p_Ohmsgraph_host[j][(pTri->cornerptr[1] - pX->X)];
								y2 = p_Ohmsgraph_host[j][(pTri->cornerptr[2] - pX->X)];
								graphdata[j][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
								if ((pos.y < 4.6) && (pos.y > 3.44) && (fabs(graphdata[j][asdf]) > maximum[j])) maximum[j] = fabs(graphdata[j][asdf]);
							};
						} else {
							int j;
							for (j = 0; j < 12; j++) {
								y0 = p_arelz_graph_host[j][(pTri->cornerptr[0] - pX->X)];
								y1 = p_arelz_graph_host[j][(pTri->cornerptr[1] - pX->X)];
								y2 = p_arelz_graph_host[j][(pTri->cornerptr[2] - pX->X)];
								graphdata[j][asdf] = wt0*y0 + wt1*y1 + wt2*y2;
								if ((pos.y < 4.8) && (pos.y > 3.44) && (fabs(graphdata[j][asdf]) > maximum[j])) maximum[j] = fabs(graphdata[j][asdf]);
							};
						};
					};
				};
			};
		}; // found triangle		
	}; // asdf	
	if ((bTdata == false) && (bAcceldata == false) && (bOhmsData == false) && (b_arelz_data == false)) {		
		maximum[3] = max(maximum[3], maximum[2]);
		maximum[2] = maximum[3];
	} else {
		// for dT graphs, let maximum be overall
		if (bTdata) {
			for (int j = 1; j <= 6; j++)
				maximum[j] = max(maximum[j], maximum[j - 1]);
			for (int j = 5; j >= 0; j--)
				maximum[j] = maximum[j + 1];
		} else {
			if (bAcceldata) {
				for (int j = 1; j < 7; j++)
					maximum[j] = maximum[0];
			} else {
				if (bOhmsData) {
					memcpy(truemax, maximum, sizeof(f64) * 20);

					// use max 0 and 1 combined:
					f64 temp = max(maximum[0], maximum[1]);
					maximum[0] = temp;
					maximum[1] = temp;
					temp = max(max(maximum[2], maximum[3]), max(maximum[4], maximum[13]));
					maximum[2] = temp;
					maximum[3] = temp;
					maximum[4] = temp; // thermal force
					maximum[13] = temp; // viscous
					temp = max(maximum[5], maximum[6]);
					maximum[5] = temp;
					maximum[6] = temp;
					temp = max(max(maximum[7], maximum[8]), max(maximum[9], maximum[10]));
					maximum[7] = temp;
					maximum[8] = temp;
					maximum[9] = temp;
					maximum[10] = temp;
					//	temp = max(maximum[11], maximum[12]); // difference, progress
					//	maximum[11] = temp;
					//	maximum[12] = temp;
					temp = max(maximum[14], maximum[17]);
					maximum[14] = temp;
					maximum[17] = temp;
					temp = max(maximum[15], maximum[16]);
					maximum[15] = temp;
					maximum[16] = temp;
				} else {
					// All same scale except for "difference" = element 10
					// ... and the arelz itself?
					int j;
					f64 temp = maximum[1];
					for (j = 2; j < 9; j++)
						temp = max(temp, maximum[j]);
					for (j = 1; j < 9; j++)
						maximum[j] = temp;
					temp = max(maximum[0], maximum[9]);
					maximum[0] = temp; 
					maximum[9] = temp; // actual vs sum

				}
			}
		}
	}
}


void RefreshGraphs(TriMesh & X, // only not const because of such as Reset_vertex_nvT
	const int iGraphsFlag)
{
	D3DXMATRIXA16 matWorld;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
	int offset_v, offset_T;
	char buff[256];
	sprintf(buff, "%5.2f ns", evaltime*1.0e9);
	f64 overc;
	char buffer[256];
	overc = 1.0 / c_;
	float x, y, z;
	float zeroplane = 0.0f;
	int i;
	int iGraph;
	

	float const MAXX = 11.0f;
	float const MAXY = 6.0f;
	long iMinor;

	switch (iGraphsFlag) {
		
	case ONE_D:

		// We are going to have to think about using LineTo the way it is done in RenderGraphs
		// let's start by rendering in the x-y plane and we can let the present camera look at it
		printf("\n\nGot to here: ONE_D\n\n");
		
		// Create data:
		Create1DGraphingData(&X);
		
		Draw1Dgraph(6, ONE_D);


		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost3[iVertex+BEGINNING_OF_CENTRAL];
			++pVertex;
			++pdata;
		}
		Graph[4].DrawSurface("Azdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_AZDOT, &X);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost4[iVertex + BEGINNING_OF_CENTRAL];
			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("Azdotdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_AZDOT, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost5[iVertex + BEGINNING_OF_CENTRAL];
			++pdata;
		}
		Graph[3].DrawSurface("Lap Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_LAPAZ, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = p_temphost6[iVertex + BEGINNING_OF_CENTRAL];
			}
			else {
				pdata->temp.x = 0.0;
			}
			++pdata;
		}
		Graph[5].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_JZ, &X);
		
		break;

	case AZSOLVERGRAPHS:

		pdata = X.pData;
		// Bear in mind: iMinor won't actually get displayed
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			pdata->temp.x = p_temphost1[iMinor]; // epsilon
			pdata->Azdot = p_temphost2[iMinor]; // Azdot0
			pdata->temp.y = p_temphost3[iMinor]; // gamma
			pdata->Az = p_temphost4[iMinor]; // Az			
			++pdata;
		}
		Graph[0].DrawSurface("epsilon",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_EPSILON, &X);

		Graph[2].DrawSurface("Azdot0",
			DATA_HEIGHT, (real *)(&(X.pData[0].Azdot)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Azdot)),
			true,
			GRAPH_AZDOT, &X);
		Graph[3].DrawSurface("regressorn",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.y)),
			true,
			GRAPH_OPTI, &X);

		Graph[4].DrawSurface("Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].Az)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Az)),
			true,
			GRAPH_AZ, &X);
		pdata = X.pData;
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			pdata->temp.x = p_temphost5[iMinor]; // epsilon
			pdata->temp.y = p_temphost6[iMinor]; // Azdot0
			++pdata;
		}
		Graph[1].DrawSurface("regressori",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_LAPAZ, &X);
		Graph[5].DrawSurface("Jacobi",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.y)),
			true,
			GRAPH_REGRESSOR, &X);

		break;
	case DTGRAPH:

		// We are going to have to think about using LineTo the way it is done in RenderGraphs
		// let's start by rendering in the x-y plane and we can let the present camera look at it
		printf("\n\nRefreshGraphs: DTGRAPHS\n\n");

		// Create data:
		Create1DGraphingData(&X, true);

		Draw1Dgraph(6, DTGRAPH);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_Tgraph_host[5][iVertex];
			++pVertex;
			++pdata;
		}
		Graph[4].DrawSurface("dTe/dt",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
			false,
			GRAPH_DTE, &X);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_Tgraph_host[7][iVertex];
			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("d/dt nTe",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_DNT, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_Tgraph_host[5][iVertex];
			++pdata;
		}
		Graph[3].DrawSurface("n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_ION_N, &X);

		Graph[5].DrawSurface("Te",
			DATA_HEIGHT, (real *)(&(X.pData[0].Te)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
			false,
			GRAPH_ELEC_T, &X);

		break;
	case OHMS2:

			// We are going to have to think about using LineTo the way it is done in RenderGraphs
			// let's start by rendering in the x-y plane and we can let the present camera look at it
			printf("\n\nRefreshGraphs: OHMS2\n\n");

			// Create data:
			Create1DGraphingData(&X, false, false, true);

			Draw1Dgraph(6, OHMS2);
			Draw1Dgraph(7, OHMSLAW);

			pVertex = X.X;
			pdata = X.pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_Ohmsgraph_host[14][iVertex];
				++pVertex;
				++pdata;
			}
			Graph[4].DrawSurface("Jz prediction",
				DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
				SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
				false,
				GRAPH_JZ, &X);

			pVertex = X.X;
			pdata = X.pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_Ohmsgraph_host[17][iVertex];
				++pVertex;
				++pdata;
			}
			Graph[5].DrawSurface("electromotive-only prediction",
				DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
				SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
				false,
				GRAPH_VE0Z, &X);

			break;

	case ACCELGRAPHS:

		// We are going to have to think about using LineTo the way it is done in RenderGraphs
		// let's start by rendering in the x-y plane and we can let the present camera look at it
		printf("\n\nRefreshGraphs: ACCELGRAPHS\n\n");

		// Create data:
		Create1DGraphingData(&X, false, true);
		
		Draw1Dgraph(6, ACCELGRAPHS);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_accelgraph_host[0][iVertex];
			pdata->temp.y = p_accelgraph_host[1][iVertex];
			++pVertex;
			++pdata;
		}
		Graph[4].DrawSurface("dvxy/dt",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_AXY, &X);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_accelgraph_host[2][iVertex];
			pdata->temp.y = p_accelgraph_host[3][iVertex];
			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("axy : v x B",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_AXY2, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_accelgraph_host[4][iVertex];
			pdata->temp.y = p_accelgraph_host[5][iVertex];
			++pdata;
		}
		Graph[3].DrawSurface("axy : pressure",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_AXY3, &X);

		Graph[5].DrawSurface("vxy",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].vxy)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false,
			GRAPH_ION_V, &X);
		
		break;

	case ARELZ:

		// We are going to have to think about using LineTo the way it is done in RenderGraphs
		// let's start by rendering in the x-y plane and we can let the present camera look at it
		printf("\n\nRefreshGraphs: ARELZ\n\n");

		// Create data:
		Create1DGraphingData(&X, false, false, false, true);

		Draw1Dgraph(6, ARELZ);

		// Graphs:
		// .. arelz
		// .. electromotive
		// .. v x B
		// .. error

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_arelz_graph_host[0][iVertex];
			pdata->temp.y = p_arelz_graph_host[0][iVertex];
			++pVertex;
			++pdata;
		}
		Graph[4].DrawSurface("arelz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_ARELZ, &X);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_arelz_graph_host[3][iVertex] +
				p_arelz_graph_host[4][iVertex];

			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("-e/m Ez_total",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_ELECTROMOTIVE, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_arelz_graph_host[5][iVertex];
			++pdata;
		}
		Graph[3].DrawSurface("arelz : v x B",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_VXBARELZ, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_arelz_graph_host[10][iVertex];
			if (pdata->temp.x > 1.0e13) {
				printf("%d %1.9E | ", iVertex, pdata->temp.x);
			}
			++pdata;
		}
		Graph[5].DrawSurface("error in sum",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_ERROR, &X);


	//	Cannot explain why maximum on graph is 1e13 not 1e5 as reported on 1D graph.


		break;
		/*
		case JXY_RHO_EXY_GRADPHI_AXYDOTOC_AXY:

		X.Setup_J(); // the others can already exist.

		Graph[4].bDisplayTimestamp = true;

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		pVertex->Adot /= c;
		++pVertex;
		}

		Graph[0].DrawSurface("Exy[statV/cm]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);

		Graph[1].DrawSurface("Adotxy/c[statV/cm]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Adot.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Adot.x)),
		true,
		GRAPH_ADOTXY, &X);

		Graph[2].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);

		Graph[3].DrawSurface("phidot[statV/s]",
		DATA_HEIGHT, (real *)(&(X.X[0].phidot)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phidot)),
		true,
		GRAPH_PHIDOT, &X);

		Graph[4].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);

		Graph[5].DrawSurface("phi[statV]",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		true,
		GRAPH_PHI, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot *= c;
		++pVertex;
		}
		break;
		case JZ_AZ_BXY_EZ_ADOTZOC_NVZ:

		X.Reset_vertex_nvT(SPECIES_ELEC);

		X.Setup_J(); // the others can already exist.

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot /= c;
		++pVertex;
		}
		Graph[4].bDisplayTimestamp = true;

		Graph[0].DrawSurface("Ez[statV/cm]",
		DATA_HEIGHT, (real *)(&(X.X[0].E.z)),
		FLAG_SEGUE_COLOUR, (real *)(&(X.X[0].E.z)),
		false, // ??
		GRAPH_EZ, &X);

		Graph[1].DrawSurface("Az",
		DATA_HEIGHT, (real *)(&(X.X[0].A.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].A.z)),
		true,
		GRAPH_AZ, &X);
		Graph[2].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.X[0].Temp.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);
		Graph[3].DrawSurface("Bxy[Gs]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].B)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].B)),
		true, // no inner mesh display: ??
		GRAPH_BXY, &X);

		Graph[4].DrawSurface("Adotz/c [statV/cm]",
		DATA_HEIGHT, (real *)(&(X.X[0].Adot.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Adot.z)),
		true,
		GRAPH_AZ, &X);

		Graph[5].colourmax = Graph[2].colourmax;
		Graph[5].DrawSurface("Elec n",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display
		GRAPH_ELEC_N, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot *= c;
		++pVertex;
		}

		break;
		case SPECIES_ELECTRON2:

		X.Reset_vertex_nvT(SPECIES_ELEC);

		Graph[0].DrawSurface("Elec n [/cc]",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false, // no inner mesh display
		GRAPH_ELEC_N, &X);

		Graph[1].DrawSurface("v_e_xy[cm/s]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].v)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false, // no inner mesh display
		GRAPH_ELEC_V, &X);

		Graph[3].DrawSurface("v_e_z[cm/s]",
		DATA_HEIGHT, (real *)(&(X.X[0].v.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].v.z)),
		false, // no inner mesh display.
		GRAPH_VEZ, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}
		Graph[2].bDisplayTimestamp = false;
		Graph[2].DrawSurface("n_i-n_e",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_NINE, &X);

		Graph[5].TickRescaling = 1.0 / kB;
		Graph[5].DrawSurface("Elec T [eV]",
		DATA_HEIGHT, (real *)(&(X.X[0].T)),
		SEGUE_COLOUR, (real *)(&(X.X[0].T)),
		false, // no inner mesh display
		GRAPH_ELEC_T, &X);
		Graph[5].TickRescaling = 1.0;

		offset_v = (real *)(&(X.X[0].v)) - (real *)(&(X.X[0]));
		offset_T = (real *)(&(X.X[0].T)) - (real *)(&(X.X[0]));

		Graph[4].SetEyePlan(GlobalPlanEye);
		Graph[4].boolDisplayMeshWireframe = true;
		Graph[4].boolClearZBufferBeforeWireframe = true;
		Graph[4].boolDisplayMainMesh = true;
		Graph[4].boolDisplayInnerMesh = false;
		Graph[4].boolDisplayScales = false;
		Graph[4].boolDisplayShadow = false;
		Graph[4].mhTech = Graph[4].mFX->GetTechniqueByName("VelociTech");
		Graph[4].colourmax = Graph[0].colourmax; // match colours
		Graph[4].SetDataWithColour(X, FLAG_VELOCITY_COLOUR, FLAG_FLAT_MESH, offset_v, offset_v,
		GRAPH_FLAT_WIRE_MESH);
		Graph[4].Render(buff, GlobalRenderLabels, &X);

		break;
		*/

case IONIZEGRAPH:
	printf("\n\nRefreshGraphs: IONIZEGRAPHS\n\n");

	// When we come to speed up graphs, make it so we can
	// just pass an array of f64. !!!!
	// Investigate graphs half an hour: what's up with the rest?

	// Move table, start running.
	// Can we bring back cutaway any how? 
	// Wanted acceleration graphs. 
	// Want to do a big run. 

	pVertex = X.X;
	pdata = X.pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_graphdata1_host[iVertex];
		pdata->temp.y = p_graphdata2_host[iVertex]; // dn/dt /n
		++pVertex;
		++pdata;
	}
	Graph[0].DrawSurface("dn/dt",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
		false,
		GRAPH_DNDT, &X);

	Graph[1].DrawSurface("dn/dt / n",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
		SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
		false,
		GRAPH_DNDT_OVER_n, &X);
	

	pVertex = X.X;
	pdata = X.pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_graphdata3_host[iVertex]; // log10 n
		++pVertex;
		++pdata;
	}
	Graph[3].DrawSurface("log10(n)",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false,
		GRAPH_LOG10N, &X);

	pVertex = X.X;
	pdata = X.pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_graphdata4_host[iVertex]; // dTe/dt
		pdata->temp.y = p_graphdata6_host[iVertex]; // n/nn
		++pVertex;
		++pdata;
	} 
	Graph[2].DrawSurface("dTe/dt[ionization]",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
		false,
		GRAPH_DTEDT, &X);

	Graph[4].DrawSurface("n_e / n_total",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
		IONISE_COLOUR, (real *)(&(X.pData[0].temp.y)),
		false,
		GRAPH_FRACTION, &X);

	pVertex = X.X;
	pdata = X.pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_graphdata5_host[iVertex]; // dvez/dt
		//if (pdata->vez != 0.0f) {
		//	pdata->temp.y = pdata->temp.x / (pdata->vez);
		//} else {
		//	pdata->temp.y = 0.0;
		//}
		++pVertex;
		++pdata;
	}
	printf("got to here 1");
	Graph[5].DrawSurface("accel ez[ionization]",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].vez)),
		false,
		GRAPH_AEZ1, &X);
	// Do we need another shader? Or can we reset limits?
	// see what scale is like.
	printf("got to here 2");

	break;




case OVERALL:
	printf("\n\nRefreshGraphs: OVERALL\n\n");

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = pdata->n + pdata->n_n;
				pdata->temp.y = pdata->n / (1.0 + pdata->temp.x);
			} else {
				pdata->temp.x = 0.0;
				pdata->temp.y = 0.0;
			}
			++pVertex;
			++pdata;
		}
		
		Graph[0].DrawSurface("n_n + n_ion",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			IONISE_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false,
			GRAPH_TOTAL_N, &X);
		
		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = (m_neutral_*pdata->n_n*pdata->v_n.x
					+ (m_ion_ + m_e_) * pdata->n*pdata->vxy.x) /
					(m_neutral_*pdata->n_n + (m_ion_ + m_e_)*pdata->n);
				pdata->temp.y = (m_neutral_*pdata->n_n*pdata->v_n.y
					+ (m_ion_ + m_e_) * pdata->n*pdata->vxy.y) /
					(m_neutral_*pdata->n_n + (m_ion_ + m_e_)*pdata->n);
			} else {
				pdata->temp.x = 0.0; pdata->temp.y = 0.0;
			}
			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("sum[n_s v_s m_s]/sum[n_s m_s]",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, // no inner mesh display
			GRAPH_TOTAL_V, &X);

		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = (pdata->n_n*pdata->Tn
					+ pdata->n*(pdata->Ti + pdata->Te)) /
					(pdata->n_n + pdata->n + pdata->n);
			} else {
				pdata->temp.x = 0.0; pdata->temp.y = 0.0;
			}
			++pVertex;
			++pdata;
		}
		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("sum[n_s T_s]/sum[n_s]",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false,
			GRAPH_TOTAL_T, &X);
		Graph[3].TickRescaling = 1.0;


		Graph[2].DrawSurface("Neutral n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n_n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].v_n)),
			false, // no inner mesh display
			GRAPH_NEUT_N, &X);
		Graph[4].DrawSurface("Neutral v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].v_n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].v_n)),
			false, // no inner mesh display
			GRAPH_NEUT_V, &X);


		Graph[5].TickRescaling = 1.0 / kB_;
		Graph[5].DrawSurface("Neutral T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Tn)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Tn)),
			false, // no inner mesh display
			GRAPH_NEUT_T, &X);
		Graph[5].TickRescaling = 1.0;
		
		break;
	case SPECIES_ION:
		printf("\n\nRefreshGraphs: SPECIES_ION\n\n");

		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("Ion T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Ti)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Ti)),
			false, // no inner mesh display
			GRAPH_ION_T, &X);
		Graph[3].TickRescaling = 1.0;

		// labels only appear on first 1 called.

		Graph[0].DrawSurface("Ion n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ION_N, &X);
		Graph[1].DrawSurface("Ion v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].vxy)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ION_V, &X);

		// These are same so double up with elec.

		Graph[5].TickRescaling = 1.0 / kB_;
		Graph[5].DrawSurface("Elec T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Te)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
			false, // no inner mesh display
			GRAPH_ELEC_T, &X);
		Graph[5].TickRescaling = 1.0;

		break;

/*	case SPECIES_ELEC:

		Graph[0].DrawSurface("Elec n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ELEC_T, &X);
		// colours == 0.0 ... because v = 0
		// First........... let's understand why surface normals come out unpredictable.
		// Then............ let's go and see what it does with y values (in Render and .fx)

		Graph[1].DrawSurface("Elec v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].vxy)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ELEC_V, &X);
		break;

		// In other cases, (and even for the above),
		// here is a good place to call the 
		// setup routines for temp variables.
		*/

case OHMSLAW:
	printf("\n\nRefreshGraphs: OHMSLAW\n\n");

		// 0. q/ m_e nu_sum 
		// 1. qn / m_e nu_sum
		// 2. nu_sum
		// 3. prediction of Jz from uniform Ez
		// 4. prediction of Jz from actual Ez
		// 5. Actual Jz
		
		// Let temphost1 = nu_en + nu_ei_effective
		// Let temphost2 = nu_en/temphost1


	// Cannot explain why, that comes out black and this doesn't.
	// Oh because colourmax has been set to 1 or not?

	// Yet the following crashes it. Bizarre? Maybe dividing by 0?
	
	overc = 1.0 / c_;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = q_ / (m_e_ * (1.0 + p_temphost1[iVertex + BEGINNING_OF_CENTRAL]));
				pdata->temp.y = p_temphost2[iVertex + BEGINNING_OF_CENTRAL]; // colour
			} else {
				pdata->temp.x = 0.0;
				pdata->temp.y = 0.0;
			}
			++pdata;
			++pVertex;
		};
		Graph[0].DrawSurface("q over m nu_effective",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			PPN_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false, // no inner mesh display.
			GRAPH_VRESPONSEOHMS, &X);

		
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n /
					(m_e_ * (1.0 + p_temphost1[iVertex + BEGINNING_OF_CENTRAL]));
				pdata->temp.y = p_temphost2[iVertex + BEGINNING_OF_CENTRAL]; // colour
			} else {
				pdata->temp.x = 0.0;
				pdata->temp.y = 0.0;
			};
			++pdata;
			++pVertex;
		};

		Graph[1].DrawSurface("qn / m nu_effective",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			PPN_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false, // no inner mesh display.
			GRAPH_CONDUCTIVITYOHMS, &X);
			

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = p_temphost1[iVertex + BEGINNING_OF_CENTRAL];
				pdata->temp.y = p_temphost2[iVertex + BEGINNING_OF_CENTRAL]; // colour
			};
			++pVertex;
			++pdata;
		};
		Graph[2].DrawSurface("nu_effective (blue=neut dominates)",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			PPN_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false, // no inner mesh display.
			GRAPH_NU_EFFECTIVE, &X);
			
		
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = EzStrength_*q_*q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n /
					(m_e_ * (1.0 + p_temphost1[iVertex + BEGINNING_OF_CENTRAL]));
			};
			++pdata;
		};
		Graph[3].DrawSurface("predict Jz (uniform Ez)",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),			
			false, // no inner mesh display.
			GRAPH_JZ, &X);
					
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = (EzStrength_
					- X.pData[iVertex + BEGINNING_OF_CENTRAL].Azdot*overc
					)*q_*q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n /
					(m_e_ * (1.0 + p_temphost1[iVertex + BEGINNING_OF_CENTRAL]));
			};
			++pdata;
		};
		Graph[4].DrawSurface("predict Jz (Ez)",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, // no inner mesh display.
			GRAPH_JZ, &X);
			
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		pVertex = X.X;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if (pVertex->flags == DOMAIN_VERTEX) {
				pdata->temp.x = q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n*
					(X.pData[iVertex + BEGINNING_OF_CENTRAL].viz - X.pData[iVertex + BEGINNING_OF_CENTRAL].vez);
			};
			++pdata;
		};
		Graph[5].DrawSurface("actual Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, // no inner mesh display.
			GRAPH_JZ, &X);

		break;

	case JZAZBXYEZ:
		printf("\n\nRefreshGraphs: JZAZBXYEZ\n\n");

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = q_ * pdata->n*(pdata->viz - pdata->vez);
			++pdata;
		};
		Graph[3].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, // no inner mesh display.
			GRAPH_JZ, &X);

		// create graph data for Ez : add Ez_strength*Ezshape to -Azdot/c
		overc = 1.0 / c_;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			X.pData[iVertex + BEGINNING_OF_CENTRAL].temp.y =
				-X.pData[iVertex + BEGINNING_OF_CENTRAL].Azdot*overc
				+ GetEzShape__(X.pData[iVertex + BEGINNING_OF_CENTRAL].pos.modulus())*EzStrength_;
		} 
		Graph[2].DrawSurface("Ez",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)), // use Jz's colour
			false, 
			GRAPH_EZ, &X);

		Graph[0].DrawSurface("Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].Az)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Az)),
			true, GRAPH_AZ, &X);

//		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//		{
//			X.pData[iVertex + BEGINNING_OF_CENTRAL].temp.x = p_B_host[iVertex + BEGINNING_OF_CENTRAL].x;
//			X.pData[iVertex + BEGINNING_OF_CENTRAL].temp.y = p_B_host[iVertex + BEGINNING_OF_CENTRAL].y;
//		}
		Graph[1].DrawSurface("Bxy",
		VELOCITY_HEIGHT, (real *)(&(X.pData[0].B.x)),
		VELOCITY_COLOUR, (real *)(&(X.pData[0].B.x)),
		false,
		GRAPH_BXY, &X);

		Graph[5].DrawSurface("vez",
			DATA_HEIGHT, (real *)(&(X.pData[0].vez)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)), // colour is for Jz?
			false, GRAPH_VEZ, &X);


		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = -p_temphost3[iVertex + BEGINNING_OF_CENTRAL]/c_;
			++pVertex;
			++pdata;
		}
		Graph[4].DrawSurface("-Azdot/c",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true,
			GRAPH_AZDOT, &X);
	//	pdata = X.pData + BEGINNING_OF_CENTRAL;
	//	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	//	{
	//		pdata->temp.x = temp_array_host[iVertex + BEGINNING_OF_CENTRAL];
	//		++pdata;
	//	};
	//	Graph[4].DrawSurface("Lap Az",
	//		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
	//		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
	//		true, GRAPH_LAPAZ, &X);
		break;

	case VIZVEZJZAZDOT:
		printf("\n\nRefreshGraphs: VIZVEZJZAZDOT\n\n");

		// Set Jz:
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = q_ * pdata->n*(pdata->viz - pdata->vez);
			++pdata;
		};

		Graph[0].DrawSurface("viz",
			DATA_HEIGHT, (real *)(&(X.pData[0].viz)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_VIZ, &X);

		Graph[1].DrawSurface("vez",
			DATA_HEIGHT, (real *)(&(X.pData[0].vez)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_VEZ, &X);

		Graph[2].DrawSurface("Azdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].Azdot)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Azdot)),
			true, GRAPH_AZDOT, &X);

		Graph[3].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_JZ, &X);

		break;
		/*
	case NEWSTUFF:

		// Too bad substep is not stated. We should divide by substep to give anything meaningful
		// in these graphs.

		// Let temphost3 = vez0
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		pdata->temp.x = p_temphost3[iVertex + BEGINNING_OF_CENTRAL];
		++pdata;
		};
		Graph[0].DrawSurface("vez0 : vez = vez0 + sigma Ez",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display.
		GRAPH_VEZ0, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		pdata->temp.x = p_OhmsCoeffs_host[iVertex + BEGINNING_OF_CENTRAL].sigma_e_zz;
		++pdata;
		};
		Graph[1].DrawSurface("sigma : vez = vez0 + sigma Ez",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display.
		GRAPH_RESPONSE, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		pdata->temp.x = q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n*
		(p_OhmsCoeffs_host[iVertex + BEGINNING_OF_CENTRAL].sigma_i_zz
		- p_OhmsCoeffs_host[iVertex + BEGINNING_OF_CENTRAL].sigma_e_zz);

		// Will show something not very useful ---- in a brief instant there
		// isn't much time for second-order (frictional) effects.
		++pdata;
		};
		Graph[2].DrawSurface("Ez=0 v addition: vez0-vez",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display.
		GRAPH_DECEL, &X);
		// Too bad substep is not stated. We should divide by substep to give anything meaningful
		// in these graphs.

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		pdata->temp.x = q_*X.pData[iVertex + BEGINNING_OF_CENTRAL].n*
		(p_OhmsCoeffs_host[iVertex + BEGINNING_OF_CENTRAL].sigma_i_zz
		- p_OhmsCoeffs_host[iVertex + BEGINNING_OF_CENTRAL].sigma_e_zz);

		// Will show something not very useful ---- in a brief instant there
		// isn't much time for second-order (frictional) effects.
		++pdata;
		};
		Graph[3].DrawSurface("dynamic conductivity q n sigma : vez = vez0 + sigma Ez",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display.
		GRAPH_DYNCONDUCTIVITY, &X);

		// create graph data for Ez : add Ez_strength*Ezshape to -Azdot/c
		overc = 1.0 / c_;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		X.pData[iVertex + BEGINNING_OF_CENTRAL].temp.y =
		-X.pData[iVertex + BEGINNING_OF_CENTRAL].Azdot*overc
		+ GetEzShape__(X.pData[iVertex + BEGINNING_OF_CENTRAL].pos.modulus())*EzStrength_;
		}
		Graph[4].DrawSurface("Ez",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)), // use Jz's colour
		false,
		GRAPH_EZ, &X);

		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
		pdata->temp.x = q_ * pdata->n*(pdata->viz - pdata->vez);
		++pdata;
		};
		Graph[5].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);

		break;*/
	case LAPAZ_AZ:
		
		printf("\n\nRefreshGraphs: LAPAZ_AZ\n\n");
		// Assume temp.x contains Lap Az
		Graph[0].DrawSurface("Lap Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true, GRAPH_LAPAZ, &X);
		Graph[1].DrawSurface("Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].Az)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Az)),
			true, GRAPH_AZ, &X);
		Graph[2].DrawSurface("Azdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].Azdot)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Azdot)),
			true, GRAPH_AZDOT, &X);
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.y = q_ * pdata->n*(pdata->viz - pdata->vez);
			++pdata;
		};
		Graph[3].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false, GRAPH_JZ, &X);

		break;
	case EXYCOMPONENTS:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("Adotxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Adot.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Adot.x)),
		true,
		GRAPH_ADOTXY, &X);
		Graph[1].DrawSurface("Grad phi",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].GradTe)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].GradTe)),
		true, // no inner mesh display: ??
		GRAPH_GRADPHI, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);

		*/
		// Set GradTe to grad phi
		break;
	case JXYAXYBZEXY:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("Axy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].A.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].A.x)),
		true,
		GRAPH_AXY, &X);
		Graph[1].DrawSurface("Bz",
		DATA_HEIGHT, (real *)(&(X.X[0].B.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].B.z)),
		true, // no inner mesh display: ??
		GRAPH_BZ, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);
		*/
		break;
	case EXY_RHO_PHI_PHIDOT:
		/*
		// For this one do n_i-n_e
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		Graph[0].DrawSurface("phi",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		true,
		GRAPH_PHI, &X);
		Graph[1].DrawSurface("phidot",
		DATA_HEIGHT, (real *)(&(X.X[0].phidot)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phidot)),
		true,
		GRAPH_PHIDOT, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("n_i-n_e",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_NINE, &X);
		*/
		break;
	case EXY_RHO_PHI_JXY:
		// create rho on pVertex->temp2.x ... 
		/*
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		X.Setup_J();

		Graph[0].DrawSurface("phi",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		false,
		GRAPH_PHI, &X);
		Graph[1].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp)),
		false, // no inner mesh display: ??
		GRAPH_JXY, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		false,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);
		*/
		break;

	case EXY_RHO_BZ_JXY:
		/*
		// create rho on pVertex->temp2.x ...
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		X.Setup_J();

		Graph[0].DrawSurface("Bz",
		DATA_HEIGHT, (real *)(&(X.X[0].B.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].B.z)),
		true, // no inner mesh display: ??
		GRAPH_BZ, &X);
		Graph[1].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp)),
		false, // no inner mesh display: ??
		GRAPH_JXY, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		false,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);
		*/
		break;

	//case SIGMA_E_J:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("sigma_e_zz",
		DATA_HEIGHT, (real *)(&(X.X[0].sigma_e.zz)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].sigma_e.zz)),
		true,
		GRAPH_SIGMA_E, &X);
		//Graph[1].DrawSurface("v_e_0.z",
		//	DATA_HEIGHT,(real *)(&(X.X[0].v_e_0.z)),
		//	AZSEGUE_COLOUR,(real *)(&(X.X[0].v_e_0.z)),
		//false, // no inner mesh display: ??
		// GRAPH_VE0Z, &X);
		Graph[1].DrawSurface("nsigma",
		DATA_HEIGHT, (real *)(&(X.X[0].xdotdot.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].xdotdot.x)),
		true, GRAPH_SIGMATEMP, &X);
		Graph[2].DrawSurface("Ez",
		DATA_HEIGHT, (real *)(&(X.X[0].E.z)),
		FLAG_AZSEGUE_COLOUR, (real *)(&(X.X[0].E.z)), // how to make SEGUE_COLOUR work?
		false, // ??
		GRAPH_EZ, &X);
		Graph[3].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.X[0].Temp.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);
		*/
	//	break;


	case TOTAL:
		
		// In this case we have to create data,
		// as we go.
		
		// Best put it here so we can see where
		// data is being populated.

		/*long iVertex;
		Vertex * pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
		if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
		{
		pVertex->n = (pVertex->Neut.mass + pVertex->Ion.mass) / pVertex->AreaCell;
		pVertex->v = (m_n*pVertex->Neut.mom + m_ion * pVertex->Ion.mom + m_e * pVertex->Elec.mom) /
		(m_n*pVertex->Neut.mass + m_ion * pVertex->Ion.mass + m_e * pVertex->Elec.mass);
		pVertex->T = (pVertex->Neut.heat + pVertex->Ion.heat + pVertex->Elec.heat) /
		(pVertex->Neut.mass + pVertex->Ion.mass + pVertex->Elec.mass);
		pVertex->Temp.x = pVertex->Ion.mass / (pVertex->Neut.mass + pVertex->Ion.mass);
		};
		++pVertex;
		}*/
		//X.CalculateTotalGraphingData();

		printf("\n\nRefreshGraphs: TOTAL\n\n");
		// ought to change this to use variables n,v,T !
		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = pdata->n + pdata->n_n;
				pdata->temp.y = pdata->n / pdata->temp.x;
			}
			++pVertex;
			++pdata;
		}
		Graph[0].DrawSurface("n_n + n_ion",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		IONISE_COLOUR, (real *)(&(X.pData[0].temp.y)),
		false,
		GRAPH_TOTAL_N, &X);


		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = (m_neutral_*pdata->n_n*pdata->v_n.x
					+ (m_ion_ + m_e_) * pdata->n*pdata->vxy.x) /
					(m_neutral_*pdata->n_n + (m_ion_ + m_e_)*pdata->n);
				pdata->temp.y = (m_neutral_*pdata->n_n*pdata->v_n.y
					+ (m_ion_ + m_e_) * pdata->n*pdata->vxy.y) /
					(m_neutral_*pdata->n_n + (m_ion_ + m_e_)*pdata->n);
			}
			++pVertex;
			++pdata;
		}
		Graph[1].DrawSurface("sum[n_s v_s m_s]/sum[n_s m_s]",
		VELOCITY_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false, // no inner mesh display
		GRAPH_TOTAL_V, &X);
		
		
		//Graph[2].DrawSurface("n_n+n_ion",
		//DATA_HEIGHT, (real *)(&(X.X[0].n)),
		//VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		//false,
		//GRAPH_TOTAL_N_II, &X);   // ok what we did here? we thought we'd colour with velocity .. but we haven't given ourselves room for 3 temp vars so drop this for now.
		
		
		pVertex = X.X;
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
			{
				pdata->temp.x = (pdata->n_n*pdata->Tn
							+  pdata->n*(pdata->Ti + pdata->Te)) /
								(pdata->n_n + pdata->n + pdata->n);
			}
			++pVertex;
			++pdata;
		}
		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("sum[n_s T_s]/sum[n_s]",
		DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
		SEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
		false,
		GRAPH_TOTAL_T, &X);
		Graph[3].TickRescaling = 1.0;
		break;
		
	};

	// Graph 2 and 4, in case of species graphs:

	switch (iGraphsFlag) {
	//case SPECIES_NEUTRAL:
	case SPECIES_ION:
	//case SPECIES_ELEC:
	//case TOTAL:

		int offset_v, offset_T;
		offset_v = (real *)(&(X.pData[0].vxy)) - (real *)(&(X.pData[0]));
		offset_T = (real *)(&(X.pData[0].Te)) - (real *)(&(X.pData[0]));

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
			Graph[2].SetDataWithColour(X, FLAG_COLOUR_MESH, FLAG_FLAT_MESH, 0, 0,
				GRAPH_FLAT_WIRE_MESH);
			Graph[2].Render(buff, GlobalRenderLabels, &X);

		} else {
			// Tell SDWC not to mess with colourmax if it's a flat mesh.

			if (GlobalColoursPlanView == 1)
			{
				// velocity
				Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("VelociTech");
				Graph[2].colourmax = Graph[0].colourmax; // match colours

				Graph[2].SetDataWithColour(X, FLAG_VELOCITY_COLOUR, FLAG_FLAT_MESH, offset_v, offset_v,
					GRAPH_FLAT_WIRE_MESH);
				Graph[2].Render(buff, GlobalRenderLabels, &X);
			};
			////else {
			////	// temperature
			////	Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("SegueTech");
			////	// SegueVS should take maximum as a parameter;
			////	// at least for colours we should prefer an absolute scale for T
			////	// Is it ever used for anything else? Not so far? eps?

			////	Graph[2].SetDataWithColour(X, FLAG_SEGUE_COLOUR, FLAG_FLAT_MESH, offset_T, offset_T,
			////		GRAPH_FLAT_WIRE_MESH);
			////	Graph[2].Render(buff, GlobalRenderLabels, &X);
			////};
		};

		// =================================================================================

		printf("\ngot to here; graph [4]:\n\n");

		Graph[4].boolDisplayKeyButton = false; // it's temperature
		Graph[4].SetEyePlan(GlobalPlanEye);
		Graph[4].boolDisplayMeshWireframe = true;
		Graph[4].boolClearZBufferBeforeWireframe = true;
		Graph[4].boolDisplayMainMesh = true;
		Graph[4].boolDisplayInnerMesh = false;
		Graph[4].boolDisplayScales = false;

		Graph[4].mhTech = Graph[4].mFX->GetTechniqueByName("SegueTech");
		
		Graph[4].SetDataWithColour(X, FLAG_SEGUE_COLOUR, FLAG_FLAT_MESH, offset_T, offset_T,
					GRAPH_FLAT_WIRE_MESH);
		Graph[4].Render(buff, GlobalRenderLabels, &X);
				
		break;
	}
	printf("End of Refreshgraphs\n");

}


IMFSinkWriter *pSinkWriter[NUMAVI];
DWORD izStream[NUMAVI];

HRESULT hresult;

LONGLONG rtStart = 0;

int main()
{
   
	printf("hello\n");


	HINSTANCE hInstance = GetModuleHandle(NULL);
	HWND hwndConsole = GetConsoleWindow();
	WCHAR szInitialFilenameAvi[512];
	MSG msg;
	HDC hdc;
	//	HACCEL hAccelTable;
	real x, y, temp;
	int i, j;
	float a1, a2, a3, a4; 
	//HWND hwndConsole;
	FILE * fp;
	extern char Functionalfilename[1024];
	
	int nDevices, iWhich;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);

		if (prop.memoryBusWidth == 384) iWhich = i;
	}
	printf("Picked %d \n", iWhich);
	getch(); 

	cudaSetDevice(iWhich); // K40?
	cudaDeviceReset();

	size_t uFree, uTotal;
	cudaMemGetInfo(&uFree, &uTotal);
	printf("Memory on device: uFree %zd uTotal %zd\n", uFree, uTotal);

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if (!SUCCEEDED(hr)) {
		printf("CoInitializeEx failed. press p\n");
		while (getch() != 'p');
		exit(23233);
	}
	hr = MFStartup(MF_VERSION);
	if (!SUCCEEDED(hr)) {
		printf("MFStartup failed. press p\n");
		while (getch() != 'p');
		exit(23234);
	}


	h = TIMESTEP;
	evaltime = 0.0; // gets updated before advance

	memset(Historic_powermax, 0, 200 * sizeof(int));
	memset(Historic_powermin, 0, 200 * sizeof(int));

	ZeroMemory(Historic_max, 512 * HISTORY * sizeof(float));
	ZeroMemory(Historic_min, 512 * HISTORY * sizeof(float));
	GlobalStepsCounter = 0; steps_remaining = 0; steps_remaining_CPU = 0;

	SetConsoleTitle("2D 1/16 annulus DPF simulation");
	Sleep(40);
	//hwndConsole = FindWindow(NULL, "2D 1/16 annulus DPF simulation");
	MoveWindow(hwndConsole, 0, 0, SCREEN_WIDTH - VIDEO_WIDTH - 10, SCREEN_HEIGHT - 30, TRUE);

	report_time(0);

	int filetag = 0;
	do { 
		filetag++;
		sprintf(Functionalfilename, FUNCTIONALFILE_START "%03d.txt", filetag);
	} while ((_access(Functionalfilename, 0)) != -1);

	printf("\n\nopening %s \n", Functionalfilename);
	fp = fopen(Functionalfilename, "w");
	if (fp == 0) {
		printf("error with %s \n", Functionalfilename);
		getch();
	}
	else {
		printf("opened %s \n", Functionalfilename);
	}; 
	fprintf(fp, "GSC evaltime Area neut.N ion.N elec.N neut.r ion.r elec.r SDneut.r SDion.r SDelec.r "
		" neut.vr neut.vth neut.vz  ion.vr ion.vth ion.vz elec.vr elec.vth elec.vz neut.heat ion.heat elec.heat neut.T ion.T elec.T "
		" neut.mnvv/3 ion.mnvv/3 elec.mnvv/3 elec.force(vxB)r within3.6 elec.Bth EE BB Heatings and dT changes - see code \n");
	fclose(fp);
	 
	X1.Initialise(1); // Set evaltime first
	X2.Initialise(2);
	X3.Initialise(3);
	printf("Got to here 1\n");	    
	{
		X4.Initialise(4);
		printf("Got to here 2\n");
		X4.CreateTilingAndResequence2(&X1);
		X4.CreateTilingAndResequence2(&X2);
		X4.CreateTilingAndResequence2(&X3);
		printf("Got to here 3\n");
		// 
		// Dropping it for now so we can pursue solving equations first.
		//  
	}
	X1.Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	X1.EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	X1.SetupMajorPBCTriArrays();
	X2.Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	X2.EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	X2.SetupMajorPBCTriArrays();
	X3.Recalculate_TriCentroids_VertexCellAreas_And_Centroids();
	X3.EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	X3.SetupMajorPBCTriArrays();
	printf("Got to here 4\n");
	X1.InitialPopulate();
	X2.InitialPopulate();
	X3.InitialPopulate();

	X1.Create4Volleys();
	X2.Create4Volleys();
	X3.Create4Volleys();

	pTriMesh = &X1;

	pX = &X1;
	pXnew = &X2;
	
	GlobalBothSystemsInUse = 0;

	printf(report_time(1));
	printf("\n");
	report_time(0);
	
	// Window setup
	LoadString(hInstance, IDS_APP_TITLE, szTitle, 1024);
	LoadString(hInstance, IDC_F2DVALS, szWindowClass, 1024);
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_F2DVALS));
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCE(IDR_MENU1);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
	if (RegisterClassEx(&wcex) == 0) {
		char buff[128];
		MessageBox(NULL, "RegisterClassEx failed", itoa(GetLastError(), buff, 10), MB_OK);
	};

	printf("SCREEN_WIDTH %d VIDEO_WIDTH %d VIDEO_HEIGHT %d \n",
		SCREEN_WIDTH, VIDEO_WIDTH, VIDEO_HEIGHT);

	hWnd = CreateWindowEx(NULL, szWindowClass, szTitle, WS_BORDER | WS_POPUP,
		SCREEN_WIDTH - VIDEO_WIDTH - 5, 0, VIDEO_WIDTH + 5, VIDEO_HEIGHT + 20, NULL, NULL, hInstance, NULL);
	if (!hWnd) {
		DWORD dword = GetLastError();
		char buff[128];
		MessageBox(NULL, "CreateWindowEx failed", itoa(dword, buff, 10), MB_OK);
		return dword;
	}
	// This is sending a message to WndProc before any of the following happens.

	ShowWindow(hWnd, SW_SHOWNORMAL);
	UpdateWindow(hWnd);

	hwndGraphics = hWnd;

	xzscale = 2.0 / 0.1; // very zoomed in. Now what?

	DXChk(Direct3D.Initialise(hWnd, hInstance, VIDEO_WIDTH, VIDEO_HEIGHT));

	// With Field Of View = PI/4 used this:
	/*
	GlobalEye.x = 0.0f;
	GlobalEye.y = 12.4f;  //7.2f;
	GlobalEye.z = -18.0f + 2.5*xzscale;//DEVICE_RADIUS_INSULATOR_OUTER*xzscale;//-17.8f+

	GlobalLookat.x = 0.4f;
	GlobalLookat.y = 3.0f;
	GlobalLookat.z = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * xzscale;

	GlobalPlanEye.x = 0.0f;
	GlobalPlanEye.y = 35.0f;
	GlobalPlanEye.z = (3.44 + 4.1)*0.5*xzscale;

	GlobalPlanEye2.x = -0.1f;
	GlobalPlanEye2.y = 19.5f;
	GlobalPlanEye2.z = 2.8*xzscale;

	GlobalPlanLookat.x = GlobalPlanEye.x;
	GlobalPlanLookat.y = 0.0f;
	GlobalPlanLookat.z = GlobalPlanEye.z + 0.0001;

	GlobalPlanLookat2.x = GlobalPlanEye2.x;
	GlobalPlanLookat2.y = 0.0f;
	GlobalPlanLookat2.z = GlobalPlanEye2.z + 0.0001;*/

	GlobalEye.x = -10.4f;
	GlobalEye.y = 16.4f;  //7.2f;
	GlobalEye.z = 44.0f;

	GlobalLookat.x = 1.20f;
	GlobalLookat.y = 3.0f;
	GlobalLookat.z = 72.2f;

	GlobalPlanEye.x = 2.9f;
	GlobalPlanEye.y = 17.97f;
	GlobalPlanEye.z = 71.95f;

	GlobalPlanEye2.x = -0.1f;
	GlobalPlanEye2.y = 19.5f;
	GlobalPlanEye2.z = 2.8*xzscale;
	 
	GlobalPlanLookat.x = GlobalPlanEye.x;
	GlobalPlanLookat.y = 0.0f;
	GlobalPlanLookat.z = GlobalPlanEye.z + 0.0001;

	GlobalPlanLookat2.x = GlobalPlanEye2.x;
	GlobalPlanLookat2.y = 0.0f;
	GlobalPlanLookat2.z = GlobalPlanEye2.z + 0.0001;
	
	newEye.x = 0.0f;
	newEye.y = 0.1f;
	newEye.z = 40.0f;
	newLookat.x = 0.0f;
	newLookat.y = 0.0f;
	newLookat.z = 72.0f;

						 // Add vectors in parallel.
	cudaError_t cudaStatus;

	if (DXChk(Graph[0].InitialiseWithoutBuffers(0, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[0].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(200);
	};
	if (DXChk(Graph[1].InitialiseWithoutBuffers(0, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[1].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(201);
	};
	if (DXChk(Graph[2].InitialiseWithoutBuffers(GRAPH_WIDTH, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalPlanEye, GlobalPlanLookat)) +
		DXChk(Graph[2].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(202);
	};
	if (DXChk(Graph[3].InitialiseWithoutBuffers(GRAPH_WIDTH, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[3].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(203);
	};
	   
	if (NUMGRAPHS > 4) {

		if (DXChk(Graph[4].InitialiseWithoutBuffers(GRAPH_WIDTH * 2, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalPlanEye, GlobalPlanLookat)) +
			DXChk(Graph[4].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};

		if (DXChk(Graph[5].InitialiseWithoutBuffers(GRAPH_WIDTH * 2, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
			DXChk(Graph[5].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};
		if (DXChk(Graph[6].InitialiseWithoutBuffers(0, 0, GRAPH_WIDTH*2, GRAPH_HEIGHT, newEye, GlobalLookat, true)) +
			DXChk(Graph[6].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};
		if (DXChk(Graph[7].InitialiseWithoutBuffers(0, GRAPH_HEIGHT, GRAPH_WIDTH * 2, GRAPH_HEIGHT, newEye, GlobalLookat, true)) +
			DXChk(Graph[7].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};
	};

	Graph[0].bDisplayTimestamp = false;
	Graph[1].bDisplayTimestamp = false;
	Graph[2].bDisplayTimestamp = false;
	Graph[3].bDisplayTimestamp = false;
	Graph[4].bDisplayTimestamp = true;
	Graph[5].bDisplayTimestamp = false;
	Graph[6].bDisplayTimestamp = true;
	Graph[7].bDisplayTimestamp = false;

	Direct3D.pd3dDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &p_backbuffer_surface);

	if (DXChk(p_backbuffer_surface->GetDC(&surfdc), 1000))
		MessageBox(NULL, "GetDC failed", "oh dear", MB_OK);

	surfbit = CreateCompatibleBitmap(surfdc, VIDEO_WIDTH, VIDEO_HEIGHT); // EXTRAHEIGHT = 90
	SelectObject(surfdc, surfbit);
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

	BitBlt(dibdc, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, surfdc, 0, 0, SRCCOPY);
	
	rtStart = 0; // timeslice : where to place frames into mp4 files.

	for (i = 0; i < NUMAVI; i++)
	{
		swprintf(szInitialFilenameAvi, L"%s%s_%s", FOLDER, szmp4[i], INITIALMP4);
	
		pSinkWriter[i] = NULL;

		hr = InitializeSinkWriter(&(pSinkWriter[i]), &(izStream[i]), szInitialFilenameAvi);

		if (!SUCCEEDED(hr)) {
			printf("Failed to create mp4 file %d %ls \n", i, szmp4[i]);
		}

		// hAvi[i] = CreateAvi(szInitialFilenameAvi, AVIFRAMEPERIOD, NULL);
		//if (hAvi[i] == 0) {
		//	printf("Failed to create avi file %d", i);
		//	getch(); getch(); getch();
		//}
	};
	
	printf("got to here: Initialized SinkWriters \n");
	getch();

	// 1000/25 = 40
	//ZeroMemory(&opts, sizeof(opts));
	//opts.fccHandler = mmioFOURCC('D', 'I', 'B', ' ');//('d','i','v','x');
	//opts.dwFlags = 8;

	//for (i = 0; i < NUMAVI; i++)
	//{
	//	hresult = SetAviVideoCompression(hAvi[i], dib, &opts, false, hWnd); // always run this for every avi file but can
	//															  // call with false as long as we know opts contains valid information. 
	//	if (hresult != 0) {
	//		printf("error: i = %d, hresult = %d", i, (long)hresult);
	//		getch(); getch(); getch();
	//	};
	//};

	counter = 0;
	//ReleaseDC(hWnd,surfdc);
	p_backbuffer_surface->ReleaseDC(surfdc);
	GlobalCutaway = true; // dies if true
	
	RefreshGraphs(*pX, GlobalSpeciesToGraph);
	
	Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);


	// Main message loop:
	memset(&msg, 0, sizeof(MSG));
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		} else {
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
		};
	};

	UnregisterClass(szWindowClass, wcex.hInstance);
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/* Auxiliary routine: printing a matrix */
void print_matrix(char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda) {
	lapack_int i, j;
	printf("\n %s\n", desc);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) printf(" %2.5E", a[i*lda + j]);
		printf("\n");
	}
}
 
/* Auxiliary routine: printing a vector of integers */
void print_int_vector(char* desc, lapack_int n, lapack_int* a) {
	lapack_int j;
	printf("\n %s\n", desc);
	for (j = 0; j < n; j++) printf(" %6i", a[j]);
	printf("\n");
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{

	f64 lowest_vez;
	long iLow, iMinor;
	Triangle * pTri;
	Vertex * pVertex;
	
	long izTri[128];
	 
	static bool bInvoked_cuSyst = false;
	static long GSCCPU = 0;
	int iAntiskips;
	int wmId, wmEvent;
	int i, j, ctr;
	PAINTSTRUCT ps;
	HDC hdc;
	real time_back_for_Adot;
	FILE * file, *fp;
	int maxeerr, count, iMin;
	WCHAR buf1000[1024];
	char buf1001[1024];
	int attempts;
	real store_h;
	char ch, o;
	int failed;
	RECT rect;
	real TotalArea, TotalCharge;
	long iVertex;
	real mass_avg, mass_SD, mass_min, mass_max;
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name
	char szFilter[1000]; // buffer for file filter
	char szfilter[256];
	char buffer[256];

	TriMesh * temp;

	static const real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * sin(PI / 32.0);
	static const real XCENTRE1 = -XCENTRE2;
	static const real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * cos(PI / 32.0);
	 
	switch (message)
	{
	case WM_CREATE: 

		// Don't ever try doing initialisation here;
		// That should be done manually from the menus.		

		break;
		 
	case WM_COMMAND:
		wmId = LOWORD(wParam);
		wmEvent = HIWORD(wParam);

		printf("\nWM_COMMAND: wmId %d\n\n", wmId);
	
		// Ensure that display menu items are consecutive IDs.
		// Parse the menu selections:
		switch (wmId)
		{
			
		case ID_DISPLAY_ONE_D:
			// printf("\a\n");
			// Don't know why resource.h is not working;
			// Maybe some #define overwrites it with 40024.
			//wmId += 50007 - 40024;
			GlobalSpeciesToGraph = ONE_D;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;
		// int const GraphFlags[NUMAVI] = { SPECIES_ION, OVERALL, JZAZBXYEZ, OHMSLAW, ONE_D, IONIZEGRAPH };
		case ID_DISPLAY_ION:
			GlobalSpeciesToGraph = SPECIES_ION;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;

		case ID_DISPLAY_TOTAL:
			GlobalSpeciesToGraph = OVERALL;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;

		case ID_DISPLAY_JZAZBXYEZ:
			GlobalSpeciesToGraph = JZAZBXYEZ;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;
		case ID_DISPLAY_IONIZEGRAPHS:
			GlobalSpeciesToGraph = IONIZEGRAPH;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;
		case ID_DISPLAY_OHMS:
			GlobalSpeciesToGraph = OHMSLAW;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;
		case ID_DISPLAY_DTGRAPH:

			GlobalSpeciesToGraph = DTGRAPH;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			break;
		case ID_DISPLAY_ACCELGRAPH:

			GlobalSpeciesToGraph = ACCELGRAPHS;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			break;

		case ID_DISPLAY_TENSOROHMS:

			GlobalSpeciesToGraph = OHMS2;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			break;

		case ID_DISPLAY_ACCELRELZ:
			GlobalSpeciesToGraph = ARELZ;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			break;
		case ID_DISPLAY_SIGMAEJ:

			i = wmId - ID_DISPLAY_NEUT;
			GlobalSpeciesToGraph = i;
			printf("\nGlobalSpeciesToGraph = %d \n", GlobalSpeciesToGraph);
			RefreshGraphs(*pX, GlobalSpeciesToGraph);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;

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

			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving camera...");
				fp = fopen(ofn.lpstrFile, "wt");
				if (fp == 0) {
					printf("save failed.\n");
				}
				else {
					fprintf(fp, "%f %f %f ", GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
					fprintf(fp, "%f %f %f ", GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
					fprintf(fp, "%f %f %f ", GlobalEye.x, GlobalEye.y, GlobalEye.z);
					fprintf(fp, "%f %f %f ", GlobalPlanLookat.x, GlobalPlanLookat.y, GlobalPlanLookat.z);
					fclose(fp);
					printf("done\n");
				};
			}
			else {
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
			if (GetOpenFileName(&ofn) == TRUE)
			{
				printf("\nloading camera...");
				fp = fopen(ofn.lpstrFile, "rt");
				if (fp == 0) {
					printf("failed.\n");
				}
				else {
					rewind(fp);
					fscanf(fp, "%f %f %f ", &(GlobalPlanEye.x), &(GlobalPlanEye.y), &(GlobalPlanEye.z));
					fscanf(fp, "%f %f %f ", &(GlobalLookat.x), &(GlobalLookat.y), &(GlobalLookat.z));
					fscanf(fp, "%f %f %f ", &(GlobalEye.x), &(GlobalEye.y), &(GlobalEye.z));
					fscanf(fp, "%f %f %f ", &(GlobalPlanLookat.x), &(GlobalPlanLookat.y), &(GlobalPlanLookat.z));
					fclose(fp);
				};
				RefreshGraphs(*pX, GlobalSpeciesToGraph); // sends data to graphs AND renders them
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			}
			else {
				printf("file error camera\n");
			};
			break; 
		case ID_FILE_LOADGPU:

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
			if (GetOpenFileName(&ofn) == TRUE)
			{
				if (bInvoked_cuSyst == false) {
					bInvoked_cuSyst = true;

					pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
					pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos();

					pX->Create4Volleys(); // THIS SHOULD NOT ALWAYS BE HERE !!
					printf("Called Create4Volleys! This should be removed in favour of loaded iVolley.\n");

					cuSyst_host.InvokeHost();
					cuSyst_host.PopulateFromTriMesh(pX);
					cuSyst_host2.InvokeHost();
					cuSyst_host2.PopulateFromTriMesh(pX);
					// transfer information.

					PerformCUDA_Invoke_Populate(
						&cuSyst_host,
						NUMVERTICES,
						pX->InnermostFrillCentroidRadius,
						pX->OutermostFrillCentroidRadius,
						pX->numStartZCurrentTriangles,
						pX->numEndZCurrentTriangles);
				};

				cuSyst_host.Load(ofn.lpstrFile);
			};
			printf("Populate *pX\n");
			cuSyst_host.PopulateTriMesh(pX);
			printf("send to device\n");
			cuSyst_host.SendToDevice(cuSyst1);
			printf("done\n");

			// Debug: redelaun on load:
			pX->RefreshVertexNeighboursOfVerticesOrdered();
	//		pX->Redelaunerize(true, true);

			// This isn't actually helpful?

			// pX->RefreshVertexNeighboursOfVerticesOrdered();
			// pX->X[89450-BEGINNING_OF_CENTRAL].GetTriIndexArray(izTri);
//			printf("89450 : %d %d %d %d %d %d \n",
//				izTri[0], izTri[1], izTri[2], izTri[3], izTri[4], izTri[5]);
//
			pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();			 
			//	pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // Obviates some of our flip calcs to replace tri n,T 
			// not sure if needed .. just for calc centroid .. they do soon get wiped out anyway.
			cuSyst_host.PopulateFromTriMesh(pX);
			cuSyst_host.SendToDevice(cuSyst1); // check this is right
			cuSyst2.CopyStructuralDetailsFrom(cuSyst1);
			cuSyst3.CopyStructuralDetailsFrom(cuSyst1);
				// Let's assume these always carry through during GPU runs.
				// It certainly does not work as it stands if you don't populate them all the same, put it that way!!
			printf("sent back re-delaunerized system\n");

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
			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving system...");
				pX->Save(ofn.lpstrFile);
				printf("done\n");
			}
			else {
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
			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving system...");
				pX->SaveText(ofn.lpstrFile);
				printf("done\n");
			}
			else {
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
			if (GetOpenFileName(&ofn) == TRUE)
			{
				pX->Load(ofn.lpstrFile);
				printf("\ndoing nothing...");
			};
		break;

		case ID_RUN_SIMULATIONSTEPS:

			GlobalSwitchBox = 0;
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG1), hWnd, SetupBox);
			// that will not return with steps_remaining unset.
			 
			if (steps_remaining > 0)
				SetTimer(hWnd, 1, 1, NULL); // 1 millisecond delay

			break;

		case ID_RUN_SIMULATIONSTEPS_CPU:

			GlobalSwitchBox = 0;
			steps_remaining_CPU = 1;
			// that will not return with steps_remaining unset.

			if (steps_remaining_CPU > 0)
				SetTimer(hWnd, 2, 1, NULL); // 1 millisecond delay

			break;

		case ID_INITIALISE_ZAPTHEBACK:

			Zap_the_back();
			printf("done");

			RefreshGraphs(*pX, GlobalSpeciesToGraph); // sends data to graphs AND renders them
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);


			break;

		case ID_RUN_STOP:

			steps_remaining = 0;
			steps_remaining_CPU = 0;
			break;
		case ID_INITIALISE_IONISATIONSTEPS:
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;

	case WM_TIMER:
		
		KillTimer(hWnd, wParam);
		report_time(0);
		if (wParam == 1)
		{
			if (bInvoked_cuSyst == false) {
				bInvoked_cuSyst = true;

				pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
				pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos();
//
//				printf("tri 340: %d %d %d \n%1.14E %1.14E \n%1.14E %1.14E \n%1.14E %1.14E\n",
//					pX->T[340].cornerptr[0] - pX->X, pX->T[340].cornerptr[1] - pX->X, pX->T[340].cornerptr[2] - pX->X,
//					pX->T[340].cornerptr[0]->pos.x, pX->T[340].cornerptr[0]->pos.y,
//					pX->T[340].cornerptr[1]->pos.x, pX->T[340].cornerptr[1]->pos.y,
//					pX->T[340].cornerptr[2]->pos.x, pX->T[340].cornerptr[2]->pos.y);
//				printf("tri 340 periodic %d \n", pX->T[340].periodic);
//				getch(); 
				 
				cuSyst_host.InvokeHost();
				cuSyst_host.PopulateFromTriMesh(pX);
				cuSyst_host2.InvokeHost();
				cuSyst_host2.PopulateFromTriMesh(pX);
				 
				//		cuSyst_host.Output("n0.txt");

				PerformCUDA_Invoke_Populate(
					&cuSyst_host,
					NUMVERTICES,
					pX->InnermostFrillCentroidRadius,
					pX->OutermostFrillCentroidRadius,
					pX->numStartZCurrentTriangles,
					pX->numEndZCurrentTriangles);
			}

			// Run 1 step:
			printf("evaltime %1.9E\n", evaltime);

			//	PerformCUDA_RunStepsAndReturnSystem_Debug(&cuSyst_host, &cuSyst_host2, pX, &X3, pXnew);

			PerformCUDA_RunStepsAndReturnSystem(&cuSyst_host);

			//	printf("Stamp GPU over CPU y/n:");
			//	do {
			//		o = getch();
			//	} while ((o != 'y') && (o != 'n'));
			//	printf("%c\n\n", o);
			//	if (o == 'y') 

				// Auto-save system:
			if (GlobalStepsCounter % DATA_SAVE_FREQUENCY == 0)
			{
				sprintf(szFile, "auto%d.dat", GlobalStepsCounter);
				// SAVE cuSyst:
				cuSyst_host.Save(szFile);
			}

			// even number of steps should lead us back to pX having it
			steps_remaining--;
			GlobalStepsCounter++;

			printf("Done steps: %d   ||   Remaining this run: %d\n\n", GlobalStepsCounter, steps_remaining);

			if ((GlobalStepsCounter % GRAPHICS_FREQUENCY == 0) ||
				(GlobalStepsCounter % REDELAUN_FREQUENCY == 0) ||
				(steps_remaining == 0))
			{
				cuSyst_host.PopulateTriMesh(pX); // vertex n is populated into the minor array available on CPU
				printf("pulled back to host\n");
			}
		}
		else {
			pX->Advance(pXnew, &X3);
			temp = pX;
			pX = pXnew;
			pXnew = temp;

			steps_remaining_CPU--;
			GSCCPU++;
			printf("Done steps CPU: %d   ||   Remaining this run: %d\n\n", GSCCPU, steps_remaining_CPU);
			  
			sprintf(buf1001, "autosaveCPU%d.dat", GSCCPU);
			pX->Save(buf1001);
			printf("saved as %s\n", buf1001);
		};
		printf("%s\n", report_time(1));
		  
		if (GlobalStepsCounter % GRAPHICS_FREQUENCY == 0)
		{
			// make video frames:
			for (i = 0; i < NUMAVI; i++)
			{
				printf("i = %d \n", i);
				RefreshGraphs(*pX, GraphFlags[i]); // sends data to graphs AND renders them
															   //	::PlanViewGraphs1(*pX);
				printf(".DISHMOPS.\n");
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
				printf("got to here 7\n");

				if (DXChk(p_backbuffer_surface->GetDC(&surfdc), 100))
					MessageBox(NULL, "GetDC failed", "oh dear", MB_OK);
				//SelectObject(surfdc,surfbit);
				BitBlt(dibdc, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, surfdc, 0, 0, SRCCOPY);
				p_backbuffer_surface->ReleaseDC(surfdc);

				//GetDIBits(dibdc, dib, 0, VIDEO_HEIGHT, dwBits, &bmi, 0);
				// just use lpvBits
				
				// getting hAvi[i] == 0 for the last one.
				// But on debug? No such thing? Same.

				printf("Adding frame to %d : \n", i);
				hresult = WriteFrame(pSinkWriter[i], izStream[i], rtStart);

				//hresult = AddAviFrame(hAvi[i], dib);
				if (hresult != 0) printf("\n******************************************************* \n"
					"hresult = %d\n********************************************** \n", hresult);
				
			};
			rtStart += VIDEO_FRAME_DURATION;

			// sprintf(szFile, "System_%d", GlobalStepsCounter);
			// pX->SaveText(szFile);
		};

		if (GlobalStepsCounter % (AVI_FILE_PINCHOFF_FREQUENCY * GRAPHICS_FREQUENCY) == 0)
		{
			for (i = 0; i < NUMAVI; i++)
			{
				// now have to pinch out avi file and make a new one
				pSinkWriter[i]->Finalize();
				// CloseAvi(hAvi[i]);

				swprintf(buf1000, L"%s%s_%d.mp4", FOLDER, szmp4[i], GlobalStepsCounter);
				//hAvi[i] = CreateAvi(buf1000, AVIFRAMEPERIOD, NULL);
				SafeRelease(&pSinkWriter[i]);
				pSinkWriter[i] = NULL;
				rtStart = 0;
				if (!SUCCEEDED(InitializeSinkWriter(&(pSinkWriter[i]), &(izStream[i]), szmp4[i])))
				{
					printf("Failed to create %ls \n", buf1000);
					getch();
				}

				//hresult = SetAviVideoCompression(hAvi[i], dib, &opts, false, hWnd);
				//if (hresult != 0) printf("\n******************************************************* \n"
				//	"SetAviVideoCompression: hresult = %d\n********************************************** \n", hresult);

			};
		};
		 
		RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
		 
		if (GlobalStepsCounter % REDELAUN_FREQUENCY == 0)
		{
			Setup_residual_array(); // We have not specifically checked that cuSyst1 is the
			// most up-to-date, but it doesn't matter really.

			pX->RefreshVertexNeighboursOfVerticesOrdered();
			long iFlips = pX->Redelaunerize(true, true);
			// Send back to GPU:
			pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();

		//	Appears in lots of places so hard to believe data is not updated.
		//	There is no wrapping on GPU?... or is there?
		//	Need to debug inside routine and find out what it is doing for these two triangles.


			//	pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // Obviates some of our flip calcs to replace tri n,T 
			// not sure if needed .. just for calc centroid .. they do soon get wiped out anyway.
			
			cuSyst_host.PopulateFromTriMesh(pX);// 1. Does it update lists? --- some had to be updated on CPU first.
			 
			// Seems to copy structural information as well as data. n is copied from n_minor on CPU.
			
			//cuSyst1.SendToHost(cuSyst_host2);
			//cuSyst_host.ReportDifferencesHost(cuSyst_host2);
			cuSyst_host.SendToDevice(cuSyst1); 
			cuSyst2.CopyStructuralDetailsFrom(cuSyst1);
			cuSyst3.CopyStructuralDetailsFrom(cuSyst1);
			// Let's assume these always carry through during GPU runs.
			// It certainly does not work as it stands if you don't populate them all the same, put it that way!!

			// We don't actually know which system is pointed to by pX1 that is the initial system
			// for the steps --- so just copy it over all of them
			cuSyst_host.SendToDevice(cuSyst2);
			cuSyst_host.SendToDevice(cuSyst3); 
			// There almost certainly is a better way. But this is unimportant for now.

			printf("sent back re-delaunerized system\n");
			 
			// Now reset A values more carefully in the sent-back system:
			 
			if (iFlips == 0) {
				printf(" NO DELAUNAY FLIPS");
				for (int sj = 0; sj < 10; sj++) printf("-\n");
			}
			if (iFlips > 0) {

				Go_visit_the_other_file();
				
			};
			
		};
		
		if (steps_remaining > 0) {
			SetTimer(hWnd, 1, DELAY_MILLISECS, NULL);
			printf("Waiting %d milliseconds to allow user input.\n", DELAY_MILLISECS);
		};
		if (steps_remaining_CPU > 0) {
			SetTimer(hWnd, 2, DELAY_MILLISECS, NULL);
			printf("Waiting %d milliseconds to allow user input.\n", DELAY_MILLISECS);
		};

		/*
		if (wParam == 1) {
			sprintf(buf1000, "autosaveGPU%d.dat", GlobalStepsCounter);
		} else {
			sprintf(buf1000, "autosaveCPU%d.dat", GSCCPU);
		}
		pX->Save(buf1000);
		printf("saved as %s\n", buf1000);
		
		lowest_vez = 0.0;
		iLow = 0;
		pTri = pX->T;
		for (iMinor = 0; iMinor < BEGINNING_OF_CENTRAL; iMinor++)
		{
			if ((pTri->u8domain_flag == DOMAIN_TRIANGLE) && (pX->pData[iMinor].vez < lowest_vez)) {
				lowest_vez = pX->pData[iMinor].vez;
				iLow = iMinor;
			}
			++pTri;
		}
		printf("Tris: lowest_vez %1.14E iLow %d \n", lowest_vez, iLow);
		iLow = 0;
		lowest_vez = 0.0;
		pVertex = pX->X;
		for (; iMinor < NMINOR; iMinor++)
		{
			if ((pVertex->flags == DOMAIN_VERTEX) && (pX->pData[iMinor].vez < lowest_vez)) {
				lowest_vez = pX->pData[iMinor].vez;
				iLow = iMinor;
			}
			++pVertex;
		}
		printf("Vertices: lowest_vez %1.14E iLow %d \n\n", lowest_vez, iLow);


		printf("save ascii?");
		do {
			o = getch();
		} while ((o != 'y') && (o != 'n'));
		printf("%c\n", o);
		if (o == 'y') {
			sprintf(buf1000, "SaveGPUtext1_trackedAA");
			pX->SaveText(buf1000);
			printf("Ascii file saved %s.\n",buf1000);
		}
		*/

		printf("steps_remaining GPU: %d  CPU: %d\n",steps_remaining, steps_remaining_CPU);
		
		
		break;

	case WM_KEYDOWN:

		switch (wParam)
		{
		case 'W':
			GlobalEye.z += 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'S':
			GlobalEye.z -= 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'A':
			GlobalEye.x -= 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'D':
			GlobalEye.x += 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'E':
			GlobalEye.y += 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'C':
			GlobalEye.y -= 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;

		case 'V':
			GlobalLookat.z -= 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'R':
			GlobalLookat.z += 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'F':
			GlobalLookat.x -= 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'G':
			GlobalLookat.x += 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'T':
			GlobalLookat.y += 0.4f;
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'B':
			GlobalLookat.y -= 0.4f;
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case '+':
			GlobalCutaway = !GlobalCutaway;
			break;
		case 'Y':
		case '<':
			GlobalEye.x = -10.4; GlobalEye.y = 16.4; GlobalEye.z = 44.0;
			GlobalLookat.x = -3.6; GlobalLookat.y = 3.0; GlobalLookat.z = 72.2;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);

			GlobalPlanEye.x = 7.1; GlobalPlanEye.y = 11.5; GlobalPlanEye.z = 71.35;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);

			break;
		case '_':
		case '-':
		case '>':
			GlobalPlanEye.x = 7.0; GlobalPlanEye.y = 14.0; GlobalPlanEye.z = 71.0;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;

		case 'U':
			GlobalPlanEye.z += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'J':
			GlobalPlanEye.z -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'H':
			GlobalPlanEye.x -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'K':
			GlobalPlanEye.x += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'I':
			GlobalPlanEye.y *= 1.25f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'M':
			GlobalPlanEye.y *= 0.8f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
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
			
		case 'Q':
			newEye.z += 5.0f;
			printf("newEye.z %1.9E\n", newEye.z);
			break;
		case 'P':
			newEye.z -= 5.0f;
			printf("newEye.z %1.9E\n", newEye.z);
			break;
		case 'X':
			newEye.y += 5.0f;			
			printf("newEye.y %1.9E\n", newEye.y);
			break;
		case 'Z':
			newEye.y -= 5.0f;
			printf("newEye.y %1.9E\n", newEye.y);
			break;
		case 'O':
			newLookat.z -= 3.0f;
			printf("newLookat.z %1.9E\n", newLookat.z);
			break;
		case ';':
		case ':':
			newLookat.z += 3.0f;
			printf("newLookat.z %1.9E\n", newLookat.z);
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);

		};

		//PlanViewGraphs1(*pX);

		RefreshGraphs(*pX, GlobalSpeciesToGraph); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		break;
	case WM_PAINT:
		// Not sure, do we want to do this?
		//	RefreshGraphs(*pX,); // sends data to graphs AND renders them
		GetUpdateRect(hWnd, &rect, FALSE);
		if (Direct3D.pd3dDevice != NULL)
			Direct3D.pd3dDevice->Present(&rect, &rect, NULL, NULL);

		ValidateRect(hWnd, NULL);
		break;
	case WM_DESTROY:
		DeleteObject(dib);
		DeleteDC(dibdc);
		for (i = 0; i < NUMAVI; i++)
		{
			pSinkWriter[i]->Finalize();
			SafeRelease(&(pSinkWriter[i]));
		}
		// CloseAvi(hAvi[i]);

		//  _controlfp_s(0, cw, _MCW_EM); // Line A
		PerformCUDA_Revoke();

		MFShutdown();
		CoUninitialize();

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
		sprintf(buffer, "New h? (present = %1.10E)", h);
		if (GlobalSwitchBox)
			SetDlgItemText(hDlg, IDC_STATIC, buffer);
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK)
		{
			// try to read data from edit control:
			GetDlgItemText(hDlg, IDC_EDIT1, buffer, 2048);
			if (GlobalSwitchBox == 0)
			{
				// 
				steps_remaining = atoi(buffer);
				if (steps_remaining >= 0)
				{
					EndDialog(hDlg, LOWORD(wParam));
				}
				else {
					MessageBox(NULL, "incorrect value", "Enter a nonnegative integer.", MB_OK);
				};
			}
			else {
				newh = atof(buffer);
				if (newh > 0.0)
				{
					EndDialog(hDlg, LOWORD(wParam));
					sprintf(string, "h = %1.10E\n", newh);
					h = newh;
					MessageBox(NULL, string, "New value of h", MB_OK);
				}
				else {
					MessageBox(NULL, "no good", "Negative h entered", MB_OK);
				};
			};
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

