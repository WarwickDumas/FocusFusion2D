
#ifndef surfacegraph_h
#define surfacegraph_h

#include "d3d.h"
#include "mesh.h"

#define NUMBER_VERTEX_ARRAYS        4
#define HISTORY										4

class surfacegraph
{
public:

	static const int SHADOWMAPRESOLUTION = 4096; // can we go higher?

	long numVerticesTotal, numTrianglesTotal; // dimensioned amts
	long numVertices[NUMBER_VERTEX_ARRAYS];
	long numTriangles[NUMBER_VERTEX_ARRAYS]; // dimensioned amts


	long numVerticesUsed[NUMBER_VERTEX_ARRAYS]; // must be less than the above!
	long numTrianglesUsed[NUMBER_VERTEX_ARRAYS];

	IDirect3DVertexBuffer9* VertexBuffer[NUMBER_VERTEX_ARRAYS];
	IDirect3DIndexBuffer9* IndexBuffer[NUMBER_VERTEX_ARRAYS];

	//IDirect3DTexture9*  texture_grid;

	D3DVIEWPORT9  vp;
	D3DVIEWPORT9  shadow_vp;

	ID3DXEffect* mFX; // able to choose different FX for different surfacegraph objects

	//D3DXMATRIXA16 matWorld;
	D3DXMATRIXA16 matView;
	D3DXMATRIXA16 matProj;
	
	DirLight mLight;
	Mtrl     mWhiteMtrl;
	
	D3DXVECTOR3 Eye,Lookat; // eye position
	D3DXVECTOR3 Eyesph;  // r, phi, theta
	

	// for shadow map:
	IDirect3DSurface9* pSurface; // to be used how?
	
	D3DXMATRIX  mLightVP;
	
	DrawableTex2D* mShadowMap;   
	

	D3DXHANDLE   mhTech;
	D3DXHANDLE   meshTech;
	D3DXHANDLE   mhWVP;
	//D3DXHANDLE   mhWorldInvTrans;
	//D3DXHANDLE   mhWorld;       // assume world matrix = identity
	D3DXHANDLE   mhTex;
	D3DXHANDLE   mhEyePos;         // for specular
	D3DXHANDLE   mhDiffuseMtrl;
	D3DXHANDLE   mhAmbientMtrl;
	D3DXHANDLE   mhLightVector;
	D3DXHANDLE   mhLightDiffuseColour;
	D3DXHANDLE   mhLightAmbientColour;
	D3DXHANDLE   mhLightSpecularColour;
	D3DXHANDLE   mhSpecularMtrl;
	D3DXHANDLE   mhSpecularPower;
	
	D3DXHANDLE   mhBuildShadowMapTech;
	D3DXHANDLE   mhLightWVP;
	D3DXHANDLE   mhShadowMap;
	D3DXHANDLE   mhSwitch;
	
	D3DXHANDLE   mhbCullNone;

	D3DXHANDLE   mhbTransparency;
	D3DXHANDLE   mhfTransparentAlpha;

	D3DXHANDLE   mhColourMax;  // for passing information to shader in case we want to do value/ColourMax

	float ymax,ymin; // for creating graphical y-values.
	float zeroplane, yscale; // for creating graphical y-values.

	float colourmax; // often wanted, and may be different from data max;
	// we use this for the value to be passed to the shader via mhColourMax.
	real store_max, store_min;
	bool label_insist_max, label_insist_min;

	// 4 bools to set before every graph:
	bool boolDisplayKeyButton, boolDisplayInnerMesh, boolDisplayMainMesh, 
		boolDisplayMeshWireframe, boolDisplayScales,
		boolClearZBufferBeforeWireframe,
		bDisplayTimestamp,
		boolDisplayShadow;
	real TickRescaling;

#define FLAG_MESH      0
#define FLAG_SEGUE     1
#define FLAG_VELOCITY  2
#define FLAG_CURRENT  3


	inline surfacegraph() {
		for (int N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
		{
			VertexBuffer[N] = NULL;
			IndexBuffer[N] = NULL;
			numVertices[N] = 0;
			numTriangles[N] = 0;
		};
	
		mShadowMap = NULL;

		bDisplayTimestamp = 0;
		TickRescaling = 1.0;
	};

	/*
	HRESULT Initialise(int , // data points per row if square grid data to be supplied
								 int , int , int , int , float , D3DXVECTOR3 ,
								 bool , 
								 int , // data points in outermost circle if radially distributed data
								 int ,    // number of circles              if radially distributed data
								 float     //  max radius              if radially distributed data
								 );*/
	
	//Call both of these:
	HRESULT InitialiseWithoutBuffers(int vpleft, int vpright, int vpwidth, int vpheight,
											    D3DXVECTOR3 in_Eye, D3DXVECTOR3 Lookat,
		bool bOrtho = false);

	// This one may need things re-dimensioning after
	HRESULT InitialiseBuffers(const TriMesh & X);
	
	HRESULT InitialiseBuffersAux(TriMesh & X, int iLevel, int NTris=0);

	//HRESULT SetDataWithSegue(System & X);
	HRESULT SetDataWithColour(const TriMesh &, int ,int,
		int,int,
		//int offset_data_slim = 0, int offset_vcolour_slim = 0, 
		
		int code = 0);

	HRESULT SetDataWithColourAux(TriMesh & X, int iLevel, int colourflag, int heightflag, int offset_data, int offset_vcolour,
		int NTris = 0 // introduced for debugging
		);

	VOID Render(const char * szTitle, bool RenderTriLabels,
		const TriMesh * pX,
				char * szLinebelow = 0);
	VOID Rendertemp(void);

	VOID RenderAux(const char * szTitle, int const iLabels, const TriMesh * pX, int iLevel);

	HRESULT SetEyePlan(const D3DXVECTOR3 & newEye);
	HRESULT SetEye_NotPlan(const D3DXVECTOR3 & newEye);
	HRESULT SetLookat_NotPlan(const D3DXVECTOR3 & newLookat);
	HRESULT SetEyeAndLookat(const D3DXVECTOR3 & newEye,
																		const D3DXVECTOR3 & newLookat);
	
	void DrawSurface(const char szname[],
				   const int heightflag,
				   const real * var_ptr_0,
				   const int colourflag,
				   const real * var_ptr_c,
				   const bool bDisplayInner,
				   const int code, // graph code, to pass to called routines - sometimes useful
				   const TriMesh * pX // for passing to SetDataWithColour and Render
										// and for working out offsets
				   );

	// helper function:
	
	void inline RenderLabel (char * text, float x, float y, float z, bool extrainfo = false, bool botleft = false, bool bColoured = false);
	void inline RenderLabel2 (char * text, float x, float y, float z, int whichline, unsigned int color = 0xff000000, bool bLong = false);
	void inline RenderText (const char * text,int lines_down);
	void inline RenderLabel3(char * text, float x, float y, float z, int whichline, unsigned int color = 0xff000000);

	~surfacegraph();
};


#endif

