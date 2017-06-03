
#ifndef surfacegraph2_cpp
#define surfacegraph2_cpp

// DrawableTex2D copy paste:
#include "d3d.h"
#include "surfacegraph_tri.h"
#include "FFxtubes.h"
#include "globals.h"

#include "mesh.h"

//extern FixedMesh Fixed;
extern int GlobalWhichLabels;
extern real GlobalRescaling;

#define PI32bit  3.14159268f

extern unsigned int cw;

extern float Historic_max[100][HISTORY]; // if max is falling, use historic maximum for graph.
extern float Historic_min[100][HISTORY];
extern int Historic_powermax[100]; // if max is falling, use historic maximum for graph.
extern int Historic_powermin[100];
extern bool boolGlobalHistory;
extern bool bCullNone;

D3D Direct3D;

void strip_0(char * buffer)
{
	char * p = buffer;
	while (*p != 'E') ++p;
	++p;
	if (*p == '+')
	{
		memmove(p,p+1,strlen(p));
	} else {
		++p;
	}
	if (*p == '0') memmove(p,p+1,strlen(p));
}

DrawableTex2D::DrawableTex2D(UINT width, UINT height, UINT mipLevels,
		D3DFORMAT texFormat, bool useDepthBuffer,
		D3DFORMAT depthFormat, D3DVIEWPORT9& viewport,  bool autoGenMips)
: mTex(0), mRTS(0), mTopSurf(0), mWidth(width), mHeight(height), 
  mMipLevels(mipLevels), mTexFormat(texFormat), mUseDepthBuffer(useDepthBuffer),
  mDepthFormat(depthFormat), mViewPort(viewport), mAutoGenMips(autoGenMips)
{
}

DrawableTex2D::~DrawableTex2D()
{
	onLostDevice();
}

IDirect3DTexture9* DrawableTex2D::d3dTex()
{
	return mTex;
}

void DrawableTex2D::onLostDevice()
{
	SAFE_RELEASE2(mTex);
	SAFE_RELEASE2(mRTS);
	SAFE_RELEASE2(mTopSurf);
}

void DrawableTex2D::onResetDevice()
{
	UINT usage = D3DUSAGE_RENDERTARGET;
	if(mAutoGenMips)
		usage |= D3DUSAGE_AUTOGENMIPMAP;

	DXChk(D3DXCreateTexture(Direct3D.pd3dDevice, mWidth, mHeight, mMipLevels, usage, mTexFormat, D3DPOOL_DEFAULT, &mTex));
	DXChk(D3DXCreateRenderToSurface(Direct3D.pd3dDevice, mWidth, mHeight, mTexFormat, mUseDepthBuffer, mDepthFormat, &mRTS));
	DXChk(mTex->GetSurfaceLevel(0, &mTopSurf));
}

void DrawableTex2D::beginScene()
{
	mRTS->BeginScene(mTopSurf, &mViewPort);
}

void DrawableTex2D::endScene()
{
	mRTS->EndScene(D3DX_FILTER_NONE);
}

// vertex.cpp copy paste:

IDirect3DVertexDeclaration9* VertexPos::Decl = 0;
IDirect3DVertexDeclaration9* VertexCol::Decl = 0;
IDirect3DVertexDeclaration9* VertexPN::Decl  = 0;
IDirect3DVertexDeclaration9* VertexPNT::Decl = 0;
IDirect3DVertexDeclaration9* VertexPNT3::Decl = 0;
IDirect3DVertexDeclaration9* VertexPNf::Decl = 0;

void D3D::InitAllVertexDeclarations()
{
	//===============================================================
	// VertexPos

	D3DVERTEXELEMENT9 VertexPosElements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexPosElements, &VertexPos::Decl));

	//===============================================================
	// VertexCol

	D3DVERTEXELEMENT9 VertexColElements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexColElements, &VertexCol::Decl));

	//===============================================================
	// VertexPN

	D3DVERTEXELEMENT9 VertexPNElements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexPNElements, &VertexPN::Decl));

	//===============================================================
	// VertexPNT

	D3DVERTEXELEMENT9 VertexPNTElements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0},
		{0, 24, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexPNTElements, &VertexPNT::Decl));

	D3DVERTEXELEMENT9 VertexPNT3Elements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0},
		{0, 24, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexPNT3Elements, &VertexPNT3::Decl));

	
	
	D3DVERTEXELEMENT9 VertexPNfElements[] = 
	{
		{0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0},
		{0, 24, D3DDECLTYPE_FLOAT1, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};	
	DXChk(pd3dDevice->CreateVertexDeclaration(VertexPNfElements, &VertexPNf::Decl));
}

void D3D::DestroyAllVertexDeclarations()
{
	SAFE_RELEASE2(VertexPos::Decl);
	SAFE_RELEASE2(VertexCol::Decl);
	SAFE_RELEASE2(VertexPN::Decl);
	SAFE_RELEASE2(VertexPNT::Decl);
	SAFE_RELEASE2(VertexPNT3::Decl);
	SAFE_RELEASE2(VertexPNf::Decl);
}




HRESULT surfacegraph::InitialiseWithoutBuffers(int vpleft, int vptop, int vpwidth, int vpheight,
											   D3DXVECTOR3 in_Eye, D3DXVECTOR3 in_Lookat)
{
	
	_controlfp_s(&cw, _EM_INEXACT | _EM_UNDERFLOW | _EM_ZERODIVIDE , _MCW_EM);


	shadow_vp.X = 0;
	shadow_vp.Y = 0;
	shadow_vp.Width = SHADOWMAPRESOLUTION;
	shadow_vp.Height = SHADOWMAPRESOLUTION;
	shadow_vp.MinZ = 0.0f;
	shadow_vp.MaxZ = 1.0f;
	
	mShadowMap = new DrawableTex2D(SHADOWMAPRESOLUTION, SHADOWMAPRESOLUTION, 1, D3DFMT_R32F, true, D3DFMT_D24X8, shadow_vp, false);
	
	//mShadowMap->onResetDevice(); // do any good?

	vp.X = vpleft;
	vp.Y = vptop;
	vp.Width = vpwidth;
	vp.Height = vpheight;
	vp.MinZ = 0.0f;
	vp.MaxZ = 0.0f;
	
	float AspectRatio = ((float)vp.Width)/(float)vp.Height;
	
	
	// Create the FX from a .fx file.
	ID3DXBuffer* errors = 0;
	DXChk(D3DXCreateEffectFromFile(Direct3D.pd3dDevice, "shadow5.fx", //"surf.fx", 
		0, 0, D3DXSHADER_DEBUG | D3DXFX_LARGEADDRESSAWARE, 0, &mFX, &errors));
	// note debug
	
	if( errors ) 
	{
		MessageBox(0, (char*)errors->GetBufferPointer(), 0, 0);	
		PostQuitMessage(3000000);
	};
	if (mFX == 0) {
		MessageBox(0, "oh","mFX==0",MB_OK);
		PostQuitMessage(1000000);
	};
		
	// Obtain handles.
	
	meshTech = mFX->GetTechniqueByName("MeshTech");

	// This will be overwritten the first time RefreshGraphs is called.

	//if (flag == FLAG_SEGUE)
	//{
		mhTech = mFX->GetTechniqueByName("SegueTech");   // Yep
	//} else {
	//	if (flag == FLAG_VELOCITY)
	//	{
	//		mhTech = mFX->GetTechniqueByName("VelociTech");
	//	} else {
	//		if (flag == FLAG_MESH)
	//		{
	//			mhTech = mFX->GetTechniqueByName("MeshTech");
	//		} else {
	//			if (flag == FLAG_CURRENT)
	//			{
	//				mhTech = mFX->GetTechniqueByName("XYZTech");
	//			} else {
	//			//	if (flag == )
	//			//	{
	//			//	}
	//				MessageBox(NULL,"strewth matey","flag not set",MB_OK);
	//			};
	//		};
	//	};
	//};

	mhWVP  = mFX->GetParameterByName(0, "gWVP");
	//mhTex = mFX->GetParameterByName(0, "WarwickTexture");
	
	mhLightVector = mFX->GetParameterByName(0, "LightVecW");

	mhLightDiffuseColour = mFX->GetParameterByName(0, "LightDiffuseColor");
	mhDiffuseMtrl = mFX->GetParameterByName(0, "DiffuseMtrl");

	mhLightAmbientColour = mFX->GetParameterByName(0, "LightAmbientColor");
	mhAmbientMtrl = mFX->GetParameterByName(0, "AmbientMtrl");

	mhLightSpecularColour = mFX->GetParameterByName(0, "LightSpecularColor");
	mhSpecularMtrl = mFX->GetParameterByName(0, "SpecularMtrl");
	mhSpecularPower = mFX->GetParameterByName(0, "SpecularPower");
	mhEyePos = mFX->GetParameterByName(0, "gEyePosW");
	
	mhBuildShadowMapTech = mFX->GetTechniqueByName("ShadowMapTech");
	mhLightWVP           = mFX->GetParameterByName(0, "gLightWVP");
	mhShadowMap          = mFX->GetParameterByName(0, "gShadowMap");

	mhSwitch             = mFX->GetParameterByName(0, "iswitch");

	mhColourMax            = mFX->GetParameterByName(0, "Maxv");

	mhbCullNone = mFX->GetParameterByName(0,"bCullNone");

	mhbTransparency = mFX->GetParameterByName(0,"bTransparent");
	mhfTransparentAlpha = mFX->GetParameterByName(0,"fTransparentAlpha");

	mShadowMap->onResetDevice();

	
	//static D3DXVECTOR3 vLookatPt( 0.0f, 1.2f, 0.0f );
	static D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );
    
	Eye = in_Eye;
	Lookat = in_Lookat;

	//Eyesph.x = 8.4f;				// r
	//Eyesph.y = 3.1415927f*0.34f;	// phi
	//Eyesph.z = -3.1415927f*0.3f;	// theta 
	//Eye.x = Eyesph.x * (sin(Eyesph.z));
	//Eye.y = Eyesph.x * (sin(Eyesph.y)*cos(Eyesph.z));
	//Eye.z = -Eyesph.x * (cos(Eyesph.y)*cos(Eyesph.z));
	
	// convert from spherical coordinates to Cartesian:
	// x means r, y means phi (affects y coord most), z means theta (affects z coord most)
// theta is angle around circle that would be made in the x-y plane
// phi is angle around circle in the y-z plane
// theta does not affect z but it does affect how phi affects z
// phi does not affect x
	// note: this is a rel pos: if we translate vLookatPt then we should translate Eye
	
	//D3DXMatrixScaling( &matWorld,1.0f,1.0f,1.0f );
	// changed from identity
	//pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );
	
	D3DXMatrixLookAtLH( &matView, &Eye, &Lookat, &vUpVec );
	
	// For the projection matrix, we set up a perspective transform (which
	// transforms geometry from 3D view space to 2D viewport space, with
	// a perspective divide making objects smaller in the distance). To build
	// a perpsective transform, we need the field of view (1/4 pi is common),
	// the aspect ratio, and the near and far clipping planes (which define at
	// what distances geometry should be no longer be rendered).
	
	// for some reason this throws "dividing by zero" exception.

	D3DXMatrixPerspectiveFovLH( &matProj, D3DX_PI / 6.0f, AspectRatio, NEAR_CLIPPING_PLANE, FAR_CLIPPING_PLANE );
	// apparently in rel coords - DO NOT set near clipping plane value to be negative
	
	
	// So here we have set:
	// matWorld, matView, matProj
	
	// set up mLight
	mLight.ambient = D3DXCOLOR(0.5f, 0.5f, 0.5f, 1.0f);
	mLight.diffuse = D3DXCOLOR(0.5f, 0.5f, 0.5f, 1.0f);
	mLight.spec    = D3DXCOLOR(0.6f, 0.6f, 0.6f, 1.0f);
	
	// setting diffuse intensity > 1 does perform as expected
	
	// set up mWhiteMtrl
	mWhiteMtrl.ambient = WHITE*1.0f;
	mWhiteMtrl.diffuse = WHITE*1.0f;
	mWhiteMtrl.spec    = WHITE*1.0f;
	mWhiteMtrl.specPower = 3.2f;
	
	// Shadow mapping:
	
	D3DXMATRIX lightView;
	D3DXVECTOR3 lightPosW(20.0f, 14.0f, -10.0f+DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*xzscale);
	D3DXVECTOR3 lightTargetW(0.0f, 0.0f, DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE*xzscale);//e.g.
//	D3DXVECTOR3 lightUpW(1.0f, 0.0f, 0.0f);   // um why is up being set to 1,0,0 ? Try 0,1,0
	D3DXVECTOR3 lightUpW(0.0f, 1.0f, 0.0f);	

	D3DXMatrixLookAtLH(&lightView, &lightPosW, &lightTargetW, &lightUpW);

	D3DXMATRIX lightLens;
	float lightFOV = D3DX_PI*0.25f; // see if this changes anything
	D3DXMatrixPerspectiveFovLH(&lightLens, lightFOV, AspectRatio, NEAR_CLIPPING_PLANE, FAR_CLIPPING_PLANE);
	D3DXMatrixOrthoLH(&lightLens,25.0f,25.0f,NEAR_CLIPPING_PLANE,FAR_CLIPPING_PLANE);
	// 10,10 ?
	mLightVP = lightView*lightLens;
	mLight.dirW = lightTargetW-lightPosW;
	D3DXVec3Normalize(&mLight.dirW, &mLight.dirW);
	
	_controlfp_s(0,cw, _MCW_EM);

	return S_OK;
}

HRESULT surfacegraph::SetEyePlan(const D3DXVECTOR3 & newEye)
{
	_controlfp_s(&cw, _EM_INEXACT | _EM_UNDERFLOW | _EM_ZERODIVIDE , _MCW_EM);

	D3DXVECTOR3 vUpVec( 0.0f, 0.0f, 1.0f );   // DIFFERENT

	Eye = newEye;
	
	Lookat.x = Eye.x;
	Lookat.y = 0.0;
	Lookat.z = Eye.z*1.000001; // quite possible that where they are equal, this is making us spin?

	D3DXMatrixLookAtLH( &matView, &Eye, &Lookat, &vUpVec );
	
	_controlfp_s(0,cw , _MCW_EM);

	return S_OK;
}

HRESULT surfacegraph::SetEyeAndLookat(const D3DXVECTOR3 & newEye,
							 const D3DXVECTOR3 & newLookat)
{
	_controlfp_s(&cw, _EM_INEXACT | _EM_UNDERFLOW | _EM_ZERODIVIDE , _MCW_EM);

	D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );  

	Eye = newEye;
	Lookat = newLookat;
	
	D3DXMatrixLookAtLH( &matView, &Eye, &Lookat, &vUpVec );
	
	_controlfp_s(0,cw , _MCW_EM);

	return S_OK;
}

HRESULT surfacegraph::SetEye_NotPlan(const D3DXVECTOR3 & newEye)
{
	_controlfp_s(&cw, _EM_INEXACT | _EM_UNDERFLOW | _EM_ZERODIVIDE , _MCW_EM);

	D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );  

	Eye = newEye;
	
	D3DXMatrixLookAtLH( &matView, &Eye, &Lookat, &vUpVec );
	
	_controlfp_s(0,cw , _MCW_EM);

	return S_OK;
}

HRESULT surfacegraph::SetLookat_NotPlan(const D3DXVECTOR3 & newLookat)
{
	_controlfp_s(&cw, _EM_INEXACT | _EM_UNDERFLOW | _EM_ZERODIVIDE , _MCW_EM);

	D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );  

	Lookat = newLookat;
	
	D3DXMatrixLookAtLH( &matView, &Eye, &Lookat, &vUpVec );
	_controlfp_s(0, cw, _MCW_EM);

	return S_OK;
}


HRESULT surfacegraph::InitialiseBuffers(const TriMesh & X)
{
	int j,failvertex,failindex;

	// Plan to deal with too large number of vertices for a buffer:
	
	// Pick a radius, split by radius, (get acceptable number in band),
	// include those that are neighbours outside the radius;
	// use triangles that apply to only these vertices.

	// Cannot assume the vertices are in any particular order.

	// For now we stuff all into one buffer.
	long numTrianglesKey;

	long numVerticesKey = X.GetNumKeyVerticesGraphics(&numTrianglesKey);

	// Dimension buffer[0] for key. 

	if ((numVerticesKey != numVertices[0]) || (numTrianglesKey != numTriangles[0]))
	{
		SAFE_RELEASE2(VertexBuffer[0]);

		failvertex = DXChk(
						Direct3D.pd3dDevice->CreateVertexBuffer(
						numVerticesKey*sizeof(VertexPNT3),
						D3DUSAGE_WRITEONLY,
						0,	
						D3DPOOL_MANAGED,
						&VertexBuffer[0],
						0),1);

		SAFE_RELEASE2(IndexBuffer[0]);

		failindex = DXChk( 
					Direct3D.pd3dDevice->CreateIndexBuffer(
						numTrianglesKey*3*sizeof(DWORD),
						D3DUSAGE_WRITEONLY,
						D3DFMT_INDEX32,
						D3DPOOL_MANAGED,
						&IndexBuffer[0],
						0),2);

		if (failvertex || failindex)
		{
			MessageBox(NULL,"serious probs","graphics mem not allocated 0",MB_OK);
			return E_FAIL;
		};
		
		numVertices[0] = numVerticesKey;
		numTriangles[0] = numTrianglesKey; // successfully dimmed this amt.

	};

	// Other times we might do something else with the key buffer.

	// Start graph array at buffer 1.

	// This quick test for now; when we use more arrays, we need to store total in another variable...

	if (this->boolDisplayInnerMesh) {
		numVerticesTotal = X.numVertices;
		numTrianglesTotal = X.numTriangles;
		Triangle * pTri = X.T;
		for (long iTri = 0; iTri < X.numTriangles; iTri++)
		{
			if (pTri->periodic) numVerticesTotal+=2; // add some for periodic
			++pTri;
		};
	} else {
		numVerticesTotal = X.numDomainVertices;
		// count triangles that do not involve inner vertices.

		// At a future time to be revealed, we shall try to include tri 'centres' on the insulator
		// in the graph. But not right now.

		// count through triangles to see which ones are in domain.
		numTrianglesTotal = 0;
		Triangle * pTri = X.T;
		for (long iTri = 0; iTri < X.numTriangles; iTri++)
		{
			if (pTri->u8domain_flag == DOMAIN_TRIANGLE){
				numTrianglesTotal++;
				if (pTri->periodic) numVerticesTotal+=2; // add some for periodic
			};
			++pTri;
		};
	};
	if (numVerticesTotal > VERTICES_PER_ARRAY) 
	{
		printf("Warning -- too many vertices for array? Code needs adding.\n");
	};
	{
		if( (numVerticesTotal != numVertices[1]) || (numTrianglesTotal != numTriangles[1]))
		{
			SAFE_RELEASE2(VertexBuffer[1]);
			failvertex = DXChk(
							Direct3D.pd3dDevice->CreateVertexBuffer(
							numVerticesTotal*sizeof(VertexPNT3),
							D3DUSAGE_WRITEONLY,
							0,	
							D3DPOOL_MANAGED,
							&VertexBuffer[1],
							0),1);

			SAFE_RELEASE2(IndexBuffer[1]);
			failindex = DXChk( 
						Direct3D.pd3dDevice->CreateIndexBuffer(
							numTrianglesTotal*3*sizeof(DWORD),
							D3DUSAGE_WRITEONLY,
							D3DFMT_INDEX32,
							D3DPOOL_MANAGED,
							&IndexBuffer[1],
							0),2);
		

			if (failvertex || failindex)
			{
				MessageBox(NULL,"serious probs","graphics mem not allocated",MB_OK);
				return E_FAIL;
			};
			
			numVertices[1] = numVerticesTotal;
			numTriangles[1] = numTrianglesTotal; // successfully dimmed this amt.


			// where is texture created for shadow map?


			// For transparent triangles:
			// take 1/3 of size.

			SAFE_RELEASE2(IndexBuffer[2]);
			failindex = DXChk( 
						Direct3D.pd3dDevice->CreateIndexBuffer(
							numTrianglesTotal*sizeof(DWORD),
							D3DUSAGE_WRITEONLY,
							D3DFMT_INDEX32,
							D3DPOOL_MANAGED,
							&IndexBuffer[2],
							0),2);

			// dummy: allocate 1 vertex only
			SAFE_RELEASE2(VertexBuffer[2]);
			failvertex = DXChk(
							Direct3D.pd3dDevice->CreateVertexBuffer(
							1*sizeof(VertexPNT3),
							D3DUSAGE_WRITEONLY,
							0,	
							D3DPOOL_MANAGED,
							&VertexBuffer[2],
							0),1);

			if (failvertex || failindex)
			{
				MessageBox(NULL,"serious probs","graphics mem not allocated",MB_OK);
				return E_FAIL;
			};

			numTriangles[2] = numTrianglesTotal/3; // successfully dimmed this amt.
			

		// er, what is the following used for?
		//????????????????????????????????????????

		//D3DSURFACE_DESC textureDesc;
		//texture_grid->GetLevelDesc(0, &textureDesc);
		//if ( textureDesc.Format != D3DFMT_X8R8G8B8 ) return E_FAIL;
		
		};
	};

	return S_OK;
}

HRESULT surfacegraph::InitialiseBuffersAux(TriMesh & X, int iLevel,
										   int NTris)
{
	int j,failvertex,failindex;// ,N

//	numVertices *= 2;
//	numTriangles *= 2;    // this is the number to dimension

	// set Number_Rows_Vertex_Array[] -- we might as well do here, it's not expensive
//	j = NUMBER_BLOCKS_Y;
//	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
//	{
//		if (j >= BLOCK_ROWS_PER_VERTEX_ARRAY)
//		{
//			j -= BLOCK_ROWS_PER_VERTEX_ARRAY;
//			Number_Rows_Vertex_Array[N] = BLOCK_ROWS_PER_VERTEX_ARRAY;
//		} else {
//			Number_Rows_Vertex_Array[N] = j;
//			j = 0;
//		};
//	};
			
	if (bScrewPinch) {
		numVerticesTotal = X.numAuxVertices[iLevel];
	} else {
		// allow for periodic extras:
		numVerticesTotal = X.GetNumVerticesGraphicsAux(iLevel);
	};
	numTrianglesTotal = X.numAuxTriangles[iLevel]; // only display the actual ones, once.
		
	numVerticesUsed[0] = 0;
	numTrianglesUsed[0] = 0;
	numVerticesUsed[1] = 0;
	numTrianglesUsed[1] = 0;

	if( (numVerticesTotal != numVertices[1]) || (numTrianglesTotal != numTriangles[1]))
	{

		SAFE_RELEASE2(VertexBuffer[1]);

		failvertex = DXChk(
						Direct3D.pd3dDevice->CreateVertexBuffer(
						numVerticesTotal*sizeof(VertexPNT3),
						D3DUSAGE_WRITEONLY,
						0,	
						D3DPOOL_MANAGED,
						&VertexBuffer[1],
						0),1);

		SAFE_RELEASE2(IndexBuffer[1]);

		failindex = DXChk( 
					Direct3D.pd3dDevice->CreateIndexBuffer(
						numTrianglesTotal*3*sizeof(DWORD),
						D3DUSAGE_WRITEONLY,
						D3DFMT_INDEX32,
						D3DPOOL_MANAGED,
						&IndexBuffer[1],
						0),2);

		if (failvertex || failindex)
		{
			MessageBox(NULL,"serious probs","graphics mem not allocated, aux",MB_OK);
			return E_FAIL;
		};
		
		numVertices[1] = numVerticesTotal;
		numTriangles[1] = numTrianglesTotal; // successfully dimmed this amt.
		if (NTris > 0) numTriangles[1] = NTris;//+50
		//if (numTriangles[1] > numTrianglesTotal)
		//		numTriangles[1] = numTrianglesTotal; // 
	};

	return S_OK;
}
/*
HRESULT surfacegraph::Initialise(int nperRow, // data points per row if square grid data to be supplied
								 int left, int top, int width, int height, float fscale, D3DXVECTOR3 in_Eye,
								 bool Polar = 0, 
								 int number_in_outermost = 0, // data points in outermost circle if radially distributed data
								 int number_circles = 0,    // number of circles              if radially distributed data
								 float radius_outermost = 0.0f    //  max radius              if radially distributed data
								 )
{
	float distance;
	int i,j;
	float radius;

	shadow_vp.X = 0;
	shadow_vp.Y = 0;
	shadow_vp.Width = SHADOWMAPRESOLUTION;
	shadow_vp.Height = SHADOWMAPRESOLUTION;
	shadow_vp.MinZ = 0.0f;
	shadow_vp.MaxZ = 1.0f;
	
	mShadowMap = new DrawableTex2D(SHADOWMAPRESOLUTION, SHADOWMAPRESOLUTION, 1, D3DFMT_R32F, true, D3DFMT_D24X8, shadow_vp, false);
	
	//mShadowMap->onResetDevice(); // do any good?

	vp.X = left;
	vp.Y = top;
	vp.Width = width;
	vp.Height = height;
	vp.MinZ = 0.0f;
	vp.MaxZ = 0.0f;

	float GlobalAspectRatio = ((float)vp.Width)/(float)vp.Height;
	

	// Create the FX from a .fx file.
	ID3DXBuffer* errors = 0;
	D3DXCreateEffectFromFile(Direct3D.pd3dDevice, "test1.fx", 
		0, 0, D3DXSHADER_DEBUG, 0, &mFX, &errors);
	// note debug

	if( errors ) 
		MessageBox(0, (char*)errors->GetBufferPointer(), 0, 0);	
	
	// Obtain handles.
	
	mhTech = mFX->GetTechniqueByName("WarwickTech");
	mhWVP  = mFX->GetParameterByName(0, "gWVP");
	mhTex = mFX->GetParameterByName(0, "WarwickTexture");

	mhLightVector = mFX->GetParameterByName(0, "LightVecW");

	mhLightDiffuseColour = mFX->GetParameterByName(0, "LightDiffuseColor");
	mhDiffuseMtrl = mFX->GetParameterByName(0, "DiffuseMtrl");

	mhLightAmbientColour = mFX->GetParameterByName(0, "LightAmbientColor");
	mhAmbientMtrl = mFX->GetParameterByName(0, "AmbientMtrl");

	mhLightSpecularColour = mFX->GetParameterByName(0, "LightSpecularColor");
	mhSpecularMtrl = mFX->GetParameterByName(0, "SpecularMtrl");
	mhSpecularPower = mFX->GetParameterByName(0, "SpecularPower");
	mhEyePos = mFX->GetParameterByName(0, "gEyePosW");
	
	mhBuildShadowMapTech = mFX->GetTechniqueByName("BuildShadowMapTech");
	mhLightWVP           = mFX->GetParameterByName(0, "gLightWVP");
	mhShadowMap          = mFX->GetParameterByName(0, "gShadowMap");

	mhSwitch             = mFX->GetParameterByName(0, "iswitch");



	mShadowMap->onResetDevice();

	
	static D3DXVECTOR3 vLookatPt( 0.0f, 1.2f, 0.0f );
	static D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );
    
	Eye = in_Eye;

	//Eyesph.x = 8.4f;				// r
	//Eyesph.y = 3.1415927f*0.34f;	// phi
	//Eyesph.z = -3.1415927f*0.3f;	// theta 
	//Eye.x = Eyesph.x * (sin(Eyesph.z));
	//Eye.y = Eyesph.x * (sin(Eyesph.y)*cos(Eyesph.z));
	//Eye.z = -Eyesph.x * (cos(Eyesph.y)*cos(Eyesph.z));
	

	// convert from spherical coordinates to Cartesian:
	// x means r, y means phi (affects y coord most), z means theta (affects z coord most)
// theta is angle around circle that would be made in the x-y plane
// phi is angle around circle in the y-z plane
// theta does not affect z but it does affect how phi affects z
// phi does not affect x
	// note: this is a rel pos: if we translate vLookatPt then we should translate Eye
	
	//D3DXMatrixScaling( &matWorld,1.0f,1.0f,1.0f );
	// changed from identity
	//pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );
	
	D3DXMatrixLookAtLH( &matView, &Eye, &vLookatPt, &vUpVec );
	
	// For the projection matrix, we set up a perspective transform (which
	// transforms geometry from 3D view space to 2D viewport space, with
	// a perspective divide making objects smaller in the distance). To build
	// a perpsective transform, we need the field of view (1/4 pi is common),
	// the aspect ratio, and the near and far clipping planes (which define at
	// what distances geometry should be no longer be rendered).
	D3DXMatrixPerspectiveFovLH( &matProj, D3DX_PI / 4.0f, GlobalAspectRatio, 1.0f, 12.0f );
	// apparently in rel coords - DO NOT set near clipping plane value to be negative
	


	// So here we have set:
	// matWorld, matView, matProj
	
	// set up mLight
	mLight.dirW    = D3DXVECTOR3(-1.0f, -0.5f, 0.5f);// remember y is height
	D3DXVec3Normalize(&mLight.dirW, &mLight.dirW);
	mLight.ambient = D3DXCOLOR(0.4f, 0.4f, 0.4f, 1.0f);
	mLight.diffuse = D3DXCOLOR(0.6f, 0.6f, 0.6f, 1.0f);
	mLight.spec    = D3DXCOLOR(0.5f, 0.5f, 0.5f, 1.0f);
	
	// setting diffuse intensity > 1 does perform as expected
	
	// set up mWhiteMtrl
	mWhiteMtrl.ambient = WHITE*1.0f;
	mWhiteMtrl.diffuse = WHITE*1.0f;
	mWhiteMtrl.spec    = WHITE*1.0f;
	mWhiteMtrl.specPower = 8.0f;
	
	// Shadow mapping:
	
	D3DXMATRIX lightView;
	D3DXVECTOR3 lightPosW(20.0f, 10.0f, -10.0f);
	D3DXVECTOR3 lightTargetW(0.0f, 0.0f, 0.0f);
//	D3DXVECTOR3 lightUpW(1.0f, 0.0f, 0.0f);   // um why is up being set to 1,0,0 ? Try 0,1,0
	D3DXVECTOR3 lightUpW(0.0f, 1.0f, 0.0f);	

	D3DXMatrixLookAtLH(&lightView, &lightPosW, &lightTargetW, &lightUpW);
		
	D3DXMATRIX lightLens;
	float lightFOV = D3DX_PI*0.25f;
	D3DXMatrixPerspectiveFovLH(&lightLens, lightFOV, 1.0f, 1.0f, 100.0f);
	
	mLightVP = lightView*lightLens;
	

		
	// From here, create different functions for Cartesian and polar

	// just have a split

	if (Polar == 0)
	{

		perRow = nperRow;

		numVertices = perRow*perRow;
		numTriangles = 2*(perRow-1)*(perRow-1);
	
		texWidth = perRow;
		texHeight = perRow;
		
		definedindices = 0;

	} else {
		perRow = nperRow;

		Ncircles = number_circles;
		// infer how many are in each interior circle accordingly
		
		distance = PI32bit*radius_outermost/(float)number_in_outermost;
				
		numVertices = 0;
		numTriangles = 0;
		
		number_in_circle = new int[Ncircles];

		for (j = number_circles-1; j > 0; j--)
		{
			radius = radius_outermost*((float)j/(float)(number_circles-1));
			
			number_in_circle[j] = (int)(PI32bit*radius/(float)distance)+1;
			if (j == number_circles-1) number_in_circle[j] = number_in_outermost;
			
			numVertices += number_in_circle[j];
			// number of triangles: there will be one on the interior side of this circle
			// for every edge in this circle. // so number-1 ;
			// There will also be one facing every point here
			
		}
		
		// circle 0 just contains the point at the centre:
		numVertices += 1;
		number_in_circle[0] = 1;
		

		for (j = number_circles-1; j > 0; j--)
		{
			numTriangles += number_in_circle[j]+number_in_circle[j-1];

		};
		numTriangles--; // circle of 1 leads to 0 triangles

		// how to handle texture?
		// can we give texture coords that are something strange?
		// Should still be Cartesian:

		texWidth = perRow;
		texHeight = perRow;
		
		indexinner = new int[numVertices];  // choose the nearest point Clockwise from this one, on the inner circle
		indexclockwise = new int[numVertices]; // the nearest point Clockwise on this same circle
		definedindices = 1;

	};
	
	if (
	
	DXChk(
	Direct3D.pd3dDevice->
		CreateVertexBuffer(
			numVertices*sizeof(VertexPNT),
			D3DUSAGE_WRITEONLY,
			0,
			D3DPOOL_MANAGED,
			&TerrainVertexBuffer,
			0)
								,1)
		+ 
	DXChk( 
	Direct3D.pd3dDevice->
		CreateIndexBuffer(
			numTriangles*3*sizeof(DWORD),
			D3DUSAGE_WRITEONLY,
			D3DFMT_INDEX32,
			D3DPOOL_MANAGED,
			&TerrainIndexBuffer,
			0)
								,2)
		+
		
	DXChk(
		D3DXCreateTexture(
		   Direct3D.pd3dDevice, 
		   texWidth,texHeight, 0,0,D3DFMT_X8R8G8B8,D3DPOOL_MANAGED,
		   &texture_grid)
							    ,3)			
		)
		return E_FAIL;
	
	D3DSURFACE_DESC textureDesc;
	texture_grid->GetLevelDesc(0, &textureDesc);
	if ( textureDesc.Format != D3DFMT_X8R8G8B8 ) return E_FAIL;
	
	nscale = fscale;

	return S_OK;

}
*/


HRESULT surfacegraph::SetDataWithColour(const TriMesh & X, 
										int colourflag, int heightflag, 
										int offset_data, int offset_vcolour,
										//int offset_data_slim, int offset_vcolour_slim, 
										int code)
										// need to add code so that we can check historic data max and for absolute per-variable colour scales
{

	DWORD * indices[NUMBER_VERTEX_ARRAYS];
	VertexPNT3 * vertices[NUMBER_VERTEX_ARRAYS];
	//int numIndicesUsed[NUMBER_VERTEX_ARRAYS]; // not used
	//int numVerticesUsed[NUMBER_VERTEX_ARRAYS]; // class member

	// Moved here from CreateSurfaceGraphs
	// because it depends on boolDisplayInnerMesh being set correctly:
	this->InitialiseBuffers(X);

	int i,N;
	real maximum, minimum, max2, min2;

	if (heightflag != FLAG_FLAT_MESH)
	{
		if (heightflag == FLAG_VELOCITY_HEIGHT)
		{
			maximum = X.ReturnL4_Velocity(offset_data,
				this->boolDisplayInnerMesh);		
			store_max = X.ReturnMaximumVelocity(offset_data,
				this->boolDisplayInnerMesh);
			
			minimum = 0.0;
			this->store_min = 0.0f;
		} else {
			if (heightflag == FLAG_VEC3_HEIGHT) 
			{ 
			//	maximum = X.ReturnMaximum3DMagnitude(offset_data,
			//		this->boolDisplayInnerMesh);
				maximum = X.ReturnL4_3DMagnitude(offset_data, 
					this->boolDisplayInnerMesh);
				store_max = X.ReturnMaximum3DMagnitude(offset_data, 
					this->boolDisplayInnerMesh);
				
				minimum = 0.0;
				this->store_min = 0.0f;
			} else {
				X.ReturnMaxMinData(offset_data, &store_max, &store_min,
					this->boolDisplayInnerMesh);
				
				X.ReturnL5Data(offset_data, &maximum, &minimum,
					this->boolDisplayInnerMesh);
			};
		} 
	} else {

		//  plan view flat mesh case:
		// ---------------------------
		zeroplane = 0.0f;
		yscale = 1.0f;
		
		if (colourflag == FLAG_VELOCITY_COLOUR)
		{
			maximum = X.ReturnL4_Velocity(offset_data,
				this->boolDisplayInnerMesh);		
			store_max = X.ReturnMaximumVelocity(offset_data,
				this->boolDisplayInnerMesh);
				
			minimum = 0.0;
			this->store_min = 0.0f;
		} else {
			if (colourflag == FLAG_CURRENT_COLOUR) 
			{ 
				maximum = X.ReturnL4_3DMagnitude(offset_data, 
					this->boolDisplayInnerMesh);
				store_max = X.ReturnMaximum3DMagnitude(offset_data, 
					this->boolDisplayInnerMesh);
				minimum = 0.0;
				this->store_min = 0.0f;
			} else {
				X.ReturnMaxMinData(offset_data, &store_max, &store_min,
					this->boolDisplayInnerMesh);
				X.ReturnL5Data(offset_data, &maximum, &minimum,
					this->boolDisplayInnerMesh);
			};
		};
	};

	if (_isnan(maximum) || (!_finite(maximum)) || _isnan(minimum) || (!_finite(minimum)) )
	{
		printf("maximum %1.5E minimum %1.5E offset %d ",maximum, minimum, offset_data);
		getch();
	};	
	
	// Now decide on the actual max to use: 
	int powermax, powermin;
	if (maximum > 0.0) {
		real logmaxbase_ours = log(maximum)/log(GRAPH_SCALE_GEOMETRIC_INCREMENT);
		powermax = (int)logmaxbase_ours+1;

		if ((boolGlobalHistory) && (powermax == Historic_powermax[code]-1))
			powermax++;

		maximum = pow(GRAPH_SCALE_GEOMETRIC_INCREMENT,powermax);
	} else {
		powermax = 0;
	};
	if (minimum < 0.0) {
		real logminbase_ours = log(-minimum)/log(GRAPH_SCALE_GEOMETRIC_INCREMENT);
		powermin = (int)logminbase_ours+1;

		if ((boolGlobalHistory) && (powermin == Historic_powermin[code]-1))
			powermin++;
		
		minimum = -pow(GRAPH_SCALE_GEOMETRIC_INCREMENT,powermin);
	} else {
		powermin = 0;
	};
	Historic_powermax[code] = powermax;
	Historic_powermin[code] = powermin;

	// Aim to set graphic y value = zeroplane y + factor * value:
	if (maximum > 1.0e40) maximum = 1.0e40;
	if (minimum > 1.0e40) minimum = 1.0e40;
	this->ymax = max(0.0,maximum);
	this->ymin = min(0.0,minimum);
	if (this->ymax > this->ymin) {
		this->yscale = (GRAPHIC_MAX_Y - GRAPHIC_MIN_Y)/(this->ymax - this->ymin);
		if (this->ymax > 0.0)
		{
			if (this->ymin < 0.0)
			{
				zeroplane = GRAPHIC_MIN_Y + 
					(GRAPHIC_MAX_Y - GRAPHIC_MIN_Y)*(-this->ymin/(this->ymax-this->ymin));
			} else {
				zeroplane = GRAPHIC_MIN_Y;
			}
		} else {
			zeroplane = GRAPHIC_MAX_Y;
		};
	} else {
		// all zero
		zeroplane = 0.0f;
		this->yscale = 1.0f;
	};

	// We set colourmax as the value we pass to shader -- 
	// and notice that sometimes we have DATA_HEIGHT but _VELOCITY_COLOUR.
	switch(colourflag)
	{
		case FLAG_VELOCITY_COLOUR:
	
		if (heightflag == FLAG_VELOCITY_HEIGHT) {
			// make max for colours match max for height:
				colourmax = maximum; 
			} else {
				colourmax = X.ReturnL4_Velocity(offset_vcolour,this->boolDisplayInnerMesh); // == 0 initially
			};
			if (_isnan(colourmax) || (!_finite(colourmax)))
			{
				printf("colourmax maximum %1.5E minimum %1.5E offset %d ",maximum, minimum, offset_vcolour);
				getch();
			};	
		break;
		case FLAG_SEGUE_COLOUR:
		case FLAG_IONISE_COLOUR:
			// For temperature want the following absolute scale;
			colourmax = 1.0;
			// use code to determine if this is temperature or what.
			break;
		case FLAG_AZSEGUE_COLOUR:
			
			colourmax = max(maximum, fabs(minimum)); // scale both + and - according to this.
			break;
		case FLAG_CURRENT_COLOUR:
			colourmax = X.ReturnL4_3DMagnitude(offset_vcolour,this->boolDisplayInnerMesh
				); // max modulus of J
			
		//	if (_isnan(colourmax) || (!_finite(colourmax)))	
			{
				printf("FLAG_CURRENT_COLOUR maximum %1.5E minimum %1.5E offset %d flag %d ",maximum, minimum, offset_vcolour, code);
		//	getch();
			};	
			break;
	};


	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	{
		if ( VertexBuffer[N] != NULL)
		{
			if (DXChk(VertexBuffer[N]->Lock(0,0,(void **)&vertices[N],0)) ||
				DXChk( IndexBuffer[N]->Lock(0,0, (void **)&indices[N],0)) )
					MessageBox(NULL,"oh dear","lock failed",MB_OK);
		};
	};
	
	if (this->boolDisplayMainMesh) // should be always on!
	{
		// At the moment this now just pours information into vertices[2], indices[2] :
		X.SetVerticesAndIndices(vertices, indices,       // better to do in the other class...
							numVertices, numTriangles, // pass it the integer counts so that it can test for overrun & redim
		                    numVerticesUsed, numTrianglesUsed,
							colourflag,heightflag, offset_data, offset_vcolour, zeroplane, yscale, this->boolDisplayInnerMesh);
	} else {
		numVerticesUsed[1] = 0;
		numTrianglesUsed[1] = 0;
	};
	
	if ((this->boolDisplayKeyButton) && (bScrewPinch == 0))
	{
		X.SetVerticesKeyButton(vertices[0],indices[0],colourmax,colourflag);
		numVerticesUsed[0] = X.GetNumKeyVerticesGraphics(&(numTrianglesUsed[0]));
	} else {
		numVerticesUsed[0] = 0;
		numTrianglesUsed[0] = 0;
	};
	
	//if (this->boolDisplayInnerMesh)
	//{
	//	X.SetVerticesAndIndicesInner(vertices[1],indices[1],
	//								numVertices[1],
	//								colourflag,heightflag,offset_data_slim,offset_vcolour_slim,
	//								zeroplane, yscale);
	//	numVerticesUsed[1] = X.numInnerVertices;
	//	numTrianglesUsed[1] = X.numInnerTriangles; 
	//} else {
	//	numVerticesUsed[1] = 0;
	//	numTrianglesUsed[1] = 0;
	//};


		//&numVerticesUsed1, &numIndicesUsed1,
		//&numVerticesUsed2, &numIndicesUsed2); // fill in these

	//for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	//	numTrianglesUsed[N] = numIndicesUsed[N]/3;

	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
		for (i = 0; i < numVerticesUsed[N]; i++)
			vertices[N][i].normal = -vertices[N][i].normal;

	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	{
		if ( VertexBuffer[N] != NULL)
		{
			VertexBuffer[N]->Unlock();
			IndexBuffer[N]->Unlock();
		};
	};
		
	return S_OK;
}

HRESULT surfacegraph::SetDataWithColourAux(TriMesh & X, int iLevel, int colourflag, 
										   int heightflag, int offset_data, int offset_vcolour,
										   int NTris // introduced for debugging
										   )
{
		
	DWORD * indices[NUMBER_VERTEX_ARRAYS];
	VertexPNT3 * vertices[NUMBER_VERTEX_ARRAYS];
	real maximum, minimum;
	int i,N;

	if (heightflag != FLAG_FLAT_MESH)
	{

		X.ReturnMaxMinDataAux(iLevel,offset_data, &maximum, &minimum);

		if (_isnan(maximum) || (!_finite(maximum)) || _isnan(minimum) || (!_finite(minimum)) )
		{
			printf("maximum %1.5E minimum %1.5E offset %d ",maximum, minimum, offset_data);
			getch();
		};	

		this->ymax = max(0.0,maximum);
		this->ymin = min(0.0,minimum);
		if (this->ymax > this->ymin) {
			this->yscale = (GRAPHIC_MAX_Y - GRAPHIC_MIN_Y)/(this->ymax - this->ymin);
			if (this->ymax > 0.0)
			{
				if (this->ymin < 0.0)
				{
					zeroplane = GRAPHIC_MIN_Y + 
						(GRAPHIC_MAX_Y - GRAPHIC_MIN_Y)*(-this->ymin/(this->ymax-this->ymin));
				} else {
					zeroplane = GRAPHIC_MIN_Y;
				}
			} else {
				zeroplane = GRAPHIC_MAX_Y;
			};
		} else {
			// all zero
			zeroplane = 0.0f;
			this->yscale = 1.0f;
		};
		this->colourmax = max(maximum, fabs(minimum)); 

	} else {
		//  plan view
		zeroplane = 0.0f;
		yscale = 1.0f;
		this->colourmax = 1.0; 
	};

	// DEBUG:
	// if (iLevel < 2) colourmax = 1.0;
	// purpose?

	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	{
		if ( VertexBuffer[N] != NULL)
		{
			if (DXChk(VertexBuffer[N]->Lock(0,0,(void **)&vertices[N],0)) ||
				DXChk(IndexBuffer[N]->Lock(0,0,(void **)&indices[N],0)) )
					MessageBox(NULL,"oh dear","lock failed",MB_OK);
		};
	};
	
	X.SetVerticesAndIndicesAux(iLevel, vertices[1], indices[1],
			numVertices[1], numTriangles[1], // pass it the dimmed counts so that it can test for memory overrun 
		    colourflag,heightflag, offset_data, offset_vcolour, 
			zeroplane, yscale,
			NTris); 
	numVerticesUsed[1] = X.numAuxVertices[iLevel];
	numTrianglesUsed[1] = X.numAuxTriangles[iLevel];
	//printf("numTrianglesUs1ed[1] %d \n",numTrianglesUsed[1] );

	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
		for (i = 0; i < numVerticesUsed[N]; i++)
			vertices[N][i].normal = -vertices[N][i].normal;

	for (N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	{
		if ( VertexBuffer[N] != NULL)
		{
			VertexBuffer[N]->Unlock();
			IndexBuffer[N]->Unlock();
		};
	};
		
	return S_OK;
}

VOID surfacegraph::RenderAux(const char * szTitle, int const iLabels, const TriMesh * pX, int iLevel)
{
	// Simplified version for auxiliary meshes.

	char buffer[128];
	D3DXMATRIXA16 matWorld;

	Direct3D.pd3dDevice->SetViewport(&vp); 

	mShadowMap->beginScene();
	Direct3D.pd3dDevice->Clear( 0, 0, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
		//0x00000000, 1.0f,0);			    
		D3DCOLOR_XRGB( 255, 255, 255 ), 1.0f, 0 );  

	DXChk(mFX->SetTechnique(mhBuildShadowMapTech));
	
	UINT numPasses = 1;
	DXChk(mFX->Begin(&numPasses, 0));	
	DXChk(mFX->SetMatrix(mhLightWVP, &(mLightVP)));
	DXChk(mFX->SetTexture(mhShadowMap, mShadowMap->d3dTex()));
	Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNT3::Decl);
	
	DXChk(mFX->CommitChanges());
	for (UINT ii = 0; ii < numPasses; ++ii)
	{
		DXChk(mFX->BeginPass(ii));
		int N = 1;
		{
			DXChk(Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[N], 0, sizeof(VertexPNT3)));
			DXChk(Direct3D.pd3dDevice->SetIndices(IndexBuffer[N]));
			DXChk(Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[N],
				0,
				numTrianglesUsed[N]));
		};
		
		DXChk(mFX->EndPass());
	};
	DXChk(mFX->End());
	mShadowMap->endScene();
	
	// ==================================================

	D3DXMatrixIdentity(&matWorld);
	Direct3D.pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );
	Direct3D.pd3dDevice->SetTransform( D3DTS_VIEW, &matView );
	Direct3D.pd3dDevice->SetTransform( D3DTS_PROJECTION, &matProj );
	Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , false);
	if ((GlobalCutaway) || (bCullNone)) {
		Direct3D.pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE );	
	} else {
		Direct3D.pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_CCW );	
	};

	if( SUCCEEDED( Direct3D.pd3dDevice->BeginScene() ) )
    {
		
		DXChk(Direct3D.pd3dDevice->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
						 D3DCOLOR_XRGB( 240, 255, 250 ), 1.0f, 0 ), 233);
		
		// The following parameters only do anything if the shader picks them up.

		DXChk(mFX->SetValue(mhLightVector, &mLight.dirW, sizeof(D3DXVECTOR3)));
		DXChk(mFX->SetValue(mhLightDiffuseColour, &mLight.diffuse, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhDiffuseMtrl, &mWhiteMtrl.diffuse, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhLightAmbientColour, &mLight.ambient, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhAmbientMtrl, &mWhiteMtrl.ambient, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhLightSpecularColour, &mLight.spec, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhSpecularPower, &mWhiteMtrl.specPower, sizeof(float)));
		DXChk(mFX->SetValue(mhSpecularMtrl, &mWhiteMtrl.spec, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhEyePos, &Eye, sizeof(D3DXVECTOR3)));
		DXChk(mFX->SetFloat(mhColourMax,colourmax));
		
		DXChk(mFX->SetBool(mhbTransparency, false));
		Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , false);	
		DXChk(mFX->SetMatrix(mhWVP, &(matView*matProj)));
		DXChk(mFX->SetMatrix(mhLightWVP, &(mLightVP)));
		DXChk(mFX->SetTexture(mhShadowMap, mShadowMap->d3dTex()));
		DXChk(mFX->SetTechnique(mhTech));
		
		DXChk(mFX->CommitChanges());
				
		Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNT3::Decl);

		mFX->Begin(&numPasses,0); // begins technique
		mFX->BeginPass(0);
		
		int N = 1;
		//for (int N = 0; N < 2; N++)
		{
			Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[N], 0, sizeof(VertexPNT3));		
			Direct3D.pd3dDevice->SetIndices(IndexBuffer[N]);
		    Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[N],
				0,
				numTrianglesUsed[N]);	
		};
		printf("verts rendered %d tris %d \n",numVerticesUsed[N],numTrianglesUsed[N]);

		mFX->EndPass();
		mFX->End();

		D3DXVECTOR3 norm(0.0,0.0,1.0);
		D3DXVECTOR3 norm2(0.0,0.0,1.0); // second tri becomes lit - weird

		if (boolDisplayMeshWireframe)
		{		
			if (boolClearZBufferBeforeWireframe) {
				Direct3D.pd3dDevice->Clear( 0, NULL, D3DCLEAR_ZBUFFER, (D3DCOLOR)0, 1.0f, 0 );
			};
			numPasses = 1;
			DXChk(mFX->SetTechnique(meshTech));
			DXChk(mFX->CommitChanges());
			mFX->Begin(&numPasses,0); // begins technique
			mFX->BeginPass(0);

			Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[1], 0, sizeof(VertexPNT3));
			Direct3D.pd3dDevice->SetIndices(IndexBuffer[1]);
			Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[1],
				0,
				numTrianglesUsed[1]);
			printf("verts rendered %d tris %d \n",numVerticesUsed[1],numTrianglesUsed[1]);

			mFX->EndPass();
			mFX->End();			
		};	

		if (iLabels == 1) {

			Vertex * pVertex = pX->AuxX[iLevel];
			for (long iVertex = 0; iVertex < pX->numAuxVertices[iLevel]; iVertex++)
			{
				sprintf(buffer,"%d",pVertex->iVolley);
				
				RenderLabel2(buffer,  // text
						pVertex->pos.x*xzscale,
						0.00001f+zeroplane,
						pVertex->pos.y*xzscale,0); 
				++pVertex;
			};
		};

		Direct3D.pd3dDevice->EndScene();
	};
}

VOID surfacegraph::Render(char * szTitle, bool RenderTriLabels, 
						  const TriMesh * pX, // = 0 by default
						  char * szLinebelow) // = 0 by default
{
	
	long tri_len, izTri[128];

	static DWORD time = timeGetTime();
	//DWORD oldtime;
	//float timestep, temporary;
	//int i;
	int iSwitch = 0;
	D3DXMATRIXA16 matWorld;
	//RECT rect;

	vertex1 linedata[10000];
	vertex1 linedata2[12];

		float x,y,z;
		char buffer[256];
		int i;
	real tempval;
	   
	//float values[11];
	//char buffer[2048];
	//D3DRECT dr;
	
	// DRAW SHADOW MAP:
	
	Direct3D.pd3dDevice->SetViewport(&vp); 
	// without this, 5 graphs, all except the last rendered, appear as small shadows on white

	mShadowMap->beginScene();

	
	Direct3D.pd3dDevice->Clear( 0, 0, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
		//0x00000000, 1.0f,0);			    
		D3DCOLOR_XRGB( 255, 255, 255 ), 1.0f, 0 );   
	// this clearly does affect shadow map successfully...         
		//0x00000000, 1.0f, 0 ); // Luna's code		// because *this* leads to everything being in shadow

	// at the moment, whatever value we put here, is staying the same.
	// in other words, even a shader with "return 0" is having no effect.

	DXChk(mFX->SetTechnique(mhBuildShadowMapTech));
	
	UINT numPasses = 1;
	DXChk(mFX->Begin(&numPasses, 0));
	
	DXChk(mFX->SetMatrix(mhLightWVP, &(mLightVP)));

	DXChk(mFX->SetTexture(mhShadowMap, mShadowMap->d3dTex()));
	
	//if (flag == FLAG_SEGUE)
	//{
	//	Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNf::Decl);
	//} else {
	//	if (flag == FLAG_VELOCITY)
	//	{
	Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNT3::Decl);
	//	} else {
	//		MessageBox(NULL,"A flag was not set.","bad news",MB_OK);
	//	};
	//};
	
	DXChk(mFX->CommitChanges());

	//printf ("Shadow\n");
	for (UINT ii = 0; ii < numPasses; ++ii)
	{
		DXChk(mFX->BeginPass(ii));
		
		// Key button etc = array 0 which casts no shadow.
		// Inner mesh = array 1
		// We do not always have it populated. Easy way for now: inner mesh casts no shadow.

	//	for (int N = 1; N <= NUMBER_VERTEX_ARRAYS; N++) 
		int N = 1;
		{
			DXChk(Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[N], 0, sizeof(VertexPNT3)));
			DXChk(Direct3D.pd3dDevice->SetIndices(IndexBuffer[N]));
			DXChk(Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[N],
				0,
				numTrianglesUsed[N]));
	//		printf("%d size %d -- ",N,numVerticesUsed[N]);
		};
		
		DXChk(mFX->EndPass());
	};
	
	DXChk(mFX->End());
	mShadowMap->endScene();
	
	//SAFE_RELEASE2(mCylinder);
	
	// for axis drawing in fixed function pipeline:
	//Direct3D.pd3dDevice->SetRenderTarget(0,0); // invalid call
	Direct3D.pd3dDevice->SetViewport(&vp);
	
	D3DXMatrixIdentity(&matWorld);
	Direct3D.pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );
	Direct3D.pd3dDevice->SetTransform( D3DTS_VIEW, &matView );
	Direct3D.pd3dDevice->SetTransform( D3DTS_PROJECTION, &matProj );
	
	Direct3D.pd3dDevice->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
						 D3DCOLOR_XRGB( 240, 255, 250 ), 1.0f, 0 );
		
	if ((GlobalCutaway) || (bCullNone)) {
		Direct3D.pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE );	
	} else {
		Direct3D.pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_CCW );	
	};
		
	if( SUCCEEDED( Direct3D.pd3dDevice->BeginScene() ) )
    {
		if (bScrewPinch == false) 
		{
			// drawing lines around base:
			
			x = (float)(-sin(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
			z = (float)(cos(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
			y = zeroplane;
			linedata[0].x = x; linedata[0].y = y; linedata[0].z = z;
			x = (float)(-sin(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
			z = (float)(cos(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
			linedata[1].x = x; linedata[1].y = y; linedata[1].z = z;
			for (i = 0; i < 12; i++)
				linedata[i].colour = 0xff000000;
			Direct3D.pd3dDevice->SetFVF(point_fvf);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));
			
			x = (float)(sin(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
			z = (float)(cos(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
			y = zeroplane;
			linedata[0].x = x; linedata[0].y = y; linedata[0].z = z;
			x = (float)(sin(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
			z = (float)(cos(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
			linedata[1].x = x; linedata[1].y = y; linedata[1].z = z;
			for (i = 0; i < 12; i++)
				linedata[i].colour = 0;
			Direct3D.pd3dDevice->SetFVF(point_fvf);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));
			
			real theta = -HALFANGLE;
			real r = 3.44;
			for (int asdf = 0; asdf < 10000; asdf++)
			{
				theta += FULLANGLE/10000.0; 
				linedata[asdf].x = -r*sin(theta)*xzscale;
				linedata[asdf].y = this->zeroplane;
				linedata[asdf].z = r*cos(theta)*xzscale;
				linedata[asdf].colour = 0;
			};
				
			Direct3D.pd3dDevice->SetFVF(point_fvf);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,9999,linedata,sizeof(vertex1));
						
			// Now do 3.6, 3.75, 3.9, 4.05, 4.2
			r = 3.45;
			for (i = 0; i < 5; i++) {
				theta = -HALFANGLE*1.1;
				r += 0.15;
				for (int asdf = 0; asdf < 10000; asdf++)
				{
					theta += FULLANGLE*1.1/10000.0; 
					linedata[asdf].x = r*sin(theta)*xzscale; // note: sin (negative) < 0
					linedata[asdf].z = r*cos(theta)*xzscale;
					
					if (theta < -0.5*HALFANGLE) {
						linedata[asdf].colour = 0xff000000;
					} else {
						linedata[asdf].colour = 0xff444444;
					};
				};					
				Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,9999,linedata,sizeof(vertex1));

				sprintf(buffer,"%1.2f",r);
				RenderLabel(buffer, linedata[5000].x,zeroplane,linedata[5000].z,true);
			};

			// Vertical lines:
			for (int iSide = 0; iSide <= 1; iSide++)
			for (int iWhich = 0; iWhich <= 1; iWhich++)
			{
				if (iWhich == 0) {
					x = (float)(-sin(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
					z = (float)(cos(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
				} else {
					x = (float)(-sin(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
					z = (float)(cos(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
				};
				
				if (iSide == 1) x = -x;

				linedata[0].x = x; linedata[0].z = z;
				linedata[1].x = x; linedata[1].z = z;
				linedata[0].colour = 0xff000000;
				linedata[1].colour = 0xff000000;
				
				linedata[0].y = GRAPHIC_MIN_Y-1.0f;  linedata[1].y = GRAPHIC_MAX_Y+2.5f;

				Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));
			};
		
		} else {
			// screw pinch: circle and semi-circle

			// I think we need a separate "pass" AND CLEAR z buffer
			// if we want stuff to appear on top.

			real theta = 0.0;
			real r = pX->OuterRadiusAttained;
			for (int asdf = 0; asdf < 10000; asdf++)
			{
				theta += 2.0*PI/10000.0; 
				linedata[asdf].x = -r*sin(theta)*xzscale;//(-TRAP_HALFWIDTH+DELTA_0*((real)asdf))*block::xzscale;
				linedata[asdf].y = this->zeroplane;
				linedata[asdf].z = (r*cos(theta)+SP_CENTRE_Y)*xzscale;
				linedata[asdf].colour = 0;
			};
			Direct3D.pd3dDevice->SetFVF(point_fvf);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,9999,linedata,sizeof(vertex1));

			for (int asdf = 0; asdf < 10000; asdf++)
			{
				theta += PI/10000.0; 
				linedata[asdf].x = -r*cos(theta)*xzscale;
				linedata[asdf].y = GRAPHIC_MAX_Y;
				linedata[asdf].z = (r*sin(theta)+SP_CENTRE_Y)*xzscale;
				linedata[asdf].colour = 0;
			};
			Direct3D.pd3dDevice->SetFVF(point_fvf);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,9999,linedata,sizeof(vertex1));

				// Vertical lines:
			for (int iSide = 0; iSide <= 1; iSide++)
			for (int iWhich = 0; iWhich <= 1; iWhich++)
			{
				
					// only really want to appear in 2 places ??
				x = (float)(-sin(PI/5.0)*pX->OuterRadiusAttained)*xzscale;
				z = (float)(cos(PI/5.0)*pX->OuterRadiusAttained+SP_CENTRE_Y)*xzscale; // just let it draw twice over
				
				if (iSide == 1) x = -x;

				linedata[0].x = x; linedata[0].z = z;
				linedata[1].x = x; linedata[1].z = z;
				linedata[0].colour = 0xff000000;
				linedata[1].colour = 0xff000000;
				
				linedata[0].y = GRAPHIC_MIN_Y-1.0f;  linedata[1].y = GRAPHIC_MAX_Y+2.5f;

				Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));
			};
		};
 	
		// The following parameters only do anything if the shader picks them up.

		DXChk(mFX->SetValue(mhLightVector, &mLight.dirW, sizeof(D3DXVECTOR3)));
		DXChk(mFX->SetValue(mhLightDiffuseColour, &mLight.diffuse, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhDiffuseMtrl, &mWhiteMtrl.diffuse, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhLightAmbientColour, &mLight.ambient, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhAmbientMtrl, &mWhiteMtrl.ambient, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhLightSpecularColour, &mLight.spec, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhSpecularPower, &mWhiteMtrl.specPower, sizeof(float)));
		DXChk(mFX->SetValue(mhSpecularMtrl, &mWhiteMtrl.spec, sizeof(D3DXCOLOR)));
		DXChk(mFX->SetValue(mhEyePos, &Eye, sizeof(D3DXVECTOR3)));
		DXChk(mFX->SetFloat(mhColourMax,colourmax));
		
		DXChk(mFX->SetBool(mhbTransparency, false));
		Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , false);	
		
		mFX->SetTechnique(mhTech);
		mFX->SetMatrix(mhWVP, &(matView*matProj));
		
		//mFX->SetTexture(mhTex, texture_grid);      
		
		// shadow map:
		DXChk(mFX->SetMatrix(mhLightWVP, &(mLightVP)));
		DXChk(mFX->SetTexture(mhShadowMap, mShadowMap->d3dTex()));
		
		//if (flag == FLAG_SEGUE)
		//{
		//	Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNf::Decl);
		Direct3D.pd3dDevice->SetVertexDeclaration(VertexPNT3::Decl);
		
		// Looks like we need to be careful how to apply different shaders.
		// Can we do it during a {Begin ... End} block ?
		// We can do it even during BeginPass if we do CommitChanges, supposedly.
		
		DXChk(mFX->SetBool(mhbTransparency, false)); // telling shader to give everything alpha = 1 until further notice.
		
		// Cycle the transparency every 2 ns:
		real timeover = evaltime;
		while (timeover > 2.0e-9) timeover -= 2.0e-9;
		timeover = fabs(timeover-1.0e-9)/1.0e-9; 		
		DXChk(mFX->SetFloat(mhfTransparentAlpha, 0.2f));
			//0.08f + 0.18f*(float)timeover));
		DXChk(mFX->CommitChanges());
		Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , true);	
		Direct3D.pd3dDevice->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
		Direct3D.pd3dDevice->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA); // try D3DBLEND_ONE
		
		DXChk(mFX->CommitChanges());
							
		numPasses = 1;
		mFX->Begin(&numPasses,0); // begins technique
		mFX->BeginPass(0);
		
		int N;
		for (int N = 0; N < 2; N++)
		{
			Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[N], 0, sizeof(VertexPNT3));		
			Direct3D.pd3dDevice->SetIndices(IndexBuffer[N]);
		    Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[N],
				0,
				numTrianglesUsed[N]);	
		};
		

		if (GlobalCutaway) {
			DXChk(mFX->SetBool(mhbTransparency, true)); // render the rest with low alpha
			DXChk(mFX->CommitChanges());

			Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[1], 0, sizeof(VertexPNT3));		
			Direct3D.pd3dDevice->SetIndices(IndexBuffer[2]);
			Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[1],
				0,
				numTrianglesUsed[2]);
		// Maybe though, this method does cause some ruckus where there are no triangles placed.
		// Try creating separate vertex array...
		};

		mFX->EndPass();

		//DXChk(mFX->SetBool(mhbTransparency, true));
		//DXChk(mFX->SetFloat(mhfTransparentAlpha, 0.08f));
		//DXChk(mFX->CommitChanges());
		

		//Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , true);	
		//Direct3D.pd3dDevice->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
		//Direct3D.pd3dDevice->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA); // try D3DBLEND_ONE
		
		// If we do it within Begin(,) ... End()
		// then the render state doesn't take.
		// If we do it outside, then what is there from the 1st pass gets overwritten for some reason.

		// The reason we can't do it a sensible way with 1 pair of buffers and 2 passes in shader is that it sees z-fighting
		// of a copy with alpha = 0(I think) of the rest of the surface -- which I cannot understand.

		//numPasses = 1;
		//mFX->BeginPass(1);

		//Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[1], 0, sizeof(VertexPNT3));		
		//Direct3D.pd3dDevice->SetIndices(IndexBuffer[2]);
		//Direct3D.pd3dDevice->DrawIndexedPrimitive(
		//		D3DPT_TRIANGLELIST,
		//		0,
		//		0,
		//		numVerticesUsed[1],
		//		0,
		//		numTrianglesUsed[2]);

		//mFX->EndPass();
		
		mFX->End();

		DXChk(mFX->SetBool(mhbTransparency, false));
		DXChk(mFX->CommitChanges());
		Direct3D.pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , false);	

		D3DXVECTOR3 norm(0.0,0.0,1.0);
		D3DXVECTOR3 norm2(0.0,0.0,1.0); // second tri becomes lit - weird

		// Now try to draw the scales:
		if (this->boolDisplayScales) {

			for (int iSide = 0; iSide <= 1; iSide++)
			for (int iWhich = 0; iWhich <= 1; iWhich++)
			{
				if (bScrewPinch == false) {
					if (iWhich == 0) {
						x = (float)(-sin(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
						z = (float)(cos(HALFANGLE)*DEVICE_RADIUS_INSULATOR_OUTER)*xzscale;
					} else {
						x = (float)(-sin(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
						z = (float)(cos(HALFANGLE)*DOMAIN_OUTER_RADIUS)*xzscale;
					};
				} else {
					// only really want to appear in 2 places ??
					x = (float)(-sin(PI/5.0)*pX->OuterRadiusAttained)*xzscale;
					z = (float)(cos(PI/5.0)*pX->OuterRadiusAttained+SP_CENTRE_Y)*xzscale; // just let it draw twice over
				}
				if (iSide == 1) x = -x;

				// New way: 
				real const lead[12] = {0.5,0.675,0.9,1.2,1.6,2.14,2.85,3.8,5.0,6.75,9.0,12.0 };
				int log_base_10, leadindex;
				real useval, leadval, leadval2, leadvalneg, leadval2neg, temp, value[5];

				real scalemax, scalemin;
				if (label_insist_max) {
					scalemax = store_max;
					scalemin = store_min;
				} else {
					scalemax = (real)ymax;
					scalemin = (real)ymin;
				}

				if (scalemax > 0.0) {
					log_base_10 = (int)(log10(scalemax));
					// careful: what if negative?
					if (scalemax < 1.0) log_base_10--; 
					// if 0.8 then we want that to say 8.00e-1 
					// keep all markers in same e-1 however.
					useval = scalemax/pow(10.0,log_base_10); // maybe to create 10^b would be often faster.
					leadindex = 2; while (useval*0.9 > lead[leadindex+1]) leadindex++;
					leadval = lead[leadindex]*pow(10.0,log_base_10);
					leadval2 = lead[leadindex-1]*pow(10.0,log_base_10);
				};
				if (scalemin < 0.0) {
					temp = -scalemin;
					log_base_10 = (int)(log10(temp));
					if (temp < 1.0) log_base_10--;
					useval = temp/pow(10.0,log_base_10);
					leadindex = 2; while (useval*0.9 > lead[leadindex+1]) leadindex++;
					leadvalneg = -lead[leadindex]*pow(10.0,log_base_10);
					leadval2neg = -lead[leadindex-1]*pow(10.0,log_base_10);
				};

				if (scalemin >= 0.0) {
					if (scalemax > 0.0) {
						value[0] = scalemax;
						value[1] = leadval;
						value[2] = leadval2;
						value[3] = 0.5*value[2];
						value[4] = 0.0;
					} else {
						// both == 0
						value[0] = 0.0; value[1] = 0.0; value[2] = 0.0; 
						value[3] = 0.0; value[4] = 0.0;
					};
				} else {
					if (scalemax > 0.0) {
						if (fabs(scalemin) > 2.0*scalemax) {
							value[0] = scalemax;
							value[1] = 0.0;
							value[2] = 0.5*leadval2neg;
							value[3] = leadval2neg;
							value[4] = scalemin;
						} else {
							if (fabs(scalemin) < 0.5*scalemax) {
								value[0] = scalemax;
								value[1] = leadval2;
								value[2] = 0.5*leadval2;
								value[3] = 0.0;
								value[4] = scalemin;
							} else {
								value[0] = scalemax;
								value[1] = leadval2;
								value[2] = 0.0;
								value[3] = leadval2neg;
								value[4] = scalemin;
							};
						};
					} else {
						value[0] = 0.0;
						value[1] = 0.5*leadval2neg;
						value[2] = leadval2neg;
						value[3] = leadvalneg;
						value[4] = scalemin;
					};
				};
				
				for (int i = 0; i < 5; i++)
				{
					sprintf(buffer,"%1.2E",value[i]*this->TickRescaling);
					RenderLabel(buffer, x, zeroplane + yscale*value[i], z); // 3D position for top-right of text
					
				printf("szTitle = %s ",szTitle);
				printf("buffer %s x %f y %f z %f\n",
					buffer,x,zeroplane+yscale*value[i],z);
				};
				
			};
		};


		// Mesh wireframe:

		if (boolDisplayMeshWireframe)
		{
			if (boolClearZBufferBeforeWireframe) {
				Direct3D.pd3dDevice->Clear( 0, NULL, 
					D3DCLEAR_ZBUFFER, (D3DCOLOR)0, 1.0f, 0 );
			};

			numPasses = 1;
			DXChk(mFX->SetTechnique(meshTech));
			DXChk(mFX->CommitChanges());

			mFX->Begin(&numPasses,0); // begins technique
//			 The default value 0 specifies that ID3DXEffect::Begin and ID3DXEffect::End will save and restore all state modified by the effect (including pixel and vertex shader constants). Valid flags can be seen at Effect State Save and Restore Flags.

			mFX->BeginPass(0);

			//if (boolDisplayInnerMesh)
			//{
			//	Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[1], 0, sizeof(VertexPNT3));
			//	Direct3D.pd3dDevice->SetIndices(IndexBuffer[1]);
			//	Direct3D.pd3dDevice->DrawIndexedPrimitive(
			//		D3DPT_TRIANGLELIST,
			//		0,
			//		0,
			//		numVerticesUsed[1],
			//		0,
			//		numTrianglesUsed[1]);
			//};

			Direct3D.pd3dDevice->SetStreamSource(0, VertexBuffer[1], 0, sizeof(VertexPNT3));
			Direct3D.pd3dDevice->SetIndices(IndexBuffer[1]);
			Direct3D.pd3dDevice->DrawIndexedPrimitive(
				D3DPT_TRIANGLELIST,
				0,
				0,
				numVerticesUsed[1],
				0,
				numTrianglesUsed[1]);

			mFX->EndPass();
			mFX->End();
		};		
		
		Direct3D.pd3dDevice->EndScene();
		
		if( SUCCEEDED( Direct3D.pd3dDevice->BeginScene() ) )
		{ // seems to make no difference
			
			// This makes the difference:
			Direct3D.pd3dDevice->Clear( 0, NULL,D3DCLEAR_ZBUFFER,
						 (D3DCOLOR)0, 1.0f, 0 );
			// prevents z-fighting: overwrite graphics from here

			// draw black line graph at cutaway:
			if (GlobalCutaway) {

				real * radiusArray8000;
				long * VertexIndexArray8000;
			//	// Render some data height labels along the line of the cutaway.

				VertexPNT3 * pPNT;
				VertexPNT3 * vertices_buffer;
				VertexIndexArray8000 = new long[8000];
				radiusArray8000 = new real[8000];

				if (radiusArray8000 == 0) {
					printf("\n\n?@#>@>#?\n\n");
					getch();
				};		
		
				long numVertsCutawayUse = pX->GetVertsRightOfCutawayLine_Sorted(VertexIndexArray8000, radiusArray8000);

				// Render a line along the cutaway? Ambitious....
				// quite a dirty way for now: exploit that there is only 1 array of vertex positions in graphic space

				// In general, better to use TriMesh object
				// and create a function to return graphic positions.
				
				long diff = pX->Xdomain-pX->X;
				if (this->boolDisplayInnerMesh) diff = 0 ;

				for (int iSubpass = 1; iSubpass < 2; iSubpass++ )
				{
					if (iSubpass == 0) {
						DXChk(VertexBuffer[1]->Lock(0,0,(void **)&vertices_buffer,D3DLOCK_READONLY));
					} else {
						DXChk(VertexBuffer[1]->Lock(0,0,(void **)&vertices_buffer,0));

						bool has_more, has_less, has_grad;
						Vertex * pVert2, *pVertex;
						int iWhich, iCorner;
						D3DXVECTOR3 newpos;
						float wt0, wt1, wt2, wttotal, dist0, dist1, dist2;
						VertexPNT3 * pPNT0, * pPNT1, * pPNT2;
						Triangle * pTri;
						
						// add here code to shift stuff about .....
						// interpolate using pos.y but having found the relevant index
						// from the TriMesh object

						for (int asdf = 0; asdf < numVertsCutawayUse; asdf++)
						{
							if (VertexIndexArray8000[asdf]-diff >= 0)
							{
								pVertex = pX->X + VertexIndexArray8000[asdf];

								// We want the tri directly to the left of it, through which (-1,0) passes.
								// 1.Get these vertex indices
								// which tri contains a point which is further and a point less far?

								real rr = pVertex->pos.x*pVertex->pos.x+pVertex->pos.y*pVertex->pos.y;

								iWhich = -1;
								tri_len = pVertex->GetTriIndexArray(izTri);
								for (i = 0; i < tri_len; i++)
								{
									pTri = pX->T+izTri[i];
									has_more = false; has_less = false; has_grad = false;
									for (iCorner =0 ; iCorner < 3; iCorner++)
									{
										pVert2 = pTri->cornerptr[iCorner];
										if (pVert2 != pVertex) 
										{
											if (pVert2->pos.x*pVert2->pos.x+pVert2->pos.y*pVert2->pos.y > rr)
											{
												has_more = true;
											} else {
												has_less = true;
											};
										};
										if (pVert2->pos.x/pVert2->pos.y < pVertex->pos.x/pVertex->pos.y)
											has_grad = true;
									};
									
									if (has_more && has_less && has_grad)
									{
										iWhich = i;										
									}									
								};
									
								if (iWhich == -1) {// give up, do nothing} 
								} else {
									pTri = pX->T + izTri[iWhich];
									if (this->boolDisplayInnerMesh == false)
									{
										while (pTri->u8domain_flag != DOMAIN_TRIANGLE) {
											iWhich--;
											if (iWhich == -1) {
												printf("give up!");
												getch();
												return;
											};
											pTri = pX->T + izTri[iWhich];
										}
									};
									// 2. shift this pos.xyz to be on the line and y-interpolated.
									
									// is our origin also the pos origin?
									pPNT = &(vertices_buffer[VertexIndexArray8000[asdf]-diff]);
									
									newpos.z = pPNT->pos.z;
									newpos.x = pPNT->pos.z*(float)(-GRADIENT_X_PER_Y)*0.5f;
									
									pPNT0 = &(vertices_buffer[(pTri->cornerptr[0]-pX->X)-diff]);
									pPNT1 = &(vertices_buffer[(pTri->cornerptr[1]-pX->X)-diff]);
									pPNT2 = &(vertices_buffer[(pTri->cornerptr[2]-pX->X)-diff]);
									
									// Possible it picks up a point that is not within ?
									if (((pTri->cornerptr[2]-pX->X)-diff < 0)
										||
										((pTri->cornerptr[1]-pX->X)-diff < 0)
										||
										((pTri->cornerptr[0]-pX->X)-diff < 0))
									{
										iWhich = iWhich;
									}

									dist0 = sqrt((pPNT0->pos.x-newpos.x)*(pPNT0->pos.x-newpos.x)
												+ (pPNT0->pos.z-newpos.z)*(pPNT0->pos.z-newpos.z));
									dist1 = sqrt((pPNT1->pos.x-newpos.x)*(pPNT1->pos.x-newpos.x)
												+ (pPNT1->pos.z-newpos.z)*(pPNT1->pos.z-newpos.z));
									dist2 = sqrt((pPNT2->pos.x-newpos.x)*(pPNT2->pos.x-newpos.x)
												+ (pPNT2->pos.z-newpos.z)*(pPNT2->pos.z-newpos.z));
									
									wt0 = 1.0f/dist0;
									wt1 = 1.0f/dist1;
									wt2 = 1.0f/dist2;
									wttotal = wt0+wt1+wt2;
									wt0 /= wttotal;
									wt1 /= wttotal;
									wt2 /= wttotal;
									newpos.y = wt0*pPNT0->pos.y + wt1*pPNT1->pos.y + wt2*pPNT2->pos.y;
									pPNT->pos = newpos;
																		
								};
							// We could even try a cheat: do that before we do the surface graph. Does it stretch across?
							};
						}; // asdf
					}; // if iSubpass == 0
			
					long diff = pX->Xdomain-pX->X;
					if (this->boolDisplayInnerMesh) {

						for (int asdf = 0; asdf < numVertsCutawayUse; asdf++)
						{
							pPNT = &(vertices_buffer[VertexIndexArray8000[asdf]]);
							linedata[asdf].x = pPNT->pos.x;
							linedata[asdf].y = pPNT->pos.y; // will this help make it show up?
							linedata[asdf].z = pPNT->pos.z;
							linedata[asdf].colour = 0;
						};
					} else {
						long unused = -1;
						for (int asdf = 0; asdf < numVertsCutawayUse; asdf++)
						{
							if (VertexIndexArray8000[asdf]-diff >= 0)
							{
								pPNT = &(vertices_buffer[VertexIndexArray8000[asdf]-diff]); 
								// relative to Xdomain here
								linedata[asdf].x = pPNT->pos.x;
								linedata[asdf].y = pPNT->pos.y; // will this help make it show up?
								linedata[asdf].z = pPNT->pos.z;
								linedata[asdf].colour = 0;
							} else {
								// inner cutaway vertex but inner are not displayed.
								unused = asdf;
							};
						};
						for (int asdf = 0; asdf <= unused; asdf++)
						{
							linedata[asdf] = linedata[unused+1];
						};
					};
					
					Direct3D.pd3dDevice->SetFVF(point_fvf);
					Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,numVertsCutawayUse-1,linedata,sizeof(vertex1));

					int asdf = 0;
			
					real r = 3.439999999;
					for (i = 0; i < 6; i++) {
						while (radiusArray8000[asdf] < r) asdf++;	

						if (this->boolDisplayInnerMesh)
						{
							pPNT = &(vertices_buffer[VertexIndexArray8000[asdf]]);
						} else {
							pPNT = &(vertices_buffer[VertexIndexArray8000[asdf]-diff]);
						};

						x = pPNT->pos.x;
						y = zeroplane;
						z = pPNT->pos.z;
						linedata[0].x = x; linedata[0].y = y; linedata[0].z = z;
						y = pPNT->pos.y;
						linedata[1].x = x; linedata[1].y = y; linedata[1].z = z;
						Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));

						tempval = (pPNT->pos.y - zeroplane)/yscale;
						sprintf(buffer,"%1.2E",tempval);
						RenderLabel(buffer, -0.5*GRADIENT_X_PER_Y*pPNT->pos.z,zeroplane,pPNT->pos.z);
						if (i == 0) r = 3.45;
						r += 0.15;
					};
					// line underneath:
					linedata[0].x = -sin(HALFANGLE*0.5)*DEVICE_RADIUS_INSULATOR_OUTER*xzscale;
					linedata[0].y = zeroplane;
					linedata[0].z = cos(HALFANGLE*0.5)*DEVICE_RADIUS_INSULATOR_OUTER*xzscale;
					linedata[1].x = -sin(HALFANGLE*0.5)*DOMAIN_OUTER_RADIUS*xzscale;
					linedata[1].y = zeroplane;
					linedata[1].z = cos(HALFANGLE*0.5)*DOMAIN_OUTER_RADIUS*xzscale;
					Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,1,linedata,sizeof(vertex1));

					// Not sure we had to lock to read anyway?
					VertexBuffer[1]->Unlock();
				}; // iSubpass
			//	
			//	// label: seek for where r > 3.61

			//	i = 0; 
			//	while (radiusArray8000[i] < 3.61) i++;

			//	pPNT = VertexBuffer[2][VertexIndexArray8000[i]].pPNT;
			//	sprintf(buffer,"%1.4E",pPNT.y*some_kind_of_y_scale_back);
			//	RenderLabel(buffer, pPNT->x,pPNT->y,pPNT->z);
			//
				delete[] VertexIndexArray8000;
				delete[] radiusArray8000;
			};

			// Now try to draw around the edge of the viewport:

			// x and y in screen coordinates, z coordinate ignored...

			linedata2[0].x = vp.X;			  linedata2[0].y = vp.Y+vp.Height-1;
			linedata2[1].x = vp.X+vp.Width-1; linedata2[1].y = vp.Y+vp.Height-1;
			linedata2[2].x = vp.X+vp.Width-1; linedata2[2].y = vp.Y;
			linedata2[3].x = vp.X;			  linedata2[3].y = vp.Y;
			linedata2[4].x = vp.X;			  linedata2[4].y = vp.Y+vp.Height-1;
			
			linedata2[0].z = 0; linedata2[1].z = 0; linedata2[2].z = 0;
			linedata2[3].z = 0; linedata2[4].z = 0;

			linedata2[0].colour = 0;
			linedata2[1].colour = 0;
			linedata2[2].colour = 0;
			linedata2[3].colour = 0;
			linedata2[4].colour = 0;

			Direct3D.pd3dDevice->SetFVF(D3DFVF_XYZRHW | D3DFVF_DIFFUSE);
			Direct3D.pd3dDevice->DrawPrimitiveUP(D3DPT_LINESTRIP,4,linedata2,sizeof(vertex1));

			RenderText(szTitle,0);
			if (szLinebelow != 0) RenderText(szLinebelow,1);

			if (RenderTriLabels)
			{
				// Let's render some on vertices instead.

				Vertex * pVertex = pX->X + 11800;
			//	for (long iVertex = 0; iVertex < pX->numVertices; iVertex++)
				for (long iVertex = 11800; iVertex < 14500; iVertex++)
				{
					if (GlobalWhichLabels == 0) {
						sprintf(buffer,"%d",iVertex);
						
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
					}
					if (GlobalWhichLabels == 1) {
						
						sprintf(buffer,"%1.2E",pVertex->n);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
					};
					if (GlobalWhichLabels == 2) {
						sprintf(buffer,"%d",iVertex);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
						sprintf(buffer,"%1.1E",pVertex->T);
						strip_0(buffer);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,1); 
					};
					
					if (GlobalWhichLabels == 3) {
						sprintf(buffer,"%1.1E",pVertex->v.x);
						strip_0(buffer);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
						sprintf(buffer,"%1.1E",pVertex->v.y);
						strip_0(buffer);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,1); 
					};
					
					if (GlobalWhichLabels == 4) {
						sprintf(buffer,"%d",iVertex);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
						sprintf(buffer,"%1.1E",pVertex->phi);
						strip_0(buffer);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,1); 
					};
					
					if (GlobalWhichLabels == 5) {
						sprintf(buffer,"%d",iVertex);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,0); 
						sprintf(buffer,"%1.1E",pVertex->Temp.x);
						strip_0(buffer);
						RenderLabel2(buffer,  // text
							pVertex->pos.x*xzscale,
							0.00001f+zeroplane,
							pVertex->pos.y*xzscale,1); 
					};
					// How often are these n,T,v going to be maintained?
					
					++pVertex;
				};
			};
			Direct3D.pd3dDevice->EndScene();
		} // if SUCCEEDED BeginScene
		else {
			printf("BeginScene (2) failed!\n\n");
			getch();
		}
	} else { // if SUCCEEDED BeginScene
		printf("BeginScene (1) failed !!\n\n");
		getch();
	};
	
}


void inline surfacegraph::RenderText (char * text, int lines_down)
{
	RECT rect;
	rect.top = vp.Y +20+30*lines_down;
	rect.right = vp.X + vp.Width-15;

	rect.bottom=rect.top+30;
	rect.left = rect.right-300;
	
	rect.top-=2;
	rect.right+=2;
	rect.left+=2;
	rect.bottom-=2;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xff000000);

	rect.top+=2;
	rect.right-=2;
	rect.left-=2;
	rect.bottom+=2;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xff99ffff);

	rect.top-=4;
	rect.bottom-=4;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xff99ffff);

	rect.right+=4;
	rect.left+=4;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xff99ffff);
	
	rect.top+=4;
	rect.bottom+=4;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xffffffff);

	rect.top-=2;
	rect.right-=2;
	rect.left-=2;
	rect.bottom-=2;

	Direct3D.g_pFont2->DrawText(NULL,text,strlen(text),&rect,DT_RIGHT|DT_VCENTER,0xff000000);
	// Cyan came out on top. Supposed to do what about it?
}

void inline surfacegraph::RenderLabel (char * text, float x, float y, float z, 
									   bool extrainfo)
	{
		RECT rect;
		D3DXVECTOR3 transformed;
		// The following was static const. That caused badness! vp needs to change.
		D3DXMATRIXA16 screenmat(((float)vp.Width)*0.5f,0.0f,0.0f,0.0f,
			                    0.0f,  -((float)vp.Height)*0.5f,0.0f,0.0f,
								0.0f,0.0f,((float)vp.MaxZ)-((float)vp.MinZ),0.0f,
							    ((float)vp.X)+((float)vp.Width)*0.5f,((float)vp.Y)+((float)vp.Height)*0.5f,((float)vp.MinZ),1.0f);

		D3DXVECTOR3 position(x,y,z);
		D3DXVECTOR3 screencoord;
		D3DXVec3TransformCoord(&transformed, &position,&(matView*matProj));

		D3DXVec3TransformCoord(&screencoord, &transformed, &screenmat);

		DWORD format = DT_TOP|DT_RIGHT;
		D3DCOLOR textcolor = 0xff000000;
		if (extrainfo) {
			format = DT_CENTER | DT_VCENTER; // also changing rect, below.
			textcolor = 0xff700022;
		}
		// see http://msdn.microsoft.com/en-us/library/windows/desktop/bb206341(v=vs.85).aspx


		rect.top = (int)screencoord.y;
		rect.right = (int)screencoord.x;
		
		rect.bottom=rect.top+30;
		rect.left=rect.right-200;

		if (extrainfo) {
			rect.bottom -= 15;
			rect.top -= 15;
			rect.left += 100;
			rect.right += 100;
		};


		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,textcolor);
		
		rect.top += 1;
		rect.bottom += 1;
		rect.left += 1;
		rect.right += 1;

		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);

		rect.top -= 2;
		rect.bottom -= 2;

		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);
		
		rect.left -= 2;
		rect.right -= 2;

		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);

		rect.top += 2;
		rect.bottom += 2;

		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);

		rect.top -= 1;
		rect.bottom -= 1;
		rect.left += 1;
		rect.right += 1;
		

		// even more white:
		rect.left -= 2;
		rect.right -=2;
		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);
		rect.left += 4;
		rect.right += 4;
		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,0xffffffff);
		rect.left -= 2;
		rect.right -=2;

		Direct3D.g_pFont->DrawText(NULL,text,strlen(text),&rect,format,textcolor);
		
	}

void inline surfacegraph::RenderLabel2 (char * text, float x, float y, float z, int whichline)
	{
		RECT rect;
		D3DXVECTOR3 position(x,y,z);
		D3DXVECTOR3 transformed;

		// see http://msdn.microsoft.com/en-us/library/windows/desktop/bb206341(v=vs.85).aspx

		D3DXMATRIXA16 screenmat(((float)vp.Width)*0.5f,0.0f,0.0f,0.0f,
			                    0.0f,  -((float)vp.Height)*0.5f,0.0f,0.0f,
								0.0f,0.0f,((float)vp.MaxZ)-((float)vp.MinZ),0.0f,
							    ((float)vp.X)+((float)vp.Width)*0.5f,((float)vp.Y)+((float)vp.Height)*0.5f,((float)vp.MinZ),1.0f);

		D3DXVECTOR3 screencoord;

		D3DXVec3TransformCoord(&transformed, &position,&(matView*matProj));
		D3DXVec3TransformCoord(&screencoord, &transformed, &screenmat);

		rect.top = (int)screencoord.y-15;
		rect.right = (int)screencoord.x+100;
		
		if ((screencoord.x > vp.X ) && (screencoord.x < vp.X+vp.Width)) {

		if (whichline == 0) {
			rect.top -= 8;
		} else {
			rect.top += 7;
		}

		rect.bottom=rect.top+30;
		rect.left=rect.right-200;

		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xff000000);
		
		rect.top += 1;
		rect.bottom += 1;
		rect.left += 1;
		rect.right += 1;

		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffffff);

		rect.top -= 2;
		rect.bottom -= 2;
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffffff);

		rect.left -= 2;
		rect.right -= 2;
		
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffffff);

		rect.top += 2;
		rect.bottom += 2;
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffffff);

		rect.top -= 3;
		rect.bottom -= 3;
		rect.left -= 1;
		rect.right -= 1;
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffff55);
		rect.left += 4;
		rect.right += 4;
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xffffff55);
		
		rect.top += 2;
		rect.bottom += 2;
		rect.left -= 2;
		rect.right -= 2;
		Direct3D.g_pFontsmall->DrawText(NULL,text,strlen(text),&rect,DT_CENTER|DT_VCENTER,0xff000000);
		// try rendering black twice and hope it ends up on top.

		}
	}
surfacegraph::~surfacegraph()
{
	for(int N = 0; N < NUMBER_VERTEX_ARRAYS; N++)
	{
		if ( VertexBuffer[N] != NULL)
			VertexBuffer[N]->Release();
		if ( IndexBuffer[N] != NULL)
			IndexBuffer[N]->Release();
	};
	
	
	delete mShadowMap;
}

#endif