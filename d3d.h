
#ifndef d3d_h
#define d3d_h

#pragma warning( disable : 4996 ) // disable deprecated warning 
#pragma warning( disable : 4995 ) // disable deprecated warning 

#define D3DXFX_LARGEADDRESS_HANDLE
//http://msdn.microsoft.com/en-gb/library/windows/desktop/bb172855(v=vs.85).aspx
//Define this value before including d3dx9.h so that your application fails to compile when attempting to pass strings into D3DXHANDLE parameters. This will aid in making sure that valid information is being passed to the runtime.
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <crtdbg.h>
#include <initguid.h>
#include <Windows.h>
#include <mmsystem.h>
#include <d3dx9.h>
#include <stdlib.h>
//#include <iostream>
//#include <vector>
#include <strsafe.h>
//#include <dinput.h>
#include <dxerr.h>
#include <commdlg.h>

#define SAFE_DELETE_ARRAY(x)  if (x != NULL) delete[] x;
#define SAFE_DELETE(x)        if (x != NULL) delete x;
#define SAFE_RELEASE(x)       if (x != NULL) x.Release();
#define SAFE_RELEASE2(x)       if (x != NULL) x->Release();



//===============================================================
struct VertexPos
{
	VertexPos():pos(0.0f, 0.0f, 0.0f){}
	VertexPos(float x, float y, float z):pos(x,y,z){}
	VertexPos(const D3DXVECTOR3& v):pos(v){}

	D3DXVECTOR3 pos;
	static IDirect3DVertexDeclaration9* Decl;
};

//===============================================================
struct VertexCol
{
	VertexCol():pos(0.0f, 0.0f, 0.0f),col(0x00000000){}
	VertexCol(float x, float y, float z, D3DCOLOR c):pos(x,y,z), col(c){}
	VertexCol(const D3DXVECTOR3& v, D3DCOLOR c):pos(v),col(c){}

	D3DXVECTOR3 pos;
	D3DCOLOR    col;
	static IDirect3DVertexDeclaration9* Decl;
};

//===============================================================
struct VertexPN
{
	VertexPN()
		:pos(0.0f, 0.0f, 0.0f),
		normal(0.0f, 0.0f, 0.0f){}
	VertexPN(float x, float y, float z, 
		float nx, float ny, float nz):pos(x,y,z), normal(nx,ny,nz){}
	VertexPN(const D3DXVECTOR3& v, const D3DXVECTOR3& n)
		:pos(v),normal(n){}

	D3DXVECTOR3 pos;
	D3DXVECTOR3 normal;
	static IDirect3DVertexDeclaration9* Decl;
};

//===============================================================
struct VertexPNT
{
	VertexPNT()
		:pos(0.0f, 0.0f, 0.0f),
		normal(0.0f, 0.0f, 0.0f),
		tex0(0.0f, 0.0f){}
	VertexPNT(float x, float y, float z, 
		float nx, float ny, float nz,
		float u, float v):pos(x,y,z), normal(nx,ny,nz), tex0(u,v){}
	VertexPNT(const D3DXVECTOR3& v, const D3DXVECTOR3& n, const D3DXVECTOR2& uv)
		:pos(v),normal(n), tex0(uv){}

	D3DXVECTOR3 pos;
	D3DXVECTOR3 normal;
	D3DXVECTOR2 tex0;

	static IDirect3DVertexDeclaration9* Decl;
};
struct VertexPNT3
{
	D3DXVECTOR3 pos;
	D3DXVECTOR3 normal;
	D3DXVECTOR3 tex0;

	// this if we pass 3 component information for making colours.

	static IDirect3DVertexDeclaration9* Decl;

	VertexPNT3()
		:pos(0.0f, 0.0f, 0.0f),
		normal(0.0f, 0.0f, 0.0f),
		tex0(0.0f, 0.0f, 0.0f){}
	VertexPNT3(float x, float y, float z, 
		float nx, float ny, float nz,
		float u, float v, float w):pos(x,y,z), normal(nx,ny,nz), tex0(u,v,w){}
	VertexPNT3(const D3DXVECTOR3& v, const D3DXVECTOR3& n, const D3DXVECTOR3& uvw)
		:pos(v),normal(n), tex0(uvw){}
};
struct VertexPNf
{
	VertexPNf()
		:pos(0.0f, 0.0f, 0.0f),
		normal(0.0f, 0.0f, 0.0f),
		value(0.0f){}

	VertexPNf(float x, float y, float z, 
		float nx, float ny, float nz,
		float v):pos(x,y,z), normal(nx,ny,nz) {value = v;}
	
	VertexPNf(const D3DXVECTOR3& v, const D3DXVECTOR3& n, const float& l)
		:pos(v),normal(n) {value = l;}

	D3DXVECTOR3 pos;
	D3DXVECTOR3 normal;
	float value;

	static IDirect3DVertexDeclaration9* Decl;
};


struct vertex1{
			float x, y, z;  // The transformed(screen space) position for the vertex. z affects z-order; set to 0
			D3DCOLOR colour;        // The vertex colour.
		};
const DWORD point_fvf=D3DFVF_XYZ|D3DFVF_DIFFUSE;
		


// global function
int inline DXChk(HRESULT hresult, int identifier = 0)
{
	// this will be called around each DX call.
	static char errorstr[2048];
	static char errordes[4096];
	FILE * jillium;
	if (FAILED(hresult))
	{
		// summat wrong
		strcpy(errorstr,DXGetErrorString(hresult));
		strcpy(errordes,DXGetErrorDescription(hresult));
		jillium = fopen("errors2.txt","a");
		fprintf(jillium,"DXChk called with identifier %d \n%s\n%s\n\n",identifier,errorstr,errordes);
		fclose(jillium);
		return 1; // we use int so that we can add them together
		// hopefully, exit safely.
	};
	// else, it worked OK
	return 0;
};

//===============================================================
// Colors and Materials and DirLight struct

const D3DXCOLOR WHITE(1.0f, 1.0f, 1.0f, 1.0f);
const D3DXCOLOR BLACK(0.0f, 0.0f, 0.0f, 1.0f);
const D3DXCOLOR RED(1.0f, 0.0f, 0.0f, 1.0f);
const D3DXCOLOR GREEN(0.0f, 1.0f, 0.0f, 1.0f);
const D3DXCOLOR BLUE(0.0f, 0.0f, 1.0f, 1.0f);

struct Mtrl
{
	Mtrl()
		:ambient(WHITE), diffuse(WHITE), spec(WHITE), specPower(8.0f){}
	Mtrl(const D3DXCOLOR& a, const D3DXCOLOR& d, 
		 const D3DXCOLOR& s, float power)
		:ambient(a), diffuse(d), spec(s), specPower(power){}
	
	D3DXCOLOR ambient;
	D3DXCOLOR diffuse;
	D3DXCOLOR spec;
	float specPower;
};

struct DirLight
{
	D3DXCOLOR ambient;
	D3DXCOLOR diffuse;
	D3DXCOLOR spec;
	D3DXVECTOR3 dirW;
};




class DrawableTex2D
{
public:
	DrawableTex2D(UINT width, UINT height, UINT mipLevels,
		D3DFORMAT texFormat, bool useDepthBuffer,
		D3DFORMAT depthFormat, D3DVIEWPORT9& viewport, bool autoGenMips);
	~DrawableTex2D();

	IDirect3DTexture9* d3dTex();

	void beginScene();
	void endScene();

	void onLostDevice();
	void onResetDevice();

//private:
	// This class is not designed to be copied.
	DrawableTex2D(const DrawableTex2D& rhs);
	DrawableTex2D& operator=(const DrawableTex2D& rhs);

//private:
	IDirect3DTexture9*    mTex;
	ID3DXRenderToSurface* mRTS;
	IDirect3DSurface9*    mTopSurf;

	UINT         mWidth;
	UINT         mHeight;
	UINT         mMipLevels;
	D3DFORMAT    mTexFormat;
	bool         mUseDepthBuffer;
	D3DFORMAT    mDepthFormat;
	D3DVIEWPORT9 mViewPort;
	bool         mAutoGenMips;
};

class D3D
{
public:
	
	LPDIRECT3D9			g_pD3D;
	LPDIRECT3DDEVICE9	pd3dDevice; // rendering device

	LPD3DXFONT	g_pFont;
	LPD3DXFONT	g_pFont2;
	LPD3DXFONT	g_pFontsmall;
	D3DXFONT_DESC logfont;

	D3D()
	{
		g_pD3D = NULL;
		pd3dDevice = NULL;
	}
	
	void InitAllVertexDeclarations(void);
	void DestroyAllVertexDeclarations(void);

	HRESULT Initialise(HWND hWnd, HINSTANCE hInst, int WindowWidth, int WindowHeight)
	{
		//
		
		D3DPRESENT_PARAMETERS d3dpp;
		HRESULT hr;

		// Create the D3D object.
		if( NULL == ( g_pD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
			return E_FAIL;
	    
		// Set up the structure used to create the D3DDevice. Since we are now
		// using more complex geometry, we will create a device with a zbuffer.
		ZeroMemory( &d3dpp, sizeof( d3dpp ) );
		d3dpp.Windowed = TRUE;
		d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
		d3dpp.BackBufferWidth = WindowWidth; // set to 0 if not fullscreen
		d3dpp.BackBufferHeight = WindowHeight; // ditto
		d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8; // UNKNOWN is OK for windowed
		// force X8R8G8B8 because we need this for capturing the backbuffer
		d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;

		d3dpp.EnableAutoDepthStencil = TRUE; // setted it to false
		d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;//D3DFMT_D32;//D3DFMT_D16; // try unknown
	// trying to force 32bit depth buffer - we'll see what happens

		d3dpp.FullScreen_RefreshRateInHz = D3DPRESENT_RATE_DEFAULT;
		d3dpp.hDeviceWindow=hWnd;
		d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;

 /*
	DXChk(g_pD3D->CheckDeviceType(D3DADAPTER_DEFAULT,//Adapter
                           D3DDEVTYPE_HAL,    //DeviceType
                           D3DFMT_X8R8G8B8,     //DisplayFormat
                           D3DFMT_X8R8G8B8,     //BackBufferFormat
                           false)             //Windowed?
						   , 1 // identifier
						   );            
*/
		if(DXChk(g_pD3D->CreateDevice( D3DADAPTER_DEFAULT, 
									  D3DDEVTYPE_HAL, hWnd, // HAL is fastest already
                                      D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE, // change to HARDWARE
									  // trying D3DCREATE_FPU_PRESERVE to see if the program stops being f'd up
                                      &d3dpp, &pd3dDevice ) 
									  )){
										  MessageBox(NULL,"CreateDevice failed","",MB_OK);
        
										  return E_FAIL;};
	
		// create a font
		ZeroMemory(&logfont,sizeof(D3DXFONT_DESC));
		logfont.Height = 16; // made bigger
		logfont.Width = 5;
		logfont.Weight = 300;
		logfont.Italic = false;
		logfont.CharSet = DEFAULT_CHARSET;
		strcpy(logfont.FaceName,"Arial");
		D3DXCreateFontIndirect(pd3dDevice, &logfont, &g_pFont);
		
		ZeroMemory(&logfont,sizeof(D3DXFONT_DESC));
		logfont.Height = 24;
		logfont.Width = 8;
		logfont.Weight = 200;
		logfont.Italic = false;
		logfont.CharSet = DEFAULT_CHARSET;
		strcpy(logfont.FaceName,"Times");
		D3DXCreateFontIndirect(pd3dDevice, &logfont, &g_pFont2);
		
		ZeroMemory(&logfont,sizeof(D3DXFONT_DESC));
		logfont.Height = 15;
		logfont.Width = 5;
		logfont.Weight = 250;
		logfont.Italic = false;
		logfont.CharSet = DEFAULT_CHARSET;
		strcpy(logfont.FaceName,"Arial");
		D3DXCreateFontIndirect(pd3dDevice, &logfont, &g_pFontsmall);
		



		InitAllVertexDeclarations();	
		
		// Does nothing: ?
		// Turn on the zbuffer
		pd3dDevice->SetRenderState( D3DRS_ZENABLE, TRUE );
		
		pd3dDevice->SetRenderState( D3DRS_ZFUNC, D3DCMP_LESS );
		
		pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_CCW );
		// temporary
		// should be                             D3DCULL_CCW _NONE

		// Turn OFF D3D lighting
		pd3dDevice->SetRenderState( D3DRS_LIGHTING, FALSE );
		
		// I guess if you are using shaders then the above has no effect on anything
		return S_OK;
		
	}

	void RenderText(char * jibbles)
	{
		pd3dDevice->BeginScene();
		RECT rect;
		//char jibbles[300];
		//char jib2[300];

		rect.bottom=50;
		rect.top = 30;
		rect.left = 20;
		rect.right=900;

	//sprintf(jibbles,"x: %f y: %f z: %f",cart.x,cart.y,cart.z);
	//sprintf(jib2,"r: %f phi: %f theta: %f",sphericalsocalled.x,sphericalsocalled.y,sphericalsocalled.z);

		g_pFont->DrawText(NULL,jibbles,strlen(jibbles),&rect,DT_TOP|DT_LEFT,0xff000000);
		rect.bottom=70;
		rect.top = 50;
		pd3dDevice->EndScene();

	}


	~D3D()
	{
		// 
		DestroyAllVertexDeclarations();

		if ( g_pFont != NULL)
			g_pFont->Release();

		if ( g_pFont2 != NULL)
			g_pFont2->Release();

		if( pd3dDevice != NULL )
			pd3dDevice->Release();

		if( g_pD3D != NULL )
			g_pD3D->Release();
	}

};


extern D3D Direct3D;

#endif