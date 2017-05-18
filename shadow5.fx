
uniform extern float4x4 gWVP;
uniform extern float3 LightVecW;
uniform extern float4 DiffuseMtrl;
uniform extern float4 AmbientMtrl;
uniform extern float4 LightDiffuseColor;
uniform extern float4 LightAmbientColor;
uniform extern float4 LightSpecularColor;

// shadow map vars:
uniform extern texture gShadowMap;
uniform extern float4x4 gLightWVP;

// further vars:
uniform extern float3 gEyePosW;      // for specular
uniform extern float  SpecularPower;
uniform extern float4 SpecularMtrl;

uniform extern float Maxv; // == colourmax

uniform extern bool bCullNone;
uniform extern bool bTransparent;

uniform extern float fTransparentAlpha;

static const float EPSILON = 0.0001f;
static const float SMAP_SIZE = 4096.0f;


void BuildShadowMapVS(float3 posL : POSITION0,
					  out float4 posH: POSITION0,
					  out float2 depth : TEXCOORD0)
{
	// Render from light's perspective
	posH = mul(float4(posL,1.0f), gLightWVP);
	
	depth = posH.zw; // propagate coordinates
}


float4 BuildShadowMapPS(float2 depth : TEXCOORD0) : COLOR
{
	return depth.x/depth.y;
}


technique ShadowMapTech
{
    pass P0
    {
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 BuildShadowMapVS();
        pixelShader  = compile ps_2_0 BuildShadowMapPS();
    }
}

sampler ShadowMapS = sampler_state
{
	Texture = <gShadowMap>;
	MinFilter = POINT;
	MagFilter = POINT;
	MipFilter = POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
};



struct OutputVS
{
	float4 posH : POSITION0;
    float4 ambient : COLOR0;
    float4 diffuse : COLOR1;
    
    float4 oProjTex : TEXCOORD0;
    
    float4 spec : TEXCOORD1;
    float4 graphcolor : TEXCOORD2;
};

// assume we set gWVP, LightVecW, AmbientMtrl, DiffuseMtrl.

OutputVS ShadowedVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float2 velocity : TEXCOORD0
					 )
{
    // Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
	float3 normalW = normalize(normal1);
	float3 r = reflect(-LightVecW, normalW);
	float s = max(dot(LightVecW, normalW), 0.0f);
	
	outVS.ambient = AmbientMtrl;
	outVS.diffuse = s*DiffuseMtrl;
	
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
	
    return outVS;
}


OutputVS AzSegueVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float3 colvarxy : TEXCOORD0
					 )
{
// for when things can be + and -
	float colvar = colvarxy.x/Maxv;
    float thresh = 0.3;
    
    // change abruptly from white to blue where it goes beneath 0.
    
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
	float3 normalW = normalize(normal1);
	
	outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);
	
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
		
	// calculate graphcolor from segue:

    // new colour scheme:	
    // blue     = -1   (0.0,0.0,1.0)   and below that, fade to black
    // cyan    = -thresh        (0,0.9,1.0)
    // yellow = 0   (0.9,0.9,0)
    // pink            (1.0,0.4,0.4)
    // orange        (1.0,0.7,0)     and above that, fade to ... light purple (1.0,0.7,1.0) ?

//    if (colvar < -1.0f)
//    {
//// won't happen
//        outVS.graphcolor.r = 0.0f;
//        outVS.graphcolor.g = 0.0f; 
//        outVS.graphcolor.b = exp(-(colvar+1.0f)*(colvar+1.0f)); // exp(-0)=1           
//    } else {
        //if (colvar < -thresh) {
        //    outVS.graphcolor.r = 0.0f;
        //    outVS.graphcolor.g = colvar*colvar;
        //    outVS.graphcolor.b = 1.0f;
       
     
        //if (colvar < -thresh) {
        //    outVS.graphcolor.r = 0.0f;
        //    outVS.graphcolor.g = 0.9f*(colvar+1.0f)/(-thresh+1.0f);
        //    outVS.graphcolor.b = 1.0f;
        //} else {
        //    if (colvar < 0.0f) {
        //        outVS.graphcolor.r = 0.9f*(colvar+thresh)/thresh;
        //        outVS.graphcolor.g = 0.9f;
        //        outVS.graphcolor.b = 1.0f-(colvar+thresh)/thresh;
        //    } else {
        //        if (colvar < thresh) {
        //            outVS.graphcolor.r = 0.9f+0.1f*colvar/thresh;
        //            outVS.graphcolor.g = 0.9f-0.5f*colvar/thresh;
        //            outVS.graphcolor.b = 0.4f*colvar/thresh;
        //        } else {
        //            if (colvar < 1.0f) {
        //                outVS.graphcolor.r = 1.0f;
        //                outVS.graphcolor.g = 0.4f+0.3f*(colvar-thresh)/(1.0-thresh);
        //                outVS.graphcolor.b = 0.4f-0.4f*(colvar-thresh)/(1.0-thresh);
        //            } else {
        //                outVS.graphcolor.r = 1.0f;
        //                outVS.graphcolor.g = 0.7f;
        //                outVS.graphcolor.b = exp(-(colvar-1.0f)*(colvar-1.0f));
        //            };
        //        };
        //    };
        //};
   // };

    //if (colvar < 0.0f)
    //{
    //    if (colvar < -thresh)
    //    {
    //        outVS.graphcolor.r = 0.0f;
    //        outVS.graphcolor.g = -colvar - thresh; // grows to 0.6
    //        outVS.graphcolor.b = 1.0f;
    //    } else {
    //        outVS.graphcolor.r = 0.0f;
    //        outVS.graphcolor.g = 0.0f;
    //        outVS.graphcolor.b = colvar/(-thresh);
    //    };
    //} else {
    //    if (colvar > thresh)
    //    {
    //        outVS.graphcolor.r = 1.0f;
    //        outVS.graphcolor.g = colvar -thresh;
    //        outVS.graphcolor.b = 0.0f;
    //    } else {
    //        outVS.graphcolor.r = colvar/thresh;
    //        outVS.graphcolor.g = 0.0f;
    //        outVS.graphcolor.b = 0.0f;
    //    };
    //};

    //outVS.graphcolor.r += 0.1f;
    //outVS.graphcolor.g += 0.1f;
    //outVS.graphcolor.b += 0.1f;	

   if (colvar < 0.0f) {
        //outVS.graphcolor.r = -0.6f*colvar*(1.0f+colvar);
        //outVS.graphcolor.g = -1.2f*colvar;
        //outVS.graphcolor.b = 1.0f - 0.1f*colvar;

// negative: blue

        outVS.graphcolor.r = 1.0f + colvar;//-0.5f*colvar*(1.0f+colvar);
if (colvar > -0.5f) {
        outVS.graphcolor.g = 1.0f + 4.0f*colvar*(1.0f+colvar); 
} else {
        outVS.graphcolor.g = 0.4f + 1.6f*colvar*(1.0f+colvar); 
}
        outVS.graphcolor.b = 1.0f;// - 0.1f*colvar;

if (colvar < -1.0f) {
outVS.graphcolor.r = 0.0f;
outVS.graphcolor.g = 0.4f;
outVS.graphcolor.b = 1.0f;
};

    } else {

// positive: red
        outVS.graphcolor.r = 1.0f;
if (colvar < 0.5f) {
        outVS.graphcolor.g = 1.0f-4.0f*colvar*(1.0-colvar);
} else {
        outVS.graphcolor.g = 0.4f-1.6f*colvar*(1.0-colvar);
}
                    // (colvar + colvar*colvar);
        outVS.graphcolor.b = 1.0f-colvar;//
                    // 1.0f - 2.5f*(colvar-0.5f)*(colvar-0.5f);
// colvar-1/2 ^2 goes from 0 to 1/4 say.
// colvar = 0 --> want (1.0,1.0,1.0)
// colvar = 0.5 --->  (1.0,0.5,0.5)
// colvar = 1.0 ---> (1.0,0.5,0.0)

if (colvar > 1.0f) {
outVS.graphcolor.r = 1.0f;
outVS.graphcolor.g = 0.4f;
outVS.graphcolor.b = 0.0f;
}
    };

    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;

    return outVS;
}

OutputVS Ionise_VS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float3 colvarxy : TEXCOORD0
					 )
{

	static const float4 yellow = float4(1.0f,1.0f,0.0f,1.0f);
	static const float4 white = float4(1.0f,1.0f,1.0f,1.0f);
	static const float4 red = float4(1.0f,0.2f,0.2f,1.0f);
	static const float4 greencyan = float4(0.1f,1.0f,0.5f,1.0f);
	static const float4 cyan = float4(0.5f,0.8f,0.9f,1.0f);
	static const float4 blue = float4(0.3f,0.8f,1.0f,1.0f);
	static const float4 pink = float4(1.0f,0.8f,0.9f,1.0f);
	static const float4 violet = float4(0.6f,0.3f,1.0f,1.0f);
	
    // For now just did this but we want to include temperature also.

	float segthresh[4] = {0.0f,0.05f, 0.95f, 1.0f};
    float4 rgb[4] = {pink,  yellow, blue, violet};
	
	float colvar = colvarxy.x;
	
	float Tvar = colvarxy.y;

    float3 hue;
	float ppn;
	
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
	float3 normalW = normalize(normal1);
	
	outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);
	
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
	
    int iWhich = 0;
    int i;
    for (i = 1; i < 4; i++)
    {
        if (colvar > segthresh[i]) iWhich = i;
    };
    // iWhich is the last one it is > .
    if (iWhich == 3) {
        outVS.graphcolor = violet;
    } else {
        ppn = (colvar-segthresh[iWhich])/(segthresh[iWhich+1]-segthresh[iWhich]);
        outVS.graphcolor = rgb[iWhich+1] * ppn+ rgb[iWhich]*(1.0f-ppn);
    };


    //outVS.graphcolor.b = colvar+0.5f*(1.0f-colvar)*colvar; // Should be ON where ionised !!
    //outVS.graphcolor.r = Tvar/8.0e-12; // should generally be smallish.
    //if (Tvar > 8.0e-12) outVS.graphcolor.r = 1.0f;
    //outVS.graphcolor.g = 0.5f;
    //if (colvar > 0.95f) outVS.graphcolor.g = 0.5f+0.5f*(colvar-0.95f)/0.05f; // turn on if near full or zero ionisation.
    //if (colvar < 0.05f) outVS.graphcolor.g = 0.5f+0.5f*(0.05f-colvar)/0.05f; // turn on if near full or zero ionisation.

    // colvar: ionisation %, presumably
    
    //outVS.graphcolor.b = colvar;
    //outVS.graphcolor.g = 1.0f;
    //outVS.graphcolor.r = colvar;
    
    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;

    return outVS;
}

OutputVS SegueVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float3 colvarxy : TEXCOORD0
					 )
{
	// just going to use magic numbers for now ... that's why they call it development.

	static const float4 yellow = float4(1.0f,1.0f,0.0f,1.0f);
	static const float4 white = float4(1.0f,1.0f,1.0f,1.0f);
	static const float4 red = float4(1.0f,0.2f,0.2f,1.0f);
	static const float4 greencyan = float4(0.0f,1.0f,0.5f,1.0f);
	static const float4 cyan = float4(0.5f,0.8f,0.9f,1.0f);
	static const float4 blue = float4(0.3f,0.5f,1.0f,1.0f);
	static const float4 pink = float4(0.8f,0.4f,0.8f,1.0f);
	static const float4 violet = float4(0.6f,0.3f,1.0f,1.0f);
	
	float segthresh[8] = {0.0f,4.0e-14, 1.2e-12, 2.7e-12, 4.8e-12, 6.75e-12,9.0e-12, 1.4e-11};
    float4 rgb[8] = {cyan, blue, greencyan,  red, yellow, white, pink, violet};
	
	float colvar = colvarxy.x/Maxv;
	
    float3 hue;
	float ppn;
	
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
	float3 normalW = normalize(normal1);
	
	outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);
	
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
	
    int iWhich = 0;
    int i;
    for (i = 1; i < 8; i++)
    {
        if (colvar > segthresh[i]) iWhich = i;
    };
    // iWhich is the last one it is > .
    if (iWhich == 7) {
        outVS.graphcolor = violet;
    } else {
        ppn = (colvar-segthresh[iWhich])/(segthresh[iWhich+1]-segthresh[iWhich]);
        outVS.graphcolor = rgb[iWhich+1] * ppn+ rgb[iWhich]*(1.0f-ppn);
	};
	// calculate graphcolor from segue:
	
    outVS.graphcolor.r += 0.05;
    outVS.graphcolor.g += 0.05;
    outVS.graphcolor.b += 0.05; // lighten up a bit
	
    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;
    
    //clip(posL.x); // does not compile

    //if (posL.x/posL.z < -0.09849f) {
    //    outVS.graphcolor.a = 0.25f; // a special flag because alpha < 1 seems to
    //    // errantly appear all over the place z-fighting with our actual thing
    
    ////    outVS.graphcolor.g = 1.0f;
    ////    outVS.graphcolor.b = 1.0f; // try what happens - why are too many things passing this test?

    //    outVS.graphcolor.r = -0.25f; // special flag
    //    // try this as flag.
    //};
    
    return outVS;
}


OutputVS SigmaVS( float3 posL : POSITION0,
				  float3 normal1 : NORMAL0,
				  float3 sigma : TEXCOORD0
				)
{
    static const float redlim = -0.5235988; // PI/6
	static const float greenlim = -2.617994; 
	static const float greenplus = 3.665191; // PI-redlim
	static const float bluelim = 1.570796; // PI/2
	static const float range = 2.0943951; // PI*2/3
	static const float TWOPI = 6.2831853;
	static const float redplus = 5.7595865; // redlim+2pi

    float3 WT; 

    WT.r = 0.299f; WT.g = 0.587f; WT.b = 0.114f; // colours from internet sites   
        
	float3 hue;
	float ppn;
	
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 	
    float3 normalW = normalize(normal1);
	
	outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);
		
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
		
	// calculate graphcolor from sigma:
	// ===================================

    // sigma.y = perp

    // Over time perp decays as proportion of sigma.x = sigma_parallel
    // When it reaches 50% is when the Hall component sigma.z is at its max of 50% sigma_parallel

    // Set most green at the max Hall point.
    // Violet = highly magnetized.

	// multiply colour by a brightness
    
    
    hue.g = 1.8f*(sigma.z/sigma.x); // may be negative; 
    if (hue.g < 0.0f) hue.g = -hue.g;
    // ~ [ nu omega / nu^2 + omega^2 ] -- should peak at 0.5 where nu = omega
    
    
    
    //hue.b = 0.2+0.8*(sigma.x-sigma.y)/sigma.x; 
    
    if (sigma.y > 0.5f*sigma.x)
    {
        hue.r = 1.0f - 2.0f*(sigma.x-sigma.y)/sigma.x;
    } else {
        hue.r = 0.0f;   
    };
        
    hue.r = hue.r + 0.32*(sigma.y-sigma.x)*(sigma.y-sigma.x)/(sigma.x*sigma.x);

    // set hue.b to make perceived luminosity constant?!

    if (WT.r*hue.r*hue.r + WT.g*hue.g*hue.g < 0.6)
    {
        hue.b = sqrt((0.6 - WT.r*hue.r*hue.r - WT.g*hue.g*hue.g)/WT.b);
    } else {
        hue.b = 0.0f;
    };
    // Note 1.0 total luminosity is high. We go for 0.6.

    if (sigma.y < 0.0f) { hue.g = 1.0; hue.r = 0.0; hue.b = 0.8;} // negative perp
    if (sigma.x < 0.0f) { hue.g = 0.5; hue.r = 1.0; hue.b = 0.2;} // negative parallel
       
    
	// probably is a neater command:
	outVS.graphcolor.r = hue.r;//(1.0f-darkness) + hue.r*darkness;
	outVS.graphcolor.g = hue.g;
	outVS.graphcolor.b = hue.b;	
		
    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;
	

    return outVS;
}
OutputVS XYZ_VS( float3 posL : POSITION0,
                                float3 normal1: NORMAL0,
                                float3 velo: TEXCOORD0     // we'll find out if this actually works...
                            )
{
    
	static const float4 white = float4(1.0f,1.0f,1.0f,1.0f);
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;

    float vr_ppn;	

	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
    float3 normalW = normalize(normal1);
		
    outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);

	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
	
	// calculate graphcolor from velocity:
	// let's convert to cylindrical polar components?
    
    // radial:
    float2 radial;
    radial.x = posL.x/sqrt(posL.x*posL.x+posL.z*posL.z);
    radial.y = posL.z/sqrt(posL.x*posL.x+posL.z*posL.z);
    
    float3 polar_v;
    polar_v.x = radial.x*velo.x + radial.y*velo.y; // radial component
    polar_v.y = -radial.y*velo.x + radial.x*velo.y; // azimuthal component - sign doesn't matter
    polar_v.z = velo.z;
    
    if ((polar_v.x == 0.0f) && (polar_v.y == 0.0f) && (polar_v.z == 0.0f))
    {
        outVS.graphcolor.r = 1.0f;
        outVS.graphcolor.g = 1.0f;
        outVS.graphcolor.b = 1.0f;
    } else {

        // Intensity per magnitude;
        // Maxv is max magnitude.

        //float3 unitv = normalize(polar_v);
        
        float radsq = velo.x*velo.x+velo.y*velo.y+velo.z*velo.z; // maybe velo.z^2 > 1e38.
        float leng = sqrt(radsq); 
        float intensity = leng/Maxv;
        float3 unitv;     

        float3 WT; WT.r = 0.299f; WT.g = 0.587f; WT.b = 0.114f; // colours from internet sites   
        
   //     //float leng = abs(velo.x)+abs(velo.y)+abs(velo.z);  // test.  

//        unitv.x = abs(polar_v.x) / leng;
  //      unitv.y = abs(polar_v.y) / leng;        
    //    unitv.z = abs(velo.z)/leng;

        //if ((unitv.z > 1.000001f) || (unitv.y > 1.000001f) || (unitv.x > 1.000001f)) // nonsense
        //{
        //        outVS.graphcolor.r = 0.0f;
        //        outVS.graphcolor.g = 1.0f;
        //        outVS.graphcolor.b = 1.0f;
        //    return outVS;
        //} 

        //if ((unitv.z < 0.0f) || (unitv.y < 0.0f) || (unitv.z < 0.0f)) // nonsense
        //{
        //        outVS.graphcolor.r = 1.0f;
        //        outVS.graphcolor.g = 0.0f;
        //        outVS.graphcolor.b = 1.0f;
        //        return outVS;
        //};
        // Try another way:
        unitv.b = asin(abs(velo.z/leng))/1.57079632f;
        unitv.g = unitv.b*0.25f;
        unitv.r = 0.0f;

        if (leng == 0.0f) unitv.g = 1.0f;


        float desiredluminositysq = WT.b + WT.g*0.25f*0.25f + 0.12f * (1.0f - unitv.b);
            // allow that the equator is more luminous than the poles.
        
        float existingluminositysq = WT.b*unitv.b*unitv.b + WT.g*unitv.g*unitv.g
                                    + WT.r*unitv.r*unitv.r;
        // max at top = 0.1506875
        
        // Now sqrt( 0.114 + 0.587*0.25*0.25 ) = 0.388 is the luminosity at the top.
        // Aim for no more than 0.5 at the equator.

        // Put r outward, g inward, yellowy to the sides
        // We could re-jig to put g at 2/3 backward on both sides, get as far as orange in the bwd direction
        
       // float percentg = 0.5f*(polar_v.x/sqrt(velo.x*velo.x+velo.y*velo.y)+1.0f);
        // want something that comes out nearer what it is for polar_v == 0 for longer.
        
        vr_ppn = polar_v.x*polar_v.x/(velo.x*velo.x+velo.y*velo.y);
    
        vr_ppn = vr_ppn * (polar_v.x/
                sqrt(velo.x*velo.x+velo.y*velo.y)); 

        //  suspect it failed as one func because 6e14^3 = 2.16e44 > 1e38, the max float value.



        //  seems we persistently may be running into mysterious errors connected with that.

        // note: take out abs() if we want to then multiply by the second bit.

        // should range between -1 and 1 as v rotates

        float percentg = 0.5f*(vr_ppn+1.0f);
            
        // now add smth * per                  centg * green + (1.0-percentg) * smth * red
        
        // Now solve .. 0.114*unitv.z*unitv.z + 0.587*(unitv.g + x * %g )^2 + 0.299*(unitv.r + x*(1-%g))^2 
            // more general in case we ever add unitv.r above
        // (0.587 %g^2 + 0.299 (1-%g)^2) x^2 + (0.587*2*unitv.g*%g + 0.299*unitv.r*2*(1-%g) ) x =
        // -0.114*unitv.z*unitv.z - 0.587*unitv.g^2 - 0.299*unitv.r^2
        float coeffxsq = WT.g*percentg*percentg + WT.r*(1.0f-percentg)*(1.0f-percentg);
        float coeffx = WT.g*2.0f*unitv.g*percentg + WT.r*2.0f*unitv.r*(1.0f-percentg);
        float RHS = desiredluminositysq - WT.b*unitv.b*unitv.b - WT.g*unitv.g*unitv.g - WT.r*unitv.r*unitv.r;
        coeffx = coeffx/(2.0f*coeffxsq); // divided by 2 for completing the square
        RHS = RHS/coeffxsq + coeffx*coeffx; // moved 0.25 coeffx^2 across to RHS
        float x = sqrt(RHS) - coeffx;
             
        unitv.r += (1.0f-percentg)*x;
        unitv.g += percentg*x;

        
        // Finally boost up more: (?!)

        
        x = sqrt(0.4/(WT.r*unitv.r*unitv.r+WT.g*unitv.g*unitv.g+WT.b*unitv.b*unitv.b));
        if (x > 1.0f) {
            unitv = x*unitv;
            // rendering some of the above a waste of time
        };



        if (radsq < 0.0f) // nonsense
        {
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 1.0f;
            outVS.graphcolor.b = 0.5f;
            return outVS;
        };
        if (leng < 0.0f) // nonsense
        {
            outVS.graphcolor.r = 0.5f;
            outVS.graphcolor.g = 1.0f;
            outVS.graphcolor.b = 1.0f;
            return outVS;
        };
        
/*
        leng = sqrt(unitv.x*unitv.x+unitv.y*unitv.y+unitv.z*unitv.z);

        if (leng < 0.9f) 
        {
// This is what happens.
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 0.9f;
            outVS.graphcolor.b = 0.9f;

           return outVS;
        };
   
        if (leng > 1.1f) 
        {
            outVS.graphcolor.r = 0.5f;
            outVS.graphcolor.g = 1.0f;
            outVS.graphcolor.b = 0.9f;

            return outVS;
        };*/

        outVS.graphcolor.r = unitv.r*intensity + (1.0f-intensity)*white.x;
        outVS.graphcolor.g = unitv.g*intensity + (1.0f-intensity)*white.y;
        outVS.graphcolor.b = unitv.b*intensity + (1.0f-intensity)*white.z;

        //if (velo.y == 0.0f) { outVS.graphcolor.g = 1.0f; };
   };
    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;

    return outVS;
}

OutputVS VelocityVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float3 velocity : TEXCOORD0
					 )
{
    static const float redlim = -0.5235988; // PI/6
	static const float greenlim = -2.617994; 
	static const float greenplus = 3.665191; // PI-redlim
	static const float bluelim = 1.570796; // PI/2
	static const float range = 2.0943951; // PI*2/3
	static const float TWOPI = 6.2831853;
	static const float redplus = 5.7595865; // redlim+2pi

	float3 hue;
	float ppn;
	
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
//    if (normal1.y < 0.0) {
    //    normal1.x = -normal1.x;
      //  normal1.y = -normal1.y;
        //normal1.z = -normal1.z;
  //      };
	
    float3 normalW = normalize(normal1);
	
	outVS.ambient = AmbientMtrl*LightAmbientColor;
	
	float s = max(dot(LightVecW, normalW), 0.0f);
	outVS.diffuse = s*DiffuseMtrl*LightDiffuseColor;
	
	float3 ToEye = normalize(gEyePosW - posL); 
	float3 r = reflect(-LightVecW, normalW);
	float3 t = pow(max(dot(r,ToEye),0.0f),SpecularPower);
	float3 spec = t*(SpecularMtrl*LightSpecularColor).rgb;
	outVS.spec = float4(spec, 0.0f);
	
	
	// generate projective texture coordinates to project shadow map on to scene
	outVS.oProjTex = mul(float4(posL,1.0f),gLightWVP);
	
	
	// calculate graphcolor from velocity:
	
	// multiply colour by a brightness
	
    //float darkness = sqrt(velocity.x*velocity.x+velocity.y*velocity.y)/Maxv;
	
    float darkness = sqrt(velocity.x*velocity.x+velocity.y*velocity.y)/Maxv;
	

// used sqrt now. getting cornetto without.

// try below to raise it to 0.8 but doesn't work because despite the following test
    // it thereafter claims pow is called with negative f.
    // Why did I mess with it? sqrt was fine.

    if (darkness < 0.00001f)
	{
		// do not call atan2 for values near 0,0
		hue.r = 1.0f;
		hue.g = 1.0f;  
		hue.b = 1.0f;  // output white
		
		// probably is a neater command:
		outVS.graphcolor.r = 1.0f;
		outVS.graphcolor.g = 1.0f;//posL.y;
		outVS.graphcolor.b = 0.0f; // yellow is our code for 'no velocity'
		// and won't be obtained by other means
	
	} else {
	
        // darkness = pow(darkness, 0.8f);
        // Triggers "pow(f,e) will not work for negative f" -- don't know why

		float theta = atan2(velocity.y,velocity.x);
		
		if (theta < greenlim || theta > bluelim)
		{
			if (theta < 0.0) { theta += TWOPI; };
			
			ppn = (greenplus-theta)/range;
			
			hue.r = 0.0f;
			hue.g = 1.0f-ppn;
			hue.b = ppn;
		} else {
			if (theta < redlim)
			{
				ppn = (theta-greenlim)/range;
				hue.r = ppn;
				hue.g = 1.0f-ppn;
				hue.b = 0.0f;
			} else {
				ppn = (bluelim-theta)/range;
				hue.r = ppn;
				hue.g = 0.0f;
				hue.b = 1.0f-ppn;
			};
		};

        // multiply hue to get luminosity.

        float3 WT;
        WT.r = 0.299f; WT.g = 0.587f; WT.b = 0.114f;
        // lumsq  = 
        // x^2 hue.r^2 0.299 + x^2 hue.g^2 0.587 + x^2 hue.b^2 0.114
        
        // Aim lumsq = 0.5:
        // x^2 = 0.5 / existing lumsq
        
        float x = sqrt(0.5/(WT.r*hue.r*hue.r+WT.g*hue.g*hue.g+WT.b*hue.b*hue.b));
        hue.r = hue.r*x;
        hue.g = hue.g*x;
        hue.b = hue.b*x;
			
		// probably is a neater command:
		outVS.graphcolor.r = (1.0f-darkness) + hue.r*darkness;
		outVS.graphcolor.g = (1.0f-darkness) + hue.g*darkness;
		outVS.graphcolor.b = (1.0f-darkness) + hue.b*darkness;
	
		
	};       // whether v near= 0
	
    outVS.graphcolor.a = 1.0f;
    if (bTransparent) outVS.graphcolor.a = fTransparentAlpha;

// temp:
/*
    outVS.graphcolor.r = 0.0f;
    outVS.graphcolor.g = 0.0f;
    outVS.graphcolor.b = 0.0f;

    if (velocity.x > 1.0f)
    {
        outVS.graphcolor.b = 1.0f;
        outVS.graphcolor.g = 1.0f;
    };
    if (velocity.x > 110.0f)
    {
        outVS.graphcolor.b = 0.0f;
        outVS.graphcolor.g = 1.0f;
        
    };
    if (velocity.x < 0.4f)
    {
        if (velocity.y > 1.0f)
        {
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 0.5f;
        };
        if (velocity.y > 140.f)
        {
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 0.2f;
            outVS.graphcolor.b = 0.2f;
	    };
    };*/
    return outVS;
}


OutputVS MeshVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0,
					 float3 velocity : TEXCOORD0
					 )
{
    static const float redlim = -0.5235988; // PI/6
	static const float greenlim = -2.617994; 
	static const float greenplus = 3.665191; // PI-redlim
	static const float bluelim = 1.570796; // PI/2
	static const float range = 2.0943951; // PI*2/3
	static const float TWOPI = 6.2831853;
	static const float redplus = 5.7595865; // redlim+2pi

	float3 hue;
	float ppn;
	
	// Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
    if (normal1.y < 0.0) {
    //    normal1.x = -normal1.x;
      //  normal1.y = -normal1.y;
        //normal1.z = -normal1.z;
        };
	float3 normalW = normalize(normal1);
	
	
    outVS.graphcolor.r = velocity.x;
    outVS.graphcolor.g = velocity.y;
    outVS.graphcolor.b = velocity.z;

/*
    if (velocity.y > 0.5f)
    {
        outVS.graphcolor.b = 1.0f;
        outVS.graphcolor.g = 0.2f;
        // colour for fixed mesh
    };

    if (velocity.x > 1.0f)
    {
        outVS.graphcolor.b = 1.0f;
        outVS.graphcolor.g = 1.0f;
    };
    if (velocity.x > 110.0f)
    {
        outVS.graphcolor.b = 0.0f;
        outVS.graphcolor.g = 1.0f;
        
    };
    if (velocity.x < 0.4f)
    {
        if (velocity.y > 1.0f)
        {
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 0.5f;
        };
        if (velocity.y > 140.f)
        {
            outVS.graphcolor.r = 1.0f;
            outVS.graphcolor.g = 0.2f;
            outVS.graphcolor.b = 0.2f;
	    };
    };*/

    return outVS;
}


float4 ShadowedPS(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0) : COLOR
{
	projTex.xy /= projTex.w;
	projTex.x = 0.5f*projTex.x + 0.5f;
	projTex.y = -0.5f*projTex.y + 0.5f;
	
	float depth = projTex.z / projTex.w;
	
	float2 texelpos = SMAP_SIZE*projTex.xy;
	
    float shadowCoeff = (tex2D(ShadowMapS, projTex.xy).r + EPSILON < depth)? 0.0f : 1.0f;
    
   // if (shadowCoeff > 0.5f) return float4(0.33*ambient.rgb + diffuse.rgb, 1.0f);

    if (shadowCoeff > 0.5f) return float4(0.0*ambient.rgb + 0.2*diffuse.rgb, 1.0f);
    
    return float4(0.0f,0.0f,1.0f,1.0f);
    
}


float4 ApplyShadowPS(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0,
				    float4 spec: TEXCOORD1,
					float4 graphcolor : TEXCOORD2 ) : COLOR
{
	projTex.xy /= projTex.w;
	projTex.x = 0.5f*projTex.x + 0.5f;
	projTex.y = -0.5f*projTex.y + 0.5f;
	
	float depth = projTex.z / projTex.w;
	float SHADOW_EPSILON = EPSILON;
	float2 texelpos = SMAP_SIZE*projTex.xy;
	float2 lerps = frac(texelpos);
    float dx = 1.0f/ SMAP_SIZE;    
    float s0 = (tex2D(ShadowMapS, projTex.xy).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s1 = (tex2D(ShadowMapS, projTex.xy + float2(dx,0.0f)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s2 = (tex2D(ShadowMapS, projTex.xy + float2(0.0f,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s3 = (tex2D(ShadowMapS, projTex.xy + float2(dx,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	
    float shadowCoeff = lerp(lerp(s0,s1,lerps.x),lerp(s2,s3,lerps.x),lerps.y);
    
    // old(?) :
    //float shadowCoeff = (tex2D(ShadowMapS, projTex.xy).r + EPSILON < depth)? 0.0f : 1.0f;

// TEMPORARY:
  //  shadowCoeff = 1.0f;

    // temp:
//    if (shadowCoeff < 0.5) return float4(0.0f,0.0f,0.0f,0.0f);
    
    if (bTransparent == 0) {
        graphcolor.a = 1.0f;
    } else {
        graphcolor.a = fTransparentAlpha;
    };

    return float4( ambient.rgb*graphcolor.rgb 
                + shadowCoeff*diffuse.rgb*graphcolor.rgb
                + shadowCoeff*spec.xyz, graphcolor.a);
    
}

float4 ApplyShadowPSNotCancelled (float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0,
				    float4 spec: TEXCOORD1,
					float4 graphcolor : TEXCOORD2 ) : COLOR
{

    clip ((graphcolor.r == -0.25f)?-1:1);
    
	projTex.xy /= projTex.w;
	projTex.x = 0.5f*projTex.x + 0.5f;
	projTex.y = -0.5f*projTex.y + 0.5f;
	
	float depth = projTex.z / projTex.w;
	float SHADOW_EPSILON = EPSILON;
	float2 texelpos = SMAP_SIZE*projTex.xy;
	float2 lerps = frac(texelpos);
    float dx = 1.0f/ SMAP_SIZE;    
    float s0 = (tex2D(ShadowMapS, projTex.xy).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s1 = (tex2D(ShadowMapS, projTex.xy + float2(dx,0.0f)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s2 = (tex2D(ShadowMapS, projTex.xy + float2(0.0f,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s3 = (tex2D(ShadowMapS, projTex.xy + float2(dx,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	
    float shadowCoeff = lerp(lerp(s0,s1,lerps.x),lerp(s2,s3,lerps.x),lerps.y);
    
    
    float4 returnvalue;
    returnvalue.rgb = ambient.rgb*graphcolor.rgb 
                + shadowCoeff*diffuse.rgb*graphcolor.rgb
                + shadowCoeff*spec.xyz;

    returnvalue.a = 1.0f;

    return returnvalue;
    
}

float4 ApplyShadowPSTransparentOnly(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0,
				    float4 spec: TEXCOORD1,
					float4 graphcolor : TEXCOORD2 ) : COLOR
{
    clip((graphcolor.a != 0.25f)?-1:1);
    // our whole thing seems to be full of alpha !=1 z-fighting with alpha = 1
    // so here I use a special flag for the bit I want to be transparent.
    
	projTex.xy /= projTex.w;
	projTex.x = 0.5f*projTex.x + 0.5f;
	projTex.y = -0.5f*projTex.y + 0.5f;
	
	float depth = projTex.z / projTex.w;
	float SHADOW_EPSILON = EPSILON;
	float2 texelpos = SMAP_SIZE*projTex.xy;
	float2 lerps = frac(texelpos);
    float dx = 1.0f/ SMAP_SIZE;    
    float s0 = (tex2D(ShadowMapS, projTex.xy).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s1 = (tex2D(ShadowMapS, projTex.xy + float2(dx,0.0f)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s2 = (tex2D(ShadowMapS, projTex.xy + float2(0.0f,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	float s3 = (tex2D(ShadowMapS, projTex.xy + float2(dx,dx)).r + SHADOW_EPSILON < depth) ? 0.5f : 1.0f;
	
    float shadowCoeff = lerp(lerp(s0,s1,lerps.x),lerp(s2,s3,lerps.x),lerps.y);
    
    float4 returnvalue;
    returnvalue.rgb = ambient.rgb*graphcolor.rgb 
                + shadowCoeff*diffuse.rgb*graphcolor.rgb
                + shadowCoeff*spec.xyz;
    
    returnvalue.a = 0.1f;
    

    return returnvalue;
    
}

float4 NoShadowPS(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0,
				    float4 spec: TEXCOORD1,
					float4 graphcolor : TEXCOORD2 ) : COLOR
{
    graphcolor.a = 1.0f; // not transparent - used for mesh drawing

    return graphcolor;    
}

float4 ClipPS(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1,
				   float4 projTex : TEXCOORD0,
				    float4 spec: TEXCOORD1,
					float4 graphcolor : TEXCOORD2 ) : COLOR
{
  //  clip ((graphcolor.a < 1.0f)?-1:1);

    if (graphcolor.a == 0.0f) graphcolor.a = 1.0f;
        // force all those points to display
        
    if (graphcolor.a < 1.0f) {
        graphcolor.b = 1.0f;
        graphcolor.r = 1.0f; 
        graphcolor.g = graphcolor.a;  // what happens?
    } else {

    };
    // clip() creates a mess even though we pass graphcolor.a = 1.0f.

    // graphcolor.a does not come through as what it is passed as. Explanation??
    
    return graphcolor;    
}


OutputVS SimplestVS( float3 posL : POSITION0,
					 float3 normal1 : NORMAL0
					 )
{
    // Zero out our output.
	OutputVS outVS = (OutputVS)0;
	
	// Transform to homogeneous clip space.
	outVS.posH = mul(float4(posL, 1.0f), gWVP);
	 
	float3 normalW = normalize(normal1);
	float3 r = reflect(-LightVecW, normalW);
	float s = max(dot(LightVecW, normalW), 0.0f);
	
	outVS.ambient = AmbientMtrl;
	outVS.diffuse = s*DiffuseMtrl;
	
    return outVS;
}


float4 SimplestPS(float4 ambient : COLOR0, 
				   float4 diffuse : COLOR1) : COLOR
{
    float shadowCoeff = 1.0f;
    return float4(0.3*ambient.rgb + shadowCoeff*diffuse.rgb, 1.0f);
}


technique WarwickTech
{
    pass P0
    {
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 ShadowedVS();
        pixelShader  = compile ps_2_0 ShadowedPS();
    }
}

technique SigmaTech
{
    pass P0
    {
      //  CullMode = CCW;
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 SigmaVS();
        pixelShader  = compile ps_2_0 ApplyShadowPS();
       // FillMode = Wireframe;
    }
}

technique VelociTech
{
    pass P0
    {
     //   CullMode = CCW;
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 VelocityVS();
        pixelShader  = compile ps_2_0 ApplyShadowPS();
       // FillMode = Wireframe;
    }
}

technique XYZTech
{
    pass P0
    {
       // CullMode = CCW;
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 XYZ_VS();
        pixelShader  = compile ps_2_0 ApplyShadowPS();
       // FillMode = Wireframe;
    }
}

technique IoniseTech
{
    pass P0
    {
     //   CullMode = CCW;
// Don't think we can put branching here. We'd have to switch techniques and that would
// be an awful pain. So let's hope that the fixed function value can be inherited here instead.

        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 Ionise_VS();
        pixelShader  = compile ps_2_0 ApplyShadowPS();
       // FillMode = Wireframe;
    }
}

technique MeshTech
{
    pass P0
    {
     //   CullMode = NONE;
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 MeshVS(); // will change this
        pixelShader  = compile ps_2_0 NoShadowPS();
        FillMode = Wireframe;
    }
}
technique SegueTech
{
    pass P0
    {
       // CullMode = NONE; // what happens?
            // It may show artifacts at narrow ridges, but it enables us to see into the cutaway.

        //AlphaBlendEnable = FALSE;
        //DestBlend = INVSRCALPHA;
        //SrcBlend = SRCALPHA;
        vertexShader = compile vs_2_0 SegueVS();
        //pixelShader  = compile ps_2_0 ApplyShadowPSNotCancelled();
        pixelShader  = compile ps_2_0 ApplyShadowPS(); // ClipPS();

        // can't seem to get to point of just taking cross-section
        // going to have to resort to doing in cpp program.
        


        // Overall problem is that when opaque is drawn after, it thinks transparent
        // is in front and therefore hiding it.

        // Cannot be addressed by setting alpha = 0; instead we will have
        // to make a different buffer within the program. 

        // Think this still makes rubbish to left of point -- because 
        // we are getting this bad alpha rubbish that did not come through
        // same vertex shader -- do not understand.

    }
    //pass P1
    //{
    //    // Specify the vertex and pixel shader associated with this pass.
    //    AlphaBlendEnable = TRUE;
    //    DestBlend = INVSRCALPHA;
    //   SrcBlend = SRCALPHA;
    // // Probably want to set depth buffer writing to false, which will enable
    // //the transparent stuff to always be drawn anywhere that it exists?
    //    // including backs of curves
    // 
    // //Not sure what will happen anyway.

    //    vertexShader = compile vs_2_0 SegueVS();
    //    pixelShader  = compile ps_2_0 ApplyShadowPS();

    //    // Strange z-fighting of surface with an alpha=0(?) copy of itself everywhere
    //    // makes transparency difficult.
    //}
}

technique AzSegueTech
{
    pass P0
    {
        //  CullMode = CCW;
        
        // Specify the vertex and pixel shader associated with this pass.
        vertexShader = compile vs_2_0 AzSegueVS();
        pixelShader  = compile ps_2_0 ApplyShadowPS();
    }
}