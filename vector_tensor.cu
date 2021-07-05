#ifndef VECTOR_TENSOR_H
#define VECTOR_TENSOR_H

#include "FFxtubes.h"
#include <conio.h>
#include <stdio.h>

// will want to do #include type.h
// for #define real, qd_or_d

#define real double
#define __CUDACC__

int const MAX_TRIS_PER_VERTEX = 12;  
#ifdef __CUDACC__

#define QUALIFIERS __host__ __device__ __forceinline__ 
#define QUALS __host__ __device__ inline
// __host__ == cpu
// __global__ == kernel
// __device__ == call from kernel

#else

#define QUALIFIERS inline
#define QUALS   inline

#endif

struct Vector2
{
    double x, y;  
	
	QUALIFIERS Vector2 (){}

	QUALIFIERS Vector2 (double newx,double newy)
	{
		x = newx; y = newy;
	}

	real QUALIFIERS dot(const Vector2 &v) const
	{
		return x*v.x+y*v.y;
	}


	// NVCC will not put up with forward declaration of friend.
	//friend Vector2 operator* (const real h,const Vector2 &v);
	//friend Vector2 operator* (const Vector2 &v,const real h);
	
	// Have to try naked definition in cpp file of prefix multiply instead.
	// But that's no good for MSVS : the definition/declaration will get 
	// parsed 0 or several times. Better declare just outside class instead.
 
	Vector2 QUALIFIERS operator -() {
		return Vector2(-x,-y); 
	}
	
	Vector2 QUALIFIERS operator -(const Vector2 &v) const
	{
		Vector2 result;
		result.x = x - v.x;
		result.y = y - v.y;
		return result;
	}
	Vector2 QUALIFIERS operator +(const Vector2 &v) const
	{
		Vector2 result;
		result.x = x + v.x;
		result.y = y + v.y;
		return result;
	}

	Vector2 QUALIFIERS operator* (const real hh) const
	{
		Vector2 result;
		result.x = hh*x; result.y = hh*y;
		return result;
	}	

	Vector2 QUALIFIERS operator /(const real h) const
	{
		Vector2 result;
		result.x = x/h;
		result.y = y/h;
		return result;
	}
	void QUALIFIERS operator += (const Vector2 &v) 
	{
		x += v.x;
		y += v.y;
	}
	void QUALIFIERS operator -= (const Vector2 &v)
	{
		x -= v.x;
		y -= v.y;
		// Don't think we should try to use return *this in NVCC. 
	}
	void QUALIFIERS operator *= (const real alpha)
	{
		x *= alpha;
		y *= alpha;
	}
	void QUALIFIERS operator /= (const real alpha)
	{
		x /= alpha;
		y /= alpha;
	}
	real QUALIFIERS modulus() const
	{
		return sqrt(x*x+y*y);
	}

	void QUALIFIERS Normalise()
	{
		real r = modulus();
		x /= r; y /= r;
	}
	
	void QUALIFIERS project_to_ins(Vector2 & result) const
	{
		// If it never gets called, hopefully NVCC won't care what is in that.
		real factor = DEVICE_RADIUS_INSULATOR_OUTER/sqrt(x*x+y*y);
		result.x = x*factor; result.y = y*factor;
	}

	void QUALIFIERS project_to_radius
				(Vector2 & result, real radius)
	{
		real factor = radius/sqrt(x*x+y*y);
		result.x = x*factor; result.y = y*factor;
	}
};

Vector2 QUALS operator* (const real hh,const Vector2 &v)
{
	return Vector2(hh*v.x,hh*v.y);
}

// NOTE: For MSVC, declaring members inline means that 
// they have to be defined in the header - and
// that seems to be okay.
// So we probably should do that with everything here.

struct Vector3
{
	real x,y,z;

	QUALIFIERS Vector3() {};
	QUALIFIERS Vector3(real a, real b, real c)
	{
		x = a; y = b; z = c;
	}

	Vector3 QUALIFIERS operator- () const
	{
		return Vector3 (-x,-y,-z);
	}

	Vector3 QUALIFIERS cross(const Vector3 & v) const
	{
		return Vector3(y*v.z-z*v.y,
			           z*v.x-x*v.z,
					   x*v.y-y*v.x);
	}
	
	Vector3 QUALIFIERS operator +(const Vector3 &v) const
	{
		return Vector3(x+v.x,y+v.y,z+v.z);
	}

	Vector3 QUALIFIERS operator -(const Vector3 &v) const
	{
		return Vector3(x-v.x,y-v.y,z-v.z);
	}

	
	Vector3 QUALIFIERS operator* (const real hh) const
	{
		return Vector3(hh*x,hh*y,hh*z);
	}
	Vector3 QUALIFIERS operator/ (const real hh) const 
	{
		return Vector3(x/hh,y/hh,z/hh);
	}
	
	void QUALIFIERS operator +=(const Vector3 &v) 
	{
		x += v.x; y += v.y; z += v.z;
	}
	void QUALIFIERS operator -=(const Vector3 &v) 
	{
		x -= v.x; y -= v.y; z -= v.z;
	}
	void QUALIFIERS operator *=(const real xx) 
	{
		x *= xx; y *= xx; z *= xx;
	}
	void QUALIFIERS operator /=(const real xx) 
	{
		x /= xx; y /= xx; z /= xx;
	}

	bool QUALIFIERS operator != (const Vector3 &v) const 
	{
		return ((v.x != x) || (v.y != y) || (v.z != z));
	}

	real QUALIFIERS dotxy(const Vector3 &v) const
	{
		return x*v.x+y*v.y;
	}
	real QUALIFIERS dotxy(const Vector2 &v) const
	{
		return x*v.x+y*v.y;
	}
	real QUALIFIERS dot(const Vector2 &v) const
	{
		return x*v.x+y*v.y;
	}
	real QUALIFIERS dot(const Vector3 &v) const
	{
		return x*v.x+y*v.y+z*v.z;
	}

	Vector2 QUALIFIERS xypart() const
	{
		Vector2 u;
		u.x = x; 
		u.y = y;
		return u;
	}

	real QUALIFIERS modulusxy()
	{
		return sqrt(x*x+y*y);
	}
	
	real QUALIFIERS modulus()
	{
		return sqrt(x*x+y*y+z*z);
	}

	void Reflect_radially(Vector2 & centre);
	void ZeroRadially(Vector2 & centre);
};

Vector3 QUALS operator* (const real hh,const Vector3 &v)
{
	return Vector3(hh*v.x,hh*v.y,hh*v.z);
}


real QUALS dotxy(const Vector2 & v1, const Vector3 & v2)
{
	return v1.x*v2.x+v1.y*v2.y;
}
real QUALS dotxy(const Vector3 & v1, const Vector2 & v2)
{
	return v1.x*v2.x+v1.y*v2.y;
}

struct Tensor2
{
	real xx, xy, yx, yy;
	QUALIFIERS Tensor2() {}
	QUALIFIERS Tensor2(real x_x, real x_y, real y_x, real y_y)
	{
		xx = x_x; xy = x_y; yx = y_x; yy = y_y;
	}
	QUALIFIERS ~Tensor2() {}
	QUALIFIERS Tensor2 operator +(const Tensor2 &X) const
	{
		return Tensor2(
			xx + X.xx,
			xy + X.xy,
			yx + X.yx,
			yy + X.yy);
	}

	QUALIFIERS Tensor2 operator *(const Tensor2 &X) const
	{
		Tensor2 result;
		// did a test: X is the one on the right.
		result.xx = xx*X.xx + xy*X.yx;
		result.xy = xx*X.xy + xy*X.yy;
		result.yx = yx*X.xx + yy*X.yx;
		result.yy = yx*X.xy + yy*X.yy;
		return result;
	}

	QUALIFIERS Tensor2 operator *(const real hh) const
	{
		return Tensor2 (hh*xx,hh*xy,hh*yx, hh*yy);
	}

	QUALIFIERS Tensor2 operator -(const Tensor2 &X) const
	{
		return Tensor2(
			xx - X.xx, xy-X.xy, yx-X.yx, yy-X.yy
			);
	}
	
	QUALIFIERS void Inverse(Tensor2 & result) const
	{
		real overdet = 1.0/(xx*yy-xy*yx);
		result.xx = yy*overdet;
		result.xy = -xy*overdet;
		result.yx = -yx*overdet;
		result.yy = xx*overdet;
	};

	QUALIFIERS Vector2 operator *(const Vector2 &v) const
	{
		return Vector2(xx*v.x+xy*v.y,yx*v.x+yy*v.y);
	}
	
	QUALIFIERS void operator +=(const Tensor2 &X)
	{
		xx += X.xx; xy += X.xy;
		yx += X.yx; yy += X.yy;
	}

	QUALIFIERS void operator *=(const real hh) 
	{
		xx *= hh; xy *= hh;
		yx *= hh; yy *= hh;
	}
};
QUALIFIERS Tensor2 operator *(const real hh, const Tensor2 &X) 
{
	return Tensor2(hh*X.xx,hh*X.xy,hh*X.yx,hh*X.yy);
}

extern Tensor2 ID2x2;
extern Tensor2 zero2x2;

struct Tensor3
{
	real xx,xy,xz,yx,yy,yz,zx,zy,zz;
	QUALIFIERS Tensor3() {};

	QUALIFIERS Tensor3(real x_x, real x_y, real x_z, 
		          real y_x, real y_y, real y_z,
				  real z_x, real z_y, real z_z) 
	{
		xx = x_x; xy = x_y; xz = x_z; 
		yx = y_x; yy = y_y; yz = y_z; 
		zx = z_x; zy = z_y; zz = z_z;
	}

	QUALIFIERS void MakeCross (const Vector3 om)
	{
		xx = 0.0;
		xy = -om.z;
		xz = om.y;
		yx = om.z;
		yy = 0.0;
		yz = -om.x;
		zx = -om.y;
		zy = om.x;
		zz = 0.0;
	}
	
	QUALIFIERS Tensor3 Inverse()
	{
		Tensor3 result;
		real det =	  xx*(yy*zz-yz*zy)
					+ xy*(zx*yz-yx*zz)
					+ xz*(yx*zy-yy*zx);
		
		// Fill in matrix of minor determinants; 
		// transposed with applied cofactors (signs)
		
		result.xx = yy*zz-yz*zy;
		result.yx = zx*yz-yx*zz; 
		result.zx = yx*zy-yy*zx;
		result.xy = zy*xz-xy*zz;
		result.yy = xx*zz-xz*zx;
		result.zy = zx*xy-xx*zy;
		result.xz = xy*yz-xz*yy;
		result.yz = yx*xz-xx*yz;
		result.zz = xx*yy-yx*xy;

		if (det != 0.0) {
			result = result / det;
		} else {
			printf("\n\nMATRIX INVERSE FAILED. Det==0\n\n\n");
			memset(&result, 0, sizeof(Tensor3));
			result.xx = 1.0; result.yy = 1.0; result.zz = 1.0;
		}
		return result; // inline so return object doesn't matter
	};

	QUALIFIERS void Inverse(Tensor3 & result)
	{
		real det = (xx*(yy*zz - yz*zy)
			+ xy*(zx*yz - yx*zz)
			+ xz*(yx*zy - yy*zx));

		if (det == 0.0) {
			printf("\n\nMATRIX INVERSE FAILED II. Det == 0\n\n\n");
			return;
		}
		real over =	1.0/det;
		
		// Fill in matrix of minor determinants; 
		// transposed with applied cofactors (signs)
		
		result.xx = (yy*zz-yz*zy)*over;
		result.yx = (zx*yz-yx*zz)*over; 
		result.zx = (yx*zy-yy*zx)*over;
		result.xy = (zy*xz-xy*zz)*over;
		result.yy = (xx*zz-xz*zx)*over;
		result.zy = (zx*xy-xx*zy)*over;
		result.xz = (xy*yz-xz*yy)*over;
		result.yz = (yx*xz-xx*yz)*over;
		result.zz = (xx*yy-yx*xy)*over;

		//return result; // inline so return object doesn't matter
	};

QUALIFIERS Tensor2 xy2x2part () const
	{
		Tensor2 res;
		res.xx = xx;
		res.xy = xy;
		res.yx = yx;
		res.yy = yy;
		return res;
	}

QUALIFIERS Tensor3 operator- () const
	{
		Tensor3 res;
		res.xx = -xx; res.xy = -xy; res.xz = -xz;
		res.yx = -yx; res.yy = -yy; res.yz = -yz;
		res.zx = -zx; res.zy = -zy; res.zz = -zz;
		return res;
	}

QUALIFIERS Vector3 operator* (const Vector3 &v) const
	{
		Vector3 res;
		res.x = xx*v.x + xy*v.y + xz*v.z;
		res.y = yx*v.x + yy*v.y + yz*v.z;
		res.z = zx*v.x + zy*v.y + zz*v.z;
		return res;
	}
	
	
QUALIFIERS Tensor3 operator* (const real hh) const
	{ 
		return Tensor3(
			hh*xx, hh*xy, hh*xz,
			hh*yx, hh*yy, hh*yz,
			hh*zx, hh*zy, hh*zz);
	};
		
QUALIFIERS Tensor3 operator/ (const real r) const
	{
		Tensor3 result;
		// did a test: X is the one on the right.
		result.xx = xx/r;
		result.xy = xy/r;
		result.xz = xz/r;
		result.yx = yx/r;
		result.yy = yy/r;
		result.yz = yz/r;
		result.zx = zx/r;
		result.zy = zy/r;
		result.zz = zz/r;
		return result;
	}

QUALIFIERS Tensor3 operator +(const Tensor3 &v) const
	{
		Tensor3 result;
		result.xx = xx + v.xx;
		result.xy = xy + v.xy;
		result.xz = xz + v.xz;
		result.yx = yx + v.yx;
		result.yy = yy + v.yy;
		result.yz = yz + v.yz;
		result.zx = zx + v.zx;
		result.zy = zy + v.zy;
		result.zz = zz + v.zz;
		return result;
	}

QUALIFIERS Tensor3 operator -(const Tensor3 &v) const
	{
		Tensor3 result;
		result.xx = xx - v.xx;
		result.xy = xy - v.xy;
		result.xz = xz - v.xz;
		result.yx = yx - v.yx;
		result.yy = yy - v.yy;
		result.yz = yz - v.yz;
		result.zx = zx - v.zx;
		result.zy = zy - v.zy;
		result.zz = zz - v.zz;
		return result;
	}
	
QUALIFIERS Tensor3 operator *(const Tensor3 &X) const
	{
		Tensor3 result;
		result.xx = xx*X.xx + xy*X.yx + xz*X.zx;
		result.xy = xx*X.xy + xy*X.yy + xz*X.zy;
		result.xz = xx*X.xz + xy*X.yz + xz*X.zz;
		result.yx = yx*X.xx + yy*X.yx + yz*X.zx;
		result.yy = yx*X.xy + yy*X.yy + yz*X.zy;
		result.yz = yx*X.xz + yy*X.yz + yz*X.zz;
		result.zx = zx*X.xx + zy*X.yx + zz*X.zx;
		result.zy = zx*X.xy + zy*X.yy + zz*X.zy;
		result.zz = zx*X.xz + zy*X.yz + zz*X.zz;
		return result;
	}
	
QUALIFIERS Tensor3 operator +=(const Tensor3 &X) 
	{
		xx += X.xx;
		xy += X.xy;
		xz += X.xz;
		yx += X.yx;
		yy += X.yy;
		yz += X.yz;
		zx += X.zx;
		zy += X.zy;
		zz += X.zz;
		return *this;
	}
	QUALIFIERS Tensor3 operator -=(const Tensor3 &X) 
	{
		xx -= X.xx;
		xy -= X.xy;
		xz -= X.xz;
		yx -= X.yx;
		yy -= X.yy;
		yz -= X.yz;
		zx -= X.zx;
		zy -= X.zy;
		zz -= X.zz;
		return *this;
	}
	
	void Make3DRotationAboutAxis(Vector3 w, real t);
	void spitout(void);
};
QUALIFIERS Tensor3 operator* (const real hh,const Tensor3 &X)
	{		
		Tensor3 result;
		result.xx = hh*X.xx;
		result.xy = hh*X.xy;
		result.xz = hh*X.xz;
		result.yx = hh*X.yx;
		result.yy = hh*X.yy;
		result.yz = hh*X.yz;
		result.zx = hh*X.zx;
		result.zy = hh*X.zy;
		result.zz = hh*X.zz;
		return result;
	}

// Not clear to me : do we want the following for NVCC to be here?
// It actually makes sense to keep "matrix" here!

struct Matrix3
{
	real a[3][3];

	QUALIFIERS void Inverse(Matrix3 & result)
	{
		// find+replace on the above

		real det =	  a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
					+ a[0][1]*(a[2][0]*a[1][2]-a[1][0]*a[2][2])
					+ a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);

		// Fill in matrix of minor determinants; 
		// transposed with applied cofactors (signs)
	
		result.a[0][0] = a[1][1]*a[2][2]-a[1][2]*a[2][1];
		result.a[1][0] = a[2][0]*a[1][2]-a[1][0]*a[2][2]; 
		result.a[2][0] = a[1][0]*a[2][1]-a[1][1]*a[2][0];
		result.a[0][1] = a[2][1]*a[0][2]-a[0][1]*a[2][2];
		result.a[1][1] = a[0][0]*a[2][2]-a[0][2]*a[2][0];
		result.a[2][1] = a[2][0]*a[0][1]-a[0][0]*a[2][1];
		result.a[0][2] = a[0][1]*a[1][2]-a[0][2]*a[1][1];
		result.a[1][2] = a[1][0]*a[0][2]-a[0][0]*a[1][2];
		result.a[2][2] = a[0][0]*a[1][1]-a[1][0]*a[0][1];

		//real * ptr = (real *)(result.a);
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			result.a[i][j] /= det; // 99% sure static array elems are contiguous but hey.
		}
	
	};

	QUALIFIERS void multiply(real RHS[3], real output[3])
	{
		output[0] = a[0][0]*RHS[0] + a[0][1]*RHS[1] + a[0][2]*RHS[2];
		output[1] = a[1][0]*RHS[0] + a[1][1]*RHS[1] + a[1][2]*RHS[2];
		output[2] = a[2][0]*RHS[0] + a[2][1]*RHS[1] + a[2][2]*RHS[2];
	};
	
};

extern Tensor3 ID3x3;
extern Tensor3 zero3x3;

struct f64_tens3mag {
	real bx, by, bz, Px, Py, Pz, Hx, Hy, Hz;
};
struct f64_vec3mag {
	real b, P, H;
};
Vector3 QUALS Make3(const Vector2 & v, const real scalar)
{
	Vector3 result;
	result.x = v.x;
	result.y = v.y;
	result.z = scalar;
	return result;
};

// Never used? :
struct Symmetric3
{
	real xx,yy,zz,xy,xz,yz;
	QUALIFIERS Symmetric3() {};
	QUALIFIERS Symmetric3(real x_x, real x_y, real y_y, real x_z, real y_z, real z_z) ;
	
	Vector3 QUALIFIERS operator* (const Vector3 &v) const;
};


// Never used? :
struct Symmetric2
{
	real xx,yy,xy;
};


#define f64 real
#define f64_vec2 Vector2
#define f64_vec3 Vector3
#define f64_tens2 Tensor2
#define f64_tens3 Tensor3
//#define u32 unsigned long


//struct vertinfo
//{
//	long flag;
//	long numTris;
//	f64_vec2 pos; 
//	long iTriIndex[MAX_TRIS_PER_VERTEX]; // 10 x 8
//};
////
//struct structural
//{
//	u32 u32corner[3];                    
//	u32 u32neigh[3];                     // 8x3
//	int iDomain_flag, iPeriodic;       
//	f64_vec2 edge_normal[3];             // 8x6
//	f64_vec2 gradT;                        
//	f64 weight[3]; // weights used for averaging at corners. :/ from CPU
//	f64_vec2 pos; // centroid
//	f64 area;                            // 8x5+8x3
//	// size ~ 24 + 8 + 48 + 40 + 24 
//	// 144 bytes or so
//	f64_vec2 coeff[3]; // for each triangle, the coefficient by which nT_cell creates pressure at vertex.
//	
//	// Demoralised from having to do vertex aggregation so let's bung this in here.
//	// In the next version we can get rid of it.
//};

struct species_f64
{
	f64 neut, ion, elec;
};

struct species_vec2
{
	f64_vec2 neut, ion, elec;
};

struct species_vec3
{
	f64_vec3 neut, ion, elec;
};

struct f64_vec4
{
	f64 x[4];
};
#endif


#undef QUALIFIERS