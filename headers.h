
// Put all the .h includes here.


// use qd as include, not as part of an library, so we need to define QD_API empty
#include "E:/focusfusion/FFxtubes/qd/include/qd/qd_config.h"
#undef QD_API
#define QD_API
#include "E:/focusfusion/FFxtubes/qd/include/qd/dd_real.h"
//#include "qd/qd_real.h"
#include "E:/focusfusion/FFxtubes/qd/include/qd/fpu.h"

//#include "qd/src/qd_const.cpp"
//#include "qd/src/qd_real.cpp"
//#include "qd/src/c_qd.cpp"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <float.h>
#include <malloc.h>
#include <memory.h>
//#include <crtdbg.h> 
#include <initguid.h>
#include <strsafe.h>
#include <io.h> // ?
#include <mmsystem.h>

#include "d3d.h"          
#include "resource.h"

// DirectX and related:
#include <d3dx9.h>
//#include <dinput.h>
#include <dxerr.h>

#include <commdlg.h>    // probably used by avi_utils

#include "globals.h"
#include "constant.h"
#include "surfacegraph_tri.h"
#include "FFxtubes.h"
#include "vector_tensor.cu"
#include "mesh.h"
