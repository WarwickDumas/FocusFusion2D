
// Version 0.51

// Been over some attempts to ameliorate local accesses -- not v successful basically.
// Correction in "Get Lap phi" routine.

// Version 0.52:

// Change Lap A, Grad A routines to load CHAR4 p_tri_per_neigh instead of loading data
// to interrogate neighbour periodic status.

// Change major area calc in the INNERMOST/OUTERMOST case.

// Note that central area calc does not look right.

// Version 0.53:

// Made changes to Reladvect_nT because it was taking wrong connection for OUTERMOST.
// Changed flag tests & treatment of Inner verts in preceding routines.

// Version 0.54:

// Adjusted area calculations as written in spec. 
// We set ins crossing tri minor area = 0, centroid on ins;
// frill area = 0, centroid on boundary.

// Version 0.6:

// Debugging and making corrections.


// ==
// Version 0.7:

// Debugging ... there is a kernel launch failure for Antiadvect Adot

// Version 0.8: 

// Change to set phi at innermost and outermost instead of trying to handle
// insulator special conditions.
// Am I sure? yes.




// PLAN: 
// Allow that on GPU we can move outside domain and it's fine, we do not change PB data.
// PB data will be only changed on CPU.
// Nonetheless we kept PBCTri lists which can be updated, unlike has_periodic alone, in case
// of moving something to its image within the domain.



// NOTES:

		// Ensure that outside the domain, n_major is recorded as 0

		// Ensure that outside the domain, resistive_heat is recorded as 0


extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;



__global__ void Kernel_CalculateTriMinorAreas_AndCentroids
			(structural * __restrict__ p_info_sharing, // for vertex positions
			 LONG3 * __restrict__ p_tri_corner_index,
			 CHAR4 * __restrict__ p_tri_perinfo,
			 // Output:
			 f64 * __restrict__ p_area_minor,
			 f64_vec2 * __restrict__ p_tri_centroid)
{
	__shared__ f64_vec2 shared_vertex_pos[SIZE_OF_MAJOR_PER_TRI_TILE];
	
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long tid =  threadIdx.x + blockIdx.x * blockDim.x;
	
	// Note that we only do a fetch with the first half of threads:
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		structural info = p_info_sharing[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];

		// Well here is a major problem.
		// mti.StartMajor is a separate value for every thread.
		// How can we make it do a contiguous access?
		
		// Suppose a 1:1 correspondence between minor blocks and major blocks...
		// that is ONE way.
		shared_vertex_pos[threadIdx.x] = info.pos;
//		shared_shorts[threadIdx.x].flag = info.flag;
//		shared_shorts[threadIdx.x].neigh_len = info.neigh_len;
		// these were never used.
	};
	// If we make an extended array then we can always go through that code.
	__syncthreads();

	// Triangle area * 2/3 is area of minor cell.

	// if (tid < Ntris) { // redundant test if we do it right
		
	LONG3 corner_index = p_tri_corner_index[tid];
	CHAR4 perinfo = p_tri_perinfo[tid];
	// Do we ever require those and not the neighbours?
	// Yes - this time for instance.
	f64_vec2 pos1, pos2, pos3;

	if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
	{
		pos1 = shared_vertex_pos[corner_index.i1-StartMajor];
	} else {
		// have to load in from global memory:
		structural info = p_info_sharing[corner_index.i1];
		pos1 = info.pos;
	}
	if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
	{
		pos2 = shared_vertex_pos[corner_index.i2-StartMajor];
	} else {
		// have to load in from global memory:
		structural info = p_info_sharing[corner_index.i2];
		pos2 = info.pos;
	}
	if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
	{
		pos3 = shared_vertex_pos[corner_index.i3-StartMajor];
	} else {
		// have to load in from global memory:
		structural info = p_info_sharing[corner_index.i3];
		pos3 = info.pos;
	}
	
	if (perinfo.per0+perinfo.per1+perinfo.per2 == 0) {
	} else {
		// In this case which ones are periodic?
		// Should we just store per flags?
		// How it should work:
		// CHAR4 perinfo: periodic, per0, per1, per2;
		if (perinfo.per0 == NEEDS_ANTI) 
			pos1 = Anticlock_rotate2(pos1);
		if (perinfo.per0 == NEEDS_CLOCK)
			pos1 = Clockwise_rotate2(pos1);
		if (perinfo.per1 == NEEDS_ANTI)
			pos2 = Anticlock_rotate2(pos2);
		if (perinfo.per1 == NEEDS_CLOCK)
			pos2 = Clockwise_rotate2(pos2);
		if (perinfo.per2 == NEEDS_ANTI)
			pos3 = Anticlock_rotate2(pos3);
		if (perinfo.per2 == NEEDS_CLOCK)
			pos3 = Clockwise_rotate2(pos3);
	};
	
	// Now we've got to decide what to do about minor cells near the edges.
	
	// Edge of memory: triangles should not continue to the edge. 
	// Ultimately the edge of memory will be mostly within cathode rods and suchlike things.
	// So we don't need to connect tri mesh beyond outermost row of vertices, even if we could.
	
	// Realise that this edge cell crosses into the insulator and so should be assigned nv_r = 0
	
	// We do not know what order the corners are given in.
	// So use fabs:
	f64 area = fabs(0.5*(   (pos2.x+pos1.x)*(pos2.y-pos1.y)
						 + (pos3.x+pos2.x)*(pos3.y-pos2.y)
						 + (pos1.x+pos3.x)*(pos1.y-pos3.y)
					)	);
	f64_vec2 centroid = THIRD*(pos1+pos2+pos3);
	if (area > 1.0e-3) {
		printf("tri %d area %1.3E pos_x %1.6E %1.6E %1.6E \n"
			   "                    pos_y %1.6E %1.6E %1.6E \n",tid,area,
			pos1.x,pos2.x,pos3.x,
			pos1.y,pos2.y,pos3.y);
	}
	if (perinfo.flag == OUTER_FRILL) 
	{
		f64_vec2 temp = 0.5*(pos1+pos2); 
		temp.project_to_radius(centroid, FRILL_CENTROID_OUTER_RADIUS_d);
		area = 1.0e-14; // == 0 but tiny is less likely to cause 1/0
	//	printf("%d %1.5E %1.5E .. %1.5E %1.5E \n",tid,temp.x,temp.y,
	//		centroid.x,centroid.y);
	}
	
	if (perinfo.flag == INNER_FRILL)
	{
		f64_vec2 temp = 0.5*(pos1+pos2); 
		temp.project_to_radius(centroid, FRILL_CENTROID_INNER_RADIUS_d);
		area = 1.0e-14; // == 0 but tiny is less likely to cause 1/0
	}
	
	if (perinfo.flag == CROSSING_INS) {
		
		f64_vec2 centroid2;
		centroid.project_to_ins(centroid2);
		centroid = centroid2;
		// The major cells will abut the insulator.
		
		// Only count the % of the area that is in the domain.
		//bool b1, b2, b3;
		//b1 = (pos1.x*pos1.x+pos1.y*pos1.y > INSULATOR_OUTER_RADIUS*INSULATOR_OUTER_RADIUS);
		//b2 = (pos2.x*pos2.x+pos2.y*pos2.y > INSULATOR_OUTER_RADIUS*INSULATOR_OUTER_RADIUS);
		//b3 = (pos3.x*pos3.x+pos3.y*pos3.y > INSULATOR_OUTER_RADIUS*INSULATOR_OUTER_RADIUS);
		
		// Save ourselves some bother for now by setting area to be near 0.
	//	area = 1.0e-14;
		// FOR NOW, legislate v = 0 in insulator-crossing tris.
		// And so avoid having to do an awkward area calculation.

		// Stick with correct area for tri as area variable.
		// Possibly we never use 'area' except for domain-related matters; if that can be
		// verified, then it's best to change to 'domain_intersection_area', however tricky.
	}
	
	p_tri_centroid[tid] = centroid;
	p_area_minor[tid] = 0.666666666666667*area;  
	if (p_area_minor[tid] < 0.0) {
		printf("kernel -- tid %d flag %d area %1.8E \n",tid,perinfo.flag,area);
	};
	
	// Perhaps we need instead to read data from neighbours to create tri minor area.
	
	// Note that we subsequently CHANGED the nodes of minor mesh to be at averages
	// so that we could average neatly A to edges. However, this means TWOTHIRDS*tri area
	// is not an exact estimate.	
}

// FOR OUTERMOST,
// 
//    |  4  \/  3   |
//  pt0|  -------  |pt3
//       0      2
//   pt1|   1    |pt2

// If we have an outer point,
// then the number of neighs is not the number of tris;
// SO EXPLOIT THIS
// Make sure that the omitted edge is the one that would go between the frill tris.

// This has to go into the reconstructing code that will
// generate the mesh with frill tris.

// ---------------------------------------------------------------------------------

// We'll want to calculate areas for triangles AND for central cells.
// But they require different codes so might as well be different kernels.
// Central area = sum of 1/6 of each neighbouring tri minor area.
			 
// So now let's write central area calc routine:
// We should be passed the pointer to the start of the central minor area array.

__global__ void Kernel_CalculateCentralMinorAreas (
			structural * __restrict__ p_info_sharing, 			 
			long * __restrict__ p_IndexTri,
			f64 * __restrict__ p_triminor_area,			 
			 // Output:
			f64 * __restrict__ p_area_minor
			 // pass output array starting from the central array start
			 )
{
	__shared__ f64 shared_area[SIZE_OF_TRI_TILE_FOR_MAJOR];
	__shared__ long Indextri[MAXNEIGH_d*threadsPerTileMajor];
	// 2*8+12*4 = 64 bytes => room for 768 threads in 48K 
	
	long index = threadIdx.x + blockIdx.x * blockDim.x;
	// Load in minor data: how to manage this? fill in with 2 strides; rely on contiguity.
	long StartMinor = blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR; // Have to do this way.
	// will this be recognised as contiguous access?
	shared_area[threadIdx.x] = 
		p_triminor_area[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
	shared_area[blockDim.x + threadIdx.x] = 
		p_triminor_area[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + blockDim.x + threadIdx.x];
	// Loaded in 2 times as many areas as central cells
	__syncthreads();
	
	//if (index < Nverts) 
	{			
		structural info = p_info_sharing[index];
		memcpy(Indextri + MAXNEIGH_d*threadIdx.x, 
			   p_IndexTri + MAXNEIGH_d*index,
			   MAXNEIGH_d*sizeof(long)); // MAXNEIGH_d should be chosen to be 12, for 1 full bus.
		f64 sum = 0.0;
#pragma unroll 12
		for (short iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			long indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if ((indextri < StartMinor) || (indextri >= StartMinor+SIZE_OF_TRI_TILE_FOR_MAJOR))
			{
				sum += p_triminor_area[indextri];
			} else {
				sum += shared_area[indextri-StartMinor];
			}
		}
		// place separation of central from edge cell at 1/3 along line.
		// Then have 1/9 area in central shard, 2/3 in edge minor,
		// so (1/9)/(2/3) = 1/6
		p_area_minor[index] = sum*SIXTH;
		if (sum < 0.0) {
			printf("kerncentral -- tid %d area %1.2E \n",index,p_area_minor[index]);
		};
	};
	
	// This may give funny results at the edges of memory, where we have added
	// areas only of shards that formed part of triangles. But that is the expected
	// behaviour here.
	
	// If we had frills with repeated corners, then we get area 0 from them.
	// If we used a special vertex then we had to do something special in setting area - perhaps we want it to = 0.

	// BUG:

	// This 1/6 only holds as long as we position the minor joins on the lines
	// between vertices. If we use (1/3)(vertex + centroid 1 + centroid 2)
	// then we should not be doing this area sum. Rather, given each pair of 
	// triangles, we can infer the area of the triangle that is part of central cell.
}



__global__ void Kernel_CalculateMajorAreas (
			structural * __restrict__ p_info,
			f64_vec2 * __restrict__ p_tri_centroid,
			long * __restrict__ pIndexTri,
			char * __restrict__ pPBCtri,
			// Output:
			f64 * __restrict__ p_area			
			)
{	
	__shared__ f64_vec2 shared_centroids[SIZE_OF_TRI_TILE_FOR_MAJOR];	
	__shared__ long Indextri[MAXNEIGH_d*threadsPerTileMajor];
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajor];
	
	// Major areas are used for rel advect (comp htg), ionisation, heating.


	long StartMinor = blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR; // Have to do this way.
	
	shared_centroids[threadIdx.x] = 
		p_tri_centroid[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
	shared_centroids[blockDim.x + threadIdx.x] = 
		p_tri_centroid[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + blockDim.x + threadIdx.x];	
	//
	//if (shared_centroids[threadIdx.x].x*shared_centroids[threadIdx.x].x
	//	+ shared_centroids[threadIdx.x].y*shared_centroids[threadIdx.x].y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
	//	shared_centroids[threadIdx.x].project_to_radius(shared_centroids[threadIdx.x],DOMAIN_OUTER_RADIUS);
	//
	//if (shared_centroids[threadIdx.x].x*shared_centroids[threadIdx.x].x
	//	+ shared_centroids[threadIdx.x].y*shared_centroids[threadIdx.x].y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
	//	shared_centroids[threadIdx.x].project_to_radius(shared_centroids[threadIdx.x],INNER_A_BOUNDARY);
	//
	//if (shared_centroids[blockDim.x + threadIdx.x].x*shared_centroids[blockDim.x + threadIdx.x].x
	//	+ shared_centroids[blockDim.x + threadIdx.x].y*shared_centroids[blockDim.x + threadIdx.x].y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
	//	shared_centroids[blockDim.x + threadIdx.x].project_to_radius(shared_centroids[blockDim.x + threadIdx.x],DOMAIN_OUTER_RADIUS);
	//
	//if (shared_centroids[blockDim.x + threadIdx.x].x*shared_centroids[blockDim.x + threadIdx.x].x
	//	+ shared_centroids[blockDim.x + threadIdx.x].y*shared_centroids[blockDim.x + threadIdx.x].y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
	//	shared_centroids[blockDim.x + threadIdx.x].project_to_radius(shared_centroids[blockDim.x + threadIdx.x],INNER_A_BOUNDARY);
	//

	__syncthreads();
	
	long index = threadIdx.x + blockIdx.x * blockDim.x;
	f64_vec2 uprev, unext;
	
	//if (index < Nverts) { // redundant test, should be
	structural info = p_info[index];
	memcpy(Indextri + MAXNEIGH_d*threadIdx.x, 
			pIndexTri + MAXNEIGH_d*index,
			MAXNEIGH_d*sizeof(long)); // MAXNEIGH_d should be chosen to be 12, for 1 full bus.
	memcpy(PBCtri + MAXNEIGH_d*threadIdx.x, 
			pPBCtri + MAXNEIGH_d*index,
			MAXNEIGH_d*sizeof(char)); // MAXNEIGH_d should be chosen to be 12, for 1 full bus.
	
	f64 grad_x_integrated_x = 0.0;
	
	// Going to do shoelace on tri centroids which must be sorted anticlockwise.
	
	// If we have a frilled e.g.OUTERMOST vertex, we shall find that
	// info.neigh_len = 4 whereas tri_len = 5. Bear in mind...
	
	if ((info.flag != OUTERMOST) && (info.flag != INNERMOST))
	{
		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + info.neigh_len-1];
		if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
			uprev = shared_centroids[indextri-StartMinor];
		} else {
			uprev = p_tri_centroid[indextri];
		}
		char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + info.neigh_len-1];
		if (PBC == NEEDS_CLOCK) {
			uprev = Clockwise_rotate2(uprev);
		}
		if (PBC == NEEDS_ANTI) {
			uprev = Anticlock_rotate2(uprev);
		}	
		short iNeigh;
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++) // iNeigh is the anticlockwise one
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
				unext = shared_centroids[indextri-StartMinor];
			} else {
				unext = p_tri_centroid[indextri];
			}
			char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + iNeigh];
			if (PBC == NEEDS_CLOCK) {
				unext = Clockwise_rotate2(unext);
			}
			if (PBC == NEEDS_ANTI) {
				unext = Anticlock_rotate2(unext);
			}	
			
			// Get edge_normal.x and average x on edge
			grad_x_integrated_x += //0.5*(unext.x+uprev.x)*edge_normal.x
									0.5*(unext.x+uprev.x)*(unext.y-uprev.y);
			uprev = unext;
		};
	} else {
		// FOR THE OUTERMOST / INNERMOST CELLS :
		// In this case we basically substituted tri_len for neigh_len:

		// Also we project frill centroid on to the inner/outer radius.
		

		// AFAICS this is NOT USED ANYWAY -- 


		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + info.neigh_len];
		if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
			uprev = shared_centroids[indextri-StartMinor];
		} else {
			uprev = p_tri_centroid[indextri];
		}
		if (uprev.x*uprev.x + uprev.y*uprev.y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
			uprev.project_to_radius(uprev,DOMAIN_OUTER_RADIUS);
		if (uprev.x*uprev.x + uprev.y*uprev.y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
			uprev.project_to_radius(uprev,INNER_A_BOUNDARY);
		char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + info.neigh_len];
		if (PBC == NEEDS_CLOCK) uprev = Clockwise_rotate2(uprev);
		if (PBC == NEEDS_ANTI) uprev = Anticlock_rotate2(uprev);
			
		short iNeigh;
		for (iNeigh = 0; iNeigh < info.neigh_len+1; iNeigh++) // iNeigh is the anticlockwise one
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
				unext = shared_centroids[indextri-StartMinor];
			} else {
				unext = p_tri_centroid[indextri];
			}			
			if (unext.x*unext.x + unext.y*unext.y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
				unext.project_to_radius(unext,DOMAIN_OUTER_RADIUS);
			if (unext.x*unext.x + unext.y*unext.y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
				unext.project_to_radius(unext,INNER_A_BOUNDARY);
			
			char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + iNeigh];
			if (PBC == NEEDS_CLOCK) unext = Clockwise_rotate2(unext);
			if (PBC == NEEDS_ANTI) unext = Anticlock_rotate2(unext);
						
			grad_x_integrated_x += 0.5*(unext.x+uprev.x)*(unext.y-uprev.y);
			
			// We do have to even count the edge looking into frills, or polygon
			// area would not be right.			
			uprev = unext;
		};
	};
	
	p_area[index] = grad_x_integrated_x;
	if ((index == 36685)) {
		printf("index %d flag %d area %1.3E \n",
			index, info.flag, grad_x_integrated_x);
		
		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + info.neigh_len];
		if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
			uprev = shared_centroids[indextri-StartMinor];
		} else {
			uprev = p_tri_centroid[indextri];
		}
		if (uprev.x*uprev.x + uprev.y*uprev.y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
		{
			// This is not what happens. centroid is #IND I think.
			printf("uprev before %1.5E %1.5E \nDOR %1.5E mod %1.5E\n",uprev.x,uprev.y,
				DOMAIN_OUTER_RADIUS,uprev.modulus());
			uprev.project_to_radius(uprev,DOMAIN_OUTER_RADIUS);
			printf("uprev after %1.6E %1.6E \n",uprev.x,uprev.y);
		};
		if (uprev.x*uprev.x + uprev.y*uprev.y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
			uprev.project_to_radius(uprev,INNER_A_BOUNDARY);
		char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + info.neigh_len];
		if (PBC == NEEDS_CLOCK) uprev = Clockwise_rotate2(uprev);
		if (PBC == NEEDS_ANTI) uprev = Anticlock_rotate2(uprev);
			
		printf("uprev %1.5E %1.5E ... %1.5E\n",uprev.x,uprev.y,uprev.modulus());
		short iNeigh;
		for (iNeigh = 0; iNeigh < info.neigh_len+1; iNeigh++) // iNeigh is the anticlockwise one
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh];
			if ((indextri >= StartMinor) && (indextri < StartMinor + SIZE_OF_TRI_TILE_FOR_MAJOR)) {
				unext = shared_centroids[indextri-StartMinor];
			} else {
				unext = p_tri_centroid[indextri];
			}			
			if (unext.x*unext.x + unext.y*unext.y > DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS)
				unext.project_to_radius(unext,DOMAIN_OUTER_RADIUS);
			if (unext.x*unext.x + unext.y*unext.y < INNER_A_BOUNDARY*INNER_A_BOUNDARY)
				unext.project_to_radius(unext,INNER_A_BOUNDARY);
			
			char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + iNeigh];
			if (PBC == NEEDS_CLOCK) unext = Clockwise_rotate2(unext);
			if (PBC == NEEDS_ANTI) unext = Anticlock_rotate2(unext);
						
			grad_x_integrated_x += 0.5*(unext.x+uprev.x)*(unext.y-uprev.y);
			
			printf("unext %1.5E %1.5E ... %1.5E area %1.5E \n",unext.x,unext.y,unext.modulus(),
				grad_x_integrated_x);
			
			// We do have to even count the edge looking into frills, or polygon
			// area would not be right.			
			uprev = unext;
			
			
		};
		
	};
}


__global__ void Kernel_Average_nT_to_tri_minors (
			LONG3 * __restrict__ p_tri_corner_index,
			CHAR4 * __restrict__ p_tri_perinfo,
			nT * __restrict__ p_nT_neut,
			nT * __restrict__ p_nT_ion,
			nT * __restrict__ p_nT_elec,
			 // Output:
			nT * __restrict__ p_minor_nT_neut,
			nT * __restrict__ p_minor_nT_ion,
			nT * __restrict__ p_minor_nT_elec
			)
{
	
	// Average by area so that we get the same total mass on minor mesh as on major.
	// We have to know intersection. It's not always 1/3 of triangle is it.
	// ??	
	// Even corner positions do not tell us intersection. We'd have to know the neighbouring
	// centroid also.

    __shared__ nT shared_nT[SIZE_OF_MAJOR_PER_TRI_TILE];
    
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		shared_nT[threadIdx.x] = p_nT_neut[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];	 
	}	
	__syncthreads();

	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	nT nT1, nT2, nT3, nT_out;		
	LONG3 corner_index;
	CHAR4 per_info = p_tri_perinfo[tid];

	corner_index = p_tri_corner_index[tid];
	// Do we ever require those and not the neighbours? Yes - this time for instance.
	
	if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
	{
		nT1 = shared_nT[corner_index.i1-StartMajor];
	} else {
		// have to load in from global memory:
		nT1 = p_nT_neut[corner_index.i1];
	}
	if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
	{
		nT2 = shared_nT[corner_index.i2-StartMajor];
	} else {
		nT2 = p_nT_neut[corner_index.i2];
	}
	if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
	{
		nT3 = shared_nT[corner_index.i3-StartMajor];
	} else {
		nT3 = p_nT_neut[corner_index.i3];
	}
	if (per_info.flag == CROSSING_INS) {
		// An idea: Ensure that outside the domain, n is recorded as 0
		int divide = 0.0;
		nT_out.n = 0.0;
		nT_out.T = 0.0;
		if (nT1.n > 0.0) {
			nT_out.n += nT1.n;
			nT_out.T += nT1.T;
			divide++;
		}
		if (nT2.n > 0.0) {
			nT_out.n += nT2.n;
			nT_out.T += nT2.T;
			divide++;
		}
		if (nT3.n > 0.0) {
			nT_out.n += nT3.n;
			nT_out.T += nT3.T;
			divide++;
		}
		nT_out.n /= (real)divide;
		nT_out.T /= (real)divide;
	} else {

		nT_out.n = THIRD*(nT1.n+nT2.n+nT3.n);
		nT_out.T = THIRD*(nT1.T+nT2.T+nT3.T);
	};
	// SO THIS IS JUST ROUGH FOR NOW? What we wanted to do:
	// Sum (Area_intersection * nT) / Sum(Area_intersection)
	// You cannot get the intersection area just from knowing the corner positions.
	
	// But do note that since centroid = (1/3)(sum of positions), (1/3) represents linear interpolation on a plane.
	p_minor_nT_neut[tid] = nT_out;
	__syncthreads();	
	
	// Now repeat same thing for each species	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		shared_nT[threadIdx.x] = p_nT_ion[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];	 
	}	
	__syncthreads();
	
	//if (tid < Ntris) {		
	if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
	{
		nT1 = shared_nT[corner_index.i1-StartMajor];
	} else {
		// have to load in from global memory:		
		nT1 = p_nT_ion[corner_index.i1];
	}
	if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
	{
		nT2 = shared_nT[corner_index.i2-StartMajor];
	} else {
		nT2 = p_nT_ion[corner_index.i2];
	}
	if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
	{
		nT3 = shared_nT[corner_index.i3-StartMajor];
	} else {
		nT3 = p_nT_ion[corner_index.i3];
	}
	if (per_info.flag == CROSSING_INS) {
		// An idea: Ensure that outside the domain, n is recorded as 0
		int divide = 0.0;
		nT_out.n = 0.0;
		nT_out.T = 0.0;
		if (nT1.n > 0.0) {
			nT_out.n += nT1.n;
			nT_out.T += nT1.T;
			divide++;
		}
		if (nT2.n > 0.0) {
			nT_out.n += nT2.n;
			nT_out.T += nT2.T;
			divide++;
		}
		if (nT3.n > 0.0) {
			nT_out.n += nT3.n;
			nT_out.T += nT3.T;
			divide++;
		}
		nT_out.n /= (real)divide;
		nT_out.T /= (real)divide;
	} else {

		nT_out.n = THIRD*(nT1.n+nT2.n+nT3.n);
		nT_out.T = THIRD*(nT1.T+nT2.T+nT3.T);
	};
	p_minor_nT_ion[tid] = nT_out;
	//};

	__syncthreads();

	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		shared_nT[threadIdx.x] = p_nT_elec[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];	 
	}	
	__syncthreads();

	//if (tid < Ntris) {		
	if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
	{
		nT1 = shared_nT[corner_index.i1-StartMajor];
	} else {
		// have to load in from global memory:
		nT1 = p_nT_elec[corner_index.i1];
	}
	if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
	{
		nT2 = shared_nT[corner_index.i2-StartMajor];
	} else {
		// have to load in from global memory:
		nT2 = p_nT_elec[corner_index.i2];
	}
	if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
	{
		nT3 = shared_nT[corner_index.i3-StartMajor];
	} else {
		// have to load in from global memory:
		nT3 = p_nT_elec[corner_index.i3];
	}
	if (per_info.flag == CROSSING_INS) {
		// An idea: Ensure that outside the domain, n is recorded as 0
		int divide = 0.0;
		nT_out.n = 0.0;
		nT_out.T = 0.0;
		if (nT1.n > 0.0) {
			nT_out.n += nT1.n;
			nT_out.T += nT1.T;
			divide++;
		}
		if (nT2.n > 0.0) {
			nT_out.n += nT2.n;
			nT_out.T += nT2.T;
			divide++;
		}
		if (nT3.n > 0.0) {
			nT_out.n += nT3.n;
			nT_out.T += nT3.T;
			divide++;
		}
		nT_out.n /= (real)divide;
		nT_out.T /= (real)divide;
	} else {

		nT_out.n = THIRD*(nT1.n+nT2.n+nT3.n);
		nT_out.T = THIRD*(nT1.T+nT2.T+nT3.T);
	};
	p_minor_nT_elec[tid] = nT_out;
	 
	// if frills have corners repeated, we end up with 1/3+2/3 --- should never matter.
	// If special vertex, probably we set nT at special vertex to 0 so 1/3+1/3.
	// nT should not be important at frills, as outermost points and innermost points
	// do not need to know pressure.
}


__global__ void Kernel_GetZCurrent(
		    CHAR4 * __restrict__ p_minor_info,
			nT * __restrict__ p_minor_nT_ion,
			nT * __restrict__ p_minor_nT_elec,
			f64_vec3 * __restrict__ p_minor_v_ion,
			f64_vec3 * __restrict__ p_minor_v_elec, // Not clear if this should be nv or {n,v} ? {n,v}
			f64 * __restrict__ p_area_minor,
			f64 * __restrict__ p_summands	)
{
	__shared__ f64 intrablock[threadsPerTileMinor];
	long tid =  threadIdx.x + blockIdx.x * blockDim.x;
	CHAR4 minor_info = p_minor_info[tid];
	// This is called for all minor cells. 
	if ((minor_info.flag == DOMAIN_MINOR) || (minor_info.flag == OUTERMOST)) {
	
		// Let DOMAIN_MINOR == DOMAIN_TRIANGLE ...
		// And if you are DOMAIN_MINOR then n,v should be meaningful.
		// Other possible values:
		// OUTERMOST_CENTRAL == OUTERMOST, OUTER_FRILL, INNERMOST_CENTRAL, INNER_FRILL, INNER_TRIANGLE,
		// CROSSING_INS, INNER_CENTRAL -- 

		f64 n_ion = p_minor_nT_ion[tid].n;
		f64 n_e = p_minor_nT_elec[tid].n;
		f64_vec3 v_ion = p_minor_v_ion[tid];
		f64_vec3 v_e = p_minor_v_elec[tid];
		f64 Iz = q*(n_ion*v_ion.z - n_e*v_e.z)*p_area_minor[tid];			
		// Lots of bus loads, hopefully all contig.
		intrablock[threadIdx.x] = Iz;
		// HERE ASSUMED that area is calculated as DOMAIN INTERSECTION AREA
		// if we start including nv in insulator-crossing tris.
		
	} else {
		intrablock[threadIdx.x] = 0.0;
	};
	__syncthreads();
	
	// Now it's the aggregation:
	int s = blockDim.x;
	int k = s/2;
	while (s != 1) {
		if (threadIdx.x < k)
		{
			intrablock[threadIdx.x] += intrablock[threadIdx.x + k];
		};
		__syncthreads();
		// Attempt to modify:
		if ((s % 2 == 1) && (threadIdx.x == k-1)){
			intrablock[threadIdx.x] += intrablock[threadIdx.x+s-1];
		}; 
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s/2;
		__syncthreads();
	};
	if (threadIdx.x == 0)
	{
		p_summands[blockIdx.x] = intrablock[0];
	};
} // Doesn't matter much if function is slow, I think it is only called for debug purposes anyway.

__global__ void Kernel_Create_v_overall_and_newpos(
			structural * __restrict__ p_info,
			f64 const h,
			nT * __restrict__ p_nT_neut,
			nT * __restrict__ p_nT_ion,
			nT * __restrict__ p_nT_elec,
			f64_vec3 * __restrict__ p_v_neut,
			f64_vec3 * __restrict__ p_v_ion,
			f64_vec3 * __restrict__ p_v_elec,
			// Output:
			structural * __restrict__ p_info_out,
			f64_vec2 * __restrict__ p_v_overall
			)
{
	long tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//if (tid < Nverts) 
	structural info = p_info[tid];
	f64_vec2 v_save;
		
	if (info.flag == DOMAIN_VERTEX) 
	{ 
		nT nT_neut, nT_ion, nT_elec;
		f64_vec3 v_n, v_i, v_e;

		nT_neut = p_nT_neut[tid];
		nT_ion = p_nT_ion[tid];
		nT_elec = p_nT_elec[tid];
		v_n = p_v_neut[tid];
		v_i = p_v_ion[tid];
		v_e = p_v_elec[tid]; // expensive loads; can we avoid function by putting it in with smth else?
		
		f64_vec3 v_overall = (m_n*nT_neut.n*v_n + m_ion*nT_ion.n*v_i + m_e*nT_elec.n*v_e)/
							 (m_n*nT_neut.n + m_ion*nT_ion.n + m_e*nT_elec.n);
		v_save.x = v_overall.x;
		v_save.y = v_overall.y;
		info.pos += h*v_save;
	} else {
		v_save.x = 0.0; v_save.y = 0.0;	
	}
	
	p_v_overall[tid] = v_save;
	p_info_out[tid] = info; // safer to do unnecessary write of whole object to get contiguity.
	
	// can we do anything else with the data?
	// We could transfer it to shared and do something with it. But there isn't anything.
}


__global__ void Kernel_Average_v_overall_to_tris (
			LONG3 * __restrict__ p_tri_corner_index,
			CHAR4 * __restrict__ p_tri_perinfo,
			f64_vec2 * __restrict__ p_v_overall,
			f64_vec2 * __restrict__ p_tri_centroid,
			 // Output:
			f64_vec2 * __restrict__ p_minor_v_overall
			)
{
	__shared__ f64_vec2 shared_v[SIZE_OF_MAJOR_PER_TRI_TILE];
    
	// Averaging as 1/3 to tris.
	// Even corner positions do not tell us intersection. We'd have to know the neighbouring
	// centroid also.	
	// Load to shared:
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		shared_v[threadIdx.x] = p_v_overall[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];	 
	}
	// Let's hope it works with that sort of index. If it doesn't we're in a tough situation.
	
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long tid =  threadIdx.x + blockIdx.x * blockDim.x;
	
	__syncthreads();
	
	f64_vec2 v0, v1, v2, v_out;		
	LONG3 corner_index;
	CHAR4 perinfo;
	
	//if (tid < Ntris) {		 // redundant check
		
	corner_index = p_tri_corner_index[tid];
	perinfo = p_tri_perinfo[tid];
	
	if ((perinfo.flag == DOMAIN_TRIANGLE) || 
		(perinfo.flag == CROSSING_INS)) 
	{
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			v0 = shared_v[corner_index.i1-StartMajor];
		} else {
			// have to load in from global memory:
			v0 = p_v_overall[corner_index.i1];
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			v1 = shared_v[corner_index.i2-StartMajor];
		} else {
			v1 = p_v_overall[corner_index.i2];
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			v2 = shared_v[corner_index.i3-StartMajor];
		} else {
			v2 = p_v_overall[corner_index.i3];
		}

		if (perinfo.per0+perinfo.per1+perinfo.per2 == 0) {
		} else {
			// In this case which ones are periodic?
			// Should we just store per flags?
			// How it should work:
			// CHAR4 perinfo: periodic, per0, per1, per2;
			if (perinfo.per0 == NEEDS_ANTI) 
				v0 = Anticlock_rotate2(v0);
			if (perinfo.per0 == NEEDS_CLOCK)
				v0 = Clockwise_rotate2(v0);
			if (perinfo.per1 == NEEDS_ANTI)
				v1 = Anticlock_rotate2(v1);
			if (perinfo.per1 == NEEDS_CLOCK)
				v1 = Clockwise_rotate2(v1);
			if (perinfo.per2 == NEEDS_ANTI)
				v2 = Anticlock_rotate2(v2);
			if (perinfo.per2 == NEEDS_CLOCK)
				v2 = Clockwise_rotate2(v2);
		};
		
		v_out = THIRD*(v0+v1+v2);

		// For insulator triangle, 
		// we should take v_overall_r = 0
		// because this tri centroid will remain on the insulator.
		// It is OK to average with places that should have v_overall = 0.
		if (perinfo.flag == CROSSING_INS)
		{
			f64_vec2 r = p_tri_centroid[tid]; // random accesses??
			//f64_vec2 rhat = r/r.modulus();
			// v_out = v_out - rhat*v_out.dot(rhat);
			v_out = v_out - r*v_out.dot(r)/(r.x*r.x+r.y*r.y);
			// Well this is kinda wrong.


			// UNDEFINED 2ND TIME


		}
	} else {
		v_out.x = 0.0; v_out.y = 0.0;
	}
	p_minor_v_overall[tid] = v_out;
}


__global__ void Kernel_Average_nnionrec_to_tris
			(
			CHAR4 * __restrict__ p_tri_perinfo,
			LONG3 * __restrict__ p_tri_corner_index,
			nn * __restrict__ p_nn_ionrec,
			nn * __restrict__ p_nn_ionrec_minor
			)
{
	__shared__ nn shared_nn[SIZE_OF_MAJOR_PER_TRI_TILE];
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		shared_nn[threadIdx.x] = p_nn_ionrec[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];	 
	}
	// Let's hope it works with that sort of index. If it doesn't we're in a tough situation.
	
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__syncthreads();
	
	nn nn0, nn1, nn2;		
	LONG3 corner_index;
	nn nn_out;

	//if (tid < Ntris) {		 // redundant check - ?
	corner_index = p_tri_corner_index[tid];
	CHAR4 perinfo = p_tri_perinfo[tid];
	
	if ((perinfo.flag == DOMAIN_TRIANGLE) || 
		(perinfo.flag == CROSSING_INS)) 
	{
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			nn0 = shared_nn[corner_index.i1-StartMajor];
		} else {
			// have to load in from global memory:
			nn0 = p_nn_ionrec[corner_index.i1];
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			nn1 = shared_nn[corner_index.i2-StartMajor];
		} else {
			nn1 = p_nn_ionrec[corner_index.i2];
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			nn2 = shared_nn[corner_index.i3-StartMajor];
		} else {
			nn2 = p_nn_ionrec[corner_index.i3];
		}
		nn_out.n_ionise = THIRD*(nn0.n_ionise+nn1.n_ionise+nn2.n_ionise);
		nn_out.n_recombine = THIRD*(nn0.n_recombine+nn1.n_recombine+nn2.n_recombine);
		
		if (perinfo.flag == CROSSING_INS)
		{
			// Ensure that we are not using silly data...
			// Assume n_ionise = 0 outside domain.
			nn_out.n_ionise = 0.5*(nn0.n_ionise+nn1.n_ionise+nn2.n_ionise);
			nn_out.n_recombine = 0.5*(nn0.n_recombine+nn1.n_recombine+nn2.n_recombine);
		}
	} else {
		nn_out.n_ionise = 0.0;
		nn_out.n_recombine = 0.0;
	}
	p_nn_ionrec_minor[tid] = nn_out;
	
}

__global__ void Kernel_RelAdvect_nT(
		real h,						
		structural * __restrict__ p_info, // Advection for domain vertices only
		long * __restrict__ pIndexTri,
	
	//	char * __restrict__ pPBCTri,
	//  do we want this - or should we just use has_periodic flag ?
	// Debatable: has_periodic flag is non-maintainable if things cross the PB.
	// However that is probably all right, we should only like doing PB manip on CPU.

		f64_vec2 * __restrict__ p_minor_centroid, // work out tri centroids b4hand
		nT * __restrict__ p_nT_neut,
		nT * __restrict__ p_nT_ion,
		nT * __restrict__ p_nT_elec, 
		nT * __restrict__ p_minor_nT_neut,
		nT * __restrict__ p_minor_nT_ion,
		nT * __restrict__ p_minor_nT_elec,
		f64_vec3 * __restrict__ p_minor_v_neut,
		f64_vec3 * __restrict__ p_minor_v_ion,
		f64_vec3 * __restrict__ p_minor_v_elec,
		f64_vec2 * __restrict__ p_minor_v_overall,
		f64 * __restrict__ p_area_old,
		f64 * __restrict__ p_area_new,
		// dest:
		nT * __restrict__ p_nT_neut_out,
		nT * __restrict__ p_nT_ion_out,
		nT * __restrict__ p_nT_elec_out
		)
{
	// Idea is, we don't need to look at other nT
	// when we do this one -- though we can afterwards
	// push through from registry into shared, take grad nT,
	// if we want.
	// It is reasonable to overwrite and do one species after another.
			
	__shared__ f64_vec2 p_tri_centroid[SIZE_OF_TRI_TILE_FOR_MAJOR];  // + 2*2
	__shared__ f64_vec2 p_nv_shared[SIZE_OF_TRI_TILE_FOR_MAJOR];	   // + 2*2
	// Note: trimmed to vxy since we do not advect v here.
	__shared__ f64 p_T_shared[SIZE_OF_TRI_TILE_FOR_MAJOR];	// +1*2
	__shared__ long Indextri[MAXNEIGH_d*threadsPerTileMajor]; // +6 doublesworth
		
	// We could increase the occupancy by changing Indextri to load 1 index at a time
	// but if we use 63 registers then I think we get only about 512 per SM anyway.
	
	// FIRM PLAN:
	// Put triangles outside the outermost row of vertices. Padding that will make us have
	// 2x the number of triangles as vertices.
	// These triangles will have v = 0 in our setup. 
	// In general though they serve no purpose?
	
	// We still need to load a periodic flag for each triangle so not loading a general flag didn't achieve much...
	// Alternatively, load a "has periodic" flag for this vertex as part of structural:
	// instead of two shorts, have short char char = neigh_len,has_periodic,general_flag
	// That seems logical - we don't need a general flag to be a short rather than a char.
		
	// Occupancy calculator says try having 192 instead of 128 in a major tile.
	// Just have to see empirically when programming is all done.
	
	// 2 ways round: with tri centroids loading in:
	//  shared 2 * (2 + 2 + 1) + 6 from indextri = 16 doubles equiv!!
	//   with vertex pos loading in:
	//  shared 2 + 2*(2 + 1) + 6 + 6 = 20 doubles equiv!!
	
	// This is simply a huge amount of data to have to deal with.
	// Was advection a full up routine before?
	// Yes - it was so bad we could not fit in IndexNeigh into shared.
	
	// One way is to CONTIGUOUSLY load Indextri on the go:
	// put arrays of member0, member1, etc.
	// Would it even recognise as a contiguous load? We could force it to.
	// That is an interesting alternative but not sure about it.
	
	// On the plus side, bus activity is reduced by doing the way we ARE doing it.
	// This way, we reuse Indextri for 3 species instead of loading x3.
	// Don't have a sense of really how long bus trips take compared to everything else.
	// That will be something to learn this time - with nvprof, nSight.
	
	// How much shared now? About 10 doubles per thread.
	// 80*256 vs 48*1024 = 192*256. 2 blocks of 256 at a time.
	// We are ending up with too few threads running.
	// Solution: Store both nv, and T, thus to compose nvT when needed.
	
	// *****
	// It would be more sensible to run thread for each triangle but with same stored data as in this block --- surely?
	// *****
	
	long StartMinor = blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR;
	long EndMinor = (blockIdx.x+1)*SIZE_OF_TRI_TILE_FOR_MAJOR;
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	f64_vec3 v_3;
	f64_vec2 v1, v2, v_overall, v_overall2; // can drop 1 of these...
	f64 area_old, area_new;
	
	structural info = p_info[index];
	//if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) 
	// TRY WITH AND WITHOUT THIS.
	// LOADING IN UNNECESSARY DATA FOR OUT-OF-DOMAIN .VS. 
	// IF IT DOES NOT TRIGGER CONTIG ACCESS PROPERLY WITHIN BRANCH.
	// ... Probably it _IS_ necessary to load unnecessary data.
	
	// ##################################################################
	// The easy and correct thing, if we are only treating those
	// that are DOMAIN/OUTERMOST, should be to only call for those blocks.
	// ##################################################################
	
	// Behaviour we want:
	// Valid edges for traffic: domain-domain
	//                          domain-outermost
	//                          outermost-outermost [optional]
	//            Not valid: traffic into insulator --- but should get v_tri_r == 0
	//                       traffic into frills    --- careful to avoid!
	//                       anything not involving domain/outermost

	{	
		{
			nT nT_temp = p_minor_nT_neut[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_3 = p_minor_v_neut[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_overall = p_minor_v_overall[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v2.x = v_3.x-v_overall.x;
			v2.y = v_3.y-v_overall.y;
			
			p_nv_shared[threadIdx.x] = nT_temp.n*v2;
			
			// **********************************************************************
			// CONSIDER: We promised that we would use the J that appears in the
			// A-dot advance formula, for flowing charge. Is that what we are doing?
			// We averaged n to the minor tile and multiplied by minor velocity rel to mesh.
			// I guess that is okay...
			// **********************************************************************
			
			p_T_shared[threadIdx.x] = nT_temp.T;
			nT_temp = p_minor_nT_neut[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_3 = p_minor_v_neut[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_overall2 = p_minor_v_overall[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v2.x = v_3.x-v_overall2.x;
			v2.y = v_3.y-v_overall2.y;
			p_nv_shared[blockDim.x + threadIdx.x] = nT_temp.n*v2;
			p_T_shared[blockDim.x + threadIdx.x] = nT_temp.T;			
		}			
		p_tri_centroid[threadIdx.x] = p_minor_centroid[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
		p_tri_centroid[blockDim.x + threadIdx.x] = p_minor_centroid[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
	}
	__syncthreads(); // Avoid putting within branch.
	
	if ((info.flag == DOMAIN_VERTEX)) 
		// || (info.flag == OUTERMOST))  // 29/06/17
	{
		// h*n*v.dot(edgenormal) is amount of traffic between major cells
		
		// Next job is to compute edge_normal
		// And decide whether this is a legit edge for traffic or not.
		
		nT nTsrc = p_nT_neut[index]; // Not needed up here...
		area_old = p_area_old[index];
		area_new = p_area_new[index];
		// hope that by putting here we get contiguous access.
		
		memcpy(Indextri + MAXNEIGH_d*threadIdx.x, 
					pIndexTri + MAXNEIGH_d*index,
					MAXNEIGH_d*sizeof(long)); // MAXNEIGH should be chosen to be 12, for 1 full bus.
		// memcpy(PBCtri + MAXNEIGH_d*threadIdx.x, 
		//			pPBCTri + MAXNEIGH_d*index,
		//			MAXNEIGH_d*sizeof(char)); // MAXNEIGH should be chosen to be 12, for 1 full bus.

		// By running threads per tri, we'd dispense with Indextri in shared and store solution (towards colour
		// array) for vertex tile instead. 
		// Then we just incur a reload to aggregate the colour array.
		
		// Easiest way:
		// Edge involves 2 centres and 2 values of nv etc
		f64_vec2 nv1, nvT1, nv2, nvT2, pos1, pos2; // lots of registers ... here 12
		
		short iNeigh1 = info.neigh_len-1; // NOTE it is possible vertex has
		// different number of neighs and triangles. What happens?

		if (info.flag == OUTERMOST) iNeigh1++; // get to end of array.

		short iNeigh2 = 0;
		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if ((indextri >= StartMinor) && (indextri < EndMinor))
		{
			nv1 = p_nv_shared[indextri-StartMinor];
			nvT1 = p_T_shared[indextri-StartMinor]*nv1; 
			pos1 = p_tri_centroid[indextri-StartMinor];

				if (index == 85400-BEGINNING_OF_CENTRAL) {
					printf("%d nv1.x %1.5E \n",indextri,nv1.x);
				};
		} else {
			nT nT1 = p_minor_nT_neut[indextri];
			v_3 = p_minor_v_neut[indextri];
			v_overall = p_minor_v_overall[indextri];
			v1.x = v_3.x-v_overall.x;
			v1.y = v_3.y-v_overall.y;
			nv1 = nT1.n*v1;
			nvT1 = nT1.T*nv1;
			pos1 = p_minor_centroid[indextri];
			// Bad news: 3 separate bus journeys.
			// We probably spend AT LEAST half our time here. Tile of 12 x 12 -> 144 within, 48 edge.
			// We could be sending a more full bus by putting nTv.
			// That would reduce costs here by 33%.
			// The increased cost would be, that when we create n,T by averaging, we have to write access only 
			// part of an nvT struct.
			// However, it is possible that a lot of these bus journeys take place at the same time that
			// other threads are NOT needing a bus journey. Consider that.

			// Stick with separate nT,v for now. We may never know, how much faster nvT would have been.
				
				if (index == 85400-BEGINNING_OF_CENTRAL) {
					printf("v_3 %1.6E %1.6E %d v_overall %1.6E %1.6E nT1.n %1.6E\n",
						v_3.x,v_3.y,indextri,v_overall.x,v_overall.y,nT1.n);
				};
				
		};


		
		//char PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if (info.has_periodic) {

			if ((pos1.x > pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Anticlock_rotate2(nv1); // ANTI is to mean that the tri is on the right, vertex on left.
				nvT1 = Anticlock_rotate2(nvT1);  // ANTI means apply anticlockwise.
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Clockwise_rotate2(nv1);
				nvT1 = Clockwise_rotate2(nvT1);
				pos1 = Clockwise_rotate2(pos1);
			};
			// Assume we always find periodic neigh to right/left of 1/4-way line, and same
			// for the point itself.
		};
		
		f64 mass, heat;
		mass = 0.0; heat = 0.0;
		
#pragma unroll 12
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh2];
			if ((indextri >= StartMinor) && (indextri < EndMinor))
			{
				nv2 = p_nv_shared[indextri-StartMinor];
				nvT2 = p_T_shared[indextri-StartMinor]*nv2;
				pos2 = p_tri_centroid[indextri-StartMinor];
			} else {
				nT nT2 = p_minor_nT_neut[indextri];
				v_3 = p_minor_v_neut[indextri];
				f64_vec2 v_overall_ = p_minor_v_overall[indextri];
				v2.x = v_3.x-v_overall_.x;
				v2.y = v_3.y-v_overall_.y;
				nv2 = nT2.n*v2;
				nvT2 = nT2.T*nv2;
				pos2 = p_minor_centroid[indextri];
			};
			
		    // Two ways to store periodic: either 3 longs in registers, or,
			// as an array of chars in shared memory.			
			// Alternative: each tri knows if it is periodic and we somehow
			// load this alongside tri centroid, as a CHAR4.
			if (info.has_periodic) {
				if ((pos2.x > pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Anticlock_rotate2(nv2); // ANTI is to mean that the tri is on the right, vertex on left.
					nvT2 = Anticlock_rotate2(nvT2);  // ANTI means apply anticlockwise.
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Clockwise_rotate2(nv2);
					nvT2 = Clockwise_rotate2(nvT2);
					pos2 = Clockwise_rotate2(pos2);
				};
				// Assume we always find periodic neigh to right/left of 1/4-way line, and same
				// for the point itself.
			};
			
			f64_vec2 edgenormal;
			edgenormal.x = pos2.y-pos1.y; // 2 is the more anticlockwise one
			edgenormal.y = pos1.x-pos2.x;
			
			// At edge of memory, whether we have extra outer tris involved or not,
			// counting all edges means we create an edge looking out of the domain.
			// It's our choice whether current can flow out of the domain or not.
			// Probably best if not. 
			// So either we need to find a way to import a flag here, OR, set vr to zero (v=0?)
			// either in extra-outer tris or in tris just inside.
			
			// The extra-outer tris are sounding more appealing all the time. Let's go for them.
			if (1) { // if legitimate edge
				f64 flow = 0.5*h*((nv1+nv2).dot(edgenormal));
				mass -= flow; // correct? -- compare
				

				flow = 0.5*h*((nvT1+nvT2).dot(edgenormal));
				heat -= flow;
			};
			
			nvT1 = nvT2;
			nv1 = nv2;
			pos1 = pos2;			
		}; // next neigh
		
		// If we did the above with triangle threads that update a solution in shared memory,
		// we could switch to half the block doing the following:
		
		if (index == 85400-BEGINNING_OF_CENTRAL) {
			printf("mass %1.10E src %1.10E area_new %1.10E\n",
				mass, nTsrc.n*area_old, area_new);
		};

		mass += nTsrc.n*area_old;
		heat += nTsrc.n*nTsrc.T*area_old;
		nT nT_out;
		nT_out.n = mass/area_new;
		nT_out.T = heat/mass;
		
		// Compressive heating:
		// We need here new area and old area:
		
		nT_out.T *= (1.0-0.666666666666667*(nT_out.n-nTsrc.n)/nTsrc.n
		                -0.111111111111111*(nT_out.n-nTsrc.n)*(nT_out.n-nTsrc.n)/(nTsrc.n*nTsrc.n));
		// Note: 2 divisions vs 1 call to pow
		

		// ?:
		// Where is recognition that we cannot flow into insulator? Depends just on v_r = 0??


		p_nT_neut_out[index] = nT_out;	
	} else {  // whether DOMAIN VERTEX 
		
		p_nT_neut_out[index] = p_nT_neut[index];
	};
	
	__syncthreads(); // avoid syncthreads within branch.
	
	// Now we allow additional inflow from OUTERMOST. 29/06/17

	// That means basically electrons moving in. Then we want to get rid of them
	// each step, send them up per z current.
	
	
	// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
	
	
	// Ready for next species:
	//if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))  
	{	// The point here is to reuse both the tri indices and the centroids.

		// We should realise that putting a DOMAIN condition on this will make it go wrong:
		// we are not loading all ins triangles this way, but we will assume we can 
		// use them, as far as I can see.

		// If we only drop whole blocks that are INNER_VERTEX -- which we should --
		// then we should be all right here -- if it's not part of this block then it's loaded
		// separately.
		
		nT nT_temp = p_minor_nT_ion[
			blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
		v_3 = p_minor_v_ion[
			blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
		v2.x = v_3.x-v_overall.x;
		v2.y = v_3.y-v_overall.y;
		p_nv_shared[threadIdx.x] = nT_temp.n*v2;
		p_T_shared[threadIdx.x] = nT_temp.T;
		
		nT_temp = p_minor_nT_ion[
			blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
		v_3 = p_minor_v_ion[
			blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
		v2.x = v_3.x-v_overall2.x;
		v2.y = v_3.y-v_overall2.y;
		p_nv_shared[blockDim.x + threadIdx.x] = nT_temp.n*v2;
		p_T_shared[blockDim.x + threadIdx.x] = nT_temp.T;			
		
	}		
	__syncthreads(); // Avoid putting within branch.
	
	f64_vec2 nv1, nvT1, pos1, nv2, nvT2, pos2;
	nT nT1, nT2;

	if ((info.flag == DOMAIN_VERTEX))// || (info.flag == OUTERMOST))  {
	{
		// h*n*v.dot(edgenormal) is amount of traffic between major cells
		
		nT nTsrc = p_nT_ion[index];
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;
		
		if (info.flag == OUTERMOST) iNeigh1++; // get to end of array.

		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if ((indextri >= StartMinor) && (indextri < EndMinor))
		{
			nv1 = p_nv_shared[indextri-StartMinor];
			nvT1 = p_T_shared[indextri-StartMinor]*nv1; // extra access to shared - nvm
			pos1 = p_tri_centroid[indextri-StartMinor];
		} else {
			nT1 = p_minor_nT_ion[indextri];
			v_3 = p_minor_v_ion[indextri];
			v_overall = p_minor_v_overall[indextri];
			v1.x = v_3.x-v_overall.x;
			v1.y = v_3.y-v_overall.y;
			nv1 = nT1.n*v1;
			nvT1 = nT1.T*nv1;
			pos1 = p_minor_centroid[indextri];
		};
		//char PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if (info.has_periodic) {
			if ((pos1.x > pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Anticlock_rotate2(nv1); // ANTI is to mean that the tri is on the right, vertex on left.
				nvT1 = Anticlock_rotate2(nvT1);  // ANTI means apply anticlockwise.
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Clockwise_rotate2(nv1);
				nvT1 = Clockwise_rotate2(nvT1);
				pos1 = Clockwise_rotate2(pos1);
			};
			// Assume we always find periodic neigh to right/left of 1/4-way line, and same
			// for the point itself.
		};
		
		f64 mass, heat;
		mass = 0.0; heat = 0.0;		
#pragma unroll 12
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++) 
			// aar - if we have an outer point
			// then the number of neighs is not the number of tris
			// SO EXPLOIT THIS
			// Make sure that the omitted edge is the one that would go between the frill tris.
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh2];
			if ((indextri >= StartMinor) && (indextri < EndMinor))
			{
				nv2 = p_nv_shared[indextri-StartMinor];
				nvT2 = p_T_shared[indextri-StartMinor]*nv2;
				pos2 = p_tri_centroid[indextri-StartMinor];
			} else {
				nT2 = p_minor_nT_ion[indextri];
				v_3 = p_minor_v_ion[indextri];
				f64_vec2 v_overall_ = p_minor_v_overall[indextri];
				v2.x = v_3.x-v_overall_.x;
				v2.y = v_3.y-v_overall_.y;
				f64_vec2 nv2 = nT2.n*v2;
				nvT2 = nT2.T*nv2;
				pos2 = p_minor_centroid[indextri];
			};
			
		    if (info.has_periodic) {
				if ((pos2.x > pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Anticlock_rotate2(nv2); // ANTI is to mean that the tri is on the right, vertex on left.
					nvT2 = Anticlock_rotate2(nvT2);  // ANTI means apply anticlockwise.
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Clockwise_rotate2(nv2);
					nvT2 = Clockwise_rotate2(nvT2);
					pos2 = Clockwise_rotate2(pos2);
				};
			};
			
			f64_vec2 edgenormal;
			edgenormal.x = pos2.y-pos1.y; // 2 is the more anticlockwise one
			edgenormal.y = pos1.x-pos2.x;
			
			if (1) { // if legitimate edge -- remember we should treat edge the same way from both sides.
				f64 flow = 0.5*h*((nv1+nv2).dot(edgenormal));
				mass -= flow; // correct? -- compare
				flow = 0.5*h*((nvT1+nvT2).dot(edgenormal));
				heat -= flow;
			};
			
			nvT1 = nvT2;
			nv1 = nv2;
			pos1 = pos2;
		}; // next neigh
		
		mass += nTsrc.n*area_old;
		heat += nTsrc.n*nTsrc.T*area_old;
		
		nT nT_out;

		nT_out.n = mass/area_new;
		nT_out.T = heat/mass;
		
		// Compressive heating:
		// We need here new area and old area:		
		nT_out.T *= (1.0-0.666666666666667*(nT_out.n-nTsrc.n)/nTsrc.n
		                -0.111111111111111*(nT_out.n-nTsrc.n)*(nT_out.n-nTsrc.n)/(nTsrc.n*nTsrc.n));
		
		p_nT_ion_out[index] = nT_out;
	} else {
		
		p_nT_ion_out[index] = p_nT_ion[index];
	}
	
	// The point here is to reuse both the tri indices and the centroids.
	
	// Ready for next species:
	//if (info.flag == DOMAIN_VERTEX) {
		// TRY WITH AND WITHOUT THIS.
		{
			nT nT_temp = p_minor_nT_elec[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_3 = p_minor_v_elec[
				blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v2.x = v_3.x-v_overall.x;
			v2.y = v_3.y-v_overall.y;
			nv2 = nT2.n*v2;
			nvT2 = nT2.T*nv2;
			p_nv_shared[threadIdx.x] = nT_temp.n*v2;
			p_T_shared[threadIdx.x] = nT_temp.T;
			
			nT_temp = p_minor_nT_elec[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v_3 = p_minor_v_elec[
				blockDim.x + blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
			v2.x = v_3.x-v_overall2.x;
			v2.y = v_3.y-v_overall2.y;
			p_nv_shared[blockDim.x + threadIdx.x] = nT_temp.n*v2;
			p_T_shared[blockDim.x + threadIdx.x] = nT_temp.T;			
		}	
	//}		
	__syncthreads(); // Avoid putting within branch.

	
	if ((info.flag == DOMAIN_VERTEX)) { // || (info.flag == OUTERMOST)) {
		nT nTsrc = p_nT_elec[index];
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;
		
		if (info.flag == OUTERMOST) iNeigh1++; // get to end of array.

		long indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if ((indextri >= StartMinor) && (indextri < EndMinor))
		{
			nv1 = p_nv_shared[indextri-StartMinor];
			nvT1 = p_T_shared[indextri-StartMinor]*nv1; // extra access to shared - nvm
			pos1 = p_tri_centroid[indextri-StartMinor];
		} else {
			nT1 = p_minor_nT_elec[indextri];
			v_3 = p_minor_v_elec[indextri];
			v_overall = p_minor_v_overall[indextri];
			v1.x = v_3.x-v_overall.x;
			v1.y = v_3.y-v_overall.y;
			nv1 = nT1.n*v1;
			nvT1 = nT1.T*nv1;
			pos1 = p_minor_centroid[indextri];
		};
		//char PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh1];
		if (info.has_periodic) {
			if ((pos1.x > pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Anticlock_rotate2(nv1); // ANTI is to mean that the tri is on the right, vertex on left.
				nvT1 = Anticlock_rotate2(nvT1);  // ANTI means apply anticlockwise.
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -pos1.y*GRADIENT_X_PER_Y*0.5) &&
				(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
			{
				nv1 = Clockwise_rotate2(nv1);
				nvT1 = Clockwise_rotate2(nvT1);
				pos1 = Clockwise_rotate2(pos1);
			};
			// Assume we always find periodic neigh to right/left of 1/4-way line, and same
			// for the point itself.
		};
		
		f64 mass, heat;
		mass = 0.0; heat = 0.0;		
#pragma unroll 12
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			indextri = Indextri[MAXNEIGH_d*threadIdx.x + iNeigh2];
			if ((indextri >= StartMinor) && (indextri < EndMinor))
			{
				nv2 = p_nv_shared[indextri-StartMinor];
				nvT2 = p_T_shared[indextri-StartMinor]*nv2;
				pos2 = p_tri_centroid[indextri-StartMinor];
			} else {
				nT2 = p_minor_nT_elec[indextri];
				v_3 = p_minor_v_elec[indextri];
				f64_vec2 v_overall_ = p_minor_v_overall[indextri];
				v2.x = v_3.x-v_overall_.x;
				v2.y = v_3.y-v_overall_.y;
				nv2 = nT2.n*v2;
				nvT2 = nT2.T*nv2;
				pos2 = p_minor_centroid[indextri];
			};
			
		    if (info.has_periodic) {
				if ((pos2.x > pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x < -info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Anticlock_rotate2(nv2); // ANTI is to mean that the tri is on the right, vertex on left.
					nvT2 = Anticlock_rotate2(nvT2);  // ANTI means apply anticlockwise.
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -pos2.y*GRADIENT_X_PER_Y*0.5) &&
					(info.pos.x > info.pos.y*GRADIENT_X_PER_Y*0.5))
				{
					nv2 = Clockwise_rotate2(nv2);
					nvT2 = Clockwise_rotate2(nvT2);
					pos2 = Clockwise_rotate2(pos2);
				};
				// Assume we always find periodic neigh to right/left of 1/4-way line, and same
				// for the point itself.
			};
			
			f64_vec2 edgenormal;
			edgenormal.x = pos2.y-pos1.y; // 2 is the more anticlockwise one
			edgenormal.y = pos1.x-pos2.x;
			
			if (1) { // if legitimate edge --- how to know if we are looking into outermost??
				// We are not loading info about neighbours. Yet it is only the neigh that knows
				// it is OUTERMOST.

				f64 flow = 0.5*h*((nv1+nv2).dot(edgenormal));
				mass -= flow; // correct? -- compare
				flow = 0.5*h*((nvT1+nvT2).dot(edgenormal));
				heat -= flow;

				// Meanwhile what if we are looking through insulator.
				// We should find there that we have insisted on v_r=0 and so v.dot(edgenormal) roughly = 0.

				// But we need to consider about outermost what to do.
				// We don't really want to be arbitrarily losing or gaining charge.
				// The answer is to include OUTERMOST flag, but, disinclude the outermost edge of an OUTERMOST vertex.
				// This can happen automatically by a CAREFUL NUMBERING of outermost tris and neighs.
				// Does it disagree with the numbering we previously considered canonical? Probably yes --> edit through :-/
			};			
			
			nvT1 = nvT2;
			nv1 = nv2;
			pos1 = pos2;
		}; // next neigh		
		mass += nTsrc.n*area_old;
		heat += nTsrc.n*nTsrc.T*area_old; 

		nT nT_out;
		nT_out.n = mass/area_new;
		nT_out.T = heat/mass;		

		// Compressive heating:
		// We need here new area and old area:		
		nT_out.T *= (1.0-0.666666666666667*(nT_out.n-nTsrc.n)/nTsrc.n
		                -0.111111111111111*(nT_out.n-nTsrc.n)*(nT_out.n-nTsrc.n)/(nTsrc.n*nTsrc.n));

		p_nT_elec_out[index] = nT_out;
	} else {
		
		p_nT_elec_out[index] = p_nT_elec[index];
	};
}


__global__ void Kernel_Populate_A_frill(
			CHAR4 * __restrict__ p_tri_info,
			f64_vec3 * __restrict__ p_A, // update own, read others
			f64_vec2 * __restrict__ p_tri_centroid,
			//LONG3 * __restrict__ p_corner_index
			LONG3 * __restrict__ p_tri_neigh_index)
{
	//long index = (blockIdx.x + BLOCK_START_OF_FRILL_SEARCH_d)*blockDim.x + threadIdx.x;
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	// load the two corner indices
	CHAR4 perinfo = p_tri_info[index];
	
	if (perinfo.flag == OUTER_FRILL) {
		
		//LONG3 cornerindex = p_corner_index[index];
		//A0 = p_A[BEGINNING_OF_CENTRAL + cornerindex.i1];
		//A1 = p_A[BEGINNING_OF_CENTRAL + cornerindex.i2];
		//if (perinfo.per0 == NEEDS_CLOCK) A0 = Clockwise_rotate2(A0);
		//if (perinfo.per1 == NEEDS_CLOCK) A1 = Clockwise_rotate2(A1);
		//if (perinfo.per0 == NEEDS_ANTI) A0 = Anticlock_rotate2(A0);
		//if (perinfo.per1 == NEEDS_ANTI) A1 = Anticlock_rotate2(A1);
		//p_A[index] = 0.5*(A0 + A1);
		
		// Just do this instead:
		LONG3 neighindex = p_tri_neigh_index[index];
		f64_vec2 cent = p_tri_centroid[index];
		f64_vec2 centneigh = p_tri_centroid[neighindex.i1];
		f64_vec3 A = p_A[neighindex.i1];
		// Axy decrease radially:
		f64 factor = sqrt((centneigh.x*centneigh.x+centneigh.y*centneigh.y)/
						  (cent.x*cent.x+cent.y*cent.y));
		A.x *= factor;
		A.y *= factor;
		p_A[index] = A;
	};
	if (perinfo.flag == INNER_FRILL) {
		
		//LONG3 cornerindex = p_corner_index[index];
		//A0 = p_A[BEGINNING_OF_CENTRAL + cornerindex.i1];
		//A1 = p_A[BEGINNING_OF_CENTRAL + cornerindex.i2];
		//if (perinfo.per0 == NEEDS_CLOCK) A0 = Clockwise_rotate2(A0);
		//if (perinfo.per1 == NEEDS_CLOCK) A1 = Clockwise_rotate2(A1);
		//if (perinfo.per0 == NEEDS_ANTI) A0 = Anticlock_rotate2(A0);
		//if (perinfo.per1 == NEEDS_ANTI) A1 = Anticlock_rotate2(A1);
		//p_A[index] = 0.5*(A0 + A1);
		
		// Just do this instead:
		LONG3 neighindex = p_tri_neigh_index[index];
		f64_vec2 cent = p_tri_centroid[index];
		f64_vec2 centneigh = p_tri_centroid[neighindex.i1];
		f64_vec3 A = p_A[neighindex.i1];
		// Axy decrease radially:
		f64 factor = sqrt((cent.x*cent.x+cent.y*cent.y)/
						(centneigh.x*centneigh.x+centneigh.y*centneigh.y));
		A.x *= factor;
		A.y *= factor;
		p_A[index] = A;
	};
}

// The same sort of routine as the following will be needed to anti-advect A,Adot,phi,phidot.
// Bad news; no way to avoid though... could we interpolate to new values? Is that really much different.
// Crude estimate of grad is okay. 


__global__ void Kernel_Compute_Grad_A_minor_antiadvect(
			f64_vec3 * __restrict__ p_A_tri,        // for creating grad
			f64_vec3 * __restrict__ p_A_vert,       // 
			f64 h,
			f64_vec2 * __restrict__ p_v_overall,    // hv = amt to anti-advect
			structural * __restrict__ p_info,       // 
			f64_vec2 * __restrict__ p_tri_centroid, // 
			CHAR4 * __restrict__ p_tri_perinfo,     // 
			CHAR4 * __restrict__ p_tri_per_neigh,
			LONG3 * __restrict__ p_corner_index,    // 
			LONG3 * __restrict__ p_neigh_tri_index, // 
			long * __restrict__ p_IndexTri,         // we said carry on using this for now.
			bool bAdd,
			f64_vec3 * __restrict__ p_Addition_Rate,
			// output:
			f64_vec3 * __restrict__ p_A_out // fill in for both tri and vert...			
			)
{
	__shared__ f64_vec3 A_tri[threadsPerTileMinor];
	__shared__ f64_vec2 tri_centroid[threadsPerTileMinor];				// 5
	__shared__ f64_vec3 A_vert[SIZE_OF_MAJOR_PER_TRI_TILE]; // +1.5
	__shared__ f64_vec2 vertex_pos[SIZE_OF_MAJOR_PER_TRI_TILE];
	
	// If we want 512 threads/SM, 12 doubles in shared per thread is limit.
	// We can accommodate 12 .. so 6 per major in addition to this but not when we have shared_per.
	// Well we could limit it to 10 tris but it's asking for trouble.
	// 6 longs = 3 doublesworth per thread
	
	__shared__ long IndexTri[SIZE_OF_MAJOR_PER_TRI_TILE*MAXNEIGH_d]; // +3
	
	// Do first with 3 dimensions Axyz at once - may be slower but we'll see.
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	long StartTri = blockIdx.x*blockDim.x; // can replace this one.
	long StartMajor = SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x; // can replace this one.
	// could replace with #define here.	
	A_tri[threadIdx.x] = p_A_tri[index];
	tri_centroid[threadIdx.x] = p_tri_centroid[index];
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE) {
		A_vert[threadIdx.x] = p_A_vert[SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x + threadIdx.x];
		structural info = p_info[SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x + threadIdx.x];
		vertex_pos[threadIdx.x] = info.pos;
	}
	
	//	shared_per[threadIdx.x] = perinfo.per0+perinfo.per1+perinfo.per2; // if periodic tri then neigh will need to be able to know.
	// note that we have to make sure CHAR4 takes up 32 bits not 4 x 32.
	// Is that the root of our problems with footprint?
	// If so, what should we do? Bitwise operations on a char?
	
	__syncthreads();
	
	f64_vec2 gradAx(0.0,0.0);
	f64_vec2 gradAy(0.0,0.0);
	f64_vec2 gradAz(0.0,0.0);
	f64_vec2 v_overall = p_v_overall[index];
	CHAR4 perinfo = p_tri_perinfo[index];
	{
		// Allow it to run through and produce nonsense for frills....		
		CHAR4 tri_rotate = p_tri_per_neigh[index];
		LONG3 corner_index = p_corner_index[index];
		LONG3 neightri = p_neigh_tri_index[index]; 
		// Note that A, as well as position, has to be ROTATED to make a contiguous image.
		// This tri minor has 3 edges with triangles and 3 edges with centrals.
				
		f64 area = 0.0;
		f64_vec2 pos0(9.0,19.0), pos1 (1.0,2.0), pos2(4.0,2.0);
		
		//	f64_vec3 Avert0,Avert1,Avert2;
		// We need 4 values at a time in order to do a side.
		// We don't need to have all 7 values (3+ 3 + itself)
		// So we'd be better just to load one quadrilateral's doings at a time, given the paucity of register and L1 space.
		// Either we store all 7x3 A-values at once + 7 positions. Or we use 4 and 4 positions at once.
		// Bear in mind a partial saving might yield big benefits.
		// HAZARD: we don't know the ordering of the points. 
		
		// Halfway house: for simplicity store all the positions already loaded.
		// A does not load from the same place anyway.
		// Then go round the quadrilaterals.

		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos0 = vertex_pos[corner_index.i1-StartMajor];
		} else {
			structural info = p_info[corner_index.i1];
			pos0 = info.pos;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos1 = vertex_pos[corner_index.i2-StartMajor];
		} else {
			structural info = p_info[corner_index.i2];
			pos1 = info.pos;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos2 = vertex_pos[corner_index.i3-StartMajor];
		} else {
			structural info = p_info[corner_index.i3];
			pos2 = info.pos;
		}
		
		if (perinfo.per0+perinfo.per1+perinfo.per2 == 0) {
		} else {
			// In this case which ones are periodic?
			// Should we just store per flags?
			// How it should work:
			// CHAR4 perinfo: periodic, per0, per1, per2;
			if (perinfo.per0 == NEEDS_ANTI) {
				pos0 = Anticlock_rotate2(pos0);
			}
			if (perinfo.per0 == NEEDS_CLOCK) { // this means the corner is off the clockwise side. Therefore anticlockwise rotated.
				pos0 = Clockwise_rotate2(pos0);
			}
			if (perinfo.per1 == NEEDS_ANTI) {
				pos1 = Anticlock_rotate2(pos1);
			}
			if (perinfo.per1 == NEEDS_CLOCK) {
				pos1 = Clockwise_rotate2(pos1);
			}
			if (perinfo.per2 == NEEDS_ANTI) {
				pos2 = Anticlock_rotate2(pos2);
			}
			if (perinfo.per2 == NEEDS_CLOCK) {
				pos2 = Clockwise_rotate2(pos2);
			}
		};

		// It worked with none of the calcs in. Now we bring back the above. Still works

		f64_vec2 u0(1.0,2.0), 
			     u1(0.0,2.0), 
				 u2(3.0,1.0); 
		// to be the positions of neighbouring centroids
	//	CHAR4 tri_rotate; // 4 chars but really using 3
	//	tri_rotate.per0 = 0; tri_rotate.per1 = 0; tri_rotate.per2 = 0;
		
		char periodic = perinfo.per0+perinfo.per1+perinfo.per2;
		
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri+blockDim.x))
		{
			u0 = tri_centroid[neightri.i1-StartTri];
		} else {
			u0 = p_tri_centroid[neightri.i1];
		};
		if (tri_rotate.per0 == NEEDS_CLOCK)
			u0 = Clockwise_rotate2(u0);
		if (tri_rotate.per0 == NEEDS_ANTI)
			u0 = Anticlock_rotate2(u0);
		
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri+blockDim.x))
		{
			u1 = tri_centroid[neightri.i2-StartTri];
		} else {
			u1 = p_tri_centroid[neightri.i2];
		}
		if (tri_rotate.per1 == NEEDS_CLOCK)
			u1 = Clockwise_rotate2(u1);
		if (tri_rotate.per1 == NEEDS_ANTI)
			u1 = Anticlock_rotate2(u1);
		
		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri+blockDim.x))
		{
			u2 = tri_centroid[neightri.i3-StartTri];
		} else {
			u2 = p_tri_centroid[neightri.i3];
		}
		if (tri_rotate.per2 == NEEDS_CLOCK)
			u2 = Clockwise_rotate2(u2);
		if (tri_rotate.per2 == NEEDS_ANTI)
			u2 = Anticlock_rotate2(u2);
		// still works

		// ............................................................................................
		// . I think working round with 4 has a disadvantage: if we get back around to one that is off-tile,
		// we have to load it all over again. Still that is only 1 out of 7 that gets duplicated.
		// Here is the best thing I can come up with: store 7 positions. That is already
		// 28 longs' worth... each A-value uses 6 of the 7 positions to have an effect.
		// Load each A-value at a time and recalc shoelace for 3 quadrilaterals. ??
			// Too complicated.
		// If we store all positions, can we finish with each A as we handle it? Yes but let's not.
		
		//f64_vec2 ourpos = tri_centroid[threadIdx.x]; // can try with and without this assignment to variable
		//f64_vec3 A0 = A_tri[threadIdx.x]; // can try with and without this assignment to variable
		f64_vec3 A_1(0.0,0.0,0.0),
				A_out(0.0,0.0,0.0),
				A_2(0.0,0.0,0.0);
		
		// Our A: A_tri[threadIdx.x]
		
		// Now fill in the A values:
		// ____________________________
		
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_1 = A_vert[corner_index.i2-StartMajor];
		} else {
			A_1 = p_A_vert[corner_index.i2];
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i3-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i3];
		}

		if (periodic == 0) {
		} else {
			if (perinfo.per1 == NEEDS_ANTI) {
				A_1 = Anticlock_rotate3(A_1);
			}
			if (perinfo.per1 == NEEDS_CLOCK) {
				A_1 = Clockwise_rotate3(A_1);
			}
			if (perinfo.per2 == NEEDS_ANTI) {
				A_2 = Anticlock_rotate3(A_2); 
			};
			if (perinfo.per2 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			};
		}

		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + blockDim.x))
		{
			A_out = A_tri[neightri.i1-StartTri];
		} else {
			A_out = p_A_tri[neightri.i1];
		}
		if (tri_rotate.per0 != 0) {
			if (tri_rotate.per0 == NEEDS_CLOCK) {
				A_out = Clockwise_rotate3(A_out);
			} else {
				A_out = Anticlock_rotate3(A_out);
			};
		};
		// ======================================================
		
	//	shoelace = (ourpos.x-u0.x)*(pos1.y-pos2.y)
	//			 + (pos1.x-pos2.x)*(u0.y-ourpos.y); // if u0 is opposite point 0
					// clock.x-anti.x
		
		// We are now going to put the corners of the minor cell at 
		// e.g. 1/3(pos1 + u0 + ourpos)
		// rather than at
		// e.g. 2/3 pos1 + 1/3 pos2
		//corner1 = 0.3333333*(pos1+u0+ourpos)
		//corner2 = 0.3333333*(pos2+u0+ourpos)
		//edgenormal.x = corner1.y-corner2.y = 0.333333(pos1.y-pos2.y) -- so no change here

		f64_vec2 edgenormal;
		edgenormal.x = (pos1.y-pos2.y)*0.333333333333333;
		edgenormal.y = (pos2.x-pos1.x)*0.333333333333333; // cut off 1/3 of the edge
		if (edgenormal.dot(pos0-pos1) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		
		// Think about averaging at typical edge.
		// Using 5/12:
		// corners are say equidistant from 3 points, so on that it would be 1/6
		// but allocate the middle half of the bar to 50/50 A_tri[threadIdx.x]+Aout.

		// tried without A_tri[threadIdx.x].z+ ...
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;

		// Now that we put minor corners at (1/3)(2 centroids+vertex), this makes even more sense.
				
		area += 0.333333333333333*
			(0.5*(pos1.x+pos2.x)+tri_centroid[threadIdx.x].x+u0.x)*edgenormal.x;

		// NOT CONSISTENT BEHAVIOUR:
			// TO HERE WAS ENOUGH TO FAIL.


		// ASSUMING ALL VALUES VALID (consider edge of memory a different case):
		//  From here on is where it gets thorny as we no longer map A_1 to vertex 1.
		// %%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%%

		A_1 = A_out; // now A_1 points at tri neigh 0
		A_out = A_2; // now looking at vertex 2
		// A_2 is now to point at tri neigh 1
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri + blockDim.x))
		{
			A_2 = A_tri[neightri.i2-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i2];
		}
		if (tri_rotate.per1 != 0) {
			if (tri_rotate.per1 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
		
	//	shoelace = (ourpos.x-pos2.x)*(u0.y-u1.y)
	//			 + (u0.x-u1.x)*(pos2.y-ourpos.y);  	

		//x1 = (2/3)pos2+(1/3)pos0;
		//x2 = (2/3)pos2+(1/3)pos1;
		//edgenormal.x = (x1.y-x2.y);
		//edgenormal.y = (x2.x-x1.x); // cut off 1/3 of the edge
		edgenormal.x = 0.333333333333333*(pos0.y-pos1.y);
		edgenormal.y = 0.333333333333333*(pos1.x-pos0.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos2-pos1) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;
		
		area += 0.333333333333333*(0.5*(u0.x+u1.x)
			+tri_centroid[threadIdx.x].x+pos2.x)*edgenormal.x;
		
		A_1 = A_out; // now A_1 points at corner 2
		A_out = A_2; // now points at tri 1
		// A_2 to point at corner 0
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i1-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i1];
		}
		if (perinfo.per0 != 0) {
			if (perinfo.per0 == NEEDS_ANTI) {
				A_2 = Anticlock_rotate3(A_2);
			}
			if (perinfo.per0 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			}							
		}
	  	
//shoelace = (ourpos.x-u1.x)*(pos2.y-pos0.y) // clock.y-anti.y
	//			 + (pos2.x-pos0.x)*(u1.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos0.y-pos2.y);
		edgenormal.y = 0.333333333333333*(pos2.x-pos0.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos1-pos0) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;
		
		area += 0.333333333333333*(0.5*(pos2.x+pos0.x)+tri_centroid[threadIdx.x].x+u1.x)*edgenormal.x;
		
		A_1 = A_out;
		A_out = A_2;
		// A_2 is now to point at tri neigh 2
		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri + blockDim.x))
		{
			A_2 = A_tri[neightri.i3-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i3];
		}
		if (tri_rotate.per2 != 0) {
			if (tri_rotate.per2 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
	//	f64 shoelace = (ourpos.x-pos0.x)*(u1.y-u2.y) // clock.y-anti.y
	//			 + (u1.x-u2.x)*(pos0.y-ourpos.y); 
			// Where is it used?

		edgenormal.x = 0.333333333333333*(pos1.y-pos2.y);
		edgenormal.y = 0.333333333333333*(pos2.x-pos1.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos0-pos1) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;
		area += 0.333333333333333*(0.5*(u2.x+u1.x)+tri_centroid[threadIdx.x].x+pos0.x)*edgenormal.x;
		
		A_1 = A_out;
		A_out = A_2;
		// A2 to be for corner 1
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i2-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i2];
		}
		if (perinfo.per1 != 0) {
			if (perinfo.per1 == NEEDS_ANTI) {
				A_2 = Anticlock_rotate3(A_2);
			}
			if (perinfo.per1 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			}							
		}
	  	
		//shoelace = (ourpos.x-u2.x)*(pos0.y-pos1.y) // clock.y-anti.y
//+ (pos0.x-pos1.x)*(u2.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos1.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos1.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos2-pos1) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;
		
		area += 0.333333333333333*(0.5*(pos0.x+pos1.x)+tri_centroid[threadIdx.x].x+u2.x)*edgenormal.x;
		
		A_1 = A_out;
		A_out = A_2;
		// A2 to be for tri 0
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + blockDim.x))
		{
			A_2 = A_tri[neightri.i1-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i1];
		}
		if (tri_rotate.per0 != 0) {
			if (tri_rotate.per0 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
		//shoelace = (ourpos.x-pos1.x)*(u2.y-u0.y) // clock.y-anti.y
		//		 + (u2.x-u0.x)*(pos1.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos2.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos2.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos1-pos2) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		gradAx += (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A_tri[threadIdx.x].x+A_out.x))*edgenormal;
		gradAy += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A_tri[threadIdx.x].y+A_out.y))*edgenormal;
		gradAz += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A_tri[threadIdx.x].z+A_out.z))*edgenormal;
		
		area += 0.333333333333333*(0.5*(u0.x+u2.x)+tri_centroid[threadIdx.x].x+pos1.x)*edgenormal.x;
		
		// CHECKED ALL THAT
		
		gradAx /= area;
		gradAy /= area;
		gradAz /= area;	
		
	}
	
	// Now we have to do something about anti-advecting:

	if ((perinfo.flag == DOMAIN_TRIANGLE) || (perinfo.flag == CROSSING_INS)) // otherwise the centroid can be assumed not moving??
	{
		f64_vec3 anti_Advect;
		anti_Advect.x = h*v_overall.dot(gradAx);
		anti_Advect.y = h*v_overall.dot(gradAy);
		anti_Advect.z = h*v_overall.dot(gradAz);
	
		p_A_out[index] = A_tri[threadIdx.x] + anti_Advect;
	} else {
		p_A_out[index] = A_tri[threadIdx.x];
	}

	// Similar routine will be needed to create grad A ... or Adot ... what a waste of calcs.
	// Is there a more sensible way: only do a mesh move every 10 steps -- ??
	// Then what do we do on the intermediate steps -- that's a problem -- flowing Eulerian fluid
	// will give the right change in pressure, but then mesh has to catch up. Still that might be a thought.
	
	// Next consideration: Lap A on central.
	
	// Idea for doing at same time: (don't do it -- too much atomicAdd, I do not trust)
	// ___ only certain major cells "belong" to this tri tile.
	// ___ write to a given output from our total effect coming from this tile's tris.
	// ____ when we hit a central cell outside this tile, send it atomicAdd to an array
	// that collects up all the extra contribs to it.
	// __ then we just reload, sum 2 things and divide
	// However, atomicAdd fp64 only exists on Compute 6.0 :-(
	// Workaround taken from http://stackoverflow.com/questions/16077464/atomicadd-for-double-on-gpu
	// Eventually decided not to use but to carry on with half the threads to target centrals in this routine.

	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE) {
		index = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x;		
		structural info = p_info[index];	

		if (info.flag == DOMAIN_VERTEX) {
			// Does branching disrupt contiguity???

			f64_vec2 v_overall = p_v_overall[BEGINNING_OF_CENTRAL + index];
			
			memcpy(IndexTri+threadIdx.x*MAXNEIGH_d, p_IndexTri+index*MAXNEIGH_d,
				sizeof(long)*MAXNEIGH_d);
			f64_vec3 A0 = A_vert[threadIdx.x]; // can ditch
			f64_vec2 u0 = vertex_pos[threadIdx.x];
			f64_vec3 A1(0.0,0.0,0.0),A2(0.0,0.0,0.0),A3(0.0,0.0,0.0);
			f64_vec2 u1(0.0,0.0),u2(1.1,1.1),u3(1.0,2.0);
			f64 //shoelace, 
				area = 0.0; 

			f64_vec2 edgenormal;
			
			// As before we need 4 A values and positions at a time. Now 3 all come from tris.
			gradAx.x = 0.0; gradAx.y = 0.0; 
			gradAy.x = 0.0; gradAy.y = 0.0; 
			gradAz.x = 0.0; gradAz.y = 0.0; 
						
			// Note that we found out, unroll can be slower if registers are used up (!) CAUTION:
			
			// Initial situation: inext = 1, i = 0, iprev = -1
			long iindextri = IndexTri[threadIdx.x*MAXNEIGH_d+info.neigh_len-1];
			// BEWARE OF FRILLED VERTCELLS: neigh_len < tri_len ??
			
			if ((iindextri >= StartTri) && (iindextri < StartTri +
				blockDim.x)) // matching code above to see what happens
				// threadsPerTileMinor))
			{
				// DOES NOT WORK WITH 2 LINES HERE.
				A3 = A_tri[iindextri-StartTri]; // this breaks it
				u3 = tri_centroid[iindextri-StartTri]; // this breaks it
			} else {
				A3 = p_A_tri[iindextri];
				u3 = p_tri_centroid[iindextri]; 
			};
			// The peculiar thing is that a very similar read happens earlier on.

			// INCONSISTENT BEHAVIOUR: now does not work with all above reads commented.
			// FAILS IF START COMMENT HERE
			
			if (info.has_periodic != 0) {
				if ((u3.x > u3.y*GRADIENT_X_PER_Y*0.5) && (u3.x < -0.5*GRADIENT_X_PER_Y*u3.y))
				{
					A3 = Anticlock_rotate3(A3);
					u3 = Anticlock_rotate2(u3);
				};
				if ((u3.x < -u3.y*GRADIENT_X_PER_Y*0.5) && (u3.x > 0.5*GRADIENT_X_PER_Y*u3.y))
				{
					A3 = Clockwise_rotate3(A3);
					u3 = Clockwise_rotate2(u3);
				};
			}
			
			iindextri = IndexTri[threadIdx.x*MAXNEIGH_d]; // + 0
			if ((iindextri >= StartTri) && (iindextri < StartTri + threadsPerTileMinor))
			{
				A2 = A_tri[iindextri-StartTri];
				u2 = tri_centroid[iindextri-StartTri];
			} else {
				A2 = p_A_tri[iindextri];
				u2 = p_tri_centroid[iindextri];
			};
			if (info.has_periodic != 0) {
				if ((u2.x > u2.y*GRADIENT_X_PER_Y*0.5) && (u2.x < -0.5*GRADIENT_X_PER_Y*u2.y))
				{
					A2 = Anticlock_rotate3(A2);
					u2 = Anticlock_rotate2(u2);
				};
				if ((u2.x < -u2.y*GRADIENT_X_PER_Y*0.5) && (u2.x > 0.5*GRADIENT_X_PER_Y*u2.y))
				{
					A2 = Clockwise_rotate3(A2);
					u2 = Clockwise_rotate2(u2);
				};
			}			
			
			int inext = 0; // will be ++ straight away.
						
#pragma unroll MAXNEIGH_d
			for (int i = 0; i < info.neigh_len; i++)  // WHY ARE WE GOING TO MAXNEIGH_d ?
			{
				inext++;
				if (inext == info.neigh_len) inext = 0; 
				// Bear in mind, this would not work for OUTERMOST.
					
				iindextri = IndexTri[threadIdx.x*MAXNEIGH_d+inext];
				if ((iindextri >= StartTri) && (iindextri < StartTri + threadsPerTileMinor))
				{
					A1 = A_tri[iindextri-StartTri];
					u1 = tri_centroid[iindextri-StartTri];
				} else {
					A1 = p_A_tri[iindextri];
					u1 = p_tri_centroid[iindextri];
				};
				if (info.has_periodic != 0) {
					if ((u1.x > 0.5*GRADIENT_X_PER_Y*u1.y) && (u1.x < -0.5*GRADIENT_X_PER_Y*u1.y))
					{
						A1 = Anticlock_rotate3(A1);
						u1 = Anticlock_rotate2(u1);
					};
					if ((u1.x < -0.5*GRADIENT_X_PER_Y*u1.y) && (u1.x > 0.5*GRADIENT_X_PER_Y*u1.y))
					{
						A1 = Clockwise_rotate3(A1);
						u1 = Clockwise_rotate2(u1);
					};
				}
				
				// So how are we going to get the corners of central cell?
				// Do we change the plan and make them the average of 2 tri centroids and the vertex?
				// That is one way, not sure I'm keen on it, not having thought about it.
				// YES, that is what we have to do.
				
				// ==============
				
				//	edge_cnr1 = (u1+u2+u0)*0.333333333333333;
				//	edge_cnr2 = (u3+u2+u0)*0.333333333333333;
				edgenormal.x = 0.333333333333333*(u1.y-u3.y);
				edgenormal.y = 0.333333333333333*(u3.x-u1.x);
				// edgenormal to point at u2:
				if ((u2-u1).dot(edgenormal) < 0.0)
				{
					edgenormal.x = -edgenormal.x; 
					edgenormal.y = -edgenormal.y;
				}
				//shoelace = (u0.x-u2.x)*(u1.y-u3.y) +
				//	       (u1.x-u3.x)*(u2.y-u0.y);
			
				gradAx += (TWELTH*(A1.x+A3.x)+FIVETWELTHS*(A0.x+A2.x))*edgenormal;
				gradAy += (TWELTH*(A1.y+A3.y)+FIVETWELTHS*(A0.y+A2.y))*edgenormal;
				gradAz += (TWELTH*(A1.z+A3.z)+FIVETWELTHS*(A0.z+A2.z))*edgenormal;

				area += (0.3333333333333333*(0.5*(u1.x+u3.x)+u2.x+u0.x))*edgenormal.x; 
				// ( grad x )_x
				
				// move round A values and positions:
				// ----------------------------------
				A3 = A2;
				u3 = u2;
				A2 = A1;
				u2 = u1;
			}

			gradAx /= area;
			gradAy /= area;
			gradAz /= area;
			
			f64_vec3 anti_Advect;
			anti_Advect.x = h*v_overall.dot(gradAx);
			anti_Advect.y = h*v_overall.dot(gradAy);
			anti_Advect.z = h*v_overall.dot(gradAz);

			// Save off: 
			if (bAdd) {
				anti_Advect += h*p_Addition_Rate[BEGINNING_OF_CENTRAL + index];
			}
			p_A_out[BEGINNING_OF_CENTRAL + index] = A_vert[threadIdx.x] + anti_Advect; // best way may be: if we know start of central stuff, can send
			
		} else { // ONLY FOR DOMAIN VERTEX
			p_A_out[BEGINNING_OF_CENTRAL + index] = A_vert[threadIdx.x];
		};
	}; // IS THREAD IN THE FIRST HALF OF THE BLOCK
	

	// =============================================================================
	// Understand the following important fact:
	// If you will use 63 registers (and this routine surely will - 
	// we have positions 7 x 2 x 2 = 28 registers, A 7 x 3 x 2 = 35 registers
	// -- though we could try taking into account, 1 dimension at a time)
	// Then the max thread throughput per SM is 512 which means that we will get
	// no penalty from using up to 12 doubles in shared memory per thread.
	// =============================================================================
	// That does mean L1 has room for only 4 doubles. It is not big compared to registry itself.
}

__global__ void Kernel_Compute_Lap_A_and_Grad_A_to_get_B_on_all_minor(
			f64_vec3 * __restrict__ p_A_tri,
			f64_vec3 * __restrict__ p_A_vert,
			structural * __restrict__ p_info,
			f64_vec2 * __restrict__ p_tri_centroid,
			CHAR4 * __restrict__ p_tri_perinfo,
			CHAR4 * __restrict__ p_tri_per_neigh,
			LONG3 * __restrict__ p_corner_index,
			LONG3 * __restrict__ p_neigh_tri_index,
			long * __restrict__ p_IndexTri,

			// output:
			f64_vec3 * __restrict__ p_Lap_A,
			f64_vec3 * __restrict__ p_Lap_A_central,
			f64_vec3 * __restrict__ p_B,
			f64_vec3 * __restrict__ p_B_central // could just infer
			)
{
	// The logic here. Lap A requires A on quadrilateral over each honey-edge.
	// Therefore we need both tri and vertex values of A at once.
	// The same applies for Lap_A_central as for Lap_A_tri.
	// Therefore we carry on to do Lap_A_central using the same data ; in fact we can
	// avoid loading Indextri because we work on the result in shared memory as we are doing tris.

	__shared__ f64_vec3 A_tri[threadsPerTileMinor];
	__shared__ f64_vec2 tri_centroid[threadsPerTileMinor];				// 5
	__shared__ f64_vec3 A_vert[SIZE_OF_MAJOR_PER_TRI_TILE]; // +1.5
	__shared__ f64_vec2 vertex_pos[SIZE_OF_MAJOR_PER_TRI_TILE];// altogether 9 doubles per thread so far here.
//	__shared__ short shared_per[threadsPerTileMinor]; // short easier to access than char maybe.
	
	// If we want 512 threads/SM, 12 doubles in shared per thread is limit.
	// We can accommodate 12 .. so 6 per major in addition to this but not when we have shared_per.
	// Well we could limit it to 10 tris but it's asking for trouble.
	// 6 longs = 3 doublesworth per thread
 
	__shared__ long IndexTri[SIZE_OF_MAJOR_PER_TRI_TILE*MAXNEIGH_d];
	// __shared__ char PBCtri[SIZE_OF_MAJOR_PER_TRI_TILE*MAXNEIGH_d];
	// Total 3+2+1.5+1+3 = 11.5 -- it ought to work -- plus shared_per. Even if that counts
	// as a long, we still just about get it.
	// It OUGHT still to run 2 blocks per SM.
	// Only half the threads will continue to the 2nd part. But on the other hand,
	// if each major thread has 6 (ind) + 2*5 + 5 = 21+ doubles, only 256 of those can run.
	// Anything else needed? Yes - the list of chars -- which is 6 bytes per thread here
	// and thus makes this all too chancy.
	// Go with the unintelligent way -- two separate routines ??
	// Note that if a triangle is not periodic itself then it's impossible that its data
	// needs to be rotated for central, since central is a corner of the triangle.
	// Therefore we can consult shared_per list instead. Okay but what if a tri is not in the list?
	// 
	
	// __shared__ f64_vec3 Lap_A_central[SIZE_OF_MAJOR_PER_TRI_TILE]; // +1.5
	
			// Let's hope atomicAdd to shared isn't as bad as we expect.
			// https://devtalk.nvidia.com/default/topic/514085/cuda-programming-and-performance/atomicadd-in-shared-memory-is-measured-slower-than-in-global-memory-timing-shared-memory-atomic-o/
			// says that it's AWFUL.

	// Factors against doing:
	// _ Must take account of periodic in applying effect from this side
	// _ Must do atomic add to an extra array to avoid conflicting with other blocks' contribs
	// _ Must do atomic add within shared memory to avoid conflicting with other threads' contribs
	
	// We got so far 7.5 to 8 doubles per go.
	// Do we want to add 3 more, put the routine following this. Yes.

	// do we also want tri centroid? probably yes really
	// Do we need shared flags?
	
	// Note that we do not need to do 3D at all to do LAP - this is something where we can use 1 dimension at a time.
	// Problem with that, we'd have to load A all over again. 
	// We could try it both ways.
	// Stripping back the solution to 1D at a time, is probably just tinkering at the edges.
	// The only thing worth comparing is if we do both that AND reload values 3x to do Ax,Ay,Az separately.
	
	// Now bear in mind: if 10 doubles is a lot for shared that is 48K, 5 doubles is already a lot for L1.
	
	// Do first with 3 dimensions Axyz at once - may be slower but we'll see.
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	long StartTri = blockIdx.x*blockDim.x; // can replace this one.
	long StartMajor = SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x; // can replace this one.
	// could replace with #define here.	
	A_tri[threadIdx.x] = p_A_tri[index];
	tri_centroid[threadIdx.x] = p_tri_centroid[index];	
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE) {
		A_vert[threadIdx.x] = p_A_vert[SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x + threadIdx.x];
		structural info = p_info[SIZE_OF_MAJOR_PER_TRI_TILE*blockIdx.x + threadIdx.x];
		vertex_pos[threadIdx.x] = info.pos;
		
	//	f64_vec3 zero(0.0,0.0,0.0);
	//	Lap_A_central[threadIdx.x] = zero;
		// To save Lap A central solution we'd need to send it to the array per this tile's colour
		// and then aggregate the results, divide by shoelace?		
	}
	
//	shared_per[threadIdx.x] = perinfo.per0+perinfo.per1+perinfo.per2; // if periodic tri then neigh will need to be able to know.
	__syncthreads();
	
	// perinfo is still in scope later on but we'd rather get rid of it.
	// The construction here is so we can get it before syncthreads, which is awkward.
	
	f64_vec3 LapA(0.0,0.0,0.0);
	f64_vec3 B(0.0,0.0,0.0);
	CHAR4 perinfo = p_tri_perinfo[index];
		
	if ((perinfo.flag != OUTER_FRILL) && (perinfo.flag != INNER_FRILL))
	{
		// We may need to find a way to AVOID doing branch around memory accesses.
		// For frills, we would get a division by zero I'd expect.
		// We probably should be splitting out tri vs central.

		f64_vec2 edgenormal; // moving this inside unfortunately did not make any gains at all.
		
		LONG3 corner_index = p_corner_index[index];
		LONG3 neightri = p_neigh_tri_index[index]; 
		CHAR4 tri_rotate = p_tri_per_neigh[index];
		
		// Note that A, as well as position, has to be ROTATED to make a contiguous image.
		// This tri minor has 3 edges with triangles and 3 edges with centrals.
		
		// To accumulate Lap_A_central at the same time:
		// * We should colour the blocks so that no two colours are shared by 1 major. That is possible.	
		// * The block outputs to its own colour array of centrals affected.	
		// * Then we aggregate the colour arrays. 
		
		// @@@@@@@@@@@@@@@@		
		// Now consider another one: what if we launched 3 threads per triangle. Same shared data for block as here.
		// Does that really help anything? Think no.
		// We need to divide by area when we've done something.
		f64 area = 0.0;
		
		f64_vec2 pos0(0.0,0.0), pos1(1.0,0.0), pos2(0.0,1.0);
		
		// DEBUG: COMMENTING FROM HERE IT WORKED.


	//	f64_vec3 Avert0,Avert1,Avert2;
		// We need 4 values at a time in order to do a side.
		// We don't need to have all 7 values (3+ 3 + itself)
		// So we'd be better just to load one quadrilateral's doings at a time, given the paucity of register and L1 space.
		// Either we store all 7x3 A-values at once + 7 positions. Or we use 4 and 4 positions at once.
		// Bear in mind a partial saving might yield big benefits.
		// HAZARD: we don't know the ordering of the points. 
		
		// Halfway house: for simplicity store all the positions already loaded.
		// A does not load from the same place anyway.
		// Then go round the quadrilaterals.
		
		// THIS BIT IS ENOUGH TO CRASH IT:
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos0 = vertex_pos[corner_index.i1-StartMajor]; // this line okay
		} else {
			structural info = p_info[corner_index.i1];
			pos0 = info.pos;	
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos1 = vertex_pos[corner_index.i2-StartMajor];
		} else {
			structural info = p_info[corner_index.i2];
			pos1 = info.pos;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos2 = vertex_pos[corner_index.i3-StartMajor];
		} else {
			structural info = p_info[corner_index.i3];
			pos2 = info.pos;
		}
		
		char periodic = perinfo.per0 + perinfo.per1 + perinfo.per2;
		
		if (periodic == 0) {
		} else {
			// In this case which ones are periodic?
			// Should we just store per flags?
			// How it should work:
			// CHAR4 perinfo: periodic, per0, per1, per2;
			if (perinfo.per0 == NEEDS_ANTI) {
				pos0 = Anticlock_rotate2(pos0);
		//		Avert0 = Anticlock_rotate2(Avert0);
			}
			if (perinfo.per0 == NEEDS_CLOCK) {
				pos0 = Clockwise_rotate2(pos0);
		//		Avert0 = Clockwise_rotate2(Avert0);
			}
			if (perinfo.per1 == NEEDS_ANTI) {
				pos1 = Anticlock_rotate2(pos1);
		//		Avert1 = Anticlock_rotate2(Avert1);
			}
			if (perinfo.per1 == NEEDS_CLOCK) {
				pos1 = Clockwise_rotate2(pos1);
		//		Avert1 = Clockwise_rotate2(Avert1);
			}
			if (perinfo.per2 == NEEDS_ANTI) {
				pos2 = Anticlock_rotate2(pos2);
		//		Avert2 = Anticlock_rotate2(Avert2);
			}
			if (perinfo.per2 == NEEDS_CLOCK) {
				pos2 = Clockwise_rotate2(pos2);
		//		Avert2 = Clockwise_rotate2(Avert2);
			}
		};
		
		f64_vec2 u0(0.0,0.0),u1(1.0,1.0),u2(1.0,3.0); // to be the positions of neighbouring centroids
			
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			u0 = tri_centroid[neightri.i1-StartTri];
		} else {
			u0 = p_tri_centroid[neightri.i1];
		}
		if (tri_rotate.per0 == NEEDS_CLOCK) {
			u0 = Clockwise_rotate2(u0);
		}
		if (tri_rotate.per0 == NEEDS_ANTI) {
			u0 = Anticlock_rotate2(u0);
		}

		// Am I correct that this is to avoid tri_neigh_per information being recorded...
		
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri + threadsPerTileMinor))
		{
			u1 = tri_centroid[neightri.i2-StartTri];
		} else {
			u1 = p_tri_centroid[neightri.i2];
		}
		if (tri_rotate.per1 == NEEDS_CLOCK) {
			u1 = Clockwise_rotate2(u1);
		}
		if (tri_rotate.per1 == NEEDS_ANTI) {
			u1 = Anticlock_rotate2(u1);
		}


		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri + threadsPerTileMinor))
		{
			u2 = tri_centroid[neightri.i3-StartTri];
		} else {
			u2 = p_tri_centroid[neightri.i3];
		}	
		if (tri_rotate.per2 == NEEDS_CLOCK) {
			u2 = Clockwise_rotate2(u2);
		}
		if (tri_rotate.per2 == NEEDS_ANTI) {
			u2 = Anticlock_rotate2(u2);
		}


		if (index == 73250)
		{
			printf("u0 1 2 %1.6E %1.6E ,  %1.6E %1.6E ,  %1.6E %1.6E \n",
					u0.x,u0.y,u1.x,u1.y,u2.x,u2.y);
			printf("pos0 1 2  %1.6E %1.6E ,  %1.6E %1.6E ,  %1.6E %1.6E \n",
					pos0.x,pos0.y,pos1.x,pos1.y,pos2.x,pos2.y);
		}

		// ............................................................................................
		
		// . I think working round with 4 has a disadvantage: if we get back around to one that is off-tile,
		// we have to load it all over again. Still that is only 1 out of 7 that gets duplicated.
		// Here is the best thing I can come up with: store 7 positions. That is already
		// 28 longs' worth... each A-value uses 6 of the 7 positions to have an effect.
		// Load each A-value at a time and recalc shoelace for 3 quadrilaterals. ??
			// Too complicated.
		// If we store all positions, can we finish with each A as we handle it? Yes but let's not.
		
		f64_vec2 ourpos = tri_centroid[threadIdx.x]; // can try with and without this assignment to variable
		f64_vec3 A0 = A_tri[threadIdx.x]; // can try with and without this assignment to variable
		f64_vec3 A_1(0.0,0.0,0.0),
			     A_out(1.0,2.0,3.0),
				 A_2(4.0,5.0,6.0);
		
		// Our A: A_tri[threadIdx.x]
		
		// Now fill in the A values:
		// ____________________________
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_1 = A_vert[corner_index.i2-StartMajor];
		} else {
			A_1 = p_A_vert[corner_index.i2];
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i3-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i3];
		}
		
		if (perinfo.per1 == NEEDS_ANTI) {
			A_1 = Anticlock_rotate3(A_1);
		}
		if (perinfo.per1 == NEEDS_CLOCK) {
			A_1 = Clockwise_rotate3(A_1);
		}
		if (perinfo.per2 == NEEDS_ANTI) {
			A_2 = Anticlock_rotate3(A_2); 
		};
		if (perinfo.per2 == NEEDS_CLOCK) {
			A_2 = Clockwise_rotate3(A_2);
		};
		
		
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			A_out = A_tri[neightri.i1-StartTri];
		} else {
			A_out = p_A_tri[neightri.i1];
		}
		if (tri_rotate.per0 != 0) {
			if (tri_rotate.per0 == NEEDS_CLOCK) {
				A_out = Clockwise_rotate3(A_out);
			} else {
				A_out = Anticlock_rotate3(A_out);
			};
		};

		// ======================================================
		
		f64 shoelace = (ourpos.x-u0.x)*(pos1.y-pos2.y)
				 + (pos1.x-pos2.x)*(u0.y-ourpos.y); // if u0 is opposite point 0
					// clock.x-anti.x
		
		// We are now going to put the corners of the minor cell at 
		// e.g. 1/3(pos1 + u0 + ourpos)
		// rather than at
		// e.g. 2/3 pos1 + 1/3 pos2
		//corner1 = 0.3333333*(pos1+u0+ourpos)
		//corner2 = 0.3333333*(pos2+u0+ourpos)
		//edgenormal.x = corner1.y-corner2.y = 0.333333(pos1.y-pos2.y) -- so no change here
		
		edgenormal.x = (pos1.y-pos2.y)*0.333333333333333;
		edgenormal.y = (pos2.x-pos1.x)*0.333333333333333; // cut off 1/3 of the edge
		if (edgenormal.dot(pos0-pos1) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		// note: same coeff to A0->grad_x as to x0 in shoelace:
		f64 coeff = ((pos1.y-pos2.y)*edgenormal.x +
					 (pos2.x-pos1.x)*edgenormal.y)/shoelace;
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		
		coeff = ((u0.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-u0.x)*edgenormal.y)/shoelace; // from top line same
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);

		// Think about averaging at typical edge.
		// Using 5/12:
		// corners are say equidistant from 3 points, so on that it would be 1/6
		// but allocate the middle half of the bar to 50/50 A0+Aout.

		// Bx = dAz/dy
		//B.x += Az_edge*edgenormal.y;
		// By = -dAz/dx
		//B.y += -Az_edge*edgenormal.x;
		// Bz = dAy/dx-dAx/dy
		//B.z += Ay_edge*edgenormal.x-Ax_edge*edgenormal.y;
		
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;
		// Now that we put minor corners at (1/3)(2 centroids+vertex), this makes even more sense.
		
		area += 0.333333333333333*(0.5*(pos1.x+pos2.x)+ourpos.x+u0.x)*edgenormal.x;
						
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		// ASSUMING ALL VALUES VALID (consider edge of memory a different case):
		//  From here on is where it gets thorny as we no longer map A_1 to vertex 1.
		// %%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%$$%%%
		
		A_1 = A_out; // now A_1 points at tri neigh 0
		A_out = A_2; // now looking at vertex 2
		// A_2 is now to point at tri neigh 1
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri + threadsPerTileMinor))
		{
			A_2 = A_tri[neightri.i2-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i2];
		}
		if (tri_rotate.per1 != 0) {
			if (tri_rotate.per1 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
		
		shoelace = (ourpos.x-pos2.x)*(u0.y-u1.y)
				 + (u0.x-u1.x)*(pos2.y-ourpos.y);  // can insert formula instead of creating var.	
		
		//x1 = (2/3)pos2+(1/3)pos0;
		//x2 = (2/3)pos2+(1/3)pos1;
		//edgenormal.x = (x1.y-x2.y);
		//edgenormal.y = (x2.x-x1.x); // cut off 1/3 of the edge
		edgenormal.x = 0.333333333333333*(pos0.y-pos1.y);
		edgenormal.y = 0.333333333333333*(pos1.x-pos0.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos2-pos1) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		coeff = ((u0.y-u1.y)*edgenormal.x +
				 (u1.x-u0.x)*edgenormal.y)/shoelace;  // This is correct - see coeff in shoelace on ourpos.y
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		
		
		//// Now do contribution to Lap A central for vertex 2:
		//if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		//{
		//	f64_vec3 addition = coeff*(A0-A_out);
		//	atomicAdd((double *)(Lap_A_solution+neightri.i2-StartTri), addition.x);
		//	atomicAdd((double *)(Lap_A_solution+neightri.i2-StartTri)+1, addition.y);
		//	atomicAdd((double *)(Lap_A_solution+neightri.i2-StartTri)+2, addition.z);
		//	// Will this simultaneously be affected by other threads? YES
		//	// So have to use atomicAdd on shared memory.
		//	
		//	// I guess we learned our lesson: it really is more of a headache to do this way
		//	// than just to write a whole separate routine for central cells.
		//	// !
		//	// the workaround atomicAdd will make it slow because of converting to long-long ?
		//	// So this is probably slower than recreating the whole routine and calculating again.
		//	// :-(
		//} else {
		//	f64_vec3 addition = coeff*(A0-A_out);
		//	atomicAdd((double *)(Lap_A_extra_array+neightri), addition.x);

		//	// We forgot something ELSE:
		//	// we have to take into account periodic orientation as well!

		//	// Okay let's scrap this attempt to create central at same time.

		//	// Unfortunately I do not see a way to overwrite part of shared memory with indices
		//	// either.
		//}

		// A_1 ~ u0, A_2 ~ u1
		coeff = ((pos2.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-pos2.x)*edgenormal.y)/shoelace;
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);
		
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;
		
		area += 0.333333333333333*(0.5*(u0.x+u1.x)+ourpos.x+pos2.x)*edgenormal.x;
						
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		A_1 = A_out; // now A_1 points at corner 2
		A_out = A_2; // now points at tri 1
		// A_2 to point at corner 0
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i1-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i1];
		}
		if (perinfo.per0 != 0) {
			if (perinfo.per0 == NEEDS_ANTI) {
				A_2 = Anticlock_rotate3(A_2);
			}
			if (perinfo.per0 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			}									
		}
	  	
		shoelace = (ourpos.x-u1.x)*(pos2.y-pos0.y) // clock.y-anti.y
				 + (pos2.x-pos0.x)*(u1.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos0.y-pos2.y);
		edgenormal.y = 0.333333333333333*(pos2.x-pos0.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos1-pos0) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		coeff = ((pos2.y-pos0.y)*edgenormal.x +
				 (pos0.x-pos2.x)*edgenormal.y)/shoelace; // see coeffs on ourpos in shoelace
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		// A_1~pos2 A_2~pos0
		coeff = ((u1.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-u1.x)*edgenormal.y)/shoelace; // something suspicious: that we had to change smth here.
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;
		
		area += 0.333333333333333*(0.5*(pos2.x+pos0.x)+ourpos.x+u1.x)*edgenormal.x;
				
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		A_1 = A_out;
		A_out = A_2;
		// A_2 is now to point at tri neigh 2
		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri + threadsPerTileMinor))
		{
			A_2 = A_tri[neightri.i3-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i3];
		}
		if (tri_rotate.per2 != 0) {
			if (tri_rotate.per2 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
		shoelace = (ourpos.x-pos0.x)*(u1.y-u2.y) // clock.y-anti.y
				 + (u1.x-u2.x)*(pos0.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos1.y-pos2.y);
		edgenormal.y = 0.333333333333333*(pos2.x-pos1.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos0-pos1) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		coeff = ((u1.y-u2.y)*edgenormal.x +
				 (u2.x-u1.x)*edgenormal.y)/shoelace; // see coeffs on ourpos in shoelace
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		// A_1~u1 A_2~u2
		coeff = ((pos0.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-pos0.x)*edgenormal.y)/shoelace; // something suspicious: that we had to change smth here.
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;

		area += THIRD*(0.5*(u2.x+u1.x)+ourpos.x+pos0.x)*edgenormal.x;
				
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		A_1 = A_out;
		A_out = A_2;
		// A2 to be for corner 1
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			A_2 = A_vert[corner_index.i2-StartMajor];
		} else {
			A_2 = p_A_vert[corner_index.i2];
		}
		if (perinfo.per1 != 0) {
			if (perinfo.per1 == NEEDS_ANTI) {
				A_2 = Anticlock_rotate3(A_2);
			}
			if (perinfo.per1 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			}									// CAREFUL WITH FLAGS N MEANINGS
		}
	  	
		shoelace = (ourpos.x-u2.x)*(pos0.y-pos1.y) // clock.y-anti.y
				 + (pos0.x-pos1.x)*(u2.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos1.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos1.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos2-pos1) > 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		coeff = ((pos0.y-pos1.y)*edgenormal.x +
				 (pos1.x-pos0.x)*edgenormal.y)/shoelace; // see coeffs on ourpos in shoelace
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		// A_1~pos0 A_2~pos1
		coeff = ((u2.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-u2.x)*edgenormal.y)/shoelace; // something suspicious: that we had to change smth here.
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;

		area += 0.333333333333333*(0.5*(pos0.x+pos1.x)+ourpos.x+u2.x)*edgenormal.x;
				
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		A_1 = A_out;
		A_out = A_2;
		// A2 to be for tri 0
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			A_2 = A_tri[neightri.i1-StartTri];
		} else {
			A_2 = p_A_tri[neightri.i1];
		}
		if (tri_rotate.per0 != 0) {
			if (tri_rotate.per0 == NEEDS_CLOCK) {
				A_2 = Clockwise_rotate3(A_2);
			} else {
				A_2 = Anticlock_rotate3(A_2);
			};
		};
		shoelace = (ourpos.x-pos1.x)*(u2.y-u0.y) // clock.y-anti.y
				 + (u2.x-u0.x)*(pos1.y-ourpos.y); 
		edgenormal.x = 0.333333333333333*(pos2.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos2.x); // cut off 1/3 of the edge
		if (edgenormal.dot(pos1-pos2) < 0.0) {
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;    
		};
		coeff = ((u2.y-u0.y)*edgenormal.x +
				 (u0.x-u2.x)*edgenormal.y)/shoelace; // see coeffs on ourpos in shoelace
		LapA.x += coeff*(A0.x-A_out.x);
		LapA.y += coeff*(A0.y-A_out.y);
		LapA.z += coeff*(A0.z-A_out.z);
		// A_1~pos0 A_2~pos1
		coeff = ((pos1.y-ourpos.y)*edgenormal.x +
				 (ourpos.x-pos1.x)*edgenormal.y)/shoelace; // something suspicious: that we had to change smth here.
		LapA.x += coeff*(A_1.x-A_2.x);
		LapA.y += coeff*(A_1.y-A_2.y);
		LapA.z += coeff*(A_1.z-A_2.z);
		B.x += (TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.y;
		B.y += -(TWELTH*(A_1.z+A_2.z)+FIVETWELTHS*(A0.z+A_out.z))*edgenormal.x;
		B.z += (TWELTH*(A_1.y+A_2.y)+FIVETWELTHS*(A0.y+A_out.y))*edgenormal.x
			 - (TWELTH*(A_1.x+A_2.x)+FIVETWELTHS*(A0.x+A_out.x))*edgenormal.y;
		
		area += 0.333333333333333*(0.5*(u0.x+u2.x)+ourpos.x+pos1.x)*edgenormal.x;
						
		if (index == 73250)
			printf("LapAx_integ _ %1.6E ; area _ %1.6E ; A_1.x %1.6E %1.6E %1.6E \n",LapA.x,area,A_1.x,A_out.x,A_2.x);

		// CHECKED ALL THAT 
		
		// Heavy calcs are actually here: six divisions!
		LapA /= (area);
		B /= (area);		
		
		if (index == 73250)
			printf("result %1.8E \n",LapA.x);
		
	} else {
		// frill - leave Lap A = B = 0
	}
	p_Lap_A[index] = LapA;
	p_B[index] = B;
	
	// Similar routine will be needed to create grad A ... or Adot ... what a waste of calcs.
	// Is there a more sensible way: only do a mesh move every 10 steps -- ??
	// Then what do we do on the intermediate steps -- that's a problem -- flowing Eulerian fluid
	// will give the right change in pressure, but then mesh has to catch up. Still that might be a thought.
	
	// Next consideration: Lap A on central.
	
	// Idea for doing at same time: (don't do it -- too much atomicAdd, I do not trust)
	// ___ only certain major cells "belong" to this tri tile.
	// ___ write to a given output from our total effect coming from this tile's tris.
	// ____ when we hit a central cell outside this tile, send it atomicAdd to an array
	// that collects up all the extra contribs to it.
	// __ then we just reload, sum 2 things and divide
	// However, atomicAdd fp64 only exists on Compute 6.0 :-(
	// Workaround taken from http://stackoverflow.com/questions/16077464/atomicadd-for-double-on-gpu
	// Eventually decided not to use but to carry on with half the threads to target centrals in this routine.
	
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE) {
		// Create Lap A for centrals.
		// Outermost has to supply good boundary conditions for the outer edge.
		
		index = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x;
		
		structural info = p_info[index];
		memcpy(IndexTri+threadIdx.x*MAXNEIGH_d, p_IndexTri+index*MAXNEIGH_d,
			sizeof(long)*MAXNEIGH_d);
		f64_vec3 A0 = A_vert[threadIdx.x]; // can ditch
		f64_vec2 u0 = vertex_pos[threadIdx.x];
		f64_vec3 A1,A2,A3;
		f64_vec2 u1,u2,u3;
		f64 shoelace, area = 0.0;
		f64_vec2 edgenormal;
		
		// As before we need 4 A values and positions at a time. Now 3 all come from tris.
		LapA.x = 0.0; LapA.y = 0.0; LapA.z = 0.0;
		B.x = 0.0; B.y = 0.0; B.z = 0.0;
		// Note that I found out, unroll can be slower if registers are used up (!) CAUTION:
		long iindextri;
		
		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
			// In this case there are extra triangles, for frills.
			iindextri = IndexTri[threadIdx.x*MAXNEIGH_d+info.neigh_len];
		} else {
			iindextri = IndexTri[threadIdx.x*MAXNEIGH_d+info.neigh_len-1];
		}
		if ((iindextri >= StartTri) && (iindextri < StartTri + threadsPerTileMinor))
		{
			A3 = A_tri[iindextri-StartTri];
			u3 = tri_centroid[iindextri-StartTri];
		} else {
			A3 = p_A_tri[iindextri];
			u3 = p_tri_centroid[iindextri];
		};
		if (info.has_periodic != 0) {
			if ((u3.x > 0.5*GRADIENT_X_PER_Y*u3.y) && (u0.x < -0.5*GRADIENT_X_PER_Y*u0.y))
			{
				A3 = Anticlock_rotate3(A3);
				u3 = Anticlock_rotate2(u3);
			};
			if ((u3.x < -0.5*GRADIENT_X_PER_Y*u3.y) && (u0.x > 0.5*GRADIENT_X_PER_Y*u0.y))
			{
				A3 = Clockwise_rotate3(A3);
				u3 = Clockwise_rotate2(u3);
			};
		}
		
		// Initial situation: inext = 1, i = 0, iprev = -1
		iindextri = IndexTri[threadIdx.x*MAXNEIGH_d]; // + 0
		if ((iindextri >= StartTri) && (iindextri < StartTri + threadsPerTileMinor))
		{
			A2 = A_tri[iindextri-StartTri];
			u2 = tri_centroid[iindextri-StartTri];
		} else {
			A2 = p_A_tri[iindextri];
			u2 = p_tri_centroid[iindextri];
		};		
		if (info.has_periodic != 0) {
			if ((u2.x > 0.5*GRADIENT_X_PER_Y*u2.y) && (u0.x < -0.5*GRADIENT_X_PER_Y*u0.y))
			{
				A2 = Anticlock_rotate3(A2);
				u2 = Anticlock_rotate2(u2);
			};
			if ((u2.x < -0.5*GRADIENT_X_PER_Y*u2.y) && (u0.x > 0.5*GRADIENT_X_PER_Y*u0.y))
			{
				A2 = Clockwise_rotate3(A2);
				u2 = Clockwise_rotate2(u2);
			};
		}
		
		short limit = info.neigh_len;
		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) 
			limit++;
		// Ordinarily, number of tri pairs = number of tris = number of neighs
		// For outermost, number of neighs = 4 but the number of tri pairs to use = 5.
		// Now we attempt to go all the way round: A and u from frills are valid and we can
		// form a quadrilateral
		
		int inext = 0; // will be ++ straight away.
#pragma unroll MAXNEIGH_d
		for (short i = 0; i < limit; i++)
		{
			inext++;
			if (inext == limit) inext = 0; 
				
			iindextri = IndexTri[threadIdx.x*MAXNEIGH_d+inext];
			if ((iindextri >= StartTri) && (iindextri < StartTri + threadsPerTileMinor))
			{
				A1 = A_tri[iindextri-StartTri];
				u1 = tri_centroid[iindextri-StartTri];
			} else {
				A1 = p_A_tri[iindextri];
				u1 = p_tri_centroid[iindextri];
			};
			if (info.has_periodic != 0) {
				if ((u1.x > 0.5*GRADIENT_X_PER_Y*u1.y) && (u0.x < -0.5*GRADIENT_X_PER_Y*u0.y))
				{
					A1 = Anticlock_rotate3(A1);
					u1 = Anticlock_rotate2(u1);
				};
				if ((u1.x < -0.5*GRADIENT_X_PER_Y*u1.y) && (u0.x > 0.5*GRADIENT_X_PER_Y*u0.y))
				{
					A1 = Clockwise_rotate3(A1);
					u1 = Clockwise_rotate2(u1);
				};
			}
					
			// Affect LapA,B:
			// ==============
			
			//	edge_cnr1 = (u1+u2+u0)*0.333333333333333;
			edgenormal.x = 0.333333333333333*(u1.y-u3.y);
			edgenormal.y = 0.333333333333333*(u3.x-u1.x);
			// edgenormal to point at u2:
			if ((u2-u1).dot(edgenormal) < 0.0)
			{
				edgenormal.x=-edgenormal.x; edgenormal.y = -edgenormal.y;
			}
			shoelace = (u0.x-u2.x)*(u1.y-u3.y) +
				       (u1.x-u3.x)*(u2.y-u0.y);
			//coeff = ((u1.y-u3.y)*edgenormal.x + (u3.x-u1.x)*edgenormal.y)/shoelace;
			//LapA += coeff*(A0-A2);
			LapA += (A0-A2)*(((u1.y-u3.y)*edgenormal.x + (u3.x-u1.x)*edgenormal.y)/shoelace);
			//coeff = ((u2.y-u0.y)*edgenormal.x + (u0.x-u2.x)*edgenormal.y)/shoelace;
			LapA += (A1-A3)*(((u2.y-u0.y)*edgenormal.x + (u0.x-u2.x)*edgenormal.y)/shoelace);
			
			B.x +=  (TWELTH*(A1.z+A3.z)+FIVETWELTHS*(A0.z+A2.z))*edgenormal.y;
		    B.y += -(TWELTH*(A1.z+A3.z)+FIVETWELTHS*(A0.z+A2.z))*edgenormal.x;
		    B.z +=  (TWELTH*(A1.y+A3.y)+FIVETWELTHS*(A0.y+A2.y))*edgenormal.x
			       -(TWELTH*(A1.x+A3.x)+FIVETWELTHS*(A0.x+A2.x))*edgenormal.y;
			area += (0.3333333333333333*(0.5*(u1.x+u3.x)+u2.x+u0.x))*edgenormal.x; 
			// ( grad x )_x
			
			if (index + BEGINNING_OF_CENTRAL == 110500)  
			{
				printf("%d LapA.x %1.6E area %1.6E A %1.6E %1.6E %1.6E %1.6E u %1.6E %1.6E , %1.6E %1.6E , %1.6E %1.6E  , %1.6E %1.6E \n",index,
					LapA.x,area,A0.x,A1.x,A2.x,A3.x,u0.x,u0.y,u1.x,u1.y,u2.x,u2.y,u3.x,u3.y);
			};
			
			// move round A values and positions:
			// ----------------------------------
			A3 = A2;
			u3 = u2;
			A2 = A1;
			u2 = u1;
		}
	
	//	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
			// Add on the outer edge: dA/dr times length of edge.
			// dAx/dr = -(Ax r)/r^2 = -Ax/r
	
		// We find a way just to go all the way round instead.

		LapA /= area;
		B /= area;
		// Save off:
		p_Lap_A_central[index] = LapA; // best way may be: if we know start of central stuff, can send
		p_B_central[index] = B;        // into the 1 array where it belongs.
	}
	
	
	// =============================================================================
	// Understand the following important fact:
	// If you will use 63 registers (and this routine surely will - 
	// we have positions 7 x 2 x 2 = 28 registers, A 7 x 3 x 2 = 35 registers
	// -- though we could try taking into account, 1 dimension at a time)
	// Then the max thread throughput per SM is 512 which means that we will get
	// no penalty from using up to 12 doubles in shared memory per thread.
	// =============================================================================
	// That does mean L1 has room for only 4 doubles. It is not big compared to registry itself.
}


__global__ void Kernel_Rel_advect_v_tris (
	f64 h,
    structural * __restrict__ p_info,
	nT * __restrict__ p_nT_minor,
	nT * __restrict__ p_nT_minor_new,
	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_v_minor,
	f64_vec2 * __restrict__ p_tri_centroid,
	LONG3 * __restrict__ p_tri_corner_index,
	LONG3 * __restrict__ p_tri_neigh_index,
	CHAR4 * __restrict__ p_tri_per_info,
	CHAR4 * __restrict__ p_tri_per_neigh, // is neighbour re-oriented rel to this

	f64 * __restrict__ p_area_old,  // get from where?
	f64 * __restrict__ p_area_new,

	f64_vec3 * __restrict__ p_v_out
	)
{
	// Idea of momentum advection
	// ==========================
	// n_tri has been inferred from n_major
	// Average nv to the edge between minors;
	// find mom flow 
	// ARE WE CLEAR ABOUT USING nv AT ALL? NEED TO CHECK CORRESPONDENCE ---
	// v = (n_k area_k v_k + additional mom)/(

	// Need rel to v_overall ...

	// Let's assume this kernel is called for threads corresp to ##triangles##.

	// This info needed to do the "more proper" way with v_edge subtracted from each v that gets averaged.
/*	__shared__ f64_vec2 tri_centroid[blockDim.x];            // + 2
	__shared__ f64_vec2 vertex_pos[SIZE_OF_MAJOR_PER_TRI_TILE];   // + 1
	__shared__ f64_vec3 p_v_tri[blockDim.x];              // + 3
	__shared__ f64_vec3 p_v_central[SIZE_OF_MAJOR_PER_TRI_TILE];   // + 1.5
	__shared__ f64 p_n_central[SIZE_OF_MAJOR_PER_TRI_TILE];   // + 0.5
	__shared__ f64 p_n_tri[blockDim.x];                   // + 1        =   9
	__shared__ f64_vec2 p_v_overall[blockDim.x]; // +2
	__shared__ f64_vec2 p_v_overall[SIZE_OF_MAJOR_PER_TRI_TILE]; // +1 needs to be limited to vertices --
	*/

	// 9+3 = 12 so that leaves no room for tri perflag - but that's OK.

	__shared__ f64_vec2 tri_centroid[threadsPerTileMinor];
	__shared__ f64_vec3 v_tri[threadsPerTileMinor];
	__shared__ f64_vec2 n_vrel_tri[threadsPerTileMinor];
	
	// For central cells, going to have to run all over again with the following
	// replaced by __shared__ long IndexTri[MAXNEIGH_d*SIZE_OF_MAJOR_PER_TRI_TILE];
	
	__shared__ f64_vec2 n_vrel_central[SIZE_OF_MAJOR_PER_TRI_TILE];
	__shared__ f64_vec3 v_central[SIZE_OF_MAJOR_PER_TRI_TILE];
	__shared__ f64_vec2 vertex_pos[SIZE_OF_MAJOR_PER_TRI_TILE];   // 2 + 1 + 3+ 1.5 +2 +1 = 10.5
	 
	// It is more certain that something vile does not go wrong, if we do stick with loading
	// tri index each central.
	// But we don't have room for that here due to sharing v_central.
	// So we basically have to write 2 routines, even doing it this way. :-[
	
	// Consider to chop and change to the alternative: how can we try to ensure that we do
	// get a contiguous access each time we do a load and go through? We can't because it may
	// do extra bus loads for some threads in-between.
	// So. Stick with inelegant ways.

	// __shared__ char shared_per[blockDim.x]; // tri periodic info --- there may be other more sensible ways though
	// I'm seeing now that there is sense in just loading a CHAR4 with the information in.
	// ?
	// Then it doesn't need to even do a load of tests - it'll decide beforehand on CPU where it needs to
	// have a periodic rotation looking at the next triangle.
	// Loading per_info for itself and putting into shared is reasonable mind you. It's not an extra load.
	// But sometimes there IS an extra load then, because we have to ask edge triangles about their periodic data
	// and that is not a contiguous fetch. 
	// Keep shared memory cleaner. Okay then. So we COULD then fit in all the other things separately if we wanted.
	// But how would we ideally do the advect formula then?
	
	// Overall v_edge comes from 
	// But actually there is no reason to average v_edge along the edge.
	// Look instead at each end.
	// There we have v_overall = average of 3. nv = average of 3. 
	
	// So then do I want to calc v_overall, push it back into each of them: sum n_i (v_i - [v_overall = avg])
	// These seem like niceties.
	
	// ______________________________________________________________________________
	
	// can create on our own triangle but how shall we create for edge?
	// Use vertices nearby edge instead??
	// These are the ones moving and therefore moving the edge, not the tri centroid.
	// IMPORTANT:
	// Another alternative way is to infer the edge motion from the 4 relevant points, but use
	// only the opposing 2, or use 5/12, to create v of species.
	
	// To get actual conservation of momentum we have to run again and divide by,
	// for each cell, NEW N_k+1 that comes from n_k+1 avged, area_k+1.

	long StartTri = blockIdx.x*threadsPerTileMinor;
	//long EndMinor = (blockIdx.x+1)*blockDim.x; // can ditch
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Valid traffic of momentum:
	
	tri_centroid[threadIdx.x] = p_tri_centroid[index];	
	f64_vec3 v_own = p_v_minor[index];
	v_tri[threadIdx.x] = v_own;
	f64 n_own = p_nT_minor[index].n;
	f64_vec2 v_overall = p_v_overall_minor[index];
	f64_vec2 nvrel;
	nvrel.x = n_own*(v_own.x - v_overall.x);
	nvrel.y = n_own*(v_own.y - v_overall.y);
	n_vrel_tri[threadIdx.x] = nvrel;
	// What makes this way better?
	// Isn't it better to put
	// store n_s, store v_overall, store v_s.
	
	CHAR4 perinfo = p_tri_per_info[index];
	// CHAR4 perneighinfo = p_tri_per_neigh[index]; 
	// 3 chars for neighs per0,1,2 to show rel rotation; 'periodic' is just padding.
	
	// If we load tri_per_info for neighbours then ?
	// If the neigh is periodic and we are not, we can tell from x-values.

	// How was it done for Lap A??
		
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE) {
		structural info = p_info[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		vertex_pos[threadIdx.x] = info.pos;
		
		v_central[threadIdx.x] = p_v_minor[ BEGINNING_OF_CENTRAL + blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x ];
		f64 n = p_nT_minor[ BEGINNING_OF_CENTRAL + blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x ].n;
		v_overall = p_v_overall_minor[ BEGINNING_OF_CENTRAL + blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x ];
		
		nvrel.x = n*(v_central[threadIdx.x].x - v_overall.x);
		nvrel.y = n*(v_central[threadIdx.x].y - v_overall.y);
		n_vrel_central[threadIdx.x] = nvrel;
		
		// Saved data vertex_pos, v_central, n_vrel_central  |  for each vertex in tile.
	}
	__syncthreads();
	
	nvrel = n_vrel_tri[threadIdx.x];
	
	if (perinfo.flag == DOMAIN_TRIANGLE)  
	{
		// The other cases:
		//               CROSSING_INS, we assume v = 0 for now
		//               OUTER_FRILL, v = 0
		//               INNER_TRIANGLE, v = 0

		//nT nTsrc = p_nT_shared[threadIdx.x]; 
		f64 area_old = p_area_old[index]; // where getting these from?
		f64 area_new = p_area_new[index];
		
		LONG3 corner_index = p_tri_corner_index[index];
		LONG3 neightri = p_tri_neigh_index[index];
		CHAR4 perneigh = p_tri_per_neigh[index];
		// Of course, if we were smart we could roll these into 3 longs
		// in both cases, because we only need 24 bits to describe index.
		// Ultimately that would be better.
		
		f64_vec2 pos0, pos1, pos2, edgenormal; 
		f64_vec2 u0,u1,u2, ownpos;
		f64_vec3 Nv(0.0,0.0,0.0);
		
		// Create pos0,1,2 and adjust for periodic:
		
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos0 = vertex_pos[corner_index.i1-StartMajor];
		} else {
			structural info = p_info[corner_index.i1];
			pos0 = info.pos;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos1 = vertex_pos[corner_index.i2-StartMajor];
		} else {
			structural info = p_info[corner_index.i2];
			pos1 = info.pos;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			pos2 = vertex_pos[corner_index.i3-StartMajor];
		} else {
			structural info = p_info[corner_index.i3];
			pos2 = info.pos;
		};
		
		if (perinfo.per0 == NEEDS_ANTI) {
			pos0 = Anticlock_rotate2(pos0);
		}
		if (perinfo.per0 == NEEDS_CLOCK) {
			pos0 = Clockwise_rotate2(pos0);
		}
		if (perinfo.per1 == NEEDS_ANTI) {
			pos1 = Anticlock_rotate2(pos1);
		}
		if (perinfo.per1 == NEEDS_CLOCK) {
			pos1 = Clockwise_rotate2(pos1);
		}
		if (perinfo.per2 == NEEDS_ANTI) {
			pos2 = Anticlock_rotate2(pos2);
		}
		if (perinfo.per2 == NEEDS_CLOCK) {
			pos2 = Clockwise_rotate2(pos2);
		}
	//	};
		// Create u0,1,2 and adjust for periodic:
	//	CHAR4 tri_rotate(0,0,0,0); // 4 chars but really using 3
		
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			u0 = tri_centroid[neightri.i1-StartTri];
		//	perneigh = shared_per[neightri.i1-StartTri];
		} else {
			u0 = p_tri_centroid[neightri.i1];
		// 	CHAR4 perinfoneigh = p_tri_per_info[neightri.i1];
		//	perneigh = perinfoneigh.periodic; // just load and use 1 char ?...
		}
		//if (perneigh != perinfo.periodic) {
		//	// Test to see if we need to rotate the neighbour centroid and A:
		//	if ((perneigh != 0) && (ownpos.x > 0.0)) {
		//		// Avoid loading per flags again: save this as a char
		//		tri_rotate.per0 = 1; // rotate it clockwise
		//		u0 = Clockwise_rotate2(u0);
		//	};
		//	if ((perinfo.periodic != 0) && (u0.x > 0.0)) {
		//		u0 = Anticlock_rotate2(u0);
		//		tri_rotate.per0 = -1;
		//	};		
		//};
		
		// ^^ Did I decide this was bad for some reason? Better to load 
		// just a char4 for periodic relationship to neighs? COULD BE.
		// When we load all of these for edge ones it's individual.
		// 64 accesses vs 256/12. 256/8 = 32 so it's better this way round.
		
		// HMM
		
		if (perneigh.per0 == NEEDS_ANTI)
			u0 = Anticlock_rotate2(u0);
		if (perneigh.per0 == NEEDS_CLOCK)
			u0 = Clockwise_rotate2(u0);
		
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri + threadsPerTileMinor))
		{
			u1 = tri_centroid[neightri.i2 - StartTri];
		} else {
			u1 = p_tri_centroid[neightri.i2];
		};
		if (perneigh.per1 == NEEDS_ANTI)
			u1 = Anticlock_rotate2(u1);
		if (perneigh.per1 == NEEDS_CLOCK)
			u1 = Clockwise_rotate2(u1);
		
		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri + threadsPerTileMinor))
		{
			u2 = tri_centroid[neightri.i3 - StartTri];
		} else {
			u2 = p_tri_centroid[neightri.i3];
		};		
		if (perneigh.per2 == NEEDS_ANTI)
			u2 = Anticlock_rotate2(u2);
		if (perneigh.per2 == NEEDS_CLOCK)
			u2 = Clockwise_rotate2(u2);
		
		// Let's say that we only need to take the average v with the opposite cell.
		
		// Edge facing tri 0:
		edgenormal.x = 0.333333333333333*(pos1.y-pos2.y);
		edgenormal.y = 0.333333333333333*(pos2.x-pos1.x);
		if ((pos0-pos1).dot(edgenormal) > 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		
		// The edge is moving with ?
		// Corners at (1/3)(ownpos + u2 + pos0),(1/3)(ownpos+u2 + pos1)
		// v_overall only really matters insofar that it has a dotproduct with edgenormal.
		
		// Think about this clearly.
		// v_overall was generated in major cells.
		// Then it was averaged out to triangles. 
		// Here the edge endpoints are formed by taking the average of 2 centroids + 1 vertex.
		// Therefore we do want to use v_overall from those 4 locations.
		f64_vec2 nvrel_prev, nvrel_out, nvrel_next;
		
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			nvrel_prev = n_vrel_central[corner_index.i2-StartMajor];
		} else {
			f64_vec3 v = p_v_minor[ BEGINNING_OF_CENTRAL + corner_index.i2];
			f64 n = p_nT_minor[ BEGINNING_OF_CENTRAL + corner_index.i2].n;
			v_overall = p_v_overall_minor[ BEGINNING_OF_CENTRAL + corner_index.i2];
			nvrel_prev.x = n*(v.x - v_overall.x);
			nvrel_prev.y = n*(v.y - v_overall.y);
		};
		if (perinfo.per1 == NEEDS_ANTI) 
			nvrel_prev = Anticlock_rotate2(nvrel_prev);
		if (perinfo.per1 == NEEDS_CLOCK)
			nvrel_prev = Clockwise_rotate2(nvrel_prev); 
		// Every single one of these rotates will need to be checked.
		
		f64_vec3 v_out, vnext;
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			nvrel_out = n_vrel_tri[neightri.i1-StartTri];
			v_out = v_tri[neightri.i1-StartTri];
		} else {
			f64_vec3 v = p_v_minor [ neightri.i1];
			f64 n = p_nT_minor[neightri.i1].n;
			v_overall = p_v_overall_minor[neightri.i1]; 
			// I do not say this is the best way. Only that it is a way.
			nvrel_out.x = n*(v.x- v_overall.x);
			nvrel_out.y = n*(v.y - v_overall.y);
			v_out = v;
		};
		if (perneigh.per0 == NEEDS_ANTI)
		{
			nvrel_out = Anticlock_rotate2(nvrel_out);
			v_out = Anticlock_rotate3(v_out);
		};
		if (perneigh.per0 == NEEDS_CLOCK)
		{
			nvrel_out = Clockwise_rotate2(nvrel_out);
			v_out = Clockwise_rotate3(v_out);
		};
		
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			nvrel_next = n_vrel_central[corner_index.i3-StartMajor];
			vnext = v_central[corner_index.i3-StartMajor];
		} else {
			vnext = p_v_minor [BEGINNING_OF_CENTRAL + corner_index.i3];
			f64 n = p_nT_minor[BEGINNING_OF_CENTRAL + corner_index.i3].n;
			v_overall = p_v_overall_minor[BEGINNING_OF_CENTRAL + corner_index.i3];
			nvrel_next.x = n*(vnext.x - v_overall.x);
			nvrel_next.y = n*(vnext.y - v_overall.y);
			// Need 'vnext' to avoid loading data twice.
		};
		// So we keep how many in memory? 3 out of 6. Then we move round.
		
		if (perinfo.per2 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
			vnext = Anticlock_rotate3(vnext);
		};
		if (perinfo.per2 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
			vnext = Clockwise_rotate3(vnext);
		};
		
		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................

		// Edge facing point 2:
		edgenormal.x = 0.333333333333333*(pos1.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos1.x);
		if ((pos2-pos1).dot(edgenormal) < 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		// Churn them around:
		// "next" is now "out"
		nvrel_prev = nvrel_out;
		v_out = vnext;
		nvrel_out = nvrel_next;
		
		// new 'next' is tri 1
		
		if ((neightri.i2 >= StartTri) && (neightri.i2 < StartTri + threadsPerTileMinor))
		{
			nvrel_next = n_vrel_tri[neightri.i2-StartTri];
			vnext = v_tri[neightri.i2-StartTri];
		} else {
			f64_vec3 v = p_v_minor [ neightri.i2];
			f64 n = p_nT_minor[neightri.i2].n;
			v_overall = p_v_overall_minor[neightri.i2]; 
			// I do not say this is the best way. Only that it is a way.
			nvrel_next.x = n*(v.x- v_overall.x);
			nvrel_next.y = n*(v.y - v_overall.y);
			vnext = v;
		};
		if (perneigh.per1 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
			vnext = Anticlock_rotate3(vnext);
		};
		if (perneigh.per1 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
			vnext = Clockwise_rotate3(vnext);
		};
		
		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................

		// Edge facing tri 1:
		edgenormal.x = 0.333333333333333*(pos2.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos2.x);
		if ((pos1-pos0).dot(edgenormal) > 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		// Churn them around:
		// "next" is now "out"
		nvrel_prev = nvrel_out;
		v_out = vnext;
		nvrel_out = nvrel_next;
		
		// new 'next' is point 0
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			nvrel_next = n_vrel_central[corner_index.i1-StartMajor];
			vnext = v_central[corner_index.i1-StartMajor];
		} else {
			vnext = p_v_minor [BEGINNING_OF_CENTRAL + corner_index.i1];
			f64 n = p_nT_minor[BEGINNING_OF_CENTRAL + corner_index.i1].n;
			v_overall = p_v_overall_minor[BEGINNING_OF_CENTRAL + corner_index.i1];
			nvrel_next.x = n*(vnext.x - v_overall.x);
			nvrel_next.y = n*(vnext.y - v_overall.y);
			// Need 'vnext' to avoid loading data twice.
		};
		if (perinfo.per0 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
			vnext = Anticlock_rotate3(vnext);
		};
		if (perinfo.per0 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
			vnext = Clockwise_rotate3(vnext);
		};

		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................

		// Edge facing point 0:
		edgenormal.x = 0.333333333333333*(pos2.y-pos1.y);
		edgenormal.y = 0.333333333333333*(pos1.x-pos2.x);
		if ((pos0-pos1).dot(edgenormal) < 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		// Churn them around:
		// "next" is now "out"
		nvrel_prev = nvrel_out;
		v_out = vnext;
		nvrel_out = nvrel_next;
		
		// new 'next' is tri 2 :
		if ((neightri.i3 >= StartTri) && (neightri.i3 < StartTri + threadsPerTileMinor))
		{
			nvrel_next = n_vrel_tri[neightri.i3-StartTri];
			vnext = v_tri[neightri.i3-StartTri];
		} else {
			f64_vec3 v = p_v_minor [ neightri.i3];
			f64 n = p_nT_minor[neightri.i3].n;
			v_overall = p_v_overall_minor[neightri.i3]; 
			// I do not say this is the best way. Only that it is a way.
			nvrel_next.x = n*(v.x- v_overall.x);
			nvrel_next.y = n*(v.y - v_overall.y);
			vnext = v;
		};
		if (perneigh.per2 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
			vnext = Anticlock_rotate3(vnext);
		};
		if (perneigh.per2 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
			vnext = Clockwise_rotate3(vnext);
		};
		
		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................

		// Edge facing tri 2:
		edgenormal.x = 0.333333333333333*(pos0.y-pos1.y);
		edgenormal.y = 0.333333333333333*(pos1.x-pos0.x);
		if ((pos2-pos1).dot(edgenormal) > 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		// Churn them around:
		// "next" is now "out"
		nvrel_prev = nvrel_out;
		v_out = vnext;
		nvrel_out = nvrel_next;
		
		// new 'next' is point 1
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE))
		{
			nvrel_next = n_vrel_central[corner_index.i2-StartMajor];
			vnext = v_central[corner_index.i2-StartMajor];
		} else {
			vnext = p_v_minor [BEGINNING_OF_CENTRAL + corner_index.i2];
			f64 n = p_nT_minor[BEGINNING_OF_CENTRAL + corner_index.i2].n;
			v_overall = p_v_overall_minor[BEGINNING_OF_CENTRAL + corner_index.i2];
			nvrel_next.x = n*(vnext.x - v_overall.x);
			nvrel_next.y = n*(vnext.y - v_overall.y);
			// Need 'vnext' to avoid loading data twice.
		};
		if (perinfo.per1 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
			vnext = Anticlock_rotate3(vnext);
		};
		if (perinfo.per1 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
			vnext = Clockwise_rotate3(vnext);
		};

		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................

		// Edge facing point 1:
		edgenormal.x = 0.333333333333333*(pos2.y-pos0.y);
		edgenormal.y = 0.333333333333333*(pos0.x-pos2.x);
		if ((pos1-pos0).dot(edgenormal) < 0.0)
		{
			edgenormal.x = -edgenormal.x;
			edgenormal.y = -edgenormal.y;
		};
		
		// Churn them around:
		// "next" is now "out"
		nvrel_prev = nvrel_out;
		v_out = vnext;
		nvrel_out = nvrel_next;
		
		// new 'next' is tri 0		
		if ((neightri.i1 >= StartTri) && (neightri.i1 < StartTri + threadsPerTileMinor))
		{
			nvrel_next = n_vrel_tri[neightri.i1-StartTri];
		//	vnext = v_tri[neightri.i1-StartTri];
		} else {
			f64_vec3 v = p_v_minor [ neightri.i1];
			f64 n = p_nT_minor[neightri.i1].n;
			v_overall = p_v_overall_minor[neightri.i1]; 
			// I do not say this is the best way. Only that it is a way.
			nvrel_next.x = n*(v.x - v_overall.x);
			nvrel_next.y = n*(v.y - v_overall.y);
		//	vnext = v;
		};
		if (perneigh.per0 == NEEDS_ANTI)
		{
			nvrel_next = Anticlock_rotate2(nvrel_next);
		//	vnext = Anticlock_rotate2(vnext);
		};
		if (perneigh.per0 == NEEDS_CLOCK)
		{
			nvrel_next = Clockwise_rotate2(nvrel_next);
		//	vnext = Clockwise_rotate2(vnext);
		};
		
		// momflow = h*(nv.dot(edgenormal))*v;
		Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_out + nvrel_out + nvrel_prev + nvrel_next).dot
			 (edgenormal))*(0.5*(v_out + v_own));
		
		// ....................................
		// that's it - that was 6. 

		// -------------------------------------------------
		Nv += n_own*v_own*area_old;  // Reused n and v : CAREFUL ?
		// Note that 'n' does get overwritten above.
		
		// save off:
		f64 dest_n = p_nT_minor_new[index].n;
		p_v_out[index] = (Nv / (dest_n*area_new));
		////if (index == 43654) {
		////	printf("43654: %1.8E %1.8E %1.8E | %1.8E %1.8E | %1.8E %1.8E | %1.8E %1.8E %1.8E \n",
		////		Nv.x,Nv.y,Nv.z,dest_n,area_new, n_own, area_old,v_own.x,v_own.y,v_own.z);
		////	// dest_n comes out 0 --- yet when we print out from host code it is not 0.

		////}
		////
	} else {
		// Not DOMAIN_TRIANGLE:

		// Set v = 0?

	};

	// Now move on to centrals with the same data in memory.
	// Unfortunately we can't -- unless we figured out how to overwrite the central n data with
	// indextri data
	// Or, do what we should have done, and make indextri[0] a contiguous fetch so no array storage is needed.
}

__global__ void Kernel_Rel_advect_v_central(
	f64 const h,
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_tri_centroid,
	nT * __restrict__ p_nT,
	nT * __restrict__ p_nT_minor,
	nT * __restrict__ p_nT_new,
	f64_vec3 * __restrict__ p_v,
	f64_vec2 * __restrict__ p_v_overall_minor,
	long * __restrict__ p_indextri,
	char * __restrict__ pPBCtri,
	f64 * __restrict__ p_area_old,
	f64 * __restrict__ p_area_new,
	f64_vec3 * __restrict__ p_v_out

	// Not making a whole lot of sense: we need nT_minor, for tris?
	)
{
	// Maybe we SHOULD change it to put indextri packed the other way ---> be able
	// to merge this into the tris routine.

	// That is the good alternative. Using scatter-not-gather with atomicAdd and doing as part of tri code is not a good way for us.
	// what other way is there?
	// We do want to have thread run for each central.
	// Or... 
	// stuck with atomic add between threads even if we could arrange
	// it to be not between blocks by running certain colours at once.
	// Gather not scatter.
	// Need indextri ... would have been far better
	// to have put contiguous storage for first, second, third index.
	// 
	// OK - stick with incredibly inelegant way for now,
	// know that we should eventually change it given time, then we can merge
	// the central calcs routine into this tri calcs routine.
	
	// nvm
	
	// Alternative for central rel v:

	__shared__ f64_vec2 tri_centroid[SIZE_OF_TRI_TILE_FOR_MAJOR]; // + 2
	__shared__ f64_vec3 v_tri[SIZE_OF_TRI_TILE_FOR_MAJOR]; // + 3
	__shared__ f64_vec2 n_vrel_tri[SIZE_OF_TRI_TILE_FOR_MAJOR];   // + 2
	
	//__shared__ char shared_per[SIZE_OF_TRI_TILE_FOR_MAJOR]; // tri periodic info
	// Perhaps better to load in PBCtri list instead.
	// I think so? Saves interrogating tris outside the tile.

	__shared__ long IndexTri[threadsPerTileMajor*MAXNEIGH_d];
	__shared__ char PBCtri[threadsPerTileMajor*MAXNEIGH_d];
	
	// per thread: 2*7 + 6 + 1.5 = 21.5 < 24
	// We'd bring down to 14 if we chose to do contiguous index loads per neigh;
	// however that feels like it has a high chance of not working, unless we did syncthreads.
	
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	
	v_tri[threadIdx.x] = p_v[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	tri_centroid[threadIdx.x] = p_tri_centroid[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	
	f64 n = p_nT_minor[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x].n;
	f64_vec2 v_overall = p_v_overall_minor[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	f64_vec2 nvrel;
	nvrel.x = n*(v_tri[threadIdx.x].x - v_overall.x);
	nvrel.y = n*(v_tri[threadIdx.x].y - v_overall.y);
	n_vrel_tri[threadIdx.x] = nvrel;
	
	long const StartTri = SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x;

	v_tri[threadIdx.x + blockDim.x] = p_v[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + blockDim.x + threadIdx.x];
	tri_centroid[threadIdx.x + blockDim.x] = p_tri_centroid[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + blockDim.x + threadIdx.x];
	n = p_nT_minor[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + blockDim.x + threadIdx.x].n;
	v_overall = p_v_overall_minor[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + blockDim.x + threadIdx.x];
	nvrel.x = n*(v_tri[threadIdx.x + blockDim.x].x - v_overall.x);
	nvrel.y = n*(v_tri[threadIdx.x + blockDim.x].y - v_overall.y);
	n_vrel_tri[threadIdx.x + blockDim.x] = nvrel;
	__syncthreads();
	
	structural info = p_info[index];
	//f64_vec2 ownpos = info.pos;
	
	if (info.flag == DOMAIN_VERTEX) {
			// otherwise???
		memcpy(IndexTri + threadIdx.x*MAXNEIGH_d,p_indextri + index*MAXNEIGH_d, sizeof(long)*MAXNEIGH_d);
		memcpy(PBCtri + threadIdx.x*MAXNEIGH_d, pPBCtri + index*MAXNEIGH_d, sizeof(char)*MAXNEIGH_d);
		
		// For each triangle abutting this central, we want to know things like --
		// where are the corners of the edge .. this requires the neighbouring centroids also.
		
		f64_vec2 edgenormal,ownpos,
			u_prev, u_out,u_next, nvrel_prev, nvrel_out, nvrel_next; // 8 x 2
		f64_vec3 Nv(0.0,0.0,0.0);										// + 3
		f64_vec3 v_out, v_next, v;									// + 9 = 28
		
		v = p_v[BEGINNING_OF_CENTRAL + index];
		n = p_nT_minor[BEGINNING_OF_CENTRAL + index].n;
		v_overall = p_v_overall_minor[BEGINNING_OF_CENTRAL + index]; 
		// ???????????????????????????????????????????????????????
		nvrel.x = n*(v.x-v_overall.x);
		nvrel.y = n*(v.y-v_overall.y);

		// Assume we load in u_prev:
		long indextri = IndexTri[threadIdx.x*MAXNEIGH_d + info.neigh_len-1]; // bad news, neigh_len is not tri_len
			
		// ###############################################################################
		// OOPS -- it's not true at the edge of memory, is it, so what will happen there?
		// ###############################################################################
			
		if ((indextri >= StartTri) && (indextri < StartTri + SIZE_OF_TRI_TILE_FOR_MAJOR))
		{
			u_prev = tri_centroid[indextri-StartTri];
			nvrel_prev = n_vrel_tri[indextri-StartTri];
		} else {
			u_prev = p_tri_centroid[indextri];
			f64_vec3 v_ = p_v[indextri];
			n = p_nT_minor[indextri].n;
			v_overall = p_v_overall_minor[indextri];
			nvrel_prev.x = n*(v_.x - v_overall.x);
			nvrel_prev.y = n*(v_.y - v_overall.y);
		};
		char PBC = PBCtri[threadIdx.x*MAXNEIGH_d + info.neigh_len-1];
		if (PBC == NEEDS_CLOCK) 
		{
			// Always check these rotate flags throughout.
			u_prev = Clockwise_rotate2(u_prev);
			nvrel_prev = Clockwise_rotate2(nvrel_prev);
		};
		if (PBC == NEEDS_ANTI) 
		{
			u_prev = Anticlock_rotate2(u_prev);
			nvrel_prev = Anticlock_rotate2(nvrel_prev);
		};

		indextri = IndexTri[0];
		if ((indextri >= StartTri) && (indextri < StartTri + SIZE_OF_TRI_TILE_FOR_MAJOR))
		{
			u_out = tri_centroid[indextri-StartTri];
			v_out = v_tri[indextri-StartTri];
			nvrel_out = n_vrel_tri[indextri-StartTri];
		} else {
			u_out = p_tri_centroid[indextri];
			v_out = p_v[indextri];
			n = p_nT_minor[indextri].n;
			v_overall = p_v_overall_minor[indextri];
			nvrel_out.x = n*(v_out.x - v_overall.x);
			nvrel_out.y = n*(v_out.y - v_overall.y);
		};
		PBC = PBCtri[0];
		if (PBC == NEEDS_CLOCK)
		{
			u_out = Clockwise_rotate2(u_out);
			nvrel_out = Clockwise_rotate2(nvrel_out);
			v_out = Clockwise_rotate3(v_out);
		};
		if (PBC == NEEDS_ANTI)
		{
			u_out = Anticlock_rotate2(u_out);
			nvrel_out = Anticlock_rotate2(nvrel_out);
			v_out = Anticlock_rotate3(v_out);
		};

		int i,inext;
		for (i = 0; i < info.neigh_len; i++)
		{
			inext = i+1; if (inext == info.neigh_len) inext = 0;
			indextri = IndexTri[threadIdx.x*MAXNEIGH_d + inext];
			if ((indextri >= StartTri) && (indextri < StartTri + SIZE_OF_TRI_TILE_FOR_MAJOR))
			{
				u_next = tri_centroid[indextri-StartTri];
				v_next = v_tri[indextri-StartTri];
				nvrel_next = n_vrel_tri[indextri-StartTri];			
			} else {
				u_next = p_tri_centroid[indextri];
				v_next = p_v[indextri];
				n = p_nT_minor[indextri].n;
				v_overall = p_v_overall_minor[indextri];
				nvrel_next.x = n*(v_next.x - v_overall.x);
				nvrel_next.y = n*(v_next.y - v_overall.y);
			}
			PBC = PBCtri[threadIdx.x*MAXNEIGH_d + inext];
			if (PBC == NEEDS_CLOCK)
			{
				u_next = Clockwise_rotate2(u_next);
				nvrel_next = Clockwise_rotate2(nvrel_next);
				v_next = Clockwise_rotate3(v_next);
			};
			if (PBC == NEEDS_ANTI)
			{
				u_next = Anticlock_rotate2(u_next);
				nvrel_next = Anticlock_rotate2(nvrel_next);
				v_next = Anticlock_rotate3(v_next);
			};
			
			// edgenormal:
			edgenormal.x = u_prev.y-u_next.y;
			edgenormal.y = u_next.x-u_prev.x;
			if ((ownpos-u_prev).dot(edgenormal) > 0.0) {
			// NOT SURE ABOUT THAT TEST ?
				edgenormal.x = -edgenormal.x;
				edgenormal.y = -edgenormal.y;
			}
			
			Nv -= h*(SIXTH*(nvrel + nvrel + nvrel_prev + nvrel_next + nvrel_out + nvrel_out).dot(edgenormal))
						*(0.5*(v_out + v));
			u_prev = u_out;
			u_out = u_next;		
			v_out = v_next;		
			nvrel_prev = nvrel_out;
			nvrel_out = nvrel_next;
		}		
		// Now how does it end?
				
		f64 area_old = p_area_old[index];
		f64 area_new = p_area_new[index];
		Nv += n*v*area_old; // CAREFUL: n and v ?
		// Probably got overwritten somewhere.
		f64 dest_n = p_nT_new[index].n;
		p_v_out[index] = (Nv / (dest_n*area_new));
	} else {
		// Not DOMAIN_VERTEX

		f64_vec3 zero(0.0,0.0,0.0);
		p_v_out[index] = zero;
	};
}


// Grad phi: first put on triangles from major

__global__ void Kernel_Compute_grad_phi_Te_centrals(
	structural * __restrict__ p_info_sharing, // for vertex positions & get has_periodic flag
	f64 * __restrict__ p_phi,
	nT * __restrict__ p_nT_elec,
	long * __restrict__ p_indexneigh,
	// Output:
	f64_vec2 * __restrict__ p_grad_phi,
	f64_vec2 * __restrict__ p_grad_Te
	)
{
	// Bad approach? : scatter instead of gather.
	// This thread works to create grad phi on tris because we otherwise,
	// having to load it in from tris, also have to load in periodic flags
	// regarding them. 
	// Easier to compute it here -- computing it multiple times for each tri
	// but that probably is cheaper. Less shared mem here than when we 
	// load to aggregate from tris - we then need to load area, grad phi, PB flag for tri
	// vs -- phi and position for major
	// Then instead of doing tri minors separately, is more efficient to put in a scatter
	// data here to affect tri minors: but
	// That requires that we load IndexTri and do a random write access???
	// Maybe we should keep the tris routine separate -- that's simplest for now.
	
	__shared__ f64 p_phi_shared[threadsPerTileMajor];
	__shared__ f64 p_Te_shared[threadsPerTileMajor];
	__shared__ f64_vec2 p_vertex_pos_shared[threadsPerTileMajor]; 
	__shared__ long indexneigh[MAXNEIGH_d*threadsPerTileMajor]; // 1 + 2 + 6 doublesworth
	
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 Te;
	p_phi_shared[threadIdx.x] = p_phi[blockIdx.x*blockDim.x + threadIdx.x];
	
	structural info = p_info_sharing[blockIdx.x*blockDim.x + threadIdx.x];
	p_vertex_pos_shared[threadIdx.x] = info.pos;
	{
		nT nTtemp = p_nT_elec[blockIdx.x*blockDim.x + threadIdx.x];
		p_Te_shared[threadIdx.x] = nTtemp.T;
		if (nTtemp.n == 0.0) {
			p_Te_shared[threadIdx.x] = 0.0; // recognise inner vertex
		}
	}
	Te = p_Te_shared[threadIdx.x];
	
	long StartMajor = blockIdx.x*blockDim.x;
	long EndMajor = StartMajor + blockDim.x;
	f64 phi1, phi2, Te1, Te2;
	f64_vec2 pos1, pos2;

	__syncthreads();
	

	if (info.flag == DOMAIN_VERTEX) {
		// Don't bother otherwise, as even though grad phi == 0, 
		
		// We do bother otherwise.
		// We are going to assume that we can set phi at the front and back.
		
		// BUT WE want to do differently for Te : for inner vertex we could skip. Does it help?
		// What to do at insulator for Te ? Set internal values of T equal to ours.
		
		
		memcpy(indexneigh + threadIdx.x*MAXNEIGH_d, p_indexneigh + MAXNEIGH_d*index, sizeof(long)*MAXNEIGH_d);
				
		f64_vec2 grad_phi_integrated(0.0,0.0);
		f64_vec2 grad_Te_integrated(0.0,0.0);
		f64 grad_x_integrated_x = 0.0;
		
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;
		// get phi,pos at edge -- & rotate if necessary
		long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh1];
		if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
		{
			phi1 = p_phi_shared[indexNeigh-StartMajor];
			pos1 = p_vertex_pos_shared[indexNeigh-StartMajor];
			Te1 = p_Te_shared[indexNeigh-StartMajor];
			if (Te1 == 0.0) Te1 = Te;
			
		} else {
			phi1 = p_phi[indexNeigh];
			structural infotemp = p_info_sharing[indexNeigh];
			pos1 = infotemp.pos;
			nT nTtemp = p_nT_elec[indexNeigh];
			Te1 = nTtemp.T;

			if (nTtemp.n == 0.0) Te1 = Te;
		};
		if (info.has_periodic) {
			if ((pos1.x > 0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Clockwise_rotate2(pos1);
			};
		};
		
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh2];
			if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
			{
				phi2 = p_phi_shared[indexNeigh-StartMajor];
				pos2 = p_vertex_pos_shared[indexNeigh-StartMajor];
				Te2 = p_Te_shared[indexNeigh-StartMajor];
				if (Te2 == 0.0) {
					Te2 = Te;
				}
			} else {
				phi2 = p_phi[indexNeigh];
				structural infotemp = p_info_sharing[indexNeigh];
				pos2 = infotemp.pos;
				nT nTtemp = p_nT_elec[indexNeigh];
				Te2 = nTtemp.T; // undefined if this neighbour is inside ins

				if (nTtemp.n == 0.0) {
					Te2 = Te; // but in shared?
				}

			};
			if (info.has_periodic) {
				if ((pos2.x > 0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Clockwise_rotate2(pos2);
				};
			};
			
			// Now we've got contiguous pos1, pos2, and own pos.
			
			f64_vec2 edge_normal;
			edge_normal.x = pos1.y-pos2.y;
			edge_normal.y = pos2.x-pos1.x;
			if (edge_normal.dot(info.pos-pos1) > 0.0) {
				edge_normal.x = -edge_normal.x;
				edge_normal.y = -edge_normal.y;
			}
			grad_phi_integrated += edge_normal*0.5*(phi1+phi2);
			grad_Te_integrated += edge_normal*0.5*(Te1+Te2);
			grad_x_integrated_x += edge_normal.x*0.5*(pos1.x+pos2.x);
			
			//if (index == 11685) {
			//	printf("11685: grad_phi_integrated %1.10E %1.10E\n"
			//		"phi12 %1.6E %1.6E edgenormal %1.6E %1.6E\n "	,
			//		grad_phi_integrated.x,grad_phi_integrated.y,
			//		phi1,phi2,edge_normal.x,edge_normal.y);
			//}

			// This should now be fine since phi values defined in insulator.
			phi1 = phi2;
			pos1 = pos2;
		}
		p_grad_phi[index] = grad_phi_integrated/grad_x_integrated_x;
		p_grad_Te[index] = grad_Te_integrated/grad_x_integrated_x;
		
	} else {
		// We could do it right for inner-vertex grad_phi, but no point.
		f64_vec2 zero(0.0,0.0);
		p_grad_phi[index] = zero;
		p_grad_Te[index] = zero;
	};
}


__global__ void Kernel_InitialisePhi(
	structural * __restrict__ p_info_sharing,
	f64 k1_phiinit, f64 k2_phiinit,
	f64 V,
	f64 * __restrict__ p_phi_out
							  )
{
	long index = blockIdx.x*blockDim.x+threadIdx.x;
	structural info = p_info_sharing[index];
	f64 r = info.pos.modulus();
	f64 result;
	if (r <= 2.8) {
		result = -V;
	} else {
		if (r >= 4.6) {
			result = V;
		} else {
			result = k1_phiinit*log(r)+k2_phiinit;
		};
	};
	p_phi_out[index] = result;

	// ?:
	// What about setting outermost/innermost phi values on a step.
}


__global__ void Kernel_GetThermalPressureCentrals(
	structural * __restrict__ p_info_sharing, // for vertex positions & get has_periodic flag
	nT * __restrict__ p_nT_neut,
	nT * __restrict__ p_nT_ion,
	nT * __restrict__ p_nT_elec,
	long * __restrict__ p_indexneigh,
	// Output:
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec
	)
{	
	__shared__ f64 p_nT_shared[threadsPerTileMajor];
	__shared__ f64_vec2 p_vertex_pos_shared[threadsPerTileMajor]; 
	__shared__ long indexneigh[MAXNEIGH_d*threadsPerTileMajor]; // 1 + 2 + 6 doublesworth
	
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	
	nT nT_temp = p_nT_neut[index];
	p_nT_shared[threadIdx.x] = nT_temp.n*nT_temp.T;
	structural info = p_info_sharing[index];
	p_vertex_pos_shared[threadIdx.x] = info.pos;
	
	__syncthreads();
	
	long StartMajor = blockIdx.x*blockDim.x;
	long EndMajor = StartMajor + blockDim.x; // not needed
	
	if (info.flag == DOMAIN_VERTEX) {
		// Don't bother otherwise, right?
		memcpy(indexneigh + threadIdx.x*MAXNEIGH_d, p_indexneigh + MAXNEIGH_d*index, sizeof(long)*MAXNEIGH_d);
		
		f64_vec2 grad_nT_integrated(0.0,0.0);
		f64 nT1, nT2;
		f64_vec2 pos1, pos2;
		f64 nT0 = p_nT_shared[threadIdx.x];
//		f64 areasum = 0.0;
		
		// Now let's be careful ... we want to integrate grad nT over the central cell
		// Probably our best bet is what? Divide by area out to neighs where it is found,
		// multiply by central area that is known.
		
		// * * ** * * ** * * ** * * ** * * ** * * ** * * ** * * ** * * ** * * ** * * ** 
		
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;		
		// get phi,pos -- & rotate if necessary
		long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh1];
		if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
		{
			nT1 = p_nT_shared[indexNeigh-StartMajor];
			pos1 = p_vertex_pos_shared[indexNeigh-StartMajor];
			if (nT1 == 0.0) {
				nT1 = nT0;
			}
		} else {
			nT nT_temp = p_nT_neut[indexNeigh];
			nT1 = nT_temp.n*nT_temp.T;
			structural infotemp = p_info_sharing[indexNeigh];
			pos1 = infotemp.pos;

			if (nT_temp.n == 0.0) {
				nT1 = nT0; // but in shared?
			}
		};
		if (info.has_periodic) {
			if ((pos1.x > 0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Clockwise_rotate2(pos1);
			};
		};
		
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh2];
			if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
			{
				nT2 = p_nT_shared[indexNeigh-StartMajor];
				pos2 = p_vertex_pos_shared[indexNeigh-StartMajor];
				if (nT2 == 0.0) {
					nT2 = nT0;
				}
			} else {
				nT nT_temp = p_nT_neut[indexNeigh];
				nT2 = nT_temp.n*nT_temp.T;
				structural infotemp = p_info_sharing[indexNeigh];
				pos2 = infotemp.pos;
				if (nT_temp.n == 0.0) {
					nT2 = nT0; // but in shared?
				}
			};
			if (info.has_periodic) {
				if ((pos2.x > 0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Clockwise_rotate2(pos2);
				};
			};
			
			// Now we've got contiguous pos1, pos2, and own pos.
			
			// Correctly, pos2 is the anticlockwise one, therefore edge_normal.x should be
			// pos2.y-pos1.y;
			f64_vec2 edge_normal;
			edge_normal.x = pos2.y-pos1.y;
			edge_normal.y = pos1.x-pos2.x;
			// Drop this:
//			if (edge_normal.dot(info.pos-pos1) > 0.0) {
//				edge_normal.x = -edge_normal.x;
//				edge_normal.y = -edge_normal.y;
//			}
			grad_nT_integrated += edge_normal*0.5*(nT1+nT2);
			//grad_x_integrated_x += edge_normal.x*0.5*(pos1.x+pos2.x);
			
			nT1 = nT2;
			pos1 = pos2;
		}
		
		// Now we took it integrated over the whole union of triangles, but,
		// we want to diminish this to the size of the central.
		// = 1/9 as much
		
		f64_vec3 add(-grad_nT_integrated.x/(9.0*m_n),
					 -grad_nT_integrated.y/(9.0*m_n),
					 0.0);
		p_MAR_neut[index] += add;
		
		if (index == 20000) {
			printf("\n\nGTPC 20000: %1.9E %1.9E \n",grad_nT_integrated.x,grad_nT_integrated.y);
			printf("nT1 nT2: %1.9E %1.9E \n\n",nT1,nT2);
		}
		
		// Note that we accumulated edge_normal*(phi0+phi1) so that it
		// cancelled out between every edge being counted each way.
		// Therefore we only need the outward facing edges, the rest cancel to 0.
	} else {
		// Not domain vertex
//		f64_vec2 zero(0.0,0.0);
//		p_grad_phi[index] = zero;
		// do nothing
	}
	__syncthreads();
	
	// Now proceed, with shared positions already stored, to do ion. Correct?
	
	nT_temp = p_nT_ion[blockIdx.x*blockDim.x + threadIdx.x];
	p_nT_shared[threadIdx.x] = nT_temp.n*nT_temp.T;
	
	__syncthreads();
	
	if (info.flag == DOMAIN_VERTEX) {
		// Don't bother otherwise, right?
		
		f64_vec2 grad_nT_integrated(0.0,0.0);
		f64 nT1, nT2;
		f64_vec2 pos1, pos2;
			
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;
		
		// get phi,pos -- & rotate if necessary
		long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh1];
		if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
		{
			nT1 = p_nT_shared[indexNeigh-StartMajor];
			pos1 = p_vertex_pos_shared[indexNeigh-StartMajor];
		} else {
			nT nT_temp = p_nT_ion[indexNeigh];
			nT1 = nT_temp.n*nT_temp.T;
			structural infotemp = p_info_sharing[indexNeigh];
			pos1 = infotemp.pos;
		};
		if (info.has_periodic) {
			if ((pos1.x > 0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Clockwise_rotate2(pos1);
			};
		};
		
		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			
			long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh2];
			if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
			{
				nT2 = p_nT_shared[indexNeigh-StartMajor];
				pos2 = p_vertex_pos_shared[indexNeigh-StartMajor];
			} else {
				nT nT_temp = p_nT_ion[indexNeigh];
				nT2 = nT_temp.n*nT_temp.T;
				structural infotemp = p_info_sharing[indexNeigh];
				pos2 = infotemp.pos;
			};
			if (info.has_periodic) {
				if ((pos2.x > 0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Clockwise_rotate2(pos2);
				};
			};
			
			// Now we've got contiguous pos1, pos2, and own pos.
			
			// Correctly, pos2 is the anticlockwise one, therefore edge_normal.x should be
			// pos2.y-pos1.y;
			f64_vec2 edge_normal;
			edge_normal.x = pos2.y-pos1.y;
			edge_normal.y = pos1.x-pos2.x;
			// Drop this:
//			if (edge_normal.dot(info.pos-pos1) > 0.0) {
//				edge_normal.x = -edge_normal.x;
//				edge_normal.y = -edge_normal.y;
//			}
			grad_nT_integrated += edge_normal*0.5*(nT1+nT2);
			//grad_x_integrated_x += edge_normal.x*0.5*(pos1.x+pos2.x);
			
			nT1 = nT2;
			pos1 = pos2;
		}
		
		// Now we took it integrated over the whole union of triangles, but,
		// we want to diminish this to the size of the central.
		// = 1/9 as much

		f64_vec3 add(-grad_nT_integrated.x/(9.0*m_ion),
					 -grad_nT_integrated.y/(9.0*m_ion),
					 0.0);
		p_MAR_ion[index] += add;
		
	};


	__syncthreads();

	// Now proceed, with shared positions already stored, to do ion. Correct?

	nT_temp = p_nT_elec[blockIdx.x*blockDim.x + threadIdx.x];
	p_nT_shared[threadIdx.x] = nT_temp.n*nT_temp.T;
	
	__syncthreads();
	
	if (info.flag == DOMAIN_VERTEX) {
		// Don't bother otherwise, right?
		f64_vec2 grad_nT_integrated(0.0,0.0);
		f64 nT1, nT2;
		f64_vec2 pos1, pos2;
			
		short iNeigh1 = info.neigh_len-1;
		short iNeigh2 = 0;
		
		// get phi,pos -- & rotate if necessary
		long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh1];
		if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
		{
			nT1 = p_nT_shared[indexNeigh-StartMajor];
			pos1 = p_vertex_pos_shared[indexNeigh-StartMajor];
		} else {
			nT nT_temp = p_nT_elec[indexNeigh];
			nT1 = nT_temp.n*nT_temp.T;
			structural infotemp = p_info_sharing[indexNeigh];
			pos1 = infotemp.pos;
		};
		if (info.has_periodic) {
			if ((pos1.x > 0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Anticlock_rotate2(pos1);
			};
			if ((pos1.x < -0.5*pos1.y*GRADIENT_X_PER_Y) &&
				(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
			{
				pos1 = Clockwise_rotate2(pos1);
			};
		};

		for (iNeigh2 = 0; iNeigh2 < info.neigh_len; iNeigh2++)
		{
			
			long indexNeigh = indexneigh[threadIdx.x*MAXNEIGH_d + iNeigh2];
			if ((indexNeigh >= StartMajor) && (indexNeigh < EndMajor))
			{
				nT2 = p_nT_shared[indexNeigh-StartMajor];
				pos2 = p_vertex_pos_shared[indexNeigh-StartMajor];
			} else {
				nT nT_temp = p_nT_elec[indexNeigh];
				nT2 = nT_temp.n*nT_temp.T;
				structural infotemp = p_info_sharing[indexNeigh];
				pos2 = infotemp.pos;
			};
			if (info.has_periodic) {
				if ((pos2.x > 0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x < -0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Anticlock_rotate2(pos2);
				};
				if ((pos2.x < -0.5*pos2.y*GRADIENT_X_PER_Y) &&
					(info.pos.x > 0.5*info.pos.y*GRADIENT_X_PER_Y))
				{
					pos2 = Clockwise_rotate2(pos2);
				};
			}			
			// Now we've got contiguous pos1, pos2, and own pos.
			
			// Correctly, pos2 is the anticlockwise one, therefore edge_normal.x should be
			// pos2.y-pos1.y;
			f64_vec2 edge_normal;
			edge_normal.x = pos2.y-pos1.y;
			edge_normal.y = pos1.x-pos2.x;
			grad_nT_integrated += edge_normal*0.5*(nT1+nT2);
			
			nT1 = nT2;
			pos1 = pos2;
		}
		
		// Now we took it integrated over the whole union of triangles, but,
		// we want to diminish this to the size of the central.
		// = 1/9 as much
		
		f64_vec3 add(-grad_nT_integrated.x/(9.0*m_e),
					 -grad_nT_integrated.y/(9.0*m_e),
					 0.0);
		p_MAR_elec[index] += add;		
	};	

	// We divided by particle mass and left in Area_central
}  


__global__ void Kernel_Compute_grad_phi_Te_tris(
	structural * __restrict__ p_info_sharing, // for vertex positions
	f64 * __restrict__ p_phi,
	nT * __restrict__ p_nT_elec,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_perinfo,
	// Output:
	f64_vec2 * __restrict__ p_grad_phi,
	f64_vec2 * __restrict__ p_GradTe
	)
{
	__shared__ f64 p_phi_shared[SIZE_OF_MAJOR_PER_TRI_TILE];
	__shared__ f64 p_Te_shared[SIZE_OF_MAJOR_PER_TRI_TILE];
	__shared__ f64_vec2 p_vertex_pos_shared[SIZE_OF_MAJOR_PER_TRI_TILE];
	
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long index =  threadIdx.x + blockIdx.x * blockDim.x;
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		p_phi_shared[threadIdx.x] = p_phi[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		nT nTtemp = p_nT_elec[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_Te_shared[threadIdx.x] = nTtemp.T;

		//if (nTtemp.n == 0.0) p_Te_shared[threadIdx.x] = 0.0; // WAY TO SIGNAL IT IS AN INNER VERTEX
		// We don't need this - because it does not matter how grad Te is defined for CROSSING_INS and others??
		// Still we may wish to try and make sure it is smth vaguely sensible, at the end, setting dT/dr = 0.

		structural info = p_info_sharing[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_vertex_pos_shared[threadIdx.x] = info.pos;
	}
	__syncthreads();
	
	CHAR4 perinfo = p_tri_perinfo[index];
		
	// Take grad on triangle:
	// first collect corner positions; if this is periodic triangle then we have to rotate em.
	if ((perinfo.flag == DOMAIN_TRIANGLE) || (perinfo.flag == CROSSING_INS)) { 
		LONG3 corner_index = p_tri_corner_index[index];
		// Do we ever require those and not the neighbours?
		// Yes - this time for instance.
		f64_vec2 pos0, pos1, pos2;
		f64 phi0,phi1,phi2, Te0, Te1, Te2;
		short iNeigh;

		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			pos0 = p_vertex_pos_shared[corner_index.i1-StartMajor];
			phi0 = p_phi_shared[corner_index.i1-StartMajor];
			Te0 = p_Te_shared[corner_index.i1-StartMajor];

			// Do not worry if this is not smth sensible, if inner vertex.		
		} else {
			// have to load in from global memory:
			structural info = p_info_sharing[corner_index.i1];
			pos0 = info.pos;
			phi0 = p_phi[corner_index.i1];
			nT nTtemp = p_nT_elec[corner_index.i1];
			Te0 = nTtemp.T;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			pos1 = p_vertex_pos_shared[corner_index.i2-StartMajor];
			phi1 = p_phi_shared[corner_index.i2-StartMajor];
			Te1 = p_Te_shared[corner_index.i2-StartMajor];
		} else {
			structural info = p_info_sharing[corner_index.i2];
			pos1 = info.pos;
			phi1 = p_phi[corner_index.i2];
			nT nTtemp = p_nT_elec[corner_index.i2];
			Te1 = nTtemp.T;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			pos2 = p_vertex_pos_shared[corner_index.i3-StartMajor];
			phi2 = p_phi_shared[corner_index.i3-StartMajor];
			Te2 = p_Te_shared[corner_index.i3-StartMajor];
		} else {
			structural info = p_info_sharing[corner_index.i3];
			pos2 = info.pos;
			phi2 = p_phi[corner_index.i3];
			nT nTtemp = p_nT_elec[corner_index.i3];
			Te2 = nTtemp.T;
		}
		
	//	if (perinfo.periodic == 0) {
	//	} else {
			// In this case which ones are periodic?
			// Should we just store per flags?
			// How it should work:
			// CHAR4 perinfo: periodic, per0, per1, per2;
		if (perinfo.per0 == NEEDS_ANTI) 
			pos0 = Anticlock_rotate2(pos0);
		if (perinfo.per0 == NEEDS_CLOCK)
			pos0 = Clockwise_rotate2(pos0);
		if (perinfo.per1 == NEEDS_ANTI)
			pos1 = Anticlock_rotate2(pos1);
		if (perinfo.per1 == NEEDS_CLOCK)
			pos1 = Clockwise_rotate2(pos1);
		if (perinfo.per2 == NEEDS_ANTI)
			pos2 = Anticlock_rotate2(pos2);
		if (perinfo.per2 == NEEDS_CLOCK)
			pos2 = Clockwise_rotate2(pos2);
	//	};		
		// To get grad phi:
		f64_vec2 grad_phi, edge_normal0, edge_normal1, edge_normal2, GradTe;
		// Integral of grad... average phi on edge . edgenormal
		// This should give the same result as the plane passing through
		// the 3 corners -- a few simple examples suggest yes.
		
		edge_normal0.x = pos2.y-pos1.y;
		edge_normal0.y = pos1.x-pos2.x;
		// Got to make sure it points out. How? Have to take
		// dot product with vector to the opposing point
		if (edge_normal0.dot(pos0-pos1) > 0.0) {
			// points to opposing point - wrong way
			edge_normal0.x = -edge_normal0.x;
			edge_normal0.y = -edge_normal0.y;
		}
		edge_normal1.x = pos2.y-pos0.y;
		edge_normal1.y = pos0.x-pos2.x;
		if (edge_normal1.dot(pos1-pos0) > 0.0) {
			edge_normal1.x = -edge_normal1.x;
			edge_normal1.y = -edge_normal1.y;
		}
		edge_normal2.x = pos1.y-pos0.y;
		edge_normal2.y = pos0.x-pos1.x;
		if (edge_normal2.dot(pos2-pos0) > 0.0) {
			edge_normal2.x = -edge_normal2.x;
			edge_normal2.y = -edge_normal2.y;
		};
		grad_phi = 
			( 0.5*(phi1 + phi2)*edge_normal0 // opposite phi0
			+ 0.5*(phi0 + phi2)*edge_normal1
			+ 0.5*(phi1 + phi0)*edge_normal2 );		
		
		GradTe = 
			( 0.5*(Te1 + Te2)*edge_normal0 // opposite phi0
			+ 0.5*(Te0 + Te2)*edge_normal1
			+ 0.5*(Te1 + Te0)*edge_normal2 );		
		// Divide by area -- easier to recalculate here than to load it in.
		f64 area = fabs(0.5*(
			   (pos1.x+pos0.x)*edge_normal2.x
			 + (pos2.x+pos1.x)*edge_normal0.x
			 + (pos0.x+pos2.x)*edge_normal1.x
			));
		grad_phi /= area;
		GradTe /= area;

		if (perinfo.flag == CROSSING_INS) {
			// Included just for 'fun' - not clear we want to use it.
			// dT/dr = 0
			f64_vec2 rhat = THIRD*(pos0+pos1+pos2);
			//rhat /= rhat.modulus();
			GradTe -= rhat*(rhat.dot(GradTe))/(rhat.x*rhat.x+rhat.y*rhat.y);
		}
		
		// Grad of phi on tri is grad for this minor within the tri:
		p_grad_phi[index] = grad_phi;
		p_GradTe[index] = GradTe;
		
		//if (index == 73400) {
		//	printf("73400 Grad phi:\n"
		//		"%d %d %d\n"
		//		"phi012 %1.9E %1.9E %1.9E\n"
		//		"pos0xy %1.9E %1.9E pos1 %1.9E %1.9E pos2  %1.9E %1.9E\n"
		//		"area %1.9E \n----------------------------------\n",
		//		corner_index.i1,corner_index.i2,corner_index.i3,
		//		phi0,phi1,phi2,
		//		pos0.x,pos0.y,pos1.x,pos1.y,pos2.x,pos2.y,
		//		area);
		//}
	} else {
		f64_vec2 zero(0.0,0.0);
		p_grad_phi[index] = zero;
		p_GradTe[index] = zero;

		// We should not need them for any non-domain triangle so it really does not matter
		// what happens inside ins -- and this is the quickest way.


	}
}

__global__ void Get_Lap_phi_on_major(
									f64 * __restrict__ p_phi,								
									structural * __restrict__ p_info_sharing,
								//	f64_vec2 * __restrict__ p_tri_centroid,
									long * __restrict__ pIndexNeigh,
									char * __restrict__ pPBCNeigh,
									// output:
									f64 * __restrict__ p_Lap_phi
									 )
{
	__shared__ f64 p_phi_shared[threadsPerTileMajor];
	__shared__ f64_vec2 p_vertex_pos_shared[threadsPerTileMajor];
	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajor];
	// So, per thread: 1 + 2 + 6 doubles = 9 doubles.
	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajor]; // + 1.5
	//__shared__ f64_vec2 tri_centroid[SIZE_OF_TRI_TILE_FOR_MAJOR]; // + 4

	// This is not good: 1 + 2 + 6 + 1.5 + 4 = 14.5 --- we said max 12 for decent throughput.
	// I think we can drop PBCneigh here and use info.has_periodic
	
	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x; 
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	
	f64 phi_clock, phi_anti, phi_out, phi;
	f64_vec2 pos_clock, pos_anti, pos_out;
	char PBC;

	p_phi_shared[threadIdx.x] = p_phi[index];
	structural info = p_info_sharing[index];
	structural info2;
	p_vertex_pos_shared[threadIdx.x] = info.pos;
	
	// We are going to want tri centroids to know the edge of the major cell.
	//tri_centroid[threadIdx.x] = p_tri_centroid[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + threadIdx.x];
	//tri_centroid[threadIdx.x + blockDim.x] = p_tri_centroid[blockIdx.x*SIZE_OF_TRI_TILE_FOR_MAJOR + blockDim.x + threadIdx.x];
	
	__syncthreads();
	
	f64 Lapphi = 0.0, Area = 0.0;
	
	long indexneigh;

	// New plan for Lap phi.
	// It exists wherever phi evolves, that is anywhere that is not outermost or innermost
	// In those cases let Lap phi = 0 and it is never to be used.

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		Area = 1.0; // lap phi = 0

	} else {
	
//case DOMAIN_VERTEX:
			
		// Now we've got to load up what we need for the edge of the major cell.
		// Did we do this anywhere else?
		phi = p_phi_shared[threadIdx.x];
		memcpy(Indexneigh + MAXNEIGH_d*threadIdx.x, 
					pIndexNeigh + MAXNEIGH_d*index,
					MAXNEIGH_d*sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d*threadIdx.x, 
					pPBCNeigh + MAXNEIGH_d*index,
					MAXNEIGH_d*sizeof(char));
		
		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len-1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = p_vertex_pos_shared[indexneigh-StartMajor];
			phi_clock = p_phi_shared[indexneigh-StartMajor];
		} else {
			info2 = p_info_sharing[indexneigh];
			pos_clock = info2.pos;
			phi_clock = p_phi[indexneigh];
		};

		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len-1];
		if (PBC == NEEDS_ANTI)
			pos_clock = Anticlock_rotate2(pos_clock);
		if (PBC == NEEDS_CLOCK)
			pos_clock = Clockwise_rotate2(pos_clock);
		
		// What about neighs and tris? Are they in the appropriate relationship?
		// How about: load vertex positions --> work out centroids --
		// we need phi from vertices anyway and we need their positions anyway. So.
	
		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = p_vertex_pos_shared[indexneigh-StartMajor];
			phi_out = p_phi_shared[indexneigh-StartMajor];
		} else {
			info2 = p_info_sharing[indexneigh];
			pos_out = info2.pos;
			phi_out = p_phi[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI)
			pos_out = Anticlock_rotate2(pos_out);
		if (PBC == NEEDS_CLOCK)
			pos_out = Clockwise_rotate2(pos_out);
		
		short iNeigh;
#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			int inext = iNeigh+1; if (inext == info.neigh_len) inext = 0; 
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + inext];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = p_vertex_pos_shared[indexneigh-StartMajor];
				phi_anti = p_phi_shared[indexneigh-StartMajor];
			} else {
				info2 = p_info_sharing[indexneigh];
				pos_anti = info2.pos;
				phi_anti = p_phi[indexneigh];
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + inext];
			if (PBC == NEEDS_ANTI)
				pos_anti = Anticlock_rotate2(pos_anti);
			if (PBC == NEEDS_CLOCK)
				pos_anti = Clockwise_rotate2(pos_anti);
			
			// Choice of using PBC list here. Alternative is what: just working from
			// 'has_periodic' flag on our own thread, and deciding from that based on x/y.
			// ___________________________________________________________________________

			// Now what to do with it?
			// Find the edge:
			f64_vec2 edgenormal;
			//vec2 = THIRD*(pos_clock + info.pos + pos_out); <--- assume this would be centroid...
			edgenormal.x = THIRD*(pos_anti.y-pos_clock.y);
			edgenormal.y = THIRD*(pos_clock.x-pos_anti.x);

			// HERE DID NOT HAVE TO USE tri_centroid AFTER ALL.
			// HOWEVER MAKE SURE WE DO THE RIGHT THING IN CASE THIS ABUTS THE INSULATOR.
			// In this case, tri centroid is meant to be projected to insulator!! 
			// ^^ !!
			// But is it important how Lap phi is calculated, if we extend it right through the domain?
			// Not sure.


			Lapphi += ( (phi - phi_out) * ( (pos_anti.y-pos_clock.y)*edgenormal.x
										  + (pos_clock.x-pos_anti.x)*edgenormal.y )  
					+	(phi_anti-phi_clock)*( (pos_out.y - info.pos.y)*edgenormal.x
										  + (info.pos.x - pos_out.x)*edgenormal.y) )

				/ ( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
				  + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) );


		//	if (index == 10000) {
		//		printf("10000: Lapphi %1.10E area %1.9E \nphi_clockoutanti %1.9E %1.9E %1.9E %1.9E\n",
		//			Lapphi,Area,
		//			phi_clock, phi_out,phi_anti, phi);
		//		printf("shoelace: %1.10E \n",
		//			( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
		//			  + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) ));
		//		printf("infopos %1.10E %1.10E \npos_out %1.10E %1.10E \npos_clk %1.10E %1.10E \npos_ant %1.10E %1.10E \n",
		//			 info.pos.x,info.pos.y,pos_out.x,pos_out.y,
		//			 pos_clock.x,pos_clock.y,pos_anti.x,pos_anti.y);
		//	};
			
			Area += 0.5*(pos_clock.x + pos_anti.x)*edgenormal.x;


/*
			if (pos_out.x*pos_out.x+pos_out.y*pos_out.y < DEVICE_INSULATOR_OUTER_RADIUS*DEVICE_INSULATOR_OUTER_RADIUS)
			{
				// Zero contribution, looking into insulator
			} else {
				if (pos_anti.x*pos_anti.x+pos_anti.y*pos_anti.y < DEVICE_INSULATOR_OUTER_RADIUS*DEVICE_INSULATOR_OUTER_RADIUS)
				{
					// assume we just look at the phi_out? No, 					
					// get grad phi from 3 points. 

					f64 shoelacedoubled =  ( (pos_clock.x + info.pos.x)*(pos_clock.y-info.pos.y) // y_anti - y_clock --- pos_clock is the highest one.
									+ (pos_clock.x + pos_out.x)*(pos_out.y-pos_clock.y)
									+ (pos_out.x + info.pos.x)*(info.pos.y-pos_out.y));
					f64_vec2 Gradphi;

					Gradphi.x = ( (phi_clock + phi)*(pos_clock.y-info.pos.y) // y_anti - y_clock --- pos_clock is the highest one.
								+ (phi_clock + phi_out)*(pos_out.y-pos_clock.y)
								+ (phi_out + phi)*(info.pos.y-pos_out.y) )
									/ shoelacedoubled;					
					Gradphi.y = ( (phi_clock + phi)*(info.pos.x-pos_clock.x) // y_anti - y_clock --- pos_clock is the highest one.
								+ (phi_clock + phi_out)*(pos_clock.x-pos_out.x)
								+ (phi_out + phi)*(pos_out.x-info.pos.x) )
									/ shoelacedoubled;
					
					Lapphi += Gradphi.dot(edgenormal);

					// We did not yet modify edgenormal, note bene.

					// And what then is the contribution for shoelace?
					
					// Should be adding up
					// integral of dx/dx
					//edgenormal.x = THIRD*(pos_anti.y-pos_clock.y);
					//edgenormal.y = THIRD*(pos_clock.x-pos_anti.x);
					Area += 0.5*(pos_clock.x + pos_anti.x)*edgenormal.x;

					// Of course for sides we are not doing this quite right, by not
					// modifying the centroid.
					
				} else {
					if (pos_clock.x*pos_clock.x+pos_clock.y*pos_clock.y < DEVICE_INSULATOR_OUTER_RADIUS*DEVICE_INSULATOR_OUTER_RADIUS)
					{
						f64 shoelacedoubled = ( (pos_anti.x + info.pos.x)*(info.pos.y-pos_anti.y) // y_anti - y_clock --- pos_clock is the highest one.
											+ (pos_anti.x + pos_out.x)*(pos_anti.y-pos_out.y)
											+ (pos_out.x + info.pos.x)*(pos_out.y-info.pos.y));
						f64_vec2 Gradphi;						
						Gradphi.x = ( (phi_anti + phi)*(info.pos.y-pos_anti.y) // y_anti - y_clock --- pos_clock is the highest one.
									+ (phi_anti + phi_out)*(pos_anti.y-pos_out.y)
									+ (phi_out + phi)*(pos_out.y-info.pos.y) )
										/ shoelacedoubled;

						Gradphi.y = ( (phi_anti + phi)*(pos_anti.x-info.pos.x) // y_anti - y_clock --- pos_clock is the highest one.
									+ (phi_anti + phi_out)*(pos_out.x-pos_anti.x)
									+ (phi_out + phi)*(info.pos.x-pos_out.x) )
										/ shoelacedoubled;
								
						Lapphi += Gradphi.dot(edgenormal);
					
						Area += 0.5*(pos_clock.x + pos_anti.x)*edgenormal.x;
						
					} else {
						// Default case.	

						//shoelace =  (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
						//	      + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y);
						// same coeff to phi for grad_x integrated as on x_0 in shoelace:
						// same coeff to phi_anti for grad_y as on y_anti in shoelace:


						Lapphi += ( (phi - phi_out) * ( (pos_anti.y-pos_clock.y)*edgenormal.x
													  + (pos_clock.x-pos_anti.x)*edgenormal.y )  
								+	(phi_anti-phi_clock)*( (pos_out.y - info.pos.y)*edgenormal.x
													  + (info.pos.x - pos_out.x)*edgenormal.y) )

							/ ( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
							  + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) );


						if (index == 10000) {
							printf("10000: Lapphi %1.10E area %1.9E \nphi_clockoutanti %1.9E %1.9E %1.9E %1.9E\n",
								Lapphi,Area,
								phi_clock, phi_out,phi_anti, phi);
							printf("shoelace: %1.10E \n",
								( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
								  + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) ));
							printf("infopos %1.10E %1.10E \npos_out %1.10E %1.10E \npos_clk %1.10E %1.10E \npos_ant %1.10E %1.10E \n",
								 info.pos.x,info.pos.y,pos_out.x,pos_out.y,
								 pos_clock.x,pos_clock.y,pos_anti.x,pos_anti.y);
						};
						

						Area += 0.5*(pos_clock.x + pos_anti.x)*edgenormal.x;
					};
				};
			};*/
			// Get away with not repositioning edge_normal ends to insulator...			
			
			// Now go round:		
			pos_clock = pos_out;
			pos_out = pos_anti;
			phi_clock = phi_out;
			phi_out = phi_anti;		
		};
	};
	//	if (index == 25000) {
	//		printf("25000: Lapphi %1.10E area %1.9E phi_clockoutanti %1.9E %1.9E %1.9E %1.9E\n",Lapphi,Area,
	//			phi_clock, phi_out,phi_anti, phi);
	//		printf("pos %1.10E %1.10E %1.10E %1.10E %1.10E %1.10E \n",
	//			pos_clock.x,pos_clock.y, pos_out.x,pos_out.y,pos_anti.x,pos_anti.y);
			// phi = 0 , Lapphi = IND , pos is filled in, area is sensible value.		
	//	}

	/*	break;
	
	case OUTERMOST:
		// In this case we have e.g. if there are 4 neighs 0,1,2,3, then just 0-1-2, 1-2-3
		// We can happily drop the d/dtheta, it's not a big deal.

		// Start with neigh 0, not neigh N-1. End with neigh N-2 for centre.

		phi = p_phi_shared[threadIdx.x];
		memcpy(Indexneigh + MAXNEIGH_d*threadIdx.x, 
					pIndexNeigh + MAXNEIGH_d*index,
					MAXNEIGH_d*sizeof(long));
		memcpy(PBCneigh + MAXNEIGH_d*threadIdx.x, 
					pPBCNeigh + MAXNEIGH_d*index,
					MAXNEIGH_d*sizeof(char));
	
		long indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_clock = p_vertex_pos_shared[indexneigh-StartMajor];
			phi_clock = p_phi_shared[indexneigh-StartMajor];
		} else {
			info2 = p_info_sharing[indexneigh];
			pos_clock = info2.pos;
			phi_clock = p_phi[indexneigh];
		};

		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI)
			pos_clock = Anticlock_rotate2(pos_clock);
		if (PBC == NEEDS_CLOCK)
			pos_clock = Clockwise_rotate2(pos_clock);
		
		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 1];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = p_vertex_pos_shared[indexneigh-StartMajor];
			phi_out = p_phi_shared[indexneigh-StartMajor];
		} else {
			info2 = p_info_sharing[indexneigh];
			pos_out = info2.pos;
			phi_out = p_phi[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 1];
		if (PBC == NEEDS_ANTI)
			pos_out = Anticlock_rotate2(pos_out);
		if (PBC == NEEDS_CLOCK)
			pos_out = Clockwise_rotate2(pos_out);
		
#pragma unroll MAXNEIGH_d
		for (iNeigh = 1; iNeigh < info.neigh_len-1; iNeigh++)
		{
			int inext = iNeigh+1;
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + inext];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = p_vertex_pos_shared[indexneigh-StartMajor];
				phi_anti = p_phi_shared[indexneigh-StartMajor];
			} else {
				info2 = p_info_sharing[indexneigh];
				pos_anti = info2.pos;
				phi_anti = p_phi[indexneigh];
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + inext];
			if (PBC == NEEDS_ANTI)
				pos_anti = Anticlock_rotate2(pos_anti);
			if (PBC == NEEDS_CLOCK)
				pos_anti = Clockwise_rotate2(pos_anti);
			
			f64_vec2 edgenormal;
			edgenormal.x = THIRD*(pos_anti.y-pos_clock.y);
			edgenormal.y = THIRD*(pos_clock.x-pos_anti.x);

			Lapphi += ( (phi - phi_out) * ( (pos_anti.y-pos_clock.y)*edgenormal.x
										  + (pos_clock.x-pos_anti.x)*edgenormal.y )
					+	(phi_anti-phi_clock)*( (pos_out.y - info.pos.y)*edgenormal.x
										  + (info.pos.x - pos_out.x)*edgenormal.y) )
							  // was:
										// + (pos_anti.y-pos_clock.y)*edgenormal.y ) )
						// divide by shoelace :
				/ ( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
				  + (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) );
			// Dividing by 0.
			
			Area += 0.5*(pos_clock.x + pos_anti.x)*edgenormal.x;
		
			if (index == 11685) {
				printf("11685: Lapphi %1.8E Area %1.8E \n"
					"phi %1.5E phi_out %1.5E anti %1.5E clk %1.5E\n"
					"shoelace %1.5E\n",
					Lapphi,Area,
					phi,phi_out,phi_anti,phi_clock,
					( (info.pos.x - pos_out.x)*(pos_anti.y - pos_clock.y)
					+ (pos_anti.x - pos_clock.x)*(pos_out.y - info.pos.y) )
					);
				printf("info.pos %1.8E %1.8E\n"
					   "pos_out  %1.8E %1.8E\n"
					   "pos_anti %1.8E %1.8E\n"
					   "pos_clk  %1.8E %1.8E\n",
					   info.pos.x,info.pos.y, pos_out.x,pos_out.y,pos_anti.x,pos_anti.y,pos_clock.x,pos_clock.y);

				// Getting Lapphi -1.#IND0000E+000 Area -8.60120926E-004
				// phi 0.00000E+000 phi_out 0.00000E+000 anti 0.00000E+000 clk 0.00000E+000
			}
			// Now go round:		
			pos_clock = pos_out;
			pos_out = pos_anti;
			phi_clock = phi_out;
			phi_out = phi_anti;	
		};
		break;		
		
	};*/

	// integral of div f = sum of [f dot edgenormal]
	// ... so here we took integral of div grad f.
	// Look at previous code. Need to collect area and divide by it.
		
	p_Lap_phi[index] = Lapphi/Area;
	
}

__global__ void Kernel_GetThermalPressureTris(
	structural * __restrict__ p_info_sharing, // for vertex positions
	nT * __restrict__ p_nT_neut,
	nT * __restrict__ p_nT_ion,
	nT * __restrict__ p_nT_elec,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_perinfo,
	// Output:
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion, 
	f64_vec3 * __restrict__ p_MAR_elec
	)
{
	// Attention: code p_MAR_neut[index] += add; 
	// implies that we zero those arrays before we come here.
	
	__shared__ f64 p_nT_shared[SIZE_OF_MAJOR_PER_TRI_TILE];
	__shared__ f64_vec2 p_vertex_pos_shared[SIZE_OF_MAJOR_PER_TRI_TILE];
	
	long StartMajor = blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE;
	long EndMajor = StartMajor + SIZE_OF_MAJOR_PER_TRI_TILE; 
	long index =  threadIdx.x + blockIdx.x * blockDim.x;
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		nT nTtemp = p_nT_neut[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_nT_shared[threadIdx.x] = nTtemp.n*nTtemp.T;
		structural info = p_info_sharing[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_vertex_pos_shared[threadIdx.x] = info.pos;
	}
	__syncthreads(); 
	// Take grad on triangle:
	// first collect corner positions; if this is periodic triangle then we have to rotate em.
	LONG3 corner_index;
	f64_vec2 edge_normal0, edge_normal1, edge_normal2;			
	CHAR4 perinfo = p_tri_perinfo[index];
	if (perinfo.flag == DOMAIN_TRIANGLE) { // ?
		corner_index = p_tri_corner_index[index];
		// Do we ever require those and not the neighbours?
		// Yes - this time for instance.
		f64 nT0, nT1, nT2;
		f64_vec2 pos0, pos1, pos2;
		
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			pos0 = p_vertex_pos_shared[corner_index.i1-StartMajor];
			nT0 = p_nT_shared[corner_index.i1-StartMajor];
		} else {
			// have to load in from global memory:
			structural info = p_info_sharing[corner_index.i1];
			pos0 = info.pos;
			nT nTtemp = p_nT_neut[corner_index.i1];
			nT0 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			pos1 = p_vertex_pos_shared[corner_index.i2-StartMajor];
			nT1 = p_nT_shared[corner_index.i2-StartMajor];
		} else {
			// have to load in from global memory:
			structural info = p_info_sharing[corner_index.i2];
			pos1 = info.pos;
			nT nTtemp = p_nT_neut[corner_index.i2];
			nT1 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			pos2 = p_vertex_pos_shared[corner_index.i3-StartMajor];
			nT2 = p_nT_shared[corner_index.i3-StartMajor];
		} else {
			// have to load in from global memory:
			structural info = p_info_sharing[corner_index.i3];
			pos2 = info.pos;
			nT nTtemp = p_nT_neut[corner_index.i3];
			nT2 = nTtemp.n*nTtemp.T;
		}
		
		// In this case which ones are periodic?
		// Should we just store per flags?
		// How it should work:
		// CHAR4 perinfo: periodic, per0, per1, per2;
		if (perinfo.per0 == NEEDS_ANTI) 
			pos0 = Anticlock_rotate2(pos0);
		if (perinfo.per0 == NEEDS_CLOCK)
			pos0 = Clockwise_rotate2(pos0);
		if (perinfo.per1 == NEEDS_ANTI)
			pos1 = Anticlock_rotate2(pos1);
		if (perinfo.per1 == NEEDS_CLOCK)
			pos1 = Clockwise_rotate2(pos1);
		if (perinfo.per2 == NEEDS_ANTI)
			pos2 = Anticlock_rotate2(pos2);
		if (perinfo.per2 == NEEDS_CLOCK)
			pos2 = Clockwise_rotate2(pos2);
		
		// Integral of grad... average phi on edge . edgenormal
		// This should give the same result as the plane passing through
		// the 3 corners -- a few simple examples suggest yes.
		
		edge_normal0.x = pos2.y-pos1.y;
		edge_normal0.y = pos1.x-pos2.x;
		// Got to make sure it points out. How? Have to take
		// dot product with vector to the opposing point
		if (edge_normal0.dot(pos0-pos1) > 0.0) {
			// points to opposing point - wrong way
			edge_normal0.x = -edge_normal0.x;
			edge_normal0.y = -edge_normal0.y;
		}
		edge_normal1.x = pos2.y-pos0.y;
		edge_normal1.y = pos0.x-pos2.x;
		if (edge_normal1.dot(pos1-pos0) > 0.0) {
			edge_normal1.x = -edge_normal1.x;
			edge_normal1.y = -edge_normal1.y;
		}
		edge_normal2.x = pos1.y-pos0.y;
		edge_normal2.y = pos0.x-pos1.x;
		if (edge_normal2.dot(pos2-pos0) > 0.0) {
			edge_normal2.x = -edge_normal2.x;
			edge_normal2.y = -edge_normal2.y;
		};
		f64_vec2 grad_nT_integrated = 
			( 0.5*(nT1 + nT2)*edge_normal0 // opposite phi0
			+ 0.5*(nT0 + nT2)*edge_normal1
			+ 0.5*(nT1 + nT0)*edge_normal2 );		
		
		// Grad of phi on tri is grad for this minor within the tri:
		//p_grad_nT_neut_integrated[index] = grad_nT_integrated;
		// NOTE WE DO NOW DIVIDE BY PARTICLE MASS
		f64_vec3 add(-grad_nT_integrated.x/m_n,
			         -grad_nT_integrated.y/m_n,
					 0.0); // MINUS
		p_MAR_neut[index] += add; 
		
	} else {
		if (perinfo.flag == CROSSING_INS) {
			
			// We don't know if it's got 1 point outside ins or 2.
			// If 1 then not a lot we can do ??

			// Contribute zero to MAR for now...
			
		} else {
						
			// leave MAR unaffected
		};
	}
	__syncthreads();
	// Now load in ion nT info:
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		nT nTtemp = p_nT_ion[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_nT_shared[threadIdx.x] = nTtemp.n*nTtemp.T;
	}
	__syncthreads();
	
	if (perinfo.flag == DOMAIN_TRIANGLE) { // ?		
		f64 nT0, nT1, nT2;		
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			nT0 = p_nT_shared[corner_index.i1-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_ion[corner_index.i1];
			nT0 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			nT1 = p_nT_shared[corner_index.i2-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_ion[corner_index.i2];
			nT1 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			nT2 = p_nT_shared[corner_index.i3-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_ion[corner_index.i3];
			nT2 = nTtemp.n*nTtemp.T;
		}
				
		// Integral of grad... average phi on edge . edgenormal
		// This should give the same result as the plane passing through
		// the 3 corners -- a few simple examples suggest yes.
		
		f64_vec2 grad_nT_integrated = 
			( 0.5*(nT1 + nT2)*edge_normal0 // opposite phi0
			+ 0.5*(nT0 + nT2)*edge_normal1
			+ 0.5*(nT1 + nT0)*edge_normal2 );		
		
		// Grad of phi on tri is grad for this minor within the tri:
		//p_grad_nT_ion_integrated[index] = grad_nT_integrated;
		f64_vec3 add(-grad_nT_integrated.x/m_ion,
					 -grad_nT_integrated.y/m_ion,
					 0.0);
		p_MAR_ion[index] += add;
		
	} else {
		f64_vec2 zero(0.0,0.0);
		//p_grad_nT_ion_integrated[index] = zero;
	}
	__syncthreads();
	
	if (threadIdx.x < SIZE_OF_MAJOR_PER_TRI_TILE)
	{
		nT nTtemp = p_nT_elec[blockIdx.x*SIZE_OF_MAJOR_PER_TRI_TILE + threadIdx.x];
		p_nT_shared[threadIdx.x] = nTtemp.n*nTtemp.T;
	}
	__syncthreads();

	if (perinfo.flag == DOMAIN_TRIANGLE) { // ?
		
		f64 nT0, nT1, nT2;		
		if ((corner_index.i1 >= StartMajor) && (corner_index.i1 < EndMajor))
		{
			nT0 = p_nT_shared[corner_index.i1-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_elec[corner_index.i1];
			nT0 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i2 >= StartMajor) && (corner_index.i2 < EndMajor))
		{
			nT1 = p_nT_shared[corner_index.i2-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_elec[corner_index.i2];
			nT1 = nTtemp.n*nTtemp.T;
		}
		if ((corner_index.i3 >= StartMajor) && (corner_index.i3 < EndMajor))
		{
			nT2 = p_nT_shared[corner_index.i3-StartMajor];
		} else {
			// have to load in from global memory:
			nT nTtemp = p_nT_elec[corner_index.i3];
			nT2 = nTtemp.n*nTtemp.T;
		}
				
		// Integral of grad... average phi on edge . edgenormal
		// This should give the same result as the plane passing through
		// the 3 corners -- a few simple examples suggest yes.
		
		f64_vec2 grad_nT_integrated = 
			( 0.5*(nT1 + nT2)*edge_normal0 // opposite phi0
			+ 0.5*(nT0 + nT2)*edge_normal1
			+ 0.5*(nT1 + nT0)*edge_normal2 );		
		
		// Grad of phi on tri is grad for this minor within the tri:
		//p_grad_nT_elec_integrated[index] = grad_nT_integrated;
		f64_vec3 add(-grad_nT_integrated.x/m_e,
					 -grad_nT_integrated.y/m_e,
					 0.0);
		p_MAR_elec[index] += add;
	} else {
		
	}
}



__global__ void Kernel_Advance_Antiadvect_phidot(
	f64 * __restrict__ p_phidot,
	f64_vec2 * __restrict__ p_v_overall,
	f64 h_use,
	f64_vec2 * __restrict__ p_grad_phidot,
	f64 * __restrict__ p_Lap_phi,
	nT * __restrict__ p_nT_ion,
	nT * __restrict__ p_nT_elec,
	// out:
	f64 * __restrict__ p_phidot_out
	)
{
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec2 move = h_use*p_v_overall[index];
	f64 Lap_phi = p_Lap_phi[index];
	nT nT_ion = p_nT_ion[index];
	nT nT_elec = p_nT_elec[index];
	f64 phidot = p_phidot[index];
	f64_vec2 grad_phidot = p_grad_phidot[index];
	
	// What it has to do:
	p_phidot_out[index] = 
		phidot + move.dot(grad_phidot)
			+ h_use*csq*(Lap_phi + FOURPI_Q*(nT_ion.n-nT_elec.n));
	
	if (index == 10000) {
		printf("phidot[10000] %1.10E movedot %1.10E Lapphi %1.10E \n",
			phidot,move.dot(grad_phidot),Lap_phi);
	};
	// CHECK SIGNS
	// We are giving the value at the moved point.
	
	// The existence of this routine is a clear inefficiency.
	// It's basically so that computing grad_phi can be separate and repeated.
	// Could we combine it with Get_Lap_phi_on_major?
	
}

__global__ void Kernel_Advance_Antiadvect_phi
(
	structural * __restrict__ p_info,
	f64 V,
	f64 * __restrict__ p_phi,
	f64_vec2 * p_v_overall_major, 
	f64 h_use,
	f64_vec2 * __restrict__ p_grad_phi_major,
	f64 * __restrict__ p_phidot	, 
	f64 * __restrict__ p_phi_out
	)
{
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	f64_vec2 move = h_use*p_v_overall_major[index];
	f64 phidot = p_phidot[index];
	f64 phi = p_phi[index];
	f64_vec2 grad_phi = p_grad_phi_major[index];
	
	f64 rr = info.pos.x*info.pos.x+info.pos.y*info.pos.y;
	f64 result;
	if (rr <= 2.8*2.8) {
		result = -V; // > 0
	} else {
		if (rr >= 4.6*4.6) {
			result = V;
		} else {
			result = phi + move.dot(grad_phi) + h_use*phidot;
		};
	};
		
	p_phi_out[index] = result; 
		
}

__global__ void Kernel_Antiadvect_A_allminor
(
	f64_vec3 * __restrict__ p_A,
	f64_vec2 * __restrict__ p_v_overall_minor,
	f64 h_use,
	f64_vec2 * __restrict__ p_grad_Ax,
	f64_vec2 * __restrict__ p_grad_Ay,
	f64_vec2 * __restrict__ p_grad_Az,
	f64_vec3 * __restrict__ p_A_out,
	bool bAdd,
	f64_vec3 * __restrict__ p_Adot
 )
{
	long index = blockDim.x*blockIdx.x + threadIdx.x;
	f64_vec2 move = h_use*p_v_overall_minor[index];
	
	f64_vec3 A_out;
	f64_vec3 A_in = p_A[index];
	A_out.x = A_in.x + move.dot(p_grad_Ax[index]);
	A_out.y = A_in.y + move.dot(p_grad_Ay[index]);
	A_out.z = A_in.z + move.dot(p_grad_Az[index]);
	if (bAdd) {
		f64_vec3 Adot = p_Adot[index];
		A_out += h_use*Adot;
	}
	p_A_out[index] = A_out;
}

__global__ void Kernel_Ionisation(
		   f64 const h,
		   structural * __restrict__ p_info,
		   f64 * __restrict__ p_area,

			nT * __restrict__ p_nT_neut_src,
			nT * __restrict__ p_nT_ion_src,
			nT * __restrict__ p_nT_elec_src,

			nT * __restrict__ p_nT_neut_use, 
			nT * __restrict__ p_nT_ion_use, 
			nT * __restrict__ p_nT_elec_use, 
			
			nn * __restrict__ p_nn_ionise_recombine,
			// Where are these used and how to avoid storing?
			// There isn't a way: we have to spread this information out to minor cells.
			
			bool b2ndpass
		   )
{
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	structural info = p_info[index]; // 3 doubles?
	
	nT nT_elec_use, nT_ion_use, nT_neut_use;   
	nT nT_elec_src, nT_ion_src, nT_neut_src;   
	//f64 n_n_plus, n_ion_plus, n_e_plus; 
	nn nirec;
	
	nT_elec_src = p_nT_elec_src[index];
	nT_neut_src = p_nT_neut_src[index];
	nT_ion_src = p_nT_ion_src[index];
		
	if (b2ndpass) {
		nT_elec_use = p_nT_elec_use[index];
		nT_ion_use = p_nT_ion_use[index];
		nT_neut_use = p_nT_neut_use[index];
	} else {
		nT_elec_use = nT_elec_src;
		nT_ion_use = nT_ion_src;
		nT_neut_use = nT_neut_src;
	};
		
	if (info.flag == DOMAIN_VERTEX) 
	{
		// . Do ionisation --> know how much ionisation change in mom 
		// (or v) of each species and rate of adding to T.
		{
			// My clever way: anticipate some change only in some of the
			// T values. But use estimated T 1/2 on the 2nd pass.

			f64 S, R, sqrtTeeV;
			if (b2ndpass == 0) {
				// Use a deliberate underestimate that takes into account some
				// expected change in Te from ionisation:		
				// For sqrt(T) we use sqrt((T_k+T_k+1)/2).
				f64 T_eV_k = nT_elec_src.T/kB; 
				f64 buildingblock = 1.0e-5*exp(-13.6/T_eV_k)/
								(13.6*(6.0*13.6+T_eV_k));
				buildingblock = buildingblock*buildingblock;
				f64 temp = 0.25*h*nT_neut_src.n*TWOTHIRDS*13.6*buildingblock;
				S = - temp + sqrt( temp*temp + T_eV_k*buildingblock );
				
				// The 2nd pass, T will be a little less if ionisation
				// is important, and we then go ahead and use that.
				
				sqrtTeeV = sqrt(T_eV_k);
				R = nT_elec_src.n*8.75e-27/
					((T_eV_k*T_eV_k)*(T_eV_k*T_eV_k)*sqrtTeeV);	// take n_i*n_e*R
										
				// Nothing fancy for recombination rate.
				// It should only be a problem in case that we are weakly ionised,
				// and at least in that case the first go will be limited by the
				// measure to avoid n < 0.
				
				nirec.n_ionise = nT_neut_src.n*nT_elec_src.n*S*h;
				nirec.n_recombine = nT_elec_src.n*nT_ion_src.n*R*h;
				f64 netionise = nirec.n_ionise - nirec.n_recombine;
			
				if ((nT_neut_src.n-netionise < 0.0) || (nT_ion_src.n+netionise < 0.0) || (nT_elec_src.n+netionise < 0.0))
				{		
					// in denom goes n_ionise/n_n and n_recombine/n_lowest
					if (nT_ion_src.n < nT_elec_src.n) {
						f64 denom = (1.0 + h*nT_elec_src.n*S + h*nT_elec_src.n*R);
						nirec.n_ionise /= denom;
						nirec.n_recombine /= denom;
					} else {
						f64 denom = (1.0 + h*nT_elec_src.n*S + h*nT_ion_src.n*R);
						nirec.n_ionise /= denom;
						nirec.n_recombine /= denom;
					};
					netionise = nirec.n_ionise - nirec.n_recombine;
				};
				
			//	n_ion_plus = nT_ion_src.n + netionise;
			//	n_n_plus = nT_neut_src.n - netionise;
			//	n_e_plus = nT_elec_src.n + netionise;
				
			} else {
				// Use Te_1/2 throughout:			
				f64 T_eV = nT_elec_use.T/kB; // large negative
				sqrtTeeV = sqrt(T_eV);
				S = 1.0e-5*sqrtTeeV*exp(-13.6/T_eV)/(13.6*(6.0*13.6+T_eV));			
				R = nT_elec_use.n*8.75e-27/((T_eV*T_eV)*(T_eV*T_eV)*sqrtTeeV);
				
				nirec.n_ionise = nT_neut_use.n*nT_elec_use.n*S*h;
				nirec.n_recombine = nT_elec_use.n*nT_ion_use.n*R*h;
				f64 netionise = nirec.n_ionise - nirec.n_recombine;
							
				// Am I right that they are getting wiped out -- so that makes a difference here
				// drastically reducing the amt of recombination because it recognises there are less there.
				
				if ((nT_neut_src.n-netionise < 0.0) || (nT_ion_src.n+netionise < 0.0) || (nT_elec_src.n+netionise < 0.0))
				{	
					f64 denom;
					if (nT_ion_src.n < nT_elec_src.n) {
						denom = (1.0 + h*nT_elec_src.n*S + h*nT_elec_src.n*R);
					} else {
						denom = (1.0 + h*nT_elec_src.n*S + h*nT_ion_src.n*R);
					};				
					nirec.n_ionise = nT_neut_src.n*nT_elec_src.n*S*h/denom;
					nirec.n_recombine = nT_elec_src.n*nT_ion_src.n*R*h/denom;			
					netionise = nirec.n_ionise - nirec.n_recombine;
				};
				
	//			n_ion_plus = nT_ion_src.n + netionise;
	//			n_n_plus = nT_neut_src.n - netionise;
	//			n_e_plus = nT_elec_src.n + netionise;
			};
		} // end of "do ionisation"
		// We now got: n_ion_plus, n_n_plus, n_e_plus, n_ionise, n_recombine.
	} else {
		// (info.flag == DOMAIN_VERTEX)
	//	n_e_plus = nT_elec_src.n;
	//	n_n_plus = nT_neut_src.n;
	//	n_ion_plus = nT_ion_src.n;
	};
	
	// Save output...
	p_nn_ionise_recombine[index] = nirec;
	
	// nT_elec_out ? We do not need it for midpoint v routine, because
	// we load n_ionise_recombine to recreate.
	//p_nT_elec_out[index].n = n_e_plus;
	//p_nT_ion_out[index].n = n_ion_plus;
	//p_nT_neut_out[index].n = n_n_plus;
	// Therefore we should probably use the midpoint routine to do this save,
	// because we will be doing a save-off of T anyway.
	// NOPE - midpoint routine applies to minor not major.
	
}

// Note: unroll can increase register pressure!

__global__ void Kernel_Midpoint_v_and_Adot (
	f64 const h,

	CHAR4 * __restrict__ p_tri_perinfo,
	nT * __restrict__ p_nT_neut_src,
	nT * __restrict__ p_nT_ion_src, 
	nT * __restrict__ p_nT_elec_src, 
	// n_k appears because it is needed as part of the midpoint step.
	// Am I serious about __restrict__ ? Yes if pass 0 as use on 1st pass

	// On 2nd pass you do need different n for half time?
	// n basically changes with ionisation
	nT * __restrict__ p_nT_neut_use, // k+1/2 on 2nd pass; on 1st pass n is adjusted by ionisation - correct?
	nT * __restrict__ p_nT_ion_use, // k or k+1/2
	nT * __restrict__ p_nT_elec_use, // k or k+1/2 --- for forming nu etc...
	nn * __restrict__ p_nn_ionise_recombine,
	// Have to load 2 additional doubles due to doing ionisation outside.
	
	f64_vec2 * __restrict__ p_tri_centroid, 
	// CAN WE MAKE THIS BE EXTENDED TO APPLY FOR CENTRAL VALUES ALSO?
	// For now have to include separate set of positions:
	structural * __restrict__ p_info,

	// We use this when we assume we are adding to v's momentum and for doing the n_k+1 part of the midpt formula
	f64_vec3 * __restrict__ p_B,
	f64_vec3 * __restrict__ p_v_n_src,
	f64_vec3 * __restrict__ p_v_ion_src,
	f64_vec3 * __restrict__ p_v_e_src,
	f64 * __restrict__ p_area,			// It's assumed to be area_k+1 but I guess it's area_k+1/2 ... too bad?
	f64_vec2 * __restrict__ p_grad_phi_half,
	f64_vec3 * __restrict__ p_Lap_A_half,
	f64_vec3 * __restrict__ p_Adot_k,

	f64_vec3 * __restrict__ p_MomAdditionRate_neut,
	f64_vec3 * __restrict__ p_MomAdditionRate_ion,
	f64_vec3 * __restrict__ p_MomAdditionRate_elec,
							// OKay let's check out if aTP was even correct.
	f64_vec2 * __restrict__ p_grad_Te,

	f64_vec3 * __restrict__ p_v_neut_out,
	f64_vec3 * __restrict__ p_v_ion_out,
	f64_vec3 * __restrict__ p_v_elec_out,

	f64 * __restrict__ p_resistive_heat_neut, // additions to NT 
	f64 * __restrict__ p_resistive_heat_ion,
	f64 * __restrict__ p_resistive_heat_elec,
	f64_vec3 * p_dAdt_out,
	
	bool b2ndPass,
	f64 const EzTuning,
	
	f64 * __restrict__ p_Iz,
	f64 * __restrict__ p_sigma_zz
	)
{
	// Still going to need to know the Ez linear relationship
	// from this function:
	__shared__ f64 Iz[threadsPerTileMinor];
	__shared__ f64 sigma_zz[threadsPerTileMinor];

	// Only dimension here what will have def been used by the time we hit the 
	// largest memory footprint:	
	f64 nu_ne_MT_over_n, nu_ni_MT_over_n, nu_eiBar, nu_ieBar, nu_eHeart; // 5 double
	Vector3 omega_ce, v_e_k, v_ion_k, v_n_k, v_n_0; // 15 double
	Vector3 v_ion_plus, v_n_plus, v_e_plus; // 9 doubles
	CHAR4 per_info;
	
	f64 n_e_plus, n_ion_plus, n_n_plus;
	nT nT_elec_use,nT_ion_use,nT_neut_use;
	nn n_ionrec;
	Vector3 Lap_A_half;

	long index = threadIdx.x + blockIdx.x*blockDim.x;
	
	omega_ce = p_B[index];
	omega_ce *= eovermc; // Trying to avoid 3 separate accesses [which would not even be contig!!]
	
	// Shane Cook p.176 We do not need to put the reads where they are needed, can put first.
	
	// It is UNCLEAR where I read that we have to put reads outside of branches.
	// I do not find corroborating sources.
	// index < Nverts is more or less guaranteed but it would of course be nice not to
	// do fetches for info.flag != DOMAIN_VERTEX.
	
	v_e_k = p_v_e_src[index];
	v_ion_k = p_v_ion_src[index];
	v_n_k = p_v_n_src[index];
	f64 area = p_area[index];
	f64_vec2 centroid;
	if (index < BEGINNING_OF_CENTRAL)
	{
		centroid = p_tri_centroid[index]; // position - only used for shaping Ez I think
	} else {
		centroid = p_info[index-BEGINNING_OF_CENTRAL].pos;
	};
	
	per_info = p_tri_perinfo[index];
	n_ionrec = p_nn_ionise_recombine[index];
	
	//n_n_plus = p_n_n_plus[index].n;
	//n_ion_plus = p_n_ion_plus[index].n;
	//n_e_plus = p_n_e_plus[index].n;

	if ((OUTPUT) && (index == REPORT)) {
		printf("v_k %1.5E %1.5E %1.5E\n",v_n_k.z,v_ion_k.z,v_e_k.z);
		printf("Bxy %1.5E %1.5E omega.z %1.5E \n",omega_ce.x/eovermc,omega_ce.y/eovermc,omega_ce.z);
	};
	
	if ((per_info.flag == DOMAIN_TRIANGLE)) // try loading data inside, outside branch...
	{
		{
			nT nT_elec_src, nT_ion_src, nT_neut_src; // register pressure?
			
			// data from time t_k:
			nT_elec_src = p_nT_elec_src[index];
			nT_ion_src = p_nT_ion_src[index];
			nT_neut_src = p_nT_neut_src[index];
			// Question whether these should be inside brace.

			// n_n_plus means neutral density at t_k+1
			n_n_plus = nT_neut_src.n + n_ionrec.n_recombine-n_ionrec.n_ionise;
			n_ion_plus = nT_ion_src.n + n_ionrec.n_ionise-n_ionrec.n_recombine;
			n_e_plus = nT_elec_src.n + n_ionrec.n_ionise-n_ionrec.n_recombine;
	
			if (b2ndPass) {
				// On the 2nd+3rd pass,
				// we prefer to have n,T at t_k+1/2 for use in time-derivatives:
				nT_elec_use = p_nT_elec_use[index];
				nT_ion_use = p_nT_ion_use[index];
				nT_neut_use = p_nT_neut_use[index];
			} else {
				nT_elec_use = nT_elec_src;
				nT_ion_use = nT_ion_src;
				nT_neut_use = nT_neut_src;
			};
	
			{
				// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
				f64 sqrt_Te,ionneut_thermal, electron_thermal,
						lnLambda, s_in_MT, s_en_MT, s_en_visc;
					
				sqrt_Te = sqrt(nT_elec_use.T);
				ionneut_thermal = sqrt(nT_ion_use.T/m_ion+nT_neut_use.T/m_n); // hopefully not sqrt(0)
				electron_thermal = sqrt_Te*over_sqrt_m_e;
				lnLambda = Get_lnLambda_d(nT_ion_use.n,nT_elec_use.T);
				
				s_in_MT = Estimate_Neutral_MT_Cross_section(nT_ion_use.T*one_over_kB);
				s_en_MT = Estimate_Neutral_MT_Cross_section(nT_elec_use.T*one_over_kB);
				s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(nT_elec_use.T*one_over_kB);
				// Need nu_ne etc to be defined:
				nu_ne_MT_over_n = s_en_MT*electron_thermal; // have to multiply by n_e for nu_ne_MT
				nu_ni_MT_over_n = s_in_MT*ionneut_thermal;
				nu_eiBar = nu_eiBarconst*kB_to_3halves*nT_ion_use.n*lnLambda/(nT_elec_use.T*sqrt_Te);
				nu_ieBar = nT_elec_use.n*nu_eiBar/nT_ion_use.n;
				nu_eHeart = 1.87*nu_eiBar + nT_neut_use.n*s_en_visc*electron_thermal;
			}								
			
			f64 Beta_ni, Beta_ne;
			
			// Get v_n (v_e, v_i):
			Beta_ne = h*0.5*(m_e/(m_e+m_n))*nu_ne_MT_over_n*nT_elec_use.n; // avoid division with a #define!
			Beta_ni = h*0.5*(m_ion/(m_ion+m_n))*nu_ni_MT_over_n*nT_ion_use.n;
			v_n_0 = 
					// ionisation addition to neutral momentum:
					((nT_neut_src.n - n_ionrec.n_ionise)*v_n_k
					+ n_ionrec.n_recombine*(m_i_over_m_n*v_ion_k+m_e_over_m_n*v_e_k))/n_n_plus;
						- Beta_ne*(v_n_k-v_e_k)
						- Beta_ni*(v_n_k-v_ion_k);

			if ((OUTPUT) && (index == REPORT)) printf("vn0 %1.8E .. %1.8E\n",v_n_0.x,				
					n_ionrec.n_recombine*(m_i_over_m_n*v_ion_k.x + m_e_over_m_n*v_e_k.x)/n_n_plus);
			
			{
				//Vector2 grad_nT_neut = p_grad_nT_neut[index];
				Vector3 MomAdditionRate = p_MomAdditionRate_neut[index];
					// We can avoid a fetch if we just store the sum(diff) of these in 1 Vector3
					// But be careful : how do we work out visc heating? Do that first.
				
				// We stored [gradnTintegrated / m_s] = d/dt Nv
				
				v_n_0 += h*( //-grad_nT_neut.x + ViscMomAdditionRate_neut.x)/(n_n_plus*m_n);
							 MomAdditionRate/(n_n_plus*area));
				
				f64 over = 1.0/(1.0 + Beta_ne + Beta_ni);
				v_n_0 *= over;
				Beta_ni *= over;
				Beta_ne *= over;
								
				if ((OUTPUT) && (index == REPORT)) printf(
					"Beta_ni %1.8E Beta_ne %1.8E n_ionise %1.6E n_recomb %1.6E \n"
					"n_n_plus %1.8E MAR %1.8E %1.8E \n"
					"area %1.8E  \n",
					Beta_ni, Beta_ne, n_ionrec.n_ionise, n_ionrec.n_recombine, n_n_plus,
					MomAdditionRate.x,MomAdditionRate.y,
					area);
				
				if ((OUTPUT) && (index == REPORT)) printf("vn0 %1.8E \n",v_n_0.x);
			} 		
			
			// Now get v_i (v_e):
			f64 total = 
				(nu_eHeart*nu_eHeart + omega_ce.x*omega_ce.x+omega_ce.y*omega_ce.y+omega_ce.z*omega_ce.z);
			
			Vector3 vec_e, vec_i, dAdt_k;
			f64 EzShape;
			{
				Vector2 grad_phi, GradTe;
				Vector3 MomAdditionRate; // We could use it first as this, union with dAdt_k
				
				// Load in more input data:
				grad_phi = p_grad_phi_half[index];
				Lap_A_half = p_Lap_A_half[index];
				dAdt_k = p_Adot_k[index];
				MomAdditionRate = p_MomAdditionRate_ion[index];
				// (TRY putting the loads outside the branch to see what happens.)
				// ***************************************************************
				
				EzShape = GetEzShape(centroid.modulus()); // It goes to 0 at the outer radius.
				
				// Set up most of vec_e, vec_i here; this vec_i may be "d" in the documents.
				vec_i =     // Ionisation affected v_i_k:
					((nT_ion_src.n-n_ionrec.n_recombine)*v_ion_k + n_ionrec.n_ionise*v_n_k)/n_ion_plus;
					
				if ((OUTPUT) && (index == REPORT))	printf("vec_i %1.8E %1.8E %1.8E\n",vec_i.x,vec_i.y,vec_i.z);
					
				vec_i -= h*0.5*moverM*omega_ce.cross(v_ion_k);		

				if ((OUTPUT) && (index == REPORT))	printf("vec_i %1.8E %1.8E %1.8E\n",vec_i.x,vec_i.y,vec_i.z);
				
				vec_i +=
					  h*qoverM*( //- grad_phi [[below]]
								 - dAdt_k/c - h*c*0.5*Lap_A_half 
								 - h*M_PI*e*(nT_ion_src.n*v_ion_k - nT_elec_src.n*v_e_k))
					// nu_ni/n * n_n = nu_in
					- h*0.5*(m_n/(m_ion+m_n))*nu_ni_MT_over_n*nT_neut_use.n*(v_ion_k-v_n_k-v_n_0)
					- h*0.5*moverM*nu_ieBar*(v_ion_k-v_e_k);
				
				if ((OUTPUT) && (index == REPORT)) printf("dAdt_kz/c  %1.8E  hc0.5 Lap_Az_half %1.8E \n"
					"n_ion vx_ion_k - n_e vx_e_k %1.8E \n"
					" n-i term x %1.8E\n"
					" nu_ieBar term x %1.8E \n",
dAdt_k.z/c,
h*c*0.5*Lap_A_half.z, nT_ion_src.n*v_ion_k.x - nT_elec_src.n*v_e_k.x, h*0.5*(m_n/(m_ion+m_n))*nu_ni_MT_over_n*nT_neut_use.n*(v_ion_k.x-v_n_k.x-v_n_0.x),
h*0.5*moverM*nu_ieBar*(v_ion_k.x-v_e_k.x));

				if ((OUTPUT) && (index == REPORT)) printf("nu_ni_MT_over_n %1.6E nT_neut_use.n %1.6E v_n_k.x %1.6E v_n_0.x %1.6E \n",
					nu_ni_MT_over_n,nT_neut_use.n,v_n_k.x,
					v_n_0.x // the culprit
					);


				if ((OUTPUT) && (index == REPORT))	printf("vec_i %1.8E %1.8E %1.8E\n",vec_i.x,vec_i.y,vec_i.z);
				
				vec_i.x -= h*qoverM*grad_phi.x;
				vec_i.y -= h*qoverM*grad_phi.y;
				vec_i.z += h*qoverM*EzShape*EzTuning;
				
				if ((OUTPUT) && (index == REPORT))	printf("vec_i %1.8E %1.8E %1.8E\n",vec_i.x,vec_i.y,vec_i.z);
				
				// -grad_nT_ion.x + ViscMomAdditionRate_ion.x
				
//				Vector3 ViscMomAdditionRate_ion = p_visc_mom_addition_rate_ion[index];
				// We can avoid a fetch if we just store the sum(diff) of these in 1 Vector3
				// But be careful : how do we work out visc heating? It has to be fetched separately anyway.
				vec_i += h*((MomAdditionRate)/(n_ion_plus*area));
				
				if ((OUTPUT) && (index == REPORT))	printf("vec_i w/press %1.5E %1.5E %1.5E\n",vec_i.x,vec_i.y,vec_i.z);
				
				
				// We almost certainly should take v += (ViscMomAddition/n_k+1)
				// The same applies to grad_nT_ion : integrate this over [t_k,t_k+1]
				// and we get the addition to momentum.
				
				// Load data for electron:
				MomAdditionRate = p_MomAdditionRate_elec[index];
				GradTe = p_grad_Te[index];
				
				// Add thermal force on ions:				
				f64 fac = 1.5*h*(nu_ieBar/(m_ion*nu_eHeart*total));
				
				vec_i.x += fac*	(// (Upsilon.xx)*GradTe.x  + Upsilon.xy*GradTe.y
							  (omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart)*GradTe.x
							+ (omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z)*GradTe.y);
							// divide by (nu*nu+omega*omega) already in fac
				vec_i.y += fac*	( 
							  (omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z)*GradTe.x
							+ (omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart)*GradTe.y);
				vec_i.z += fac* (
							  (omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y)*GradTe.x
							+ (omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x)*GradTe.y);
				
		//		if (index == 15936)	printf("vec_i %1.5E \n",vec_i.z);
				
				// Add Upsilon part of collisional term:
				fac = h*0.5*0.9*moverM*nu_eiBar*nu_ieBar/(nu_eHeart*total);
				vec_i.x += fac*(	(omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart)*(v_ion_k.x-v_e_k.x)
							+	(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z)*(v_ion_k.y-v_e_k.y)
							+	(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y)*(v_ion_k.z-v_e_k.z)
							);
				vec_i.y += fac*(	(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z)*(v_ion_k.x-v_e_k.x)
							+	(omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart)*(v_ion_k.y-v_e_k.y)
							+	(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x)*(v_ion_k.z-v_e_k.z)
							);
				vec_i.z += fac*(	(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y)*(v_ion_k.x-v_e_k.x)
							+	(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x)*(v_ion_k.y-v_e_k.y)
							+	(omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart)*(v_ion_k.z-v_e_k.z)
							);
								
				if ((OUTPUT) && (index == REPORT)) {
					printf("%d vik %1.8E %1.8E %1.8E \nvec_i %1.8E %1.8E %1.8E \n",
						index,v_ion_k.x,v_ion_k.y,v_ion_k.z,vec_i.x,vec_i.y,vec_i.z);
				};
				










				vec_e = ((nT_elec_src.n-n_ionrec.n_recombine)*v_e_k + n_ionrec.n_ionise*v_n_k)/n_e_plus;
				
				if ((OUTPUT) && (index == REPORT)) printf("v_e_k %1.14E %1.14E %1.14E\n",v_e_k.x,v_e_k.y,v_e_k.z);
				
				vec_e += h*0.5*omega_ce.cross(v_e_k)
					- h*eoverm*(// -grad_phi // below
							- dAdt_k/c - h*c*0.5*Lap_A_half
							- h*M_PI*e*(nT_ion_src.n*v_ion_k - nT_elec_src.n*v_e_k))
					
					- 0.5*h*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n*(v_e_k-v_n_k-v_n_0)
					- 0.5*h*nu_eiBar*(v_e_k-v_ion_k);
				
				vec_e.x += h*eoverm*grad_phi.x ;
				vec_e.y += h*eoverm*grad_phi.y;
				vec_e.z += -h*eoverm*EzShape*EzTuning;
				
				//vec_e.x += h*( (-grad_nT_e.x )/(n_e_plus*m_e));
				
				vec_e += h*(MomAdditionRate/(n_e_plus*area)); // MAR = d/dt (Neve)
								
				// Add thermal force to electrons:			
				fac = -(1.5*h*nu_eiBar/(m_e*nu_eHeart*total));
					
				vec_e.x += fac*(
							  (omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart)*GradTe.x
							+ (omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z)*GradTe.y);
				vec_e.y += fac*(
							  (omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z)*GradTe.x
							+ (omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart)*GradTe.y);
				vec_e.z += fac*(
							  (omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y)*GradTe.x
							+ (omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x)*GradTe.y);
							
				if ((OUTPUT) && (index == REPORT)) {
					printf("vec_e total %1.14E %1.14E %1.14E\n",vec_e.x,vec_e.y,vec_e.z);
					printf("h*eoverm*grad_phi %1.14E %1.14E \n",h*eoverm*grad_phi.x,h*eoverm*grad_phi.y);
					printf("h*(MomAdditionRate/(n_e_plus*area)) %1.14E %1.14E \n",
						h*(MomAdditionRate.x/(n_e_plus*area)),
						h*(MomAdditionRate.y/(n_e_plus*area)));
					printf("h*0.5*omega_ce.cross(v_e_k).z %1.14E \n",
						h*0.5*(omega_ce.cross(v_e_k)).z);
					printf("h*eoverm*dAdt_k/c %1.14E \n" 
						"h*eoverm*h*c*0.5*Lap_A_half %1.14E\n"
						" h*eoverm*h*M_PI*e*() %1.14E\n"
						"0.5*h*...*(v_e_k-v_n_k-v_n_0) %1.14E\n"
						"0.5*h*nu_eiBar*(v_e_k-v_ion_k) %1.14E\n",
						h*eoverm*dAdt_k.z/c ,
						h*eoverm*h*c*0.5*Lap_A_half.z,
						h*eoverm*h*M_PI*e*(nT_ion_src.n*v_ion_k.z - nT_elec_src.n*v_e_k.z),
						// = 8e-18*2e17*(viz-vez) = 1.6 (viz-vez) = 1.6(-vez) = 1e6.
						// Where is the term that cancels its impact?
						0.5*h*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n*(v_e_k-v_n_k-v_n_0).z,
						0.5*h*nu_eiBar*(v_e_k-v_ion_k).z);
					printf("-h*eoverm*EzShape*EzTuning %1.14E\n",
						-h*eoverm*EzShape*EzTuning);
					printf("thermal contrib z %1.14E\n",
						fac*(
							  (omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y)*GradTe.x
							+ (omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x)*GradTe.y));
				};
				// Add Upsilon part of collisional term:
				fac = 0.5*h*0.9*nu_eiBar*nu_eiBar/(nu_eHeart*total);
				
				vec_e.x += fac*(
							  (omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart)*(v_e_k.x-v_ion_k.x)
							+ (omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z)*(v_e_k.y-v_ion_k.y)
							+ (omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y)*(v_e_k.z-v_ion_k.z));
				vec_e.y += fac*(
							  (omega_ce.y*omega_ce.x + nu_eHeart*omega_ce.z)*(v_e_k.x-v_ion_k.x)
							+ (omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart)*(v_e_k.y-v_ion_k.y)
							+ (omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x)*(v_e_k.z-v_ion_k.z));
				vec_e.z += fac*(
							  (omega_ce.z*omega_ce.x - nu_eHeart*omega_ce.y)*(v_e_k.x-v_ion_k.x)
							+ (omega_ce.z*omega_ce.y + nu_eHeart*omega_ce.x)*(v_e_k.y-v_ion_k.y)
							+ (omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart)*(v_e_k.z-v_ion_k.z));
			
				if ((OUTPUT) && (index == REPORT))
				{
					printf("contrib_[e-i] %1.14E\n",
							fac*(
						  (omega_ce.z*omega_ce.x - nu_eHeart*omega_ce.y)*(v_e_k.x-v_ion_k.x)
							+ (omega_ce.z*omega_ce.y + nu_eHeart*omega_ce.x)*(v_e_k.y-v_ion_k.y)
							+ (omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart)*(v_e_k.z-v_ion_k.z)));
				
					printf("vec_e %1.14E %1.14E %1.14E \n",
						vec_e.x,vec_e.y,vec_e.z);
				};
			}

			Tensor3 Tens1, Tens2, Tens3;
			// We are going to need Tens1, Tens2 again
			// BUT have to reallocate BECAUSE ...
			// we don't want them to be created prior to this.
			// and we can't stand to put heating in this same scope
			// which also has Tens3.

			// Tens1 is going to be "G"
			
			// Set Tens3 = Upsilon_eHeart:
			//// nu = nu_eHeart, omega = 
			//{
			//	f64 total = nu_eHeart*nu_eHeart+
			//				omega_ce.x*omega_ce.x + omega_ce.y*omega_ce.y
			//				+omega_ce.x*omega_ce.z;

			//	Upsilon_eHeart.xx = (nu_eHeart*nu_eHeart +omega_ce.x*omega_ce.x)/total;
			//	Upsilon_eHeart.xy = (omega_ce.x*omega_ce.y-nu_eHeart*omega_ce.z)/total;
			//	Upsilon_eHeart.xz = (omega_ce.x*omega_ce.z+nu_eHeart*omega_ce.y)/total;
			//	Upsilon_eHeart.yx = 
			//}

			// Upsilon is used 8 times.
			// But it would keep getting wiped out.
			// So it's a real problem. Storing it is another 9 doubles which is bad.
			// Try to edit to at least copy-paste the code...

			f64 fac = h*0.5*moverM*0.9*nu_eiBar*nu_ieBar/(nu_eHeart*total);
			// We inserted 1/total into fac.
			
			Tens1.xx = 1.0 
					 // + no contrib from omega_ci x
					   + (h*0.5*m_n/(m_ion+m_n))*nu_ni_MT_over_n*nT_neut_use.n
										*(1.0-Beta_ni)
					   + h*0.5*moverM*nu_ieBar
					   + h*h*e*e*M_PI* n_ion_plus / m_ion
						;
			Tens1.yy = Tens1.xx;
			Tens1.zz = Tens1.xx;
			Tens1.xx -= fac*//Upsilon_eHeart.xx/total; 
				// division by "total = nu*nu+omega*omega" is in fac.
							(nu_eHeart*nu_eHeart + omega_ce.x*omega_ce.x);
			Tens1.yy -= fac*(nu_eHeart*nu_eHeart + omega_ce.y*omega_ce.y);
			Tens1.zz -= fac*(nu_eHeart*nu_eHeart + omega_ce.z*omega_ce.z);
			
			Tens1.xy = -h*0.5*moverM*omega_ce.z
					 - fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
			Tens1.xz = h*0.5*moverM*omega_ce.y
					 - fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
			Tens1.yx = h*0.5*moverM*omega_ce.z
					 - fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
			Tens1.yz = -h*0.5*moverM*omega_ce.x
					 - fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
			Tens1.zx = -h*0.5*moverM*omega_ce.y
					 - fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
			Tens1.zy = h*0.5*moverM*omega_ce.x
					 - fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
			// ... replace omega_ci = omega_ce*moverM ... 
			
			// Formula for Upsilon_eHeart comes from Krook model subsection in model document.
			// We will prefer not to create omega_ci vector of course!!!
			
			Tens1.Inverse(Tens2); // Tens2 now = G^-1
						
			// Now create F:
			fac = h*0.5*0.9*nu_eiBar*nu_eiBar/(nu_eHeart*total);
			
			Tens3.xx = -h*0.5*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n*Beta_ni
							-h*0.5*nu_eiBar
								- (h*h*e*e*M_PI*over_m_e) * n_ion_plus;
			Tens3.yy = Tens3.xx;
			Tens3.zz = Tens3.xx;
			Tens3.xx += fac*(nu_eHeart*nu_eHeart + omega_ce.x*omega_ce.x);
			Tens3.yy += fac*(nu_eHeart*nu_eHeart + omega_ce.y*omega_ce.y);
			Tens3.zz += fac*(nu_eHeart*nu_eHeart + omega_ce.z*omega_ce.z);

			Tens3.xy = fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
			Tens3.xz = fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
			Tens3.yx = fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
			Tens3.yz = fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
			Tens3.zx = fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
			Tens3.zy = fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
			
			Tens1 = Tens3*Tens2; 

			// Contents now: { F G^-1, G^-1, F }
			// Now create the vector for v_e:
			// vec_e = d' - F G^-1 d

			vec_e -= Tens1*vec_i;
		
			if ((OUTPUT) && (index == REPORT))
				printf("modified vec_e \n %1.14E %1.14E %1.14E\n",
					vec_e.x,vec_e.y,vec_e.z);
		
			// Let's watch out:
			// this means if we change EzExt then we change vec_e.z and vec_i.z
			// directly, but we also change vec_e via vec_i.

			// We need to store that from this point because we are about to wipe out Tens1.

			Vector3 vec_e_effect_of_EzTuning;
			vec_e_effect_of_EzTuning.x = -Tens1.xz*(h*qoverM*EzShape);
			vec_e_effect_of_EzTuning.y = -Tens1.yz*(h*qoverM*EzShape);
			vec_e_effect_of_EzTuning.z = -h*eoverm*EzShape-Tens1.zz*(h*qoverM*EzShape);
			
			// Contents now: { F G^-1, G^-1, F }
			
			// Populate Tens3 as U. Multiply to get Tens2 = FG^-1 U
			
			Tens3.xx = -0.5*h*(m_n/(m_ion+m_n))*nu_ni_MT_over_n*nT_neut_use.n*Beta_ne 
						- 0.5*h*moverM*nu_ieBar
						- h*h*e*qoverM* M_PI *  n_e_plus;
			Tens3.yy = Tens3.xx;
			Tens3.zz = Tens3.xx;
			fac = 0.5*h*moverM*0.9*nu_eiBar*nu_ieBar/(nu_eHeart*total);
			Tens3.xx += fac*(omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart);
			Tens3.yy += fac*(omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart);
			Tens3.zz += fac*(omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart);
			
			Tens3.xy = fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
			Tens3.xz = fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
			Tens3.yx = fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
			Tens3.yz = fac*(omega_ce.z*omega_ce.y - nu_eHeart*omega_ce.x);
			Tens3.zx = fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
			Tens3.zy = fac*(omega_ce.z*omega_ce.y + nu_eHeart*omega_ce.x);
			
			// We really could do with storing Upsilon somehow.

			Tens2 = Tens1*Tens3;
			
			// Tens1 = V - F G^-1 U
			// V:
			Tens1.xx = 1.0 + h*0.5*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n
										*(1.0-Beta_ne)
						+ h*0.5*nu_eiBar + h*h*e*eoverm* M_PI* n_e_plus;
			Tens1.yy = Tens1.xx;
			Tens1.zz = Tens1.xx;
			fac = -0.5*h*0.9*nu_eiBar*nu_eiBar/(nu_eHeart*total);
			Tens1.xx += fac*(omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart);
			Tens1.yy += fac*(omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart);
			Tens1.zz += fac*(omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart);
			
			Tens1.xy = h*0.5*omega_ce.z
					 + fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
			Tens1.xz = -h*0.5*omega_ce.y
					 + fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
			Tens1.yx = -h*0.5*omega_ce.z
					 + fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
			Tens1.yz = h*0.5*omega_ce.x
					 + fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
			Tens1.zx = h*0.5*omega_ce.y
					 + fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
			Tens1.zy = -h*0.5*omega_ce.x
					 + fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
			
			if ((OUTPUT) && (index == REPORT))
				printf(	"nu_eiBar %1.14E n_e_plus %1.14E nu_en_MT %1.14E\n"
						"V \n%1.14E %1.14E %1.14E \n%1.14E %1.14E %1.14E \n%1.14E %1.14E %1.14E \n\n",
						nu_eiBar,n_e_plus,nu_ne_MT_over_n*nT_neut_use.n,
						Tens1.xx,Tens1.xy,Tens1.xz,
						Tens1.yx,Tens1.yy,Tens1.yz,
						Tens1.zx,Tens1.zy,Tens1.zz);
		
			Tens1 -= Tens2; // Tens1 = V - F G^-1 U
		
			if ((OUTPUT) && (index == REPORT))
				printf("V-FG^-1U \n%1.14E %1.14E %1.14E \n%1.14E %1.14E %1.14E \n%1.14E %1.14E %1.14E \n\n",
					Tens1.xx,Tens1.xy,Tens1.xz,
					Tens1.yx,Tens1.yy,Tens1.yz,
					Tens1.zx,Tens1.zy,Tens1.zz);
			
			// Now calculate v_e:
			// Two cases: on 1st pass we should
			// -- insert the 
			
			Tens1.Inverse(Tens2);
			v_e_plus = Tens2*vec_e;		// Here is v_e_k+1
			
			// DEBUG:
		//	f64_vec3 vec_e0 = vec_e;
		//	vec_e0.z += h*eoverm*EzShape*EzTuning;
		//	f64_vec3 v_e_0 = Tens2*vec_e0;
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
			if ((OUTPUT) && (index == REPORT)) {
				printf("(V-FG^-1U)^-1 \n %1.14E %1.14E %1.14E ) %1.14E = %1.14E\n%1.14E %1.14E %1.14E ) %1.14E = %1.14E \n%1.14E %1.14E %1.14E ) %1.14E = %1.14E \n\n",
					Tens2.xx,Tens2.xy,Tens2.xz,vec_e.x,v_e_plus.x,
					Tens2.yx,Tens2.yy,Tens2.yz,vec_e.y,v_e_plus.y,
					Tens2.zx,Tens2.zy,Tens2.zz,vec_e.z,v_e_plus.z);
				printf("\n");
				// Test relationship:
				printf("h %1.14E (1-hh) %1.14E (1+..)/(1+..) %1.14E\n",
					h, (1.0-h*h*e*eoverm*M_PI*nT_elec_src.n),
					
					(1.0-h*h*e*eoverm*M_PI*nT_elec_src.n)/(1.0+h*h*e*eoverm*M_PI*n_e_plus));
				printf("vek.z %1.14E rat*vekz %1.14E vez_k+1 %1.14E \nek ne+ %1.14E %1.14E\n************\n",
					v_e_k.z,
				v_e_k.z*(1.0-h*h*e*eoverm*M_PI*nT_elec_src.n)/(1.0+h*h*e*eoverm*M_PI*n_e_plus),
				v_e_plus.z,
				nT_elec_src.n,n_e_plus);
			};
			
			// Effect of EzTuning:
			
			// Now we come to part of the routine where we have to record the effects of scaling the external Ez field.
			// We want to record an aggregated total: Sum of z current = Iz0 + sigma_zz EzTuning.
			{
				Vector3 ve_plus_of_EzTuning = Tens2*vec_e_effect_of_EzTuning;
				// Sadly there appears not to be a way round storing this.

				// Performance: Some changes resulted in lower stack frame, higher loads+stores. 
				//              We should realise that way outside L1, this is a worsening. NVM.

				// Can we preserve U = Tens3?
				// Now recreate G which we overwrote:
				
				Tens1.xx = 1.0 
						 // + no contrib from omega_ci x
						   + (h*0.5*m_n/(m_ion+m_n))*nu_ni_MT_over_n*nT_neut_use.n
											*(1.0-Beta_ni)
						   + h*0.5*moverM*nu_ieBar
						   + h*h*e*e* M_PI* n_ion_plus / m_ion
							;
				Tens1.yy = Tens1.xx;
				Tens1.zz = Tens1.xx;
				Tens1.xx -= fac*//Upsilon_eHeart.xx/total; 
					// division by "total = nu*nu+omega*omega" is in fac.
								(nu_eHeart*nu_eHeart + omega_ce.x*omega_ce.x);
				Tens1.yy -= fac*(nu_eHeart*nu_eHeart + omega_ce.y*omega_ce.y);
				Tens1.zz -= fac*(nu_eHeart*nu_eHeart + omega_ce.z*omega_ce.z);
				
				Tens1.xy = -h*0.5*moverM*omega_ce.z
						 - fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
				Tens1.xz = h*0.5*moverM*omega_ce.y
						 - fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
				Tens1.yx = h*0.5*moverM*omega_ce.z
						 - fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
				Tens1.yz = -h*0.5*moverM*omega_ce.x
						 - fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
				Tens1.zx = -h*0.5*moverM*omega_ce.y
						 - fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
				Tens1.zy = h*0.5*moverM*omega_ce.x
						 - fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
				// ... replace omega_ci = omega_ce*moverM ... 
				
				// Formula for Upsilon_eHeart comes from Krook model subsection in model document.
				// We will prefer not to create omega_ci vector of course!!!
				
				Tens1.Inverse(Tens2); // Tens2 now = G^-1
				v_ion_plus = Tens2*(vec_i - Tens3*v_e_plus);  // Here is v_i_k+1
				
				Iz[threadIdx.x] = q*area*(v_ion_plus.z*n_ion_plus - v_e_plus.z*n_e_plus);
				{ 
					real viz_plus_of_EzTuning;
					{
						Vector3 temp = Tens3*ve_plus_of_EzTuning;
						viz_plus_of_EzTuning = Tens2.zz*h*qoverM*EzShape
										 - Tens2.zx*temp.x
										 - Tens2.zy*temp.y
										 - Tens2.zz*temp.z;
					}
					sigma_zz[threadIdx.x] = q*area*(viz_plus_of_EzTuning*n_ion_plus - ve_plus_of_EzTuning.z*n_e_plus);

					if ((OUTPUT) && (index == REPORT)) {
						printf("sigma_zz[threadIdx.x]: %1.14E \n"
							"viz_plus_of_EzTuning*n_ion_plus %1.14E\n"
							"ve_plus_of_EzTuning.z*n_e_plus %1.14E\n"
							"vec_e_effec %1.14E %1.14E %1.14E\n",
							sigma_zz[threadIdx.x],
							viz_plus_of_EzTuning*n_ion_plus,
							ve_plus_of_EzTuning.z*n_e_plus,
							vec_e_effect_of_EzTuning);
					};

					//Bug: sigma_zz calc appeared in wrong place, above, and gave wrong ion values.

				}
				if ((OUTPUT) && (index == REPORT)) {
					printf("Iz: %1.14E ion %1.14E e %1.14E\n"
						   "old: %1.14E ion %1.14E e %1.14E\n------\n",
						   Iz[threadIdx.x], q*area*v_ion_plus.z*n_ion_plus,-q*area*v_e_plus.z*n_e_plus,
						   q*area*(v_ion_k.z*nT_ion_src.n-v_e_k.z*nT_elec_src.n),
									q*area*v_ion_k.z*nT_ion_src.n,-q*area*v_e_k.z*nT_elec_src.n);
				}
				
				if ((OUTPUT) && (index == REPORT)) 
					//(Iz[threadIdx.x] > 1.0e6) || 
				 //  ( (Iz[threadIdx.x] != Iz[threadIdx.x])
				//		&& (threadIdx.x == 32))
				{ // && (index < BEGINNING_OF_CENTRAL)){
					printf("!__ %d %d | Iz %1.14E sig %1.14E ne %1.14E vez %1.14E r %1.14E\n",
							index, per_info.flag,
							//q*area*(v_ion_plus.z*n_ion_plus - v_e_0.z*n_e_plus),
							Iz[threadIdx.x],
							sigma_zz[threadIdx.x],
							n_e_plus, v_e_plus.z,
							centroid.modulus());
				};				
			} // ve_plus_of_EzTuning goes out of scope
			
			v_n_plus = v_n_0 + Beta_ne*v_e_plus + Beta_ni*v_ion_plus;  // Here is v_n_k+1
			
			// v_e =  (V-F G^-1 U) ^-1 ( vec_e_0 )
			//		+ EzTuning (V-F G^-1 U) ^-1 ( vec_e_1 )
			// v_i = G^-1 (d - U ve)
			
			// Now:
			if (b2ndPass) {
				
				p_dAdt_out[index] = dAdt_k + h*c*c*(Lap_A_half + TWO_PI_OVER_C*
					//(J_k+J_k+1)
						q*(nT_ion_src.n*v_ion_k-nT_elec_src.n*v_e_k +
							n_ion_plus*v_ion_plus-n_e_plus*v_e_plus)
													);
				// The Jk comes from what was implied earlier: our n_plus as figured here.
				// . v_e_plus needs here to be the estimate from our "best guess" Ez_ext.
				// . Both J's, k and k+1, need to correspond to the evolution of rho.
			};
			// Lap_A_half is the only variable that is only in scope in this bracket.
			// We really should try putting writes outside braces.
		}
		
		if (b2ndPass == 0) {
			// WE NO LONGER WANT TO DO THIS: No save-off of n,T on minor cells.

//			nT_neut_use.n = (nT_neut_src.n+nT_neut_use.n)*0.5;
//			nT_neut_use.T = (nT_neut_src.T+nT_neut_use.T)*0.5;
//			nT_ion_use.n = (nT_ion_src.n+nT_ion_use.n)*0.5;
//			nT_ion_use.T = (nT_ion_src.T+nT_ion_use.T)*0.5;
//			nT_elec_use.n = (nT_elec_src.n+nT_elec_use.n)*0.5;
//			nT_elec_use.T = (nT_elec_src.T+nT_elec_use.T)*0.5;

			v_n_plus = 0.5*(v_n_plus+v_n_k);
			v_ion_plus = 0.5*(v_ion_plus+v_ion_k);
			v_e_plus = 0.5*(v_e_plus+v_e_k);
			// Tween back to output half-time system
		} 
		
//		p_nT_neut_out[index] = nT_neut_use;
//		p_nT_ion_out[index] = nT_ion_use;
//		p_nT_elec_out[index] = nT_elec_use;
		// Save them off in the heating routine that takes place on majors, not here.

		p_v_neut_out[index] = v_n_plus;
		p_v_ion_out[index] = v_ion_plus;
		p_v_elec_out[index] = v_e_plus;
		// On 1st pass we use this v to calculate viscosity.
		
		// Time to sort out heating contribution: (ie frictional or "resistive" heating)

		f64 NnTn_addition, NiTi_addition, NeTe_addition;
		// Inelastic friction heating:
		NiTi_addition = area* THIRD*m_ion*n_ionrec.n_ionise*((v_ion_k-v_n_k).dot(v_ion_k-v_n_k));
		NnTn_addition = area* THIRD*m_ion*n_ionrec.n_recombine*((v_ion_k-v_n_k).dot(v_ion_k-v_n_k));
		NeTe_addition = area* THIRD*m_e*(n_ionrec.n_ionise + n_ionrec.n_recombine)*((v_e_k-v_n_k).dot(v_e_k-v_n_k));
	
		{
			f64 total = 
				(nu_eHeart*nu_eHeart + omega_ce.x*omega_ce.x+omega_ce.y*omega_ce.y+omega_ce.z*omega_ce.z);
			Tensor3 Tens1;
			Tens1.xx = h*nu_eiBar ;
			Tens1.yy = Tens1.xx;
			Tens1.zz = Tens1.xx;
			f64 fac = -h*0.9*nu_eiBar*nu_eiBar/(nu_eHeart*total);
			Tens1.xx += fac*(omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart);
			Tens1.yy += fac*(omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart);
			Tens1.zz += fac*(omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart);			
			Tens1.xy = fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
			Tens1.xz = fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
			Tens1.yx = fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
			Tens1.yz = fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
			Tens1.zx = fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
			Tens1.zy = fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
			
			// This was e-i resistive heating:
			NeTe_addition += 
				area* SIXTH*n_e_plus*TWOTHIRDS*m_e*(
				// rate of change of ve. dot(ve-vi), integrated:
							(Tens1*(v_e_k-v_ion_k)).dot(v_e_k-v_ion_k)
						+ 
							(Tens1*(v_e_k-v_ion_k+v_e_plus-v_ion_plus)).dot
								(v_e_k-v_ion_k+v_e_plus-v_ion_plus) // 0.25 cancels with 4
						+   (Tens1*(v_e_plus-v_ion_plus)).dot(v_e_plus-v_ion_plus)
						);
		}
		{
			// Inelastic frictional heating:
			
			// Maybe this is actually FRICTIONAL heating e-n, i-n ;
			// I think that's what we're actually looking at here.

			f64 M_in = m_n*m_ion/((m_n+m_ion)*(m_n+m_ion));
//			f64 M_en = m_n*m_e/((m_n+m_e)*(m_n+m_e));
//			f64 M_ie = m_ion*m_e/((m_ion+m_e)*(m_ion+m_e));
			
			NeTe_addition += area * SIXTH*n_e_plus*TWOTHIRDS*m_e*(
				h*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n*(
					  (v_e_k-v_n_k).dot(v_e_k-v_n_k)
					+ (v_e_k-v_n_k + v_e_plus - v_n_plus).dot(v_e_k-v_n_k + v_e_plus - v_n_plus)
					+ (v_e_plus-v_n_plus).dot(v_e_plus-v_n_plus)
												));
			
			f64 v_ni_diff_sq = SIXTH*((v_n_k-v_ion_k).dot(v_n_k-v_ion_k)
					+ (v_n_k-v_ion_k+v_n_plus-v_ion_plus).dot(v_n_k-v_ion_k+v_n_plus-v_ion_plus)
					+ (v_n_plus-v_ion_plus).dot(v_n_plus-v_ion_plus));
			
			NiTi_addition += area * n_ion_plus*TWOTHIRDS*m_n*
							h*M_in*nu_ni_MT_over_n*nT_neut_use.n*v_ni_diff_sq;
			
			NnTn_addition += area * n_n_plus*TWOTHIRDS*m_ion*
							h*M_in*nu_ni_MT_over_n*nT_ion_use.n*v_ni_diff_sq;
					
			// We can deduce T_k+1 afterwards from n_k+1 T_k+1.
			// OR, we can rearrange conservative equations to be for T_k+1.
		}
		
		// NOTE HERE WE PUT " = "
		// Rather than, adding -- which we might want to do if we put visc+cond+thermoelectric
		// into same slots.
		// This is the addition to NT.
		p_resistive_heat_neut[index] = NnTn_addition;
		p_resistive_heat_ion[index] = NiTi_addition;
		p_resistive_heat_elec[index] = NeTe_addition;

	} else {  // NOT (info.flag == DOMAIN_VERTEX) ...

		p_resistive_heat_neut[index] = 0.0;
		p_resistive_heat_ion[index] = 0.0;
		p_resistive_heat_elec[index] = 0.0; // Or save some writes by doing cudaMemset beforehand.

		if (per_info.flag == OUTERMOST) {
//			p_nT_neut_out[index] = nT_neut_src;
//			p_nT_ion_out[index] = nT_ion_src;
//			p_nT_elec_out[index] = nT_elec_src;
			p_v_neut_out[index] = v_n_k;
			p_v_ion_out[index] = v_ion_k;
			p_v_elec_out[index] = v_e_k;
			// Populate with something to avoid mishaps.
		}

		Vector3 dAdt_k,four_pi_over_c_J;
	//	Lap_A_half = p_Lap_A_half[index];
		dAdt_k = p_Adot_k[index];
		
		// ReverseJ calc: this is for the reverse z current at the edge of the anode:
		
		four_pi_over_c_J.x = 0.0;
		four_pi_over_c_J.y = 0.0;
		four_pi_over_c_J.z = 0.0;
		
		if (per_info.flag == REVERSE_JZ_TRI) {
			// . Need to go through program and identify if there
			// are times we test for OUT_OF_DOMAIN - think none.
			
			four_pi_over_c_J.z = four_pi_over_c_ReverseJz;
			
		};
		
	//	if ((index >= ReverseJzIndexStart) && (index < ReverseJzIndexEnd))
		{
			// This will no longer work -- we resequenced so they are out of order.
		} 
		
		Vector3 Adot_plus = dAdt_k + h*c*c*(Lap_A_half + four_pi_over_c_J);
		
	//	if ((OUTPUT) && (index == REPORT))
	//		printf("Adot %1.5E Lap_A_half %1.5E 4pi/cJ %1.5E Adot+ %1.5E\n",
	//			dAdt_k.z, Lap_A_half.z, four_pi_over_c_J.z, Adot_plus.z);

		p_dAdt_out[index] = Adot_plus;
		sigma_zz[threadIdx.x] = 0.0;
		Iz[threadIdx.x] = 0.0;
	};

	//} else { // index < Nverts
	//	sigma_zz[threadIdx.x] = 0.0;
	//	Iz[threadIdx.x] = 0.0;
	//};
	// Aggregate:
	__syncthreads();

	int s = blockDim.x;
	int k = s/2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + k];
			Iz[threadIdx.x] += Iz[threadIdx.x + k];
		};
		__syncthreads();
		
		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k-1)){
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x+s-1];
			Iz[threadIdx.x] += Iz[threadIdx.x+s-1];			
		}; 
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s/2;
		__syncthreads();
	};
		
	if (threadIdx.x == 0)
	{
		p_sigma_zz[blockIdx.x] = sigma_zz[0];
		p_Iz[blockIdx.x] = Iz[0];		
	}
}
__global__ void Kernel_Heating_routine(
		f64 const h,
		structural * __restrict__ p_info,
		long * __restrict__ p_IndexTri,

		nT * __restrict__ p_nT_neut_src, // major
		nT * __restrict__ p_nT_ion_src,
		nT * __restrict__ p_nT_elec_src,
		nn * __restrict__ p_nn_ionrec,    // major
		// If we want "use" then it comes in as the output variable.

		f64_vec3 * __restrict__ p_B_major,  // major

	//	f64 * __restrict__ p_visccond_heatrate_neut,
	//	f64 * __restrict__ p_visccond_heatrate_ion,
	//	f64 * __restrict__ p_visccond_heatrate_elec, 
		// We could get rid and use the central slots from the resistive heating.
	
		// Defined on minor:
		f64 * __restrict__ p_resistive_heat_neut, // minor
		f64 * __restrict__ p_resistive_heat_ion,
		f64 * __restrict__ p_resistive_heat_elec,  // to include inelastic frictional effects.
		// What about ion-neutral frictional heating? Where was that included??
	
		f64 * __restrict__ p_area_cell, // major

		nT * __restrict__ p_nT_neut_out, // major
		nT * __restrict__ p_nT_ion_out,
		nT * __restrict__ p_nT_elec_out,
		bool b2ndPass // on '2ndpass', load nT_neut_use. 
								)
{
	// Temperature advance:
	__shared__ f64 resistive_neut[SIZE_OF_TRI_TILE_FOR_MAJOR];
	__shared__ f64 resistive_ion[SIZE_OF_TRI_TILE_FOR_MAJOR];
	__shared__ f64 resistive_elec[SIZE_OF_TRI_TILE_FOR_MAJOR]; // 6 doubles equiv
	
	__shared__ long indextri[MAXNEIGH_d*threadsPerTileMajor]; // 6 doubles equiv
	
	resistive_neut[threadIdx.x]
					= p_resistive_heat_neut[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	resistive_neut[threadIdx.x + threadsPerTileMajor] 
					= p_resistive_heat_neut[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadsPerTileMajor + threadIdx.x];
	
	resistive_ion[threadIdx.x]
					= p_resistive_heat_ion[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	resistive_ion[threadIdx.x + threadsPerTileMajor] 
					= p_resistive_heat_ion[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadsPerTileMajor + threadIdx.x];
	
	resistive_elec[threadIdx.x]
					= p_resistive_heat_elec[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadIdx.x];
	resistive_elec[threadIdx.x + threadsPerTileMajor] 
					= p_resistive_heat_elec[SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x + threadsPerTileMajor + threadIdx.x];
	
	__syncthreads();
	

	f64 niTi, nnTn, neTe;				
	nT  nT_ion_src, nT_elec_src, nT_neut_src,
		nT_neut_use, nT_ion_use, nT_elec_use;
	f64 n_e_plus, n_ion_plus, n_n_plus, area;
	
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	long StartTri = SIZE_OF_TRI_TILE_FOR_MAJOR*blockIdx.x;
	
	memcpy(indextri + MAXNEIGH_d*threadIdx.x, 
		   p_IndexTri + index*MAXNEIGH_d, sizeof(long)*MAXNEIGH_d);
	
	// Do we also want to gather v ? No, we can use from centrals.
	// Remember to collect resistive heat from centrals as well.
	
	
	// n_ionrec is defined on all minors but we use the central==vertex
	// value to actually evolve n --- that is what I assume here.
	
	area = p_area_cell[index];
	structural info = p_info[index];

	nT_neut_src = p_nT_neut_src[index];
	nT_ion_src = p_nT_ion_src[index];
	nT_elec_src = p_nT_elec_src[index];	

	if (info.flag == DOMAIN_VERTEX) {
		
		if (b2ndPass) {
			nT_neut_use = p_nT_neut_out[index];
			nT_ion_use = p_nT_ion_out[index];
			nT_elec_use = p_nT_elec_out[index];
		} else {
			nT_neut_use = nT_neut_src;
			nT_ion_use = nT_ion_src;
			nT_elec_use = nT_elec_src;
		}	
		

		nn n_ionrec = p_nn_ionrec[index];  
		n_n_plus = nT_neut_src.n + n_ionrec.n_recombine-n_ionrec.n_ionise;
		n_ion_plus = nT_ion_src.n + n_ionrec.n_ionise-n_ionrec.n_recombine;
		n_e_plus = nT_elec_src.n + n_ionrec.n_ionise-n_ionrec.n_recombine;


		// worked with more commented.

		niTi = (nT_ion_src.n-n_ionrec.n_recombine)*nT_ion_src.T 
			  + 0.5*n_ionrec.n_ionise*nT_neut_src.T;
		
		nnTn = (nT_neut_src.n-n_ionrec.n_ionise)*nT_neut_src.T
			  + n_ionrec.n_recombine*(nT_elec_src.T+nT_ion_src.T )
			  + n_ionrec.n_recombine*TWOTHIRDS*13.6*kB;
		
		neTe = (nT_elec_src.n-n_ionrec.n_recombine)*nT_elec_src.T
			  + 0.5*n_ionrec.n_ionise*nT_neut_src.T
			  - n_ionrec.n_ionise*TWOTHIRDS*13.6*kB;	
		
		if ((OUTPUT) && (index == REPORT)){
				printf(
					"Tsrc  %1.5E %1.5E %1.5E \n"
					"nT ionise %1.5E %1.5E %1.5E \n",
					nT_neut_src.T,nT_ion_src.T,nT_elec_src.T,
					nnTn,niTi,neTe);
			};
		// This will serve as part of the right hand side for including heat transfers.
		
		// Visc+cond heat addition:
		// ------------------------
		// DKE = 1/2 m n v.v 
		// We should associate a heating amount with each wall that will be positive.
		// ( That works out nicely for offset! )
		// That means we need to do a fetch. We can't work out visc htg without knowing
		// neighbour v, which means we might as well store it - correct?
		// If we are adding to v then we are increasing or decreasing DKE here -- but 
		// then we want net + heating appearing in this and the neighbour.
		// So that leaves us having to do a fetch always.
		
		// & Include heat conduction heat addition in same step.
		// ------------------------
		{
			// CAREFUL ABOUT WHETHER THESE WERE CREATED DIVIDING BY AREA.
			
			// Either we will reinstate these here, or, 
			// we will proceed by putting the necessary heat into
			// what is now the "resistive" variable, major/central part.

	//		nnTn += h*p_visccond_heatrate_neut[index];
	//		niTi += h*p_visccond_heatrate_ion[index];
	//		neTe += h*p_visccond_heatrate_elec[index];
		}
		
		// Now drag in the resistive heat rates INCLUDING its own central.
		{
			f64 neut_resistive = 0.0, ion_resistive = 0.0, elec_resistive = 0.0;
			long iTri;
			for (iTri = 0; iTri < info.neigh_len; iTri++)
			{
				// CAREFUL of cases where we are at the edge.
				long index_tri = indextri[threadIdx.x*MAXNEIGH_d + iTri];
				if ((index_tri >= StartTri) && (index_tri < StartTri + SIZE_OF_TRI_TILE_FOR_MAJOR))
				{
					neut_resistive += resistive_neut[index_tri-StartTri];
					ion_resistive += resistive_ion[index_tri-StartTri];
					elec_resistive += resistive_elec[index_tri-StartTri];
				} else {
					neut_resistive += p_resistive_heat_neut[index_tri];
					ion_resistive += p_resistive_heat_ion[index_tri];
					elec_resistive += p_resistive_heat_elec[index_tri];
				};
			}
			neut_resistive *= THIRD;
			ion_resistive *= THIRD;
			elec_resistive *= THIRD;
			// Try __syncthreads here...
			// Add the values for central cell:
			neut_resistive += p_resistive_heat_neut[BEGINNING_OF_CENTRAL + index];
			ion_resistive += p_resistive_heat_ion[BEGINNING_OF_CENTRAL + index];
			elec_resistive += p_resistive_heat_elec[BEGINNING_OF_CENTRAL + index];
			nnTn += neut_resistive/area;
			niTi += ion_resistive/area;
			neTe += elec_resistive/area; // These were the additions to NT
		}
		
		// NO CRASH IF PUT IT HERE.


		// So we have now to collect things like:
		// nu_eHeart, nu_eiBar :
		f64 nu_ne_MT_over_n, nu_ni_MT_over_n, nu_eiBar, nu_ieBar, nu_eHeart; // 5 double
		Vector3 omega_ce = eovermc*p_B_major[index];
		{
			f64 sqrt_Te = sqrt(nT_elec_use.T);
			f64 s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(nT_elec_use.T*one_over_kB);
			f64 electron_thermal = sqrt_Te*over_sqrt_m_e;
			f64 ionneut_thermal = sqrt(nT_ion_use.T/m_ion+nT_neut_use.T/m_n); // hopefully not sqrt(0)
			f64 lnLambda = Get_lnLambda_d(nT_ion_use.n,nT_elec_use.T);
			f64 s_in_MT = Estimate_Neutral_MT_Cross_section(nT_ion_use.T*one_over_kB);
			f64 s_en_MT = Estimate_Neutral_MT_Cross_section(nT_elec_use.T*one_over_kB);
			nu_ne_MT_over_n = s_en_MT*electron_thermal; // have to multiply by n_e for nu_ne_MT
			nu_ni_MT_over_n = s_in_MT*ionneut_thermal;
			nu_eiBar = nu_eiBarconst*kB_to_3halves*nT_ion_use.n*lnLambda/(nT_elec_use.T*sqrt_Te);
			nu_ieBar = nT_elec_use.n*nu_eiBar/nT_ion_use.n;
			nu_eHeart = 1.87*nu_eiBar + 
							nT_neut_use.n*s_en_visc*electron_thermal;
		}
		
		// From here on doing the inter-species heat exchange:
		Tensor3 Tens1; 	
		{
			f64 M_in = m_n*m_ion/((m_n+m_ion)*(m_n+m_ion));
			f64 M_en = m_n*m_e/((m_n+m_e)*(m_n+m_e));
			f64 M_ie = m_ion*m_e/((m_ion+m_e)*(m_ion+m_e));
			
			// See section 10.3.1, June 2016 doc.
			// Seems good idea to do this in heat, or manipulate equivalently.
			
			// d/dt(NT) = U NT
			// Add to the RH vector, h*0.5*U*NT_k:
			Tens1.xx = -2.0*(M_in*nu_ni_MT_over_n*nT_ion_use.n + M_en*nu_ne_MT_over_n*nT_elec_use.n);
			Tens1.xy = 2.0*M_in*nu_ni_MT_over_n*nT_neut_use.n;
			Tens1.xz = 2.0*M_en*nu_ne_MT_over_n*nT_neut_use.n;
			
			Tens1.yx = 2.0*M_in*nu_ni_MT_over_n*nT_ion_use.n;
			Tens1.yy = -2.0*(M_in*nu_ni_MT_over_n*nT_neut_use.n
						   + M_ie*nu_ieBar);
			Tens1.yz = 2.0*M_ie*nu_eiBar;
			
			Tens1.zx = 2.0*M_en*nu_ne_MT_over_n*nT_elec_use.n;
			Tens1.zy = 2.0*M_ie*nu_ieBar;
			Tens1.zz = -2.0*(M_ie*nu_eiBar + M_en*nu_ne_MT_over_n*nT_neut_use.n);
		}
		
		// Midpoint: 
		// d/dt (nT) = U
		// (nT)_k+1 = (1 - h/2 U)^-1 (1+h/2 U) (nT)_k
		
		nnTn += h*0.5*(Tens1.xx*(nT_neut_src.n*nT_neut_src.T)
					   + Tens1.xy*(nT_ion_src.n*nT_ion_src.T)
					   + Tens1.xz*(nT_elec_src.n*nT_elec_src.T)
					   );
		niTi += h*0.5*(Tens1.yx*(nT_neut_src.n*nT_neut_src.T)
					   + Tens1.yy*(nT_ion_src.n*nT_ion_src.T)
					   + Tens1.yz*(nT_elec_src.n*nT_elec_src.T)
					   );
		neTe += h*0.5*(Tens1.zx*(nT_neut_src.n*nT_neut_src.T)
					   + Tens1.zy*(nT_ion_src.n*nT_ion_src.T)
					   + Tens1.zz*(nT_elec_src.n*nT_elec_src.T)
					   );
		
		// Matrix is 1 - h*0.5*U
		
		Tens1.xx = 1.0-h*0.5*Tens1.xx;
		Tens1.xy = -h*0.5*Tens1.xy;
		Tens1.xz = -h*0.5*Tens1.xz;
		
		Tens1.yx = -h*0.5*Tens1.yx;
		Tens1.yy = 1.0-h*0.5*Tens1.yy;
		Tens1.yz = -h*0.5*Tens1.yz;
		
		Tens1.zx = -h*0.5*Tens1.zx;
		Tens1.zy = -h*0.5*Tens1.zy;
		Tens1.zz = 1.0-h*0.5*Tens1.zz;
		
		
		if ((OUTPUT) && (index == REPORT)) {
				printf("nT_before %1.5E %1.5E %1.5E \n",
					nnTn,niTi,neTe);
			};
		{
			Tensor3 Tens2;
			Tens1.Inverse(Tens2);
			Vector3 RH,LH;
			RH.x = nnTn;
			RH.y = niTi;
			RH.z = neTe;
			LH = Tens2*RH;
			nnTn = LH.x;
			niTi = LH.y;
			neTe = LH.z;				
		}
		if ((OUTPUT) && (index == REPORT)) {
				printf("nT_after %1.5E %1.5E %1.5E \n",
					nnTn,niTi,neTe);
		};
		// END OF COMMENT


		// Overwrite any old rubbish in memory so that we can save off the output:
		nT_neut_use.n = n_n_plus;
		nT_neut_use.T = nnTn/n_n_plus;
		nT_ion_use.n = n_ion_plus;
		nT_ion_use.T = niTi/n_ion_plus;
		nT_elec_use.n = n_e_plus;
		nT_elec_use.T = neTe/n_e_plus;	
		if (b2ndPass == false) {
			// Tween back to halfway if this is the first pass:
			nT_neut_use.n = 0.5*(nT_neut_src.n + nT_neut_use.n);
			nT_ion_use.n = 0.5*(nT_ion_src.n + nT_ion_use.n);
			nT_elec_use.n = 0.5*(nT_elec_src.n + nT_elec_use.n);
			nT_neut_use.T = 0.5*(nT_neut_src.T + nT_neut_use.T);
			nT_ion_use.T = 0.5*(nT_ion_src.T + nT_ion_use.T);
			nT_elec_use.T = 0.5*(nT_elec_src.T + nT_elec_use.T);
		};
		
		//if ((OUTPUT) && (index == REPORT))
		//	printf("Te %1.5E \n################\n",nT_elec_use.T);
		// */ this was the end of commenting
	} else {
		
		// Not DOMAIN_VERTEX :

		nT_neut_use = nT_neut_src;
		nT_ion_use = nT_ion_src;
		nT_elec_use = nT_elec_src;
	};

	p_nT_neut_out[index] = nT_neut_use;
	p_nT_ion_out[index] = nT_ion_use;
	p_nT_elec_out[index] = nT_elec_use;
}



