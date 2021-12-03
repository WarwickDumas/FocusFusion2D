 
#include "cuda_struct.h"
#pragma once
#ifndef CUSYSTCU
#define CUSYSTCU

extern real evaltime;
extern long GlobalStepsCounter;
extern bool GlobalSuppressSuccessVerbosity;
extern f64 * p_graphdata1_host, *p_graphdata2_host, *p_graphdata3_host, *p_graphdata4_host, *p_graphdata5_host, *p_graphdata6_host;
extern f64 * p_Tgraph_host[9];
extern f64 * p_accelgraph_host[12];
extern f64 * p_Ohmsgraph_host[20];
extern f64 * p_arelz_graph_host[12];
extern f64 * p_temphost5;

__host__ bool Call(cudaError_t cudaStatus, char str[])
{
	if (cudaStatus == cudaSuccess) {
		if (strncmp(str,"cudaMemcpy",8) != 0)
			if (GlobalSuppressSuccessVerbosity == false) printf("\tSuccess: %s ||| \n",str);
		return false;
	} else {
		printf("Error: %s\nReturned %d : %s\n",
			str, cudaStatus, cudaGetErrorString(cudaStatus));
		printf("press o\n");
		while (getch() != 'o');
		PerformCUDA_Revoke();
		exit(2030);
	};
	return true;
} 
   

cuSyst::cuSyst(){
	bInvoked = false;
	bInvokedHost = false;
}

int cuSyst::Invoke()
{
	 Nverts = NUMVERTICES;
	 Ntris = NUMTRIANGLES; // FFxtubes.h
	 Nminor = Nverts + Ntris;

	if (bInvoked == false) {

		if (
			   (!CallMAC(cudaMalloc((void**)&p_info, Nminor * sizeof(structural))))

			&& (!CallMAC(cudaMalloc((void**)&p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_izNeigh_TriMinor, Ntris*6 * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBC_triminor, Ntris * 6 * sizeof(char))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_corner_index, Ntris * sizeof(LONG3))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_neigh_index, Ntris * sizeof(LONG3))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4))))
			&& (!CallMAC(cudaMalloc((void**)&p_who_am_I_to_corner, Ntris * sizeof(LONG3))))

			&& (!CallMAC(cudaMalloc((void**)&p_n_major, Nverts * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_n_minor, Nminor * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_T_minor, Nminor * sizeof(T3))))

			&& (!CallMAC(cudaMalloc((void**)&p_AAdot, Nminor * sizeof(AAdot))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_v_n, Nminor * sizeof(f64_vec3))))
			&& (!CallMAC(cudaMalloc((void**)&p_vie, Nminor * sizeof(v4))))
			&& (!CallMAC(cudaMalloc((void**)&p_B, Nminor * sizeof(f64_vec3))))

			&& (!CallMAC(cudaMalloc((void**)&p_Lap_Az, Nminor * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_v_overall_minor, Nminor * sizeof(f64_vec2))))
			&& (!CallMAC(cudaMalloc((void**)&p_n_upwind_minor, Nminor * sizeof(nvals))))
						
			&& (!CallMAC(cudaMalloc((void**)&p_AreaMinor, Nminor * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_AreaMajor, Nverts * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_cc, Nminor*sizeof(f64_vec2))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_iVolley, Nverts * sizeof(char))))

			)
		{
			bInvoked = true;
			//Zero();
			printf("Dimensioned for MAXNEIGH_d = %d\n", MAXNEIGH_d);
			return 0;
		}
		else {
			printf("There was an error in dimensioning Systdata object.\n");
			getch();	getch();
			return 1;
		};
	}
	else {
		if (Nverts != NUMVERTICES) { printf("cuSyst Error - Nverts %d != N %d\n", Nverts, NUMVERTICES); getch(); }
		return 2;
	};
}
int cuSyst::InvokeHost()
{
	Nverts = NUMVERTICES;
	Ntris = NUMTRIANGLES;
	Nminor = Nverts + Ntris;
	p_info = ( structural * )malloc(Nminor* sizeof(structural));
		
	p_izTri_vert = ( long *)malloc(Nverts*MAXNEIGH_d * sizeof(long));
	p_izNeigh_vert = (long * )malloc(Nverts*MAXNEIGH_d * sizeof(long));
	p_szPBCtri_vert = (char * )malloc(Nverts*MAXNEIGH_d * sizeof(char));
	p_szPBCneigh_vert = (char *)malloc(Nverts*MAXNEIGH_d * sizeof(char));

	p_izNeigh_TriMinor = (long * )malloc(Ntris * 6 * sizeof(long));
	p_szPBC_triminor = (char * )malloc(Ntris * 6 * sizeof(char));
	p_tri_corner_index = ( LONG3 *)malloc(Ntris * sizeof(LONG3));
	p_tri_periodic_corner_flags = (CHAR4 *)malloc(Ntris * sizeof(CHAR4));
	p_tri_neigh_index = (LONG3 *)malloc(Ntris * sizeof(LONG3));
	p_tri_periodic_neigh_flags = (CHAR4 *)malloc(Ntris * sizeof(CHAR4));
	p_who_am_I_to_corner = (LONG3 * )malloc(Ntris * sizeof(LONG3));

	p_n_major = (nvals * )malloc(Nverts * sizeof(nvals));
	p_n_minor = (nvals * )malloc(Nminor * sizeof(nvals));
	p_T_minor = (T3 * )malloc(Nminor * sizeof(T3));

	p_AAdot = ( AAdot *)malloc(Nminor * sizeof(AAdot));

	p_v_n = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));
	p_vie = (v4 * )malloc(Nminor * sizeof(v4));
	p_B = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));

	p_Lap_Az = (f64 * )malloc(Nminor * sizeof(f64));
	p_v_overall_minor = (f64_vec2 *)malloc(Nminor * sizeof(f64_vec2));
	p_n_upwind_minor = (nvals *)malloc(Nminor * sizeof(nvals));

	p_AreaMinor = (f64 * )malloc(Nminor * sizeof(f64));
	p_AreaMajor = (f64 * )malloc(Nverts * sizeof(f64));

	p_cc = (f64_vec2 *)malloc(Nminor * sizeof(f64));
	
	p_iVolley = (char *)malloc(Nverts * sizeof(char));

	
	if (p_cc == 0) {
		printf("failed to invokeHost the cusyst.\n");
		getch();
		return 1;
	}
	else {
		bInvokedHost = true;
		return 0;
	};
}
cuSyst::~cuSyst(){
	if (bInvoked)
	{

		cudaFree(p_info);
		cudaFree(p_izTri_vert);
		cudaFree(p_izNeigh_vert);
		cudaFree(p_szPBCtri_vert);
		cudaFree(p_szPBCneigh_vert);
		cudaFree(p_izNeigh_TriMinor);
		cudaFree(p_szPBC_triminor);
		cudaFree(p_tri_corner_index);
		cudaFree(p_tri_periodic_corner_flags);
		cudaFree(p_tri_neigh_index);
		cudaFree(p_tri_periodic_neigh_flags);
		cudaFree(p_who_am_I_to_corner);
		cudaFree(p_n_major);
		cudaFree(p_n_minor);
		cudaFree(p_n_upwind_minor);
		cudaFree(p_T_minor);
		cudaFree(p_AAdot);
		cudaFree(p_v_n);
		cudaFree(p_vie);
		cudaFree(p_B);
		cudaFree(p_Lap_Az);
		cudaFree(p_v_overall_minor);
		cudaFree(p_AreaMinor);
		cudaFree(p_AreaMajor);
		cudaFree(p_cc);
		cudaFree(p_iVolley);

	}
	if (bInvokedHost) {

free(p_info);
free(p_izTri_vert);
free(p_izNeigh_vert);
free(p_szPBCtri_vert);
free(p_szPBCneigh_vert);
free(p_izNeigh_TriMinor);
free(p_szPBC_triminor);
free(p_tri_corner_index);
free(p_tri_periodic_corner_flags);
free(p_tri_neigh_index);
free(p_tri_periodic_neigh_flags);
free(p_who_am_I_to_corner);
free(p_n_major);
free(p_n_minor);
free(p_n_upwind_minor);
free(p_T_minor);
free(p_AAdot);
free(p_v_n);
free(p_vie);
free(p_B);
free(p_Lap_Az);
free(p_v_overall_minor);
free(p_AreaMinor);
free(p_AreaMajor);
free(p_cc);
free(p_iVolley);
	};
}

void cuSyst::SaveGraphs(const char filename[])
{
	FILE * fp = fopen(filename, "wb");
	if (fp == 0) { printf("open %s failed\n\n", filename); getch();  return; }
	else { printf("opened file %s ..", filename); }

	long filevers = 1;
	fwrite(&filevers, sizeof(long), 1, fp);
	fwrite(&Nverts, sizeof(long), 1, fp);
	fwrite(&Ntris, sizeof(long), 1, fp);

	fwrite(&GlobalStepsCounter, sizeof(long), 1, fp);
	fwrite(&evaltime, sizeof(f64), 1, fp);

	fwrite(p_info, sizeof(structural), NMINOR, fp);

	fwrite(p_izTri_vert, sizeof(long), Nverts*MAXNEIGH_d, fp);

	fwrite(p_izNeigh_vert, sizeof(long), Nverts*MAXNEIGH_d, fp);
	fwrite(p_szPBCtri_vert, sizeof(char), Nverts*MAXNEIGH_d, fp);
	fwrite(p_szPBCneigh_vert, sizeof(char), Nverts*MAXNEIGH_d, fp);

	fwrite(p_izNeigh_TriMinor, sizeof(long), Ntris * 6, fp);
	fwrite(p_szPBC_triminor, sizeof(char), Ntris * 6, fp);
	fwrite(p_tri_corner_index, sizeof(LONG3), Ntris, fp);
	fwrite(p_tri_periodic_corner_flags, sizeof(CHAR4), Ntris, fp);
	fwrite(p_tri_neigh_index, sizeof(LONG3), Ntris, fp);
	fwrite(p_tri_periodic_neigh_flags, sizeof(CHAR4), Ntris, fp);
	fwrite(p_who_am_I_to_corner, sizeof(LONG3), Ntris, fp);

	fwrite(p_iVolley, sizeof(char), Nverts, fp); // Not changed yet in load.

	fwrite(p_n_major, sizeof(nvals), Nverts, fp);
	fwrite(p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3), Nverts, fp);
	fwrite(p_AAdot + BEGINNING_OF_CENTRAL, sizeof(AAdot), Nverts, fp);
	fwrite(p_v_n + BEGINNING_OF_CENTRAL, sizeof(f64_vec3), Nverts, fp);
	fwrite(p_vie + BEGINNING_OF_CENTRAL, sizeof(v4), Nverts, fp);
	fwrite(p_B + BEGINNING_OF_CENTRAL, sizeof(f64_vec3), Nverts, fp);
	fwrite(p_AreaMajor, sizeof(f64), Nverts, fp);
	// Now save the graphing data that we use ...
	fwrite(p_graphdata1_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	fwrite(p_graphdata2_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	fwrite(p_graphdata3_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	fwrite(p_graphdata4_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	fwrite(p_graphdata5_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	fwrite(p_graphdata6_host + BEGINNING_OF_CENTRAL, sizeof(f64),Nverts, fp);
	int i;
	for (i = 0; i < 9; i++)
		fwrite(p_Tgraph_host[i], sizeof(f64),NUMVERTICES, fp);
	for (i = 0; i < 12; i++)
		fwrite(p_accelgraph_host[i], sizeof(f64),NUMVERTICES, fp);
	//for (i = 0; i < 20; i++)
	//	fwrite(p_Ohmsgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64));
	// skip ohmsgraph...
	for (i = 0; i < 12; i++)
		fwrite(p_arelz_graph_host[i], sizeof(f64),NUMVERTICES, fp);

	// We only really wanted to save 1D graphs of these, so maybe we should have stuck to that!

	fwrite(p_temphost5 + BEGINNING_OF_CENTRAL, sizeof(f64),NUMVERTICES, fp); // Lap Az

	// so that makes 33 extra doubles per vertex so far. We have to expect the total size will be the same.


	//	fwrite(p_Lap_Az, Nminor * sizeof(f64));
	//	fwrite(p_v_overall_minor, Nminor * sizeof(f64_vec2));
	//	fwrite(p_n_upwind_minor, Nminor * sizeof(nvals));

	

	fclose(fp);
	printf("File save done.\n");
}
void cuSyst::Save(const char filename[])
{
	FILE * fp = fopen(filename, "wb");
	if (fp == 0) { printf("open %s failed\n\n", filename); getch();  return; }
	else { printf("opened file %s ..", filename); }

	long filevers = 1;
	fwrite(&filevers, sizeof(long),1,fp);
	fwrite(&Nverts, sizeof(long),1,fp);
	fwrite(&Ntris, sizeof(long),1,fp);

	fwrite(&GlobalStepsCounter, sizeof(long), 1, fp);
	fwrite(&evaltime, sizeof(f64), 1, fp);

	fwrite(p_info, sizeof(structural),NMINOR, fp);

	fwrite(p_izTri_vert, sizeof(long), Nverts*MAXNEIGH_d, fp);

	fwrite(p_izNeigh_vert, sizeof(long),Nverts*MAXNEIGH_d, fp);
	fwrite(p_szPBCtri_vert, sizeof(char),Nverts*MAXNEIGH_d, fp);
	fwrite(p_szPBCneigh_vert, sizeof(char),Nverts*MAXNEIGH_d, fp);

	fwrite(p_izNeigh_TriMinor, sizeof(long), Ntris * 6 , fp);
	fwrite(p_szPBC_triminor, sizeof(char), Ntris * 6, fp);
	fwrite(p_tri_corner_index, sizeof(LONG3), Ntris , fp);
	fwrite(p_tri_periodic_corner_flags, sizeof(CHAR4), Ntris, fp);
	fwrite(p_tri_neigh_index, sizeof(LONG3), Ntris , fp);
	fwrite(p_tri_periodic_neigh_flags, sizeof(CHAR4), Ntris , fp);
	fwrite(p_who_am_I_to_corner, sizeof(LONG3), Ntris, fp);

	fwrite(p_iVolley, sizeof(char), Nverts, fp); // Not changed yet in load.

	fwrite(p_n_major, sizeof(nvals), Nverts, fp);
	fwrite(p_n_minor, sizeof(nvals), Nminor,fp);
	fwrite(p_T_minor, sizeof(T3), Nminor,fp);

	fwrite(p_AAdot, sizeof(AAdot), Nminor,fp);

	fwrite(p_v_n, sizeof(f64_vec3), Nminor , fp);
	fwrite(p_vie, sizeof(v4), Nminor , fp);
	fwrite(p_B, sizeof(f64_vec3), Nminor , fp);

//	fwrite(p_Lap_Az, Nminor * sizeof(f64));
//	fwrite(p_v_overall_minor, Nminor * sizeof(f64_vec2));
//	fwrite(p_n_upwind_minor, Nminor * sizeof(nvals));

	fwrite(p_AreaMinor, sizeof(f64), Nminor, fp);
	fwrite(p_AreaMajor, sizeof(f64), Nverts, fp);
	
	fclose(fp);
	printf("File save done.\n");
}

void cuSyst::Load(const char filename[])
{
	FILE * fp = fopen(filename, "rb");
	if (fp == 0) { printf("open %s failed\n\n", filename); getch();  return; } 
	else { printf("opened file %s ..", filename); }
	rewind(fp);
	long Nverttest, Ntritest, filevers;
	fread(&filevers, sizeof(long), 1, fp);
	fread(&Nverttest, sizeof(long), 1, fp);
	fread(&Ntritest, sizeof(long), 1, fp);

	if ( (filevers != 1) || (Nverttest != Nverts) || (Ntritest != Ntris) ) {
		printf("filevers %d Nverts Ntris %d %d File: %d %d \n\n", filevers, Nverts, Ntris, Nverttest, Ntritest);
		return;
	}
	
	fread(&GlobalStepsCounter, sizeof(long), 1, fp);
	fread(&evaltime, sizeof(f64), 1, fp);

	fread(p_info, sizeof(structural), NMINOR, fp);

	fread(p_izTri_vert, sizeof(long), Nverts*MAXNEIGH_d, fp);
	fread(p_izNeigh_vert, sizeof(long), Nverts*MAXNEIGH_d, fp);
	fread(p_szPBCtri_vert, sizeof(char), Nverts*MAXNEIGH_d, fp);
	fread(p_szPBCneigh_vert, sizeof(char), Nverts*MAXNEIGH_d, fp);

	fread(p_izNeigh_TriMinor, sizeof(long), Ntris * 6, fp);
	fread(p_szPBC_triminor, sizeof(char), Ntris * 6, fp);
	fread(p_tri_corner_index, sizeof(LONG3), Ntris, fp);
	fread(p_tri_periodic_corner_flags, sizeof(CHAR4), Ntris, fp);
	fread(p_tri_neigh_index, sizeof(LONG3), Ntris, fp);
	fread(p_tri_periodic_neigh_flags, sizeof(CHAR4), Ntris, fp);
	fread(p_who_am_I_to_corner, sizeof(LONG3), Ntris, fp);

	// p_iVolley
	fread(p_iVolley, sizeof(char), Nverts, fp);

	fread(p_n_major, sizeof(nvals), Nverts, fp);
	fread(p_n_minor, sizeof(nvals), NMINOR, fp);
	fread(p_T_minor, sizeof(T3), NMINOR, fp);
	fread(p_AAdot, sizeof(AAdot), NMINOR, fp);
	fread(p_v_n, sizeof(f64_vec3), NMINOR, fp);
	fread(p_vie, sizeof(v4), NMINOR, fp);
	fread(p_B, sizeof(f64_vec3), NMINOR, fp);

	//	fread(p_Lap_Az, Nminor * sizeof(f64));
	//	fread(p_v_overall_minor, Nminor * sizeof(f64_vec2));
	//	fread(p_n_upwind_minor, Nminor * sizeof(nvals));

	fread(p_AreaMinor, sizeof(f64), NMINOR, fp);
	fread(p_AreaMajor, sizeof(f64), Nverts, fp);

	fclose(fp);
	printf("File read done.\n");
}


void cuSyst::SendToHost(cuSyst & Xhost)
{
	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	if ((!CallMAC(cudaMemcpy(Xhost.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBCtri_vert, p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBCneigh_vert, p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_izNeigh_TriMinor, p_izNeigh_TriMinor, Ntris * 6 * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyDeviceToHost)))
		
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_periodic_corner_flags, p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_neigh_index, p_tri_neigh_index, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_periodic_neigh_flags, p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_iVolley, p_iVolley, Nverts * sizeof(char), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_v_overall_minor, p_v_overall_minor, Nminor * sizeof(f64_vec2), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyDeviceToHost)))
		)
	{
		// success - do nothing
	}
	else {
		printf("cudaMemcpy error");
		getch();
	}
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");

}
void cuSyst::SendToDevice(cuSyst & Xdevice)
{
	//printf("Value sending to device [92250 * 6 + 1]: %d", p_szPBC_triminor[92250 * 6 + 1]);
	//getch();

	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	if (
		   (!CallMAC(cudaMemcpy(Xdevice.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBCtri_vert, p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBCneigh_vert, p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_izNeigh_TriMinor, p_izNeigh_TriMinor, Ntris * 6 * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_periodic_corner_flags, p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_neigh_index, p_tri_neigh_index, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_periodic_neigh_flags, p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_iVolley, p_iVolley, Nverts * sizeof(char), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_v_overall_minor, p_v_overall_minor, Nminor * sizeof(f64_vec2), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyHostToDevice)))

		)
	{

	}
	else {
		printf("SendToDevice error"); getch();
	}
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");
}


void cuSyst::ReportDifferencesHost(cuSyst &X2)
{
	long iMinor, iVertex, iTri;

	printf("\nDIFFERENCES:\n");
	for (iMinor = 0; iMinor < Nminor; iMinor++)
	{
		if (X2.p_info[iMinor].flag != p_info[iMinor].flag) printf("%d flag %d %d \n",iMinor, X2.p_info[iMinor].flag, p_info[iMinor].flag);
		if (X2.p_info[iMinor].pos.x != p_info[iMinor].pos.x) printf("%d pos %1.9E %1.9E \n", iMinor, X2.p_info[iMinor].pos.x, p_info[iMinor].pos.x);
	}
	for (iVertex = 0; iVertex < Nverts; iVertex++)
	{
		if (memcmp(X2.p_izTri_vert + iVertex*MAXNEIGH_d, p_izTri_vert + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long)) != 0) {
			printf("vertex %d izTri_vert \n", iVertex);
		}
		if (memcmp(X2.p_izNeigh_vert + iVertex*MAXNEIGH_d, p_izNeigh_vert + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long)) != 0) {
			printf("vertex %d izNeigh_vert \n", iVertex);
		}
		if (memcmp(X2.p_szPBCtri_vert + iVertex*MAXNEIGH_d, p_szPBCtri_vert + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char)) != 0) {
			printf("vertex %d szPBCtri_vert \n", iVertex);
		}
		if (memcmp(X2.p_szPBCneigh_vert + iVertex*MAXNEIGH_d, p_szPBCneigh_vert + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char)) != 0) {
			printf("vertex %d szPBCneigh_vert \n", iVertex);
		}
	};

	for (iTri = 0; iTri < Ntris; iTri++)
	{
		if (memcmp(X2.p_izNeigh_TriMinor + iTri*6, p_izNeigh_TriMinor + iTri*6, 6 * sizeof(long)) != 0) {
			printf("tri %d izNeigh_TriMinor \n", iTri);
		}
		if (memcmp(X2.p_szPBC_triminor + iTri * 6, p_szPBC_triminor + iTri * 6, 6 * sizeof(char)) != 0) {
			printf("tri %d szPBC_triminor \n", iTri);
		}
		if (memcmp(X2.p_tri_corner_index + iTri, p_tri_corner_index + iTri, sizeof(LONG3)) != 0) {
			printf("tri %d p_tri_corner_index %d %d %d | %d %d %d \n", iTri,
				(X2.p_tri_corner_index + iTri)->i1, (X2.p_tri_corner_index + iTri)->i2, (X2.p_tri_corner_index + iTri)->i3,
				(p_tri_corner_index + iTri)->i1, (p_tri_corner_index + iTri)->i2, (p_tri_corner_index + iTri)->i3);
		}
		if (memcmp(X2.p_tri_periodic_corner_flags + iTri, p_tri_periodic_corner_flags + iTri, sizeof(CHAR4)) != 0) {
			printf("tri %d p_tri_periodic_corner_flags \n", iTri);
		}
		if (memcmp(X2.p_tri_neigh_index + iTri, p_tri_neigh_index + iTri, sizeof(LONG3)) != 0) {
			printf("tri %d p_tri_neigh_index \n", iTri);
		}
		if (memcmp(X2.p_tri_periodic_neigh_flags + iTri, p_tri_periodic_neigh_flags + iTri, sizeof(CHAR4)) != 0) {
			printf("tri %d p_tri_periodic_neigh_flags \n", iTri);
		}
		if (memcmp(X2.p_who_am_I_to_corner + iTri, p_who_am_I_to_corner + iTri, sizeof(LONG3)) != 0) {
			printf("tri %d p_who_am_I_to_corner \n", iTri);
		}
	};
	for (iVertex = 0; iVertex < Nverts; iVertex++)
	{
		if (memcmp(X2.p_n_major + iVertex, p_n_major + iVertex, sizeof(nvals)) != 0) {
			printf("n_major %d %1.10E %1.10E \n", iVertex, (X2.p_n_major + iVertex)->n, (p_n_major + iVertex)->n);
		}
	};
	for (iMinor = 0; iMinor < Nminor; iMinor++)
	{
		if (memcmp(X2.p_n_minor + iMinor, p_n_minor + iMinor, sizeof(nvals)) != 0) {
			printf("n_minor %d %1.10E %1.10E \n", iMinor, (X2.p_n_minor + iMinor)->n, (p_n_minor + iMinor)->n);
		} // hmm
		if (memcmp(X2.p_T_minor + iMinor, p_T_minor + iMinor, sizeof(T3)) != 0) {
			printf("T_minor %d %1.10E %1.10E \n", iMinor, (X2.p_T_minor + iMinor)->Te, (p_T_minor + iMinor)->Te);
		} 
		if (memcmp(X2.p_AAdot + iMinor, p_AAdot + iMinor, sizeof(AAdot)) != 0) {
			printf("AAdot %d %1.10E %1.10E \n", iMinor, (X2.p_AAdot + iMinor)->Azdot, (p_AAdot + iMinor)->Azdot);
		}
		if (memcmp(X2.p_v_n + iMinor, p_v_n + iMinor, sizeof(f64_vec3)) != 0) {
			printf("v_n %d %1.10E %1.10E \n", iMinor, (X2.p_v_n + iMinor)->x, (p_v_n + iMinor)->x);
		}
		if (memcmp(X2.p_vie + iMinor, p_vie + iMinor, sizeof(v4)) != 0) {
			printf("vie %d %1.10E %1.10E \n", iMinor, (X2.p_vie + iMinor)->vez, (p_vie + iMinor)->vez);
		}
		if (memcmp(X2.p_B + iMinor, p_B + iMinor, sizeof(f64_vec3)) != 0) {
			printf("B %d %1.10E %1.10E \n", iMinor, (X2.p_B + iMinor)->x, (p_B + iMinor)->x);
		}
		if (memcmp(X2.p_AreaMinor + iMinor, p_AreaMinor + iMinor, sizeof(f64)) != 0) {
			printf("AreaMinor %d %1.10E %1.10E \n", iMinor, *(X2.p_AreaMinor + iMinor), *(p_AreaMinor + iMinor));
		}
	}
	printf("Difference detection done! \n\n");

}

void cuSyst::Output(const char * filename)
{
	FILE * fp = fopen(filename, "w");
	if (fp != 0) {

		long i;
		for (i = 0; i < NUMVERTICES; i++)
		{
			fprintf(fp, "izTri %d : %d %d %d %d %d %d \n",
				i, p_izTri_vert[MAXNEIGH*i + 0],
				p_izTri_vert[MAXNEIGH*i + 1],
				p_izTri_vert[MAXNEIGH*i + 2],
				p_izTri_vert[MAXNEIGH*i + 3],
				p_izTri_vert[MAXNEIGH*i + 4],
				p_izTri_vert[MAXNEIGH*i + 5]);
		}
		for (i = 0; i < NMINOR; i++)
		{
			fprintf(fp, "%d pos %1.14E %1.14E T %1.14E %1.14E %1.14E n %1.14E vxy %1.14E %1.14E vez %1.14E\n",
				i, p_info[i].pos.x, p_info[i].pos.y, p_T_minor[i].Tn, p_T_minor[i].Ti, p_T_minor[i].Te,
				p_n_minor[i].n, p_vie[i].vxy.x, p_vie[i].vxy.y, p_vie[i].vez );
		}
		fclose(fp);
	}
	else {
		printf("file error: cannot open %s ..\n", filename);
	}
}
void cuSyst::PopulateFromTriMesh(TriMesh * pX)
{
	// AsSUMES THIS cuSyst has been allocated on the host.
	// USES pTri->cent

	// Variables on host are called TriMinorNeighLists and TriMinorPBCLists
	memcpy(p_izNeigh_TriMinor, pX->TriMinorNeighLists, Ntris * 6 * sizeof(long)); // pointless that we duplicate it but nvm
	memcpy(p_szPBC_triminor, pX->TriMinorPBCLists, Ntris * 6 * sizeof(char));

	if ((Nverts != pX->numVertices) ||
		(Ntris != pX->numTriangles))
	{
		printf("ERROR (nVerts %d != pX->numVertices %d) || (nTris != pX->numTriangles)\n",
			Nverts, pX->numVertices);
		getch();
		return;
	}

	plasma_data data;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, &(pX->pData[iMinor]), sizeof(plasma_data));
		p_n_minor[iMinor].n = data.n;
		p_n_minor[iMinor].n_n = data.n_n;
		if (iMinor >= BEGINNING_OF_CENTRAL) {
			p_n_major[iMinor - BEGINNING_OF_CENTRAL].n = data.n;
			p_n_major[iMinor - BEGINNING_OF_CENTRAL].n_n = data.n_n;

		}
		p_T_minor[iMinor].Tn = data.Tn;
		p_T_minor[iMinor].Ti = data.Ti;
		p_T_minor[iMinor].Te = data.Te;
		p_AAdot[iMinor].Az = data.Az;
		p_AAdot[iMinor].Azdot = data.Azdot;
		p_v_n[iMinor] = data.v_n;
		p_vie[iMinor].vxy = data.vxy;
		p_vie[iMinor].vez = data.vez;
		p_vie[iMinor].viz = data.viz;

		//if (iMinor == 25964 - BEGINNING_OF_CENTRAL) {
		//	printf("iMinor %d p_vie[iMinor].vez %1.10E viz %1.10E\n", iMinor, p_vie[iMinor].vez, p_vie[iMinor].viz);
		////	getch();
		//}

		p_B[iMinor] = data.B;
		p_AreaMinor[iMinor] = pX->AreaMinorArray[iMinor];
	}
	
	pX->SetupMajorPBCTriArrays();
	// AreaMajor??? pVertex->AreaCell?
	Vertex * pVertex;
	pVertex = pX->X;
	long izTri[MAXNEIGH],izNeigh[MAXNEIGH];
	char szPBCtri[MAXNEIGH], szPBCneigh[MAXNEIGH];
	short tri_len, neigh_len;
	long iVertex;
	short i;
	structural info;
	for (iVertex = 0; iVertex < Nverts; iVertex++)
	{
		
		tri_len = pVertex->GetTriIndexArray(izTri);
		info.neigh_len = tri_len;
		memset(izTri+tri_len, 0, sizeof(long)*(MAXNEIGH-tri_len));
		memcpy(p_izTri_vert + iVertex*MAXNEIGH, izTri, sizeof(long)*MAXNEIGH);

		neigh_len = pVertex->GetNeighIndexArray(izNeigh);
		memset(izNeigh + neigh_len, 0, sizeof(long)*(MAXNEIGH - neigh_len));
		memcpy(p_izNeigh_vert + iVertex*MAXNEIGH,izNeigh, sizeof(long)*MAXNEIGH);
		
		// For INNERMOST, tri_len != neigh_len. 5 tris inc frills, 4 neighs.
		
		// PB lists:
		memset(szPBCtri + tri_len, 0, sizeof(char)*(MAXNEIGH - tri_len));
		memcpy(szPBCtri, pX->MajorTriPBC[iVertex], sizeof(char)*tri_len);
		memcpy(p_szPBCtri_vert + iVertex*MAXNEIGH, szPBCtri, sizeof(char)*MAXNEIGH);
		
		memset(szPBCneigh, 0, sizeof(char)*MAXNEIGH);
		for (i = 0; i < neigh_len; i++)
		{
			if ((pX->T + izTri[i])->periodic == 0) {
				// do nothing: neighbour must be contiguous
				// tris >= neighs
			} else {
				if (((pX->X + izNeigh[i])->pos.x > 0.0) && (pVertex->pos.x < 0.0))
					szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
				if (((pX->X + izNeigh[i])->pos.x < 0.0) && (pVertex->pos.x > 0.0))
					szPBCneigh[i] = ROTATE_ME_CLOCKWISE;
			};
		}
		memcpy(p_szPBCneigh_vert + iVertex*MAXNEIGH, szPBCneigh, sizeof(char)*MAXNEIGH);
		info.flag = pVertex->flags;
		info.pos = pVertex->pos;
		p_info[iVertex + BEGINNING_OF_CENTRAL] = info;

		p_iVolley[iVertex] = (char)(pVertex->iVolley);
		++pVertex;
	};
	

	long iTri; 
	// Triangle structural?
	Triangle * pTri = pX->T;
	for (iTri = 0; iTri < Ntris; iTri++)
	{
		LONG3 tri_corner_index;
		CHAR4 tri_periodic_corner_flags;
		LONG3 who_am_I_to_corner;
		LONG3 tri_neigh_index;
		CHAR4 tri_periodic_neigh_flags;

		tri_corner_index.i1 = pTri->cornerptr[0] - pX->X;
		tri_corner_index.i2 = pTri->cornerptr[1] - pX->X;
		tri_corner_index.i3 = pTri->cornerptr[2] - pX->X;
		p_tri_corner_index[iTri] = tri_corner_index;
		tri_neigh_index.i1 = pTri->neighbours[0] - pX->T;
		tri_neigh_index.i2 = pTri->neighbours[1] - pX->T;
		tri_neigh_index.i3 = pTri->neighbours[2] - pX->T;
		p_tri_neigh_index[iTri] = tri_neigh_index;

		tri_len = pTri->cornerptr[0]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i1 = i;
		}
		tri_len = pTri->cornerptr[1]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i2 = i;
		}
		tri_len = pTri->cornerptr[2]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i3 = i;
		}
		p_who_am_I_to_corner[iTri] = who_am_I_to_corner;
		
		memset(&tri_periodic_corner_flags, 0, sizeof(CHAR4));
		tri_periodic_corner_flags.flag = pTri->u8domain_flag;
		if (pTri->periodic != 0) {
			if (pTri->cornerptr[0]->pos.x > 0.0) tri_periodic_corner_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[1]->pos.x > 0.0) tri_periodic_corner_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[2]->pos.x > 0.0) tri_periodic_corner_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
		}
		p_tri_periodic_corner_flags[iTri] = tri_periodic_corner_flags;
				
		memset(&tri_periodic_neigh_flags, 0, sizeof(CHAR4));
		tri_periodic_neigh_flags.flag = pTri->u8domain_flag;
		if ((pTri->periodic == 0) && (pTri->cent.x > 0.0)) {
			if (pTri->neighbours[0]->periodic != 0) 
				tri_periodic_neigh_flags.per0 = ROTATE_ME_CLOCKWISE;
			if (pTri->neighbours[1]->periodic != 0)
				tri_periodic_neigh_flags.per1 = ROTATE_ME_CLOCKWISE;
			if (pTri->neighbours[2]->periodic != 0)
				tri_periodic_neigh_flags.per2 = ROTATE_ME_CLOCKWISE;
		} else {
			// if we are NOT periodic but on left, neighs are not rotated rel to us.
			// If we ARE periodic but neigh is not and neigh cent > 0.0 then it is rotated.
			if (pTri->periodic != 0) {
				if ((pTri->neighbours[0]->periodic == 0) && (pTri->neighbours[0]->cent.x > 0.0))
					tri_periodic_neigh_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
				if ((pTri->neighbours[1]->periodic == 0) && (pTri->neighbours[1]->cent.x > 0.0))
					tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
				if ((pTri->neighbours[2]->periodic == 0) && (pTri->neighbours[2]->cent.x > 0.0))
					tri_periodic_neigh_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
			}
		}
		p_tri_periodic_neigh_flags[iTri] = tri_periodic_neigh_flags;
		info.pos = pTri->cent;
		info.flag = pTri->u8domain_flag;
		info.neigh_len = 6;
		p_info[iTri] = info;
		++pTri;
	};
	
}

void cuSyst::CopyStructuralDetailsFrom(cuSyst & src) // this assume both live on device
{
	// info contains flag .... do we know that?
	cudaMemcpy(p_info, src.p_info, sizeof(structural)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izTri_vert, src.p_izTri_vert, sizeof(long)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izNeigh_vert, src.p_izNeigh_vert, sizeof(long)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBCtri_vert, src.p_szPBCtri_vert, sizeof(char)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBCneigh_vert, src.p_szPBCneigh_vert, sizeof(char)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izNeigh_TriMinor, src.p_izNeigh_TriMinor, sizeof(long)*6*Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBC_triminor, src.p_szPBC_triminor, sizeof(char)*6*Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_corner_index, src.p_tri_corner_index, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_periodic_corner_flags, src.p_tri_periodic_corner_flags, sizeof(CHAR4) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_neigh_index, src.p_tri_neigh_index, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_periodic_neigh_flags, src.p_tri_periodic_neigh_flags, sizeof(CHAR4) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_who_am_I_to_corner, src.p_who_am_I_to_corner, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_iVolley, src.p_iVolley, sizeof(char)*Nverts, cudaMemcpyDeviceToDevice);
	// find another way would be better. Just a waste of memory and processing having duplicate info, creates unnecessary risks.
}

void cuSyst::PopulateTriMesh(TriMesh * pX)
{
	// AsSUMES THIS cuSyst has been allocated on the host.

	long izTri[MAXNEIGH], izNeigh[MAXNEIGH];
	char szPBCtri[MAXNEIGH], szPBCneigh[MAXNEIGH];
	short tri_len, neigh_len;

	plasma_data data;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		data.n = p_n_minor[iMinor].n ;
		data.n_n = p_n_minor[iMinor].n_n ;
		if (iMinor >= BEGINNING_OF_CENTRAL) {
			data.n = p_n_major[iMinor - BEGINNING_OF_CENTRAL].n ;
			data.n_n = p_n_major[iMinor - BEGINNING_OF_CENTRAL].n_n ;
		} 
		data.Tn = p_T_minor[iMinor].Tn;
		data.Ti = p_T_minor[iMinor].Ti ;
		data.Te = p_T_minor[iMinor].Te ;
		data.Az = p_AAdot[iMinor].Az ;
		data.Azdot = p_AAdot[iMinor].Azdot ;
		data.v_n = p_v_n[iMinor] ;
		data.vxy = p_vie[iMinor].vxy;
		data.vez = p_vie[iMinor].vez;
		data.viz = p_vie[iMinor].viz ;
		data.B = p_B[iMinor] ;
		
		data.pos = p_info[iMinor].pos;

		memcpy(&(pX->pData[iMinor]), &data, sizeof(plasma_data));
		pX->AreaMinorArray[iMinor] = p_AreaMinor[iMinor];
	};

	// UNTIL we go the whole hog and change graphs to cuSyst.

	structural info;
	long iTri, iVertex;

	Vertex * pVertex = pX->X;
	for (iVertex = 0; iVertex < Nverts; iVertex++)
	{
		info = p_info[iVertex + BEGINNING_OF_CENTRAL];
		pVertex->pos = info.pos;
		pVertex->flags = info.flag;

		//tri_len = pVertex->GetTriIndexArray(izTri);
		//info.neigh_len = tri_len;
		//memset(izTri + tri_len, 0, sizeof(long)*(MAXNEIGH - tri_len));
		//memcpy(p_izTri_vert + iVertex*MAXNEIGH, izTri, sizeof(long)*MAXNEIGH);

		memcpy(izTri, p_izTri_vert + iVertex*MAXNEIGH, sizeof(long)*MAXNEIGH);
		pVertex->SetTriIndexArray(izTri, info.neigh_len); // FOR SOME REASON WE PUT == tri_len when we x-ferred????
		//neigh_len = pVertex->GetNeighIndexArray(izNeigh); 
		//memset(izNeigh + neigh_len, 0, sizeof(long)*(MAXNEIGH - neigh_len));
		//memcpy(p_izNeigh_vert + iVertex*MAXNEIGH, izNeigh, sizeof(long)*MAXNEIGH);

		memcpy(izNeigh, p_izNeigh_vert + iVertex*MAXNEIGH, sizeof(long)*MAXNEIGH);
		pVertex->SetNeighIndexArray(izNeigh, info.neigh_len + (((info.flag == INNERMOST) || (info.flag == OUTERMOST))? -1:0));
		// For INNERMOST, tri_len != neigh_len. 5 tris inc frills, 4 neighs.
		
		// PB lists:
		//memcpy(szPBCtri, pX->MajorTriPBC[iVertex], sizeof(char)*tri_len);
		memcpy(szPBCtri, p_szPBCtri_vert + iVertex*MAXNEIGH, sizeof(char)*MAXNEIGH);
		memcpy(pX->MajorTriPBC[iVertex], szPBCtri, sizeof(char)*MAXNEIGH);
		pVertex->iVolley = p_iVolley[iVertex];

		++pVertex;
	}
	
	printf(".....");

	// Triangle structural?
	Triangle * pTri = pX->T;
	for (iTri = 0; iTri < Ntris; iTri++)
	{
		LONG3 tri_corner_index;
		CHAR4 tri_periodic_corner_flags;
		LONG3 who_am_I_to_corner;
		LONG3 tri_neigh_index;
		CHAR4 tri_periodic_neigh_flags;

		tri_corner_index = p_tri_corner_index[iTri];
		pTri->cornerptr[0] = pX->X + tri_corner_index.i1;
		pTri->cornerptr[1] = pX->X + tri_corner_index.i2;
		pTri->cornerptr[2] = pX->X + tri_corner_index.i3;

		tri_neigh_index = p_tri_neigh_index[iTri];
		pTri->neighbours[0] = tri_neigh_index.i1 + pX->T;
		pTri->neighbours[1] = tri_neigh_index.i2 + pX->T;
		pTri->neighbours[2] = tri_neigh_index.i3 + pX->T;
		
		tri_periodic_corner_flags = p_tri_periodic_corner_flags[iTri];
		pTri->periodic = ((tri_periodic_corner_flags.per0 == ROTATE_ME_ANTICLOCKWISE) ? 1 : 0)
			+ ((tri_periodic_corner_flags.per1 == ROTATE_ME_ANTICLOCKWISE) ? 1 : 0)
			+ ((tri_periodic_corner_flags.per2 == ROTATE_ME_ANTICLOCKWISE) ? 1 : 0);
		if (pTri->periodic > 0) pTri->periodic = 3 - pTri->periodic;
		// CPU periodic is how many need to be clockwise rotated.

		info = p_info[iTri];
		pTri->cent = info.pos;
		pTri->u8domain_flag = info.flag;
		++pTri;
	}
	
}
                            
#endif
