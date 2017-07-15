

#include "cuda_struct.h"
#include <conio.h>
#include <malloc.h>
#include <stdio.h>
#include <memory.h>


//#define NOCUDA


// moved down because we want to run without cuda for a minute
#ifdef __CUDACC__


// When built into 1, this must go inside #ifdef __CUDACC__
Systdata::Systdata() {
	bInvoked = false;
	bInvokedHost = false;
	Nverts = 0;	
	Ntris = 0;
	Nminor = 0;
}

void Systdata::Invoke (long N){
	printf("Systdata::Invoke N %d \n",
									// => predicted bytes to allocate %d\n",
			N);	//,(sizeof(f64)*(2+6+6+9+9+6+2+1+1+3+)				

	Nverts = N;
	Ntris = 2*Nverts;
	Nminor = Ntris+Nverts;

	if (bInvoked == false) {

		if (
			( !CallMAC( cudaMalloc((void**)&p_phi,Nverts*sizeof(f64)) ) )
		&&	( !CallMAC( cudaMalloc((void**)&p_phidot,Nverts*sizeof(f64)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_A,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_Adot,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_nT_neut_minor,Nminor*sizeof(nT)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_nT_ion_minor,Nminor*sizeof(nT)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_nT_elec_minor,Nminor*sizeof(nT)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_v_neut,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_v_ion,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_v_elec,Nminor*sizeof(f64_vec3)) ) )

		// Going to need nT for minor also. ...

		&&  ( !CallMAC( cudaMalloc((void**)&p_MomAdditionRate_neut,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_MomAdditionRate_ion,Nminor*sizeof(f64_vec3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_MomAdditionRate_elec,Nminor*sizeof(f64_vec3)) ) )

		&&  ( !CallMAC( cudaMalloc((void**)&p_intercell_heatrate_neut,Nverts*sizeof(f64)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_intercell_heatrate_ion,Nverts*sizeof(f64)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_intercell_heatrate_elec,Nverts*sizeof(f64)) ) )

		// Think about this. n,T are stored on major so heatrate should be calc'd for major.

		&&  ( !CallMAC( cudaMalloc((void**)&p_area,Nverts*sizeof(f64)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_area_minor,Nminor*sizeof(f64)) ) )

		&&  ( !CallMAC( cudaMalloc((void**)&p_info,Nverts*sizeof(structural)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_tri_perinfo,Nminor*sizeof(CHAR4)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_tri_per_neigh,Ntris*sizeof(CHAR4)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_tri_corner_index,Ntris*sizeof(LONG3)) ) )
		&&  ( !CallMAC( cudaMalloc((void**)&p_neigh_tri_index,Ntris*sizeof(LONG3)) ) )

		&&  ( !CallMAC( cudaMalloc((void**)&pIndexNeigh,Nverts*MAXNEIGH_d*sizeof(long))))
		&&  ( !CallMAC( cudaMalloc((void**)&pIndexTri,Nverts*MAXNEIGH_d*sizeof(long))))
	    &&  ( !CallMAC( cudaMalloc((void**)&pPBCneigh,Nverts*MAXNEIGH_d*sizeof(char))))
		&&  ( !CallMAC( cudaMalloc((void**)&pPBCtri,Nverts*MAXNEIGH_d*sizeof(char))))

		&&  ( !CallMAC( cudaMalloc((void**)&p_grad_phi,Nminor*sizeof(f64_vec2))))
	    &&  ( !CallMAC( cudaMalloc((void**)&p_B,Nminor*sizeof(f64_vec3))))
		&&  ( !CallMAC( cudaMalloc((void**)&p_Lap_A,Nminor*sizeof(f64_vec3))))
		&&  ( !CallMAC( cudaMalloc((void**)&p_GradTe,Nminor*sizeof(f64_vec2))))
		&&  ( !CallMAC( cudaMalloc((void**)&p_v_overall,Nminor*sizeof(f64_vec2))))

		// v_overall is a funny one. We are putting tris first then major -- correct?

		&&  ( !CallMAC( cudaMalloc((void**)&p_tri_centroid, Ntris*sizeof(f64_vec2))))
		
		        )
	{
			bInvoked = true;
			Zero();
			printf("Dimensioned for MAXNEIGH_d = %d\n",MAXNEIGH_d);
		} else {
			printf("There was an error in dimensioning Systdata object.\n");
			getch();	getch();
		};
	} else {
		if (Nverts != N) { printf("Systdata Error - Nverts %d != N %d\n",Nverts,N); getch(); }
	};
}
//#else
	
void Systdata::InvokeHost (long N){
	printf("Systdata::InvokeHost N %d\n",N);

	Nverts = N;
	Ntris = 2*Nverts;
	Nminor = Ntris+Nverts;

	if (bInvokedHost == false) {

		p_phi = (f64 *)malloc(Nverts*sizeof(f64));
		p_phidot = (f64 *)malloc(Nverts*sizeof(f64));
		p_A = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_Adot = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_nT_neut_minor = (nT *)malloc(Nminor*sizeof(nT));
		p_nT_ion_minor  = (nT *)malloc(Nminor*sizeof(nT));
		p_nT_elec_minor = (nT *)malloc(Nminor*sizeof(nT));
		p_v_neut = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_v_ion = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_v_elec = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		
		p_MomAdditionRate_neut = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_MomAdditionRate_ion = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_MomAdditionRate_elec = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));

		p_intercell_heatrate_neut = (f64 *)malloc(Nverts*sizeof(f64));
		p_intercell_heatrate_ion = (f64 *)malloc(Nverts*sizeof(f64));
		p_intercell_heatrate_elec = (f64 *)malloc(Nverts*sizeof(f64));

		p_area = (f64 *)malloc(Nverts*sizeof(f64));
		p_area_minor = (f64 *)malloc(Nminor*sizeof(f64));

		p_info = (structural *)malloc(Nverts*sizeof(structural));
		p_tri_perinfo = (CHAR4 *)malloc(Ntris*sizeof(CHAR4));
		p_tri_per_neigh = (CHAR4 *)malloc(Nminor*sizeof(CHAR4));
		p_tri_corner_index = (LONG3 *)malloc(Ntris*sizeof(LONG3));
		p_neigh_tri_index = (LONG3 *)malloc(Ntris*sizeof(LONG3));
		
		pIndexNeigh = (long *)malloc(Nverts*MAXNEIGH_d*sizeof(long));
		pIndexTri = (long *)malloc(Nverts*MAXNEIGH_d*sizeof(long));
		pPBCneigh = (char *)malloc(Nverts*MAXNEIGH_d*sizeof(char));
		pPBCtri = (char *)malloc(Nverts*MAXNEIGH_d*sizeof(char));
		
		p_grad_phi = (f64_vec2 *)malloc(Nminor*sizeof(f64_vec2));
		p_B = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_Lap_A = (f64_vec3 *)malloc(Nminor*sizeof(f64_vec3));
		p_GradTe = (f64_vec2 *)malloc(Nminor*sizeof(f64_vec2));
		p_v_overall = (f64_vec2 *)malloc(Nminor*sizeof(f64_vec2));
		
		p_tri_centroid = (f64_vec2 *)malloc(Ntris*sizeof(f64_vec2));
		
		if ((p_tri_centroid != 0)
			&&
			(p_neigh_tri_index != 0)) // ?
		{
			bInvokedHost = true;
			ZeroHost();
			printf("Dimensioned for MAXNEIGH_d = %d\n",MAXNEIGH_d);
		} else {
			printf("There was an error in dimensioning Systdata object.\n");
			getch();
		};
	} else {
		if (Nverts != N) { printf("Systdata Error - Nverts %d != N %d\n",Nverts,N);
							getch(); }
	};
}

void Systdata::RevokeHost() {
	if (bInvokedHost) 
	{
		free(p_phi);
		free(p_phidot);
		free(p_A);
		free(p_Adot);
		free(p_nT_neut_minor);
		free(p_nT_ion_minor);
		free(p_nT_elec_minor);
		free(p_v_neut);
		free(p_v_ion);
		free(p_v_elec);
		free(p_MomAdditionRate_neut);
		free(p_MomAdditionRate_ion);
		free(p_MomAdditionRate_elec);
		free(p_intercell_heatrate_neut);
		free(p_intercell_heatrate_ion);
		free(p_intercell_heatrate_elec);
		free(p_area);
		free(p_area_minor);
		free(p_info);
		free(p_tri_perinfo);
		free(p_tri_per_neigh);
		free(p_tri_corner_index);
		free(p_neigh_tri_index);
		free(pIndexNeigh);
		free(pIndexTri);
		free(pPBCneigh);
		free(pPBCtri);
	//	free(p_store_grad_phi);
		free(p_B);
	//	free(p_store_Lap_A);
		free(p_GradTe);
		free(p_v_overall);
		free(p_tri_centroid);
	}
}
//#endif

//#ifdef __CUDACC__

void Systdata::Zero () {
	if (bInvoked)
	{

		CallMAC( cudaMemset(p_phi,0,Nverts*sizeof(f64)) );
		CallMAC( cudaMemset(p_phidot,0,Nverts*sizeof(f64)) ) ;
		CallMAC( cudaMemset(p_A,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_Adot,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_nT_neut_minor,0,Nminor*sizeof(nT)) ) ;
		CallMAC( cudaMemset(p_nT_ion_minor,0,Nminor*sizeof(nT)) ) ;
		CallMAC( cudaMemset(p_nT_elec_minor,0,Nminor*sizeof(nT)) ) ;

		CallMAC( cudaMemset(p_v_neut,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_v_ion,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_v_elec,0,Nminor*sizeof(f64_vec3)) ) ;
		
		CallMAC( cudaMemset(p_MomAdditionRate_neut,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_MomAdditionRate_ion,0,Nminor*sizeof(f64_vec3)) ) ;
		CallMAC( cudaMemset(p_MomAdditionRate_elec,0,Nminor*sizeof(f64_vec3)) ) ;

		CallMAC( cudaMemset(p_intercell_heatrate_neut,0,Nverts*sizeof(f64)) ) ;
		CallMAC( cudaMemset(p_intercell_heatrate_ion,0,Nverts*sizeof(f64)) ) ;
		CallMAC( cudaMemset(p_intercell_heatrate_elec,0,Nverts*sizeof(f64)) ) ;
		
		CallMAC( cudaMemset(p_area,0,Nverts*sizeof(f64)) ) ;
		CallMAC( cudaMemset(p_area_minor,0,Nminor*sizeof(f64)) ) ;
		
		CallMAC( cudaMemset(p_info,0,Nverts*sizeof(structural)) ) ;
		CallMAC( cudaMemset(p_tri_perinfo,0,Nminor*sizeof(CHAR4)) ) ;
		CallMAC( cudaMemset(p_tri_per_neigh,0,Ntris*sizeof(CHAR4)) ) ;
		CallMAC( cudaMemset(p_tri_corner_index,0,Ntris*sizeof(LONG3)) ) ;
		CallMAC( cudaMemset(p_neigh_tri_index,0,Ntris*sizeof(LONG3)) ) ;
		
		CallMAC( cudaMemset(pIndexNeigh,0,Nverts*MAXNEIGH_d*sizeof(long)));
		CallMAC( cudaMemset(pIndexTri,0,Nverts*MAXNEIGH_d*sizeof(long)));
	    CallMAC( cudaMemset(pPBCneigh,0,Nverts*MAXNEIGH_d*sizeof(char)));
	    CallMAC( cudaMemset(pPBCtri,0,Nverts*MAXNEIGH_d*sizeof(char)));

		CallMAC( cudaMemset(p_grad_phi,0,Nminor*sizeof(f64_vec2)));
	    CallMAC( cudaMemset(p_B,0,Nminor*sizeof(f64_vec3)));
		CallMAC( cudaMemset(p_Lap_A,0,Nminor*sizeof(f64_vec3)));
		CallMAC( cudaMemset(p_GradTe,0,Nminor*sizeof(f64_vec2)));
		CallMAC( cudaMemset(p_v_overall,0,Nminor*sizeof(f64_vec2)));

		CallMAC( cudaMemset(p_tri_centroid,0,Ntris*sizeof(f64_vec2)));


	} else {
		printf("Zero called before InvokeHost.\n"); getch();
	};
}
//#else
		
void Systdata::ZeroHost () {
	if (bInvokedHost)
	{
		memset(p_phi,0,Nverts*sizeof(f64));
		memset(p_phidot,0,Nverts*sizeof(f64));
		memset(p_A,0,Nminor*sizeof(f64_vec3)) ;
		memset(p_Adot,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_nT_neut_minor,0,Nminor*sizeof(nT))  ;
		memset(p_nT_ion_minor,0,Nminor*sizeof(nT))  ;
		memset(p_nT_elec_minor,0,Nminor*sizeof(nT))  ;
		memset(p_v_neut,0,Nminor*sizeof(f64_vec3)) ;
		memset(p_v_ion,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_v_elec,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_MomAdditionRate_neut,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_MomAdditionRate_ion,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_MomAdditionRate_elec,0,Nminor*sizeof(f64_vec3))  ;
		memset(p_intercell_heatrate_neut,0,Nverts*sizeof(f64))  ;
		memset(p_intercell_heatrate_ion,0,Nverts*sizeof(f64))  ;
		memset(p_intercell_heatrate_elec,0,Nverts*sizeof(f64))  ;
		
		memset(p_area,0,Nverts*sizeof(f64))  ;
		memset(p_info,0,Nverts*sizeof(structural)) ;
		memset(pIndexNeigh,0,Nverts*MAXNEIGH_d*sizeof(long));
		memset(pIndexTri,0,Nverts*MAXNEIGH_d*sizeof(long));
	    memset(pPBCneigh,0,Nverts*MAXNEIGH_d*sizeof(char));
	    memset(pPBCtri,0,Nverts*MAXNEIGH_d*sizeof(char));
		memset(p_grad_phi,0,Nverts*sizeof(f64_vec2));
		memset(p_Lap_A,0,Nverts*sizeof(f64_vec3));
	    memset(p_B,0,Nverts*sizeof(f64_vec3));
	    memset(p_GradTe,0,Nverts*sizeof(f64_vec2));
	    memset(p_v_overall,0,Nverts*sizeof(f64_vec2));

	} else {
		printf("Zero called before InvokeHost.\n"); getch();
	};

}

int Systdata::SaveHost(const char str[])
{
	FILE * fp;
	fp = fopen(str,"wb");
	if (fp == 0) {
		printf("Failed to open %s \n",str);
		getch();
		return 1;
	} else {
		printf(" %s opened \n",str);
		
		fwrite(&Nverts,sizeof(long),1,fp);
		fwrite(&Ntris,sizeof(long),1,fp);
		fwrite(&Nminor,sizeof(long),1,fp);
		fwrite(&numReverseJzTris,sizeof(long),1,fp);

		fwrite(&EzTuning, sizeof(f64),1,fp);
		fwrite(&evaltime, sizeof(f64),1,fp);

		fwrite(&InnermostFrillCentroidRadius, sizeof(f64),1,fp);
		fwrite(&OutermostFrillCentroidRadius, sizeof(f64),1,fp);
		
		// Data:
		// =====
		
		fwrite(p_phi,sizeof(f64),Nverts,fp);
		fwrite(p_phidot,sizeof(f64),Nverts,fp);

		fwrite(p_A,sizeof(f64_vec3),Nminor,fp);
		fwrite(p_Adot,sizeof(f64_vec3),Nminor,fp);

		fwrite(p_nT_neut_minor,sizeof(nT),Nminor,fp);
		fwrite(p_nT_ion_minor,sizeof(nT),Nminor,fp);
		fwrite(p_nT_elec_minor,sizeof(nT),Nminor,fp);

		fwrite(p_v_neut,sizeof(f64_vec3),Nminor,fp);
		fwrite(p_v_ion,sizeof(f64_vec3),Nminor,fp);
		fwrite(p_v_elec,sizeof(f64_vec3),Nminor,fp);

		fwrite(p_area,sizeof(f64),Nverts,fp);
		fwrite(p_area_minor,sizeof(f64),Nminor,fp);
		
		// Structural information:
		// =======================

		fwrite(p_info,sizeof(structural),Nverts,fp);
		fwrite(p_tri_perinfo,sizeof(CHAR4),Nminor,fp);
		fwrite(p_tri_per_neigh,sizeof(CHAR4),Ntris,fp);
		fwrite(p_tri_corner_index,sizeof(LONG3),Ntris,fp);
		fwrite(p_neigh_tri_index,sizeof(LONG3),Ntris,fp);

		fwrite(pIndexNeigh,sizeof(long),MAXNEIGH_d*Nverts,fp);
		fwrite(pIndexTri,sizeof(long),MAXNEIGH_d*Nverts,fp);
		fwrite(pPBCneigh,sizeof(char),MAXNEIGH_d*Nverts,fp);
		fwrite(pPBCtri,sizeof(char),MAXNEIGH_d*Nverts,fp);
		fwrite(p_tri_centroid,sizeof(f64_vec2),Ntris,fp);

		// Derivatives:
		// =============

		// Skip grad phi, Lap A, grad Te.
		fwrite(p_B,sizeof(f64_vec3),Nminor,fp);

		fclose(fp);
		printf("%s closed\n",str);
	};
	return 0;
}

int Systdata::LoadHost(const char str[])
{
	// fwrite is suited to flatpacks:
	FILE * fp;
	fp = fopen(str,"rb");
	
	if (fp == 0) {
		printf("Failed to open %s \n",str);
		getch();
		return 1;
	} else {
		printf(" %s opened to read \n",str);

		rewind(fp);
	
		long Nverts_file, Ntris_file, Nminor_file, numRevJz_file;

		fread(&Nverts_file,sizeof(long),1,fp);
		fread(&Ntris_file,sizeof(long),1,fp);
		fread(&Nminor_file,sizeof(long),1,fp);
		fread(&numRevJz_file,sizeof(long),1,fp);

		if (Nverts_file != Nverts) {
			printf("Nverts_file %d Nverts %d\n",Nverts_file,Nverts);
			getch();
			return 1;
		};
		if (Ntris_file != Ntris) {
			printf("Ntris_file %d Ntris %d\n",Ntris_file,Ntris);
			getch();
			return 1;
		};
		if (Nminor_file != Nminor) {
			printf("Nminor_file %d Nminor %d\n",Nminor_file,Nminor);
			getch();
			return 1;
		};
		if (numRevJz_file != numReverseJzTris) {
			printf("numRevJz_file %d ours %d\n",numRevJz_file,numReverseJzTris);
			getch();
			return 1;
		};

		fread(&EzTuning, sizeof(f64),1,fp);
		fread(&evaltime, sizeof(f64),1,fp);
		fread(&InnermostFrillCentroidRadius, sizeof(f64),1,fp);
		fread(&OutermostFrillCentroidRadius, sizeof(f64),1,fp);
		


		// Data:
		// =====
		fread(p_phi,sizeof(f64),Nverts,fp);
		fread(p_phidot,sizeof(f64),Nverts,fp);

		fread(p_A,sizeof(f64_vec3),Nminor,fp);
		fread(p_Adot,sizeof(f64_vec3),Nminor,fp);

		fread(p_nT_neut_minor,sizeof(nT),Nminor,fp);
		fread(p_nT_ion_minor,sizeof(nT),Nminor,fp);
		fread(p_nT_elec_minor,sizeof(nT),Nminor,fp);

		fread(p_v_neut,sizeof(f64_vec3),Nminor,fp);
		fread(p_v_ion,sizeof(f64_vec3),Nminor,fp);
		fread(p_v_elec,sizeof(f64_vec3),Nminor,fp);

		fread(p_area,sizeof(f64),Nverts,fp);
		fread(p_area_minor,sizeof(f64),Nminor,fp);
		
		// Structural information:
		// =======================

		fread(p_info,sizeof(structural),Nverts,fp);
		fread(p_tri_perinfo,sizeof(CHAR4),Nminor,fp);
		fread(p_tri_per_neigh,sizeof(CHAR4),Ntris,fp);
		fread(p_tri_corner_index,sizeof(LONG3),Ntris,fp);
		fread(p_neigh_tri_index,sizeof(LONG3),Ntris,fp);

		fread(pIndexNeigh,sizeof(long),MAXNEIGH_d*Nverts,fp);
		fread(pIndexTri,sizeof(long),MAXNEIGH_d*Nverts,fp);
		fread(pPBCneigh,sizeof(char),MAXNEIGH_d*Nverts,fp);
		fread(pPBCtri,sizeof(char),MAXNEIGH_d*Nverts,fp);
		fread(p_tri_centroid,sizeof(f64_vec2),Ntris,fp);

		// Derivatives:
		// =============

		// Skip grad phi, Lap A, grad Te.
		fread(p_B,sizeof(f64_vec3),Nminor,fp);

		fclose(fp);
		printf("%s closed\n",str);
	};
	return 0;
}

#endif

#ifdef __CUDACC__

Systdata::~Systdata() {
	if (bInvoked) 
	{
		cudaFree(p_phi);
		cudaFree(p_phidot);
		cudaFree(p_A);
		cudaFree(p_Adot);
		cudaFree(p_nT_neut_minor);
		cudaFree(p_nT_ion_minor);
		cudaFree(p_nT_elec_minor);
		cudaFree(p_v_neut);
		cudaFree(p_v_ion);
		cudaFree(p_v_elec);
		cudaFree(p_MomAdditionRate_neut);
		cudaFree(p_MomAdditionRate_ion);
		cudaFree(p_MomAdditionRate_elec);
		cudaFree(p_intercell_heatrate_neut);
		cudaFree(p_intercell_heatrate_ion);
		cudaFree(p_intercell_heatrate_elec);
		cudaFree(p_area);
		cudaFree(p_area_minor);
		cudaFree(p_info);
		cudaFree(p_tri_perinfo);
		cudaFree(p_tri_per_neigh);
		cudaFree(p_tri_corner_index);
		cudaFree(p_neigh_tri_index);
		cudaFree(pIndexNeigh);
		cudaFree(pIndexTri);
		cudaFree(pPBCneigh);
		cudaFree(pPBCtri);
		cudaFree(p_grad_phi);
		cudaFree(p_B);
		cudaFree(p_Lap_A);
		cudaFree(p_GradTe);
		cudaFree(p_v_overall);
		cudaFree(p_tri_centroid);
	}
}
#else

#ifdef NOCUDA
//// Temporary to compile without CUDA:
Systdata::~Systdata() {
	// do nothing
}

#endif


// I see difficulties...
	// This destructor will be different;
	// It cannot exist for the host object.

#endif