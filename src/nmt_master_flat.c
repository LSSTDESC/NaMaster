#include "utils.h"

static void purify_generic_flat(nmt_field_flat *fl,flouble *mask,fcomplex **walm0,flouble **maps_in,fcomplex **alms_out)
{
  if(fl->pure_b || fl->pure_e) {
    nmt_purify_flat(fl,mask,walm0,maps_in,maps_in,alms_out);
  }
  else {
    int im1;
    for(im1=0;im1<fl->nmaps;im1++)
      fs_map_product(fl->fs,maps_in[im1],mask,maps_in[im1]);
    fs_map2alm(fl->fs,1,2*fl->pol,maps_in,alms_out);
  }
}

void nmt_workspace_flat_free(nmt_workspace_flat *w)
{
  int ii;
  gsl_permutation_free(w->coupling_matrix_perm);
  gsl_matrix_free(w->coupling_matrix_binned_gsl);
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    free(w->coupling_matrix_unbinned[ii]);
  free(w->coupling_matrix_unbinned);
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    free(w->coupling_matrix_binned[ii]);
  free(w->coupling_matrix_binned);
  free(w->n_cells);
  nmt_bins_flat_free(w->bin);
  nmt_flatsky_info_free(w->fs);
  free(w->mask1);
  free(w->mask2);
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  free(w->maskprod);
#endif //_ENABLE_FLAT_THEORY_ACCURATE
  free(w);
}

static nmt_workspace_flat *nmt_workspace_flat_new(int ncls,nmt_flatsky_info *fs,
						  nmt_binning_scheme_flat *bin,
						  flouble lmn_x,flouble lmx_x,
						  flouble lmn_y,flouble lmx_y,int is_teb)
{
  int ii,ib=0;
  nmt_workspace_flat *w=my_malloc(sizeof(nmt_workspace_flat));
  w->is_teb=is_teb;
  w->ncls=ncls;

  w->ellcut_x[0]=lmn_x;
  w->ellcut_x[1]=lmx_x;
  w->ellcut_y[0]=lmn_y;
  w->ellcut_y[1]=lmx_y;

  w->bin=nmt_bins_flat_create(bin->n_bands,bin->ell_0_list,bin->ell_f_list);
  w->lmax=w->bin->ell_f_list[w->bin->n_bands-1];

  w->fs=nmt_flatsky_info_alloc(fs->nx,fs->ny,fs->lx,fs->ly);

  w->mask1=my_malloc(fs->npix*sizeof(flouble));
  w->mask2=my_malloc(fs->npix*sizeof(flouble));
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  w->maskprod=my_malloc(w->fs->npix*sizeof(flouble));
#endif //_ENABLE_FLAT_THEORY_ACCURATE

  w->n_cells=my_calloc(w->bin->n_bands,sizeof(int));

  w->coupling_matrix_unbinned=my_malloc(w->ncls*w->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    w->coupling_matrix_unbinned[ii]=my_calloc(w->ncls*w->fs->n_ell,sizeof(flouble));

  w->coupling_matrix_binned=my_malloc(w->ncls*w->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    w->coupling_matrix_binned[ii]=my_calloc(w->ncls*w->bin->n_bands,sizeof(flouble));

  w->coupling_matrix_binned_gsl=gsl_matrix_alloc(w->ncls*w->bin->n_bands,w->ncls*w->bin->n_bands);
  w->coupling_matrix_perm=gsl_permutation_alloc(w->ncls*w->bin->n_bands);

  return w;
}

nmt_workspace_flat *nmt_workspace_flat_read(char *fname)
{
  int ii,nx,ny;
  flouble lx,ly;
  nmt_workspace_flat *w=my_malloc(sizeof(nmt_workspace_flat));
  FILE *fi=my_fopen(fname,"rb");

  my_fread(&(w->is_teb),sizeof(int),1,fi);
  my_fread(&(w->ncls),sizeof(int),1,fi);

  my_fread(w->ellcut_x,sizeof(flouble),2,fi);
  my_fread(w->ellcut_y,sizeof(flouble),2,fi);

  my_fread(&(w->pe1),sizeof(int),1,fi);
  my_fread(&(w->pe2),sizeof(int),1,fi);
  my_fread(&(w->pb1),sizeof(int),1,fi);
  my_fread(&(w->pb2),sizeof(int),1,fi);

  my_fread(&nx,sizeof(int),1,fi);
  my_fread(&ny,sizeof(int),1,fi);
  my_fread(&lx,sizeof(flouble),1,fi);
  my_fread(&ly,sizeof(flouble),1,fi);
  w->fs=nmt_flatsky_info_alloc(nx,ny,lx,ly);

  w->mask1=my_malloc(w->fs->npix*sizeof(flouble));
  my_fread(w->mask1,sizeof(flouble),w->fs->npix,fi);
  w->mask2=my_malloc(w->fs->npix*sizeof(flouble));
  my_fread(w->mask2,sizeof(flouble),w->fs->npix,fi);
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  w->maskprod=my_malloc(w->fs->npix*sizeof(flouble));
  my_fread(w->maskprod,sizeof(flouble),w->fs->npix,fi);
#endif //_ENABLE_FLAT_THEORY_ACCURATE

  w->bin=my_malloc(sizeof(nmt_binning_scheme_flat));
  my_fread(&(w->bin->n_bands),sizeof(int),1,fi);
  w->bin->ell_0_list=my_malloc(w->bin->n_bands*sizeof(flouble));
  w->bin->ell_f_list=my_malloc(w->bin->n_bands*sizeof(flouble));
  my_fread(w->bin->ell_0_list,sizeof(flouble),w->bin->n_bands,fi);
  my_fread(w->bin->ell_f_list,sizeof(flouble),w->bin->n_bands,fi);
  w->lmax=w->bin->ell_f_list[w->bin->n_bands-1];

  w->n_cells=my_malloc(w->bin->n_bands*sizeof(int));
  my_fread(w->n_cells,sizeof(int),w->bin->n_bands,fi);

  w->coupling_matrix_unbinned=my_malloc(w->ncls*w->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++) {
    w->coupling_matrix_unbinned[ii]=my_malloc(w->ncls*w->fs->n_ell*sizeof(flouble));
    my_fread(w->coupling_matrix_unbinned[ii],sizeof(flouble),w->ncls*w->fs->n_ell,fi);
  }

  w->coupling_matrix_binned=my_malloc(w->ncls*w->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++) {
    w->coupling_matrix_binned[ii]=my_malloc(w->ncls*w->bin->n_bands*sizeof(flouble));
    my_fread(w->coupling_matrix_binned[ii],sizeof(flouble),w->ncls*w->bin->n_bands,fi);
  }

  w->coupling_matrix_binned_gsl=gsl_matrix_alloc(w->ncls*w->bin->n_bands,w->ncls*w->bin->n_bands);
  w->coupling_matrix_perm=gsl_permutation_alloc(w->ncls*w->bin->n_bands);
  gsl_matrix_fread(fi,w->coupling_matrix_binned_gsl);
  gsl_permutation_fread(fi,w->coupling_matrix_perm);
  fclose(fi);

  return w;
}

void nmt_workspace_flat_write(nmt_workspace_flat *w,char *fname)
{
  int ii;
  FILE *fo=my_fopen(fname,"wb");

  my_fwrite(&(w->is_teb),sizeof(int),1,fo);
  my_fwrite(&(w->ncls),sizeof(int),1,fo);
  my_fwrite(w->ellcut_x,sizeof(flouble),2,fo);
  my_fwrite(w->ellcut_y,sizeof(flouble),2,fo);

  my_fwrite(&(w->pe1),sizeof(int),1,fo);
  my_fwrite(&(w->pe2),sizeof(int),1,fo);
  my_fwrite(&(w->pb1),sizeof(int),1,fo);
  my_fwrite(&(w->pb2),sizeof(int),1,fo);

  my_fwrite(&(w->fs->nx),sizeof(int),1,fo);
  my_fwrite(&(w->fs->ny),sizeof(int),1,fo);
  my_fwrite(&(w->fs->lx),sizeof(flouble),1,fo);
  my_fwrite(&(w->fs->ly),sizeof(flouble),1,fo);

  my_fwrite(w->mask1,sizeof(flouble),w->fs->npix,fo);
  my_fwrite(w->mask2,sizeof(flouble),w->fs->npix,fo);
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  my_fwrite(w->maskprod,sizeof(flouble),w->fs->npix,fo);
#endif //_ENABLE_FLAT_THEORY_ACCURATE

  my_fwrite(&(w->bin->n_bands),sizeof(int),1,fo);
  my_fwrite(w->bin->ell_0_list,sizeof(flouble),w->bin->n_bands,fo);
  my_fwrite(w->bin->ell_f_list,sizeof(flouble),w->bin->n_bands,fo);

  my_fwrite(w->n_cells,sizeof(int),w->bin->n_bands,fo);

  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    my_fwrite(w->coupling_matrix_unbinned[ii],sizeof(flouble),w->ncls*w->fs->n_ell,fo);

  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    my_fwrite(w->coupling_matrix_binned[ii],sizeof(flouble),w->ncls*w->bin->n_bands,fo);

  gsl_matrix_fwrite(fo,w->coupling_matrix_binned_gsl);
  gsl_permutation_fwrite(fo,w->coupling_matrix_perm);
  
  fclose(fo);
}

static int check_flatsky_infos(nmt_flatsky_info *fs1,nmt_flatsky_info *fs2)
{
  if(fs1->nx!=fs2->nx) return 1;
  if(fs1->ny!=fs2->ny) return 1;
  if(fs1->lx!=fs2->lx) return 1;
  if(fs1->ly!=fs2->ly) return 1;
  return 0;
}

nmt_workspace_flat *nmt_compute_coupling_matrix_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						     nmt_binning_scheme_flat *bin,
						     flouble lmn_x,flouble lmx_x,
						     flouble lmn_y,flouble lmx_y,int is_teb)
{
  if(check_flatsky_infos(fl1->fs,fl2->fs))
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can only correlate fields defined on the same pixels!\n");

  int n_cl=fl1->nmaps*fl2->nmaps;
  if(is_teb) {
    if(!((fl1->pol==0) && (fl2->pol==1)))
      report_error(NMT_ERROR_INCONSISTENT,"For T-E-B MCM the first input field must be spin-0 and the se\
cond spin-2\n");
    n_cl=7;
  }
  
  int ii;
  nmt_workspace_flat *w=nmt_workspace_flat_new(n_cl,fl1->fs,bin,
					       lmn_x,lmx_x,lmn_y,lmx_y,is_teb);
  nmt_flatsky_info *fs=fl1->fs;
  w->pe1=fl1->pure_e;
  w->pe2=fl2->pure_e;
  w->pb1=fl1->pure_b;
  w->pb2=fl2->pure_b;

  for(ii=0;ii<fl1->fs->npix;ii++)
    w->mask1[ii]=fl1->mask[ii];
  for(ii=0;ii<fl2->fs->npix;ii++)
    w->mask2[ii]=fl2->mask[ii];
  
  fcomplex *cmask1,*cmask2;
  flouble *maskprod,*cosarr,*sinarr,*kmodarr,*beamprod;
  int *i_band,*i_band_nocut,*i_ring;
  cmask1=dftw_malloc(fs->ny*(fs->nx/2+1)*sizeof(fcomplex));
  fs_map2alm(fl1->fs,1,0,&(fl1->mask),&cmask1);
  if(fl1==fl2)
    cmask2=cmask1;
  else {
    cmask2=dftw_malloc(fs->ny*(fs->nx/2+1)*sizeof(fcomplex));
    fs_map2alm(fl2->fs,1,0,&(fl2->mask),&cmask2);
  }
  i_ring=my_malloc(w->fs->npix*sizeof(int));
  i_band=my_malloc(w->fs->npix*sizeof(int));
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  maskprod=w->maskprod;
#else //_ENABLE_FLAT_THEORY_ACCURATE
  maskprod=my_malloc(w->fs->npix*sizeof(flouble));
#endif //_ENABLE_FLAT_THEORY_ACCURATE
  i_band_nocut=my_malloc(w->fs->npix*sizeof(int));
  kmodarr=dftw_malloc(w->fs->npix*sizeof(flouble));
  beamprod=dftw_malloc(w->fs->npix*sizeof(flouble));
  if(w->ncls>1) {
    cosarr=dftw_malloc(w->fs->npix*sizeof(flouble));
    sinarr=dftw_malloc(w->fs->npix*sizeof(flouble));
  }

  int *x_out_range,*y_out_range;
  x_out_range=my_calloc(fs->nx,sizeof(int));
  y_out_range=my_calloc(fs->ny,sizeof(int));
  for(ii=0;ii<fs->nx;ii++) {
    flouble k;
    if(2*ii<=fs->nx) k=ii*2*M_PI/fs->lx;
    else k=-(fs->nx-ii)*2*M_PI/fs->lx;
    if((k<=w->ellcut_x[1]) && (k>=w->ellcut_x[0]))
      x_out_range[ii]=1;
  }
  for(ii=0;ii<fs->ny;ii++) {
    flouble k;
    if(2*ii<=fs->ny) k=ii*2*M_PI/fs->ly;
    else k=-(fs->ny-ii)*2*M_PI/fs->ly;
    if((k<=w->ellcut_y[1]) && (k>=w->ellcut_y[0]))
      y_out_range[ii]=1;
  }

#pragma omp parallel default(none)				\
  shared(fl1,fl2,fs,cmask1,cmask2,w,i_ring,i_band,i_band_nocut)	\
  shared(cosarr,sinarr,kmodarr,beamprod,maskprod)		\
  shared(x_out_range,y_out_range)
  {
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;
    int iy1,ix1;
    int *n_cells_thr=my_calloc(w->bin->n_bands,sizeof(int));
    gsl_interp_accel *intacc_beam=gsl_interp_accel_alloc();

#pragma omp for
    for(iy1=0;iy1<fs->ny;iy1++) {
      flouble ky;
      int ik=0;
      if(2*iy1<=fs->ny)
	ky=iy1*dky;
      else
	ky=-(fs->ny-iy1)*dky;
      for(ix1=0;ix1<fs->nx;ix1++) {
	flouble kx,kmod,beam1,beam2;
	int ix_here,index_here,index;
	index=ix1+fs->nx*iy1;
	if(2*ix1<=fs->nx) {
	  kx=ix1*dkx;
	  ix_here=ix1;
	}
	else {
	  kx=-(fs->nx-ix1)*dkx;
	  ix_here=fs->nx-ix1;
	}
	index_here=ix_here+(fs->nx/2+1)*iy1;
	
	maskprod[index]=(creal(cmask1[index_here])*creal(cmask2[index_here])+
			 cimag(cmask1[index_here])*cimag(cmask2[index_here]));

	kmod=sqrt(kx*kx+ky*ky);
	beam1=nmt_k_function_eval(fl1->beam,kmod,intacc_beam);
	beam2=nmt_k_function_eval(fl2->beam,kmod,intacc_beam);	
	kmodarr[index]=kmod;
	beamprod[index]=beam1*beam2;
	ik=nmt_bins_flat_search_fast(w->bin,kmod,ik);
	
	if(y_out_range[iy1] || x_out_range[ix1])
	  i_band[index]=-1;
	else {
	  if(ik>=0) {
	    i_band[index]=ik;
	    n_cells_thr[ik]++;
	  }
	  else
	    i_band[index]=-1;
	}
	i_band_nocut[index]=ik;
	i_ring[index]=(int)(kmod*w->fs->i_dell);
	if((i_ring[index]<0) || (i_ring[index]>=w->fs->n_ell))
	  i_ring[index]=-1;
	if(w->ncls>1) {
	  flouble c,s;
	  if(kmod>0) {
	    c=kx/kmod;
	    s=ky/kmod;
	  }
	  else {
	    c=1.;
	    s=0.;
	  }
	  cosarr[index]=c*c-s*s;
	  sinarr[index]=2*s*c;
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(iy1=0;iy1<w->bin->n_bands;iy1++)
	w->n_cells[iy1]+=n_cells_thr[iy1];
    } //end omp critical
    free(n_cells_thr);
    gsl_interp_accel_free(intacc_beam);
  } //end omp parallel
  free(x_out_range);
  free(y_out_range);

#pragma omp parallel default(none)			\
  shared(fs,i_ring,i_band,i_band_nocut,w)		\
  shared(cosarr,sinarr,kmodarr,maskprod,beamprod)
  {
    int iy1,ix1,ix2,iy2;
    int pe1=w->pe1,pe2=w->pe2,pb1=w->pb1,pb2=w->pb2;
    int pure_any=pe1 || pb1 || pe2 || pb2;
    flouble **coup_unbinned_thr,**coup_binned_thr;
    coup_unbinned_thr=my_malloc(w->bin->n_bands*w->ncls*sizeof(flouble *));
    for(iy1=0;iy1<w->bin->n_bands*w->ncls;iy1++)
      coup_unbinned_thr[iy1]=my_calloc(w->fs->n_ell*w->ncls,sizeof(flouble));
    coup_binned_thr=my_malloc(w->bin->n_bands*w->ncls*sizeof(flouble *));
    for(iy1=0;iy1<w->bin->n_bands*w->ncls;iy1++)
      coup_binned_thr[iy1]=my_calloc(w->bin->n_bands*w->ncls,sizeof(flouble));

#pragma omp for
    for(iy1=0;iy1<fs->ny;iy1++) {
      for(ix1=0;ix1<fs->nx;ix1++) {
	int index1=ix1+fs->nx*iy1;
	int ik1=i_band[index1];
	if(ik1>=0) {
	  flouble inv_k1=0;
	  ik1*=w->ncls;
	  if((index1>0) && (w->ncls>1))
	    inv_k1=1./kmodarr[index1];
	  for(iy2=0;iy2<fs->ny;iy2++) {
	    for(ix2=0;ix2<fs->nx;ix2++) {
	      int index2=ix2+fs->nx*iy2;
	      int ir2=i_ring[index2];
	      int ik2=i_band_nocut[index2];
	      flouble cdiff=1,sdiff=0,kr=1,mp;
	      int index;
	      int iy=iy1-iy2;
	      int ix=ix1-ix2;
	      if(iy<0) iy+=fs->ny;
	      if(ix<0) ix+=fs->nx;
	      ik2*=w->ncls;
	      ir2*=w->ncls;
	      index=ix+fs->nx*iy;
	      
	      if(w->ncls>1) {
		cdiff=cosarr[index1]*cosarr[index2]+sinarr[index1]*sinarr[index2];
		sdiff=sinarr[index1]*cosarr[index2]-cosarr[index1]*sinarr[index2];
		if((index1==0) && (index2==0))
		  kr=1;
		else
		  kr=kmodarr[index2]*inv_k1;
		kr*=kr;
	      }
	      mp=maskprod[index]*beamprod[index2];
	      
	      if(w->ncls==1) {
		if(ir2>=0)
		  coup_unbinned_thr[ik1+0][ir2+0]+=mp;
		if(ik2>=0)
		  coup_binned_thr[ik1+0][ik2+0]+=mp;
	      }
	      else if(w->ncls==2) {
		flouble fc[2],fs[2];
		fc[0]=cdiff*mp;
		fs[0]=sdiff*mp;
		if(pure_any) {
		  fc[1]=kr*mp; fs[1]=0;
		}
		if(ir2>=0) {
		  coup_unbinned_thr[ik1+0][ir2+0]+=fc[pe1+pe2]; //TE,TE
		  coup_unbinned_thr[ik1+0][ir2+1]-=fs[pe1+pe2]; //TE,TB
		  coup_unbinned_thr[ik1+1][ir2+0]+=fs[pb1+pb2]; //TB,TE
		  coup_unbinned_thr[ik1+1][ir2+1]+=fc[pb1+pb2]; //TB,TB
		}
		if(ik2>=0) {
		  coup_binned_thr[ik1+0][ik2+0]+=fc[pe1+pe2]; //TE,TE
		  coup_binned_thr[ik1+0][ik2+1]-=fs[pe1+pe2]; //TE,TB
		  coup_binned_thr[ik1+1][ik2+0]+=fs[pb1+pb2]; //TB,TE
		  coup_binned_thr[ik1+1][ik2+1]+=fc[pb1+pb2]; //TB,TB
		}
	      }
	      else if(w->ncls==4) {
		flouble fc[2],fs[2];
		fc[0]=cdiff; fs[0]=sdiff;
		if(pure_any) {
		  fc[1]=kr; fs[1]=0;
		}
		if(ir2>=0) {
		  coup_unbinned_thr[ik1+0][ir2+0]+=fc[pe1]*fc[pe2]*mp; //EE,EE
		  coup_unbinned_thr[ik1+0][ir2+1]-=fc[pe1]*fs[pe2]*mp; //EE,EB
		  coup_unbinned_thr[ik1+0][ir2+2]-=fs[pe1]*fc[pe2]*mp; //EE,BE
		  coup_unbinned_thr[ik1+0][ir2+3]+=fs[pe1]*fs[pe2]*mp; //EE,BB
		  coup_unbinned_thr[ik1+1][ir2+0]+=fc[pe1]*fs[pb2]*mp; //EB,EE
		  coup_unbinned_thr[ik1+1][ir2+1]+=fc[pe1]*fc[pb2]*mp; //EB,EB
		  coup_unbinned_thr[ik1+1][ir2+2]-=fs[pe1]*fs[pb2]*mp; //EB,BE
		  coup_unbinned_thr[ik1+1][ir2+3]-=fs[pe1]*fc[pb2]*mp; //EB,BB
		  coup_unbinned_thr[ik1+2][ir2+0]+=fs[pb1]*fc[pe2]*mp; //BE,EE
		  coup_unbinned_thr[ik1+2][ir2+1]-=fs[pb1]*fs[pe2]*mp; //BE,EB
		  coup_unbinned_thr[ik1+2][ir2+2]+=fc[pb1]*fc[pe2]*mp; //BE,BE
		  coup_unbinned_thr[ik1+2][ir2+3]-=fc[pb1]*fs[pe2]*mp; //BE,BB
		  coup_unbinned_thr[ik1+3][ir2+0]+=fs[pb1]*fs[pb2]*mp; //BB,EE
		  coup_unbinned_thr[ik1+3][ir2+1]+=fs[pb1]*fc[pb2]*mp; //BB,EB
		  coup_unbinned_thr[ik1+3][ir2+2]+=fc[pb1]*fs[pb2]*mp; //BB,BE
		  coup_unbinned_thr[ik1+3][ir2+3]+=fc[pb1]*fc[pb2]*mp; //BB,BB
		}
		if(ik2>=0) {
		  coup_binned_thr[ik1+0][ik2+0]+=fc[pe1]*fc[pe2]*mp; //EE,EE
		  coup_binned_thr[ik1+0][ik2+1]-=fc[pe1]*fs[pe2]*mp; //EE,EB
		  coup_binned_thr[ik1+0][ik2+2]-=fs[pe1]*fc[pe2]*mp; //EE,BE
		  coup_binned_thr[ik1+0][ik2+3]+=fs[pe1]*fs[pe2]*mp; //EE,BB
		  coup_binned_thr[ik1+1][ik2+0]+=fc[pe1]*fs[pb2]*mp; //EB,EE
		  coup_binned_thr[ik1+1][ik2+1]+=fc[pe1]*fc[pb2]*mp; //EB,EB
		  coup_binned_thr[ik1+1][ik2+2]-=fs[pe1]*fs[pb2]*mp; //EB,BE
		  coup_binned_thr[ik1+1][ik2+3]-=fs[pe1]*fc[pb2]*mp; //EB,BB
		  coup_binned_thr[ik1+2][ik2+0]+=fs[pb1]*fc[pe2]*mp; //BE,EE
		  coup_binned_thr[ik1+2][ik2+1]-=fs[pb1]*fs[pe2]*mp; //BE,EB
		  coup_binned_thr[ik1+2][ik2+2]+=fc[pb1]*fc[pe2]*mp; //BE,BE
		  coup_binned_thr[ik1+2][ik2+3]-=fc[pb1]*fs[pe2]*mp; //BE,BB
		  coup_binned_thr[ik1+3][ik2+0]+=fs[pb1]*fs[pb2]*mp; //BB,EE
		  coup_binned_thr[ik1+3][ik2+1]+=fs[pb1]*fc[pb2]*mp; //BB,EB
		  coup_binned_thr[ik1+3][ik2+2]+=fc[pb1]*fs[pb2]*mp; //BB,BE
		  coup_binned_thr[ik1+3][ik2+3]+=fc[pb1]*fc[pb2]*mp; //BB,BB
		}
	      }
	      else if(w->ncls==7) {
		flouble fc[2],fs[2];
		fc[0]=cdiff; fs[0]=sdiff;
		if(pure_any) {
		  fc[1]=kr; fs[1]=0;
		}
		if(ir2>=0) {
		  coup_unbinned_thr[ik1+0][ir2+0]+=mp; //TT,TT
		  coup_unbinned_thr[ik1+1][ir2+1]+=fc[pe1+pe2]*mp; //TE,TE
		  coup_unbinned_thr[ik1+1][ir2+2]-=fs[pe1+pe2]*mp; //TE,TB
		  coup_unbinned_thr[ik1+2][ir2+1]+=fs[pb1+pb2]*mp; //TB,TE
		  coup_unbinned_thr[ik1+2][ir2+2]+=fc[pb1+pb2]*mp; //TB,TB
		  coup_unbinned_thr[ik1+3][ir2+3]+=fc[pe2]*fc[pe2]*mp; //EE,EE
		  coup_unbinned_thr[ik1+3][ir2+4]-=fc[pe2]*fs[pe2]*mp; //EE,EB
		  coup_unbinned_thr[ik1+3][ir2+5]-=fs[pe2]*fc[pe2]*mp; //EE,BE
		  coup_unbinned_thr[ik1+3][ir2+6]+=fs[pe2]*fs[pe2]*mp; //EE,BB
		  coup_unbinned_thr[ik1+4][ir2+3]+=fc[pe2]*fs[pb2]*mp; //EB,EE
		  coup_unbinned_thr[ik1+4][ir2+4]+=fc[pe2]*fc[pb2]*mp; //EB,EB
		  coup_unbinned_thr[ik1+4][ir2+5]-=fs[pe2]*fs[pb2]*mp; //EB,BE
		  coup_unbinned_thr[ik1+4][ir2+6]-=fs[pe2]*fc[pb2]*mp; //EB,BB
		  coup_unbinned_thr[ik1+5][ir2+3]+=fs[pb2]*fc[pe2]*mp; //BE,EE
		  coup_unbinned_thr[ik1+5][ir2+4]-=fs[pb2]*fs[pe2]*mp; //BE,EB
		  coup_unbinned_thr[ik1+5][ir2+5]+=fc[pb2]*fc[pe2]*mp; //BE,BE
		  coup_unbinned_thr[ik1+5][ir2+6]-=fc[pb2]*fs[pe2]*mp; //BE,BB
		  coup_unbinned_thr[ik1+6][ir2+3]+=fs[pb2]*fs[pb2]*mp; //BB,EE
		  coup_unbinned_thr[ik1+6][ir2+4]+=fs[pb2]*fc[pb2]*mp; //BB,EB
		  coup_unbinned_thr[ik1+6][ir2+5]+=fc[pb2]*fs[pb2]*mp; //BB,BE
		  coup_unbinned_thr[ik1+6][ir2+6]+=fc[pb2]*fc[pb2]*mp; //BB,BB
		}
		if(ik2>=0) {
		  coup_binned_thr[ik1+0][ik2+0]+=mp; //TT,TT
		  coup_binned_thr[ik1+1][ik2+1]+=fc[pe1+pe2]*mp; //TE,TE
		  coup_binned_thr[ik1+1][ik2+2]-=fs[pe1+pe2]*mp; //TE,TB
		  coup_binned_thr[ik1+2][ik2+1]+=fs[pb1+pb2]*mp; //TB,TE
		  coup_binned_thr[ik1+2][ik2+2]+=fc[pb1+pb2]*mp; //TB,TB
		  coup_binned_thr[ik1+3][ik2+3]+=fc[pe2]*fc[pe2]*mp; //EE,EE
		  coup_binned_thr[ik1+3][ik2+4]-=fc[pe2]*fs[pe2]*mp; //EE,EB
		  coup_binned_thr[ik1+3][ik2+5]-=fs[pe2]*fc[pe2]*mp; //EE,BE
		  coup_binned_thr[ik1+3][ik2+6]+=fs[pe2]*fs[pe2]*mp; //EE,BB
		  coup_binned_thr[ik1+4][ik2+3]+=fc[pe2]*fs[pb2]*mp; //EB,EE
		  coup_binned_thr[ik1+4][ik2+4]+=fc[pe2]*fc[pb2]*mp; //EB,EB
		  coup_binned_thr[ik1+4][ik2+5]-=fs[pe2]*fs[pb2]*mp; //EB,BE
		  coup_binned_thr[ik1+4][ik2+6]-=fs[pe2]*fc[pb2]*mp; //EB,BB
		  coup_binned_thr[ik1+5][ik2+3]+=fs[pb2]*fc[pe2]*mp; //BE,EE
		  coup_binned_thr[ik1+5][ik2+4]-=fs[pb2]*fs[pe2]*mp; //BE,EB
		  coup_binned_thr[ik1+5][ik2+5]+=fc[pb2]*fc[pe2]*mp; //BE,BE
		  coup_binned_thr[ik1+5][ik2+6]-=fc[pb2]*fs[pe2]*mp; //BE,BB
		  coup_binned_thr[ik1+6][ik2+3]+=fs[pb2]*fs[pb2]*mp; //BB,EE
		  coup_binned_thr[ik1+6][ik2+4]+=fs[pb2]*fc[pb2]*mp; //BB,EB
		  coup_binned_thr[ik1+6][ik2+5]+=fc[pb2]*fs[pb2]*mp; //BB,BE
		  coup_binned_thr[ik1+6][ik2+6]+=fc[pb2]*fc[pb2]*mp; //BB,BB
		}
	      }		
	    }
	  }
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(iy1=0;iy1<w->ncls*w->bin->n_bands;iy1++) {
	for(iy2=0;iy2<w->ncls*w->bin->n_bands;iy2++)
	  w->coupling_matrix_binned[iy1][iy2]+=coup_binned_thr[iy1][iy2];
	for(iy2=0;iy2<w->ncls*w->fs->n_ell;iy2++)
	  w->coupling_matrix_unbinned[iy1][iy2]+=coup_unbinned_thr[iy1][iy2];
      }
    } //end omp critical

    for(iy1=0;iy1<w->bin->n_bands*w->ncls;iy1++) {
      free(coup_unbinned_thr[iy1]);
      free(coup_binned_thr[iy1]);
    }
    free(coup_unbinned_thr);
    free(coup_binned_thr);
  } //end omp parallel

#pragma omp parallel default(none) \
  shared(w,fs)
  {
    int il1;
    flouble fac_norm=4*M_PI*M_PI/(fs->lx*fs->lx*fs->ly*fs->ly);

#pragma omp for
    for(il1=0;il1<w->bin->n_bands;il1++) {
      int icl1;
      flouble norm;
      if(w->n_cells[il1]>0)
	norm=fac_norm/w->n_cells[il1];
      else
	norm=0;
      for(icl1=0;icl1<w->ncls;icl1++) {
	int il2;
	for(il2=0;il2<w->fs->n_ell;il2++) {
	  int icl2;
	  for(icl2=0;icl2<w->ncls;icl2++)
	    w->coupling_matrix_unbinned[w->ncls*il1+icl1][w->ncls*il2+icl2]*=norm;
	}
	for(il2=0;il2<w->bin->n_bands;il2++) {
	  int icl2;
	  for(icl2=0;icl2<w->ncls;icl2++)
	    w->coupling_matrix_binned[w->ncls*il1+icl1][w->ncls*il2+icl2]*=norm;
	}
      }
    } //end omp for
  } //end omp parallel

  int icl_a,icl_b,ib2,ib3,sig;
  for(icl_a=0;icl_a<w->ncls;icl_a++) {
    for(icl_b=0;icl_b<w->ncls;icl_b++) {
      for(ib2=0;ib2<w->bin->n_bands;ib2++) {
	for(ib3=0;ib3<w->bin->n_bands;ib3++) {
	  gsl_matrix_set(w->coupling_matrix_binned_gsl,w->ncls*ib2+icl_a,w->ncls*ib3+icl_b,
			 w->coupling_matrix_binned[w->ncls*ib2+icl_a][w->ncls*ib3+icl_b]);
	}
      }
    }
  }
  gsl_linalg_LU_decomp(w->coupling_matrix_binned_gsl,w->coupling_matrix_perm,&sig);

  dftw_free(cmask1);
  if(fl1!=fl2)
    dftw_free(cmask2);
  free(i_ring);
  free(i_band);
  free(i_band_nocut);
  dftw_free(kmodarr);
  dftw_free(beamprod);
#ifndef _ENABLE_FLAT_THEORY_ACCURATE
  free(maskprod);
#endif //_ENABLE_FLAT_THEORY_ACCURATE
  if(w->ncls>1) {
    dftw_free(cosarr);
    dftw_free(sinarr);
  }

  return w;
}

void nmt_compute_deprojection_bias_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
					nmt_binning_scheme_flat *bin,
					flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y,
					int nl_prop,flouble *l_prop,flouble **cl_proposal,
					flouble **cl_bias)
{
  //Placeholder
  int ii;
  long ip;
  int nspec=fl1->nmaps*fl2->nmaps;
  flouble **cl_dum=my_malloc(nspec*sizeof(flouble *));
  nmt_k_function **cl_proposal_f=my_malloc(nspec*sizeof(nmt_k_function *));
  for(ii=0;ii<nspec;ii++) {
    cl_dum[ii]=my_calloc(bin->n_bands,sizeof(flouble));
    cl_proposal_f[ii]=nmt_k_function_alloc(nl_prop,l_prop,cl_proposal[ii],cl_proposal[ii][0],0,0);
    for(ip=0;ip<bin->n_bands;ip++)
      cl_bias[ii][ip]=0;
  }

  if(check_flatsky_infos(fl1->fs,fl2->fs))
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can only correlate fields defined on the same pixels!\n");

  //TODO: some terms (e.g. C^ab*SHT[w*g^j]) could be precomputed
  //TODO: if fl1=fl2 F2=F3
  //Allocate dummy maps and alms
  flouble **map_1_dum=my_malloc(fl1->nmaps*sizeof(flouble *));
  fcomplex **alm_1_dum=my_malloc(fl1->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl1->nmaps;ii++) {
    map_1_dum[ii]=dftw_malloc(fl1->npix*sizeof(flouble));
    alm_1_dum[ii]=dftw_malloc(fl1->fs->ny*(fl1->fs->nx/2+1)*sizeof(fcomplex));
  }
  flouble **map_2_dum=my_malloc(fl2->nmaps*sizeof(flouble *));
  fcomplex **alm_2_dum=my_malloc(fl2->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl2->nmaps;ii++) {
    map_2_dum[ii]=dftw_malloc(fl2->npix*sizeof(flouble));
    alm_2_dum[ii]=dftw_malloc(fl2->fs->ny*(fl2->fs->nx/2+1)*sizeof(fcomplex));
  }

  if(fl2->ntemp>0) {
    int iti;
    for(iti=0;iti<fl2->ntemp;iti++) {
      int itj;
      for(itj=0;itj<fl2->ntemp;itj++) {
	int im1,im2;
	double nij=gsl_matrix_get(fl2->matrix_M,iti,itj);
	//w*g^j
	for(im2=0;im2<fl2->nmaps;im2++)
	  fs_map_product(fl2->fs,fl2->temp[itj][im2],fl2->mask,map_2_dum[im2]);
	//DFT[w*g^j]
	fs_map2alm(fl2->fs,1,2*fl2->pol,map_2_dum,alm_2_dum);
	//C^ab*DFT[w*g^j]
	for(im1=0;im1<fl1->nmaps;im1++) {
	  fs_zero_alm(fl1->fs,alm_1_dum[im1]);
	  for(im2=0;im2<fl2->nmaps;im2++)
	    fs_alter_alm(fl2->fs,-1.,alm_2_dum[im2],alm_1_dum[im1],cl_proposal_f[im1*fl2->nmaps+im2],1);
	}
	//DFT^-1[C^ab*DFT[w*g^j]]
	fs_alm2map(fl1->fs,1,2*fl1->pol,map_1_dum,alm_1_dum);
	//DFT[v*DFT^-1[C^ab*DFT[w*g^j]]]
	purify_generic_flat(fl1,fl1->mask,fl1->a_mask,map_1_dum,alm_1_dum);
	//Sum_m(DFT[v*DFT^-1[C^ab*DFT[w*g^j]]]*g^i*)/(2l+1)
	fs_alm2cl(fl1->fs,bin,alm_1_dum,fl2->a_temp[iti],fl1->pol,fl2->pol,cl_dum,
		  lmn_x,lmx_x,lmn_y,lmx_y);
	for(im1=0;im1<nspec;im1++) {
	  for(ip=0;ip<bin->n_bands;ip++)
	    cl_bias[im1][ip]-=cl_dum[im1][ip]*nij;
	}
      }
    }
  }

  if(fl1->ntemp>0) {
    int iti;
    for(iti=0;iti<fl1->ntemp;iti++) {
      int itj;
      for(itj=0;itj<fl1->ntemp;itj++) {
	int im1,im2;
	double mij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	//v*f^j
	for(im1=0;im1<fl1->nmaps;im1++)
	  fs_map_product(fl1->fs,fl1->temp[itj][im1],fl1->mask,map_1_dum[im1]);
	//DFT[v*f^j]
	fs_map2alm(fl1->fs,1,2*fl1->pol,map_1_dum,alm_1_dum);
	//C^abT*DFT[v*f^j]
	for(im2=0;im2<fl2->nmaps;im2++) {
	  fs_zero_alm(fl2->fs,alm_2_dum[im2]);
	  for(im1=0;im1<fl1->nmaps;im1++)
	    fs_alter_alm(fl1->fs,-1.,alm_1_dum[im1],alm_2_dum[im2],cl_proposal_f[im1*fl2->nmaps+im2],1);
	}
	//DFT^-1[C^abT*DFT[v*f^j]]
	fs_alm2map(fl2->fs,1,2*fl2->pol,map_2_dum,alm_2_dum);
	//DFT[w*DFT^-1[C^abT*DFT[v*f^j]]]
	purify_generic_flat(fl2,fl2->mask,fl2->a_mask,map_2_dum,alm_2_dum);
	//Sum_m(f^i*DFT[w*DFT^-1[C^abT*DFT[v*f^j]]]^*)/(2l+1)
	fs_alm2cl(fl1->fs,bin,fl1->a_temp[iti],alm_2_dum,fl1->pol,fl2->pol,cl_dum,
		  lmn_x,lmx_x,lmn_y,lmx_y);
	for(im1=0;im1<nspec;im1++) {
	  for(ip=0;ip<bin->n_bands;ip++)
	    cl_bias[im1][ip]-=cl_dum[im1][ip]*mij;
	}
      }
    }
  }

  if((fl1->ntemp>0) && (fl2->ntemp>0)) {
    int iti,itj,itp,itq,im1,im2;
    flouble *mat_prod=my_calloc(fl1->ntemp*fl2->ntemp,sizeof(flouble));
    for(itj=0;itj<fl1->ntemp;itj++) {
      for(itq=0;itq<fl2->ntemp;itq++) {
	//w*g^q
	for(im2=0;im2<fl2->nmaps;im2++)
	  fs_map_product(fl2->fs,fl2->temp[itq][im2],fl2->mask,map_2_dum[im2]);
	//DFT[w*g^q]
	fs_map2alm(fl2->fs,1,2*fl2->pol,map_2_dum,alm_2_dum);
	//C^ab*DFT[w*g^q]
	for(im1=0;im1<fl1->nmaps;im1++) {
	  fs_zero_alm(fl1->fs,alm_1_dum[im1]);
	  for(im2=0;im2<fl2->nmaps;im2++)
	    fs_alter_alm(fl2->fs,-1.,alm_2_dum[im2],alm_1_dum[im1],cl_proposal_f[im1*fl2->nmaps+im2],1);
	}
	//DFT^-1[C^ab*DFT[w*g^q]]
	fs_alm2map(fl1->fs,1,2*fl1->pol,map_1_dum,alm_1_dum);
	for(im1=0;im1<fl1->nmaps;im1++) {
	  //v*DFT^-1[C^ab*DFT[w*g^q]]
	  fs_map_product(fl1->fs,map_1_dum[im1],fl1->mask,map_1_dum[im1]);
	  //Int[f^jT*v*DFT^-1[C^ab*DFT[w*g^q]]]
	  mat_prod[itj*fl2->ntemp+itq]+=fs_map_dot(fl1->fs,map_1_dum[im1],fl1->temp[itj][im1]);
	}
      }
    }

    for(iti=0;iti<fl1->ntemp;iti++) {
      for(itp=0;itp<fl2->ntemp;itp++) {
	//Sum_m(f^i*g^p*)/(2l+1)
	fs_alm2cl(fl1->fs,bin,fl1->a_temp[iti],fl2->a_temp[itp],fl1->pol,fl2->pol,cl_dum,
		  lmn_x,lmx_x,lmn_y,lmx_y);
	for(itj=0;itj<fl1->ntemp;itj++) {
	  double mij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	  for(itq=0;itq<fl2->ntemp;itq++) {
	    double npq=gsl_matrix_get(fl2->matrix_M,itp,itq);
	    for(im1=0;im1<nspec;im1++) {
	      for(ip=0;ip<bin->n_bands;ip++)
		cl_bias[im1][ip]+=cl_dum[im1][ip]*mat_prod[itj*fl2->ntemp+itq]*mij*npq;
	    }
	  }
	}
      }
    }

    free(mat_prod);
  }

  for(ii=0;ii<fl1->nmaps;ii++) {
    dftw_free(map_1_dum[ii]);
    dftw_free(alm_1_dum[ii]);
  }
  free(map_1_dum);
  free(alm_1_dum);
  for(ii=0;ii<fl2->nmaps;ii++) {
    dftw_free(map_2_dum[ii]);
    dftw_free(alm_2_dum[ii]);
  }
  free(map_2_dum);
  free(alm_2_dum);
  for(ii=0;ii<nspec;ii++) {
    free(cl_dum[ii]);
    nmt_k_function_free(cl_proposal_f[ii]);
  }
  free(cl_proposal_f);
  free(cl_dum);

  return;
}

#ifdef _ENABLE_FLAT_THEORY_ACCURATE
void nmt_couple_cl_l_flat_accurate(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out)
{
  //Zero input array
  int ii;
  for(ii=0;ii<w->ncls;ii++) {
    int jj;
    for(jj=0;jj<w->bin->n_bands;jj++)
      cl_out[ii][jj]=0;
  }

  //Precompute angles, mode lengths and band indices
  flouble *cosarr,*sinarr,*kmodarr;
  int *i_band=my_malloc(w->fs->npix*sizeof(int));
  flouble *clmaps=my_malloc(w->ncls*w->fs->npix*sizeof(flouble));
  nmt_k_function **fcl=my_malloc(w->ncls*sizeof(nmt_k_function *));
  for(ii=0;ii<w->ncls;ii++)
    fcl[ii]=nmt_k_function_alloc(nl,larr,cl_in[ii],cl_in[ii][0],0.,0);
  if(w->ncls>1) {
    kmodarr=dftw_malloc(w->fs->npix*sizeof(flouble));
    cosarr=dftw_malloc(w->fs->npix*sizeof(flouble));
    sinarr=dftw_malloc(w->fs->npix*sizeof(flouble));
  }

  int *x_out_range,*y_out_range;
  x_out_range=my_calloc(w->fs->nx,sizeof(int));
  y_out_range=my_calloc(w->fs->ny,sizeof(int));
  for(ii=0;ii<w->fs->nx;ii++) {
    flouble k;
    if(2*ii<=w->fs->nx) k=ii*2*M_PI/w->fs->lx;
    else k=-(w->fs->nx-ii)*2*M_PI/w->fs->lx;
    if((k<=w->ellcut_x[1]) && (k>=w->ellcut_x[0]))
      x_out_range[ii]=1;
  }
  for(ii=0;ii<w->fs->ny;ii++) {
    flouble k;
    if(2*ii<=w->fs->ny) k=ii*2*M_PI/w->fs->ly;
    else k=-(w->fs->ny-ii)*2*M_PI/w->fs->ly;
    if((k<=w->ellcut_y[1]) && (k>=w->ellcut_y[0]))
      y_out_range[ii]=1;
  }

#pragma omp parallel default(none)		\
  shared(w,i_band,cosarr,sinarr,kmodarr)	\
  shared(x_out_range,y_out_range,fcl,clmaps)
  {
    flouble dkx=2*M_PI/w->fs->lx;
    flouble dky=2*M_PI/w->fs->ly;
    int iy1,ix1;
    gsl_interp_accel *intacc=gsl_interp_accel_alloc();

#pragma omp for
    for(iy1=0;iy1<w->fs->ny;iy1++) {
      flouble ky;
      int ik=0;
      if(2*iy1<=w->fs->ny)
	ky=iy1*dky;
      else
	ky=-(w->fs->ny-iy1)*dky;
      for(ix1=0;ix1<w->fs->nx;ix1++) {
	flouble kx,kmod;
	int ix_here,index,ic;
	index=ix1+w->fs->nx*iy1;
	if(2*ix1<=w->fs->nx) {
	  kx=ix1*dkx;
	  ix_here=ix1;
	}
	else {
	  kx=-(w->fs->nx-ix1)*dkx;
	  ix_here=w->fs->nx-ix1;
	}

	kmod=sqrt(kx*kx+ky*ky);
	for(ic=0;ic<w->ncls;ic++)
	  clmaps[ic+index*w->ncls]=nmt_k_function_eval(fcl[ic],kmod,intacc);

	if(y_out_range[iy1] || x_out_range[ix1])
	  i_band[index]=-1;
	else {
	  int ic;
	  ik=nmt_bins_flat_search_fast(w->bin,kmod,ik);
	  if(ik>=0)
	    i_band[index]=ik;
	  else
	    i_band[index]=-1;
	  if(w->ncls>1) {
	    flouble c,s;
	    if(kmod>0) {
	      c=kx/kmod;
	      s=ky/kmod;
	    }
	    else {
	      c=1.;
	      s=0.;
	    }
	    kmodarr[index]=kmod;
	    cosarr[index]=c*c-s*s;
	    sinarr[index]=2*s*c;
	  }	
	}
      }
    } //end omp for
    gsl_interp_accel_free(intacc);
  } //end omp parallel
  free(x_out_range);
  free(y_out_range);
  for(ii=0;ii<w->ncls;ii++)
    nmt_k_function_free(fcl[ii]);
  free(fcl);

#pragma omp parallel default(none)			\
  shared(i_band,w,cosarr,sinarr,kmodarr,clmaps,cl_out)
  {
    int iy1,ix1,ix2,iy2;
    int pe1=w->pe1,pe2=w->pe2,pb1=w->pb1,pb2=w->pb2;
    int pure_any=pe1 || pb1 || pe2 || pb2;
    flouble **cl_out_thr=my_malloc(w->ncls*sizeof(flouble *));
    for(iy1=0;iy1<w->ncls;iy1++)
      cl_out_thr[iy1]=my_calloc(w->bin->n_bands,sizeof(flouble));

#pragma omp for
    for(iy1=0;iy1<w->fs->ny;iy1++) {
      for(ix1=0;ix1<w->fs->nx;ix1++) {
	int index1=ix1+w->fs->nx*iy1;
	int ik1=i_band[index1];
	if(ik1>=0) {
	  flouble inv_k1=0;
	  if((index1>0) && (w->ncls>1))
	    inv_k1=1./kmodarr[index1];
	  for(iy2=0;iy2<w->fs->ny;iy2++) {
	    for(ix2=0;ix2<w->fs->nx;ix2++) {
	      int index;
	      flouble mp,cdiff=1,sdiff=0,kr=1;
	      int index2=ix2+w->fs->nx*iy2;
	      flouble *cls_in=&(clmaps[w->ncls*index2]);
	      int iy=iy1-iy2;
	      int ix=ix1-ix2;
	      if(iy<0) iy+=w->fs->ny;
	      if(ix<0) ix+=w->fs->nx;
	      index=ix+w->fs->nx*iy;

	      if(w->ncls>1) {
		cdiff=cosarr[index1]*cosarr[index2]+sinarr[index1]*sinarr[index2];
		sdiff=sinarr[index1]*cosarr[index2]-cosarr[index1]*sinarr[index2];
		if((index1==0) && (index2==0))
		  kr=1;
		else
		  kr=kmodarr[index2]*inv_k1;
		kr*=kr;
	      }
	      mp=w->maskprod[index];

	      if(w->ncls==1) {
		cl_out_thr[0][ik1]+=mp*cls_in[0];
	      }
	      if(w->ncls==2) {
		flouble fc[2],fs[2];
		fc[0]=cdiff*mp;
		fs[0]=sdiff*mp;
		if(pure_any) {
		  fc[1]=kr*mp; fs[1]=0;
		}
		cl_out_thr[0][ik1]+=fc[pe1+pe2]*cls_in[0]; //TE,TE
		cl_out_thr[0][ik1]-=fs[pe1+pe2]*cls_in[1]; //TE,TB
		cl_out_thr[1][ik1]+=fs[pb1+pb2]*cls_in[0]; //TB,TE
		cl_out_thr[1][ik1]+=fc[pb1+pb2]*cls_in[1]; //TB,TB
	      }
	      if(w->ncls==4) {
		flouble fc[2],fs[2];
		fc[0]=cdiff; fs[0]=sdiff;
		if(pure_any) {
		  fc[1]=kr; fs[1]=0;
		}
		cl_out_thr[0][ik1]+=fc[pe1]*fc[pe2]*mp*cls_in[0]; //EE,EE
		cl_out_thr[0][ik1]-=fc[pe1]*fs[pe2]*mp*cls_in[1]; //EE,EB
		cl_out_thr[0][ik1]-=fs[pe1]*fc[pe2]*mp*cls_in[2]; //EE,BE
		cl_out_thr[0][ik1]+=fs[pe1]*fs[pe2]*mp*cls_in[3]; //EE,BB
		cl_out_thr[1][ik1]+=fc[pe1]*fs[pb2]*mp*cls_in[0]; //EB,EE
		cl_out_thr[1][ik1]+=fc[pe1]*fc[pb2]*mp*cls_in[1]; //EB,EB
		cl_out_thr[1][ik1]-=fs[pe1]*fs[pb2]*mp*cls_in[2]; //EB,BE
		cl_out_thr[1][ik1]-=fs[pe1]*fc[pb2]*mp*cls_in[3]; //EB,BB
		cl_out_thr[2][ik1]+=fs[pb1]*fc[pe2]*mp*cls_in[0]; //BE,EE
		cl_out_thr[2][ik1]-=fs[pb1]*fs[pe2]*mp*cls_in[1]; //BE,EB
		cl_out_thr[2][ik1]+=fc[pb1]*fc[pe2]*mp*cls_in[2]; //BE,BE
		cl_out_thr[2][ik1]-=fc[pb1]*fs[pe2]*mp*cls_in[3]; //BE,BB
		cl_out_thr[3][ik1]+=fs[pb1]*fs[pb2]*mp*cls_in[0]; //BB,EE
		cl_out_thr[3][ik1]+=fs[pb1]*fc[pb2]*mp*cls_in[1]; //BB,EB
		cl_out_thr[3][ik1]+=fc[pb1]*fs[pb2]*mp*cls_in[2]; //BB,BE
		cl_out_thr[3][ik1]+=fc[pb1]*fc[pb2]*mp*cls_in[3]; //BB,BB
	      }
	    }
	  }
	}
      }
    } //end omp for
#pragma omp critical
    {
      for(iy1=0;iy1<w->ncls;iy1++) {
	for(iy2=0;iy2<w->bin->n_bands;iy2++)
	  cl_out[iy1][iy2]+=cl_out_thr[iy1][iy2];
      }
    } //end omp critical

    for(iy1=0;iy1<w->ncls;iy1++)
      free(cl_out_thr[iy1]);
    free(cl_out_thr);
  } //end omp parallel

  int il1;
  flouble fac_norm=4*M_PI*M_PI/(w->fs->lx*w->fs->lx*w->fs->ly*w->fs->ly);
  for(il1=0;il1<w->bin->n_bands;il1++) {
    int icl1;
    flouble norm;
    if(w->n_cells[il1]>0)
      norm=fac_norm/w->n_cells[il1];
    else
      norm=0;
    for(icl1=0;icl1<w->ncls;icl1++)
      cl_out[icl1][il1]*=norm;
  }

  free(i_band);
  free(clmaps);
  if(w->ncls>1) {
    dftw_free(kmodarr);
    dftw_free(cosarr);
    dftw_free(sinarr);
  }
}
#endif //_ENABLE_FLAT_THEORY_ACCURATE

void nmt_couple_cl_l_flat_fast(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out)
{
  int ii;
  flouble *cl_in_rings=my_calloc(w->ncls*w->fs->n_ell,sizeof(flouble));
  int *n_cells=my_calloc(w->fs->n_ell,sizeof(int));
  nmt_k_function **fcl=my_malloc(w->ncls*sizeof(nmt_k_function *));

  for(ii=0;ii<w->ncls;ii++)
    fcl[ii]=nmt_k_function_alloc(nl,larr,cl_in[ii],cl_in[ii][0],0.,0);

  //Interpolate input power spectrum onto grid and bin into rings
#pragma omp parallel default(none)		\
  shared(w,fcl,cl_in_rings,n_cells)
  {
    int iy1,ix1;
    flouble dkx=2*M_PI/w->fs->lx;
    flouble dky=2*M_PI/w->fs->ly;
    flouble *cl_in_rings_thr=my_calloc(w->ncls*w->fs->n_ell,sizeof(flouble));
    int *n_cells_thr=my_calloc(w->fs->n_ell,sizeof(int));
    gsl_interp_accel *intacc=gsl_interp_accel_alloc();

#pragma omp for
    for(iy1=0;iy1<w->fs->ny;iy1++) {
      flouble ky;
      if(2*iy1<=w->fs->ny)
	ky=iy1*dky;
      else
	ky=-(w->fs->ny-iy1)*dky;
      for(ix1=0;ix1<w->fs->nx;ix1++) {
	flouble kx,kmod;
	int ir;
	if(2*ix1<=w->fs->nx)
	  kx=ix1*dkx;
	else
	  kx=-(w->fs->nx-ix1)*dkx;
	kmod=sqrt(kx*kx+ky*ky);
	ir=(int)(kmod*w->fs->i_dell);
	if(ir<w->fs->n_ell) {
	  int ic,ind0=ir*w->ncls;
	  n_cells_thr[ir]++;
	  for(ic=0;ic<w->ncls;ic++)
	    cl_in_rings_thr[ind0+ic]+=nmt_k_function_eval(fcl[ic],kmod,intacc);
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(iy1=0;iy1<w->fs->n_ell;iy1++)
	n_cells[iy1]+=n_cells_thr[iy1];
      for(iy1=0;iy1<w->fs->n_ell*w->ncls;iy1++)
	cl_in_rings[iy1]+=cl_in_rings_thr[iy1];
    } //end omp critical
    
    free(cl_in_rings_thr);
    free(n_cells_thr);
    gsl_interp_accel_free(intacc);
  } //end omp parallel

  for(ii=0;ii<w->fs->n_ell;ii++) {
    int ic;
    for(ic=0;ic<w->ncls;ic++) {
      if(n_cells[ii]>0) 
	cl_in_rings[ii*w->ncls+ic]/=n_cells[ii];
    }
  }

  //Convolve with mode-coupling matrix
  for(ii=0;ii<w->ncls;ii++) {
    int i1;
    for(i1=0;i1<w->bin->n_bands;i1++) {
      int ind2,ind1=i1*w->ncls+ii;
      cl_out[ii][i1]=0;
      for(ind2=0;ind2<w->ncls*w->fs->n_ell;ind2++)
	cl_out[ii][i1]+=w->coupling_matrix_unbinned[ind1][ind2]*cl_in_rings[ind2];
    }
  }

  //Free up
  free(cl_in_rings);
  free(n_cells);
  for(ii=0;ii<w->ncls;ii++)
    nmt_k_function_free(fcl[ii]);
  free(fcl);
}

void nmt_couple_cl_l_flat_quick(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out)
{
  int ii;
  flouble **cell_in=my_malloc(w->ncls*sizeof(flouble *));
  gsl_interp_accel *intacc=gsl_interp_accel_alloc();
  for(ii=0;ii<w->ncls;ii++) {
    nmt_k_function *fcl=nmt_k_function_alloc(nl,larr,cl_in[ii],cl_in[ii][0],0.,0);
    cell_in[ii]=my_calloc(w->bin->n_bands,sizeof(flouble));

    int iy;
    flouble dkx=2*M_PI/w->fs->lx;
    flouble dky=2*M_PI/w->fs->ly;
    for(iy=0;iy<w->fs->ny;iy++) {
      flouble ky;
      int ik=0;
      if(2*iy<=w->fs->ny)
	ky=iy*dky;
      else
	ky=-(w->fs->ny-iy)*dky;
      if((ky>w->ellcut_y[1]) || (ky<w->ellcut_y[0])) {
	int ix;
	for(ix=0;ix<w->fs->nx;ix++) {
	  flouble kx;
	  if(2*ix<=w->fs->nx)
	    kx=ix*dkx;
	  else
	    kx=-(w->fs->nx-ix)*dkx;
	  if((kx>w->ellcut_x[1]) || (kx<w->ellcut_x[0])) {
	    double kmod=sqrt(kx*kx+ky*ky);
	    ik=nmt_bins_flat_search_fast(w->bin,kmod,ik);
	    if(ik>=0)
	      cell_in[ii][ik]+=nmt_k_function_eval(fcl,kmod,intacc);
	  }
	}
      }
    }

    for(iy=0;iy<w->bin->n_bands;iy++) {
      if(w->n_cells[iy]>0)
	cell_in[ii][iy]/=w->n_cells[iy];
      else
	cell_in[ii][iy]=0;
    }
    nmt_k_function_free(fcl);
  }
  gsl_interp_accel_free(intacc);

  int icl1;
  for(icl1=0;icl1<w->ncls;icl1++) {
    int i1;
    for(i1=0;i1<w->bin->n_bands;i1++) {
      int icl2;
      int ind1=i1*w->ncls+icl1;
      cl_out[icl1][i1]=0;
      for(icl2=0;icl2<w->ncls;icl2++) {
	int i2;
	for(i2=0;i2<w->bin->n_bands;i2++) {
	  int ind2=i2*w->ncls+icl2;
	  cl_out[icl1][i1]+=w->coupling_matrix_binned[ind1][ind2]*cell_in[icl2][i2];
	}
      }
    }
  }
  
  for(ii=0;ii<w->ncls;ii++)
    free(cell_in[ii]);
  free(cell_in);
}

void nmt_decouple_cl_l_flat(nmt_workspace_flat *w,flouble **cl_in,flouble **cl_noise_in,
			    flouble **cl_bias,flouble **cl_out)
{
  int icl,ib2;
  gsl_vector *dl_map_bad_b=gsl_vector_alloc(w->ncls*w->bin->n_bands);
  gsl_vector *dl_map_good_b=gsl_vector_alloc(w->ncls*w->bin->n_bands);

  //Bin coupled power spectrum
  for(icl=0;icl<w->ncls;icl++) {
    for(ib2=0;ib2<w->bin->n_bands;ib2++) {
      gsl_vector_set(dl_map_bad_b,w->ncls*ib2+icl,
		     cl_in[icl][ib2]-cl_noise_in[icl][ib2]-cl_bias[icl][ib2]);
    }
  }
  
  gsl_linalg_LU_solve(w->coupling_matrix_binned_gsl,w->coupling_matrix_perm,dl_map_bad_b,dl_map_good_b);
  for(icl=0;icl<w->ncls;icl++) {
    for(ib2=0;ib2<w->bin->n_bands;ib2++)
      cl_out[icl][ib2]=gsl_vector_get(dl_map_good_b,w->ncls*ib2+icl);
  }

  gsl_vector_free(dl_map_bad_b);
  gsl_vector_free(dl_map_good_b);
}

void nmt_compute_coupled_cell_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
				   nmt_binning_scheme_flat *bin,flouble **cl_out,
				   flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y)
{
  if(check_flatsky_infos(fl1->fs,fl2->fs))
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can only correlate fields defined on the same pixels!\n");
  fs_alm2cl(fl1->fs,bin,fl1->alms,fl2->alms,fl1->pol,fl2->pol,cl_out,lmn_x,lmx_x,lmn_y,lmx_y);
}

nmt_workspace_flat *nmt_compute_power_spectra_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						   nmt_binning_scheme_flat *bin,
						   flouble lmn_x,flouble lmx_x,
						   flouble lmn_y,flouble lmx_y,
						   nmt_workspace_flat *w0,flouble **cl_noise,
						   int nl_prop,flouble *l_prop,flouble **cl_prop,
						   flouble **cl_out)
{
  int ii;
  flouble **cl_bias,**cl_data;
  nmt_workspace_flat *w;

  if(w0==NULL)
    w=nmt_compute_coupling_matrix_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,0);
  else {
    w=w0;
    if((check_flatsky_infos(fl1->fs,w->fs)) || (check_flatsky_infos(fl2->fs,w->fs)))
      report_error(NMT_ERROR_CONSISTENT_RESO,"Input workspace has different pixels!\n");
    if(bin->n_bands!=w->bin->n_bands)
      report_error(NMT_ERROR_CONSISTENT_RESO,"Input workspace has different bandpowers!\n");
  }

  cl_bias=my_malloc(w->ncls*sizeof(flouble *));
  cl_data=my_malloc(w->ncls*sizeof(flouble *));
  for(ii=0;ii<w->ncls;ii++) {
    cl_bias[ii]=my_calloc(w->bin->n_bands,sizeof(flouble));
    cl_data[ii]=my_calloc(w->bin->n_bands,sizeof(flouble));
  }
  nmt_compute_coupled_cell_flat(fl1,fl2,bin,cl_data,lmn_x,lmx_x,lmn_y,lmx_y);
  nmt_compute_deprojection_bias_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,
				     nl_prop,l_prop,cl_prop,cl_bias);
  nmt_decouple_cl_l_flat(w,cl_data,cl_noise,cl_bias,cl_out);
  for(ii=0;ii<w->ncls;ii++) {
    free(cl_bias[ii]);
    free(cl_data[ii]);
  }
  free(cl_bias);
  free(cl_data);

  return w;
}
