#include "config.h"
#include "utils.h"

nmt_curvedsky_info *nmt_curvedsky_info_copy(nmt_curvedsky_info *cs_in)
{
  nmt_curvedsky_info *cs_out=my_malloc(sizeof(nmt_curvedsky_info));
  cs_out->is_healpix=cs_in->is_healpix;
  cs_out->n_eq=cs_in->n_eq;
  cs_out->lmax_sht=cs_in->lmax_sht;
  cs_out->nx_short=cs_in->nx_short;
  cs_out->nx=cs_in->nx;
  cs_out->ny=cs_in->ny;
  cs_out->npix=cs_in->npix;
  cs_out->Delta_theta=cs_in->Delta_theta;
  cs_out->Delta_phi=cs_in->Delta_phi;
  cs_out->phi0=cs_in->phi0;
  cs_out->theta0=cs_in->theta0;

  return cs_out;
}

nmt_curvedsky_info *nmt_curvedsky_info_alloc(int is_healpix,long nside,
					     int lmax_sht,
					     int nx0,int ny0,flouble Dtheta,flouble Dphi,
					     flouble phi0,flouble theta0)
{
  nmt_curvedsky_info *cs=my_malloc(sizeof(nmt_curvedsky_info));
  cs->is_healpix=is_healpix;
  
  if(cs->is_healpix) {
    long nside_test=1;
    int ispow2=0;
    while((nside_test<=16384) && (!ispow2)) {
      if(nside==nside_test)
	ispow2=1;
      else
	nside_test*=2;
    }
    if(!ispow2)
      report_error(NMT_ERROR_VALUE,"Nside must be a power of 2\n");
    cs->n_eq=nside;
    cs->npix=12*nside*nside;
  }
  else {
    //Basic sanity checks
    if((Dtheta>=M_PI) || (Dphi>=2*M_PI) || (Dtheta<=0) || (Dphi<=0)
       || (fabs(phi0)>2*M_PI) || (theta0<0) || (theta0>M_PI))
      report_error(NMT_ERROR_VALUE,"Wrong pixel sizes or reference coordinates\n");
    if((nx0<=0) || (ny0<=0))
      report_error(NMT_ERROR_VALUE,"Wrong map dimensions\n");

    //Check map fits on the sphere
    if(theta0-(ny0-1)*Dtheta<-0.01*Dtheta)
      report_error(NMT_ERROR_VALUE,"Input map doesn't really fit on the sphere\n");
    long nx=(int)round(2*M_PI / Dphi);
    if(nx0>nx)
      report_error(NMT_ERROR_VALUE,"Seems like you're wrapping the sphere more than once\n");

    //Check pixel size values
    if((fabs(M_PI/Dtheta-round(M_PI/Dtheta))>0.01) || (fabs(2*M_PI/Dphi-round(2*M_PI/Dphi))>0.01))
      report_error(NMT_ERROR_VALUE,"The pixel size must fit in a Clenshaw-Curtis grid\n");
    
    cs->n_eq=fmin(M_PI/Dtheta, M_PI/Dphi) / 2;
    cs->nx_short=nx0;
    cs->nx=nx;
    cs->ny=ny0;
    cs->npix=nx*ny0;
    cs->Delta_phi=Dphi;
    cs->Delta_theta=Dtheta;
    cs->phi0=phi0;
    cs->theta0=theta0;
  }

  // set lmax values, affects calls to he_get_lmax
  if (lmax_sht <= 0)
    cs->lmax_sht=he_get_largest_possible_lmax(cs);
  else
    cs->lmax_sht=lmax_sht;
  
  return cs;
}

flouble *nmt_extend_CAR_map(nmt_curvedsky_info *cs,flouble *map_in)
{
  flouble *map_out=my_calloc(cs->npix,sizeof(flouble));

  if(cs->is_healpix)
    memcpy(map_out,map_in,cs->npix*sizeof(flouble));
  else {
    if(cs->nx!=cs->nx_short) {
      int jj;
      for(jj=0;jj<cs->ny;jj++)
	memcpy(map_out+jj*cs->nx,map_in+jj*cs->nx_short,cs->nx_short*sizeof(flouble));
    }
    else
      memcpy(map_out,map_in,cs->npix*sizeof(flouble));
  }
  
  return map_out;
}

int nmt_diff_curvedsky_info(nmt_curvedsky_info *c1, nmt_curvedsky_info *c2)
{
  if(c1->is_healpix)
    return (c2->is_healpix && (c1->n_eq==c2->n_eq) && (c1->lmax_sht==c2->lmax_sht));
  else
    return ((!(c2->is_healpix)) && (c1->nx_short==c2->nx_short) && (c1->nx==c2->nx) && (c1->ny==c2->ny)
	    && (fabs(c1->phi0-c2->phi0)<1e-6) && (fabs(c1->theta0-c2->theta0)<1e-6)
	    && (fabs(c1->Delta_theta-c2->Delta_theta)<1e-6)
	    && (fabs(c1->Delta_phi-c2->Delta_phi)<1e-6)
	    && (c1->lmax_sht==c2->lmax_sht));
}
  
void nmt_field_free(nmt_field *fl)
{
  int imap,itemp;

  free(fl->cs);
  free(fl->beam);

  for(imap=0;imap<fl->nmaps;imap++) {
    free(fl->maps[imap]);
    free(fl->alms[imap]);
  }
  free(fl->mask);
  if(fl->ntemp>0) {
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++) {
	free(fl->temp[itemp][imap]);
	free(fl->a_temp[itemp][imap]);
      }
    }
  }

  free(fl->alms);
  free(fl->maps);
  if(fl->ntemp>0) {
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      free(fl->temp[itemp]);
      free(fl->a_temp[itemp]);
    }
    free(fl->temp);
    free(fl->a_temp);
    gsl_matrix_free(fl->matrix_M);
  }
  if(fl->a_mask!=NULL) {
    for(imap=0;imap<fl->nmaps;imap++)
      free(fl->a_mask[imap]);
    free(fl->a_mask);
  }
  free(fl);
}

//Takes maps_in and purifies into maps_out, using mask and its SHT walm0.
//SHT of purified map in alms
//Information about what to purify, pixelization etc. in fl
void nmt_purify(nmt_field *fl,flouble *mask,fcomplex **walm0,
		flouble **maps_in,flouble **maps_out,fcomplex **alms,
		int niter)
{
  long ip;
  int imap,mm,ll;
  int purify[2]={0,0};
  flouble *f_l=my_malloc((fl->lmax+1)*sizeof(flouble));
  flouble  **pmap=my_malloc(fl->nmaps*sizeof(flouble *));
  flouble  **wmap=my_malloc(fl->nmaps*sizeof(flouble *));
  fcomplex **walm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **palm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **alm_out=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(imap=0;imap<fl->nmaps;imap++) {
    walm[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    palm[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    pmap[imap]=my_calloc(fl->npix,sizeof(flouble));
    wmap[imap]=my_calloc(fl->npix,sizeof(flouble));
    memcpy(walm[imap],walm0[imap],he_nalms(fl->lmax)*sizeof(fcomplex));
    alm_out[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
  }

  if(fl->pure_e)
    purify[0]=1;
  if(fl->pure_b)
    purify[1]=1;

  //Product with spin-0 mask
  for(imap=0;imap<fl->nmaps;imap++)
    he_map_product(fl->cs,maps_in[imap],mask,pmap[imap]);
  //Compute SHT and store in alm_out
  he_map2alm(fl->cs,fl->lmax,1,2,pmap      ,alm_out,niter);

  //Compute spin-1 mask
  for(ll=0;ll<=fl->lmax;ll++) //The minus sign is because of the definition of E-modes
    f_l[ll]=-sqrt((ll+1.)*ll);
  he_alter_alm(fl->lmax,-1.,walm[0],walm[0],f_l,0); //TODO: There may be a -1 sign here
  he_alm2map(fl->cs,fl->lmax,1,1,wmap,walm);
  //Product with spin-1 mask
  for(ip=0;ip<fl->npix;ip++) {
    pmap[0][ip]=wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip];
    pmap[1][ip]=wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip];
  }
  //Compute SHT, multiply by 2*sqrt((l+1)!(l-2)!/((l-1)!(l+2)!)) and add to alm_out
  he_map2alm(fl->cs,fl->lmax,1,1,pmap       ,palm  ,niter);
  for(ll=0;ll<=fl->lmax;ll++) {
    if(ll>1)
      f_l[ll]=2./sqrt((ll+2.)*(ll-1.));
    else
      f_l[ll]=0;
  }
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
      for(mm=0;mm<=fl->lmax;mm++) {
	for(ll=mm;ll<=fl->lmax;ll++) {
	  long index=he_indexlm(ll,mm,fl->lmax);
	  alm_out[imap][index]+=f_l[ll]*palm[imap][index];
	}
      }
    }
  }

  //Compute spin-2 mask
  for(ll=0;ll<=fl->lmax;ll++) { //The extra minus sign is because of the scalar SHT below (E-mode def for s=0)
    if(ll>1)
      f_l[ll]=-sqrt((ll+2.)*(ll-1.));
    else
      f_l[ll]=0;
  }
  he_alter_alm(fl->lmax,-1.,walm[0],walm[0],f_l,0); //TODO: There may be a -1 sign here
  he_alm2map(fl->cs,fl->lmax,1,2,wmap,walm);
  //Product with spin-2 mask
  for(ip=0;ip<fl->npix;ip++) {
    pmap[0][ip]=wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip];
    pmap[1][ip]=wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip];
  }
  //Compute SHT, multiply by sqrt((l-2)!/(l+2)!) and add to alm_out
  he_map2alm(fl->cs,fl->lmax,2,0,pmap,palm,niter);
  for(ll=0;ll<=fl->lmax;ll++) {
    if(ll>1)
      f_l[ll]=1./sqrt((ll+2.)*(ll+1.)*ll*(ll-1.));
    else
      f_l[ll]=0;
  }
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
      for(mm=0;mm<=fl->lmax;mm++) {
	for(ll=mm;ll<=fl->lmax;ll++) {
	  long index=he_indexlm(ll,mm,fl->lmax);
	  alm_out[imap][index]+=f_l[ll]*palm[imap][index];
	}
      }
    }
  }

  for(imap=0;imap<fl->nmaps;imap++)
    memcpy(alms[imap],alm_out[imap],he_nalms(fl->lmax)*sizeof(fcomplex));
  he_alm2map(fl->cs,fl->lmax,1,2,maps_out,alms);

  for(imap=0;imap<fl->nmaps;imap++) {
    free(pmap[imap]);
    free(wmap[imap]);
    free(palm[imap]);
    free(walm[imap]);
    free(alm_out[imap]);
  }
  free(pmap);
  free(wmap);
  free(palm);
  free(walm);
  free(alm_out);
  free(f_l);
}

nmt_field *nmt_field_alloc_sph(nmt_curvedsky_info *cs,flouble *mask,int pol,flouble **maps,
			       int ntemp,flouble ***temp,flouble *beam,
			       int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv,
			       int niter,int masked_input)
{
  int ii,itemp,itemp2,imap;
  long npix_short;
  nmt_field *fl=my_malloc(sizeof(nmt_field));
  fl->cs=nmt_curvedsky_info_copy(cs);
  fl->lmax=he_get_lmax(fl->cs);
  fl->npix=cs->npix;
  fl->pol=pol;
  if(pol) fl->nmaps=2;
  else fl->nmaps=1;
  fl->ntemp=ntemp;

  if(fl->cs->is_healpix)
    npix_short=fl->cs->npix;
  else
    npix_short=fl->cs->nx_short*fl->cs->ny;
  
  fl->pure_e=0;
  fl->pure_b=0;
  if(pol) {
    if(pure_e)
      fl->pure_e=1;
    if(pure_b)
      fl->pure_b=1;
  }

  fl->beam=my_malloc((fl->lmax+1)*sizeof(flouble));
  if(beam==NULL) {
    for(ii=0;ii<=fl->lmax;ii++)
      fl->beam[ii]=1.;
  }
  else
    memcpy(fl->beam,beam,(fl->lmax+1)*sizeof(flouble));

  fl->mask=nmt_extend_CAR_map(fl->cs,mask);

  fl->maps=my_malloc(fl->nmaps*sizeof(flouble *));
  for(imap=0;imap<fl->nmaps;imap++) {//Need to multiply observed map by mask first
    fl->maps[imap]=nmt_extend_CAR_map(fl->cs,maps[imap]);
    if(masked_input) { //If input maps are already masked, we need to unmask them for purification.
      if(fl->pol && (fl->pure_e || fl->pure_b)) {
	long ip;
	for(ip=0;ip<npix_short;ip++) {
          if(mask[ip]>0)
            maps[imap][ip]/=mask[ip];
        }
      }
    }
    else { //Otherwise, multiply by the mask and store
      he_map_product(fl->cs,fl->maps[imap],fl->mask,fl->maps[imap]);
    }
  }

  if(fl->ntemp>0) {
    fl->temp=my_malloc(fl->ntemp*sizeof(flouble **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->temp[itemp]=my_malloc(fl->nmaps*sizeof(flouble *));
      for(imap=0;imap<fl->nmaps;imap++) {
	fl->temp[itemp][imap]=nmt_extend_CAR_map(fl->cs,temp[itemp][imap]);
        if(masked_input) { //If input maps are already masked, we need to unmask them for purification.
          if(fl->pol && (fl->pure_e || fl->pure_b)) {
            long ip;
            for(ip=0;ip<npix_short;ip++) {
              if(mask[ip]>0)
                temp[itemp][imap][ip]/=mask[ip];
            }
          }
        }
        else { //Otherwise, multiply by the mask and store
          he_map_product(fl->cs,fl->temp[itemp][imap],fl->mask,fl->temp[itemp][imap]);
        }
      }
    }

    //Compute normalization matrix
    fl->matrix_M=gsl_matrix_alloc(fl->ntemp,fl->ntemp);
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(itemp2=itemp;itemp2<fl->ntemp;itemp2++) {
	flouble matrix_element=0;
	for(imap=0;imap<fl->nmaps;imap++)
	  matrix_element+=he_map_dot(fl->cs,fl->temp[itemp][imap],fl->temp[itemp2][imap]);
	gsl_matrix_set(fl->matrix_M,itemp,itemp2,matrix_element);
	if(itemp2!=itemp)
	  gsl_matrix_set(fl->matrix_M,itemp2,itemp,matrix_element);
      }
    }
    moore_penrose_pinv(fl->matrix_M,tol_pinv);

    //Deproject
    flouble *prods=my_calloc(fl->ntemp,sizeof(flouble));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++)
	prods[itemp]+=he_map_dot(fl->cs,fl->temp[itemp][imap],fl->maps[imap]);
    }
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      flouble alpha=0;
      for(itemp2=0;itemp2<fl->ntemp;itemp2++) {
	double mij=gsl_matrix_get(fl->matrix_M,itemp,itemp2);
	alpha+=mij*prods[itemp2];
      }
#ifdef _DEBUG
      printf("alpha_%d = %lE\n",itemp,alpha);
#endif //_DEBUG
      for(imap=0;imap<fl->nmaps;imap++) {
	long ip;
	for(ip=0;ip<fl->npix;ip++)
	  fl->maps[imap][ip]-=alpha*fl->temp[itemp][imap][ip]; //Correct masked field
	if(fl->pol && (fl->pure_e || fl->pure_b)) { //Correct unmasked field too if purifying
	  for(ip=0;ip<npix_short;ip++)
	    maps[imap][ip]-=alpha*temp[itemp][imap][ip]; 
	}
      }
    }
    free(prods);
  }

  //Allocate harmonic coefficients
  fl->alms=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->alms[ii]=my_malloc(he_nalms(fl->lmax)*sizeof(fcomplex));
  if(fl->ntemp>0) {
    fl->a_temp=my_malloc(fl->ntemp*sizeof(fcomplex **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->a_temp[itemp]=my_malloc(fl->nmaps*sizeof(fcomplex *));
      for(imap=0;imap<fl->nmaps;imap++)
	fl->a_temp[itemp][imap]=my_malloc(he_nalms(fl->lmax)*sizeof(fcomplex));
    }
  }

  if(fl->pol && (fl->pure_e || fl->pure_b)) {
    //If purification is needed:
    // 1- Compute mask alms
    // 2- Purify de-contaminated map
    // 3- Compute purified contaminants

    //Compute mask SHT (store in fl->a_mask)
    fl->a_mask=my_malloc(fl->nmaps*sizeof(fcomplex *));
    for(imap=0;imap<fl->nmaps;imap++)
      fl->a_mask[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    he_map2alm(fl->cs,fl->lmax,1,0,&(fl->mask),fl->a_mask,n_iter_mask_purify);

    //Purify map
    flouble **mdum=my_malloc(fl->nmaps*sizeof(flouble *));
    for(imap=0;imap<fl->nmaps;imap++)
      mdum[imap]=nmt_extend_CAR_map(fl->cs,maps[imap]);
    nmt_purify(fl,fl->mask,fl->a_mask,mdum,fl->maps,fl->alms,niter);
    for(imap=0;imap<fl->nmaps;imap++)
      free(mdum[imap]);
    
    //Compute purified contaminant SHTs
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
	for(imap=0;imap<fl->nmaps;imap++)
	  mdum[imap]=nmt_extend_CAR_map(fl->cs,temp[itemp][imap]);
	nmt_purify(fl,fl->mask,fl->a_mask,mdum,fl->temp[itemp],fl->a_temp[itemp],niter);
	for(imap=0;imap<fl->nmaps;imap++) {//Store non-pure map
	  he_map_product(fl->cs,mdum[imap],fl->mask,fl->temp[itemp][imap]);
	  free(mdum[imap]);
	}
      }
      //IMPORTANT: at this stage, fl->maps and fl->alms contain the purified map and SH coefficients
      //           However, although fl->a_temp contains the purified SH coefficients,
      //           fl->temp contains the ***non-purified*** maps. This is to speed up the calculation
      //           of the deprojection bias.
    }
    free(mdum);
  }
  else {
    //If no purification, just multiply by mask and SHT
    fl->a_mask=NULL; //No need to store extra-pure mask harmonic coefficients

    //Masked map and harmonic coefficients
    he_map2alm(fl->cs,fl->lmax,1,2*fl->pol,fl->maps,fl->alms,niter);

    //Compute template harmonic coefficients too
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++)
	he_map2alm(fl->cs,fl->lmax,1,2*fl->pol,fl->temp[itemp],fl->a_temp[itemp],niter);
    }
  }

  return fl;
}

flouble **nmt_synfast_sph(nmt_curvedsky_info *cs,int nfields,int *spin_arr,int lmax,
			  flouble **cells,flouble **beam_fields,int seed)
{
  int ifield,imap;
  int nmaps=0;
  long npix=cs->npix;
  int lmax_cs=he_get_lmax(cs);
  flouble **beam,**maps;
  fcomplex **alms;
  for(ifield=0;ifield<nfields;ifield++) {
    int nmp=1;
    if(spin_arr[ifield]) nmp=2;
    nmaps+=nmp;
  }

  imap=0;
  beam=my_malloc(nmaps*sizeof(flouble *));
  maps=my_malloc(nmaps*sizeof(flouble *));
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    for(imp=0;imp<nmp;imp++) {
      beam[imap+imp]=my_malloc((lmax+1)*sizeof(flouble));
      maps[imap+imp]=my_malloc(npix*sizeof(flouble));
      memcpy(beam[imap+imp],beam_fields[ifield],(lmax+1)*sizeof(flouble));
    }
    imap+=nmp;
  }

  imap=0;
  alms=he_synalm(cs,nmaps,lmax_cs,cells,beam,seed);
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    he_alm2map(cs,lmax_cs,1,spin_arr[ifield],&(maps[imap]),&(alms[imap]));
    for(imp=0;imp<nmp;imp++) {
      free(alms[imap+imp]);
      free(beam[imap+imp]);
    }
    imap+=nmp;
  }
  free(alms);
  free(beam);
  return maps;
}

nmt_field *nmt_field_read(int is_healpix,char *fname_mask,char *fname_maps,char *fname_temp,
			  char *fname_beam,int pol,int pure_e,int pure_b,
			  int n_iter_mask_purify,double tol_pinv,int niter)
{
  nmt_curvedsky_info *cs,*cs_dum;
  nmt_field *fl;
  flouble *beam;
  flouble *mask;
  flouble **maps;
  flouble ***temp;
  int ii,ntemp,itemp,imap,lmax,nmaps=1;
  if(pol) nmaps=2;

  cs=my_malloc(sizeof(nmt_curvedsky_info));
  cs->is_healpix=is_healpix;
  cs_dum=my_malloc(sizeof(nmt_curvedsky_info));
  cs_dum->is_healpix=is_healpix;

  //Read mask and compute nside, lmax etc.
  mask=he_read_map(fname_mask,cs,0);
  lmax=he_get_lmax(cs);

  //Read beam
  if(!strcmp(fname_beam,"none"))
    beam=NULL;
  else {
    FILE *fi=my_fopen(fname_beam,"r");
    int nlines=my_linecount(fi); rewind(fi);
    if(nlines<=lmax)
      report_error(NMT_ERROR_READ,"Beam file must have more than %d rows and two columns\n",lmax);
    beam=my_malloc((lmax+1)*sizeof(flouble));
    for(ii=0;ii<=lmax;ii++) {
      int l;
      double b;
      int stat=fscanf(fi,"%d %lf\n",&l,&b);
      if(stat!=2)
	report_error(NMT_ERROR_READ,"Error reading file %s, line %d\n",fname_beam,ii+1);
      if((l>lmax) || (l<0))
	report_error(NMT_ERROR_READ,"Wrong multipole %d\n",l);
      beam[l]=b;
    }
  }

  //Read data maps
  maps=my_malloc(nmaps*sizeof(flouble *));
  for(ii=0;ii<nmaps;ii++) {
    maps[ii]=he_read_map(fname_maps,cs_dum,ii);
    if(!(nmt_diff_curvedsky_info(cs,cs_dum)))
      report_error(NMT_ERROR_READ,"Wrong pixelization\n");
  }

  //Read templates and deproject
  if(strcmp(fname_temp,"none")) {
    int ncols,isnest;
    free(cs_dum);
    cs_dum=he_get_file_params(fname_temp,is_healpix,&ncols,&isnest);
    if(!(nmt_diff_curvedsky_info(cs,cs_dum)))
      report_error(NMT_ERROR_READ,"Wrong pixelization\n");
    if((ncols==0) || (ncols%nmaps!=0))
      report_error(NMT_ERROR_READ,"Not enough templates in file %s\n",fname_temp);
    ntemp=ncols/nmaps;
    temp=my_malloc(ntemp*sizeof(flouble **));
    for(itemp=0;itemp<ntemp;itemp++) {
      temp[itemp]=my_malloc(nmaps*sizeof(flouble *));
      for(imap=0;imap<nmaps;imap++)
	temp[itemp][imap]=he_read_map(fname_temp,cs_dum,itemp*nmaps+imap);
    }
  }
  else {
    ntemp=0;
    temp=NULL;
  }

  fl=nmt_field_alloc_sph(cs,mask,pol,maps,ntemp,temp,beam,pure_e,pure_b,
			 n_iter_mask_purify,tol_pinv,niter,0);

  if(beam!=NULL)
    free(beam);
  free(mask);
  for(imap=0;imap<nmaps;imap++)
    free(maps[imap]);
  free(maps);
  if(ntemp>0) {
    for(itemp=0;itemp<ntemp;itemp++) {
      for(imap=0;imap<nmaps;imap++)
	free(temp[itemp][imap]);
      free(temp[itemp]);
    }
    free(temp);
  }
  free(cs);
  free(cs_dum);
  
  return fl;
}
