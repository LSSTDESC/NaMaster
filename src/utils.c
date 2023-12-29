#include "config.h"
#include "utils.h"

#include <setjmp.h>
jmp_buf nmt_exception_buffer;
int nmt_exception_status;
int nmt_error_policy=EXIT_ON_ERROR;
char nmt_error_message[256]="No error\n";

void set_error_policy(int i)
{
  nmt_error_policy=i;
}

void report_error(int level,char *fmt,...)
{
  va_list args;
  char msg[256];

  va_start(args,fmt);
  vsprintf(msg,fmt,args);
  va_end(args);
  
  if(level) {
    if(nmt_error_policy==EXIT_ON_ERROR) {
      fprintf(stderr," Fatal error: %s",msg);
      exit(level);
    }
    else {
      sprintf(nmt_error_message,"%s",msg);
      throw(level);
    }
  }
  else
    fprintf(stderr," Warning: %s",msg);
}

void *my_malloc(size_t size)
{
  void *outptr=malloc(size);
  if(outptr==NULL) report_error(NMT_ERROR_MEMORY,"Out of memory\n");

  return outptr;
}

void *my_calloc(size_t nmemb,size_t size)
{
  void *outptr=calloc(nmemb,size);
  if(outptr==NULL)
    report_error(NMT_ERROR_MEMORY,"Out of memory\n");

  return outptr;
}

gsl_rng *init_rng(unsigned int seed)
{
  gsl_rng *rng=gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng,seed);

  return rng;
}

double rng_01(gsl_rng *rng)
{
  double result=gsl_rng_uniform(rng);
  return result;
}

void rng_gauss(gsl_rng *rng,double *r1,double *r2)
{
  double phase=2*M_PI*rng_01(rng);
  double u=sqrt(-2*log(1-rng_01(rng)));
  *r1=u*cos(phase);
  *r2=u*sin(phase);
}

void end_rng(gsl_rng *rng)
{
  gsl_rng_free(rng);
}

int drc3jj_000(int il2,int il3,int *l1min_out,int *l1max_out,
	       double *lfac,double *thrcof,int size)
{
  int ii,l1max,l1min,nfin;
  l1max=il2+il3;
  l1min=abs(il2-il3);
  *l1max_out=l1max;
  *l1min_out=l1min;

  if(l1max-l1min<0) //Check for meaningful values
    report_error(NMT_ERROR_WIG3J,"WTF?\n");

  if(l1max==l1min) { //If it's only one value:
    int sign2=1;
    if(l1min & 1)
      sign2=-1;
    thrcof[0]=sign2/sqrt((double)(l1min+il2+il3+1));
    return 0;
  }

  nfin=l1max-l1min+1;
  if(nfin>size) {//Check there's enough space
    report_error(NMT_ERROR_WIG3J,"Output array is too small %d\n",nfin);
    return 2;
  }

  for(ii=0;ii<nfin;ii++) {
    int il1=l1min+ii;
    int J=il1+il2+il3;
    int g,sign;
    double lth;
    if(J&1) { //Odd, continue
      thrcof[ii]=0;
      continue;
    }

    //g=J/2
    g=J>>1;

    // Overall sign
    if(g&1)
      sign=-1;
    else
      sign=1;

    lth = 0.5*(lfac[J-2*il1]+lfac[J-2*il2]+lfac[J-2*il3]-lfac[J+1]);
    lth += lfac[g]-lfac[g-il1]-lfac[g-il2]-lfac[g-il3];
    thrcof[ii]=sign*exp(lth);
  }
  return 0;
}

//Returns all non-zero wigner-3j symbols
// il2 (in) : l2
// il3 (in) : l3
// im2 (in) : m2
// im3 (in) : m3
// l1min_out (out) : min value for l1
// l1max_out (out) : max value for l1
// thrcof (out) : array with the values of the wigner-3j
// size (in) : size allocated for thrcof
int drc3jj(int il2,int il3,int im2, int im3,int *l1min_out,
	   int *l1max_out,double *thrcof,int size)
{
  int sign1,sign2,nfin,im1,l1max,l1min,ii,lstep;
  int converging,nstep2,nfinp1,index,nlim;
  double newfac,c1,c2,sum1,sum2,a1,a2,a1s,a2s,dv,denom,c1old,oldfac,l1,l2,l3,m1,m2,m3;
  double x,x1,x2,x3,y,y1,y2,y3,sumfor,sumbac,sumuni,cnorm,thresh,ratio;
  double huge=sqrt(1.79E308/20.0);
  double srhuge=sqrt(huge);
  double tiny=1./huge;
  double srtiny=1./srhuge;

  im1=-im2-im3;
  l2=(double)il2; l3=(double)il3;
  m1=(double)im1; m2=(double)im2; m3=(double)im3;
  
  if((abs(il2+im2-il3+im3))%2==0)
    sign2=1;
  else
    sign2=-1;
  
  //l1 bounds
  l1max=il2+il3;
  l1min=NMT_MAX((abs(il2-il3)),(abs(im1)));
  *l1max_out=l1max;
  *l1min_out=l1min;

  if((il2-abs(im2)<0)||(il3-abs(im3)<0)) {
    for(ii=0;ii<=l1max-l1min;ii++)
      thrcof[ii]=0;
    return 0;
  }
  
  if(l1max-l1min<0) //Check for meaningful values
    report_error(NMT_ERROR_WIG3J,"WTF?\n");
  
  if(l1max==l1min) { //If it's only one value:
    thrcof[0]=sign2/sqrt(l1min+l2+l3+1);
    return 0;
  }
  else {
    nfin=l1max-l1min+1;
    if(nfin>size) //Check there's enough space
      report_error(NMT_ERROR_WIG3J,"Output array is too small %d\n",nfin);
    else {
      l1=l1min;
      newfac=0.;
      c1=0.;
      sum1=(l1+l1+1)*tiny;
      thrcof[0]=srtiny;
      
      lstep=0;
      converging=1;
      while((lstep<nfin-1)&&(converging)) { //Forward series
	lstep++;
	l1++; //order
	
	oldfac=newfac;
	a1=(l1+l2+l3+1)*(l1-l2+l3)*(l1+l2-l3)*(-l1+l2+l3+1);
	a2=(l1+m1)*(l1-m1);
	newfac=sqrt(a1*a2);
	
	if(l1>1) {
	  dv=-l2*(l2+1)*m1+l3*(l3+1)*m1+l1*(l1-1)*(m3-m2);
	  denom=(l1-1)*newfac;
	  if(lstep>1)
	    c1old=fabs(c1);
	  c1=-(l1+l1-1)*dv/denom;
	}
	else {
	  c1=-(l1+l1-1)*l1*(m3-m2)/newfac;
	}
	
	if(lstep<=1) {
	  x=srtiny*c1;
	  thrcof[1]=x;
	  sum1+=tiny*(l1+l1+1)*c1*c1;
	}
	else {
	  c2=-l1*oldfac/denom;
	  x=c1*thrcof[lstep-1]+c2*thrcof[lstep-2];
	  thrcof[lstep]=x;
	  sumfor=sum1;
	  sum1+=(l1+l1+1)*x*x;
	  if(lstep<nfin-1) {
	    if(fabs(x)>=srhuge) {
	      for(ii=0;ii<=lstep;ii++) {
		if(fabs(thrcof[ii])<srtiny)
		  thrcof[ii]=0;
		thrcof[ii]/=srhuge;
	      }
	      sum1/=huge;
	      sumfor/=huge;
	      x/=srhuge;
	    }
	    
	    if(c1old<=fabs(c1))
	      converging=0;
	  }
	}
      }
      
      if(nfin>2) {
	x1=x;
	x2=thrcof[lstep-1];
	x3=thrcof[lstep-2];
	nstep2=nfin-lstep-1+3;
	
	nfinp1=nfin+1;
	l1=l1max;
	thrcof[nfin-1]=srtiny;
	sum2=tiny*(l1+l1+1);
	
	l1+=2;
	lstep=0;
	while(lstep<nstep2-1) { //Backward series
	  lstep++;
	  l1--;
	  
	  oldfac=newfac;
	  a1s=(l1+l2+l3)*(l1-l2+l3-1)*(l1+l2-l3-1)*(-l1+l2+l3+2);
	  a2s=(l1+m1-1)*(l1-m1-1);
	  newfac=sqrt(a1s*a2s);
	  
	  dv=-l2*(l2+1)*m1+l3*(l3+1)*m1+l1*(l1-1)*(m3-m2);
	  denom=l1*newfac;
	  c1=-(l1+l1-1)*dv/denom;
	  if(lstep<=1) {
	    y=srtiny*c1;
	    thrcof[nfin-2]=y;
	    sumbac=sum2;
	    sum2+=tiny*(l1+l1-3)*c1*c1;
	  }
	  else {
	    c2=-(l1-1)*oldfac/denom;
	    y=c1*thrcof[nfin-lstep]+c2*thrcof[nfinp1-lstep]; //is the index ok??
	    if(lstep!=nstep2-1) {
	      thrcof[nfin-lstep-1]=y; //is the index ok??
	      sumbac=sum2;
	      sum2+=(l1+l1-3)*y*y;
	      if(fabs(y)>=srhuge) {
		for(ii=0;ii<=lstep;ii++) {
		  index=nfin-ii-1; //is the index ok??
		  if(fabs(thrcof[index])<srtiny)
		    thrcof[index]=0;
		  thrcof[index]=thrcof[index]/srhuge;
		}
		sum2/=huge;
		sumbac/=huge;
	      }
	    }
	  }
	}
	
	y3=y;
	y2=thrcof[nfin-lstep]; //is the index ok??
	y1=thrcof[nfinp1-lstep]; //is the index ok??
	
	ratio=(x1*y1+x2*y2+x3*y3)/(x1*x1+x2*x2+x3*x3);
	nlim=nfin-nstep2+1;
	
	if(fabs(ratio)<1) {
	  nlim++;
	  ratio=1./ratio;
	  for(ii=nlim-1;ii<nfin;ii++) //is the index ok??
	    thrcof[ii]*=ratio;
	  sumuni=ratio*ratio*sumbac+sumfor;
	}
	else {
	  for(ii=0;ii<nlim;ii++)
	    thrcof[ii]*=ratio;
	  sumuni=ratio*ratio*sumfor+sumbac;
	}
      }
      else
	sumuni=sum1;
      
      cnorm=1./sqrt(sumuni);
      sign1 = copysign(1., thrcof[nfin-1]);
      
      if(sign1*sign2<=0)
	cnorm=-cnorm;
      if(fabs(cnorm)>=1) {
	for(ii=0;ii<nfin;ii++)
	  thrcof[ii]*=cnorm;
	return 0;
      }
      else {
	thresh=tiny/fabs(cnorm);
	for(ii=0;ii<nfin;ii++) {
	  if(fabs(thrcof[ii])<thresh)
	    thrcof[ii]=0;
	  thrcof[ii]*=cnorm;
	}
	return 0;
      }
    } //Size is good
  } //Doing for many l1s
  
  return 2;
}

void moore_penrose_pinv(gsl_matrix *M,double threshold)
{
  if(M->size1!=M->size2)
    report_error(NMT_ERROR_PINV,"Matrix must be square\n");
  
  gsl_eigen_symmv_workspace *w=gsl_eigen_symmv_alloc(M->size1);
  gsl_vector *eval=gsl_vector_alloc(M->size1); 
  gsl_matrix *evec=gsl_matrix_alloc(M->size1,M->size2);
  gsl_matrix *Mc=gsl_matrix_alloc(M->size1,M->size2);
  gsl_matrix_memcpy(Mc,M);

  //Compute eigenvectors and eigenvalues
  gsl_eigen_symmv(Mc,eval,evec,w);

  //Compute inverse eigenvalues (non-singular)
  int ii;
  double eval_max=gsl_vector_max(eval);
  gsl_matrix_set_zero(M);
  for(ii=0;ii<M->size1;ii++) {
    double v=gsl_vector_get(eval,ii);
    if(v<threshold*eval_max)
      gsl_matrix_set(M,ii,ii,0);
    else
      gsl_matrix_set(M,ii,ii,1./v);
  }

  //Put it back together
  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,M,evec,0,Mc); // Lambda * E^T
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,evec,Mc,0,M); // E * Lambda * E^T
  
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  gsl_matrix_free(Mc);
  gsl_eigen_symmv_free(w);
}

//Covariance index handling
#define IND_COV_0 0
#define IND_COV_P 1
#define IND_COV_MP 2
#define IND_COV_MM 3
#define IND_COV_00 0
#define IND_COV_0P 1
#define IND_COV_PP 2
#define IND_COV_MMP 3
#define IND_COV_MMM 4
#define IND_COV_ZERO 5
static const int covar_indices[16]={IND_COV_P   ,IND_COV_MM  ,IND_COV_MP  ,IND_COV_ZERO,
				    IND_COV_MP  ,IND_COV_P   ,IND_COV_ZERO,IND_COV_MP  ,
				    IND_COV_MM  ,IND_COV_ZERO,IND_COV_P   ,IND_COV_MM  ,
				    IND_COV_ZERO,IND_COV_MM  ,IND_COV_MP  ,IND_COV_P};

static int cov_get_coupling_index(int nma,int nmb,int ia1,int ib1,int ia2,int ib2)
{
  if(nma==1) {
    if(nmb==1) //TT,TT
      return IND_COV_0;
    else {
      if(ib1==ib2) //TE,TE or TB,TB
	return IND_COV_0;
      else
	return IND_COV_ZERO; //TE,TB or TB,TE
    }
  }
  else {
    if(nmb==1) {
      if(ia1==ia2) //ET,ET or BT,BT
	return IND_COV_0;
      else
	return IND_COV_ZERO; //ET,BT or BT,ET
    }
    else {//22,22
      int icl1=ib1+2*ia1;
      int icl2=ib2+2*ia2;
      int ind=icl2+4*icl1;
      return covar_indices[ind];
    }
  }
}

int cov_get_coupling_pair_index(int na,int nc,int nb,int nd,
				int ia1,int ia2,int ic1,int ic2,
				int ib1,int ib2,int id1,int id2)
{
  int ind1=cov_get_coupling_index(na,nc,ia1,ic1,ia2,ic2);
  int ind2=cov_get_coupling_index(nb,nd,ib1,id1,ib2,id2);
  
  if((ind1==IND_COV_0) && (ind2==IND_COV_0))
    return IND_COV_00;
  if(((ind1==IND_COV_0) && (ind2==IND_COV_P)) ||
     ((ind1==IND_COV_P) && (ind2==IND_COV_0)))
    return IND_COV_0P;
  if((ind1==IND_COV_P) && (ind2==IND_COV_P))
    return IND_COV_PP;
  if(((ind1==IND_COV_MP) && (ind2==IND_COV_MP)) ||
     ((ind1==IND_COV_MM) && (ind2==IND_COV_MM)))
    return IND_COV_MMM;
  if(((ind1==IND_COV_MP) && (ind2==IND_COV_MM)) ||
     ((ind1==IND_COV_MM) && (ind2==IND_COV_MP)))
    return IND_COV_MMP;
  return IND_COV_ZERO;
}
