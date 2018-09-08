import numpy as np
#import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
import os

#This script describes the functionality of the flat-sky version of pymaster

#We start by defining the flat-sky field
Lx=72.*np.pi/180; Ly=48.*np.pi/180;
Nx=602; Ny=410;
#Nx=301; Ny=205;

#Let's now create a mask:
mask=np.ones(Nx*Ny)
xarr=np.ones(Ny)[:,None]*np.arange(Nx)[None,:]*Lx/Nx
yarr=np.ones(Nx)[None,:]*np.arange(Ny)[:,None]*Ly/Ny
def dig_hole(x,y,r) :
    rad=(np.sqrt((xarr-x)**2+(yarr-y)**2)).flatten()
    return np.where(rad<r)[0]
mask[dig_hole(0.3*Lx,0.6*Ly ,0.05*np.sqrt(Lx*Ly))]=0.
mask[dig_hole(0.7*Lx,0.12*Ly,0.07*np.sqrt(Lx*Ly))]=0.
mask[dig_hole(0.7*Lx,0.8*Ly ,0.03*np.sqrt(Lx*Ly))]=0.
mask[np.where(xarr.flatten()<Lx/16.)]=0; mask[np.where(xarr.flatten()>15*Lx/16.)]=0;
mask[np.where(yarr.flatten()<Ly/16.)]=0; mask[np.where(yarr.flatten()>15*Ly/16.)]=0;
mask=mask.reshape([Ny,Nx])
mask=nmt.mask_apodization_flat(mask,Lx,Ly,aposize=2.,apotype="C1");
plt.figure(); plt.imshow(mask,interpolation='nearest',origin='lower'); plt.colorbar()

#Binning scheme
l0_bins=np.arange(Nx/8)*8*np.pi/Lx
lf_bins=(np.arange(Nx/8)+1)*8*np.pi/Lx
b=nmt.NmtBinFlat(l0_bins,lf_bins)
ells_uncoupled=b.get_effective_ells()

#Let's create a fictitious theoretical power spectrum to generate
#Gaussian simulations:
larr=np.arange(3000.)
clarr=((larr+50.)/300.)**(-1.1)+0.5

#This function will generate random fields
def get_sample_field() :
    mpt=nmt.synfast_flat(Nx,Ny,Lx,Ly,np.array([clarr]),[0])[0]
    return nmt.NmtFieldFlat(Lx,Ly,mask,[mpt])

#Convenience function from sample_workspaces.py for flat-sky fields
def compute_master(f_a,f_b,wsp) :
    cl_coupled=nmt.compute_coupled_cell_flat(f_a,f_b,b);
    cl_decoupled=wsp.decouple_cell(cl_coupled)

    return cl_decoupled

#Let's generate one particular sample and its power spectrum
print("Field")
f0=get_sample_field()
plt.figure(); plt.imshow(f0.get_maps()[0]*mask,interpolation='nearest',origin='lower'); plt.colorbar()
print("Workspace")
w=nmt.NmtWorkspaceFlat();
if not os.path.isfile("w_flat_covar.dat") :
    w.compute_coupling_matrix(f0,f0,b)
    w.write_to("w_flat_covar.dat");
w.read_from("w_flat_covar.dat")
cl_0=compute_master(f0,f0,w)[0]

#Let's now compute the Gaussian estimate of the covariance!
print("Covariance")
#First we generate a NmtCovarianceWorkspaceFlat object to precompute
#and store the necessary coupling coefficients
cw=nmt.NmtCovarianceWorkspaceFlat();
if not os.path.isfile("cw_flat.dat") :
    cw.compute_coupling_coefficients(w,w) #<- Thisi is the time-consuming operation
    cw.write_to("cw_flat.dat")
cw.read_from("cw_flat.dat")
covar=nmt.gaussian_covariance_flat(cw,larr,clarr,clarr,clarr,clarr)

#Let's now compute the sample covariance
print("Sample covariance")
nsamp=1000
covar_sample=np.zeros([len(cl_0),len(cl_0)])
mean_sample=np.zeros(len(cl_0))
for i in np.arange(nsamp) :
    print(i)
    f=get_sample_field()
    cl=compute_master(f,f,w)[0]
    covar_sample+=cl[None,:]*cl[:,None]
    mean_sample+=cl
mean_sample/=nsamp
covar_sample=covar_sample/nsamp-mean_sample[None,:]*mean_sample[:,None]

#Let's plot them:
plt.figure();
plt.plot(ells_uncoupled[0:],np.fabs(np.diag(covar,k=0)),'r-',label='0-th diag., theory');
plt.plot(ells_uncoupled[0:],np.fabs(np.diag(covar_sample,k=0)),'b-',label='0-th diag., 10K sims'); 
plt.plot(ells_uncoupled[1:],np.fabs(np.diag(covar,k=1)),'r--',label='1-st diag., theory');
plt.plot(ells_uncoupled[1:],np.fabs(np.diag(covar_sample,k=1)),'b--',label='1-st diag., !0K sims');
plt.xlabel('$\\ell$',fontsize=16)
plt.ylabel('${\\rm diag}({\\rm Cov})$',fontsize=16)
plt.legend(loc='upper right',frameon=False)
plt.loglog()
plt.savefig("diags.png",bbox_inches='tight')
plt.figure(); plt.title("Correlation matrix residuals");
plt.imshow((covar-covar_sample)/np.sqrt(np.diag(covar)[None,:]*np.diag(covar)[:,None]),
           origin='lower',interpolation='nearest');
plt.colorbar()
plt.savefig("corr_res.png",bbox_inches='tight');
plt.show()
