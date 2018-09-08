import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

#This script showcases the ability of namaster to compute Gaussian
#estimates of the covariance matrix. This is currently only
#supported for spin-0 fields
#A similar example for flat-sky fields can be found in test/sample_covariance_flat.py

#HEALPix map resolution
nside=256

#We start by creating some synthetic masks and maps with contaminants.
#Here we will focus on the auto-correlation of a spin-1 field.
#a) Read and apodize mask
mask=nmt.mask_apodization(hp.read_map("mask.fits",verbose=False),1.,apotype="Smooth")

#Let's now create a fictitious theoretical power spectrum to generate
#Gaussian realizations:
larr=np.arange(3*nside)
clarr=((larr+1.)/80.)**(-1.1)+1.

#This routine generates a scalar Gaussian random field based on this
#power spectrum
def get_sample_field() :
    mp=hp.synfast(clarr,nside,verbose=False)
    return nmt.NmtField(mask,[mp])

#We also copy this function from sample_workspaces.py. It computes
#power spectra given a pair of fields and a workspace.
def compute_master(f_a,f_b,wsp) :
    cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
    cl_decoupled=wsp.decouple_cell(cl_coupled)

    return cl_decoupled

#Let's generate one particular sample and its power spectrum.
print("Field")
f0=get_sample_field()
b=nmt.NmtBin(nside,nlb=20) #We will use 20 multipoles per bandpower.
print("Workspace")
w=nmt.NmtWorkspace()
w.compute_coupling_matrix(f0,f0,b)
cl_0=compute_master(f0,f0,w)[0]

#Let's now compute the Gaussian estimate of the covariance!
print("Covariance")
#First we generate a NmtCovarianceWorkspace object to precompute
#and store the necessary coupling coefficients
cw=nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(w,w) #<- This is the time-consuming operation
covar=nmt.gaussian_covariance(cw,clarr,clarr,clarr,clarr)

#Let's now compute the sample covariance
print("Sample covariance")
nsamp=100
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
plt.figure(); plt.imshow(covar,origin='lower',interpolation='nearest')
plt.figure(); plt.imshow(covar_sample,origin='lower',interpolation='nearest')
plt.figure(); plt.imshow(covar-covar_sample,origin='lower',interpolation='nearest')
plt.show()
