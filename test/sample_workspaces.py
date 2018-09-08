import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

#This script showcases the use of NmtWorkspace objects to speed up the
#computation of power spectra for many pairs of fields with the same masks.

#HEALPix map resolution
nside=256

#We start by creating some synthetic masks and maps with contaminants.
#Here we will focus on the cross-correlation of a spin-2 and a spin-1 field.
#a) Read and apodize mask
mask=nmt.mask_apodization(hp.read_map("mask.fits",verbose=False),1.,apotype="Smooth")
#b) Read maps
mp_t,mp_q,mp_u=hp.read_map("maps.fits",field=[0,1,2],verbose=False)
#c) Read contaminants maps
tm_t,tm_q,tm_u=hp.read_map("temp.fits",field=[0,1,2],verbose=False)
#d) Create contaminated fields
#   Spin-0
f0=nmt.NmtField(mask,[mp_t+tm_t],templates=[[tm_t]])
#   Spin-2
f2=nmt.NmtField(mask,[mp_q+tm_q,mp_u+tm_u],templates=[[tm_q,tm_u]])
#e) Create binning scheme. We will use 20 multipoles per bandpower.
b=nmt.NmtBin(nside,nlb=20)
#f) Finally, we read our best guess for the true power spectrum. We will
#   use this to:
#   i)  Compute the bias to the power spectrum from contaminant cleaning
#   ii) Generate random realizations of our fields to compute the errors
l,cltt,clee,clbb,clte=np.loadtxt("cls.txt",unpack=True)
cl_02_th=np.array([clte,np.zeros_like(clte)])

#We then generate an NmtWorkspace object that we use to compute and store
#the mode coupling matrix. Note that this matrix depends only on the masks
#of the two fields to correlate, but not on the maps themselves (in this
#case both maps are the same.
w=nmt.NmtWorkspace()
w.compute_coupling_matrix(f0,f2,b)

#Since we suspect that our maps are contaminated (that's why we passed the
#contaminant templates as arguments to the NmtField constructor), we also
#need to compute the bias to the power spectrum caused by contaminant
#cleaning (deprojection bias).
cl_bias=nmt.deprojection_bias(f0,f2,cl_02_th)

#The function defined below will compute the power spectrum between two
#NmtFields f_a and f_b, using the coupling matrix stored in the
#NmtWorkspace wsp and subtracting the deprojection bias clb.
#Note that the most expensive operations in the MASTER algorithm are
#the computation of the coupling matrix and the deprojection bias. Since
#these two objects are precomputed, this function should be pretty fast!
def compute_master(f_a,f_b,wsp,clb) :
    #Compute the power spectrum (a la anafast) of the masked fields
    #Note that we only use n_iter=0 here to speed up the computation,
    #but the default value of 3 is recommended in general.
    cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
    #Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled=wsp.decouple_cell(cl_coupled,cl_bias=clb)

    return cl_decoupled

#OK, we can now compute the power spectrum of our two input fields
cl_master=compute_master(f0,f2,w,cl_bias)

#Let's now compute the errors on this estimator using 100 Gaussian random
#simulations. In a realistic scenario you'd want to compute the full
#covariance matrix, but let's keep things simple.
nsim=100
cl_mean=np.zeros_like(cl_master)
cl_std=np.zeros_like(cl_master)
for i in np.arange(nsim) :
    print("%d-th simulation"%i)
    t,q,u=hp.synfast([cltt,clee,clbb,clte],nside,verbose=False)
    f0_sim=nmt.NmtField(mask,[t],templates=[[tm_t]])
    f2_sim=nmt.NmtField(mask,[q,u],templates=[[tm_q,tm_u]])
    cl=compute_master(f0_sim,f2_sim,w,cl_bias)
    cl_mean+=cl
    cl_std+=cl*cl
cl_mean/=nsim
cl_std=np.sqrt(cl_std/nsim-cl_mean*cl_mean)

#One final thing needs to be done before we can compare the result with
#the theory. The theory power spectrum must be binned into bandpowers in
#the same manner the data has. This is straightforward to do using just
#two nested function calls.
cl_02_th_binned=w.decouple_cell(w.couple_cell(cl_02_th))

#Now let's plot the result!
plt.plot(b.get_effective_ells(),cl_02_th_binned[0],'r-',label='True power spectrum')
plt.plot(b.get_effective_ells(),cl_02_th_binned[1],'g-')
plt.errorbar(b.get_effective_ells(),cl_master[0],yerr=cl_std[0],
             fmt='ro',label='MASTER estimate (TE)')
plt.errorbar(b.get_effective_ells(),cl_master[1],yerr=cl_std[1],
             fmt='bo',label='MASTER estimate (TB)')
plt.ylim([-0.03,0.03]);
plt.legend(loc='upper right')
plt.xlabel('$\\ell$',fontsize=16); plt.ylabel('$C_\\ell$',fontsize=16);
plt.show()
