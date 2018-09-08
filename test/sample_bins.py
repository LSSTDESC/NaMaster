import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

#This script showcases the use of the NmtBin structure to define bandpowers.

#HEALPix map resolution
nside=256

#Initialize binning scheme with bandpowers of constant width
#(4 multipoles per bin)
bin1=nmt.NmtBin(nside,nlb=4)

#Initialize binning scheme with custom-made bandpowers.
#In this case we simply manually choose these bandpowers to also have
#4 multipoles per bin.
ells=np.arange(3*nside,dtype='int32') #Array of multipoles
weights=0.25*np.ones_like(ells) #Array of weights
bpws=-1+np.zeros_like(ells) #Array of bandpower indices
i=0;
while 4*(i+1)+2<3*nside :
    bpws[4*i+2:4*(i+1)+2]=i
    i+=1
bin2=nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=weights)

#At this stage bin1 and bin2 should be identical
print(np.sum(bin1.get_effective_ells()-bin2.get_effective_ells()))

#Array with effective multipole per bandpower
ell_eff=bin1.get_effective_ells()

#Bandpower info:
print("Bandpower info:")
print(" %d bandpowers"%(bin1.get_n_bands()))
print("The columns in the following table are:")
print(" [1]-band index, [2]-list of multipoles, [3]-list of weights, [4]=effective multipole")
for i in range(bin1.get_n_bands()) :
    print(i, bin1.get_ell_list(i), bin1.get_weight_list(i), ell_eff[i])
print("")

#Binning a power spectrum
#Read the TT power spectrum
data=np.loadtxt("cls.txt",unpack=True);
ell_arr=data[0]; cl_tt=data[1]
#Bin the power spectrum into bandpowers
cl_tt_binned=bin1.bin_cell(np.array([cl_tt]))
#Unbin bandpowers
cl_tt_binned_unbinned=bin1.unbin_cell(cl_tt_binned)
#Plot all to see differences
plt.plot(ell_arr,cl_tt                   ,'r-',label='Original $C_\\ell$')
plt.plot(ell_eff,cl_tt_binned[0]         ,'g-',label='Binned $C_\\ell$')
plt.plot(ell_arr,cl_tt_binned_unbinned[0],'b-',label='Binned-unbinned $C_\\ell$')
plt.loglog()
plt.legend(loc='upper right',frameon=False)
plt.show()
