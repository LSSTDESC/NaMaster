import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import healpy as hp

np.random.seed(1234)

l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
cltt[:2]=0
fl = np.sqrt(l*(l+1))
fl[0] = 1
fl = 1./fl

nside_out = 64
# Generate spin-0 field
m = hp.synfast(cltt[:3*nside_out], nside_out, new=True, verbose=False)
# Compute alm / sqrt(l*(l+1)) and then its derivatives
a = hp.map2alm(m)
a = hp.almxfl(a, fl)
_, mdth, mdph = hp.alm2map_der1(a, nside_out)

hp.write_map("mps_sp1.fits", [m, mdth, mdph], overwrite=True)

msk = np.ones_like(m)
f0 = nmt.NmtField(msk, [m], spin=0, n_iter=0)
f1 = nmt.NmtField(msk, [mdth, mdph], spin=1, n_iter=0)
cl00 = nmt.compute_coupled_cell(f0, f0)
cl01 = nmt.compute_coupled_cell(f0, f1)
cl11 = nmt.compute_coupled_cell(f1, f1)

msk = hp.read_map("msk.fits", verbose=False)
msk_bin = np.ones_like(msk)
msk_bin[msk <= 0] = 0
f0 = nmt.NmtField(msk, [m*msk_bin], spin=0)
f1 = nmt.NmtField(msk, [mdth*msk_bin, mdph*msk_bin], spin=1)
b = nmt.NmtBin(nside_out,nlb=16)
w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
w00.write_to('bm_sp1_w00.fits')
w01 = nmt.NmtWorkspace.from_fields(f0, f1, b)
w01.write_to('bm_sp1_w01.fits')
w11 = nmt.NmtWorkspace.from_fields(f1, f1, b)
w11.write_to('bm_sp1_w11.fits')
c00 = w00.decouple_cell(nmt.compute_coupled_cell(f0, f0))
c01 = w01.decouple_cell(nmt.compute_coupled_cell(f0, f1))
c11 = w11.decouple_cell(nmt.compute_coupled_cell(f1, f1))
leff = b.get_effective_ells()
np.savetxt('bm_sp1_c00.txt', np.transpose([leff, c00[0]]))
np.savetxt('bm_sp1_c01.txt', np.transpose([leff, c01[0], c01[1]]))
np.savetxt('bm_sp1_c11.txt', np.transpose([leff, c11[0], c11[1], c11[2], c11[3]]))

plt.figure()
plt.plot(leff, c00[0]/c00[0]-1, 'k-')
plt.plot(leff, c01[0]/c00[0]-1, 'r-')
plt.plot(leff, c11[0]/c00[0]-1, 'b-')
plt.plot(leff, c01[1]/c00[0], 'r--')
plt.plot(leff, c11[2]/c00[0], 'b--')

plt.figure()
plt.plot(l[:3*nside_out], cl00[0], 'k-')
plt.plot(l[:3*nside_out], cl01[0], 'r-')
plt.plot(l[:3*nside_out], cl11[0], 'b-')
plt.plot(l[:3*nside_out], cl01[1], 'r--')
plt.plot(l[:3*nside_out], cl11[3], 'b--')
plt.show()
