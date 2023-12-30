import numpy as np
import healpy as hp
import pymaster as nmt

msk = hp.read_map("msk.fits")
mt, mq, mu = hp.read_map("mps.fits", field=[0, 1, 2])
ct, cq, cu = hp.read_map("tmp.fits", field=[0, 1, 2])
sig2 = np.ones_like(mt)

f2 = nmt.NmtField(msk, [mq, mu], templates=[[cq, cu]])
cln = nmt.uncorr_noise_deprojection_bias(f2, sig2)

np.savetxt("bm_uncorr_noise_dp.txt", cln.T)
import matplotlib.pyplot as plt
plt.plot(cln[0], 'k-')
plt.plot(cln[3], 'r-')
plt.plot(cln[1], 'b--')
plt.plot(cln[2], 'c--')
plt.show()
