import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

nside = 256

mp_t, mp_q, mp_u = hp.read_map("maps.fits",
                               field=[0, 1, 2],
                               verbose=False)
mask = np.ones(hp.nside2npix(nside))
cls = hp.anafast([mp_t, mp_q, mp_u], pol=True)
aft_tt, aft_ee, aft_bb, aft_te, aft_eb, aft_tb = cls

b = nmt.NmtBin.from_nside_linear(nside, 1)
larr = b.get_effective_ells()

f0 = nmt.NmtField(mask, [mp_t])
f2 = nmt.NmtField(mask, [mp_q, mp_u])
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0, f0, b)
w02 = nmt.NmtWorkspace()
w02.compute_coupling_matrix(f0, f2, b)
w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(f2, f2, b)


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled


nmt_tt = compute_master(f0, f0, w00)[0]
nmt_te, nmt_tb = compute_master(f0, f2, w02)
nmt_ee, nmt_eb, nmt_be, nmt_bb = compute_master(f2, f2, w22)


def compare_cl(aft_cl, nmt_cl, title):
    print(np.sqrt(np.sum((aft_cl[2:2*nside] -
                          nmt_cl[:2*nside-2])**2) / (len(nmt_cl)-2)))
    plt.figure()
    plt.title(title)
    plt.plot(larr[:2*nside-2], aft_cl[2:2*nside]-nmt_cl[:2*nside-2])


compare_cl(aft_tt, nmt_tt, 'TT')
compare_cl(aft_te, nmt_te, 'TE')
compare_cl(aft_tb, nmt_tb, 'TB')
compare_cl(aft_ee, nmt_ee, 'EE')
compare_cl(aft_eb, nmt_eb, 'EB')
compare_cl(aft_eb, nmt_be, 'BE')
compare_cl(aft_bb, nmt_bb, 'BB')
plt.show()
