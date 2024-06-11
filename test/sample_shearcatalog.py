import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

# This script showcases the use of catalog-based pseudo-C_ell computation from
# Wolz et al. 2024 (in prep.) in the case of a masked spin-2 field catalog.

# We start by creating a pixel-level survey mask
nside = 128
mask = hp.ud_grade(
    hp.read_map("/home/wolz/pclcat/data/WISExSCOSmask_equatorial.fits.gz"),
    nside_out=nside
)
plt.clf()
hp.mollview(mask, title="Survey mask")
plt.savefig("sample_shearcatalog_mask.png")
plt.tight_layout()
plt.show()

# We then draw 1e7 random sources uniformly distributed across the masked sky
num_sources = 1e7
cth = -1 + 2*np.random.rand(int(num_sources))
phi = 2*np.pi*np.random.rand(int(num_sources))
sth = np.sqrt(1 - cth**2)
angles = np.array([sth*np.cos(phi), sth*np.sin(phi), cth])
ipix = hp.vec2pix(nside, *angles)
good = mask[ipix] > 0
positions = np.array([np.arccos(cth[good]), phi[good]])

# For simplicity, these sources have uniform weights.
weights = np.ones_like(positions[0])

# Then, we generate a Gaussian spin-2 field with only E-modes, to mimic e.g. a
# cosmic shear field.
ls = np.arange(3*nside)
cl = 1./(ls + 10.)**2.
cl0 = 0*cl
map_Q, map_U = hp.synfast([cl0, cl, cl0, cl0], nside, new=True)[1:]
catalog_Q = map_Q[ipix][good]
catalog_U = map_U[ipix][good]

# Now, we compute the mode coupling matrix of the source catalog.
nmt_bin = nmt.NmtBin.from_nside_linear(nside, nlb=10)
lb = nmt_bin.get_effective_ells()
f = nmt.NmtFieldCatalog(positions, weights, None, lmax=nmt_bin.lmax,
                        lmax_mask=nmt_bin.lmax, spin=2)
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f, f, nmt_bin)

# We compute and plot the survey mask's pseudo power spectrum, highlighting
# the contribution of the mask shot noise.
pcl_mask = hp.alm2cl(f.get_mask_alms())
plt.clf()
plt.plot(pcl_mask - f.Nw, label="Catalog-based", color="darkorange", ls="-")
plt.plot(pcl_mask, color="darkorange", alpha=0.5, ls=":", label="Uncorrected")
plt.loglog()
plt.axhline(f.Nw, color="k", linestyle="--", label=r"$N_w$")
plt.ylabel(r"$C_\ell^w$", fontsize=16)
plt.xlabel(r"$\ell$", fontsize=16)
plt.legend(fontsize=13)
plt.savefig("sample_shearcatalog_mask_pcl.png", bbox_inches="tight")
plt.show()

# We then compute the coupled spin-2 pseudo power spectrum 
f = nmt.NmtFieldCatalog(positions, weights, [catalog_Q, catalog_U],
                        lmax=3*nside-1, spin=2)
pcl = nmt.compute_coupled_cell(f, f)

# We need to divide it by the pixel window function imprinted on the source
# field values by the (pixelized) density map.
pcl /= hp.pixwin(nside)**2

# Finally, we compute the decoupled power spectra, the binned theory
# expectation, and plot them.
clb = wsp.decouple_cell(pcl)
clb_theory = wsp.decouple_cell(wsp.couple_cell([cl, cl0, cl0, cl0]))

plt.clf()
plt.plot(lb, clb[0], "bo", label="EE")
plt.plot(lb, clb[3], "ro", label="BB")
plt.plot(lb, clb_theory[0], "k-", label="Expected EE")
plt.loglog()
plt.ylabel(r"$C_\ell^{\rm decoupled}$", fontsize=16)
plt.xlabel(r"$\ell$", fontsize=16)
plt.legend(fontsize=13)
plt.savefig("sample_shearcatalog_decoupled.png", bbox_inches="tight")
plt.show()
