import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

# This script showcases the use of catalog-based pseudo-C_ell computation from
# Wolz et al. 2024 (in prep.) in the case of a source clustering catalog.

# We start by initializing a lognormally distributed source density field at
# a given nside and a given source density and survey mask.
nside = 128
source_density = 0.001  # source number density per square arcmin

mask = hp.ud_grade(
    hp.read_map("/home/wolz/pclcat/data/WISExSCOSmask_equatorial.fits.gz"),
    nside_out=nside
)
ls = np.arange(3*nside)
cl = 1./(ls + 10.)**2.
npix = hp.nside2npix(nside)
pix_area_arcmin = (4*np.pi/npix)*(180*60/np.pi)**2
deltaG = hp.synfast(cl, nside)
nmap_src = source_density*pix_area_arcmin*np.exp(deltaG
                                                 - 0.5*np.std(deltaG)**2)

# We then draw a Cox-distributed source catalog with lognormal intensity
# measure (i.e., a Poisson-sampled source catalog whose intensity is also
# stochastic).
num_src_mean = source_density * pix_area_arcmin
num_sources = int(np.amax(nmap_src*mask)*len(nmap_src))

# probability map for a random source to be sampled at a given pixel
pmap_src = mask*nmap_src/np.amax(mask*nmap_src)

# draw random source positions with continous random angles
cth = -1 + 2*np.random.rand(int(num_sources))
phi = 2*np.pi*np.random.rand(int(num_sources))
sth = np.sqrt(1 - cth**2)
unif = np.random.rand(num_sources)
angles = np.array([sth*np.cos(phi), sth*np.sin(phi), cth])
ipix = hp.vec2pix(nside, *angles)
good = unif <= pmap_src[ipix]
positions = np.array([np.arccos(cth[good]), phi[good]])

# We then draw Poisson-distributed random source catalog with 50 times higher
# number density than the source catalog.
nran_factor = 50
num_ran_mean = num_src_mean * nran_factor
nmap_ran = num_ran_mean * mask
pmap_ran = nmap_ran/np.amax(nmap_ran)
num_randoms = int(np.amax(nmap_ran)*len(nmap_ran))
cth = -1 + 2*np.random.rand(int(num_randoms))
phi = 2*np.pi*np.random.rand(int(num_randoms))
sth = np.sqrt(1 - cth**2)
unif = np.random.rand(num_randoms)
angles = np.array([sth*np.cos(phi), sth*np.sin(phi), cth])
ipix = hp.vec2pix(nside, *angles)
good = unif <= pmap_ran[ipix]
positions_rand = np.array([np.arccos(cth[good]), phi[good]])

# We assign uniform weights to all sources and randoms.
weights = np.ones(len(positions[0])).astype(np.float64)
weights_rand = np.ones(len(positions_rand[0])).astype(np.float64)

# Now, we compute the mode coupling matrix at the positions and weights of the
# random catalog.
nmt_bin = nmt.NmtBin.from_nside_linear(nside, nlb=10)
lb = nmt_bin.get_effective_ells()
f = nmt.NmtFieldCatalogClustering(
    positions, weights, positions_rand, weights_rand, 3*nside-1
)
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f, f, nmt_bin)

# We compute the clustering field by passing both the source catalog and the
# random catalog.
pcl_uncorrected = hp.alm2cl(f.alm)
pcl = nmt.compute_coupled_cell(f, f)

# We plot both the shot-noise corrected and uncorrected coupled overdensity
# power spectrum.
plt.clf()
plt.plot(pcl.flatten(), label="Catalog-based", color="darkorange", ls="-")
plt.plot(pcl_uncorrected.flatten(), color="darkorange", alpha=0.5, ls=":",
         label="Uncorrected")
plt.loglog()
plt.xlabel(r"$\ell$", fontsize=16)
plt.ylabel(r"$C_\ell^{\delta\delta,\, {\rm coupled}}$", fontsize=16)
plt.axhline(f.Nf, color="k", linestyle="--", label=r"$N_f$")
plt.ylim((f.Nf/1000, None))
plt.legend(fontsize=13)
plt.savefig("sample_clusteringcatalog_coupled.png", bbox_inches="tight")
plt.show()

# We need to divide it by the pixel window function imprinted on the source
# field values by the (pixelized) density map.
pcl /= hp.pixwin(nside)**2

# Now, we compute the decoupled overdensity power spectrum
clb = wsp.decouple_cell(pcl).flatten()
clb_uncorrected = wsp.decouple_cell(pcl_uncorrected).flatten()

# Finally, we compute the binned theory power spectrum and compare with data.
delta = nmap_src/np.mean(nmap_src) - 1
cl_clean = np.array([hp.anafast(delta)])[0, :3*nside]
pcl_clean = wsp.couple_cell(cl_clean.reshape((1, -1)))
cl_clean = wsp.decouple_cell(pcl_clean).flatten()

plt.clf()
plt.plot(lb, clb, "bo", label="Catalog-based")
plt.plot(lb, clb_uncorrected, "bo", mfc="w", label="Uncorrected")
plt.plot(lb, cl_clean, "k-", label="Expectation")
plt.loglog()
plt.legend(fontsize=13)
plt.xlabel(r"$\ell$", fontsize=16)
plt.ylabel(r"$C_\ell^{\delta\delta,\, {\rm decoupled}}$", fontsize=16)
plt.savefig("sample_clusteringcatalog_decoupled.png", bbox_inches="tight")
plt.show()
