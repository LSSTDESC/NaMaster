import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS

# This script showcases the use of the NaMaster to compute power spectra
# for curved-sky fields with rectangular pixelization.
#
# Note that NaMaster does not support any kind of rectangular pixellization.
# The specific kind supported is pixels defined using the CAR (Plate-Carree)
# projection and with Clenshaw-Curtis weights (i.e. the WCS reference pixel
# must lie on the equator, and the full latitude range must be divided
# exactly into pixels, with one pixel centre at both poles.

# Fields with rectangular pixelization are created from a WCS object that
# defines the geometry of the map.
hdul = fits.open("benchmarks/msk_car.fits")
wcs = WCS(hdul[0].header)
hdul.close()

# Read input maps
# a) Read mask
mask = fits.open("benchmarks/msk_car.fits")[0].data
# b) Read maps
mp_t, mp_q, mp_u = fits.open("benchmarks/mps_car.fits")[0].data
# You can also read and use contaminant maps in the same fashion.
# We'll skip that step here.

# # # #  Create fields
# Create spin-0 field. Pass a WCS structure do define the rectangular pixels.
f0 = nmt.NmtField(mask, [mp_t], wcs=wcs, n_iter=0)
# Create spin-2 field
f2 = nmt.NmtField(mask, [mp_q, mp_u], wcs=wcs, n_iter=0)

# Let's check out the maps.
# First the original map
plt.figure()
plt.title("Original map")
plt.imshow(mp_t, interpolation='nearest', origin='lower')
# Now the map processed after creating the NmtField. Note that `get_maps()`
# will return flattened maps, so you need to undo that.
plt.figure()
plt.title("Map from NmtField")
plt.imshow(f0.get_maps().reshape([mp_t.shape[0], -1]),
           interpolation='nearest', origin='lower')
plt.show()

# You can now use these NmtFields just like you would use HEALPix-based
# ones in terms of power spectrum estimation.
