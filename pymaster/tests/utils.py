import numpy as np


def normdiff(v1, v2):
    return np.amax(np.fabs(v1-v2))


def read_flat_map(filename, i_map=0):
    """
    Reads a flat-sky map and the details of its pixelization scheme.
    The latter are returned as a FlatMapInfo object.
    i_map : map to read. If -1, all maps will be read.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    hdul = fits.open(filename)
    w = WCS(hdul[0].header)

    maps = hdul[i_map].data
    ny, nx = maps.shape
    hdul.close()

    return w, maps
