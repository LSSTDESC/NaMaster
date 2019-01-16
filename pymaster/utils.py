from pymaster import nmtlib as lib
import numpy as np


def mask_apodization(mask_in, aposize, apotype="C1"):
    """
    Apodizes a mask with an given apodization scale using different methods.

    :param mask_in: input mask, provided as an array of floats corresponding to a HEALPix map in RING order.
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: "C1", "C2" and "Smooth". See the description of the C-function nmt_apodize_mask in the C API documentation for a full description of these methods.
    :return: apodized mask as a HEALPix map
    """
    return lib.apomask(mask_in.astype("float64"), len(mask_in), aposize, apotype)


def mask_apodization_flat(mask_in, lx, ly, aposize, apotype="C1"):
    """
    Apodizes a flat-sky mask with an given apodization scale using different methods.

    :param mask_in: input mask, provided as a 2D array (ny,nx) of floats.
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: "C1", "C2" and "Smooth". See the description of the C-function nmt_apodize_mask in the C API documentation for a full description of these methods.
    :return: apodized mask as a 2D array (ny,nx)
    """
    if mask_in.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    nx = len(mask_in[0])
    ny = len(mask_in)
    mask_apo_flat = lib.apomask_flat(
        nx, ny, lx, ly, mask_in.flatten().astype("float64"), nx * ny, aposize, apotype
    )
    return mask_apo_flat.reshape([ny, nx])


def synfast_spherical(nside, cls, spin_arr, beam=None, seed=-1,wcs=None,map_shape=None):
    """
    Generates a full-sky Gaussian random field according to a given power spectrum. This function should produce outputs similar to healpy's synfast.

    :param int nside: HEALpix resolution parameter. If you want rectangular pixel maps, ignore this parameter and pass a WCS object as `wcs` (see below).
    :param array-like cls: array containing power spectra. Shape should be [n_cls][n_ell], where n_cls is the number of power spectra needed to define all the fields. This should be n_cls = n_maps * (n_maps + 1) / 2, where n_maps is the total number of maps required (1 for each spin-0 field, 2 for each spin-2 field). Power spectra must be provided only for the upper-triangular part in row-major order (e.g. if n_maps is 3, there will be 6 power spectra ordered as [1-1,1-2,1-3,2-2,2-3,3-3]. 
    :param array-like spin_arr: array containing the spins of all the fields to generate.
    :param beam array-like: 2D array containing the instrumental beam of each field to simulate (the output map(s) will be convolved with it)
    :param int seed: RNG seed. If negative, will use a random seed.
    :param wcs: a WCS object (see http://docs.astropy.org/en/stable/wcs/index.html).
    :param map_shape: a tuple `(ny,nx)` with the size of the map in the longitude and latitude directions. Must be provided if `wcs` is not `None` (i.e. if using rectangular pixels).
    :return: a number of full-sky maps (1 for each spin-0 field, 2 for each spin-2 field).
    """
    if seed < 0:
        seed = np.random.randint(50000000)

    if wcs is None :
        is_healpix=1
        npix=12*nside*nside
        #Make all WCS variables dummy
        nx=-1; ny=-1;
        delta_phi=-1; delta_theta=-1;
        phi0=-1; theta0=-1;
    else :
        is_healpix=0
        try:
            ny,nx=map_shape
        except:
            raise ValueError("Map dimensions must be provided if using rectangular pixels")
        npix=nx*ny
        delta_phi,delta_theta=wcs.wcs.cdelt
        phi0,theta0=wcs.wcs.crval

    spin_arr = np.array(spin_arr).astype(np.int32)
    nfields = len(spin_arr)

    if np.sum((spin_arr == 0) | (spin_arr == 2)) != nfields:
        raise ValueError("All spins must be 0 or 2")
    nmaps = int(1 * np.sum(spin_arr == 0) + 2 * np.sum(spin_arr == 2))

    ncls = (nmaps * (nmaps + 1)) // 2
    if ncls != len(cls):
        raise ValueError(
            "Must provide all Cls necessary to simulate all fields (%d)." % ncls
        )
    lmax = len(cls[0]) - 1

    if beam is None:
        beam = np.ones([nfields, lmax + 1])
    else:
        if len(beam) != nfields:
            raise ValueError("Must provide one beam per field")
        if len(beam[0]) != lmax + 1:
            raise ValueError(
                "The beam should have as many multipoles as the power spectrum"
            )

    data = lib.synfast_new(is_healpix, nside, nx, ny,
                           delta_phi, delta_theta, theta0, phi0,
                           spin_arr, seed, cls, beam, nmaps * npix)

    if is_healpix :
        maps = data.reshape([nmaps, npix])
    else :
        maps = data.reshape([nmaps, ny, nx])

    return maps


def synfast_flat(nx, ny, lx, ly, cls, spin_arr, beam=None, seed=-1):
    """
    Generates a flat-sky Gaussian random field according to a given power spectrum. This function is the flat-sky equivalent of healpy's synfast.

    :param int nx: number of pixels in the x-axis
    :param int ny: number of pixels in the y-axis
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param array-like cls: array containing power spectra. Shape should be [n_cls][n_ell], where n_cls is the number of power spectra needed to define all the fields. This should be n_cls = n_maps * (n_maps + 1) / 2, where n_maps is the total number of maps required (1 for each spin-0 field, 2 for each spin-2 field). Power spectra must be provided only for the upper-triangular part in row-major order (e.g. if n_maps is 3, there will be 6 power spectra ordered as [1-1,1-2,1-3,2-2,2-3,3-3]. 
    :param array-like spin_arr: array containing the spins of all the fields to generate.
    :param beam array-like: 2D array containing the instrumental beam of each field to simulate (the output map(s) will be convolved with it)
    :param int seed: RNG seed. If negative, will use a random seed.
    :return: a number of arrays (1 for each spin-0 field, 2 for each spin-2 field) of size (ny,nx) containing the simulated maps.
    """
    if seed < 0:
        seed = np.random.randint(50000000)

    spin_arr = np.array(spin_arr).astype(np.int32)
    nfields = len(spin_arr)

    if np.sum((spin_arr == 0) | (spin_arr == 2)) != nfields:
        raise ValueError("All spins must be 0 or 2")
    nmaps = int(1 * np.sum(spin_arr == 0) + 2 * np.sum(spin_arr == 2))

    ncls = (nmaps * (nmaps + 1)) // 2
    if ncls != len(cls):
        raise ValueError(
            "Must provide all Cls necessary to simulate all fields (%d)." % ncls
        )
    lmax = len(cls[0]) - 1

    if beam is None:
        beam = np.ones([nfields, lmax + 1])
    else:
        if len(beam) != nfields:
            raise ValueError("Must provide one beam per field")
        if len(beam[0]) != lmax + 1:
            raise ValueError(
                "The beam should have as many multipoles as the power spectrum"
            )

    data = lib.synfast_new_flat(
        nx, ny, lx, ly, spin_arr, seed, cls, beam, nmaps * nx * ny
    )

    maps = data.reshape([nmaps, ny, nx])

    return maps
