from pymaster import nmtlib as lib
import numpy as np


class NmtWCSTranslator(object):
    """
    This class takes care of interpreting a WCS object in \
    terms of a Clenshaw-Curtis grid.

    :param wcs: a WCS object (see \
        http://docs.astropy.org/en/stable/wcs/index.html).
    :param axes: shape of the maps you want to analyze.
    """

    def __init__(self, wcs, axes):
        if wcs is None:
            is_healpix = 1
            nside = 2
            while 12 * nside * nside != axes[0]:
                nside *= 2
                if nside > 65536:
                    raise ValueError("Something is wrong "
                                     "with your input arrays")
            npix = 12*nside*nside
            flip_th = False
            flip_ph = False
            theta_min = -1
            theta_max = -1
            phi0 = -1
            dth = -1
            dph = -1
            nx = -1
            ny = -1
        else:
            is_healpix = 0
            nside = -1

            d_ra, d_dec = wcs.wcs.cdelt[:2]
            ra0, dec0 = wcs.wcs.crval[:2]
            ix0, iy0 = wcs.wcs.crpix[:2]
            typra = wcs.wcs.ctype[0]
            typdec = wcs.wcs.ctype[1]
            try:
                ny, nx = axes
            except:
                raise ValueError("Input maps must be 2D if not HEALPix")
            npix = ny*nx

            dth = np.fabs(np.radians(d_dec))
            dph = np.fabs(np.radians(d_ra))

            # Check if projection type is CAR
            if not ((typra[-3:] == 'CAR') and (typdec[-3:] == 'CAR')):
                raise ValueError("Maps must have CAR pixelization")

            # Check if reference pixel is consistent with CC
            if np.fabs(dec0) > 1E-3:
                raise ValueError("Reference pixel must be at the equator")

            # Check d_ra, d_dec are CC
            if (np.fabs(round(2*np.pi/dph) - 2 * np.pi / dph) > 0.01) or \
               (np.fabs(round(np.pi/dth) - np.pi / dth) > 0.01):
                raise ValueError("The pixels should divide the sphere exactly")

            # Is colatitude decreasing? (i.e. is declination increasing?)
            flip_th = d_dec > 0
            flip_ph = d_ra < 0

            # Get map edges
            # Theta
            coord0 = np.zeros([1, len(wcs.wcs.crpix)])
            coord1 = coord0.copy()
            coord1[0, 1] = ny-1
            edges = [wcs.wcs_pix2world(coord0, 0)[0, 1],
                     wcs.wcs_pix2world(coord1, 0)[0, 1]]
            theta_min = np.radians(90-max(edges))
            theta_max = np.radians(90-min(edges))
            # Phi
            coord0 = np.zeros([1, len(wcs.wcs.crpix)])
            if flip_ph:
                coord0[0, 0] = nx-1
            phi0 = wcs.wcs_pix2world(coord0, 0)[0, 0]
            if np.isnan(phi0):
                raise ValueError("There is something wrong with the azimuths")
            phi0 = np.radians(phi0)

            # Check if ix0,iy0 + ra0, dec0 mean this is CC
            if (theta_min < 0) or (theta_max > np.pi) or np.isnan(edges).any():
                raise ValueError("The colatitude map edges are "
                                 "outside the sphere")

            if np.fabs(nx*d_ra) > 360:
                raise ValueError("Seems like you're wrapping the "
                                 "sphere more than once")

        # Store values
        self.is_healpix = is_healpix
        self.nside = nside
        self.npix = npix
        self.nx = nx
        self.ny = ny
        self.flip_th = flip_th
        self.flip_ph = flip_ph
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.d_theta = dth
        self.d_phi = dph
        self.phi0 = phi0

    def get_lmax(self):
        return lib.get_lmax_py(self.is_healpix, self.nside, self.nx, self.ny,
                               self.d_phi, self.d_theta, self.phi0,
                               self.theta_max)


def mask_apodization(mask_in, aposize, apotype="C1"):
    """
    Apodizes a mask with an given apodization scale using different methods.

    :param mask_in: input mask, provided as an array of floats \
        corresponding to a HEALPix map in RING order.
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: \
        "C1", "C2" and "Smooth". See the description of the C-function \
        nmt_apodize_mask in the C API documentation for a full description \
        of these methods.
    :return: apodized mask as a HEALPix map
    """
    return lib.apomask(mask_in.astype("float64"), len(mask_in),
                       aposize, apotype)


def mask_apodization_flat(mask_in, lx, ly, aposize, apotype="C1"):
    """
    Apodizes a flat-sky mask with an given apodization scale using \
    different methods.

    :param mask_in: input mask, provided as a 2D array (ny,nx) of floats.
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: "C1", \
       "C2" and "Smooth". See the description of the C-function \
       nmt_apodize_mask in the C API documentation for a full \
       description of these methods.
    :return: apodized mask as a 2D array (ny,nx)
    """
    if mask_in.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    nx = len(mask_in[0])
    ny = len(mask_in)
    mask_apo_flat = lib.apomask_flat(
        nx, ny, lx, ly, mask_in.flatten().astype("float64"),
        nx * ny, aposize, apotype
    )
    return mask_apo_flat.reshape([ny, nx])


def synfast_spherical(nside, cls, spin_arr, beam=None, seed=-1, wcs=None):
    """
    Generates a full-sky Gaussian random field according to a given \
    power spectrum. This function should produce outputs similar to \
    healpy's synfast.

    :param int nside: HEALpix resolution parameter. If you want \
        rectangular pixel maps, ignore this parameter and pass a \
        WCS object as `wcs` (see below).
    :param array-like cls: array containing power spectra. Shape \
        should be [n_cls][n_ell], where n_cls is the number of power \
        spectra needed to define all the fields. This should be \
        n_cls = n_maps * (n_maps + 1) / 2, where n_maps is the total \
        number of maps required (1 for each spin-0 field, 2 for each \
        spin>0 field). Power spectra must be provided only for the \
        upper-triangular part in row-major order (e.g. if n_maps is \
        3, there will be 6 power spectra ordered as \
        [1-1,1-2,1-3,2-2,2-3,3-3].
    :param array-like spin_arr: array containing the spins of all the \
        fields to generate.
    :param beam array-like: 2D array containing the instrumental beam \
        of each field to simulate (the output map(s) will be convolved \
        with it)
    :param int seed: RNG seed. If negative, will use a random seed.
    :param wcs: a WCS object \
        (see http://docs.astropy.org/en/stable/wcs/index.html).
    :return: a number of full-sky maps (1 for each spin-0 field, 2 for \
        each spin-2 field).
    """
    if seed < 0:
        seed = np.random.randint(50000000)

    if wcs is None:
        wtshape = [12*nside*nside]
    else:
        n_ra = int(round(360./np.fabs(wcs.wcs.cdelt[0])))
        n_dec = int(round(180./np.fabs(wcs.wcs.cdelt[1])))+1
        wtshape = [n_dec, n_ra]
    wt = NmtWCSTranslator(wcs, wtshape)

    if wcs is not None:  # Check that theta_min and theta_max are 0 and pi
        if (np.fabs(wt.theta_min) > 1E-4) or \
           (np.fabs(wt.theta_max - np.pi) > 1E-4):
            raise ValueError("Given your WCS, the map wouldn't cover the "
                             "whole sphere exactly")

    spin_arr = np.array(spin_arr).astype(np.int32)
    nfields = len(spin_arr)

    if np.any(spin_arr < 0):
        raise ValueError("Spins must be positive")
    nmaps = int(1 * np.sum(spin_arr == 0) + 2 * np.sum(spin_arr != 0))

    ncls = (nmaps * (nmaps + 1)) // 2
    if ncls != len(cls):
        raise ValueError(
            "Must provide all Cls necessary to simulate all "
            "fields (%d)." % ncls
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

    data = lib.synfast_new(wt.is_healpix, wt.nside, wt.nx, wt.ny,
                           wt.d_phi, wt.d_theta, wt.phi0, wt.theta_max,
                           spin_arr, seed, cls, beam, nmaps * wt.npix)

    if wt.is_healpix:
        maps = data.reshape([nmaps, wt.npix])
    else:
        maps = data.reshape([nmaps, wt.ny, wt.nx])
        if wt.flip_th:
            maps = maps[:, ::-1, :]
        if wt.flip_ph:
            maps = maps[:, :, ::-1]

    return maps


def synfast_flat(nx, ny, lx, ly, cls, spin_arr, beam=None, seed=-1):
    """
    Generates a flat-sky Gaussian random field according to a given power \
    spectrum. This function is the flat-sky equivalent of healpy's synfast.

    :param int nx: number of pixels in the x-axis
    :param int ny: number of pixels in the y-axis
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param array-like cls: array containing power spectra. Shape should be \
        [n_cls][n_ell], where n_cls is the number of power spectra needed \
        to define all the fields. This should be \
        n_cls = n_maps * (n_maps + 1) / 2, where n_maps is the total number \
        of maps required (1 for each spin-0 field, 2 for each spin>0 field). \
        Power spectra must be provided only for the upper-triangular part in \
        row-major order (e.g. if n_maps is 3, there will be 6 power spectra \
        ordered as [1-1,1-2,1-3,2-2,2-3,3-3].
    :param array-like spin_arr: array containing the spins of all the fields \
        to generate.
    :param beam array-like: 2D array containing the instrumental beam of each \
        field to simulate (the output map(s) will be convolved with it)
    :param int seed: RNG seed. If negative, will use a random seed.
    :return: a number of arrays (1 for each spin-0 field, 2 for each \
        spin-2 field) of size (ny,nx) containing the simulated maps.
    """
    if seed < 0:
        seed = np.random.randint(50000000)

    spin_arr = np.array(spin_arr).astype(np.int32)
    nfields = len(spin_arr)

    if np.any(spin_arr < 0):
        raise ValueError("Spins must be positive")
    nmaps = int(1 * np.sum(spin_arr == 0) + 2 * np.sum(spin_arr != 0))

    ncls = (nmaps * (nmaps + 1)) // 2
    if ncls != len(cls):
        raise ValueError(
            "Must provide all Cls necessary to simulate all "
            "fields (%d)." % ncls
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
