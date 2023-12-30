from pymaster import nmtlib as lib
import numpy as np
import os
import healpy as hp


def _setenv(name, value, keep=False):
    """Set the named environment variable to the given value. If keep==False
    (the default), existing values are overwritten. If the value is None, then
    it's deleted from the environment. If keep==True, then this function does
    nothing if the variable already has a value."""
    if name in os.environ and keep:
        return
    elif name in os.environ and value is None:
        del os.environ[name]
    elif value is not None:
        os.environ[name] = str(value)


def _getenv(name, default=None):
    """Return the value of the named environment variable, or default
    if it's not set"""
    try:
        return os.environ[name]
    except KeyError:
        return default


# From pixell:
# Initialize DUCC's thread num variable from OMP's if it's not already set.
# This must be done before importing ducc0 for the first time. Doing this
# limits wasted memory from ducc allocating too big a thread pool. For computes
# with many cores, this can save GBs of memory.
_setenv("DUCC0_NUM_THREADS", _getenv("OMP_NUM_THREADS"), keep=True)
import ducc0  # noqa


class _NmtMapInfo(object):
    def __init__(self, nring, theta, phi0, nphi, weight,
                 is_CAR=False, nx_short=-1, nx_full=-1):
        self.nring = nring
        self.theta = theta
        self.phi0 = phi0
        self.nphi = nphi
        self.is_CAR = is_CAR
        self.nx_short = nx_short
        self.nx_full = nx_full
        # Compute the starting point of each ring
        off = np.cumsum(nphi)
        off = np.concatenate([[0], off[:-1]])
        self.offsets = off.astype(np.uint64, copy=False)
        self.weight = weight

    @classmethod
    def from_rectpix(cls, n_theta, theta_min, d_theta,
                     n_phi, d_phi, phi0):
        th = theta_min + np.arange(n_theta+1)*d_theta
        nring = n_theta
        theta = th[:-1].astype(np.float64)
        phi0 = np.zeros(n_theta, dtype=np.float64)+phi0
        nx_short = n_phi
        nx_full = int(2*np.pi/d_phi)
        nphi = np.zeros(n_theta, dtype=np.uint64)+nx_full

        # CC weights according to ducc0
        i_theta_0 = int(theta_min/d_theta+0.5)
        # +1 below because both poles get a ring
        nth_full = int(np.pi/d_theta+0.5)+1
        w = ducc0.sht.experimental.get_gridweights('CC', nth_full)
        w *= d_phi/(2*np.pi)
        weight_th = w[i_theta_0:i_theta_0+n_theta]
        # Alternatively, we could use the pixel area as below
        # weight_th = (np.cos(th[:-1])-np.cos(th[1:]))*d_phi

        return cls(nring=nring, theta=theta, phi0=phi0,
                   nphi=nphi, weight=weight_th, is_CAR=True,
                   nx_short=nx_short, nx_full=nx_full)

    @classmethod
    def from_nside(cls, nside):
        npix = 12*nside*nside
        rings = np.arange(4*nside-1)
        nring = len(rings)
        theta = np.zeros(nring, np.float64)
        phi0 = np.zeros(nring, np.float64)
        nphi = np.zeros(nring, np.uint64)
        rings = rings+1
        northrings = np.where(rings > 2*nside,
                              4*nside-rings,
                              rings)
        # Handle polar cap
        cap = np.where(northrings < nside)[0]
        theta[cap] = 2*np.arcsin(northrings[cap]/(6**0.5*nside))
        nphi[cap] = 4*northrings[cap]
        phi0[cap] = np.pi/(4*northrings[cap])
        # Handle rest
        rest = np.where(northrings >= nside)[0]
        theta[rest] = np.arccos((2*nside-northrings[rest]) *
                                (8*nside/npix))
        nphi[rest] = 4*nside
        phi0[rest] = np.pi/(4*nside) * \
            (((northrings[rest]-nside) & 1) == 0)
        # Above assumed northern hemisphere. Fix southern
        south = np.where(northrings != rings)[0]
        theta[south] = np.pi-theta[south]
        weight = 4*np.pi/npix
        return cls(nring=nring, theta=theta, phi0=phi0,
                   nphi=nphi, weight=weight)

    def pad_map(self, maps):
        if not self.is_CAR:
            return maps
        # Shapes
        pre = maps.shape[:-1]
        shape_short = pre + (self.nring, self.nx_short)
        shape_pad = pre + (self.nring, self.nx_full)
        shape_flat = pre + (self.nring*self.nx_full,)
        # Init to zero
        maps_pad = np.zeros(shape_pad)
        # Copy over
        maps_pad[..., :self.nx_short] = maps.reshape(shape_short)
        # Flatten
        maps_pad = maps_pad.reshape(shape_flat)
        return maps_pad

    def unpad_map(self, maps):
        if not self.is_CAR:
            return maps
        pre = maps.shape[:-1]
        shape_pad = pre + (self.nring, self.nx_full)
        shape_flat = pre + (self.nring*self.nx_short,)
        maps_unpad = maps.reshape(shape_pad)[..., :self.nx_short]
        return maps_unpad.reshape(shape_flat)

    def times_weight(self, m):
        if self.is_CAR:  # Weight depends on theta
            npix = m.shape[-1]
            nx = npix // self.nring
            pre = m.shape[:-1]
            shape_2d = pre + (self.nring, nx)
            shape_flat = pre + (self.nring*nx,)
            return (m.reshape(shape_2d) *
                    self.weight[:, None]).reshape(shape_flat)
        else:  # Weight is constant in healpix
            return m*self.weight

    def dot_map(self, m1, m2):
        return np.sum(self.times_weight(m1*m2))


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
            is_healpix = True
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
            minfo = _NmtMapInfo.from_nside(nside)
        else:
            is_healpix = False
            nside = -1

            d_ra, d_dec = wcs.wcs.cdelt[:2]
            ra0, dec0 = wcs.wcs.crval[:2]
            ix0, iy0 = wcs.wcs.crpix[:2]
            typra = wcs.wcs.ctype[0]
            typdec = wcs.wcs.ctype[1]
            try:
                ny, nx = axes
            except TypeError:
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

            minfo = _NmtMapInfo.from_rectpix(ny, theta_min, dth,
                                             nx, dph, phi0)

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
        self.minfo = minfo

    def __eq__(self, other):
        # Is it the same thing?
        if self is other:
            return True
        # Is it the same type?
        if type(other) is not type(self):
            return False

        # If HEALPix
        if self.is_healpix:
            if not other.is_healpix:
                return False
            if other.nside != self.nside:
                return False
            return True

        # If CAR
        if other.is_healpix:
            return False
        for att in ['npix', 'nx', 'ny', 'theta_min',
                    'theta_max', 'phi0', 'd_theta', 'd_phi']:
            a_this = getattr(self, att, None)
            a_other = getattr(other, att, None)
            if a_this != a_other:
                return False
        return True

    def reform_map(self, maps):
        if not self._map_compatible(maps):
            raise ValueError("Incompatible map!")
        if self.is_healpix:
            return maps

        # Flipping dimensions
        if self.flip_th:
            maps = maps[..., ::-1, :]
        if self.flip_ph:
            maps = maps[..., :, ::-1]
        # Flatten last two dimensions
        maps = maps.reshape(maps.shape[:-2]+(self.npix,))
        return maps

    def _map_compatible(self, mp):
        if self.is_healpix:
            return mp.shape[-1] == self.npix
        else:
            return mp.shape[-2:] == (self.ny, self.nx)

    def get_lmax(self):
        if self.is_healpix:
            return 3*self.nside-1
        else:
            dxmin = min(self.d_theta, self.d_phi)
            return int(np.pi/dxmin)


class AlmInfo(object):
    def __init__(self, lmax):
        self.lmax = lmax
        self.mmax = self.lmax
        m = np.arange(self.mmax+1)
        self.mstart = (m*(2*self.lmax+1-m)//2).astype(np.uint64, copy=False)
        self.nelem = int(np.max(self.mstart) + (self.lmax+1))

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return False
        if self.lmax != other.lmax:
            return False
        return True


def mask_apodization(mask_in, aposize, apotype="C1"):
    """
    Apodizes a mask with an given apodization scale using different methods. \
    A given pixel is determined to be "masked" if its value is 0.

    :param mask_in: input mask, provided as an array of floats \
        corresponding to a HEALPix map in RING order.
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: \
        "C1", "C2" and "Smooth". See the description of the C-function \
        nmt_apodize_mask in the C API documentation for a full description \
        of these methods.
    :return: apodized mask as a HEALPix map
    """
    if apotype not in ['C1', 'C2', 'Smooth']:
        raise ValueError(f"Apodization type {apotype} unknown. "
                         "Choose from ['C1', 'C2', 'Smooth']")

    m = lib.apomask(mask_in.astype("float64"), len(mask_in),
                    aposize, apotype)

    if apotype != 'Smooth':
        return m

    # Smooth
    wt = NmtWCSTranslator(None, mask_in.shape)
    lmax = 3*wt.nside-1
    ainfo = AlmInfo(lmax)
    ls = np.arange(lmax+1)
    beam = np.exp(-0.5*ls*(ls+1)*np.radians(aposize)**2)
    alm = map2alm(np.array([m]), 0, wt.minfo, ainfo, n_iter=3)[0]
    alm = hp.almxfl(alm, beam, mmax=ainfo.mmax)
    m = alm2map(np.array([alm]), 0, wt.minfo, ainfo)[0]
    # Multiply by original mask
    m *= mask_in
    return m


def mask_apodization_flat(mask_in, lx, ly, aposize, apotype="C1"):
    """
    Apodizes a flat-sky mask with an given apodization scale using \
    different methods. A given pixel is determined to be "masked" if \
    its value is 0.

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


def synfast_spherical(nside, cls, spin_arr, beam=None, seed=-1,
                      wcs=None, lmax=None):
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

    if lmax is None:
        lmax = wt.get_lmax()

    spin_arr = np.array(spin_arr).astype(np.int32)
    nfields = len(spin_arr)

    if np.any(spin_arr < 0):
        raise ValueError("Spins must be positive")
    nmap_arr = np.array([1+int(s != 0) for s in spin_arr])
    map_first = np.concatenate([[0], np.cumsum(nmap_arr)[:-1]])
    nmaps = np.sum(nmap_arr)

    ncls = (nmaps * (nmaps + 1)) // 2
    if ncls != len(cls):
        raise ValueError(
            f"Must provide all Cls necessary to simulate all field ({ncls}).")
    lmax_cls = len(cls[0]) - 1
    lmax = min(lmax_cls, lmax)
    ainfo = AlmInfo(lmax)

    # 1. Generate alms
    # Note that, if `new=False` stops being allowed in healpy, we'll need
    # to change the Cl ordering.
    alms = np.array(hp.synalm(cls, lmax=lmax, mmax=lmax, new=False))

    # 2. Multiply by beam
    if beam is not None:
        if len(beam) != nfields:
            raise ValueError("Must provide one beam per field")
        if len(beam[0]) < lmax + 1:
            raise ValueError(f"The beam should be provided to ell = {lmax}")
        beam_per = []
        for ifield, n in enumerate(nmap_arr):
            for i in range(n):
                beam_per.append(beam[ifield])
        alms = np.array([hp.almxfl(alm, bl, mmax=lmax)
                         for alm, bl in zip(alms, beam_per)])

    # 3. SHT back to real space
    maps = np.concatenate([alm2map(alms[i0:i0+n], s, wt.minfo, ainfo)
                           for i0, n, s in zip(map_first, nmap_arr, spin_arr)])

    if wt.is_healpix:
        maps = maps.reshape([nmaps, wt.npix])
    else:
        maps = maps.reshape([nmaps, wt.ny, wt.nx])
        if wt.flip_th:
            maps = maps[:, ::-1, :]
        if wt.flip_ph:
            maps = maps[:, :, ::-1]

    return maps


def _toeplitz_sanity(l_toeplitz, l_exact, dl_band, lmax, fl1, fl2):
    if l_toeplitz > 0:
        if (fl1.pure_e or fl1.pure_b or
                fl2.pure_e or fl2.pure_b):
            raise ValueError("Can't use Toeplitz approximation "
                             "with purification.")
        if (l_exact <= 0) or (dl_band < 0):
            raise ValueError("`l_exact` and `dl_band` must be "
                             "positive numbers")
        if l_exact > l_toeplitz:
            raise ValueError("`l_exact` must be `<= l_toeplitz")
        if ((l_toeplitz >= lmax) or (l_exact >= lmax) or
                (dl_band >= lmax)):
            raise ValueError("`l_toeplitz`, `l_exact` and `dl_band` "
                             "must be smaller than `l_max`")


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


def moore_penrose_pinvh(mat, w_thr):
    if (w_thr is None) or (w_thr <= 0):
        return np.linalg.inv(mat)

    w, v = np.linalg.eigh(mat)
    badw = w < w_thr*np.max(w)
    w_inv = 1./w
    w_inv[badw] = 0.
    pinv = np.dot(v, np.dot(np.diag(w_inv), v.T))
    return pinv


def _ducc_kwargs(spin, map_info, alm_info):
    kwargs = {'spin': spin, 'theta': map_info.theta, 'nphi': map_info.nphi,
              'phi0': map_info.phi0, 'ringstart': map_info.offsets,
              'lmax': alm_info.lmax, 'mmax': alm_info.mmax,
              'mstart': alm_info.mstart, 'nthreads': 0}
    return kwargs


def _alm2map_ducc0(alm, spin, map_info, alm_info):
    kwargs = _ducc_kwargs(spin, map_info, alm_info)
    maps = ducc0.sht.experimental.synthesis(alm=alm, **kwargs)
    return maps


def _map2alm_ducc0(map, spin, map_info, alm_info):
    kwargs = _ducc_kwargs(spin, map_info, alm_info)
    alm = ducc0.sht.experimental.adjoint_synthesis(
        map=map_info.times_weight(map), **kwargs)
    return alm


def _alm2map_healpy(alm, spin, map_info, alm_info):
    if map_info.is_CAR:
        raise ValueError("Can't use healpy for CAR maps")
    kwargs = {'lmax': alm_info.lmax, 'mmax': alm_info.mmax}
    if spin == 0:
        map = [hp.alm2map(alm, mside=map_info.nside, verbose=False,
                          **kwargs)]
    else:
        map = hp.alm2map_spin(alm, map_info.nside, spin, **kwargs)
    return np.array(map)


def _map2alm_healpy(map, spin, map_info, alm_info):
    if map_info.is_CAR:
        raise ValueError("Can't use healpy for CAR maps")
    kwargs = {'lmax': alm_info.lmax, 'mmax': alm_info.mmax}
    if spin == 0:
        alm = [hp.map2alm(map, **kwargs)]
    else:
        alm = hp.map2alm_spin(map, spin, **kwargs)
    return np.array(alm)


_m2a_d = {'ducc': _map2alm_ducc0,
          'healpy': _map2alm_healpy}
_a2m_d = {'ducc': _alm2map_ducc0,
          'healpy': _alm2map_healpy}


def map2alm(map, spin, map_info, alm_info, n_iter=0,
            sht_calculator='ducc'):
    map = map_info.pad_map(map)
    m2a = _m2a_d[sht_calculator]
    a2m = _a2m_d[sht_calculator]
    alm = m2a(map, spin, map_info, alm_info)
    for i in range(n_iter):
        dmap = a2m(alm, spin, map_info, alm_info)-map
        alm -= m2a(dmap, spin, map_info, alm_info)
    return alm


def alm2map(alm, spin, map_info, alm_info,
            sht_calculator='ducc'):
    a2m = _a2m_d[sht_calculator]
    map = a2m(alm, spin, map_info, alm_info)
    return map_info.unpad_map(map)
