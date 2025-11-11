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


try:
    # From pixell:
    # Initialize DUCC's thread num variable from OMP's if it's not already set.
    # This must be done before importing ducc0 for the first time. Doing this
    # limits wasted memory from ducc allocating too big a thread pool. For
    # computes with many cores, this can save GBs of memory.
    _setenv("DUCC0_NUM_THREADS", _getenv("OMP_NUM_THREADS"), keep=True)
    import ducc0  # noqa
    HAVE_DUCC = True
except ModuleNotFoundError:
    HAVE_DUCC = False


class NmtParams(object):
    """ Class that holds the values of several parameters controlling the
    default behaviour of different NaMaster methods. Currently these are:

    - ``sht_calculator``: the software package to use to calculate \
      spherical harmonic transforms. Defaults to \
      `ducc <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_, with the other \
      choice being `healpy <https://healpy.readthedocs.io/en/latest/>`_.
    - ``n_iter_default``: number of iterations to use when computing \
      `map2alm` spherical harmonic transforms of most maps. Defaults to 3.
    - ``n_iter_mask_default``: number of iterations to use when computing \
      `map2alm` spherical harmonic transforms of masks. Defaults to 3.
    - ``tol_pinv_default``: relative eigenvalue threshold to use when \
      computing matrix pseudo-inverses.

    All variables can be changed using the ``set_`` methods described below,
    and their current values can be checked with :meth:`get_default_params`.
    Note that, except for ``sht_calculator``, all of these variables can
    be tweaked when calling various NaMaster functions. The
    values stored in this object only hold the values they default to if
    they are not set in those function calls.
    """
    def __init__(self):
        self.sht_calculator = 'ducc' if HAVE_DUCC else 'healpy'
        self.n_iter_default = 3
        self.n_iter_mask_default = 3
        self.tol_pinv_default = 1E-10


nmt_params = NmtParams()


def set_sht_calculator(calc_name):
    """ Select default spherical harmonic transform calculator.

    Args:
        calc_name (:obj:`str`): Calculator name. Allowed options
            are ``'ducc'`` or ``'healpy'``.
    """
    if calc_name not in ['ducc', 'healpy']:
        raise KeyError("SHT calculator must be 'ducc' or 'healpy'")
    if (calc_name == 'ducc') and (not HAVE_DUCC):
        raise ValueError("ducc not found. "
                         "Select a different SHT calculator.")
    nmt_params.sht_calculator = calc_name


def set_n_iter_default(n_iter, mask=False):
    """ Select the number of Jacobi iterations to use when computing
    `map2alm` spherical harmonic transforms.

    Args:
        n_iter (:obj:`int`): Number of iterations.
        mask (:obj:`bool`): If ``True``, ``n_iter`` will be the
            number of iterations used when computing mask transforms.
            Otherwise, it will be the number used for all other
            transforms.
    """
    if n_iter < 0:
        raise ValueError('n_iter must be positive.')
    if mask:
        nmt_params.n_iter_mask_default = int(n_iter)
    else:
        nmt_params.n_iter_default = int(n_iter)


def set_tol_pinv_default(tol_pinv):
    """ Select the relative eigenvalue threshold to use when
    computing matrix pseudo-inverses. Check the docstring of
    :meth:`moore_penrose_pinvh`.

    Args:
        tol_pinv (:obj:`float`): Relative eigenvalue threshold.
    """
    if (tol_pinv > 1) or (tol_pinv < 0):
        raise ValueError('tol_pinv must be between 0 and 1.')
    nmt_params.tol_pinv_default = tol_pinv


def get_default_params():
    """ Returns a dictionary with the current values of all
    default parameters.
    """
    return {k: getattr(nmt_params, k)
            for k in ['sht_calculator',
                      'n_iter_default',
                      'n_iter_mask_default',
                      'tol_pinv_default']}


class _SHTInfo(object):
    def __init__(self, nring, theta, phi0, nphi, weight,
                 is_CAR=False, nx_short=-1, nx_full=-1, nside=-1):
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
        self.nside = nside

    @classmethod
    def from_rectpix(cls, n_theta, theta_min, d_theta,
                     n_phi, d_phi, phi0):
        th = theta_min + np.arange(n_theta+1)*d_theta
        nring = n_theta
        theta = th[:-1].astype(np.float64)
        phi0 = np.zeros(n_theta, dtype=np.float64)+phi0
        nx_short = n_phi
        nx_full = int(2*np.pi/d_phi+0.5)
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
                   nphi=nphi, weight=weight, nside=nside)

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


class NmtMapInfo(object):
    """
    This object contains information about the pixelization of
    a curved-sky map. :obj:`NmtMapInfo` objects can be compared
    with one another using ``==`` and ``!=`` to check for fields
    with compatible pixelizations.

    Args:
        wcs (`WCS`): a WCS object (see the `astropy
            <http://docs.astropy.org/en/stable/wcs/index.html>`_
            documentation). If ``None``, HEALPix pixelization is
            assumed, and ``axes`` should be a 1-element
            sequence with the number of pixels of the map.
        axes (`array`): shape of the maps (length-2 for CAR maps,
            length-1 for HEALPix).
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
            si = _SHTInfo.from_nside(nside)
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
            except ValueError:
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

            if np.fabs(nx*d_ra) > 360+0.1*np.fabs(d_ra):
                raise ValueError("Seems like you're wrapping the "
                                 "sphere more than once")

            si = _SHTInfo.from_rectpix(ny, theta_min, dth,
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
        self.si = si

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
        """ Modifies a map to make it compatible with the
        standards used by NaMaster for map manipulation (e.g.
        spherical harmonic transforms). This includes
        flattening 2D maps, and flipping their coordinate
        axes if required by their associated WCS information.
        HEALPix maps are unmodified.

        Args:
            maps (`array`): 2D (for HEALPix) or 3D (for CAR)
                array containing a set of maps.

        Returns:
            (`array`): Reformed map.
        """
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
        """ Returns the default maximum multipole :math:`\\ell_{\\rm max}`
        associated with this pixelization scheme. This will be
        :math:`3 N_{\\rm side}-1` for HEALPix or
        :math:`\\pi/{\\rm min}(\\Delta\\theta,\\Delta\\varphi)` for
        CAR maps (with :math:`\\Delta\\theta` and :math:`\\Delta\\varphi`
        the constant intervals of colatitude and azimuth in radians).
        """
        if self.is_healpix:
            return 3*self.nside-1
        else:
            dxmin = min(self.d_theta, self.d_phi)
            return int(np.pi/dxmin)


class NmtAlmInfo(object):
    """ Object containing information useful to manipulate sets of
    spherical harmonic coefficients :math:`a_{\\ell m}`.

    Args:
        lmax (:obj:`int`): Maximum multipole :math:`\\ell_{\\rm max}`
            to which the :math:`a_{\\ell m}` s are calculated.
    """
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
    """ Apodizes a mask with an given apodization scale using different
    methods. A given pixel is determined to be "masked" if its value is 0.
    This method only works for HEALPix maps. Three apodization methods are
    currently implemented:

    - **"C1"** apodization: all pixels are multiplied by a factor :math:`f` \
      given by

      .. math::
        f=\\left\\{
        \\begin{array}{cc}
            x-\\sin(2\\pi x)/(2\\pi) & x<1\\\\
            1 & {\\rm otherwise}
        \\end{array}
        \\right..

      where :math:`x=\\sqrt{(1-\\cos\\theta)/(1-\\cos\\theta_*)}`, with
      :math:`\\theta_*` the apodization scale, and :math:`\\theta` the
      separation between a pixel and its nearest masked pixel (i.e. where
      the mask takes a zero value).
    - **"C2"** apodization: similar to "C1", but using the apodization
      function:

      .. math::
        f=\\left\\{
        \\begin{array}{cc}
            \\frac{1}{2}\\left[1-\\cos(\\pi x)\\right] & x<1\\\\
            1 & {\\rm otherwise}
        \\end{array}
        \\right..

    - **"Smooth"** apodization: this is carried out in three steps:

      1. All pixels within a disk of radius :math:`2.5\\theta_*` of a
         masked pixel are masked.
      2. The resulting map is smoothed with a Gaussian window function
         with standard deviation :math:`\\theta_*`.
      3. The resulting map is multiplied by the original mask to ensure
         that all pixels that were previously masked are still masked.

    .. note::
      Note that, confusingly, the definition of the "C1" and "C2"
      apodization windows above is the opposite of the similar :math:`C^1`
      and :math:`C^2` functions in
      `Grain et al. 2009 <https://arxiv.org/abs/0903.2350>`_. This is due
      to a typo in the early development of NaMaster, which may be too
      disruptive to fix at this stage. This may be modified in a future
      release. Until then, users should rely on the documentation above,
      and not the defitions in Grain et al. 2009.

    Args:
        mask_in (`array`): Input mask, provided as an array of floats
            corresponding to a HEALPix map in RING order.
        aposize (:obj:`float`): Apodization scale in degrees.
        apotype (:obj:`str`): Apodization type.

    Returns:
        (`array`): Apodized mask.
    """
    if apotype not in ['C1', 'C2', 'Smooth']:
        raise ValueError(f"Apodization type {apotype} unknown. "
                         "Choose from ['C1', 'C2', 'Smooth']")

    m = lib.apomask(mask_in.astype("float64"), len(mask_in),
                    aposize, apotype)

    if apotype != 'Smooth':
        return m

    # Smooth
    minfo = NmtMapInfo(None, mask_in.shape)
    lmax = 3*minfo.nside-1
    ainfo = NmtAlmInfo(lmax)
    ls = np.arange(lmax+1)
    beam = np.exp(-0.5*ls*(ls+1)*np.radians(aposize)**2)
    alm = map2alm(np.array([m]), 0, minfo, ainfo,
                  n_iter=3)[0]
    alm = hp.almxfl(alm, beam, mmax=ainfo.mmax)
    m = alm2map(np.array([alm]), 0, minfo, ainfo)[0]
    # Multiply by original mask
    m *= mask_in
    return m


def mask_apodization_flat(mask_in, lx, ly, aposize, apotype="C1"):
    """ Apodizes a flat-sky mask. See the docstrings of
    :meth:`mask_apodization` for a description of the different
    methods implemented.

    Args:
        mask_in (`array`): Input mask, provided as a 2D array of
            shape ``(ny, nx)``.
        lx (:obj:`float`): Patch size in the x axis (radians).
        ly (:obj:`float`): Patch size in the y axis (radians).
        aposize (:obj:`float`): Apodization scale in degrees.
        apotype (:obj:`str`): Apodization type.

    Returns:
        (`array`): Apodized mask.
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
    """ Generates full-sky Gaussian random fields according to a given
    power spectrum. This function should produce outputs similar to
    healpy's `synfast <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.synfast.html>`_.

    Args:
        nside (:obj:`int`): HEALpix resolution parameter.
            If you want rectangular pixel maps, ignore this parameter
            and pass a WCS object as ``wcs`` (see below).
        cls (`array`): Array contiaining power spectra. Shape
            should be ``(n_cls, n_ell)``, where ``n_cls`` is the
            number of power spectra needed to define all the fields.
            This should be ``n_cls = n_maps * (n_maps + 1) / 2``,
            where ``n_maps`` is the total number of maps required
            (1 for each spin-0 field, 2 for each spin>0 field).
            Power spectra must be provided only for the upper-triangular
            part of the full power spectrum matrix in row-major order
            (e.g. if ``n_maps=3``, there will be 6 power spectra ordered
            as [1x1, 1x2, 1x3, 2x2, 2x3, 3x3]).
        spin_arr (`array`): Array containing the spins of all the
            fields to generate.
        beam (`array`): 2D array containing the instrumental beam
            of each field to simulate (the output map(s) will be
            convolved with it).
        seed (:obj:`int`): Seed for the random number generator. If
            negative, a random seed will be used.
        wcs (`WCS`): A WCS object if using rectangular pixels (see `the astropy
            documentation
            <http://docs.astropy.org/en/stable/wcs/index.html>`_).
        lmax (:obj:`int`): Maximum multipole up to which the spherical
            harmonic coefficients of the maps will be generated.

    Returns:
        (`array`): A set of full-sky maps (1 for each spin-0 field, 2 for \
            each spin-s field).
    """  # noqa
    if seed < 0:
        seed = np.random.randint(50000000)

    if wcs is None:
        wtshape = [12*nside*nside]
    else:
        n_ra = int(round(360./np.fabs(wcs.wcs.cdelt[0])))
        n_dec = int(round(180./np.fabs(wcs.wcs.cdelt[1])))+1
        wtshape = [n_dec, n_ra]
    minfo = NmtMapInfo(wcs, wtshape)

    if wcs is not None:  # Check that theta_min and theta_max are 0 and pi
        if (np.fabs(minfo.theta_min) > 1E-4) or \
           (np.fabs(minfo.theta_max - np.pi) > 1E-4):
            raise ValueError("Given your WCS, the map wouldn't cover the "
                             "whole sphere exactly")

    if lmax is None:
        lmax = minfo.get_lmax()

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
    ainfo = NmtAlmInfo(lmax)

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
    maps = np.concatenate([alm2map(alms[i0:i0+n], s, minfo, ainfo)
                           for i0, n, s in zip(map_first, nmap_arr, spin_arr)])

    if minfo.is_healpix:
        maps = maps.reshape([nmaps, minfo.npix])
    else:
        maps = maps.reshape([nmaps, minfo.ny, minfo.nx])
        if minfo.flip_th:
            maps = maps[:, ::-1, :]
        if minfo.flip_ph:
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
    Generates flat-sky Gaussian random fields according to a given
    power spectrum. This is the flat-sky equivalent of
    :meth:`synfast_spherical`.

    Args:
        nx (:obj:`int`): Number of pixels in the x axis.
        ny (:obj:`int`): Number of pixels in the y axis.
        lx (:obj:`float`): Patch size in the x axis (radians).
        ly (:obj:`float`): Patch size in the y axis (radians).
        cls (`array`): Array contiaining power spectra. Shape
            should be ``(n_cls, n_ell)``, where ``n_cls`` is the
            number of power spectra needed to define all the fields.
            This should be ``n_cls = n_maps * (n_maps + 1) / 2``,
            where ``n_maps`` is the total number of maps required
            (1 for each spin-0 field, 2 for each spin>0 field).
            Power spectra must be provided only for the upper-triangular
            part of the full power spectrum matrix in row-major order
            (e.g. if ``n_maps=3``, there will be 6 power spectra ordered
            as [1x1, 1x2, 1x3, 2x2, 2x3, 3x3]). The power spectra are
            assumed to be sampled at all integer multipoles from
            :math:`\\ell=0` to ``n_ell-1``.
        spin_arr (`array`): Array containing the spins of all the
            fields to generate.
        beam (`array`): 2D array containing the instrumental beam
            of each field to simulate (the output map(s) will be
            convolved with it).
        seed (:obj:`int`): Seed for the random number generator. If
            negative, a random seed will be used.

    Returns:
        (`array`): An array of flat-sky maps (1 for each spin-0 field, 2 for \
            each spin-s field) with shape ``(nmaps, ny, nx)``.
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


def moore_penrose_pinvh(mat, tol_pinv):
    """ Compute the Moore-Penrose pseudo-inverse of a
    Hermitian matrix. This is done by diagonalising
    the matrix, setting the inverse of all eigenvales
    below a given threshold to zero, and reconstructing
    the inverse matrix from the modified eigenvalues.
    The inverse of all eigenvalues smaller than a factor
    ``tol_pinv`` times the largest eigenvalue will be set
    to zero. If ``tol_pinv <= 0``, the standard inverse
    is computed.

    Args:
        mat (`array`): 2D Hermitian matrix.
        tol_pinv (:obj:`float`): Relative eigenvalue
            threshold.

    Returns:
        (`array`): Pseudo-inverse of input matrix.
    """
    if (tol_pinv is None) or (tol_pinv <= 0):
        return np.linalg.inv(mat)

    w, v = np.linalg.eigh(mat)
    goodw = w >= tol_pinv*np.max(w)
    w_inv = np.zeros_like(w)
    w_inv[goodw] = 1./w[goodw]
    pinv = np.dot(v, np.dot(np.diag(w_inv), v.T))
    return pinv


def _ducc_kwargs(spin, sht_info, alm_info):
    kwargs = {'spin': spin, 'theta': sht_info.theta, 'nphi': sht_info.nphi,
              'phi0': sht_info.phi0, 'ringstart': sht_info.offsets,
              'lmax': alm_info.lmax, 'mmax': alm_info.mmax,
              'mstart': alm_info.mstart, 'nthreads': 0}
    return kwargs


def _alm2map_ducc0(alm, spin, sht_info, alm_info):
    kwargs = _ducc_kwargs(spin, sht_info, alm_info)
    maps = ducc0.sht.experimental.synthesis(alm=alm, **kwargs)
    return maps


def _map2alm_ducc0(map, spin, sht_info, alm_info):
    kwargs = _ducc_kwargs(spin, sht_info, alm_info)
    alm = ducc0.sht.experimental.adjoint_synthesis(
        map=sht_info.times_weight(map), **kwargs)
    return alm


def _alm2map_healpy(alm, spin, sht_info, alm_info):
    if sht_info.is_CAR:
        raise ValueError("Can't use healpy for CAR maps")
    kwargs = {'lmax': alm_info.lmax, 'mmax': alm_info.mmax}
    if spin == 0:
        map = [hp.alm2map(alm[0], nside=sht_info.nside, **kwargs)]
    else:
        map = hp.alm2map_spin(alm, sht_info.nside, spin, **kwargs)
    return np.array(map)


def _map2alm_healpy(map, spin, sht_info, alm_info):
    if sht_info.is_CAR:
        raise ValueError("Can't use healpy for CAR maps")
    kwargs = {'lmax': alm_info.lmax, 'mmax': alm_info.mmax}
    if spin == 0:
        alm = [hp.map2alm(map[0], iter=0, **kwargs)]
    else:
        alm = hp.map2alm_spin(map, spin, **kwargs)
    return np.array(alm)


_m2a_d = {'ducc': _map2alm_ducc0,
          'healpy': _map2alm_healpy}
_a2m_d = {'ducc': _alm2map_ducc0,
          'healpy': _alm2map_healpy}


def _alm2catalog_ducc0(alms, positions, spin, lmax):
    alms = np.atleast_2d(alms)
    values = ducc0.sht.synthesis_general(alm=alms, spin=spin,
                                         lmax=lmax, loc=positions.T,
                                         epsilon=1E-5,
                                         nthreads=0)
    return values


def _catalog2alm_ducc0(values, positions, spin, lmax):
    values = np.atleast_2d(values)
    alm = ducc0.sht.adjoint_synthesis_general(lmax=lmax, map=values,
                                              loc=positions.T, spin=int(spin),
                                              epsilon=1E-5,
                                              nthreads=0)
    return alm


def map2alm(map, spin, map_info, alm_info, *, n_iter):
    """ Computes the spherical harmonic transform (SHT)
    for a set of input maps. The SHT implementation to
    be used can be selected via :meth:`set_sht_calculator`
    (see the documentation of :class:`NmtParams`).

    Args:
        map (`array`): 2D array with shape ``(nmaps, npix)``
            where  ``nmaps`` is either 1 (for spin-0 fields)
            or 2 (for spin-s fields), and ``npix`` is the
            number of pixels. If using CAR rectangular
            pixels, the map should be provided flattened.
        spin (:obj:`int`): Field spin.
        map_info (:class:`NmtMapInfo`): Object describing
            the pixelization of the input map.
        alm_info (:class:`NmtAlmInfo`): Object describing
            the structure of the output spherical harmonic
            coefficients.
        n_iter (:obj:`int`): Number of Jacobi iterations used
            to improve the accuracy of the SHT.

    Returns:
        (`array`): Harmonic coefficients :math:`a_{\\ell m}` \
            of the input map. A set of two arrays (E and B \
            modes) is returned if ``spin>0``.
    """
    map = map_info.si.pad_map(map)
    m2a = _m2a_d[nmt_params.sht_calculator]
    a2m = _a2m_d[nmt_params.sht_calculator]
    alm = m2a(map, spin, map_info.si, alm_info)
    for i in range(n_iter):
        dmap = a2m(alm, spin, map_info.si, alm_info)-map
        alm -= m2a(dmap, spin, map_info.si, alm_info)
    return alm


def alm2map(alm, spin, map_info, alm_info):
    """ Computes the inverse spherical harmonic transform
    (SHT) for a set of input :math:`a_{\\ell m}` s. The SHT
    implementation to be used can be selected via
    :meth:`set_sht_calculator` (see the documentation of
    :class:`NmtParams`).

    Args:
        alm (`array`): 2D array with shape ``(nmaps, n_lm)``
            where  ``nmaps`` is either 1 (for spin-0 fields)
            or 2 (for spin-s fields), and ``n_lm`` is the
            number of harmonic coefficients.
        spin (:obj:`int`): Field spin.
        map_info (:class:`NmtMapInfo`): Object describing
            the pixelization of the output map.
        alm_info (:class:`NmtAlmInfo`): Object describing
            the structure of the input spherical harmonic
            coefficients.

    Returns:
        (`array`): Map reconstructed from the input \
            :math:`a_{\\ell m}` s. A set of two arrays \
            (e.g. Q and U Stokes parameters) is returned \
            if ``spin>0``.
    """
    a2m = _a2m_d[nmt_params.sht_calculator]
    map = a2m(alm, spin, map_info.si, alm_info)
    return map_info.si.unpad_map(map)
