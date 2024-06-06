from pymaster import nmtlib as lib
import numpy as np


def _get_bpw_arrays_linear(lmax, nlb):
    ells = np.arange(lmax+1, dtype=np.int32)
    bpws = ((ells-2) // nlb).astype(np.int32)
    bpws[:2] = -1
    # Remove last bandpower if smaller
    if np.sum(bpws == bpws[-1]) != nlb:
        bad = bpws == bpws[-1]
        bpws[bad] = -1
    return ells, bpws


class NmtBin(object):
    """:obj:`NmtBin` objects define the set of bandpowers used in the
    computation of the pseudo-:math:`C_\\ell` estimator. The
    definition of bandpowers is described in Section 2.1.3 of the
    `NaMaster paper <https://arxiv.org/abs/1809.09603>`_, and Section
    3.6 of the scientific documentation. We provide several convenience
    constructors that cover a range of common use cases requiring
    fewer parameters (see :meth:`NmtBin.from_nside_linear`,
    :meth:`NmtBin.from_lmax_linear` and :meth:`NmtBin.from_edges`).

    Args:
        ells (`array`): Array of integers corresponding to different
            multipoles.
        bpws (`array`): Array of integers that assign the multipoles
            in ``ells`` to different bandpowers. All negative values
            will be ignored.
        lmax (:obj:`int`): Maximum :math:`\\ell` to be considered by these
            bandpowers, and by any calculation that uses them (e.g.
            mode-coupling matrices, band limit used when computing
            spherical harmonic coefficients of input maps, etc.).
            If ``None``, the maximum of ``ells`` will be used.
        weights (`array`): Array of floats corresponding to the
            weights associated to each multipole in ``ells``. The sum
            of weights within each bandpower will be automatically
            normalized to 1. If ``None``, uniform weights are assumed.
        f_ell (`array`): If present, this is array represents an
            :math:`\\ell`-dependent function that will be multiplied
            by all pseudo-:math:`C_\\ell` computations carried out
            using this bandpower scheme.
    """
    def __init__(self, *, bpws, ells, lmax=None, weights=None,
                 f_ell=None):
        self.bin = None

        if lmax is None:
            lmax = np.amax(ells)
        self.lmax = int(lmax)
        if weights is None:
            weights = np.ones(len(ells))
        if f_ell is None:
            f_ell = np.ones(len(ells))
        self.bin = lib.bins_create_py(bpws.astype(np.int32),
                                      ells.astype(np.int32),
                                      weights, f_ell, self.lmax)

    @classmethod
    def from_nside_linear(cls, nside, nlb, is_Dell=False, f_ell=None):
        """ Convenience constructor for HEALPix maps with linear
        binning, starting at :math:`\\ell=2`, and up to
        :math:`\\ell=3 N_{\\rm side}-1`. Although this will also be the
        maximum multipole associated with this :obj:`NmtBin` object,
        only bandpowers containing a total of ``nlb`` multipoles within
        this range will be used (i.e. the last bin will be discarded
        if not complete).

        Args:
            nside (:obj:`int`): HEALPix :math:`N_{\\rm side}` resolution
                parameter of the maps you intend to correlate.
            nlb (:obj:`int`): Integer value corresponding to a constant
                bandpower width. I.e. the bandpowers will be defined as
                consecutive sets of ``nlb`` multipoles from
                :math:`\\ell=2` with equal weights.
            is_Dell (:obj:`bool`): If ``True``, the output of all
                pseudo-:math:`C_\\ell` computations carried out using
                this bandpower scheme (e.g. from
                :meth:`~pymaster.workspaces.NmtWorkspace.decouple_cell`)
                will be multiplied by :math:`\\ell (\\ell + 1) / 2 \\pi`
                (no prefactor otherwise).
            f_ell (`array`): If present, this is array represents an
                :math:`\\ell`-dependent function that will be multiplied by
                all pseudo-:math:`C_\\ell` computations carried out using
                this bandpower scheme. If not ``None``, the value of
                ``is_Dell`` is ignored. If provided, it must be sampled at
                all :math:`\\ell` s up to (and including)
                :math:`3 N_{\\rm side}-1`.
        """
        ells, bpws = _get_bpw_arrays_linear(3*nside-1, nlb)
        weights = np.ones(len(ells))
        if is_Dell and (f_ell is None):
            f_ell = ells * (ells+1) / (2*np.pi)
        return cls(lmax=3*nside-1, bpws=bpws, ells=ells, weights=weights,
                   f_ell=f_ell)

    @classmethod
    def from_lmax_linear(cls, lmax, nlb, is_Dell=False, f_ell=None):
        """ Convenience constructor for generic linear binning, starting
        at :math:`\\ell=2`, and up to :math:`\\ell=` ``lmax``. Although this
        will also be the maximum multipole associated with this :obj:`NmtBin`
        object, only bandpowers containing a total of ``nlb`` multipoles within
        this range will be used (i.e. the last bin will be discarded
        if not complete).

        Args:
            lmax (:obj:`int`): Integer value corresponding to the maximum
                multipole to be used.
            nlb (:obj:`int`): Integer value corresponding to a constant
                bandpower width. I.e. the bandpowers will be defined as
                consecutive sets of ``nlb`` multipoles from :math:`\\ell=2`
                with equal weights.
            is_Dell (:obj:`bool`): If ``True``, the output of all
                pseudo-:math:`C_\\ell` computations carried out using this
                bandpower scheme (e.g. from
                :meth:`~pymaster.workspaces.NmtWorkspace.decouple_cell`)
                will be multiplied by :math:`\\ell (\\ell + 1) / 2 \\pi`
                (no prefactor otherwise).
            f_ell (`array`): If present, this is array represents an
                :math:`\\ell`-dependent function that will be multiplied by
                all pseudo-:math:`C_\\ell` computations carried out using
                this bandpower scheme. If not ``None``, the value of
                ``is_Dell`` is ignored. If provided, it must be sampled at
                all :math:`\\ell` s up to (and including)
                :math:`\\ell_{\\rm max}`.
        """
        ells, bpws = _get_bpw_arrays_linear(lmax, nlb)
        weights = np.ones(len(ells))
        if is_Dell and (f_ell is None):
            f_ell = ells * (ells+1) / (2*np.pi)
        return cls(lmax=lmax, bpws=bpws, ells=ells, weights=weights,
                   f_ell=f_ell)

    @classmethod
    def from_edges(cls, ell_ini, ell_end, is_Dell=False, f_ell=None):
        """
        Convenience constructor for general equal-weight bands.
        All :math:`\\ell` s in the interval ``[ell_ini, ell_end)`` will be
        binned with equal weights across the band.

        Args:
            ell_ini (`array`): Array of integers containing the lower edges
                of each bandpower.
            ell_end (`array`): Array containing the upper edges of each
                bandpower.
            is_Dell (:obj:`bool`): If `True`, the output of all
                pseudo-:math:`C_\\ell` computations carried out using this
                bandpower scheme (e.g. from
                :meth:`~pymaster.workspaces.NmtWorkspace.decouple_cell`)
                will be multiplied by :math:`\\ell (\\ell + 1) / 2 \\pi`
                (no prefactor otherwise).
            f_ell (`array`): If present, this is array represents an
                :math:`\\ell`-dependent function that will be multiplied by
                all pseudo-:math:`C_\\ell` computations carried out using
                this bandpower scheme. If not ``None``, the value of
                ``is_Dell`` is ignored. If provided, it must be sampled at
                all :math:`\\ell` s covered by ``ell_ini`` and ``ell_end``.
        """
        ells, bpws, weights = [], [], []
        for ib, (li, le) in enumerate(zip(ell_ini, ell_end)):
            nlb = int(le - li)
            ells += list(range(li, le))
            bpws += [ib] * nlb
            weights += [1./nlb] * nlb
        ells = np.array(ells)
        bpws = np.array(bpws)
        weights = np.array(weights)
        if is_Dell and (f_ell is None):
            f_ell = ells * (ells+1) / (2*np.pi)
        return cls(lmax=np.amax(ells), bpws=bpws, ells=ells, weights=weights,
                   f_ell=f_ell)

    def __del__(self):
        if getattr(self, 'bin', None) is not None:
            if lib.bins_free is not None:
                lib.bins_free(self.bin)
            self.bin = None

    def get_n_bands(self):
        """ Returns the number of bandpowers stored in this
        object.

        Returns:
            (:obj:`int`): Number of bandpowers.
        """
        return self.bin.n_bands

    def get_nell_list(self):
        """ Returns an array with the number of multipoles
        in each bandpower stored in this object.

        Returns:
            (`array`): Number of multipoles per bandpower.
        """
        return lib.get_nell_list(self.bin, self.bin.n_bands)

    def get_ell_min(self, b):
        """ Returns the minimum ell value used by bandpower with
        index ``b``.

        Args:
            b (:obj:`int`): Bandpower index.

        Returns:
            (:obj:`int`): Minimum :math:`\\ell` value.
        """
        return self.get_ell_list(b)[0]

    def get_ell_max(self, b):
        """ Returns the maximum ell value used by bandpower with
        index ``b``.

        Args:
            b (:obj:`int`): Bandpower index.

        Returns:
            (:obj:`int`): Maximum :math:`\\ell` value.
        """
        return self.get_ell_list(b)[-1]

    def get_ell_list(self, b):
        """ Returns an array with the multipoles in the
        ``b``-th bandpower

        Args:
            b (:obj:`int`): Bandpower index.

        Returns:
            (`array`): Array of multipoles associated with bandpower
            ``b``.
        """
        return lib.get_ell_list(self.bin, int(b),
                                lib.get_nell(self.bin, int(b)))

    def get_weight_list(self, b):
        """ Returns an array with the weights associated with each
        multipole in the ``b``-th bandpower


        Args:
            b (:obj:`int`): Bandpower index.

        Returns:
            (`array`): Weights associated to multipoles in bandpower
            ``b``.
        """
        return lib.get_weight_list(self.bin, int(b),
                                   lib.get_nell(self.bin, int(b)))

    def get_effective_ells(self):
        """ Returns an array with the effective multipole of each
        bandpower. These are computed as a weighted average of the
        multipoles within each bin.

        Returns:
            (`array`): Effective multipoles for each bandpower.
        """
        return lib.get_ell_eff(self.bin, self.bin.n_bands)

    def bin_cell(self, cls_in):
        """ Bins a power spectrum into bandpowers. This is carried
        out as a weighted average over the multipoles in each bandpower.

        Args:
            cls_in (`array`): 1 or 2-D array of power spectra.

        Returns:
            (`array`): Array of bandpowers.
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != self.lmax + 1):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.bin_cl(self.bin, cls_in, len(cls_in) * self.bin.n_bands)
        clout = np.reshape(cl1d, [len(cls_in), self.bin.n_bands])
        if oned:
            clout = clout[0]
        return clout

    def unbin_cell(self, cls_in):
        """ Un-bins a set of bandpowers into a power spectrum. This is
        simply done by assigning a constant value for every multipole in
        each bandpower.

        Args:
            cls_in (`array`): 1 or 2-D array of bandpowers.

        Returns:
            (`array`): Array of power spectra.
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != self.bin.n_bands):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.unbin_cl(self.bin, cls_in,
                            int(len(cls_in) * (self.lmax + 1)))
        clout = np.reshape(cl1d, [len(cls_in), self.lmax + 1])
        if oned:
            clout = clout[0]
        return clout


class NmtBinFlat(object):
    """ An :obj:`NmtBinFlat` object defines the set of bandpowers used in
    the computation of the pseudo-:math:`C_\\ell` estimator. The definition
    of bandpowers is described in Section 2.5.1 of the
    `NaMaster paper <https://arxiv.org/abs/1809.09603>`_, or
    Section 3.6 of the scientific documentation. Note that NaMaster only
    supports top-hat bandpowers for flat-sky power spectra.

    Args:
        l0 (`array`): Array of floats corresponding to the lower bound of
            each bandpower.
        lf (`array`): Array of floats corresponding to the upper bound of
            each bandpower. ``lf`` should have the same shape as ``l0``.
    """

    def __init__(self, l0, lf):
        self.bin = None
        self.bin = lib.bins_flat_create_py(l0, lf)

    def __del__(self):
        if self.bin is not None:
            if lib.bins_flat_free is not None:
                lib.bins_flat_free(self.bin)
            self.bin = None

    def get_n_bands(self):
        """
        Returns the number of bandpowers stored in this object

        Returns:
            (:obj:`int`): Number of bandpowers.
        """
        return self.bin.n_bands

    def get_effective_ells(self):
        """
        Returns an array with the effective multipole associated with
        each bandpower. These are computed as a weighted average of
        the multipoles within each bin.

        Returns:
            (`array`): Effective multipoles for each bandpower.
        """
        return lib.get_ell_eff_flat(self.bin, self.bin.n_bands)

    def bin_cell(self, ells, cls_in):
        """
        Bins a power spectrum into bandpowers. This is carried out
        as a weighted average over the multipoles in each bin.

        Args:
            ells (`array`): Multipole values at which the input
                power spectra are sampled.
            cls_in (`array`): 1 or 2-D array of input power
                spectra.

        Returns:
            (`array`): Array of bandpowers.
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != len(ells)):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.bin_cl_flat(self.bin, ells, cls_in,
                               len(cls_in) * self.bin.n_bands)
        clout = np.reshape(cl1d, [len(cls_in), self.bin.n_bands])
        if oned:
            clout = clout[0]
        return clout

    def unbin_cell(self, cls_in, ells):
        """
        Un-bins a set of bandpowers into power spectra. This is
        simply done by assigning a constant value for every multipole
        in each bandpower.

        Args:
            cls_in (`array`): 1 or 2-D array of bandpowers.
            ells (`array`): Array of multipoles at which the input
                power spectra are evaluated.

        Returns:
            (`array`): Array of power spectra.
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != self.bin.n_bands):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.unbin_cl_flat(self.bin, cls_in, ells,
                                 len(cls_in) * len(ells))
        clout = np.reshape(cl1d, [len(cls_in), len(ells)])
        if oned:
            clout = clout[0]
        return clout
