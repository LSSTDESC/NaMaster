from pymaster import nmtlib as lib
import numpy as np


class NmtBin(object):
    """
    An NmtBin object defines the set of bandpowers used in the computation of the pseudo-Cl estimator. The definition of bandpowers is described in Section 3.6 of the scientific documentation.

    :param int nside: HEALPix nside resolution parameter of the maps you intend to correlate. The maximum multipole considered for bandpowers will be 3*nside-1.
    :param array-like ells: array of integers corresponding to different multipoles
    :param array-like bpws: array of integers that assign the multipoles in ells to different bandpowers
    :param array-like weights: array of floats corresponding to the weights associated to each multipole in ells. The sum of weights within each bandpower is normalized to 1.
    :param int nlb: integer value corresponding to a constant bandpower width. I.e. the bandpowers will be defined as consecutive sets of nlb multipoles from l=2 to l=lmax (see below) with equal weights. If this argument is provided, the values of ells, bpws and weights are ignored.
    :param int lmax: integer value corresponding to the maximum multiple used by these bandpowers. If None, it will be set to 3*nside-1. In any case the actual maximum multipole will be chosen as the minimum of lmax, 3*nside-1 and the maximum element of ells.
    """

    def __init__(self, nside, bpws=None, ells=None, weights=None, nlb=None, lmax=None):
        self.bin = None

        if (bpws is None) and (ells is None) and (weights is None) and (nlb is None):
            raise KeyError("Must supply bandpower arrays or constant bandpower width")

        if lmax is None:
            lmax_in = 3 * nside - 1
        else:
            lmax_in = lmax

        if nlb is None:
            if (bpws is None) and (ells is None) and (weights is None):
                raise KeyError("Must provide bpws, ells and weights")
            ell_max = min(3 * nside - 1, lmax_in)
            self.bin = lib.bins_create_py(
                bpws.astype(np.int32), ells.astype(np.int32), weights, int(ell_max)
            )
        else:
            ell_max = min(3 * nside - 1, lmax_in)
            self.bin = lib.bins_constant(nlb, ell_max)
        self.lmax = ell_max

    def __del__(self):
        if self.bin is not None:
            lib.bins_free(self.bin)
            self.bin = None

    def get_n_bands(self):
        """
        Returns the number of bandpowers stored in this object

        :return: number of bandpowers
        """
        return self.bin.n_bands

    def get_nell_list(self):
        """
        Returns an array with the number of multipoles in each bandpower stored in this object

        :return: number of multipoles per bandpower
        """
        return lib.get_nell_list(self.bin, self.bin.n_bands)

    def get_ell_list(self, ibin):
        """
        Returns an array with the multipoles in the ibin-th bandpower

        :param int ibin: bandpower index
        :return: multipoles associated with bandpower ibin
        """
        return lib.get_ell_list(self.bin, ibin, lib.get_nell(self.bin, ibin))

    def get_weight_list(self, ibin):
        """
        Returns an array with the weights associated to each multipole in the ibin-th bandpower

        :param int ibin: bandpower index
        :return: weights associated to multipoles in bandpower ibin
        """
        return lib.get_weight_list(self.bin, ibin, lib.get_nell(self.bin, ibin))

    def get_effective_ells(self):
        """
        Returns an array with the effective multipole associated to each bandpower. These are computed as a weighted average of the multipoles within each bandpower.

        :return: effective multipoles for each bandpower
        """
        return lib.get_ell_eff(self.bin, self.bin.n_bands)

    def bin_cell(self, cls_in):
        """
        Bins a power spectrum into bandpowers. This is carried out as a weighted average over the multipoles in each bandpower.

        :param array-like cls_in: 2D array of power spectra
        :return: array of bandpowers
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
        """
        Un-bins a set of bandpowers into a power spectrum. This is simply done by assigning a constant value for every multipole in each bandpower (corresponding to the value of that bandpower).

        :param array-like cls_in: array of bandpowers
        :return: array of power spectra
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != self.bin.n_bands):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.unbin_cl(self.bin, cls_in, len(cls_in) * (self.lmax + 1))
        clout = np.reshape(cl1d, [len(cls_in), self.lmax + 1])
        if oned:
            clout = clout[0]
        return clout


class NmtBinFlat(object):
    """
    An NmtBinFlat object defines the set of bandpowers used in the computation of the pseudo-Cl estimator. The definition of bandpowers is described in Section 3.6 of the scientific documentation. Note that currently pymaster only supports top-hat bandpowers for flat-sky power spectra.

    :param array-like l0: array of floats corresponding to the lower bound of each bandpower.
    :param array-like lf: array of floats corresponding to the upper bound of each bandpower. lf should have the same shape as l0
    """

    def __init__(self, l0, lf):
        self.bin = None
        self.bin = lib.bins_flat_create_py(l0, lf)

    def __del__(self):
        if self.bin is not None:
            lib.bins_flat_free(self.bin)
            self.bin = None

    def get_n_bands(self):
        """
        Returns the number of bandpowers stored in this object

        :return: number of bandpowers
        """
        return self.bin.n_bands

    def get_effective_ells(self):
        """
        Returns an array with the effective multipole associated to each bandpower. These are computed as a weighted average of the multipoles within each bandpower.

        :return: effective multipoles for each bandpower
        """
        return lib.get_ell_eff_flat(self.bin, self.bin.n_bands)

    def bin_cell(self, ells, cls_in):
        """
        Bins a power spectrum into bandpowers. This is carried out as a weighted average over the multipoles in each bandpower.

        :param array-like ells: multipole values at which the input power spectra are defined
        :param array-like cls_in: 2D array of input power spectra
        :return: array of bandpowers
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != len(ells)):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.bin_cl_flat(self.bin, ells, cls_in, len(cls_in) * self.bin.n_bands)
        clout = np.reshape(cl1d, [len(cls_in), self.bin.n_bands])
        if oned:
            clout = clout[0]
        return clout

    def unbin_cell(self, cls_in, ells):
        """
        Un-bins a set of bandpowers into a power spectrum. This is simply done by assigning a constant value for every multipole in each bandpower (corresponding to the value of that bandpower).

        :param array-like cls_in: array of bandpowers
        :param array-like ells: array of multipoles at which the power spectra should be intepolated
        :return: array of power spectra
        """
        oned = False
        if cls_in.ndim != 2:
            oned = True
            cls_in = np.array([cls_in])
        if (cls_in.ndim > 2) or (len(cls_in[0]) != self.bin.n_bands):
            raise ValueError("Input Cl has wrong size")
        cl1d = lib.unbin_cl_flat(self.bin, cls_in, ells, len(cls_in) * len(ells))
        clout = np.reshape(cl1d, [len(cls_in), len(ells)])
        if oned:
            clout = clout[0]
        return clout
