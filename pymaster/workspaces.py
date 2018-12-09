from pymaster import nmtlib as lib
import numpy as np


class NmtWorkspace(object):
    """
    NmtWorkspace objects are used to compute and store the coupling matrix associated with an incomplete sky coverage, and used in the MASTER algorithm. When initialized, this object is practically empty. The information describing the coupling matrix must be computed or read from a file afterwards.
    """

    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            lib.workspace_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """
        Reads the contents of an NmtWorkspace object from a file (encoded using an internal binary format).

        :param str fname: input file name
        """
        if self.wsp is not None:
            lib.workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_workspace(fname)

    def compute_coupling_matrix(self, fl1, fl2, bins, is_teb=False):
        """
        Computes coupling matrix associated with the cross-power spectrum of two NmtFields and an NmtBin binning scheme. Note that the mode coupling matrix will only contain ells up to the maximum multipole included in the NmtBin bandpowers.

        :param NmtField fl1,fl2: fields to correlate
        :param NmtBin bin: binning scheme
        :param boolean is_teb: if true, all mode-coupling matrices (0-0,0-2,2-2) will be computed at the same time. In this case, fl1 must be a spin-0 field and fl1 must be spin-2.
        """
        if self.wsp is not None:
            lib.workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.comp_coupling_matrix(fl1.fl, fl2.fl, bins.bin, int(is_teb))

    def write_to(self, fname):
        """
        Writes the contents of an NmtWorkspace object to a file (encoded using an internal binary format).

        :param str fname: output file name
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before writing")
        lib.write_workspace(self.wsp, fname)

    def get_coupling_matrix(self) :
        """
        Returns the currently stored mode-coupling matrix.

        :return: mode-coupling matrix. The matrix will have shape `[nrows,nrows]`, with `nrows = n_cls * n_ells`, where `n_cls` is the number of power spectra (1, 2 or 4 for spin0-0, spin0-2 and spin2-2 correlations) and `n_ells = lmax + 1` (normally `lmax = 3 * nside - 1`). The assumed ordering of power spectra is such that the `l`-th element of the `i`-th power spectrum be stored with index `l * n_cls + i`.
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before getting a MCM")
        nrows=(self.wsp.lmax+1)*self.wsp.ncls
        return lib.get_mcm(self.wsp,nrows*nrows).reshape([nrows,nrows])

    def update_coupling_matrix(self,new_matrix) :
        """
        Updates the stored mode-coupling matrix.

        The new matrix (`new_matrix`) must have shape `[nrows,nrows]`, with `nrows = n_cls * n_ells`, where `n_cls` is the number of power spectra (1, 2 or 4 for spin0-0, spin0-2 and spin2-2 correlations) and `n_ells = lmax + 1` (normally `lmax = 3 * nside - 1`). The assumed ordering of power spectra is such that the `l`-th element of the `i`-th power spectrum be stored with index `l * n_cls + i`.

        :param new_matrix: matrix that will replace the mode-coupling matrix.
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before updating MCM")
        if len(new_matrix)!=(self.wsp.lmax+1)*self.wsp.ncls :
            raise ValueError("Input matrix has an inconsistent size")
        lib.update_mcm(self.wsp,len(new_matrix),new_matrix.flatten())

    def couple_cell(self, cl_in):
        """
        Convolves a set of input power spectra with a coupling matrix (see Eq. 6 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspace object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields).
        :return: coupled power spectrum
        """
        if (len(cl_in) != self.wsp.ncls) or (len(cl_in[0]) < self.wsp.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape")
        cl1d = lib.couple_cell_py(self.wsp, cl_in, self.wsp.ncls * (self.wsp.lmax + 1))
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.lmax + 1])
        return clout

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """
        Decouples a set of pseudo-Cl power spectra into a set of bandpowers by inverting the binned coupling matrix (se Eq. 4 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspace object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field, 4 for two spin-2 fields and 7 if this NmtWorkspace was created using `is_teb=True`).
        :param cl_bias: bias to the power spectrum associated to contaminant residuals (optional). This can be computed through :func:`pymaster.deprojection_bias`.
        :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations).
        :return: set of decoupled bandpowers
        """
        if (len(cl_in) != self.wsp.ncls) or (len(cl_in[0]) < self.wsp.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape")
        if cl_bias is not None:
            if (len(cl_bias) != self.wsp.ncls) or (len(cl_bias[0]) < self.wsp.lmax + 1):
                raise ValueError("Input bias power spectrum has wrong shape")
            clb = cl_bias.copy()
        else:
            clb = np.zeros_like(cl_in)
        if cl_noise is not None:
            if (len(cl_noise) != self.wsp.ncls) or (
                len(cl_noise[0]) < self.wsp.lmax + 1
            ):
                raise ValueError("Input noise power spectrum has wrong shape")
            cln = cl_noise.copy()
        else:
            cln = np.zeros_like(cl_in)

        cl1d = lib.decouple_cell_py(
            self.wsp, cl_in, cln, clb, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])

        return clout


class NmtWorkspaceFlat(object):
    """
    NmtWorkspaceFlat objects are used to compute and store the coupling matrix associated with an incomplete sky coverage, and used in the flat-sky version of the MASTER algorithm. When initialized, this object is practically empty. The information describing the coupling matrix must be computed or read from a file afterwards.
    """

    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            lib.workspace_flat_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """
        Reads the contents of an NmtWorkspaceFlat object from a file (encoded using an internal binary format).

        :param str fname: input file name
        """
        if self.wsp is not None:
            lib.workspace_flat_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_workspace_flat(fname)

    def compute_coupling_matrix(
            self, fl1, fl2, bins, ell_cut_x=[1., -1.], ell_cut_y=[1., -1.], is_teb=False
    ):
        """
        Computes coupling matrix associated with the cross-power spectrum of two NmtFieldFlats and an NmtBinFlat binning scheme.

        :param NmtFieldFlat fl1,fl2: fields to correlate
        :param NmtBinFlat bin: binning scheme
        :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
        :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
        :param boolean is_teb: if true, all mode-coupling matrices (0-0,0-2,2-2) will be computed at the same time. In this case, fl1 must be a spin-0 field and fl1 must be spin-2.
        """
        if self.wsp is not None:
            lib.workspace_flat_free(self.wsp)
            self.wsp = None

        self.wsp = lib.comp_coupling_matrix_flat(
            fl1.fl,
            fl2.fl,
            bins.bin,
            ell_cut_x[0],
            ell_cut_x[1],
            ell_cut_y[0],
            ell_cut_y[1],
            int(is_teb),
        )

    def write_to(self, fname):
        """
        Writes the contents of an NmtWorkspaceFlat object to a file (encoded using an internal binary format).

        :param str fname: output file name
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before writing")
        lib.write_workspace_flat(self.wsp, fname)

    def couple_cell(self, ells, cl_in):
        """
        Convolves a set of input power spectra with a coupling matrix (see Eq. 6 of the C API documentation).

        :param ells: list of multipoles on which the input power spectra are defined
        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspaceFlat object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields).
        :return: coupled power spectrum. The coupled power spectra are returned at the multipoles returned by calling :func:`get_ell_sampling` for any of the fields that were used to generate the workspace.
        """
        if (len(cl_in) != self.wsp.ncls) or (len(cl_in[0]) != len(ells)):
            raise ValueError("Input power spectrum has wrong shape")
        cl1d = lib.couple_cell_py_flat(
            self.wsp, ells, cl_in, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])
        return clout

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """
        Decouples a set of pseudo-Cl power spectra into a set of bandpowers by inverting the binned coupling matrix (se Eq. 4 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspaceFlat object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field, 4 for two spin-2 fields and 7 if this NmtWorkspaceFlat was created using `is_teb=True`). These power spectra must be defined at the multipoles returned by :func:`get_ell_sampling` for any of the fields used to create the workspace.
        :param cl_bias: bias to the power spectrum associated to contaminant residuals (optional). This can be computed through :func:`pymaster.deprojection_bias_flat`.
        :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations).
        :return: set of decoupled bandpowers
        """
        if (len(cl_in) != self.wsp.ncls) or (len(cl_in[0]) != self.wsp.bin.n_bands):
            raise ValueError("Input power spectrum has wrong shape")
        if cl_bias is not None:
            if (len(cl_bias) != self.wsp.ncls) or (
                len(cl_bias[0]) != self.wsp.bin.n_bands
            ):
                raise ValueError("Input bias power spectrum has wrong shape")
            clb = cl_bias.copy()
        else:
            clb = np.zeros_like(cl_in)
        if cl_noise is not None:
            if (len(cl_noise) != self.wsp.ncls) or (
                len(cl_noise[0]) != self.wsp.bin.n_bands
            ):
                raise ValueError("Input noise power spectrum has wrong shape")
            cln = cl_noise.copy()
        else:
            cln = np.zeros_like(cl_in)

        cl1d = lib.decouple_cell_py_flat(
            self.wsp, cl_in, cln, clb, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])

        return clout


def deprojection_bias(f1, f2, cls_guess):
    """
    Computes the bias associated to contaminant removal to the cross-pseudo-Cl of two fields.

    :param NmtField f1,f2: fields to correlate
    :param cls_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2.
    :return: deprojection bias power spectra.
    """
    if len(cls_guess) != f1.fl.nmaps * f2.fl.nmaps:
        raise ValueError("Proposal Cell doesn't match number of maps")
    if len(cls_guess[0]) != f1.fl.lmax + 1:
        raise ValueError("Proposal Cell doesn't match map resolution")
    cl1d = lib.comp_deproj_bias(
        f1.fl, f2.fl, cls_guess, len(cls_guess) * len(cls_guess[0])
    )
    cl2d = np.reshape(cl1d, [len(cls_guess), len(cls_guess[0])])

    return cl2d


def uncorr_noise_deprojection_bias(f1, map_var):
    """
    Computes the bias associated to contaminant removal in the presence of uncorrelated inhomogeneous noise to the auto-pseudo-Cl of a given field f1.

    :param NmtField f1: fields to correlate
    :param map_cls_guess: array containing a HEALPix map corresponding to the local noise variance (in one sterad).
    :return: deprojection bias power spectra.
    """
    ncls = f1.fl.nmaps * f1.fl.nmaps
    nells = f1.fl.lmax + 1
    if len(map_var) != f1.fl.npix:
        raise ValueError("Variance map doesn't match map resolution")
    cl1d = lib.comp_uncorr_noise_deproj_bias(f1.fl, map_var, ncls * nells)
    cl2d = np.reshape(cl1d, [ncls, nells])

    return cl2d


def deprojection_bias_flat(
    f1, f2, b, ells, cls_guess, ell_cut_x=[1., -1.], ell_cut_y=[1., -1.]
):
    """
    Computes the bias associated to contaminant removal to the cross-pseudo-Cl of two flat-sky fields. The returned power spectrum is defined at the multipoles returned by the method :func:`get_ell_sampling` of either f1 or f2.

    :param NmtFieldFlat f1,f2: fields to correlate
    :param NmtBinFlat b: binning scheme defining output bandpower
    :param ells: list of multipoles on which the proposal power spectra are defined
    :param cls_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2.
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: deprojection bias power spectra.
    """
    if len(cls_guess) != f1.fl.nmaps * f2.fl.nmaps:
        raise ValueError("Proposal Cell doesn't match number of maps")
    if len(cls_guess[0]) != len(ells):
        raise ValueError("cls_guess and ells must have the same length")
    cl1d = lib.comp_deproj_bias_flat(
        f1.fl,
        f2.fl,
        b.bin,
        ell_cut_x[0],
        ell_cut_x[1],
        ell_cut_y[0],
        ell_cut_y[1],
        ells,
        cls_guess,
        f1.fl.nmaps * f2.fl.nmaps * b.bin.n_bands,
    )
    cl2d = np.reshape(cl1d, [f1.fl.nmaps * f2.fl.nmaps, b.bin.n_bands])

    return cl2d


def compute_coupled_cell(f1, f2):
    """
    Computes the full-sky angular power spectra of two masked fields (f1 and f2) without aiming to deconvolve the mode-coupling matrix. Effectively, this is equivalent to calling the usual HEALPix anafast routine on the masked and contaminant-cleaned maps.

    :param NmtField f1,f2: fields to correlate
    :return: array of coupled power spectra
    """
    if f1.fl.nside != f2.fl.nside:
        raise ValueError("Fields must have same resolution")

    cl1d = lib.comp_pspec_coupled(
        f1.fl, f2.fl, f1.fl.nmaps * f2.fl.nmaps * (f1.fl.lmax + 1)
    )
    clout = np.reshape(cl1d, [f1.fl.nmaps * f2.fl.nmaps, f1.fl.lmax + 1])

    return clout


def compute_coupled_cell_flat(f1, f2, b, ell_cut_x=[1., -1.], ell_cut_y=[1., -1.]):
    """
    Computes the angular power spectra of two masked flat-sky fields (f1 and f2) without aiming to deconvolve the mode-coupling matrix. Effectively, this is equivalent to computing the map FFTs and averaging over rings of wavenumber.  The returned power spectrum is defined at the multipoles returned by the method :func:`get_ell_sampling` of either f1 or f2.

    :param NmtFieldFlat f1,f2: fields to correlate
    :param NmtBinFlat b: binning scheme defining output bandpower
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: array of coupled power spectra
    """
    if (f1.nx != f2.nx) or (f1.ny != f2.ny):
        raise ValueError("Fields must have same resolution")

    cl1d = lib.comp_pspec_coupled_flat(
        f1.fl,
        f2.fl,
        b.bin,
        f1.fl.nmaps * f2.fl.nmaps * b.bin.n_bands,
        ell_cut_x[0],
        ell_cut_x[1],
        ell_cut_y[0],
        ell_cut_y[1],
    )
    clout = np.reshape(cl1d, [f1.fl.nmaps * f2.fl.nmaps, b.bin.n_bands])

    return clout


def compute_full_master(f1, f2, b, cl_noise=None, cl_guess=None, workspace=None):
    """
    Computes the full MASTER estimate of the power spectrum of two fields (f1 and f2). This is equivalent to successively calling:

    - :func:`pymaster.NmtWorkspace.compute_coupling_matrix`
    - :func:`pymaster.deprojection_bias`
    - :func:`pymaster.compute_coupled_cell`
    - :func:`pymaster.NmtWorkspace.decouple_cell`

    :param NmtField f1,f2: fields to correlate
    :param NmtBin b: binning scheme defining output bandpower
    :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations) (optional).
    :param cl_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2. Needed only to compute the contaminant cleaning bias (optional).
    :param NmtWorkspace workspace: object containing the mode-coupling matrix associated with an incomplete sky coverage. If provided, the function will skip the computation of the mode-coupling matrix and use the information encoded in this object.
    :return: set of decoupled bandpowers
    """
    if f1.fl.nside != f2.fl.nside:
        raise ValueError("Fields must have same resolution")
    if cl_noise is not None:
        if len(cl_noise) != f1.fl.nmaps * f2.fl.nmaps:
            raise ValueError("Wrong length for noise power spectrum")
        cln = cl_noise.copy()
    else:
        cln = np.zeros([f1.fl.nmaps * f2.fl.nmaps, 3 * f1.fl.nside])
    if cl_guess is not None:
        if len(cl_guess) != f1.fl.nmaps * f2.fl.nmaps:
            raise ValueError("Wrong length for guess power spectrum")
        clg = cl_guess.copy()
    else:
        clg = np.zeros([f1.fl.nmaps * f2.fl.nmaps, 3 * f1.fl.nside])

    if workspace is None:
        cl1d = lib.comp_pspec(
            f1.fl, f2.fl, b.bin, None, cln, clg, len(cln) * b.bin.n_bands
        )
    else:
        cl1d = lib.comp_pspec(
            f1.fl, f2.fl, b.bin, workspace.wsp, cln, clg, len(cln) * b.bin.n_bands
        )

    clout = np.reshape(cl1d, [len(cln), b.bin.n_bands])

    return clout


def compute_full_master_flat(
    f1,
    f2,
    b,
    cl_noise=None,
    cl_guess=None,
    ells_guess=None,
    workspace=None,
    ell_cut_x=[1., -1.],
    ell_cut_y=[1., -1.],
):
    """
    Computes the full MASTER estimate of the power spectrum of two flat-sky fields (f1 and f2). This is equivalent to successively calling:

    - :func:`pymaster.NmtWorkspaceFlat.compute_coupling_matrix`
    - :func:`pymaster.deprojection_bias_flat`
    - :func:`pymaster.compute_coupled_cell_flat`
    - :func:`pymaster.NmtWorkspaceFlat.decouple_cell`

    :param NmtFieldFlat f1,f2: fields to correlate
    :param NmtBinFlat b: binning scheme defining output bandpower
    :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations) (optional).  This power spectrum should correspond to the bandpowers defined by b.
    :param cl_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2. Needed only to compute the contaminant cleaning bias (optional).
    :param ells_guess: multipoles at which cl_guess is defined.
    :param NmtWorkspaceFlat workspace: object containing the mode-coupling matrix associated with an incomplete sky coverage. If provided, the function will skip the computation of the mode-coupling matrix and use the information encoded in this object.
    :param int nell_rebin: number of sub-intervals into which the base k-intervals will be sub-sampled to compute the coupling matrix
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: set of decoupled bandpowers
    """
    if (f1.nx != f2.nx) or (f1.ny != f2.ny):
        raise ValueError("Fields must have same resolution")
    if cl_noise is not None:
        if (len(cl_noise) != f1.fl.nmaps * f2.fl.nmaps) or (
            len(cl_noise[0]) != b.bin.n_bands
        ):
            raise ValueError("Wrong length for noise power spectrum")
        cln = cl_noise.copy()
    else:
        cln = np.zeros([f1.fl.nmaps * f2.fl.nmaps, b.bin.n_bands])
    if cl_guess is not None:
        if ells_guess is None:
            raise ValueError("Must provide ell-values for cl_guess")
        if (len(cl_guess) != f1.fl.nmaps * f2.fl.nmaps) or (
            len(cl_guess[0]) != len(ells_guess)
        ):
            raise ValueError("Wrong length for guess power spectrum")
        lf = ells_guess.copy()
        clg = cl_guess.copy()
    else:
        lf = b.get_effective_ells()
        clg = np.zeros([f1.fl.nmaps * f2.fl.nmaps, b.bin.n_bands])

    if workspace is None:
        cl1d = lib.comp_pspec_flat(
            f1.fl,
            f2.fl,
            b.bin,
            None,
            cln,
            lf,
            clg,
            len(cln) * b.bin.n_bands,
            ell_cut_x[0],
            ell_cut_x[1],
            ell_cut_y[0],
            ell_cut_y[1],
        )
    else:
        cl1d = lib.comp_pspec_flat(
            f1.fl,
            f2.fl,
            b.bin,
            workspace.wsp,
            cln,
            lf,
            clg,
            len(cln) * b.bin.n_bands,
            ell_cut_x[0],
            ell_cut_x[1],
            ell_cut_y[0],
            ell_cut_y[1],
        )

    clout = np.reshape(cl1d, [len(cln), b.bin.n_bands])

    return clout
