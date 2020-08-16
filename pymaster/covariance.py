from pymaster import nmtlib as lib
import numpy as np


class NmtCovarianceWorkspace(object):
    """
    NmtCovarianceWorkspace objects are used to compute and store \
    the coupling coefficients needed to calculate the Gaussian \
    covariance matrix under the approximations in astro-ph/0307515 \
    and arXiv/1609.09730. When initialized, this object is \
    practically empty. The information describing the coupling \
    coefficients must be computed or read from a file afterwards.
    """
    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """
        Reads the contents of an NmtCovarianceWorkspace object \
        from a FITS file.

        :param str fname: input file name
        """
        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_covar_workspace(fname)

    def compute_coupling_coefficients(self, fla1, fla2,
                                      flb1=None, flb2=None,
                                      lmax=None, n_iter=3):
        """
        Computes coupling coefficients of the Gaussian covariance \
        between the power spectra of two pairs of NmtField objects \
        (fla1, fla2, flb1 and flb2). Note that you can reuse this \
        workspace for the covariance of power spectra between any \
        pairs of fields as long as the fields have the same masks \
        as those passed to this function, and as long as the binning \
        scheme used are also the same.

        :param NmtField fla1,fla2: fields contributing to the first \
            power spectrum whose covariance you want to compute.
        :param NmtField flb1,flb2: fields contributing to the second \
            power spectrum whose covariance you want to compute. If \
            None, fla1,fla2 will be used.
        :param n_iter: number of iterations when computing a_lms.
        """
        if flb1 is None:
            flb1 = fla1
        if flb2 is None:
            flb2 = fla2

        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None

        ns = fla1.fl.cs.n_eq
        if (fla2.fl.cs.n_eq != ns) or (flb1.fl.cs.n_eq != ns) or \
           (flb2.fl.cs.n_eq != ns):
            raise ValueError("Everything should have the same resolution!")

        if lmax is None:
            lmax = lib.get_lmax_from_cs_py(fla1.fl.cs)

        self.wsp = lib.covar_workspace_init_py(fla1.fl, fla2.fl, flb1.fl,
                                               flb2.fl, lmax, n_iter)

    def write_to(self, fname):
        """
        Writes the contents of an NmtCovarianceWorkspace object to a FITS file.

        :param str fname: output file name
        """
        if self.wsp is None:
            raise ValueError("Must initialize workspace before writing")
        lib.write_covar_workspace(self.wsp, "!"+fname)


class NmtCovarianceWorkspaceFlat(object):
    """
    NmtCovarianceWorkspaceFlat objects are used to compute and store the \
    coupling coefficients needed to calculate the Gaussian covariance \
    matrix under a flat-sky version the Efstathiou approximations in \
    astro-ph/0307515 and arXiv/1609.09730. When initialized, this object \
    is practically empty. The information describing the coupling \
    coefficients must be computed or read from a file afterwards.
    """
    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            lib.covar_workspace_flat_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """
        Reads the contents of an NmtCovarianceWorkspaceFlat object from a \
        FITS file.

        :param str fname: input file name
        """
        if self.wsp is not None:
            lib.covar_workspace_flat_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_covar_workspace_flat(fname)

    def compute_coupling_coefficients(self, fla1, fla2, bin_a,
                                      flb1=None, flb2=None, bin_b=None):
        """
        Computes coupling coefficients of the Gaussian covariance between \
        the power spectra of two pairs of NmtFieldFlat objects (fla1, fla2, \
        flb1 and flb2). Note that you can reuse this workspace for the \
        covariance of power spectra between any pairs of fields as long \
        as the fields have the same masks as those passed to this function, \
        and as long as the binning scheme used are also the same.

        :param NmtFieldFlat fla1,fla2: fields contributing to the first \
            power spectrum whose covariance you want to compute.
        :param NmtBinFlat bin_a: binning scheme for the first power \
            spectrum.
        :param NmtFieldFlat flb1,flb2: fields contributing to the second \
            power spectrum whose covariance you want to compute. If None, \
            fla1,fla2 will be used.
        :param NmtBinFlat bin_b: binning scheme for the second power \
            spectrum. If none, bin_a will be used.
        """
        if flb1 is None:
            flb1 = fla1
        if flb2 is None:
            flb2 = fla2
        if bin_b is None:
            bin_b = bin_a

        if (fla1.fl.fs.nx != fla2.fl.fs.nx) or \
           (fla1.fl.fs.ny != fla2.fl.fs.ny) or \
           (fla1.fl.fs.nx != flb1.fl.fs.nx) or \
           (fla1.fl.fs.ny != flb1.fl.fs.ny) or \
           (fla1.fl.fs.nx != flb2.fl.fs.nx) or \
           (fla1.fl.fs.ny != flb2.fl.fs.ny):
            raise ValueError("Everything should have the same resolution!")

        if self.wsp is not None:
            lib.covar_workspace_flat_free(self.wsp)
            self.wsp = None
        self.wsp = lib.covar_workspace_flat_init_py(fla1.fl, fla2.fl,
                                                    bin_a.bin,
                                                    flb1.fl, flb2.fl,
                                                    bin_b.bin)

    def write_to(self, fname):
        """
        Writes the contents of an NmtCovarianceWorkspaceFlat object to \
        a FITS file.

        :param str fname: output file name
        """
        if self.wsp is None:
            raise ValueError("Must initialize workspace before writing")
        lib.write_covar_workspace_flat(self.wsp, "!"+fname)


def gaussian_covariance(cw, spin_a1, spin_a2, spin_b1, spin_b2,
                        cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None,
                        coupled=False):
    """
    Computes Gaussian covariance matrix for power spectra using the \
    information precomputed in cw (a NmtCovarianceWorkspace object). \
    cw should have been initialized using four NmtField objects (let's \
    call them a1, a2, b1 and b2), corresponding to the two pairs of \
    fields whose power spectra we want the covariance of. These power \
    spectra should have been computed using two NmtWorkspace objects, \
    wa and wb, which must be passed as arguments of this function (the \
    power spectrum for fields a1 and a2 was computed with wa, and that \
    of b1 and b2 with wb). Using the same notation, clXnYm should be a \
    prediction for the power spectrum between fields Xn and Ym. These \
    predicted input power spectra should be defined for all \
    ells <=3*nside (where nside is the HEALPix resolution parameter of \
    the fields that were correlated).

    :param NmtCovarianceWorkspace cw: workspaces containing the \
        precomputed coupling coefficients.
    :param int spin_Xn: spin for the n-th field in pair X.
    :param clXnYm: prediction for the cross-power spectrum between \
        fields Xn and Ym.
    :param NmtWorkspace wX: workspace containing the mode-coupling \
        matrix pair X. If `wb` is `None`, the code will assume `wb=wa`.
    :param coupled: if `True`, the covariance matrix of the
        mode-coupled pseudo-Cls will be computed. Otherwise it'll be
        the covariance of decoupled bandpowers.
    """
    nm_a1 = 2 if spin_a1 else 1
    nm_a2 = 2 if spin_a2 else 1
    nm_b1 = 2 if spin_b1 else 1
    nm_b2 = 2 if spin_b2 else 1

    if wb is None:
        wb = wa

    if (wa.wsp.ncls != nm_a1*nm_a2) or (wb.wsp.ncls != nm_b1*nm_b2):
        raise ValueError("Input spins do not match input workspaces")

    if (len(cla1b1) != nm_a1*nm_b1) or \
       (len(cla1b2) != nm_a1*nm_b2) or \
       (len(cla2b1) != nm_a2*nm_b1) or \
       (len(cla2b2) != nm_a2*nm_b2):
        raise ValueError("Input spins do not match input power"
                         "spectrum shapes")

    if (len(cla1b1[0]) < cw.wsp.lmax + 1) or \
       (len(cla1b2[0]) < cw.wsp.lmax + 1) or \
       (len(cla2b1[0]) < cw.wsp.lmax + 1) or \
       (len(cla2b2[0]) < cw.wsp.lmax + 1):
        raise ValueError("Input C_ls have a weird length")

    if coupled:
        len_a = wa.wsp.ncls * (cw.wsp.lmax+1)
        len_b = wb.wsp.ncls * (cw.wsp.lmax+1)

        covar = lib.comp_gaussian_covariance_coupled(
            cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
            wa.wsp, wb.wsp, cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
        )
    else:
        len_a = wa.wsp.ncls * wa.wsp.bin.n_bands
        len_b = wb.wsp.ncls * wa.wsp.bin.n_bands

        covar = lib.comp_gaussian_covariance(
            cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
            wa.wsp, wb.wsp, cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
        )

    return covar.reshape([len_a, len_b])


def gaussian_covariance_flat(cw, spin_a1, spin_a2, spin_b1, spin_b2, larr,
                             cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None):
    """
    Computes Gaussian covariance matrix for flat-sky power spectra using \
    the information precomputed in cw (a NmtCovarianceWorkspaceFlat object). \
    cw should have been initialized using four NmtFieldFlat objects (let's \
    call them a1, a2, b1 and b2), corresponding to the two pairs of fields \
    whose power spectra we want the covariance of. These power spectra should \
    have been computed using two NmtWorkspaceFlat objects, wa and wb, which \
    must be passed as arguments of this function (the power spectrum for \
    fields a1 and a2 was computed with wa, and that of b1 and b2 with wb). \
    Using the same notation, clXnYm should be a prediction for the power \
    spectrum between fields Xn and Ym. These predicted input power spectra \
    should be defined in a sufficiently well sampled range of ells given the \
    map properties from which the power spectra were computed. The values of \
    ell at which they are sampled are given by larr.

    Please note that, while the method used to estimate these covariance \
    is sufficiently accurate in a large number of scenarios, it is based on \
    a numerical approximation, and its accuracy should be assessed if in \
    doubt. In particular, we discourage users from using it to compute any \
    covariance matrix involving B-mode components.

    :param NmtCovarianceWorkspaceFlat cw: workspaces containing the \
        precomputed coupling coefficients.
    :param int spin_Xn: spin for the n-th field in pair X.
    :param larr: values of ell at which the following power spectra are \
        computed.
    :param clXnYm: prediction for the cross-power spectrum between fields \
        Xn and Ym.
    :param NmtWorkspaceFlat wX: workspace containing the mode-coupling matrix \
        pair X. If `wb` is `None`, the code will assume `wb=wa`.
    """
    nm_a1 = 2 if spin_a1 else 1
    nm_a2 = 2 if spin_a2 else 1
    nm_b1 = 2 if spin_b1 else 1
    nm_b2 = 2 if spin_b2 else 1

    if wb is None:
        wb = wa

    if (wa.wsp.ncls != nm_a1*nm_a2) or (wb.wsp.ncls != nm_b1*nm_b2):
        raise ValueError("Input spins do not match input workspaces")

    if (len(cla1b1) != nm_a1*nm_b1) or \
       (len(cla1b2) != nm_a1*nm_b2) or \
       (len(cla2b1) != nm_a2*nm_b1) or \
       (len(cla2b2) != nm_a2*nm_b2):
        raise ValueError("Input spins do not match input power"
                         "spectrum shapes")

    if (
        (len(cla1b1[0]) != len(larr))
        or (len(cla1b2[0]) != len(larr))
        or (len(cla2b1[0]) != len(larr))
        or (len(cla2b2[0]) != len(larr))
    ):
        raise ValueError("Input C_ls have a weird length")
    len_a = wa.wsp.ncls * cw.wsp.bin.n_bands
    len_b = wb.wsp.ncls * cw.wsp.bin.n_bands

    covar1d = lib.comp_gaussian_covariance_flat(
        cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
        wa.wsp, wb.wsp, larr, cla1b1, cla1b2, cla2b1, cla2b2,
        len_a * len_b
    )
    covar = np.reshape(covar1d, [len_a, len_b])
    return covar
