import numpy as np
import healpy as hp
from pymaster import nmtlib as lib
import pymaster.utils as ut
import warnings


class NmtCovarianceWorkspace(object):
    """ :obj:`NmtCovarianceWorkspace` objects are used to compute and
    store the coupling coefficients needed to calculate the Gaussian
    covariance matrix of angular power spectra under the approximations
    described in in `Garcia-Garcia et al. 2019
    <https://arxiv.org/abs/1906.11765>`_ (see also
    `Efstathiou et al. 2003 <https://arxiv.org/abs/astro-ph/0307515>`_,
    and `Couchot et al. 2016 <https://arxiv.org/abs/1609.09730>`_).

    :obj:`NmtCovarianceWorkspace` objects may be constructed from a set
    of :obj:`~pymaster.field.NmtField` objects, describing the masks
    of the fields being correlated, or may be read from a file.
    We recommend using the class methods :meth:`from_fields` and
    :meth:`from_file` to create new :obj:`NmtCovarianceWorkspace` objects,
    rather than using the main constructor.

    Args:
        fla1 (:class:`~pymaster.field.NmtField`): First field contributing
            to the first power spectrum whose covariance you want to
            compute.
        fla2 (:class:`~pymaster.field.NmtField`): Second field contributing
            to the first power spectrum whose covariance you want to
            compute.
        flb1 (:class:`~pymaster.field.NmtField`): As ``fla1`` for the
            second power spectrum. If ``None``, it will be set to
            ``fla1``.
        flb2 (:class:`~pymaster.field.NmtField`): As ``fla2`` for the
            second power spectrum. If ``None``, it will be set to
            ``fla2``.
        l_toeplitz (:obj:`int`): If a positive number, the Toeplitz
            approximation described in `Louis et al. 2020
            <https://arxiv.org/abs/2010.14344>`_ will be used.
            In that case, this quantity corresponds to
            :math:`\\ell_{\\rm toeplitz}` in Fig. 3 of that paper.
        l_exact (:obj:`int`): If ``l_toeplitz>0``, it corresponds to
            :math:`\\ell_{\\rm exact}` in Fig. 3 of the paper.
            Ignored if ``l_toeplitz<=0``.
        dl_band (:obj:`int`): If ``l_toeplitz>0``, this quantity
            corresponds to :math:`\\Delta \\ell_{\\rm band}` in Fig.
            3 of the paper. Ignored if ``l_toeplitz<=0``.
        spin0_only (:obj:`bool`): If ``True``, only spin-0 combinations
            of the mode-coupling coefficients will be computed and stored.
        fname (:obj:`str`): Input file name. If not `None`, the values of
            all input fields will be ignored, and all mode-coupling
            coefficients will be read from file.
        force_spin_only (:obj:`bool`): If ``True``, only spin-0
            combinations of the mode-coupling coefficients will
            be read and stored.
    """
    def __init__(self, fla1=None, fla2=None, flb1=None, flb2=None,
                 l_toeplitz=-1, l_exact=-1, dl_band=-1,
                 spin0_only=False, fname=None, force_spin0_only=False):
        self.wsp = None

        if ((fla1 is None) and (fla2 is None) and (fname is None)):
            warnings.warn("The bare constructor for `NmtCovarianceWorkspace` "
                          "objects is deprecated and will be removed "
                          "in future versions of NaMaster. Consider "
                          "using the class methods "
                          "`from_fields` and `from_file`, or pass "
                          "the necessary arguments to the constructor.",
                          category=DeprecationWarning)
            return

        if (fname is not None):
            self.read_from(fname, force_spin0_only=force_spin0_only)
            return

        self.compute_coupling_coefficients(fla1, fla2,
                                           flb1=flb1, flb2=flb2,
                                           l_toeplitz=l_toeplitz,
                                           l_exact=l_exact,
                                           dl_band=dl_band,
                                           spin0_only=spin0_only)

    @classmethod
    def from_fields(cls, fla1, fla2, flb1=None, flb2=None,
                    l_toeplitz=-1, l_exact=-1, dl_band=-1,
                    spin0_only=False):
        """ Creates an :obj:`NmtCovarianceWorkspace` object containing the
        mode-coupling coefficients of the Gaussian covariance
        between the power spectra of two pairs of
        :class:`~pymaster.field.NmtField` objects (``fla1``, ``fla2``,
        ``flb1``, and ``flb2``). Note that you can reuse this
        workspace for the covariance of power spectra between any
        pairs of fields as long as the fields have the same masks
        as those passed to this function, and as long as the binning
        schemes used are also the same.

        Args:
            fla1 (:class:`~pymaster.field.NmtField`): First field contributing
                to the first power spectrum whose covariance you want to
                compute.
            fla2 (:class:`~pymaster.field.NmtField`): Second field contributing
                to the first power spectrum whose covariance you want to
                compute.
            flb1 (:class:`~pymaster.field.NmtField`): As ``fla1`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla1``.
            flb2 (:class:`~pymaster.field.NmtField`): As ``fla2`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla2``.
            l_toeplitz (:obj:`int`): If a positive number, the Toeplitz
                approximation described in `Louis et al. 2020
                <https://arxiv.org/abs/2010.14344>`_ will be used.
                In that case, this quantity corresponds to
                :math:`\\ell_{\\rm toeplitz}` in Fig. 3 of that paper.
            l_exact (:obj:`int`): If ``l_toeplitz>0``, it corresponds to
                :math:`\\ell_{\\rm exact}` in Fig. 3 of the paper.
                Ignored if ``l_toeplitz<=0``.
            dl_band (:obj:`int`): If ``l_toeplitz>0``, this quantity
                corresponds to :math:`\\Delta \\ell_{\\rm band}` in Fig.
                3 of the paper. Ignored if ``l_toeplitz<=0``.
            spin0_only (:obj:`bool`): If ``True``, only spin-0 combinations
                of the mode-coupling coefficients will be computed and stored.
        """
        return cls(fla1=fla1, fla2=fla2, flb1=flb1, flb2=flb2,
                   l_toeplitz=l_toeplitz, l_exact=l_exact,
                   dl_band=dl_band, spin0_only=spin0_only)

    @classmethod
    def from_file(cls, fname, force_spin0_only=False):
        """ Creates an :obj:`NmtCovarianceWorkspace` object from the
        mode-coupling coefficients stored in a FITS file.
        See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name.
            force_spin_only (:obj:`bool`): If ``True``, only spin-0
                combinations of the mode-coupling coefficients will
                be read and stored.
        """
        return cls(fname=fname, force_spin0_only=force_spin0_only)

    def __del__(self):
        if self.wsp is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp)
            self.wsp = None

    def read_from(self, fname, force_spin0_only=False):
        """ Reads the contents of an :obj:`NmtCovarianceWorkspace`
        object from a FITS file.

        Args:
            fname (:obj:`str`): Input file name.
            force_spin_only (:obj:`bool`): If ``True``, only spin-0
                combinations of the mode-coupling coefficients will
                be read and stored.
        """
        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_covar_workspace(fname,
                                            int(force_spin0_only))

    def compute_coupling_coefficients(self, fla1, fla2,
                                      flb1=None, flb2=None, *,
                                      l_toeplitz=-1,
                                      l_exact=-1, dl_band=-1,
                                      spin0_only=False):
        """ Computes coupling coefficients of the Gaussian covariance
        between the power spectra of two pairs of
        :class:`~pymaster.field.NmtField` objects (``fla1``, ``fla2``,
        ``flb1``, and ``flb2``). Note that you can reuse this
        workspace for the covariance of power spectra between any
        pairs of fields as long as the fields have the same masks
        as those passed to this function, and as long as the binning
        schemes used are also the same.

        Args:
            fla1 (:class:`~pymaster.field.NmtField`): First field contributing
                to the first power spectrum whose covariance you want to
                compute.
            fla2 (:class:`~pymaster.field.NmtField`): Second field contributing
                to the first power spectrum whose covariance you want to
                compute.
            flb1 (:class:`~pymaster.field.NmtField`): As ``fla1`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla1``.
            flb2 (:class:`~pymaster.field.NmtField`): As ``fla2`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla2``.
            l_toeplitz (:obj:`int`): If a positive number, the Toeplitz
                approximation described in `Louis et al. 2020
                <https://arxiv.org/abs/2010.14344>`_ will be used.
                In that case, this quantity corresponds to
                :math:`\\ell_{\\rm toeplitz}` in Fig. 3 of that paper.
            l_exact (:obj:`int`): If ``l_toeplitz>0``, it corresponds to
                :math:`\\ell_{\\rm exact}` in Fig. 3 of the paper.
                Ignored if ``l_toeplitz<=0``.
            dl_band (:obj:`int`): If ``l_toeplitz>0``, this quantity
                corresponds to :math:`\\Delta \\ell_{\\rm band}` in Fig.
                3 of the paper. Ignored if ``l_toeplitz<=0``.
            spin0_only (:obj:`bool`): If ``True``, only spin-0 combinations
                of the mode-coupling coefficients will be computed and stored.
        """
        if flb1 is None:
            flb1 = fla1
        if flb2 is None:
            flb2 = fla2

        if np.any([fla1.anisotropic_mask, fla2.anisotropic_mask,
                   flb1.anisotropic_mask, flb2.anisotropic_mask]):
            raise NotImplementedError("Covariance matrix estimation not "
                                      "implemented for anisotropic weights.")

        if (not (fla1.is_compatible(fla2) and
                 fla1.is_compatible(flb1) and
                 fla1.is_compatible(flb2))):
            raise ValueError("Fields have incompatible pixelizations.")

        ut._toeplitz_sanity(l_toeplitz, l_exact, dl_band,
                            fla1.ainfo.lmax, fla1, flb1)

        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None

        def get_mask_prod_cl(f1_p1, f2_p1, f1_p2, f2_p2):
            mask_p1 = f1_p1.get_mask()*f2_p1.get_mask()
            alm_p1 = ut.map2alm(np.array([mask_p1]), 0,
                                f1_p1.minfo, f1_p1.ainfo_mask,
                                n_iter=f1_p1.n_iter_mask)[0]
            mask_p2 = f1_p2.get_mask()*f2_p2.get_mask()
            alm_p2 = ut.map2alm(np.array([mask_p2]), 0,
                                f1_p2.minfo, f1_p2.ainfo_mask,
                                n_iter=f1_p2.n_iter_mask)[0]
            return hp.alm2cl(alm_p1, alm_p2, lmax=f1_p1.ainfo_mask.lmax)

        pcl_mask_11_22 = get_mask_prod_cl(fla1, flb1, fla2, flb2)
        pcl_mask_12_21 = get_mask_prod_cl(fla1, flb2, fla2, flb1)
        self.wsp = lib.covar_workspace_init_py(pcl_mask_11_22,
                                               pcl_mask_12_21,
                                               int(fla1.ainfo.lmax),
                                               int(fla1.ainfo_mask.lmax),
                                               l_toeplitz, l_exact, dl_band,
                                               int(spin0_only))

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtCovarianceWorkspace`
        object to a FITS file.

        Args:
            fname (:obj:`str`): Output file name.
        """
        if self.wsp is None:
            raise ValueError("Must initialize workspace before writing")
        lib.write_covar_workspace(self.wsp, "!"+fname)


class NmtCovarianceWorkspaceFlat(object):
    """ :obj:`NmtCovarianceWorkspaceFlat` objects are used to compute and
    store the coupling coefficients needed to calculate the Gaussian
    covariance matrix of angular power spectra using a flat-sky version
    of the approximations described in `Garcia-Garcia et al. 2019
    <https://arxiv.org/abs/1906.11765>`_. When initialized, this object
    is practically empty. The information describing the coupling
    coefficients must be computed or read from a file afterwards.
    """
    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            if lib.covar_workspace_flat_free is not None:
                lib.covar_workspace_flat_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """ Reads the contents of an :obj:`NmtCovarianceWorkspaceFlat`
        object from a FITS file.

        Args:
            fname (:obj:`str`): Input file name.
        """
        if self.wsp is not None:
            lib.covar_workspace_flat_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_covar_workspace_flat(fname)

    def compute_coupling_coefficients(self, fla1, fla2, bin_a,
                                      flb1=None, flb2=None, bin_b=None):
        """ Computes coupling coefficients of the Gaussian covariance
        between the power spectra of two pairs of
        :class:`~pymaster.field.NmtFieldFlat` objects (``fla1``, ``fla2``,
        ``flb1``, and ``flb2``). Note that you can reuse this
        workspace for the covariance of power spectra between any
        pairs of fields as long as the fields have the same masks
        as those passed to this function, and as long as the binning
        schemes used are also the same.

        Args:
            fla1 (:class:`~pymaster.field.NmtFieldFlat`): First field
                contributing to the first power spectrum whose covariance
                you want to compute.
            fla2 (:class:`~pymaster.field.NmtFieldFlat`): Second field
                contributing to the first power spectrum whose covariance
                you want to compute.
            bin_a (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme for the
                first power spectrum.
            flb1 (:class:`~pymaster.field.NmtFieldFlat`): As ``fla1`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla1``.
            flb2 (:class:`~pymaster.field.NmtFieldFlat`): As ``fla2`` for the
                second power spectrum. If ``None``, it will be set to
                ``fla2``.
            bin_b (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme for the
                second power spectrum. If ``None``, ``bin_a`` will be used.
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
        """ Writes the contents of an :obj:`NmtCovarianceWorkspaceFlat` object
        to a FITS file.

        Args:
            fname (:obj:`str`): Output file name.
        """
        if self.wsp is None:
            raise ValueError("Must initialize workspace before writing")
        lib.write_covar_workspace_flat(self.wsp, "!"+fname)


def gaussian_covariance(cw, spin_a1, spin_a2, spin_b1, spin_b2,
                        cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None,
                        coupled=False):
    """ Computes the Gaussian covariance matrix for power spectra using the
    information precomputed in cw (a :class:`NmtCovarianceWorkspace`
    object). ``cw`` should have been initialized using four
    :class:`~pymaster.field.NmtField` objects (let's call them `a1`,
    `a2`, `b1`, and `b2`), corresponding to the two pairs of fields
    whose power spectra we want the covariance of. These power spectra
    should have been computed using two
    :class:`~pymaster.workspaces.NmtWorkspace` objects, ``wa`` and
    ``wb``, which must be passed as arguments of this function (the
    power spectrum for fields `a1` and `a2` was computed with ``wa``,
    and that of `b1` and `b2` with ``wb``). Using the same notation,
    ``clXnYm`` should be a prediction for the power spectrum between
    fields `Xn` and `Ym`. These predicted input power spectra should
    be defined for all multipoles :math:`\\ell` up to the
    :math:`\\ell_{\\rm max}` with which all fields were constructed.

    .. note::
        Note that, as suggested in
        `Nicola et al. 2020 <https://arxiv.org/abs/2010.09717>`_
        (the so-called "improved narrow-kernel approximation"), an
        optimal choice for the input power spectra would be the
        mode-coupled version of the true power spectra of the
        corresponding fields divided by the average of the product
        of the associated masks across the sky (Eq. 2.36 in the paper).
        Often, a good substitute for this can be obtained as the
        pseudo-:math:`C_\\ell` of the associated maps (e.g. computed via
        :meth:`~pymaster.workspaces.compute_coupled_cell`), divided
        by the same mean mask product.

    Args:
        cw (:obj:`NmtCovarianceWorkspace`): Workspace containing the
            precomputed coupling coefficients.
        spin_a1 (:obj:`int`): Spin of field `a1`.
        spin_a2 (:obj:`int`): Spin of field `a2`.
        spin_b1 (:obj:`int`): Spin of field `b1`.
        spin_b2 (:obj:`int`): Spin of field `b2`.
        cla1b1 (`array`): Prediction for the cross-power spectrum
            between fields `a1` and `b1`.
        cla1b2 (`array`): As `cla1b1` for fields `a1` and `b2`.
        cla2b1 (`array`): As `cla1b1` for fields `a2` and `b1`.
        cla2b2 (`array`): As `cla1b1` for fields `a2` and `b2`.
        wa (:class:`~pymaster.workspaces.NmtWorkspace`): Workspace
            containing the mode-coupling matrix for the first power
            spectrum (that of fields `a1` and `a2`).
        wb (:class:`~pymaster.workspaces.NmtWorkspace`): As ``wa``
            for the second power spectrum (that of fields `b1` and
            `b2`). If ``None``, ``wa`` will be used instead.
        coupled (:obj:`bool`): If ``True``, the covariance matrix
            of the mode-coupled pseudo-:math:`C_\\ell` s will be
            computed. Otherwise it'll be the covariance of
            mode-decoupled bandpowers.
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
        raise ValueError("Input C_ls have a weird length. "
                         f"Expected {cw.wsp.lmax+1}, but got "
                         f"({len(cla1b1[0])}, {len(cla1b2[0])}, "
                         f"{len(cla2b1[0])}, {len(cla2b2[0])}).")

    if coupled:
        len_a = wa.wsp.ncls * (cw.wsp.lmax+1)
        len_b = wb.wsp.ncls * (cw.wsp.lmax+1)
        wa.check_unbinned()
        wb.check_unbinned()

        covar = lib.comp_gaussian_covariance_coupled(
            cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
            wa.wsp, wb.wsp, cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
        )
    else:
        len_a = wa.wsp.ncls * wa.wsp.bin.n_bands
        len_b = wb.wsp.ncls * wb.wsp.bin.n_bands

        covar = lib.comp_gaussian_covariance(
            cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
            wa.wsp, wb.wsp, cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
        )

    return covar.reshape([len_a, len_b])


def gaussian_covariance_flat(cw, spin_a1, spin_a2, spin_b1, spin_b2, larr,
                             cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None):
    """ As :meth:`gaussian_covariance` but for the flat-sky versions of all
    quantities involved. The only difference with :meth:`gaussian_covariance`
    is that all power spectra must have been sampled at the input
    multipoles ``larr``.
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
        raise ValueError("Input C_ls have a weird length. "
                         f"Expected {len(larr)}, but got "
                         f"({len(cla1b1[0])}, {len(cla1b2[0])}, "
                         f"{len(cla2b1[0])}, {len(cla2b2[0])}).")
    len_a = wa.wsp.ncls * cw.wsp.bin.n_bands
    len_b = wb.wsp.ncls * cw.wsp.bin.n_bands

    covar1d = lib.comp_gaussian_covariance_flat(
        cw.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
        wa.wsp, wb.wsp, larr, cla1b1, cla1b2, cla2b1, cla2b2,
        len_a * len_b
    )
    covar = np.reshape(covar1d, [len_a, len_b])
    return covar
