import numpy as np
import healpy as hp
from pymaster import nmtlib as lib
import pymaster.utils as ut
from pymaster import (compute_coupled_cell, NmtBin, NmtWorkspace,
                      NmtFieldCatalog)


def _get_mask_prod_alm(f1, f2):
    # If we have catalog and map, make sure catalog goes
    # first
    fa, fb = (f1, f2) if _is_catalog(f1) else (f2, f1)

    # Check they have the same lmax_mask
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("Fields have incompatible pixelizations.")

    # Check which case we are dealing with
    if _is_catalog(fa):
        if _is_catalog(fb):
            option = 'cat_cat'
        else:
            option = 'cat_map'
    else:
        option = 'map_map'

    if option == 'map_map':
        if not fa.is_compatible(fb):  # Check they can be multiplied
            raise ValueError("Fields have incompatible pixelizations.")
        mask_p = fa.get_mask()*fb.get_mask()
        minfo = fa.minfo
    else:
        # The first field is a catalog
        mask_a, nside_a = fa.get_catalog_mask_map()
        minfo = ut.NmtMapInfo(None, [len(mask_a)])
        if option == 'cat_map':
            if fb.minfo.is_healpix:
                mask_b = hp.ud_grade(fb.get_mask(), nside_out=nside_a)
            else:  # Need to reproject CAR into healpix
                wlm_b = fb.get_mask_alms()
                mask_b = ut.alm2map(np.array([wlm_b]), 0, minfo,
                                    fb.ainfo_mask).squeeze()
            mask_p = mask_a * mask_b
        else:  # cat-cat
            auto = fa is fb
            if auto:
                mask_b, nside_b = mask_a, nside_a
            else:
                mask_b, nside_b = fb.get_catalog_mask_map()
            assert nside_a == nside_b
            mask_p = mask_a * mask_b
            if auto:  # Subtract self-pair contribution
                mask2_a, nside2_a = fa.get_catalog_mask_squared_map()
                assert nside_a == nside2_a
                mask_p -= mask2_a
    return mask_p, minfo


def _get_mask_prod_cl(f1_p1, f2_p1, f1_p2, f2_p2):
    mask_p1, minfo_p1 = _get_mask_prod_alm(f1_p1, f2_p1)
    alm_p1 = ut.map2alm(np.array([mask_p1]), 0,
                        minfo_p1, f1_p1.ainfo_mask,
                        n_iter=f1_p1.n_iter_mask)[0]
    mask_p2, minfo_p2 = _get_mask_prod_alm(f1_p2, f2_p2)
    alm_p2 = ut.map2alm(np.array([mask_p2]), 0,
                        minfo_p2, f1_p2.ainfo_mask,
                        n_iter=f1_p2.n_iter_mask)[0]
    return hp.alm2cl(alm_p1, alm_p2, lmax=f1_p1.ainfo_mask.lmax)


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
        all_spins (:obj:`bool`): If ``True``, coupling coefficients for
            all spin combinations will be calculated. Otherwise, only the
            spin combination determined by the input fields will be
            considered. The default value is ``True``, but setting it
            to ``False`` will generally lead to faster results and
            better memory usage (at the expense of some flexibility).
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
        fname (:obj:`str`): Input file name. If not `None`, the values of
            all input fields will be ignored, and all mode-coupling
            coefficients will be read from file.
    """
    def __init__(self, fla1, fla2, flb1=None, flb2=None,
                 all_spins=True, l_toeplitz=-1, l_exact=-1,
                 dl_band=-1, fname=None):
        self.wsp = None
        if (fname is not None):
            self._read_from(fname)
            return

        if flb1 is None:
            flb1 = fla1
        if flb2 is None:
            flb2 = fla2

        self.all_spins = all_spins
        self.spin_a1 = fla1.spin
        self.spin_a2 = fla2.spin
        self.spin_b1 = flb1.spin
        self.spin_b2 = flb2.spin

        self._compute_coupling_coefficients(fla1, fla2, flb1, flb2,
                                            all_spins=all_spins,
                                            l_toeplitz=l_toeplitz,
                                            l_exact=l_exact,
                                            dl_band=dl_band)

    @classmethod
    def from_fields(cls, fla1, fla2, flb1=None, flb2=None, *,
                    all_spins=True, l_toeplitz=-1, l_exact=-1,
                    dl_band=-1):
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
            all_spins (:obj:`bool`): If ``True``, coupling coefficients for
                all spin combinations will be calculated. Otherwise, only the
                spin combination determined by the input fields will be
                considered.
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
        """
        return cls(fla1=fla1, fla2=fla2, flb1=flb1, flb2=flb2,
                   all_spins=all_spins, l_toeplitz=l_toeplitz,
                   l_exact=l_exact, dl_band=dl_band)

    @classmethod
    def from_file(cls, fname):
        """ Creates an :obj:`NmtCovarianceWorkspace` object from the
        mode-coupling coefficients stored in a FITS file.
        See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name.
        """
        return cls(None, None, fname=fname)

    def __del__(self):
        if self.wsp is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp)
            self.wsp = None

    def _read_from(self, fname):
        """ Reads the contents of an :obj:`NmtCovarianceWorkspace`
        object from a FITS file.

        Args:
            fname (:obj:`str`): Input file name.
        """
        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_covar_workspace(fname)
        self.all_spins = bool(self.wsp.all_spins)
        self.spin_a1 = self.wsp.spin_a1
        self.spin_a2 = self.wsp.spin_a2
        self.spin_b1 = self.wsp.spin_b1
        self.spin_b2 = self.wsp.spin_b2

    def _compute_coupling_coefficients(self, fla1, fla2,
                                       flb1, flb2, *,
                                       all_spins=True,
                                       l_toeplitz=-1,
                                       l_exact=-1, dl_band=-1):
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
                second power spectrum.
            flb2 (:class:`~pymaster.field.NmtField`): As ``fla2`` for the
                second power spectrum.
            all_spins (:obj:`bool`): If ``True``, coupling coefficients for
                all spin combinations will be calculated. Otherwise, only the
                spin combination determined by the input fields will be
                considered.
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
        """
        if np.any([fla1.anisotropic_mask, fla2.anisotropic_mask,
                   flb1.anisotropic_mask, flb2.anisotropic_mask]):
            raise NotImplementedError("Covariance matrix estimation not "
                                      "implemented for anisotropic weights.")

        ut._toeplitz_sanity(l_toeplitz, l_exact, dl_band,
                            fla1.ainfo.lmax, fla1, flb1)

        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None

        pcl_mask_11_22 = _get_mask_prod_cl(fla1, flb1, fla2, flb2)
        pcl_mask_12_21 = _get_mask_prod_cl(fla1, flb2, fla2, flb1)
        self.wsp = lib.covar_workspace_init_py(int(fla1.spin), int(fla2.spin),
                                               int(flb1.spin), int(flb2.spin),
                                               pcl_mask_11_22,
                                               pcl_mask_12_21,
                                               int(all_spins), 0, 1, 1,
                                               int(fla1.ainfo.lmax),
                                               int(fla1.ainfo_mask.lmax),
                                               l_toeplitz, l_exact, dl_band)

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtCovarianceWorkspace`
        object to a FITS file.

        Args:
            fname (:obj:`str`): Output file name.
        """
        lib.write_covar_workspace(self.wsp, "!"+fname)

    def gaussian_covariance(self, cla1b1, cla1b2, cla2b1, cla2b2,
                            wa, wb=None, coupled=False,
                            spins=None):
        """ Computes the Gaussian covariance matrix for power spectra
        using the information precomputed in this
        :class:`NmtCovarianceWorkspace` object). Let us call the four
        fields used to initialise this workspace `a1`, `a2`, `b1`, and
        `b2`, corresponding to the two pairs of fields whose power
        spectra we want the covariance of. These power spectra should
        have been computed using two
        :class:`~pymaster.workspaces.NmtWorkspace` objects, ``wa`` and
        ``wb``, which must be passed as arguments of this method (the
        power spectrum for fields `a1` and `a2` was computed with ``wa``,
        and that of `b1` and `b2` with ``wb``). Using the same notation,
        ``clXnYm`` should be a prediction for the power spectrum between
        fields `Xn` and `Ym`. These predicted input power spectra should
        be defined for all multipoles :math:`\\ell` up to the
        :math:`\\ell_{\\rm max}` with which all fields were constructed.

        .. note::
            Note that, as suggested in
            `Nicola et al. 2020 <https://arxiv.org/abs/2010.09717>`_
            (the so-called "improved narrow-kernel approximation" - iNKA),
            an optimal choice for the input power spectra would be the
            mode-coupled version of the true power spectra of the
            corresponding fields divided by the average of the product
            of the associated masks across the sky (Eq. 2.36 in the paper).
            Often, a good substitute for this can be obtained as the
            pseudo-:math:`C_\\ell` of the associated maps (e.g. computed via
            :meth:`~pymaster.workspaces.compute_coupled_cell`), divided
            by the same mean mask product. The convenience function
            :meth:`get_iNKA_cell` may be used to calculate this
            spectrum under the iNKA.

        Args:
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
            spins (`array`): A list of 4 integers containing the
                spins of the fields whose power spectrum covariance
                one wishes to calculate. Note that you can only select
                arbitrary spin combinations if you created this object
                using ``all_spins=True``. If ``None``, the spin
                combination is determined by the fields used to
                create this object.
        """
        if spins is not None:
            if not self.all_spins:
                if ((spins[0] != self.spin_a1) or
                        (spins[1] != self.spin_a2) or
                        (spins[2] != self.spin_b1) or
                        (spins[3] != self.spin_b2)):
                    raise ValueError(
                        "The input spins do not coincide with those of "
                        "the fields used to initialise this object. If "
                        "you want to use arbitrary spin combinations, "
                        "use `all_spins=True` when initialising this "
                        "class.")
            if len(spins) != 4:
                raise ValueError("`spins` must have 4 elements.")
            spin_a1, spin_a2, spin_b1, spin_b2 = spins
        else:
            spin_a1 = self.spin_a1
            spin_a2 = self.spin_a2
            spin_b1 = self.spin_b1
            spin_b2 = self.spin_b2
        nm_a1 = 2 if spin_a1 else 1
        nm_a2 = 2 if spin_a2 else 1
        nm_b1 = 2 if spin_b1 else 1
        nm_b2 = 2 if spin_b2 else 1

        if wb is None:
            wb = wa

        if (wa.wsp.ncls != nm_a1*nm_a2) or (wb.wsp.ncls != nm_b1*nm_b2):
            raise ValueError("Field spins do not match input workspaces")

        if (len(cla1b1) != nm_a1*nm_b1) or \
           (len(cla1b2) != nm_a1*nm_b2) or \
           (len(cla2b1) != nm_a2*nm_b1) or \
           (len(cla2b2) != nm_a2*nm_b2):
            raise ValueError("Field spins do not match input power"
                             "spectrum shapes")

        if (len(cla1b1[0]) < self.wsp.lmax + 1) or \
           (len(cla1b2[0]) < self.wsp.lmax + 1) or \
           (len(cla2b1[0]) < self.wsp.lmax + 1) or \
           (len(cla2b2[0]) < self.wsp.lmax + 1):
            raise ValueError("Input C_ls have a weird length. "
                             f"Expected {self.wsp.lmax+1}, but got "
                             f"({len(cla1b1[0])}, {len(cla1b2[0])}, "
                             f"{len(cla2b1[0])}, {len(cla2b2[0])}).")

        if coupled:
            len_a = wa.wsp.ncls * (self.wsp.lmax+1)
            len_b = wb.wsp.ncls * (self.wsp.lmax+1)
            wa.check_unbinned()
            wb.check_unbinned()

            covar = lib.comp_gaussian_covariance_coupled(
                self.wsp, int(spin_a1), int(spin_a2),
                int(spin_b1), int(spin_b2), wa.wsp, wb.wsp, 1, 1,
                cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
            )
        else:
            len_a = wa.wsp.ncls * wa.wsp.bin.n_bands
            len_b = wb.wsp.ncls * wb.wsp.bin.n_bands

            covar = lib.comp_gaussian_covariance(
                self.wsp, int(spin_a1), int(spin_a2),
                int(spin_b1), int(spin_b2), wa.wsp, wb.wsp, 1, 1,
                cla1b1, cla1b2, cla2b1, cla2b2, len_a * len_b
            )

        return covar.reshape([len_a, len_b])


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

    def gaussian_covariance(self,
                            spin_a1, spin_a2, spin_b1, spin_b2, larr,
                            cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None):
        """ As :meth:`NmtCovarianceWorkspace.gaussian_covariance` but for the
        flat-sky versions of all quantities involved. The only difference with
        is that all power spectra must have been sampled at the input
        multipoles ``larr``, and the spins of all fields must be specified.
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
        len_a = wa.wsp.ncls * self.wsp.bin.n_bands
        len_b = wb.wsp.ncls * self.wsp.bin.n_bands

        covar1d = lib.comp_gaussian_covariance_flat(
            self.wsp, spin_a1, spin_a2, spin_b1, spin_b2,
            wa.wsp, wb.wsp, larr, cla1b1, cla1b2, cla2b1, cla2b2,
            len_a * len_b)

        covar = np.reshape(covar1d, [len_a, len_b])
        return covar


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

    .. warning::
        This function is deprecated and will be removed in a future
        version of NaMaster. Use the
        :meth:`NmtCovarianceWorkspace.gaussian_covariance` method
        instead.

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
    return cw.gaussian_covariance(cla1b1, cla1b2, cla2b1, cla2b2,
                                  wa, wb=wb, coupled=coupled,
                                  spins=[spin_a1, spin_a2,
                                         spin_b1, spin_b2])


def gaussian_covariance_flat(cw, spin_a1, spin_a2, spin_b1, spin_b2, larr,
                             cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None):
    """ As :meth:`gaussian_covariance` but for the flat-sky versions of all
    quantities involved. The only difference with :meth:`gaussian_covariance`
    is that all power spectra must have been sampled at the input
    multipoles ``larr``.

    .. warning::
        This function is deprecated and will be removed in a future
        version of NaMaster. Use the
        :meth:`NmtCovarianceWorkspaceFlat.gaussian_covariance` method
        instead.
    """
    return cw.gaussian_covariance(spin_a1, spin_a2, spin_b1, spin_b2,
                                  larr, cla1b1, cla1b2, cla2b1, cla2b2,
                                  wa, wb=wb)


def _is_catalog(f):
    return isinstance(f, NmtFieldCatalog)


def get_iNKA_cell(fla, flb, cl_guess=None, w=None):
    """ Returns the power spectrum that should be used in the
    calculation of the Gaussian covariance matrix according to the
    improved Narrow-Kernel Approximation (iNKA) of
    `Nicola et al. 2020 <https://arxiv.org/abs/2010.09717>`_. This
    can then be used, for instance, as input for
    :meth:`NmtCovarianceWorkspace.gaussian_covariance`.

    The two fields whose power spectra we need must be compatible.
    This means that, at least, they must be represented in harmonic
    space up to the same maximum multipole. If they are also
    compatible at the map level, the effective sky fraction used in
    the iNKA will be calculated from the product of their masks.
    Otherwise, their harmonic-space spectrum will be used.

    Args:
        fla (:class:`~pymaster.field.NmtField`): First field whose
            power spectrum we want to calculate.
        flb (:class:`~pymaster.field.NmtField`): Second field whose
            power spectrum we want to calculate.
        cl_guess (`array`): A guess for the true power spectra between
            ``fla`` and ``flb``. The number of power spectra must
            correspond to the spins of the two fields in question. If
            ``None``, the pseudo-:math:`C_\\ell` between the two fields
            will be used instead.
        w (:class:`~pymaster.workspaces.NmtWorkspace`): Workspace
            containing the mode-coupling matrix for these two fields.
            This is only required if ``cl_guess`` is not ``None``.
            If needed but ``None``, the mode-coupling matrix will be
            calculated on the fly.

    Returns:
        (`array`): power spectrum to be used in covariance calculations.
    """
    if not fla.is_compatible(flb, strict=False):
        raise ValueError("Fields have incompatible pixelizations")

    # 1. Compute fsky as the mean of the mask product.

    # If both fields are compatible at the map level, just take
    # the product of their maps and average. Otherwise use
    # Parseval's theorem and do it from their harmonic spectrum.
    use_map_product = fla.is_compatible(flb)

    if use_map_product:
        wawb = np.mean(fla.get_mask()*flb.get_mask())
    else:
        lmax = fla.ainfo_mask.lmax
        walm = fla.get_mask_alms()
        wblm = flb.get_mask_alms()
        clw = hp.alm2cl(walm, wblm, lmax=lmax)
        ls = np.arange(lmax+1)
        # Correct for catalogs
        if _is_catalog(fla) and _is_catalog(flb):
            phi_a = fla.get_ipd_kernel(lmax)
            phi_b = flb.get_ipd_kernel(lmax)
            # Subtract shot noise
            if fla is flb:
                clw = clw - fla.Nw
            # Multiply by kernels
            clw = clw * phi_a * phi_b
        wawb = np.sum((2*ls+1)*clw)/(4*np.pi)

    # 2. Compute pseudo-Cl

    # If no guess Cl is provided, compute it from the data.
    if cl_guess is None:
        pcl_ab = compute_coupled_cell(fla, flb)
        # Note that we don't need to worry abot catalogs
        # here, since the function above already subtracts
        # the shot-noise contribution.
    else:
        # We'll need to calculate the MCM if not available
        if w is None:
            # Just some token bins that go to the right lmax
            b = NmtBin.from_lmax_linear(
                fla.ainfo.lmax, nlb=int(fla.ainfo.lmax//10))
            w = NmtWorkspace.from_fields(fla, flb, b)
        pcl_ab = w.couple_cell(cl_guess)

    # 3. Return ratio
    return pcl_ab / wawb
