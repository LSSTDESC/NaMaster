import numpy as np
import healpy as hp
from pymaster import nmtlib as lib
import pymaster.utils as ut
import pymaster.master as mt
from pymaster import (compute_coupled_cell, NmtBin, NmtWorkspace,
                      NmtFieldCatalog)


def _get_mask_prod_alm(f1, f2):
    # If we have catalog and map, make sure catalog goes
    # first
    fa, fb = (f1, f2) if _is_mask_catalog(f1) else (f2, f1)

    # Check they have the same lmax_mask
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("Fields have incompatible pixelizations.")

    # Check which case we are dealing with
    if _is_mask_catalog(fa):
        if _is_mask_catalog(fb):
            option = 'cat_cat'
        else:
            option = 'cat_map'
    else:
        option = 'map_map'

    if option == 'map_map':
        minfo = fa.minfo
        if fa.is_compatible(fb):
            mask_p = fa.get_mask()*fb.get_mask()
        else:
            mask_a = fa.get_mask()
            if minfo.is_healpix and fb.minfo.is_healpix:
                mask_b = hp.ud_grade(fb.get_mask(), nside_out=minfo.nside)
            else:
                wlm_b = fb.get_mask_alms()
                mask_b = ut.alm2map(np.array([wlm_b]), 0,
                                    minfo, fb.ainfo_mask).squeeze()
            mask_p = mask_a * mask_b
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
    mask_p_alm = ut.map2alm(np.array([mask_p]), 0,
                            minfo, fa.ainfo_mask,
                            n_iter=fa.n_iter_mask)[0]
    return mask_p_alm, minfo


class NmtCovarianceWorkspaceOld(object):
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
            coefficients will be read from file."""
    def __init__(self, fla1, fla2, flb1=None, flb2=None,
                 all_spins=False, l_toeplitz=-1, l_exact=-1,
                 dl_band=-1, fname=None):
        self.wsp = None
        self.wsp_SN = None
        self.wsp_NS = None
        self.wsp_NN = None
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
                    all_spins=False, l_toeplitz=-1, l_exact=-1,
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
    def from_file(cls, fname, fname_SN=None, fname_NS=None, fname_NN=None):
        """ Creates an :obj:`NmtCovarianceWorkspace` object from the
        mode-coupling coefficients stored in a FITS file.
        See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name."""
        return cls(None, None, fname=fname)

    def __del__(self):
        if self.wsp is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp)
            self.wsp = None
        if self.wsp_SN is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp_SN)
            self.wsp_SN = None
        if self.wsp_NS is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp_NS)
            self.wsp_NS = None
        if self.wsp_NN is not None:
            if lib.covar_workspace_free is not None:
                lib.covar_workspace_free(self.wsp_NN)
            self.wsp_NN = None

    def _read_from(self, fname):
        """ Reads the contents of an :obj:`NmtCovarianceWorkspace`
        object from a FITS file.

        Args:
            fname (:obj:`str`): Input file name."""
        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None
        import fitsio as fts

        f = fts.FITS(fname)
        h = f['CWSP_PRIMARY'].read_header()
        self.lmax = h['LMAX']
        self.lmax_mask = h['LMAX_MASK'] if 'LMAX_MASK' in h else self.lmax
        if 'ALL_SPINS' in h:
            self.all_spins = h['ALL_SPINS']
            self.spin_a1 = h['SPIN_A1']
            self.spin_a2 = h['SPIN_A2']
            self.spin_b1 = h['SPIN_B1']
            self.spin_b2 = h['SPIN_B2']
        else:
            self.all_spins = 1
            self.spin_a1 = self.spin_a2 = self.spin_b1 = self.spin_b2 = 0
        self.has_SN = np.array([False, False])
        self.has_NS = np.array([False, False])
        self.has_NN = np.array([False, False])

        # Read the coupling coefficients
        xi_types = ['00_1122', '00_1221', '02_1122', '02_1221',
                    '22P_1122', '22P_1221', '22M_1122', '22M_1221']
        xis = {'': {}, 'SN': {}, 'NS': {}, 'NN': {}}
        # Loop over the different signal-noise combinations
        for prefix in ['', 'SN', 'NS', 'NN']:
            xi = xis[prefix]
            xi_any = False
            # Read all stored coupling coefficients
            for n in xi_types:
                if f'XI{prefix+n}' in f:
                    xi_any = True
                    xi[n] = f[f'XI{prefix + n}'].read()
                    if xi[n].shape != (self.lmax+1, self.lmax+1):
                        raise ValueError(f"XI{prefix + n} shape "
                                         f"does not match expected dimensions")
                    xi[n] = xi[n].flatten()
                else:
                    xi[n] = np.array([0.0])
            if not xi_any:
                xis[prefix] = None

        # Create all C-level workspaces
        self.wsp = lib.covar_workspace_init_from_xi(
            self.spin_a1, self.spin_a2, self.spin_b1, self.spin_b2,
            self.all_spins, self.lmax, self.lmax_mask,
            xis['']['00_1122'], xis['']['00_1221'],
            xis['']['02_1122'], xis['']['02_1221'],
            xis['']['22P_1122'], xis['']['22P_1221'],
            xis['']['22M_1122'], xis['']['22M_1221'])
        if xis['SN'] is not None:
            if self.wsp_SN is not None:
                lib.covar_workspace_free(self.wsp_SN)
                self.wsp_SN = None
            self.wsp_SN = lib.covar_workspace_init_from_xi(
                self.spin_a1, self.spin_a2, self.spin_b1, self.spin_b2,
                self.all_spins, self.lmax, self.lmax_mask,
                xis['SN']['00_1122'], xis['SN']['00_1221'],
                xis['SN']['02_1122'], xis['SN']['02_1221'],
                xis['SN']['22P_1122'], xis['SN']['22P_1221'],
                xis['SN']['22M_1122'], xis['SN']['22M_1221'])
            self.has_SN = np.array([self.wsp_SN.has_1122 > 0,
                                    self.wsp_SN.has_1221 > 0])
        if xis['NS'] is not None:
            if self.wsp_NS is not None:
                lib.covar_workspace_free(self.wsp_NS)
                self.wsp_NS = None
            self.wsp_NS = lib.covar_workspace_init_from_xi(
                self.spin_a1, self.spin_a2, self.spin_b1, self.spin_b2,
                self.all_spins, self.lmax, self.lmax_mask,
                xis['NS']['00_1122'], xis['NS']['00_1221'],
                xis['NS']['02_1122'], xis['NS']['02_1221'],
                xis['NS']['22P_1122'], xis['NS']['22P_1221'],
                xis['NS']['22M_1122'], xis['NS']['22M_1221'])
            self.has_NS = np.array([self.wsp_NS.has_1122 > 0,
                                    self.wsp_NS.has_1221 > 0])
        if xis['NN'] is not None:
            if self.wsp_NN is not None:
                lib.covar_workspace_free(self.wsp_NN)
                self.wsp_NN = None
            self.wsp_NN = lib.covar_workspace_init_from_xi(
                self.spin_a1, self.spin_a2, self.spin_b1, self.spin_b2,
                self.all_spins, self.lmax, self.lmax_mask,
                xis['NN']['00_1122'], xis['NN']['00_1221'],
                xis['NN']['02_1122'], xis['NN']['02_1221'],
                xis['NN']['22P_1122'], xis['NN']['22P_1221'],
                xis['NN']['22M_1122'], xis['NN']['22M_1221'])
            self.has_NN = np.array([self.wsp_NN.has_1122 > 0,
                                    self.wsp_NN.has_1221 > 0])
        f.close()

    def _compute_coupling_coefficients(self, fla1, fla2,
                                       flb1, flb2, *,
                                       all_spins=False,
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
        self.has_SN = np.array([False, False])
        self.has_NS = np.array([False, False])
        self.has_NN = np.array([False, False])
        if np.any([fla1.anisotropic_mask, fla2.anisotropic_mask,
                   flb1.anisotropic_mask, flb2.anisotropic_mask]):
            raise NotImplementedError("Covariance matrix estimation not "
                                      "implemented for anisotropic weights.")

        lmax = fla1.ainfo.lmax
        lmax_mask = fla1.ainfo_mask.lmax
        self.lmax = lmax
        self.lmax_mask = lmax_mask
        ut._toeplitz_sanity(l_toeplitz, l_exact, dl_band,
                            lmax, fla1, flb1)

        if self.wsp is not None:
            lib.covar_workspace_free(self.wsp)
            self.wsp = None

        def get_wsp(pcl_1122, pcl_1221, has_1122, has_1221):
            wsp = lib.covar_workspace_init_py(int(fla1.spin), int(fla2.spin),
                                              int(flb1.spin), int(flb2.spin),
                                              pcl_1122, pcl_1221,
                                              int(all_spins), 0,
                                              int(has_1122), int(has_1221),
                                              int(fla1.ainfo.lmax),
                                              int(fla1.ainfo_mask.lmax),
                                              l_toeplitz, l_exact, dl_band)
            return wsp

        s11_lm, _ = _get_mask_prod_alm(fla1, flb1)
        s22_lm, _ = _get_mask_prod_alm(fla2, flb2)
        s12_lm, _ = _get_mask_prod_alm(fla1, flb2)
        s21_lm, _ = _get_mask_prod_alm(fla2, flb1)
        pcl_mask_S11_S22 = hp.alm2cl(s11_lm, s22_lm, lmax=lmax_mask)
        pcl_mask_S12_S21 = hp.alm2cl(s12_lm, s21_lm, lmax=lmax_mask)

        self.wsp = get_wsp(pcl_mask_S11_S22, pcl_mask_S12_S21, 1, 1)

        # Compute coupling coefficients for catalog-based field combinations
        is_catalog_any = (_is_catalog(fla1) or _is_catalog(fla2) or
                          _is_catalog(flb1) or _is_catalog(flb2))
        if not is_catalog_any:
            return

        has_1122_NS = has_1221_NS = has_1122_SN = has_1221_SN = False
        has_1122_NN = has_1221_NN = False
        pcl_mask_N11_S22 = np.zeros_like(pcl_mask_S11_S22)
        pcl_mask_N12_S21 = np.zeros_like(pcl_mask_S11_S22)
        pcl_mask_S11_N22 = np.zeros_like(pcl_mask_S11_S22)
        pcl_mask_S12_N21 = np.zeros_like(pcl_mask_S11_S22)
        pcl_mask_N11_N22 = np.zeros_like(pcl_mask_S11_S22)
        pcl_mask_N12_N21 = np.zeros_like(pcl_mask_S11_S22)

        lmx = fla1.ainfo_mask.lmax
        n11_lm = None
        n22_lm = None

        if ((fla1 is flb1) or (fla1 is flb2)) and _is_catalog(fla1):
            n11_lm = fla1.get_catalog_variance_alm()
        if ((fla2 is flb1) or (fla2 is flb2)) and _is_catalog(fla2):
            if (n11_lm is not None) and (fla2 is fla1):
                n22_lm = n11_lm
            else:
                n22_lm = fla2.get_catalog_variance_alm()

        # Here's some horrible combinatorics
        if fla1 is flb1 and _is_catalog(fla1) and _is_catalog(flb1):
            has_1122_NS = True
            # Calculate pcl_mask_N11_S22
            pcl_mask_N11_S22 = hp.alm2cl(n11_lm, s22_lm, lmax=lmx)
            if fla2 is flb2 and _is_catalog(fla2) and _is_catalog(flb2):
                has_1122_NN = True
                # Calculate pcl_mask_N11_N22
                pcl_mask_N11_N22 = hp.alm2cl(n11_lm, n22_lm, lmax=lmx)
                if fla1 is fla2 and not fla1.is_clustering:
                    # Correct the four-point cumulant
                    prefac = 1/(4*np.pi)
                    corr_noise = prefac * np.sum(
                        (np.sum(fla1.field**2,
                                axis=0)/fla1.nmaps)**2
                        )
                    pcl_mask_N11_N22 = pcl_mask_N11_N22 - corr_noise
        if fla2 is flb2 and _is_catalog(fla2) and _is_catalog(flb2):
            has_1122_SN = True
            # Calculate pcl_mask_S11_N22
            pcl_mask_S11_N22 = hp.alm2cl(s11_lm, n22_lm)
        if fla1 is flb2 and _is_catalog(fla1) and _is_catalog(flb2):
            has_1221_NS = True
            # Calculate pcl_mask_N12_S21
            pcl_mask_N12_S21 = hp.alm2cl(n11_lm, s21_lm, lmax=lmx)
            if fla2 is flb1 and _is_catalog(fla2) and _is_catalog(flb1):
                has_1221_NN = True
                # Calcuate pcl_mask_N12_N21
                pcl_mask_N12_N21 = hp.alm2cl(n11_lm, n22_lm, lmax=lmx)
                if fla1 is fla2 and not fla1.is_clustering:
                    # Correct the four-point cumulant
                    prefac = 1/(4*np.pi)
                    corr_noise = prefac * np.sum(
                        (np.sum(fla1.field**2,
                                axis=0)/fla1.nmaps)**2
                        )
                    pcl_mask_N12_N21 = pcl_mask_N12_N21 - corr_noise
        if fla2 is flb1 and _is_catalog(fla1) and _is_catalog(flb1):
            has_1221_SN = True
            # Calculate pcl_mask_S12_N21
            pcl_mask_S12_N21 = hp.alm2cl(s12_lm, n22_lm)

        self.has_NS = np.array([has_1122_NS, has_1221_NS])
        self.has_SN = np.array([has_1122_SN, has_1221_SN])
        self.has_NN = np.array([has_1122_NN, has_1221_NN])

        # TODO: we are not taking advantage of cases
        # when fla1=fla2 or flb1=flb2
        if self.has_NS.any():
            self.wsp_NS = get_wsp(pcl_mask_N11_S22, pcl_mask_N12_S21,
                                  has_1122_NS, has_1221_NS)
        if self.has_SN.any():
            self.wsp_SN = get_wsp(pcl_mask_S11_N22, pcl_mask_S12_N21,
                                  has_1122_SN, has_1221_SN)
        if self.has_NN.any():
            self.wsp_NN = get_wsp(pcl_mask_N11_N22, pcl_mask_N12_N21,
                                  has_1122_NN, has_1221_NN)

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtCovarianceWorkspace`
        object to a FITS file.

        Args:
            fname (:obj:`str`): Output file name."""
        import fitsio as fts

        # Write header with global information
        f = fts.FITS(fname, 'rw', clobber=True)
        h = {'LMAX': self.wsp.lmax,
             'LMAX_MASK': self.wsp.lmax_mask,
             'ALL_SPINS': self.wsp.all_spins,
             'SPIN_A1': self.wsp.spin_a1,
             'SPIN_A2': self.wsp.spin_a2,
             'SPIN_B1': self.wsp.spin_b1,
             'SPIN_B2': self.wsp.spin_b2}
        f.write(np.ones((1, 1)), header=h, extname='CWSP_PRIMARY')

        def write_wsp(w, prefix):
            # This function writes the coupling coefficients of a
            # workspace to a FITS HDU.
            if w is None:
                return
            for i, n in enumerate(['00_1122', '00_1221',
                                   '02_1122', '02_1221',
                                   '22P_1122', '22P_1221',
                                   '22M_1122', '22M_1221']):
                exists, xi = lib.get_cw_xi(w, i, (w.lmax+1)**2)
                if exists:
                    f.write(xi.reshape((w.lmax+1, w.lmax+1)),
                            extname=f'XI{prefix + n}')

        # Write the coupling coefficients of all workspaces to the FITS file
        write_wsp(self.wsp, '')
        write_wsp(self.wsp_SN, 'SN')
        write_wsp(self.wsp_NS, 'NS')
        write_wsp(self.wsp_NN, 'NN')

        f.close()

    def gaussian_covariance(self, cla1b1, cla1b2, cla2b1, cla2b2,
                            wa, wb=None, coupled=False, spins=None):
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

            covar_SS = lib.comp_gaussian_covariance_coupled(
                self.wsp, int(spin_a1), int(spin_a2),
                int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                cla1b1, cla1b2, cla2b1, cla2b2, 0, 0, 0, 0,
                len_a * len_b
            )

            covar_NN = covar_NS = covar_SN = np.zeros_like(covar_SS)
            if self.has_NN.any():
                covar_NN = lib.comp_gaussian_covariance_coupled(
                    self.wsp_NN, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    np.ones_like(cla1b1), np.ones_like(cla1b2),
                    np.ones_like(cla2b1), np.ones_like(cla2b2),
                    1, 1, 1, 1, len_a * len_b)
            if self.has_NS.any():
                covar_NS = lib.comp_gaussian_covariance_coupled(
                    self.wsp_NS, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    np.ones_like(cla1b1), np.ones_like(cla1b2),
                    cla2b1, cla2b2, 1, 1, 0, 0, len_a * len_b)
            if self.has_SN.any():
                covar_SN = lib.comp_gaussian_covariance_coupled(
                    self.wsp_SN, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    cla1b1, cla1b2, np.ones_like(cla2b1),
                    np.ones_like(cla2b2), 0, 0, 1, 1,
                    len_a * len_b)
        else:
            len_a = wa.wsp.ncls * wa.wsp.bin.n_bands
            len_b = wb.wsp.ncls * wb.wsp.bin.n_bands

            covar_SS = lib.comp_gaussian_covariance(
                self.wsp, int(spin_a1), int(spin_a2),
                int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                cla1b1, cla1b2, cla2b1, cla2b2, 0, 0, 0, 0,
                len_a * len_b
            )

            covar_NN = covar_NS = covar_SN = np.zeros_like(covar_SS)
            if self.has_NN.any():
                covar_NN = lib.comp_gaussian_covariance(
                    self.wsp_NN, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    np.ones_like(cla1b1), np.ones_like(cla1b2),
                    np.ones_like(cla2b1), np.ones_like(cla2b2),
                    1, 1, 1, 1, len_a * len_b)
            if self.has_NS.any():
                covar_NS = lib.comp_gaussian_covariance(
                    self.wsp_NS, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    np.ones_like(cla1b1), np.ones_like(cla1b2),
                    cla2b1, cla2b2, 1, 1, 0, 0, len_a * len_b)
            if self.has_SN.any():
                covar_SN = lib.comp_gaussian_covariance(
                    self.wsp_SN, int(spin_a1), int(spin_a2),
                    int(spin_b1), int(spin_b2), wa.wsp, wb.wsp,
                    cla1b1, cla1b2, np.ones_like(cla2b1),
                    np.ones_like(cla2b2), 0, 0, 1, 1,
                    len_a * len_b)

        covar = covar_SS+covar_SN+covar_NS+covar_NN
        return covar.reshape([len_a, len_b])


class _NmtCovIdxHandler(object):
    _xi_comb = {(0, 0, 0, 0): [(0, 0), (0, 0)],
                (0, 0, 0, 2): [(0, 0), (0, 0)],
                (0, 0, 2, 0): [(0, 0), (0, 0)],
                (0, 0, 2, 2): [(0, 0), (0, 0)],
                (0, 2, 0, 0): [(0, 0), (0, 0)],
                (0, 2, 0, 2): [(0, 2), (0, 0)],
                (0, 2, 2, 0): [(0, 0), (0, 2)],
                (0, 2, 2, 2): [(0, 2), (0, 2)],
                (2, 0, 0, 0): [(0, 0), (0, 0)],
                (2, 0, 0, 2): [(0, 0), (0, 2)],
                (2, 0, 2, 0): [(0, 2), (0, 0)],
                (2, 0, 2, 2): [(0, 2), (0, 2)],
                (2, 2, 0, 0): [(0, 0), (0, 0)],
                (2, 2, 0, 2): [(0, 2), (0, 2)],
                (2, 2, 2, 0): [(0, 2), (0, 2)],
                (2, 2, 2, 2): [(2, 2), (2, 2)]}
    _icoup_spin2 = ['+', '--', '-+', 'Z',
                    '-+', '+', 'Z', '-+',
                    '--', 'Z', '+', '--',
                    'Z', '--', '-+', '+']

    def _coupling_index_signal(self, nma, nmb,
                               ia1, ib1, ia2, ib2):
        if nma == 1:
            if nmb == 1:  # TT,TT
                return '0'
            else:  # TX,TY
                if ib1 == ib2:  # TE,TE or TB,TB
                    return '0'
                else:
                    return 'Z'
        else:
            if nmb == 1:  # XT,YT
                if ia1 == ia2:  # ET,ET or BT,BT
                    return '0'
                else:
                    return 'Z'
            else:  # XY,WZ
                iclp = ib2+2*ia2+4*(ib1+2*ia1)
                return self._icoup_spin2[iclp]

    def _coupling_index_noise(self, nma, nmb, ia, ib):
        if nma != nmb:
            return 'Z'
        if nma == 1:
            return '0'
        return ['+', '-+', '--', '+'][ib+2*ia]

    def _pair_xi_index(self, ind1, ind2):
        xi = None
        sign = 1
        if (ind1, ind2) == ('0', '0'):
            xi = '00'
        elif (ind1, ind2) in [('0', '+'), ('+', '0')]:
            xi = '0s'
        elif (ind1, ind2) == ('+', '+'):
            xi = 'pp'
        elif (ind1, ind2) in [('-+', '-+'), ('--', '--')]:
            xi = 'mm'
            sign = -1
        elif (ind1, ind2) in [('-+', '--'), ('--', '-+')]:
            xi = 'mm'
        return sign, xi

    def _get_covar_terms_NN(self, ia, ib, ic, id, wick0):
        if wick0:
            ind1 = self._coupling_index_noise(
                self.nmaps[0], self.nmaps[2], ia, ic)
            ind2 = self._coupling_index_noise(
                self.nmaps[1], self.nmaps[3], ib, id)
        else:
            ind1 = self._coupling_index_noise(
                self.nmaps[0], self.nmaps[3], ia, id)
            ind2 = self._coupling_index_noise(
                self.nmaps[1], self.nmaps[2], ib, ic)
        sign, xi = self._pair_xi_index(ind1, ind2)
        return xi, sign

    def _get_covar_terms_SN(self, ia, ib, ic, id,
                            wick0, is_NS=False):
        indices = []
        xis = []
        signs = []
        if is_NS:
            iA, iB = ib, ia
            nmpA, nmpB = self.nmaps[1], self.nmaps[0]
        else:
            iA, iB = ia, ib
            nmpA, nmpB = self.nmaps[0], self.nmaps[1]
        if wick0:
            nmpx, nmpy = self.nmaps[2], self.nmaps[3]
            ix, iy = ic, id
        else:
            nmpx, nmpy = self.nmaps[3], self.nmaps[2]
            ix, iy = id, ic
        for iA2 in range(nmpA):
            for ix2 in range(nmpx):
                ind1 = self._coupling_index_signal(
                    nmpA, nmpx, iA, ix, iA2, ix2)
                ind2 = self._coupling_index_noise(
                    nmpB, nmpy, iB, iy)
                sign, xi = self._pair_xi_index(ind1, ind2)
                if xi is not None:
                    indices.append(ix2+nmpx*iA2)
                    xis.append(xi)
                    signs.append(sign)
        indices = np.array(indices, dtype=int)
        signs = np.array(signs)
        return indices, xis, signs

    def _get_covar_terms_SS(self,
                            ia, ib, ic, id, wick0):
        indices = []
        xis = []
        signs = []
        for ia2 in range(self.nmaps[0]):
            for ib2 in range(self.nmaps[1]):
                for ic2 in range(self.nmaps[2]):
                    for id2 in range(self.nmaps[3]):
                        if wick0:
                            ind1 = self._coupling_index_signal(
                                self.nmaps[0], self.nmaps[2],
                                ia, ic, ia2, ic2)
                            ind2 = self._coupling_index_signal(
                                self.nmaps[1], self.nmaps[3],
                                ib, id, ib2, id2)
                        else:
                            ind1 = self._coupling_index_signal(
                                self.nmaps[0], self.nmaps[3],
                                ia, id, ia2, id2)
                            ind2 = self._coupling_index_signal(
                                self.nmaps[1], self.nmaps[2],
                                ib, ic, ib2, ic2)
                        sign, xi = self._pair_xi_index(ind1, ind2)
                        if xi is not None:
                            if wick0:
                                indices.append([ic2+self.nmaps[2]*ia2,
                                                id2+self.nmaps[3]*ib2])
                            else:
                                indices.append([id2+self.nmaps[3]*ia2,
                                                ic2+self.nmaps[2]*ib2])
                            xis.append(xi)
                            signs.append(sign)
        indices = np.array(indices, dtype=int)
        signs = np.array(signs)
        return indices, xis, signs

    def __init__(self, sa1, sa2, sb1, sb2):
        self.spins = (sa1, sa2, sb1, sb2)
        self.nmaps = [2 if s > 0 else 1 for s in self.spins]
        if not (set(self.spins) <= set((0, 2))):
            raise ValueError("Covariance matrix estimation is only "
                             "implemented for spin-0 and spin-2 fields.")
        # List of spin combinations for which MASTER coefficients
        # are needed
        self.xi_sp_combs = self._xi_comb.get(self.spins)

        # Precalculate all combinations needed
        # for the signal-signal covariance
        self.info_SS_1122 = []
        self.info_SS_1221 = []
        self.info_SN_1122 = []
        self.info_SN_1221 = []
        self.info_NS_1122 = []
        self.info_NS_1221 = []
        self.info_NN_1122 = []
        self.info_NN_1221 = []
        for ia in range(self.nmaps[0]):
            for ib in range(self.nmaps[1]):
                inf1122_SS = []
                inf1221_SS = []
                inf1122_SN = []
                inf1221_SN = []
                inf1122_NS = []
                inf1221_NS = []
                inf1122_NN = []
                inf1221_NN = []
                for ic in range(self.nmaps[2]):
                    for id in range(self.nmaps[3]):
                        # SS
                        ids, xs, sgns = self._get_covar_terms_SS(
                            ia, ib, ic, id, wick0=True)
                        inf1122_SS.append([ids, sgns, xs])
                        ids, xs, sgns = self._get_covar_terms_SS(
                            ia, ib, ic, id, wick0=False)
                        inf1221_SS.append([ids, sgns, xs])
                        # SN
                        ids, xs, sgns = self._get_covar_terms_SN(
                            ia, ib, ic, id, wick0=True, is_NS=False)
                        inf1122_SN.append([ids, sgns, xs])
                        ids, xs, sgns = self._get_covar_terms_SN(
                            ia, ib, ic, id, wick0=False, is_NS=False)
                        inf1221_SN.append([ids, sgns, xs])
                        # NS
                        ids, xs, sgns = self._get_covar_terms_SN(
                            ia, ib, ic, id, wick0=True, is_NS=True)
                        inf1122_NS.append([ids, sgns, xs])
                        ids, xs, sgns = self._get_covar_terms_SN(
                            ia, ib, ic, id, wick0=False, is_NS=True)
                        inf1221_NS.append([ids, sgns, xs])
                        # NN
                        xi, sgn = self._get_covar_terms_NN(
                            ia, ib, ic, id, wick0=True)
                        inf1122_NN.append([sgn, xi])
                        xi, sgn = self._get_covar_terms_NN(
                            ia, ib, ic, id, wick0=False)
                        inf1221_NN.append([sgn, xi])
                self.info_SS_1122.append(inf1122_SS)
                self.info_SS_1221.append(inf1221_SS)
                self.info_SN_1122.append(inf1122_SN)
                self.info_SN_1221.append(inf1221_SN)
                self.info_NS_1122.append(inf1122_NS)
                self.info_NS_1221.append(inf1221_NS)
                self.info_NN_1122.append(inf1122_NN)
                self.info_NN_1221.append(inf1221_NN)


def _sqz(d, k, i):
    x = d.get(k)
    if x is not None:
        if k == '00':
            assert x.ndim == 3
            return x[i]
        else:
            assert x.ndim == 4
            return x[0][i]
    return None


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
        fname (:obj:`str`): Input file name. If not `None`, the values of
            all input fields will be ignored, and all mode-coupling
            coefficients will be read from file."""
    def __init__(self, fla1, fla2, flb1=None, flb2=None,
                 l_toeplitz=-1, l_exact=-1,
                 dl_band=-1, fname=None):
        if (fname is not None):
            self._read_from(fname)
            self._post_init()
            return

        if flb1 is None:
            flb1 = fla1
        if flb2 is None:
            flb2 = fla2

        self.spin_a1 = fla1.spin
        self.spin_a2 = fla2.spin
        self.spin_b1 = flb1.spin
        self.spin_b2 = flb2.spin
        self.lmax = fla1.ainfo.lmax
        self.lmax_mask = fla1.ainfo_mask.lmax
        self.l_toeplitz = l_toeplitz
        self.l_exact = l_exact
        self.dl_band = dl_band

        self._post_init()
        self._compute_coupling_coefficients(fla1, fla2, flb1, flb2)

    def _post_init(self):
        self._idxh = _NmtCovIdxHandler(self.spin_a1, self.spin_a2,
                                       self.spin_b1, self.spin_b2)
        self.nclsa = self._idxh.nmaps[0] * self._idxh.nmaps[1]
        self.nclsb = self._idxh.nmaps[2] * self._idxh.nmaps[3]

    def _get_covariance_xis(self, *, pcl_1122=None, pcl_1221=None):
        has_1122 = pcl_1122 is not None
        has_1221 = pcl_1221 is not None

        sp_comb = self._idxh.xi_sp_combs
        if sp_comb is None:
            raise ValueError("Invalid combination of spins: "
                             f"({self.spin_a1}, {self.spin_a2},"
                             f"{self.spin_b1}, {self.spin_b2})")
        xi_eq = sp_comb[0] == sp_comb[1]
        i_1122, i_1221 = -1, -1
        if has_1122:
            i_1122 = 0
            if has_1221:
                i_1221 = 1
        elif has_1221:
            i_1221 = 0

        # Compute all MASTER coefficients needed for the covariance matrix
        xis = [{}, {}]
        if xi_eq:  # Both 1122 and 1221 require the same MASTER coefficients
            pcls = []
            if has_1122:
                pcls.append(pcl_1122)
            if has_1221:
                pcls.append(pcl_1221)
            pcls = np.array(pcls)
            d = mt.get_master_coefficients(
                pcls, self.lmax, sp_comb[0][0], sp_comb[0][1],
                is_teb=False, pure_any=False,
                l_toeplitz=self.l_toeplitz, l_exact=self.l_exact,
                dl_band=self.dl_band)
            if has_1122:
                xis[0] = {k: _sqz(d, k, i_1122)
                          for k in ['00', '0s', 'pp', 'mm']}
            if has_1221:
                xis[1] = {k: _sqz(d, k, i_1221)
                          for k in ['00', '0s', 'pp', 'mm']}
        else:  # Different coefficients needed for 1122 and 1221
            if has_1122:
                d = mt.get_master_coefficients(
                    pcl_1122[None, :], self.lmax, sp_comb[0][0], sp_comb[0][1],
                    is_teb=False, pure_any=False,
                    l_toeplitz=self.l_toeplitz, l_exact=self.l_exact,
                    dl_band=self.dl_band)
                xis[0] = {k: _sqz(d, k, 0) for k in ['00', '0s', 'pp', 'mm']}
            if has_1221:
                d = mt.get_master_coefficients(
                    pcl_1221[None, :], self.lmax, sp_comb[1][0], sp_comb[1][1],
                    is_teb=False, pure_any=False,
                    l_toeplitz=self.l_toeplitz, l_exact=self.l_exact,
                    dl_band=self.dl_band)
                xis[1] = {k: _sqz(d, k, 0) for k in ['00', '0s', 'pp', 'mm']}

        return xis

    @classmethod
    def from_fields(cls, fla1, fla2, flb1=None, flb2=None, *,
                    l_toeplitz=-1, l_exact=-1, dl_band=-1):
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
        """
        return cls(fla1=fla1, fla2=fla2, flb1=flb1, flb2=flb2,
                   l_toeplitz=l_toeplitz, l_exact=l_exact, dl_band=dl_band)

    @classmethod
    def from_file(cls, fname):
        """ Creates an :obj:`NmtCovarianceWorkspace` object from the
        mode-coupling coefficients stored in a FITS file.
        See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name."""
        return cls(None, None, fname=fname)

    def _read_from(self, fname):
        """ Reads the contents of an :obj:`NmtCovarianceWorkspace`
        object from a FITS file.

        Args:
            fname (:obj:`str`): Input file name."""
        import fitsio as fts

        f = fts.FITS(fname)
        h = f['CWSP_PRIMARY'].read_header()
        self.lmax = h['LMAX']
        self.lmax_mask = h['LMAX_MASK'] if 'LMAX_MASK' in h else self.lmax
        if 'ALL_SPINS' in h:
            self.spin_a1 = h['SPIN_A1']
            self.spin_a2 = h['SPIN_A2']
            self.spin_b1 = h['SPIN_B1']
            self.spin_b2 = h['SPIN_B2']
        else:
            self.spin_a1 = self.spin_a2 = self.spin_b1 = self.spin_b2 = 0
        self.l_toeplitz = h['L_TOEPLITZ'] if 'L_TOEPLITZ' in h else -1
        self.l_exact = h['L_EXACT'] if 'L_EXACT' in h else -1
        self.dl_band = h['DL_BAND'] if 'DL_BAND' in h else -1
        self.has_SN = np.array([False, False])
        self.has_NS = np.array([False, False])
        self.has_NN = np.array([False, False])

        self.xiSS = [{}, {}]
        self.xiSN = [{}, {}]
        self.xiNS = [{}, {}]
        self.xiNN = [{}, {}]

        for xilist, xiname in zip([self.xiSS, self.xiSN, self.xiNS, self.xiNN],
                                  ['', 'SN', 'NS', 'NN']):
            for n1, n2 in zip(['00', '0s', 'pp', 'mm'],
                              ['00', '02', '22P', '22M']):
                for i, wick in enumerate(['1122', '1221']):
                    name = 'XI'+xiname+n2+'_'+wick
                    xilist[i][n1] = None
                    if name in f:
                        xi = f[name].read()
                        if xi.shape != (self.lmax+1, self.lmax+1):
                            raise ValueError(
                                f"{name} shape "
                                "does not match expected dimensions")
                        xilist[i][n1] = xi
        f.close()
        self.has_SN = [np.any([self.xiSN[i][k] is not None
                               for k in ['00', '0s', 'pp', 'mm']])
                       for i in range(2)]
        self.has_SN = np.array(self.has_SN)
        self.has_NS = [np.any([self.xiNS[i][k] is not None
                               for k in ['00', '0s', 'pp', 'mm']])
                       for i in range(2)]
        self.has_NS = np.array(self.has_NS)
        self.has_NN = [np.any([self.xiNN[i][k] is not None
                               for k in ['00', '0s', 'pp', 'mm']])
                       for i in range(2)]
        self.has_NN = np.array(self.has_NN)

    def _compute_coupling_coefficients(self, fla1, fla2, flb1, flb2):
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
        """
        self.has_SN = np.array([False, False])
        self.has_NS = np.array([False, False])
        self.has_NN = np.array([False, False])
        self.xiSN = [{k: None for k in ['00', '0s', 'pp', 'mm']}
                     for _ in range(2)]
        self.xiNS = [{k: None for k in ['00', '0s', 'pp', 'mm']}
                     for _ in range(2)]
        self.xiNN = [{k: None for k in ['00', '0s', 'pp', 'mm']}
                     for _ in range(2)]
        if np.any([fla1.anisotropic_mask, fla2.anisotropic_mask,
                   flb1.anisotropic_mask, flb2.anisotropic_mask]):
            raise NotImplementedError("Covariance matrix estimation not "
                                      "implemented for anisotropic weights.")

        lmax = fla1.ainfo.lmax
        lmax_mask = fla1.ainfo_mask.lmax
        self.lmax = lmax
        self.lmax_mask = lmax_mask
        ut._toeplitz_sanity(self.l_toeplitz, self.l_exact, self.dl_band,
                            lmax, fla1, flb1)

        s11_lm, _ = _get_mask_prod_alm(fla1, flb1)
        s22_lm, _ = _get_mask_prod_alm(fla2, flb2)
        s12_lm, _ = _get_mask_prod_alm(fla1, flb2)
        s21_lm, _ = _get_mask_prod_alm(fla2, flb1)
        pcl_mask_S11_S22 = hp.alm2cl(s11_lm, s22_lm, lmax=lmax_mask)
        pcl_mask_S12_S21 = hp.alm2cl(s12_lm, s21_lm, lmax=lmax_mask)

        self.xiSS = self._get_covariance_xis(pcl_1122=pcl_mask_S11_S22,
                                             pcl_1221=pcl_mask_S12_S21)

        # Compute coupling coefficients for catalog-based field combinations
        is_catalog_any = (_is_catalog(fla1) or _is_catalog(fla2) or
                          _is_catalog(flb1) or _is_catalog(flb2))
        if not is_catalog_any:
            return

        has_1122_NS = has_1221_NS = has_1122_SN = has_1221_SN = False
        has_1122_NN = has_1221_NN = False
        pcl_mask_N11_S22 = None
        pcl_mask_N12_S21 = None
        pcl_mask_S11_N22 = None
        pcl_mask_S12_N21 = None
        pcl_mask_N11_N22 = None
        pcl_mask_N12_N21 = None

        lmx = fla1.ainfo_mask.lmax
        n11_lm = None
        n22_lm = None

        if ((fla1 is flb1) or (fla1 is flb2)) and _is_catalog(fla1):
            n11_lm = fla1.get_catalog_variance_alm()
        if ((fla2 is flb1) or (fla2 is flb2)) and _is_catalog(fla2):
            if (n11_lm is not None) and (fla2 is fla1):
                n22_lm = n11_lm
            else:
                n22_lm = fla2.get_catalog_variance_alm()

        # Here's some horrible combinatorics
        if fla1 is flb1 and _is_catalog(fla1) and _is_catalog(flb1):
            has_1122_NS = True
            # Calculate pcl_mask_N11_S22
            pcl_mask_N11_S22 = hp.alm2cl(n11_lm, s22_lm, lmax=lmx)
            if fla2 is flb2 and _is_catalog(fla2) and _is_catalog(flb2):
                has_1122_NN = True
                # Calculate pcl_mask_N11_N22
                pcl_mask_N11_N22 = hp.alm2cl(n11_lm, n22_lm, lmax=lmx)
                if fla1 is fla2 and not fla1.is_clustering:
                    # Correct the four-point cumulant
                    prefac = 1/(4*np.pi)
                    corr_noise = prefac * np.sum(
                        (np.sum(fla1.field**2,
                                axis=0)/fla1.nmaps)**2
                        )
                    pcl_mask_N11_N22 = pcl_mask_N11_N22 - corr_noise
        if fla2 is flb2 and _is_catalog(fla2) and _is_catalog(flb2):
            has_1122_SN = True
            # Calculate pcl_mask_S11_N22
            pcl_mask_S11_N22 = hp.alm2cl(s11_lm, n22_lm)
        if fla1 is flb2 and _is_catalog(fla1) and _is_catalog(flb2):
            has_1221_NS = True
            # Calculate pcl_mask_N12_S21
            pcl_mask_N12_S21 = hp.alm2cl(n11_lm, s21_lm, lmax=lmx)
            if fla2 is flb1 and _is_catalog(fla2) and _is_catalog(flb1):
                has_1221_NN = True
                # Calcuate pcl_mask_N12_N21
                pcl_mask_N12_N21 = hp.alm2cl(n11_lm, n22_lm, lmax=lmx)
                if fla1 is fla2 and not fla1.is_clustering:
                    # Correct the four-point cumulant
                    prefac = 1/(4*np.pi)
                    corr_noise = prefac * np.sum(
                        (np.sum(fla1.field**2,
                                axis=0)/fla1.nmaps)**2
                        )
                    pcl_mask_N12_N21 = pcl_mask_N12_N21 - corr_noise
        if fla2 is flb1 and _is_catalog(fla1) and _is_catalog(flb1):
            has_1221_SN = True
            # Calculate pcl_mask_S12_N21
            pcl_mask_S12_N21 = hp.alm2cl(s12_lm, n22_lm)

        self.has_NS = np.array([has_1122_NS, has_1221_NS])
        self.has_SN = np.array([has_1122_SN, has_1221_SN])
        self.has_NN = np.array([has_1122_NN, has_1221_NN])

        # TODO: we are not taking advantage of cases
        # when fla1=fla2 or flb1=flb2
        if self.has_NS.any():
            self.xiNS = self._get_covariance_xis(
                pcl_1122=pcl_mask_N11_S22, pcl_1221=pcl_mask_N12_S21)
        if self.has_SN.any():
            self.xiSN = self._get_covariance_xis(
                pcl_1122=pcl_mask_S11_N22, pcl_1221=pcl_mask_S12_N21)
        if self.has_NN.any():
            self.xiNN = self._get_covariance_xis(
                pcl_1122=pcl_mask_N11_N22, pcl_1221=pcl_mask_N12_N21)

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtCovarianceWorkspace`
        object to a FITS file.

        Args:
            fname (:obj:`str`): Output file name."""
        import fitsio as fts

        # Write header with global information
        f = fts.FITS(fname, 'rw', clobber=True)
        h = {'LMAX': self.lmax,
             'LMAX_MASK': self.lmax_mask,
             'SPIN_A1': self.spin_a1,
             'SPIN_A2': self.spin_a2,
             'SPIN_B1': self.spin_b1,
             'SPIN_B2': self.spin_b2,
             'L_TOEPLITZ': self.l_toeplitz,
             'L_EXACT': self.l_exact,
             'DL_BAND': self.dl_band}
        f.write(np.ones((1, 1)), header=h, extname='CWSP_PRIMARY')

        def write_xi(w, prefix):
            # This function writes the coupling coefficients of a
            # workspace to a FITS HDU.
            if w is None:
                return
            for n1, n2 in zip(['00', '0s', 'pp', 'mm'],
                              ['00', '02', '22P', '22M']):
                for i, wick in enumerate(['1122', '1221']):
                    xi = w[i][n1]
                    if xi is not None:
                        f.write(xi.reshape((self.lmax+1, self.lmax+1)),
                                extname=f'XI{prefix + n2}_{wick}')

        # Write the coupling coefficients of all workspaces to the FITS file
        write_xi(self.xiSS, '')
        write_xi(self.xiSN, 'SN')
        write_xi(self.xiNS, 'NS')
        write_xi(self.xiNN, 'NN')

        f.close()

    def gaussian_covariance(self, cla1b1, cla1b2, cla2b1, cla2b2,
                            wa, wb=None, coupled=False):
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
        """
        nm_a1, nm_a2, nm_b1, nm_b2 = self._idxh.nmaps

        if wb is None:
            wb = wa

        if (wa.ncls != nm_a1*nm_a2) or (wb.ncls != nm_b1*nm_b2):
            raise ValueError("Field spins do not match input workspaces")

        if (len(cla1b1) != nm_a1*nm_b1) or \
           (len(cla1b2) != nm_a1*nm_b2) or \
           (len(cla2b1) != nm_a2*nm_b1) or \
           (len(cla2b2) != nm_a2*nm_b2):
            raise ValueError("Field spins do not match input power"
                             "spectrum shapes")

        if (len(cla1b1[0]) < self.lmax + 1) or \
           (len(cla1b2[0]) < self.lmax + 1) or \
           (len(cla2b1[0]) < self.lmax + 1) or \
           (len(cla2b2[0]) < self.lmax + 1):
            raise ValueError("Input C_ls have a weird length. "
                             f"Expected {self.lmax+1}, but got "
                             f"({len(cla1b1[0])}, {len(cla1b2[0])}, "
                             f"{len(cla2b1[0])}, {len(cla2b2[0])}).")
        if (wa.lmax != self.lmax) or (wb.lmax != self.lmax):
            raise ValueError("Input workspaces have a different lmax "
                             "than the covariance workspace."
                             f" Expected {self.lmax}, but got ({wa.lmax}, {wb.lmax}).")  # noqa: E501

        # Symmetrized power spectra
        cla1b1 = np.array(cla1b1)
        cla1b2 = np.array(cla1b2)
        cla2b1 = np.array(cla2b1)
        cla2b2 = np.array(cla2b2)
        clprod_1122 = cla1b1[:, None, :, None] * cla2b2[None, :, None, :]
        clprod_1122 = 0.5*(clprod_1122 + np.swapaxes(clprod_1122, 2, 3))
        clprod_1221 = cla1b2[:, None, :, None] * cla2b1[None, :, None, :]
        clprod_1221 = 0.5*(clprod_1221 + np.swapaxes(clprod_1221, 2, 3))
        clt1122_NS = 0.5*(cla2b2[:, None, :]+cla2b2[:, :, None])
        clt1221_NS = 0.5*(cla2b1[:, None, :]+cla2b1[:, :, None])
        clt1122_SN = 0.5*(cla1b1[:, None, :]+cla1b1[:, :, None])
        clt1221_SN = 0.5*(cla1b2[:, None, :]+cla1b2[:, :, None])
        nba = self.lmax+1 if coupled else wa.nbands
        nbb = self.lmax+1 if coupled else wb.nbands
        len_a = wa.ncls * nba
        len_b = wb.ncls * nbb

        covar = np.zeros([self.nclsa, self.nclsb, nba, nbb])
        for ia in range(self.nclsa):
            for ib in range(self.nclsb):
                cov = np.zeros([self.lmax+1, self.lmax+1])
                # Signal=signal
                for (i, j), sign, sxi in zip(*self._idxh.info_SS_1122[ia][ib]):
                    cov += (sign * clprod_1122[i, j] * self.xiSS[0][sxi])
                for (i, j), sign, sxi in zip(*self._idxh.info_SS_1221[ia][ib]):
                    cov += (sign * clprod_1221[i, j] * self.xiSS[1][sxi])
                # Signal-noise
                if self.has_SN[0]:  # 1122
                    for i, sign, sxi in zip(*self._idxh.info_SN_1122[ia][ib]):
                        cov += sign*self.xiSN[0][sxi]*clt1122_SN[i]
                if self.has_SN[1]:  # 1221
                    for i, sign, sxi in zip(*self._idxh.info_SN_1221[ia][ib]):
                        cov += sign*self.xiSN[1][sxi]*clt1221_SN[i]
                # Noise-signal
                if self.has_NS[0]:  # 1122
                    for i, sign, sxi in zip(*self._idxh.info_NS_1122[ia][ib]):
                        cov += sign*self.xiNS[0][sxi]*clt1122_NS[i]
                if self.has_NS[1]:  # 1221
                    for i, sign, sxi in zip(*self._idxh.info_NS_1221[ia][ib]):
                        cov += sign*self.xiNS[1][sxi]*clt1221_NS[i]
                # Noise-noise
                if self.has_NN[0]:  # 1122
                    sign, sxi = self._idxh.info_NN_1122[ia][ib]
                    if sxi is not None:
                        cov += sign*self.xiNN[0][sxi]
                if self.has_NN[1]:  # 1221
                    sign, sxi = self._idxh.info_NN_1221[ia][ib]
                    if sxi is not None:
                        cov += sign*self.xiNN[1][sxi]

                # Bin if needed
                if not coupled:
                    # Bin rows and transpose
                    cov = np.array([wb.bins.bin_cell(row) for row in cov]).T
                    # Bin former columns and transpose back
                    cov = np.array([wa.bins.bin_cell(col) for col in cov]).T

                covar[ia, ib, :, :] = cov

        # [Nl, Np, Nl, Np]
        covar = np.transpose(covar, axes=[2, 0, 3, 1])
        # Flatten both sides
        covar = covar.reshape([len_a, len_b])
        # Decouple if needed
        if not coupled:
            imcma = np.linalg.inv(wa.mcm_binned)
            imcmb = np.linalg.inv(wb.mcm_binned)
            covar = np.einsum('ik,jl,kl->ij', imcma, imcmb, covar)
        return covar


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
    if ((spin_a1 != cw.spin_a1) or (spin_a2 != cw.spin_a2) or
            (spin_b1 != cw.spin_b1) or (spin_b2 != cw.spin_b2)):
        raise ValueError("Requested spins do not match those used "
                         "to initialize the workspace")
    return cw.gaussian_covariance(cla1b1, cla1b2, cla2b1, cla2b2,
                                  wa, wb=wb, coupled=coupled)


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


def _is_mask_catalog(f):
    if isinstance(f, NmtFieldCatalog):
        if f.mask is not None:
            return False
        return True
    return False


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
            phi_a = 1 if fla.mask is not None else fla.get_cloud_kernel(lmax)
            phi_b = 1 if flb.mask is not None else flb.get_cloud_kernel(lmax)
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
