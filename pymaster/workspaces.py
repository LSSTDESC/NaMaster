from pymaster import nmtlib as lib
import pymaster.utils as ut
import pymaster.master as mst
from pymaster.bins import NmtBin
import numpy as np
import healpy as hp
import warnings


class NmtWorkspace(object):
    """ :obj:`NmtWorkspace` objects are used to compute and store the
    mode-coupling matrix associated with an incomplete sky coverage,
    and used in the MASTER algorithm. :obj:`NmtWorkspace` objects can be
    initialised from a pair of :class:`~pymaster.field.NmtField` objects
    and an :class:`~pymaster.bins.NmtBin` object, containing information
    about the masks involved and the :math:`\\ell` binning scheme, or
    read from a file where the mode-coupling matrix was stored.

    We recommend using the class methods :meth:`from_fields` and
    :meth:`from_file` to create new :obj:`NmtWorkspace` objects,
    rather than using the main constructor.

    Args:
        fl1 (:class:`~pymaster.field.NmtField`): First field being
            correlated.
        fl2 (:class:`~pymaster.field.NmtField`): Second field being
            correlated.
        bins (:class:`~pymaster.bins.NmtBin`): Binning scheme.
        is_teb (:obj:`bool`): If ``True``, all mode-coupling matrices
            (0-0,0-s,s-s) will be computed at the same time. In this
            case, ``fl1`` must be a spin-0 field and ``fl2`` must be
            spin-s.
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
        fname (:obj:`str`): Input file name. If not `None`, this
            workspace will be initialised from file, and the values
            of ``fl1``, ``fl2``, and ``bin`` will be ignored.
        read_unbinned_MCM (:obj:`bool`): If ``False``, the unbinned
            mode-coupling matrix will not be read. This can save
            significant IO time.
        normalization (:obj:`str`): Normalization convention to use for
            the bandpower window functions. Two options supported:
            `'MASTER'` (default) corresponds to the standard inversion
            of the binned mode-coupling matrix. `'FKP'` simply divides
            by the mean of the mask product, forcing a unit response
            to an input white spectrum.
    """
    def __init__(self, fl1=None, fl2=None, bins=None, is_teb=False,
                 l_toeplitz=-1, l_exact=-1, dl_band=-1, fname=None,
                 normalization='MASTER'):
        self.mcm = None
        self.mcm_binned = None
        self.lmax = None
        self.lmax_mask = None
        self.nbands = None
        self.bpws = None
        self.bins = None
        self.spin1 = None
        self.spin2 = None
        self.aniso1 = None
        self.aniso2 = None
        self.nmaps1 = None
        self.nmaps2 = None
        self.ncls = None
        self.beam1 = None
        self.beam2 = None
        self.pure_e1 = None
        self.pure_b1 = None
        self.pure_e2 = None
        self.pure_b2 = None
        self.is_teb = None
        self.l_toeplitz = None
        self.l_exact = None
        self.dl_band = None
        self.pcl_mask = None
        self.norm_type = None
        self.normalization = None
        self.wawb = None

        if ((fl1 is None) and (fl2 is None) and (bins is None) and
                (fname is None)):
            warnings.warn("The bare constructor for `NmtWorkspace` "
                          "objects is deprecated and will be removed "
                          "in future versions of NaMaster. Consider "
                          "using the class methods "
                          "`from_fields` and `from_file`, or pass "
                          "the necessary arguments to the constructor.",
                          category=DeprecationWarning)
            return

        if (fname is not None):
            self.read_from(fname)
            return

        self.compute_coupling_matrix(
            fl1, fl2, bins, is_teb=is_teb,
            l_toeplitz=l_toeplitz, l_exact=l_exact, dl_band=dl_band,
            normalization=normalization)

    @classmethod
    def from_fields(cls, fl1, fl2, bins, is_teb=False,
                    l_toeplitz=-1, l_exact=-1, dl_band=-1,
                    normalization='MASTER'):
        """ Creates an :obj:`NmtWorkspace` object containing the
        mode-coupling matrix associated with the cross-power spectrum of
        two :class:`~pymaster.field.NmtField` s
        and an :class:`~pymaster.bins.NmtBin` binning scheme. Note that
        the mode-coupling matrix will only contain :math:`\\ell` s up
        to the maximum multipole included in the bandpowers, which should
        match the :math:`\\ell_{\\rm max}` of the fields as well.

        Args:
            fl1 (:class:`~pymaster.field.NmtField`): First field to correlate.
            fl2 (:class:`~pymaster.field.NmtField`): Second field to correlate.
            bins (:class:`~pymaster.bins.NmtBin`): Binning scheme.
            is_teb (:obj:`bool`): If ``True``, all mode-coupling matrices
                (0-0,0-s,s-s) will be computed at the same time. In this
                case, ``fl1`` must be a spin-0 field and ``fl2`` must be
                spin-s.
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
            normalization (:obj:`str`): Normalization convention to use
                for the bandpower window functions. Two options
                supported: `'MASTER'` (default) corresponds to the
                standard inversion of the binned mode-coupling matrix.
                `'FKP'` simply divides by the mean of the mask product,
                forcing a unit response to an input white spectrum.
        """
        return cls(fl1=fl1, fl2=fl2, bins=bins, is_teb=is_teb,
                   l_toeplitz=l_toeplitz, l_exact=l_exact,
                   dl_band=dl_band, normalization=normalization)

    @classmethod
    def from_file(cls, fname):
        """ Creates an :obj:`NmtWorkspace` object from a mode-coupling
        matrix stored in a FITS file. See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name.
        """
        return cls(fname=fname)

    def read_from(self, fname):
        """ Reads the contents of an :obj:`NmtWorkspace` object from a
        FITS file.

        Args:
            fname (:obj:`str`): Input file name.
        """
        import fitsio as fts

        f = fts.FITS(fname)
        # Write header information
        h = f['WSP_PRIMARY'].read_header()
        self.lmax = h['LMAX']
        self.lmax_mask = h['LMAX_MASK']
        self.is_teb = bool(h['IS_TEB'])
        self.ncls = h['NCLS']
        self.norm_type = h['NORM_TYPE'] if 'NORM_TYPE' in h else 0
        self.normalization = 'MASTER' if self.norm_type == 0 else 'FKP'
        self.wawb = h['WAWB'] if 'WAWB' in h else 0.0
        s1 = h['SPIN1'] if 'SPIN1' in h else -1
        s2 = h['SPIN2'] if 'SPIN2' in h else -1
        if s1 < 0 or s2 < 0:
            if self.ncls == 1:
                s1 = s2 = 0
            elif self.ncls == 2:
                s1 = 0
                s2 = 2
            elif self.ncls == 4:
                s1 = s2 = 2
        self.spin1 = s1
        self.spin2 = s2
        self.aniso1 = bool(h['ANISO1']) if 'ANISO1' in h else False
        self.aniso2 = bool(h['ANISO2']) if 'ANISO2' in h else False
        self.nmaps1 = 2 if self.spin1 > 0 else 1
        self.nmaps2 = 2 if self.spin2 > 0 else 1
        self.pure_e1 = bool(h['PURE_E1']) if 'PURE_E1' in h else False
        self.pure_b1 = bool(h['PURE_B1']) if 'PURE_B1' in h else False
        self.pure_e2 = bool(h['PURE_E2']) if 'PURE_E2' in h else False
        self.pure_b2 = bool(h['PURE_B2']) if 'PURE_B2' in h else False
        self.l_toeplitz = h['L_TOEPLITZ'] if 'L_TOEPLITZ' in h else -1
        self.l_exact = h['L_EXACT'] if 'L_EXACT' in h else -1
        self.dl_band = h['DL_BAND'] if 'DL_BAND' in h else -1

        # Read the mode-coupling matrix
        self.mcm = f['WSP_PRIMARY'].read()

        # Read beams
        hdu = f['BEAMS']
        # This is only to support legacy files. To be removed
        # in the future.
        if 'BEAMS' in hdu.get_colnames():
            # Assume both fields had the same beam
            beams = hdu['BEAMS'].read()
            self.beam1 = np.sqrt(beams)
            self.beam2 = np.sqrt(beams)
        else:
            self.beam1 = hdu['BEAM1'].read()
            self.beam2 = hdu['BEAM2'].read()

        # Read mask PCL
        self.pcl_mask = f['PCL_MASKS']['PCL_MASKS'].read()

        # Read binning scheme
        extname = 'BINS' if 'BINS' in f else 'BANDPOWERS'
        self.bins = NmtBin._from_fits_file(f, extname=extname)
        self.nbands = self.bins.get_n_bands()

        # Get bandpowers
        self.bpws, self.mcm_binned = self._postproc_mcm(self.mcm)
        f.close()

    def update_beams(self, beam1, beam2):
        """ Update beams associated with this mode-coupling matrix.
        This is significantly faster than recomputing the matrix from
        scratch.

        Args:
            beam1 (`array`): First beam, in the form of a 1D array
                with the beam sampled at all integer multipoles up
                to the maximum :math:`\\ell` with which this
                workspace was initialised.
            beam2 (`array`): Second beam.
        """
        b1arr = isinstance(beam1, (list, tuple, np.ndarray))
        b2arr = isinstance(beam2, (list, tuple, np.ndarray))
        if ((not b1arr) or (not b2arr)):
            raise ValueError("The new beams must be provided as arrays")

        if ((len(beam1) != self.lmax+1) or
                (len(beam2) != self.lmax+1)):
            raise ValueError("The new beams must go up to"
                             " ell = %d" % self.lmax)
        self.beam1 = beam1
        self.beam2 = beam2
        # Rebin MCM
        self.bpws, self.mcm_binned = self._postproc_mcm(self.mcm)

    def update_bins(self, bins):
        """ Update binning associated with this mode-coupling matrix.
        This is significantly faster than recomputing the matrix from
        scratch.

        Args:
            bins (:class:`~pymaster.bins.NmtBin`): New binning scheme.
        """
        if bins.bin.ell_max != self.lmax:
            raise ValueError("The new binning scheme has a different "
                             "maximum multipole than the fields used to "
                             "compute this workspace.")
        self.bins = bins
        self.nbands = bins.get_n_bands()
        # Rebin MCM
        self.bpws, self.mcm_binned = self._postproc_mcm(self.mcm)

    def _postproc_mcm(self, mcm):
        mcm_binned = self.bins._bin_mcm(mcm, self.norm_type, self.wawb,
                                        self.beam1, self.beam2,
                                        oneside=False)
        mcm_binned = mcm_binned.reshape([self.ncls*self.nbands,
                                         self.ncls*self.nbands])
        bpws = self.bins._bin_mcm(mcm, self.norm_type, self.wawb,
                                  self.beam1, self.beam2, oneside=True)
        # TODO: regularise inversion for catalog-based cases, where one
        # of the eigenvalues (corresponding to white noise) should be
        imcm_binned = np.linalg.inv(mcm_binned)

        bpws = np.dot(imcm_binned, bpws)
        return bpws, mcm_binned

    def _get_mcm_anisotropic(self, aniso1, aniso2, pclm_00,
                             pclm_0e, pclm_0b, pclm_e0, pclm_b0,
                             pclm_ee, pclm_eb, pclm_be, pclm_bb):
        sg00 = 1
        m00 = mst.get_general_coupling_matrix(
            pclm_00*sg00, self.spin1, self.spin2,
            self.spin1, self.spin2, parity="both")
        m0e = m0b = me0 = mb0 = mee = meb = mbe = mbb = 0*m00
        if aniso2:
            sg0s = (-1)**self.spin2
            m0e = mst.get_general_coupling_matrix(
                pclm_0e*sg0s, self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
            m0b = mst.get_general_coupling_matrix(
                pclm_0b*sg0s, self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
        if aniso1:
            sgs0 = (-1)**self.spin1
            me0 = mst.get_general_coupling_matrix(
                pclm_e0*sgs0, -self.spin1, self.spin2,
                self.spin1, self.spin2, parity="both")
            mb0 = mst.get_general_coupling_matrix(
                pclm_b0*sgs0, self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
        if aniso1 and aniso2:
            sgss = (-1)**(self.spin1+self.spin2)
            mee = mst.get_general_coupling_matrix(
                pclm_ee*sgss, -self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
            meb = mst.get_general_coupling_matrix(
                pclm_eb*sgss, -self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
            mbe = mst.get_general_coupling_matrix(
                pclm_be*sgss, -self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")
            mbb = mst.get_general_coupling_matrix(
                pclm_bb*sgss, -self.spin1, -self.spin2,
                self.spin1, self.spin2, parity="both")

        mcm = np.zeros([self.lmax+1, self.ncls,
                        self.lmax+1, self.ncls])
        if (self.spin1 == 0) and (self.spin2 != 0):
            # s=0 kills odd-parity terms
            mcm[:, 0, :, 0] = m00[0]-m0e[0]
            mcm[:, 0, :, 1] = -m0b[0]
            mcm[:, 1, :, 0] = -m0b[0]
            mcm[:, 1, :, 1] = m00[0]+m0e[0]
        if (self.spin1 != 0) and (self.spin2 == 0):
            # s=0 kills odd-parity terms
            mcm[:, 0, :, 0] = m00[0]-me0[0]
            mcm[:, 0, :, 1] = -mb0[0]
            mcm[:, 1, :, 0] = -mb0[0]
            mcm[:, 1, :, 1] = m00[0]+me0[0]
        if (self.spin1 != 0) and (self.spin2 != 0):
            mcm[:, 0, :, 0] = m00[0]-m0e[0]-me0[0]+mee[0]+mbb[1]
            mcm[:, 0, :, 1] = -m0b[0]+meb[0]-mb0[1]-mbe[1]
            mcm[:, 0, :, 2] = -mb0[0]+mbe[0]-m0b[1]-meb[1]
            mcm[:, 0, :, 3] = mbb[0]+m00[1]+m0e[1]+me0[1]+mee[1]
            mcm[:, 1, :, 0] = -m0b[0]+meb[0]+mb0[1]-mbe[1]
            mcm[:, 1, :, 1] = m00[0]+m0e[0]-me0[0]-mee[0]-mbb[1]
            mcm[:, 1, :, 2] = mbb[0]-m00[1]+m0e[1]-me0[1]+mee[1]
            mcm[:, 1, :, 3] = -mb0[0]-mbe[0]+m0b[1]+meb[1]
            mcm[:, 2, :, 0] = -mb0[0]+mbe[0]+m0b[1]-meb[1]
            mcm[:, 2, :, 1] = mbb[0]-m00[1]-m0e[1]+me0[1]+mee[1]
            mcm[:, 2, :, 2] = m00[0]-m0e[0]+me0[0]-mee[0]-mbb[1]
            mcm[:, 2, :, 3] = -m0b[0]-meb[0]+mb0[1]+mbe[1]
            mcm[:, 3, :, 0] = mbb[0]+m00[1]-m0e[1]-me0[1]+mee[1]
            mcm[:, 3, :, 1] = -mb0[0]-mbe[0]-m0b[1]+meb[1]
            mcm[:, 3, :, 2] = -m0b[0]-meb[0]-mb0[1]+mbe[1]
            mcm[:, 3, :, 3] = m00[0]+m0e[0]+me0[0]+mee[0]+mbb[1]
        return mcm.reshape([self.ncls*(self.lmax+1),
                            self.ncls*(self.lmax+1)])

    def _get_mcm(self):
        pure_any = (self.pure_e1 or self.pure_b1 or
                    self.pure_e2 or self.pure_b2)
        d = mst.get_master_coefficients(self.pcl_mask, self.lmax,
                                        spin1=self.spin1, spin2=self.spin2,
                                        is_teb=self.is_teb,
                                        pure_any=pure_any,
                                        l_toeplitz=self.l_toeplitz,
                                        l_exact=self.l_exact,
                                        dl_band=self.dl_band)
        ls = np.arange(self.lmax+1)
        ipe1 = int(self.pure_e1)
        ipb1 = int(self.pure_b1)
        ipe2 = int(self.pure_e2)
        ipb2 = int(self.pure_b2)
        sign = (-1)**(self.spin1+self.spin2)

        mcm = np.zeros([self.lmax+1, self.ncls,
                        self.lmax+1, self.ncls])
        lfac = (2*ls+1)[None, :]
        if self.ncls == 1:
            sign = 1
            mcm[:, 0, :, 0] = d['00']*lfac*sign
        elif self.ncls == 2:
            mcm[:, 0, :, 0] = d['0s'][ipe1+ipe2]*lfac*sign
            mcm[:, 1, :, 1] = d['0s'][ipb1+ipb2]*lfac*sign
        elif self.ncls == 4:
            mcm[:, 0, :, 0] = d['pp'][ipe1+ipe2]*lfac*sign
            mcm[:, 1, :, 1] = d['pp'][ipe1+ipb2]*lfac*sign
            mcm[:, 2, :, 2] = d['pp'][ipb1+ipe2]*lfac*sign
            mcm[:, 3, :, 3] = d['pp'][ipb1+ipb2]*lfac*sign
            mcm[:, 0, :, 3] = d['mm'][ipe1+ipe2]*lfac*sign
            mcm[:, 1, :, 2] = -d['mm'][ipe1+ipb2]*lfac*sign
            mcm[:, 2, :, 1] = -d['mm'][ipb1+ipe2]*lfac*sign
            mcm[:, 3, :, 0] = d['mm'][ipb1+ipb2]*lfac*sign
        elif self.ncls == 7:
            mcm[:, 0, :, 0] = d['00']*lfac
            mcm[:, 1, :, 1] = d['0s'][ipe2]*lfac*sign
            mcm[:, 2, :, 2] = d['0s'][ipb2]*lfac*sign
            mcm[:, 3, :, 3] = d['pp'][ipe2+ipe2]*lfac
            mcm[:, 4, :, 4] = d['pp'][ipe2+ipb2]*lfac
            mcm[:, 5, :, 5] = d['pp'][ipb2+ipe2]*lfac
            mcm[:, 6, :, 6] = d['pp'][ipb2+ipb2]*lfac
            mcm[:, 3, :, 6] = d['mm'][ipe2+ipe2]*lfac
            mcm[:, 4, :, 5] = -d['mm'][ipe2+ipb2]*lfac
            mcm[:, 5, :, 4] = -d['mm'][ipb2+ipe2]*lfac
            mcm[:, 6, :, 3] = d['mm'][ipb2+ipb2]*lfac
        return mcm.reshape([self.ncls*(self.lmax+1),
                            self.ncls*(self.lmax+1)])

    def compute_coupling_matrix(self, fl1, fl2, bins, is_teb=False,
                                l_toeplitz=-1, l_exact=-1, dl_band=-1,
                                normalization='MASTER'):
        """ Computes the mode-coupling matrix associated with the
        cross-power spectrum of two :class:`~pymaster.field.NmtField` s
        and an :class:`~pymaster.bins.NmtBin` binning scheme. Note that
        the mode-coupling matrix will only contain :math:`\\ell` s up
        to the maximum multipole included in the bandpowers, which should
        match the :math:`\\ell_{\\rm max}` of the fields as well.

        Args:
            fl1 (:class:`~pymaster.field.NmtField`): First field to correlate.
            fl2 (:class:`~pymaster.field.NmtField`): Second field to correlate.
            bins (:class:`~pymaster.bins.NmtBin`): Binning scheme.
            is_teb (:obj:`bool`): If ``True``, all mode-coupling matrices
                (0-0,0-s,s-s) will be computed at the same time. In this
                case, ``fl1`` must be a spin-0 field and ``fl2`` must be
                spin-s.
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
            normalization (:obj:`str`): Normalization convention to use for
                the bandpower window functions. Two options supported:
                `'MASTER'` (default) corresponds to the standard inversion
                of the binned mode-coupling matrix. `'FKP'` simply divides
                by the mean of the mask product, forcing a unit response to
                an input white spectrum.
        """
        if not fl1.is_compatible(fl2, strict=False):
            raise ValueError("Fields have incompatible pixelizations.")
        if fl1.ainfo.lmax != bins.lmax:
            raise ValueError(f"Maximum multipoles in bins ({bins.lmax}) "
                             f"and fields ({fl1.ainfo.lmax}) "
                             "are not the same.")

        if is_teb and ((fl1.spin != 0) or (fl2.spin == 0)):
            raise ValueError("If is_teb=True, fl1 must be spin-0 and fl2 "
                             "must be spin-s.")

        self.spin1 = fl1.spin
        self.spin2 = fl2.spin
        self.beam1 = fl1.beam
        self.beam2 = fl2.beam
        self.pure_e1 = fl1.pure_e
        self.pure_b1 = fl1.pure_b
        self.pure_e2 = fl2.pure_e
        self.pure_b2 = fl2.pure_b
        self.lmax = fl1.ainfo.lmax
        self.lmax_mask = fl1.ainfo_mask.lmax
        self.is_teb = is_teb
        self.bins = bins
        self.l_toeplitz = l_toeplitz
        self.l_exact = l_exact
        self.dl_band = dl_band
        self.aniso1 = fl1.anisotropic_mask
        self.aniso2 = fl2.anisotropic_mask
        self.nbands = bins.get_n_bands()

        anisotropic_mask_any = self.aniso1 or self.aniso2
        if anisotropic_mask_any and (l_toeplitz >= 0):
            raise NotImplementedError("Toeplitz approximation not "
                                      "implemented for anisotropic masks.")
        ut._toeplitz_sanity(l_toeplitz, l_exact, dl_band,
                            bins.bin.ell_max, fl1, fl2)

        # Get mask PCL
        alm1 = fl1.get_mask_alms()
        Nw = 0
        if fl2 is fl1:
            alm2 = alm1
            Nw = fl1.Nw
        else:
            alm2 = fl2.get_mask_alms()
        pcl_mask = hp.alm2cl(alm1, alm2, lmax=fl1.ainfo_mask.lmax)
        if anisotropic_mask_any:
            pcl0 = pcl_mask * 0
            pclm_00 = pcl_mask
            pclm_0e = pclm_0b = pclm_e0 = pclm_b0 = pcl0
            pclm_ee = pclm_eb = pclm_be = pclm_bb = pcl0
            if self.aniso1:
                alm1a = fl1.get_anisotropic_mask_alms()
            if self.aniso2:
                alm2a = fl2.get_anisotropic_mask_alms()
            if self.aniso2:
                pclm_0e = hp.alm2cl(alm1, alm2a[0], lmax=fl1.ainfo_mask.lmax)
                pclm_0b = hp.alm2cl(alm1, alm2a[1], lmax=fl1.ainfo_mask.lmax)
            if self.aniso1:
                pclm_e0 = hp.alm2cl(alm1a[0], alm2, lmax=fl1.ainfo_mask.lmax)
                pclm_b0 = hp.alm2cl(alm1a[1], alm2, lmax=fl1.ainfo_mask.lmax)
                if self.aniso2:
                    pclm_ee = hp.alm2cl(alm1a[0], alm2a[0],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_eb = hp.alm2cl(alm1a[0], alm2a[1],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_be = hp.alm2cl(alm1a[1], alm2a[0],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_bb = hp.alm2cl(alm1a[1], alm2a[1],
                                        lmax=fl1.ainfo_mask.lmax)

        if normalization == 'MASTER':
            self.norm_type = 0
        elif normalization == 'FKP':
            self.norm_type = 1
        else:
            raise ValueError(f"Unknown normalization type {normalization}. "
                             "Allowed options are 'MASTER' and 'FKP'.")
        self.normalization = normalization

        self.wawb = 0
        if self.norm_type == 1:
            if fl1.is_catalog or fl2.is_catalog:
                if fl2 is fl1:
                    self.wawb = fl1.Nw
                else:
                    raise ValueError("Cannot use FKP normalisation for "
                                     "catalog fields unless they are the "
                                     "same field.")
            else:
                msk1 = fl1.get_mask()
                msk2 = fl2.get_mask()
                self.wawb = fl1.minfo.si.dot_map(msk1, msk2)/(4*np.pi)

        self.nmaps1 = 2 if self.spin1 > 0 else 1
        self.nmaps2 = 2 if self.spin2 > 0 else 1
        if self.is_teb:
            self.ncls = 7
        else:
            self.ncls = self.nmaps1 * self.nmaps2

        self.pcl_mask = pcl_mask.flatten()-Nw
        if anisotropic_mask_any:
            self.mcm = self._get_mcm_anisotropic(
                self.aniso1, self.aniso2, pclm_00,
                pclm_0e, pclm_0b, pclm_e0, pclm_b0,
                pclm_ee, pclm_eb, pclm_be, pclm_bb)
        else:
            self.mcm = self._get_mcm()
        self.bpws, self.mcm_binned = self._postproc_mcm(self.mcm)

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtWorkspace` object
        to a FITS file.

        Args:
            fname (:obj:`str`): Output file name
        """
        import fitsio as fts

        # Write header with global information
        f = fts.FITS(fname, 'rw', clobber=True)

        h = {'LMAX': self.lmax,
             'LMAX_MASK': self.lmax_mask,
             'IS_TEB': self.is_teb,
             'NCLS': self.ncls,
             'NORM_TYPE': self.norm_type,
             'WAWB': self.wawb,
             'SPIN1': self.spin1,
             'SPIN2': self.spin2,
             'ANISO1': self.aniso1,
             'ANISO2': self.aniso2,
             'PURE_E1': self.pure_e1,
             'PURE_B1': self.pure_b1,
             'PURE_E2': self.pure_e2,
             'PURE_B2': self.pure_b2,
             'L_TOEPLITZ': self.l_toeplitz,
             'L_EXACT': self.l_exact,
             'DL_BAND': self.dl_band}
        f.write(self.mcm, header=h, extname='WSP_PRIMARY')

        # Write beams
        ls = np.arange(self.lmax+1, dtype=np.int32)
        f.write([ls, self.beam1, self.beam2],
                names=['L', 'BEAM1', 'BEAM2'],
                extname='BEAMS')
        # Write mask PCL
        f.write([ls, self.pcl_mask],
                names=['L', 'PCL_MASKS'],
                extname='PCL_MASKS')

        # Write binning scheme
        self.bins._to_fits_file(f, extname='BANDPOWERS')
        f.close()

    def get_coupling_matrix(self):
        """ Returns the currently stored mode-coupling matrix.

        Returns:
            (`array`): Mode-coupling matrix. The matrix will have shape
            ``(nrows,nrows)``, with ``nrows = n_cls * n_ells``, where
            ``n_cls`` is the number of power spectra (1, 2 or 4 for
            spin 0-0, spin 0-s and spin s-s correlations), and
            ``n_ells = lmax + 1``, and ``lmax`` is the maximum multipole
            associated with this workspace. The assumed ordering of power
            spectra is such that the ``L``-th element of the ``i``-th power
            spectrum be stored with index ``L * n_cls + i``.
        """
        return self.mcm

    def update_coupling_matrix(self, new_matrix):
        """
        Updates the stored mode-coupling matrix. The new matrix
        (``new_matrix``) must have shape ``(nrows,nrows)``.
        See docstring of :meth:`~NmtWorkspace.get_coupling_matrix` for an
        explanation of the size and ordering of this matrix.

        Args:
            new_matrix (`array`): Matrix that will replace the mode-coupling
                matrix.
        """
        rowsize = (self.lmax + 1) * self.ncls
        if new_matrix.shape != (rowsize, rowsize):
            raise ValueError("Input matrix has an inconsistent shape. "
                             f"Expected {(rowsize, rowsize)}, "
                             f"but got {new_matrix.shape}.")
        self.mcm = new_matrix
        # Bin new MCM
        self.bpws, self.mcm_binned = self._postproc_mcm(self.mcm)

    def couple_cell(self, cl_in):
        """ Convolves a set of input power spectra with a coupling matrix
        (see Eq. 9 of the NaMaster paper).

        Args:
            cl_in (`array`): Set of input power spectra. The number of power
                spectra must correspond to the spins of the two fields that
                this :obj:`NmtWorkspace` object was initialized with (i.e. 1
                for two spin-0 fields, 2 for one spin-0 field and one spin-s
                field, and 4 for two spin-s fields).

        Returns:
            (`array`): Mode-coupled power spectra.
        """
        if (len(cl_in) != self.ncls) or \
           (len(cl_in[0]) < self.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape. "
                             f"Expected ({self.ncls}, {self.lmax+1}), "
                             f"but got {cl_in.shape}.")

        # Shorten C_ells if they're too long
        cl_in = np.array(cl_in)[:, :self.lmax+1]
        # Multiply by beams
        cl_in = cl_in * (self.beam1*self.beam2)[None, :]
        cl1d = np.dot(self.mcm, cl_in.T.flatten())
        clout = np.reshape(cl1d, [self.lmax + 1, self.ncls]).T
        return clout

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """ Decouples a set of pseudo-:math:`C_\\ell` power spectra into a
        set of bandpowers by inverting the binned coupling matrix (se Eq.
        16 of the NaMaster paper).

        Args:
            cl_in (`array`): Set of input power spectra. The number of power
                spectra must correspond to the spins of the two fields that
                this :obj:`NmtWorkspace` object was initialized with (i.e. 1
                for two spin-0 fields, 2 for one spin-0 field and one spin-s
                field, 4 for two spin-s fields, and 7 if this
                :obj:`NmtWorkspace` was created using ``is_teb=True``).
            cl_bias (`array`): Bias to the power spectrum associated with
                contaminant residuals (optional). This can be computed through
                :func:`deprojection_bias`.
            cl_noise (`array`): Noise bias (i.e. angular
                pseudo-:math:`C_\\ell` of masked noise realizations).

        Returns:
            (`array`): Set of decoupled bandpowers.
        """
        if (len(cl_in) != self.ncls) or \
           (len(cl_in[0]) < self.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape. "
                             f"Expected ({self.ncls}, {self.lmax+1}), "
                             f"but got {cl_in.shape}")
        if cl_bias is not None:
            if (len(cl_bias) != self.ncls) or \
               (len(cl_bias[0]) < self.lmax + 1):
                raise ValueError(
                    "Input bias power spectrum has wrong shape. "
                    f"Expected ({self.ncls}, {self.lmax+1}), "
                    f"but got {cl_bias.shape}")
            clb = cl_bias.copy()
        else:
            clb = np.zeros_like(cl_in)
        if cl_noise is not None:
            if (len(cl_noise) != self.ncls) or \
               (len(cl_noise[0]) < self.lmax + 1):
                raise ValueError(
                    "Input noise power spectrum has wrong shape. "
                    f"Expected ({self.ncls}, {self.lmax+1}), "
                    f"but got {cl_noise.shape}")
            cln = cl_noise.copy()
        else:
            cln = np.zeros_like(cl_in)

        cltot = cl_in-clb-cln  # [ncls, lmax+1]
        cl1d = self.bins.bin_cell(cltot)  # [ncls, nband]
        cl1d = np.linalg.solve(self.mcm_binned, cl1d.T.flatten())
        clout = np.reshape(cl1d, [self.nbands, self.ncls]).T

        return clout

    def get_bandpower_windows(self):
        """ Get bandpower window functions. Convolve the theory power spectra
        with these as an alternative to the combination of function calls \
        ``w.decouple_cell(w.couple_cell(cls_theory))``. See Eqs. 18 and
        19 of the NaMaster paper.

        As an example consider the power spectrum of two spin-2 fields. In
        this case, the estimated bandpowers would have shape ``[4, n_bpw]``,
        where ``n_bpw`` is the number of bandpowers. The unbinned power
        spectra would have shape ``[4, lmax+1]``, where ``lmax`` is the
        maximum multipole under study. The bandpower window functions would
        then have shape ``[4, n_bpw, 4, lmax+1]`` and, for example, the
        window function at indices ``[0, b1, 3, ell2]`` quantifies the
        amount of :math:`BB` power at :math:`\\ell=` ``ell2`` that is leaked
        into the ``b1``-th :math:`EE` bandpower.

        Returns:
            (`array`): Bandpower windows with shape \
                ``(n_cls, n_bpws, n_cls, lmax+1)``.
        """
        return np.transpose(self.bpws.reshape([self.nbands,
                                               self.ncls,
                                               self.lmax+1,
                                               self.ncls]),
                            axes=[1, 0, 3, 2])


class NmtWorkspaceFlat(object):
    """ :obj:`NmtWorkspaceFlat` objects are used to compute and store the
    mode-coupling matrix associated with an incomplete sky coverage, and
    used in the flat-sky version of the MASTER algorithm. When initialized,
    this object is practically empty. The information describing the
    coupling matrix must be computed or read from a file afterwards.
    """
    def __init__(self):
        self.wsp = None

    def __del__(self):
        if self.wsp is not None:
            if lib.workspace_flat_free is not None:
                lib.workspace_flat_free(self.wsp)
            self.wsp = None

    def read_from(self, fname):
        """ Reads the contents of an :obj:`NmtWorkspaceFlat` object from a
        FITS file.

        Args:
            fname (:obj:`str`): Input file name.
        """
        if self.wsp is not None:
            lib.workspace_flat_free(self.wsp)
            self.wsp = None

        import fitsio as fts
        f = fts.FITS(fname)

        # Workspace info
        h = f['WSP_PRIMARY'].read_header()
        lmax = h['LMAX']
        lcut_X_I = h['ELLCUT_X_I']
        lcut_X_F = h['ELLCUT_X_F']
        lcut_Y_I = h['ELLCUT_Y_I']
        lcut_Y_F = h['ELLCUT_Y_F']
        pure_e1 = h['PURE_E1']
        pure_e2 = h['PURE_E2']
        pure_b1 = h['PURE_B1']
        pure_b2 = h['PURE_B2']
        is_teb = h['IS_TEB']
        ncls = h['NCLS']
        mcm = f['WSP_PRIMARY'].read()

        # Flatsky info
        h = f['FS_INFO'].read_header()
        nx = h['NX']
        ny = h['NY']
        npix = h['NPIX']
        lx = h['LX']
        ly = h['LY']
        pixsize = h['PIXSIZE']
        dell = h['DELL']
        i_dell = h['I_DELL']
        lmin = f['FS_INFO'].read()['L_MIN']

        # N_cells
        n_cells = f['N_CELLS'].read()['N_CELLS']

        # Binned mcm
        mcm_binned = d = f['MCM_BINNED'].read()
        mcm_binned_gsl = f['MCM_BINNED_GSL'].read()
        mcm_perm = f['MCM_PERM'].read()

        # Binning
        d = f['BINS_SUMMARY'].read()
        ell_0 = d['ELL_0']
        ell_f = d['ELL_F']

        f.close()

        self.wsp = lib.workspace_flat_from_data(
            int(ncls), lmax,
            lcut_X_I, lcut_X_F,
            lcut_Y_I, lcut_Y_F,
            int(pure_e1), int(pure_e2),
            int(pure_b1), int(pure_b2), int(is_teb),
            n_cells.astype(np.int32),
            int(nx), int(ny), npix, lx, ly, pixsize, dell, i_dell,
            lmin.astype(np.float64),
            ell_0.astype(np.float64), ell_f.astype(np.float64),
            mcm.astype(np.float64), mcm_binned.astype(np.float64),
            mcm_binned_gsl.astype(np.float64), mcm_perm.astype(np.int32))

    def compute_coupling_matrix(self, fl1, fl2, bins, ell_cut_x=[1., -1.],
                                ell_cut_y=[1., -1.], is_teb=False):
        """ Computes mode-coupling matrix associated with the cross-power
        spectrum of two :class:`~pymaster.field.NmtFieldFlat` s and an
        :class:`~pymaster.bins.NmtBinFlat` binning scheme.

        Args:
            fl1 (:class:`~pymaster.field.NmtFieldFlat`): First field to
                correlate.
            fl2 (:class:`~pymaster.field.NmtFieldFlat`): Second field to
                correlate.
            bin (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme.
            ell_cut_x (`array`): Sequence of two elements determining the
                range of :math:`l_x` to remove from the calculation. No
                Fourier modes removed by default.
            ell_cut_y (`array`): Sequence of two elements determining the
                range of :math:`l_y` to remove from the calculation. No
                Fourier modes removed by default.
            is_teb (:obj:`bool`): If ``True``, all mode-coupling matrices
                (0-0,0-s,s-s) will be computed at the same time. In this
                case, ``fl1`` must be a spin-0 field and ``fl2`` must be
                spin-s.
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
        """ Writes the contents of an :obj:`NmtWorkspaceFlat` object
        to a FITS file.

        Args:
            fname (:obj:`str`): Output file name.
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before "
                               "writing")

        import fitsio as fts

        nbands = self.wsp.bin.n_bands
        nells = self.wsp.fs.n_ell
        ncls = self.wsp.ncls
        lmax = self.wsp.lmax
        lcx_i, lcx_f, lcy_i, lcy_f = lib.wsp_flat_get_lcuts(self.wsp, 4)
        mcm = lib.wsp_flat_get_mcm(self.wsp, int(1), int(0),
                                   ncls**2*nbands*nells).reshape([ncls*nbands,
                                                                  ncls*nells])
        # Write header with global information
        f = fts.FITS(fname, 'rw', clobber=True)
        h = {'LMAX': lmax,
             'ELLCUT_X_I': lcx_i,
             'ELLCUT_X_F': lcx_f,
             'ELLCUT_Y_I': lcy_i,
             'ELLCUT_Y_F': lcy_f,
             'PURE_E1': self.wsp.pe1,
             'PURE_E2': self.wsp.pe2,
             'PURE_B1': self.wsp.pb1,
             'PURE_B2': self.wsp.pb2,
             'IS_TEB': self.wsp.is_teb,
             'NCLS': self.wsp.ncls}
        f.write(mcm, header=h, extname='WSP_PRIMARY')

        h = {'NX': self.wsp.fs.nx,
             'NY': self.wsp.fs.ny,
             'NPIX': self.wsp.fs.npix,
             'LX': self.wsp.fs.lx,
             'LY': self.wsp.fs.ly,
             'PIXSIZE': self.wsp.fs.pixsize,
             'DELL': self.wsp.fs.dell,
             'I_DELL': self.wsp.fs.i_dell}
        lmin = lib.wsp_flat_get_fs_ellmin(self.wsp, nells)
        f.write([lmin], names=['L_MIN'], header=h, extname='FS_INFO')

        n_cells = lib.wsp_flat_get_n_cells(self.wsp, nbands)
        f.write([n_cells], names=['N_CELLS'], extname='N_CELLS')

        mcm_binned = lib.wsp_flat_get_mcm(
            self.wsp, int(0), int(0),
            ncls**2*nbands**2).reshape([ncls*nbands,
                                        ncls*nbands])
        f.write(mcm_binned, extname='MCM_BINNED')

        mcm_binned_gsl = lib.wsp_flat_get_mcm(
            self.wsp, int(0), int(1),
            (ncls*nbands)**2).reshape([ncls*nbands,
                                       ncls*nbands])
        f.write(mcm_binned_gsl, extname='MCM_BINNED_GSL')

        mcm_perm = lib.wsp_flat_get_perm(self.wsp, ncls*nbands)
        f.write(mcm_perm, extname='MCM_PERM')

        ell_0, ell_f = lib.wsp_flat_get_bin_ls(
            self.wsp, 2*nbands).reshape([2, -1])
        f.write([ell_0, ell_f], names=['ELL_0', 'ELL_F'],
                extname='BINS_SUMMARY')

        f.close()

    def couple_cell(self, ells, cl_in):
        """ Convolves a set of input power spectra with a coupling
        matrix (see Eq. 42 of the NaMaster paper).


        Args:
            ells (`array`): List of multipoles on which the input power
                spectra are defined.
            cl_in (`array`): Set of input power spectra. The number of power
                spectra must correspond to the spins of the two fields that
                this :obj:`NmtWorkspace` object was initialized with (i.e. 1
                for two spin-0 fields, 2 for one spin-0 field and one spin-s
                field, and 4 for two spin-s fields).

        Returns:
            (`array`): Mode-coupled power spectra. The coupled power spectra \
                are returned at the multipoles returned by calling \
                :meth:`~pymaster.field.NmtFieldFlat.get_ell_sampling` for \
                any of the fields that were used to generate the workspace.
        """
        if (len(cl_in) != self.wsp.ncls) or (len(cl_in[0]) != len(ells)):
            raise ValueError("Input power spectrum has wrong shape. "
                             f"Expected ({self.wsp.ncls}, {len(ells)}, "
                             f"but got {cl_in.shape}.")
        cl1d = lib.couple_cell_py_flat(
            self.wsp, ells, cl_in, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])
        return clout

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """ Decouples a set of pseudo-:math:`C_\\ell` power spectra into a
        set of bandpowers by inverting the binned coupling matrix (see
        Eq. 47 of the NaMaster paper).

        Args:
            cl_in (`array`): Set of input power spectra. The number of power
                spectra must correspond to the spins of the two fields that
                this :obj:`NmtWorkspace` object was initialized with (i.e. 1
                for two spin-0 fields, 2 for one spin-0 field and one spin-s
                field, 4 for two spin-s fields, and 7 if this
                :obj:`NmtWorkspace` was created using ``is_teb=True``). These
                power spectra must be defined at the multipoles returned by
                :meth:`~pymaster.field.NmtFieldFlat.get_ell_sampling` for
                any of the fields used to create the workspace.
            cl_bias (`array`): Bias to the power spectrum associated with
                contaminant residuals (optional). This can be computed through
                :func:`deprojection_bias_flat`.
            cl_noise (`array`): Noise bias (i.e. angular
                pseudo-:math:`C_\\ell` of masked noise realisations).

        Returns:
            (`array`): Set of decoupled bandpowers.
        """
        if (len(cl_in) != self.wsp.ncls) or \
           (len(cl_in[0]) != self.wsp.bin.n_bands):
            raise ValueError(
                "Input power spectrum has wrong shape. "
                f"Expected ({self.wsp.ncls}, {self.wsp.bin.n_bands}), "
                f"but got {cl_in.shape}")
        if cl_bias is not None:
            if (len(cl_bias) != self.wsp.ncls) or \
               (len(cl_bias[0]) != self.wsp.bin.n_bands):
                raise ValueError(
                    "Input bias power spectrum has wrong shape. "
                    f"Expected ({self.wsp.ncls}, {self.wsp.bin.n_bands}), "
                    f"but got {cl_bias.shape}.")
            clb = cl_bias.copy()
        else:
            clb = np.zeros_like(cl_in)
        if cl_noise is not None:
            if (len(cl_noise) != self.wsp.ncls) or \
               (len(cl_noise[0]) != self.wsp.bin.n_bands):
                raise ValueError(
                    "Input noise power spectrum has wrong shape. "
                    f"Expected ({self.wsp.ncls}, {self.wsp.bin.n_bands}), "
                    f"but got {cl_noise.shape}.")
            cln = cl_noise.copy()
        else:
            cln = np.zeros_like(cl_in)

        cl1d = lib.decouple_cell_py_flat(
            self.wsp, cl_in, cln, clb, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])

        return clout


def deprojection_bias(f1, f2, cl_guess, n_iter=None):
    """ Computes the bias associated to contaminant removal to the
    cross-pseudo-:math:`C_\\ell` of two fields. See Eq. 26 in the NaMaster
    paper.

    Args:
        f1 (:class:`~pymaster.field.NmtField`): First field to correlate.
        f2 (:class:`~pymaster.field.NmtField`): Second field to correlate.
        cl_guess (`array`): Array of power spectra corresponding to a
            best-guess of the true power spectra of ``f1`` and ``f2``.
        n_iter (:obj:`int`): Number of iterations when computing
            :math:`a_{\\ell m}` s. See docstring of
            :class:`~pymaster.field.NmtField`.

    Returns:
        (`array`): Deprojection bias pseudo-:math:`C_\\ell`.
    """
    if n_iter is None:
        n_iter = ut.nmt_params.n_iter_default

    if not f1.is_compatible(f2):
        raise ValueError("Fields have incompatible pixelizations.")

    def purify_if_needed(fld, mp):
        if fld.pure_e or fld.pure_b:
            # Compute mask alms if needed
            amask = fld.get_mask_alms()
            return fld._purify(fld.mask, amask, mp,
                               n_iter=n_iter, return_maps=False,
                               task=[fld.pure_e, fld.pure_b])
        else:
            return ut.map2alm(mp*fld.mask[None, :], fld.spin,
                              fld.minfo, fld.ainfo, n_iter=n_iter)

    pcl_shape = (f1.nmaps * f2.nmaps, f1.ainfo.lmax+1)
    if cl_guess.shape != pcl_shape:
        raise ValueError(
            f"Guess Cl should have shape {pcl_shape}")
    clg = cl_guess.reshape([f1.nmaps, f2.nmaps, f1.ainfo.lmax+1])

    if f1.lite or f2.lite:
        raise ValueError("Can't compute deprojection bias for "
                         "lightweight fields")

    clb = np.zeros((f1.nmaps, f2.nmaps, f1.ainfo.lmax+1))

    # Compute ff part
    if f1.n_temp > 0:
        pcl_ff = np.zeros((f1.n_temp, f1.n_temp,
                           f1.nmaps, f2.nmaps,
                           f1.ainfo.lmax+1))
        for ij, tj in enumerate(f1.temp):
            # SHT(v*fj)
            ftild_j = ut.map2alm(tj*f1.mask[None, :], f1.spin,
                                 f1.minfo, f1.ainfo, n_iter=n_iter)
            # C^ba*SHT[v*fj]
            ftild_j = np.array([
                np.sum([hp.almxfl(ftild_j[m], clg[m, n],
                                  mmax=f1.ainfo.mmax)
                        for m in range(f1.nmaps)], axis=0)
                for n in range(f2.nmaps)])
            # SHT^-1[C^ba*SHT[v*fj]]
            ftild_j = ut.alm2map(ftild_j, f2.spin, f2.minfo,
                                 f2.ainfo)
            # SHT[w*SHT^-1[C^ba*SHT[v*fj]]]
            ftild_j = purify_if_needed(f2, ftild_j)
            for ii, f_i in enumerate(f1.alm_temp):
                clij = np.array([[hp.alm2cl(a1, a2, lmax=f1.ainfo.lmax)
                                  for a2 in ftild_j]
                                 for a1 in f_i])
                pcl_ff[ii, ij, :, :, :] = clij
        clb -= np.einsum('ij,ijklm', f1.iM, pcl_ff)

    # Compute gg part and fg part
    if f2.n_temp > 0:
        pcl_gg = np.zeros((f2.n_temp, f2.n_temp,
                           f1.nmaps, f2.nmaps,
                           f1.ainfo.lmax+1))
        if f1.n_temp > 0:
            prod_fg = np.zeros((f1.n_temp, f2.n_temp))
            pcl_fg = np.zeros((f1.n_temp, f2.n_temp,
                               f1.nmaps, f2.nmaps,
                               f1.ainfo.lmax+1))

        for ij, tj in enumerate(f2.temp):
            # SHT(w*gj)
            gtild_j = ut.map2alm(tj*f2.mask[None, :], f2.spin,
                                 f2.minfo, f2.ainfo, n_iter=n_iter)
            # C^ab*SHT[w*gj]
            gtild_j = np.array([
                np.sum([hp.almxfl(gtild_j[n], clg[m, n],
                                  mmax=f2.ainfo.mmax)
                        for n in range(f2.nmaps)], axis=0)
                for m in range(f1.nmaps)])
            # SHT^-1[C^ab*SHT[w*gj]]
            gtild_j = ut.alm2map(gtild_j, f1.spin, f1.minfo,
                                 f1.ainfo)
            if f1.n_temp > 0:
                # Int[f^i*v*SHT^-1[C^ab*SHT[w*gj]]]
                for ii, ti in enumerate(f1.temp):
                    prod_fg[ii, ij] = f1.minfo.si.dot_map(
                        ti, gtild_j*f1.mask[None, :])

            # SHT[v*SHT^-1[C^ab*SHT[w*gj]]]
            gtild_j = purify_if_needed(f1, gtild_j)

            # PCL[g_i, gtild_j]
            for ii, g_i in enumerate(f2.alm_temp):
                clij = np.array([[hp.alm2cl(a1, a2, lmax=f1.ainfo.lmax)
                                  for a2 in g_i]
                                 for a1 in gtild_j])
                pcl_gg[ii, ij, :, :, :] = clij

        clb -= np.einsum('ij,ijklm', f2.iM, pcl_gg)
        if f1.n_temp > 0:
            # PCL[f_i, g_j]
            pcl_fg = np.array([[[[hp.alm2cl(a1, a2, lmax=f1.ainfo.lmax)
                                  for a2 in gj]
                                 for a1 in fi]
                                for gj in f2.alm_temp]
                               for fi in f1.alm_temp])
            clb += np.einsum('ij,rs,jr,isklm',
                             f1.iM, f2.iM, prod_fg, pcl_fg)
    return clb.reshape(pcl_shape)


def uncorr_noise_deprojection_bias(f1, map_var, n_iter=None):
    """ Computes the bias associated to contaminant removal in the
    presence of uncorrelated inhomogeneous noise to the
    auto-pseudo-:math:`C_\\ell` of a given field.

    Args:
        f1 (:class:`~pymaster.field.NmtField`): Field being correlated.
        map_var (`array`): Single map containing the local noise
            variance in one steradian. The map should have the same
            pixelization used by ``f1``.
        n_iter (:obj:`int`): Number of iterations when computing
            :math:`a_{\\ell m}` s. See docstring of
            :class:`~pymaster.field.NmtField`.

    Returns:
        (`array`): Deprojection bias pseudo-:math:`C_\\ell`.
    """
    if f1.lite:
        raise ValueError("Can't compute deprojection bias for "
                         "lightweight fields")
    if n_iter is None:
        n_iter = ut.nmt_params.n_iter_default

    # Flatten in case it's a 2D map
    sig2 = map_var.flatten()
    if len(sig2) != f1.minfo.npix:
        raise ValueError("Variance map doesn't match map resolution")

    pcl_shape = (f1.nmaps * f1.nmaps, f1.ainfo.lmax+1)

    # Return if no contamination
    if f1.n_temp == 0:
        return np.zeros(pcl_shape)

    clb = np.zeros((f1.nmaps, f1.nmaps, f1.ainfo.lmax+1))

    # First term in Eq. 39 of the NaMaster paper
    pcl_ff = np.zeros((f1.n_temp, f1.n_temp,
                       f1.nmaps, f1.nmaps,
                       f1.ainfo.lmax+1))
    for j, fj in enumerate(f1.temp):
        # SHT(v^2 sig^2 f_j)
        fj_v_s = ut.map2alm(fj*(f1.mask**2*sig2)[None, :], f1.spin,
                            f1.minfo, f1.ainfo, n_iter=n_iter)
        for i, fi in enumerate(f1.alm_temp):
            cl = np.array([[hp.alm2cl(a1, a2, lmax=f1.ainfo.lmax)
                            for a1 in fi]
                           for a2 in fj_v_s])
            pcl_ff[i, j, :, :, :] = cl
    clb -= 2*np.einsum('ij,ijklm', f1.iM, pcl_ff)

    # Second term in Eq. 39 of the namaster paper
    # PCL(fi, fs)
    pcl_ff = np.array([[[[hp.alm2cl(a1, a2, lmax=f1.ainfo.lmax)
                          for a2 in fs]
                         for a1 in fi]
                        for fs in f1.alm_temp]
                       for fi in f1.alm_temp])
    # Int[fj * fr * v^2 * sig^2]
    prod_ff = np.array([[
        f1.minfo.si.dot_map(fj, fr*(f1.mask**2*sig2)[None, :])
        for fr in f1.temp] for fj in f1.temp])
    clb += np.einsum('ij,rs,jr,isklm', f1.iM, f1.iM, prod_ff, pcl_ff)

    return clb.reshape(pcl_shape)


def deprojection_bias_flat(f1, f2, b, ells, cl_guess,
                           ell_cut_x=[1., -1.], ell_cut_y=[1., -1.]):
    """ Computes the bias associated to contaminant removal to the
    cross-pseudo-:math:`C_\\ell` of two flat-sky fields. See Eq. 50 in
    the NaMaster paper.

    Args:
        f1 (:class:`~pymaster.field.NmtFieldFlat`): First field to
            correlate.
        f2 (:class:`~pymaster.field.NmtFieldFlat`): Second field to
            correlate.
        b (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme defining
            the output bandpowers.
        ells (`array`): List of multipoles on which the guess power
            spectra are defined.
        cl_guess (`array`): Array of power spectra corresponding to a
            best-guess of the true power spectra of ``f1`` and ``f2``.
        ell_cut_x (`array`): Sequence of two elements determining the
            range of :math:`l_x` to remove from the calculation. No
            Fourier modes removed by default.
        ell_cut_y (`array`): Sequence of two elements determining the
            range of :math:`l_y` to remove from the calculation. No
            Fourier modes removed by default.

    Returns:
        (`array`): Deprojection bias pseudo-:math:`C_\\ell`.
    """
    if len(cl_guess) != f1.fl.nmaps * f2.fl.nmaps:
        raise ValueError("Proposal Cell doesn't match number of maps")
    if len(cl_guess[0]) != len(ells):
        raise ValueError("cl_guess and ells must have the same length")
    cl1d = lib.comp_deproj_bias_flat(
        f1.fl,
        f2.fl,
        b.bin,
        ell_cut_x[0],
        ell_cut_x[1],
        ell_cut_y[0],
        ell_cut_y[1],
        ells,
        cl_guess,
        f1.fl.nmaps * f2.fl.nmaps * b.bin.n_bands,
    )
    cl2d = np.reshape(cl1d, [f1.fl.nmaps * f2.fl.nmaps, b.bin.n_bands])

    return cl2d


def compute_coupled_cell(f1, f2):
    """ Computes the full-sky pseudo-:math:`C_\\ell` of two masked
    fields (``f1`` and ``f2``) without aiming to deconvolve the
    mode-coupling matrix (Eq. 7 of the NaMaster paper). Effectively,
    this is equivalent to calling the usual HEALPix `anafast
    <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.anafast.html>`_
    routine on the masked and contaminant-cleaned maps.

    Args:
        f1 (:class:`~pymaster.field.NmtField`): First field to
            correlate.
        f2 (:class:`~pymaster.field.NmtField`): Second field to
            correlate.

    Returns:
        (`array`): Array of coupled pseudo-:math:`C_\\ell` s.
    """  # noqa
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("You're trying to correlate incompatible fields")
    alm1 = f1.get_alms()
    alm2 = f2.get_alms()
    ncl = len(alm1) * len(alm2)
    lmax = min(f1.ainfo.lmax, f2.ainfo.lmax)

    Nf = 0
    if f2 is f1:
        Nf = f1.Nf

    cls = np.array([[hp.alm2cl(a1, a2, lmax=lmax)
                     for a2 in alm2] for a1 in alm1])
    if Nf != 0:
        for i in range(len(alm1)):
            cls[i, i, :] -= Nf
    cls = cls.reshape([ncl, lmax+1])
    return cls


def compute_coupled_cell_flat(f1, f2, b, ell_cut_x=[1., -1.],
                              ell_cut_y=[1., -1.]):
    """ Computes the flat-sky pseudo-:math:`C_\\ell` of two masked
    fields (``f1`` and ``f2``) without aiming to deconvolve the
    mode-coupling matrix (Eq. 42 of the NaMaster paper). Effectively,
    this is equivalent to computing the map FFTs and
    averaging over rings of wavenumber.  The returned power
    spectrum is defined at the multipoles returned by the
    method :meth:`~pytest.field.NmtFieldFlat.get_ell_sampling`
    of either ``f1`` or ``f2``.

    Args:
        f1 (:class:`~pymaster.field.NmtFieldFlat`): First field to
            correlate.
        f2 (:class:`~pymaster.field.NmtFieldFlat`): Second field to
            correlate.
        b (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme defining
            the output bandpowers.
        ell_cut_x (`array`): Sequence of two elements determining the
            range of :math:`l_x` to remove from the calculation. No
            Fourier modes removed by default.
        ell_cut_y (`array`): Sequence of two elements determining the
            range of :math:`l_y` to remove from the calculation. No
            Fourier modes removed by default.

    Returns:
        (`array`): Array of coupled pseudo-:math:`C_\\ell` s.
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


def compute_full_master(f1, f2, b=None, cl_noise=None, cl_guess=None,
                        workspace=None, l_toeplitz=-1, l_exact=-1, dl_band=-1,
                        normalization='MASTER'):
    """ Computes the full MASTER estimate of the power spectrum of two
    fields (``f1`` and ``f2``). This is equivalent to sequentially calling:

    - :meth:`NmtWorkspace.compute_coupling_matrix`
    - :meth:`deprojection_bias`
    - :meth:`compute_coupled_cell`
    - :meth:`NmtWorkspace.decouple_cell`


    Args:
        fl1 (:class:`~pymaster.field.NmtField`): First field to
            correlate.
        fl2 (:class:`~pymaster.field.NmtField`): Second field to
            correlate.
        b (:class:`~pymaster.bins.NmtBin`): Binning scheme.
        cl_noise (`array`): Noise bias (i.e. angular
            pseudo-:math:`C_\\ell` of masked noise realizations).
        cl_guess (`array`): Array of power spectra corresponding to a
            best-guess of the true power spectra of ``f1`` and ``f2``.
        workspace (:class:`~pymaster.workspaces.NmtWorkspace`):
            Object containing the mode-coupling matrix associated with
            an incomplete sky coverage. If provided, the function will
            skip the computation of the mode-coupling matrix and use
            the information encoded in this object.
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
        normalization (:obj:`str`): Normalization convention to use for
            the bandpower window functions. Two options supported:
            `'MASTER'` (default) corresponds to the standard inversion
            of the binned mode-coupling matrix. `'FKP'` simply divides
            by the mean of the mask product, forcing a unit response
            to an input white spectrum.

    Returns:
        (`array`): Set of decoupled bandpowers.
    """
    if (b is None) and (workspace is None):
        raise SyntaxError("Must supply either workspace or bins.")
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("Fields have incompatible pixelizations.")
    pcl_shape = (f1.nmaps * f2.nmaps, f1.ainfo.lmax+1)

    if cl_noise is not None:
        if cl_noise.shape != pcl_shape:
            raise ValueError(
                f"Noise Cl should have shape {pcl_shape}")
        pcln = cl_noise
    else:
        pcln = np.zeros(pcl_shape)
    if cl_guess is not None:
        if cl_guess.shape != pcl_shape:
            raise ValueError(
                f"Guess Cl should have shape {pcl_shape}")
        clg = cl_guess
    else:
        clg = np.zeros(pcl_shape)

    # Data power spectrum
    pcld = compute_coupled_cell(f1, f2)
    # Deprojection bias
    pclb = deprojection_bias(f1, f2, clg)

    if workspace is None:
        w = NmtWorkspace.from_fields(
            fl1=f1, fl2=f2, bins=b,
            l_toeplitz=l_toeplitz,
            l_exact=l_exact, dl_band=dl_band,
            normalization=normalization)
    else:
        w = workspace

    return w.decouple_cell(pcld - pclb - pcln)


def compute_full_master_flat(f1, f2, b, cl_noise=None, cl_guess=None,
                             ells_guess=None, workspace=None,
                             ell_cut_x=[1., -1.], ell_cut_y=[1., -1.]):
    """
    Computes the full MASTER estimate of the power spectrum of two
    flat-sky fields (``f1`` and ``f2``). This is equivalent to
    sequentially calling:

    - :meth:`NmtWorkspaceFlat.compute_coupling_matrix`
    - :meth:`deprojection_bias_flat`
    - :meth:`compute_coupled_cell_flat`
    - :meth:`NmtWorkspaceFlat.decouple_cell`

    Args:
        f1 (:class:`~pymaster.field.NmtFieldFlat`): First field to
            correlate.
        f2 (:class:`~pymaster.field.NmtFieldFlat`): Second field to
            correlate.
        b (:class:`~pymaster.bins.NmtBinFlat`): Binning scheme defining
            the output bandpowers.
        cl_noise (`array`): Noise bias (i.e. angular
            pseudo-:math:`C_\\ell` of masked noise realisations).
        cl_guess (`array`): Array of power spectra corresponding to a
            best-guess of the true power spectra of ``f1`` and ``f2``.
        ells_guess (`array`): List of multipoles on which the guess power
            spectra are defined.
        workspace (:class:`~pymaster.workspaces.NmtWorkspaceFlat`):
            Object containing the mode-coupling matrix associated with
            an incomplete sky coverage. If provided, the function will
            skip the computation of the mode-coupling matrix and use
            the information encoded in this object.
        ell_cut_x (`array`): Sequence of two elements determining the
            range of :math:`l_x` to remove from the calculation. No
            Fourier modes removed by default.
        ell_cut_y (`array`): Sequence of two elements determining the
            range of :math:`l_y` to remove from the calculation. No
            Fourier modes removed by default.

    Returns:
        (`array`): Set of decoupled bandpowers.
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
