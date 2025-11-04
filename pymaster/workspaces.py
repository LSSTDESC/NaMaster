from pymaster import nmtlib as lib
import pymaster.utils as ut
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
                 read_unbinned_MCM=True, normalization='MASTER'):
        self.wsp = None
        self.has_unbinned = False

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
            self.read_from(fname, read_unbinned_MCM=read_unbinned_MCM)
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
    def from_file(cls, fname, read_unbinned_MCM=True):
        """ Creates an :obj:`NmtWorkspace` object from a mode-coupling
        matrix stored in a FITS file. See :meth:`write_to`.

        Args:
            fname (:obj:`str`): Input file name.
            read_unbinned_MCM (:obj:`bool`): If ``False``, the unbinned
                mode-coupling matrix will not be read. This can save
                significant IO time.
        """
        return cls(fname=fname, read_unbinned_MCM=read_unbinned_MCM)

    def __del__(self):
        if self.wsp is not None:
            if lib.workspace_free is not None:
                lib.workspace_free(self.wsp)
            self.wsp = None

    def check_unbinned(self):
        """ Raises an error if this workspace does not contain the
        unbinned mode-coupling matrix.
        """
        if not self.has_unbinned:
            raise ValueError("This workspace does not store the unbinned "
                             "mode-coupling matrix.")

    def read_from(self, fname, read_unbinned_MCM=True):
        """ Reads the contents of an :obj:`NmtWorkspace` object from a
        FITS file.

        Args:
            fname (:obj:`str`): Input file name.
            read_unbinned_MCM (:obj:`bool`): If ``False``, the unbinned
                mode-coupling matrix will not be read. This can save
                significant IO time.
        """
        if self.wsp is not None:
            lib.workspace_free(self.wsp)
            self.wsp = None
        self.wsp = lib.read_workspace(fname, int(read_unbinned_MCM))
        self.has_unbinned = read_unbinned_MCM

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

        lmax = self.wsp.lmax_fields
        if (len(beam1) <= lmax) or (len(beam2) <= lmax):
            raise ValueError("The new beams must go up to ell = %d" % lmax)
        lib.wsp_update_beams(self.wsp, beam1, beam2)

    def update_bins(self, bins):
        """ Update binning associated with this mode-coupling matrix.
        This is significantly faster than recomputing the matrix from
        scratch.

        Args:
            bins (:class:`~pymaster.bins.NmtBin`): New binning scheme.
        """
        if self.wsp is None:
            raise ValueError("Can't update bins without first computing "
                             "the mode-coupling matrix")
        if bins.bin is None:
            raise ValueError("Can't replace with uninitialized bins")
        lib.wsp_update_bins(self.wsp, bins.bin)

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
        if self.wsp is not None:
            lib.workspace_free(self.wsp)
            self.wsp = None

        anisotropic_mask_any = fl1.anisotropic_mask or fl2.anisotropic_mask
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
            if fl1.anisotropic_mask:
                alm1a = fl1.get_anisotropic_mask_alms()
            if fl2.anisotropic_mask:
                alm2a = fl2.get_anisotropic_mask_alms()
            if fl2.anisotropic_mask:
                pclm_0e = hp.alm2cl(alm1, alm2a[0], lmax=fl1.ainfo_mask.lmax)
                pclm_0b = hp.alm2cl(alm1, alm2a[1], lmax=fl1.ainfo_mask.lmax)
            if fl1.anisotropic_mask:
                pclm_e0 = hp.alm2cl(alm1a[0], alm2, lmax=fl1.ainfo_mask.lmax)
                pclm_b0 = hp.alm2cl(alm1a[1], alm2, lmax=fl1.ainfo_mask.lmax)
                if fl2.anisotropic_mask:
                    pclm_ee = hp.alm2cl(alm1a[0], alm2a[0],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_eb = hp.alm2cl(alm1a[0], alm2a[1],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_be = hp.alm2cl(alm1a[1], alm2a[0],
                                        lmax=fl1.ainfo_mask.lmax)
                    pclm_bb = hp.alm2cl(alm1a[1], alm2a[1],
                                        lmax=fl1.ainfo_mask.lmax)

        if normalization == 'MASTER':
            norm_type = 0
        elif normalization == 'FKP':
            norm_type = 1
        else:
            raise ValueError(f"Unknown normalization type {normalization}. "
                             "Allowed options are 'MASTER' and 'FKP'.")

        wawb = 0
        if norm_type == 1:
            if fl1.is_catalog or fl2.is_catalog:
                if fl2 is fl1:
                    wawb = fl1.Nw
                else:
                    raise ValueError("Cannot use FKP normalisation for "
                                     "catalog fields unless they are the "
                                     "same field.")
            else:
                msk1 = fl1.get_mask()
                msk2 = fl2.get_mask()
                wawb = fl1.minfo.si.dot_map(msk1, msk2)/(4*np.pi)

        if anisotropic_mask_any:
            self.wsp = lib.comp_coupling_matrix_anisotropic(
                int(fl1.spin), int(fl2.spin),
                int(fl1.anisotropic_mask), int(fl2.anisotropic_mask),
                int(fl1.ainfo.lmax), int(fl1.ainfo_mask.lmax),
                pclm_00, pclm_0e, pclm_0b, pclm_e0, pclm_b0,
                pclm_ee, pclm_eb, pclm_be, pclm_bb,
                fl1.beam, fl2.beam, bins.bin,
                int(norm_type), wawb)
        else:
            self.wsp = lib.comp_coupling_matrix(
                int(fl1.spin), int(fl2.spin),
                int(fl1.ainfo.lmax), int(fl1.ainfo_mask.lmax),
                int(fl1.pure_e), int(fl1.pure_b),
                int(fl2.pure_e), int(fl2.pure_b),
                int(norm_type), wawb,
                fl1.beam, fl2.beam, pcl_mask.flatten()-Nw,
                bins.bin, int(is_teb), l_toeplitz, l_exact, dl_band)
        self.has_unbinned = True

    def write_to(self, fname):
        """ Writes the contents of an :obj:`NmtWorkspace` object
        to a FITS file.

        Args:
            fname (:obj:`str`): Output file name
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before writing")
        self.check_unbinned()
        lib.write_workspace(self.wsp, "!"+fname)

    def get_coupling_matrix(self):
        """ Returns the currently stored mode-coupling matrix.

        Returns:
            (`array`): Mode-coupling matrix. The matrix will have shape
            ``(nrows,nrows)``, with ``nrows = n_cls * n_ells``, where
            ``n_cls`` is the number of power spectra (1, 2 or 4 for
            spin 0-0, spin 0-2 and spin 2-2 correlations), and
            ``n_ells = lmax + 1``, and ``lmax`` is the maximum multipole
            associated with this workspace. The assumed ordering of power
            spectra is such that the ``L``-th element of the ``i``-th power
            spectrum be stored with index ``L * n_cls + i``.
        """
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before "
                               "getting a MCM")
        self.check_unbinned()
        nrows = (self.wsp.lmax + 1) * self.wsp.ncls
        return lib.get_mcm(self.wsp, nrows * nrows).reshape([nrows, nrows])

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
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before updating MCM")
        self.check_unbinned()
        if len(new_matrix) != (self.wsp.lmax + 1) * self.wsp.ncls:
            raise ValueError("Input matrix has an inconsistent size. "
                             f"Expected {(self.wsp.lmax+1)*self.wsp.ncls}, "
                             f"but got {len(new_matrix)}.")
        lib.update_mcm(self.wsp, len(new_matrix), new_matrix.flatten())

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
        if (len(cl_in) != self.wsp.ncls) or \
           (len(cl_in[0]) < self.wsp.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape. "
                             f"Expected ({self.wsp.ncls}, {self.wsp.lmax+1}), "
                             f"bu got {cl_in.shape}.")
        self.check_unbinned()

        # Shorten C_ells if they're too long
        cl_in = np.array(cl_in)[:, :self.wsp.lmax+1]
        cl1d = lib.couple_cell_py(self.wsp, cl_in,
                                  self.wsp.ncls * (self.wsp.lmax + 1))
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.lmax + 1])
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
        if (len(cl_in) != self.wsp.ncls) or \
           (len(cl_in[0]) < self.wsp.lmax + 1):
            raise ValueError("Input power spectrum has wrong shape. "
                             f"Expected ({self.wsp.ncls}, {self.wsp.lmax+1}), "
                             f"but got {cl_in.shape}")
        if cl_bias is not None:
            if (len(cl_bias) != self.wsp.ncls) or \
               (len(cl_bias[0]) < self.wsp.lmax + 1):
                raise ValueError(
                    "Input bias power spectrum has wrong shape. "
                    f"Expected ({self.wsp.ncls}, {self.wsp.lmax+1}), "
                    f"but got {cl_bias.shape}")
            clb = cl_bias.copy()
        else:
            clb = np.zeros_like(cl_in)
        if cl_noise is not None:
            if (len(cl_noise) != self.wsp.ncls) or \
               (len(cl_noise[0]) < self.wsp.lmax + 1):
                raise ValueError(
                    "Input noise power spectrum has wrong shape. "
                    f"Expected ({self.wsp.ncls}, {self.wsp.lmax+1}), "
                    f"but got {cl_noise.shape}")
            cln = cl_noise.copy()
        else:
            cln = np.zeros_like(cl_in)

        cl1d = lib.decouple_cell_py(
            self.wsp, cl_in, cln, clb, self.wsp.ncls * self.wsp.bin.n_bands
        )
        clout = np.reshape(cl1d, [self.wsp.ncls, self.wsp.bin.n_bands])

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
        self.check_unbinned()
        d = lib.get_bandpower_windows(self.wsp,
                                      self.wsp.ncls * self.wsp.bin.n_bands *
                                      self.wsp.ncls * (self.wsp.lmax+1))
        return np.transpose(d.reshape([self.wsp.bin.n_bands,
                                       self.wsp.ncls,
                                       self.wsp.lmax+1,
                                       self.wsp.ncls]),
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
        self.wsp = lib.read_workspace_flat(fname)

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
        lib.write_workspace_flat(self.wsp, "!"+fname)

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


def get_general_coupling_matrix(pcl_mask, s1, s2, n1, n2):
    """ Returns a general mode-coupling matrix of the form

    .. math::
      M_{\\ell \\ell'}=\\sum_{\\ell''}
      \\frac{(2\\ell'+1)(2\\ell''+1)}{4\\pi}
      \\tilde{C}^{uv}_\\ell
      \\left(\\begin{array}{ccc}
      \\ell & \\ell' & \\ell'' \\\\
      n_1 & -s_1 & s_1-n_1
      \\end{array}\\right)
      \\left(\\begin{array}{ccc}
      \\ell & \\ell' & \\ell'' \\\\
      n_2 & -s_2 & s_2-n_2
      \\end{array}\\right)

    Args:
        pcl_mask (`array`): 1D array containing the power spectrum
          of the masks :math:`\\tilde{C}_\\ell^{uw}`.
        s1 (:obj:`int`): spin index :math:`s_1` above.
        s2 (:obj:`int`): spin index :math:`s_2` above.
        n1 (:obj:`int`): spin index :math:`n_1` above.
        n2 (:obj:`int`): spin index :math:`n_2` above.

    Returns:
        (`array`): 2D array of shape ``[nl, nl]``, where ``nl`` is
        the size of ``pcl_mask``, containing the mode-coupling
        matrix for multipoles from 0 to ``nl-1``.
    """

    lmax = len(pcl_mask)-1
    xi = lib.comp_general_coupling_matrix(
        int(s1), int(s2), int(n1),
        int(n2), int(lmax),
        pcl_mask, int((lmax+1)**2))
    xi = xi.reshape([lmax+1, lmax+1])
    return xi
