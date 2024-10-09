from pymaster import nmtlib as lib
import numpy as np
import healpy as hp
import pymaster.utils as ut


class NmtField(object):
    """ An :obj:`NmtField` object contains all the information
    describing the fields to correlate, including their observed
    maps, masks and contaminant templates.

    Args:
        mask (`array`): Array containing a map corresponding to the
            field's mask. Should be 1-dimensional for a HEALPix map or
            2-dimensional for a map with rectangular (CAR) pixelization.
        maps (`array`): Array containing the observed maps for this
            field. Should be at least 2-dimensional. The first
            dimension corresponds to the number of maps, which should
            be 1 for a spin-0 field and 2 otherwise. The other
            dimensions should be either ``[npix]`` for HEALPix maps or
            ``[ny,nx]`` for maps with rectangular pixels (with the
            `y` and `x` dimensions corresponding to latitude and longitude
            respectively). For a spin>0 field, the two maps passed should
            be the usual Q/U Stokes parameters for polarization, or
            :math:`e_1`/:math:`e_2` (:math:`\\gamma_1`/:math:`\\gamma_2` etc.)
            in the case of cosmic shear. It is important to note that
            NaMaster uses the same polarization convention as HEALPix (i.e.
            with the growing colatitude :math:`\\theta`). It is however more
            common for galaxy ellipticities to be provided using the IAU
            convention (e.g. growing declination). In this case, the sign of
            the :math:`e_2`/:math:`\\gamma_2` map should be swapped before
            using it to create an :obj:`NmtField`. See more
            `here <https://healpix.jpl.nasa.gov/html/intronode12.htm>`_.
            If ``None``, this field will only contain a mask but no maps.
            The field can then be used to compute a mode-coupling matrix,
            for instance, but not actual power spectra.
        spin (:obj:`int`): Spin of this field. If ``None`` it will
            default to 0 or 2 if ``maps`` contains 1 or 2 maps, respectively.
        templates (`array`): Array containing a set of contaminant templates
            for this field. This array should have shape ``[ntemp,nmap,...]``,
            where ``ntemp`` is the number of templates, ``nmap`` should be 1
            for spin-0 fields and 2 otherwise. The other dimensions should be
            ``[npix]`` for HEALPix maps or ``[ny,nx]`` for maps with
            rectangular pixels. The best-fit contribution from each
            contaminant is automatically removed from the maps unless
            ``templates=None``.
        beam (`array`): Spherical harmonic transform of the instrumental beam
            (assumed to be rotationally symmetric - i.e. no :math:`m`
            dependence). If ``None``, no beam will be corrected for. Otherwise,
            this array should have at least as many elements as the maximum
            multipole sampled by the maps + 1 (see ``lmax``). Note that
            NaMaster does not automatically correct for pixel window function
            effects, as this is not always appropriate. If you want to correct
            for these effects, you should include them pixel window function
            in the beam or account for it when interpreting the measured
            power spectrum.
        purify_e (:obj:`bool`): Purify E-modes?
        purify_b (:obj:`bool`): Purify B-modes?
        n_iter (:obj:`int`): Number of iterations when computing the
            :math:`a_{\\ell m}` s of the input maps. See the documentation of
            :meth:`~pymaster.utils.map2alm`. If ``None``, it will default to
            the internal value (see documentation of
            :class:`~pymaster.utils.NmtParams`), which can be accessed via
            :meth:`~pymaster.utils.get_default_params`, and modified via
            :meth:`~pymaster.utils.set_n_iter_default`.
        n_iter_mask (:obj:`int`): Number of iterations when computing the
            spherical harmonic transform of the mask. If ``None``, it will
            default to the internal value (see documentation of
            :class:`~pymaster.utils.NmtParams`), which can be accessed via
            :meth:`~pymaster.utils.get_default_params`,
            and modified via :meth:`~pymaster.utils.set_n_iter_default`.
        lmax (:obj:`int`): Maximum multipole up to which map power spectra
            will be computed. If negative or zero, the maximum multipole given
            the map resolution will be used (e.g. :math:`3N_{\\rm side}-1`
            for HEALPix maps).
        lmax_mask (:obj:`int`): Maximum multipole up to which the power
            spectrum of the mask will be computed. If negative or zero, the
            maximum multipole given the map resolution will be used (e.g.
            :math:`3N_{\\rm side}-1` for HEALPix maps).
        tol_pinv (:obj:`float`): When computing the pseudo-inverse of the
            contaminant covariance matrix. See documentation of
            :meth:`~pymaster.utils.moore_penrose_pinvh`. Only relevant if
            passing contaminant templates that are likely to be highly
            correlated. If ``None``, it will default to the internal value,
            which can be accessed via
            :meth:`~pymaster.utils.get_default_params`, and modified via
            :meth:`~pymaster.utils.set_tol_pinv_default`.
        wcs (`WCS`): A WCS object if using rectangular (CAR) pixels (see
            `the astropy documentation
            <http://docs.astropy.org/en/stable/wcs/index.html>`_).
        masked_on_input (:obj:`bool`): Set to ``True`` if the input maps and
            templates are already multiplied by the mask. Note that this is
            not advisable if you're using purification, as correcting for this
            usually incurs inaccuracies around the mask edges that may lead
            to significantly biased power spectra.
        lite (:obj:`bool`): Set to ``True`` if you want to only store the bare
            minimum necessary to run a standard pseudo-:math:`C_\\ell` with
            deprojection and purification, but you don't care about
            deprojection bias. This will significantly reduce the memory taken
            up by the resulting object.
        mask_22 (`array`): Array containing the mask corresponding to the
            ``22`` component of an anisotropic weight matrix. If not ``None``
            the ``mask`` argument will be interpreted as the ``11`` component
            of the same matrix. The off-diagonal component of the weight
            matrix must then be passed as ``mask_12`` (below). If ``mask_22``
            and ``mask_12`` are ``None``, isotropic weighting is assumed,
            described by ``mask``.
        mask_12 (`array`): Array containing the off-diagonal component of the
            anisotropic weight matrix. See ``mask_22`` above.
    """
    def __init__(self, mask, maps, *, spin=None, templates=None, beam=None,
                 purify_e=False, purify_b=False, n_iter=None, n_iter_mask=None,
                 tol_pinv=None, wcs=None, lmax=None, lmax_mask=None,
                 masked_on_input=False, lite=False,
                 mask_22=None, mask_12=None):
        if n_iter_mask is None:
            n_iter_mask = ut.nmt_params.n_iter_mask_default
        if n_iter is None:
            n_iter = ut.nmt_params.n_iter_default
        if tol_pinv is None:
            tol_pinv = ut.nmt_params.tol_pinv_default

        # 0. Preliminary initializations
        # These first attributes are compulsory for all fields
        self.lite = lite
        self.mask = None
        self.beam = None
        self.n_iter = n_iter
        self.n_iter_mask = n_iter_mask
        self.pure_e = purify_e
        self.pure_b = purify_b
        # The alm is only required for non-mask-only maps
        self.alm = None
        # The remaining attributes are only required for non-lite maps
        self.maps = None
        self.temp = None
        self.alm_temp = None
        # The mask alms are only stored if computed for non-lite maps
        self.alm_mask = None
        self.n_temp = 0
        # Nw, Nf, and alpha are protected and made available as @property below
        self._Nw = 0
        self._Nf = 0
        self._alpha = None
        self.is_catalog = False
        # Anisotropic mask
        self.anisotropic_mask = False
        self.mask_a = None
        self.alm_mask_a = None

        # 1. Store mask and beam

        # 1.1 Check for anisotropic masks
        if (mask_22 is not None) or (mask_12 is not None):
            # Check that all anisotropic components are provided
            if (mask_22 is None) or (mask_12 is None):
                raise ValueError("Both mask_22 and mask_12 must be passed if "
                                 "either of them is passed.")
            # Check that the noise covariance that the masks derive from
            # is positive-definite
            mask_scale = np.mean(mask+mask_22)
            if np.any(mask*mask_22-mask_12**2 < -1E-5*mask_scale):
                raise ValueError("The anisotropic mask does not seem to be "
                                 "positive-definite.")
            mask0 = (0.5*(mask + mask_22)).astype(np.float64)
            mask1 = (0.5*(mask - mask_22)).astype(np.float64)
            mask2 = mask_12.astype(np.float64)
            mask = mask0
            self.anisotropic_mask = True
        else:
            # This ensures the mask will have the right type
            # and endianness (can cause issues when read from
            # some FITS files).
            mask = mask.astype(np.float64)

        # 1.2 get all spatial info from the masks
        self.minfo = ut.NmtMapInfo(wcs, mask.shape)
        self.mask = self.minfo.reform_map(mask)
        if self.anisotropic_mask:
            self.mask_a = self.minfo.reform_map(np.array([mask1, mask2]))
        if lmax is None:
            lmax = self.minfo.get_lmax()
        if lmax_mask is None:
            lmax_mask = self.minfo.get_lmax()
        self.ainfo = ut.NmtAlmInfo(lmax)
        self.ainfo_mask = ut.NmtAlmInfo(lmax_mask)

        # Beam
        if isinstance(beam, (list, tuple, np.ndarray)):
            if len(beam) <= lmax:
                raise ValueError("Input beam must have at least %d elements "
                                 "given the input map resolution" % (lmax+1))
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.ones(lmax+1)
            else:
                raise ValueError("Input beam can only be an array or None\n")
        self.beam = beam_use

        # If mask-only, just return
        if maps is None:
            if spin is None:
                raise ValueError("Please supply field spin")
            self.spin = spin
            if (self.spin == 0) and self.anisotropic_mask:
                raise ValueError("Anisotropic masks can't be used "
                                 "with scalar fields.")
            self.nmaps = 2 if spin else 1
            return

        # 2. Check maps
        # As for the mask, ensure dtype is float to avoid
        # issues when reading the map from a fits file
        maps = np.array(maps, dtype=np.float64)
        if (len(maps) != 1) and (len(maps) != 2):
            raise ValueError("Must supply 1 or 2 maps per field")

        if spin is None:
            if len(maps) == 1:
                spin = 0
            else:
                spin = 2
        else:
            if (((spin != 0) and len(maps) == 1) or
                    ((spin == 0) and len(maps) != 1)):
                raise ValueError("Spin-zero fields are "
                                 "associated with a single map")
        self.spin = spin
        self.nmaps = 2 if spin else 1

        maps = self.minfo.reform_map(maps)

        pure_any = self.pure_e or self.pure_b
        if pure_any and self.spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")

        # 2.1 Check that no bells nor whistles are requested for anisotropic
        # masks, since these are not supported.
        if self.anisotropic_mask:
            if self.spin == 0:
                raise ValueError("Anisotropic masks can't be used with "
                                 "scalar fields.")
            if pure_any:
                raise NotImplementedError("Purification not implemented for "
                                          "anisotropic masks")
            if templates is not None:
                raise NotImplementedError("Contaminant deprojection not "
                                          "supported for anisotropic masks.")

        # 3. Check templates
        if isinstance(templates, (list, tuple, np.ndarray)):
            templates = np.array(templates, dtype=np.float64)
            if (len(templates[0]) != len(maps)):
                raise ValueError("Each template must have the same number of "
                                 "components as the map.")
            templates = self.minfo.reform_map(templates)
            self.n_temp = len(templates)
        else:
            if templates is not None:
                raise ValueError("Input templates can only be an array "
                                 "or None")
        w_temp = templates is not None

        # 4. Temporarily store unmasked maps if purifying
        maps_unmasked = None
        temp_unmasked = None
        if pure_any:
            maps_unmasked = np.array(maps)
            if w_temp:
                temp_unmasked = np.array(templates)
            if masked_on_input:
                good = self.mask > 0
                goodm = self.mask[good]
                maps_unmasked[:, good] /= goodm[None, :]
                if w_temp:
                    temp_unmasked[:, :, good] /= goodm[None, None, :]

        # 5. Mask all maps
        if w_temp:
            templates = np.array(templates)
        if not masked_on_input:
            if self.anisotropic_mask:
                maps = (maps*self.mask[None, :] +
                        np.array([maps[0]*self.mask_a[0] +
                                  maps[1]*self.mask_a[1],
                                  maps[0]*self.mask_a[1] -
                                  maps[1]*self.mask_a[0]]))
            else:
                maps *= self.mask[None, :]
            if w_temp:
                templates *= self.mask[None, None, :]

        # 6. Compute template normalization matrix and deproject
        if w_temp:
            M = np.array([[self.minfo.si.dot_map(t1, t2)
                           for t1 in templates]
                          for t2 in templates])
            iM = ut.moore_penrose_pinvh(M, tol_pinv)
            prods = np.array([self.minfo.si.dot_map(t, maps)
                              for t in templates])
            alphas = np.dot(iM, prods)
            self.iM = iM
            self.alphas = alphas
            maps = maps - np.sum(alphas[:, None, None]*templates,
                                 axis=0)
            # Do it also for unmasked maps if needed
            if pure_any:
                maps_unmasked = maps_unmasked - \
                    np.sum(alphas[:, None, None]*temp_unmasked, axis=0)

        # 7. Compute alms
        # - If purifying, do so at the same time.
        alm_temp = None
        alm_mask = None
        if pure_any:
            task = [self.pure_e, self.pure_b]
            alm_mask = self.get_mask_alms()
            self.alm, maps = self._purify(self.mask, alm_mask, maps_unmasked,
                                          n_iter=n_iter, task=task)
            if w_temp and (not self.lite):
                alm_temp = np.array([self._purify(self.mask, alm_mask, t,
                                                  n_iter=n_iter,
                                                  task=task)[0]
                                     for t in temp_unmasked])
                # IMPORTANT: at this stage, maps and self.alm contain the
                # purified map and SH coefficients. However, although alm_temp
                # contains the purified SH coefficients, templates contains the
                # ***non-purified*** maps. This is to speed up the calculation
                # of the deprojection bias.
        else:
            self.alm = ut.map2alm(maps, self.spin, self.minfo,
                                  self.ainfo, n_iter=n_iter)
            if w_temp and (not self.lite):
                alm_temp = np.array([ut.map2alm(t, self.spin, self.minfo,
                                                self.ainfo, n_iter=n_iter)
                                     for t in templates])

        # 8. Store any additional information needed
        if not self.lite:
            self.maps = maps.copy()
            if w_temp:
                self.temp = templates.copy()
                self.alm_temp = alm_temp

    def is_compatible(self, other, strict=True):
        """ Returns ``True`` if the this :obj:`NmtField`
        is compatible with that of another one (``other``).

        Args:
            other: field to compare with
            strict: if ``True``, the fields are compared at the pixel-
                and harmonic-space levels. Otherwise, only the
                harmonic-space resolution is compared. This is relevant
                if only power spectra between fields are necessary, but
                no pixel-level operations (e.g. map multiplications).
        """
        if strict:
            if self.minfo != other.minfo:
                return False
        if self.ainfo_mask != other.ainfo_mask:
            return False
        if self.ainfo != other.ainfo:
            return False
        return True

    def get_mask(self):
        """ Get this field's mask.

        Returns:
            (`array`): 1D array containing the field's mask.
        """
        if self.mask is None:
            raise ValueError("Input mask unavailable for this field")
        return self.mask

    def get_anisotropic_mask(self):
        """ Get this field's anisotropic mask components.

        Returns:
            (`array`): 2D array containing the field's anisotropic mask.
        """
        if self.mask_a is None:
            raise ValueError("Input anisotropic mask unavailable "
                             "for this field")
        return self.mask_a

    def get_mask_alms(self):
        """ Get the :math:`a_{\\ell m}` coefficients of this field's mask.
        Note that, in most cases, the mask :math:`a_{\\ell n}` s are not
        computed when generating the field. When calling this function for
        the first time, if they have not been calculated, they will be
        (which may be a slow operation), and stored internally for any future
        calls.

        Returns:
            (`array`): 1D array containing the mask :math:`a_{\\ell m}` s.
        """
        if self.alm_mask is None:
            amask = ut.map2alm(np.array([self.mask]), 0,
                               self.minfo, self.ainfo_mask,
                               n_iter=self.n_iter_mask)[0]
            if not self.lite:  # Store while we're at it
                self.alm_mask = amask
        else:
            amask = self.alm_mask
        return amask

    def get_anisotropic_mask_alms(self):
        """ Get the :math:`a_{\\ell m}` coefficients of this field's
        anisotropic mask components.

        Returns:
            (`array`): 2D array containing the mask :math:`a_{\\ell m}` s.
        """
        if self.mask_a is None:
            raise ValueError("This field does not have an anisotropic mask.")

        if self.alm_mask_a is None:
            amask = ut.map2alm(self.mask_a, 2*self.spin,
                               self.minfo, self.ainfo_mask,
                               n_iter=self.n_iter_mask)
            if not self.lite:  # Store while we're at it
                self.alm_mask_a = amask
        else:
            amask = self.alm_mask_a
        return amask

    def get_maps(self):
        """ Get this field's set of maps. The returned maps will be
        masked, contaminant-deprojected, and purified (if so required
        when generating this :obj:`NmtField`).

        Returns:
            (`array`): 2D array containing the field's maps.
        """
        if self.maps is None:
            raise ValueError("Input maps unavailable for this field")
        return self.maps

    def get_alms(self):
        """ Get the :math:`a_{\\ell m}` coefficients of this field. They
        include the effects of masking, as well as contaminant deprojection
        and purification (if required when generating this
        :obj:`NmtField`).

        Returns:
            (`array`): 2D array containing the field's :math:`a_{\\ell m}` s.
        """
        if self.alm is None:
            raise ValueError("Mask-only fields have no alms")
        return self.alm

    def get_templates(self):
        """ Get this field's set of contaminant templates maps. The
        returned maps will have the mask applied to them.

        Returns:
            (`array`): 3D array containing the field's contaminant maps.
        """
        if self.temp is None:
            raise ValueError("Input templates unavailable for this field")
        return self.temp

    def _purify(self, mask, alm_mask, maps_u, n_iter, task=[False, True],
                return_maps=True):
        # 1. Spin-0 mask bit
        # Multiply by mask
        maps = maps_u*mask[None, :]
        # Compute alms
        alms = ut.map2alm(maps, 2, self.minfo,
                          self.ainfo_mask, n_iter=n_iter)

        # 2. Spin-1 mask bit
        # Compute spin-1 mask
        ls = np.arange(self.ainfo_mask.lmax+1)
        # The minus sign is because of the definition of E-modes
        fl = -np.sqrt((ls+1.0)*ls)
        walm = hp.almxfl(alm_mask, fl,
                         mmax=self.ainfo_mask.mmax)
        walm = np.array([walm, walm*0])
        wmap = ut.alm2map(walm, 1, self.minfo, self.ainfo_mask)
        # Product with spin-1 mask
        maps = np.array([wmap[0]*maps_u[0]+wmap[1]*maps_u[1],
                         wmap[0]*maps_u[1]-wmap[1]*maps_u[0]])
        # Compute SHT, multiply by
        # 2*sqrt((l+1)!(l-2)!/((l-1)!(l+2)!)) and add to alms
        palm = ut.map2alm(maps, 1, self.minfo,
                          self.ainfo, n_iter=n_iter)
        fl[2:] = 2/np.sqrt((ls[2:]+2.0)*(ls[2:]-1.0))
        fl[:2] = 0
        for ipol, purify in enumerate(task):
            if purify:
                alms[ipol] += hp.almxfl(palm[ipol],
                                        fl[:self.ainfo.lmax+1],
                                        mmax=self.ainfo.mmax)

        # 3. Spin-2 mask bit
        # Compute spin-0 mask
        # The extra minus sign is because of the scalar SHT below
        # (E-mode definition for spin=0).
        fl[2:] = -np.sqrt((ls[2:]+2.0)*(ls[2:]-1.0))
        fl[:2] = 0
        walm[0] = hp.almxfl(walm[0], fl, mmax=self.ainfo_mask.mmax)
        wmap = ut.alm2map(walm, 2, self.minfo, self.ainfo_mask)
        # Product with spin-2 mask
        maps = np.array([wmap[0]*maps_u[0]+wmap[1]*maps_u[1],
                         wmap[0]*maps_u[1]-wmap[1]*maps_u[0]])
        # Compute SHT, multiply by
        # sqrt((l-2)!/(l+2)!) and add to alms
        palm = np.array([ut.map2alm(np.array([m]), 0, self.minfo,
                                    self.ainfo, n_iter=n_iter)[0]
                         for m in maps])
        fl[2:] = 1/np.sqrt((ls[2:]+2.0)*(ls[2:]+1.0) *
                           ls[2:]*(ls[2:]-1))
        fl[:2] = 0
        for ipol, purify in enumerate(task):
            if purify:
                alms[ipol] += hp.almxfl(palm[ipol],
                                        fl[:self.ainfo.lmax+1],
                                        mmax=self.ainfo.mmax)

        if return_maps:
            # 4. Compute purified map if needed
            maps = ut.alm2map(alms, 2, self.minfo, self.ainfo)
            return alms, maps
        return alms

    @property
    def Nw(self):
        """ Shot noise contribution associated with the mask for
        catalog-based fields (``0`` for standard fields).
        """
        return self._Nw

    @property
    def Nf(self):
        """ Shot noise contribution to the weighted field power
        spectrum for catalog-based fields (``0`` for standard
        fields).
        """
        return self._Nf

    @property
    def alpha(self):
        """ Ratio of random to data catalog sources (``None``
        for standard fields or catalog-based fields without
        randoms).
        """
        return self._alpha


class NmtFieldFlat(object):
    """ An :obj:`NmtFieldFlat` object contains all the information
    describing the flat-sky fields to correlate, including their observed
    maps, masks and contaminant templates.

    Args:
        lx (:obj:`float`): Size of the patch in the `x` direction (in
            radians).
        ly (:obj:`float`): Size of the patch in the `y` direction (in
            radians).
        mask (`array`): 2D array of shape ``(ny, nx)`` containing the
            field's mask.
        maps (`array`): Array containing the observed maps for this
            field. Should be at 3-dimensional. The first dimension
            corresponds to the number of maps, which should be 1 for a
            spin-0 field and 2 otherwise. The other dimensions should be
            ``(ny, nx)``. If ``None``, this field will only contain a
            mask but no maps. The field can then be used to compute a
            mode-coupling matrix, for instance, but not actual power
            spectra.
        spin (:obj:`int`): Spin of this field. If ``None`` it will
            default to 0 or 2 if ``maps`` contains 1 or 2 maps, respectively.
        templates (`array`): Array containing a set of contaminant templates
            for this field. This array should have shape
            ``[ntemp,nmap,ny,nx]``, where ``ntemp`` is the number of
            templates, and ``nmap`` should be 1 for spin-0 fields and, 2
            otherwise. The best-fit contribution from each contaminant is
            automatically removed from the maps unless ``templates=None``.
        beam (`array`): Spherical harmonic transform of the instrumental beam
            (assumed to be rotationally symmetric). Should be 2D, with shape
            ``(2,nl)``. ``beam[0]`` contains the values of :math:`\\ell` for
            which de beam is defined, with ``beam[1]`` containing the
            corresponding beam values. If None, no beam will be corrected for.
            Note that NaMaster does not automatically correct for pixel window
            function effects, as this is not always appropriate. If you want
            to correct for these effects, you should include them pixel window
            function in the beam or account for it when interpreting the
            measured power spectrum.
        purify_e (:obj:`bool`): Purify E-modes?
        purify_b (:obj:`bool`): Purify B-modes?
        tol_pinv (:obj:`float`): When computing the pseudo-inverse of the
            contaminant covariance matrix. See documentation of
            :meth:`~pymaster.utils.moore_penrose_pinvh`. Only relevant if
            passing contaminant templates that are likely to be highly
            correlated. If ``None``, it will default to the internal value,
            which can be accessed via
            :meth:`~pymaster.utils.get_default_params`, and modified via
            :meth:`~pymaster.utils.set_tol_pinv_default`.
        masked_on_input (:obj:`bool`): Set to ``True`` if the input maps and
            templates are already multiplied by the mask. Note that this is
            not advisable if you're using purification, as correcting for this
            usually incurs inaccuracies around the mask edges that may lead
            to significantly biased power spectra.
        lite (:obj:`bool`): Set to ``True`` if you want to only store the bare
            minimum necessary to run a standard pseudo-:math:`C_\\ell` with
            deprojection and purification, but you don't care about
            deprojection bias. This will significantly reduce the memory taken
            up by the resulting object.
    """
    def __init__(self, lx, ly, mask, maps, spin=None, templates=None,
                 beam=None, purify_e=False, purify_b=False,
                 tol_pinv=None, masked_on_input=False, lite=False):
        self.fl = None

        if tol_pinv is None:
            tol_pinv = ut.nmt_params.tol_pinv_default

        pure_e = 0
        if purify_e:
            pure_e = 1
        pure_b = 0
        if purify_b:
            pure_b = 1
        masked_input = 0
        if masked_on_input:
            masked_input = 1

        if (lx < 0) or (ly < 0):
            raise ValueError("Must supply sensible dimensions for "
                             "flat-sky field")
        # Flatten arrays and check dimensions
        shape_2D = np.shape(mask)
        self.ny = shape_2D[0]
        self.nx = shape_2D[1]

        if maps is None:
            mask_only = True
            if spin is None:
                raise ValueError("Please supply field spin")
            lite = True
        else:
            mask_only = False
            # As in the curved case, to ensure right type and endianness (and
            # solve the problems when reading it from a fits file)
            maps = np.array(maps, dtype=np.float64)

            nmaps = len(maps)
            if (nmaps != 1) and (nmaps != 2):
                raise ValueError("Must supply 1 or 2 maps per field")

            if spin is None:
                if nmaps == 1:
                    spin = 0
                else:
                    spin = 2
            else:
                if (((spin != 0) and nmaps == 1) or
                        ((spin == 0) and nmaps != 1)):
                    raise ValueError("Spin-zero fields are "
                                     "associated with a single map")

        if (pure_e or pure_b) and spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")

        # Flatten mask
        msk = (mask.astype(np.float64)).flatten()

        if (not mask_only):
            # Flatten maps
            mps = []
            for m in maps:
                if np.shape(m) != shape_2D:
                    raise ValueError("Mask and maps don't have the same shape")
                mps.append((m.astype(np.float64)).flatten())
            mps = np.array(mps)

        # Flatten templates
        if isinstance(templates, (list, tuple, np.ndarray)):
            tmps = []
            for t in templates:
                tmp = []
                if len(t) != nmaps:
                    raise ValueError("Maps and templates should have the "
                                     "same number of maps")
                for m in t:
                    if np.shape(m) != shape_2D:
                        raise ValueError("Mask and templates don't have "
                                         "the same shape")
                    tmp.append((m.astype(np.float64)).flatten())
                tmps.append(tmp)
            tmps = np.array(tmps)
        else:
            if templates is not None:
                raise ValueError("Input templates can only be an array "
                                 "or None")

        # Form beam
        if isinstance(beam, (list, tuple, np.ndarray)):
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.array([[-1.], [-1.]])
            else:
                raise ValueError("Input beam can only be an array or "
                                 "None")

        if mask_only:
            self.fl = lib.field_alloc_empty_flat(self.nx, self.ny,
                                                 lx, ly, spin,
                                                 msk, beam_use,
                                                 pure_e, pure_b)
        else:
            # Generate field
            if isinstance(templates, (list, tuple, np.ndarray)):
                self.fl = lib.field_alloc_new_flat(self.nx, self.ny, lx, ly,
                                                   spin, msk, mps, tmps,
                                                   beam_use, pure_e, pure_b,
                                                   tol_pinv, masked_input,
                                                   int(lite))
            else:
                self.fl = lib.field_alloc_new_notemp_flat(self.nx, self.ny,
                                                          lx, ly, spin,
                                                          msk, mps,
                                                          beam_use, pure_e,
                                                          pure_b, masked_input,
                                                          int(lite))
        self.lite = lite

    def __del__(self):
        if self.fl is not None:
            if lib.field_flat_free is not None:
                lib.field_flat_free(self.fl)
            self.fl = None

    def get_mask(self):
        """ Returns this field's mask as a 2D array with shape
        ``(ny, nx)``.

        Returns:
            (`array`): Mask.
        """
        msk = lib.get_mask_flat(self.fl,
                                int(self.fl.npix)).reshape([self.ny,
                                                            self.nx])

        return msk

    def get_maps(self):
        """ Returns a 3D array with shape ``(nmap, ny, nx)`` corresponding
        to the observed maps for this field. If the field was initialized
        with contaminant templates, the maps returned by this function
        have their best-fit contribution from these contaminants removed.

        Returns:
            (`array`): Maps.
        """
        if self.lite:
            raise ValueError("Input maps unavailable for lightweight fields. "
                             "To use this function, create an `NmtFieldFlat` "
                             "object with `lite=False`.")
        maps = np.zeros([self.fl.nmaps, self.fl.npix])
        for imap in range(self.fl.nmaps):
            maps[imap, :] = lib.get_map_flat(self.fl, imap, int(self.fl.npix))
        mps = maps.reshape([self.fl.nmaps, self.ny, self.nx])

        return mps

    def get_templates(self):
        """ Returns a 4D array with shape ``(ntemp, nmap, ny, nx)``,
        corresponding to the contaminant templates passed when initializing
        this field.

        Return:
            (`array`): Contaminant maps.
        """
        if self.lite:
            raise ValueError("Input maps unavailable for lightweight fields. "
                             "To use this function, create an `NmtFieldFlat` "
                             "object with `lite=False`.")
        temp = np.zeros([self.fl.ntemp, self.fl.nmaps, self.fl.npix])
        for itemp in range(self.fl.ntemp):
            for imap in range(self.fl.nmaps):
                temp[itemp, imap, :] = lib.get_temp_flat(
                    self.fl, itemp, imap, int(self.fl.npix)
                )
        tmps = temp.reshape([self.fl.ntemp, self.fl.nmaps, self.ny, self.nx])

        return tmps

    def get_ell_sampling(self):
        """ Returns the finest :math:`\\ell` sampling at which intermediate
        power spectra calculated involving this field are evaluated.

        Return:
            (`array`): Array of :math:`\\ell` values.
        """
        ells = lib.get_ell_sampling_flat(self.fl, int(self.fl.fs.n_ell))
        return ells


class NmtFieldCatalog(NmtField):
    """ An :obj:`NmtFieldCatalog` object contains all the information
    describing a field sampled at the discrete positions of
    a given catalog of sources. If you want to estimate clustering power
    spectra directly from catalogs, use :obj:`NmtFieldCatalogClustering`
    instead. The mathematical background for these fields was laid out
    in `Baleato & White 2024 <https://arxiv.org/abs/2312.12285>`_ , and
    expanded in Wolz et al. 2024.

    Args:
        positions (`array`): Source positions, provided as a list or array
            of 2 arrays. If ``lonlat`` is True, the arrays should contain the
            longitude and latitude of the sources, in this order, and in
            degrees (e.g. R.A. and Dec. if using Equatorial coordinates).
            Otherwise, the arrays should contain the colatitude and
            longitude of each source in radians (i.e. the spherical
            coordinates :math:`(\\theta,\\phi)`).
        weights (`array`): An array containing the weight assigned to
            each source.
        field (`array`): A list of 1 or 2 arrays (for spin-0 and spin-s
            fields, respectively), containing the value of the field
            whose power spectra we aim to calculate at the positions of
            each source.
        lmax (:obj:`int`): Maximum multipole up to which the spherical
            harmonics of this field will be computed.
        lmax_mask (:obj:`int`): Maximum multipole up to which the spherical
            harmonics of this field's mask will be computed. If ``None``,
            it will default to ``lmax``. Note that you should explore the
            sensitivity of the recovered :math:`C_\\ell` to the choice of
            ``lmax_mask``, and enlarge it if necessary.
        spin (:obj:`int`): Spin of this field. If ``None`` it will
            default to 0 or 2 if ``field`` contains 1 or 2 arrays,
            respectively.
        beam (`array`): Spherical harmonic transform of the instrumental beam
            (assumed to be rotationally symmetric - i.e. no :math:`m`
            dependence) associated with any smoothing that the field that
            was sampled on these catalog sources may have undergone. If
            ``None``, no beam will be corrected for. Otherwise, this array
            should have at least of size ``lmax+1``.
        field_is_weighted (:obj:`bool`): Set to ``True`` if the input field has
            already been multiplied by the source weights.
        lonlat (:obj:`bool`): If ``True``, longitude and latitude in degrees
            are provided as input. If ``False``, colatitude and longitude in
            radians are provided instead.
    """
    def __init__(self, positions, weights, field, lmax, lmax_mask=None,
                 spin=None, beam=None, field_is_weighted=False, lonlat=False):
        # 0. Preliminary initializations
        if ut.HAVE_DUCC:
            self.sht_calculator = 'ducc'
        else:
            raise ValueError("DUCC is needed, but currently not installed.")

        # These first attributes are compulsory for all fields
        self.lite = True
        self.mask = None
        self.beam = None
        self.n_iter = None
        self.n_iter_mask = None
        self.pure_e = False
        self.pure_b = False
        self.alm = None
        self.alm_mask = None
        self._Nw = 0.
        self._Nf = 0.
        self.ainfo = ut.NmtAlmInfo(lmax)
        self._alpha = None
        self.is_catalog = True

        # The remaining attributes are only required for non-lite maps
        self.maps = None
        self.temp = None
        self.alm_temp = None
        self.minfo = None
        self.n_temp = 0
        self.anisotropic_mask = False
        self.mask_a = None
        self.alm_mask_a = None

        # Sanity checks on positions and weights
        positions = np.array(positions, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        if np.shape(positions) != (2, len(weights)):
            raise ValueError("Positions must be 2D array of shape"
                             f" {(2, len(weights))}.")
        if lonlat:
            # Put in radians and swap order
            positions = np.pi/180.*positions[::-1]
            # Go from latitude to colatitude
            positions[0] = np.pi/2 - positions[0]
        if not (np.logical_and(positions[0] >= 0.,
                               positions[0] <= np.pi)).all():
            if lonlat:
                raise ValueError(
                    "Second dimension of positions must be latitude in "
                    "degrees, between -90 and 90."
                )
            else:
                raise ValueError(
                    "First dimension of positions must be colatitude in "
                    "radians, between 0 and pi."
                )
        if not (np.logical_and(positions[1] >= 0.,
                               positions[1] <= 2*np.pi)).all():
            if lonlat:
                raise ValueError(
                    "First dimension of positions must be longitude in "
                    "degrees, between 0 and 360."
                )
            else:
                raise ValueError(
                    "Second dimension of positions must be longitude in "
                    "radians, between 0 and 2*pi."
                )

        # Compute mask shot noise
        self._Nw = np.sum(weights**2.)/(4.*np.pi)

        # 1. Compute mask alms and beam
        # Sanity checks
        if lmax_mask is None:
            lmax_mask = lmax
        self.ainfo_mask = ut.NmtAlmInfo(lmax_mask)
        # Mask alms
        self.alm_mask = ut._catalog2alm_ducc0(weights, positions,
                                              spin=0, lmax=lmax_mask)[0]
        # Beam
        if isinstance(beam, (list, tuple, np.ndarray)):
            if len(beam) <= lmax:
                raise ValueError("Input beam must have at least %d elements "
                                 "given the input maximum "
                                 "multipole" % (lmax+1))
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.ones(lmax+1)
            else:
                raise ValueError("Input beam can only be an array or None\n")
        self.beam = beam_use

        # 2. Compute field alms
        # If only positions and weights, just return here
        if field is None:
            if spin is None:
                raise ValueError("If field is None, spin needs to be "
                                 "provided.")
            self.spin = spin
            return

        # Ensure it's 2D
        self.field = np.atleast_2d(np.array(field, dtype=np.float64))
        # Set spin
        if spin is None:
            spin = 0 if len(self.field) == 1 else 2
        self.spin = spin
        self.nmaps = 2 if spin else 1
        nsrcs = len(weights)
        if self.field.shape != (self.nmaps, nsrcs):
            raise ValueError(f"Field should have shape {(self.nmaps, nsrcs)}.")
        if not field_is_weighted:
            self.field *= weights
        self.alm = ut._catalog2alm_ducc0(self.field, positions,
                                         spin=spin, lmax=lmax)

        # 3. Compute field noise bias on mask pseudo-C_ell
        self._Nf = np.sum(self.field**2)/(4*np.pi*self.nmaps)


class NmtFieldCatalogClustering(NmtField):
    """ An :obj:`NmtFieldCatalogClustering` object contains all the
    information describing a field represented by the position and density
    of discrete sources, with a survey footprint characterised by a set of
    random points or through a standard mask.

    Args:
        positions (`array`): Source positions, provided as a list or array
            of 2 arrays. If ``lonlat`` is True, the arrays should contain the
            longitude and latitude of the sources, in this order, and in
            degrees (e.g. R.A. and Dec. if using Equatorial coordinates).
            Otherwise, the arrays should contain the colatitude and
            longitude of each source in radians (i.e. the spherical
            coordinates :math:`(\\theta,\\phi)`).
        weights (`array`): An array containing the weight assigned to
            each source.
        positions_rand (`array`): As ``positions`` for the random catalog.
        weights_rand (`array`): As ``weights`` for the random catalog.
        lmax (:obj:`int`): Maximum multipole up to which the spherical
            harmonics of this field will be computed.
        lonlat (:obj:`bool`): If ``True``, longitude and latitude in degrees
            are provided as input. If ``False``, colatitude and longitude in
            radians are provided instead.
    """
    def __init__(self, positions, weights, positions_rand, weights_rand,
                 lmax, lonlat=False):
        # Preliminary initializations
        if ut.HAVE_DUCC:
            self.sht_calculator = 'ducc'
        else:
            raise ValueError("DUCC is needed, but currently not installed.")

        # These first attributes are compulsory for all fields
        self.lite = True
        self.mask = None
        self.beam = np.ones(lmax+1)
        self.n_iter = None
        self.n_iter_mask = None
        self.pure_e = False
        self.pure_b = False
        self.alm = None
        self.alm_mask = None
        self._Nw = 0.
        self._Nf = 0.
        self.ainfo = ut.NmtAlmInfo(lmax)
        self._alpha = 0
        self.spin = 0
        self.is_catalog = True

        # The remaining attributes are only required for non-lite maps
        self.maps = None
        self.temp = None
        self.alm_temp = None
        self.minfo = None
        self.n_temp = 0
        self.anisotropic_mask = False
        self.mask_a = None
        self.alm_mask_a = None

        # Sanity checks on positions and weights
        def process_pos_w(pos, w, kind):
            pos = np.array(pos, dtype=np.float64)
            w = np.array(w, dtype=np.float64)
            if np.shape(pos) != (2, len(w)):
                raise ValueError(f"{kind} positions must be 2D array of"
                                 f" shape {(2, len(w))}.")
            if lonlat:
                # Put in radians and swap order
                pos = np.pi/180.*pos[::-1]
                # Go from latitude to colatitude
                pos[0] = np.pi/2 - pos[0]
            if not (np.logical_and(pos[0] >= 0., pos[0] <= np.pi)).all():
                if lonlat:
                    raise ValueError(
                        f"Second dimension of {kind} positions must be "
                        "latitude in degrees, between -90 and 90."
                    )
                else:
                    raise ValueError(
                        f"First dimension of {kind} positions must be "
                        "colatitude in radians, between 0 and pi."
                    )
            if not (np.logical_and(pos[1] >= 0., pos[1] <= 2*np.pi)).all():
                if lonlat:
                    raise ValueError(
                        f"First dimension of {kind} positions must be "
                        "longitude in degrees, between 0 and 360."
                    )
                else:
                    raise ValueError(
                        f"Second dimension of {kind} positions must be "
                        "longitude in radians, between 0 and 2*pi."
                    )
            return pos, w

        positions, weights = process_pos_w(positions, weights, "data")
        positions_rand, weights_rand = process_pos_w(positions_rand,
                                                     weights_rand, "random")

        # Compute alpha
        self._alpha = np.sum(weights)/np.sum(weights_rand)

        # Compute mask shot noise
        self._Nw = self._alpha**2*np.sum(weights_rand**2.)/(4.*np.pi)

        # Compute mask alms
        # Sanity checks
        self.ainfo_mask = ut.NmtAlmInfo(lmax)
        # Mask alms
        self.alm_mask = ut._catalog2alm_ducc0(weights_rand*self._alpha,
                                              positions_rand,
                                              spin=0, lmax=lmax)

        # Compute field alms
        self.alm = ut._catalog2alm_ducc0(weights, positions,
                                         spin=0, lmax=lmax)-self.alm_mask
        self.alm_mask = self.alm_mask[0]

        # Compute field shot noise
        self._Nf = np.sum(weights**2)/(4*np.pi)+self._Nw
