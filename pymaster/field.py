from pymaster import nmtlib as lib
import numpy as np
import healpy as hp
import pymaster.utils as ut


class NmtField(object):
    """
    An NmtField object contains all the information describing the fields to
    correlate, including their observed maps, masks and contaminant templates.

    :param mask: array containing a map corresponding to the field's mask. \
        Should be 1-dimensional for a HEALPix map or 2-dimensional for a map \
        with rectangular pixelization.
    :param maps: array containing the observed maps for this field. Should be \
        at least 2-dimensional. The first dimension corresponds to the number \
        of maps, which should be 1 for a spin-0 field and 2 otherwise. \
        The other dimensions should be [npix] for HEALPix maps or \
        [ny,nx] for maps with rectangular pixels. For a spin>0 field, the two \
        maps to pass should be the usual Q/U Stokes parameters for \
        polarization, or e1/e2 (gamma1/gamma2 etc.) in the case of cosmic \
        shear. It is important to note that NaMaster uses the same \
        polarization convention as HEALPix (i.e. with the x-coordinate \
        growing with increasing colatitude theta). It is however more common \
        for galaxy ellipticities to be provided using the IAU convention \
        (i.e. x grows with declination). In this case, the sign of the \
        e2/gamma2 map should be swapped before using it to create an \
        NmtField. See more \
        `here <https://healpix.jpl.nasa.gov/html/intronode12.htm>`_ . \
        If `None`, this field will only contain a mask but no maps. The field \
        can then be used to compute a mode-coupling matrix, for instance, \
        but not actual power spectra.
    :param spin: field's spin. If `None` it will be set to 0 if there is
        a single map on input, and will default to 2 if there are 2 maps.
    :param templates: array containing a set of contaminant templates for \
        this field. This array should have shape [ntemp][nmap]..., where \
        ntemp is the number of templates, nmap should be 1 for spin-0 fields \
        and 2 otherwise. The other dimensions should be [npix] for \
        HEALPix maps or [ny,nx] for maps with rectangular pixels. The \
        best-fit contribution from each contaminant is automatically removed \
        from the maps unless templates=None.
    :param beam: spherical harmonic transform of the instrumental beam \
        (assumed to be rotationally symmetric - i.e. no m dependence). If \
        None, no beam will be corrected for. Otherwise, this array should \
        have at least as many elements as the maximum multipole sampled by \
        the maps + 1 (e.g. if a HEALPix map, it should contain 3*nside \
        elements, corresponding to multipoles from 0 to 3*nside-1).
    :param purify_e: use pure E-modes?
    :param purify_b: use pure B-modes?
    :param n_iter_mask_purify: number of iterations used to compute an \
        accurate SHT of the mask when using E/B purification.
    :param tol_pinv: when computing the pseudo-inverse of the contaminant \
        covariance matrix, all eigenvalues below tol_pinv * max_eval will be \
        treated as singular values, where max_eval is the largest eigenvalue. \
        Only relevant if passing contaminant templates that are likely to be \
        highly correlated.
    :param wcs: a WCS object if using rectangular pixels (see \
        http://docs.astropy.org/en/stable/wcs/index.html).
    :param n_iter: number of iterations when computing a_lms.
    :param lmax_sht: maximum multipole up to which map power spectra will be \
        computed. If negative or zero, the maximum multipole given the map \
        resolution will be used (e.g. 3 * nside - 1 for HEALPix maps).
    :param masked_on_input: set to `True` if input maps and templates are \
        already multiplied by the masks. Note that this is not advisable \
        if you're using purification.
    :param lite: set to `True` if you want to only store the bare minimum \
        necessary to run a standard pseudo-Cl with deprojection and \
        purification, but you don't care about deprojection bias. This \
        will reduce the memory taken up by the resulting object.
    """
    def __init__(self, mask, maps, spin=None, templates=None, beam=None,
                 purify_e=False, purify_b=False, n_iter_mask=3,
                 tol_pinv=1E-10, wcs=None, n_iter=3, lmax=-1, lmax_mask=-1,
                 masked_on_input=False, lite=False):
        # 0. Preliminary initializations
        # These first attributes are compulsory for all fields
        self.lite = lite
        self.mask = None
        self.beam = None
        self.n_iter = n_iter
        self.n_iter_mask = n_iter_mask
        # The alm is only required for non-mask-only maps
        self.alm = None
        # The remaining attributes are only required for non-lite maps
        self.maps = None
        self.temp = None
        self.alm_temp = None
        # The mask alms are only stored if computed for non-lite maps
        self.alm_mask = None
        self.n_temp = 0

        # 1. Store mask and beam
        # This ensures the mask will have the right type
        # and endianness (can cause issues when read from
        # some FITS files).
        mask = mask.astype(np.float64)
        self.wt = ut.NmtWCSTranslator(wcs, mask.shape)
        self.mask = self.wt.reform_map(mask)
        if lmax <= 0:
            lmax = self.wt.get_lmax()
        if lmax_mask <= 0:
            lmax_mask = self.wt.get_lmax()
        self.ainfo = ut.AlmInfo(lmax)
        self.ainfo_mask = ut.AlmInfo(lmax_mask)

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

        maps = self.wt.reform_map(maps)

        pure_any = purify_e or purify_b
        if pure_any and self.spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")
        self.pure_e = purify_e
        self.pure_b = purify_b

        # 3. Check templates
        if isinstance(templates, (list, tuple, np.ndarray)):
            templates = np.array(templates, dtype=np.float64)
            if (len(templates[0]) != 1) and (len(templates[0]) != 2):
                raise ValueError("Must supply 1 or 2 maps per field")
            templates = self.wt.reform_map(templates)
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
                good = mask > 0
                goodm = mask[good]
                maps_unmasked[:, good] /= goodm[None, :]
                if w_temp:
                    temp_unmasked[:, :, good] /= goodm[None, None, :]

        # 5. Mask all maps
        maps = np.array(maps)
        if w_temp:
            templates = np.array(templates)
        if not masked_on_input:
            maps *= self.mask[None, :]
            if w_temp:
                templates *= self.mask[None, None, :]

        # 6. Compute template normalization matrix and deproject
        if w_temp:
            M = np.array([[self.wt.minfo.dot_map(t1, t2)
                           for t1 in templates]
                          for t2 in templates])
            iM = ut.moore_penrose_pinvh(M, tol_pinv)
            prods = np.array([self.wt.minfo.dot_map(t, maps)
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
            self.alm, maps = self._purify(mask, alm_mask, maps_unmasked,
                                          n_iter=n_iter, task=task)
            if w_temp and (not self.lite):
                alm_temp = np.array([self._purify(mask, alm_mask, t,
                                                  n_iter=n_iter,
                                                  task=task)[0]
                                     for t in temp_unmasked])
                # IMPORTANT: at this stage, maps and self.alm contain the
                # purified map and SH coefficients. However, although alm_temp
                # contains the purified SH coefficients, templates contains the
                # ***non-purified*** maps. This is to speed up the calculation
                # of the deprojection bias.
        else:
            self.alm = ut.map2alm(maps, self.spin, self.wt.minfo,
                                  self.ainfo, n_iter=n_iter)
            if w_temp and (not self.lite):
                alm_temp = np.array([ut.map2alm(t, self.spin, self.wt.minfo,
                                                self.ainfo, n_iter=n_iter)
                                     for t in templates])

        # 8. Store any additional information needed
        if not self.lite:
            self.maps = maps.copy()
            if w_temp:
                self.temp = templates.copy()
                self.alm_temp = alm_temp

    def is_compatible(self, other):
        if self.wt != other.wt:
            return False
        if self.ainfo != other.ainfo:
            return False
        if self.ainfo_mask != other.ainfo_mask:
            return False
        return True

    def get_mask(self):
        return self.mask

    def get_maps(self):
        if self.maps is None:
            raise ValueError("Input maps unavailable for this field")
        return self.maps

    def get_alms(self):
        if self.alm is None:
            raise ValueError("Mask-only fields have no alms")
        return self.alm

    def get_mask_alms(self):
        if self.alm_mask is None:
            amask = ut.map2alm(np.array([self.mask]), 0,
                               self.wt.minfo, self.ainfo_mask,
                               n_iter=self.n_iter_mask)[0]
            if not self.lite:  # Store while we're at it
                self.alm_mask = amask
        else:
            amask = self.alm_mask
        return amask

    def get_templates(self):
        if self.temp is None:
            raise ValueError("Input templates unavailable for this field")
        return self.temp

    def _purify(self, mask, alm_mask, maps_u, n_iter, task=[False, True],
                return_maps=True):
        # 1. Spin-0 mask bit
        # Multiply by mask
        maps = maps_u*mask[None, :]
        # Compute alms
        alms = ut.map2alm(maps, 2, self.wt.minfo,
                          self.ainfo_mask, n_iter=n_iter)

        # 2. Spin-1 mask bit
        # Compute spin-1 mask
        ls = np.arange(self.ainfo_mask.lmax+1)
        # The minus sign is because of the definition of E-modes
        fl = -np.sqrt((ls+1.0)*ls)
        walm = hp.almxfl(alm_mask, fl,
                         mmax=self.ainfo_mask.mmax)
        walm = np.array([walm, walm*0])
        wmap = ut.alm2map(walm, 1, self.wt.minfo, self.ainfo_mask)
        # Product with spin-1 mask
        maps = np.array([wmap[0]*maps_u[0]+wmap[1]*maps_u[1],
                         wmap[0]*maps_u[1]-wmap[1]*maps_u[0]])
        # Compute SHT, multiply by
        # 2*sqrt((l+1)!(l-2)!/((l-1)!(l+2)!)) and add to alms
        palm = ut.map2alm(maps, 1, self.wt.minfo,
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
        wmap = ut.alm2map(walm, 2, self.wt.minfo, self.ainfo_mask)
        # Product with spin-2 mask
        maps = np.array([wmap[0]*maps_u[0]+wmap[1]*maps_u[1],
                         wmap[0]*maps_u[1]-wmap[1]*maps_u[0]])
        # Compute SHT, multiply by
        # sqrt((l-2)!/(l+2)!) and add to alms
        palm = np.array([ut.map2alm(np.array([m]), 0, self.wt.minfo,
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
            maps = ut.alm2map(alms, 2, self.wt.minfo, self.ainfo)
            return alms, maps
        return alms


class NmtFieldFlat(object):
    """
    An NmtFieldFlat object contains all the information describing the \
    flat-sky fields to correlate, including their observed maps, masks \
    and contaminant templates.

    :param float lx,ly: size of the patch in the x and y directions (in \
        radians)
    :param mask: 2D array (nx,ny) containing a HEALPix map corresponding \
        to the field's mask.
    :param maps: 2 2D arrays (nmaps,nx,ny) containing the observed maps \
        for this field. The first dimension corresponds to the number of \
        maps, which should be 1 for a spin-0 field and 2 otherwise. \
        If `None`, this field will only contain a mask but no maps. The field \
        can then be used to compute a mode-coupling matrix, for instance, \
        but not actual power spectra.
    :param spin: field's spin. If `None` it will be set to 0 if there is
        a single map on input, and will default to 2 if there are 2 maps.
    :param templates: array of maps (ntemp,nmaps,nx,ny) containing a set \
        of contaminant templates for this field. This array should have \
        shape [ntemp][nmap][nx][ny], where ntemp is the number of \
        templates, nmap should be 1 for spin-0 fields and 2 for spin-2 \
        fields, and nx,ny define the patch. The best-fit contribution \
        from each contaminant is automatically removed from the maps \
        unless templates=None
    :param beam: 2D array (2,nl) defining the FT of the instrumental beam \
        (assumed to be rotationally symmetric). beam[0] should contain \
        the values of l for which de beam is defined, with beam[1] \
        containing the beam values. If None, no beam will be corrected \
        for.
    :param purify_e: use pure E-modes?
    :param purify_b: use pure B-modes?
    :param tol_pinv: when computing the pseudo-inverse of the contaminant \
        covariance matrix, all eigenvalues below tol_pinv * max_eval will \
        be treated as singular values, where max_eval is the largest \
        eigenvalue. Only relevant if passing contaminant templates that \
        are likely to be highly correlated.
    :param masked_on_input: set to `True` if input maps and templates are
        already multiplied by the masks. Note that this is not advisable
        if you're using purification.
    :param lite: set to `True` if you want to only store the bare minimum \
        necessary to run a standard pseudo-Cl with deprojection and \
        purification, but you don't care about deprojection bias. This \
        will reduce the memory taken up by the resulting object.
    """
    def __init__(self, lx, ly, mask, maps, spin=None, templates=None,
                 beam=None, purify_e=False, purify_b=False,
                 tol_pinv=1E-10, masked_on_input=False, lite=False):
        self.fl = None

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
        """
        Returns this field's mask as a 2D array ([ny][nx]).

        :return: 2D mask.
        """
        msk = lib.get_mask_flat(self.fl,
                                int(self.fl.npix)).reshape([self.ny,
                                                            self.nx])

        return msk

    def get_maps(self):
        """
        Returns a 3D array ([nmap][ny][nx]) corresponding to the observed \
        maps for this field. If the field was initialized with contaminant \
        templates, the maps returned by this function have their best-fit \
        contribution from these contaminants removed.

        :return: 3D array of flat-sky maps
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
        """
        Returns a 4D array ([ntemp][nmap][ny][nx]) corresponding to the \
        contaminant templates passed when initializing this field.

        :return: 4D array of flat-sky maps
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
