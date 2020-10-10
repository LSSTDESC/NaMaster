from pymaster import nmtlib as lib
import numpy as np
from pymaster.utils import NmtWCSTranslator


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
                 purify_e=False, purify_b=False, n_iter_mask_purify=3,
                 tol_pinv=1E-10, wcs=None, n_iter=3, lmax_sht=-1,
                 masked_on_input=False, lite=False):
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

        wt = NmtWCSTranslator(wcs, mask.shape)
        if wt.is_healpix == 0:
            if wt.flip_th:
                mask = mask[::-1, :]
            if wt.flip_ph:
                mask = mask[:, ::-1]
            mask = mask.reshape(wt.npix)

        if maps is None:
            mask_only = True
            if spin is None:
                raise ValueError("Please supply field spin")
            lite = True
        else:
            mask_only = False
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

            if wt.is_healpix and (len(maps[0]) != len(mask)):
                raise ValueError("All maps must have the same resolution")

        if (pure_e or pure_b) and spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")

        # Flatten if 2D maps
        if (not mask_only) and (wt.is_healpix == 0):
            try:
                maps = np.array(maps)
                if wt.flip_th:
                    maps = maps[:, ::-1, :]
                if wt.flip_ph:
                    maps = maps[:, :, ::-1]
                maps = maps.reshape([len(maps), wt.npix])
            except:
                raise ValueError("Input maps have the wrong shape")

        if isinstance(templates, (list, tuple, np.ndarray)):
            ntemp = len(templates)
            if (len(templates[0]) != 1) and (len(templates[0]) != 2):
                raise ValueError("Must supply 1 or 2 maps per field")

            if wt.is_healpix == 0:  # Flatten if 2D maps
                try:
                    templates = np.array(templates)
                    if wt.flip_th:
                        templates = templates[:, :, ::-1, :]
                    if wt.flip_ph:
                        templates = templates[:, :, :, ::-1]
                    templates = templates.reshape([ntemp, len(maps), wt.npix])
                except:
                    raise ValueError("Input templates have the wrong shape")

            if len(templates[0][0]) != len(mask):
                raise ValueError("All maps must have the same resolution")
        else:
            if templates is not None:
                raise ValueError("Input templates can only be an array "
                                 "or None\n")

        if lmax_sht > 0:
            lmax = lmax_sht
        else:
            lmax = wt.get_lmax()

        if isinstance(beam, (list, tuple, np.ndarray)):
            if len(beam) <= lmax:
                raise ValueError("Input beam must have at least %d elements "
                                 "given the input map resolution" % (lmax))
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.ones(lmax+1)
            else:
                raise ValueError("Input beam can only be an array or None\n")

        if mask_only:
            self.fl = lib.field_alloc_empty(wt.is_healpix, wt.nside, lmax_sht,
                                            wt.nx, wt.ny,
                                            wt.d_phi, wt.d_theta,
                                            wt.phi0, wt.theta_max, spin,
                                            mask, beam_use, pure_e, pure_b,
                                            n_iter_mask_purify)
        else:
            if isinstance(templates, (list, tuple, np.ndarray)):
                self.fl = lib.field_alloc_new(wt.is_healpix, wt.nside,
                                              lmax_sht, wt.nx, wt.ny,
                                              wt.d_phi, wt.d_theta,
                                              wt.phi0, wt.theta_max, spin,
                                              mask, maps, templates, beam_use,
                                              pure_e, pure_b,
                                              n_iter_mask_purify,
                                              tol_pinv, n_iter, masked_input,
                                              int(lite))
            else:
                self.fl = lib.field_alloc_new_notemp(
                    wt.is_healpix, wt.nside, lmax_sht, wt.nx, wt.ny, wt.d_phi,
                    wt.d_theta, wt.phi0, wt.theta_max, spin,
                    mask, maps, beam_use, pure_e, pure_b, n_iter_mask_purify,
                    n_iter, masked_input, int(lite))
        self.lite = lite

    def __del__(self):
        if self.fl is not None:
            lib.field_free(self.fl)
            self.fl = None

    def get_maps(self):
        """
        Returns a 2D array ([nmap][npix]) corresponding to the observed maps \
        for this field. If the field was initialized with contaminant \
        templates, the maps returned by this function have their best-fit \
        contribution from these contaminants removed.

        :return: 2D array of maps
        """
        if self.lite:
            raise ValueError("Input maps unavailable for lightweight fields")
        maps = np.zeros([self.fl.nmaps, self.fl.npix])
        for imap in range(self.fl.nmaps):
            maps[imap, :] = lib.get_map(self.fl, imap, int(self.fl.npix))
        else:
            mps = maps
        return mps

    def get_templates(self):
        """
        Returns a 3D array ([ntemp][nmap][npix]) corresponding to the \
        contaminant templates passed when initializing this field.

        :return: 3D array of maps
        """
        if self.lite:
            raise ValueError("Input maps unavailable for lightweight fields")
        temp = np.zeros([self.fl.ntemp, self.fl.nmaps, self.fl.npix])
        for itemp in range(self.fl.ntemp):
            for imap in range(self.fl.nmaps):
                temp[itemp, imap, :] = lib.get_temp(
                    self.fl, itemp, imap, int(self.fl.npix)
                )
        else:
            tmps = temp
        return tmps


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
            lib.field_flat_free(self.fl)
            self.fl = None

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
