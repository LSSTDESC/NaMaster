from pymaster import nmtlib as lib
import numpy as np


class NmtField(object):
    """
    An NmtField object contains all the information describing the fields to correlate, including their observed maps, masks and contaminant templates.

    :param mask: array containing a HEALPix map corresponding to the field's mask.
    :param maps: 2D array containing the observed maps for this field. The first dimension corresponds to the number of maps, which should be 1 for a spin-0 field and 2 for a spin-2 field.
    :param templates: 3D array containing a set of contaminant templates for this field. This array should have shape [ntemp][nmap][npix], where ntemp is the number of templates, nmap should be 1 for spin-0 fields and 2 for spin-2 fields, and npix is the number of pixels per map. The best-fit contribution from each contaminant is automatically removed from the maps unless templates=None
    :param beam: spherical harmonic transform of the instrumental beam (assumed to be rotationally symmetric - i.e. no m dependence). If None, no beam will be corrected for. Otherwise, this array should have 3*nside elements, corresponding to multipoles from 0 to 3*nside-1.
    :param purify_e: use pure E-modes?
    :param purify_b: use pure B-modes?
    :param n_iter_mask_purify: number of iterations used to compute an accurate SHT of the mask when using E/B purification
    :param tol_pinv: when computing the pseudo-inverse of the contaminant covariance matrix, all eigenvalues below tol_pinv * max_eval will be treated as singular values, where max_eval is the largest eigenvalue. Only relevant if passing contaminant templates that are likely to be highly correlated.

    """

    def __init__(
        self,
        mask,
        maps,
        templates=None,
        beam=None,
        purify_e=False,
        purify_b=False,
        n_iter_mask_purify=3,
        tol_pinv=1E-10,
    ):
        self.fl = None

        pure_e = 0
        if purify_e:
            pure_e = 1
        pure_b = 0
        if purify_b:
            pure_b = 1

        nside = 2
        while 12 * nside * nside != len(mask):
            nside *= 2
            if nside > 65536:
                raise ValueError("Something is wrong with your input arrays")

        if (len(maps) != 1) and (len(maps) != 2):
            raise ValueError("Must supply 1 or 2 maps per field")
        if len(maps[0]) != len(mask):
            raise ValueError("All maps must have the same resolution")

        if isinstance(templates, (list, tuple, np.ndarray)):
            if (len(templates[0]) != 1) and (len(templates[0]) != 2):
                raise ValueError("Must supply 1 or 2 maps per field")
            if len(templates[0][0]) != len(mask):
                raise ValueError("All maps must have the same resolution")
        else:
            if templates is not None:
                raise ValueError("Input templates can only be an array or None\n")

        if isinstance(beam, (list, tuple, np.ndarray)):
            if len(beam) != 3 * nside:
                raise ValueError("Input beam must have 3*nside elements")
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.ones(3 * nside)
            else:
                raise ValueError("Input beam can only be an array or None\n")

        if isinstance(templates, (list, tuple, np.ndarray)):
            self.fl = lib.field_alloc_new(
                mask,
                maps,
                templates,
                beam_use,
                pure_e,
                pure_b,
                n_iter_mask_purify,
                tol_pinv,
            )
        else:
            self.fl = lib.field_alloc_new_notemp(
                mask, maps, beam_use, pure_e, pure_b, n_iter_mask_purify
            )

    def __del__(self):
        if self.fl is not None:
            lib.field_free(self.fl)
            self.fl = None

    def get_maps(self):
        """
        Returns a 2D array ([nmap][npix]) corresponding to the observed maps for this field. If the field was initialized with contaminant templates, the maps returned by this function have their best-fit contribution from these contaminants removed.
        
        :return: 2D array of HEALPix maps
        """
        maps = np.zeros([self.fl.nmaps, self.fl.npix])
        for imap in range(self.fl.nmaps):
            maps[imap, :] = lib.get_map(self.fl, imap, int(self.fl.npix))
        else:
            mps = maps
        return mps

    def get_templates(self):
        """
        Returns a 3D array ([ntemp][nmap][npix]) corresponding to the contaminant templates passed when initializing this field.

        :return: 3D array of HEALPix maps
        """
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
    An NmtFieldFlat object contains all the information describing the flat-sky fields to correlate, including their observed maps, masks and contaminant templates.

    :param float lx,ly: size of the patch in the x and y directions (in radians)
    :param mask: 2D array (nx,ny) containing a HEALPix map corresponding to the field's mask.
    :param maps: 2 2D arrays (nmaps,nx,ny) containing the observed maps for this field. The first dimension corresponds to the number of maps, which should be 1 for a spin-0 field and 2 for a spin-2 field.
    :param templates: array of maps (ntemp,nmaps,nx,ny) containing a set of contaminant templates for this field. This array should have shape [ntemp][nmap][nx][ny], where ntemp is the number of templates, nmap should be 1 for spin-0 fields and 2 for spin-2 fields, and nx,ny define the patch. The best-fit contribution from each contaminant is automatically removed from the maps unless templates=None
    :param beam: 2D array (2,nl) defining the FT of the instrumental beam (assumed to be rotationally symmetric). beam[0] should contain the values of l for which de beam is defined, with beam[1] containing the beam values. If None, no beam will be corrected for.
    :param purify_e: use pure E-modes?
    :param purify_b: use pure B-modes?
    :param tol_pinv: when computing the pseudo-inverse of the contaminant covariance matrix, all eigenvalues below tol_pinv * max_eval will be treated as singular values, where max_eval is the largest eigenvalue. Only relevant if passing contaminant templates that are likely to be highly correlated.

    """

    def __init__(
        self,
        lx,
        ly,
        mask,
        maps,
        templates=None,
        beam=None,
        purify_e=False,
        purify_b=False,
        tol_pinv=1E-10,
    ):
        self.fl = None

        pure_e = 0
        if purify_e:
            pure_e = 1
        pure_b = 0
        if purify_b:
            pure_b = 1

        if (lx < 0) or (ly < 0):
            raise ValueError("Must supply sensible dimensions for flat-sky field")
        # Flatten arrays and check dimensions
        shape_2D = np.shape(mask)
        nmaps = len(maps)
        self.ny = shape_2D[0]
        self.nx = shape_2D[1]
        if (nmaps != 1) and (nmaps != 2):
            raise ValueError("Must supply 1 or 2 maps per field")

        # Flatten mask
        msk = (mask.astype(np.float64)).flatten()

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
                    raise ValueError(
                        "Maps and templates should have the same number of maps"
                    )
                for m in t:
                    if np.shape(m) != shape_2D:
                        raise ValueError("Mask and templates don't have the same shape")
                    tmp.append((m.astype(np.float64)).flatten())
                tmps.append(tmp)
            tmps = np.array(tmps)
        else:
            if templates is not None:
                raise ValueError("Input templates can only be an array or None")

        # Form beam
        if isinstance(beam, (list, tuple, np.ndarray)):
            beam_use = beam
        else:
            if beam is None:
                beam_use = np.array([[-1.], [-1.]])
            else:
                raise ValueError("Input beam can only be an array or None")

        # Generate field
        if isinstance(templates, (list, tuple, np.ndarray)):
            self.fl = lib.field_alloc_new_flat(
                self.nx,
                self.ny,
                lx,
                ly,
                msk,
                mps,
                tmps,
                beam_use,
                pure_e,
                pure_b,
                tol_pinv,
            )
        else:
            self.fl = lib.field_alloc_new_notemp_flat(
                self.nx, self.ny, lx, ly, msk, mps, beam_use, pure_e, pure_b
            )

    def __del__(self):
        if self.fl is not None:
            lib.field_flat_free(self.fl)
            self.fl = None

    def get_maps(self):
        """
        Returns a 3D array ([nmap][ny][nx]) corresponding to the observed maps for this field. If the field was initialized with contaminant templates, the maps returned by this function have their best-fit contribution from these contaminants removed.
        
        :return: 3D array of flat-sky maps
        """
        maps = np.zeros([self.fl.nmaps, self.fl.npix])
        for imap in range(self.fl.nmaps):
            maps[imap, :] = lib.get_map_flat(self.fl, imap, int(self.fl.npix))
        mps = maps.reshape([self.fl.nmaps, self.ny, self.nx])

        return mps

    def get_templates(self):
        """
        Returns a 4D array ([ntemp][nmap][ny][nx]) corresponding to the contaminant templates passed when initializing this field.

        :return: 4D array of flat-sky maps
        """
        temp = np.zeros([self.fl.ntemp, self.fl.nmaps, self.fl.npix])
        for itemp in range(self.fl.ntemp):
            for imap in range(self.fl.nmaps):
                temp[itemp, imap, :] = lib.get_temp_flat(
                    self.fl, itemp, imap, int(self.fl.npix)
                )
        tmps = temp.reshape([self.fl.ntemp, self.fl.nmaps, self.ny, self.nx])

        return tmps
