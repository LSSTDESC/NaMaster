from pymaster import nmtlib as lib
import numpy as np


def get_master_coefficients(pcl_mask, lmax, spin1, spin2,
                            is_teb=False, pure_any=False,
                            l_toeplitz=-1, l_exact=-1, dl_band=-1):
    """
    Calculate the master coefficients for a set of mask power spectra
    and a given set of spin and purification combinations.

    Args:
        pcl_mask (`array`): The power spectrum of the mask(s). You
            can provide multiple masks at once by passing a 2D array
            of shape ``(nmask, lmax_mask+1)``, where ``nmask`` is the number
            of masks and ``lmax_mask`` is the maximum multipole of the
            mask power spectra.
        lmax (:obj:`int`): The maximum multipole to consider.
        spin1 (:obj:`int`): The spin of the first field.
        spin2 (:obj:`int`): The spin of the second field.
        pure_any (bool): Whether to return coupling coefficients for
            purified fields.
        is_teb (:obj:`bool`): If ``True``, all mode-coupling matrices
            (0-0,0-s,s-s) will be computed at the same time. In this
            case, ``spin1`` must be 0 and ``spin2`` must be non-zero.
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

    Returns:
        `dict`: A dictionary containing the master coefficients for
        the requested spin and purification combinations. The dictionary
        contains the following keys:
            - ``'00'``: The 0-0 coupling coefficients, or ``None``.
            - ``'0s'``: The 0-s coupling coefficients, or ``None``.
            - ``'pp'``: The +/+ coupling coefficients, or ``None``.
            - ``'mm'``: The -/- coupling coefficients, or ``None``.
            - ``'pure_any'``: The value of the ``pure_any`` argument.
            - ``'toeplitz'``: the Toeplitz approximation parameters.
            - ``'spins'``: A tuple containing the spins of the two fields.
            - ``'lmax'``: The maximum multipole considered.
            - ``'lmax_mask'``: The maximum multipole of the mask power spectra.
        Note that, if ``pure_any`` is ``True``, the 0-s and +/+ coupling
        coefficients will contain two and three sets of coefficients,
        respectively. The first set corresponds to the standard MASTER
        coefficients without purification. The second set corresponds
        to the coefficients when one of the fields is purified. The third set
        corresponds to the coefficients when both fields are purified.
    """
    # Give mask power spectra the right shape
    pcl_mask = np.atleast_2d(pcl_mask)
    nmask, nls_mask = pcl_mask.shape
    ls = np.arange(nls_mask)
    pcl_mask = pcl_mask * ((2*ls + 1) / (4 * np.pi))[None, :]
    lmax_mask = nls_mask-1
    nls = lmax+1

    # Sanity checks on input flags
    if is_teb and not (spin1 == 0 and spin2 != 0):
        raise ValueError("is_teb is only valid for spin1=0 and spin2!=0")
    if pure_any and not (spin1 == 2 or spin2 == 2):
        raise ValueError("pure_any is only valid for spin 2")

    # Decide what we need to calculate
    has_00 = (spin1 == 0 and spin2 == 0) or is_teb
    has_0s = (((spin1 != spin2) and
              (spin1 == 0 or spin2 == 0)) or is_teb)
    has_ss = (spin1 != 0 and spin2 != 0) or is_teb
    if pure_any:
        npure_0s, npure_ss = 2, 3
    else:
        npure_0s, npure_ss = 1, 1

    # Calculate output size
    n_xis = has_00 * 1 + has_0s * npure_0s + 2 * has_ss * npure_ss
    size = n_xis * nls**2 * nmask

    # Calculate the coefficients
    d = lib.get_xis(int(lmax), int(lmax_mask), pcl_mask,
                    int(spin1), int(spin2), int(pure_any),
                    int(is_teb), int(l_toeplitz),
                    int(l_exact), int(dl_band), size)
    d = d.reshape((n_xis, nmask, nls, nls))
    if nmask == 1:
        d = d.reshape((n_xis, nls, nls))

    # Place everything in a dictionary
    xi_dict = {}
    xi_dict['00'] = d[0] if has_00 else None
    start_0s = has_00 * 1
    end_0s = start_0s + has_0s * npure_0s
    xi_dict['0s'] = d[start_0s:end_0s] if has_0s else None
    start_pp = end_0s
    end_pp = start_pp + has_ss * npure_ss
    xi_dict['pp'] = d[start_pp:end_pp] if has_ss else None
    start_mm = end_pp
    end_mm = start_mm + has_ss * npure_ss
    xi_dict['mm'] = d[start_mm:end_mm] if has_ss else None
    xi_dict['pure_any'] = pure_any
    xi_dict['toeplitz'] = {'l_toeplitz': l_toeplitz,
                           'l_exact': l_exact,
                           'dl_band': dl_band}
    xi_dict['spins'] = (spin1, spin2)
    xi_dict['lmax'] = lmax
    xi_dict['lmax_mask'] = lmax_mask
    return xi_dict


def get_general_coupling_matrix(pcl_mask, s1, s2, n1, n2,
                                parity="all"):
    """ Returns a general mode-coupling matrix of the form

    .. math::
      M_{\\ell \\ell'}=\\sum_{\\ell''}
      \\frac{(2\\ell'+1)(2\\ell''+1)}{4\\pi}
      \\tilde{C}^{uv}_{\\ell''}\\,
      P_{\\ell+\\ell'+\\ell''}\\,
      \\left(\\begin{array}{ccc}
      \\ell & \\ell' & \\ell'' \\\\
      n_1 & -s_1 & s_1-n_1
      \\end{array}\\right)
      \\left(\\begin{array}{ccc}
      \\ell & \\ell' & \\ell'' \\\\
      n_2 & -s_2 & s_2-n_2
      \\end{array}\\right)

    Where :math:`P_L=1` if ``parity="all"``,
    :math:`P_L=(1+(-1)^L)/2` if ``parity="even"``,
    and :math:`P_L=(1-(-1)^L)/2` if ``parity="odd"``.

    Args:
        pcl_mask (`array`): 1D array containing the power spectrum
          of the masks :math:`\\tilde{C}_\\ell^{uw}`.
        s1 (:obj:`int`): spin index :math:`s_1` above.
        s2 (:obj:`int`): spin index :math:`s_2` above.
        n1 (:obj:`int`): spin index :math:`n_1` above.
        n2 (:obj:`int`): spin index :math:`n_2` above.
        parity (:obj:`str`): One of ``"all"``, ``"even"``,
            ``"odd"``. of ``both``. Determines the parity
            of the mode-coupling matrix. If ``both``,
            both even and odd matrices are returned in a
            3D array of shape ``[2, nl, nl]``,

    Returns:
        (`array`): 2D array of shape ``[nl, nl]``, where ``nl`` is
        the size of ``pcl_mask``, containing the mode-coupling
        matrix for multipoles from 0 to ``nl-1``. If ``parity`` is
        ``"both"``, a 3D array of shape ``[2, nl, nl]`` is returned,
        where the the even and odd matrices are returned in the first
        and second indices, respectively.
    """

    if parity == 'all':
        par = 0
    elif parity == 'even':
        par = 1
    elif parity == 'odd':
        par = -1
    elif parity == 'both':
        par = 2
    else:
        raise ValueError("`parity` must be \"all\", "
                         "\"even\", \"odd\", or \"both\".")
    lmax = len(pcl_mask)-1
    size = (lmax+1)**2
    if par == 2:
        size *= 2
    xi = lib.comp_general_coupling_matrix(
        int(s1), int(s2), int(n1),
        int(n2), int(par), int(lmax),
        pcl_mask, int(size))
    if par == 2:
        xi = xi.reshape([2, lmax+1, lmax+1])
    else:
        xi = xi.reshape([lmax+1, lmax+1])
    return xi
