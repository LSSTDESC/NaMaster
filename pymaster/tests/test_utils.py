import numpy as np
import pytest
import pymaster as nmt


def test_params_set_get():
    # SHT calculator
    bak = nmt.get_default_params()['sht_calculator']
    nmt.set_sht_calculator('healpy')
    assert nmt.get_default_params()['sht_calculator'] == 'healpy'
    nmt.set_sht_calculator(bak)

    # n_iter
    bak = nmt.get_default_params()['n_iter_default']
    nmt.set_n_iter_default(5)
    assert nmt.get_default_params()['n_iter_default'] == 5
    nmt.set_n_iter_default(bak)

    # n_iter_mask
    bak = nmt.get_default_params()['n_iter_mask_default']
    nmt.set_n_iter_default(5, mask=True)
    assert nmt.get_default_params()['n_iter_mask_default'] == 5
    nmt.set_n_iter_default(bak, mask=True)

    # tol_pinv
    bak = nmt.get_default_params()['tol_pinv_default']
    nmt.set_tol_pinv_default(1E-3)
    assert nmt.get_default_params()['tol_pinv_default'] == 1E-3
    nmt.set_tol_pinv_default(bak)

    # Wrong SHT calculator
    with pytest.raises(KeyError):
        nmt.set_sht_calculator('healpyy')

    # Fake us not having ducc
    nmt.utils.HAVE_DUCC = False
    with pytest.raises(ValueError):
        nmt.set_sht_calculator('ducc')
    nmt.utils.HAVE_DUCC = True
    nmt.set_sht_calculator('ducc')

    # Negative n_iter
    with pytest.raises(ValueError):
        nmt.set_n_iter_default(-1)

    # Wrong tolerance
    with pytest.raises(ValueError):
        nmt.set_tol_pinv_default(2)


def test_moore_penrose_pinv():
    # Unit vector
    v = np.random.rand(3)
    v /= np.sqrt(np.dot(v, v))

    # Matrix with single eigenvalue
    m = np.outer(v, v)
    # Pseudo-inverse
    im = nmt.utils.moore_penrose_pinvh(m, 1E-4)
    # Diagonalise
    w, e = np.linalg.eigh(im)
    # Pick up largest eigval
    e = e[:, w > 0.1].squeeze()
    # Should only have two non-zero eigvals
    assert np.sum(np.fabs(w) < 1E-15) == 2
    # Check e is parallel to v
    assert np.isclose(np.fabs(np.dot(e, v)),
                      np.sqrt(np.dot(e, e)))

    # For invertible matrix, we just get
    # the inverse
    w = np.array([1, 2, 3])
    m = np.diag(w)
    im = nmt.utils.moore_penrose_pinvh(m, None)
    imb = np.diag(1/w)
    assert np.allclose(im, imb)


def test_ducc_catalog2alm():
    import pymaster.utils as ut
    import numpy as np
    import healpy as hp

    nside = 128
    npix = int(hp.nside2npix(nside))
    pixel_area = np.pi*4./npix
    lmax = 128

    # input alms
    alm_in = np.zeros((3, hp.Alm.getidx(lmax, lmax, lmax)+1), dtype="complex")
    alm_in[0, hp.Alm.getidx(lmax, 2, 2)] = 2.
    alm_in[1, hp.Alm.getidx(lmax, 2, 0)] = 1.
    alm_in[2, hp.Alm.getidx(lmax, 3, 1)] = 2.

    map = hp.alm2map(alm_in, nside, lmax=lmax, mmax=lmax)
    th, ph = hp.pix2ang(nside, np.arange(npix))

    # spin 0
    alm_cat_0 = pixel_area*ut._catalog2alm_ducc0(
        map[0], np.array([th, ph]), 0, lmax
    )
    # spin 2
    alm_cat_2 = pixel_area*ut._catalog2alm_ducc0(
        map[1:3], np.array([th, ph]), 2, lmax
    )

    assert np.all(np.absolute(alm_cat_0[0] - alm_in[0]) < 1E-4)
    assert np.all(np.absolute(alm_cat_2[:2] - alm_in[1:3]) < 1E-4)
