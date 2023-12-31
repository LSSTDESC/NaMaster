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
