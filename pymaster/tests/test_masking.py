import pytest
import numpy as np
import pymaster as nmt
import healpy as hp


class MaskingTester(object):
    def __init__(self):
        self.nside = 256
        self.th0 = np.pi/4
        self.msk = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.th, ph = hp.pix2ang(self.nside,
                                 np.arange(hp.nside2npix(self.nside),
                                           dtype=int))
        self.msk[self.th < self.th0] = 1.
        self.aposize = 2.
        self.inv_x2thr = 1./(1-np.cos(np.radians(self.aposize)))


MT = MaskingTester()


def test_mask_errors():
    with pytest.raises(RuntimeError):  # Badly shaped input
        nmt.mask_apodization(MT.msk[:13], MT.aposize, apotype="C1")
    with pytest.raises(RuntimeError):  # Negative apodization
        nmt.mask_apodization(MT.msk, -MT.aposize, apotype="C1")
    with pytest.raises(ValueError):  # Wrong apodization type
        nmt.mask_apodization(MT.msk, MT.aposize, apotype="C3")
    with pytest.raises(RuntimeError):  # Aposize too small
        nmt.mask_apodization(MT.msk[:12*2**2], 1., apotype='C1')
    with pytest.raises(RuntimeError):  # Aposize too small
        nmt.mask_apodization(MT.msk[:12*2**2], 1., apotype='Smooth')


def test_mask_c1():
    msk_apo = nmt.mask_apodization(MT.msk, MT.aposize,
                                   apotype="C1")
    # Below transition
    assert (msk_apo[MT.th > MT.th0] < 1E-10).all()
    # Above transition
    assert (np.fabs(msk_apo[MT.th <
                            MT.th0 -
                            np.radians(MT.aposize)]-1.) < 1E-10).all()
    # Within transition
    x = np.sqrt((1-np.cos(MT.th-MT.th0))*MT.inv_x2thr)
    f = x-np.sin(x*2*np.pi)/(2*np.pi)
    ind_transition = ((MT.th < MT.th0) &
                      (MT.th > MT.th0 -
                       np.radians(MT.aposize)))
    assert (np.fabs(msk_apo[ind_transition] -
                    f[ind_transition]) < 2E-2).all()


def test_mask_c2():
    msk_apo = nmt.mask_apodization(MT.msk, MT.aposize,
                                   apotype="C2")
    # Below transition
    assert (msk_apo[MT.th > MT.th0] < 1E-10).all()
    # Above transition
    assert (np.fabs(msk_apo[MT.th < MT.th0 -
                            np.radians(MT.aposize)]-1.) < 1E-10).all()
    # Within transition
    x = np.sqrt((1-np.cos(MT.th-MT.th0))*MT.inv_x2thr)
    f = 0.5*(1-np.cos(x*np.pi))
    ind_transition = ((MT.th < MT.th0) &
                      (MT.th > MT.th0 -
                       np.radians(MT.aposize)))
    assert (np.fabs(msk_apo[ind_transition] -
                    f[ind_transition]) < 2E-2).all()


def test_mask_smooth():
    # Here there's no analytical expression, so we just do some basic
    # sanity checks
    msk_apo = nmt.mask_apodization(MT.msk, MT.aposize,
                                   apotype="Smooth")
    # We've removed area
    assert np.mean(msk_apo) < np.mean(MT.msk)
    # Mask is positive or zero
    assert np.all(msk_apo >= 0)
    # We haven't suppressed it too much
    assert np.fabs(np.amax(msk_apo)-1) < 1E-3
    # All masked pixels are still masked
    assert np.all(msk_apo[MT.msk == 0] == 0)
