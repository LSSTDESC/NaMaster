import pytest
import numpy as np
import pymaster as nmt


class MaskingTesterFlat(object):
    def __init__(self):
        self.nx = self.ny = 200
        self.lx = self.ly = np.radians(10.)
        self.msk = np.zeros([self.ny, self.nx])
        self.msk[:self.ny//2, :] = 1.
        self.aposize = 1.
        self.inv_xthr = 1./np.radians(self.aposize)
        self.ioff = self.ny//2-int(np.radians(self.aposize)/(self.ly/self.ny))


MT = MaskingTesterFlat()


def test_mask_flat_errors():
    with pytest.raises(ValueError):  # Badly shaped input
        nmt.mask_apodization_flat(MT.msk[0], MT.lx,
                                  MT.ly, MT.aposize,
                                  apotype="C1")
    with pytest.raises(RuntimeError):  # Negative apodization
        nmt.mask_apodization_flat(MT.msk, MT.lx, MT.ly,
                                  -MT.aposize, apotype="C1")
    with pytest.raises(RuntimeError):  # Wrong apodization type
        nmt.mask_apodization_flat(MT.msk, MT.lx, MT.ly,
                                  MT.aposize, apotype="C3")


def test_mask_flat_c1():
    msk_apo = nmt.mask_apodization_flat(MT.msk, MT.lx, MT.ly,
                                        MT.aposize, apotype="C1")
    # Below transition
    assert (msk_apo[MT.ny//2:, :] < 1E-10).all()
    # Above transition
    assert (np.fabs(msk_apo[:MT.ioff, :]-1.) < 1E-10).all()
    # Within transition
    ind_transition = np.arange(MT.ioff, MT.ny//2, dtype=int)
    x = MT.inv_xthr*np.fabs((MT.ny/2.-ind_transition)*MT.ly/MT.ny)
    f = x-np.sin(x*2*np.pi)/(2*np.pi)
    assert (np.fabs(msk_apo[ind_transition, :] - f[:, None])
            < 1E-10).all()


def test_mask_flat_c2():
    msk_apo = nmt.mask_apodization_flat(MT.msk, MT.lx,
                                        MT.ly, MT.aposize,
                                        apotype="C2")
    # Below transition
    assert (msk_apo[MT.ny//2:, :] < 1E-10).all()
    # Above transition
    assert (np.fabs(msk_apo[:MT.ioff, :]-1.) < 1E-10).all()
    # Within transition
    ind_transition = np.arange(MT.ioff, MT.ny//2, dtype=int)
    x = MT.inv_xthr*np.fabs((MT.ny/2.-ind_transition)*MT.ly/MT.ny)
    f = 0.5*(1-np.cos(x*np.pi))
    assert (np.fabs(msk_apo[ind_transition, :] -
                    f[:, None]) < 1E-10).all()
