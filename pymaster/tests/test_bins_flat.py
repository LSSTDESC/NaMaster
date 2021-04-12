import pytest
import pymaster as nmt
import numpy as np
from .utils import normdiff


class BinTesterFlat(object):
    def __init__(self):
        self.nlb = 5
        self.nbands = 399
        larr = np.arange(self.nbands+1)*self.nlb+2
        self.b = nmt.NmtBinFlat(larr[:-1], larr[1:])


BT = BinTesterFlat()


def test_bins_flat_errors():
    # Tests raised exceptions
    with pytest.raises(ValueError):
        BT.b.bin_cell(np.arange(3), np.random.randn(3, 3, 3))
    with pytest.raises(ValueError):
        BT.b.bin_cell(np.arange(4), np.random.randn(3, 3))
    with pytest.raises(ValueError):
        BT.b.unbin_cell(np.arange(3), np.random.randn(3, 3, 3))
    with pytest.raises(ValueError):
        BT.b.unbin_cell(np.arange(4), np.random.randn(3, 3))


def test_bins_flat_alloc():
    # Tests bandpower properties
    assert (BT.b.get_n_bands() == BT.nbands)
    assert (normdiff((np.arange(BT.nbands)+0.5)*BT.nlb+2,
                     BT.b.get_effective_ells()) < 1E-5)


def test_bins_flat_binning():
    # Tests binning
    cl = np.arange((BT.nbands+2)*BT.nlb)
    cl_b = BT.b.bin_cell(cl, cl)
    cl_u = BT.b.unbin_cell(cl_b, cl)
    assert (normdiff(cl_b, BT.b.get_effective_ells()) < 1E-5)
    i_bin = (cl.astype(int) - 2)//BT.nlb
    igood = (i_bin >= 0) & (i_bin < BT.nbands)
    assert (normdiff(cl_u[igood], cl_b[i_bin[igood]]) < 1E-5)
