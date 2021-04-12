import pytest
import numpy as np
import pymaster as nmt
import healpy as hp


class SynfastTester(object):
    def __init__(self):
        self.nside = 128
        self.lmax = 3*self.nside-1
        self.larr = np.arange(int(self.lmax+1))
        self.lpivot = self.nside*0.5
        self.alpha_pivot = 1.
        self.cltt = (2*self.lpivot/(self.larr+self.lpivot))**self.alpha_pivot
        self.clee = self.cltt.copy()
        self.clbb = self.cltt.copy()
        self.clte = np.zeros_like(self.cltt)
        self.cleb = np.zeros_like(self.cltt)
        self.cltb = np.zeros_like(self.cltt)
        self.cl1 = np.array([self.cltt])
        self.cl2 = np.array([self.clee, self.cleb,
                             self.clbb])
        self.cl12 = np.array([self.cltt, self.clte, self.cltb,
                              self.clee, self.cleb,
                              self.clbb])
        self.cl22 = np.array([self.clee, self.cleb, self.cleb, self.cleb,
                              self.clbb, self.cleb, self.cleb,
                              self.clee, self.cleb,
                              self.clbb])
        self.beam = np.ones_like(self.cltt)

    def anafast(self, mps1, spin1, mps2=None, spin2=None):
        msk = np.ones(hp.nside2npix(self.nside))
        f1 = nmt.NmtField(msk, mps1, spin=spin1, n_iter=0)
        if mps2 is None:
            f2 = f1
        else:
            f2 = nmt.NmtField(msk, mps2, spin=spin2, n_iter=0)
        return nmt.compute_coupled_cell(f1, f2)

    def synfast_stats(self, spin):
        # Temperature only
        m_t = nmt.synfast_spherical(self.nside, self.cl1, [0],
                                    beam=np.array([self.beam]),
                                    seed=1234)
        # Polarization
        m_p1 = nmt.synfast_spherical(self.nside, self.cl12, [0, spin],
                                     beam=np.array([self.beam, self.beam]),
                                     seed=1234)
        ctt1 = self.anafast(m_t, 0)[0]
        ctt2 = self.anafast([m_p1[0]], 0)[0]
        cte2, ctb2 = self.anafast([m_p1[0]], 0,
                                  m_p1[1:], spin)
        cee2, ceb2, _, cbb2 = self.anafast(m_p1[1:], spin,
                                           m_p1[1:], spin)

        def get_diff(c_d, c_t, c11, c22, c12, facsig=5):
            diff = np.fabs(c_d-c_t)  # Residuals
            # 1-sigma expected errors
            sig = np.sqrt((c11*c22+c12**2)/(2*self.larr+1.))
            return diff[2*self.nside] < facsig*sig[2*self.nside]

        # Check TT
        assert (get_diff(ctt1, self.cltt, self.cltt,
                         self.cltt, self.cltt).all())
        assert (get_diff(ctt2, self.cltt, self.cltt,
                         self.cltt, self.cltt).all())
        # Check EE
        assert (get_diff(cee2, self.clee, self.clee,
                         self.clee, self.clee).all())
        # Check BB
        assert (get_diff(cbb2, self.clbb, self.clbb,
                         self.clbb, self.clbb).all())
        # Check TE
        assert (get_diff(cte2, self.clte, self.cltt,
                         self.clee, self.clte).all())
        # Check EB
        assert (get_diff(ceb2, self.cleb, self.clbb,
                         self.clee, self.cleb).all())
        # Check TB
        assert (get_diff(ctb2, self.cltb, self.clbb,
                         self.cltt, self.cltb).all())


ST = SynfastTester()


def test_synfast_errors():
    with pytest.raises(ValueError):  # Negative spin
        nmt.synfast_spherical(ST.nside, ST.cl1, [-1], seed=1234)
    with pytest.raises(ValueError):  # Not enough power spectra
        nmt.synfast_spherical(ST.nside, ST.cl2, [0, 2], seed=1234)
    with pytest.raises(ValueError):  # Not enough beams
        nmt.synfast_spherical(ST.nside, ST.cl12, [0, 2],
                              beam=np.array([ST.beam]), seed=1234)
    with pytest.raises(ValueError):  # Inconsistent beam
        nmt.synfast_spherical(ST.nside, ST.cl12, [0, 2],
                              beam=np.array([ST.beam[:15],
                                             ST.beam[:15]]),
                              seed=1234)
    m = nmt.synfast_spherical(ST.nside, ST.cl12, [0, 2],
                              beam=None, seed=-1)
    assert m.shape == (3, hp.nside2npix(ST.nside))


@pytest.mark.parametrize('spin', [1, 2])
def test_synfast_spin(spin):
    ST.synfast_stats(spin)
