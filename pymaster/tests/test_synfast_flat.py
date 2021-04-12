import pytest
import numpy as np
import pymaster as nmt


class SynfastTesterFlat(object):
    def __init__(self):
        self.nx = 141
        self.ny = 311
        self.lx = np.radians(1.)
        self.ly = np.radians(1.)
        self.nbpw = 30
        self.lmax = np.sqrt((self.nx*np.pi/self.lx)**2 +
                            (self.ny*np.pi/self.ly)**2)
        self.lpivot = self.lmax/6.
        self.alpha_pivot = 1.
        self.larr = np.arange(int(self.lmax)+1)
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

    def anafast(self, mps, spin):
        if mps.ndim == 2:
            scalar_input = True
        else:
            scalar_input = False

        k_x = np.fft.rfftfreq(self.nx, self.lx/(2*np.pi*self.nx))
        k_y = np.fft.fftfreq(self.ny, self.ly/(2*np.pi*self.ny))
        k_mod = np.sqrt(k_x[None, :]**2+k_y[:, None]**2)
        dkvol = (2*np.pi)**2/(self.lx*self.ly)
        fft_norm = self.lx*self.ly/(2*np.pi*self.nx*self.ny)

        krange = [0, np.amax(k_mod)]
        kbins = max(self.nx, self.ny)//8
        nk, bk = np.histogram(k_mod.flatten(), range=krange,
                              bins=kbins)
        kk, bk = np.histogram(k_mod.flatten(), range=krange,
                              bins=kbins, weights=k_mod.flatten())
        kmean = kk/nk

        def compute_cl_single(alm1, alm2):
            almabs2 = (np.real(alm1)*np.real(alm2) +
                       np.imag(alm1)*np.imag(alm2)).flatten()
            pk, bk = np.histogram(k_mod.flatten(), range=krange,
                                  bins=kbins, weights=almabs2)
            return pk/nk

        if scalar_input:
            alms = np.fft.rfftn(mps)*fft_norm
            cls = compute_cl_single(alms, alms)
        else:
            alms_tqu = np.array([np.fft.rfftn(m)*fft_norm
                                 for m in mps])

            k_mod[0, 0] = 1E-16
            cosk = k_x[None, :]/k_mod
            cosk[0, 0] = 1.
            sink = k_y[:, None]/k_mod
            sink[0, 0] = 0.
            k_mod[0, 0] = 0
            if spin == 1:
                cos2k = cosk
                sin2k = sink
            else:
                cos2k = cosk**2-sink**2
                sin2k = 2*sink*cosk
            a_t = alms_tqu[0, :, :]
            a_e = cos2k*alms_tqu[1, :, :]-sin2k*alms_tqu[2, :, :]
            a_b = sin2k*alms_tqu[1, :, :]+cos2k*alms_tqu[2, :, :]
            cls = []
            cls.append(compute_cl_single(a_t, a_t))
            cls.append(compute_cl_single(a_e, a_e))
            cls.append(compute_cl_single(a_b, a_b))
            cls.append(compute_cl_single(a_t, a_e))
            cls.append(compute_cl_single(a_e, a_b))
            cls.append(compute_cl_single(a_t, a_b))
            cls = np.array(cls)

        return kmean, nk, cls*dkvol

    def synfast_flat_stats(self, spin):
        # Temperature only
        m_t = nmt.synfast_flat(self.nx, self.ny, self.lx, self.ly,
                               self.cl1, [0], beam=np.array([self.beam]),
                               seed=1234)[0]
        # Polarization
        m_p1 = nmt.synfast_flat(self.nx, self.ny, self.lx, self.ly,
                                self.cl12, [0, spin],
                                beam=np.array([self.beam, self.beam]),
                                seed=1234)
        km, nk, ctt1 = self.anafast(m_t, 0)
        km, nk, [ctt2, cee2, cbb2, cte2, ceb2, ctb2] = self.anafast(m_p1, spin)
        lint = km.astype(int)

        def get_diff(c_d, c_t, c11, c22, c12, nmodes, facsig=5):
            diff = np.fabs(c_d-c_t[lint])  # Residuals
            sig = np.sqrt((c11[lint]*c22[lint]+c12[lint]**2)/nmodes)
            return diff < facsig*sig

        # Check TT
        assert (get_diff(ctt1, self.cltt, self.cltt,
                         self.cltt, self.cltt, nk).all())
        assert (get_diff(ctt2, self.cltt, self.cltt,
                         self.cltt, self.cltt, nk).all())
        # Check EE
        assert (get_diff(cee2, self.clee, self.clee,
                         self.clee, self.clee, nk).all())
        # Check BB
        assert (get_diff(cbb2, self.clbb, self.clbb,
                         self.clbb, self.clbb, nk).all())
        # Check TE
        assert (get_diff(cte2, self.clte, self.cltt,
                         self.clee, self.clte, nk).all())
        # Check EB
        assert (get_diff(ceb2, self.cleb, self.clbb,
                         self.clee, self.cleb, nk).all())
        # Check TB
        assert (get_diff(ctb2, self.cltb, self.cltt,
                         self.clbb, self.cltb, nk).all())


ST = SynfastTesterFlat()


def test_synfast_flat_errors():
    with pytest.raises(ValueError):  # Negative spin
        nmt.synfast_flat(ST.nx, ST.ny, ST.lx, ST.ly,
                         ST.cl1, [-1], beam=ST.beam, seed=1234)
    with pytest.raises(ValueError):  # Not enough power spectra
        nmt.synfast_flat(ST.nx, ST.ny, ST.lx, ST.ly,
                         ST.cl2, [0, 2],
                         beam=np.array([ST.beam, ST.beam]),
                         seed=1234)
    with pytest.raises(ValueError):  # Not enough beams
        nmt.synfast_flat(ST.nx, ST.ny, ST.lx, ST.ly,
                         ST.cl12, [0, 2],
                         beam=np.array([ST.beam]), seed=1234)
    with pytest.raises(ValueError):  # Inconsistent beam
        nmt.synfast_flat(ST.nx, ST.ny, ST.lx, ST.ly,
                         ST.cl12, [0, 2],
                         beam=np.array([ST.beam[:15],
                                        ST.beam[:15]]),
                         seed=1234)
    with pytest.raises(RuntimeError):  # Negative dimensions
        nmt.synfast_flat(ST.nx, ST.ny, -ST.lx, ST.ly,
                         ST.cl2, [2],
                         beam=np.array([ST.beam]), seed=1234)
    m = nmt.synfast_flat(ST.nx, ST.ny, ST.lx, ST.ly,
                         ST.cl12, [0, 2],
                         beam=None, seed=-1)
    assert m.shape == (3, ST.ny, ST.nx)


@pytest.mark.parametrize('spin', [1, 2])
def test_synfast_flat_spin(spin):
    ST.synfast_flat_stats(spin)
