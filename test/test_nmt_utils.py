import unittest
import numpy as np
import pymaster as nmt
import healpy as hp


class TestUtilsMaskSph(unittest.TestCase):
    def setUp(self):
        self.nside = 256
        self.th0 = np.pi/4
        self.msk = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.th, ph = hp.pix2ang(self.nside,
                                 np.arange(hp.nside2npix(self.nside),
                                           dtype=int))
        self.msk[self.th < self.th0] = 1.
        self.aposize = 2.
        self.inv_x2thr = 1./(1-np.cos(np.radians(self.aposize)))

    def test_mask_errors(self):
        with self.assertRaises(RuntimeError):  # Badly shaped input
            nmt.mask_apodization(self.msk[:13], self.aposize, apotype="C1")
        with self.assertRaises(RuntimeError):  # Negative apodization
            nmt.mask_apodization(self.msk, -self.aposize, apotype="C1")
        with self.assertRaises(RuntimeError):  # Wrong apodization type
            nmt.mask_apodization(self.msk, self.aposize, apotype="C3")
        with self.assertRaises(RuntimeError):  # Aposize too small
            nmt.mask_apodization(self.msk[:12*2**2], 1., apotype='C1')
        with self.assertRaises(RuntimeError):
            nmt.mask_apodization(self.msk[:12*2**2], 1., apotype='Smooth')

    def test_mask_c1(self):
        msk_apo = nmt.mask_apodization(self.msk, self.aposize,
                                       apotype="C1")
        # Below transition
        self.assertTrue((msk_apo[self.th > self.th0] < 1E-10).all())
        # Above transition
        self.assertTrue((np.fabs(msk_apo[self.th <
                                         self.th0 -
                                         np.radians(self.aposize)]-1.) <
                         1E-10).all())
        # Within transition
        x = np.sqrt((1-np.cos(self.th-self.th0))*self.inv_x2thr)
        f = x-np.sin(x*2*np.pi)/(2*np.pi)
        ind_transition = ((self.th < self.th0) &
                          (self.th > self.th0 -
                           np.radians(self.aposize)))
        self.assertTrue((np.fabs(msk_apo[ind_transition] -
                                 f[ind_transition]) < 2E-2).all())

    def test_mask_c2(self):
        msk_apo = nmt.mask_apodization(self.msk, self.aposize,
                                       apotype="C2")
        # Below transition
        self.assertTrue((msk_apo[self.th > self.th0] < 1E-10).all())
        # Above transition
        self.assertTrue((np.fabs(msk_apo[self.th < self.th0 -
                                         np.radians(self.aposize)]-1.) <
                         1E-10).all())
        # Within transition
        x = np.sqrt((1-np.cos(self.th-self.th0))*self.inv_x2thr)
        f = 0.5*(1-np.cos(x*np.pi))
        ind_transition = ((self.th < self.th0) &
                          (self.th > self.th0 -
                           np.radians(self.aposize)))
        self.assertTrue((np.fabs(msk_apo[ind_transition] -
                                 f[ind_transition]) < 2E-2).all())


class TestUtilsMaskFsk(unittest.TestCase):
    def setUp(self):
        self.nx = self.ny = 200
        self.lx = self.ly = np.radians(10.)
        self.msk = np.zeros([self.ny, self.nx])
        self.msk[:self.ny//2, :] = 1.
        self.aposize = 1.
        self.inv_xthr = 1./np.radians(self.aposize)
        self.ioff = self.ny//2-int(np.radians(self.aposize)/(self.ly/self.ny))

    def test_mask_flat_errors(self):
        with self.assertRaises(ValueError):  # Badly shaped input
            nmt.mask_apodization_flat(self.msk[0], self.lx,
                                      self.ly, self.aposize,
                                      apotype="C1")
        with self.assertRaises(RuntimeError):  # Negative apodization
            nmt.mask_apodization_flat(self.msk, self.lx, self.ly,
                                      -self.aposize, apotype="C1")
        with self.assertRaises(RuntimeError):  # Wrong apodization type
            nmt.mask_apodization_flat(self.msk, self.lx, self.ly,
                                      self.aposize, apotype="C3")

    def test_mask_flat_c1(self):
        msk_apo = nmt.mask_apodization_flat(self.msk, self.lx, self.ly,
                                            self.aposize, apotype="C1")
        # Below transition
        self.assertTrue((msk_apo[self.ny//2:, :] < 1E-10).all())
        # Above transition
        self.assertTrue((np.fabs(msk_apo[:self.ioff, :]-1.) <
                         1E-10).all())
        # Within transition
        ind_transition = np.arange(self.ioff, self.ny//2, dtype=int)
        x = self.inv_xthr*np.fabs((self.ny/2.-ind_transition)*self.ly/self.ny)
        f = x-np.sin(x*2*np.pi)/(2*np.pi)
        self.assertTrue((np.fabs(msk_apo[ind_transition, :] -
                                 f[:, None]) <
                         1E-10).all())

    def test_mask_flat_c2(self):
        msk_apo = nmt.mask_apodization_flat(self.msk, self.lx,
                                            self.ly, self.aposize,
                                            apotype="C2")
        # Below transition
        self.assertTrue((msk_apo[self.ny//2:, :] < 1E-10).all())
        # Above transition
        self.assertTrue((np.fabs(msk_apo[:self.ioff, :]-1.) < 1E-10).all())
        # Within transition
        ind_transition = np.arange(self.ioff, self.ny//2, dtype=int)
        x = self.inv_xthr*np.fabs((self.ny/2.-ind_transition)*self.ly/self.ny)
        f = 0.5*(1-np.cos(x*np.pi))
        self.assertTrue((np.fabs(msk_apo[ind_transition, :] -
                                 f[:, None]) < 1E-10).all())


if __name__ == '__main__':
    unittest.main()
