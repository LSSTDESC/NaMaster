import unittest
import numpy as np
import pymaster as nmt
from .testutils import normdiff

#Unit tests associated with the NmtBin and NmtBinFlat classes

class TestBinsSph(unittest.TestCase) :
    def setUp(self) :
        #Generates two equivalent binning schemes using two different initialization paths.
        self.nside=1024
        self.lmax=2000
        self.nlb=4
        self.bc=nmt.NmtBin(self.nside,nlb=4,lmax=self.lmax)
        ells=np.arange(self.lmax-4,dtype=int)+2
        bpws=(ells-2)//4
        weights=0.25*np.ones(self.lmax-4)
        self.bv=nmt.NmtBin(self.nside,bpws=bpws,ells=ells,weights=weights,lmax=self.lmax)
        
    def test_bins_errors(self) :
        #Tests raised exceptions
        ells=np.arange(self.lmax-4,dtype=int)+2
        bpws=(ells-2)//4
        weights=0.25*np.ones(self.lmax-4)
        weights[16:20]=0
        with self.assertRaises(RuntimeError):
            b=nmt.NmtBin(self.nside,bpws=bpws,ells=ells,weights=weights,lmax=self.lmax)
        with self.assertRaises(ValueError):
            self.bv.bin_cell(np.random.randn(3,3,3))
        with self.assertRaises(ValueError):
            self.bv.unbin_cell(np.random.randn(3,3,3))

    def test_bins_constant(self) :
        #Tests constant bandpower initialization
        self.assertEqual(self.bc.get_n_bands(),(self.lmax-2)//self.nlb)
        self.assertEqual(self.bc.get_ell_list(5)[2],2+self.nlb*5+2)
        b=nmt.NmtBin(1024,nlb=4,lmax=2000)
        self.assertEqual(b.bin.ell_max,2000)

    def test_bins_variable(self) :
        #Tests variable bandpower initialization
        self.assertEqual(self.bv.get_n_bands(),(self.lmax-2)//self.nlb)
        self.assertEqual(self.bv.get_n_bands(),self.bc.get_n_bands())
        for i in range(self.bv.get_n_bands()) :
            self.assertTrue((self.bv.get_ell_list(i)==self.bc.get_ell_list(i)).all())
            self.assertTrue(np.fabs(np.sum(self.bv.get_weight_list(i))-1.)<1E-5)
        self.assertTrue(normdiff(self.bv.get_effective_ells(),
                                 (2+self.nlb*np.arange(self.bv.get_n_bands())+0.5*(self.nlb-1)))<1E-5)

    def test_bins_binning(self) :
        #Tests C_l binning and unbinning
        cls=np.arange(self.lmax+1,dtype=float)
        cl_b=self.bv.bin_cell(cls)
        cl_u=self.bv.unbin_cell(cl_b)
        cl_b_p=np.mean(cls[2:2+self.nlb*((self.lmax-2)//self.nlb)].reshape([-1,self.nlb]),axis=1)
        self.assertTrue(normdiff(cl_b_p,cl_b)<1E-5)
        cl_u_p=(cl_b[:,None]*np.ones([len(cl_b),self.nlb])).flatten()
        self.assertTrue(normdiff(cl_u_p,cl_u[2:2+self.nlb*((self.lmax-2)//self.nlb)])<1E-5)
        
class TestBinsFsk(unittest.TestCase) :
    def setUp(self) :
        self.nlb=5
        self.nbands=399
        larr=np.arange(self.nbands+1)*self.nlb+2
        self.b=nmt.NmtBinFlat(larr[:-1],larr[1:])

    def test_bins_flat_errors(self) :
        #Tests raised exceptions
        with self.assertRaises(ValueError):
            self.b.bin_cell(np.arange(3),np.random.randn(3,3,3))
        with self.assertRaises(ValueError):
            self.b.bin_cell(np.arange(4),np.random.randn(3,3))
        with self.assertRaises(ValueError):
            self.b.unbin_cell(np.arange(3),np.random.randn(3,3,3))
        with self.assertRaises(ValueError):
            self.b.unbin_cell(np.arange(4),np.random.randn(3,3))

    def test_bins_flat_alloc(self) :
        #Tests bandpower properties
        self.assertTrue(self.b.get_n_bands()==self.nbands)
        self.assertTrue(normdiff((np.arange(self.nbands)+0.5)*self.nlb+2,
                                 self.b.get_effective_ells())<1E-5)

    def test_bins_flat_binning(self) :
        #Tests binning
        cl=np.arange((self.nbands+2)*self.nlb)
        cl_b=self.b.bin_cell(cl,cl)
        cl_u=self.b.unbin_cell(cl_b,cl)
        self.assertTrue(normdiff(cl_b,self.b.get_effective_ells())<1E-5)
        i_bin=(cl.astype(int)-2)//self.nlb
        igood=(i_bin>=0) & (i_bin<self.nbands)
        self.assertTrue(normdiff(cl_u[igood],cl_b[i_bin[igood]])<1E-5)

if __name__ == '__main__':
    unittest.main()
