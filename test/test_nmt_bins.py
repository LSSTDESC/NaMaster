from unittest import TestCase
import numpy as np
import pymaster as nmt

class TestBinsSph(TestCase) :
    def norm(self,v1,v2) :
        return np.amax(np.fabs(v1-v2))
    
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

    def test_bins_binning(self) :
        #Tests C_l binning and unbinning
        cls=np.arange(self.lmax+1,dtype=float)
        cl_b=self.bv.bin_cell(cls)
        cl_u=self.bv.unbin_cell(cl_b)
        cl_b_p=np.mean(cls[2:2+self.nlb*((self.lmax-2)//self.nlb)].reshape([-1,self.nlb]),axis=1)
        self.assertTrue(self.norm(cl_b_p,cl_b)<1E-5)
        cl_u_p=(cl_b[:,None]*np.ones([len(cl_b),self.nlb])).flatten()
        self.assertTrue(self.norm(cl_u_p,cl_u[2:2+self.nlb*((self.lmax-2)//self.nlb)])<1E-5)
        
    def test_bins_errors(self) :
        #Tests raised exceptions
        print("\nIgnore all of these error messages:\n_________")
        ells=np.arange(self.lmax-4,dtype=int)+2
        bpws=(ells-2)//4
        weights=0.25*np.ones(self.lmax-4)
        weights[16:20]=0
        try:
            b=nmt.NmtBin(self.nside,bpws=bpws,ells=ells,weights=weights,lmax=self.lmax)
        except:
            pass
        try:
            self.bv.bin_cell(self,np.random.randn(3,3,3))
        except:
            pass
        try:
            self.bv.unbin_cell(self,np.random.randn(3,3,3))
        except:
            pass
        print("_________")

    def test_bins_constant(self) :
        #Tests constant bandpower initialization
        self.assertEqual(self.bc.get_n_bands(),(self.lmax-2)//self.nlb)
        self.assertEqual(self.bc.get_ell_list(5)[2],2+self.nlb*5+2)

    def test_bins_variable(self) :
        #Tests variable bandpower initialization
        self.assertEqual(self.bv.get_n_bands(),(self.lmax-2)//self.nlb)
        self.assertEqual(self.bv.get_n_bands(),self.bc.get_n_bands())
        for i in range(self.bv.get_n_bands()) :
            self.assertTrue((self.bv.get_ell_list(i)==self.bc.get_ell_list(i)).all())
            self.assertTrue(np.fabs(np.sum(self.bv.get_weight_list(i))-1.)<1E-5)
        self.assertTrue(self.norm(self.bv.get_effective_ells(),
                                  (2+self.nlb*np.arange(self.bv.get_n_bands())+0.5*(self.nlb-1)))<1E-5)

if __name__ == '__main__':
    unittest.main()
