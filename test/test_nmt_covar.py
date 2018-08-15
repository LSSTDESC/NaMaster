import unittest 
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
from .testutils import normdiff, read_flat_map

#Unit tests associated with the NmtField and NmtFieldFlat classes

class TestCovarFsk(unittest.TestCase) :
    def setUp(self) :
        #This is to avoid showing an ugly warning that has nothing to do with pymaster
        warnings.simplefilter("ignore", ResourceWarning)

        self.w=nmt.NmtWorkspaceFlat()
        self.w.read_from("test/benchmarks/bm_f_nc_np_w00.dat")
        
        l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("test/benchmarks/cls_lss.txt",unpack=True)
        self.l=l
        self.cltt=cltt+nltt
        
    def test_workspace_covar_flat_benchmark(self) :
        cw=nmt.NmtCovarianceWorkspaceFlat()
        cw.compute_coupling_coefficients(self.w,self.w)

        covar=nmt.gaussian_covariance_flat(cw,self.l,self.cltt,self.cltt,self.cltt,self.cltt)
        covar_bench=np.loadtxt("test/benchmarks/bm_f_nc_np_cov.txt",unpack=True)
        self.assertTrue((np.fabs(covar-covar_bench)<=np.fmin(np.fabs(covar),np.fabs(covar_bench))*1E-5).all())

    def test_workspace_covar_flat_errors(self) :
        cw=nmt.NmtCovarianceWorkspaceFlat()

        with self.assertRaises(ValueError) : #Write uninitialized
            cw.write_to("wsp.dat");
        
        cw.compute_coupling_coefficients(self.w,self.w)  #All good
        self.assertEqual(cw.wsp.bin.n_bands,self.w.wsp.bin.n_bands)

        with self.assertRaises(RuntimeError) : #Write uninitialized
            cw.write_to("tests/wsp.dat");

        cw.read_from('test/benchmarks/bm_f_nc_np_cw00.dat') #Correct reading
        self.assertEqual(cw.wsp.bin.n_bands,self.w.wsp.bin.n_bands)

        #gaussian_covariance
        with self.assertRaises(ValueError) : #Wrong input power spectra
            nmt.gaussian_covariance_flat(cw,self.l,self.cltt,self.cltt,self.cltt,self.cltt[:15])

        with self.assertRaises(RuntimeError) : #Incorrect reading
            cw.read_from('none')
        w2=nmt.NmtWorkspaceFlat()
        w2.read_from("test/benchmarks/bm_f_nc_np_w00.dat")
        w2.wsp.fs.nx=self.w.wsp.fs.nx//2
        with self.assertRaises(ValueError) : #Incompatible resolutions
            cw.compute_coupling_coefficients(self.w,w2)
        w2.wsp.fs.nx=self.w.wsp.fs.nx
        w2.wsp.bin.n_bands=self.w.wsp.bin.n_bands//2
        with self.assertRaises(RuntimeError) : #Incompatible resolutions
            cw.compute_coupling_coefficients(self.w,w2)
        w2.wsp.bin.n_bands=self.w.wsp.bin.n_bands

        w2.read_from("test/benchmarks/bm_f_nc_np_w02.dat")
        with self.assertRaises(ValueError) : #Spin-2
            cw.compute_coupling_coefficients(self.w,w2)
        
class TestCovarSph(unittest.TestCase) :
    def setUp(self) :
        #This is to avoid showing an ugly warning that has nothing to do with pymaster
        warnings.simplefilter("ignore", ResourceWarning)

        self.w=nmt.NmtWorkspace()
        self.w.read_from("test/benchmarks/bm_nc_np_w00.dat")
        self.nside=self.w.wsp.nside
        
        l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("test/benchmarks/cls_lss.txt",unpack=True)
        self.l=l[:3*self.nside]
        self.cltt=cltt[:3*self.nside]+nltt[:3*self.nside]
                                                                                
    def test_workspace_covar_benchmark(self) :
        cw=nmt.NmtCovarianceWorkspace()
        cw.compute_coupling_coefficients(self.w,self.w)

        covar=nmt.gaussian_covariance(cw,self.cltt,self.cltt,self.cltt,self.cltt)
        covar_bench=np.loadtxt("test/benchmarks/bm_nc_np_cov.txt",unpack=True)
        self.assertTrue((np.fabs(covar-covar_bench)<=np.fmin(np.fabs(covar),np.fabs(covar_bench))*1E-5).all())
                    
    def test_workspace_covar_errors(self) :
        cw=nmt.NmtCovarianceWorkspace()

        with self.assertRaises(ValueError) : #Write uninitialized
            cw.write_to("wsp.dat");
            
        cw.compute_coupling_coefficients(self.w,self.w) #All good
        self.assertEqual(cw.wsp.nside,self.w.wsp.nside)
        self.assertEqual(cw.wsp.lmax_a,self.w.wsp.lmax)
        self.assertEqual(cw.wsp.lmax_b,self.w.wsp.lmax)

        with self.assertRaises(RuntimeError) : #Write uninitialized
            cw.write_to("tests/wsp.dat");

        cw.read_from('test/benchmarks/bm_nc_np_cw00.dat') #Correct reading
        self.assertEqual(cw.wsp.nside,self.w.wsp.nside)
        self.assertEqual(cw.wsp.lmax_a,self.w.wsp.lmax)
        self.assertEqual(cw.wsp.lmax_b,self.w.wsp.lmax)

        #gaussian_covariance
        with self.assertRaises(ValueError) : #Wrong input power spectra
            nmt.gaussian_covariance(cw,self.cltt,self.cltt,self.cltt,self.cltt[:15])
        
        with self.assertRaises(RuntimeError) : #Incorrect reading
            cw.read_from('none')
        w2=nmt.NmtWorkspace()
        w2.read_from("test/benchmarks/bm_nc_np_w00.dat")
        w2.wsp.nside=self.w.wsp.nside//2
        with self.assertRaises(ValueError) : #Incompatible resolutions
            cw.compute_coupling_coefficients(self.w,w2)
        w2.wsp.nside=self.w.wsp.nside
        w2.wsp.lmax=self.w.wsp.lmax//2
        with self.assertRaises(RuntimeError) : #Incompatible resolutions
            cw.compute_coupling_coefficients(self.w,w2)
        w2.wsp.lmax=self.w.wsp.lmax

        w2.read_from("test/benchmarks/bm_nc_np_w02.dat")
        with self.assertRaises(ValueError) : #Spin-2
            cw.compute_coupling_coefficients(self.w,w2)
        
if __name__ == '__main__':
    unittest.main()
