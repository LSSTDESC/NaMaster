import unittest 
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
from .testutils import normdiff

#Unit tests associated with the NmtField and NmtFieldFlat classes
class TestFieldSph(unittest.TestCase) :
    def setUp(self) :
        #This is to avoid showing an ugly warning that has nothing to do with pymaster
        warnings.simplefilter("ignore", ResourceWarning)

        self.nside=64
        self.lmax=3*self.nside-1
        self.ntemp=5
        self.npix=hp.nside2npix(self.nside)
        self.msk=np.ones(self.npix)
        self.mps=np.zeros([3,self.npix])
        self.tmp=np.zeros([self.ntemp,3,self.npix])
        self.beam=np.ones(self.lmax+1)

        th,ph=hp.pix2ang(self.nside,np.arange(self.npix))
        sth=np.sin(th); cth=np.cos(th)
        self.mps[0]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
        self.mps[1]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
        self.mps[2]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.
        for i in range(self.ntemp) :
            self.tmp[i][0]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
            self.tmp[i][1]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
            self.tmp[i][2]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.

    def test_field_alloc(self) :
        #No templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam)
        f2p=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                         purify_e=True,purify_b=True,n_iter_mask_purify=10)
        self.assertTrue(normdiff(f0.get_maps()[0],self.mps[0]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[0]))>np.mean(np.fabs(f2p.get_maps()[0]-
                                                                                 self.mps[1]*self.msk)))
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[1]))>np.mean(np.fabs(f2p.get_maps()[1]-
                                                                                 self.mps[2]*self.msk)))
        self.assertEqual(len(f0.get_templates()),0)
        self.assertEqual(len(f2.get_templates()),0)
        self.assertEqual(len(f2p.get_templates()),0)

        #With templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],
                        templates=np.array([[t[0]] for t in self.tmp]),
                        beam=self.beam)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],
                        templates=np.array([[t[1],t[2]] for t in self.tmp]),
                        beam=self.beam)
        #Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],0*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),5)
        self.assertEqual(len(f2.get_templates()),5)

    def test_field_error(self) :
        with self.assertRaises(ValueError) : #Incorrect mask size
            f=nmt.NmtField(self.msk[:15],self.mps)
        with self.assertRaises(ValueError) : #Incorrect map size
            f=nmt.NmtField(self.msk,[self.mps[0,:15]])
        with self.assertRaises(ValueError) : #Incorrect template size
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[self.tmp[0,0,:15]]])
        with self.assertRaises(ValueError) : #Passing 3 maps!
            f=nmt.NmtField(self.msk,self.mps)
        with self.assertRaises(ValueError) : #Passing 3 template maps!
            f=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],templates=self.tmp)
        with self.assertRaises(ValueError) : #Passing crap as templates
            f=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],templates=1)
        with self.assertRaises(ValueError) : #Passing wrong beam
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam[:30])
        with self.assertRaises(ValueError) : #Passing crap as beam
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=1)

class TestFieldFsk(unittest.TestCase) :
    def setUp(self) :
        self.ntemp=5
        self.nx=141
        self.ny=311
        self.lx=np.pi/180
        self.ly=np.pi/180
        self.npix=self.nx*self.ny
        self.msk=np.ones([self.ny,self.nx])
        self.lmax=np.sqrt((self.nx*np.pi/self.lx)**2+(self.ny*np.pi/self.ly)**2)
        self.nell=30
        xarr=np.arange(self.nx)*self.lx/self.nx
        yarr=np.arange(self.ny)*self.ly/self.ny
        i0_x=2; i0_y=3;
        k0_x=i0_x*2*np.pi/self.lx
        k0_y=i0_y*2*np.pi/self.ly
        phase=k0_x*xarr[None,:]+k0_y*yarr[:,None]
        cphi0=k0_x/np.sqrt(k0_x**2+k0_y**2)
        sphi0=k0_y/np.sqrt(k0_x**2+k0_y**2)
        c2phi0=cphi0**2-sphi0**2
        s2phi0=2*sphi0*cphi0
        self.mps=np.array([2*np.pi*np.cos(phase)/(self.lx*self.ly),
                           2*np.pi*c2phi0*np.cos(phase)/(self.lx*self.ly),
                           -2*np.pi*s2phi0*np.cos(phase)/(self.lx*self.ly)])
        self.tmp=np.array([self.mps.copy() for i in range(self.ntemp)])
        self.beam=np.array([np.arange(self.nell)*self.lmax/(self.nell-1.),np.ones(self.nell)])

    def test_field_flat_alloc(self) :
        #No templates
        f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=self.beam)
        f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam)
        f2p=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                             purify_e=True,purify_b=True)
        self.assertTrue(normdiff(f0.get_maps()[0],self.mps[0]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),0)
        self.assertEqual(len(f2.get_templates()),0)
        self.assertEqual(len(f2p.get_templates()),0)

        #With templates
        f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=self.beam,
                            templates=np.array([[t[0]] for t in self.tmp]))
        f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                            templates=np.array([[t[1],t[2]] for t in self.tmp]))
        #Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],0*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),5)
        self.assertEqual(len(f2.get_templates()),5)

    def test_field_flat_errors(self) :
        with self.assertRaises(ValueError) : #Incorrect map sizes
            f=nmt.NmtFieldFlat(self.lx,-self.ly,self.msk,[self.mps[0]])
        with self.assertRaises(ValueError) :
            f=nmt.NmtFieldFlat(-self.lx,self.ly,self.msk,[self.mps[0]])
        with self.assertRaises(ValueError) : #Mismatching map dimensions
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0,:self.ny//2]])
        with self.assertRaises(ValueError) : #Mismatching template dimensions
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],
                               templates=np.array([[t[0,:self.ny//2]] for t in self.tmp]))
        with self.assertRaises(ValueError) : #Passing 3 templates!
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,self.mps)
        with self.assertRaises(ValueError) : #Passing 3 templates!
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],templates=self.tmp)
        with self.assertRaises(ValueError) : #Passing crap templates
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],
                               templates=1)
        with self.assertRaises(ValueError) : #Passing crap beam
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=1)
            
if __name__ == '__main__':
    unittest.main()
