# v1.1
- Faster MCM via symmetry (thanks to Thibaut Louis) (#111)
- Covariance calculable for coupled spectra (#113)
- Arbitrary spins (#115)
- Additional binning functionality (thanks to Joe Zuntz) (#116)
- Lightweight and map-less fields (#117)

# v1.0.2 changes
- Pip-installable (#95).
- Installation instructions on rtd (#95).

# v1.0.1 changes
- Input maps and templates can be masked (#93).
- Input theory power spectra are culled to lmax automatically (#90).
- Libsharp, HEALPix and libnmt automatically installed (#84).
- Code is now PEP8 compliant (#86).
- Workspaces are now stored as FITS files (which should make them easier to read externally) (#83).
- More convenient NmtBin constructors (#80).
- Lmax_mask, lmax_sht, lmax_bins (#74).
- Parallelized Gaussian covariance (#73).
- Ability to recover bandpower window functions easily (#55)


# v1.0 changes
- Support for rectangular pixelizations (CAR pixelization with Clenshaw-Curtis separation).
- Support for spin-0 and spin-2 covariance matrices.
- Convenience functions to recover bandpower windows and the unbinned mode-coupling matrix.
- Arbitrary ell-weighting (e.g. can compute D_l = l*(l+1)*C_l/2pi).
- Simultaneous calculation of the MCM for all spin-0 and spin-2 combinations.
