#!/bin/bash

nside_lss=1024
nside_cmb=256

if [ ! -f lambda_sfd_ebv.fits ] ; then
    echo "Downloading reddening map"
    wget https://lambda.gsfc.nasa.gov/data/foregrounds/SFD/lambda_sfd_ebv.fits
fi

if [ ! -f mask_lss.fits ] ; then
    echo "Missing mask!"
    exit
fi

if [ ! -f star_template.fits ] ; then
    echo "Missing star template!"
    exit
fi

if [ ! -f nz.txt ] ; then
    echo "Missing N(z)"
    exit
fi

echo "Generating LSS power spectra"
python get_lss_cls.py --plot

echo "Generating LSS mask"
python get_lss_mask.py --plot --nside ${nside_lss} --nholes 100 --rholes 1.

echo "Generating LSS contaminant templates"
python get_lss_contaminants.py --plot --nside ${nside_lss}

echo "Generating CMB mask"
python get_cmb_mask.py --plot --nside ${nside_cmb}

echo "Generating CMB contaminant templates"
python get_cmb_contaminants.py --plot --nside ${nside_cmb}

echo "Generating flat-sky LSS contaminant templates"
python get_lss_contaminants_flat.py --plot

echo "Generating flat-sky CMB mask"
python get_cmb_mask_flat.py --plot

echo "Generating flat-sky CMB contaminant templates"
python get_cmb_contaminants_flat.py --plot
exit
