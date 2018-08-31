#!/bin/bash

mkdir -p output

for nside in 64 256 1024 4096
do
    for task in field mcm pure deproj pure_deproj
    do
	for ncores in 1 2 4 8 16
	do
	    if [ ${nside} -gt 256 ]
	    then
		ncomp=10
	    else
		ncomp=100
	    fi
	    addqueue -q cmb -s -n 1x${ncores} -m 4 ./timings -nside ${nside} -ncomp ${ncomp} -do_${task} -out output/${task}_ns${nside}_nc${ncores}.txt
	done
    done
done
