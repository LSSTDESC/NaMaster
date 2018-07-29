#!/bin/bash

#exc="addqueue -q cmb -s -n 1x12 -m 2.0 /usr/local/shared/python/2.7.6-gcc/bin/python"
exc="python"
nsims=1000
which_partition="regular"
timelim_hou=00
timelim_min=15
timelim_sec=00
run_in_nodes=1

mkdir -p tests_sph
for nside in 512
do
    if [ ${nside} -eq 64 ]
    then	
	timelim_hou=00
	timelim_min=05
	timelim_sec=00
    elif [ ${nside} -eq 256 ]
    then
	timelim_hou=00
	timelim_min=15
	timelim_sec=00
    elif [ ${nside} -eq 512 ]
    then
	timelim_hou=00
	timelim_min=30
	timelim_sec=00
    fi
    for msk in 1
    do
	for cont in 0 1
	do
	    if [ ${cont} -eq 0 ] #Run pure B
	    then
		for aposize in 1. 2. 5. 10. 20.
		do
		    for pureB in 0 1
		    do
			command="${exc} check_sph_pure.py ${nside} ${cont} ${nsims} 0 ${aposize} C1 0 ${pureB}"
			if [ ${run_in_nodes} -eq 1 ]
			then
			    runfile=tests_sph/run_batch_ns${nside}_mask${msk}_apo${aposize}_cont${cont}_pure${pureB}.sh
			    cat > ${runfile} <<EOF
#!/bin/bash -l
#SBATCH --partition ${which_partition}
##SBATCH --qos premium
#SBATCH --nodes 1
#SBATCH --time=${timelim_hou}:${timelim_min}:${timelim_sec}
#SBATCH --job-name=sph_${nside}_${aposize}_${cont}_${pureB}
#SBATCH --account=m1727
#SBATCH -C haswell
module load python/2.7-anaconda
srun -n 1 ${command}
EOF
			    
			    cat ${runfile}
			    #sbatch ${runfile}
			else
			    echo ${command}
			    #${command}
			fi
			echo " "
		    done
		done
	    fi
	    for aposize in 1.
	    do
		command="${exc} check_sph.py ${nside} ${msk} ${cont} ${nsims} 0 ${aposize}"
		if [ ${run_in_nodes} -eq 1 ]
		then
		    runfile=tests_sph/run_batch_ns${nside}_mask${msk}_apo${aposize}_cont${cont}_nopure.sh
		    cat > ${runfile} <<EOF
#!/bin/bash -l
#SBATCH --partition ${which_partition}
##SBATCH --qos premium
#SBATCH --nodes 1
#SBATCH --time=${timelim_hou}:${timelim_min}:${timelim_sec}
#SBATCH --job-name=sph_${nside}_${aposize}_${cont}
#SBATCH --account=m1727
#SBATCH -C haswell
module load python/2.7-anaconda
srun -n 1 ${command}
EOF
		
		    cat ${runfile}
		    #sbatch ${runfile}
		else
		    echo ${command}
		    #${command}
		fi
		echo " "
	    done
	done
    done
done

mkdir -p tests_flat
for msk in 1
do
    for cont in 0 1
    do
	for aposize in 0. 0.1
	do
	    command="check_flat.py ${msk} ${cont} ${nsims} 0 ${aposize}"
	    if [ ${run_in_nodes} -eq 1 ]
	    then
		runfile=tests_flat/run_batch_mask${msk}_apo${aposize}_cont${cont}_nopure.sh
		cat > ${runfile} <<EOF
#!/bin/bash -l
#SBATCH --partition ${which_partition}
##SBATCH --qos premium
#SBATCH --nodes 1
#SBATCH --time=${timelim_hou}:${timelim_min}:${timelim_sec}
#SBATCH --job-name=flt_${aposize}_${cont}
#SBATCH --account=m1727
#SBATCH -C haswell
module load python/2.7-anaconda
srun -n 1 python ${command}
EOF
		
		cat ${runfile}
		#sbatch ${runfile}
	    else
		echo ${command}
		#${command}
	    fi
	    echo " "
	done
    done
done

exc="python"
for nside in 64 256
do
    for msk in 1
    do
	for cont in 0 1
	do
	    for aposize in 1.
	    do
		command="${exc} check_sph.py ${nside} ${msk} ${cont} ${nsims} 1 ${aposize}"
		echo ${command}
		#${command}
		echo " "
	    done
	done
    done
done

for msk in 1
do
    for cont in 0 1
    do
	for aposize in 0. 0.1
	do
	    command="${exc} check_flat.py ${msk} ${cont} ${nsims} 1 ${aposize}"
	    echo ${command}
	    #${command}
	    echo " "
	done
    done
done
