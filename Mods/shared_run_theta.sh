#!/bin/bash
###########################
#
#Scritp that reads parameter file and
#runs the integration routine sequentially for each parameter
#in the file
#
#
###########################

#Readibg parameter file

param_arr=$(awk -F= '{print $1}' params_thet2)
jobname="thetasweep_kappa_0.75_N30"  #JOBNAME importan to declare -has to be descriptive
pow=$PWD #saving the current working directory


#General info about the job
date_in="`date "+%Y-%m-%d-%H-%M-%S"`"
echo $date_in

#loop over the parameters
for param_val in ${param_arr[@]}; do

	echo "${pow} is the current working directory"
	echo "${param_val}"

	#create one temporary directory per parameter
	jname="${jobname}_${param_val}"

	#creating temp directory in scratch to run
	dir="${TMPDIR}/${jname}_${date_in}"
	mkdir -v "${dir}"

	#moving seeds, parameters and TBG_landau.out to the new dir
	echo "started copy of job files...."
    cp Bubble_ep.py "${dir}"
    cp Hamiltonian.py  "${dir}"
    cp MoireLattice.py  "${dir}"
    cp params_thet2  "${dir}"
	cp -r dispersions "${dir}"
	cp shared_job.sh "${dir}"
	#entering the temp directory, running and coming back
	echo "ended copy of job files...."

	#entering the temp directory, running and coming back
	echo "submitting job...."
	cd "${dir}"
	sbatch -J "${jname}" --export=ALL,param_val="${param_val}" shared_job.sh
	sleep 1
	echo "moving back to ${pow} ...."
	cd "${pow}"

	sleep 1


done

