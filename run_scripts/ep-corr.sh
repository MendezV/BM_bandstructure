#!/bin/bash
###########################
#
#Scritp that reads parameter file and
#runs the integration routine sequentially for each parameter
#in the file
#
#
###########################
export OPENBLAS_NUM_THREADS=20

#generic parameters 
Lattice_size=24
filling_seed=0
twist_angle=1.05
kappa=0.75
Mode_HF=0
phonon_polarization='T'

#needed prerequisites for the run
parameter_file='params'
dire_to_mods='../Mods/'

#Reading parameter file
param_arr=$(awk -F= '{print $1}' ${parameter_file})
echo ${param_arr}

jobname="trans_0"  #JOBNAME importan to declare -has to be descriptive

#General info about the job
date_in="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_in}" >inforun
echo '....Jobs started running' >>  inforun

#Temporary directories where the runs will take place
dire_to_temps="../temp/temp_${jobname}_${date_in}"
rm -rf "${dire_to_temps}"
mkdir "${dire_to_temps}"

#loop over the parameters
for param_val in ${param_arr[@]}; do

	#create one temporary directory per parameter
	dire=""${dire_to_temps}"/${jobname}_${param_val}"
	rm -rf "${dire}"
	mkdir -vp "${dire}"

	cp ${dire_to_mods}Bubble_ep_BM.py  "${dire}"
    cp ${dire_to_mods}Dispersion.py  "${dire}"
    cp ${dire_to_mods}MoireLattice.py  "${dire}"
    cp ${parameter_file}  "${dire}"
	cp -r dispersions "${dire}"
	#entering the temp directory, running and coming back
	cd "${dire}"
	echo "parameters: L " ${Lattice_size} " nu " ${filling_seed} " th " ${param_val} " kap " ${kappa} " HF " ${Mode_HF} " phLT " ${phonon_polarization} >> output.out

	nohup time python3 -u Bubble_ep_BM.py ${Lattice_size} ${filling_seed} ${param_val} ${kappa} ${Mode_HF} ${phonon_polarization} >> output.out	
	
	cd "../../../Mods"
	sleep 1

done

wait

#general info about the job as it ends
date_fin="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_fin}" >>inforun
echo 'Jobs finished running'>>inforun

#moving files to the data directory and tidying up
dire_to_data="../data/${jobname}_${date_fin}"
mkdir "${dire_to_data}"
mv "${dire_to_temps}"/* "${dire_to_data}"
mv inforun "${dire_to_data}"
rm -r "${dire_to_temps}"
