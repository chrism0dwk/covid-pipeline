#!/bin/bash

#####################################################
# Name: enqueue_pipeline.sh			    #
# Author: Chris Jewell <c.jewell@lancaster.ac.uk>   #
# Created: 2021-02-07				    #
# License: MIT					    #
# Copyright: Chris Jewell 2021			    #
# Purpose: Launch a COVID pipeline on a SGE cluster #
# Usage: enqueue_pipeline.sh <ISO8601 date>	    #
#####################################################



while getopts 'c:d:s:r:' OPTION; do
    case "$OPTION" in
	c)
	    arg_config=${OPTARG}
	    ;;
	d)
	    arg_date=${OPTARG}
	    ;;
	s)
	    arg_staging_dir=${OPTARG}
	    ;;
	r)
	    arg_results_dir=${OPTARG}
	    ;;
	:)
	    exit 1
	    ;;
	*)
	    echo "usage: enqueue_pipeline.sh -c <config file> -d <ISO6801 date> -s <staging dir> -r <results_dir>"
	    exit 1
	    ;;
    esac
done


# Fix for Lancaster HEC environment
source /usr/shared_apps/admin/etc/sge/switch-gpu.sh

# Environment variables for date and results staging
JOBNAME=c19_$arg_date_$arg_config
STDOUT=$arg_staging_dir/stdout.txt

# Final results dir
RESULTS_DEST_DIR=/mnt/covid/c19/covid_pipeline

mkdir -p ${arg_staging_dir}
qsub -N $JOBNAME -o $STDOUT \
     pipeline.sge \
     -c "$arg_config" \
     -d "$arg_date" \
     -s "$arg_staging_dir" \
     -r "$arg_results_dir"

exit 0
