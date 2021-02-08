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

# Fix for Lancaster HEC environment
source /usr/shared_apps/admin/etc/sge/switch-gpu.sh

# Environment variables for date and results staging
export COVIDREFDATE=$@
export COVIDRESULTSDIR=${global_scratch}/covid_pipeline/$COVIDREFDATE
JOBNAME=c19_$COVIDREFDATE
STDOUT=$COVIDRESULTSDIR/stdout.txt

# Final results dir
RESULTS_DEST_DIR=/mnt/covid/c19/covid_pipeline

# Email success/fail
MAILTO="c.jewell@lancaster.ac.uk"



stdout_str() {
    if [ -f "$STDOUT" ]
    then
	STDOUT_CONTENT=$(<$STDOUT)
    fi
    cat <<EOF 
===STDOUT===
$STDOUT_CONTENT
===STDOUT===
EOF
}    



email_results() {
    stdout_str | mail -s "[COVID Pipeline] ${1:-"Unknown"}" $MAILTO
}



error_exit() {
    MSG=${1:-"Unknown Error"}
    email_results "Error $MSG"
    echo $MSG 1>&2
    exit 1
}

mkdir -p $COVIDRESULTSDIR ||
    error_exit "Cannot create output directory $COVIDRESULTSDIR"

qsub -sync y -N c19_$COVIDREFDATE -o $STDOUT \
     -v COVIDREFDATE="$COVIDREFDATE" -v COVIDRESULTSDIR="$COVIDRESULTSDIR" \
     pipeline.sge >>$STDOUT 2>&1 || error_exit "Pipeline job failed"

rsync -a ${COVIDRESULTSDIR} ${RESULTS_DEST_DIR} >>$STDOUT 2>&1 ||
    error_exit "Error copying $COVIDRESULTSDIR to $RESULTS_DEST_DIR"

email_results "Success"
exit 0

