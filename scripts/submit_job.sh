#!/bin/bash

MCKENZIE_HOOK=~/mckenzie/scripts/hook.sh

res=$(sbatch $1 $2)


jobId=`echo $res | sed -E 's/Submitted batch job ([0-9]+)/\1/'`

if [ "$jobId" != "$res" ]
then

    
    ${MCKENZIE_HOOK} -a 1 -i $jobId > /dev/null
    
    jobName=$(cat $2 | grep \"name\"\: | sed -E 's/.+\"name\": \"(.*)\"\,/\1/')
    jobTag=$(cat $2 | grep \"tag\"\: | sed -E 's/.+\"tag\": \"(.*)\"\,/\1/')
    
    ${MCKENZIE_HOOK} -i $jobId -n $jobTag/$jobName > /dev/null

    echo "Batch job ID $jobId -> $jobTag/$jobName"
else
    echo "Error submitting job!"
    echo res
fi