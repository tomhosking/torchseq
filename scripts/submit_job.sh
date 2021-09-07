#!/bin/bash

MCKENZIE_HOOK=~/mckenzie/scripts/hook.sh



POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --config)
    CONFIG="$2"
    POSITIONAL+=("$1") # save it in an array for later
    POSITIONAL+=("$2") # save it in an array for later
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters




res=$(sbatch $@)


jobId=`echo $res | sed -E 's/Submitted batch job ([0-9]+)/\1/'`

partition=`scontrol show job $jobId | grep "Partition=([^\s]+)" -Po | sed s/Partition=//`

if [ "$jobId" != "$res" ]
then
    
    ${MCKENZIE_HOOK} -a 1 -i $jobId -p $partition > /dev/null
    
    jobName=$(cat $CONFIG | grep \"name\"\: | sed -E 's/.+\"name\": \"(.*)\"\,/\1/')
    jobTag=$(cat $CONFIG | grep \"tag\"\: | sed -E 's/.+\"tag\": \"(.*)\"\,/\1/')

    echo "Batch job ID $jobId -> $jobTag/$jobName"
    
    ${MCKENZIE_HOOK} -i $jobId -p $partition -n $jobTag/$jobName > /dev/null
    sleep 5
else
    echo "Error submitting job!"
    echo res
fi