#!/usr/bin/env /bin/bash

# wrapper for the single-file, zOptical/zWriteCSV-V3-CSV-based classification, to be run locally

# note: a dedicated virtual environment might be required...
# from the ~ folder:
# > python3 -m venv env_classify
# > source env_classify/bin/activate
# > pip install --upgrade pip
# > pip install numpy
# > pip install scikit-learn
# then navigate back to the script folder

while read IN1 IN2 IN3 IN4 IN5 IN6 ; do
    
    maindirexec="." #$IN1
    filedataname=$IN2
    outpreddataname=$IN3
    outhistdataname=$IN4
    filecls0name=$IN5
    filecls1name=$IN6
    
    echo "running job with:"
    echo "vvv"
    echo $maindirexec
    echo $filedataname
    echo $outpreddataname
    echo $outhistdataname
    echo $filecls0name
    echo $filecls1name
    echo "^^^"

    time python classify.py $filedataname $outpreddataname $outhistdataname $filecls0name $filecls1name
    
    echo "---"
    
done < "condor_submit/condor_args.dat" | awk '{ print $1,$2,$3,$4,$5,$6 }'
