#!/bin/bash

# wrapper for the single-file, zOptical/zWriteCSV-V3-CSV-based classification on HTCondor; to be run with condor_run.sub

export EOS_MGM_URL=root://eosuser.cern.ch

python3 -m venv env_temp
source env_temp/bin/activate
pip install --upgrade pip
pip install numpy
pip install scikit-learn

maindirexec=$1
filedataname=$2
outpreddataname=$3
outhistdataname=$4
filecls0name=$5
filecls1name=$6

echo "running job with:"
echo "vvv"
echo $maindirexec
echo $filedataname
echo $outpreddataname
echo $outhistdataname
echo $filecls0name
echo $filecls1name
echo "^^^"

mkdir -p output_pred
mkdir -p output_hist
mkdir -p output_info
eos cp $maindirexec/classify.py .

time python classify.py $filedataname $outpreddataname $outhistdataname $filecls0name $filecls1name

deactivate

eos cp -r output_pred $maindirexec
eos cp -r output_hist $maindirexec
