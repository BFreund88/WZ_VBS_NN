#!/bin/bash 
:: Random hyper parameter optimization in batch mode for BDT

source /home/zp/freund/.bashrc
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.14.04-x86_64-slc6-gcc62-opt" 

nest=$(( ( ( RANDOM % 20 )  + 1 ) * 100  ))
depth=$(( ( RANDOM % 5 )  + 1 ))
lr=$(python -c "import random;print(random.randint(0, 60)*0.001)")
opt=0

name=${opt}_${nest}_${depth}_${lr}_

echo "==> Running the BDT Optimisation"

python2 OPT_VBS_BDT.py --v 2 \
    --opt $opt \
    --depth $depth \
    --lr $lr \
    --output $name

exit 0