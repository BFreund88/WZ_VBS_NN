#!/bin/bash 
:: Random hyper parameter optimization in batch mode for NN

source /home/zp/freund/.bashrc
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.14.04-x86_64-slc6-gcc62-opt"

numlayer=$(( ( RANDOM % 5 )  + 1 ))
numn=$(( ( ( RANDOM %30 )  + 1 ) * 10 ))
epochs=200
:: opt=$(( ( RANDOM % 2 ) ))
dropout=$(python -c "import random;print(random.randint(0, 60)*0.01)")
patience=$(( ( RANDOM % 20 )  + 1 )) 
lrrate=$(python -c "import random;print(random.randint(1, 20)*0.001)")

name=$lrrate_${numlayer}_${numn}_${dropout}_${patience}_

python OPT_VBS_NN.py --v 2 \
    --lr $lrrate \
    --epoch $epochs \
    --numn $numn \
    --numlayer $numlayer \
    --dropout $dropout \
    --patience $patience \
    --output $name


exit 0