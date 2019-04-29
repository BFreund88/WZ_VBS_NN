name=WZ_HyperOpt
echo "bash launch.sh |& tee $PWD/logs/$name.LOG" \
    | qsub -v "NAME=$name,DATA_PREFIX=$data_prefix,TYPE=$type" \
    -N $name \
    -d $PWD \
    -l nice=0 \
    -j oe \
    -o $PWD/submit 
