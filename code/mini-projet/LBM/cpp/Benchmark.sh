#!/usr/bin/env bash

BASEDIR=$PWD
NX_base=42
NY_base=18

for scale_factor in 1 5 10 15 20 30 50 100 1000 5000 10000
do  

    NX=$(($NX_base * $scale_factor))
    NY=$(($NY_base * $scale_factor))
    echo $NX
    echo $NY

    cd $BASEDIR/LBM_cuda/build/nvcc/src/

    sed -i "11 s/nx=\([0-9]\+\)/nx=$NX/g" './flowAroundCylinder.ini'
    sed -i "12 s/ny=\([0-9]\+\)/ny=$NY/g" './flowAroundCylinder.ini'
    echo "Launching GPU solver"
    ./lbmFlowAroundCylinder

    cd $BASEDIR/LBM_cpp/build_seq/src/
    sed -i "11 s/nx=\([0-9]\+\)/nx=$NX/g" './flowAroundCylinder.ini'
    sed -i "12 s/ny=\([0-9]\+\)/ny=$NY/g" './flowAroundCylinder.ini'
    echo "Launching sequencial solver"
    ./lbmFlowAroundCylinder

done