#!/bin/sh
#PBS -N test_vecAdd
#PBS -l nodes=1:ppn=1:gpus=1

cd $PBS_O_WORKDIR

./firstTest
