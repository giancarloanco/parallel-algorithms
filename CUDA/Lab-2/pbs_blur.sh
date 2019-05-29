#!/bin/sh
#PBS -N Blur
#PBS -l nodes=n009:ppn=1:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -std=c++11 blur.cu -o blur
./blur
