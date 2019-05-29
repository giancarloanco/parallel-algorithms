#!/bin/sh
#PBS -N GreyscaleConversion
#PBS -l nodes=n009:ppn=1:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -std=c++11 greyscale_conversion.cu -o greyscale
./greyscale
