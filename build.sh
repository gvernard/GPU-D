#! /bin/bash -f
#nvcc -D_DEBUG -lgomp -arch=compute_11 --compiler-options "-fno-strict-aliasing -Wall -fopenmp" -O3 -o lenser_gpu lensDriver.cpp lensFuncs.cpp help.cpp cudaLens.cu
nvcc -D_DEBUG -lgomp -arch=compute_60 --compiler-options "-fno-strict-aliasing -Wall -fopenmp" -O3 -o lenser_gpu lensDriver.cpp lensFuncs.cpp help.cpp cudaLens.cu
