#!/bin/bash
gcc -g -Wall -fopenmp -lm -o openmp openmp.cpp
./openmp $1 $2