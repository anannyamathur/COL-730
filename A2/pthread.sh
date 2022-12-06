#!/bin/bash
gcc -g -Wall -o pth pthread.cpp -lpthread -lm
./pth $1 $2