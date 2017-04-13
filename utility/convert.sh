#!/bin/bash

# Usage: bash convert.sh <num-of-cluster> <num-of-gpu>
# Make sure you have the csv2binary executable in the same folder

num_of_cluster=$1
num_of_gpu=$2
prefix=$3
for i in {0..${num_of_cluster}}
do
	for j in {0..${num_of_gpu}}
	do
		./csv2binary ${prefix}_${i}_${j}.csv ${prefix}_${i}_${j}.dat
	done
done
