#!/bin/bash

# Usage: bash convert.sh <num-of-cluster> <num-of-gpu>
# Make sure you have the csv2binary executable in the same folder

num_of_cluster=$1
num_of_gpu=$2
for i in {0..${num_of_cluster}}
do
	for j in {0..${num_of_gpu}}
	do
		./csv2binary sift_big_${i}_${j}.csv sift_big_${i}_${j}.dat
	done
done
