#!/bin/bash

# Usage: ./split.sh <prefix> <num_of_cluster> <num_of_gpu>

prefix=$1
num_of_cluster=$(($2 - 1))
num_of_gpu=$3

for cluster in `seq 0 $num_of_cluster`
do
	echo "Spliting table" $(($cluster + 1))

	# call split
	total_lines=$(wc -l ${prefix}_${cluster}.csv | cut -f1 -d' ')
	lines_per_file=$(((total_lines + num_of_gpu - 1) / num_of_gpu))
	split --lines=$lines_per_file -da 5 --additional-suffix=.csv ${prefix}_${cluster}.csv

	# rename files
	mv x00000.csv ${prefix}_${cluster}_0.csv
	for file in `find . -name 'x*.csv'`
	do
		mv $file `echo $file | sed "s!./x0*!${prefix}_${cluster}_!"`
	done
done

echo "All done"
