#!/bin/bash

# split files into 2 for loading on 2 GPUs
for cluster in $(find . -name "*.csv")
do
	lines=$(wc -l $cluster | cut -f1 -d' ')
	first_half=$((lines/2))
	second_half=$((lines-first_half))
	head -n $first_half $cluster > ${cluster%.csv}_0.csv
	tail -n $second_half $cluster > ${cluster%.csv}_1.csv
done
