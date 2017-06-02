###########################
#                         #
# This is a sample script #
# used for pre-clustering #
#                         #
###########################

import pandas as pd
from sklearn.cluster import KMeans

# read data from csv
print "Start reading file"
input_csv = pd.read_csv("../original-dataset/sift_4.5m.csv", header=None)

# perform KMeans clustering
print "Start clustering"
k = 20
kmeans_model = KMeans(n_init=1, n_clusters=k).fit(input_csv.iloc[:, :])
labels = kmeans_model.labels_

# save result to K csv files
print "Start generating output"

output_csvs = list()
for i in range(k):
    f = open("sift_big_" + str(i) + ".csv", "w")
    output_csvs.append(f)

for i in range(input_csv.shape[0]):
    label_i = labels[i]
    output_csvs[label_i].write(','.join(input_csv.iloc[i,:].to_csv(header=False, index=False).rstrip("\n").split("\n")) + "\n")

for output in output_csvs:
    output.close()
