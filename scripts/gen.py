#####################################
#                                   #
# Generating queries from CSV files #
#                                   #
#####################################

import json

if __name__ == "__main__":
    count = 500
    queries = dict()
    queries["topk"] = 100
    queries["queries"] = list()
    with open("sift_big_0.csv", "r") as data1:
        for line in data1.readlines()[:count]:
            numbers = line.rstrip("\n").split(",")[:-1]
            numbers = [int(x) for x in numbers]
            queries["queries"].append(numbers)
    with open("sift_big_1.csv", "r") as data2:
        for line in data2.readlines()[:count]:
            numbers = line.rstrip("\n").split(",")[:-1]
            numbers = [int(x) for x in numbers]
            queries["queries"].append(numbers)
    print json.dumps(queries)
