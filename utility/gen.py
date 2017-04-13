import json

def add_data(filename, queries, count, cluster):
    with open(filename, "r") as data:
        for line in data.readlines()[:count]:
            numbers = line.rstrip("\n").split(",")
            numbers = [int(x) for x in numbers]
            single_query = dict()
            single_query["content"] = numbers
            single_query["clusters"] = [cluster]
            queries["queries"].append(single_query)

if __name__ == "__main__":
    cluster_count = 20
    count = 25
    queries = dict()
    queries["topk"] = 1000
    queries["queries"] = list()

    for i in range(cluster_count):
        add_data("sift_big_" + str(i) + "_0.csv", queries, count, i)
        add_data("sift_big_" + str(i) + "_1.csv", queries, count, i)
    print json.dumps(queries)
