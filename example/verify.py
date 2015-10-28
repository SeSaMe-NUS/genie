
#data_file = open("/media/hd1/home/luanwenhao/TestData2Wenhao/sift_hash/sift_hash_100k.csv", "r")
#data_file = open("sift_test_1k.csv", "r")
#data_file = open("/media/hd1/home/zhoujingbo/data/sift/vivo_sift_wulifu/data/vivo_first_try/train/train.csv");
data_file = open("/media/hd1/home/zhoujingbo/workspace/python/LazySVM/data/ocr/hashed/numCF25/ocr_trn_dataSize-3400000_numCF-25_numDim-1155_r-2.65_domainBits-13.csv");

l = list(data_file)

while(True):
	radius = input("Raidus: ")
	query = input("Query line: ")
	query_data = l[query].split(",")
	#del query_data[-1]
	data_str = raw_input("Data lines: ")
	data_list = data_str.split(", ")
	data_list = [int(i) for i in data_list]

	for i in range(len(data_list)):
		count = 0
		index = data_list[i]
		#data = l[index + 1].split(",")
		data = l[index ].split(",")
		#del data[-1]
		for j in range(len(query_data)):
			if(len(query_data[j])<1 or len(data[j])<1) :
				break;
			if abs(int(query_data[j]) - int(data[j])) <= radius:
				count += 1
		print "data "+str(index)+": "+str(count)


