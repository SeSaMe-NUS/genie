inFileName = raw_input('Please enter the data file namae: ')
numQuery = int(raw_input('Please enter number of queries: '))
selectivity = float(raw_input('Please enter selectivity: '))
weight = int(raw_input('Please enter weight: '))

confirmMsg = 'Parameters: file %s, numQuery %d, selectivity %.4f, weight %d.\n' \
			 'Generating query file...\n'%(inFileName, numQuery, selectivity, weight)
print confirmMsg

outFileName = inFileName[:-4]+'_query'+inFileName[-4:]
outNumQuery = 0

inFile = open(inFileName, 'r')
outFile = open(outFileName, 'w')

queryFormat = '%d,%d,%d,'+str(selectivity)+','+str(weight)+'\n'

for line in inFile:
	if outNumQuery >= numQuery: break
	if not line or line == '\n': continue
	tokens = line.split(',')
	for dim in range(len(tokens)):
		if not tokens[dim]: continue
		try:
			value = int(tokens[dim])
		except:
			continue
		queryMsg = queryFormat%(outNumQuery, dim, value)
		outFile.write(queryMsg)
	outNumQuery += 1

inFile.close()
outFile.close()

finishMsg = 'Query file %s generated with %d queries!\n'%(outFileName, outNumQuery)
print finishMsg