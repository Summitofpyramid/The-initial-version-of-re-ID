#!/usr/bin/python
'''
	usage: cmc.py file1.csv [file2.csv | file3.csv ]
	
	Take a list of .csv files from the command line arguments
	and plot them together in a CMC plot. 
	Each .csv file contains a Similarity matrix of mxn elements.
		m is the number of items in the test gallery 
		n is the number of probe items 
		at each location[i,j] the .csv contain the Similarity value
		between the probe j and the gallery item i.
		Similarity is higher at 0 
'''

import os
import sys
import csv
import operator
import numpy as np
import matplotlib.pyplot as plt

listOfTables = sys.argv[1:]
nameOfTables = [ os.path.splitext(name)[0] for name in listOfTables]

'''
	Takes a .csv file with mxn items and provides the 
	rank of the probes in the instance attribute probesRank.
	
	The table assumes the first n elements of the m rows (gallery items)
	correspond to the correct classification of the probes items. That means that 
	table at row[j,j] with j < n correspond to the similitud value of probe j 
	with its correct match in the gallery
'''
class SimilarityTable:

	def __init__(self, tableFilename):
		self.table = []
		self.m = 0   # number of classes in gallery
		self.n = 0   # number of probes
		self.probesRank = []
		with open(tableFilename, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			firstRow = True;
			table = []
			for row in spamreader:
				self.m = self.m +  1
				rowData = [float(x.strip()) for x in row]
				numOfProbes = len(rowData)
				if self.n == 0:
					self.n = numOfProbes
				elif self.n != numOfProbes:
					self.__error("Table is malformed.\n")
					self.__error("row: %d has: %d elements and should have %d \n" %\
					 (self.m, numOfProbes, self.n))
					break; 
				table.append(rowData)
			self.table = np.array(table)
		self.__rankProbes()
	
	def __error(self, message):
		sys.stderr.write(message)
		
	def __printInfo(self):
		print("Similarity Table with %d probles and %d gallery classes\n" %\
			(self.n, self.m))
			
	def __rankProbes(self):
		for colIdx in range(self.n):
			col =  [ (i,v) for i, v in enumerate(self.table[:,colIdx])]
			col = sorted(col, key=operator.itemgetter(1))
			rank = [ i + 1  for i, v in enumerate(col) if v[0] == colIdx]
			self.probesRank.append(rank[0])
		
class CMCPlot:

	def __init__(self):
		plt.xlabel('Rank')
		plt.ylabel('Probability')
		plt.title('CMC')
		plt.grid(True)

	def addCurves(self, list,listNames):
		for idx, obj in enumerate(list):
			self.addCurve(obj, listNames[idx])
		
	def addCurve(self, table, name):
		prob = []
		for rank in range(1, table.m + 1):
			counter = len([x for x in table.probesRank if x <= rank])
			prob.append( float(counter) / float(table.m))
		handle, = plt.plot(range(1, table.m + 1), prob, label=name)
		plt.axis([1, table.m + 1, 0, 1 ])
				
	def legend(self):
		plt.legend(loc='lower right')
		
	def show(self):
		plt.show()
					

sim = [SimilarityTable(t) for t in listOfTables]

cmc = CMCPlot()
cmc.addCurves(sim, nameOfTables)
cmc.legend()
cmc.show()



