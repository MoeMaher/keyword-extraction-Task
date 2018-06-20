import nltk
import numpy, sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import os
from os import listdir

import sys
import subprocess
import glob
from os import path

data_w = []
data_key =	[]
data_title = []
keyCount = []
flag2 = False
for file in listdir('/home/maher/keywordExtraction/Hulth2003/training'):
	flag = True
	if file.endswith('abstr'):
		f = file 	
		tagname = f[:-5]+ 'uncontr'
		ff = open(os.path.join('/home/maher/keywordExtraction/Hulth2003/training', f))
		tag = open(os.path.join('/home/maher/keywordExtraction/Hulth2003/training', tagname))
		
		keywords = tag.readlines()
		keywords = ' '.join(keywords)
		keywords = keywords.replace(";","")
		keywords = keywords.replace("(","")
		keywords = keywords.replace(")","")
		keywords = keywords.replace("[","")
		keywords = keywords.replace("]","")
		keywords = keywords.replace("/","")
		keywords = keywords.replace("'s","")
		keywords = ' '.join(nltk.word_tokenize(keywords))
		keywords = keywords.split(' ')
		keyCount += keywords
		list = ff.readlines()
		# print(len(keywords))
		
		for i in range(len(list)):
			sent = nltk.word_tokenize(list[i])
			for w in sent:
				data_w.append(w)
				if flag:
					data_key.append(1) 
				else:
					data_key.append(0)						
				if w in keywords:
					data_title.append(1)
				else:
					data_title.append(0)	
			if not (flag and list[i].endswith('\n') and list[i+1].startswith('\t')):
				flag = False
		data_w.append('#')
		data_key.append(0)
		data_title.append(0)

				
keyCount = set(keyCount)
blah = set(data_w)
print(len(keyCount)/len(blah))
import os
os.remove("/home/maher/keywordExtraction/datasetWordsTokenized.txt")
orig_stdout = sys.stdout			
output = open('/home/maher/keywordExtraction/datasetWordsTokenized.txt', 'w')
sys.stdout = output
for i in range(len(data_w)):
	print(data_w[i], data_key[i], data_title[i])
	
sys.stdout = orig_stdout
output.close()


os.remove("/home/maher/keywordExtraction/wordsTokenized.txt")

orig_stdout = sys.stdout			
output = open('/home/maher/keywordExtraction/wordsTokenized.txt', 'w')
sys.stdout = output

for i in range(len(data_w)):
	if data_w[i] == "":
		break 
		
	print(data_w[i])
	
sys.stdout = orig_stdout
output.close()





















