import nltk
from sklearn.naive_bayes import GaussianNB
from random import shuffle
import sys
#------------------------------------------

from process_data import *
from extract_lexical import *
from ngram import *
from unigram_swn import *
from tfidf import *
from aspect_noun import *
from chi_square import *
#------------------------------------------


n,wt,noun_count,feature_no=1,15,100,500
#data=data[:50]+data[-50:]

f=open('logs.txt','w')
f.close()
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
parameter=1
binn=0

if len(sys.argv)==1:
	print 'using full.review for testing, With all possible combinations'
	filename='full.review'
	data = process(filename)
	print len(data),'reviews in all\n','-'*30
	target=[]
	for i in data:
		target.append(i[1])
	for i in range(18):
		if i>=2 and i<=16:
			for j in range(7):
				for k in range(5):
					predict(data,target,n,wt,noun_count,parameter=i,p2=j,p3=k+1,binn=1,num_features=feature_no)
		elif i==1 or i==17:
			for l in range(5):
				predict(data,target,n,wt,noun_count,parameter=i,p2=0,p3=l+1,num_features=feature_no)
				
elif len(sys.argv)==6:
	filename=str(sys.argv[1])
	data = process(filename)
	#data=data[:50]+data[-50:]
	print len(data),'reviews in all\n','-'*30
	target=[]
	for i in data:
		target.append(i[1])
	para1=int(sys.argv[2])
	para2=int(sys.argv[3])
	para3=int(sys.argv[4])
	para4=int(sys.argv[5])
	predict(data,target,n,wt,noun_count,parameter=para1,p2=para2,p3=para3,binn=para4,num_features=feature_no)
	
elif len(sys.argv)==5:
	filename=str(sys.argv[1])
	data = process(filename)
	#data=data[:50]+data[-50:]
	print len(data),'reviews in all\n','-'*30
	target=[]
	for i in data:
		target.append(i[1])
	para1=int(sys.argv[2])
	para2=int(sys.argv[3])
	para3=int(sys.argv[4])
	predict(data,target,n,wt,noun_count,parameter=para1,p2=para2,p3=para3,binn=0,num_features=feature_no)
	
else:
	print 'Usage: fillename parameter1 parameter2 parameter3 bin_key'
	exit(0)


#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------






###########################################################################
'''
Usage: fillename parameter1 parameter2 parameter3 bin_key(optional) 
	where filname-(name of resourse file), 
	parameter1 - index of feature extraction/weighing method,
	parameter2 - index of feature selection method,
	parameter3 - index of prediction algorithm
	bin_key 	 - 0/1 depending if binning is to be done.(optional).default is 0
	
Legends:
Parameter 1:
		1=  FS1(lexical)
		2=  FS2 (Ngram Count)
		3=  FS3(SWN)
		4=  FS4(Aspect Extraction)
		5=  FS3(SWN) + FS4(Aspect Extraction)
		6=  FS2(Simple Ngram Count) + FS3(SWN) + FS4(Aspect Extraction)
		7=  FS1(Lexical Features) + FS2(Simple Ngram Count) + FS3(SWN) + FS4(Aspect Extraction)
		8=  tfidf with tf=b, idf=default		
		9=  tfidf with tf=b, idf=s
		10= tfidf with tf=b, idf=sp
		11= tfidf with tf=a, idf=default
		12= tfidf with tf=a, idf=s
		13= tfidf with tf=a, idf=sp
		14= tfidf with tf=o, idf=default
		15= tfidf with tf=o, idf=s
		16= tfidf with tf=o, idf=sp
Parameter 2:
		0= No Feature selection
		1= InformationGain Selection
		2= GainRatio Selection
		3= ChiSquare Selection
		4= Pointwise Mutual information
		5= Categorial Proportional Difference
		6= KL Divergence
Parameter 3:
		1= GaussianNB
		2= Logistic Regression
		3= RandomForestClassifier
		4= SVC with linear kernel
		5= SVC with radial kernel
		
		
Other Parameters with descriptions:
					n=	Represents n-gram (1-unigram, 2-bi-gram, etc)
					wt=	The minimum frequency of n-gram to be selection in feature extraction
	noun_count=	Number of noun features to be extracted, based on frequency
	feature_no=	Number of features to be selected in feature selection algorithms
'''
###########################################################################





