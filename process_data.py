from process_data import *
from extract_lexical import *
from ngram import *
from unigram_swn import *
from tfidf import *
from aspect_noun import *
from statistics import *
from chi_square import *
from cpd import *
from pmi import *
from informationGain import *
from gainRatio import *
from jsd import *
from sklearn.tree import DecisionTreeClassifier

def process(filename):
	data=[]
	review=[None,None]
	f=open(filename)
	read_rating=0
	read_review=0
	c=0
	for line in f:	
		
		if read_rating ==1:
			#print line[0],
			c=int(line[0])
			#print c,
			if c<3:
				review[1]=0
			else:
				review[1]=1
			read_rating=0
		elif read_review ==1:
			if review[0]==None:
				review[0]=line
			else:
				review[0]+=(line)
		
		if line=='<rating>\n':
			read_rating=1
		elif line=='<review_text>\n':
			read_review=1
		elif line=='</review_text>\n':
			read_review=0
			review[0]=review[0][0:len(review[0])-16]
			data.append(review)
			#print review
			review=[None,None]
		
	return data
		
#----------------------------------------------------------------------------

def add_matrix(fs1,fs2):
	
	#print len(fs1),len(fs2)
	#print len(fs1[0]),len(fs2[0])
	#print 'entered adding fs1,fs2'
	for i in range(len(fs1)):
		fs1[i]+=fs2[i]
	#print len(fs1[0])
	return fs1


#----------------------------------------------------------------------------	


def predict(data,target,n=1,wt=15,noun_count=100,parameter=2,p2=2,p3=3,binn=0,num_features=-1):
	
	
	if parameter==1:
		fe_lexical_matrix=lexical(data)
		s='Using FS1(lexical)'
		statistics(fe_lexical_matrix,target,p3,s,p2=0)
		
	elif parameter==2:
		fs_ngram_matrix=fs_ngram(data,n,15)
		s='Using FS2 (Ngram Count)'
		fs_ngram_matrix = select(fs_ngram_matrix,target,p2,num_features)
		statistics(fs_ngram_matrix,target,p3,s,p2)
	elif parameter==3:
		fs_swn_matrix=swn_features(data)
		s='Using FS3(SWN)'
		fs_swn_matrix=select(fs_swn_matrix,target,p2,num_features)
		statistics(fs_swn_matrix,target,p3,s,p2)
	elif parameter==4:
		noun_matrix=aspect_extract(data,noun_count)
		s='Using FS4(Aspect Extraction)'
		noun_matrix = select(noun_matrix,target,p2,num_features)
		statistics(noun_matrix,target,p3,s,p2)
	elif parameter==5:
		fs_swn_matrix=swn_features(data)
		noun_matrix=aspect_extract(data,noun_count)
		matrix34=add_matrix(fs_swn_matrix,noun_matrix)
		s='Using FS3(SWN) + FS4(Aspect Extraction)'
		matrix34 = select(matrix34,target,p2,num_features)
		statistics(matrix34,target,p3,s,p2)
	elif parameter==6:
		fs_ngram_matrix=fs_ngram(data,n,wt)
		fs_swn_matrix=swn_features(data)
		noun_matrix=aspect_extract(data,noun_count)
		matrix34=add_matrix(fs_swn_matrix,noun_matrix)
		matrix234=add_matrix(fs_ngram_matrix,matrix34)
		matrix234 = select(matrix234,target,p2,num_features)
		s='Using FS2(Simple Ngram Count) + FS3(SWN) + FS4(Aspect Extraction)'
		statistics(matrix234,target,p3,s,p2)
	elif parameter==7:
		fe_lexical_matrix=lexical(data)
		fs_ngram_matrix=fs_ngram(data,n,wt)
		fs_swn_matrix=swn_features(data)
		noun_matrix=aspect_extract(data,noun_count)
		matrix34=add_matrix(fs_swn_matrix,noun_matrix)
		matrix234=add_matrix(fs_ngram_matrix,matrix34)
		matrix1234=add_matrix(fe_lexical_matrix,matrix234)
		s='Using FS1(Lexical Features) + FS2(Simple Ngram Count) + FS3(SWN) + FS4(Aspect Extraction)'
		matrix1234 = select(matrix1234,target,p2,num_features)
		statistics(matrix1234,target,p3,s,p2)

	elif parameter==8:
		fs_tfidf_matrix = tfidf(data,1,tf='b',idf='d')
		msg='Using FS3(tfidf) with tf=b, idf=default bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==9:
		fs_tfidf_matrix = tfidf(data,1,tf='b',idf='s')
		msg='Using FS3(tfidf) with tf=b, idf=s bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==10:
		fs_tfidf_matrix = tfidf(data,1,tf='b',idf='sp')
		msg='Using FS3(tfidf) with tf=b, idf=sp bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==11:
		fs_tfidf_matrix = tfidf(data,1,tf='a',idf='d')
		msg='Using FS3(tfidf) with tf=a, idf=default bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==12:
		fs_tfidf_matrix = tfidf(data,1,tf='a',idf='s')
		msg='Using FS3(tfidf) with tf=a, idf=s bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==13:
		fs_tfidf_matrix = tfidf(data,1,tf='a',idf='sp')
		msg='Using FS3(tfidf) with tf=a, idf=sp bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==14:
		fs_tfidf_matrix = tfidf(data,1,tf='o',idf='d')
		msg='Using FS3(tfidf) with tf=o, idf=default bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==15:
		fs_tfidf_matrix = tfidf(data,1,tf='o',idf='s')
		msg='Using FS3(tfidf) with tf=o, idf=s bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==16:
		fs_tfidf_matrix = tfidf(data,1,tf='o',idf='sp')
		msg='Using FS3(tfidf) with tf=o, idf=sp bin='+str(binn)
		if binn==1:
			fs_tfidf_matrix=bin2(fs_tfidf_matrix)
		fs_tfidf_matrix = select(fs_tfidf_matrix,target,p2,num_features)
		statistics(fs_tfidf_matrix,target,p3,msg,p2)
	elif parameter==17:
		fe_lexical_matrix=vocab(data)
		s='Using FS1.5(lexical Vocabulary)'
		statistics(fe_lexical_matrix,target,p3,s,p2=0)
	else:
		print 'wrong parameter'
		exit(0)
	

def select(matrix,target,p2,num_features=-1):
	
	if p2==0:
		return matrix
	elif p2==1:
		return informationGain(matrix,target,num_features)
	elif p2==2:
		return gainRatio(matrix,target,num_features)
	elif p2==3:
		return chiSquare(matrix,target,num_features)
	elif p2==4:
		return pmi(matrix,target,num_features)
	elif p2==5:
		return cpd(matrix,target,num_features)
	elif p2==6:
		return jsd(matrix,target,num_features)		 
	else:
		print 'Invalid parameter p2'
		exit(0)
		
		
		
		
		
		
		
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def bin2(matrix):
	max=[]
	min=[]
	mid=[]
	new_matrix=[]
	for i in range(len(matrix)):
		F=[]
		for j in range(len(matrix[0])):
			F.append(0)
		new_matrix.append(F)
	
	for i in range(len(matrix[0])):#features
		max.append(0)
		min.append(1000000)
		for j in range(len(matrix)):#review
			if matrix[j][i]>max[i]:
				max[i]=matrix[j][i]
			if matrix[j][i]<min[i]:
				min[i]=matrix[j][i]
	
	#print len(matrix)==len(new_matrix) and len(matrix[0])==len(new_matrix[0])
	
	for i in range(len(matrix[0])):
		mid.append((max[i]+min[i])/2.0)
		for j in range(len(matrix)):
			if matrix[j][i]>mid[i]:
				#print 'change',
				new_matrix[j][i]=1
				
	return new_matrix
	#for i in range(len(min)):
		#print min[i],max[i]

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
