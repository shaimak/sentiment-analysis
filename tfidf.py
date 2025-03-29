import nltk
from numpy import *
import math

#
#
#
#

def tfidf(data,n,tf='b',idf='d'):
	fe_ngram=[]	# all ngrams
	#fs_ngram=[] # selected ngrams
	tf_scores=[]
	tfidf_ngram=[]
	tfidf_scores=[]
	tf_matrix=[]
	
	#Collecting all ngrams
	ngram_array=[]
	N=len(data)
	avg_word_count=0
	for i in data:
		ngram_review=[]
		review=i[0]
		tokens=nltk.word_tokenize(review.lower())
		tokens_alpha=[t for t in tokens if t.isalpha()]
		text=nltk.Text(tokens_alpha)
		ngram_review=nltk.ngrams(text,n)
		ngram_array+=ngram_review
	#print 'finished collecting ngrams '
	#Using the text as a whole
	fdist_ngram=nltk.FreqDist(ngram_array)
	fe_ngram=fdist_ngram.keys() # all ngrams
	avg_word_count=len(ngram_array)/(0.0+len(data))
	
	'''Filling df,df1,df2 arrays'''
	#fill
	df_array=[]
	df1_array=[]
	df2_array=[]
	dl_array=[]
	N1,N2=0,0
	max_tf=0
	for i in range(len(fe_ngram)):
		df_array.append(0.0)
		df1_array.append(0.0)
		df2_array.append(0.0)
		#dl_array.append(0.0)
		
	for i in data:# for each review
		ngram_review=[]
		review=i[0]
		rating=i[1]
		#print rating
		if rating>0:
			N1+=1
		else:
			N2+=1
		tokens=nltk.word_tokenize(review.lower())
		tokens_alpha=[t for t in tokens if t.isalpha()]
		
		text=nltk.Text(tokens_alpha)
		ngram_review=nltk.ngrams(text,n)
		fdist_review=nltk.FreqDist(ngram_review)
		keys_review=fdist_review.keys()
		if fdist_review[keys_review[0]]>max_tf:
			max_tf=fdist_review[keys_review[0]]
		
		
		for j in range(len(fe_ngram)): #j is the feature
			if fe_ngram[j] in ngram_review:
				df_array[j]+=1				
				if rating >0:
					df1_array[j]+=1
				else:
					df2_array[j]+=1
	
	#for other cases which require to calculate df,df1,df2
	if tf=='b':
		for i in data: #i[0] is review
			F=[]
			review=i[0]
			tokens=nltk.word_tokenize(review.lower())
			tokens_alpha=[t for t in tokens if t.isalpha()]
			text=nltk.Text(tokens_alpha)
			ngram_review=nltk.ngrams(text,n)
			fdist_review=nltk.FreqDist(ngram_review)
			
			for j in range(len(fe_ngram)):
				Tfidf_f=fdist_review[fe_ngram[j]]*(math.log(N,2)-math.log(df_array[j],2)+0.0)
				if Tfidf_f>0:
					F.append(1)
				else:
					F.append(0)
			tf_matrix.append(F)
		
		if idf=='d':
			return tf_matrix
		elif idf=='s':
			for i in range(len(fe_ngram)): #for each feature
				Delta=math.log(N1*df2_array[i]+0.5,2)-math.log(N2*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=Delta
			
			return tf_matrix
		elif idf=='sp':
			for i in range(len(fe_ngram)): #for each feature
				DeltaSP=math.log((N1-df1_array[i])*df2_array[i]+0.5,2)-math.log((N2-df2_array[i])*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=DeltaSP
			
			return tf_matrix
		else:
			print 'No valid IDF weighting entered'
			exit(2)
	
	elif tf=='a':
		for i in data: #i[0] is review
			F=[]
			review=i[0]
			tokens=nltk.word_tokenize(review.lower())
			tokens_alpha=[t for t in tokens if t.isalpha()]
			text=nltk.Text(tokens_alpha)
			ngram_review=nltk.ngrams(text,n)
			fdist_review=nltk.FreqDist(ngram_review)
			for j in range(len(fe_ngram)):
				augmented=0.5+0.5*fdist_review[fe_ngram[j]]/max_tf
				F.append(augmented)
			tf_matrix.append(F)
			
		if idf=='d':
			return tf_matrix
		elif idf=='s':
			for i in range(len(fe_ngram)): #for each feature
				Delta=math.log(N1*df2_array[i]+0.5,2)-math.log(N2*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=Delta
			
			return tf_matrix
		elif idf=='sp':
			for i in range(len(fe_ngram)): #for each feature
				DeltaSP=math.log((N1-df1_array[i])*df2_array[i]+0.5,2)-math.log((N2-df2_array[i])*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=DeltaSP
			
			return tf_matrix
		else:
			print 'No valid IDF weighting entered'
			exit(2)
	elif tf=='o':
		for i in data: #i[0] is review
			F=[]
			review=i[0]
			tokens=nltk.word_tokenize(review.lower())
			tokens_alpha=[t for t in tokens if t.isalpha()]
			text=nltk.Text(tokens_alpha)
			ngram_review=nltk.ngrams(text,n)
			fdist_review=nltk.FreqDist(ngram_review)
			for j in range(len(fe_ngram)):
				g=fdist_review[fe_ngram[j]]
				o_bm25=(2.2*g)/(1.2*(.05+.95*len(ngram_review)/avg_word_count) + g)
				F.append(o_bm25)
			tf_matrix.append(F)
		if idf=='d':
			return tf_matrix
		elif idf=='s':
			for i in range(len(fe_ngram)): #for each feature
				Delta=math.log(N1*df2_array[i]+0.5,2)-math.log(N2*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=Delta
			
			return tf_matrix
		elif idf=='sp':
			for i in range(len(fe_ngram)): #for each feature
				DeltaSP=math.log((N1-df1_array[i])*df2_array[i]+0.5,2)-math.log((N2-df2_array[i])*df1_array[i]+0.5,2)
				for j in range(len(tf_matrix)):
					tf_matrix[j][i]*=DeltaSP
			return tf_matrix
		else:
			print 'No valid IDF weighting entered'
			exit(2)
#--------------------------------------------------------------------------

##########################################################################################################################################




if __name__ == "__main__":
	import nltk
	from sklearn.naive_bayes import GaussianNB
	#import sklearn
	#------------------------------------------
	from process_data import *
	from statistics import *
	from sklearn.metrics import precision_recall_fscore_support
	#from sklearn.metrics import accuracy_score
	from sklearn.cross_validation import train_test_split
	#from numpy  import *
	#------------------------------------------
	
	#wr=open('logs.txt','w')
	path='' #not using now
	filename1='full.review'
	filename2='test.review'
	n,wt,noun_count=1,15,50
	scheme='fixed_features'
	data=[]
	fe_lexical=[]
	noun_array=[]
	data = process(filename1)
	data= data[:5]+data[-5:]
	print len(data),'reviews in all\n','-'*30
	gnb = GaussianNB()
	fs_tfidf_matrix=[]
	s=''
	size=len(data)
	f=open('logs.txt','w')
	f.close()
	''' All Functions '''
	#------------------------------------------------------------------
	
	target=[]
	for i in data:
		target.append(i[1])
	#print len(target),len(fs_tfidf_matrix)
	#------------------------------------------------------------------
	
	#---------------------------------------------------------------------------------------------
	#---------------------------------------------------------------------------------------------
	
	fs_tfidf_matrix = tfidf(data,1,tf='b',idf='d')
	msg='Using FS3(tfidf) with tf=b, idf=default'
	statistics(fs_tfidf_matrix,target,msg)
	
	fs_tfidf_matrix = tfidf(data,1,tf='b',idf='s')
	msg='Using FS3(tfidf) with tf=b, idf=s'
	statistics(fs_tfidf_matrix,target,msg)
	
	fs_tfidf_matrix = tfidf(data,1,tf='b',idf='sp')
	msg='Using FS3(tfidf) with tf=a, idf=default'
	statistics(fs_tfidf_matrix,target,msg)
	
	
	fs_tfidf_matrix = tfidf(data,1,tf='a',idf='d')
	msg='Using FS3(tfidf) with tf=a, idf=default'
	statistics(fs_tfidf_matrix,target,msg)
	
	fs_tfidf_matrix = tfidf(data,1,tf='a',idf='s')
	msg='Using FS3(tfidf) with tf=a, idf=s'
	statistics(fs_tfidf_matrix,target,msg)
	
	fs_tfidf_matrix = tfidf(data,1,tf='a',idf='sp')
	msg='Using FS3(tfidf) with tf=a, idf=sp'
	statistics(fs_tfidf_matrix,target,msg)
	
	fs_tfidf_matrix = tfidf(data,1,tf='o')
	msg='Using FS3(tfidf) with tf=o'
	statistics(fs_tfidf_matrix,target,msg)
