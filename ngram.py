import nltk
from numpy import *

'''n-gram '''
def fs_ngram(data,n,wt):
	fe_ngram=[]
	fs_ngram=[]
	ngram_array=[]
	for i in data:
		NG=[]
		review=i[0]
		review=review.lower()
		tokens=nltk.word_tokenize(review)
		tokens_alpha=[k for k in tokens if k.isalpha()]
		#print len(review)
		text_i=nltk.Text(tokens_alpha)
		NG=nltk.ngrams(text_i,n)
		#print nltk.unigram
		fe_ngram+=NG
		#print len(NG),len(fe_ngram)
	#----------------------------------------------------------------
	''' Triming down the fe_ngram array based on wts'''
	fdist_ngram=nltk.FreqDist(fe_ngram)
	keys_ngram=fdist_ngram.keys()
	#print len(keys_ngram)
	for i in range(len(keys_ngram)):
		if fdist_ngram[keys_ngram[i]] >=wt:
			fs_ngram.append(keys_ngram[i])
		else:
			break
	if len(fs_ngram)==0:
		print 'maximum wt of',n,'gram is',fdist_ngram[keys_ngram[0]],'\nPick a value less than this'
		exit(0)
	
	#----------------------------------------------------------------
	'''Filling up the new matrix FS2 i.e. lexical_array, Presence wise'''
	print len(fs_ngram),'Features containing ngrams'
	#print 'maximum wt of',n,'gram is',fdist_ngram[keys_ngram[0]]
	#print fs_ngram
	#print NG
	num_reviews=len(data)
	num_features=len(fs_ngram)
	#print number_reviews
	for i in range(num_reviews):
		F=[]
		NG_t=[]
		review_t=data[i][0]
		review_t=review_t.lower()
		tokens_t=nltk.word_tokenize(review_t)
		tokens_alpha_t=[e for e in tokens_t if e.isalpha()]
		text_t=nltk.Text(tokens_alpha_t)
		NG_t=nltk.ngrams(text_t,n)
		#print NG_t
		for t in range(num_features):
			if fs_ngram[t] in NG_t:
				F.append(1)
			else:
				F.append(0)
			#print len(F),
		ngram_array.append(F)
	#ngram_array.append(fs_ngram)
	return ngram_array
	
	


##########################################################################################################################################




if __name__ == "__main__":
	
	from nltk.tag.simplify import simplify_wsj_tag
	import nltk
	from sklearn.naive_bayes import GaussianNB
	#import sklearn
	#------------------------------------------
	from process_data import *
	from statistics import *
	from sklearn.metrics import precision_recall_fscore_support
	#from sklearn.metrics import accuracy_score
	from sklearn.cross_validation import train_test_split
	#------------------------------------------
	
	
	filename='full.review'
	n,wt=1,5
	scheme='fixed_features'
	data = process(filename)
	data= data[:40]+data[-40:]
	print len(data),'reviews in all\n','-'*30
	gnb = GaussianNB()
	size=len(data)
	f=open('logs.txt','w')
	f.close()
	#------------------------------------------------------------------
	
	target=[]
	for i in data:
		target.append(i[1])
	
	#---------------------------------------------------------------------------------------------
	#---------------------------------------------------------------------------------------------
		
		
	fs_ngram_matrix=fs_ngram(data,n,wt)
	s='Using FS2 (Ngram Count)'
	statistics(fs_ngram_matrix,target,s)
