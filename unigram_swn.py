from sentiwordnet import *
from nltk.corpus import stopwords
import nltk
from nltk.tag.simplify import simplify_wsj_tag

def swn_features(data):
	swn = SentiWordNetCorpusReader("SentiWordNet_3.0.0.txt")
	N=len(data)
	fe_unigram=[]
	unigram_array=[]
	fs_swn=[]
	fs_score=[]
	
	for i in data:
		unigram_review=[]
		review=i[0]
		review=review.lower()
		tokens=nltk.word_tokenize(review)
		tokens_alpha=[k for k in tokens if k.isalpha() and k not in stopwords.words('english')]
		unigram_review=tokens_alpha # not really copying
		unigram_array+=unigram_review
	
	text_unigram=nltk.Text(unigram_array)
	fdist_unigram=nltk.FreqDist(text_unigram)
	fe_unigram=fdist_unigram.keys()
	
	
	
	test_tagged=nltk.pos_tag(fe_unigram)
	test_tagged = [(word, simplify_wsj_tag(tag)) for word, tag in test_tagged]
	
	
	
	'''Using Sentiment subjectivity scoring to fill fs_swn''' 
	for i in fe_unigram:
		score_list=[]
		append=0
		
		# adjective		
		score_list= swn.senti_synsets(i,'a')
		if len(score_list)>0:
			total_positive=0.0
			total_negative=0.0
			for k in score_list:
				total_positive+=k.pos_score
				total_negative+=k.neg_score
			pos_score = total_positive/(0.0+len(score_list))
			neg_score = total_negative/(0.0+len(score_list))
			if (pos_score+neg_score) >=0.5 and append==0:
				fs_swn.append(i)
				append=1
				if pos_score>neg_score:
					fs_score.append(pos_score)
				else:
					fs_score.append(0.0-neg_score)
		
		#adverb
		score_list= swn.senti_synsets(i,'r')
		if len(score_list)>0:
			total_positive=0.0
			total_negative=0.0
			for k in score_list:
				total_positive+=k.pos_score
				total_negative+=k.neg_score
			pos_score = total_positive/(0.0+len(score_list))
			neg_score = total_negative/(0.0+len(score_list))
			if (pos_score+neg_score) >=0.5 and append==0:
				fs_swn.append(i)
				append=1
				if pos_score>neg_score:
					fs_score.append(pos_score)
				else:
					fs_score.append(0.0-neg_score)
		
		#Verb
		score_list= swn.senti_synsets(i,'v')
		if len(score_list)>0:
			total_positive=0.0
			total_negative=0.0
			for k in score_list:
				total_positive+=k.pos_score
				total_negative+=k.neg_score
			pos_score = total_positive/(0.0+len(score_list))
			neg_score = total_negative/(0.0+len(score_list))
			if (pos_score+neg_score) >=0.5 and append==0:
				fs_swn.append(i)
				append=1
				if pos_score>neg_score:
					fs_score.append(pos_score)
				else:
					fs_score.append(0.0-neg_score)
			
		

	
	
	'''Filling up the new matrix FS3'''
	#print 'Filling up the new matrix FS3'
	unigram_array=[]
	num_reviews=len(data)
	num_features=len(fs_swn)
	
	for i in data: #i[0] is review
		F=[]
		review=i[0]
		review=review.lower()
		tokens=nltk.word_tokenize(review)
		tokens_alpha=[k for k in tokens if k.isalpha() and k not in stopwords.words('english')]
		#tagged_words=nltk.pos_tag(tokens_alpha)
		#tagged_words = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_words]
		#print tagged_words[2]
		for j in range(len(fs_swn)): #fs_swn[j] is feature
			if fs_swn[j] in tokens_alpha:
				F.append(fs_score[j])
			else:
				F.append(0)
				#print j
		unigram_array.append(F)
		
	#c=[]
	#for i in range(len(fs_swn)):
		#c.append(0)
		#for j in range(len(unigram_array)): 
			#if unigram_array[j][i]!=0:
	    	#c[i]+=1
	#print c
	
	#print unigram_array
	return unigram_array
	
	
	
	
	
##################################################################################################################################


if __name__=='__main__':
	
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
	data = process(filename)
	data= data[:20]+data[-20:]
	print len(data),'reviews in all\n','-'*30
	gnb = GaussianNB()
	f=open('logs.txt','w')
	f.close()
	target=[]
	#------------------------------------------------------------------
	for i in data:
		target.append(i[1])
	#------------------------------------------------------------------
	
	fs_swn_matrix=swn_features(data)
	s='Using FS3(SWN)'
	statistics(fs_swn_matrix,target,s)
	
