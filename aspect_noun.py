import nltk
from nltk.corpus import stopwords
from nltk.tag.simplify import simplify_wsj_tag



def aspect_extract(data,noun_count=-1):
	fe_nouns=[]
	fe_nouns_scores=[]
	unigram_array=[]
	
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
	fe_nouns=[word for word,tag in test_tagged if tag=='N' or tag=='NP' or tag=='FW'] # foreign word or place, or technical word
	fe_nouns_scores=[fdist_unigram[score] for score in fe_nouns]
	fe_nouns_scores, fe_nouns = (list(x) for x in zip(*sorted(zip(fe_nouns_scores, fe_nouns),reverse=True)))
	
	#print fe_nouns[:10]
	noun_array=fe_nouns[:noun_count]
	
	'''Filling the matrix'''
	noun_matrix=[]
	for i in data:
		F=[0 for x in range(len(noun_array))]
		review=i[0]
		review=review.lower()
		tokens=nltk.word_tokenize(review)
		unigrams=[k for k in tokens if k.isalpha()]
		
		for j in range(len(noun_array)):
			if noun_array[j] in unigrams:
				F[j]+=1
		noun_matrix.append(F)
		
	#print noun_matrix
	#c=[]
	#for i in range(len(noun_array)):
		#c.append(0)
		#for j in range(len(noun_matrix)): 
			#if noun_matrix[j][i]!=0:
				#c[i]+=1
	#print c
	return noun_matrix



##########################################################################################################################################




if __name__ == "__main__":
	from nltk.tag.simplify import simplify_wsj_tag
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
	noun_count=-1
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
		
		
	noun_matrix=aspect_extract(data,noun_count)
	s='Using FS4(Aspect Extraction)'
	statistics(noun_matrix,target,s)
