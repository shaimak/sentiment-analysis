import nltk
import re
import math
def lexical(data):
	lexical_array=[]	#length 251, 1st f0 has ID
	ID=0
	
	'''Function Words -150 in number'''
	
	function_words=['a','about','above','after','all','although','am','among','an','and','another','any','anybody','anyone','anything','are','around','as','at','be','because','before','behind','below','beside','between','both','but','by','can','cos','do','down','each','either','enough','every','everybody','everyone','everything','few','following','for','from','have','he','her','him','i','if','in','including','inside','into','is','it','its','latter','less','like','little','lots','many','me','more','most','much','must','my','near','need','neither','no','nobody','none','nor','nothing','of','off','on','once','one','onto','opposite','or','our','outside','over','own','past','per','plenty','plus','regarding','same','several','she','should','since','so','some','somebody','someone','something','such','than','that','the','their','them','these','they','this','those','though','through','till','to','toward','towards','under','unless','unlike','untill','up','upon','us','used','via','we','what','whatever','when','where','whether','which','while','who','whoever','whom','whose','will','with','within','without','worth','would','yes','you','your']
	
	
	
	'''filling up all 251 features for each review'''
	for i in data:
		prev_c='w'
		F=[]
		M1,M2=0.0,0.0
		fx,v2=0,0
		for j in range(251):
			F.append(0)
		
		review=i[0]
		ID=0
		F[0]=ID #f0
		F[1]=len(review)
		NL=0 #new lines
		NP=0 #new Paragraphs
		#Filling up f2-f53
		for c in review:
			
			if c.isalpha():
				F[2]+=1
				if c.isupper():
					F[3]+=1
				
				d=c.upper()
				#print d,
				F[ord(d)-58]+=1
				#print d,ord(d)-58,'-',
			
			elif c.isdigit():
				F[4]+=1
			elif c==' ':
				F[5]+=1
			elif c=='	':
				F[6]+=1
			else:
				e=ord(c)
				if e in [35,36,37,38]:
					F[e-2]+=1
				elif e in [42,43]:
					F[e-5]+=1
				elif e == 45:
					F[39]+=1
				elif e == 47:
					F[40]+=1
				elif e in [60,61,62]:
					F[e-19]+=1
				elif e == 64:
					F[44]+=1
				elif e in [91,92,93,94,95]:
					F[e-46]+=1
				elif e in [123,124,125,126]:
					F[e-73]+=1
						
				elif e == 10:									#' \n '
					NL+=1
					if ord(prev_c)==10:
						NP+=1
				elif e == 46:									#' . '
					F[88]+=1
				elif e == 44: 								#' , '
					F[89]+=1
				elif e == 63:									#' ? '
					F[90]+=1
				elif e == 33:									#' ! '
					F[91]+=1
				elif e == 59:									#' ; '
					F[92]+=1
				elif e == 58:									#' : '
					F[93]+=1
				elif e == 39:									#" ' "
					F[94]+=1
				elif e == 34:									#' " '
					F[95]+=1
					
					#print c
			prev_c=c
#-------------------------------------------------------------------------------------		

		for k in range(54):
			if k!=0 and k!=1:
				F[k]/=F[1]+0.0		
		#print F[:54]
		tokens=nltk.word_tokenize(review.lower())
		tokens_alpha=[k for k in tokens if k.isalpha()]
		text_alpha=nltk.Text(tokens_alpha)
		fdist_alpha=nltk.FreqDist(text_alpha)
		text=nltk.Text(tokens)
		fdist=nltk.FreqDist(text)
		keys=fdist_alpha.keys()
		F[54]=len(tokens_alpha)
		w=[w for w in tokens if len(w)<4]
		F[55]=len(w)
		
#-------------------------------------------------------------------------------------
		for t in tokens_alpha:
			F[56]+=len(t)
		
		F[57]=F[56]/(len(tokens_alpha)+0.0)
		F[58]=len(review)/(NL+1.0)
		F[59]=len(tokens)/(NL+1.0)
		F[60]=len(fdist.keys())/(F[54]+0.0)
		F[61]=len(fdist.hapaxes())/(len(tokens)+0.0)
		#f62
		for t in tokens_alpha:
			if t.isalpha():
				#print t,len(t),'-',
				if len(t) == 1:
					F[68]+=1
				elif len(t) == 2:
					F[62]+=1
					F[69]+=1
				elif len(t) == 3:
					F[70]+=1
				elif len(t) == 4:
					F[71]+=1
				elif len(t) == 5:
					F[72]+=1
				elif len(t) == 6:
					F[73]+=1
				elif len(t) == 7:
					F[74]+=1
				elif len(t) == 8:
					F[75]+=1
				elif len(t) == 9:
					F[76]+=1
				elif len(t) == 10:
					F[77]+=1
				elif len(t) == 11:
					F[78]+=1
				elif len(t) == 12:
					F[79]+=1
				elif len(t) == 13:
					F[80]+=1
				elif len(t) == 14:
					F[81]+=1
				elif len(t) == 15:
					F[82]+=1
				elif len(t) == 16:
					F[83]+=1
				elif len(t) == 17:
					F[84]+=1
				elif len(t) == 18:
					F[85]+=1
				elif len(t) == 19:
					F[86]+=1
				elif len(t) == 20:
					F[87]+=1
				
				
				#print t.lower()
				for l in range(150):
					if t.lower() == function_words[l]:
						F[l+96]+=1.0/F[54]
						break
			
#------------------------------------------------------------------------		
		for v in range(20):
			F[v+68]/=F[54]+0.0
		
		sentences=re.split(r'[.!?]+',review)		
		for sen in sentences:
			if len(sen)>2:
				F[247]+=1
		F[246]=NL+1
		F[248]=NP+1
		F[249]=F[247]/(F[248]+0.0)
		F[250]=F[1]/(F[248]+0.0)
		M1=F[54]
		maximum_freq=fdist[keys[0]]
		#print maximum_freq,keys[0]
#------------------------------------------------------------------------		
		for u in range(maximum_freq+1):
			fx=0
			for k in keys:
				if fdist[k]==u:
					 fx+=1
			M2+=fx*u*u
		F[63]=100*(M2-M1)/(M1*M1*1.0) #yule's K measure : Doubt: the lower the better
			
		for r in range(maximum_freq+1):
			vr=0
			if r!=0:
				for k in keys:
					if fdist[k] == r:
						vr+=1
			F[64]+=vr*r*(r-1)/ ( (len(tokens_alpha)+0.001)*(len(tokens_alpha)-1.0001) ) # Simpson's D measure
		
		for k in keys:
			if fdist[k] == 2:
				v2+=1
				#print k
		
		
		F[65]=v2/(len(keys)+0.0)  # Sichel's S measure
		
		F[66]=pow(M1,pow(len(keys),-0.17)) #Brunet's W measure
		#print len(fdist_alpha.hapaxes())
		if len(keys)==0:
			print 'oops',i
		#else:
			#print len(keys),
		F[67]=1
		#100*math.log(len(tokens_alpha))/(1.0- len(fdist_alpha.hapaxes())/(len(keys)+0.0001))
		 
		F[2]/=F[1]+0.0
		F[3]/=F[1]+0.0
		F[4]/=F[1]+0.0
		F[5]/=F[1]+0.0
		F[6]/=F[1]+0.0
		
		F[55]/=F[54]+0.0
		F[56]/=F[1]+0.0
		F[1]/=1000 #normalising
		F[250]/=500 #normalising
		F[58]/=400 #normalising
		F[57]/=70
		F[62]/=28
		F[59]/=79
		F[54]/=160
		#print F[3]
		#print F[63:68]
		if F[2]>1:
			print F[2]
		lexical_array.append(F)
		
	return lexical_array
	
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------



def vocab(data):
	fe_lexical_vocab=[]
	
	
	for i in data:
		V=[] #vocab_array
		for j in range(5):
			V.append(0)
		
		M1,M2=0.0,0.0
		fx,v2=0,0
		review=i[0]
		tokens=nltk.word_tokenize(review.lower())
		tokens_alpha=[k for k in tokens if k.isalpha()]
		text_alpha=nltk.Text(tokens_alpha)
		fdist_alpha=nltk.FreqDist(text_alpha)
		keys=fdist_alpha.keys()
		maximum_freq=fdist_alpha[keys[0]]
		M1=len(tokens_alpha)
		


		for u in range(maximum_freq+1):
			fx=0
			for k in keys:
				if fdist_alpha[k]==u:
					 fx+=1
			M2+=fx*u*u
		V[0]=100*(M2-M1)/(M1*M1*1.0) #yule's K measure : Doubt: the lower the better
			
		for r in range(maximum_freq+1):
			vr=0
			if r!=0:
				for k in keys:
					if fdist_alpha[k] == r:
						vr+=1
			#if len(tokens_alpha) ==0 or len(tokens_alpha)==1:
				#print 'true problem',tokens_alpha
			V[1]+=vr*r*(r-1)/ ( (len(tokens_alpha)+0.0001)*(len(tokens_alpha)-0.999) ) # Simpson's D measure
		
		for k in keys:
			if fdist_alpha[k] == 2:
				v2+=1
				#print k
		
		
		V[2]=v2/(len(keys)+0.0)  # Sichel's S measure
		if len(keys)==0:
			print i
		V[3]=pow(M1,pow(len(keys),-0.17)) #Brunet's W measure
		if len(keys)==0:
			print 'found'
		V[4]=100*math.log(len(tokens_alpha))/(1.0- len(fdist_alpha.hapaxes())/(len(keys)+0.001))
		fe_lexical_vocab.append(V)
	
	return fe_lexical_vocab

	
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
	
	fe_lexical_matrix=lexical(data)
	s='Using FS1(lexical)'
	statistics(fe_lexical_matrix,target,s)
	
	fe_lexical_matrix=vocab(data)
	s='Using FS1.5(lexical Vocabulary)'
	statistics(fe_lexical_matrix,target,s)
	
