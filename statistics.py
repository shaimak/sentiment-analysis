import numpy as np
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
#----------------------------------------------------------------------------	
def statistics2(matrix,target,p3,msg,p2=0):
	
	#Deciding predicting function
	if p3==1:
		clf=GaussianNB()
		msg3=' and Predicting using gaussianNB'
	elif p3==2:
		clf=LogisticRegression()
		msg3=' and Predicting using Logistic Regression'
	elif p3==3:
		clf = RandomForestClassifier(n_estimators=10)
		msg3=' and Predicting using RandomForestClassifier'
	elif p3==4:
		clf = SVC(kernel='linear')
		msg3=' and Predicting using SVC with linear kernel'
	elif p3==5:
		clf = SVC()
		msg3=' and Predicting using SVC with radial kernel'
	else:
		print 'Invalid parameter p3'
		exit(0)
	
	#Deciding msg2 i.e. selecting msg
	if p2==0:
		msg2=', with No feature selection'
	elif p2==1:
		msg2=', with InformationGain Selection'
	elif p2==2:
		msg2=', with GainRatio Selection'
	elif p2==3:
		msg2=', with ChiSquare Selection'
	elif p2==4:
		msg2=', with Pointwise Mutual information'
	elif p2==5:
		msg2=', with Categorial Proportional Difference'
	elif p2==6:
		msg2=', with KL Divergence'
	else:
		print 'Invalid paramter p2 in statistics function','p3=',p3
		exit(0)
	
	msg+=msg2+msg3
	tp,tn,fp,fn=0.0,0.0,0.0,0.0
	print msg
	with open("logs.txt", "a") as myfile:
		mean_p,mean_r,mean_f,mean_a=0.0,0.0,0.0,0.0
		n_fold=10
		for i in range(n_fold):
			x_train,x_test,y_train,y_test=train_test_split(matrix,target,test_size=.40,random_state=i*100)
			y_pred=clf.fit(x_train,y_train).predict(x_test)
			p,r,f,s=precision_recall_fscore_support(y_test,y_pred,average='macro')
			#a = accuracy_score(y_test, y_pred)
			mean_p+=p/n_fold
			mean_r+=r/n_fold
			mean_f+=f/n_fold
			mat=confusion_matrix(y_test,y_pred)
			tp=mat[0][0]
			tn=mat[1][1]
			fp=mat[0][1]
			fn=mat[1][0]
			a=(tp+tn)/(tp+tn+fp+fn+0.0001)
			mean_a+=a/n_fold
		print mean_p,mean_r,mean_f,mean_a
		myfile.write(str(msg))
		#myfile.write('\n'+'-'*30)	
		#accuracy=(tp+tn)/(tp+tn+fp+fn+.0001)
		#precision=tp/(tp+fp+.0001)
		#recall=tp/(tp+fn+.0001)
		#f_measure=2*precision*recall/(precision+recall+.0001)
	
		myfile.write('\n'+'-'*30+'\n'+'precision ='+str(mean_p)+'\n')
		myfile.write('recall ='+str(mean_r)+'\n')
		myfile.write('accuracy ='+str(mean_a)+'\n')
		myfile.write('F-measure ='+str(mean_f)+'\n')
		myfile.write('-'*60+'\n')
		
		
	
		

		
#----------------------------------------------------------------------------	
def statistics(matrix,target,p3,msg,p2=0):
	
	#Deciding predicting function
	if p3==1:
		clf=GaussianNB()
		msg3=' and Predicting using gaussianNB'
	elif p3==2:
		clf=LogisticRegression()
		msg3=' and Predicting using Logistic Regression'
	elif p3==3:
		clf = RandomForestClassifier(n_estimators=10)
		msg3=' and Predicting using RandomForestClassifier'
	elif p3==4:
		clf = SVC(kernel='linear')
		msg3=' and Predicting using SVC with linear kernel'
	elif p3==5:
		clf = SVC()
		msg3=' and Predicting using SVC with radial kernel'
	else:
		print 'Invalid parameter p3'
		exit(0)
	
	#Deciding msg2 i.e. selecting msg
	if p2==0:
		msg2=', with No Feature selection'
	elif p2==1:
		msg2=', with InformationGain Selection'
	elif p2==2:
		msg2=', with GainRatio Selection'
	elif p2==3:
		msg2=', with ChiSquare Selection'
	elif p2==4:
		msg2=', with Pointwise Mutual information'
	elif p2==5:
		msg2=', with Categorial Proportional Difference'
	elif p2==6:
		msg2=', with KL Divergence'
	else:
		print 'Invalid paramter p2 in statistics function','p3=',p3
		exit(0)
	
	msg+=msg2+msg3

	X = np.array(matrix)
	Y = np.array(target)
	print msg
	n_fold=3
	tp,tn,fp,fn=0.0,0.0,0.0,0.0
	with open("logs.txt", "a") as myfile:
		mean_p,mean_r,mean_f,mean_a=0.0,0.0,0.0,0.0
		skf = cross_validation.StratifiedKFold(Y,n_fold)
		for train_index, test_index in skf:
			#print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			y_pred=clf.fit(X_train,y_train).predict(X_test)
			p,r,f,s=precision_recall_fscore_support(y_test,y_pred,average='macro')
			#a = accuracy_score(y_test, y_pred)
			mean_p+=p/n_fold
			mean_r+=r/n_fold
			mean_f+=f/n_fold
			mat=confusion_matrix(y_test,y_pred)
			tp=mat[0][0]
			tn=mat[1][1]
			fp=mat[0][1]
			fn=mat[1][0]
			a=(tp+tn)/(tp+tn+fp+fn+0.0001)
			mean_a+=a/n_fold
			#print y_test,y_pred,confusion_matrix(y_test,y_pred)
		print mean_p,mean_r,mean_f,mean_a
		myfile.write(str(msg))
		
	
		myfile.write('\n'+'-'*30+'\n'+'precision ='+str(mean_p)+'\n')
		myfile.write('recall ='+str(mean_r)+'\n')
		myfile.write('accuracy ='+str(mean_a)+'\n')
		myfile.write('F-measure ='+str(mean_f)+'\n')
		myfile.write('-'*60+'\n')
		

