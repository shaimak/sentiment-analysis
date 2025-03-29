import math

def jsd(matrix,target,num_features=-1):
	fe_jsd=[]
	fe_ranking=[]
	new_matrix=[]
	n1,n2=0.0,0.0
	delta=0.00001
	
	for i in target:
		if i==1:
			n1+=1
		else:
			n2+=1
	p1=n1/(n1+n2+delta)
	p2=n2/(n1+n2+delta)
	entropy_C=-p1*math.log(p1,2)-p2*math.log(p2,2)
	
	for i in range(len(matrix[0])):# i is feature
		
		a,b,c,d=0,0,0,0
		cf1,cf2,cf3,cf4=0.0,0.0,0.0,0.0
		fc1,fc2,fc3,fc4=0.0,0.0,0.0,0.0
		fe_ranking.append(i)
		entropy_CF_P=0.0
		entropy_CF_NP=0.0
		entropy_FC_P=0.0
		entropy_FC_NP=0.0
		entropy_F=0.0
		
		for j in range(len(matrix)): #j is class	
			if matrix[j][i]!=0:#if feature is present
				if target[j]==1:
					a+=1
				else:
					b+=1
			else:
				if target[j]==1:
					c+=1
				else:
					d+=1
		if a+b+c+d != len(matrix):
			print 'Error in calculation',j,i
		
		#Entropy(C/f)
		cf1=a/(a+b+delta)
		cf2=b/(a+b+delta)
		cf3=c/(c+d+delta)
		cf4=d/(c+d+delta)
		n1=(a+b)/(a+b+c+d+0.0)
		n2=(c+d)/(a+b+c+d+0.0)
		entropy_CF_P  =-cf1*math.log(cf1+delta,2) - cf2*math.log(cf2+delta,2)
		entropy_CF_NP = -cf3*math.log(cf3+delta,2) - cf4*math.log(cf4+delta,2)
		entropy_CF=n1*entropy_CF_P + n2*entropy_CF_NP
		
		#Entropy(f/C)
		fc1=a/(a+c+delta)
		fc2=c/(a+c+delta)
		fc3=b/(b+d+delta)
		fc4=d/(b+d+delta)
		n3=(a+c)/(a+b+c+d+0.0)
		n4=(b+d)/(a+b+c+d+0.0)
		entropy_FC_P  =-fc1*math.log(fc1+delta,2) - fc2*math.log(fc2+delta,2)
		entropy_FC_NP = -fc3*math.log(fc3+delta,2) - fc4*math.log(fc4+delta,2)
		entropy_FC=n3*entropy_FC_P + n4*entropy_FC_NP
		
		#Enropy(f)
		entropy_F=-n1*math.log(n1+delta,2) - n2*math.log(n2+delta,2)
		
		
		#Score
		score= 0.5*(entropy_CF - entropy_C) + 0.5*(entropy_FC - entropy_F)
		#print entropy_CF - entropy_C,entropy_FC -entropy_F, score
		fe_jsd.append(score)
	fe_jsd, fe_ranking = (list(x) for x in zip(*sorted(zip(fe_jsd, fe_ranking),reverse=False)))
	
	#for i in range(len(fe_jsd)):
		#print fe_jsd[i],fe_ranking[i]
	
	fe_ranking=fe_ranking[:num_features]
	'''Creating and filling the new matrix'''
	
	for i in range(len(matrix)):#i is review
		F=[]
		for j in range(len(matrix[0])):
			if j in fe_ranking:
				F.append(matrix[i][j])
		new_matrix.append(F)
	
	#print len(new_matrix),len(new_matrix[0])
	return new_matrix
