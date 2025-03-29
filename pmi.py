
def pmi(matrix,target,num_features=0):
	fe_scores=[]
	fe_ranking=[]
	new_matrix=[]
	delta=0.00001
	
	for i in range(len(matrix[0])):# i is feature
		a,b,c,d=0,0,0,0
		score=0.0
		fe_ranking.append(i)
		
		for j in range(len(matrix)): #j is class
			
			if matrix[j][i]!=0:
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
		
		#print a,b,c,d
		s1=a/(a+b+delta)/(a+c+delta)
		s2=b/(a+b+delta)/(b+d+delta)
		if s1>s2:
			score=s1
		else:
			score=s2
		
		fe_scores.append(score)
	
	fe_scores, fe_ranking = (list(x) for x in zip(*sorted(zip(fe_scores, fe_ranking),reverse=True)))
	
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
