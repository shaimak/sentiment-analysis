

def chiSquare(matrix,target,num_features=-1):
	fe_scores=[]
	fe_ranking=[]
	new_matrix=[]
	delta=0.000001
	
	#both X(f,c) will be same since we have binary cases in classes
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
		if a*d-b*c==0:
			#print a,b,c,d
			score=0
		else:
			score= (a*d-b*c)*(a*d-b*c)/(a+c+delta)/(b+d+delta)/(a+b+delta)/(c+d+delta)/1.0
			#print score
		fe_scores.append(score)
	
	fe_scores, fe_ranking = (list(x) for x in zip(*sorted(zip(fe_scores, fe_ranking),reverse=True)))
	#for i in range(len(fe_scores)):
		#if fe_scores[i]>0:
		#print fe_scores[i],fe_ranking[i]
	
	fe_ranking=fe_ranking[:num_features]
	'''Creating and filling the new matrix'''
	
	for i in range(len(matrix)):#i is review
		F=[]
		for j in range(len(matrix[0])):
			if j in fe_ranking:
				F.append(matrix[i][j])
		new_matrix.append(F)
	
	print len(new_matrix),len(new_matrix[0])
	return new_matrix
	
