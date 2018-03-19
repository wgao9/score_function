import numpy as np
import scipy.spatial as ss
from math import exp
N = 1000
d = 2

def S_4_estimator(x, y, k=100, bw=1.0, adaptive_bw = True, tr=999):

    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    assert tr <= len(x)-1, "Set tr smaller than num.samples - 1"
    assert len(x) == len(y), "Set num.x = num.y"
    N = len(x)
    d = len(x[0])

    S_4 = np.zeros((d,d,d,d))
    tree = ss.cKDTree(x)
    
    for i in range(N):
      lists = tree.query(x[i], k+1, p=2)
      if adaptive_bw:
	  bw = lists[0][k]
#      list_knn = lists[1][1:tr+1]
      M_0 = 0
      M_1 = np.matrix(np.zeros((d,1)))
      M_2 = np.matrix(np.zeros((d,d)))

      for neighbor in range(N):
	if neighbor == i:
		continue
        dis = np.matrix(x[neighbor] - x[i]).reshape(d,1)
        M_0 += exp(-dis.transpose()*dis/(2*bw**2))
        M_1 += (dis)*exp(-dis.transpose()*dis/(2*bw**2))
        M_2 += (dis*dis.transpose())*exp(-dis.transpose()*dis/(2*bw**2))
        
      mu = M_1/M_0
      Sigma = M_2/M_0 - M_1*M_1.transpose()/(M_0**2)
      Sigma_inv = np.linalg.inv(Sigma)
      a_1 = Sigma_inv*mu
      A_2 = Sigma_inv - bw**(-2)*np.identity(d) 
     
      for j_1 in range(d): 
        for j_2 in range(d):
          for j_3 in range(d):
            for j_4 in range(d):
              sums = 0
              sums += a_1[j_1]*a_1[j_2]*a_1[j_3]*a_1[j_4]
              sums -= A_2[j_1,j_2]*a_1[j_3]*a_1[j_4]
              sums -= A_2[j_1,j_3]*a_1[j_2]*a_1[j_4]
              sums -= A_2[j_1,j_4]*a_1[j_2]*a_1[j_3]
              sums -= A_2[j_2,j_3]*a_1[j_1]*a_1[j_4]
              sums -= A_2[j_2,j_4]*a_1[j_1]*a_1[j_3]
              sums -= A_2[j_3,j_4]*a_1[j_1]*a_1[j_2]
              sums += A_2[j_1,j_2]*A_2[j_3,j_4]
              sums += A_2[j_1,j_3]*A_2[j_2,j_4]
              sums += A_2[j_1,j_4]*A_2[j_2,j_3]
              S_4[j_1,j_2,j_3,j_4] += (1.0/N)*y[i]*sums
    return S_4
          
def S_4_gaussian(x, y):
    assert len(x) == len(y), "Set num.x = num.y"
    N = len(x)
    d = len(x[0])

    S_4 = np.zeros((d,d,d,d))
    for i in range(N):
        for j_1 in range(d):
     	    for j_2 in range(d):
	        for j_3 in range(d):
		    for j_4 in range(d):
		        sums = 0.0
 		    	sums += x[i,j_1]*x[i,j_2]*x[i,j_3]*x[i,j_4]
			sums -= x[i,j_1]*x[i,j_2]
		    	sums -= x[i,j_1]*x[i,j_3]
		    	sums -= x[i,j_1]*x[i,j_4]
		    	sums -= x[i,j_2]*x[i,j_3]
		    	sums -= x[i,j_2]*x[i,j_4]
		    	sums -= x[i,j_3]*x[i,j_4]
		    	sums += 3
		    	S_4[j_1, j_2, j_3, j_4] += (1.0/N)*y[i]*sums
    return S_4
    
def S_4_value(S_4, b_1, b_2, b_3, b_4):
    d = len(S_4)
    sums = 0
    for j_1 in range(d):
      for j_2 in range(d):
        for j_3 in range(d):
          for j_4 in range(d):
            sums += S_4[j_1,j_2,j_3,j_4]*b_1[j_1]*b_2[j_2]*b_3[j_3]*b_4[j_4]
    return sums

x = np.random.normal(0.0,1.0,size=(N,d))
y = x[:,0]**4
b_1 = np.array([1,0])
b_2 = np.array([1,0])
b_3 = np.array([1,0])
b_4 = np.array([1,0])
hat_S_4 = S_4_estimator(x,y)
true_S_4 = S_4_gaussian(x,y)
print(hat_S_4)
print(true_S_4)
print(S_4_value(hat_S_4, b_1, b_2, b_3, b_4))
print(S_4_value(true_S_4, b_1, b_2, b_3, b_4))
