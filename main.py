import numpy as np
import scipy.spatial as ss
from math import exp
N = 1000
d = 2

def S_4_estimator(x, y, k=50, tr=99):

    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    assert tr <= len(x)-1, "Set tr smaller than num.samples - 1"
    assert len(x) == len(y), "Set num.x = num.y"
    N = len(x)
    d = len(x[0])

    S_4 = np.zeros((d,d,d,d))
    tree = ss.cKDTree(x)
    
    for i in range(N):
      lists = tree.query(x[i], tr+1, p=2)
      bw = 1.0
      list_knn = lists[1][1:tr+1]
      M_0 = 0
      M_1 = np.matrix(np.zeros((d,1)))
      M_2 = np.matrix(np.zeros((d,d)))

      for neighbor in list_knn:
        dis = np.matrix(x[neighbor] - x[i]).reshape(d,1)
        M_0 += exp(-dis.transpose()*dis/(2*bw**2))
        M_1 += (dis)*exp(-dis.transpose()*dis/(2*bw**2))
        M_2 += (dis*dis.transpose())*exp(-dis.transpose()*dis/(2*bw**2))
        
      mu = M_1/M_0
      Sigma_inv = np.linalg.inv(M_2/M_0 - M_1.transpose()*M_1/(M_0**2))
      a_1 = Sigma_inv*mu
      A_2 = Sigma_inv - bw**(-2)*np.identity(d) 
      print(a_1)
      print(A_2)
      
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
      
def S_4_value(S_4, b_1, b_2, b_3, b_4):
    d = len(S_4)
    sums = 0
    for j_1 in range(d):
      for j_2 in range(d):
        for j_3 in range(d):
          for j_4 in range(d):
            sums += S_4[j_1,j_2,j_3,j_4]*b_1[j_1]*b_2[j_2]*b_3[j_3]*b_4[j_4]
    return sums
    
def S_4_gaussian(x, b_1, b_2, b_3, b_4):
    ret = np.inner(b_1,x)*np.inner(b_2,x)*np.inner(b_3,x)*np.inner(b_4,x)
    ret -= np.inner(b_1,x)*np.inner(b_2,x)*np.inner(b_3,b_4)
    ret -= np.inner(b_1,x)*np.inner(b_3,x)*np.inner(b_2,b_4)
    ret -= np.inner(b_1,x)*np.inner(b_4,x)*np.inner(b_2,b_3)
    ret -= np.inner(b_2,x)*np.inner(b_3,x)*np.inner(b_1,b_4)
    ret -= np.inner(b_2,x)*np.inner(b_4,x)*np.inner(b_1,b_3)
    ret -= np.inner(b_3,x)*np.inner(b_4,x)*np.inner(b_1,b_2)
    ret += np.inner(b_1,b_2)*np.inner(b_3,b_4)
    ret += np.inner(b_1,b_3)*np.inner(b_2,b_4)
    ret += np.inner(b_1,b_4)*np.inner(b_2,b_3)
    return ret
    
x = np.random.normal(0.0,1.0,size=(N,d))
y = x[:,0]
b_1 = np.array([1,0])
b_2 = np.array([1,0])
b_3 = np.array([1,0])
b_4 = np.array([1,0])
S_4 = S_4_estimator(x,y)
print(S_4)
print(S_4_value(S_4, b_1, b_2, b_3, b_4))