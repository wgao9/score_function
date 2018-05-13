import numpy as np
import scipy.spatial as ss
from math import exp
from scipy.stats import multivariate_normal
import time
d = 2

def S_2_estimator(x, x_0, bw):
    N = len(x)
    d = len(x[0])

    S_2 = np.zeros((d,d))
    M_0 = 0
    M_1 = np.matrix(np.zeros((d,1)))
    M_2 = np.matrix(np.zeros((d,d)))

    for i in range(N):
        dis = np.matrix(x[i] - x_0).reshape(d,1)
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
		S_2[j_1,j_2] += a_1[j_1]*a_1[j_2]
		S_2[j_1,j_2] -= A_2[j_1,j_2]
    return S_2


def S_2_gaussian(x):
    d = len(x)
    S_2 = np.zeros((d,d))
    eye = np.identity(d)

    for j_1 in range(d):
	for j_2 in range(d):
	    S_2[j_1,j_2] += x[j_1]*x[j_2]
	    S_2[j_1,j_2] -= eye[j_1,j_2]
    return S_2

def S_2_mixed(x, mu_1, mu_2):
    S_2_1 = S_2_gaussian(x-mu_1)
    S_2_2 = S_2_gaussian(x-mu_2)
    density_1 = multivariate_normal.pdf(x, mean=mu_1, cov=np.identity(d))
    density_2 = multivariate_normal.pdf(x, mean=mu_2, cov=np.identity(d))
    return (density_1*S_2_1+density_2*S_2_2)/(density_1+density_2)

def S_4_estimator(x, x_0, bw):
    N = len(x)
    d = len(x[0])

    S_4 = np.zeros((d,d,d,d))
    M_0 = 0
    M_1 = np.matrix(np.zeros((d,1)))
    M_2 = np.matrix(np.zeros((d,d)))

    for i in range(N):
        dis = np.matrix(x[i] - x_0).reshape(d,1)
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
                  S_4[j_1,j_2,j_3,j_4] += a_1[j_1]*a_1[j_2]*a_1[j_3]*a_1[j_4]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_1,j_2]*a_1[j_3]*a_1[j_4]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_1,j_3]*a_1[j_2]*a_1[j_4]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_1,j_4]*a_1[j_2]*a_1[j_3]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_2,j_3]*a_1[j_1]*a_1[j_4]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_2,j_4]*a_1[j_1]*a_1[j_3]
                  S_4[j_1,j_2,j_3,j_4] -= A_2[j_3,j_4]*a_1[j_1]*a_1[j_2]
                  S_4[j_1,j_2,j_3,j_4] += A_2[j_1,j_2]*A_2[j_3,j_4]
                  S_4[j_1,j_2,j_3,j_4] += A_2[j_1,j_3]*A_2[j_2,j_4]
                  S_4[j_1,j_2,j_3,j_4] += A_2[j_1,j_4]*A_2[j_2,j_3]
    return S_4

          
def S_4_gaussian(x):
    d = len(x)
    S_4 = np.zeros((d,d,d,d))
    eye = np.identity(d)

    for j_1 in range(d):
     	for j_2 in range(d):
	    for j_3 in range(d):
		for j_4 in range(d):
 		    S_4[j_1,j_2,j_3,j_4] += x[j_1]*x[j_2]*x[j_3]*x[j_4]
	            S_4[j_1,j_2,j_3,j_4] -= x[j_1]*x[j_2]*eye[j_3,j_4]
		    S_4[j_1,j_2,j_3,j_4] -= x[j_1]*x[j_3]*eye[j_2,j_4]
		    S_4[j_1,j_2,j_3,j_4] -= x[j_1]*x[j_4]*eye[j_2,j_3]
		    S_4[j_1,j_2,j_3,j_4] -= x[j_2]*x[j_3]*eye[j_1,j_4]
		    S_4[j_1,j_2,j_3,j_4] -= x[j_2]*x[j_4]*eye[j_1,j_3]
		    S_4[j_1,j_2,j_3,j_4] -= x[j_3]*x[j_4]*eye[j_1,j_2]
		    S_4[j_1,j_2,j_3,j_4] += eye[j_1,j_2]*eye[j_3,j_4]
		    S_4[j_1,j_2,j_3,j_4] += eye[j_1,j_3]*eye[j_2,j_4]
		    S_4[j_1,j_2,j_3,j_4] += eye[j_1,j_4]*eye[j_2,j_3]

    return S_4

def S_4_mixed(x, mu_1, mu_2):
    S_4_1 = S_4_gaussian(x-mu_1)
    S_4_2 = S_4_gaussian(x-mu_2)
    density_1 = multivariate_normal.pdf(x, mean=mu_1, cov=np.identity(d))
    density_2 = multivariate_normal.pdf(x, mean=mu_2, cov=np.identity(d))
    return (density_1*S_4_1+density_2*S_4_2)/(density_1+density_2)

def SP_2(S_2, T_2):
    return np.linalg.norm(S_2-T_2,2)

def Fro_4(S_4,T_4):
    S_4_v = S_4.flatten()
    T_4_v = T_4.flatten()
    return np.linalg.norm(S_4_v-T_4_v,2)


def experiment(bw, N_list, T, order=2, mixed=True):
	#Experiment; order={2,4}, distribution={Gaussian, mixed Gaussian}
	#initialization
	mean_err, std_err = [], []
	per_95_err, per_75_err, median_err, per_25_err, per_5_err = [], [], [], [], []
	start_time = time.time()
	mu_1, mu_2 = [1,1], [-1,-1]

	#Generate Testing Points
        if mixed == True:
		x_0_left = np.random.multivariate_normal(mu_2,[[1,0],[0,1]],T/2)
		x_0_right = np.random.multivariate_normal(mu_2,[[1,0],[0,1]],T/2)
		x_0 = np.concatenate((x_0_left, x_0_right), axis=0)
	else:
		x_0 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],T)

	#For each number of samples
	for n in N_list:
		errors = []
	
		#For each test point
		for t in range(T):
			#Generate samples from distribution
        		if mixed == True:
				if np.linalg.norm(x_0[t]-mu_1,2) > 2.0 and np.linalg.norm(x_0[t]-mu_2,2) > 2.0:
					continue
				x_left = np.random.multivariate_normal(mu_1,[[1,0],[0,1]], n/2)
				x_right = np.random.multivariate_normal(mu_2,[[1,0],[0,1]], n/2)
				x = np.concatenate((x_left,x_right), axis=0)
			else:
				if np.linalg.norm(x_0[t],2) > 2.0:
					continue
				x = np.random.multivariate_normal([0,0],[[1,0],[0,1]], n)
			
			#Estimate
			if mixed == True:
				if order == 2:
					S_hat = S_2_estimator(x, x_0[t], bw*n**(-1.0/(d+6)))
					S_true = S_2_mixed(x_0[t], mu_1, mu_2)
				elif order == 4:
					S_hat = S_4_estimator(x, x_0[t], bw*n**(-1.0/(d+6)))
					S_true = S_4_mixed(x_0[t], mu_1, mu_2)
			else:
				if order == 2:
					S_hat = S_2_estimator(x, x_0[t], bw*n**(-1.0/(d+6)))
					S_true = S_2_gaussian(x_0[t])
				elif order == 4:
					S_hat = S_4_estimator(x, x_0[t], bw*n**(-1.0/(d+6)))
					S_true = S_4_gaussian(x_0[t])

			#Computing Error
			if order == 2:
				errors.append(SP_2(S_hat, S_true))
			elif order == 4:
				errors.append(Fro_4(S_hat, S_true))
			
			if t%100 == 99:
				print("Experiment I: Num of sample: %d, Num of try: %d, time: %.4f"%(n,t,time.time()-start_time))

		print(np.mean(errors))
		print(np.std(errors))
                print(np.percentile(errors,95))
                print(np.percentile(errors,75))
		print(np.median(errors))
                print(np.percentile(errors,25))
		print(np.percentile(errors,5))

		mean_err.append(np.mean(errors))
		std_err.append(np.std(errors))
		per_95_err.append(np.percentile(errors,95))
		per_75_err.append(np.percentile(errors,75))
		median_err.append(np.median(errors))
		per_25_err.append(np.percentile(errors,25))
		per_5_err.append(np.percentile(errors,5))

	return mean_err, std_err, per_95_err, per_75_err, median_err, per_25_err, per_5_err 

def print_list(name, l):
	name += "["
	for i in l:
		name += "%.6f  "%i
	name += "]"
	print(name)

def main():
	#N_list = [64,128,256,512,1024]
	N_list = [1024]

        #mean_S_2, std_S_2, per_95_S_2, per_75_S_2, median_S_2, per_25_S_2, per_5_S_2 = experiment(6.0, N_list, T=10000, order=2, mixed=False)
        mean_S_4, std_S_4 , per_95_S_4, per_75_S_4, median_S_4, per_25_S_4, per_5_S_4 = experiment(10.0, N_list, T=50000, order=4, mixed=False)
        #mean_S_2_mixed, std_S_2_mixed, per_95_S_2_mixed, per_75_S_2_mixed, median_S_2_mixed, per_25_S_2_mixed, per_5_S_2_mixed = experiment(1.5, N_list, T=10000, order=2, mixed=True)
       # mean_S_4_mixed, std_S_4_mixed, per_95_S_4_mixed, per_75_S_4_mixed, median_S_4_mixed, per_25_S_4_mixed, per_5_S_4_mixed = experiment(2.5, N_list, T=50000, order=4, mixed=True)
	'''
        print_list("Gaussian S_2: Mean = ",mean_S_2)
        print_list("Gaussian S_2: Std = ",std_S_2)
	print_list("Gaussian S_2: 95% = ",per_95_S_2)
	print_list("Gaussian S_2: 75% = ",per_75_S_2)
	print_list("Gaussian S_2: 50% = ",median_S_2)
	print_list("Gaussian S_2: 25% = ",per_25_S_2)
	print_list("Gaussian S_2: 5% = ",per_5_S_2)
       
	
        print_list("Gaussian S_4: Mean = ",mean_S_4)
        print_list("Gaussian S_4: Std = ",std_S_4)
	print_list("Gaussian S_4: 95% = ",per_95_S_4)
	print_list("Gaussian S_4: 75% = ",per_75_S_4)
	print_list("Gaussian S_4: 50% = ",median_S_4)
	print_list("Gaussian S_4: 25% = ",per_25_S_4)
	print_list("Gaussian S_4: 5% = ",per_5_S_4)

	'''
        print_list("Mixed Gaussian S_2: Mean = ",mean_S_2_mixed)
        print_list("Mixed Gaussian S_2: Std = ",std_S_2_mixed)
	print_list("Mixed Gaussian S_2: 95% = ",per_95_S_2_mixed)
	print_list("Mixed Gaussian S_2: 75% = ",per_75_S_2_mixed)
	print_list("Mixed Gaussian S_2: 50% = ",median_S_2_mixed)
	print_list("Mixed Gaussian S_2: 25% = ",per_25_S_2_mixed)
	print_list("Mixed Gaussian S_2: 5% = ",per_5_S_2_mixed)
       
	'''
        print_list("Mixed Gaussian S_4: Mean = ",mean_S_4_mixed)
        print_list("Mixed Gaussian S_4: Std = ",std_S_4_mixed)
	print_list("Mixed Gaussian S_4: 95% = ",per_95_S_4_mixed)
	print_list("Mixed Gaussian S_4: 75% = ",per_75_S_4_mixed)
	print_list("Mixed Gaussian S_4: 50% = ",median_S_4_mixed)
	print_list("Mixed Gaussian S_4: 25% = ",per_25_S_4_mixed)
	print_list("Mixed Gaussian S_4: 5% = ",per_5_S_4_mixed)
	'''

if __name__ == "__main__":
    main()
		
