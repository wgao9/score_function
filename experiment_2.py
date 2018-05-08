import numpy as np
import scipy as sp
import tensorflow as tf
import estimator
import time

d = 2
n = 2**13
lr = 1e-3
steps = 10000
mu = 5.0
lamb = 0.0

def generating_data(n, d, mixed=False):
	if mixed == True:
		x_left = np.random.multivariate_normal(np.ones(d),np.identity(d),n/2)
		x_right = np.random.multivariate_normal(-np.ones(d),np.identity(d),n/2)
		return np.concatenate((x_left,x_right),axis=0)
	else:
		return np.random.multivariate_normal(np.zeros(d),np.identity(d),n)
        
def generating_weights(d):
	return np.identity(d)

def weight_loss(A_1, A_2):
	Q = np.abs(np.matmul(np.linalg.inv(A_2), A_1))
	return min(1 - Q.max(0).min(), 1-Q.max(1).min())

#Model input x and label y
x = tf.placeholder(tf.float32,[None,d])
y = tf.placeholder(tf.float32,[None])
S_4 = tf.placeholder(tf.float32,[d,d,d,d])

#Model y=\sum_i ReLU(a_i^T x)
hat_A = tf.Variable(tf.random_normal([d,d], stddev=tf.sqrt(1.0/d)))
hat_y = tf.reduce_sum(tf.nn.relu(tf.matmul(x, hat_A)), axis=1)

#Loss Function
l_2_loss = tf.reduce_mean(tf.square(hat_y - y))
optimizer = tf.train.AdamOptimizer(lr)
l_2_train = optimizer.minimize(l_2_loss)

#Tensor Loss Function
A_jk = tf.einsum('im,jm,kn,ln->ijkl', hat_A, hat_A, hat_A, hat_A)
A_jj = tf.einsum('im,jm,km,lm->ijkl', hat_A, hat_A, hat_A, hat_A)
reg = tf.reduce_sum(tf.square(tf.diag_part(tf.matmul(tf.transpose(hat_A), hat_A) - tf.eye(d))))
tensor_loss = tf.reduce_sum(tf.multiply(S_4, -A_jk+(mu+1)*A_jj))+lamb*reg
optimizer = tf.train.GradientDescentOptimizer(lr)
tensor_train = optimizer.minimize(tensor_loss)

#Generating Data
train_x = generating_data(n, d, mixed=True)
A = generating_weights(d)
train_z = np.matmul(train_x,A)
train_y = np.sum(train_z*(train_z>0.0), axis=1)
train_S_4 = estimator.S_4_estimator(train_x, train_y, bw_factor=1.5)
#train_S_4 = estimator.S_4_gaussian(train_x, train_y)
print(train_S_4)

#Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(tf.assign(hat_A, A))
loss_tensor_0 = sess.run(tensor_loss, {S_4:train_S_4})
sess.run(init)

losses_A = []
losses_tensor = []
start_time = time.time()

f = open("mixed_LLSFE.dat", 'w')
f.write("#Mixed Input, LLSF Estimator; n=1024, d=2, lr=0.001, mu=5\n")
f.write("#Steps		Objective	weight_loss")
#Training
for i in range(steps):
	sess.run(tensor_train, {S_4: train_S_4})
	sess.run(tf.assign(hat_A, tf.nn.l2_normalize(hat_A,[0])))
	if i%10 == 0:
		loss_tensor = sess.run(tensor_loss, {S_4: train_S_4})
		loss_A = weight_loss(sess.run(hat_A), A)
		print ("Mixed: Step %d, loss_tensor = %.6f (goal: %.6f), loss_A = %.6f, time = %.6f"%(i, loss_tensor, loss_tensor_0, loss_A, time.time()-start_time))
		f.write("%d	%.6f	%.6f \n"%(i, loss_tensor, loss_A))
		if loss_tensor < loss_tensor_0-0.1 and loss_A > 0.1:
			print("Warning!")
			#print(train_S_4)
			
			print(sess.run(hat_A, {S_4:train_S_4}))
			#print(sess.run(-A_jk+(mu+1)*A_jj, {S_4:train_S_4}))
		#print(sess.run(A_jj, {S_4:train_S_4}))
		
		losses_tensor.append(loss_tensor)
		losses_A.append(loss_A)
print losses_A
