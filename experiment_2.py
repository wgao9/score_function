import numpy as np
import scipy as sp
import tensorflow as tf
import estimator
import time

d = 50
n = 2**13
lr_l_2 = 5e-3
lr_tensor = 5e-5
steps = 10000
mu = 30.0
lamb = 0.0

def generating_data(n, d, mixed=False, exponential=False):
	if exponential == True:
		x = np.random.exponential(1.0,size=(n,d))
		sgn = 2*np.random.randint(2,size=(n,d))-1
		return np.multiply(sgn,x)
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
optimizer_l_2 = tf.train.AdamOptimizer(lr_l_2)
l_2_train = optimizer_l_2.minimize(l_2_loss)

#Tensor Loss Function
A_jk = tf.einsum('im,jm,kn,ln->ijkl', hat_A, hat_A, hat_A, hat_A)
A_jj = tf.einsum('im,jm,km,lm->ijkl', hat_A, hat_A, hat_A, hat_A)
reg = tf.reduce_sum(tf.square(tf.diag_part(tf.matmul(tf.transpose(hat_A), hat_A) - tf.eye(d))))
tensor_loss = tf.reduce_sum(tf.multiply(S_4, -A_jk+(mu+1)*A_jj))+lamb*reg
optimizer_tensor = tf.train.GradientDescentOptimizer(lr_tensor)
tensor_train = optimizer_tensor.minimize(tensor_loss)

#Generating Data
train_x = generating_data(n, d, mixed=False, exponential = True)
A = generating_weights(d)
train_z = np.matmul(train_x,A)
train_y = np.sum(train_z*(train_z>0.0), axis=1)
#train_S_4 = estimator.S_4_estimator(train_x, train_y, bw_factor=1.5)
#train_S_4 = estimator.S_4_gaussian(train_x, train_y)
#train_S_4 = estimator.S_4_exponential(train_x,train_y)
#print(train_S_4)

#Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(tf.assign(hat_A, A))
loss_l2_0 = sess.run(l_2_loss, {x: train_x, y: train_y})
#loss_tensor_0 = sess.run(tensor_loss, {S_4:train_S_4})
sess.run(init)

losses_A = []
losses = []
start_time = time.time()

f = open("exponential_l2.dat", 'w')
f.write("#Exponential symmetric Input (p(x)=e^{-|x|}/2), L-2 loss; n=8192, d=50, lr=0.005\n")
f.write("#Steps		Objective	weight_loss")

#Training
for i in range(steps):
	sess.run(l_2_train, {x: train_x, y:train_y})
#	sess.run(tensor_train, {S_4: train_S_4})
	sess.run(tf.assign(hat_A, tf.nn.l2_normalize(hat_A,[0])))
	if i%10 == 0:
		loss_l_2 = sess.run(l_2_loss, {x: train_x, y: train_y})
		#loss_tensor = sess.run(tensor_loss, {S_4: train_S_4})
		loss_A = weight_loss(sess.run(hat_A), A)
		print ("Exponential(mu=%.1f): Step %d, loss_tensor = %.6f (goal: %.6f), loss_A = %.6f, time = %.6f"%(mu, i, loss_l_2, loss_l2_0, loss_A, time.time()-start_time))
		f.write("%d	%.6f	%.6f \n"%(i, loss_l_2, loss_A))
		if loss_l_2 < loss_l2_0-0.1 and loss_A > 0.1:
			print("Warning!")
			#print(train_S_4)
			
			#print(sess.run(hat_A, {S_4:train_S_4}))
			#print(sess.run(-A_jk+(mu+1)*A_jj, {S_4:train_S_4}))
		#print(sess.run(A_jj, {S_4:train_S_4}))
		
		losses.append(loss_l_2)
		losses_A.append(loss_A)
print losses_A
