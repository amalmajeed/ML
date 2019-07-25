#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt


GLOBAL_MIN = np.inf
THETA0 = 0
THETA1 = 0

def hypothesis(th0,th1,x):
	return(th0+th1*x)

def cost_function(th0,th1,ip,op):
	global GLOBAL_MIN,THETA1,THETA0
	pred_err = []
	for i,val in enumerate(ip):
		pred_err.append(op[i]-hypothesis(th0,th1,val))
	MSE = (0.5/len(pred_err))*sum([i**2 for i in pred_err])
	if(MSE < GLOBAL_MIN):
		GLOBAL_MIN = MSE
		THETA0 = th0
		THETA1 = th1
	return MSE


input_train = np.array(range(1,50))
label_train = np.array(range(1,50))
#predict_train = np.array([])
MSE_train = []
parameter_train = []
#theta0_error_train = []
#theta1_error_train = []

theta0 , theta1 = 1 , 0.5

learning_rate = 0.004

# Hypothesis

# predict_train = theta0 + theta1*input_train
# error_train = label_train - predict_train
# jtheta = sum([i**2 for i in error_train])

for i in range(0,len(input_train)):
	htheta = hypothesis(theta0,theta1,input_train[i])
	y = label_train[i]
	e = y - htheta
	theta0 = theta0 + learning_rate*e
	theta1 = theta1 + learning_rate*e*input_train[i]
	parameter_train.append((theta1,theta0))
	MSE_train.append(cost_function(theta0,theta1,input_train,label_train))

print(input_train)
#print(parameter_train)
theta1_train = [i for i,j in parameter_train]
#print(MSE_train)
print(f"GLOBAL_MIN : {GLOBAL_MIN}\n THETA0 : {THETA0}\n THETA1 : {THETA1}\n")

#Error = sum( [ (i-j)**2 for i,j in zip(predict_train, label_train)]) # ERROR = Sum of Squares of 

#Cost = 0.5 * Error # COST FUNCTION CALCULATION








# boundary = [(i,j) for i,j in zip(range(-50,51),[m**2 for m in range(-50,51)])]
# x = [i for i,j in boundary]
# y = [j for i,j in boundary]












plt.xlim(0, 60)
plt.ylim(0, 100)
plt.gca().set_aspect('equal', adjustable='box')

plt.plot(range(1,50),MSE_train)
plt.show()


