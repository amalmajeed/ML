#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt


GLOBAL_MIN = np.inf
THETA0 = 0
THETA1 = 0

def op_map(optrain):
	op = [i*3 for i in optrain]
	return op

def hypothesis(th0,th1,x):
	return(th0+th1*x)

def cost_function(th0,th1,ip,op):
	"""
			AVERAGE MEAN SQUARED ERROR COST FUNCTION
	"""
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

def GradientDescent(theta0,theta1,learning_rate,input_train,label_train):
	MSE_train = []
	parameter_train = []
	for i in range(0,len(input_train)):
		htheta = hypothesis(theta0,theta1,input_train[i])
		y = label_train[i]
		e = y - htheta
		theta0 = theta0 + learning_rate*e # Bias Update
		theta1 = theta1 + learning_rate*e*input_train[i] # Weight Update
		parameter_train.append((theta1,theta0))
		MSE_train.append(cost_function(theta0,theta1,input_train,label_train))
	return MSE_train,parameter_train


def PlotCostFunction(x,y,xlim1,xlim2,ylim1,ylim2):
	plt.xlim(xlim1,xlim2)
	plt.ylim(ylim1,ylim2)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.plot(x,y,'ro')
	plt.show()


def __main__():
	input_train = np.array(range(1,100))
	label_train = np.array(op_map(range(1,100)))
	theta0 , theta1 = 1 , 0.5
	learning_rate = 0.000025
	print(f"Input Dataset : {input_train}")
	MSE , PARAM = GradientDescent(theta0,theta1,learning_rate,input_train,label_train)
	theta1_train = [i for i,j in PARAM]
	print(f"GLOBAL_MIN : {GLOBAL_MIN}\n THETA0 : {THETA0}\n THETA1 : {THETA1}\n")
	print(f"MEAN SQUARED ERRORS : {MSE}\n THETA1 : {theta1_train}")
	MSE_SC = [i/2 for i in MSE]
	PlotCostFunction(theta1_train,MSE_SC,xlim1=-5,xlim2=5,ylim1=0,ylim2=10)



if __name__ =="__main__":
	__main__()
