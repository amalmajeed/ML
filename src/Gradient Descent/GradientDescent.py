#!/usr/local/bin/python3



import numpy as np
import matplotlib.pyplot as plt
import random


GLOBAL_MIN = np.inf
THETA0 = 0
THETA1 = 0
EPOCH = 200
LEARNING_RATE = 0.000129

def op_map(optrain):
	op = [(i*3+random.uniform(-10,10)) for i in optrain]
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
	global EPOCH
	MSE_train = []
	th0 = theta0
	th1 = theta1
	mse = 0
	parameter_train = []
	for epoch in range(EPOCH):
		for i in range(0,len(input_train)):
			htheta = hypothesis(th0,th1,input_train[i])
			y = label_train[i]
			e = y - htheta
			#print(f"y : {y:.4f} htheta : {htheta:.4f} e : {e:.4f}")
			th0 = th0 + learning_rate*e # Bias Update
			th1 = th1 + learning_rate*e*input_train[i] # Weight Update
			mse = cost_function(th0,th1,input_train,label_train)
			parameter_train.append((th1,th0))
			MSE_train.append(mse)
		print(f"epoch : {epoch:04d} theta1/weight : {th1:.4f} theta0/bias : {th0:.4f} pred : {htheta:.4f} error : {e:.4f}")
		#parameter_train.append((th1,th0))
		#MSE_train.append(cost_function(th0,th1,input_train,label_train))
	return MSE_train,parameter_train


def PlotCostFunction(x,y,xlim1,xlim2,ylim1,ylim2):
	plt.xlim(xlim1,xlim2)
	plt.ylim(ylim1,ylim2)
	plt.gca().set_aspect('auto', adjustable='box')
	plt.plot(x,y,'ro')
	plt.show()

def PlotRegressionData(ip,label,xlim1,xlim2,ylim1,ylim2):
	global THETA1,THETA0
	final_op_train = []
	for i in ip:
		final_op_train.append(hypothesis(THETA0,THETA1,i)) 
	plt.plot(ip,final_op_train,'b--')
	plt.plot(ip,label,'ro')
	plt.xlim(0,100)
	plt.ylim(0,500)
	plt.gca().set_aspect('auto', adjustable='box')
	plt.show()



def __main__():
	global LEARNING_RATE
	input_train = np.array(range(0,100))
	label_train = np.array(op_map(range(0,100)))
	theta0 , theta1 = 0.5 , 3
	
	print(f"Input Dataset : {input_train}")
	MSE , PARAM = GradientDescent(theta0,theta1,LEARNING_RATE,input_train,label_train)
	theta1_train = [i for i,j in PARAM]
	#print(f"GLOBAL_MIN : {GLOBAL_MIN}\n THETA0 : {THETA0}\n THETA1 : {THETA1}\n")
	#print(f"MEAN SQUARED ERRORS : {MSE}\n THETA1 : {theta1_train}")
	print(f"MEAN SQUARED ERRORS : {MSE}")
	#Scaled down mean square errors
	MSE_SC = [i/2 for i in MSE]
	
	PlotCostFunction(theta1_train,MSE,xlim1=0,xlim2=50,ylim1=0,ylim2=200)
	#PlotRegressionData(input_train,label_train,xlim1=0,xlim2=100,ylim1=0,ylim2=500)



if __name__ =="__main__":
	__main__()






















# Author - Amal Majeed