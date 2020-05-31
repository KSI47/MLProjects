import pandas as pd 
import numpy as np
import operator
import matplotlib.pyplot as plt
import math 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data_train=pd.read_csv('./train.csv') #load train data
X=data_train.drop(['Phase'], axis=1) #select features
Y=data_train['Phase'] # select labels
C_array=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] # create a set of hyperparameters test values
dic_acc_cross={} #dictionnary for cross validation accuracies, to calculate the mean
dic_acc={} #dictionnary for mean accuracies for each values of c
for c in C_array: # for each value of c
	for i in range(5): # for each value of c, we split the data 5 different times and calculate the accuracy in each and then take the mean to compare
		X_train, X_valid, Y_train, Y_valid= train_test_split(X,Y, test_size=0.3, random_state=20) #split data to train set and cross validation set 
		svm_model=SVC(C=c,gamma='auto') #gamma_auto=1/N with N= number of features
		svm_model.fit(X_train, Y_train) #fitting the model 
		Y_predict_valid= svm_model.predict(X_valid) #Predict the labels of X_valid using the fitted model
		acc_cross=accuracy_score(Y_valid, Y_predict_valid) # SCORE= Right_predictions / Total size of data validation data
		dic_acc_cross[i]=acc_cross*100 # store the accuracy in pourcentage
	acc_moy=0
	for k in dic_acc_cross:
		print( dic_acc_cross[k]) # print the 5 accuracies
	for k in dic_acc_cross:
		acc_moy+=dic_acc_cross[k] 
	acc_moy=acc_moy/5 # calculate the mean
	dic_acc[c]=acc_moy # store the mean
	print("mean accuracy score for C=" + format(c) + " is " + format(acc_moy) + " \n")
c_array_log=list(dic_acc.keys())
for i in range(len(c_array_log)):
	c_array_log[i] = math.log(c_array_log[i],10) # calculate the logarithm value of each c value in order to plot 
plt.plot(c_array_log,list(dic_acc.values()),'ro') # plot all the accuracies according to each value of c 
plt.ylabel("Accuracy")
plt.xlabel("C values in log")
plt.show()
c_opt=max(dic_acc.items(), key=operator.itemgetter(1))[0] # choose the C value according to the maximum accuracy mean
X_train, X_valid, Y_train, Y_valid= train_test_split(X,Y, test_size=0.3, random_state=20)
svm_opt=SVC(C=c_opt,gamma='auto') # recreate the model using that optimal C value
svm_opt.fit(X_train,Y_train) # fit the optimal model
data_test=pd.read_csv('./test.csv')# load test data
X_test=data_test.drop(['Phase'], axis=1) # select features 
Y_test=data_test['Phase'] # select labels
Y_predict_test= svm_opt.predict(X_test) # predict the X_test labels
confusion= np.array(confusion_matrix(Y_test, Y_predict_test, labels=["'D'", "'H'"])) # calculate the confusion matrix
plot_cm = pd.DataFrame(confusion, index=['repos', 'attente'], columns=['predicted_repos', 'predicted_attente']) # create a DF of confusion matrix with labeled axis
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # plot confusion matrix for data test
                print(plot_cm)
print("accuracy score on Test_data is for C=" + format(c) + " is " + format(accuracy_score(Y_test, Y_predict_test)*100) + " \n") # calculate and print the test score 




