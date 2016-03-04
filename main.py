import numpy as np
import  random
import perceptron
import pocket
import perceptron_bias
import modified_perceptron as mp

#Trainig data and Testing data
data =np.genfromtxt("/home/rutuja/Downloads/heart.data", delimiter=",",comments="#")
random.shuffle(data)
y = np.zeros(len(data))
X = np.zeros((270,13))
for i in range(len(data)):
    y[i]= data[i,1]
for j in range(len(data)):
    X[j,:] = data[j,2:15]
a = np.amax(X)
b = np.amin(X)

#Data Standardization
X_mean = np.mean(X,0)
X_std = np.std(X,0)

X_data = (X - X_mean)/X_std
print np.amax(X_data)
print np.amin(X_data)
X_data_train = X_data[0:170,:]
X_data_test = X_data[170:270,:]

#Data normalization
min_X = np.amin(X)
max_X = np.amax(X)
N = max_X - min_X
X_scale = (b-a)*((X - min_X)/N) - 1

X_scale_train = X_scale[0:170,:]
X_scale_test=X_scale[170:270,:]

#Data witout normalization or standardization
X_train = X[0:170,:]
X_test = X[170:270,:]
y_train = y[0:170]
y_test = y[170:270]

#Training
p = perceptron.Perceptron()
#p = perceptron_bias.Perceptron()
#p = pocket.Perceptron()
#p = mp.Perceptron()
w = p.fit(X_data_train,y_train)
c = 0.0001
M = 100
#Testing
false_classification = 0.0
for j in range(len(X_data_test)):
    if y_test[j] * (np.dot(w,X_data_test[j])+b1) <= 0:
       false_classification += 1
print false_classification
Eout = false_classification/M
print Eout
