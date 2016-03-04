import numpy as np
from matplotlib import pyplot as plt

class Perceptron :
 
    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""
 
    def __init__(self, max_iterations=1000, learning_rate=0.2) :
 
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
 
    def fit(self, X, y) :
        self.w = np.random.uniform(-1,1,len(X[0]))
        converged = False
        c = 0.0001
        M = 3500
        iterations =0
        a = [None] * 10000
        while (not converged and iterations < self.max_iterations) :
            converged = True
            j=0
            lambda_i = np.zeros((10000,2))
            misclassifications =0.0
            for i in range(len(X)) :
                if y[i] * (self.discriminant(X[i])) < c*(np.linalg.norm(self.w)) :
                    lambda_i[j][0] = y[i] * (self.discriminant(X[i]))
                    lambda_i[j][1] = i
                    converged = False
                    j = j + 1
                    misclassifications += 1
            a[iterations]= misclassifications
            if not converged:
                ind = np.argmax(lambda_i,0)
                index = lambda_i[ind[0]][1]
                self.w = self.w + self.learning_rate*y[index]*X[index]
            iterations += 1
        k = np.amin(a)
        Ein = k/M
        print Ein
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        return self.w
        
    def discriminant(self, x) :
        return np.dot(self.w, x)
 
    def predict(self, X) :
 
        scores = np.dot(self.w, X)
        return np.sign(scores)   
                
        
                            
                
                    

        
