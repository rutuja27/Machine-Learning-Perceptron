import numpy as np
from matplotlib import pyplot as plt
 
class Perceptron :
 
    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""
 
    def __init__(self, max_iterations=1000, learning_rate=0.2) :
 
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
 
    def fit(self, X, y) :
        self.w = np.zeros(len(X[0]))
        self.wpocket = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        a = [0.0]*1000
        N = 170
        
        while (not converged and iterations < self.max_iterations) :
            converged = True
            misclassifications = 0.0
            for i in range(len(X)) :
                if y[i] * (self.discriminant(X[i])) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
                    misclassifications += 1
            a[iterations]= misclassifications
            if iterations == 0:
                self.wpocket = self.w
                k = a[iterations]
            else:
                if a[iterations]< k:
                    self.wpocket = self.w
                    k = a[iterations]
            #plot_data(X, y, self.w)
            iterations += 1
"""Calculation of Ein"""            
        Ein = k/N
        print Ein
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        return (self.w,b)
    
    
    def discriminant(self, x) :
        return np.dot(self.w, x)
 
    def predict(self, X) :
 
        scores = np.dot(self.w, X)
        return np.sign(scores)
 
 
def generate_separable_data(N) :
    #xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    #print w,w.shape
    X = np.random.uniform(-1, 1, [N, 2])
    #print X,X.shape
    y = np.sign(np.dot(X, w))
    return X,y,w
 
def plot_data(X, y, w):
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    a = -w[0]/w[1]
    pts = np.linspace(-1,1)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0],X[i][1],cols[y[i]]+'o')
    plt.show()
 
if __name__=='__main__' :
    X,y,w = generate_separable_data(270)
    p = Perceptron()
    p.fit(X,y)
