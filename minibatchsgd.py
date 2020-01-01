import numpy as np
import random

'''
order of stuff:
1. define __init__ , set the number of layers, size array, bias matrix and weight matrices
2. feedforward function to calc the result for a given input
3. the sgd function to train the net using sgd.It takes in the training data, shuffles it, splits it into mini batches and send the shuffle and send it to the update function
4. the update fn to compute and update weight and bias mats by starting to send a single minibatch one at a time to the backpropagation function
5. backprop func to compute the backprop, it will first feedforward, get to the result, store each step in an array, and then come back layer by layer to the orignal values.
6. evaluate to calc the sample test cases in each epoch
7,8. sigmoid and sigmoid's derivative
'''

class Network:
    
    def __init__ (self,sizes):
        self.layersCount = len(sizes)
        self.sizes = sizes
        #init biases, each bias should have elements same as number
        #of neurons in the current layer [one bias for each elem]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        #init weights with dimension no of elems in curr layer * 
        # no of elems in previous layer (this has to be multiplied 
        # with output from previous layer)
        #so we'll have to combine previous and next layer neurons
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def sgd(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        testData = list(testData)
        trainingData = list(trainingData)
        if testData : nTest = len(testData)
        n = len(trainingData)
        
        #split the data sets by shuffling, a new shuffle in each epoch
        for j in range(epochs):
            #shuffle now
            random.shuffle(trainingData)
            miniBatches = [trainingData[k : k + miniBatchSize] for k in range(0,n,miniBatchSize)]
            #now send the mini batches for updation
            for miniBatch in miniBatches:
                self.update(miniBatch, eta)
                
            if testData:
                print(f"Epoch {j} : {self.evaluate(testData)} / {nTest} correct results")
            else:
                print(f"Epoch {j} complete")
    
    def update(self, miniBatch, eta):
            #make 2 new arrays init with zeros, which would store the updated weight and biases matrices returned by the backprop function
        updWeight = [np.zeros(w.shape) for w in self.weights]
        updBias = [np.zeros(b.shape) for b in self.biases]
            
            #now send the dataset one by one into the backprop function
        for x,y in miniBatch :
            changeWeight, changeBias = self.backprop(x,y)
            #update the upd mats with the returned values
            updWeight = [ow + delw for ow,delw in zip(updWeight, changeWeight)]
            updBias = [ob + delb for ob, delb in zip(updBias, changeBias)]
        # now after every miniBatch is over, update the original weights and biases
        # we know that the changes will be original - eta/n * change in cost function (which is being returned into the above matrices), we have made our functions in such a way1
        self.weights = [w - (eta/len(miniBatch))*updW for w, updW in zip(self.weights, updWeight)]
        self.biases = [b - (eta/len(miniBatch))*updB for b, updB in zip(self.biases, updBias)]
        
    def backprop(self, x, y):
        # first again init a mat of zeros to store the new values (the values to be returned)
        delWeights = [np.zeros(w.shape) for w in self.weights]
        delBiases = [np.zeros(b.shape) for b in self.biases]
        
        #first feed forward and find the output, storing the z each time as it has to be used again
        activation = x
        activations = [x] # to store activations of all layer
        zs = [] # to store z mats
        
        # iterate throgh the values, feedforward
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # now we are ready wuth our activations and z matrices
        # now, start backpropagation
        
        #backprop
        # 1. for the last layer, backprop = derivative of C . sigmoid dash of the last z matrix, we have cosidered the learning rate and length in the update function
        delta = (activations[-1] - y)*sigmoidDash(zs[-1])
        delWeights[-1] = np.dot(delta, activations[-2].transpose())
        delBiases[-1] = delta
        
        #2. for the remaining layers, go back deeper and use the del l = w l+1 T . del l+1 * sigmoid dash zl formula
        for l in range(2, self.layersCount):
            z = zs[-l]
            sd = sigmoidDash(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta ) * sd
            delWeights[-l] = np.dot(delta, activations[-l-1].transpose())
            delBiases[-l] = delta
        return (delWeights, delBiases)

    def evaluate(self, test_data) :
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
def sigmoidDash(z):
        val = sigmoid(z)
        return val*(1-val)
    
    ############# copy pasted function for testing

      
        