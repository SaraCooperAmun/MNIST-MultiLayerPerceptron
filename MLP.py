import numpy as np
from numpy import *
import csv
import math
import matplotlib.pyplot as plt

trainX = []
trainY = []
testX = []
testY = []
with open('fashion-mnist_train.csv', 'rb') as infile:
       reader = csv.reader(infile)
       next(reader, None)  
       for row in reader:
           for i in range(len(row)):
               row[i] = int(row[i])
           trainX.append(row[1:])
           trainY.append(row[0])
with open('fashion-mnist_test.csv', 'rb') as infile:
       reader = csv.reader(infile)
       next(reader, None)  
       for row in reader:
           for i in range(len(row)):
               row[i] = int(row[i])
           testX.append(row[1:])
           testY.append(row[0])

trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
trainY = np.reshape(trainY, (-1,1)) 

    
trainXF = trainX/float(255)  

trainYV = vectorize_array(trainY)

testX = np.asarray(testX) 
testY = np.asarray(testY)
testY = np.reshape(testY, (-1,1)) 
testXF = testX/float(255)  

testYV = vectorize_array(testY) 




def vectorize_array(d): #for dataset with 10 clothes
    return np.array([vectorize(y) for y in d])


def vectorize(i):
    v = np.zeros(10)
    v[i] = 1.0
    return v

def vectorize_array2(d): #for dataset with 6 clothes
    return np.array([vectorize2(y) for y in d])


def vectorize2(i):
    v = np.zeros(6)
    v[i] = 1.0
    return v



def reducedataset(trainY, trainXF, testY, testXF):  #to create dataset2, with only 6 clothes
    index6 = np.where(trainY == 6)[0]

    index7 = np.where(trainY == 7)[0]

    index8 = np.where(trainY == 8)[0]

    index9 = np.where(trainY == 9)[0]



    array = np.concatenate((index6,index7,index8, index9), axis = None)
    Ydata = np.delete(trainY, array)
    Ydata = np.reshape(Ydata, (-1,1))
    trainYV2 = vectorize_array2(Ydata)
    trainXF2 =  np.delete(trainXF, array, axis = 0)




    indexTest6 = np.where(testY == 6)[0]

    indexTest7 = np.where(testY == 7)[0]

    indexTest8 = np.where(testY == 8)[0]

    indexTest9 = np.where(testY == 9)[0]



    arrayTest = np.concatenate((indexTest6,indexTest7,indexTest8, indexTest9), axis = None)
    Ytest = np.delete(testY, arrayTest)
    Ytest = np.reshape(Ytest, (-1,1))
    testYV2 = vectorize_array2(Ytest)
    testXF2= np.delete(testXF, arrayTest, axis = 0)

    return trainXF2, trainYV2, testXF2, testYV2

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)


def predict (x,y,W1,b1,W2,b2): #for predicting in the case of a 1 hidden layer neural network. Change activation function if necessary
    z1 = np.dot(x, W1) + b1  
    a1 = np.tanh(z1)
    #a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    exp_scores = np.exp(z2 - np.max(z2))
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    preds = np.argmax(out, axis=1)
    predsTest = np.asarray(preds)
    resultTest = y.argmax(axis=1)
    count = 0
    for i in range(len(predsTest)):  
                if predsTest[i] == resultTest[i]:
                    count+=1
    testAccuracy = count*100 / float( len(predsTest))
                    
    
    return testAccuracy

def predict2Layers (x,y,W1,b1,W2,b2, W3, b3): #for predicting in the case of a 2 hidden layer neural network. Change activation function if necessary
    z1 = np.dot(x, W1) + b1   
    #a1 = np.tanh(z1)
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    #a2 = np.tanh(z2)
    a2 = relu(z2)
    z3 =  np.dot(a2, W3) + b3
    exp_scores = np.exp(z3 - np.max(z3))
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    preds = np.argmax(out, axis=1)
    predsTest = np.asarray(preds)
    resultTest = y.argmax(axis=1)
    count = 0
    for i in range(len(predsTest)):  
                if predsTest[i] == resultTest[i]:
                    count+=1
    testAccuracy = count*100 / float( len(predsTest))
                    
    
    return testAccuracy



#1 hidden layer network with regularisation
def oneLayerRegularization(learning_rate, hidden_nodes, epochs, regul, trainXF, trainYV, testXF, testYV, testY, trainY):#for full and reduced dataset, with different activation functions, number of epochs, learning rates, and hidden neurons.

   # Xtrain= trainXF     #Select these lines if the full dataset (with 10 classes) needs to be tested
   # Ytrain = trainYV
   # Xtest = testXF
   # Ytest =testYV 
    #If only 6 clothes use these instead
    trainXF2, trainYV2, testXF2, testYV2 = reducedataset(trainY, trainXF, testY, testXF)
    Xtrain = trainXF2
    Ytrain = trainYV2
    Xtest = testXF2
    Ytest = testYV2

    input_dim = Xtrain.shape[1] 
    output_dim = Ytrain.shape[1]
    W1 = np.random.randn(input_dim, hidden_nodes) / np.sqrt(input_dim) 
    b1 = np.zeros((1, hidden_nodes)) 
    W2 = np.random.randn(hidden_nodes, output_dim) / np.sqrt(hidden_nodes) 
    b2 = np.zeros((1, output_dim)) 
    previous_cost = float('inf')
    count = 0
    costTraining = []
    costValidation = []
    finish = False
    epoch = 0
    index = []
    while (finish == False) and (epoch < epochs):
        
        samples = Xtrain.shape[0]
        split = int(2* samples/3)
        Xdata = Xtrain[:split]
        Ydata = Ytrain[:split]
        validationX = Xtrain[split:samples]
        validationY = Ytrain[split:samples]



        m = Xdata.shape[0] 
        z1 = Xdata.dot(W1) + b1
       # a1 = np.tanh(z1)
        a1 = relu(z1)

        z2 = np.dot(a1, W2) + b2  
        exp_scores = np.exp(z2 - np.max(z2)) 
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 

        dW2 = np.dot(a1.T, (output-Ydata)) / float(m)
        db2 = (np.sum(output-Ydata, axis=0, keepdims=True)) / float(m)
        #delta2 = np.dot((output-Ydata), W2.T) * (1 - np.power(z1, 2)) 
        delta2 = np.dot((output-Ydata), W2.T) * drelu(z1)
        dW1 = np.dot(Xdata.T, delta2)/ float(m)
        db1 = np.sum(delta2, axis=0) / float(m)
        dW2 += regul * W2
        dW1 += regul * W1
        W2 -= learning_rate* dW2
        W1 -= learning_rate * dW1
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1  
        if (epoch % 10 == 0 and epoch!=0):
            z1 = np.dot(Xdata, W1) + b1 
           # a1 = np.tanh(z1)
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            exp_scores = np.exp(z2 - np.max(z2))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsTraining = np.asarray(preds) 
            loss = - np.sum([y * np.log(x) for x, y in zip(out, Ydata)]) 
            loss += regul/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
            costTrain =  1/float(m) * loss
            costTraining.append(costTrain)

            z1 = np.dot(validationX, W1) + b1 
            #a1 = np.tanh(z1)
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            exp_scores = np.exp(z2 - np.max(z2))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsValidation = np.asarray(preds)  
            loss = - np.sum([y * np.log(x) for x, y in zip(out, validationY)]) 
            costVal =  1/float(m) * loss
            costValidation.append(costVal)
            index.append(epoch)
            if (previous_cost !=float('inf')):
                if (costVal > previous_cost): 
                    finish = True
            previous_cost = costVal

            print epoch, costVal
        epoch+=1
            
    result = Ydata.argmax(axis=1)
    count = 0
    for i in range(len(predsTraining)):  
                if predsTraining[i] == result[i]:
                    count+=1
    count2 = 0
    result2 = validationY.argmax(axis =1)
    for i in range(len(predsValidation)):  
                if predsValidation[i] == result2[i]:
                    count2+=1
    print "Training accuracy: ",  count*100 / float( len(predsTraining))
    print "Validation accuracy: ",  count2*100 / float( len(predsValidation))
    correct = predict(Xtest, Ytest, W1,b1,W2,b2)
    print "Test accuracy: %", correct
    
    

    return index, costTraining, costValidation


#2 hiden layer network with regularisation

def twoLayerRegularization(learning_rate, hidden_nodes, epochs, regul, trainXF, trainYV, testXF, testYV, testY, trainY):
    
    Xtrain= trainXF
    Ytrain = trainYV
    Xtest = testXF
    Ytest =testYV
    
    #If only 6 clothes use these instead
    #trainXF2, trainYV2, testXF2, testYV2 = reducedataset(trainY, trainXF, testY, testXF)
    #Xtrain = trainXF2
    #Ytrain = trainYV2
    #Xtest = testXF2
    #Ytest = testYV2

    input_dim = Xtrain.shape[1] 
    output_dim = Ytrain.shape[1]
    regul = 0.01 #test for 0.05, 0.01 and 0.001
    previous_cost = float('inf')
    i = 0
    count = 0
    costTraining = []
    costValidation = []
    finish = False
    epoch = 0
    index = []
    W1 = np.random.randn(input_dim, hidden_nodes) / np.sqrt(input_dim) 
    b1 = np.zeros((1, hidden_nodes)) 
    W2 = np.random.randn(hidden_nodes, hidden_nodes) / np.sqrt(hidden_nodes) 
    b2 = np.zeros((1, hidden_nodes))
    W3 = np.random.randn(hidden_nodes, output_dim) / np.sqrt(hidden_nodes)
    b3 = np.zeros((1, output_dim))

    while (finish == False) and (epoch < epochs):
        samples = Xtrain.shape[0]
        split = int(2* samples/3)
        Xdata = Xtrain[:split]
        Ydata = Ytrain[:split]
        validationX = Xtrain[split:samples]
        validationY = Ytrain[split:samples]


        m = Xdata.shape[0] 
        z1 = Xdata.dot(W1) + b1
        #a1 = np.tanh(z1)
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2  
        #a2 = np.tanh(z2)
        a2 = relu(z2)
        z3 =  np.dot(a2, W3) + b3
        exp_scores = np.exp(z3 - np.max(z3))
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        dW3 = np.dot(a2.T, (output-Ydata)) / float(m)
        db3 = (np.sum(output-Ydata, axis=0, keepdims=True)) / float(m)
        #delta2 = np.dot((output-Ydata), W3.T) * (1 - np.power(z2, 2)) 
        delta2 = np.dot((output-Ydata), W3.T) * drelu(z2)
        dW2 = np.dot(a1.T, delta2) / float(m)

        db2 = np.sum(delta2, axis=0) / float(m)
        #delta1 = np.dot(delta2, W2.T) * (1 - np.power(z1, 2)) 
        delta1 = np.dot(delta2, W2.T) * drelu(z1)
        dW1 = np.dot(Xdata.T, delta1)/ float(m)
        db1 = np.sum(delta1, axis=0) / float(m)
        dW2 += regul * W2
        dW1 += regul * W1
        W2 -= learning_rate* dW2
        W1 -= learning_rate * dW1
        W3 -= learning_rate * dW3
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1
        b3 -= learning_rate * db3
        if (epoch % 10 == 0 and epoch!=0):
            z1 = np.dot(Xdata, W1) + b1
            #a1 = np.tanh(z1)
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2                                          
            #a2 = np.tanh(z2)
            a2 = relu(z2)
            z3 =  np.dot(a2, W3) + b3
            exp_scores = np.exp(z3 - np.max(z3))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsTraining = np.asarray(preds)  
            loss = - np.sum([y * np.log(x) for x, y in zip(out, Ydata)]) 
            loss += regul/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))+  np.sum(np.square(W2))) 
            costTrain =  1/float(m) * loss
            costTraining.append(costTrain)

            z1 = np.dot(validationX, W1) + b1  
            #a1 = np.tanh(z1)
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            #a2 = np.tanh(z2)
            a2 = relu(z2)
            z3 =  np.dot(a2, W3) + b3
            exp_scores = np.exp(z3 -np.max(z3))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsValidation = np.asarray(preds)  
            loss = - np.sum([y * np.log(x) for x, y in zip(out, validationY)]) 
            costVal =  1/float(m) * loss
            costValidation.append(costVal)
            index.append(epoch)
            if (previous_cost !=float('inf')):
                if (costVal > previous_cost): 
                
                    finish = True
            previous_cost = costVal
            
            print epoch, costVal

        epoch+=1
            
    result = Ydata.argmax(axis=1)
    count = 0
    for i in range(len(predsTraining)):  
                if predsTraining[i] == result[i]:
                    count+=1
    count2 = 0
    result2 = validationY.argmax(axis =1)
    for i in range(len(predsValidation)):  
                if predsValidation[i] == result2[i]:
                    count2+=1
    print "Training accuracy: ",  count*100 / float( len(predsTraining))
    print "Validation accuracy: ",  count2*100 / float( len(predsValidation))
    correct = predict2Layers(testXF, testYV, W1,b1,W2,b2,W3,b3)
    print "Test accuracy: %", correct
    

    return index, costTraining, costValidation



#One hidden layer network without regularisation

def oneLayerNoRegularization (learning_rate, hidden_nodes, epochs, trainXF, trainYV, testXF, testYV, testY, trainY):

    Xtrain= trainXF
    Ytrain = trainYV
    Xtest = testXF
    Ytest =testYV
    
    #If only 6 clothes use these instead
    #trainXF2, trainYV2, testXF2, testYV2 = reducedataset(trainY, trainXF, testY, testXF)
    #Xtrain = trainXF2
    #Ytrain = trainYV2
    #Xtest = testXF2
    #Ytest = testYV2

    input_dim = Xtrain.shape[1] 
    output_dim = Ytrain.shape[1]
    previous_cost = float('inf')
    count = 0
    costTraining = []
    costValidation = []
    finish = False
    epoch = 0
    index = []
    learning_rate = 0.01 #test with 0.01 and 0.001
    W1 = np.random.randn(input_dim, hidden_nodes) / np.sqrt(input_dim) 
    b1 = np.zeros((1, hidden_nodes)) 
    W2 = np.random.randn(hidden_nodes, output_dim) / np.sqrt(hidden_nodes) 
    b2 = np.zeros((1, output_dim)) 

    for epoch in range(epochs):
        Xdata = Xtrain
        Ydata = Ytrain

        m = Xdata.shape[0] 
        #z1 = Xdata.dot(W1) + b1
        a1 = np.tanh(z1)
        a1 = relu(z1)
        z2 = a1.dot(W2) + b2  
        exp_scores = np.exp(z2 - np.max(z2)) 
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 

        dW2 = np.dot(a1.T, (output-Ydata)) / float(m)
        db2 = (np.sum(output-Ydata, axis=0, keepdims=True)) / float(m)
        delta2 = np.dot((output-Ydata), W2.T) * (1 - np.power(z1, 2)) 
        #delta2 = np.dot((output-Ydata), W2.T) * drelu(z1)
        dW1 = np.dot(Xdata.T, delta2)/ float(m)
        db1 = np.sum(delta2, axis=0) / float(m)

        W2 -= learning_rate* dW2
        W1 -= learning_rate * dW1
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1  
        if (epoch % 10 == 0 and epoch!=0):
            z1 = np.dot(Xdata, W1) + b1  
            a1 = np.tanh(z1)
           # a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            exp_scores = np.exp(z2 - np.max(z2))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsTraining = np.asarray(preds)  
            loss = - np.sum([y * np.log(x) for x, y in zip(out, Ydata)]) 
            cost =  1/float(m) * loss
            costTraining.append(cost)
            index.append(epoch)



            print epoch, cost
            result = Ydata.argmax(axis=1)
    count = 0
    for i in range(len(predsTraining)):  #OH MY GOOD THIS IS OKAY!! IT GETS 200 K OUT OF 2000 BUT STILL
                if predsTraining[i] == result[i]:
                    count+=1
    print "Training accuracy: %",  count*100 / float( len(predsTraining))
    correct = predict(Xtest, Ytest, W1,b1,W2,b2)
    print "Test accuracy: %", correct
    return index, costTraining

#2 hidden layer network with no regularisation

def twoLayersNoRegularization (learning_rate, hidden_nodes, epochs, trainXF, trainYV, testXF, testYV, testY, trainY):
  
    Xtrain= trainXF
    Ytrain = trainYV
    Xtest = testXF
    Ytest =testYV
    
    #If only 6 clothes use these instead
    #trainXF2, trainYV2, testXF2, testYV2 = reducedataset(trainY, trainXF, testY, testXF)
    #Xtrain = trainXF2
    #Ytrain = trainYV2
    #Xtest = testXF2
    #Ytest = testYV2

    input_dim = Xtrain.shape[1] 
    output_dim = Ytrain.shape[1]
    previous_cost = float('inf')
    i = 0
    count = 0
    costTraining = []
    costValidation = []
    finish = False
    epoch = 0
    index = []
    W1 = np.random.randn(input_dim, hidden_nodes) / np.sqrt(input_dim) 
    b1 = np.zeros((1, hidden_nodes)) 
    W2 = np.random.randn(hidden_nodes, hidden_nodes) / np.sqrt(hidden_nodes) 
    b2 = np.zeros((1, hidden_nodes))
    W3 = np.random.randn(hidden_nodes, output_dim) / np.sqrt(hidden_nodes)
    b3 = np.zeros((1, output_dim))
    for epoch in range(epochs):
        Xdata = Xtrain
        Ydata = Ytrain


        m = Xdata.shape[0] 
        z1 = Xdata.dot(W1) + b1
        #a1 = np.tanh(z1)
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2   
        #a2 = np.tanh(z2)
        a2 = relu(z2)
        z3 =  np.dot(a2, W3) + b3
        exp_scores = np.exp(z3 - np.max(z3))

        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        dW3 = np.dot(a2.T, (output-Ydata)) / float(m)
        db3 = (np.sum(output-Ydata, axis=0, keepdims=True)) / float(m)
        #delta2 = np.dot((output-Ydata), W3.T) * (1 - np.power(z2, 2)) 
        delta2 = np.dot((output-Ydata), W3.T) * drelu(z2)
        dW2 = np.dot(a1.T, delta2) / float(m)

        db2 = np.sum(delta2, axis=0) / float(m)
        #delta1 = np.dot(delta2, W2.T) * (1 - np.power(z1, 2)) 
        delta1 = np.dot(delta2, W2.T) * drelu(z1)
        dW1 = np.dot(Xdata.T, delta1)/ float(m)
        db1 = np.sum(delta1, axis=0) / float(m)

        W2 -= learning_rate* dW2
        W1 -= learning_rate * dW1
        W3 -= learning_rate * dW3
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1
        b3 -= learning_rate * db3
        if (epoch % 10 == 0 and epoch!=0):
            z1 = np.dot(Xdata, W1) + b1
            #a1 = np.tanh(z1)
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2  
            #a2 = np.tanh(z2)
            a2 = relu(z2)
            z3 =  np.dot(a2, W3) + b3
            exp_scores = np.exp(z3 - np.max(z3))
            out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            predsTraining = np.asarray(preds)  
            loss = - np.sum([y * np.log(x) for x, y in zip(out, Ydata)]) 

            cost =  1/float(m) * loss
            costTraining.append(cost)
            index.append(epoch)


            
            print epoch, cost

    
    result = Ydata.argmax(axis=1)
    count = 0
    for i in range(len(predsTraining)):  
                if predsTraining[i] == result[i]:
                    count+=1

    print "Training accuracy: %",  count*100 / float( len(predsTraining))
    correct = predict2Layers(Xtest, Ytest, W1,b1,W2,b2,W3, b3)
    print "Test accuracy: %", correct
    return index, costTraining




colors = ['r', 'g', 'b']

#Plot the training and validation cost functions for the 2 layered network  for different regularisation lambda
epochs = 80

learning_rate = 0.01

hidden_nodes = 128

regul = [0.05, 0.01, 0.001]

for i in range(len(regul)):

    index, costTraining, costValidation =twoLayerRegularization(learning_rate, hidden_nodes, epochs, regul[i], trainXF, trainYV, testXF, testYV, testY, trainY)
    plt.figure(i+1)
    plt.title('Regularization: ' + str(regul[i]))
    plt.plot(index, costTraining, 'r', label = "Training")
    plt.plot(index, costValidation, 'b', label = "Validation")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()



#Visualise cost function of a 2layered network with no regularisation

epochs = 80

learning_rate = 0.01

hidden_nodes = 128


index, costTraining = twoLayersNoRegularization (learning_rate, hidden_nodes, epochs, trainXF, trainYV, testXF, testYV, testY, trainY)
plt.plot(index, costTraining, 'b')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()    



#Visualising the effet of using different number of neurons for the activation function tanh. 
epochs = 40
learning_rate = 0.01
hidden_nodes = [64, 128, 248]
costsPlot = []
for i in range(len(hidden_nodes)):

    index, costTraining = oneLayerNoRegularization(learning_rate, hidden_nodes[i], epochs, trainXF, trainYV, testXF, testYV, testY, trainY)
    plt.plot(index, costTraining, colors[i] , label = str(hidden_nodes[i]))


plt.xlabel('epoch')
plt.ylabel('cost')
legend = plt.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



#Plot for different learning rates
learning_rates = [0.01, 0.001]
epochs = 80
hidden_nodes = 128
for i in range(len(hidden_nodes)):

    index, costTraining = oneLayerNoRegularization(learning_rate[i], hidden_nodes, epochs, trainXF, trainYV, testXF, testYV, testY, trainY)
    plt.plot(index, costTraining, colors[i] , label = str(learning_rates[i]))


plt.xlabel('epoch')
plt.ylabel('cost')
legend = plt.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



#Plot the training and validation accuracies for classifying 6 clothes using ReLU, epochs = 80 and hidden nodes = 128. 
epochs = 80
hidden_nodes = 128
regul = 0.01

index, costTraining, costValidation = oneLayerRegularization(learning_rate, hidden_nodes, epochs, regul, trainXF, trainYV, testXF, testYV, testY, trainY)

plt.title('Classification of 6 clothes')
plt.plot(index, costTraining, 'r', label = "Training")
plt.plot(index, costValidation, 'b', label = "Validation")
plt.xlabel('epoch')
plt.ylabel('cost')
legend = plt.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()




    

