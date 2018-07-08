import pandas as pd
import numpy as np
from scipy.special import expit, logit


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'class']
dataset = pd.read_csv(url, names=names)

array = dataset.values


data1 = array[:,1:6]
data3 = array[:,7:10]
data = np.hstack((data1, data3)).astype(np.float32)

for i in range(data.shape[1]):
    for j in range(data.shape[0]):
        mean = data[:,i].mean()
        std = data[:,i].std()
        data[j,i] = (data[j,i] - mean)/std

print(data.shape)
traindata = data[0:600]

print(data.shape)
testdata = data[600:699]

        
classes = array[:,10:11]
classes = classes.reshape(classes.shape[0],)

for i in range(classes.shape[0]):
    if classes[i] == 2:
        classes[i] = 0
    if classes[i] == 4:
        classes[i] = 1

trainclasses = classes[0:600]  
testclasses = classes[600:699]
        

                    
def loglikelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1+np.exp(scores)))
    return ll

    
def sigmoid(scores):
    return expit(scores)

    
def logistic_regression(features, target, steps, learning_rate):
    intercept = np.ones((features.shape[0],1))
    features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1])
    
    for step in range(steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        error = target - predictions
        gradient = np.array(np.dot(features.T, error), dtype= np.float32)
        weights += learning_rate * gradient
        if step%1000 == 0:
            print(loglikelihood(features, target, weights))
    return weights

weights = logistic_regression(traindata, trainclasses, 15000, 5e-5)

traindata_with_intercept = np.hstack((np.ones((traindata.shape[0], 1)),traindata))
final_scores = np.dot(traindata_with_intercept, weights)
trainpreds = np.round(sigmoid(final_scores))


testdata_with_intercept = np.hstack((np.ones((testdata.shape[0], 1)),testdata))
final_scores = np.dot(testdata_with_intercept, weights)
testpreds = np.round(sigmoid(final_scores))

def accuracy(preds, target):
    count = 0
    length = target.shape[0]
    for i in range(length):
        if preds[i] == target[i]:
            count+=1
    return count/length

print('The training set accuracy is {}'.format(accuracy(trainpreds, trainclasses)))
print('The testing set accuracy is {}'.format(accuracy(testpreds, testclasses)))
        