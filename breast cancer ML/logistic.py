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

        
traindata = data[0:600]
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
        


def cost(features, scores, target):
    cost = -(1/features.shape[0])*np.sum(np.dot(target,np.log(expit(scores))+(1-target)*np.log(1-expit(scores))))
    return cost
                                         

def gradient(features, scores, target, weight):
    gradient = np.sum(np.dot(expit(scores) - target, features[:,weight]))
    return gradient
    
def sigmoid(scores):
    return expit(scores)

      
def logistic_regression(features, target, steps, learning_rate):
    intercept = np.ones((features.shape[0],1))
    features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1])
    
    for step in range(steps):
        scores = np.array(np.dot(features, weights),dtype=np.float32)
        predictions = sigmoid(scores)
        error = target - predictions
#         gradient = np.array(np.gradient(loglikelihood(features, classes, weights)),dtype=np.float32)
#         gradient = np.array(np.dot(features.T, error),dtype=np.float32)
        for w in range(weights.shape[0]):
            grad = gradient(features, scores, target, w)
            weights[w] -= learning_rate * grad
#         weights -= learning_rate * gradient
        if step%1000 == 0:
            print(cost(features, scores, classes))
    return weights

weights = logistic_regression(data, classes, 11000, 5e-7)

# data_with_intercept = np.hstack((np.ones((data.shape[0], 1)),data))
# final_scores = np.array(np.dot(data_with_intercept, weights),dtype=np.float32)
# preds = np.round(sigmoid(final_scores))

# def accuracy(preds, target):
#     count = 0
#     length = target.shape[0]
#     for i in range(length):
#         if preds[i] == target[i]:
#             count+=1
#     return count/length

# print('The accuracy is {}'.format(accuracy(preds, classes)))
        
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
    
    