import pandas as pd
import numpy as np


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

def mean_sq_error(features, preds, target):
#     error = np.square(target - preds)
    error = (1/(2*features.shape[0])) * np.power(np.sum(target - preds),2)
    return error

def gradient(features, preds, target, weights, weight):
#     gradient = -2 * np.dot(features.T, target) + 2 * np.dot(features.T, features).dot(weights)
    gradient = -(1/features.shape[0]) * np.sum(np.dot((target - preds), features[:,weight]))
    return gradient
             

def linear_regression(features, target, learning_rate, num_steps):
    intercept = np.ones((features.shape[0],1))
    features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1])
    
    for i in range(num_steps):
        preds = np.dot(features, weights)
        error = mean_sq_error(features, preds, target)
        if i%100 == 0:
            print(error)
        for w in range(weights.shape[0]):
            grad = gradient(features, preds, target, weights, w)
            weights[w] -= learning_rate * grad
    return weights

weights = linear_regression(data, classes, 0.04, 3000)
intercept = np.ones((data.shape[0],1))
# data_with_intercept = np.hstack((intercept, data))   
# preds = np.dot(data_with_intercept, weights)
    
traindata_with_intercept = np.hstack((np.ones((traindata.shape[0], 1)),traindata))
final_scores = np.dot(traindata_with_intercept, weights)
trainpreds = np.round(np.dot(traindata_with_intercept, weights))


testdata_with_intercept = np.hstack((np.ones((testdata.shape[0], 1)),testdata))
final_scores = np.dot(testdata_with_intercept, weights)
testpreds = np.round(np.dot(testdata_with_intercept, weights))

# for i in range(preds.shape[0]):
#     if preds[i] < 0.5:
#         preds[i]= 0
#     else: 
#         preds[i]= 1
    
def accuracy(preds, target):
    count = 0
    length = target.shape[0]
    for i in range(length):
        if preds[i] == target[i]:
            count+=1
    return 100*count/length


# print('The accuracy is {}'.format(accuracy(preds, classes)))    
print('The training set accuracy is {}'.format(accuracy(trainpreds, trainclasses)))
print('The testing set accuracy is {}'.format(accuracy(testpreds, testclasses)))             
    