import pandas as pd
import numpy as np 
import math
import pandas as pd
import numpy as np


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'class']
dataset = pd.read_csv(url, names=names)


array = dataset.values


dataset = np.hstack((data1, data4)).astype(np.float32)
data = np.hstack((data1, data3)).astype(np.float32)

for i in range(data.shape[1]):
    for j in range(data.shape[0]):
        mean = data[:,i].mean()
        std = data[:,i].std()
        data[j,i] = (data[j,i] - mean)/std

traindata = dataset[0:600]
testdata = dataset[600:699]


classes = array[:,10:11]
classes = classes.reshape(classes.shape[0],)


trainclasses = classes[0:600]  
testclasses = classes[600:699]

#makes dataset into a list!
#could've been array
def test_split(index, value, dataset):
    left = list()
    right = list()
    for instance in dataset:
        if instance[index] < value:
            left.append(instance)
        else:
            right.append(instance)
    return left, right        

#on iterations the datatset it turned into dictionary causes problems




def parententropy(groups, classes):
    parententropy = 0
    n_instances = float(sum([len(group) for group in groups]))
    count = 0.0
    for class_val in classes:
        for group in groups:
            count += [row[-1] for row in group].count(class_val)
        p1 = count/n_instances
        parententropy -= p1 * math.log(p1, 2)
    return parententropy


def info_gain(groups, classes):
    gain = parententropy(groups, classes)
    n_instances = float(sum([len(group) for group in groups]))
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        for class_val in classes:
            p2 = [row[-1] for row in group].count(class_val) / size
            if p2 == 0:
                continue
            gain -= p2 * math.log(p2, 2) * (size / n_instances)
    return gain


# def entropy(groups):
#     yvals = list()
#     for group in groups:
#         yvals.append([row[-1] for row in group])
    

def get_split(dataset):
#     max = np.zeros((dataset.shape[1]-1,))
    min = 999
    class_values = list(set(row[-1] for row in dataset))
    for i in range(len(dataset[0])-1):
        for row in dataset:
            value = row[i]
            index = i
            groups = test_split(index, value, dataset)
            gain = info_gain(groups, class_values)
            if gain < min:
                b_index, b_value, min, b_groups = index, value, gain, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
                
    

def to_terminal(group):
    classes = [row[-1] for row in group]
    return max(set(classes), key = classes.count)

#recursive function
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    #check for no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    #check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    #process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
    
# Build a decision tree 
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        #does it contain another level?
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        #does it contain another level?
        #to_terminal makes it a value instead of dictionary entry
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
#time to build the algorithm!
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions
    

predictions = decision_tree(traindata, testdata, 5, 10)

def accuracy(predictions):
    count = 0
    for p in range(len(predictions)):
        if predictions[p] == testclasses[p]:
            count+=1
    return 100 * count/len(predictions)

print ('The accuracy is {}'.format(accuracy(predictions)))