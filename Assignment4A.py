""" 000825410_1.csv, 000825410_2.csv this two files are linearly seperable and I have
determined that using the accuracy of Perceptron Learning Algorithm to solve the dataset. Whilst, other two files 000825410_3.csv, 000825410_4.csv has the accuracy just above 50% which says that the files are not linerly seperable.

Prerak Patel, Student, Mohawk College, 2020
"""

import numpy as np
import csv

## READ IT
def read(file):
    train_data_file = open(file,"r")

    # creating CSV readers
    csv_reader = csv.reader(train_data_file, delimiter=",")

    # declaring the arrays for the storing data
    data_set = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for row in csv_reader:
        data_set += [[str(num) for num in row]]

    # converting the data set to np array for slicing
    data_set = np.array(data_set)

    # shuffling the array for each run
    np.random.shuffle(data_set)

    data_set = np.array((data_set.astype(np.float)))

    # slicing the data_set for the data
    data = data_set[:,:-1]
    # slicing the data_set for the labels
    labels = data_set[:,-1]

    # using the first 80% of the data set into training data
    train_data = data[:int(len(data_set)*0.8)]
    train_labels = labels[:int(len(data_set)*0.8)]

    # using the other 20% of the data set into testing data
    test_data = data[int(len(data_set)*0.8):]
    test_labels = labels[int(len(data_set)*0.8):]
    return train_data, train_labels, test_data, test_labels

## Training the data
def fit(train_data,train_labels,epochs):
    # initialize the weights (ws), threshold (t), and learning rate (lr)
    ws = [0]*len(train_data[0])
    t = 0
    lr = 0.1
    # Adjusting the weights number of times the epochs
    for i in range(epochs):
        # looping through train_data
        for index in range(len(train_data)):
            # training data
            xs = train_data[index]
            # # target label
            target = train_labels[index]
            # compute perceptron output using sum(ws * xs)
            a = sum(ws*xs)

            # if the sum is less than thresold output 0
            if a <= t:
                output = 0.0
            # else the sum is greater than thresold output 1
            elif a > t:
                output = 1.0

            # if output < target
            if output < target:
                #   w = w + x * lr, t = t – lr
                ws = ws + xs * lr
                t = t - lr

            # else if output > target
            elif output > target:
                #   w = w - x * lr, t = t + lr
                ws = ws - xs * lr
                t = t + lr
    return ws, t

## Testing the data
def predict(test_data, test_labels, ws, t):
    correct_pred = 0
    # looping through testing data
    for index in range(len(test_data)):
        xs = test_data[index]
        target = test_labels[index]
        # compute perceptron output using sum(ws * xs)
        a = sum(ws*xs)

        # if activation <= thresold
        if a <= t:
            output = 0.0
            # if target and output matches
            if target == output:
                correct_pred += 1

        # if activation > thresold
        elif a > t:
            output = 1.0
            # if target and output matches
            if target == output:
                correct_pred += 1
    return (correct_pred*100)/len(test_labels)

# creating file array
files = ['000825410_1.csv', '000825410_2.csv', '000825410_3.csv', '000825410_4.csv']

# looping through each file and run the algorithm on each file
for eachFile in files:
    train_data, train_labels, test_data, test_labels = read(eachFile)
    weights, thresold = fit(train_data,train_labels,1000)
    accuracy = predict(test_data, test_labels, weights, thresold)
    print(eachFile + ": " + str(np.around(accuracy, decimals=1)) + "% W: " + str(np.around(weights[0:3], decimals=1)) + " T: " + str(np.around(thresold, decimals=1)))


