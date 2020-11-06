
import mido
from mido import MidiFile
import os
from sklearn import svm
import numpy as np
from pre_process import *


def feature_extraction(vector):
    result = []
    notes = np.array(vector)
    #result.append(notes.min())
    result.append(notes.max())
    result.append(np.average(vector))

    repeated_note = np.zeros(150)
    cross_times = 0
    positive = True
    diff_average = 0.0
    diff_max = 0
    for idx in range(len(vector)-1):

        repeated_note[vector[idx]] = repeated_note[vector[idx]] + 1

        diff = vector[idx] - vector[idx + 1]
        curr_positive = False
        if diff < 0:
            curr_positive = False
        else:
            curr_positive = True

        if curr_positive != positive:
            cross_times = cross_times + 1

        positive = curr_positive

        if abs(diff) > abs(diff_max):
            diff_max = abs(diff)

        diff_average = diff_average + diff

    result.append(diff_max)
    #result.append(diff_average/len(vector))

    max_repeated_note_index = 0
    max_repeated_note = 0
    for idx in range(len(repeated_note)):
        if repeated_note[idx] > max_repeated_note:
            max_repeated_note = repeated_note[idx]
            max_repeated_note_index = idx

    result.append(max_repeated_note_index)
    result.append(cross_times)

    return result


def feature_extraction_2(vector):
    result = []
    notes = np.array(vector)

    repeated_note = np.zeros(128)
    diff_max = 0
    for idx in range(len(vector)-1):

        repeated_note[vector[idx]] = repeated_note[vector[idx]] + 1

        diff = vector[idx] - vector[idx + 1]

        if abs(diff) > abs(diff_max):
            diff_max = diff

    for note in repeated_note:
        result.append(note)
    # result.append(diff_max)
    # result.append(diff_average/len(vector))

    result.append(notes.min())
    result.append(notes.max())
    result.append(np.average(vector))
    # max_repeated_note_index = 0
    # max_repeated_note = 0
    # for idx in range(len(repeated_note)):
    #     if repeated_note[idx] > max_repeated_note:
    #         max_repeated_note = repeated_note[idx]
    #         max_repeated_note_index = idx

    # result.append(max_repeated_note_index)
    #result.append(cross_times)
    # print(len(result))

    return result


model = train_predictor()
# First you have to find all the files in the folder
train_dir = os.listdir('train_set/')
test_dir = os.listdir('python_test_set/')

# Reading Our Train and Test Dataset
train_x = {}
train_y = {}

test_x = {}
test_y = {}

# Then Reading our Y Data. train_y and test_y

filepath = 'trainLabels.txt'
with open(filepath) as fp:
    line = fp.readline()
    counter = 1
    while line:
        temp = line.split(',')
        train_y[temp[0]] = int(temp[1][0])
        #print(temp[0] + ' , ' + train_y[temp[0]])
        line = fp.readline()
        counter += 1

print(len(train_y))

filepath2 = 'testLabel.txt'
with open(filepath2) as fp:
    line = fp.readline()
    counter = 1
    while line:
        temp = line.split(',')
        test_y[temp[0]] = int(temp[1][0])
        #print(temp[0] + ' , ' + test_y[temp[0]])
        line = fp.readline()
        counter += 1

print(len(test_y))



# Then Reading our X Data
# Train Data

for file in train_dir:
    mid_file = MidiFile('train_set/' + file)
    file_name = mid_file.filename.split('/')[-1]
    #print(file_name)
    vector = []

    for i, track in enumerate(mid_file.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                if msg.velocity != 0:
                    vector.append(msg.note)

    train_x[file_name] = vector
    #print(train_y[file_name])

print(len(train_x))

# Test Data



for file in test_dir:
    mid_file = MidiFile('python_test_set/' + file)
    file_name = mid_file.filename.split('/')[-1]

    vector = []

    for i, track in enumerate(mid_file.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                if msg.velocity != 0:
                    vector.append(msg.note)
    pre_pro_vector = pre_process(vector, model)
    test_x[file_name] = pre_pro_vector
    #print(test_x[file_name])

print(len(test_x))

# Creating Classifier
classifier = svm.SVC()

# Training Our Model
# Before Training our Model we should fix our data cause its not on the correct format and order

train_x_data = []
train_y_data = []

train_x_keys = list(train_x.keys())

print(train_x_keys)

for key in train_x_keys:
    print(key)
    print(train_y[key])
    train_x_data.append(feature_extraction(list(train_x[key])))
    train_y_data.append(train_y[key])

print(train_x_data[0])
print(train_x_data[1])
print(train_x_data[2])
print(train_x_data[3])
print(train_x_data[4])
print(train_x_data[5])

print(len(train_x_data))
print(len(train_y_data))

# Now Training our Model
classifier.fit(train_x_data, train_y_data)


# Evaluate our Model
error = 0.0

test_x_keys = list(test_x.keys())


for key in test_x_keys:
    print(100*'-')
    print(key)
    data = feature_extraction(list(test_x[key]))
    print(data)
    predict = classifier.predict(np.reshape(data, (-1,5)))
    print(predict)
    print(test_y[key])
    if predict != test_y[key]:
        error = error + 1
        print('error')

print('error = ' + str(error))
print('accuracy = ' + str(1 - (error/len(test_x_keys))))











