import mido 
from mido import MidiFile 
import os
from sklearn import svm 
import numpy as np


def find_difference(vector1, vector2):

    note_differences = []

    for i in range(len(vector1)):
        if vector1[i] != vector2[i]:
            note_differences.append(i)

    
    return note_differences



# First you have to find all the files in the folder
# Train Data
audio_files = os.listdir('validation/groundTruth/')

# Creating Our Train Dataset
data_x = [] # features
data_y = [] # labels

for file in audio_files:
    mid_file = MidiFile('validation/groundTruth/' + file)
    #print(mid_file)

    vector = []

    for i, track in enumerate(mid_file.tracks): 
        for msg in track: 
            if hasattr(msg, 'note'): 
                vector.append(msg.note)

    #print(vector)

    #print(vector)
    #print(100*'-')

    for idx, msg in enumerate(vector[3:-4]):
        #print(idx)

        feature_vector = vector[idx:idx+3] + vector[idx+4:idx+7]
        data_x.append(feature_vector)
        data_y.append(msg)
        #print(feature_vector)
    #print(vector[3:-3])
    #print(100*'*')

print(len(data_x))
print(100*'-')
print(len(data_y))

# Creating Classifier
classifier = svm.SVC()

# Training Our Model
classifier.fit(data_x, data_y)


#label = classifier.predict(np.reshape(data_x[0], (-1,6)))

#print(label)

# Predicting on Query data
error = 0.0
all = 0.0
audio_query_files = os.listdir('validation/query')

for file in audio_query_files:
    mid_file = MidiFile('validation/query/' + file)
    #print(mid_file)

    vector = []

    for i, track in enumerate(mid_file.tracks): 
        for msg in track: 
            if hasattr(msg, 'note'): 
                vector.append(msg.note)

    #print(vector)
    #print(100*'*')

    predicted_vector = vector[0:3]

    for idx, msg in enumerate(vector[0:-7]):
        #print(idx)

        feature_vector = vector[idx:idx+3] + vector[idx+4:idx+7]
        predict_msg = classifier.predict(np.reshape(feature_vector, (-1,6)))
        predicted_vector.append(predict_msg[0])

    predicted_vector.append(vector[len(vector)-4])
    predicted_vector.append(vector[len(vector)-3])
    predicted_vector.append(vector[len(vector)-2])
    predicted_vector.append(vector[len(vector)-1])
    

    #print(predicted_vector)
    #print(100*'-')

    mid_file = MidiFile('validation/groundTruth/' + file)
    #print(mid_file)

    truth_vector = []

    for i, track in enumerate(mid_file.tracks): 
        for msg in track: 
            if hasattr(msg, 'note'): 
                truth_vector.append(msg.note)

    #print(truth_vector)
    #print(100*'%')

    note_differences = find_difference(predicted_vector, truth_vector[:-1])

    #print(note_differences)

    all = all + len(predicted_vector)
    error = error + len(note_differences)

print((all-error)/all)

    

