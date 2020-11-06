import mido
from mido import MidiFile
import os
from sklearn import svm
import numpy as np

def pre_process(vector, model):

    #model
    pre_pro_vector = []

    for idx in range(len(vector)):

        if vector[idx] == 0 and idx > 11:
            predict = model.predict(np.reshape(vector[idx-10:idx], (-1,10)))
            vector[idx] = predict

    pre_pro_vector = vector    

    return pre_pro_vector

def train_predictor():

    train_dir = os.listdir('train_predict_set/')

    # Reading Our Train and Test Dataset
    data_x = [] # features
    data_y = [] # labels
    counter = 0
    for file in train_dir:
        mid_file = MidiFile('train_predict_set/' + file)
        print(counter)
        counter = counter + 1
        # file_name = mid_file.filename.split('/')[-1]
        #print(file_name)
        vector = []

        for i, track in enumerate(mid_file.tracks):
            for msg in track:
                if hasattr(msg, 'note'):
                    if msg.velocity != 0:
                        vector.append(msg.note)

        for idx in range(1, len(vector), 20):

            if idx < len(vector) - 11:
                feature_vector = vector[idx:idx+10]
                data_x.append(feature_vector)
                data_y.append(vector[idx+10])
        #print(train_y[file_name])

    print(len(data_x))
    # Creating Classifier
    classifier = svm.SVC()

    # Training Our Model
    classifier.fit(data_x, data_y)

    return classifier
    

    







    
