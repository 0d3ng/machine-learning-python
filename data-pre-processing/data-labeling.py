import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

test_labels = ['green', 'red', 'black']
encoder_values = encoder.transform(test_labels)
print("Labels {}".format(test_labels))
print("Encoder {}".format(list(encoder_values)))
encoder_values = [3,0,4,1]
decoder_list = encoder.inverse_transform(encoder_values)
print("Encoded value= {}".format(encoder_values))
print("Decode labels= {}".format(list(decoder_list)))
