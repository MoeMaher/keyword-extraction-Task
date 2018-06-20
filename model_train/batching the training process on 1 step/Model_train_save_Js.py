import nltk
import gensim
from nltk.probability import FreqDist
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional, Dropout,Embedding
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflowjs as tfjs
from scipy.sparse import csr_matrix


path_to_save_classifier = '/home/wessam/Desktop/Maher/newClassifier'
path_to_word2vec        = '/home/wessam/Desktop/Maher/GoogleNews-vectors-negative300.bin.gz'
path_to_datasetWords    = '/home/wessam/Desktop/Maher/datasetWordsTokenized.txt'

#############################################################################

# a call back function that print the accuracy and the confusion matrix by the end of each epoch
class TestCallback(Callback):
    def __init__(self, test_data, y_test, common_words):
        self.test_data = test_data
        self.y_test = y_test
        self.common_words = common_words
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        
        pred = (self.model.predict(x),0).flatten()
        print(pred[0])
        for i in range(len(pred)) :
            if(pred[i] > 0.7 and self.y_test[i] not in self.common_words):
                pred[i] = 1
            else :
                pred[i] = 0
        print(pred[0])
        print(confusion_matrix(y.flatten(),pred ))
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\\nTesting loss: {}, acc: {}\\n'.format(loss, acc))

# returns the indices of the words and the strange words will be -1
def indexEncoder(strs, bagOfWords):
    out = []
    for word in strs:
        if word in bagOfWords:
            out.append([bagOfWords.index(word)])
        else:
            out.append([-1])
    return out
    
def createBagOfWords(words):
    return list(set(words))

#############################################################################

print('loading the word2vec module ...')
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True  )

#############################################################################

print('loading used words ...')
f = open(path_to_datasetWords)
lines = f.readlines()

#############################################################################
# tokenize and create trains

strings = [""] * len(lines)
Y_train = [0] * len(lines)
title = [0] * len(lines) 


for i in range(len(lines)) :
	l = nltk.word_tokenize(lines[i])
	strings[i] = l[0]
	title[i] = l[1]
	Y_train[i] = l[2]

#############################################################################
# parsing the Y train

Y_train_main = [int(label) for label in Y_train]
X_train_main = strings

#############################################################################
# collecting the 100 most common words to discard it when predicting

print('analizing words ...')
freqd = FreqDist(X_train_main)
common_words = [ w[0] for w in freqd.most_common(100)]
with open("common_words.txt", "wb") as fp:   #Pickling
    pickle.dump(common_words, fp)

#############################################################################

print('processing the base words ...')
x_train = X_train_main
y_train = Y_train_main

y_train = [y_train[i] for i in range(len(y_train)) if x_train[i] in model.vocab]
x_train = [word for word in x_train if word in model.vocab] # array of words
# x_train = list(nltk.bigrams(x_train)) # I was gonna use bigrams
y_train = y_train[:len(x_train)]

print(len(x_train))
print(len(y_train))

# this is used in developing envirnoment while testing the hyper parameters

# y_test = [ y_test[i] for i in range(len(y_test)) if x_test[i] in model.vocab and x_test[i] in x_train ]
# x_test = [ word for word in x_test if word in model.vocab and word in x_train] # array of words
# # x_test = list(nltk.bigrams(x_test))
# y_test = y_test[:len(x_test)]

bag_of_words = createBagOfWords(x_train);

print('encoding the words ...')
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(x_train)
integer_encoded = indexEncoder(x_train, bag_of_words)

print(integer_encoded[0:10])

onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_train = onehot_encoder.fit_transform(integer_encoded)
X_train = np.array([[w] for w in X_train])

# test_integer_encoded = label_encoder.transform(x_test)
# test_integer_encoded = test_integer_encoded.reshape(len(test_integer_encoded), 1)
# X_test = onehot_encoder.transform(test_integer_encoded)
# X_test = np.array([[w] for w in X_test])

Y_train = np.array(y_train) # answers to vectors
# X_train = np.array([[np.append(model[word],freqd.freq(word))] for word in x_train]) # array of vectors

# Y_test = np.array(y_test) # answers to vectors
# X_test = np.array([[np.append(model[word],freqd.freq(word))] for word in x_test]) # array of vectors

#############################################################################

# print('saving x_train ..')
# with open("x_train.txt", "wb") as fp:   #Pickling
#     pickle.dump(x_train, fp)
# print('saving y_train ..')
# with open("y_train.txt", "wb") as fp:   #Pickling
#     pickle.dump(y_train, fp)
# print('saving x_test ..')
# with open("x_test.txt", "wb") as fp:   #Pickling
#     pickle.dump(x_test, fp)
# print('saving y_tr ..')
# with open("y_test.txt", "wb") as fp:   #Pickling
#     pickle.dump(y_test, fp)


#############################################################################

print('intializing the classifier ...')
RNNClassifier = Sequential()
RNNClassifier.add(LSTM(124,  input_shape = (None, len(X_train[0][0]))))
RNNClassifier.add(Dense(64, activation = 'sigmoid'))
RNNClassifier.add(Dense(1, activation='sigmoid'))

#############################################################################

RNNClassifier.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

#############################################################################


print('fitting the classifier ...')
history = RNNClassifier.fit(X_train, Y_train, batch_size=300, epochs=20)
        #   callbacks=[TestCallback((X_test, Y_test), x_train, common_words )])

#############################################################################

# save the model for python uses! #########

print(RNNClassifier.evaluate(X_test, Y_test, verbose=0))
        print('\\nTesting loss: {}, acc: {}\\n'.format(loss, acc))

model_json = RNNClassifier.to_json()
with open("CNNClassifier.json", "w") as json_file:
    json_file.write(model_json)
serialize weights to HDF5
RNNClassifier.save_weights("model.h5") # TODO: check right path
print("Saved model to disk")

# save the model for Javascript uses! ######
 
    # tfjs.converters.save_keras_model(RNNClassifier, path_to_save_classifier)

#############################################################################
    
# save the tools needed for further predictions.

thefile = open("common_words.txt", 'w')
for item in common_words:
  thefile.write("%s\n" % item)

thefile = open("bag_of_words.txt", 'w')
for item in bag_of_words:
  thefile.write("%s\n" % item)

