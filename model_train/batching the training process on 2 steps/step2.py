
# Start empty line

# end empty line
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
import scipy


path_to_save_classifier = '/home/wessam/Desktop/Maher/newClassifier'
path_to_word2vec        = '/home/wessam/Desktop/Maher/GoogleNews-vectors-negative300.bin.gz'
path_to_datasetWords    = '/home/wessam/Desktop/Maher/datasetWordsTokenized.txt'


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

# In[3]:


X_train_loaded = scipy.sparse.load_npz('/home/wessam/Desktop/Maher/savedSparse/X_train_sparse.npz')
Y_train_loaded = scipy.sparse.load_npz('/home/wessam/Desktop/Maher/savedSparse/Y_train_sparse.npz')


# In[5]:


X_train = np.array(X_train_loaded.A)
Y_train = np.array(Y_train_loaded.A)


# In[10]:


X_train = np.reshape(X_train, (108846, 1, 11000))


# In[ ]:



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
# RNNClassifier.add(Dense(600, activation = 'sigmoid',  input_shape = (None, 10024)))
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

# print(RNNClassifier.evaluate(X_test, Y_test, verbose=0))
        # print('\\nTesting loss: {}, acc: {}\\n'.format(loss, acc))

# model_json = RNNClassifier.to_json()
# with open("CNNClassifier.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
# RNNClassifier.save_weights("model.h5") # TODO: check right path
# print("Saved model to disk")
 
tfjs.converters.save_keras_model(RNNClassifier, path_to_save_classifier)

#############################################################################
    

thefile = open("common_words.txt", 'w')
for item in common_words:
  thefile.write("%s\n" % item)

thefile = open("bag_of_words.txt", 'w')
for item in bag_of_words:
  thefile.write("%s\n" % item)

