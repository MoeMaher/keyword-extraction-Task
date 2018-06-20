
# coding: utf-8

# In[2]:


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


# In[3]:


path_to_save_classifier = '/home/wessam/Desktop/Maher/newClassifier'
path_to_word2vec        = '/home/wessam/Desktop/Maher/GoogleNews-vectors-negative300.bin.gz'
path_to_datasetWords    = '/home/wessam/Desktop/Maher/datasetWordsTokenized.txt'
#############################################################################

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

strings = [""] * len(lines)
Y_train = [0] * len(lines)
title = [0] * len(lines) 


for i in range(len(lines)) :
	l = nltk.word_tokenize(lines[i])
	strings[i] = l[0]
	title[i] = l[1]
	Y_train[i] = l[2]

#############################################################################

Y_train_main = [int(label) for label in Y_train]
X_train_main = strings

#############################################################################

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
# x_train = list(nltk.bigrams(x_train))
y_train = y_train[:len(x_train)]

print(len(x_train))
print(len(y_train))

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


# In[4]:


onehot_encoder = OneHotEncoder(sparse=True)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)


# In[15]:


X_train_sparse = onehot_encoder.fit_transform(integer_encoded)
Y_train_sparse = csr_matrix(np.array(y_train))


# In[16]:


scipy.sparse.save_npz('/home/wessam/Desktop/Maher/savedSparse/X_train_sparse.npz', X_train_sparse)
scipy.sparse.save_npz('/home/wessam/Desktop/Maher/savedSparse/Y_train_sparse.npz', Y_train_sparse)

