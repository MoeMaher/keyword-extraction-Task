
# coding: utf-8

# # Keywords extraction

# **Things need to be configured:**
# 
# - The weird ratio = P(mostProbable)-P(word) > (P(word)/2)
# - The accuracy ratio = 0.7
# - Remove Top N common keywords; N = 30

# ## Imports

# In[1]:


import pandas as pd
import nltk
from nltk.tokenize import regexp_tokenize
import nltk.data
import pattern3
import re
from collections import Counter
from textblob import TextBlob
from nltk.corpus import brown
word_list = brown.words()
from nltk.probability import FreqDist
import os.path
import nltk
import gensim
from keras.models import model_from_json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
import csv


# ## Discover and visualize the data to gain insights

# ## getting the data ready

# In[2]:


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


# In[3]:


docs = pd.read_csv('documents-fixed.csv')


# In[4]:


docs.shape


# In[5]:


docs['content'][0]


# In[6]:


# docs["core_std"].value_counts()


# In[7]:


# Concatinate all words from all the documents.
allDocuments = ''
for i in range(len(docs)):
    if(not isinstance(docs['content'][i], type(0.0))):
        allDocuments = allDocuments + str(docs['content'][i])
print(len(allDocuments))


# ### Intializing Variables

# In[8]:


CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 


# In[9]:


WORDS = Counter(['m'])


# ### creating pipeline functions for handling the data (content) 

# functions that handle the text format and style of writing

# In[10]:


# converts the tokens to lowercase/uppercase
def convert_letters(tokens, style = "lower"):
    if (style == "lower"):
        tokens = [token.lower() for token in tokens]
    else :
        tokens = [token.upper() for token in tokens]
    return(tokens)

# remove blancs from text 
def remove_blanc(tokens):
    tokens = [token.strip() for token in tokens]
    return(tokens)

# expand contractions ex. this's -> this is
def expand_contractions(sentence, contraction_mapping=CONTRACTION_MAP): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 

# convert the text into unicode
def remove_accent(tokens):
    tokens = [unidecode.unidecode(token) for token in tokens]
    return(tokens)

# remove the stopwords from the tokenized text 
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(wordlist, stopwords=stopword_list):
    return [w for w in wordlist if w not in stopwords]


# functions that handle the spelling mistakes 

# In[11]:


def words(text): return re.findall(r'\w+', text.lower())

# Probability of `word`.
def P(word, N=sum(WORDS.values())): 
    global WORDS
    return WORDS[word] / N

# Most probable spelling correction for word relative to the corpus in WORDS.
def correct(word): 
    if len(word) > 1:
        return max(candidates(word), key=P)
    else:
        return word
    
# if there is another word in the documents similar to the input world with a relativily high
# occurance in the documnets return the it
# else return the input word 
def properify(word): 
    mostProbable = max(candidates_weird(word), key=P)
    if(known([word]) and P(mostProbable)-P(word) > (P(word)/2) and len(word) > 2):
        return mostProbable
    else:
        return word

# Generate possible spelling corrections for word
def candidates(word): 
    return  known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

# Generate possible spelling similar to the word
def candidates_weird(word): 
    return known(edits1(word)) or known(edits2(word)) or [word]

# The subset of `words` that appear in the dictionary of WORDS
def known(words): 
    global WORDS
    return set(w for w in words if w in WORDS)

# All edits that are one edit away from `word`.
def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

# All edits that are two edits away from `word`.
def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# impractical dont use it
# All edits that are three edits away from `word`.
def edits3(word): 
    return (e3 for e1 in edits1(word) for e3 in edits2(e1))

# take a list of tokens and call correct on each token
def correct_from_tokens(tokens):
    global WORDS
    WORDS = Counter(words(' '.join(word_list)))
    return [correct(w) for w in tokens]
    
# take a list of tokens and call properify on each token
def remove_weird_from_tokens(tokens):
    global WORDS
    WORDS = Counter(words(allDocuments))
    return [properify(w) for w in tokens]
    
# ** not used **
def text_blob_clean(tokens):
    cleanBlob = TextBlob(' '.join(tokens))
    return cleanBlob.correct()
    


# ## Importing the trained module

# In[12]:


# the location of the classifier
path_to_classifier = '/home/maher/Desktop/optimaKeywordsTask/savedModel'

# predict method take an input the document tokenized and output a list of predictions the 
# rest of the parameters are self explanatory
def predict(inWords, bag_of_words, most_common, classifier, label_encoder, onehot_encoder, accuracy):

    inWords = [w for w in inWords if w in bag_of_words]

    test_integer_encoded = label_encoder.transform(inWords)
    test_integer_encoded = test_integer_encoded.reshape(len(test_integer_encoded), 1)
    X = onehot_encoder.transform(test_integer_encoded)
    X = np.array([[w] for w in X])

    pred = classifier.predict(X)
    stemmer = SnowballStemmer("english")
    simi_out = [inWords[i] for i in range(len(inWords)) if pred[i] > accuracy]
    stemmed_out = []
    out = []
    for word in simi_out:
        if stemmer.stem(word) not in stemmed_out:
            out.append(word)
            stemmed_out.append(stemmer.stem(word))

    return np.unique(out)



print('loading the classifier ...')
# load json and create model
# TODO: change the path 
json_file = open(path_to_classifier+'/classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_to_classifier + "/model.h5")
print("Loaded model from disk")

print('compiling the classifier')
loaded_model.compile(loss='binary_crossentropy',
                     optimizer= 'RMSprop',
                     metrics=['accuracy'])

# loading LabelEncoder, OneHot Encoder, stopwords and the bag of words used 
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(path_to_classifier + '/label_encoder.npy')

with open(path_to_classifier + "/onehot_encoder.txt", "rb") as fp:   # Unpickling
    onehot_encoder = pickle.load(fp)
with open(path_to_classifier + "/common_words.txt", "rb") as fp:   # Unpickling
    common_words = pickle.load(fp)
with open(path_to_classifier + "/bag_of_words.txt", "rb") as fp:   # Unpickling
    bag_of_words = pickle.load(fp)



# method getTags take as an input the document tokenized and the accuracy needed and return the predictions  
def getTags(tokens, accuracy):
    tags = predict(tokens, bag_of_words, common_words, loaded_model, label_encoder, onehot_encoder, accuracy)
    return tags


# In[13]:


def clean(inputDoc):
    global WORDS
    inputDocList = regexp_tokenize(expand_contractions(inputDoc), pattern='\w+|\$[\d\.]+|\S+')

    no_weird_words = remove_weird_from_tokens(remove_blanc(convert_letters(inputDocList)))

    cleanDoc = correct_from_tokens(no_weird_words)
    return cleanDoc


# In[17]:


savedKeywordsPath = 'CleanDocs.txt';

def saveCleanDocs():
    finishedDocuments = 0
    if(os.path.exists(savedKeywordsPath)):   
        f = open(savedKeywordsPath, 'r')
        listedFile = list(f)
        finishedDocuments = int(listedFile[len(listedFile)-1].split('#')[1])
    else:
        f = open(savedKeywordsPath, 'w')
        f.write("#0#\n")
        f.close()  
    
    f = open('CleanDocs.txt', 'a+')
    try:
        f.write("%s\n" % ' '.join(clean(docs['content'][finishedDocuments])))
        f.write("#%s#\n" % str(finishedDocuments+1))
        print(finishedDocuments)
        f.close()
    except Exception:
        print('error in %d', finishedDocuments)
        f.write("#%s#\n" % str(finishedDocuments+1))
        
    


# In[20]:


for i in range(1,4100):
    saveCleanDocs()

