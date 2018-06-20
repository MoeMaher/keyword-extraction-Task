# Keywords Extraction task

## getting started

start by looking at keywordsExtraction.ipynb in which, data manipulation and loading the model and using it 
then take a look at how the model is created in /model_train 

I didn’t know if you were testing, just solving the problem or implementing models from scratch, so I didn’t try use libraries from the box. all the code is created from scratch except small snippet in the notebook that i still modified

## problem 

the problem here is that this model is trained over hulth data-set and only knows its vocab so here it will not do great.

## Improvements
due to the lake of time there is alot of things i wanted to try and implement
- Make spelling correction algorithm like in google’s they train a model on a very large data-set to make a binary classification is this word is suitable in this sentence or not, I think this way we can use word embedding, or google pre-trained model to vectorize the words in 300D vectors better the one hot encoding, anyway and then get all the possible words with one or two differences from the word and see if the confidence of this word in this sentence is above a predefined threshold, then this word can replace the old one. 

- The data-set sometimes talks about things that not necessarily has a common word like for example Mm which is a lesson about how to teach children the letter M, so we have to focus on unsupervised keyword extraction with of course the Part of Speech (POS) restriction , as an initial suggestions, I found some useful graph based methods like what TextRank is doing 

- Train the model on more data-set as its trained over Hulth 2003 data-set which is mainly physics and engineering papers, this was an already planned improvement for the Js library.

- Perform the computation on the data in a fixed pipeline, for ex. Yamal.
