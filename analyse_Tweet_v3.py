from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#Utiliser le même source de randomness
np.random.seed(7)
tokenizer = Tokenizer()

#ouvrir le dataset
dataset = pd.read_csv("Tweets-airline-sentiment.csv", delimiter=",")
X = dataset.text
Y = dataset.airline_sentiment
Y2 = to_categorical(Y)

tokenizer.fit_on_texts(X)
encoded_docsX = tokenizer.texts_to_matrix(X, mode='count')

#Diviser le jeu de donnée 67% pour l'entrainement et 33% pour tester
X_train, X_test, Y_train, Y_test = train_test_split(encoded_docsX, Y, test_size=0.33, random_state=7)

nbre_ligne = len(tokenizer.word_index) + 1

# create model
model = Sequential()
model.add(Dense(250, input_dim=nbre_ligne, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=2, batch_size=10)

dataset_Y_prediction = model.predict_classes(X_test)
#print(dataset_Y_prediction)
#print(type(dataset_Y_prediction))
n=0
for i in dataset_Y_prediction:
   print(i, n)
   n=n+1
