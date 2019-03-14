from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

#Utiliser le même source de randomness
np.random.seed(7)
tokenizer = Tokenizer()

#ouvrir le dataset
dataset = pd.read_csv("JeuDeDonnee.csv", delimiter=",")
X = dataset.text
Y = dataset.airline_sentiment
Y2 = to_categorical(Y)

tokenizer.fit_on_texts(X)
encoded_docsX = tokenizer.texts_to_matrix(X, mode='count')

#Diviser le jeu de donnée 80% pour l'entrainement et 20% pour valider
X_train, X_test, Y_train, Y_test = train_test_split(encoded_docsX, Y2, test_size=0.20, random_state=7)

nbre_ligne = len(tokenizer.word_index) + 1

# create model
model = Sequential()
model.add(Dense(1000, input_dim=nbre_ligne, activation='relu'))
model.add(Dense(750, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Stopper l'entrainnement jusqu'à que le model ne s'améliore plus
stopEntrainnement = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# ajuster le modèle
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=1, callbacks=[stopEntrainnement])

#Partie Diagramme
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Fin partie diagramme

#Evaluer le model après l'entrainement
print("Après l'entrainement: ")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Loss: ", loss, " Accuracy: ", accuracy)

#Permet de prédire les émotions dans la partie Validation
dataset_Y_prediction = model.predict_classes(X_test)

#Enregistrer les prédictions dans un fichier (validation)
n=0
f = open("validation.txt", 'w', encoding = 'utf-8')
for i in dataset_Y_prediction:
   f.write("Pour ligne: ")
   f.write(str(n))
   f.write(" valeur: ")
   f.write(str(i))
   f.write('\n')
   n=n+1

f.close()


#Test un jeu de donnée
dataset = pd.read_csv("test.csv", delimiter=",")
XTest = dataset.text

encoded_docsXTest = tokenizer.texts_to_matrix(XTest, mode='count')
predictionTest = model.predict_classes(encoded_docsXTest)
l=0
f = open("test.txt", 'w', encoding = 'utf-8')
for i in predictionTest:
   f.write("Pour ligne: ")
   f.write(str(l))
   f.write(" valeur: ")
   f.write(str(i))
   f.write('\n')
   l=l+1

f.close()


#Partie Test en temps réel
while(1):
   print('Entrer qqche: ')
   x = input()
   x_test = [x]
   encoded_docsX2 = tokenizer.texts_to_matrix(x_test, mode='count')
   prediction = model.predict_classes(encoded_docsX2)
   print(prediction)
   print("\n")
   
   #lossTest, accuracy = model.evaluate(x_test, prediction, verbose=False)
   #print("Loss: ", lossTest, " Accuracy: ", accuracy)
