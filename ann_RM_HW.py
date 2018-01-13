# Artifical Neural Networks

# Install Theano
# Install Tensorflow
# Install Keras

#=================================================================================================
# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
labelencoder_X_gender = LabelEncoder()

X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting dataset to training dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#=================================================================================================
# ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) old implementation
# Dense vytvara hidden unit a parameter units udava kolko bude mat hidden layer neuronov (tip ako to spravit je urobit (pocet vstupnych+pocet vystupov)/2 ako priemer)
# kernel_initializer zabezpecuje ze sa inicializuju spravne W (weights) aby boli male ale nie nulove
# inpit_dim toto definuje kolko je neuronov v vstupnej vrstve a staci to  zadefinovat raz
# activation zadava aka bude aktivacna funkcia v tomto pripade je to Rectifier funkcia
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', input_dim = 11, activation = 'relu'))
# tu pridavame Dropuot regularizacia aby sme predchadzali overfittingu
# spravit to vlastne to ze tento prvej hidden layer povie ze 10% z neuronov ma vynechat
# odporuca sa ist maximalne po 0.5
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# tu pridavame Dropuot regularizacia aby sme predchadzali overfittingu
# spravit to vlastne to ze tento prvej hidden layer povie ze 10% z neuronov ma vynechat
# odporuca sa ist maximalne po 0.5
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# compile funkcia vlastne zadefinuje ako bude fungovat trenovanie celej siete (backword propagation)
# parameter optimazer udava aku metodu zvolis na ziskanie optimalnych Ws, v tomto pripade pouzije Stochastic Gradient Descent a v distribucii ktora sa vola ADAM
# loss je vlastne zadefinovanie Cost funkcie, cize loss funkcia je vlastne cost funcia a binary_crossentropy je vlastne cost funkcia pre logistic regresion
# metrics udava ako budu evaulovane jednotlive kroky
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# toto trebuje uz siet a batch_size hovori ze zober kazdych 10 prikladov a vykonaj trenovanie
# nb_epoch hovorio tom kolko krat sa cele ucenie ma zopakovat
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Zmena pravdepodobnosti na hodnoty TRUE a FALSE podla trashold
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#=================================================================================================
# Homework
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

# predikcia noveho sa robi tak ze sa pouzije uz vytvoreny classifier a do neho sa vlozi nove pole
# toto pole ale musi byt samozrejme s dummy premennymi ak ich mame a tiez musi byt aplikovane feature scaling ak som ho pouzil aj pri vytvarani modelu
new_prediction = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

#=================================================================================================
# ANN evaulation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# tutok si zadefinujeme zase nasu ANN, ale tentokrat vo funkcii lebo tuto funkciu budeme potrebovat ako vstupny parameter
# pre vytvaranie KerasClassifier objektu
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', input_dim = 11, activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# vytvorenie objektu Keras Classifier pre 10 batchov a 100 epoch
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoche = 100)

# toto spusti cele ucenie ANN parameter cv urcuje na kolko celkov sa rozdelia treningove data,
# vlastne ta 10tka znamena ze sa ucenie bude opakovat 10 krat na 10 roznych casti treningovych dat
# odporuca sa pouzivat hodnotu 10
# n_job = -1 znamena ze na vypocet sa pouzije cela kapacita CPU
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_job = -1)

# vypocitanie priemernej hodnoty z natrenovancych presnosti
mean = accuracies.mean()

# vypocitanie variance
variance = accuracies.std()











#=================================================================================================