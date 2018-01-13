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

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) old implementation
# Dense vytvara hidden unit a parameter units udava kolko bude mat hidden layer neuronov (tip ako to spravit je urobit (pocet vstupnych+pocet vystupov)/2 ako priemer)
# kernel_initializer zabezpecuje ze sa inicializuju spravne W (weights) aby boli male ale nie nulove
# inpit_dim toto definuje kolko je neuronov v vstupnej vrstve a staci to  zadefinovat raz
# activation zadava aka bude aktivacna funkcia v tomto pripade je to Rectifier funkcia
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', input_dim = 11, activation = 'relu'))

# Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

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