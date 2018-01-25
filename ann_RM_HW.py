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
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

# toto spusti cele ucenie ANN parameter cv urcuje na kolko celkov sa rozdelia treningove data,
# vlastne ta 10tka znamena ze sa ucenie bude opakovat 10 krat na 10 roznych casti treningovych dat
# odporuca sa pouzivat hodnotu 10
# n_job = -1 znamena ze na vypocet sa pouzije cela kapacita CPU ale na mojom pocitaci nefunguje -1 a preto musim zadat tam 1
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

# vypocitanie priemernej hodnoty z natrenovancych presnosti
mean = accuracies.mean()

# vypocitanie variance
variance = accuracies.std()

#=================================================================================================
# ANN tunning - ked tuningujem ANN tak mozem upravovat nasledovne veci" 
# 1. mozem zmenit strukturu ANN to znamena ze mozem pridavat alebo uberat jednotlive hidden layers 
# 2. alebo menit pocet neuronov v jednotlivych skrytych vrstvach
# 3. alebo sa hrat s parametrami ako je optimizer, batch_size a nb_epoche

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# funkcia ktora bude vytvarat nasu ANN, rozdiel oproti predchadzajucemu prikladu je ze mame tu parameter pre
# optimizer aby sme mohli skusat rozne kombinacieho optimizera
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', input_dim = 11, activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# vytvorenie objektu Keras Classifier ale tento krat bez velkosti batchu a poctu epoch lebo to budeme tunit
classifier = KerasClassifier(build_fn = build_classifier)

# toto je zadefinovanie roznych moznosti pre 3 parametre ktore budeme testovat v roznych kombinaciach
parameters = { 'batch_size': [1, 5, 25, 32],
              'nb_epoch': [50, 100, 500],
              'optimizer': ['adam', 'rmsprop']}

# Grid search je metoda ktoru pouzijeme na tuning ANN.
# parameter estimator je vlastne nasa ANN ktoru sme zadefinovali v premennej classifier
# parameter param_grid je zoznam nasich parametrov s jednotlivymi hodnotami ktore chceme otestovat
# parameter scoring definuje ako bude ANN evaulovana, lebo toto tiez vie spravit
# cv je cross validation a definuje pocet zlozeni pre closs validation
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

# toto nam vrati hodnoty pre najlepsie parametre ktore vysli z testovania a tiez aka je presnot pre tieto parametre
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#=================================================================================================
# ANN tunning - Home work - get the gold medal for the accuracy over 86%

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# funkcia ktora bude vytvarat nasu ANN, rozdiel oproti predchadzajucemu prikladu je ze mame tu parameter pre
# optimizer aby sme mohli skusat rozne kombinacieho optimizera
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', input_dim = 11, activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
# pridavam novu skrytu vrstvu aby som zvysil acuracy    8345 8075 8206
    

    
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# vytvorenie objektu Keras Classifier ale tento krat bez velkosti batchu a poctu epoch lebo to budeme tunit
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1, nb_epoch = 500)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

# vypocitanie priemernej hodnoty z natrenovancych presnosti
mean = accuracies.mean()

# vypocitanie variance
variance = accuracies.std()











































































#=================================================================================================