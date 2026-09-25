import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Carga el dataset de Iris
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names) 
df['species'] = y
print('------------------------------')
print(df.head())
'''Dividir en Entrenamiento y prueba
separamos una parte de los datos para entrenar el modelo y otra para probar su rendimiento.'''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)
'''Normalización de los datos
Algoritmo Knn son sensibles a la escala de los datos. Por eso,normalizamos las caracteristicas:'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''Entrenar modelo
creamos y entrenamos el modelo KNN con k=3 vecinos:'''
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)
'''Hacer Predicciones y Evaluar'''
y_pred = model.predict(X_test_scaled)
print('Precision:', model.score(X_test_scaled, y_test)) 
print('------------------------------------------------------------')
print('\nInforme de Clasificacion:') 
print('------------------------------------------------------------')
print(classification_report(y_test, y_pred,target_names=iris.target_names))
print('------------------------------------------------------------')
print('\nMatriz de Confusion:')
print('------------------------------------------------------------')
print(confusion_matrix(y_test, y_pred))
print('------------------------------------------------------------')
'''listo en menos de 20 lineas de codigo, has creado un modelo de Machine Learning funcional.
El KNN alcanza del 95% de presicion en el dataset de iris,Este pipeline,carga datos,dividir,normalizar,entrenar y evaluar el modelo.''' 
 