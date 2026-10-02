import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#Pruebas para sabe la info
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.cvs')
print(df.info())