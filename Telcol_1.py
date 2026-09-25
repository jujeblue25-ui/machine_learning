import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Verificar dimensiones (esperamos 7043 filas y 21 columnas)
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# 3. Ver estructura general y tipos de datos
df.info()

# Convertir espacios en blanco a NaN y luego a float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

# Rellenar los valores nulos con 0 (ya que son clientes nuevos con 0 meses de antigüedad)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Comprobar que ya es numérica
print("Tipo de dato de TotalCharges:", df['TotalCharges'].dtype)

# Eliminar customerID del DataFrame
df = df.drop(columns=['customerID'])

# Convertir Churn: 'Yes' -> 1, 'No' -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Verificar la proporción de la variable objetivo
print("Distribución de Churn:")
print(df['Churn'].value_counts(normalize=True))

# Configuración visual
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 5))

# 1. Relación entre Tipo de Contrato y Churn
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Cancelación por Tipo de Contrato')
plt.xlabel('Tipo de Contrato')
plt.ylabel('Cantidad de Clientes')

# 2. Distribución de Antigüedad (tenure) según Churn
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, element="step")
plt.title('Distribución de Antigüedad (Meses) por Churn')
plt.xlabel('Meses de Antigüedad (tenure)')
plt.ylabel('Cantidad de Clientes')

plt.tight_layout()
plt.show()

# 1. Separar la variable dependiente / objetivo (y) de las variables independientes / predictoras (X)
X = df.drop(columns=['Churn'])  # Todo el DataFrame excepto la columna a predecir
y = df['Churn']                 # La columna objetivo (ya convertida a 0 y 1)

# 2. Aplicar One-Hot Encoding a todas las columnas categóricas de texto en 'X'
# pd.get_dummies convierte textos en columnas binarias (0 y 1) de manera automática
X_encoded = pd.get_dummies(X, drop_first=True)

print("Número original de columnas en X:", X.shape[1])
print("Número de columnas en X tras One-Hot Encoding:", X_encoded.shape[1])


# Dividir la matriz X_encoded y el vector y
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.20,       # 20% reservado para pruebas (1,409 clientes)
    random_state=42,      # Fija una semilla para que los resultados sean reproducibles
    stratify=y            # Mantiene la misma proporción de Churn (26.5%) en train y test
)

print(f"Conjunto de Entrenamiento: {X_train.shape[0]} filas")
print(f"Conjunto de Prueba: {X_test.shape[0]} filas")