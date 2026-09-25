import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset y limpiar (mantén tu código previo intacto hasta la visualización)
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ==========================================
# CONFIGURACIÓN VISUAL MEJORADA
# ==========================================
sns.set_theme(style="whitegrid")
# Definimos colores personalizados: Azul para "No Churn", Rojo para "Churn"
colores_churn = ["#C40DB4", "#07C455"]

fig = plt.figure(figsize=(16, 12))

# 1. Relación entre Tipo de Contrato y Churn
ax1 = fig.add_subplot(2, 2, 1)
sns.countplot(data=df, x='Contract', hue='Churn', palette=colores_churn)
ax1.set_title('Cancelación por Tipo de Contrato', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tipo de Contrato', fontsize=12)
ax1.set_ylabel('Cantidad de Clientes', fontsize=12)

# 2. Boxplot: Estadística detallada de Antigüedad por Churn
ax2 = fig.add_subplot(2, 2, 2)
sns.boxplot(data=df, x='Churn', y='tenure', hue='Churn', palette=colores_churn, legend=False)
ax2.set_title('Estadística de Antigüedad (Cuartiles y Mediana)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Churn (0 = No, 1 = Sí)', fontsize=12)
ax2.set_ylabel('Meses de Antigüedad (tenure)', fontsize=12)

# 3. Distribución con KDE (Suavizada para ver volumen de clientes)
ax3 = fig.add_subplot(2, 2, 3)
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette=colores_churn, alpha=0.6, linewidth=2)
ax3.set_title('Curva de Densidad de Antigüedad', fontsize=14, fontweight='bold')
ax3.set_xlabel('Meses de Antigüedad (tenure)', fontsize=12)
ax3.set_ylabel('Densidad de Clientes', fontsize=12)

# 4. Tendencia/Predicción: Probabilidad de Churn vs Antigüedad
ax4 = fig.add_subplot(2, 2, 4)
# Calculamos la probabilidad empírica de churn por cada mes
churn_prob = df.groupby('tenure')['Churn'].mean().reset_index()
sns.regplot(data=churn_prob, x='tenure', y='Churn', 
            scatter_kws={'alpha':0.6, 'color':'#55A868', 's': 20}, 
            line_kws={'color':'#C44E52', 'linewidth': 3}, 
            order=3) # Regresión polinómica para capturar la curva de predicción
ax4.set_title('Línea de Tendencia: Probabilidad de Cancelación', fontsize=14, fontweight='bold')
ax4.set_xlabel('Meses de Antigüedad (tenure)', fontsize=12)
ax4.set_ylabel('Probabilidad Estimada de Churn', fontsize=12)

plt.tight_layout(pad=3.0)
plt.show()

# ==========================================
# MACHINE LEARNING (Tu código original)
# ==========================================
X = df.drop(columns=['Churn'])  
y = df['Churn']                 
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)