import pandas as pd
#PASO 1.CARGAR Y LIMPIAR LOS DATOS
df = pd.read_csv('titanic (1).csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
df['Embarked'].fillna('S', inplace = True)

age_1 = df[df['Pclass'] == 1] ['Age'].median()
age_2 = df[df['Pclass'] == 2] ['Age'].median()
age_3 = df[df['Pclass'] == 3] ['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        if row['Pclass'] == 2:
            return age_2
        return age_3
    return row["Age"]

df['Age'] = df.apply(fill_age, axis = 1)

def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0
df['Sex'] = df['Sex'].apply(fill_sex)
# Paso 2. creando un modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('Survived', axis = 1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Classifier = KNeighborsClassifier(n_neighbors = 5)
Classifier.fit(X_train, y_train)

y_pred = Classifier.predict(X_test)
print('porcentaje de resultados predichos correctamente:', accuracy_score(y_test, y_pred) * 100)
print('Confucion matrix:')
print(confusion_matrix(y_test, y_pred))