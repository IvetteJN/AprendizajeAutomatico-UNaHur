import numpy as np
import pandas as pd

# Importamos el dataset
data = pd.read_csv("titanic_train.csv")

# Exploración de los datos
# print(data.shape)
# print(data.head())
# print(data.describe())
# print(data.info())


# def zscore(data):
#     out = (data - data.mean()) / data.std()
#     return out


"""Estandarización de datos numéricos
Para que las distintas variables numéricas puedan ser comparables entre sí es necesario
estandarizarlas. Por ejemplo, si tomamos la estrategia Z-score todas nuestras variables
van a tener una media igual a 0 y un desvío estandard igual a 1.
Para estandarizar nuestros datos usando Z-score, scikit-learn nos ofrece StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)"""

# Vamos a quedarnos con las columnas que sean float64
# col = ["SibSp", "Parch", "Age", "Fare"]

# print(data[col])

# from sklearn.preprocessing import StandardScaler

# # Aplicamos la estandarizacion a las variables de interes
# scaler = StandardScaler()
# scaler = scaler.fit(data[col])
# data[col] = scaler.transform(data[col])
# # data[col] = scaler.fit_transform(data[col]) hago los dos pasos anteriores, en uno.

# print(data.describe())

"""Codificación de datos categóricos
Para muchos modelos de aprendizaje automático es necesario que todas las variables sean
numéricas, por lo que vamos a tener que transformar las variables categóricas de tal forma
que puedan ser representadas con números.
Si las categorias pueden ser traducidas como variables "ordinales", simplemente podemos
reemplazarlas por numeros. Ej: "Bajo": 0, "Medio": 1 y "Alto": 2.
Pero si las categorias no tienen ningun orden intrínseco, suele ser una mejor opción crear
variables "dummies", es decir, nuevas columnas con valores 0 ó 1 para cada una de las
categorias. Esto se conoce como One Hot Encoder."""

# En primer lugar, vamos a eliminar la variable "Name" y "Ticket" porque no nos van a ser de utilidad
data.drop(["Name", "Ticket"], axis=1, inplace=True)

# Por ejemplo, veamos cuantas mujeres y hombres estuvieron embarcados:
print(data["Sex"].value_counts())

# Vamos a transformar en dummies las varibles Sex y Embarked
data = pd.get_dummies(data, columns=["Sex", "Embarked"], prefix=["Sex", "Embarked"])

print(data.head())

# Si vemos cuantos 1's tenemos en Sex_female y Sex_male debería coincidir con la cantidad de mujeres y hombres que teniamos antes
print(data["Sex_male"].sum())

print(data["Sex_female"].sum())

"""Imputación de datos faltantes
En el curso anterior vimos el método .fillna() para llenar los datos faltantes. 
Ahora vamos a usar los métodos que nos ofrece scikit-learn.
SimpleImputer: nos permite llevar acabo las estrategias básicas de imputación de datos. 
Entre las opciones que tenemos son: mean, median, most frequent, constant.
IterativeImputer: permite realizar la imputación de los datos faltantes teniendo en cuenta 
todos los atributos del dataset (imputación multivariada)."""

# Tenemos datos faltantes?
# print(data.isna().sum())

# Eliminamos la columna Cabin porque tiene demasiados datos faltantes (no resulta informativa)
data.drop("Cabin", axis=1, inplace=True)

# Importamos la clase SimpleImputer
from sklearn.impute import SimpleImputer

# Creamos un objeto SimpleImputer con la estrategia elegida
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# Ajustamos los datos al objeto imputer (en este caso sería calcular el promedio de la columna)
imputer = imputer.fit(data)

# Guardamos los nombres de las columnas
col = data.columns

# Imputamos los datos faltantes
data = pd.DataFrame(imputer.transform(data), columns=col)

print(data.isna().sum())

"""Detección de datos atípicos(outliers)
scikit-learn nos ofrece una gran variedad de estrategias para detectar outliers. 
Un resumen de las opciones disponibles puede encontrarse aqui
En esta oportunidad vamos a utilizar un método que se llama Local Outlier Factor (LOF). 
La idea detrás de este método es sencilla: va a considerar outlier a cualquier punto que 
esté muy alejado de todos los demás, en otras palabras, un punto que tenga una muy baja 
densidad de datos cerca."""

from sklearn.neighbors import LocalOutlierFactor

# .fit_predict() va a devolver para cada una de mis filas si el método la predijo como un outlier (-1) o no (1).

lof_outlier = LocalOutlierFactor(n_neighbors=5)
outlier_scores = lof_outlier.fit_predict(data)

# Cantidad de outliers detectados
print(sum(outlier_scores == -1))

# Elimino los outliers de mis datos
mask = outlier_scores != -1
data = data.loc[mask, :]

print(data.shape)

"""¡Listo! Luego de todos estos pasos ya estamos preparados para comenzar a darle forma a un 
modelo de aprendizaje automático. De todas formas, en el proceso de crear y entrenar nuestro 
modelo es común que surja algún problema que nos obligue a volver un par de pasos atrás y 
repensar el preprocesamiento que hicimos. ¡Asi que a estar atentos!"""
