""" cambié el nombre del archivo de pandas.py a using_library porque me arrojaba Attributeerror: module 'pandas' has no attribute 'read_csv'
instalación librería panda. 
instalación matplotlib"""

# Ejercicio 2
# Para este ejercicio vamos a trabajar con datos provenientes del Gobierno de la Ciudad de Buenos Aires sobre la administración de las vacunas de COVID-19

# a. Importe el datos desde la página del Gobierno de CABA (este punto se los dejamos resuelto)

# Ejercicio 2
# Para este ejercicio vamos a trabajar con datos provenientes del Gobierno de la Ciudad de Buenos Aires sobre la administración de las vacunas de COVID-19

# a. Importe el datos desde la página del Gobierno de CABA (este punto se los dejamos resuelto)

import pandas as pd

data = pd.read_csv(
    "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-salud/plan-de-vacunacion-covid-19/dataset_total_vacunas.csv"
)
# b. ¿Cuántas filas tiene la tabla? ¿Y cuántas columnas?

# print(data.shape)

# c. ¿Qué tipos de datos tienen las columnas del dataset?

# Imprime la información que posee el dataset: columnas, sus nombres, datos no nulos y tipo de datos
# print(data.info())
# Imprime el tipo de datos que posee cada campo del dataset
# print(data.dtypes)

# d. Realice una tabla de frecuencias absolutas y otra de frecuencias relativas de las variables GRUPO_ETARIO y TIPO_EFECTOR (ayudita: .value_count())

# Imprime los primeros registros del dataset
# print(data.head())
# Imprime los últimos registros del dataset
# print(data.tail())
# Imprime una muestra de los registros del dataset
# print(data.sample(5))
# Imprime cantidad de registros, media, mediana, registro menor y mayor, desviación estandar.
# print(data.describe())

# Trae todos los datos de la columna grupo etario
# print(data["GRUPO_ETARIO"])
# Trae los datos de la columna por grupo etario, de una manera más limpia. frecuencia absoluta
# print(data["GRUPO_ETARIO"].value_counts())
# Traemos los datos de tipo efector. frecuencia absoluta
# print(data["TIPO_EFECTOR"].value_counts())
# El total de datos del dataset
# print(len(data))
# calculamos la frecuencia relativa de cada uno de los campos requeridos (grupo etario y tipo efector)
# print(data['GRUPO_ETARIO'].value_counts() / data['GRUPO_ETARIO'].count())
# print(data['GRUPO_ETARIO'].value_counts() / len(data))
# print(data['TIPO_EFECTOR'].value_counts() / len(data))
# print(data['TIPO_EFECTOR'].value_counts() / data['TIPO_EFECTOR'].count())

# e. Realice un análisis de descriptivo de las variables numéricas, utilizando los principales estadísticos
# de tendencia central (media, mediana, etc.) y de dispersión (desvío estándar). ¿Qué diría de la distribución
# de estas variables? ¿Es simétrica?

# print(data.describe())
# La distribución es asimétrica, los datos de una de las variables es superior sobre las demás. La mayoría de los datos
# tienen valor cercano a cero. La media está cerca de 60, parándonos ahí, las dos mitades se observan asimétricas.
# La mediana es 1. La mitad de mis datos, tienen como valor 1 o menos. (50%)

# f. Realice un histograma y un boxplot de la variable DOSIS_1. ¿Qué conclusiones puede establecer?

# creamos un histograma:
# print(data["DOSIS_1"].hist())
# El histograma es horrible porque la distribución de datos es asimétrica, por lo que las barras se ven desproporcionadas y no se puede hacer un
# análisis descripptivo de los datos.
# Modificamos la distribución agregando mas bins(barras):
# print(data['DOSIS_1'].hist(bins=100))
# La distribución de las barras sigue siendo desproporcionada, por lo que transformamos los datos del eje logarítmicamente:
# print(data['DOSIS_1'].hist(bins = 100, log = True))

# creamos un boxplot para ver la distribución de los datos:
# import matplotlib.pyplot as plt

# boxplot = data.boxplot(column=["DOSIS_1"])
# boxplot.plot()
# plt.show()

# Se adjunta el resultado como Figure_1.png

# g. ¿Cuál es la media para la DOSIS_1 para las mujeres de 91 años o más?
# data_query = data[(data['GENERO'] == 'F') & (data['GRUPO_ETARIO'] == '91 o mas')]

# si quisiéramos conocer la media para la DOSIS_1 para las mujeres mayores de 71 años:
# data_query2 = data[(data['GENERO'] == 'F') & (data['GRUPO_ETARIO'] == '91 o mas') | (data['GRUPO_ETARIO'] == '81 a 90') | (data['GRUPO_ETARIO'] == '71 a 80')]

# como podemos ver, creamos un nuevo dataset con los datos que son de nuestro interés (data_query y data_query2)
# print(data_query2)
# print(data_query)

# Obtenemos la media:
# print(data_query['DOSIS_1'].mean())

# h. La variable **TIPO_EFECTOR** tiene tres valores posibles: "Privado", "Público" y "Público nacional". Pero a nosotros solo nos interesa
# distinguir entre ámbitos públicos y privados. Reemplace en la variable **TIPO_EFECTOR** los valores de "Público nacional" por "Público".
# (ayudita: `.replace()`)

# print(data['TIPO_EFECTOR'].replace('Público nacional', 'Público', inplace = True))
# print(data['TIPO_EFECTOR'].value_counts())

# i. ¿Existen datos faltantes en nuestro dataset? En caso afirmativo, proponga una manera de tratarlos e impleméntela.

# isna nos trae los datos faltantes, sum los suma y nos da el total de datos faltantes en el dataset
# print(data.isna().sum())

# podemos rellenarlos con un valor absoluto:
# print(data['DOSIS_3'].fillna(0, inplace = True))

# corroboramos:
# print(data.isna().sum())
# podemos ver que los datos faltantes ahora son reemplazados con el valor 0, dándonos, además, 0 valores faltantes en el campo DOSIS_3

# Podemos borrar los campos con datos faltantes:
# print(data.drop(['ID_CARGA'], axis = 1, inplace = True))
# si no colocamos true, el valor por defecto será false y nos va a devolver una consulta con los datos eliminados, pero no se eliminarán del dataset.
# corroboramos si se eliminó el campo:
# print(data.isna().sum())

# j. Haga una nueva tabla tomando solamente los datos de las personas de "30 o menos". Analice si existe una correlación entre la DOSIS_1 y la
# DOSIS_2 para esos casos. (si, se puede inferir que la cantidad de personas que recibió la primera dosis es mayor a la de la segunda.
# Realice un gráfico de puntos para ver gráficamente la relación entre estas dos variables. Figura_2

data_query3 = data[(data["GRUPO_ETARIO"] == "30 o menos")]
print(data_query3)

import matplotlib.pyplot as plt

boxplot = data.boxplot(column=["DOSIS_1", "DOSIS_2"])
boxplot.plot()
plt.show()
