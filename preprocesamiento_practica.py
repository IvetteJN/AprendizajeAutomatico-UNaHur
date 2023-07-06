import numpy as np
import pandas as pd

data = pd.read_csv("melb_data.csv")

data.shape  # Nos indica la cantidad de registros, y la cantidad de columnas

data.info()  # Nos indica el tipo de datos, el nombre de las columnas y la cantidad de datos en cada una

data.describe()

"""Todos tienen al menos un ambiente (min) max 10 ambientes. 
En rooms se nota que es una variable amigable, tenemos números bastante coherentes. 
En precio tenemos números muy variables. van desde 85mil a 9mill. (pero razonable)

Geopandas: permite trabajar con datos gps, coordenadas, creación de mapas, etc 
(tratamiento de datos demográficos, geográficos, sociales, etc)"""

# Estandarización de los datos
# Standard Scaler

from sklearn.preprocessing import StandardScaler

data.columns

"""Los datos tipo fecha hay que tratarlos de manera de transformarlos en datos tratables 
para hacer cuentas. La variable price no la tocamos porque es la que queremos tener 
en cuenta en el aprendizaje automático (objetivo)
Leer cómo funciona standardScaler"""

# hacemos una lista de las variables que queremos estandarizar
col_a_estandarizar = [
    "Rooms",
    "Distance",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
]
# Instanciamos el objeto
scaler = StandardScaler()
# fiteamos los datos de interés
scaler = scaler.fit(data[col_a_estandarizar])
# y los transformamos
data[col_a_estandarizar] = scaler.transform(data[col_a_estandarizar])

data.describe()

"""MinMaxScaler. from sklearn.preprocessing import StandardScaler hacemos una lista de las 
variables que queremos estandarizar col_a_estandarizar = ['Rooms', 'Distance', 'Bedroom2', 
'Bathroom', 'Car', 'Landsize', 'BuildingArea'] Instanciamos el objeto scaler = MinMaxScaler() 
fiteamos los datos de interés scaler = scaler.fit(data[col_a_estandarizar]) y los transformamos 
data[col_a_estandarizar] = scaler.transform(data[col_a_estandarizar])"""

# Codificar variables categóricas

data.columns

"""Si hacemos el siguiente código, nos devuelve la cantidad de filas que tengo, en las que 
se repite cada una de las categorías. Sabemos cual es mi categoría mayoritaria, minoritaria, 
si hay mucha diferencia entre categorías. Podría darse el caso de tener categorías muy chiquitas, 
muchas, pero muy minoritarias (1 o 2 datos). Lo que me conviene hacer es agruparlas en una 
especie de categoría ("otros"), donde bajo la cantidad de categorías."""

data["Suburb"].value_counts()

"""Tenemos muchas categorías. Si las pasamos a todas a dummies, vamos a estar agregando... 
muchas columnas. Cada una de las categorías se va a transformar en una nueva columna en el 
dataset. cuando son tan minoritarias(un dato por categoría). ¿Qué hago con esos datos? Va a 
depender del contexto. Va a estar relacionado con lo que yo quiero predecir. Imaginemos que 
lo que se quiere predecir está dentro de las categorías con mayor cantidad de datos... 
En ese caso, las minoritarias no me interesan, así que las puedo eliminar.(pueden hacer ruido) 
o podemos agruparlas en una sola categoría, del tipo "otras"."""

data["Suburb"].replace(
    {
        "Sandhurst": "Otros",
        "Bullengarook": "Otros",
        "Croydon South": "Otros",
        "Montrose": "Otros",
        "Monbulk": "Otros",
    }
)

# Otra opción sería:
# data['Suburb'].replace(['Sandhurst', 'Bullengarook', 'Croydon South', 'Montrose', 'Monbulk'], ['Otros])

# O también:
# otras = data['Suburb'].value_counts().loc[lambda x : x < 10].index

# otras

# Podemos también asignar los datos a una nueva variable y trabajar directamente con ella:
sub = data["Suburb"].value_counts()

type(sub)
# Nos devuelve una serie de pandas:
# pandas.core.series.Series

sub

data["Regionname"].value_counts()
# La suma de todos los registros:
len(data)

data.shape

# Frecuencia relativa:
data["Regionname"].value_counts() / len(data)

# Si queremos ver los datos en porcentaje, los dividimos por la longitud de datos y los multiplicamos por 100
(data["Regionname"].value_counts() / len(data)) * 100

# Para conocer la cantidad de valores únicos que hay en cierta columna. Cuántas categorías hay dentro de esta categoría
data["Regionname"].unique()

# Si unimos cada región con suburbio, no va a ser muy informativo. Argentino.porteño, argentino.cordobes, chileno.santiagueño, boliviano.paceño, etc
