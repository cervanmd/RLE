#usamos pandas para manipular bases de datos
import pandas as pd, numpy as np

#yo normalmente tengo mis bases de datos en drive, pero aquí se puede cambiar la línea para abrir cualquier archivo con texto delimitado de diferentes maneras
#import os
#from google.colab import drive
#drive.mount("/content/gdrive")



#en la literatura normalmente se usa df o "dataframe" para definir al conjunto de datos con el que trabajaremos
df = pd.read_csv('/content/gdrive/My Drive/diabetes/datarawdiabetes2.txt', sep="\t")

#imprimimos los primeros elementos del conjunto de datos.
df.head()

#normalizar los datos antes de comenzar es una buena práctica
from sklearn import preprocessing
x = df.values #retorna un arreglo numpy
StandardScaler = preprocessing.MinMaxScaler() #definimos cuál normalización utilizaremos, en este caso un mínimo-máximo
x_scaled = StandardScaler.fit_transform(x) #se realiza la transformación y guardaamos en una nueva variable
df11 = pd.DataFrame(x_scaled, columns = df.columns) #solo para actualizar nuestro "dataframe"

df=df11
df.head() #comprobamos que se haya actualizado


#la literatura señala que la base de datos se tiene que dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split


X = df.drop(["ComplicacionBi", "id","TCHOLU","STATUS"],axis=1) #la función ".drop" elimina parámetros de la base de datos. 
y = df['STATUS'] #establecemos la variable "STATUS" como variable de interés
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) #se realiza la partición prueba y entrenamiento, "test_size" hace referencia a la proporción de división, en este caso, mitad y mitad.
#random_state hace referencia a la distribución aleatoria de estas particiones. Este asegur una aleatoriedad en la partición cada vez que se ejecute el código


