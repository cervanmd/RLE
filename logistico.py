#usamos pandas para manipular bases de datos
import pandas as pd, numpy as np

#yo normalmente tengo mis bases de datos en drive, pero aquí se puede cambiar la línea para abrir cualquier archivo con texto delimitado de diferentes maneras
import os
from google.colab import drive
drive.mount("/content/gdrive")

#en la literatura normalmente se usa df o "dataframe" para definir al conjunto de datos con el que trabajaremos
df = pd.read_csv('/content/gdrive/My Drive/diabetes/datarawdiabetes2.txt', sep="\t")

#imprimimos los primeros elementos del conjunto de datos.
df.head()
