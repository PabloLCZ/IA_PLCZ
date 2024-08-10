#Importamos las librerias como tambien damos una semilla para que se generen los numeros randon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
 


#a estatura le generamos 100 estatruras con un rango de valores como ser 1.5, maximo de 2.1 y 100 valores 
estaturas = np.random.uniform(1.5, 2.1, 100)

#creamos una lista para los pesos
pesos = []  

for estatura in estaturas:
   peso_min = 18.5 * (estatura ** 2)
   peso_max = 24.9 * (estatura ** 2)
   desviacion = (peso_max - peso_min) * 0.1
   peso = np.random.normal(loc=(peso_min + peso_max) / 2, scale=desviacion)
   peso = np.clip(peso, peso_min, peso_max)
   pesos.append(peso)
#Por cada estatura de la lista estatura vamos sacando los valores
#for estatura in estaturas:
 #   # sacamos un peso minimo y maximo con el IMC de una persona normal
  #  pesominimo = 18.5 * (estatura ** 2)  
   # pesomaximo = 24.9 * (estatura ** 2)  
    ##con esos valores sacamos el peso con un numero randon que tendra como minimo el peso minimo y maximo el peso maximo
    #peso = np.random.uniform(pesominimo, pesomaximo)
    ##ese peso se le pone a la lista peso
    #pesos.append(peso) 



# Guardamos en un dataframe las Estaturas y los pesos calculados y lo imprimimos
dataEstatura = pd.DataFrame({
    'Estatura': estaturas,
    'Peso': pesos
})

print(dataEstatura)


# Con este codigo indicamos que con los datos del dataEstaturaframe de la columna Estatura y Peso 
# sacamos donde estarian en una grafica
plt.scatter(dataEstatura['Estatura'], dataEstatura['Peso'], color='purple')

# Título del gráfico
plt.title('Estatura vs Peso (Original)')  

#le damos los valores de x a Estatura y y para Peso y mostramos la grafica
plt.xlabel('Estatura')  
plt.ylabel('Peso')  
plt.show()  


#realizamos el ajuste para una curva polinomica de grado 2
# transformamos los datos en el caso de Estatura de combierte en una matriz con una columna para su mejor procesamiento
XO = dataEstatura['Estatura'].values.reshape(-1, 1)  # Estatura (característica X)

#y el caso de peso no se toca ya que solo se convierte en una array porque es la etiqueta o el valor que se quiere predecir
y = dataEstatura['Peso'].values  # Peso (variable objetivo y)

#Imprimimos solo 10 para demostracion
print(XO[:10])



# Se crea un objeto de de transformacion polinomica y se pone de que grado en este caso de grado 2
Curvapoly = PolynomialFeatures(degree=2)
# Genera nuevas características polinómicas basadas en X donde valores cuadraticos como el sesgo o intercepto
XO_poly = Curvapoly.fit_transform(XO)  

# Crea el modelo de regresion lineal en este caso una parabola, luego se lo entrena con los valores de x_poly y los valores del vector y
CurvaModelo = LinearRegression()
#ajusta el CurvaModelo a los datos buscando los sesgo o intercepto con el pendiente que es la variable objetivo y
CurvaModelo.fit(XO_poly, y)

# intenta predecir las variables de peso para la x_poly que tenia altura
Y_Entrenado = CurvaModelo.predict(XO_poly)

# despues calcula el error cuadratico medio que es la precision del modelo donde y son los valores dependientes
#que estan fijas y Y_Entrenado es lo que el modelo saco, entonces mse que es el error cuadratico medio
#que representa el promedio de los cuadrados de las diferencias entre y y Y_Entrenado
mse = mean_squared_error(y, Y_Entrenado)
print(f"Error cuadrático medio es de  (MSE): {mse:.2f}")

# Imprimimos los puntos originale, 
# luego creamos la curva o parabola con los valores x y los valores calculados a todo se le pone label para su interpretacion del legend
plt.scatter(dataEstatura['Estatura'], dataEstatura['Peso'], color='orange', label='Datos')  # Puntos originales
plt.plot(dataEstatura['Estatura'], Y_Entrenado, color='red', label='Curva ajustada (grado 2)')  # Curva ajustada
plt.title('Curva Ajustada para Estatura vs Peso')  
plt.xlabel('Estatura') 
plt.ylabel('Peso')  
plt.legend()  
plt.show() 