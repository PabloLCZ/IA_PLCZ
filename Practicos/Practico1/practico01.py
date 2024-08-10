#Importamos las librerias como tambien damos una semilla para que se generen los numeros randon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(42)  


#a estatura le generamos 100 estatruras con un rango de valores como ser 1.5, maximo de 2.1 y 100 valores 
estaturas = np.random.uniform(1.5, 2.1, 100)

#creamos una lista para los pesos
pesos = []  

#Por cada estatura de la lista estatura vamos sacando los valores
for estatura in estaturas:
    # sacamos un peso minimo y maximo con el IMC de una persona normal
    peso_min = 18.5 * (estatura ** 2)  
    peso_max = 24.9 * (estatura ** 2)  
    #con esos valores sacamos el peso con un numero randon que tendra como minimo el peso minimo y maximo el peso maximo
    peso = np.random.uniform(peso_min, peso_max)
    #ese peso se le pone a la lista peso
    pesos.append(peso) 


    # Calcular peso mínimo y máximo de mejor manera asi para que el modelo sea mas preciso 
#for estatura in estaturas:
 #   peso_min = 18.5 * (estatura ** 2)
  #  peso_max = 24.9 * (estatura ** 2)
   # 
    # Usar una distribución normal para simular el peso, con una desviación estándar del 10% del rango
    #desviacion = (peso_max - peso_min) * 0.1
    #peso = np.random.normal(loc=(peso_min + peso_max) / 2, scale=desviacion)
   # 
    # Asegurarse de que el peso esté dentro del rango
   # peso = np.clip(peso, peso_min, peso_max)
    
    # Agregar el peso a la lista
   # pesos.append(peso)

# Guardamos en un dataframe las Estaturas y los pesos calculados y lo imprimimos
data = pd.DataFrame({
    'Estatura (m)': estaturas,
    'Peso (kg)': pesos
})

print(data)


# Con este codigo indicamos que con los datos del dataframe de la columna Estatura (m) y Peso (Kg) 
# sacamos donde estarian en una grafica
plt.scatter(data['Estatura (m)'], data['Peso (kg)'], color='purple')

# Título del gráfico
plt.title('Estatura vs Peso (Original)')  

#le damos los valores de x a Estatura y y para Peso y mostramos la grafica
plt.xlabel('Estatura (m)')  
plt.ylabel('Peso (kg)')  
plt.show()  


#realizamos el ajuste para una curva polinomica de grado 2
# transformamos los datos en el caso de Estatura de combierte en una matriz con una columna para su mejor procesamiento
X = data['Estatura (m)'].values.reshape(-1, 1)  # Estatura (característica X)

#y el caso de peso no se toca ya que solo se convierte en una array porque es la etiqueta o el valor que se quiere predecir
y = data['Peso (kg)'].values  # Peso (variable objetivo y)

#Imprimimos solo 10 para demostracion
print(X[:10])



# Se crea un objeto de de transformacion polinomica y se pone de que grado en este caso de grado 2
poly = PolynomialFeatures(degree=2)
# Genera nuevas características polinómicas basadas en X donde valores cuadraticos como el sesgo o intercepto
X_poly = poly.fit_transform(X)  

# Crea el modelo de regresion lineal en este caso una parabola, luego se lo entrena con los valores de x_poly y los valores del vector y
model = LinearRegression()
#ajusta el modelo a los datos buscando los sesgo o intercepto con el pendiente que es la variable objetivo y
model.fit(X_poly, y)

# intenta predecir las variables de peso para la x_poly que tenia altura
y_pred = model.predict(X_poly)

# despues calcula el error cuadratico medio que es la precision del modelo donde y son los valores dependientes
#que estan fijas y y_pred es lo que el modelo saco, entonces mse que es el error cuadratico medio
#que representa el promedio de los cuadrados de las diferencias entre y y y_pred
mse = mean_squared_error(y, y_pred)
print(f"Error cuadrático medio (MSE): {mse:.2f}")

# Imprimimos los puntos originale, 
# luego creamos la curva o parabola con los valores x y los valores calculados a todo se le pone label para su interpretacion del legend
plt.scatter(data['Estatura (m)'], data['Peso (kg)'], color='orange', label='Datos')  # Puntos originales
plt.plot(data['Estatura (m)'], y_pred, color='red', label='Curva ajustada (grado 2)')  # Curva ajustada
plt.title('Curva Ajustada para Estatura vs Peso')  # Título del gráfico
plt.xlabel('Estatura (m)')  # Etiqueta del eje X
plt.ylabel('Peso (kg)')  # Etiqueta del eje Y
plt.legend()  # Mostrar la leyenda para identificar la curva ajustada
plt.show()  # Mostrar el gráfico