import numpy as np

# Cargar el archivo CSV sin encabezado para trabajar con los datos directamente
data = np.genfromtxt('/mnt/data/test.csv', delimiter=',', dtype=str, skip_header=1)

# 1. Crear la columna 'Customer Feedback' basada en la columna 'satisfaction' (última columna)
feedback = np.where(data[:, -1] == 'satisfied', 'Positive experience', 'Neutral or Negative experience')

# 2. Crear la columna 'Requested Special Assistance' basada en la columna 'Age' (suponiendo que está en el índice 3)
special_assistance = np.where(data[:, 3].astype(int) > 60, 'True', 'False')

# 3. Crear la columna 'Flight Code' basada en el 'id' (suponiendo que el 'id' está en la primera columna)
flight_codes = np.array([f'FL{int(id_val):05d}' for id_val in data[:, 1]])

# Crear la columna de contador '#' que cuenta cuántas filas hay
num_filas = np.arange(1, data.shape[0] + 1).reshape(-1, 1)  # Esto crea una columna desde 1 hasta el número de filas

# Unir la columna '#' y las nuevas columnas al dataset original
data_modificado = np.column_stack((num_filas, data, feedback, special_assistance, flight_codes))

# Definir los nombres de las columnas, incluyendo la nueva columna '#'
column_names = ['#', 'id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 
                'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Satisfaction', 
                'Customer Feedback', 'Requested Special Assistance', 'Flight Code']

# Guardar el nuevo dataset en un archivo CSV llamado 'dataset_sintetico.csv'
np.savetxt('dataset_sintetico.csv', data_modificado, delimiter=',', fmt='%s', header=",".join(column_names), comments='')

print("El dataset sintético ha sido guardado como 'dataset_sintetico.csv'.")
