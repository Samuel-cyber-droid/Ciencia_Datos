import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar solo los primeros 100 registros y las columnas específicas
columnas = ['red_id', 'contaminante_id', 'lectura', 'fc_lectura', 'hora_lectura']
df = pd.read_csv('Datos2020.csv', usecols=columnas)

# Crear columna 'estados' basada en 'red_id' según el diccionario de datos
def mapear_estado(red_id):
    # Mapeo de red_id a estados basado en el diccionario de datos
    mapeo_estados = {
        'MIN': 'Ciudad de México',
        'XAL': 'Veracruz',
        'POZ': 'Veracruz',
        'VER': 'Veracruz',
        'CHIEST': 'Chihuahua',
        'CJU2': 'Chihuahua',
        'HGO': 'Hidalgo',
        'CHI': 'Chihuahua',
        'CMX': 'Ciudad de México',
        'AGS': 'Aguascalientes',
        'TIJ': 'Baja California',
        'COAH1': 'Coahuila',
        'DGO': 'Durango',
        'GDP': 'Durango',
        'LER': 'Durango',
        'ABA': 'Guanajuato',
        'CLY': 'Guanajuato',
        'GTO': 'Guanajuato',
        'IRP': 'Guanajuato',
        'LEON': 'Guanajuato',
        'PUR': 'Guanajuato',
        'SAL': 'Guanajuato',
        'SPZ': 'Guanajuato',
        'SMA': 'Guanajuato',
        'SIL': 'Guanajuato',
        'GDL': 'Jalisco',
        'TLC': 'Estado de México',
        'MLM': 'Michoacán',
        'CUA': 'Morelos',
        'CUE': 'Morelos',
        'OCU': 'Morelos',
        'ZAC': 'Morelos',
        'MTY': 'Nuevo León',
        'PUE': 'Puebla',
        'COR': 'Querétaro',
        'MAR': 'Querétaro',
        'SJR': 'Querétaro',
        'SQ': 'Querétaro',
        'SLP': 'San Luis Potosí',
        'CEN': 'Tabasco',
        'MID': 'Yucatán',
        'ZA': 'Zacatecas',
        'MXCM': 'Baja California'
    }

    return mapeo_estados.get(red_id, 'Desconocido')

# Diccionario de coordenadas (latitud, longitud) por estado
coordenadas_estados = {
    'Aguascalientes': (21.885256, -102.291568),
    'Baja California': (30.840634, -115.283758),
    'Chihuahua': (28.632996, -106.069100),
    'Ciudad de México': (19.432608, -99.133209),
    'Coahuila': (27.058676, -101.706829),
    'Durango': (24.027720, -104.653175),
    'Estado de México': (19.496873, -99.723267),
    'Guanajuato': (21.019015, -101.257359),
    'Hidalgo': (20.091096, -98.762387),
    'Jalisco': (20.659538, -103.349437),
    'Michoacán': (19.566519, -101.706829),
    'Morelos': (18.681305, -99.101350),
    'Nuevo León': (25.592172, -99.996195),
    'Puebla': (19.041440, -98.206273),
    'Querétaro': (20.588793, -100.389888),
    'San Luis Potosí': (22.156470, -100.985540),
    'Tabasco': (17.840917, -92.618927),
    'Veracruz': (19.173773, -96.134224),
    'Yucatán': (20.709879, -89.094338),
    'Zacatecas': (22.770925, -102.583245),
    'Desconocido': (0, 0)  # Para estados no mapeados
}

# Función para obtener coordenadas basadas en el estado
def obtener_coordenadas(estado):
    return coordenadas_estados.get(estado, (0, 0))

# Agregar columna 'estados'
df['estados'] = df['red_id'].apply(mapear_estado)

# Agregar columnas de latitud y longitud
df['latitud'] = df['estados'].apply(lambda x: obtener_coordenadas(x)[0])
df['longitud'] = df['estados'].apply(lambda x: obtener_coordenadas(x)[1])

# Renombrar columnas según lo solicitado
df = df.rename(columns={'red_id': 'estacion_id'})

# Guardar el nuevo CSV con las columnas agregadas
df.to_csv('Datos2020_con_estados_y_coordenadas.csv', index=False)

# 1. Información básica del dataset
print("=" * 50)
print("INFORMACIÓN BÁSICA DEL DATASET")
print("=" * 50)
print(f"Dimensiones: {df.shape}")
print("\nPrimeras 10 filas:")
print(df.head(10))

print("\n" + "=" * 50)
print("INFORMACIÓN DE COLUMNAS")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("ESTADÍSTICOS DESCRIPTIVOS")
print("=" * 50)
print(df.describe())

# 2. Análisis de valores nulos
print("\n" + "=" * 50)
print("VALORES NULOS POR COLUMNA")
print("=" * 50)
print(df.isnull().sum())

# 3. Análisis de las variables categóricas
print("\n" + "=" * 50)
print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("=" * 50)

print("\nDistribución de estados:")
estados_counts = df['estados'].value_counts()
print(estados_counts)

print("\nDistribución de contaminante_id:")
contaminante_counts = df['contaminante_id'].value_counts()
print(contaminante_counts)

# Visualización de distribuciones categóricas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de barras para estados
estados_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Distribución de estados')
ax1.set_xlabel('Estado')
ax1.set_ylabel('Frecuencia')
ax1.tick_params(axis='x', rotation=45)

# Gráfico de barras para contaminante_id
contaminante_counts.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Distribución de contaminante_id')
ax2.set_xlabel('Tipo de Contaminante')
ax2.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# 4. Análisis de la variable numérica (lectura)
print("\n" + "=" * 50)
print("ANÁLISIS DE LECTURAS")
print("=" * 50)

# Histograma de lecturas
plt.figure(figsize=(10, 6))
sns.histplot(df['lectura'], bins=20, kde=True, color='purple')
plt.title('Distribución de Valores de Lectura')
plt.xlabel('Valor de Lectura')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot de lecturas
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['lectura'], color='orange')
plt.title('Diagrama de Caja de Valores de Lectura')
plt.xlabel('Valor de Lectura')
plt.show()

# 5. Análisis temporal básico
print("\n" + "=" * 50)
print("ANÁLISIS TEMPORAL BÁSICO")
print("=" * 50)

# Convertir a datetime
df['fc_lectura'] = pd.to_datetime(df['fc_lectura'], format='%d/%m/%y')

print(f"Rango de fechas: De {df['fc_lectura'].min()} a {df['fc_lectura'].max()}")

# Agrupar por fecha y ver estadísticas
lecturas_por_fecha = df.groupby('fc_lectura')['lectura'].agg(['mean', 'min', 'max', 'count'])
print("\nEstadísticas de lecturas por fecha:")
print(lecturas_por_fecha)

# Serie temporal de lecturas
plt.figure(figsize=(14, 7))
df.set_index('fc_lectura')['lectura'].plot(color='blue')
plt.title('Evolución Temporal de las Lecturas')
plt.xlabel('Fecha')
plt.ylabel('Valor de Lectura')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Relación entre variables categóricas y lecturas
print("\n" + "=" * 50)
print("RELACIÓN ENTRE VARIABLES CATEGÓRICAS Y LECTURAS")
print("=" * 50)

# Boxplot por tipo de contaminante
plt.figure(figsize=(10, 6))
sns.boxplot(x='contaminante_id', y='lectura', data=df)
plt.title('Distribución de Lecturas por Tipo de Contaminante')
plt.xlabel('Tipo de Contaminante')
plt.ylabel('Valor de Lectura')
plt.show()

# Boxplot por estado
plt.figure(figsize=(12, 6))
sns.boxplot(x='estados', y='lectura', data=df)
plt.title('Distribución de Lecturas por Estado')
plt.xlabel('Estado')
plt.ylabel('Valor de Lectura')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Análisis de valores atípicos
print("\n" + "=" * 50)
print("IDENTIFICACIÓN DE VALORES ATÍPICOS")
print("=" * 50)

# Calcular rango intercuartílico (IQR)
Q1 = df['lectura'].quantile(0.25)
Q3 = df['lectura'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

atipicos = df[(df['lectura'] < limite_inferior) | (df['lectura'] > limite_superior)]
print(f"Número de valores atípicos: {len(atipicos)}")
print("\nValores atípicos:")
print(atipicos[['estados', 'contaminante_id', 'lectura', 'fc_lectura']])

# 8. Resumen ejecutivo
print("\n" + "=" * 50)
print("RESUMEN EJECUTIVO")
print("=" * 50)
print(f"- Total de registros: {len(df)}")
print(f"- Estados presentes: {', '.join(df['estados'].unique())}")
print(f"- Contaminantes medidos: {', '.join(df['contaminante_id'].unique())}")
print(
    f"- Rango de fechas: {df['fc_lectura'].min().strftime('%Y-%m-%d')} a {df['fc_lectura'].max().strftime('%Y-%m-%d')}")
print(f"- Rango de valores de lectura: {df['lectura'].min():.2f} to {df['lectura'].max():.2f}")
print(f"- Valores atípicos detectados: {len(atipicos)}")

# 9. Análisis adicional por estado
print("\n" + "=" * 50)
print("ESTADÍSTICAS POR ESTADO")
print("=" * 50)
estadisticas_por_estado = df.groupby('estados')['lectura'].agg(['count', 'mean', 'std', 'min', 'max'])
print(estadisticas_por_estado)

# 10. Análisis de correlación entre fecha y lectura
print("\n" + "=" * 50)
print("CORRELACIÓN ENTRE FECHA Y LECTURA")
print("=" * 50)
# Convertir fecha a numérico para calcular correlación
df['fc_lectura_num'] = df['fc_lectura'].astype('int64')
correlacion = df['fc_lectura_num'].corr(df['lectura'])
print(f"Correlación entre fecha y lectura: {correlacion:.4f}")

# 11. Mapa de calor por coordenadas (nuevo análisis)
print("\n" + "=" * 50)
print("ANÁLISIS GEOGRÁFICO")
print("=" * 50)

# Mostrar coordenadas únicas por estado
coordenadas_unicas = df[['estados', 'latitud', 'longitud']].drop_duplicates()
print("Coordenadas por estado:")
print(coordenadas_unicas)

# Gráfico de dispersión por coordenadas
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitud'], df['latitud'], c=df['lectura'],
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Nivel de Lectura')
plt.title('Distribución Geográfica de Lecturas por Estado')
plt.xlabel('Longitud')
plt.ylabel('Latitud')

# Agregar etiquetas de estados
for i, row in coordenadas_unicas.iterrows():
    plt.annotate(row['estados'], (row['longitud'], row['latitud']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("NUEVO ARCHIVO CSV GUARDADO")
print("=" * 50)
print("El nuevo archivo 'Datos2020_con_estados_y_coordenadas.csv' ha sido guardado")
print("con las columnas 'estados', 'latitud' y 'longitud' agregadas")
print("y la columna 'red_id' renombrada como 'estacion_id'")