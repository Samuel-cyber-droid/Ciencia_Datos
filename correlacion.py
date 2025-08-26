import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Cargar el dataset
df = pd.read_csv('Datos2020_con_estados.csv')

# Convertir fecha a datetime
df['fc_lectura'] = pd.to_datetime(df['fc_lectura'], format='%d/%m/%y')

# Extraer información temporal
df['hora'] = df['hora_lectura'] - 1  # Convertir a formato 0-23
df['dia_semana'] = df['fc_lectura'].dt.dayofweek  # 0=Lunes, 6=Domingo
df['mes'] = df['fc_lectura'].dt.month

# Crear una tabla pivote para análisis de correlación
pivot_data = df.pivot_table(
    values='lectura',
    index=['fc_lectura', 'hora', 'dia_semana', 'mes'],
    columns='contaminante_id',
    aggfunc='mean'
).reset_index()

# Calcular matriz de correlación
correlation_columns = ['hora', 'dia_semana', 'mes'] + list(df['contaminante_id'].unique())
correlation_matrix = pivot_data[correlation_columns].corr()

# Visualizar matriz de correlación
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap='RdBu_r',
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.8}
)
plt.title('Matriz de Correlación entre Contaminantes, Hora y Día', fontsize=16)
plt.tight_layout()
plt.show()

# Análisis de correlaciones específicas
print("="*60)
print("ANÁLISIS DE CORRELACIONES TEMPORALES")
print("="*60)

# Correlaciones entre hora y contaminantes
hora_correlations = correlation_matrix.loc['hora', df['contaminante_id'].unique()].sort_values(ascending=False)
print("\nCorrelación entre Hora del Día y Contaminantes:")
for contaminante, corr_value in hora_correlations.items():
    print(f"{contaminante}: {corr_value:.3f}")

# Correlaciones entre día de la semana y contaminantes
dia_correlations = correlation_matrix.loc['dia_semana', df['contaminante_id'].unique()].sort_values(ascending=False)
print("\nCorrelación entre Día de la Semana y Contaminantes:")
for contaminante, corr_value in dia_correlations.items():
    print(f"{contaminante}: {corr_value:.3f}")

# Correlaciones entre mes y contaminantes
mes_correlations = correlation_matrix.loc['mes', df['contaminante_id'].unique()].sort_values(ascending=False)
print("\nCorrelación entre Mes del Año y Contaminantes:")
for contaminante, corr_value in mes_correlations.items():
    print(f"{contaminante}: {corr_value:.3f}")

# Visualización de patrones temporales
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Patrón por hora del día
hora_promedio = df.groupby(['hora', 'contaminante_id'])['lectura'].mean().reset_index()
for contaminante in df['contaminante_id'].unique():
    subset = hora_promedio[hora_promedio['contaminante_id'] == contaminante]
    axes[0, 0].plot(subset['hora'], subset['lectura'], label=contaminante, marker='o')
axes[0, 0].set_title('Patrón de Contaminación por Hora del Día')
axes[0, 0].set_xlabel('Hora del Día (0-23)')
axes[0, 0].set_ylabel('Nivel de Contaminación Promedio')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Patrón por día de la semana
dia_promedio = df.groupby(['dia_semana', 'contaminante_id'])['lectura'].mean().reset_index()
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
for contaminante in df['contaminante_id'].unique():
    subset = dia_promedio[dia_promedio['contaminante_id'] == contaminante]
    axes[0, 1].plot(subset['dia_semana'], subset['lectura'], label=contaminante, marker='o')
axes[0, 1].set_title('Patrón de Contaminación por Día de la Semana')
axes[0, 1].set_xlabel('Día de la Semana')
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(dias, rotation=45)
axes[0, 1].set_ylabel('Nivel de Contaminación Promedio')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Patrón por mes del año
mes_promedio = df.groupby(['mes', 'contaminante_id'])['lectura'].mean().reset_index()
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
for contaminante in df['contaminante_id'].unique():
    subset = mes_promedio[mes_promedio['contaminante_id'] == contaminante]
    axes[1, 0].plot(subset['mes'], subset['lectura'], label=contaminante, marker='o')
axes[1, 0].set_title('Patrón de Contaminación por Mes del Año')
axes[1, 0].set_xlabel('Mes del Año')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels(meses)
axes[1, 0].set_ylabel('Nivel de Contaminación Promedio')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Boxplot por hora del día para el contaminante principal
contaminante_principal = df['contaminante_id'].value_counts().index[0]
df_principal = df[df['contaminante_id'] == contaminante_principal]
sns.boxplot(x='hora', y='lectura', data=df_principal, ax=axes[1, 1])
axes[1, 1].set_title(f'Distribución de {contaminante_principal} por Hora del Día')
axes[1, 1].set_xlabel('Hora del Día (0-23)')
axes[1, 1].set_ylabel('Nivel de Contaminación')

plt.tight_layout()
plt.show()

# Resumen de hallazgos
print("="*60)
print("RESUMEN DE HALLazGOS TEMPORALES")
print("="*60)

# Identificar correlaciones significativas
umbral_correlacion = 0.3
correlaciones_significativas = {}

for variable in ['hora', 'dia_semana', 'mes']:
    for contaminante in df['contaminante_id'].unique():
        if contaminante in correlation_matrix.columns and variable in correlation_matrix.index:
            corr_val = correlation_matrix.loc[variable, contaminante]
            if abs(corr_val) > umbral_correlacion:
                if variable not in correlaciones_significativas:
                    correlaciones_significativas[variable] = []
                correlaciones_significativas[variable].append((contaminante, corr_val))

print("\nCorrelaciones significativas (|r| > 0.3):")
for variable, correlaciones in correlaciones_significativas.items():
    print(f"\n{variable.upper()}:")
    for contaminante, corr_val in sorted(correlaciones, key=lambda x: abs(x[1]), reverse=True):
        direccion = "positiva" if corr_val > 0 else "negativa"
        print(f"  {contaminante}: {corr_val:.3f} ({direccion})")