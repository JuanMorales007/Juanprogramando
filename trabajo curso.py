###categoria_seguridad_alimentaria
#####region == "Costa"
# Importamos numpy para realizar operaciones numéricas eficientes.
import numpy as np

# Pandas nos permitirá trabajar con conjuntos de datos estructurados.
import pandas as pd

# Desde sklearn.model_selection importaremos funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold

# Utilizaremos sklearn.preprocessing para preprocesar nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler

# sklearn.metrics nos proporcionará métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score

# statsmodels.api nos permitirá realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm

# Por último, matplotlib.pyplot nos ayudará a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt
import os
datos = pd.read_csv("C:/Users/Usuario/Documents/JIME/Juan/Python/sample_endi_model_10p (1).txt", sep=";")
datos = datos[~datos["dcronica"].isna()]
datos = datos[~datos["categoria_seguridad_alimentaria"].isna()]
datos = datos[~datos["region"].isna()]
# Filtrar los datos para incluir solo la región "Costa"
datos["region"] = datos["region"].apply(lambda x: "Costa" if x == 1 else "Sierra" if x == 2 else "Oriente")
datos_costa = datos[datos['region'] == "Costa"]
datos_costa.groupby("categoria_seguridad_alimentaria").size()

for i in datos_costa:
    datos_costa = datos_costa[~datos_costa[i].isna()]
######

# Seleccionar las variables de interés
variables_interes = ['region', "sexo", 'categoria_seguridad_alimentaria',"n_hijos"]

# Crear un nuevo DataFrame con las variables de interés
nueva_base_datos = datos_costa[variables_interes]

variables_categoricas = [ "sexo", 'categoria_seguridad_alimentaria']
variables_numericas = ['n_hijos']
# Asegurarse de que las variables definidas existan en el DataFrame
variables_categoricas = [var for var in variables_categoricas if var in datos_costa.columns]
variables_numericas = [var for var in variables_numericas if var in datos_costa.columns]

# Verificar las variables categóricas y numéricas
print("Variables categóricas:", variables_categoricas)
print("Variables numéricas:", variables_numericas)
######
transformador = StandardScaler()
datos_escalados = datos_costa.copy()

###########
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   "categoria_seguridad_alimentaria_Inseguridad media","categoria_seguridad_alimentaria_Inseguridad leve","categoria_seguridad_alimentaria_Seguridad"]]
y = datos_dummies["dcronica"]

#####
weights = datos_dummies['fexp_nino']

###Separación de las muestras en entramiento (train) y prubea (test)

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

####Ajuste del modelo

modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#########3
# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)
###########################
# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_class == y_test

###Validación cruzada
# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

##### Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)
print(precision_promedio)

#####
plt.hist(accuracy_scores, bins=30, edgecolor='black')
# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()


####Vamos a crear un histograma para visualizar la distribución de los coeficientes estimados para la variable “n_hijos”:
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

######
###¿Cuál es el valor del parámetro asociado a la variable 
##clave si ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?
# Obtener el valor del parámetro asociado a la variable 'categoria_seguridad_alimentaria_Inseguridad media'
coef_inseguridad_media = result.params['categoria_seguridad_alimentaria_Inseguridad media']
print(coef_inseguridad_media)
# Obtener el p-valor asociado a este coeficiente
p_valor_inseguridad_media = result.pvalues['categoria_seguridad_alimentaria_Inseguridad media']

print("Valor del parámetro asociado a 'categoria_seguridad_alimentaria_Inseguridad media':", coef_inseguridad_media)
print("P-valor asociado:", p_valor_inseguridad_media)

# Verificar significancia
if p_valor_inseguridad_media < 0.05:
    print("El parámetro asociado a 'categoria_seguridad_alimentaria_Inseguridad media' es significativo.")
else:
    print("El parámetro asociado a 'categoria_seguridad_alimentaria_Inseguridad media' no es significativo.")

####El p-valor asociado a la variable categoria_seguridad_alimentaria_Inseguridad media 
###es aproximadamente 7.77e-05, lo que indica que es extremadamente pequeño. Esto sugiere que la variable es altamente significativa en el modelo de regresión logística.

##### Dado que este coeficiente es negativo, sugiere que a medida 
#que la categoría de seguridad alimentaria se mueve desde "Inseguridad media" hacia categorías de mayor seguridad alimentaria, la log-odds de la variable de respuesta (en este caso, la probabilidad de cronicidad) disminuye.



######################################
##base no filtrada
#Precisión promedio de validación cruzada: 0.731372549019608

###base filtrada: 
#Precisión promedio de validación cruzada: 0.7535714285714286

###la presicion promedio del modelo aumenta 
