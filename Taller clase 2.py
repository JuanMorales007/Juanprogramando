## frase divertida 
variable_frase = "Mi nombre es Juan y soy un joven profesor de amor"
##lista de numeros 
edad_de_mis_compas_de_eco2 =[21,20,22,21,21,22]
##diccionarios
mis_materias_que_me_tienemal = {"eco_2" : "L-M-V","mercado" : "L-V","formulacion" : "M","gestion" : "V","investigacion" : "L-M"}
print(mis_materias_que_me_tienemal)
############ 
vector_enteros=[10]*7
print(vector_enteros)

vector_flotantes=[7.2]*5
print(vector_flotantes)

diccionario = {"entero" : vector_enteros, "flotante" : vector_flotantes}
print(diccionario)

#CADENAS
cadena_simple = 'Hola a todos!'
cadena_doble=["Estudio economía"," estoy cursando sexto semestre"]
print(cadena_doble)
print(cadena_simple)
# una tabla Excel usando pandas
import pandas as pd
imp_sri= pd.read_excel("ventas_SRI.xlsx")
print(imp_sri)
