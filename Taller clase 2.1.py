
## frase divertida 
variable_frase = "Mi nombre es Juan y soy un joven profesor de amor"
##lista de numeros 
edad_de_mis_compas =[21,20,22,21,21,22]
##diccionarios
animes_que_recomiendo= {"anime1" : "Monster","Anime2" : "pluto","anime3" : "SNK","anime4" : "vinlandsaga","anime5" : "87"}
print(animes_que_recomiendo)
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