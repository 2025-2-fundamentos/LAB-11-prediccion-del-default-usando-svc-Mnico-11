import os
import gzip
import json
import pickle
import zipfile
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Paso 1: Limpieza de los datos
def clean_data(df):
    df = df.copy()                                                      #? Se crea una copia del DataFrame para evitar modificar el original
    df = df.drop('ID', axis = 1)                                        #? Elimina la columna "ID" que no es relevante para el modelo
    df = df.rename(columns={'default payment next month': 'default'})   #? Renombra la columna de objetivo
    df = df.dropna()                                                    #? Elimina las filas con valores nulos
    df = df[(df['EDUCATION'] != 0 ) & (df['MARRIAGE'] != 0)]            #? Elimina registros con valores no válidos en EDUCATION o MARRIAGE
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4                        #? Agrupa los valores de EDUCATION mayores a 4 en la categoría "others"

    return df

# Paso 3: Crear el pipeline del modelo
def model():
    categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']              #? Columnas categóricas
    numeric_columns = [
                "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"
                ]                                                       #? Columnas numéricas

    preprocessor = ColumnTransformer(                                                     #? Transformación de columnas categóricas y numéricas
                    transformers = [
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), #? OneHotEncoding para las columnas categóricas
                    ('scaler', StandardScaler(), numeric_columns)                         #? Estandarización de las variables numéricas
                    ],
                    remainder ='passthrough'                                              #? Las columnas restantes se mantienen sin cambios
                    )

    selector  = SelectKBest(score_func = f_classif)                           #? Selección de las K mejores característica

    pipeline = Pipeline(steps = [                                             #? Modelo de clasificador (SVM)
                        ('preprocessor', preprocessor),                       #? Primero, se aplica la preprocesamiento
                        ('pca', PCA()),                                       #? Se aplica PCA para reducir la dimensionalidad
                        ("selectkbest", selector ),                           #? Selección de las K mejores característica
                        ('classifier', SVC(kernel = 'rbf', random_state=42))  #? Se ajusta el clasificador SVM con kernel RBF
                                ]
                        )

    return pipeline                                                           #? Devuelve el pipeline completo

def optimize_hyperparameters(model, n_splits, x_train, y_train, scoring):     #? Paso 4: Optimización de hiperparámetros
    estimator = GridSearchCV(
        estimator = model,
        param_grid = {
            'pca__n_components': [20, 21],                                    #? Número de componentes principales para PCA
            'selectkbest__k': [12],                                           #? Selección de las mejores 12 características
            'classifier__kernel': ['rbf'],                                    #? Tipo de kernel para SVM
            'classifier__gamma': [0.099]                                      #? Valor de gamma para el kernel RBF
        },
        cv=n_splits,                                                          #? Validación cruzada con 10 splits
        refit=True,
        verbose=0,                                                            #? No mostrar detalles adicionales del proceso
        scoring=scoring,                                                      #? Métrica de precisión balanceada
        return_train_score=False                                              #? No devolver las métricas de entrenamiento
    )
    estimator.fit(x_train, y_train)                                           #? Ajusta el modelo con los datos de entrenamiento

    return estimator

def metrics(model, x_train, y_train, x_test, y_test):                        # Paso 6: Calcular y guardar las métricas
    y_train_pred = model.predict(x_train)                                    #? Predicciones en los datos de entrenamiento
    y_test_pred = model.predict(x_test)                                      #? Predicciones en los datos de prueba

    train_metrics = {                                                        #? Métricas para el conjunto de entrenamiento
        'type': 'metrics',
        'dataset': 'train',
        'precision': (precision_score(y_train, y_train_pred, average='binary')),
        'balanced_accuracy': (balanced_accuracy_score(y_train, y_train_pred)),
        'recall': (recall_score(y_train, y_train_pred, average='binary')),
        'f1_score': (f1_score(y_train, y_train_pred, average='binary'))
    }

    test_metrics = {                                                         #? Métricas para el conjunto de prueba
        'type': 'metrics',
        'dataset': 'test',
        'precision': (precision_score(y_test, y_test_pred, average='binary')),
        'balanced_accuracy':(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': (recall_score(y_test, y_test_pred, average='binary')),
        'f1_score': (f1_score(y_test, y_test_pred, average='binary'))
    }

    return train_metrics, test_metrics                                       #? Devuelve las métricas de ambos conjuntos

def confusion_matrices(model, x_train, y_train, x_test, y_test):             # Paso 7: Calcular las matrices de confusión
    y_train_pred = model.predict(x_train)                                    #? Predicciones en los datos de entrenamiento
    y_test_pred = model.predict(x_test)                                      #? Predicciones en los datos de prueba

    cm_train = confusion_matrix(y_train, y_train_pred)                       #? Matriz de confusión para el conjunto de entrenamiento
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()                #? Matriz de confusión para el conjunto de prueba

    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()


    train_matrix = {
        'type': 'cm_matrix',
        'dataset': 'train', 
        'true_0': {'predicted_0': int(tn_train),
                    'predicted_1': int(fp_train)},
        'true_1': {'predicted_0': int(fn_train),
                    'predicted_1': int(tp_train)}
    }

    test_matrix = {
        'type': 'cm_matrix',
        'dataset': 'test', 
        'true_0': {'predicted_0': int(tn_test),
                    'predicted_1': int(fp_test)},
        'true_1': {'predicted_0': int(fn_test),
                    'predicted_1': int(tp_test)}
    }

    return train_matrix, test_matrix                                        #? Devuelve las matrices de confusión para ambos conjuntos

def save_model(model):                                      #? Paso 5: Guardar el modelo
    os.makedirs('files/models', exist_ok = True)

    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def save_metrics(metrics):
    os.makedirs('files/output', exist_ok=True)

    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")



file_Test = 'files/input/test_data.csv.zip'     #? Limpia los datos de prueba
file_Train = 'files/input/train_data.csv.zip'   #? Limpia los datos de entrenamient


with zipfile.ZipFile(file_Test, 'r') as zip:
    with zip.open('test_default_of_credit_card_clients.csv') as f:
        df_Test = pd.read_csv(f)


with zipfile.ZipFile(file_Train, 'r') as zip:
    with zip.open('train_default_of_credit_card_clients.csv') as f:
        df_Train = pd.read_csv(f)


df_Test = clean_data(df_Test)
df_Train = clean_data(df_Train)


x_train, y_train = df_Train.drop('default', axis=1), df_Train['default']
x_test, y_test = df_Test.drop('default', axis=1), df_Test['default']


model_pipeline = model()                                                                                 #? Crea el pipeline
model_pipeline = optimize_hyperparameters(model_pipeline, 10, x_train, y_train, 'balanced_accuracy')     #? Optimiza los hiperparámetros


save_model(model_pipeline)

train_metrics, test_metrics = metrics(model_pipeline, x_train, y_train, x_test, y_test)                  #? Calcular y guardar las métricas
train_matrix, test_matrix = confusion_matrices(model_pipeline, x_train, y_train, x_test, y_test)         #? Calcular las matrices de confusión

save_metrics([train_metrics, test_metrics, train_matrix, test_matrix])                                   #? Guardar las métricas y matrices de confusión