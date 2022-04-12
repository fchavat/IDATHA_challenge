# Desafío de Computer Vision para IDATHA

Este repositorio contiene los archivos relativos al desafío de Computer Vision de IDATHA.

La idea del desafío fue seleccionar un *dataset* de *Kaggle*.
El que seleccioné es *ASL Alphaber*  (https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

El tiempo que se le destinó al desafío fueron más o menos 15 horas.

## Sobre el conjunto elegido

El conjunto que seleccioné tiene poca diversidad en sus elementos. Para cada una de las letras del alfabeto se tiene una serie de elementos que fueron creados por la misma persona, sin variar demasiado el contexto de la imagen, generando al final un conjunto de datos que si bien es suficiente para desarrollar un modelo y probar si aprende, no parecería ser bueno para lograr un modelo que generalice y pueda predecir símbolos del abecedario de señas con buen rendimiento. Esto lo comprobé al probar el modelo sobre un conjunto aparte (https://www.kaggle.com/datasets/danrasband/asl-alphabet-test), donde se obtuvo un valor de exactitud (*accuracy*) muy bajo (\~0.32) en comparación con el obtenido en validación (\~0.85). Aún así, creo que fue posible pasar por las etapas de implementación y entrenamiento sin saltar nada, por lo que decidí no cambiar el dataset luego de haber notado esto.

## Reproducir el experimento

 - **Versión de Python utilizada**: 3.8.10

Instrucciones
 - Instalar requerimientos

        pip install -r requirements.txt

 - Crear carpeta *dataset* en *root*
 - Descargar dataset *ASL Alphabet* (https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
 - Descomprimir *dataset* dentro de la carpeta *dataset*
   - Debería quedar así: dataset/asl_alphabet_train/[A,B,...]
 - Descargar el *dataset* *ASL Alphabet Test* (https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)
 - Descomprimir *dataset* dentro de *dataset/test*
   - Debería quedar así: *dataset/test/[A,B,...]*
 - Levantar *desafio.ipynb* con alguna herramienta que soporte Jupyter Notebook

Ahora ya debería ser posible ejecutar las celdas sin problema.

## Ejecutar Rest API
Para ejecutar la aplicación Rest API de Flask nos posicionamos en *root* y ejecutamos

        FLASK_ENV=development FLASK_APP=app.py flask run

Ahora ya debería ser posible hacer uso de la API.
Implementé un par de *templates* para permitir visualizar las llamadas.

Si levantamos la aplicación (posiblemente en *127.0.0.1:5000*) podemos ir a home (*"/"*) para usar el modelo con un *file uploader* o sacando fotos con una *webcam*.

Si vamos a "*/model_info*" se muestra la información del modelo usado.
