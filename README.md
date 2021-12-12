# Pràctica Kaggle APC UAB 2021-22
### Nom: Bhupinder Kumar Ram Ram 
### DATASET: Electrical Fault detection and classification
### URL: [kaggle](https://www.kaggle.com/esathyaprakash/electrical-fault-detection-and-classification)
## Resum
Nuestra base de datos nos da la información sobre 3 líneas con su respectiva corriente y 3 fuentes de energía sobre un sistema electrónico. A partir de esta información, tenemos dos tipos de problemas de clasificación: <br>

    1. Detectar si hay un fallo en el sistema eléctrico.
    2. Clasificar que tipo de falla hay en el sistema eléctrico.
De los datos del dataset, nuestro target es el Output (S).
Siendo un 0 o 1. Con lo cual es un problema logístico o de clasificación.
Y los demás siendo features, en este caso nos interesan todos, ja todas nos aportan información sobre el sistema eléctrico.
Siendo todos valores numéricos.

### Objectius del dataset
Detectar si hay un fallo o no en el sistema eléctrico.

### Preprocessat
Para poder visualizar mejor los resultados, vamos a realizar un PCA sobre nuestras features, actualmente son 6, y vamos a dejarlos en 2D, además cada vez que vayamos a reducir las dimensiones.<br>
Utilizando un escalado estándar, uno de mínimos y máximos para ver como quedan los datos en un eje de 2 dimensiones, y como afecta este al resultado del aprendizaje.
### Model

| Models | Hiperparametres | Mètrica | Temps | 
| :---: | :---: | :---: | :---: |
| SVC | kernel=poly, c=1.0, degree=2 |  0.93 precision | 1.391 |
| SVC | kernel=rbf, c=1.0 | - | - |
| Gradient Boosting Classifier | default | - | - |
| Logistic Regression | default | - | - |
| Perceptron | default | - | - |

## Conclusions
Según los resultados que hemos vistos en notebook, los mejores modelos para predecir dicho problema, cuando un sistema electrónica puede sufrir un fallo según las fuentes de energía y la corriente por cada línea serían:<br> 
Utilizando un modelo clasificador de <b>Máquinas de vectores de soporte</b> con el kernel `RBF`, o una <b>red neuronal</b>. <br>
Y como un clasificador que no haya aportado ningún resultado relevante, sería el regresor logístico, ya que como se puede ver apreciar esta clasificando todo de una forma incorrecta. <br>
Además con la `ConfusionMatrixDisplay` podemos ver los como los false positives y false negatives, para entender mucho mejor como de bien está trabajando el clasificador.

