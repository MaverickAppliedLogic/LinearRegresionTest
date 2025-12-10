import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Punto de entrada principal: este bloque solo se ejecuta si el script
# se ejecuta directamente (no cuando se importa como módulo).
if __name__ == "__main__":

    # Generamos 70 valores de x en el intervalo [-20, 20)
    # np.random.rand(70, 1) -> valores uniformes en [0, 1)
    # * 40 -> [0, 40), luego -20 -> [-20, 20)
    x = -20 + 40 * np.random.rand(70, 1)

    # Pendiente "real" de la relación lineal subyacente
    m = 15 / 40

    # Generamos y como una relación lineal con ruido gaussiano
    # y ≈ m * x + ruido, donde el ruido tiene desviación típica 2
    y = m * x + 2 * np.random.randn(70, 1)

    # --------- FIGURA 1: Visualización de todos los datos ---------
    fig, ax = plt.subplots()
    # Nube de puntos (x, y) generados
    ax.scatter(x, y)
    ax.set_title('Datos')

    # --------- División en entrenamiento y test ---------
    # Separar los datos en conjunto de entrenamiento (70%) y test (30%)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x, y, test_size=0.3
    )

    # --------- FIGURA 2: Datos de entrenamiento ---------
    fig1, ax1 = plt.subplots()
    # Mostramos solo los puntos usados para entrenar el modelo
    ax1.scatter(x_train1, y_train1, s=50, edgecolors='none')
    ax1.set_title('Datos de entrenamiento')

    # --------- Entrenamiento del modelo de regresión lineal ---------
    # Creamos el objeto de regresión lineal
    regr = linear_model.LinearRegression()
    # Ajustamos el modelo a los datos de entrenamiento
    regr.fit(x_train1, y_train1)

    # --------- Evaluación del modelo: R^2 en train y test ---------
    # R^2 en el conjunto de entrenamiento
    coeff_train = regr.score(x_train1, y_train1)
    print('Coeficiente de determinacion R2 en entrenamiento: {}'.format(coeff_train))

    # R^2 en el conjunto de test (generalización)
    coef_test = regr.score(x_test1, y_test1)
    print('Coeficiente de determinacion R2 en test: {}'.format(coef_test))

    # --------- FIGURA 3: Datos train/test + recta de regresión ---------
    fig2, ax2 = plt.subplots()

    # Puntos de entrenamiento en verde
    ax2.scatter(
        x_train1, y_train1,
        s=50, edgecolors='green', label='entrenamiento'
    )
    # Puntos de test con borde azul y sin relleno
    ax2.scatter(
        x_test1, y_test1,
        c='none', s=50, edgecolors='blue', label='test'
    )

    # Número de puntos para dibujar la recta de regresión de forma suave
    nx = 100
    # Obtenemos los límites actuales del eje x
    x_min, x_max = plt.xlim()
    # Generamos 100 puntos equiespaciados entre x_min y x_max
    xx = np.linspace(x_min, x_max, nx)
    # Los convertimos a vector columna (shape (nx, 1)) para usar con predict
    xx_v = xx.reshape(-1, 1)

    # Predecimos los valores de y sobre la rejilla xx_v usando el modelo
    y_pred = regr.predict(xx_v)

    # Dibujamos la recta de regresión sobre el gráfico
    ax2.plot(xx, y_pred, color='black', label='recta regresion')
    ax2.set_title('Recta de regresion')
    ax2.legend()

    # Mostramos todas las figuras
    plt.show()
