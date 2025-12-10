
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

if __name__ == "__main__":

    x = -20 + 40 * np.random.rand(70, 1)
    m = 15/40
    y = m * x + 2 * np.random.randn(70, 1)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title('Datos')

    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.3)
    fig1, ax1 = plt.subplots()

    ax1.scatter(x_train1, y_train1, s=50, edgecolors='none')
    ax1.set_title('Datos de entrenamiento')

    regr = linear_model.LinearRegression()
    regr.fit(x_train1, y_train1)

    coeff_train = regr.score(x_train1, y_train1)
    print('Coeficiente de determinacion R2 en entrenamiento: {}'.format(coeff_train))

    coef_test = regr.score(x_test1, y_test1)
    print('Coeficiente de determinacion R2 en test: {}'.format(coef_test))

    fig2, ax2 = plt.subplots()

    ax2.scatter(x_train1, y_train1, s=50, edgecolors='green', label='entrenamiento')
    ax2.scatter(x_test1, y_test1, c='none', s=50, edgecolors='blue', label='test')

    nx = 100
    x_min, x_max = plt.xlim()
    xx = np.linspace(x_min, x_max, nx)
    xx_v = xx.reshape(-1, 1)

    y_pred = regr.predict(xx_v)
    ax2.plot(xx, y_pred, color='black', label='recta regresion')
    ax2.set_title('Recta de regresion')
    ax2.legend()
    plt.show()
