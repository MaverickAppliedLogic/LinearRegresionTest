
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n = int(input('Numero de interaciones: '))
    print('Pesos iniciales')
    a0 = float(input('a0: '))
    a1 = float(input('a1: '))
    a2 = float(input('a2: '))
    p = [a0, a1, a2]
    x0 = 1
    y = np.zeros(n)
    E = np.zeros(n)
    grad = []

    for i in range(n):
        x1 = float(input('Entrada_1: '))
        x2 = float(input('Entrada_2: '))
        X = [x0, x1, x2]
        z = float(input('Salida deseada: '))
        s = 0
        for j in range(3):
            s += p[3 * i + j] * X[j]
        y[i] = s
        print('Entrada sigmoide: ', y[i])
        salida = 1 / (1 + np.exp(-y[i]))
        E[i] = (salida - z) ** 2
        delta = 0.5
        k = 2 * salida * (1 - salida) * (salida - z)
        parcial_a = k * x1
        parcial_b = k * x2
        parcial_c = k
        gr = np.sqrt(parcial_a ** 2 + parcial_b ** 2 + parcial_c ** 2)
        grad = grad + [gr]
        p = p + [p[3 * i] - delta * parcial_c, p[3 * i + 1] - delta * parcial_a, p[3 * i + 2] - delta * parcial_b]
        print('Pesos: ', p)
    print('Error: ', E)
    print('Salidas: ', y)
    for i in range(int(len(p) / 3)):
        print(p[3 * i: 3 * i + 3])
    plt.plot(0, 0, 'bo', 0, 1, 'bo', 1, 0, 'bo', 1, 1, 'r')
    x = np.linspace(-0.1, 1.1, 100)
    for i in range(4):
        a0, a1, a2 = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        y = -(a1 / a2) * x + (-a0 / a2)
        plt.subplot(2, 2, i + 1)
        plt.plot(0, 0, 'bo', 0, 1, 'bo', 1, 0, 'bo', 1, 1, 'r')
        plt.plot(x, y)
    plt.show()
