import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(0)
    X = np.r_[np.random.randn(70, 2) - [2, 2],
          np.random.randn(70, 2) + [2, 2]]

    Y= [0] * 70 + [1] * 70

    fig, ax = plt.subplots()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state= 1234, shuffle=True)


    modelo = svm.SVC(C=100, kernel='linear', random_state = 123)
    mod = modelo.fit(X_train, Y_train)

    x = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 50)
    y = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 50)

    Y, X = np.meshgrid(y, x)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    print(grid.shape)

    pred_grid = modelo.predict(grid)

    ax.scatter(grid[:,0], grid[:, 1], c=pred_grid, alpha=0.2)
    ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, alpha= 1 )

    ax.scatter(modelo.support_vectors_[:, 0], modelo.support_vectors_[:, 1],
               s=200, linewidth = 1, facecolors= 'none',edgecolors= 'black')

    ax.contour(X, Y, modelo.decision_function(grid).reshape(X.shape), colors= 'k',
               levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])
    plt.show()
    print(modelo.score(X_test, Y_test))

