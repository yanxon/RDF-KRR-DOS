# Load data
import numpy as np
import matplotlib.pyplot as plt

dataset_X = np.loadtxt('X_.txt', dtype = 'float')
dataset_Y = np.loadtxt('Y_.txt', dtype = 'float')

dataset_Y = np.asarray(dataset_Y).reshape(len(dataset_Y),1)

# Next is to preprocess the data we obtain!

# Let's get rid of some high value of DOS

X = []
Y = []

for i in range(len(dataset_Y)):
    if dataset_Y[i]> 1.0:
        pass
    else:
        X.append(dataset_X[i,:])
        Y.append(dataset_Y[i])

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
estimator = GridSearchCV(KernelRidge(kernel='laplacian', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
estimator.fit(X_train, Y_train)

estimator.best_estimator_

estimator2 = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=0.01, kernel='laplacian', kernel_params=None)
estimator2.fit(X_train, Y_train)
y_predicted = estimator2.predict(X_test)
r2= estimator2.score(X_test, Y_test, sample_weight=None)
print('r^2 = ', r2)

plt.scatter(y_predicted, Y_test, c='green') # this is messed up fix the axis
plt.title('DOS: Actual vs Predicted-- 1012 Materials')
plt.text(1.6,.3, '$r^2$: 0.18145')
plt.xlabel('y_predicted')
plt.ylabel('Y_test')
plt.savefig('C:\\Users\\Carbon\\Desktop\\KNN.png')
plt.show()
#np.savetxt('/scratch/yanxonh/python', r2)