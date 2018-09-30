"""
This is a machine learing (ML) script coded with Kernel Ridge Regression algorithm.
Variables: Density of states at Fermi level (X) and radial distribution function (Y).
"""

# Load data
import numpy as np
import matplotlib.pyplot as plt

X_sp = np.loadtxt('X_sp_metals.txt', dtype = 'float')
Y_sp = np.loadtxt('Y_sp_metals.txt', dtype = 'float')
sg_sp = np.loadtxt('sg_sp_metals.txt', dtype = 'int')
compounds_sp = np.loadtxt('compounds_sp_metals.txt', dtype = 'str')

# Try with shorter RDF
X_sp = X_sp[:,10:30]

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Training with Grid search
X_train, X_test, Y_train, Y_test = train_test_split(X_sp, Y_sp, test_size = 0.1, random_state=13)
estimator = GridSearchCV(KernelRidge(kernel='laplacian', gamma=0.1), cv=10, 
                         param_grid={"alpha": [1e6, 1e5, 1e4, 1e3, 100, 10, 1e0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
                         "gamma": np.logspace(-5, 5)})
estimator.fit(X_train, Y_train)

# Get the best parameter
estimator.best_estimator_

# Train with the best parameter
estimator2 = KernelRidge(alpha=0.01, coef0=1, degree=3, gamma=0.1, kernel='laplacian', kernel_params=None)
estimator2.fit(X_train, Y_train)
y_predicted = estimator2.predict(X_test)
r2= estimator2.score(X_test, Y_test, sample_weight=None)
print('r^2 = ', r2)

# Mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_predicted, Y_test)

# Plotting

Y_testing=[]
sg_testing=[]
for i in range(85):
    for j in range(844):
        if Y_test[i,0] == Y_sp[j,0]:
            sg_testing.append(sg_sp[j])
            break

Y_test_triclinic = []
y_pred_triclinic = []
Y_test_monoclinic = []
y_pred_monoclinic = []
Y_test_orthorhombic = []
y_pred_orthorhombic = []
Y_test_tetragonal = []
y_pred_tetragonal = []
Y_test_trigonal = []
y_pred_trigonal = []
Y_test_hexagonal = []
y_pred_hexagonal = []
Y_test_cubic = []
y_pred_cubic = []

for i, sg in enumerate(sg_testing):
    if sg in [1,2]:
        Y_test_triclinic.append(Y_test[i,:])
        y_pred_triclinic.append(y_predicted[i,:])
    elif sg in np.arange(3,16):
        Y_test_monoclinic.append(Y_test[i,:])
        y_pred_monoclinic.append(y_predicted[i,:])
    elif sg in np.arange(16,75):
        Y_test_orthorhombic.append(Y_test[i,:])
        y_pred_orthorhombic.append(y_predicted[i,:])
    elif sg in np.arange(75,143):
        Y_test_tetragonal.append(Y_test[i,:])
        y_pred_tetragonal.append(y_predicted[i,:])
    elif sg in np.arange(143,168):
        Y_test_trigonal.append(Y_test[i,:])
        y_pred_trigonal.append(y_predicted[i,:])
    elif sg in np.arange(168,195):
        Y_test_hexagonal.append(Y_test[i,:])
        y_pred_hexagonal.append(y_predicted[i,:])
    elif sg in np.arange(195,231):
        Y_test_cubic.append(Y_test[i,:])
        y_pred_cubic.append(y_predicted[i,:])
        
tri = plt.scatter(y_pred_triclinic, Y_test_triclinic, c='green')
mono = plt.scatter(y_pred_monoclinic, Y_test_monoclinic, c='red')
ortho = plt.scatter(y_pred_orthorhombic, Y_test_orthorhombic, c='black')
tetra = plt.scatter(y_pred_tetragonal, Y_test_tetragonal, c='gold')
trig = plt.scatter(y_pred_trigonal, Y_test_trigonal, c='purple')
hexa = plt.scatter(y_pred_hexagonal, Y_test_hexagonal, c='darkcyan')
cub = plt.scatter(y_pred_cubic, Y_test_cubic, c='gray')

plt.title('DOS: Actual vs Predicted -- 844 crystals')
plt.text(1.6,.3, '$r^2$: 0.6540')
plt.legend((tri, mono, ortho, tetra, trig, hexa, cub),
           ('trigonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic'),
           loc='lower right')
plt.xlabel('y_predicted')
plt.ylabel('Y_test')
plt.savefig('C:\\Users\\Carbon\\Desktop\\KRR_0.1.png')
plt.show()
