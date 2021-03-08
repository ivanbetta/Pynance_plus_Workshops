
# Machine Learning

"""
The objective of this code is the discussion of some fundamental preprocessing 
techniques.
"""

# Scaling Datasets

"""
Many algorithms (such as logistic regression, SVM & Neural Networks) show better 
performance when the dataset has a feature-wise null mean. Therefore, one of 
the most important preprocessing steps is so-called zero-centering.[1]
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
'https://scikit-learn.org/stable/'

'Set random seed for reproducibility'
np.random.seed(1000)
nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]

'Create the dataset'
X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

'Perform scaling'
ss = StandardScaler()
X_ss = ss.fit_transform(X)

rs = RobustScaler(quantile_range=(10, 90))
X_rs = rs.fit_transform(X)

mms = MinMaxScaler(feature_range=(-1, 1))
X_mms = mms.fit_transform(X)

'Show the results'
sns.set()

fig, ax = plt.subplots(2, 2, figsize=(22, 15), sharex=True, sharey=True)

ax[0, 0].scatter(X[:, 0], X[:, 1], s=50)
ax[0, 0].set_xlim([-6, 6])
ax[0, 0].set_ylim([-6, 6])
ax[0, 0].set_ylabel(r'$x_1$', fontsize=16)
ax[0, 0].set_title('Original dataset', fontsize=18)

ax[0, 1].scatter(X_mms[:, 0], X_mms[:, 1], s=50)
ax[0, 1].set_xlim([-6, 6])
ax[0, 1].set_ylim([-6, 6])
ax[0, 1].set_title(r'Min-Max scaling (-1, 1)', fontsize=18)

ax[1, 0].scatter(X_ss[:, 0], X_ss[:, 1], s=50)
ax[1, 0].set_xlim([-6, 6])
ax[1, 0].set_ylim([-6, 6])
ax[1, 0].set_xlabel(r'$x_0$', fontsize=16)
ax[1, 0].set_ylabel(r'$x_1$', fontsize=16)
ax[1, 0].set_title(r'Standard scaling ($\mu=0$ and $\sigma=1$)', fontsize=18)

ax[1, 1].scatter(X_rs[:, 0], X_rs[:, 1], s=50)
ax[1, 1].set_xlim([-6, 6])
ax[1, 1].set_ylim([-6, 6])
ax[1, 1].set_xlabel(r'$x_0$', fontsize=16)
ax[1, 1].set_title(r'Robust scaling based on ($10^{th}, 90^{th}$) quantiles', fontsize=18)

plt.show()

# Importance of Feature Scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

RANDOM_STATE = 42
FIG_SIZE = (10, 7)

features, target = load_wine(return_X_y=True)

'Make a train/test split using 30% test size'
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

'Fit to data and predict using pipelined GNB and PCA.'
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

'Fit to data and predict using pipelined scaling, GNB and PCA.'
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

'Show prediction accuracies in scaled and unscaled data.'
print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

'Extract PCA from pipeline'
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

'Show first principal components'
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

'Use PCA without and with scale on X_train data for visualization.'
X_train_transformed = pca.transform(X_train)
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

'visualize standardized vs. untouched dataset with PCA performed'
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Training dataset after PCA')
ax2.set_title('Standardized training dataset after PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()

# Normalization

"""
(Not to be confused with statistical normalization, which is more complex and 
generic approach) Consist of transforming each vector into a corresponfing one 
with unit norma given a predefined norm (for example, L2).
Contrary to other methods, normalizing a dataset leads to a projection where 
the existing relationships are kept only in terms of angular distance.
Natural Language Processing (NPL), two feature vectors are different in proportion 
to the angle they form, while they are almost insensitive to Euclidean distance.[1]
"""

from sklearn.preprocessing import Normalizer

'Set random seed for reproducibility'
np.random.seed(1000)


nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]

X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

'Perform normalization'
nz = Normalizer(norm='l2')
X_nz = nz.fit_transform(X)

'Show the results'
sns.set()

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(X_nz[:, 0], X_nz[:, 1], s=50)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_xlabel(r'$x_0$', fontsize=16)
ax.set_ylabel(r'$x_1$', fontsize=16)
ax.set_title(r'Normalized dataset ($L_2$ norm = 1)', fontsize=16)

plt.show()

'Compute a test example'
X_test = [
    [-4., 0.],
    [-1., 3.]
]

Y_test = nz.transform(X_test)

'Print the degree (in radians)'
print(np.arccos(np.dot(Y_test[0], Y_test[1])))

# Whitening

"""
Is the operation of imposing an identity covariance matrix to a zero-centered 
dataset.[1]
"""

'Set random seed for reproducibility'
np.random.seed(1000)
nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]

def zero_center(X):
    return X - np.mean(X, axis=0)

def whiten(X, correct=True):
    Xc = zero_center(X)
    _, L, V = np.linalg.svd(Xc)
    W = np.dot(V.T, np.diag(1.0 / L))
    return np.dot(Xc, W) * np.sqrt(X.shape[0]) if correct else 1.0

'Create the dataset'
X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

'Perform whitening'
X_whiten = whiten(X)

'Show the results'
sns.set()

fig, ax = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

ax[0].scatter(X[:, 0], X[:, 1], s=50)
ax[0].set_xlim([-6, 6])
ax[0].set_ylim([-6, 6])
ax[0].set_xlabel(r'$x_0$', fontsize=18)
ax[0].set_ylabel(r'$x_1$', fontsize=18)
ax[0].set_title('Original dataset', fontsize=18)

ax[1].scatter(X_whiten[:, 0], X_whiten[:, 1], s=50)
ax[1].set_xlim([-6, 6])
ax[1].set_ylim([-6, 6])
ax[1].set_xlabel(r'$x_0$', fontsize=18)
ax[1].set_title(r'Whitened dataset', fontsize=18)

plt.show()

'Show original and whitened covariance matrices'
print(np.cov(X.T))
print(np.cov(X_whiten.T))

# Training, validation & test sets

"""
Depending on the nature of the problem, it's possible to choose a split percentage 
ratio of 70%-30%, which is a good practice in machine learning, where the datasets 
are relatively small, or a higher training percentage of 80%, 90%, or up to 99% 
for deep learning tasks where numerosity of the sample is very high.
"""

np.random.seed(5)

nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]

Y = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.7,random_state=1000)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

# Cross-validation

"""
A valid method to detect the problem of wrongly selected test sets is provided 
by the cross-validation (CV) technique.
K-Fold is the cross validation approach we are going to use, the idea is to split 
the whole dataset X into a moving test set and a training set made up the 
remaining part. The size of the test set is determined by the number of folds, 
so that during k iterations, the test covers the whole original dataset.
... the right choice of k is a problem itself; however, in practice, a value in 
the range [5,15] is often the most reasonable default choice. [1] 
"""

import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler

'Learning curves for a Logistic Regression classification'

'Set random seed for reproducibility'
np.random.seed(1000)

'Create the dataset'
X, Y = make_classification(n_samples=500, n_classes=5, n_features=50, n_informative=10,
                           n_redundant=5, n_clusters_per_class=3, random_state=1000)

'Scale the dataset'
ss = StandardScaler()
X = ss.fit_transform(X)

'Perform a CV with 10 folds and a Logistic Regression'
lr = LogisticRegression(solver="lbfgs", multi_class="auto", random_state=1000)

splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=1000)
train_sizes = np.linspace(0.1, 1.0, 20)

'Compute the learning curves'
lr_train_sizes, lr_train_scores, lr_test_scores = learning_curve(lr, X, Y, cv=splits, train_sizes=train_sizes,
                                                                 n_jobs=joblib.cpu_count(), scoring="accuracy",
                                                                 shuffle=True, random_state=1000)

'Plot the scores'
sns.set()

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(lr_train_sizes, np.mean(lr_train_scores, axis=1), "o-", label="Training")
ax.plot(lr_train_sizes, np.mean(lr_test_scores, axis=1), "o-", label="Test")
ax.set_xlabel('Training set size', fontsize=18)
ax.set_ylabel('Average accuracy', fontsize=18)
ax.set_xticks(lr_train_sizes)
ax.grid(True)
ax.legend(fontsize=16)

plt.show()

'Average cross-validation accuracy for a different number pf folds'

'Set random seed for reproducibility'
np.random.seed(1000)

'Create the dataset'
X, Y = make_classification(n_samples=500, n_classes=5, n_features=50, n_informative=10,
                           n_redundant=5, n_clusters_per_class=3, random_state=1000)

'Scale the dataset'
ss = StandardScaler()
X = ss.fit_transform(X)

'Perform the evaluation for different number of folds'
mean_scores = []
cvs = [x for x in range(5, 100, 10)]

for cv in cvs:
    score = cross_val_score(LogisticRegression(solver="lbfgs", multi_class="auto", random_state=1000),
                            X, Y, scoring="accuracy", n_jobs=joblib.cpu_count(), cv=cv)
    mean_scores.append(np.mean(score))

'Plot the scores'
sns.set()

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(cvs, mean_scores, 'o-')
ax.set_xlabel('Number of folds / Training set size', fontsize=16)
ax.set_ylabel('Average accuracy', fontsize=16)
ax.set_xticks(cvs)
ax.set_xticklabels(['{} / {}'.format(x, int(500 * (x - 1) / x)) for x in cvs], fontsize=15)
ax.grid(True)

plt.show()

"""
Below the discussion of loss function, which are proxies
that allow us to measure the error made by a machine learning problem.[1]
"""

'https://scikit-learn.org/stable/user_guide.html'

# Mean Square Error

import numpy as np 
from sklearn.metrics import mean_squared_error 

'Using scikit-learn'
'Given values'
Y_true = [1,1,2,2,4] 
  
'Calculated values'
Y_pred = [0.6,1.29,1.99,2.69,3.4]   

'Calculation of Mean Squared Error (MSE)'
print(mean_squared_error(Y_true,Y_pred))

'Using numpy'

'Given values'
Y_true = [1,1,2,2,4] 
  
'Calculated values'
Y_pred = [0.6,1.29,1.99,2.69,3.4] 
  
'Mean Squared Error'
MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
print(MSE)

# HuberRegressor vs Ridge on dataset with strong outliers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge

'Generate toy data.'
rng = np.random.RandomState(0)
X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0,
                       bias=100.0)

'Add four strong outliers to the dataset.'
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.
X_outliers[2:, :] += X.min() - X.mean() / 4.
y_outliers[:2] += y.min() - y.mean() / 4.
y_outliers[2:] += y.max() + y.mean() / 4.
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, 'b.')

'Fit the huber regressor over a series of epsilon values.'
colors = ['r-', 'b-', 'y-', 'm-']

x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1.35, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

'Fit a ridge regressor to compare it to huber regressor.'
ridge = Ridge(alpha=0.0, random_state=0, normalize=True)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, 'g-', label="ridge regression")

plt.title("Comparison of HuberRegressor vs Ridge")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()

# References

'[1] Guiseppe Bonaccorso (2020). Mastering Machine Learning Algorithms.'
'https://www.packtpub.com/product/mastering-machine-learning-algorithms-second-edition/9781838820299'
'https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing'
'https://scikit-learn.org/stable/auto_examples/index.html'