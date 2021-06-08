import numpy as np
from joblib import dump
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

dt = np.loadtxt("round5.csv", delimiter=",", dtype=str)
idx = np.random.choice(dt.shape[0], size=dt.shape[0], replace=False)
dt = dt[idx, :]

X = dt[:,:-1].astype(np.float32)
cutoff = int(X.shape[0]*0.75)
X_train = X[:cutoff,:]
X_test = X[cutoff:,:]

y = dt[:,-1].astype(np.float)
y = y.astype(np.int)
y_train = y[:cutoff]
y_test = y[cutoff:]

#print(X_train.shape, y_train.shape)

clf_svm = SVC(probability=True, kernel='rbf')
parameters = {'gamma':[0.001, 0.01, 0.1], 'C':[1, 10]}
clf_svm = GridSearchCV(clf_svm, parameters)
clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6,  max_samples=0.8333, bootstrap=False)

clf = clf_svm.fit(X_train, y_train)

p = clf.predict_proba(X_train)
print("Training accuracy: ", clf.score(X_train,y_train))
print("Training AUC: ", roc_auc_score(y_train, p[:,1]))

p = clf.predict_proba(X_test)
print("Test accuracy: ", clf.score(X_test,y_test))
print("Test AUC: ", roc_auc_score(y_test, p[:,1]))
print("Test cross-entropy: ", log_loss(y_test, p))

clf = clf_svm.fit(X, y)
dump(clf, 'round5.joblib')

