import numpy as np
from Utils import *
from sys import argv
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,mutual_info_classif
from sklearn.ensemble import AdaBoostClassifier
# STEP 1: GET N SAMPLES FROM DATABASE AND PREPROCESS
hm_path = argv[1]
N_SAMPLES = 10000
depth=0

X,Y,names = load_data(hm_path,N_SAMPLES, depth)
X_norm = (X - X.min(0)) / (X.ptp(0) + 0.000001)
#feat_variances = np.var(X, axis=0)
scores = computeMetrics(X_norm,Y,names)

feat_variances = np.var(X_norm, axis=0)
feat_means   = np.mean(X_norm, axis=0)
#print feat_means, feat_variances

clf = svm.SVC()
clf.fit(X,Y)
score_cv = cross_val_score(clf, X, Y, cv=5, n_jobs=-1)

print np.mean(score_cv)

clf.fit(X_norm,Y)
score_cv = cross_val_score(clf, X_norm, Y, cv=5, n_jobs=-1)

print np.mean(score_cv)




for k in [1,3,5,10,20,50,100]:
	clf = AdaBoostClassifier(base_estimator = svm.SVC(probability=True), n_estimators=k)
	clf.fit(X_norm,Y)
	score_cv = cross_val_score(clf, X_norm, Y, cv=5, n_jobs=-1)
	precision_cv = cross_val_score(clf, X_norm, Y, cv=5, n_jobs=-1, scoring='precision')
	print 'k=',k,' ',
	avg_score = np.average([np.mean(score_cv), np.mean(precision_cv)])


	print np.mean(score_cv), np.mean(precision_cv), avg_score
