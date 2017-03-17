from os import listdir
import os.path
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,mutual_info_classif,chi2
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.feature_selection import SelectFromModel
from sklearn import tree

def load_data(data_path, n_samples, depth):
	path0 = os.path.join(data_path,'CU_SPLIT_0')
	path1 = os.path.join(data_path,'CU_SPLIT_1')

	rem_cols = []
	rem_feats = ['POC','X', 'Y','W','D','FINAL_PU','FINAL_TU','FINAL_FME','TR_SKIP']
	#rem_feats += ['UP_SPLIT','LEFT_SPLIT','UPLEFT_SPLIT','UPRIGHT_SPLIT']
	#rem_feats += ['UP_DEPTH','LEFT_DEPTH','UPLEFT_DEPTH','UPRIGHT_DEPTH']
	rem_feats += ['SSE','TIME_ms']
	#rem_feats += ['COST_2NxN', 'COST_Nx2N']

	file_list = [x for x in listdir(path0) if '_d'+str(depth) in x]
	n_files = len(file_list)
	
	samples_per_file = n_samples/n_files
	samples = np.array([])
	names = None
	
	for p in file_list:
		if not names:
			names = open(os.path.join(path0,p),'r').readline().replace('TR_SKIP','CBF').replace('CNF','TR_SKIP').split(',')
			for rem in rem_feats:
				rem_cols.append(names.index(rem))

		if samples.size == 0:
			samples = np.genfromtxt(os.path.join(path0,p),delimiter=',',skip_header=1,max_rows=samples_per_file/2)
		else:
			tmp = np.genfromtxt(os.path.join(path0,p),delimiter=',',skip_header=1,max_rows=samples_per_file/2)
			samples = np.concatenate((samples,tmp))
		p = p.replace('split0','split1')
		tmp = np.genfromtxt(os.path.join(path1,p),delimiter=',',skip_header=1,max_rows=samples_per_file/2)
		samples = np.concatenate((samples,tmp))

	n_cols = samples.shape[1]
	samples = np.delete(samples, rem_cols, 1) # delete first cols and some last ones (labels)
	names = [x for x in names if x not in rem_feats]
	fout = open('samples_d'+str(depth)+'.csv','w')
	fout.write(','.join(names))
	fout.close()

	np.savetxt(open('samples_d'+str(depth)+'.csv','a'),samples, fmt='%g',delimiter=',')

	print samples.shape, len(names)
	samples_x = samples[:,:-1]
	samples_y = samples[:,-1]
	return samples_x,samples_y,names


def plotCorrelMap(x,labels):
	corrmat = np.absolute(np.corrcoef(x_norm.T))
	plt.imshow(corrmat, cmap='bwr')
	plt.xticks(range(corrmat.shape[0]),labels, rotation='vertical')
	plt.yticks(range(corrmat.shape[0]),labels)
	plt.grid(False)
	plt.colorbar()
	plt.tight_layout()
	plt.show()


def selectKBestIter(x, y,clf):
	for k in range(1,x.shape[1]):
		X_new = SelectKBest(mutual_info_classif, k=k).fit_transform(x, y)
		clf.fit(X_new,y)

		score_cv = cross_val_score(clf, X_new, Y, cv=5, n_jobs=-1)
		print 'k=',k,' ',
		print np.mean(score_cv)

def featureSelectionSVM(x, y,scores):

	#get IDX of feature only
	scores_sort = [x[0] for x in sorted(scores, key=lambda x: -x[2])]
	print ' Accuracy Precision Average'
	best_avg_score = 0
	for k in range(1,len(names)-1):
		rem_cols = [x for x in range(len(names)-1) if x not in scores_sort[:k]]

		x_sub = np.delete(X_norm, rem_cols, 1) # delete selected cols

		clf = svm.SVC()

		clf.fit(x_sub,Y)
		score_cv = cross_val_score(clf, x_sub, Y, cv=5, n_jobs=-1)
		precision_cv = cross_val_score(clf, x_sub, Y, cv=5, n_jobs=-1, scoring='precision')
		print 'k=',k,' ',
		avg_score = np.average([np.mean(score_cv), np.mean(precision_cv),np.mean(precision_cv)])
		print np.mean(score_cv), np.mean(precision_cv), avg_score

		if avg_score >= best_avg_score:
			best_avg_score = avg_score
			best_feats = [names[x] for x in scores_sort[:k]]
			best_feat_idx = [x for x in scores_sort[:k]]

	for k in range(1,len(names)-1):
		print ' '.join([names[x] for x in scores_sort[:k]])

	return best_avg_score, best_feats, best_feat_idx

def computeMetrics(x, y, labels):
	f_scores = f_classif(x,y)

	mutual_infos = mutual_info_classif(x,y)
	chip2= chi2(x,y)
	scores = []
	for i in range(len(labels)-1):
		X_sub = x[:,i].reshape(x.shape[0],1)
		clf = svm.SVC(kernel='rbf')
		clf.fit(X_sub,y)
		score_cv_svm1 = cross_val_score(clf, X_sub, y, cv=5, n_jobs=-1, scoring='precision')
		score_cv_svm2 = cross_val_score(clf, X_sub, y, cv=5, n_jobs=-1, scoring='accuracy')
		score_cv_svm =  (score_cv_svm1*2.0 + score_cv_svm2) / 3.0
		clf = tree.DecisionTreeClassifier()

		clf.fit(X_sub,y)
		score_cv_tree1 = cross_val_score(clf, X_sub, y, cv=5, n_jobs=-1, scoring='precision')
		score_cv_tree2 = cross_val_score(clf, X_sub, y, cv=5, n_jobs=-1, scoring='accuracy')
		score_cv_tree =  (score_cv_tree1*2.0 + score_cv_tree2) / 3.0

		scores.append([i,labels[i],np.mean(score_cv_svm), np.mean(score_cv_tree), f_scores[0][i], mutual_infos[i], chip2[0][i]])

	return scores
