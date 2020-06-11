# -*- coding: utf-8 -*-
"""
We test hyperopt parameter tuning on artificial highly imbalanced data
"""

import pprint
from collections import Counter

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as imb_pipeline
# from sklearn import datasets
# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import RepeatedStratifiedKFold
from scoring_methods import gini_scorer, custom_scorer, new_gini_scorer

from model_zoo import model_zoo

from get_sahulat_data import load_sahulat_data

pp = pprint.PrettyPrinter(indent=4)

global random_state 
random_state = 42


X, Y = load_sahulat_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, shuffle=True,  random_state=random_state, stratify=Y)
scale = MinMaxScaler()
counter = Counter(Y)
print(counter)

def clean_best(params):
    del params['scale']
    del params['normalize']
    # del params['scorer']
    # del params['model']
    return params



from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             log_loss, make_scorer, precision_recall_curve,
                             r2_score, roc_auc_score)
def testing_the_hyperparameters():
	print('Testing best params on the unseen data ...\n' )
	cleaned_best = clean_best(best)
	rf = RandomForestClassifier(**cleaned_best)
	rf.fit(X_train, Y_train)
	Y_pred = rf.predict(X_test)
	Y_pred_prba = rf.predict_proba(X_test)
	accuracy_score_value = accuracy_score(y_true=Y_test, y_pred=Y_pred)
	r2_score_value = r2_score(y_true=Y_test, y_pred=Y_pred)
	f1_score_value = f1_score(y_true=Y_test, y_pred=Y_pred)
	roc_auc_value = roc_auc_score(y_true=Y_test, y_score=Y_pred)
	g_v = normalized_gini(solution=Y_test, submission=Y_pred)
	g_v_proba = normalized_gini_new(solution=Y_test, submission=Y_pred_proba)
	conf_matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
	tn, fp, fn, tp = conf_matrix.ravel()
	print(f'tn={tn} ')
	print(f'fp={fp} ')
	print(f'fn={fn} ')
	print(f'tp={tp}\n')
	recall = tp / (tp + fn)
	precision = tp / (tp + fp)
	F1 = 2 * recall * precision / (recall + precision)
	model_name = 'best_hyperopted_model'
	print('Model Performance:{}'.format(model_name))
	print('accuracy_score is: {:0.4f} '.format(accuracy_score_value))
	print('roc-auc value is {:0.4f}.'.format(roc_auc_value))
	print('gini score is {:0.4f}.'.format(g_v))
	print('new gini score is {:0.4f}.'.format(g_v_proba))
	print(f'recall = {recall}')
	print(f'precision = {precision}\n')
	print(f'F1 = {F1}\n')
	print('conf_matrix = \n')
	print(conf_matrix)

# global best_cross_val_score
# best_cross_val_score = 0
def pick_animal_from_the_zoo(zoopark = model_zoo): 
	'''''
		we select best model from the zoopark

	"'''
	trials = Trials()
	# best_cross_val_score = 0
	best_model = None
	for model, params in model_zoo.items():
		cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
		# cv = 5
		def hyperopt_cross_val_custom_score(params):
			X_ = X_train[:]
			# if 'normalize' in params:
			# 	if params['normalize'] == 1:
			# 		X_ = normalize(X_)
			del params['normalize']
			# if 'scale' in params:
			# 	if params['scale'] == 1:
			# 		X_ = scale.fit_transform(X_)
			del params['scale']
			# scoring = gini_scorer
			scoring = new_gini_scorer
			adasyn = SMOTE()
			clf = model(**params)
			pipeline = imb_pipeline([('sampling', adasyn), ('class', clf)])
			# pipeline = clf
			score = cross_val_score(estimator=pipeline,
													X=X_, 
													y=Y_train, 
													scoring=scoring, 
													cv=cv).mean()
			return score
		
		def loss_function(params):
			cross_val_score = hyperopt_cross_val_custom_score(params=params)	
			loss_value = 1-cross_val_score
			return {'loss': loss_value, 'status': STATUS_OK}
		
		print(f'model = {model}')
		print(f'params={params}')
		best = fmin(
					fn = loss_function, 
					space=params, 
					algo=tpe.suggest, 
					max_evals=50, 
					trials=trials,
					return_argmin=False
					)
		print("best:\n")
		pp.pprint(best)
		best_model = model
	cleaned_best = clean_best(best)
	return cleaned_best, best_model
	

if  __name__ == '__main__':
	_, _ = pick_animal_from_the_zoo(zoopark=model_zoo)
